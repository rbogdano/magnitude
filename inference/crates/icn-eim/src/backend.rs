//! `CompletionBackend` over a vLLM container's chat-completions endpoint.
//!
//! The trait is synchronous with an event callback, while the transport is async. `icn-api`
//! invokes `complete` inside `tokio::task::spawn_blocking`, so blocking on a runtime handle
//! here is correct: the caller is already on a blocking thread, not a runtime worker.
//!
//! ICN's previous out-of-process backend proxied length-prefixed JSON over a child process's
//! stdio. This is the same architecture with a different transport — every value crossing the
//! boundary was already serializable, which is why swapping stdio for HTTP touches nothing
//! above this crate.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use futures_util::StreamExt as _;
use icn_contracts::{
    ChatRequest, ChatTemplateRequest, CompletionBackend, FinishReason, Generation,
    GenerationMetrics, GenerationSnapshot, InferenceError, InferenceEvent, InferenceProgress,
    InferenceStreamEvent, ModelProperties, PreparedChatInfo,
};
use tokio::runtime::Handle;

use crate::request::to_vllm_request;
use crate::sse::{ChunkOutcome, ToolCallAccumulator, decode_chunk, next_sse_frame, sse_data};

/// How long to wait for the whole response. Generous, because a long agentic turn on CPU can
/// legitimately run for minutes; the caller cancels through the event callback instead.
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Bounded so a runaway server cannot exhaust memory through a single frame.
const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

pub struct EimCompletionBackend {
    /// The serving-configuration identity ICN addresses this model by.
    model_id: String,
    /// Base URL of the container, e.g. `http://127.0.0.1:43117`.
    endpoint: String,
    /// What the server calls the model, read from `GET /v1/models` at readiness.
    served_model_name: String,
    properties: ModelProperties,
    client: reqwest::Client,
    runtime: Handle,
    request_timeout: Duration,
}

impl EimCompletionBackend {
    pub fn new(
        model_id: impl Into<String>,
        endpoint: impl Into<String>,
        served_model_name: impl Into<String>,
        properties: ModelProperties,
        runtime: Handle,
    ) -> Result<Self, InferenceError> {
        let client = reqwest::Client::builder()
            // The container is on loopback; a proxy would break the connection entirely.
            .no_proxy()
            .build()
            .map_err(|error| InferenceError::Backend(format!("http client: {error}")))?;
        Ok(Self {
            model_id: model_id.into(),
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            served_model_name: served_model_name.into(),
            properties,
            client,
            runtime,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        })
    }

    #[must_use]
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.endpoint)
    }
}

/// Everything accumulated while the stream runs.
struct StreamState {
    text: String,
    reasoning: String,
    tool_calls: ToolCallAccumulator,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    generated_tokens: usize,
    finish_reason: Option<String>,
    started_streaming: bool,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    saw_done: bool,
}

impl StreamState {
    fn new() -> Self {
        Self {
            text: String::new(),
            reasoning: String::new(),
            tool_calls: ToolCallAccumulator::default(),
            prompt_tokens: 0,
            cached_prompt_tokens: 0,
            generated_tokens: 0,
            finish_reason: None,
            started_streaming: false,
            first_token_at: None,
            last_token_at: None,
            saw_done: false,
        }
    }

    fn snapshot(&self, metrics: GenerationMetrics) -> GenerationSnapshot {
        GenerationSnapshot {
            cached_prompt_tokens: self.cached_prompt_tokens,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            metrics,
        }
    }

    /// Metrics measured on this side of the wire.
    ///
    /// vLLM's OpenAI endpoint reports no per-request timing, so queue, sampler, parser, and
    /// draft figures stay zero rather than being invented. `prompt_ms` is time to first token:
    /// on a streaming request that interval *is* prefill plus scheduling, and attributing it
    /// to the prompt is the closest honest reading available.
    fn metrics(&self, dispatched_at: Instant) -> GenerationMetrics {
        let time_to_first_token_ms = self.first_token_at.map_or(0.0, |at| {
            at.duration_since(dispatched_at).as_secs_f64() * 1000.0
        });
        let decode_ms = match (self.first_token_at, self.last_token_at) {
            (Some(first), Some(last)) => last.duration_since(first).as_secs_f64() * 1000.0,
            _ => 0.0,
        };
        let decode_tokens_per_second = if decode_ms > 0.0 && self.generated_tokens > 1 {
            // The first token is attributed to prefill, so decode rate covers the rest.
            (self.generated_tokens as f64 - 1.0) / (decode_ms / 1000.0)
        } else {
            0.0
        };
        let prompt_tokens_per_second = if time_to_first_token_ms > 0.0 && self.prompt_tokens > 0 {
            self.prompt_tokens as f64 / (time_to_first_token_ms / 1000.0)
        } else {
            0.0
        };

        GenerationMetrics {
            queue_ms: 0.0,
            prompt_ms: time_to_first_token_ms,
            decode_ms,
            time_to_first_token_ms,
            prompt_tokens_per_second,
            decode_tokens_per_second,
            sampler_ms: 0.0,
            parser_ms: 0.0,
            draft_tokens: 0,
            accepted_draft_tokens: 0,
            draft_ms: 0.0,
            verification_ms: 0.0,
        }
    }
}

impl CompletionBackend for EimCompletionBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn properties(&self) -> Result<ModelProperties, InferenceError> {
        Ok(self.properties.clone())
    }

    /// Templating happens inside the container, so ICN cannot render a prompt for this model.
    ///
    /// Returning an error rather than a fabricated prompt is deliberate: a caller that needs a
    /// real rendered prompt would otherwise silently receive a fiction. No caller in Magnitude
    /// uses this path for local models.
    fn apply_template(
        &self,
        _request: ChatTemplateRequest,
    ) -> Result<PreparedChatInfo, InferenceError> {
        Err(InferenceError::Backend(
            "templating is performed by the vLLM server; ICN does not render prompts for \
             container-backed models"
                .to_owned(),
        ))
    }

    fn complete(
        &self,
        request: ChatRequest,
        on_event: &mut dyn FnMut(InferenceStreamEvent) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        let timings_per_token = request.timings_per_token;
        let body = to_vllm_request(&request, &self.served_model_name);
        let url = self.chat_completions_url();
        let cancelled = Arc::new(AtomicBool::new(false));

        // A callback error must stop the stream rather than be swallowed, and it is the error
        // the caller sees, not a generic transport failure.
        let mut callback_error: Option<InferenceError> = None;
        let mut state = StreamState::new();
        let dispatched_at = Instant::now();

        let mut emit =
            |event: InferenceStreamEvent, callback_error: &mut Option<InferenceError>| -> bool {
                match on_event(event) {
                    Ok(()) => true,
                    Err(error) => {
                        *callback_error = Some(error);
                        cancelled.store(true, Ordering::Relaxed);
                        false
                    }
                }
            };

        if !emit(
            InferenceStreamEvent {
                delta: InferenceEvent::Progress(InferenceProgress::Preparing),
                timings: None,
            },
            &mut callback_error,
        ) {
            return Err(callback_error.unwrap_or(InferenceError::Cancelled));
        }

        let transport = self.runtime.block_on(async {
            let response = self
                .client
                .post(&url)
                .timeout(self.request_timeout)
                .json(&body)
                .send()
                .await
                .map_err(|error| InferenceError::Backend(format!("request failed: {error}")))?;

            let status = response.status();
            if !status.is_success() {
                let detail = response.text().await.unwrap_or_default();
                let detail = detail.trim();
                let detail = if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {}", &detail[..detail.len().min(500)])
                };
                return Err(match status.as_u16() {
                    // The container is up but does not serve this name. That is a
                    // configuration mistake, not a transient failure.
                    404 => InferenceError::InvalidConfig(format!(
                        "served model `{}` is unknown to the container{detail}",
                        self.served_model_name
                    )),
                    503 => InferenceError::Overloaded,
                    _ => InferenceError::Backend(format!("HTTP {status}{detail}")),
                });
            }

            let mut stream = response.bytes_stream();
            let mut buffer: Vec<u8> = Vec::new();

            while let Some(chunk) = stream.next().await {
                if cancelled.load(Ordering::Relaxed) {
                    return Err(InferenceError::Cancelled);
                }
                let chunk = chunk
                    .map_err(|error| InferenceError::Backend(format!("stream failed: {error}")))?;
                buffer.extend_from_slice(&chunk);
                if buffer.len() > MAX_FRAME_BYTES {
                    return Err(InferenceError::Backend(format!(
                        "server-sent event frame exceeded {MAX_FRAME_BYTES} bytes"
                    )));
                }

                while let Some((frame, consumed)) = next_sse_frame(&buffer) {
                    buffer.drain(..consumed);
                    let Some(payload) = sse_data(&frame) else {
                        continue;
                    };
                    match decode_chunk(&payload) {
                        Ok(ChunkOutcome::Done) => {
                            state.saw_done = true;
                        }
                        Ok(ChunkOutcome::Error(message)) => {
                            return Err(InferenceError::Backend(message));
                        }
                        Ok(ChunkOutcome::Chunk(chunk)) => {
                            if !absorb(
                                &chunk,
                                &mut state,
                                dispatched_at,
                                timings_per_token,
                                &mut emit,
                                &mut callback_error,
                            ) {
                                return Err(InferenceError::Cancelled);
                            }
                        }
                        Err(error) => {
                            return Err(InferenceError::Backend(format!(
                                "undecodable chunk: {error}"
                            )));
                        }
                    }
                }
            }
            Ok(())
        });

        // A cancellation caused by the callback reports the callback's error, which carries
        // why the consumer stopped listening.
        if let Some(error) = callback_error {
            return Err(error);
        }
        transport?;

        if !state.saw_done && state.finish_reason.is_none() {
            return Err(InferenceError::Backend(
                "stream ended without a finish reason or [DONE] sentinel".to_owned(),
            ));
        }

        let metrics = state.metrics(dispatched_at);
        Ok(Generation {
            text: state.text,
            reasoning: state.reasoning,
            tool_calls: state.tool_calls.finish(),
            cached_prompt_tokens: state.cached_prompt_tokens,
            prompt_tokens: state.prompt_tokens,
            generated_tokens: state.generated_tokens,
            finish_reason: resolve_finish_reason(state.finish_reason.as_deref(), &metrics),
            metrics,
        })
    }
}

/// Applies one decoded chunk to the accumulated state, emitting events as it goes.
///
/// Returns false when the consumer's callback asked to stop.
fn absorb(
    chunk: &crate::sse::ChatChunk,
    state: &mut StreamState,
    dispatched_at: Instant,
    timings_per_token: bool,
    emit: &mut impl FnMut(InferenceStreamEvent, &mut Option<InferenceError>) -> bool,
    callback_error: &mut Option<InferenceError>,
) -> bool {
    if let Some(usage) = chunk.usage {
        state.prompt_tokens = usage.prompt_tokens;
        state.cached_prompt_tokens = usage.cached_prompt_tokens;
        // The server's completion count is authoritative over our delta count, which cannot
        // see tokens that produced no text.
        if usage.completion_tokens > 0 {
            state.generated_tokens = usage.completion_tokens;
        }
    }
    if let Some(reason) = &chunk.finish_reason {
        state.finish_reason = Some(reason.clone());
    }

    let semantic =
        chunk.content.is_some() || chunk.reasoning.is_some() || !chunk.tool_calls.is_empty();
    if semantic {
        let now = Instant::now();
        if state.first_token_at.is_none() {
            state.first_token_at = Some(now);
        }
        state.last_token_at = Some(now);
    }

    // `StreamStart` marks the beginning of the assistant turn and must precede any delta.
    if semantic && !state.started_streaming {
        state.started_streaming = true;
        if !emit(
            InferenceStreamEvent {
                delta: InferenceEvent::StreamStart,
                timings: None,
            },
            callback_error,
        ) {
            return false;
        }
        if !emit(
            InferenceStreamEvent {
                delta: InferenceEvent::Progress(InferenceProgress::Generating),
                timings: None,
            },
            callback_error,
        ) {
            return false;
        }
    }

    let timings = |state: &StreamState| {
        timings_per_token.then(|| state.snapshot(state.metrics(dispatched_at)))
    };

    if let Some(text) = &chunk.reasoning {
        state.reasoning.push_str(text);
        let snapshot = timings(state);
        if !emit(
            InferenceStreamEvent {
                delta: InferenceEvent::ReasoningDelta { text: text.clone() },
                timings: snapshot,
            },
            callback_error,
        ) {
            return false;
        }
    }

    if let Some(text) = &chunk.content {
        state.text.push_str(text);
        // Only counted when the server sent no usage; otherwise usage wins.
        if state.generated_tokens == 0 || chunk.usage.is_none() {
            state.generated_tokens = state.generated_tokens.saturating_add(1);
        }
        let snapshot = timings(state);
        if !emit(
            InferenceStreamEvent {
                delta: InferenceEvent::ContentDelta { text: text.clone() },
                timings: snapshot,
            },
            callback_error,
        ) {
            return false;
        }
    }

    for call in &chunk.tool_calls {
        state.tool_calls.absorb(call);
        let snapshot = timings(state);
        if !emit(
            InferenceStreamEvent {
                delta: InferenceEvent::ToolCallDelta {
                    index: call.index,
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                },
                timings: snapshot,
            },
            callback_error,
        ) {
            return false;
        }
    }

    true
}

/// A stream that produced tool calls but no stated reason still finished for tool calls.
fn resolve_finish_reason(reason: Option<&str>, _metrics: &GenerationMetrics) -> FinishReason {
    crate::sse::finish_reason(reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::{ChatChunk, ToolCallDelta, UsageDelta};

    fn collect(
        chunks: &[ChatChunk],
        timings_per_token: bool,
    ) -> (StreamState, Vec<InferenceEvent>) {
        let mut state = StreamState::new();
        let dispatched_at = Instant::now();
        let mut events = Vec::new();
        let mut error = None;
        let mut emit = |event: InferenceStreamEvent, _: &mut Option<InferenceError>| {
            events.push(event.delta);
            true
        };
        for chunk in chunks {
            assert!(absorb(
                chunk,
                &mut state,
                dispatched_at,
                timings_per_token,
                &mut emit,
                &mut error,
            ));
        }
        (state, events)
    }

    fn content(text: &str) -> ChatChunk {
        ChatChunk {
            content: Some(text.to_owned()),
            ..ChatChunk::default()
        }
    }

    #[test]
    fn emits_stream_start_before_the_first_delta_exactly_once() {
        let (_, events) = collect(&[content("Hel"), content("lo")], false);

        assert!(matches!(events[0], InferenceEvent::StreamStart));
        assert!(matches!(
            events[1],
            InferenceEvent::Progress(InferenceProgress::Generating)
        ));
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, InferenceEvent::StreamStart))
                .count(),
            1
        );
    }

    #[test]
    fn accumulates_text_across_deltas() {
        let (state, _) = collect(&[content("Hel"), content("lo, "), content("world")], false);

        assert_eq!(state.text, "Hello, world");
    }

    #[test]
    fn keeps_reasoning_separate_from_content() {
        let reasoning = ChatChunk {
            reasoning: Some("thinking".to_owned()),
            ..ChatChunk::default()
        };
        let (state, events) = collect(&[reasoning, content("answer")], false);

        assert_eq!(state.reasoning, "thinking");
        assert_eq!(state.text, "answer");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, InferenceEvent::ReasoningDelta { .. }))
        );
    }

    #[test]
    fn a_usage_report_overrides_the_counted_delta_total() {
        let usage = ChatChunk {
            usage: Some(UsageDelta {
                prompt_tokens: 1200,
                completion_tokens: 48,
                cached_prompt_tokens: 1024,
            }),
            ..ChatChunk::default()
        };
        // Three text deltas, but the server says 48 tokens: a token can produce no text.
        let (state, _) = collect(&[content("a"), content("b"), content("c"), usage], false);

        assert_eq!(state.generated_tokens, 48);
        assert_eq!(state.prompt_tokens, 1200);
        assert_eq!(state.cached_prompt_tokens, 1024);
    }

    #[test]
    fn counts_deltas_when_the_server_reports_no_usage() {
        let (state, _) = collect(&[content("a"), content("b")], false);

        assert_eq!(state.generated_tokens, 2);
    }

    /// Runs one chunk through `absorb` and returns the full events, timings included.
    fn absorb_once(timings_per_token: bool) -> Vec<InferenceStreamEvent> {
        let mut events = Vec::new();
        {
            let mut error = None;
            let mut emit = |event: InferenceStreamEvent, _: &mut Option<InferenceError>| {
                events.push(event);
                true
            };
            absorb(
                &content("x"),
                &mut StreamState::new(),
                Instant::now(),
                timings_per_token,
                &mut emit,
                &mut error,
            );
        }
        events
    }

    #[test]
    fn attaches_timings_only_when_requested() {
        assert!(
            absorb_once(false)
                .iter()
                .all(|event| event.timings.is_none())
        );

        let requested = absorb_once(true);
        let content_deltas = requested
            .iter()
            .filter(|event| matches!(event.delta, InferenceEvent::ContentDelta { .. }))
            .collect::<Vec<_>>();
        assert!(!content_deltas.is_empty());
        assert!(content_deltas.iter().all(|event| event.timings.is_some()));
        // StreamStart and progress markers carry no per-token snapshot.
        assert!(
            requested
                .iter()
                .filter(|event| matches!(event.delta, InferenceEvent::StreamStart))
                .all(|event| event.timings.is_none())
        );
    }

    #[test]
    fn a_snapshot_reports_the_tokens_produced_so_far() {
        let mut state = StreamState::new();
        state.generated_tokens = 3;

        assert_eq!(
            state
                .snapshot(state.metrics(Instant::now()))
                .generated_tokens,
            3
        );
    }

    #[test]
    fn forwards_tool_call_deltas_and_assembles_them() {
        let opening = ChatChunk {
            tool_calls: vec![ToolCallDelta {
                index: 0,
                id: Some("call_1".to_owned()),
                name: Some("read_file".to_owned()),
                arguments: "{\"pa".to_owned(),
            }],
            ..ChatChunk::default()
        };
        let closing = ChatChunk {
            tool_calls: vec![ToolCallDelta {
                index: 0,
                id: None,
                name: None,
                arguments: "th\":\"a.rs\"}".to_owned(),
            }],
            ..ChatChunk::default()
        };
        let (state, events) = collect(&[opening, closing], false);

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, InferenceEvent::ToolCallDelta { .. }))
                .count(),
            2,
            "each fragment is forwarded so the consumer can render progress"
        );
        let calls = state.tool_calls.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"path":"a.rs"}"#);
    }

    #[test]
    fn a_usage_only_chunk_starts_no_assistant_turn() {
        let usage = ChatChunk {
            usage: Some(UsageDelta {
                prompt_tokens: 5,
                completion_tokens: 0,
                cached_prompt_tokens: 0,
            }),
            ..ChatChunk::default()
        };
        let (state, events) = collect(&[usage], false);

        assert!(!state.started_streaming);
        assert!(events.is_empty(), "no semantic content, no events");
    }

    #[test]
    fn stops_absorbing_when_the_consumer_rejects_an_event() {
        let mut state = StreamState::new();
        let mut error = None;
        let mut emit = |_: InferenceStreamEvent, error: &mut Option<InferenceError>| {
            *error = Some(InferenceError::Cancelled);
            false
        };

        assert!(!absorb(
            &content("x"),
            &mut state,
            Instant::now(),
            false,
            &mut emit,
            &mut error,
        ));
        assert!(matches!(error, Some(InferenceError::Cancelled)));
    }

    #[test]
    fn metrics_leave_unavailable_figures_at_zero() {
        let (state, _) = collect(&[content("a"), content("b")], false);
        let metrics = state.metrics(Instant::now() - Duration::from_millis(100));

        // vLLM's OpenAI endpoint reports none of these, so they are not invented.
        assert_eq!(metrics.queue_ms, 0.0);
        assert_eq!(metrics.sampler_ms, 0.0);
        assert_eq!(metrics.parser_ms, 0.0);
        assert_eq!(metrics.draft_tokens, 0);
        assert_eq!(metrics.verification_ms, 0.0);
        // Time to first token is measured, and prompt time is attributed to it.
        assert!(metrics.time_to_first_token_ms > 0.0);
        assert_eq!(metrics.prompt_ms, metrics.time_to_first_token_ms);
    }

    #[test]
    fn a_single_token_response_reports_no_decode_rate() {
        let (state, _) = collect(&[content("only")], false);
        let metrics = state.metrics(Instant::now() - Duration::from_millis(50));

        // One token is entirely prefill; a decode rate would be meaningless.
        assert_eq!(metrics.decode_tokens_per_second, 0.0);
    }
}
