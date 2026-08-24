//! Server-sent event framing and OpenAI chat-chunk decoding for the vLLM transport.
//!
//! Framing and delta extraction started as `benchmark-runner`'s endpoint client, which was
//! already a working reqwest OpenAI streaming client. Two things changed. It accumulated a
//! whole response into strings, which suits a benchmark but not `CompletionBackend::complete`,
//! so decoding is per-chunk and the caller emits events as they arrive. And it read only
//! `reasoning_content`, so the other three spellings servers actually use are handled here,
//! mirroring the precedence Magnitude's TypeScript `customEndpointChunkDecoder` already
//! established for the same wire formats.

use std::collections::BTreeMap;

use serde_json::Value;

/// Splits the next complete SSE frame out of a byte buffer, tolerating both LF and CRLF.
///
/// Returns the frame body and how many bytes to consume, or `None` when the buffer does not
/// yet hold a terminated frame.
#[must_use]
pub fn next_sse_frame(buffer: &[u8]) -> Option<(Vec<u8>, usize)> {
    for index in 0..buffer.len().saturating_sub(1) {
        if buffer[index..].starts_with(b"\n\n") {
            return Some((buffer[..index].to_vec(), index + 2));
        }
        if buffer[index..].starts_with(b"\r\n\r\n") {
            return Some((buffer[..index].to_vec(), index + 4));
        }
    }
    None
}

/// Joins the `data:` lines of one frame. Comment and `event:` lines are ignored.
#[must_use]
pub fn sse_data(frame: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(frame);
    let data = text
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>();
    (!data.is_empty()).then(|| data.join("\n"))
}

/// One incremental tool call. vLLM streams `arguments` in fragments, so these accumulate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

/// Token accounting from the final chunk, when `stream_options.include_usage` was requested.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UsageDelta {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prefix-cache hits, when the server reports them.
    pub cached_prompt_tokens: usize,
}

/// The decoded content of one `data:` payload.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatChunk {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCallDelta>,
    pub finish_reason: Option<String>,
    pub usage: Option<UsageDelta>,
}

impl ChatChunk {
    /// Whether this chunk carries anything a consumer should be told about.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_none()
            && self.reasoning.is_none()
            && self.tool_calls.is_empty()
            && self.finish_reason.is_none()
            && self.usage.is_none()
    }
}

/// What one SSE payload turned out to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChunkOutcome {
    /// The `[DONE]` sentinel: the stream is complete.
    Done,
    Chunk(ChatChunk),
    /// The server reported an error mid-stream, as a JSON `error` member.
    Error(String),
}

#[derive(Debug, thiserror::Error)]
#[error("could not decode chat chunk: {source}")]
pub struct ChunkDecodeError {
    #[source]
    pub source: serde_json::Error,
    pub payload: String,
}

/// Decodes one `data:` payload.
pub fn decode_chunk(payload: &str) -> Result<ChunkOutcome, ChunkDecodeError> {
    let payload = payload.trim();
    if payload == "[DONE]" {
        return Ok(ChunkOutcome::Done);
    }

    let value: Value = serde_json::from_str(payload).map_err(|source| ChunkDecodeError {
        source,
        payload: payload.to_owned(),
    })?;

    if let Some(error) = value.get("error").filter(|error| !error.is_null()) {
        return Ok(ChunkOutcome::Error(
            error
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| error.to_string(), str::to_owned),
        ));
    }

    let mut chunk = ChatChunk {
        usage: decode_usage(&value),
        ..ChatChunk::default()
    };

    for choice in value
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            chunk.finish_reason = Some(reason.to_owned());
        }
        let Some(delta) = choice.get("delta") else {
            continue;
        };

        if let Some(text) = non_empty_text(delta.get("content")) {
            chunk.content = Some(match chunk.content.take() {
                Some(mut existing) => {
                    existing.push_str(&text);
                    existing
                }
                None => text,
            });
        }

        if let Some(text) = decode_reasoning(delta) {
            chunk.reasoning = Some(match chunk.reasoning.take() {
                Some(mut existing) => {
                    existing.push_str(&text);
                    existing
                }
                None => text,
            });
        }

        chunk.tool_calls.extend(decode_tool_calls(delta));
    }

    Ok(ChunkOutcome::Chunk(chunk))
}

/// Reasoning text, taking the first spelling that carries anything.
///
/// Servers disagree: vLLM emits `reasoning_content`, OpenRouter emits `reasoning` and
/// sometimes a structured `reasoning_details`, and Anthropic-shaped shims emit `thinking`.
/// Some emit two representations of the same text, so first-wins is what prevents a
/// duplicated thought stream. The order matches Magnitude's TypeScript decoder exactly.
fn decode_reasoning(delta: &Value) -> Option<String> {
    non_empty_text(delta.get("reasoning_content"))
        .or_else(|| non_empty_text(delta.get("reasoning")))
        .or_else(|| decode_reasoning_details(delta.get("reasoning_details")))
        .or_else(|| non_empty_text(delta.get("thinking")))
}

/// Concatenates the `reasoning.text` entries of a structured `reasoning_details` array,
/// ignoring other detail types such as redacted or signature blocks.
fn decode_reasoning_details(details: Option<&Value>) -> Option<String> {
    let text = details?
        .as_array()?
        .iter()
        .filter(|detail| {
            detail.get("type").and_then(Value::as_str) == Some("reasoning.text")
        })
        .filter_map(|detail| non_empty_text(detail.get("text")))
        .collect::<String>();
    (!text.is_empty()).then_some(text)
}

fn decode_tool_calls(delta: &Value) -> Vec<ToolCallDelta> {
    delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|raw| {
            let function = raw.get("function");
            ToolCallDelta {
                index: raw
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|index| usize::try_from(index).ok())
                    .unwrap_or(0),
                id: non_empty_text(raw.get("id")),
                name: function.and_then(|function| non_empty_text(function.get("name"))),
                // An empty fragment is meaningful here only as "no new argument text".
                arguments: function
                    .and_then(|function| non_empty_text(function.get("arguments")))
                    .unwrap_or_default(),
            }
        })
        .collect()
}

fn decode_usage(value: &Value) -> Option<UsageDelta> {
    let usage = value.get("usage").filter(|usage| !usage.is_null())?;
    let count = |name: &str| -> usize {
        usage
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0)
    };
    Some(UsageDelta {
        prompt_tokens: count("prompt_tokens"),
        completion_tokens: count("completion_tokens"),
        cached_prompt_tokens: usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0),
    })
}

fn non_empty_text(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

/// Accumulates streamed tool-call fragments into whole calls, keyed by their stream index.
#[derive(Clone, Debug, Default)]
pub struct ToolCallAccumulator {
    calls: BTreeMap<usize, ToolCallDelta>,
}

impl ToolCallAccumulator {
    pub fn absorb(&mut self, delta: &ToolCallDelta) {
        let call = self.calls.entry(delta.index).or_insert_with(|| ToolCallDelta {
            index: delta.index,
            ..ToolCallDelta::default()
        });
        if let Some(id) = &delta.id {
            call.id = Some(id.clone());
        }
        if let Some(name) = &delta.name {
            call.name = Some(name.clone());
        }
        call.arguments.push_str(&delta.arguments);
    }

    /// The completed calls, in stream order.
    #[must_use]
    pub fn finish(self) -> Vec<icn_contracts::ToolCall> {
        self.calls
            .into_values()
            .map(|call| icn_contracts::ToolCall {
                id: call.id.unwrap_or_default(),
                name: call.name.unwrap_or_default(),
                arguments: call.arguments,
            })
            .collect()
    }
}

/// Maps OpenAI's `finish_reason` onto ICN's vocabulary.
///
/// An absent or unrecognized reason becomes `Stop`, which is what a stream that simply ended
/// means. `length` is preserved because the agent uses it to decide whether to continue.
#[must_use]
pub fn finish_reason(reason: Option<&str>) -> icn_contracts::FinishReason {
    match reason {
        Some("length") => icn_contracts::FinishReason::Length,
        Some("tool_calls") | Some("function_call") => icn_contracts::FinishReason::ToolCalls,
        _ => icn_contracts::FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(payload: &str) -> ChatChunk {
        match decode_chunk(payload).expect("decodable") {
            ChunkOutcome::Chunk(chunk) => chunk,
            other => panic!("expected a chunk, got {other:?}"),
        }
    }

    #[test]
    fn splits_frames_on_lf_and_crlf() {
        let (frame, consumed) = next_sse_frame(b"data: one\n\ndata: two\n\n").expect("a frame");
        assert_eq!(frame, b"data: one");
        assert_eq!(consumed, 11);

        let (frame, consumed) =
            next_sse_frame(b"data: one\r\n\r\ndata: two").expect("a crlf frame");
        assert_eq!(frame, b"data: one");
        assert_eq!(consumed, 13);
    }

    #[test]
    fn withholds_an_unterminated_frame() {
        assert!(next_sse_frame(b"data: partial").is_none());
    }

    #[test]
    fn joins_multiline_data_and_ignores_other_fields() {
        assert_eq!(
            sse_data(b": keep-alive\nevent: message\ndata: first\ndata: second"),
            Some("first\nsecond".to_owned())
        );
        assert_eq!(sse_data(b": keep-alive only"), None);
    }

    #[test]
    fn recognizes_the_done_sentinel() {
        assert_eq!(decode_chunk("[DONE]").expect("done"), ChunkOutcome::Done);
        assert_eq!(decode_chunk("  [DONE] ").expect("done"), ChunkOutcome::Done);
    }

    #[test]
    fn decodes_a_content_delta() {
        let chunk = chunk(r#"{"choices":[{"delta":{"content":"Hel"}}]}"#);

        assert_eq!(chunk.content.as_deref(), Some("Hel"));
        assert_eq!(chunk.reasoning, None);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn treats_an_empty_content_string_as_no_delta() {
        // Servers send empty deltas as keep-alives; forwarding them would emit blank tokens.
        assert!(chunk(r#"{"choices":[{"delta":{"content":""}}]}"#).is_empty());
    }

    #[test]
    fn prefers_reasoning_content_over_every_other_spelling() {
        let chunk = chunk(
            r#"{"choices":[{"delta":{
                "reasoning_content":"canonical",
                "reasoning":"openrouter",
                "thinking":"anthropic-shim"
            }}]}"#,
        );

        assert_eq!(chunk.reasoning.as_deref(), Some("canonical"));
    }

    #[test]
    fn falls_back_to_reasoning_then_details_then_thinking() {
        assert_eq!(
            chunk(r#"{"choices":[{"delta":{"reasoning":"second"}}]}"#)
                .reasoning
                .as_deref(),
            Some("second")
        );
        assert_eq!(
            chunk(
                r#"{"choices":[{"delta":{"reasoning_details":[
                    {"type":"reasoning.text","text":"third"}
                ]}}]}"#
            )
            .reasoning
            .as_deref(),
            Some("third")
        );
        assert_eq!(
            chunk(r#"{"choices":[{"delta":{"thinking":"fourth"}}]}"#)
                .reasoning
                .as_deref(),
            Some("fourth")
        );
    }

    #[test]
    fn does_not_duplicate_reasoning_sent_in_two_representations() {
        // OpenRouter sends `reasoning` and `reasoning_details` describing the same text.
        // Emitting both would show the user a doubled thought stream.
        let chunk = chunk(
            r#"{"choices":[{"delta":{
                "reasoning":"thought",
                "reasoning_details":[{"type":"reasoning.text","text":"thought"}]
            }}]}"#,
        );

        assert_eq!(chunk.reasoning.as_deref(), Some("thought"));
    }

    #[test]
    fn ignores_reasoning_detail_types_other_than_text() {
        let chunk = chunk(
            r#"{"choices":[{"delta":{"reasoning_details":[
                {"type":"reasoning.encrypted","data":"opaque"},
                {"type":"reasoning.text","text":"visible"},
                {"type":"reasoning.signature","signature":"sig"}
            ]}}]}"#,
        );

        assert_eq!(chunk.reasoning.as_deref(), Some("visible"));
    }

    #[test]
    fn decodes_incremental_tool_calls() {
        let opening = chunk(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"id":"call_1","function":{"name":"read_file","arguments":"{\"pa"}}
            ]}}]}"#,
        );
        let continuation = chunk(
            r#"{"choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"th\":\"a.rs\"}"}}
            ]}}]}"#,
        );

        let mut accumulator = ToolCallAccumulator::default();
        for delta in opening.tool_calls.iter().chain(continuation.tool_calls.iter()) {
            accumulator.absorb(delta);
        }
        let calls = accumulator.finish();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, r#"{"path":"a.rs"}"#);
    }

    #[test]
    fn keeps_parallel_tool_calls_separate_and_ordered() {
        let mut accumulator = ToolCallAccumulator::default();
        // Deliberately absorb the second call first; output must still be index-ordered.
        for delta in [
            ToolCallDelta { index: 1, id: Some("b".into()), name: Some("second".into()), arguments: "{}".into() },
            ToolCallDelta { index: 0, id: Some("a".into()), name: Some("first".into()), arguments: "{}".into() },
        ] {
            accumulator.absorb(&delta);
        }
        let calls = accumulator.finish();

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "first");
        assert_eq!(calls[1].name, "second");
    }

    #[test]
    fn decodes_usage_including_cached_prompt_tokens() {
        let chunk = chunk(
            r#"{"choices":[],"usage":{
                "prompt_tokens":1200,"completion_tokens":48,"total_tokens":1248,
                "prompt_tokens_details":{"cached_tokens":1024}
            }}"#,
        );
        let usage = chunk.usage.expect("usage");

        assert_eq!(usage.prompt_tokens, 1200);
        assert_eq!(usage.completion_tokens, 48);
        assert_eq!(usage.cached_prompt_tokens, 1024);
    }

    #[test]
    fn tolerates_usage_without_a_details_member() {
        let usage = chunk(r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}"#)
            .usage
            .expect("usage");

        assert_eq!(usage.cached_prompt_tokens, 0);
    }

    #[test]
    fn surfaces_a_mid_stream_error_with_its_message() {
        assert_eq!(
            decode_chunk(r#"{"error":{"message":"model not found","type":"NotFoundError"}}"#)
                .expect("decodable"),
            ChunkOutcome::Error("model not found".to_owned())
        );
    }

    #[test]
    fn surfaces_a_structureless_error_as_its_json() {
        match decode_chunk(r#"{"error":"overloaded"}"#).expect("decodable") {
            ChunkOutcome::Error(message) => assert!(message.contains("overloaded")),
            other => panic!("expected an error, got {other:?}"),
        }
    }

    #[test]
    fn reports_undecodable_payloads_with_the_offending_text() {
        let error = decode_chunk("{not json").expect_err("should fail");

        assert_eq!(error.payload, "{not json");
    }

    #[test]
    fn maps_finish_reasons_onto_icn_vocabulary() {
        use icn_contracts::FinishReason;

        assert_eq!(finish_reason(Some("stop")), FinishReason::Stop);
        assert_eq!(finish_reason(Some("length")), FinishReason::Length);
        assert_eq!(finish_reason(Some("tool_calls")), FinishReason::ToolCalls);
        // Older servers say function_call for the same thing.
        assert_eq!(finish_reason(Some("function_call")), FinishReason::ToolCalls);
        // An ended stream with no stated reason is a normal stop, not a failure.
        assert_eq!(finish_reason(None), FinishReason::Stop);
        assert_eq!(finish_reason(Some("something_new")), FinishReason::Stop);
    }

    #[test]
    fn decodes_a_finish_reason_carried_without_a_delta() {
        let chunk = chunk(r#"{"choices":[{"finish_reason":"tool_calls","delta":{}}]}"#);

        assert_eq!(chunk.finish_reason.as_deref(), Some("tool_calls"));
        assert!(chunk.content.is_none());
    }
}
