//! `ChatRequest` to vLLM `POST /v1/chat/completions`.
//!
//! `CompletionBackend::complete` receives structured messages, not a rendered prompt —
//! templating is a separate seam — so this is a field-for-field mapping rather than a
//! tokenizer. That is why no Jinja engine or tokenizer is needed on the Rust side at all.
//!
//! Two ICN concepts have no vLLM equivalent and are deliberately dropped rather than faked:
//! `cache_prompt` (vLLM's prefix caching is a server-side decision) and `timings_per_token`
//! (there is no per-token timing channel, so metrics are measured client-side instead).

use std::collections::BTreeMap;

use base64::Engine as _;
use icn_contracts::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatRole, ReasoningControl,
    ResponseFormat, ToolChoice,
};
use serde_json::{Map, Value, json};

/// Builds the request body vLLM expects.
///
/// `served_model_name` must be what the server reported from `GET /v1/models`, not the Hugging
/// Face identifier: vLLM validates the `model` member against its own served name, and EIM's
/// own test helper carries a comment warning about exactly this mismatch.
#[must_use]
pub fn to_vllm_request(
    request: &ChatRequest,
    served_model_name: &str,
    served_context_tokens: u32,
) -> Value {
    let template = &request.template;

    let mut body = Map::new();
    body.insert("model".into(), json!(served_model_name));
    body.insert("stream".into(), json!(true));
    // Without this the final chunk carries no token accounting and `Generation` would have to
    // guess its own prompt and completion counts.
    body.insert("stream_options".into(), json!({ "include_usage": true }));
    body.insert(
        "messages".into(),
        Value::Array(template.messages.iter().map(to_vllm_message).collect()),
    );

    // Omitted when it would not leave room for a prompt, rather than forwarded into a certain
    // refusal. vLLM enforces `prompt + completion <= max-model-len` and rejects the whole request
    // with "This model's maximum context length is N tokens, however you requested ..."; with the
    // parameter absent it generates until the window is full, which is what a request for the whole
    // window means. llama.cpp behaved that way by truncating, so callers were built expecting it.
    let fits_a_prompt = served_context_tokens == 0 || request.max_tokens < served_context_tokens;
    if request.max_tokens > 0 && fits_a_prompt {
        body.insert("max_tokens".into(), json!(request.max_tokens));
    }
    body.insert("temperature".into(), json!(request.temperature));
    body.insert("top_p".into(), json!(request.top_p));
    if request.seed > 0 {
        body.insert("seed".into(), json!(request.seed));
    }
    if !request.stop.is_empty() {
        body.insert("stop".into(), json!(request.stop));
    }
    if request.ignore_eos {
        // A vLLM extension, not core OpenAI, but supported and needed for benchmarking.
        body.insert("ignore_eos".into(), json!(true));
    }

    if !template.tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(
                template
                    .tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters,
                            },
                        })
                    })
                    .collect(),
            ),
        );
        body.insert(
            "parallel_tool_calls".into(),
            json!(template.parallel_tool_calls),
        );
    }

    if let Some(tool_choice) = to_vllm_tool_choice(&template.tool_choice) {
        body.insert("tool_choice".into(), tool_choice);
    }

    if let Some(response_format) = to_vllm_response_format(&template.response_format) {
        body.insert("response_format".into(), response_format);
    }

    let template_kwargs = to_chat_template_kwargs(&template.reasoning, &template.template_args);
    if !template_kwargs.is_empty() {
        body.insert(
            "chat_template_kwargs".into(),
            Value::Object(template_kwargs.into_iter().collect()),
        );
    }

    Value::Object(body)
}

fn to_vllm_message(message: &ChatMessage) -> Value {
    let mut rendered = Map::new();
    rendered.insert("role".into(), json!(role_name(message.role)));

    match &message.content {
        None => {
            // An assistant turn that only carries tool calls still needs a content member;
            // omitting it makes some templates render nothing at all.
            if !message.tool_calls.is_empty() {
                rendered.insert("content".into(), Value::Null);
            } else {
                rendered.insert("content".into(), json!(""));
            }
        }
        Some(ChatContent::Text(text)) => {
            rendered.insert("content".into(), json!(text));
        }
        Some(ChatContent::Parts(parts)) => {
            rendered.insert(
                "content".into(),
                Value::Array(parts.iter().map(to_vllm_content_part).collect()),
            );
        }
    }

    if let Some(reasoning) = &message.reasoning {
        // Passing prior reasoning back lets a template that preserves it rebuild the turn.
        rendered.insert("reasoning_content".into(), json!(reasoning));
    }
    if !message.tool_calls.is_empty() {
        rendered.insert(
            "tool_calls".into(),
            Value::Array(
                message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": { "name": call.name, "arguments": call.arguments },
                        })
                    })
                    .collect(),
            ),
        );
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        rendered.insert("tool_call_id".into(), json!(tool_call_id));
    }

    Value::Object(rendered)
}

fn to_vllm_content_part(part: &ChatContentPart) -> Value {
    match part {
        ChatContentPart::Text { text } => json!({ "type": "text", "text": text }),
        ChatContentPart::Image(image) => {
            // ICN holds validated local bytes; vLLM takes a data URI.
            let encoded = base64::engine::general_purpose::STANDARD.encode(image.bytes());
            json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{};base64,{encoded}", image.media_type()) },
            })
        }
    }
}

const fn role_name(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}

fn to_vllm_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        // Auto is the server default; sending it adds nothing and some builds reject it
        // alongside an empty tool list.
        ToolChoice::Auto => None,
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Function { name } => Some(json!({
            "type": "function",
            "function": { "name": name },
        })),
        // vLLM has no allowed-tools member. Degrading to the mode alone keeps the request
        // valid; the caller has already narrowed `tools` to the allowed set.
        ToolChoice::AllowedTools { mode, .. } => Some(match mode {
            icn_contracts::AllowedToolsMode::Required => json!("required"),
            icn_contracts::AllowedToolsMode::Auto => json!("auto"),
        }),
    }
}

fn to_vllm_response_format(format: &ResponseFormat) -> Option<Value> {
    match format {
        ResponseFormat::Text => None,
        ResponseFormat::JsonObject => Some(json!({ "type": "json_object" })),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => Some(json!({
            "type": "json_schema",
            "json_schema": { "name": name, "schema": schema, "strict": strict },
        })),
        // GBNF has no OpenAI-shaped equivalent; vLLM exposes it as a guided-decoding extra.
        ResponseFormat::Grammar { grammar } => Some(json!({
            "type": "text",
            "guided_grammar": grammar,
        })),
    }
}

/// Reasoning control and caller template arguments, merged into one `chat_template_kwargs`.
///
/// Caller arguments are applied first so a resolved reasoning profile wins on conflict: the
/// profile is what the effort mapping promised, and silently overriding it would make the
/// selected effort a lie.
fn to_chat_template_kwargs(
    reasoning: &ReasoningControl,
    template_args: &BTreeMap<String, Value>,
) -> BTreeMap<String, Value> {
    let mut kwargs: BTreeMap<String, Value> = template_args.clone();

    match reasoning {
        ReasoningControl::ModelDefault => {}
        ReasoningControl::Disabled => {
            kwargs.insert("enable_thinking".into(), json!(false));
        }
        ReasoningControl::Enabled { .. } => {
            // `budget_tokens` has no vLLM equivalent; the template switch is all we can honor.
            kwargs.insert("enable_thinking".into(), json!(true));
        }
        ReasoningControl::Resolved { controls, .. } => {
            for (name, value) in &controls.template_args {
                kwargs.insert(name.clone(), value.clone());
            }
            if let Some(enable_thinking) = controls.enable_thinking {
                kwargs.insert("enable_thinking".into(), json!(enable_thinking));
            }
        }
    }

    kwargs
}

#[cfg(test)]
mod tests {
    /// Comfortably above every `max_tokens` these cases use, so the completion bound is forwarded.
    const SERVED_CONTEXT: u32 = 32_768;

    use super::*;
    use icn_contracts::{
        AllowedToolsMode, AutomaticReasoningBudget, ChatTemplateRequest, ImageInput,
        NativeReasoningControls, NormalizedReasoningEffort, ToolCall, ToolDefinition,
    };

    fn request(template: ChatTemplateRequest) -> ChatRequest {
        ChatRequest {
            template,
            stop: Vec::new(),
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            seed: 0,
            cache_prompt: true,
            ignore_eos: false,
            timings_per_token: false,
        }
    }

    fn template(messages: Vec<ChatMessage>) -> ChatTemplateRequest {
        ChatTemplateRequest {
            messages,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            reasoning: ReasoningControl::ModelDefault,
            response_format: ResponseFormat::Text,
            template_args: BTreeMap::new(),
        }
    }

    fn user(text: &str) -> ChatMessage {
        ChatMessage::text(ChatRole::User, text)
    }

    #[test]
    fn always_streams_and_asks_for_usage() {
        let body = to_vllm_request(
            &request(template(vec![user("hi")])),
            "Qwen/Qwen3-8B",
            SERVED_CONTEXT,
        );

        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"]["include_usage"], json!(true));
        assert_eq!(body["model"], json!("Qwen/Qwen3-8B"));
    }

    #[test]
    fn uses_the_served_model_name_not_the_catalog_identifier() {
        // vLLM validates `model` against its own served name; EIM's helper warns about this.
        let body = to_vllm_request(
            &request(template(vec![user("hi")])),
            "served-alias",
            SERVED_CONTEXT,
        );

        assert_eq!(body["model"], json!("served-alias"));
    }

    #[test]
    fn drops_cache_prompt_and_timings_per_token() {
        let mut chat = request(template(vec![user("hi")]));
        chat.cache_prompt = true;
        chat.timings_per_token = true;
        let body = to_vllm_request(&chat, "m", SERVED_CONTEXT);

        // Neither has a vLLM equivalent; sending an unknown member risks a 400.
        assert!(body.get("cache_prompt").is_none());
        assert!(body.get("timings_per_token").is_none());
    }

    #[test]
    fn omits_a_completion_bound_that_leaves_no_room_for_a_prompt() {
        // vLLM enforces `prompt + completion <= max-model-len` and refuses the whole request:
        // "This model's maximum context length is 16384 tokens. However you requested 16384 output
        // tokens and your prompt contains ...". Omitting the bound means "until the window is full",
        // which is what was asked for and what llama.cpp did by truncating.
        let mut chat = request(template(vec![user("hi")]));
        chat.max_tokens = 16_384;

        let body = to_vllm_request(&chat, "m", 16_384);

        assert!(body.get("max_tokens").is_none(), "{body}");
    }

    #[test]
    fn forwards_a_completion_bound_the_window_can_hold() {
        let mut chat = request(template(vec![user("hi")]));
        chat.max_tokens = 8_192;

        let body = to_vllm_request(&chat, "m", 16_384);

        assert_eq!(body["max_tokens"], serde_json::json!(8_192));
    }

    #[test]
    fn an_unknown_served_context_forwards_the_bound_unchanged() {
        // Better to let the engine speak than to silently drop a caller's limit on a guess.
        let mut chat = request(template(vec![user("hi")]));
        chat.max_tokens = 99_999;

        let body = to_vllm_request(&chat, "m", 0);

        assert_eq!(body["max_tokens"], serde_json::json!(99_999));
    }

    #[test]
    fn omits_a_zero_seed_and_zero_max_tokens() {
        let mut chat = request(template(vec![user("hi")]));
        chat.seed = 0;
        chat.max_tokens = 0;
        let body = to_vllm_request(&chat, "m", SERVED_CONTEXT);

        // Zero means unset in ICN; forwarding it would pin sampling to seed 0.
        assert!(body.get("seed").is_none());
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn forwards_stop_sequences_and_ignore_eos_only_when_set() {
        let mut chat = request(template(vec![user("hi")]));
        assert!(
            to_vllm_request(&chat, "m", SERVED_CONTEXT)
                .get("stop")
                .is_none()
        );
        assert!(
            to_vllm_request(&chat, "m", SERVED_CONTEXT)
                .get("ignore_eos")
                .is_none()
        );

        chat.stop = vec!["\n\n".to_owned()];
        chat.ignore_eos = true;
        let body = to_vllm_request(&chat, "m", SERVED_CONTEXT);

        assert_eq!(body["stop"], json!(["\n\n"]));
        assert_eq!(body["ignore_eos"], json!(true));
    }

    #[test]
    fn renders_an_image_part_as_a_data_uri() {
        let mut chat = template(vec![ChatMessage {
            role: ChatRole::User,
            content: Some(ChatContent::Parts(vec![
                ChatContentPart::Text {
                    text: "what is this".to_owned(),
                },
                ChatContentPart::Image(ImageInput::new("image/png", vec![1_u8, 2, 3])),
            ])),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }]);
        chat.tool_choice = ToolChoice::Auto;
        let body = to_vllm_request(&request(chat), "m", SERVED_CONTEXT);

        let parts = body["messages"][0]["content"].as_array().expect("parts");
        assert_eq!(parts[0]["type"], json!("text"));
        assert_eq!(parts[1]["type"], json!("image_url"));
        assert_eq!(
            parts[1]["image_url"]["url"],
            json!("data:image/png;base64,AQID")
        );
    }

    #[test]
    fn renders_a_tool_call_turn_and_its_result() {
        let assistant = ChatMessage {
            role: ChatRole::Assistant,
            content: None,
            reasoning: Some("I should read it".to_owned()),
            tool_calls: vec![ToolCall {
                id: "call_1".to_owned(),
                name: "read_file".to_owned(),
                arguments: r#"{"path":"a.rs"}"#.to_owned(),
            }],
            tool_call_id: None,
        };
        let result = ChatMessage {
            role: ChatRole::Tool,
            content: Some(ChatContent::Text("fn main() {}".to_owned())),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: Some("call_1".to_owned()),
        };
        let body = to_vllm_request(
            &request(template(vec![assistant, result])),
            "m",
            SERVED_CONTEXT,
        );

        let turn = &body["messages"][0];
        assert_eq!(turn["role"], json!("assistant"));
        // A tool-call-only turn keeps an explicit null content rather than omitting it.
        assert_eq!(turn["content"], Value::Null);
        assert_eq!(turn["reasoning_content"], json!("I should read it"));
        assert_eq!(turn["tool_calls"][0]["id"], json!("call_1"));
        assert_eq!(turn["tool_calls"][0]["type"], json!("function"));
        assert_eq!(
            turn["tool_calls"][0]["function"]["name"],
            json!("read_file")
        );

        assert_eq!(body["messages"][1]["role"], json!("tool"));
        assert_eq!(body["messages"][1]["tool_call_id"], json!("call_1"));
    }

    #[test]
    fn sends_tools_with_parallel_tool_calls() {
        let mut chat = template(vec![user("read a.rs")]);
        chat.tools = vec![ToolDefinition {
            name: "read_file".to_owned(),
            description: Some("Read a file".to_owned()),
            parameters: json!({ "type": "object", "properties": {} }),
        }];
        chat.parallel_tool_calls = false;
        let body = to_vllm_request(&request(chat), "m", SERVED_CONTEXT);

        assert_eq!(body["tools"][0]["type"], json!("function"));
        assert_eq!(body["tools"][0]["function"]["name"], json!("read_file"));
        assert_eq!(body["parallel_tool_calls"], json!(false));
    }

    #[test]
    fn omits_tool_members_entirely_when_no_tools_are_offered() {
        let body = to_vllm_request(&request(template(vec![user("hi")])), "m", SERVED_CONTEXT);

        assert!(body.get("tools").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
        // Auto is the server default; sending it with no tools is rejected by some builds.
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn maps_every_tool_choice_variant() {
        let choice = |choice: ToolChoice| {
            let mut chat = template(vec![user("hi")]);
            chat.tool_choice = choice;
            to_vllm_request(&request(chat), "m", SERVED_CONTEXT)
                .get("tool_choice")
                .cloned()
        };

        assert_eq!(choice(ToolChoice::Auto), None);
        assert_eq!(choice(ToolChoice::None), Some(json!("none")));
        assert_eq!(choice(ToolChoice::Required), Some(json!("required")));
        assert_eq!(
            choice(ToolChoice::Function {
                name: "read_file".to_owned()
            }),
            Some(json!({ "type": "function", "function": { "name": "read_file" } }))
        );
        // vLLM has no allowed-tools member, so this degrades to the bare mode.
        assert_eq!(
            choice(ToolChoice::AllowedTools {
                mode: AllowedToolsMode::Required,
                names: vec!["read_file".to_owned()],
            }),
            Some(json!("required"))
        );
        assert_eq!(
            choice(ToolChoice::AllowedTools {
                mode: AllowedToolsMode::Auto,
                names: Vec::new(),
            }),
            Some(json!("auto"))
        );
    }

    #[test]
    fn maps_every_response_format_variant() {
        let format = |format: ResponseFormat| {
            let mut chat = template(vec![user("hi")]);
            chat.response_format = format;
            to_vllm_request(&request(chat), "m", SERVED_CONTEXT)
                .get("response_format")
                .cloned()
        };

        assert_eq!(format(ResponseFormat::Text), None);
        assert_eq!(
            format(ResponseFormat::JsonObject),
            Some(json!({ "type": "json_object" }))
        );

        let schema = format(ResponseFormat::JsonSchema {
            name: "answer".to_owned(),
            schema: json!({ "type": "object" }),
            strict: true,
        })
        .expect("json schema");
        assert_eq!(schema["type"], json!("json_schema"));
        assert_eq!(schema["json_schema"]["name"], json!("answer"));
        assert_eq!(schema["json_schema"]["strict"], json!(true));

        let grammar = format(ResponseFormat::Grammar {
            grammar: "root ::= \"yes\"".to_owned(),
        })
        .expect("grammar");
        assert_eq!(grammar["guided_grammar"], json!("root ::= \"yes\""));
    }

    #[test]
    fn disabled_reasoning_switches_the_template_off() {
        let mut chat = template(vec![user("hi")]);
        chat.reasoning = ReasoningControl::Disabled;
        let body = to_vllm_request(&request(chat), "m", SERVED_CONTEXT);

        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            json!(false)
        );
    }

    #[test]
    fn enabled_reasoning_switches_it_on_and_ignores_the_budget() {
        let mut chat = template(vec![user("hi")]);
        chat.reasoning = ReasoningControl::Enabled {
            budget_tokens: Some(2048),
        };
        let body = to_vllm_request(&request(chat), "m", SERVED_CONTEXT);

        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], json!(true));
        // vLLM has no thinking-budget control; pretending otherwise would mislead the caller.
        assert!(body["chat_template_kwargs"].get("budget_tokens").is_none());
    }

    #[test]
    fn model_default_reasoning_sends_no_template_arguments() {
        let body = to_vllm_request(&request(template(vec![user("hi")])), "m", SERVED_CONTEXT);

        assert!(body.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn a_resolved_profile_overrides_caller_template_arguments() {
        let mut chat = template(vec![user("hi")]);
        chat.template_args = BTreeMap::from([
            ("enable_thinking".to_owned(), json!(true)),
            ("caller_only".to_owned(), json!("kept")),
        ]);
        chat.reasoning = ReasoningControl::Resolved {
            effort: NormalizedReasoningEffort("none".into()),
            controls: NativeReasoningControls {
                enable_thinking: Some(false),
                template_args: BTreeMap::from([("profile_only".to_owned(), json!(1))]),
            },
            automatic_budget: AutomaticReasoningBudget::Disabled,
            explicit_budget_tokens: None,
            template_fingerprint: "fp".to_owned(),
        };
        let kwargs = &to_vllm_request(&request(chat), "m", SERVED_CONTEXT)["chat_template_kwargs"];

        // The resolved effort wins: otherwise the selected effort would be a lie.
        assert_eq!(kwargs["enable_thinking"], json!(false));
        assert_eq!(kwargs["profile_only"], json!(1));
        // Unrelated caller arguments survive.
        assert_eq!(kwargs["caller_only"], json!("kept"));
    }

    #[test]
    fn caller_template_arguments_survive_without_any_reasoning_control() {
        let mut chat = template(vec![user("hi")]);
        chat.template_args = BTreeMap::from([("custom".to_owned(), json!("value"))]);
        let body = to_vllm_request(&request(chat), "m", SERVED_CONTEXT);

        assert_eq!(body["chat_template_kwargs"]["custom"], json!("value"));
    }
}
