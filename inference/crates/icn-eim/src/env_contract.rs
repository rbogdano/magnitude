//! The `INFERENCE_*` environment a serving container is launched with.
//!
//! EIM reads its entire configuration from environment variables once at startup, so this map
//! *is* the launch contract. Two entries carry most of the weight.
//!
//! `INFERENCE_PROFILE_ID` pins the profile explicitly rather than letting EIM's selector score
//! candidates. That is what makes the RAM estimate trustworthy: the estimate assumed a
//! particular tensor-parallel size, and auto-selection could pick another.
//!
//! `INFERENCE_ENGINE_ARGS` carries the arguments no shipped EIM profile sets. Two of them are
//! not optional for an agent: `max-model-len`, because otherwise vLLM takes the model's full
//! trained context and may refuse to start once the KV cache is sized on CPU; and the
//! tool-call parser, because without it vLLM returns tool calls as prose and the agent loop
//! silently falls apart.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use crate::docker::cli::ProxySettings;

/// In-container weight cache, matching EIM's default `INFERENCE_CACHE_PATH`.
pub const CONTAINER_CACHE_PATH: &str = "/workspace/model-cache";

/// EIM's own port. Never overridden: the base image's `HEALTHCHECK` hardcodes
/// `localhost:8000`, so a different in-container port would leave the container
/// permanently unhealthy even while serving correctly.
pub const CONTAINER_PORT: u16 = 8000;

/// Everything that varies per launch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchEnvironment {
    /// Hugging Face repository identifier, e.g. `Qwen/Qwen3-30B-A3B`.
    pub model_id: String,
    /// The EIM profile to pin, e.g. `vllm-xeon-bf16-tp2`.
    pub profile_id: String,
    /// `--max-model-len`. Must equal the context the RAM estimate was computed for.
    pub context_tokens: u32,
    /// `--max-num-seqs`, the admitted concurrent sequence count.
    pub parallel_sequences: u32,
    /// The name the server should advertise. Read back from `/v1/models` regardless.
    pub served_model_name: String,
    /// vLLM's tool-call parser for this model family. `None` marks a model whose tool calls
    /// cannot be parsed, which must also be reflected in its declared capabilities.
    pub tool_call_parser: Option<String>,
    /// vLLM's reasoning parser, for models that emit a separate thinking channel.
    pub reasoning_parser: Option<String>,
    /// Only for gated repositories, and only when configured.
    pub hf_token: Option<String>,
    /// Host proxy settings. Required on a network without direct internet access: the daemon
    /// may have a proxy for image pulls while the container has none, and vLLM downloads
    /// weights from inside the container.
    pub proxy: ProxySettings,
    /// Whether the weight cache is mounted read-only, which it is once Magnitude prefetches.
    pub offline_weights: bool,
    /// Additional engine arguments, merged last so an operator can override anything.
    pub engine_args_override: BTreeMap<String, Value>,
}

impl LaunchEnvironment {
    /// Builds the complete container environment.
    #[must_use]
    pub fn to_environment(&self) -> BTreeMap<String, String> {
        let mut environment = BTreeMap::new();

        environment.insert("INFERENCE_MODEL_ID".to_owned(), self.model_id.clone());
        environment.insert("INFERENCE_PROFILE_ID".to_owned(), self.profile_id.clone());
        environment.insert(
            "INFERENCE_CACHE_PATH".to_owned(),
            CONTAINER_CACHE_PATH.to_owned(),
        );
        environment.insert("INFERENCE_ENGINE_ARGS".to_owned(), self.engine_args_json());
        // Matches every shipped profile. Set here as well because a profile's `env_vars` never
        // override the runtime environment, so being explicit avoids relying on that ordering.
        environment.insert("VLLM_DO_NOT_TRACK".to_owned(), "1".to_owned());

        if self.offline_weights {
            // Turn a cache miss into an immediate, legible failure instead of a silent
            // multi-hundred-gigabyte download behind an opaque progress indicator.
            environment.insert("HF_HUB_OFFLINE".to_owned(), "1".to_owned());
            environment.insert("TRANSFORMERS_OFFLINE".to_owned(), "1".to_owned());
        } else {
            environment.extend(self.proxy.to_environment());
        }

        if let Some(token) = &self.hf_token {
            environment.insert("HF_TOKEN".to_owned(), token.clone());
        }

        environment
    }

    /// The `INFERENCE_ENGINE_ARGS` payload, merged over whatever the profile declares.
    ///
    /// Keys are kebab-case because that is what EIM serializes onto the vLLM command line, and
    /// what its validator checks against vLLM's real argument parser.
    #[must_use]
    pub fn engine_args_json(&self) -> String {
        let mut args = Map::new();
        args.insert("max-model-len".into(), json!(self.context_tokens));
        args.insert("max-num-seqs".into(), json!(self.parallel_sequences.max(1)));
        args.insert(
            "served-model-name".into(),
            json!(self.served_model_name.clone()),
        );

        if let Some(parser) = &self.tool_call_parser {
            // Both are required: the flag enables parsing, the parser names the format.
            args.insert("enable-auto-tool-choice".into(), json!(true));
            args.insert("tool-call-parser".into(), json!(parser));
        }
        if let Some(parser) = &self.reasoning_parser {
            args.insert("reasoning-parser".into(), json!(parser));
        }

        for (name, value) in &self.engine_args_override {
            args.insert(name.clone(), value.clone());
        }

        // Serialization cannot fail for a map of scalars, but a hard-coded fallback keeps the
        // launch path infallible rather than panicking on an impossible branch.
        serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".to_owned())
    }

    /// The mount argument for the weight cache.
    ///
    /// Read-only once Magnitude owns prefetching, which is also what makes one cache directory
    /// safely shareable across containers.
    #[must_use]
    pub fn cache_mount(&self, host_cache_path: &str) -> String {
        let mode = if self.offline_weights { "ro" } else { "rw" };
        format!("{host_cache_path}:{CONTAINER_CACHE_PATH}:{mode}")
    }
}

/// vLLM's tool-call parser for a model family, or `None` when no parser is known.
///
/// Derived from the model identifier because EIM's `metadata.yaml` records nothing about tool
/// calling — its tags only ever say `text-generation`, `chat`, `instruction`, `reasoning`, or
/// `multimodal`. Every mapping here is a claim that must be confirmed on real hardware with a
/// multi-turn tool loop; a model whose parser cannot be confirmed belongs in the `None` branch,
/// where the catalog will mark it unusable rather than letting the agent fail silently.
#[must_use]
pub fn tool_call_parser_for(canonical_name: &str) -> Option<&'static str> {
    let name = canonical_name.to_ascii_lowercase();

    // Base models have no chat template at all, so they can never carry tool calls. Checked
    // first because their names would otherwise match a family prefix below.
    if name == "google/gemma-7b" || name == "zai-org/glm-4-9b-hf" {
        return None;
    }
    // Mistral gained a function-calling template in v0.3; v0.2 does not have one.
    if name.starts_with("mistralai/mistral-7b-instruct-v0.2") {
        return None;
    }

    if name.starts_with("qwen/qwen3") {
        return Some("hermes");
    }
    if name.starts_with("openai/gpt-oss") {
        return Some("openai");
    }
    if name.starts_with("ibm-granite/granite") {
        return Some("granite");
    }
    if name.starts_with("meta-llama/llama-4") {
        return Some("llama4_pythonic");
    }
    if name.starts_with("meta-llama/llama-3") {
        return Some("llama3_json");
    }
    None
}

/// vLLM's reasoning parser for a model that emits a separate thinking channel.
#[must_use]
pub fn reasoning_parser_for(canonical_name: &str) -> Option<&'static str> {
    let name = canonical_name.to_ascii_lowercase();
    if name.starts_with("qwen/qwen3") {
        return Some("qwen3");
    }
    if name.starts_with("openai/gpt-oss") {
        return Some("openai_gptoss");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch() -> LaunchEnvironment {
        LaunchEnvironment {
            model_id: "Qwen/Qwen3-30B-A3B".to_owned(),
            profile_id: "vllm-xeon-bf16-tp2".to_owned(),
            context_tokens: 32_768,
            parallel_sequences: 4,
            served_model_name: "Qwen/Qwen3-30B-A3B".to_owned(),
            tool_call_parser: Some("hermes".to_owned()),
            reasoning_parser: Some("qwen3".to_owned()),
            hf_token: None,
            proxy: ProxySettings::default(),
            offline_weights: true,
            engine_args_override: BTreeMap::new(),
        }
    }

    fn engine_args(launch: &LaunchEnvironment) -> Value {
        serde_json::from_str(&launch.engine_args_json()).expect("valid JSON")
    }

    #[test]
    fn pins_the_profile_rather_than_letting_the_selector_score() {
        // Auto-selection could pick a different tensor-parallel size than the RAM estimate
        // assumed, which would make the estimate meaningless.
        let environment = launch().to_environment();

        assert_eq!(
            environment.get("INFERENCE_PROFILE_ID").map(String::as_str),
            Some("vllm-xeon-bf16-tp2")
        );
        assert_eq!(
            environment.get("INFERENCE_MODEL_ID").map(String::as_str),
            Some("Qwen/Qwen3-30B-A3B")
        );
    }

    #[test]
    fn never_sets_the_container_port() {
        // The base image's HEALTHCHECK hardcodes localhost:8000.
        assert!(!launch().to_environment().contains_key("INFERENCE_PORT"));
        assert_eq!(CONTAINER_PORT, 8000);
    }

    #[test]
    fn never_sets_the_engine_or_the_metric() {
        let environment = launch().to_environment();

        // Only vllm is implemented; sglang parses but raises NotImplementedError.
        assert!(!environment.contains_key("INFERENCE_ENGINE"));
        // No shipped profile declares a metric, so the variable is a no-op that would only
        // suggest a control that does not exist.
        assert!(!environment.contains_key("INFERENCE_METRIC"));
        // Reported but unused in selection, and the profile already pins tensor parallelism.
        assert!(!environment.contains_key("INFERENCE_ACCELERATOR_COUNT"));
    }

    #[test]
    fn always_injects_the_arguments_no_eim_profile_sets() {
        let args = engine_args(&launch());

        // Without max-model-len vLLM takes the trained context and may refuse to start once
        // the KV cache is sized.
        assert_eq!(args["max-model-len"], json!(32_768));
        assert_eq!(args["max-num-seqs"], json!(4));
        // Without both of these vLLM returns tool calls as prose.
        assert_eq!(args["enable-auto-tool-choice"], json!(true));
        assert_eq!(args["tool-call-parser"], json!("hermes"));
        assert_eq!(args["reasoning-parser"], json!("qwen3"));
    }

    #[test]
    fn omits_tool_arguments_for_a_model_with_no_parser() {
        let no_tools = LaunchEnvironment {
            tool_call_parser: None,
            reasoning_parser: None,
            ..launch()
        };
        let args = engine_args(&no_tools);

        // Enabling auto tool choice without a parser is worse than leaving both off.
        assert!(args.get("enable-auto-tool-choice").is_none());
        assert!(args.get("tool-call-parser").is_none());
        assert!(args.get("reasoning-parser").is_none());
    }

    #[test]
    fn clamps_a_zero_sequence_count_to_one() {
        let degenerate = LaunchEnvironment {
            parallel_sequences: 0,
            ..launch()
        };

        assert_eq!(engine_args(&degenerate)["max-num-seqs"], json!(1));
    }

    #[test]
    fn an_operator_override_wins_over_every_injected_argument() {
        let overridden = LaunchEnvironment {
            engine_args_override: BTreeMap::from([
                ("max-model-len".to_owned(), json!(8_192)),
                ("tool-call-parser".to_owned(), json!("mistral")),
                ("quantization".to_owned(), json!("awq")),
            ]),
            ..launch()
        };
        let args = engine_args(&overridden);

        assert_eq!(args["max-model-len"], json!(8_192));
        assert_eq!(args["tool-call-parser"], json!("mistral"));
        // An argument no shipped profile uses at all is still reachable this way.
        assert_eq!(args["quantization"], json!("awq"));
    }

    #[test]
    fn forces_offline_weights_when_the_cache_is_read_only() {
        let environment = launch().to_environment();

        // A cache miss becomes an instant failure rather than an invisible re-download.
        assert_eq!(
            environment.get("HF_HUB_OFFLINE").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            environment.get("TRANSFORMERS_OFFLINE").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            launch().cache_mount("/var/lib/magnitude/eim"),
            "/var/lib/magnitude/eim:/workspace/model-cache:ro"
        );
    }

    #[test]
    fn passes_the_proxy_only_when_the_container_must_reach_the_network() {
        let proxy = ProxySettings {
            http_proxy: Some("http://proxy-dmz.intel.com:911".to_owned()),
            https_proxy: Some("http://proxy-dmz.intel.com:912".to_owned()),
            no_proxy: Some("localhost,127.0.0.1,.intel.com".to_owned()),
        };

        // Offline: the proxy is irrelevant and would only be noise.
        let offline = LaunchEnvironment {
            proxy: proxy.clone(),
            offline_weights: true,
            ..launch()
        }
        .to_environment();
        assert!(!offline.contains_key("HTTPS_PROXY"));

        // Downloading: without this vLLM cannot reach Hugging Face on a proxied network, and
        // the failure looks like a missing model rather than a network problem.
        let downloading = LaunchEnvironment {
            proxy,
            offline_weights: false,
            ..launch()
        }
        .to_environment();
        assert_eq!(
            downloading.get("HTTPS_PROXY").map(String::as_str),
            Some("http://proxy-dmz.intel.com:912")
        );
        assert!(
            downloading.contains_key("https_proxy"),
            "both cases are needed"
        );
        assert!(!downloading.contains_key("HF_HUB_OFFLINE"));
        assert_eq!(
            LaunchEnvironment {
                offline_weights: false,
                ..launch()
            }
            .cache_mount("/var/lib/magnitude/eim"),
            "/var/lib/magnitude/eim:/workspace/model-cache:rw"
        );
    }

    #[test]
    fn includes_a_hugging_face_token_only_when_configured() {
        assert!(!launch().to_environment().contains_key("HF_TOKEN"));

        let gated = LaunchEnvironment {
            hf_token: Some("hf_secret".to_owned()),
            ..launch()
        };
        assert_eq!(
            gated.to_environment().get("HF_TOKEN").map(String::as_str),
            Some("hf_secret")
        );
    }

    #[test]
    fn maps_the_eim_catalog_onto_vllm_tool_call_parsers() {
        for (model, parser) in [
            ("Qwen/Qwen3-4B", Some("hermes")),
            ("Qwen/Qwen3-8B", Some("hermes")),
            ("Qwen/Qwen3-30B-A3B", Some("hermes")),
            ("Qwen/Qwen3-VL-30B-A3B-Instruct", Some("hermes")),
            ("openai/gpt-oss-20b", Some("openai")),
            ("ibm-granite/granite-3.2-2b-instruct", Some("granite")),
            ("meta-llama/Llama-3.1-8B-Instruct", Some("llama3_json")),
            ("meta-llama/Llama-3.2-3B-Instruct", Some("llama3_json")),
            (
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                Some("llama4_pythonic"),
            ),
        ] {
            assert_eq!(tool_call_parser_for(model), parser, "model: {model}");
        }
    }

    #[test]
    fn reports_no_parser_for_models_that_cannot_carry_tool_calls() {
        for model in [
            // Base models: no chat template at all.
            "google/gemma-7b",
            "zai-org/glm-4-9b-hf",
            // v0.3 added function calling; v0.2 has no such template.
            "mistralai/Mistral-7B-Instruct-v0.2",
            // Gemma has no native function-calling template.
            "google/gemma-4-E4B-it",
            // No standard vLLM parser for these Phi-4 variants.
            "microsoft/Phi-4-reasoning",
            "microsoft/Phi-4-multimodal-instruct",
        ] {
            assert_eq!(tool_call_parser_for(model), None, "model: {model}");
        }
    }

    #[test]
    fn matches_model_names_case_insensitively() {
        // Catalog identifiers and Hugging Face repositories disagree about capitalization.
        assert_eq!(tool_call_parser_for("qwen/QWEN3-8B"), Some("hermes"));
        assert_eq!(
            tool_call_parser_for("META-LLAMA/Llama-3.1-8B-Instruct"),
            Some("llama3_json")
        );
    }

    #[test]
    fn assigns_reasoning_parsers_only_to_models_with_a_thinking_channel() {
        assert_eq!(reasoning_parser_for("Qwen/Qwen3-30B-A3B"), Some("qwen3"));
        assert_eq!(
            reasoning_parser_for("openai/gpt-oss-20b"),
            Some("openai_gptoss")
        );
        assert_eq!(
            reasoning_parser_for("meta-llama/Llama-3.1-8B-Instruct"),
            None
        );
        assert_eq!(
            reasoning_parser_for("ibm-granite/granite-3.2-2b-instruct"),
            None
        );
    }
}
