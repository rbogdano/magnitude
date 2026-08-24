//! Synthesizing `ModelProperties` for a container-backed model.
//!
//! `icn-api` calls `properties()` unconditionally on three paths, so every field must have a
//! value even though a vLLM container exposes far less about itself than a loaded GGUF did.
//! Where a field has no container equivalent it gets an honest sentinel rather than a plausible
//! fabrication: `chat_template` says the template lives server-side instead of inventing Jinja,
//! and `execution` reports defaults because every one of its members — GPU layer counts, KV
//! cache types, flash attention — is a llama.cpp concept with no vLLM analogue.

use std::collections::BTreeMap;
use std::path::PathBuf;

use icn_contracts::{
    AutomaticReasoningBudget, ExecutionConfig, ExecutionConfigReport, ModelModalities,
    ModelProperties, NativeReasoningControls, NormalizedReasoningEffort, ReasoningEffortMapping,
    ReasoningProfile, SpeculativeDecodingRuntimeProperties, TemplateCapabilities,
};
use sha2::{Digest, Sha256};

/// The sentinel recorded where a llama.cpp chat template would have been.
///
/// Visible in `/props`, so an operator reading it learns the truth rather than seeing a
/// convincing-looking template that ICN never applies.
pub const SERVER_SIDE_TEMPLATE: &str = "eim-vllm-server-side";

/// Reasoning support as declared by the catalog overlay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningDeclaration {
    /// Effort names in ascending order. Empty when the model does not reason.
    pub efforts: Vec<String>,
    pub default_effort: Option<String>,
}

impl ReasoningDeclaration {
    /// A model with no thinking mode.
    #[must_use]
    pub fn unsupported() -> Self {
        Self {
            efforts: Vec::new(),
            default_effort: None,
        }
    }

    /// The common dual-mode shape: `none` turns thinking off, anything else turns it on.
    #[must_use]
    pub fn dual_mode(default_effort: &str) -> Self {
        Self {
            efforts: vec!["none".to_owned(), "high".to_owned()],
            default_effort: Some(default_effort.to_owned()),
        }
    }

    #[must_use]
    pub fn supported(&self) -> bool {
        !self.efforts.is_empty()
    }
}

/// Everything needed to describe a served model, gathered from the catalog overlay plus the
/// live server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPropertiesSpec {
    /// The name the server reported from `GET /v1/models`, which may differ from the catalog
    /// identifier and is what requests must address.
    pub served_model_name: String,
    /// In-container weight cache directory. Never opened by ICN; recorded for diagnostics.
    pub container_model_path: PathBuf,
    pub model_size_bytes: u64,
    /// From the model's `config.json` `architectures[0]`, e.g. `Qwen3MoeForCausalLM`.
    pub architecture: Option<String>,
    /// `--max-model-len` as passed to vLLM. Authoritative, because we chose it.
    pub context_tokens: u32,
    /// `max_position_embeddings` from `config.json`.
    pub training_context_tokens: u32,
    /// `sliding_window`, or zero when the model has none.
    pub sliding_window_tokens: i32,
    /// Whether the fork configured a tool-call parser for this model. When false, vLLM emits
    /// tool calls as plain text and the agent loop cannot use them.
    pub tools: bool,
    pub reasoning: ReasoningDeclaration,
    pub modalities: ModelModalities,
    /// Container image content digest, part of the template fingerprint.
    pub image_digest: String,
    /// The pinned EIM profile, part of the template fingerprint.
    pub eim_profile_id: String,
}

impl ModelPropertiesSpec {
    /// Identity of the exact serving setup, recomputed whenever anything material changes.
    ///
    /// `validate_apply_template_request` and request finalization compare a resolved reasoning
    /// control's fingerprint against this, so it must be stable for one deployment and must
    /// change when the image, the profile, the served name, or the reasoning shape changes.
    #[must_use]
    pub fn template_fingerprint(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(self.image_digest.as_bytes());
        digest.update([0]);
        digest.update(self.eim_profile_id.as_bytes());
        digest.update([0]);
        digest.update(self.served_model_name.as_bytes());
        digest.update([0]);
        digest.update(
            self.reasoning
                .default_effort
                .as_deref()
                .unwrap_or("")
                .as_bytes(),
        );
        for effort in &self.reasoning.efforts {
            digest.update([0]);
            digest.update(effort.as_bytes());
        }
        digest.update([0]);
        digest.update([u8::from(self.tools)]);
        format!("{:x}", digest.finalize())
    }

    /// Builds the properties ICN publishes for this model.
    #[must_use]
    pub fn to_model_properties(&self) -> ModelProperties {
        let fingerprint = self.template_fingerprint();
        ModelProperties {
            model_path: self.container_model_path.clone(),
            model_size_bytes: self.model_size_bytes,
            architecture: self.architecture.clone(),
            name: Some(self.served_model_name.clone()),
            context_tokens: self.context_tokens,
            training_context_tokens: self.training_context_tokens,
            sliding_window_tokens: self.sliding_window_tokens,
            chat_template: SERVER_SIDE_TEMPLATE.to_owned(),
            capabilities: self.template_capabilities(),
            reasoning: self.reasoning_profile(&fingerprint),
            modalities: self.modalities,
            // No EIM profile declares any speculative-decoding argument, so saying so is
            // more useful than reporting a disabled configuration that was never considered.
            speculative: SpeculativeDecodingRuntimeProperties::Disabled {
                reason: "eim_profiles_do_not_declare_speculative_decoding".to_owned(),
            },
            execution: ExecutionConfigReport {
                requested: ExecutionConfig::default(),
                resolved: ExecutionConfig::default(),
            },
            template_fingerprint: fingerprint,
        }
    }

    fn template_capabilities(&self) -> TemplateCapabilities {
        TemplateCapabilities {
            string_content: true,
            // Only a multimodal model accepts the typed content-part form.
            typed_content: self.modalities.vision || self.modalities.audio || self.modalities.video,
            tools: self.tools,
            tool_calls: self.tools,
            parallel_tool_calls: self.tools,
            system_role: true,
            preserve_reasoning: self.reasoning.supported(),
            // vLLM streams tool-call arguments as a JSON string, never as an object.
            object_arguments: false,
            enable_thinking: self.reasoning.supported(),
        }
    }

    fn reasoning_profile(&self, fingerprint: &str) -> ReasoningProfile {
        ReasoningProfile {
            default_effort: self
                .reasoning
                .default_effort
                .as_ref()
                .map(|effort| NormalizedReasoningEffort(effort.clone())),
            mappings: self
                .reasoning
                .efforts
                .iter()
                .map(|effort| ReasoningEffortMapping {
                    effort: NormalizedReasoningEffort(effort.clone()),
                    controls: NativeReasoningControls {
                        // `none` is the only effort that turns thinking off; every other
                        // level maps to the same template switch, since vLLM exposes no
                        // graduated thinking control.
                        enable_thinking: Some(effort != "none"),
                        template_args: BTreeMap::new(),
                    },
                    automatic_budget: AutomaticReasoningBudget::Disabled,
                })
                .collect(),
            template_fingerprint: fingerprint.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ModelPropertiesSpec {
        ModelPropertiesSpec {
            served_model_name: "Qwen/Qwen3-30B-A3B".to_owned(),
            container_model_path: PathBuf::from("/workspace/model-cache/Qwen/Qwen3-30B-A3B"),
            model_size_bytes: 61_000_000_000,
            architecture: Some("Qwen3MoeForCausalLM".to_owned()),
            context_tokens: 32_768,
            training_context_tokens: 40_960,
            sliding_window_tokens: 0,
            tools: true,
            reasoning: ReasoningDeclaration::dual_mode("high"),
            modalities: ModelModalities::default(),
            image_digest: "d0508be469accfdd25a1a97b5a50ece67db5c5ca23031cbed45fdbf8d0a7bb89"
                .to_owned(),
            eim_profile_id: "vllm-xeon-bf16-tp2".to_owned(),
        }
    }

    #[test]
    fn reports_the_served_name_and_our_chosen_context() {
        let properties = spec().to_model_properties();

        assert_eq!(properties.name.as_deref(), Some("Qwen/Qwen3-30B-A3B"));
        // The context is what we passed as --max-model-len, not the model's trained maximum.
        assert_eq!(properties.context_tokens, 32_768);
        assert_eq!(properties.training_context_tokens, 40_960);
    }

    #[test]
    fn records_an_honest_sentinel_instead_of_inventing_a_chat_template() {
        assert_eq!(
            spec().to_model_properties().chat_template,
            SERVER_SIDE_TEMPLATE
        );
    }

    #[test]
    fn states_why_speculative_decoding_is_unavailable() {
        match spec().to_model_properties().speculative {
            SpeculativeDecodingRuntimeProperties::Disabled { reason } => {
                assert_eq!(reason, "eim_profiles_do_not_declare_speculative_decoding");
            }
            other => panic!("expected Disabled, got {other:?}"),
        }
    }

    #[test]
    fn ties_tool_capabilities_to_whether_a_parser_is_configured() {
        let with_tools = spec().to_model_properties().capabilities;
        assert!(with_tools.tools && with_tools.tool_calls && with_tools.parallel_tool_calls);

        // Without a parser vLLM returns tool calls as prose, so claiming support would make
        // the agent loop fail silently instead of the model being marked unusable.
        let without = ModelPropertiesSpec {
            tools: false,
            ..spec()
        }
        .to_model_properties()
        .capabilities;
        assert!(!without.tools && !without.tool_calls && !without.parallel_tool_calls);
    }

    #[test]
    fn never_claims_object_shaped_tool_arguments() {
        // vLLM streams `arguments` as a JSON string in every version we target.
        assert!(!spec().to_model_properties().capabilities.object_arguments);
    }

    #[test]
    fn advertises_typed_content_only_for_multimodal_models() {
        assert!(!spec().to_model_properties().capabilities.typed_content);

        let vision = ModelPropertiesSpec {
            modalities: ModelModalities {
                vision: true,
                audio: false,
                video: false,
            },
            ..spec()
        };
        assert!(vision.to_model_properties().capabilities.typed_content);
    }

    #[test]
    fn maps_dual_mode_reasoning_onto_the_template_switch() {
        let reasoning = spec().to_model_properties().reasoning;

        assert_eq!(
            reasoning
                .default_effort
                .as_ref()
                .map(|effort| effort.0.as_str()),
            Some("high")
        );
        assert_eq!(reasoning.mappings.len(), 2);

        let none = reasoning
            .mappings
            .iter()
            .find(|mapping| mapping.effort.0 == "none")
            .expect("a none mapping");
        assert_eq!(none.controls.enable_thinking, Some(false));

        let high = reasoning
            .mappings
            .iter()
            .find(|mapping| mapping.effort.0 == "high")
            .expect("a high mapping");
        assert_eq!(high.controls.enable_thinking, Some(true));
    }

    #[test]
    fn a_non_reasoning_model_declares_no_efforts() {
        let properties = ModelPropertiesSpec {
            reasoning: ReasoningDeclaration::unsupported(),
            ..spec()
        }
        .to_model_properties();

        assert!(properties.reasoning.mappings.is_empty());
        assert!(properties.reasoning.default_effort.is_none());
        // These drive the client's reasoning affordances; both must be off.
        assert!(!properties.capabilities.enable_thinking);
        assert!(!properties.capabilities.preserve_reasoning);
    }

    #[test]
    fn the_fingerprint_is_stable_for_one_deployment() {
        assert_eq!(spec().template_fingerprint(), spec().template_fingerprint());
        assert_eq!(spec().template_fingerprint().len(), 64);
    }

    #[test]
    fn the_fingerprint_changes_with_every_material_input() {
        let baseline = spec().template_fingerprint();

        let variants = [
            ModelPropertiesSpec {
                image_digest: "0".repeat(64),
                ..spec()
            },
            ModelPropertiesSpec {
                eim_profile_id: "vllm-xeon-bf16-tp1".to_owned(),
                ..spec()
            },
            ModelPropertiesSpec {
                served_model_name: "other".to_owned(),
                ..spec()
            },
            ModelPropertiesSpec {
                reasoning: ReasoningDeclaration::unsupported(),
                ..spec()
            },
            ModelPropertiesSpec {
                tools: false,
                ..spec()
            },
        ];
        for variant in variants {
            assert_ne!(
                variant.template_fingerprint(),
                baseline,
                "fingerprint must change: {variant:?}"
            );
        }
    }

    #[test]
    fn the_reasoning_profile_carries_the_same_fingerprint_as_the_properties() {
        let properties = spec().to_model_properties();

        // A mismatch here makes every resolved-reasoning request fail validation.
        assert_eq!(
            properties.reasoning.template_fingerprint,
            properties.template_fingerprint
        );
    }
}
