//! Package identity for a container-backed model.
//!
//! The native inventory derived identity from a canonicalized local path plus per-file content
//! digests. A container-backed model has neither: its weights live wherever the engine put them
//! inside the container, and what actually determines behavior is the serving image.
//!
//! So the image is the identity. The client's schema requires exactly 64 lowercase hex
//! characters, and a Docker image content digest already is a sha256 — but only once the image
//! exists locally. Before that, the digest is derived from the image reference so the value is
//! deterministic, computable offline, and changes exactly when the pinned image does. That is
//! the semantics the client's update state wants: a changed digest means a new version.

use icn_contracts::models::{
    ModelFile, ModelFileId, ModelFileRole, ModelPackage, ModelPackageProperties, ModelPackageSource,
};
use sha2::{Digest, Sha256};

use crate::controller::EimModelDefinition;

/// The digest recorded for a model whose image is not present yet.
///
/// Deterministic in the image reference alone, so two ICNs describing the same pinned image agree
/// without either having pulled it.
#[must_use]
pub fn provisional_digest(image: &str) -> String {
    format!("{:x}", Sha256::digest(image.as_bytes()))
}

/// Builds the package the client sees for one model.
#[must_use]
pub fn model_package(definition: &EimModelDefinition) -> ModelPackage {
    ModelPackage {
        id: definition.package_id.clone(),
        source: ModelPackageSource::HuggingFace {
            repository: definition.canonical_name.clone(),
            // Pinning a commit belongs with acquisition, which does not exist yet. `main` is the
            // honest placeholder: it says the weights are whatever the engine resolves.
            revision: "main".to_owned(),
        },
        files: vec![ModelFile {
            id: ModelFileId(format!("{}/image", definition.package_id.0)),
            // Relative to the in-container weight cache, in EIM's Local Directory layout.
            path: std::path::PathBuf::from(&definition.canonical_name),
            role: ModelFileRole::Weights,
            size_bytes: definition.weight_bytes,
            tensor_storage_bytes: Some(definition.geometry.total_parameters * 2),
            sha256: provisional_digest(&definition.image),
        }],
        // No projector and no draft model: vLLM bundles a vision tower into the same repository,
        // and no shipped EIM profile declares speculative decoding.
        relationships: Vec::new(),
        properties: ModelPackageProperties {
            format: "safetensors".to_owned(),
            quantization: "bf16".to_owned(),
            quantization_name: "BF16".to_owned(),
            architecture: definition
                .architecture
                .clone()
                .unwrap_or_else(|| "unknown".to_owned()),
            maximum_context_length: Some(definition.geometry.max_position_embeddings),
            intrinsic_model_id: Some(definition.canonical_name.clone()),
            intrinsic_quality_id: Some("bf16".to_owned()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::ModelGeometry;
    use crate::properties::ReasoningDeclaration;
    use icn_contracts::ModelModalities;
    use icn_contracts::models::{ModelPackageId, ModelServingConfigurationId};

    fn definition() -> EimModelDefinition {
        EimModelDefinition {
            configuration_id: ModelServingConfigurationId("cfg".to_owned()),
            package_id: ModelPackageId("eim--Qwen--Qwen3-8B--v1".to_owned()),
            catalog_model_id: "qwen-qwen3-8b".to_owned(),
            catalog_variant_id: "vllm-bf16:tp2".to_owned(),
            canonical_name: "Qwen/Qwen3-8B".to_owned(),
            display_name: "Qwen3 8B".to_owned(),
            variant_label: "bf16".to_owned(),
            description: String::new(),
            release_date: "2025-04-29".to_owned(),
            license: "Apache-2.0".to_owned(),
            quality_score: 24.0,
            quality_score_provenance: "curated".to_owned(),
            image: "magnitude-eim-xeon-qwen-qwen3-8b:v1".to_owned(),
            eim_profile_id: "vllm-xeon-bf16-tp2".to_owned(),
            tensor_parallel_size: 2,
            context_tokens: 32_768,
            geometry: ModelGeometry {
                total_parameters: 8_200_000_000,
                weight_bytes: None,
                active_parameters: 8_200_000_000,
                num_hidden_layers: 36,
                num_key_value_heads: 8,
                head_dim: 128,
                max_position_embeddings: 40_960,
                sliding_window: None,
                vision: false,
            },
            architecture: Some("Qwen3ForCausalLM".to_owned()),
            sliding_window_tokens: 0,
            reasoning: ReasoningDeclaration::dual_mode("high"),
            modalities: ModelModalities::default(),
            tool_call_parser: Some("hermes".to_owned()),
            reasoning_parser: Some("qwen3".to_owned()),
            hf_token_required: false,
            weight_bytes: 16_400_000_000,
        }
    }

    #[test]
    fn the_digest_is_sixty_four_lowercase_hex_characters() {
        let digest = provisional_digest("magnitude-eim-xeon-qwen-qwen3-8b:v1");

        // The client's schema enforces exactly this shape.
        assert_eq!(digest.len(), 64);
        assert!(
            digest
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn the_digest_changes_only_when_the_image_does() {
        let baseline = provisional_digest("magnitude-eim-xeon-qwen-qwen3-8b:v1");

        assert_eq!(
            baseline,
            provisional_digest("magnitude-eim-xeon-qwen-qwen3-8b:v1"),
            "the same reference must always agree"
        );
        assert_ne!(
            baseline,
            provisional_digest("magnitude-eim-xeon-qwen-qwen3-8b:v2"),
            "a new pinned image is a new version"
        );
    }

    #[test]
    fn the_package_carries_one_weights_file_identified_by_the_image() {
        let package = model_package(&definition());

        assert_eq!(package.files.len(), 1);
        assert_eq!(package.files[0].role, ModelFileRole::Weights);
        assert_eq!(
            package.files[0].sha256,
            provisional_digest(&definition().image)
        );
        // Tensor storage is the bf16 footprint, which is what the client shows as model size.
        assert_eq!(package.files[0].tensor_storage_bytes, Some(16_400_000_000));
    }

    #[test]
    fn declares_safetensors_rather_than_gguf() {
        let package = model_package(&definition());

        assert_eq!(package.properties.format, "safetensors");
        assert_eq!(package.properties.quantization, "bf16");
        assert_eq!(
            package.properties.maximum_context_length,
            Some(40_960),
            "the model's trained context, not the serving limit"
        );
    }

    #[test]
    fn has_no_relationships_because_there_is_no_projector_or_draft() {
        assert!(model_package(&definition()).relationships.is_empty());
    }
}
