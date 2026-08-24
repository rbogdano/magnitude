//! The catalog ICN publishes to the client.
//!
//! `GET /v1/models` is not optional: ACN calls it while building its layer graph and fails to
//! become ready if it errors, so a container-backed ICN must answer it before the client can
//! start at all.
//!
//! What it reports is every model in the table, whether or not this host can serve it. A model
//! that does not fit, is gated without a credential, or has no tool-call parser stays listed
//! with its reasons legible — hiding it would leave the user unable to learn why a model they
//! expected is absent. Capabilities carry that judgement: `tools: false` is what marks a model
//! the agent cannot use.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use icn_contracts::InventoryError;
use icn_contracts::models::{
    CatalogDiagnostic, CatalogModel, CatalogModelId, CatalogModelLocalState, CatalogModels,
    CatalogVariantId, ModelCapabilities, ModelFailure, ModelParameterization,
    ModelReasoningCapabilities, ModelReleaseDate, ModelServingConfiguration,
    ModelServingConfigurationId, ModelsResponse, ReconcileCatalogModelRequest,
    ReconcileCatalogModelResponse, ServableModelBundle, ServingProfile,
};
use icn_contracts::models::{
    InstalledCatalogAttribution, InstalledModelPackage, InstalledModelPackages,
    InstalledModelPackagesResponse, ModelBundleInput, ModelDownload, ModelDownloadId,
    ModelDownloads, ModelDownloadsResponse, ModelPackageId, ModelPackageInspection,
    ModelPackageInstallationOrigin, RecommendableModel, RecommendableModelCatalog,
    RecommendableModelCatalogProvider, RemoveInstalledModelPackageResponse,
    ResolvedServableModelBundle, StartModelDownloadRequest, StartModelDownloadResponse,
};

use crate::controller::EimModelDefinition;

/// Serves the catalog from the model table.
pub struct EimCatalog {
    definitions: Arc<BTreeMap<ModelServingConfigurationId, EimModelDefinition>>,
    packages: Arc<dyn InstalledImageProbe>,
}

/// Whether a model's serving image is present locally.
///
/// Abstracted so the projection is testable without a daemon, and because "installed" will grow
/// to mean image *and* cached weights once acquisition lands.
pub trait InstalledImageProbe: Send + Sync + 'static {
    fn is_installed(&self, image: &str) -> BoxFuture<'_, bool>;
}

/// Reports nothing as installed. Used before acquisition exists and in tests.
pub struct NoImagesInstalled;

impl InstalledImageProbe for NoImagesInstalled {
    fn is_installed(&self, _image: &str) -> BoxFuture<'_, bool> {
        Box::pin(async { false })
    }
}

/// Asks Docker whether the serving image is present.
pub struct DockerImageProbe {
    docker: crate::docker::DockerCli,
}

impl DockerImageProbe {
    #[must_use]
    pub fn new(docker: crate::docker::DockerCli) -> Self {
        Self { docker }
    }
}

impl InstalledImageProbe for DockerImageProbe {
    fn is_installed(&self, image: &str) -> BoxFuture<'_, bool> {
        let image = image.to_owned();
        Box::pin(async move {
            // A failed inspect means absent, not an error: the client asks constantly and a
            // transient daemon hiccup should not surface as a catalog failure.
            matches!(self.docker.image_summary(&image).await, Ok(Some(_)))
        })
    }
}

impl EimCatalog {
    #[must_use]
    pub fn new(
        definitions: Arc<BTreeMap<ModelServingConfigurationId, EimModelDefinition>>,
        packages: Arc<dyn InstalledImageProbe>,
    ) -> Self {
        Self {
            definitions,
            packages,
        }
    }

    /// The capabilities the client uses to decide whether a model is usable.
    ///
    /// `tools` follows the parser: with none configured, vLLM returns tool calls as prose, and
    /// claiming support would let the agent loop fail silently instead of the model being
    /// visibly marked unusable.
    fn capabilities(definition: &EimModelDefinition) -> ModelCapabilities {
        let reasoning_supported = !definition.reasoning.efforts.is_empty();
        ModelCapabilities {
            vision: definition.modalities.vision,
            tools: definition.tool_call_parser.is_some(),
            // vLLM drives structured output through its guided-decoding backend, available for
            // any served model.
            structured_output: true,
            reasoning: ModelReasoningCapabilities {
                supported: reasoning_supported,
                efforts: definition.reasoning.efforts.clone(),
                default_effort: definition.reasoning.default_effort.clone(),
            },
        }
    }

    /// Plain-language notes about a model, shown beside it.
    ///
    /// This is where an unusable model explains itself. Marking one unavailable without saying
    /// why leaves the user to guess, and for a container-backed catalog most of the reasons are
    /// specific and actionable: a missing credential, an engine that cannot parse the model's
    /// tool calls, or a model with no chat template at all.
    fn quality_evidence(definition: &EimModelDefinition) -> Vec<String> {
        let mut evidence = Vec::new();

        if definition.tool_call_parser.is_none() {
            evidence.push(
                "The engine has no tool-call parser for this model family, so the agent cannot \
                 use it. It can still hold a conversation."
                    .to_owned(),
            );
        }
        if definition.hf_token_required {
            evidence.push(
                "Gated repository: set HF_TOKEN to download it, and accept the model's licence \
                 on Hugging Face first."
                    .to_owned(),
            );
        }
        if definition.geometry.active_parameters < definition.geometry.total_parameters {
            evidence.push(format!(
                "Mixture of experts: {:.0}B parameters resident, {:.1}B active per token, so it \
                 runs closer to the speed of the smaller number.",
                definition.geometry.total_parameters as f64 / 1e9,
                definition.geometry.active_parameters as f64 / 1e9,
            ));
        }
        if definition.context_tokens < definition.geometry.max_position_embeddings {
            evidence.push(format!(
                "Served at {} tokens of context rather than its trained {}, because the key-value \
                 cache is sized from this and the engine can refuse to start when it does not fit.",
                definition.context_tokens, definition.geometry.max_position_embeddings,
            ));
        }
        evidence.push(format!(
            "Served by vLLM on Intel Xeon through an EIM container, profile {}.",
            definition.eim_profile_id
        ));
        evidence
    }

    fn parameterization(definition: &EimModelDefinition) -> ModelParameterization {
        let total = definition.geometry.total_parameters;
        let active = definition.geometry.active_parameters;
        if active < total {
            ModelParameterization::MixtureOfExperts {
                total_parameters: total,
                active_parameters: active,
            }
        } else {
            ModelParameterization::Dense {
                total_parameters: total,
            }
        }
    }

    fn configuration(definition: &EimModelDefinition) -> ModelServingConfiguration {
        ModelServingConfiguration {
            id: definition.configuration_id.clone(),
            // A container-backed model is always a single package: there is no draft model,
            // because no shipped EIM profile declares speculative decoding.
            bundle: ServableModelBundle::Standalone {
                package: crate::package::model_package(definition),
            },
            profile: ServingProfile {
                context_length: definition.context_tokens,
            },
        }
    }

    /// The installed-package record for one model, shared by both surfaces that report it.
    ///
    /// The catalog's local state and `/v1/models/installed` describe the same thing, so they are
    /// built from one function. Reporting an installed model with no packages in one of them
    /// would be a lie in the data even where nothing currently reads it.
    fn installed_package(definition: &EimModelDefinition) -> InstalledModelPackage {
        InstalledModelPackage {
            package: crate::package::model_package(definition),
            // The in-container weight cache. Not a host path: ICN never opens it, and reporting a
            // host path would imply the weights are reachable from here.
            path: std::path::PathBuf::from(crate::env_contract::CONTAINER_CACHE_PATH)
                .join(&definition.canonical_name),
            origin: ModelPackageInstallationOrigin::Magnitude,
            inspection: ModelPackageInspection::Inspected {
                capabilities: Self::capabilities(definition),
            },
            catalog_attribution: InstalledCatalogAttribution::Attributed {
                model_id: CatalogModelId(definition.catalog_model_id.clone()),
                variant_id: CatalogVariantId(definition.catalog_variant_id.clone()),
            },
        }
    }

    /// Turns one definition into a catalog entry, or a diagnostic when its data is unusable.
    fn entry(
        definition: &EimModelDefinition,
        installed: bool,
    ) -> Result<CatalogModel, CatalogDiagnostic> {
        let model_id = CatalogModelId(definition.catalog_model_id.clone());
        let variant_id = CatalogVariantId(definition.catalog_variant_id.clone());
        let release_date =
            ModelReleaseDate::new(definition.release_date.clone()).map_err(|message| {
                CatalogDiagnostic {
                    model_id: model_id.clone(),
                    variant_id: variant_id.clone(),
                    failure: ModelFailure {
                        code: "eim_catalog_invalid".to_owned(),
                        message,
                        retryable: false,
                    },
                }
            })?;

        Ok(CatalogModel {
            model_id,
            variant_id,
            desired_configuration: Self::configuration(definition),
            display_name: definition.display_name.clone(),
            variant_label: definition.variant_label.clone(),
            description: definition.description.clone(),
            release_date,
            license: definition.license.clone(),
            capabilities: Self::capabilities(definition),
            parameterization: Self::parameterization(definition),
            quality_score: definition.quality_score,
            quality_score_provenance: definition.quality_score_provenance.clone(),
            // Every shipped EIM profile is bf16, so there is no fidelity axis to rank on and a
            // constant is the honest value. It makes the term drop out of the client's ranking.
            fidelity_rank: 100,
            quantization_aware: false,
            quality_evidence: Self::quality_evidence(definition),
            local_state: if installed {
                // Reported as up to date: the image digest is the version, so a present image is
                // by definition the version the table asked for.
                CatalogModelLocalState::Installed {
                    installation: icn_contracts::models::CatalogModelInstallation {
                        // Runnable describes what a load would use; whether it fits is decided
                        // by the assessment, not here.
                        effective_configuration:
                            icn_contracts::models::CatalogModelEffectiveConfiguration::Runnable {
                                configuration: Self::configuration(definition),
                            },
                        packages: vec![Self::installed_package(definition)],
                    },
                    update_state: icn_contracts::models::CatalogModelUpdateState::Current,
                }
            } else {
                CatalogModelLocalState::NotInstalled
            },
        })
    }
}

impl CatalogModels for EimCatalog {
    fn list(&self) -> BoxFuture<'_, Result<ModelsResponse, InventoryError>> {
        Box::pin(async move {
            let mut catalog_models = Vec::new();
            let mut diagnostics = Vec::new();
            for definition in self.definitions.values() {
                let installed = self.packages.is_installed(&definition.image).await;
                match Self::entry(definition, installed) {
                    Ok(model) => catalog_models.push(model),
                    // A malformed entry is reported rather than dropped, so a typo in the table
                    // is visible in the client instead of silently shortening the catalog.
                    Err(diagnostic) => diagnostics.push(diagnostic),
                }
            }
            Ok(ModelsResponse {
                // The table is static for a process lifetime, so there is nothing to revise.
                revision: 1,
                reconciliation_complete: true,
                catalog_models,
                // Every package this ICN knows about comes from the table, so nothing is
                // uncatalogued by construction.
                uncatalogued_packages: Vec::new(),
                diagnostics,
            })
        })
    }

    fn reconcile(
        &self,
        request: ReconcileCatalogModelRequest,
    ) -> BoxFuture<'_, Result<ReconcileCatalogModelResponse, InventoryError>> {
        Box::pin(async move {
            // Acquisition -- building or pulling the image and prefetching weights -- is not
            // implemented yet. Refusing explicitly is better than reporting an admitted download
            // that will never progress, which the client would wait on indefinitely.
            Err(InventoryError::Unsupported(format!(
                "installing {}/{} is not implemented for container-backed models yet",
                request.model_id.0, request.variant_id.0
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::ModelGeometry;
    use crate::properties::ReasoningDeclaration;
    use icn_contracts::ModelModalities;
    use icn_contracts::models::ModelPackageId;

    fn definition() -> EimModelDefinition {
        EimModelDefinition {
            configuration_id: ModelServingConfigurationId("eim-qwen3-8b-ctx32768".to_owned()),
            package_id: ModelPackageId("eim--Qwen--Qwen3-8B--v1".to_owned()),
            catalog_model_id: "qwen-qwen3-8b".to_owned(),
            catalog_variant_id: "vllm-bf16:tp2".to_owned(),
            canonical_name: "Qwen/Qwen3-8B".to_owned(),
            display_name: "Qwen3 8B".to_owned(),
            variant_label: "bf16".to_owned(),
            description: "Dense language model from Alibaba's Qwen3 series.".to_owned(),
            release_date: "2025-04-29".to_owned(),
            license: "Apache-2.0".to_owned(),
            quality_score: 24.0,
            quality_score_provenance: "curated_public_benchmarks".to_owned(),
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

    fn catalog(definitions: Vec<EimModelDefinition>) -> EimCatalog {
        EimCatalog::new(
            Arc::new(
                definitions
                    .into_iter()
                    .map(|definition| (definition.configuration_id.clone(), definition))
                    .collect(),
            ),
            Arc::new(NoImagesInstalled),
        )
    }

    #[tokio::test]
    async fn lists_a_model_the_client_can_decode() {
        let response = catalog(vec![definition()]).list().await.expect("a catalog");

        assert_eq!(response.catalog_models.len(), 1);
        assert!(response.diagnostics.is_empty());
        assert!(response.reconciliation_complete);

        let model = &response.catalog_models[0];
        assert_eq!(model.model_id.0, "qwen-qwen3-8b");
        // Two colon-separated components, which the client's schema requires.
        assert_eq!(model.variant_id.0.split(':').count(), 2);
        assert_eq!(model.desired_configuration.profile.context_length, 32_768);
    }

    #[tokio::test]
    async fn reports_tool_support_from_the_configured_parser() {
        let with_parser = catalog(vec![definition()]).list().await.expect("a catalog");
        assert!(with_parser.catalog_models[0].capabilities.tools);

        // Without a parser vLLM returns tool calls as prose, so the model must be marked
        // unusable rather than quietly failing inside the agent loop.
        let without = catalog(vec![EimModelDefinition {
            tool_call_parser: None,
            ..definition()
        }])
        .list()
        .await
        .expect("a catalog");
        assert!(!without.catalog_models[0].capabilities.tools);
    }

    #[tokio::test]
    async fn a_model_without_a_parser_says_why_it_cannot_be_used() {
        let response = catalog(vec![EimModelDefinition {
            tool_call_parser: None,
            ..definition()
        }])
        .list()
        .await
        .expect("a catalog");
        let evidence = &response.catalog_models[0].quality_evidence;

        assert!(
            evidence
                .iter()
                .any(|note| note.contains("tool-call parser")),
            "{evidence:?}"
        );
        // And that it is a capability limit, not a malfunction.
        assert!(
            evidence
                .iter()
                .any(|note| note.contains("hold a conversation")),
            "{evidence:?}"
        );
    }

    #[tokio::test]
    async fn a_gated_model_names_the_credential_it_needs() {
        let response = catalog(vec![EimModelDefinition {
            hf_token_required: true,
            ..definition()
        }])
        .list()
        .await
        .expect("a catalog");

        assert!(
            response.catalog_models[0]
                .quality_evidence
                .iter()
                .any(|note| note.contains("HF_TOKEN")),
            "{:?}",
            response.catalog_models[0].quality_evidence
        );
    }

    #[tokio::test]
    async fn explains_a_context_shorter_than_the_model_was_trained_for() {
        // Otherwise a user comparing the served context against the model card sees a
        // discrepancy with no explanation.
        let response = catalog(vec![definition()]).list().await.expect("a catalog");

        assert!(
            response.catalog_models[0]
                .quality_evidence
                .iter()
                .any(|note| note.contains("32768") && note.contains("40960")),
            "{:?}",
            response.catalog_models[0].quality_evidence
        );
    }

    #[tokio::test]
    async fn a_serveable_model_still_says_how_it_is_served() {
        let response = catalog(vec![definition()]).list().await.expect("a catalog");
        let evidence = &response.catalog_models[0].quality_evidence;

        assert!(
            evidence.iter().any(|note| note.contains("EIM container")),
            "{evidence:?}"
        );
        // No complaint about tools or a credential, because neither applies.
        assert!(
            !evidence
                .iter()
                .any(|note| note.contains("tool-call parser"))
        );
        assert!(!evidence.iter().any(|note| note.contains("HF_TOKEN")));
    }

    #[tokio::test]
    async fn distinguishes_dense_from_mixture_of_experts() {
        let dense = catalog(vec![definition()]).list().await.expect("a catalog");
        assert!(matches!(
            dense.catalog_models[0].parameterization,
            ModelParameterization::Dense { .. }
        ));

        let moe = catalog(vec![EimModelDefinition {
            geometry: ModelGeometry {
                total_parameters: 30_500_000_000,
                weight_bytes: None,
                active_parameters: 3_300_000_000,
                ..definition().geometry
            },
            ..definition()
        }])
        .list()
        .await
        .expect("a catalog");
        match moe.catalog_models[0].parameterization {
            ModelParameterization::MixtureOfExperts {
                total_parameters,
                active_parameters,
            } => {
                // The client's schema requires active strictly below total.
                assert!(active_parameters < total_parameters);
            }
            ref other => panic!("expected a mixture of experts, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn carries_the_declared_reasoning_efforts() {
        let response = catalog(vec![definition()]).list().await.expect("a catalog");
        let reasoning = &response.catalog_models[0].capabilities.reasoning;

        assert!(reasoning.supported);
        assert_eq!(
            reasoning.efforts,
            vec!["none".to_owned(), "high".to_owned()]
        );
        // The client's schema requires the default to be one of the offered efforts.
        assert!(
            reasoning
                .efforts
                .contains(reasoning.default_effort.as_ref().expect("a default"))
        );
    }

    #[tokio::test]
    async fn a_non_reasoning_model_declares_no_efforts() {
        let response = catalog(vec![EimModelDefinition {
            reasoning: ReasoningDeclaration::unsupported(),
            ..definition()
        }])
        .list()
        .await
        .expect("a catalog");
        let reasoning = &response.catalog_models[0].capabilities.reasoning;

        assert!(!reasoning.supported);
        assert!(reasoning.efforts.is_empty());
        assert!(reasoning.default_effort.is_none());
    }

    #[tokio::test]
    async fn reports_a_malformed_entry_as_a_diagnostic_rather_than_dropping_it() {
        // A typo in the table must be visible in the client, not silently shorten the catalog.
        let response = catalog(vec![EimModelDefinition {
            release_date: "2025-13-45".to_owned(),
            ..definition()
        }])
        .list()
        .await
        .expect("a catalog");

        assert!(response.catalog_models.is_empty());
        assert_eq!(response.diagnostics.len(), 1);
        assert_eq!(response.diagnostics[0].failure.code, "eim_catalog_invalid");
        assert!(!response.diagnostics[0].failure.retryable);
    }

    #[tokio::test]
    async fn reports_a_present_image_as_installed_and_current() {
        struct AllInstalled;
        impl InstalledImageProbe for AllInstalled {
            fn is_installed(&self, _image: &str) -> BoxFuture<'_, bool> {
                Box::pin(async { true })
            }
        }

        let catalog = EimCatalog::new(
            Arc::new(BTreeMap::from([(
                definition().configuration_id,
                definition(),
            )])),
            Arc::new(AllInstalled),
        );
        let response = catalog.list().await.expect("a catalog");

        match &response.catalog_models[0].local_state {
            CatalogModelLocalState::Installed { update_state, .. } => {
                // The image digest is the version, so a present image is the asked-for version.
                assert!(matches!(
                    update_state,
                    icn_contracts::models::CatalogModelUpdateState::Current
                ));
            }
            other => panic!("expected Installed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn both_surfaces_describe_an_installed_model_identically() {
        struct AllInstalled;
        impl InstalledImageProbe for AllInstalled {
            fn is_installed(&self, _image: &str) -> BoxFuture<'_, bool> {
                Box::pin(async { true })
            }
        }
        let catalog = EimCatalog::new(
            Arc::new(BTreeMap::from([(
                definition().configuration_id,
                definition(),
            )])),
            Arc::new(AllInstalled),
        );

        let from_catalog = match &catalog.list().await.expect("a catalog").catalog_models[0]
            .local_state
        {
            CatalogModelLocalState::Installed { installation, .. } => installation.packages.clone(),
            other => panic!("expected Installed, got {other:?}"),
        };
        let from_installed = catalog
            .list_installed()
            .await
            .expect("installed packages")
            .packages;

        // The client derives its installed state from the second surface but reads the first for
        // update and configuration; disagreement between them would be a latent trap.
        assert_eq!(from_catalog, from_installed);
        assert_eq!(from_catalog.len(), 1);
        assert_eq!(from_catalog[0].package.id, definition().package_id);
    }

    #[tokio::test]
    async fn refuses_installation_explicitly_instead_of_admitting_it() {
        // Reporting an admitted download that never progresses would leave the client waiting.
        let error = catalog(vec![definition()])
            .reconcile(ReconcileCatalogModelRequest {
                model_id: CatalogModelId("qwen-qwen3-8b".to_owned()),
                variant_id: CatalogVariantId("vllm-bf16:tp2".to_owned()),
            })
            .await
            .expect_err("installation is not implemented yet");

        assert!(matches!(error, InventoryError::Unsupported(_)));
    }

    #[tokio::test]
    async fn an_empty_table_is_an_empty_catalog_not_a_failure() {
        // ACN fails to become ready if this endpoint errors, so having no models must still
        // produce a valid response.
        let response = catalog(Vec::new()).list().await.expect("a catalog");

        assert!(response.catalog_models.is_empty());
        assert!(response.diagnostics.is_empty());
    }
}

/// The recommendable catalog: what the client ranks when suggesting a model.
///
/// Every entry the table describes is offered. Whether one is *serveable* is decided by the
/// assessment, not by omission from this list -- the client is what explains an unavailable
/// model, and it cannot explain what it never receives.
impl RecommendableModelCatalogProvider for EimCatalog {
    fn catalog(&self) -> BoxFuture<'_, Result<RecommendableModelCatalog, InventoryError>> {
        Box::pin(async move {
            let mut models = Vec::new();
            let mut diagnostics = Vec::new();
            for definition in self.definitions.values() {
                match Self::entry(definition, false) {
                    Ok(entry) => models.push(RecommendableModel {
                        model_id: entry.model_id,
                        variant_id: entry.variant_id,
                        configuration: entry.desired_configuration,
                        display_name: entry.display_name,
                        variant_label: entry.variant_label,
                        description: entry.description,
                        release_date: entry.release_date,
                        license: entry.license,
                        capabilities: entry.capabilities,
                        parameterization: entry.parameterization,
                        quality_score: entry.quality_score,
                        quality_score_provenance: entry.quality_score_provenance,
                        fidelity_rank: entry.fidelity_rank,
                        quantization_aware: entry.quantization_aware,
                        quality_evidence: entry.quality_evidence,
                    }),
                    Err(diagnostic) => diagnostics.push(diagnostic),
                }
            }
            Ok(RecommendableModelCatalog {
                models,
                diagnostics,
            })
        })
    }
}

/// Installed packages, meaning models whose serving image is present.
impl InstalledModelPackages for EimCatalog {
    fn list_installed(
        &self,
    ) -> BoxFuture<'_, Result<InstalledModelPackagesResponse, InventoryError>> {
        Box::pin(async move {
            let mut packages = Vec::new();
            for definition in self.definitions.values() {
                if !self.packages.is_installed(&definition.image).await {
                    continue;
                }
                packages.push(Self::installed_package(definition));
            }
            Ok(InstalledModelPackagesResponse {
                revision: 1,
                reconciliation_complete: true,
                packages,
            })
        })
    }

    /// Resolving a bundle to on-disk artifacts is meaningless here: the weights are inside a
    /// container, and the load path addresses a serving configuration rather than files.
    fn resolve_bundle(
        &self,
        _bundle: ModelBundleInput,
    ) -> BoxFuture<'_, Result<ResolvedServableModelBundle, InventoryError>> {
        Box::pin(async {
            Err(InventoryError::Unsupported(
                "container-backed models are not resolved to local artifacts".to_owned(),
            ))
        })
    }

    /// Removal is owned by the controller, which also has to refuse while a model is resident.
    fn remove_installed(
        &self,
        package_id: &ModelPackageId,
    ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>> {
        let package_id = package_id.clone();
        Box::pin(async move {
            Err(InventoryError::Unsupported(format!(
                "removing {} is handled through the model controller",
                package_id.0
            )))
        })
    }
}

/// Downloads, which do not exist yet.
///
/// The endpoint still has to answer: ACN polls it while building its layer graph and treats an
/// error as fatal. An empty list is truthful -- nothing is downloading -- while starting one is
/// refused explicitly rather than admitted and left to stall.
pub struct EimDownloads;

impl ModelDownloads for EimDownloads {
    fn start(
        &self,
        _request: StartModelDownloadRequest,
    ) -> BoxFuture<'_, Result<StartModelDownloadResponse, InventoryError>> {
        Box::pin(async {
            Err(InventoryError::Unsupported(
                "weight prefetching is not implemented for container-backed models yet".to_owned(),
            ))
        })
    }

    fn list(&self) -> BoxFuture<'_, Result<ModelDownloadsResponse, InventoryError>> {
        Box::pin(async {
            Ok(ModelDownloadsResponse {
                downloads: Vec::new(),
            })
        })
    }

    fn cancel(&self, id: &ModelDownloadId) -> BoxFuture<'_, Result<ModelDownload, InventoryError>> {
        let id = id.clone();
        Box::pin(async move { Err(InventoryError::NotFound(id.0)) })
    }

    fn acknowledge_failure(
        &self,
        id: &ModelDownloadId,
    ) -> BoxFuture<'_, Result<ModelDownload, InventoryError>> {
        let id = id.clone();
        Box::pin(async move { Err(InventoryError::NotFound(id.0)) })
    }
}
