//! `POST /v1/models/assess`: whether this host can serve a model, and how fast.
//!
//! Not optional, and not merely informational. The client's projections gate on
//! `servingState._tag === "Assessed"`, so a model without an assessment is not offerable — an ICN
//! that cannot answer this shows the user "assessing models for this machine failed" and an empty
//! picker, whatever else works.
//!
//! Two things make the response easy to get wrong in a way that is invisible from here, because
//! both are enforced by the client's schema rather than by this contract:
//!
//! - `ModelAssessment::is_valid_for` demands arithmetically exact `remainingBytes` and a
//!   `capacityBytes` equal to the topology's *total* for that domain, not its stable capacity.
//! - The performance ladder must be non-empty, strictly ascending, and end exactly at the served
//!   context, with `0 < lower <= estimated <= upper`.
//!
//! A violation of either makes the whole assessment undecodable rather than imprecise, which
//! presents as a model missing from the catalog with no reason attached. Both are asserted here.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use icn_contracts::models::{
    AssessModelRequest, AssessModelResult, AssessModelsRequest, AssessModelsResponse,
    AssessmentEnvironmentId, MemoryAssessment, ModelAssessment, ModelAssessmentId, ModelAssessor,
    ModelBundleInput, ModelFailure, ModelPackageId, ModelPackageOperand,
    ModelServingConfigurationId, ServingProfile,
};
use icn_contracts::{HardwareSnapshot, InventoryError, MemoryDomainId};

use crate::controller::EimModelDefinition;
use crate::estimate::{MemoryEstimate, ServingShape};

/// The concurrent sequence count assessments are computed for.
///
/// One, because this answers "can this host serve the model" for a single user, and the load path
/// assesses its own admission separately with the count it was actually asked for. Charging a
/// four-way key-value cache here would report a model as not fitting that serves one person fine.
const ASSESSED_PARALLEL_SEQUENCES: u32 = 1;

/// Everything about the host that an assessment depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentHost {
    /// Total physical system memory, which is what the client's validity check compares against.
    pub total_bytes: u64,
    /// Physical minus Magnitude's reserve policy: what a model may actually occupy.
    pub stable_bytes: u64,
    /// Decides the roofline's bandwidth and which tensor-parallel sizes are usable at all.
    pub numa_nodes: u32,
}

impl AssessmentHost {
    /// Reads the figures out of the snapshot ICN publishes.
    ///
    /// Derived from the snapshot rather than sampled independently, because the client revalidates
    /// an assessment against the topology built from that same snapshot. Two separate readings of
    /// system memory would differ by whatever was allocated in between and make every assessment
    /// undecodable, which presents as an empty model picker.
    #[must_use]
    pub fn from_snapshot(snapshot: &HardwareSnapshot, numa_nodes: u32) -> Self {
        let system = snapshot
            .memory_domains
            .iter()
            .find(|domain| domain.id.is_system());
        Self {
            total_bytes: system.map_or(0, |domain| domain.total_capacity_bytes),
            stable_bytes: system.map_or(0, |domain| domain.stable_capacity_bytes),
            numa_nodes,
        }
    }
}

/// Answers `/v1/models/assess` from catalog geometry.
pub struct EimModelAssessor {
    definitions: Arc<BTreeMap<ModelServingConfigurationId, EimModelDefinition>>,
    host: Arc<dyn Fn() -> AssessmentHost + Send + Sync>,
}

impl EimModelAssessor {
    pub fn new(
        definitions: Arc<BTreeMap<ModelServingConfigurationId, EimModelDefinition>>,
        host: Arc<dyn Fn() -> AssessmentHost + Send + Sync>,
    ) -> Self {
        Self { definitions, host }
    }

    /// Finds the definition a request's bundle refers to.
    ///
    /// By package identity, because that is the one field a request is guaranteed to carry: the
    /// client asks about a model it has only seen in the catalog, so it has a package but no
    /// serving-configuration id of its own choosing.
    fn definition(&self, bundle: &ModelBundleInput) -> Option<&EimModelDefinition> {
        let package_id = match bundle {
            ModelBundleInput::Standalone { package } => package_identity(package),
            ModelBundleInput::SpeculativeDecoding { target, .. } => package_identity(target),
        }?;
        self.definitions
            .values()
            .find(|definition| definition.package_id == package_id)
    }

    fn assess_one(&self, request: &AssessModelRequest, host: AssessmentHost) -> AssessModelResult {
        let Some(definition) = self.definition(&request.bundle) else {
            return AssessModelResult::InvalidBundle {
                request_id: request.request_id.clone(),
                failure: ModelFailure {
                    code: "eim_model_unknown".to_owned(),
                    message: "this bundle is not a model in the EIM catalog".to_owned(),
                    retryable: false,
                },
            };
        };

        // An empty profile list is a request about the model's own served context, which is the
        // only configuration this fork can serve it in anyway.
        let profiles: Vec<(ServingProfile, Vec<u32>)> = if request.profiles.is_empty() {
            vec![(
                ServingProfile {
                    context_length: definition.context_tokens,
                },
                Vec::new(),
            )]
        } else {
            request
                .profiles
                .iter()
                .map(|entry| {
                    (
                        entry.profile.clone(),
                        entry.performance_context_tokens.clone(),
                    )
                })
                .collect()
        };

        AssessModelResult::Assessed {
            request_id: request.request_id.clone(),
            profiles: profiles
                .into_iter()
                .map(|(profile, requested_depths)| {
                    assessment(definition, &profile, &requested_depths, host)
                })
                .collect(),
        }
    }
}

fn package_identity(operand: &ModelPackageOperand) -> Option<ModelPackageId> {
    match operand {
        ModelPackageOperand::Installed { package_id } => Some(package_id.clone()),
        ModelPackageOperand::SourceBacked { package } => Some(package.id.clone()),
    }
}

/// The assessment for one model at one profile.
fn assessment(
    definition: &EimModelDefinition,
    profile: &ServingProfile,
    requested_depths: &[u32],
    host: AssessmentHost,
) -> ModelAssessment {
    // The client may ask about a context the model was not trained for. Serving it would silently
    // change what the estimate covers, so the request is honored as asked and the load path is
    // where a context becomes real.
    let context_tokens = profile.context_length.max(1);
    let estimate = MemoryEstimate::compute(
        &definition.geometry,
        &ServingShape {
            context_tokens,
            parallel_sequences: ASSESSED_PARALLEL_SEQUENCES,
            tensor_parallel_size: definition.tensor_parallel_size,
        },
    );

    let configuration = definition.serving_configuration(context_tokens);
    let assessment_id = ModelAssessmentId(format!(
        "{}-ctx{}-tp{}",
        definition.configuration_id.0, context_tokens, definition.tensor_parallel_size
    ));
    let memory = vec![memory_assessment(&estimate, host)];

    // The engine cannot parse this model's tool calls, so the agent cannot use it. Reported as
    // incompatible rather than as a fit, because "it loads but the agent loop silently fails" is
    // the worse of the two answers to give.
    if definition.tool_call_parser.is_none() {
        return ModelAssessment::Incompatible {
            configuration,
            failure: ModelFailure {
                code: "eim_no_tool_call_parser".to_owned(),
                message: format!(
                    "no vLLM tool-call parser is known for {}, so the agent cannot use it",
                    definition.canonical_name
                ),
                retryable: false,
            },
        };
    }

    let reserve = host.total_bytes.saturating_sub(host.stable_bytes);
    if estimate.required_bytes > host.stable_bytes {
        return ModelAssessment::DoesNotFit {
            configuration,
            assessment_id,
            memory,
            limiting_resource: "system memory".to_owned(),
            deficit_bytes: estimate
                .required_bytes
                .saturating_sub(host.stable_bytes.saturating_sub(reserve).max(0)),
        };
    }

    ModelAssessment::Fits {
        configuration,
        assessment_id,
        memory,
        performance: crate::perf::evidence(
            &definition.geometry,
            host.numa_nodes,
            requested_depths,
            context_tokens,
        ),
    }
}

/// The one system-memory domain, with the arithmetic the client re-checks.
fn memory_assessment(estimate: &MemoryEstimate, host: AssessmentHost) -> MemoryAssessment {
    // Total, not stable. `is_valid_for` compares this against `topology.capacity(domain)`, and the
    // reserve is carried separately in `compatibilityReserveBytes` -- putting it in both places
    // makes the response fail validation and the model vanish.
    let capacity_bytes = host.total_bytes;
    let compatibility_reserve_bytes = host.total_bytes.saturating_sub(host.stable_bytes);
    let remaining_bytes = (i128::from(capacity_bytes)
        - i128::from(compatibility_reserve_bytes)
        - i128::from(estimate.required_bytes))
    .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;

    MemoryAssessment {
        memory_domain_id: MemoryDomainId::system(),
        capacity_bytes,
        required_bytes: estimate.required_bytes,
        compatibility_reserve_bytes,
        remaining_bytes,
    }
}

impl ModelAssessor for EimModelAssessor {
    fn assess(
        &self,
        request: AssessModelsRequest,
    ) -> BoxFuture<'_, Result<AssessModelsResponse, InventoryError>> {
        Box::pin(async move {
            let host = (self.host)();
            Ok(AssessModelsResponse {
                // Keyed to the host facts the assessments were computed against, so a client
                // holding results from a different environment can tell.
                environment_id: AssessmentEnvironmentId(format!(
                    "eim-{}-numa{}-mem{}",
                    env!("CARGO_PKG_VERSION"),
                    host.numa_nodes,
                    host.total_bytes
                )),
                results: request
                    .requests
                    .iter()
                    .map(|entry| self.assess_one(entry, host))
                    .collect(),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icn_contracts::MemoryTopology;
    use icn_contracts::models::{ModelAssessmentProfile, ModelAssessmentRequestId};

    /// The snapshot ICN publishes on whatever machine runs this.
    ///
    /// Read from the real host rather than fabricated, because the point of these tests is that the
    /// assessment's arithmetic agrees with the topology the client revalidates against -- and two
    /// independent readings of system memory would not.
    fn snapshot() -> HardwareSnapshot {
        icn_hardware::discover_hardware(
            icn_hardware::CapacityPolicy::default(),
            "eim-test",
            &icn_hardware::HostTopology {
                cpu_model: Some("test".to_owned()),
                logical_cores: 8,
                numa_nodes: 2,
            },
        )
    }

    fn host() -> AssessmentHost {
        AssessmentHost::from_snapshot(&snapshot(), 2)
    }

    fn definitions(
        models: Vec<EimModelDefinition>,
    ) -> Arc<BTreeMap<ModelServingConfigurationId, EimModelDefinition>> {
        Arc::new(
            models
                .into_iter()
                .map(|definition| (definition.configuration_id.clone(), definition))
                .collect(),
        )
    }

    fn assessor(models: Vec<EimModelDefinition>) -> EimModelAssessor {
        EimModelAssessor::new(definitions(models), Arc::new(host))
    }

    fn request(definition: &EimModelDefinition, depths: Vec<u32>) -> AssessModelsRequest {
        AssessModelsRequest {
            requests: vec![AssessModelRequest {
                request_id: ModelAssessmentRequestId("r-1".to_owned()),
                bundle: ModelBundleInput::Standalone {
                    package: ModelPackageOperand::SourceBacked {
                        package: crate::package::model_package(definition),
                    },
                },
                profiles: vec![ModelAssessmentProfile {
                    profile: ServingProfile {
                        context_length: definition.context_tokens,
                    },
                    performance_context_tokens: depths,
                }],
            }],
        }
    }

    /// The topology ICN publishes, which is what the client validates an assessment against.
    fn topology() -> MemoryTopology {
        MemoryTopology::from_snapshot(&snapshot()).expect("a valid topology")
    }

    #[tokio::test]
    async fn a_serveable_model_fits_and_carries_a_performance_ladder() {
        let definition = crate::controller::tests_support::definition();
        let response = assessor(vec![definition.clone()])
            .assess(request(&definition, vec![4_096, 16_384]))
            .await
            .expect("an assessment");

        assert_eq!(response.results.len(), 1);
        let AssessModelResult::Assessed { profiles, .. } = &response.results[0] else {
            panic!("expected Assessed, got {:?}", response.results[0]);
        };
        let ModelAssessment::Fits { performance, .. } = &profiles[0] else {
            panic!("an 8B model fits in 242 GB: {:?}", profiles[0]);
        };
        assert_eq!(
            performance.last().expect("a sample").context_tokens,
            definition.context_tokens,
            "the ladder must end at the served context or the client rejects it"
        );
    }

    #[tokio::test]
    async fn the_memory_arithmetic_is_what_the_client_revalidates() {
        // `is_valid_for` recomputes `remainingBytes` and compares `capacityBytes` against the
        // topology's total. Getting either wrong makes the model vanish with no reason shown.
        let definition = crate::controller::tests_support::definition();
        let response = assessor(vec![definition.clone()])
            .assess(request(&definition, Vec::new()))
            .await
            .expect("an assessment");

        let AssessModelResult::Assessed { profiles, .. } = &response.results[0] else {
            panic!("expected Assessed");
        };
        assert!(
            profiles[0].is_valid_for(&topology()),
            "{:?} must validate against {:?}",
            profiles[0],
            topology()
        );
    }

    #[tokio::test]
    async fn a_model_with_no_tool_call_parser_is_incompatible_rather_than_fitting() {
        // It would load. Reporting it as a fit is what lets the agent loop fail silently.
        let mut definition = crate::controller::tests_support::definition();
        definition.tool_call_parser = None;

        let response = assessor(vec![definition.clone()])
            .assess(request(&definition, Vec::new()))
            .await
            .expect("an assessment");

        let AssessModelResult::Assessed { profiles, .. } = &response.results[0] else {
            panic!("expected Assessed");
        };
        match &profiles[0] {
            ModelAssessment::Incompatible { failure, .. } => {
                assert_eq!(failure.code, "eim_no_tool_call_parser");
                assert!(!failure.retryable);
            }
            other => panic!("expected Incompatible, got {other:?}"),
        }
        assert!(profiles[0].is_valid_for(&topology()));
    }

    #[tokio::test]
    async fn a_model_larger_than_the_host_does_not_fit() {
        // Llama-4-Scout: 109B parameters is 218 GB of bf16 weights, past a 242 GB budget once the
        // cache and runtime are charged. The one entry in the catalog this must reject.
        let mut definition = crate::controller::tests_support::definition();
        definition.geometry.total_parameters = 109_000_000_000;
        definition.geometry.active_parameters = 17_000_000_000;
        definition.geometry.weight_bytes = None;

        let response = assessor(vec![definition.clone()])
            .assess(request(&definition, Vec::new()))
            .await
            .expect("an assessment");

        let AssessModelResult::Assessed { profiles, .. } = &response.results[0] else {
            panic!("expected Assessed");
        };
        match &profiles[0] {
            ModelAssessment::DoesNotFit {
                limiting_resource, ..
            } => assert_eq!(limiting_resource, "system memory"),
            other => panic!("expected DoesNotFit, got {other:?}"),
        }
        assert!(
            profiles[0].is_valid_for(&topology()),
            "a rejection still has to be decodable"
        );
    }

    #[tokio::test]
    async fn an_unknown_bundle_is_reported_rather_than_failing_the_batch() {
        // The client assesses many models in one call. One unrecognised entry must not cost the
        // others their assessments, which is what "assessment could not be completed" looks like.
        let definition = crate::controller::tests_support::definition();
        let mut stranger = definition.clone();
        stranger.package_id = ModelPackageId("eim--nobody--nothing--v1".to_owned());

        let mut batch = request(&definition, Vec::new());
        batch.requests.push(AssessModelRequest {
            request_id: ModelAssessmentRequestId("r-2".to_owned()),
            bundle: ModelBundleInput::Standalone {
                package: ModelPackageOperand::SourceBacked {
                    package: crate::package::model_package(&stranger),
                },
            },
            profiles: Vec::new(),
        });

        let response = assessor(vec![definition])
            .assess(batch)
            .await
            .expect("an assessment");

        assert!(matches!(
            response.results[0],
            AssessModelResult::Assessed { .. }
        ));
        assert!(matches!(
            response.results[1],
            AssessModelResult::InvalidBundle { .. }
        ));
    }

    #[tokio::test]
    async fn a_request_with_no_profiles_is_assessed_at_the_served_context() {
        let definition = crate::controller::tests_support::definition();
        let mut batch = request(&definition, Vec::new());
        batch.requests[0].profiles = Vec::new();

        let response = assessor(vec![definition.clone()])
            .assess(batch)
            .await
            .expect("an assessment");

        let AssessModelResult::Assessed { profiles, .. } = &response.results[0] else {
            panic!("expected Assessed");
        };
        assert_eq!(profiles.len(), 1);
        let ModelAssessment::Fits { configuration, .. } = &profiles[0] else {
            panic!("expected Fits");
        };
        assert_eq!(
            configuration.profile.context_length,
            definition.context_tokens
        );
    }

    #[tokio::test]
    async fn every_shipped_model_produces_a_decodable_assessment() {
        // The whole catalog in one call, which is what the client actually does on startup. One
        // undecodable entry is an empty picker.
        let models = EimModelDefinition::shipped_table().expect("the shipped table");
        let assessor = assessor(models.clone());
        let batch = AssessModelsRequest {
            requests: models
                .iter()
                .enumerate()
                .map(|(index, definition)| AssessModelRequest {
                    request_id: ModelAssessmentRequestId(format!("r-{index}")),
                    bundle: ModelBundleInput::Standalone {
                        package: ModelPackageOperand::SourceBacked {
                            package: crate::package::model_package(definition),
                        },
                    },
                    profiles: Vec::new(),
                })
                .collect(),
        };

        let response = assessor.assess(batch).await.expect("an assessment");

        assert_eq!(response.results.len(), models.len());
        for result in &response.results {
            let AssessModelResult::Assessed { profiles, .. } = result else {
                panic!("every catalog model must be assessed, got {result:?}");
            };
            for profile in profiles {
                assert!(
                    profile.is_valid_for(&topology()),
                    "undecodable assessment: {profile:?}"
                );
            }
        }
    }
}
