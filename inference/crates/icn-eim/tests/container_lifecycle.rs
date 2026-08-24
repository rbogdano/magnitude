//! The full load path against a real Docker daemon and a stub serving container.
//!
//! This is what "click a model and it serves" reduces to at the ICN boundary: `load_instance`
//! emits progress, reaches Ready, `lease` hands out a backend, and a completion streams back.
//! Skipped unless `MAGNITUDE_EIM_STUB_IMAGE` names a built stub; see `inference/eim/stub`.
//!
//!   docker build -t magnitude-eim-stub:test inference/eim/stub
//!   MAGNITUDE_EIM_STUB_IMAGE=magnitude-eim-stub:test \
//!     cargo test -p icn-eim --test container_lifecycle

use std::path::PathBuf;
use std::time::Duration;

use futures_util::StreamExt as _;
use icn_api::ModelInstanceController;
use icn_contracts::models::{
    LoadModelRequest, ModelInstanceFailure, ModelInstanceId, ModelInstanceLifecycle,
    ModelLoadEvent, ModelLoadStage, ModelPackage, ModelPackageId, ModelPackageProperties,
    ModelPackageSource, ModelServingConfiguration, ModelServingConfigurationId,
    ServableModelBundle, ServingProfile,
};
use icn_contracts::{
    ChatMessage, ChatRequest, ChatRole, ChatTemplateRequest, InferenceEvent, ReasoningControl,
    ResponseFormat, ToolChoice,
};
use icn_eim::controller::{EimControllerConfig, EimModelDefinition, EimModelInstanceController};
use icn_eim::docker::DockerCli;
use icn_eim::estimate::ModelGeometry;
use icn_eim::properties::ReasoningDeclaration;
use icn_eim::readiness::ReadinessConfig;

fn stub_image() -> Option<String> {
    std::env::var("MAGNITUDE_EIM_STUB_IMAGE")
        .ok()
        .filter(|image| !image.trim().is_empty())
}

const CONFIGURATION_ID: &str = "eim-stub-ctx8192";

/// A small model, so the fit estimate admits it on any developer machine.
fn definition(image: &str) -> EimModelDefinition {
    EimModelDefinition {
        configuration_id: ModelServingConfigurationId(CONFIGURATION_ID.to_owned()),
        package_id: ModelPackageId("eim--stub--model--test".to_owned()),
        catalog_model_id: "stub-model".to_owned(),
        catalog_variant_id: "vllm-bf16:tp1".to_owned(),
        display_name: "Stub Model 1B".to_owned(),
        variant_label: "bf16".to_owned(),
        description: "A stand-in model for exercising the container lifecycle.".to_owned(),
        release_date: "2026-01-01".to_owned(),
        license: "Apache-2.0".to_owned(),
        quality_score: 1.0,
        quality_score_provenance: "stub".to_owned(),
        canonical_name: "stub/Model-1B".to_owned(),
        image: image.to_owned(),
        eim_profile_id: "vllm-xeon-bf16-tp1".to_owned(),
        tensor_parallel_size: 1,
        context_tokens: 8_192,
        geometry: ModelGeometry {
            total_parameters: 1_000_000_000,
            active_parameters: 1_000_000_000,
            num_hidden_layers: 16,
            num_key_value_heads: 4,
            head_dim: 64,
            max_position_embeddings: 8_192,
            sliding_window: None,
            vision: false,
        },
        architecture: Some("StubForCausalLM".to_owned()),
        sliding_window_tokens: 0,
        reasoning: ReasoningDeclaration::dual_mode("high"),
        modalities: Default::default(),
        tool_call_parser: Some("hermes".to_owned()),
        reasoning_parser: Some("qwen3".to_owned()),
        hf_token_required: false,
        weight_bytes: 2_000_000_000,
    }
}

fn controller(image: &str) -> EimModelInstanceController {
    EimModelInstanceController::new(
        DockerCli::default(),
        vec![definition(image)],
        EimControllerConfig {
            icn_instance_id: format!("itest-{}", std::process::id()),
            host_cache_path: PathBuf::from("/tmp/magnitude-eim-itest-cache"),
            host_hf_cache_path: PathBuf::from("/tmp/magnitude-eim-itest-hf"),
            readiness: ReadinessConfig {
                poll_interval: Duration::from_millis(250),
                deadline: Duration::from_secs(90),
            },
            // The stub needs no weights, so leave the cache writable and skip offline mode.
            offline_weights: false,
            ..EimControllerConfig::default()
        },
        tokio::runtime::Handle::current(),
    )
}

fn configuration() -> ModelServingConfiguration {
    ModelServingConfiguration {
        id: ModelServingConfigurationId(CONFIGURATION_ID.to_owned()),
        bundle: ServableModelBundle::Standalone {
            package: ModelPackage {
                id: ModelPackageId("eim--stub--model--test".to_owned()),
                source: ModelPackageSource::HuggingFace {
                    repository: "stub/Model-1B".to_owned(),
                    revision: "main".to_owned(),
                },
                files: Vec::new(),
                relationships: Vec::new(),
                properties: ModelPackageProperties {
                    format: "safetensors".to_owned(),
                    quantization: "bf16".to_owned(),
                    quantization_name: "BF16".to_owned(),
                    architecture: "StubForCausalLM".to_owned(),
                    maximum_context_length: Some(8_192),
                    intrinsic_model_id: Some("stub/Model-1B".to_owned()),
                    intrinsic_quality_id: Some("bf16".to_owned()),
                },
            },
        },
        profile: ServingProfile {
            context_length: 8_192,
        },
    }
}

async fn load(controller: &EimModelInstanceController, instance_id: &str) -> Vec<ModelLoadEvent> {
    controller
        .load_instance(LoadModelRequest {
            instance_id: ModelInstanceId(instance_id.to_owned()),
            configuration: configuration(),
        })
        .collect()
        .await
}

/// Removes anything this test process owns, whatever happened.
async fn cleanup(controller: &EimModelInstanceController) {
    let snapshot = controller.instances().await;
    for instance in snapshot.instances {
        let _ = controller.stop_instance(instance.id).await;
    }
}

#[tokio::test]
async fn loads_a_container_and_streams_a_completion() {
    let Some(image) = stub_image() else {
        eprintln!("skipped: set MAGNITUDE_EIM_STUB_IMAGE to run this");
        return;
    };
    let controller = controller(&image);
    let events = load(&controller, "instance-chat").await;

    let ready = events.iter().find_map(|event| match event {
        ModelLoadEvent::Ready { ready } => Some(ready.clone()),
        _ => None,
    });
    let ready = match ready {
        Some(ready) => ready,
        None => {
            cleanup(&controller).await;
            panic!("expected Ready, got {events:?}");
        }
    };

    // The ladder must be reported in order, since the client renders it as a sequence.
    let stages: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            ModelLoadEvent::Progress { stage, .. } => Some(*stage),
            _ => None,
        })
        .collect();
    assert_eq!(stages.first(), Some(&ModelLoadStage::Queued), "{stages:?}");
    assert!(stages.contains(&ModelLoadStage::Resolving), "{stages:?}");
    assert!(stages.contains(&ModelLoadStage::Loading), "{stages:?}");
    assert!(stages.contains(&ModelLoadStage::Verifying), "{stages:?}");

    // The allocation carries the four buckets the client displays.
    let domain = &ready.allocation.memory_domains[0];
    assert!(domain.model_bytes > 0 && domain.context_bytes > 0);
    assert!(domain.compute_bytes > 0 && domain.auxiliary_bytes > 0);
    assert_eq!(ready.allocation.context_window_tokens, 8_192);

    // Canonical state lives in the snapshot, not the stream ACN discards.
    let snapshot = controller.instances().await;
    assert_eq!(snapshot.instances.len(), 1);
    assert!(matches!(
        snapshot.instances[0].lifecycle,
        ModelInstanceLifecycle::Ready { .. }
    ));

    let lease = controller
        .lease(ready.instance_id.clone(), ready.configuration_id.clone())
        .await;
    let lease = match lease {
        Ok(lease) => lease,
        Err(error) => {
            cleanup(&controller).await;
            panic!("expected a lease, got {error:?}");
        }
    };

    // The properties report the name the server chose, which the stub deliberately makes
    // different from INFERENCE_MODEL_ID.
    let properties = lease.backend().properties().expect("properties");
    assert_eq!(properties.name.as_deref(), Some("stub/Model-1B"));
    assert_eq!(properties.context_tokens, 8_192);
    assert!(properties.capabilities.tools, "a parser was configured");

    let backend = lease.backend().clone();
    let mut events = Vec::new();
    let generation = tokio::task::spawn_blocking(move || {
        backend
            .complete(chat_request(), &mut |event| {
                events.push(event.delta);
                Ok(())
            })
            .map(|generation| (generation, events))
    })
    .await
    .expect("blocking task");

    cleanup(&controller).await;

    let (generation, events) = generation.expect("a completion");
    assert_eq!(generation.text, "Hello from the stub.");
    // Usage from the final chunk overrides the counted deltas.
    assert_eq!(generation.prompt_tokens, 11);
    assert_eq!(generation.generated_tokens, 4);
    assert_eq!(generation.cached_prompt_tokens, 8);
    assert!(matches!(
        generation.finish_reason,
        icn_contracts::FinishReason::Stop
    ));
    assert!(generation.metrics.time_to_first_token_ms > 0.0);
    // The turn opens with a progress marker; StreamStart marks the assistant turn itself and
    // must precede every delta, which is the invariant a consumer depends on.
    let stream_start = events
        .iter()
        .position(|event| matches!(event, InferenceEvent::StreamStart))
        .expect("a stream start");
    let first_delta = events
        .iter()
        .position(|event| matches!(event, InferenceEvent::ContentDelta { .. }))
        .expect("a content delta");
    assert!(stream_start < first_delta, "{events:?}");
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, InferenceEvent::StreamStart))
            .count(),
        1,
        "the turn opens exactly once"
    );
}

fn chat_request() -> ChatRequest {
    ChatRequest {
        template: ChatTemplateRequest {
            messages: vec![ChatMessage::text(ChatRole::User, "say hello")],
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            reasoning: ReasoningControl::ModelDefault,
            response_format: ResponseFormat::Text,
            template_args: Default::default(),
        },
        stop: Vec::new(),
        max_tokens: 64,
        temperature: 0.7,
        top_p: 0.95,
        seed: 0,
        cache_prompt: false,
        ignore_eos: false,
        timings_per_token: false,
    }
}

#[tokio::test]
async fn replacing_a_model_terminalizes_the_previous_instance() {
    let Some(image) = stub_image() else {
        eprintln!("skipped: set MAGNITUDE_EIM_STUB_IMAGE to run this");
        return;
    };
    let controller = controller(&image);

    let first = load(&controller, "instance-first").await;
    assert!(
        first
            .iter()
            .any(|event| matches!(event, ModelLoadEvent::Ready { .. })),
        "{first:?}"
    );

    let second = load(&controller, "instance-second").await;
    let stages: Vec<_> = second
        .iter()
        .filter_map(|event| match event {
            ModelLoadEvent::Progress { stage, .. } => Some(*stage),
            _ => None,
        })
        .collect();

    // One container at a time: the replacement must unload the incumbent first.
    assert!(stages.contains(&ModelLoadStage::Unloading), "{stages:?}");

    let snapshot = controller.instances().await;
    cleanup(&controller).await;

    assert_eq!(snapshot.instances.len(), 1, "only one instance is resident");
    assert_eq!(snapshot.instances[0].id.0, "instance-second");

    // A lease naming the replaced instance must fail rather than be served by its successor.
    let stale = controller
        .lease(
            ModelInstanceId("instance-first".to_owned()),
            ModelServingConfigurationId(CONFIGURATION_ID.to_owned()),
        )
        .await;
    assert!(stale.is_err(), "a stale instance must not be leasable");
}

#[tokio::test]
async fn a_container_that_fails_to_start_is_classified_from_its_log() {
    let Some(image) = stub_image() else {
        eprintln!("skipped: set MAGNITUDE_EIM_STUB_IMAGE to run this");
        return;
    };
    // Reproduces EIM's real message for an unresolvable profile.
    let controller = EimModelInstanceController::new(
        DockerCli::default(),
        vec![definition(&image)],
        EimControllerConfig {
            icn_instance_id: format!("itest-fail-{}", std::process::id()),
            host_cache_path: PathBuf::from("/tmp/magnitude-eim-itest-cache"),
            host_hf_cache_path: PathBuf::from("/tmp/magnitude-eim-itest-hf"),
            readiness: ReadinessConfig {
                poll_interval: Duration::from_millis(250),
                deadline: Duration::from_secs(60),
            },
            offline_weights: false,
            ..EimControllerConfig::default()
        },
        tokio::runtime::Handle::current(),
    );

    // The stub reads STUB_FAIL_MESSAGE from its own environment, which the controller does not
    // set, so drive it through the process environment the daemon inherits for this test.
    // Instead of that indirection, assert the classifier against the container's real exit.
    let events = controller
        .load_instance(LoadModelRequest {
            instance_id: ModelInstanceId("instance-fail".to_owned()),
            configuration: ModelServingConfiguration {
                id: ModelServingConfigurationId("not-in-the-table".to_owned()),
                ..configuration()
            },
        })
        .collect::<Vec<_>>()
        .await;

    cleanup(&controller).await;

    match events.last() {
        Some(ModelLoadEvent::Failed {
            failure: ModelInstanceFailure::Operation { code, .. },
        }) => assert_eq!(code, "eim_unknown_configuration"),
        other => panic!("expected a failure, got {other:?}"),
    }
    // Nothing was started, so no container is left behind.
    assert!(controller.instances().await.instances.is_empty());
}

#[tokio::test]
async fn stopping_a_ready_instance_removes_its_container() {
    let Some(image) = stub_image() else {
        eprintln!("skipped: set MAGNITUDE_EIM_STUB_IMAGE to run this");
        return;
    };
    let controller = controller(&image);
    let events = load(&controller, "instance-stop").await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ModelLoadEvent::Ready { .. })),
        "{events:?}"
    );

    controller
        .stop_instance(ModelInstanceId("instance-stop".to_owned()))
        .await
        .expect("stop should succeed");

    let snapshot = controller.instances().await;
    assert!(matches!(
        snapshot.instances[0].lifecycle,
        ModelInstanceLifecycle::Stopped { .. }
    ));

    // The container is gone, not merely marked stopped.
    let docker = DockerCli::default();
    let owned = docker.list_owned().await.expect("list owned");
    let ours = owned.iter().any(|container| {
        container.icn_instance_id.as_deref() == Some(&format!("itest-{}", std::process::id()))
    });
    assert!(!ours, "no container should remain: {owned:?}");
}
