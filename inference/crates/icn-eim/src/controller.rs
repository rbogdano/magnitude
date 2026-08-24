//! `ModelInstanceController` over Docker.
//!
//! This is the seam the client's "click a model" path lands on: the CLI chooser calls ACN,
//! which calls `loadModelInstance`, which reaches `load_instance` here. Previously that started
//! a llama.cpp worker process; now it starts an EIM container. Nothing above ICN changes.
//!
//! One container is resident at a time. That was already the invariant for the native
//! controller, and it stays: switching models terminalizes the previous instance before the
//! replacement becomes Ready. For containers it is also the safe default, since two resident
//! models would double the memory budget the fit estimate was computed against.
//!
//! Canonical lifecycle lives in `instances()`, not in the load stream. ACN drains the stream
//! for progress and discards it, reading state from the snapshot and the invalidation watch, so
//! a dropped stream must never lose a transition.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use futures_util::StreamExt as _;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use icn_api::{ModelInstanceController, ModelInstanceLease};
use icn_contracts::models::{
    LoadModelReady, LoadModelRequest, ModelInstance, ModelInstanceAllocation, ModelInstanceFailure,
    ModelInstanceId, ModelInstanceLifecycle, ModelInstanceMemoryDomain, ModelInstancesInvalidation,
    ModelInstancesSnapshot, ModelLoadEvent, ModelLoadPlan, ModelLoadStage, ModelPackageId,
    ModelReleaseReason, ModelServingConfigurationId, ModelStoppingAllocation,
    PreviewModelLoadRequest, RemoveInstalledModelPackageResponse,
};
use icn_contracts::{CompletionBackend, InventoryError, MemoryDomainId, ModelModalities};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::{ReceiverStream, WatchStream};

use crate::docker::DockerCli;
use crate::docker::cli::{ContainerSpec, ProxySettings};
use crate::docker::naming::{ContainerLabels, container_name};
use crate::env_contract::{CONTAINER_PORT, LaunchEnvironment};
use crate::estimate::{HostBudget, MemoryEstimate, ModelGeometry, ServingShape};
use crate::memory::SystemMemoryObserver;
use crate::properties::{ModelPropertiesSpec, ReasoningDeclaration};
use crate::readiness::{
    DockerContainerWatch, HttpServerProbe, ReadinessConfig, ReadinessOutcome, await_ready,
};

/// Everything the controller needs to serve one catalog model.
///
/// Stage 2 builds these from the EIM catalog plus the geometry overlay; stage 1 accepts them
/// directly so the container path can be exercised before the catalog exists.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EimModelDefinition {
    pub configuration_id: ModelServingConfigurationId,
    pub package_id: ModelPackageId,
    pub catalog_model_id: String,
    /// Hugging Face repository identifier, also `INFERENCE_MODEL_ID`.
    pub canonical_name: String,
    /// Fully qualified serving image reference.
    pub image: String,
    pub eim_profile_id: String,
    pub tensor_parallel_size: u32,
    pub context_tokens: u32,
    pub geometry: ModelGeometry,
    pub architecture: Option<String>,
    pub sliding_window_tokens: i32,
    pub reasoning: ReasoningDeclaration,
    pub modalities: ModelModalities,
    /// `None` marks a model whose tool calls vLLM cannot parse. The catalog must mark such a
    /// model unusable rather than letting the agent loop fail silently.
    #[serde(default)]
    pub tool_call_parser: Option<String>,
    #[serde(default)]
    pub reasoning_parser: Option<String>,
    #[serde(default)]
    pub hf_token_required: bool,
    pub weight_bytes: u64,
}

impl EimModelDefinition {
    /// Reads a stage-1 model table from JSON.
    ///
    /// Stage 2 replaces this with the generated EIM catalog plus its geometry overlay. Until
    /// that overlay is authored from each model's `config.json`, supplying the table as data
    /// is the honest option: inventing layer counts and key-value head counts would make the
    /// RAM estimate confidently wrong.
    pub fn load_table(path: &std::path::Path) -> Result<Vec<Self>, String> {
        let source = std::fs::read_to_string(path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let definitions: Vec<Self> = serde_json::from_str(&source)
            .map_err(|error| format!("could not parse {}: {error}", path.display()))?;

        let mut seen = std::collections::BTreeSet::new();
        for definition in &definitions {
            if !seen.insert(definition.configuration_id.clone()) {
                return Err(format!(
                    "duplicate serving configuration {}",
                    definition.configuration_id.0
                ));
            }
            if definition.tool_call_parser.is_some() && definition.context_tokens == 0 {
                return Err(format!(
                    "{} declares tool calling but no context length",
                    definition.canonical_name
                ));
            }
        }
        Ok(definitions)
    }
}

/// Host-wide launch settings.
#[derive(Clone, Debug)]
pub struct EimControllerConfig {
    /// The `--instance-id` this ICN was started with, written onto every container.
    pub icn_instance_id: String,
    /// Host directory mounted as the in-container weight cache, in EIM's Local Directory layout.
    pub host_cache_path: PathBuf,
    /// Host directory mounted as the engine's Hugging Face cache. Required: without it a model
    /// not already in the Local Directory layout downloads into the container's ephemeral
    /// filesystem and is re-downloaded on every start.
    pub host_hf_cache_path: PathBuf,
    pub proxy: ProxySettings,
    /// Only needed for gated repositories.
    pub hf_token: Option<String>,
    pub readiness: ReadinessConfig,
    /// Weights are prefetched by Magnitude, so the cache mounts read-only and the container
    /// runs offline. Stage 1 sets this false and lets the container download.
    pub offline_weights: bool,
    /// The container memory ceiling as a fraction of the estimate, in percent. Above 100 so a
    /// slightly low estimate does not OOM a healthy container, but bounded so a runaway one is
    /// killed instead of taking the host down.
    pub memory_limit_percent: u64,
    /// Shared memory for the container. Docker's 64 MB default breaks tensor parallelism:
    /// vLLM's workers broadcast through shared memory, and the failure surfaces only as a gloo
    /// "connection closed by peer" from the worker that survived.
    pub shm_size_bytes: u64,
    /// How many NUMA nodes the host exposes, used to size the per-node memory reservation vLLM
    /// asks for. Zero means unknown, which makes the reservation fall back to its ceiling.
    pub numa_nodes: u32,
}

impl Default for EimControllerConfig {
    fn default() -> Self {
        Self {
            icn_instance_id: "standalone".to_owned(),
            host_cache_path: PathBuf::from("/var/lib/magnitude/eim/model-cache"),
            host_hf_cache_path: PathBuf::from("/var/lib/magnitude/eim/hf-cache"),
            proxy: ProxySettings::from_environment(),
            hf_token: std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
            readiness: ReadinessConfig::default(),
            offline_weights: false,
            memory_limit_percent: 115,
            shm_size_bytes: 16 * 1024 * 1024 * 1024,
            numa_nodes: 1,
        }
    }
}

/// The one resident container, if any.
struct Resident {
    instance_id: ModelInstanceId,
    configuration_id: ModelServingConfigurationId,
    container_name: String,
    lifecycle: ModelInstanceLifecycle,
    backend: Option<Arc<dyn CompletionBackend>>,
    last_activity: Instant,
}

#[derive(Clone, Copy, Debug)]
struct ResidencyPolicy {
    generation: u64,
    idle_timeout: Duration,
}

struct ControllerState {
    revision: u64,
    resident: Option<Resident>,
    residency: ResidencyPolicy,
}

/// Shared, cloneable controller internals, so a load task can outlive the borrow that spawned it.
#[derive(Clone)]
struct Shared {
    docker: DockerCli,
    definitions: Arc<BTreeMap<ModelServingConfigurationId, EimModelDefinition>>,
    config: Arc<EimControllerConfig>,
    runtime: Handle,
    state: Arc<Mutex<ControllerState>>,
    revision_tx: Arc<watch::Sender<u64>>,
    memory: Arc<SystemMemoryObserver>,
}

pub struct EimModelInstanceController {
    shared: Shared,
    aliases: Arc<RwLock<BTreeSet<String>>>,
    /// Serializes load and stop so two clients cannot race a replacement.
    mutation: Arc<tokio::sync::Mutex<()>>,
}

/// How often idleness is checked. Coarse on purpose: the timeout is measured in minutes, and a
/// tighter loop would only add wakeups.
const IDLE_CHECK_INTERVAL: Duration = Duration::from_secs(30);

/// Whether a resident instance has been idle long enough to release.
///
/// Only a Ready instance is eligible. Reaping one that is still loading would race the load it
/// is part of, and a stopped or failed one has nothing left to release.
fn is_idle_past(
    lifecycle: &ModelInstanceLifecycle,
    idle_for: Duration,
    idle_timeout: Duration,
) -> bool {
    matches!(lifecycle, ModelInstanceLifecycle::Ready { .. })
        && !idle_timeout.is_zero()
        && idle_for >= idle_timeout
}

impl EimModelInstanceController {
    pub fn new(
        docker: DockerCli,
        definitions: Vec<EimModelDefinition>,
        config: EimControllerConfig,
        runtime: Handle,
    ) -> Self {
        let (revision_tx, _) = watch::channel(0);
        let controller = Self {
            shared: Shared {
                docker,
                definitions: Arc::new(
                    definitions
                        .into_iter()
                        .map(|definition| (definition.configuration_id.clone(), definition))
                        .collect(),
                ),
                config: Arc::new(config),
                runtime,
                state: Arc::new(Mutex::new(ControllerState {
                    revision: 0,
                    resident: None,
                    residency: ResidencyPolicy {
                        generation: 0,
                        idle_timeout: Duration::from_secs(60 * 60),
                    },
                })),
                revision_tx: Arc::new(revision_tx),
                memory: Arc::new(SystemMemoryObserver::new()),
            },
            aliases: Arc::new(RwLock::new(BTreeSet::new())),
            mutation: Arc::new(tokio::sync::Mutex::new(())),
        };
        controller.start_idle_reaper();
        controller
    }

    /// Releases the resident container once it has been idle past the client's timeout.
    ///
    /// The client sets that timeout — an hour while connected, ten minutes after it
    /// disconnects — and expects ICN to enforce it. Without this a forgotten session would
    /// hold tens of gigabytes indefinitely.
    fn start_idle_reaper(&self) {
        let shared = self.shared.clone();
        let mutation = Arc::clone(&self.mutation);
        self.shared.runtime.spawn(async move {
            loop {
                tokio::time::sleep(IDLE_CHECK_INTERVAL).await;
                let expired = {
                    let state = shared.state.lock().expect("controller state lock");
                    state.resident.as_ref().is_some_and(|resident| {
                        is_idle_past(
                            &resident.lifecycle,
                            resident.last_activity.elapsed(),
                            state.residency.idle_timeout,
                        )
                    })
                };
                if !expired {
                    continue;
                }
                // Take the same lock a load or stop would, so reaping cannot interleave with a
                // replacement half-way through.
                let _guard = mutation.lock().await;
                let still_expired = {
                    let state = shared.state.lock().expect("controller state lock");
                    state.resident.as_ref().is_some_and(|resident| {
                        is_idle_past(
                            &resident.lifecycle,
                            resident.last_activity.elapsed(),
                            state.residency.idle_timeout,
                        )
                    })
                };
                if still_expired {
                    tracing::info!("releasing an idle model instance");
                    shared
                        .release_resident(ModelReleaseReason::IdleTimeout)
                        .await;
                }
            }
        });
    }

    /// Removes containers left behind by a previous, no-longer-running ICN.
    ///
    /// Runs at boot. Without it a `SIGKILL`ed ICN — which is exactly how the client's shutdown
    /// ends if the graceful window elapses — leaves a container holding tens of gigabytes with
    /// nothing tracking it.
    pub async fn reap_orphans(&self) -> Vec<String> {
        let owned = match self.shared.docker.list_owned().await {
            Ok(owned) => owned,
            Err(error) => {
                tracing::warn!(%error, "could not list ICN-owned containers");
                return Vec::new();
            }
        };
        let mut reaped = Vec::new();
        for container in owned {
            if !container.is_orphan_of(&self.shared.config.icn_instance_id, pid_is_alive) {
                continue;
            }
            tracing::info!(
                container = %container.name,
                owner = ?container.icn_instance_id,
                "removing orphaned container"
            );
            let _ = self.shared.docker.stop(&container.name, 5).await;
            if self.shared.docker.remove(&container.name).await.is_ok() {
                reaped.push(container.name);
            }
        }
        reaped
    }
}

/// Whether a recorded owner process is still running.
#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    // Signal 0 performs the permission and existence checks without delivering anything.
    // SAFETY: `kill` with signal 0 has no side effects on the target process.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(unix))]
fn pid_is_alive(_pid: u32) -> bool {
    // Without a cheap liveness check, err toward leaving containers alone; an operator can
    // remove them, whereas killing a live ICN's container breaks a working session.
    true
}

impl Shared {
    fn definition(
        &self,
        configuration_id: &ModelServingConfigurationId,
    ) -> Result<EimModelDefinition, InventoryError> {
        self.definitions
            .get(configuration_id)
            .cloned()
            .ok_or_else(|| InventoryError::NotFound(configuration_id.0.clone()))
    }

    fn host_budget(&self) -> HostBudget {
        let stable_capacity_bytes = self
            .memory
            .sample()
            .map(|sample| {
                sample
                    .physical_capacity_bytes
                    .saturating_sub(sample.abort_reserve_bytes())
            })
            .unwrap_or(0);
        HostBudget {
            stable_capacity_bytes,
            container_limit_bytes: None,
        }
    }

    /// The shape a load would use: the largest admitted sequence count that fits.
    fn plan(&self, definition: &EimModelDefinition) -> Result<(ServingShape, MemoryEstimate), ()> {
        let budget = self.host_budget();
        let shape = crate::estimate::largest_fitting_shape(
            &definition.geometry,
            definition.context_tokens,
            definition.tensor_parallel_size,
            &budget,
        )
        .ok_or(())?;
        Ok((shape, MemoryEstimate::compute(&definition.geometry, &shape)))
    }

    fn bump_revision(&self) -> u64 {
        let revision = {
            let mut state = self.state.lock().expect("controller state lock");
            state.revision += 1;
            state.revision
        };
        // A dropped receiver is normal: nothing is watching until a client subscribes.
        let _ = self.revision_tx.send(revision);
        revision
    }

    fn set_lifecycle(&self, lifecycle: ModelInstanceLifecycle) {
        {
            let mut state = self.state.lock().expect("controller state lock");
            if let Some(resident) = state.resident.as_mut() {
                resident.lifecycle = lifecycle;
            }
        }
        self.bump_revision();
    }

    fn snapshot(&self) -> ModelInstancesSnapshot {
        let state = self.state.lock().expect("controller state lock");
        ModelInstancesSnapshot {
            revision: state.revision,
            instances: state
                .resident
                .as_ref()
                .map(|resident| ModelInstance {
                    id: resident.instance_id.clone(),
                    configuration_id: resident.configuration_id.clone(),
                    lifecycle: resident.lifecycle.clone(),
                })
                .into_iter()
                .collect(),
        }
    }

    /// Stops and removes the resident container, if there is one.
    async fn release_resident(&self, reason: ModelReleaseReason) {
        let (container_name, allocation) = {
            let mut state = self.state.lock().expect("controller state lock");
            let Some(resident) = state.resident.as_mut() else {
                return;
            };
            let allocation = match &resident.lifecycle {
                ModelInstanceLifecycle::Ready { allocation } => ModelStoppingAllocation::Resident {
                    allocation: allocation.clone(),
                },
                _ => ModelStoppingAllocation::Planned { allocation: None },
            };
            resident.lifecycle = ModelInstanceLifecycle::Stopping {
                reason,
                allocation: allocation.clone(),
            };
            // Dropping the backend closes its HTTP client before the container goes away.
            resident.backend = None;
            (resident.container_name.clone(), allocation)
        };
        let _ = allocation;
        self.bump_revision();

        // vLLM is PID 1 through `os.execv`, so SIGTERM reaches it directly and a real shutdown
        // window is worth giving.
        let _ = self.docker.stop(&container_name, 30).await;
        let _ = self.docker.remove(&container_name).await;

        {
            let mut state = self.state.lock().expect("controller state lock");
            if let Some(resident) = state.resident.as_mut() {
                resident.lifecycle = ModelInstanceLifecycle::Stopped { reason };
            }
        }
        self.bump_revision();
    }

    fn fail(&self, failure: ModelInstanceFailure) {
        self.set_lifecycle(ModelInstanceLifecycle::Failed { failure });
    }
}

fn operation_failure(
    code: &str,
    message: impl Into<String>,
    retryable: bool,
) -> ModelInstanceFailure {
    ModelInstanceFailure::Operation {
        code: code.to_owned(),
        message: message.into(),
        retryable,
    }
}

/// Builds the `LowMemory` failure the client renders with real byte figures.
fn low_memory_failure(
    required_bytes: u64,
    budget: &HostBudget,
    parallel_sequences: u32,
) -> ModelInstanceFailure {
    let usable = budget.usable_bytes();
    let deficit = required_bytes.saturating_sub(usable);
    ModelInstanceFailure::LowMemory {
        code: "low_memory".to_owned(),
        message: format!(
            "requires {:.1} GB but only {:.1} GB is available on this host",
            required_bytes as f64 / 1e9,
            usable as f64 / 1e9,
        ),
        // Freeing memory makes this succeed, so it is worth offering a retry.
        retryable: true,
        required_system_memory_bytes: required_bytes,
        allocation_headroom_bytes: usable,
        system_reserve_bytes: budget.stable_capacity_bytes.saturating_sub(usable),
        load_boundary_bytes: usable,
        minimum_additional_available_bytes: deficit,
        parallel_sequences: parallel_sequences.max(1),
    }
}

fn allocation_of(shape: &ServingShape, estimate: &MemoryEstimate) -> ModelInstanceAllocation {
    ModelInstanceAllocation {
        context_window_tokens: shape.context_tokens,
        parallel_sequences: shape.parallel_sequences,
        physical_context_tokens: shape
            .context_tokens
            .saturating_mul(shape.parallel_sequences.max(1)),
        memory_domains: vec![ModelInstanceMemoryDomain {
            memory_domain_id: MemoryDomainId::system(),
            model_bytes: estimate.weight_bytes,
            context_bytes: estimate.kv_bytes,
            compute_bytes: estimate.activation_bytes,
            auxiliary_bytes: estimate.runtime_bytes,
        }],
    }
}

fn plan_of(shape: &ServingShape, estimate: &MemoryEstimate) -> ModelLoadPlan {
    ModelLoadPlan {
        context_window_tokens: shape.context_tokens,
        parallel_sequences: shape.parallel_sequences,
        physical_context_tokens: shape
            .context_tokens
            .saturating_mul(shape.parallel_sequences.max(1)),
        required_system_memory_bytes: estimate.required_bytes,
    }
}

/// Binds an ephemeral loopback port and releases it for the container to take.
///
/// A race is possible between releasing and publishing, which the caller retries on.
fn allocate_host_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|error| error.to_string())?;
    listener
        .local_addr()
        .map(|address| address.port())
        .map_err(|error| error.to_string())
}

impl ModelInstanceController for EimModelInstanceController {
    fn set_residency_policy(
        &self,
        generation: u64,
        idle_timeout: Duration,
    ) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(async move {
            let mut state = self.shared.state.lock().expect("controller state lock");
            // A stale generation must not undo a newer policy; the client bumps it on every
            // connection change.
            if generation >= state.residency.generation {
                state.residency = ResidencyPolicy {
                    generation,
                    idle_timeout,
                };
            }
            Ok(())
        })
    }

    /// Computes the plan without touching Docker.
    ///
    /// The plan describes what a load *would* allocate, and the client asks for it while
    /// browsing models. Making it depend on the image being present would fail for a model
    /// that has not been installed yet, which is precisely when a user wants to see the cost.
    fn preview_load(
        &self,
        request: PreviewModelLoadRequest,
    ) -> BoxFuture<'_, Result<ModelLoadPlan, InventoryError>> {
        Box::pin(async move {
            let definition = self.shared.definition(&request.configuration.id)?;
            let (shape, estimate) = self.shared.plan(&definition).map_err(|()| {
                InventoryError::NotReady(format!(
                    "{} does not fit this host",
                    definition.canonical_name
                ))
            })?;
            Ok(plan_of(&shape, &estimate))
        })
    }

    fn load_instance(&self, request: LoadModelRequest) -> BoxStream<'static, ModelLoadEvent> {
        let shared = self.shared.clone();
        let mutation = Arc::clone(&self.mutation);
        // Bounded so a client that stops reading applies backpressure rather than letting
        // events accumulate; the canonical state is the snapshot regardless.
        let (events, receiver) = mpsc::channel(32);

        self.shared.runtime.spawn(async move {
            let _guard = mutation.lock().await;
            run_load(shared, request, events).await;
        });

        ReceiverStream::new(receiver).boxed()
    }

    fn stop_instance(
        &self,
        instance_id: ModelInstanceId,
    ) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(async move {
            let _guard = self.mutation.lock().await;
            {
                let state = self.shared.state.lock().expect("controller state lock");
                match state.resident.as_ref() {
                    Some(resident) if resident.instance_id == instance_id => {}
                    // Stopping something already gone is a success, so an idle reaper racing a
                    // user's stop does not surface an error.
                    _ => return Ok(()),
                }
            }
            self.shared
                .release_resident(ModelReleaseReason::UserStop)
                .await;
            Ok(())
        })
    }

    fn instances(&self) -> BoxFuture<'_, ModelInstancesSnapshot> {
        Box::pin(async move { self.shared.snapshot() })
    }

    fn watch_instances(&self) -> BoxStream<'static, ModelInstancesInvalidation> {
        WatchStream::new(self.shared.revision_tx.subscribe())
            .map(|revision| ModelInstancesInvalidation { revision })
            .boxed()
    }

    fn remove_installed(
        &self,
        package_id: ModelPackageId,
    ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>> {
        Box::pin(async move {
            let definition = self
                .shared
                .definitions
                .values()
                .find(|definition| definition.package_id == package_id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(package_id.0.clone()))?;

            {
                let state = self.shared.state.lock().expect("controller state lock");
                if let Some(resident) = state.resident.as_ref()
                    && resident.configuration_id == definition.configuration_id
                    && matches!(resident.lifecycle, ModelInstanceLifecycle::Ready { .. })
                {
                    return Err(InventoryError::Loaded(definition.canonical_name));
                }
            }

            // Stage 1 reclaims the image. The weight cache is removed alongside it once
            // Magnitude owns prefetching, since otherwise "uninstall" frees a megabyte of
            // layer and leaves tens of gigabytes of weights behind.
            let invocation = self
                .shared
                .docker
                .invoke(&[
                    "rmi".to_owned(),
                    "--force".to_owned(),
                    definition.image.clone(),
                ])
                .await
                .map_err(|error| InventoryError::Io(error.to_string()))?;

            Ok(RemoveInstalledModelPackageResponse {
                package_id,
                removed: invocation.succeeded() || invocation.stderr.contains("No such image"),
            })
        })
    }

    fn lease(
        &self,
        instance_id: ModelInstanceId,
        configuration_id: ModelServingConfigurationId,
    ) -> BoxFuture<'_, Result<ModelInstanceLease, InventoryError>> {
        Box::pin(async move {
            let backend =
                {
                    let mut state = self.shared.state.lock().expect("controller state lock");
                    let resident = state.resident.as_mut().ok_or_else(|| {
                        InventoryError::NotReady("no model is resident".to_owned())
                    })?;
                    // The exact pair must match. A request naming a stale instance must fail rather
                    // than be silently served by whatever replaced it.
                    if resident.instance_id != instance_id
                        || resident.configuration_id != configuration_id
                    {
                        return Err(InventoryError::NotFound(format!(
                            "instance {} is not resident",
                            instance_id.0
                        )));
                    }
                    if !matches!(resident.lifecycle, ModelInstanceLifecycle::Ready { .. }) {
                        return Err(InventoryError::NotReady(format!(
                            "instance {} is not ready",
                            instance_id.0
                        )));
                    }
                    // Leasing counts as activity, so a long conversation is never idle-reaped.
                    resident.last_activity = Instant::now();
                    resident.backend.clone().ok_or_else(|| {
                        InventoryError::NotReady("backend is not attached".to_owned())
                    })?
                };

            let aliases = Arc::new(self.aliases.read().expect("alias lock").clone());
            let state = Arc::clone(&self.shared.state);
            Ok(ModelInstanceLease::new(
                backend,
                instance_id,
                configuration_id,
                aliases,
                move || {
                    if let Ok(mut state) = state.lock()
                        && let Some(resident) = state.resident.as_mut()
                    {
                        resident.last_activity = Instant::now();
                    }
                },
            ))
        })
    }

    fn add_alias(&self, alias: String) {
        if let Ok(mut aliases) = self.aliases.write() {
            aliases.insert(alias);
        }
    }
}

/// The load ladder. Emits progress as it goes and leaves canonical state in the snapshot.
async fn run_load(shared: Shared, request: LoadModelRequest, events: mpsc::Sender<ModelLoadEvent>) {
    // A closed receiver means the client stopped reading; the load continues, because ACN
    // deliberately treats the stream as advisory and reads state from the snapshot.
    let emit = |event: ModelLoadEvent| {
        let events = events.clone();
        async move {
            let _ = events.send(event).await;
        }
    };

    emit(ModelLoadEvent::Progress {
        stage: ModelLoadStage::Queued,
        fraction: None,
        plan: None,
    })
    .await;

    let configuration_id = request.configuration.id.clone();
    let definition = match shared.definition(&configuration_id) {
        Ok(definition) => definition,
        Err(error) => {
            emit(ModelLoadEvent::Failed {
                failure: operation_failure("eim_unknown_configuration", error.to_string(), false),
            })
            .await;
            return;
        }
    };

    let budget = shared.host_budget();
    let (shape, estimate) = match shared.plan(&definition) {
        Ok(planned) => planned,
        Err(()) => {
            let required = MemoryEstimate::compute(
                &definition.geometry,
                &ServingShape {
                    context_tokens: definition.context_tokens,
                    parallel_sequences: 1,
                    tensor_parallel_size: definition.tensor_parallel_size,
                },
            )
            .required_bytes;
            emit(ModelLoadEvent::Failed {
                failure: low_memory_failure(required, &budget, 1),
            })
            .await;
            return;
        }
    };
    let plan = plan_of(&shape, &estimate);

    if definition.hf_token_required && shared.config.hf_token.is_none() {
        emit(ModelLoadEvent::Failed {
            failure: operation_failure(
                "eim_model_gated",
                format!(
                    "{} is a gated repository; set HF_TOKEN to use it",
                    definition.canonical_name
                ),
                false,
            ),
        })
        .await;
        return;
    }

    // Resolving: confirm the serving image is present. Building or pulling it belongs to
    // acquisition, so a missing image here is a configuration problem, not a transient one.
    emit(ModelLoadEvent::Progress {
        stage: ModelLoadStage::Resolving,
        fraction: None,
        plan: Some(plan.clone()),
    })
    .await;

    let image = match shared.docker.image_summary(&definition.image).await {
        Ok(Some(image)) => image,
        Ok(None) => {
            emit(ModelLoadEvent::Failed {
                failure: operation_failure(
                    "eim_image_missing",
                    format!("serving image {} is not present locally", definition.image),
                    false,
                ),
            })
            .await;
            return;
        }
        Err(error) => {
            emit(ModelLoadEvent::Failed {
                failure: operation_failure("eim_docker_failed", error.to_string(), true),
            })
            .await;
            return;
        }
    };

    // Unloading: one container at a time, so the previous instance terminalizes first.
    let had_resident = {
        let state = shared.state.lock().expect("controller state lock");
        state.resident.is_some()
    };
    if had_resident {
        emit(ModelLoadEvent::Progress {
            stage: ModelLoadStage::Unloading,
            fraction: None,
            plan: Some(plan.clone()),
        })
        .await;
        shared
            .release_resident(ModelReleaseReason::Replacement)
            .await;
    }

    let host_port = match allocate_host_port() {
        Ok(port) => port,
        Err(error) => {
            emit(ModelLoadEvent::Failed {
                failure: operation_failure("eim_port_allocation_failed", error, true),
            })
            .await;
            return;
        }
    };
    let container = container_name(&shared.config.icn_instance_id, &request.instance_id.0);

    // vLLM's CPU backend reserves a fraction of each NUMA node rather than an absolute size,
    // and its default of 0.92 fails on a host with anything else resident. No shipped EIM
    // profile sets it, so the reservation is derived from the estimate here.
    let node_capacity_bytes = if shared.config.numa_nodes > 1 {
        budget.stable_capacity_bytes / u64::from(shared.config.numa_nodes)
    } else {
        budget.stable_capacity_bytes
    };
    let cpu_utilization = crate::estimate::cpu_memory_utilization(
        &estimate,
        shape.tensor_parallel_size,
        node_capacity_bytes,
    );

    let launch = LaunchEnvironment {
        model_id: definition.canonical_name.clone(),
        profile_id: definition.eim_profile_id.clone(),
        context_tokens: shape.context_tokens,
        parallel_sequences: shape.parallel_sequences,
        served_model_name: definition.canonical_name.clone(),
        tool_call_parser: definition.tool_call_parser.clone(),
        reasoning_parser: definition.reasoning_parser.clone(),
        hf_token: definition
            .hf_token_required
            .then(|| shared.config.hf_token.clone())
            .flatten(),
        proxy: shared.config.proxy.clone(),
        offline_weights: shared.config.offline_weights,
        cpu_memory_utilization: Some(cpu_utilization),
        engine_args_override: BTreeMap::new(),
    };

    let spec = ContainerSpec {
        name: container.clone(),
        image: definition.image.clone(),
        host_port,
        container_port: CONTAINER_PORT,
        labels: ContainerLabels {
            icn_instance_id: shared.config.icn_instance_id.clone(),
            model_instance_id: request.instance_id.0.clone(),
            configuration_id: configuration_id.0.clone(),
            catalog_model_id: definition.catalog_model_id.clone(),
            eim_profile_id: definition.eim_profile_id.clone(),
            pid: std::process::id(),
        }
        .to_args(),
        environment: launch.to_environment(),
        mounts: vec![launch.cache_mount(&shared.config.host_cache_path.to_string_lossy())],
        memory_limit_bytes: Some(
            estimate
                .required_bytes
                .saturating_mul(shared.config.memory_limit_percent)
                / 100,
        ),
        stop_timeout_seconds: 30,
        shm_size_bytes: Some(shared.config.shm_size_bytes),
        // Lets vLLM bind worker memory to a NUMA node. Without it every worker logs
        // `numa_migrate_pages failed` and runs unbound, losing exactly the locality that made
        // a tensor-parallel split worth doing on a multi-socket host.
        capabilities: vec!["SYS_NICE".to_owned()],
        command: Vec::new(),
    };

    // Register the instance as Loading before starting, so a client polling the snapshot sees
    // the transition even if it never reads the stream.
    {
        let mut state = shared.state.lock().expect("controller state lock");
        state.resident = Some(Resident {
            instance_id: request.instance_id.clone(),
            configuration_id: configuration_id.clone(),
            container_name: container.clone(),
            lifecycle: ModelInstanceLifecycle::Loading {
                stage: ModelLoadStage::Loading,
                progress: None,
                planned_allocation: Some(plan.clone()),
            },
            backend: None,
            last_activity: Instant::now(),
        });
    }
    shared.bump_revision();

    emit(ModelLoadEvent::Progress {
        stage: ModelLoadStage::Loading,
        fraction: None,
        plan: Some(plan.clone()),
    })
    .await;

    if let Err(error) = shared.docker.run(&spec).await {
        shared.fail(operation_failure(
            "eim_container_start_failed",
            error.to_string(),
            true,
        ));
        emit(ModelLoadEvent::Failed {
            failure: operation_failure("eim_container_start_failed", error.to_string(), true),
        })
        .await;
        return;
    }

    let endpoint = format!("http://127.0.0.1:{host_port}");
    let probe = match HttpServerProbe::new(&endpoint) {
        Ok(probe) => probe,
        Err(error) => {
            let _ = shared.docker.remove(&container).await;
            shared.fail(operation_failure(
                "eim_http_client_failed",
                error.clone(),
                false,
            ));
            emit(ModelLoadEvent::Failed {
                failure: operation_failure("eim_http_client_failed", error, false),
            })
            .await;
            return;
        }
    };
    let watch = DockerContainerWatch::new(shared.docker.clone(), container.clone());

    let outcome = await_ready(&watch, &probe, shared.config.readiness, |_elapsed| {}).await;

    let served_model_name = match outcome {
        ReadinessOutcome::Ready {
            served_model_name, ..
        } => served_model_name,
        ReadinessOutcome::ContainerExited { state, .. } => {
            let logs = shared
                .docker
                .logs_tail(&container, 200)
                .await
                .unwrap_or_default();
            let failure = if state.oom_killed {
                // The container hit its own ceiling: report it with real figures rather than as
                // an opaque exit, so the user learns the model is too large for this host.
                low_memory_failure(estimate.required_bytes, &budget, shape.parallel_sequences)
            } else {
                classify_startup_failure(state.exit_code, &logs)
            };
            let _ = shared.docker.remove(&container).await;
            shared.fail(failure.clone());
            emit(ModelLoadEvent::Failed { failure }).await;
            return;
        }
        ReadinessOutcome::TimedOut { elapsed } => {
            let logs = shared
                .docker
                .logs_tail(&container, 200)
                .await
                .unwrap_or_default();
            let failure = operation_failure(
                "eim_readiness_timeout",
                format!(
                    "container did not start serving within {}s: {}",
                    elapsed.as_secs(),
                    last_log_line(&logs)
                ),
                true,
            );
            let _ = shared.docker.stop(&container, 10).await;
            let _ = shared.docker.remove(&container).await;
            shared.fail(failure.clone());
            emit(ModelLoadEvent::Failed { failure }).await;
            return;
        }
    };

    emit(ModelLoadEvent::Progress {
        stage: ModelLoadStage::Verifying,
        fraction: None,
        plan: Some(plan.clone()),
    })
    .await;

    let properties = ModelPropertiesSpec {
        served_model_name: served_model_name.clone(),
        container_model_path: PathBuf::from(crate::env_contract::CONTAINER_CACHE_PATH)
            .join(&definition.canonical_name),
        model_size_bytes: definition.weight_bytes,
        architecture: definition.architecture.clone(),
        context_tokens: shape.context_tokens,
        training_context_tokens: definition.geometry.max_position_embeddings,
        sliding_window_tokens: definition.sliding_window_tokens,
        tools: definition.tool_call_parser.is_some(),
        reasoning: definition.reasoning.clone(),
        modalities: definition.modalities,
        image_digest: image.content_digest.clone(),
        eim_profile_id: definition.eim_profile_id.clone(),
    }
    .to_model_properties();

    let backend = match crate::backend::EimCompletionBackend::new(
        configuration_id.0.clone(),
        endpoint,
        served_model_name,
        properties,
        shared.runtime.clone(),
    ) {
        Ok(backend) => Arc::new(backend) as Arc<dyn CompletionBackend>,
        Err(error) => {
            let _ = shared.docker.stop(&container, 10).await;
            let _ = shared.docker.remove(&container).await;
            let failure = operation_failure("eim_backend_failed", error.to_string(), false);
            shared.fail(failure.clone());
            emit(ModelLoadEvent::Failed { failure }).await;
            return;
        }
    };

    let allocation = allocation_of(&shape, &estimate);
    {
        let mut state = shared.state.lock().expect("controller state lock");
        if let Some(resident) = state.resident.as_mut() {
            resident.backend = Some(backend);
            resident.lifecycle = ModelInstanceLifecycle::Ready {
                allocation: allocation.clone(),
            };
            resident.last_activity = Instant::now();
        }
    }
    shared.bump_revision();

    emit(ModelLoadEvent::Ready {
        ready: LoadModelReady {
            instance_id: request.instance_id,
            configuration_id,
            allocation,
        },
    })
    .await;
}

fn last_log_line(logs: &str) -> String {
    logs.lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("(no container output)")
        .chars()
        .take(300)
        .collect()
}

/// Maps a failed container start onto a code the client can explain.
///
/// EIM exits 0 or 1 and nothing else, so the useful signal is in the log text. Each pattern is
/// a stable message from EIM's own configuration and profile handling, or from vLLM.
fn classify_startup_failure(exit_code: i64, logs: &str) -> ModelInstanceFailure {
    const PATTERNS: &[(&str, &str, bool)] = &[
        (
            "INFERENCE_MODEL_ID is required",
            "eim_config_invalid",
            false,
        ),
        (
            "No compatible profile found",
            "eim_profile_unavailable",
            false,
        ),
        ("none are auto-selectable", "eim_profile_manual_only", false),
        ("not found in registry", "eim_profile_id_unknown", false),
        ("gated repo", "eim_model_gated", false),
        ("Access to model", "eim_model_gated", false),
        ("401 Client Error", "eim_model_gated", false),
        ("403 Client Error", "eim_model_gated", false),
        (
            "Cannot find the requested files",
            "eim_weights_missing",
            true,
        ),
        ("HF_HUB_OFFLINE", "eim_weights_missing", true),
        ("Illegal instruction", "eim_cpu_unsupported", false),
        ("SIGILL", "eim_cpu_unsupported", false),
        (
            "engine_args validation failed",
            "eim_engine_args_invalid",
            false,
        ),
    ];

    for (needle, code, retryable) in PATTERNS {
        if logs.contains(needle) {
            return operation_failure(
                code,
                format!("container exited with {exit_code}: {}", last_log_line(logs)),
                *retryable,
            );
        }
    }
    operation_failure(
        "eim_engine_start_failed",
        format!("container exited with {exit_code}: {}", last_log_line(logs)),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> ModelGeometry {
        ModelGeometry {
            total_parameters: 8_200_000_000,
            active_parameters: 8_200_000_000,
            num_hidden_layers: 36,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 40_960,
            sliding_window: None,
            vision: false,
        }
    }

    fn definition() -> EimModelDefinition {
        EimModelDefinition {
            configuration_id: ModelServingConfigurationId("eim-qwen3-8b-ctx32768".to_owned()),
            package_id: ModelPackageId("eim--Qwen--Qwen3-8B--v1".to_owned()),
            catalog_model_id: "qwen-qwen3-8b".to_owned(),
            canonical_name: "Qwen/Qwen3-8B".to_owned(),
            image: "magnitude-eim-xeon-qwen-qwen3-8b:v1".to_owned(),
            eim_profile_id: "vllm-xeon-bf16-tp2".to_owned(),
            tensor_parallel_size: 2,
            context_tokens: 32_768,
            geometry: geometry(),
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

    fn controller() -> EimModelInstanceController {
        EimModelInstanceController::new(
            DockerCli::new("magnitude-nonexistent-docker"),
            vec![definition()],
            EimControllerConfig::default(),
            Handle::current(),
        )
    }

    #[tokio::test]
    async fn starts_with_no_resident_instance() {
        let snapshot = controller().instances().await;

        assert_eq!(snapshot.revision, 0);
        assert!(snapshot.instances.is_empty());
    }

    #[tokio::test]
    async fn previews_a_plan_without_touching_docker() {
        // The Docker command is deliberately bogus: a preview must not need the daemon, because
        // the client asks for it while browsing models that are not installed.
        let plan = controller()
            .preview_load(PreviewModelLoadRequest {
                configuration: icn_contracts::models::ModelServingConfiguration {
                    id: definition().configuration_id,
                    bundle: icn_contracts::models::ServableModelBundle::Standalone {
                        package: test_package(),
                    },
                    profile: icn_contracts::models::ServingProfile {
                        context_length: 32_768,
                    },
                },
            })
            .await
            .expect("a plan");

        assert_eq!(plan.context_window_tokens, 32_768);
        assert!(plan.parallel_sequences >= 1);
        assert_eq!(
            plan.physical_context_tokens,
            plan.context_window_tokens * plan.parallel_sequences
        );
        assert!(plan.required_system_memory_bytes > 16_400_000_000);
    }

    fn test_package() -> icn_contracts::models::ModelPackage {
        icn_contracts::models::ModelPackage {
            id: definition().package_id,
            source: icn_contracts::models::ModelPackageSource::HuggingFace {
                repository: "Qwen/Qwen3-8B".to_owned(),
                revision: "main".to_owned(),
            },
            files: Vec::new(),
            relationships: Vec::new(),
            properties: icn_contracts::models::ModelPackageProperties {
                format: "safetensors".to_owned(),
                quantization: "bf16".to_owned(),
                quantization_name: "BF16".to_owned(),
                architecture: "Qwen3ForCausalLM".to_owned(),
                maximum_context_length: Some(40_960),
                intrinsic_model_id: Some("Qwen/Qwen3-8B".to_owned()),
                intrinsic_quality_id: Some("bf16".to_owned()),
            },
        }
    }

    #[tokio::test]
    async fn previewing_an_unknown_configuration_is_not_found() {
        let error = controller()
            .preview_load(PreviewModelLoadRequest {
                configuration: icn_contracts::models::ModelServingConfiguration {
                    id: ModelServingConfigurationId("nonexistent".to_owned()),
                    bundle: icn_contracts::models::ServableModelBundle::Standalone {
                        package: test_package(),
                    },
                    profile: icn_contracts::models::ServingProfile {
                        context_length: 4096,
                    },
                },
            })
            .await
            .expect_err("should not be found");

        assert!(matches!(error, InventoryError::NotFound(_)));
    }

    #[tokio::test]
    async fn leasing_without_a_resident_instance_is_not_ready() {
        let outcome = controller()
            .lease(
                ModelInstanceId("instance-1".to_owned()),
                definition().configuration_id,
            )
            .await;

        match outcome {
            Err(InventoryError::NotReady(_)) => {}
            Err(other) => panic!("expected NotReady, got {other:?}"),
            Ok(_) => panic!("a lease must not be granted with nothing resident"),
        }
    }

    #[tokio::test]
    async fn stopping_an_unknown_instance_succeeds() {
        // An idle reaper racing a user's stop must not surface an error.
        controller()
            .stop_instance(ModelInstanceId("never-existed".to_owned()))
            .await
            .expect("stopping something absent is a success");
    }

    #[tokio::test]
    async fn a_stale_residency_generation_does_not_undo_a_newer_policy() {
        let controller = controller();
        controller
            .set_residency_policy(5, Duration::from_secs(600))
            .await
            .expect("policy accepted");
        controller
            .set_residency_policy(3, Duration::from_secs(1))
            .await
            .expect("policy accepted");

        let state = controller.shared.state.lock().expect("lock");
        assert_eq!(state.residency.generation, 5);
        assert_eq!(state.residency.idle_timeout, Duration::from_secs(600));
    }

    #[tokio::test]
    async fn aliases_accumulate_for_leases() {
        let controller = controller();
        controller.add_alias("qwen3".to_owned());
        controller.add_alias("default".to_owned());

        let aliases = controller.aliases.read().expect("lock");
        assert!(aliases.contains("qwen3") && aliases.contains("default"));
    }

    #[tokio::test]
    async fn a_load_of_an_unknown_configuration_fails_without_starting_anything() {
        let controller = controller();
        let events: Vec<_> = controller
            .load_instance(LoadModelRequest {
                instance_id: ModelInstanceId("instance-1".to_owned()),
                configuration: icn_contracts::models::ModelServingConfiguration {
                    id: ModelServingConfigurationId("nonexistent".to_owned()),
                    bundle: icn_contracts::models::ServableModelBundle::Standalone {
                        package: test_package(),
                    },
                    profile: icn_contracts::models::ServingProfile {
                        context_length: 4096,
                    },
                },
            })
            .collect()
            .await;

        assert!(matches!(
            events.first(),
            Some(ModelLoadEvent::Progress {
                stage: ModelLoadStage::Queued,
                ..
            })
        ));
        match events.last() {
            Some(ModelLoadEvent::Failed {
                failure:
                    ModelInstanceFailure::Operation {
                        code, retryable, ..
                    },
            }) => {
                assert_eq!(code, "eim_unknown_configuration");
                assert!(
                    !retryable,
                    "a nonexistent configuration will not fix itself"
                );
            }
            other => panic!("expected a failure, got {other:?}"),
        }
        // Nothing was registered, so no phantom instance is left in the snapshot.
        assert!(controller.instances().await.instances.is_empty());
    }

    #[tokio::test]
    async fn a_missing_serving_image_fails_as_a_configuration_problem() {
        let controller = controller();
        let events: Vec<_> = controller
            .load_instance(LoadModelRequest {
                instance_id: ModelInstanceId("instance-1".to_owned()),
                configuration: icn_contracts::models::ModelServingConfiguration {
                    id: definition().configuration_id,
                    bundle: icn_contracts::models::ServableModelBundle::Standalone {
                        package: test_package(),
                    },
                    profile: icn_contracts::models::ServingProfile {
                        context_length: 32_768,
                    },
                },
            })
            .collect()
            .await;

        // The bogus docker command makes image inspection fail, which is reported as a Docker
        // failure rather than being mistaken for a missing image.
        match events.last() {
            Some(ModelLoadEvent::Failed {
                failure:
                    ModelInstanceFailure::Operation {
                        code, retryable, ..
                    },
            }) => {
                assert_eq!(code, "eim_docker_failed");
                assert!(retryable, "the daemon may come back");
            }
            other => panic!("expected a failure, got {other:?}"),
        }
        // Resolving must be reported with the plan attached, since that is the first point a
        // client can show what the load would cost.
        assert!(events.iter().any(|event| matches!(
            event,
            ModelLoadEvent::Progress {
                stage: ModelLoadStage::Resolving,
                plan: Some(_),
                ..
            }
        )));
    }

    #[test]
    fn classifies_every_stable_eim_startup_message() {
        for (log, expected_code, expected_retryable) in [
            (
                "Error: INFERENCE_MODEL_ID is required but not set",
                "eim_config_invalid",
                false,
            ),
            (
                "Error: No compatible profile found for model=...",
                "eim_profile_unavailable",
                false,
            ),
            (
                "Compatible profiles found but none are auto-selectable",
                "eim_profile_manual_only",
                false,
            ),
            (
                "Specified profile ID 'x' not found in registry",
                "eim_profile_id_unknown",
                false,
            ),
            (
                "You are trying to access a gated repo",
                "eim_model_gated",
                false,
            ),
            ("401 Client Error: Unauthorized", "eim_model_gated", false),
            (
                "Cannot find the requested files in the disk cache",
                "eim_weights_missing",
                true,
            ),
            (
                "Illegal instruction (core dumped)",
                "eim_cpu_unsupported",
                false,
            ),
            (
                "vLLM engine_args validation failed",
                "eim_engine_args_invalid",
                false,
            ),
            (
                "something nobody has seen before",
                "eim_engine_start_failed",
                true,
            ),
        ] {
            match classify_startup_failure(1, log) {
                ModelInstanceFailure::Operation {
                    code, retryable, ..
                } => {
                    assert_eq!(code, expected_code, "log: {log}");
                    assert_eq!(retryable, expected_retryable, "log: {log}");
                }
                other => panic!("expected an operation failure, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_startup_failure_message_carries_the_last_log_line() {
        let failure =
            classify_startup_failure(1, "first line\nError: No compatible profile found\n\n");

        match failure {
            ModelInstanceFailure::Operation { message, .. } => {
                assert!(message.contains("exited with 1"));
                assert!(message.contains("No compatible profile found"), "{message}");
            }
            other => panic!("expected an operation failure, got {other:?}"),
        }
    }

    #[test]
    fn low_memory_failure_reports_the_real_shortfall() {
        let budget = HostBudget {
            stable_capacity_bytes: 242_000_000_000,
            container_limit_bytes: None,
        };
        match low_memory_failure(255_000_000_000, &budget, 2) {
            ModelInstanceFailure::LowMemory {
                code,
                required_system_memory_bytes,
                allocation_headroom_bytes,
                minimum_additional_available_bytes,
                parallel_sequences,
                retryable,
                message,
                system_reserve_bytes: _,
                load_boundary_bytes: _,
            } => {
                assert_eq!(code, "low_memory");
                assert_eq!(required_system_memory_bytes, 255_000_000_000);
                assert_eq!(allocation_headroom_bytes, 242_000_000_000);
                assert_eq!(minimum_additional_available_bytes, 13_000_000_000);
                assert_eq!(parallel_sequences, 2);
                // Freeing memory makes it work, so a retry is worth offering.
                assert!(retryable);
                assert!(message.contains("255.0 GB"), "{message}");
            }
            other => panic!("expected LowMemory, got {other:?}"),
        }
    }

    #[test]
    fn an_allocation_reports_the_four_memory_buckets() {
        let shape = ServingShape {
            context_tokens: 32_768,
            parallel_sequences: 2,
            tensor_parallel_size: 2,
        };
        let estimate = MemoryEstimate::compute(&geometry(), &shape);
        let allocation = allocation_of(&shape, &estimate);

        assert_eq!(allocation.memory_domains.len(), 1);
        let domain = &allocation.memory_domains[0];
        // These four are what the client displays, so none may be lumped into another.
        assert_eq!(domain.model_bytes, estimate.weight_bytes);
        assert_eq!(domain.context_bytes, estimate.kv_bytes);
        assert_eq!(domain.compute_bytes, estimate.activation_bytes);
        assert_eq!(domain.auxiliary_bytes, estimate.runtime_bytes);
        assert_eq!(allocation.physical_context_tokens, 65_536);
    }

    #[tokio::test]
    async fn reaping_tolerates_an_unreachable_daemon() {
        // Boot must not fail because Docker is briefly unavailable; the preflight reports that.
        assert!(controller().reap_orphans().await.is_empty());
    }

    #[test]
    fn only_a_ready_instance_is_idle_reaped() {
        let ready = ModelInstanceLifecycle::Ready {
            allocation: allocation_of(
                &ServingShape {
                    context_tokens: 4096,
                    parallel_sequences: 1,
                    tensor_parallel_size: 1,
                },
                &MemoryEstimate::compute(
                    &geometry(),
                    &ServingShape {
                        context_tokens: 4096,
                        parallel_sequences: 1,
                        tensor_parallel_size: 1,
                    },
                ),
            ),
        };
        let timeout = Duration::from_secs(600);

        assert!(is_idle_past(&ready, Duration::from_secs(601), timeout));
        assert!(!is_idle_past(&ready, Duration::from_secs(599), timeout));

        // Reaping mid-load would race the load it belongs to.
        let loading = ModelInstanceLifecycle::Loading {
            stage: ModelLoadStage::Loading,
            progress: None,
            planned_allocation: None,
        };
        assert!(!is_idle_past(&loading, Duration::from_secs(9_999), timeout));

        // Already released: nothing left to do.
        let stopped = ModelInstanceLifecycle::Stopped {
            reason: ModelReleaseReason::UserStop,
        };
        assert!(!is_idle_past(&stopped, Duration::from_secs(9_999), timeout));

        // A zero timeout means "never reap", not "reap immediately".
        assert!(!is_idle_past(
            &ready,
            Duration::from_secs(9_999),
            Duration::ZERO
        ));
    }

    #[test]
    fn a_live_process_is_recognized() {
        // Our own pid is alive, so a container we own is never reaped.
        assert!(pid_is_alive(std::process::id()));
    }
}
