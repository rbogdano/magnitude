//! Installing a container-backed model.
//!
//! Two things have to be true before a model can be served: its serving image is on the host, and
//! its weights are in EIM's Local Directory layout under the mounted cache. This drives both and
//! reports them through the download surface the client already has, because that surface is
//! where progress and cancellation live — `reconcile` returning `Current` gives the client
//! nothing to show and nothing to cancel, which for a step measured in tens of gigabytes is not
//! an acceptable shape.
//!
//! The stages are used for what they say. `Resolving` covers the image, which may be pulled or
//! built and reports no bytes; `CheckingSpace` is the disk pre-check; `Downloading` is the weight
//! fetch and is the only stage that moves the byte counters; `Verifying` re-checks what landed;
//! `Publishing` writes the manifest that makes the result recognisable on a later start.
//!
//! Ordering is deliberate. The image is resolved first because every way it can fail — no
//! checkout, no registry, an unknown model — is a configuration mistake that should surface in
//! seconds rather than after a 60 GB download.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use icn_contracts::InventoryError;
use icn_contracts::inventory::{DownloadFailure, DownloadStage};
use icn_contracts::models::{
    ModelDownload, ModelDownloadId, ModelDownloadState, ModelPackageId, ServableModelBundle,
};

use crate::controller::EimModelDefinition;
use crate::docker::image::{ImageError, ImageResolver, ImageStage};
use crate::weights::{self, WeightError, WeightSource};

/// How often the byte counters are published while a file streams.
///
/// The client polls the download list roughly every second, so a finer cadence only costs lock
/// traffic. Coarser and a fetch looks stalled.
const PROGRESS_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

/// Everything an acquisition needs that is not the model itself.
pub struct AcquisitionContext {
    pub resolver: Arc<ImageResolver>,
    pub weights: Arc<WeightSource>,
    /// Host directory mounted as the in-container weight cache.
    pub cache_root: PathBuf,
}

struct Record {
    id: ModelDownloadId,
    bundle: ServableModelBundle,
    state: ModelDownloadState,
    cancelled: Arc<AtomicBool>,
}

/// Tracks installations in flight and the ones that finished.
///
/// Records are kept after they complete: the client's projection reads the download list to learn
/// why an install failed, and a record dropped on completion would make a failure indistinguish-
/// able from a request that never happened.
pub struct EimAcquisitions {
    context: Arc<AcquisitionContext>,
    records: Arc<Mutex<Vec<Record>>>,
    tasks: Mutex<HashMap<String, tokio::task::JoinHandle<()>>>,
    next_id: AtomicU64,
}

impl EimAcquisitions {
    #[must_use]
    pub fn new(context: AcquisitionContext) -> Self {
        Self {
            context: Arc::new(context),
            records: Arc::new(Mutex::new(Vec::new())),
            tasks: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Whether this model's weights are completely present.
    pub async fn weights_present(&self, definition: &EimModelDefinition) -> bool {
        weights::is_present(&self.context.cache_root, &definition.canonical_name).await
    }

    /// Removes a model's weights, so uninstalling reclaims the gigabytes rather than a thin layer.
    pub async fn remove_weights(&self, definition: &EimModelDefinition) -> std::io::Result<()> {
        weights::remove(&self.context.cache_root, &definition.canonical_name).await
    }

    /// Starts installing one model, or returns the identifier of the attempt already running.
    ///
    /// Idempotent on purpose: the client retries a reconciliation whenever its projection has not
    /// caught up yet, and a second Docker build racing the first would be destructive.
    pub fn begin(&self, definition: &EimModelDefinition) -> ModelDownloadId {
        let bundle = ServableModelBundle::Standalone {
            package: crate::package::model_package(definition),
        };
        if let Some(existing) = self.active_for(&definition.package_id) {
            return existing;
        }

        let id = ModelDownloadId(format!(
            "eim-install-{}",
            self.next_id.fetch_add(1, Ordering::Relaxed)
        ));
        let cancelled = Arc::new(AtomicBool::new(false));
        self.records.lock().expect("records").push(Record {
            id: id.clone(),
            bundle,
            state: ModelDownloadState::Pending {
                completed_bytes: 0,
                total_bytes: definition.weight_bytes,
            },
            cancelled: Arc::clone(&cancelled),
        });

        let handle = tokio::spawn(run(
            Arc::clone(&self.context),
            Arc::clone(&self.records),
            id.clone(),
            definition.clone(),
            cancelled,
        ));
        self.tasks
            .lock()
            .expect("tasks")
            .insert(id.0.clone(), handle);
        id
    }

    /// The identifier of an unfinished attempt at this package, if there is one.
    fn active_for(&self, package_id: &ModelPackageId) -> Option<ModelDownloadId> {
        let records = self.records.lock().expect("records");
        records
            .iter()
            .rev()
            .find(|record| {
                bundle_package_id(&record.bundle) == Some(package_id)
                    && matches!(
                        record.state,
                        ModelDownloadState::Pending { .. } | ModelDownloadState::Downloading { .. }
                    )
            })
            .map(|record| record.id.clone())
    }

    #[must_use]
    pub fn list(&self) -> Vec<ModelDownload> {
        self.records
            .lock()
            .expect("records")
            .iter()
            .map(|record| ModelDownload {
                id: record.id.clone(),
                bundle: record.bundle.clone(),
                state: record.state.clone(),
            })
            .collect()
    }

    /// Stops an installation. The fetch notices between chunks and removes what it had written.
    pub fn cancel(&self, id: &ModelDownloadId) -> Result<ModelDownload, InventoryError> {
        let mut records = self.records.lock().expect("records");
        let record = records
            .iter_mut()
            .find(|record| record.id == *id)
            .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
        record.cancelled.store(true, Ordering::Relaxed);
        let (completed_bytes, total_bytes) = byte_counts(&record.state);
        // The task also writes `Cancelled` when it unwinds, but recording it here means a caller
        // that polls immediately does not see the request as still running.
        record.state = ModelDownloadState::Cancelled {
            completed_bytes,
            total_bytes,
        };
        Ok(ModelDownload {
            id: record.id.clone(),
            bundle: record.bundle.clone(),
            state: record.state.clone(),
        })
    }

    /// Marks a failure as seen, which is what returns the model to an offerable state.
    pub fn acknowledge_failure(
        &self,
        id: &ModelDownloadId,
    ) -> Result<ModelDownload, InventoryError> {
        let mut records = self.records.lock().expect("records");
        let record = records
            .iter_mut()
            .find(|record| record.id == *id)
            .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
        if let ModelDownloadState::Failed {
            completed_bytes,
            total_bytes,
            failure,
            ..
        } = &record.state
        {
            record.state = ModelDownloadState::Failed {
                completed_bytes: *completed_bytes,
                total_bytes: *total_bytes,
                failure: failure.clone(),
                acknowledged: true,
            };
        }
        Ok(ModelDownload {
            id: record.id.clone(),
            bundle: record.bundle.clone(),
            state: record.state.clone(),
        })
    }
}

fn bundle_package_id(bundle: &ServableModelBundle) -> Option<&ModelPackageId> {
    match bundle {
        ServableModelBundle::Standalone { package } => Some(&package.id),
        _ => None,
    }
}

fn byte_counts(state: &ModelDownloadState) -> (u64, u64) {
    match state {
        ModelDownloadState::Pending {
            completed_bytes,
            total_bytes,
        }
        | ModelDownloadState::Downloading {
            completed_bytes,
            total_bytes,
            ..
        }
        | ModelDownloadState::Failed {
            completed_bytes,
            total_bytes,
            ..
        }
        | ModelDownloadState::Cancelled {
            completed_bytes,
            total_bytes,
        } => (*completed_bytes, *total_bytes),
        ModelDownloadState::Completed => (0, 0),
    }
}

/// The whole installation, as the spawned task runs it.
async fn run(
    context: Arc<AcquisitionContext>,
    records: Arc<Mutex<Vec<Record>>>,
    id: ModelDownloadId,
    definition: EimModelDefinition,
    cancelled: Arc<AtomicBool>,
) {
    let outcome = install(&context, &records, &id, &definition, &cancelled).await;
    let mut guard = records.lock().expect("records");
    let Some(record) = guard.iter_mut().find(|record| record.id == id) else {
        return;
    };
    // A cancellation already recorded by `cancel` is not overwritten: the request is what the
    // user did, and reporting it as a failure afterwards would misattribute the cause.
    if matches!(record.state, ModelDownloadState::Cancelled { .. }) {
        return;
    }
    let (completed_bytes, total_bytes) = byte_counts(&record.state);
    record.state = match outcome {
        Ok(()) => ModelDownloadState::Completed,
        Err(AcquisitionFailure::Cancelled) => ModelDownloadState::Cancelled {
            completed_bytes,
            total_bytes,
        },
        Err(failure) => {
            tracing::warn!(
                model = %definition.canonical_name,
                download = %id.0,
                "installing a container-backed model failed: {failure}"
            );
            ModelDownloadState::Failed {
                completed_bytes,
                total_bytes,
                failure: failure.into_download_failure(),
                acknowledged: false,
            }
        }
    };
}

#[derive(Debug, thiserror::Error)]
enum AcquisitionFailure {
    #[error("the serving image could not be prepared: {0}")]
    Image(#[from] ImageError),
    #[error("the weights could not be fetched: {0}")]
    Weights(#[from] WeightError),
    #[error("cancelled")]
    Cancelled,
}

impl AcquisitionFailure {
    fn into_download_failure(self) -> DownloadFailure {
        match self {
            // An absent image with nowhere to get it from, a missing checkout, or a malformed
            // identifier are all "the source cannot give us this", which is what the operator
            // needs to act on. A daemon or build failure is reported verbatim instead, because
            // the message is the only thing that distinguishes them.
            Self::Image(ImageError::Absent { .. } | ImageError::MissingSource { .. }) => {
                DownloadFailure::SourceUnavailable
            }
            Self::Image(error) => DownloadFailure::Internal {
                message: error.to_string(),
            },
            Self::Weights(WeightError::Gated { .. }) => DownloadFailure::SourceUnavailable,
            Self::Weights(WeightError::Unreadable { .. } | WeightError::NoWeights { .. }) => {
                DownloadFailure::SourceUnavailable
            }
            Self::Weights(WeightError::Transport(error)) => {
                if error.is_connect() || error.is_timeout() {
                    DownloadFailure::NetworkUnavailable
                } else {
                    DownloadFailure::Interrupted
                }
            }
            Self::Weights(WeightError::Corrupt { .. } | WeightError::ShortFile { .. }) => {
                DownloadFailure::CorruptDownload
            }
            Self::Weights(WeightError::Storage { .. }) => DownloadFailure::LocalStorageFailure,
            Self::Weights(WeightError::InsufficientSpace {
                required_bytes,
                available_bytes,
            }) => DownloadFailure::InsufficientDiskSpace {
                required_bytes,
                available_bytes,
            },
            // A malformed proxy URL is configuration, not the network being down. It reaches here
            // only if the client is rebuilt mid-run, since startup validates it.
            Self::Weights(error @ WeightError::Proxy { .. }) => DownloadFailure::Internal {
                message: error.to_string(),
            },
            Self::Weights(WeightError::Cancelled) | Self::Cancelled => DownloadFailure::Interrupted,
        }
    }
}

async fn install(
    context: &AcquisitionContext,
    records: &Arc<Mutex<Vec<Record>>>,
    id: &ModelDownloadId,
    definition: &EimModelDefinition,
    cancelled: &Arc<AtomicBool>,
) -> Result<(), AcquisitionFailure> {
    let publish = |stage: DownloadStage, completed: u64, total: u64, rate: Option<u64>| {
        let mut guard = records.lock().expect("records");
        if let Some(record) = guard.iter_mut().find(|record| record.id == *id)
            && !matches!(record.state, ModelDownloadState::Cancelled { .. })
        {
            record.state = ModelDownloadState::Downloading {
                stage,
                completed_bytes: completed,
                total_bytes: total,
                bytes_per_second: rate,
            };
        }
    };

    // The image first: every way it fails is a configuration mistake, and finding out in seconds
    // is worth more than the ordering costs.
    publish(DownloadStage::Resolving, 0, definition.weight_bytes, None);
    let outcome = context
        .resolver
        .resolve(&definition.image, &definition.canonical_name, |stage| {
            tracing::info!(
                model = %definition.canonical_name,
                image = %definition.image,
                "{}", ImageStage::describe(stage)
            );
        })
        .await?;
    tracing::info!(
        model = %definition.canonical_name,
        digest = %outcome.summary().content_digest,
        "serving image ready"
    );
    if cancelled.load(Ordering::Relaxed) {
        return Err(AcquisitionFailure::Cancelled);
    }

    if weights::is_present(&context.cache_root, &definition.canonical_name).await {
        tracing::info!(model = %definition.canonical_name, "weights already cached");
        return Ok(());
    }

    let snapshot = context
        .weights
        .snapshot(&definition.canonical_name)
        .await
        .map_err(AcquisitionFailure::Weights)?;
    tracing::info!(
        model = %definition.canonical_name,
        revision = %snapshot.revision,
        files = snapshot.files.len(),
        bytes = snapshot.total_bytes,
        "resolved repository"
    );

    publish(DownloadStage::CheckingSpace, 0, snapshot.total_bytes, None);
    if let Some(available) = weights::available_bytes(&context.cache_root) {
        // A tenth on top: a filesystem filled to the last byte by a model download is a host that
        // cannot write logs either.
        let required = snapshot
            .total_bytes
            .saturating_add(snapshot.total_bytes / 10);
        if available < required {
            return Err(AcquisitionFailure::Weights(
                WeightError::InsufficientSpace {
                    required_bytes: required,
                    available_bytes: available,
                },
            ));
        }
    }

    let started = std::time::Instant::now();
    let mut last_published = std::time::Instant::now();
    publish(DownloadStage::Downloading, 0, snapshot.total_bytes, None);
    context
        .weights
        .fetch(
            &snapshot,
            &weights::model_directory(&context.cache_root, &definition.canonical_name),
            cancelled,
            |progress| {
                if last_published.elapsed() < PROGRESS_INTERVAL {
                    return;
                }
                last_published = std::time::Instant::now();
                let seconds = started.elapsed().as_secs_f64();
                let rate = (seconds > 0.5).then(|| {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let rate = (progress.completed_bytes as f64 / seconds) as u64;
                    rate
                });
                publish(
                    DownloadStage::Downloading,
                    progress.completed_bytes,
                    progress.total_bytes,
                    rate,
                );
            },
        )
        .await
        .map_err(AcquisitionFailure::Weights)?;

    publish(
        DownloadStage::Verifying,
        snapshot.total_bytes,
        snapshot.total_bytes,
        None,
    );
    let directory = weights::model_directory(&context.cache_root, &definition.canonical_name);
    for file in &snapshot.files {
        let path = directory.join(&file.path);
        let actual = tokio::fs::metadata(&path).await.map(|m| m.len()).ok();
        if actual != Some(file.size_bytes) {
            return Err(AcquisitionFailure::Weights(WeightError::ShortFile {
                path: file.path.clone(),
                expected_bytes: file.size_bytes,
                actual_bytes: actual.unwrap_or(0),
            }));
        }
    }

    publish(
        DownloadStage::Publishing,
        snapshot.total_bytes,
        snapshot.total_bytes,
        None,
    );
    weights::write_manifest(&context.cache_root, &snapshot)
        .await
        .map_err(AcquisitionFailure::Weights)?;
    tracing::info!(
        model = %definition.canonical_name,
        seconds = started.elapsed().as_secs(),
        "weights installed"
    );
    Ok(())
}

/// The catalog's view of installation, so `reconcile` can start one without owning the machinery.
pub trait CatalogAcquisition: Send + Sync + 'static {
    /// Both halves of "installed": the serving image and the weights.
    fn is_installed<'a>(&'a self, definition: &'a EimModelDefinition) -> BoxFuture<'a, bool>;

    /// Begins installing, returning the download the client should follow.
    fn begin<'a>(
        &'a self,
        definition: &'a EimModelDefinition,
    ) -> BoxFuture<'a, Result<ModelDownloadId, InventoryError>>;

    /// Removes a model's weights.
    fn remove<'a>(
        &'a self,
        definition: &'a EimModelDefinition,
    ) -> BoxFuture<'a, Result<(), InventoryError>>;
}

/// Installation backed by Docker and Hugging Face.
pub struct DockerAcquisition {
    acquisitions: Arc<EimAcquisitions>,
    docker: crate::docker::DockerCli,
}

impl DockerAcquisition {
    #[must_use]
    pub fn new(acquisitions: Arc<EimAcquisitions>, docker: crate::docker::DockerCli) -> Self {
        Self {
            acquisitions,
            docker,
        }
    }
}

impl CatalogAcquisition for DockerAcquisition {
    fn is_installed<'a>(&'a self, definition: &'a EimModelDefinition) -> BoxFuture<'a, bool> {
        Box::pin(async move {
            // Both, because either alone produces a container that fails: an image without
            // weights cannot serve offline, and weights without an image cannot start at all.
            if !self.acquisitions.weights_present(definition).await {
                return false;
            }
            self.docker
                .image_summary(&definition.image)
                .await
                .ok()
                .flatten()
                .is_some()
        })
    }

    fn begin<'a>(
        &'a self,
        definition: &'a EimModelDefinition,
    ) -> BoxFuture<'a, Result<ModelDownloadId, InventoryError>> {
        Box::pin(async move { Ok(self.acquisitions.begin(definition)) })
    }

    fn remove<'a>(
        &'a self,
        definition: &'a EimModelDefinition,
    ) -> BoxFuture<'a, Result<(), InventoryError>> {
        Box::pin(async move {
            self.acquisitions
                .remove_weights(definition)
                .await
                .map_err(|error| {
                    InventoryError::Internal(format!(
                        "removing weights for {}: {error}",
                        definition.canonical_name
                    ))
                })
        })
    }
}

/// Reports nothing installed and refuses to install. Used by `--fake` and in tests.
pub struct NoAcquisition;

impl CatalogAcquisition for NoAcquisition {
    fn is_installed<'a>(&'a self, _definition: &'a EimModelDefinition) -> BoxFuture<'a, bool> {
        Box::pin(async { false })
    }

    fn begin<'a>(
        &'a self,
        definition: &'a EimModelDefinition,
    ) -> BoxFuture<'a, Result<ModelDownloadId, InventoryError>> {
        Box::pin(async move {
            Err(InventoryError::Unsupported(format!(
                "this ICN cannot install {}: no container runtime is configured",
                definition.canonical_name
            )))
        })
    }

    fn remove<'a>(
        &'a self,
        _definition: &'a EimModelDefinition,
    ) -> BoxFuture<'a, Result<(), InventoryError>> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docker::DockerCli;
    use crate::docker::image::ImageSource;

    fn definition(name: &str, package: &str) -> EimModelDefinition {
        let mut definition = crate::controller::tests_support::definition();
        definition.canonical_name = name.to_owned();
        definition.package_id = ModelPackageId(package.to_owned());
        definition
    }

    fn acquisitions(cache_root: PathBuf) -> EimAcquisitions {
        let docker = DockerCli::new("docker-that-does-not-exist");
        EimAcquisitions::new(AcquisitionContext {
            resolver: Arc::new(ImageResolver::new(docker, ImageSource::PresentOnly)),
            weights: Arc::new(WeightSource::new(
                reqwest::Client::new(),
                "http://127.0.0.1:1".to_owned(),
                None,
            )),
            cache_root,
        })
    }

    fn temporary(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "magnitude-acquisition-{label}-{}",
            std::process::id()
        ))
    }

    #[tokio::test]
    async fn an_admitted_install_is_immediately_visible_as_a_download() {
        let acquisitions = acquisitions(temporary("visible"));
        let definition = definition("Qwen/Qwen3-8B", "eim--Qwen--Qwen3-8B--v1");

        let id = acquisitions.begin(&definition);
        let downloads = acquisitions.list();

        // The client waits on the download it was handed. If the record appeared only once the
        // task had scheduled, that wait could observe an empty list and give up.
        assert_eq!(downloads.len(), 1);
        assert_eq!(downloads[0].id, id);
        assert!(matches!(
            downloads[0].bundle,
            ServableModelBundle::Standalone { .. }
        ));
    }

    #[tokio::test]
    async fn the_download_carries_the_bundle_the_catalog_publishes() {
        let acquisitions = acquisitions(temporary("bundle"));
        let definition = definition("Qwen/Qwen3-8B", "eim--Qwen--Qwen3-8B--v1");

        acquisitions.begin(&definition);
        let downloads = acquisitions.list();

        // The client matches a download to a model by package identity alone. A different
        // package id here and the progress would attach to nothing.
        assert_eq!(
            bundle_package_id(&downloads[0].bundle),
            Some(&definition.package_id),
        );
    }

    #[tokio::test]
    async fn asking_twice_joins_the_attempt_already_running() {
        let acquisitions = acquisitions(temporary("idempotent"));
        let definition = definition("Qwen/Qwen3-8B", "eim--Qwen--Qwen3-8B--v1");

        let first = acquisitions.begin(&definition);
        let second = acquisitions.begin(&definition);

        // The client re-reconciles whenever its projection has not caught up. A second build
        // racing the first would fight over the same image tag.
        assert_eq!(first, second);
        assert_eq!(acquisitions.list().len(), 1);
    }

    #[tokio::test]
    async fn two_models_install_independently() {
        let acquisitions = acquisitions(temporary("independent"));

        let first = acquisitions.begin(&definition("Qwen/Qwen3-8B", "eim--Qwen--Qwen3-8B--v1"));
        let second = acquisitions.begin(&definition("Qwen/Qwen3-4B", "eim--Qwen--Qwen3-4B--v1"));

        assert_ne!(first, second);
        assert_eq!(acquisitions.list().len(), 2);
    }

    #[tokio::test]
    async fn cancelling_is_recorded_before_the_task_notices() {
        let acquisitions = acquisitions(temporary("cancel"));
        let definition = definition("Qwen/Qwen3-8B", "eim--Qwen--Qwen3-8B--v1");
        let id = acquisitions.begin(&definition);

        let cancelled = acquisitions.cancel(&id).expect("cancel");

        assert!(matches!(
            cancelled.state,
            ModelDownloadState::Cancelled { .. }
        ));
    }

    #[tokio::test]
    async fn cancelling_an_unknown_download_is_not_found() {
        let acquisitions = acquisitions(temporary("unknown"));

        let error = acquisitions
            .cancel(&ModelDownloadId("nope".to_owned()))
            .expect_err("unknown");

        assert!(matches!(error, InventoryError::NotFound(_)));
    }

    #[tokio::test]
    async fn a_failure_can_be_acknowledged_once_it_has_happened() {
        let acquisitions = acquisitions(temporary("acknowledge"));
        let definition = definition("Qwen/Qwen3-8B", "eim--Qwen--Qwen3-8B--v1");
        let id = acquisitions.begin(&definition);

        // The resolver is `PresentOnly` against a nonexistent docker command, so the attempt
        // fails without touching the network.
        let failure = loop {
            let downloads = acquisitions.list();
            let state = downloads[0].state.clone();
            if matches!(state, ModelDownloadState::Failed { .. }) {
                break state;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        };
        assert!(matches!(
            failure,
            ModelDownloadState::Failed {
                acknowledged: false,
                ..
            }
        ));

        let acknowledged = acquisitions.acknowledge_failure(&id).expect("acknowledge");

        assert!(matches!(
            acknowledged.state,
            ModelDownloadState::Failed {
                acknowledged: true,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn an_unreachable_daemon_reports_what_docker_said() {
        let acquisitions = acquisitions(temporary("daemon"));
        let definition = definition("Qwen/Qwen3-8B", "eim--Qwen--Qwen3-8B--v1");
        acquisitions.begin(&definition);

        let failure = loop {
            let state = acquisitions.list()[0].state.clone();
            if let ModelDownloadState::Failed { failure, .. } = state {
                break failure;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        };

        // A broken container runtime is not the same problem as a model that cannot be obtained,
        // and flattening the two would send the operator looking in the wrong place. The message
        // is the only thing that distinguishes them, so it is carried verbatim.
        match failure {
            DownloadFailure::Internal { message } => {
                assert!(message.contains("docker"), "unexpected message: {message}");
            }
            other => panic!("expected the daemon failure to be reported, got {other:?}"),
        }
    }

    #[test]
    fn an_image_with_nowhere_to_come_from_is_an_unavailable_source() {
        // The pre-provisioned deployment shape: the operator has to push the image. Not a
        // network problem, and it must not read as one.
        assert_eq!(
            AcquisitionFailure::Image(ImageError::Absent {
                image: "magnitude-eim-xeon-qwen-qwen3-8b:v1".to_owned(),
            })
            .into_download_failure(),
            DownloadFailure::SourceUnavailable,
        );
        assert_eq!(
            AcquisitionFailure::Image(ImageError::MissingSource {
                path: PathBuf::from("/nonexistent/eim"),
            })
            .into_download_failure(),
            DownloadFailure::SourceUnavailable,
        );
    }

    #[test]
    fn a_gated_repository_is_an_unavailable_source_not_a_corrupt_download() {
        assert_eq!(
            AcquisitionFailure::Weights(WeightError::Gated {
                repository: "meta-llama/Llama-3.1-8B-Instruct".to_owned(),
            })
            .into_download_failure(),
            DownloadFailure::SourceUnavailable,
        );
    }

    #[test]
    fn a_checksum_mismatch_is_reported_as_corruption() {
        // The one failure worth retrying automatically, and the one that must never be published
        // as a complete install.
        assert_eq!(
            AcquisitionFailure::Weights(WeightError::Corrupt {
                path: "model-00001-of-00005.safetensors".to_owned(),
                expected: "a".repeat(64),
                actual: "b".repeat(64),
            })
            .into_download_failure(),
            DownloadFailure::CorruptDownload,
        );
    }

    #[test]
    fn a_full_disk_carries_both_byte_figures() {
        assert_eq!(
            AcquisitionFailure::Weights(WeightError::InsufficientSpace {
                required_bytes: 18_040_000_000,
                available_bytes: 4_000_000_000,
            })
            .into_download_failure(),
            DownloadFailure::InsufficientDiskSpace {
                required_bytes: 18_040_000_000,
                available_bytes: 4_000_000_000,
            },
        );
    }

    #[tokio::test]
    async fn refusing_to_install_names_the_model() {
        let error = NoAcquisition
            .begin(&definition("Qwen/Qwen3-8B", "eim--Qwen--Qwen3-8B--v1"))
            .await
            .expect_err("unsupported");

        match error {
            InventoryError::Unsupported(message) => assert!(message.contains("Qwen/Qwen3-8B")),
            other => panic!("expected an explicit refusal, got {other:?}"),
        }
    }
}
