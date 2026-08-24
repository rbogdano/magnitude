//! ICN composition root.
//!
//! What this used to be: a 6,700-line file dominated by native machinery — a persistent
//! planning worker pool, hardware calibration microbenchmarks, an instance registry over
//! llama.cpp child processes, and a native model assessor. All of it existed to plan and run
//! inference in ICN's own address space.
//!
//! What it is now: wiring. Inference happens in an Intel EIM container, so ICN's job is to
//! confirm Docker is usable, describe the host, and hand `icn-api` the collaborators it asks
//! for. The HTTP boundary, its OpenAPI surface, and the generated TypeScript protocol are
//! untouched, which is what lets the client stack keep working unchanged.
//!
//! Removing calibration is also why startup is now fast: ACN allows 150 seconds for the ICN
//! process to become ready, and that budget used to be spent running microbenchmarks before
//! the first request could be served.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use icn_api::{AppState, FakeBackend, ServerIdentity, app};
use icn_contracts::bootstrap_protocol::{
    IcnStartupBackend, IcnStartupProgressRecord, IcnStartupProgressRecordType, IcnStartupRecord,
    IcnStartupRecordType,
};
use icn_contracts::{HardwareProvider, HardwareSnapshot, InventoryError};
use icn_eim::catalog::{DockerImageProbe, EimCatalog, EimDownloads};
use icn_eim::controller::{EimControllerConfig, EimModelDefinition, EimModelInstanceController};
use icn_eim::docker::cli::ProxySettings;
use icn_eim::docker::{DockerCli, DockerPreflight};
use icn_hardware::{CapacityPolicy, HostTopology};
use tower_http::trace::{DefaultOnResponse, TraceLayer};

mod build_identity;
mod telemetry;

/// Docker Engine API version that supports every flag the container driver uses.
const MINIMUM_DOCKER_API_VERSION: (u32, u32) = (1, 41);

#[derive(Debug, Parser)]
#[command(
    name = "magnitude-icn",
    version,
    about = "Magnitude inference control node"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
// Clap's flat `serve` command intentionally keeps its complete execution profile visible in
// `--help`; boxing individual flags would only optimize the one-time CLI parse allocation.
#[allow(clippy::large_enum_variant)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        /// Opaque owner-provided identity echoed by the startup and health protocols.
        #[arg(long, default_value = "standalone")]
        instance_id: String,
        /// Exit when the private owning process closes stdin.
        #[arg(long)]
        exit_on_stdin_eof: bool,
        /// Private owner capability. Prefer the environment-backed form used by managed launch.
        #[arg(long, env = "MAGNITUDE_ICN_AUTH_TOKEN", hide_env_values = true)]
        auth_token: Option<String>,
        /// Deterministic in-memory backend used only by protocol tests. Skips the Docker
        /// preflight, so `bun icn:dev` keeps working on a machine with no daemon.
        #[arg(long)]
        fake: bool,
        /// Magnitude-owned model inventory and weight cache root.
        #[arg(long, visible_alias = "models-dir")]
        model_store: Option<PathBuf>,
        /// Magnitude-owned root for all disposable derived cache data.
        #[arg(long)]
        cache_root: Option<PathBuf>,
        /// Additional read-only Hugging Face hub cache roots.
        #[arg(long = "hf-cache", visible_alias = "hf-cache-dir")]
        hf_caches: Vec<PathBuf>,
        /// Verified release or prepared development installation.
        #[arg(long)]
        installation: Option<PathBuf>,
        /// Command used to reach Docker, for sites that need a wrapper such as `sudo -n docker`.
        #[arg(long, env = "MAGNITUDE_DOCKER_COMMAND", default_value = "docker")]
        docker_command: String,
        /// JSON table of servable EIM models. Stage 2 replaces this with the generated catalog
        /// and its geometry overlay; until that overlay exists, supplying the table as data
        /// avoids inventing the layer and head counts the RAM estimate depends on.
        #[arg(long, env = "MAGNITUDE_EIM_CATALOG")]
        eim_catalog: Option<PathBuf>,
    },
    /// Report whether this host can serve models: Docker reachability and host capacity.
    Doctor {
        #[arg(long)]
        json: bool,
    },
    Version {
        #[arg(long)]
        json: bool,
    },
}

/// Describes the host as a single system-memory domain holding one CPU device.
///
/// The snapshot is rebuilt per request rather than cached, because `current_free_bytes` is what
/// the client uses to show live headroom.
struct EimHardware {
    policy: CapacityPolicy,
    eim_build: String,
    host: HostTopology,
}

impl HardwareProvider for EimHardware {
    fn snapshot(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<HardwareSnapshot, InventoryError>> + Send + '_>,
    > {
        Box::pin(async move {
            Ok(icn_hardware::discover_hardware(
                self.policy,
                self.eim_build.clone(),
                &self.host,
            ))
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve {
            bind,
            instance_id,
            exit_on_stdin_eof,
            auth_token,
            fake,
            model_store,
            cache_root,
            hf_caches,
            installation,
            docker_command,
            eim_catalog,
        } => {
            if exit_on_stdin_eof {
                install_parent_stdin_guard();
            }
            let _telemetry = telemetry::init(true)?;
            let eim_build = build_identity::eim_build();

            // Stage 1 accepts the storage and installation paths the client passes but does not
            // yet own an inventory; stage 2 restores icn-models and puts them to work.
            tracing::info!(
                model_store = ?model_store,
                cache_root = ?cache_root,
                hf_caches = ?hf_caches,
                installation = ?installation,
                "storage roots recorded"
            );

            let host = HostTopology::observe();
            tracing::info!(
                cpu_model = ?host.cpu_model,
                logical_cores = host.logical_cores,
                // EIM's profile selector rejects a tensor-parallel size greater than the NUMA
                // node count, so this decides which serving profiles are usable at all.
                numa_nodes = host.numa_nodes,
                "host topology observed"
            );

            let mut controller = None;
            let mut catalog = None;
            let mut downloads = None;
            let mut state = if fake {
                AppState::new(FakeBackend::new("icn-fake", "Hello from ICN."))
            } else {
                let docker = DockerCli::from_command_line(&docker_command);
                let preflight = run_docker_preflight(&docker, &host).await?;
                tracing::info!(
                    server_version = %preflight.server_version,
                    api_version = %preflight.api_version,
                    storage_driver = %preflight.storage_driver,
                    "docker daemon ready"
                );

                let definitions = match &eim_catalog {
                    Some(path) => EimModelDefinition::load_table(path)
                        .map_err(|error| anyhow::anyhow!(error))?,
                    None => Vec::new(),
                };
                tracing::info!(
                    servable_models = definitions.len(),
                    catalog = ?eim_catalog,
                    "EIM model table loaded"
                );

                let eim = Arc::new(EimModelInstanceController::new(
                    docker.clone(),
                    definitions,
                    EimControllerConfig {
                        icn_instance_id: instance_id.clone(),
                        host_cache_path: cache_root
                            .clone()
                            .unwrap_or_else(|| PathBuf::from("/var/lib/magnitude/eim"))
                            .join("eim/model-cache"),
                        proxy: ProxySettings::from_environment(),
                        ..EimControllerConfig::default()
                    },
                    tokio::runtime::Handle::current(),
                ));
                // Before serving, remove containers a killed predecessor left holding memory.
                let reaped = eim.reap_orphans().await;
                if !reaped.is_empty() {
                    tracing::info!(?reaped, "removed orphaned containers");
                }
                // ACN calls GET /v1/models while building its layer graph and fails to become
                // ready if it errors, so the catalog is not optional.
                catalog = Some(Arc::new(EimCatalog::new(
                    eim.definitions(),
                    Arc::new(DockerImageProbe::new(docker)),
                )));
                downloads = Some(Arc::new(EimDownloads));
                controller = Some(eim);
                AppState::model_free()
            }
            .with_hardware(Arc::new(EimHardware {
                policy: CapacityPolicy::default(),
                eim_build: eim_build.clone(),
                host,
            }))
            .with_identity(ServerIdentity {
                instance_id: instance_id.clone(),
                api_version: 1,
                native_build: eim_build.clone(),
            });

            if let Some(catalog) = catalog {
                // The same catalog answers all three model surfaces the client polls, so they
                // can never describe different models.
                state = state
                    .with_catalog_models(catalog.clone())
                    .with_recommendable_catalog(catalog.clone())
                    .with_installed_packages(catalog);
            }
            if let Some(downloads) = downloads {
                state = state.with_model_downloads(downloads);
            }
            if let Some(controller) = controller {
                state = state.with_model_controller(controller);
            }
            if let Some(auth_token) = auth_token {
                state = state.with_authorization(auth_token);
            }

            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .with_context(|| format!("failed to bind {bind}"))?;
            let address = listener
                .local_addr()
                .context("failed to read bound address")?;
            let startup = IcnStartupRecord {
                record_type: IcnStartupRecordType::IcnReady,
                protocol_version: 1,
                origin: format!("http://{address}"),
                instance_id,
                pid: std::process::id(),
                api_version: 1,
                native_build: eim_build,
            };
            // The owner reads this line from stdout to learn the origin and prove identity.
            println!("MAGNITUDE_ICN_READY {}", serde_json::to_string(&startup)?);
            tracing::info!(
                service.name = telemetry::SERVICE_NAME,
                server.address = %address,
                "ICN server ready"
            );

            let app = app(state).layer(
                TraceLayer::new_for_http()
                    .make_span_with(telemetry::http_request_span)
                    .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
            );
            axum::serve(listener, app)
                .with_graceful_shutdown(interrupt_signal())
                .await?;
            tracing::info!("ICN server stopped");
        }
        Command::Doctor { json } => {
            let docker = DockerCli::default();
            let report = doctor_report(&docker).await;
            if json {
                println!("{}", serde_json::to_string(&report)?);
            } else {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            if !report.usable {
                std::process::exit(1);
            }
        }
        Command::Version { json } => {
            let identity = build_identity::identity();
            if json {
                println!("{}", serde_json::to_string(&identity)?);
            } else {
                println!("{}", serde_json::to_string_pretty(&identity)?);
            }
        }
    }

    Ok(())
}

/// Confirms the daemon is reachable and new enough, reporting through the startup progress
/// channel so the client shows "Checking Docker" rather than a silent pause.
///
/// A failure here is fatal: without Docker there is no way to serve a model, and starting
/// anyway would leave the client with an empty model list and no explanation.
async fn run_docker_preflight(
    docker: &DockerCli,
    host: &HostTopology,
) -> anyhow::Result<DockerPreflight> {
    emit_startup_progress(host);
    let preflight = docker
        .preflight()
        .await
        .context("Docker daemon is not reachable; ICN cannot serve models without it")?;

    let api_version = parse_api_version(&preflight.api_version);
    anyhow::ensure!(
        api_version >= Some(MINIMUM_DOCKER_API_VERSION),
        "Docker Engine API {} is older than the required {}.{}",
        preflight.api_version,
        MINIMUM_DOCKER_API_VERSION.0,
        MINIMUM_DOCKER_API_VERSION.1,
    );
    Ok(preflight)
}

/// Tells the owner which hardware is being prepared while the Docker preflight runs.
///
/// The record carries a backend and a hardware label rather than free text, so the label is
/// where the container story goes. `Cpu` is the honest variant: vLLM in the EIM image serves
/// from system memory on Xeon cores, and there is no accelerator for ICN to claim.
fn emit_startup_progress(host: &HostTopology) {
    let record = IcnStartupProgressRecord {
        record_type: IcnStartupProgressRecordType::PreparingBackend,
        backend: IcnStartupBackend::Cpu {
            hardware_label: match &host.cpu_model {
                Some(model) => format!("{model} (vLLM container)"),
                None => "CPU (vLLM container)".to_owned(),
            },
        },
    };
    if let Ok(encoded) = serde_json::to_string(&record) {
        println!("MAGNITUDE_ICN_PROGRESS {encoded}");
    }
}

fn parse_api_version(version: &str) -> Option<(u32, u32)> {
    let (major, minor) = version.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// What `magnitude-icn doctor` reports.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct DoctorReport {
    /// Whether this host can serve a model at all.
    usable: bool,
    docker: Option<DoctorDocker>,
    docker_error: Option<String>,
    cpu_model: Option<String>,
    logical_cores: usize,
    numa_nodes: usize,
    system_memory_bytes: u64,
    /// Serving profiles are limited by NUMA node count, so say which are possible.
    max_tensor_parallel_size: usize,
    eim_build: String,
    eim_base_image: &'static str,
    /// Serving images are named `<prefix>-<org>-<model>`; an operator needs this to know what
    /// to build or pull.
    eim_image_tag_prefix: &'static str,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct DoctorDocker {
    server_version: String,
    api_version: String,
    storage_driver: String,
    docker_root_dir: String,
}

async fn doctor_report(docker: &DockerCli) -> DoctorReport {
    let host = HostTopology::observe();
    let (docker_report, docker_error) = match docker.preflight().await {
        Ok(preflight) => (
            Some(DoctorDocker {
                server_version: preflight.server_version,
                api_version: preflight.api_version,
                storage_driver: preflight.storage_driver,
                docker_root_dir: preflight.docker_root_dir,
            }),
            None,
        ),
        Err(error) => (None, Some(error.to_string())),
    };
    let system_memory_bytes = icn_hardware::observe_system_memory()
        .map(|observation| observation.physical_capacity_bytes)
        .unwrap_or_default();

    DoctorReport {
        usable: docker_report.is_some() && system_memory_bytes > 0,
        docker: docker_report,
        docker_error,
        cpu_model: host.cpu_model.clone(),
        logical_cores: host.logical_cores,
        numa_nodes: host.numa_nodes,
        system_memory_bytes,
        max_tensor_parallel_size: host.numa_nodes,
        eim_build: build_identity::eim_build(),
        eim_base_image: build_identity::EIM_BASE_IMAGE,
        eim_image_tag_prefix: build_identity::EIM_IMAGE_TAG_PREFIX,
    }
}

fn install_parent_stdin_guard() {
    // This thread starts before telemetry initialization. An ordinary ACN shutdown signals ICN
    // first; abrupt owner loss closes the private pipe and must terminate ICN regardless.
    std::thread::spawn(move || {
        use std::io::Read as _;

        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0_u8; 1];
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) | Err(_) => std::process::exit(0),
                Ok(_) => {}
            }
        }
    });
}

#[cfg(unix)]
async fn interrupt_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate()).expect("SIGTERM handler must install");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = terminate.recv() => {},
    }
}

#[cfg(not(unix))]
async fn interrupt_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_docker_api_version() {
        assert_eq!(parse_api_version("1.51"), Some((1, 51)));
        assert_eq!(parse_api_version(" 1.41 "), Some((1, 41)));
        assert_eq!(parse_api_version("nonsense"), None);
        assert_eq!(parse_api_version("1"), None);
    }

    #[test]
    fn the_minimum_api_version_admits_current_daemons_and_rejects_old_ones() {
        assert!(parse_api_version("1.51") >= Some(MINIMUM_DOCKER_API_VERSION));
        assert!(parse_api_version("1.41") >= Some(MINIMUM_DOCKER_API_VERSION));
        assert!(parse_api_version("1.40") < Some(MINIMUM_DOCKER_API_VERSION));
        // An unparseable version must not be treated as acceptable.
        assert!(parse_api_version("unknown") < Some(MINIMUM_DOCKER_API_VERSION));
    }

    #[tokio::test]
    async fn doctor_reports_unusable_when_docker_is_absent() {
        // A binary that cannot be executed stands in for an unreachable daemon.
        let report = doctor_report(&DockerCli::new("magnitude-nonexistent-docker")).await;

        assert!(!report.usable);
        assert!(report.docker.is_none());
        assert!(report.docker_error.is_some());
        // Host facts are still reported, so the failure message can be specific.
        assert!(report.logical_cores >= 1);
        assert!(report.numa_nodes >= 1);
    }

    #[tokio::test]
    async fn the_hardware_provider_describes_one_system_domain() {
        let hardware = EimHardware {
            policy: CapacityPolicy::default(),
            eim_build: "eim_test".to_owned(),
            host: HostTopology::observe(),
        };
        let snapshot = hardware.snapshot().await.expect("a snapshot");

        assert_eq!(snapshot.memory_domains.len(), 1);
        assert_eq!(snapshot.native_build, "eim_test");
        assert_eq!(
            snapshot.enabled_backends,
            vec![icn_hardware::XEON_VLLM_BACKEND.to_owned()]
        );
    }
}
