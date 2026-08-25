//! `docker` CLI invocations.
//!
//! Every call goes through [`DockerCli::invoke`], which records argv, exit status, stdout,
//! stderr and duration into a [`DockerInvocation`]. That record is what gets attached to a
//! failure, because EIM's exit codes are only ever 0 or 1 — the useful signal is in stderr.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::process::Command;

use super::inspect::{ContainerState, ImageSummary};
use super::naming::{
    CONFIGURATION_LABEL, ICN_INSTANCE_LABEL, MODEL_INSTANCE_LABEL, OWNER_LABEL, OWNER_VALUE,
    OwnedContainer, PID_LABEL,
};

/// One completed `docker` process, kept for diagnostics.
#[derive(Clone, Debug)]
pub struct DockerInvocation {
    pub argv: Vec<String>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub duration: Duration,
}

impl DockerInvocation {
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0)
    }

    /// A one-line summary safe to put in a log or a failure message.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "`{}` exited with {} in {}ms: {}",
            self.argv.join(" "),
            self.exit_code
                .map_or_else(|| "signal".to_owned(), |code| code.to_string()),
            self.duration.as_millis(),
            self.stderr.trim().lines().next().unwrap_or("(no stderr)"),
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DockerError {
    #[error("could not execute `{binary}`: {source}")]
    Spawn {
        binary: String,
        #[source]
        source: std::io::Error,
    },
    #[error("docker command failed: {}", invocation.summary())]
    Failed { invocation: Box<DockerInvocation> },
    #[error("could not decode docker output as JSON: {source}")]
    Decode {
        #[source]
        source: serde_json::Error,
        payload: String,
    },
    #[error("docker daemon is not reachable: {0}")]
    DaemonUnreachable(String),
}

/// What the boot-time preflight learned about the daemon.
///
/// This replaces the native hardware calibration step that used to run before ICN could serve.
/// Removing that step is the single largest cut in ICN startup latency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerPreflight {
    pub server_version: String,
    pub api_version: String,
    pub storage_driver: String,
    pub docker_root_dir: String,
    /// Total memory the daemon reports, which can differ from the host when it runs in a VM.
    pub daemon_total_memory_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct RawVersion {
    #[serde(rename = "Server")]
    server: RawVersionServer,
}

#[derive(Debug, Deserialize)]
struct RawVersionServer {
    #[serde(rename = "Version", default)]
    version: String,
    #[serde(rename = "ApiVersion", default)]
    api_version: String,
}

#[derive(Debug, Deserialize)]
struct RawInfo {
    #[serde(rename = "Driver", default)]
    driver: String,
    #[serde(rename = "DockerRootDir", default)]
    docker_root_dir: String,
    #[serde(rename = "MemTotal", default)]
    mem_total: u64,
}

/// Proxy settings the container needs in order to reach Hugging Face.
///
/// On a corporate network the Docker daemon may already be configured with a proxy for image
/// pulls while the container's own environment has none. vLLM downloads model weights from
/// inside the container, so without these it fails on what looks like a missing model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProxySettings {
    pub http_proxy: Option<String>,
    pub https_proxy: Option<String>,
    pub no_proxy: Option<String>,
}

impl ProxySettings {
    /// Reads the conventional environment variables, preferring uppercase.
    #[must_use]
    pub fn from_environment() -> Self {
        let read = |upper: &str, lower: &str| {
            std::env::var(upper)
                .or_else(|_| std::env::var(lower))
                .ok()
                .filter(|value| !value.trim().is_empty())
        };
        Self {
            http_proxy: read("HTTP_PROXY", "http_proxy"),
            https_proxy: read("HTTPS_PROXY", "https_proxy"),
            no_proxy: read("NO_PROXY", "no_proxy"),
        }
    }

    /// Renders the settings as container environment variables, in both cases, because
    /// different Python libraries read different spellings.
    #[must_use]
    pub fn to_environment(&self) -> BTreeMap<String, String> {
        let mut environment = BTreeMap::new();
        let mut insert = |name: &str, value: &Option<String>| {
            if let Some(value) = value {
                environment.insert(name.to_ascii_uppercase(), value.clone());
                environment.insert(name.to_ascii_lowercase(), value.clone());
            }
        };
        insert("http_proxy", &self.http_proxy);
        insert("https_proxy", &self.https_proxy);
        insert("no_proxy", &self.no_proxy);
        environment
    }

    #[must_use]
    pub fn is_configured(&self) -> bool {
        self.http_proxy.is_some() || self.https_proxy.is_some()
    }
}

/// A `docker run` request, kept as data so argv construction stays testable without a daemon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerSpec {
    pub name: String,
    pub image: String,
    /// Host port published on loopback only. The served API has no auth and no TLS, so it must
    /// never be reachable off-box.
    pub host_port: u16,
    /// Always 8000: the baked `HEALTHCHECK` in the EIM base image hardcodes `localhost:8000`,
    /// so changing the in-container port would make the container permanently unhealthy.
    pub container_port: u16,
    pub labels: Vec<String>,
    pub environment: BTreeMap<String, String>,
    /// `host:container:mode` mounts, typically the weight cache read-only.
    pub mounts: Vec<String>,
    /// Hard memory ceiling, so a bad estimate OOM-kills the container instead of the host.
    pub memory_limit_bytes: Option<u64>,
    /// Seconds `docker stop` waits before `SIGKILL`. vLLM is PID 1 via `os.execv`.
    pub stop_timeout_seconds: u32,
    /// Shared memory for the container, in bytes.
    ///
    /// Docker defaults to 64 MB, which is far too little for tensor parallelism: vLLM's workers
    /// broadcast through shared memory and `shm_broadcast` fails to start, surfacing only as a
    /// gloo "connection closed by peer" from the surviving worker. Observed on a real Xeon with
    /// `tensor-parallel-size: 2`.
    pub shm_size_bytes: Option<u64>,
    /// Linux capabilities to add.
    ///
    /// `SYS_NICE` is what lets vLLM bind worker memory to a NUMA node; without it every worker
    /// logs `numa_migrate_pages failed. errno: 1` and runs unbound, which on a multi-socket host
    /// costs the cross-socket bandwidth the tensor-parallel split existed to avoid.
    pub capabilities: Vec<String>,
    /// Command arguments. Empty means EIM's default `serve`.
    pub command: Vec<String>,
}

impl ContainerSpec {
    /// Builds the full argv for `docker run`.
    #[must_use]
    pub fn to_run_args(&self) -> Vec<String> {
        let mut args = vec![
            "run".to_owned(),
            "--detach".to_owned(),
            "--name".to_owned(),
            self.name.clone(),
        ];
        args.extend(self.labels.iter().cloned());
        args.push("--publish".to_owned());
        args.push(format!(
            "127.0.0.1:{}:{}",
            self.host_port, self.container_port
        ));
        if let Some(limit) = self.memory_limit_bytes {
            args.push("--memory".to_owned());
            args.push(limit.to_string());
            // Without this the container swaps instead of failing, which turns a fast, legible
            // OOM into an unbounded slowdown.
            args.push("--memory-swap".to_owned());
            args.push(limit.to_string());
        }
        args.push("--stop-timeout".to_owned());
        args.push(self.stop_timeout_seconds.to_string());
        if let Some(shm_size) = self.shm_size_bytes {
            args.push("--shm-size".to_owned());
            args.push(shm_size.to_string());
        }
        for capability in &self.capabilities {
            args.push("--cap-add".to_owned());
            args.push(capability.clone());
        }
        for mount in &self.mounts {
            args.push("--volume".to_owned());
            args.push(mount.clone());
        }
        for (name, value) in &self.environment {
            args.push("--env".to_owned());
            args.push(format!("{name}={value}"));
        }
        args.push(self.image.clone());
        args.extend(self.command.iter().cloned());
        args
    }
}

/// Runs `docker` subcommands.
#[derive(Clone, Debug)]
pub struct DockerCli {
    /// Usually `docker`; overridable so a site can point at `podman` or a sudo wrapper.
    program: String,
    leading_args: Vec<String>,
}

impl Default for DockerCli {
    fn default() -> Self {
        Self::new("docker")
    }
}

impl DockerCli {
    #[must_use]
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            leading_args: Vec::new(),
        }
    }

    /// Parses a command string such as `sudo -n docker` into a program plus leading arguments.
    #[must_use]
    pub fn from_command_line(command: &str) -> Self {
        let mut parts = command.split_whitespace().map(str::to_owned);
        match parts.next() {
            Some(program) => Self {
                program,
                leading_args: parts.collect(),
            },
            None => Self::default(),
        }
    }

    fn argv(&self, args: &[String]) -> Vec<String> {
        let mut argv = vec![self.program.clone()];
        argv.extend(self.leading_args.iter().cloned());
        argv.extend(args.iter().cloned());
        argv
    }

    /// Executes one `docker` subcommand and captures everything about it.
    pub async fn invoke(&self, args: &[String]) -> Result<DockerInvocation, DockerError> {
        let argv = self.argv(args);
        let started = Instant::now();
        let output = Command::new(&self.program)
            .args(self.leading_args.iter().chain(args.iter()))
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|source| DockerError::Spawn {
                binary: self.program.clone(),
                source,
            })?;

        Ok(DockerInvocation {
            argv,
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            duration: started.elapsed(),
        })
    }

    /// Executes a subcommand and fails unless it exited zero.
    async fn invoke_ok(&self, args: &[String]) -> Result<DockerInvocation, DockerError> {
        let invocation = self.invoke(args).await?;
        if invocation.succeeded() {
            Ok(invocation)
        } else {
            Err(DockerError::Failed {
                invocation: Box::new(invocation),
            })
        }
    }

    /// Confirms the daemon is reachable and reports what it is.
    pub async fn preflight(&self) -> Result<DockerPreflight, DockerError> {
        let version = self
            .invoke(&[
                "version".to_owned(),
                "--format".to_owned(),
                "{{json .}}".to_owned(),
            ])
            .await?;
        if !version.succeeded() {
            // A client-only failure here means the daemon is down or unreachable, which is the
            // one startup condition the client UI must be able to explain.
            return Err(DockerError::DaemonUnreachable(
                version
                    .stderr
                    .trim()
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_owned(),
            ));
        }
        let parsed: RawVersion =
            serde_json::from_str(version.stdout.trim()).map_err(|source| DockerError::Decode {
                source,
                payload: version.stdout.clone(),
            })?;

        let info = self
            .invoke_ok(&[
                "info".to_owned(),
                "--format".to_owned(),
                "{{json .}}".to_owned(),
            ])
            .await?;
        let info: RawInfo =
            serde_json::from_str(info.stdout.trim()).map_err(|source| DockerError::Decode {
                source,
                payload: info.stdout.clone(),
            })?;

        Ok(DockerPreflight {
            server_version: parsed.server.version,
            api_version: parsed.server.api_version,
            storage_driver: info.driver,
            docker_root_dir: info.docker_root_dir,
            daemon_total_memory_bytes: info.mem_total,
        })
    }

    /// Starts a container and returns its id.
    pub async fn run(&self, spec: &ContainerSpec) -> Result<String, DockerError> {
        let invocation = self.invoke_ok(&spec.to_run_args()).await?;
        Ok(invocation.stdout.trim().to_owned())
    }

    /// Reads `.State`. Returns `None` when the container no longer exists.
    pub async fn container_state(&self, name: &str) -> Result<Option<ContainerState>, DockerError> {
        let invocation = self
            .invoke(&[
                "inspect".to_owned(),
                "--format".to_owned(),
                "{{json .State}}".to_owned(),
                name.to_owned(),
            ])
            .await?;
        if !invocation.succeeded() {
            return Ok(None);
        }
        ContainerState::from_json(&invocation.stdout)
            .map(Some)
            .map_err(|source| DockerError::Decode {
                source,
                payload: invocation.stdout,
            })
    }

    /// Reads an image summary. Returns `None` when the image is not present locally.
    pub async fn image_summary(&self, image: &str) -> Result<Option<ImageSummary>, DockerError> {
        let invocation = self
            .invoke(&[
                "image".to_owned(),
                "inspect".to_owned(),
                "--format".to_owned(),
                "{{json .}}".to_owned(),
                image.to_owned(),
            ])
            .await?;
        if !invocation.succeeded() {
            return Ok(None);
        }
        // `docker image inspect` emits an array unless a format is given; with `{{json .}}` it
        // emits one object per image, newline separated. Take the first.
        let first = invocation.stdout.trim().lines().next().unwrap_or_default();
        ImageSummary::from_json(first)
            .map(Some)
            .map_err(|source| DockerError::Decode {
                source,
                payload: invocation.stdout,
            })
    }

    /// Tail of the container log, for attaching to a failure.
    pub async fn logs_tail(&self, name: &str, lines: u32) -> Result<String, DockerError> {
        let invocation = self
            .invoke(&[
                "logs".to_owned(),
                "--tail".to_owned(),
                lines.to_string(),
                name.to_owned(),
            ])
            .await?;
        // vLLM writes to both streams; the interesting failure text is usually on stderr.
        Ok(format!("{}{}", invocation.stdout, invocation.stderr))
    }

    /// Stops a container, giving vLLM time to shut down before `SIGKILL`.
    pub async fn stop(&self, name: &str, timeout_seconds: u32) -> Result<(), DockerError> {
        let invocation = self
            .invoke(&[
                "stop".to_owned(),
                "--timeout".to_owned(),
                timeout_seconds.to_string(),
                name.to_owned(),
            ])
            .await?;
        // A container that is already gone is a success for our purposes.
        if invocation.succeeded() || invocation.stderr.contains("No such container") {
            Ok(())
        } else {
            Err(DockerError::Failed {
                invocation: Box::new(invocation),
            })
        }
    }

    /// Removes a container, forcing if it is still running.
    pub async fn remove(&self, name: &str) -> Result<(), DockerError> {
        let invocation = self
            .invoke(&["rm".to_owned(), "--force".to_owned(), name.to_owned()])
            .await?;
        if invocation.succeeded() || invocation.stderr.contains("No such container") {
            Ok(())
        } else {
            Err(DockerError::Failed {
                invocation: Box::new(invocation),
            })
        }
    }

    /// Lists every ICN-owned container, running or not, for orphan reaping.
    pub async fn list_owned(&self) -> Result<Vec<OwnedContainer>, DockerError> {
        let invocation = self
            .invoke_ok(&[
                "ps".to_owned(),
                "--all".to_owned(),
                "--filter".to_owned(),
                format!("label={OWNER_LABEL}={OWNER_VALUE}"),
                "--format".to_owned(),
                "{{json .}}".to_owned(),
            ])
            .await?;

        let mut owned = Vec::new();
        for line in invocation
            .stdout
            .lines()
            .filter(|line| !line.trim().is_empty())
        {
            #[derive(Deserialize)]
            struct RawPs {
                #[serde(rename = "ID", default)]
                id: String,
                #[serde(rename = "Names", default)]
                names: String,
                #[serde(rename = "Labels", default)]
                labels: String,
                #[serde(rename = "State", default)]
                state: String,
                #[serde(rename = "Ports", default)]
                ports: String,
            }
            let raw: RawPs = serde_json::from_str(line).map_err(|source| DockerError::Decode {
                source,
                payload: line.to_owned(),
            })?;
            let labels = parse_ps_labels(&raw.labels);
            owned.push(OwnedContainer {
                id: raw.id,
                name: raw.names,
                icn_instance_id: labels.get(ICN_INSTANCE_LABEL).cloned(),
                pid: labels.get(PID_LABEL).and_then(|pid| pid.parse().ok()),
                configuration_id: labels.get(CONFIGURATION_LABEL).cloned(),
                model_instance_id: labels.get(MODEL_INSTANCE_LABEL).cloned(),
                host_port: parse_published_port(&raw.ports),
                running: raw.state == "running",
            });
        }
        Ok(owned)
    }
}

/// Reads the loopback port from `docker ps`'s `Ports` column.
///
/// Rendered as `127.0.0.1:41234->8000/tcp`, possibly several comma-separated entries. Only the
/// mapping to the container's own port matters, since that is the served API.
fn parse_published_port(ports: &str) -> Option<u16> {
    ports.split(',').find_map(|entry| {
        let entry = entry.trim();
        let (published, target) = entry.split_once("->")?;
        if !target.starts_with(&crate::env_contract::CONTAINER_PORT.to_string()) {
            return None;
        }
        published.rsplit(':').next()?.parse().ok()
    })
}

/// `docker ps --format '{{json .}}'` renders labels as a comma-separated `key=value` string.
fn parse_ps_labels(labels: &str) -> BTreeMap<String, String> {
    labels
        .split(',')
        .filter_map(|entry| {
            let (key, value) = entry.split_once('=')?;
            Some((key.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ContainerSpec {
        ContainerSpec {
            name: "magnitude-eim-icn1-mi1".to_owned(),
            image: "magnitude-eim-xeon-qwen-qwen3-8b:v1".to_owned(),
            host_port: 43_117,
            container_port: 8000,
            labels: vec!["--label".to_owned(), "dev.magnitude.owner=icn".to_owned()],
            environment: BTreeMap::from([
                ("INFERENCE_MODEL_ID".to_owned(), "Qwen/Qwen3-8B".to_owned()),
                (
                    "INFERENCE_PROFILE_ID".to_owned(),
                    "vllm-xeon-bf16-tp2".to_owned(),
                ),
            ]),
            mounts: vec!["/var/lib/magnitude/eim:/workspace/model-cache:ro".to_owned()],
            memory_limit_bytes: Some(83_000_000_000),
            stop_timeout_seconds: 30,
            shm_size_bytes: Some(16 * 1024 * 1024 * 1024),
            capabilities: vec!["SYS_NICE".to_owned()],
            command: Vec::new(),
        }
    }

    #[test]
    fn publishes_the_host_port_on_loopback_only() {
        let args = spec().to_run_args();
        let published = args
            .iter()
            .position(|arg| arg == "--publish")
            .map(|index| args[index + 1].clone())
            .expect("a published port");

        // The served API has no auth and no TLS; binding 0.0.0.0 would expose it off-box.
        assert_eq!(published, "127.0.0.1:43117:8000");
    }

    #[test]
    fn keeps_the_container_port_at_8000() {
        // The base image's HEALTHCHECK hardcodes localhost:8000.
        assert!(
            spec()
                .to_run_args()
                .contains(&"127.0.0.1:43117:8000".to_owned())
        );
    }

    #[test]
    fn forbids_swap_whenever_a_memory_limit_is_set() {
        let args = spec().to_run_args();
        let limit = 83_000_000_000_u64.to_string();

        let memory = args
            .iter()
            .position(|arg| arg == "--memory")
            .expect("limit");
        let swap = args
            .iter()
            .position(|arg| arg == "--memory-swap")
            .expect("swap limit");
        assert_eq!(args[memory + 1], limit);
        // Equal values disable swap, turning a bad estimate into a fast OOM instead of thrash.
        assert_eq!(args[swap + 1], limit);
    }

    #[test]
    fn omits_memory_flags_when_no_limit_is_configured() {
        let unlimited = ContainerSpec {
            memory_limit_bytes: None,
            ..spec()
        };
        let args = unlimited.to_run_args();

        assert!(!args.contains(&"--memory".to_owned()));
        assert!(!args.contains(&"--memory-swap".to_owned()));
    }

    #[test]
    fn reads_the_published_loopback_port_docker_reports() {
        // What `docker ps` renders for a container this controller started. Reading it back is
        // what lets a successor reach a model it did not start itself.
        assert_eq!(
            parse_published_port("127.0.0.1:40767->8000/tcp"),
            Some(40_767)
        );
    }

    #[test]
    fn ignores_a_mapping_to_a_port_the_engine_does_not_serve() {
        // The served API is always on the container's own port; anything else is not it.
        assert_eq!(parse_published_port("127.0.0.1:40767->9000/tcp"), None);
        assert_eq!(
            parse_published_port("127.0.0.1:5000->9000/tcp, 127.0.0.1:40767->8000/tcp"),
            Some(40_767)
        );
    }

    #[test]
    fn an_unpublished_container_reports_no_port() {
        // Nothing to adopt: without a published port the server cannot be reached at all.
        assert_eq!(parse_published_port(""), None);
        assert_eq!(parse_published_port("8000/tcp"), None);
    }

    #[test]
    fn renders_environment_and_mounts_as_separate_flags() {
        let args = spec().to_run_args();

        assert!(args.contains(&"INFERENCE_MODEL_ID=Qwen/Qwen3-8B".to_owned()));
        assert!(args.contains(&"INFERENCE_PROFILE_ID=vllm-xeon-bf16-tp2".to_owned()));
        assert!(args.contains(&"/var/lib/magnitude/eim:/workspace/model-cache:ro".to_owned()));
        assert_eq!(args.iter().filter(|arg| *arg == "--env").count(), 2);
        assert_eq!(args.iter().filter(|arg| *arg == "--volume").count(), 1);
    }

    #[test]
    fn puts_the_image_last_so_no_flag_is_mistaken_for_it() {
        let args = spec().to_run_args();

        assert_eq!(
            args.last().map(String::as_str),
            Some("magnitude-eim-xeon-qwen-qwen3-8b:v1")
        );
        assert_eq!(args.first().map(String::as_str), Some("run"));
    }

    #[test]
    fn appends_a_command_after_the_image() {
        let dry_run = ContainerSpec {
            command: vec![
                "dry-run".to_owned(),
                "--format".to_owned(),
                "json".to_owned(),
            ],
            ..spec()
        };
        let args = dry_run.to_run_args();
        let image = args
            .iter()
            .position(|arg| arg == "magnitude-eim-xeon-qwen-qwen3-8b:v1")
            .expect("image");

        assert_eq!(&args[image + 1..], &["dry-run", "--format", "json"]);
    }

    #[test]
    fn raises_shared_memory_above_the_docker_default() {
        let args = spec().to_run_args();
        let shm = args
            .iter()
            .position(|arg| arg == "--shm-size")
            .map(|index| args[index + 1].parse::<u64>().expect("a byte count"))
            .expect("a shared memory size");

        // Docker's 64 MB default makes vLLM's tensor-parallel shm_broadcast fail to start, and
        // the only symptom is a gloo "connection closed by peer" from the surviving worker.
        assert!(shm > 64 * 1024 * 1024, "{shm}");
    }

    #[test]
    fn adds_the_capability_numa_binding_needs() {
        let args = spec().to_run_args();
        let index = args
            .iter()
            .position(|arg| arg == "--cap-add")
            .expect("a capability");

        // Without SYS_NICE every worker logs `numa_migrate_pages failed. errno: 1` and runs
        // unbound, losing the locality the tensor-parallel split was for.
        assert_eq!(args[index + 1], "SYS_NICE");
    }

    #[test]
    fn omits_shared_memory_and_capabilities_when_unset() {
        let plain = ContainerSpec {
            shm_size_bytes: None,
            capabilities: Vec::new(),
            ..spec()
        };
        let args = plain.to_run_args();

        assert!(!args.contains(&"--shm-size".to_owned()));
        assert!(!args.contains(&"--cap-add".to_owned()));
    }

    #[test]
    fn supports_a_wrapped_docker_command_line() {
        let cli = DockerCli::from_command_line("sudo -n docker");

        assert_eq!(
            cli.argv(&["ps".to_owned()]),
            vec!["sudo", "-n", "docker", "ps"]
        );
    }

    #[test]
    fn parses_the_comma_separated_label_string_docker_ps_emits() {
        let labels = parse_ps_labels(
            "dev.magnitude.owner=icn,dev.magnitude.pid=4242,dev.magnitude.icn-instance-id=icn-1",
        );

        assert_eq!(labels.get(OWNER_LABEL).map(String::as_str), Some("icn"));
        assert_eq!(labels.get(PID_LABEL).map(String::as_str), Some("4242"));
        assert_eq!(
            labels.get(ICN_INSTANCE_LABEL).map(String::as_str),
            Some("icn-1")
        );
    }

    #[test]
    fn proxy_settings_render_both_letter_cases() {
        let proxy = ProxySettings {
            http_proxy: Some("http://proxy-dmz.intel.com:911".to_owned()),
            https_proxy: Some("http://proxy-dmz.intel.com:912".to_owned()),
            no_proxy: Some("localhost,127.0.0.1,.intel.com".to_owned()),
        };
        let environment = proxy.to_environment();

        // Python's requests reads lowercase; some libraries read uppercase. Pass both.
        assert_eq!(environment.len(), 6);
        assert_eq!(
            environment.get("HTTPS_PROXY").map(String::as_str),
            Some("http://proxy-dmz.intel.com:912")
        );
        assert_eq!(
            environment.get("https_proxy").map(String::as_str),
            Some("http://proxy-dmz.intel.com:912")
        );
        assert!(proxy.is_configured());
    }

    #[test]
    fn an_unset_proxy_contributes_no_environment() {
        let proxy = ProxySettings::default();

        assert!(proxy.to_environment().is_empty());
        assert!(!proxy.is_configured());
    }
}
