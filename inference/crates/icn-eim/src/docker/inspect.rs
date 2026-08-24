//! Typed decoding of `docker inspect --format '{{json ...}}'` output.
//!
//! `ContainerState` is what readiness polling checks on every iteration. Without it a crashed
//! container burns the whole multi-minute health timeout before reporting anything useful,
//! which is the failure mode EIM's own harness explicitly guards against.

use serde::Deserialize;

/// Docker's container status vocabulary. Anything unrecognized decodes to `Unknown` rather
/// than failing, so a new Docker status cannot wedge lifecycle handling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContainerStatus {
    Created,
    Running,
    Paused,
    Restarting,
    Removing,
    Exited,
    Dead,
    Unknown(String),
}

impl ContainerStatus {
    #[must_use]
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    /// Whether the container has stopped for good. `Restarting` is deliberately excluded:
    /// EIM containers have no restart policy, so seeing it means someone set one externally.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Exited | Self::Dead | Self::Removing)
    }
}

impl From<&str> for ContainerStatus {
    fn from(value: &str) -> Self {
        match value {
            "created" => Self::Created,
            "running" => Self::Running,
            "paused" => Self::Paused,
            "restarting" => Self::Restarting,
            "removing" => Self::Removing,
            "exited" => Self::Exited,
            "dead" => Self::Dead,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawContainerState {
    #[serde(rename = "Status", default)]
    status: String,
    #[serde(rename = "ExitCode", default)]
    exit_code: i64,
    #[serde(rename = "OOMKilled", default)]
    oom_killed: bool,
    #[serde(rename = "Error", default)]
    error: String,
}

/// The subset of `.State` that lifecycle decisions depend on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerState {
    pub status: ContainerStatus,
    pub exit_code: i64,
    /// Set when the kernel OOM killer took the container. This is how a wrong RAM estimate
    /// surfaces as a legible `LowMemory` failure instead of a mystery exit.
    pub oom_killed: bool,
    pub error: Option<String>,
}

impl ContainerState {
    /// Decodes one `docker inspect --format '{{json .State}}'` payload.
    pub fn from_json(payload: &str) -> Result<Self, serde_json::Error> {
        let raw: RawContainerState = serde_json::from_str(payload.trim())?;
        Ok(Self {
            status: ContainerStatus::from(raw.status.as_str()),
            exit_code: raw.exit_code,
            oom_killed: raw.oom_killed,
            error: Some(raw.error).filter(|error| !error.is_empty()),
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawImageSummary {
    #[serde(rename = "Id", default)]
    id: String,
    #[serde(rename = "Size", default)]
    size: u64,
    #[serde(rename = "RepoDigests", default)]
    repo_digests: Vec<String>,
}

/// What `docker image inspect` tells us about a resolved image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSummary {
    /// The content digest, `sha256:` prefix stripped. Used as the package identity that the
    /// local-model protocol requires to be 64 hex characters.
    pub content_digest: String,
    pub size_bytes: u64,
    pub repo_digests: Vec<String>,
}

impl ImageSummary {
    pub fn from_json(payload: &str) -> Result<Self, serde_json::Error> {
        let raw: RawImageSummary = serde_json::from_str(payload.trim())?;
        Ok(Self {
            content_digest: raw
                .id
                .strip_prefix("sha256:")
                .unwrap_or(&raw.id)
                .to_owned(),
            size_bytes: raw.size,
            repo_digests: raw.repo_digests,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_running_container() {
        let state = ContainerState::from_json(
            r#"{"Status":"running","Running":true,"ExitCode":0,"OOMKilled":false,"Error":""}"#,
        )
        .expect("valid state");

        assert!(state.status.is_running());
        assert!(!state.status.is_terminal());
        assert_eq!(state.error, None);
    }

    #[test]
    fn decodes_an_oom_killed_container() {
        let state = ContainerState::from_json(
            r#"{"Status":"exited","ExitCode":137,"OOMKilled":true,"Error":""}"#,
        )
        .expect("valid state");

        assert!(state.status.is_terminal());
        assert!(state.oom_killed);
        assert_eq!(state.exit_code, 137);
    }

    #[test]
    fn decodes_a_start_failure_with_an_error_message() {
        let state = ContainerState::from_json(
            r#"{"Status":"dead","ExitCode":1,"OOMKilled":false,"Error":"oci runtime error"}"#,
        )
        .expect("valid state");

        assert_eq!(state.error.as_deref(), Some("oci runtime error"));
    }

    #[test]
    fn an_unknown_status_decodes_instead_of_failing() {
        let state = ContainerState::from_json(r#"{"Status":"hibernating"}"#).expect("valid state");

        assert_eq!(
            state.status,
            ContainerStatus::Unknown("hibernating".to_owned())
        );
        assert!(!state.status.is_running());
        assert!(!state.status.is_terminal());
    }

    #[test]
    fn strips_the_sha256_prefix_from_an_image_id() {
        let digest = "d0508be469accfdd25a1a97b5a50ece67db5c5ca23031cbed45fdbf8d0a7bb89";
        let summary = ImageSummary::from_json(&format!(
            r#"{{"Id":"sha256:{digest}","Size":1234,"RepoDigests":["intel/x@sha256:{digest}"]}}"#
        ))
        .expect("valid image");

        // The local-model protocol requires exactly 64 lowercase hex characters here.
        assert_eq!(summary.content_digest.len(), 64);
        assert_eq!(summary.content_digest, digest);
        assert_eq!(summary.size_bytes, 1234);
        assert_eq!(summary.repo_digests.len(), 1);
    }
}
