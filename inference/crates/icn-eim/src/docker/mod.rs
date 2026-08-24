//! The container driver.
//!
//! Every operation shells out to the `docker` CLI rather than talking to the Engine socket
//! directly. That is a deliberate choice: `DOCKER_HOST`, `docker context`, rootless Docker,
//! Podman-as-docker, and a `sudo -n docker` wrapper all work for free, and there is no Engine
//! API version to negotiate. EIM's own integration harness drives containers the same way.
//!
//! One hard rule: never parse human-readable `docker` output. Every invocation asks for JSON
//! and is decoded with serde, so a cosmetic change to the CLI's tables cannot silently break
//! lifecycle decisions.

pub mod cli;
pub mod inspect;
pub mod naming;

pub use cli::{DockerCli, DockerError, DockerInvocation, DockerPreflight};
pub use inspect::{ContainerState, ContainerStatus, ImageSummary};
pub use naming::{ContainerLabels, OWNER_LABEL, OWNER_VALUE, OwnedContainer};
