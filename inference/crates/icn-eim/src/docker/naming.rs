//! Container naming and the label set that makes orphan reaping possible.
//!
//! A vLLM container can hold tens of gigabytes resident. If ICN is killed without unwinding —
//! and the TypeScript lifecycle does exactly that, `SIGTERM` then `SIGKILL` 500 ms later — an
//! unlabeled container would survive invisibly. Labels let a fresh ICN find and remove
//! containers left behind by a previous instance of itself.

/// Marks a container as ICN-owned. Reaping only ever considers containers carrying this.
pub const OWNER_LABEL: &str = "dev.magnitude.owner";
pub const OWNER_VALUE: &str = "icn";

pub const ICN_INSTANCE_LABEL: &str = "dev.magnitude.icn-instance-id";
pub const MODEL_INSTANCE_LABEL: &str = "dev.magnitude.model-instance-id";
pub const CONFIGURATION_LABEL: &str = "dev.magnitude.configuration-id";
pub const CATALOG_MODEL_LABEL: &str = "dev.magnitude.catalog-model-id";
pub const PROFILE_LABEL: &str = "dev.magnitude.eim-profile-id";
pub const PID_LABEL: &str = "dev.magnitude.pid";

/// Identity written onto every container ICN starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerLabels {
    /// The `--instance-id` this ICN process was started with.
    pub icn_instance_id: String,
    pub model_instance_id: String,
    pub configuration_id: String,
    pub catalog_model_id: String,
    pub eim_profile_id: String,
    /// ICN's own pid, so a reaper can tell a live owner from a dead one.
    pub pid: u32,
}

impl ContainerLabels {
    /// Renders the label set as `--label key=value` arguments.
    #[must_use]
    pub fn to_args(&self) -> Vec<String> {
        [
            (OWNER_LABEL, OWNER_VALUE.to_owned()),
            (ICN_INSTANCE_LABEL, self.icn_instance_id.clone()),
            (MODEL_INSTANCE_LABEL, self.model_instance_id.clone()),
            (CONFIGURATION_LABEL, self.configuration_id.clone()),
            (CATALOG_MODEL_LABEL, self.catalog_model_id.clone()),
            (PROFILE_LABEL, self.eim_profile_id.clone()),
            (PID_LABEL, self.pid.to_string()),
        ]
        .into_iter()
        .flat_map(|(key, value)| ["--label".to_owned(), format!("{key}={value}")])
        .collect()
    }
}

/// Container names are `magnitude-eim-<icn instance>-<model instance>`, sanitized to what
/// Docker accepts: `[a-zA-Z0-9][a-zA-Z0-9_.-]*`.
#[must_use]
pub fn container_name(icn_instance_id: &str, model_instance_id: &str) -> String {
    format!(
        "magnitude-eim-{}-{}",
        sanitize(icn_instance_id),
        sanitize(model_instance_id)
    )
}

fn sanitize(value: &str) -> String {
    let sanitized: String = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
                character
            } else {
                '-'
            }
        })
        .collect();
    // Docker requires the first character to be alphanumeric.
    match sanitized.chars().next() {
        Some(first) if first.is_ascii_alphanumeric() => sanitized,
        _ => format!("x{sanitized}"),
    }
}

/// An ICN-owned container discovered during reaping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedContainer {
    pub id: String,
    pub name: String,
    pub icn_instance_id: Option<String>,
    pub pid: Option<u32>,
    /// Which serving configuration it was started for, so a successor can tell whether the model
    /// it is serving is still one this build knows how to address.
    pub configuration_id: Option<String>,
    /// The instance identity its owner published, carried forward when a successor adopts it.
    pub model_instance_id: Option<String>,
    /// Loopback port the served API is published on, read back so a successor can reach it without
    /// having recorded anything itself.
    pub host_port: Option<u16>,
    /// Whether the container is running right now, as opposed to exited.
    pub running: bool,
}

impl OwnedContainer {
    /// Whether this container's owning process is gone.
    ///
    /// Liveness is the only discriminator, and deliberately so. The instance id must not exempt a
    /// container: ACN launches ICN with a *stable* identity, so a restarted ICN carries its
    /// predecessor's instance id, and treating a matching id as proof of ownership made every
    /// container survive an ICN that was killed rather than shut down. The next ICN then could not
    /// start that model at all, because the container name is derived from the same identity and
    /// collided — a permanent failure needing manual cleanup, reached by the ordinary path of the
    /// client ending a session with `SIGKILL` after its grace period.
    ///
    /// A container recorded against our own pid is the one we are running. A live pid that is not
    /// ours belongs to a second ICN alongside this one and is left alone.
    #[must_use]
    pub fn is_orphan_of(&self, current_pid: u32, pid_is_alive: impl Fn(u32) -> bool) -> bool {
        match self.pid {
            Some(pid) if pid == current_pid => false,
            Some(pid) => !pid_is_alive(pid),
            // No pid recorded means we cannot prove an owner is alive; treat it as an orphan.
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_docker_safe_container_name() {
        assert_eq!(
            container_name("abc123", "inst-7"),
            "magnitude-eim-abc123-inst-7"
        );
        // Slashes and colons are common in identifiers and illegal in container names.
        assert_eq!(
            container_name("a/b:c", "Qwen/Qwen3-8B"),
            "magnitude-eim-a-b-c-Qwen-Qwen3-8B"
        );
        // A leading non-alphanumeric character is prefixed rather than dropped.
        assert!(container_name("-lead", "x").starts_with("magnitude-eim-x-lead"));
    }

    #[test]
    fn renders_every_label_as_a_flag_and_value_pair() {
        let labels = ContainerLabels {
            icn_instance_id: "icn-1".to_owned(),
            model_instance_id: "mi-1".to_owned(),
            configuration_id: "cfg-1".to_owned(),
            catalog_model_id: "qwen-qwen3-8b".to_owned(),
            eim_profile_id: "vllm-xeon-bf16-tp2".to_owned(),
            pid: 4242,
        };
        let args = labels.to_args();

        assert_eq!(args.len(), 14, "seven labels, each a flag plus a value");
        assert_eq!(args.iter().filter(|arg| *arg == "--label").count(), 7);
        assert!(args.contains(&format!("{OWNER_LABEL}={OWNER_VALUE}")));
        assert!(args.contains(&"dev.magnitude.pid=4242".to_owned()));
        assert!(args.contains(&"dev.magnitude.eim-profile-id=vllm-xeon-bf16-tp2".to_owned()));
    }

    /// The pid this ICN would record on a container it starts.
    const OURS: u32 = 4242;

    #[test]
    fn the_container_we_are_running_is_never_an_orphan() {
        let container = OwnedContainer {
            id: "deadbeef".to_owned(),
            name: "magnitude-eim-icn-1-mi-1".to_owned(),
            icn_instance_id: Some("icn-1".to_owned()),
            pid: Some(OURS),
            configuration_id: None,
            model_instance_id: None,
            host_port: None,
            running: true,
        };

        // Even a liveness check that claims nothing is alive must not reach our own container.
        assert!(!container.is_orphan_of(OURS, |_| false));
    }

    #[test]
    fn a_container_from_a_dead_icn_is_an_orphan() {
        let container = OwnedContainer {
            id: "deadbeef".to_owned(),
            name: "magnitude-eim-icn-0-mi-9".to_owned(),
            icn_instance_id: Some("icn-0".to_owned()),
            pid: Some(999_999),
            configuration_id: None,
            model_instance_id: None,
            host_port: None,
            running: true,
        };

        assert!(container.is_orphan_of(OURS, |_| false));
        // A second ICN that is still running owns its containers; leave them alone.
        assert!(!container.is_orphan_of(OURS, |_| true));
    }

    #[test]
    fn a_restarted_icn_reclaims_its_predecessors_containers() {
        // The identity is stable across restarts because ACN supplies it, so the predecessor's
        // container carries *our* instance id with a dead pid. Exempting it on the matching id
        // left the name permanently taken and the model unservable -- reached by the ordinary path
        // of a client killing ICN after its shutdown grace period.
        let container = OwnedContainer {
            id: "deadbeef".to_owned(),
            name: "magnitude-eim-icn-1-mi-1".to_owned(),
            icn_instance_id: Some("icn-1".to_owned()),
            pid: Some(999_999),
            configuration_id: None,
            model_instance_id: None,
            host_port: None,
            running: true,
        };

        assert!(
            container.is_orphan_of(OURS, |_| false),
            "a matching instance id is not proof that an owner is alive"
        );
    }

    #[test]
    fn a_container_without_a_recorded_pid_is_an_orphan() {
        let container = OwnedContainer {
            id: "deadbeef".to_owned(),
            name: "magnitude-eim-unknown".to_owned(),
            icn_instance_id: None,
            pid: None,
            configuration_id: None,
            model_instance_id: None,
            host_port: None,
            running: true,
        };

        assert!(container.is_orphan_of(OURS, |_| true));
    }
}
