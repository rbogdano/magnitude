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
}

impl OwnedContainer {
    /// Whether this container belongs to a different, no-longer-running ICN.
    ///
    /// Containers whose recorded pid is still alive are left alone even when the instance id
    /// differs, because a second ICN may legitimately be running alongside this one.
    #[must_use]
    pub fn is_orphan_of(&self, current_instance_id: &str, pid_is_alive: impl Fn(u32) -> bool) -> bool {
        if self.icn_instance_id.as_deref() == Some(current_instance_id) {
            return false;
        }
        match self.pid {
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

    #[test]
    fn our_own_containers_are_never_orphans() {
        let container = OwnedContainer {
            id: "deadbeef".to_owned(),
            name: "magnitude-eim-icn-1-mi-1".to_owned(),
            icn_instance_id: Some("icn-1".to_owned()),
            pid: Some(1),
        };

        assert!(!container.is_orphan_of("icn-1", |_| false));
    }

    #[test]
    fn a_container_from_a_dead_icn_is_an_orphan() {
        let container = OwnedContainer {
            id: "deadbeef".to_owned(),
            name: "magnitude-eim-icn-0-mi-9".to_owned(),
            icn_instance_id: Some("icn-0".to_owned()),
            pid: Some(999_999),
        };

        assert!(container.is_orphan_of("icn-1", |_| false));
        // A second ICN that is still running owns its containers; leave them alone.
        assert!(!container.is_orphan_of("icn-1", |_| true));
    }

    #[test]
    fn a_container_without_a_recorded_pid_is_an_orphan() {
        let container = OwnedContainer {
            id: "deadbeef".to_owned(),
            name: "magnitude-eim-unknown".to_owned(),
            icn_instance_id: None,
            pid: None,
        };

        assert!(container.is_orphan_of("icn-1", |_| true));
    }
}
