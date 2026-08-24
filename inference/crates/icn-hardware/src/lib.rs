//! Host memory policy and CPU-only hardware discovery for the EIM container backend.
//!
//! The llama.cpp `common/fit` planning path this crate used to wrap is gone. Model fit is
//! now estimated from catalog geometry in `icn-eim`, and inference runs in a vLLM container
//! that ICN never shares an address space with. What remains here is the part that was never
//! native: the one authoritative system-memory policy, and enough host discovery to build a
//! `HardwareSnapshot` describing a CPU with system RAM.
//!
//! There is deliberately no device enumeration. A container-backed engine exposes no devices
//! to ICN, and inventing them would put fiction into the memory topology that assessments
//! are validated against.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use icn_contracts::{
    HardwareDevice, HardwareDeviceId, HardwareDeviceKind, HardwareMemoryDomain,
    HardwareMemoryDomainKind, HardwareSnapshot, HardwareSystemMemory, MemoryDomainId,
};
use sha2::{Digest, Sha256};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

const GIB: u64 = 1024 * 1024 * 1024;

/// Identity of the single device ICN reports for a container-backed engine.
pub const XEON_VLLM_BACKEND: &str = "xeon-vllm-cpu";

/// The system-memory policy shared by model assessment, load admission, and runtime supervision.
///
/// Each reserve is the larger of a fraction of physical memory and an absolute floor. Keeping
/// these values in the hardware layer gives every caller one authoritative policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMemoryThresholds {
    pub assess_reserve_bytes: u64,
    pub abort_reserve_bytes: u64,
}

#[must_use]
pub fn system_memory_thresholds(total_bytes: u64) -> SystemMemoryThresholds {
    SystemMemoryThresholds {
        assess_reserve_bytes: (total_bytes / 10).max(2 * GIB),
        abort_reserve_bytes: (total_bytes / 20).max(GIB),
    }
}

/// Cross-platform system-memory facts used by inference allocation.
///
/// Platform-specific constraints are collapsed into the binding allocation capacity and
/// headroom. Callers never need to know which operating-system mechanism supplied the limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMemoryObservation {
    pub physical_capacity_bytes: u64,
    pub physical_available_bytes: u64,
    pub allocation_capacity_bytes: u64,
    pub allocation_headroom_bytes: u64,
}

pub fn observe_system_memory() -> Result<SystemMemoryObservation, String> {
    let mut system = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );
    system.refresh_memory_specifics(MemoryRefreshKind::everything());
    normalize_system_memory(system.total_memory(), system.available_memory())
}

/// Normalizes an already-sampled physical-memory observation into ICN's allocation contract.
///
/// Long-lived observers can reuse their platform sampler while this function keeps operating-
/// system allocation constraints private to the hardware layer.
pub fn normalize_system_memory(
    physical_capacity_bytes: u64,
    physical_available_bytes: u64,
) -> Result<SystemMemoryObservation, String> {
    if physical_capacity_bytes == 0 || physical_available_bytes > physical_capacity_bytes {
        return Err(format!(
            "invalid system memory observation: total={physical_capacity_bytes}, available={physical_available_bytes}"
        ));
    }
    let platform_limit = platform_allocation_limit()?;
    Ok(SystemMemoryObservation {
        physical_capacity_bytes,
        physical_available_bytes,
        allocation_capacity_bytes: platform_limit.map_or(physical_capacity_bytes, |limit| {
            physical_capacity_bytes.min(limit.capacity)
        }),
        allocation_headroom_bytes: platform_limit.map_or(physical_available_bytes, |limit| {
            physical_available_bytes.min(limit.headroom)
        }),
    })
}

#[derive(Clone, Copy)]
struct PlatformAllocationLimit {
    capacity: u64,
    headroom: u64,
}

#[cfg(not(windows))]
fn platform_allocation_limit() -> Result<Option<PlatformAllocationLimit>, String> {
    Ok(None)
}

#[cfg(windows)]
fn platform_allocation_limit() -> Result<Option<PlatformAllocationLimit>, String> {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::System::ProcessStatus::{GetPerformanceInfo, PERFORMANCE_INFORMATION};

    // SAFETY: the structure is initialized to its documented size and remains valid for the call.
    unsafe {
        let mut performance: PERFORMANCE_INFORMATION = zeroed();
        performance.cb = size_of::<PERFORMANCE_INFORMATION>() as u32;
        if GetPerformanceInfo(&mut performance, performance.cb) == 0 {
            return Err(format!(
                "GetPerformanceInfo failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        let page_size = performance.PageSize as u64;
        let capacity = (performance.CommitLimit as u64).saturating_mul(page_size);
        let allocated = (performance.CommitTotal as u64).saturating_mul(page_size);
        Ok(Some(PlatformAllocationLimit {
            capacity,
            headroom: capacity.saturating_sub(allocated),
        }))
    }
}

/// Stable ICN assessment-capacity policy. It intentionally uses total capacity,
/// not volatile process-external free memory or application warning policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct CapacityPolicy {
    pub reserve_bytes_per_domain: u64,
    #[serde(default)]
    pub system_reserve_bytes: Option<u64>,
}

impl CapacityPolicy {
    #[must_use]
    pub fn reserve_for_domain(self, domain: &MemoryDomainId) -> u64 {
        if domain.is_system() {
            self.system_reserve_bytes
                .unwrap_or(self.reserve_bytes_per_domain)
        } else {
            self.reserve_bytes_per_domain
        }
    }
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self {
            reserve_bytes_per_domain: 1536 * 1024 * 1024,
            system_reserve_bytes: None,
        }
    }
}

/// Host CPU and NUMA facts that decide which EIM serving profile is compatible.
///
/// EIM's profile selector rejects a `tensor-parallel-size` greater than the host's NUMA node
/// count, so this is not cosmetic: it is the input that picks `tp1` over `tp2`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostTopology {
    pub cpu_model: Option<String>,
    pub logical_cores: usize,
    pub numa_nodes: usize,
}

impl HostTopology {
    pub fn observe() -> Self {
        let mut system = System::new_with_specifics(RefreshKind::nothing());
        system.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing());
        Self {
            cpu_model: system
                .cpus()
                .first()
                .map(|cpu| cpu.brand().trim().to_owned())
                .filter(|brand| !brand.is_empty()),
            logical_cores: std::thread::available_parallelism().map_or(1, |value| value.get()),
            numa_nodes: observe_numa_nodes(),
        }
    }
}

/// Counts online NUMA nodes. Falls back to one node when the host does not expose the sysfs
/// topology (non-Linux, or a container without `/sys` mounted), because a single node is the
/// conservative answer: it makes the profile selector prefer `tp1`, which always works.
fn observe_numa_nodes() -> usize {
    let Ok(online) = std::fs::read_to_string("/sys/devices/system/node/online") else {
        return 1;
    };
    let count: usize = online
        .trim()
        .split(',')
        .filter_map(|range| {
            let mut bounds = range.splitn(2, '-');
            let start = bounds.next()?.trim().parse::<usize>().ok()?;
            match bounds.next() {
                None => Some(1),
                Some(end) => end.trim().parse::<usize>().ok()?.checked_sub(start)?.checked_add(1),
            }
        })
        .sum();
    count.max(1)
}

/// Describe the host as a single system-memory domain holding one CPU device.
#[must_use]
pub fn discover_hardware(
    policy: CapacityPolicy,
    eim_build: impl Into<String>,
    host: &HostTopology,
) -> HardwareSnapshot {
    let mut system = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );
    system.refresh_memory();
    let total_bytes = system.total_memory();
    let available_bytes = system.available_memory();
    let thresholds = system_memory_thresholds(total_bytes);
    let system_memory = HardwareSystemMemory {
        physical_capacity_bytes: total_bytes,
        physical_available_bytes: available_bytes,
        allocation_capacity_bytes: total_bytes,
        allocation_headroom_bytes: available_bytes,
        assess_reserve_bytes: thresholds.assess_reserve_bytes,
        abort_reserve_bytes: thresholds.abort_reserve_bytes,
    };

    let device = HardwareDevice {
        id: HardwareDeviceId::new(format!("{XEON_VLLM_BACKEND}:0")),
        native_index: 0,
        backend: XEON_VLLM_BACKEND.to_owned(),
        physical_id: None,
        name: host
            .cpu_model
            .clone()
            .unwrap_or_else(|| "CPU".to_owned()),
        description: format!(
            "{} ({} logical cores, {} NUMA node{})",
            host.cpu_model.as_deref().unwrap_or("CPU"),
            host.logical_cores,
            host.numa_nodes,
            if host.numa_nodes == 1 { "" } else { "s" },
        ),
        kind: HardwareDeviceKind::Cpu,
        memory_limit: None,
    };

    let domains = vec![HardwareMemoryDomain {
        id: MemoryDomainId::system(),
        kind: HardwareMemoryDomainKind::System,
        total_capacity_bytes: total_bytes,
        stable_capacity_bytes: total_bytes
            .saturating_sub(policy.reserve_for_domain(&MemoryDomainId::system())),
        current_free_bytes: Some(available_bytes),
        shares_system_memory: true,
        devices: vec![device],
    }];

    HardwareSnapshot {
        captured_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs()),
        platform: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        system_product_name: discover_system_product_name(std::env::consts::OS),
        cpu_model: host.cpu_model.clone(),
        logical_cores: host.logical_cores,
        system_memory,
        native_build: eim_build.into(),
        enabled_backends: vec![XEON_VLLM_BACKEND.to_owned()],
        topology_fingerprint: topology_fingerprint(&domains),
        memory_domains: domains,
    }
}

/// Apply one capacity policy to an inventory observation before constructing its topology.
///
/// Assessment consumers receive only the resulting snapshot/topology; policy never participates
/// in location resolution or accounting.
#[must_use]
pub fn with_capacity_policy(
    mut snapshot: HardwareSnapshot,
    policy: CapacityPolicy,
) -> HardwareSnapshot {
    for domain in &mut snapshot.memory_domains {
        domain.stable_capacity_bytes = domain
            .total_capacity_bytes
            .saturating_sub(policy.reserve_for_domain(&domain.id));
    }
    snapshot.topology_fingerprint = topology_fingerprint(&snapshot.memory_domains);
    snapshot
}

fn topology_fingerprint(domains: &[HardwareMemoryDomain]) -> String {
    let topology_material = domains
        .iter()
        .map(|domain| {
            (
                &domain.id,
                &domain.kind,
                domain.total_capacity_bytes,
                domain.stable_capacity_bytes,
                domain.shares_system_memory,
                domain
                    .devices
                    .iter()
                    .map(|device| {
                        (
                            &device.id,
                            device.native_index,
                            &device.backend,
                            &device.physical_id,
                            &device.kind,
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let topology_material = serde_json::to_vec(&topology_material).unwrap_or_default();
    format!("{:x}", Sha256::digest(topology_material))
}

fn discover_system_product_name(platform: &str) -> Option<String> {
    match platform {
        "linux" => [
            "/sys/devices/virtual/dmi/id/product_name",
            "/sys/class/dmi/id/product_name",
            "/proc/device-tree/model",
        ]
        .into_iter()
        .find_map(|path| {
            std::fs::read_to_string(path)
                .ok()
                .map(|value| value.trim_end_matches('\0').trim().to_owned())
                .filter(|value| !value.is_empty())
        }),
        _ => None,
    }
}

/// Unused placeholder kept so callers can key caches by domain without importing BTreeMap.
pub type DomainCapacities = BTreeMap<MemoryDomainId, u64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserves_the_larger_of_a_tenth_and_two_gibibytes() {
        // A 256 GB host: the tenth dominates.
        let large = system_memory_thresholds(256 * 1000 * 1000 * 1000);
        assert_eq!(large.assess_reserve_bytes, 256 * 1000 * 1000 * 1000 / 10);
        assert_eq!(large.abort_reserve_bytes, 256 * 1000 * 1000 * 1000 / 20);

        // An 8 GiB host: the absolute floors dominate.
        let small = system_memory_thresholds(8 * GIB);
        assert_eq!(small.assess_reserve_bytes, 2 * GIB);
        assert_eq!(small.abort_reserve_bytes, GIB);
    }

    #[test]
    fn rejects_an_incoherent_memory_observation() {
        assert!(normalize_system_memory(0, 0).is_err());
        assert!(normalize_system_memory(GIB, 2 * GIB).is_err());
        assert!(normalize_system_memory(2 * GIB, GIB).is_ok());
    }

    #[test]
    fn discovers_a_single_system_domain_with_one_cpu_device() {
        let host = HostTopology {
            cpu_model: Some("Intel(R) Xeon(R) 6767P".to_owned()),
            logical_cores: 256,
            numa_nodes: 2,
        };
        let snapshot = discover_hardware(CapacityPolicy::default(), "eim-test", &host);

        assert_eq!(snapshot.memory_domains.len(), 1);
        assert_eq!(snapshot.memory_domains[0].id, MemoryDomainId::system());
        assert_eq!(snapshot.memory_domains[0].devices.len(), 1);
        assert_eq!(snapshot.enabled_backends, vec![XEON_VLLM_BACKEND.to_owned()]);
        assert_eq!(snapshot.logical_cores, 256);
        assert!(!snapshot.topology_fingerprint.is_empty());
        // The snapshot must be convertible to the topology assessments are validated against.
        assert!(icn_contracts::MemoryTopology::from_snapshot(&snapshot).is_some());
    }

    #[test]
    fn numa_node_count_never_reports_zero() {
        // Whatever the host exposes, a profile selector must never see zero nodes.
        assert!(observe_numa_nodes() >= 1);
    }
}
