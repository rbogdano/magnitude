//! Model fit from catalog geometry.
//!
//! This replaces `icn_hardware::plan_load_with_backend`, which asked llama.cpp's `common/fit`
//! to plan a load against real device memory. A vLLM container cannot be asked that question
//! from outside, so fit is computed instead from six numbers per model, recorded in the catalog
//! overlay at authoring time and read from each model's Hugging Face `config.json`.
//!
//! The estimate is deliberately simple and slightly conservative. On the measured target host
//! (Xeon 6767P, 269.9 GB physical, ~243 GB after the reserve policy) it is informational far
//! more often than it is a gate: only one model in the EIM catalog, Llama-4-Scout, actually
//! fails to fit. Its value is therefore in what it shows the user — "this will use 72 GB of
//! your 243 GB" — which is why the four memory buckets are filled in honestly rather than
//! lumped into `auxiliary_bytes`.

use icn_contracts::{
    HardwareAssessment, HardwareDeficit, HardwareMemory, HardwareMemoryDomainAssessment,
    HardwareProfile, HardwareRecommendation, MemoryDomainId, MemoryTopology,
};

const BF16_BYTES_PER_PARAMETER: u64 = 2;
const GIB: u64 = 1024 * 1024 * 1024;

/// Python, torch, IPEX, and the vLLM process itself, before any model state.
const RUNTIME_BASELINE_BYTES: u64 = 2 * GIB;
/// Floor for worker activations and the compiled graph.
const MINIMUM_ACTIVATION_BYTES: u64 = GIB;
/// Activations scale with model size; a tenth of weight bytes tracks observed vLLM CPU behavior.
const ACTIVATION_FRACTION_OF_WEIGHTS: u64 = 10;
/// A vision tower and image-token KV are not in the text geometry, so charge a flat margin.
const VISION_TOWER_BYTES: u64 = 4 * GIB;

/// vLLM admits at most this many concurrent sequences on a dynamically sized load.
///
/// Mirrors `MAX_DYNAMIC_PARALLEL_SEQUENCES` from the native controller so that switching
/// backends does not silently change how much context a user gets.
pub const MAX_DYNAMIC_PARALLEL_SEQUENCES: u32 = 4;

/// The per-model numbers the RAM formula needs, all from Hugging Face `config.json`.
///
/// `total_parameters` is what the host must hold: for a mixture of experts every expert stays
/// resident even though only a few are active per token. `active_parameters` is what memory
/// bandwidth must stream per token, so it drives throughput, not capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelGeometry {
    pub total_parameters: u64,
    /// Measured on-disk weight bytes, when known.
    ///
    /// Preferred over `total_parameters * 2` because a published checkpoint is not always bf16.
    /// `openai/gpt-oss-20b` ships in MXFP4 and weighs 13.8 GB, which the parameter count would
    /// have overestimated threefold -- and it is the resident bytes, not the parameter count,
    /// that decide whether a model fits.
    #[serde(default)]
    pub weight_bytes: Option<u64>,
    pub active_parameters: u64,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub max_position_embeddings: u32,
    #[serde(default)]
    pub sliding_window: Option<u32>,
    #[serde(default)]
    pub vision: bool,
}

impl ModelGeometry {
    /// Bytes of KV cache one token occupies across all layers, for both K and V.
    #[must_use]
    pub fn kv_bytes_per_token(&self) -> u64 {
        2 * u64::from(self.num_hidden_layers)
            * u64::from(self.num_key_value_heads)
            * u64::from(self.head_dim)
            * BF16_BYTES_PER_PARAMETER
    }
}

/// How the container will be launched. `tensor_parallel_size` comes from the EIM profile that
/// `INFERENCE_PROFILE_ID` pins, so the estimate and the container can never disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServingShape {
    pub context_tokens: u32,
    pub parallel_sequences: u32,
    pub tensor_parallel_size: u32,
}

/// The four buckets the UI displays, plus their sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryEstimate {
    /// Model weights: `total_parameters` at bf16.
    pub weight_bytes: u64,
    /// KV cache for the full context across every parallel sequence.
    pub kv_bytes: u64,
    /// Worker activations and the compiled graph, plus any vision tower.
    pub activation_bytes: u64,
    /// The vLLM process baseline, multiplied by tensor-parallel rank count.
    pub runtime_bytes: u64,
    pub required_bytes: u64,
}

impl MemoryEstimate {
    #[must_use]
    pub fn compute(geometry: &ModelGeometry, shape: &ServingShape) -> Self {
        let weight_bytes = geometry.weight_bytes.unwrap_or_else(|| {
            geometry
                .total_parameters
                .saturating_mul(BF16_BYTES_PER_PARAMETER)
        });

        let kv_bytes = geometry
            .kv_bytes_per_token()
            .saturating_mul(u64::from(shape.context_tokens))
            .saturating_mul(u64::from(shape.parallel_sequences.max(1)));

        let mut activation_bytes =
            (weight_bytes / ACTIVATION_FRACTION_OF_WEIGHTS).max(MINIMUM_ACTIVATION_BYTES);
        if geometry.vision {
            activation_bytes = activation_bytes.saturating_add(VISION_TOWER_BYTES);
        }

        // `distributed-executor-backend: mp` in every EIM profile means one full process per
        // tensor-parallel rank, so the baseline is paid per rank rather than once.
        let runtime_bytes =
            RUNTIME_BASELINE_BYTES.saturating_mul(u64::from(shape.tensor_parallel_size.max(1)));

        Self {
            weight_bytes,
            kv_bytes,
            activation_bytes,
            runtime_bytes,
            required_bytes: weight_bytes
                .saturating_add(kv_bytes)
                .saturating_add(activation_bytes)
                .saturating_add(runtime_bytes),
        }
    }

    fn domain_assessment(&self, usable_capacity_bytes: u64) -> HardwareMemoryDomainAssessment {
        HardwareMemoryDomainAssessment {
            memory_domain: MemoryDomainId::system(),
            model_bytes: self.weight_bytes,
            context_bytes: self.kv_bytes,
            compute_bytes: self.activation_bytes,
            auxiliary_bytes: self.runtime_bytes,
            required_bytes: self.required_bytes,
            usable_capacity_bytes,
            margin_bytes: i128::from(usable_capacity_bytes)
                .saturating_sub(i128::from(self.required_bytes))
                .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64,
        }
    }
}

/// Everything the assessor needs to know about the host and the container limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBudget {
    /// Stable system capacity from the topology, i.e. physical minus the reserve policy.
    pub stable_capacity_bytes: u64,
    /// An operator-pinned `--memory` ceiling, when one is configured.
    pub container_limit_bytes: Option<u64>,
}

impl HostBudget {
    /// Reads the stable system capacity out of a validated topology.
    #[must_use]
    pub fn from_topology(topology: &MemoryTopology, container_limit_bytes: Option<u64>) -> Self {
        Self {
            stable_capacity_bytes: topology
                .stable_capacity(&MemoryDomainId::system())
                .unwrap_or(0),
            container_limit_bytes,
        }
    }

    /// The binding budget: a container cannot exceed either the host reserve or its own limit.
    #[must_use]
    pub fn usable_bytes(&self) -> u64 {
        match self.container_limit_bytes {
            Some(limit) => self.stable_capacity_bytes.min(limit),
            None => self.stable_capacity_bytes,
        }
    }
}

/// Memory the container holds before any vLLM worker starts.
///
/// vLLM checks its requested reservation against memory *currently available* rather than against
/// the ceiling, and by the time a worker initializes the python interpreter, torch and EIM's
/// launcher are already resident. Measured at 1.94 GiB on the Xeon; carried at double that,
/// because underestimating it is a refusal to start and overestimating it costs a rounding error
/// on a host with hundreds of gigabytes.
const STARTUP_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Never reserve less than this: vLLM needs room for its own bookkeeping beyond the model.
const MINIMUM_CPU_UTILIZATION: f64 = 0.05;
/// Never claim the whole ceiling: `STARTUP_RESERVE_BYTES` has to stay outside the workers.
const MAXIMUM_CPU_UTILIZATION: f64 = 0.95;

/// The container memory ceiling and the engine's reservation fraction, which must agree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContainerMemoryPlan {
    /// `docker run --memory`, and `--memory-swap` set to the same value.
    pub limit_bytes: u64,
    /// `--gpu-memory-utilization`, which on the CPU backend is a fraction of `limit_bytes`.
    pub utilization: f64,
}

/// Derives both container memory controls from one estimate.
///
/// They are produced together because they multiply, and every way of getting this wrong was
/// observed on real hardware rather than reasoned about. vLLM 0.26 on the CPU backend:
///
/// ```text
/// Auto set (5.37/32.72) GiB for KV cache on node 1, with 11.11 GiB requested memory for the
/// worker. 5.74 GiB memory was consumed by non-kv usages.
/// ```
/// ```text
/// ValueError: Available memory on node 0 (30.78/32.72 GiB) on startup is less than desired CPU
/// memory utilization (0.95, 31.08 GiB).
/// ```
///
/// Three facts follow, and each one broke a load before it was known:
///
/// 1. The fraction is of the container's cgroup limit, not of a NUMA node. Treating it as a node
///    fraction while the cgroup is far smaller shrinks the real budget by the ratio between them,
///    which is what left a healthy 4B model with 1.47 GiB of key-value cache where it needed 18.
/// 2. *Each* tensor-parallel worker claims that fraction independently, so the reservation across
///    the container is `utilization x limit x ranks`. Ignoring the multiplication overcommits the
///    cgroup and earns an out-of-memory kill part-way through loading.
/// 3. The check is against memory currently available, not against the ceiling, so the ceiling
///    must exceed the workers' share by whatever is already resident.
///
/// Hence: the ceiling is the estimate with its slack *plus* the startup reserve, and the fraction
/// hands the ranks exactly the part that is not the reserve.
#[must_use]
pub fn container_memory_plan(
    estimate: &MemoryEstimate,
    tensor_parallel_size: u32,
    memory_limit_percent: u64,
) -> ContainerMemoryPlan {
    let ranks = u64::from(tensor_parallel_size.max(1));
    // What the ranks may hold between them: the estimate with the configured slack, so a slightly
    // low estimate does not fail and a runaway one is still killed.
    let workers_share = estimate
        .required_bytes
        .saturating_mul(memory_limit_percent.max(100))
        / 100;
    let limit_bytes = workers_share.saturating_add(STARTUP_RESERVE_BYTES);

    #[allow(clippy::cast_precision_loss)]
    let utilization = if limit_bytes == 0 {
        MAXIMUM_CPU_UTILIZATION
    } else {
        (workers_share as f64 / (limit_bytes as f64 * ranks as f64))
            .clamp(MINIMUM_CPU_UTILIZATION, MAXIMUM_CPU_UTILIZATION)
    };

    ContainerMemoryPlan {
        limit_bytes,
        utilization,
    }
}

/// A model is `Recommended` only when it leaves comfortable headroom; at 70% or more of the
/// budget it still loads, but the user is told it is tight.
const CONSTRAINED_UTILIZATION_NUMERATOR: u64 = 7;
const CONSTRAINED_UTILIZATION_DENOMINATOR: u64 = 10;

/// Build the assessment ICN's load admission and the client UI both consume.
#[must_use]
pub fn assess(
    geometry: &ModelGeometry,
    shape: &ServingShape,
    budget: &HostBudget,
    device: &str,
) -> HardwareAssessment {
    let estimate = MemoryEstimate::compute(geometry, shape);
    let usable = budget.usable_bytes();
    let profile = HardwareProfile {
        context_length: shape.context_tokens,
        acceleration: icn_hardware::XEON_VLLM_BACKEND.to_owned(),
        device: device.to_owned(),
    };

    if estimate.required_bytes <= usable {
        let constrained = estimate
            .required_bytes
            .saturating_mul(CONSTRAINED_UTILIZATION_DENOMINATOR)
            > usable.saturating_mul(CONSTRAINED_UTILIZATION_NUMERATOR);
        return HardwareAssessment::Fits {
            profile,
            memory: HardwareMemory {
                required_bytes: estimate.required_bytes,
                usable_capacity_bytes: usable,
                headroom_bytes: usable.saturating_sub(estimate.required_bytes),
                domains: vec![estimate.domain_assessment(usable)],
                device_constraints: Vec::new(),
            },
            recommendation: if constrained {
                HardwareRecommendation::Constrained
            } else {
                HardwareRecommendation::Recommended
            },
        };
    }

    HardwareAssessment::DoesNotFit {
        profile,
        memory: HardwareDeficit {
            required_bytes: estimate.required_bytes,
            usable_capacity_bytes: usable,
            deficit_bytes: estimate.required_bytes.saturating_sub(usable),
            domains: vec![estimate.domain_assessment(usable)],
            device_constraints: Vec::new(),
        },
        limiting_resource: "system memory".to_owned(),
        alternative: shortest_fitting_context(geometry, shape, budget).map(|context_tokens| {
            HardwareProfile {
                context_length: context_tokens,
                acceleration: icn_hardware::XEON_VLLM_BACKEND.to_owned(),
                device: device.to_owned(),
            }
        }),
    }
}

/// The largest parallel-sequence count that still fits, or `None` when even one does not.
///
/// Ascending rather than descending so the returned shape is the smallest admission that
/// satisfies the request, matching how the native controller enumerated candidates.
#[must_use]
pub fn largest_fitting_shape(
    geometry: &ModelGeometry,
    context_tokens: u32,
    tensor_parallel_size: u32,
    budget: &HostBudget,
) -> Option<ServingShape> {
    let usable = budget.usable_bytes();
    (1..=MAX_DYNAMIC_PARALLEL_SEQUENCES)
        .map(|parallel_sequences| ServingShape {
            context_tokens,
            parallel_sequences,
            tensor_parallel_size,
        })
        .take_while(|shape| MemoryEstimate::compute(geometry, shape).required_bytes <= usable)
        .last()
}

/// A shorter context that would fit, found by halving. Returns `None` when weights alone
/// exceed the budget, which is the Llama-4-Scout case: no context length can rescue it.
fn shortest_fitting_context(
    geometry: &ModelGeometry,
    shape: &ServingShape,
    budget: &HostBudget,
) -> Option<u32> {
    let usable = budget.usable_bytes();
    let mut context_tokens = shape.context_tokens / 2;
    while context_tokens >= 2048 {
        let candidate = ServingShape {
            context_tokens,
            parallel_sequences: 1,
            tensor_parallel_size: shape.tensor_parallel_size,
        };
        if MemoryEstimate::compute(geometry, &candidate).required_bytes <= usable {
            return Some(context_tokens);
        }
        context_tokens /= 2;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured target host: `magnitude-icn doctor` on the Xeon 6767P reports exactly
    /// this many bytes of physical memory (251 GiB). Using the measured figure rather than the
    /// marketing "256 GB" matters: it moves the budget to ~243 GB, which is what decides how
    /// much margin the Llama-4-Scout rejection actually has.
    const TARGET_HOST_PHYSICAL_BYTES: u64 = 269_853_134_848;

    fn target_host_budget() -> HostBudget {
        let physical = TARGET_HOST_PHYSICAL_BYTES;
        HostBudget {
            stable_capacity_bytes: physical
                - icn_hardware::system_memory_thresholds(physical).assess_reserve_bytes,
            container_limit_bytes: None,
        }
    }

    fn qwen3_30b_a3b() -> ModelGeometry {
        ModelGeometry {
            total_parameters: 30_500_000_000,
            weight_bytes: None,
            active_parameters: 3_300_000_000,
            num_hidden_layers: 48,
            num_key_value_heads: 4,
            head_dim: 128,
            max_position_embeddings: 40_960,
            sliding_window: None,
            vision: false,
        }
    }

    fn llama4_scout() -> ModelGeometry {
        ModelGeometry {
            total_parameters: 109_000_000_000,
            weight_bytes: None,
            active_parameters: 17_000_000_000,
            num_hidden_layers: 48,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 131_072,
            sliding_window: None,
            vision: true,
        }
    }

    fn shape(context_tokens: u32, tensor_parallel_size: u32) -> ServingShape {
        ServingShape {
            context_tokens,
            parallel_sequences: 1,
            tensor_parallel_size,
        }
    }

    /// What the Xeon 6767P reports per NUMA node: vLLM's own error message named
    /// 125.94 GiB, which is half of the host's 251 GiB.
    const NODE_CAPACITY_BYTES: u64 = 135_236_616_192;

    /// What vLLM 0.26 reported on the Xeon at a 32.72 GiB ceiling, and what the engine already
    /// held when it checked. Kept as constants because they are the only evidence for the formula.
    const MEASURED_LIMIT_GIB: f64 = 32.72;
    const MEASURED_WORKER_REQUEST_GIB: f64 = 11.11;
    const MEASURED_UTILIZATION: f64 = 0.3396;
    const MEASURED_RESIDENT_AT_STARTUP_GIB: f64 = 32.72 - 30.78;

    fn plan(geometry: &ModelGeometry, ranks: u32) -> ContainerMemoryPlan {
        container_memory_plan(
            &MemoryEstimate::compute(geometry, &shape(32_768, ranks)),
            ranks,
            115,
        )
    }

    #[test]
    fn the_fraction_is_of_the_container_limit_not_of_a_numa_node() {
        // The arithmetic that identifies the denominator: 0.3396 x 32.72 GiB = 11.11 GiB. The
        // NUMA node was 125.94 GiB, so it is not that.
        let implied = MEASURED_UTILIZATION * MEASURED_LIMIT_GIB;

        assert!(
            (implied - MEASURED_WORKER_REQUEST_GIB).abs() < 0.02,
            "the container limit is the denominator: {implied} vs {MEASURED_WORKER_REQUEST_GIB}"
        );
    }

    #[test]
    fn the_ranks_together_stay_inside_the_ceiling() {
        // Each worker claims the fraction independently, so exceeding the ceiling here is an
        // out-of-memory kill part-way through loading rather than a legible refusal.
        for ranks in [1, 2, 4, 8] {
            let plan = plan(&qwen3_30b_a3b(), ranks);
            let reserved = plan.utilization * plan.limit_bytes as f64 * f64::from(ranks);

            assert!(
                reserved <= plan.limit_bytes as f64,
                "{ranks} ranks reserve {reserved} of a {} ceiling",
                plan.limit_bytes
            );
        }
    }

    #[test]
    fn the_ranks_together_still_cover_the_estimate() {
        // The other side of the same constraint: too small a fraction and the key-value cache does
        // not fit, which is how a healthy 4B model was left with 1.47 GiB where it needed 18.
        for ranks in [1, 2, 4] {
            let estimate = MemoryEstimate::compute(&qwen3_30b_a3b(), &shape(32_768, ranks));
            let plan = container_memory_plan(&estimate, ranks, 115);
            let reserved = plan.utilization * plan.limit_bytes as f64 * f64::from(ranks);

            assert!(
                reserved >= estimate.required_bytes as f64,
                "{ranks} ranks reserve {reserved} for an estimate of {}",
                estimate.required_bytes
            );
        }
    }

    #[test]
    fn the_ceiling_leaves_room_for_what_is_resident_before_the_workers_start() {
        // vLLM compares its request against available memory, not against the ceiling. If the
        // workers' share were the whole ceiling, the launcher's own footprint would refuse it.
        for ranks in [1, 2] {
            for geometry in [qwen3_8b(), qwen3_30b_a3b()] {
                let plan = plan(&geometry, ranks);
                let per_worker = plan.utilization * plan.limit_bytes as f64;
                let available =
                    plan.limit_bytes as f64 - MEASURED_RESIDENT_AT_STARTUP_GIB * GIB as f64;

                assert!(
                    per_worker <= available,
                    "a worker asks for {per_worker} where {available} is available"
                );
            }
        }
    }

    #[test]
    fn a_small_model_gets_the_reserve_too() {
        // A fractional headroom would give a 2 GB model 300 MB of slack, which does not cover a
        // launcher measured at 1.94 GiB. The reserve is therefore absolute.
        let tiny = ModelGeometry {
            total_parameters: 500_000_000,
            weight_bytes: Some(1_000_000_000),
            active_parameters: 500_000_000,
            ..qwen3_8b()
        };
        let estimate = MemoryEstimate::compute(&tiny, &shape(2_048, 1));
        let plan = container_memory_plan(&estimate, 1, 115);

        assert!(
            plan.limit_bytes - (plan.utilization * plan.limit_bytes as f64) as u64
                >= MEASURED_RESIDENT_AT_STARTUP_GIB as u64 * 1024 * 1024 * 1024,
            "plan={plan:?}"
        );
    }

    #[test]
    fn more_ranks_each_claim_less() {
        assert!(plan(&qwen3_30b_a3b(), 1).utilization > plan(&qwen3_30b_a3b(), 2).utilization);
    }

    #[test]
    fn never_claims_the_whole_ceiling() {
        assert!(plan(&qwen3_8b(), 1).utilization <= MAXIMUM_CPU_UTILIZATION);
    }

    #[test]
    fn never_claims_a_sliver() {
        assert!(plan(&qwen3_8b(), 64).utilization >= MINIMUM_CPU_UTILIZATION);
    }

    #[test]
    fn a_zero_rank_count_is_treated_as_one() {
        assert_eq!(plan(&qwen3_8b(), 0), plan(&qwen3_8b(), 1));
    }

    #[test]
    fn a_ceiling_below_the_estimate_is_raised_to_it() {
        // A percentage under 100 would hand the engine less than the model needs, which is not a
        // ceiling but a guaranteed failure.
        let estimate = MemoryEstimate::compute(&qwen3_8b(), &shape(32_768, 1));

        assert_eq!(
            container_memory_plan(&estimate, 1, 50),
            container_memory_plan(&estimate, 1, 100),
        );
    }

    fn qwen3_8b() -> ModelGeometry {
        ModelGeometry {
            total_parameters: 8_200_000_000,
            weight_bytes: None,
            active_parameters: 8_200_000_000,
            num_hidden_layers: 36,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 40_960,
            sliding_window: None,
            vision: false,
        }
    }

    #[test]
    fn buckets_sum_to_the_required_total() {
        let estimate = MemoryEstimate::compute(&qwen3_30b_a3b(), &shape(32_768, 2));

        assert_eq!(
            estimate.required_bytes,
            estimate.weight_bytes
                + estimate.kv_bytes
                + estimate.activation_bytes
                + estimate.runtime_bytes,
        );
    }

    #[test]
    fn prefers_measured_weight_bytes_over_the_parameter_count() {
        // gpt-oss-20b ships in MXFP4: 13.8 GB on disk for ~21 billion parameters, which
        // `parameters * 2` would have called 42 GB. Resident bytes are what decide fit.
        let measured = ModelGeometry {
            total_parameters: 21_000_000_000,
            active_parameters: 3_600_000_000,
            weight_bytes: Some(13_800_000_000),
            num_hidden_layers: 24,
            num_key_value_heads: 8,
            head_dim: 64,
            max_position_embeddings: 131_072,
            sliding_window: Some(128),
            vision: false,
        };

        assert_eq!(
            MemoryEstimate::compute(&measured, &shape(32_768, 1)).weight_bytes,
            13_800_000_000
        );
        // Without a measurement the bf16 assumption still applies.
        let derived = ModelGeometry {
            weight_bytes: None,
            ..measured
        };
        assert_eq!(
            MemoryEstimate::compute(&derived, &shape(32_768, 1)).weight_bytes,
            42_000_000_000
        );
    }

    #[test]
    fn charges_total_parameters_for_a_mixture_of_experts() {
        // All experts stay resident in host RAM even though 3.3B are active per token.
        let estimate = MemoryEstimate::compute(&qwen3_30b_a3b(), &shape(32_768, 1));

        assert_eq!(estimate.weight_bytes, 61_000_000_000);
    }

    #[test]
    fn charges_the_runtime_baseline_once_per_tensor_parallel_rank() {
        let tp1 = MemoryEstimate::compute(&qwen3_30b_a3b(), &shape(32_768, 1));
        let tp2 = MemoryEstimate::compute(&qwen3_30b_a3b(), &shape(32_768, 2));

        // `distributed-executor-backend: mp` spawns one process per rank.
        assert_eq!(tp2.runtime_bytes, 2 * tp1.runtime_bytes);
        assert_eq!(
            tp2.required_bytes - tp1.required_bytes,
            RUNTIME_BASELINE_BYTES
        );
    }

    #[test]
    fn kv_cache_scales_with_context_and_parallel_sequences() {
        let geometry = qwen3_30b_a3b();
        let one = MemoryEstimate::compute(&geometry, &shape(16_384, 2));
        let two = MemoryEstimate::compute(
            &geometry,
            &ServingShape {
                context_tokens: 16_384,
                parallel_sequences: 2,
                tensor_parallel_size: 2,
            },
        );

        assert_eq!(two.kv_bytes, 2 * one.kv_bytes);
    }

    #[test]
    fn qwen3_30b_a3b_fits_the_target_host_with_room_to_spare() {
        let assessment = assess(
            &qwen3_30b_a3b(),
            &shape(32_768, 2),
            &target_host_budget(),
            "Intel(R) Xeon(R) 6767P",
        );

        match assessment {
            HardwareAssessment::Fits {
                memory,
                recommendation,
                ..
            } => {
                // Roughly 72 GB against a ~243 GB budget.
                assert!(
                    memory.required_bytes < 80_000_000_000,
                    "{}",
                    memory.required_bytes
                );
                assert!(memory.headroom_bytes > 150_000_000_000);
                assert_eq!(recommendation, HardwareRecommendation::Recommended);
                assert_eq!(memory.domains.len(), 1);
                assert!(memory.domains[0].margin_bytes > 0);
            }
            other => panic!("expected Fits, got {other:?}"),
        }
    }

    #[test]
    fn llama4_scout_does_not_fit_and_no_shorter_context_rescues_it() {
        // 109B parameters at bf16 is 218 GB of weights alone. With activations, the vision
        // tower, per-rank runtime and KV cache it needs ~255 GB against a ~243 GB budget --
        // only a 5% margin, so this doubles as a sensitivity test on the formula's constants.
        // Its real point is to reject the model before the user waits out a 218 GB download
        // and is then OOM-killed.
        let assessment = assess(
            &llama4_scout(),
            &shape(32_768, 2),
            &target_host_budget(),
            "Intel(R) Xeon(R) 6767P",
        );

        match assessment {
            HardwareAssessment::DoesNotFit {
                memory,
                limiting_resource,
                alternative,
                ..
            } => {
                assert!(
                    memory.required_bytes > TARGET_HOST_PHYSICAL_BYTES / 10 * 9,
                    "{}",
                    memory.required_bytes
                );
                assert!(memory.deficit_bytes > 0);
                assert_eq!(limiting_resource, "system memory");
                // Weights dominate, so shortening the context cannot help.
                assert!(
                    alternative.is_none(),
                    "unexpected alternative: {alternative:?}"
                );
            }
            other => panic!("expected DoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn offers_a_shorter_context_when_the_kv_cache_is_what_overflows() {
        // A model whose weights fit easily but whose full-context KV cache does not.
        let geometry = ModelGeometry {
            total_parameters: 8_000_000_000,
            weight_bytes: None,
            active_parameters: 8_000_000_000,
            num_hidden_layers: 64,
            num_key_value_heads: 64,
            head_dim: 128,
            max_position_embeddings: 1_000_000,
            sliding_window: None,
            vision: false,
        };
        let budget = HostBudget {
            stable_capacity_bytes: 40 * GIB,
            container_limit_bytes: None,
        };

        match assess(&geometry, &shape(524_288, 1), &budget, "cpu") {
            HardwareAssessment::DoesNotFit { alternative, .. } => {
                let alternative = alternative.expect("a shorter context should fit");
                assert!(alternative.context_length < 524_288);
                assert!(alternative.context_length >= 2048);
            }
            other => panic!("expected DoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn charges_a_margin_for_a_vision_tower() {
        let text_only = ModelGeometry {
            vision: false,
            ..qwen3_30b_a3b()
        };
        let multimodal = ModelGeometry {
            vision: true,
            ..qwen3_30b_a3b()
        };

        let delta = MemoryEstimate::compute(&multimodal, &shape(32_768, 2)).required_bytes
            - MemoryEstimate::compute(&text_only, &shape(32_768, 2)).required_bytes;

        assert_eq!(delta, VISION_TOWER_BYTES);
    }

    #[test]
    fn a_container_limit_binds_tighter_than_the_host_reserve() {
        let budget = HostBudget {
            stable_capacity_bytes: 230 * GIB,
            container_limit_bytes: Some(64 * GIB),
        };

        assert_eq!(budget.usable_bytes(), 64 * GIB);
        // Qwen3-30B-A3B needs ~72 GB, so a 64 GB container ceiling rejects it.
        assert!(matches!(
            assess(&qwen3_30b_a3b(), &shape(32_768, 2), &budget, "cpu"),
            HardwareAssessment::DoesNotFit { .. }
        ));
    }

    #[test]
    fn reports_constrained_when_utilization_passes_seventy_percent() {
        let estimate = MemoryEstimate::compute(&qwen3_30b_a3b(), &shape(32_768, 2));
        // Set the budget so the model needs just over 70% of it.
        let budget = HostBudget {
            stable_capacity_bytes: estimate.required_bytes * 10 / 7 - 1,
            container_limit_bytes: None,
        };

        assert!(matches!(
            assess(&qwen3_30b_a3b(), &shape(32_768, 2), &budget, "cpu"),
            HardwareAssessment::Fits {
                recommendation: HardwareRecommendation::Constrained,
                ..
            }
        ));
    }

    #[test]
    fn picks_the_largest_parallel_sequence_count_that_fits() {
        let geometry = qwen3_30b_a3b();

        // Generous budget: the cap, not memory, is the limit.
        let generous =
            largest_fitting_shape(&geometry, 32_768, 2, &target_host_budget()).expect("should fit");
        assert_eq!(generous.parallel_sequences, MAX_DYNAMIC_PARALLEL_SEQUENCES);

        // A budget that admits exactly one sequence.
        let one_sequence_bytes = MemoryEstimate::compute(
            &geometry,
            &ServingShape {
                context_tokens: 32_768,
                parallel_sequences: 1,
                tensor_parallel_size: 2,
            },
        )
        .required_bytes;
        let tight = largest_fitting_shape(
            &geometry,
            32_768,
            2,
            &HostBudget {
                stable_capacity_bytes: one_sequence_bytes,
                container_limit_bytes: None,
            },
        )
        .expect("one sequence should fit");
        assert_eq!(tight.parallel_sequences, 1);
    }

    #[test]
    fn reports_no_fitting_shape_when_even_one_sequence_overflows() {
        assert!(
            largest_fitting_shape(&llama4_scout(), 32_768, 2, &target_host_budget(),).is_none()
        );
    }
}
