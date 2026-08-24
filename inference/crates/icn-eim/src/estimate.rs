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
        let weight_bytes = geometry
            .total_parameters
            .saturating_mul(BF16_BYTES_PER_PARAMETER);

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

/// Headroom above the estimate when telling vLLM how much of a NUMA node to reserve.
///
/// The estimate is a model of allocation, not a measurement of it, so the reservation is asked
/// for with margin. Too tight and the engine runs out mid-load; too loose and it refuses to
/// start because the node does not have that much free.
const CPU_UTILIZATION_SAFETY_NUMERATOR: u64 = 3;
const CPU_UTILIZATION_SAFETY_DENOMINATOR: u64 = 2;

/// Never reserve less than this fraction of a node: vLLM needs room for its own bookkeeping
/// beyond what the model accounts for.
const MINIMUM_CPU_UTILIZATION: f64 = 0.05;
/// Never ask for more than this. vLLM compares the request against *currently free* memory, so
/// asking for nearly the whole node fails on any host that is doing anything else at all --
/// which is exactly how the default of 0.92 fails on an almost idle machine.
const MAXIMUM_CPU_UTILIZATION: f64 = 0.85;

/// The fraction of one NUMA node vLLM should reserve, for `--gpu-memory-utilization`.
///
/// Despite the name, that flag controls CPU memory on the CPU backend, and it is a fraction of
/// a *node* rather than an absolute size. This is the value that actually decides whether a
/// container starts: vLLM's default of 0.92 asks for 92% of every node and fails on a host with
/// anything else resident. No shipped EIM profile sets it.
///
/// With `distributed-executor-backend: mp` each tensor-parallel rank binds one node and holds
/// its shard, so the per-node requirement is the estimate divided by the rank count.
#[must_use]
pub fn cpu_memory_utilization(
    estimate: &MemoryEstimate,
    tensor_parallel_size: u32,
    node_capacity_bytes: u64,
) -> f64 {
    if node_capacity_bytes == 0 {
        return MAXIMUM_CPU_UTILIZATION;
    }
    let ranks = u64::from(tensor_parallel_size.max(1));
    let per_node = estimate
        .required_bytes
        .div_ceil(ranks)
        .saturating_mul(CPU_UTILIZATION_SAFETY_NUMERATOR)
        / CPU_UTILIZATION_SAFETY_DENOMINATOR;

    let fraction = per_node as f64 / node_capacity_bytes as f64;
    fraction.clamp(MINIMUM_CPU_UTILIZATION, MAXIMUM_CPU_UTILIZATION)
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

    #[test]
    fn reserves_a_node_fraction_sized_to_the_model_rather_than_the_host() {
        // The default of 0.92 asks for 92% of a node and fails on an almost idle machine; that
        // is the failure this exists to prevent.
        let estimate = MemoryEstimate::compute(&qwen3_8b(), &shape(32_768, 2));
        let utilization = cpu_memory_utilization(&estimate, 2, NODE_CAPACITY_BYTES);

        assert!(utilization > 0.05, "{utilization}");
        assert!(
            utilization < 0.3,
            "an 8B model needs a small slice of a 135 GB node: {utilization}"
        );
    }

    #[test]
    fn a_larger_model_reserves_a_larger_fraction() {
        let small = cpu_memory_utilization(
            &MemoryEstimate::compute(&qwen3_8b(), &shape(32_768, 2)),
            2,
            NODE_CAPACITY_BYTES,
        );
        let large = cpu_memory_utilization(
            &MemoryEstimate::compute(&qwen3_30b_a3b(), &shape(32_768, 2)),
            2,
            NODE_CAPACITY_BYTES,
        );

        assert!(large > small, "small={small} large={large}");
    }

    #[test]
    fn more_ranks_reserve_less_of_each_node() {
        // Each rank binds one node and holds only its shard.
        let estimate = MemoryEstimate::compute(&qwen3_30b_a3b(), &shape(32_768, 2));
        let one_rank = cpu_memory_utilization(&estimate, 1, NODE_CAPACITY_BYTES);
        let two_ranks = cpu_memory_utilization(&estimate, 2, NODE_CAPACITY_BYTES);

        assert!(two_ranks < one_rank, "one={one_rank} two={two_ranks}");
    }

    #[test]
    fn never_asks_for_nearly_a_whole_node() {
        // A model far larger than a node must still produce a request vLLM can satisfy against
        // free memory rather than one guaranteed to be refused.
        let enormous = MemoryEstimate::compute(&llama4_scout(), &shape(32_768, 1));

        assert_eq!(
            cpu_memory_utilization(&enormous, 1, NODE_CAPACITY_BYTES),
            MAXIMUM_CPU_UTILIZATION
        );
    }

    #[test]
    fn never_asks_for_a_sliver() {
        let tiny = ModelGeometry {
            total_parameters: 100_000_000,
            active_parameters: 100_000_000,
            ..qwen3_8b()
        };
        let utilization = cpu_memory_utilization(
            &MemoryEstimate::compute(&tiny, &shape(2_048, 1)),
            1,
            NODE_CAPACITY_BYTES,
        );

        assert!(utilization >= MINIMUM_CPU_UTILIZATION);
    }

    #[test]
    fn falls_back_to_the_ceiling_when_node_capacity_is_unknown() {
        let estimate = MemoryEstimate::compute(&qwen3_8b(), &shape(32_768, 2));

        assert_eq!(
            cpu_memory_utilization(&estimate, 2, 0),
            MAXIMUM_CPU_UTILIZATION
        );
    }

    fn qwen3_8b() -> ModelGeometry {
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
