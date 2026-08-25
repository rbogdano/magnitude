//! Estimated single-stream decode throughput.
//!
//! Decode on a CPU host is bound by memory bandwidth rather than arithmetic: producing one token
//! means reading every active parameter and the occupied key-value cache. So the estimate is a
//! roofline — achievable bandwidth divided by the bytes one token must move — and its most
//! consequential property is that it uses *active* parameters. A mixture of experts holds every
//! expert resident but streams only a few per token, so it estimates far faster than a dense model
//! of the same weight, which is the correct reading on a bandwidth-bound machine.
//!
//! This is an estimate and says so. Nothing here is a measurement of the model being asked about,
//! every result carries `Low` confidence, and the interval is wide enough to admit being wrong in
//! either direction. Presenting it as observed throughput would be a fabrication.

use icn_contracts::models::{PerformanceConfidence, PerformanceEvidence};

use crate::estimate::ModelGeometry;

const BF16_BYTES_PER_PARAMETER: u64 = 2;

/// Achievable read bandwidth per NUMA node, in bytes per second.
///
/// Calibrated rather than assumed, from one observation: a dense 8.2B model at two ranks decoded
/// 29.6 tokens per second on a Xeon 6767P, which puts 8.2e9 x 2 x 29.6 ~= 485 GB/s across two
/// nodes. That is the only number in this module with any evidence behind it, and it comes from a
/// single model on a single machine — hence the wide interval and the low confidence.
///
/// It is deliberately *achievable* rather than theoretical peak. A theoretical figure derived from
/// channel count and DDR5 speed would be roughly 50% optimistic and would need a fudge factor
/// applied to it, which is a guess wearing a calculation's clothes.
const ACHIEVABLE_BYTES_PER_SECOND_PER_NODE: f64 = 243_000_000_000.0;

/// How far the interval spreads either side of the estimate.
///
/// Chosen to be embarrassingly wide, because the underlying figure is one measurement extrapolated
/// across a whole catalog. A narrow interval around a number this weakly evidenced would claim a
/// precision that does not exist.
const INTERVAL_FACTOR: f64 = 1.6;

/// The smallest rate worth reporting. The schema requires a strictly positive lower bound, and a
/// model slow enough to approach this is unusable rather than slow.
const MINIMUM_TOKENS_PER_SECOND: f64 = 0.01;

/// Estimated decode rate at one occupied-context depth.
#[must_use]
pub fn tokens_per_second(geometry: &ModelGeometry, numa_nodes: u32, context_tokens: u32) -> f64 {
    let bandwidth = ACHIEVABLE_BYTES_PER_SECOND_PER_NODE * f64::from(numa_nodes.max(1));
    // Active parameters, not total: the whole point of the shape.
    let weight_bytes = geometry
        .active_parameters
        .saturating_mul(BF16_BYTES_PER_PARAMETER);
    let cache_bytes = geometry
        .kv_bytes_per_token()
        .saturating_mul(u64::from(context_tokens));
    #[allow(clippy::cast_precision_loss)]
    let per_token = (weight_bytes.saturating_add(cache_bytes)) as f64;
    if per_token <= 0.0 {
        return MINIMUM_TOKENS_PER_SECOND;
    }
    (bandwidth / per_token).max(MINIMUM_TOKENS_PER_SECOND)
}

/// The context depths to report, given what the client asked for and the served context.
///
/// The client's schema requires a non-empty, strictly ascending list whose last entry is exactly
/// the served context, and rejects the whole assessment otherwise — so requested depths are
/// normalized here rather than trusted. A violation makes the assessment undecodable instead of
/// merely imprecise, which would leave the model invisible with no stated reason.
#[must_use]
pub fn context_ladder(requested: &[u32], served_context_tokens: u32) -> Vec<u32> {
    let served = served_context_tokens.max(1);
    let mut depths: Vec<u32> = requested
        .iter()
        .copied()
        .filter(|tokens| *tokens > 0 && *tokens < served)
        .collect();
    depths.sort_unstable();
    depths.dedup();
    depths.push(served);
    depths
}

/// Estimates throughput at each requested depth.
#[must_use]
pub fn evidence(
    geometry: &ModelGeometry,
    numa_nodes: u32,
    requested: &[u32],
    served_context_tokens: u32,
) -> Vec<PerformanceEvidence> {
    context_ladder(requested, served_context_tokens)
        .into_iter()
        .map(|context_tokens| {
            let estimated = tokens_per_second(geometry, numa_nodes, context_tokens);
            PerformanceEvidence {
                context_tokens,
                lower_tokens_per_second: (estimated / INTERVAL_FACTOR)
                    .max(MINIMUM_TOKENS_PER_SECOND),
                estimated_tokens_per_second: estimated,
                upper_tokens_per_second: estimated * INTERVAL_FACTOR,
                // Never anything else here. Confidence rises only on observed generations, and
                // none are observed at assessment time -- the model is not even downloaded yet.
                confidence: PerformanceConfidence::Low,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn qwen3_30b_a3b() -> ModelGeometry {
        ModelGeometry {
            total_parameters: 30_500_000_000,
            active_parameters: 3_300_000_000,
            ..qwen3_8b()
        }
    }

    #[test]
    fn reproduces_the_measurement_it_was_calibrated_from() {
        // 29.6 tokens per second, dense 8.2B, two nodes, short context. If this drifts, the
        // constant has been changed without re-deriving it from an observation.
        let estimated = tokens_per_second(&qwen3_8b(), 2, 512);

        assert!(
            (estimated - 29.6).abs() < 1.5,
            "expected roughly the observed 29.6 tok/s, got {estimated}"
        );
    }

    #[test]
    fn a_mixture_of_experts_estimates_far_faster_than_its_weight_suggests() {
        // The single most consequential property of this shape on a bandwidth-bound host: 30B
        // resident, 3.3B active, so it should outrun a dense 8B rather than trail it.
        let dense = tokens_per_second(&qwen3_8b(), 2, 4_096);
        let sparse = tokens_per_second(&qwen3_30b_a3b(), 2, 4_096);

        assert!(sparse > dense * 2.0, "dense={dense} sparse={sparse}");
    }

    #[test]
    fn longer_context_estimates_slower() {
        let short = tokens_per_second(&qwen3_8b(), 2, 1_024);
        let long = tokens_per_second(&qwen3_8b(), 2, 32_768);

        assert!(long < short, "short={short} long={long}");
    }

    #[test]
    fn more_nodes_estimate_faster() {
        assert!(
            tokens_per_second(&qwen3_8b(), 2, 4_096) > tokens_per_second(&qwen3_8b(), 1, 4_096)
        );
        // A host reporting no nodes is treated as one rather than dividing by zero.
        assert_eq!(
            tokens_per_second(&qwen3_8b(), 0, 4_096),
            tokens_per_second(&qwen3_8b(), 1, 4_096)
        );
    }

    #[test]
    fn the_ladder_always_ends_at_the_served_context() {
        // The client's schema requires exactly this and rejects the whole assessment otherwise,
        // which would leave the model invisible with no reason given.
        assert_eq!(
            context_ladder(&[4_096, 16_384], 32_768),
            vec![4_096, 16_384, 32_768]
        );
        assert_eq!(context_ladder(&[], 8_192), vec![8_192]);
    }

    #[test]
    fn the_ladder_is_strictly_ascending_whatever_was_asked_for() {
        // Duplicates, zeroes, reversed order and depths past the served context all have to
        // normalize rather than propagate into an undecodable response.
        assert_eq!(
            context_ladder(&[16_384, 4_096, 4_096, 0, 99_999, 32_768], 32_768),
            vec![4_096, 16_384, 32_768],
        );
    }

    #[test]
    fn every_sample_satisfies_the_schemas_ordering_of_bounds() {
        let samples = evidence(&qwen3_30b_a3b(), 2, &[2_048, 8_192], 32_768);

        assert_eq!(samples.len(), 3);
        for sample in &samples {
            assert!(sample.lower_tokens_per_second > 0.0, "{sample:?}");
            assert!(
                sample.lower_tokens_per_second <= sample.estimated_tokens_per_second,
                "{sample:?}"
            );
            assert!(
                sample.estimated_tokens_per_second <= sample.upper_tokens_per_second,
                "{sample:?}"
            );
            assert_eq!(sample.confidence, PerformanceConfidence::Low);
        }
        assert_eq!(samples.last().expect("a sample").context_tokens, 32_768);
    }

    #[test]
    fn an_enormous_model_still_reports_a_positive_rate() {
        // The schema needs a strictly positive lower bound even where the honest answer is
        // "unusably slow", and reporting zero would make the assessment undecodable.
        let enormous = ModelGeometry {
            total_parameters: 2_000_000_000_000,
            active_parameters: 2_000_000_000_000,
            ..qwen3_8b()
        };

        let samples = evidence(&enormous, 1, &[], 131_072);

        assert!(samples[0].lower_tokens_per_second > 0.0, "{samples:?}");
    }
}
