---
applies_to:
  - inference/crates/icn-contracts/src/inventory.rs
  - inference/crates/icn-contracts/src/models.rs
  - inference/crates/icn-eim/src/estimate.rs
  - inference/crates/icn-eim/src/catalog.rs
  - packages/acn/src/local-model-recommendation-policy.ts
  - packages/acn/src/local-model-recommendations.ts
---

# Generation-performance estimation

## Contract

ICN estimates single-user decode throughput at several occupied-context depths for one serving
configuration. The ordered estimates are advisory evidence for recommendations. They never change
capacity, authorize a load, or replace observed runtime timing.

An estimate is not a measurement, and its confidence says which it is. Nothing may present an
estimate as though throughput had been observed.

## Where the numbers come from

Decode on a CPU host is bound by memory bandwidth, not arithmetic. Each generated token requires
reading the active parameters and the occupied key-value cache, so the estimate is a roofline:
achievable bandwidth divided by the bytes one token must move.

Active parameters, not total, drive it. A mixture of experts holds every expert resident but streams
only a few per token, so it estimates far faster than a dense model of the same weight — which is
the single most consequential property of this shape on a bandwidth-bound host.

Achievable bandwidth is a fraction of the host's theoretical peak. That fraction is an assumption
until measured, and it is the least certain input in the estimate.

## Two tiers

A configuration that has never run reports a **cold** estimate from the roofline, at low confidence,
with a wide interval. It is likely to be wrong in either direction, and most wrong for a mixture of
experts and for long context.

Once real generations have been observed for that image, profile, host and context depth, the
estimate becomes **warm**: measured throughput, with confidence rising and the interval narrowing as
samples accumulate. Warm estimates supersede the roofline for the configuration they describe and
never for another.

A warm estimate is keyed by the serving image digest as well as the host. A different image is a
different engine build with different accepted arguments, so its measurements do not transfer.

## Shape requirements

Samples are ordered by strictly increasing context depth, the last sample sits exactly at the
configuration's served context, and every rate satisfies `0 < lower <= estimated <= upper`. These
are not stylistic: the client's schema enforces them, and a violation makes the whole assessment
undecodable rather than merely imprecise.

## What this does not do

It does not decide whether a model fits. Capacity is `info/inference/fit-estimation.md`, and the
memory reservation the engine itself honors is separate again — see `design/icn/eim-containers.md`.

It does not rank models. The client's recommendation policy combines this estimate with curated
capability and runtime memory; throughput is one term among several.

It does not model concurrency. These are single-user figures. Aggregate throughput under several
simultaneous sessions is a different quantity, unmeasured here, and one that batching makes behave
unlike the single-stream case — a mixture of experts in particular loses its advantage as more
sequences activate more experts per step.

## Acceptance criteria

- Every reported sample set is non-empty, strictly ascending by context, and ends at the served
  context.
- A cold estimate reports low confidence; confidence only rises on observed generations.
- Measurements are keyed by image digest, profile, host and context depth, and never reused across
  a different image.
- No estimate is described as measured throughput anywhere it reaches the user.
