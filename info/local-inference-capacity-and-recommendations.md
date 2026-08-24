# Local inference capacity and recommendations

ICN is the only authority for inference hardware, model fit, the model catalog, and active runtime
state. CLI and web actions call ACN RPCs; ACN translates those actions to the generated ICN client.
ACN never treats its own host as the inference machine.

Inference runs in an Intel EIM container, not in ICN's address space. ICN starts and stops that
container and speaks HTTP to it over loopback. Clients still never talk to the container: ICN remains
the only route, and there is no alternate transport or externally supplied endpoint.

## The catalog

The catalog is Magnitude-owned metadata over the models EIM ships. For each one it records the
Hugging Face repository, the EIM serving profile to pin, the served context, geometry, curated
quality evidence, licence, and which vLLM parsers the model needs for tool calls and reasoning.

Geometry and weight sizes are fetched from Hugging Face rather than asserted, because the memory
estimate depends on them. Quality scores are curated estimates and their provenance says so: no
benchmark is run here, and reporting an estimate as a measurement would be a fabrication.

Every catalog model is listed whether or not this host can serve it. A model that does not fit, is
gated without a credential, or has no tool-call parser stays visible with a machine-readable reason
and a plain-language note. The onboarding chooser is the deliberate exception — it is a guided
first-run flow and offers only models that can actually be served.

Served context is often below a model's trained maximum. The key-value cache is sized from it, and on
the CPU backend the engine can refuse to start when that cache does not fit, so the catalog says why
whenever the two differ.

## Capacity

`GET /v1/hardware` is the hardware and live-memory authority. It reports one system memory domain
holding one CPU device: a container-backed engine exposes no devices to ICN, and inventing them would
put fiction into the topology that assessments are validated against.

Host NUMA node count is load-bearing rather than cosmetic. EIM's profile selector rejects a
tensor-parallel size above it, so it decides which serving profiles are usable at all, and
`magnitude-icn doctor` reports the maximum.

Fit is estimated from catalog geometry rather than measured, and is separate from the memory
reservation the engine itself honors. See `info/inference/fit-estimation.md`.

## Recommendations

Ranking uses curated capability, an estimated generation speed, and runtime memory. Two of those
inputs behave differently than they did with a local GGUF inventory.

Speed starts as a memory-bandwidth roofline over active parameters, since decode on a CPU host is
bandwidth-bound. That makes a mixture of experts rank far above a dense model of the same weight, and
it is an estimate until real throughput has been observed for that model on that host.

Quantization fidelity no longer discriminates. Every EIM serving profile is bf16, so the fidelity term
is constant across candidates and drops out of the ranking. `Smartest` therefore resolves toward
capability, which for an all-bf16 catalog is the correct reading rather than a degradation.

## Residency

At most one model is resident at a time. Loading terminalizes the previous instance before the
replacement becomes Ready, because the memory estimate is reasoned against the whole host budget and
two resident models would silently double what that budget assumed.

The client sets an idle timeout — longer while connected, shorter after it disconnects — and ICN
enforces it by releasing the container. ACN persists only user profile and ordinary slot selections;
it keeps no competing artifact index, endpoint binding, or active-model record.

The local provider ID is `local`. Its catalog is projected from what ICN reports, demand loading uses
ICN runtime control, and generation streams through ICN chat.

## Weights

Weights are not managed by Magnitude yet. A model absent from the mounted cache is downloaded by the
engine inside the container, which means a fresh container downloads it again — accepted for now, and
the reason installing a model through the catalog is refused explicitly rather than admitted and left
to stall.
