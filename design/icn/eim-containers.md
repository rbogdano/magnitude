---
applies_to:
  - inference/crates/icn-eim/**
  - inference/crates/icn-hardware/**
  - inference/crates/icn-server/**
  - inference/eim/**
  - inference/eim-pin.toml
---

# EIM container backend

## Contract

ICN serves a model by running one Intel EIM container: a Docker image that resolves a validated
vLLM profile at startup and exposes an OpenAI-compatible API. ICN owns that container's whole
lifetime — it decides what to start, when to stop it, and removes what it left behind — and reaches
it only over loopback HTTP.

This replaces the native inference engine. There is no llama.cpp, no fit planner, no per-accelerator
feature graph, and no in-process model state. `grep -rn llama_cpp_2 inference/crates/` must stay
empty.

The boundary is the two traits `icn-api` already defines. `CompletionBackend` carries inference;
`ModelInstanceController` carries lifecycle. Nothing container-specific may appear in
`icn-contracts` or `icn-api`: preserving that separation is what keeps the generated client protocol
unchanged, and `bun icn:check-generated` is the check that says so.

## One resident container

At most one container is Ready at a time. Loading a model terminalizes the previous instance before
the replacement becomes Ready, and a lease naming a replaced instance fails rather than being served
by its successor.

This was already the invariant for the native controller. It carries over for a stronger reason: the
memory estimate is computed against the whole host budget, so two resident models would silently
double what that budget was reasoned about.

Canonical lifecycle lives in the instance snapshot, never in the load stream. The client drains that
stream for progress and discards it, so an instance is registered as Loading *before* its container
starts — a client that only polls must still observe every transition.

## Memory is estimated, and vLLM's own control is separate

Fit is computed from six geometry numbers per model, not measured. Weights are charged at their
measured on-disk size when known, because a published checkpoint is not always bf16, and for a
mixture of experts every expert is resident even though few are active per token.

The estimate is not what governs the engine's allocation. vLLM's CPU backend reserves a *fraction of
each NUMA node* through `--gpu-memory-utilization`, misleadingly named on that backend, and defaults
to 0.92 — which fails on a host doing anything at all. That reservation is derived from the estimate
and is the value that decides whether a container starts. The two quantities must not be conflated.

On a large host the estimate is informational far more often than it is a gate. Its value is
therefore in what it displays, which is why the four memory buckets — weights, KV cache,
activations, runtime — are filled in separately rather than lumped together.

## Model data is fetched, capability claims are load-bearing

Geometry comes from each model's Hugging Face `config.json` and weight sizes from its safetensors
index. A guessed layer count yields an estimate that is confidently wrong, which is worse than an
absent one; when a number cannot be obtained, that must be stated rather than filled in. Quality
scores are curated estimates and say so in their provenance, because no benchmark was run here.

A model's `toolCallParser` decides whether the catalog presents it as usable. A wrong entry does not
produce an error: vLLM returns tool calls as prose and the agent loop fails silently. A parser that
has not been observed working on real hardware is a claim, and a model whose parser is unknown is
marked unusable rather than optimistically offered.

## Everything is listed, nothing is hidden

The catalog reports every model in the table, whether or not this host can serve it. A model that
does not fit, is gated without a credential, or has no tool-call parser stays listed, carrying both
a machine-readable reason and a plain-language note. Omitting it would leave the user unable to
learn why a model they expected is absent.

The onboarding chooser is the exception, deliberately: it is a guided first-run flow rather than a
catalog browser, and it presents only models that can actually be served.

## Container launch invariants

Each of these was established by watching a real vLLM refuse to start, and two of them contradict
what EIM's own documentation implies. Absence from documentation is not evidence.

- The container port is always 8000. The base image's `HEALTHCHECK` hardcodes `localhost:8000`, so
  changing it leaves a working container permanently unhealthy.
- Ports are published on loopback only. The served surface has no authentication and no TLS.
- Shared memory must exceed Docker's 64 MB default whenever tensor parallelism is used, or vLLM's
  workers cannot broadcast and the only symptom is a gloo "connection closed by peer".
- `CAP_SYS_NICE` is required for NUMA binding; without it workers run unbound and lose the locality
  the tensor-parallel split existed to provide.
- `--memory-swap` equals `--memory`, so a bad estimate is a fast, legible kill rather than unbounded
  thrash. An out-of-memory kill is reported as a low-memory failure with real byte figures.
- Proxy variables are passed in whenever the container may download, because the daemon can have a
  proxy for image pulls while the container has none — and that failure reads as a missing model.
- `INFERENCE_PROFILE_ID` is always pinned. Auto-selection could choose a different tensor-parallel
  size than the estimate assumed, which would make the estimate meaningless.

## Orphans

Every container carries the owning ICN's instance identity and process id. A restarted ICN removes
containers whose owner is gone, and leaves alone those whose recorded process is still alive. This
is not optional bookkeeping: the client's shutdown ends in `SIGKILL` after a short grace period, and
an unlabelled container would survive invisibly holding tens of gigabytes.

## Failures name their cause

EIM exits 0 or 1 and nothing else, so a container that fails to start is classified by matching its
log against EIM's stable messages. Each code distinguishes something the operator would do
differently: an unreachable daemon, an absent image, a missing checkout, a gated repository,
weights absent from an offline cache, or an unsupported CPU. A daemon-level oddity is retryable; a
configuration mistake is not.

## Acceptance criteria

- `grep -rn llama_cpp_2 inference/crates/` is empty.
- `bun icn:check-generated` passes with no diff.
- At most one instance is present in the snapshot at any time, and a stale instance cannot be leased.
- Every model in the table appears in the catalog; an unusable one carries a reason and a note.
- A container that dies during startup is detected from its state within one poll interval, not by
  waiting out the readiness budget.
- Stopping an instance removes its container, and a restarted ICN reaps what a killed predecessor
  left behind.
- The served model name is read from `GET /v1/models` rather than assumed from any identifier.

## Not yet implemented

Weight prefetching does not exist, and installing a model through the catalog is refused explicitly
rather than admitted and left to stall. A model absent from the mounted cache is downloaded by the
engine into the container, and is therefore downloaded again for each fresh container — accepted for
now. Only Qwen3's tool-call parser has been confirmed on real hardware.
