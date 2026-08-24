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

## Weights are fetched by Magnitude, not by the container

EIM resolves `INFERENCE_MODEL_ID` against `<cache>/<org>/<model>/` before asking the engine to
download, and calls that layout an explicit, pre-populated one. It is a plain copy of the
repository's files, so Magnitude populates it over ordinary HTTP and the container never reaches
the network.

That is where the progress bar comes from. Installing is the longest step in serving a model for
the first time, and the client's download surface is the only place that carries
`completedBytes`/`totalBytes`/`bytesPerSecond`. Letting the engine download instead leaves the
longest phase of the longest operation with nothing to show, no way to cancel, and a fresh
download for every container.

Selection is a denylist. A repository's non-weight files are kilobytes, so keeping an unrecognised
one costs nothing while omitting a needed one fails a load for no visible reason. What the denylist
is for is duplicate weights: `openai/gpt-oss-20b` publishes the same parameters three times and
`mistralai/Mistral-7B-Instruct-v0.2` twice, which is 27 GB and 15 GB of waste respectively.

Installed therefore means both halves — the serving image is present *and* the weights are
complete against a recorded manifest. Either alone yields a container that cannot serve.

## Memory is estimated, and vLLM's own control is separate

Fit is computed from six geometry numbers per model, not measured. Weights are charged at their
measured on-disk size when known, because a published checkpoint is not always bf16, and for a
mixture of experts every expert is resident even though few are active per token.

The estimate is not what governs the engine's allocation. That is `--gpu-memory-utilization`,
misleadingly named on the CPU backend, and it interacts with the container's memory ceiling in
three ways that were each established by watching a load fail:

- The fraction is of the **container's cgroup limit**, not of a NUMA node. Reading it as a node
  fraction while the cgroup is far smaller shrinks the real budget by the ratio between them.
- **Each** tensor-parallel worker claims that fraction independently, so the reservation across the
  container is `utilization x limit x ranks`. Ignoring the multiplication overcommits the cgroup.
- The check is against memory *currently available*, not against the ceiling, so the ceiling must
  exceed the workers' share by whatever the interpreter and EIM's launcher already hold.

Because the two multiply, they are derived together from one estimate and never set independently:
the ceiling is the estimate with its slack plus a fixed startup reserve, and the fraction hands the
ranks the part that is not the reserve.

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

Every container carries the owning ICN's instance identity and process id. Liveness of that pid is
what decides whether a container is abandoned — never the instance identity, which ACN keeps stable
across restarts, so a successor carries its predecessor's id and would otherwise adopt containers it
cannot use. The container name is derived from the same identity, so an adopted container also makes
the model unservable until someone removes it by hand.

Shutdown releases what this process owns, and that is a separate job from reaping: the reaper skips
containers whose owner is alive, and on the way out that owner is us. A clean exit that skipped it
would leave the whole model's memory held until some later ICN happened to start.

None of this is optional bookkeeping. The client's shutdown ends in `SIGKILL` after a short grace
period, so the boot-time sweep is the only thing standing between that and tens of gigabytes held by
an invisible container.

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
- A plain `serve` with no EIM flags publishes the full catalog, because that is how ACN starts ICN.
- A container that dies during startup is detected from its state within one poll interval, not by
  waiting out the readiness budget.
- Stopping an instance removes its container, and a restarted ICN reaps what a killed predecessor
  left behind.
- Selecting any listed, serveable model installs it and then serves it: the image is prepared, the
  weights are fetched with byte-accurate progress, and the model reaches Ready.
- Uninstalling reclaims the weights as well as the image.
- The completion callback is invoked outside any async context, and a test proves it by blocking in
  the callback the way the API handler does.
- The served model name is read from `GET /v1/models` rather than assumed from any identifier.

## The synchronous seam is bridged by a channel, not by blocking

`CompletionBackend::complete` is synchronous with an event callback while the transport is async.
The bridge is a bounded channel: the request runs as a task on the runtime and hands decoded chunks
to the synchronous caller, which is the thread that invokes the callback.

Not `Handle::block_on`, which is the obvious bridge and is wrong. It establishes an async context
on the calling thread, and the callback `icn-api` supplies performs a blocking send into the
client's event channel — which panics inside an async context. Every unit and container test passed
while this was broken, because their callbacks only appended to a vector. The suite now supplies a
callback that blocks exactly as the real one does.

## Not yet implemented

Two of the six serveable models have had their tool-call parser confirmed on real hardware, Qwen3
and Granite. The remaining four are still claims.

Granite is worth recording because it shows how the claim can be wrong in more than one direction.
`granite` extracts its calls correctly but leaks the `<tool_call>` marker into the visible message,
one token at a time; `hermes` keeps the message clean and *corrupts the arguments*, because Granite
3.2 wraps its call in a list that parser mis-slices. Corrupt arguments are the worse failure, so
`granite` is the entry and the leaked marker is filtered here instead. A parser that merely looks
plausible can be wrong in a way that only a real multi-step tool loop reveals.

Gated repositories need a Hugging Face credential with somewhere to store it and a licence
acceptance to surface.
