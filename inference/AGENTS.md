# Inference development instructions

Inference does not run in this process. It runs in an Intel EIM container: a Docker image that
resolves a validated vLLM profile at startup and serves an OpenAI-compatible API. ICN's job is to
decide which model to serve, start and stop the container, and carry chat over HTTP.

The llama.cpp fork this directory used to maintain is gone, along with the native planner, the
per-accelerator feature graph, and the parity harness. `grep -rn llama_cpp_2 crates/` returns
nothing, and it should stay that way.

## Where the boundaries are

`icn-contracts` is the contract layer and has no backend dependencies. `icn-api` owns the HTTP and
OpenAPI surface and depends only on contracts. Neither should learn anything container-specific:
that is what `icn-eim` is for, behind the two traits the API already defines — `CompletionBackend`
for inference and `ModelInstanceController` for lifecycle.

`icn-server` is wiring. If it grows logic, that logic probably belongs in `icn-eim`.

## The contract must not move

The generated TypeScript protocol and `openapi.json` are derived from `icn-api`. The client stack
depends on them, and the whole point of this design is that it keeps working unchanged.

    bun icn:check-generated    # must pass with no diff

Adding a field to any `ToSchema` type, or changing an enum variant, breaks this. When a contract
change is genuinely needed, make it deliberately and regenerate in the same commit — never as a
side effect.

## Claims about a container need evidence

Four things this fork assumed and got wrong, each discovered only by starting a real vLLM:

- vLLM's CPU backend reserves a fraction of each NUMA node, not the model's size, and defaults to
  0.92 — which fails on a host doing anything at all.
- Docker's default 64 MB of shared memory is far too little for tensor parallelism, and the only
  symptom is a gloo "connection closed by peer".
- NUMA binding needs `CAP_SYS_NICE`.
- `INFERENCE_CACHE_PATH` is consulted only for weights already laid out there.

Two of those contradicted what EIM's documentation implied. Absence from the documentation is not
evidence. Before asserting that a container needs no particular flag, capability or mount, start
one and watch.

Unit tests cover argv construction, decoding, and the memory formula. Anything that depends on a
daemon belongs in an integration test gated on an environment variable, so the suite still runs on
a machine without Docker. See `eim/VERIFY.md` for what cannot be automated and why.

## Model data is fetched, never invented

The memory estimate needs six geometry numbers per model that EIM's `metadata.yaml` does not
record. They are fetched from each model's Hugging Face `config.json`, and weight sizes from its
safetensors index — measured rather than derived, because a published checkpoint is not always
bf16. `eim/generate-models.py` states the provenance of every field.

A guessed layer count produces an estimate that is confidently wrong, which is worse than an
absent one. If a number cannot be obtained, say so rather than filling it in.

## Capability claims are load-bearing

`toolCallParser` decides whether the catalog presents a model as usable. A wrong entry does not
produce an error: vLLM returns tool calls as prose and the agent loop falls apart silently. Only
Qwen3's parser has been confirmed on real hardware. Treat the rest as unverified, and prefer
marking a model unusable over claiming support that has not been observed.
