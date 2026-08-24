# Magnitude ICN

The Inference Control Node. It decides which model to serve, owns that model's lifecycle, and
carries chat between the agent and the engine.

The engine is not in this process. Inference runs in an Intel EIM container: a Docker image that
matches the host to the model, resolves a validated vLLM profile at startup, and serves an
OpenAI-compatible API on port 8000. ICN starts and stops that container and speaks HTTP to it.

    crates/icn-contracts   transport- and backend-neutral contracts
    crates/icn-api         the HTTP and OpenAPI boundary
    crates/icn-eim         the container backend: Docker, vLLM transport, model fit
    crates/icn-hardware    host memory policy and CPU discovery
    crates/icn-server      composition root
    crates/icn-utils

`icn-contracts` and `icn-api` know nothing about containers. Everything specific to EIM sits behind
the two traits the API already defines: `CompletionBackend` for inference, `ModelInstanceController`
for lifecycle. That separation is why swapping the engine left the client stack untouched.

`crates/icn-models` is excluded from the workspace. It was built for a GGUF inventory on local disk
and needs retargeting to whole safetensors repositories before it can come back.

## Pins

`eim-pin.toml` records the EIM revision serving images are built from and the vLLM base image those
layers sit on. Both feed `eim_build()`, which the client compares three times during startup, so a
mismatched installation is rejected by an invariant that already existed for this class of problem.

There is no native source to pin. A base-image change alters accepted engine arguments and
tool-call parser names, which is why it invalidates a prepared installation exactly as a bindings
change once did.

## First five minutes

    bun run dev:version          # a development version selects the development installation
    bun icn:build                # builds the binary and stages target/development
    bun icn:doctor               # is this host able to serve at all?

`doctor` answers the question that matters before anything else: whether the Docker daemon is
reachable, and what the host looks like. On a dual-socket machine it reports two NUMA nodes and
therefore a maximum tensor-parallel size of two, which is the fact that decides which EIM serving
profiles are usable.

    bun icn:dev                  # deterministic in-memory backend, no daemon needed
    bun icn:serve                # build and serve for real

Serving a model needs a built image and the model table:

    magnitude-icn serve --eim-catalog eim/models.json --eim-source ~/eim

With `--eim-source` a missing image is built from that EIM checkout; with `--eim-registry` it is
pulled instead, which is the recommended production shape because a build is multi-gigabyte and
multi-minute. With neither, a missing image is reported rather than fetched: starting a large build
unasked on an operator's host is not a reasonable default.

Installing a model prepares that image and then fetches its weights from Hugging Face into the
layout EIM reads, so export proxy variables before starting on a network without direct access —
the daemon's proxy configuration does not reach this process, and the resulting timeout reads like a
Hugging Face outage rather than missing configuration.

EIM publishes no images of its own — its CI builds with `push: false` — so one of those two is
required.

## Build and verification

    cargo test --manifest-path Cargo.toml --workspace
    cargo clippy --manifest-path Cargo.toml --workspace --all-targets -- -D warnings
    bun icn:check-generated      # the generated client protocol must not drift

Integration suites need a real daemon and skip without one:

    docker build -t magnitude-eim-stub:test eim/stub
    MAGNITUDE_EIM_STUB_IMAGE=magnitude-eim-stub:test cargo test -p icn-eim --test container_lifecycle
    MAGNITUDE_EIM_SOURCE=~/eim cargo test -p icn-eim --test image_resolution

## Testing philosophy

Anything decidable without a daemon is a unit test: argv construction, chunk decoding, the memory
formula, failure classification. That keeps the suite runnable on a laptop, which matters because
the real serving path is not — `vllm/vllm-openai-cpu` needs AVX-512, and a model load takes minutes
where it works at all.

`eim/stub` stands in for a serving container so the lifecycle can be exercised anyway. It reports a
served model name unrelated to `INFERENCE_MODEL_ID` on purpose: vLLM validates requests against its
own served name, so code that assumes the two agree fails loudly here rather than in production.

What cannot be automated is written down instead. `eim/VERIFY.md` lists the manual checks, including
four things a hand-run container will not forgive. A probe script that tried to automate the daemon
check was removed after it reported "not ready" for a daemon that was ready: a check whose failures
are indistinguishable from real ones is worse than none.
