# Verifying the container backend by hand

Automated coverage stops at the ICN boundary: Rust unit tests, the two container integration
suites, and the zero-diff contract check. Above that the CLI is a TUI whose headless mode is
disabled upstream, so bringing the daemon up is a manual step.

An earlier version of this directory shipped a probe script that started the daemon and polled for
readiness. It was removed after three attempts: it could not locate the daemon's ephemeral port
reliably and reported "not ready" for a daemon that was in fact ready. A check that produces false
negatives is worse than no check, because its failures are indistinguishable from real ones.

## Bring the daemon up

```sh
bun run dev:version          # a development version selects the development ICN installation
bun run icn:build            # stages inference/target/development
HOME=$(mktemp -d) bun run packages/acn/src/serve.ts
```

The separate home directory keeps a check from touching real sessions or models. The daemon binds
an ephemeral loopback port and does not announce it, so find it and ask:

```sh
ss -tlnp | grep bun
curl -s http://127.0.0.1:<port>/health | python3 -m json.tool
```

Ready looks like this, and reaching it is the whole point:

```json
{ "service": "magnitude-acn", "state": { "_tag": "Ready" } }
```

`Ready` is reachable only if ICN was resolved, its identity, API version and capabilities were
verified, it reported its startup record, its `/health` identity matched, and every model surface
the client's layer graph builds over answered. It is not a liveness check; it is the container
backend's integration test.

## Check ICN on its own

```sh
inference/target/development/bin/magnitude-icn doctor
inference/target/development/bin/magnitude-icn serve --bind 127.0.0.1:0 --instance-id probe
```

Then, against the origin printed in `MAGNITUDE_ICN_READY`:

```sh
for path in /health /v1/hardware /v1/models /v1/models/installed \
            /v1/models/catalog /v1/models/downloads /v1/models/instances; do
  printf '%-28s %s\n' "$path" "$(curl -s -o /dev/null -w '%{http_code}' "$ORIGIN$path")"
done
```

All seven must return 200. A 500 means a collaborator is missing from `AppState` and the daemon
will refuse to become ready — which is how the catalog, installed-packages and downloads surfaces
came to be implemented in the first place.

## Serve a model end to end

Needs a built serving image and a model table. From an EIM checkout:

```sh
docker build -f docker/Dockerfile.inference-xeon-base \
  --build-arg PARENT_REGISTRY=docker.io \
  --build-arg PARENT_REPOSITORY=vllm/vllm-openai-cpu \
  --build-arg PARENT_TAG=v0.26.0 -t magnitude-eim-xeon-base:v1 .

docker build -f docker/Dockerfile.inference \
  --build-arg BASE_IMAGE=magnitude-eim-xeon-base:v1 \
  --build-arg ACCELERATOR_FAMILY=xeon \
  --build-arg ORG=Qwen --build-arg MODEL=Qwen3-8B \
  -t magnitude-eim-xeon-qwen-qwen3-8b:v1 .
```

Then start ICN with `--eim-catalog <table.json>`. Four things this will not forgive, all of them
learned by watching a real vLLM refuse to start:

- **`gpu-memory-utilization`** must be set. On the CPU backend that flag controls CPU memory and
  defaults to 0.92 of every NUMA node, which fails on a host doing anything at all. The launch
  contract derives it, so this only matters when running a container by hand.
- **`--shm-size`** must exceed Docker's 64 MB default whenever tensor parallelism is used. Without
  it vLLM's workers cannot broadcast and the only symptom is a gloo "connection closed by peer".
- **`--cap-add SYS_NICE`** is what lets vLLM bind worker memory to a NUMA node.
- **Proxy variables** must reach the container on a network without direct internet access.
  Without them the weight download fails in a way that reads as a missing model.

Weights are not preserved between containers unless the engine's Hugging Face cache is mounted;
that is accepted behavior for now, so expect a re-download on every fresh container.
