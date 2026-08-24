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

- **`gpu-memory-utilization`** must be set, and it is subtler than its name. On the CPU backend it
  is a fraction of the *container's* memory limit, not of a NUMA node; every tensor-parallel worker
  claims that fraction independently; and vLLM checks it against memory currently available rather
  than against the limit. Running a container by hand with `--memory L` and a fraction picked for a
  NUMA node gets you either `Available memory on node 0 ... is less than desired CPU memory
  utilization` or a key-value cache far too small to hold the context. The launch contract derives
  the limit and the fraction together, so this only matters by hand.
- **`--shm-size`** must exceed Docker's 64 MB default whenever tensor parallelism is used. Without
  it vLLM's workers cannot broadcast and the only symptom is a gloo "connection closed by peer".
- **`--cap-add SYS_NICE`** is what lets vLLM bind worker memory to a NUMA node.
- **Proxy variables** must reach whatever downloads. Magnitude fetches weights in-process, so the
  variables have to be in ICN's own environment — the Docker daemon's proxy configuration does not
  reach it, and the resulting timeout reads like a Hugging Face outage.

## Installing a model

Selecting a model should install and then serve it. Check the whole path rather than the ends:

    curl -s -XPOST localhost:8080/v1/models/catalog/reconcile \
      -H 'content-type: application/json' \
      -d '{"modelId":"qwen-qwen3-4b","variantId":"vllm-bf16:tp1"}'

The reply must be `DownloadAdmitted` and return immediately — a reply that blocks for the length of
the download is the bug this shape exists to prevent. Then `GET /v1/models/downloads` should walk
`resolving` (the image) into `downloading` with byte counts that move and a plausible transfer rate,
and finish `Completed`. Afterwards `GET /v1/models` must report the model `Installed`, and the
weights must be on disk as a plain repository copy at `<cache>/eim/model-cache/<org>/<model>/`.

Two things worth confirming by eye, because both were wrong once. The container log should say
`Found model in local directory format`, which is what proves the prefetch is being used rather than
a second download happening invisibly inside the container. And a completion should stream through
`POST /v1/chat/completions` — not just reach `Ready` — because the load path and the inference path
fail independently, and a model that loads but cannot stream looks healthy in the snapshot.
