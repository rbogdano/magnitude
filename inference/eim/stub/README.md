# EIM stub container

A stand-in for an Intel EIM serving container, used to exercise ICN's container lifecycle,
readiness polling, and chat transport without running vLLM.

It exists because the real path is not testable on a developer machine: `vllm/vllm-openai-cpu`
needs AVX-512, and even where that is available a model load takes minutes. The stub answers in
milliseconds and can be told to fail in specific ways.

Build it:

```sh
docker build -t magnitude-eim-stub:test inference/eim/stub
```

Then run the container integration tests:

```sh
MAGNITUDE_EIM_STUB_IMAGE=magnitude-eim-stub:test \
  cargo test --manifest-path inference/Cargo.toml -p icn-eim --test container_lifecycle
```

## Deliberate infidelity

The stub reports a served model name unrelated to `INFERENCE_MODEL_ID`. vLLM validates requests
against its own `served_model_name` rather than the Hugging Face identifier, and EIM's own test
helper carries a warning comment about the mismatch. Reporting a different name means any code
that assumes the two are the same fails loudly here rather than in production.

## Knobs

| Variable | Effect |
|---|---|
| `STUB_SLOW_START` | seconds of 503 from `/health` before reporting ready |
| `STUB_CRASH_AFTER` | exit 17 this many seconds after start |
| `STUB_FAIL_MESSAGE` | print to stderr and exit 1 immediately, to exercise log classification against EIM's real failure strings |
| `STUB_REASONING` | emit `reasoning_content` deltas before the answer |
| `STUB_TOOL_CALL` | emit an incremental tool call instead of prose |
