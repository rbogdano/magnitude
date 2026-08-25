//! Intel EIM container backend for ICN.
//!
//! ICN used to run llama.cpp in a child process it owned. Here it owns a Docker container
//! instead: an Intel Inference Microservices image that resolves a validated vLLM profile at
//! startup and serves an OpenAI-compatible API on port 8000. The seams ICN already had are
//! unchanged — `ModelInstanceController` for lifecycle, `CompletionBackend` for inference —
//! so everything above this crate keeps working against the same generated HTTP contract.
//!
//! Module map:
//!   - `acquisition` — installing a model: the serving image, then the weights
//!   - `backend`  — CompletionBackend over the container's chat-completions endpoint
//!   - `assessor` — `/v1/models/assess`: whether the host can serve a model, and how fast
//!   - `catalog`  — the model catalog ICN publishes; ACN cannot start without it
//!   - `controller` — ModelInstanceController: one resident container, load/stop/lease
//!   - `package`  — package identity derived from the serving image rather than a file path
//!   - `env_contract` — the INFERENCE_* launch contract, including the tool-call parser
//!   - `docker`  — the container driver: `docker` CLI invocations, never scraping human output
//!   - `estimate` — model fit from catalog geometry, replacing the native `common/fit` planner
//!   - `memory`   — live system-memory sampling and the admission/eviction thresholds
//!   - `properties` — ModelProperties synthesis for a container-backed model
//!   - `perf`     — estimated decode throughput, a memory-bandwidth roofline
//!   - `readiness` — waiting for a container to start serving, and reading its served name
//!   - `request`  — ChatRequest to a vLLM chat-completions body
//!   - `sse`      — server-sent event framing and OpenAI chat-chunk decoding
//!   - `weights`  — fetching a repository into EIM's Local Directory layout

pub mod acquisition;
pub mod assessor;
pub mod backend;
pub mod catalog;
pub mod controller;
pub mod docker;
pub mod env_contract;
pub mod estimate;
pub mod memory;
pub mod package;
pub mod perf;
pub mod properties;
pub mod readiness;
pub mod request;
pub mod sse;
pub mod weights;
