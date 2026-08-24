//! Intel EIM container backend for ICN.
//!
//! ICN used to run llama.cpp in a child process it owned. Here it owns a Docker container
//! instead: an Intel Inference Microservices image that resolves a validated vLLM profile at
//! startup and serves an OpenAI-compatible API on port 8000. The seams ICN already had are
//! unchanged — `ModelInstanceController` for lifecycle, `CompletionBackend` for inference —
//! so everything above this crate keeps working against the same generated HTTP contract.
//!
//! Module map:
//!   - `backend`  — CompletionBackend over the container's chat-completions endpoint
//!   - `docker`  — the container driver: `docker` CLI invocations, never scraping human output
//!   - `estimate` — model fit from catalog geometry, replacing the native `common/fit` planner
//!   - `memory`   — live system-memory sampling and the admission/eviction thresholds
//!   - `properties` — ModelProperties synthesis for a container-backed model
//!   - `request`  — ChatRequest to a vLLM chat-completions body
//!   - `sse`      — server-sent event framing and OpenAI chat-chunk decoding

pub mod backend;
pub mod docker;
pub mod estimate;
pub mod memory;
pub mod properties;
pub mod request;
pub mod sse;
