//! Intel EIM container backend for ICN.
//!
//! ICN used to run llama.cpp in a child process it owned. Here it owns a Docker container
//! instead: an Intel Inference Microservices image that resolves a validated vLLM profile at
//! startup and serves an OpenAI-compatible API on port 8000. The seams ICN already had are
//! unchanged — `ModelInstanceController` for lifecycle, `CompletionBackend` for inference —
//! so everything above this crate keeps working against the same generated HTTP contract.
//!
//! Module map:
//!   - `docker`  — the container driver: `docker` CLI invocations, never scraping human output
//!   - `estimate` — model fit from catalog geometry, replacing the native `common/fit` planner

pub mod docker;
pub mod estimate;
