//! Waiting for a container to start serving.
//!
//! Two details decide whether this behaves well. First, the container's state is checked on
//! every iteration, not just the HTTP endpoint: a container that crashed during vLLM startup
//! would otherwise burn the entire multi-minute deadline before reporting anything, and EIM's
//! own integration harness guards against exactly that. Second, readiness is a 200 status with
//! an *empty* body — checking for content would never succeed.
//!
//! Once healthy, the served model name is read from `GET /v1/models` rather than assumed. vLLM
//! validates requests against its own `served_model_name`, and EIM's test helper carries a
//! comment warning about this mismatch.

use std::time::Duration;

use futures_util::future::BoxFuture;
use serde_json::Value;

use crate::docker::inspect::ContainerState;

/// How long readiness may take, and how often to look.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadinessConfig {
    pub poll_interval: Duration,
    /// Total budget. The base image's `HEALTHCHECK --start-period=60s` is misleading: EIM's own
    /// harness allows 420 seconds, and 300 for a 0.6B model on a warm cache. A large
    /// mixture-of-experts model loading from disk on CPU can legitimately take much longer.
    pub deadline: Duration,
}

impl Default for ReadinessConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(2),
            deadline: Duration::from_secs(600),
        }
    }
}

/// Why waiting ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadinessOutcome {
    Ready {
        /// What the server calls the model. Requests must use this, not the catalog identifier.
        served_model_name: String,
        elapsed: Duration,
    },
    /// The container stopped before it became healthy.
    ContainerExited {
        state: ContainerState,
        elapsed: Duration,
    },
    /// Still not healthy when the budget ran out, with the container still running.
    TimedOut { elapsed: Duration },
}

/// The container's own liveness, separate from whether it serves yet.
pub trait ContainerWatch: Send + Sync {
    /// `None` when the container no longer exists.
    fn state(&self) -> BoxFuture<'_, Result<Option<ContainerState>, String>>;
}

/// The served HTTP surface.
pub trait ServerProbe: Send + Sync {
    /// The status code of `GET /health`, or an error when the socket is not accepting yet.
    fn health(&self) -> BoxFuture<'_, Result<u16, String>>;
    /// The raw body of `GET /v1/models`.
    fn models(&self) -> BoxFuture<'_, Result<String, String>>;
}

/// Reads `data[0].id` out of an OpenAI `GET /v1/models` body.
///
/// Returns `None` rather than failing when the shape is unexpected, so the caller can fall back
/// to the catalog identifier and still serve. A wrong name produces a clear 404 on the first
/// request, which is far easier to diagnose than a refusal to start.
#[must_use]
pub fn parse_served_model_name(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body.trim()).ok()?;
    value
        .get("data")?
        .as_array()?
        .first()?
        .get("id")?
        .as_str()
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
}

/// Polls until the container serves, dies, or the deadline passes.
///
/// `on_tick` is called once per iteration with the elapsed time so the caller can emit load
/// progress. It is deliberately not given a fraction: only the caller knows the learned
/// duration prior to divide by.
pub async fn await_ready(
    watch: &dyn ContainerWatch,
    probe: &dyn ServerProbe,
    config: ReadinessConfig,
    mut on_tick: impl FnMut(Duration),
) -> ReadinessOutcome {
    let mut elapsed = Duration::ZERO;

    loop {
        // Liveness first. A container that has already exited must short-circuit rather than
        // wait out the deadline on an endpoint that will never answer.
        match watch.state().await {
            Ok(Some(state)) if state.status.is_terminal() => {
                return ReadinessOutcome::ContainerExited { state, elapsed };
            }
            Ok(None) => {
                // Removed out of band. Report it as an exit with no detail rather than
                // pretending the wait can continue.
                return ReadinessOutcome::ContainerExited {
                    state: ContainerState {
                        status: crate::docker::inspect::ContainerStatus::Unknown(
                            "removed".to_owned(),
                        ),
                        exit_code: 0,
                        oom_killed: false,
                        error: Some("container no longer exists".to_owned()),
                    },
                    elapsed,
                };
            }
            // A failed inspect is transient: the daemon may be briefly busy. Keep waiting.
            Ok(Some(_)) | Err(_) => {}
        }

        // 200 with an empty body is the ready signal; any other status means not yet.
        if probe.health().await == Ok(200) {
            let served_model_name = match probe.models().await {
                Ok(body) => parse_served_model_name(&body),
                Err(_) => None,
            };
            if let Some(served_model_name) = served_model_name {
                return ReadinessOutcome::Ready {
                    served_model_name,
                    elapsed,
                };
            }
            // Healthy but the model list is not answering yet. vLLM briefly serves /health
            // before its OpenAPI routes are mounted, so this is worth another iteration
            // rather than a failure.
        }

        if elapsed >= config.deadline {
            return ReadinessOutcome::TimedOut { elapsed };
        }
        tokio::time::sleep(config.poll_interval).await;
        elapsed = elapsed.saturating_add(config.poll_interval);
        on_tick(elapsed);
    }
}

/// Container liveness through the `docker` CLI.
pub struct DockerContainerWatch {
    docker: crate::docker::DockerCli,
    container_name: String,
}

impl DockerContainerWatch {
    #[must_use]
    pub fn new(docker: crate::docker::DockerCli, container_name: impl Into<String>) -> Self {
        Self {
            docker,
            container_name: container_name.into(),
        }
    }
}

impl ContainerWatch for DockerContainerWatch {
    fn state(&self) -> BoxFuture<'_, Result<Option<ContainerState>, String>> {
        Box::pin(async move {
            self.docker
                .container_state(&self.container_name)
                .await
                .map_err(|error| error.to_string())
        })
    }
}

/// The served surface over HTTP.
pub struct HttpServerProbe {
    client: reqwest::Client,
    endpoint: String,
}

impl HttpServerProbe {
    /// `endpoint` is the container's base URL, e.g. `http://127.0.0.1:43117`.
    pub fn new(endpoint: impl Into<String>) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            // The container is published on loopback; a corporate proxy must not intercept it.
            .no_proxy()
            // Short per-probe timeout: the loop, not the request, owns the overall budget.
            .timeout(Duration::from_secs(5))
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            client,
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
        })
    }
}

impl ServerProbe for HttpServerProbe {
    fn health(&self) -> BoxFuture<'_, Result<u16, String>> {
        Box::pin(async move {
            self.client
                .get(format!("{}/health", self.endpoint))
                .send()
                .await
                .map(|response| response.status().as_u16())
                .map_err(|error| error.to_string())
        })
    }

    fn models(&self) -> BoxFuture<'_, Result<String, String>> {
        Box::pin(async move {
            let response = self
                .client
                .get(format!("{}/v1/models", self.endpoint))
                .send()
                .await
                .map_err(|error| error.to_string())?;
            response.text().await.map_err(|error| error.to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::docker::inspect::ContainerStatus;

    struct ScriptedWatch {
        states: Mutex<Vec<Option<ContainerState>>>,
        running: Option<ContainerState>,
    }

    impl ScriptedWatch {
        fn always_running() -> Self {
            Self {
                states: Mutex::new(Vec::new()),
                running: Some(running_state()),
            }
        }

        fn then(states: Vec<Option<ContainerState>>) -> Self {
            Self {
                states: Mutex::new(states),
                running: Some(running_state()),
            }
        }
    }

    impl ContainerWatch for ScriptedWatch {
        fn state(&self) -> BoxFuture<'_, Result<Option<ContainerState>, String>> {
            Box::pin(async move {
                let mut states = self.states.lock().expect("lock");
                if states.is_empty() {
                    Ok(self.running.clone())
                } else {
                    Ok(states.remove(0))
                }
            })
        }
    }

    fn running_state() -> ContainerState {
        ContainerState {
            status: ContainerStatus::Running,
            exit_code: 0,
            oom_killed: false,
            error: None,
        }
    }

    fn exited_state(exit_code: i64, oom_killed: bool) -> ContainerState {
        ContainerState {
            status: ContainerStatus::Exited,
            exit_code,
            oom_killed,
            error: None,
        }
    }

    struct ScriptedProbe {
        health_codes: Mutex<Vec<Result<u16, String>>>,
        models_body: String,
        health_calls: AtomicUsize,
    }

    impl ScriptedProbe {
        fn new(health_codes: Vec<Result<u16, String>>, models_body: &str) -> Self {
            Self {
                health_codes: Mutex::new(health_codes),
                models_body: models_body.to_owned(),
                health_calls: AtomicUsize::new(0),
            }
        }
    }

    impl ServerProbe for ScriptedProbe {
        fn health(&self) -> BoxFuture<'_, Result<u16, String>> {
            Box::pin(async move {
                self.health_calls.fetch_add(1, Ordering::Relaxed);
                let mut codes = self.health_codes.lock().expect("lock");
                if codes.is_empty() {
                    Err("connection refused".to_owned())
                } else {
                    codes.remove(0)
                }
            })
        }

        fn models(&self) -> BoxFuture<'_, Result<String, String>> {
            Box::pin(async move { Ok(self.models_body.clone()) })
        }
    }

    const MODELS_BODY: &str =
        r#"{"object":"list","data":[{"id":"Qwen/Qwen3-8B","object":"model"}]}"#;

    fn fast() -> ReadinessConfig {
        ReadinessConfig {
            poll_interval: Duration::from_millis(1),
            deadline: Duration::from_millis(50),
        }
    }

    #[tokio::test]
    async fn reports_ready_with_the_served_model_name() {
        let outcome = await_ready(
            &ScriptedWatch::always_running(),
            &ScriptedProbe::new(vec![Ok(503), Ok(503), Ok(200)], MODELS_BODY),
            fast(),
            |_| {},
        )
        .await;

        match outcome {
            ReadinessOutcome::Ready {
                served_model_name, ..
            } => assert_eq!(served_model_name, "Qwen/Qwen3-8B"),
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn short_circuits_when_the_container_exits() {
        // The whole point: a crash must not wait out the deadline.
        let outcome = await_ready(
            &ScriptedWatch::then(vec![Some(running_state()), Some(exited_state(1, false))]),
            &ScriptedProbe::new(vec![Ok(503), Ok(503), Ok(503)], MODELS_BODY),
            ReadinessConfig {
                poll_interval: Duration::from_millis(1),
                deadline: Duration::from_secs(600),
            },
            |_| {},
        )
        .await;

        match outcome {
            ReadinessOutcome::ContainerExited { state, elapsed } => {
                assert_eq!(state.exit_code, 1);
                // Detected on the second iteration, nowhere near the 600 second budget.
                assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
            }
            other => panic!("expected ContainerExited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn surfaces_an_oom_kill_as_an_exit() {
        let outcome = await_ready(
            &ScriptedWatch::then(vec![Some(exited_state(137, true))]),
            &ScriptedProbe::new(vec![], MODELS_BODY),
            fast(),
            |_| {},
        )
        .await;

        match outcome {
            ReadinessOutcome::ContainerExited { state, .. } => {
                // This is how a wrong RAM estimate becomes a legible LowMemory failure.
                assert!(state.oom_killed);
                assert_eq!(state.exit_code, 137);
            }
            other => panic!("expected ContainerExited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn treats_a_vanished_container_as_an_exit() {
        let outcome = await_ready(
            &ScriptedWatch::then(vec![None]),
            &ScriptedProbe::new(vec![], MODELS_BODY),
            fast(),
            |_| {},
        )
        .await;

        assert!(matches!(outcome, ReadinessOutcome::ContainerExited { .. }));
    }

    #[tokio::test]
    async fn keeps_waiting_through_a_transient_inspect_failure() {
        struct FlakyWatch {
            calls: AtomicUsize,
        }
        impl ContainerWatch for FlakyWatch {
            fn state(&self) -> BoxFuture<'_, Result<Option<ContainerState>, String>> {
                Box::pin(async move {
                    // First look fails, as a briefly busy daemon would.
                    if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                        Err("daemon busy".to_owned())
                    } else {
                        Ok(Some(running_state()))
                    }
                })
            }
        }

        let outcome = await_ready(
            &FlakyWatch {
                calls: AtomicUsize::new(0),
            },
            &ScriptedProbe::new(vec![Err("refused".to_owned()), Ok(200)], MODELS_BODY),
            fast(),
            |_| {},
        )
        .await;

        assert!(matches!(outcome, ReadinessOutcome::Ready { .. }));
    }

    #[tokio::test]
    async fn keeps_waiting_when_health_passes_before_the_model_list_answers() {
        // vLLM answers /health before its OpenAPI routes are mounted.
        struct LateModels {
            calls: AtomicUsize,
        }
        impl ServerProbe for LateModels {
            fn health(&self) -> BoxFuture<'_, Result<u16, String>> {
                Box::pin(async { Ok(200) })
            }
            fn models(&self) -> BoxFuture<'_, Result<String, String>> {
                Box::pin(async move {
                    if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                        Ok(r#"{"object":"list","data":[]}"#.to_owned())
                    } else {
                        Ok(MODELS_BODY.to_owned())
                    }
                })
            }
        }

        let outcome = await_ready(
            &ScriptedWatch::always_running(),
            &LateModels {
                calls: AtomicUsize::new(0),
            },
            fast(),
            |_| {},
        )
        .await;

        assert!(matches!(outcome, ReadinessOutcome::Ready { .. }));
    }

    #[tokio::test]
    async fn times_out_while_the_container_is_still_running() {
        let outcome = await_ready(
            &ScriptedWatch::always_running(),
            &ScriptedProbe::new(vec![], MODELS_BODY),
            fast(),
            |_| {},
        )
        .await;

        match outcome {
            ReadinessOutcome::TimedOut { elapsed } => {
                assert!(elapsed >= Duration::from_millis(50), "{elapsed:?}");
            }
            other => panic!("expected TimedOut, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reports_progress_on_every_iteration() {
        let ticks = Mutex::new(Vec::new());
        let _ = await_ready(
            &ScriptedWatch::always_running(),
            &ScriptedProbe::new(vec![], MODELS_BODY),
            fast(),
            |elapsed| ticks.lock().expect("lock").push(elapsed),
        )
        .await;

        let ticks = ticks.into_inner().expect("lock");
        assert!(ticks.len() > 5, "expected repeated progress, got {ticks:?}");
        // Monotonic, so a caller can drive a non-regressing progress bar.
        assert!(ticks.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn parses_the_served_model_name() {
        assert_eq!(
            parse_served_model_name(MODELS_BODY).as_deref(),
            Some("Qwen/Qwen3-8B")
        );
        // A deliberately different served name is exactly the case this exists for.
        assert_eq!(
            parse_served_model_name(r#"{"data":[{"id":"served-alias"}]}"#).as_deref(),
            Some("served-alias")
        );
    }

    #[test]
    fn returns_none_for_shapes_it_cannot_read() {
        for body in [
            "",
            "not json",
            r#"{"data":[]}"#,
            r#"{"data":[{}]}"#,
            r#"{"data":[{"id":""}]}"#,
            r#"{"object":"list"}"#,
        ] {
            assert_eq!(parse_served_model_name(body), None, "body: {body}");
        }
    }
}
