//! Integration test for the container driver against a real Docker daemon.
//!
//! Skipped unless `MAGNITUDE_EIM_TEST_IMAGE` names an image that can run a small HTTP server,
//! mirroring EIM's own convention of requiring an image name rather than hardcoding one. A
//! plain `python:3.12-slim` is enough — the point is to exercise `docker run`, the readiness
//! loop, state short-circuiting, and teardown against the real daemon, not to run vLLM.
//!
//!   MAGNITUDE_EIM_TEST_IMAGE=python:3.12-slim cargo test -p icn-eim --test container_readiness
//!
//! Every container it creates carries the ICN owner label and is removed on the way out, so a
//! failed run leaves nothing behind that the reaper would not find.

use std::collections::BTreeMap;
use std::time::Duration;

use icn_eim::docker::DockerCli;
use icn_eim::docker::cli::ContainerSpec;
use icn_eim::docker::naming::{ContainerLabels, OWNER_LABEL, OWNER_VALUE};
use icn_eim::readiness::{
    DockerContainerWatch, HttpServerProbe, ReadinessConfig, ReadinessOutcome, await_ready,
};

/// A server that answers `/health` with 503 for `--slow-start` seconds and then 200-empty, and
/// serves a `/v1/models` list under a deliberately different name than any catalog identifier.
///
/// The differing name is the assertion that matters: it proves the served name is read from the
/// server rather than assumed, which is the mistake EIM's own helper warns about.
const STUB_SERVER: &str = r#"
import http.server, json, os, sys, time
started = time.time()
slow = float(os.environ.get("STUB_SLOW_START", "0"))
crash_after = float(os.environ.get("STUB_CRASH_AFTER", "0"))

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if crash_after and time.time() - started > crash_after:
            os._exit(17)
        if self.path == "/health":
            ready = time.time() - started >= slow
            self.send_response(200 if ready else 503)
            self.send_header("content-length", "0")
            self.end_headers()
            return
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [
                {"id": "stub-served-name", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404); self.send_header("content-length", "0"); self.end_headers()

if crash_after:
    # Exit even with no request arriving, so a readiness loop sees a dead container.
    import threading
    threading.Timer(crash_after, lambda: os._exit(17)).start()

http.server.HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
"#;

fn test_image() -> Option<String> {
    std::env::var("MAGNITUDE_EIM_TEST_IMAGE")
        .ok()
        .filter(|image| !image.trim().is_empty())
}

fn labels(name: &str) -> Vec<String> {
    ContainerLabels {
        icn_instance_id: "integration-test".to_owned(),
        model_instance_id: name.to_owned(),
        configuration_id: "test-configuration".to_owned(),
        catalog_model_id: "test-model".to_owned(),
        eim_profile_id: "vllm-xeon-bf16-tp1".to_owned(),
        pid: std::process::id(),
    }
    .to_args()
}

/// Binds an ephemeral loopback port and releases it, the same way the launcher does.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind an ephemeral port");
    listener.local_addr().expect("read the bound port").port()
}

fn spec(
    image: &str,
    name: &str,
    port: u16,
    environment: BTreeMap<String, String>,
) -> ContainerSpec {
    ContainerSpec {
        name: name.to_owned(),
        image: image.to_owned(),
        host_port: port,
        container_port: 8000,
        labels: labels(name),
        environment,
        mounts: Vec::new(),
        memory_limit_bytes: None,
        stop_timeout_seconds: 5,
        command: vec!["python".to_owned(), "-c".to_owned(), STUB_SERVER.to_owned()],
    }
}

async fn teardown(docker: &DockerCli, name: &str) {
    let _ = docker.stop(name, 2).await;
    let _ = docker.remove(name).await;
}

#[tokio::test]
async fn reads_the_served_model_name_from_a_real_container() {
    let Some(image) = test_image() else {
        eprintln!("skipped: set MAGNITUDE_EIM_TEST_IMAGE to run this");
        return;
    };
    let docker = DockerCli::default();
    let name = "magnitude-eim-itest-ready";
    teardown(&docker, name).await;

    let port = free_port();
    // A slow start proves the loop actually polls rather than succeeding on the first look.
    let environment = BTreeMap::from([("STUB_SLOW_START".to_owned(), "4".to_owned())]);
    docker
        .run(&spec(&image, name, port, environment))
        .await
        .expect("container should start");

    let outcome = await_ready(
        &DockerContainerWatch::new(docker.clone(), name),
        &HttpServerProbe::new(format!("http://127.0.0.1:{port}")).expect("probe"),
        ReadinessConfig {
            poll_interval: Duration::from_millis(500),
            deadline: Duration::from_secs(60),
        },
        |_| {},
    )
    .await;

    teardown(&docker, name).await;

    match outcome {
        ReadinessOutcome::Ready {
            served_model_name,
            elapsed,
        } => {
            // Read from /v1/models, not assumed from any identifier we passed in.
            assert_eq!(served_model_name, "stub-served-name");
            // It waited for the 503 window rather than reporting ready immediately.
            assert!(elapsed >= Duration::from_secs(3), "{elapsed:?}");
        }
        other => panic!("expected Ready, got {other:?}"),
    }
}

#[tokio::test]
async fn short_circuits_on_a_container_that_dies_during_startup() {
    let Some(image) = test_image() else {
        eprintln!("skipped: set MAGNITUDE_EIM_TEST_IMAGE to run this");
        return;
    };
    let docker = DockerCli::default();
    let name = "magnitude-eim-itest-crash";
    teardown(&docker, name).await;

    let port = free_port();
    let environment = BTreeMap::from([
        // Never becomes healthy, and exits shortly after starting.
        ("STUB_SLOW_START".to_owned(), "9999".to_owned()),
        ("STUB_CRASH_AFTER".to_owned(), "3".to_owned()),
    ]);
    docker
        .run(&spec(&image, name, port, environment))
        .await
        .expect("container should start");

    let started = std::time::Instant::now();
    let outcome = await_ready(
        &DockerContainerWatch::new(docker.clone(), name),
        &HttpServerProbe::new(format!("http://127.0.0.1:{port}")).expect("probe"),
        ReadinessConfig {
            poll_interval: Duration::from_millis(500),
            // A generous budget: the test is that the crash is detected instead of waited out.
            deadline: Duration::from_secs(300),
        },
        |_| {},
    )
    .await;
    let wall_clock = started.elapsed();

    let logs = docker.logs_tail(name, 50).await.unwrap_or_default();
    teardown(&docker, name).await;

    match outcome {
        ReadinessOutcome::ContainerExited { state, .. } => {
            assert_eq!(state.exit_code, 17, "logs: {logs}");
            assert!(!state.oom_killed);
        }
        other => panic!("expected ContainerExited, got {other:?} (logs: {logs})"),
    }
    // Without the per-iteration state check this would have taken the full 300 seconds.
    assert!(wall_clock < Duration::from_secs(60), "{wall_clock:?}");
}

#[tokio::test]
async fn labels_every_container_so_the_reaper_can_find_it() {
    let Some(image) = test_image() else {
        eprintln!("skipped: set MAGNITUDE_EIM_TEST_IMAGE to run this");
        return;
    };
    let docker = DockerCli::default();
    let name = "magnitude-eim-itest-labels";
    teardown(&docker, name).await;

    let port = free_port();
    docker
        .run(&spec(&image, name, port, BTreeMap::new()))
        .await
        .expect("container should start");

    let owned = docker.list_owned().await.expect("list owned containers");
    let ours = owned.iter().find(|container| container.name == name);

    teardown(&docker, name).await;

    let ours = ours.expect("our container should carry the owner label");
    assert_eq!(ours.icn_instance_id.as_deref(), Some("integration-test"));
    assert_eq!(ours.pid, Some(std::process::id()));
    // A container belonging to this very process is never an orphan.
    assert!(!ours.is_orphan_of("integration-test", |_| false));
    // One belonging to a different, dead ICN is.
    assert!(ours.is_orphan_of("some-other-icn", |_| false));
    assert!(
        !OWNER_LABEL.is_empty() && !OWNER_VALUE.is_empty(),
        "owner label constants must be set"
    );
}

#[tokio::test]
async fn stop_and_remove_are_idempotent() {
    let Some(_) = test_image() else {
        eprintln!("skipped: set MAGNITUDE_EIM_TEST_IMAGE to run this");
        return;
    };
    let docker = DockerCli::default();

    // Teardown runs on failure paths too, so acting on an absent container must succeed rather
    // than mask the original error.
    docker
        .stop("magnitude-eim-itest-absent", 1)
        .await
        .expect("stopping an absent container is a success");
    docker
        .remove("magnitude-eim-itest-absent")
        .await
        .expect("removing an absent container is a success");
    assert!(
        docker
            .container_state("magnitude-eim-itest-absent")
            .await
            .expect("inspect should not error")
            .is_none()
    );
}
