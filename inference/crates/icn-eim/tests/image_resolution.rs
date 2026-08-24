//! Builds a serving image from a real EIM checkout, against a real Docker daemon.
//!
//! Skipped unless `MAGNITUDE_EIM_SOURCE` points at a checkout. EIM publishes no images, so
//! building is the default path rather than an edge case, and it is the part of resolution that
//! cannot be checked by inspecting argv: the build arguments have to actually satisfy EIM's
//! Dockerfiles.
//!
//!   MAGNITUDE_EIM_SOURCE=~/eim cargo test -p icn-eim --test image_resolution

use std::path::PathBuf;

use icn_eim::docker::DockerCli;
use icn_eim::docker::image::{ImageOutcome, ImageResolver, ImageSource, ImageStage};

fn eim_source() -> Option<PathBuf> {
    std::env::var("MAGNITUDE_EIM_SOURCE")
        .ok()
        .map(PathBuf::from)
        .filter(|path| path.join("docker/Dockerfile.inference").is_file())
}

/// A tag no other test or operator will be holding, so removing it is safe.
const MODEL_IMAGE: &str = "magnitude-eim-itest-qwen-qwen3-8b:resolve";
const BASE_IMAGE: &str = "magnitude-eim-itest-base:resolve";

fn build_source(source: PathBuf) -> ImageSource {
    ImageSource::Build {
        eim_source: source,
        // From EIM's own assets/xeon/base/config.yaml.
        parent_registry: "docker.io".to_owned(),
        parent_repository: "vllm/vllm-openai-cpu".to_owned(),
        parent_tag: "v0.26.0".to_owned(),
        base_image: BASE_IMAGE.to_owned(),
    }
}

async fn remove(docker: &DockerCli, image: &str) {
    let _ = docker
        .invoke(&["rmi".to_owned(), "--force".to_owned(), image.to_owned()])
        .await;
}

#[tokio::test]
async fn builds_a_serving_image_from_an_eim_checkout() {
    let Some(source) = eim_source() else {
        eprintln!("skipped: set MAGNITUDE_EIM_SOURCE to an EIM checkout to run this");
        return;
    };
    let docker = DockerCli::default();
    remove(&docker, MODEL_IMAGE).await;

    let resolver = ImageResolver::new(docker.clone(), build_source(source));
    let mut stages = Vec::new();
    let outcome = resolver
        .resolve(MODEL_IMAGE, "Qwen/Qwen3-8B", |stage| stages.push(stage))
        .await;

    let built = match outcome {
        Ok(outcome) => outcome,
        Err(error) => {
            remove(&docker, MODEL_IMAGE).await;
            panic!("resolution failed: {error}");
        }
    };

    // Present afterwards, with a digest the local-model protocol can carry.
    let summary = docker
        .image_summary(MODEL_IMAGE)
        .await
        .expect("inspect succeeds")
        .expect("the image exists");
    remove(&docker, MODEL_IMAGE).await;

    assert!(matches!(built, ImageOutcome::Built(_)), "{built:?}");
    assert_eq!(summary.content_digest.len(), 64);
    assert!(summary.size_bytes > 0);
    // The model layer is thin: it copies profile YAML onto a shared base, so almost all of the
    // size comes from the vLLM parent.
    assert!(stages.contains(&ImageStage::BuildingModel), "{stages:?}");
    assert_eq!(stages.first(), Some(&ImageStage::Inspecting), "{stages:?}");
}

#[tokio::test]
async fn a_second_resolution_reuses_the_image_instead_of_rebuilding() {
    let Some(source) = eim_source() else {
        eprintln!("skipped: set MAGNITUDE_EIM_SOURCE to an EIM checkout to run this");
        return;
    };
    let docker = DockerCli::default();
    let resolver = ImageResolver::new(docker.clone(), build_source(source));

    let first = resolver
        .resolve(MODEL_IMAGE, "Qwen/Qwen3-8B", |_| {})
        .await
        .expect("first resolution builds");
    let mut stages = Vec::new();
    let second = resolver
        .resolve(MODEL_IMAGE, "Qwen/Qwen3-8B", |stage| stages.push(stage))
        .await
        .expect("second resolution reuses");

    remove(&docker, MODEL_IMAGE).await;

    assert!(matches!(first, ImageOutcome::Built(_)), "{first:?}");
    // The second must not rebuild: every model load resolves, and rebuilding each time would
    // add minutes to a load that should be instant.
    assert!(
        matches!(second, ImageOutcome::AlreadyPresent(_)),
        "{second:?}"
    );
    assert_eq!(stages, vec![ImageStage::Inspecting]);
    assert_eq!(
        first.summary().content_digest,
        second.summary().content_digest,
        "the same build must produce the same identity"
    );
}

#[tokio::test]
async fn refuses_to_build_a_model_that_eim_has_no_profiles_for() {
    let Some(source) = eim_source() else {
        eprintln!("skipped: set MAGNITUDE_EIM_SOURCE to an EIM checkout to run this");
        return;
    };
    let docker = DockerCli::default();
    let image = "magnitude-eim-itest-unknown:resolve";
    remove(&docker, image).await;

    // EIM's model Dockerfile copies assets/xeon/<org>/<model>/, so a model it does not ship
    // cannot be built. Failing here is better than producing an image that cannot resolve a
    // profile and only fails once a user tries to load it.
    let outcome = resolver_for(&docker, source)
        .resolve(image, "nobody/NotAModel-1B", |_| {})
        .await;

    remove(&docker, image).await;
    assert!(outcome.is_err(), "a model without profiles must not build");
}

fn resolver_for(docker: &DockerCli, source: PathBuf) -> ImageResolver {
    ImageResolver::new(docker.clone(), build_source(source))
}
