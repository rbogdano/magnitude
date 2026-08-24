//! Getting a serving image onto the host.
//!
//! EIM publishes none of its images -- its CI builds with `push: false` and Docker Hub has no
//! `intel/inference-xeon-*` -- so building locally is the default path rather than a fallback.
//! An operator who has pushed images to a private registry can pull instead, which is the
//! recommended production shape because a build is multi-gigabyte and multi-minute.
//!
//! The build is two-stage, matching EIM's own Dockerfiles: a base image carrying the runtime and
//! general profiles, then a thin per-model layer adding that model's validated profiles and
//! pinning `INFERENCE_MODEL_ID`. The base is shared, so only the second stage runs per model and
//! it copies a handful of YAML files.

use std::path::{Path, PathBuf};

use super::cli::{DockerCli, DockerError};
use super::inspect::ImageSummary;

/// Where a serving image comes from when it is not already present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSource {
    /// Pull from a registry that already holds the images.
    Registry,
    /// Build from a pinned EIM checkout.
    Build {
        /// Root of the EIM source tree, which is the Docker build context.
        eim_source: PathBuf,
        /// Registry host of the vLLM base image, from EIM's own `assets/xeon/base/config.yaml`.
        parent_registry: String,
        parent_repository: String,
        parent_tag: String,
        /// Tag for the intermediate base image this fork builds.
        base_image: String,
    },
    /// Neither pull nor build. Reports a missing image as a configuration problem, which is what
    /// an operator who pre-provisions images wants.
    PresentOnly,
}

/// How an image came to be available.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageOutcome {
    AlreadyPresent(ImageSummary),
    Pulled(ImageSummary),
    Built(ImageSummary),
}

impl ImageOutcome {
    #[must_use]
    pub fn summary(&self) -> &ImageSummary {
        match self {
            Self::AlreadyPresent(summary) | Self::Pulled(summary) | Self::Built(summary) => summary,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("docker failed: {0}")]
    Docker(#[from] DockerError),
    #[error(
        "serving image {image} is not present and this host is configured not to build or pull it"
    )]
    Absent { image: String },
    #[error("EIM source tree is missing at {}; set MAGNITUDE_EIM_SOURCE to a checkout", path.display())]
    MissingSource { path: PathBuf },
    #[error("model identifier {0:?} is not in org/model form")]
    MalformedModelId(String),
    #[error("docker reported no image after a successful {stage}")]
    NoImageAfter { stage: &'static str },
}

/// Splits a Hugging Face identifier into the organization and model names EIM's build expects.
pub fn split_model_id(canonical_name: &str) -> Result<(&str, &str), ImageError> {
    match canonical_name.split_once('/') {
        Some((org, model)) if !org.is_empty() && !model.is_empty() => Ok((org, model)),
        _ => Err(ImageError::MalformedModelId(canonical_name.to_owned())),
    }
}

/// Which stage of resolution is running, so the caller can report progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageStage {
    Inspecting,
    Pulling,
    BuildingBase,
    BuildingModel,
}

impl ImageStage {
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            Self::Inspecting => "checking for the serving image",
            Self::Pulling => "pulling the serving image",
            Self::BuildingBase => "building the EIM base image",
            Self::BuildingModel => "building the model image",
        }
    }
}

pub struct ImageResolver {
    docker: DockerCli,
    source: ImageSource,
}

impl ImageResolver {
    #[must_use]
    pub fn new(docker: DockerCli, source: ImageSource) -> Self {
        Self { docker, source }
    }

    /// Arguments for `docker build` of the shared base image.
    #[must_use]
    pub fn base_build_args(
        eim_source: &Path,
        parent_registry: &str,
        parent_repository: &str,
        parent_tag: &str,
        base_image: &str,
    ) -> Vec<String> {
        vec![
            "build".to_owned(),
            "--file".to_owned(),
            eim_source
                .join("docker/Dockerfile.inference-xeon-base")
                .to_string_lossy()
                .into_owned(),
            "--build-arg".to_owned(),
            format!("PARENT_REGISTRY={parent_registry}"),
            "--build-arg".to_owned(),
            format!("PARENT_REPOSITORY={parent_repository}"),
            "--build-arg".to_owned(),
            format!("PARENT_TAG={parent_tag}"),
            "--tag".to_owned(),
            base_image.to_owned(),
            // The EIM tree is the build context: the Dockerfile copies its runtime sources and
            // general profiles from paths relative to it.
            eim_source.to_string_lossy().into_owned(),
        ]
    }

    /// Arguments for `docker build` of one model's thin layer.
    pub fn model_build_args(
        eim_source: &Path,
        base_image: &str,
        canonical_name: &str,
        image: &str,
    ) -> Result<Vec<String>, ImageError> {
        let (org, model) = split_model_id(canonical_name)?;
        Ok(vec![
            "build".to_owned(),
            "--file".to_owned(),
            eim_source
                .join("docker/Dockerfile.inference")
                .to_string_lossy()
                .into_owned(),
            "--build-arg".to_owned(),
            format!("BASE_IMAGE={base_image}"),
            // Only Xeon exists: EIM's GPU and NPU detectors are unimplemented stubs.
            "--build-arg".to_owned(),
            "ACCELERATOR_FAMILY=xeon".to_owned(),
            "--build-arg".to_owned(),
            format!("ORG={org}"),
            "--build-arg".to_owned(),
            format!("MODEL={model}"),
            "--tag".to_owned(),
            image.to_owned(),
            eim_source.to_string_lossy().into_owned(),
        ])
    }

    /// Ensures the image is present, pulling or building only if it is not.
    ///
    /// `on_stage` is called before each step so a caller can report what is happening: a base
    /// image build downloads several gigabytes and takes minutes, which is indistinguishable
    /// from a hang without it.
    pub async fn resolve(
        &self,
        image: &str,
        canonical_name: &str,
        mut on_stage: impl FnMut(ImageStage),
    ) -> Result<ImageOutcome, ImageError> {
        // Configuration is checked before the daemon is consulted. A missing checkout is worth
        // naming precisely -- the alternative is a Docker error about an unreadable build
        // context, which points nowhere useful -- and it costs nothing to find out early.
        if let ImageSource::Build { eim_source, .. } = &self.source
            && !eim_source.join("docker/Dockerfile.inference").is_file()
        {
            return Err(ImageError::MissingSource {
                path: eim_source.clone(),
            });
        }

        on_stage(ImageStage::Inspecting);
        if let Some(summary) = self.docker.image_summary(image).await? {
            return Ok(ImageOutcome::AlreadyPresent(summary));
        }

        match &self.source {
            ImageSource::PresentOnly => Err(ImageError::Absent {
                image: image.to_owned(),
            }),
            ImageSource::Registry => {
                on_stage(ImageStage::Pulling);
                self.docker
                    .invoke(&["pull".to_owned(), image.to_owned()])
                    .await?
                    .succeeded()
                    .then_some(())
                    .ok_or(ImageError::NoImageAfter { stage: "pull" })?;
                self.summary_after(image, "pull")
                    .await
                    .map(ImageOutcome::Pulled)
            }
            ImageSource::Build {
                eim_source,
                parent_registry,
                parent_repository,
                parent_tag,
                base_image,
            } => {
                // The base is shared across every model, so build it only when absent.
                if self.docker.image_summary(base_image).await?.is_none() {
                    on_stage(ImageStage::BuildingBase);
                    let args = Self::base_build_args(
                        eim_source,
                        parent_registry,
                        parent_repository,
                        parent_tag,
                        base_image,
                    );
                    let invocation = self.docker.invoke(&args).await?;
                    if !invocation.succeeded() {
                        return Err(ImageError::Docker(DockerError::Failed {
                            invocation: Box::new(invocation),
                        }));
                    }
                }

                on_stage(ImageStage::BuildingModel);
                let args = Self::model_build_args(eim_source, base_image, canonical_name, image)?;
                let invocation = self.docker.invoke(&args).await?;
                if !invocation.succeeded() {
                    return Err(ImageError::Docker(DockerError::Failed {
                        invocation: Box::new(invocation),
                    }));
                }
                self.summary_after(image, "build")
                    .await
                    .map(ImageOutcome::Built)
            }
        }
    }

    async fn summary_after(
        &self,
        image: &str,
        stage: &'static str,
    ) -> Result<ImageSummary, ImageError> {
        self.docker
            .image_summary(image)
            .await?
            .ok_or(ImageError::NoImageAfter { stage })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_a_hugging_face_identifier() {
        assert_eq!(
            split_model_id("Qwen/Qwen3-8B").unwrap(),
            ("Qwen", "Qwen3-8B")
        );
        assert_eq!(
            split_model_id("ibm-granite/granite-3.2-2b-instruct").unwrap(),
            ("ibm-granite", "granite-3.2-2b-instruct")
        );
    }

    #[test]
    fn rejects_an_identifier_without_an_organization() {
        for name in ["Qwen3-8B", "/Qwen3-8B", "Qwen/", ""] {
            assert!(split_model_id(name).is_err(), "name: {name}");
        }
    }

    #[test]
    fn base_build_pins_the_vllm_parent_from_eims_own_configuration() {
        let args = ImageResolver::base_build_args(
            Path::new("/srv/eim"),
            "docker.io",
            "vllm/vllm-openai-cpu",
            "v0.26.0",
            "magnitude-eim-xeon-base:v1",
        );

        assert!(args.contains(&"PARENT_REGISTRY=docker.io".to_owned()));
        assert!(args.contains(&"PARENT_REPOSITORY=vllm/vllm-openai-cpu".to_owned()));
        assert!(args.contains(&"PARENT_TAG=v0.26.0".to_owned()));
        assert!(args.contains(&"/srv/eim/docker/Dockerfile.inference-xeon-base".to_owned()));
        // The EIM tree is the context; its Dockerfile copies sources relative to it.
        assert_eq!(args.last().map(String::as_str), Some("/srv/eim"));
    }

    #[test]
    fn model_build_layers_onto_the_base_and_names_the_model() {
        let args = ImageResolver::model_build_args(
            Path::new("/srv/eim"),
            "magnitude-eim-xeon-base:v1",
            "Qwen/Qwen3-8B",
            "magnitude-eim-xeon-qwen-qwen3-8b:v1",
        )
        .expect("well-formed identifier");

        assert!(args.contains(&"BASE_IMAGE=magnitude-eim-xeon-base:v1".to_owned()));
        assert!(args.contains(&"ORG=Qwen".to_owned()));
        assert!(args.contains(&"MODEL=Qwen3-8B".to_owned()));
        // Only Xeon exists; EIM's other accelerator detectors are unimplemented.
        assert!(args.contains(&"ACCELERATOR_FAMILY=xeon".to_owned()));
        assert!(args.contains(&"/srv/eim/docker/Dockerfile.inference".to_owned()));
    }

    #[test]
    fn a_malformed_identifier_fails_before_docker_is_invoked() {
        assert!(matches!(
            ImageResolver::model_build_args(
                Path::new("/srv/eim"),
                "base:v1",
                "no-organization",
                "image:v1",
            ),
            Err(ImageError::MalformedModelId(_))
        ));
    }

    #[tokio::test]
    async fn an_unusable_daemon_is_reported_as_a_docker_failure() {
        // Not as an absent image: "Docker is broken" and "this model is not installed" call for
        // different responses from the operator, so they must not collapse into one error.
        let resolver = ImageResolver::new(
            DockerCli::new("magnitude-nonexistent-docker"),
            ImageSource::PresentOnly,
        );
        let error = resolver
            .resolve("missing:v1", "Qwen/Qwen3-8B", |_| {})
            .await
            .expect_err("the daemon cannot be reached");

        assert!(matches!(error, ImageError::Docker(_)), "{error:?}");
    }

    #[tokio::test]
    async fn a_missing_checkout_is_named_rather_than_left_to_docker() {
        let resolver = ImageResolver::new(
            DockerCli::new("magnitude-nonexistent-docker"),
            ImageSource::Build {
                eim_source: PathBuf::from("/definitely/not/here"),
                parent_registry: "docker.io".to_owned(),
                parent_repository: "vllm/vllm-openai-cpu".to_owned(),
                parent_tag: "v0.26.0".to_owned(),
                base_image: "magnitude-eim-xeon-base:v1".to_owned(),
            },
        );
        let error = resolver
            .resolve("missing:v1", "Qwen/Qwen3-8B", |_| {})
            .await
            .expect_err("there is no checkout");

        // Docker would otherwise complain about an unreadable context, which points nowhere.
        assert!(matches!(error, ImageError::MissingSource { .. }));
    }

    #[test]
    fn every_stage_describes_itself_for_progress_reporting() {
        for stage in [
            ImageStage::Inspecting,
            ImageStage::Pulling,
            ImageStage::BuildingBase,
            ImageStage::BuildingModel,
        ] {
            assert!(!stage.describe().is_empty(), "{stage:?}");
        }
        // A base build downloads gigabytes; without a description it is indistinguishable from
        // a hang.
        assert!(ImageStage::BuildingBase.describe().contains("base"));
    }
}
