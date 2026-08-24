use icn_contracts::bootstrap_protocol::IcnBinaryIdentity;
use sha2::{Digest, Sha256};

pub(crate) const EIM_REVISION: &str = env!("ICN_EIM_REVISION");
pub(crate) const EIM_BASE_IMAGE: &str = env!("ICN_EIM_BASE_IMAGE");
pub(crate) const EIM_IMAGE_TAG_PREFIX: &str = env!("ICN_EIM_IMAGE_TAG_PREFIX");
pub(crate) const TARGET: &str = env!("ICN_BUILD_TARGET");
pub(crate) const PROFILE: &str = env!("ICN_BUILD_PROFILE");
pub(crate) const RUSTC_VERSION: &str = env!("ICN_RUSTC_VERSION");

/// One acceleration story: vLLM on Intel Xeon inside the EIM image.
///
/// The previous list was decided at compile time by the Metal, CUDA, and Vulkan feature graph.
/// A container-backed engine has no such choice to make — what it can use is whatever the
/// daemon and the image support, discovered at runtime by the Docker preflight.
pub(crate) fn enabled_backends() -> Vec<&'static str> {
    vec![icn_hardware::XEON_VLLM_BACKEND]
}

pub(crate) fn identity() -> IcnBinaryIdentity {
    IcnBinaryIdentity {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        api_version: 1,
        native_build: eim_build(),
        backend_module_abi: backend_module_abi(),
        // Unchanged deliberately: ACN refuses to start unless the binary advertises every
        // capability it needs, and a container-backed ICN still provides all seven. Dropping
        // one here would be a silent contract break rather than a compile error.
        capabilities: [
            "hardware",
            "model_catalog",
            "model_installed",
            "model_assessment",
            "model_downloads",
            "model_residency",
            "chat_streaming",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        target: TARGET.to_owned(),
        profile: PROFILE.to_owned(),
        rustc: RUSTC_VERSION.to_owned(),
        backends: enabled_backends().into_iter().map(str::to_owned).collect(),
    }
}

/// Identity of the serving stack, in place of the native build hash.
///
/// Keeps the `native_build` field name because it crosses the generated protocol boundary and
/// ACN compares it three times during startup. What it now digests is the pair that actually
/// determines behavior: the EIM revision the images are built from, and the vLLM base image
/// they sit on. A base-image change alters accepted engine arguments and tool-call parser
/// names, so it must invalidate a prepared installation exactly as a bindings change did.
pub(crate) fn eim_build() -> String {
    let mut digest = Sha256::new();
    digest.update(EIM_REVISION.as_bytes());
    digest.update([0]);
    digest.update(EIM_BASE_IMAGE.as_bytes());
    format!("eim_{:x}", digest.finalize())
}

pub(crate) fn backend_module_abi() -> String {
    format!("eim-vllm-{EIM_BASE_IMAGE}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_identity_pins_the_eim_revision_and_base_image() {
        assert_eq!(EIM_REVISION.len(), 40, "a full git revision");
        assert!(
            EIM_BASE_IMAGE.contains(':'),
            "an image reference with a tag"
        );
        assert!(!EIM_IMAGE_TAG_PREFIX.is_empty());

        let identity = identity();
        assert_eq!(identity.native_build, eim_build());
        assert!(identity.native_build.starts_with("eim_"));
        assert!(!identity.target.is_empty());
    }

    #[test]
    fn reports_exactly_the_container_backend() {
        assert_eq!(enabled_backends(), vec![icn_hardware::XEON_VLLM_BACKEND]);
    }

    #[test]
    fn advertises_every_capability_acn_requires() {
        // ACN's IcnBinaryResolutionConfig lists these as requiredCapabilities; a missing one
        // prevents ACN from ever becoming ready.
        let identity = identity();
        for capability in [
            "hardware",
            "model_catalog",
            "model_installed",
            "model_assessment",
            "model_downloads",
            "model_residency",
            "chat_streaming",
        ] {
            assert!(
                identity.capabilities.contains(&capability.to_owned()),
                "missing capability {capability}"
            );
        }
    }

    #[test]
    fn the_build_identity_changes_with_the_base_image() {
        // Two different pins must not produce the same build hash, or ACN could accept an
        // installation whose container arguments no longer match.
        let mut digest = Sha256::new();
        digest.update(EIM_REVISION.as_bytes());
        digest.update([0]);
        digest.update(b"docker.io/vllm/vllm-openai-cpu:v0.0.0");
        assert_ne!(format!("eim_{:x}", digest.finalize()), eim_build());
    }
}
