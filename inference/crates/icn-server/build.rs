use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));
    let pin_path = manifest_dir.join("../../eim-pin.toml");
    println!("cargo:rerun-if-changed={}", pin_path.display());

    let pin = fs::read_to_string(&pin_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", pin_path.display()));
    let eim_revision = table_value(&pin, "eim", "revision")
        .unwrap_or_else(|| panic!("missing eim.revision in {}", pin_path.display()));
    let base_image_part = |key: &str| {
        table_value(&pin, "base_image", key)
            .unwrap_or_else(|| panic!("missing base_image.{key} in {}", pin_path.display()))
    };
    let base_image = format!(
        "{}/{}:{}",
        base_image_part("registry_host"),
        base_image_part("base_repository"),
        base_image_part("base_tag"),
    );
    let image_tag_prefix = table_value(&pin, "images", "tag_prefix")
        .unwrap_or_else(|| panic!("missing images.tag_prefix in {}", pin_path.display()));

    emit("ICN_EIM_REVISION", &eim_revision);
    emit("ICN_EIM_BASE_IMAGE", &base_image);
    emit("ICN_EIM_IMAGE_TAG_PREFIX", &image_tag_prefix);
    emit(
        "ICN_BUILD_TARGET",
        &env::var("TARGET").expect("TARGET must be set"),
    );
    emit(
        "ICN_BUILD_PROFILE",
        &env::var("PROFILE").expect("PROFILE must be set"),
    );
    emit("ICN_RUSTC_VERSION", &rustc_version());
}

fn table_value(source: &str, wanted_table: &str, wanted_key: &str) -> Option<String> {
    let mut table = None;
    for raw_line in source.lines() {
        let line = raw_line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            table = Some(name.trim());
            continue;
        }
        if table != Some(wanted_table) {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == wanted_key {
            return value
                .trim()
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .map(str::to_owned);
        }
    }
    None
}

fn rustc_version() -> String {
    let rustc = env::var_os("RUSTC").expect("RUSTC must be set");
    let output = Command::new(&rustc)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| {
            panic!("failed to execute {}: {error}", Path::new(&rustc).display())
        });
    assert!(output.status.success(), "rustc --version failed");
    String::from_utf8(output.stdout)
        .expect("rustc --version must be UTF-8")
        .trim()
        .to_owned()
}

fn emit(name: &str, value: &str) {
    println!("cargo:rustc-env={name}={value}");
}
