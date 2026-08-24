//! Fetching a model's weights into EIM's Local Directory layout.
//!
//! EIM resolves `INFERENCE_MODEL_ID` against `<cache>/<org>/<model>/` first and only falls back
//! to letting the engine download — `ModelCacheResolver` calls that layout "an explicit,
//! pre-populated layout", and it is a plain copy of the repository's files with no hub cache
//! structure, no blobs and no symlinks. Populating it is therefore something Magnitude can do
//! itself, over ordinary HTTP.
//!
//! That is worth doing rather than leaving to the container for three reasons. The download is
//! the longest step in serving a model for the first time and this is the only place it can be
//! given a real byte-accurate progress bar, because the client's download surface already
//! carries `completedBytes`/`totalBytes`/`bytesPerSecond` and polls it. A populated directory is
//! reused by every later container instead of being fetched again into an ephemeral filesystem.
//! And once the weights are local the container needs no network at all, which is what lets the
//! cache mount read-only.
//!
//! Selection is a denylist, not an allowlist. A repository's non-weight files are kilobytes, so
//! keeping an unrecognised one costs nothing while omitting a needed one produces a load failure
//! whose cause is invisible — the asymmetry decides the direction. What the denylist is actually
//! for is the several repositories that publish the same weights twice: `openai/gpt-oss-20b`
//! ships `metal/model.bin` and `original/model.safetensors` alongside its shards, which is 27 GB
//! of duplicate out of 41 GB, and `mistralai/Mistral-7B-Instruct-v0.2` ships 15 GB of
//! `pytorch_model-*.bin` beside its safetensors.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

/// Public Hugging Face endpoint. Overridable so a test can point at a local server.
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// The file name recording what was fetched, kept beside the model directories rather than
/// inside one so the engine never sees a file the repository does not contain.
const MANIFEST_DIRECTORY: &str = ".magnitude";

#[derive(Debug, thiserror::Error)]
pub enum WeightError {
    #[error("hugging face request failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("repository {repository} is not readable: HTTP {status}")]
    Unreadable { repository: String, status: u16 },
    #[error("repository {repository} requires a Hugging Face access token")]
    Gated { repository: String },
    #[error("repository {repository} listed no weight files")]
    NoWeights { repository: String },
    #[error("{path} arrived as {actual_bytes} bytes, expected {expected_bytes}")]
    ShortFile {
        path: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
    #[error("{path} failed its checksum: expected {expected}, computed {actual}")]
    Corrupt {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("writing {}: {source}", path.display())]
    Storage {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{required_bytes} bytes are needed and {available_bytes} are free on the weight cache")]
    InsufficientSpace {
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error("{name} is not a usable proxy URL: {url}")]
    Proxy { name: &'static str, url: String },
    #[error("cancelled")]
    Cancelled,
}

/// One file of a repository, as Hugging Face describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    pub path: String,
    pub size_bytes: u64,
    /// The LFS object digest, present for every large file. Plain small files have only a git
    /// blob id, which is a SHA-1 over different bytes and so cannot verify the content.
    pub sha256: Option<String>,
}

/// What a repository holds at one revision, after selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepositorySnapshot {
    pub repository: String,
    /// The resolved commit. Pinned so the listing and the downloads cannot disagree if the
    /// branch moves while a multi-gigabyte fetch is in flight.
    pub revision: String,
    pub files: Vec<RemoteFile>,
    pub total_bytes: u64,
}

#[derive(Deserialize)]
struct RepositoryInfo {
    sha: Option<String>,
    #[serde(default)]
    gated: serde_json::Value,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<TreeEntryLfs>,
}

#[derive(Deserialize)]
struct TreeEntryLfs {
    oid: String,
}

/// Directories that duplicate the weights in another framework's format, or hold documentation
/// assets. Each was observed in one of the catalog's own repositories.
const SKIPPED_DIRECTORIES: &[&str] = &[
    "original", // Llama and gpt-oss keep pre-conversion checkpoints here
    "metal",    // gpt-oss ships a 13.7 GB Metal build of the same weights
    "onnx", "openvino", "coreml", "tflite",
    "figures",  // Phi-4-multimodal documentation images
    "examples", // Phi-4-multimodal sample audio
];

/// Extensions that are never loaded by vLLM.
const SKIPPED_EXTENSIONS: &[&str] = &[
    "pth",
    "msgpack",
    "h5",
    "onnx",
    "onnx_data",
    "gguf",
    "tflite",
    "mlmodel",
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "svg",
    "webp",
    "wav",
    "mp3",
    "mp4",
    "md",
];

/// Extensions that hold weights in a format superseded by safetensors.
const SUPERSEDED_WEIGHT_EXTENSIONS: &[&str] = &["bin", "pt", "ckpt"];

fn extension_of(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .map(|extension| extension.to_string_lossy().to_ascii_lowercase())
}

fn leading_directory(path: &str) -> Option<&str> {
    path.split_once('/').map(|(head, _)| head)
}

/// Whether one entry is needed to serve the model.
///
/// `has_safetensors` is a property of the whole tree, which is why selection cannot be decided
/// per file in isolation: a repository that publishes only `pytorch_model.bin` still needs it.
#[must_use]
pub fn is_required(path: &str, has_safetensors: bool) -> bool {
    if path == ".gitattributes" {
        return false;
    }
    if let Some(directory) = leading_directory(path)
        && SKIPPED_DIRECTORIES.contains(&directory)
    {
        return false;
    }
    match extension_of(path) {
        Some(extension) if SKIPPED_EXTENSIONS.contains(&extension.as_str()) => false,
        Some(extension) if SUPERSEDED_WEIGHT_EXTENSIONS.contains(&extension.as_str()) => {
            !has_safetensors
        }
        // `pytorch_model.bin.index.json` names shards this selection dropped, and vLLM prefers
        // whichever index matches the weights it found. Keeping a stale one is noise at best.
        _ if path.ends_with("pytorch_model.bin.index.json") => !has_safetensors,
        _ => true,
    }
}

/// Selects the files to fetch from a full tree listing.
fn select(entries: Vec<RemoteFile>) -> Vec<RemoteFile> {
    let has_safetensors = entries.iter().any(|entry| {
        extension_of(&entry.path).as_deref() == Some("safetensors")
            && leading_directory(&entry.path)
                .is_none_or(|directory| !SKIPPED_DIRECTORIES.contains(&directory))
    });
    let mut selected: Vec<RemoteFile> = entries
        .into_iter()
        .filter(|entry| is_required(&entry.path, has_safetensors))
        .collect();
    // Ordered so a resumed fetch walks the same sequence, and so the small configuration files
    // land first: a repository whose weights are unreachable then fails before gigabytes move.
    selected.sort_by(|left, right| {
        left.size_bytes
            .cmp(&right.size_bytes)
            .then_with(|| left.path.cmp(&right.path))
    });
    selected
}

/// Free bytes on the filesystem holding `path`, or `None` when it cannot be determined.
///
/// Checked before a fetch because the failure it prevents is the worst-behaved one available: a
/// filesystem filling up part-way through a 60 GB download takes the whole host's writes with
/// it, and the client's download surface has a variant that names the two numbers instead.
#[must_use]
pub fn available_bytes(path: &Path) -> Option<u64> {
    let mut probe = path;
    let target = loop {
        if probe.exists() {
            break probe;
        }
        probe = probe.parent()?;
    };
    let raw = std::ffi::CString::new(target.as_os_str().as_encoded_bytes()).ok()?;
    // SAFETY: `raw` is a valid NUL-terminated path and `stats` is a POD output parameter that
    // statvfs fully initializes when it returns zero.
    unsafe {
        let mut stats = std::mem::zeroed::<libc::statvfs>();
        if libc::statvfs(raw.as_ptr(), &raw mut stats) != 0 {
            return None;
        }
        // Blocks available to an unprivileged writer, times the fragment size. Not `f_bfree`,
        // which includes the reserve only root may consume.
        Some((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64))
    }
}

/// Progress of a weight fetch, reported after every chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FetchProgress {
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

enum Scheme {
    Http,
    Https,
}

/// Reads repositories from Hugging Face.
pub struct WeightSource {
    client: reqwest::Client,
    endpoint: String,
    token: Option<String>,
}

impl WeightSource {
    #[must_use]
    pub fn new(client: reqwest::Client, endpoint: String, token: Option<String>) -> Self {
        Self {
            client,
            endpoint,
            token,
        }
    }

    /// Builds a source that honors the same proxy the container is given.
    ///
    /// Configured from `ProxySettings` rather than left to reqwest's own environment detection, so
    /// there is exactly one place proxy configuration comes from. Two independent readers of the
    /// same variables is how a host ends up pulling images successfully while weight downloads
    /// time out — the daemon has a proxy and the process does not, and the symptom looks like a
    /// Hugging Face outage.
    pub fn through_proxy(
        endpoint: String,
        token: Option<String>,
        proxy: &crate::docker::cli::ProxySettings,
    ) -> Result<Self, WeightError> {
        let no_proxy = proxy
            .no_proxy
            .as_deref()
            .and_then(reqwest::NoProxy::from_string);
        let mut builder = reqwest::Client::builder()
            .user_agent(concat!("magnitude-icn/", env!("CARGO_PKG_VERSION")))
            // No total timeout: one safetensors shard is several gigabytes and a legitimate fetch
            // runs for minutes. The connect timeout is what separates an unreachable endpoint
            // from a slow one.
            .connect_timeout(std::time::Duration::from_secs(30))
            .pool_idle_timeout(std::time::Duration::from_secs(90));
        for (name, url, scheme) in [
            ("HTTPS_PROXY", proxy.https_proxy.as_deref(), Scheme::Https),
            ("HTTP_PROXY", proxy.http_proxy.as_deref(), Scheme::Http),
        ] {
            let Some(url) = url else { continue };
            let configured = match scheme {
                Scheme::Https => reqwest::Proxy::https(url),
                Scheme::Http => reqwest::Proxy::http(url),
            }
            .map_err(|_| WeightError::Proxy {
                name,
                url: url.to_owned(),
            })?;
            builder = builder.proxy(configured.no_proxy(no_proxy.clone()));
        }
        Ok(Self::new(
            builder.build().map_err(WeightError::Transport)?,
            endpoint,
            token,
        ))
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// Resolves the branch to a commit and lists the files worth fetching.
    pub async fn snapshot(&self, repository: &str) -> Result<RepositorySnapshot, WeightError> {
        let info = self
            .authorized(
                self.client
                    .get(format!("{}/api/models/{repository}", self.endpoint)),
            )
            .send()
            .await?;
        let status = info.status();
        if !status.is_success() {
            return Err(
                if status == reqwest::StatusCode::UNAUTHORIZED
                    || status == reqwest::StatusCode::FORBIDDEN
                {
                    WeightError::Gated {
                        repository: repository.to_owned(),
                    }
                } else {
                    WeightError::Unreadable {
                        repository: repository.to_owned(),
                        status: status.as_u16(),
                    }
                },
            );
        }
        let info: RepositoryInfo = info.json().await?;
        // A revision is always available in practice; falling back to the branch keeps the fetch
        // possible rather than failing over a field that is only there to pin.
        let revision = info.sha.unwrap_or_else(|| "main".to_owned());
        if info.gated.as_bool() == Some(true) && self.token.is_none() {
            return Err(WeightError::Gated {
                repository: repository.to_owned(),
            });
        }

        let entries = self.tree(repository, &revision).await?;
        let files = select(entries);
        if files.is_empty() {
            return Err(WeightError::NoWeights {
                repository: repository.to_owned(),
            });
        }
        let total_bytes = files.iter().map(|file| file.size_bytes).sum();
        Ok(RepositorySnapshot {
            repository: repository.to_owned(),
            revision,
            files,
            total_bytes,
        })
    }

    async fn tree(&self, repository: &str, revision: &str) -> Result<Vec<RemoteFile>, WeightError> {
        let mut url = format!(
            "{}/api/models/{repository}/tree/{revision}?recursive=1",
            self.endpoint
        );
        let mut entries = Vec::new();
        loop {
            let response = self
                .authorized(self.client.get(&url))
                .send()
                .await?
                .error_for_status()?;
            // Large repositories page. Ours do not, but a listing silently truncated at the page
            // limit would omit weight shards and produce a model that loads to a confusing error.
            let next = response
                .headers()
                .get(reqwest::header::LINK)
                .and_then(|value| value.to_str().ok())
                .and_then(next_page_link);
            let page: Vec<TreeEntry> = response.json().await?;
            entries.extend(
                page.into_iter()
                    .filter(|entry| entry.kind == "file")
                    .map(|entry| RemoteFile {
                        path: entry.path,
                        size_bytes: entry.size,
                        sha256: entry.lfs.map(|lfs| lfs.oid),
                    }),
            );
            match next {
                Some(link) => url = link,
                None => return Ok(entries),
            }
        }
    }

    /// Fetches every selected file into `<destination>/<org>/<model>/`, reporting cumulative
    /// bytes as it goes.
    ///
    /// A file already present at its expected size is skipped rather than re-verified: hashing
    /// 60 GB to confirm what the manifest already records would cost more than the fetch it is
    /// meant to avoid. Content is verified as it streams, which is when it is free.
    pub async fn fetch(
        &self,
        snapshot: &RepositorySnapshot,
        model_directory: &Path,
        cancelled: &Arc<AtomicBool>,
        mut on_progress: impl FnMut(FetchProgress),
    ) -> Result<(), WeightError> {
        tokio::fs::create_dir_all(model_directory)
            .await
            .map_err(|source| WeightError::Storage {
                path: model_directory.to_owned(),
                source,
            })?;

        let mut completed_bytes = 0u64;
        for file in &snapshot.files {
            if cancelled.load(Ordering::Relaxed) {
                return Err(WeightError::Cancelled);
            }
            let destination = model_directory.join(&file.path);
            if tokio::fs::metadata(&destination)
                .await
                .is_ok_and(|metadata| metadata.len() == file.size_bytes)
            {
                completed_bytes += file.size_bytes;
                on_progress(FetchProgress {
                    completed_bytes,
                    total_bytes: snapshot.total_bytes,
                });
                continue;
            }
            self.fetch_one(snapshot, file, &destination, cancelled, |written| {
                on_progress(FetchProgress {
                    completed_bytes: completed_bytes + written,
                    total_bytes: snapshot.total_bytes,
                });
            })
            .await?;
            completed_bytes += file.size_bytes;
        }
        Ok(())
    }

    async fn fetch_one(
        &self,
        snapshot: &RepositorySnapshot,
        file: &RemoteFile,
        destination: &Path,
        cancelled: &Arc<AtomicBool>,
        mut on_written: impl FnMut(u64),
    ) -> Result<(), WeightError> {
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| WeightError::Storage {
                    path: parent.to_owned(),
                    source,
                })?;
        }

        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint, snapshot.repository, snapshot.revision, file.path
        );
        let response = self.authorized(self.client.get(&url)).send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(
                if status == reqwest::StatusCode::UNAUTHORIZED
                    || status == reqwest::StatusCode::FORBIDDEN
                {
                    WeightError::Gated {
                        repository: snapshot.repository.clone(),
                    }
                } else {
                    WeightError::Unreadable {
                        repository: snapshot.repository.clone(),
                        status: status.as_u16(),
                    }
                },
            );
        }

        // Written beside the target and renamed on success, so an interrupted fetch never leaves
        // a short file that the size check above would later mistake for a complete one.
        let partial = partial_path(destination);
        let mut sink =
            tokio::fs::File::create(&partial)
                .await
                .map_err(|source| WeightError::Storage {
                    path: partial.clone(),
                    source,
                })?;
        let mut digest = Sha256::new();
        let mut written = 0u64;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            if cancelled.load(Ordering::Relaxed) {
                let _ = tokio::fs::remove_file(&partial).await;
                return Err(WeightError::Cancelled);
            }
            let chunk = chunk?;
            digest.update(&chunk);
            sink.write_all(&chunk)
                .await
                .map_err(|source| storage_error(&partial, source, snapshot.total_bytes))?;
            written += chunk.len() as u64;
            on_written(written);
        }
        sink.flush()
            .await
            .map_err(|source| storage_error(&partial, source, snapshot.total_bytes))?;
        drop(sink);

        if written != file.size_bytes {
            let _ = tokio::fs::remove_file(&partial).await;
            return Err(WeightError::ShortFile {
                path: file.path.clone(),
                expected_bytes: file.size_bytes,
                actual_bytes: written,
            });
        }
        if let Some(expected) = &file.sha256 {
            let actual = format!("{:x}", digest.finalize());
            if &actual != expected {
                let _ = tokio::fs::remove_file(&partial).await;
                return Err(WeightError::Corrupt {
                    path: file.path.clone(),
                    expected: expected.clone(),
                    actual,
                });
            }
        }
        tokio::fs::rename(&partial, destination)
            .await
            .map_err(|source| WeightError::Storage {
                path: destination.to_owned(),
                source,
            })
    }
}

/// Where a file is written while it is still arriving.
///
/// A suffix on the whole name rather than a replaced extension: `model-00001-of-00016.safetensors`
/// and `pytorch_model.bin.index.json` both have to survive the round trip unambiguously.
fn partial_path(destination: &Path) -> PathBuf {
    let mut name = destination.file_name().unwrap_or_default().to_os_string();
    name.push(".magnitude-partial");
    destination.with_file_name(name)
}

/// Classifies a write failure, keeping "the disk filled up" distinguishable from every other
/// storage problem. It is worth its own variant because the client renders the two byte figures,
/// and because it is the one failure the operator can act on directly.
fn storage_error(path: &Path, source: std::io::Error, required_bytes: u64) -> WeightError {
    if source.raw_os_error() == Some(libc::ENOSPC) {
        return WeightError::InsufficientSpace {
            required_bytes,
            available_bytes: available_bytes(path).unwrap_or(0),
        };
    }
    WeightError::Storage {
        path: path.to_owned(),
        source,
    }
}

/// Extracts the `rel="next"` target from a `Link` header.
fn next_page_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        if !part.contains("rel=\"next\"") {
            return None;
        }
        let start = part.find('<')? + 1;
        let end = part.find('>')?;
        (start < end).then(|| part[start..end].to_owned())
    })
}

/// What a completed fetch recorded, so a later start can tell a complete directory from a
/// partial one without hashing gigabytes.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightManifest {
    pub repository: String,
    pub revision: String,
    pub total_bytes: u64,
    pub files: Vec<ManifestFile>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestFile {
    pub path: String,
    pub size_bytes: u64,
}

impl WeightManifest {
    #[must_use]
    pub fn of(snapshot: &RepositorySnapshot) -> Self {
        Self {
            repository: snapshot.repository.clone(),
            revision: snapshot.revision.clone(),
            total_bytes: snapshot.total_bytes,
            files: snapshot
                .files
                .iter()
                .map(|file| ManifestFile {
                    path: file.path.clone(),
                    size_bytes: file.size_bytes,
                })
                .collect(),
        }
    }
}

/// Where the manifest for one repository lives, given the cache root.
#[must_use]
pub fn manifest_path(cache_root: &Path, repository: &str) -> PathBuf {
    cache_root
        .join(MANIFEST_DIRECTORY)
        .join(format!("{}.json", repository.replace('/', "--")))
}

/// The Local Directory EIM resolves `INFERENCE_MODEL_ID` to.
#[must_use]
pub fn model_directory(cache_root: &Path, repository: &str) -> PathBuf {
    repository
        .split('/')
        .fold(cache_root.to_owned(), |path, part| path.join(part))
}

/// Records a completed fetch.
pub async fn write_manifest(
    cache_root: &Path,
    snapshot: &RepositorySnapshot,
) -> Result<(), WeightError> {
    let path = manifest_path(cache_root, &snapshot.repository);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|source| WeightError::Storage {
                path: parent.to_owned(),
                source,
            })?;
    }
    let body =
        serde_json::to_vec_pretty(&WeightManifest::of(snapshot)).unwrap_or_else(|_| b"{}".to_vec());
    tokio::fs::write(&path, body)
        .await
        .map_err(|source| WeightError::Storage { path, source })
}

/// Whether the weights for one repository are completely present.
///
/// Every recorded file must exist at its recorded size. Size alone is deliberate: the content
/// was verified against its checksum while it streamed, and rehashing tens of gigabytes on every
/// catalog listing would make `GET /v1/models` unusable.
pub async fn is_present(cache_root: &Path, repository: &str) -> bool {
    let Ok(body) = tokio::fs::read(manifest_path(cache_root, repository)).await else {
        return false;
    };
    let Ok(manifest) = serde_json::from_slice::<WeightManifest>(&body) else {
        return false;
    };
    let directory = model_directory(cache_root, repository);
    for file in &manifest.files {
        let path = directory.join(&file.path);
        match tokio::fs::metadata(&path).await {
            Ok(metadata) if metadata.len() == file.size_bytes => {}
            _ => return false,
        }
    }
    true
}

/// Removes the weights and the manifest for one repository.
pub async fn remove(cache_root: &Path, repository: &str) -> std::io::Result<()> {
    let directory = model_directory(cache_root, repository);
    if tokio::fs::metadata(&directory).await.is_ok() {
        tokio::fs::remove_dir_all(&directory).await?;
    }
    let manifest = manifest_path(cache_root, repository);
    if tokio::fs::metadata(&manifest).await.is_ok() {
        tokio::fs::remove_file(&manifest).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, size_bytes: u64) -> RemoteFile {
        RemoteFile {
            path: path.to_owned(),
            size_bytes,
            sha256: None,
        }
    }

    fn selected(entries: Vec<RemoteFile>) -> Vec<String> {
        select(entries)
            .into_iter()
            .map(|entry| entry.path)
            .collect()
    }

    #[test]
    fn keeps_the_files_vllm_loads() {
        let paths = selected(vec![
            file("config.json", 726),
            file("generation_config.json", 239),
            file("model.safetensors.index.json", 32_819),
            file("model-00001-of-00003.safetensors", 3_957_900_840),
            file("tokenizer.json", 11_422_654),
            file("tokenizer_config.json", 9_732),
            file("vocab.json", 2_776_833),
            file("merges.txt", 1_671_853),
        ]);

        assert_eq!(paths.len(), 8, "every one of these is loaded: {paths:?}");
    }

    #[test]
    fn keeps_a_chat_template_and_a_preprocessor_configuration() {
        // gpt-oss ships its harmony template as a separate file, and Qwen3-VL its vision and
        // video preprocessor configurations. Dropping any of them changes how requests render.
        let paths = selected(vec![
            file("config.json", 1_806),
            file("chat_template.jinja", 16_738),
            file("chat_template.json", 5_499),
            file("preprocessor_config.json", 390),
            file("video_preprocessor_config.json", 385),
            file("model.safetensors", 100),
        ]);

        assert!(paths.contains(&"chat_template.jinja".to_owned()));
        assert!(paths.contains(&"chat_template.json".to_owned()));
        assert!(paths.contains(&"preprocessor_config.json".to_owned()));
        assert!(paths.contains(&"video_preprocessor_config.json".to_owned()));
    }

    #[test]
    fn drops_the_duplicate_weights_gpt_oss_publishes() {
        // The real tree: 41 GB listed, 13.8 GB of it actually loaded.
        let paths = selected(vec![
            file("config.json", 1_806),
            file("metal/model.bin", 13_750_886_400),
            file("model-00000-of-00002.safetensors", 4_792_272_488),
            file("model-00001-of-00002.safetensors", 4_798_702_184),
            file("model-00002-of-00002.safetensors", 4_170_342_232),
            file("model.safetensors.index.json", 36_355),
            file("original/config.json", 376),
            file("original/model.safetensors", 13_761_300_984),
            file("tokenizer.json", 27_868_174),
        ]);

        assert!(!paths.iter().any(|path| path.starts_with("metal/")));
        assert!(!paths.iter().any(|path| path.starts_with("original/")));
        assert_eq!(
            paths
                .iter()
                .filter(|path| path.ends_with(".safetensors"))
                .count(),
            3,
            "only the sharded copy is fetched"
        );
    }

    #[test]
    fn prefers_safetensors_over_the_pytorch_copy() {
        // Mistral v0.2 publishes both. Fetching each is 15 GB of waste.
        let paths = selected(vec![
            file("model-00001-of-00003.safetensors", 4_943_162_336),
            file("model.safetensors.index.json", 25_125),
            file("pytorch_model-00001-of-00003.bin", 4_943_184_288),
            file("pytorch_model.bin.index.json", 23_950),
        ]);

        assert_eq!(
            paths,
            vec![
                "model.safetensors.index.json".to_owned(),
                "model-00001-of-00003.safetensors".to_owned(),
            ]
        );
    }

    #[test]
    fn keeps_the_pytorch_copy_when_it_is_the_only_one() {
        let paths = selected(vec![
            file("config.json", 596),
            file("pytorch_model-00001-of-00003.bin", 4_943_184_288),
            file("pytorch_model.bin.index.json", 23_950),
        ]);

        assert!(paths.contains(&"pytorch_model-00001-of-00003.bin".to_owned()));
        assert!(paths.contains(&"pytorch_model.bin.index.json".to_owned()));
    }

    #[test]
    fn keeps_remote_code_and_adapters_but_not_documentation() {
        // Phi-4-multimodal needs its `.py` architecture and both LoRA adapters, and ships a
        // technical report, sample audio and radar charts it does not need.
        let paths = selected(vec![
            file("modeling_phi4mm.py", 116_057),
            file("configuration_phi4mm.py", 11_014),
            file("speech-lora/adapter_model.safetensors", 922_782_296),
            file("vision-lora/adapter_config.json", 464),
            file("model-00001-of-00003.safetensors", 4_997_504_848),
            file("phi_4_mm.tech_report.02252025.pdf", 5_295_165),
            file("figures/vision_radar.png", 173_821),
            file("examples/what_is_shown_in_this_image.wav", 112_844),
            file("README.md", 65_395),
            file("SECURITY.md", 2_656),
        ]);

        assert!(paths.contains(&"modeling_phi4mm.py".to_owned()));
        assert!(paths.contains(&"speech-lora/adapter_model.safetensors".to_owned()));
        assert!(paths.contains(&"vision-lora/adapter_config.json".to_owned()));
        assert!(!paths.iter().any(|path| path.ends_with(".pdf")));
        assert!(!paths.iter().any(|path| path.ends_with(".md")));
        assert!(!paths.iter().any(|path| path.starts_with("figures/")));
        assert!(!paths.iter().any(|path| path.starts_with("examples/")));
    }

    #[test]
    fn keeps_an_unrecognised_file_rather_than_guessing() {
        // The denylist direction matters: an omitted file fails a load for no visible reason,
        // while a spare kilobyte costs nothing.
        assert!(is_required("configuration.json", true));
        assert!(is_required("added_tokens.json", true));
        assert!(is_required("tokenizer.model", true));
        assert!(is_required("something-new.json", true));
    }

    #[test]
    fn small_files_are_fetched_first() {
        let order = selected(vec![
            file("model-00001-of-00002.safetensors", 4_000_000_000),
            file("config.json", 726),
            file("tokenizer.json", 11_422_654),
        ]);

        assert_eq!(
            order,
            vec![
                "config.json".to_owned(),
                "tokenizer.json".to_owned(),
                "model-00001-of-00002.safetensors".to_owned(),
            ],
            "an unreachable repository should fail before gigabytes move"
        );
    }

    #[test]
    fn the_local_directory_is_the_layout_eim_resolves() {
        assert_eq!(
            model_directory(Path::new("/cache"), "Qwen/Qwen3-8B"),
            PathBuf::from("/cache/Qwen/Qwen3-8B"),
        );
    }

    #[test]
    fn the_manifest_sits_outside_the_model_directory() {
        let manifest = manifest_path(Path::new("/cache"), "Qwen/Qwen3-8B");

        assert_eq!(
            manifest,
            PathBuf::from("/cache/.magnitude/Qwen--Qwen3-8B.json"),
            "the engine must never see a file the repository does not contain"
        );
        assert!(!manifest.starts_with("/cache/Qwen/Qwen3-8B"));
    }

    #[test]
    fn a_partial_file_keeps_its_whole_name() {
        assert_eq!(
            partial_path(Path::new(
                "/cache/Qwen/Qwen3-8B/model-00001-of-00005.safetensors"
            )),
            PathBuf::from(
                "/cache/Qwen/Qwen3-8B/model-00001-of-00005.safetensors.magnitude-partial"
            ),
            "replacing the extension would collide across shards"
        );
        assert_eq!(
            partial_path(Path::new("/cache/x/pytorch_model.bin.index.json")),
            PathBuf::from("/cache/x/pytorch_model.bin.index.json.magnitude-partial"),
        );
    }

    #[test]
    fn reads_the_next_page_from_a_link_header() {
        assert_eq!(
            next_page_link(
                "<https://huggingface.co/api/models/x/tree/main?cursor=abc>; rel=\"next\""
            ),
            Some("https://huggingface.co/api/models/x/tree/main?cursor=abc".to_owned()),
        );
        assert_eq!(next_page_link("<https://example.test>; rel=\"prev\""), None);
        assert_eq!(next_page_link(""), None);
    }

    #[tokio::test]
    async fn an_absent_manifest_is_not_present() {
        let directory =
            std::env::temp_dir().join(format!("magnitude-weights-absent-{}", std::process::id()));
        assert!(!is_present(&directory, "Qwen/Qwen3-8B").await);
    }

    #[tokio::test]
    async fn a_manifest_whose_files_are_missing_is_not_present() {
        let root =
            std::env::temp_dir().join(format!("magnitude-weights-partial-{}", std::process::id()));
        let _ = tokio::fs::remove_dir_all(&root).await;
        let snapshot = RepositorySnapshot {
            repository: "Qwen/Qwen3-8B".to_owned(),
            revision: "abc".to_owned(),
            files: vec![file("config.json", 4)],
            total_bytes: 4,
        };
        write_manifest(&root, &snapshot).await.expect("manifest");

        assert!(
            !is_present(&root, "Qwen/Qwen3-8B").await,
            "a manifest without its files must not report the model as installed"
        );

        tokio::fs::create_dir_all(model_directory(&root, "Qwen/Qwen3-8B"))
            .await
            .expect("directory");
        tokio::fs::write(
            model_directory(&root, "Qwen/Qwen3-8B").join("config.json"),
            b"{}\n\n",
        )
        .await
        .expect("write");

        assert!(is_present(&root, "Qwen/Qwen3-8B").await);

        remove(&root, "Qwen/Qwen3-8B").await.expect("remove");
        assert!(!is_present(&root, "Qwen/Qwen3-8B").await);
        let _ = tokio::fs::remove_dir_all(&root).await;
    }
}
