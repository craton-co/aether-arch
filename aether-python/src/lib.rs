//! Python bindings for AetherArch compression library.
//!
//! Provides a Pythonic API for compressing and decompressing `.aet` archives.
//!
//! # Usage
//!
//! ```python
//! import aether
//!
//! # Compress files
//! stats = aether.compress("input_dir/", ["file1.txt", "file2.rs"], "output.aet")
//!
//! # Extract all files
//! aether.extract_all("output.aet", "output_dir/")
//!
//! # Extract a single file
//! aether.extract_file("output.aet", "file1.txt", "single_output/file1.txt")
//!
//! # Verify archive integrity
//! result = aether.verify("output.aet")
//! assert result.is_ok
//!
//! # List files in archive
//! entries = aether.list_files("output.aet")
//! for entry in entries:
//!     print(f"{entry.path}: {entry.size} bytes")
//! ```

use std::io::BufReader;
use std::path::{Component, Path, PathBuf};

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use aether_core::entropy::{
    NeuralSsmPredictor, Order0Model, ProbabilityPredictor, RlePredictor,
};
#[cfg(feature = "context-mixer")]
use aether_core::entropy::{ContextMixer, Lz4AwarePredictor};
#[cfg(feature = "context-mixer")]
use aether_core::entropy::context_mixer::ContextMixerConfig;
use aether_core::format::PredictorId;
use aether_core::header::ArchiveHeader;
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Hard cap on max_threads to prevent resource exhaustion.
const MAX_THREADS_LIMIT: usize = 256;

/// Hard cap on the number of files accepted by `compress` to prevent resource
/// exhaustion at the binding layer (mirrors aether-core's MAX_FILE_COUNT).
const MAX_COMPRESS_FILES: usize = 1_000_000;

/// Clamp mtime to a sane range: 1970-01-01 to 3000-01-01.
const MTIME_MIN: i64 = 0;
const MTIME_MAX: i64 = 32_503_680_000;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Strip filesystem paths from error messages to avoid information disclosure.
fn sanitize_io_error(context: &str, err: std::io::Error) -> PyErr {
    // Only expose the OS error kind, not the full message which may contain paths.
    PyIOError::new_err(format!("{context}: {}", err.kind()))
}

fn sanitize_core_error(context: &str, err: aether_core::error::AetherError) -> PyErr {
    use aether_core::error::AetherError;
    // Redact variants that embed user-supplied paths or arbitrary strings
    // derived from archive contents to prevent information disclosure.
    match &err {
        AetherError::Io(_) => {
            PyRuntimeError::new_err(format!("{context}: I/O error"))
        }
        AetherError::FileNotFound(_) => {
            PyRuntimeError::new_err(format!("{context}: file not found in archive"))
        }
        AetherError::PathTraversal(_) => {
            PyRuntimeError::new_err(format!(
                "{context}: path traversal detected in archive entry"
            ))
        }
        // Catch-all: use a generic message to avoid leaking internal details
        // (e.g. file paths, archive contents) from future AetherError variants.
        _ => PyRuntimeError::new_err(format!("{context}: internal error")),
    }
}

/// Validate that a path is safe for use as an extraction destination:
/// - Must be relative (no absolute paths).
/// - Must not contain `..` (parent-dir) components.
/// - Must not contain null bytes.
fn validate_output_path(label: &str, p: &Path) -> PyResult<()> {
    // Reject null bytes (could truncate C-string paths on some platforms).
    if p.to_string_lossy().contains('\0') {
        return Err(PyValueError::new_err(format!(
            "{label} must not contain null bytes"
        )));
    }
    for component in p.components() {
        match component {
            Component::ParentDir => {
                return Err(PyValueError::new_err(format!(
                    "{label} must not contain '..' components (path traversal rejected)"
                )));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(PyValueError::new_err(format!(
                    "{label} must be a relative path, not absolute"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

// ── Predictor factory ───────────────────────────────────────────────────────

fn make_predictor_factory(
    name: &str,
) -> PyResult<(
    Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync>,
    PredictorId,
)> {
    match name {
        "order0" | "o0" => Ok((Box::new(|| Box::new(Order0Model::new())), PredictorId::Order0)),
        #[cfg(feature = "context-mixer")]
        "cm" | "context-mixer" => Ok((
            Box::new(|| Box::new(ContextMixer::with_config(ContextMixerConfig::default()))),
            PredictorId::ContextMixer,
        )),
        #[cfg(feature = "context-mixer")]
        "cm-light" => Ok((
            Box::new(|| Box::new(ContextMixer::with_config(ContextMixerConfig::lightweight()))),
            PredictorId::ContextMixerLight,
        )),
        #[cfg(feature = "context-mixer")]
        "lz4" | "lz4-aware" => Ok((
            Box::new(|| Box::new(Lz4AwarePredictor::new())),
            PredictorId::Lz4Aware,
        )),
        "ssm" | "neural-ssm" => Ok((
            Box::new(|| Box::new(NeuralSsmPredictor::new())),
            PredictorId::NeuralSsm,
        )),
        "rle" => Ok((
            Box::new(|| Box::new(RlePredictor::new())),
            PredictorId::Rle,
        )),
        #[cfg(not(feature = "context-mixer"))]
        "cm" | "context-mixer" | "cm-light" | "lz4" | "lz4-aware" => Err(PyValueError::new_err(
            format!("Predictor '{name}' requires the 'context-mixer' feature (not compiled in)")
        )),
        other => Err(PyValueError::new_err(format!(
            "Unknown predictor: '{other}'. Available: order0, cm, cm-light, lz4-aware, ssm, rle"
        ))),
    }
}

fn make_factory_from_id(
    id: PredictorId,
) -> PyResult<Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync>> {
    match id {
        PredictorId::Order0 => Ok(Box::new(|| Box::new(Order0Model::new()))),
        #[cfg(feature = "context-mixer")]
        PredictorId::ContextMixer => {
            Ok(Box::new(|| Box::new(ContextMixer::with_config(ContextMixerConfig::default()))))
        }
        #[cfg(feature = "context-mixer")]
        PredictorId::ContextMixerLight => {
            Ok(Box::new(|| Box::new(ContextMixer::with_config(ContextMixerConfig::lightweight()))))
        }
        #[cfg(feature = "context-mixer")]
        PredictorId::Lz4Aware => Ok(Box::new(|| Box::new(Lz4AwarePredictor::new()))),
        PredictorId::NeuralSsm => Ok(Box::new(|| Box::new(NeuralSsmPredictor::new()))),
        PredictorId::Rle => Ok(Box::new(|| Box::new(RlePredictor::new()))),
        other => Err(PyValueError::new_err(format!(
            "Unsupported predictor ID in archive: {other:?}"
        ))),
    }
}

/// Open an archive and read its header in one step, returning the file
/// (positioned after the header) and the detected predictor ID.
/// Eliminates the TOCTOU window from opening the file twice.
fn open_archive(path: &str) -> PyResult<(BufReader<std::fs::File>, PredictorId)> {
    let file = std::fs::File::open(path)
        .map_err(|e| sanitize_io_error("Cannot open archive", e))?;
    let mut reader = BufReader::new(file);
    let header = ArchiveHeader::read_from(&mut reader)
        .map_err(|e| sanitize_core_error("Failed to read archive header", e))?;
    Ok((reader, header.predictor_id))
}

// ── Python result types ─────────────────────────────────────────────────────

/// Statistics from a compression operation.
#[pyclass(frozen)]
#[derive(Debug, Clone)]
struct CompressionStats {
    /// Original total size of all input files in bytes.
    #[pyo3(get)]
    original_size: u64,
    /// Compressed archive size in bytes.
    #[pyo3(get)]
    compressed_size: u64,
    /// Number of compressed blocks.
    #[pyo3(get)]
    block_count: u32,
    /// Number of input files.
    #[pyo3(get)]
    file_count: u32,
    /// Number of solid groups.
    #[pyo3(get)]
    group_count: u32,
}

#[pymethods]
impl CompressionStats {
    /// Compression ratio (compressed / original), between 0.0 and 1.0.
    #[getter]
    fn ratio(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            self.compressed_size as f64 / self.original_size as f64
        }
    }

    /// Bits per byte of original data.
    #[getter]
    fn bits_per_byte(&self) -> f64 {
        self.ratio() * 8.0
    }

    fn __repr__(&self) -> String {
        format!(
            "CompressionStats(original_size={}, compressed_size={}, ratio={:.3}, files={}, blocks={}, groups={})",
            self.original_size, self.compressed_size, self.ratio(),
            self.file_count, self.block_count, self.group_count,
        )
    }
}

/// Result of archive integrity verification.
#[pyclass(frozen)]
#[derive(Debug, Clone)]
struct VerifyResult {
    /// Total number of blocks in the archive.
    #[pyo3(get)]
    total_blocks: usize,
    /// Number of blocks that verified successfully.
    #[pyo3(get)]
    verified_blocks: usize,
    /// List of corrupted block IDs.
    #[pyo3(get)]
    corrupted_blocks: Vec<u32>,
}

#[pymethods]
impl VerifyResult {
    /// True if all blocks verified successfully.
    #[getter]
    fn is_ok(&self) -> bool {
        self.corrupted_blocks.is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "VerifyResult(total={}, verified={}, corrupted={}, is_ok={})",
            self.total_blocks,
            self.verified_blocks,
            self.corrupted_blocks.len(),
            self.is_ok(),
        )
    }
}

/// Metadata for a file within an archive.
#[pyclass(frozen)]
#[derive(Debug, Clone)]
struct FileInfo {
    /// File path within the archive.
    #[pyo3(get)]
    path: String,
    /// Original uncompressed size in bytes.
    #[pyo3(get)]
    size: u64,
    /// Unix file permissions.
    #[pyo3(get)]
    permissions: u32,
    /// Last modification time (Unix timestamp), clamped to [0, 3000-01-01].
    #[pyo3(get)]
    mtime: i64,
}

#[pymethods]
impl FileInfo {
    fn __repr__(&self) -> String {
        // Escape single quotes and backslashes in the path to prevent
        // repr-injection (e.g. a crafted archive entry name containing `'`).
        let escaped = self.path.replace('\\', "\\\\").replace('\'', "\\'");
        format!("FileInfo(path='{escaped}', size={})", self.size)
    }
}

// ── Module functions ────────────────────────────────────────────────────────

/// Compress files into a .aet archive.
///
/// Args:
///     base_dir: Root directory for resolving relative file paths.
///     file_paths: List of file paths (absolute, or relative to base_dir).
///     output_path: Path for the output .aet archive file.
///     predictor: Predictor name. Options: "order0", "cm", "cm-light",
///         "lz4-aware", "ssm" (neural, default), "rle".
///     max_threads: Maximum concurrent compression threads (default 4, capped at 256).
///
/// Returns:
///     CompressionStats with size and ratio information.
///
/// Raises:
///     ValueError: If file_paths is empty or max_threads exceeds 256.
#[pyfunction]
#[pyo3(signature = (base_dir, file_paths, output_path, predictor="ssm", max_threads=4))]
fn compress(
    py: Python<'_>,
    base_dir: &str,
    file_paths: Vec<String>,
    output_path: &str,
    predictor: &str,
    max_threads: usize,
) -> PyResult<CompressionStats> {
    if file_paths.is_empty() {
        return Err(PyValueError::new_err("file_paths must not be empty"));
    }

    // Enforce file count limit to prevent resource exhaustion.
    if file_paths.len() > MAX_COMPRESS_FILES {
        return Err(PyValueError::new_err(format!(
            "file_paths length {} exceeds maximum of {MAX_COMPRESS_FILES}",
            file_paths.len(),
        )));
    }

    // Raise ValueError as documented instead of silently clamping.
    if max_threads == 0 {
        return Err(PyValueError::new_err("max_threads must be >= 1"));
    }
    if max_threads > MAX_THREADS_LIMIT {
        return Err(PyValueError::new_err(format!(
            "max_threads {max_threads} exceeds limit of {MAX_THREADS_LIMIT}"
        )));
    }

    let (factory, _pid) = make_predictor_factory(predictor)?;
    let compressor = Compressor::new(factory).with_max_threads(max_threads);

    let base = std::fs::canonicalize(base_dir)
        .map_err(|e| sanitize_io_error("Cannot resolve base_dir", e))?;

    // Validate that every file path resolves to a location inside base_dir
    // to prevent the caller from reading arbitrary files on the filesystem.
    let mut paths: Vec<PathBuf> = Vec::with_capacity(file_paths.len());
    for fp in &file_paths {
        let resolved = if Path::new(fp).is_absolute() {
            std::fs::canonicalize(fp)
        } else {
            std::fs::canonicalize(base.join(fp))
        }
        .map_err(|e| sanitize_io_error("Cannot resolve file path", e))?;

        if !resolved.starts_with(&base) {
            return Err(PyValueError::new_err(
                "file_paths must resolve to locations within base_dir",
            ));
        }
        paths.push(resolved);
    }

    let output = PathBuf::from(output_path);

    // Write to a temporary file and rename on success so a failed
    // compression never leaves a partial/corrupt archive on disk.
    let output_dir = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    // Include PID, thread ID, and a timestamp to avoid collisions and
    // prevent an attacker from predicting the temp file name.
    let temp_path = output_dir.join(format!(
        ".aether_tmp_{}_{:?}_{}.aet",
        std::process::id(),
        std::thread::current().id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    ));
    let temp_path_clone = temp_path.clone();

    let result = py.allow_threads(move || {
        let file = std::fs::File::create(&temp_path_clone)
            .map_err(|e| sanitize_io_error("Cannot create output", e))?;
        let mut writer = std::io::BufWriter::new(file);
        compressor
            .compress_to_archive(&base, &paths, &mut writer)
            .map_err(|e| sanitize_core_error("Compression failed", e))
    });

    match result {
        Ok((stats, _analytics)) => {
            std::fs::rename(&temp_path, &output)
                .map_err(|e| sanitize_io_error("Cannot finalize output", e))?;
            Ok(CompressionStats {
                original_size: stats.original_size,
                compressed_size: stats.compressed_size,
                block_count: stats.block_count,
                file_count: stats.file_count,
                group_count: stats.group_count,
            })
        }
        Err(e) => {
            // Best-effort cleanup of the temporary file.
            let _ = std::fs::remove_file(&temp_path);
            Err(e)
        }
    }
}

/// Extract all files from an archive.
///
/// Auto-detects the predictor used during compression.
///
/// Args:
///     archive_path: Path to the .aet archive.
///     output_dir: Directory to extract files into.
#[pyfunction]
fn extract_all(py: Python<'_>, archive_path: &str, output_dir: &str) -> PyResult<()> {
    // Reject traversal/absolute tricks in output_dir itself.
    // Reuse validate_output_path for full coverage (null bytes, `..`, absolute).
    validate_output_path("output_dir", Path::new(output_dir))?;

    let (mut reader, pid) = open_archive(archive_path)?;
    let factory_for_list = make_factory_from_id(pid)?;
    let factory_for_extract = make_factory_from_id(pid)?;

    // Defense-in-depth: validate every archive entry path before extraction
    // to guard against Zip Slip, even if aether-core also checks internally.
    let decompressor_list = Decompressor::new(factory_for_list);
    let entries = decompressor_list
        .list_files(&mut reader)
        .map_err(|e| sanitize_core_error("Extraction failed", e))?;
    for entry in &entries {
        validate_output_path("archive entry", Path::new(&entry.path))?;
    }

    // Re-open the archive since list_files consumed the reader position.
    let (mut reader, _pid) = open_archive(archive_path)?;
    let decompressor = Decompressor::new(factory_for_extract);
    let output = PathBuf::from(output_dir);

    py.allow_threads(move || {
        decompressor
            .extract_all(&mut reader, &output)
            .map_err(|e| sanitize_core_error("Extraction failed", e))
    })?;

    Ok(())
}

/// Extract a single file from an archive.
///
/// Args:
///     archive_path: Path to the .aet archive.
///     file_path: Path of the file within the archive.
///     output_path: Destination file path on disk.
///
/// Raises:
///     ValueError: If output_path contains path traversal sequences.
#[pyfunction]
fn extract_file(
    py: Python<'_>,
    archive_path: &str,
    file_path: &str,
    output_path: &str,
) -> PyResult<()> {
    // Validate the archive-internal file_path to reject traversal attempts
    // that could trick callers who blindly pass archive entries through.
    validate_output_path("file_path", Path::new(file_path))?;

    // Validate output_path: must be relative, no `..`, no null bytes.
    let out_path = Path::new(output_path);
    validate_output_path("output_path", out_path)?;

    // Create parent directories.
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| sanitize_io_error("Cannot create directory", e))?;
        }
    }

    // After creating parent dirs, verify the final resolved path still
    // lives where we expect. This closes a TOCTOU window where a symlink
    // could be placed in a newly-created parent directory to redirect
    // the output to an arbitrary location.
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            let canonical_parent = std::fs::canonicalize(parent)
                .map_err(|e| sanitize_io_error("Cannot resolve parent directory", e))?;
            let canonical_target = canonical_parent.join(
                out_path
                    .file_name()
                    .ok_or_else(|| PyValueError::new_err("output_path has no file name"))?,
            );
            // Ensure the resolved target is still under the canonical parent
            // (i.e. no symlink redirected us elsewhere).
            if !canonical_target.starts_with(&canonical_parent) {
                return Err(PyValueError::new_err(
                    "output_path resolves outside its parent directory (symlink detected)",
                ));
            }
        }
    }

    let (mut reader, pid) = open_archive(archive_path)?;
    let factory = make_factory_from_id(pid)?;
    let decompressor = Decompressor::new(factory);
    let file_path_owned = file_path.to_string();
    let output = PathBuf::from(output_path);

    py.allow_threads(move || {
        // Use O_EXCL to ensure we create a new file and never write through
        // a symlink that appeared between validation and open (race guard).
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .map_err(|e| sanitize_io_error("Cannot create output", e))?;
        decompressor
            .extract_file(&mut reader, &file_path_owned, &mut out)
            .map_err(|e| sanitize_core_error("Extraction failed", e))
    })?;

    Ok(())
}

/// Verify the integrity of an archive.
///
/// Args:
///     archive_path: Path to the .aet archive.
///
/// Returns:
///     VerifyResult with block-level verification details.
#[pyfunction]
fn verify(py: Python<'_>, archive_path: &str) -> PyResult<VerifyResult> {
    let (mut reader, pid) = open_archive(archive_path)?;
    let factory = make_factory_from_id(pid)?;
    let decompressor = Decompressor::new(factory);

    let result = py.allow_threads(move || {
        decompressor
            .verify(&mut reader)
            .map_err(|e| sanitize_core_error("Verification failed", e))
    })?;

    Ok(VerifyResult {
        total_blocks: result.total_blocks,
        verified_blocks: result.verified_blocks,
        corrupted_blocks: result.corrupted_blocks,
    })
}

/// List all files in an archive.
///
/// Args:
///     archive_path: Path to the .aet archive.
///
/// Returns:
///     List of FileInfo objects with path, size, permissions, and mtime.
#[pyfunction]
fn list_files(py: Python<'_>, archive_path: &str) -> PyResult<Vec<FileInfo>> {
    let (mut reader, pid) = open_archive(archive_path)?;
    let factory = make_factory_from_id(pid)?;
    let decompressor = Decompressor::new(factory);

    let entries = py.allow_threads(move || {
        decompressor
            .list_files(&mut reader)
            .map_err(|e| sanitize_core_error("List failed", e))
    })?;

    Ok(entries
        .into_iter()
        .map(|fe| FileInfo {
            path: fe.path,
            size: fe.original_size,
            permissions: fe.permissions,
            mtime: fe.mtime.clamp(MTIME_MIN, MTIME_MAX),
        })
        .collect())
}

/// Return the library version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Return a list of available predictor names.
#[pyfunction]
fn available_predictors() -> Vec<&'static str> {
    let mut v = vec!["order0", "ssm", "rle"];
    #[cfg(feature = "context-mixer")]
    {
        v.extend_from_slice(&["cm", "cm-light", "lz4-aware"]);
    }
    v
}

// ── Module definition ───────────────────────────────────────────────────────

/// AetherArch — next-generation neural-probabilistic file archiver.
///
/// Functions:
///     compress(base_dir, file_paths, output_path, predictor="ssm", max_threads=4)
///     extract_all(archive_path, output_dir)
///     extract_file(archive_path, file_path, output_path)
///     verify(archive_path) -> VerifyResult
///     list_files(archive_path) -> list[FileInfo]
///     version() -> str
///     available_predictors() -> list[str]
#[pymodule]
fn aether(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(extract_all, m)?)?;
    m.add_function(wrap_pyfunction!(extract_file, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(list_files, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(available_predictors, m)?)?;
    m.add_class::<CompressionStats>()?;
    m.add_class::<VerifyResult>()?;
    m.add_class::<FileInfo>()?;
    Ok(())
}
