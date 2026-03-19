//! WebAssembly bindings for AetherArch decompression.
//!
//! Provides a decompress-only API for browser-based archive reading.
//! Uses `--no-default-features` to exclude ContextMixer (~100 MiB) and lz4_flex (C code).
//!
//! # Building
//!
//! ```sh
//! # Install wasm target
//! rustup target add wasm32-unknown-unknown
//!
//! # Build (requires wasm-pack for JS bindings)
//! wasm-pack build aether-wasm --target web
//! ```
//!
//! # Limitations
//!
//! - Decompress-only: compression is not exposed
//! - Only NeuralSsm, RLE, Order0, and Mtf predictors available
//! - Archives using ContextMixer or LZ4-aware predictors cannot be decompressed
//! - Zstd-compressed blocks require the zstd crate (uses C FFI), which may not
//!   compile to wasm without wasm32-wasi or emscripten target

use wasm_bindgen::prelude::*;

use std::io::{self, Cursor, Write};

use aether_core::entropy::{
    MtfPredictor, NeuralSsmPredictor, Order0Model, ProbabilityPredictor, RlePredictor,
};
use aether_core::format::PredictorId;
use aether_core::header::FileEntry;
use aether_core::pipeline::decompress::Decompressor;

/// Sanitize an internal error message before returning it to JavaScript.
///
/// Strips potentially sensitive information (file paths, memory addresses,
/// internal struct names) while preserving the actionable error description.
fn sanitize_error(msg: &str) -> String {
    // Truncate excessively long error messages that could leak internal state.
    // Use char_indices to avoid panicking on multi-byte UTF-8 boundaries.
    const MAX_ERROR_LEN: usize = 256;
    if msg.len() > MAX_ERROR_LEN {
        let boundary = msg
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= MAX_ERROR_LEN)
            .last()
            .unwrap_or(0);
        format!("{}... (truncated)", &msg[..boundary])
    } else {
        msg.to_string()
    }
}

/// Returns true if a normalized path contains traversal sequences or is absolute.
fn is_path_traversal(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    // Reject absolute paths and traversal sequences
    normalized.starts_with('/')
        || normalized.contains("/../")
        || normalized.starts_with("../")
        || normalized.ends_with("/..")
        || normalized == ".."
        || normalized.contains("://")
}

/// A writer wrapper that enforces a maximum byte limit, returning an error
/// if the write would exceed the cap. Prevents decompression bombs from
/// exhausting wasm linear memory before a post-check can fire.
struct LimitedWriter {
    buf: Vec<u8>,
    limit: u64,
    written: u64,
}

impl LimitedWriter {
    fn new(limit: u64) -> Self {
        Self {
            buf: Vec::new(),
            limit,
            written: 0,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.buf
    }
}

impl Write for LimitedWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.written = self.written.saturating_add(data.len() as u64);
        if self.written > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "decompressed output exceeds wasm extraction limit",
            ));
        }
        self.buf.write(data)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buf.flush()
    }
}

/// Maximum number of file entries returned by `list_files` to prevent
/// excessive JSON serialization in the wasm memory space.
const MAX_WASM_LIST_ENTRIES: usize = 100_000;

/// Maximum decompressed file size allowed in `extract_file` to prevent
/// wasm linear memory exhaustion.  Wasm memory is typically limited to
/// 2–4 GiB; keeping the extraction cap well below that avoids OOM aborts.
const MAX_WASM_EXTRACT_BYTES: u64 = 256 * 1024 * 1024; // 256 MiB

/// V11 fix: Maximum total decompressed size across all files in an archive.
/// Even if individual files are within the per-file limit, many files could
/// exhaust wasm memory. Cap total extraction at 512 MiB.
const MAX_WASM_TOTAL_DECOMPRESSED: u64 = 512 * 1024 * 1024; // 512 MiB

/// Validated archive metadata returned by `make_decompressor_with_entries`.
struct ValidatedArchive {
    decompressor: Decompressor,
    entries: Vec<FileEntry>,
}

/// Create a decompressor and extract file entries from the archive metadata.
///
/// Performs all shared validation upfront: predictor support, entry count cap,
/// path traversal rejection, and total decompressed size check.
fn make_decompressor_with_entries(archive_bytes: &[u8]) -> Result<ValidatedArchive, JsError> {
    let metadata = Decompressor::read_metadata_streaming(&mut &archive_bytes[..])
        .map_err(|e| JsError::new(&sanitize_error(&e.to_string())))?;

    let factory: Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync> = match metadata
        .header
        .predictor_id
    {
        PredictorId::Order0 => Box::new(|| Box::new(Order0Model::new())),
        PredictorId::Rle => Box::new(|| Box::new(RlePredictor::new())),
        PredictorId::NeuralSsm => Box::new(|| Box::new(NeuralSsmPredictor::new())),
        PredictorId::Mtf => Box::new(|| Box::new(MtfPredictor::new())),
        _other => {
            return Err(JsError::new(
                    "Unsupported predictor in wasm build — only Order0, Rle, NeuralSsm, and Mtf are available",
                ));
        }
    };

    let entries = metadata.file_entries;

    // Enforce entry count cap to prevent excessive memory use during parsing
    if entries.len() > MAX_WASM_LIST_ENTRIES {
        return Err(JsError::new(&format!(
            "Archive contains too many files (limit: {})",
            MAX_WASM_LIST_ENTRIES
        )));
    }

    // Reject paths with traversal sequences to protect downstream consumers
    for entry in &entries {
        if is_path_traversal(&entry.path) {
            return Err(JsError::new(
                "Archive contains path traversal sequences and cannot be processed",
            ));
        }
    }

    // Validate total decompressed size (saturating to avoid u64 overflow)
    let total_size: u64 = entries
        .iter()
        .fold(0u64, |acc, e| acc.saturating_add(e.original_size));
    if total_size > MAX_WASM_TOTAL_DECOMPRESSED {
        return Err(JsError::new(&format!(
            "Archive total decompressed size exceeds wasm limit of {} bytes",
            MAX_WASM_TOTAL_DECOMPRESSED,
        )));
    }

    Ok(ValidatedArchive {
        decompressor: Decompressor::new(factory),
        entries,
    })
}

/// Create a decompressor that auto-detects the predictor from archive metadata.
fn make_decompressor(archive_bytes: &[u8]) -> Result<Decompressor, JsError> {
    make_decompressor_with_entries(archive_bytes).map(|v| v.decompressor)
}

/// Verify archive integrity from a byte buffer.
///
/// Returns a JSON string with verification results.
/// Auto-detects the predictor from the archive header.
#[wasm_bindgen]
pub fn verify(archive_bytes: &[u8]) -> Result<String, JsError> {
    // S15 security fix: catch panics to prevent wasm abort.
    //
    // SAFETY INVARIANT (V6): AssertUnwindSafe is sound here because each call
    // creates a fresh Decompressor and Cursor on the stack — no mutable static
    // or shared state is accessed. If Decompressor or its predictor factory
    // ever captures shared mutable state (e.g. a global cache), this usage
    // becomes unsound and must be revisited.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| verify_inner(archive_bytes)))
        .unwrap_or_else(|_| Err(JsError::new("Internal error during verification")))
}

fn verify_inner(archive_bytes: &[u8]) -> Result<String, JsError> {
    let decompressor = make_decompressor(archive_bytes)?;
    let mut reader = Cursor::new(archive_bytes);

    let result = decompressor
        .verify(&mut reader)
        .map_err(|e| JsError::new(&sanitize_error(&e.to_string())))?;

    #[derive(serde::Serialize)]
    struct VerifyResult {
        ok: bool,
        verified_blocks: u64,
        total_blocks: u64,
        corrupted: Vec<u32>,
    }

    let output = VerifyResult {
        ok: result.is_ok(),
        verified_blocks: result.verified_blocks as u64,
        total_blocks: result.total_blocks as u64,
        corrupted: result.corrupted_blocks.clone(),
    };

    serde_json::to_string(&output).map_err(|_| JsError::new("JSON serialization failed"))
}

/// List files in an archive from a byte buffer.
///
/// Returns a JSON array of file entries.
/// Auto-detects the predictor from the archive header.
#[wasm_bindgen]
pub fn list_files(archive_bytes: &[u8]) -> Result<String, JsError> {
    // S15 security fix: catch panics to prevent wasm abort.
    // SAFETY INVARIANT (V6): see verify() comment — same reasoning applies.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        list_files_inner(archive_bytes)
    }))
    .unwrap_or_else(|_| Err(JsError::new("Internal error during list_files")))
}

fn list_files_inner(archive_bytes: &[u8]) -> Result<String, JsError> {
    // Entry count, path traversal, and total size are validated in make_decompressor_with_entries.
    let archive = make_decompressor_with_entries(archive_bytes)?;

    #[derive(serde::Serialize)]
    struct FileInfo {
        path: String,
        size: u64,
    }

    let file_infos: Vec<FileInfo> = archive
        .entries
        .iter()
        .map(|e| FileInfo {
            path: e.path.replace('\\', "/"),
            size: e.original_size,
        })
        .collect();

    serde_json::to_string(&file_infos).map_err(|_| JsError::new("JSON serialization failed"))
}

/// Decompress a single file from an archive by path.
///
/// Returns the decompressed file content as bytes.
/// Auto-detects the predictor from the archive header.
#[wasm_bindgen]
pub fn extract_file(archive_bytes: &[u8], file_path: &str) -> Result<Vec<u8>, JsError> {
    // S15 security fix: catch panics to prevent wasm abort.
    // SAFETY INVARIANT (V6): see verify() comment — same reasoning applies.
    let file_path_owned = file_path.to_string();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        extract_file_inner(archive_bytes, &file_path_owned)
    }))
    .unwrap_or_else(|_| Err(JsError::new("Internal error during extract_file")))
}

fn extract_file_inner(archive_bytes: &[u8], file_path: &str) -> Result<Vec<u8>, JsError> {
    // Entry count, path traversal, and total size are validated in make_decompressor_with_entries.
    let archive = make_decompressor_with_entries(archive_bytes)?;
    let mut reader = Cursor::new(archive_bytes);

    let entry = archive
        .entries
        .iter()
        .find(|e| e.path == file_path || e.path.replace('\\', "/") == file_path)
        .ok_or_else(|| JsError::new("File not found in archive"))?;

    if entry.original_size > MAX_WASM_EXTRACT_BYTES {
        return Err(JsError::new("File size exceeds wasm extraction limit"));
    }

    // V5 fix: use the entry's actual path for extraction, not the caller's
    // potentially-different path string, to ensure the metadata pre-check
    // and the actual extraction target the same file.
    let matched_path = entry.path.clone();

    // Use LimitedWriter to abort mid-stream if decompressed output exceeds the
    // cap. This prevents decompression bombs from exhausting wasm linear memory
    // before a post-check could fire.
    let mut output = LimitedWriter::new(MAX_WASM_EXTRACT_BYTES);
    archive
        .decompressor
        .extract_file(&mut reader, &matched_path, &mut output)
        .map_err(|e| JsError::new(&sanitize_error(&e.to_string())))?;

    Ok(output.into_inner())
}
