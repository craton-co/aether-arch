// Fix #15: enforce explicit unsafe blocks inside unsafe functions.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

//! C FFI bindings for AetherArch compression library.
//!
//! Provides an opaque-handle API for embedding AetherArch in C/C++ programs.
//! All functions return `i32` status codes: 0 for success, negative for errors.
//! Use [`aether_last_error`] to retrieve a human-readable error string.
//!
//! # Thread Safety
//!
//! Each `AetherCompressor` / `AetherDecompressor` handle is **not** thread-safe.
//! Create separate handles per thread, or synchronize externally.
//! Concurrent access to the same handle from multiple threads is undefined behavior.
//!
//! # Memory Management
//!
//! All handles returned by `aether_*_new` must be freed with the corresponding
//! `aether_*_free` function. Lists returned by `aether_list` must be freed with
//! `aether_file_list_free`.
//!
//! # Dynamic Loading
//!
//! This library must **not** be unloaded via `dlclose()` (or `FreeLibrary()` on
//! Windows) while any thread that has called an FFI function is still alive.
//! Doing so invalidates thread-local storage and causes undefined behavior.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "context-mixer")]
use aether_core::entropy::context_mixer::ContextMixerConfig;
#[cfg(feature = "context-mixer")]
use aether_core::entropy::{ContextMixer, Lz4AwarePredictor};
use aether_core::entropy::{NeuralSsmPredictor, Order0Model, ProbabilityPredictor, RlePredictor};
use aether_core::format::PredictorId;
use aether_core::header::ArchiveHeader;
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

// ── Error handling ──────────────────────────────────────────────────────────

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Maximum error message length to prevent unbounded thread-local growth.
const MAX_ERROR_LEN: usize = 1024;

/// Maximum number of files accepted via FFI for both compression and extraction.
/// Fix #16: shared limit to prevent OOM from directory-bomb attacks during extraction
/// (M9 already enforced this for compression only).
const MAX_FFI_FILES: usize = 1_000_000;

fn set_last_error(msg: String) {
    LAST_ERROR.with(|cell| {
        // Replace interior null bytes with '?' so the message isn't silently truncated
        let sanitized = msg.replace('\0', "?");
        // Fix #10: sanitize OS-level error details that may leak internal paths,
        // usernames, or filesystem structure to untrusted callers.
        let mut sanitized = sanitize_error_message(&sanitized);
        // Cap error message length to prevent unbounded thread-local growth.
        if sanitized.len() > MAX_ERROR_LEN {
            sanitized.truncate(MAX_ERROR_LEN);
            sanitized.push_str("...");
        }
        // Use try_borrow_mut to avoid panicking if the RefCell is already
        // borrowed (e.g. due to a panic during a previous borrow_mut scope).
        if let Ok(mut guard) = cell.try_borrow_mut() {
            *guard = CString::new(sanitized).ok();
        }
    });
}

/// Strip OS-specific path details from error messages while keeping the
/// error class (e.g. "permission denied", "not found") useful for debugging.
///
/// # Fix #16: improved sanitizer coverage
///
/// Now also redacts:
/// - Relative paths with `..` components (e.g. `../../etc/passwd`)
/// - Windows UNC paths (e.g. `\\server\share\file`)
/// - Paths embedded in archive entry error messages
fn sanitize_error_message(msg: &str) -> String {
    let mut result = String::with_capacity(msg.len());
    let bytes = msg.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let ch = bytes[i];

        // Detect relative paths starting with ../ or ..\
        if ch == b'.'
            && i + 2 < len
            && bytes[i + 1] == b'.'
            && (bytes[i + 2] == b'/' || bytes[i + 2] == b'\\')
        {
            // Scan forward past the entire path
            while i < len
                && !bytes[i].is_ascii_whitespace()
                && bytes[i] != b'"'
                && bytes[i] != b'\''
            {
                i += 1;
            }
            result.push_str("<path>");
            continue;
        }

        // Detect Unix absolute paths: / followed by alphanumeric
        if ch == b'/' && i + 1 < len && bytes[i + 1].is_ascii_alphanumeric() {
            let start = i;
            // Scan forward past path characters
            while i < len && is_path_char(bytes[i]) {
                i += 1;
            }
            // Only redact if it looks like a real path (has at least 2 slashes)
            let segment = &msg[start..i];
            if segment.matches('/').count() >= 2 {
                result.push_str("<path>");
            } else {
                result.push_str(segment);
            }
            continue;
        }

        // Detect Windows absolute paths: C:\ or C:/
        if i + 2 < len
            && ch.is_ascii_alphabetic()
            && bytes[i + 1] == b':'
            && (bytes[i + 2] == b'\\' || bytes[i + 2] == b'/')
        {
            // Scan forward past path characters
            while i < len
                && !bytes[i].is_ascii_whitespace()
                && bytes[i] != b'"'
                && bytes[i] != b'\''
            {
                i += 1;
            }
            result.push_str("<path>");
            continue;
        }

        // Detect Windows UNC paths: \\server\share or //server/share
        if i + 1 < len
            && ((ch == b'\\' && bytes[i + 1] == b'\\') || (ch == b'/' && bytes[i + 1] == b'/'))
            && i + 2 < len
            && bytes[i + 2].is_ascii_alphanumeric()
        {
            while i < len
                && !bytes[i].is_ascii_whitespace()
                && bytes[i] != b'"'
                && bytes[i] != b'\''
            {
                i += 1;
            }
            result.push_str("<path>");
            continue;
        }

        result.push(ch as char);
        i += 1;
    }

    result
}

/// Helper: returns true if the byte is a plausible path character.
#[inline]
fn is_path_char(b: u8) -> bool {
    b == b'/' || b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-' || b == b'+'
}

/// Return a **caller-owned** copy of the last error message, or `NULL` if no
/// error has occurred.
///
/// The caller must free the returned pointer with [`aether_error_free`].
///
/// # H1 security fix
///
/// Previous versions returned a borrowed pointer into thread-local storage,
/// which became dangling after the next FFI call. This version returns an
/// owned copy that is valid until explicitly freed.
#[no_mangle]
pub extern "C" fn aether_last_error() -> *mut c_char {
    mark_ffi_used();
    LAST_ERROR.with(|cell| {
        // Use try_borrow to avoid panicking if the RefCell is already
        // mutably borrowed (e.g. re-entrant call during error handling).
        let borrow = match cell.try_borrow() {
            Ok(b) => b,
            Err(_) => return ptr::null_mut(),
        };
        match &*borrow {
            Some(s) => {
                // Return an owned copy so the caller is not racing with TLS
                match CString::new(s.to_bytes()) {
                    Ok(copy) => copy.into_raw(),
                    Err(_) => ptr::null_mut(),
                }
            }
            None => ptr::null_mut(),
        }
    })
}

/// Free an error string returned by [`aether_last_error`].
///
/// No-op if `err` is `NULL`.
#[no_mangle]
pub extern "C" fn aether_error_free(err: *mut c_char) {
    if !err.is_null() {
        unsafe {
            drop(CString::from_raw(err));
        }
    }
}

/// Return the library version string (e.g. "0.2.2").
///
/// The returned pointer is valid for the lifetime of the library and must
/// **not** be freed by the caller.
#[no_mangle]
pub extern "C" fn aether_version() -> *const c_char {
    // L3 fix: use a CStr constant for clarity and safety.
    // The trailing \0 is explicit so this cannot accidentally become a
    // buffer over-read if the pattern is modified.
    static VERSION: &CStr =
        match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
            Ok(c) => c,
            Err(_) => panic!("CARGO_PKG_VERSION contains an interior null byte"),
        };
    VERSION.as_ptr()
}

// ── Fix #13: dlclose guard ───────────────────────────────────────────────────

/// Tracks whether any FFI function has been called on this thread.
/// Once set, unloading the library (dlclose / FreeLibrary) while the thread
/// is still alive is undefined behavior because thread-local storage is
/// invalidated.
static FFI_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Mark the library as actively used. Call at the top of every public FFI
/// function that touches thread-local state.
///
/// Uses `Ordering::Relaxed` intentionally: this is a monotonic flag (false→true)
/// and is not used as a synchronization primitive. A stale `false` read is
/// acceptable — it only means the caller is more conservative about dlclose.
#[inline]
fn mark_ffi_used() {
    FFI_INITIALIZED.store(true, Ordering::Relaxed);
}

/// Check if the library has been used. Useful for C consumers that want to
/// verify before calling `dlclose`.
///
/// Returns 1 if any FFI function has been called, 0 otherwise.
#[no_mangle]
pub extern "C" fn aether_is_active() -> i32 {
    if FFI_INITIALIZED.load(Ordering::Relaxed) {
        1
    } else {
        0
    }
}

// ── Status codes ────────────────────────────────────────────────────────────

/// Operation completed successfully.
pub const AETHER_OK: i32 = 0;
/// A required pointer argument was NULL.
pub const AETHER_ERR_NULL_PTR: i32 = -1;
/// A string argument contained invalid UTF-8.
pub const AETHER_ERR_INVALID_UTF8: i32 = -2;
/// An I/O error occurred (file not found, permission denied, etc.).
pub const AETHER_ERR_IO: i32 = -3;
/// Compression or decompression failed.
pub const AETHER_ERR_COMPRESSION: i32 = -4;
/// Invalid or unsupported predictor ID.
pub const AETHER_ERR_INVALID_PREDICTOR: i32 = -5;
/// Archive format error (corrupt header, bad magic, etc.).
pub const AETHER_ERR_ARCHIVE: i32 = -6;
/// Path traversal detected in archive entry.
pub const AETHER_ERR_PATH_TRAVERSAL: i32 = -7;

// ── Predictor ID enum (mirrors PredictorId) ─────────────────────────────────

/// Predictor algorithm selector for C callers.
///
/// **Note:** This enum is `#[repr(u16)]` but C callers should use the
/// `AETHER_PREDICTOR_ID_*` constants. The FFI functions accept a raw `u16`
/// and validate it, so passing an out-of-range value is safe (returns an
/// error, not undefined behavior).
#[repr(u16)]
#[derive(Debug, Clone, Copy)]
pub enum AetherPredictorId {
    /// Simple order-0 frequency model (~1 KiB memory).
    Order0 = 0,
    /// PAQ-inspired context mixer (~100 MiB memory).
    ContextMixer = 1,
    /// Neural SSM + RLE hybrid (best quality, ~25 KiB memory).
    NeuralSsm = 2,
    /// Lightweight context mixer (~4 MiB memory).
    ContextMixerLight = 3,
    /// LZ4-aware FSM predictor.
    Lz4Aware = 4,
    /// Hierarchical RLE predictor for BWT streams.
    Rle = 5,
}

/// M1 security fix: convert a raw u16 to a validated AetherPredictorId.
/// Returns `Err` with an error code for out-of-range values instead of UB.
fn validated_predictor_id(raw: u16) -> Result<AetherPredictorId, i32> {
    match raw {
        0 => Ok(AetherPredictorId::Order0),
        1 => Ok(AetherPredictorId::ContextMixer),
        2 => Ok(AetherPredictorId::NeuralSsm),
        3 => Ok(AetherPredictorId::ContextMixerLight),
        4 => Ok(AetherPredictorId::Lz4Aware),
        5 => Ok(AetherPredictorId::Rle),
        _ => {
            set_last_error(format!("Invalid predictor ID: {raw}"));
            Err(AETHER_ERR_INVALID_PREDICTOR)
        }
    }
}

fn predictor_id_to_internal(id: AetherPredictorId) -> PredictorId {
    match id {
        AetherPredictorId::Order0 => PredictorId::Order0,
        AetherPredictorId::ContextMixer => PredictorId::ContextMixer,
        AetherPredictorId::NeuralSsm => PredictorId::NeuralSsm,
        AetherPredictorId::ContextMixerLight => PredictorId::ContextMixerLight,
        AetherPredictorId::Lz4Aware => PredictorId::Lz4Aware,
        AetherPredictorId::Rle => PredictorId::Rle,
    }
}

fn make_factory(
    id: PredictorId,
) -> Result<Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync>, String> {
    match id {
        PredictorId::Order0 => Ok(Box::new(|| Box::new(Order0Model::new()))),
        #[cfg(feature = "context-mixer")]
        PredictorId::ContextMixer => Ok(Box::new(|| {
            Box::new(ContextMixer::with_config(ContextMixerConfig::default()))
        })),
        #[cfg(feature = "context-mixer")]
        PredictorId::ContextMixerLight => Ok(Box::new(|| {
            Box::new(ContextMixer::with_config(ContextMixerConfig::lightweight()))
        })),
        #[cfg(feature = "context-mixer")]
        PredictorId::Lz4Aware => Ok(Box::new(|| Box::new(Lz4AwarePredictor::new()))),
        PredictorId::NeuralSsm => Ok(Box::new(|| Box::new(NeuralSsmPredictor::new()))),
        PredictorId::Rle => Ok(Box::new(|| Box::new(RlePredictor::new()))),
        PredictorId::ZstdOnly => Ok(Box::new(|| Box::new(Order0Model::new()))),
        #[cfg(not(feature = "context-mixer"))]
        PredictorId::ContextMixer | PredictorId::ContextMixerLight | PredictorId::Lz4Aware => Err(
            format!("Predictor {:?} requires the 'context-mixer' feature", id),
        ),
        // Fix #11: explicitly error instead of silently falling back to
        // NeuralSsm for unknown PredictorId variants. Silent fallback risks
        // data corruption if aether-core adds new variants.
        other => Err(format!("Unsupported predictor: {other:?}")),
    }
}

// ── Helper: C string → Path ─────────────────────────────────────────────────

/// Maximum length (in bytes) we will scan for a null terminator.
/// Fix #6: prevents unbounded memory reads from a non-terminated C string.
const MAX_CSTR_LEN: usize = 64 * 1024; // 64 KiB — generous for any path

/// Safely convert a C string pointer to a `CStr` with bounded scanning.
///
/// Scans at most `MAX_CSTR_LEN` bytes **one byte at a time** looking for a
/// null terminator. Returns an error if the pointer is null or no terminator
/// is found.
///
/// # Fix #16: safe page-boundary scanning
///
/// Previous versions created a 64 KiB slice from the pointer unconditionally,
/// which could read past the allocation into unmapped memory if the string
/// was near a page boundary. This version reads byte-by-byte to avoid
/// accessing memory beyond the actual null terminator.
///
/// # Safety
/// The pointer must point to a valid C string (null-terminated) whose length
/// (including the terminator) does not exceed `MAX_CSTR_LEN` bytes.
unsafe fn bounded_cstr<'a>(s: *const c_char) -> Result<&'a CStr, i32> {
    if s.is_null() {
        set_last_error("Null pointer argument".into());
        return Err(AETHER_ERR_NULL_PTR);
    }
    // Scan byte-by-byte to avoid reading past the allocation boundary.
    // This is critical when the string is near the end of a mapped page.
    let mut len: usize = 0;
    unsafe {
        while len < MAX_CSTR_LEN {
            if *s.add(len) == 0 {
                // SAFETY: we found a null terminator at offset `len`, so
                // CStr::from_ptr will read exactly `len + 1` bytes.
                return Ok(CStr::from_ptr(s));
            }
            len += 1;
        }
    }
    set_last_error("C string exceeds maximum length or is not null-terminated".into());
    Err(AETHER_ERR_INVALID_UTF8)
}

/// Convert a C string pointer to an owned `PathBuf`.
///
/// # Safety
/// The pointer must be valid for reading up to `MAX_CSTR_LEN` bytes.
unsafe fn cstr_to_pathbuf(s: *const c_char) -> Result<PathBuf, i32> {
    let c = unsafe { bounded_cstr(s)? };
    let s = c.to_str().map_err(|_| {
        set_last_error("Invalid UTF-8 in path".into());
        AETHER_ERR_INVALID_UTF8
    })?;
    Ok(PathBuf::from(s))
}

/// Convert a C string pointer to an owned `String`.
///
/// # Safety
/// The pointer must be valid for reading up to `MAX_CSTR_LEN` bytes.
unsafe fn cstr_to_string(s: *const c_char) -> Result<String, i32> {
    let c = unsafe { bounded_cstr(s)? };
    c.to_str().map(|s| s.to_owned()).map_err(|_| {
        set_last_error("Invalid UTF-8 in string".into());
        AETHER_ERR_INVALID_UTF8
    })
}

// ── Path traversal protection ───────────────────────────────────────────────

/// M4 security fix: validate that `target` does not escape `base_dir`.
///
/// Prevents zip-slip attacks where a malicious archive contains entries
/// like `../../etc/passwd` that write outside the output directory.
///
/// Fix #4: also rejects paths whose resolved destination is a symlink,
/// preventing symlink-based traversal.
/// Fix #5: also rejects Windows drive-relative paths like `C:file.txt`
/// and extended-length prefixes like `\\?\`.
fn validate_no_path_traversal(base_dir: &Path, target: &Path) -> Result<(), i32> {
    let target_str = target.to_string_lossy();

    // Reject empty paths
    if target_str.is_empty() {
        set_last_error("Empty path in archive entry".into());
        return Err(AETHER_ERR_PATH_TRAVERSAL);
    }

    // Reject absolute paths (Unix `/...` or Windows `C:\...`, `\\...`)
    // Fix #16: do not embed the raw path in error messages to avoid leaking
    // archive internals. The sanitizer may not catch all path patterns.
    if target.is_absolute() || target_str.starts_with('/') || target_str.starts_with('\\') {
        set_last_error("Absolute path in archive entry".into());
        return Err(AETHER_ERR_PATH_TRAVERSAL);
    }

    // Fix #5: reject Windows drive-relative paths (e.g. "C:file.txt") and
    // extended-length path prefixes (e.g. "\\?\C:\...").
    // A colon at position 1 indicates a drive letter (e.g. "C:foo").
    let bytes = target_str.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        set_last_error("Drive-relative path in archive entry".into());
        return Err(AETHER_ERR_PATH_TRAVERSAL);
    }
    // Reject \\?\ extended-length and \\.\  device paths
    if target_str.starts_with(r"\\?\") || target_str.starts_with(r"\\.\") {
        set_last_error("Extended-length path in archive entry".into());
        return Err(AETHER_ERR_PATH_TRAVERSAL);
    }

    // Track depth using raw string splitting for cross-platform reliability.
    // std::path::Component may normalize differently on Windows vs Unix.
    // A path like "a/b/../c" is fine (depth stays >= 0), but "../../etc"
    // escapes (depth goes negative).
    let mut depth: i32 = 0;
    for component in target_str.split(&['/', '\\']) {
        match component {
            ".." => depth -= 1,
            "" | "." => {}
            _ => depth += 1,
        }
        if depth < 0 {
            set_last_error("Path traversal detected in archive entry".into());
            return Err(AETHER_ERR_PATH_TRAVERSAL);
        }
    }

    // Fix #4: if the resolved target location already exists on disk, check
    // whether any component in the path is a symlink. This prevents an attacker
    // from planting a symlink at e.g. "innocent_dir" → "/etc" so that
    // "innocent_dir/file.txt" writes to "/etc/file.txt".
    let resolved = base_dir.join(target);
    // Walk each prefix of the resolved path to detect symlinks at any level.
    // We only need to check components under base_dir (the extraction root).
    let mut check_path = base_dir.to_path_buf();
    for component in target.components() {
        check_path.push(component);
        match check_path.symlink_metadata() {
            Ok(meta) if meta.file_type().is_symlink() => {
                set_last_error("Symlink in extraction path".into());
                return Err(AETHER_ERR_PATH_TRAVERSAL);
            }
            _ => {}
        }
    }
    // Also verify the final resolved path doesn't escape base_dir via
    // canonicalization (catches symlink chains we might have missed).
    if resolved.exists() {
        if let (Ok(canon_base), Ok(canon_target)) =
            (base_dir.canonicalize(), resolved.canonicalize())
        {
            if !canon_target.starts_with(&canon_base) {
                set_last_error("Resolved path escapes output directory".into());
                return Err(AETHER_ERR_PATH_TRAVERSAL);
            }
        }
    }

    Ok(())
}

// ── Compressor ──────────────────────────────────────────────────────────────

/// Opaque compressor handle.
pub struct AetherCompressor {
    inner: Compressor,
}

/// Create a new compressor with the specified predictor.
///
/// # M1 security fix
///
/// Accepts a raw `u16` instead of the enum type to prevent undefined
/// behavior when C callers pass out-of-range values.
///
/// Returns `NULL` on failure (check `aether_last_error()`).
#[no_mangle]
pub extern "C" fn aether_compressor_new(predictor_id: u16) -> *mut AetherCompressor {
    mark_ffi_used();
    let id = match validated_predictor_id(predictor_id) {
        Ok(id) => id,
        Err(_) => return ptr::null_mut(),
    };
    let pid = predictor_id_to_internal(id);
    match make_factory(pid) {
        Ok(factory) => {
            let compressor = Compressor::new(factory);
            Box::into_raw(Box::new(AetherCompressor { inner: compressor }))
        }
        Err(msg) => {
            set_last_error(msg);
            ptr::null_mut()
        }
    }
}

/// Set the maximum number of concurrent compression threads.
///
/// Set to 0 for unlimited (uses all available cores).
/// Default is 4 for memory backpressure.
///
/// # M3 security fix
///
/// Takes `*mut` consistently with other mutating functions.
///
/// # Fix #14
///
/// Returns `AETHER_OK` on success, `AETHER_ERR_NULL_PTR` if compressor
/// is `NULL`. Previously returned `void`, silently ignoring null pointers.
#[no_mangle]
#[must_use]
pub extern "C" fn aether_compressor_set_max_threads(
    compressor: *mut AetherCompressor,
    max_threads: u32,
) -> i32 {
    if compressor.is_null() {
        set_last_error("Null compressor".into());
        return AETHER_ERR_NULL_PTR;
    }
    let c = unsafe { &mut *compressor };
    c.inner.set_max_threads(max_threads as usize);
    AETHER_OK
}

/// Free a compressor handle and set the caller's pointer to `NULL`.
///
/// No-op if `*compressor` is `NULL` or `compressor` is `NULL`.
///
/// # H2 security fix
///
/// Accepts `*mut *mut` so the caller's pointer is nulled after free,
/// preventing double-free and use-after-free bugs.
#[no_mangle]
pub extern "C" fn aether_compressor_free(compressor: *mut *mut AetherCompressor) {
    if compressor.is_null() {
        return;
    }
    let ptr = unsafe { *compressor };
    if !ptr.is_null() {
        unsafe {
            *compressor = ptr::null_mut();
            drop(Box::from_raw(ptr));
        }
    }
}

/// Compress files into a `.aet` archive.
///
/// - `base_dir`: root directory for resolving relative file paths.
/// - `file_paths`: array of C strings with file paths relative to `base_dir`.
/// - `file_count`: number of entries in `file_paths`.
/// - `output_path`: path for the output `.aet` archive.
///
/// # M3 security fix
///
/// Takes `*mut` for the compressor handle for consistent aliasing semantics.
///
/// Returns `AETHER_OK` (0) on success, negative error code on failure.
#[no_mangle]
#[must_use]
pub extern "C" fn aether_compress(
    compressor: *mut AetherCompressor,
    base_dir: *const c_char,
    file_paths: *const *const c_char,
    file_count: u32,
    output_path: *const c_char,
) -> i32 {
    // Validate arguments
    if compressor.is_null() || file_paths.is_null() {
        set_last_error("Null pointer argument".into());
        return AETHER_ERR_NULL_PTR;
    }

    let base = match unsafe { cstr_to_pathbuf(base_dir) } {
        Ok(p) => p,
        Err(code) => return code,
    };
    let out = match unsafe { cstr_to_pathbuf(output_path) } {
        Ok(p) => p,
        Err(code) => return code,
    };

    if file_count == 0 {
        set_last_error("No files to compress".into());
        return AETHER_ERR_COMPRESSION;
    }
    // M9 security fix: bound file_count to prevent OOM from malicious callers.
    if file_count as usize > MAX_FFI_FILES {
        set_last_error(format!(
            "File count {file_count} exceeds maximum {MAX_FFI_FILES}"
        ));
        return AETHER_ERR_COMPRESSION;
    }

    // Collect file paths. The caller guarantees file_paths has file_count entries.
    let mut paths = Vec::with_capacity(file_count as usize);
    for i in 0..file_count as usize {
        let raw = unsafe { *file_paths.add(i) };
        match unsafe { cstr_to_string(raw) } {
            Ok(s) => paths.push(PathBuf::from(s)),
            Err(code) => return code,
        }
    }

    // Fix #3: use &mut to match the *mut pointer type, preventing UB from
    // concurrent &/&mut aliasing if a C caller races set_max_threads.
    let c = unsafe { &mut *compressor };

    // Create output file
    let file = match std::fs::File::create(out) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(format!("Failed to create output file: {e}"));
            return AETHER_ERR_IO;
        }
    };
    let mut writer = std::io::BufWriter::new(file);

    match c.inner.compress_to_archive(&base, &paths, &mut writer) {
        Ok((_stats, _analytics)) => AETHER_OK,
        Err(e) => {
            set_last_error(format!("Compression failed: {e}"));
            AETHER_ERR_COMPRESSION
        }
    }
}

// ── Decompressor ────────────────────────────────────────────────────────────

/// Opaque decompressor handle.
pub struct AetherDecompressor {
    inner: Decompressor,
}

/// Create a new decompressor with the specified predictor.
///
/// # M1 security fix
///
/// Accepts a raw `u16` instead of the enum type to prevent undefined
/// behavior when C callers pass out-of-range values.
///
/// Returns `NULL` on failure (check `aether_last_error()`).
#[no_mangle]
pub extern "C" fn aether_decompressor_new(predictor_id: u16) -> *mut AetherDecompressor {
    mark_ffi_used();
    let id = match validated_predictor_id(predictor_id) {
        Ok(id) => id,
        Err(_) => return ptr::null_mut(),
    };
    let pid = predictor_id_to_internal(id);
    match make_factory(pid) {
        Ok(factory) => {
            let decompressor = Decompressor::new(factory);
            Box::into_raw(Box::new(AetherDecompressor {
                inner: decompressor,
            }))
        }
        Err(msg) => {
            set_last_error(msg);
            ptr::null_mut()
        }
    }
}

/// Create a decompressor that auto-detects the predictor from an archive.
///
/// Reads the archive header to determine which predictor was used during
/// compression, then creates a decompressor configured for that predictor.
///
/// Returns `NULL` on failure (check `aether_last_error()`).
#[no_mangle]
pub extern "C" fn aether_decompressor_auto(archive_path: *const c_char) -> *mut AetherDecompressor {
    mark_ffi_used();
    let path = match unsafe { cstr_to_pathbuf(archive_path) } {
        Ok(p) => p,
        Err(_) => return ptr::null_mut(),
    };

    let pid = match detect_predictor(&path) {
        Ok(id) => id,
        Err(msg) => {
            set_last_error(msg);
            return ptr::null_mut();
        }
    };

    match make_factory(pid) {
        Ok(factory) => {
            let decompressor = Decompressor::new(factory);
            Box::into_raw(Box::new(AetherDecompressor {
                inner: decompressor,
            }))
        }
        Err(msg) => {
            set_last_error(msg);
            ptr::null_mut()
        }
    }
}

/// Free a decompressor handle and set the caller's pointer to `NULL`.
///
/// No-op if `*decompressor` is `NULL` or `decompressor` is `NULL`.
///
/// # H2 security fix
///
/// Accepts `*mut *mut` so the caller's pointer is nulled after free,
/// preventing double-free and use-after-free bugs.
#[no_mangle]
pub extern "C" fn aether_decompressor_free(decompressor: *mut *mut AetherDecompressor) {
    if decompressor.is_null() {
        return;
    }
    let ptr = unsafe { *decompressor };
    if !ptr.is_null() {
        unsafe {
            *decompressor = ptr::null_mut();
            drop(Box::from_raw(ptr));
        }
    }
}

/// Extract all files from an archive to the given output directory.
///
/// # M4 security fix
///
/// Validates that no extracted file path escapes the output directory
/// (zip-slip protection). If a path traversal is detected, returns
/// `AETHER_ERR_PATH_TRAVERSAL` (-7).
///
/// # Security: output directory requirements
///
/// The output directory (`output_dir`) **must not** be writable by untrusted
/// users. Symlink and path-traversal checks are performed before extraction
/// begins, but a local attacker with write access to the output tree can
/// race the check (TOCTOU) by replacing a directory with a symlink between
/// validation and the actual file write. For maximum safety, extract into a
/// directory owned exclusively by the calling process (e.g. a `mkdtemp`
/// result with `0700` permissions).
///
/// Returns `AETHER_OK` (0) on success, negative error code on failure.
#[no_mangle]
#[must_use]
pub extern "C" fn aether_extract_all(
    decompressor: *mut AetherDecompressor,
    archive_path: *const c_char,
    output_dir: *const c_char,
) -> i32 {
    if decompressor.is_null() {
        set_last_error("Null decompressor".into());
        return AETHER_ERR_NULL_PTR;
    }

    let archive = match unsafe { cstr_to_pathbuf(archive_path) } {
        Ok(p) => p,
        Err(code) => return code,
    };
    let outdir = match unsafe { cstr_to_pathbuf(output_dir) } {
        Ok(p) => p,
        Err(code) => return code,
    };

    let d = unsafe { &mut *decompressor };

    use std::io::Seek;

    // Fix #1 (TOCTOU): open the file once and use the same handle for both
    // validation and extraction. This prevents an attacker from swapping the
    // file between validation and extraction.
    let mut file = match std::fs::File::open(&archive) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(format!("Failed to open archive: {e}"));
            return AETHER_ERR_IO;
        }
    };

    // M4 security fix: pre-validate all archive entries for path traversal
    // before extracting anything.
    //
    // Fix #16: also enforce an entry count limit to prevent directory-bomb
    // attacks where a malicious archive contains millions of entries.
    {
        let entries = match d.inner.list_files(&mut file) {
            Ok(e) => e,
            Err(e) => {
                set_last_error(format!("Failed to list archive entries: {e}"));
                return AETHER_ERR_ARCHIVE;
            }
        };
        if entries.len() > MAX_FFI_FILES {
            set_last_error(format!(
                "Archive contains {} entries, exceeds maximum {}",
                entries.len(),
                MAX_FFI_FILES
            ));
            return AETHER_ERR_ARCHIVE;
        }
        for entry in &entries {
            if let Err(code) = validate_no_path_traversal(&outdir, Path::new(&entry.path)) {
                return code;
            }
        }
    }

    // Seek back to the beginning so extract_all reads from the start.
    if let Err(e) = file.seek(std::io::SeekFrom::Start(0)) {
        set_last_error(format!("Failed to seek archive: {e}"));
        return AETHER_ERR_IO;
    }

    // Fix #2: validate output directory is not a symlink itself to prevent
    // extraction to an attacker-controlled location.
    if let Ok(meta) = outdir.symlink_metadata() {
        if meta.file_type().is_symlink() {
            set_last_error("Output directory is a symlink".into());
            return AETHER_ERR_PATH_TRAVERSAL;
        }
    }

    // Ensure the output directory exists before extraction (core's
    // validate_resolved_path requires it for canonicalize).
    if let Err(e) = std::fs::create_dir_all(&outdir) {
        set_last_error(format!("Failed to create output directory: {e}"));
        return AETHER_ERR_IO;
    }

    match d.inner.extract_all(&mut file, &outdir) {
        Ok(()) => AETHER_OK,
        Err(e) => {
            set_last_error(format!("Extraction failed: {e}"));
            AETHER_ERR_ARCHIVE
        }
    }
}

/// Extract a single file from an archive.
///
/// - `file_path`: the path within the archive to extract.
/// - `output_path`: destination file path on disk.
///
/// # M4 security fix
///
/// Validates that the in-archive `file_path` does not contain path traversal.
///
/// # Security: caller responsibility for `output_path`
///
/// The `output_path` is **chosen by the caller** and is not validated against
/// a base directory. The caller is responsible for ensuring that `output_path`
/// is a safe location. The parent directory of `output_path` is checked for
/// symlinks, but the caller should still sanitize this value if it originates
/// from untrusted input.
///
/// Returns `AETHER_OK` (0) on success, negative error code on failure.
#[no_mangle]
#[must_use]
pub extern "C" fn aether_extract_file(
    decompressor: *mut AetherDecompressor,
    archive_path: *const c_char,
    file_path: *const c_char,
    output_path: *const c_char,
) -> i32 {
    if decompressor.is_null() {
        set_last_error("Null decompressor".into());
        return AETHER_ERR_NULL_PTR;
    }

    let archive = match unsafe { cstr_to_pathbuf(archive_path) } {
        Ok(p) => p,
        Err(code) => return code,
    };
    let file_name = match unsafe { cstr_to_string(file_path) } {
        Ok(s) => s,
        Err(code) => return code,
    };
    let out = match unsafe { cstr_to_pathbuf(output_path) } {
        Ok(p) => p,
        Err(code) => return code,
    };

    // M4 security fix: validate the in-archive path doesn't contain traversal
    if let Err(code) = validate_no_path_traversal(Path::new("."), Path::new(&file_name)) {
        return code;
    }

    // Fix #9: validate the output path doesn't point to a symlink, preventing
    // an attacker from tricking extraction into overwriting arbitrary files.
    if let Some(parent) = out.parent() {
        if parent.exists() {
            match parent.symlink_metadata() {
                Ok(meta) if meta.file_type().is_symlink() => {
                    set_last_error("Output parent directory is a symlink".into());
                    return AETHER_ERR_PATH_TRAVERSAL;
                }
                _ => {}
            }
        }
    }

    let d = unsafe { &mut *decompressor };
    let mut archive_file = match std::fs::File::open(archive) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(format!("Failed to open archive: {e}"));
            return AETHER_ERR_IO;
        }
    };

    // Create parent directories for output.
    // L2 security fix: call create_dir_all() unconditionally (it's idempotent)
    // to avoid TOCTOU race between exists() check and creation.
    if let Some(parent) = out.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            set_last_error(format!("Failed to create output directory: {e}"));
            return AETHER_ERR_IO;
        }
    }

    let mut out_file = match std::fs::File::create(out) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(format!("Failed to create output file: {e}"));
            return AETHER_ERR_IO;
        }
    };

    match d
        .inner
        .extract_file(&mut archive_file, &file_name, &mut out_file)
    {
        Ok(()) => AETHER_OK,
        Err(e) => {
            set_last_error(format!("File extraction failed: {e}"));
            AETHER_ERR_ARCHIVE
        }
    }
}

// ── File listing ────────────────────────────────────────────────────────────

/// File information returned by `aether_list`.
#[repr(C)]
pub struct AetherFileInfo {
    /// File path (C string, owned by this struct).
    pub path: *mut c_char,
    /// Original uncompressed size in bytes.
    pub original_size: u64,
    /// Unix permissions (e.g. 0o644).
    pub permissions: u32,
    /// Last modification time (Unix timestamp).
    pub mtime: i64,
}

/// Opaque file list handle returned by [`aether_list`].
///
/// # H3 security fix
///
/// Embeds the entry count so [`aether_file_list_free`] does not depend on
/// the caller passing the correct count, eliminating heap-corruption risk.
///
/// # Fix #12: safety warning
///
/// **Do not modify `count` or `entries` after receiving this struct.**
/// Callers should use [`aether_file_list_count`] and
/// [`aether_file_list_get`] to access entries safely.
/// Mutating `count` before calling [`aether_file_list_free`] causes
/// undefined behavior (heap corruption).
#[repr(C)]
pub struct AetherFileList {
    /// Pointer to the array of [`AetherFileInfo`] entries.
    /// **Read-only** — do not modify.
    pub entries: *mut AetherFileInfo,
    /// Number of entries in the array.
    /// **Read-only** — do not modify.
    pub count: u32,
}

/// List all files in an archive.
///
/// On success, `*list_out` is populated with a heap-allocated
/// [`AetherFileList`] containing the file entries and count.
///
/// The caller must free the result with [`aether_file_list_free`].
///
/// Returns `AETHER_OK` (0) on success, negative error code on failure.
#[no_mangle]
#[must_use]
pub extern "C" fn aether_list(
    decompressor: *mut AetherDecompressor,
    archive_path: *const c_char,
    list_out: *mut *mut AetherFileList,
) -> i32 {
    if decompressor.is_null() || list_out.is_null() {
        set_last_error("Null pointer argument".into());
        return AETHER_ERR_NULL_PTR;
    }

    let archive = match unsafe { cstr_to_pathbuf(archive_path) } {
        Ok(p) => p,
        Err(code) => return code,
    };

    // Fix #16: use &mut consistently with the *mut pointer type to prevent
    // aliasing UB if a C caller concurrently calls extract (which also takes &mut).
    let d = unsafe { &mut *decompressor };
    let mut file = match std::fs::File::open(archive) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(format!("Failed to open archive: {e}"));
            return AETHER_ERR_IO;
        }
    };

    let entries = match d.inner.list_files(&mut file) {
        Ok(e) => e,
        Err(e) => {
            set_last_error(format!("List failed: {e}"));
            return AETHER_ERR_ARCHIVE;
        }
    };

    let count = entries.len();
    // L1 fix: check that count fits in u32 before truncating.
    if count > u32::MAX as usize {
        set_last_error(format!(
            "Archive contains {count} entries, exceeds u32::MAX"
        ));
        return AETHER_ERR_ARCHIVE;
    }

    let mut infos: Vec<AetherFileInfo> = Vec::with_capacity(count);
    for fe in entries {
        let path_cstr = match CString::new(fe.path.as_str()) {
            Ok(c) => c,
            Err(_) => {
                // Clean up already-allocated CStrings before returning
                for info in &mut infos {
                    if !info.path.is_null() {
                        unsafe {
                            drop(CString::from_raw(info.path));
                        }
                        info.path = ptr::null_mut();
                    }
                }
                set_last_error(format!("File path contains null byte: {}", fe.path));
                return AETHER_ERR_ARCHIVE;
            }
        };
        infos.push(AetherFileInfo {
            path: path_cstr.into_raw(),
            original_size: fe.original_size,
            permissions: fe.permissions,
            mtime: fe.mtime,
        });
    }

    // S3 security fix: convert to boxed slice so capacity == length.
    // This ensures aether_file_list_free can safely reconstruct and drop
    // the allocation without capacity mismatch (Vec may over-allocate).
    let mut boxed = infos.into_boxed_slice();
    let entries_ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed); // Ownership transferred to AetherFileList

    // H3 security fix: embed count in the returned struct so the free
    // function doesn't depend on the caller passing the correct value.
    let file_list = Box::new(AetherFileList {
        entries: entries_ptr,
        count: count as u32,
    });

    unsafe {
        *list_out = Box::into_raw(file_list);
    }

    AETHER_OK
}

/// Fix #12: safe accessor — return the number of entries in a file list.
///
/// Returns 0 if `list` is `NULL`.
#[no_mangle]
pub extern "C" fn aether_file_list_count(list: *const AetherFileList) -> u32 {
    if list.is_null() {
        return 0;
    }
    unsafe { (*list).count }
}

/// Fix #12: safe accessor — return a pointer to the `index`-th file info entry.
///
/// Returns `NULL` if `list` is `NULL` or `index` is out of bounds.
/// The returned pointer is valid until [`aether_file_list_free`] is called.
#[no_mangle]
pub extern "C" fn aether_file_list_get(
    list: *const AetherFileList,
    index: u32,
) -> *const AetherFileInfo {
    if list.is_null() {
        return ptr::null();
    }
    let file_list = unsafe { &*list };
    if index >= file_list.count || file_list.entries.is_null() {
        return ptr::null();
    }
    unsafe { file_list.entries.add(index as usize) }
}

/// Free a file list returned by [`aether_list`] and null the caller's pointer.
///
/// No-op if `*list` is `NULL` or `list` is `NULL`.
///
/// # H3 security fix
///
/// The count is now embedded in the [`AetherFileList`] struct, so the caller
/// cannot pass an incorrect count. This eliminates the heap-corruption risk
/// from the previous API where count was a separate parameter.
///
/// # Fix #8
///
/// Now accepts `*mut *mut` to null the caller's pointer after free,
/// consistent with the H2 pattern used by compressor/decompressor free.
///
/// # Fix #7
///
/// Eliminates aliasing UB: uses a single `from_raw_parts_mut` to free both
/// the path strings and the entries array without creating overlapping mutable
/// references.
#[no_mangle]
pub extern "C" fn aether_file_list_free(list: *mut *mut AetherFileList) {
    if list.is_null() {
        return;
    }
    let list_ptr = unsafe { *list };
    if list_ptr.is_null() {
        return;
    }
    unsafe {
        // Null the caller's pointer first to prevent use-after-free.
        *list = ptr::null_mut();

        let file_list = Box::from_raw(list_ptr);
        if !file_list.entries.is_null() {
            let count = file_list.count as usize;
            let entries_ptr = file_list.entries;
            // Drop the AetherFileList box first (it doesn't own the entries
            // allocation), then work with the entries array exclusively.
            drop(file_list);

            // Reconstruct the boxed slice. This is the sole owner of the
            // allocation, avoiding aliasing UB from the previous approach.
            let mut entries_box =
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(entries_ptr, count));
            // Free each owned path string.
            for info in entries_box.iter_mut() {
                if !info.path.is_null() {
                    drop(CString::from_raw(info.path));
                    info.path = ptr::null_mut();
                }
            }
            // entries_box drops here, freeing the array allocation.
        }
    }
}

// ── Verification ────────────────────────────────────────────────────────────

/// Result of archive verification.
#[repr(C)]
pub struct AetherVerifyResult {
    /// Total number of blocks in the archive.
    pub total_blocks: u32,
    /// Number of blocks that verified successfully.
    pub verified_blocks: u32,
    /// Number of corrupted blocks detected.
    pub corrupted_count: u32,
    /// 1 if all blocks verified, 0 if any corruption detected.
    pub is_ok: i32,
}

/// Verify the integrity of an archive.
///
/// On success, `*result_out` is populated with verification details.
///
/// Returns `AETHER_OK` (0) on success (even if corruption is detected — check
/// `result_out->is_ok`). Returns a negative error code only on I/O or parsing errors.
#[no_mangle]
#[must_use]
pub extern "C" fn aether_verify(
    decompressor: *mut AetherDecompressor,
    archive_path: *const c_char,
    result_out: *mut AetherVerifyResult,
) -> i32 {
    if decompressor.is_null() || result_out.is_null() {
        set_last_error("Null pointer argument".into());
        return AETHER_ERR_NULL_PTR;
    }

    let archive = match unsafe { cstr_to_pathbuf(archive_path) } {
        Ok(p) => p,
        Err(code) => return code,
    };

    // Fix #16: use &mut consistently to prevent aliasing UB.
    let d = unsafe { &mut *decompressor };
    let mut file = match std::fs::File::open(archive) {
        Ok(f) => f,
        Err(e) => {
            set_last_error(format!("Failed to open archive: {e}"));
            return AETHER_ERR_IO;
        }
    };

    match d.inner.verify(&mut file) {
        Ok(result) => {
            // M2 fix: use saturating conversions instead of truncating casts.
            let total = u32::try_from(result.total_blocks).unwrap_or(u32::MAX);
            let verified = u32::try_from(result.verified_blocks).unwrap_or(u32::MAX);
            let corrupted = u32::try_from(result.corrupted_blocks.len()).unwrap_or(u32::MAX);
            unsafe {
                (*result_out) = AetherVerifyResult {
                    total_blocks: total,
                    verified_blocks: verified,
                    corrupted_count: corrupted,
                    is_ok: if result.is_ok() { 1 } else { 0 },
                };
            }
            AETHER_OK
        }
        Err(e) => {
            set_last_error(format!("Verification failed: {e}"));
            AETHER_ERR_ARCHIVE
        }
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn detect_predictor(path: &Path) -> Result<PredictorId, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("Cannot open archive: {e}"))?;
    let header = ArchiveHeader::read_from(&mut f)
        .map_err(|e| format!("Failed to read archive header: {e}"))?;
    Ok(header.predictor_id)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn version_is_not_null() {
        let v = aether_version();
        assert!(!v.is_null());
        let s = unsafe { CStr::from_ptr(v) };
        assert!(s.to_str().unwrap().contains('.'));
    }

    #[test]
    fn last_error_initially_null() {
        let e = aether_last_error();
        assert!(e.is_null());
    }

    #[test]
    fn last_error_owned_survives_next_call() {
        // H1 regression test: the returned error string must survive
        // subsequent FFI calls that overwrite LAST_ERROR.
        set_last_error("first error".into());
        let err1 = aether_last_error();
        assert!(!err1.is_null());

        // Overwrite LAST_ERROR
        set_last_error("second error".into());

        // err1 must still be valid (owned copy, not a borrowed pointer)
        let s = unsafe { CStr::from_ptr(err1) };
        assert_eq!(s.to_str().unwrap(), "first error");
        aether_error_free(err1);

        // Clean up second error
        let err2 = aether_last_error();
        aether_error_free(err2);
    }

    #[test]
    fn error_free_null_is_safe() {
        aether_error_free(ptr::null_mut());
    }

    #[test]
    fn compressor_lifecycle() {
        let c = aether_compressor_new(AetherPredictorId::NeuralSsm as u16);
        assert!(!c.is_null());
        let mut c = c;
        aether_compressor_free(&mut c);
        assert!(c.is_null()); // H2: pointer nulled after free
    }

    #[test]
    fn decompressor_lifecycle() {
        let d = aether_decompressor_new(AetherPredictorId::Order0 as u16);
        assert!(!d.is_null());
        let mut d = d;
        aether_decompressor_free(&mut d);
        assert!(d.is_null()); // H2: pointer nulled after free
    }

    #[test]
    fn double_free_is_safe() {
        // H2 regression test: calling free twice must not crash.
        let c = aether_compressor_new(AetherPredictorId::NeuralSsm as u16);
        assert!(!c.is_null());
        let mut c = c;
        aether_compressor_free(&mut c);
        assert!(c.is_null());
        aether_compressor_free(&mut c); // second free is a no-op
    }

    #[test]
    fn null_compressor_free_is_safe() {
        let mut c: *mut AetherCompressor = ptr::null_mut();
        aether_compressor_free(&mut c);
    }

    #[test]
    fn null_decompressor_free_is_safe() {
        let mut d: *mut AetherDecompressor = ptr::null_mut();
        aether_decompressor_free(&mut d);
    }

    #[test]
    fn null_ptr_to_free_is_safe() {
        // Passing a null outer pointer
        aether_compressor_free(ptr::null_mut());
        aether_decompressor_free(ptr::null_mut());
    }

    #[test]
    fn invalid_predictor_id_returns_null() {
        // M1 regression test: out-of-range predictor IDs must not cause UB.
        let c = aether_compressor_new(u16::MAX);
        assert!(c.is_null());
        let err = aether_last_error();
        assert!(!err.is_null());
        let s = unsafe { CStr::from_ptr(err) };
        assert!(s.to_str().unwrap().contains("Invalid predictor ID"));
        aether_error_free(err);

        let d = aether_decompressor_new(9999);
        assert!(d.is_null());
        let err = aether_last_error();
        aether_error_free(err);
    }

    #[test]
    fn null_args_return_error() {
        let code = aether_compress(ptr::null_mut(), ptr::null(), ptr::null(), 0, ptr::null());
        assert!(code < 0);
    }

    #[test]
    fn extract_null_decompressor() {
        let code = aether_extract_all(ptr::null_mut(), ptr::null(), ptr::null());
        assert!(code < 0);
    }

    #[test]
    fn verify_null_decompressor() {
        let mut result = AetherVerifyResult {
            total_blocks: 0,
            verified_blocks: 0,
            corrupted_count: 0,
            is_ok: 0,
        };
        let code = aether_verify(ptr::null_mut(), ptr::null(), &mut result);
        assert!(code < 0);
    }

    #[test]
    fn path_traversal_rejected() {
        // M4 regression test: paths with .. must be rejected.
        let base = Path::new("/tmp/output");
        assert!(validate_no_path_traversal(base, Path::new("normal/file.txt")).is_ok());
        assert!(validate_no_path_traversal(base, Path::new("../../etc/passwd")).is_err());
        assert!(validate_no_path_traversal(base, Path::new("a/../../../etc/shadow")).is_err());
        assert!(validate_no_path_traversal(base, Path::new("a/b/../c")).is_ok());
        // stays within
    }

    #[test]
    fn absolute_path_in_archive_rejected() {
        // M4 regression test: absolute paths must be rejected.
        let base = Path::new("/tmp/output");
        assert!(validate_no_path_traversal(base, Path::new("/etc/passwd")).is_err());
    }

    #[test]
    fn windows_drive_relative_path_rejected() {
        // Fix #5 regression test: Windows drive-relative paths must be rejected.
        let base = Path::new("/tmp/output");
        assert!(validate_no_path_traversal(base, Path::new("C:file.txt")).is_err());
        assert!(validate_no_path_traversal(base, Path::new("D:secret.txt")).is_err());
    }

    #[test]
    fn empty_path_rejected() {
        let base = Path::new("/tmp/output");
        assert!(validate_no_path_traversal(base, Path::new("")).is_err());
    }

    #[test]
    fn file_list_free_null_is_safe() {
        // Fix #8: null pointers should be no-ops
        aether_file_list_free(ptr::null_mut());
        let mut null_list: *mut AetherFileList = ptr::null_mut();
        aether_file_list_free(&mut null_list);
    }

    #[test]
    fn set_max_threads_returns_error_on_null() {
        // Fix #14: should return error code instead of silently ignoring
        let code = aether_compressor_set_max_threads(ptr::null_mut(), 4);
        assert_eq!(code, AETHER_ERR_NULL_PTR);
    }

    #[test]
    fn error_sanitization() {
        // Fix #10: error messages should not leak absolute paths
        set_last_error("Failed to open /home/user/secret/file.txt".into());
        let err = aether_last_error();
        assert!(!err.is_null());
        let s = unsafe { CStr::from_ptr(err) }.to_str().unwrap();
        assert!(!s.contains("/home/user"), "Error message leaked path: {s}");
        assert!(
            s.contains("<path>"),
            "Error should contain <path> placeholder: {s}"
        );
        aether_error_free(err);
    }

    #[test]
    fn error_sanitization_windows_paths() {
        // Fix #16: Windows absolute paths must be redacted
        let sanitized = sanitize_error_message(r"Failed to open C:\Users\admin\secret.txt");
        assert!(
            !sanitized.contains("admin"),
            "Windows path leaked: {sanitized}"
        );
        assert!(
            sanitized.contains("<path>"),
            "Should contain <path>: {sanitized}"
        );
    }

    #[test]
    fn error_sanitization_relative_traversal() {
        // Fix #16: relative traversal paths must be redacted
        let sanitized = sanitize_error_message("Bad entry: ../../etc/passwd in archive");
        assert!(
            !sanitized.contains("etc/passwd"),
            "Relative path leaked: {sanitized}"
        );
        assert!(
            sanitized.contains("<path>"),
            "Should contain <path>: {sanitized}"
        );
    }

    #[test]
    fn error_sanitization_unc_paths() {
        // Fix #16: UNC paths must be redacted
        let sanitized = sanitize_error_message(r"Access denied: \\server\share\secret.docx");
        assert!(
            !sanitized.contains("server"),
            "UNC path leaked: {sanitized}"
        );
        assert!(
            sanitized.contains("<path>"),
            "Should contain <path>: {sanitized}"
        );
    }

    #[test]
    fn compress_exceeds_max_files() {
        // Fix #16: file_count exceeding MAX_FFI_FILES must return an error
        let c = aether_compressor_new(AetherPredictorId::NeuralSsm as u16);
        assert!(!c.is_null());
        let base = CString::new("/tmp").unwrap();
        let dummy = CString::new("/tmp/a.txt").unwrap();
        let files = [dummy.as_ptr()];
        let out = CString::new("/tmp/out.aet").unwrap();
        // Pass a count exceeding the limit (we only have 1 pointer, but the
        // limit check happens before dereferencing entries beyond index 0)
        let code = aether_compress(c, base.as_ptr(), files.as_ptr(), 1_000_001, out.as_ptr());
        assert_eq!(code, AETHER_ERR_COMPRESSION);
        let err = aether_last_error();
        assert!(!err.is_null());
        let s = unsafe { CStr::from_ptr(err) }.to_str().unwrap();
        assert!(
            s.contains("exceeds maximum"),
            "Error should mention limit: {s}"
        );
        aether_error_free(err);
        let mut c = c;
        aether_compressor_free(&mut c);
    }

    #[test]
    fn path_traversal_error_does_not_leak_entry_name() {
        // Fix #16: error messages from path traversal should not contain
        // the raw malicious path
        let base = Path::new("/tmp/output");
        let malicious = Path::new("../../etc/shadow");
        let result = validate_no_path_traversal(base, malicious);
        assert!(result.is_err());
        let err = aether_last_error();
        assert!(!err.is_null());
        let s = unsafe { CStr::from_ptr(err) }.to_str().unwrap();
        assert!(!s.contains("etc/shadow"), "Error leaked entry path: {s}");
        aether_error_free(err);
    }

    #[test]
    fn file_list_accessors() {
        // Fix #12: test accessor functions with null
        assert_eq!(aether_file_list_count(ptr::null()), 0);
        assert!(aether_file_list_get(ptr::null(), 0).is_null());
    }

    #[test]
    fn roundtrip_via_ffi() {
        use std::io::Write;

        // Create temp directories
        let tmp = tempfile::tempdir().unwrap();
        let input_dir = tmp.path().join("input");
        let output_dir = tmp.path().join("output");
        let archive = tmp.path().join("test.aet");
        std::fs::create_dir_all(&input_dir).unwrap();

        // Create test files
        let mut f = std::fs::File::create(input_dir.join("hello.txt")).unwrap();
        f.write_all(b"Hello, AetherArch FFI!").unwrap();
        drop(f);

        let mut f = std::fs::File::create(input_dir.join("data.bin")).unwrap();
        f.write_all(&[0u8; 256]).unwrap();
        drop(f);

        // Compress
        let mut c = aether_compressor_new(AetherPredictorId::NeuralSsm as u16);
        assert!(!c.is_null());

        let base = CString::new(input_dir.to_str().unwrap()).unwrap();
        let file1 = CString::new(input_dir.join("hello.txt").to_str().unwrap()).unwrap();
        let file2 = CString::new(input_dir.join("data.bin").to_str().unwrap()).unwrap();
        let files = [file1.as_ptr(), file2.as_ptr()];
        let out = CString::new(archive.to_str().unwrap()).unwrap();

        let code = aether_compress(c, base.as_ptr(), files.as_ptr(), 2, out.as_ptr());
        if code != AETHER_OK {
            let err = aether_last_error();
            let msg = if err.is_null() {
                "unknown error".to_string()
            } else {
                let s = unsafe { CStr::from_ptr(err) }.to_string_lossy().to_string();
                aether_error_free(err);
                s
            };
            panic!("Compress failed with code {code}: {msg}");
        }
        aether_compressor_free(&mut c);
        assert!(c.is_null());

        // Auto-detect predictor
        let mut d = aether_decompressor_auto(out.as_ptr());
        assert!(!d.is_null());

        // List files
        let mut list: *mut AetherFileList = ptr::null_mut();
        let code = aether_list(d, out.as_ptr(), &mut list);
        assert_eq!(code, AETHER_OK, "List failed");
        assert!(!list.is_null());

        let file_list = unsafe { &*list };
        assert_eq!(file_list.count, 2);

        // Check file names
        let infos =
            unsafe { std::slice::from_raw_parts(file_list.entries, file_list.count as usize) };
        let names: Vec<String> = infos
            .iter()
            .map(|i| {
                unsafe { CStr::from_ptr(i.path) }
                    .to_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert!(names.contains(&"hello.txt".to_string()));
        assert!(names.contains(&"data.bin".to_string()));

        aether_file_list_free(&mut list);
        assert!(list.is_null()); // Fix #8: pointer nulled after free

        // Verify
        let mut result = AetherVerifyResult {
            total_blocks: 0,
            verified_blocks: 0,
            corrupted_count: 0,
            is_ok: 0,
        };
        let code = aether_verify(d, out.as_ptr(), &mut result);
        assert_eq!(code, AETHER_OK, "Verify failed");
        assert_eq!(result.is_ok, 1, "Archive corrupted");

        // Extract
        let outdir = CString::new(output_dir.to_str().unwrap()).unwrap();
        let code = aether_extract_all(d, out.as_ptr(), outdir.as_ptr());
        if code != AETHER_OK {
            let err = aether_last_error();
            let msg = if err.is_null() {
                "unknown error".to_string()
            } else {
                let s = unsafe { CStr::from_ptr(err) }.to_string_lossy().to_string();
                aether_error_free(err);
                s
            };
            panic!("Extract failed with code {code}: {msg}");
        }

        aether_decompressor_free(&mut d);
        assert!(d.is_null());

        // Verify extracted content
        let hello = std::fs::read(output_dir.join("hello.txt")).unwrap();
        assert_eq!(hello, b"Hello, AetherArch FFI!");
        let data = std::fs::read(output_dir.join("data.bin")).unwrap();
        assert_eq!(data.len(), 256);
        assert!(data.iter().all(|&b| b == 0));
    }
}
