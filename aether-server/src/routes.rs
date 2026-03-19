use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Multipart, State};
use axum::http::header;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use serde_json::json;

use aether_core::entropy::{NeuralSsmPredictor, Order0Model, ProbabilityPredictor};
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

use crate::error::ApiError;
use crate::AppState;

/// Validate the API key from State. Returns Ok(()) if no key is configured
/// or if the provided key matches.
///
/// V1 fix: Use HMAC-based comparison to prevent both timing and length
/// side-channel attacks. Both sides are hashed before comparison, so the
/// attacker cannot learn the key length from response timing.
pub fn check_auth(headers: &HeaderMap, state: &AppState) -> Result<(), ApiError> {
    if let Some(ref expected) = state.api_key {
        let provided = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        // Hash both keys with a fixed context so the comparison is always
        // over equal-length digests, leaking neither length nor content.
        let expected_hash = blake3::hash(expected.as_bytes());
        let provided_hash = blake3::hash(provided.as_bytes());
        use subtle::ConstantTimeEq;
        if !bool::from(expected_hash.as_bytes().ct_eq(provided_hash.as_bytes())) {
            return Err(ApiError::Unauthorized("Invalid or missing API key".into()));
        }
    }
    Ok(())
}

/// Maximum time allowed for a single compression or decompression operation.
/// Reduced from 5 min to 120s to limit resource exhaustion from slow requests.
const BLOCKING_TIMEOUT: Duration = Duration::from_secs(120);

/// Maximum size of a single uploaded field (256 MiB).
const MAX_FIELD_SIZE: usize = 256 * 1024 * 1024;

/// V2/V3 fix: Server-specific total decompression limit (512 MiB).
/// Much tighter than the library's 8 GiB limit to prevent disk exhaustion
/// from decompression bombs via the REST API.
const MAX_SERVER_DECOMPRESSED_SIZE: u64 = 512 * 1024 * 1024;

/// V2/V3 fix: Maximum number of files the server will extract.
/// Prevents inode exhaustion from archives with millions of tiny files.
const MAX_SERVER_FILE_COUNT: u32 = 10_000;

/// Windows reserved device names that must not be used as filenames.
const WINDOWS_RESERVED: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM0", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
    "COM8", "COM9", "LPT0", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// POST /compress — multipart upload → .aet archive download
pub async fn compress(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    check_auth(&headers, &state)?;
    let tmp_input = tempfile::tempdir().map_err(|e| ApiError::Internal(e.to_string()))?;
    let mut file_count = 0u32;
    let mut used_names = HashSet::new();
    let mut name_counters: HashMap<String, u32> = HashMap::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        // Enforce file count limit on upload to prevent inode exhaustion.
        if file_count >= MAX_SERVER_FILE_COUNT {
            return Err(ApiError::BadRequest(format!(
                "Upload exceeds maximum file count of {}",
                MAX_SERVER_FILE_COUNT,
            )));
        }

        let filename = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("file_{file_count}"));

        let data = match field.bytes().await {
            Ok(b) => {
                if b.len() > MAX_FIELD_SIZE {
                    return Err(ApiError::BadRequest(format!(
                        "Field exceeds maximum size of {} bytes",
                        MAX_FIELD_SIZE,
                    )));
                }
                b
            }
            Err(e) => return Err(ApiError::BadRequest(e.to_string())),
        };

        // Security fix: sanitize filenames to prevent path traversal.
        let mut safe_filename = sanitize_filename(&filename, file_count);

        // Deduplicate: if the sanitized name collides, use a monotonic
        // counter per base name for O(1) dedup instead of linear probing.
        if !used_names.insert(safe_filename.clone()) {
            let counter = name_counters.entry(safe_filename.clone()).or_insert(1);
            let base = safe_filename.clone();
            loop {
                safe_filename = format!("{counter}_{base}");
                *counter = counter.checked_add(1).ok_or_else(|| {
                    ApiError::BadRequest("Filename deduplication overflow".into())
                })?;
                if used_names.insert(safe_filename.clone()) {
                    break;
                }
            }
        }

        // Sanitized filenames never contain path separators, so join is flat.
        let path = tmp_input.path().join(&safe_filename);
        std::fs::write(&path, &data)?;
        file_count += 1;
    }

    if file_count == 0 {
        return Err(ApiError::BadRequest("No files uploaded".into()));
    }

    let files: Vec<_> = std::fs::read_dir(tmp_input.path())?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .map(|e| e.path())
        .collect();

    let base_dir = tmp_input.path().to_path_buf();
    let archive = tokio::time::timeout(
        BLOCKING_TIMEOUT,
        tokio::task::spawn_blocking(move || -> Result<Vec<u8>, ApiError> {
            let compressor = Compressor::new(|| Box::new(NeuralSsmPredictor::new()));
            let mut cursor = Cursor::new(Vec::new());
            compressor
                .compress_to_archive(&base_dir, &files, &mut cursor)
                .map_err(ApiError::from)?;
            Ok(cursor.into_inner())
        }),
    )
    .await
    .map_err(|_| ApiError::Internal("Compression timed out".into()))?
    .map_err(|e| ApiError::Internal(e.to_string()))??;

    Ok((
        [
            (header::CONTENT_TYPE, "application/octet-stream"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"archive.aet\"",
            ),
        ],
        archive,
    ))
}

/// V4 fix: Read and validate a single archive field from multipart upload.
/// Enforces MAX_FIELD_SIZE on all endpoints, not just /compress.
async fn read_archive_field(multipart: &mut Multipart) -> Result<Vec<u8>, ApiError> {
    let field = match multipart.next_field().await {
        Ok(Some(f)) => f,
        Ok(None) => return Err(ApiError::BadRequest("No archive uploaded".into())),
        Err(e) => {
            tracing::debug!("Multipart read error: {e}");
            return Err(ApiError::BadRequest("Invalid upload".into()));
        }
    };
    let data = field
        .bytes()
        .await
        .map_err(|e| ApiError::BadRequest(e.to_string()))?;
    if data.len() > MAX_FIELD_SIZE {
        return Err(ApiError::BadRequest(format!(
            "Upload exceeds maximum size of {} bytes",
            MAX_FIELD_SIZE,
        )));
    }
    Ok(data.to_vec())
}

/// POST /extract — .aet upload → JSON file listing with sizes
pub async fn extract(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    check_auth(&headers, &state)?;
    let data = read_archive_field(&mut multipart).await?;

    let result = tokio::time::timeout(
        BLOCKING_TIMEOUT,
        tokio::task::spawn_blocking(move || -> Result<serde_json::Value, ApiError> {
            let metadata =
                Decompressor::read_metadata_streaming(&mut (&data[..])).map_err(ApiError::from)?;

            // V2/V3 fix: reject archives with too many files for the server
            if metadata.file_entries.len() > MAX_SERVER_FILE_COUNT as usize {
                return Err(ApiError::BadRequest(format!(
                    "Archive contains {} files, exceeding server limit of {}",
                    metadata.file_entries.len(),
                    MAX_SERVER_FILE_COUNT,
                )));
            }

            // V2/V3 fix: reject archives whose declared total size exceeds server limit
            let declared_total: u64 = metadata.file_entries.iter().map(|e| e.original_size).sum();
            if declared_total > MAX_SERVER_DECOMPRESSED_SIZE {
                return Err(ApiError::BadRequest(format!(
                    "Archive declared total size {} bytes exceeds server limit of {} bytes",
                    declared_total, MAX_SERVER_DECOMPRESSED_SIZE,
                )));
            }

            let factory = make_factory_from_id(metadata.header.predictor_id)?;
            let decompressor = Decompressor::new(factory);
            let tmp = tempfile::tempdir().map_err(|e| ApiError::Internal(e.to_string()))?;
            let mut cursor = Cursor::new(&data);
            decompressor
                .extract_all(&mut cursor, tmp.path())
                .map_err(ApiError::from)?;

            // Validate that extracted files stayed within the temp directory,
            // and enforce cumulative size + file count limits post-extraction.
            let mut files = Vec::new();
            collect_extracted(tmp.path(), tmp.path(), &mut files)?;

            let file_list: Vec<_> = files
                .iter()
                .map(|(name, data)| json!({ "name": name, "size": data.len() }))
                .collect();

            Ok(json!({ "files": file_list, "count": files.len() }))
        }),
    )
    .await
    .map_err(|_| ApiError::Internal("Extraction timed out".into()))?
    .map_err(|e| ApiError::Internal(e.to_string()))??;

    Ok(axum::Json(result))
}

/// POST /verify — .aet upload → JSON verification result
pub async fn verify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    check_auth(&headers, &state)?;
    let data = read_archive_field(&mut multipart).await?;

    let result = tokio::time::timeout(
        BLOCKING_TIMEOUT,
        tokio::task::spawn_blocking(move || -> Result<serde_json::Value, ApiError> {
            let metadata =
                Decompressor::read_metadata_streaming(&mut (&data[..])).map_err(ApiError::from)?;
            let factory = make_factory_from_id(metadata.header.predictor_id)?;

            let decompressor = Decompressor::new(factory);
            let mut cursor = Cursor::new(&data);
            let result = decompressor.verify(&mut cursor).map_err(ApiError::from)?;

            Ok(json!({
                "ok": result.is_ok(),
                "total_blocks": result.total_blocks,
                "verified_blocks": result.verified_blocks,
                "corrupted_blocks": result.corrupted_blocks,
            }))
        }),
    )
    .await
    .map_err(|_| ApiError::Internal("Verification timed out".into()))?
    .map_err(|e| ApiError::Internal(e.to_string()))??;

    Ok(axum::Json(result))
}

/// POST /list — list files in an uploaded archive
pub async fn list(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    check_auth(&headers, &state)?;
    let data = read_archive_field(&mut multipart).await?;

    let result = tokio::time::timeout(
        BLOCKING_TIMEOUT,
        tokio::task::spawn_blocking(move || -> Result<serde_json::Value, ApiError> {
            let entries =
                Decompressor::list_files_streaming(&mut (&data[..])).map_err(ApiError::from)?;

            let file_list: Vec<_> = entries
                .iter()
                .map(|e| json!({ "path": e.path, "size": e.original_size }))
                .collect();

            Ok(json!({ "files": file_list, "count": entries.len() }))
        }),
    )
    .await
    .map_err(|_| ApiError::Internal("List operation timed out".into()))?
    .map_err(|e| ApiError::Internal(e.to_string()))??;

    Ok(axum::Json(result))
}

/// GET /health
pub async fn health() -> impl IntoResponse {
    axum::Json(json!({ "status": "ok" }))
}

/// GET /version
pub async fn version(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, ApiError> {
    check_auth(&headers, &state)?;
    Ok(axum::Json(json!({
        "name": "aether-server",
        "version": env!("CARGO_PKG_VERSION"),
        "format_version": format!(
            "{}.{}",
            aether_core::format::FORMAT_VERSION_MAJOR,
            aether_core::format::FORMAT_VERSION_MINOR
        ),
    })))
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_factory_from_id(
    id: aether_core::format::PredictorId,
) -> Result<Box<dyn Fn() -> Box<dyn ProbabilityPredictor> + Send + Sync>, ApiError> {
    use aether_core::format::PredictorId;
    match id {
        PredictorId::Order0 => Ok(Box::new(|| Box::new(Order0Model::new()))),
        PredictorId::NeuralSsm | PredictorId::ZstdOnly => {
            Ok(Box::new(|| Box::new(NeuralSsmPredictor::new())))
        }
        PredictorId::Rle => Ok(Box::new(|| {
            Box::new(aether_core::entropy::RlePredictor::new())
        })),
        other => Err(ApiError::BadRequest(format!(
            "Archive uses unknown predictor ID {:?}",
            other
        ))),
    }
}

/// Sanitize an uploaded filename to prevent path traversal and filesystem attacks.
///
/// Uses an allowlist approach: only ASCII alphanumeric, '.', '-', '_' are kept.
/// Also blocks Windows reserved device names and strips trailing dots/spaces.
pub(crate) fn sanitize_filename(filename: &str, fallback_index: u32) -> String {
    // Allowlist: keep only safe ASCII characters.
    let safe: String = filename
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();

    // Replace consecutive dots to prevent ".." traversal.
    let mut result = String::with_capacity(safe.len());
    let mut prev_dot = false;
    for c in safe.chars() {
        if c == '.' {
            if prev_dot {
                // Skip consecutive dots entirely (don't even insert underscore)
                continue;
            } else {
                result.push('.');
                prev_dot = true;
            }
        } else {
            prev_dot = false;
            result.push(c);
        }
    }

    // Strip leading dots/underscores (hidden files, leading separators).
    let result = result.trim_start_matches(['_', '.']).to_string();

    // Strip trailing dots, underscores, and spaces (Windows NTFS issue).
    let result = result.trim_end_matches(['.', ' ', '_']).to_string();

    // Block Windows reserved device names (CON, NUL, COM1, etc.).
    let stem = result.split('.').next().unwrap_or("").to_ascii_uppercase();
    if WINDOWS_RESERVED.contains(&stem.as_str()) {
        return format!("file_{fallback_index}");
    }

    if result.is_empty() {
        format!("file_{fallback_index}")
    } else {
        result
    }
}

/// Iterative directory traversal to collect extracted files.
/// Uses an explicit stack instead of recursion to prevent stack overflow
/// from deeply nested archive directories.
///
/// Enforces cumulative size and file count limits to catch decompression bombs
/// that lie about their declared sizes in metadata.
fn collect_extracted(
    root: &std::path::Path,
    dir: &std::path::Path,
    files: &mut Vec<(String, Vec<u8>)>,
) -> Result<(), ApiError> {
    let mut stack = vec![dir.to_path_buf()];
    let mut total_bytes: u64 = 0;
    let mut file_count: u32 = 0;

    while let Some(current_dir) = stack.pop() {
        for entry in std::fs::read_dir(&current_dir)? {
            let entry = entry?;
            let path = entry.path();

            // H2/H4 security fix: use symlink_metadata() to detect symlinks without
            // following them. Reject symlinks to prevent arbitrary file read attacks
            // via malicious archives containing symlinks to sensitive server files.
            let meta = std::fs::symlink_metadata(&path)
                .map_err(|e| ApiError::Internal(format!("Cannot stat extracted file: {e}")))?;
            if meta.file_type().is_symlink() {
                return Err(ApiError::BadRequest(
                    "Archive contains a symlink, which is not permitted".into(),
                ));
            }

            if meta.is_dir() {
                stack.push(path);
            } else {
                // Enforce file count limit post-extraction.
                file_count = file_count
                    .checked_add(1)
                    .ok_or_else(|| ApiError::BadRequest("Extracted file count overflow".into()))?;
                if file_count > MAX_SERVER_FILE_COUNT {
                    return Err(ApiError::BadRequest(
                        "Extracted file count exceeds server limit".into(),
                    ));
                }

                let rel = path
                    .strip_prefix(root)
                    .map_err(|_| {
                        ApiError::BadRequest("Extracted path escapes output directory".into())
                    })?
                    .to_string_lossy()
                    .replace('\\', "/");
                let data = std::fs::read(&path)?;

                // Enforce cumulative decompressed size limit.
                total_bytes += u64::try_from(data.len()).unwrap_or(u64::MAX);
                if total_bytes > MAX_SERVER_DECOMPRESSED_SIZE {
                    return Err(ApiError::BadRequest(
                        "Extracted data exceeds server decompression size limit".into(),
                    ));
                }

                files.push((rel, data));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{header, HeaderMap, HeaderValue};

    // ── Filename sanitization tests ─────────────────────────────────────

    #[test]
    fn sanitize_plain_filename() {
        assert_eq!(sanitize_filename("hello.txt", 0), "hello.txt");
    }

    #[test]
    fn sanitize_path_traversal_unix() {
        let result = sanitize_filename("../../../etc/passwd", 0);
        assert!(!result.contains('/'), "sanitized name must not contain /");
        assert!(!result.contains(".."), "sanitized name must not contain ..");
        assert!(
            result.contains("etc"),
            "original basename components preserved"
        );
        assert!(result.contains("passwd"));
    }

    #[test]
    fn sanitize_path_traversal_windows() {
        let result = sanitize_filename("..\\..\\Windows\\System32\\config", 0);
        assert!(
            !result.contains('\\'),
            "sanitized name must not contain backslash"
        );
        assert!(!result.contains(".."), "sanitized name must not contain ..");
    }

    #[test]
    fn sanitize_forward_slash_replaced() {
        let result = sanitize_filename("path/to/file.txt", 0);
        assert!(!result.contains('/'));
        assert!(result.contains("file.txt"));
    }

    #[test]
    fn sanitize_backslash_replaced() {
        let result = sanitize_filename("path\\to\\file.txt", 0);
        assert!(!result.contains('\\'));
        assert!(result.contains("file.txt"));
    }

    #[test]
    fn sanitize_dot_dot_replaced() {
        let result = sanitize_filename("..file.txt", 0);
        assert!(!result.contains(".."));
    }

    #[test]
    fn sanitize_empty_becomes_fallback() {
        let result = sanitize_filename("///", 5);
        assert_eq!(result, "file_5");
    }

    #[test]
    fn sanitize_only_dots_and_slashes() {
        let result = sanitize_filename("../../..", 3);
        assert_eq!(result, "file_3");
    }

    #[test]
    fn sanitize_preserves_extensions() {
        assert_eq!(sanitize_filename("report.pdf", 0), "report.pdf");
        assert_eq!(sanitize_filename("data.tar.gz", 0), "data.tar.gz");
    }

    #[test]
    fn sanitize_mixed_traversal() {
        let result = sanitize_filename("../foo/..\\bar/baz.txt", 0);
        assert!(!result.contains('/'));
        assert!(!result.contains('\\'));
        assert!(!result.contains(".."));
        assert!(result.contains("baz.txt"));
    }

    #[test]
    fn sanitize_nul_bytes_stripped() {
        let result = sanitize_filename("evil\0.txt", 0);
        assert!(!result.contains('\0'), "NUL bytes must be stripped");
        assert!(result.contains("evil"));
        assert!(result.contains(".txt"));
    }

    #[test]
    fn sanitize_control_chars_stripped() {
        let result = sanitize_filename("file\x01\x02\x1f.txt", 0);
        assert!(
            !result.chars().any(|c| c.is_control()),
            "control chars must be stripped"
        );
        assert!(result.contains("file"));
    }

    #[test]
    fn sanitize_windows_ads_colon_stripped() {
        let result = sanitize_filename("file.txt:Zone.Identifier", 0);
        assert!(!result.contains(':'), "colons must be stripped");
    }

    #[test]
    fn sanitize_nul_truncation_attack() {
        let result = sanitize_filename("safe.txt\0../../etc/passwd", 0);
        assert!(!result.contains('\0'));
        assert!(!result.contains('/'));
        assert!(!result.contains(".."));
    }

    #[test]
    fn sanitize_windows_reserved_con() {
        assert_eq!(sanitize_filename("CON", 0), "file_0");
        assert_eq!(sanitize_filename("con.txt", 0), "file_0");
        assert_eq!(sanitize_filename("Con.log", 0), "file_0");
    }

    #[test]
    fn sanitize_windows_reserved_nul() {
        assert_eq!(sanitize_filename("NUL", 1), "file_1");
        assert_eq!(sanitize_filename("nul.txt", 1), "file_1");
    }

    #[test]
    fn sanitize_windows_reserved_com1() {
        assert_eq!(sanitize_filename("COM1", 2), "file_2");
        assert_eq!(sanitize_filename("com1.txt", 2), "file_2");
    }

    #[test]
    fn sanitize_windows_reserved_lpt1() {
        assert_eq!(sanitize_filename("LPT1", 3), "file_3");
    }

    #[test]
    fn sanitize_trailing_dots_stripped() {
        assert_eq!(sanitize_filename("file.txt.", 0), "file.txt");
        assert_eq!(sanitize_filename("file...", 0), "file");
    }

    #[test]
    fn sanitize_non_ascii_replaced() {
        // Unicode chars replaced with underscores
        let result = sanitize_filename("café.txt", 0);
        assert!(result.is_ascii());
        assert!(result.contains(".txt"));
    }

    // ── check_auth tests ────────────────────────────────────────────────
    //
    // Tests now pass AppState directly instead of mutating env vars,
    // making them thread-safe.

    #[test]
    fn check_auth_no_api_key_set_accepts_any_request() {
        let state = AppState { api_key: None };
        let headers = HeaderMap::new();
        assert!(check_auth(&headers, &state).is_ok());
    }

    #[test]
    fn check_auth_with_empty_headers_no_panic() {
        let state = AppState { api_key: None };
        let headers = HeaderMap::new();
        let _ = check_auth(&headers, &state);
    }

    #[test]
    fn check_auth_with_bearer_header_no_panic() {
        let state = AppState { api_key: None };
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-key-12345"),
        );
        let _ = check_auth(&headers, &state);
    }

    #[test]
    fn check_auth_with_non_bearer_prefix_no_panic() {
        let state = AppState { api_key: None };
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        let _ = check_auth(&headers, &state);
    }

    #[test]
    fn check_auth_correct_key_accepts() {
        let state = AppState {
            api_key: Some("test-secret-key-42".into()),
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-secret-key-42"),
        );
        assert!(check_auth(&headers, &state).is_ok());
    }

    #[test]
    fn check_auth_wrong_key_rejects() {
        let state = AppState {
            api_key: Some("correct-key".into()),
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer wrong-key"),
        );
        assert!(check_auth(&headers, &state).is_err());
    }

    #[test]
    fn check_auth_missing_header_rejects_when_key_set() {
        let state = AppState {
            api_key: Some("some-key".into()),
        };
        let headers = HeaderMap::new();
        assert!(check_auth(&headers, &state).is_err());
    }

    #[test]
    fn check_auth_different_length_key_rejects() {
        let state = AppState {
            api_key: Some("short".into()),
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer a-much-longer-key-than-expected"),
        );
        assert!(check_auth(&headers, &state).is_err());
    }

    // ── collect_extracted tests ──────────────────────────────────────────

    #[test]
    fn collect_extracted_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mut files = Vec::new();
        collect_extracted(tmp.path(), tmp.path(), &mut files).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn collect_extracted_flat_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), b"hello").unwrap();
        std::fs::write(tmp.path().join("b.txt"), b"world").unwrap();
        let mut files = Vec::new();
        collect_extracted(tmp.path(), tmp.path(), &mut files).unwrap();
        assert_eq!(files.len(), 2);
        let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"a.txt"));
        assert!(names.contains(&"b.txt"));
    }

    #[test]
    fn collect_extracted_nested_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("nested.txt"), b"data").unwrap();
        let mut files = Vec::new();
        collect_extracted(tmp.path(), tmp.path(), &mut files).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "sub/nested.txt");
    }

    #[cfg(unix)]
    #[test]
    fn collect_extracted_rejects_symlinks() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("real.txt"), b"data").unwrap();
        symlink(tmp.path().join("real.txt"), tmp.path().join("link.txt")).unwrap();
        let mut files = Vec::new();
        let result = collect_extracted(tmp.path(), tmp.path(), &mut files);
        assert!(result.is_err(), "symlinks should be rejected");
    }

    // ── health / version response tests ─────────────────────────────────

    #[tokio::test]
    async fn health_returns_ok() {
        use axum::response::IntoResponse;
        use http_body_util::BodyExt;

        let resp = health().await.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::OK);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }
}
