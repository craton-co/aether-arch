#![allow(deprecated)]
//! Cloud storage backend adapters for reading/writing archives.
//!
//! Provides a `StorageBackend` trait and implementations for S3, GCS, and Azure Blob.
//! The `CloudReader` adapter implements `Read + Seek` using range requests, enabling
//! seekable decompression directly from cloud storage.
//!
//! # Security
//!
//! Implementations **must** verify TLS certificates on all connections.
//! Do not disable certificate verification, even for testing — use a local
//! mock server or the provided in-memory `MockBackend` in tests instead.
//!
//! All cloud paths are validated against traversal attacks, null bytes, and
//! header-injection characters before being passed to backends.
//!
//! # Status
//!
//! The S3, GCS, and Azure backends are currently **stubs** that return
//! errors for all operations.  Real SDK integration is tracked in the roadmap.
//!
//! All cloud functionality is behind the `cloud` feature flag.

pub mod azure;
pub mod gcs;
pub mod s3;

use crate::error::{AetherError, Result};

/// Maximum number of bytes to buffer in a single prefetch for `CloudReader`.
const DEFAULT_BUFFER_SIZE: usize = 256 * 1024; // 256 KiB

/// Maximum number of bytes to fetch in a single `read_range` call.
/// Prevents OOM when a malicious server reports an enormous `total_size`.
const MAX_FETCH_SIZE: u64 = 16 * 1024 * 1024; // 16 MiB

/// Validate a cloud storage path.
///
/// Rejects paths containing traversal sequences, null bytes, or characters
/// that could be used for HTTP header injection.
pub fn validate_cloud_path(path: &str) -> Result<()> {
    if path.is_empty() {
        return Err(AetherError::InvalidCloudPath(
            "path must not be empty".into(),
        ));
    }

    if path.contains('\0') {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain null bytes".into(),
        ));
    }

    // Block newlines/carriage returns that could enable HTTP header injection
    if path.contains('\n') || path.contains('\r') {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain newline characters".into(),
        ));
    }

    // Block backslashes — some backends or OS layers interpret these as
    // path separators, which could enable traversal (e.g. `..\..`).
    if path.contains('\\') {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain backslashes".into(),
        ));
    }

    // Block path traversal (literal)
    if path.contains("..") {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain '..' traversal sequences".into(),
        ));
    }

    // Block percent-encoded traversal and dangerous characters.
    // This catches cases where a downstream component might decode the path.
    let lower = path.to_ascii_lowercase();
    if lower.contains("%2e%2e") || lower.contains("%2e.") || lower.contains(".%2e") {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain encoded traversal sequences".into(),
        ));
    }
    // Double-encoded traversal: %252e decodes to %2e, which decodes to '.'
    if lower.contains("%252e") {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain double-encoded traversal sequences".into(),
        ));
    }
    // Triple-encoded and beyond: block any %25 sequence to prevent
    // arbitrary levels of recursive decoding.
    if lower.contains("%25") {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain multiply-encoded sequences".into(),
        ));
    }
    // %2f = '/', %5c = '\'  — encoded separators can bypass literal checks
    if lower.contains("%2f") || lower.contains("%5c") {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain encoded path separators".into(),
        ));
    }
    // %00 = null byte
    if lower.contains("%00") {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain encoded null bytes".into(),
        ));
    }
    // Overlong UTF-8 encodings: %c0%ae is an overlong encoding of '.'
    // that some parsers accept. Block known overlong sequences.
    if lower.contains("%c0%ae") || lower.contains("%c0%af") {
        return Err(AetherError::InvalidCloudPath(
            "path must not contain overlong UTF-8 encoded sequences".into(),
        ));
    }
    // Unicode fullwidth characters that could be normalized to ASCII
    // equivalents by some backends (e.g. U+FF0E fullwidth period,
    // U+FF0F fullwidth solidus, U+FF3C fullwidth reverse solidus).
    for ch in path.chars() {
        if ch == '\u{FF0E}' || ch == '\u{FF0F}' || ch == '\u{FF3C}' {
            return Err(AetherError::InvalidCloudPath(
                "path must not contain Unicode fullwidth characters that could be normalized to path separators or dots".into(),
            ));
        }
    }

    // Block absolute paths (they should be relative to bucket/container root)
    if path.starts_with('/') {
        return Err(AetherError::InvalidCloudPath(
            "path must not start with '/'".into(),
        ));
    }

    Ok(())
}

/// A validated cloud storage path.
///
/// This newtype can only be constructed through [`ValidatedPath::new`], which
/// calls [`validate_cloud_path`].  Passing `ValidatedPath` instead of raw
/// `&str` through the API ensures that validation cannot be accidentally
/// skipped.
///
/// # Warning
///
/// Do **not** derive `Deserialize` on this type — doing so would bypass
/// validation.  If serde support is needed, implement `TryFrom<String>` and
/// use `#[serde(try_from = "String")]` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedPath(String);

impl ValidatedPath {
    /// Create a new `ValidatedPath`, returning an error if validation fails.
    #[must_use = "validation result must be checked"]
    pub fn new(path: String) -> Result<Self> {
        validate_cloud_path(&path)?;
        Ok(Self(path))
    }

    /// Borrow the inner path string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ValidatedPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ValidatedPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Trait for cloud storage backends.
///
/// Implementations provide basic operations needed for archive read/write:
/// - `read_range`: Read a byte range (for seekable decompression)
/// - `write`: Upload complete data
/// - `delete`: Remove an object
/// - `size`: Get object size (for seek calculations)
/// - `exists`: Check if object exists
///
/// All `path` arguments use [`ValidatedPath`], which enforces validation at
/// the type level.  Backend implementations can trust that the path has been
/// checked for traversal attacks, null bytes, and header-injection characters.
pub trait StorageBackend: Send + Sync {
    /// Read a byte range from the object.
    ///
    /// Returns the bytes in `[offset, offset + length)`.
    fn read_range(&self, path: &ValidatedPath, offset: u64, length: u64) -> Result<Vec<u8>>;

    /// Write (upload) data to the given path.
    fn write(&self, path: &ValidatedPath, data: &[u8]) -> Result<()>;

    /// Delete the object at the given path.
    fn delete(&self, path: &ValidatedPath) -> Result<()>;

    /// Get the size of the object in bytes.
    fn size(&self, path: &ValidatedPath) -> Result<u64>;

    /// Check if the object exists.
    fn exists(&self, path: &ValidatedPath) -> Result<bool>;
}

/// A `Read + Seek` adapter over a `StorageBackend`.
///
/// Translates `read()` and `seek()` calls into range requests, enabling
/// seekable decompression directly from cloud storage without downloading
/// the entire archive.
///
/// Includes an internal read-ahead buffer to avoid per-byte network
/// round-trips.
///
/// # Path safety
///
/// The `path` is validated on construction via [`validate_cloud_path`].
///
/// # Note on `total_size`
///
/// The remote object size is queried once at construction and cached.  If
/// the remote object may change while this reader is alive, call
/// [`refresh_size`](Self::refresh_size) to re-query.
pub struct CloudReader<B: StorageBackend> {
    backend: B,
    path: ValidatedPath,
    position: u64,
    total_size: u64,
    /// Internal read-ahead buffer.
    buf: Vec<u8>,
    /// Byte offset in the remote object where `buf[0]` starts.
    buf_start: u64,
}

impl<B: StorageBackend> CloudReader<B> {
    /// Create a new `CloudReader` for the given object.
    ///
    /// Validates `path` and queries the object size on creation.
    pub fn new(backend: B, path: String) -> Result<Self> {
        let validated = ValidatedPath::new(path)?;
        let total_size = backend.size(&validated)?;
        Ok(Self {
            backend,
            path: validated,
            position: 0,
            total_size,
            buf: Vec::new(),
            buf_start: 0,
        })
    }

    /// Re-query the remote object size.
    ///
    /// Use this if the remote object may have been modified since
    /// construction (mitigates TOCTOU issues).
    pub fn refresh_size(&mut self) -> Result<()> {
        self.total_size = self.backend.size(&self.path)?;
        // Invalidate buffer — the remote content may have changed.
        self.buf.clear();
        Ok(())
    }

    /// Return the cached total size of the remote object.
    pub fn total_size(&self) -> u64 {
        self.total_size
    }
}

impl<B: StorageBackend> std::io::Read for CloudReader<B> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.position >= self.total_size {
            return Ok(0);
        }

        let remaining = self.total_size - self.position;
        let wanted = (buf.len() as u64).min(remaining) as usize;

        // Check if the internal buffer covers [position, position + wanted).
        let buf_end = self.buf_start + self.buf.len() as u64;
        if self.position >= self.buf_start && self.position + wanted as u64 <= buf_end {
            // Serve from buffer.
            let offset = (self.position - self.buf_start) as usize;
            buf[..wanted].copy_from_slice(&self.buf[offset..offset + wanted]);
            self.position += wanted as u64;
            return Ok(wanted);
        }

        // Buffer miss — fetch a new chunk from the backend.
        // Cap at MAX_FETCH_SIZE to prevent OOM if total_size is enormous
        // (e.g. a malicious server reporting u64::MAX).
        let fetch_len = (DEFAULT_BUFFER_SIZE as u64)
            .max(wanted as u64)
            .min(self.total_size - self.position)
            .min(MAX_FETCH_SIZE);

        let data = self
            .backend
            .read_range(&self.path, self.position, fetch_len)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        let n = data.len().min(wanted);
        buf[..n].copy_from_slice(&data[..n]);

        // Store the remainder in the internal buffer for subsequent reads.
        self.buf_start = self.position;
        self.buf = data;

        self.position += n as u64;
        Ok(n)
    }
}

impl<B: StorageBackend> std::io::Seek for CloudReader<B> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let new_pos: i64 = match pos {
            std::io::SeekFrom::Start(offset) => i64::try_from(offset).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "Seek offset overflow")
            })?,
            std::io::SeekFrom::End(offset) => i64::try_from(self.total_size)
                .ok()
                .and_then(|s| s.checked_add(offset))
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "Seek from end overflow")
                })?,
            std::io::SeekFrom::Current(offset) => i64::try_from(self.position)
                .ok()
                .and_then(|p| p.checked_add(offset))
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Seek from current overflow",
                    )
                })?,
        };

        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Seek before start of file",
            ));
        }

        let new_pos = new_pos as u64;

        // Clamp to total_size — seeking past EOF is not meaningful for
        // a fixed-size remote object and could cause out-of-range requests.
        self.position = new_pos.min(self.total_size);

        // Invalidate the buffer if the new position falls outside it.
        // This prevents serving stale data and frees memory sooner.
        let buf_end = self.buf_start + self.buf.len() as u64;
        if self.position < self.buf_start || self.position >= buf_end {
            self.buf.clear();
        }

        Ok(self.position)
    }
}

/// Parse a cloud URL into (scheme, bucket, key).
///
/// Supported schemes:
/// - `s3://bucket/key` -> `("s3", "bucket", "key")`
/// - `gs://bucket/key` -> `("gs", "bucket", "key")`
/// - `az://container/blob` -> `("az", "container", "blob")`
///
/// # Validation
///
/// Returns an error if the URL contains null bytes, newlines, empty
/// bucket/key components, or an unsupported scheme.
///
/// The returned `bucket` and `key` are borrowed from the input `url`, so
/// the caller must keep `url` alive for as long as the returned references
/// are used.
#[must_use = "parsed URL result must be checked"]
pub fn parse_cloud_url(url: &str) -> Result<(&str, &str, &str)> {
    // Reject dangerous characters before parsing.
    if url.contains('\0') {
        return Err(AetherError::InvalidCloudUrl(
            "URL must not contain null bytes".into(),
        ));
    }
    if url.contains('\n') || url.contains('\r') {
        return Err(AetherError::InvalidCloudUrl(
            "URL must not contain newline characters".into(),
        ));
    }

    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| AetherError::InvalidCloudUrl("missing '://' separator".into()))?;

    let (bucket, key) = rest
        .split_once('/')
        .ok_or_else(|| AetherError::InvalidCloudUrl("missing key after bucket".into()))?;

    if bucket.is_empty() {
        return Err(AetherError::InvalidCloudUrl(
            "bucket name must not be empty".into(),
        ));
    }

    if key.is_empty() {
        return Err(AetherError::InvalidCloudUrl(
            "object key must not be empty".into(),
        ));
    }

    // Block literal path traversal in the key.
    if key.contains("..") {
        return Err(AetherError::InvalidCloudUrl(
            "URL key must not contain '..' traversal sequences".into(),
        ));
    }

    // Validate bucket name format.
    // S3: 3-63 chars, lowercase alphanumeric, hyphens, dots.
    // GCS: 3-63 chars, lowercase alphanumeric, hyphens, dots, underscores.
    // Azure containers: 3-63 chars, lowercase alphanumeric, hyphens.
    // We use a permissive superset that covers all three providers.
    if bucket.len() < 3 || bucket.len() > 63 {
        return Err(AetherError::InvalidCloudUrl(
            "bucket name must be 3-63 characters".into(),
        ));
    }
    if !bucket
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.' || c == '_')
    {
        return Err(AetherError::InvalidCloudUrl(
            "bucket name must contain only lowercase letters, digits, hyphens, dots, and underscores".into(),
        ));
    }

    // Reject percent-encoded traversal sequences, separators, and null bytes.
    let lower = url.to_ascii_lowercase();

    // Encoded traversal: %2e = '.'
    if lower.contains("%2e%2e") || lower.contains("%2e.") || lower.contains(".%2e") {
        return Err(AetherError::InvalidCloudUrl(
            "URL contains encoded traversal sequence".into(),
        ));
    }
    // Double-encoded traversal: %252e decodes to %2e, which decodes to '.'
    if lower.contains("%252e") {
        return Err(AetherError::InvalidCloudUrl(
            "URL contains double-encoded traversal sequence".into(),
        ));
    }
    // Encoded path separators: %2f = '/', %5c = '\'
    if lower.contains("%2f") || lower.contains("%5c") {
        return Err(AetherError::InvalidCloudUrl(
            "URL contains encoded path separator".into(),
        ));
    }
    // Encoded null byte
    if lower.contains("%00") {
        return Err(AetherError::InvalidCloudUrl(
            "URL contains encoded null byte".into(),
        ));
    }
    // Triple+ encoded: block %25 to prevent arbitrary recursive decoding
    if lower.contains("%25") {
        return Err(AetherError::InvalidCloudUrl(
            "URL contains multiply-encoded sequences".into(),
        ));
    }
    // Backslashes (some backends interpret as separator)
    if key.contains('\\') {
        return Err(AetherError::InvalidCloudUrl(
            "URL key must not contain backslashes".into(),
        ));
    }
    // Overlong UTF-8 encodings: %c0%ae is an overlong encoding of '.'
    if lower.contains("%c0%ae") || lower.contains("%c0%af") {
        return Err(AetherError::InvalidCloudUrl(
            "URL contains overlong UTF-8 encoded sequences".into(),
        ));
    }
    // Unicode fullwidth characters that could normalize to dangerous ASCII
    for ch in url.chars() {
        if ch == '\u{FF0E}' || ch == '\u{FF0F}' || ch == '\u{FF3C}' {
            return Err(AetherError::InvalidCloudUrl(
                "URL contains Unicode fullwidth characters that could be normalized".into(),
            ));
        }
    }

    match scheme {
        "s3" | "gs" | "az" => Ok((scheme, bucket, key)),
        _ => Err(AetherError::InvalidCloudUrl(format!(
            "unsupported scheme '{scheme}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MockBackend ────────────────────────────────────────────────

    struct MockBackend {
        data: Vec<u8>,
    }

    impl StorageBackend for MockBackend {
        fn read_range(&self, _path: &ValidatedPath, offset: u64, length: u64) -> Result<Vec<u8>> {
            let start = usize::try_from(offset)
                .map_err(|_| AetherError::CloudStorage("offset exceeds platform usize".into()))?;
            let end = usize::try_from(offset.saturating_add(length))
                .map_err(|_| AetherError::CloudStorage("range exceeds platform usize".into()))?;
            let end = end.min(self.data.len());
            if start > self.data.len() {
                return Err(AetherError::CloudStorage(
                    "offset past end of object".into(),
                ));
            }
            Ok(self.data[start..end].to_vec())
        }
        fn write(&self, _path: &ValidatedPath, _data: &[u8]) -> Result<()> {
            Ok(())
        }
        fn delete(&self, _path: &ValidatedPath) -> Result<()> {
            Ok(())
        }
        fn size(&self, _path: &ValidatedPath) -> Result<u64> {
            Ok(self.data.len() as u64)
        }
        fn exists(&self, _path: &ValidatedPath) -> Result<bool> {
            Ok(true)
        }
    }

    fn mock_reader(data: Vec<u8>) -> CloudReader<MockBackend> {
        let backend = MockBackend { data };
        CloudReader::new(backend, "test/path.aet".to_string()).unwrap()
    }

    // ── parse_cloud_url tests ──────────────────────────────────────

    #[test]
    fn parse_s3_url() {
        let (scheme, bucket, key) = parse_cloud_url("s3://my-bucket/path/to/archive.aet").unwrap();
        assert_eq!(scheme, "s3");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/archive.aet");
    }

    #[test]
    fn parse_gs_url() {
        let (scheme, bucket, key) = parse_cloud_url("gs://bucket/key.aet").unwrap();
        assert_eq!(scheme, "gs");
        assert_eq!(bucket, "bucket");
        assert_eq!(key, "key.aet");
    }

    #[test]
    fn parse_az_url() {
        let (scheme, bucket, key) = parse_cloud_url("az://container/blob.aet").unwrap();
        assert_eq!(scheme, "az");
        assert_eq!(bucket, "container");
        assert_eq!(key, "blob.aet");
    }

    #[test]
    fn parse_invalid_url_no_scheme() {
        assert!(parse_cloud_url("invalid").is_err());
    }

    #[test]
    fn parse_invalid_url_bad_scheme() {
        assert!(parse_cloud_url("ftp://bucket/key").is_err());
    }

    #[test]
    fn parse_url_empty_bucket() {
        assert!(parse_cloud_url("s3:///key").is_err());
    }

    #[test]
    fn parse_url_empty_key() {
        assert!(parse_cloud_url("s3://bucket/").is_err());
    }

    #[test]
    fn parse_url_missing_key() {
        assert!(parse_cloud_url("s3://bucket").is_err());
    }

    #[test]
    fn parse_url_null_byte() {
        assert!(parse_cloud_url("s3://bucket/key\0.aet").is_err());
    }

    #[test]
    fn parse_url_newline() {
        assert!(parse_cloud_url("s3://bucket/key\n.aet").is_err());
        assert!(parse_cloud_url("s3://bucket/key\r.aet").is_err());
    }

    #[test]
    fn parse_url_encoded_traversal() {
        assert!(parse_cloud_url("s3://bucket/%2e%2e/etc/passwd").is_err());
    }

    #[test]
    fn parse_url_encoded_null() {
        assert!(parse_cloud_url("s3://bucket/key%00.aet").is_err());
    }

    #[test]
    fn parse_url_encoded_slash() {
        assert!(parse_cloud_url("s3://bucket/foo%2fbar").is_err());
        assert!(parse_cloud_url("s3://bucket/foo%5cbar").is_err());
    }

    #[test]
    fn parse_url_double_encoded_traversal() {
        assert!(parse_cloud_url("s3://bucket/%252e%252e/etc/passwd").is_err());
    }

    #[test]
    fn parse_url_backslash_in_key() {
        assert!(parse_cloud_url("s3://bucket/foo\\bar").is_err());
    }

    // ── validate_cloud_path tests ──────────────────────────────────

    #[test]
    fn valid_path() {
        assert!(validate_cloud_path("some/path/to/file.aet").is_ok());
    }

    #[test]
    fn path_empty() {
        assert!(validate_cloud_path("").is_err());
    }

    #[test]
    fn path_traversal() {
        assert!(validate_cloud_path("../etc/passwd").is_err());
        assert!(validate_cloud_path("foo/../../bar").is_err());
    }

    #[test]
    fn path_null_byte() {
        assert!(validate_cloud_path("foo\0bar").is_err());
    }

    #[test]
    fn path_newline() {
        assert!(validate_cloud_path("foo\nbar").is_err());
        assert!(validate_cloud_path("foo\rbar").is_err());
    }

    #[test]
    fn path_absolute() {
        assert!(validate_cloud_path("/absolute/path").is_err());
    }

    #[test]
    fn path_backslash() {
        assert!(validate_cloud_path("foo\\bar").is_err());
        assert!(validate_cloud_path("..\\..\\etc\\passwd").is_err());
    }

    #[test]
    fn path_encoded_traversal() {
        assert!(validate_cloud_path("%2e%2e/etc/passwd").is_err());
        assert!(validate_cloud_path("foo/%2e./bar").is_err());
        assert!(validate_cloud_path("foo/.%2e/bar").is_err());
    }

    #[test]
    fn path_encoded_separators() {
        assert!(validate_cloud_path("foo%2fbar").is_err());
        assert!(validate_cloud_path("foo%5cbar").is_err());
    }

    #[test]
    fn path_encoded_null() {
        assert!(validate_cloud_path("foo%00bar").is_err());
    }

    #[test]
    fn validated_path_newtype() {
        assert!(ValidatedPath::new("valid/path.aet".to_string()).is_ok());
        assert!(ValidatedPath::new("../traversal".to_string()).is_err());
        let vp = ValidatedPath::new("test/path.aet".to_string()).unwrap();
        assert_eq!(vp.as_str(), "test/path.aet");
    }

    // ── CloudReader tests ──────────────────────────────────────────

    #[test]
    fn cloud_reader_rejects_invalid_path() {
        let backend = MockBackend { data: vec![0; 10] };
        assert!(CloudReader::new(backend, "../traversal".to_string()).is_err());
    }

    #[test]
    fn cloud_reader_seek_and_read() {
        let mut reader = mock_reader((0..100u8).collect());

        // Read first 10 bytes
        let mut buf = [0u8; 10];
        use std::io::Read;
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // Seek to position 50
        use std::io::Seek;
        reader.seek(std::io::SeekFrom::Start(50)).unwrap();
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]);

        // Seek from end
        reader.seek(std::io::SeekFrom::End(-5)).unwrap();
        let mut small_buf = [0u8; 5];
        reader.read_exact(&mut small_buf).unwrap();
        assert_eq!(small_buf, [95, 96, 97, 98, 99]);
    }

    #[test]
    fn cloud_reader_seek_past_end_clamps() {
        use std::io::{Read, Seek};

        let mut reader = mock_reader((0..10u8).collect());

        // Seek way past end — should clamp to total_size
        let pos = reader.seek(std::io::SeekFrom::Start(9999)).unwrap();
        assert_eq!(pos, 10); // clamped to total_size

        // Read should return 0 (EOF)
        let mut buf = [0u8; 1];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn cloud_reader_seek_before_start_errors() {
        use std::io::Seek;

        let mut reader = mock_reader((0..10u8).collect());
        let result = reader.seek(std::io::SeekFrom::Current(-1));
        assert!(result.is_err());
    }

    #[test]
    fn cloud_reader_zero_size_object() {
        use std::io::Read;

        let mut reader = mock_reader(vec![]);
        let mut buf = [0u8; 1];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn cloud_reader_buffered_sequential_reads() {
        use std::io::Read;

        let mut reader = mock_reader((0..=255).cycle().take(1024).collect());

        // Many small reads should be served from the prefetch buffer
        // without calling read_range for each one.
        let mut total = Vec::new();
        let mut buf = [0u8; 7]; // intentionally odd size
        loop {
            let n = reader.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            total.extend_from_slice(&buf[..n]);
        }
        let expected: Vec<u8> = (0..=255).cycle().take(1024).collect();
        assert_eq!(total, expected);
    }

    #[test]
    fn cloud_reader_refresh_size() {
        let mut reader = mock_reader((0..10u8).collect());
        assert_eq!(reader.total_size(), 10);
        // refresh_size re-queries (same backend, same answer here)
        reader.refresh_size().unwrap();
        assert_eq!(reader.total_size(), 10);
    }

    // ── New: overlong/fullwidth/multi-encoding path tests ─────────

    #[test]
    fn path_double_encoded_traversal() {
        assert!(validate_cloud_path("%252e%252e/etc/passwd").is_err());
    }

    #[test]
    fn path_triple_encoded_blocked_by_percent25() {
        assert!(validate_cloud_path("%25252e/foo").is_err());
    }

    #[test]
    fn path_overlong_utf8_dot() {
        assert!(validate_cloud_path("%c0%ae%c0%ae/etc/passwd").is_err());
    }

    #[test]
    fn path_overlong_utf8_slash() {
        assert!(validate_cloud_path("foo%c0%afbar").is_err());
    }

    #[test]
    fn path_unicode_fullwidth_period() {
        assert!(validate_cloud_path("foo\u{FF0E}\u{FF0E}/bar").is_err());
    }

    #[test]
    fn path_unicode_fullwidth_slash() {
        assert!(validate_cloud_path("foo\u{FF0F}bar").is_err());
    }

    #[test]
    fn path_unicode_fullwidth_backslash() {
        assert!(validate_cloud_path("foo\u{FF3C}bar").is_err());
    }

    // ── New: parse_cloud_url key traversal & bucket validation ────

    #[test]
    fn parse_url_key_literal_traversal() {
        assert!(parse_cloud_url("s3://bucket/foo/../secret").is_err());
        assert!(parse_cloud_url("s3://bucket/../etc/passwd").is_err());
    }

    #[test]
    fn parse_url_bucket_too_short() {
        assert!(parse_cloud_url("s3://ab/key.aet").is_err());
    }

    #[test]
    fn parse_url_bucket_too_long() {
        let long_bucket = "a".repeat(64);
        assert!(parse_cloud_url(&format!("s3://{long_bucket}/key.aet")).is_err());
    }

    #[test]
    fn parse_url_bucket_invalid_chars() {
        assert!(parse_cloud_url("s3://BUCKET/key.aet").is_err());
        assert!(parse_cloud_url("s3://buck et/key.aet").is_err());
        assert!(parse_cloud_url("s3://buck@t/key.aet").is_err());
    }

    #[test]
    fn parse_url_bucket_valid_special_chars() {
        // Dots, hyphens, underscores are allowed
        assert!(parse_cloud_url("s3://my-bucket.v1_data/key.aet").is_ok());
    }

    #[test]
    fn parse_url_overlong_utf8() {
        assert!(parse_cloud_url("s3://bucket/%c0%ae%c0%ae/foo").is_err());
    }

    #[test]
    fn parse_url_fullwidth_chars() {
        assert!(parse_cloud_url("s3://bucket/foo\u{FF0E}\u{FF0E}/bar").is_err());
    }

    #[test]
    fn parse_url_multiply_encoded() {
        assert!(parse_cloud_url("s3://bucket/foo%2525bar").is_err());
    }
}
