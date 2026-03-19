use thiserror::Error;

/// All errors that can occur within AetherArch operations.
///
/// This enum is `#[non_exhaustive]` — new error variants may be added in
/// future versions without a semver-breaking change.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum AetherError {
    #[error("Invalid magic bytes: expected AetherArch header")]
    InvalidMagic,

    #[error("Unsupported format version {major}.{minor}")]
    UnsupportedVersion { major: u8, minor: u8 },

    #[error("Block checksum mismatch at block {block_id}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        block_id: u32,
        expected: String,
        actual: String,
    },

    #[error("Header CRC mismatch: expected {expected:#010x}, got {actual:#010x}")]
    HeaderCrcMismatch { expected: u32, actual: u32 },

    #[error("Block CRC mismatch at block {block_id}")]
    BlockCrcMismatch { block_id: u32 },

    #[error("Invalid block magic at offset {offset}")]
    InvalidBlockMagic { offset: u64 },

    #[error("Footer magic mismatch")]
    InvalidFooterMagic,

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Encryption error: {0}")]
    Encryption(String),

    #[error("Decryption error: {0}")]
    Decryption(String),

    #[error("Predictor error: {0}")]
    Predictor(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Block {0} not found in index")]
    BlockNotFound(u32),

    #[error("File not found in archive: {0}")]
    FileNotFound(String),

    #[error("Truncated archive: expected at least {expected} bytes, got {actual}")]
    TruncatedArchive { expected: u64, actual: u64 },

    #[error("Unknown compression method: {0}")]
    UnknownCompressionMethod(u8),

    #[error("Unknown predictor ID: {0}")]
    UnknownPredictorId(u16),

    #[error("Unknown content type: {0}")]
    UnknownContentType(u16),

    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),

    #[error("Path traversal detected in archive entry: {0}")]
    PathTraversal(String),

    #[error("Cloud storage error: {0}")]
    CloudStorage(String),

    #[error("Invalid cloud path: {0}")]
    InvalidCloudPath(String),

    #[error("Invalid cloud URL: {0}")]
    InvalidCloudUrl(String),

    #[error("Nonce mismatch for block {block_id}: stored nonce does not match derived nonce")]
    NonceMismatch { block_id: u32 },

    #[error("Password verification failed: incorrect password or corrupted header")]
    PasswordVerificationFailed,

    #[error("Non-UTF-8 file path in archive")]
    InvalidUtf8Path,

    #[error("Header integrity check failed")]
    HeaderIntegrityMismatch,

    #[error("File already exists (no_clobber mode): {0}")]
    FileAlreadyExists(String),
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, AetherError>;
