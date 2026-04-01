use crate::error::{AetherError, Result};

// ── Magic & Version ──────────────────────────────────────────────────────────

/// Magic bytes identifying an AetherArch archive: `0xAE "ther" 0x00 <major> <minor>`.
pub const MAGIC: [u8; 8] = [0xAE, 0x74, 0x68, 0x65, 0x72, 0x00, 0x01, 0x00];

/// Format major version number.
pub const FORMAT_VERSION_MAJOR: u8 = 0x01;

/// Format minor version number.
/// Bumped from 0 to 1 for BLAKE3 header integrity and encryption header v2.
pub const FORMAT_VERSION_MINOR: u8 = 0x01;

/// Magic bytes at the start of each compressed data block.
pub const BLOCK_MAGIC: u32 = 0xB10C_AE01;

/// Magic bytes at the end of the archive footer (`0xAE "END"`).
pub const FOOTER_MAGIC: u32 = 0xAE45_4E44;

// ── Size Constants ───────────────────────────────────────────────────────────

/// Size of the fixed archive header in bytes.
pub const ARCHIVE_HEADER_SIZE: usize = 48;

/// Size of each block header in bytes.
pub const BLOCK_HEADER_SIZE: usize = 28;

/// Size of each block trailer in bytes.
pub const BLOCK_TRAILER_SIZE: usize = 36;

/// Size of each block index entry in bytes.
pub const BLOCK_INDEX_ENTRY_SIZE: usize = 24;

/// Size of the fixed archive footer in bytes.
pub const ARCHIVE_FOOTER_SIZE: usize = 32;

/// Size of each solid group table entry in bytes.
pub const SOLID_GROUP_ENTRY_SIZE: usize = 24;

// ── Safety Limits ───────────────────────────────────────────────────────────

/// Maximum decompressed block size (64 MiB). Prevents OOM from crafted archives
/// claiming enormous uncompressed sizes.
pub const MAX_DECOMPRESSED_BLOCK_SIZE: usize = 64 * 1024 * 1024;

/// Maximum number of files allowed in a single archive. Prevents excessive
/// memory allocation from crafted headers.
pub const MAX_FILE_COUNT: u32 = 1_000_000;

/// Maximum number of blocks allowed in a single archive.
pub const MAX_BLOCK_COUNT: u32 = 10_000_000;

/// Maximum number of solid groups allowed in a single archive.
pub const MAX_SOLID_GROUP_COUNT: u32 = 100_000;

/// Maximum file path length in bytes (stored as u16, but capped lower for sanity).
pub const MAX_PATH_LENGTH: usize = 4096;

/// Maximum total decompressed output per archive (8 GiB).
///
/// Prevents decompression bomb attacks where a crafted archive contains many
/// small compressed blocks that expand to enormous total output. Without this
/// limit, an archive with 10 million 64 MiB blocks could decompress to 640 PB.
pub const MAX_TOTAL_DECOMPRESSED_SIZE: u64 = 8 * 1024 * 1024 * 1024;

/// Maximum total compressed bytes read per archive (16 GiB).
///
/// Prevents memory exhaustion from crafted archives that claim many blocks
/// with large `compressed_size` fields. Each block's payload is allocated
/// before decompression, so without this limit an archive could force
/// sequential allocation of up to `MAX_BLOCK_COUNT * MAX_DECOMPRESSED_BLOCK_SIZE`.
pub const MAX_TOTAL_COMPRESSED_READ_SIZE: u64 = 16 * 1024 * 1024 * 1024;

/// Maximum capacity hint for `Vec::with_capacity` when sizes come from
/// untrusted archive metadata. Prevents speculative over-allocation from
/// crafted headers that claim millions of entries.
pub const MAX_PREALLOC_CAPACITY: usize = 4096;

/// Minimum Argon2id memory cost (16 MiB in KiB). Prevents crafted archives
/// from setting trivially low parameters that make brute-force feasible.
pub const MIN_ARGON2_M_COST: u32 = 16 * 1024;

/// Minimum Argon2id iteration count.
pub const MIN_ARGON2_T_COST: u32 = 2;

/// Minimum Argon2id parallelism lanes.
pub const MIN_ARGON2_P_COST: u32 = 1;

/// Maximum Argon2id memory cost (m_cost) allowed from archive headers (1 GiB).
/// Prevents crafted archives from forcing extreme memory allocation during
/// key derivation.
///
/// This is the single source of truth for Argon2 bounds — used by both the
/// encryption module (compression-time validation) and the decompression
/// module (archive-read-time validation).
pub const MAX_ARGON2_M_COST: u32 = 1_048_576; // 1 GiB in KiB

/// Maximum Argon2id time cost (t_cost) allowed from archive headers.
pub const MAX_ARGON2_T_COST: u32 = 16;

/// Maximum Argon2id parallelism cost (p_cost) allowed from archive headers.
pub const MAX_ARGON2_P_COST: u32 = 16;

/// Permission mask applied to file modes extracted from archives.
/// Strips setuid (4000), setgid (2000), and sticky (1000) bits to prevent
/// privilege escalation from crafted archives.
pub const SAFE_PERMISSION_MASK: u32 = 0o0777;

/// Maximum total input size during compression (8 GiB).
///
/// Prevents unbounded memory consumption since the compressor reads all file
/// data into memory before compression. Without this limit, a large file list
/// could exhaust available RAM.
pub const MAX_TOTAL_INPUT_SIZE: u64 = 8 * 1024 * 1024 * 1024;

/// Size of the password verification tag in the encryption header.
/// BLAKE3 keyed hash of a known constant, used for fast-fail on wrong password.
pub const VERIFICATION_TAG_SIZE: usize = 32;

/// Reserved block_id used for metadata encryption (file table + group table).
/// Must never collide with real block IDs (real IDs start at 0 and increment).
pub const ENCRYPTED_METADATA_BLOCK_ID: u32 = u32::MAX;

// ── Routing Thresholds ──────────────────────────────────────────────────────

/// BWT compression ratio below which LZ77 is skipped (BWT won decisively).
/// At 55% compression, BWT has reduced the data enough that LZ77 is unlikely
/// to beat it, and the predictor sync can be skipped for speed.
pub const BWT_DECISIVE_RATIO: usize = 55;

// ── Enums ────────────────────────────────────────────────────────────────────

/// Identifies which predictor was used to compress the archive.
///
/// Stored as a `u16` in the archive header. The decompressor uses this to
/// instantiate the correct predictor for decoding.
#[non_exhaustive]
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictorId {
    Order0 = 0x0000,
    ContextMixer = 0x0001,
    NeuralSsm = 0x0002,
    ContextMixerLight = 0x0003,
    Lz4Aware = 0x0004,
    /// RLE-aware context-class predictor for BWT+MTF+RLE streams.
    Rle = 0x0005,
    /// MTF-aware predictor for BWT+MTF preprocessed data.
    Mtf = 0x0006,
    ZstdOnly = 0x00FF,
}

impl PredictorId {
    pub fn from_u16(v: u16) -> Result<Self> {
        match v {
            0x0000 => Ok(Self::Order0),
            0x0001 => Ok(Self::ContextMixer),
            0x0002 => Ok(Self::NeuralSsm),
            0x0003 => Ok(Self::ContextMixerLight),
            0x0004 => Ok(Self::Lz4Aware),
            0x0005 => Ok(Self::Rle),
            0x0006 => Ok(Self::Mtf),
            0x00FF => Ok(Self::ZstdOnly),
            other => Err(AetherError::UnknownPredictorId(other)),
        }
    }
}

/// Per-block compression method stored in the block header.
///
/// Each chunk is independently compressed with the method that produces
/// the smallest output. The routing cascade in `pipeline::router` tries
/// them in order and picks the winner.
#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionMethod {
    /// Predictor + Range Coding (main path)
    PredictorRans = 0,
    /// Zstandard fallback for high-entropy data
    Zstd = 1,
    /// Stored uncompressed (incompressible data)
    Store = 2,
    /// LZ4 preprocessing + Predictor + Range Coding
    LzPredictorRans = 3,
    /// LZ77 preprocessing (min-match-3, lazy) + Predictor + Range Coding
    Lz77PredictorRans = 4,
    /// BWT + MTF preprocessing + Predictor + Range Coding (best for text)
    BwtPredictorRans = 5,
    /// Byte-plane splitting + per-plane Range Coding (best for numeric/float arrays)
    BytePlanePredictorRans = 6,
}

impl CompressionMethod {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::PredictorRans),
            1 => Ok(Self::Zstd),
            2 => Ok(Self::Store),
            3 => Ok(Self::LzPredictorRans),
            4 => Ok(Self::Lz77PredictorRans),
            5 => Ok(Self::BwtPredictorRans),
            6 => Ok(Self::BytePlanePredictorRans),
            other => Err(AetherError::UnknownCompressionMethod(other)),
        }
    }
}

/// Semantic content type used for solid grouping.
///
/// Files with the same content type are grouped together in solid blocks,
/// allowing the predictor to learn cross-file patterns within each group.
#[non_exhaustive]
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    Mixed = 0,
    Text = 1,
    BinaryStructured = 2,
    BinaryRandom = 3,
    Image = 4,
    Executable = 5,
    /// Numeric arrays / tensor data (float weights, embeddings, etc.)
    NumericData = 6,
}

impl ContentType {
    pub fn from_u16(v: u16) -> Result<Self> {
        match v {
            0 => Ok(Self::Mixed),
            1 => Ok(Self::Text),
            2 => Ok(Self::BinaryStructured),
            3 => Ok(Self::BinaryRandom),
            4 => Ok(Self::Image),
            5 => Ok(Self::Executable),
            6 => Ok(Self::NumericData),
            other => Err(AetherError::UnknownContentType(other)),
        }
    }
}

// ── Header Flags ─────────────────────────────────────────────────────────────

/// Header flag: archive contains a neural model predictor.
pub const FLAG_HAS_NEURAL_MODEL: u16 = 1 << 0;

/// Header flag: archive uses solid grouping (files grouped by content type).
pub const FLAG_SOLID_ARCHIVE: u16 = 1 << 1;

/// Header flag: archive is encrypted. Reserved for future use.
pub const FLAG_ENCRYPTED: u16 = 1 << 2;

/// Header flag: archive was compressed with a dictionary (requires matching .aed file).
pub const FLAG_HAS_DICTIONARY: u16 = 1 << 3;

// ── Shannon Entropy ──────────────────────────────────────────────────────────

/// Calculate Shannon entropy of a byte slice.
///
/// Returns bits per byte in the range `[0.0, 8.0]`.
/// - 0.0 means perfectly uniform (all identical bytes).
/// - 8.0 means maximum entropy (uniformly random).
pub fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut freq = [0u64; 256];
    for &byte in data {
        freq[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0f64;

    for &count in &freq {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_of_empty() {
        assert_eq!(shannon_entropy(&[]), 0.0);
    }

    #[test]
    fn entropy_of_uniform() {
        // All same byte → zero entropy
        let data = vec![0x42u8; 1000];
        assert!((shannon_entropy(&data) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn entropy_of_two_symbols() {
        // Equal mix of two symbols → 1.0 bit/byte
        let mut data = vec![0u8; 1000];
        data[0..500].fill(1);
        let e = shannon_entropy(&data);
        assert!((e - 1.0).abs() < 0.01, "Expected ~1.0, got {e}");
    }

    #[test]
    fn entropy_of_random_is_high() {
        // All 256 byte values equally → 8.0 bits/byte
        let mut data = Vec::with_capacity(256 * 100);
        for _ in 0..100 {
            for b in 0u8..=255 {
                data.push(b);
            }
        }
        let e = shannon_entropy(&data);
        assert!((e - 8.0).abs() < 0.01, "Expected ~8.0, got {e}");
    }

    #[test]
    fn predictor_id_roundtrip() {
        for id in [
            PredictorId::Order0,
            PredictorId::ContextMixer,
            PredictorId::NeuralSsm,
            PredictorId::ContextMixerLight,
            PredictorId::Lz4Aware,
            PredictorId::Rle,
            PredictorId::ZstdOnly,
        ] {
            let v = id as u16;
            assert_eq!(PredictorId::from_u16(v).unwrap(), id);
        }
    }

    #[test]
    fn compression_method_roundtrip() {
        for m in [
            CompressionMethod::PredictorRans,
            CompressionMethod::Zstd,
            CompressionMethod::Store,
            CompressionMethod::LzPredictorRans,
            CompressionMethod::Lz77PredictorRans,
            CompressionMethod::BwtPredictorRans,
            CompressionMethod::BytePlanePredictorRans,
        ] {
            let v = m as u8;
            assert_eq!(CompressionMethod::from_u8(v).unwrap(), m);
        }
    }

    #[test]
    fn content_type_roundtrip() {
        for ct in [
            ContentType::Mixed,
            ContentType::Text,
            ContentType::BinaryStructured,
            ContentType::BinaryRandom,
            ContentType::Image,
            ContentType::Executable,
            ContentType::NumericData,
        ] {
            let v = ct as u16;
            assert_eq!(ContentType::from_u16(v).unwrap(), ct);
        }
    }
}
