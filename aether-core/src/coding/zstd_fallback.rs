//! Zstandard fallback compression for high-entropy data.
//!
//! When the entropy analyzer determines that data is too random for the
//! neural/context predictor to beat zstd, we fall back to zstd which
//! handles near-random data efficiently.

use std::io::Read;

use crate::error::{AetherError, Result};

/// Default zstd compression level (balance of speed and ratio).
pub const ZSTD_COMPRESSION_LEVEL: i32 = 3;

/// Compress a data block using Zstandard.
pub fn compress(data: &[u8]) -> Result<Vec<u8>> {
    zstd::encode_all(std::io::Cursor::new(data), ZSTD_COMPRESSION_LEVEL)
        .map_err(|e| AetherError::Compression(format!("zstd compress: {e}")))
}

/// Maximum decompressible output size for zstd fallback.
///
/// Aligned with [`crate::format::MAX_DECOMPRESSED_BLOCK_SIZE`] (64 MiB) to
/// prevent decompression bombs from crafted zstd frames that declare enormous
/// output sizes in their frame header.
const MAX_ZSTD_DECODE_SIZE: usize = crate::format::MAX_DECOMPRESSED_BLOCK_SIZE;

/// Decompress a Zstandard-compressed data block.
///
/// `expected_size` must match the original uncompressed length.  The output is
/// validated against both `expected_size` and `MAX_ZSTD_DECODE_SIZE` to guard
/// against decompression bombs.
///
/// Uses a capacity-bounded read to prevent the zstd frame header from
/// triggering an allocation larger than `expected_size` (V1 security fix:
/// `zstd::decode_all` trusts the frame header's declared size, so a crafted
/// frame claiming gigabytes would OOM before the size-mismatch check).
pub fn decompress(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    if expected_size > MAX_ZSTD_DECODE_SIZE {
        return Err(AetherError::Decompression(format!(
            "zstd decode size {} exceeds safety limit {}",
            expected_size, MAX_ZSTD_DECODE_SIZE,
        )));
    }

    let mut decoder = zstd::stream::Decoder::new(std::io::Cursor::new(data))
        .map_err(|e| AetherError::Decompression(format!("zstd decoder init: {e}")))?;

    // Allocate exactly expected_size and read into it with a hard limit.
    // Read at most expected_size + 1 so we can detect frames that produce
    // more output than expected without allocating unbounded memory.
    let limit = expected_size.saturating_add(1);
    let mut decompressed = Vec::with_capacity(expected_size);
    let bytes_read = std::io::Read::take(&mut decoder, limit as u64)
        .read_to_end(&mut decompressed)
        .map_err(|e| AetherError::Decompression(format!("zstd decompress: {e}")))?;

    if bytes_read != expected_size {
        return Err(AetherError::Decompression(format!(
            "zstd decode size mismatch: expected {expected_size}, got {bytes_read}",
        )));
    }

    Ok(decompressed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let data: &[u8] = &[];
        let compressed = compress(data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn roundtrip_text() {
        let data = b"Hello, World! This is a test of zstd compression. Repeated text helps. Repeated text helps.";
        let compressed = compress(data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn roundtrip_binary() {
        let data: Vec<u8> = (0..10_000).map(|i| (i * 7 % 256) as u8).collect();
        let compressed = compress(&data).unwrap();
        let decompressed = decompress(&compressed, data.len()).unwrap();
        assert_eq!(data, decompressed);
    }

    #[test]
    fn compresses_repetitive_data() {
        let data = vec![0x42u8; 100_000];
        let compressed = compress(&data).unwrap();
        assert!(
            compressed.len() < data.len() / 10,
            "zstd should compress 100K identical bytes well: {} -> {}",
            data.len(),
            compressed.len()
        );
    }

    #[test]
    fn decode_rejects_oversized() {
        let data = b"small";
        let compressed = compress(data).unwrap();
        let err = decompress(&compressed, MAX_ZSTD_DECODE_SIZE + 1);
        assert!(
            err.is_err(),
            "should reject expected_size beyond safety limit"
        );
    }

    #[test]
    fn decode_rejects_size_mismatch() {
        let data = b"test data for mismatch";
        let compressed = compress(data).unwrap();
        let err = decompress(&compressed, data.len() + 10);
        assert!(err.is_err(), "should reject mismatched expected_size");
    }
}
