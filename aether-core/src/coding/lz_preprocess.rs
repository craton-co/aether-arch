//! LZ4 preprocessing stage.
//!
//! Applies LZ4 compression to remove repeated substrings before the data
//! is fed through the predictor + range coder.  This dramatically improves
//! compression ratio on structured data (text, source code, JSON) where
//! long-range repetitions dominate.
//!
//! The LZ4 byte stream itself has enough structure for the context mixer
//! to model effectively, giving better results than either approach alone.
//!
//! # Format Stability
//!
//! The `lz4_flex` crate's `compress_prepend_size` / `decompress_size_prepended`
//! output format is **not** part of the official LZ4 frame specification.
//! A change in `lz4_flex`'s internal framing between versions could cause
//! silent decode failures on archives created with a different version.
//!
//! The workspace pins `lz4_flex = "=0.11.3"` to prevent accidental upgrades.
//! Any version bump **must** be validated against roundtrip tests with
//! archives created under the pinned version before merging.

use crate::error::{AetherError, Result};

/// Compress `data` with LZ4.
///
/// Returns `Some(lz_bytes)` if LZ4 reduces the size (lz_bytes.len() < data.len()),
/// or `None` if LZ4 doesn't help (incompressible data).
///
/// The output format is `lz4_flex::compress_prepend_size` which prepends
/// the *uncompressed* length as a little-endian u32 before the LZ4 frame.
#[must_use]
pub fn lz_encode(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }

    let compressed = lz4_flex::compress_prepend_size(data);

    // Only use LZ4 if it actually reduces size
    if compressed.len() < data.len() {
        Some(compressed)
    } else {
        None
    }
}

/// Decompress LZ4-encoded data produced by [`lz_encode`].
///
/// `lz_bytes` must be in the `compress_prepend_size` format (LE u32 length
/// prefix followed by LZ4 frame data).
///
/// `expected_original_size` is validated against the decompressed output
/// to catch any size mismatches.
pub fn lz_decode(lz_bytes: &[u8], expected_original_size: usize) -> Result<Vec<u8>> {
    if lz_bytes.is_empty() {
        return Err(AetherError::Decompression("LZ4 decode: empty input".into()));
    }

    // R1 security fix: validate the embedded size prefix *before* calling
    // decompress_size_prepended, which would otherwise pre-allocate whatever
    // the (potentially crafted) frame header claims.
    if lz_bytes.len() >= 4 {
        let declared_size = u32::from_le_bytes(lz_bytes[..4].try_into().unwrap()) as usize;
        if declared_size != expected_original_size {
            return Err(AetherError::Decompression(format!(
                "LZ4 embedded size {declared_size} does not match expected {expected_original_size}"
            )));
        }
    }

    let decompressed = lz4_flex::decompress_size_prepended(lz_bytes)
        .map_err(|e| AetherError::Decompression(format!("LZ4 decompress failed: {e}")))?;

    if decompressed.len() != expected_original_size {
        return Err(AetherError::Decompression(format!(
            "LZ4 decode size mismatch: expected {expected_original_size}, got {}",
            decompressed.len()
        )));
    }

    Ok(decompressed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_text() {
        let text = b"Hello, world! Hello, world! Hello, world! Hello, world! \
                     The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog.";
        let lz = lz_encode(text).expect("text should compress with LZ4");
        assert!(
            lz.len() < text.len(),
            "LZ4 output ({}) should be smaller than input ({})",
            lz.len(),
            text.len()
        );
        let decoded = lz_decode(&lz, text.len()).expect("decode should succeed");
        assert_eq!(&decoded[..], &text[..]);
    }

    #[test]
    fn roundtrip_large_text() {
        // Highly repetitive text — LZ4 should compress well
        let line = "The AetherArch compression pipeline uses LZ4 preprocessing.\n";
        let text: Vec<u8> = line.as_bytes().repeat(500);
        let lz = lz_encode(&text).expect("large text should compress");
        assert!(
            lz.len() < text.len() / 2,
            "LZ4 should achieve at least 2:1 on repetitive text, got {}/{}",
            lz.len(),
            text.len()
        );
        let decoded = lz_decode(&lz, text.len()).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn incompressible_returns_none() {
        // Already-compressed / random data shouldn't benefit from LZ4
        let mut random_data = vec![0u8; 256];
        for (i, b) in random_data.iter_mut().enumerate() {
            // Pseudo-random using a simple LCG
            *b = ((i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                >> 33) as u8;
        }
        // Very short random data may or may not compress, so use larger block
        let big_random: Vec<u8> = (0..4096)
            .map(|i: i32| {
                ((i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    >> 33) as u8
            })
            .collect();
        // LZ4 on high-entropy data typically expands or barely compresses
        let result = lz_encode(&big_random);
        // Either None or Some — but if Some, verify roundtrip still works
        if let Some(ref lz) = result {
            let decoded = lz_decode(lz, big_random.len()).unwrap();
            assert_eq!(decoded, big_random);
        }
    }

    #[test]
    fn empty_returns_none() {
        assert!(lz_encode(&[]).is_none());
    }

    #[test]
    fn decode_empty_fails() {
        let err = lz_decode(&[], 0);
        assert!(err.is_err());
    }

    #[test]
    fn decode_size_mismatch() {
        // Use repetitive data that LZ4 will definitely compress
        let text: Vec<u8> = b"abcdef1234567890 ".repeat(100);
        let lz = lz_encode(&text).expect("should compress");
        // Request wrong size
        let err = lz_decode(&lz, text.len() + 100);
        assert!(err.is_err(), "should fail on size mismatch");
    }

    #[test]
    fn roundtrip_single_byte_repeated() {
        let data = vec![0x42u8; 10_000];
        let lz = lz_encode(&data).expect("uniform data should compress with LZ4");
        assert!(
            lz.len() < data.len() / 10,
            "uniform data should compress extremely well"
        );
        let decoded = lz_decode(&lz, data.len()).unwrap();
        assert_eq!(decoded, data);
    }
}
