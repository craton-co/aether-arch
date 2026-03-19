//! Content-Defined Chunking using FastCDC v2020.
//!
//! Breaks input data into variable-size chunks whose boundaries are determined
//! by the content itself (not fixed offsets). This means inserting bytes at the
//! start of a file only affects nearby chunk boundaries, preserving deduplication.

use crate::format::shannon_entropy;

/// Minimum chunk size (16 KiB).
pub const MIN_CHUNK_SIZE: u32 = 16 * 1024;
/// Average chunk size (512 KiB).
pub const AVG_CHUNK_SIZE: u32 = 512 * 1024;
/// Maximum chunk size (4 MiB).
pub const MAX_CHUNK_SIZE: u32 = 4 * 1024 * 1024;

/// A content-defined chunk with precomputed metadata.
#[derive(Debug)]
pub struct Chunk {
    /// Offset within the original input stream.
    pub offset: u64,
    /// Length in bytes.
    pub length: usize,
    /// The raw chunk data.
    pub data: Vec<u8>,
    /// BLAKE3 hash of `data`.
    pub blake3_hash: [u8; 32],
    /// Shannon entropy of `data` (bits per byte, 0.0..8.0).
    pub entropy: f64,
}

/// Break a byte slice into content-defined chunks.
///
/// Each chunk has its BLAKE3 hash and Shannon entropy precomputed,
/// ready for dedup detection and routing decisions.
pub fn chunk_data(data: &[u8]) -> Vec<Chunk> {
    if data.is_empty() {
        return Vec::new();
    }

    let chunker =
        fastcdc::v2020::FastCDC::new(data, MIN_CHUNK_SIZE, AVG_CHUNK_SIZE, MAX_CHUNK_SIZE);
    let mut chunks = Vec::new();

    for entry in chunker {
        let chunk_data = data[entry.offset..entry.offset + entry.length].to_vec();
        let hash = blake3::hash(&chunk_data);
        let entropy = shannon_entropy(&chunk_data);

        chunks.push(Chunk {
            offset: entry.offset as u64,
            length: entry.length,
            data: chunk_data,
            blake3_hash: *hash.as_bytes(),
            entropy,
        });
    }

    chunks
}

/// Break a byte slice into fixed-size blocks (fallback for small inputs).
pub fn chunk_fixed(data: &[u8], block_size: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut offset = 0usize;

    while offset < data.len() {
        let end = (offset + block_size).min(data.len());
        let chunk_data = data[offset..end].to_vec();
        let hash = blake3::hash(&chunk_data);
        let entropy = shannon_entropy(&chunk_data);

        chunks.push(Chunk {
            offset: offset as u64,
            length: chunk_data.len(),
            data: chunk_data,
            blake3_hash: *hash.as_bytes(),
            entropy,
        });
        offset = end;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_empty() {
        let chunks = chunk_data(&[]);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_small_data() {
        // Data smaller than MIN_CHUNK_SIZE → single chunk
        let data = vec![0x42u8; 1000];
        let chunks = chunk_data(&data);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].length, 1000);
        assert_eq!(chunks[0].data, data);
    }

    #[test]
    fn chunk_deterministic() {
        let data: Vec<u8> = (0..200_000).map(|i| (i % 256) as u8).collect();
        let chunks1 = chunk_data(&data);
        let chunks2 = chunk_data(&data);

        assert_eq!(chunks1.len(), chunks2.len());
        for (a, b) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.length, b.length);
            assert_eq!(a.blake3_hash, b.blake3_hash);
        }
    }

    #[test]
    fn chunk_covers_all_data() {
        let data: Vec<u8> = (0..500_000).map(|i| (i * 7 % 256) as u8).collect();
        let chunks = chunk_data(&data);

        let total_len: usize = chunks.iter().map(|c| c.length).sum();
        assert_eq!(total_len, data.len());

        // Chunks are contiguous
        let mut expected_offset = 0u64;
        for chunk in &chunks {
            assert_eq!(chunk.offset, expected_offset);
            expected_offset += chunk.length as u64;
        }
    }

    #[test]
    fn chunk_entropy_computed() {
        let data = vec![0u8; 100_000];
        let chunks = chunk_data(&data);
        for chunk in &chunks {
            assert!(chunk.entropy < 0.01, "Uniform data should have ~0 entropy");
        }
    }

    #[test]
    fn fixed_chunking() {
        let data = vec![0xABu8; 10_000];
        let chunks = chunk_fixed(&data, 4096);
        assert_eq!(chunks.len(), 3); // 4096 + 4096 + 1808
        assert_eq!(chunks[0].length, 4096);
        assert_eq!(chunks[1].length, 4096);
        assert_eq!(chunks[2].length, 1808);
    }
}
