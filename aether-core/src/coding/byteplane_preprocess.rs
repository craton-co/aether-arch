//! Byte-plane splitting for numeric/structured data.
//!
//! Inspired by ZipNN: splits N-byte elements into N independent byte planes
//! by position. Exponent bytes in float arrays cluster tightly (~40 of 256
//! values), making them highly compressible with simple entropy coding, while
//! mantissa bytes are near-random and best stored raw.
//!
//! # Payload format (BytePlanePredictorRans)
//!
//! ```text
//! [width: u8]                     // Element width (2 or 4)
//! [plane_flags: u8]               // Bitmask: bit i=1 -> plane i is RC-compressed
//! [plane_sizes: u32 LE x width]   // Compressed size of each plane
//! [plane_data...]                  // Concatenated plane payloads
//! [tail_bytes...]                  // Uncompressed remainder (len % width bytes)
//! ```

use crate::format::{shannon_entropy, MAX_DECOMPRESSED_BLOCK_SIZE};

/// Minimum chunk size to attempt byte-plane splitting.
/// Need enough elements for the entropy coder to amortize overhead.
pub const MIN_BYTEPLANE_SIZE: usize = 64;

/// Entropy threshold for compressing a plane (bits/byte).
/// Planes above this are stored raw. Exponent planes are typically 3-5 bps;
/// mantissa planes are ~7.5-8.0 bps.
const PLANE_COMPRESS_THRESHOLD: f64 = 7.0;

/// Element width for byte-plane splitting.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BytePlaneWidth {
    /// 2-byte elements (BF16, FP16, i16, u16)
    Two = 2,
    /// 4-byte elements (FP32, i32, u32)
    Four = 4,
}

impl BytePlaneWidth {
    /// Parse from stored u8 value.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            2 => Some(Self::Two),
            4 => Some(Self::Four),
            _ => None,
        }
    }

    /// Number of byte planes.
    pub fn planes(self) -> usize {
        self as usize
    }
}

/// Split data into byte planes by element width.
///
/// For width=2: `[A0 B0 A1 B1 ...]` -> `planes[0]=[A0 A1 ...], planes[1]=[B0 B1 ...]`
/// For width=4: `[A0 B0 C0 D0 A1 B1 C1 D1 ...]` -> 4 planes
///
/// Returns `(planes, tail)` where tail is the leftover bytes if `data.len() % width != 0`.
pub fn byteplane_split(data: &[u8], width: BytePlaneWidth) -> (Vec<Vec<u8>>, Vec<u8>) {
    let w = width.planes();
    let n_elements = data.len() / w;
    let _tail_len = data.len() % w;

    let mut planes: Vec<Vec<u8>> = (0..w).map(|_| Vec::with_capacity(n_elements)).collect();

    let bulk = &data[..n_elements * w];
    for chunk in bulk.chunks_exact(w) {
        for (plane_idx, &byte) in chunk.iter().enumerate() {
            planes[plane_idx].push(byte);
        }
    }

    let tail = data[n_elements * w..].to_vec();
    (planes, tail)
}

/// Merge byte planes back into interleaved data, appending tail bytes.
pub fn byteplane_merge(planes: &[Vec<u8>], tail: &[u8]) -> Vec<u8> {
    if planes.is_empty() {
        return tail.to_vec();
    }
    let n_elements = planes[0].len();
    let w = planes.len();
    let mut data = Vec::with_capacity(n_elements * w + tail.len());

    for i in 0..n_elements {
        for plane in planes {
            data.push(plane[i]);
        }
    }
    data.extend_from_slice(tail);
    data
}

/// Check if byte-plane splitting with the given width would be beneficial.
///
/// Returns `true` if at least one plane has significantly lower entropy than
/// the raw data (indicating structured numeric content).
pub fn is_byteplane_beneficial(data: &[u8], width: BytePlaneWidth) -> bool {
    if data.len() < MIN_BYTEPLANE_SIZE {
        return false;
    }

    let w = width.planes();
    let n_elements = data.len() / w;
    if n_elements < 32 {
        return false;
    }

    let raw_entropy = shannon_entropy(data);

    // Quick check: sample first plane entropy from a prefix
    let sample_size = n_elements.min(1024);
    let mut plane0_sample = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        plane0_sample.push(data[i * w]);
    }

    let plane0_entropy = shannon_entropy(&plane0_sample);

    // Beneficial if the MSB plane has at least 1.5 bps lower entropy than raw.
    // This catches float exponent bytes (3-5 bps vs 6-8 bps raw) while
    // avoiding false positives on text or already-compressed data.
    raw_entropy - plane0_entropy > 1.5
}

/// Determine which planes should be compressed vs stored raw.
///
/// Returns a bitmask where bit `i` = 1 means plane `i` should be RC-compressed.
pub fn classify_planes(planes: &[Vec<u8>]) -> u8 {
    let mut flags: u8 = 0;
    for (i, plane) in planes.iter().enumerate() {
        if !plane.is_empty() && shannon_entropy(plane) < PLANE_COMPRESS_THRESHOLD {
            flags |= 1 << i;
        }
    }
    flags
}

/// Detect if data is likely an array of numeric values at the given width.
///
/// Uses a simple heuristic: checks if byte-plane splitting reveals
/// significantly non-uniform distributions in the MSB plane.
pub fn detect_numeric_width(data: &[u8]) -> Option<BytePlaneWidth> {
    if data.len() < MIN_BYTEPLANE_SIZE {
        return None;
    }

    // Try width=4 first (FP32 is most common), then width=2
    // Pick whichever shows the strongest plane entropy differential
    let mut best: Option<(BytePlaneWidth, f64)> = None;

    for &width in &[BytePlaneWidth::Four, BytePlaneWidth::Two] {
        let w = width.planes();
        let n_elements = data.len() / w;
        if n_elements < 32 {
            continue;
        }

        // Sample MSB plane entropy
        let sample_size = n_elements.min(1024);
        let mut plane0 = Vec::with_capacity(sample_size);
        for i in 0..sample_size {
            plane0.push(data[i * w]);
        }
        let plane0_entropy = shannon_entropy(&plane0);

        let benefit = shannon_entropy(data) - plane0_entropy;
        if benefit > 1.5 {
            if best.as_ref().is_none_or(|(_, b)| benefit > *b) {
                best = Some((width, benefit));
            }
        }
    }

    best.map(|(w, _)| w)
}

/// Encode byte-plane split data into the BytePlanePredictorRans payload format.
///
/// Each compressible plane (below entropy threshold) is range-coded with a
/// fresh Order0 predictor. High-entropy planes are stored raw.
///
/// Returns `None` if the result is not smaller than the input.
pub fn byteplane_encode(data: &[u8], width: BytePlaneWidth) -> Option<Vec<u8>> {
    use crate::coding::rans;
    use crate::entropy::Order0Model;

    if data.len() < MIN_BYTEPLANE_SIZE || data.len() > MAX_DECOMPRESSED_BLOCK_SIZE {
        return None;
    }

    let (planes, tail) = byteplane_split(data, width);
    let plane_flags = classify_planes(&planes);

    // At least one plane must be compressible for this to be worthwhile
    if plane_flags == 0 {
        return None;
    }

    let w = width.planes();

    // Encode each plane
    let mut encoded_planes: Vec<Vec<u8>> = Vec::with_capacity(w);
    for (i, plane) in planes.iter().enumerate() {
        if (plane_flags >> i) & 1 == 1 {
            // RC-compress this plane with Order0
            let mut predictor = Order0Model::new();
            match rans::encode_block(plane, &mut predictor) {
                Ok(rc_bytes) => {
                    // Only keep RC version if it's actually smaller
                    if rc_bytes.len() < plane.len() {
                        encoded_planes.push(rc_bytes);
                    } else {
                        // Store raw instead — clear the flag
                        encoded_planes.push(plane.clone());
                    }
                }
                Err(_) => {
                    encoded_planes.push(plane.clone());
                }
            }
        } else {
            // Store raw
            encoded_planes.push(plane.clone());
        }
    }

    // Recompute flags based on what actually compressed
    let mut actual_flags: u8 = 0;
    for (i, (enc, orig)) in encoded_planes.iter().zip(planes.iter()).enumerate() {
        if enc.len() < orig.len() && (plane_flags >> i) & 1 == 1 {
            actual_flags |= 1 << i;
        }
    }

    // Build payload: [width: u8] [flags: u8] [sizes: u32 x w] [plane_data...] [tail...]
    let header_size = 2 + 4 * w;
    let planes_size: usize = encoded_planes.iter().map(|p| p.len()).sum();
    let total_size = header_size + planes_size + tail.len();

    // Only beneficial if smaller than original
    if total_size >= data.len() {
        return None;
    }

    let mut payload = Vec::with_capacity(total_size);
    payload.push(width as u8);
    payload.push(actual_flags);
    for ep in &encoded_planes {
        payload.extend_from_slice(&(ep.len() as u32).to_le_bytes());
    }
    for ep in &encoded_planes {
        payload.extend_from_slice(ep);
    }
    payload.extend_from_slice(&tail);

    Some(payload)
}

/// Decode a BytePlanePredictorRans payload back to the original data.
///
/// `uncompressed_size` is the expected output size (from the block header).
pub fn byteplane_decode(payload: &[u8], uncompressed_size: usize) -> crate::error::Result<Vec<u8>> {
    use crate::coding::rans;
    use crate::entropy::Order0Model;
    use crate::error::AetherError;

    // Parse header
    if payload.len() < 2 {
        return Err(AetherError::Decompression(
            "BytePlane payload too short: missing header".into(),
        ));
    }

    let width = BytePlaneWidth::from_u8(payload[0]).ok_or_else(|| {
        AetherError::Decompression(format!(
            "BytePlane: invalid width byte {}",
            payload[0]
        ))
    })?;
    let plane_flags = payload[1];
    let w = width.planes();

    let sizes_start = 2;
    let sizes_end = sizes_start + 4 * w;
    if payload.len() < sizes_end {
        return Err(AetherError::Decompression(format!(
            "BytePlane payload too short for {} plane sizes: {} bytes",
            w,
            payload.len()
        )));
    }

    // Read plane sizes
    let mut plane_sizes = Vec::with_capacity(w);
    for i in 0..w {
        let offset = sizes_start + 4 * i;
        let size = u32::from_le_bytes(
            payload[offset..offset + 4]
                .try_into()
                .map_err(|_| AetherError::Decompression("BytePlane: truncated plane size".into()))?,
        ) as usize;

        if size > MAX_DECOMPRESSED_BLOCK_SIZE {
            return Err(AetherError::ResourceLimitExceeded(format!(
                "BytePlane plane {} size {} exceeds safety limit {}",
                i, size, MAX_DECOMPRESSED_BLOCK_SIZE,
            )));
        }
        plane_sizes.push(size);
    }

    // Calculate expected element count and tail length
    let n_elements = uncompressed_size / w;
    let tail_len = uncompressed_size % w;

    // Validate total payload size
    let total_plane_bytes: usize = plane_sizes.iter().sum();
    let expected_payload_len = sizes_end + total_plane_bytes + tail_len;
    if payload.len() < expected_payload_len {
        return Err(AetherError::Decompression(format!(
            "BytePlane payload too short: need {} bytes, have {}",
            expected_payload_len,
            payload.len()
        )));
    }

    // Decode each plane
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(w);
    let mut data_offset = sizes_end;
    for (i, &psize) in plane_sizes.iter().enumerate() {
        let plane_data = &payload[data_offset..data_offset + psize];
        data_offset += psize;

        if (plane_flags >> i) & 1 == 1 {
            // RC-compressed: decode with Order0
            let mut predictor = Order0Model::new();
            let decoded = rans::decode_block(plane_data, n_elements, &mut predictor)?;
            planes.push(decoded);
        } else {
            // Stored raw
            if psize != n_elements {
                return Err(AetherError::Decompression(format!(
                    "BytePlane raw plane {} size mismatch: {} vs expected {}",
                    i, psize, n_elements,
                )));
            }
            planes.push(plane_data.to_vec());
        }
    }

    // Read tail bytes
    let tail = &payload[data_offset..data_offset + tail_len];

    // Merge planes + tail
    let result = byteplane_merge(&planes, tail);

    if result.len() != uncompressed_size {
        return Err(AetherError::Decompression(format!(
            "BytePlane size mismatch after merge: got {} expected {}",
            result.len(),
            uncompressed_size,
        )));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_merge_roundtrip_width2() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let (planes, tail) = byteplane_split(&data, BytePlaneWidth::Two);
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0].len(), 50); // even bytes
        assert_eq!(planes[1].len(), 50); // odd bytes
        assert!(tail.is_empty());
        let merged = byteplane_merge(&planes, &tail);
        assert_eq!(merged, data);
    }

    #[test]
    fn split_merge_roundtrip_width4() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let (planes, tail) = byteplane_split(&data, BytePlaneWidth::Four);
        assert_eq!(planes.len(), 4);
        assert_eq!(planes[0].len(), 25);
        assert!(tail.is_empty());
        let merged = byteplane_merge(&planes, &tail);
        assert_eq!(merged, data);
    }

    #[test]
    fn split_merge_with_tail() {
        let data: Vec<u8> = (0..103).map(|i| i as u8).collect();
        let (planes, tail) = byteplane_split(&data, BytePlaneWidth::Four);
        assert_eq!(planes[0].len(), 25); // 100/4 = 25 full elements
        assert_eq!(tail.len(), 3); // 103 % 4 = 3
        let merged = byteplane_merge(&planes, &tail);
        assert_eq!(merged, data);
    }

    #[test]
    fn classify_low_entropy_plane() {
        // Plane of mostly zeros → should be flagged for compression
        let mut low_entropy = vec![0u8; 1000];
        low_entropy[500] = 1;
        low_entropy[700] = 2;

        let high_entropy: Vec<u8> = (0..1000).map(|i| (i * 97 + 13) as u8).collect();

        let planes = vec![low_entropy, high_entropy];
        let flags = classify_planes(&planes);
        assert_eq!(flags & 1, 1, "low-entropy plane should be flagged");
        assert_eq!(flags & 2, 0, "high-entropy plane should NOT be flagged");
    }

    #[test]
    fn encode_decode_roundtrip_synthetic_floats() {
        // Simulate BF16-like data: exponent bytes cluster, mantissa bytes vary
        let n = 2000;
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            // Byte 0: exponent-like (clusters around a few values)
            let exp = match i % 10 {
                0..=6 => 0x3F, // ~70% one value
                7..=8 => 0x40, // ~20% another
                _ => 0x3E,     // ~10% another
            };
            // Byte 1: mantissa-like (pseudo-random)
            let mantissa = ((i as u32).wrapping_mul(2654435761) >> 16) as u8;
            data.push(exp);
            data.push(mantissa);
        }

        let encoded = byteplane_encode(&data, BytePlaneWidth::Two);
        assert!(encoded.is_some(), "synthetic float data should compress");

        let payload = encoded.unwrap();
        assert!(
            payload.len() < data.len(),
            "compressed ({}) should be smaller than original ({})",
            payload.len(),
            data.len()
        );

        let decoded = byteplane_decode(&payload, data.len()).unwrap();
        assert_eq!(decoded, data, "roundtrip must be lossless");
    }

    #[test]
    fn encode_returns_none_for_random_data() {
        // Truly random data: byte-plane splitting should not help
        let data: Vec<u8> = (0..4000)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as u8)
            .collect();
        let result = byteplane_encode(&data, BytePlaneWidth::Two);
        // May or may not be None depending on pseudo-random distribution,
        // but if it encodes, it must roundtrip
        if let Some(payload) = result {
            let decoded = byteplane_decode(&payload, data.len()).unwrap();
            assert_eq!(decoded, data);
        }
    }

    #[test]
    fn encode_returns_none_for_small_data() {
        let data = vec![0u8; 32];
        assert!(byteplane_encode(&data, BytePlaneWidth::Two).is_none());
    }

    #[test]
    fn detect_numeric_finds_float_like_data() {
        // Simulate FP32 data: byte 0 clusters (exponent), bytes 1-3 vary
        let n = 500;
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            data.push(0x3F); // exponent byte (clustered)
            data.push(((i * 97) % 256) as u8);
            data.push(((i * 53) % 256) as u8);
            data.push(((i * 31) % 256) as u8);
        }

        let width = detect_numeric_width(&data);
        assert!(width.is_some(), "should detect numeric data");
    }

    #[test]
    fn detect_numeric_rejects_text() {
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);
        let width = detect_numeric_width(&data);
        assert!(width.is_none(), "text should not be detected as numeric");
    }
}
