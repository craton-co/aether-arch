//! Burrows-Wheeler Transform + Move-to-Front preprocessing.
//!
//! BWT rearranges data so that similar contexts cluster together.
//! MTF converts the clustered data into a stream of mostly small values
//! (0s and 1s), which entropy codes compress extremely well.
//!
//! BWT is computed via `libsais` (Ilya Grebnov's linear-time SA-IS
//! algorithm, via C FFI) using the **doubled-text trick**: we sort
//! suffixes of T+T and keep only positions < n to obtain the cyclic
//! rotation SA.  This is O(n) via SA-IS (vs O(n log n) with divsufsort),
//! with the same ~10n peak memory for the doubled text + suffix array.
//!
//! Output format: `[primary_index: u32 LE] [MTF data...]`
//! Total size = input size + 4 bytes.
//!
//! This achieves bzip2-class compression ratios on text data, typically
//! beating gzip by 10-20%.

use crate::error::{AetherError, Result};

/// Maximum decodable output size for BWT+MTF decode.
///
/// Aligned with [`crate::format::MAX_DECOMPRESSED_BLOCK_SIZE`] (64 MiB) to
/// guard against OOM from corrupted size fields in crafted archives.
const MAX_BWT_DECODE_SIZE: usize = crate::format::MAX_DECOMPRESSED_BLOCK_SIZE;

// ── BWT Constants ───────────────────────────────────────────────────────────

/// Maximum input size for BWT encoding (8 MiB).
///
/// The doubled-text trick allocates `2 * n` bytes for the text and
/// `libsais` allocates `~4 * 2n = 8n` bytes for the suffix array,
/// giving a total peak memory of roughly **10× the input size**.
///
/// With 8 MiB input → ~80 MiB peak per BWT call.  Since solid groups are
/// compressed in parallel (one rayon task per group), limiting BWT input
/// prevents OOM on memory-constrained targets.
///
/// The maximum FastCDC chunk size is 4 MiB (`MAX_CHUNK_SIZE`), so this
/// constant provides a 2× safety margin.  If chunk sizes are increased in
/// the future, this constant should be updated to match.
pub const MAX_BWT_INPUT_SIZE: usize = 8 * 1024 * 1024;

// ── BWT (Burrows-Wheeler Transform) ─────────────────────────────────────────

#[cfg(feature = "bwt-encode")]
/// Compute BWT using the doubled-text trick with `libsais` SA-IS algorithm.
///
/// We sort suffixes of T+T (text concatenated with itself) using `libsais`
/// (linear-time SA-IS) and keep only positions < n.  This produces a valid
/// cyclic rotation suffix array, from which we extract the BWT.
///
/// Returns (bwt_output, primary_index).
/// Time: O(n) (libsais SA-IS on 2n bytes). Memory: ~10n peak.
///
/// Returns `Err` for inputs exceeding [`MAX_BWT_INPUT_SIZE`].
fn bwt_encode(data: &[u8]) -> std::result::Result<(Vec<u8>, u32), &'static str> {
    let n = data.len();
    if n == 0 {
        return Ok((vec![], 0));
    }
    if n == 1 {
        return Ok((data.to_vec(), 0));
    }
    if n > MAX_BWT_INPUT_SIZE {
        return Err("BWT input exceeds MAX_BWT_INPUT_SIZE");
    }

    // Build T+T.  Positions 0..n in the suffix array of T+T are in the same
    // relative order as cyclic rotations of T (each suffix in T+T starting at
    // i < n has ≥ n characters = the full cyclic rotation at i, so ties in
    // the first n characters are impossible between distinct positions).
    let mut doubled = Vec::with_capacity(2 * n);
    doubled.extend_from_slice(data);
    doubled.extend_from_slice(data);

    // Build suffix array of T+T using libsais (linear-time SA-IS).
    let doubled_len = doubled.len();
    debug_assert!(doubled_len <= i32::MAX as usize, "doubled_len overflows i32 for libsais");
    let mut sa = vec![0i32; doubled_len];
    // SAFETY: libsais reads doubled_len bytes from doubled and writes
    // doubled_len i32 entries to sa.  Both buffers are correctly sized.
    let rc = unsafe {
        libsais_sys::libsais::libsais(
            doubled.as_ptr(),
            sa.as_mut_ptr(),
            doubled_len as i32,
            0,                    // fs: no extra space
            std::ptr::null_mut(), // freq: not needed
        )
    };
    if rc != 0 {
        return Err("libsais suffix array construction failed");
    }

    // Scan in sorted order, keep only positions < n to build the cyclic BWT.
    let mut bwt = Vec::with_capacity(n);
    let mut primary_index = 0u32;

    for &s in &sa {
        let pos = s as usize;
        if pos < n {
            if pos == 0 {
                primary_index = bwt.len() as u32;
            }
            // BWT[rank] = character just before this rotation (cyclically)
            bwt.push(data[(pos + n - 1) % n]);
            if bwt.len() == n {
                break; // All n rotations collected
            }
        }
    }

    Ok((bwt, primary_index))
}

/// Inverse BWT using LF-mapping.
fn bwt_decode(bwt: &[u8], primary_index: u32) -> std::result::Result<Vec<u8>, &'static str> {
    let n = bwt.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if primary_index as usize >= n {
        return Err("BWT primary index out of bounds");
    }

    // Count character frequencies
    let mut count = [0u32; 256];
    for &b in bwt {
        count[b as usize] += 1;
    }

    // Cumulative counts (C array)
    let mut cumul = [0u32; 256];
    let mut sum = 0u32;
    for i in 0..256 {
        cumul[i] = sum;
        sum += count[i];
    }

    // Build LF-mapping: LF[i] = C[bwt[i]] + rank of bwt[i] among equal chars before i
    let mut lf = vec![0u32; n];
    let mut occ = [0u32; 256];
    for i in 0..n {
        let c = bwt[i] as usize;
        lf[i] = cumul[c] + occ[c];
        occ[c] += 1;
    }

    // Reconstruct original string by following LF-mapping backwards
    let mut output = vec![0u8; n];
    let mut idx = primary_index as usize;
    for i in (0..n).rev() {
        if idx >= n {
            return Err("BWT LF-mapping out of bounds");
        }
        output[i] = bwt[idx];
        // L1 defense-in-depth: validate LF result before next iteration
        let next_idx = lf[idx] as usize;
        if next_idx >= n && i > 0 {
            return Err("BWT LF-mapping produced out-of-bounds index");
        }
        idx = next_idx;
    }

    Ok(output)
}

// ── MTF (Move-to-Front Transform) ──────────────────────────────────────────

/// Move-to-Front encode: converts byte stream into indices into a dynamic list.
/// Frequently repeated bytes get index 0 (most common after BWT).
///
/// Uses stack-allocated `[u8; 256]` arrays for cache efficiency:
/// - `pos[byte]` = current rank → O(1) lookup (no linear scan)
/// - Shift loop is O(rank), not O(256), so rank-0 symbols cost zero
fn mtf_encode(data: &[u8]) -> Vec<u8> {
    let mut list = [0u8; 256]; // list[rank] = byte value
    let mut pos = [0u8; 256]; // pos[byte] = current rank
    for i in 0..256usize {
        list[i] = i as u8;
        pos[i] = i as u8;
    }

    let mut output = Vec::with_capacity(data.len());
    for &byte in data {
        let rank = pos[byte as usize];
        output.push(rank);
        if rank > 0 {
            let r = rank as usize;
            // Shift list[0..r] → list[1..=r] and update pos for each moved byte.
            // Loop runs O(rank) times; for BWT output rank is usually 0-5.
            for i in (1..=r).rev() {
                let b = list[i - 1];
                list[i] = b;
                pos[b as usize] = i as u8;
            }
            list[0] = byte;
            pos[byte as usize] = 0;
        }
    }

    output
}

/// Inverse Move-to-Front: converts indices back into byte stream.
fn mtf_decode(data: &[u8]) -> Vec<u8> {
    let mut list = [0u8; 256]; // list[rank] = byte value
    for (i, item) in list.iter_mut().enumerate() {
        *item = i as u8;
    }

    let mut output = Vec::with_capacity(data.len());
    for &rank in data {
        let byte = list[rank as usize];
        output.push(byte);
        if rank > 0 {
            let r = rank as usize;
            for i in (1..=r).rev() {
                list[i] = list[i - 1];
            }
            list[0] = byte;
        }
    }

    output
}

// ── Zero-Run RLE (RUNA/RUNB, bzip2-style) ──────────────────────────────────

/// Apply bijective base-2 zero-run encoding to MTF data.
///
/// - MTF value 0 runs → RUNA(0)/RUNB(1) sequences (bijective base-2)
/// - MTF values 1-254 → shifted to 2-255
/// - Returns None if MTF data contains value 255 (can't be represented)
#[must_use]
pub fn rle_encode(mtf_data: &[u8]) -> Option<Vec<u8>> {
    // Check if representable (value 255 can't be shifted to 256)
    if mtf_data.contains(&255) {
        return None;
    }

    let mut output = Vec::with_capacity(mtf_data.len());
    let mut i = 0;

    while i < mtf_data.len() {
        if mtf_data[i] == 0 {
            // Count the zero run
            let mut run_len: u32 = 0;
            while i < mtf_data.len() && mtf_data[i] == 0 {
                run_len += 1;
                i += 1;
            }
            // Encode run_len using bijective base-2
            // RUNA=0 (digit value 1), RUNB=1 (digit value 2)
            encode_run_length(&mut output, run_len);
        } else {
            // Non-zero MTF value: shift by 1 (1→2, 2→3, ..., 254→255)
            output.push(mtf_data[i] + 1);
            i += 1;
        }
    }

    Some(output)
}

/// Encode a run length N using bijective base-2 numeration.
///
/// Digits: RUNA=0 (value 1), RUNB=1 (value 2).
/// Emitted least-significant-digit first.
fn encode_run_length(output: &mut Vec<u8>, mut n: u32) {
    // Bijective base-2: digits d where d ∈ {1,2}
    // n = d_0 * 2^0 + d_1 * 2^1 + ... + d_k * 2^k
    // RUNA = digit value 1 (symbol 0), RUNB = digit value 2 (symbol 1)
    while n > 0 {
        n -= 1;
        output.push((n & 1) as u8); // 0=RUNA, 1=RUNB
        n >>= 1;
    }
}

/// Decode bijective base-2 zero-run encoded data back to MTF stream.
pub fn rle_decode(rle_data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(expected_size);
    let mut i = 0;

    while i < rle_data.len() {
        if rle_data[i] <= 1 {
            // Start of a zero run — read RUNA/RUNB digits
            let mut run_len: u64 = 0;
            let mut power: u64 = 1;
            while i < rle_data.len() && rle_data[i] <= 1 {
                // digit value = rle_data[i] + 1 (RUNA=1, RUNB=2)
                run_len = run_len.saturating_add((rle_data[i] as u64 + 1).saturating_mul(power));
                power = power.saturating_mul(2);
                i += 1;
            }
            // H6 security fix: reject if run_len exceeds remaining capacity instead
            // of silently truncating (which could mask corrupted/malicious archives).
            let remaining = expected_size.saturating_sub(output.len()) as u64;
            if run_len > remaining {
                return Err(AetherError::Decompression(format!(
                    "RLE zero-run length {} exceeds remaining output capacity {}",
                    run_len, remaining,
                )));
            }
            let run_len = run_len as usize;
            output.resize(output.len() + run_len, 0);
        } else {
            // H7 security fix: check capacity *before* pushing to avoid
            // allocating one byte past expected_size.
            if output.len() >= expected_size {
                return Err(AetherError::Decompression(format!(
                    "RLE decode: non-zero symbol at position {} would exceed expected size {}",
                    i, expected_size,
                )));
            }
            // Shifted non-zero value: subtract 1 (2→1, 3→2, ..., 255→254)
            output.push(rle_data[i] - 1);
            i += 1;
        }
    }

    if output.len() != expected_size {
        return Err(AetherError::Decompression(format!(
            "RLE decode size mismatch: got {}, expected {expected_size}",
            output.len()
        )));
    }

    Ok(output)
}

// ── Combined API ────────────────────────────────────────────────────────────

#[cfg(feature = "bwt-encode")]
/// Apply BWT + MTF preprocessing.
///
/// Returns `Ok((primary_index, mtf_data))` or `Err` if the input exceeds
/// [`MAX_BWT_INPUT_SIZE`].
pub fn bwt_mtf_encode_parts(data: &[u8]) -> Result<(u32, Vec<u8>)> {
    let (bwt_data, primary_index) =
        bwt_encode(data).map_err(|e| AetherError::ResourceLimitExceeded(e.to_string()))?;
    let mtf_data = mtf_encode(&bwt_data);
    Ok((primary_index, mtf_data))
}

#[cfg(feature = "bwt-encode")]
/// Apply BWT + MTF preprocessing (legacy format).
///
/// Returns encoded bytes: `[primary_index: u32 LE] [MTF data...]`
/// Output size = input size + 4 bytes.
///
/// Returns `Err` if the input exceeds [`MAX_BWT_INPUT_SIZE`].
pub fn bwt_mtf_encode(data: &[u8]) -> Result<Vec<u8>> {
    let (primary_index, mtf_data) = bwt_mtf_encode_parts(data)?;
    let mut output = Vec::with_capacity(4 + mtf_data.len());
    output.extend_from_slice(&primary_index.to_le_bytes());
    output.extend_from_slice(&mtf_data);
    Ok(output)
}

/// Reverse BWT + MTF preprocessing.
///
/// Input: `[primary_index: u32 LE] [MTF data...]`
/// Returns original data.
pub fn bwt_mtf_decode(encoded: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    if expected_size > MAX_BWT_DECODE_SIZE {
        return Err(AetherError::Decompression(format!(
            "BWT+MTF decode size {} exceeds safety limit {}",
            expected_size, MAX_BWT_DECODE_SIZE,
        )));
    }
    if encoded.len() < 4 {
        return Err(AetherError::Decompression(
            "BWT+MTF data too short for primary index".into(),
        ));
    }

    let primary_index = u32::from_le_bytes(encoded[..4].try_into().map_err(|_| {
        AetherError::Decompression("BWT+MTF data too short for primary index".into())
    })?);
    let mtf_data = &encoded[4..];

    if mtf_data.len() != expected_size {
        return Err(AetherError::Decompression(format!(
            "BWT+MTF size mismatch: got {}, expected {expected_size}",
            mtf_data.len()
        )));
    }

    bwt_mtf_decode_parts(primary_index, mtf_data, expected_size)
}

/// Reverse BWT + MTF from separate primary_index and mtf_data.
pub fn bwt_mtf_decode_parts(
    primary_index: u32,
    mtf_data: &[u8],
    expected_size: usize,
) -> Result<Vec<u8>> {
    if expected_size > MAX_BWT_DECODE_SIZE {
        return Err(AetherError::Decompression(format!(
            "BWT+MTF decode size {} exceeds safety limit {}",
            expected_size, MAX_BWT_DECODE_SIZE,
        )));
    }
    let bwt_data = mtf_decode(mtf_data);
    let original = bwt_decode(&bwt_data, primary_index)
        .map_err(|e| AetherError::Decompression(format!("BWT decode failed: {e}")))?;

    if original.len() != expected_size {
        return Err(AetherError::Decompression(format!(
            "BWT+MTF decode size mismatch: got {}, expected {expected_size}",
            original.len()
        )));
    }

    Ok(original)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bwt_roundtrip_simple() {
        let data = b"banana";
        let (bwt, idx) = bwt_encode(data).unwrap();
        let decoded = bwt_decode(&bwt, idx).unwrap();
        assert_eq!(&decoded, data);
    }

    #[test]
    fn bwt_roundtrip_text() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let (bwt, idx) = bwt_encode(data).unwrap();
        let decoded = bwt_decode(&bwt, idx).unwrap();
        assert_eq!(&decoded, data);
    }

    #[test]
    fn mtf_roundtrip() {
        let data = b"abracadabra";
        let encoded = mtf_encode(data);
        let decoded = mtf_decode(&encoded);
        assert_eq!(&decoded, data);
    }

    #[test]
    fn mtf_clusters_repeated_bytes() {
        // BWT output typically has runs of same byte → MTF gives lots of 0s
        let data = vec![b'a'; 100];
        let encoded = mtf_encode(&data);
        // First byte gets some index, all subsequent should be 0
        assert!(encoded[1..].iter().all(|&b| b == 0));
    }

    #[test]
    fn combined_roundtrip_text() {
        let text = b"Hello, world! Hello, world! This is a BWT test. \
                     The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog.";
        let encoded = bwt_mtf_encode(text).unwrap();
        let decoded = bwt_mtf_decode(&encoded, text.len()).unwrap();
        assert_eq!(&decoded[..], &text[..]);
    }

    #[test]
    fn combined_roundtrip_repetitive() {
        let text: Vec<u8> = b"ABCABCABCABCABC".repeat(100);
        let encoded = bwt_mtf_encode(&text).unwrap();
        let decoded = bwt_mtf_decode(&encoded, text.len()).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn mtf_output_mostly_small() {
        // Longer text after BWT should produce mostly small MTF values
        let line = b"the quick brown fox jumps over the lazy dog again and again. ";
        let text: Vec<u8> = line.repeat(20);
        let (bwt, _) = bwt_encode(&text).unwrap();
        let mtf = mtf_encode(&bwt);

        let small_count = mtf.iter().filter(|&&b| b < 10).count();
        let ratio = small_count as f64 / mtf.len() as f64;
        assert!(
            ratio > 0.5,
            "Expected >50% small MTF values, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn decode_validates_size() {
        let text = b"test data for size validation";
        let encoded = bwt_mtf_encode(text).unwrap();
        let err = bwt_mtf_decode(&encoded, text.len() + 10);
        assert!(err.is_err());
    }

    #[test]
    fn single_byte_roundtrip() {
        let data = b"X";
        let encoded = bwt_mtf_encode(data).unwrap();
        let decoded = bwt_mtf_decode(&encoded, data.len()).unwrap();
        assert_eq!(&decoded[..], &data[..]);
    }

    #[test]
    fn uniform_data_roundtrip() {
        let data = vec![0x42u8; 10_000];
        let encoded = bwt_mtf_encode(&data).unwrap();
        let decoded = bwt_mtf_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
    }

    // ── RLE tests ─────────────────────────────────────────────────────

    #[test]
    fn rle_roundtrip_basic() {
        // Simple MTF-like data: runs of zeros interspersed with small values
        let mtf = vec![0, 0, 0, 0, 0, 3, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1];
        let rle = rle_encode(&mtf).unwrap();
        let decoded = rle_decode(&rle, mtf.len()).unwrap();
        assert_eq!(decoded, mtf);
    }

    #[test]
    fn rle_roundtrip_single_zeros() {
        let mtf = vec![0, 5, 0, 3, 0];
        let rle = rle_encode(&mtf).unwrap();
        let decoded = rle_decode(&rle, mtf.len()).unwrap();
        assert_eq!(decoded, mtf);
    }

    #[test]
    fn rle_roundtrip_no_zeros() {
        let mtf = vec![1, 2, 3, 4, 5];
        let rle = rle_encode(&mtf).unwrap();
        // No zeros → all values shifted by 1
        assert_eq!(rle, vec![2, 3, 4, 5, 6]);
        let decoded = rle_decode(&rle, mtf.len()).unwrap();
        assert_eq!(decoded, mtf);
    }

    #[test]
    fn rle_roundtrip_all_zeros() {
        let mtf = vec![0; 1000];
        let rle = rle_encode(&mtf).unwrap();
        // Run of 1000 zeros → bijective base-2 of 1000 → ~10 symbols
        assert!(
            rle.len() < 15,
            "RLE of 1000 zeros should be < 15 symbols, got {}",
            rle.len()
        );
        let decoded = rle_decode(&rle, mtf.len()).unwrap();
        assert_eq!(decoded, mtf);
    }

    #[test]
    fn rle_rejects_value_255() {
        let mtf = vec![0, 1, 255, 3];
        assert!(rle_encode(&mtf).is_none());
    }

    #[test]
    fn rle_run_lengths() {
        // Verify specific run length encodings
        for n in 1..=100u32 {
            let mtf = vec![0u8; n as usize];
            let rle = rle_encode(&mtf).unwrap();
            let decoded = rle_decode(&rle, mtf.len()).unwrap();
            assert_eq!(decoded, mtf, "Failed roundtrip for run length {n}");
        }
    }

    #[test]
    fn rle_compresses_real_bwt_output() {
        // BWT+MTF of repetitive text → lots of zero runs
        let line = b"the quick brown fox jumps over the lazy dog again and again. ";
        let text: Vec<u8> = line.repeat(20);
        let (_, mtf_data) = bwt_mtf_encode_parts(&text).unwrap();
        let rle = rle_encode(&mtf_data).unwrap();
        // RLE should significantly reduce the data size
        let ratio = rle.len() as f64 / mtf_data.len() as f64;
        assert!(
            ratio < 0.8,
            "RLE should reduce MTF data by >20%, got ratio {:.2}",
            ratio
        );
    }

    #[test]
    fn rle_full_pipeline_roundtrip() {
        // Full BWT+MTF+RLE roundtrip
        let text = b"Hello, world! Hello, world! This is a BWT+RLE test. \
                     The quick brown fox jumps over the lazy dog.";
        let (primary_index, mtf_data) = bwt_mtf_encode_parts(text).unwrap();
        let rle = rle_encode(&mtf_data).unwrap();
        let mtf_decoded = rle_decode(&rle, mtf_data.len()).unwrap();
        let original = bwt_mtf_decode_parts(primary_index, &mtf_decoded, text.len()).unwrap();
        assert_eq!(&original[..], &text[..]);
    }
}
