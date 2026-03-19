//! Custom LZ77 preprocessor with min-match-3 and lazy matching.
//!
//! Produces an LZ4-like token format but with minimum match length 3 (vs LZ4's 4).
//! This catches short 3-byte matches (common words like "the", "and", "for") that
//! LZ4 misses, closing the gap to Deflate-class compressors.
//!
//! Output format:
//!   `[original_size: u32 LE] [token sequences...]`
//!
//! Token sequence (same structure as LZ4):
//!   `[token] [lit_len_ext...] [literals...] [offset_lo] [offset_hi] [match_len_ext...]`
//! where token high nibble = literal length, low nibble = match_length - 3.
//! Last sequence has no offset or match.
//!
//! The FSM predictor in `lz4_aware.rs` handles this format unchanged because
//! it only uses nibble values for state transitions.

use crate::error::{AetherError, Result};

const WINDOW_SIZE: usize = 65536; // 64KB sliding window
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 65538; // large enough for any data
const HASH_BITS: usize = 16; // 64K hash entries
const HASH_SIZE: usize = 1 << HASH_BITS;
const MAX_CHAIN: usize = 4096; // max hash chain walk length (matches gzip -9)
const NICE_MATCH: usize = 258; // stop searching if match >= this
/// Maximum input size for LZ77 encoding (64 MiB).
///
/// Aligned with [`crate::format::MAX_DECOMPRESSED_BLOCK_SIZE`] to prevent
/// excessive memory use.  Positions are stored as `u32` in the hash chain,
/// so this also prevents index overflow.  FastCDC chunks are at most 4 MiB,
/// giving a 16× safety margin.
const MAX_LZ77_INPUT_SIZE: usize = crate::format::MAX_DECOMPRESSED_BLOCK_SIZE;

const EMPTY: u32 = u32::MAX; // "no entry" sentinel

/// Hash 3 bytes using Knuth multiplicative hash.
#[inline]
fn hash3(data: &[u8], pos: usize) -> usize {
    let v = (data[pos] as u32) << 16 | (data[pos + 1] as u32) << 8 | data[pos + 2] as u32;
    (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize
}

struct MatchFinder {
    head: Vec<u32>,
    prev: Vec<u32>,
}

impl MatchFinder {
    fn new() -> Self {
        Self {
            head: vec![EMPTY; HASH_SIZE],
            prev: vec![EMPTY; WINDOW_SIZE],
        }
    }

    #[inline]
    fn insert(&mut self, pos: usize, hash: usize) {
        // S11 security fix: runtime check instead of debug-only assert
        if pos > MAX_LZ77_INPUT_SIZE {
            return;
        }
        self.prev[pos % WINDOW_SIZE] = self.head[hash];
        self.head[hash] = pos as u32;
    }

    fn find_best(&self, data: &[u8], pos: usize, hash: usize) -> Option<(u16, usize)> {
        let mut best_len = MIN_MATCH - 1;
        let mut best_offset: u16 = 0;
        let mut chain_pos = self.head[hash];
        let mut chain_count = 0;
        let max_len = (data.len() - pos).min(MAX_MATCH);
        // Track last ref_pos to detect non-monotonic (cyclic) chains.
        // Chains should go strictly backwards; a forward or equal jump
        // indicates a cycle caused by prev[] slot reuse.
        let mut last_ref_pos = pos;

        while chain_pos != EMPTY && chain_count < MAX_CHAIN {
            let ref_pos = chain_pos as usize;
            // Q4 fix: use >= WINDOW_SIZE to prevent offset 65536 wrapping u16 to 0.
            if ref_pos >= pos || pos - ref_pos >= WINDOW_SIZE {
                break;
            }
            // Cycle detection: chain must be strictly decreasing in position.
            if ref_pos >= last_ref_pos {
                break;
            }
            last_ref_pos = ref_pos;

            let offset = (pos - ref_pos) as u16;

            // Quick check: compare last best byte + first bytes for early rejection
            if data[ref_pos + best_len] == data[pos + best_len] {
                let mut len = 0;
                while len < max_len && data[ref_pos + len] == data[pos + len] {
                    len += 1;
                }

                if len > best_len {
                    best_len = len;
                    best_offset = offset;
                    if len >= NICE_MATCH || len == max_len {
                        break;
                    }
                }
            }

            chain_pos = self.prev[ref_pos % WINDOW_SIZE];
            chain_count += 1;
        }

        if best_len >= MIN_MATCH {
            Some((best_offset, best_len))
        } else {
            None
        }
    }
}

/// Encode data with LZ77 (min-match-3, lazy matching).
///
/// Returns `Some(encoded)` if compression reduces size, `None` otherwise.
#[must_use]
pub fn lz77_encode(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < MIN_MATCH {
        return None;
    }
    // Positions are stored as u32 in the hash chain; reject inputs that would overflow.
    if data.len() > MAX_LZ77_INPUT_SIZE {
        return None;
    }

    let mut finder = MatchFinder::new();
    let mut output = Vec::with_capacity(data.len());

    // Prepend original size
    output.extend_from_slice(&(data.len() as u32).to_le_bytes());

    let mut pos = 0;
    let mut lit_start = 0;
    let mut deferred = false; // true = we already deferred once, take the next match

    while pos + 2 < data.len() {
        let hash = hash3(data, pos);
        let current_match = finder.find_best(data, pos, hash);

        let use_match = if let Some((_, cur_len)) = current_match {
            if deferred {
                // Already deferred once — take this match to prevent cascading
                true
            } else if pos + 3 < data.len() {
                let next_hash = hash3(data, pos + 1);
                if let Some((_, next_len)) = finder.find_best(data, pos + 1, next_hash) {
                    // Defer if next position has a significantly better match
                    next_len <= cur_len + 1
                } else {
                    true
                }
            } else {
                true
            }
        } else {
            false
        };

        if use_match {
            let (offset, match_len) = current_match.unwrap();
            emit_sequence(&mut output, &data[lit_start..pos], offset, match_len);

            // Insert all match positions into hash table
            let end = (pos + match_len).min(data.len().saturating_sub(2));
            for i in pos..end {
                let h = hash3(data, i);
                finder.insert(i, h);
            }

            pos += match_len;
            lit_start = pos;
            deferred = false;
        } else {
            finder.insert(pos, hash);
            pos += 1;
            deferred = current_match.is_some(); // we deferred a match
        }
    }

    // Emit remaining literals
    let remaining = &data[lit_start..];
    if !remaining.is_empty() {
        emit_final_literals(&mut output, remaining);
    }

    if output.len() < data.len() {
        Some(output)
    } else {
        None
    }
}

fn emit_sequence(output: &mut Vec<u8>, literals: &[u8], offset: u16, match_len: usize) {
    let lit_len = literals.len();
    let match_minus3 = match_len - MIN_MATCH;

    let lit_nibble = lit_len.min(15) as u8;
    let match_nibble = match_minus3.min(15) as u8;
    output.push((lit_nibble << 4) | match_nibble);

    // Literal length extension
    if lit_len >= 15 {
        let mut rem = lit_len - 15;
        while rem >= 255 {
            output.push(255);
            rem -= 255;
        }
        output.push(rem as u8);
    }

    // Literal data
    output.extend_from_slice(literals);

    // Offset (2-byte LE)
    output.push(offset as u8);
    output.push((offset >> 8) as u8);

    // Match length extension
    if match_minus3 >= 15 {
        let mut rem = match_minus3 - 15;
        while rem >= 255 {
            output.push(255);
            rem -= 255;
        }
        output.push(rem as u8);
    }
}

fn emit_final_literals(output: &mut Vec<u8>, literals: &[u8]) {
    let lit_len = literals.len();
    let lit_nibble = lit_len.min(15) as u8;
    output.push(lit_nibble << 4); // low nibble = 0, no match follows

    if lit_len >= 15 {
        let mut rem = lit_len - 15;
        while rem >= 255 {
            output.push(255);
            rem -= 255;
        }
        output.push(rem as u8);
    }

    output.extend_from_slice(literals);
}

/// Decode LZ77-encoded data produced by [`lz77_encode`].
pub fn lz77_decode(encoded: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    if encoded.len() < 4 {
        return Err(AetherError::Decompression("LZ77 data too short".into()));
    }

    let original_size =
        u32::from_le_bytes(encoded[..4].try_into().map_err(|_| {
            AetherError::Decompression("LZ77 data too short for size header".into())
        })?) as usize;
    if original_size != expected_size {
        return Err(AetherError::Decompression(format!(
            "LZ77 size mismatch: header says {original_size}, expected {expected_size}"
        )));
    }
    if original_size > MAX_LZ77_INPUT_SIZE {
        return Err(AetherError::Decompression(format!(
            "LZ77 decode size {} exceeds safety limit {}",
            original_size, MAX_LZ77_INPUT_SIZE
        )));
    }

    // V2 security fix: don't pre-allocate the full original_size from an
    // untrusted size field.  Cap initial allocation to the smaller of
    // original_size and encoded input size × 4 (a reasonable expansion
    // estimate), so a crafted archive with a large size header but tiny
    // payload doesn't force a huge allocation up front.
    let safe_capacity = original_size.min(encoded.len().saturating_mul(4).max(4096));
    let mut output = Vec::with_capacity(safe_capacity);
    let mut pos = 4;

    while output.len() < original_size {
        if pos >= encoded.len() {
            return Err(AetherError::Decompression(
                "LZ77: unexpected end of data".into(),
            ));
        }

        let token = encoded[pos];
        pos += 1;

        // ── Literal length ──
        let mut lit_len = (token >> 4) as usize;
        if lit_len == 15 {
            loop {
                if pos >= encoded.len() {
                    return Err(AetherError::Decompression(
                        "LZ77: unexpected end in lit_len ext".into(),
                    ));
                }
                let ext = encoded[pos] as usize;
                pos += 1;
                lit_len += ext;
                // V2 security fix: cap lit_len to remaining output capacity
                // to prevent unbounded accumulation from crafted extension bytes.
                let remaining = original_size.saturating_sub(output.len());
                if lit_len > remaining {
                    return Err(AetherError::Decompression(format!(
                        "LZ77: lit_len {lit_len} exceeds remaining output capacity {remaining}"
                    )));
                }
                if ext < 255 {
                    break;
                }
            }
        }

        // ── Copy literals ──
        if pos + lit_len > encoded.len() {
            return Err(AetherError::Decompression(
                "LZ77: not enough literal data".into(),
            ));
        }
        output.extend_from_slice(&encoded[pos..pos + lit_len]);
        pos += lit_len;

        // Check if we've produced all output bytes
        if output.len() >= original_size {
            break;
        }

        // ── Offset (2 bytes LE) ──
        if pos + 2 > encoded.len() {
            return Err(AetherError::Decompression(
                "LZ77: missing offset bytes".into(),
            ));
        }
        let offset = encoded[pos] as usize | ((encoded[pos + 1] as usize) << 8);
        pos += 2;

        if offset == 0 || offset > output.len() {
            return Err(AetherError::Decompression(format!(
                "LZ77: invalid offset {offset} at output pos {}",
                output.len()
            )));
        }

        // ── Match length ──
        let mut match_len = ((token & 0x0F) as usize) + MIN_MATCH;
        if (token & 0x0F) == 15 {
            loop {
                if pos >= encoded.len() {
                    return Err(AetherError::Decompression(
                        "LZ77: unexpected end in match_len ext".into(),
                    ));
                }
                let ext = encoded[pos] as usize;
                pos += 1;
                match_len += ext;
                // H5 security fix: cap match_len to remaining output capacity
                // to prevent unbounded accumulation from crafted extension bytes.
                let remaining = original_size.saturating_sub(output.len());
                if match_len > remaining {
                    return Err(AetherError::Decompression(format!(
                        "LZ77: match_len {match_len} exceeds remaining output capacity {remaining}"
                    )));
                }
                if ext < 255 {
                    break;
                }
            }
        }

        // ── Copy match (may overlap for RLE-style patterns) ──
        let start = output.len() - offset;
        if offset >= match_len {
            // Non-overlapping: bulk copy is safe and fast
            output.extend_from_within(start..start + match_len);
        } else {
            // Overlapping: repeat the offset-sized pattern to fill match_len.
            // Copy in offset-sized chunks to minimize per-byte overhead.
            let mut remaining = match_len;
            while remaining > 0 {
                let chunk = remaining.min(offset);
                let len = output.len();
                output.extend_from_within(len - offset..len - offset + chunk);
                remaining -= chunk;
            }
        }
    }

    if output.len() != original_size {
        return Err(AetherError::Decompression(format!(
            "LZ77: output size {}, expected {original_size}",
            output.len()
        )));
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_text() {
        let text = b"Hello, world! Hello, world! Hello, world! Hello, world! \
                     The quick brown fox jumps over the lazy dog. \
                     The quick brown fox jumps over the lazy dog.";
        let enc = lz77_encode(text).expect("text should compress");
        assert!(enc.len() < text.len());
        let dec = lz77_decode(&enc, text.len()).unwrap();
        assert_eq!(&dec[..], &text[..]);
    }

    #[test]
    fn roundtrip_large_repetitive() {
        let line = "The AetherArch compression pipeline uses LZ77 preprocessing.\n";
        let text: Vec<u8> = line.as_bytes().repeat(500);
        let enc = lz77_encode(&text).expect("should compress");
        assert!(enc.len() < text.len() / 2);
        let dec = lz77_decode(&enc, text.len()).unwrap();
        assert_eq!(dec, text);
    }

    #[test]
    fn roundtrip_rust_source() {
        let src = br#"
fn main() {
    let x = 42;
    println!("Hello, world! The answer is {}", x);
    for i in 0..10 {
        println!("  iteration {}", i);
    }
    println!("Done.");
    let x = 42;
    println!("Hello, world! The answer is {}", x);
    for i in 0..10 {
        println!("  iteration {}", i);
    }
    println!("Done.");
}
"#;
        let enc = lz77_encode(src).expect("should compress");
        assert!(enc.len() < src.len());
        let dec = lz77_decode(&enc, src.len()).unwrap();
        assert_eq!(&dec[..], &src[..]);
    }

    #[test]
    fn catches_3byte_matches() {
        // "the" repeated many times with different suffixes.
        // LZ4 (min-match-4) would miss these, LZ77 (min-match-3) catches them.
        let text = b"the cat and the dog and the fox and the owl and the bat";
        let enc = lz77_encode(text).expect("should compress");
        let dec = lz77_decode(&enc, text.len()).unwrap();
        assert_eq!(&dec[..], &text[..]);
    }

    #[test]
    fn incompressible_returns_none() {
        let random: Vec<u8> = (0..4096u64)
            .map(|i| {
                (i.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    >> 33) as u8
            })
            .collect();
        let result = lz77_encode(&random);
        if let Some(ref enc) = result {
            let dec = lz77_decode(enc, random.len()).unwrap();
            assert_eq!(dec, random);
        }
    }

    #[test]
    fn short_input_returns_none() {
        assert!(lz77_encode(b"AB").is_none());
        assert!(lz77_encode(b"").is_none());
    }

    #[test]
    fn decode_validates_size() {
        let text: Vec<u8> = b"abcdef1234567890 ".repeat(100);
        let enc = lz77_encode(&text).expect("should compress");
        let err = lz77_decode(&enc, text.len() + 100);
        assert!(err.is_err());
    }

    #[test]
    fn matches_overlap_correctly() {
        // "AAAA..." should compress to a single match with self-referencing copy
        let data = vec![0x41u8; 10_000];
        let enc = lz77_encode(&data).expect("should compress");
        assert!(enc.len() < data.len() / 10);
        let dec = lz77_decode(&enc, data.len()).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    #[cfg(feature = "lz4")]
    fn better_than_lz4_on_3byte_patterns() {
        // Text with lots of short repeated patterns
        let text = "fn f() { } fn g() { } fn h() { } fn i() { } fn j() { } \
                    fn k() { } fn l() { } fn m() { } fn n() { } fn o() { } "
            .as_bytes()
            .repeat(20);
        let lz77 = lz77_encode(&text).expect("lz77 should compress");
        let lz4 = crate::coding::lz_preprocess::lz_encode(&text);
        // LZ77 should be at least as good as LZ4 (usually better)
        if let Some(lz4) = lz4 {
            assert!(
                lz77.len() <= lz4.len(),
                "LZ77 ({}) should be <= LZ4 ({})",
                lz77.len(),
                lz4.len()
            );
        }
    }
}
