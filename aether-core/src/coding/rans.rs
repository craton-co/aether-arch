//! Custom byte-aligned range coder optimized for adaptive prediction.
//!
//! Replaces the `constriction` crate with a purpose-built implementation that:
//! - Works directly with 15-bit CDF tables (no per-symbol model allocation)
//! - Produces byte output natively (no `u32` → `u8` conversion layer)
//! - Uses carry-propagating shift_low for correct output
//!
//! The range coder is paired with a CDF converter (`probs_to_cdf`) that
//! quantizes the predictor's `[f32; 256]` distribution into a monotone
//! cumulative frequency table with 15-bit precision.

use crate::entropy::ProbabilityPredictor;
use crate::error::{AetherError, Result};

// ─── Constants ───────────────────────────────────────────────────────────

/// CDF precision: 15 bits.  Total probability mass = 32 768.
pub const PROB_BITS: u32 = 15;
pub const PROB_TOTAL: u32 = 1 << PROB_BITS; // 32768

/// Range is renormalized when it drops below this threshold (2^24).
const TOP: u32 = 1 << 24;

/// Resolve a cumulative frequency to its symbol.
///
/// BWT+MTF+RLE streams are heavily concentrated on RUNA/RUNB (symbols 0 and
/// 1), so two prefix checks avoid the full binary search for the common case.
/// The remaining symbols use a bounded binary search starting at symbol 2.
#[inline(always)]
fn find_symbol(cdf: &[u16; 257], freq: u32) -> usize {
    if freq < cdf[1] as u32 {
        return 0;
    }
    if freq < cdf[2] as u32 {
        return 1;
    }

    let mut lo = 2usize;
    let mut hi = 256usize;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if (cdf[mid + 1] as u32) <= freq {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// ─── CDF Conversion ─────────────────────────────────────────────────────

/// Convert a 256-element `f32` probability distribution to a cumulative
/// frequency table with 15-bit precision.
///
/// Guarantees:
/// - `cdf[0] == 0`
/// - `cdf[256] == PROB_TOTAL` (32 768)
/// - `cdf[i+1] > cdf[i]` for all `i` (every symbol is encodable)
#[inline]
pub fn probs_to_cdf(probs: &[f32; 256]) -> [u16; 257] {
    let mut cdf = [0u16; 257];

    // Cumulative rounding: each CDF entry is the rounded cumulative sum
    // scaled to PROB_TOTAL.  This minimizes total quantization error.
    // V4 security fix: clamp individual probabilities to non-negative finite
    // values before accumulation.  A predictor could produce NaN, Inf, or
    // negative values for individual symbols; clamping here prevents `cum`
    // from going backwards or becoming NaN during the scan below.
    let mut clamped = [0.0f64; 256];
    for i in 0..256 {
        let p = probs[i] as f64;
        clamped[i] = if p.is_finite() && p > 0.0 { p } else { 0.0 };
    }

    let sum: f64 = clamped.iter().sum();

    // V3 security fix: if the predictor produces all-zero (or all-garbage)
    // probabilities, fall back to uniform immediately rather than dividing
    // by zero or propagating garbage through the fix-up passes.
    if sum <= 0.0 {
        for (i, cdf_val) in cdf.iter_mut().enumerate() {
            *cdf_val = (i as u32 * PROB_TOTAL / 256) as u16;
        }
        cdf[256] = PROB_TOTAL as u16;
        return cdf;
    }

    let scale: f64 = PROB_TOTAL as f64 / sum;

    let mut cum: f64 = 0.0;
    for i in 0..256 {
        cdf[i] = (cum * scale + 0.5) as u16;
        cum += clamped[i];
    }
    cdf[256] = PROB_TOTAL as u16;

    // Ensure strict monotonicity (minimum freq = 1 per symbol).
    for i in 0..256 {
        if cdf[i + 1] <= cdf[i] {
            cdf[i + 1] = cdf[i] + 1;
        }
    }

    // If the forward fix-up pushed cdf[256] past PROB_TOTAL, the rounded
    // CDF doesn't fit.  Rather than a backward clamp (which can silently
    // degrade peaked distributions to near-uniform), use a guaranteed
    // proportional redistribution: give every symbol freq=1 first, then
    // distribute the remaining mass proportionally to clamped probs.
    if cdf[256] != PROB_TOTAL as u16 {
        const N: usize = 256;
        let remaining_mass = PROB_TOTAL as usize - N; // 32768 - 256 = 32512

        // Compute proportional share of the remaining mass for each symbol.
        let mut freqs = [1u16; N]; // every symbol gets at least 1
        let mut distributed = 0usize;
        for i in 0..N {
            let share = (clamped[i] / sum * remaining_mass as f64) as usize;
            freqs[i] += share as u16;
            distributed += share;
        }

        // Distribute leftover from rounding to the highest-probability
        // symbols (greedy, preserves distribution shape).
        let mut leftover = remaining_mass - distributed;
        if leftover > 0 {
            // Build index sorted by descending probability
            let mut indices: Vec<usize> = (0..N).collect();
            indices.sort_unstable_by(|&a, &b| {
                clamped[b]
                    .partial_cmp(&clamped[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &idx in &indices {
                if leftover == 0 {
                    break;
                }
                freqs[idx] += 1;
                leftover -= 1;
            }
        }

        // Rebuild CDF from freqs
        cdf[0] = 0;
        for i in 0..N {
            cdf[i + 1] = cdf[i] + freqs[i];
        }

        // S12 safety net: if floating-point edge cases still broke the
        // invariant, fall back to uniform.
        if cdf[256] != PROB_TOTAL as u16 {
            for (i, cdf_val) in cdf.iter_mut().enumerate() {
                *cdf_val = (i as u32 * PROB_TOTAL / 256) as u16;
            }
            cdf[256] = PROB_TOTAL as u16;
        }
    }

    debug_assert_eq!(cdf[0], 0, "CDF must start at 0");
    debug_assert_eq!(cdf[256], PROB_TOTAL as u16, "CDF must end at PROB_TOTAL");
    // Verify strict monotonicity in debug builds
    debug_assert!(
        (0..256).all(|i| cdf[i + 1] > cdf[i]),
        "CDF is not strictly monotonic",
    );

    cdf
}

// ─── Range Encoder ───────────────────────────────────────────────────────

/// Byte-aligned range encoder with carry propagation (LZMA-style).
pub struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u32,
    output: Vec<u8>,
    /// V8 safety: set to true if the cache overflow valve fires, indicating
    /// the output stream is potentially corrupt and should not be used.
    cache_overflow: bool,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            output: Vec::new(),
            cache_overflow: false,
        }
    }

    /// Encode a symbol given its CDF interval.
    #[inline(always)]
    pub fn encode_cdf(&mut self, symbol: u8, cdf: &[u16; 257]) {
        let s = symbol as usize;
        self.encode_interval(cdf[s], cdf[s + 1]);
    }

    /// Encode a symbol given only its `(cdf_lo, cdf_hi)` interval.
    ///
    /// Equivalent to `encode_cdf(symbol, &cdf)` when `cdf_lo == cdf[symbol]`
    /// and `cdf_hi == cdf[symbol + 1]`, but skips materialising the other
    /// 255 entries — the encoder never reads them. Paired with
    /// [`ProbabilityPredictor::query_cdf`]
    /// in the encode hot loop.
    #[inline(always)]
    pub fn encode_interval(&mut self, cdf_lo: u16, cdf_hi: u16) {
        let cum = cdf_lo as u32;
        let freq = (cdf_hi - cdf_lo) as u32;
        debug_assert!(freq > 0, "zero-frequency symbol (cdf_lo == cdf_hi)");

        let r = self.range / PROB_TOTAL;
        // V6: after renormalization range >= TOP (2^24), so r >= 2^24 / 2^15 = 512.
        debug_assert!(
            r > 0,
            "range/PROB_TOTAL must be positive after renormalization"
        );
        self.low += cum as u64 * r as u64;
        self.range = freq * r;

        // Renormalize: shift out resolved top bytes.
        while self.range < TOP {
            self.shift_low();
            self.range <<= 8;
        }
    }

    /// Carry-propagating byte output.
    #[inline(always)]
    fn shift_low(&mut self) {
        let low_hi = (self.low >> 32) as u8;
        if (self.low as u32) < 0xFF00_0000 || low_hi != 0 {
            self.output.push(self.cache.wrapping_add(low_hi));
            let fill = 0xFFu8.wrapping_add(low_hi);
            for _ in 0..self.cache_size.saturating_sub(1) {
                self.output.push(fill);
            }
            self.cache = (self.low >> 24) as u8;
            self.cache_size = 0;
        }
        self.cache_size += 1;
        // R3: cache_size grows when carry resolution is deferred.  In the
        // worst case it can approach the input length.  Cap at 16 MiB to
        // prevent unbounded memory growth from pathological predictors.
        if self.cache_size > 1 << 24 {
            // V8 security fix: flag the stream as corrupt rather than silently
            // flushing bytes the decoder doesn't expect.  The old code wrote
            // arbitrary cache bytes here, producing output that would decode
            // to wrong data.  Callers (encode_block) check this flag via
            // finish() and return an error.
            self.cache_overflow = true;
            // Still flush to prevent OOM, but the output is now invalid.
            self.output.push(self.cache);
            for _ in 0..self.cache_size.saturating_sub(2) {
                self.output.push(0xFF);
            }
            self.cache = (self.low >> 24) as u8;
            self.cache_size = 1;
        }
        self.low = ((self.low as u32) << 8) as u64;
    }

    /// Flush remaining state and return the compressed byte stream.
    ///
    /// Returns `Err` if the cache overflow safety valve fired during
    /// encoding, which means the output is corrupt (V8 security fix).
    pub fn finish(mut self) -> std::result::Result<Vec<u8>, &'static str> {
        for _ in 0..5 {
            self.shift_low();
        }
        if self.cache_overflow {
            Err("range encoder cache overflow: pathological predictor produced corrupt output")
        } else {
            Ok(self.output)
        }
    }
}

// ─── Range Decoder ───────────────────────────────────────────────────────

/// Byte-aligned range decoder (matched to `RangeEncoder`).
pub struct RangeDecoder<'a> {
    code: u32,
    range: u32,
    input: &'a [u8],
    pos: usize,
    /// Number of virtual zero bytes consumed past the end of input.
    /// A small overread (≤5) is normal due to the encoder's flush padding.
    /// Large overread indicates a truncated or corrupted stream.
    eof_overread: usize,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut dec = Self {
            code: 0,
            range: 0xFFFF_FFFF,
            input: data,
            pos: 0,
            eof_overread: 0,
        };
        // Initialise `code` from the first 5 bytes (matches encoder flush).
        for _ in 0..5 {
            dec.code = (dec.code << 8) | dec.read_byte() as u32;
        }
        dec
    }

    #[inline(always)]
    fn read_byte(&mut self) -> u8 {
        if self.pos < self.input.len() {
            let b = self.input[self.pos];
            self.pos += 1;
            b
        } else {
            self.eof_overread += 1;
            0
        }
    }

    /// Decode one symbol using a CDF table.
    #[inline(always)]
    pub fn decode_cdf(&mut self, cdf: &[u16; 257]) -> u8 {
        let r = self.range / PROB_TOTAL;
        // V6: after renormalization range >= TOP (2^24), so r >= 2^24 / 2^15 = 512.
        debug_assert!(
            r > 0,
            "range/PROB_TOTAL must be positive after renormalization"
        );
        let freq = (self.code / r).min(PROB_TOTAL - 1);

        let symbol = find_symbol(cdf, freq);

        let cum = cdf[symbol] as u32;
        let sym_freq = (cdf[symbol + 1] - cdf[symbol]) as u32;
        debug_assert!(sym_freq > 0, "zero-frequency symbol {symbol}");

        self.code -= cum * r;
        self.range = sym_freq * r;

        // Renormalize.
        while self.range < TOP {
            self.code = (self.code << 8) | self.read_byte() as u32;
            self.range <<= 8;
        }

        symbol as u8
    }
}

// ─── Block-level API ─────────────────────────────────────────────────────

/// Encode a block of data using predictor-guided range coding.
///
/// Returns compressed data as a byte vector.
/// The predictor is reset at the start, then advanced byte-by-byte.
#[must_use = "encode_block returns the compressed data; discarding it silently loses the V8 cache overflow error"]
pub fn encode_block<P: ProbabilityPredictor + ?Sized>(
    data: &[u8],
    predictor: &mut P,
) -> Result<Vec<u8>> {
    encode_block_inner(data, predictor, true)
}

/// Encode a block WITHOUT resetting the predictor first.
///
/// Used for cross-block predictor state carry within a solid group:
/// blocks 2+ reuse the predictor state from the previous block's
/// predict/update calls, giving the predictor prior context.
pub fn encode_block_continuing<P: ProbabilityPredictor + ?Sized>(
    data: &[u8],
    predictor: &mut P,
) -> Result<Vec<u8>> {
    encode_block_inner(data, predictor, false)
}

fn encode_block_inner<P: ProbabilityPredictor + ?Sized>(
    data: &[u8],
    predictor: &mut P,
    reset: bool,
) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    if reset {
        predictor.reset();
    }
    let mut enc = RangeEncoder::new();

    for &byte in data {
        // Encode-only fast path: ask the predictor for only the two CDF
        // entries the range coder consumes (cdf[byte], cdf[byte+1]).
        // The trait's default impl falls back to predict_cdf + slice, so
        // predictors that haven't migrated are unaffected.
        let (lo, hi) = predictor.query_cdf(byte);
        enc.encode_interval(lo, hi);
        predictor.update(byte);
    }

    enc.finish().map_err(|e| AetherError::Compression(e.into()))
}

/// Maximum decodable block size for range decoder output.
///
/// Aligned with [`crate::format::MAX_DECOMPRESSED_BLOCK_SIZE`] (64 MiB) to
/// prevent silent truncation of legitimate blocks.  Previously this was
/// 16 MiB, which could corrupt decompression of chunks near the old limit
/// or any future increase in `MAX_CHUNK_SIZE`.
///
/// The range decoder allocates `expected_len` bytes up-front, so this
/// constant also guards against OOM from corrupted size fields in crafted
/// archives.
const MAX_DECODE_SIZE: usize = crate::format::MAX_DECOMPRESSED_BLOCK_SIZE;

/// Decode a block from compressed bytes back to original data.
///
/// `expected_len` must match the original uncompressed length exactly.
/// The predictor must be the same type that was used for encoding.
pub fn decode_block<P: ProbabilityPredictor + ?Sized>(
    compressed: &[u8],
    expected_len: usize,
    predictor: &mut P,
) -> Result<Vec<u8>> {
    decode_block_inner(compressed, expected_len, predictor, true)
}

/// Decode a block WITHOUT resetting the predictor first.
///
/// Used for cross-block predictor state carry: the predictor continues
/// from the state left by the previous block's decode+update calls.
pub fn decode_block_continuing<P: ProbabilityPredictor + ?Sized>(
    compressed: &[u8],
    expected_len: usize,
    predictor: &mut P,
) -> Result<Vec<u8>> {
    decode_block_inner(compressed, expected_len, predictor, false)
}

fn decode_block_inner<P: ProbabilityPredictor + ?Sized>(
    compressed: &[u8],
    expected_len: usize,
    predictor: &mut P,
    reset: bool,
) -> Result<Vec<u8>> {
    if expected_len == 0 {
        return Ok(Vec::new());
    }
    if compressed.is_empty() {
        return Err(AetherError::Decompression(
            "Empty compressed data for non-zero expected length".into(),
        ));
    }
    if expected_len > MAX_DECODE_SIZE {
        return Err(AetherError::Decompression(format!(
            "Decode size {} exceeds safety limit {}",
            expected_len, MAX_DECODE_SIZE
        )));
    }

    if reset {
        predictor.reset();
    }
    let mut dec = RangeDecoder::new(compressed);
    // V2 security fix: don't pre-allocate the full expected_len from an
    // untrusted size field.  Cap the initial allocation to the smaller of
    // expected_len and the compressed input size × 4 (a reasonable expansion
    // estimate), so a crafted archive with expected_len = 64 MiB but only a
    // few bytes of compressed data doesn't force a huge allocation up front.
    let safe_capacity = expected_len.min(compressed.len().saturating_mul(4).max(4096));
    let mut output = Vec::with_capacity(safe_capacity);

    for _ in 0..expected_len {
        let cdf = predictor.predict_cdf();
        let byte = dec.decode_cdf(&cdf);
        predictor.update(byte);
        output.push(byte);
    }

    // V9 security fix: detect truncated/corrupted compressed streams.
    // The encoder flushes 5 bytes, so up to 5 virtual zero reads past EOF
    // are normal.  Anything beyond that means the decoder was synthesizing
    // data from zeros — the output is silently garbage.
    const MAX_NORMAL_OVERREAD: usize = 5;
    if dec.eof_overread > MAX_NORMAL_OVERREAD {
        return Err(AetherError::Decompression(format!(
            "Range decoder read {} bytes past end of input (truncated stream?)",
            dec.eof_overread,
        )));
    }

    Ok(output)
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "context-mixer")]
    use crate::entropy::context_mixer::{ContextMixer, ContextMixerConfig};
    use crate::entropy::order0::Order0Model;

    #[test]
    fn prefix_symbol_lookup_matches_exhaustive_reference() {
        let mut frequencies = [1u16; 256];
        frequencies[0] = 12_000;
        frequencies[1] = 8_000;
        frequencies[2] = 4_000;
        let assigned: u32 = frequencies.iter().map(|&value| value as u32).sum();
        frequencies[255] = frequencies[255].saturating_add((PROB_TOTAL - assigned) as u16);

        let mut cdf = [0u16; 257];
        for symbol in 0..256 {
            cdf[symbol + 1] = cdf[symbol] + frequencies[symbol];
        }
        assert_eq!(cdf[256], PROB_TOTAL as u16);

        for freq in 0..PROB_TOTAL {
            let expected = (0..256)
                .find(|&symbol| freq < cdf[symbol + 1] as u32)
                .unwrap();
            assert_eq!(find_symbol(&cdf, freq), expected);
        }
    }

    /// Reference (pre-optimization) implementation of `probs_to_cdf`,
    /// kept verbatim so we can prove bit-identity against the optimized
    /// version on a wide test corpus. NOT exported — tests only.
    ///
    /// If you change this, also update the production `probs_to_cdf` and
    /// re-run [`probs_to_cdf_matches_reference_on_diverse_inputs`].
    #[allow(clippy::needless_range_loop)]
    fn probs_to_cdf_reference(probs: &[f32; 256]) -> [u16; 257] {
        let mut cdf = [0u16; 257];
        let mut clamped = [0.0f64; 256];
        for i in 0..256 {
            let p = probs[i] as f64;
            clamped[i] = if p.is_finite() && p > 0.0 { p } else { 0.0 };
        }
        let sum: f64 = clamped.iter().sum();
        if sum <= 0.0 {
            for (i, cdf_val) in cdf.iter_mut().enumerate() {
                *cdf_val = (i as u32 * PROB_TOTAL / 256) as u16;
            }
            cdf[256] = PROB_TOTAL as u16;
            return cdf;
        }
        let scale: f64 = PROB_TOTAL as f64 / sum;
        let mut cum: f64 = 0.0;
        for i in 0..256 {
            cdf[i] = (cum * scale + 0.5) as u16;
            cum += clamped[i];
        }
        cdf[256] = PROB_TOTAL as u16;
        for i in 0..256 {
            if cdf[i + 1] <= cdf[i] {
                cdf[i + 1] = cdf[i] + 1;
            }
        }
        if cdf[256] != PROB_TOTAL as u16 {
            const N: usize = 256;
            let remaining_mass = PROB_TOTAL as usize - N;
            let mut freqs = [1u16; N];
            let mut distributed = 0usize;
            for i in 0..N {
                let share = (clamped[i] / sum * remaining_mass as f64) as usize;
                freqs[i] += share as u16;
                distributed += share;
            }
            let mut leftover = remaining_mass - distributed;
            if leftover > 0 {
                let mut indices: Vec<usize> = (0..N).collect();
                indices.sort_unstable_by(|&a, &b| {
                    clamped[b]
                        .partial_cmp(&clamped[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for &idx in &indices {
                    if leftover == 0 {
                        break;
                    }
                    freqs[idx] += 1;
                    leftover -= 1;
                }
            }
            cdf[0] = 0;
            for i in 0..N {
                cdf[i + 1] = cdf[i] + freqs[i];
            }
            if cdf[256] != PROB_TOTAL as u16 {
                for (i, cdf_val) in cdf.iter_mut().enumerate() {
                    *cdf_val = (i as u32 * PROB_TOTAL / 256) as u16;
                }
                cdf[256] = PROB_TOTAL as u16;
            }
        }
        cdf
    }

    /// The optimized `probs_to_cdf` must produce a CDF that is byte-exactly
    /// equal to the reference for ALL inputs we care about. This is the
    /// primary backward-compat guard: existing archives encode bytes
    /// against the reference algorithm, and any deviation silently
    /// corrupts decompression.
    #[test]
    fn probs_to_cdf_matches_reference_on_diverse_inputs() {
        // Test corpus covers: uniform, peaked (overshoot territory),
        // near-uniform with one tiny entry, all-tied-tiny (MIN_PROB clamp
        // collapse), incremental-decay, and a pseudo-random distribution.
        let mut cases: Vec<[f32; 256]> = Vec::new();

        // 1. Uniform.
        cases.push([1.0 / 256.0; 256]);

        // 2. Peaked: one symbol dominant, rest tiny (NeuralSSM after RLE
        //    training). Forces overshoot.
        let mut peaked = [1e-6f32; 256];
        peaked[0] = 0.99;
        cases.push(peaked);

        // 3. Two peaks (RUNA + RUNB pattern after BWT+MTF+RLE).
        let mut two_peak = [1e-6f32; 256];
        two_peak[0] = 0.6;
        two_peak[1] = 0.39;
        cases.push(two_peak);

        // 4. All entries clamped to MIN_PROB equivalent — maximum tie
        //    pressure on the leftover-sort branch.
        cases.push([1e-7f32; 256]);

        // 5. Geometric decay (typical Order0 mid-training).
        let mut geom = [0.0f32; 256];
        let mut v = 0.5f32;
        for slot in geom.iter_mut() {
            *slot = v.max(1e-7);
            v *= 0.97;
        }
        cases.push(geom);

        // 6. Pseudo-random LCG distribution.
        let mut rng_state = 0x9e37_79b9u32;
        let mut rand_probs = [0.0f32; 256];
        for slot in rand_probs.iter_mut() {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            *slot = (rng_state >> 8) as f32 / (1u32 << 24) as f32;
        }
        cases.push(rand_probs);

        // 7. Sparse: only 5 non-zero entries (heavy overshoot path).
        let mut sparse = [1e-8f32; 256];
        sparse[10] = 0.5;
        sparse[50] = 0.3;
        sparse[100] = 0.15;
        sparse[150] = 0.04;
        sparse[200] = 0.01;
        cases.push(sparse);

        for (idx, probs) in cases.iter().enumerate() {
            let opt = probs_to_cdf(probs);
            let reference = probs_to_cdf_reference(probs);
            assert_eq!(
                opt, reference,
                "probs_to_cdf diverged from reference on case #{idx}"
            );
        }
    }

    /// Microbench: compare optimized `probs_to_cdf` against the reference
    /// implementation by replaying the NeuralSSM predict-loop on the same
    /// English text corpus the criterion benchmarks use. Run with
    /// `--ignored --nocapture` to see the numbers.
    ///
    /// This is repeatable (single tight loop, minimal harness noise) and
    /// directly measures the function we're optimizing — unlike the
    /// criterion `compress_ssm` bench which has ±30% machine variance.
    #[test]
    #[ignore]
    #[cfg(feature = "bwt-encode")]
    fn probs_to_cdf_microbench_vs_reference() {
        use crate::entropy::NeuralSsmPredictor;

        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tests")
            .join("fixtures")
            .join("large");
        let text = match std::fs::read(dir.join("english.txt")) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("[microbench] english.txt fixture missing, skipping");
                return;
            }
        };
        let bwt_data = match crate::coding::bwt_preprocess::bwt_mtf_encode(&text) {
            Ok(d) => d,
            Err(_) => return,
        };
        eprintln!("[microbench] BWT-encoded corpus: {} bytes", bwt_data.len());

        // Run the NeuralSSM predict() → probs_to_cdf() pipeline, which is
        // structurally identical to what predict_cdf does internally and
        // hits the same overshoot ratio (~98.76%).
        let bench = |name: &str, cdf_fn: fn(&[f32; 256]) -> [u16; 257]| {
            let mut pred = NeuralSsmPredictor::new();
            let start = std::time::Instant::now();
            let mut total_sample = 0u64; // prevent dead-code elimination
            for &byte in &bwt_data {
                let probs = pred.predict();
                let cdf = cdf_fn(&probs);
                total_sample = total_sample.wrapping_add(cdf[byte as usize] as u64);
                pred.update(byte);
            }
            let elapsed = start.elapsed();
            let mbps = bwt_data.len() as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0);
            eprintln!(
                "[{}] {:?}  ({:.3} MiB/s)  sample={}",
                name, elapsed, mbps, total_sample
            );
            elapsed
        };

        // Warm-up (file I/O, BWT, predictor init) is already done; cache
        // is hot. Run reference first, then optimized.
        let reference_time = bench("reference", probs_to_cdf_reference);
        let optimized_time = bench("optimized", probs_to_cdf);

        let speedup = reference_time.as_secs_f64() / optimized_time.as_secs_f64();
        let delta_pct = (1.0 - optimized_time.as_secs_f64() / reference_time.as_secs_f64()) * 100.0;
        eprintln!(
            "[delta] optimized is {:+.2}% vs reference (speedup {:.3}x)",
            delta_pct, speedup
        );
    }

    // ── CDF sanity ──────────────────────────────────────────────────

    #[test]
    fn cdf_uniform() {
        let probs = [1.0f32 / 256.0; 256];
        let cdf = probs_to_cdf(&probs);
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[256], PROB_TOTAL as u16);
        for i in 0..256 {
            assert!(cdf[i + 1] > cdf[i], "monotonicity at {i}");
        }
    }

    #[test]
    fn cdf_peaked() {
        let mut probs = [1e-10f32; 256];
        probs[42] = 0.999;
        let cdf = probs_to_cdf(&probs);
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[256], PROB_TOTAL as u16);
        for i in 0..256 {
            assert!(cdf[i + 1] > cdf[i], "monotonicity at {i}");
        }
        // Symbol 42 should get the lion's share.
        let freq_42 = cdf[43] - cdf[42];
        assert!(freq_42 > (PROB_TOTAL as u16) / 2, "freq_42 = {freq_42}");
    }

    // ── Roundtrip tests ─────────────────────────────────────────────

    #[test]
    fn roundtrip_empty() {
        let mut enc_pred = Order0Model::new();
        let mut dec_pred = Order0Model::new();

        let compressed = encode_block(&[], &mut enc_pred).unwrap();
        let decoded = decode_block(&compressed, 0, &mut dec_pred).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn roundtrip_single_byte() {
        let mut enc_pred = Order0Model::new();
        let mut dec_pred = Order0Model::new();

        let data = [42u8];
        let compressed = encode_block(&data, &mut enc_pred).unwrap();
        let decoded = decode_block(&compressed, data.len(), &mut dec_pred).unwrap();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn roundtrip_uniform_data() {
        let mut enc_pred = Order0Model::new();
        let mut dec_pred = Order0Model::new();

        let data = vec![0xAA; 1000];
        let compressed = encode_block(&data, &mut enc_pred).unwrap();
        let decoded = decode_block(&compressed, data.len(), &mut dec_pred).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn roundtrip_english_text() {
        let mut enc_pred = Order0Model::new();
        let mut dec_pred = Order0Model::new();

        let data = b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        let compressed = encode_block(data, &mut enc_pred).unwrap();
        let decoded = decode_block(&compressed, data.len(), &mut dec_pred).unwrap();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn roundtrip_all_byte_values() {
        let mut enc_pred = Order0Model::new();
        let mut dec_pred = Order0Model::new();

        let data: Vec<u8> = (0..=255).collect();
        let compressed = encode_block(&data, &mut enc_pred).unwrap();
        let decoded = decode_block(&compressed, data.len(), &mut dec_pred).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn compresses_repetitive_data() {
        let mut pred = Order0Model::new();

        let data = vec![0x42; 10_000];
        let compressed = encode_block(&data, &mut pred).unwrap();

        // Highly repetitive data should compress very well
        assert!(
            compressed.len() < data.len() / 5,
            "Expected significant compression: {} -> {}",
            data.len(),
            compressed.len()
        );
    }

    #[test]
    #[cfg(feature = "context-mixer")]
    fn roundtrip_with_context_mixer() {
        let data = b"Hello! This tests the context mixer predictor with range coding. \
                     The mixer should handle this text correctly. Repeated patterns \
                     help the mixer learn: help the mixer learn: help the mixer learn.";

        let mut enc_pred = ContextMixer::with_config(ContextMixerConfig::lightweight());
        let compressed = encode_block(data, &mut enc_pred).unwrap();

        let mut dec_pred = ContextMixer::with_config(ContextMixerConfig::lightweight());
        let decoded = decode_block(&compressed, data.len(), &mut dec_pred).unwrap();

        assert_eq!(data.to_vec(), decoded);
    }

    fn roundtrip_test(
        data: Vec<u8>,
        predictor_factory: impl Fn() -> Box<dyn ProbabilityPredictor>,
    ) {
        let mut enc_pred = predictor_factory();
        let compressed = encode_block(&data, enc_pred.as_mut()).expect("encode");

        let mut dec_pred = predictor_factory();
        let decoded = decode_block(&compressed, data.len(), dec_pred.as_mut()).expect("decode");
        assert_eq!(
            data,
            decoded,
            "data roundtrip mismatch at len={}",
            data.len()
        );
    }

    #[test]
    fn roundtrip_json_sized_data_order0() {
        // Test with JSON-like data similar to data.json (726 bytes)
        let json = br#"{
  "name": "AetherArch",
  "version": "0.1.0",
  "description": "Next-generation file archiver",
  "authors": ["Anonymous"],
  "features": [
    "neural-probabilistic prediction",
    "context-mixing compression",
    "content-defined chunking",
    "semantic solid archiving",
    "adaptive entropy routing",
    "range coding via constriction",
    "zstandard fallback",
    "BLAKE3 integrity checksums"
  ],
  "benchmarks": {
    "english_text_bpb": 2.5,
    "source_code_bpb": 3.0,
    "binary_data_bpb": 6.5,
    "random_data_bpb": 8.0
  },
  "config": {
    "min_chunk_size": 4096,
    "avg_chunk_size": 65536,
    "max_chunk_size": 524288,
    "high_entropy_threshold": 7.5,
    "incompressible_threshold": 7.95
  }
}
"#;
        roundtrip_test(json.to_vec(), || Box::new(Order0Model::new()));
    }

    #[test]
    fn roundtrip_various_sizes_order0() {
        // Test roundtrip with various data sizes from 1 to 2000 bytes
        for size in [1, 10, 50, 100, 200, 500, 726, 809, 1000, 1098, 2000] {
            let data: Vec<u8> = (0..size).map(|i| (i % 97 + 32) as u8).collect();
            roundtrip_test(data, || Box::new(Order0Model::new()));
        }
    }

    #[test]
    #[cfg(feature = "context-mixer")]
    fn context_mixer_compresses_better_than_order0() {
        let data = b"ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB\
                     CDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCDCD";

        let mut o0 = Order0Model::new();
        let comp_o0 = encode_block(data, &mut o0).unwrap();

        let mut cm = ContextMixer::with_config(ContextMixerConfig::lightweight());
        let comp_cm = encode_block(data, &mut cm).unwrap();

        // Context mixer should compress better on patterned data
        assert!(
            comp_cm.len() <= comp_o0.len(),
            "Context mixer ({}) should be <= order-0 ({}) on patterned data",
            comp_cm.len(),
            comp_o0.len(),
        );
    }
}
