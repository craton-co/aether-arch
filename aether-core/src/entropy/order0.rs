//! Order-0 adaptive frequency model.
//!
//! The simplest possible predictor: it tracks how often each byte value has
//! appeared so far and uses the observed frequencies as the probability
//! distribution. No context is considered — hence "order-0".
//!
//! This serves as the baseline predictor and a building block for testing
//! the range coding pipeline.

use super::traits::ProbabilityPredictor;
use crate::format::PredictorId;

/// Largest `total` for which [`ExactDivisor`] is provably exact.
///
/// The magic-number bound below needs `total^2 * (PROB_TOTAL + 1) <= 2^SHIFT`.
/// The model rescales at 1_000_000, so this ceiling is never reached in
/// practice; [`ExactDivisor::div`] falls back to a hardware divide if it is.
const MAX_MAGIC_TOTAL: u32 = 1 << 20;

/// Exact `n / total` by multiply-and-shift, for a `total` that stays fixed
/// across a whole CDF sweep.
///
/// Both `predict_cdf` and `decode_symbol` divide by `self.total` once per
/// symbol — up to 256 times to resolve a single byte. A 64-bit hardware
/// divide is ~20-40 cycles; a widening multiply plus a shift is ~4, and this
/// is the dominant cost of decoding a byte-plane block.
///
/// With `M = floor(2^K / d) + 1` and `e = M*d - 2^K` (so `1 <= e <= d`),
/// `n*M / 2^K = n/d + n*e/(d * 2^K)`, which floors to `floor(n/d)` exactly
/// when `n * e < 2^K`. Here `n = cum * PROB_TOTAL + total/2 < total *
/// (PROB_TOTAL + 1)` and `e <= total`, so `total^2 * (PROB_TOTAL + 1) <= 2^K`
/// is sufficient — satisfied by `K = 56` for every `total <= 2^20`.
#[derive(Clone, Copy)]
struct ExactDivisor {
    total: u32,
    magic: u128,
    exact: bool,
}

impl ExactDivisor {
    const SHIFT: u32 = 56;

    #[inline]
    fn new(total: u32) -> Self {
        let exact = total > 0 && total <= MAX_MAGIC_TOTAL;
        let magic = if exact {
            ((1u128 << Self::SHIFT) / total as u128) + 1
        } else {
            0
        };
        Self {
            total,
            magic,
            exact,
        }
    }

    #[inline(always)]
    fn div(&self, n: u64) -> u64 {
        if self.exact {
            ((n as u128 * self.magic) >> Self::SHIFT) as u64
        } else {
            n / self.total as u64
        }
    }
}

/// Adaptive order-0 (unigram) frequency model with Laplace smoothing.
pub struct Order0Model {
    /// Frequency count for each byte value. Starts at 1 (Laplace prior).
    counts: [u32; 256],
    /// Sum of all counts. Maintained incrementally for speed.
    total: u32,
}

impl Order0Model {
    pub fn new() -> Self {
        Self {
            counts: [1; 256],
            total: 256, // 256 * 1
        }
    }
}

impl Default for Order0Model {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityPredictor for Order0Model {
    fn predict(&mut self) -> [f32; 256] {
        let mut probs = [0.0f32; 256];
        let total = self.total as f32;
        for (i, prob) in probs.iter_mut().enumerate() {
            *prob = self.counts[i] as f32 / total;
        }
        probs
    }

    /// Build CDF directly from integer counts, bypassing float conversion.
    fn predict_cdf(&mut self) -> [u16; 257] {
        use crate::coding::rans::PROB_TOTAL;

        let mut cdf = [0u16; 257];

        // Scale integer counts to 15-bit CDF using cumulative rounding.
        // `ExactDivisor` replaces 256 hardware divides with 256 multiplies
        // and produces bit-identical results.
        let total = self.total;
        let divisor = ExactDivisor::new(total);
        let half = (total / 2) as u64;
        let mut cum = 0u64;
        for (i, cdf_val) in cdf.iter_mut().enumerate().take(256) {
            *cdf_val = divisor.div(cum * PROB_TOTAL as u64 + half) as u16;
            cum += self.counts[i] as u64;
        }
        cdf[256] = PROB_TOTAL as u16;

        // Ensure strict monotonicity (Laplace prior guarantees counts >= 1,
        // but quantization can still collapse small symbols).
        for i in 0..256 {
            if cdf[i + 1] <= cdf[i] {
                cdf[i + 1] = cdf[i] + 1;
            }
        }

        // If fixup overshot, fall back to general probs_to_cdf.
        if cdf[256] != PROB_TOTAL as u16 {
            return crate::coding::rans::probs_to_cdf(&self.predict());
        }

        cdf
    }

    fn update(&mut self, byte: u8) {
        self.counts[byte as usize] += 1;
        self.total += 1;

        // Periodic rescaling to adapt to local statistics and prevent overflow.
        // Halve all counts (minimum 1) when total exceeds threshold.
        if self.total > 1_000_000 {
            self.total = 0;
            for c in self.counts.iter_mut() {
                *c = (*c >> 1).max(1);
                self.total += *c;
            }
        }
    }

    fn reset(&mut self) {
        self.counts = [1; 256];
        self.total = 256;
    }

    /// Fast path: read only the two CDF entries the encoder needs
    /// (`cdf[byte]` and `cdf[byte+1]`), bit-identical to what `predict_cdf`
    /// would produce.
    ///
    /// **Correctness contract.** The decoder uses `predict_cdf`, which:
    /// 1. computes each `cdf[i]` by cumulative integer rounding, then
    /// 2. applies a forward monotonicity fix-up that can chain — bumping
    ///    `cdf[i]` may force bumping `cdf[i+1]`, and so on, and
    /// 3. falls back to [`crate::coding::rans::probs_to_cdf`] entirely if
    ///    that fix-up pushed `cdf[256]` past `PROB_TOTAL`.
    ///
    /// All three steps must be reproduced exactly. Step 3 is the subtle
    /// one: it is a *whole-table* decision, so it cannot be observed from a
    /// sweep that stops at `byte + 1`. An earlier revision only checked the
    /// `s == 255` anchor and trusted the invariant elsewhere; that silently
    /// desynchronised encoder and decoder on skewed distributions (e.g. the
    /// float-exponent planes fed to `byteplane_encode`), producing archives
    /// whose blocks failed to decode. See
    /// `query_cdf_matches_predict_cdf_on_skewed_counts`.
    ///
    /// Two regimes:
    ///
    /// * `total <= PROB_TOTAL` — every symbol holds at least one count
    ///   (Laplace prior), so every rounded gap is `>= 1`, the fix-up never
    ///   fires and overshoot is impossible. The partial sweep up to
    ///   `byte + 1` is then exact, and the upper `254 - s` rounding
    ///   divisions, the fix-up sweep, and the 514-byte `[u16; 257]` return
    ///   are all skipped.
    /// * `total > PROB_TOTAL` — a rounded gap can collapse to zero, so we
    ///   replay the full forward sweep (without materialising the table)
    ///   and delegate to `predict_cdf` when it would have overshot.
    fn query_cdf(&mut self, byte: u8) -> (u16, u16) {
        use crate::coding::rans::PROB_TOTAL;
        let s = byte as usize;
        let divisor = ExactDivisor::new(self.total);
        let total = self.total as u64;
        let half = total / 2;
        let scale = PROB_TOTAL as u64;

        if total <= scale {
            // No fix-up possible: gap_i = round((cum+c_i)·S/T) - round(cum·S/T)
            // is at least floor(c_i · S / T) >= 1 because c_i >= 1 and T <= S.
            let mut cum: u64 = 0;
            for &count in &self.counts[..s] {
                cum += count as u64;
            }
            let lo = divisor.div(cum * scale + half) as u16;
            let hi = if s == 255 {
                PROB_TOTAL as u16
            } else {
                divisor.div((cum + self.counts[s] as u64) * scale + half) as u16
            };
            return (lo, hi);
        }

        // Full forward sweep, mirroring predict_cdf's rounding + chained
        // monotonicity fix-up, but keeping only the two entries we need.
        // `prev` ends as cdf[255], which is what decides the overshoot.
        let mut prev: u16 = 0; // cdf[0] == 0
        let mut cum: u64 = 0;
        let mut cdf_s: u16 = 0;
        let mut cdf_s1: u16 = 0;
        for i in 1..256 {
            cum += self.counts[i - 1] as u64;
            let mut val = divisor.div(cum * scale + half) as u16;
            if val <= prev {
                val = prev + 1;
            }
            if i == s {
                cdf_s = val;
            } else if i == s + 1 {
                cdf_s1 = val;
            }
            prev = val;
        }

        // predict_cdf pins cdf[256] = PROB_TOTAL and then bumps it when
        // cdf[255] >= PROB_TOTAL; that bump is exactly its overshoot
        // condition, which rebuilds the whole table via probs_to_cdf.
        if prev >= PROB_TOTAL as u16 {
            let cdf = self.predict_cdf();
            return (cdf[s], cdf[s + 1]);
        }
        if s == 255 {
            cdf_s1 = PROB_TOTAL as u16;
        }

        (cdf_s, cdf_s1)
    }

    // No `decode_symbol` override.
    //
    // The obvious one — resolve the symbol during the rounding sweep instead
    // of materialising the table — measured *slower* than the trait default
    // (0.90x on a BWT+MTF+RLE stream, 0.52x on a skewed byte plane). Two
    // reasons, both structural:
    //
    //  * The forward monotonicity fix-up chains, so unlike
    //    `NeuralSsmPredictor` the boundaries cannot be binary-searched;
    //    the sweep runs to `i = 255` either way.
    //  * The overshoot fallback is a whole-table decision, only known at
    //    `i = 255`. When it fires — which is the common case on the skewed
    //    planes byte-plane blocks are made of — the sweep is wasted and the
    //    table is rebuilt anyway, so the fused loop pays for it twice.
    //
    // `predict_cdf` + the decoder's `find_symbol` is the faster shape here.
    // The divide-free `ExactDivisor` above is where this model's decode win
    // actually comes from (2.25x on CDF construction).

    fn name(&self) -> &str {
        "order-0"
    }

    fn predictor_id(&self) -> PredictorId {
        PredictorId::Order0
    }

    fn save_state(&self) -> Option<Vec<u8>> {
        // Format: [version: u8] [u32; 256] counts in little-endian
        let mut buf = Vec::with_capacity(1 + 256 * 4);
        buf.push(1); // version 1
        for &c in &self.counts {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        Some(buf)
    }

    fn load_state(&mut self, data: &[u8]) -> bool {
        if data.len() != 1 + 256 * 4 {
            return false;
        }
        if data[0] != 1 {
            return false; // Unknown version
        }
        let data = &data[1..];
        // Maximum allowed count per symbol. This bounds the skew an adversarial
        // payload can introduce. 2M per symbol × 256 symbols fits in u32 and
        // is well above any count the model would reach organically (rescale
        // fires at 1M total).
        const MAX_COUNT: u32 = 2_000_000;

        let mut counts = [0u32; 256];
        let mut total: u64 = 0;
        for (i, count) in counts.iter_mut().enumerate() {
            let offset = i * 4;
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            let c = u32::from_le_bytes(bytes);
            if c == 0 {
                return false; // Laplace prior requires all counts >= 1
            }
            if c > MAX_COUNT {
                return false; // Reject adversarially skewed distributions
            }
            *count = c;
            total += c as u64;
        }
        if total > u32::MAX as u64 {
            return false; // Would overflow u32 total
        }
        self.counts = counts;
        self.total = total as u32;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_is_uniform() {
        let mut model = Order0Model::new();
        let probs = model.predict();

        // All probabilities should be equal (1/256)
        let expected = 1.0 / 256.0;
        for &p in &probs {
            assert!((p - expected).abs() < 1e-6);
        }

        // Sum should be ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn prediction_adapts_after_updates() {
        let mut model = Order0Model::new();

        // Feed a bunch of 'A's
        for _ in 0..1000 {
            model.update(b'A');
        }

        let probs = model.predict();
        // 'A' should have much higher probability than other bytes
        let p_a = probs[b'A' as usize];
        let p_other = probs[0]; // 0x00, which was never fed
        assert!(p_a > p_other * 10.0, "P('A') = {p_a}, P(0x00) = {p_other}");
    }

    #[test]
    fn prediction_always_valid_distribution() {
        let mut model = Order0Model::new();

        for byte in 0..=255u8 {
            let probs = model.predict();

            // All positive
            for &p in &probs {
                assert!(p > 0.0, "All probabilities must be > 0");
            }

            // Sum to ~1.0
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "Sum = {sum}, expected ~1.0");

            model.update(byte);
        }
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut model = Order0Model::new();

        for _ in 0..500 {
            model.update(42);
        }

        model.reset();
        let probs = model.predict();
        let expected = 1.0 / 256.0;
        for &p in &probs {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    /// `ExactDivisor` must agree with hardware division across the whole
    /// operand range the CDF sweep can produce, including the boundaries
    /// where the magic-number bound is tightest.
    #[test]
    fn exact_divisor_matches_hardware_division() {
        use crate::coding::rans::PROB_TOTAL;

        let totals = [
            256u32,
            257,
            1_000,
            32_767,
            32_768,
            32_769,
            65_536,
            999_999,
            1_000_001,
            MAX_MAGIC_TOTAL,
            MAX_MAGIC_TOTAL + 1, // falls back to hardware divide
        ];

        for total in totals {
            let divisor = ExactDivisor::new(total);
            let half = (total / 2) as u64;
            // `cum` ranges over [0, total]; sample the ends densely and the
            // middle on a stride, since the operand is monotone in `cum`.
            let mut cums: Vec<u64> = (0..=64u64).collect();
            cums.extend((0..=64u64).map(|k| total as u64 - k.min(total as u64)));
            cums.extend((0..512u64).map(|k| k * (total as u64) / 512));
            for cum in cums {
                let cum = cum.min(total as u64);
                let n = cum * PROB_TOTAL as u64 + half;
                assert_eq!(
                    divisor.div(n),
                    n / total as u64,
                    "ExactDivisor mismatch: n={n}, total={total}",
                );
            }
        }
    }

    /// Differential guard for the encode-side `query_cdf` fast path.
    ///
    /// The decoder always calls `predict_cdf`, so any disagreement between
    /// the two silently desynchronises the range coder. This walks a set of
    /// count distributions chosen to exercise both `query_cdf` regimes —
    /// including heavily skewed ones where `predict_cdf`'s monotonicity
    /// fix-up overshoots and it rebuilds the table via `probs_to_cdf`.
    #[test]
    fn query_cdf_matches_predict_cdf_on_skewed_counts() {
        // (dominant symbol, repetitions) — large repetition counts push
        // `total` far past PROB_TOTAL, which is what collapses the rounded
        // gaps of the rare symbols to zero.
        let shapes: [(u8, usize); 6] = [
            (0, 100),
            (0, 32_000),
            (7, 200_000),
            (255, 500_000),
            (128, 900_000),
            (3, 1_200_000), // crosses the rescale threshold
        ];

        for (dominant, reps) in shapes {
            let mut model = Order0Model::new();
            for i in 0..reps {
                // Mostly the dominant symbol, with a thin tail of others so
                // the distribution is skewed rather than degenerate.
                if i % 4096 == 0 {
                    model.update(((i / 4096) % 256) as u8);
                } else {
                    model.update(dominant);
                }
            }

            let reference = model.predict_cdf();
            for symbol in 0..=255u8 {
                let (lo, hi) = model.query_cdf(symbol);
                let s = symbol as usize;
                assert_eq!(
                    (lo, hi),
                    (reference[s], reference[s + 1]),
                    "query_cdf disagrees with predict_cdf for symbol {symbol}                      (dominant={dominant}, reps={reps}, total={})",
                    model.total,
                );
            }
        }
    }

    /// Round-trip guard at the range-coder level: encode with `query_cdf`
    /// (what `rans::encode_block` uses) and decode with `predict_cdf`.
    #[test]
    fn skewed_stream_round_trips_through_range_coder() {
        // A float-exponent-like plane: one dominant value, a handful of
        // neighbours, and a long thin tail. This is the shape that broke
        // byte-plane decoding before the `query_cdf` overshoot fix.
        let mut data = Vec::with_capacity(200_000);
        let mut x: u32 = 0x1234_5678;
        for _ in 0..200_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let r = x >> 20; // 12 bits
            data.push(match r {
                0..=3800 => 130,
                3801..=4000 => 129,
                4001..=4050 => 131,
                _ => (r % 256) as u8,
            });
        }

        let mut encoder = Order0Model::new();
        let encoded = crate::coding::rans::encode_block(&data, &mut encoder).unwrap();

        let mut decoder = Order0Model::new();
        let decoded =
            crate::coding::rans::decode_block(&encoded, data.len(), &mut decoder).unwrap();

        assert_eq!(decoded, data, "Order0 range-coder round-trip mismatch");
    }

    #[test]
    fn rescaling_prevents_overflow() {
        let mut model = Order0Model::new();

        // Push past the rescaling threshold
        for _ in 0..2_000_000 {
            model.update(0);
        }

        // Model should still produce valid probabilities
        let probs = model.predict();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "Sum after rescaling = {sum}");

        // Byte 0 should dominate
        assert!(probs[0] > 0.9);
    }
}
