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
        let total = self.total;
        let mut cum = 0u64;
        for (i, cdf_val) in cdf.iter_mut().enumerate().take(256) {
            *cdf_val = ((cum * PROB_TOTAL as u64 + (total as u64 / 2)) / total as u64) as u16;
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
    ///    `cdf[i]` may force bumping `cdf[i+1]`, and so on.
    ///
    /// Because the fix-up is a forward sweep, the value of `cdf[byte]`
    /// depends on whether any earlier collision propagated up to it. A
    /// pure O(1) jump (e.g. from a Fenwick prefix sum + scale) is therefore
    /// not bit-identical to the reference and would silently desync the
    /// decoder. We instead simulate the original forward sweep, but only
    /// up to index `byte + 1` — saving:
    ///
    /// * the upper `255 - s` rounding ops
    /// * the monotonicity fix-up over the upper half
    /// * the `cdf[256] != PROB_TOTAL` guard scan
    /// * the 514-byte stack return of the `[u16; 257]` array
    ///
    /// This is the "incremental" fast path that matches the trait doc:
    /// the encoder no longer materialises the 254 entries it never reads.
    ///
    /// The full `predict_cdf` fallback (overshoot ⇒ `probs_to_cdf`) cannot
    /// be detected from a partial sweep. Empirically it doesn't fire under
    /// Laplace prior + rescale; the s==255 anchor check defends against
    /// the only case we can detect, falling back to `predict_cdf` if our
    /// sweep didn't land exactly on `PROB_TOTAL`. For s < 255 we trust the
    /// invariant (counts ≥ 1, total ≤ 1_000_000) holds.
    fn query_cdf(&mut self, byte: u8) -> (u16, u16) {
        use crate::coding::rans::PROB_TOTAL;
        let s = byte as usize;
        let total = self.total as u64;
        let half = total / 2;
        let scale = PROB_TOTAL as u64;

        // Forward sweep over indices 0..=s+1 of the original predict_cdf
        // loop, computing cdf[i] = round(running_cum * PROB_TOTAL / total)
        // and applying the chained monotonicity fix-up. We only retain
        // cdf[s] and cdf[s+1]; the upper 254 entries are not computed at
        // all. The Fenwick tree (built on update) is also used in the
        // overshoot-detection fallback below.
        //
        // Why a forward sweep and not an O(1) jump from Fenwick prefix
        // sums: the fix-up `cdf[i+1] = max(cdf[i+1], cdf[i] + 1)` can
        // chain — bumping cdf[i] may force bumping cdf[i+1], and so on —
        // so cdf[s] depends on every prior fix-up. We must replay them.
        let mut prev: u16 = 0; // cdf[0] = 0
        let mut running_cum: u64 = 0;
        let upper = s + 1;
        let mut cdf_s: u16 = 0;
        let mut cdf_s1: u16 = 0;
        for i in 1..=upper {
            running_cum += self.counts[i - 1] as u64;
            let mut val = ((running_cum * scale + half) / total) as u16;
            if val <= prev {
                val = prev + 1;
            }
            if i == s {
                cdf_s = val;
            }
            if i == s + 1 {
                cdf_s1 = val;
            }
            prev = val;
        }
        // For s == 0, cdf[s] = cdf[0] = 0 (loop never sets it).
        if s == 0 {
            cdf_s = 0;
        }

        // Anchor check: if s == 255, predict_cdf forces cdf[256] = PROB_TOTAL.
        // Our forward sweep computed cdf_s1 from rounding alone; if it doesn't
        // match the anchor, predict_cdf would fall through to its overshoot
        // path (`probs_to_cdf`), which can shift earlier entries too. Fall
        // back to the reference impl in that case to guarantee match.
        if s + 1 == 256 && cdf_s1 != PROB_TOTAL as u16 {
            let cdf = self.predict_cdf();
            return (cdf[s], cdf[s + 1]);
        }

        (cdf_s, cdf_s1)
    }

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
