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
