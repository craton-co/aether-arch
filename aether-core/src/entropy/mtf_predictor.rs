//! Predictor specialized for BWT+MTF output streams.
//!
//! After BWT+MTF, the data is a stream of small non-negative integers where:
//! - Value 0 dominates (repeated context byte, often ~60%)
//! - Runs of 0s are very common (same symbol repeating)
//! - Small values (0-10) account for ~90%+ of the data
//!
//! This predictor uses:
//! - Run-length context when in a zero run (captures run statistics)
//! - Order-1 context otherwise (captures transition patterns)
//! - Order-0 fallback for the first byte
//! - Virtual pseudocounts (alpha) for fast adaptation

use super::traits::ProbabilityPredictor;
use crate::format::PredictorId;

/// Order-1 contexts (256 possible previous byte values).
const ORDER1_SIZE: usize = 256;
/// Run-length buckets: 0, 1, 2, 3, 4, 5, 6, 7-15, 16-31, 32-63, 64+
const RUN_BUCKETS: usize = 11;
/// Rescale threshold for u16 count tables (per-entry).
const RESCALE_THRESHOLD: u16 = 16000;
/// Rescale threshold for u32 totals — prevents overflow even when no single
/// entry reaches `RESCALE_THRESHOLD` (e.g. uniform input across all 256 symbols).
const TOTAL_RESCALE_THRESHOLD: u32 = 1_000_000;
/// Pseudocount per symbol during prediction.
const ALPHA: f32 = 0.1;
/// Total virtual pseudocount mass across all 256 symbols.
const ALPHA_TOTAL: f32 = 256.0 * ALPHA;

/// Map a zero-run length to a bucket index (0..RUN_BUCKETS).
#[inline]
fn run_bucket(run_len: u32) -> usize {
    match run_len {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 5,
        6 => 6,
        7..=15 => 7,
        16..=31 => 8,
        32..=63 => 9,
        _ => 10,
    }
}

/// MTF-aware predictor for BWT+MTF preprocessed data.
///
/// Uses run-length context for zero runs and order-1 for transitions.
/// Memory: ~136 KiB.
pub struct MtfPredictor {
    /// Run-length context: run_bucket → counts per byte value.
    /// Used when the previous byte was 0 (in a potential zero run).
    run_ctx: Vec<[u16; 256]>,
    /// Cached totals per run context.
    run_ctx_totals: Vec<u32>,
    /// Order-1 table: prev_byte → counts per byte value.
    /// Used when the previous byte was non-zero.
    order1: Vec<[u16; 256]>,
    /// Cached totals per order-1 context.
    order1_totals: Vec<u32>,
    /// Order-0 fallback counts (u16 to match run_ctx/order1 tables).
    order0: [u16; 256],
    order0_total: u32,
    /// Current zero run length.
    run_len: u32,
    /// Previous byte.
    prev: u8,
    /// Whether we've seen at least one byte.
    has_context: bool,
}

impl MtfPredictor {
    pub fn new() -> Self {
        Self {
            run_ctx: vec![[0u16; 256]; RUN_BUCKETS],
            run_ctx_totals: vec![0u32; RUN_BUCKETS],
            order1: vec![[0u16; 256]; ORDER1_SIZE],
            order1_totals: vec![0u32; ORDER1_SIZE],
            order0: [0u16; 256],
            order0_total: 0,
            run_len: 0,
            prev: 0,
            has_context: false,
        }
    }

    #[inline]
    fn predict_from(entry: &[u16; 256], total: u32) -> [f32; 256] {
        let denom = total as f32 + ALPHA_TOTAL;
        let mut probs = [0.0f32; 256];
        for i in 0..256 {
            probs[i] = (entry[i] as f32 + ALPHA) / denom;
        }
        probs
    }
}

impl Default for MtfPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityPredictor for MtfPredictor {
    fn predict(&mut self) -> [f32; 256] {
        if !self.has_context {
            // Order-0 fallback for the first byte (u16 counts, same as other tables)
            let denom = self.order0_total as f32 + ALPHA_TOTAL;
            let mut probs = [0.0f32; 256];
            for (i, prob) in probs.iter_mut().enumerate() {
                *prob = (self.order0[i] as f32 + ALPHA) / denom;
            }
            return probs;
        }

        if self.prev == 0 {
            // In a zero run — use run-length context
            let bucket = run_bucket(self.run_len);
            Self::predict_from(&self.run_ctx[bucket], self.run_ctx_totals[bucket])
        } else {
            // After a non-zero byte — use order-1
            let ctx = self.prev as usize;
            Self::predict_from(&self.order1[ctx], self.order1_totals[ctx])
        }
    }

    fn update(&mut self, byte: u8) {
        let b = byte as usize;

        // Update the model that was used for prediction
        if !self.has_context {
            // First byte — update order-0 only
        } else if self.prev == 0 {
            // Was in run context
            let bucket = run_bucket(self.run_len);
            let entry = &mut self.run_ctx[bucket];
            entry[b] = entry[b].saturating_add(1);
            self.run_ctx_totals[bucket] += 1;
            if entry[b] >= RESCALE_THRESHOLD
                || self.run_ctx_totals[bucket] >= TOTAL_RESCALE_THRESHOLD
            {
                self.run_ctx_totals[bucket] = 0;
                for c in entry.iter_mut() {
                    *c >>= 1;
                    self.run_ctx_totals[bucket] += *c as u32;
                }
            }
        } else {
            // Was in order-1 context
            let ctx = self.prev as usize;
            let entry = &mut self.order1[ctx];
            entry[b] = entry[b].saturating_add(1);
            self.order1_totals[ctx] += 1;
            if entry[b] >= RESCALE_THRESHOLD || self.order1_totals[ctx] >= TOTAL_RESCALE_THRESHOLD {
                self.order1_totals[ctx] = 0;
                for c in entry.iter_mut() {
                    *c >>= 1;
                    self.order1_totals[ctx] += *c as u32;
                }
            }
        }

        // Update order-0 always (u16 counts, consistent with run_ctx/order1)
        self.order0[b] = self.order0[b].saturating_add(1);
        self.order0_total += 1;
        if self.order0[b] >= RESCALE_THRESHOLD || self.order0_total >= TOTAL_RESCALE_THRESHOLD {
            self.order0_total = 0;
            for c in self.order0.iter_mut() {
                *c >>= 1;
                self.order0_total += *c as u32;
            }
        }

        // Update run counter
        if byte == 0 {
            self.run_len = self.run_len.saturating_add(1);
        } else {
            self.run_len = 0;
        }

        self.prev = byte;
        self.has_context = true;
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn name(&self) -> &str {
        "mtf"
    }

    fn predictor_id(&self) -> PredictorId {
        PredictorId::Mtf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_uniform_then_adapts() {
        let mut pred = MtfPredictor::new();
        let probs = pred.predict();
        // First prediction should be ~uniform
        let diff = (probs[0] - probs[100]).abs();
        assert!(diff < 0.001, "Initial should be ~uniform, diff={diff}");

        // Feed enough data to populate run-length buckets
        for _ in 0..200 {
            pred.predict();
            pred.update(0);
        }
        let probs = pred.predict();
        assert!(probs[0] > 0.3, "After 200 zeros, p(0) > 0.3: {}", probs[0]);
    }

    #[test]
    fn learns_runs_of_zeros() {
        let mut pred = MtfPredictor::new();
        for _ in 0..100 {
            pred.predict();
            pred.update(0);
        }
        let probs = pred.predict();
        assert!(
            probs[0] > 0.5,
            "After 100 zeros, p(0) > 0.5, got {}",
            probs[0]
        );
    }

    #[test]
    fn run_context_improves_with_length() {
        let mut pred = MtfPredictor::new();
        // Feed pattern: runs of 10 zeros followed by value 3
        for _ in 0..100 {
            for _ in 0..10 {
                pred.predict();
                pred.update(0);
            }
            pred.predict();
            pred.update(3);
        }
        // After many such patterns, p(0) should be very high during a run
        for _ in 0..5 {
            pred.predict();
            pred.update(0);
        }
        let probs = pred.predict();
        assert!(
            probs[0] > 0.7,
            "Mid-run p(0) should be high, got {}",
            probs[0]
        );
    }

    #[test]
    fn roundtrip_with_range_coder() {
        use crate::coding::rans;

        let mtf_data: Vec<u8> = (0..1000)
            .map(|i| match i % 10 {
                0..=6 => 0,
                7 => 1,
                8 => 2,
                _ => (i % 20) as u8,
            })
            .collect();

        let mut enc = MtfPredictor::new();
        let compressed = rans::encode_block(&mtf_data, &mut enc).unwrap();

        let mut dec = MtfPredictor::new();
        let decoded = rans::decode_block(&compressed, mtf_data.len(), &mut dec).unwrap();

        assert_eq!(mtf_data, decoded);
    }
}
