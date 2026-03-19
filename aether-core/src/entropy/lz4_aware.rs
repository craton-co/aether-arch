//! LZ4/LZ77-aware predictor that exploits the structure of LZ byte streams.
//!
//! Instead of treating all bytes uniformly, this predictor maintains a finite
//! state machine (FSM) that tracks position within the LZ token format:
//! size prefix → token → \[lit_len_ext\] → literals → match_offset → \[match_len_ext\] → token → ...
//!
//! Each FSM state has a specialized sub-predictor tuned for the byte patterns
//! in that position. The key insight is that the Literals sub-predictor maintains
//! its own context buffer of *only literal bytes* (not interleaved LZ control
//! bytes), giving it clean multi-order context on the actual data.
//!
//! The literal predictor uses order-1 through order-3 context tables with
//! adaptive multiplicative mixing for best compression.

use super::traits::ProbabilityPredictor;
use crate::format::PredictorId;

// ── FSM States ──────────────────────────────────────────────────────────────

/// Position within the LZ byte stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lz4ParseState {
    /// First 4 bytes: LE u32 uncompressed size (from compress_prepend_size).
    SizePrefix { byte_index: u8 },

    /// The token byte: high nibble = literal_length, low nibble = match_length - N.
    Token,

    /// Literal length extension bytes (when token high nibble == 15).
    LitLenExt,

    /// Literal data bytes (the actual uncompressed data embedded in LZ output).
    Literals { remaining: usize },

    /// Match offset low byte (first of 2-byte LE u16).
    MatchOffsetLow,

    /// Match offset high byte (second of 2-byte LE u16).
    MatchOffsetHigh,

    /// Match length extension bytes (when token low nibble == 15).
    MatchLenExt,
}

// ── Sub-predictor helpers ───────────────────────────────────────────────────

/// Simple order-0 frequency model.
struct FreqModel {
    counts: [u32; 256],
    total: u32,
}

impl FreqModel {
    fn new() -> Self {
        Self {
            counts: [1; 256],
            total: 256,
        }
    }

    fn predict(&self) -> [f32; 256] {
        let inv = 1.0 / self.total as f32;
        let mut probs = [0.0f32; 256];
        for (i, prob) in probs.iter_mut().enumerate() {
            *prob = self.counts[i] as f32 * inv;
        }
        probs
    }

    fn update(&mut self, byte: u8) {
        self.counts[byte as usize] += 1;
        self.total += 1;
        if self.total > 500_000 {
            self.rescale();
        }
    }

    fn rescale(&mut self) {
        self.total = 0;
        for c in self.counts.iter_mut() {
            *c = (*c >> 1).max(1);
            self.total += *c;
        }
    }

    fn reset(&mut self) {
        self.counts = [1; 256];
        self.total = 256;
    }
}

/// FNV-1a hash of context bytes.
///
/// Uses the standard FNV-1a algorithm with fixed constants, which is
/// deterministic across all Rust versions and platforms (unlike
/// `DefaultHasher` whose algorithm and seed are not guaranteed stable).
#[inline]
fn hash_context(data: &[u8]) -> usize {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h as usize
}

// ── Constants ───────────────────────────────────────────────────────────────

/// Hash table bits for literal order-1 model (16K entries = ~8 MiB).
const LIT_ORDER1_BITS: usize = 14;
/// Hash table bits for literal order-2 model (16K entries = ~8 MiB).
const LIT_ORDER2_BITS: usize = 14;
/// Hash table bits for literal order-3 model (8K entries = ~4 MiB).
const LIT_ORDER3_BITS: usize = 13;
/// Token context table size (keyed on previous token).
const TOKEN_TABLE_SIZE: usize = 256;
/// Max rescale threshold for u16 hash table entries.
const HASH_TABLE_RESCALE: u16 = 8000;
/// Number of context bytes to keep for literal predictor.
const LIT_CONTEXT_LEN: usize = 4;
/// Maximum literal length to prevent FSM from staying in Literals state
/// for an unreasonable number of bytes on malformed input. LZ4 frames
/// are limited to ~4 GiB, so 256 MiB is a generous upper bound.
const MAX_LIT_LEN: usize = 256 * 1024 * 1024;

// ── Adaptive Mixer ─────────────────────────────────────────────────────────

/// Multiplicative weight mixer for literal order models.
struct LiteralMixer {
    weights: [f64; 3],
}

impl LiteralMixer {
    fn new() -> Self {
        Self {
            // Bias toward higher orders initially
            weights: [1.0, 3.0, 6.0],
        }
    }

    /// Mix up to 3 predictions (order-1, order-2, order-3).
    fn mix(&self, preds: &[Option<[f32; 256]>; 3]) -> [f32; 256] {
        let mut result = [0.0f32; 256];
        let mut w_sum = 0.0f64;

        for (i, pred) in preds.iter().enumerate() {
            if let Some(p) = pred {
                w_sum += self.weights[i];
                for j in 0..256 {
                    result[j] += (self.weights[i] as f32) * p[j];
                }
            }
        }

        if w_sum > 0.0 {
            let inv = 1.0 / w_sum as f32;
            for item in result.iter_mut() {
                *item *= inv;
            }
            result
        } else {
            [1.0 / 256.0; 256]
        }
    }

    /// Update weights based on how well each model predicted the actual byte.
    fn update(&mut self, preds: &[Option<[f32; 256]>; 3], actual: u8) {
        for (i, pred) in preds.iter().enumerate() {
            if let Some(p) = pred {
                let score = p[actual as usize] as f64;
                self.weights[i] *= score.max(1e-8);
            }
        }

        // Always normalize to prevent precision loss from accumulated
        // multiplicative updates. Without this, weights drift toward 0 and
        // lose relative precision in f64.
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            let inv = 1.0 / sum;
            for w in &mut self.weights {
                *w *= inv;
            }
        } else {
            // Weights collapsed — reset to equal.
            let n = self.weights.len() as f64;
            for w in &mut self.weights {
                *w = 1.0 / n;
            }
        }
    }

    fn reset(&mut self) {
        self.weights = [1.0, 3.0, 6.0];
    }
}

// ── Main Predictor ──────────────────────────────────────────────────────────

/// LZ-aware predictor using FSM-guided specialized sub-models.
///
/// Total memory: ~20 MiB (dominated by literal hash tables).
pub struct Lz4AwarePredictor {
    // ── FSM state ──
    state: Lz4ParseState,
    lit_len_accum: usize,
    match_len_nibble: u8,
    last_token: u8,

    // ── Sub-predictors ──
    /// Size prefix: 4 independent order-0 models (one per byte position).
    size_models: [FreqModel; 4],

    /// Token: order-1 model keyed on previous token byte.
    token_table: Vec<[u16; 256]>,

    /// Literal length extension: order-0 model.
    lit_ext_model: FreqModel,

    /// Literal data: order-1 hash table.
    lit_order1: Vec<[u16; 256]>,
    /// Literal data: order-2 hash table.
    lit_order2: Vec<[u16; 256]>,
    /// Literal data: order-3 hash table.
    lit_order3: Vec<[u16; 256]>,
    /// Context buffer: only literal bytes (not control bytes).
    lit_context: std::collections::VecDeque<u8>,
    /// Adaptive mixer for literal sub-models.
    lit_mixer: LiteralMixer,
    /// Cached literal predictions from predict_literal(), reused in update_literal().
    cached_lit_preds: [Option<[f32; 256]>; 3],
    /// True after predict() has been called and before update() consumes it.
    /// Prevents update() from using stale cached predictions.
    predict_called: bool,

    /// Match offset low byte: order-0 model.
    offset_low_model: FreqModel,
    /// Match offset high byte: order-0 model.
    offset_high_model: FreqModel,

    /// Match length extension: order-0 model.
    match_ext_model: FreqModel,
}

impl Lz4AwarePredictor {
    pub fn new() -> Self {
        let lit_order1_size = 1 << LIT_ORDER1_BITS;
        let lit_order2_size = 1 << LIT_ORDER2_BITS;
        let lit_order3_size = 1 << LIT_ORDER3_BITS;

        Self {
            state: Lz4ParseState::SizePrefix { byte_index: 0 },
            lit_len_accum: 0,
            match_len_nibble: 0,
            last_token: 0,

            size_models: [
                FreqModel::new(),
                FreqModel::new(),
                FreqModel::new(),
                FreqModel::new(),
            ],
            token_table: vec![[1u16; 256]; TOKEN_TABLE_SIZE],
            lit_ext_model: FreqModel::new(),
            lit_order1: vec![[1u16; 256]; lit_order1_size],
            lit_order2: vec![[1u16; 256]; lit_order2_size],
            lit_order3: vec![[1u16; 256]; lit_order3_size],
            lit_context: std::collections::VecDeque::with_capacity(LIT_CONTEXT_LEN + 1),
            lit_mixer: LiteralMixer::new(),
            cached_lit_preds: [None; 3],
            offset_low_model: FreqModel::new(),
            offset_high_model: FreqModel::new(),
            match_ext_model: FreqModel::new(),
            predict_called: false,
        }
    }

    // ── Predict methods per state ──

    fn predict_size_prefix(&self, idx: usize) -> [f32; 256] {
        self.size_models[idx].predict()
    }

    fn predict_token(&self) -> [f32; 256] {
        let entry = &self.token_table[self.last_token as usize];
        let total: u32 = entry.iter().map(|&c| c as u32).sum();
        let inv = 1.0 / total as f32;
        let mut probs = [0.0f32; 256];
        for i in 0..256 {
            probs[i] = entry[i] as f32 * inv;
        }
        probs
    }

    fn hash_entry_predict(table: &[[u16; 256]], hash_bits: usize, ctx: &[u8]) -> [f32; 256] {
        let hash = hash_context(ctx);
        let entry = &table[hash & ((1 << hash_bits) - 1)];
        let total: u32 = entry.iter().map(|&c| c as u32).sum();
        let inv = 1.0 / total as f32;
        let mut p = [0.0f32; 256];
        for i in 0..256 {
            p[i] = entry[i] as f32 * inv;
        }
        p
    }

    fn predict_literal(&mut self) -> [f32; 256] {
        let ctx = self.lit_context.make_contiguous();
        let ctx_len = ctx.len();

        let o1 = if ctx_len >= 1 {
            Some(Self::hash_entry_predict(
                &self.lit_order1,
                LIT_ORDER1_BITS,
                &ctx[ctx_len - 1..],
            ))
        } else {
            None
        };

        let o2 = if ctx_len >= 2 {
            Some(Self::hash_entry_predict(
                &self.lit_order2,
                LIT_ORDER2_BITS,
                &ctx[ctx_len - 2..],
            ))
        } else {
            None
        };

        let o3 = if ctx_len >= 3 {
            Some(Self::hash_entry_predict(
                &self.lit_order3,
                LIT_ORDER3_BITS,
                &ctx[ctx_len - 3..],
            ))
        } else {
            None
        };

        // Cache predictions for reuse in update_literal()
        self.cached_lit_preds = [o1, o2, o3];
        self.lit_mixer.mix(&[o1, o2, o3])
    }

    // ── Update methods per state ──

    fn update_token(&mut self, byte: u8) {
        let entry = &mut self.token_table[self.last_token as usize];
        entry[byte as usize] = entry[byte as usize].saturating_add(1);
        if entry[byte as usize] >= HASH_TABLE_RESCALE {
            for c in entry.iter_mut() {
                *c = (*c >> 1).max(1);
            }
        }
    }

    fn hash_entry_update(table: &mut [[u16; 256]], hash_bits: usize, ctx: &[u8], byte: u8) {
        let hash = hash_context(ctx);
        let idx = hash & ((1 << hash_bits) - 1);
        let entry = &mut table[idx];
        entry[byte as usize] = entry[byte as usize].saturating_add(1);
        if entry[byte as usize] >= HASH_TABLE_RESCALE {
            for c in entry.iter_mut() {
                *c = (*c >> 1).max(1);
            }
        }
    }

    fn update_literal(&mut self, byte: u8) {
        // Reuse predictions cached by predict_literal() only if predict() was
        // called first (prevents stale cache from a previous predict/update cycle).
        let preds = if self.predict_called {
            std::mem::take(&mut self.cached_lit_preds)
        } else {
            [None; 3]
        };
        let ctx = self.lit_context.make_contiguous();
        let ctx_len = ctx.len();

        let [o1, o2, o3] = if preds[0].is_some() || preds[1].is_some() || preds[2].is_some() {
            preds
        } else {
            // Fallback: compute predictions if cache was empty
            let o1 = if ctx_len >= 1 {
                Some(Self::hash_entry_predict(
                    &self.lit_order1,
                    LIT_ORDER1_BITS,
                    &ctx[ctx_len - 1..],
                ))
            } else {
                None
            };
            let o2 = if ctx_len >= 2 {
                Some(Self::hash_entry_predict(
                    &self.lit_order2,
                    LIT_ORDER2_BITS,
                    &ctx[ctx_len - 2..],
                ))
            } else {
                None
            };
            let o3 = if ctx_len >= 3 {
                Some(Self::hash_entry_predict(
                    &self.lit_order3,
                    LIT_ORDER3_BITS,
                    &ctx[ctx_len - 3..],
                ))
            } else {
                None
            };
            [o1, o2, o3]
        };

        // Update mixer weights
        self.lit_mixer.update(&[o1, o2, o3], byte);

        // Update order-1 table
        if ctx_len >= 1 {
            Self::hash_entry_update(
                &mut self.lit_order1,
                LIT_ORDER1_BITS,
                &ctx[ctx_len - 1..],
                byte,
            );
        }

        // Update order-2 table
        if ctx_len >= 2 {
            Self::hash_entry_update(
                &mut self.lit_order2,
                LIT_ORDER2_BITS,
                &ctx[ctx_len - 2..],
                byte,
            );
        }

        // Update order-3 table
        if ctx_len >= 3 {
            Self::hash_entry_update(
                &mut self.lit_order3,
                LIT_ORDER3_BITS,
                &ctx[ctx_len - 3..],
                byte,
            );
        }

        // Update literal context buffer (VecDeque: O(1) push_back + pop_front)
        self.lit_context.push_back(byte);
        if self.lit_context.len() > LIT_CONTEXT_LEN {
            self.lit_context.pop_front();
        }
    }

    // ── FSM transition ──

    fn next_state(&self, byte: u8) -> Lz4ParseState {
        match self.state {
            Lz4ParseState::SizePrefix { byte_index } => {
                if byte_index < 3 {
                    Lz4ParseState::SizePrefix {
                        byte_index: byte_index + 1,
                    }
                } else {
                    Lz4ParseState::Token
                }
            }
            Lz4ParseState::Token => {
                let lit_len_nibble = byte >> 4;
                if lit_len_nibble == 15 {
                    Lz4ParseState::LitLenExt
                } else if lit_len_nibble > 0 {
                    Lz4ParseState::Literals {
                        remaining: lit_len_nibble as usize,
                    }
                } else {
                    Lz4ParseState::MatchOffsetLow
                }
            }
            Lz4ParseState::LitLenExt => {
                if byte == 255 {
                    Lz4ParseState::LitLenExt
                } else {
                    // lit_len_accum includes the initial 15 from the token nibble
                    // plus all extension bytes. A value of 0 here means the stream
                    // is malformed; skip to MatchOffsetLow to avoid a zero-remaining
                    // Literals state that would desync predict/update.
                    if self.lit_len_accum > 0 {
                        Lz4ParseState::Literals {
                            remaining: self.lit_len_accum,
                        }
                    } else {
                        Lz4ParseState::MatchOffsetLow
                    }
                }
            }
            Lz4ParseState::Literals { remaining } => {
                if remaining > 1 {
                    Lz4ParseState::Literals {
                        remaining: remaining - 1,
                    }
                } else {
                    Lz4ParseState::MatchOffsetLow
                }
            }
            Lz4ParseState::MatchOffsetLow => Lz4ParseState::MatchOffsetHigh,
            Lz4ParseState::MatchOffsetHigh => {
                if self.match_len_nibble == 15 {
                    Lz4ParseState::MatchLenExt
                } else {
                    Lz4ParseState::Token
                }
            }
            Lz4ParseState::MatchLenExt => {
                if byte == 255 {
                    Lz4ParseState::MatchLenExt
                } else {
                    Lz4ParseState::Token
                }
            }
        }
    }
}

impl Default for Lz4AwarePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityPredictor for Lz4AwarePredictor {
    fn predict(&mut self) -> [f32; 256] {
        self.predict_called = true;
        match self.state {
            Lz4ParseState::SizePrefix { byte_index } => {
                self.predict_size_prefix(byte_index as usize)
            }
            Lz4ParseState::Token => self.predict_token(),
            Lz4ParseState::LitLenExt => self.lit_ext_model.predict(),
            Lz4ParseState::Literals { .. } => self.predict_literal(),
            Lz4ParseState::MatchOffsetLow => self.offset_low_model.predict(),
            Lz4ParseState::MatchOffsetHigh => self.offset_high_model.predict(),
            Lz4ParseState::MatchLenExt => self.match_ext_model.predict(),
        }
    }

    fn update(&mut self, byte: u8) {
        match self.state {
            Lz4ParseState::SizePrefix { byte_index } => {
                self.size_models[byte_index as usize].update(byte);
            }
            Lz4ParseState::Token => {
                self.update_token(byte);
                self.match_len_nibble = byte & 0x0F;
                let lit_len_nibble = byte >> 4;
                if lit_len_nibble == 15 {
                    self.lit_len_accum = 15;
                }
                self.last_token = byte;
            }
            Lz4ParseState::LitLenExt => {
                self.lit_ext_model.update(byte);
                self.lit_len_accum = self
                    .lit_len_accum
                    .saturating_add(byte as usize)
                    .min(MAX_LIT_LEN);
            }
            Lz4ParseState::Literals { .. } => {
                self.update_literal(byte);
            }
            Lz4ParseState::MatchOffsetLow => {
                self.offset_low_model.update(byte);
            }
            Lz4ParseState::MatchOffsetHigh => {
                self.offset_high_model.update(byte);
            }
            Lz4ParseState::MatchLenExt => {
                self.match_ext_model.update(byte);
            }
        }

        self.state = self.next_state(byte);
        self.predict_called = false;
    }

    fn reset(&mut self) {
        self.state = Lz4ParseState::SizePrefix { byte_index: 0 };
        self.lit_len_accum = 0;
        self.match_len_nibble = 0;
        self.last_token = 0;

        for m in &mut self.size_models {
            m.reset();
        }
        for entry in self.token_table.iter_mut() {
            *entry = [1u16; 256];
        }
        self.lit_ext_model.reset();
        for entry in self.lit_order1.iter_mut() {
            *entry = [1u16; 256];
        }
        for entry in self.lit_order2.iter_mut() {
            *entry = [1u16; 256];
        }
        for entry in self.lit_order3.iter_mut() {
            *entry = [1u16; 256];
        }
        self.lit_context.clear();
        self.lit_mixer.reset();
        self.cached_lit_preds = [None; 3];
        self.predict_called = false;
        self.offset_low_model.reset();
        self.offset_high_model.reset();
        self.match_ext_model.reset();
    }

    fn name(&self) -> &str {
        "lz4-aware"
    }

    fn predictor_id(&self) -> PredictorId {
        PredictorId::Lz4Aware
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_size_prefix() {
        let pred = Lz4AwarePredictor::new();
        assert_eq!(pred.state, Lz4ParseState::SizePrefix { byte_index: 0 });
    }

    #[test]
    fn fsm_transitions_size_prefix() {
        let mut pred = Lz4AwarePredictor::new();
        for i in 0..4u8 {
            pred.predict();
            pred.update(i);
        }
        assert_eq!(pred.state, Lz4ParseState::Token);
    }

    #[test]
    fn fsm_transitions_simple_token() {
        let mut pred = Lz4AwarePredictor::new();
        for i in 0..4u8 {
            pred.predict();
            pred.update(i);
        }
        assert_eq!(pred.state, Lz4ParseState::Token);

        pred.predict();
        pred.update(0x30);
        assert_eq!(pred.state, Lz4ParseState::Literals { remaining: 3 });

        for _ in 0..3 {
            pred.predict();
            pred.update(b'A');
        }
        assert_eq!(pred.state, Lz4ParseState::MatchOffsetLow);

        pred.predict();
        pred.update(0x05);
        assert_eq!(pred.state, Lz4ParseState::MatchOffsetHigh);

        pred.predict();
        pred.update(0x00);
        assert_eq!(pred.state, Lz4ParseState::Token);
    }

    #[test]
    fn fsm_transitions_extended_literal_length() {
        let mut pred = Lz4AwarePredictor::new();
        for i in 0..4u8 {
            pred.predict();
            pred.update(i);
        }

        pred.predict();
        pred.update(0xF4);
        assert_eq!(pred.state, Lz4ParseState::LitLenExt);
        assert_eq!(pred.lit_len_accum, 15);
        assert_eq!(pred.match_len_nibble, 4);

        pred.predict();
        pred.update(10);
        assert_eq!(pred.state, Lz4ParseState::Literals { remaining: 25 });
    }

    #[test]
    fn fsm_transitions_extended_match_length() {
        let mut pred = Lz4AwarePredictor::new();
        for i in 0..4u8 {
            pred.predict();
            pred.update(i);
        }

        pred.predict();
        pred.update(0x1F);
        assert_eq!(pred.state, Lz4ParseState::Literals { remaining: 1 });

        pred.predict();
        pred.update(b'X');
        assert_eq!(pred.state, Lz4ParseState::MatchOffsetLow);

        pred.predict();
        pred.update(0x01);
        pred.predict();
        pred.update(0x00);
        assert_eq!(pred.state, Lz4ParseState::MatchLenExt);

        pred.predict();
        pred.update(255);
        assert_eq!(pred.state, Lz4ParseState::MatchLenExt);

        pred.predict();
        pred.update(5);
        assert_eq!(pred.state, Lz4ParseState::Token);
    }

    #[test]
    fn fsm_parses_real_lz4_stream() {
        let input = b"Hello, world! Hello, world! Hello, world! Hello, world! \
                      The quick brown fox jumps over the lazy dog. \
                      The quick brown fox jumps over the lazy dog.";
        let lz4_data = lz4_flex::compress_prepend_size(input);

        let mut pred = Lz4AwarePredictor::new();
        pred.reset();

        for &byte in &lz4_data {
            let probs = pred.predict();
            assert!(
                probs.iter().all(|&p| p > 0.0),
                "All probabilities must be positive"
            );
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.05,
                "Probabilities should sum to ~1.0, got {sum}"
            );
            pred.update(byte);
        }
    }

    #[test]
    fn deterministic() {
        let input = b"Test data for determinism check. Test data for determinism check.";
        let lz4_data = lz4_flex::compress_prepend_size(input);

        let mut pred1 = Lz4AwarePredictor::new();
        let mut pred2 = Lz4AwarePredictor::new();

        for &byte in &lz4_data {
            let p1 = pred1.predict();
            let p2 = pred2.predict();
            assert_eq!(p1, p2, "Predictions must be identical");
            pred1.update(byte);
            pred2.update(byte);
        }
    }

    #[test]
    fn reset_restores_initial_state() {
        let input = b"Some data to feed through the predictor before reset.";
        let lz4_data = lz4_flex::compress_prepend_size(input);

        let mut pred = Lz4AwarePredictor::new();
        let initial_probs = pred.predict();

        for &byte in &lz4_data {
            pred.predict();
            pred.update(byte);
        }

        pred.reset();
        let after_reset_probs = pred.predict();
        assert_eq!(
            initial_probs, after_reset_probs,
            "Reset should restore initial prediction"
        );
        assert_eq!(pred.state, Lz4ParseState::SizePrefix { byte_index: 0 });
    }

    #[test]
    fn roundtrip_with_range_coder() {
        use crate::coding::rans;

        let input = b"Hello, world! Hello, world! Hello, world! \
                      This is a test of the LZ4-aware predictor with range coding.";
        let lz4_data = lz4_flex::compress_prepend_size(input);

        let mut enc_pred = Lz4AwarePredictor::new();
        let compressed =
            rans::encode_block(&lz4_data, &mut enc_pred).expect("encode should succeed");

        let mut dec_pred = Lz4AwarePredictor::new();
        let decoded = rans::decode_block(&compressed, lz4_data.len(), &mut dec_pred)
            .expect("decode should succeed");

        assert_eq!(lz4_data, decoded, "Range coding roundtrip failed");
    }

    #[test]
    fn compresses_lz4_stream() {
        use crate::coding::rans;

        let line = "The quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = line.as_bytes().repeat(100);
        let lz4_data = lz4_flex::compress_prepend_size(&input);

        let mut lz4_pred = Lz4AwarePredictor::new();
        let lz4_bytes = rans::encode_block(&lz4_data, &mut lz4_pred)
            .expect("encode with lz4-aware should succeed");
        let lz4_size = lz4_bytes.len();

        assert!(
            lz4_size < lz4_data.len(),
            "LZ4-aware predictor should compress the LZ4 stream: RC output {} >= LZ4 input {}",
            lz4_size,
            lz4_data.len()
        );
    }
}
