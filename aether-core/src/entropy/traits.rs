//! Core abstraction for byte-level probability prediction.
//!
//! Every predictor — from the simple order-0 model to the neural SSM —
//! implements [`ProbabilityPredictor`]. The range coder consumes
//! the predictions without knowing which predictor produced them.

use crate::format::PredictorId;

/// A byte-level probability predictor.
///
/// The contract:
/// 1. Call [`predict`](Self::predict) to get P(next_byte) — a 256-element probability distribution.
/// 2. The caller (encoder) encodes the actual byte using this distribution.
/// 3. Call [`update`](Self::update) with the actual byte to advance the predictor's state.
/// 4. Repeat for every byte in the block.
///
/// The **decompressor** runs the exact same predict/update sequence, so
/// it reconstructs the identical probability distributions and can decode.
///
/// # Determinism
///
/// Implementations **must** be deterministic: given the same sequence of
/// `update` calls, `predict` must always return bit-identical results,
/// regardless of platform. This is the fundamental correctness requirement.
pub trait ProbabilityPredictor: Send {
    /// Produce a probability distribution over the next byte (256 values).
    ///
    /// The returned array should sum to approximately 1.0. All entries must
    /// be positive (≥ some small epsilon) to ensure every byte is encodable.
    fn predict(&mut self) -> [f32; 256];

    /// Return a 15-bit cumulative frequency table for the next byte.
    ///
    /// `cdf[0] == 0`, `cdf[256] == 32768`, and `cdf[i+1] > cdf[i]` for all `i`.
    /// The default converts from [`predict`](Self::predict) via [`crate::coding::rans::probs_to_cdf`].
    /// Predictors can override for speed (e.g. computing CDF from integer counts).
    fn predict_cdf(&mut self) -> [u16; 257] {
        let probs = self.predict();
        crate::coding::rans::probs_to_cdf(&probs)
    }

    /// Return `(cdf[byte], cdf[byte+1])` — the only two CDF entries the range
    /// coder needs to encode `byte`.
    ///
    /// **Encode-only fast path.** The decoder must scan the full CDF (it has
    /// a codeword in `[0, PROB_TOTAL)` and must find which symbol's interval
    /// contains it), so this method is only useful on the compress side.
    ///
    /// The default implementation calls [`predict_cdf`](Self::predict_cdf) and
    /// slices, so existing predictors continue working unchanged. Predictors
    /// where building the full 257-entry table is expensive (e.g. predictors
    /// that maintain integer counts or do per-byte mixing) should override
    /// this for an O(1) or O(symbol_count) fast path that skips materialising
    /// the 254 entries the encoder never reads.
    ///
    /// The contract must match `predict_cdf`: after a `query_cdf(byte)` call,
    /// the next `update(byte)` must advance state in a way that is
    /// bit-identical to what `predict_cdf()` + `update(byte)` would produce
    /// at the corresponding `(cdf[byte], cdf[byte+1])` pair. This is what
    /// keeps encode/decode in lockstep — the decoder still uses
    /// `predict_cdf`, so the two paths must agree on the same CDF.
    fn query_cdf(&mut self, byte: u8) -> (u16, u16) {
        let cdf = self.predict_cdf();
        let s = byte as usize;
        (cdf[s], cdf[s + 1])
    }

    /// Feed a confirmed byte to update internal state.
    ///
    /// Called after encoding (compressor) or decoding (decompressor) each byte.
    fn update(&mut self, byte: u8);

    /// Reset predictor state to initial conditions.
    ///
    /// Called at block boundaries so each block can be decompressed independently.
    fn reset(&mut self);

    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Identifier stored in the archive header.
    fn predictor_id(&self) -> PredictorId;

    /// Serialize the predictor's learned state for dictionary pretraining.
    ///
    /// Returns `None` if this predictor does not support serialization.
    /// The returned bytes can be passed to a new instance via [`load_state`](Self::load_state)
    /// to initialize it with pretrained weights.
    fn save_state(&self) -> Option<Vec<u8>> {
        None
    }

    /// Load pretrained state from bytes previously returned by [`save_state`](Self::save_state).
    ///
    /// Returns `true` if state was successfully loaded, `false` if this
    /// predictor does not support serialization or the data is invalid.
    fn load_state(&mut self, _data: &[u8]) -> bool {
        false
    }

    /// Stage A: the per-block coding baseline this predictor will reset to,
    /// if a dictionary has been installed as a baseline. Used by the router
    /// to propagate the baseline onto the internal coding predictors it
    /// builds per chunk (e.g. the BWT path's NeuralSSM). Returns `None` when
    /// no baseline is set or the predictor doesn't support dictionaries.
    fn coding_baseline(&self) -> Option<&[u8]> {
        None
    }

    /// Stage A: install a dictionary state as this predictor's per-block
    /// reset baseline. Returns `true` if applied. Default: unsupported.
    fn set_coding_baseline(&mut self, _state: &[u8]) -> bool {
        false
    }
}

/// Validate that a probability distribution is well-formed for use with the
/// range coder. Returns `Ok(())` if valid, or `Err(reason)` if not.
///
/// This is primarily intended for use in fuzz targets and debug assertions.
/// A valid distribution must:
/// - Have all entries > 0 (every byte must be encodable)
/// - Sum to approximately 1.0 (within `tolerance`)
/// - Contain no NaN or infinite values
pub fn validate_distribution(probs: &[f32; 256], tolerance: f32) -> Result<(), &'static str> {
    let mut sum = 0.0f64;
    for &p in probs.iter() {
        if !p.is_finite() {
            return Err("distribution contains NaN or infinity");
        }
        if p <= 0.0 {
            return Err("distribution contains non-positive probability");
        }
        sum += p as f64;
    }
    if (sum - 1.0).abs() > tolerance as f64 {
        return Err("distribution does not sum to ~1.0");
    }
    Ok(())
}

/// Fuzz-friendly helper: feed arbitrary bytes through a predictor and verify
/// every prediction is a valid distribution. Panics on the first invalid
/// distribution (suitable for use with `cargo-fuzz`).
///
/// # Example (in a fuzz target)
/// ```ignore
/// fuzz_target!(|data: &[u8]| {
///     let mut pred = Order0Model::new();
///     fuzz_predict_update(&mut pred, data);
/// });
/// ```
pub fn fuzz_predict_update(pred: &mut dyn ProbabilityPredictor, data: &[u8]) {
    for &byte in data {
        let probs = pred.predict();
        if let Err(reason) = validate_distribution(&probs, 0.02) {
            panic!("invalid distribution after byte {byte:#04x}: {reason}");
        }
        pred.update(byte);
    }
}

/// Fuzz-friendly helper: attempt to load arbitrary bytes as predictor state,
/// then verify the predictor still produces valid distributions.
///
/// Returns `true` if `load_state` accepted the data and all subsequent
/// predictions were valid; `false` if `load_state` correctly rejected it.
/// Panics if `load_state` accepted but subsequent predictions are invalid
/// (indicating a validation gap in `load_state`).
pub fn fuzz_load_state(
    pred: &mut dyn ProbabilityPredictor,
    state_data: &[u8],
    test_bytes: &[u8],
) -> bool {
    if !pred.load_state(state_data) {
        return false;
    }
    // State was accepted — verify predictions are still valid.
    for &byte in test_bytes {
        let probs = pred.predict();
        if let Err(reason) = validate_distribution(&probs, 0.02) {
            panic!(
                "load_state accepted data but predictor produces invalid distribution: {reason}"
            );
        }
        pred.update(byte);
    }
    true
}
