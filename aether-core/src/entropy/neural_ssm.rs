//! Neural State Space Model predictor for byte-level probability prediction.
//!
//! Combines a diagonal linear SSM (long-range context) with an RlePredictor
//! baseline (immediate context). The SSM maintains a hidden state vector via
//! exponential moving averages at multiple time scales, then uses two online
//! logistic-regression classifiers for binary decisions:
//!
//! 1. Run symbol (0-1) vs literal (≥2)?
//! 2. RUNA (0) vs RUNB (1)?
//!
//! Literal value predictions are delegated to the RlePredictor's counting model.
//! An adaptive mixer blends SSM and RLE predictions based on recent performance,
//! so the SSM can only help (never hurt) relative to pure RlePredictor.
//!
//! Architecture:
//!   - Embedding: byte → D-dimensional vector (fixed, deterministic)
//!   - SSM update: h\[d\] = a\[d\] * h\[d\] + (1 - a\[d\]) * embed\[byte\]\[d\]  (EMA)
//!   - Binary classifiers: sigmoid(dot(w, h) + bias), trained via online SGD
//!   - Mixer: exponential moving average of per-expert log-likelihood
//!
//! Only 2*(D+1) = 66 learnable parameters (at D=32) → adapts rapidly from scratch.
//! Memory: ~25 KiB total (embedding table + SSM state + RlePredictor).

use super::rle_predictor::RlePredictor;
use super::traits::ProbabilityPredictor;
use crate::format::PredictorId;

/// Maximum SSM hidden state dimension (compile-time upper bound).
const D_MAX: usize = 32;

/// Minimum probability to ensure every byte is encodable.
const MIN_PROB: f32 = 1e-6;

/// Number of order-2 context buckets for literal prediction.
///
/// Stage B: raised from 8 (4 prev-classes × 2 prev_prev-classes) to 64
/// (8 × 8). On the BWT+MTF+RLE stream the "literals" are MTF ranks, which
/// cluster heavily near zero, so finer conditioning on the previous two
/// ranks sharpens the literal distribution. Pure integer counting — adds
/// no floating-point math, so cross-platform determinism is unaffected.
const NUM_O2_CTX: usize = 64;

/// Pseudocount per literal value in order-2 model.
const O2_ALPHA: f32 = 0.5;

/// Stage B confidence constant for the order-2 literal blend. The effective
/// blend weight is `o2_lit_blend * obs / (obs + O2_CONF_K)`, so a context
/// reaches half its target blend after ~`O2_CONF_K` observations. This lets
/// the finer (64-context) model defer to the global RLE distribution while a
/// context is still sparse — fixing the regression a fixed blend caused when
/// contexts were subdivided. Deterministic integer/f32 math only.
const O2_CONF_K: f32 = 32.0;

/// Default blend weight for order-2 literal model (0 = pure RLE, 1 = pure order-2).
/// Tuned via greedy sweep on Silesia corpus (202 MiB, 12 files). 0.3 wins over 0.1
/// on both the text subset (dickens+nci) and the internal 87 KiB corpus.
const DEFAULT_O2_LIT_BLEND: f32 = 0.3;

/// Default hyperparameters — retuned via greedy sweep on Silesia corpus (March 2026).
/// Previous defaults (D=20, lr=0.02, o2=0.1) were tuned on 87 KiB structured text only.
/// Silesia sweep findings: D=32 > D=20 on diverse data; lr=0.01 better for larger D;
/// o2=0.3 gives biggest gain (better literal context on real-world text diversity).
const DEFAULT_D: usize = 32;
const DEFAULT_LR: f32 = 0.01;
const DEFAULT_MIX_DECAY: f32 = 0.995;
const DEFAULT_WARMUP: u32 = 0;
const DEFAULT_MIX_SENSITIVITY: f32 = 100.0;
const DEFAULT_MAX_ALPHA: f32 = 0.9;
const DEFAULT_DECAY_LO: f32 = 0.5;
const DEFAULT_DECAY_HI: f32 = 0.999;

/// Configuration for NeuralSsmPredictor hyperparameters.
#[derive(Clone, Debug)]
pub struct NeuralSsmConfig {
    /// Hidden state dimension (1..=D_MAX).
    pub d: usize,
    /// Learning rate for SGD on binary classifiers.
    pub lr: f32,
    /// EMA decay for performance tracking.
    pub mix_decay: f32,
    /// Warmup steps before SSM can contribute.
    pub warmup: u32,
    /// Sensitivity of mixing weight to performance difference.
    pub mix_sensitivity: f32,
    /// Maximum mixing weight for SSM.
    pub max_alpha: f32,
    /// Lowest decay rate (shortest memory).
    pub decay_lo: f32,
    /// Highest decay rate (longest memory).
    pub decay_hi: f32,
    /// Blend weight for order-2 literal context model.
    pub o2_lit_blend: f32,
    /// Minimum observations before order-2 model contributes.
    pub o2_min_obs: u32,
}

impl Default for NeuralSsmConfig {
    fn default() -> Self {
        Self {
            d: DEFAULT_D,
            lr: DEFAULT_LR,
            mix_decay: DEFAULT_MIX_DECAY,
            warmup: DEFAULT_WARMUP,
            mix_sensitivity: DEFAULT_MIX_SENSITIVITY,
            max_alpha: DEFAULT_MAX_ALPHA,
            decay_lo: DEFAULT_DECAY_LO,
            decay_hi: DEFAULT_DECAY_HI,
            o2_lit_blend: DEFAULT_O2_LIT_BLEND,
            o2_min_obs: 10,
        }
    }
}

/// Deterministic pseudo-random hash for reproducible initialization.
/// Maps a u32 seed to a value in [-1.0, 1.0] with uniform distribution.
#[inline]
fn det_hash(seed: u32) -> f32 {
    let mut x = seed.wrapping_add(0x9e37_79b9);
    x = ((x >> 16) ^ x).wrapping_mul(0x0045_d9f3);
    x = ((x >> 16) ^ x).wrapping_mul(0x0045_d9f3);
    x = (x >> 16) ^ x;
    // Use only 23 bits (f32 mantissa width) to avoid rounding bias.
    // Shift right by 9 to get a value in [0, 2^23), divide by 2^23,
    // then scale to [-1.0, 1.0].
    let mantissa_bits = (x >> 9) as f32; // [0, 8388608)
    mantissa_bits / 4_194_304.0 - 1.0 // [0, 2.0) - 1.0 = [-1.0, 1.0)
}

const SIGMOID_MIN: f32 = -20.0;
const SIGMOID_MAX: f32 = 20.0;
const SIGMOID_STEPS_PER_UNIT: usize = 256;
const SIGMOID_LUT_LEN: usize = ((SIGMOID_MAX - SIGMOID_MIN) as usize * SIGMOID_STEPS_PER_UNIT) + 1;

/// Logistic sigmoid lookup table.
///
/// The expensive `exp()` calls happen once during process initialization.
/// Hot-path predictions use a 1/256-step table with linear interpolation,
/// retaining close agreement with the exact logistic curve without calling
/// the platform math library for every encoded or decoded byte.
static SIGMOID_LUT: std::sync::LazyLock<Box<[f32]>> = std::sync::LazyLock::new(|| {
    (0..SIGMOID_LUT_LEN)
        .map(|index| {
            let x = SIGMOID_MIN + index as f32 / SIGMOID_STEPS_PER_UNIT as f32;
            1.0 / (1.0 + (-x).exp())
        })
        .collect()
});

#[inline]
fn sigmoid(x: f32) -> f32 {
    let scaled = (x.clamp(SIGMOID_MIN, SIGMOID_MAX) - SIGMOID_MIN) * SIGMOID_STEPS_PER_UNIT as f32;
    let index = scaled as usize;
    if index + 1 >= SIGMOID_LUT_LEN {
        return SIGMOID_LUT[SIGMOID_LUT_LEN - 1];
    }
    let fraction = scaled - index as f32;
    let lo = SIGMOID_LUT[index];
    lo + (SIGMOID_LUT[index + 1] - lo) * fraction
}

/// Fast natural logarithm approximation (max error ~0.1%).
/// Uses the IEEE 754 bit representation to extract the exponent, then
/// applies a minimax polynomial correction.  Sufficient for log-likelihood
/// tracking in the adaptive mixer (which only needs relative comparisons).
#[allow(dead_code)]
#[inline]
fn fast_ln(x: f32) -> f32 {
    // Safety: f32 → u32 bit reinterpret is safe for any finite positive f32.
    debug_assert!(x > 0.0);
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as f32 - 127.0;
    // Mantissa in [1, 2) — reconstruct as f32.
    let mantissa = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    // Minimax polynomial for ln(m) on [1, 2): degree-2, max error ~0.003
    let ln_m = -1.7417939 + mantissa * (2.8212026 + mantissa * (-1.0794568));
    (exponent + ln_m) * core::f32::consts::LN_2
}

/// Neural SSM predictor with adaptive mixing.
pub struct NeuralSsmPredictor {
    // ── Config ────────────────────────────────────────────────
    cfg: NeuralSsmConfig,

    // ── RLE baseline ──────────────────────────────────────────
    rle: RlePredictor,

    // ── SSM hidden state ──────────────────────────────────────
    /// Hidden state vector [d]. Each dimension tracks an EMA at a different timescale.
    h: [f32; D_MAX],
    /// Diagonal decay rates [d].
    a: [f32; D_MAX],
    /// Precomputed 1.0 - a[i] for vectorization.
    a_inv: [f32; D_MAX],
    /// Input embeddings [256][d]. Fixed, deterministic initialization.
    embed: Box<[[f32; D_MAX]; 256]>,

    // ── Binary classifiers (online SGD) ───────────────────────
    /// Weights for p(run_symbol | h).
    w_run: [f32; D_MAX],
    /// Bias for p(run_symbol | h).
    b_run: f32,
    /// Weights for p(RUNA | run, h).
    w_runa: [f32; D_MAX],
    /// Bias for p(RUNA | run, h).
    b_runa: f32,

    // ── Adaptive mixer ────────────────────────────────────────
    /// EMA of SSM log-likelihood (for binary decisions only).
    ssm_perf: f32,
    /// EMA of RLE log-likelihood (for binary decisions only).
    rle_perf: f32,

    // ── Order-2 literal context model ─────────────────────────
    /// Literal counts per order-2 context [NUM_O2_CTX][254].
    o2_lit_counts: Box<[[u32; 254]; NUM_O2_CTX]>,
    /// Total literal counts per context.
    o2_lit_totals: [u32; NUM_O2_CTX],
    /// Previous two bytes for context hashing.
    prev_byte: u8,
    prev_prev_byte: u8,

    // ── Cached predictions ────────────────────────────────────
    last_rle_probs: [f32; 256],
    last_ssm_p_run: f32,
    last_ssm_p_runa: f32,
    last_rle_p_run: f32,
    last_rle_p_runa: f32,

    step: u32,

    /// Stage A: optional per-block reset baseline (a serialized predictor
    /// state from a pretrained dictionary). When set, `reset()` restores
    /// THIS state instead of zeroing — so every block starts from the
    /// dictionary's learned distribution while remaining independently
    /// decodable (seekability preserved). NOT part of `save_state`; it is a
    /// coding-time seed supplied identically at encode and decode.
    dict_baseline: Option<Vec<u8>>,
}

impl NeuralSsmPredictor {
    pub fn new() -> Self {
        Self::with_config(NeuralSsmConfig::default())
    }

    pub fn with_config(cfg: NeuralSsmConfig) -> Self {
        let d = cfg.d.clamp(1, D_MAX);
        // Validate config fields to prevent divergence from untrusted input.
        let lr = if cfg.lr.is_finite() && cfg.lr >= 0.0 {
            cfg.lr.min(1.0)
        } else {
            DEFAULT_LR
        };
        let mix_decay = if cfg.mix_decay.is_finite() {
            cfg.mix_decay.clamp(0.0, 1.0)
        } else {
            DEFAULT_MIX_DECAY
        };
        let mix_sensitivity = if cfg.mix_sensitivity.is_finite() && cfg.mix_sensitivity >= 0.0 {
            cfg.mix_sensitivity.min(1000.0)
        } else {
            DEFAULT_MIX_SENSITIVITY
        };
        let max_alpha = if cfg.max_alpha.is_finite() {
            cfg.max_alpha.clamp(0.0, 1.0)
        } else {
            DEFAULT_MAX_ALPHA
        };
        let decay_lo = if cfg.decay_lo.is_finite() {
            cfg.decay_lo.clamp(0.0, 1.0)
        } else {
            DEFAULT_DECAY_LO
        };
        let decay_hi = if cfg.decay_hi.is_finite() {
            cfg.decay_hi.clamp(decay_lo, 1.0)
        } else {
            DEFAULT_DECAY_HI.max(decay_lo)
        };
        let o2_lit_blend = if cfg.o2_lit_blend.is_finite() {
            cfg.o2_lit_blend.clamp(0.0, 1.0)
        } else {
            DEFAULT_O2_LIT_BLEND
        };
        let cfg = NeuralSsmConfig {
            d,
            lr,
            mix_decay,
            mix_sensitivity,
            max_alpha,
            decay_lo,
            decay_hi,
            o2_lit_blend,
            o2_min_obs: cfg.o2_min_obs,
            warmup: cfg.warmup,
        };

        // Decay rates: linearly spaced from decay_lo to decay_hi
        let mut a = [0.0f32; D_MAX];
        for (i, a_val) in a.iter_mut().enumerate().take(d) {
            let t = i as f32 / (d - 1).max(1) as f32;
            *a_val = decay_lo + t * (decay_hi - decay_lo);
        }

        // Precompute 1 - decay for vectorization
        let mut a_inv = [0.0f32; D_MAX];
        for i in 0..d {
            a_inv[i] = 1.0 - a[i];
        }

        // Embeddings: deterministic pseudo-random in [-1, 1]
        let mut embed = Box::new([[0.0f32; D_MAX]; 256]);
        for sym in 0..256 {
            for i in 0..d {
                let raw = det_hash((sym as u32).wrapping_mul(997).wrapping_add(i as u32 + 42));
                embed[sym][i] = raw;
            }
        }

        Self {
            cfg: NeuralSsmConfig { d, ..cfg },
            rle: RlePredictor::new(),
            h: [0.0; D_MAX],
            a,
            a_inv,
            embed,
            w_run: [0.0; D_MAX],
            b_run: 0.0,
            w_runa: [0.0; D_MAX],
            b_runa: 0.0,
            ssm_perf: 0.0,
            rle_perf: 0.0,
            o2_lit_counts: Box::new([[0u32; 254]; NUM_O2_CTX]),
            o2_lit_totals: [0u32; NUM_O2_CTX],
            prev_byte: 0xFF, // sentinel: no previous byte
            prev_prev_byte: 0xFF,
            last_rle_probs: [1.0 / 256.0; 256],
            last_ssm_p_run: 0.5,
            last_ssm_p_runa: 0.5,
            last_rle_p_run: 0.5,
            last_rle_p_runa: 0.5,
            step: 0,
            dict_baseline: None,
        }
    }

    /// Stage A: set a pretrained dictionary state as the per-block reset
    /// baseline. Returns `true` if the state is valid and was applied (the
    /// predictor is immediately reset to it); `false` leaves the predictor
    /// unchanged. The same state must be supplied at decode time.
    pub fn set_dict_baseline(&mut self, state: &[u8]) -> bool {
        // Validate by loading into self; load_state commits only on success.
        if !self.load_state(state) {
            return false;
        }
        self.dict_baseline = Some(state.to_vec());
        true
    }

    /// Compute SSM's binary predictions from hidden state.
    fn ssm_binary_predict(&self) -> (f32, f32) {
        let d = self.cfg.d;
        let mut dot_run = self.b_run;
        let mut dot_runa = self.b_runa;
        for i in 0..d {
            dot_run += self.w_run[i] * self.h[i];
            dot_runa += self.w_runa[i] * self.h[i];
        }
        (sigmoid(dot_run), sigmoid(dot_runa))
    }

    /// Hash previous two bytes into an order-2 context index for literals.
    ///
    /// 8 buckets per byte (64 contexts total). MTF ranks cluster near zero,
    /// so the low ranks get exact buckets and larger ranks are grouped
    /// logarithmically. `bucket(0..=3)` are exact; then 4-5, 6-9, 10-19, 20+.
    #[inline]
    fn rank_bucket(b: u8) -> usize {
        match b {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4..=5 => 4,
            6..=9 => 5,
            10..=19 => 6,
            _ => 7,
        }
    }

    fn o2_context(prev: u8, prev_prev: u8) -> usize {
        // 8 prev-buckets × 8 prev_prev-buckets = 64 contexts.
        Self::rank_bucket(prev) * 8 + Self::rank_bucket(prev_prev)
    }

    /// Compute mixing weight alpha for the SSM component.
    /// Returns 0 when SSM is worse than RLE (safe fallback).
    #[inline]
    fn mixing_alpha(&self) -> f32 {
        if self.step < self.cfg.warmup {
            return 0.0;
        }
        let diff = self.ssm_perf - self.rle_perf;
        if diff <= 0.0 {
            return 0.0; // SSM is worse, don't use it
        }
        (diff * self.cfg.mix_sensitivity).min(self.cfg.max_alpha)
    }

    /// Build the clamped, unnormalized symbol weights shared by the encoder
    /// interval query and decoder CDF paths.
    #[inline]
    fn model_weights(&mut self) -> ([f32; 256], f32) {
        let rle_probs = self.rle.predict();
        self.last_rle_probs = rle_probs;

        let rle_p_run = rle_probs[0] + rle_probs[1];
        let rle_p_runa = if rle_p_run > MIN_PROB {
            rle_probs[0] / rle_p_run
        } else {
            0.5
        };
        self.last_rle_p_run = rle_p_run;
        self.last_rle_p_runa = rle_p_runa;

        let (ssm_p_run, ssm_p_runa) = self.ssm_binary_predict();
        self.last_ssm_p_run = ssm_p_run;
        self.last_ssm_p_runa = ssm_p_runa;

        let alpha = self.mixing_alpha();
        let p_run = alpha * ssm_p_run + (1.0 - alpha) * rle_p_run;
        let p_runa = alpha * ssm_p_runa + (1.0 - alpha) * rle_p_runa;

        let inv_rle_lit = 1.0 / (1.0 - rle_p_run).max(MIN_PROB);
        let o2_ctx = Self::o2_context(self.prev_byte, self.prev_prev_byte);
        let o2_obs = self.o2_lit_totals[o2_ctx];
        let inv_o2_total = 1.0 / (o2_obs as f32 + 254.0 * O2_ALPHA);
        let eff_blend = self.cfg.o2_lit_blend * (o2_obs as f32 / (o2_obs as f32 + O2_CONF_K));
        let rle_weight = 1.0 - eff_blend;
        let use_o2 = eff_blend > 0.0;

        let mut weights = [0.0f32; 256];
        weights[0] = (p_run * p_runa).max(MIN_PROB);
        weights[1] = (p_run * (1.0 - p_runa)).max(MIN_PROB);
        let p_lit = 1.0 - p_run;
        let mut sum = weights[0] + weights[1];

        for i in 2..256 {
            let rle_p = rle_probs[i] * inv_rle_lit;
            let lit_p = if use_o2 {
                let o2_p = (self.o2_lit_counts[o2_ctx][i - 2] as f32 + O2_ALPHA) * inv_o2_total;
                rle_weight * rle_p + eff_blend * o2_p
            } else {
                rle_p
            };
            let weight = (p_lit * lit_p).max(MIN_PROB);
            weights[i] = weight;
            sum += weight;
        }

        (weights, sum)
    }

    /// Quantize cumulative mass while reserving one count per symbol.
    #[inline]
    fn quantized_boundary(cumulative: f64, total: f64, symbol_index: usize) -> u16 {
        use crate::coding::rans::PROB_TOTAL;
        const RESERVED: u32 = 256;
        let scalable = PROB_TOTAL - RESERVED;
        let value = symbol_index as u32 + (cumulative * scalable as f64 / total).floor() as u32;
        value.min(PROB_TOTAL) as u16
    }
}

impl Default for NeuralSsmPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityPredictor for NeuralSsmPredictor {
    #[inline]
    fn predict(&mut self) -> [f32; 256] {
        let (mut probs, sum) = self.model_weights();
        let inv = 1.0 / sum;
        for prob in probs.iter_mut() {
            *prob *= inv;
        }

        probs
    }

    /// Build 15-bit CDF directly, bypassing the `[f32; 256]` → probs_to_cdf() path.
    ///
    /// Uses the same cumulative-rounding approach as probs_to_cdf but avoids
    /// the expensive fixup path (sort + redistribution) since we build the
    /// probabilities ourselves and can guarantee they're well-behaved.
    #[inline]
    fn predict_cdf(&mut self) -> [u16; 257] {
        use crate::coding::rans::PROB_TOTAL;

        let (weights, sum) = self.model_weights();
        let mut monotone_cdf = [0u16; 257];
        let mut cumulative = 0.0f64;
        let total = sum as f64;
        for i in 0..256 {
            monotone_cdf[i] = Self::quantized_boundary(cumulative, total, i);
            cumulative += weights[i] as f64;
        }
        monotone_cdf[256] = PROB_TOTAL as u16;
        return monotone_cdf;

        // Retained as a compile-disabled reference for ratio/performance
        // comparisons against the pre-0.3 quantizer.
        #[cfg(any())]
        {
            // 1. Get RLE baseline prediction
            let rle_probs = self.rle.predict();
            self.last_rle_probs = rle_probs;

            // 2. Extract RLE's binary decisions
            let rle_p_run = rle_probs[0] + rle_probs[1];
            let rle_p_runa = if rle_p_run > MIN_PROB {
                rle_probs[0] / rle_p_run
            } else {
                0.5
            };
            self.last_rle_p_run = rle_p_run;
            self.last_rle_p_runa = rle_p_runa;

            // 3. Get SSM's binary predictions
            let (ssm_p_run, ssm_p_runa) = self.ssm_binary_predict();
            self.last_ssm_p_run = ssm_p_run;
            self.last_ssm_p_runa = ssm_p_runa;

            // 4. Mix binary decisions
            let alpha = self.mixing_alpha();
            let beta = 1.0 - alpha;
            let p_run = alpha * ssm_p_run + beta * rle_p_run;
            let p_runa = alpha * ssm_p_runa + beta * rle_p_runa;

            // 5. Build CDF directly via cumulative rounding in f32.
            // Same algorithm as probs_to_cdf's core loop but avoids:
            // - f64 upcasting of all 256 probabilities
            // - the expensive fixup path (sort + proportional redistribution)
            // - materializing a separate [f32;256] intermediate for predict()
            let rle_p_lit_total = (1.0 - rle_p_run).max(MIN_PROB);
            let o2_ctx = Self::o2_context(self.prev_byte, self.prev_prev_byte);
            let o2_obs = self.o2_lit_totals[o2_ctx];
            let o2_total = o2_obs as f32 + 254.0 * O2_ALPHA;
            // Confidence-weighted blend (Stage B) — MUST match predict() exactly so
            // the [f32;256] and direct-CDF paths agree.
            let eff_blend = self.cfg.o2_lit_blend * (o2_obs as f32 / (o2_obs as f32 + O2_CONF_K));

            // First pass: compute raw probs and sum.
            // Precompute reciprocals to replace 254+ divisions with multiplies.
            let inv_rle_lit = 1.0 / rle_p_lit_total;
            let inv_o2_total = 1.0 / o2_total;
            let rle_weight = 1.0 - eff_blend;
            let use_o2 = eff_blend > 0.0;

            let mut raw = [0.0f32; 256];
            raw[0] = (p_run * p_runa).max(MIN_PROB);
            raw[1] = (p_run * (1.0 - p_runa)).max(MIN_PROB);
            let p_lit = 1.0 - p_run;
            let mut sum = raw[0] + raw[1];
            for i in 2..256 {
                let rle_p = rle_probs[i] * inv_rle_lit;
                let lit_p = if use_o2 {
                    let o2_p = (self.o2_lit_counts[o2_ctx][i - 2] as f32 + O2_ALPHA) * inv_o2_total;
                    rle_weight * rle_p + eff_blend * o2_p
                } else {
                    rle_p
                };
                let p = (p_lit * lit_p).max(MIN_PROB);
                raw[i] = p;
                sum += p;
            }

            // Cumulative rounding: same precision as probs_to_cdf but in f32.
            //
            // ── Early-exit overshoot detection ────────────────────────────
            //
            // On real NeuralSSM data (BWT+MTF+RLE of English text), the
            // overshoot fallback fires on ~98.76% of bytes — measured via
            // the `query_cdf_overshoot_rate_on_bench_corpus` diagnostic.
            // Each overshoot wastes both the f32 cumulative-rounding sweep
            // AND the f32 monotonicity fix-up before calling `probs_to_cdf`.
            //
            // Observation: overshoot is GUARANTEED whenever the rounded gap
            // `cur - prev` is zero anywhere in the interior. The fix-up loop
            // would then bump `cdf[i+1] = cdf[i] + 1`, and any such bump
            // pushes `cdf[256]` past `PROB_TOTAL` (since the rounded
            // `cdf[256]` is anchored at `PROB_TOTAL` and bumps only ever
            // increase). So we can break out of the rounding loop the
            // instant we see `cur <= prev` and fall straight to
            // `probs_to_cdf` — bit-identical to running the full f32 path
            // and then hitting the same fallback.
            //
            // For peaked NeuralSSM distributions this typically triggers at
            // i ≈ 3..10 (just past the RUNA/RUNB peak), so we skip ~250
            // iterations of pass 2 + all 256 of pass 3 on every overshoot
            // byte. The `predict_cdf_early_exit_microbench` test measures a
            // **+19.08% speedup (1.24x)** on the BWT-encoded English corpus
            // the `compress_ssm` criterion bench uses, with bit-identity to
            // the reference verified by `predict_cdf_early_exit_matches_reference`.
            let scale = PROB_TOTAL as f32 / sum;
            let mut cdf = [0u16; 257];
            let mut cum = 0.0f32;
            let mut prev: u16 = 0;
            for i in 0..256 {
                let cur = (cum * scale + 0.5) as u16;
                // i > 0 because cdf[0] = 0 by initialization and the first
                // rounded value (i=0) is also 0 — the `<=` would trigger
                // spuriously. From i=1 onward, `cur <= prev` means a zero
                // rounded gap → fix-up will bump → overshoot guaranteed.
                if i > 0 && cur <= prev {
                    return crate::coding::rans::probs_to_cdf(&raw);
                }
                cdf[i] = cur;
                prev = cur;
                cum += raw[i];
            }
            cdf[256] = PROB_TOTAL as u16;

            // Ensure strict monotonicity. With the early-exit above, all
            // interior gaps are guaranteed >= 1, so the only entry the
            // fix-up can touch is cdf[256] (when cdf[255] rounds up to
            // PROB_TOTAL on a floating-point boundary). We still need this
            // loop for that edge case, but it's a no-op for indices 0..255
            // in the hot path.
            for i in 0..256 {
                if cdf[i + 1] <= cdf[i] {
                    cdf[i + 1] = cdf[i] + 1;
                }
            }

            // If forward fixup overshot, fall back to full probs_to_cdf.
            if cdf[256] != PROB_TOTAL as u16 {
                return crate::coding::rans::probs_to_cdf(&raw);
            }

            cdf
        }
    }

    /// Encode-only fast path that computes just the selected symbol interval.
    #[inline]
    fn query_cdf(&mut self, byte: u8) -> (u16, u16) {
        let (weights, sum) = self.model_weights();
        let symbol = byte as usize;
        let mut cumulative = 0.0f64;
        for &weight in &weights[..symbol] {
            cumulative += weight as f64;
        }
        let lo = Self::quantized_boundary(cumulative, sum as f64, symbol);
        cumulative += weights[symbol] as f64;
        let hi = if symbol == 255 {
            crate::coding::rans::PROB_TOTAL as u16
        } else {
            Self::quantized_boundary(cumulative, sum as f64, symbol + 1)
        };
        (lo, hi)
    }

    #[inline]
    fn update(&mut self, byte: u8) {
        let y = byte as usize;
        let d = self.cfg.d;
        let lr = self.cfg.lr;
        let decay = self.cfg.mix_decay;

        // ── Update mixer performance (EMA of log-likelihood) ──────────
        // Compare SSM vs RLE on binary decisions only.
        let (ssm_ll, rle_ll) = if byte <= 1 {
            let ssm_run_ll = self.last_ssm_p_run.max(MIN_PROB).ln();
            let rle_run_ll = self.last_rle_p_run.max(MIN_PROB).ln();
            let ssm_ab = if byte == 0 {
                self.last_ssm_p_runa.max(MIN_PROB).ln()
            } else {
                (1.0 - self.last_ssm_p_runa).max(MIN_PROB).ln()
            };
            let rle_ab = if byte == 0 {
                self.last_rle_p_runa.max(MIN_PROB).ln()
            } else {
                (1.0 - self.last_rle_p_runa).max(MIN_PROB).ln()
            };
            (ssm_run_ll + ssm_ab, rle_run_ll + rle_ab)
        } else {
            let ssm_lit_ll = (1.0 - self.last_ssm_p_run).max(MIN_PROB).ln();
            let rle_lit_ll = (1.0 - self.last_rle_p_run).max(MIN_PROB).ln();
            (ssm_lit_ll, rle_lit_ll)
        };
        self.ssm_perf = decay * self.ssm_perf + (1.0 - decay) * ssm_ll;
        self.rle_perf = decay * self.rle_perf + (1.0 - decay) * rle_ll;

        // ── Update RLE predictor ──────────────────────────────────────
        self.rle.update(byte);

        // ── Online SGD for SSM binary classifiers ─────────────────────
        const WEIGHT_CLIP: f32 = 10.0;

        // Run-vs-lit classifier: always update
        let y_run = if byte <= 1 { 1.0f32 } else { 0.0 };
        let err_run = self.last_ssm_p_run - y_run; // sigmoid CE gradient
        let lr_err_run = lr * err_run;
        for i in 0..d {
            self.w_run[i] =
                (self.w_run[i] - lr_err_run * self.h[i]).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
        }
        self.b_run = (self.b_run - lr_err_run).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);

        // RUNA-vs-RUNB classifier: only update when byte is a run symbol
        if byte <= 1 {
            let y_runa = if byte == 0 { 1.0f32 } else { 0.0 };
            let err_runa = self.last_ssm_p_runa - y_runa;
            let lr_err_runa = lr * err_runa;
            for i in 0..d {
                self.w_runa[i] =
                    (self.w_runa[i] - lr_err_runa * self.h[i]).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
            }
            self.b_runa = (self.b_runa - lr_err_runa).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
        }

        // ── Update order-2 literal context model ───────────────────
        if byte >= 2 {
            let o2_ctx = Self::o2_context(self.prev_byte, self.prev_prev_byte);
            let idx = (byte - 2) as usize;
            self.o2_lit_counts[o2_ctx][idx] += 1;
            self.o2_lit_totals[o2_ctx] += 1;
            if self.o2_lit_totals[o2_ctx] > 100_000 {
                self.o2_lit_totals[o2_ctx] = 0;
                for v in self.o2_lit_counts[o2_ctx].iter_mut() {
                    *v >>= 1;
                    self.o2_lit_totals[o2_ctx] += *v;
                }
            }
        }

        // ── SSM state update: EMA with per-dimension decay ────────────
        let emb = &self.embed[y];
        for (i, h_val) in self.h.iter_mut().enumerate().take(d) {
            *h_val = self.a[i] * *h_val + self.a_inv[i] * emb[i];
        }

        self.prev_prev_byte = self.prev_byte;
        self.prev_byte = byte;
        self.step += 1;
    }

    fn reset(&mut self) {
        // Stage A: if a dictionary baseline is set, restore THAT state so
        // every block starts from the pretrained distribution. We clone the
        // bytes first to avoid borrowing self while load_state mutates it;
        // the baseline was validated in set_dict_baseline, so this succeeds.
        if let Some(baseline) = self.dict_baseline.clone() {
            let ok = self.load_state(&baseline);
            debug_assert!(ok, "dict_baseline was validated on set, must reload");
            // load_state does not touch dict_baseline, so it persists.
            return;
        }
        // Zero only mutable state in-place. Deterministic fields (cfg, a, a_inv,
        // embed) are pure functions of config and never need recomputation.
        // This avoids 2 Box re-allocations and 8192 det_hash() calls per reset.
        self.h.fill(0.0);
        self.w_run.fill(0.0);
        self.b_run = 0.0;
        self.w_runa.fill(0.0);
        self.b_runa = 0.0;
        self.ssm_perf = 0.0;
        self.rle_perf = 0.0;
        for ctx in self.o2_lit_counts.iter_mut() {
            ctx.fill(0);
        }
        self.o2_lit_totals.fill(0);
        self.prev_byte = 0xFF;
        self.prev_prev_byte = 0xFF;
        self.last_rle_probs.fill(1.0 / 256.0);
        self.last_ssm_p_run = 0.5;
        self.last_ssm_p_runa = 0.5;
        self.last_rle_p_run = 0.5;
        self.last_rle_p_runa = 0.5;
        self.step = 0;
        self.rle.reset();
    }

    fn coding_baseline(&self) -> Option<&[u8]> {
        self.dict_baseline.as_deref()
    }

    fn set_coding_baseline(&mut self, state: &[u8]) -> bool {
        self.set_dict_baseline(state)
    }

    fn name(&self) -> &str {
        "neural-ssm"
    }

    fn predictor_id(&self) -> PredictorId {
        PredictorId::NeuralSsm
    }

    fn save_state(&self) -> Option<Vec<u8>> {
        let d = self.cfg.d;
        // Format: [version: u8] [d: u32] [h: f32*d] [w_run: f32*d] [b_run: f32]
        //         [w_runa: f32*d] [b_runa: f32] [ssm_perf: f32] [rle_perf: f32]
        //         [o2_lit_counts: u32 * NUM_O2_CTX * 254] [o2_lit_totals: u32 * NUM_O2_CTX]
        //         [prev_byte: u8] [prev_prev_byte: u8] [step: u32]
        //         [rle_state_len: u32] [rle_state: bytes]
        let mut buf = Vec::new();
        buf.push(1); // version 1
        buf.extend_from_slice(&(d as u32).to_le_bytes());
        for i in 0..d {
            buf.extend_from_slice(&self.h[i].to_le_bytes());
        }
        for i in 0..d {
            buf.extend_from_slice(&self.w_run[i].to_le_bytes());
        }
        buf.extend_from_slice(&self.b_run.to_le_bytes());
        for i in 0..d {
            buf.extend_from_slice(&self.w_runa[i].to_le_bytes());
        }
        buf.extend_from_slice(&self.b_runa.to_le_bytes());
        buf.extend_from_slice(&self.ssm_perf.to_le_bytes());
        buf.extend_from_slice(&self.rle_perf.to_le_bytes());
        for ctx in 0..NUM_O2_CTX {
            for &c in &self.o2_lit_counts[ctx] {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        }
        for &t in &self.o2_lit_totals {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        buf.push(self.prev_byte);
        buf.push(self.prev_prev_byte);
        buf.extend_from_slice(&self.step.to_le_bytes());
        // Include RLE sub-predictor state
        if let Some(rle_state) = self.rle.save_state() {
            buf.extend_from_slice(&(rle_state.len() as u32).to_le_bytes());
            buf.extend_from_slice(&rle_state);
        } else {
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
        Some(buf)
    }

    fn load_state(&mut self, data: &[u8]) -> bool {
        /// Read a u32 from `data` at `off`, advancing `off` by 4.
        /// Returns `None` if out of bounds.
        fn read_u32(data: &[u8], off: &mut usize) -> Option<u32> {
            let end = (*off).checked_add(4)?;
            let bytes: [u8; 4] = data.get(*off..end)?.try_into().ok()?;
            *off = end;
            Some(u32::from_le_bytes(bytes))
        }
        /// Read a finite f32 from `data` at `off`, advancing `off` by 4.
        /// Returns `None` if out of bounds or if the value is NaN/Infinity.
        fn read_finite_f32(data: &[u8], off: &mut usize) -> Option<f32> {
            let v = f32::from_bits(read_u32(data, off)?);
            if v.is_finite() {
                Some(v)
            } else {
                None
            }
        }
        /// Read a bounded f32 — finite and within [-bound, bound].
        fn read_bounded_f32(data: &[u8], off: &mut usize, bound: f32) -> Option<f32> {
            let v = read_finite_f32(data, off)?;
            if v.abs() <= bound {
                Some(v)
            } else {
                None
            }
        }
        // Maximum allowed magnitude for learned weights. The online SGD clips
        // weights to ±WEIGHT_CLIP (10.0), so anything larger is adversarial.
        const WEIGHT_BOUND: f32 = 10.0;
        // Maximum allowed hidden state magnitude. With decay in [0,1] and
        // embeddings in [-1,1], hidden state stays bounded; 5.0 is generous.
        const HIDDEN_BOUND: f32 = 5.0;
        // Maximum per-symbol count in order-2 literal model (rescales at 100K total).
        const MAX_O2_COUNT: u32 = 200_000;

        if data.is_empty() || data[0] != 1 {
            return false; // Missing or unknown version
        }
        let mut off = 1; // skip version byte
        let d = read_u32(data, &mut off).unwrap_or(u32::MAX) as usize;
        if d > D_MAX || d != self.cfg.d {
            return false;
        }
        // Check minimum expected size upfront (version byte already consumed)
        let min_size =
            1 + 4 + d * 4 * 3 + 4 * 4 + NUM_O2_CTX * 254 * 4 + NUM_O2_CTX * 4 + 2 + 4 + 4;
        if data.len() < min_size {
            return false;
        }
        // Parse into temporaries to avoid leaving self in a half-loaded state.
        let mut h = [0.0f32; D_MAX];
        let mut w_run = [0.0f32; D_MAX];
        let mut w_runa = [0.0f32; D_MAX];
        for h_val in h.iter_mut().take(d) {
            *h_val = match read_bounded_f32(data, &mut off, HIDDEN_BOUND) {
                Some(v) => v,
                None => return false,
            };
        }
        for w_val in w_run.iter_mut().take(d) {
            *w_val = match read_bounded_f32(data, &mut off, WEIGHT_BOUND) {
                Some(v) => v,
                None => return false,
            };
        }
        let b_run = match read_bounded_f32(data, &mut off, WEIGHT_BOUND) {
            Some(v) => v,
            None => return false,
        };
        for w_val in w_runa.iter_mut().take(d) {
            *w_val = match read_bounded_f32(data, &mut off, WEIGHT_BOUND) {
                Some(v) => v,
                None => return false,
            };
        }
        let b_runa = match read_bounded_f32(data, &mut off, WEIGHT_BOUND) {
            Some(v) => v,
            None => return false,
        };
        let ssm_perf = match read_finite_f32(data, &mut off) {
            Some(v) => v,
            None => return false,
        };
        let rle_perf = match read_finite_f32(data, &mut off) {
            Some(v) => v,
            None => return false,
        };

        // Layout MUST match save_state: all counts (NUM_O2_CTX * 254) as one
        // contiguous block, THEN all totals (NUM_O2_CTX) as a second block.
        // (Prior to the fix, load read these interleaved per-context, which
        // silently corrupted every dictionary whose contexts weren't all
        // empty — the consistency check below would reject a context's total
        // read from the next context's first count.)
        let mut o2_lit_counts = [[0u32; 254]; NUM_O2_CTX];
        let mut o2_lit_totals = [0u32; NUM_O2_CTX];
        let mut computed_totals = [0u64; NUM_O2_CTX];
        for ctx in 0..NUM_O2_CTX {
            for lit_count in o2_lit_counts[ctx].iter_mut() {
                let c = match read_u32(data, &mut off) {
                    Some(v) => v,
                    None => return false,
                };
                if c > MAX_O2_COUNT {
                    return false; // Reject adversarially large counts
                }
                *lit_count = c;
                computed_totals[ctx] += c as u64;
            }
        }
        for ctx in 0..NUM_O2_CTX {
            o2_lit_totals[ctx] = match read_u32(data, &mut off) {
                Some(v) => v,
                None => return false,
            };
            // Validate totals are consistent with counts
            if computed_totals[ctx] > u32::MAX as u64
                || o2_lit_totals[ctx] != computed_totals[ctx] as u32
            {
                return false;
            }
        }
        if off + 2 > data.len() {
            return false;
        }
        let prev_byte = data[off];
        off += 1;
        let prev_prev_byte = data[off];
        off += 1;
        let step = match read_u32(data, &mut off) {
            Some(v) => v,
            None => return false,
        };

        // Load RLE sub-predictor state
        let rle_len = match read_u32(data, &mut off) {
            Some(v) => v as usize,
            None => return false,
        };
        if rle_len > 0 {
            if off.checked_add(rle_len).is_none_or(|end| end > data.len()) {
                return false;
            }
            if !self.rle.load_state(&data[off..off + rle_len]) {
                return false;
            }
            off += rle_len;
        }

        // Reject trailing bytes — strict parsing prevents silent acceptance
        // of truncated or extended state blobs.
        if off != data.len() {
            return false;
        }

        // All validation passed — commit state
        self.h = h;
        self.w_run = w_run;
        self.b_run = b_run;
        self.w_runa = w_runa;
        self.b_runa = b_runa;
        self.ssm_perf = ssm_perf;
        self.rle_perf = rle_perf;
        *self.o2_lit_counts = o2_lit_counts;
        self.o2_lit_totals = o2_lit_totals;
        self.prev_byte = prev_byte;
        self.prev_prev_byte = prev_prev_byte;
        self.step = step;
        true
    }
}

#[cfg(test)]
impl NeuralSsmPredictor {
    /// Reference implementation of `predict_cdf` WITHOUT the early-exit
    /// overshoot detection — used by the microbench and bit-identity
    /// tests to validate the production version above. Test-only; never
    /// called from compress/decompress paths.
    #[allow(dead_code)]
    pub(crate) fn predict_cdf_no_early_exit(&mut self) -> [u16; 257] {
        use crate::coding::rans::PROB_TOTAL;

        // Replay pass 1 (compute raw[], sum) IDENTICALLY to predict_cdf
        // above — including the state writes to last_* and the
        // ssm_binary_predict() side effects, since both paths are
        // expected to leave the predictor in the same post-predict
        // state. The bit-identity test relies on this.
        let rle_probs = self.rle.predict();
        self.last_rle_probs = rle_probs;

        let rle_p_run = rle_probs[0] + rle_probs[1];
        let rle_p_runa = if rle_p_run > MIN_PROB {
            rle_probs[0] / rle_p_run
        } else {
            0.5
        };
        self.last_rle_p_run = rle_p_run;
        self.last_rle_p_runa = rle_p_runa;

        let (ssm_p_run, ssm_p_runa) = self.ssm_binary_predict();
        self.last_ssm_p_run = ssm_p_run;
        self.last_ssm_p_runa = ssm_p_runa;

        let alpha = self.mixing_alpha();
        let beta = 1.0 - alpha;
        let p_run = alpha * ssm_p_run + beta * rle_p_run;
        let p_runa = alpha * ssm_p_runa + beta * rle_p_runa;

        let rle_p_lit_total = (1.0 - rle_p_run).max(MIN_PROB);
        let o2_ctx = Self::o2_context(self.prev_byte, self.prev_prev_byte);
        let o2_obs = self.o2_lit_totals[o2_ctx];
        let o2_total = o2_obs as f32 + 254.0 * O2_ALPHA;
        // Must mirror predict_cdf()'s confidence-weighted blend exactly.
        let eff_blend = self.cfg.o2_lit_blend * (o2_obs as f32 / (o2_obs as f32 + O2_CONF_K));

        let inv_rle_lit = 1.0 / rle_p_lit_total;
        let inv_o2_total = 1.0 / o2_total;
        let rle_weight = 1.0 - eff_blend;
        let use_o2 = eff_blend > 0.0;

        let mut raw = [0.0f32; 256];
        raw[0] = (p_run * p_runa).max(MIN_PROB);
        raw[1] = (p_run * (1.0 - p_runa)).max(MIN_PROB);
        let p_lit = 1.0 - p_run;
        let mut sum = raw[0] + raw[1];
        for i in 2..256 {
            let rle_p = rle_probs[i] * inv_rle_lit;
            let lit_p = if use_o2 {
                let o2_p = (self.o2_lit_counts[o2_ctx][i - 2] as f32 + O2_ALPHA) * inv_o2_total;
                rle_weight * rle_p + eff_blend * o2_p
            } else {
                rle_p
            };
            let p = (p_lit * lit_p).max(MIN_PROB);
            raw[i] = p;
            sum += p;
        }

        // Pass 2-3 WITHOUT early-exit — full sweeps, then check overshoot
        // at the end. This is the structure the original predict_cdf had
        // before the early-exit optimization was added.
        let scale = PROB_TOTAL as f32 / sum;
        let mut cdf = [0u16; 257];
        let mut cum = 0.0f32;
        for i in 0..256 {
            cdf[i] = (cum * scale + 0.5) as u16;
            cum += raw[i];
        }
        cdf[256] = PROB_TOTAL as u16;
        for i in 0..256 {
            if cdf[i + 1] <= cdf[i] {
                cdf[i + 1] = cdf[i] + 1;
            }
        }
        if cdf[256] != PROB_TOTAL as u16 {
            return crate::coding::rans::probs_to_cdf(&raw);
        }
        cdf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_lookup_tracks_exact_curve() {
        for step in -2000..=2000 {
            let x = step as f32 / 100.0;
            let exact = 1.0 / (1.0 + (-x).exp());
            let error = (sigmoid(x) - exact).abs();
            assert!(error < 0.000_001, "sigmoid LUT error {error} at x={x}");
        }
    }

    /// Bit-identity guard for the encode-only query and decoder CDF paths.
    #[test]
    fn query_cdf_matches_decoder_cdf() {
        // Use the same bias as roundtrip_with_range_coder: heavy RUNA/RUNB
        // distribution that drives the overshoot path on essentially
        // every byte.
        let mut data = Vec::with_capacity(4096);
        let mut state = 0x9e37_79b9u32;
        for _ in 0..4096 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = match (state >> 24) & 0x7 {
                0..=3 => 0,
                4 => 1,
                _ => ((state >> 8) & 0xFF) as u8,
            };
            data.push(b);
        }

        let mut query_predictor = NeuralSsmPredictor::new();
        let mut cdf_predictor = NeuralSsmPredictor::new();
        for (step, &byte) in data.iter().enumerate() {
            let interval = query_predictor.query_cdf(byte);
            let cdf = cdf_predictor.predict_cdf();
            let symbol = byte as usize;
            assert_eq!(
                interval,
                (cdf[symbol], cdf[symbol + 1]),
                "CDF interval diverged at step {step} (byte={byte})"
            );
            query_predictor.update(byte);
            cdf_predictor.update(byte);
        }
    }

    /// Microbench: measure the speedup from the early-exit overshoot
    /// detection on the same BWT-encoded English corpus the criterion
    /// `compress_ssm` bench uses. Run with `--ignored --nocapture` to
    /// see numbers. Repeatable (tight loop, no harness noise).
    #[test]
    #[ignore]
    #[cfg(feature = "bwt-encode")]
    fn query_cdf_microbench() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("tests")
            .join("fixtures")
            .join("large");
        let text = match std::fs::read(dir.join("english.txt")) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("[microbench] fixture missing, skipping");
                return;
            }
        };
        let bwt_data = match crate::coding::bwt_preprocess::bwt_mtf_encode(&text) {
            Ok(d) => d,
            Err(_) => return,
        };
        eprintln!("[microbench] corpus: {} bytes", bwt_data.len());

        // Reference first (no early exit), then optimized. Cache is warm
        // for both — the bench corpus fits in L3.
        let mut pred_ref = NeuralSsmPredictor::new();
        let mut sample_ref = 0u64;
        let start = std::time::Instant::now();
        for &byte in &bwt_data {
            let cdf = pred_ref.predict_cdf();
            sample_ref = sample_ref
                .wrapping_add(cdf[byte as usize] as u64)
                .wrapping_add(cdf[byte as usize + 1] as u64);
            pred_ref.update(byte);
        }
        let ref_time = start.elapsed();

        let mut pred_opt = NeuralSsmPredictor::new();
        let mut sample_opt = 0u64;
        let start = std::time::Instant::now();
        for &byte in &bwt_data {
            let (lo, hi) = pred_opt.query_cdf(byte);
            sample_opt = sample_opt.wrapping_add(lo as u64).wrapping_add(hi as u64);
            pred_opt.update(byte);
        }
        let opt_time = start.elapsed();

        // The samples should match — bit-identity proven again.
        assert_eq!(sample_ref, sample_opt, "query_cdf diverged from full CDF");

        let ref_mbps = bwt_data.len() as f64 / ref_time.as_secs_f64() / (1024.0 * 1024.0);
        let opt_mbps = bwt_data.len() as f64 / opt_time.as_secs_f64() / (1024.0 * 1024.0);
        let delta = (1.0 - opt_time.as_secs_f64() / ref_time.as_secs_f64()) * 100.0;
        let speedup = ref_time.as_secs_f64() / opt_time.as_secs_f64();
        eprintln!("[full CDF    ] {:?}  ({:.3} MiB/s)", ref_time, ref_mbps);
        eprintln!("[query CDF   ] {:?}  ({:.3} MiB/s)", opt_time, opt_mbps);
        eprintln!(
            "[delta] early-exit is {:+.2}% faster (speedup {:.3}x)",
            delta, speedup
        );
    }

    #[test]
    fn starts_near_uniform() {
        let mut pred = NeuralSsmPredictor::new();
        let probs = pred.predict();
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Probabilities should sum to ~1.0, got {sum}"
        );
        // During warmup (alpha=0), should match RlePredictor
    }

    #[test]
    fn probs_sum_to_one_throughout() {
        let mut pred = NeuralSsmPredictor::new();
        let stream: Vec<u8> = vec![0, 1, 0, 3, 0, 0, 5, 0, 0, 0, 3, 0, 1, 4, 2, 0];
        for &b in &stream {
            let probs = pred.predict();
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Probabilities must sum to ~1.0, got {sum}"
            );
            assert!(
                probs.iter().all(|&p| p >= 0.0),
                "All probabilities must be non-negative"
            );
            pred.update(b);
        }
    }

    #[test]
    fn adapts_to_run_heavy_stream() {
        let mut pred = NeuralSsmPredictor::new();
        // Feed lots of RUNA (0) — typical for BWT+MTF+RLE
        for _ in 0..200 {
            pred.predict();
            pred.update(0);
        }
        let probs = pred.predict();
        // p(RUNA) + p(RUNB) should be high
        let p_run = probs[0] + probs[1];
        assert!(
            p_run > 0.5,
            "After many run symbols, p(run) should be > 0.5, got {p_run}"
        );
    }

    #[test]
    fn ssm_trains_binary_classifiers() {
        let mut pred = NeuralSsmPredictor::new();
        // Train on alternating 0, 0, 0, 3 pattern (75% run, 25% literal)
        let pattern: Vec<u8> = (0..400).map(|i| if i % 4 < 3 { 0 } else { 3 }).collect();
        for &b in &pattern {
            pred.predict();
            pred.update(b);
        }
        // SSM should have learned p(run) ≈ 0.75
        let (ssm_p_run, _) = pred.ssm_binary_predict();
        assert!(
            ssm_p_run > 0.6,
            "SSM should learn p(run) > 0.6, got {ssm_p_run}"
        );
    }

    #[test]
    fn mixing_starts_at_zero() {
        let pred = NeuralSsmPredictor::new();
        assert_eq!(pred.mixing_alpha(), 0.0, "Alpha should be 0 before warmup");
    }

    #[test]
    fn deterministic_predictions() {
        let stream: Vec<u8> = (0..200).map(|i| (i % 5) as u8).collect();

        let mut pred1 = NeuralSsmPredictor::new();
        let mut pred2 = NeuralSsmPredictor::new();

        for &b in &stream {
            let p1 = pred1.predict();
            let p2 = pred2.predict();
            assert_eq!(p1, p2, "Predictions must be identical for same input");
            pred1.update(b);
            pred2.update(b);
        }
    }

    #[test]
    fn roundtrip_with_range_coder() {
        use crate::coding::rans;

        let rle_data: Vec<u8> = (0..500)
            .map(|i| match i % 7 {
                0..=2 => 0,
                3 => 1,
                4 => 3,
                5 => 0,
                _ => 5,
            })
            .collect();

        let mut enc = NeuralSsmPredictor::new();
        let compressed = rans::encode_block(&rle_data, &mut enc).unwrap();

        let mut dec = NeuralSsmPredictor::new();
        let decoded = rans::decode_block(&compressed, rle_data.len(), &mut dec).unwrap();

        assert_eq!(rle_data, decoded);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut pred = NeuralSsmPredictor::new();
        // Dirty every field by training on varied data
        let dirty: Vec<u8> = (0..200).map(|i| (i * 7 + 3) as u8).collect();
        for &b in &dirty {
            pred.predict();
            pred.update(b);
        }
        pred.reset();

        // Verify field-by-field against a fresh instance
        let mut fresh = NeuralSsmPredictor::new();
        assert_eq!(pred.h, fresh.h, "h mismatch after reset");
        assert_eq!(pred.w_run, fresh.w_run, "w_run mismatch");
        assert_eq!(pred.b_run, fresh.b_run, "b_run mismatch");
        assert_eq!(pred.w_runa, fresh.w_runa, "w_runa mismatch");
        assert_eq!(pred.b_runa, fresh.b_runa, "b_runa mismatch");
        assert_eq!(pred.ssm_perf, fresh.ssm_perf, "ssm_perf mismatch");
        assert_eq!(pred.rle_perf, fresh.rle_perf, "rle_perf mismatch");
        assert_eq!(
            pred.o2_lit_totals, fresh.o2_lit_totals,
            "o2_lit_totals mismatch"
        );
        assert_eq!(pred.prev_byte, fresh.prev_byte, "prev_byte mismatch");
        assert_eq!(
            pred.prev_prev_byte, fresh.prev_prev_byte,
            "prev_prev_byte mismatch"
        );
        assert_eq!(
            pred.last_rle_probs, fresh.last_rle_probs,
            "last_rle_probs mismatch"
        );
        assert_eq!(
            pred.last_ssm_p_run, fresh.last_ssm_p_run,
            "last_ssm_p_run mismatch"
        );
        assert_eq!(
            pred.last_ssm_p_runa, fresh.last_ssm_p_runa,
            "last_ssm_p_runa mismatch"
        );
        assert_eq!(
            pred.last_rle_p_run, fresh.last_rle_p_run,
            "last_rle_p_run mismatch"
        );
        assert_eq!(
            pred.last_rle_p_runa, fresh.last_rle_p_runa,
            "last_rle_p_runa mismatch"
        );
        assert_eq!(pred.step, fresh.step, "step mismatch");
        assert_eq!(
            *pred.o2_lit_counts, *fresh.o2_lit_counts,
            "o2_lit_counts mismatch"
        );

        // Prediction equivalence: feed same stream to both, verify identical output
        let stream: Vec<u8> = (0..100).map(|i| (i % 5) as u8).collect();
        for &b in &stream {
            let p1 = pred.predict();
            let p2 = fresh.predict();
            assert_eq!(
                p1, p2,
                "Prediction diverged after reset at step {}",
                pred.step
            );
            pred.update(b);
            fresh.update(b);
        }
    }

    #[test]
    fn save_load_state_roundtrip_preserves_predictions() {
        // Train on varied data so multiple order-2 contexts are non-empty.
        // Empty contexts (the previous tests' regime) hid the counts/totals
        // block-vs-interleaved layout bug because 0 == 0 always validated.
        let mut pred = NeuralSsmPredictor::new();
        let training: Vec<u8> = (0..5000u32)
            .map(|i| ((i * 31 + i / 7) % 256) as u8)
            .collect();
        for &b in &training {
            pred.predict();
            pred.update(b);
        }

        let state = pred.save_state().expect("ssm supports save_state");
        let mut loaded = NeuralSsmPredictor::new();
        assert!(
            loaded.load_state(&state),
            "load_state must accept its own save_state output"
        );

        // Learned tables must survive the round-trip exactly.
        assert_eq!(pred.o2_lit_totals, loaded.o2_lit_totals, "o2_lit_totals");
        assert_eq!(*pred.o2_lit_counts, *loaded.o2_lit_counts, "o2_lit_counts");
        assert_eq!(pred.w_run, loaded.w_run, "w_run");
        assert_eq!(pred.w_runa, loaded.w_runa, "w_runa");
        assert_eq!(pred.step, loaded.step, "step");

        // Behavioral equality: predict() before update() on both so the
        // cached last_* fields are set identically each step.
        let stream: Vec<u8> = (0..300u32).map(|i| (i % 256) as u8).collect();
        for &b in &stream {
            assert_eq!(
                pred.predict(),
                loaded.predict(),
                "prediction diverged after save/load at step {}",
                pred.step
            );
            pred.update(b);
            loaded.update(b);
        }
    }

    #[test]
    fn reset_restores_dict_baseline_not_zero() {
        // Build a representative baseline by training, then capture it.
        let training: Vec<u8> = (0..3000u32)
            .map(|i| ((i * 7 + i / 3) % 256) as u8)
            .collect();
        let mut trained = NeuralSsmPredictor::new();
        for &b in &training {
            trained.predict();
            trained.update(b);
        }
        let baseline = trained.save_state().unwrap();

        // Installing the baseline immediately sets state to it.
        let mut pred = NeuralSsmPredictor::new();
        assert!(pred.set_dict_baseline(&baseline));
        assert_eq!(pred.save_state().unwrap(), baseline);
        assert_eq!(pred.coding_baseline(), Some(baseline.as_slice()));

        // Dirty the state, then reset() must restore the BASELINE, not zero.
        for &b in b"completely different bytes 12345" {
            pred.predict();
            pred.update(b);
        }
        pred.reset();
        assert_eq!(
            pred.save_state().unwrap(),
            baseline,
            "reset() with a dict baseline must restore the baseline, not zero"
        );

        // Without a baseline, reset() returns to the fresh (zeroed) state.
        let mut plain = NeuralSsmPredictor::new();
        plain.update(5);
        plain.reset();
        assert_eq!(
            plain.save_state().unwrap(),
            NeuralSsmPredictor::new().save_state().unwrap(),
            "reset() without a baseline must zero the state"
        );
    }

    #[test]
    fn reset_prediction_equivalence() {
        // Full roundtrip: train, reset, retrain — must match fresh predictor
        let stream: Vec<u8> = (0..150).map(|i| (i * 13 + 7) as u8).collect();

        let mut pred = NeuralSsmPredictor::new();
        for &b in &stream {
            pred.predict();
            pred.update(b);
        }
        pred.reset();

        let mut fresh = NeuralSsmPredictor::new();
        for &b in &stream {
            let p1 = pred.predict();
            let p2 = fresh.predict();
            assert_eq!(p1, p2, "Predictions must match after reset");
            pred.update(b);
            fresh.update(b);
        }
    }

    /// Measure cross-entropy (bits per byte) of a predictor on given data.
    /// Measure bits-per-byte and elapsed time for a predictor on given data.
    fn measure_bpb(pred: &mut dyn ProbabilityPredictor, data: &[u8]) -> (f64, std::time::Duration) {
        let start = std::time::Instant::now();
        let mut total_bits = 0.0f64;
        for &byte in data {
            let probs = pred.predict();
            let p = (probs[byte as usize] as f64).max(1e-12);
            total_bits -= p.log2();
            pred.update(byte);
        }
        let elapsed = start.elapsed();
        (total_bits / data.len() as f64, elapsed)
    }

    /// Load all large test fixtures, run BWT+MTF+RLE, return the RLE stream.
    fn load_rle_corpus() -> Vec<u8> {
        use crate::coding::bwt_preprocess;

        let mut all_data = Vec::new();
        // Try both workspace root and crate-relative paths
        let candidates = ["tests/fixtures/large", "../tests/fixtures/large"];
        let fixture_dir = candidates
            .iter()
            .map(std::path::Path::new)
            .find(|p| p.exists())
            .expect("Cannot find tests/fixtures/large");
        if fixture_dir.exists() {
            for entry in std::fs::read_dir(fixture_dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_file() {
                    let data = std::fs::read(&path).unwrap();
                    all_data.extend_from_slice(&data);
                }
            }
        }
        assert!(!all_data.is_empty(), "No test fixtures found");

        let (_, mtf_data) =
            bwt_preprocess::bwt_mtf_encode_parts(&all_data).expect("BWT encode failed");
        bwt_preprocess::rle_encode(&mtf_data).expect("RLE encoding failed")
    }

    #[test]
    #[ignore] // Slow: runs full hyperparameter sweep (~90 KiB BWT+MTF+RLE corpus)
    fn sweep_hyperparameters() {
        let rle_data = load_rle_corpus();
        eprintln!("RLE corpus: {} bytes", rle_data.len());

        // Baseline: pure RlePredictor
        let mut rle_pred = RlePredictor::new();
        let (rle_bpb, rle_time) = measure_bpb(&mut rle_pred, &rle_data);
        let rle_speed = rle_data.len() as f64 / rle_time.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!(
            "RlePredictor baseline: {:.4} bpb  ({:.1} MiB/s, {:.1?})",
            rle_bpb, rle_speed, rle_time
        );

        // Current default
        let mut default_pred = NeuralSsmPredictor::new();
        let (default_bpb, default_time) = measure_bpb(&mut default_pred, &rle_data);
        let default_speed = rle_data.len() as f64 / default_time.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!(
            "NeuralSSM default: {:.4} bpb  ({:.1} MiB/s, {:.1?})",
            default_bpb, default_speed, default_time
        );

        let mut best_bpb = default_bpb;
        let mut best_label = String::from("default");

        // ── Sweep D ──
        for &d in &[4, 8, 12, 16, 20, 24, 32] {
            let cfg = NeuralSsmConfig {
                d,
                ..Default::default()
            };
            let mut pred = NeuralSsmPredictor::with_config(cfg);
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!("  D={d:>2}: {bpb:.4} bpb  ({spd:.1} MiB/s){marker}");
            if bpb < best_bpb {
                best_bpb = bpb;
                best_label = format!("D={d}");
            }
        }

        // ── Sweep LR (with best D so far) ──
        let best_d: usize = best_label
            .strip_prefix("D=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_D);
        for &lr in &[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5] {
            let cfg = NeuralSsmConfig {
                d: best_d,
                lr,
                ..Default::default()
            };
            let mut pred = NeuralSsmPredictor::with_config(cfg);
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!("  D={best_d}, lr={lr:.2}: {bpb:.4} bpb  ({spd:.1} MiB/s){marker}");
            if bpb < best_bpb {
                best_bpb = bpb;
                best_label = format!("D={best_d}, lr={lr}");
            }
        }

        // ── Sweep warmup ──
        let best_lr: f32 = best_label
            .split("lr=")
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_LR);
        for &warmup in &[0, 25, 50, 100, 200, 500] {
            let cfg = NeuralSsmConfig {
                d: best_d,
                lr: best_lr,
                warmup,
                ..Default::default()
            };
            let mut pred = NeuralSsmPredictor::with_config(cfg);
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!("  warmup={warmup}: {bpb:.4} bpb  ({spd:.1} MiB/s){marker}");
            if bpb < best_bpb {
                best_bpb = bpb;
                best_label = format!("D={best_d}, lr={best_lr}, warmup={warmup}");
            }
        }

        // ── Sweep decay range ──
        let best_warmup: u32 = best_label
            .split("warmup=")
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_WARMUP);
        for &(lo, hi) in &[
            (0.5, 0.999),
            (0.7, 0.999),
            (0.85, 0.999),
            (0.9, 0.999),
            (0.9, 0.9999),
            (0.95, 0.9999),
        ] {
            let cfg = NeuralSsmConfig {
                d: best_d,
                lr: best_lr,
                warmup: best_warmup,
                decay_lo: lo,
                decay_hi: hi,
                ..Default::default()
            };
            let mut pred = NeuralSsmPredictor::with_config(cfg);
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!("  decay=[{lo:.2}, {hi:.4}]: {bpb:.4} bpb  ({spd:.1} MiB/s){marker}");
            if bpb < best_bpb {
                best_bpb = bpb;
                best_label =
                    format!("D={best_d}, lr={best_lr}, warmup={best_warmup}, decay=[{lo},{hi}]");
            }
        }

        // ── Sweep mix_sensitivity + max_alpha ──
        for &sens in &[10.0, 20.0, 30.0, 50.0, 100.0] {
            for &alpha in &[0.3, 0.5, 0.7, 0.9] {
                let cfg = NeuralSsmConfig {
                    d: best_d,
                    lr: best_lr,
                    warmup: best_warmup,
                    mix_sensitivity: sens,
                    max_alpha: alpha,
                    ..Default::default()
                };
                let mut pred = NeuralSsmPredictor::with_config(cfg);
                let (bpb, t) = measure_bpb(&mut pred, &rle_data);
                let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
                if bpb < best_bpb {
                    best_bpb = bpb;
                    best_label = format!("sens={sens}, alpha={alpha}");
                    eprintln!(
                        "  sens={sens:.0}, alpha={alpha:.1}: {bpb:.4} bpb  ({spd:.1} MiB/s) ***"
                    );
                }
            }
        }

        // ── Sweep order-2 literal blend ──
        eprintln!("  --- order-2 literal blend ---");
        for &blend in &[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5] {
            for &min_obs in &[5, 10, 20, 50] {
                let cfg = NeuralSsmConfig {
                    o2_lit_blend: blend,
                    o2_min_obs: min_obs,
                    ..Default::default()
                };
                let mut pred = NeuralSsmPredictor::with_config(cfg);
                let (bpb, t) = measure_bpb(&mut pred, &rle_data);
                let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
                if bpb < best_bpb {
                    best_bpb = bpb;
                    best_label = format!("o2_blend={blend}, min_obs={min_obs}");
                    eprintln!("  o2_blend={blend:.2}, min_obs={min_obs}: {bpb:.4} bpb  ({spd:.1} MiB/s) ***");
                } else if blend == 0.0 && min_obs == 5 {
                    eprintln!("  o2_blend=0 (disabled): {bpb:.4} bpb  ({spd:.1} MiB/s)");
                }
            }
        }

        eprintln!("\n=== BEST: {best_label} → {best_bpb:.4} bpb (RLE baseline: {rle_bpb:.4}) ===");
        // The SSM should not be worse than pure RLE
        assert!(
            best_bpb <= rle_bpb + 0.001,
            "SSM should not be worse than RLE"
        );
    }

    #[test]
    #[ignore] // Slow: runs head-to-head comparison on ~90 KiB BWT+MTF+RLE corpus
    fn head_to_head_configs() {
        let rle_data = load_rle_corpus();
        eprintln!("RLE corpus: {} bytes", rle_data.len());

        let configs: Vec<(&str, NeuralSsmConfig)> = vec![
            (
                "RlePredictor only",
                NeuralSsmConfig {
                    max_alpha: 0.0,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=4 lr=0.05 o2=0",
                NeuralSsmConfig {
                    d: 4,
                    lr: 0.05,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=4 lr=0.05 o2=0.3",
                NeuralSsmConfig {
                    d: 4,
                    lr: 0.05,
                    o2_lit_blend: 0.3,
                    ..Default::default()
                },
            ),
            (
                "D=20 lr=0.02 o2=0",
                NeuralSsmConfig {
                    d: 20,
                    lr: 0.02,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=20 lr=0.02 o2=0.1",
                NeuralSsmConfig {
                    d: 20,
                    lr: 0.02,
                    o2_lit_blend: 0.1,
                    ..Default::default()
                },
            ),
            (
                "D=20 lr=0.02 o2=0.3",
                NeuralSsmConfig {
                    d: 20,
                    lr: 0.02,
                    o2_lit_blend: 0.3,
                    ..Default::default()
                },
            ),
            (
                "D=16 lr=0.02 o2=0",
                NeuralSsmConfig {
                    d: 16,
                    lr: 0.02,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=16 lr=0.05 o2=0",
                NeuralSsmConfig {
                    d: 16,
                    lr: 0.05,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=8 lr=0.05 o2=0",
                NeuralSsmConfig {
                    d: 8,
                    lr: 0.05,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=20 lr=0.03 o2=0",
                NeuralSsmConfig {
                    d: 20,
                    lr: 0.03,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=20 lr=0.01 o2=0",
                NeuralSsmConfig {
                    d: 20,
                    lr: 0.01,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
        ];

        let mut rle_pred = RlePredictor::new();
        let (rle_bpb, rle_time) = measure_bpb(&mut rle_pred, &rle_data);
        let rle_speed = rle_data.len() as f64 / rle_time.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!(
            "  {:30} → {:.4} bpb  ({:.1} MiB/s, {:.1?})",
            "RlePredictor baseline", rle_bpb, rle_speed, rle_time
        );

        let mut best_bpb = f64::MAX;
        let mut best_name = "";
        for (name, cfg) in &configs {
            let mut pred = NeuralSsmPredictor::with_config(cfg.clone());
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!(
                "  {:30} → {:.4} bpb  ({:.1} MiB/s, {:.1?}){}",
                name, bpb, spd, t, marker
            );
            if bpb < best_bpb {
                best_bpb = bpb;
                best_name = name;
            }
        }
        eprintln!("\n  BEST: {} → {:.4} bpb", best_name, best_bpb);
    }

    // ── Silesia corpus tests (opt-in via SILESIA_PATH env var) ─────────────

    /// Load files from a directory (up to `max_bytes`), run BWT+MTF+RLE and return the
    /// concatenated RLE stream alongside a list of included file descriptions.
    ///
    /// Files are processed in 512 KiB chunks (matching the real pipeline's FastCDC max
    /// block size), so BWT runs on manageable blocks rather than whole multi-MB files.
    /// Chunks where `rle_encode` returns None (binary data with MTF byte 255) are skipped
    /// — in the real pipeline those chunks are routed to LZ77 or Zstd instead.
    fn load_rle_corpus_from(dir: &str, max_bytes: usize) -> (Vec<u8>, Vec<String>) {
        use crate::coding::bwt_preprocess;
        const CHUNK_SIZE: usize = 512 * 1024; // 512 KiB — matches FastCDC max block
        let mut rle_stream: Vec<u8> = Vec::new();
        let mut used: Vec<String> = Vec::new();
        let mut skipped_files: Vec<String> = Vec::new();
        let mut bytes_processed = 0usize;
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|_| panic!("Cannot read directory: {dir}"))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        paths.sort(); // deterministic order
        for path in &paths {
            if bytes_processed >= max_bytes {
                break;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let remaining = max_bytes - bytes_processed;
            let file_data = std::fs::read(path).unwrap();
            let take = file_data.len().min(remaining);
            let data = &file_data[..take];

            // Process file in 512 KiB BWT chunks (mirrors the real routing pipeline).
            let mut file_rle_bytes = 0usize;
            let mut file_chunks_ok = 0u32;
            let mut file_chunks_skip = 0u32;
            for chunk_start in (0..take).step_by(CHUNK_SIZE) {
                let chunk_end = (chunk_start + CHUNK_SIZE).min(take);
                let chunk = &data[chunk_start..chunk_end];
                let Ok((_, mtf_data)) = bwt_preprocess::bwt_mtf_encode_parts(chunk) else {
                    file_chunks_skip += 1;
                    continue;
                };
                match bwt_preprocess::rle_encode(&mtf_data) {
                    Some(rle) => {
                        rle_stream.extend_from_slice(&rle);
                        file_rle_bytes += rle.len();
                        file_chunks_ok += 1;
                    }
                    None => {
                        file_chunks_skip += 1;
                    }
                }
            }

            if file_chunks_ok > 0 {
                bytes_processed += take;
                used.push(format!(
                    "{name} ({:.1} MiB → {} RLE bytes, {file_chunks_skip} chunks skipped)",
                    take as f64 / 1_048_576.0,
                    file_rle_bytes
                ));
            } else {
                skipped_files.push(format!(
                    "{name} ({:.1} MiB, all chunks binary)",
                    take as f64 / 1_048_576.0
                ));
            }
        }
        assert!(
            !rle_stream.is_empty(),
            "No RLE-encodable data found in {dir}"
        );
        if !skipped_files.is_empty() {
            eprintln!(
                "  [fully skipped {} file(s): {}]",
                skipped_files.len(),
                skipped_files.join(", ")
            );
        }
        (rle_stream, used)
    }

    /// Hyperparameter sweep on the Silesia corpus.
    /// Run with:  SILESIA_PATH=/tmp/silesia cargo test -p aether-core --release \
    ///              -- neural_ssm::tests::sweep_on_silesia --nocapture --ignored
    #[test]
    #[ignore]
    fn sweep_on_silesia() {
        let silesia_path =
            std::env::var("SILESIA_PATH").unwrap_or_else(|_| "/tmp/silesia".to_string());
        if !std::path::Path::new(&silesia_path).exists() {
            eprintln!("SKIP: Silesia corpus not found at {silesia_path}");
            eprintln!("      Set SILESIA_PATH or place files in /tmp/silesia/");
            return;
        }
        // Use up to 32 MiB so the sweep completes in a reasonable time (~10 min).
        // Text-heavy files come first alphabetically: dickens, nci, osdb, reymont.
        let max_bytes = 32 * 1024 * 1024;
        let (rle_data, used_files) = load_rle_corpus_from(&silesia_path, max_bytes);
        eprintln!(
            "Silesia RLE corpus: {} bytes (capped at {} MiB)",
            rle_data.len(),
            max_bytes / (1024 * 1024)
        );
        for f in &used_files {
            eprintln!("  included: {f}");
        }

        let mut rle_pred = RlePredictor::new();
        let (rle_bpb, rle_time) = measure_bpb(&mut rle_pred, &rle_data);
        let rle_speed = rle_data.len() as f64 / rle_time.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!(
            "RlePredictor baseline: {:.4} bpb  ({:.1} MiB/s, {:.1?})",
            rle_bpb, rle_speed, rle_time
        );

        let mut default_pred = NeuralSsmPredictor::new();
        let (default_bpb, default_time) = measure_bpb(&mut default_pred, &rle_data);
        let default_speed = rle_data.len() as f64 / default_time.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!(
            "NeuralSSM default (D=20,lr=0.02,o2=0.1): {:.4} bpb  ({:.1} MiB/s, {:.1?})",
            default_bpb, default_speed, default_time
        );

        let mut best_bpb = default_bpb;
        let mut best_label = String::from("default");

        // ── Sweep D ──
        eprintln!("--- D sweep ---");
        for &d in &[4usize, 8, 12, 16, 20, 24, 32] {
            let cfg = NeuralSsmConfig {
                d,
                ..Default::default()
            };
            let mut pred = NeuralSsmPredictor::with_config(cfg);
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!("  D={d:>2}: {bpb:.4} bpb  ({spd:.1} MiB/s){marker}");
            if bpb < best_bpb {
                best_bpb = bpb;
                best_label = format!("D={d}");
            }
        }

        // ── Sweep LR (with best D) ──
        let best_d: usize = best_label
            .strip_prefix("D=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_D);
        eprintln!("--- lr sweep (D={best_d}) ---");
        for &lr in &[0.005f32, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2] {
            let cfg = NeuralSsmConfig {
                d: best_d,
                lr,
                ..Default::default()
            };
            let mut pred = NeuralSsmPredictor::with_config(cfg);
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!("  lr={lr:.3}: {bpb:.4} bpb  ({spd:.1} MiB/s){marker}");
            if bpb < best_bpb {
                best_bpb = bpb;
                best_label = format!("D={best_d},lr={lr}");
            }
        }

        // ── Sweep decay range ──
        let best_lr: f32 = best_label
            .split("lr=")
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_LR);
        eprintln!("--- decay sweep (D={best_d},lr={best_lr}) ---");
        for &(lo, hi) in &[
            (0.5f32, 0.999f32),
            (0.7, 0.999),
            (0.85, 0.999),
            (0.9, 0.999),
            (0.9, 0.9999),
            (0.95, 0.9999),
            (0.5, 0.9999),
        ] {
            let cfg = NeuralSsmConfig {
                d: best_d,
                lr: best_lr,
                decay_lo: lo,
                decay_hi: hi,
                ..Default::default()
            };
            let mut pred = NeuralSsmPredictor::with_config(cfg);
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!("  decay=[{lo:.2},{hi:.4}]: {bpb:.4} bpb  ({spd:.1} MiB/s){marker}");
            if bpb < best_bpb {
                best_bpb = bpb;
                best_label = format!("D={best_d},lr={best_lr},decay=[{lo},{hi}]");
            }
        }

        // ── Sweep mix sensitivity + max_alpha ──
        eprintln!("--- mixer sweep ---");
        for &sens in &[10.0f32, 30.0, 50.0, 100.0, 200.0] {
            for &alpha in &[0.5f32, 0.7, 0.9] {
                let cfg = NeuralSsmConfig {
                    d: best_d,
                    lr: best_lr,
                    mix_sensitivity: sens,
                    max_alpha: alpha,
                    ..Default::default()
                };
                let mut pred = NeuralSsmPredictor::with_config(cfg);
                let (bpb, t) = measure_bpb(&mut pred, &rle_data);
                let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
                if bpb < best_bpb {
                    best_bpb = bpb;
                    best_label = format!("sens={sens},alpha={alpha}");
                    eprintln!(
                        "  sens={sens:.0},alpha={alpha:.1}: {bpb:.4} bpb  ({spd:.1} MiB/s) ***"
                    );
                }
            }
        }

        // ── Sweep o2 blend ──
        eprintln!("--- o2 blend sweep ---");
        for &blend in &[0.0f32, 0.05, 0.1, 0.15, 0.2, 0.3] {
            for &min_obs in &[5u32, 10, 20] {
                let cfg = NeuralSsmConfig {
                    o2_lit_blend: blend,
                    o2_min_obs: min_obs,
                    ..Default::default()
                };
                let mut pred = NeuralSsmPredictor::with_config(cfg);
                let (bpb, t) = measure_bpb(&mut pred, &rle_data);
                let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
                if bpb < best_bpb {
                    best_bpb = bpb;
                    best_label = format!("o2={blend},min={min_obs}");
                    eprintln!(
                        "  o2={blend:.2},min_obs={min_obs}: {bpb:.4} bpb  ({spd:.1} MiB/s) ***"
                    );
                } else if blend == 0.0 && min_obs == 5 {
                    eprintln!("  o2=0 (disabled): {bpb:.4} bpb  ({spd:.1} MiB/s)");
                }
            }
        }

        eprintln!("\n=== SILESIA BEST: {best_label} → {best_bpb:.4} bpb  (RLE baseline: {rle_bpb:.4}) ===");
        assert!(
            best_bpb <= rle_bpb + 0.001,
            "SSM should not be worse than RLE"
        );
    }

    /// Head-to-head on Silesia: compare key configs including the current default.
    /// Run with:  SILESIA_PATH=/tmp/silesia cargo test -p aether-core --release \
    ///              -- neural_ssm::tests::head_to_head_on_silesia --nocapture --ignored
    #[test]
    #[ignore]
    fn head_to_head_on_silesia() {
        let silesia_path =
            std::env::var("SILESIA_PATH").unwrap_or_else(|_| "/tmp/silesia".to_string());
        if !std::path::Path::new(&silesia_path).exists() {
            eprintln!("SKIP: Silesia corpus not found at {silesia_path}");
            return;
        }
        let max_bytes = 32 * 1024 * 1024;
        let (rle_data, used_files) = load_rle_corpus_from(&silesia_path, max_bytes);
        eprintln!("Silesia RLE corpus: {} bytes", rle_data.len());
        for f in &used_files {
            eprintln!("  included: {f}");
        }

        let configs: Vec<(&str, NeuralSsmConfig)> = vec![
            (
                "RleOnly (alpha=0)",
                NeuralSsmConfig {
                    max_alpha: 0.0,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=20 lr=0.02 o2=0.1 ★",
                NeuralSsmConfig {
                    d: 20,
                    lr: 0.02,
                    o2_lit_blend: 0.1,
                    ..Default::default()
                },
            ),
            (
                "D=32 lr=0.02 o2=0",
                NeuralSsmConfig {
                    d: 32,
                    lr: 0.02,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=32 lr=0.02 o2=0.1",
                NeuralSsmConfig {
                    d: 32,
                    lr: 0.02,
                    o2_lit_blend: 0.1,
                    ..Default::default()
                },
            ),
            (
                "D=32 lr=0.01 o2=0.1",
                NeuralSsmConfig {
                    d: 32,
                    lr: 0.01,
                    o2_lit_blend: 0.1,
                    ..Default::default()
                },
            ),
            (
                "D=20 lr=0.01 o2=0.1",
                NeuralSsmConfig {
                    d: 20,
                    lr: 0.01,
                    o2_lit_blend: 0.1,
                    ..Default::default()
                },
            ),
            (
                "D=20 lr=0.02 o2=0",
                NeuralSsmConfig {
                    d: 20,
                    lr: 0.02,
                    o2_lit_blend: 0.0,
                    ..Default::default()
                },
            ),
            (
                "D=12 lr=0.02 o2=0.1",
                NeuralSsmConfig {
                    d: 12,
                    lr: 0.02,
                    o2_lit_blend: 0.1,
                    ..Default::default()
                },
            ),
            (
                "D=32 decay=[0.5,0.9999]",
                NeuralSsmConfig {
                    d: 32,
                    lr: 0.02,
                    decay_lo: 0.5,
                    decay_hi: 0.9999,
                    ..Default::default()
                },
            ),
        ];

        let mut rle_pred = RlePredictor::new();
        let (rle_bpb, rle_time) = measure_bpb(&mut rle_pred, &rle_data);
        let rle_spd = rle_data.len() as f64 / rle_time.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!(
            "  {:35} → {:.4} bpb  ({:.1} MiB/s)",
            "RlePredictor baseline", rle_bpb, rle_spd
        );

        let mut best_bpb = f64::MAX;
        let mut best_name = "";
        for (name, cfg) in &configs {
            let mut pred = NeuralSsmPredictor::with_config(cfg.clone());
            let (bpb, t) = measure_bpb(&mut pred, &rle_data);
            let spd = rle_data.len() as f64 / t.as_secs_f64() / (1024.0 * 1024.0);
            let marker = if bpb < best_bpb { " ***" } else { "" };
            eprintln!(
                "  {:35} → {:.4} bpb  ({:.1} MiB/s, {:.0?}){}",
                name, bpb, spd, t, marker
            );
            if bpb < best_bpb {
                best_bpb = bpb;
                best_name = name;
            }
        }
        eprintln!(
            "\n  SILESIA BEST: {} → {:.4} bpb  (baseline: {:.4})",
            best_name, best_bpb, rle_bpb
        );
    }
}
