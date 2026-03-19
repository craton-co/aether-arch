//! Multi-order context-mixing predictor (PAQ-inspired).
//!
//! Combines predictions from order-1 through order-8 context models using
//! logistic mixing in log-odds space. Mixing weights are adapted online
//! via gradient descent, giving more weight to models that predict well.
//!
//! This is the V1 "secret weapon" — it should achieve ~2.0–2.5 bits/byte
//! on English text (vs gzip's ~3.0, zstd's ~2.7).

use super::traits::ProbabilityPredictor;
use crate::format::PredictorId;

// ── Single Order-N Model ─────────────────────────────────────────────────────

/// A single order-N context model.
///
/// Maintains a hash table mapping `hash(last N bytes)` → `[u16; 256]` frequency
/// counts. Uses FNV-1a hashing and Laplace smoothing.
struct OrderNModel {
    order: usize,
    /// Hash table: each entry is a 256-element frequency array.
    table: Vec<[u16; 256]>,
    table_mask: usize,
}

/// Maximum allowed table_bits to prevent excessive memory allocation.
/// 20 bits = 1M entries * 512 bytes = 512 MiB per model, which is already very large.
const MAX_TABLE_BITS: usize = 20;

/// Maximum number of sub-models in a context mixer. Prevents unbounded
/// memory and CPU from configs with thousands of tiny models.
const MAX_NUM_MODELS: usize = 16;

impl OrderNModel {
    fn new(order: usize, table_bits: usize) -> Self {
        let capped_bits = table_bits.min(MAX_TABLE_BITS);
        let size = 1usize << capped_bits;
        Self {
            order,
            table: vec![[1u16; 256]; size],
            table_mask: size - 1,
        }
    }

    /// Predict P(byte | context) for this single order model.
    fn predict(&self, context: &[u8]) -> [f32; 256] {
        if context.len() < self.order {
            // Not enough context — return uniform
            return [1.0 / 256.0; 256];
        }

        let ctx = &context[context.len() - self.order..];
        let hash = Self::hash_context(ctx);
        let counts = &self.table[hash & self.table_mask];
        let total: u32 = counts.iter().map(|&c| c as u32).sum();
        let inv_total = 1.0 / total as f32;

        let mut probs = [0.0f32; 256];
        for i in 0..256 {
            probs[i] = counts[i] as f32 * inv_total;
        }
        probs
    }

    /// Update the frequency table with an observed byte.
    fn update(&mut self, context: &[u8], byte: u8) {
        if context.len() < self.order {
            return;
        }

        let ctx = &context[context.len() - self.order..];
        let hash = Self::hash_context(ctx);
        let counts = &mut self.table[hash & self.table_mask];
        counts[byte as usize] = counts[byte as usize].saturating_add(1);

        // Rescale if any count gets too high (prevents overflow, keeps adaptive)
        if counts[byte as usize] >= 8000 {
            for c in counts.iter_mut() {
                *c = (*c >> 1).max(1);
            }
        }
    }

    /// FNV-1a hash of a context byte slice.
    ///
    /// Uses the standard FNV-1a algorithm with fixed constants, which is
    /// deterministic across all Rust versions and platforms (unlike
    /// `DefaultHasher` whose algorithm and seed are not guaranteed stable).
    fn hash_context(ctx: &[u8]) -> usize {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        for &b in ctx {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3); // FNV prime
        }
        h as usize
    }

    fn reset(&mut self) {
        for entry in self.table.iter_mut() {
            *entry = [1u16; 256];
        }
    }

    /// Approximate memory usage in bytes.
    fn memory_usage(&self) -> usize {
        self.table.len() * 256 * std::mem::size_of::<u16>()
    }
}

// ── Context Mixer ────────────────────────────────────────────────────────────

/// Context-mixing predictor combining multiple order-N models.
///
/// Uses logistic mixing: each model's prediction is converted to log-odds,
/// weighted, summed, and converted back to probability. Weights are adapted
/// online using a simplified gradient descent.
///
/// # Cross-Platform Determinism
///
/// **Warning**: This predictor uses `f64::ln()` and `f64::exp()` in the
/// mixing step, which are implemented by the platform's math library and
/// may differ by 1-2 ULP across architectures (x86 vs ARM vs WASM).
/// Archives compressed with `ContextMixer` are **not guaranteed to
/// decompress identically on all platforms**. For cross-platform
/// deterministic archives, prefer [`RlePredictor`](super::RlePredictor)
/// or [`Order0Model`](super::Order0Model).
///
/// # Memory
///
/// The default configuration allocates **~100 MiB** for hash tables
/// (see [`ContextMixerConfig::default`]).  When used with parallel
/// solid-group compression (rayon), each worker thread creates its own
/// `ContextMixer` instance.  With 8 parallel groups, peak memory can
/// reach **~800 MiB** just for predictors — enough to thrash the L3
/// cache on most desktop CPUs.
///
/// **Recommendations for parallel workloads:**
/// - Use [`ContextMixerConfig::lightweight`] (~4 MiB per instance)
/// - Or limit rayon's thread pool size via `RAYON_NUM_THREADS`
/// - Or use [`NeuralSsmPredictor`](super::NeuralSsmPredictor) (~25 KiB)
///   which provides better compression ratios at a fraction of the memory.
pub struct ContextMixer {
    models: Vec<OrderNModel>,
    /// Mixing weights (positive, sum to ~1.0).
    weights: Vec<f64>,
    /// Learning rate for online weight adaptation.
    learning_rate: f64,
    /// Rolling context buffer (last `max_context` bytes).
    context_buf: std::collections::VecDeque<u8>,
    max_context: usize,
    /// Whether this is a lightweight configuration (for header ID).
    is_lightweight: bool,
    /// Cached per-model predictions from the last `predict()` call,
    /// reused in `update()` to avoid double computation.
    cached_predictions: Vec<[f32; 256]>,
    /// True after predict() and before update() consumes cache.
    predict_called: bool,
}

impl ContextMixer {
    /// Create a new context mixer with default configuration.
    ///
    /// Uses order-1 through order-6 models. Total memory: ~100 MiB.
    pub fn new() -> Self {
        Self::with_config(ContextMixerConfig::default())
    }

    /// Maximum total memory budget (512 MiB). Configs exceeding this are
    /// rejected and fall back to [`ContextMixerConfig::lightweight`].
    const MAX_TOTAL_MEMORY: usize = 512 * 1024 * 1024;

    /// Create with explicit configuration.
    ///
    /// If the config would exceed `Self::MAX_TOTAL_MEMORY`, it silently falls
    /// back to [`ContextMixerConfig::lightweight`] to prevent OOM.
    pub fn with_config(config: ContextMixerConfig) -> Self {
        // Validate model count and memory budget before allocating.
        // Fall back to lightweight config if limits are exceeded.
        let config = if config.orders.len() > MAX_NUM_MODELS || config.orders.is_empty() {
            ContextMixerConfig::lightweight()
        } else {
            config
        };
        let estimated_mem: usize = config
            .orders
            .iter()
            .map(|&(_, bits)| {
                let capped = bits.min(MAX_TABLE_BITS);
                (1usize << capped) * 256 * std::mem::size_of::<u16>()
            })
            .sum();
        let config = if estimated_mem > Self::MAX_TOTAL_MEMORY {
            ContextMixerConfig::lightweight()
        } else {
            config
        };

        let models: Vec<OrderNModel> = config
            .orders
            .iter()
            .map(|&(order, table_bits)| OrderNModel::new(order, table_bits))
            .collect();

        let n = models.len();
        let is_lightweight = config.is_lightweight;
        Self {
            models,
            weights: vec![1.0 / n as f64; n],
            learning_rate: config.learning_rate,
            context_buf: std::collections::VecDeque::with_capacity(config.max_context + 1),
            max_context: config.max_context,
            is_lightweight,
            cached_predictions: Vec::new(),
            predict_called: false,
        }
    }

    /// Approximate total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.models.iter().map(|m| m.memory_usage()).sum::<usize>()
            + self.context_buf.capacity()
            + self.weights.len() * 8
    }

    /// Combine predictions from all models in log-odds space.
    fn mix_predictions(&self, predictions: &[[f32; 256]]) -> [f32; 256] {
        let mut mixed = [0.0f64; 256];

        for (pred, &weight) in predictions.iter().zip(&self.weights) {
            for i in 0..256 {
                // Clamp to avoid log(0) or log(inf)
                let p = (pred[i] as f64).clamp(1e-12, 1.0 - 1e-12);
                // Convert to log-odds (logit)
                let logit = (p / (1.0 - p)).ln();
                mixed[i] += weight * logit;
            }
        }

        // Convert back to probabilities via sigmoid, then normalize
        let mut probs = [0.0f32; 256];
        let mut sum = 0.0f64;
        for i in 0..256 {
            // Clamp logit to avoid exp overflow
            let clamped = mixed[i].clamp(-20.0, 20.0);
            let p = 1.0 / (1.0 + (-clamped).exp());
            probs[i] = p as f32;
            sum += p;
        }

        // Normalize to sum to 1.0, ensuring minimum probability for every byte
        if sum > 0.0 {
            let inv_sum = 1.0 / sum as f32;
            for p in probs.iter_mut() {
                *p *= inv_sum;
                if *p < 1e-7 {
                    *p = 1e-7;
                }
            }
            // Renormalize after flooring to maintain sum == 1.0
            let floored_sum: f32 = probs.iter().sum();
            if floored_sum > 0.0 {
                let inv = 1.0 / floored_sum;
                for p in probs.iter_mut() {
                    *p *= inv;
                }
            }
        }

        probs
    }

    /// Update mixing weights based on which models predicted well.
    fn update_weights(&mut self, predictions: &[[f32; 256]], actual_byte: u8) {
        let target = actual_byte as usize;

        for (i, pred) in predictions.iter().enumerate() {
            let p = (pred[target] as f64).clamp(1e-12, 1.0);
            // Increase weight proportional to log-probability (reward good predictions)
            self.weights[i] *= (1.0 + self.learning_rate * p.ln()).max(0.01);
        }

        // Normalize weights
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in self.weights.iter_mut() {
                *w /= sum;
            }
        }
    }
}

impl Default for ContextMixer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityPredictor for ContextMixer {
    fn predict(&mut self) -> [f32; 256] {
        let ctx = self.context_buf.make_contiguous();
        let predictions: Vec<[f32; 256]> = self.models.iter().map(|m| m.predict(ctx)).collect();
        let mixed = self.mix_predictions(&predictions);
        self.cached_predictions = predictions;
        self.predict_called = true;
        mixed
    }

    fn update(&mut self, byte: u8) {
        // Reuse predictions cached by predict() only if predict() was called
        // first in this cycle. Prevents stale cache from previous cycles.
        let predictions = if self.predict_called && !self.cached_predictions.is_empty() {
            self.predict_called = false;
            std::mem::take(&mut self.cached_predictions)
        } else {
            self.predict_called = false;
            let ctx = self.context_buf.make_contiguous();
            self.models.iter().map(|m| m.predict(ctx)).collect()
        };

        // Adapt mixing weights
        self.update_weights(&predictions, byte);

        // Update each sub-model's frequency tables
        let ctx = self.context_buf.make_contiguous();
        for model in &mut self.models {
            model.update(ctx, byte);
        }

        // Maintain context buffer (VecDeque: O(1) push_back + pop_front)
        self.context_buf.push_back(byte);
        if self.context_buf.len() > self.max_context {
            self.context_buf.pop_front();
        }
    }

    fn reset(&mut self) {
        for model in &mut self.models {
            model.reset();
        }
        self.weights = vec![1.0 / self.models.len() as f64; self.models.len()];
        self.context_buf.clear();
        self.cached_predictions.clear();
        self.predict_called = false;
    }

    fn name(&self) -> &str {
        "context-mixer"
    }

    fn predictor_id(&self) -> PredictorId {
        if self.is_lightweight {
            PredictorId::ContextMixerLight
        } else {
            PredictorId::ContextMixer
        }
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the context mixer.
pub struct ContextMixerConfig {
    /// (order, table_bits) pairs for each sub-model.
    /// `table_bits` determines the hash table size: 2^table_bits entries.
    pub orders: Vec<(usize, usize)>,
    /// Whether this is a lightweight config (affects PredictorId in header).
    pub is_lightweight: bool,
    /// Learning rate for mixing weight adaptation.
    pub learning_rate: f64,
    /// Maximum context length to keep in the rolling buffer.
    pub max_context: usize,
}

impl Default for ContextMixerConfig {
    /// Default: order 1–6 with moderate table sizes.
    /// Total memory: ~100 MiB.
    ///
    /// **Warning**: This allocation is per-predictor.  With rayon-parallel
    /// solid groups, total memory = `~100 MiB × num_groups`.  Consider
    /// [`ContextMixerConfig::lightweight`] or [`NeuralSsmPredictor`](super::NeuralSsmPredictor) for
    /// parallel workloads.
    fn default() -> Self {
        Self {
            orders: vec![
                (1, 14), // 16K entries = 8 MiB
                (2, 16), // 64K entries = 32 MiB
                (3, 16), // 64K entries = 32 MiB
                (4, 15), // 32K entries = 16 MiB
                (5, 14), // 16K entries = 8 MiB
                (6, 13), // 8K entries = 4 MiB
            ],
            is_lightweight: false,
            learning_rate: 0.002,
            max_context: 8,
        }
    }
}

impl ContextMixerConfig {
    /// Lightweight config for testing (~4 MiB).
    pub fn lightweight() -> Self {
        Self {
            orders: vec![
                (1, 12), // 4K entries
                (2, 12), // 4K entries
                (3, 11), // 2K entries
            ],
            is_lightweight: true,
            learning_rate: 0.005,
            max_context: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_prediction_is_roughly_uniform() {
        let mut mixer = ContextMixer::with_config(ContextMixerConfig::lightweight());
        let probs = mixer.predict();

        // All should be positive
        for &p in &probs {
            assert!(p > 0.0);
        }

        // Sum should be ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Initial prediction sum = {sum}");
    }

    #[test]
    fn adapts_to_repeated_pattern() {
        let mut mixer = ContextMixer::with_config(ContextMixerConfig::lightweight());

        // Feed "ABABAB..." pattern
        let pattern = b"AB";
        for _ in 0..500 {
            for &b in pattern {
                mixer.update(b);
            }
        }

        // After seeing "...AB", the mixer should predict 'A' with high probability
        // (since 'A' always follows 'B' in the pattern)
        let probs = mixer.predict();
        let p_a = probs[b'A' as usize];
        let p_b = probs[b'B' as usize];

        // 'A' should be likely (it follows 'B')
        assert!(
            p_a > 0.3,
            "Expected P('A') > 0.3 after AB pattern, got {p_a}"
        );
        // Together they should be significantly above uniform (2/256 = 0.0078)
        assert!(
            p_a + p_b > 0.5,
            "P('A') + P('B') = {} should be well above uniform",
            p_a + p_b
        );
    }

    #[test]
    fn reset_works() {
        let mut mixer = ContextMixer::with_config(ContextMixerConfig::lightweight());

        for _ in 0..1000 {
            mixer.update(42);
        }

        mixer.reset();

        // Should be back to approximately uniform
        let probs = mixer.predict();
        let max_p = probs.iter().cloned().fold(0.0f32, f32::max);
        let min_p = probs.iter().cloned().fold(1.0f32, f32::min);
        assert!(
            (max_p - min_p) < 0.01,
            "After reset, max={max_p} min={min_p} — should be ~uniform"
        );
    }

    #[test]
    fn prediction_always_valid() {
        let mut mixer = ContextMixer::with_config(ContextMixerConfig::lightweight());

        let text = b"Hello, World! This is a test of the context mixing predictor.";
        for &byte in text.iter() {
            let probs = mixer.predict();

            // All positive
            for &p in &probs {
                assert!(p > 0.0, "Zero probability found");
            }

            // Sum to ~1.0
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.05,
                "Sum = {sum} at byte '{}'",
                byte as char
            );

            mixer.update(byte);
        }
    }

    #[test]
    fn deterministic() {
        let text = b"The quick brown fox jumps over the lazy dog. ";

        let run = || {
            let mut mixer = ContextMixer::with_config(ContextMixerConfig::lightweight());
            let mut all_probs = Vec::new();
            for &byte in text.iter() {
                all_probs.push(mixer.predict());
                mixer.update(byte);
            }
            all_probs
        };

        let probs1 = run();
        let probs2 = run();

        for (i, (a, b)) in probs1.iter().zip(probs2.iter()).enumerate() {
            for j in 0..256 {
                assert_eq!(
                    a[j].to_bits(),
                    b[j].to_bits(),
                    "Non-deterministic at byte {i}, symbol {j}"
                );
            }
        }
    }
}
