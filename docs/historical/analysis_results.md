# AetherArch Ratio & Speed Improvements (Amended)

Based on an analysis of the AetherArch pipeline and a post-mortem of the `perf/router-decode-optimization` branch, here are targeted suggestions to improve both compression ratio and compression/decompression speed.

## 🚀 Speed Improvements

1. **Zero-Allocation Predictor Reset (The `router-decode-optimization` Fix)**
   - **Post-Mortem:** The `perf/router-decode-optimization` branch attempted to save memory allocations by reusing a `scratch = NeuralSsmPredictor::new()` instance across the `BwtPredictorRans`, `Lz77PredictorRans`, and `PredictorRans` trials in `router.rs`. However, it inadvertently made speed worse. Why? Because `rans::encode_block` calls `predictor.reset()`, and `NeuralSsmPredictor::reset()` is implemented as `*self = Self::with_config(...)`. This means it *still* performs two heavy `Box::new` allocations (~33 KiB) and loop-recalculates $256 \times 32 = 8,192$ deterministic floats via `det_hash()` *for every chunk trial*. Reusing `scratch` simply shifted when the heavy penalty occurred, adding overhead without skipping the allocations.
   - **Suggestion:** Rewrite `NeuralSsmPredictor::reset(&mut self)` to be completely in-place. The `embed` array is deterministic and constant — it never needs to be recomputed. By simply calling `.fill(0)` on `self.h`, `self.o2_lit_counts`, `self.w_run`, etc., and resetting the primitives (`self.step = 0`), you will achieve the true zero-allocation promise of the branch. This single change will drastically increase throughput while perfectly preserving the ratio.

2. **Parallelization Defaults (`compress.rs`)**
   - **Current:** `DEFAULT_MAX_THREADS` is hardcoded to `4`.
   - **Suggestion:** Dynamically scale `max_threads` based on `std::thread::available_parallelism()`, bounded by available system RAM. Since predictors like `NeuralSsmPredictor` require heap allocations (and PAQ mixers are even heavier), polling system memory before spawning thread pools avoids OOMs while maximizing throughput on modern 16/32-core systems.

3. **BWT Entropy Skip Tuning (`router.rs`)**
   - **Current:** `BWT_ENTROPY_SKIP` is conservatively set to `7.0`. 
   - **Suggestion:** Computing the suffix array (via `libsais`) is very expensive. Reducing this threshold to `6.0` or `6.5` ensures that only highly structured data pays the penalty of BWT. Data with entropy around `6.5-7.0` is rarely clustered efficiently by BWT before MTF, so `Lz77PredictorRans` or `PredictorRans` will likely win anyway while skipping `libsais` makes the process significantly faster.

4. **LZ4 vs LZ77 Preprocessing (`router.rs`)**
   - **Current:** Both `lz77_preprocess` and `lz4_flex` (`LzPredictorRans` behind the `lz4` feature) exist.
   - **Suggestion:** Fully default to `lz4_flex` (or zstd negative compression levels) instead of bespoke `lz77_preprocess` for the LZ step. `lz4_flex` provides SIMD-accelerated matching which drastically speeds up the LZ search phase prior to range coding.

---

## 📈 Ratio Improvements

1. **Larger Maximum Chunk Size (`chunker.rs`)**
   - **Current:** `MAX_CHUNK_SIZE` is capped at `4 MiB`.
   - **Suggestion:** Increase the max chunk size to `16 MiB` or `32 MiB`. BWT clustering and LZ77 matching rely heavily on long-range repetitions. While small chunks help threading, `4 MiB` splits large text corpuses (or JSON lines) too frequently, resetting the BWT sequence and degrading the overall ratio. A larger unified block provides much better BWT runs.

2. **Executable Filters (BCJ) (`analyzer.rs`)**
   - **Current:** `ContentType::Executable` (ELF, PE, Mach-O) mostly relies on the generic `recommend_method_for()` which routes to `PredictorRans`.
   - **Suggestion:** Implement a Branch/Call/Jump (BCJ) filter (similar to XZ/7-Zip) for x86/x64 and ARM. Normalizing relative jump offsets into absolute addresses before compression significantly clusters byte repetitions, allowing BWT/LZ77 to find far more matches in compiled binaries.

3. **Dedicated Float Splitting for High-Bit Data**
   - **Current:** `byteplane_preprocess` detects numeric width for Numpy/Safetensors. 
   - **Suggestion:** Many scientific datasets compress far better if you not only byte-plane split, but additionally apply a Delta filter across planes (subtracting the previous value). Adding an explicit delta-encoding step before `BytePlanePredictorRans` shrinks the dynamic range substantially.
