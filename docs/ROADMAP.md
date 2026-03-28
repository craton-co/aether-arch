# AetherArch Roadmap

Production readiness, licensing, and monetization plan for AetherArch 0.2.3+.

## Phase 1: Foundation (Weeks 1-4) — COMPLETE

Infrastructure, licensing, and critical safety fixes required before any public release.

### Project Infrastructure
- [x] Add CHANGELOG.md (keep-a-changelog format)
- [x] Add CONTRIBUTING.md
- [x] Add SECURITY.md (responsible disclosure policy)
- [x] Add MSRV policy to workspace Cargo.toml
- [x] Add GitHub Actions CI (test, clippy, fmt, audit)
- [x] Add `deny.toml` for cargo-deny (license + CVE audit)

### Code Safety (High severity)
- [x] Add `#[non_exhaustive]` to all public enums
- [x] Add named constants for magic thresholds (BWT_DECISIVE_RATIO)
- [x] Add bounds checking on archive-supplied sizes before allocation
- [x] Add `MAX_DECOMPRESSED_BLOCK_SIZE` safety cap (64 MiB)
- [x] Add `MAX_FILE_COUNT` / `MAX_BLOCK_COUNT` sanity limits

### Documentation
- [x] Add `///` rustdoc to all public API items
- [ ] Run `RUSTDOCFLAGS="-D warnings" cargo doc` and fix all warnings
- [x] Add `examples/` directory with basic usage
  - `basic_compress.rs` — Compress files to .aet archive with analytics
  - `basic_decompress.rs` — Extract + verify archive integrity
  - `streaming_extract.rs` — Two-phase streaming decompression from Read source

### Enterprise Feature Gate
- [x] Add `enterprise` feature flag to aether-core/Cargo.toml
- [x] Document open-source vs enterprise split in ROADMAP

---

## Phase 2: Quality & Security (Weeks 5-8) — COMPLETE

Hardening for external use and first public release.

### Critical Correctness/Safety Fixes
- [x] Align `MAX_DECODE_SIZE` in `rans.rs` with `MAX_DECOMPRESSED_BLOCK_SIZE` (64 MiB)
  - Previous 16 MiB limit could silently corrupt blocks near the limit
  - Added bounds checks on `encoded_len` / `lz_len` in `decompress_chunk`
- [x] Add `MAX_BWT_INPUT_SIZE` guard (8 MiB) to prevent OOM from doubled-text allocation
  - BWT allocates ~10× input for doubled text + suffix array
  - With parallel groups, unguarded input causes ~80 MiB × num_threads peak
  - `bwt_mtf_encode_parts` now returns `Result`; router treats oversized chunks as BWT-not-applicable
- [x] Fix router double-encode determinism (predictor runs twice on winning path)
  - Replaced re-encode sync with `sync_predictor` for LZ77/PredictorRans paths
  - Added matching `sync_predictor` calls in decompressor for all predictor-based paths
  - Eliminates redundant encode pass and cross-block state contamination from trial encodes

### Performance/Robustness Fixes
- [x] Add `ContextMixer` memory warning (~100 MiB per instance, thrashes L3 cache for parallel compression)
  - Added `# Memory` rustdoc section to `ContextMixer` struct and default config
  - Recommends `lightweight()` config or `NeuralSsmPredictor` for parallel workloads
- [x] Verify streaming predictor carry-over between blocks in same group
  - Confirmed `HashMap<u32, Box<dyn ProbabilityPredictor>>` correctly reuses predictors
  - Added `sync_predictor` calls in decompressor to match compressor symmetry
- [x] Improve error messages with byte offsets, block IDs, and group IDs
  - `decompress_block` errors now include block ID, archive offset, group ID, and method
  - `decompress_block_streaming` errors include block ID, group ID, and method
  - BWT/LZ77/LZ4 payload errors include sizes for corruption diagnosis
- [x] Pin `lz4_flex = "=0.11.3"` and add format compatibility guard
  - `lz4_flex`'s `compress_prepend_size` format is not part of the LZ4 frame spec
  - Version upgrade could cause silent decode failures on existing archives
  - Added warning in `lz_preprocess.rs` module docs

### Testing
- [x] Add fuzz targets for `read_metadata_streaming()`, `decode_block()`, block header parsing
  - 4 targets: `fuzz_block_header`, `fuzz_streaming_metadata`, `fuzz_decode_block`, `fuzz_range_coder`
  - Uses `cargo-fuzz` / `libfuzzer-sys`
- [x] Add `#[ignore]` to slow tests (hyperparameter sweep)
  - `sweep_hyperparameters`, `head_to_head_configs`, `sweep_on_silesia`, `head_to_head_on_silesia`
- [ ] Run AddressSanitizer / Miri on unsafe code paths
- [ ] Achieve 80%+ code coverage on core compression/decompression paths

### Code Quality
- [x] Split `decompress.rs` into `decompress_seekable.rs` and `decompress_streaming.rs`
  - Shared types in `decompress.rs`, seekable methods in `decompress_seekable.rs`, streaming in `decompress_streaming.rs`
  - Public API unchanged (`aether_core::pipeline::decompress::Decompressor`)
- [x] Give `RlePredictor` its own `PredictorId` variant (`Rle = 0x0005`)
  - CLI updated with `"rle"` predictor option and auto-detection from archive header
- [x] Make `ContextMixer` and `lz4_flex` paths opt-in via feature flags
  - `context-mixer` feature gates `ContextMixer`, `Lz4AwarePredictor`, and related tests
  - `lz4` feature gates `lz_preprocess` module and `lz4_flex` dependency
  - Both in `default` features — existing builds unchanged
  - `LzPredictorRans` decompression returns clear error when `lz4` disabled (archives remain parseable)
- [x] Add memory backpressure to parallel compression (limit concurrent group size)
  - `Compressor::with_max_threads()` builder method (default: 4 concurrent groups)
  - Bounded `rayon::ThreadPoolBuilder` when `max_threads > 0`; global pool when `0` (unlimited)
  - Peak memory capped at ~`max_threads × (predictor_size + group_data)`

### Security
- [ ] Commission third-party security review of format parser (~$5K)
- [ ] Apply for CII Best Practices Badge
- [ ] Register as CVE Numbering Authority (CNA)

### Release
- [ ] Freeze format as 1.0 with migration strategy
- [ ] Create GitHub release with prebuilt binaries (Windows, Linux, macOS)

---

## Phase 3: Ecosystem (Weeks 9-16) — COMPLETE

Grow adoption and lay groundwork for revenue.

### Bindings
- [x] C FFI crate (`aether-ffi`) with `aether.h` header via cbindgen
  - Lifecycle: `aet_compressor_new()` / `aet_compressor_free()`, `aet_decompressor_new()` / `aet_decompressor_free()`
  - Operations: `aet_compress()`, `aet_extract()`, `aet_verify()`, `aet_version()`
  - Error handling: `aet_last_error()` returns thread-local error strings
  - 10 unit tests covering lifecycle, null safety, and full roundtrip
- [x] Python bindings via PyO3 (`aether-python` crate, excluded from workspace)
  - `aether.compress()` / `aether.extract()` / `aether.verify()` / `aether.list_files()`
  - Password-based encryption support
  - Type stubs for IDE autocompletion
- [x] Wasm target for browser-based decompression (`aether-wasm` crate)
  - Decompress-only API: `verify()`, `list_files()`, `extract_file()`
  - Uses `--no-default-features` (NeuralSsm + RLE + Order0 only)
  - wasm-bindgen bindings for JS interop

### Features (Enterprise)
- [x] Encryption (AES-256-GCM, ChaCha20-Poly1305) — `enterprise` feature
  - Argon2id KDF (64 MiB memory, 3 iterations, 4 lanes)
  - Per-block nonces (master_nonce XOR block_id) for random-access decryption
  - Encrypt-after-compress: preserves block-level random access
  - 57-byte EncryptionHeader after archive header (backward compatible)
  - CLI: `--password` and `--cipher` flags on compress/extract/verify
  - 7 integration tests + 11 unit tests
- [x] Multi-threaded operations (compression & decompression) — `enterprise` feature
  - Two-phase: sequential I/O (read all blocks) → parallel CPU (decompress per-group)
  - One predictor per solid group (correct state evolution)
  - `Decompressor::with_max_threads()`: 0=unlimited, 1=sequential, N=bounded
  - CLI: `--threads` / `-t` flag on extract command
  - 5 integration tests (unlimited, bounded, sequential-vs-parallel, encrypted+parallel)
- [x] Cloud storage backends (S3, GCS, Azure) — `enterprise` feature (via `cloud` flag)
- [ ] Archive splitting and spanning
- [ ] Archive repair/recovery (parity blocks)
- [ ] Solid append (add files without full rewrite)

### Performance
- [x] Replace `cdivsufsort` C binding with `divsufsort` crate, then `libsais` SA-IS (O(n) linear time)
  - Re-evaluated libsais: SA-based BWT (without doubled text) is incompatible with cyclic-rotation
    LF-mapping decoder (different primary index semantics). Direct `libsais_bwt` roundtrip fails.
  - Solution: use `libsais` for O(n) suffix array construction on doubled text T+T, preserving
    cyclic rotation extraction. Faster than divsufsort O(n log n) with identical BWT output.
  - C FFI concern resolved: project already depends on C FFI via `zstd-sys`.
  - Feature-gated behind `bwt-encode` (default on); wasm decompress-only unaffected.
- [x] Criterion benchmarks (`cargo bench -p aether-core`)
  - Roundtrip (compress + decompress) with Order0 and NeuralSSM predictors
  - BWT+MTF+RLE encode/decode throughput
  - Range coder encode/decode throughput
  - Predictor predict+update cycle throughput (Order0, NeuralSSM)
- [x] Profile and optimize hot paths (BWT sort, range coder inner loop)
  - Binary search in range coder decode (256→8 comparisons)
  - `#[inline]` hints on predictor predict/update hot paths
  - Precomputed `a_inv` array for NeuralSSM EMA vectorization
- [x] Early entropy-based BWT skip — skip SA construction for high-entropy chunks (>7.0 bps)
  - Text is typically 4-5 bps; near-random data above 7.0 bps skips expensive SA entirely
  - Tuned from 6.5→7.0 bps: 6.5 caused 0.84% ratio regression on Silesia by skipping BWT-beneficial chunks
- [ ] Target 2+ MiB/s compression, 5+ MiB/s decompression

### Marketing
- [ ] Technical blog post about NeuralSSM predictor design
- [ ] Comprehensive benchmark vs zstd, brotli, lzma, bzip2 on Silesia + Canterbury
- [ ] Submit to Hacker News / r/rust / r/programming

---

## Phase 4: Benchmarks, Examples & Wasm (0.2.2) — COMPLETE

Cross-cutting improvements: examples, external benchmark comparison, documentation refresh, performance optimization, and Wasm target.

### Examples
- [x] `basic_compress.rs` — Compress files with analytics output
- [x] `basic_decompress.rs` — Extract, verify, and byte-compare
- [x] `streaming_extract.rs` — Two-phase streaming decompression

### Benchmarks
- [x] `aet bench --compare` flag for external tool comparison (gzip, bzip2, xz, zstd)
- [x] Updated BENCHMARKS.md with 0.2.1 numbers and external comparisons
- [x] Test suite updated to 179 tests (135 unit + 42 integration + 2 doc)

### Performance
- [x] `#[inline]` hints on predictor hot paths (predict, update, mixing_alpha)
- [x] Precomputed `a_inv` array in NeuralSSM for EMA vectorization
- [x] Range coder binary search already present from 0.1.7

### Wasm
- [x] `aether-wasm` crate with wasm-bindgen (decompress-only API)

---

## Phase 5: Performance Optimization (0.2.3) — COMPLETE

Systematic speed optimization of compression and decompression hot paths.

**Baseline**: 27.37% ratio (2.190 bpb), 0.3 MB/s compress, 0.4 MB/s decompress on 87.1 KiB internal corpus (pre-enlargement).

### Kept
- [x] Direct CDF override in `predict_cdf()` — 2.6-3.4× predictor speedup (CDF built in-place, skips `probs_to_cdf()`)
- [x] Division→multiplication in predictor hot loops — 10% SSM predictor speedup, +20% end-to-end (1.1→1.3 comp, 1.2→1.5 MB/s)
- [x] LTO + `codegen-units=1` — marginal (<5%), kept as release default

### Reverted / Not Feasible
- [x] AVX2/SIMD vectorization — CPU frequency penalty negates gains on mixed scalar/SIMD workload
- [x] `fast_ln()` in SSM mixer — ratio regression 27.37%→27.54% (mixer weighting precision-sensitive)
- [x] 16 order-2 context buckets — ratio regression 27.37%→27.47% (count fragmentation on small corpus)
- [x] Unrolled binary search in `decode_cdf()` — 16% slower (instruction cache pressure)
- [x] SA-based BWT (direct, without doubled text) — confirmed incompatible with cyclic-rotation decoder.
  Re-evaluated in 0.2.4: `libsais` now used for SA construction on doubled text instead (see Phase 3).

### Deferred
- [ ] Cross-block predictor state carry — requires format versioning, 4 decompressor call sites
- [ ] Parallel intra-group block decompression
- [ ] Profile-guided optimization (PGO)

**Final numbers (2.6 MiB internal corpus, post-enlargement)**:
- **Ratio**: 2.75% (0.220 bpb) — synthetic corpus is highly repetitive; Silesia is authoritative
- **Compression**: 1.0 MB/s
- **Decompression**: 2.1 MB/s
- **Silesia (202 MiB)**: 26.45% (2.116 bpb), 0.2/0.3 MB/s

---

## Phase 6: Security Hardening & Open Source Readiness (0.2.3) — IN PROGRESS

Code security audit, documentation refresh, and GitHub infrastructure for public release.

### Security Fixes (Critical)
- [x] Replace weak PRNG (SystemTime + stack address) with OS CSPRNG (`getrandom` crate) for nonce/salt generation
- [x] Fix path traversal in `aether-server` `collect_extracted()` — `strip_prefix` failure now returns error
- [x] Fix unsound `'static` lifetime in FFI (`cstr_to_path`) — replaced with owned `PathBuf`/`String`
- [x] Fix FFI `aether_list()` memory leak on `CString::new()` failure — now cleans up prior allocations
- [x] Fix `try_into().unwrap()` on untrusted input in 20+ locations (rans, bwt, lz77, router, neural_ssm, rle_predictor, order0)
- [x] Fix CDF underflow in range coder backward pass — `saturating_sub(1)` prevents u16 wraparound
- [x] Add API key authentication to `aether-server` (`AETHER_API_KEY` env var, Bearer token)

### Code Quality
- [x] Replace heap-allocated CRC buffers with stack arrays in header/block/footer
- [x] Fix `.ok().flatten()` multipart error swallowing in server endpoints
- [x] Safe `read_u32`/`read_f32` helpers for `load_state()` in NeuralSSM, RLE, Order0 predictors
- [ ] Run `RUSTDOCFLAGS="-D warnings" cargo doc` and fix all warnings
- [ ] Run AddressSanitizer / Miri on unsafe code paths
- [ ] Achieve 80%+ code coverage on core compression/decompression paths

### Documentation
- [x] Update SECURITY.md with encryption side-channel considerations and server auth note
- [x] Update CONTRIBUTING.md with DCO policy, fuzz testing instructions, unsafe code policy
- [x] Update ARCHITECTURE.md with EncryptionHeader (57B) and dictionary hash format
- [x] Update PREDICTORS.md benchmarks from v0.1.2 to v0.2.2
- [ ] Update PRESENTATION.md performance numbers to match v0.2.2

### GitHub Infrastructure
- [x] Fix repository URL to `craton-co/aether-arch` across all Cargo.toml files
- [x] Add `docs.rs` metadata to all crate Cargo.toml files
- [x] Add GitHub issue templates (bug report, feature request)
- [x] Add pull request template
- [x] Add `.github/dependabot.yml` (Cargo + GitHub Actions, `lz4_flex` pinned)
- [x] Add release workflow (cross-compile Linux/macOS/Windows, GitHub Releases)
- [ ] Publish to crates.io (`aether-core`, `aether-cli`)
- [ ] Set up GitHub Pages for rustdoc hosting

### Certifications
- [ ] Commission third-party security review of format parser
- [ ] Apply for OpenSSF Scorecard (automated)
- [ ] Apply for CII Best Practices Badge

---

## Research Directions

Ideas and experiments not yet committed to the roadmap. See git history for previously completed items.

### Predictor Selection per Solid Group
Allow per-group predictor selection based on content type. Text groups → BWT + NeuralSSM, binary structured → LZ77 + ContextMixer, images/random → Zstd. Store per-group predictor choice in `SolidGroupEntry`.

### NeuralSsmPredictor: Larger SSM with Feature Hashing
Expand order-2 to order-3/4 literal context, use larger hash tables (256 or 1024 contexts instead of 8), add SSM hidden state features to literal prediction.

### Improved LZ77 (Optimal Parsing)
Implement optimal parsing (Storer-Szymanski / price-based), extend window to 256 KiB+, add secondary hash for longer matches.

### Progress Bars and Better CLI UX
Implement `indicatif` progress bars, per-file compression ratios, `--quiet`/`--verbose` flags, color output.

### Pre-trained Neural Model (V2)
Train a small Mamba/S4-style SSM (~200K params) on a large text corpus, export to safetensors, use `candle-core`/`candle-nn` for pure-Rust inference. INT8 quantization for deterministic cross-platform behavior. Could reach 1.8-2.0 bpb on text.

### Speculative Decoding
Use a fast "draft" model to speculatively predict N bytes ahead, verify with full neural model. 2-4x throughput improvement if draft agreement exceeds ~75%.

### Delta and Diff Compression
Detect similar files via BLAKE3 or rolling hash, delta-encode against each other before compression. Useful for version control archives and backup sets.

### Memory-Mapped I/O
Use memory-mapped files for input, process chunks lazily without loading entire files.

### Known Limitations
- **Small files**: Archive format overhead (~468 bytes) significant for files under 10 KiB
- **Speed**: ~1 MiB/s compression — suitable for archival, not real-time
- **Floating-point determinism**: f32 arithmetic may differ across CPU architectures (x86 vs ARM FMA)
- **No symlink support**: Symbolic links are followed, not preserved
- **No async I/O**: All I/O is synchronous
- **32-bit block sizes**: Individual blocks limited to 4 GiB

---

## Dependencies

| Dependency | Blocks | Status |
|-----------|--------|--------|
| Git repository initialized | Phase 1 CI/CD | **Done** |
| Format freeze | Phase 2 crates.io publish | Not started |
| Encryption implementation | Phase 3 enterprise, FIPS cert | **Done** |
| Pure-Rust SA construction | Eliminating C FFI risk | **Done** (`divsufsort` crate) |

---

Copyright 2024-2026 Craton Software Company Licensed under Apache-2.0.
