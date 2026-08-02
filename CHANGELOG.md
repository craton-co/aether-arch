# Changelog

All notable changes to AetherArch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-30

### Added

- Archival, balanced, and fast per-group compression profiles.
- Ratio-safe x86/x86-64 BCJ preprocessing for ELF, PE, and Mach-O executables.
- Prefix-compressed file-table paths, enabled only when the table shrinks.
- A text/log/binary/image/tiny-file benchmark matrix runner.

### Changed

- Compression and transformed training use zero-copy borrowed chunk views.
- NeuralSSM uses an encode-only CDF interval query, monotone quantization, and
  an interpolated sigmoid lookup table.
- Range decoding accelerates common RUNA/RUNB symbols before binary search.
- BWT entropy routing and transformed training share the 7.0 bps threshold.
- Threaded builds parallelize independent chunks within solid groups while
  retaining deterministic output order.

## [0.3.0-rc1] - TBD

### Planned
- Archive splitting and spanning: multi-part archives across multiple files/disks
- Archive repair: parity block generation and recovery tool
- Format Freeze: finalize `.aet` binary format specification for 1.0
- Third-party security audit and FIPS 140-2 evaluation

## [0.2.3] - 2026-03-18

### Added
- **Direct CDF override**: `predict_cdf()` overrides on `NeuralSsmPredictor` and `Order0Model` build CDF tables in-place without `probs_to_cdf()` conversion — 2.6-3.4× predictor speedup
- **Division→multiplication optimization**: Precomputed reciprocals in `RlePredictor::predict()` and `NeuralSsmPredictor::predict()/predict_cdf()` — replaces 254 divisions per call with 1 division + 254 multiplies; 10% SSM predictor speedup, +20% end-to-end throughput
- **LTO + codegen-units=1**: Release profile optimizations for marginal (<5%) gains
- **CLA Assistant**: GitHub Actions workflow for contributor license agreement enforcement
- **Dependabot action bumps**: `actions/checkout` v6, `actions/upload-artifact` v7, `actions/download-artifact` v8, `dtolnay/rust-toolchain` 1.100.0
- **Enlarged internal corpus**: Test fixtures expanded from 87.1 KiB to 2.6 MiB (english.txt 2 MiB, source.rs 315 KiB, mixed.json 299 KiB) for more representative speed benchmarks

### Fixed
- **Content-type detection**: Fixed type mismatch in `analyzer.rs` text detection (`contains` call), improving routing decisions on Silesia corpus

### Changed
- **Silesia ratio**: 26.45% (was 29.12% in 0.2.2) — improved content-type detection leads to better routing
- **Silesia speed**: 0.2 MB/s compress, 0.3 MB/s decompress (889s on 202 MiB)
- **Internal corpus**: 2.75% ratio (0.220 bpb) on enlarged 2.6 MiB corpus; 1.0 MB/s compress, 2.1 MB/s decompress

### Investigated & Reverted
- AVX2/SIMD vectorization on SSM hot loops — CPU frequency penalty negates gains
- `fast_ln()` polynomial approximation in SSM mixer — ratio regression (+0.17%)
- 16 order-2 context buckets (was 8) — count fragmentation on small corpus (+0.10%)
- Unrolled binary search in `decode_cdf()` — 16% slower due to instruction cache pressure
- SA-based BWT via SA-IS — mathematically incorrect for cyclic rotations

## [0.2.2] - 2026-02-26

### Added
- **Enhanced enterprise gating**: `threading` and `cloud` modules now gated behind the `enterprise` feature flag.
- **Examples directory**: `basic_compress.rs`, `basic_decompress.rs`, `streaming_extract.rs` in `aether-core/examples/`
- **Benchmark comparison**: `aet bench --compare` flag for external tool comparison (gzip, bzip2, xz, zstd)
- **Wasm crate** (`aether-wasm`): wasm-bindgen decompress-only API (`verify`, `list_files`, `extract_file`)
- **Performance optimization**: `#[inline]` on predictor hot paths, precomputed `a_inv` in NeuralSSM EMA loop

### Changed
- **BENCHMARKS.md**: Updated with 0.2.1 numbers, external tool comparison tables, 179 tests

## [0.2.1] - 2026-02-13

### Added
- **Compression analytics**: `CompressionAnalytics` struct with per-method block counts, byte distributions, group stats, timing; CLI `--analytics` flag
- **Dictionary pretraining**: `Dictionary` module for training/saving/loading `.aed` files; `save_state()`/`load_state()` on `ProbabilityPredictor` trait; implemented for Order0, RLE, NeuralSSM; `FLAG_HAS_DICTIONARY` header flag; `Compressor/Decompressor::with_dictionary()`; CLI `aet train` command and `--dictionary` flag on compress/extract
- **Archive migration tool**: `Migrator` struct in `pipeline::migrate` for decompress→recompress with new settings (predictor change, dictionary, encryption); CLI `aet migrate` command
- **REST API server** (`aether-server` crate): `axum`-based HTTP server with `/compress`, `/extract`, `/verify`, `/list`, `/health`, `/version` endpoints; multipart upload; configurable port and max upload size
- **Cloud storage adapters**: `StorageBackend` trait with `CloudReader` (Read+Seek via range requests); S3, GCS, Azure Blob stub implementations; URL parser for `s3://`, `gs://`, `az://` schemes; 5 unit tests
- **C FFI crate** (`aether-ffi`): `aether.h` header via cbindgen, lifecycle/compression/decompression/error APIs, 10 unit tests
- **Python bindings** (`aether-python`): PyO3 module with `compress()`, `extract()`, `verify()`, `list_files()`, encryption support, type stubs
- **Encryption** (enterprise feature): AES-256-GCM and ChaCha20-Poly1305 with Argon2id KDF, per-block nonces, 57-byte EncryptionHeader, CLI `--password`/`--cipher` flags, 18 tests
- **Multi-threaded decompression** (enterprise feature): two-phase (sequential I/O → parallel CPU) across solid groups, `Decompressor::with_max_threads()`, CLI `--threads` flag, 5 integration tests
- **Pure-Rust suffix array**: replaced `cdivsufsort` C binding with `divsufsort` crate (eliminates unsafe FFI)
- **Criterion benchmarks**: roundtrip, BWT, range coder, and predictor benchmarks (`cargo bench -p aether-core`)
- ROADMAP.md with phased production readiness plan
- CHANGELOG.md (this file)
- CONTRIBUTING.md with development guidelines
- SECURITY.md with responsible disclosure policy
- Apache-2.0 license
- GitHub Actions CI pipeline (test, clippy, fmt, audit)
- `deny.toml` for cargo-deny license and CVE auditing
- `#[non_exhaustive]` on all public enums for forward compatibility
- Bounds checking on archive-supplied sizes before memory allocation
- `MAX_DECOMPRESSED_BLOCK_SIZE` (64 MiB) safety cap
- `MAX_FILE_COUNT` and `MAX_BLOCK_COUNT` sanity limits
- Named constant `BWT_DECISIVE_RATIO` (was magic 0.55)
- MSRV policy: Rust 1.75.0
- `enterprise` feature flag for future gated features
- Comprehensive `///` rustdoc on all public API items
- `MAX_BWT_INPUT_SIZE` (8 MiB) guard to prevent OOM from BWT doubled-text allocation
- Bounds checks on `encoded_len` / `lz_len` in decompressor (crafted archive defense)
- `ContextMixer` memory warning rustdoc for parallel compression workloads
- Format stability warning in `lz_preprocess.rs` module docs
- Fuzz targets: `fuzz_block_header`, `fuzz_streaming_metadata`, `fuzz_decode_block`, `fuzz_range_coder`
- `PredictorId::Rle` variant (0x0005) — `RlePredictor` now has its own predictor ID
- `context-mixer` feature flag: gates `ContextMixer`, `Lz4AwarePredictor`, and related tests
- `lz4` feature flag: gates `lz_preprocess` module and `lz4_flex` dependency
- Memory backpressure: `Compressor::with_max_threads()` limits concurrent group compression (default 4)
- `#[ignore]` on slow hyperparameter sweep tests

### Changed
- `MAX_DECODE_SIZE` in range coder now aligned with `MAX_DECOMPRESSED_BLOCK_SIZE` (was hardcoded 16 MiB)
- Router sync section: replaced redundant `encode_block` re-encode with `sync_predictor`
  for LZ77 and PredictorRans paths (eliminates double-encode, fixes cross-block state contamination)
- Decompressor: added `sync_predictor` calls after LZ77/LZ4/PredictorRans decode
  to maintain symmetric cross-block state with compressor
- Error messages in `decompress_block` / `decompress_block_streaming` now include
  block ID, archive offset, group ID, and compression method
- `bwt_mtf_encode_parts` / `bwt_mtf_encode` now return `Result` (was infallible)
- `lz4_flex` pinned to `=0.11.3` (format stability)
- Split `decompress.rs` into `decompress.rs` (shared types), `decompress_seekable.rs`,
  and `decompress_streaming.rs` — public API unchanged
- `lz4_flex` is now an optional dependency (enabled via `lz4` feature, in `default`)
- Parallel compression uses bounded `rayon::ThreadPoolBuilder` instead of global pool

## [0.1.8] — 2026-01-20

### Added
- Streaming decompression (`Read`-only path, no `Seek` required)
  - `read_metadata_streaming()` for sequential metadata parsing
  - `extract_all_streaming()` / `extract_with_streaming_metadata()` two-phase API
  - `verify_streaming()` / `list_files_streaming()` full streaming support
  - Per-group predictors via `HashMap<u32, Box<dyn ProbabilityPredictor>>`
  - CLI: `"-"` stdin sentinel for Extract, List, Verify commands
- Skip `sync_predictor` during decompression optimization
  - `predictor_state_flag` in BlockHeader (byte 13)
  - `CompressedChunk.predictor_synced` tracking during compression
  - Backward compatible: old archives always sync (same as 0.1.7)
- 9 new integration tests: streaming roundtrip (4), streaming verify (2),
  streaming list (1), metadata detection (1), two-phase extraction (1)

## [0.1.7] — 2025-11-15

### Changed
- Custom range coder replacing `constriction` crate (LZMA-style carry-propagating
  encoder + subtraction-based decoder)
- `predict_cdf()` method on `ProbabilityPredictor` trait (returns `[u16; 257]` CDF)
- Chunk sizes: MIN 4 KiB → 16 KiB, AVG 64 KiB → 512 KiB, MAX 512 KiB → 4096 KiB
- RLE decoder hardening: saturating arithmetic, MAX_DECODE_SIZE=16 MiB guard
- Extract command now shows MiB/s speed
- `wrapping_mul` in LCG test generators, mid-payload corruption offsets

## [0.1.6] — 2025-09-22

### Added
- NeuralSsmPredictor: diagonal SSM + RLE baseline + order-2 context
- Adaptive mixer with EMA log-likelihood sensitivity
- Silesia-tuned hyperparameters (D=32, lr=0.01, o2_blend=0.30)

## [0.1.5] — 2025-07-28

### Added
- BWT + MTF + RUNA/RUNB RLE preprocessing pipeline
- Custom LZ77 encoder (min-match-3, lazy matching, 64KB window)
- Adaptive routing cascade: BWT → LZ77 → plain RC → Zstd → Store
- RlePredictor: hierarchical 3-context predictor for RLE streams
- Semantic solid grouping by content type
- Parallel group compression via rayon

## [0.1.0] — 2025-05-16

### Added
- Initial AetherArch implementation
- `.aet` binary archive format
- Order0Model, ContextMixer, Lz4AwarePredictor
- FastCDC content-defined chunking
- BLAKE3 integrity checksums
- CRC32 header/trailer verification
- CLI tool (`aet`) with compress, extract, list, verify, bench commands

---

Copyright 2024-2026 Craton Software Company Licensed under Apache-2.0.
