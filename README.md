# AetherArch (.aet)

[![CI](https://github.com/craton-co/aether-arch/actions/workflows/ci.yml/badge.svg)](https://github.com/craton-co/aether-arch/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![MSRV: 1.85.0](https://img.shields.io/badge/MSRV-1.85.0-brightgreen.svg)](https://blog.rust-lang.org/)

A next-generation file archiver built in Rust that combines neural-probabilistic prediction with custom range coding, content-defined chunking, semantic solid archiving, and adaptive routing.

**Status**: 0.2.6 — 293 tests (128 unit + 87 integration + 28 FFI + 41 server + 1 doc), 6 crates, encryption, streaming, dictionary pretraining, REST API, Wasm target

```
Tool             Ratio     Speed (comp)    Corpus
────────────────────────────────────────────────────────────────
AetherArch      21.25%    0.13 MB/s        Webster 40 MiB (dictionary text)
xz -9           20.00%    0.79 MB/s        Webster 40 MiB  (best ratio)
brotli -q11     20.25%    0.19 MB/s        Webster 40 MiB
zstd -19        20.75%    1.36 MB/s        Webster 40 MiB
bzip2 -9        20.75%    7.53 MB/s        Webster 40 MiB
gzip -9         30.00%    12.3 MB/s        Webster 40 MiB
```

AetherArch ranks **2nd on the Webster dictionary benchmark** (only 1.25% behind xz best-in-class),
beating brotli, zstd, bzip2, and gzip on compression ratio. See [docs/v2.6/BENCHMARK_V2_6.md](docs/v2.6/BENCHMARK_V2_6.md) for full Silesia results.

## How It Works

AetherArch replaces the fixed Huffman/LZ77 model of gzip with a multi-stage adaptive pipeline:

```
Input files
  │
  ▼
Content-Defined Chunking (FastCDC, 16-512-4096 KiB)
  │
  ▼
Entropy Analysis + Content-Type Detection
  │
  ▼
Semantic Solid Grouping (by file type)
  │
  ▼
Adaptive Routing (per-chunk, picks smallest):
  ├─ BWT + MTF + RLE → Neural SSM predictor + Range coding
  ├─ LZ77 (min-match-3, 64KB window) → Predictor + Range coding
  ├─ Plain predictor + Range coding
  ├─ Zstd fallback (level 3)
  └─ Store (uncompressed)
  │
  ▼
Archive assembly with BLAKE3 integrity checksums
```

The BWT+MTF+RLE path is the primary compression path for structured data. A Burrows-Wheeler Transform clusters similar contexts, Move-to-Front encoding converts to small integers, and bijective RUNA/RUNB run-length encoding compacts zero runs. The resulting stream is then modeled by a Neural SSM predictor — a diagonal state-space model with online-learning sigmoid classifiers — and compressed via a custom byte-aligned range coder with 15-bit CDF precision.

The Neural SSM predictor uses D=32 exponential moving averages as hidden state, two online sigmoid classifiers (SGD lr=0.01), and blends an order-2 literal context at weight 0.30. These hyperparameters were tuned by greedy sweep on the Silesia corpus.

## Quick Start

### Build

```bash
cargo build --release
```

The binary is at `target/release/aet` (or `aet.exe` on Windows).

### Compress

```bash
aet compress mydir/ -o archive.aet
aet compress file1.txt file2.rs -o archive.aet
aet compress mydir/ -o archive.aet --predictor cm    # explicit predictor
aet compress mydir/ -o archive.aet --analytics        # show compression stats
```

### Extract

```bash
aet extract archive.aet -o output_dir/
aet extract archive.aet -f path/to/file.txt -o .     # single file
aet extract archive.aet -o output_dir/ --threads 4    # parallel decompression (auto-scales by default)
cat archive.aet | aet extract - -o output_dir/        # streaming from stdin
```

### Encryption

```bash
aet compress mydir/ -o archive.aet --password secret
aet compress mydir/ -o archive.aet --password secret --cipher chacha20
aet extract archive.aet -o output_dir/ --password secret
```

Supported ciphers: `aes256gcm` (default), `chacha20` (ChaCha20-Poly1305). Key derivation: Argon2id.

### Dictionary Pretraining

```bash
aet train --output domain.aed training_data/    # train a dictionary
aet compress mydir/ -o archive.aet --dictionary domain.aed
aet extract archive.aet -o output_dir/ --dictionary domain.aed
```

### Archive Migration

```bash
aet migrate old.aet -o new.aet --predictor ssm         # change predictor
aet migrate old.aet -o new.aet --dictionary domain.aed  # add dictionary
```

### List Contents

```bash
aet list archive.aet
aet list archive.aet --long    # detailed: sizes, groups, BLAKE3 hashes
cat archive.aet | aet list -   # streaming from stdin
```

### Verify Integrity

```bash
aet verify archive.aet
cat archive.aet | aet verify - # streaming from stdin
```

### Benchmark

```bash
aet bench mydir/ -P order0,cm,cm-light,lz4-aware,ssm
aet bench mydir/ --compare                             # compare with gzip, bzip2, xz, zstd
```

## Library Usage

```rust
use aether_core::pipeline::compress::Compressor;
use aether_core::pipeline::decompress::Decompressor;

// Compress
let stats = Compressor::new()
    .compress_to_archive(&["mydir/"], "archive.aet")?;

// Extract
Decompressor::new()
    .extract_all("archive.aet", "output/")?;
```

See [`examples/`](aether-core/examples/) for streaming, dictionary, and analytics usage.

## Available Predictors

| Name | CLI Flag | Description | Speed | Memory |
|------|----------|-------------|-------|--------|
| Order-0 | `order0` | Byte frequency counting with Laplace smoothing | Fast | 1 KiB |
| Context Mixer | `cm` | PAQ-inspired multi-order (1-8) logistic mixing | Very slow | ~100 MiB |
| Context Mixer Light | `cm-light` | Orders 1-6, smaller tables | Slow | ~25 MiB |
| LZ4-Aware | `lz4-aware` | FSM tracking LZ4 token structure | Medium | ~8 MiB |
| Neural SSM | `ssm` | Diagonal SSM + RLE predictor + order-2 context | Medium | ~25 KiB |
| RLE | `rle` | Hierarchical 3-context RLE stream predictor | Fast | ~3 KiB |

The predictor choice is stored in the archive header and auto-detected during extraction. All predictors produce identical output on the current test corpus because the BWT+RLE path always wins and uses its own internal Neural SSM predictor.

## Project Structure

```
aether/
├── Cargo.toml                         # Workspace root
├── aether-core/                       # Library crate (all compression logic)
│   ├── src/
│   │   ├── lib.rs                     # Public API re-exports
│   │   ├── error.rs                   # Error types (AetherError)
│   │   ├── format.rs                  # Constants, enums, shannon_entropy()
│   │   ├── header.rs                  # Archive/file/group/footer structs
│   │   ├── block.rs                   # Block header/trailer/index
│   │   ├── chunker.rs                 # FastCDC content-defined chunking
│   │   ├── analyzer.rs               # Entropy analysis, content detection
│   │   ├── grouper.rs                # Semantic solid grouping
│   │   ├── dictionary.rs             # Dictionary training/saving/loading (.aed)
│   │   ├── crypto/                    # Encryption (AES-256-GCM, ChaCha20-Poly1305)
│   │   ├── cloud/                     # Cloud storage backends (S3, GCS, Azure)
│   │   ├── entropy/                   # Probability predictors
│   │   │   ├── traits.rs             # ProbabilityPredictor trait
│   │   │   ├── order0.rs             # Baseline frequency model
│   │   │   ├── context_mixer.rs      # Multi-order context mixing
│   │   │   ├── lz4_aware.rs          # FSM-based LZ4 stream predictor
│   │   │   ├── rle_predictor.rs      # Hierarchical RLE stream predictor
│   │   │   ├── mtf_predictor.rs      # MTF-aware predictor (legacy)
│   │   │   └── neural_ssm.rs         # Neural SSM + RLE hybrid (best)
│   │   ├── coding/                    # Entropy coding + preprocessing
│   │   │   ├── rans.rs               # Custom byte-aligned range coding
│   │   │   ├── zstd_fallback.rs      # Zstd passthrough
│   │   │   ├── bwt_preprocess.rs     # BWT + MTF + RUNA/RUNB RLE
│   │   │   ├── lz77_preprocess.rs    # Custom LZ77 (min-match-3)
│   │   │   └── lz_preprocess.rs      # LZ4 via lz4_flex
│   │   └── pipeline/                  # Orchestration
│   │       ├── router.rs             # Adaptive chunk routing
│   │       ├── compress.rs           # Compression pipeline
│   │       ├── decompress.rs         # Shared decompression types
│   │       ├── decompress_seekable.rs # Seekable decompression (Seek+Read)
│   │       ├── decompress_streaming.rs # Streaming decompression (Read-only)
│   │       ├── analytics.rs          # Compression analytics
│   │       └── migrate.rs            # Archive migration tool
│   ├── benches/
│   │   └── compression.rs            # Criterion benchmarks
│   ├── examples/                      # Usage examples
│   │   ├── basic_compress.rs
│   │   ├── basic_decompress.rs
│   │   └── streaming_extract.rs
│   ├── fuzz/                          # Fuzz targets (cargo-fuzz)
│   └── tests/
│       └── integration.rs            # 42 integration tests
├── aether-cli/                        # Binary crate (CLI tool)
│   └── src/main.rs                   # clap-based CLI
├── aether-ffi/                        # C FFI crate (cbindgen header)
│   └── src/lib.rs                    # aet_compress/extract/verify C API
├── aether-server/                     # REST API server (axum)
│   └── src/main.rs                   # /compress, /extract, /verify, /list
├── aether-wasm/                       # WebAssembly bindings (decompress-only)
│   └── src/lib.rs                    # verify, list_files, extract_file
├── aether-python/                     # Python bindings via PyO3 (excluded from workspace)
└── tests/fixtures/                    # Test corpus
    ├── sample/                        # Small files (~2.6 KiB)
    └── large/                         # Benchmark corpus (~87 KiB)
```

293 tests (128 unit + 87 integration + 28 FFI + 41 server + 1 doc, 5 ignored).

## .aet Archive Format

Binary, little-endian. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for full specification.

```
┌─ Archive Header (48 bytes) ─────────────────┐
│  Magic, flags, predictor ID, counts, offsets │
├─ Encryption Header (57 bytes, optional) ────┤
│  Cipher, salt, KDF params, nonce             │
├─ Dictionary Hash (32 bytes, optional) ──────┤
│  BLAKE3 hash of pretrained dictionary state  │
├─ File Table (variable) ─────────────────────┤
│  Per file: path, size, BLAKE3, group, perms  │
├─ Solid Group Table (24 bytes each) ──────────┤
│  Group ID, content type, method, block range │
├─ Compressed Blocks (variable) ──────────────┤
│  Each: header(28B) + payload + trailer(36B)  │
├─ Block Index (24 bytes each) ────────────────┤
│  Block offsets for random-access seeking     │
├─ Archive Footer (32 bytes) ──────────────────┤
│  Redundant offsets + counts + magic          │
└──────────────────────────────────────────────┘
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `fastcdc` | 3 | Content-defined chunking (v2020 algorithm) |
| `zstd` | 0.13 | Fallback compression for high-entropy data |
| `blake3` | 1 | BLAKE3 integrity checksums |
| `lz4_flex` | =0.11.3 | LZ4 compression (pure Rust, pinned for format stability) |
| `divsufsort` | 2 | Pure-Rust suffix array for BWT |
| `byteorder` | 1 | Little-endian binary serialization |
| `thiserror` | 2 | Error type derivation |
| `clap` | 4 | CLI argument parsing |
| `rayon` | 1 | Parallel inter-group compression |
| `crc32fast` | 1 | CRC32 checksums for headers |
| `serde` | 1 | Serialization support |
| `tracing` | 0.1 | Structured logging |

## Testing

```bash
# Run all tests
cargo test --workspace --release

# Run with output visible
cargo test --workspace --release -- --nocapture

# Run specific test
cargo test -p aether-core --release -- neural_ssm::tests::head_to_head_configs --nocapture

# Run benchmarks (Criterion)
cargo bench -p aether-core

# Run CLI benchmarks
cargo run --release -p aether-cli -- bench tests/fixtures/large/ -P order0,cm,ssm

# Run fuzz targets
cargo +nightly fuzz run fuzz_decode_block -p aether-core
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Deep technical design, format specification, predictor internals
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — Compression results, test suite output, performance data
- [docs/ROADMAP.md](docs/ROADMAP.md) — Production readiness plan, research directions, open-source vs enterprise split
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [CONTRIBUTING.md](CONTRIBUTING.md) — Development guidelines
- [SECURITY.md](SECURITY.md) — Vulnerability reporting and security policy
- [docs/PREDICTORS.md](docs/PREDICTORS.md) — Detailed predictor and compression method reference
- [docs/PRESENTATION.md](docs/PRESENTATION.md) — Project overview slides

### V2.6 Release Docs

- [docs/v2.6/BENCHMARK_V2_6.md](docs/v2.6/BENCHMARK_V2_6.md) — Full Silesia benchmark vs gzip, brotli, xz, zstd, bzip2, lz4
- [docs/v2.6/IMPLEMENTATION_REPORT_V2_6.md](docs/v2.6/IMPLEMENTATION_REPORT_V2_6.md) — All 5 optimizations explained
- [docs/v2.6/THREADING_DESIGN.md](docs/v2.6/THREADING_DESIGN.md) — Dynamic threading architecture and scaling analysis
- [docs/v2.6/STATUS_REPORT.md](docs/v2.6/STATUS_REPORT.md) — Deployment readiness checklist
- [docs/v2.6/V2_6_QUICK_REFERENCE.md](docs/v2.6/V2_6_QUICK_REFERENCE.md) — Quick command reference

## Installation

```bash
# From source
cargo install --path aether-cli

# The binary is named `aet`
aet --help
```

Minimum supported Rust version: **1.85.0**

## What's Missing

AetherArch is pre-1.0 software. Key gaps before production use:

- **Speed**: ~2.0 MiB/s on text/code (internal corpus, V2.6); ~0.1 MiB/s on large Silesia files. Suitable for archival, not real-time compression.
- **Format not frozen**: The `.aet` binary format may change in future versions. No migration guarantee yet.
- **No symlink support**: Symbolic links are followed and stored as regular files.
- **No archive append**: Adding files requires a full rewrite of the archive.
- **Cloud backends are stubs**: S3/GCS/Azure adapters define the trait but have no SDK integration.
- **Floating-point determinism**: f32 arithmetic may differ across CPU architectures (x86 vs ARM FMA), which could affect cross-platform archive portability.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full production readiness plan.

## License

AetherArch is licensed under the [Apache License, Version 2.0](LICENSE).

Copyright 2024-2026 Craton Software Company

Enterprise features (encryption, multi-threaded decompression, cloud storage backends)
are included in the source under the same Apache 2.0 license. Organizations using
enterprise features in production may purchase a commercial support license — contact
us at [legal@craton.com.ar](mailto:legal@craton.com.ar) for details.
