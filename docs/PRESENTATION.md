# AetherArch — Project Presentation

> Historical 0.2.3 presentation. For the current 0.3.0 architecture and
> profiles, see `README.md`, `docs/ARCHITECTURE.md`, and `BENCHMARKS.md`.

---

## Slide 1 · What Is AetherArch?

> A next-generation file archiver built in Rust that replaces static entropy coding with an online neural-probabilistic model.

- **Format**: `.aet` — custom binary archive with BLAKE3 integrity
- **Language**: Rust (safe, no unsafe blocks in core library)
- **Codebase**: 6 crates (core, cli, ffi, server, wasm, python), 285 tests
- **Status**: 0.2.3 — dictionary pretraining, REST API, cloud backends, Wasm target, streaming decompression, performance optimizations

---

## Slide 2 · The Problem with Existing Archivers

Standard tools use **fixed pipelines** designed in the 1990s:

| Tool | Algorithm | Limitation |
|------|-----------|------------|
| gzip | LZ77 + Huffman | Fixed 32 KB window, static coding |
| bzip2 | BWT + Huffman | Good transform, weak entropy coder |
| xz | LZMA2 | Great ratio, very slow compression |
| zstd | LZ77 + Huffman/ANS | Speed-focused, less ratio on text |
| brotli | LZ77 + context modeling | Large dictionary, slow at max level |
| lz4 | LZ77 (minimal) | Extreme speed, poor ratio |

**AetherArch's approach**: separate *modeling* from *coding*.
Any predictor can be plugged in. A custom byte-aligned range coder adapts to whatever probabilities it receives.

---

## Slide 3 · Architecture at a Glance

```
Input files
  │
  ├─ Content-type detection  (ELF, PE, JPEG, ZIP, text…)
  ├─ Semantic solid grouping (files of same type share a predictor)
  ├─ Content-defined chunking (FastCDC  16–512–4096 KiB)
  │
  ├─ Groups compressed in PARALLEL via rayon ──────────────────────┐
  │      (each group: independent predictor, concurrent threads)   │
  │                                                                 │
  └─ Per-chunk adaptive routing (picks smallest):                  │
       ┌──────────────────────────────────────────────┐            │
       │  BWT + MTF + RLE → Neural SSM → Range coding │  ← main   │
       │  LZ77 (min-match-3, 64 KB) → Range coding    │  [skip if │
       │  Plain predictor → Range coding               │   BWT<55%]│
       │  Zstd level 3  (high-entropy fallback)        │            │
       │  Store          (incompressible data)         │            │
       └──────────────────────────────────────────────┘            │
  ◄───────────────────────────────────────────────────────────────┘
  Sequential write (deterministic archive layout)
```

**Key insight**: the BWT transform clusters identical contexts — a perfect setup for an online neural predictor.

---

## Slide 4 · The Compression Stack

Each layer adds a measurable gain (87.1 KiB internal corpus):

```
                     bpb     ratio    gain vs previous
                   ───────  ───────  ─────────────────
Raw Order-0         4.769   59.6%    —  (baseline)
+ ContextMixer      4.195   52.4%    -12%
+ LZ4 preprocess    3.218   40.2%    -23%
+ LZ77 preprocess   2.671   33.4%    -17%
+ BWT transform     2.414   30.2%    -10%
+ RUNA/RUNB RLE     2.223   27.8%     -8%   ← beat gzip-9
+ RlePredictor      2.202   27.5%     -1%
+ NeuralSsmPredict  2.186   27.3%     -1%
```

---

## Slide 5 · Results — Benchmarks

### Silesia Corpus (202 MiB, 12 files) — Industry Standard

```
                      AetherArch   gzip -9   bzip2 -9
  ────────────────────────────────────────────────────
  Overall ratio          26.45%    31.91%     25.72%
  ────────────────────────────────────────────────────
  vs gzip-9         17.1% smaller  ——
  vs bzip2-9                       ——      2.8% larger
```

**17.1% smaller than gzip-9**, closing in on bzip2-9 (only 2.8% gap).

Silesia compression speed: ~0.2 MiB/s (202 MiB of diverse data with large BWT blocks).

### Internal Corpus (87.1 KiB) — `aet bench --compare`

```
Tool              Comp MB/s         Size      Ratio  Bits/byte
--------------------------------------------------------------
brotli -q 11           0.6     21.7 KiB     24.89%      1.991
xz -9                  1.9     23.6 KiB     27.12%      2.169
AetherArch (ssm)       1.3     23.8 KiB     27.37%      2.190
bzip2 -9               5.8     23.9 KiB     27.40%      2.192
zstd -19               0.9     24.2 KiB     27.78%      2.222
gzip -9                3.4     25.4 KiB     29.14%      2.331
lz4 -9                 2.1     32.2 KiB     36.92%      2.953
```

AetherArch beats **zstd -19** and gzip -9 on ratio. brotli -q 11 leads on this text-heavy corpus (large context model). lz4 trades ratio for decompression speed.

**Speed note**: Internal corpus speed (~1.3 MiB/s) is higher than Silesia (~0.2 MiB/s) because small files fit in cache and have lower BWT sort overhead per byte.

---

## Slide 6 · The Neural SSM Predictor

The distinguishing architectural choice: a **diagonal linear State Space Model** fused with a hierarchical RLE predictor.

```
Input byte
    │
    ├─ Embedding: byte → 32-dimensional vector
    │
    ├─ SSM update: h[d] = α[d]·h[d] + (1−α[d])·embed[byte][d]
    │  (32 exponential moving averages at different timescales: 0.5 … 0.999)
    │
    ├─ Binary classifier 1: sigmoid(w_run · h)  → P(run symbol)
    ├─ Binary classifier 2: sigmoid(w_runa · h) → P(RUNA vs RUNB)
    │
    ├─ RlePredictor baseline (3-context hierarchical model)
    │
    ├─ Adaptive mixer: weight SSM vs RLE by recent log-likelihood
    │  → SSM only contributes when it outperforms pure RLE
    │
    └─ Order-2 literal context blend (30% weight, 8 hash buckets)
```

**66 learnable parameters. ~25 KiB memory. Adapts from scratch per block.**
Retuned via greedy sweep on Silesia (8.8 MiB RLE stream): D=32, lr=0.01, o2=0.30.
Improvement: 3.4265 bpb → 3.4121 bpb on Silesia (−0.0144 bpb vs old defaults).

The interesting result is that **66 online parameters** — adapting from scratch per block with no pre-training — produce any measurable improvement at all over the pure hierarchical RLE baseline. The SSM's multi-timescale EMA state captures run-length patterns that a fixed prior cannot represent.

---

## Slide 7 · Test Coverage

**285 tests — 0 failures** (5 ignored hyperparameter sweeps)

```
  128 unit tests      42 + 9 + 27 + 9 integration tests      2 doc tests
  ─────────────────   ──────────────────────────────────       ───────────
  Analyzer & detect    Roundtrip (9 tests)                     Cloud S3
  Block serialization  Single-file extraction (5)              Decompressor
  FastCDC chunking     Corruption detection (5)
  BWT / MTF / RLE      Determinism (3 tests)
  LZ77 / LZ4 codec    Format & metadata (4)
  Range coding         Edge cases (2 tests)
  All 6 predictors     Streaming decomp (9 tests)
  Header / footer CRC  Dictionary (3 tests)
  Cloud URL parsing    Migration (1 test)
  Dictionary/Analytics Security (27 tests)
  Compressor builders  FFI / Server (9 + 41 tests)
```

Every roundtrip test verifies decompression with **BLAKE3 hash comparison**.
Corruption tests flip individual bytes and assert detection.
4 fuzz targets: block header, streaming metadata, decode block, range coder.

---

## Slide 8 · Predictor Comparison

Two corpora: internal 50 KiB RLE stream and Silesia 8.8 MiB RLE stream.

**Internal corpus (50 KiB):**
```
Predictor                         bpb      Speed
─────────────────────────────────────────────────
RlePredictor (baseline)          3.9754   7.4 MiB/s
D=20 lr=0.02 o2=0.1  (0.1.4)    3.9293   1.2 MiB/s   ← old default
D=32 lr=0.01 o2=0.3  (0.1.5)    3.9320   1.0 MiB/s   ← new default
```

**Silesia corpus (8.8 MiB RLE, text files — primary benchmark):**
```
Predictor                         bpb      Speed
─────────────────────────────────────────────────
RlePredictor (baseline)          3.5052   4.9 MiB/s
D=20 lr=0.02 o2=0.1  (0.1.4)    3.4265   0.8 MiB/s   ← old default
D=32 lr=0.01 o2=0.3  (0.1.5)    3.4121   1.0 MiB/s   ← new default  ★ best
```

The new defaults sacrifice a tiny 0.003 bpb on the small corpus but gain 0.014 bpb on Silesia.
The 5× speed overhead of NeuralSSM over pure RlePredictor is dominated by the predict+update loop.

---

## Slide 9 · Performance Engineering (0.1.5 → 0.2.3)

### Speed progression (87.1 KiB internal corpus)

```
Version   Comp MB/s   Decomp MB/s   Key change
──────────────────────────────────────────────────────
0.1.5        ~0.6         —          Baseline
0.1.6        ~0.9         —          O(rank) MTF, LZ77/sync early-exit
0.1.7        ~0.4        ~0.4        Larger chunks (512 KiB), more BWT work
0.1.8        ~0.9        ~1.2        Sync-skip decompression optimization
0.2.1        ~1.1        ~1.2        #[inline] hints, precomputed EMA arrays
0.2.2        ~1.3        ~1.5        Direct CDF overrides, div→mul optimization
```

### Key optimizations

1. **O(rank) MTF** — stack `[u8; 256]` arrays, O(1) lookup, O(avg_rank) shift
2. **LZ77 Early-Exit** — skip when BWT < 55% (saves O(n) hash-chain + RC pass)
3. **Sync-Predictor Skip** — `predictor_state_flag` in BlockHeader skips O(n) sync
4. **Custom Range Coder** — replaced `constriction` with LZMA-style byte-aligned coder
5. **Binary Search CDF Decode** — 256→8 comparisons in range decoder
6. **Direct CDF Override** — bypass probs_to_cdf() with f32 cumulative rounding (2.6× predictor speedup)
7. **Division→Multiplication** — precompute reciprocals in 254-element literal loops (+20% e2e)
8. **LTO + codegen-units=1** — whole-program optimization in release builds

### V2.3 optimization investigation results

```
  ✅ Kept: direct CDF override, div→mul, LTO          → +20% speed, ratio preserved
  ❌ Reverted: fast_ln (ratio +0.17%), 16 o2 buckets (ratio +0.10%)
  ❌ Reverted: AVX2/SIMD (CPU frequency penalty), unrolled search (cache pressure)
  ❌ Not feasible: SA-based BWT (wrong for cyclic rotations), C FFI (removed in V2.0)
  ⏳ Deferred: cross-block predictor state (needs format v2)
```

### Bottleneck analysis (0.2.3)

```
  1. BWT suffix array construction  (~50%)   ← dominant, requires C FFI for speedup
  2. NeuralSsm predict+update       (~25%)   ← optimized with CDF + div→mul
  3. Range coding                    (~15%)   ← 176 MiB/s encode, 39 MiB/s decode
  4. I/O, hashing, format overhead   (~10%)
```

**Memory scaling:** BWT allocates **~10× chunk size** peak RSS per thread. At 4 MiB max chunk: ~40 MiB/thread.

---

## Slide 10 · Archive Format

Self-describing, random-access, integrity-checked.

```
┌─────────────────────────────────┐
│  Header           48 B          │  magic · flags · predictor ID · offsets
│  [EncryptionHdr   57 B]         │  cipher · salt · KDF params · nonce (opt.)
│  [DictionaryHash  32 B]         │  BLAKE3 of dictionary state (optional)
│  File Table       variable      │  path · size · BLAKE3 · group assignment
│  Group Table      24 B × N      │  content type · method · block range
│  Blocks           variable      │  header(28B) + payload + trailer(36B)
│  Block Index      24 B × N      │  offset table for random-access seeks
│  Footer           32 B          │  redundant counts + magic
└─────────────────────────────────┘
```

- **Integrity**: BLAKE3 per file + CRC32 per block header/trailer
- **Random access**: extract any single file without decompressing others
- **Self-describing**: predictor ID stored in header, auto-detected on extract
- **Encryption**: AES-256-GCM / ChaCha20-Poly1305 with Argon2id KDF (enterprise)
- **Dictionary**: optional pretrained predictor state for domain-specific compression

---

## Slide 11 · Roadmap

### Completed — Core (0.1.5–0.1.8)
- ✅ NeuralSSM tuning on Silesia (D=32, lr=0.01, o2=0.30)
- ✅ Parallel group compression (rayon), pure-Rust BWT (divsufsort)
- ✅ Custom range coder (LZMA-style), 4 MiB chunks
- ✅ Streaming decompression, sync-skip optimization

### Completed — Safety & Quality (0.2.0)
- ✅ Fuzz targets, feature flags, memory backpressure
- ✅ BWT OOM guard, bounds checks, error enrichment

### Completed — Ecosystem (0.2.1)
- ✅ C FFI (cbindgen), Python bindings (PyO3)
- ✅ Encryption: AES-256-GCM / ChaCha20-Poly1305 (enterprise)
- ✅ Multi-threaded decompression (enterprise)
- ✅ Dictionary pretraining, compression analytics
- ✅ Archive migration tool, REST API server, cloud backends

### Completed — Benchmarks & Wasm (0.2.2)
- ✅ Examples directory, `aet bench --compare` external tools
- ✅ Wasm crate (decompress-only via wasm-bindgen)
- ✅ Performance optimization (#[inline], EMA vectorization)

### Completed — V2.3 Performance Optimization
- ✅ Direct CDF overrides (NeuralSSM + Order0): bypass probs_to_cdf(), 2.6-3.4× predictor speedup
- ✅ Division→multiplication in predictor hot loops: +20% end-to-end speed
- ✅ LTO + codegen-units=1 release profile
- ✅ Investigated & documented: AVX2 (freq penalty), SA-based BWT (incorrect), fast_ln (ratio loss)

### Future
- **Faster BWT**: pure-Rust SA-IS with cyclic BWT support (biggest remaining bottleneck, ~50% of wall time)
- **Cross-block predictor state carry** (needs format v2)
- **Pre-trained neural model** — offline-trained SSM via `candle-core`
- **Archive splitting/spanning**, **repair/recovery**
- **crates.io publish**, format freeze, security audit

---

## Summary

| | |
|--|--|
| **What** | Neural-probabilistic file archiver in Rust |
| **How** | BWT + MTF + RUNA/RUNB RLE + diagonal SSM + range coding |
| **Silesia result** | 26.45% ratio — beats gzip-9 by 17.1%; only 2.8% behind bzip2-9 |
| **vs zstd/brotli/lz4** | Beats zstd -19 on text (27.37% vs 27.78%); brotli -q 11 leads (24.89%) |
| **Best case** | x-ray binary: 52.6% vs gzip's 71.3% — 26% smaller |
| **Tests** | 285 passing (128 unit + 87 integration + 28 FFI + 41 server + 1 doc) |
| **Crates** | 6 crates: core, cli, ffi, server, wasm, python |
| **Model** | 66 learnable params (D=32), online, no pre-trained weights |
| **Comp speed (0.2.3)** | ~0.2 MiB/s compress, ~0.3 MiB/s decompress on Silesia |
| **Decomp speed (0.2.3)** | Custom range coder, sync-skip, CDF optimization, ~889s on 202 MiB |
| **Bottleneck** | BWT sort (~50%) · NeuralSsm predict+update (~25%) · Range coding (~15%) |
| **Enterprise** | Encryption, parallel decompress, dictionary, REST API, cloud storage |

---

Copyright 2024-2026 Craton Software Company Licensed under Apache-2.0.
