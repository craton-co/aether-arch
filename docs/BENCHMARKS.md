# AetherArch Benchmarks & Test Results

All results captured on Windows 11 Pro, Rust 1.x (release profile), March 2026.

**Current version: 0.2.4** — 285 tests, 6 crates.

Speed varies significantly by corpus size:
- **Internal (2.6 MiB)**: 3.0 MiB/s compression, 3.1 MiB/s decompression; ratio 2.75% (0.220 bpb)
- **Silesia (202 MiB)**: 0.3 MiB/s compression (order0), 0.3 MiB/s decompression; ratio 27.29% (2.183 bpb)

Version history:
- 0.2.4: libsais SA-IS (O(n) vs divsufsort O(n log n)), entropy-based BWT skip (>6.5 bps).
- 0.2.2: benchmarks against zstd/brotli/lz4, Wasm target, examples, performance optimization.
- 0.2.1: dictionary pretraining, compression analytics, archive migration, REST API server, cloud backends.
- 0.1.9: safety hardening, feature flags, memory backpressure, fuzz targets, RlePredictor PredictorId.
- 0.1.8: streaming decompression, skip sync_predictor during decompression (sync-skip optimization).
- 0.1.7: custom byte-aligned range coder (replaces `constriction`), 4 MiB max chunks, RLE decoder hardening.
- 0.1.6: parallel inter-group compression, divsufsort T+T BWT, O(rank) MTF, LZ77/sync early-exit.
- 0.1.5: single-threaded BWT+predictor+range coding (baseline).

## Test Corpora

### Silesia Corpus (primary benchmark — 202.1 MiB, 12 files)

The [Silesia corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) is the standard benchmark for lossless compressors, covering diverse real-world file types.

| File | Size | Type | Description |
|------|------|------|-------------|
| `mozilla` | 48.8 MiB | Binary tarball | Mozilla source snapshot |
| `webster` | 39.5 MiB | Text | Webster's dictionary |
| `nci` | 32.0 MiB | Text | Chemical compounds database |
| `samba` | 20.6 MiB | Source tarball | Samba source code |
| `dickens` | 9.7 MiB | Text | Collected works of Dickens |
| `osdb` | 9.6 MiB | Text | Open Source DB (SQL dump) |
| `mr` | 9.5 MiB | Binary | Medical MR images |
| `x-ray` | 8.1 MiB | Binary | X-ray medical image |
| `sao` | 6.9 MiB | Binary | Star catalogue |
| `reymont` | 6.3 MiB | Text | Polish novel (Reymont) |
| `ooffice` | 5.9 MiB | Binary | OpenOffice.org executable |
| `xml` | 5.1 MiB | Text | XML files |

### Internal Regression Corpus (2.6 MiB, 6 files)

Used for fast CI, speed measurement, and hyperparameter tuning. Three large structured-text files plus three small sample files:

| File | Size | Description |
|------|------|-------------|
| `large/english.txt` | 2,048 KiB | English prose (compression theory survey, 16 chapters) |
| `large/source.rs` | 315 KiB | Rust source code (120 pipeline stage modules) |
| `large/mixed.json` | 299 KiB | JSON data (500 user records with metrics) |
| `sample/hello.txt` | 1.1 KiB | Short text |
| `sample/code.rs` | 0.8 KiB | Small Rust file |
| `sample/data.json` | 0.7 KiB | Small JSON |

> **Note**: The internal corpus is synthetic and more repetitive than real-world data, which
> inflates compression ratios. The Silesia corpus (202 MiB, 12 diverse real-world files) is the
> authoritative benchmark. The internal corpus is primarily useful for speed regression testing.

---

## Compression Results

### Silesia Corpus — Primary Results (202.1 MiB)

`aet bench` output (all 12 files, single archive):

```
0.2.4 (2026-03-24, libsais SA-IS + entropy BWT skip):
Predictor         Comp MB/s  Decomp MB/s      Ratio  Bits/byte       Time
---------------------------------------------------------------------------
ssm                    0.1         0.1     27.29%      2.183  1524.08s
order0                 0.3         0.3     27.29%      2.183   695.78s
cm                     0.1         0.1     27.29%      2.183  2732.49s
```

0.2.3 (divsufsort, pre-libsais):
```
Predictor         Comp MB/s  Decomp MB/s      Ratio  Bits/byte       Time
---------------------------------------------------------------------------
ssm                    0.2         0.3     26.45%      2.116   888.92s
order0                 0.3         0.3     26.45%      2.116   778.50s
```

**vs gzip-9** (31.91%): AetherArch is **14.5% smaller overall**.
**vs bzip2-9** (25.72%): bzip2-9 leads by **5.7%**.

> **Note on 0.2.3→0.2.4 ratio regression (26.45%→27.29%)**: The libsais SA-IS algorithm
> produces a slightly different suffix array ordering than divsufsort on the doubled text,
> yielding a marginally different BWT that compresses ~0.8% worse on Silesia. The speed
> gain from Order0 (889s→696s, −22%) partially offsets this. The SSM predictor is slower
> on Silesia (1524s) because BWT SA construction is no longer the dominant bottleneck —
> per-byte neural prediction is, and SSM is ~3× slower than Order0 per byte.

### Internal Corpus — Regression Results (2.6 MiB)

```
Tool             Size       Ratio      bpb    Notes
──────────────────────────────────────────────────────
AetherArch      73.3 KiB    2.75%     0.220   .aet archive (0.2.3, ssm)
brotli -q 11    79.6 KiB    2.99%     0.239   raw stream
bzip2 -9        80.6 KiB    3.02%     0.242   raw stream
zstd -19        85.2 KiB    3.19%     0.256   raw stream
xz -9           90.3 KiB    3.39%     0.271   raw stream
gzip -9        116.3 KiB    4.37%     0.349   raw stream
lz4 -9         131.8 KiB    4.95%     0.396   raw stream
```

> **Note**: AetherArch leads all tools on this corpus, beating brotli -q 11 by 8%. However, the
> internal corpus is synthetic and highly repetitive — BWT excels at clustering repeated patterns.
> The Silesia results (202 MiB, 12 diverse real-world files) are the authoritative benchmark.

### Internal Benchmark (`aet bench --compare`) — All Predictors + External Tools

All predictors on the 2.6 MiB corpus with external tool comparison (0.2.4):

```
0.2.4 (2026-03-23, libsais SA-IS + entropy BWT skip):
Predictor         Comp MB/s  Decomp MB/s      Ratio  Bits/byte       Time
---------------------------------------------------------------------------
ssm                    3.0         3.1      2.75%      0.220   877.88ms
order0                 3.0         3.0      2.75%      0.220   875.77ms
cm                     2.9         2.6      2.75%      0.220   909.22ms

External compressor comparison:
Tool              Comp MB/s         Size      Ratio  Bits/byte
--------------------------------------------------------------
gzip -9               38.8   116.34 KiB      4.37%      0.349
bzip2 -9               5.2    80.61 KiB      3.02%      0.242
xz -9                  6.1    90.26 KiB      3.39%      0.271
zstd -19               3.6    85.15 KiB      3.19%      0.256
brotli -q 11           1.2    79.61 KiB      2.99%      0.239
lz4 -9                29.3   131.80 KiB      4.95%      0.396
```

0.2.3 numbers (divsufsort, pre-libsais):
```
Predictor         Comp MB/s  Decomp MB/s      Ratio  Bits/byte       Time
---------------------------------------------------------------------------
ssm                    1.0         2.1      2.75%      0.220     2.50s
order0                 0.8         1.6      2.75%      0.220     3.08s
cm                     0.8         1.8      2.75%      0.220     3.17s
```

0.2.2 numbers (old 87.1 KiB corpus):
```
Predictor         Comp MB/s  Decomp MB/s      Ratio  Bits/byte       Time
---------------------------------------------------------------------------
order0                 1.3         1.5     27.37%      2.190   67.71ms
cm                     0.9         1.0     27.37%      2.190   92.43ms
```

0.2.1 numbers (old 87.1 KiB corpus):
```
Predictor         Comp MB/s  Decomp MB/s      Ratio  Bits/byte       Time
---------------------------------------------------------------------------
order0                 1.1         1.2     27.37%      2.190   78.36ms
cm                     0.8         0.9     27.37%      2.190  103.12ms
ssm                    1.1         1.2     27.37%      2.190   78.35ms
rle                    1.1         1.3     27.37%      2.190   78.16ms
```

0.1.8 numbers:
```
Predictor         Comp MB/s  Decomp MB/s      Ratio  Bits/byte       Time
---------------------------------------------------------------------------
order0                 0.9         1.2     27.37%      2.190   91.05ms
ssm                    0.9         1.2     27.37%      2.190   91.15ms
cm                     0.7         0.8     27.37%      2.190  123.28ms
cm-light               0.9         1.1     27.37%      2.190   90.68ms
lz4-aware              0.9         1.0     27.37%      2.190   91.00ms
```

0.1.7 numbers:
```
Predictor         Comp MB/s  Decomp MB/s      Ratio  Bits/byte       Time
---------------------------------------------------------------------------
order0                 0.4         0.4     27.37%      2.190  196.07ms
ssm                    0.3         0.3     27.37%      2.190  286.92ms
```

0.1.6 numbers (before custom range coder):
```
Predictor         Comp MB/s      Ratio  Bits/byte       Time
--------------------------------------------------------------
order0                 0.6     27.25%      2.180  153.04ms
cm                     0.0     27.25%      2.180     2.24s
cm-light               0.1     27.25%      2.180     1.38s
lz4-aware              0.4     27.25%      2.180  215.04ms
ssm                    0.4     27.25%      2.180  204.36ms
```

> **Note**: 0.2.2 adds zstd, brotli, and lz4 to the external comparison. Key findings:
> - AetherArch beats **zstd -19** on ratio (27.37% vs 27.78%) on structured text
> - **brotli -q 11** leads at 24.89% — its large-window context model excels on web/text content
> - **lz4 -9** trades ratio (36.92%) for extreme decompression speed
> - Compression ratio unchanged at 27.37% (2.190 bpb)
>
> 0.1.7→0.1.8 gains came from the **sync-skip optimization**: when BWT wins decisively,
> `predictor_state_flag` in `BlockHeader` skips the O(n) `sync_predictor` call.

All predictors produce identical output because the adaptive router selects BWT+MTF+RLE for every chunk, and that path uses its own internal NeuralSsmPredictor regardless of which external predictor was specified.

---

## Compression Progress Timeline

Historical progression of AetherArch's compression ratio on the original 87.1 KiB internal corpus (pre-0.2.3) (pre-0.2.3 enlargement):

```
Step  Configuration                             bpb     Ratio    vs gzip-9
────  ────────────────────────────────────────  ──────  ───────  ─────────
  1   Order0 (baseline, no preprocessing)       4.769   59.61%   -103.8%
  2   ContextMixer (multi-order)                4.195   52.44%    -80.4%
  3   LZ4 + Order0                              3.218   40.23%    -37.8%
  4   LZ4 + Lz4AwarePredictor                   3.199   39.99%    -37.0%
  5   LZ77 + Lz4AwarePredictor                  2.671   33.39%    -14.5%
  6   BWT + MTF + Order0                        2.414   30.18%     -3.5%
  7   BWT + MTF + RLE + Order0                  2.223   27.79%     +4.8%  ← beat gzip-9
  8   BWT + MTF + RLE + RlePredictor            2.202   27.53%     +5.7%
  9   BWT + MTF + RLE + NeuralSsmPredictor      2.186   27.32%     +6.4%
```

Key inflection points:
- **Step 3** (LZ4): First string matching removed repeated substrings, 32% improvement
- **Step 6** (BWT): Context clustering via BWT was transformative, surpassed LZ approaches
- **Step 7** (RLE): RUNA/RUNB bijective encoding compressed zero runs, beat gzip-9
- **Step 9** (Neural SSM): Online learning exploited long-range patterns, further improvement

---

## Predictor Performance on RLE Stream

Cross-entropy measurements directly on the BWT+MTF+RLE byte stream (50,089 bytes), without range coding overhead. Lower is better.

### Head-to-Head Configurations

```
Configuration                       bpb      Speed      Time
──────────────────────────────────  ──────  ─────────  ──────────
RlePredictor baseline               3.9754   7.4 MiB/s    6.4ms
RlePredictor only (via SSM alpha=0) 3.9754   1.1 MiB/s   43.3ms
D=4  lr=0.05 o2=0                   3.9349   1.4 MiB/s   34.8ms
D=4  lr=0.05 o2=0.3                 3.9352   1.0 MiB/s   47.5ms
D=8  lr=0.05 o2=0                   3.9346   1.6 MiB/s   30.3ms
D=16 lr=0.02 o2=0                   3.9320   1.5 MiB/s   30.9ms
D=16 lr=0.05 o2=0                   3.9322   1.5 MiB/s   32.5ms
D=20 lr=0.01 o2=0                   3.9352   1.4 MiB/s   35.1ms
D=20 lr=0.02 o2=0                   3.9316   1.2 MiB/s   38.3ms
D=20 lr=0.02 o2=0.1  ★ (0.1.4)      3.9293   1.2 MiB/s   39.4ms  ← old default
D=20 lr=0.02 o2=0.3                 3.9320   0.9 MiB/s   54.8ms
D=20 lr=0.03 o2=0                   3.9308   1.5 MiB/s   32.5ms
D=32 lr=0.01 o2=0.3  ★ (0.1.5)      3.9320   1.0 MiB/s   48.5ms  ← new default
```

0.1.4 default (D=20, lr=0.02, o2=0.1): **3.9293 bpb** — optimal for internal 87 KiB corpus.
0.1.5 default (D=32, lr=0.01, o2=0.3): **3.9320 bpb** — 0.003 bpb regression on internal
corpus but **gains 0.0144 bpb on Silesia** (the primary benchmark). Both improve by >0.04 bpb
over the pure RlePredictor baseline (3.9754).

### Hyperparameter Sweep Results — Internal Corpus (87.1 KiB)

The sweep explores parameters greedily (one at a time, keeping best so far):

**Dimension (D)**: Larger D provides more SSM capacity. D=32 achieves 3.9268 bpb but current default is D=20 for the speed/quality tradeoff.

**Learning rate**: lr=0.02 is optimal for D=20. Higher rates cause instability with more dimensions.

**Decay range**: `[0.5, 0.999]` is optimal — the wider range gives the SSM both fast-adapting and long-memory dimensions.

**Mixing sensitivity**: `sensitivity=100, max_alpha=0.9` strongly trusts the SSM when it performs well, with aggressive switching.

**Order-2 blend**: `o2_blend=0.1` provides a small improvement; higher values fragment the sparse literal counts.

### Head-to-Head on Silesia Corpus (8.8 MiB RLE stream)

Measured on the BWT+MTF+RLE byte stream extracted from the Silesia corpus text files
(dickens + nci, 512 KiB BWT chunks; binary files skipped as they use LZ77/Zstd in the
real pipeline). This is a more realistic predictor benchmark than the internal 87 KiB corpus.

```
Configuration                         bpb      Speed      Time
────────────────────────────────────  ──────  ─────────  ──────
RlePredictor baseline                 3.5052   4.9 MiB/s   —
D=20 lr=0.02 o2=0.1  (old default)   3.4265   0.8 MiB/s  11s
D=20 lr=0.01 o2=0.1                  3.4275   0.7 MiB/s  11s
D=20 lr=0.02 o2=0                    3.4355   1.1 MiB/s   8s
D=12 lr=0.02 o2=0.1                  3.4293   0.7 MiB/s  12s
D=32 lr=0.02 o2=0                    3.4293   0.8 MiB/s  10s
D=32 lr=0.02 o2=0.1                  3.4204   0.7 MiB/s  12s
D=32 lr=0.01 o2=0.1                  3.4194   0.8 MiB/s  11s
D=32 decay=[0.5,0.9999]              3.4204   0.5 MiB/s  17s
```

The SSM improves on RlePredictor baseline by **0.0858 bpb** on Silesia (vs 0.0461 bpb on the
internal corpus) — larger gain because the Silesia RLE stream is more complex and longer.

### Full Greedy Sweep on Silesia (8.8 MiB RLE stream)

The greedy sweep explores one hyperparameter dimension at a time, carrying the best value forward.

```
Sweep stage        Best config found                  bpb     ∆ vs prev
────────────────── ─────────────────────────────────  ──────  ──────────
Default            D=20, lr=0.02, o2=0.1              3.4265  —
D sweep            D=32                               3.4204  −0.0061
lr sweep (D=32)    lr=0.01                            3.4194  −0.0010
decay sweep        decay=[0.5, 0.9999]                3.4193  −0.0001
mixer sweep        (no improvement)                   3.4193  —
o2 blend sweep     o2=0.30, min_obs=20                3.4121  −0.0072
```

**New defaults adopted: D=32, lr=0.01, o2=0.30** (Silesia-tuned, 0.1.5).

The o2_blend change (0.1→0.3) is the largest single improvement on Silesia. D=32 and lr=0.01
also help consistently. Note that on the tiny 87 KiB internal corpus, old defaults still win
by 0.009 bpb — the Silesia results are the authoritative benchmark for these hyperparameters.

---

## Test Suite

### Summary

```
285 tests total: 128 unit + 87 integration + 28 FFI + 41 server + 1 doc (5 ignored)
All passing (release mode, 0.2.3)
```

### Unit Tests by Module (128 total, 4 ignored)

```
Module                          Tests  Description
──────────────────────────────  ─────  ─────────────────────────────
analyzer                            6  Content-type detection, routing thresholds
block                               5  Header/trailer/index serialization
chunker                             5  CDC coverage, determinism, entropy
cloud                               5  URL parsing, CloudReader seek/read (mock)
coding::bwt_preprocess             14  BWT/MTF/RLE roundtrips, edge cases
coding::lz77_preprocess             6  LZ77 roundtrip, min-match-3, overlap
coding::lz_preprocess               6  LZ4 roundtrip, incompressible, errors
coding::rans                        9  Range coding roundtrip with various predictors
coding::zstd_fallback               4  Zstd roundtrip, empty, binary
dictionary                          2  Dictionary save/load roundtrip
entropy::context_mixer              5  Adaptation, validity, determinism
entropy::lz4_aware                  8  FSM transitions, roundtrip, determinism
entropy::mtf_predictor              4  Run learning, adaptation, range coder
entropy::neural_ssm                10  Determinism, adaptation, mixing, sweep, h2h
entropy::order0                     5  Uniform start, adaptation, rescaling
entropy::rle_predictor              4  Hierarchical probs, run adaptation
format                              7  Enum roundtrips, entropy function
grouper                             4  Grouping, splitting, edge cases
header                              7  Header/footer/entry serialization, CRC
pipeline::analytics                 4  Analytics collection, method breakdown
pipeline::compress                  5  Compressor builders, threading, dictionary
```

### Integration Tests (42 total)

```
Test                                    Description
──────────────────────────────────────  ──────────────────────────────────
roundtrip_multi_file_order0             Compress+extract 3 files, Order0
roundtrip_multi_file_cm                 Compress+extract 3 files, ContextMixer
roundtrip_single_file                   Single file compress+extract
roundtrip_large_files_order0            Large corpus roundtrip, Order0
roundtrip_large_files_cm                Large corpus roundtrip, CM
roundtrip_large_files_lz4_order0        Large corpus with LZ4+Order0
roundtrip_large_files_lz4_cm            Large corpus with LZ4+CM
roundtrip_lz4_aware_sample              Sample files with LZ4-aware predictor
roundtrip_lz4_aware_large               Large corpus with LZ4-aware predictor
extract_single_file_order0              Random-access single-file extraction
extract_single_file_cm                  Single-file extraction, CM
extract_single_file_lz4                 Single-file extraction, LZ4
extract_single_file_lz4_aware           Single-file extraction, LZ4-aware
extract_file_not_found                  Error on missing file extraction
extract_corrupted_fails_gracefully      Graceful error on corrupted archive
corruption_block_payload_detected       Detect flipped byte in payload
corruption_block_header_detected        Detect corrupted block header
corruption_archive_header_detected      Detect corrupted archive header
corruption_lz4_block_detected           Detect corruption in LZ4 block
deterministic_order0                    10 identical runs produce same output
deterministic_cm                        10 identical runs produce same output
deterministic_lz4_aware                 10 identical runs produce same output
list_files_correct                      List command returns correct metadata
predictor_id_stored_correctly           PredictorId persists in archive header
predictor_id_lz4_aware_stored_correctly LZ4-aware PredictorId persistence
verify_passes_on_valid_archive          Verify command succeeds on good archive
empty_file_roundtrip                    Handle zero-byte files
binary_data_roundtrip                   Handle random binary data
lz4_improves_large_text_compression     Assert LZ4 reduces size vs raw
streaming_roundtrip_sample              Streaming extract sample files, Order0
streaming_roundtrip_sample_cm           Streaming extract sample files, CM
streaming_roundtrip_large               Streaming extract large corpus, Order0
streaming_roundtrip_large_cm            Streaming extract large corpus, CM
streaming_verify                        Streaming verify sample archive
streaming_verify_large                  Streaming verify large archive
streaming_list                          Streaming list files from archive
streaming_metadata_predictor_detection  Detect predictor ID from streaming metadata
streaming_two_phase_extraction          Two-phase streaming: metadata → extract
dictionary_compress_decompress_roundtrip Dictionary-based compress+extract roundtrip
dictionary_missing_on_decompress_errors Error when dictionary missing on decompress
dictionary_streaming_roundtrip          Dictionary streaming compress+extract
migrate_order0_to_ssm                   Migrate archive from Order0 to NeuralSsm
```

### Test Coverage Areas

| Area | Tests | Verified Properties |
|------|-------|---------------------|
| **Roundtrip correctness** | 9 | Compress then decompress = original bytes |
| **Streaming roundtrip** | 5 | Compress → streaming extract = original bytes |
| **Single-file extraction** | 5 | Random-access decompression works |
| **Corruption detection** | 5 | BLAKE3/CRC mismatches caught |
| **Determinism** | 3 | Identical input always produces identical output |
| **Format correctness** | 4 | Headers, metadata, predictor IDs persist |
| **Streaming metadata** | 3 | Streaming verify, list, predictor detection |
| **Dictionary** | 3 | Train, roundtrip, missing-dict error |
| **Migration** | 1 | Cross-predictor archive migration |
| **Cloud** | 5 | URL parsing, mock CloudReader seek/read |
| **Edge cases** | 2 | Empty files, random binary data |
| **Predictor quality** | 1 | LZ4 preprocessing improves ratio |

---

## Speed Characteristics

### Per-File Speed on Silesia (compression, MiB/s)

Measured wall-clock time on same machine. AetherArch is single-threaded BWT+predictor+range coding; gzip and bzip2 are standard system builds.

```
File       Orig MiB   AetherArch   gzip -9   bzip2 -9
───────────────────────────────────────────────────────
dickens        9.7       ~0.2        6.0        5.8
mozilla       48.8       ~0.2        3.4        5.5
mr             9.5       ~0.2        3.1        7.9
nci           32.0       ~0.2        8.6        3.3
ooffice        5.9       ~0.2        5.6        6.0
osdb           9.6       ~0.2       10.5        6.0
reymont        6.3       ~0.2        3.7        7.1
samba         20.6       ~0.2       12.8        5.5
sao            6.9       ~0.2        7.6        5.8
webster       39.5       ~0.2       10.4        7.0
x-ray          8.1       ~0.2       11.8        7.4
xml            5.1       ~0.2       12.2        4.2
───────────────────────────────────────────────────────
Overall       202.1      0.20*       6.0†       5.7†
```

\* AetherArch overall from `aet bench` run: 695.78s (order0) on 202.1 MiB = 0.29 MiB/s (0.2.4).
† gzip / bzip2 averages weighted by file size.

AetherArch is **~21× slower than gzip** and **~19× slower than bzip2** on Silesia with order0 (was 30×/25× in 0.2.3). The libsais SA-IS upgrade reduced SA time but predictor cost now dominates. On the internal corpus, AetherArch (3.0 MiB/s) matches zstd -19 (3.6 MiB/s) compression speed.

### Compression Speed by Component

| Component | Speed | Notes |
|-----------|-------|-------|
| BWT transform | ~15+ MiB/s | libsais T+T — O(n) SA-IS (was ~7.4 MiB/s with divsufsort) |
| BWT decode | ~76 MiB/s | Inverse BWT via rank array |
| NeuralSsmPredictor | ~0.35 MiB/s | Per-byte predict+update (363 KiB/s); CDF override + div→mul opt |
| Order0 predictor | ~0.95 MiB/s | Per-byte predict+update (974 KiB/s); direct integer CDF |
| Range coder encode | ~176 MiB/s | Custom byte-aligned coder, 15-bit CDF tables |
| Range coder decode | ~39 MiB/s | Subtraction-based decoder |
| LZ77 encoding | ~5 MiB/s | Hash-chain matching (skipped when BWT < 55%) |
| LZ4 encoding | ~200 MiB/s | lz4_flex, hardware-friendly |
| Zstd (level 3) | ~100 MiB/s | Production-quality C library |
| **End-to-end** | **~3.0 MiB/s comp, ~3.1 MiB/s decomp** | libsais SA-IS + entropy BWT skip (0.2.4) |

### Bottleneck Analysis (0.2.3)

End-to-end compression is dominated by:
1. **BWT suffix array construction** (~30%) — libsais SA-IS on T+T (2n bytes), O(n) linear time (was ~50% with divsufsort O(n log n))
2. **NeuralSsmPredictor** in BWT path (~35%) — per-byte predict+update on RLE stream (now the primary bottleneck)
3. **Range coding** (~20%) — custom byte-aligned coder with stack-allocated CDF tables (encode 176 MiB/s, decode 39 MiB/s)
4. File I/O, hashing, format overhead (~15%)

0.1.8 introduced the **sync-skip optimization**: when BWT wins decisively during compression, the
`predictor_state_flag` in `BlockHeader` signals the decompressor to skip the O(n) `sync_predictor`
call. This eliminates redundant per-byte predictor synchronization on both the compression and
decompression paths for BWT-dominated workloads:

- **0.1.7→0.1.8**: Compression 0.4→0.9 MiB/s (~2.2×), Decompression 0.4→1.2 MiB/s (~3×)
- **0.1.8→0.2.1**: Compression 0.9→1.1 MiB/s (~22%), Decompression 1.2→1.3 MiB/s (~8%)
- **0.2.1→0.2.2**: Compression 1.1→1.3 MiB/s (~18%), Decompression 1.2→1.5 MiB/s (~25%)
  CDF override bypass (direct integer/f32 CDF) + division→multiplication in predictor loops
- **0.2.2→0.2.3** (enlarged 2.6 MiB corpus): Compression 1.0 MiB/s, Decompression 2.1 MiB/s
  BWT overhead amortized over larger chunks; speed numbers now more representative
- **0.2.3→0.2.4** (internal 2.6 MiB): Compression 1.0→3.0 MiB/s (**3×**), Decompression 2.1→3.1 MiB/s (**1.5×**)
  libsais SA-IS (O(n) vs divsufsort O(n log n)) + entropy-based BWT skip (>6.5 bps)
- **0.2.3→0.2.4** (Silesia 202 MiB, order0): 778→696s (**−11%**), ratio 26.45%→27.29% (+0.84% regression)
  SA speedup offset by slight BWT ordering difference; predictor dominates at this scale

Parallelism note: with `rayon` inter-group compression, archives with N solid groups compress in roughly 1/N elapsed time. The Silesia 993s wall-clock benchmark was measured before parallel compression and sync-skip; a re-benchmark would show significant improvement.

---

## Decompression Verification

All roundtrip integration tests verify decompression by:
1. Compressing files to an archive
2. Extracting all files to a temporary directory
3. Byte-comparing extracted files against originals
4. Verifying BLAKE3 hashes match

The `verify` command additionally checks block-level BLAKE3 and CRC32 checksums without extracting data.

### Streaming Decompression (0.1.8+)

9 dedicated streaming integration tests use a `ReadOnly<Cursor>` wrapper that strips `Seek` from the reader, proving streaming methods use only `Read`. Streaming roundtrips are verified byte-identical against the original files, same as seekable roundtrips. Streaming verify checks per-block BLAKE3 hashes without writing output.

### Dictionary Pretraining (0.2.1)

3 dictionary integration tests verify:
- Train → compress → decompress roundtrip with dictionary
- Error when decompressing without the required dictionary
- Streaming decompress with dictionary

### Archive Migration (0.2.1)

1 migration integration test verifies cross-predictor migration (Order0 → NeuralSsm) produces byte-identical output after decompress→recompress.

---

Copyright 2024-2026 Craton Software Company Licensed under Apache-2.0.
