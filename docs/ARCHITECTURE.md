# AetherArch Architecture

This document describes the internal design of AetherArch in detail: the archive format, compression pipeline, predictor models, preprocessing transforms, and how they all fit together.

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Archive Format Specification](#archive-format-specification)
3. [Compression Pipeline](#compression-pipeline)
4. [Preprocessing Transforms](#preprocessing-transforms)
5. [Probability Predictors](#probability-predictors)
6. [Range Coding](#range-coding)
7. [Adaptive Routing](#adaptive-routing)
8. [Module Reference](#module-reference)

---

## Design Philosophy

Traditional archivers (gzip, zip) use a fixed LZ77+Huffman pipeline. AetherArch instead separates **modeling** (predicting what byte comes next) from **coding** (encoding the actual byte given the prediction). This separation allows plugging in increasingly sophisticated models without changing the coding layer.

Key design decisions:

- **Custom Range Coding**: We use a purpose-built byte-aligned range coder (LZMA-style carry-propagating encoder + subtraction-based decoder) with 15-bit CDF precision. This eliminates per-symbol model allocation overhead and works directly with the predictor's CDF tables. The predictor produces `P(byte_N | bytes_0..N-1)` in forward order — Range Coding encodes/decodes in the same direction.

- **Online learning, no pre-trained weights**: All predictors adapt from scratch on each block. This means no model files to distribute and guaranteed deterministic behavior — the decompressor runs the identical predict/update loop to reproduce probabilities.

- **Adaptive routing**: Instead of committing to one compression method, each chunk tries multiple transforms (BWT, LZ77, plain) and picks the smallest result. The overhead of trying multiple paths is negligible compared to the I/O cost.

- **Semantic solid grouping**: Files are grouped by content type (text, images, executables) so predictors see similar data in sequence, improving cross-file learning within a solid group.

---

## Archive Format Specification

All integers are little-endian. The format is designed for both streaming writes and random-access reads.

### Layout Overview

```
Offset  Section                     Size
──────  ──────────────────────────  ──────────────
0       Archive Header              48 bytes (fixed)
48      File Table                  variable
        Solid Group Table           24 bytes x group_count
        Compressed Blocks           variable
        Block Index                 24 bytes x block_count
        Archive Footer              32 bytes (fixed)
```

### Archive Header (48 bytes)

```
Offset  Field                Type      Description
──────  ───────────────────  ────────  ──────────────────────────
0       magic                [u8; 8]   0xAE "ther" 0x00 0x01 0x02
8       flags                u16       Bitfield (see below)
10      predictor_id         u16       PredictorId enum value
12      file_count           u32       Number of files in archive
16      solid_group_count    u32       Number of solid groups
20      block_count          u32       Total compressed blocks
24      file_table_offset    u64       Byte offset of file table
32      block_index_offset   u64       Byte offset of block index
40      checksum             u32       CRC32 of bytes 0..39
44      reserved             u32       Must be 0
```

**Header flags**:
- Bit 0 (`0x0001`): `FLAG_HAS_NEURAL_MODEL` — archive uses neural predictor
- Bit 1 (`0x0002`): `FLAG_SOLID_ARCHIVE` — solid grouping enabled
- Bit 2 (`0x0004`): `FLAG_ENCRYPTED` — archive uses AES-256-GCM or ChaCha20-Poly1305 encryption
- Bit 3 (`0x0008`): `FLAG_HAS_DICTIONARY` — archive includes a pretrained dictionary
- Bit 4 (`0x0010`): `FLAG_PATH_PREFIXES` — file paths share prefixes with the previous entry

### Encryption Header (57 bytes, optional)

Present immediately after the Archive Header when `FLAG_ENCRYPTED` is set.

```
Offset  Field                Type         Description
──────  ───────────────────  ──────────   ──────────────────────────
0       cipher_id            u8           0x01 = AES-256-GCM, 0x02 = ChaCha20-Poly1305
1       salt                 [u8; 32]     Argon2id salt (random)
33      m_cost               u32 LE       Argon2id memory cost in KiB (default: 65536 = 64 MiB)
37      t_cost               u32 LE       Argon2id time cost / iterations (default: 3)
41      p_cost               u32 LE       Argon2id parallelism lanes (default: 4)
45      nonce                [u8; 12]     Master nonce (random, from OS CSPRNG)
```

Per-block nonces are derived as `master_nonce XOR block_id` (little-endian u32
XORed into the first 4 bytes of the nonce). This enables random-access
decryption without storing separate nonces per block.

### Dictionary Hash (32 bytes, optional)

Present immediately after the Encryption Header (or after the Archive Header
if unencrypted) when `FLAG_HAS_DICTIONARY` is set.

```
Field                Type         Description
───────────────────  ──────────   ──────────────────────────
dictionary_hash      [u8; 32]     BLAKE3 hash of the dictionary used during compression
```

The decompressor uses this hash to verify that the correct dictionary is
provided before attempting decompression.

### File Table Entry (variable length)

Written sequentially after the archive header.

```
Field                Type         Description
───────────────────  ───────────  ─────────────────────────
path_len             u32          Length of file path string
path                 [u8; N]      UTF-8 file path
original_size        u64          Uncompressed file size
blake3_hash          [u8; 32]     BLAKE3 hash of original content
solid_group_id       u32          Which solid group this file belongs to
chunk_start_idx      u32          Index of first chunk
chunk_count          u32          Number of chunks for this file
permissions          u32          Unix-style file permissions
mtime                i64          Modification time (Unix timestamp)
```

### Solid Group Entry (24 bytes)

```
Field                Type         Description
───────────────────  ───────────  ─────────────────────────
group_id             u32          Unique group identifier
content_type         u16          ContentType enum value
compression_method   u8           Suggested compression method
padding              u8           Reserved (0)
first_block_idx      u32          Index of first block in group
block_count          u32          Number of blocks in group
file_count           u32          Number of files in group
```

### Block Header (28 bytes)

Each compressed block is prefixed by this header.

```
Field                Type         Description
───────────────────  ───────────  ─────────────────────────
magic                u32          0xB10CAE01
block_id             u32          Unique block ID
solid_group_id       u32          Group this block belongs to
method               u8           CompressionMethod enum value
predictor_state      u8           1 if predictor state is reset
compressed_size      u32          Size of compressed payload
uncompressed_size    u32          Original data size
crc                  u32          CRC32 of header bytes 0..23
padding              [u8; 2]      Reserved (0)
```

### Block Trailer (36 bytes)

Follows the compressed payload for each block.

```
Field                Type         Description
───────────────────  ───────────  ─────────────────────────
content_blake3       [u8; 32]     BLAKE3 of original uncompressed data
crc                  u32          CRC32 of trailer bytes 0..31
```

### Block Index Entry (24 bytes)

Written after all blocks, enables random-access seeking.

```
Field                Type         Description
───────────────────  ───────────  ─────────────────────────
block_id             u32          Block identifier
archive_offset       u64          Absolute byte offset in archive
compressed_size      u32          Size of compressed payload
uncompressed_size    u32          Original data size
solid_group_id       u32          Group identifier
```

### Archive Footer (32 bytes)

Redundant metadata at the end of the file for backward seeking.

```
Field                Type         Description
───────────────────  ───────────  ─────────────────────────
block_index_offset   u64          Offset of block index section
file_table_offset    u64          Offset of file table section
block_count          u32          Number of blocks
file_count           u32          Number of files
crc                  u32          CRC32 of footer bytes 0..27
magic                u32          0xAE454E44 ("AE" + "END")
```

### Compression Method Values

| Value | Name | Description |
|-------|------|-------------|
| 0 | `PredictorRans` | Direct predictor + range coding |
| 1 | `Zstd` | Zstandard (level 3) |
| 2 | `Store` | Uncompressed |
| 3 | `LzPredictorRans` | LZ4 + predictor + range coding |
| 4 | `Lz77PredictorRans` | LZ77 + predictor + range coding |
| 5 | `BwtPredictorRans` | BWT + MTF + RLE + predictor + range coding |
| 6 | `BytePlanePredictorRans` | Numeric byte-plane splitting + range coding |
| 7 | `BcjZstd` | x86/x86-64 BCJ normalization + Zstandard |

### Predictor ID Values

| Value | Name | Description |
|-------|------|-------------|
| 0x0000 | `Order0` | Byte frequency counting |
| 0x0001 | `ContextMixer` | Multi-order context mixer (full) |
| 0x0002 | `NeuralSsm` | Neural SSM + RLE hybrid |
| 0x0003 | `ContextMixerLight` | Context mixer (lightweight) |
| 0x0004 | `Lz4Aware` | LZ4 FSM predictor |
| 0x0005 | `Rle` | Hierarchical RLE stream predictor |
| 0x00FF | `ZstdOnly` | Zstd-only mode (no predictor) |

---

## Compression Pipeline

### High-Level Flow

```
                    ┌──────────────┐
                    │  Input Files │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Scan &     │  BLAKE3 hash, entropy, content type
                    │   Analyze    │  (analyzer.rs)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Semantic   │  Group by ContentType
                    │   Grouping   │  (grouper.rs)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Per-Group   │  Files concatenated within group
                    │  Chunking    │  FastCDC (16-512-8192 KiB)
                    └──────┬───────┘  (chunker.rs)
                           │
              ┌────────────▼────────────┐
              │    Adaptive Routing     │  (router.rs)
              │    (per chunk)          │
              └─┬────┬────┬────┬────┬──┘
                │    │    │    │    │
          ┌─────▼┐ ┌─▼──┐│ ┌──▼┐ ┌▼────┐
          │ BWT  │ │LZ77││ │RC │ │Zstd │  Pick smallest
          │+MTF  │ │    ││ │   │ │     │
          │+RLE  │ │    ││ │   │ │     │
          │→RC   │ │→RC ││ │   │ │     │
          └──────┘ └────┘│ └───┘ └─────┘
                         │ Store
                           │
                    ┌──────▼───────┐
                    │   Archive    │  Header, file table, groups,
                    │   Assembly   │  blocks, index, footer
                    └──────────────┘
                    (compress.rs, writer.rs)
```

### Pipeline Steps (compress.rs)

1. **Scan files**: Read each file, compute BLAKE3 hash, measure Shannon entropy, detect content type from magic bytes and file extension.

2. **Group files**: `grouper.rs` clusters files by `ContentType` (Text, Image, Executable, etc.). Each group becomes a solid group where the predictor's state carries across files, improving compression.

3. **Write header placeholder**: Write a 48-byte archive header with zero offsets (patched later).

4. **Write file table**: Serialize all `FileEntry` structs (paths, sizes, hashes, group assignments).

5. **Write solid group table**: Serialize `SolidGroupEntry` structs.

6. **Compress blocks**: Groups are compressed in **parallel** via `rayon::par_iter()` — each solid group has an independent predictor, so groups can be compressed concurrently without shared mutable state. Within each group, chunks remain sequential so the predictor state carries across blocks. Results are buffered in memory (Phase A), then written to the archive in deterministic group order (Phase B) for reproducible byte layout regardless of thread scheduling.

7. **Write block index**: Record each block's archive offset and sizes for random-access decompression.

8. **Write footer**: Redundant offsets and counts.

9. **Patch header**: Seek back to byte 0 and overwrite the header with actual offsets and counts.

### Decompression (decompress.rs)

1. Read footer (last 32 bytes) to locate block index and file table.
2. Read file table and block index.
3. For full extraction: decompress blocks sequentially, maintaining predictor state per group.
4. For single-file extraction: use block index to seek directly to the relevant blocks.
5. After decompressing each block, verify BLAKE3 hash against the stored trailer.

---

## Preprocessing Transforms

### BWT + MTF + RLE (bwt_preprocess.rs)

The most effective transform chain for structured data. Used as the primary compression path.

**Burrows-Wheeler Transform (BWT)**: Sorts all cyclic rotations of the input, producing output where similar contexts cluster together. A byte preceded by "th" will appear near other bytes preceded by "th". This makes the output highly predictable.

Implementation uses the **doubled-text trick** with `divsufsort` (pure-Rust suffix array crate): sort suffixes of T+T (the input text concatenated with itself), then keep only the n positions < n. Any two positions i,j < n in T+T have at least n characters to compare — exactly the cyclic rotation content — so the filtered suffix array equals the cyclic rotation SA. This gives O(n log n) suffix array construction in practice, replacing the previous O(n log² n) prefix-doubling approach.

**Move-to-Front (MTF)**: Maintains a dynamic ranking of 256 byte values. Each input byte is encoded as its current rank, then promoted to rank 0. Recently-seen bytes get small ranks. After BWT, this produces many zeros and small values.

Implementation uses stack-allocated `[u8; 256]` arrays (`pos[byte] = rank`, `list[rank] = byte`) for O(1) rank lookup and O(average_rank) shift per symbol — replacing the previous O(256) `Vec::position()` + `Vec::remove()` + `Vec::insert()` per symbol. For BWT output, over 60% of symbols have rank 0 (zero-cost) and the average rank is typically 2–5.

**RUNA/RUNB Run-Length Encoding**: Encodes runs of zero using bijective base-2 encoding:
- Value 0 (RUNA) and 1 (RUNB) encode zero-run lengths
- Values 2-255 encode MTF values 1-254
- Run length decoding: `RUNA` = 1, `RUNB` = 2, `RUNA RUNA` = 3, `RUNB RUNA` = 4, etc. (bijective base-2)

**Payload format**: `[flags: u8] [primary_index: u32 LE] [encoded_len: u32 LE] [range-coded data]`
- flags bit 0: 1 = RLE applied, 0 = raw MTF output

This transform chain reduces an 87 KiB English text corpus from 4.769 bpb (raw Order-0) to 2.223 bpb before applying any sophisticated predictor.

### LZ77 (lz77_preprocess.rs)

Custom LZ77 implementation with:
- 64 KiB sliding window
- Minimum match length 3 (not 4 like LZ4)
- Maximum match length 65,538
- Hash-chain match finder (16-bit hash, chain depth 4,096)
- Lazy matching with nice-match threshold at 258
- Token format: `[original_size: u32 LE] [token_stream]`

Captures short repeated patterns that BWT misses, particularly effective for structured source code.

### LZ4 (lz_preprocess.rs)

Thin wrapper around the `lz4_flex` crate. Returns `None` if LZ4 doesn't reduce size. Used as a fast preprocessing step when BWT is too slow or doesn't help.

---

## Probability Predictors

All predictors implement the `ProbabilityPredictor` trait:

```rust
pub trait ProbabilityPredictor: Send {
    fn predict(&mut self) -> [f32; 256];    // P(next_byte)
    fn predict_cdf(&mut self) -> [u16; 257]; // 15-bit CDF (default: via predict())
    fn update(&mut self, byte: u8);         // advance state
    fn reset(&mut self);                    // reset for new block
    fn name(&self) -> &str;
    fn predictor_id(&self) -> PredictorId;
}
```

The contract: call `predict()` to get the probability distribution, then `update()` with the actual byte. The decompressor runs the same loop to reproduce identical probabilities.

### Order-0 Model (order0.rs)

The simplest predictor. Counts byte frequencies with Laplace smoothing (pseudocount = 1). Rescales when total exceeds 1,000,000 to prevent overflow and maintain adaptation.

- Memory: ~1 KiB
- Typical: 4.769 bpb on raw text, 2.223 bpb on BWT+RLE stream

### Context Mixer (context_mixer.rs)

PAQ-inspired multi-order context mixer. Maintains independent hash-table models for orders 1 through 8 (full) or 1 through 6 (lightweight). Predictions are combined in log-odds space with adaptive gradient-based weight updates.

Each `OrderNModel` uses FNV-1a hashing to map N-byte contexts to byte frequency count arrays. Weights are updated after each byte using the gradient of log-loss.

- Memory: ~100 MiB (full) or ~25 MiB (lightweight)
- Typical: 4.195 bpb on raw text
- Speed: Very slow (~0.03 MiB/s) due to large hash-table lookups

### LZ4-Aware Predictor (lz4_aware.rs)

A finite-state-machine predictor that understands the LZ4 byte stream format. Tracks which part of the LZ4 token structure the current byte belongs to and dispatches to specialized sub-predictors.

**FSM States**:
```
SizePrefix(0..3) → Token → [LitLenExt] → Literals(n)
    → MatchOffsetLow → MatchOffsetHigh → [MatchLenExt] → Token → ...
```

The key insight: the Literals sub-predictor maintains its own context buffer of only literal bytes (not interleaved LZ4 control bytes), giving it clean multi-order context on the actual text data.

- Memory: ~8 MiB
- Typical: 3.199 bpb on LZ4 stream, 2.671 bpb on LZ77 stream

### RLE Predictor (rle_predictor.rs)

Hierarchical predictor designed specifically for the RUNA/RUNB RLE stream produced by BWT+MTF+RLE. Uses three context classes:

| Context | Condition | Description |
|---------|-----------|-------------|
| `CTX_START` | First byte | Initial state |
| `CTX_IN_RUN` | Previous was 0 or 1 | Inside a zero run |
| `CTX_AFTER_LIT` | Previous was >= 2 | After a literal value |

For each context, the predictor makes hierarchical decisions:
1. **Run vs Literal**: Is the next byte a run symbol (0-1) or a literal (>=2)? Binary model with alpha=0.5.
2. **If run**: RUNA (0) or RUNB (1)? Binary model.
3. **If literal**: Which value (2-255)? 254-way counting model with alpha=0.1.

- Memory: ~3 KiB
- Typical: 2.202 bpb on BWT+RLE stream

### Neural SSM Predictor (neural_ssm.rs)

The most sophisticated predictor. A hybrid that combines a diagonal linear State Space Model with the RlePredictor baseline and an order-2 literal context model.

**Architecture**:

```
Input byte
    │
    ├──► Embedding lookup (256 × D deterministic vectors)
    │
    ├──► SSM state update: h[d] = a[d]*h[d] + (1-a[d])*embed[byte][d]
    │    (exponential moving average at D different timescales)
    │
    ├──► Binary classifier 1: sigmoid(w_run . h + b_run)  →  P(run | SSM)
    ├──► Binary classifier 2: sigmoid(w_runa . h + b_runa) →  P(RUNA | run, SSM)
    │
    ├──► RlePredictor: P(run), P(RUNA), P(literal value)
    │
    ├──► Adaptive mixer:
    │    alpha = f(EMA_ssm_loglik - EMA_rle_loglik)
    │    P_mixed = alpha * P_ssm + (1-alpha) * P_rle
    │
    └──► Order-2 literal context blending (8 hash contexts, blend=0.1)
         Final P(literal_v) = 0.9 * P_rle(v) + 0.1 * P_o2(v)
```

**Hyperparameters** (tuned via greedy sweep on Silesia corpus):

| Parameter | Value | Description |
|-----------|-------|-------------|
| D | 32 | Hidden state dimension |
| lr | 0.01 | SGD learning rate for classifiers |
| mix_decay | 0.995 | EMA decay for performance tracking |
| warmup | 0 | Steps before SSM contributes |
| mix_sensitivity | 100.0 | How fast alpha responds to performance |
| max_alpha | 0.9 | Maximum SSM mixing weight |
| decay_lo | 0.5 | Fastest EMA decay rate |
| decay_hi | 0.999 | Slowest EMA decay rate |
| o2_lit_blend | 0.30 | Order-2 literal blend weight |
| o2_min_obs | 10 | Min observations before O2 kicks in |

**Key properties**:
- Only 66 learnable parameters: 2 classifiers x (D weights + 1 bias) = 2 x (32 + 1) = 66
- Adapts from scratch on each block (no pre-trained weights)
- Adaptive mixer ensures the SSM can only help, never hurt, relative to pure RLE
- Memory: ~25 KiB total
- Typical: 2.186 bpb on BWT+RLE stream (0.016 bpb improvement over RLE alone)

### MTF Predictor (mtf_predictor.rs)

A run-length-aware predictor for raw MTF data (without RLE). Maintains order-1 context tables and run-length-bucketed context tables. Superseded by the RLE predictor pipeline.

- Memory: ~136 KiB
- Status: Legacy, kept for compatibility

---

## Range Coding

### How It Works (rans.rs)

Range coding is an arithmetic coding variant that encodes a sequence of symbols against per-symbol probability distributions.

For each byte position:
1. The predictor produces a 15-bit cumulative frequency table (CDF) via `predict_cdf()`
2. The range encoder narrows the current interval by the predicted probability of the actual byte
3. A byte predicted with 99% probability costs ~0.015 bits; a byte predicted with 0.4% costs ~8 bits

The key insight: if the predictor is good at predicting bytes, each byte costs very few bits — approaching the theoretical Shannon entropy of the source given the model.

### Implementation

Custom byte-aligned range coder (replaced `constriction` in 0.1.7):

- **Encoder**: LZMA-style carry-propagating encoder. Maintains a 64-bit `low` register and 32-bit `range`. When range drops below 2^24, shifts out a byte via a cache system that handles carry propagation.
- **Decoder**: Subtraction-based decoder. Reads 4 bytes to initialize `code`, then narrows range on each symbol decode. Uses binary search over the CDF to find the decoded symbol.
- **CDF precision**: 15 bits (PROB_TOTAL = 32768). Each symbol guaranteed minimum frequency 1. Stack-allocated `[u16; 257]` tables — no heap allocation per symbol.

```rust
// Encoding
let mut encoder = RangeEncoder::new();
for &byte in data {
    let cdf = predictor.predict_cdf();  // [u16; 257], 15-bit CDF
    encoder.encode_cdf(byte as usize, &cdf);
    predictor.update(byte);
}
let compressed = encoder.finish();  // Vec<u8>

// Decoding (identical predictor loop)
let mut decoder = RangeDecoder::new(&compressed);
for _ in 0..expected_len {
    let cdf = predictor.predict_cdf();
    let byte = decoder.decode_cdf(&cdf);
    predictor.update(byte);
}
```

The predictor is reset at the start of each block (`predictor.reset()`), ensuring blocks can be decompressed independently. The `predict_cdf()` trait method has a default implementation that converts from `predict()` via `probs_to_cdf()`, but predictors can override it to compute CDFs directly from integer counts for better performance.

---

## Adaptive Routing

Routing is profile-aware per semantic solid group:

- `archival` evaluates the full ratio-first cascade.
- `balanced` retains archival routing for text/numeric groups and uses
  BCJ/Zstandard/Store for executable, image, and binary groups.
- `fast` uses BCJ/Zstandard/Store for every group.

### The Routing Cascade (router.rs)

When the analyzer recommends `PredictorRans` for a chunk, the router tries multiple transforms and picks the smallest:

```
Step 1: BWT + MTF + RLE → NeuralSsmPredictor + range coding
        → BwtPredictorRans (if smaller than original)

Step 2: LZ77 → predictor + range coding          [SKIPPED if BWT < 55%]
        → Lz77PredictorRans (if smaller than Step 1 or original)

Step 3: Plain predictor + range coding
        → PredictorRans (if Steps 1-2 failed and this is smaller)

Step 4: Zstd (level 3)
        → Zstd (if everything else expanded)

Step 5: Store (uncompressed)
        → Store (last resort)
```

**BWT early-exit**: If the BWT path compressed to less than 55% of the original chunk size (`bwt_decisive`), Steps 2 and 3 are skipped. For structured text, BWT typically achieves 25–35% ratio, so the LZ77 encode + range coding pass is avoided entirely on text content.

The BWT path uses its own internal `NeuralSsmPredictor` regardless of which predictor the user selected via CLI, because the NeuralSSM is specifically designed for the RUNA/RUNB stream that BWT+MTF+RLE produces.

### Predictor Sync

Predictor-coded blocks reset at block boundaries. Under the `threading` feature,
chunks within a solid group therefore use independent predictors and are
compressed in parallel; ordered collection preserves deterministic block
layout. Dictionary baselines are installed on every per-chunk predictor.

**Sync skip**: When BWT wins decisively (`bwt_decisive`), the sync step is omitted. Subsequent chunks of the same content type will also use BWT (with its own internal predictor), so the group predictor's cross-block state will not be consumed by any LZ77 or plain-RC path block. This avoids an O(n) per-byte predict+update pass through the group predictor for each chunk — a significant saving for expensive predictors like NeuralSsm or ContextMixer.

### When Routing Skips PredictorRans

High-entropy data (>7.5 bpb) goes directly to Zstd. Near-random data (>7.95 bpb) is stored uncompressed. These thresholds are in `analyzer.rs`.

---

## Module Reference

### Core Types

| File | Lines | Description |
|------|-------|-------------|
| `error.rs` | 63 | `AetherError` enum with 16 variants |
| `format.rs` | 228 | Magic bytes, size constants, `CompressionMethod`, `PredictorId`, `ContentType`, `shannon_entropy()` |
| `header.rs` | 425 | `ArchiveHeader`, `FileEntry`, `SolidGroupEntry`, `ArchiveFooter` with read/write |
| `block.rs` | 250 | `BlockHeader`, `BlockTrailer`, `BlockIndexEntry` with read/write |

### Analysis & Grouping

| File | Lines | Description |
|------|-------|-------------|
| `chunker.rs` | 190 | FastCDC (16-512-8192 KiB), owned public chunks + zero-copy internal views |
| `analyzer.rs` | 267 | Content-type detection (ELF, PE, JPEG, PNG, PDF, ZIP, etc.), entropy routing |
| `grouper.rs` | 176 | Semantic solid grouping by content type |

### Entropy Predictors

| File | Lines | Description |
|------|-------|-------------|
| `entropy/traits.rs` | 57 | `ProbabilityPredictor` trait (with `predict_cdf()`) |
| `entropy/order0.rs` | 164 | Order-0 frequency model |
| `entropy/context_mixer.rs` | 448 | Multi-order logistic context mixer |
| `entropy/lz4_aware.rs` | 783 | FSM-based LZ4 stream predictor |
| `entropy/rle_predictor.rs` | 249 | Hierarchical RLE stream predictor |
| `entropy/neural_ssm.rs` | 759 | Diagonal SSM + RLE hybrid with adaptive mixer |
| `entropy/mtf_predictor.rs` | 276 | Run-length-aware MTF predictor (legacy) |

### Coding & Preprocessing

| File | Lines | Description |
|------|-------|-------------|
| `coding/rans.rs` | ~350 | Custom byte-aligned range coder (15-bit CDF) |
| `coding/zstd_fallback.rs` | 63 | Zstd compress/decompress wrapper |
| `coding/bwt_preprocess.rs` | 506 | BWT + MTF + RUNA/RUNB RLE |
| `coding/lz77_preprocess.rs` | 450 | Custom LZ77 with min-match-3 |
| `coding/lz_preprocess.rs` | 150 | LZ4 via lz4_flex |

### Pipeline

| File | Lines | Description |
|------|-------|-------------|
| `pipeline/router.rs` | 267 | Adaptive chunk routing (try BWT/LZ77/RC/Zstd/Store) |
| `pipeline/compress.rs` | 299 | Full compression orchestration |
| `pipeline/decompress.rs` | 311 | Decompression, extraction, verification |

### CLI

| File | Lines | Description |
|------|-------|-------------|
| `aether-cli/src/main.rs` | ~435 | clap-based CLI: compress/extract/list/verify/bench |

---

Copyright 2024-2026 Craton Software Company Licensed under Apache-2.0.
