# AetherArch Predictors and Compression Methods

## Overview

AetherArch uses a three-level compression strategy:

1. **Preprocessing** - BWT+MTF+RLE (primary path for text), LZ77, or LZ4 removes redundancy from raw data.
2. **Predictor** - A probabilistic model that predicts the next byte of the preprocessed stream. The predictor produces a probability distribution over all 256 byte values.
3. **Entropy coder** - A custom byte-aligned range coder (15-bit CDF precision) that encodes the actual byte using the predicted distribution. Better predictions = smaller output.

The **adaptive router** inspects each chunk's Shannon entropy and selects the best compression cascade per-block: LZ4+Predictor+RC, plain Predictor+RC, Zstd, or Store.

---

## Predictors

### `order0` - Order-0 Frequency Model

| Property | Value |
|----------|-------|
| CLI name | `order0`, `o0` |
| ID | `0x0000` |
| Memory | ~1 KiB |
| Speed | Fast |
| Compression | Moderate |

A simple byte frequency counter. Maintains a 256-element frequency table and predicts each byte proportional to how often it has appeared so far. Uses Laplace smoothing (add-1) and periodic rescaling to prevent overflow.

**Best for**: Quick compression where speed matters more than ratio. Baseline model.

**Typical performance**: ~4.5-5.0 bits/byte on English text.

---

### `cm` - Context Mixer

| Property | Value |
|----------|-------|
| CLI name | `cm`, `context-mixer` |
| ID | `0x0001` |
| Memory | ~100 MiB |
| Speed | Slow |
| Compression | Best |

PAQ-inspired multi-order context mixing. Combines predictions from 6 independent order-N models (order 1 through 6) using logistic mixing in log-odds space. Mixing weights are adapted online via gradient descent, giving more weight to models that predict well for the current data.

Each `OrderNModel` uses a hash table mapping `hash(last N bytes)` to a 256-element frequency array, with FNV-1a hashing and Laplace smoothing.

**Configuration (default):**

| Order | Table bits | Entries | Memory |
|-------|-----------|---------|--------|
| 1 | 14 | 16K | ~8 MiB |
| 2 | 16 | 64K | ~32 MiB |
| 3 | 16 | 64K | ~32 MiB |
| 4 | 15 | 32K | ~16 MiB |
| 5 | 14 | 16K | ~8 MiB |
| 6 | 13 | 8K | ~4 MiB |

Learning rate: 0.002, Max context: 8 bytes.

**Best for**: Maximum compression on structured data (text, source code, JSON, XML).

**Typical performance**: ~4.0-4.2 bits/byte on English text.

---

### `cm-light` - Lightweight Context Mixer

| Property | Value |
|----------|-------|
| CLI name | `cm-light` |
| ID | `0x0003` |
| Memory | ~4 MiB |
| Speed | Medium |
| Compression | Good |

A stripped-down context mixer with smaller tables and fewer orders. Uses only order 1-3 models. Good balance between compression ratio and resource usage.

**Configuration:**

| Order | Table bits | Entries |
|-------|-----------|---------|
| 1 | 12 | 4K |
| 2 | 12 | 4K |
| 3 | 11 | 2K |

Learning rate: 0.005, Max context: 4 bytes.

**Best for**: Moderate compression with lower memory usage. Good default for systems with limited RAM.

---

### `lz4-aware` - LZ4-Aware FSM Predictor

| Property | Value |
|----------|-------|
| CLI name | `lz4`, `lz4-aware` |
| ID | `0x0004` |
| Memory | ~8.2 MiB |
| Speed | Medium |
| Compression | Best (with LZ4 preprocessing) |

A specialized predictor that understands the internal structure of LZ4 byte streams. Maintains a finite state machine (FSM) tracking position within the LZ4 format and dispatches to specialized sub-predictors per byte role.

**FSM States:**

```
SizePrefix(0..3) → Token → [LitLenExt] → Literals(n) → MatchOffsetLow → MatchOffsetHigh → [MatchLenExt] → Token → ...
```

**Sub-predictors per state:**

| State | Model | Memory | Rationale |
|-------|-------|--------|-----------|
| SizePrefix | 4× Order-0 (per position) | 4 KiB | Size bytes are position-dependent |
| Token | Order-1 keyed on previous token | 128 KiB | Token patterns repeat |
| LitLenExt | Order-0 | 1 KiB | 255 dominates until final byte |
| Literals | Order-1 + Order-2 hash tables, linear blend | ~8 MiB | Core workhorse — models literal-only context |
| MatchOffsetLow | Order-0 | 1 KiB | Small offsets common |
| MatchOffsetHigh | Order-0 | 1 KiB | Often 0x00 |
| MatchLenExt | Order-0 | 1 KiB | Similar to LitLenExt |

**Key insight**: The Literals sub-predictor maintains its own context buffer of *only literal bytes* (not interleaved LZ4 control bytes). This gives it clean order-1/order-2 context on the actual text data, rather than context polluted by control bytes.

**Literal blend**: 40% order-1 + 60% order-2 when both have context, fallback to order-1 or uniform.

**Best for**: Maximum compression when used with LZ4 preprocessing. Designed specifically for the `LzPredictorRans` compression path.

**Typical performance**: ~3.2 bits/byte on English text (with LZ4 preprocessing).

---

### `ssm` - Neural SSM Predictor (Default for BWT path)

| Property | Value |
|----------|-------|
| CLI name | `ssm`, `neural-ssm` |
| ID | `0x0002` |
| Memory | ~25 KiB |
| Speed | Medium |
| Compression | Best (on BWT+RLE streams) |

A hybrid predictor combining a diagonal linear State Space Model with the RlePredictor baseline and an order-2 literal context model. This is the most sophisticated predictor and produces the best compression ratios on structured data.

**Architecture:**

```
Input byte
    │
    ├─ Embedding: byte → 32-dimensional vector
    ├─ SSM update: h[d] = a[d]*h[d] + (1-a[d])*embed[byte][d]
    │  (32 exponential moving averages at timescales 0.5 … 0.999)
    ├─ Binary classifier 1: sigmoid(w_run · h) → P(run symbol)
    ├─ Binary classifier 2: sigmoid(w_runa · h) → P(RUNA vs RUNB)
    ├─ RlePredictor baseline (3-context hierarchical model)
    ├─ Adaptive mixer: weight SSM vs RLE by recent log-likelihood
    └─ Order-2 literal context blend (30% weight, 8 hash buckets)
```

**Key properties:**
- 66 learnable parameters: 2 classifiers x (D weights + 1 bias) = 2 x (32 + 1)
- Adapts from scratch on each block (no pre-trained weights)
- Adaptive mixer ensures SSM can only help, never hurt, relative to pure RLE
- Hyperparameters tuned via greedy sweep on Silesia corpus: D=32, lr=0.01, o2_blend=0.30

**Best for**: BWT+MTF+RLE streams (used automatically by the BWT routing path).

**Typical performance**: ~2.186 bpb on BWT+RLE stream (vs 2.202 for pure RLE, 2.223 for Order-0).

---

### `rle` - RLE Stream Predictor

| Property | Value |
|----------|-------|
| CLI name | `rle` |
| ID | `0x0005` |
| Memory | ~3 KiB |
| Speed | Fast |
| Compression | Good (on BWT+RLE streams) |

Hierarchical predictor designed specifically for the RUNA/RUNB RLE stream produced by BWT+MTF+RLE. Uses three context classes (start, in-run, after-literal) with specialized binary/counting models for each.

**Best for**: Baseline predictor for RLE streams. Used as the foundation inside NeuralSsmPredictor.

**Typical performance**: ~2.202 bpb on BWT+RLE stream.

---

## Per-Block Compression Methods

Each block in an archive is independently compressed using one of these methods, chosen automatically by the adaptive router:

### `BwtPredictorRans` - BWT + MTF + RLE + Predictor + Range Coding (Primary)

The primary compression path for structured data. Input is transformed by the Burrows-Wheeler Transform to cluster similar contexts, then Move-to-Front encoding converts to small integers, and bijective RUNA/RUNB run-length encoding compacts zero runs. The resulting stream is modeled by NeuralSsmPredictor and compressed via range coding.

**Payload format**: `[flags: u8] [primary_index: u32 LE] [encoded_len: u32 LE] [range-coded data]`

This path always uses NeuralSsmPredictor internally, regardless of the user-selected predictor.

### `Lz77PredictorRans` - LZ77 + Predictor + Range Coding

Custom LZ77 with min-match-3, 64 KB window, lazy matching. Effective for structured source code and data with short repeated patterns that BWT may miss. Skipped when BWT achieves < 55% ratio.

### `LzPredictorRans` - LZ4 + Predictor + Range Coding

Data is first transformed by LZ4 to remove repeated substrings, then the resulting LZ4 byte stream is fed through the predictor + range coder.

**Payload format**: `[lz_len: u32 LE] [range-coded LZ4 bytes]`

### `PredictorRans` - Predictor + Range Coding (Fallback)

Used when preprocessing transforms (BWT, LZ77, LZ4) don't reduce data size but the predictor can still model the byte distribution effectively.

### `Zstd` - Zstandard Fallback

Used when chunk entropy is between 7.5 and 7.95 bits/byte, or when all predictor-based paths expanded the data. Falls back to Zstandard (level 3).

### `Store` - Raw Storage

Used when chunk entropy exceeds 7.95 bits/byte (incompressible data). Applied to already-compressed data (JPEG, PNG, ZIP, encrypted data).

---

## Adaptive Routing Cascade

The router tries multiple paths per chunk and picks the smallest result:

```
Shannon Entropy of chunk
    |
    |  <= 7.5 bpb    -->  Adaptive routing cascade:
    |                      |
    |                      Step 1: BWT + MTF + RLE → NeuralSsm + RC  (BwtPredictorRans)
    |                      Step 2: LZ77 → Predictor + RC             (Lz77PredictorRans)
    |                              [SKIPPED if BWT achieved < 55% ratio]
    |                      Step 3: Plain Predictor + RC              (PredictorRans)
    |                      Step 4: Zstd level 3                      (Zstd)
    |                      Step 5: Store                             (Store)
    |                      → Pick smallest output
    |
    |  <= 7.95 bpb   -->  Zstd (zstandard fallback)
    |
    |  > 7.95 bpb    -->  Store (raw, no compression)
```

---

## Content-Type Detection and Solid Grouping

Before compression, files are analyzed and grouped by content type:

| Content Type | Detection |
|-------------|-----------|
| Text | ASCII/UTF-8 content, extensions: .txt, .md, .rs, .py, .js, .json, .xml, .html, .css, .toml, .yaml, .csv |
| Image | Magic bytes for JPEG (FFD8FF), PNG (89504E47) |
| Executable | Magic bytes for ELF (7F454C46), PE (MZ) |
| BinaryStructured | Non-text binary with patterns |
| BinaryRandom | High-entropy binary |
| Mixed | Everything else |

Files of the same content type are placed in **solid groups** that share a single predictor instance. This lets the predictor learn cross-file patterns within a group (e.g., common keywords across all `.rs` files).

---

## Integrity Verification

Every archive includes a multi-level integrity chain:

| Level | Algorithm | Protects |
|-------|-----------|----------|
| Archive Header | CRC-32 | Header metadata (48 bytes) |
| Block Header | CRC-32 | Per-block metadata (28 bytes) |
| Block Content | BLAKE3 | Decompressed block data |
| Block Trailer | CRC-32 | BLAKE3 hash field (36 bytes) |
| File | BLAKE3 | Reassembled file content |
| Archive Footer | CRC-32 | Footer metadata (32 bytes) |

Run `aet verify <archive>` to check all integrity levels.

---

## Benchmark Results

On an 87.1 KiB structured text corpus (English prose, Rust source code, JSON):

### Current Results (v0.2.3, BWT+MTF+RLE routing)

| Predictor | Path | Ratio | Bits/byte | Speed |
|-----------|------|-------|-----------|-------|
| `ssm` (NeuralSSM) | BWT+MTF+RLE | 27.37% | 2.190 | ~1.1 MiB/s |
| `rle` (RlePredictor) | BWT+MTF+RLE | 27.52% | 2.202 | ~1.2 MiB/s |
| `order0` | BWT+MTF+RLE | 27.79% | 2.223 | ~1.3 MiB/s |

### External Tool Comparison (v0.2.3)

| Tool | Ratio | Bits/byte |
|------|-------|-----------|
| AetherArch (ssm) | 27.37% | 2.190 |
| bzip2 -9 | 27.40% | 2.192 |
| xz -9 | 27.12% | 2.169 |
| gzip -9 | 29.14% | 2.331 |

### Legacy Results (v0.1.2, LZ4 preprocessing)

| Predictor | Ratio | Bits/byte | Speed |
|-----------|-------|-----------|-------|
| `lz4-aware` | 39.99% | 3.199 | ~0.8 MiB/s |
| `order0` | 40.23% | 3.218 | ~1.2 MiB/s |
| `cm-light` | 42.99% | 3.439 | ~0.1 MiB/s |

The BWT+MTF+RLE pipeline (introduced in v0.1.8) replaced LZ4 preprocessing as the primary compression path, improving compression by ~12 percentage points. The NeuralSSM predictor's SSM long-range context provides the best ratio on BWT+RLE streams.

---

Copyright 2024-2026 Craton Software Company Licensed under Apache-2.0.
