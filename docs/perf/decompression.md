# Decompression Speed — Analysis and Changes

**Status**: Landed in the decompression-speed branch (post-0.3.0).
**Scope**: the read path — `aet extract`, `aet verify`, `Decompressor::*`,
`router::decompress_chunk`, the range decoder, and the inverse transforms.

This is the decompression counterpart to the compression optimization work
retired in [`docs/internal/retired/deep-research-aether-opt.md`][retired].
That effort targeted the write path; the numbers it left behind said
decompression was roughly as slow as compression, and nothing in it had
looked at why.

[retired]: ../internal/retired/deep-research-aether-opt.md

---

## Summary of findings

Two of the four largest wins were not optimizations at all.

1. **Two decode paths were broken.** Archives the compressor accepted could
   not be extracted — a byte-plane block or an LZ77 block written by
   0.3.0 could fail with "range decoder read N bytes past end of input".
   Nothing measurable can be said about the speed of a path that does not
   produce the right answer, so this came first.

2. **The decoder ran a full predictor pass over data it had already
   decoded.** For the `Zstd` and `Store` blocks an archival or balanced
   archive is made of where the data is high-entropy, this was the entire
   cost of decompression — the actual codec was a rounding error next to
   it.

3. **The per-symbol CDF was built in full and read twice.** The range decoder
   needs one interval; `predict_cdf` computed all 257 boundaries.

4. **Parallel decompression existed but did not scale**, and was gated behind
   the `enterprise` feature, so released binaries never used it.

---

## Where the time actually goes

Measured with `cargo bench -p aether-core --bench decompression`, which
isolates the decode loop, each compression method, each inverse transform,
and the whole-archive paths. Numbers below are from the branch tip; see
"Measurement" for why absolute figures on a shared machine are only
comparable within a run.

| Stage | Throughput | Comment |
|---|---:|---|
| Range decode + NeuralSSM (`decode/ssm_bwt_stream`) | ~0.4 MiB/s of coded symbols | Dominates every predictor-backed block |
| Range decode + Order0 | ~0.2 MiB/s of coded symbols | Backs every byte-plane block |
| Inverse BWT + MTF, 128 KiB | 33 MiB/s | |
| Inverse BWT + MTF, 512 KiB | 25 MiB/s | |
| Inverse BWT + MTF, 2 MiB | 12 MiB/s | ~25% of a 2 MiB BWT block |
| Byte-plane decode, 512 KiB | 0.26 MiB/s sequential, 0.68 MiB/s over 4 planes | Order0 rANS per plane |
| RLE decode | ~840 MiB/s | Negligible |
| LZ77 decode | ~1.0 GiB/s | Negligible |
| Zstd decode | ~380 MiB/s | Negligible |
| Store | ~6 GiB/s | A memcpy |
| Archive metadata parse | ~2 us | Negligible until file counts get large |

The shape is stark: **entropy decoding is decompression.** Everything else
put together is under a tenth of it on the archival profile. That is the
same conclusion the compression work reached about its own hot loop, and it
sets the priority order for everything below.

---

## The correctness gates

### `Order0Model::query_cdf` did not reproduce `predict_cdf`

The encoder commits to intervals from `query_cdf`; the decoder resolves
symbols from `predict_cdf`. They must agree bit for bit.

`predict_cdf` has three stages: cumulative integer rounding, a forward
monotonicity fix-up, and — if that fix-up pushes `cdf[256]` past
`PROB_TOTAL` — a complete rebuild via `probs_to_cdf`. The encode-side fast
path added in 0.3.0 reproduced the first two but treated the third as
unreachable, checking only the `s == 255` anchor.

It is not unreachable. Overshoot needs a rounded gap to collapse to zero,
which needs `total > PROB_TOTAL` and a skewed distribution — exactly the
float-exponent planes that `byteplane_encode` range-codes with Order0. On
those, encoder and decoder disagreed on every symbol.

The fix splits into two regimes. When `total <= PROB_TOTAL` every symbol
holds at least one count, so no gap can collapse, no fix-up fires and
overshoot is impossible — the partial sweep is exact and stops at
`byte + 1`. Otherwise the full forward sweep is replayed without
materialising the table, delegating to `predict_cdf` when it would have
overshot.

**Any archive written by 0.3.0 containing a byte-plane block may be
unreadable.** The fix repairs the encoder; it cannot repair bytes already
written.

### The router decoded predictor payloads with the wrong predictor

`compress_chunk` range-codes every predictor-backed payload — BWT, LZ77,
plain, byte-plane — with a **scratch `NeuralSsmPredictor`** it owns, not
with the caller's group predictor. It has done so since the first release.

`decompress_chunk` decoded the BWT path with a matching scratch predictor
but decoded `PredictorRans`, `Lz77PredictorRans` and `LzPredictorRans` with
the *group* predictor. When the group predictor was also NeuralSSM the two
happened to coincide. When it was anything else — including `cm`, the CLI's
default — they did not, and the block failed to decode.

Both sides now build the same predictor through one `coding_predictor()`
helper.

---

## The optimizations

### 1. Delete the predictor sync pass

Both sides used to run a full `predict`/`update` over every chunk's
plaintext, to keep group predictor state "in sync". Nothing consumed that
state: every payload is coded by a scratch predictor that `encode_block` /
`decode_block` reset before the first symbol. The parallel compression path
already depended on this — it builds a fresh predictor per chunk and still
emits byte-identical archives.

On the decode side the pass cost one NeuralSSM step per *output* byte. It
ran whenever the block header's `predictor_state_flag` said the encoder had
synced — that is, for `Zstd` and `Store` blocks chosen by the analyzer or as
the routing cascade's fallback, and for `Lz77PredictorRans` and
`PredictorRans` wins. (BWT, byte-plane, BCJ and the whole fast profile
already skipped it.) For an incompressible block the pass *was* the
decompression:

| Method | Before | After |
|---|---:|---:|
| `Store`, 256 KiB | 423 ms | 39.5 us |

`CompressedChunk::predictor_synced` and the block header's
`predictor_state_flag` are still written and read, so the format is
unchanged and a predictor with genuine cross-block state could reintroduce
the pass without a format break.

### 2. Fuse the range decoder with the predictor

`predict_cdf` spends one `quantized_boundary` — an `f64` multiply, divide
and floor — on each of 256 boundaries, writes a 257-entry table, returns it
by value, and the decoder reads two entries of it.

`RangeDecoder::decode_cdf` is now split into `decode_freq` + `advance`, and
`ProbabilityPredictor::decode_symbol(freq)` resolves the symbol itself. The
default implementation is the old `predict_cdf` + `find_symbol`, so every
predictor keeps working.

`NeuralSsmPredictor` overrides it. Its boundaries are non-decreasing in the
symbol index, so a binary search runs directly over them and evaluates about
nine — not 256.

This is the step [`per-symbol-cdf.md`](per-symbol-cdf.md) predicted would
lose. That analysis assumed the decode-side search would call `query_cdf`
nine times, each re-running the O(256) model pass. The model pass runs
**once**; only the quantisation is searched. Measured **1.87x** on the
decode loop, in-process against the old path.

### 3. Stop copying the RLE baseline per symbol

`model_weights` called `RlePredictor::predict()`, which built a
`[f32; 256]`, returned it by value, and cached it into a `last_rle_probs`
field nothing read — over a kilobyte of memory traffic per predicted byte,
all of it folded into `weights` immediately. The baseline is now consumed
through `RlePredictor::model_parts()` in decomposed form, with the
arithmetic sequence preserved exactly.

### 4. Divide-free CDF construction for Order0

Up to 256 hardware divides per symbol became a widening multiply and a
shift, with the magic number chosen so the result is provably exact over
the whole operand range. **2.25x** on Order0 CDF construction.

A `decode_symbol` override for Order0 was implemented and then **removed**:
0.90x on a BWT stream, 0.52x on a skewed byte plane. Its fix-up chains
forward, so the boundaries cannot be binary-searched, and its overshoot
fallback is only detectable at the end of the sweep — when it fires, a fused
loop does the work twice. The reasoning is recorded at the call site.

### 5. Decode from the baseline, not from a predictor

`decompress_chunk` reads exactly one thing off the predictor it is handed:
the dictionary coding baseline. Passing the baseline directly
(`decompress_chunk_with_baseline`) removes predictor construction from the
decode path entirely — a `ContextMixer` is ~100 MiB per instance, and one
was built per solid group. With no dictionary configured, nothing is built
at all.

It also removes a resource cap rather than enforcing one: the
`MAX_SOLID_GROUP_COUNT` ceiling on the streaming and verify predictor maps
existed to stop a crafted archive with a unique `solid_group_id` per block
from exhausting memory. Nothing is allocated per group any more.

### 6. Parallelise across blocks, not solid groups

Blocks are independently decodable: a block's output is a function of its
payload, its method, its size and the dictionary baseline, and of nothing
else. The parallel path used to make one rayon task per solid group, so an
archive with one large group and several small ones parallelised no better
than sequential.

One task per block lets rayon work-steal, which matters because block sizes
within an archive span two orders of magnitude. Measured on a 57-file
archive: 3.93s at one thread, 1.91s at eight.

Parallel decompression moved from the `enterprise` gate to `threading`, and
`aether-cli` enables `threading` by default — released binaries should use
the machine they run on. Output is byte-identical at every thread count.

### 7. Two-chain inverse BWT

The LF walk misses cache on almost every output byte and each jump depends
on the last, so the CPU cannot run ahead. Two changes, picked by block size:

* Packing `lf` and the byte into one `u32` halves the misses.
* Above 1 MiB, walking `LF` backwards and its inverse `Q` forwards at the
  same time puts two independent misses in flight. `Q` walked from
  `primary_index` emits the output forwards exactly as `LF` emits it
  backwards, because the permutation is a single n-cycle.

| Block | Two arrays | Packed | Two chains |
|---:|---:|---:|---:|
| 64 KiB | 60 MiB/s | 84 MiB/s | 69 MiB/s |
| 512 KiB | 24 MiB/s | 27 MiB/s | 31 MiB/s |
| 2048 KiB | 10 MiB/s | 11 MiB/s | 16 MiB/s |

### 8. Parallelise the two things left holding a whole core

**Byte-plane planes.** A byte-plane block is two or four independent Order0
streams, and that decode is by far the most expensive part of the method —
the planes are the only parallelism it has. Decoding them concurrently under
`threading` measures **2.6x** (1.99 s to 752 ms on a 512 KiB float array).
Nested inside the per-block parallelism, rayon's work-stealing absorbs it.

**Verification.** `verify` decodes every block, so it scales exactly like
extraction — but it must not stop at the first failure, since finding bad
blocks is the point and one corrupt block must not hide the state of
everything after it. Both the read and the decode therefore record failures
per block and continue. Measured through the CLI: a 57-file source tree goes
2879 ms to 912 ms (**3.16x**), a 2 MiB float corpus 14.1 s to 5.3 s
(**2.67x**).

`verify_reports_the_same_corruption_at_every_thread_count` corrupts two block
payloads and checks that every thread count reports exactly the same two.

### 9. The decompress-only build

`aether-core --no-default-features` — and therefore `aether-wasm`, which is
decompress-only by design — did not compile. `bwt-encode` gates the
libsais-backed forward transform, but the router's BWT trial and the
transformed dictionary trainer called it ungated, so the crate only built when
another workspace member turned the feature back on through Cargo's feature
unification. `cargo build --workspace` hid it; `cargo build -p aether-wasm`
did not.

Decoding `BwtPredictorRans` blocks was never gated, so the fix is confined to
the encode side: the router's BWT trial is skipped and the cascade falls
through, and `Dictionary::train_transformed` returns an explanatory error
rather than not existing.

### 10. Reassembly and I/O

* Extraction takes each block out of the array as it consumes it, instead of
  holding every decompressed block until the last file is written. A
  single-block file hands its buffer straight through with no copy.
* The trailing `combined[..end].to_vec()` — a second full copy of every
  extracted file — is now an in-place truncate.
* The CLI wraps the archive in a `BufReader`. The seekable path reads the
  file table, group table and block index one small struct at a time, which
  was a syscall per entry on an unbuffered handle.
* `aet extract` reports throughput over the bytes it produced, not the bytes
  it read. An archive-size rate makes a decompressor look faster the better
  the compressor did.

---

## End-to-end results

`main` (0.3.0) against the branch tip, same benchmark binary source, run
**alternately** so the host's background load lands on both arms. Two rounds;
both are shown because the spread between them is the machine, not the code —
which is exactly why the per-round ratio is the number to read.

A second confirmation session on a busier host reproduced the same shape at
the low end of the range: `archive/extract_all` 1.98x / 2.21x,
`archive/verify` 1.12x / 2.82x, `decode/ssm_bwt_stream` 1.76x / 2.41x. Take
**2-3x** as the honest interval for whole-archive extraction rather than any
single figure; the 1.12x `verify` round had base and new at 658 ms and 586 ms
while the very next round had 912 ms and 323 ms, which is the host talking.

| Benchmark | main | branch | speedup |
|---|---:|---:|---:|
| `archive/extract_all` | 1128 ms / 799 ms | 376 ms / 307 ms | **3.00x / 2.61x** |
| `archive/verify` | 1003 ms / 697 ms | 327 ms / 263 ms | **3.07x / 2.65x** |
| `decode/ssm_bwt_stream` | 408 ms / 247 ms | 220 ms / 126 ms | 1.86x / 1.96x |
| `method/text_archival_BwtPredictorRans` | 195 ms / 335 ms | 134 ms / 112 ms | 1.46x / 3.00x |
| `method/floats_BwtPredictorRans` | 1085 ms / 1083 ms | 721 ms / 595 ms | 1.51x / 1.82x |
| `method/random_Store` | 423 ms / 391 ms | 39.5 us / 41.8 us | **~10000x** |
| `transform/bwt_mtf_decode` (512 KiB) | 24.5 ms / 30.4 ms | 23.1 ms / 18.7 ms | 1.06x / 1.62x |
| `method/text_fast_Zstd` | 670 us / 467 us | 561 us / 499 us | 1.19x / 0.94x |
| `transform/rle_decode` | 535 us / 579 us | 528 us / 556 us | 1.01x / 1.04x |
| `transform/lz77_decode` | 413 us / 390 us | 377 us / 429 us | 1.10x / 0.91x |
| `archive/read_metadata` | 2.7 us / 2.2 us | 3.0 us / 2.2 us | 0.89x / 1.03x |

Reading the table:

- **Whole-archive extraction is 2.6–3.0x faster, single-threaded.** The
  benchmark builds `aether-core` with default features, so `threading` is off
  in both arms — thread scaling is on top of this, not part of it.
- The stages that were already fast (Zstd, RLE, LZ77, metadata) are unchanged;
  their columns are noise around 1.0x, which is the control this table needs
  to be trustworthy.
- `method/random_Store` is the sync pass. 423 ms to memcpy 256 KiB was not a
  codec cost.

### End to end through the CLI

`aet extract` wall-clock, minimum of three alternating runs on an 8-core host.
Both binaries were given byte-identical archives (verified with `cmp`).

| Corpus | main | branch, `-t 1` | branch, `-t 0` |
|---|---:|---:|---:|
| 2.63 MiB text, 3 files | 1043 ms | 725 ms (1.44x) | 544 ms (**1.92x**) |
| 932 KiB source tree, 57 files | 2217 ms | 1616 ms (1.37x) | 703 ms (**3.15x**) |

These are smaller than the `archive/*` figures above, for two reasons worth
being explicit about:

- Both corpora are pure text, so every block takes the BWT path. None of them
  pay the sync pass that made `method/random_Store` a four-order-of-magnitude
  result — an archive containing incompressible data gains far more.
- CLI timings include process start, header parsing, and writing the extracted
  files, none of which got faster.

The 57-file archive is where thread scaling shows: 57 blocks give rayon
something to steal. The 3-file archive has three, one of which is 2 MiB, so it
is bounded by that single block.

---

## Measurement

The development host is shared and frequently at 100% CPU from unrelated
processes. Wall-clock measurements on it vary by a factor of two to four
between runs, which is larger than most of the effects being measured.

Three rules were used, and are worth keeping:

1. **Compare inside one process.** Where both the old and new code paths can
   be compiled together, run them alternately in one binary and take the
   **minimum** of several repetitions. Background load then affects both
   arms about equally, and the minimum is the run least polluted by it. Every
   per-stage figure quoted above was obtained this way.
2. **Never compare across separate benchmark runs.** Criterion's
   `--save-baseline` is only trustworthy if both arms ran under the same
   machine conditions; on this host they routinely did not.
3. **Report throughput over decompressed bytes.** The archive-size rate is
   not a decompression speed.

`scripts/decompression-matrix.ps1` applies the same rules at the CLI level:
compression is untimed setup, extraction is repeated and reported as minimum
and median, thread counts are swept, and `verify` and single-file extraction
are measured alongside full extraction.

---

## What was considered and not done

* **Order0 `decode_symbol`.** Implemented, measured slower, removed. See
  above.
* **A quantiser that cannot overshoot.** `NeuralSsmPredictor` reserves one
  count per symbol, so its boundaries are monotone by construction and it
  never needs a fallback. Order0 does not, and switching it would change
  every Order0 archive's bitstream. It is the natural thing to do at a
  format break, and would remove both the fallback cost and the class of bug
  that the `query_cdf` desync belonged to.
* **Streaming extraction with bounded memory.** `extract_all` still decodes
  every block before writing any file, so peak memory is proportional to the
  uncompressed archive. Reassembly now releases blocks as it goes, but the
  decode phase does not. Bounding it properly means interleaving decode and
  write per file, which the block index already makes possible.
* **More than two inverse-BWT chains.** `LF` and `Q` give two entry points
  into the cycle for free. A third would need a position in the middle of
  the cycle, and finding one costs the walk you were trying to avoid.
* **Parallelism inside a block.** The adaptive predictor makes symbol
  decoding strictly sequential within a block. Parallelism is bounded by
  block count, which for text can be low: FastCDC produced a *single* 2 MiB
  chunk for a 2 MiB English text file. Chunking policy is a compression-side
  decision with ratio consequences, so it was left alone, but it is the
  ceiling on how much threads can help small archives.
