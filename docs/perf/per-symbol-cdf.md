# Per-Symbol CDF API — Design & Extension Plan

**Status**: Prototype landed (Order0 only). Encode-asymmetric.
**Target**: 2–3× speedup on the predictor (currently 92% of compress wall-clock).

## Motivation

The hot loop in `aether-core/src/coding/rans.rs::encode_block_inner` is:

```rust
for &byte in data {
    let cdf = predictor.predict_cdf();   // builds & returns [u16; 257]
    enc.encode_cdf(byte, &cdf);          // reads cdf[byte], cdf[byte+1]
    predictor.update(byte);
}
```

For every byte, the predictor materialises a 257-entry CDF, returns it by
value (514-byte stack copy), and the encoder reads exactly two entries. The
other 254 entries are wasted. With `predict_cdf` being the bulk of compress
time, this is the long-pole refactor.

## What changed in this prototype

### 1. New trait method (encode-only)

`ProbabilityPredictor::query_cdf(&mut self, byte: u8) -> (u16, u16)`
returns `(cdf[byte], cdf[byte+1])`. Default impl calls `predict_cdf` and
slices — every existing predictor keeps working unchanged.

Files: `aether-core/src/entropy/traits.rs` (~line 50).

### 2. Encoder uses it

`encode_block_inner` now calls `query_cdf(byte)`, plus a new
`RangeEncoder::encode_interval(cdf_lo, cdf_hi)` that takes just the two
entries (the old `encode_cdf(byte, &cdf)` is retained and now delegates).

Files: `aether-core/src/coding/rans.rs` (`encode_interval`,
`encode_block_inner`).

### 3. Order0 migrated to a real fast path

`Order0Model::query_cdf` replaces the O(256) full-CDF build with an O(byte)
forward sweep that simulates the same cumulative-rounding + monotonicity
fix-up but only up to index `byte + 1`. The upper 254 entries are never
computed, and the function returns 4 bytes instead of 514.

The trade-off vs a pure O(log 256) Fenwick lookup: the original
`predict_cdf` runs a forward fix-up (`cdf[i+1] = max(cdf[i+1], cdf[i]+1)`)
that can chain. A pure prefix-sum + scale jump is **not bit-identical to
the decoder's `predict_cdf`** and would silently desync. Replaying the
sweep up to `byte+1` is the smallest correct shortcut.

Files: `aether-core/src/entropy/order0.rs` (`query_cdf`).

## Why decode is left alone

Decode is genuinely different: it has a codeword in `[0, PROB_TOTAL)` and
must find the symbol whose CDF interval contains it. With access only to
`query_cdf(byte) -> (lo, hi)` per byte, the decoder would need either:

* **Linear scan** — call `query_cdf(b)` for `b = 0..256` until the interval
  matches. 256× slowdown vs the current full-CDF binary search.
* **Binary search** — call `query_cdf(mid)` and pick the half containing
  the codeword. 8 queries per byte. Each `query_cdf` is roughly half the
  cost of `predict_cdf`, so this is ~4× slower than the current single
  `predict_cdf` + in-array binary search.

Neither wins. The full-CDF cost in `predict_cdf` is fundamental for decode.
The win is **encode-asymmetric**.

A future change *could* convert `predict_cdf` itself to return a
"CDF handle" (e.g. `[u16; 257]` lazily cached in the predictor, returned
by reference), avoiding the 514-byte stack return for decode too. That's
a separate refactor not attempted here.

## Extension plan (ranked by expected impact)

| Rank | Predictor      | Fast path | Expected encode win |
|------|----------------|-----------|---------------------|
| 1    | NeuralSSM      | per-symbol probability + cached scale | ~2–3× (it's 92% of compress) |
| 2    | RLE            | binary-decision short-circuit | ~1.3–1.5× |
| 3    | ContextMixer   | per-symbol mixed-prob lookup | ~1.2–1.5× |
| 4    | Lz4Aware       | Order0-style cumulative skip | ~1.2× |
| 5    | Mtf            | already cheap; modest 1.05–1.1× |

**TL;DR of extension plan**: NeuralSSM is the prize — moving it to a
per-symbol fast path is where the 2–3× whole-pipeline number lives.
The other four predictors get smaller, mechanical wins by mirroring
Order0's "compute only up to `byte+1`" approach over their own
specialised CDF builders.

### 1. NeuralSSM (the prize)

Today's `NeuralSsmPredictor::predict_cdf` (`aether-core/src/entropy/neural_ssm.rs:433-520`)
runs three passes over 256 entries:

1. Compute `raw[i] = mixed_prob(i)` for `i ∈ {0, 1}` (binary decisions)
   and `i ∈ [2, 256)` (literal distribution blending RLE + O2 + scale).
   Accumulates `sum` for normalisation.
2. Cumulative-rounding to CDF using `scale = PROB_TOTAL / sum`.
3. Forward monotonicity fix-up; overshoot fallback to `probs_to_cdf`.

The literal-distribution loop (pass 1, `i ∈ [2, 256)`) is the long pole:
~254 multiplies, conditional O2 lookup, and a per-symbol clamp.

**Proposed fast path.** Maintain `running_sum` of raw probabilities as
state across `update` calls — it's expensive only because `update` slightly
shifts the SSM weights and RLE probs. The reformulation:

* Keep the current `raw[i]` formula but compute it **on demand for one i**
  inside `query_cdf`.
* Cache `sum` (the normaliser) by running pass 1 ONCE per
  `(predict, update)` cycle — but only as a scalar accumulator, not a
  256-entry array. Total ops ~same as today's pass 1.
* `scale = PROB_TOTAL / sum` cached.
* `query_cdf(byte)`: compute `raw[byte]` and the running prefix sum
  `cum[byte] = Σ_{j<byte} raw[j]` ON THE FLY. With `sum` known up front,
  cumulative rounding for just `cdf[byte]` and `cdf[byte+1]` collapses to
  two scalar `round(cum * scale)` ops — the forward fix-up chain still
  needs replaying for correctness but only over `0..=byte+1` (as in Order0).
* Better: pre-tabulate `raw[]` AND `cum[]` in one fused pass during update
  (still O(256) but writes scalars to a packed array, fits in L1). Then
  `query_cdf` is O(byte) at worst, and skipping ContextMixer-blended O2
  for unobserved contexts gives an early-exit.

The single-normalised-lookup-per-byte hint from the task spec:

```
For byte b:
    raw_b   = mix_alpha * ssm_pb + mix_beta * rle_pb
    cdf_b   = round((Σ_{j<b} raw_j) * PROB_TOTAL / Σ_{j<256} raw_j)
    cdf_b1  = round((Σ_{j≤b} raw_j) * PROB_TOTAL / Σ_{j<256} raw_j)
```

With `sum` cached, `cdf_b1 - cdf_b` is just `round(raw_b * scale)` — and
we need `cdf_b` itself, so the prefix sum is unavoidable. But we can
either (a) maintain it incrementally on update, or (b) compute it lazily
via a Fenwick over raw probs (with the same forward-fix-up caveat).

**Realistic expected win**: pass 1 (the O2-blended literal loop) still
runs once per byte to refresh `sum`, but pass 2 (CDF write) and pass 3
(monotonicity fix-up) drop from O(256) to O(byte) average ~128. Plus the
514-byte return is gone. End-to-end NeuralSSM encode should drop ~30–40%,
which on the 92%-predictor budget is the headline 2–3× number.

### 2. RLE predictor (`aether-core/src/entropy/rle_predictor.rs`)

The RLE predictor's CDF heavily favours bytes 0 and 1 (RUNA/RUNB). Most
encoded bytes are 0 or 1, so `query_cdf(0)` and `query_cdf(1)` are common.
Fast path:

* For `byte ∈ {0, 1}`: read the two binary-decision probabilities, scale
  to 15 bits, return. O(1).
* For `byte ≥ 2`: fall back to the full literal-distribution build —
  rare in practice (literals are the minority on BWT-encoded data).

Expected win: 1.3–1.5× on RLE encode, dominated by the common-case
short-circuit.

### 3. ContextMixer (`aether-core/src/entropy/context_mixer.rs`)

Mixes ~8 sub-predictor distributions with learned weights. Building the
full 256-entry mixed distribution is the expensive part. Fast path
mirrors Order0: compute only `mixed[0..=byte+1]` for the cumulative sweep.
This skips the 254 upper-half mixes but doesn't help much when `byte` is
late in the alphabet. Combined with caching the per-sub-predictor CDFs
(they don't change between `query_cdf` and `update` for static mixers)
this could win ~1.2–1.5×.

### 4. Lz4Aware (`aether-core/src/entropy/lz4_aware.rs`)

Mostly an Order0-equivalent with an LZ4-token-aware reset. Same fast path
as Order0; expected 1.2× win, limited by the smaller absolute baseline
cost.

### 5. MTF (`aether-core/src/entropy/mtf_predictor.rs`)

Already very cheap (small contexts). The query_cdf default — `predict_cdf`
then slice — is probably fine; any "fast path" would save only the stack
copy, ~5% win at most.

## Decoder fast path sketch (and why we shouldn't bother yet)

The natural decode-side equivalent of `query_cdf` is "find the symbol s
where `cdf[s] ≤ codeword < cdf[s+1]`" using O(log 256) = 8 `query_cdf`
calls — a binary search over symbols:

```rust
let mut lo = 0u8;
let mut hi = 255u8;
while lo < hi {
    let mid = ((lo as u16 + hi as u16) >> 1) as u8;
    let (_, mid_hi) = predictor.query_cdf(mid);
    if codeword < mid_hi { hi = mid; } else { lo = mid + 1; }
}
let sym = lo;
// also need final (cdf[sym], cdf[sym+1]) for renorm — one more query_cdf
let (sym_lo, sym_hi) = predictor.query_cdf(sym);
```

This needs ~9 `query_cdf` calls per decoded byte. For NeuralSSM where
`query_cdf` still runs pass 1 (O(256) sum accumulation), this is 9×
slower than today. For Order0 where `query_cdf` is O(byte) average 128,
the 9 calls average ~9×128 = 1152 ops vs predict_cdf's 256 ops — 4.5×
slower. **Net loss** on every predictor we've considered.

The only way decode wins is if `query_cdf` becomes truly O(1) — which
requires the predictor to fully cache the CDF as state. At that point
you might as well return the cached CDF reference from `predict_cdf` and
get back to today's algorithm with one less allocation.

Conclusion: leave decode on `predict_cdf` indefinitely. The encode-only
asymmetry is the right answer.

## Measured impact (this prototype)

See report. TL;DR: Order0 encode shows a meaningful but variance-noisy
speedup (`compress_order0`), NeuralSSM is unchanged (default path).
Variance on this machine is ±30%; take single numbers with a grain of
salt.

## Open questions

* The `query_cdf` default impl returns a 4-byte tuple, which is great —
  but the predictor still builds the full 257 internally. For predictors
  where that's the only path, there's no win. Should we make `query_cdf`
  the primary method and have `predict_cdf` default-impl through it?
  Probably not — decode still needs the full table.
* Should `RangeEncoder::encode_cdf` be deprecated? It now just delegates,
  so it's harmless. Leaving it for source-compat.
