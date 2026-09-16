# AetherArch Benchmarks

Performance measurements and tuning workflows for the `aet` archiver.

## Workload Matrix

Use the checked-in matrix runner for repeatable text, log, executable/binary,
image, and tiny-file measurements across all compression profiles:

```powershell
pwsh ./scripts/benchmark-matrix.ps1 `
  -Binary C:\path\to\aet-aether-opt-019fb2fb.exe `
  -DatasetRoot C:\datasets\aether `
  -Iterations 10 `
  -OutputCsv benchmark-matrix.csv
```

The dataset root contains `text`, `logs`, `binaries`, `images`, and `tiny`
directories. The runner records input/archive bytes, ratio, compression and
extraction time, and throughput for `archival`, `balanced`, and `fast`.
Missing workload directories are reported and skipped. Keep published results
separate from historical values in `docs/BENCHMARKS.md`.

## Decompression

The compression matrix above times one extraction per case — enough to catch a
regression, not enough to reason about read speed. Two decompression-specific
tools sit alongside it.

### Per-stage benchmarks

```bash
cargo bench -p aether-core --bench decompression
```

Four groups, each isolating a stage so a change can be attributed rather than
lost in end-to-end noise:

| Group | What it measures |
|---|---|
| `decode/*` | The range-decoder hot loop per predictor. This is the stage that dominates every predictor-backed block. |
| `method/*` | `router::decompress_chunk` per compression method, on payloads produced by the real encoder. |
| `transform/*` | The inverse transforms: BWT+MTF, RLE, LZ77, byte-plane. |
| `archive/*` | Metadata parsing, whole-archive extraction, `verify`. |

### Workload matrix

```powershell
pwsh ./scripts/decompression-matrix.ps1 `
  -Binary C:\path\to\aet.exe `
  -DatasetRoot C:\datasets\aether `
  -Repetitions 5 `
  -Threads 1,2,4,0 `
  -OutputCsv decompression-matrix.csv
```

Same dataset layout as the compression matrix (`text`, `logs`, `binaries`,
`images`, `tiny`). Differences that matter:

- Compression runs once per case as **untimed setup**.
- Extraction is repeated and reported as **minimum and median**. Compare the
  minimum across builds — it is the run least polluted by whatever else the
  host was doing.
- Thread counts are swept, so scaling is measured rather than assumed.
- `verify` (decode without writing files) and single-file extraction (random
  access via the block index) are measured alongside full extraction; they are
  separate user-visible operations with different costs.
- Throughput is reported over **decompressed** bytes. An archive-size rate
  makes a decompressor look faster the better the compressor did.

### Measuring on a busy machine

Wall-clock on a shared development host varies by a factor of two to four
between runs — larger than most effects worth measuring. Three rules:

1. **Compare inside one process.** Where both code paths can be compiled
   together, run them alternately in one binary and take the minimum of
   several repetitions. Background load then affects both arms equally.
2. **Never compare across separate benchmark runs.** Criterion's
   `--save-baseline` is only trustworthy when both arms ran under the same
   machine conditions.
3. **Watch the ratio, not the absolute.** A 40% swing in both arms of the same
   run is the host, not the code.

See [`docs/perf/decompression.md`](docs/perf/decompression.md) for the
analysis these tools were built to support.

## Profile-Guided Optimization

AetherArch's hot path is byte-level entropy coding: millions of small
`predict()` / range-coder calls per second through
`aether-core/src/coding/rans.rs` and `aether-core/src/entropy/neural_ssm.rs`.
Branch direction and inlining choices in those files are exactly what PGO
biases well, so we ship a one-shot wrapper.

### One-liner

```powershell
pwsh ./scripts/pgo.ps1
```

The script:

1. Verifies `cargo-pgo` and `llvm-profdata` are available (instructions printed
   if not).
2. Builds an instrumented `aet` into `target/<host>/release-pgo/`.
3. Runs a training workload — compress + extract of `english.txt`,
   `source.rs`, and `mixed.json` from `tests/fixtures/large/`, three iterations
   each, into a temp dir that's cleaned up afterward.
4. Rebuilds the same crate with the merged profile via
   `cargo pgo optimize build` and prints the final binary path plus a
   `llvm-profdata` summary of the merged profile.

### Expected wins

Based on the Rust PGO literature (Rust compiler itself, ripgrep, hyperfine),
expect roughly **5–15% throughput improvement on encode and decode** for
predictor- and range-coder-bound workloads. Not measured in this repo yet —
plug in `aet bench --compare` against the PGO binary if you want concrete
numbers for your hardware.

### When to re-run

Re-run `scripts/pgo.ps1` whenever you materially change:

- `aether-core/src/coding/rans.rs` (range coder hot loop, CDF construction)
- `aether-core/src/entropy/neural_ssm.rs` or other `ProbabilityPredictor` impls
- The routing cascade in `aether-core/src/pipeline/router.rs`
- Branch-heavy code on the compress / decompress paths

Cosmetic edits (docstrings, formatting, test-only code) don't need a rebuild;
the previous PGO binary stays accurate.

### Not a default build

PGO is **not** part of `cargo build --release`. The standard release profile
is unchanged — same flags, same target dir, same output. PGO artifacts live in
the dedicated `release-pgo` profile under a separate output directory so
instrumentation runtime never leaks into normal builds.

If you `cargo build --profile release-pgo` directly (without the wrapper) you
get a plain release build that just happens to live in the wrong folder — no
PGO. Always go through `scripts/pgo.ps1`.
