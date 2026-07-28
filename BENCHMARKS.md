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
