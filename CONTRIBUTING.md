# Contributing to AetherArch

Thank you for your interest in contributing to AetherArch! This document provides
guidelines and information for contributors.

## License

AetherArch is licensed under Apache-2.0. By contributing, you agree that your
contributions will be licensed under the same terms. You retain copyright to
your contributions. All contributors must sign the Contributor License Agreement (CLA).

### Developer Certificate of Origin (DCO)

All commits must include a `Signed-off-by` line certifying that you wrote
or have the right to submit the code under the project's license. Add it
automatically with `git commit -s`. The DCO text is available at
<https://developercertificate.org/>.

## Code of Conduct

As a contributor, you are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md)
at all times. We are committed to providing a welcoming and inclusive experience
for everyone.

## Getting Started

### Prerequisites

- Rust toolchain (see `rust-version` in `Cargo.toml` for MSRV)
- Git

### Building

```bash
cargo build --workspace --release
```

### Running Tests

```bash
# All tests
cargo test --workspace --release

# Specific crate
cargo test -p aether-core --release

# With output
cargo test --workspace --release -- --nocapture
```

### Linting

```bash
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

## Development Workflow

1. **Fork and branch**: Create a feature branch from `main`.
2. **Make changes**: Follow the code style guidelines below.
3. **Test**: Ensure all existing tests pass and add tests for new functionality.
4. **Lint**: Run clippy and rustfmt before committing.
5. **Commit**: Use clear, descriptive commit messages.
6. **Pull request**: Open a PR against `main` with a description of changes.

## Code Style

- Follow standard Rust conventions (`rustfmt` defaults).
- All public items must have `///` rustdoc comments.
- Use `thiserror` for error types, `anyhow` only in the CLI binary.
- Prefer `Result<T>` (from `crate::error`) over raw `std::result::Result`.
- All predictors must implement the `ProbabilityPredictor` trait.
- Determinism is critical: predictors must produce bit-identical output
  across platforms for the same input sequence.

## Architecture Guidelines

- **`aether-core`**: Library crate. No I/O to stdout/stderr. No CLI dependencies.
- **`aether-cli`**: Binary crate. Thin wrapper over `aether-core` public API.
- **Entropy predictors**: One file per predictor in `entropy/`. Must implement `ProbabilityPredictor`.
- **Coding transforms**: One file per transform in `coding/`. Encode/decode must be inverse operations.
- **Pipeline**: `router.rs` decides method per chunk, `compress.rs` orchestrates end-to-end,
  `decompress.rs` handles extraction.

### Adding a New Predictor

1. Create `aether-core/src/entropy/your_predictor.rs`.
2. Implement the `ProbabilityPredictor` trait (see `traits.rs`):
   - `predict() -> [f32; 256]` — return byte probability distribution
   - `predict_cdf() -> [u16; 257]` — return CDF for range coder (15-bit precision)
   - `update(byte: u8)` — update internal state after observing a byte
   - `name() -> &str`, `predictor_id() -> PredictorId`
   - `save_state() / load_state()` — for dictionary pretraining support
3. Add a `PredictorId` variant in `format.rs` and wire it into `PredictorId::from_u16()`.
4. Register the predictor in `router.rs` (`make_factory`) and `aether-cli/src/main.rs`.
5. Add unit tests: determinism, adaptation, roundtrip with range coder.
6. **Critical**: predictions must be deterministic — same input sequence must produce
   bit-identical CDF output on all platforms (beware f32 FMA differences on ARM vs x86).

### Fuzz Testing

AetherArch maintains fuzz targets in `fuzz/` using `cargo-fuzz` / `libfuzzer-sys`.
Run them with:

```bash
cargo +nightly fuzz run fuzz_block_header
cargo +nightly fuzz run fuzz_streaming_metadata
cargo +nightly fuzz run fuzz_decode_block
cargo +nightly fuzz run fuzz_range_coder
```

When adding new parsing code that handles untrusted input, please add a
corresponding fuzz target. See existing targets in `fuzz/fuzz_targets/`
for examples.

### Unsafe Code Policy

`aether-core` avoids `unsafe` code. The only `unsafe` in the workspace lives
in `aether-ffi/src/lib.rs` (C FFI boundary, inherently unsafe). If you
believe `unsafe` is necessary for performance:

1. Document the safety invariants in a `// SAFETY:` comment.
2. Add a fuzz target covering the unsafe path.
3. Explain in the PR why a safe alternative is insufficient.

### Dependency Notes

- **`zstd`**: Wraps the C `libzstd` library via `zstd-sys`. Requires a C compiler at build time.
  The `zstd-sys` crate vendors the C source and compiles it via `cc`. This is the only
  non-Rust dependency in the default build.
- **`lz4_flex`**: Pure Rust, pinned to `=0.11.3` for format stability. Do not upgrade without
  verifying that existing archives still decompress correctly.

## What Needs Help

Check the [docs/ROADMAP.md](docs/ROADMAP.md) for current priorities. High-impact areas:

- **Performance**: Target 2+ MiB/s compression, 5+ MiB/s decompression.
- **Platform testing**: Verify on Linux, macOS, and Windows.
- **Code coverage**: Achieve 80%+ coverage on core compression/decompression paths.
- **Format freeze**: Help validate the `.aet` format spec for 1.0 stability.

## Reporting Bugs

- Use GitHub Issues with a clear title and reproduction steps.
- Include: OS, Rust version, AetherArch version, sample input (if possible).
- For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## Questions?

Open a GitHub Discussion or reach out via Issues.

---

Copyright 2024-2026 Craton Software Company Licensed under Apache-2.0.
