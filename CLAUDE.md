# AetherArch - Claude Code Configuration

**Project**: AetherArch (.aet) — next-generation file archiver in Rust  
**Version**: 0.3.0
**Status**: 330 tests (164 unit + 96 integration + 28 FFI + 41 server + 1 doc, 7 ignored)  
**Workspace**: `aether-core`, `aether-cli` (`aet`), `aether-ffi`, `aether-server`, `aether-wasm`

---

## Development Standards

### Technology Stack
- **Language**: Rust 1.70+
- **Build**: Cargo workspace (5 crates)
- **Testing**: cargo test (unit, integration, FFI)
- **Benchmarking**: Criterion (`benches/compression.rs`, `benches/decompression.rs`)
- **Code Quality**: clippy, rustfmt
- **CI/CD**: GitHub Actions (.github/workflows/)

### Command Scripts
```bash
cargo build              # Debug build
cargo build --release   # Optimized release build
cargo test              # Run all tests (285 total)
cargo test --release   # Faster test execution
cargo bench -p aether-core  # Criterion benchmarks (compression + decompression)
cargo bench -p aether-core --bench decompression  # decode path only
cargo clippy -- -D warnings # Linting (must pass)
cargo fmt --check       # Format validation
aet bench --compare     # Compare vs gzip/bzip2/xz/zstd
pwsh ./scripts/decompression-matrix.ps1 -Binary ... -DatasetRoot ...  # read-path matrix
```

### Directory Organization
```
aether-core/
  src/
    chunker.rs          # FastCDC v2020 chunking
    coding/             # BWT, LZ77, RLE preprocessing
    entropy/            # Predictors: Neural SSM, RLE, Order0, ContextMixer
    pipeline/           # Router, compression, decompression
    crypto/             # AES-256-GCM, ChaCha20-Poly1305, Argon2id
    dictionary.rs       # Dictionary training/loading (.aed format)
    format.rs           # CompressionMethod, PredictorId enums
    cloud/              # StorageBackend trait, CloudReader
  benches/              # Criterion benchmarks
  examples/             # basic_compress, basic_decompress, streaming_extract
  tests/                # Integration tests

aether-cli/src/main.rs       # CLI: compress, extract, verify, list, train, migrate, bench
aether-ffi/src/lib.rs        # C FFI bindings
aether-server/src/main.rs    # REST API (axum)
aether-wasm/src/lib.rs       # WebAssembly bindings (decompress-only)
```

### Naming Conventions
- **Functions**: snake_case
- **Types**: PascalCase
- **Constants**: UPPER_SNAKE_CASE
- **Test modules**: `#[cfg(test)] mod tests { ... }`
- **Compression methods**: CompressionMethod enum variants (PredictorRans=0, Zstd=1, Store=2, etc.)
- **Predictor types**: PredictorId enum (Order0=0, ContextMixer=1, NeuralSsm=2, etc.)

### Code Practices
- **Error handling**: `Result<T>` types with context via `?` operator
- **Memory safety**: No `unsafe` except FFI boundary and validated crypto operations
- **Performance**: Inline hot paths (predictors, entropy coder); profile with benchmarks before optimizing
- **Thread safety**: Use `rayon` for parallel decompression; predictors must support `Send + Sync`
- **API design**: Public traits are stable; internal types prefixed with `_` if private implementation details

### Key Architecture Decisions

**[DECISION] Routing Cascade**
- BWT+MTF+RLE → LZ77 → Plain RC → Zstd → Store
- Why: Entropy-based adaptive routing picks smallest compressed form
- Update when: Adding new compression methods (update router.rs and format.rs)

**[DECISION] Block Independence**
- Every predictor-backed payload is coded by a scratch predictor that
  `encode_block`/`decode_block` reset per block, so no cross-block group
  predictor state is ever consumed
- Why: it is what makes parallel compression, parallel decompression and
  single-file random access all valid, and it removed a per-output-byte
  predictor pass from the decode path
- How to apply: decode through `router::decompress_chunk_with_baseline` and
  pass the dictionary baseline, not a predictor. The `predictor_synced` /
  `predictor_state_flag` fields are still written and read for format
  stability but no path consumes them — do not reintroduce a sync pass
  without a predictor that genuinely carries state across blocks

**[DECISION] Per-Block Encryption**
- Master nonce XOR block_id enables random-access decryption without reading sequentially
- Why: Supports seekable decompression in encrypted archives
- How to apply: Never change nonce derivation in crypto/mod.rs without updating all paths

**[DECISION] Streaming vs Seekable Decompression**
- Streaming: Read-only, sequential I/O (stdin support via "-" sentinel)
- Seekable: Full random access (default, requires Seek trait)
- Why: Matches use cases; streaming is memory-efficient for pipelines
- How to apply: Choose decompress_streaming.rs for pipes, decompress_seekable.rs for files

### Performance Targets
- **Internal (2.6 MiB)**: 2.75% ratio (0.220 bpb), 3.0 comp MB/s, 3.1 decomp MB/s
- **Silesia (202 MiB)**: 26.45% ratio (2.116 bpb), 0.2 comp MB/s, 0.3 decomp MB/s
- **Compression**: Prioritize ratio over speed; entropy coder ~1 MiB/s is acceptable
- **Decompression**: entropy decoding is the whole cost — everything else put
  together is under a tenth of it on the archival profile. Measure with
  `cargo bench -p aether-core --bench decompression`, which isolates the
  decode loop, each method, each inverse transform and the archive paths.
  See `docs/perf/decompression.md`
- **Benchmarking on a loaded machine**: wall-clock here varies 2-4x between
  runs. Compare old and new code paths *inside one process* and take the
  minimum of several repetitions; never compare across separate runs

### Quality Assurance

**Testing Strategy**
- **Unit tests**: Alongside source code, test public APIs and invariants
- **Integration tests**: End-to-end roundtrip (compress → decompress → verify)
- **FFI tests**: C binding lifecycle and error handling
- **Server tests**: REST endpoint contracts
- **Property tests**: RLE decode, entropy coder (synthetic data)
- **Focus**: User behavior (can I extract files? Does verification pass?) not implementation

**External Dependencies**
- Prefer well-established crates (zstd-sys, libbrotli)
- Crypto: ring, chacha20poly1305 (audited, stable)
- Avoid single-author unmaintained crates
- Pin versions for security-critical dependencies

**Supply Chain Security**
```toml
# Enforce minimum package age (7 days) to reduce zero-day risk
[package]
publish = true
```

**Enforcement Rules** (RED LINES)
1. ❌ No `unsafe` code outside crypto/ffi boundaries without explicit review
2. ❌ No secrets in code (API keys, passwords) — use environment variables
3. ❌ No hardcoded file paths (use configurable paths)
4. ❌ All public APIs must have docstrings with examples
5. ❌ Breaking changes require deprecation warnings in prior version

### Documentation
- **API docs**: `cargo doc --open` must render correctly
- **Examples**: `aether-core/examples/` show common workflows
- **Benchmarks**: Document in BENCHMARKS.md with Criterion results
- **Gotchas**: Document in memory files (`CLAUDE.md` this file)

---

## Code Review Framework

### Always Check

1. **Test Coverage**: New public APIs must have at least one test
   - Roundtrip compress → decompress → verify
   - Error cases (corrupted data, invalid headers, OOM bounds)

2. **Memory Safety**: 
   - `unsafe` blocks justified and marked `// SAFETY: <reason>`
   - BWT: Check MAX_BWT_INPUT_SIZE prevents 10× amplification
   - RLE decode: Verify MAX_DECODE_SIZE guards (64 MiB)
   - No unbounded allocations

3. **Error Handling**:
   - All Results propagated with context (not swallowed)
   - Decompression errors include block ID, archive offset, method
   - No panics in library code (only CLI/server can panic)

4. **Encryption & Secrets**:
   - No hardcoded keys or nonces
   - Argon2id parameters documented (64 MiB, 3 iter, 4 lanes)
   - Per-block nonce derivation unchanged

5. **Predictor State**:
   - All ProbabilityPredictor impls must update internal state on predict/encode
   - Streaming path carries state via HashMap<u32, predictor>
   - sync_predictor called at group boundaries (unless predictor_synced=true)

6. **Dependency Changes**:
   - Justify new crate additions (e.g., "needed for neural SSM optimization")
   - Pin versions for security-critical libs (crypto, compression)
   - Run `cargo audit` — no high-severity vulns allowed

### Security Checklist

- ❌ No hardcoded credentials (passwords, API keys, tokens)
- ❌ No buffer overflows (checked bounds in range coder, RLE decoder)
- ❌ No path traversal (cloud URLs validated)
- ❌ Argon2id params protect against brute-force (64 MiB memory minimum)
- ❌ AEAD ciphers used correctly (nonce never reused with same key)
- ✅ Encrypt-after-compress (not before, preserves block-level access)

### Performance Checklist

- **N+1 Issues**: Group predictor state reused within group (not recreated per block)
- **Unbounded Operations**: FastCDC window bounded (512 KiB avg, 8 MiB max)
- **Memory Leaks**: Rayon thread pool properly bounded; no stale predictors
- **Caching**: Dictionary state precomputed once (not per-file)
- **Hot Paths**: NeuralSsmPredictor.predict() and RangeEncoder inline-enabled

### Skip (Don't Review Closely)

- ✅ Auto-generated code (cbindgen output, bindgen)
- ✅ Formatting-only changes (run `cargo fmt` first)
- ✅ Version bumps alone (Cargo.toml PATCH increments)
- ✅ Benchmark-only changes (Criterion results)

### Severity Markers

- 🔴 **Blocking**: Memory safety bug, panics in library, secret leak, breaking API without deprecation
- 🟡 **Non-Critical**: Style inconsistency, redundant code, optimization opportunity
- 🟣 **Pre-Existing**: Known limitation documented elsewhere

### Documentation Sync

Ensure PR updates docs if it changes:
- Public API signatures → update rustdoc comments
- Predictor behavior → update CLAUDE.md [DECISION] section
- Compression method routing → update comments in router.rs
- New CLI flags → update `aet --help` description
- Performance characteristics → update benchmarks in BENCHMARKS.md

---

## Gotchas & Troubleshooting

### [GOTCHA] BWT Memory Amplification
- **Issue**: BWT on large inputs allocates 10× memory (SA construction)
- **Solution**: MAX_BWT_INPUT_SIZE = 8 MiB enforced; larger chunks skip BWT
- **Entropy-based skip**: Chunks >=7.0 bps (very random) bypass SA-IS

### [GOTCHA] Predictor State Drift
- **Issue**: Streaming path must carry predictor HashMap across blocks
- **Solution**: `decompress_streaming.rs` maintains HashMap<u32, predictor>; verify sync at group boundaries
- **Check**: If verification fails unexpectedly, inspect predictor sync in decompress logs

### [GOTCHA] Nonce Reuse in Encryption
- **Issue**: Reusing nonce+key with ChaCha20-Poly1305 breaks security
- **Solution**: Master nonce XOR block_id; never reuse with same key
- **Verify**: crypto/mod.rs `maybe_decrypt_payload()` always derives unique nonce

### [GOTCHA] Range Coder Precision
- **Issue**: 15-bit CDF precision can overflow with extreme distributions
- **Solution**: probs_to_cdf() uses saturating arithmetic; verify on synthetic worst-case data
- **Test**: See `fuzz/decode_block` target

### [GOTCHA] lz4_flex Versioning
- **Pinned**: "=0.11.6" for format stability (upgrades require compatibility testing)
- **Why**: LZ4 format compatibility is subtle; unpin only with extensive testing

### [GOTCHA] Rayon Thread Pool Scope
- **Issue**: Predictors must be `Send + Sync`; mutable static is unsafe
- **Solution**: Create predictors on main thread, move ownership to rayon workers
- **Check**: Compile with `RUSTFLAGS="-Z sanitizer=thread"` to detect races

---

## Handoff Template

When closing a session, document:
1. **Current progress**: What was completed in this session?
2. **Blockers**: What's stuck? Why? What's the diagnosis?
3. **Next steps**: Specific files/functions to tackle
4. **Unresolved**: Any ambiguous architectural choices? Link to memory files.
5. **Performance notes**: Any benchmarks run? Surprising results?

**File**: `~/.claude/handoffs/YYYY-MM-DD_HH-MM_<session-id>.md` (max 1500 tokens)

Example:
```
# Session 2026-04-14 — AetherArch Handoff

## Progress
- Implemented streaming decompression predictor sync
- Fixed entropy-based BWT skip logic
- 4 new integration tests passing

## Blockers
- RLE decoder still panics on corrupted sparse data (fuzz/sparse_rle_decode)
  - Issue: Saturation arithmetic doesn't catch all edge cases
  - Diagnosis: Need to review RlePredictor update logic

## Next Steps
- Fix RLE panic in bwt_preprocess.rs:92
- Add property test for RLE round-trip
- Benchmark streaming vs seekable on 100 MiB files

## Unresolved
- Should Order0 state include frequency tables? (memory tradeoff)
- See CLAUDE.md [DECISION] section — currently deferring

## Notes
- Silesia benchmark shows 2-hour turnaround; blocking further optimization work
```

---

## Libraries for Reference

| Area | Library | Purpose |
|------|---------|---------|
| Compression | zstd-sys | Fallback fast compression |
| Crypto | chacha20poly1305, aes-gcm | AEAD ciphers |
| KDF | argon2 | Password-based key derivation |
| Hashing | blake3 | Dictionary verification |
| Parallel | rayon | Parallel compression (per chunk) and decompression (per block) |
| Fuzzing | libfuzzer | Crash detection |
| Benchmarking | criterion | Performance profiling |
| WebAssembly | wasm-bindgen | JS FFI |
| HTTP | axum, tokio | Server framework |

---

## Useful Links

- **Format**: `aether-core/src/format.rs` — CompressionMethod, PredictorId enums
- **Router**: `aether-core/src/pipeline/router.rs` — adaptive routing cascade
- **Streaming**: `aether-core/src/pipeline/decompress_streaming.rs` — predictor carry-over
- **CLI**: `aether-cli/src/main.rs` — command definitions
- **Server**: `aether-server/src/main.rs` — REST API endpoints
- **Tests**: `cargo test --lib` (unit), `cargo test --test` (integration)

---

**Last Updated**: 2026-04-14  
**Maintainer**: AetherArch Team  
**Questions?**: Check git log, memory files, or run with `--verbose` flag
