# AetherArch Local CI Testing Guide

This guide explains how to run all GitHub Actions CI jobs locally before uploading/pushing code.

## Overview

The AetherArch project uses GitHub Actions for continuous integration with the following jobs:

| Job | Description | Status |
|-----|-------------|--------|
| **test** | Build & test on Linux, Windows, macOS | Multi-platform |
| **clippy** | Lint with Clippy (strict warnings) | Linux |
| **fmt** | Code format check with rustfmt | Linux |
| **deny** | Dependency security check | Linux |
| **doc** | Documentation build | Linux |
| **msrv** | Minimum Supported Rust Version check | Linux (1.85.0) |
| **cla** | Contributor License Agreement | GitHub only |
| **release** | Build artifacts on tag | Conditional |

## Quick Start

### Run All CI Checks Locally

```bash
chmod +x ci_test_suite.sh
./ci_test_suite.sh
```

This will sequentially run:
1. Format check (rustfmt)
2. Linting (Clippy)
3. Build (release)
4. Test suite (all tests)
5. Documentation build
6. MSRV check (Rust 1.85.0)
7. Dependency security (cargo-deny)
8. Feature flag combinations

### Individual Tests

If you only want to run specific checks:

```bash
# Format only
cargo fmt --all -- --check

# Auto-fix formatting
cargo fmt --all

# Clippy linting
cargo clippy --workspace -- -D warnings

# Build
cargo build --workspace --release

# Tests
cargo test --workspace --release

# Documentation
RUSTDOCFLAGS="-Dwarnings" cargo doc --workspace --no-deps

# MSRV (Rust 1.85.0)
cargo +1.85.0 check --workspace

# Dependency check
cargo deny check

# Feature combinations
cargo check --workspace                           # default features
cargo check --workspace --no-default-features   # minimal
cargo check --workspace --all-features          # all features
```

## What Each Check Does

### 1. Format Check (`rustfmt`)

Ensures code follows Rust formatting standards.

```bash
cargo fmt --all -- --check  # Check without modifying
cargo fmt --all             # Auto-fix
```

**Fix needed?** Run `cargo fmt --all` before committing.

### 2. Linting (`clippy`)

Catches common mistakes and suggests improvements.

```bash
cargo clippy --workspace -- -D warnings
```

The `-D warnings` flag treats warnings as errors, matching GitHub Actions behavior.

### 3. Build (`cargo build`)

Compiles all crates in release mode.

```bash
cargo build --workspace --release
```

Release mode enables optimizations and catches more errors.

### 4. Tests (`cargo test`)

Runs all unit, integration, and doc tests.

```bash
cargo test --workspace --release

# Including ignored tests (slow tests)
cargo test --workspace --release -- --include-ignored
```

### 5. Documentation (`cargo doc`)

Builds all API documentation and checks for broken links.

```bash
RUSTDOCFLAGS="-Dwarnings" cargo doc --workspace --no-deps
```

The `--no-deps` flag skips generating docs for dependencies (faster).

### 6. MSRV Check (Minimum Supported Rust Version)

Ensures code compiles on Rust 1.85.0.

```bash
# Install if needed
rustup install 1.85.0

# Check
cargo +1.85.0 check --workspace
```

**Note:** Check only (not full build) for speed.

### 7. Dependency Security (`cargo-deny`)

Scans dependencies for known security vulnerabilities and license issues.

```bash
# Install if not present
cargo install cargo-deny

# Check
cargo deny check
```

### 8. Feature Combinations

Tests that features can be disabled/enabled independently:

```bash
# Default features
cargo check --workspace

# Minimal (no defaults)
cargo check --workspace --no-default-features

# All features
cargo check --workspace --all-features
```

## Troubleshooting

### Format Check Fails

```bash
# See what changes are needed
cargo fmt --all -- --check

# Auto-fix
cargo fmt --all

# Re-check
cargo fmt --all -- --check
```

### Clippy Warnings

```bash
# See detailed warnings
cargo clippy --workspace -- -D warnings

# Fix common issues (some are auto-fixable)
cargo clippy --fix --allow-dirty

# For specific crates
cargo clippy -p aether-core -- -D warnings
```

### Test Failures

```bash
# Run with backtrace
RUST_BACKTRACE=1 cargo test --workspace --release

# Run single test
cargo test test_name -- --exact

# Run with logging
RUST_LOG=debug cargo test --workspace --release
```

### MSRV Issues

If code uses features from newer Rust versions:

```bash
# Check what features are available in 1.85.0
rustup doc

# Revert to compatible syntax
# Common issues:
# - `is_none_or()` -> use `Option::and_then()` or pattern match
# - `?` in main/tests -> use `Result::map_err()`
```

## Integration with Git Workflow

### Before Committing

```bash
# Run all checks
./ci_test_suite.sh

# If everything passes
git add -A
git commit -m "description"
git push origin branch-name
```

### Before Opening PR

Ensure local tests pass:

```bash
./ci_test_suite.sh && echo "Ready for PR!" || echo "Fix failures above"
```

### Creating a PR

```bash
# Using GitHub CLI
gh pr create --base main --head your-branch

# Or manually on GitHub.com
# - Base: main
# - Compare: your-branch
```

## CI Workflow in GitHub Actions

After pushing, GitHub Actions will automatically run:

1. **Multi-platform tests** (Linux, Windows, macOS)
2. All checks listed above
3. Release build (if tagged)

These additional platform tests can't be run locally without Docker, but the core checks are covered.

## Docker Testing (Optional)

For full GitHub Actions simulation with Docker:

### Using `act`

```bash
# List available jobs
act --list

# Run specific job
act -j test

# Run with local Docker (requires Docker daemon)
act --container-architecture linux/amd64
```

### Docker Daemon Issues (Windows)

If `docker` command fails:

1. Start Docker Desktop
2. Wait for "Docker daemon started" notification
3. Re-run tests

## Performance Tips

1. **Use `--release` flag**: Builds take longer but better match CI
2. **Cache dependencies**: First run downloads dependencies; subsequent runs are faster
3. **Run tests in sequence**: Don't parallelize if disk I/O is the bottleneck
4. **Skip ignored tests**: `cargo test --workspace --release` (excludes slow tests by default)

## Expected Runtimes

On a typical development machine (4 CPU, 8GB RAM):

| Check | Time |
|-------|------|
| Format | ~5 sec |
| Clippy | ~30 sec |
| Build | ~60 sec |
| Tests | ~120 sec |
| Docs | ~30 sec |
| MSRV | ~45 sec |
| Cargo Deny | ~10 sec |
| **Total** | **~300 sec (~5 min)** |

## Files in This Guide

- `ci_test_suite.sh` - Main test runner (all checks)
- `run_ci_locally.sh` - Alternative runner using `act`
- `Dockerfile.ci` - Docker image for containerized testing
- `run_ci_tests.sh` / `run_ci_tests.bat` - Docker-based runners
- `.github/workflows/ci.yml` - Actual GitHub Actions workflow

## Further Reading

- [GitHub Actions Workflow](.github/workflows/ci.yml)
- [act - Run GitHub Actions Locally](https://github.com/nektos/act)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Clippy Lints](https://doc.rust-lang.org/clippy/)
