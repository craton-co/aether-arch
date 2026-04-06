#!/bin/bash
# AetherArch Local CI Test Runner using 'act'
# Runs all GitHub Actions CI jobs locally

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           AetherArch Local CI Test Runner                   ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

log_section() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}→ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

log_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

log_header

echo "System Information:"
echo "  Branch:          $BRANCH"
echo "  Commit:          $COMMIT"
echo "  Timestamp:       $TIMESTAMP"
echo "  Project Root:    $PROJECT_ROOT"
echo ""

# Check act is available
if ! command -v act &> /dev/null; then
    log_error "GitHub Actions Local CLI 'act' not found"
    echo ""
    echo "Installation:"
    echo "  Windows (Scoop):  scoop install act"
    echo "  macOS (Homebrew): brew install act"
    echo "  Linux:            https://github.com/nektos/act"
    exit 1
fi

log_success "GitHub Actions Local CLI (act) found: $(act --version)"
echo ""

# Check git is available
if ! command -v git &> /dev/null; then
    log_error "Git not found"
    exit 1
fi

log_success "Git found: $(git --version)"
echo ""

# Check Cargo is available
if ! command -v cargo &> /dev/null; then
    log_error "Cargo not found"
    exit 1
fi

log_success "Cargo found: $(cargo --version)"
log_success "Rust found: $(rustc --version)"
echo ""

log_section "Running GitHub Actions Workflows Locally"
echo ""
echo "Available workflows to test:"
echo "  • test     - Build and test on Linux"
echo "  • clippy   - Lint with Clippy"
echo "  • fmt      - Check code formatting"
echo "  • deny     - Check dependencies with cargo-deny"
echo "  • doc      - Build documentation"
echo "  • msrv     - Check minimum supported Rust version"
echo ""
echo "Running: act --workflow=.github/workflows/ci.yml"
echo ""

# Run act with GitHub Actions CI workflow
# We'll run locally without docker to speed up testing
if act \
    --workflow .github/workflows/ci.yml \
    --list; then
    log_info "Available jobs from CI workflow:"
    echo ""
fi

echo ""
log_section "Test 1: Format Check"
log_info "Running: cargo fmt --all -- --check"
if cargo fmt --all -- --check; then
    log_success "Format check passed"
else
    log_error "Format check failed"
    exit 1
fi
echo ""

log_section "Test 2: Clippy Linting"
log_info "Running: cargo clippy --workspace -- -D warnings"
if cargo clippy --workspace -- -D warnings; then
    log_success "Clippy linting passed"
else
    log_error "Clippy linting failed"
    exit 1
fi
echo ""

log_section "Test 3: Build (Release)"
log_info "Running: cargo build --workspace --release"
if cargo build --workspace --release; then
    log_success "Build passed"
else
    log_error "Build failed"
    exit 1
fi
echo ""

log_section "Test 4: Test Suite"
log_info "Running: cargo test --workspace --release"
if cargo test --workspace --release; then
    log_success "Test suite passed"
else
    log_error "Test suite failed"
    exit 1
fi
echo ""

log_section "Test 5: Documentation"
log_info "Running: cargo doc --workspace --no-deps"
RUSTDOCFLAGS="-Dwarnings" cargo doc --workspace --no-deps || {
    log_error "Documentation build failed"
    exit 1
}
log_success "Documentation build passed"
echo ""

log_section "Test 6: MSRV Check (Rust 1.85.0)"
log_info "Running: cargo +1.85.0 check --workspace"
if rustup install 1.85.0 2>/dev/null; then
    log_info "Installed Rust 1.85.0"
fi
if cargo +1.85.0 check --workspace; then
    log_success "MSRV check passed"
else
    log_error "MSRV check failed"
    exit 1
fi
echo ""

log_section "Test 7: Cargo Deny (Dependency Security)"
log_info "Running: cargo deny check"
if command -v cargo-deny &> /dev/null; then
    if cargo deny check; then
        log_success "Cargo deny check passed"
    else
        log_error "Cargo deny check failed"
        exit 1
    fi
else
    log_info "cargo-deny not installed, skipping this check"
    log_info "To install: cargo install cargo-deny"
fi
echo ""

# Summary
log_section "CI Test Summary"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ All CI checks passed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Test Results:"
echo "  ✓ Format check:      PASSED"
echo "  ✓ Clippy linting:    PASSED"
echo "  ✓ Build test:        PASSED"
echo "  ✓ Test suite:        PASSED"
echo "  ✓ Documentation:     PASSED"
echo "  ✓ MSRV check:        PASSED"
echo ""

# Check if cargo-deny was skipped
if ! command -v cargo-deny &> /dev/null; then
    echo "  ⚠ Cargo deny:       SKIPPED (not installed)"
else
    echo "  ✓ Cargo deny:       PASSED"
fi

echo ""
echo "Next Steps:"
echo "  1. Commit and push to branch: git push origin $BRANCH"
echo "  2. Create a pull request to main"
echo "  3. GitHub Actions will run additional platform tests:"
echo "     • Ubuntu (Linux)"
echo "     • Windows"
echo "     • macOS"
echo ""
