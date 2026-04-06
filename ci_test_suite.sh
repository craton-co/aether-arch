#!/bin/bash
# Comprehensive CI Test Suite for AetherArch
# Tests all GitHub Actions CI jobs locally

set +e  # Don't exit on first error, we want to run all tests

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test results tracking
declare -A test_results
tests_passed=0
tests_failed=0
tests_skipped=0

log_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║      AetherArch CI Test Suite - Complete Validation         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

log_section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

run_test() {
    local test_name="$1"
    local test_command="$2"

    echo -n "Testing: $test_name ... "

    if eval "$test_command" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        test_results["$test_name"]="PASSED"
        ((tests_passed++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        test_results["$test_name"]="FAILED"
        ((tests_failed++))
        return 1
    fi
}

log_header

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)

echo "System Information:"
echo "  Branch:    $BRANCH"
echo "  Commit:    $COMMIT"
echo "  Rust:      $(rustc --version)"
echo "  Cargo:     $(cargo --version)"
echo ""

log_section "1. FORMAT CHECK (rustfmt)"
run_test "Format check" "cargo fmt --all -- --check"
if [ $? -ne 0 ]; then
    echo "  Hint: Run 'cargo fmt --all' to auto-fix formatting"
fi
echo ""

log_section "2. LINTING (Clippy)"
run_test "Clippy warnings" "cargo clippy --workspace -- -D warnings"
echo ""

log_section "3. BUILD (Release)"
run_test "Build release" "cargo build --workspace --release"
echo ""

log_section "4. UNIT & INTEGRATION TESTS"
run_test "Test suite" "cargo test --workspace --release"
echo ""

log_section "5. DOCUMENTATION"
RUSTDOCFLAGS="-Dwarnings" run_test "Doc build" "cargo doc --workspace --no-deps"
echo ""

log_section "6. MSRV CHECK (Rust 1.85.0)"
# Check if 1.85.0 is installed
if rustup toolchain list | grep -q "1.85.0"; then
    run_test "MSRV (1.85.0)" "cargo +1.85.0 check --workspace"
else
    echo "Installing Rust 1.85.0..."
    rustup install 1.85.0 >/dev/null 2>&1
    run_test "MSRV (1.85.0)" "cargo +1.85.0 check --workspace"
fi
echo ""

log_section "7. DEPENDENCY SECURITY (cargo-deny)"
if command -v cargo-deny &> /dev/null; then
    run_test "Cargo deny check" "cargo deny check"
else
    echo -e "${YELLOW}⊘ SKIPPED${NC} - cargo-deny not installed"
    ((tests_skipped++))
    test_results["Cargo deny"]="SKIPPED"
fi
echo ""

log_section "8. FEATURE FLAGS & COMBINATIONS"
echo "Testing feature combinations..."
echo ""

run_test "Default features" "cargo check --workspace"
run_test "No default features" "cargo check --workspace --no-default-features"
run_test "All features" "cargo check --workspace --all-features"
echo ""

log_section "9. TEST VARIANT: Vec allocation stress"
run_test "Tests (including ignored)" "cargo test --workspace --release -- --include-ignored"
echo ""

# Summary
log_section "TEST RESULTS SUMMARY"
echo ""

echo -e "Total Tests Run: $((tests_passed + tests_failed))"
echo -e "  ${GREEN}✓ Passed: $tests_passed${NC}"
echo -e "  ${RED}✗ Failed: $tests_failed${NC}"

if [ $tests_skipped -gt 0 ]; then
    echo -e "  ${YELLOW}⊘ Skipped: $tests_skipped${NC}"
fi

echo ""
echo "Detailed Results:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for test_name in "${!test_results[@]}"; do
    result="${test_results[$test_name]}"
    case "$result" in
        PASSED)
            echo -e "  ${GREEN}✓${NC} $test_name"
            ;;
        FAILED)
            echo -e "  ${RED}✗${NC} $test_name"
            ;;
        SKIPPED)
            echo -e "  ${YELLOW}⊘${NC} $test_name"
            ;;
    esac
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $tests_failed -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║       ✓ ALL CI CHECKS PASSED SUCCESSFULLY!                  ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Ready to push:"
    echo "  git add -A"
    echo "  git commit -m 'refactor: Auto-format code with rustfmt'"
    echo "  git push origin $BRANCH"
    echo ""
    echo "Then create PR: gh pr create --base main --head $BRANCH"
    echo ""
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║           ✗ SOME CI CHECKS FAILED                           ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Failed tests: $tests_failed"
    echo ""
    exit 1
fi
