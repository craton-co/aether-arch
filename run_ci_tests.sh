#!/bin/bash
# AetherArch Local CI Test Runner
# Runs all GitHub Actions CI jobs in Docker containers

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse --short HEAD)
IMAGE_NAME="aether-ci-test"
CONTAINER_NAME="aether-ci-${BRANCH}-${COMMIT}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           AetherArch Local CI Test Runner                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Branch:          $BRANCH"
echo "Commit:          $COMMIT"
echo "Container Name:  $CONTAINER_NAME"
echo "Timestamp:       $TIMESTAMP"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Docker image exists, rebuild if needed
log_section "Building Docker Image"
if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Image $IMAGE_NAME exists. Rebuilding to ensure fresh dependencies..."
fi
docker build -f Dockerfile.ci -t "$IMAGE_NAME" . || {
    log_error "Docker build failed"
    exit 1
}
log_success "Docker image built: $IMAGE_NAME"
echo ""

# Run container with all CI tests
log_section "Running CI Tests in Container"
echo "Command: docker run --rm --name \"$CONTAINER_NAME\" \"$IMAGE_NAME\""
echo ""

if docker run \
    --rm \
    --name "$CONTAINER_NAME" \
    --cpus 4 \
    --memory 8g \
    -v "$PROJECT_ROOT:/workspace" \
    "$IMAGE_NAME"; then
    log_success "All CI tests passed!"
else
    log_error "CI tests failed!"
    exit 1
fi
echo ""

# Run individual test groups for detailed reporting
log_section "Running Individual Test Groups"

# Test - Release build
log_section "1. Build (Release)"
docker run --rm \
    --name "$CONTAINER_NAME-build" \
    --cpus 4 \
    --memory 8g \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    cargo build --workspace --release && \
    log_success "Build test passed" || {
    log_error "Build test failed"
    exit 1
}
echo ""

# Test - Unit tests
log_section "2. Tests (Unit + Integration)"
docker run --rm \
    --name "$CONTAINER_NAME-test" \
    --cpus 4 \
    --memory 8g \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    cargo test --workspace --release && \
    log_success "Test suite passed" || {
    log_error "Test suite failed"
    exit 1
}
echo ""

# Clippy linting
log_section "3. Clippy Linting"
docker run --rm \
    --name "$CONTAINER_NAME-clippy" \
    --cpus 4 \
    --memory 8g \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c "cargo clippy --workspace -- -D warnings" && \
    log_success "Clippy linting passed" || {
    log_error "Clippy linting failed"
    exit 1
}
echo ""

# Format check
log_section "4. Format Check (rustfmt)"
docker run --rm \
    --name "$CONTAINER_NAME-fmt" \
    --cpus 4 \
    --memory 8g \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c "cargo fmt --all -- --check" && \
    log_success "Format check passed" || {
    log_error "Format check failed"
    exit 1
}
echo ""

# Documentation build
log_section "5. Documentation Build"
docker run --rm \
    --name "$CONTAINER_NAME-doc" \
    --cpus 4 \
    --memory 8g \
    -e RUSTDOCFLAGS="-Dwarnings" \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c "cargo doc --workspace --no-deps" && \
    log_success "Documentation build passed" || {
    log_error "Documentation build failed"
    exit 1
}
echo ""

# MSRV check
log_section "6. MSRV Check (Rust 1.85.0)"
docker run --rm \
    --name "$CONTAINER_NAME-msrv" \
    --cpus 4 \
    --memory 8g \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c "rustup install 1.85.0 && cargo +1.85.0 check --workspace" && \
    log_success "MSRV check passed" || {
    log_error "MSRV check failed"
    exit 1
}
echo ""

# Summary
log_section "CI Test Summary"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ All CI checks passed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Results Summary:"
echo "  ✓ Build test:        PASSED"
echo "  ✓ Test suite:        PASSED"
echo "  ✓ Clippy linting:    PASSED"
echo "  ✓ Format check:      PASSED"
echo "  ✓ Documentation:     PASSED"
echo "  ✓ MSRV check:        PASSED"
echo ""
echo "Next Steps:"
echo "  1. Push to feature branch: git push origin $BRANCH"
echo "  2. Create pull request to main"
echo "  3. GitHub Actions will run additional platform tests (Windows, macOS)"
echo ""
