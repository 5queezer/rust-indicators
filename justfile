# Rust Indicators Project Justfile
# ================================

# Default command: Run comprehensive test suite
default: test-all

# Run all test variations in sequence
test-all:
    @echo "Running comprehensive test suite..."
    @echo "1/4: Running standard tests..."
    cargo test
    @echo "2/4: Running tests with GPU features..."
    cargo test --features gpu
    @echo "3/4: Running tests with CUDA features..."
    cargo test --features cuda
    @echo "4/4: Running tests with no default features..."
    cargo test --no-default-features
    @echo "All tests completed!"

# Run the VPIN benchmark comparing CPU vs CUDA performance
benchmark:
    @echo "Running VPIN benchmark (CPU vs CUDA)..."
    @echo "Note: Requires CUDA feature to be meaningful"
    cargo run --bin simple_vpin_benchmark --features cuda

# Validate backend selection logic
validate:
    @echo "Running backend selection validation..."
    @echo "This tests the backend selection logic without Python dependencies"
    cargo run --bin validate_backend

# Individual test commands for convenience
test:
    cargo test

test-gpu:
    cargo test --features gpu

test-cuda:
    cargo test --features cuda

test-no-default:
    cargo test --no-default-features

# Build commands
build:
    cargo build

build-release:
    cargo build --release

# Clean build artifacts
clean:
    cargo clean

# Check code without building
check:
    cargo check

# Format code
fmt:
    cargo fmt

# Run clippy linter
clippy:
    cargo clippy

# Show available commands
help:
    @just --list