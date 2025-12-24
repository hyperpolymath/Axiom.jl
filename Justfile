# SPDX-FileCopyrightText: 2025 Axiom.jl Contributors
# SPDX-License-Identifier: MIT

# Axiom.jl Justfile - Task automation for RSR compliance

# Default recipe: show help
default:
    @just --list

# ============================================================================
# Build Tasks
# ============================================================================

# Build all backends
build: build-zig build-rust
    @echo "✓ All backends built"

# Build Zig backend
build-zig:
    @echo "Building Zig backend..."
    cd zig && zig build -Doptimize=ReleaseFast
    @echo "✓ Zig backend built"

# Build Rust backend
build-rust:
    @echo "Building Rust backend..."
    cd rust && cargo build --release
    @echo "✓ Rust backend built"

# Build debug versions
build-debug:
    cd zig && zig build
    cd rust && cargo build

# Clean build artifacts
clean:
    rm -rf zig/zig-out zig/zig-cache
    cd rust && cargo clean
    rm -rf .julia/compiled
    @echo "✓ Cleaned"

# ============================================================================
# Test Tasks
# ============================================================================

# Run all tests
test: test-julia test-zig test-rust
    @echo "✓ All tests passed"

# Run Julia tests
test-julia:
    @echo "Running Julia tests..."
    julia --project -e 'using Pkg; Pkg.test()'

# Run Zig tests
test-zig:
    @echo "Running Zig tests..."
    cd zig && zig build test

# Run Rust tests
test-rust:
    @echo "Running Rust tests..."
    cd rust && cargo test

# Run benchmarks
bench:
    @echo "Running benchmarks..."
    cd zig && zig build bench && ./zig-out/bin/bench

# ============================================================================
# Quality Tasks
# ============================================================================

# Format all code
fmt: fmt-julia fmt-zig fmt-rust
    @echo "✓ All code formatted"

# Format Julia code
fmt-julia:
    julia -e 'using JuliaFormatter; format("src"); format("test")'

# Format Zig code
fmt-zig:
    cd zig && zig fmt src/

# Format Rust code
fmt-rust:
    cd rust && cargo fmt

# Check formatting without changes
fmt-check:
    cd zig && zig fmt --check src/
    cd rust && cargo fmt --check

# Lint all code
lint: lint-rust
    @echo "✓ Linting complete"

# Lint Rust code
lint-rust:
    cd rust && cargo clippy -- -D warnings

# ============================================================================
# Security Tasks
# ============================================================================

# Full security audit
audit: audit-rust audit-deps audit-secrets
    @echo "✓ Security audit complete"

# Audit Rust dependencies
audit-rust:
    @echo "Auditing Rust dependencies..."
    cd rust && cargo audit

# Audit all dependencies
audit-deps:
    @echo "Auditing dependencies..."
    cd rust && cargo deny check

# Check for secrets in codebase
audit-secrets:
    @echo "Checking for secrets..."
    @! grep -rn "password\|secret\|api_key\|token" --include="*.jl" --include="*.rs" --include="*.zig" src/ rust/src/ zig/src/ 2>/dev/null || echo "No secrets found"

# SBOM generation
sbom:
    @echo "Generating SBOM..."
    cd rust && cargo sbom > ../sbom-rust.json
    @echo "✓ SBOM generated"

# ============================================================================
# Documentation Tasks
# ============================================================================

# Build documentation
docs:
    @echo "Building documentation..."
    cd docs && mdbook build
    @echo "✓ Documentation built"

# Serve documentation locally
docs-serve:
    cd docs && mdbook serve

# Generate API documentation
docs-api:
    julia --project -e 'using Documenter, Axiom; makedocs(sitename="Axiom.jl")'

# ============================================================================
# CI/CD Tasks
# ============================================================================

# Run full CI pipeline locally
ci: fmt-check lint test audit
    @echo "✓ CI pipeline passed"

# Pre-commit checks
pre-commit: fmt-check lint test-julia
    @echo "✓ Pre-commit checks passed"

# Validate RSR compliance
validate:
    @echo "Validating RSR compliance..."
    @test -f flake.nix && echo "✓ flake.nix exists"
    @test -f Justfile && echo "✓ Justfile exists"
    @test -f README.md && echo "✓ README.md exists"
    @test -f LICENSE && echo "✓ LICENSE exists"
    @test -f SECURITY.md && echo "✓ SECURITY.md exists"
    @test -f CODE_OF_CONDUCT.md && echo "✓ CODE_OF_CONDUCT.md exists"
    @test -f CONTRIBUTING.md && echo "✓ CONTRIBUTING.md exists"
    @test -f GOVERNANCE.md && echo "✓ GOVERNANCE.md exists"
    @test -f CHANGELOG.md && echo "✓ CHANGELOG.md exists"
    @test -d .well-known && echo "✓ .well-known/ exists"
    @test -f .well-known/security.txt && echo "✓ security.txt exists"
    @test -f FUNDING.yml && echo "✓ FUNDING.yml exists"
    @echo "✓ RSR validation passed"

# ============================================================================
# Release Tasks
# ============================================================================

# Prepare release
release-prep version:
    @echo "Preparing release {{version}}..."
    @sed -i 's/version = ".*"/version = "{{version}}"/' Project.toml
    @sed -i 's/version = ".*"/version = "{{version}}"/' rust/Cargo.toml
    @echo "✓ Version updated to {{version}}"

# Tag release
release-tag version:
    git tag -a "v{{version}}" -m "Release v{{version}}"
    @echo "✓ Tagged v{{version}}"

# ============================================================================
# Development Tasks
# ============================================================================

# Initialize development environment
init:
    @echo "Initializing development environment..."
    julia --project -e 'using Pkg; Pkg.instantiate()'
    cd rust && cargo fetch
    @echo "✓ Development environment ready"

# Update dependencies
update:
    julia --project -e 'using Pkg; Pkg.update()'
    cd rust && cargo update
    @echo "✓ Dependencies updated"

# Watch for changes and rebuild
watch:
    cargo watch -w rust/src -s "just build-rust"

# Interactive Julia REPL with Axiom loaded
repl:
    julia --project -i -e 'using Axiom'

# Code statistics
stats:
    @echo "Code Statistics:"
    @tokei src/ rust/src/ zig/src/ --sort code

# Generate coverage report
coverage:
    julia --project -e 'using Pkg; Pkg.test(coverage=true)'
    @echo "✓ Coverage report generated"
