# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl - Development Tasks
set shell := ["bash", "-uc"]
set dotenv-load := true

project := "Axiom.jl"

# Show all recipes
default:
    @just --list --unsorted

# Run Julia test suite
test:
    julia --project=. -e 'using Pkg; Pkg.test()'

# Run tests with verbose output
test-verbose:
    julia --project=. test/runtests.jl

# Instantiate project dependencies
deps:
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Update dependencies
update:
    julia --project=. -e 'using Pkg; Pkg.update()'

# Start Julia REPL with project loaded
repl:
    julia --project=. -e 'using Axiom; println("Axiom.jl v$(Axiom.VERSION) loaded")'

# Run benchmarks
bench:
    julia --project=. benchmark/benchmarks.jl

# Build Zig backend (.so)
build-zig:
    cd zig && zig build -Doptimize=ReleaseFast
    @echo "Zig library built at zig/zig-out/lib/libaxiom_zig.so"

# Build all native backends
build-backends: build-zig

# Run with Zig backend
run-zig: build-zig
    AXIOM_ZIG_LIB=zig/zig-out/lib/libaxiom_zig.so julia --project=. -e 'using Axiom; println("Backend: ", typeof(current_backend()))'

# Check code quality
lint:
    @echo "Checking editorconfig..."
    editorconfig-checker || true
    @echo "Checking SPDX headers..."
    grep -rL "SPDX-License-Identifier" src/ --include="*.jl" || echo "All files have SPDX headers"

# Clean build artifacts
clean:
    rm -rf zig/zig-out zig/zig-cache result
    @echo "Build artifacts cleaned"

# Run panic-attack security scan
scan:
    panic-attack assail . --output /tmp/axiom-jl-scan.json
    @echo "Scan results at /tmp/axiom-jl-scan.json"

# Pre-commit checks
pre-commit: test lint
    @echo "All checks passed!"

# Show project status
status:
    @echo "=== Axiom.jl Status ==="
    @echo "Julia version:"
    @julia --version
    @echo ""
    @echo "Zig backend:"
    @test -f zig/zig-out/lib/libaxiom_zig.so && echo "  Built" || echo "  Not built (run: just build-zig)"
