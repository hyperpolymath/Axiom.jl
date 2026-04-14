# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl - Development Tasks
set shell := ["bash", "-uc"]
set dotenv-load := true

import? "contractile.just"

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

# Run panic-attacker pre-commit scan
assail:
    @command -v panic-attack >/dev/null 2>&1 && panic-attack assail . || echo "panic-attack not found — install from https://github.com/hyperpolymath/panic-attacker"

# ═══════════════════════════════════════════════════════════════════════════════
# ONBOARDING & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

# Check all required toolchain dependencies and report health
doctor:
    #!/usr/bin/env bash
    echo "═══════════════════════════════════════════════════"
    echo "  Axiom.Jl Doctor — Toolchain Health Check"
    echo "═══════════════════════════════════════════════════"
    echo ""
    PASS=0; FAIL=0; WARN=0
    check() {
        local name="$1" cmd="$2" min="$3"
        if command -v "$cmd" >/dev/null 2>&1; then
            VER=$("$cmd" --version 2>&1 | head -1)
            echo "  [OK]   $name — $VER"
            PASS=$((PASS + 1))
        else
            echo "  [FAIL] $name — not found (need $min+)"
            FAIL=$((FAIL + 1))
        fi
    }
    check "just"              just      "1.25" 
    check "git"               git       "2.40" 
    check "Zig"               zig       "0.13" 
    check "Julia"             julia     "1.10" 
# Optional tools
if command -v panic-attack >/dev/null 2>&1; then
    echo "  [OK]   panic-attack — available"
    PASS=$((PASS + 1))
else
    echo "  [WARN] panic-attack — not found (pre-commit scanner)"
    WARN=$((WARN + 1))
fi
    echo ""
    echo "  Result: $PASS passed, $FAIL failed, $WARN warnings"
    if [ "$FAIL" -gt 0 ]; then
        echo "  Run 'just heal' to attempt automatic repair."
        exit 1
    fi
    echo "  All required tools present."

# Attempt to automatically install missing tools
heal:
    #!/usr/bin/env bash
    echo "═══════════════════════════════════════════════════"
    echo "  Axiom.Jl Heal — Automatic Tool Installation"
    echo "═══════════════════════════════════════════════════"
    echo ""
if ! command -v just >/dev/null 2>&1; then
    echo "Installing just..."
    cargo install just 2>/dev/null || echo "Install just from https://just.systems"
fi
    echo ""
    echo "Heal complete. Run 'just doctor' to verify."

# Guided tour of the project structure and key concepts
tour:
    #!/usr/bin/env bash
    echo "═══════════════════════════════════════════════════"
    echo "  Axiom.Jl — Guided Tour"
    echo "═══════════════════════════════════════════════════"
    echo ""
    echo '// SPDX-License-Identifier: MPL-2.0'
    echo ""
    echo "Key directories:"
    echo "  src/                      Source code" 
    echo "  ffi/                      Foreign function interface (Zig)" 
    echo "  docs/                     Documentation" 
    echo "  tests/                    Test suite" 
    echo "  test/                     Test suite" 
    echo "  .github/workflows/        CI/CD workflows" 
    echo "  contractiles/             Must/Trust/Dust contracts" 
    echo "  .machine_readable/        Machine-readable metadata" 
    echo "  examples/                 Usage examples" 
    echo ""
    echo "Quick commands:"
    echo "  just doctor    Check toolchain health"
    echo "  just heal      Fix missing tools"
    echo "  just help-me   Common workflows"
    echo "  just default   List all recipes"
    echo ""
    echo "Read more: README.adoc, EXPLAINME.adoc"

# Show help for common workflows
help-me:
    #!/usr/bin/env bash
    echo "═══════════════════════════════════════════════════"
    echo "  Axiom.Jl — Common Workflows"
    echo "═══════════════════════════════════════════════════"
    echo ""
echo "FIRST TIME SETUP:"
echo "  just doctor           Check toolchain"
echo "  just heal             Fix missing tools"
echo "" 
echo "PRE-COMMIT:"
echo "  just assail           Run panic-attacker scan"
echo ""
echo "LEARN:"
echo "  just tour             Guided project tour"
echo "  just default          List all recipes" 


# Print the current CRG grade (reads from READINESS.md '**Current Grade:** X' line)
crg-grade:
    @grade=$$(grep -oP '(?<=\*\*Current Grade:\*\* )[A-FX]' READINESS.md 2>/dev/null | head -1); \
    [ -z "$$grade" ] && grade="X"; \
    echo "$$grade"

# Generate a shields.io badge markdown for the current CRG grade
# Looks for '**Current Grade:** X' in READINESS.md; falls back to X
crg-badge:
    @grade=$$(grep -oP '(?<=\*\*Current Grade:\*\* )[A-FX]' READINESS.md 2>/dev/null | head -1); \
    [ -z "$$grade" ] && grade="X"; \
    case "$$grade" in \
      A) color="brightgreen" ;; B) color="green" ;; C) color="yellow" ;; \
      D) color="orange" ;; E) color="red" ;; F) color="critical" ;; \
      *) color="lightgrey" ;; esac; \
    echo "[![CRG $$grade](https://img.shields.io/badge/CRG-$$grade-$$color?style=flat-square)](https://github.com/hyperpolymath/standards/tree/main/component-readiness-grades)"
