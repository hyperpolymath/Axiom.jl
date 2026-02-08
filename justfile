# SPDX-License-Identifier: PMPL-1.0-or-later
# justfile - Axiom.jl
# Julia library with Rust backend for axiomatic computation
set shell := ["bash", "-uc"]

project := "Axiom.jl"

# Show all recipes
default:
    @just --list --unsorted

# Build with nix
build:
    nix build

# Build and show output path
build-show:
    nix build --print-out-paths

# Enter dev shell
develop:
    nix develop

# Check flake
check:
    nix flake check

# Update flake inputs
update:
    nix flake update

# Show flake info
info:
    nix flake info

# Format nix files
fmt:
    nixfmt *.nix || nix fmt

# Run nix linter
lint:
    statix check . || true

# Clean
clean:
    rm -rf result

# Show derivation
show-drv:
    nix derivation show

# Run tests
test: check

# All pre-commit checks
pre-commit: check
    @echo "All checks passed!"
