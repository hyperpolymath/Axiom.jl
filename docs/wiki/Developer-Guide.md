# Developer Guide

This guide defines the development workflow and release gates for Axiom.jl.

## Prerequisites

- Julia `1.10+`
- Git
- Optional for backend work:
  - Rust toolchain
  - CUDA / ROCm / Metal runtimes

## Local Setup

```bash
cd Axiom.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Build and Test

Run the full baseline pipeline before merging:

```bash
julia --project=. -e 'using Pkg; Pkg.build(); Pkg.precompile(); Pkg.test()'
```

## One-Command Readiness Gate

Run the consolidated release-readiness checks:

```bash
./scripts/readiness-check.sh
```

Useful toggles:

- `AXIOM_READINESS_RUN_RUST=0` disables Rust parity/smoke checks.
- `AXIOM_READINESS_RUN_COPROCESSOR=0` disables coprocessor strategy/resilience checks.
- `AXIOM_READINESS_RUN_GPU_PERF=0` disables GPU resilience/performance evidence checks.
- `AXIOM_READINESS_ALLOW_SKIPS=1` allows skipped checks without failing the run.
- `AXIOM_COPROCESSOR_SELF_HEAL=0` disables coprocessor self-healing fallback (useful for failure-path testing).
- `JULIA_BIN=/path/to/julia` selects a specific Julia binary.

## Runtime Smoke Tests

Run quick runtime checks after unit tests:

```bash
julia --project=. test/ci/runtime_smoke.jl
```

## Backend Parity (CPU vs Rust)

Run parity checks when Rust backend is available:

```bash
cargo build --release --manifest-path rust/Cargo.toml
AXIOM_RUST_LIB=$PWD/rust/target/release/libaxiom_core.so julia --project=. test/ci/backend_parity.jl
```

Tolerance budgets used by CI parity checks:

- `matmul`: `atol=1e-4`, `rtol=1e-4`
- `dense`: `atol=1e-4`, `rtol=1e-4`
- `conv2d`: `atol=2e-4`, `rtol=2e-4`
- `normalization`: `atol=1e-4`, `rtol=1e-4`
- `activations`: `atol=1e-5`, `rtol=1e-5`

## GPU Fallback and Hardware Smoke

Run explicit fallback checks (no GPU required):

```bash
julia --project=. test/ci/gpu_fallback.jl
```

Run hardware smoke checks on GPU runners:

```bash
AXIOM_GPU_BACKEND=cuda AXIOM_GPU_REQUIRED=1 julia --project=. test/ci/gpu_hardware_smoke.jl
AXIOM_GPU_BACKEND=rocm AXIOM_GPU_REQUIRED=1 julia --project=. test/ci/gpu_hardware_smoke.jl
AXIOM_GPU_BACKEND=metal AXIOM_GPU_REQUIRED=1 julia --project=. test/ci/gpu_hardware_smoke.jl
```

Run GPU resilience diagnostics and generate machine-readable baseline evidence:

```bash
julia --project=. test/ci/gpu_resilience.jl
julia --project=. scripts/gpu-performance-evidence.jl
```

Generate backend-specific evidence on dedicated hardware runners:

```bash
AXIOM_GPU_BASELINE_BACKEND=cuda AXIOM_GPU_REQUIRED=1 julia --project=. scripts/gpu-performance-evidence.jl
AXIOM_GPU_BASELINE_BACKEND=rocm AXIOM_GPU_REQUIRED=1 julia --project=. scripts/gpu-performance-evidence.jl
AXIOM_GPU_BASELINE_BACKEND=metal AXIOM_GPU_REQUIRED=1 julia --project=. scripts/gpu-performance-evidence.jl
```

`scripts/gpu-performance-evidence.jl` reads optional thresholds from:

- `AXIOM_GPU_BASELINE_PATH` (default: `benchmark/gpu_performance_baseline.json`)
- `AXIOM_GPU_BASELINE_ENFORCE` (set to `1` to fail on regressions vs baseline)
- `AXIOM_GPU_MAX_REGRESSION_RATIO` (default: `1.20`)

## Non-GPU Coprocessor Strategy

Run deterministic strategy/fallback checks for TPU/NPU/DSP/FPGA targets:

```bash
julia --project=. test/ci/coprocessor_strategy.jl
```

Generate a machine-readable capability/evidence artifact:

```bash
julia --project=. scripts/coprocessor-evidence.jl
```

Run coprocessor resilience diagnostics (self-healing + fault-tolerance counters):

```bash
julia --project=. test/ci/coprocessor_resilience.jl
```

Generate coprocessor resilience evidence artifact:

```bash
julia --project=. scripts/coprocessor-resilience-evidence.jl
```

## Certificate Integrity Checks

Run certificate reproducibility and tamper-detection checks:

```bash
julia --project=. test/ci/certificate_integrity.jl
```

## Proof-Assistant Bundle Reconciliation

When Lean/Coq/Isabelle artifacts are updated, reconcile manifest status:

```bash
julia --project=. -e 'using Axiom; reconcile_proof_bundle("build/proofs/my_model.obligations.json")'
```

Run deterministic CI reconciliation coverage:

```bash
julia --project=. test/ci/proof_bundle_reconciliation.jl
```

Generate proof bundle evidence for CI artifacts/review:

```bash
julia --project=. scripts/proof-bundle-evidence.jl
```

## Quality Gates

Use these checks to keep production paths clean:

```bash
rg -n "TO[D]O|FIXM[E]|TB[D]|OPEN_ITEM|FIX_ITEM|XXX|HACK" src test ext
```

If a marker is required (for templates or roadmap planning), keep it out of production code paths (`src`, `ext`, `test`) and explain it in review notes.

## Documentation Expectations

- User-facing behavior changes require updates in:
  - `docs/wiki/User-Guide.md`
  - `docs/wiki/Home.md`
- Developer workflow changes require updates in:
  - `docs/wiki/Developer-Guide.md`
- Claims in README/wiki must match tested behavior.

## Release Checklist

1. `Pkg.build`, `Pkg.precompile`, and `Pkg.test` pass on a clean environment.
2. Runtime smoke checks pass.
3. No unresolved production work-marker markers in `src`, `ext`, or `test`.
4. CPU vs Rust parity and certificate integrity CI checks pass.
5. README/wiki claims are aligned with actual implementation status.
6. Version metadata is consistent (`Project.toml` and `Axiom.VERSION`).
