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
- `AXIOM_READINESS_RUN_COULD_PACKAGING=0` disables model package/registry could-item checks.
- `AXIOM_READINESS_RUN_COULD_OPTIMIZATION=0` disables optimization-pass could-item checks.
- `AXIOM_READINESS_RUN_COULD_TELEMETRY=0` disables verification-telemetry could-item checks.
- `AXIOM_READINESS_ALLOW_SKIPS=1` allows skipped checks without failing the run.
- `AXIOM_COPROCESSOR_SELF_HEAL=0` disables coprocessor self-healing fallback (useful for failure-path testing).
- `AXIOM_TPU_REQUIRED=1` enforces strict TPU mode (no compile/runtime fallback).
- `AXIOM_NPU_REQUIRED=1` enforces strict NPU mode (no compile/runtime fallback).
- `AXIOM_DSP_REQUIRED=1` enforces strict DSP mode (no compile/runtime fallback).
- `AXIOM_MATH_REQUIRED=1` enforces strict MATH mode (no compile/runtime fallback).
- `AXIOM_COPROCESSOR_REQUIRED=1` enforces strict mode for all coprocessor backends unless backend-specific flags override.
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

Run deterministic strategy/fallback checks for TPU/NPU/PPU/MATH/FPGA/DSP targets:

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

## Could: Model Packaging + Registry

Run packaging/registry workflow checks:

```bash
julia --project=. test/ci/model_package_registry.jl
```

Generate machine-readable package/registry evidence artifacts:

```bash
julia --project=. scripts/model-package-evidence.jl
```

## Could: Optimization Pass Evidence

Run optimization pass behavior checks:

```bash
julia --project=. test/ci/optimization_passes.jl
```

Generate optimization benchmark/drift evidence:

```bash
julia --project=. scripts/optimization-evidence.jl
```

## Could: Verification Telemetry

Run structured telemetry coverage checks:

```bash
julia --project=. test/ci/verification_telemetry.jl
```

Generate verification telemetry evidence artifact:

```bash
julia --project=. scripts/verification-telemetry-evidence.jl
```

## TPU/NPU/DSP/MATH Strict Mode (Production Gate)

Run strict-mode behavior checks for TPU/NPU/DSP/MATH fallback blocking and hook requirements:

```bash
julia --project=. test/ci/tpu_required_mode.jl
julia --project=. test/ci/npu_required_mode.jl
julia --project=. test/ci/dsp_required_mode.jl
julia --project=. test/ci/math_required_mode.jl
```

Generate machine-readable TPU/NPU/DSP/MATH strict-mode evidence:

```bash
julia --project=. scripts/tpu-strict-evidence.jl
julia --project=. scripts/npu-strict-evidence.jl
julia --project=. scripts/dsp-strict-evidence.jl
julia --project=. scripts/math-strict-evidence.jl
```

Use `templates/AxiomTPUExtSkeleton.jl`, `templates/AxiomNPUExtSkeleton.jl`, `templates/AxiomDSPExtSkeleton.jl`, and `templates/AxiomMathExtSkeleton.jl` as starters for extension hook overrides.

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
