# Axiom.jl Roadmap

## Current Baseline (v1.0.0)

Axiom.jl provides:
- Core tensor/layer pipeline in Julia
- Verification checks (`@ensure`, property checks, certificates)
- Optional Rust backend integration hooks
- GPU extension hooks for CUDA/ROCm/Metal

Current focus is stabilization: production-grade build/test/runtime reliability, accurate docs, and explicit feature status.

## Near-Term Plan

### Must
- [ ] Complete backend parity and reliability for CPU + Rust + GPU extension paths.
- [ ] Harden verification/certificate workflows for repeatable CI and artifact integrity.
- [ ] Keep README/wiki claims aligned with tested behavior.

Must completion gates:
- [ ] Core-op parity tests pass on CPU Julia and Rust backend (matmul, dense, conv, normalization, activations) with documented tolerance budgets.
- [ ] GPU extension paths (CUDA/ROCm/Metal) have deterministic CI jobs where hardware is available, and explicit fallback-behavior tests where it is not.
- [ ] `instantiate/build/precompile/test` succeeds in CI on supported Julia versions without manual steps.
- [ ] Runtime smoke tests for documented examples pass on CPU and at least one accelerated backend.
- [ ] No unresolved `TODO`/`FIXME`/`TBD` markers in `src`, `ext`, and `test` for release-scoped areas.

### Should
- [ ] Improve performance benchmarking and regression tracking across backends.
- [ ] Expand verification property coverage and diagnostics.
- [ ] Strengthen release automation and compatibility testing.

Should completion gates:
- [ ] Baseline benchmark suite published for CPU Julia, Rust, and GPU extension paths with trend tracking.
- [ ] Verification diagnostics include actionable counterexample metadata and failure categorization.
- [ ] Compatibility matrix is validated across OS/Julia combinations used by supported deployments.
- [ ] Release process produces versioned artifacts and changelog validation automatically.

### Could
- [ ] Add richer model packaging and registry workflows.
- [ ] Expand advanced optimization passes (fusion, mixed precision).
- [ ] Add deeper observability tooling for runtime verification paths.

Could completion gates:
- [ ] Packaging format includes model metadata, verification claims, and reproducible hashes.
- [ ] Optional optimization passes are benchmarked and guarded behind explicit flags.
- [ ] Verification runtime emits structured telemetry suitable for dashboards/incident analysis.

## Deferred Commitments (Tracked)

These are intentionally tracked roadmap promises, not removed features:

1. `from_pytorch(...)` import API
2. `to_onnx(...)` export API
3. Production-hardened GPU paths across CUDA/ROCm/Metal
4. Non-GPU accelerators (TPU/NPU/DSP/FPGA) backend strategy
5. Proof-assistant export improvements beyond skeleton artifacts

See `docs/wiki/Roadmap-Commitments.md` for stage mapping and acceptance criteria.

## Definition of Done for "Production Ready"

1. Clean `instantiate/build/precompile/test` on supported Julia versions.
2. Runtime smoke tests for documented examples pass.
3. No unresolved work-marker markers in `src`, `ext`, and `test`.
4. Public docs match implemented APIs and backend support status.
