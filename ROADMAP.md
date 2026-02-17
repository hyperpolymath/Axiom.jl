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
- [x] Complete backend parity and reliability for CPU + Rust + GPU extension paths.
- [x] Harden verification/certificate workflows for repeatable CI and artifact integrity.
- [x] Keep README/wiki claims aligned with tested behavior.

Must execution order (sorted):
1. Backend parity/reliability (CPU + Rust + GPU extensions) - Completed (2026-02-16).
2. Verification/certificate workflow hardening - Completed (2026-02-16).
3. README/wiki claim alignment sweep - Completed (2026-02-16).

Must completion gates:
- [x] Core-op parity tests pass on CPU Julia and Rust backend (matmul, dense, conv, normalization, activations) with documented tolerance budgets.
- [x] GPU extension paths (CUDA/ROCm/Metal) have deterministic CI jobs where hardware is available, and explicit fallback-behavior tests where it is not.
- [x] `instantiate/build/precompile/test` succeeds in CI on supported Julia versions without manual steps.
- [x] Runtime smoke tests for documented examples pass on CPU and at least one accelerated backend.
- [x] No unresolved legacy triad markers (`TO[D]O`/`FIXM[E]`/`TB[D]`) in `src`, `ext`, and `test` for release-scoped areas.
- [x] README/wiki claims are audited against implemented APIs and CI-tested behavior, with roadmap links for deferred features.

Must progress snapshot (2026-02-16):
- [x] CI workflow now runs explicit `instantiate/build/precompile/test` steps across supported Julia versions.
- [x] Added backend parity script with documented tolerance budgets: `test/ci/backend_parity.jl`.
- [x] Added runtime smoke checks for CPU and accelerated paths: `test/ci/runtime_smoke.jl`.
- [x] Added deterministic GPU fallback checks and optional hardware smoke jobs: `test/ci/gpu_fallback.jl`, `test/ci/gpu_hardware_smoke.jl`.
- [x] Added non-GPU accelerator strategy checks and CI coverage: `test/ci/coprocessor_strategy.jl`.
- [x] Added certificate integrity CI checks and digest-report artifacts: `test/ci/certificate_integrity.jl`, `.github/workflows/verify-certificates.yml`.
- [x] Added in-tree gRPC unary protobuf binary-wire support (`application/grpc`) with JSON bridge fallback (`application/grpc+json`).
- [x] Added direct `.pt/.pth/.ckpt` import bridge and expanded ONNX export coverage (Dense/Conv/Norm/Pool + activations).
- [x] Added consolidated readiness gate script for local/CI release checks: `scripts/readiness-check.sh`.

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

Status update (2026-02-17):
- `from_pytorch(...)`: canonical descriptor import + direct `.pt/.pth/.ckpt` bridge shipped (via `scripts/pytorch_to_axiom_descriptor.py`).
- `to_onnx(...)`: exporter now covers Dense/Conv/Norm/Pool + common activations for supported `Sequential`/`Pipeline` models.
- GPU hardening: deterministic fallback CI + optional hardware-smoke CI are in place; compiled GPU wrappers dispatch through extension hooks, and device-range guards are now enforced across CUDA/ROCm/Metal with out-of-range fallback tests.
- Non-GPU accelerator strategy: target backends, detection, compiled dispatch, fallback/strategy CI, and capability/evidence reporting are in place (`coprocessor_capability_report`, `scripts/coprocessor-evidence.jl`); production kernels remain in progress.
- Proof-assistant exports: obligation manifest bundles, certificate markers, and automated assistant reconciliation are shipped (`proof_obligation_manifest`, `export_proof_bundle`, `proof_assistant_obligation_report`, `reconcile_proof_bundle`); full assistant proof replay remains a Stage 4 track.

See `docs/wiki/Roadmap-Commitments.md` for stage mapping and acceptance criteria.

## Definition of Done for "Production Ready"

1. Clean `instantiate/build/precompile/test` on supported Julia versions.
2. Runtime smoke tests for documented examples pass.
3. No unresolved work-marker markers in `src`, `ext`, and `test`.
4. Public docs match implemented APIs and backend support status.
