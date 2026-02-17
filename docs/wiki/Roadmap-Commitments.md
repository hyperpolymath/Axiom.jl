# Roadmap and Tracked Commitments

This page tracks roadmap commitments for Axiom.jl and records what is already shipped versus what remains to reach full production scope.

## Tracked Commitments from README/Wiki

These items are roadmap promises with explicit implementation status and acceptance criteria.

| Commitment | Status | Target Stage | Acceptance Criteria |
|---|---|---|---|
| `from_pytorch(...)` model import | Baseline shipped (descriptor + direct `.pt/.pth/.ckpt` bridge + CI interop smoke) | Stage 3 | Keep bridge stable across representative checkpoints and maintain parity validation matrix |
| `to_onnx(...)` export | Baseline shipped (Dense/Conv/Norm/Pool export + deterministic/CI interop smoke coverage) | Stage 3 | Expand operator coverage and add interoperability/round-trip validation matrix |
| CPU + Rust + GPU extension backend parity/reliability | Baseline shipped (CI gates + readiness gate) | Stage 2 | Core-op parity tests on CPU+Rust, deterministic GPU extension tests (or fallback tests where unavailable), and CI/runtime smoke coverage |
| TPU/NPU/PPU/MATH/FPGA/DSP backends | In progress (targets + strategy CI + capability/evidence reporting + runtime self-healing diagnostics/resilience evidence + TPU/NPU/DSP/MATH strict-mode gates shipped) | Stage 5+ | At least one production-grade non-GPU accelerator backend with strict-mode fallback controls, CI coverage, resilience diagnostics, and benchmark evidence |
| Accelerator rollout sequence | In progress (Maths/Physics basics shipped; cryptographic coprocessor + FPGA production-ready next; VPU/QPU basics queued; remaining production hardening in v2) | Stage 5+ | Keep roadmap/docs/CI aligned with the sequencing policy and close each stage with evidence artifacts |
| REST/gRPC/GraphQL serving parity | In progress (REST/GraphQL + gRPC binary/json bridge shipped) | Stage 2-3 | External runtime interoperability/perf hardening and expanded conformance coverage |
| GPU production hardening (CUDA/ROCm/Metal) | Baseline shipped (fallback + optional hardware CI + compiled hook dispatch + device-range guards + self-healing/runtime diagnostics + performance evidence artifacts) | Stage 2 | Extension-backed kernels, deterministic tests, fallback behavior, and backend-specific performance baselines |
| Verification/certificate workflow hardening | Baseline shipped (integrity CI + readiness gate) | Stage 2 | Repeatable certificate serialization checks, tamper-detection tests, and digest-report artifacts in CI |
| Proof assistant export beyond skeleton artifacts | In progress (obligation manifest + status metadata + automated reconciliation + deterministic CI/evidence coverage shipped) | Stage 4 | Preserve machine-checkable proof obligations/status metadata, keep reconciliation/evidence CI green, and complete assistant proof replay |
| Model packaging + registry workflow | Baseline shipped (`MODEL_PACKAGE_FORMAT`, `export_model_package`, `build_registry_entry`, CI + evidence artifacts) | Stage 3 | Package manifests carry metadata/claims/hashes and registry entries can be generated deterministically |
| Optimization evidence track | Baseline shipped (explicit optimization flags + CI/evidence coverage) | Stage 2-3 | Keep optimization passes behind explicit flags and publish drift/latency evidence for each release |
| Verification telemetry observability | Baseline shipped (`verification_result_telemetry`, `verification_telemetry_report`, CI/evidence artifacts) | Stage 2-3 | Structured telemetry suitable for dashboards/incident workflows is emitted and tested |

Latest readiness verification (2026-02-17): `scripts/readiness-check.sh` => Passed: 29, Failed: 0, Skipped: 0.

If roadmap wording and README/wiki claims diverge, the roadmap is the source of truth.

## Release Roadmap

- v0.1: Core framework, DSL, verification basics (shipped).
- v0.2: Full Rust backend, GPU support.
- v0.3: Hugging Face integration, model zoo.
- v0.4: Advanced proofs, SMT integration.
- v1.0: Production readiness, industry certifications.

Source: `README.adoc`.

## Prioritized, Staged Plan

### Stage 1: Verification Foundations (Now -> v0.2)

- Finish SMT integration hardening (timeouts, caching, Rust runner optional).
- Implement proof serialization for audit trails.
- Expand property coverage in `@prove` with clearer error reporting.

### Stage 2: Backend Parity and Performance (v0.2)

- Rust backend feature parity for core ops (matmul, conv, norms, activations).
- Maintain GPU abstraction hooks and deterministic fallback/hardware CI coverage (CUDA/ROCm/Metal).
- Add benchmarking + regressions for Rust vs Julia.

### Stage 3: Ecosystem and Model Zoo (v0.3)

- Hugging Face import integration.
- Curated model zoo with verification-ready templates.
- Packaging for reuse (pretrained weights + metadata).

### Stage 4: Advanced Proofs (v0.4)

- Extend SMT properties (quantifiers, non-linear properties where feasible).
- Proof certificates export and verification workflow.
- Formal proof tooling integration (long-term research track).

### Stage 5: Production Readiness (v1.0)

- Security hardening, sandboxed solver execution.
- Release engineering, CI/CD maturity, signed artifacts.
- Compliance and industry certification readiness.

## Planned Architecture Enhancements

- GPU backends (CUDA/ROCm).
- Distributed training.
- Quantization (INT8/INT4).
- Sparse tensor ops.
- JIT kernel fusion.

Source: `docs/wiki/Architecture.md`.

## Planned Ecosystem Integrations

- Model zoo expansion (audio, reinforcement learning).
- TensorRT, CoreML, Edge TPU.
- MLflow, Weights & Biases.

Source: `docs/wiki/Framework-Comparison.md`.

## Backend Roadmap

- CUDA and Metal Rust backends (planned).
- GPU performance and kernel optimization.

Source: `docs/wiki/Rust-Backend.md`.

## Open Engineering Work Items

- SMT solver integration hardening: `src/dsl/prove.jl`.
- Optimization passes (fusion, mixed precision, Float16): `src/backends/abstract.jl`, `src/dsl/pipeline.jl`.
- Rust codegen/compile hooks: `src/backends/abstract.jl`.
- CUDA op lowering and GPU kernel parity: `src/backends/abstract.jl`.

## Maintainer Coverage Gaps

Maintainership is marked Unassigned in:
- Core Julia implementation
- Rust backend
- Legacy prototype cleanup and de-scoping docs
- Verification (@ensure, @prove, certificates)
- Documentation
- CI/CD

Source: `MAINTAINERS.md`.
