# Roadmap and Tracked Commitments

This page tracks roadmap commitments for Axiom.jl and explicitly records feature promises that are not yet implemented.

## Deferred Promises from README/Wiki

These items were presented as capabilities in earlier docs but are not yet stable public APIs. They are now tracked as roadmap commitments.

| Commitment | Status | Target Stage | Acceptance Criteria |
|---|---|---|---|
| `from_pytorch(...)` model import | Planned | Stage 3 | Load representative PyTorch checkpoints, parity tests vs source model, documented API and error handling |
| `to_onnx(...)` export | Planned | Stage 3 | Export verified models to ONNX with shape/property metadata and round-trip smoke test |
| CPU + Rust + GPU extension backend parity/reliability | In progress (CI gates landed) | Stage 2 | Core-op parity tests on CPU+Rust, deterministic GPU extension tests (or fallback tests where unavailable), and CI/runtime smoke coverage |
| TPU/NPU/DSP/FPGA backends | In progress (core targets shipped) | Stage 5+ | At least one production-grade non-GPU accelerator backend with CI coverage and benchmark evidence |
| REST/gRPC/GraphQL serving parity | In progress | Stage 2-3 | REST + GraphQL runtime endpoints stable, gRPC network runtime integrated against generated proto contracts |
| GPU production hardening (CUDA/ROCm/Metal) | In progress (fallback + optional hardware CI landed) | Stage 2 | Extension-backed kernels, deterministic tests, fallback behavior, and backend-specific performance baselines |
| Verification/certificate workflow hardening | In progress (integrity CI landed) | Stage 2 | Repeatable certificate serialization checks, tamper-detection tests, and digest-report artifacts in CI |
| Proof assistant export without manual placeholders | Planned | Stage 4 | Export artifacts that include machine-checkable proof obligations and explicit proof status metadata |

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
- Establish GPU abstraction hooks (CUDA/ROCm planned).
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
- Zig backend
- Verification (@ensure, @prove, certificates)
- Documentation
- CI/CD

Source: `MAINTAINERS.md`.
