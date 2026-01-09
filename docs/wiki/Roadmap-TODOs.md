# Roadmap and TODOs

This page consolidates planned work and in-code TODOs for Axiom.jl. It is a living snapshot; see the referenced files for detail.

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

## Open TODOs in Code

- SMT solver integration: `src/dsl/prove.jl`.
- Proof serialization: `src/dsl/prove.jl`.
- Optimization passes (fusion, mixed precision, Float16): `src/backends/abstract.jl`, `src/dsl/pipeline.jl`.
- Rust codegen/compile hooks: `src/backends/abstract.jl`.
- CUDA op lowering: `src/backends/abstract.jl`.

## Maintainer Coverage Gaps

Maintainership is marked TBD in:
- Core Julia implementation
- Rust backend
- Zig backend
- Verification (@ensure, @prove, certificates)
- Documentation
- CI/CD

Source: `MAINTAINERS.md`.
