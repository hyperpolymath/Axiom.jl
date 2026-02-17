;; SPDX-License-Identifier: PMPL-1.0-or-later
;; META.scm - Project metadata and architectural decisions for Axiom.jl

(define project-meta
  `((version . "1.1.0")
    (architecture-decisions
      ((id . "adr-001")
       (title . "Fallback-first accelerator design")
       (status . "accepted")
       (decision . "All accelerator paths must expose deterministic fallback behavior with strict-mode overrides.")
       (evidence . ("test/ci/gpu_fallback.jl" "test/ci/coprocessor_strategy.jl" "test/ci/coprocessor_resilience.jl")))
      ((id . "adr-002")
       (title . "Machine-readable evidence artifacts")
       (status . "accepted")
       (decision . "Each major reliability track emits structured evidence JSON for CI/review pipelines.")
       (evidence . ("scripts/gpu-performance-evidence.jl"
                    "scripts/coprocessor-evidence.jl"
                    "scripts/proof-bundle-evidence.jl"
                    "scripts/model-package-evidence.jl"
                    "scripts/optimization-evidence.jl"
                    "scripts/verification-telemetry-evidence.jl")))
      ((id . "adr-003")
       (title . "Roadmap sequence for non-GPU accelerators")
       (status . "accepted")
       (decision . "Maths/Physics basics first, then cryptographic coprocessor + FPGA production-ready, then VPU/QPU basics, remaining production hardening in v2.")
       (evidence . ("ROADMAP.adoc" "ROADMAP.md" "docs/wiki/Roadmap-Commitments.md"))))
    (development-practices
      ((code-style . "julia+rust standard")
       (security . "fallback-safe + integrity checks + signed release workflow")
       (versioning . "semver")
       (documentation . "asciidoc + markdown + machine-readable scm")
       (branching . "trunk-based")))
    (design-rationale
      ((dependability . "Prefer explicit fallbacks, strict-mode opt-in, and telemetry over silent failure.")
       (interoperability . "Keep PyTorch/ONNX and serving APIs aligned with deterministic CI smoke tests.")
       (governance . "Roadmap + readiness gate + evidence artifacts are source-of-truth for release claims.")))))
