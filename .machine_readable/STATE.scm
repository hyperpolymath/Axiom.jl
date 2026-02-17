;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Project state tracking for Axiom.jl

(define-module (state axiom)
  #:use-module (ice-9 match)
  #:export (state get-completion-percentage get-blockers get-milestone))

(define state
  '((metadata
      (version . "1.1.0")
      (schema-version . "1.0.0")
      (created . "2025-01-23")
      (updated . "2026-02-17")
      (project . "Axiom.jl")
      (repo . "https://github.com/hyperpolymath/Axiom.jl"))

    (project-context
      (name . "Axiom.jl")
      (tagline . "Provably correct machine learning with compile-time verification")
      (tech-stack . ("Julia" "Rust" "SMT solvers"))
      (target-platforms . ("Linux" "macOS" "Windows")))

    (current-position
      (phase . "stabilization-and-roadmap-execution")
      (overall-completion . 88)
      (components
        ((name . "CPU + Rust + GPU extension reliability")
         (status . "baseline-shipped")
         (completion . 95)
         (notes . "Parity/smoke CI + readiness gate + fallback diagnostics shipped"))
        ((name . "Verification and certificate integrity")
         (status . "baseline-shipped")
         (completion . 92)
         (notes . "Certificate integrity CI + proof-bundle reconciliation/evidence shipped"))
        ((name . "PyTorch import + ONNX export")
         (status . "baseline-shipped")
         (completion . 85)
         (notes . "Descriptor + direct checkpoint bridge and ONNX baseline coverage in CI"))
        ((name . "Coprocessor strategy (TPU/NPU/PPU/MATH/FPGA/DSP)")
         (status . "in-progress")
         (completion . 72)
         (notes . "Detection/dispatch/fallback diagnostics/resilience + strict TPU/NPU/DSP gates shipped; production kernels remain"))
        ((name . "Model packaging + registry workflow")
         (status . "baseline-shipped")
         (completion . 82)
         (notes . "Deterministic package manifest + registry entry APIs with CI/evidence"))
        ((name . "Optimization pass evidence")
         (status . "baseline-shipped")
         (completion . 78)
         (notes . "Explicit optimize/precision paths with CI and benchmark/drift evidence"))
        ((name . "Verification telemetry observability")
         (status . "baseline-shipped")
         (completion . 80)
         (notes . "Structured run payload + summary telemetry APIs with CI/evidence"))
        ((name . "Proof assistant replay")
         (status . "in-progress")
         (completion . 62)
         (notes . "Manifest/bundle/reconciliation shipped; full assistant replay remains Stage 4"))
        ((name . "@prove SMT integration")
         (status . "partial")
         (completion . 58)
         (notes . "Core structure exists; advanced solver integration/hardening remains"))))

      (working-features
        "Compile-time shape verification"
        "Core layers + training stack"
        "CPU/Rust runtime parity"
        "GPU fallback + resilience diagnostics"
        "Non-GPU coprocessor strategy/fallback diagnostics"
        "TPU/NPU/DSP strict-mode gates"
        "PyTorch checkpoint bridge + ONNX baseline export"
        "Proof assistant obligation manifests + reconciliation"
        "Model package + registry manifests with reproducible hashes"
        "Structured verification telemetry APIs"))

    (route-to-mvp
      (milestones
        ((name . "Must gates")
         (target-date . "2026-02-16")
         (status . "completed")
         (items
           "CPU + Rust + GPU extension reliability baseline"
           "Verification/certificate hardening baseline"
           "README/wiki claim alignment baseline"))
        ((name . "Could baselines")
         (target-date . "2026-02-17")
         (status . "completed")
         (items
           "Model package + registry manifests"
           "Optimization pass CI + evidence"
           "Verification telemetry CI + evidence"))
        ((name . "Accelerator rollout")
         (target-date . "2026-03-31")
         (status . "in-progress")
         (items
           "Maths/Physics basics shipped"
           "Cryptographic coprocessor + FPGA production-ready next"
           "VPU + QPU basics next"
           "Remaining accelerator production hardening in v2"))
        ((name . "Proof assistant Stage 4")
         (target-date . "2026-04-30")
         (status . "in-progress")
         (items
           "Keep bundle reconciliation CI green"
           "Complete assistant proof replay beyond skeleton artifacts"))
        ((name . "JuliaHub readiness")
         (target-date . "2026-05-15")
         (status . "in-progress")
         (items
           "Pass add/precompile/build/test/runtime/formalism gates"
           "Freeze release docs and machine-readable state"
           "Publish tagged release artifacts"))))

    (blockers-and-issues
      (critical
        ())
      (high
        ("Coprocessor production kernels (crypto/FPGA first in current sequence)"
         "Proof assistant full replay remains incomplete"))
      (medium
        ("Advanced SMT hardening for @prove track"
         "Compatibility/perf matrix expansion for release automation"))
      (low
        ("Model zoo expansion"
         "Quantization/distributed training tracks"))))

    (critical-next-actions
      (immediate
        "Complete cryptographic coprocessor production-ready baseline"
        "Complete FPGA production-ready baseline"
        "Keep roadmap/docs/machine state aligned with shipped CI evidence")
      (this-week
        "Land VPU and QPU basics (non-production)"
        "Run full readiness gate including could-item evidence scripts"
        "Stabilize release notes + roadmap commitment tables")
      (this-month
        "Advance proof assistant replay stage"
        "Expand compatibility/performance matrix toward JuliaHub release gate"
        "Prepare release candidate docs/artifacts"))

    (session-history
      ((date . "2026-02-17")
       (accomplishments
         "Coprocessor strategy/resilience + strict TPU/NPU/DSP gates verified green"
         "Added PPU/Math basics in strategy detection/capability/resilience coverage"
         "Added model package/registry, optimization evidence, and verification telemetry baselines"
         "Updated roadmap/docs to reflect accelerator rollout sequencing policy"))
      ((date . "2026-02-16")
       (accomplishments
         "Completed must-phase reliability/certificate/doc-alignment gates"
         "Integrated readiness gate script with CI coverage"))))))

;; Helper functions
(define (get-completion-percentage)
  "Get overall project completion percentage"
  (assoc-ref (assoc-ref state 'current-position) 'overall-completion))

(define (get-blockers priority)
  "Get blockers by priority (:critical, :high, :medium, :low)"
  (let ((blockers (assoc-ref state 'blockers-and-issues)))
    (assoc-ref blockers priority)))

(define (get-milestone name)
  "Get milestone by name"
  (let* ((route (assoc-ref state 'route-to-mvp))
         (milestones (assoc-ref route 'milestones)))
    (assoc name milestones)))
