;; SPDX-License-Identifier: PMPL-1.0-or-later
;; PLAYBOOK.scm - Operational runbook for Axiom.jl

(define playbook
  `((version . "1.1.0")
    (procedures
      ((verify-readiness
         . (("baseline" . "./scripts/readiness-check.sh")
            ("no-rust" . "AXIOM_READINESS_RUN_RUST=0 ./scripts/readiness-check.sh")
            ("full" . "AXIOM_READINESS_ALLOW_SKIPS=0 ./scripts/readiness-check.sh")))
       (coprocessor
         . (("strategy" . "julia --project=. test/ci/coprocessor_strategy.jl")
            ("resilience" . "julia --project=. test/ci/coprocessor_resilience.jl")
            ("strict-tpu" . "julia --project=. test/ci/tpu_required_mode.jl")
            ("strict-npu" . "julia --project=. test/ci/npu_required_mode.jl")
            ("strict-dsp" . "julia --project=. test/ci/dsp_required_mode.jl")))
       (could-baselines
         . (("package-registry" . "julia --project=. test/ci/model_package_registry.jl")
            ("optimization" . "julia --project=. test/ci/optimization_passes.jl")
            ("telemetry" . "julia --project=. test/ci/verification_telemetry.jl")
            ("package-evidence" . "julia --project=. scripts/model-package-evidence.jl")
            ("optimization-evidence" . "julia --project=. scripts/optimization-evidence.jl")
            ("telemetry-evidence" . "julia --project=. scripts/verification-telemetry-evidence.jl")))
       (proof-bundle
         . (("reconcile-ci" . "julia --project=. test/ci/proof_bundle_reconciliation.jl")
            ("evidence" . "julia --project=. scripts/proof-bundle-evidence.jl")))))
    (alerts
      ((hard-fail
         . ("readiness-check reports failed checks"
            "strict-mode test fails when backend is required"
            "proof bundle reconciliation status regresses"))
       (soft-fail
         . ("performance evidence regression vs baseline"
            "telemetry evidence missing or malformed"
            "docs/roadmap alignment check fails"))))
    (contacts
      ((maintainer . "Axiom.jl maintainers")
       (repo . "https://github.com/hyperpolymath/Axiom.jl")))))
