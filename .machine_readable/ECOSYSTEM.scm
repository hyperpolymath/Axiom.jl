;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm - Project relationship mapping for Axiom.jl

(ecosystem
  (version "1.0")
  (name "Axiom.jl")
  (type "framework")
  (purpose "Provably correct machine learning framework for Julia with formal verification, multi-backend acceleration, and proof assistant integration")

  (position-in-ecosystem
    (role "primary-tool")
    (layer "application")
    (description "Julia ML framework combining compile-time shape verification, SMT-backed formal property guarantees, and high-performance Rust/Zig compute backends. Targets safety-critical ML (medical, autonomous, finance)."))

  (related-projects
    ((verisimdb . ((relationship . "potential-consumer")
                   (description . "Verification similarity database - could ingest Axiom.jl proof certificates")))
     (panic-attacker . ((relationship . "tooling")
                        (description . "Security scanner - should be run against Axiom.jl codebase")))
     (echidna . ((relationship . "tooling")
                 (description . "Proof auditor - should verify Axiom.jl's verification claims")))
     (hypatia . ((relationship . "integration")
                 (description . "Neurosymbolic CI/CD - should orchestrate Axiom.jl verification pipeline")))
     (eclexia . ((relationship . "sibling-project")
                 (description . "Programming language with shadow pricing - similar formal verification goals")))
     (language-bridges . ((relationship . "pattern-reference")
                          (description . "Zig FFI bridge patterns used by Axiom.jl's zig backend")))))

  (what-this-is
    "A Julia package that makes ML models provably correct by combining:
     1. Compile-time tensor shape verification via @axiom macro
     2. Runtime property assertions via @ensure macro
     3. Formal proof generation via @prove macro backed by SMT solvers (Z3, CVC5)
     4. Proof certificate export to Lean 4, Coq, Isabelle/HOL
     5. High-performance backends in Rust (Rayon parallelism) and Zig (SIMD)
     6. GPU acceleration via CUDA, ROCm, Metal package extensions
     7. PyTorch model import/export for interoperability")

  (what-this-is-not
    ("Not a general-purpose deep learning framework (use Flux.jl/Lux.jl for that)"
     "Not a replacement for Julia's type system (complements it)"
     "Not a standalone theorem prover (delegates to Z3/CVC5/Lean)")))
