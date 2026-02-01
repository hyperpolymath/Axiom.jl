;; SPDX-License-Identifier: PMPL-1.0-or-later
;; ECOSYSTEM.scm - Project relationship mapping
;; Media Type: application/vnd.ecosystem+scm

(ecosystem
  (version "1.0")
  (name "Axiom.jl")
  (type "ml-verification-framework")
  (purpose "Provably correct machine learning framework with compile-time shape verification, formal property guarantees, and high-performance Rust/Zig backends")

  (position-in-ecosystem
    (role "verification-foundation")
    (layer "application-library")
    (description "Provides compile-time correctness guarantees for ML systems, bridging formal methods with practical machine learning through shape verification, property proofs, and performant native backends"))

  (related-projects
    ((name . "ProvenCrypto.jl")
     (relationship . "sibling-standard")
     (description . "Formally verified cryptography - both projects use formal verification for correctness guarantees")
     (integration . "Shared verification techniques, potential for verified ML crypto applications like homomorphic encryption inference"))
    ((name . "PolyglotFormalisms.jl")
     (relationship . "potential-consumer")
     (description . "Multi-language formal verification framework - could integrate Axiom's @prove annotations")
     (integration . "PolyglotFormalisms could translate Axiom proofs across verification languages, enabling cross-platform ML verification"))
    ((name . "Causals.jl")
     (relationship . "potential-consumer")
     (description . "Causal inference framework - needs verified ML for fairness and causal effect estimation")
     (integration . "Axiom can provide shape-verified causal models, provably correct confounding adjustment, formal fairness guarantees"))
    ((name . "BowtieRisk.jl")
     (relationship . "potential-consumer")
     (description . "Risk analysis framework - could use Axiom for AI safety hazard analysis")
     (integration . "Model ML system failures with verified properties, quantify risks of shape mismatches, prove safety barriers"))
    ((name . "PyTorch")
     (relationship . "inspiration")
     (description . "Industry-standard ML framework - Axiom provides PyTorch model import with verification layer")
     (integration . "Import PyTorch models, add compile-time shape checks, export to verified Rust/Zig backends"))
    ((name . "Lean")
     (relationship . "inspiration")
     (description . "Proof assistant - Axiom adapts dependent type techniques for practical ML verification")
     (integration . "Compatible proof approaches, potential backend target for deep verification")))

  (what-this-is
    "A Julia ML framework with compile-time shape verification preventing runtime dimension errors"
    "Formal property system (@prove annotations) for correctness guarantees beyond types"
    "Rust and Zig performance backends for production deployment with zero-cost abstractions"
    "PyTorch model import pipeline with automatic verification layer insertion"
    "Dependent type inference for tensor shapes eliminating manual annotations"
    "Provable fairness guarantees for ML models through formal constraints"
    "Static analysis of gradient flows preventing vanishing/exploding gradients"
    "Contract-based API ensuring pre/post-conditions on model operations"
    "Integration with Julia's scientific computing ecosystem"
    "Transparent performance model showing where verification overhead occurs")

  (what-this-is-not
    "Not a general-purpose proof assistant (like Lean/Coq) - specialized for ML verification"
    "Not focused on neural architecture search - verifies given architectures"
    "Not a distributed training framework - focuses on correctness, not scale"
    "Not production-ready yet - research/early-stage development"
    "Not backward-compatible with PyTorch API - adds verification layer"
    "Not designed for dynamic computation graphs - static shapes required for verification"
    "Not a model compression tool - verification may prevent some optimizations"
    "Not interactive proving - automated verification or compilation failure"
    "Not certified for safety-critical systems yet - formal foundation being built"
    "Not replacing existing ML frameworks - complementary verification layer"))
