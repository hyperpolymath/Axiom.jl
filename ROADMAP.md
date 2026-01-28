# Axiom.jl Development Roadmap

## Current State (v1.0)

Production-ready theorem proving and verification framework:
- Multi-backend architecture (Julia, Zig, Rust)
- SMTLib integration (Z3 solver interface)
- Proof search with strategies (forward/backward chaining)
- Proof export to Coq, Isabelle, Lean, Idris
- Expression manipulation and unification
- Comprehensive security hardening (20+ @assert → proper validation)

**Status:** Complete with security fixes. **Known limitation:** Rust backend compilation issue (num_traits dependency) - Julia backend fully functional.

---

## v1.0 → v1.2 Roadmap (Near-term)

### v1.1 - Proof Automation & Usability (3-6 months)

**MUST:**
- [ ] **Fix Rust backend** - Resolve num_traits dependency issue in packages/rust/Cargo.toml
- [ ] **Automated tactic learning** - Suggest proof tactics based on goal similarity to past proofs
- [ ] **Interactive proof assistant** - REPL with proof state visualization (goals, assumptions)
- [ ] **Proof library** - Pre-proven lemmas for arithmetic, logic, set theory, algebra

**SHOULD:**
- [ ] **Sledgehammer-style automation** - Call external provers (E, Vampire, SPASS) and reconstruct proofs
- [ ] **Counter-example generation** - Use model finders (Nitpick, Nunchaku) to disprove false conjectures
- [ ] **Proof by induction** - Automated induction tactic with well-founded recursion
- [ ] **Simplification engine** - Rewrite rules, normalization, term ordering

**COULD:**
- [ ] **Jupyter integration** - Pluto.jl notebook for literate theorem proving
- [ ] **LaTeX export** - Pretty-print proofs for publication (natural deduction trees, sequent calculus)
- [ ] **Proof visualization** - Graphical proof trees (Makie.jl)

### v1.2 - Advanced Logics & Integration (6-12 months)

**MUST:**
- [ ] **Higher-order logic** - HOL support (function types, lambda calculus)
- [ ] **Dependent type theory** - CIC (Calculus of Inductive Constructions) backend
- [ ] **Program verification** - Hoare logic for Julia code (pre/post conditions, loop invariants)
- [ ] **Integration with PolyglotFormalisms.jl** - Import/export TLA+, Alloy, Z specs

**SHOULD:**
- [ ] **Linear logic** - Resource-aware reasoning (session types, concurrency)
- [ ] **Modal logic** - Temporal logic (LTL, CTL), epistemic logic
- [ ] **Separation logic** - Memory safety verification (heap reasoning)
- [ ] **Integration with ProvenCrypto.jl** - Verify crypto protocol implementations

**COULD:**
- [ ] **Category theory library** - Functors, natural transformations, limits, adjunctions
- [ ] **Homotopy type theory** - Univalence, higher inductive types
- [ ] **Quantum logic** - Verify quantum algorithms (Grover, Shor)

---

## v1.3+ Roadmap (Speculative)

### Research Frontiers

**AI-Assisted Theorem Proving:**
- Neural theorem provers (GPT-f, PACT, Thor style)
- Reinforcement learning for tactic selection (AlphaProof approach)
- Autoformalization (natural language → formal statement)
- Proof repair (fix broken proofs after library changes)

**Verified Software at Scale:**
- Verified compilers (CompCert-style for Julia/Rust)
- Verified operating systems (seL4-style microkernel)
- Verified cryptography (full Fiat-Crypto integration)
- Verified machine learning (neural network correctness proofs)

**Formal Mathematics:**
- Formalize major theorems (Kepler conjecture, Feit-Thompson, Hales-Jewett)
- Integration with Lean's mathlib (import/export)
- Automated conjecture generation (discover new theorems)
- Collaborative proof engineering (distributed theorem proving)

**Quantum & Beyond:**
- Quantum Hoare logic (verify quantum circuits)
- Linear dependent types (quantum resource management)
- Topological quantum computing verification
- Non-classical logics (paraconsistent, fuzzy, many-valued)

### Ecosystem Integration

- **Symbolics.jl:** Symbolic math integrated with theorem proving
- **ModelingToolkit.jl:** Verify correctness of numerical simulations
- **DifferentialEquations.jl:** Prove stability, convergence properties
- **Optimization.jl/JuMP.jl:** Certified optimization (verified optimality)

### Ambitious Features

- **Theorem proving foundation model** - Pre-trained on mathlib, Archive of Formal Proofs, Coq stdlib
- **Autonomous mathematician** - AI agent that formulates and proves conjectures
- **Global proof repository** - Federated database of verified theorems (blockchain-backed?)
- **Formal verification as a service** - Cloud API for on-demand proof checking

---

## Migration Path

**v1.0 → v1.1:** Backward compatible (Rust backend fix, proof automation additive)
**v1.1 → v1.2:** Mostly compatible (dependent types may require new expression types)
**v1.2 → v1.3+:** Breaking changes likely (AI features, HoTT need fundamental redesign)

## Community Goals

- **Formalize 100 theorems** from "100 Theorems" list by v1.2
- **Publication at ITP conference** (Interactive Theorem Proving) by v1.2
- **Integration with major provers** (Coq, Isabelle, Lean, Metamath) by v1.2
- **Verification of real Julia package** (e.g., LinearAlgebra.jl subset) by v1.2

## Critical Next Steps (Pre-v1.1)

1. **Fix Rust backend compilation** - Add num_traits to packages/rust/Cargo.toml
2. **Performance benchmarking** - Compare proof search against Prover9, E prover
3. **Documentation** - Tutorial, API reference, example proofs
4. **Community outreach** - Blog post, JuliaCon talk proposal
