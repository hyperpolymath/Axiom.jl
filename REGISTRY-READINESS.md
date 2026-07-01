<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->
<!-- SPDX-FileCopyrightText: 2024-2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk> -->

# Axiom.jl — Honest Registry-Readiness Assessment

Last updated: 2026-05-18. This is a deliberately blunt self-assessment, not
marketing. It exists because the Julia General registry maintainers
(rightly) flagged this org for high-volume, LLM-generated, incoherent
package submissions. The only thing that rebuilds that trust is honesty.

## What was actually wrong (and is now fixed)

| Problem (real) | Status |
|---|---|
| `Project.toml` had **9 fabricated dependency UUIDs** (`11111111-…`) + 9 phantom extensions for packages that do not exist | ✅ Removed |
| Hard dep on **unregistered** `AcceleratorGate` | ✅ Vendored internally; dependency dropped |
| Docs claimed "283 tests / 99% / Release Candidate", broken cert workflow described as working, `@prove` described as proving | ✅ README/EXPLAINME rewritten to the truth; false claims removed |
| 100 erroring tests (`size(::Tensor,::Int)` undefined) | ✅ Fixed (real root cause) |
| `generate_certificate` threw `FieldError` (`result.mode` nonexistent) — broke the shipped example | ✅ Fixed (`verification_mode` kwarg) |
| `sum(::Tensor; dims)` undefined — documented usage failed | ✅ Fixed |
| Licensing self-contradictory: `Project.toml` MPL-2.0 vs source headers PMPL-1.0 (non-OSI) | ✅ Relicensed consistently to **MPL-2.0** (OSI), REUSE.toml added |
| LLM-tell files in package root (`0-AI-MANIFEST.a2ml`, `llm-warmup-*.md`) | ✅ Removed |

## Honest current limitations (NOT hidden)

- **`@prove` is experimental** — it runs but returns `unknown`; it does not
  discharge proofs. Documented as such; not a registration claim.
- **No native GPU acceleration of Axiom's own.** GPU/accelerator support is
  optional extensions over real packages (CUDA/AMDGPU/Metal/PyCall) only.
- **Axiom is downstream of Zygote** for AD; it implements none itself.
- The accelerator "backends" (TPU/NPU/FPGA/…) are **type stubs**, not
  working hardware backends. The README's prior-art section states plainly
  that Flux.jl/Lux.jl/CUDA.jl/KernelAbstractions.jl are more capable for
  real training on real hardware; Axiom's only distinct contribution is
  first-class runtime property-checking + packaging + multi-protocol
  serving as named APIs.

## Registry checklist

- [x] No fabricated UUIDs / phantom deps
- [x] All deps registered (after vendoring AcceleratorGate)
- [x] OSI license (MPL-2.0), consistent, REUSE.toml
- [x] Honest README with usage example + prior-art comparison
- [x] `Pkg.test()` green (verified repeatedly, incl. post-relicense)
- [x] Aqua.jl quality gate: **now a real, wired-in CI gate**
      (`test/aqua.jl`, included from `test/runtests.jl`), not just a
      self-assessment claim. `Aqua.test_all(Axiom)` passes with **zero
      overrides** -- ambiguities, unbound type parameters, undefined
      exports, project-extras, stale deps, `[compat]` bounds (incl.
      `julia`), piracy, and persistent tasks all pass clean. Getting there
      required two real fixes, not exemptions:
      - Three genuine unbound-type-parameter bugs in
        `src/layers/normalization.jl` (`LayerNorm`, `InstanceNorm`,
        `GroupNorm`): their auto-generated default constructors left `T`
        (and `S`) unbound because the only `T`-typed fields were
        `Union{_, Nothing}` (nullable when `affine=false`). Fixed with
        explicit inner constructors that bind `T`/`S` from the type
        application itself.
      - The stale `JSON3` dependency (its sole consumer,
        `src/integrations/huggingface.jl`, is dead code never
        `include()`d from `src/Axiom.jl`) is now actually removed from
        `Project.toml`'s `[deps]` -- this bullet had previously been
        marked done without a real Aqua run to verify it; it was not.
- [x] JET.jl static analysis gate: **new** (`test/jet.jl`), scoped to
      Axiom's own code via JET's native `target_modules` config.
      `report_package(Axiom)` finds 66 possible errors when run
      unscoped (essentially all rooted in transitive deps: JSON/
      StructUtils conversions, LinearAlgebra QR internals, Zygote adjoint
      plumbing); scoped to `target_modules = (Axiom,)` it finds **zero**,
      verified by direct report enumeration, not just by trusting the
      scoping logic.
- [x] Documenter.jl docs build (`docs/make.jl`, `docs/src/`), with
      `doctest = true`, builds clean locally (`julia --project=docs
      docs/make.jl`, exit 0, no warnings). API reference split across 4
      pages by source-file area because a single-page `@autodocs` dump
      of Axiom's full public surface exceeds Documenter's HTML
      `size_threshold` sanity check.
- [x] Julia test-gate wired into CI (`.github/workflows/julia-test.yml`):
      builds the hybrid Ed448+Dilithium5 crypto cdylib (`crypto/`) before
      running `Pkg.test()`, so `test/verification/hybrid_signing_tests.jl`
      actually exercises the real signing path in CI instead of always
      taking the `@test_skip` branch (the pre-existing `ci.yml`
      `julia-compat` job never built the crypto shim, so those 27 tests
      were silently skipped in every CI run to date).
- [ ] **Human maintainer track** — NOT a code task: genuine Julia
      community participation, engaging the registry maintainers as a
      person, and reducing org-wide package sprawl. This, not tooling, is
      the actual gate. Registration should not be re-attempted until this
      is genuinely addressed.

## Position

This package is now technically honest and defensible. It is **not**
re-submitted to General by tooling. Whether/when to register is a human
decision contingent on the community-trust track above.
