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
- [x] Aqua.jl quality gate: unbound-params / undefined-exports /
      project-extras pass; stale-deps **now clean** (orphaned
      `src/integrations/huggingface.jl` + its sole-consumer dep `JSON3`
      removed — they were unreachable dead code)
- [ ] **Human maintainer track** — NOT a code task: genuine Julia
      community participation, engaging the registry maintainers as a
      person, and reducing org-wide package sprawl. This, not tooling, is
      the actual gate. Registration should not be re-attempted until this
      is genuinely addressed.

## Position

This package is now technically honest and defensible. It is **not**
re-submitted to General by tooling. Whether/when to register is a human
decision contingent on the community-trust track above.
