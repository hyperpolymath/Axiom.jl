<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<!-- TOPOLOGY.md — Project architecture map and completion dashboard -->
<!-- Last updated: 2026-02-19 -->

# Axiom.jl — Project Topology

## System Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          EXTERNALS / ECOSYSTEM          │
                        ├─────────────────────────────────────────┤
                        │  ┌────────────┐      ┌────────────┐     │
                        │  │  PyTorch   │ ───▶ │    ONNX    │     │
                        │  │ Checkpoints│      │   Models   │     │
                        │  └─────┬──────┘      └─────▲──────┘     │
                        │        │                   │            │
                        └────────┼───────────────────┼────────────┘
                                 │                   │
                                 ▼                   │
                        ┌────────────────────────────┼────────────┐
                        │           APPLICATION LAYER             │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │     @axiom       │      │  Solvers │ │
                        │  │      DSL         │ ───▶ │ (Z3/CVC5)│ │
                        │  └────────┬─────────┘      └─────▲────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────┴────┐ │
                        │  │   Verification   │      │  @prove  │ │
                        │  │     Engine       │ ───▶ │  Bridge  │ │
                        │  └────────┬─────────┘      └──────────┘ │
                        │           │                             │
                        │  ┌────────▼─────────┐      ┌──────────┐ │
                        │  │     Model        │      │Telemetry │ │
                        │  │   Structure      │ ───▶ │ Observ.  │ │
                        │  └────────┬─────────┘      └──────────┘ │
                        │           │                             │
                        │  ┌────────▼─────────┐      ┌──────────┐ │
                        │  │      Core        │      │ Proof    │ │
                        │  │     Layers       │ ◀──▶ │ Assistant│ │
                        │  └──────────────────┘      └──────────┘ │
                        └───────────┬──────────────────────┬──────┘
                                    │                      │
                        ┌───────────▼──────────────────────▼──────┐
                        │           RUNTIME BACKENDS              │
                        ├─────────────────────────────────────────┤
                        │  ┌──────────────────┐      ┌──────────┐ │
                        │  │   Julia (Dev)    │      │   Rust   │ │
                        │  │     Backend      │ ◀──▶ │ Backend  │ │
                        │  └────────┬─────────┘      └─────┬────┘ │
                        │           │                      │      │
                        │  ┌────────▼─────────┐      ┌─────▼────┐ │
                        │  │    GPU / CUDA    │      │Accellerat│ │
                        │  │     Fallbacks    │      │ Dispatch │ │
                        │  └──────────────────┘      └──────────┘ │
                        └───────────┬──────────────────────┬──────┘
                                    │                      │
                        ┌───────────▼──────────────────────▼──────┐
                        │         REPO INFRASTRUCTURE             │
                        │  .machine_readable/ (state, meta)       │
                        │  .github/workflows/ (RSR Gate)          │
                        │  Project.toml, Cargo.toml               │
                        └─────────────────────────────────────────┘
```

## Completion Dashboard

```
COMPONENT                          STATUS              NOTES
─────────────────────────────────  ──────────────────  ─────────────────────────────────
CORE VERIFICATION & DSL
  @axiom DSL (src/dsl)              █████████░  90%    Stable, near final
  Shape Verification (src/types)    ██████████ 100%    Production ready
  @prove SMT Integration            ██████░░░░  60%    Functional, needs hardening

MODEL & PROOFS
  Neural Layers (src/layers)        ██████████ 100%    Full coverage (Dense/Conv/etc)
  Proof Assistant Stage 4           ██████░░░░  60%    Reconciliation shipped
  Certificate Generation            █████████░  90%    Integrity CI in place

INTEROP & ECOSYSTEM
  PyTorch Bridge (src/interop)      █████████░  85%    Checkpoint + Descriptor support
  ONNX Export (src/interop)         █████████░  85%    Baseline coverage
  Model Registry API                ████████░░  82%    Deterministic manifests

BACKENDS & PERFORMANCE
  Julia Native Backend              ██████████ 100%    Primary dev path
  Rust Core (rust/)                 ██████████ 100%    Feature parity with Julia
  GPU/CUDA Support                  █████████░  95%    Fallbacks + resilience verified
  Accelerator Gate                  ███████░░░  72%    Strict gates + detection shipped

REPO INFRASTRUCTURE
  .machine_readable/ (STATE.scm)    ██████████ 100%    Updated to v1.1.0
  .github/workflows/ (CI)           ██████████ 100%    RSR standard compliance
  Verification Telemetry            ████████░░  80%    Structured APIs baseline

─────────────────────────────────────────────────────────────────────────────
OVERALL:                            █████████░  92%    Stabilization Phase
```

## Key Dependencies

```
Core DSL ──────► Verification Engine ──────► Certificate Generation
                                                   │
Julia Backend ──────► Rust Backend ──────► Accelerator Support
                                                   │
Model Metadata ──────► Registry Workflow ────► COMPLETE
```

## Update Protocol

This file is maintained by both humans and AI agents. When updating:

1. **After completing a component**: Change its bar and percentage
2. **After adding a component**: Add a new row in the appropriate section
3. **After architectural changes**: Update the ASCII diagram
4. **Date**: Update the `Last updated` comment at the top of this file

Progress bars use: `█` (filled) and `░` (empty), 10 characters wide.
Percentages: 0%, 10%, 20%, ... 100% (in 10% increments).
