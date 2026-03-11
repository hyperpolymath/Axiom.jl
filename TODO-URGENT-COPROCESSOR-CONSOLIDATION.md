# URGENT: Coprocessor Backend Consolidation

**Date:** 2026-02-27
**Priority:** URGENT
**Status:** NOT STARTED

## Problem

`src/backends/abstract.jl` is **2,857 lines** mixing 6+ concerns in one file.
Total backend code: 4,346 lines across 4 files. The abstract.jl monolith is
unreadable, untestable, and blocks contributors from understanding the
extension points.

## Current State (4 files)

| File | Lines | Contents |
|------|-------|----------|
| `abstract.jl` | 2,857 | Backend types, SmartBackend router, coprocessor hooks, layer forwarding, compilation targets — ALL IN ONE FILE |
| `zig_ffi.jl` | 634 | Zig FFI implementation |
| `gpu_hooks.jl` | 470 | CUDA/Metal/ROCm hooks |
| `julia_backend.jl` | 385 | Pure Julia reference impl |

## Proposed Consolidation (7 files)

| New File | Contents | Est. Lines |
|----------|----------|-----------|
| `abstract.jl` (REDUCED) | Only: AbstractBackend base, 15 backend type defs, CompilationTarget, registry | ~200 |
| `compute.jl` | All Julia backend activation implementations (relu, sigmoid, gelu, etc.) | ~400 |
| `accelerators.jl` | GPU hooks (CUDA/Metal/ROCm), TPU/NPU/DSP/PPU/FPGA/VPU/QPU/Crypto defs | ~350 |
| `coprocessor_hooks.jl` | 9 coprocessor extension hooks + fallback dispatch | ~130 |
| `routing.jl` | SmartBackend dispatch tables + backend-aware layer forwarding | ~250 |
| `zig_ffi.jl` (KEEP) | Zig FFI — already well-scoped | 634 |
| `julia_backend.jl` (KEEP) | Reference impl — already well-scoped | 385 |

## 15 Backend Types (for reference)

JuliaBackend, ZigBackend, CUDABackend, MetalBackend, ROCmBackend,
TPUBackend, NPUBackend, DSPBackend, PPUBackend, MathBackend,
FPGABackend, VPUBackend, QPUBackend, CryptoBackend, SmartBackend

## 9 Coprocessor Hooks

`backend_coprocessor_matmul`, `_conv2d`, `_relu`, `_softmax`,
`_batchnorm`, `_layernorm`, `_maxpool2d`, `_avgpool2d`, `_global_avgpool2d`

## Motivation

IDApTIK's coprocessors were consolidated from 10 files to 3 on 2026-02-27
(Compute, Security, IO). Same pattern applies here: group by concern,
not by implementation detail. The 2,857-line abstract.jl is the worst
offender — it needs to be split into at least 4 focused files.

## Rules

- Every backend type, every hook, every activation MUST be preserved
- SmartBackend dispatch tables (from 2026-02-20 benchmarks) must not change
- Zero functionality loss — only file reorganisation
- Update `include()` statements in the module wrapper accordingly
