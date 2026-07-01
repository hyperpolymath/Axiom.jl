<!--
SPDX-License-Identifier: CC-BY-SA-4.0
SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
-->

# Backends, Serving & Interop

Accelerator backend abstraction (CPU/Zig/GPU/coprocessor), model serving
(REST/GraphQL/gRPC), PyTorch/ONNX interop, and model metadata/packaging.

```@autodocs
Modules = [Axiom]
Public = true
Private = false
Pages = [
    "backends/abstract.jl",
    "backends/julia_backend.jl",
    "backends/gpu_hooks.jl",
    "backends/zig_ffi.jl",
    "serving/api.jl",
    "integrations/interop.jl",
    "model_metadata.jl",
    "model_packaging.jl",
]
```
