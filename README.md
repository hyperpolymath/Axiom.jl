<!--
SPDX-License-Identifier: CC-BY-SA-4.0
SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
-->

[![OpenSSF Best Practices](https://img.shields.io/badge/OpenSSF-Best_Practices-green?logo=opensourcesecurity)](https://www.bestpractices.dev/en/projects/new?repo_url=https://github.com/hyperpolymath/Axiom.jl)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/) <embed
src="https://api.thegreenwebfoundation.org/greencheckimage/github.com"
data-link="https://www.thegreenwebfoundation.org/green-web-check/?url=github.com" />
image:<a href="https://img.shields.io/badge/Julia-1.10+-9558B2?logo=julia"
data-link="https://julialang.org/">Julia</a>
[![Zig](https://img.shields.io/badge/Zig-Backend-F7A41D?logo=zig)](https://ziglang.org/)

**Provably Correct Machine Learning**

*Machine learning where bugs are caught at compile time, not in
production.*

<div id="toc">

</div>

# What is Axiom.jl?

Axiom.jl is a **next-generation ML framework** that combines:

- **Compile-time verification** - Shape errors caught before runtime

- **Formal guarantees** - Verification checks and certificate workflows

- **Optional acceleration** - Zig/GPU backend paths with explicit
  fallback behavior

- **Julia elegance** - Express models as mathematical specifications

```julia
using Axiom

model = Sequential(
    Dense(784, 128, relu),
    Dense(128, 10),
    Softmax()
)

x = Tensor(randn(Float32, 16, 784))
y = model(x)
result = verify(model, properties=[ValidProbabilities(), FiniteOutput()], data=[(x, nothing)])
@assert result.passed
```

# Features

## Compile-Time Shape Verification

```julia
# PyTorch: Runtime error after hours of training
# Axiom.jl: Compile error in milliseconds

@axiom BrokenModel begin
    input :: Tensor{Float32, (224, 224, 3)}
    features = input |> Conv(64, (3,3))
    output = features |> Dense(10)  # COMPILE ERROR!
    # "Shape mismatch: Conv output is (222,222,64), Dense expects vector"
end
```

## Formal Verification

```julia
@axiom SafeClassifier begin
    # ...
    @ensure valid_probabilities(output)    # Runtime assertion
    @prove ∀x. sum(softmax(x)) == 1.0      # Experimental proof workflow
end

# Generate verification certificates
cert = verify(model) |> generate_certificate
save_certificate(cert, "fda_submission.cert")
```

## Model Interoperability

```julia
# Import from a PyTorch checkpoint (.pt/.pth/.ckpt) via built-in Python bridge
# (requires python3 + torch in the selected runtime)
model = from_pytorch("model.pt")

# Or import canonical descriptor JSON
model2 = from_pytorch("model.pytorch.json")

# Export supported models to ONNX
to_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

Current scope:

- `from_pytorch(…)`: canonical descriptor import + direct
  `.pt/.pth/.ckpt` bridge.

- `to_onnx(…)`: export for `Sequential`/`Pipeline` models built from
  Dense/Conv/Norm/Pool + common activations.

## High Performance

```julia
# Development: Julia backend
model = Sequential(Dense(784, 128, relu), Dense(128, 10))

# Production path: optional Zig backend
prod_model = compile(model, backend=ZigBackend("/path/to/libaxiom_zig.so"), optimize=:aggressive)
```

## Coprocessor Targets

> **Status — skeleton, not accelerated execution.** Coprocessor support today is
> a *detection + dispatch* surface: `detect_coprocessor()` probes for devices and
> `compile(…, backend=cop)` routes through the backend interface, but the
> per-device compute kernels are not yet implemented, so execution gracefully
> falls back to the Julia backend. Treat this as a forward-looking API, not
> hardware-accelerated inference.

```julia
# Non-GPU accelerator targets with self-healing fallback
cop = detect_coprocessor()  # TPU/NPU/VPU/QPU/PPU/MATH/CRYPTO/FPGA/DSP or nothing
if cop !== nothing
    model_accel = compile(model, backend=cop, verify=false)
end
```

## Model Packaging + Registry Manifests

```julia
metadata = create_metadata(
    model;
    name="my-model",
    architecture="Sequential",
    version="1.0.0",
)
verify_and_claim!(metadata, "FiniteOutput", "verified=true; source=ci")

bundle = export_model_package(model, metadata, "build/model_package")
entry = build_registry_entry(bundle["manifest"]; channel="stable")
export_registry_entry(entry, "build/model_package/registry-entry.json")
```

## Verification Telemetry

```julia
reset_verification_telemetry!()
result = verify(model, properties=[FiniteOutput()], data=[(x, nothing)])
run_payload = verification_result_telemetry(result; source="inference-gate")
summary = verification_telemetry_report()
```

## Serving APIs

```julia
# REST
rest_server = serve_rest(model; host="0.0.0.0", port=8080, background=true)

# GraphQL
graphql_server = serve_graphql(model; host="0.0.0.0", port=8081, background=true)

# gRPC bridge server + contract generation
# - binary unary protobuf (`application/grpc`)
# - JSON bridge mode (`application/grpc+json`)
grpc_server = serve_grpc(model; host="0.0.0.0", port=50051, background=true)
generate_grpc_proto("axiom_inference.proto")
```

## Interop APIs

```julia
# PyTorch import (checkpoint bridge or canonical descriptor JSON)
model = from_pytorch("model.pt")
model = from_pytorch("model.pytorch.json")

# ONNX export (Dense/Conv/Norm/Pool + common activations)
to_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

# Quick Start

## Installation

```julia
using Pkg
Pkg.add("Axiom")
```

## Hello World

```julia
using Axiom

# Define a simple classifier
model = Sequential(
    Dense(784, 256, relu),
    Dense(256, 10),
    Softmax()
)

# Generate sample data
x = randn(Float32, 32, 784)

# Inference
predictions = model(x)

# Verify properties
@ensure all(sum(predictions, dims=2) .≈ 1.0)
```

## With @axiom DSL

```julia
using Axiom

@axiom MNISTClassifier begin
    input :: Tensor{Float32, (:batch, 28, 28, 1)}
    output :: Probabilities(10)

    features = input |> Conv(32, (3,3)) |> ReLU |> MaxPool((2,2))
    features = features |> Conv(64, (3,3)) |> ReLU |> MaxPool((2,2))
    flat = features |> GlobalAvgPool() |> Flatten
    output = flat |> Dense(64, 10) |> Softmax

    @ensure valid_probabilities(output)
end

model = MNISTClassifier()
```

# Why Axiom.jl?

## The Problem

ML models are deployed in critical applications:

- Medical diagnosis

- Autonomous vehicles

- Financial systems

- Criminal justice

Yet our tools allow bugs to slip through to production.

## The Solution

Axiom.jl catches bugs **before** they cause harm:

| Issue | PyTorch | Axiom.jl |
|----|----|----|
| Shape mismatch | Runtime crash | Compile error |
| NaN in output | Silent failure | Detected/proven |
| Invalid probabilities | Undetected | Checkable with verification properties |
| Adversarial fragility | Unknown | Roadmap / partial |

# Documentation

- [Home](docs/wiki/Home.md) - Start here

- [User Guide](docs/wiki/User-Guide.md) - Install, infer, verify

- [Developer Guide](docs/wiki/Developer-Guide.md) - Build/test/release
  workflow

- [Release Checklist](RELEASE-CHECKLIST.adoc) - Pre-release and
  release-day gates

- [Vision](docs/wiki/Vision.md) - Why we built this

- [@axiom DSL](docs/wiki/Axiom-DSL.md) - Model definition guide

- [Verification](docs/wiki/Verification.md) - @ensure and @prove

- [Migration Guide](docs/wiki/Migration-Guide.md) - From PyTorch

- [FAQ](docs/wiki/FAQ.md) - Common questions

- [Roadmap](ROADMAP.md) - Tracked commitments and delivery criteria

# Project Structure

    Axiom.jl/
    ├── src/                 # Julia source
    │   ├── Axiom.jl        # Main module
    │   ├── types/          # Tensor type system
    │   ├── layers/         # Neural network layers
    │   ├── dsl/            # @axiom macro system
    │   ├── verification/   # @ensure, @prove
    │   ├── training/       # Optimizers, loss functions
    │   └── backends/       # Backend abstraction (15 backends)
    ├── zig/                # Zig native backend
    │   └── src/           # matmul, conv, norm, attention, etc.
    ├── ext/                # GPU package extensions (CUDA, ROCm, Metal)
    ├── test/               # Test suite
    ├── examples/           # Example models
    └── docs/               # Documentation & wiki

# Roadmap

- [x] **v0.1** - Core framework, DSL, verification basics

- [ ] **v0.2** - Full Zig backend, GPU support

- [ ] **v0.3** - Hugging Face integration, model zoo

- [ ] **v0.4** - Advanced proofs, SMT integration

- [ ] **v1.0** - Production ready, industry certifications

# Contributing

We welcome contributions! See
<a href="CONTRIBUTING.md" class="md">CONTRIBUTING</a>.

- Bug reports and feature requests

- Documentation improvements

- New layers and operations

- Performance optimizations

- Verification methods

# Julia-First Verification

Axiom’s proof system is **Julia-native by default**. SMT solving runs
through `packages/SMTLib.jl` with no native backend dependency. The Zig
SMT runner is an optional backend you can enable for hardened subprocess
control.

Julia-native example:

```julia
@prove ∃x. x > 0
```

Optional Zig runner:

```bash
export AXIOM_SMT_RUNNER=zig
export AXIOM_ZIG_LIB=/path/to/libaxiom_zig.so
export AXIOM_SMT_SOLVER=z3
```

```julia
@prove ∃x. x > 0
```

# License

Licensed under the [Mozilla Public License 2.0
(MPL-2.0)](https://www.mozilla.org/en-US/MPL/2.0/).

This project’s philosophy of sharing is guided by the [Palimpsest
License
(PMPL-1.0)](https://github.com/hyperpolymath/palimpsest-license) — a
next-generation, quantum-safe, AI-aware copyleft license designed for
the post-quantum era, with built-in provisions for machine learning
model governance, reproducible builds, and ethical compute. The
Palimpsest License is currently in development and undergoing community
review. Until it achieves full SPDX recognition, Axiom.jl is distributed
under MPL-2.0, on which the Palimpsest License is based. The spirit is
the same: share improvements, keep the ecosystem open, and give
downstream users the freedom to build on verified foundations.

# Acknowledgments

Axiom.jl builds on the shoulders of giants:

- [Julia](https://julialang.org/) - The language

- <a href="https://fluxml.ai/" class="jl">Flux</a> - Inspiration for
  Julia ML

- [Zig](https://ziglang.org/) - Native performance backend

- [PyTorch](https://pytorch.org/) - Ecosystem compatibility

------------------------------------------------------------------------

*The future of ML is verified.*

[Get Started](docs/wiki/Home.md) \| [Release
Checklist](RELEASE-CHECKLIST.adoc) \| [Roadmap](ROADMAP.md) \| [Star on
GitHub](https://github.com/hyperpolymath/Axiom.jl)
