<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# User Guide

This guide covers day-1 usage of Axiom.jl for model definition, inference, and verification.

## Requirements

- Julia `1.10+`
- CPU-only usage works out of the box
- Optional accelerators:
  - NVIDIA: `CUDA.jl`
  - AMD: `AMDGPU.jl`
  - Apple Silicon: `Metal.jl`

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/hyperpolymath/Axiom.jl")
```

For local development:

```julia
using Pkg
Pkg.develop(path=".")
Pkg.instantiate()
```

## Quick Inference

```julia
using Axiom

model = Sequential(
    Dense(784, 128, relu),
    Dense(128, 10),
    Softmax()
)

x = Tensor(randn(Float32, 32, 784))
y = model(x)
@show size(y.data)
```

## Verification

```julia
using Axiom

model = Sequential(Dense(10, 5, relu), Dense(5, 3), Softmax())
x = Tensor(randn(Float32, 4, 10))
data = [(x, nothing)]

result = verify(model, properties=[ValidProbabilities(), FiniteOutput()], data=data)
println(result)
```

## Certificates

```julia
cert = generate_certificate(model, result, model_name="demo-model")
save_certificate(cert, "demo.cert")
loaded = load_certificate("demo.cert")
@show verify_certificate(loaded)
```

## Optional Backends

CPU is the default:

```julia
set_backend!(JuliaBackend())
```

Zig backend (requires a compiled shared library):

```julia
set_backend!(ZigBackend("/path/to/libaxiom_zig.so"))
```

GPU backend (extension package required):

```julia
using CUDA
set_backend!(CUDABackend(0))
```

## Accelerator Scope

- Implemented today:
  - CPU + Zig backend
  - GPU backends via extensions (CUDA/ROCm/Metal)
  - Coprocessor backend targets (`TPUBackend`, `NPUBackend`, `PPUBackend`, `MathBackend`, `FPGABackend`, `DSPBackend`) with detection and CPU fallback
- Still in progress:
  - Production-grade runtime kernels for non-GPU coprocessors

## Packaging + Registry Baseline

```julia
metadata = create_metadata(model; name="demo", architecture="Sequential", version="1.0.0")
verify_and_claim!(metadata, "FiniteOutput", "verified=true; source=manual")
bundle = export_model_package(model, metadata, "build/model_package")
entry = build_registry_entry(bundle["manifest"]; channel="stable")
```

## Verification Telemetry Baseline

```julia
reset_verification_telemetry!()
result = verify(model, properties=[FiniteOutput()], data=[(x, nothing)])
verification_result_telemetry(result; source="manual-check")
verification_telemetry_report()
```

## Serving APIs (REST, GraphQL, gRPC)

REST server:

```julia
using Axiom

model = Sequential(Dense(10, 5, relu), Dense(5, 3), Softmax())
server = serve_rest(model; host="127.0.0.1", port=8080, background=true)
# close(server) when done
```

GraphQL server:

```julia
using Axiom

model = Sequential(Dense(10, 5, relu), Dense(5, 3), Softmax())
server = serve_graphql(model; host="127.0.0.1", port=8081, background=true)
# close(server) when done
```

gRPC bridge server + proto contract:

```julia
using Axiom

model = Sequential(Dense(10, 5, relu), Dense(5, 3), Softmax())
server = serve_grpc(model; host="127.0.0.1", port=50051, background=true)
# close(server) when done

generate_grpc_proto("axiom_inference.proto")
grpc_support_status()
```

`generate_grpc_proto`, in-process handlers, and an in-tree network gRPC bridge are included.
The bridge serves `POST /axiom.v1.AxiomInference/Predict` and `POST|GET /axiom.v1.AxiomInference/Health`.
Supported content types:
- `application/grpc` (binary unary protobuf frames)
- `application/grpc+json` (JSON bridge mode)

## Interop APIs (PyTorch import / ONNX export)

```julia
using Axiom

# Import a PyTorch checkpoint directly (requires python3 + torch)
model = from_pytorch("model.pt")

# Or import a canonical descriptor (axiom.pytorch.sequential.v1)
model = from_pytorch("model.pytorch.json")

# Export to ONNX (Dense/Conv/Norm/Pool + common activations)
to_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

## Troubleshooting

- Precompile issues:
  - Run `Pkg.instantiate()` then `Pkg.precompile()`
- GPU not detected:
  - Ensure the corresponding extension package is installed and functional
- Verification warning about missing data:
  - Provide `data=[(input_tensor, labels_or_nothing)]` to `verify`
