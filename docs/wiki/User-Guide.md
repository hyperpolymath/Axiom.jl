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

Rust backend (requires a compiled shared library):

```julia
set_backend!(RustBackend("/path/to/libaxiom_core.so"))
```

GPU backend (extension package required):

```julia
using CUDA
set_backend!(CUDABackend(0))
```

## Accelerator Scope

- Implemented today:
  - CPU + Rust backend
  - GPU backends via extensions (CUDA/ROCm/Metal)
  - Coprocessor backend targets (`TPUBackend`, `NPUBackend`, `DSPBackend`, `FPGABackend`) with detection and CPU fallback
- Still in progress:
  - Production-grade runtime kernels for non-GPU coprocessors

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

gRPC contract + handlers:

```julia
using Axiom

generate_grpc_proto("axiom_inference.proto")
grpc_support_status()
```

`generate_grpc_proto` and the in-process `grpc_predict`/`grpc_health` handlers are included in-tree.
Network gRPC serving is intended to use the generated proto with your chosen gRPC runtime.

## Troubleshooting

- Precompile issues:
  - Run `Pkg.instantiate()` then `Pkg.precompile()`
- GPU not detected:
  - Ensure the corresponding extension package is installed and functional
- Verification warning about missing data:
  - Provide `data=[(input_tensor, labels_or_nothing)]` to `verify`
