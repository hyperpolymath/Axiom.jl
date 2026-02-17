# Architecture

> A deep dive into Axiom.jl's design philosophy and implementation

## Overview

Axiom.jl is built on a multi-runtime architecture that combines Julia ergonomics with optional Rust acceleration and extension-based GPU/coprocessor targets. The goal is **mathematical elegance without sacrificing speed**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Space                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Julia DSL (@axiom)                           ││
│  │  Mathematical notation • Type inference • REPL exploration      ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                      Verification Layer                              │
│  ┌──────────────────────┐  ┌───────────────────────────────────────┐│
│  │  @ensure (Runtime)   │  │  @prove (Compile-time)                ││
│  │  Shape assertions    │  │  SMT-based verification               ││
│  │  Value constraints   │  │  Proof certificates                   ││
│  └──────────────────────┘  └───────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                      Computation Backends                            │
│  ┌─────────────────┐ ┌──────────────────┐ ┌───────────────────────┐│
│  │  Julia Backend  │ │   Rust Backend   │ │   Accelerator Targets ││
│  │  Pure Julia     │ │   FFI via ccall  │ │   TPU/NPU/DSP/FPGA    ││
│  │  Portable       │ │   Rayon parallel │ │   Fallback-first flow ││
│  └─────────────────┘ └──────────────────┘ └───────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                        Hardware Layer                                │
│  ┌───────────────┐ ┌───────────────┐ ┌────────────────────────────┐│
│  │     CPU       │ │     GPU       │ │     Accelerators           ││
│  │  AVX/NEON     │ │  CUDA/ROCm    │ │   TPU/NPU/Custom           ││
│  └───────────────┘ └───────────────┘ └────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Correctness by Construction

Unlike frameworks that bolt on type checking as an afterthought, Axiom.jl was designed from the ground up with verification in mind:

```julia
# The type system prevents shape mismatches at compile time
struct Tensor{T, Shape}
    data::Array{T}
end

# This won't compile - shapes don't match
# invalid_add(a::Tensor{Float32, (784,)}, b::Tensor{Float32, (10,)}) = a + b
```

### 2. Zero-Cost Abstractions

High-level constructs compile to efficient code:

```julia
# This beautiful DSL...
@axiom function classify(x)
    x |> Dense(784 => 256) |> relu |> Dense(256 => 10) |> softmax
end

# ...generates code equivalent to hand-written SIMD loops
```

### 3. Explicit is Better Than Implicit

No hidden magic. Every operation is transparent:

```julia
# You always know what backend you're using
using Axiom: RustBackend, JuliaBackend

model = @axiom backend=RustBackend() begin
    Dense(784 => 256, activation=relu)
end

# Explicit verification boundaries
@ensure output isa Tensor{Float32, (10,)} "Output must be 10-class probabilities"
```

## Layer Architecture

### The Tensor Type

At the heart of Axiom.jl is the `Tensor` type - a parametric type that carries shape information:

```julia
struct Tensor{T<:Number, Shape<:Tuple}
    data::Array{T}

    function Tensor{T, Shape}(data::Array{T}) where {T, Shape}
        expected_shape = Tuple(Shape.parameters)
        @assert size(data) == expected_shape "Shape mismatch: expected $expected_shape, got $(size(data))"
        new{T, Shape}(data)
    end
end
```

**Why this matters:**
- Shape errors caught at compile time, not runtime
- Type inference propagates shapes through operations
- Documentation is built into the type signature

### Layer Abstraction

Every layer follows a consistent interface:

```julia
abstract type AbstractLayer end

# Required methods
function forward(layer::AbstractLayer, x::AbstractArray) end
function parameters(layer::AbstractLayer)::Vector{AbstractArray} end
function trainable(layer::AbstractLayer)::Bool end

# Optional: custom backward pass
function backward(layer::AbstractLayer, grad::AbstractArray) end
```

### Shape Inference

Shapes flow through the computation graph:

```julia
# Input shape: (batch, 784)
x = Tensor{Float32, (batch, 784)}(data)

# Dense layer: 784 -> 256
layer = Dense(784 => 256)

# Output shape: (batch, 256) - automatically inferred
y = forward(layer, x)  # Type: Tensor{Float32, (batch, 256)}
```

## Backend System

### Backend Selection

Axiom.jl automatically selects the best backend:

```julia
function select_backend()
    if rust_available()
        return RustBackend()  # Parallel by default
    else
        return JuliaBackend()  # Always available
    end
end
```

### Julia Backend

Pure Julia implementation - no external dependencies:

```julia
struct JuliaBackend <: AbstractBackend end

function forward(::JuliaBackend, layer::Dense, x::AbstractArray)
    y = x * layer.weight
    if layer.use_bias
        y .+= layer.bias'
    end
    return y
end
```

**Pros:**
- No compilation step
- Works everywhere
- Easy to debug

**Cons:**
- Single-threaded by default
- No SIMD optimization

### Rust Backend

High-performance parallel computing:

```julia
struct RustBackend <: AbstractBackend end

function forward(::RustBackend, layer::Dense, x::AbstractArray)
    ccall((:rust_dense_forward, RUST_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Cint, Cint, Cint),
          x, layer.weight, layer.bias, output,
          batch_size, in_features, out_features)
end
```

**Rust Implementation:**
```rust
#[no_mangle]
pub extern "C" fn rust_dense_forward(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    output: *mut f32,
    batch: i32,
    in_features: i32,
    out_features: i32,
) {
    // Parallel over batch dimension using Rayon
    (0..batch).into_par_iter().for_each(|b| {
        // BLAS-style matrix-vector multiplication
        for o in 0..out_features {
            let mut sum = unsafe { *bias.add(o as usize) };
            for i in 0..in_features {
                sum += unsafe {
                    *input.add((b * in_features + i) as usize)
                    * *weight.add((i * out_features + o) as usize)
                };
            }
            unsafe { *output.add((b * out_features + o) as usize) = sum };
        }
    });
}
```

### Coprocessor Targets (TPU/NPU/DSP/FPGA)

Axiom exposes non-GPU accelerator targets through backend handles and
fallback-safe dispatch:

```julia
cop = detect_coprocessor()  # TPU/NPU/DSP/FPGA or nothing
if cop !== nothing
    compiled = compile(model, backend=cop, verify=false, optimize=:none)
end
```

Current state:
- Strategy and fallback behavior are implemented and CI-covered.
- Extension hooks are available for backend-specific kernel overrides.
- Production kernels for specific coprocessors remain roadmap work.

## Memory Management

### Tensor Memory Layout

Tensors are stored in row-major (C) order:

```julia
# Logical shape: (batch=2, height=3, width=4)
# Memory layout: [batch0_h0_w0, batch0_h0_w1, ..., batch1_h2_w3]
```

### Buffer Reuse

The computation graph tracks buffer lifetimes:

```julia
# These operations reuse buffers when safe
y1 = relu(x)      # Allocates buffer A
y2 = sigmoid(y1)  # Can reuse buffer A (y1 no longer needed)
```

### Gradient Checkpointing

For large models, recompute instead of store:

```julia
@axiom checkpoint=true function transformer_block(x)
    # Activations will be recomputed during backward pass
    # Saves O(n) memory at cost of O(1) extra compute
end
```

## Verification System

### Runtime Verification (@ensure)

```julia
@ensure property description
```

Checks properties at runtime with descriptive errors:

```julia
function forward(layer::Dense, x)
    @ensure size(x, 2) == layer.in_features "Input features must match layer"
    @ensure all(isfinite, x) "Input contains NaN or Inf"

    y = x * layer.weight'

    @ensure size(y, 2) == layer.out_features "Output shape mismatch"
    return y
end
```

### Compile-Time Verification (@prove)

```julia
@prove property
```

Uses SMT solvers to prove properties statically:

```julia
@prove output_shape(model) == (batch_size, num_classes)
@prove all_weights_bounded(model, -10.0, 10.0)
@prove gradient_exists(model)
```

### Verification Certificates

Generate auditable proof artifacts:

```julia
cert = verify_model(model, properties)

# Generate certificate
save_certificate(cert, "model_v1.cert")

# Certificate contains:
# - Model hash
# - Properties verified
# - Proof traces
# - Timestamp
```

## Compilation Pipeline

```
Source Code (@axiom)
       │
       ▼
┌──────────────────┐
│  Macro Expansion │  Convert DSL to Julia IR
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  Shape Inference │  Propagate tensor shapes
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  Verification    │  Check @ensure/@prove
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ Backend Lowering │  Dispatch to Rust/Julia/accelerator targets
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  LLVM/Native     │  Final code generation
└──────────────────┘
```

## Extension Points

### Custom Layers

```julia
struct MyLayer <: AbstractLayer
    param::Array{Float32}
end

Axiom.forward(layer::MyLayer, x) = custom_op(x, layer.param)
Axiom.parameters(layer::MyLayer) = [layer.param]
```

### Custom Backends

```julia
struct CustomAcceleratorBackend <: AbstractBackend end

function Axiom.forward(::CustomAcceleratorBackend, layer::Dense, x)
    # Dispatch to custom hardware
end
```

### Custom Verifiers

```julia
struct CustomProperty <: VerificationProperty
    name::String
    check::Function
end

function verify(prop::CustomProperty, model, x)
    prop.check(model, x)
end
```

## Performance Characteristics

| Operation | Julia | Rust | Notes |
|-----------|-------|------|-------|
| MatMul 256×256 | 1.0× | 2.1× | Rust backend parity tracked in CI |
| MatMul 1024×1024 | 1.0× | 2.8× | Representative CPU/Rust scaling |
| Conv2D 3×3 | 1.0× | 1.8× | Kernel parity and tolerance-tested |
| LayerNorm | 1.0× | 1.5× | Memory-bound behavior |
| Softmax | 1.0× | 1.2× | Limited by exp() |
| Compile Time | 0s | ~30s | Rust toolchain build cost |
| Binary Size | 0 | ~2MB | Shared-library footprint depends on build |

## Thread Safety

### Immutable Tensors

Tensors are immutable by default - operations return new tensors:

```julia
y = x + 1  # Creates new tensor, x unchanged
```

### Explicit Mutability

When mutation is needed, use explicit in-place operations:

```julia
y .= x .+ 1  # Modifies y in-place
```

### Backend Thread Safety

- **Julia**: Single-threaded, safe
- **Rust**: Thread-safe via Rayon, uses work-stealing
- **Accelerator targets**: fallback-safe strategy; runtime kernels are backend-specific roadmap work

## Future Architecture

### Planned Enhancements

1. **GPU Support**: CUDA/ROCm backends
2. **Distributed Training**: Multi-node parallelism
3. **Quantization**: INT8/INT4 inference
4. **Sparse Operations**: Efficient sparse tensor support
5. **JIT Compilation**: Runtime kernel fusion

### Research Directions

1. **Differentiable Verification**: Gradients through verification
2. **Neural Architecture Search**: With built-in correctness
3. **Formal Proofs**: Machine-checked correctness proofs

---

*Next: [Performance Tuning](Performance-Tuning.md) for optimization strategies*
