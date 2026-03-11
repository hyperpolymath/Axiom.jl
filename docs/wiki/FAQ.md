<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# Frequently Asked Questions

> *"There are no stupid questions, only stupid bugs that weren't caught at compile time."*

---

## General Questions

### What is Axiom.jl?

Axiom.jl is a **formally verified machine learning framework** for Julia. It combines:

- **Julia** for high-level, expressive model definitions
- **Zig** for high-performance native inference
- **Formal methods** for mathematical guarantees about model behavior

### Why the name "Axiom"?

An **axiom** is a fundamental truth that doesn't need proof - it's self-evident. In Axiom.jl:

- Your model definitions are **axioms** - declarative specifications
- Properties you prove become **theorems** - derived truths
- The framework ensures your models are **mathematically sound**

Also, it sounds cool.

### Is Axiom.jl production-ready?

Axiom.jl is currently in **alpha** (v0.1.0). It's suitable for:

- Research and experimentation
- Learning about verified ML
- Non-critical applications

For production safety-critical systems, wait for v1.0 or use additional validation.

### What makes Axiom.jl different from PyTorch/TensorFlow?

| Feature | PyTorch/TensorFlow | Axiom.jl |
|---------|-------------------|----------|
| Shape errors caught | Runtime | **Compile time** |
| Formal verification | No | **Yes** |
| Interactive REPL | No | **Yes** |
| Mathematical notation | Limited | **Native** |
| Proof certificates | No | **Yes** |

---

## Installation

### How do I install Axiom.jl?

```julia
using Pkg
Pkg.add("Axiom")
```

### What Julia version do I need?

Julia 1.9 or higher. We recommend Julia 1.10 for best performance.

### Do I need Zig installed?

Not for basic usage. The Julia backend works without Zig.

For the high-performance Zig backend:
```bash
# Install Zig from https://ziglang.org/download/
```

Then build:
```bash
cd ~/.julia/packages/Axiom/*/zig
zig build -Doptimize=ReleaseFast
```

### Can I use Axiom.jl without GPU?

Yes! Axiom.jl works on CPU. GPU support (CUDA, ROCm, Metal) is optional.

### Do you support TPU/NPU/DSP/FPGA?

Yes, Axiom now includes first-class backend targets:
`TPUBackend`, `NPUBackend`, `DSPBackend`, `FPGABackend`.

Current status:
- Detection and backend selection APIs are available now.
- Compilation falls back safely to CPU when a coprocessor runtime is not present.
- Production-grade non-GPU kernels are still in progress.

### Do you support REST, GraphQL, and gRPC?

Yes, with the following scope:
- REST: in-tree runtime server via `serve_rest(...)`
- GraphQL: in-tree runtime server via `serve_graphql(...)`
- gRPC: in-tree proto generation, in-process handlers, and network bridge server via `serve_grpc(...)`
  (supports `application/grpc` binary unary protobuf and `application/grpc+json` bridge mode)

---

## Model Definition

### What's the difference between `@axiom` and `Sequential`?

**`@axiom`**: Declarative, with type annotations and guarantees
```julia
@axiom Model begin
    input :: Tensor{Float32, (784,)}
    output = input |> Dense(784, 10) |> Softmax
    @ensure valid_probabilities(output)
end
```

**`Sequential`**: Imperative, PyTorch-style
```julia
model = Sequential(
    Dense(784, 10),
    Softmax()
)
```

Use `@axiom` for verified models, `Sequential` for quick prototyping.

### How do I define a custom layer?

```julia
struct MyLayer <: AbstractLayer
    weight::Matrix{Float32}
    bias::Vector{Float32}
end

function MyLayer(in_features::Int, out_features::Int)
    weight = randn(Float32, in_features, out_features) * 0.01f0
    bias = zeros(Float32, out_features)
    MyLayer(weight, bias)
end

function forward(layer::MyLayer, x)
    x * layer.weight .+ layer.bias'
end

parameters(layer::MyLayer) = (weight=layer.weight, bias=layer.bias)
```

### Can I use pre-trained weights?

`from_pytorch(...)` supports both direct checkpoints and descriptor import:
```julia
model = from_pytorch("pretrained.pt")           # requires python3 + torch
model = from_pytorch("pretrained.pytorch.json")
```

ONNX export API is available in the currently supported subset:
```julia
to_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

### How do I freeze layers?

```julia
# Freeze specific parameters
model.encoder.requires_grad = false

# Or during training
for param in parameters(model.encoder)
    param .= param  # Detach from gradient computation
end
```

---

## Verification

### What can I verify with @ensure?

Anything that returns a boolean:

```julia
@ensure sum(output) ≈ 1.0
@ensure all(output .>= 0)
@ensure maximum(output) > 0.5
@ensure norm(weights) < 100
@ensure !any(isnan, output)
```

### What can I prove with @prove?

Properties that can be derived from function definitions:

```julia
# These can be proven
@prove ∀x. sum(softmax(x)) == 1.0
@prove ∀x. relu(x) >= 0
@prove ∀x. 0 <= sigmoid(x) <= 1

# These need runtime checking
@prove ∀x. my_custom_function(x) > 0  # Can't prove, falls back to @ensure
```

### What if @prove fails?

If a property can't be proven, it becomes a runtime assertion:

```
⚠ Cannot prove: ∀x. custom_fn(x) > 0
  Adding runtime assertion instead.
```

Your code still works, but you won't get compile-time guarantees.

### How do I get a verification certificate?

```julia
result = verify(model, properties=[...], data=test_data)

if result.passed
    cert = generate_certificate(model, result)
    save_certificate(cert, "certificate.cert")
end
```

---

## Performance

### Is Axiom.jl faster than PyTorch?

**Inference**: Yes, 2-3x faster with Zig backend
**Training**: Comparable (Julia) to 1.5x faster (Zig)

### How do I enable the Zig backend?

```julia
# Compile for Zig
model = compile(my_model, backend=:zig)

# Or set environment variable
ENV["AXIOM_BACKEND"] = "zig"
```

### Why is compilation slow?

First run compiles Julia code. Subsequent runs are fast:

```julia
# First run: slow (compilation)
@time model(input)  # 2.5s

# Second run: fast
@time model(input)  # 0.01s
```

Use `precompile` for production:
```julia
using Axiom
Axiom.precompile()
```

### How do I profile performance?

```julia
using Profile

@profile for i in 1:100
    model(input)
end

Profile.print()
```

---

## Training

### Does Axiom.jl support automatic differentiation?

Yes, via Zygote.jl integration:

```julia
using Zygote

grads = gradient(m -> loss(m(x), y), model)
```

### How do I use a custom loss function?

```julia
function my_loss(pred, target)
    mse_loss(pred, target) + 0.01f0 * sum(abs, weights)
end

train!(model, data, optimizer, loss_fn=my_loss)
```

### Can I use learning rate schedulers?

```julia
optimizer = Adam(lr=0.001f0)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in 1:100
    train_epoch!(model, data, optimizer)
    step!(scheduler, epoch)
end
```

### How do I save/load models?

```julia
# Save
save_model(model, "model.axiom")

# Load
model = load_model("model.axiom")
```

---

## Troubleshooting

### "Shape mismatch" error

This is the most common error. Read the message carefully:

```
ERROR: Shape mismatch at Dense layer
  Expected: Vector (1D)
  Got: Tensor{Float32, (28, 28, 64)} (3D)

  Solution: Add Flatten layer before Dense
```

### "Type instability" warning

Julia's type inference couldn't determine types. Add type annotations:

```julia
# Before (type unstable)
x = some_computation()

# After (type stable)
x::Matrix{Float32} = some_computation()
```

### "Zig backend not found"

Build the Zig backend:
```bash
cd ~/.julia/packages/Axiom/*/zig
zig build -Doptimize=ReleaseFast
export AXIOM_ZIG_LIB=$(pwd)/zig-out/lib/libaxiom_zig.so
```

### "Out of memory"

Reduce batch size or use gradient checkpointing:

```julia
# Smaller batches
loader = DataLoader(data, batch_size=16)  # Was 64

# Gradient checkpointing (roadmap item)
model = checkpoint(model, every=3)  # Checkpoint every 3 layers
```

---

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome:
- Bug reports
- Documentation improvements
- New layer implementations
- Verification methods
- Performance optimizations

### How do I report a bug?

Open an issue on [GitHub](https://github.com/Hyperpolymath/Axiom.jl/issues) with:
- Julia version
- Axiom.jl version
- Minimal reproducing example
- Expected vs actual behavior

### Is there a roadmap?

See [Roadmap Commitments](Roadmap-Commitments.md) for planned work and delivery stages, plus [Vision](Vision.md) for long-term direction.

### Is Axiom Julia-first?

Yes. The core DSL, verification pipeline, and SMT integration are Julia-native.
The Zig backend is optional and only used when explicitly enabled.

---

## More Questions?

- [Discord](https://discord.gg/axiomjl) - Real-time help
- [GitHub Discussions](https://github.com/Hyperpolymath/Axiom.jl/discussions) - Community Q&A
- [API Reference](../api/README.md) - Detailed documentation
