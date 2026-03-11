<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
<div align="center">

# Axiom.jl

### *Provably Correct Machine Learning*

[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org/)
[![Rust](https://img.shields.io/badge/Rust-Backend-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-PMPL--1.0-blue.svg)](../../LICENSE)

**The ML framework where bugs are caught before runtime, not after deployment.**

[Quick Start](#-quick-start) | [User Guide](User-Guide.md) | [Developer Guide](Developer-Guide.md) | [Release Checklist](../../RELEASE-CHECKLIST.adoc) | [Vision](Vision.md) | [Roadmap](../../ROADMAP.md) | [API Reference](API-Reference.md)

</div>

---

## What is Axiom.jl?

Axiom.jl is a **revolutionary machine learning framework** that combines:

- **Julia's expressiveness** - Write models as mathematical expressions
- **Rust's performance** - Production-grade speed and safety
- **Formal verification** - Prove properties about your models
- **PyTorch compatibility** - Import existing models seamlessly

```julia
@axiom ImageClassifier begin
    input :: Tensor{Float32, (224, 224, 3)}
    output :: Probabilities(1000)

    features = input |> ResNet50(pretrained=true) |> GlobalAvgPool()
    output = features |> Dense(2048, 1000) |> Softmax

    # These are GUARANTEED at compile time
    @ensure sum(output) ≈ 1.0
    @ensure all(output .>= 0)
end
```

---

## Why Axiom?

### The Problem with PyTorch

```python
# PyTorch: Fails at runtime (after you've waited 3 hours for training)
class BrokenModel(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc1 = nn.Linear(100, 10)  # Wrong size!

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)  # RuntimeError: size mismatch
        return x
```

### The Axiom Solution

```julia
# Axiom.jl: Fails at COMPILE time (immediately)
@axiom BrokenModel begin
    input :: Tensor{Float32, (224, 224, 3)}

    features = input |> Conv(64, (3,3))
    output = features |> Dense(10)  # COMPILE ERROR!
    # Shape mismatch: Conv output is (222, 222, 64)
    # Dense expects vector. Add Flatten layer.
end
```

**Result**: Hours saved. Bugs caught. Sleep restored.

---

## Quick Start

### Installation

```julia
using Pkg
Pkg.add("Axiom")
```

### Your First Model (30 seconds)

```julia
using Axiom

# Define a verified model
model = Sequential(
    Dense(784, 256, relu),
    Dense(256, 10),
    Softmax()
)

# Run inference
output = model(randn(Float32, 32, 784))

# Verify properties
@ensure all(sum(output, dims=2) .≈ 1.0)
```

### Verification-First Workflow (60 seconds)

```julia
using Axiom

# Define and verify a model
model = Sequential(
    Dense(784, 128, relu),
    Dense(128, 10),
    Softmax()
)

sample = Tensor(randn(Float32, 16, 784))
batch = [(sample, nothing)]
result = verify(model, properties=[ValidProbabilities(), FiniteOutput()], data=batch)
```

---

## Optional Backends

Axiom is Julia-first. The Rust backend is optional and used only when you
explicitly enable it (e.g., for high-performance kernels or SMT runner
hardening). Most users can ignore it entirely.

---

## Core Features

| Feature | PyTorch | TensorFlow | Axiom.jl |
|---------|---------|------------|----------|
| Shape checking | Runtime | Runtime | **Compile time** |
| Formal proofs | No | No | **Yes** |
| REPL exploration | No | No | **Yes** |
| Performance | Good | Good | **Better** (Rust) |
| Safety certification | No | No | **Yes** |
| Learning curve | Low | Medium | Low |

---

## Documentation Map

### Getting Started
- [User Guide](User-Guide.md) - Installation and day-1 workflows
- [Developer Guide](Developer-Guide.md) - Local development, checks, release flow
- [Release Checklist](../../RELEASE-CHECKLIST.adoc) - Pre-release and release-day checklist
- [Vision & Philosophy](Vision.md) - Why Axiom.jl exists
- [Tutorials](Tutorials.md) - From beginner to expert
- [PyTorch Migration](Migration-Guide.md) - Coming from PyTorch?
- [FAQ](FAQ.md) - Common questions answered

### Core Concepts
- [@axiom DSL](Axiom-DSL.md) - The declarative model definition
- [Verification System](Verification.md) - @ensure and @prove
- [Architecture](Architecture.md) - Deep dive into design
- [Rust Backend](Rust-Backend.md) - Performance architecture

### API Reference
- [Complete API Reference](API-Reference.md) - All functions, types, macros
- Layers - Dense, Conv, Pool, etc.
- Activations - ReLU, GELU, Softmax, etc.
- Optimizers - Adam, SGD, etc.
- Loss Functions - CrossEntropy, MSE, etc.

### Production
- [Performance Tuning](Performance-Tuning.md) - Optimize for speed
- [Safety-Critical Applications](Safety-Critical.md) - FDA, ISO 26262, DO-178C
- [Certification Readiness](Certification-Readiness.md) - Compliance checklists by domain
- [Deployment Guide](Deployment.md) - Server, edge, cloud
- [Ecosystem & Integrations](Ecosystem.md) - Connect to Julia/Python world

### Compare
- [Framework Comparison](Framework-Comparison.md) - vs PyTorch, TensorFlow, JAX

---

## Interactive Examples

Try these in your Julia REPL:

<details>
<summary><b>Example 1: Image Classification</b></summary>

```julia
using Axiom

@axiom CIFAR10Classifier begin
    input :: Image(32, 32, 3)
    output :: Probabilities(10)

    # Convolutional feature extractor
    conv1 = input |> Conv(32, (3,3), padding=:same) |> BatchNorm() |> ReLU
    conv2 = conv1 |> Conv(64, (3,3), padding=:same) |> BatchNorm() |> ReLU
    pool1 = conv2 |> MaxPool((2,2))

    conv3 = pool1 |> Conv(128, (3,3), padding=:same) |> BatchNorm() |> ReLU
    pool2 = conv3 |> MaxPool((2,2))

    # Classifier head
    flat = pool2 |> GlobalAvgPool()
    output = flat |> Dense(128, 10) |> Softmax

    @ensure valid_probabilities(output)
end

model = CIFAR10Classifier()
```
</details>

<details>
<summary><b>Example 2: Transformer Block</b></summary>

```julia
using Axiom

@axiom TransformerBlock begin
    input :: Tensor{Float32, (:batch, :seq, 512)}
    output :: Tensor{Float32, (:batch, :seq, 512)}

    # Multi-head attention
    attn = input |> MultiHeadAttention(heads=8, dim=512)
    add1 = input + attn  # Residual connection
    norm1 = add1 |> LayerNorm(512)

    # Feed-forward
    ff = norm1 |> Dense(512, 2048, gelu) |> Dense(2048, 512)
    add2 = norm1 + ff  # Residual connection
    output = add2 |> LayerNorm(512)

    @ensure shape(output) == shape(input)
end
```
</details>

<details>
<summary><b>Example 3: Verified Medical AI</b></summary>

```julia
using Axiom

@axiom MedicalDiagnosis begin
    input :: MedicalImage(512, 512, 1)
    output :: Diagnosis(5)  # 5 possible conditions

    # Feature extraction
    features = input |> ResNet50(pretrained=true)

    # Classifier
    output = features |> Dense(2048, 5) |> Softmax

    # CRITICAL: Medical safety guarantees
    @ensure valid_probabilities(output)
    @ensure no_nan(output)
    @ensure confidence_bounded(output, min=0.6)  # Require high confidence

    # For FDA approval: formal properties
    @prove ∀x. sum(output(x)) == 1.0
    @prove ∀x ε. (ε < 0.01) ⟹ stable(output(x), output(x + ε))
end
```
</details>

---

## Benchmarks

Performance comparison on common tasks:

| Task | PyTorch | Axiom.jl (Julia) | Axiom.jl (Rust) |
|------|---------|------------------|-----------------|
| MNIST training | 1.0x | 1.1x | **1.8x** |
| ResNet inference | 1.0x | 0.9x | **2.3x** |
| BERT inference | 1.0x | 1.0x | **2.1x** |
| Compilation | 30s | 2s | 5s |

*Benchmarks on Intel i9-12900K, RTX 3090. Your mileage may vary.*

---

## Community

- [GitHub Discussions](https://github.com/Hyperpolymath/Axiom.jl/discussions) - Questions and ideas
- [Discord](https://discord.gg/axiomjl) - Real-time chat
- [Twitter](https://twitter.com/axiomjl) - Updates and news
- [Contributing](CONTRIBUTING.md) - Help us build the future

---

## What's Next?

1. **[Read the Vision](Vision.md)** - Understand why we built this
2. **[Install Axiom.jl](Installation.md)** - Get started in 2 minutes
3. **[Build Your First Model](../tutorials/first-model.md)** - Hands-on tutorial
4. **[Join the Community](https://discord.gg/axiomjl)** - Get help and share

---

<div align="center">

**Built with by the Axiom.jl team**

*The future of ML is verified.*

</div>
