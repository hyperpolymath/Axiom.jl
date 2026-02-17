# PyTorch to Axiom.jl Migration Guide

> *"Your models, but better."*

---

## Why Migrate?

| What You Get | PyTorch | Axiom.jl |
|--------------|---------|----------|
| Compile-time shape checking | | |
| Formal verification | | |
| 2-3x faster inference | | |
| REPL exploration | | |
| Keep existing models | | |

**The best part**: You don't have to rewrite anything. Import and go.

---

## Quick Migration (5 minutes)

### Step 1: Install

```julia
using Pkg
Pkg.add("Axiom")
```

### Step 2: Import Your Model

```julia
using Axiom

# Load a checkpoint directly (requires python3 + torch)
model = from_pytorch("my_model.pt")

# Or load an exported canonical descriptor
model = from_pytorch("my_model.pytorch.json")

# That's it! Now using Axiom.jl
output = model(input)
```

### Step 3: Verify (Optional but Recommended)

```julia
result = verify(model, properties=[ValidProbabilities()])
println(result)  # ✓ PASSED
```

---

## Migration Levels

### Level 1: Import Only (Zero Effort)

```julia
# Just load and use
model = from_pytorch("model.pt")
output = model(input)
```

**Benefits**: Faster inference, verification available

### Level 2: Add Verification (Minimal Effort)

```julia
# Load model
model = from_pytorch("model.pytorch.json")

# Add verification
verified_model = @axiom VerifiedModel begin
    input :: Tensor{Float32, (224, 224, 3)}
    output = input |> model |> Softmax

    @ensure valid_probabilities(output)
end
```

**Benefits**: Guaranteed properties, better error messages

### Level 3: Full Rewrite (Maximum Benefits)

```julia
# Native Axiom.jl definition
@axiom MyModel begin
    input :: Image(224, 224, 3)
    output :: Probabilities(1000)

    features = input |> ResNet50() |> GlobalAvgPool()
    output = features |> Dense(2048, 1000) |> Softmax

    @ensure valid_probabilities(output)
    @prove ∀x. no_nan(output(x))
end
```

**Benefits**: Full compile-time checking, formal proofs, maximum performance

---

## Layer-by-Layer Translation

### Linear / Dense

**PyTorch**:
```python
nn.Linear(in_features, out_features, bias=True)
```

**Axiom.jl**:
```julia
Dense(in_features, out_features, activation; bias=true)
```

**Example**:
```python
# PyTorch
self.fc1 = nn.Linear(784, 256)
x = F.relu(self.fc1(x))

# Axiom.jl
Dense(784, 256, relu)
# Or: Dense(784, 256) |> ReLU
```

### Conv2d

**PyTorch**:
```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

**Axiom.jl**:
```julia
Conv2d(in_channels, out_channels, kernel_size; stride=1, padding=0)
```

**Example**:
```python
# PyTorch
self.conv1 = nn.Conv2d(3, 64, 3, padding=1)

# Axiom.jl
Conv2d(3, 64, (3, 3), padding=1)
# Or: Conv(3, 64, (3, 3), padding=:same)
```

### BatchNorm

**PyTorch**:
```python
nn.BatchNorm2d(num_features)
```

**Axiom.jl**:
```julia
BatchNorm(num_features)
```

### MaxPool / AvgPool

**PyTorch**:
```python
nn.MaxPool2d(kernel_size, stride=None)
nn.AvgPool2d(kernel_size, stride=None)
```

**Axiom.jl**:
```julia
MaxPool(kernel_size; stride=kernel_size)
AvgPool(kernel_size; stride=kernel_size)
```

### Dropout

**PyTorch**:
```python
nn.Dropout(p=0.5)
```

**Axiom.jl**:
```julia
Dropout(0.5)
```

### Activations

| PyTorch | Axiom.jl (function) | Axiom.jl (layer) |
|---------|---------------------|------------------|
| `F.relu(x)` | `relu(x)` | `ReLU()` |
| `F.sigmoid(x)` | `sigmoid(x)` | `Sigmoid()` |
| `torch.tanh(x)` | `tanh(x)` | `Tanh()` |
| `F.softmax(x, dim=-1)` | `softmax(x)` | `Softmax()` |
| `F.gelu(x)` | `gelu(x)` | `GELU()` |
| `F.leaky_relu(x, 0.1)` | `leaky_relu(x, 0.1)` | `LeakyReLU(0.1)` |

### Sequential

**PyTorch**:
```python
nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

**Axiom.jl**:
```julia
Sequential(
    Dense(784, 256, relu),
    Dense(256, 10)
)
# Or with explicit layers:
Sequential(
    Dense(784, 256),
    ReLU(),
    Dense(256, 10)
)
```

---

## Complete Model Translation

### PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
```

### Axiom.jl

```julia
using Axiom

@axiom MNISTClassifier begin
    input :: Image(28, 28, 1)
    output :: Probabilities(10)

    # Conv block 1
    x = input |> Conv(32, (3,3), padding=:same) |> ReLU |> MaxPool((2,2))

    # Conv block 2
    x = x |> Conv(64, (3,3), padding=:same) |> ReLU |> MaxPool((2,2))

    # Classifier
    x = x |> Flatten
    x = x |> Dense(64 * 7 * 7, 128, relu) |> Dropout(0.5)
    output = x |> Dense(128, 10) |> Softmax

    # Guarantees (impossible in PyTorch!)
    @ensure valid_probabilities(output)
end
```

**Key differences**:
1. Shape declared upfront
2. No manual `view()` / reshape
3. Built-in guarantees
4. Cleaner syntax

---

## Training Code Translation

### PyTorch

```python
model = MNISTClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
```

### Axiom.jl

```julia
model = MNISTClassifier()
optimizer = Adam(lr=0.001f0)

history = train!(model, dataloader, optimizer,
    epochs = 10,
    loss_fn = crossentropy
)
```

Or for more control:

```julia
model = MNISTClassifier()
optimizer = Adam(lr=0.001f0)

for epoch in 1:10
    for (batch_x, batch_y) in dataloader
        # Forward
        output = model(batch_x)
        loss = crossentropy(output, batch_y)

        # Backward (using Zygote.jl)
        grads = gradient(m -> crossentropy(m(batch_x), batch_y), model)

        # Update
        step!(optimizer, parameters(model), grads)
    end
end
```

---

## Loading Pre-trained Models

### From exported descriptor JSON

```julia
# Canonical descriptor format
model = from_pytorch("resnet50.pytorch.json")
```

Direct `.pt/.pth/.ckpt` loading is supported through the built-in bridge script
(`scripts/pytorch_to_axiom_descriptor.py`) and requires `python3` + `torch`.

### From Hugging Face (Roadmap)

Direct `from_huggingface(...)` import is not currently shipped.
Use one of the current interop paths:

```julia
# 1) Export or bridge to a PyTorch checkpoint/descriptor
model = from_pytorch("model.pt")
model = from_pytorch("model.pytorch.json")

# 2) Export to ONNX from external tooling, then import via your ONNX flow
to_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

### From ONNX

```julia
model = from_onnx("model.onnx")
```

---

## Data Loading

### PyTorch DataLoader

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Axiom.jl DataLoader

```julia
loader = DataLoader((X, y), batch_size=32, shuffle=true)

# Iterate
for (batch_x, batch_y) in loader
    # ...
end
```

---

## Common Gotchas

### 1. Shape Convention

**PyTorch**: `(N, C, H, W)` - Batch, Channel, Height, Width

**Axiom.jl**: `(N, H, W, C)` - Batch, Height, Width, Channel

The `from_pytorch` function handles this automatically.

### 2. Indexing

**PyTorch**: 0-indexed

**Julia**: 1-indexed

### 3. Weight Layout

**PyTorch Linear**: `(out_features, in_features)`

**Axiom.jl Dense**: `(in_features, out_features)`

Automatically transposed during import.

### 4. Softmax Dimension

**PyTorch**: `F.softmax(x, dim=-1)` (must specify)

**Axiom.jl**: `softmax(x)` (last dim by default)

---

## Verification Bonus

After migration, you get verification for free:

```julia
# Verify your migrated model
model = from_pytorch("model.pytorch.json")

result = verify(model,
    properties = [
        ValidProbabilities(),
        FiniteOutput(),
        LocalLipschitz(0.01, 0.1)
    ],
    data = test_loader
)

# Generate certificate for deployment
cert = generate_certificate(model, result)
```

---

## Performance Comparison

After migration, you can compile for production:

```julia
# Development (Julia backend)
dev_model = from_pytorch("model.pytorch.json")

# Production (Rust backend) - 2-3x faster
prod_model = compile(dev_model, backend=:rust, optimize=:aggressive)

# Benchmark
@time dev_model(test_input)   # 0.012s
@time prod_model(test_input)  # 0.004s
```

---

## Need Help?

- [FAQ](FAQ.md) - Common questions
- [Discord](https://discord.gg/axiomjl) - Real-time help
- [GitHub Issues](https://github.com/Hyperpolymath/Axiom.jl/issues) - Bug reports
