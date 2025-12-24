# The @axiom DSL

> *Declare what you want, not how to compute it.*

---

## Overview

The `@axiom` macro is the heart of Axiom.jl. It lets you define neural networks as **mathematical specifications** with **built-in guarantees**.

```julia
@axiom ModelName begin
    # Type declarations
    input :: InputType
    output :: OutputType

    # Layer composition
    layer1 = input |> SomeLayer(...)
    layer2 = layer1 |> AnotherLayer(...)
    output = layer2 |> FinalLayer(...)

    # Guarantees
    @ensure condition1
    @prove property1
end
```

---

## Quick Reference

```julia
# Minimal example
@axiom Classifier begin
    input :: Tensor{Float32, (28, 28, 1)}
    output :: Tensor{Float32, (10,)}

    output = input |> Flatten |> Dense(784, 10) |> Softmax
end

# Create instance
model = Classifier()

# Use it
predictions = model(image_batch)
```

---

## Type Declarations

### Static Shapes

```julia
# Exact shape known at compile time
input :: Tensor{Float32, (224, 224, 3)}
output :: Tensor{Float32, (1000,)}
```

### Dynamic Dimensions

```julia
# Batch dimension can vary
input :: Tensor{Float32, (:batch, 784)}
output :: Tensor{Float32, (:batch, 10)}

# Sequence length can vary
input :: Tensor{Float32, (:batch, :seq, 512)}
```

### Semantic Types

```julia
# High-level type aliases
input :: Image(224, 224, 3)
output :: Probabilities(1000)

# These expand to Tensor types with constraints
# Probabilities guarantees: sum=1, all>=0
```

---

## Layer Composition

### Pipeline Operator

The `|>` operator chains layers:

```julia
# Each step transforms the data
features = input |> Conv(64, (3,3)) |> ReLU |> MaxPool((2,2))
```

This is equivalent to:

```julia
features = MaxPool((2,2))(ReLU()(Conv(64, (3,3))(input)))
```

But much more readable!

### Named Intermediate Values

```julia
@axiom CNN begin
    input :: Image(32, 32, 3)
    output :: Probabilities(10)

    # Name each stage for clarity
    conv1 = input |> Conv(32, (3,3)) |> BatchNorm() |> ReLU
    pool1 = conv1 |> MaxPool((2,2))

    conv2 = pool1 |> Conv(64, (3,3)) |> BatchNorm() |> ReLU
    pool2 = conv2 |> MaxPool((2,2))

    flat = pool2 |> GlobalAvgPool() |> Flatten

    output = flat |> Dense(64, 10) |> Softmax
end
```

### Parallel Paths

```julia
@axiom InceptionModule begin
    input :: Tensor{Float32, (:batch, 28, 28, 256)}
    output :: Tensor{Float32, (:batch, 28, 28, 256)}

    # Multiple parallel paths
    path1 = input |> Conv(64, (1,1))
    path2 = input |> Conv(64, (1,1)) |> Conv(64, (3,3), padding=:same)
    path3 = input |> Conv(64, (1,1)) |> Conv(64, (5,5), padding=:same)
    path4 = input |> MaxPool((3,3), stride=1, padding=:same) |> Conv(64, (1,1))

    # Concatenate along channel dimension
    output = concat(path1, path2, path3, path4, dim=4)
end
```

### Residual Connections

```julia
@axiom ResBlock begin
    input :: Tensor{Float32, (:batch, H, W, C)}
    output :: Tensor{Float32, (:batch, H, W, C)}

    # Residual branch
    residual = input |> Conv(C, (3,3), padding=:same) |> BatchNorm() |> ReLU
    residual = residual |> Conv(C, (3,3), padding=:same) |> BatchNorm()

    # Skip connection
    output = (input + residual) |> ReLU

    @ensure shape(output) == shape(input)
end
```

---

## Guarantees

### @ensure: Runtime Assertions

```julia
@axiom SafeClassifier begin
    # ...

    # Checked at runtime for every forward pass
    @ensure sum(output) ≈ 1.0
    @ensure all(output .>= 0)
    @ensure !any(isnan, output)
end
```

### @prove: Compile-Time Proofs

```julia
@axiom VerifiedClassifier begin
    # ...

    # Proven at compile time (if possible)
    @prove ∀x. sum(softmax(x)) == 1.0
    @prove ∀x. all(softmax(x) .>= 0)
end
```

If a property can't be proven, it becomes a runtime assertion with a warning.

---

## Complete Example

```julia
@axiom VGG16 begin
    input :: Image(224, 224, 3)
    output :: Probabilities(1000)

    # Block 1
    x = input |> Conv(64, (3,3), padding=:same) |> ReLU
    x = x |> Conv(64, (3,3), padding=:same) |> ReLU
    x = x |> MaxPool((2,2))

    # Block 2
    x = x |> Conv(128, (3,3), padding=:same) |> ReLU
    x = x |> Conv(128, (3,3), padding=:same) |> ReLU
    x = x |> MaxPool((2,2))

    # Block 3
    x = x |> Conv(256, (3,3), padding=:same) |> ReLU
    x = x |> Conv(256, (3,3), padding=:same) |> ReLU
    x = x |> Conv(256, (3,3), padding=:same) |> ReLU
    x = x |> MaxPool((2,2))

    # Block 4
    x = x |> Conv(512, (3,3), padding=:same) |> ReLU
    x = x |> Conv(512, (3,3), padding=:same) |> ReLU
    x = x |> Conv(512, (3,3), padding=:same) |> ReLU
    x = x |> MaxPool((2,2))

    # Block 5
    x = x |> Conv(512, (3,3), padding=:same) |> ReLU
    x = x |> Conv(512, (3,3), padding=:same) |> ReLU
    x = x |> Conv(512, (3,3), padding=:same) |> ReLU
    x = x |> MaxPool((2,2))

    # Classifier
    x = x |> Flatten
    x = x |> Dense(4096, relu) |> Dropout(0.5)
    x = x |> Dense(4096, relu) |> Dropout(0.5)
    output = x |> Dense(1000) |> Softmax

    # Guarantees
    @ensure valid_probabilities(output)
    @ensure no_nan(output)
end
```

---

## Comparison with PyTorch

### PyTorch

```python
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        # Have to manually track shapes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # ... 13 more layers ...
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Is 7*7 right? Hope so!
            nn.ReLU(),
            nn.Dropout(0.5),
            # ...
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Magic reshape
        x = self.classifier(x)
        return F.softmax(x, dim=1)
```

### Axiom.jl

```julia
@axiom VGG16 begin
    input :: Image(224, 224, 3)
    output :: Probabilities(1000)

    # Shapes are verified at compile time
    # No manual tracking needed
    # No "magic" reshapes

    x = input |> Conv(64, (3,3), padding=:same) |> ReLU
    # ... (see full example above)

    @ensure valid_probabilities(output)
end
```

**Key differences**:
- Shapes declared upfront, verified by compiler
- No manual `view()` calls
- Guarantees are part of the model definition
- Errors caught immediately, not at runtime

---

## Advanced Features

### Conditional Layers

```julia
@axiom ConditionalModel begin
    # ...

    # Training vs inference
    x = if training
        x |> Dropout(0.5)
    else
        x
    end
end
```

### Parameter Sharing

```julia
@axiom SiameseNetwork begin
    input1 :: Image(224, 224, 3)
    input2 :: Image(224, 224, 3)
    output :: Tensor{Float32, (1,)}

    # Shared encoder
    encoder = Sequential(
        Conv(64, (3,3)) |> ReLU |> MaxPool((2,2)),
        Conv(128, (3,3)) |> ReLU |> MaxPool((2,2)),
        Flatten,
        Dense(256)
    )

    # Same encoder applied to both inputs
    emb1 = input1 |> encoder
    emb2 = input2 |> encoder

    # Compare embeddings
    diff = abs(emb1 - emb2)
    output = diff |> Dense(1) |> Sigmoid
end
```

### Attention Mechanisms

```julia
@axiom Attention begin
    query :: Tensor{Float32, (:batch, :seq, 512)}
    key :: Tensor{Float32, (:batch, :seq, 512)}
    value :: Tensor{Float32, (:batch, :seq, 512)}
    output :: Tensor{Float32, (:batch, :seq, 512)}

    # Scaled dot-product attention
    scores = query @ transpose(key) / sqrt(512)
    weights = scores |> Softmax
    output = weights @ value

    @ensure shape(output) == shape(query)
end
```

---

## Error Messages

Axiom.jl provides helpful error messages:

```julia
@axiom BadModel begin
    input :: Tensor{Float32, (28, 28, 1)}
    output = input |> Dense(10)  # Error!
end

# ERROR: Shape mismatch at Dense layer
#
# Expected input: Vector or 2D Matrix
# Got: Tensor{Float32, (28, 28, 1)} (3D)
#
# Problem: Dense layer expects flattened input
#          Your input has shape (28, 28, 1)
#
# Solution: Add Flatten layer before Dense
#
#     output = input |> Flatten |> Dense(784, 10)
#                      ^^^^^^^^
#
# Would you like me to apply this fix? [y/N]
```

---

## Next Steps

- [Verification System](Verification.md) - Deep dive into @ensure and @prove
- [Type System](Type-System.md) - How shapes are tracked
- [Custom Layers](Custom-Layers.md) - Building your own layers
- [API Reference](../api/README.md) - Complete layer documentation
