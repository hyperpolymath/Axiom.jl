<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# API Reference

> Complete reference for all Axiom.jl functions, types, and macros

## Core Types

### Tensor{T, Shape}

The fundamental data structure for all neural network operations.

```julia
struct Tensor{T<:Number, Shape<:Tuple}
    data::Array{T}
end
```

**Type Parameters:**
- `T`: Element type (e.g., `Float32`, `Float64`, `Float16`)
- `Shape`: Tuple of dimensions (e.g., `(784,)`, `(32, 28, 28, 1)`)

**Constructors:**
```julia
# From existing array (shape inferred)
Tensor(data::Array{T}) where T

# With explicit shape
Tensor{Float32, (784,)}(data)

# Zeros
zeros(Tensor{Float32, (100, 784)})

# Random (normal distribution)
randn(Tensor{Float32, (batch, features)})

# Random (uniform)
rand(Tensor{Float32, (batch, features)})

# From range
Tensor(1:100)
```

**Properties:**
```julia
size(t::Tensor)      # Returns shape tuple
length(t::Tensor)    # Total number of elements
eltype(t::Tensor)    # Element type
ndims(t::Tensor)     # Number of dimensions
```

**Operations:**
```julia
# Arithmetic
a + b, a - b, a * b, a / b
a .+ b, a .- b, a .* b, a ./ b  # Broadcasting

# Indexing
t[1, :]              # Slice
t[1:10, :]           # Range
view(t, 1:10, :)     # View (no copy)

# Reshaping
reshape(t, new_shape)
flatten(t)
squeeze(t)
unsqueeze(t, dim)

# Reduction
sum(t), sum(t, dims=1)
mean(t), mean(t, dims=1)
maximum(t), minimum(t)
argmax(t), argmin(t)
```

---

## Layers

### Dense

Fully connected (linear) layer.

```julia
Dense(in_features => out_features;
      activation=identity,
      use_bias=true,
      init=glorot_uniform)
```

**Parameters:**
- `in_features::Int`: Input dimension
- `out_features::Int`: Output dimension
- `activation`: Activation function (default: `identity`)
- `use_bias::Bool`: Include bias term (default: `true`)
- `init`: Weight initialization function

**Example:**
```julia
layer = Dense(784 => 256, activation=relu)
y = forward(layer, x)  # x: (batch, 784) → y: (batch, 256)
```

**Shapes:**
- Input: `(batch, in_features)`
- Output: `(batch, out_features)`
- Weight: `(out_features, in_features)`
- Bias: `(out_features,)`

---

### Conv2D

2D Convolution layer.

```julia
Conv2D(in_channels => out_channels, kernel_size;
       stride=1,
       padding=0,
       dilation=1,
       groups=1,
       activation=identity,
       use_bias=true)
```

**Parameters:**
- `in_channels::Int`: Number of input channels
- `out_channels::Int`: Number of output channels
- `kernel_size::Tuple{Int,Int}`: Kernel dimensions `(H, W)`
- `stride`: Stride (default: `1` or `(1, 1)`)
- `padding`: Padding (default: `0` or `(0, 0)`)
- `dilation`: Dilation (default: `1`)
- `groups`: Grouped convolution (default: `1`)
- `activation`: Activation function

**Example:**
```julia
conv = Conv2D(3 => 64, (3, 3), padding=1, activation=relu)
y = forward(conv, x)  # x: (batch, H, W, 3) → y: (batch, H, W, 64)
```

**Shapes (NHWC format):**
- Input: `(batch, height, width, in_channels)`
- Output: `(batch, out_height, out_width, out_channels)`
- Weight: `(kernel_h, kernel_w, in_channels, out_channels)`

---

### Conv1D

1D Convolution layer.

```julia
Conv1D(in_channels => out_channels, kernel_size;
       stride=1,
       padding=0,
       activation=identity)
```

**Shapes:**
- Input: `(batch, length, channels)`
- Output: `(batch, out_length, out_channels)`

---

### MaxPool2D / AvgPool2D

Pooling layers.

```julia
MaxPool2D(kernel_size; stride=kernel_size, padding=0)
AvgPool2D(kernel_size; stride=kernel_size, padding=0)
```

**Example:**
```julia
pool = MaxPool2D((2, 2))
y = forward(pool, x)  # Halves spatial dimensions
```

---

### GlobalAvgPool2D / GlobalMaxPool2D

Global pooling (reduces to single value per channel).

```julia
gap = GlobalAvgPool2D()
y = forward(gap, x)  # x: (batch, H, W, C) → y: (batch, C)
```

---

### BatchNorm

Batch normalization.

```julia
BatchNorm(num_features;
          eps=1e-5,
          momentum=0.1,
          affine=true,
          track_running_stats=true)
```

**Example:**
```julia
bn = BatchNorm(64)
y = forward(bn, x)  # Normalizes over batch dimension
```

---

### LayerNorm

Layer normalization.

```julia
LayerNorm(normalized_shape;
          eps=1e-5,
          elementwise_affine=true)
```

**Example:**
```julia
ln = LayerNorm(768)  # For transformer hidden size
y = forward(ln, x)
```

---

### Dropout

Dropout regularization.

```julia
Dropout(p=0.5)
```

**Note:** Only active during training.

```julia
dropout = Dropout(0.3)
y = forward(dropout, x)          # Training: applies dropout
y = forward(dropout, x, training=false)  # Inference: identity
```

---

### Flatten

Flatten spatial dimensions.

```julia
flatten = Flatten()
y = forward(flatten, x)  # x: (batch, H, W, C) → y: (batch, H*W*C)
```

---

### Sequential / Chain

Container for sequential layers.

```julia
model = Sequential(
    Dense(784 => 256, activation=relu),
    Dense(256 => 10, activation=softmax)
)

# Or using @axiom
model = @axiom begin
    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)
end
```

---

## Activation Functions

All activation functions can be used as:
1. Layer activation: `Dense(10 => 10, activation=relu)`
2. Standalone: `y = relu(x)`
3. In-place: `relu!(x)`

### Available Activations

| Function | Formula | Use Case |
|----------|---------|----------|
| `relu` | `max(0, x)` | Default choice |
| `relu6` | `min(max(0, x), 6)` | Mobile nets |
| `leaky_relu(x, α=0.01)` | `x > 0 ? x : αx` | Dying ReLU fix |
| `elu(x, α=1.0)` | `x > 0 ? x : α(exp(x)-1)` | Smooth at 0 |
| `selu` | Scaled ELU | Self-normalizing |
| `gelu` | Gaussian error | Transformers |
| `sigmoid` | `1/(1+exp(-x))` | Binary output |
| `tanh` | `(exp(x)-exp(-x))/(exp(x)+exp(-x))` | Bounded output |
| `softmax` | `exp(x)/sum(exp(x))` | Classification |
| `log_softmax` | `log(softmax(x))` | NLL loss |
| `swish` / `silu` | `x * sigmoid(x)` | Modern nets |
| `mish` | `x * tanh(softplus(x))` | Alternative to swish |
| `softplus` | `log(1 + exp(x))` | Smooth ReLU |
| `hard_sigmoid` | Piecewise linear | Fast sigmoid |
| `hard_swish` | Piecewise | Fast swish |

**Examples:**
```julia
# As layer activation
Dense(10 => 10, activation=gelu)

# Standalone
y = gelu(x)

# With parameters
y = leaky_relu(x, 0.2)

# In-place
relu!(x)
```

---

## Loss Functions

### CrossEntropyLoss

For multi-class classification.

```julia
loss = CrossEntropyLoss(;
    reduction=:mean,  # :mean, :sum, :none
    weight=nothing,   # Class weights
    ignore_index=-1,  # Index to ignore
    label_smoothing=0.0
)

l = loss(predictions, targets)
```

**Example:**
```julia
loss_fn = CrossEntropyLoss()
predictions = rand(Float32, 32, 10)  # (batch, classes)
targets = rand(1:10, 32)             # Class indices
l = loss_fn(predictions, targets)
```

---

### BinaryCrossEntropyLoss

For binary classification.

```julia
loss = BinaryCrossEntropyLoss(;
    reduction=:mean,
    pos_weight=nothing
)
```

---

### MSELoss

Mean squared error.

```julia
loss = MSELoss(reduction=:mean)
l = loss(predictions, targets)
```

---

### L1Loss

Mean absolute error.

```julia
loss = L1Loss(reduction=:mean)
```

---

### HuberLoss

Smooth L1 loss.

```julia
loss = HuberLoss(delta=1.0, reduction=:mean)
```

---

## Optimizers

### Adam

Adaptive moment estimation.

```julia
optimizer = Adam(;
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)
```

**Usage:**
```julia
opt = Adam(lr=0.001)
for epoch in 1:100
    grads = gradient(model, x, y)
    update!(opt, model, grads)
end
```

---

### SGD

Stochastic gradient descent.

```julia
optimizer = SGD(;
    lr=0.01,
    momentum=0.0,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=false
)
```

---

### AdamW

Adam with decoupled weight decay.

```julia
optimizer = AdamW(;
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

---

### RMSprop

```julia
optimizer = RMSprop(;
    lr=0.01,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0.0,
    momentum=0.0
)
```

---

## DSL Macros

### @axiom

Define a model with the Axiom DSL.

```julia
model = @axiom [options] begin
    layer1
    layer2
    ...
end
```

**Options:**
- `backend=ZigBackend("/path/to/libaxiom_zig.so")`: Backend selection
- `verified=true`: Enable verification
- `name="MyModel"`: Model name

**Example:**
```julia
model = @axiom backend=JuliaBackend() verified=true begin
    @ensure size(x, 2) == 784 "Input must be 784-dim"
    Dense(784 => 256, activation=relu)
    Dropout(0.5)
    Dense(256 => 10, activation=softmax)
    @ensure size(output, 2) == 10 "Output must be 10 classes"
end
```

---

### @ensure

Runtime assertion.

```julia
@ensure condition "error message"
```

**Examples:**
```julia
@ensure size(x, 1) == batch_size "Batch size mismatch"
@ensure all(isfinite, x) "Input contains NaN/Inf"
@ensure 0 ≤ probability ≤ 1 "Invalid probability"
```

---

### @prove

Compile-time verification.

```julia
@prove Property args...
```

**Available Properties:**
```julia
@prove BoundedOutputs(min, max) model
@prove Lipschitz(constant) model
@prove Monotonic(region) model
@prove Robust(epsilon) model
```

---

## Training Functions

### train!

High-level training function.

```julia
train!(model, train_loader;
    epochs=10,
    optimizer=Adam(),
    loss=CrossEntropyLoss(),
    val_loader=nothing,
    callbacks=[],
    verbose=true
)
```

**Example:**
```julia
train!(model, train_loader,
    epochs=50,
    optimizer=Adam(lr=0.001),
    loss=CrossEntropyLoss(),
    val_loader=val_loader,
    callbacks=[
        EarlyStopping(patience=5),
        ModelCheckpoint("best.axiom")
    ]
)
```

---

### gradient

Compute gradients.

```julia
grads = gradient(model, x, y, loss_fn)
```

---

### value_and_gradient

Compute loss and gradients together.

```julia
loss, grads = value_and_gradient(model, x, y, loss_fn)
```

---

### update!

Apply gradients to model.

```julia
update!(optimizer, model, grads)
```

---

## Backends

### JuliaBackend

Pure Julia implementation.

```julia
backend = JuliaBackend()
model = @axiom backend=JuliaBackend() begin
    Dense(10 => 10)
end
```

---

### ZigBackend

High-performance Zig native implementation.

```julia
backend = ZigBackend()

# Check availability
zig_available()  # Returns Bool
```

---

### Coprocessor Targets

Non-GPU accelerator targets with fallback-safe compilation strategy.

```julia
backend = detect_coprocessor()  # TPU/NPU/DSP/FPGA or nothing
if backend !== nothing
    model_accel = compile(model, backend=backend, verify=false, optimize=:none)
end
```

---

## Verification

### VerificationProperty

Base type for verification properties.

```julia
abstract type VerificationProperty end

struct BoundedOutputs <: VerificationProperty
    min_val::Float32
    max_val::Float32
end

struct LipschitzProperty <: VerificationProperty
    constant::Float32
end
```

---

### verify

Verify a property on a model.

```julia
result = verify(property, model, input_bounds)
```

---

### generate_certificate

Generate verification certificate.

```julia
cert = generate_certificate(model, properties)
```

---

### save_certificate / load_certificate

```julia
save_certificate(cert, "model.cert")
cert = load_certificate("model.cert")
```

---

## I/O Functions

### save_model / load_model

```julia
save_model(model, "model.axiom")
model = load_model("model.axiom")
```

---

### to_onnx / export_onnx

Export a supported model to ONNX format.
Current coverage includes `Dense`, `Conv2d`, `BatchNorm`, `LayerNorm`,
`MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` (output `(1,1)`), `GlobalAvgPool`,
`Flatten`, and common activations in `Sequential`/`Pipeline` models.

```julia
to_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

---

### from_pytorch / load_pytorch

Load a PyTorch model from:
- canonical JSON descriptor (`axiom.pytorch.sequential.v1`)
- direct `.pt/.pth/.ckpt` checkpoints via the built-in Python bridge (`python3` + `torch`)

```julia
model = from_pytorch("model.pt")
model = from_pytorch("model.pytorch.json")
```

---

## Utilities

### parameters

Get all model parameters.

```julia
params = parameters(model)  # Vector of arrays
```

---

### num_parameters

Count parameters.

```julia
n = num_parameters(model)
```

---

### set_training! / training_mode

```julia
set_training!(model, true)   # Training mode
set_training!(model, false)  # Inference mode

# Or functional
model_train = training_mode(model, true)
```

---

### summary

Print model summary.

```julia
summary(model)

# Output:
# ┌─────────────────────────────────────────┐
# │ Axiom Model Summary                     │
# ├─────────────────────────────────────────┤
# │ Layer          │ Output Shape │ Params  │
# ├─────────────────────────────────────────┤
# │ Dense-1        │ (batch, 256) │ 200,960 │
# │ Dense-2        │ (batch, 10)  │ 2,570   │
# ├─────────────────────────────────────────┤
# │ Total params: 203,530                   │
# │ Trainable params: 203,530               │
# └─────────────────────────────────────────┘
```

---

## Data Loading

### DataLoader

Batch data loading with shuffling.

```julia
loader = DataLoader(dataset;
    batch_size=32,
    shuffle=true,
    drop_last=false
)

for (x, y) in loader
    # x: (batch_size, ...), y: labels
end
```

---

*For more examples, see [Tutorials](Tutorials.md)*
