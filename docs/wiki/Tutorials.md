# Tutorials

> From beginner to expert: hands-on guides for every skill level

## Beginner Tutorials

### Tutorial 1: Your First Neural Network

Let's build a simple classifier for the classic MNIST dataset.

```julia
using Axiom
using MLDatasets: MNIST

# Load data
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# Reshape: (28, 28, N) → (N, 784)
train_x = reshape(Float32.(train_x), :, 60000)'
test_x = reshape(Float32.(test_x), :, 10000)'

# Define model
model = @axiom begin
    Dense(784 => 256, activation=relu)
    Dense(256 => 128, activation=relu)
    Dense(128 => 10, activation=softmax)
end

# Train
train!(model, DataLoader((train_x, train_y), batch_size=32),
    epochs=10,
    optimizer=Adam(lr=0.001),
    loss=CrossEntropyLoss()
)

# Evaluate
predictions = forward(model, test_x)
accuracy = mean(argmax(predictions, dims=2) .== test_y)
println("Test accuracy: $(accuracy * 100)%")
```

**What you learned:**
- Loading data
- Defining a model with `@axiom`
- Training with `train!`
- Making predictions

---

### Tutorial 2: Adding Verification

Make your model safer with runtime checks.

```julia
model = @axiom begin
    # Input validation
    @ensure size(x, 2) == 784 "Input must be 784-dimensional"
    @ensure all(0 .≤ x .≤ 1) "Pixels must be normalized [0, 1]"

    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)

    # Output validation
    @ensure size(output, 2) == 10 "Must output 10 classes"
    @ensure all(0 .≤ output .≤ 1) "Output must be probabilities"
end

# Test with invalid input
try
    bad_input = rand(Float32, 1, 100)  # Wrong size!
    forward(model, bad_input)
catch e
    println("Caught error: ", e.msg)
    # "Input must be 784-dimensional"
end
```

**What you learned:**
- Using `@ensure` for runtime validation
- Catching errors before they propagate
- Documenting expectations in code

---

### Tutorial 3: Image Classification with CNNs

Build a convolutional neural network for image classification.

```julia
using Axiom

# Define CNN
model = @axiom begin
    # Input: (batch, 28, 28, 1)
    Conv2D(1 => 32, (3, 3), activation=relu, padding=1)
    MaxPool2D((2, 2))
    # Now: (batch, 14, 14, 32)

    Conv2D(32 => 64, (3, 3), activation=relu, padding=1)
    MaxPool2D((2, 2))
    # Now: (batch, 7, 7, 64)

    Flatten()
    # Now: (batch, 3136)

    Dense(3136 => 128, activation=relu)
    Dropout(0.5)
    Dense(128 => 10, activation=softmax)
end

# Print model summary
summary(model)

# Reshape data for CNN: (N, 28, 28, 1)
train_x_cnn = reshape(train_x', 28, 28, 1, :)
train_x_cnn = permutedims(train_x_cnn, (4, 1, 2, 3))  # NHWC

# Train
train!(model, DataLoader((train_x_cnn, train_y), batch_size=64),
    epochs=5,
    optimizer=Adam(lr=0.001)
)
```

**What you learned:**
- Building CNNs with `Conv2D`
- Pooling layers
- Flattening for fully connected layers
- Dropout regularization

---

## Intermediate Tutorials

### Tutorial 4: Custom Layers

Create your own layer types.

```julia
using Axiom

# Define a custom residual block
struct ResidualBlock <: AbstractLayer
    conv1::Conv2D
    conv2::Conv2D
    shortcut::Union{Conv2D, Nothing}
end

function ResidualBlock(in_channels, out_channels; stride=1)
    conv1 = Conv2D(in_channels => out_channels, (3, 3),
                   stride=stride, padding=1, activation=relu)
    conv2 = Conv2D(out_channels => out_channels, (3, 3),
                   padding=1)

    # Shortcut connection if dimensions change
    shortcut = if in_channels != out_channels || stride != 1
        Conv2D(in_channels => out_channels, (1, 1), stride=stride)
    else
        nothing
    end

    ResidualBlock(conv1, conv2, shortcut)
end

# Define forward pass
function Axiom.forward(block::ResidualBlock, x)
    identity = isnothing(block.shortcut) ? x : forward(block.shortcut, x)
    out = forward(block.conv1, x)
    out = forward(block.conv2, out)
    return relu.(out .+ identity)  # Residual connection!
end

# Get parameters
function Axiom.parameters(block::ResidualBlock)
    params = [parameters(block.conv1)..., parameters(block.conv2)...]
    if !isnothing(block.shortcut)
        append!(params, parameters(block.shortcut))
    end
    return params
end

# Use in a model
model = @axiom begin
    Conv2D(3 => 64, (7, 7), stride=2, padding=3, activation=relu)
    MaxPool2D((3, 3), stride=2, padding=1)
    ResidualBlock(64, 64)
    ResidualBlock(64, 128, stride=2)
    GlobalAvgPool2D()
    Dense(128 => 10, activation=softmax)
end
```

**What you learned:**
- Extending `AbstractLayer`
- Implementing `forward` and `parameters`
- Residual connections

---

### Tutorial 5: Transfer Learning

Use pretrained weights in your model.

```julia
using Axiom

# Load pretrained model (hypothetical)
pretrained = load_model("resnet18_imagenet.axiom")

# Freeze pretrained layers
for layer in pretrained.layers[1:end-1]
    set_trainable!(layer, false)
end

# Replace final layer for our task
model = @axiom begin
    pretrained.layers[1:end-1]...  # All but last layer
    Dense(512 => 100, activation=softmax)  # New classifier
end

# Only train the new layer
train!(model, train_loader,
    epochs=10,
    optimizer=Adam(lr=0.001)
)

# Optional: Fine-tune everything with lower learning rate
for layer in model.layers
    set_trainable!(layer, true)
end

train!(model, train_loader,
    epochs=5,
    optimizer=Adam(lr=0.0001)  # Lower LR for fine-tuning
)
```

**What you learned:**
- Loading pretrained models
- Freezing layers
- Transfer learning workflow

---

### Tutorial 6: Custom Training Loop

Full control over the training process.

```julia
using Axiom

model = @axiom begin
    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)
end

optimizer = Adam(lr=0.001)
loss_fn = CrossEntropyLoss()

# Training metrics
train_losses = Float32[]
val_accuracies = Float32[]

for epoch in 1:50
    epoch_loss = 0.0f0
    n_batches = 0

    for (x, y) in train_loader
        # Forward pass
        predictions = forward(model, x)

        # Compute loss
        loss = loss_fn(predictions, y)
        epoch_loss += loss

        # Backward pass
        grads = gradient(model, x, y, loss_fn)

        # Update weights
        update!(optimizer, model, grads)

        n_batches += 1
    end

    # Record training loss
    push!(train_losses, epoch_loss / n_batches)

    # Validation
    correct = 0
    total = 0
    for (x, y) in val_loader
        pred = forward(model, x, training=false)
        correct += sum(argmax(pred, dims=2) .== y)
        total += length(y)
    end
    val_acc = correct / total
    push!(val_accuracies, val_acc)

    println("Epoch $epoch: Loss=$(train_losses[end]), Val Acc=$(val_acc)")

    # Early stopping
    if length(val_accuracies) > 5
        if all(val_accuracies[end-4:end] .< maximum(val_accuracies[1:end-5]))
            println("Early stopping!")
            break
        end
    end
end
```

**What you learned:**
- Manual training loop
- Computing and recording metrics
- Implementing early stopping

---

## Advanced Tutorials

### Tutorial 7: Formal Verification

Prove properties about your model.

```julia
using Axiom
using Axiom.Verification

# Train a model for safety-critical task
model = @axiom verified=true begin
    Dense(10 => 32, activation=relu)
    Dense(32 => 32, activation=relu)
    Dense(32 => 1, activation=sigmoid)
end

# Define properties to verify
properties = [
    # Output is always in [0, 1]
    BoundedOutputs(0.0f0, 1.0f0),

    # Model is Lipschitz continuous
    LipschitzProperty(10.0f0),

    # Robust to small perturbations
    RobustnessProperty(0.01f0),
]

# Run verification
for prop in properties
    println("Verifying: $prop")
    result = verify(prop, model, input_bounds=(-1.0f0, 1.0f0))
    if result.verified
        println("  ✓ Verified!")
    else
        println("  ✗ Counterexample: $(result.counterexample)")
    end
end

# Generate certificate
cert = generate_certificate(model, properties)
save_certificate(cert, "model_certificate.json")

println("Certificate generated: $(cert.certificate_id)")
```

**What you learned:**
- Defining verification properties
- Running formal verification
- Generating proof certificates

---

### Tutorial 8: Multi-Backend Deployment

Optimize for different deployment targets.

```julia
using Axiom

# Define model architecture once
architecture = quote
    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)
end

# Create versions for different backends

# Julia: Maximum compatibility
model_julia = @axiom backend=JuliaBackend() $architecture

# Rust: Best for parallel workloads
model_rust = @axiom backend=RustBackend() $architecture

# Train once (use fastest available)
best_backend = rust_available() ? RustBackend() : JuliaBackend()
model = @axiom backend=best_backend $architecture
train!(model, train_loader, epochs=10)

# Copy weights to all versions
for target_model in [model_julia, model_rust]
    copy_weights!(target_model, model)
end

# Benchmark
x = rand(Float32, 100, 784)

using BenchmarkTools
println("Julia backend:")
@btime forward($model_julia, $x)

if rust_available()
    println("Rust backend:")
    @btime forward($model_rust, $x)
end
```

**What you learned:**
- Backend selection
- Weight transfer between backends
- Benchmarking inference

---

### Tutorial 9: Building a Transformer

Implement attention mechanisms.

```julia
using Axiom

# Multi-head attention layer
struct MultiHeadAttention <: AbstractLayer
    num_heads::Int
    head_dim::Int
    d_model::Int
    W_q::Dense
    W_k::Dense
    W_v::Dense
    W_o::Dense
end

function MultiHeadAttention(d_model, num_heads)
    @assert d_model % num_heads == 0
    head_dim = d_model ÷ num_heads

    MultiHeadAttention(
        num_heads,
        head_dim,
        d_model,
        Dense(d_model => d_model),
        Dense(d_model => d_model),
        Dense(d_model => d_model),
        Dense(d_model => d_model)
    )
end

function Axiom.forward(mha::MultiHeadAttention, x; mask=nothing)
    batch, seq_len, d_model = size(x)

    # Project Q, K, V
    Q = forward(mha.W_q, x)
    K = forward(mha.W_k, x)
    V = forward(mha.W_v, x)

    # Reshape for multi-head: (batch, seq, d) → (batch, heads, seq, head_dim)
    Q = reshape(Q, batch, seq_len, mha.num_heads, mha.head_dim)
    Q = permutedims(Q, (1, 3, 2, 4))
    K = reshape(K, batch, seq_len, mha.num_heads, mha.head_dim)
    K = permutedims(K, (1, 3, 2, 4))
    V = reshape(V, batch, seq_len, mha.num_heads, mha.head_dim)
    V = permutedims(V, (1, 3, 2, 4))

    # Scaled dot-product attention
    scale = sqrt(Float32(mha.head_dim))
    scores = batched_mul(Q, permutedims(K, (1, 2, 4, 3))) / scale

    # Apply mask if provided
    if !isnothing(mask)
        scores = scores .+ (1 .- mask) .* -1e9
    end

    attn_weights = softmax(scores, dims=4)
    attn_output = batched_mul(attn_weights, V)

    # Reshape back: (batch, heads, seq, head_dim) → (batch, seq, d)
    attn_output = permutedims(attn_output, (1, 3, 2, 4))
    attn_output = reshape(attn_output, batch, seq_len, d_model)

    # Final projection
    return forward(mha.W_o, attn_output)
end

# Transformer encoder layer
struct TransformerEncoderLayer <: AbstractLayer
    self_attn::MultiHeadAttention
    ffn::Sequential
    norm1::LayerNorm
    norm2::LayerNorm
    dropout::Dropout
end

function TransformerEncoderLayer(d_model, num_heads, d_ff; dropout=0.1)
    TransformerEncoderLayer(
        MultiHeadAttention(d_model, num_heads),
        Sequential(
            Dense(d_model => d_ff, activation=gelu),
            Dense(d_ff => d_model)
        ),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Dropout(dropout)
    )
end

function Axiom.forward(layer::TransformerEncoderLayer, x; mask=nothing)
    # Self-attention with residual
    attn_out = forward(layer.self_attn, x, mask=mask)
    x = forward(layer.norm1, x .+ forward(layer.dropout, attn_out))

    # FFN with residual
    ffn_out = forward(layer.ffn, x)
    x = forward(layer.norm2, x .+ forward(layer.dropout, ffn_out))

    return x
end

# Full transformer encoder
model = @axiom begin
    # Embedding (vocabulary -> d_model)
    Embedding(vocab_size, d_model)
    PositionalEncoding(max_seq_len, d_model)

    # Encoder layers
    TransformerEncoderLayer(d_model, 8, 2048)
    TransformerEncoderLayer(d_model, 8, 2048)
    TransformerEncoderLayer(d_model, 8, 2048)

    # Classification head
    GlobalAvgPool1D()
    Dense(d_model => num_classes, activation=softmax)
end
```

**What you learned:**
- Attention mechanism implementation
- Layer composition
- Transformer architecture

---

### Tutorial 10: Distributed Training (Coming Soon)

Train on multiple machines.

```julia
using Axiom
using Axiom.Distributed

# Initialize distributed environment
init_distributed()

world_size = get_world_size()
rank = get_rank()

println("Process $rank of $world_size")

# Partition data
train_loader = distributed_dataloader(dataset,
    batch_size=32,
    shuffle=true
)

# Wrap model for distributed training
model = @axiom begin
    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)
end
model = DistributedDataParallel(model)

# Training (gradients synchronized automatically)
train!(model, train_loader,
    epochs=10,
    optimizer=Adam(lr=0.001 * world_size)  # Scale LR
)

# Only save on rank 0
if rank == 0
    save_model(model, "distributed_model.axiom")
end
```

---

## Project Ideas

### Beginner Projects
1. **Digit Recognizer** - MNIST classifier with verification
2. **Sentiment Analysis** - Simple text classification
3. **Linear Regression** - Verify model is actually linear

### Intermediate Projects
4. **Image Classifier** - CIFAR-10 with CNNs
5. **Text Generator** - Character-level RNN/LSTM
6. **Autoencoder** - Dimensionality reduction

### Advanced Projects
7. **Verified Medical AI** - Classifier with formal guarantees
8. **Autonomous Controller** - RL agent with safety constraints
9. **Custom Hardware** - Deploy with accelerator fallback strategy (TPU/NPU/DSP/FPGA targets)

---

*Need help? Join our [Discord](https://discord.gg/axiom-jl) or open an [issue](https://github.com/your-org/Axiom.jl/issues)!*
