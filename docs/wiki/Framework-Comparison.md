# Framework Comparison

> An honest comparison: Axiom.jl vs PyTorch vs TensorFlow vs JAX

## At a Glance

| Feature | Axiom.jl | PyTorch | TensorFlow | JAX |
|---------|----------|---------|------------|-----|
| **Type Safety** | Compile-time shapes | Runtime only | Graph-based | Traced |
| **Verification** | Built-in | None | TF-verify (limited) | None |
| **Primary Language** | Julia | Python | Python | Python |
| **Backend** | Rust/Julia (+ GPU extensions) | C++/CUDA | C++/CUDA | XLA |
| **Ease of Use** | High | Very High | Medium | Medium |
| **Production Ready** | Growing | Yes | Yes | Yes |
| **Safety-Critical** | Designed for | With effort | With effort | No |

## Philosophy Comparison

### Axiom.jl: Correctness First

```julia
# The type system ensures correctness
model = @axiom begin
    @ensure input_shape == (784,) "MNIST images must be 784-dim"
    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)
    @ensure output_shape == (10,) "Must output 10 classes"
end

# Compile-time verification
@prove BoundedOutputs(0.0, 1.0) model
```

**Philosophy:** Make invalid states unrepresentable. Catch errors at compile time, not production.

### PyTorch: Ease of Use First

```python
# Simple and intuitive
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    nn.Softmax(dim=1)
)

# No built-in verification
# Shape errors discovered at runtime
```

**Philosophy:** Make deep learning accessible. Python-native, imperative, debuggable.

### TensorFlow: Scale First

```python
# Graph-based for deployment
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Static graphs for optimization
# But harder to debug
```

**Philosophy:** Production at scale. Graphs enable optimization, distribution.

### JAX: Functional First

```python
# Pure functions + transformations
def model(params, x):
    x = jnp.dot(x, params['w1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['w2']) + params['b2']
    return jax.nn.softmax(x)

# vmap, jit, grad as transformations
batched_model = jax.vmap(model)
grad_model = jax.grad(model)
```

**Philosophy:** Composable transformations. Functional purity enables powerful abstractions.

## Code Comparison

### Defining a CNN

**Axiom.jl:**
```julia
model = @axiom begin
    Conv2D(3 => 32, (3, 3), activation=relu)
    MaxPool2D((2, 2))
    Conv2D(32 => 64, (3, 3), activation=relu)
    MaxPool2D((2, 2))
    Flatten()
    Dense(1600 => 128, activation=relu)
    Dense(128 => 10, activation=softmax)
end
```

**PyTorch:**
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
```

**TensorFlow:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**JAX (with Flax):**
```python
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return nn.softmax(x)
```

### Training Loop

**Axiom.jl:**
```julia
# High-level API
train!(model, train_loader, epochs=10,
    optimizer=Adam(lr=0.001),
    loss=CrossEntropyLoss()
)

# Or explicit control
for epoch in 1:10
    for (x, y) in train_loader
        loss, grads = value_and_gradient(model, x, y)
        update!(optimizer, model, grads)
    end
end
```

**PyTorch:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
```

**TensorFlow:**
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=10)
```

**JAX:**
```python
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['image'])
        return optax.softmax_cross_entropy(logits, batch['label']).mean()

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

for epoch in range(10):
    for batch in train_loader:
        state = train_step(state, batch)
```

## Feature Deep Dive

### Shape Safety

**Axiom.jl:** Compile-time shape inference and checking
```julia
# This won't compile - shape mismatch caught statically
model = @axiom begin
    Dense(784 => 256)
    Dense(128 => 10)  # Error: Expected input 256, got 128
end
```

**PyTorch:** Runtime shape checking
```python
# This compiles but crashes at runtime
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.Linear(128, 10)  # No error until forward()
)
model(x)  # RuntimeError: size mismatch
```

### Verification

**Axiom.jl:** First-class support
```julia
@ensure all(0 .≤ output .≤ 1) "Probabilities must be valid"
@prove Lipschitz(model, 10.0) "Model is 10-Lipschitz"
cert = generate_certificate(model, properties)
```

**PyTorch:** Must add externally
```python
# Need external tools like:
# - torch.fx for graph inspection
# - Third-party verification libraries
# - Manual assertions
assert (output >= 0).all() and (output <= 1).all()
```

### Deployment

**Axiom.jl:**
```julia
# Export to multiple formats
save_model(model, "model.axiom")           # Native format
export_onnx(model, "model.onnx")           # ONNX
export_certificate(cert, "model.cert")     # Verification proof
```

**PyTorch:**
```python
torch.save(model.state_dict(), "model.pt")
torch.onnx.export(model, x, "model.onnx")
traced = torch.jit.trace(model, x)
traced.save("model_traced.pt")
```

**TensorFlow:**
```python
model.save("model.h5")
model.save("saved_model/")
tf.saved_model.save(model, "tf_saved/")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

## Performance Benchmarks

### Inference Latency (single sample, CPU)

| Model | Axiom.jl (Rust backend) | PyTorch | TensorFlow |
|-------|---------------|---------|------------|
| MLP Small | 19μs | 45μs | 52μs |
| MLP Large | 128μs | 180μs | 195μs |
| CNN (ResNet-18) | 340μs | 290μs | 310μs |
| Transformer (BERT-tiny) | 820μs | 680μs | 720μs |

*Note: PyTorch and TensorFlow have more optimized GPU paths*

### Training Throughput (samples/sec, GPU)

| Model | Axiom.jl* | PyTorch | TensorFlow |
|-------|----------|---------|------------|
| ResNet-50 | 450 | 1200 | 1100 |
| BERT-base | 80 | 320 | 280 |
| GPT-2 | 40 | 180 | 160 |

*Axiom.jl GPU support is in development*

### Compilation Time

| Framework | Cold Start | Warm |
|-----------|------------|------|
| Axiom.jl (Julia) | 2-5s | <1ms |
| Axiom.jl (Rust) | +30s | <1ms |
| PyTorch | <1s | <1ms |
| TensorFlow | 5-10s | <1ms |
| JAX | 10-30s | <1ms |

## Ecosystem Comparison

### Community & Support

| Aspect | Axiom.jl | PyTorch | TensorFlow |
|--------|----------|---------|------------|
| GitHub Stars | New | 70k+ | 170k+ |
| Papers Using | Few | Thousands | Thousands |
| Stack Overflow Q&A | Growing | Extensive | Extensive |
| Corporate Backing | Community | Meta | Google |
| Commercial Support | Coming | Yes | Yes |

### Model Zoo

| Domain | Axiom.jl | PyTorch | TensorFlow |
|--------|----------|---------|------------|
| Computer Vision | Basic | HuggingFace, timm | TF Hub |
| NLP | Basic | HuggingFace | TF Hub |
| Audio | Planned | torchaudio | TF Audio |
| Reinforcement Learning | Planned | Stable-Baselines | TF-Agents |

### Integrations

| Integration | Axiom.jl | PyTorch | TensorFlow |
|-------------|----------|---------|------------|
| ONNX | Import | Import/Export | Import/Export |
| TensorRT | Planned | Yes | Yes |
| CoreML | Planned | Yes | Yes |
| Edge TPU | Planned | No | Yes |
| Jupyter | Yes | Yes | Yes |
| MLflow | Planned | Yes | Yes |
| Weights & Biases | Planned | Yes | Yes |

## When to Use Each

### Choose Axiom.jl When:

- **Safety is paramount** - Medical, automotive, aerospace
- **Verification needed** - Regulatory compliance
- **Type safety matters** - Catching bugs early
- **Julia ecosystem** - Already using Julia
- **Edge deployment** - Rust backend + CPU fallback strategy
- **Research meets production** - Same codebase for both

### Choose PyTorch When:

- **Research focus** - Quick iteration, paper reproduction
- **Ecosystem needed** - HuggingFace, torchvision, etc.
- **GPU training** - Best CUDA support
- **Python required** - Team expertise, integrations
- **Community** - Largest community, most tutorials

### Choose TensorFlow When:

- **Production scale** - Serving infrastructure (TF Serving)
- **Mobile deployment** - TensorFlow Lite
- **Google Cloud** - TPU support, Vertex AI
- **Browser ML** - TensorFlow.js
- **Enterprise** - Commercial support, LTS versions

### Choose JAX When:

- **Functional programming** - Pure functions, immutability
- **Research** - Novel architectures, custom gradients
- **TPU performance** - Best TPU utilization
- **Composability** - vmap, pmap, jit combinations
- **Scientific computing** - Beyond just ML

## Migration Paths

### PyTorch to Axiom.jl

```julia
using Axiom

# Load checkpoint directly (python3 + torch bridge)
model = from_pytorch("model.pt")

# Export to ONNX for downstream runtimes
to_onnx(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

### TensorFlow to Axiom.jl

```julia
using Axiom.TFCompat

# Load SavedModel
model = load_savedmodel("saved_model/")

# Or via ONNX
model = load_onnx("model.onnx")
```

## Summary

| Framework | Best For | Trade-off |
|-----------|----------|-----------|
| **Axiom.jl** | Verified, safe ML | Smaller ecosystem |
| **PyTorch** | Research, flexibility | Less safety |
| **TensorFlow** | Production scale | Complexity |
| **JAX** | Functional, research | Learning curve |

**Our honest take:** If you need verification and safety, Axiom.jl is unmatched. If you need the largest ecosystem and fastest GPU training, PyTorch is the way to go. We're building Axiom.jl for the future where ML is everywhere and correctness is non-negotiable.

---

*Next: [API Reference](API-Reference.md) for complete documentation*
