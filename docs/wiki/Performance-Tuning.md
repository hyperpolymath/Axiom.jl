# Performance Tuning

> Maximize throughput, minimize latency, achieve production-grade performance

## Performance Philosophy

Axiom.jl is designed for **predictable performance**. Unlike frameworks that rely on JIT compilation with unpredictable warm-up, Axiom.jl provides:

1. **Consistent latency** - No compilation pauses during inference
2. **Memory efficiency** - Explicit control over allocations
3. **Backend choice** - Pick the right tool for your hardware

## Quick Wins

### 1. Choose the Right Backend

```julia
# Check what's available
using Axiom

println("Zig available: ", zig_available())
println("Rust available: ", rust_available())

# Explicit backend selection
model = @axiom backend=ZigBackend() begin
    Dense(784 => 256, activation=relu)
    Dense(256 => 10, activation=softmax)
end
```

**Backend Selection Guide:**

| Scenario | Recommended Backend |
|----------|---------------------|
| Quick prototyping | JuliaBackend |
| Production inference | ZigBackend |
| Parallel training | RustBackend |
| Maximum compatibility | JuliaBackend |
| Embedded/Edge | ZigBackend |

### 2. Use Float32 (Not Float64)

```julia
# Slow - Float64 is default in Julia
x = rand(784)  # Float64

# Fast - explicit Float32
x = rand(Float32, 784)

# Even faster - use our typed tensors
x = Tensor{Float32, (784,)}(rand(Float32, 784))
```

### 3. Batch Your Inputs

```julia
# Slow - one at a time
for sample in dataset
    predict(model, sample)
end

# Fast - batched inference
batch = stack(dataset[1:32])
predict(model, batch)  # 10-50× faster
```

## Memory Optimization

### Pre-allocate Buffers

```julia
# Create reusable buffers
struct InferenceContext
    input_buffer::Array{Float32, 2}
    hidden_buffer::Array{Float32, 2}
    output_buffer::Array{Float32, 2}
end

function create_context(batch_size, model)
    InferenceContext(
        zeros(Float32, batch_size, 784),
        zeros(Float32, batch_size, 256),
        zeros(Float32, batch_size, 10)
    )
end

# Reuse during inference
function predict!(ctx::InferenceContext, model, x)
    copyto!(ctx.input_buffer, x)
    forward!(ctx.hidden_buffer, model.layers[1], ctx.input_buffer)
    forward!(ctx.output_buffer, model.layers[2], ctx.hidden_buffer)
    return ctx.output_buffer
end
```

### Avoid Intermediate Allocations

```julia
# Bad - allocates intermediate arrays
function forward_slow(model, x)
    h1 = x * model.W1        # Allocation 1
    h2 = relu.(h1)           # Allocation 2
    h3 = h2 * model.W2       # Allocation 3
    return softmax(h3)       # Allocation 4
end

# Good - fused operations
function forward_fast!(y, model, x, buffer)
    mul!(buffer, x, model.W1)      # In-place
    relu!(buffer)                   # In-place
    mul!(y, buffer, model.W2)      # In-place
    softmax!(y)                     # In-place
end
```

### Memory Layout Matters

```julia
# NHWC (batch, height, width, channels) is optimal for:
# - CPU inference
# - Depthwise convolutions
# - Memory coalescing

# NCHW is optimal for:
# - GPU inference (cuDNN)
# - Standard convolutions

# Axiom.jl uses NHWC by default
x = zeros(Float32, 32, 224, 224, 3)  # batch=32, 224×224, RGB
```

## Computation Optimization

### Matrix Multiplication Tuning

The backends use tiled algorithms optimized for cache:

```julia
# Tile size affects cache utilization
# Default: 64×64 tiles fit in L1 cache (16KB)

# For larger L1 caches (32KB+), you can increase:
# (Requires recompiling Zig/Rust backend)
const TILE_SIZE = 128
```

### SIMD Utilization

The Zig backend uses explicit SIMD:

```zig
// 8-wide SIMD vectors (AVX)
const Vec = @Vector(8, f32);

// Aligned memory access is faster
// Ensure arrays are 32-byte aligned
```

**Check SIMD availability:**
```julia
# CPU features detection
using CpuId
println("AVX: ", cpufeature(:AVX))
println("AVX2: ", cpufeature(:AVX2))
println("AVX512: ", cpufeature(:AVX512F))
```

### Activation Function Speed

From fastest to slowest:

1. **ReLU** - Single comparison, branchless SIMD
2. **ReLU6** - Two comparisons
3. **Leaky ReLU** - One comparison + multiply
4. **Swish/SiLU** - Sigmoid + multiply
5. **GELU** - tanh approximation
6. **Softmax** - exp + reduction

```julia
# Use ReLU when possible
@axiom function fast_net(x)
    Dense(784 => 256, activation=relu) |>   # Fast
    Dense(256 => 10, activation=softmax)    # Necessary for classification
end

# GELU only when accuracy matters
@axiom function accurate_net(x)
    Dense(784 => 256, activation=gelu) |>   # Slower but better gradients
    Dense(256 => 10, activation=softmax)
end
```

## Profiling

### Built-in Timing

```julia
using Axiom: @timed_forward

# Time individual layers
@timed_forward model x

# Output:
# Layer 1 (Dense): 0.23ms
# Layer 2 (ReLU):  0.01ms
# Layer 3 (Dense): 0.18ms
# Total: 0.42ms
```

### Memory Profiling

```julia
using Axiom: @memory_profile

@memory_profile begin
    y = forward(model, x)
end

# Output:
# Allocations: 3
# Total memory: 128 KB
# Peak memory: 64 KB
```

### Benchmark Comparison

```julia
using BenchmarkTools

x = rand(Float32, 32, 784)

# Compare backends
@btime forward($model_julia, $x)   # Julia backend
@btime forward($model_rust, $x)    # Rust backend
@btime forward($model_zig, $x)     # Zig backend
```

## Parallelization

### Batch Parallelism

```julia
# Rust backend automatically parallelizes over batches
model = @axiom backend=RustBackend() begin
    Dense(784 => 256)
end

# Each batch element processed in parallel
y = forward(model, rand(Float32, 1000, 784))
```

### Data Parallelism

```julia
using Base.Threads

# Parallel inference over multiple inputs
results = Vector{Array{Float32}}(undef, length(inputs))

@threads for i in eachindex(inputs)
    results[i] = forward(model, inputs[i])
end
```

### Model Parallelism

For very large models:

```julia
# Split model across computation
model_part1 = @axiom begin
    Dense(784 => 4096)
    Dense(4096 => 4096)
end

model_part2 = @axiom begin
    Dense(4096 => 4096)
    Dense(4096 => 10)
end

function forward_parallel(x)
    h = forward(model_part1, x)  # Can run on device 1
    return forward(model_part2, h)  # Can run on device 2
end
```

## Inference Optimization

### Static Shape Compilation

When shapes are known at compile time:

```julia
# Dynamic shapes - slower
function forward_dynamic(model, x)
    # Shape checked at runtime
end

# Static shapes - faster
function forward_static(model, x::Tensor{Float32, (32, 784)})
    # Shape known at compile time - optimizations possible
end
```

### Warm-up Elimination

```julia
# Force compilation before timing
function warmup!(model, example_input)
    # Run once to trigger compilation
    _ = forward(model, example_input)

    # Clear any lazy allocations
    GC.gc()
end

warmup!(model, x)
# Now all subsequent calls have consistent latency
```

### Quantization (Coming Soon)

```julia
# INT8 quantization for faster inference
model_int8 = quantize(model, Int8)

# 4× faster, 4× smaller
y = forward(model_int8, x)
```

## Common Bottlenecks

### 1. Memory Bandwidth

**Symptom:** Large tensors, simple operations slow
**Solution:**
- Use smaller batch sizes
- Fuse operations
- Use in-place operations

### 2. Cache Misses

**Symptom:** Large matrices slow, small ones fast
**Solution:**
- Use tiled algorithms (default in backends)
- Ensure data locality
- Prefer NHWC layout

### 3. Branch Misprediction

**Symptom:** Operations with conditionals slow
**Solution:**
- Use branchless operations (ReLU instead of if/else)
- Sort data to improve prediction

### 4. Garbage Collection

**Symptom:** Periodic slowdowns
**Solution:**
- Pre-allocate buffers
- Use in-place operations
- Disable GC during critical sections

```julia
GC.enable(false)
try
    # Critical inference path
    y = forward(model, x)
finally
    GC.enable(true)
end
```

## Production Checklist

- [ ] Using Float32 (not Float64)
- [ ] Batched inference enabled
- [ ] Appropriate backend selected
- [ ] Buffers pre-allocated
- [ ] Warm-up completed before timing
- [ ] GC pauses minimized
- [ ] Memory layout matches access pattern
- [ ] Profiling shows no bottlenecks
- [ ] Latency meets requirements
- [ ] Throughput meets requirements

## Benchmark Reference

Benchmarks on Intel Core i9-12900K (single-threaded):

| Model | Batch | Julia | Rust | Zig |
|-------|-------|-------|------|-----|
| MLP (784→256→10) | 1 | 45μs | 22μs | 19μs |
| MLP (784→256→10) | 32 | 320μs | 145μs | 128μs |
| ResNet-18 block | 1 | 890μs | 380μs | 340μs |
| Transformer layer | 1 | 2.1ms | 950μs | 820μs |
| LayerNorm (768) | 32 | 28μs | 18μs | 15μs |
| Softmax (10000) | 32 | 85μs | 62μs | 58μs |

---

*Next: [Safety-Critical Applications](Safety-Critical.md) for deploying in regulated environments*
