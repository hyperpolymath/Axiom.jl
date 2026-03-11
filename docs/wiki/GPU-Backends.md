<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# GPU Backend Support

Axiom.jl provides GPU acceleration through package extensions for CUDA (NVIDIA), ROCm (AMD), and Metal (Apple Silicon).

## Quick Start

### Auto-Detection

The simplest way to use GPU acceleration:

```julia
using Axiom

# Auto-detect and use available GPU
if (gpu_backend = detect_gpu()) !== nothing
    set_backend!(gpu_backend)
    @info "Using GPU: $(typeof(gpu_backend))"
else
    @info "No GPU found, using CPU"
end
```

### Manual Selection

Explicitly select a GPU backend:

```julia
using Axiom

# NVIDIA CUDA
set_backend!(CUDABackend(0))  # Device 0

# AMD ROCm
set_backend!(ROCmBackend(0))

# Apple Metal
set_backend!(MetalBackend(0))
```

## Installation

### CUDA (NVIDIA)

```julia
using Pkg
Pkg.add("CUDA")

using CUDA
using Axiom

# Check CUDA is functional
if CUDA.functional()
    set_backend!(CUDABackend(0))
    @info "CUDA ready with $(CUDA.name(CUDA.device()))"
end
```

### ROCm (AMD)

```julia
using Pkg
Pkg.add("AMDGPU")

using AMDGPU
using Axiom

if AMDGPU.functional()
    set_backend!(ROCmBackend(0))
    @info "ROCm ready with $(AMDGPU.device_name())"
end
```

### Metal (Apple Silicon)

```julia
using Pkg
Pkg.add("Metal")

using Metal
using Axiom

if Metal.functional()
    set_backend!(MetalBackend(0))
    @info "Metal ready on Apple Silicon"
end
```

## Usage

### Training on GPU

```julia
using Axiom
using CUDA  # Load GPU extension

# Set backend
set_backend!(CUDABackend(0))

# Define model (same code as CPU)
model = @axiom begin
    Dense(784, 128)
    relu
    Dense(128, 10)
    softmax
end

# Train (automatically uses GPU)
optimizer = Adam(lr=0.001)
loss = CrossEntropyLoss()

train!(model, train_data, optimizer, loss, epochs=10)
```

### Inference on GPU

```julia
using Axiom
using CUDA

# Load pretrained model
model = load_model("my_model.jl")

# Compile for GPU
model_gpu = compile(model, backend=CUDABackend(0), optimize=:aggressive)

# Run inference
predictions = model_gpu(test_input)
```

### Mixed CPU/GPU Workflow

Use different backends for different operations:

```julia
using Axiom
using CUDA

# Data preprocessing on CPU
preprocessed = @with_backend JuliaBackend() begin
    normalize(raw_data)
end

# Model forward pass on GPU
predictions = @with_backend CUDABackend(0) begin
    model(preprocessed)
end

# Postprocessing on CPU
results = @with_backend JuliaBackend() begin
    argmax(predictions, dims=2)
end
```

## Multi-GPU Support

### Data Parallel Training

```julia
using Axiom
using CUDA

# Distribute batch across GPUs
function train_data_parallel!(model, data, gpus=[0, 1, 2, 3])
    batch_size = size(data, 1)
    mini_batch_size = batch_size ÷ length(gpus)

    # Split data across devices
    results = @async for (i, gpu) in enumerate(gpus)
        start_idx = (i-1) * mini_batch_size + 1
        end_idx = i * mini_batch_size
        batch = data[start_idx:end_idx, :]

        @with_backend CUDABackend(gpu) begin
            forward(model, batch)
        end
    end

    # Aggregate gradients
    # ... gradient synchronization ...
end
```

### Model Parallel Training

For models too large for a single GPU:

```julia
# Split model across GPUs
model_part1 = Dense(784, 4096) |> CUDABackend(0)
model_part2 = Dense(4096, 4096) |> CUDABackend(1)
model_part3 = Dense(4096, 10) |> CUDABackend(2)

# Forward pass transfers between GPUs
x_gpu0 = forward(model_part1, x)
x_gpu1 = backend_to_gpu(CUDABackend(1), backend_to_cpu(CUDABackend(0), x_gpu0))
x_gpu1 = forward(model_part2, x_gpu1)
# ... etc ...
```

## Performance Tips

### Memory Management

```julia
# Explicit synchronization for accurate timing
backend_synchronize(current_backend())
@time predictions = model(x)
backend_synchronize(current_backend())

# Transfer to GPU once, reuse
x_gpu = backend_to_gpu(current_backend(), x)
for epoch in 1:100
    # Reuse x_gpu, avoid repeated transfers
    loss = compute_loss(model, x_gpu, y_gpu)
end
```

### Compilation Optimization

```julia
# Aggressive optimization for inference
model_optimized = compile(
    model,
    backend=CUDABackend(0),
    optimize=:aggressive,
    precision=:float16  # Use FP16 for faster inference
)

# Mixed precision for training (FP16 forward, FP32 gradients)
model_mixed = compile(
    model,
    backend=CUDABackend(0),
    precision=:mixed
)
```

## Verification on GPU

GPU operations can be verified for correctness:

```julia
using Axiom
using CUDA

# Enable verification even on GPU
model_verified = compile(
    model,
    backend=CUDABackend(0),
    verify=true  # Checks properties still hold on GPU
)

# Verify specific properties
@prove ∀x ∈ inputs. is_finite(model_verified(x))
```

## Benchmarking

Compare CPU vs GPU performance:

```julia
using BenchmarkTools
using Axiom
using CUDA

model = Dense(1024, 1024)
x = randn(Float32, 1024, 128)

# CPU benchmark
set_backend!(JuliaBackend())
@benchmark forward($model, $x)

# GPU benchmark
set_backend!(CUDABackend(0))
@benchmark begin
    result = forward($model, $x)
    backend_synchronize(current_backend())  # Important for accurate timing
end
```

## Troubleshooting

### CUDA Out of Memory

```julia
# Reduce batch size
batch_size = 32  # Instead of 128

# Use gradient checkpointing (saves memory)
model_checkpointed = enable_checkpointing(model)

# Clear GPU cache
CUDA.reclaim()
```

### Slow First Run

GPU kernels are compiled JIT on first use:

```julia
# Warmup run
model_gpu = compile(model, backend=CUDABackend(0))
_ = model_gpu(randn(Float32, input_size))  # Trigger compilation

# Now benchmark
@benchmark model_gpu($test_input)
```

### Backend Not Available

```julia
# Check what's available
println("CUDA available: ", cuda_available())
println("ROCm available: ", rocm_available())
println("Metal available: ", metal_available())

# Auto-fallback to CPU
backend = detect_gpu()
if backend === nothing
    @warn "No GPU detected, using CPU"
    backend = JuliaBackend()
end
set_backend!(backend)
```

## Implementation Notes

GPU backends are implemented as [package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)):

- `ext/AxiomCUDAExt.jl` - CUDA.jl integration
- `ext/AxiomAMDGPUExt.jl` - AMDGPU.jl integration
- `ext/AxiomMetalExt.jl` - Metal.jl integration

These are loaded automatically when the corresponding GPU package is imported.

## See Also

- [Backend Comparison](Framework-Comparison.md)
- [Performance Tuning](Performance-Tuning.md)
- [Issue #12 - GPU Abstraction Hooks](https://github.com/hyperpolymath/Axiom.jl/issues/12)
