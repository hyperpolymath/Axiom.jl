# SPDX-License-Identifier: PMPL-1.0-or-later
# GPU Backend Hooks for Axiom.jl
#
# This file defines the interface for GPU backends (CUDA, ROCm, Metal).
# Backends are implemented as package extensions loaded conditionally.
#
# Refs: Issue #12 - GPU abstraction hooks

# ROCmBackend struct is defined in abstract.jl alongside CUDABackend/MetalBackend

# ============================================================================
# GPU Detection and Initialization
# ============================================================================

"""
    detect_gpu() -> Union{AbstractBackend, Nothing}

Auto-detect available GPU and return appropriate backend.

Returns `nothing` if no GPU is available.

# Example
```julia
if (backend = detect_gpu()) !== nothing
    set_backend!(backend)
    @info "Using GPU backend: \$(typeof(backend))"
else
    @info "No GPU detected, using CPU"
end
```
"""
function detect_gpu()
    # Try CUDA (NVIDIA)
    if cuda_available()
        device = cuda_device_count() > 0 ? 0 : -1
        if device >= 0
            return CUDABackend(device)
        end
    end

    # Try ROCm (AMD)
    if rocm_available()
        device = rocm_device_count() > 0 ? 0 : -1
        if device >= 0
            return ROCmBackend(device)
        end
    end

    # Try Metal (Apple Silicon)
    if metal_available()
        device = metal_device_count() > 0 ? 0 : -1
        if device >= 0
            return MetalBackend(device)
        end
    end

    # No GPU found
    nothing
end

"""
    rocm_available() -> Bool

Check if ROCm is available.
"""
function rocm_available()
    forced = _backend_env_available("AXIOM_ROCM_AVAILABLE")
    forced !== nothing && return forced
    false
end

"""
    cuda_device_count() -> Int

Get number of available CUDA devices.
"""
function cuda_device_count()
    forced = _backend_env_count("AXIOM_CUDA_AVAILABLE", "AXIOM_CUDA_DEVICE_COUNT")
    forced !== nothing && return forced
    0
end

"""
    rocm_device_count() -> Int

Get number of available ROCm devices.
"""
function rocm_device_count()
    forced = _backend_env_count("AXIOM_ROCM_AVAILABLE", "AXIOM_ROCM_DEVICE_COUNT")
    forced !== nothing && return forced
    0
end

"""
    metal_device_count() -> Int

Get number of available Metal devices.
"""
function metal_device_count()
    forced = _backend_env_count("AXIOM_METAL_AVAILABLE", "AXIOM_METAL_DEVICE_COUNT")
    forced !== nothing && return forced
    metal_available() ? 1 : 0
end

function _gpu_hook_overrides(backend_type::DataType)
    hooks = Dict{String, Bool}()
    for (name, hook) in (
        ("backend_gpu_matmul", backend_gpu_matmul),
        ("backend_gpu_conv2d", backend_gpu_conv2d),
        ("backend_gpu_relu", backend_gpu_relu),
        ("backend_gpu_softmax", backend_gpu_softmax),
        ("backend_gpu_batchnorm", backend_gpu_batchnorm),
    )
        hooks[name] = _hook_has_backend_override(hook, backend_type)
    end
    hooks
end

"""
    gpu_capability_report() -> Dict{String,Any}

Return machine-readable GPU capability status, including hook coverage and
runtime self-healing diagnostics.
"""
function gpu_capability_report()
    selected = detect_gpu()
    backends = Dict{String, Any}()
    strategy = [
        ("CUDA", CUDABackend, cuda_available, cuda_device_count),
        ("ROCm", ROCmBackend, rocm_available, rocm_device_count),
        ("Metal", MetalBackend, metal_available, metal_device_count),
    ]

    for (label, backend_type, available_fn, count_fn) in strategy
        available = available_fn()
        count = count_fn()
        hooks = _gpu_hook_overrides(backend_type)
        kernel_hooks_loaded = !isempty(hooks) && all(values(hooks))
        backends[label] = Dict(
            "available" => available,
            "device_count" => count,
            "compilable" => available && count > 0,
            "kernel_hooks_loaded" => kernel_hooks_loaded,
            "hook_overrides" => hooks,
        )
    end

    Dict(
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "strategy_order" => ["CUDA", "ROCm", "Metal"],
        "selected_backend" => selected === nothing ? nothing : string(typeof(selected)),
        "self_healing_enabled" => _gpu_self_healing_enabled(),
        "runtime_diagnostics" => gpu_runtime_diagnostics(),
        "backends" => backends,
    )
end

"""
    select_device!(backend::CUDABackend, device::Int)

Select specific CUDA device.
"""
function select_device!(backend::CUDABackend, device::Int)
    # In full implementation: CUDA.device!(device)
    @warn "CUDA device selection requires CUDA.jl extension"
    CUDABackend(device)
end

"""
    select_device!(backend::ROCmBackend, device::Int)

Select specific ROCm device.
"""
function select_device!(backend::ROCmBackend, device::Int)
    # In full implementation: AMDGPU.device!(device)
    @warn "ROCm device selection requires AMDGPU.jl extension"
    ROCmBackend(device)
end

"""
    select_device!(backend::MetalBackend, device::Int)

Metal device selection (no-op on macOS - OS manages).
"""
function select_device!(backend::MetalBackend, device::Int)
    @debug "Metal device selection managed by macOS"
    MetalBackend(device)
end

# ============================================================================
# GPU Backend Operations Interface
# ============================================================================
#
# These functions define the expected interface for GPU backends.
# Implementations are provided via package extensions (ext/):
#   - ext/AxiomCUDAExt.jl - CUDA.jl extension
#   - ext/AxiomAMDGPUExt.jl - AMDGPU.jl extension
#   - ext/AxiomMetalExt.jl - Metal.jl extension

"""
    backend_gpu_matmul(backend::Union{CUDABackend,ROCmBackend,MetalBackend}, A, B)

Matrix multiplication on GPU.

# Implementation Requirements
- Transfer data to GPU if not already there
- Execute BLAS gemm on device
- Return GPU array (conversion to CPU handled by caller if needed)
"""
function backend_gpu_matmul end

"""
    backend_gpu_conv2d(backend, input, weight, bias, stride, padding)

2D convolution on GPU.

# Implementation Requirements
- Use cuDNN (CUDA), MIOpen (ROCm), or MPS (Metal)
- Support common padding modes: :same, :valid
- Handle bias addition if provided
"""
function backend_gpu_conv2d end

"""
    backend_gpu_relu(backend, x)

ReLU activation on GPU.

# Implementation Requirements
- Element-wise max(0, x)
- In-place or out-of-place based on backend capabilities
"""
function backend_gpu_relu end

"""
    backend_gpu_softmax(backend, x, dim)

Softmax on GPU.

# Implementation Requirements
- Numerically stable implementation (subtract max)
- Along specified dimension
"""
function backend_gpu_softmax end

"""
    backend_gpu_batchnorm(backend, x, gamma, beta, mean, var, eps)

Batch normalization on GPU.

# Implementation Requirements
- Use fused kernel if available (cuDNN/MIOpen/MPS)
- Handle training vs inference mode
"""
function backend_gpu_batchnorm end

"""
    backend_to_gpu(backend, x::AbstractArray)

Transfer array to GPU.

# Implementation Requirements
- CUDA: convert to CuArray
- ROCm: convert to ROCArray
- Metal: convert to MtlArray
- No-op if already on device
"""
function backend_to_gpu end

"""
    backend_to_cpu(backend, x_gpu)

Transfer array from GPU to CPU.

# Implementation Requirements
- Convert GPU array to standard Julia Array
- No-op if already on CPU
"""
function backend_to_cpu end

"""
    backend_synchronize(backend)

Synchronize GPU operations.

# Implementation Requirements
- Block until all GPU work completes
- CUDA: CUDA.synchronize()
- ROCm: AMDGPU.synchronize()
- Metal: Metal.synchronize()
"""
function backend_synchronize end

# ============================================================================
# Fallback Implementations (CPU)
# ============================================================================

# Default fallbacks that execute on CPU with a warning
for op in [:backend_gpu_matmul, :backend_gpu_conv2d, :backend_gpu_relu,
           :backend_gpu_softmax, :backend_gpu_batchnorm]
    @eval function $op(backend::Union{CUDABackend,ROCmBackend,MetalBackend}, args...)
        @warn "GPU extension not loaded for $(typeof(backend)), falling back to CPU" maxlog=1
        backend_name = typeof(backend)
        julia_backend = JuliaBackend()
        # Call the corresponding CPU operation
        cpu_op = Symbol(replace(string($op), "gpu_" => ""))
        if isdefined(@__MODULE__, cpu_op)
            return getfield(@__MODULE__, cpu_op)(julia_backend, args...)
        else
            error("No fallback implementation for $op")
        end
    end
end

backend_to_gpu(::Union{CUDABackend,ROCmBackend,MetalBackend}, x::AbstractArray) = x
backend_to_cpu(::Union{CUDABackend,ROCmBackend,MetalBackend}, x) = x
backend_synchronize(::Union{CUDABackend,ROCmBackend,MetalBackend}) = nothing

# ============================================================================
# ROCm Compiled Model
# ============================================================================

"""
    ROCmCompiledModel

Model wrapper that dispatches operations to ROCm backend (AMD GPUs).
"""
struct ROCmCompiledModel{M}
    model::M
    backend::ROCmBackend
end

function forward(rm::ROCmCompiledModel, x)
    @debug "ROCmCompiledModel requires tensor input for accelerated path; falling back to model forward"
    forward(rm.model, x)
end

function forward(rm::ROCmCompiledModel, x::AbstractTensor)
    _gpu_forward_with_recovery(rm.model, x, rm.backend)
end

(rm::ROCmCompiledModel)(x) = forward(rm, x)

parameters(rm::ROCmCompiledModel) = parameters(rm.model)
output_shape(rm::ROCmCompiledModel, input_shape) = output_shape(rm.model, input_shape)

function compile_to_backend(model, backend::ROCmBackend)
    @info "Compiling to ROCm backend on device $(backend.device)..."

    # Check ROCm availability
    if !rocm_available()
        _gpu_record_diagnostic!(backend, "compile_fallbacks")
        @warn "ROCm not available, falling back to Julia backend"
        return model
    end
    device_count = rocm_device_count()
    if backend.device < 0 || backend.device >= device_count
        _gpu_record_diagnostic!(backend, "compile_fallbacks")
        @warn "ROCm device $(backend.device) out of range (available: 0:$(max(0, device_count - 1))), falling back to Julia backend"
        return model
    end

    # Wrap model for ROCm execution
    ROCmCompiledModel(model, backend)
end

# ============================================================================
# Extension Loading Documentation
# ============================================================================

"""
# GPU Backend Extensions

Axiom.jl supports GPU acceleration through package extensions.

## CUDA (NVIDIA GPUs)

```julia
using CUDA
using Axiom

# Auto-detect and use GPU
backend = detect_gpu()
set_backend!(backend)

# Or explicitly select CUDA
set_backend!(CUDABackend(0))  # Device 0

# Compile model for GPU
model_gpu = compile(model, backend=CUDABackend(0))
result = model_gpu(input)
```

## ROCm (AMD GPUs)

```julia
using AMDGPU
using Axiom

set_backend!(ROCmBackend(0))
model_gpu = compile(model, backend=ROCmBackend(0))
```

## Metal (Apple Silicon)

```julia
using Metal
using Axiom

# Metal available on M1/M2/M3 Macs
if metal_available()
    set_backend!(MetalBackend(0))
    model_gpu = compile(model, backend=MetalBackend(0))
end
```

## Implementation Checklist for Extension Developers

Each GPU backend extension must provide:

1. **Backend operations:**
   - `backend_gpu_matmul`
   - `backend_gpu_conv2d`
   - `backend_gpu_relu`
   - `backend_gpu_softmax`
   - `backend_gpu_batchnorm`

2. **Memory management:**
   - `backend_to_gpu` - Transfer to GPU memory
   - `backend_to_cpu` - Transfer to CPU memory
   - `backend_synchronize` - Synchronize operations

3. **Device management:**
   - Override `cuda_available()` / `rocm_available()` / `metal_available()`
   - Implement `select_device!` if needed

4. **Testing:**
   - Numerical correctness vs CPU reference
   - Memory leak checks
   - Performance benchmarks

## Extension Template

See `ext/AxiomCUDAExt.jl` for reference implementation.

```julia
# ext/AxiomCUDAExt.jl
module AxiomCUDAExt

using CUDA
using Axiom

# Override availability check
Axiom.cuda_available() = CUDA.functional()
Axiom.cuda_device_count() = CUDA.ndevices()

# Implement GPU operations
function Axiom.backend_gpu_matmul(::Axiom.CUDABackend, A, B)
    A_gpu = CuArray(A)
    B_gpu = CuArray(B)
    result = A_gpu * B_gpu
    Array(result)  # Return CPU array
end

# ... implement other operations ...

end  # module
```
"""
const GPU_BACKEND_DOCS = nothing
