# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl CUDA Extension
#
# Provides GPU acceleration via NVIDIA CUDA.
# Automatically loaded when CUDA.jl is imported.

module AxiomCUDAExt

using CUDA
using Axiom

# ============================================================================
# Availability Detection
# ============================================================================

function Axiom.cuda_available()
    forced = Axiom._backend_env_available("AXIOM_CUDA_AVAILABLE")
    forced !== nothing && return forced
    CUDA.functional()
end

function Axiom.cuda_device_count()
    forced = Axiom._backend_env_count("AXIOM_CUDA_AVAILABLE", "AXIOM_CUDA_DEVICE_COUNT")
    forced !== nothing && return forced
    CUDA.ndevices()
end

# ============================================================================
# GPU Operations
# ============================================================================

function Axiom.backend_gpu_matmul(::Axiom.CUDABackend, A::AbstractMatrix, B::AbstractMatrix)
    A_gpu = CuArray(A)
    B_gpu = CuArray(B)
    result_gpu = A_gpu * B_gpu  # Uses cuBLAS gemm
    Array(result_gpu)  # Transfer back to CPU
end

function Axiom.backend_gpu_relu(::Axiom.CUDABackend, x::AbstractArray)
    x_gpu = CuArray(x)
    result_gpu = max.(x_gpu, 0)
    Array(result_gpu)
end

function Axiom.backend_gpu_softmax(::Axiom.CUDABackend, x::AbstractArray, dim::Int)
    x_gpu = CuArray(x)
    # Numerically stable softmax
    x_max = maximum(x_gpu, dims=dim)
    x_exp = exp.(x_gpu .- x_max)
    result_gpu = x_exp ./ sum(x_exp, dims=dim)
    Array(result_gpu)
end

# ============================================================================
# Memory Management
# ============================================================================

Axiom.backend_to_gpu(::Axiom.CUDABackend, x::AbstractArray) = CuArray(x)
Axiom.backend_to_cpu(::Axiom.CUDABackend, x_gpu::CuArray) = Array(x_gpu)
Axiom.backend_synchronize(::Axiom.CUDABackend) = CUDA.synchronize()

end  # module AxiomCUDAExt
