# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl AMDGPU Extension
#
# Provides GPU acceleration via AMD ROCm.
# Automatically loaded when AMDGPU.jl is imported.

module AxiomAMDGPUExt

using AMDGPU
using Axiom

# ============================================================================
# Availability Detection
# ============================================================================

function Axiom.rocm_available()
    forced = Axiom._backend_env_available("AXIOM_ROCM_AVAILABLE")
    forced !== nothing && return forced
    AMDGPU.functional()
end

function Axiom.rocm_device_count()
    forced = Axiom._backend_env_count("AXIOM_ROCM_AVAILABLE", "AXIOM_ROCM_DEVICE_COUNT")
    forced !== nothing && return forced
    AMDGPU.ndevices()
end

# ============================================================================
# GPU Operations
# ============================================================================

function Axiom.backend_gpu_matmul(::Axiom.ROCmBackend, A::AbstractMatrix, B::AbstractMatrix)
    A_gpu = ROCArray(A)
    B_gpu = ROCArray(B)
    result_gpu = A_gpu * B_gpu  # Uses rocBLAS gemm
    Array(result_gpu)
end

function Axiom.backend_gpu_relu(::Axiom.ROCmBackend, x::AbstractArray)
    x_gpu = ROCArray(x)
    result_gpu = max.(x_gpu, 0)
    Array(result_gpu)
end

function Axiom.backend_gpu_softmax(::Axiom.ROCmBackend, x::AbstractArray, dim::Int)
    x_gpu = ROCArray(x)
    x_max = maximum(x_gpu, dims=dim)
    x_exp = exp.(x_gpu .- x_max)
    result_gpu = x_exp ./ sum(x_exp, dims=dim)
    Array(result_gpu)
end

# ============================================================================
# Memory Management
# ============================================================================

Axiom.backend_to_gpu(::Axiom.ROCmBackend, x::AbstractArray) = ROCArray(x)
Axiom.backend_to_cpu(::Axiom.ROCmBackend, x_gpu::ROCArray) = Array(x_gpu)
Axiom.backend_synchronize(::Axiom.ROCmBackend) = AMDGPU.synchronize()

end  # module AxiomAMDGPUExt
