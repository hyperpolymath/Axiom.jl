# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Metal Extension
#
# Provides GPU acceleration via Apple Metal (M1/M2/M3).
# Automatically loaded when Metal.jl is imported.

module AxiomMetalExt

using Metal
using Axiom

# ============================================================================
# Availability Detection
# ============================================================================

function Axiom.metal_available()
    forced = Axiom._backend_env_available("AXIOM_METAL_AVAILABLE")
    forced !== nothing && return forced
    Metal.functional()
end

# ============================================================================
# GPU Operations
# ============================================================================

function Axiom.backend_gpu_matmul(::Axiom.MetalBackend, A::AbstractMatrix, B::AbstractMatrix)
    A_gpu = MtlArray(A)
    B_gpu = MtlArray(B)
    result_gpu = A_gpu * B_gpu  # Uses Metal Performance Shaders
    Array(result_gpu)
end

function Axiom.backend_gpu_relu(::Axiom.MetalBackend, x::AbstractArray)
    x_gpu = MtlArray(x)
    result_gpu = max.(x_gpu, 0)
    Array(result_gpu)
end

function Axiom.backend_gpu_softmax(::Axiom.MetalBackend, x::AbstractArray, dim::Int)
    x_gpu = MtlArray(x)
    x_max = maximum(x_gpu, dims=dim)
    x_exp = exp.(x_gpu .- x_max)
    result_gpu = x_exp ./ sum(x_exp, dims=dim)
    Array(result_gpu)
end

# ============================================================================
# Memory Management
# ============================================================================

Axiom.backend_to_gpu(::Axiom.MetalBackend, x::AbstractArray) = MtlArray(x)
Axiom.backend_to_cpu(::Axiom.MetalBackend, x_gpu::MtlArray) = Array(x_gpu)
Axiom.backend_synchronize(::Axiom.MetalBackend) = Metal.synchronize()

end  # module AxiomMetalExt
