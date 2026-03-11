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

function Axiom.backend_gpu_conv2d(::Axiom.ROCmBackend, input::AbstractArray{Float32,4},
                                  weight::AbstractArray{Float32,4},
                                  bias::Union{AbstractVector{Float32},Nothing},
                                  stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    input_gpu = ROCArray(input)
    weight_gpu = ROCArray(weight)

    N, H, W, C_in = size(input)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if pH > 0 || pW > 0
        padded = AMDGPU.zeros(Float32, N, H + 2*pH, W + 2*pW, C_in)
        padded[:, pH+1:pH+H, pW+1:pW+W, :] .= input_gpu
        input_gpu = padded
    end

    w_2d = reshape(weight_gpu, :, C_out)
    output_gpu = AMDGPU.zeros(Float32, N, H_out, W_out, C_out)
    for n in 1:N
        col = AMDGPU.zeros(Float32, H_out * W_out, kH * kW * C_in)
        idx = 1
        for i in 1:H_out, j in 1:W_out
            hs = (i-1)*sH+1; ws = (j-1)*sW+1
            col[idx:idx, :] .= reshape(input_gpu[n, hs:hs+kH-1, ws:ws+kW-1, :], 1, :)
            idx += 1
        end
        output_gpu[n, :, :, :] .= reshape(col * w_2d, H_out, W_out, C_out)
    end

    if bias !== nothing
        bias_gpu = ROCArray(bias)
        for oc in 1:C_out
            output_gpu[:, :, :, oc] .+= bias_gpu[oc]
        end
    end
    Array(output_gpu)
end

function Axiom.backend_gpu_batchnorm(::Axiom.ROCmBackend, x::AbstractArray{Float32},
                                     gamma::AbstractVector{Float32}, beta::AbstractVector{Float32},
                                     running_mean::AbstractVector{Float32}, running_var::AbstractVector{Float32},
                                     eps::Float32, training::Bool)
    x_gpu = ROCArray(x)
    γ = ROCArray(reshape(gamma, ones(Int, ndims(x)-1)..., :))
    β = ROCArray(reshape(beta, ones(Int, ndims(x)-1)..., :))
    μ = ROCArray(reshape(running_mean, ones(Int, ndims(x)-1)..., :))
    σ² = ROCArray(reshape(running_var, ones(Int, ndims(x)-1)..., :))
    result = γ .* ((x_gpu .- μ) ./ sqrt.(σ² .+ eps)) .+ β
    Array(result)
end

function Axiom.backend_gpu_maxpool2d(::Axiom.ROCmBackend, input::AbstractArray{Float32,4},
                                     kernel_size::Tuple{Int,Int}, stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    N, H, W, C = size(input)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1; W_out = div(W + 2*pW - kW, sW) + 1

    input_gpu = ROCArray(input)
    if pH > 0 || pW > 0
        padded = AMDGPU.fill(Float32(-Inf), N, H + 2*pH, W + 2*pW, C)
        padded[:, pH+1:pH+H, pW+1:pW+W, :] .= input_gpu
        input_gpu = padded
    end

    output_gpu = AMDGPU.zeros(Float32, N, H_out, W_out, C)
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i-1)*sH+1; ws = (j-1)*sW+1
        output_gpu[n,i,j,c] = maximum(input_gpu[n, hs:hs+kH-1, ws:ws+kW-1, c])
    end
    Array(output_gpu)
end

function Axiom.backend_gpu_avgpool2d(::Axiom.ROCmBackend, input::AbstractArray{Float32,4},
                                     kernel_size::Tuple{Int,Int}, stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    N, H, W, C = size(input)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1; W_out = div(W + 2*pW - kW, sW) + 1

    input_gpu = ROCArray(input)
    if pH > 0 || pW > 0
        padded = AMDGPU.zeros(Float32, N, H + 2*pH, W + 2*pW, C)
        padded[:, pH+1:pH+H, pW+1:pW+W, :] .= input_gpu
        input_gpu = padded
    end

    output_gpu = AMDGPU.zeros(Float32, N, H_out, W_out, C)
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i-1)*sH+1; ws = (j-1)*sW+1
        output_gpu[n,i,j,c] = sum(input_gpu[n, hs:hs+kH-1, ws:ws+kW-1, c]) / (kH * kW)
    end
    Array(output_gpu)
end

# Backend-level dispatch
function Axiom.backend_matmul(::Axiom.ROCmBackend, A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32})
    Array(ROCArray(A) * ROCArray(B))
end

function Axiom.backend_conv2d(::Axiom.ROCmBackend, input::Array{Float32,4}, weight::Array{Float32,4},
                              bias::Union{Vector{Float32},Nothing}, stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    Axiom.backend_gpu_conv2d(Axiom.ROCmBackend(0), input, weight, bias, stride, padding)
end

function Axiom.backend_batchnorm(::Axiom.ROCmBackend, x::Array{Float32}, gamma::Vector{Float32},
                                 beta::Vector{Float32}, running_mean::Vector{Float32},
                                 running_var::Vector{Float32}, eps::Float32, training::Bool)
    Axiom.backend_gpu_batchnorm(Axiom.ROCmBackend(0), x, gamma, beta, running_mean, running_var, eps, training)
end

# ============================================================================
# Memory Management
# ============================================================================

Axiom.backend_to_gpu(::Axiom.ROCmBackend, x::AbstractArray) = ROCArray(x)
Axiom.backend_to_cpu(::Axiom.ROCmBackend, x_gpu::ROCArray) = Array(x_gpu)
Axiom.backend_synchronize(::Axiom.ROCmBackend) = AMDGPU.synchronize()

end  # module AxiomAMDGPUExt
