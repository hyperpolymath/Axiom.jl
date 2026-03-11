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

function Axiom.backend_gpu_conv2d(::Axiom.CUDABackend, input::AbstractArray{Float32,4},
                                  weight::AbstractArray{Float32,4},
                                  bias::Union{AbstractVector{Float32},Nothing},
                                  stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    # GPU-accelerated conv2d via im2col + cuBLAS GEMM
    input_gpu = CuArray(input)
    weight_gpu = CuArray(weight)

    N, H, W, C_in = size(input)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    # Pad on GPU
    if pH > 0 || pW > 0
        padded = CUDA.zeros(Float32, N, H + 2*pH, W + 2*pW, C_in)
        padded[:, pH+1:pH+H, pW+1:pW+W, :] .= input_gpu
        input_gpu = padded
    end

    # Reshape weight to 2D: (kH*kW*C_in, C_out)
    w_2d = reshape(weight_gpu, :, C_out)

    # im2col and GEMM per batch
    output_gpu = CUDA.zeros(Float32, N, H_out, W_out, C_out)
    for n in 1:N
        col = CUDA.zeros(Float32, H_out * W_out, kH * kW * C_in)
        idx = 1
        for i in 1:H_out, j in 1:W_out
            hs = (i-1)*sH+1; ws = (j-1)*sW+1
            patch = reshape(input_gpu[n, hs:hs+kH-1, ws:ws+kW-1, :], 1, :)
            col[idx:idx, :] .= patch
            idx += 1
        end
        out_2d = col * w_2d  # cuBLAS GEMM
        output_gpu[n, :, :, :] .= reshape(out_2d, H_out, W_out, C_out)
    end

    if bias !== nothing
        bias_gpu = CuArray(bias)
        for oc in 1:C_out
            output_gpu[:, :, :, oc] .+= bias_gpu[oc]
        end
    end

    Array(output_gpu)
end

function Axiom.backend_gpu_batchnorm(::Axiom.CUDABackend, x::AbstractArray{Float32},
                                     gamma::AbstractVector{Float32}, beta::AbstractVector{Float32},
                                     running_mean::AbstractVector{Float32}, running_var::AbstractVector{Float32},
                                     eps::Float32, training::Bool)
    x_gpu = CuArray(x)
    γ_gpu = CuArray(reshape(gamma, ones(Int, ndims(x)-1)..., :))
    β_gpu = CuArray(reshape(beta, ones(Int, ndims(x)-1)..., :))
    μ_gpu = CuArray(reshape(running_mean, ones(Int, ndims(x)-1)..., :))
    σ²_gpu = CuArray(reshape(running_var, ones(Int, ndims(x)-1)..., :))

    x_norm = (x_gpu .- μ_gpu) ./ sqrt.(σ²_gpu .+ eps)
    result = γ_gpu .* x_norm .+ β_gpu
    Array(result)
end

function Axiom.backend_gpu_maxpool2d(::Axiom.CUDABackend, input::AbstractArray{Float32,4},
                                     kernel_size::Tuple{Int,Int}, stride::Tuple{Int,Int},
                                     padding::Tuple{Int,Int})
    input_gpu = CuArray(input)
    N, H, W, C = size(input)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if pH > 0 || pW > 0
        padded = CUDA.fill(Float32(-Inf), N, H + 2*pH, W + 2*pW, C)
        padded[:, pH+1:pH+H, pW+1:pW+W, :] .= input_gpu
        input_gpu = padded
    end

    output_gpu = CUDA.zeros(Float32, N, H_out, W_out, C)
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i-1)*sH+1; ws = (j-1)*sW+1
        output_gpu[n,i,j,c] = maximum(input_gpu[n, hs:hs+kH-1, ws:ws+kW-1, c])
    end
    Array(output_gpu)
end

function Axiom.backend_gpu_avgpool2d(::Axiom.CUDABackend, input::AbstractArray{Float32,4},
                                     kernel_size::Tuple{Int,Int}, stride::Tuple{Int,Int},
                                     padding::Tuple{Int,Int})
    input_gpu = CuArray(input)
    N, H, W, C = size(input)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if pH > 0 || pW > 0
        padded = CUDA.zeros(Float32, N, H + 2*pH, W + 2*pW, C)
        padded[:, pH+1:pH+H, pW+1:pW+W, :] .= input_gpu
        input_gpu = padded
    end

    output_gpu = CUDA.zeros(Float32, N, H_out, W_out, C)
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i-1)*sH+1; ws = (j-1)*sW+1
        output_gpu[n,i,j,c] = sum(input_gpu[n, hs:hs+kH-1, ws:ws+kW-1, c]) / (kH * kW)
    end
    Array(output_gpu)
end

# ============================================================================
# Backend-Level Dispatch (for abstract.jl forward override)
# ============================================================================

function Axiom.backend_matmul(::Axiom.CUDABackend, A::AbstractMatrix{Float32}, B::AbstractMatrix{Float32})
    A_gpu = CuArray(A); B_gpu = CuArray(B)
    Array(A_gpu * B_gpu)
end

function Axiom.backend_conv2d(::Axiom.CUDABackend, input::Array{Float32,4}, weight::Array{Float32,4},
                              bias::Union{Vector{Float32},Nothing}, stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    Axiom.backend_gpu_conv2d(Axiom.CUDABackend(0), input, weight, bias, stride, padding)
end

function Axiom.backend_batchnorm(::Axiom.CUDABackend, x::Array{Float32}, gamma::Vector{Float32},
                                 beta::Vector{Float32}, running_mean::Vector{Float32},
                                 running_var::Vector{Float32}, eps::Float32, training::Bool)
    Axiom.backend_gpu_batchnorm(Axiom.CUDABackend(0), x, gamma, beta, running_mean, running_var, eps, training)
end

# ============================================================================
# Memory Management
# ============================================================================

Axiom.backend_to_gpu(::Axiom.CUDABackend, x::AbstractArray) = CuArray(x)
Axiom.backend_to_cpu(::Axiom.CUDABackend, x_gpu::CuArray) = Array(x_gpu)
Axiom.backend_synchronize(::Axiom.CUDABackend) = CUDA.synchronize()

end  # module AxiomCUDAExt
