# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl FPGA Extension
#
# Provides pipeline-oriented tensor operations for FPGA accelerators.
# FPGAs excel at custom data-path pipelines: every operation is modelled as a
# streaming pipeline stage with explicit data-flow between stages.
#
# Key optimisations:
#   - Streaming matrix multiply pipeline (row-by-row accumulation)
#   - Sliding-window convolution pipeline (no im2col materialisation)
#   - Pipelined reduction for norms and pooling
#
# Activated when the `FPGASynthesis` weak-dep is loaded.

module AxiomFPGAExt

using FPGASynthesis
using Axiom

# ============================================================================
# Pipeline Helpers
# ============================================================================

"""
    _streaming_matmul(A, B)

Row-streaming matrix multiply that mirrors FPGA systolic pipeline execution.
Processes A one row at a time, streaming the entire B matrix through
a multiply-accumulate pipeline per output element.
"""
function _streaming_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    m, k = size(A)
    _, n = size(B)
    C = zeros(T, m, n)
    # Each row of A streams through independently (pipeline parallelism)
    for i in 1:m
        # Pipeline stage: dot-product accumulation across K dimension
        for j in 1:n
            acc = zero(T)
            for p in 1:k
                acc += A[i, p] * B[p, j]
            end
            C[i, j] = acc
        end
    end
    C
end

"""
    _sliding_window_conv(x, w, sH, sW, pH, pW)

Sliding-window convolution pipeline for a single spatial volume (H, W, C_in).
Instead of materialising the full im2col matrix, processes one output position
at a time through a multiply-accumulate pipeline — matching FPGA line-buffer
convolution architectures.
"""
function _sliding_window_conv(
    x::AbstractArray{T,3}, w::AbstractArray{T,4},
    sH::Int, sW::Int, pH::Int, pW::Int,
) where {T}
    H, W, C_in = size(x)
    kH, kW, _, C_out = size(w)
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    # Line buffer: pad only the spatial dims
    if pH > 0 || pW > 0
        xp = zeros(T, H + 2*pH, W + 2*pW, C_in)
        xp[pH+1:pH+H, pW+1:pW+W, :] .= x
    else
        xp = x
    end

    out = zeros(T, H_out, W_out, C_out)
    # Sliding window pipeline: one output pixel per clock (conceptually)
    for i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1
        ws = (j - 1) * sW + 1
        for oc in 1:C_out
            acc = zero(T)
            for ki in 1:kH, kj in 1:kW, ic in 1:C_in
                acc += xp[hs + ki - 1, ws + kj - 1, ic] * w[ki, kj, ic, oc]
            end
            out[i, j, oc] = acc
        end
    end
    out
end

# ============================================================================
# Coprocessor Hooks
# ============================================================================

function Axiom.backend_coprocessor_matmul(
    ::Axiom.FPGABackend,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T <: AbstractFloat}
    _streaming_matmul(A, B)
end

function Axiom.backend_coprocessor_matmul(
    backend::Axiom.FPGABackend,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    T = promote_type(eltype(A), eltype(B))
    T = T <: AbstractFloat ? T : Float32
    _streaming_matmul(T.(A), T.(B))
end

function Axiom.backend_coprocessor_conv2d(
    ::Axiom.FPGABackend,
    input::AbstractArray{T,4},
    weight::AbstractArray{T,4},
    bias::Union{AbstractVector{T},Nothing},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int},
) where {T}
    N = size(input, 1)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride; pH, pW = padding
    H_out = div(size(input, 2) + 2*pH - kH, sH) + 1
    W_out = div(size(input, 3) + 2*pW - kW, sW) + 1

    output = zeros(T, N, H_out, W_out, C_out)
    for n in 1:N
        output[n, :, :, :] .= _sliding_window_conv(
            view(input, n, :, :, :), weight, sH, sW, pH, pW,
        )
    end

    if bias !== nothing
        for oc in 1:C_out
            output[:, :, :, oc] .+= bias[oc]
        end
    end
    output
end

function Axiom.backend_coprocessor_relu(
    ::Axiom.FPGABackend,
    x::AbstractArray{T},
) where {T}
    # FPGA: single-cycle comparator per element in the pipeline
    max.(x, zero(T))
end

function Axiom.backend_coprocessor_softmax(
    ::Axiom.FPGABackend,
    x::AbstractArray{T},
    dim::Int,
) where {T}
    # Two-pass pipeline: (1) streaming max-reduction, (2) exp + sum + divide
    x_max = maximum(x, dims=dim)
    x_exp = exp.(x .- x_max)
    x_exp ./ sum(x_exp, dims=dim)
end

function Axiom.backend_coprocessor_batchnorm(
    ::Axiom.FPGABackend,
    x::AbstractArray{T},
    gamma::AbstractVector{T},
    beta::AbstractVector{T},
    running_mean::AbstractVector{T},
    running_var::AbstractVector{T},
    eps::T,
    training::Bool,
) where {T}
    nd = ndims(x)
    shape = ntuple(i -> i == nd ? length(gamma) : 1, nd)
    γ = reshape(gamma, shape)
    β = reshape(beta, shape)
    μ = reshape(running_mean, shape)
    σ² = reshape(running_var, shape)
    # Pipelined normalise: precompute reciprocal sqrt (FPGA can pipeline divisions)
    inv_std = one(T) ./ sqrt.(σ² .+ eps)
    γ .* (x .- μ) .* inv_std .+ β
end

function Axiom.backend_coprocessor_layernorm(
    ::Axiom.FPGABackend,
    x::AbstractArray{T},
    gamma::AbstractArray{T},
    beta::AbstractArray{T},
    normalized_shape::Tuple,
    eps::T,
) where {T}
    n_norm = length(normalized_shape)
    nd = ndims(x)
    reduce_dims = ntuple(i -> nd - n_norm + i, n_norm)
    n = prod(size(x, d) for d in reduce_dims)
    μ = sum(x, dims=reduce_dims) ./ T(n)
    diff = x .- μ
    σ² = sum(diff .^ 2, dims=reduce_dims) ./ T(n)
    inv_std = one(T) ./ sqrt.(σ² .+ eps)
    gamma .* diff .* inv_std .+ beta
end

function Axiom.backend_coprocessor_maxpool2d(
    ::Axiom.FPGABackend,
    input::AbstractArray{T,4},
    kernel_size::Tuple{Int,Int},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int},
) where {T}
    N, H, W, C = size(input)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if pH > 0 || pW > 0
        padded = fill(T(-Inf), N, H + 2*pH, W + 2*pW, C)
        padded[:, pH+1:pH+H, pW+1:pW+W, :] .= input
    else
        padded = input
    end

    output = Array{T}(undef, N, H_out, W_out, C)
    # Pipeline: comparator tree per kernel window
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
        output[n, i, j, c] = maximum(@view padded[n, hs:hs+kH-1, ws:ws+kW-1, c])
    end
    output
end

function Axiom.backend_coprocessor_avgpool2d(
    ::Axiom.FPGABackend,
    input::AbstractArray{T,4},
    kernel_size::Tuple{Int,Int},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int},
    count_include_pad::Bool=true,
) where {T}
    N, H, W, C = size(input)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if pH > 0 || pW > 0
        padded = zeros(T, N, H + 2*pH, W + 2*pW, C)
        padded[:, pH+1:pH+H, pW+1:pW+W, :] .= input
    else
        padded = input
    end

    output = Array{T}(undef, N, H_out, W_out, C)
    inv_k = one(T) / T(kH * kW)
    # Pipeline: adder tree + multiply-by-reciprocal
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
        output[n, i, j, c] = sum(@view padded[n, hs:hs+kH-1, ws:ws+kW-1, c]) * inv_k
    end
    output
end

function Axiom.backend_coprocessor_global_avgpool2d(
    ::Axiom.FPGABackend,
    input::AbstractArray{T,4},
) where {T}
    N, H, W, C = size(input)
    output = Array{T}(undef, N, C)
    inv_hw = one(T) / T(H * W)
    for n in 1:N, c in 1:C
        output[n, c] = sum(@view input[n, :, :, c]) * inv_hw
    end
    output
end

end # module AxiomFPGAExt
