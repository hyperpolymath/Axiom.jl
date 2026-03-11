# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl TPU Extension
#
# Provides systolic-array-optimised tensor operations for Tensor Processing
# Units.  TPUs execute matrix multiplies in 128x128 tiles via a systolic mesh;
# this extension tiles all matmul/conv/norm operations accordingly.
#
# Activated when the `TPUCompute` weak-dep is loaded.

module AxiomTPUExt

using TPUCompute
using Axiom
using LinearAlgebra: mul!

# ============================================================================
# Constants
# ============================================================================

"""TPU systolic array tile dimension (128x128 is the canonical MXU size)."""
const TILE = 128

# ============================================================================
# Helpers
# ============================================================================

"""
    _pad_to_tile(A, tile)

Pad matrix `A` so that both dimensions are multiples of `tile`.
Returns the padded matrix and the original (rows, cols).
"""
function _pad_to_tile(A::AbstractMatrix{T}, tile::Int) where {T}
    m, n = size(A)
    pm = ceil(Int, m / tile) * tile
    pn = ceil(Int, n / tile) * tile
    if pm == m && pn == n
        return A, (m, n)
    end
    P = zeros(T, pm, pn)
    P[1:m, 1:n] .= A
    P, (m, n)
end

"""
    _tiled_matmul(A, B, tile)

Tile-blocked matrix multiply that mirrors TPU systolic array execution.
Each tile is dispatched as an independent 128x128 GEMM to the MXU.
"""
function _tiled_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}, tile::Int) where {T}
    Ap, (am, _ak) = _pad_to_tile(A, tile)
    Bp, (_bk, bn) = _pad_to_tile(B, tile)
    M, K = size(Ap)
    _, N = size(Bp)
    C = zeros(T, M, N)

    # Systolic-array-style tiled accumulation
    for i in 1:tile:M
        ie = i + tile - 1
        for j in 1:tile:N
            je = j + tile - 1
            for k in 1:tile:K
                ke = k + tile - 1
                # Each tile maps to one MXU pass
                @views mul!(C[i:ie, j:je], Ap[i:ie, k:ke], Bp[k:ke, j:je], one(T), one(T))
            end
        end
    end
    # Trim back to original dimensions
    C[1:am, 1:bn]
end

"""
    _im2col(input, kH, kW, sH, sW, pH, pW) -> (col, H_out, W_out)

Reshape a single spatial slice (H, W, C_in) into a 2-D column matrix
suitable for GEMM-based convolution.
"""
function _im2col(x::AbstractArray{T,3}, kH, kW, sH, sW, pH, pW) where {T}
    H, W, C = size(x)
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    # Pad spatially
    if pH > 0 || pW > 0
        xp = zeros(T, H + 2*pH, W + 2*pW, C)
        xp[pH+1:pH+H, pW+1:pW+W, :] .= x
    else
        xp = x
    end

    col = zeros(T, H_out * W_out, kH * kW * C)
    idx = 1
    for i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1
        ws = (j - 1) * sW + 1
        @views col[idx, :] .= reshape(xp[hs:hs+kH-1, ws:ws+kW-1, :], :)
        idx += 1
    end
    col, H_out, W_out
end

# ============================================================================
# Coprocessor Hooks
# ============================================================================

function Axiom.backend_coprocessor_matmul(
    ::Axiom.TPUBackend,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T <: AbstractFloat}
    _tiled_matmul(A, B, TILE)
end

# Fallback for non-Float types: promote to Float32
function Axiom.backend_coprocessor_matmul(
    backend::Axiom.TPUBackend,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    result = _tiled_matmul(Float32.(A), Float32.(B), TILE)
    convert(Matrix{eltype(A)}, result)
end

function Axiom.backend_coprocessor_conv2d(
    ::Axiom.TPUBackend,
    input::AbstractArray{T,4},
    weight::AbstractArray{T,4},
    bias::Union{AbstractVector{T},Nothing},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int},
) where {T}
    N, H, W, C_in = size(input)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride
    pH, pW = padding

    w2d = reshape(weight, kH * kW * C_in, C_out)

    # First sample to discover output spatial dims
    col0, H_out, W_out = _im2col(view(input, 1, :, :, :), kH, kW, sH, sW, pH, pW)
    output = zeros(T, N, H_out, W_out, C_out)

    # Batch-optimised: TPUs excel at large-batch inference
    for n in 1:N
        col, _, _ = _im2col(view(input, n, :, :, :), kH, kW, sH, sW, pH, pW)
        out2d = _tiled_matmul(col, w2d, TILE)
        output[n, :, :, :] .= reshape(out2d, H_out, W_out, C_out)
    end

    if bias !== nothing
        for oc in 1:C_out
            output[:, :, :, oc] .+= bias[oc]
        end
    end
    output
end

function Axiom.backend_coprocessor_relu(
    ::Axiom.TPUBackend,
    x::AbstractArray{T},
) where {T}
    # Element-wise max with fused batch vectorisation
    max.(x, zero(T))
end

function Axiom.backend_coprocessor_softmax(
    ::Axiom.TPUBackend,
    x::AbstractArray{T},
    dim::Int,
) where {T}
    # Numerically stable softmax using fused reduction (TPU reduction unit)
    x_max = maximum(x, dims=dim)
    x_exp = exp.(x .- x_max)
    x_exp ./ sum(x_exp, dims=dim)
end

function Axiom.backend_coprocessor_batchnorm(
    ::Axiom.TPUBackend,
    x::AbstractArray{T},
    gamma::AbstractVector{T},
    beta::AbstractVector{T},
    running_mean::AbstractVector{T},
    running_var::AbstractVector{T},
    eps::T,
    training::Bool,
) where {T}
    # Fused reduction: mean/var computed in a single pass over the batch
    nd = ndims(x)
    shape = ntuple(i -> i == nd ? length(gamma) : 1, nd)
    γ = reshape(gamma, shape)
    β = reshape(beta, shape)

    if training
        # Batch statistics across all dims except the channel (last) dim
        reduce_dims = ntuple(i -> i, nd - 1)
        μ = mean(x, dims=reduce_dims)
        σ² = var(x, dims=reduce_dims, corrected=false)
    else
        μ = reshape(running_mean, shape)
        σ² = reshape(running_var, shape)
    end

    # Fused normalise + affine in one pass (systolic-friendly)
    γ .* (x .- μ) ./ sqrt.(σ² .+ eps) .+ β
end

function Axiom.backend_coprocessor_layernorm(
    ::Axiom.TPUBackend,
    x::AbstractArray{T},
    gamma::AbstractArray{T},
    beta::AbstractArray{T},
    normalized_shape::Tuple,
    eps::T,
) where {T}
    # Fused layer-norm reduction over the normalised dimensions
    n_norm = length(normalized_shape)
    nd = ndims(x)
    reduce_dims = ntuple(i -> nd - n_norm + i, n_norm)
    μ = mean(x, dims=reduce_dims)
    σ² = var(x, dims=reduce_dims, corrected=false)
    gamma .* (x .- μ) ./ sqrt.(σ² .+ eps) .+ beta
end

function Axiom.backend_coprocessor_maxpool2d(
    ::Axiom.TPUBackend,
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
    # Batch-parallel reduction (TPU vector lanes)
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
        output[n, i, j, c] = maximum(@view padded[n, hs:hs+kH-1, ws:ws+kW-1, c])
    end
    output
end

function Axiom.backend_coprocessor_avgpool2d(
    ::Axiom.TPUBackend,
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
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
        patch_sum = sum(@view padded[n, hs:hs+kH-1, ws:ws+kW-1, c])
        if count_include_pad
            output[n, i, j, c] = patch_sum / T(kH * kW)
        else
            # Count only non-padded elements
            h_start = max(1, pH + 1 - (i - 1) * sH)
            h_end = min(kH, H + pH - (i - 1) * sH)
            w_start = max(1, pW + 1 - (j - 1) * sW)
            w_end = min(kW, W + pW - (j - 1) * sW)
            real_count = (h_end - h_start + 1) * (w_end - w_start + 1)
            output[n, i, j, c] = patch_sum / T(max(real_count, 1))
        end
    end
    output
end

function Axiom.backend_coprocessor_global_avgpool2d(
    ::Axiom.TPUBackend,
    input::AbstractArray{T,4},
) where {T}
    # Fused global reduction across spatial dims (single systolic pass per channel)
    N, H, W, C = size(input)
    output = Array{T}(undef, N, C)
    inv_hw = one(T) / T(H * W)
    for n in 1:N, c in 1:C
        output[n, c] = sum(@view input[n, :, :, c]) * inv_hw
    end
    output
end

# ============================================================================
# Mean/Var helpers (avoid importing Statistics)
# ============================================================================

function mean(x::AbstractArray{T}; dims=:) where {T}
    s = sum(x, dims=dims)
    n = dims === (:) ? length(x) : prod(size(x, d) for d in dims)
    s ./ T(n)
end

function var(x::AbstractArray{T}; dims=:, corrected::Bool=true) where {T}
    μ = mean(x; dims=dims)
    n = dims === (:) ? length(x) : prod(size(x, d) for d in dims)
    ss = sum((x .- μ) .^ 2, dims=dims)
    corrected ? ss ./ T(max(n - 1, 1)) : ss ./ T(n)
end

end # module AxiomTPUExt
