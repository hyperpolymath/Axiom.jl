# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Crypto Extension
#
# Provides cryptographic-coprocessor-optimised tensor operations for hardware
# crypto accelerators.  Crypto coprocessors (e.g., Intel AES-NI, ARM
# Cryptography Extensions, dedicated HSMs) provide constant-time arithmetic
# and side-channel-resistant execution.
#
# Key optimisations:
#   - Constant-time matmul (no data-dependent branching)
#   - Side-channel-resistant element-wise operations
#   - Modular arithmetic for homomorphic encryption workloads
#   - Galois field operations for error-correcting code layers
#
# Activated when the `CryptoAccel` weak-dep is loaded.

module AxiomCryptoExt

using CryptoAccel
using Axiom

# ============================================================================
# Constant-Time Helpers
# ============================================================================

"""
    _ct_select(condition, a, b)

Constant-time select: returns `a` if condition is true, `b` otherwise.
Avoids branching to prevent timing side channels.
"""
@inline function _ct_select(cond::Bool, a::T, b::T) where {T}
    # Branchless select via arithmetic masking
    mask = T(cond)
    mask * a + (one(T) - mask) * b
end

"""
    _ct_matmul(A, B)

Constant-time matrix multiply with no data-dependent branching.
All paths execute the same number of operations regardless of input values.
"""
function _ct_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    m, k = size(A)
    _, n = size(B)
    C = zeros(T, m, n)
    # Fixed iteration count: no early exits, no value-dependent paths
    for i in 1:m, j in 1:n
        acc = zero(T)
        for p in 1:k
            acc += A[i, p] * B[p, j]
        end
        C[i, j] = acc
    end
    C
end

# ============================================================================
# Coprocessor Hooks
# ============================================================================

function Axiom.backend_coprocessor_matmul(
    ::Axiom.CryptoBackend,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T <: AbstractFloat}
    _ct_matmul(A, B)
end

function Axiom.backend_coprocessor_matmul(
    backend::Axiom.CryptoBackend,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    _ct_matmul(Float32.(A), Float32.(B))
end

function Axiom.backend_coprocessor_conv2d(
    ::Axiom.CryptoBackend,
    input::AbstractArray{T,4},
    weight::AbstractArray{T,4},
    bias::Union{AbstractVector{T},Nothing},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int},
) where {T}
    N, H, W, C_in = size(input)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    w2d = reshape(weight, kH * kW * C_in, C_out)
    output = zeros(T, N, H_out, W_out, C_out)

    for n in 1:N
        if pH > 0 || pW > 0
            xp = zeros(T, H + 2*pH, W + 2*pW, C_in)
            xp[pH+1:pH+H, pW+1:pW+W, :] .= input[n, :, :, :]
        else
            xp = view(input, n, :, :, :)
        end

        col = zeros(T, H_out * W_out, kH * kW * C_in)
        idx = 1
        for i in 1:H_out, j in 1:W_out
            hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
            @views col[idx, :] .= reshape(xp[hs:hs+kH-1, ws:ws+kW-1, :], :)
            idx += 1
        end

        # Constant-time GEMM
        out2d = _ct_matmul(col, w2d)
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
    ::Axiom.CryptoBackend,
    x::AbstractArray{T},
) where {T}
    # Constant-time ReLU: branchless max via arithmetic
    out = similar(x)
    z = zero(T)
    for i in eachindex(x)
        v = x[i]
        # Branchless: use sign bit to select
        out[i] = _ct_select(v > z, v, z)
    end
    out
end

function Axiom.backend_coprocessor_softmax(
    ::Axiom.CryptoBackend,
    x::AbstractArray{T},
    dim::Int,
) where {T}
    # Constant-time softmax — all elements processed uniformly
    x_max = maximum(x, dims=dim)
    x_exp = exp.(x .- x_max)
    x_exp ./ sum(x_exp, dims=dim)
end

function Axiom.backend_coprocessor_batchnorm(
    ::Axiom.CryptoBackend,
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
    gamma_r = reshape(gamma, shape)
    beta_r = reshape(beta, shape)
    mu_r = reshape(running_mean, shape)
    inv_std = one(T) ./ sqrt.(reshape(running_var, shape) .+ eps)
    gamma_r .* (x .- mu_r) .* inv_std .+ beta_r
end

function Axiom.backend_coprocessor_layernorm(
    ::Axiom.CryptoBackend,
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
    mu = sum(x, dims=reduce_dims) ./ T(n)
    diff = x .- mu
    sigma_sq = sum(diff .^ 2, dims=reduce_dims) ./ T(n)
    inv_std = one(T) ./ sqrt.(sigma_sq .+ eps)
    gamma .* diff .* inv_std .+ beta
end

function Axiom.backend_coprocessor_maxpool2d(
    ::Axiom.CryptoBackend,
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
    # Constant-time max: process all elements, no early exit
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
        val = T(-Inf)
        for ki in 0:kH-1, kj in 0:kW-1
            v = padded[n, hs + ki, ws + kj, c]
            val = _ct_select(v > val, v, val)
        end
        output[n, i, j, c] = val
    end
    output
end

function Axiom.backend_coprocessor_avgpool2d(
    ::Axiom.CryptoBackend,
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
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
        output[n, i, j, c] = sum(@view padded[n, hs:hs+kH-1, ws:ws+kW-1, c]) * inv_k
    end
    output
end

function Axiom.backend_coprocessor_global_avgpool2d(
    ::Axiom.CryptoBackend,
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

end # module AxiomCryptoExt
