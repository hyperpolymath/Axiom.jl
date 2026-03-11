# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl VPU Extension
#
# Provides SIMD-vector-optimised tensor operations for Vector/Vision
# Processing Units.  VPUs are wide SIMD engines (256-512 bit lanes) found on
# DSP-class or vision-class SoCs (e.g., Intel Movidius, Cadence P6).
#
# Key optimisations:
#   - SIMD-tiled matrix multiply (process SIMD_WIDTH elements per cycle)
#   - Vectorised im2col convolution
#   - Element-wise SIMD ops for activations and pooling
#
# Activated when the `VPURuntime` weak-dep is loaded.

module AxiomVPUExt

using VPURuntime
using Axiom

# ============================================================================
# Constants
# ============================================================================

"""
Number of Float32 elements that fit in a single SIMD register.
Typical VPUs have 256-bit or 512-bit lanes → 8 or 16 floats.
"""
const SIMD_WIDTH = 8

# ============================================================================
# SIMD Helpers
# ============================================================================

"""
    _simd_tiled_matmul(A, B, sw)

SIMD-tiled matrix multiply: inner loop processes `sw` elements at a time
to exploit wide vector registers.
"""
function _simd_tiled_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}, sw::Int) where {T}
    m, k = size(A)
    _, n = size(B)
    C = zeros(T, m, n)

    # Tile the N (column) dimension in chunks of SIMD_WIDTH
    n_tiles = div(n, sw)
    n_rem = n - n_tiles * sw

    for i in 1:m
        # Full SIMD-width tiles
        for jt in 0:n_tiles-1
            j0 = jt * sw
            # Accumulate SIMD_WIDTH outputs simultaneously
            acc = zeros(T, sw)
            for p in 1:k
                a_val = A[i, p]
                @simd for s in 1:sw
                    @inbounds acc[s] += a_val * B[p, j0 + s]
                end
            end
            @simd for s in 1:sw
                @inbounds C[i, j0 + s] = acc[s]
            end
        end
        # Remainder (scalar tail)
        for j in (n_tiles * sw + 1):n
            acc = zero(T)
            for p in 1:k
                acc += A[i, p] * B[p, j]
            end
            C[i, j] = acc
        end
    end
    C
end

# ============================================================================
# Coprocessor Hooks
# ============================================================================

function Axiom.backend_coprocessor_matmul(
    ::Axiom.VPUBackend,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T <: AbstractFloat}
    _simd_tiled_matmul(A, B, SIMD_WIDTH)
end

function Axiom.backend_coprocessor_matmul(
    backend::Axiom.VPUBackend,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    _simd_tiled_matmul(Float32.(A), Float32.(B), SIMD_WIDTH)
end

function Axiom.backend_coprocessor_conv2d(
    ::Axiom.VPUBackend,
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

        # Vectorised im2col: build column matrix
        col = zeros(T, H_out * W_out, kH * kW * C_in)
        idx = 1
        for i in 1:H_out, j in 1:W_out
            hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
            patch = @view xp[hs:hs+kH-1, ws:ws+kW-1, :]
            # SIMD copy of the flattened patch
            flat = reshape(patch, :)
            len = length(flat)
            @simd for s in 1:len
                @inbounds col[idx, s] = flat[s]
            end
            idx += 1
        end

        # SIMD-tiled GEMM
        out2d = _simd_tiled_matmul(col, w2d, SIMD_WIDTH)
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
    ::Axiom.VPUBackend,
    x::AbstractArray{T},
) where {T}
    # SIMD element-wise max (vectorised across the flat array)
    out = similar(x)
    z = zero(T)
    @simd for i in eachindex(x)
        @inbounds out[i] = x[i] > z ? x[i] : z
    end
    out
end

function Axiom.backend_coprocessor_softmax(
    ::Axiom.VPUBackend,
    x::AbstractArray{T},
    dim::Int,
) where {T}
    x_max = maximum(x, dims=dim)
    x_shifted = x .- x_max
    # SIMD vectorised exp
    x_exp = similar(x)
    @simd for i in eachindex(x_shifted)
        @inbounds x_exp[i] = exp(x_shifted[i])
    end
    x_exp ./ sum(x_exp, dims=dim)
end

function Axiom.backend_coprocessor_batchnorm(
    ::Axiom.VPUBackend,
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
    inv_std = one(T) ./ sqrt.(reshape(running_var, shape) .+ eps)
    # Fused SIMD: (x - μ) * inv_std * γ + β
    γ .* (x .- μ) .* inv_std .+ β
end

function Axiom.backend_coprocessor_layernorm(
    ::Axiom.VPUBackend,
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
    ::Axiom.VPUBackend,
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
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
        # SIMD reduction: find max in kernel window
        val = T(-Inf)
        @simd for ki in 0:kH-1
            for kj in 0:kW-1
                @inbounds v = padded[n, hs + ki, ws + kj, c]
                val = v > val ? v : val
            end
        end
        output[n, i, j, c] = val
    end
    output
end

function Axiom.backend_coprocessor_avgpool2d(
    ::Axiom.VPUBackend,
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
        acc = zero(T)
        @simd for ki in 0:kH-1
            for kj in 0:kW-1
                @inbounds acc += padded[n, hs + ki, ws + kj, c]
            end
        end
        output[n, i, j, c] = acc * inv_k
    end
    output
end

function Axiom.backend_coprocessor_global_avgpool2d(
    ::Axiom.VPUBackend,
    input::AbstractArray{T,4},
) where {T}
    N, H, W, C = size(input)
    output = Array{T}(undef, N, C)
    inv_hw = one(T) / T(H * W)
    for n in 1:N, c in 1:C
        acc = zero(T)
        @simd for idx in eachindex(@view input[n, :, :, c])
            @inbounds acc += input[n, :, :, c][idx]
        end
        output[n, c] = acc * inv_hw
    end
    output
end

end # module AxiomVPUExt
