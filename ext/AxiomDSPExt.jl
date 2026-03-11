# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl DSP Extension
#
# Provides FFT-based convolution and signal-processing-optimised operations
# for Digital Signal Processors.  DSPs have native hardware support for FFT
# butterflies, MAC (multiply-accumulate) units, and circular buffers.
#
# Key optimisations:
#   - FFT-based convolution (O(N log N) vs O(N*K) for large kernels)
#   - Frequency-domain filtering for signal processing layers
#   - Efficient MAC-based matmul and reductions
#
# Activated when the `DSPLibs` weak-dep is loaded.

module AxiomDSPExt

using DSPLibs
using Axiom

# ============================================================================
# FFT Convolution Helpers
# ============================================================================

"""
    _fft_conv1d(signal, kernel)

1-D convolution via FFT: pad to power-of-two, multiply in frequency domain,
IFFT back.  This is O(N log N) vs O(N*K) for direct convolution.
"""
function _fft_conv1d(signal::AbstractVector{T}, kernel::AbstractVector{T}) where {T}
    ns = length(signal)
    nk = length(kernel)
    # Output length for "valid" linear convolution
    n_out = ns + nk - 1
    # Pad to next power of 2 for FFT efficiency
    n_fft = nextpow(2, n_out)

    sig_padded = zeros(Complex{T}, n_fft)
    ker_padded = zeros(Complex{T}, n_fft)
    sig_padded[1:ns] .= signal
    ker_padded[1:nk] .= kernel

    # Forward FFT (Cooley-Tukey radix-2 DIT)
    _fft_inplace!(sig_padded)
    _fft_inplace!(ker_padded)

    # Pointwise multiply in frequency domain
    result_freq = sig_padded .* ker_padded

    # Inverse FFT
    _ifft_inplace!(result_freq)

    real.(result_freq[1:n_out])
end

"""
    _fft_inplace!(x)

In-place radix-2 decimation-in-time FFT.
DSP hardware implements this as a butterfly network.
"""
function _fft_inplace!(x::AbstractVector{Complex{T}}) where {T}
    n = length(x)
    n <= 1 && return x

    # Bit-reversal permutation
    j = 1
    for i in 1:n-1
        if i < j
            x[i], x[j] = x[j], x[i]
        end
        m = n >> 1
        while m >= 2 && j > m
            j -= m
            m >>= 1
        end
        j += m
    end

    # Butterfly stages
    len = 2
    while len <= n
        half = len >> 1
        w_base = T(-2) * T(pi) / T(len)
        for start in 1:len:n
            for k in 0:half-1
                w = exp(Complex{T}(0, w_base * k))
                u = x[start + k]
                v = x[start + k + half] * w
                x[start + k]        = u + v
                x[start + k + half] = u - v
            end
        end
        len <<= 1
    end
    x
end

"""
    _ifft_inplace!(x)

In-place inverse FFT (conjugate + forward FFT + conjugate + scale).
"""
function _ifft_inplace!(x::AbstractVector{Complex{T}}) where {T}
    n = length(x)
    x .= conj.(x)
    _fft_inplace!(x)
    x .= conj.(x) ./ T(n)
    x
end

# ============================================================================
# Coprocessor Hooks
# ============================================================================

function Axiom.backend_coprocessor_matmul(
    ::Axiom.DSPBackend,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T <: AbstractFloat}
    # MAC-unit-optimised matmul: inner loop mirrors DSP MAC pipeline
    m, k = size(A)
    _, n = size(B)
    C = zeros(T, m, n)
    for i in 1:m, j in 1:n
        acc = zero(T)
        # Single-cycle MAC per iteration on DSP hardware
        for p in 1:k
            acc = muladd(A[i, p], B[p, j], acc)
        end
        C[i, j] = acc
    end
    C
end

function Axiom.backend_coprocessor_matmul(
    backend::Axiom.DSPBackend,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    Axiom.backend_coprocessor_matmul(backend, Float32.(A), Float32.(B))
end

function Axiom.backend_coprocessor_conv2d(
    ::Axiom.DSPBackend,
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

    output = zeros(T, N, H_out, W_out, C_out)

    # Decide: FFT conv for large kernels, direct for small
    use_fft = kH * kW >= 25  # 5x5 or larger → FFT is faster on DSP

    for n in 1:N
        if pH > 0 || pW > 0
            xp = zeros(T, H + 2*pH, W + 2*pW, C_in)
            xp[pH+1:pH+H, pW+1:pW+W, :] .= input[n, :, :, :]
        else
            xp = input[n, :, :, :]
        end

        if use_fft
            # FFT-based: convolve each input/output channel pair via 1-D FFT
            # over flattened spatial dims (row-major scan)
            for oc in 1:C_out
                acc = zeros(T, H + 2*pH, W + 2*pW)
                for ic in 1:C_in
                    sig = reshape(xp[:, :, ic], :)
                    ker = reshape(weight[:, :, ic, oc], :)
                    conv_result = _fft_conv1d(sig, ker)
                    # Extract valid region (simplified: take strided subset)
                    full_h = H + 2*pH
                    full_w = W + 2*pW
                    for ci in 1:min(length(conv_result), full_h * full_w)
                        acc[ci] += conv_result[ci]
                    end
                end
                # Strided extraction from the full convolution result
                for i in 1:H_out, j in 1:W_out
                    hi = (i - 1) * sH + 1; wi = (j - 1) * sW + 1
                    if hi <= size(acc, 1) && wi <= size(acc, 2)
                        output[n, i, j, oc] = acc[hi, wi]
                    end
                end
            end
        else
            # Direct conv with MAC units
            for i in 1:H_out, j in 1:W_out
                hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
                for oc in 1:C_out
                    acc = zero(T)
                    for ki in 1:kH, kj in 1:kW, ic in 1:C_in
                        acc = muladd(
                            xp[hs + ki - 1, ws + kj - 1, ic],
                            weight[ki, kj, ic, oc],
                            acc,
                        )
                    end
                    output[n, i, j, oc] = acc
                end
            end
        end
    end

    if bias !== nothing
        for oc in 1:C_out
            output[:, :, :, oc] .+= bias[oc]
        end
    end
    output
end

function Axiom.backend_coprocessor_relu(
    ::Axiom.DSPBackend,
    x::AbstractArray{T},
) where {T}
    max.(x, zero(T))
end

function Axiom.backend_coprocessor_softmax(
    ::Axiom.DSPBackend,
    x::AbstractArray{T},
    dim::Int,
) where {T}
    x_max = maximum(x, dims=dim)
    x_exp = exp.(x .- x_max)
    x_exp ./ sum(x_exp, dims=dim)
end

function Axiom.backend_coprocessor_batchnorm(
    ::Axiom.DSPBackend,
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
    γ .* (x .- μ) ./ sqrt.(σ² .+ eps) .+ β
end

function Axiom.backend_coprocessor_layernorm(
    ::Axiom.DSPBackend,
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
    σ² = sum((x .- μ) .^ 2, dims=reduce_dims) ./ T(n)
    gamma .* (x .- μ) ./ sqrt.(σ² .+ eps) .+ beta
end

function Axiom.backend_coprocessor_maxpool2d(
    ::Axiom.DSPBackend,
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
        output[n, i, j, c] = maximum(@view padded[n, hs:hs+kH-1, ws:ws+kW-1, c])
    end
    output
end

function Axiom.backend_coprocessor_avgpool2d(
    ::Axiom.DSPBackend,
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
    ::Axiom.DSPBackend,
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

end # module AxiomDSPExt
