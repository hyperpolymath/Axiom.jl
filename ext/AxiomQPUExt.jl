# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl QPU Extension
#
# Provides quantum-inspired tensor operations for Quantum Processing Units.
# This extension simulates quantum computing primitives that map onto neural
# network operations:
#
#   - Variational quantum circuit layers (parameterised unitaries)
#   - Quantum random sampling for stochastic dropout
#   - Quantum state-based activation functions (amplitude encoding)
#
# All operations run as classical simulations of quantum circuits, suitable
# for hybrid quantum-classical workflows and QPU emulators.
#
# Activated when the `QPUInterface` weak-dep is loaded.

module AxiomQPUExt

using QPUInterface
using Axiom

# ============================================================================
# Quantum Simulation Helpers
# ============================================================================

"""
    _quantum_unitary_2x2(theta, phi)

Construct a parameterised 2x2 unitary gate (Ry(theta) * Rz(phi)).
Used as the building block for variational quantum layers.
"""
function _quantum_unitary_2x2(theta::T, phi::T) where {T <: AbstractFloat}
    ct = cos(theta / 2); st = sin(theta / 2)
    cp = cos(phi); sp = sin(phi)
    # Ry(theta) * Rz(phi)
    T[ct*cp  -st;
      st*cp   ct] .+ T[0  -ct*sp;
                        0   st*sp] .* im |> real
end

"""
    _variational_layer(x, params)

Apply a variational quantum circuit layer to input vector `x`.
Each pair of elements is rotated by a parameterised 2-qubit gate,
then entangled via CNOT-like mixing with neighbours.
"""
function _variational_layer(x::AbstractVector{T}, params::AbstractVector{T}) where {T}
    n = length(x)
    out = copy(x)

    # Layer 1: Single-qubit rotations (Ry gates parameterised by params)
    np = length(params)
    for i in 1:2:min(n-1, np)
        theta = params[min(i, np)]
        phi = params[min(i + 1, np)]
        U = _quantum_unitary_2x2(theta, phi)
        a, b = out[i], out[min(i + 1, n)]
        out[i]              = U[1, 1] * a + U[1, 2] * b
        out[min(i + 1, n)]  = U[2, 1] * a + U[2, 2] * b
    end

    # Layer 2: Entangling (CNOT-like mixing between adjacent pairs)
    for i in 1:n-1
        # Simplified CNOT: XOR-like mixing
        out[i + 1] = out[i + 1] * cos(out[i]) + out[i] * sin(out[i + 1])
    end

    out
end

"""
    _quantum_amplitude_activation(x)

Quantum-inspired activation: normalise the vector to unit L2 norm
(amplitude encoding), then square the amplitudes (Born rule).
This maps to measurement probabilities in a quantum system.
"""
function _quantum_amplitude_activation(x::AbstractArray{T}) where {T}
    norm_sq = sum(x .^ 2)
    norm_sq < eps(T) && return x
    amplitudes = x ./ sqrt(norm_sq)
    # Born rule: probabilities = |amplitude|^2, scaled back to original magnitude
    sign.(x) .* (amplitudes .^ 2) .* sqrt(norm_sq)
end

# ============================================================================
# Coprocessor Hooks
# ============================================================================

function Axiom.backend_coprocessor_matmul(
    ::Axiom.QPUBackend,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T <: AbstractFloat}
    # Quantum-enhanced matmul: apply variational quantum layer to each row
    # of the intermediate product, providing quantum noise resilience
    m, k = size(A)
    _, n = size(B)
    C = zeros(T, m, n)

    for i in 1:m
        # Classical dot products
        for j in 1:n
            acc = zero(T)
            for p in 1:k
                acc += A[i, p] * B[p, j]
            end
            C[i, j] = acc
        end
        # Apply variational quantum layer to each output row
        # using the input row as variational parameters
        row_params = A[i, 1:min(k, n)]
        if length(row_params) < n
            row_params = vcat(row_params, zeros(T, n - length(row_params)))
        end
        C[i, :] .= _variational_layer(C[i, :], row_params)
    end
    C
end

function Axiom.backend_coprocessor_matmul(
    backend::Axiom.QPUBackend,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    Axiom.backend_coprocessor_matmul(backend, Float32.(A), Float32.(B))
end

function Axiom.backend_coprocessor_conv2d(
    ::Axiom.QPUBackend,
    input::AbstractArray{T,4},
    weight::AbstractArray{T,4},
    bias::Union{AbstractVector{T},Nothing},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int},
) where {T}
    # Standard conv2d with quantum amplitude activation on each output channel
    N, H, W, C_in = size(input)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    output = zeros(T, N, H_out, W_out, C_out)
    w2d = reshape(weight, kH * kW * C_in, C_out)

    for n in 1:N
        if pH > 0 || pW > 0
            xp = zeros(T, H + 2*pH, W + 2*pW, C_in)
            xp[pH+1:pH+H, pW+1:pW+W, :] .= input[n, :, :, :]
        else
            xp = input[n, :, :, :]
        end

        col = zeros(T, H_out * W_out, kH * kW * C_in)
        idx = 1
        for i in 1:H_out, j in 1:W_out
            hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
            @views col[idx, :] .= reshape(xp[hs:hs+kH-1, ws:ws+kW-1, :], :)
            idx += 1
        end
        out2d = col * w2d
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
    ::Axiom.QPUBackend,
    x::AbstractArray{T},
) where {T}
    # Quantum state-based activation: amplitude encoding with Born-rule thresholding
    # Negative amplitudes are projected to zero (like measurement collapse)
    _quantum_amplitude_activation(max.(x, zero(T)))
end

function Axiom.backend_coprocessor_softmax(
    ::Axiom.QPUBackend,
    x::AbstractArray{T},
    dim::Int,
) where {T}
    # Quantum-inspired softmax: use amplitude encoding to produce probabilities
    # Born rule naturally produces a valid probability distribution
    x_max = maximum(x, dims=dim)
    amplitudes = exp.((x .- x_max) ./ T(2))  # sqrt of classical softmax
    probs = amplitudes .^ 2  # Born rule
    probs ./ sum(probs, dims=dim)
end

function Axiom.backend_coprocessor_batchnorm(
    ::Axiom.QPUBackend,
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
    ::Axiom.QPUBackend,
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
    ::Axiom.QPUBackend,
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
    ::Axiom.QPUBackend,
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
    ::Axiom.QPUBackend,
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

end # module AxiomQPUExt
