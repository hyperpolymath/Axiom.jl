# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl PPU Extension
#
# Provides physics-informed neural network operations for Physics Processing
# Units.  PPUs are specialised coprocessors for rigid-body dynamics, fluid
# simulation, and constraint solving (e.g., PhysX/Havok-class silicon).
#
# Key optimisations:
#   - Physics-informed matmul with energy conservation regularisation
#   - Hamiltonian-preserving activations (symplectic ReLU)
#   - Conservation-law-aware normalisation
#   - Rigid body / fluid simulation integration via physics residuals
#
# Activated when the `PPUCompute` weak-dep is loaded.

module AxiomPPUExt

using PPUCompute
using Axiom

# ============================================================================
# Physics Helpers
# ============================================================================

"""
    _energy_preserving_matmul(A, B)

Matrix multiply with energy conservation regularisation.
After the standard product, the result is rescaled so that the Frobenius
norm is bounded by the geometric mean of the input norms — preventing
unbounded energy injection in physics-informed layers.
"""
function _energy_preserving_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    C = A * B

    # Energy bound: ||C||_F <= sqrt(||A||_F * ||B||_F)
    norm_A = sqrt(sum(A .^ 2))
    norm_B = sqrt(sum(B .^ 2))
    norm_C = sqrt(sum(C .^ 2))
    energy_bound = sqrt(norm_A * norm_B)

    if norm_C > energy_bound && norm_C > eps(T)
        # Rescale to preserve energy bound (PPU hardware clamp)
        C .*= (energy_bound / norm_C)
    end
    C
end

"""
    _symplectic_relu(x)

Symplectic ReLU: preserves the symplectic structure of Hamiltonian systems.
For each pair of elements (q, p) interpreted as position-momentum pairs,
only applies ReLU to the position component while preserving momentum.
For odd-length inputs, the last element gets standard ReLU.
"""
function _symplectic_relu(x::AbstractArray{T}) where {T}
    out = copy(x)
    n = length(out)
    flat = vec(out)
    # Pairs: (q1, p1, q2, p2, ...)
    # ReLU on q (odd indices), preserve p (even indices)
    for i in 1:2:n
        flat[i] = max(flat[i], zero(T))
        # p (even index) is left unchanged — momentum is conserved
    end
    reshape(flat, size(x))
end

# ============================================================================
# Coprocessor Hooks
# ============================================================================

function Axiom.backend_coprocessor_matmul(
    ::Axiom.PPUBackend,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T <: AbstractFloat}
    _energy_preserving_matmul(A, B)
end

function Axiom.backend_coprocessor_matmul(
    backend::Axiom.PPUBackend,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    _energy_preserving_matmul(Float32.(A), Float32.(B))
end

function Axiom.backend_coprocessor_conv2d(
    ::Axiom.PPUBackend,
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
            xp = input[n, :, :, :]
        end

        col = zeros(T, H_out * W_out, kH * kW * C_in)
        idx = 1
        for i in 1:H_out, j in 1:W_out
            hs = (i - 1) * sH + 1; ws = (j - 1) * sW + 1
            @views col[idx, :] .= reshape(xp[hs:hs+kH-1, ws:ws+kW-1, :], :)
            idx += 1
        end
        # Energy-preserving GEMM
        out2d = _energy_preserving_matmul(col, w2d)
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
    ::Axiom.PPUBackend,
    x::AbstractArray{T},
) where {T}
    _symplectic_relu(x)
end

function Axiom.backend_coprocessor_softmax(
    ::Axiom.PPUBackend,
    x::AbstractArray{T},
    dim::Int,
) where {T}
    # Energy-normalised softmax: Boltzmann distribution interpretation
    # Temperature parameter T=1 gives standard softmax; PPU uses physical energy units
    x_max = maximum(x, dims=dim)
    x_exp = exp.(x .- x_max)
    x_exp ./ sum(x_exp, dims=dim)
end

function Axiom.backend_coprocessor_batchnorm(
    ::Axiom.PPUBackend,
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

    # Physics-informed normalisation: preserve the sign structure
    # (important for velocity/momentum fields where sign = direction)
    x_norm = (x .- μ) ./ sqrt.(σ² .+ eps)
    result = γ .* x_norm .+ β

    # Energy conservation check: bound output energy
    in_energy = sum(x .^ 2)
    out_energy = sum(result .^ 2)
    if out_energy > in_energy * T(2) && out_energy > eps(T)
        result .*= sqrt(in_energy / out_energy)
    end
    result
end

function Axiom.backend_coprocessor_layernorm(
    ::Axiom.PPUBackend,
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
    ::Axiom.PPUBackend,
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
    ::Axiom.PPUBackend,
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
    ::Axiom.PPUBackend,
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

end # module AxiomPPUExt
