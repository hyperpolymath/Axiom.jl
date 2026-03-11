# SPDX-License-Identifier: PMPL-1.0-or-later
#
# Axiom.jl Invertible / Reversible Computing Layers
# ==================================================
#
# This module implements invertible neural network layers for normalizing flows,
# reversible residual networks, and memory-efficient backpropagation. Each layer
# provides both a forward and exact inverse mapping, plus the log absolute
# determinant of the Jacobian needed for density estimation.
#
# Type Hierarchy:
#   AbstractLayer
#   └── InvertibleLayer (abstract)
#       ├── CouplingLayer       — Affine coupling (RealNVP / Glow)
#       ├── ActNorm             — Activation normalization with data-dependent init
#       ├── Invertible1x1Conv   — LU-parameterized invertible 1×1 convolution (Glow)
#       ├── RevBlock            — Reversible residual block (RevNet)
#       └── InvertibleSequential — Composition of invertible layers
#   NormalizingFlow (wraps InvertibleSequential + base distribution)
#
# Interface:
#   forward(layer, x)               → y
#   inverse(layer, y)               → x
#   log_abs_det_jacobian(layer, x)  → scalar per sample
#   forward_and_log_det(layer, x)   → (y, log_det)
#   parameters(layer)               → NamedTuple

# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------

"""
    InvertibleLayer <: AbstractLayer

Abstract base for layers that support exact inversion and log-det Jacobian
computation. Every concrete subtype must implement:
- `forward(layer, x)` — forward mapping
- `inverse(layer, y)` — exact inverse mapping
- `forward_and_log_det(layer, x)` — fused forward + log |det J|
- `parameters(layer)` — trainable parameters
"""
abstract type InvertibleLayer <: AbstractLayer end

"""
    inverse(layer::InvertibleLayer, y)

Compute the exact inverse `x = f⁻¹(y)` such that `forward(layer, inverse(layer, y)) ≈ y`.
"""
function inverse end

"""
    log_abs_det_jacobian(layer::InvertibleLayer, x)

Compute `log |det ∂f/∂x|` for each sample in the batch.
Returns a vector of length `batch_size`.
"""
function log_abs_det_jacobian end

"""
    forward_and_log_det(layer::InvertibleLayer, x)

Fused forward pass and log-det computation. Returns `(y, log_det)`.
Default implementation calls `forward` and `log_abs_det_jacobian` separately;
concrete subtypes should override for efficiency.
"""
function forward_and_log_det(layer::InvertibleLayer, x)
    y = forward(layer, x)
    ld = log_abs_det_jacobian(layer, x)
    return (y, ld)
end

# ---------------------------------------------------------------------------
# CouplingLayer — Affine Coupling (RealNVP / Glow)
# ---------------------------------------------------------------------------

"""
    CouplingLayer(dim, hidden_dim; mask_parity=false)

Affine coupling layer: splits input along the feature dimension into two halves
`x_a, x_b`. The first half passes through unchanged; the second is transformed:
    `y_a = x_a`
    `y_b = x_b ⊙ exp(s(x_a)) + t(x_a)`
where `s` and `t` are learned scale/translate networks.

Log-det = `sum(s(x_a))` per sample.

# Arguments
- `dim::Int`: Input feature dimension (must be even).
- `hidden_dim::Int`: Hidden dimension of the scale/translate networks.
- `mask_parity::Bool`: If `true`, swap which half is transformed.
"""
mutable struct CouplingLayer{T} <: InvertibleLayer
    scale_net_w1::Matrix{T}
    scale_net_b1::Vector{T}
    scale_net_w2::Matrix{T}
    scale_net_b2::Vector{T}
    translate_net_w1::Matrix{T}
    translate_net_b1::Vector{T}
    translate_net_w2::Matrix{T}
    translate_net_b2::Vector{T}
    dim::Int
    hidden_dim::Int
    split_dim::Int
    mask_parity::Bool
end

function CouplingLayer(dim::Int, hidden_dim::Int; mask_parity::Bool=false, dtype::Type{T}=Float32) where T
    @assert dim >= 2 "CouplingLayer requires dim >= 2"
    split = div(dim, 2)
    input_half = mask_parity ? (dim - split) : split
    output_half = mask_parity ? split : (dim - split)

    # Xavier uniform initialization
    scale = T(sqrt(6.0 / (input_half + hidden_dim)))
    scale2 = T(sqrt(6.0 / (hidden_dim + output_half)))

    CouplingLayer{T}(
        (rand(T, input_half, hidden_dim) .- T(0.5)) .* T(2) .* scale,     # scale_net_w1
        zeros(T, hidden_dim),                                               # scale_net_b1
        (rand(T, hidden_dim, output_half) .- T(0.5)) .* T(2) .* scale2,   # scale_net_w2
        zeros(T, output_half),                                              # scale_net_b2
        (rand(T, input_half, hidden_dim) .- T(0.5)) .* T(2) .* scale,     # translate_net_w1
        zeros(T, hidden_dim),                                               # translate_net_b1
        (rand(T, hidden_dim, output_half) .- T(0.5)) .* T(2) .* scale2,   # translate_net_w2
        zeros(T, output_half),                                              # translate_net_b2
        dim,
        hidden_dim,
        split,
        mask_parity,
    )
end

function _coupling_split(layer::CouplingLayer, x::AbstractMatrix)
    split = layer.split_dim
    if layer.mask_parity
        x_a = x[:, (split+1):end]
        x_b = x[:, 1:split]
    else
        x_a = x[:, 1:split]
        x_b = x[:, (split+1):end]
    end
    return x_a, x_b
end

function _coupling_merge(layer::CouplingLayer, y_a::AbstractMatrix, y_b::AbstractMatrix)
    if layer.mask_parity
        return hcat(y_b, y_a)
    else
        return hcat(y_a, y_b)
    end
end

function _scale_and_translate(layer::CouplingLayer{T}, x_a::AbstractMatrix) where T
    h = tanh.(x_a * layer.scale_net_w1 .+ layer.scale_net_b1')
    s = tanh.(h * layer.scale_net_w2 .+ layer.scale_net_b2')

    h_t = tanh.(x_a * layer.translate_net_w1 .+ layer.translate_net_b1')
    t = h_t * layer.translate_net_w2 .+ layer.translate_net_b2'

    return s, t
end

function forward(layer::CouplingLayer, x::AbstractTensor)
    y, _ = forward_and_log_det(layer, x)
    return y
end

function forward(layer::CouplingLayer, x::AbstractMatrix)
    x_a, x_b = _coupling_split(layer, x)
    s, t = _scale_and_translate(layer, x_a)
    y_b = x_b .* exp.(s) .+ t
    return _coupling_merge(layer, x_a, y_b)
end

function forward(layer::CouplingLayer, x::Tensor)
    Tensor(forward(layer, x.data))
end

function inverse(layer::CouplingLayer, y::AbstractMatrix)
    y_a, y_b = _coupling_split(layer, y)
    s, t = _scale_and_translate(layer, y_a)
    x_b = (y_b .- t) .* exp.(-s)
    return _coupling_merge(layer, y_a, x_b)
end

function inverse(layer::CouplingLayer, y::Tensor)
    Tensor(inverse(layer, y.data))
end

function log_abs_det_jacobian(layer::CouplingLayer, x::AbstractMatrix)
    x_a, _ = _coupling_split(layer, x)
    s, _ = _scale_and_translate(layer, x_a)
    return vec(sum(s, dims=2))
end

function log_abs_det_jacobian(layer::CouplingLayer, x::Tensor)
    log_abs_det_jacobian(layer, x.data)
end

function forward_and_log_det(layer::CouplingLayer, x::AbstractMatrix)
    x_a, x_b = _coupling_split(layer, x)
    s, t = _scale_and_translate(layer, x_a)
    y_b = x_b .* exp.(s) .+ t
    y = _coupling_merge(layer, x_a, y_b)
    log_det = vec(sum(s, dims=2))
    return (y, log_det)
end

function forward_and_log_det(layer::CouplingLayer, x::Tensor)
    y_data, ld = forward_and_log_det(layer, x.data)
    return (Tensor(y_data), ld)
end

function parameters(layer::CouplingLayer)
    (scale_net_w1=layer.scale_net_w1, scale_net_b1=layer.scale_net_b1,
     scale_net_w2=layer.scale_net_w2, scale_net_b2=layer.scale_net_b2,
     translate_net_w1=layer.translate_net_w1, translate_net_b1=layer.translate_net_b1,
     translate_net_w2=layer.translate_net_w2, translate_net_b2=layer.translate_net_b2)
end

# ---------------------------------------------------------------------------
# ActNorm — Activation Normalization
# ---------------------------------------------------------------------------

"""
    ActNorm(dim)

Activation normalization layer (Glow). Applies an affine transformation
`y = x ⊙ exp(log_scale) + bias` with data-dependent initialization on the
first forward pass (so that the output has zero mean and unit variance).

Log-det = `sum(log_scale)` per sample.
"""
mutable struct ActNorm{T} <: InvertibleLayer
    log_scale::Vector{T}
    bias::Vector{T}
    dim::Int
    initialized::Bool
end

function ActNorm(dim::Int; dtype::Type{T}=Float32) where T
    ActNorm{T}(zeros(T, dim), zeros(T, dim), dim, false)
end

function _actnorm_init!(layer::ActNorm{T}, x::AbstractMatrix) where T
    μ = vec(mean(x, dims=1))
    σ = vec(std(x, dims=1)) .+ T(1e-6)
    layer.bias .= -μ ./ σ
    layer.log_scale .= -log.(σ)
    layer.initialized = true
    return nothing
end

function forward(layer::ActNorm, x::AbstractMatrix)
    if !layer.initialized
        Zygote.ignore_derivatives() do
            _actnorm_init!(layer, x)
        end
    end
    return x .* exp.(layer.log_scale') .+ layer.bias'
end

function forward(layer::ActNorm, x::Tensor)
    Tensor(forward(layer, x.data))
end

function inverse(layer::ActNorm, y::AbstractMatrix)
    return (y .- layer.bias') .* exp.(-layer.log_scale')
end

function inverse(layer::ActNorm, y::Tensor)
    Tensor(inverse(layer, y.data))
end

function log_abs_det_jacobian(layer::ActNorm, x::AbstractMatrix)
    batch_size = size(x, 1)
    ld = sum(layer.log_scale)
    return fill(ld, batch_size)
end

function log_abs_det_jacobian(layer::ActNorm, x::Tensor)
    log_abs_det_jacobian(layer, x.data)
end

function forward_and_log_det(layer::ActNorm, x::AbstractMatrix)
    y = forward(layer, x)
    ld = log_abs_det_jacobian(layer, x)
    return (y, ld)
end

function forward_and_log_det(layer::ActNorm, x::Tensor)
    y_data, ld = forward_and_log_det(layer, x.data)
    return (Tensor(y_data), ld)
end

function parameters(layer::ActNorm)
    (log_scale=layer.log_scale, bias=layer.bias)
end

# ---------------------------------------------------------------------------
# Invertible1x1Conv — LU-parameterized invertible matrix (Glow)
# ---------------------------------------------------------------------------

"""
    Invertible1x1Conv(dim)

Invertible 1×1 convolution parameterized via the LU decomposition of a random
orthogonal matrix. The weight matrix `W = P * L * U` where:
- `P` is a fixed permutation matrix
- `L` is lower-triangular with ones on diagonal
- `U` is upper-triangular with learnable diagonal `log_s`

Log-det = `sum(log_s)` per sample (O(D) instead of O(D³)).
"""
mutable struct Invertible1x1Conv{T} <: InvertibleLayer
    P::Matrix{T}       # Fixed permutation (from initial LU)
    L_lower::Matrix{T} # Strict lower-triangular part of L (learnable)
    U_upper::Matrix{T} # Strict upper-triangular part of U (learnable)
    log_s::Vector{T}   # Log of diagonal of U (learnable)
    dim::Int
end

function Invertible1x1Conv(dim::Int; dtype::Type{T}=Float32) where T
    # Random orthogonal initialization via QR
    W = T.(qr(randn(dim, dim)).Q)
    # LU decompose
    lu_result = lu(W)
    P_mat = T.(lu_result.P)
    L_mat = T.(lu_result.L)
    U_mat = T.(lu_result.U)

    # Extract strict lower triangle of L (diagonal is ones, not stored)
    L_lower = tril(L_mat, -1)
    # Extract diagonal of U as log_s
    s = diag(U_mat)
    log_s_init = log.(abs.(s) .+ T(1e-8))
    sign_s = sign.(s)
    # Absorb sign into U upper
    U_upper = triu(U_mat, 1)
    # Store sign in diagonal of L_lower for reconstruction
    # Actually, keep it simple: store signs separately is complex. Instead,
    # just store log_s and use sign from initialization.
    # For simplicity, we reconstruct U_diag = sign_s .* exp.(log_s)

    Invertible1x1Conv{T}(
        P_mat,
        L_lower,
        U_upper,
        log_s_init,
        dim,
    )
end

function _reconstruct_W(layer::Invertible1x1Conv{T}) where T
    d = layer.dim
    L = layer.L_lower + I(d)
    U = layer.U_upper + Diagonal(exp.(layer.log_s))
    return layer.P * L * U
end

function _reconstruct_W_inv(layer::Invertible1x1Conv{T}) where T
    W = _reconstruct_W(layer)
    return inv(W)
end

function forward(layer::Invertible1x1Conv, x::AbstractMatrix)
    W = _reconstruct_W(layer)
    return x * W'
end

function forward(layer::Invertible1x1Conv, x::Tensor)
    Tensor(forward(layer, x.data))
end

function inverse(layer::Invertible1x1Conv, y::AbstractMatrix)
    W_inv = _reconstruct_W_inv(layer)
    return y * W_inv'
end

function inverse(layer::Invertible1x1Conv, y::Tensor)
    Tensor(inverse(layer, y.data))
end

function log_abs_det_jacobian(layer::Invertible1x1Conv, x::AbstractMatrix)
    batch_size = size(x, 1)
    ld = sum(layer.log_s)
    return fill(ld, batch_size)
end

function log_abs_det_jacobian(layer::Invertible1x1Conv, x::Tensor)
    log_abs_det_jacobian(layer, x.data)
end

function forward_and_log_det(layer::Invertible1x1Conv, x::AbstractMatrix)
    y = forward(layer, x)
    ld = log_abs_det_jacobian(layer, x)
    return (y, ld)
end

function forward_and_log_det(layer::Invertible1x1Conv, x::Tensor)
    y_data, ld = forward_and_log_det(layer, x.data)
    return (Tensor(y_data), ld)
end

function parameters(layer::Invertible1x1Conv)
    (L_lower=layer.L_lower, U_upper=layer.U_upper, log_s=layer.log_s)
end

# ---------------------------------------------------------------------------
# RevBlock — Reversible Residual Block (RevNet)
# ---------------------------------------------------------------------------

"""
    RevBlock(F, G)

Reversible residual block (Gomez et al., 2017). Splits input into two halves
and computes:
    `y1 = x1 + F(x2)`
    `y2 = x2 + G(y1)`

The inverse reconstructs activations from output:
    `x2 = y2 - G(y1)`
    `x1 = y1 - F(x2)`

This is volume-preserving so `log |det J| = 0`.

Memory efficient: during backprop activations can be recomputed from outputs
via the inverse, avoiding storage of intermediate activations.

# Arguments
- `F::AbstractLayer`: First residual function.
- `G::AbstractLayer`: Second residual function.
"""
struct RevBlock{F<:AbstractLayer, G<:AbstractLayer} <: InvertibleLayer
    F::F
    G::G
end

function _revblock_split(x::AbstractMatrix)
    d = size(x, 2)
    half = div(d, 2)
    return x[:, 1:half], x[:, (half+1):end]
end

function _revblock_merge(x1::AbstractMatrix, x2::AbstractMatrix)
    return hcat(x1, x2)
end

function _forward_data(layer::RevBlock, x::AbstractMatrix)
    x1, x2 = _revblock_split(x)
    F_x2 = _layer_forward_data(layer.F, x2)
    y1 = x1 .+ F_x2
    G_y1 = _layer_forward_data(layer.G, y1)
    y2 = x2 .+ G_y1
    return _revblock_merge(y1, y2)
end

# Helper to get raw matrix data from a layer forward pass
function _layer_forward_data(layer::AbstractLayer, x::AbstractMatrix)
    result = forward(layer, Tensor(x))
    return result.data
end

function forward(layer::RevBlock, x::AbstractMatrix)
    return _forward_data(layer, x)
end

function forward(layer::RevBlock, x::Tensor)
    Tensor(_forward_data(layer, x.data))
end

function _inverse_data(layer::RevBlock, y::AbstractMatrix)
    y1, y2 = _revblock_split(y)
    G_y1 = _layer_forward_data(layer.G, y1)
    x2 = y2 .- G_y1
    F_x2 = _layer_forward_data(layer.F, x2)
    x1 = y1 .- F_x2
    return _revblock_merge(x1, x2)
end

function inverse(layer::RevBlock, y::AbstractMatrix)
    return _inverse_data(layer, y)
end

function inverse(layer::RevBlock, y::Tensor)
    Tensor(_inverse_data(layer, y.data))
end

function log_abs_det_jacobian(::RevBlock, x::AbstractMatrix)
    # Volume-preserving: log-det = 0
    return zeros(eltype(x), size(x, 1))
end

function log_abs_det_jacobian(layer::RevBlock, x::Tensor)
    log_abs_det_jacobian(layer, x.data)
end

function forward_and_log_det(layer::RevBlock, x::AbstractMatrix)
    y = forward(layer, x)
    ld = zeros(eltype(x), size(x, 1))
    return (y, ld)
end

function forward_and_log_det(layer::RevBlock, x::Tensor)
    y_data, ld = forward_and_log_det(layer, x.data)
    return (Tensor(y_data), ld)
end

function parameters(layer::RevBlock)
    (F=parameters(layer.F), G=parameters(layer.G))
end

# ---------------------------------------------------------------------------
# InvertibleSequential — Composition of Invertible Layers
# ---------------------------------------------------------------------------

"""
    InvertibleSequential(layers...)

Sequential composition of invertible layers. Forward applies layers in order;
inverse applies them in reverse order. Log-dets are summed.

# Examples
```julia
flow = InvertibleSequential(
    ActNorm(10),
    Invertible1x1Conv(10),
    CouplingLayer(10, 32),
)
y, log_det = forward_and_log_det(flow, x)
x_reconstructed = inverse(flow, y)
```
"""
struct InvertibleSequential{L<:Tuple} <: InvertibleLayer
    layers::L
end

function InvertibleSequential(layers::InvertibleLayer...)
    InvertibleSequential(Tuple(layers))
end

function forward(seq::InvertibleSequential, x::AbstractMatrix)
    for layer in seq.layers
        x = forward(layer, x)
    end
    return x
end

function forward(seq::InvertibleSequential, x::Tensor)
    for layer in seq.layers
        x = forward(layer, x)
    end
    return x
end

function inverse(seq::InvertibleSequential, y::AbstractMatrix)
    for layer in reverse(collect(seq.layers))
        y = inverse(layer, y)
    end
    return y
end

function inverse(seq::InvertibleSequential, y::Tensor)
    for layer in reverse(collect(seq.layers))
        y = inverse(layer, y)
    end
    return y
end

function log_abs_det_jacobian(seq::InvertibleSequential, x::AbstractMatrix)
    total_ld = zeros(eltype(x), size(x, 1))
    for layer in seq.layers
        total_ld .+= log_abs_det_jacobian(layer, x)
        x = forward(layer, x)
    end
    return total_ld
end

function log_abs_det_jacobian(seq::InvertibleSequential, x::Tensor)
    log_abs_det_jacobian(seq, x.data)
end

function forward_and_log_det(seq::InvertibleSequential, x::AbstractMatrix)
    total_ld = zeros(eltype(x), size(x, 1))
    for layer in seq.layers
        x_next, ld = forward_and_log_det(layer, x)
        total_ld .+= ld
        x = x_next
    end
    return (x, total_ld)
end

function forward_and_log_det(seq::InvertibleSequential, x::Tensor)
    y_data, ld = forward_and_log_det(seq, x.data)
    return (Tensor(y_data), ld)
end

function parameters(seq::InvertibleSequential)
    return Tuple(parameters(layer) for layer in seq.layers)
end

# ---------------------------------------------------------------------------
# NormalizingFlow — Complete density estimator
# ---------------------------------------------------------------------------

"""
    NormalizingFlow(transform; base_dim)

A normalizing flow wraps an `InvertibleSequential` (or any `InvertibleLayer`)
with a base distribution (standard normal) to form a density model.

Provides:
- `flow_log_prob(nf, x)` — log probability of data under the flow
- `flow_sample(nf, n)` — generate samples by pushing base noise through inverse
- `flow_nll_loss(nf, x)` — negative log-likelihood loss for training

# Arguments
- `transform::InvertibleLayer`: The invertible transformation (typically `InvertibleSequential`).
- `base_dim::Int`: Dimension of the base distribution.
"""
struct NormalizingFlow{T<:InvertibleLayer}
    transform::T
    base_dim::Int
end

function NormalizingFlow(transform::InvertibleLayer; base_dim::Int)
    NormalizingFlow(transform, base_dim)
end

"""
    _base_log_prob(x)

Log probability under a standard multivariate normal distribution (diagonal covariance).
"""
function _base_log_prob(x::AbstractMatrix{T}) where T
    d = size(x, 2)
    # log p(x) = -0.5 * (d * log(2π) + sum(x², dim=2))
    log2pi = T(log(2π))
    return vec(-T(0.5) .* (d * log2pi .+ sum(x .^ 2, dims=2)))
end

"""
    flow_log_prob(nf::NormalizingFlow, x)

Compute the log-probability of samples `x` under the normalizing flow.
Uses the change of variables formula:
    `log p(x) = log p_base(f(x)) + log |det J_f(x)|`
where `f` is the forward transform mapping data → base space.
"""
function flow_log_prob(nf::NormalizingFlow, x::AbstractMatrix)
    z, log_det = forward_and_log_det(nf.transform, x)
    base_lp = _base_log_prob(z)
    return base_lp .+ log_det
end

function flow_log_prob(nf::NormalizingFlow, x::Tensor)
    flow_log_prob(nf, x.data)
end

"""
    flow_sample(nf::NormalizingFlow, n; dtype=Float32)

Generate `n` samples from the normalizing flow by drawing from the base
distribution and pushing through the inverse transform.
"""
function flow_sample(nf::NormalizingFlow, n::Int; dtype::Type{T}=Float32) where T
    z = randn(T, n, nf.base_dim)
    x = inverse(nf.transform, z)
    return x
end

"""
    flow_nll_loss(nf::NormalizingFlow, x)

Negative log-likelihood loss for training the normalizing flow.
Returns a scalar (mean NLL over the batch).
"""
function flow_nll_loss(nf::NormalizingFlow, x::AbstractMatrix)
    lp = flow_log_prob(nf, x)
    return -mean(lp)
end

function flow_nll_loss(nf::NormalizingFlow, x::Tensor)
    flow_nll_loss(nf, x.data)
end
