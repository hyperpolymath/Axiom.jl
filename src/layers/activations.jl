# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Activation Functions
#
# All activations are implemented as both functions and callable layers.

# ============================================================================
# Activation Functions
# ============================================================================

"""
    relu(x)

Rectified Linear Unit: max(0, x)

# Properties (provable)
- ∀x. relu(x) >= 0
- ∀x. relu(x) <= max(0, x)
"""
relu(x::AbstractTensor) = Tensor(max.(zero(eltype(x.data)), x.data))
relu(x::AbstractArray) = max.(zero(eltype(x)), x)
relu(x::Number) = max(zero(x), x)

"""
    leaky_relu(x, α=0.01)

Leaky ReLU: x if x > 0, else α*x
"""
leaky_relu(x::AbstractTensor, α=0.01f0) = Tensor(max.(α .* x.data, x.data))
leaky_relu(x::AbstractArray, α=0.01f0) = max.(α .* x, x)
leaky_relu(x::Number, α=0.01f0) = x > 0 ? x : α * x

"""
    elu(x, α=1.0)

Exponential Linear Unit: x if x > 0, else α*(exp(x) - 1)
"""
elu(x::AbstractTensor, α=1.0f0) = Tensor(ifelse.(x.data .> 0, x.data, α .* (exp.(x.data) .- 1)))
elu(x::AbstractArray, α=1.0f0) = ifelse.(x .> 0, x, α .* (exp.(x) .- 1))
elu(x::Number, α=1.0f0) = x > 0 ? x : α * (exp(x) - 1)

"""
    selu(x)

Scaled ELU for self-normalizing networks.
"""
const SELU_ALPHA = 1.6732632423543772f0
const SELU_SCALE = 1.0507009873554805f0

selu(x::AbstractTensor) = Tensor(SELU_SCALE .* ifelse.(x.data .> 0, x.data, SELU_ALPHA .* (exp.(x.data) .- 1)))
selu(x::AbstractArray) = SELU_SCALE .* ifelse.(x .> 0, x, SELU_ALPHA .* (exp.(x) .- 1))

"""
    gelu(x)

Gaussian Error Linear Unit (used in Transformers).
"""
function gelu(x::AbstractTensor)
    # Approximation: 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    c = sqrt(2.0f0 / π)
    Tensor(0.5f0 .* x.data .* (1 .+ tanh.(c .* (x.data .+ 0.044715f0 .* x.data .^ 3))))
end

function gelu(x::AbstractArray)
    # Approximation: 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    c = sqrt(2.0f0 / π)
    0.5f0 .* x .* (1 .+ tanh.(c .* (x .+ 0.044715f0 .* x .^ 3)))
end

"""
    sigmoid(x)

Sigmoid activation: 1 / (1 + exp(-x))

# Properties (provable)
- ∀x. 0 <= sigmoid(x) <= 1
"""
sigmoid(x::AbstractTensor) = Tensor(1 ./ (1 .+ exp.(-x.data)))
sigmoid(x::AbstractArray) = 1 ./ (1 .+ exp.(-x))
sigmoid(x::Number) = 1 / (1 + exp(-x))

"""
    tanh(x)

Hyperbolic tangent (re-exported from Base).

# Properties (provable)
- ∀x. -1 <= tanh(x) <= 1
"""
# tanh is already in Base

"""
    softmax(x; dims=ndims(x))

Softmax activation (normalized exponential).

# Properties (provable)
- ∀x. sum(softmax(x)) ≈ 1.0
- ∀x. all(softmax(x) .>= 0)
- ∀x. all(softmax(x) .<= 1)
"""
function softmax(x::AbstractTensor; dims=ndims(x.data))
    # Numerically stable softmax
    x_max = maximum(x.data, dims=dims)
    exp_x = exp.(x.data .- x_max)
    Tensor(exp_x ./ sum(exp_x, dims=dims))
end

function softmax(x::AbstractArray; dims=ndims(x))
    # Numerically stable softmax
    x_max = maximum(x, dims=dims)
    exp_x = exp.(x .- x_max)
    exp_x ./ sum(exp_x, dims=dims)
end

"""
    log_softmax(x; dims=ndims(x))

Log of softmax (more numerically stable than log(softmax(x))).
"""
function log_softmax(x::AbstractTensor; dims=ndims(x.data))
    x_max = maximum(x.data, dims=dims)
    shifted = x.data .- x_max
    Tensor(shifted .- log.(sum(exp.(shifted), dims=dims)))
end

function log_softmax(x::AbstractArray; dims=ndims(x))
    x_max = maximum(x, dims=dims)
    shifted = x .- x_max
    shifted .- log.(sum(exp.(shifted), dims=dims))
end

"""
    softplus(x)

Softplus: log(1 + exp(x)), smooth approximation to ReLU.
"""
softplus(x::AbstractTensor) = Tensor(log1p.(exp.(x.data)))
softplus(x::AbstractArray) = log1p.(exp.(x))

"""
    softsign(x)

Softsign: x / (1 + |x|)
"""
softsign(x::AbstractTensor) = Tensor(x.data ./ (1 .+ abs.(x.data)))
softsign(x::AbstractArray) = x ./ (1 .+ abs.(x))

"""
    swish(x)

Swish activation: x * sigmoid(x)
Also known as SiLU (Sigmoid Linear Unit).
"""
swish(x::AbstractTensor) = Tensor(x.data .* sigmoid(x.data))
swish(x::AbstractArray) = x .* sigmoid(x)

# Alias
const silu = swish

"""
    mish(x)

Mish activation: x * tanh(softplus(x))
"""
mish(x::AbstractTensor) = Tensor(x.data .* tanh.(softplus(x.data)))
mish(x::AbstractArray) = x .* tanh.(softplus(x))

"""
    hardswish(x)

Hard Swish (efficient approximation to Swish).
"""
function hardswish(x::AbstractTensor)
    Tensor(x.data .* relu6.(x.data .+ 3) ./ 6)
end

function hardswish(x::AbstractArray)
    x .* relu6.(x .+ 3) ./ 6
end

"""
    relu6(x)

ReLU capped at 6: min(max(0, x), 6)
"""
relu6(x::AbstractTensor) = Tensor(min.(max.(zero(eltype(x.data)), x.data), 6))
relu6(x::AbstractArray) = min.(max.(zero(eltype(x)), x), 6)
relu6(x::Number) = min(max(zero(x), x), oftype(x, 6))

"""
    hardsigmoid(x)

Hard Sigmoid (efficient approximation to Sigmoid).
"""
hardsigmoid(x::AbstractTensor) = Tensor(max.(zero(eltype(x.data)), min.(one(eltype(x.data)), (x.data .+ 3) ./ 6)))
hardsigmoid(x::AbstractArray) = max.(zero(eltype(x)), min.(one(eltype(x)), (x .+ 3) ./ 6))

# ============================================================================
# Activation Layers (for use in pipelines)
# ============================================================================

"""
    ReLU()

ReLU activation layer.
"""
struct ReLU <: StatelessLayer end
forward(::ReLU, x) = relu(x)
output_shape(::ReLU, input_shape) = input_shape

"""
    LeakyReLU(α=0.01)

Leaky ReLU activation layer.
"""
struct LeakyReLU <: StatelessLayer
    α::Float32
end
LeakyReLU() = LeakyReLU(0.01f0)
forward(l::LeakyReLU, x) = leaky_relu(x, l.α)
output_shape(::LeakyReLU, input_shape) = input_shape

"""
    ELU(α=1.0)

ELU activation layer.
"""
struct ELU <: StatelessLayer
    α::Float32
end
ELU() = ELU(1.0f0)
forward(l::ELU, x) = elu(x, l.α)
output_shape(::ELU, input_shape) = input_shape

"""
    SELU()

SELU activation layer.
"""
struct SELU <: StatelessLayer end
forward(::SELU, x) = selu(x)
output_shape(::SELU, input_shape) = input_shape

"""
    GELU()

GELU activation layer.
"""
struct GELU <: StatelessLayer end
forward(::GELU, x) = gelu(x)
output_shape(::GELU, input_shape) = input_shape

"""
    Sigmoid()

Sigmoid activation layer.
"""
struct Sigmoid <: StatelessLayer end
forward(::Sigmoid, x) = sigmoid(x)
output_shape(::Sigmoid, input_shape) = input_shape

"""
    Tanh()

Tanh activation layer.
"""
struct Tanh <: StatelessLayer end
forward(::Tanh, x) = tanh.(x)
output_shape(::Tanh, input_shape) = input_shape

"""
    Softmax(; dims=-1)

Softmax activation layer.
"""
struct Softmax <: StatelessLayer
    dims::Int
end
Softmax(; dims=-1) = Softmax(dims)


function forward(s::Softmax, x)
    d = s.dims == -1 ? ndims(x) : s.dims
    softmax(x, dims=d)
end
output_shape(::Softmax, input_shape) = input_shape

"""
    LogSoftmax(; dims=-1)

Log Softmax activation layer.
"""
struct LogSoftmax <: StatelessLayer
    dims::Int
end
LogSoftmax(; dims=-1) = LogSoftmax(dims)


function forward(s::LogSoftmax, x)
    d = s.dims == -1 ? ndims(x) : s.dims
    log_softmax(x, dims=d)
end
output_shape(::LogSoftmax, input_shape) = input_shape

"""
    Swish()

Swish activation layer.
"""
struct Swish <: StatelessLayer end
forward(::Swish, x) = swish(x)
output_shape(::Swish, input_shape) = input_shape

"""
    Mish()

Mish activation layer.
"""
struct Mish <: StatelessLayer end
forward(::Mish, x) = mish(x)
output_shape(::Mish, input_shape) = input_shape

# ============================================================================
# PReLU (Parametric ReLU) - has learnable parameters
# ============================================================================

"""
    PReLU(num_parameters=1)

Parametric ReLU with learnable slope.
"""
mutable struct PReLU{T} <: AbstractLayer
    α::Vector{T}
end

function PReLU(num_parameters::Int=1; dtype::Type{T}=Float32) where T
    PReLU{T}(fill(T(0.25), num_parameters))
end

function forward(p::PReLU, x::AbstractTensor)
    # Broadcast α appropriately
    x_data = x.data
    pos = max.(zero(eltype(x_data)), x_data)
    neg = p.α .* min.(zero(eltype(x_data)), x_data)
    Tensor(pos .+ neg)
end

function forward(p::PReLU, x::AbstractArray)
    # Broadcast α appropriately
    pos = max.(zero(eltype(x)), x)
    neg = p.α .* min.(zero(eltype(x)), x)
    pos .+ neg
end

parameters(p::PReLU) = (α = p.α,)
output_shape(::PReLU, input_shape) = input_shape
