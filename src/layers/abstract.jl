# Axiom.jl Abstract Layer Types
#
# Base types and interfaces for all layers.

"""
    AbstractLayer

Base type for all neural network layers in Axiom.jl.
All layers must implement:
- `forward(layer, x)` - forward pass
- `parameters(layer)` - return trainable parameters
- `output_shape(layer, input_shape)` - compute output shape
"""
abstract type AbstractLayer end

# Required interface
"""
    forward(layer, x)

Compute forward pass of layer.
"""
function forward end

"""
    parameters(layer)

Return NamedTuple of trainable parameters.
"""
function parameters(layer::AbstractLayer)
    NamedTuple()
end

"""
    output_shape(layer, input_shape)

Compute output shape given input shape.
"""
function output_shape end

"""
    verify_input_shape(layer, x)

Verify that input shape is compatible with layer.
"""
function verify_input_shape(layer::AbstractLayer, x)
    # Default: no verification (override in specific layers)
    true
end

# Layer state
"""
    trainable(layer) -> Bool

Check if layer has trainable parameters.
"""
trainable(layer::AbstractLayer) = !isempty(parameters(layer))

"""
    set_training!(layer, mode::Bool)

Set layer to training or evaluation mode.
"""
function set_training!(layer::AbstractLayer, mode::Bool)
    if hasfield(typeof(layer), :training)
        layer.training = mode
    end
end

# Parameter access
"""
    nparams(layer) -> Int

Count total number of parameters.
"""
function nparams(layer::AbstractLayer)
    sum(length, values(parameters(layer)), init=0)
end

# Layer composition
"""
    (layer::AbstractLayer)(x)

Make layers callable.
"""
(layer::AbstractLayer)(x) = forward(layer, x)

# Pretty printing
function Base.show(io::IO, layer::AbstractLayer)
    print(io, "$(typeof(layer).name.name)(")
    show_layer_params(io, layer)
    print(io, ")")
end

function show_layer_params(io::IO, layer::AbstractLayer)
    # Override in specific layers
end

# Weight initialization schemes
"""
    AbstractInitializer

Base type for weight initializers.
"""
abstract type AbstractInitializer end

"""
    GlorotUniform()

Glorot uniform initialization (Xavier).
"""
struct GlorotUniform <: AbstractInitializer end

function (init::GlorotUniform)(dims...)
    fan_in = dims[1]
    fan_out = length(dims) > 1 ? dims[2] : dims[1]
    limit = sqrt(6.0f0 / (fan_in + fan_out))
    (rand(Float32, dims...) .- 0.5f0) .* 2 .* limit
end

"""
    GlorotNormal()

Glorot normal initialization (Xavier).
"""
struct GlorotNormal <: AbstractInitializer end

function (init::GlorotNormal)(dims...)
    fan_in = dims[1]
    fan_out = length(dims) > 1 ? dims[2] : dims[1]
    std = sqrt(2.0f0 / (fan_in + fan_out))
    randn(Float32, dims...) .* std
end

"""
    HeUniform()

He uniform initialization (for ReLU).
"""
struct HeUniform <: AbstractInitializer end

function (init::HeUniform)(dims...)
    fan_in = dims[1]
    limit = sqrt(6.0f0 / fan_in)
    (rand(Float32, dims...) .- 0.5f0) .* 2 .* limit
end

"""
    HeNormal()

He normal initialization (for ReLU).
"""
struct HeNormal <: AbstractInitializer end

function (init::HeNormal)(dims...)
    fan_in = dims[1]
    std = sqrt(2.0f0 / fan_in)
    randn(Float32, dims...) .* std
end

"""
    Zeros()

Zero initialization.
"""
struct Zeros <: AbstractInitializer end

(init::Zeros)(dims...) = zeros(Float32, dims...)

"""
    Ones()

One initialization.
"""
struct Ones <: AbstractInitializer end

(init::Ones)(dims...) = ones(Float32, dims...)

# Default initializers
const DEFAULT_WEIGHT_INIT = GlorotUniform()
const DEFAULT_BIAS_INIT = Zeros()

# Stateless layer marker
"""
    StatelessLayer

Marker type for layers without trainable parameters.
"""
abstract type StatelessLayer <: AbstractLayer end

parameters(::StatelessLayer) = NamedTuple()
