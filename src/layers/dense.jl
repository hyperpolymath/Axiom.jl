# Axiom.jl Dense (Fully Connected) Layer
#
# Linear transformation: y = Wx + b

"""
    Dense(in_features, out_features; activation=identity, bias=true, init=GlorotUniform())

Fully connected layer.

# Arguments
- `in_features`: Number of input features
- `out_features`: Number of output features
- `activation`: Activation function (default: identity)
- `bias`: Whether to include bias term (default: true)
- `init`: Weight initializer (default: GlorotUniform)

# Examples
```julia
dense = Dense(784, 128)           # Linear layer
dense = Dense(128, 64, relu)      # With ReLU activation
dense = Dense(64, 10, bias=false) # No bias
```
"""
mutable struct Dense{T, F} <: AbstractLayer
    weight::Matrix{T}
    bias::Union{Vector{T}, Nothing}
    activation::F
    in_features::Int
    out_features::Int
end

function Dense(
    in_features::Int,
    out_features::Int,
    activation::F = identity;
    bias::Bool = true,
    init::AbstractInitializer = DEFAULT_WEIGHT_INIT,
    bias_init::AbstractInitializer = DEFAULT_BIAS_INIT,
    dtype::Type{T} = Float32
) where {T, F}
    weight = T.(init(in_features, out_features))
    b = bias ? T.(bias_init(out_features)) : nothing

    Dense{T, F}(weight, b, activation, in_features, out_features)
end

function forward(d::Dense, x::AbstractArray)
    # x: (batch, in_features) or (in_features,)
    # output: (batch, out_features) or (out_features,)

    if ndims(x) == 1
        y = d.weight' * x
    else
        y = x * d.weight
    end

    if d.bias !== nothing
        y = y .+ d.bias'
    end

    d.activation(y)
end

function parameters(d::Dense)
    if d.bias !== nothing
        (weight = d.weight, bias = d.bias)
    else
        (weight = d.weight,)
    end
end

function output_shape(d::Dense, input_shape)
    if length(input_shape) == 1
        (d.out_features,)
    else
        (input_shape[1], d.out_features)
    end
end

function verify_input_shape(d::Dense, x)
    in_dim = ndims(x) == 1 ? length(x) : size(x, 2)
    if in_dim != d.in_features
        throw(DimensionMismatch(
            "Dense layer expects input with $(d.in_features) features, got $in_dim"
        ))
    end
    true
end

function show_layer_params(io::IO, d::Dense)
    print(io, "$(d.in_features) => $(d.out_features)")
    if d.activation !== identity
        print(io, ", $(d.activation)")
    end
    if d.bias === nothing
        print(io, ", bias=false")
    end
end

# Alias for compatibility
const Linear = Dense

"""
    Bilinear(in1_features, in2_features, out_features)

Bilinear layer: y = x1' * W * x2 + b
"""
mutable struct Bilinear{T, F} <: AbstractLayer
    weight::Array{T, 3}
    bias::Union{Vector{T}, Nothing}
    activation::F
    in1_features::Int
    in2_features::Int
    out_features::Int
end

function Bilinear(
    in1_features::Int,
    in2_features::Int,
    out_features::Int,
    activation::F = identity;
    bias::Bool = true,
    dtype::Type{T} = Float32
) where {T, F}
    weight = randn(T, in1_features, in2_features, out_features) .* sqrt(2.0f0 / (in1_features + in2_features))
    b = bias ? zeros(T, out_features) : nothing

    Bilinear{T, F}(weight, b, activation, in1_features, in2_features, out_features)
end

function forward(bl::Bilinear, x1::AbstractArray, x2::AbstractArray)
    # Batch bilinear operation
    batch_size = size(x1, 1)
    out = zeros(eltype(x1), batch_size, bl.out_features)

    for i in 1:bl.out_features
        out[:, i] = sum(x1 .* (x2 * bl.weight[:, :, i]'), dims=2)
    end

    if bl.bias !== nothing
        out = out .+ bl.bias'
    end

    bl.activation(out)
end

function parameters(bl::Bilinear)
    if bl.bias !== nothing
        (weight = bl.weight, bias = bl.bias)
    else
        (weight = bl.weight,)
    end
end
