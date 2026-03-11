# SPDX-License-Identifier: PMPL-1.0-or-later
#
# Axiom.jl Dense and Bilinear Layer Implementations
# ===============================================
#
# This file provides the implementations for fundamental neural network layers:
# the `Dense` (fully connected) layer and the `Bilinear` layer. These layers
# are crucial building blocks for constructing a wide variety of neural network
# architectures, from simple multi-layer perceptrons to more complex models
# that require interaction between different feature sets.
#
# Dense Layer (`Dense`):
# --------------------
# The `Dense` layer (also known as a fully connected layer or affine layer)
# performs a linear transformation on its input data, followed by an optional
# activation function. Its mathematical operation is typically represented as:
# `y = f(Wx + b)`
# where `x` is the input vector, `W` is the weight matrix, `b` is the bias vector,
# and `f` is the activation function. It connects every input neuron to every
# output neuron.
#
# Bilinear Layer (`Bilinear`):
# ---------------------------
# The `Bilinear` layer is designed to capture pairwise interactions between
# two different input feature sets. Its operation is more complex than a
# simple dense layer, often formulated as:
# `y_k = x1' * W_k * x2 + b_k`
# where `x1` and `x2` are input vectors, `W_k` is a tensor representing the
# interaction weights for the k-th output feature, and `b_k` is the bias.
# This type of layer is useful in models requiring attention mechanisms,
# multimodal fusion, or collaborative filtering.
#
# Both layers inherit from `AbstractLayer` and adhere to its interface,
# providing implementations for `forward`, `parameters`, `output_shape`, etc.
#

"""
    Dense(in_features, out_features; activation=identity, bias=true, init=GlorotUniform(), bias_init=Zeros(), dtype=Float32)

A `Dense` layer (also known as a fully connected layer or affine transformation)
implements the linear operation `y = Wx + b`, optionally followed by an
activation function `f`. Every input feature is connected to every output feature.

# Fields
- `weight::Matrix{T}`: The weight matrix of size `(in_features, out_features)`.
- `bias::Union{Vector{T}, Nothing}`: The bias vector of size `(out_features,)`. If `bias` is `false` during construction, this field will be `nothing`.
- `activation::F`: The activation function `f` applied to the linear output (e.g., `relu`, `sigmoid`, `identity`).
- `in_features::Int`: The number of input features expected by the layer.
- `out_features::Int`: The number of output features produced by the layer.

# Examples
```julia
# A basic linear layer without activation or bias (using defaults)
dense1 = Dense(784, 128)

# A layer with ReLU activation
dense2 = Dense(128, 64, relu)

# A layer without a bias term
dense3 = Dense(64, 10, bias=false)

# A layer with specific initialization
dense4 = Dense(20, 5, init=HeNormal(), bias_init=Ones())
```
"""
mutable struct Dense{T, F} <: AbstractLayer
    weight::Matrix{T}
    bias::Union{Vector{T}, Nothing}
    activation::F
    in_features::Int
    out_features::Int
end

"""
    Dense(
        in_features::Int,
        out_features::Int,
        activation::F = identity;
        bias::Bool = true,
        init::AbstractInitializer = DEFAULT_WEIGHT_INIT,
        bias_init::AbstractInitializer = DEFAULT_BIAS_INIT,
        dtype::Type{T} = Float32
    ) where {T, F}

Constructs a `Dense` layer.

Arguments:
- `in_features::Int`: The number of input features this layer expects. Must be positive.
- `out_features::Int`: The number of output features this layer produces. Must be positive.
- `activation::F`: The activation function to apply after the linear transformation.
                   Can be any Julia function or callable object. Defaults to `identity`.

Keyword Arguments:
- `bias::Bool`: If `true`, a bias term `b` is added to the output. Defaults to `true`.
- `init::AbstractInitializer`: The initializer strategy for the `weight` matrix.
                               Defaults to `DEFAULT_WEIGHT_INIT` (GlorotUniform).
- `bias_init::AbstractInitializer`: The initializer strategy for the `bias` vector.
                                    Defaults to `DEFAULT_BIAS_INIT` (Zeros).
- `dtype::Type{T}`: The desired element type for the `weight` and `bias` parameters.
                    Defaults to `Float32`.

Returns:
- A `Dense{T, F}` instance.

Throws:
- `AssertionError`: If `in_features` or `out_features` are not positive.
"""
function Dense(
    in_features::Int,
    out_features::Int,
    activation::F = identity;
    bias::Bool = true,
    init::AbstractInitializer = DEFAULT_WEIGHT_INIT,
    bias_init::AbstractInitializer = DEFAULT_BIAS_INIT,
    dtype::Type{T} = Float32
) where {T, F}
    @assert in_features > 0 "in_features must be positive."
    @assert out_features > 0 "out_features must be positive."

    weight = T.(init(in_features, out_features)) # Initializes as (in_features, out_features) matrix
    b = bias ? T.(bias_init(out_features)) : nothing # Initializes as (out_features,) vector

    Dense{T, F}(weight, b, activation, in_features, out_features)
end

# forward(d::Dense, x::AbstractTensor) is defined in backends/abstract.jl
# with backend-aware dispatch (routes through Zig/GPU when active).

"""
    parameters(d::Dense) -> NamedTuple

Returns a `NamedTuple` containing the trainable parameters of the `Dense` layer.
These parameters are typically optimized during the training process.

Arguments:
- `d::Dense`: The `Dense` layer instance.

Returns:
- A `NamedTuple` with keys `:weight` and (optionally) `:bias`.
    - If the layer was constructed with `bias=true`, the `NamedTuple` will contain
      `:weight => d.weight` and `:bias => d.bias`.
    - If the layer was constructed with `bias=false`, the `NamedTuple` will only
      contain `:weight => d.weight`.

# Example
```julia
dense_with_bias = Dense(10, 5)
params_with_bias = parameters(dense_with_bias) # (weight = Matrix{Float32}, bias = Vector{Float32})

dense_no_bias = Dense(10, 5, bias=false)
params_no_bias = parameters(dense_no_bias) # (weight = Matrix{Float32},)
```
"""
function parameters(d::Dense)
    if d.bias !== nothing
        (weight = d.weight, bias = d.bias)
    else
        (weight = d.weight,)
    end
end

"""
    output_shape(d::Dense, input_shape::Type{Shape{input}}) -> Shape

Infers the compile-time output `Shape` of the `Dense` layer given the
`input_shape`. This function is essential for Axiom.jl's static shape
checking system, ensuring dimensional compatibility across the neural
network architecture.

Arguments:
- `d::Dense`: The `Dense` layer instance.
- `input_shape::Type{Shape{input}}`: A `Type{Shape{input}}` representing the
                                       compile-time shape of the input tensor.
                                       Expected to be either 1D (`Shape{Tuple{num_features}}`)
                                       or 2D (`Shape{Tuple{batch_size, num_features}}`).

Returns:
- A `Shape` type representing the inferred output dimensions.
    - If input is 1D `(num_features,)`, the output will be 1D `(d.out_features,)`.
    - If input is 2D `(batch_size, num_features)`, the output will be 2D `(batch_size, d.out_features)`.

# Details:
- The function assumes the `input_shape` is compatible with the layer's `d.in_features`
  (i.e., `num_features` matches `d.in_features`). Runtime checks for this are
  performed by `verify_input_shape`.
- The batch dimension (if present) is propagated directly to the output shape,
  including `:dynamic` if `batch_size` is `typeof(:dynamic)`.

# Throws
- `ErrorException`: If `input_shape` has an unsupported number of dimensions (e.g., 3D or more).

# Examples
```julia
dense = Dense(10, 5) # in_features=10, out_features=5

# 1D input shape: (10,) -> (5,)
output_s1 = output_shape(dense, Shape{Tuple{10}}) # Shape{Tuple{5}}

# 2D input shape: (batch_size=32, 10) -> (32, 5)
output_s2 = output_shape(dense, Shape{Tuple{32, 10}}) # Shape{Tuple{32, 5}}

# 2D input shape with dynamic batch: (typeof(:dynamic), 10) -> (typeof(:dynamic), 5)
output_s3 = output_shape(dense, Shape{Tuple{typeof(:dynamic), 10}}) # Shape{Tuple{typeof(:dynamic), 5}}
```
"""
function output_shape(d::Dense, input_shape::Type{Shape{input}}) where input
    if length(input) == 1
        return Shape{Tuple{d.out_features}}
    elseif length(input) == 2
        return Shape{Tuple{input[1], d.out_features}}
    else
        error("Dense layer expects 1D or 2D input shape, got $(length(input))D")
    end
end

"""
    verify_input_shape(d::Dense, x::AbstractTensor) -> Bool

Performs runtime verification of the input tensor `x`'s feature dimension
against the `Dense` layer's `d.in_features`. This check ensures that the
number of features in the input data matches what the dense layer expects,
preventing runtime errors that compile-time shape inference alone might not
catch for dynamic dimensions.

Arguments:
- `d::Dense`: The `Dense` layer instance.
- `x::AbstractTensor`: The input data to the layer.

Returns:
- `Bool`: Always returns `true` if the verification passes. This function
          throws an error on failure.

Throws:
- `DimensionMismatch`: If the feature dimension of `x` does not match `d.in_features`.

# Details:
- For a 1D input `x` (single sample), `length(x)` is considered the input feature dimension.
- For a 2D input `x` (batch of samples), `size(x, 2)` (the second dimension) is
  considered the input feature dimension.

# Examples
```julia
dense_layer = Dense(10, 5)

# Valid 1D input
input_1d_ok = rand(10)
verify_input_shape(dense_layer, input_1d_ok) # Returns true

# Valid 2D input
input_2d_ok = rand(32, 10)
verify_input_shape(dense_layer, input_2d_ok) # Returns true

# Note: Invalid input dimensions will throw DimensionMismatch
# e.g., verify_input_shape(dense_layer, rand(8)) would fail
```
"""
function verify_input_shape(d::Dense, x::AbstractTensor)
    in_dim = ndims(x) == 1 ? length(x.data) : size(x.data, 2)
    if in_dim != d.in_features
        throw(DimensionMismatch(
            "Dense layer expects input with $(d.in_features) features, got $in_dim"
        ))
    end
    true
end

"""
    show_layer_params(io::IO, d::Dense)

Overrides the default `show_layer_params` for the `Dense` layer to provide
a concise and informative representation of its key parameters. This function
is called internally by `Base.show` for `AbstractLayer` types.

Arguments:
- `io::IO`: The I/O stream to which the parameters are printed.
- `d::Dense`: The `Dense` layer instance whose parameters are to be displayed.

# Output Format:
The output typically follows the format `in_features => out_features`,
with additional information for non-identity activation functions or when
the bias term is omitted.

# Example
```julia
# Default Dense layer
dense1 = Dense(784, 128)
# Output might look like: Dense(784 => 128)

# Dense layer with ReLU activation
dense2 = Dense(128, 64, relu)
# Output might look like: Dense(128 => 64, relu)

# Dense layer with no bias
dense3 = Dense(64, 10, bias=false)
# Output might look like: Dense(64 => 10, bias=false)
```
"""
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
    Bilinear(in1_features, in2_features, out_features; activation=identity, bias=true, dtype=Float32)

A `Bilinear` layer is designed to capture pairwise multiplicative interactions
between two distinct input feature vectors, `x1` and `x2`. For each output
feature `k`, it computes a bilinear form:
`y_k = x1' * W_k * x2 + b_k`
where `W_k` is a matrix of interaction weights specific to the k-th output.
This layer is particularly useful in applications such as:
-   **Attention Mechanisms**: Modeling interactions between query and key vectors.
-   **Recommender Systems**: Capturing user-item preferences.
-   **Multi-modal Fusion**: Combining features from different modalities.
-   **Knowledge Graph Embeddings**: Representing relationships between entities.

# Fields
- `weight::Array{T, 3}`: A 3D array of weights of size `(in1_features, in2_features, out_features)`.
                         `weight[:, :, k]` is the interaction matrix `W_k` for the k-th output.
- `bias::Union{Vector{T}, Nothing}`: The bias vector of size `(out_features,)`. If `bias` is `false` during construction, this field will be `nothing`.
- `activation::F`: The activation function `f` applied to the bilinear output (e.g., `relu`, `sigmoid`, `identity`).
- `in1_features::Int`: The number of features in the first input vector `x1`.
- `in2_features::Int`: The number of features in the second input vector `x2`.
- `out_features::Int`: The number of output features produced by the layer.

# Examples
```julia
# A bilinear layer taking two 10-feature inputs and producing 5 output features
bilinear_layer = Bilinear(10, 10, 5)

# A bilinear layer with ReLU activation
bilinear_relu = Bilinear(20, 15, 8, relu)
```
"""
mutable struct Bilinear{T, F} <: AbstractLayer
    weight::Array{T, 3}
    bias::Union{Vector{T}, Nothing}
    activation::F
    in1_features::Int
    in2_features::Int
    out_features::Int
end

"""
    Bilinear(
        in1_features::Int,
        in2_features::Int,
        out_features::Int,
        activation::F = identity;
        bias::Bool = true,
        dtype::Type{T} = Float32
    ) where {T, F}

Constructs a `Bilinear` layer.

Arguments:
- `in1_features::Int`: The number of features expected for the first input `x1`. Must be positive.
- `in2_features::Int`: The number of features expected for the second input `x2`. Must be positive.
- `out_features::Int`: The number of output features this layer produces. Must be positive.
- `activation::F`: The activation function to apply after the bilinear transformation.
                   Can be any Julia function or callable object. Defaults to `identity`.

Keyword Arguments:
- `bias::Bool`: If `true`, a bias term is added to the output. Defaults to `true`.
- `dtype::Type{T}`: The desired element type for the `weight` and `bias` parameters.
                    Defaults to `Float32`.

Returns:
- A `Bilinear{T, F}` instance.

Throws:
- `AssertionError`: If `in1_features`, `in2_features`, or `out_features` are not positive.

# Weight Initialization:
The `weight` array is initialized using a normal distribution scaled by `sqrt(2.0 / (in1_features + in2_features))`,
similar to Glorot/Xavier initialization. The `bias` is initialized to zeros if enabled.
"""
function Bilinear(
    in1_features::Int,
    in2_features::Int,
    out_features::Int,
    activation::F = identity;
    bias::Bool = true,
    dtype::Type{T} = Float32
) where {T, F}
    @assert in1_features > 0 "in1_features must be positive."
    @assert in2_features > 0 "in2_features must be positive."
    @assert out_features > 0 "out_features must be positive."

    # Initializing weights with a scaled random normal distribution
    weight = randn(T, in1_features, in2_features, out_features) .* T(sqrt(2.0 / (in1_features + in2_features)))
    b = bias ? zeros(T, out_features) : nothing

    Bilinear{T, F}(weight, b, activation, in1_features, in2_features, out_features)
end

"""
    forward(bl::Bilinear, x1::AbstractArray, x2::AbstractArray)

Performs the forward pass through the `Bilinear` layer. This involves computing
the bilinear interaction between two input feature vectors (`x1` and `x2`)
for each output feature, applying an optional bias, and then an activation function.

Arguments:
- `bl::Bilinear`: The `Bilinear` layer instance.
- `x1::AbstractArray`: The first input feature tensor. Expected to be a 2D array
                      of shape `(batch_size, bl.in1_features)`.
- `x2::AbstractArray`: The second input feature tensor. Expected to be a 2D array
                      of shape `(batch_size, bl.in2_features)`.

Returns:
- An `AbstractArray` representing the output of the layer, of shape
  `(batch_size, bl.out_features)`.

# Details:
The operation is performed for each sample in the batch and for each output feature.
For a batch of size `B`:
- `x1` has shape `(B, in1_features)`
- `x2` has shape `(B, in2_features)`
- `bl.weight` has shape `(in1_features, in2_features, out_features)`

The core computation for each `k` in `1:bl.out_features` is conceptualized as:
`output_k_batch = sum(x1 .* (x2 * bl.weight[:, :, k]'), dims=2)`
This performs `x1_batch .* ( (x2_batch @ W_k.T) )` and then sums over the relevant dimensions.

- **Bias Addition**: If `bl.bias` is not `nothing`, the bias vector is added to the
  bilinear output (broadcasting across the batch dimension).
- **Activation**: Finally, the `bl.activation` function is applied element-wise to the result.

# Examples
```julia
bilinear_layer = Bilinear(5, 7, 3) # in1=5, in2=7, out=3
batch_size = 16

input1 = rand(batch_size, 5) # Input features for x1
input2 = rand(batch_size, 7) # Input features for x2

output = forward(bilinear_layer, input1, input2)
# output will be a Matrix{Float32} of size (16, 3)
```
"""
function forward(bl::Bilinear, x1::AbstractArray, x2::AbstractArray)
    # Batch bilinear operation
    batch_size = size(x1, 1)
    out = zeros(eltype(x1), batch_size, bl.out_features)

    # Ensure input dimensions match layer's expected features
    @assert size(x1, 2) == bl.in1_features "Mismatch in x1 features: expected $(bl.in1_features), got $(size(x1, 2))"
    @assert size(x2, 2) == bl.in2_features "Mismatch in x2 features: expected $(bl.in2_features), got $(size(x2, 2))"
    @assert size(x1, 1) == size(x2, 1) "Batch sizes of x1 and x2 must match."

    # Loop through each output feature to compute its bilinear component
    for i in 1:bl.out_features
        # bl.weight[:, :, i]' is W_k^T
        # x2 * bl.weight[:, :, i]' gives (batch_size, in2_features) * (in2_features, in1_features) -> (batch_size, in1_features)
        # x1 .* (...) performs element-wise product
        # sum(..., dims=2) sums out the feature dimension, leaving (batch_size, 1)
        out[:, i] = sum(x1 .* (x2 * bl.weight[:, :, i]'), dims=2)
    end

    if bl.bias !== nothing
        out = out .+ bl.bias'
    end

    bl.activation(out)
end

"""
    parameters(bl::Bilinear) -> NamedTuple

Returns a `NamedTuple` containing the trainable parameters of the `Bilinear` layer.
These parameters are typically optimized during the training process.

Arguments:
- `bl::Bilinear`: The `Bilinear` layer instance.

Returns:
- A `NamedTuple` with keys `:weight` and (optionally) `:bias`.
    - If the layer was constructed with `bias=true`, the `NamedTuple` will contain
      `:weight => bl.weight` and `:bias => bl.bias`.
    - If the layer was constructed with `bias=false`, the `NamedTuple` will only
      contain `:weight => bl.weight`.

# Example
```julia
bilinear_with_bias = Bilinear(5, 7, 3)
params_with_bias = parameters(bilinear_with_bias) # (weight = Array{Float32, 3}, bias = Vector{Float32})

bilinear_no_bias = Bilinear(5, 7, 3, bias=false)
params_no_bias = parameters(bilinear_no_bias) # (weight = Array{Float32, 3},)
```
"""
function parameters(bl::Bilinear)
    if bl.bias !== nothing
        (weight = bl.weight, bias = bl.bias)
    else
        (weight = bl.weight,)
    end
end
