# SPDX-License-Identifier: PMPL-1.0-or-later
#
# Axiom.jl Abstract Layer Types and Interface Definitions
# ======================================================
#
# This file defines the foundational abstract types and interfaces that govern
# the structure and behavior of all neural network layers within Axiom.jl.
# By establishing a clear contract for layer implementations, this module
# ensures consistency, extensibility, and compatibility across the entire
# neural network framework.
#
# Importance of Abstraction:
# --------------------------
# -   **Modularity**: Promotes the development of independent, reusable layer components.
# -   **Extensibility**: Simplifies the process of adding new custom layers by
#     providing a well-defined blueprint.
# -   **Type Safety**: Ensures that functions operating on `AbstractLayer` can
#     expect certain methods to exist, enhancing robustness.
# -   **Polymorphism**: Allows generic functions to operate on diverse layer types,
#     treating them uniformly.
#
# Core Layer Interface (Contract for Concrete Layers):
# ---------------------------------------------------
# Any concrete layer `struct` in Axiom.jl that inherits from `AbstractLayer`
# is expected to implement (or adhere to the conventions of) the following
# methods:
# -   `forward(layer, x)`: Defines the computational logic of the layer.
# -   `parameters(layer)`: Returns a collection of trainable parameters.
# -   `output_shape(layer, input_shape)`: Infers the output tensor shape.
# -   `verify_input_shape(layer, x)`: (Optional) Performs runtime input shape validation.
#
# Weight Initialization Strategies:
# ---------------------------------
# This module also introduces an abstract type for `AbstractInitializer` and
# provides several common weight initialization schemes (e.g., Glorot, He, Zeros, Ones).
# These strategies are crucial for stable and effective training of deep neural networks.
#
#

"""
    AbstractLayer

An abstract base type for all neural network layers in Axiom.jl.
This type defines the fundamental interface that all concrete layers must
adhere to, enabling a consistent and polymorphic way to build, train, and
reason about neural network architectures.

All concrete layers inheriting from `AbstractLayer` are expected to implement
the following methods to fully integrate into the Axiom.jl framework:

Required Methods:
- `forward(layer, x)`: Computes the forward pass of the layer, transforming
                       input `x` into output.
- `parameters(layer)`: Returns a `NamedTuple` or collection of the layer's
                       trainable parameters (e.g., weights, biases). Layers
                       without trainable parameters (e.g., activation functions)
                       can implement a trivial `parameters` method.
- `output_shape(layer, input_shape)`: Infers the compile-time `Shape` of the
                                       layer's output given the `input_shape`.
                                       This is crucial for Axiom.jl's static
                                       shape checking.

Optional/Commonly Implemented Methods:
- `verify_input_shape(layer, x)`: (Runtime) Performs additional checks on the
                                  input tensor's shape or properties beyond
                                  what compile-time checks can ensure.
- `set_training!(layer, mode::Bool)`: Sets the layer's mode to training or
                                      evaluation (e.g., for `Dropout` or `BatchNorm`).

# Examples of concrete layers:
- `Dense`
- `Conv2D`
- `ReLU`
"""
abstract type AbstractLayer end

# Required interface
"""
    forward(layer::AbstractLayer, x::AbstractTensor)

Computes the forward pass of a neural network `layer`. This is the primary
method that defines the layer's functional behavior, transforming an input
tensor `x` into an output tensor. All concrete `AbstractLayer` subtypes
must implement this method.

Arguments:
- `layer`: The `AbstractLayer` instance through which the input `x` is passed.
- `x`: An `AbstractTensor` representing the input data to the layer. The shape
       and element type of `x` must be compatible with the `layer`'s design.

Returns:
- An `AbstractTensor` representing the output of the layer. Its shape and
  element type depend on the specific layer and input.

# Example
```julia
# Assuming a Dense layer is defined elsewhere
dense_layer = Dense(10, 5) # Input features=10, output features=5
input_tensor = axiom_randn(10) # 1D input tensor with 10 features

output_tensor = forward(dense_layer, input_tensor) # Computes the forward pass
# output_tensor will be a 1D tensor with 5 features
```
"""
function forward end

"""
    parameters(layer::AbstractLayer) -> NamedTuple

Returns a `NamedTuple` containing the trainable parameters (e.g., weights, biases)
of the given `layer`. This method is crucial for integrating layers with optimization
algorithms, allowing optimizers to efficiently access and update only the relevant
parameters during the training process.

Arguments:
- `layer`: The `AbstractLayer` instance from which to retrieve parameters.

Returns:
- A `NamedTuple` where keys are parameter names (e.g., `:weight`, `:bias`) and
  values are the corresponding parameter tensors or arrays.
  For layers without trainable parameters, an empty `NamedTuple()` is returned.

# Example
```julia
# Assuming a Dense layer is defined elsewhere
dense_layer = Dense(10, 5) # Input features=10, output features=5

params = parameters(dense_layer)
# params might look like (weight = Tensor{...}, bias = Tensor{...})

# For a stateless layer (e.g., ReLU), it would return an empty NamedTuple
relu_layer = ReLU()
params_relu = parameters(relu_layer) # NamedTuple()
```
"""
function parameters(layer::AbstractLayer)
    NamedTuple()
end

"""
    output_shape(layer::AbstractLayer, input_shape::Type{Shape{input}}) -> Shape

Infers the compile-time `Shape` of the output tensor that `layer` would produce,
given the `input_shape`. This method is critical for Axiom.jl's static shape
checking system, allowing the framework to verify architectural consistency
and prevent shape mismatches before runtime. All concrete `AbstractLayer` subtypes
must implement this method.

Arguments:
- `layer`: The `AbstractLayer` instance for which to infer the output shape.
- `input_shape`: A `Type{Shape{input}}` representing the compile-time shape
                 of the tensor that would be fed into the layer.

Returns:
- A `Shape` type representing the inferred output dimensions.

# Example
```julia
# Assuming a Dense layer is defined elsewhere
dense_layer = Dense(10, 5) # Input features=10, output features=5
input_shape = Shape{Tuple{typeof(:dynamic), 10}} # Batch, features

output_s = output_shape(dense_layer, input_shape)
# output_s would be Shape{Tuple{typeof(:dynamic), 5}}
```
"""
function output_shape end

"""
    verify_input_shape(layer::AbstractLayer, x::AbstractTensor) -> Bool

Performs runtime verification of the input tensor `x`'s shape or other properties
against the expectations of the `layer`. This method complements compile-time
shape checks by allowing for more dynamic or context-dependent validation that
cannot be expressed purely at the type level.

Arguments:
- `layer`: The `AbstractLayer` instance for which the input is being verified.
- `x`: An `AbstractTensor` representing the input data to the layer.

Returns:
- `Bool`: `true` if the input shape and properties are compatible with the layer,
          `false` otherwise. This method typically throws an `ErrorException`
          (or a more specific exception like `DimensionMismatch`) on failure.

# Default Implementation:
The default implementation simply returns `true`, implying no specific runtime
verification is performed unless overridden by a concrete layer.

# When to Override:
Concrete layers should override this method if:
- They have dynamic input size constraints that cannot be fully captured by
  compile-time `Shape` parameters (e.g., minimum feature dimensions).
- They require specific properties of the input data that are not part of its
  `Shape` (e.g., non-negative values, specific range).

# Example
Concrete layers can override this method to add runtime validation.
For instance, a layer might check minimum batch size or specific constraints
that cannot be expressed at compile-time.
"""
function verify_input_shape(layer::AbstractLayer, x::AbstractTensor)
    # Default: no specific runtime verification. Concrete layers may override this.
    true
end

# Layer state
"""
    trainable(layer) -> Bool

Check if layer has trainable parameters.
"""
trainable(layer::AbstractLayer) = !isempty(parameters(layer))

"""
    set_training!(layer::AbstractLayer, mode::Bool)

Sets the operating mode of a `layer` to either training (`true`) or evaluation (`false`).
This method is crucial for layers whose behavior differs between training and inference,
such as `Dropout` (which is active during training but inactive during evaluation)
or `BatchNorm` (which updates running statistics during training but uses fixed
statistics during evaluation).

Arguments:
- `layer`: The `AbstractLayer` instance whose mode is to be set.
- `mode`: A `Bool` value. `true` for training mode, `false` for evaluation (inference) mode.

# Default Implementation:
The default implementation checks if the `layer` `hasfield` named `:training`. If it does,
it sets `layer.training = mode`. Layers that require different behaviors based on the
mode are expected to have a `training::Bool` field and/or override this method
to implement their specific logic.

# When to Override:
Concrete layers that need to change their internal state or logic based on
training/evaluation mode (e.g., to enable/disable dropout, use population
statistics in BatchNorm) should override this method.

# Example
Layers with a `training` field can use this method to switch between
training and evaluation modes. For instance, Dropout layers are active
during training but disabled during evaluation.
"""
function set_training!(layer::AbstractLayer, mode::Bool)
    if hasfield(typeof(layer), :training)
        layer.training = mode
    end
end

# Parameter access
"""
    nparams(layer::AbstractLayer) -> Int

Counts the total number of trainable parameters in a `layer`. This is a utility
function for understanding the complexity and memory footprint of a neural
network model. It sums the `length` of all parameter arrays returned by
`parameters(layer)`.

Arguments:
- `layer`: The `AbstractLayer` instance for which to count parameters.

Returns:
- `Int`: The total number of trainable parameters in the layer.

# Example
```julia
# Assuming a Dense layer is defined elsewhere
dense_layer = Dense(10, 5) # Input features=10, output features=5

num_params = nparams(dense_layer)
# num_params will be (10*5 for weights) + (5 for bias) = 55
```
"""
function nparams(layer::AbstractLayer)
    sum(length, values(parameters(layer)), init=0)
end

# Layer composition
"""
    (layer::AbstractLayer)(x::AbstractTensor)

Enables `AbstractLayer` objects to be called directly as functions, providing
convenient syntactic sugar for performing a forward pass. This method
simply delegates the call to the `forward(layer, x)` function.

Arguments:
- `layer`: The `AbstractLayer` instance to be called.
- `x`: An `AbstractTensor` representing the input data to the layer.

Returns:
- An `AbstractTensor` representing the output of the layer, as returned by `forward(layer, x)`.

# Example
```julia
# Assuming a Dense layer is defined elsewhere
dense_layer = Dense(10, 5) # Input features=10, output features=5
input_tensor = axiom_randn(10)

# Direct call (syntactic sugar)
output_tensor_direct = dense_layer(input_tensor)

# Equivalent to
output_tensor_forward = forward(dense_layer, input_tensor)
```
"""
(layer::AbstractLayer)(x::AbstractTensor) = forward(layer, x)
(layer::AbstractLayer)(x::AbstractArray) = forward(layer, Tensor(x))

# Pretty printing
"""
    Base.show(io::IO, layer::AbstractLayer)

Provides a concise, single-line string representation of an `AbstractLayer`
object for console output. This method typically displays the layer's type
name and delegates to `show_layer_params` to include specific parameters.
"""
function Base.show(io::IO, layer::AbstractLayer)
    print(io, "$(typeof(layer).name.name)(")
    show_layer_params(io, layer)
    print(io, ")")
end

"""
    show_layer_params(io::IO, layer::AbstractLayer)

A helper function intended to be overridden by concrete `AbstractLayer` subtypes.
Its purpose is to print the specific parameters or configuration details of a
layer within the `Base.show` method.

# Default Implementation:
The default implementation does nothing, resulting in `LayerName()` output.

# How to Override:
Concrete layers should implement `show_layer_params` to print a comma-separated
list of their fields or relevant attributes.

# Example
Concrete layers override this to print their specific parameters,
such as in_features and out_features for a Dense layer.
"""
function show_layer_params(io::IO, layer::AbstractLayer)
    # Override in specific layers
end

# Weight initialization schemes
"""
    AbstractInitializer

An abstract base type for all weight initialization schemes in Axiom.jl.
This type defines a common interface for various strategies used to initialize
the weights and biases of neural network layers, which is crucial for stable
and effective training.

Concrete initializers inheriting from `AbstractInitializer` must implement
the following method:
- `(init::Initializer)(dims...)`: A callable method that takes variable
  dimensions (`dims...`) as input and returns an initialized array of weights.
"""
abstract type AbstractInitializer end

"""
    GlorotUniform()

Implements Glorot uniform initialization, also known as Xavier uniform
initialization. This strategy is commonly used for initializing the weights
of neural network layers, particularly effective for layers with sigmoid
or tanh activation functions. It aims to keep the variance of activations
and gradients approximately constant across layers, preventing signals from
vanishing or exploding.

The weights are drawn from a uniform distribution `U(-limit, limit)`, where:
`limit = sqrt(6 / (fan_in + fan_out))`

Arguments:
- `dims...`: A variable number of integer arguments specifying the dimensions
             of the weight tensor to be initialized. `dims[1]` is typically
             `fan_in` (input features), and `dims[2]` (if available) is `fan_out`
             (output features).

Returns:
- An `Array{Float32}` with dimensions `dims...`, initialized with values from
  the Glorot uniform distribution.

# Example
```julia
init_weights = GlorotUniform()
weights = init_weights(10, 5) # Initialize a 10x5 weight matrix
# Values will be in [-limit, limit] based on fan_in=10, fan_out=5
```
"""
struct GlorotUniform <: AbstractInitializer end

function (init::GlorotUniform)(dims...)
    fan_in = dims[1]
    fan_out = length(dims) > 1 ? dims[2] : dims[1] # For 1D, assume fan_out = fan_in
    limit = Float32(sqrt(6.0 / (fan_in + fan_out)))
    (rand(Float32, dims...) .- 0.5f0) .* 2 .* limit # Scale random [0,1) to [-limit, limit)
end

"""
    GlorotNormal()

Implements Glorot normal initialization, also known as Xavier normal
initialization. Similar to `GlorotUniform`, this strategy is effective for
layers with sigmoid or tanh activation functions, aiming to maintain a
consistent variance of activations and gradients across layers.

The weights are drawn from a normal distribution `N(0, std^2)`, where:
`std = sqrt(2 / (fan_in + fan_out))`

Arguments:
- `dims...`: A variable number of integer arguments specifying the dimensions
             of the weight tensor to be initialized. `dims[1]` is typically
             `fan_in` (input features), and `dims[2]` (if available) is `fan_out`
             (output features).

Returns:
- An `Array{Float32}` with dimensions `dims...`, initialized with values from
  the Glorot normal distribution.

# Example
```julia
init_weights = GlorotNormal()
weights = init_weights(10, 5) # Initialize a 10x5 weight matrix
# Values will be drawn from N(0, std^2) based on fan_in=10, fan_out=5
```
"""
struct GlorotNormal <: AbstractInitializer end

function (init::GlorotNormal)(dims...)
    fan_in = dims[1]
    fan_out = length(dims) > 1 ? dims[2] : dims[1] # For 1D, assume fan_out = fan_in
    std = Float32(sqrt(2.0 / (fan_in + fan_out)))
    randn(Float32, dims...) .* std
end

"""
    HeUniform()

Implements He uniform initialization, a robust strategy for initializing weights
in neural networks, particularly effective when using ReLU (Rectified Linear Unit)
or its variants as activation functions. He initialization aims to mitigate the
vanishing/exploding gradient problem by maintaining consistent variance in both
the forward and backward passes.

The weights are drawn from a uniform distribution `U(-limit, limit)`, where:
`limit = sqrt(6 / fan_in)`

Arguments:
- `dims...`: A variable number of integer arguments specifying the dimensions
             of the weight tensor to be initialized. `dims[1]` is typically
             `fan_in` (input features).

Returns:
- An `Array{Float32}` with dimensions `dims...`, initialized with values from
  the He uniform distribution.

# Example
```julia
init_weights = HeUniform()
weights = init_weights(10, 5) # Initialize a 10x5 weight matrix
# Values will be in [-limit, limit] based on fan_in=10
```
"""
struct HeUniform <: AbstractInitializer end

function (init::HeUniform)(dims...)
    fan_in = dims[1]
    limit = Float32(sqrt(6.0 / fan_in))
    (rand(Float32, dims...) .- 0.5f0) .* 2 .* limit
end

"""
    HeNormal()

Implements He normal initialization. Similar to `HeUniform`, this strategy is
highly effective for initializing weights in neural networks that utilize
ReLU or its variants as activation functions. It aims to prevent vanishing
or exploding gradients by maintaining a consistent variance in signal propagation.

The weights are drawn from a normal distribution `N(0, std^2)`, where:
`std = sqrt(2 / fan_in)`

Arguments:
- `dims...`: A variable number of integer arguments specifying the dimensions
             of the weight tensor to be initialized. `dims[1]` is typically
             `fan_in` (input features).

Returns:
- An `Array{Float32}` with dimensions `dims...`, initialized with values from
  the He normal distribution.

# Example
```julia
init_weights = HeNormal()
weights = init_weights(10, 5) # Initialize a 10x5 weight matrix
# Values will be drawn from N(0, std^2) based on fan_in=10
```
"""
struct HeNormal <: AbstractInitializer end

function (init::HeNormal)(dims...)
    fan_in = dims[1]
    std = Float32(sqrt(2.0 / fan_in))
    randn(Float32, dims...) .* std
end

"""
    Zeros()

Implements zero initialization. This strategy initializes all weights or biases
of a layer to zero. It is commonly used for biases, as initializing them to
zero ensures that the layer has no initial preference or offset. For weights,
it is generally avoided for all layers as it can lead to all neurons learning
the same features in subsequent layers (the "symmetry breaking" problem).

Arguments:
- `dims...`: A variable number of integer arguments specifying the dimensions
             of the tensor to be initialized.

Returns:
- An `Array{Float32}` with dimensions `dims...`, with all elements set to `0.0f0`.

# Example
```julia
init_biases = Zeros()
biases = init_biases(5) # Initialize a 1D bias vector of 5 zeros
```
"""
struct Zeros <: AbstractInitializer end

(init::Zeros)(dims...) = zeros(Float32, dims...)

"""
    Ones()

Implements one initialization. This strategy initializes all weights or biases
of a layer to one. While less common for general weights, it can be useful
for specific components like certain gating mechanisms in recurrent neural
networks, for debugging purposes, or for parts of normalization layers where
an initial scaling factor of one is desired.

Arguments:
- `dims...`: A variable number of integer arguments specifying the dimensions
             of the tensor to be initialized.

Returns:
- An `Array{Float32}` with dimensions `dims...`, with all elements set to `1.0f0`.

# Example
```julia
init_gamma = Ones()
gamma_weights = init_gamma(10) # Initialize a 1D vector of 10 ones
```
"""
struct Ones <: AbstractInitializer end

(init::Ones)(dims...) = ones(Float32, dims...)

# Default initializers
"""
    DEFAULT_WEIGHT_INIT

A constant representing the default weight initialization strategy used
throughout Axiom.jl for new layers. By default, it uses `GlorotUniform()`,
which is a suitable general-purpose initializer for a wide range of
activation functions, helping to stabilize training.
"""
const DEFAULT_WEIGHT_INIT = GlorotUniform()

"""
    DEFAULT_BIAS_INIT

A constant representing the default bias initialization strategy used
throughout Axiom.jl for new layers. By default, it uses `Zeros()`,
initializing all biases to zero, which is a common and often effective
practice as it introduces no initial bias in the layer's output.
"""
const DEFAULT_BIAS_INIT = Zeros()

# Stateless layer marker
"""
    StatelessLayer <: AbstractLayer

An abstract marker type for neural network layers that do not have any
trainable parameters. Layers inheriting from `StatelessLayer` (e.g.,
activation functions like ReLU, pooling layers like MaxPool, or structural
layers like Flatten) implicitly return an empty `NamedTuple()` when `parameters(layer)`
is called. This simplifies parameter management and optimization for layers
that only perform a fixed transformation.

Layers that should inherit from `StatelessLayer`:
-   Activation functions (e.g., `ReLU`, `Sigmoid`)
-   Pooling layers (e.g., `MaxPool2D`, `GlobalAvgPool`)
-   Structural layers (e.g., `Flatten`)
-   Dropout layers (if `p` is not considered trainable)

"""
abstract type StatelessLayer <: AbstractLayer end

parameters(::StatelessLayer) = NamedTuple() # Stateless layers have no trainable parameters

"""
    Dropout(p=0.5)

Dropout regularization layer. During training, randomly zeroes elements with
probability `p` and scales remaining elements by `1/(1-p)`. During evaluation
(inference), acts as identity.
"""
struct Dropout <: StatelessLayer
    p::Float64
end

Dropout(; p::Real=0.5) = Dropout(Float64(p))
Dropout(p::Real) = Dropout(Float64(p))

function forward(d::Dropout, x::AbstractArray; training::Bool=false)
    backend_dropout(current_backend(), x, eltype(x)(d.p), training)
end

function (d::Dropout)(x::AbstractArray; training::Bool=false)
    forward(d, x; training=training)
end
