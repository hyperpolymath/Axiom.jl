# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Pipeline System
#
# Functional composition of layers using the |> operator.
# Enables automatic fusion and optimization.

import Base: |>

"""
    x |> layer

Apply layer to input, with shape verification.
This extends Julia's pipe operator for Axiom.jl layers.
"""
function (|>)(x::AbstractTensor, layer::AbstractLayer)
    # Verify input shape compatibility
    verify_input_shape(layer, x)

    # Apply layer
    output = forward(layer, x)

    # Wrap result in appropriate tensor type
    output
end

# Also support function-style layers
function (|>)(x::AbstractTensor, f::Function)
    f(x)
end

"""
    Pipeline{Layers}

A composed sequence of layers that acts as a single layer.
"""
struct Pipeline{Layers<:Tuple} <: AbstractLayer
    layers::Layers

    # Inner constructor to ensure layers is a tuple
    function Pipeline(layers::Layers) where Layers<:Tuple
        new{Layers}(layers)
    end
end

function Pipeline(layers...)
    Pipeline(Tuple(layers)) # Ensure layers is always a tuple
end

# Forward through pipeline
function forward(p::Pipeline, x::AbstractTensor)
    for layer in p.layers
        x = forward(layer, x)
    end
    x
end

# Shape inference for pipeline
function output_shape(p::Pipeline, input_shape)
    shape = input_shape
    for layer in p.layers
        shape = output_shape(layer, shape)
    end
    shape
end

"""
    Chain(layers...)

Alias for Pipeline - compatible with Flux.jl naming.
"""
const Chain = Pipeline

"""
    Sequential(layers...)

Another alias for Pipeline - compatible with PyTorch naming.
"""
const Sequential = Pipeline

"""
    compose(f, g)

Compose two layers: compose(f, g)(x) = g(f(x))
"""
compose(f::AbstractLayer, g::AbstractLayer) = Pipeline(f, g)

"""
    ∘(g, f)

Function composition operator for layers: (g ∘ f)(x) = g(f(x))
"""
Base.:∘(g::AbstractLayer, f::AbstractLayer) = compose(f, g)

function build_pipeline(expr::Expr)
    if expr.head == :call && expr.args[1] == :|>
        input_or_pipeline = expr.args[2]
        layer = expr.args[3]

        if input_or_pipeline isa Expr && input_or_pipeline.head == :call && input_or_pipeline.args[1] == :|>
            # Nested pipeline: (a |> b) |> c
            inner = build_pipeline(input_or_pipeline)
            return (inner..., layer)
        else
            # Base case: input |> layer
            return (layer,)
        end
    end
    error("Not a pipeline expression")
end

"""
    @pipeline expr

Build an optimized pipeline from a chain of |> operations.

# Example
```julia
pipeline = @pipeline input |> Dense(784, 128) |> ReLU |> Dense(128, 10)
# Equivalent to: Pipeline(Dense(784, 128), ReLU(), Dense(128, 10))
```
"""
macro pipeline(expr)
    layers = build_pipeline(expr)
    :(Pipeline($(map(esc, layers)...)))
end

# Parallel pipelines
"""
    Parallel(pipelines...)

Run multiple pipelines in parallel and concatenate outputs.
"""
struct Parallel{Pipelines} <: AbstractLayer
    pipelines::Pipelines
end

Parallel(pipelines...) = Parallel(pipelines)

function forward(p::Parallel, x)
    outputs = [forward(pipeline, x) for pipeline in p.pipelines]
    # Concatenate along feature dimension
    cat(outputs..., dims=ndims(outputs[1]))
end

# Residual connection
"""
    Residual(block)

Residual connection: output = input + block(input)
"""
struct Residual{Block} <: AbstractLayer
    block::Block
end

function forward(r::Residual, x)
    x + forward(r.block, x)
end

"""
    SkipConnection(block, connection)

General skip connection: output = connection(input, block(input))
"""
struct SkipConnection{Block, Connection} <: AbstractLayer
    block::Block
    connection::Connection  # Function like (x, fx) -> x + fx
end

function SkipConnection(block)
    SkipConnection(block, +)
end

function forward(s::SkipConnection, x)
    s.connection(x, forward(s.block, x))
end

# Conditional execution
"""
    Conditional(predicate, if_true, if_false)

Conditionally apply one of two layers based on predicate.
"""
struct Conditional{Pred, IfTrue, IfFalse} <: AbstractLayer
    predicate::Pred
    if_true::IfTrue
    if_false::IfFalse
end

function forward(c::Conditional, x)
    if c.predicate(x)
        forward(c.if_true, x)
    else
        forward(c.if_false, x)
    end
end

function optimize_pipeline(p::Pipeline)
    layers = collect(p.layers)
    optimized = AbstractLayer[]
    i = 1

    while i <= length(layers)
        layer = layers[i]

        # Optimization 1: Fuse consecutive element-wise activations
        if i < length(layers) && is_elementwise(layer) && is_elementwise(layers[i + 1])
            fused = FusedActivation(layer, layers[i + 1])
            push!(optimized, fused)
            i += 2
            continue
        end

        # Optimization 2: Fuse BatchNorm with preceding Dense/Conv layer
        if i < length(layers) && is_linear_layer(layer) && is_batchnorm(layers[i + 1])
            fused = fuse_linear_batchnorm(layer, layers[i + 1])
            push!(optimized, fused)
            i += 2
            continue
        end

        # Optimization 3: Eliminate identity/redundant operations
        if is_identity_op(layer)
            i += 1
            continue
        end

        push!(optimized, layer)
        i += 1
    end

    Pipeline(Tuple(optimized))
end

is_elementwise(::StatelessLayer) = true
is_elementwise(::AbstractLayer) = false

is_linear_layer(::Dense) = true
is_linear_layer(::Conv2d) = true
is_linear_layer(::Conv1d) = true
is_linear_layer(::AbstractLayer) = false

is_batchnorm(::BatchNorm) = true
is_batchnorm(::AbstractLayer) = false

is_identity_op(::AbstractLayer) = false

"""
    FusedActivation

Fused consecutive element-wise activations.
"""
struct FusedActivation{A, B} <: StatelessLayer
    first::A
    second::B
end

function forward(f::FusedActivation, x)
    forward(f.second, forward(f.first, x))
end

output_shape(f::FusedActivation, input_shape) = output_shape(f.second, output_shape(f.first, input_shape))

"""
    FusedLinearBatchNorm

Fused Dense/Conv layer with BatchNorm for inference.
The BatchNorm parameters are folded into the linear layer weights.
"""
struct FusedLinearBatchNorm{L<:AbstractLayer} <: AbstractLayer
    layer::L
end

function fuse_linear_batchnorm(linear::Dense{T}, bn::BatchNorm{T}) where T
    # Fold BatchNorm into Dense weights: W_fused = W * γ / sqrt(σ² + ε)
    # b_fused = γ * (b - μ) / sqrt(σ² + ε) + β

    γ = bn.affine ? bn.γ : ones(T, bn.num_features)
    β = bn.affine ? bn.β : zeros(T, bn.num_features)
    μ = bn.running_mean
    σ = sqrt.(bn.running_var .+ bn.eps)

    scale = γ ./ σ

    # Fused weight: scale each output column
    weight_fused = linear.weight .* scale'

    # Fused bias
    if linear.bias !== nothing
        bias_fused = scale .* (linear.bias .- μ) .+ β
    else
        bias_fused = scale .* (-μ) .+ β
    end

    fused_dense = Dense{T, typeof(linear.activation)}(
        weight_fused, bias_fused, linear.activation,
        linear.in_features, linear.out_features
    )

    FusedLinearBatchNorm(fused_dense)
end

function fuse_linear_batchnorm(conv::Conv2d{T}, bn::BatchNorm{T}) where T
    # Fold BatchNorm into Conv2d weights
    γ = bn.affine ? bn.γ : ones(T, bn.num_features)
    β = bn.affine ? bn.β : zeros(T, bn.num_features)
    μ = bn.running_mean
    σ = sqrt.(bn.running_var .+ bn.eps)

    scale = γ ./ σ

    # Scale each output channel
    weight_fused = similar(conv.weight)
    for oc in 1:conv.out_channels
        weight_fused[:, :, :, oc] = conv.weight[:, :, :, oc] .* scale[oc]
    end

    # Fused bias
    if conv.bias !== nothing
        bias_fused = scale .* (conv.bias .- μ) .+ β
    else
        bias_fused = scale .* (-μ) .+ β
    end

    fused_conv = Conv2d{T}(
        weight_fused, bias_fused, conv.stride, conv.padding,
        conv.dilation, conv.groups, conv.in_channels, conv.out_channels, conv.kernel_size
    )

    FusedLinearBatchNorm(fused_conv)
end

forward(f::FusedLinearBatchNorm, x) = forward(f.layer, x)
parameters(f::FusedLinearBatchNorm) = parameters(f.layer)
output_shape(f::FusedLinearBatchNorm, input_shape) = output_shape(f.layer, input_shape)

"""
    trace_shapes(pipeline, input_shape)

Trace shapes through a pipeline for debugging.
"""
function trace_shapes(p::Pipeline, input_shape)
    println("Input: $input_shape")
    shape = input_shape

    for (i, layer) in enumerate(p.layers)
        shape = output_shape(layer, shape)
        println("  Layer $i ($(typeof(layer))): $shape")
    end

    println("Output: $shape")
    shape
end
