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
struct Pipeline{Layers} <: AbstractLayer
    layers::Layers
end

function Pipeline(layers...)
    Pipeline(layers)
end

# Forward through pipeline
function forward(p::Pipeline, x)
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

# Pipeline construction from |>
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

# Pipeline optimization
"""
    optimize_pipeline(pipeline) -> Pipeline

Optimize a pipeline by fusing compatible operations.
"""
function optimize_pipeline(p::Pipeline)
    # TODO: Implement fusion optimizations:
    # - Fuse element-wise operations
    # - Eliminate redundant reshapes
    # - Combine batch norms with linear layers
    p
end

# Shape debugging
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
