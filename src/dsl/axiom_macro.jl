# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl @axiom DSL Macro
#
# The core of Axiom.jl's declarative model definition.
# Transforms high-level model specifications into verified, optimized code.

"""
    @axiom name body

Define a verified machine learning model with compile-time guarantees.

# Syntax
```julia
@axiom ModelName begin
    input :: TensorType
    output :: TensorType

    # Layer definitions
    layer1 = input |> SomeLayer(...)
    layer2 = layer1 |> AnotherLayer(...)
    output = layer2 |> FinalLayer(...)

    # Invariants (checked at compile time where possible)
    @ensure property1
    @ensure property2
end
```

# Example
```julia
@axiom Classifier begin
    input :: Tensor{Float32, (28, 28, 1)}
    output :: Tensor{Float32, (10,)}

    features = input |> Flatten |> Dense(128, relu) |> Dense(64, relu)
    logits = features |> Dense(10)
    output = logits |> Softmax

    @ensure sum(output) ≈ 1.0
    @ensure all(output .>= 0)
end
```
"""
macro axiom(name, body)
    # Parse the model definition
    parsed = parse_axiom_body(body)

    # Generate the model struct and methods
    generate_axiom_code(name, parsed)
end

"""
Parsed representation of an @axiom body.
"""
struct AxiomDefinition
    input_type::Union{Expr, Nothing}
    output_type::Union{Expr, Nothing}
    layers::Vector{Pair{Symbol, Union{Expr, Symbol}}}
    ensures::Vector{Expr}
    proves::Vector{Expr}
end

"""
    parse_axiom_body(body::Expr) -> AxiomDefinition

Parse the `begin...end` block of an `@axiom` macro call into a structured
`AxiomDefinition` object. It identifies input/output type annotations,
layer assignments, and `@ensure`/`@prove` macros.
"""
function parse_axiom_body(body::Expr)
    @assert body.head == :block "Expected begin...end block"

    input_type = nothing
    output_type = nothing
    layers = Pair{Symbol, Union{Expr, Symbol}}[]
    ensures = Expr[]
    proves = Expr[]

    for expr in body.args
        # Skip line numbers
        expr isa LineNumberNode && continue

        if expr isa Expr
            if expr.head == :(::)
                # Type annotation: input :: Type or output :: Type
                name, typ = _parse_type_annotation(expr)
                if name == :input
                    input_type = typ
                elseif name == :output
                    output_type = typ
                end
            elseif expr.head == :(=)
                # Assignment: layer = expr
                name, value = _parse_assignment(expr)
                push!(layers, name => value)
            elseif expr.head == :macrocall
                # Macro: @ensure or @prove
                macro_type, macro_expr = _parse_macro_call(expr)
                if macro_type == :ensure
                    push!(ensures, macro_expr)
                elseif macro_type == :prove
                    push!(proves, macro_expr)
                end
            end
        end
    end

    AxiomDefinition(input_type, output_type, layers, ensures, proves)
end

function _parse_type_annotation(expr::Expr)
    @assert expr.head == :(::)
    name, typ = expr.args
    return name, typ
end

function _parse_assignment(expr::Expr)
    @assert expr.head == :(=)
    name, value = expr.args
    return name, value
end

function _parse_macro_call(expr::Expr)
    @assert expr.head == :macrocall
    macro_name = string(expr.args[1])
    if macro_name == "@ensure"
        return :ensure, expr.args[3]  # Skip macro name and line number
    elseif macro_name == "@prove"
        return :prove, expr.args[3]
    else
        error("Unsupported macro in @axiom block: $(macro_name)")
    end
end

"""
    generate_axiom_code(name::Symbol, def::AxiomDefinition) -> Expr

Generate the Julia code for a model struct and its methods based on the
parsed `AxiomDefinition`. This includes the struct definition, layer
initialization, the forward pass function, and any `@ensure` checks.
"""
function generate_axiom_code(name::Symbol, def::AxiomDefinition)
    layer_fields, layer_inits = _generate_layer_fields_and_inits(def)
    persistent_layer_names = [field.args[1] for field in layer_fields]
    forward_body = _generate_forward_body(def)
    ensure_checks = _generate_ensure_checks(def)

    # Generate the struct and methods
    quote
        struct $(esc(name)) <: AxiomModel
            $(layer_fields...)

            function $(esc(name))()
                $(layer_inits...)
                new($([:($n) for n in persistent_layer_names]...))
            end
        end

        function (m::$(esc(name)))(input)
            $(forward_body...)
            $(ensure_checks...)
            return output
        end

        # Store metadata for verification
        Axiom._axiom_metadata[$(QuoteNode(name))] = $(QuoteNode(def))
    end
end

function _generate_layer_fields_and_inits(def::AxiomDefinition)
    layer_fields = []
    layer_inits = []

    for (layer_name, layer_expr) in def.layers
        # Skip 'output' as it's a computed value
        layer_name == :output && continue

        # Persist only constructor-safe expressions that do not depend on `input`.
        if !is_pipeline_expr(layer_expr) && !_expr_mentions_symbol(layer_expr, :input)
            push!(layer_fields, :($layer_name::Any))
            push!(layer_inits, :($layer_name = $(esc(layer_expr))))
        end
    end
    return layer_fields, layer_inits
end

function _expr_mentions_symbol(expr, sym::Symbol)
    if expr isa Symbol
        return expr == sym
    elseif expr isa Expr
        return any(arg -> _expr_mentions_symbol(arg, sym), expr.args)
    else
        return false
    end
end

function _generate_forward_body(def::AxiomDefinition)
    forward_body = []
    for (layer_name, layer_expr) in def.layers
        push!(forward_body, :($layer_name = $(transform_pipeline(layer_expr))))
    end
    return forward_body
end

function _generate_ensure_checks(def::AxiomDefinition)
    ensure_checks = []
    for ensure_expr in def.ensures
        check_name = gensym("ensure")
        push!(ensure_checks, quote
            $check_name = $(esc(ensure_expr))
            if !$check_name
                throw(AxiomViolation($(string(ensure_expr)), "Ensure condition failed"))
            end
        end)
    end
    return ensure_checks
end

"""
    is_pipeline_expr(expr) -> Bool

Check if a given expression is a pipeline expression using the `|>` operator.
"""
function is_pipeline_expr(expr::Union{Expr, Symbol})
    expr isa Expr && expr.head == :call && expr.args[1] == :|>
end

"""
    transform_pipeline(expr::Expr) -> Expr

Recursively transform a pipeline expression `a |> b |> c` into nested
function calls `c(b(a))`.
"""
function transform_pipeline(expr::Union{Expr, Symbol})
    if !is_pipeline_expr(expr)
        return esc(expr)
    end

    input, layer = expr.args[2], expr.args[3]

    # Recursively transform nested pipelines
    if is_pipeline_expr(input)
        input = transform_pipeline(input)
    else
        input = esc(input)
    end

    # Apply the layer
    :($layer($input))
end

"""
Base type for all Axiom models.
"""
abstract type AxiomModel end

"""
Exception for axiom violations.
"""
struct AxiomViolation <: Exception
    property::String
    message::String
end

function Base.showerror(io::IO, e::AxiomViolation)
    println(io, "Axiom Violation: $(e.message)")
    println(io, "  Property: $(e.property)")
end

# Global metadata storage
const _axiom_metadata = Dict{Symbol, AxiomDefinition}()

# Convenience macro for quick model definition
"""
    @model name layers...

Quick model definition without full @axiom ceremony.

# Example
```julia
model = @model Sequential(
    Flatten,
    Dense(784, 128, relu),
    Dense(128, 10),
    Softmax
)
```
"""
macro model(expr)
    if expr.head == :call && expr.args[1] == :Sequential
        layers = expr.args[2:end]
        return :(Sequential($(map(esc, layers)...)))
    end
    error("@model expects Sequential(...) syntax")
end
