# SPDX-License-Identifier: MPL-2.0
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

# ===========================================================================
# Compile-time shape verification
#
# Walks the declared layer chain at MACRO-EXPANSION time, tracking the tensor
# shape from the `input ::` declaration through each layer assignment. A
# *provable* shape mismatch — e.g. a `Dense` whose declared `in_features`
# disagrees with the running feature dimension, or a computed `output` that
# contradicts the `output ::` declaration — raises an error during expansion,
# i.e. a genuine compile-time error before the model is constructed or run.
#
# Sound-but-incomplete BY DESIGN. A running shape is either a concrete
# `Vector` of dims (each an `Int`, or `:dynamic` for a `:batch`/`:dynamic`
# wildcard) or `nothing` (statically unknown). Anything that cannot be
# resolved from literals — Conv/Pool output geometry, non-literal layer args,
# unrecognised layers — collapses the running shape to `nothing` and is passed
# through unchecked, so a *valid* model can never receive a false compile
# error. Only definite, literal-provable contradictions are rejected.
# ===========================================================================

# Layer names known to preserve tensor shape (activations / normalisation).
const _SHAPE_PRESERVING_LAYERS = Set(Symbol.([
    "Softmax", "LogSoftmax", "relu", "ReLU", "sigmoid", "Sigmoid", "tanh",
    "Tanh", "gelu", "GELU", "elu", "ELU", "swish", "SiLU", "leakyrelu",
    "LeakyReLU", "Dropout", "BatchNorm", "LayerNorm", "GroupNorm", "Identity",
    "softplus", "mish", "Mish",
]))

# Extract the shape dims from an `input ::` / `output ::` type AST of the form
# `Tensor{ElemType, (d1, d2, ...)}`. Returns a Vector of dims (Int or :dynamic)
# or `nothing` if it is not in the recognised 2-parameter shorthand form.
function _extract_shape_dims(typ)
    typ isa Expr || return nothing
    typ.head === :curly || return nothing
    length(typ.args) >= 3 || return nothing          # Tensor, ElemType, shape
    shape_ast = typ.args[3]
    (shape_ast isa Expr && shape_ast.head === :tuple) || return nothing
    return Any[_norm_dim(d) for d in shape_ast.args]
end

# Normalise one dim AST node to an Int, or :dynamic for any symbolic/wildcard
# dim (:batch, :dynamic, :, or any other symbol). Unknown → :dynamic keeps the
# analysis sound (a wildcard never triggers a mismatch).
function _norm_dim(d)
    d isa Integer && return Int(d)
    if d isa QuoteNode
        return :dynamic
    end
    return :dynamic
end

# (head, args) for a layer application: `Dense(784,256,relu)` -> (:Dense, [784,256,relu]);
# a bare `Softmax` symbol -> (:Softmax, []); anything else -> (nothing, []).
function _layer_head_args(layer)
    if layer isa Symbol
        return (layer, Any[])
    elseif layer isa Expr && layer.head === :call
        return (layer.args[1], layer.args[2:end])
    else
        return (nothing, Any[])
    end
end

_lit_int(x) = x isa Integer ? Int(x) : nothing

# Apply one layer to the running shape. Returns (new_shape, mismatch_msg_or_nothing).
# `shape === nothing` means statically unknown: never checked, stays unknown.
function _shape_after_layer(layer, shape)
    head, args = _layer_head_args(layer)

    if head === :Dense
        in_f = length(args) >= 1 ? _lit_int(args[1]) : nothing
        out_f = length(args) >= 2 ? _lit_int(args[2]) : nothing
        shape === nothing && return (nothing, nothing)
        if in_f !== nothing && !isempty(shape)
            last_dim = shape[end]
            if last_dim isa Int && last_dim != in_f
                return (shape, "Dense layer expects $in_f input feature(s), but the incoming tensor has $last_dim")
            end
        end
        # Output feature dim = out_f if known, else unknown (:dynamic).
        isempty(shape) && return (nothing, nothing)
        new_last = out_f === nothing ? :dynamic : out_f
        return (Any[shape[1:end-1]..., new_last], nothing)

    elseif head === :Flatten || layer === :Flatten
        shape === nothing && return (nothing, nothing)
        # Only resolvable under the (batch, dims...) convention: a wildcard
        # leading dim with all-Int trailing dims. Otherwise -> unknown.
        if length(shape) >= 2 && shape[1] === :dynamic && all(x -> x isa Int, shape[2:end])
            return (Any[:dynamic, prod(shape[2:end])], nothing)
        end
        return (nothing, nothing)

    elseif head in _SHAPE_PRESERVING_LAYERS
        return (shape, nothing)                        # shape unchanged (may be nothing)

    else
        # Conv/Pool geometry and any unrecognised layer: cannot resolve
        # soundly -> collapse to unknown so nothing downstream false-errors.
        return (nothing, nothing)
    end
end

# Flatten a pipeline AST `a |> L1 |> L2 |> ...` into (base, [L1, L2, ...]).
function _flatten_pipeline(expr)
    layers = Any[]
    cur = expr
    while is_pipeline_expr(cur)
        pushfirst!(layers, cur.args[3])
        cur = cur.args[2]
    end
    return (cur, layers)
end

_fmt_shape(dims) = "(" * join(map(d -> d === :dynamic ? ":batch" : string(d), dims), ", ") * ")"

"""
    _verify_axiom_shapes(name::Symbol, def::AxiomDefinition)

Compile-time shape check for an `@axiom` body. Raises an error during macro
expansion on a provable shape mismatch; returns silently otherwise (including
when the shapes are not statically resolvable).
"""
function _verify_axiom_shapes(name::Symbol, def::AxiomDefinition)
    in_dims = def.input_type === nothing ? nothing : _extract_shape_dims(def.input_type)
    in_dims === nothing && return nothing            # no parseable input shape

    shapes = Dict{Symbol, Any}(:input => in_dims)

    for (lname, lexpr) in def.layers
        base, chain = _flatten_pipeline(lexpr)
        (base isa Symbol && haskey(shapes, base)) || continue
        cur = shapes[base]
        for layer in chain
            new_shape, msg = _shape_after_layer(layer, cur)
            if msg !== nothing
                error("@axiom $(name): compile-time shape mismatch in `$(lname)`.\n" *
                      "  $msg.\n" *
                      "  Running shape entering this layer: $(_fmt_shape(cur)).")
            end
            cur = new_shape
        end
        shapes[lname] = cur
    end

    # Computed `output` vs declared `output ::` — reject a definite contradiction.
    if haskey(shapes, :output) && shapes[:output] isa Vector && def.output_type !== nothing
        out_dims = _extract_shape_dims(def.output_type)
        computed = shapes[:output]
        if out_dims !== nothing && length(out_dims) == length(computed)
            for (d_decl, d_comp) in zip(out_dims, computed)
                if d_decl isa Int && d_comp isa Int && d_decl != d_comp
                    error("@axiom $(name): declared output shape $(_fmt_shape(out_dims)) " *
                          "contradicts the computed output shape $(_fmt_shape(computed)).")
                end
            end
        end
    end
    return nothing
end

"""
    generate_axiom_code(name::Symbol, def::AxiomDefinition) -> Expr

Generate the Julia code for a model struct and its methods based on the
parsed `AxiomDefinition`. This includes the struct definition, layer
initialization, the forward pass function, and any `@ensure` checks.
"""
function generate_axiom_code(name::Symbol, def::AxiomDefinition)
    # Compile-time shape verification: runs NOW, during macro expansion, so a
    # provable shape mismatch is a genuine compile-time error (before the model
    # is ever constructed or run). Sound-but-incomplete — see _verify_axiom_shapes.
    _verify_axiom_shapes(name, def)

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
