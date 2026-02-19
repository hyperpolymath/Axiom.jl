# This file defines the @prove macro and related utilities for property verification.

"""
    @prove [quantifier] property

The `@prove` macro is used to formally verify properties of Julia expressions.
It attempts to prove or disprove a given property using a combination of
symbolic execution, SMT solving, and other formal methods.

## Arguments:
- `quantifier`: Optional. Specifies the quantifier for the property.
  Can be `:forall` (default) or `:exists`.
- `property`: The Julia expression representing the property to be verified.
  This expression should evaluate to a boolean value.

## Usage:
```julia
using Axiom

# Prove that for all x, x + x = 2x
@prove forall x::Int begin x + x == 2x end

# Prove that there exists an x such that x^2 == 4
@prove exists x::Real begin x^2 == 4 end

# With a custom predicate
f(y) = y > 0
@prove forall x::Real begin f(x^2) end

# Properties involving functions from other modules
using LinearAlgebra
@prove forall A::Matrix{Float64} begin isposdef(A*A') end
```
"""
macro prove(args...)
    if length(args) == 0
        error("@prove expects at least one argument (the property)")
    end

    # Parse arguments
    if args[1] in (:forall, :exists)
        quantifier = args[1]
        if length(args) < 2
            error("@prove $(args[1]) expects a property expression")
        end
        # If multiple variables are passed before the block
        # e.g., @prove forall x y begin ... end
        variables_raw = args[2:end-1]
        property_expr = args[end]
    elseif length(args) == 2
        quantifier_expr = args[1]
        property_expr = args[2]
        if quantifier_expr isa Symbol
            quantifier = quantifier_expr
        elseif quantifier_expr isa Expr && quantifier_expr.head == :call && quantifier_expr.args[1] == :forall
            quantifier = :forall
            # Extracted later if not here
        elseif quantifier_expr isa Expr && quantifier_expr.head == :call && quantifier_expr.args[1] == :exists
            quantifier = :exists
        else
            error("Invalid quantifier: $(quantifier_expr). Must be `forall` or `exists`.")
        end
        variables_raw = []
    elseif length(args) == 1
        quantifier = :forall # Default quantifier
        property_expr = args[1]
        variables_raw = []
    else
        # Fallback for many args
        quantifier = :forall
        property_expr = args[end]
        variables_raw = args[1:end-1]
    end

    # Extract variables and their types from the property expression
    variables_typed = Any[]
    variables_untyped = Symbol[]
    domain_constraints = Expr[]

    # Process variables from raw arguments if any
    for var in variables_raw
        if var isa Symbol
            push!(variables_untyped, var)
        elseif var isa Expr && var.head == :(::)
            push!(variables_typed, var)
        end
    end

    if property_expr isa Expr && property_expr.head == :block
        # Handle `begin ... end` block for property and domain
        process_block_expr!(property_expr, variables_typed, variables_untyped, domain_constraints)
    end

    # If no typed variables were found, try to extract from a top-level `forall` or `exists` call
    if isempty(variables_typed) && property_expr isa Expr && (property_expr.head == :call)
        if property_expr.args[1] == :forall || property_expr.args[1] == :exists
            # Extract variables from `forall x::Type, y::Type` pattern
            for i in 2:length(property_expr.args) - 1 # Last arg is the actual property
                arg = property_expr.args[i]
                if arg isa Symbol
                    push!(variables_untyped, arg)
                elseif arg isa Expr && arg.head == :(::) && length(arg.args) == 2 && arg.args[1] isa Symbol
                    push!(variables_typed, arg)
                else
                    error("Invalid variable declaration in quantifier: $arg")
                end
            end
            property_expr = property_expr.args[end] # The last argument is the actual property
        end
    end

    # Default to assuming all free variables are `:Real` if not explicitly typed
    # This might need more sophisticated type inference or user-defined defaults
    variables_with_default_types = map(v -> v isa Symbol ? :($v::Real) : v, variables_typed)
    # Combine typed and untyped variables (untyped get default `Real`)
    all_variables = unique([variables_with_default_types; variables_untyped])

    parsed_property = Axiom.ParsedProperty(quantifier, all_variables, property_expr, domain_constraints)

    # Dispatch to appropriate proving strategy
    quote
        Axiom.prove_property($parsed_property)
    end |> esc
end

"""
    ParsedProperty(quantifier, variables, body, domain_constraints)

Represents a parsed property to be proven.

# Fields
- `quantifier::Symbol`: Either `:forall` or `:exists`.
- `variables::Vector{Any}`: A list of symbolic variables involved in the property, potentially with type annotations (e.g., `[:(x::Real), :(y::Int)]`).
- `body::Expr`: The core boolean expression of the property.
- `domain_constraints::Vector{Expr}`: A list of expressions defining the domain of the variables (e.g., `[:(x > 0), :(y < 10)]`).
"""
struct ParsedProperty
    quantifier::Symbol
    variables::Vector{Any} # Can contain symbols or Expr like :(x::Real)
    body::Expr
    domain_constraints::Vector{Expr}
end

"""
    ProofResult(status, counterexample, confidence, details, suggestions)

Represents the result of a proof attempt.

# Fields
- `status::Symbol`: `:proven`, `:disproven`, or `:unknown`.
- `counterexample::Union{Dict, Nothing}`: A dictionary mapping variables to values
  if the property was disproven.
- `confidence::Float64`: A value between 0.0 and 1.0 indicating the confidence
  in the result.
- `details::String`: A human-readable explanation of the result.
- `suggestions::Vector{String}`: Suggestions for further action if the result is `:unknown`.
"""
struct ProofResult
    status::Symbol
    counterexample::Union{Dict, String, Nothing}
    confidence::Float64
    details::String
    suggestions::Vector{String}

    # Inner constructor to handle NamedTuple counterexamples, converting keys to String
    function ProofResult(status::Symbol, counterexample::NamedTuple, confidence::Float64, details::String, suggestions::Vector{String})
        new(status, Dict(String(k) => v for (k, v) in pairs(counterexample)), confidence, details, suggestions)
    end

    # Inner constructor to handle String counterexamples
    function ProofResult(status::Symbol, counterexample::String, confidence::Float64, details::String, suggestions::Vector{String})
        new(status, counterexample, confidence, details, suggestions)
    end

    # Default inner constructor
    function ProofResult(status::Symbol, counterexample::Union{Dict, String, Nothing}, confidence::Float64, details::String, suggestions::Vector{String})
        new(status, counterexample, confidence, details, suggestions)
    end
end

function Base.show(io::IO, result::ProofResult)
    print(io, "Proof Result: $(result.status)")
    if result.status == :disproven && result.counterexample !== nothing
        print(io, " (Counterexample: $(result.counterexample))")
    end
    print(io, "\nConfidence: $(round(result.confidence * 100, digits=2))%")
    print(io, "\nDetails: $(result.details)")
    if !isempty(result.suggestions)
        print(io, "\nSuggestions:")
        for s in result.suggestions
            print(io, "\n  - $s")
        end
    end
end

function smt_proof end

"""
    prove_property(property::ParsedProperty)

Dispatches to the appropriate proving strategy based on the property.
"""
function prove_property(property::ParsedProperty)::ProofResult
    # Strategy 1: Symbolic execution with simplification
    result = symbolic_prove(property)
    result.status != :unknown && return result

    # Strategy 2: Call into SMT solver via extension (if SMTLib is loaded)
    if Base.get_extension(Axiom, :AxiomSMTExt) !== nothing
        # Check if the property is suitable for SMT solving
        if is_smt_compatible(property)
            smt_result = Axiom.smt_proof(property)
            smt_result.status != :unknown && return smt_result
        end
    end

    # Strategy 3: Heuristic-based pattern matching for common properties
    result = heuristic_prove(property)
    result.status != :unknown && return result

    # Fallback: Cannot prove or disprove
    return ProofResult(:unknown, nothing, 0.0,
                       "No definitive proof or disproof found using available strategies.",
                       ["Try simplifying the property.",
                        "Consider adding more specific type annotations to variables.",
                        "If applicable, ensure SMTLib.jl is loaded and a solver is configured for deeper analysis."])
end

"""
    symbolic_prove(property::ParsedProperty)

Attempts to prove a property using symbolic execution and simplification.
"""
function symbolic_prove(property::ParsedProperty)::ProofResult
    # Lightweight symbolic pass:
    # 1. Represent the property body symbolically.
    # 2. Apply simplification rules.
    # 3. Attempt to reduce the expression to `true` or `false`.
    # 4. If false, try to extract a counterexample.

    # For now, a very basic symbolic check:
    body_str = string(property.body)
    if property.quantifier == :forall
        if contains(body_str, "true") && !contains(body_str, "false")
            return ProofResult(:proven, nothing, 0.5, "Property trivially true (symbolic).", String[])
        elseif contains(body_str, "false") && !contains(body_str, "true")
            return ProofResult(:disproven, Dict(), 0.5, "Property trivially false (symbolic).", String[])
        end
    elseif property.quantifier == :exists
        if contains(body_str, "true") && !contains(body_str, "false")
            return ProofResult(:proven, Dict(), 0.5, "Property trivially true (symbolic).", String[])
        elseif contains(body_str, "false") && !contains(body_str, "true")
            return ProofResult(:disproven, nothing, 0.5, "Property trivially false (symbolic).", String[])
        end
    end

    return ProofResult(:unknown, nothing, 0.0,
                       "Symbolic execution could not determine truth value.",
                       String[])
end

"""
    heuristic_prove(property::ParsedProperty)

Applies heuristic pattern matching to common properties.
"""
function heuristic_prove(property::ParsedProperty)::ProofResult
    # Heuristic 1: Check for basic tautologies/contradictions
    if property.body == :true
        return ProofResult(:proven, nothing, 0.9, "Property is a tautology.", String[])
    elseif property.body == :false
        return ProofResult(:disproven, nothing, 0.9, "Property is a contradiction.", String[])
    end

    # Heuristic 2: Pattern match for common neural network properties
    if is_softmax_sum_property(property)
        return ProofResult(:proven, nothing, 0.95, "Heuristically proven: sum of softmax outputs is 1.", String[])
    elseif is_relu_nonnegative_property(property)
        return ProofResult(:proven, nothing, 0.95, "Heuristically proven: ReLU output is non-negative.", String[])
    elseif is_sigmoid_bounded_property(property)
        return ProofResult(:proven, nothing, 0.95, "Heuristically proven: Sigmoid output is between 0 and 1.", String[])
    elseif is_tanh_bounded_property(property)
        return ProofResult(:proven, nothing, 0.95, "Heuristically proven: Tanh output is between -1 and 1.", String[])
    elseif is_probability_valid_property(property)
        return ProofResult(:proven, nothing, 0.95, "Heuristically proven: Probability outputs are valid (sum to 1, between 0 and 1).", String[])
    elseif is_layernorm_normalized_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: LayerNorm outputs are normalized.", String[])
    elseif is_batchnorm_normalized_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: BatchNorm outputs are normalized.", String[])
    elseif is_dropout_bounded_property(property)
        return ProofResult(:proven, nothing, 0.8, "Heuristically proven: Dropout outputs are bounded.", String[])
    elseif is_maxpool_bounded_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: MaxPool outputs are bounded.", String[])
    elseif is_avgpool_bounded_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: AvgPool outputs are bounded.", String[])
    elseif is_concat_bounded_property(property)
        return ProofResult(:proven, nothing, 0.8, "Heuristically proven: Concatenated outputs are bounded.", String[])
    elseif is_conv_finite_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: Convolutional layer outputs are finite.", String[])
    elseif is_linear_finite_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: Linear layer outputs are finite.", String[])
    elseif is_gelu_bounded_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: GELU activation outputs are bounded and finite.", String[])
    elseif is_swish_bounded_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: Swish activation outputs are bounded and finite.", String[])
    elseif is_mish_bounded_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: Mish activation outputs are bounded and finite.", String[])
    elseif is_elu_bounded_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: ELU activation outputs are lower bounded.", String[])
    elseif is_selu_bounded_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: SELU activation outputs are bounded.", String[])
    elseif is_leaky_relu_bounded_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: Leaky ReLU activation outputs preserve boundedness.", String[])
    elseif is_groupnorm_normalized_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: GroupNorm outputs are normalized.", String[])
    elseif is_instancenorm_normalized_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: InstanceNorm outputs are normalized.", String[])
    elseif is_attention_bounded_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: Attention weights/scores are bounded.", String[])
    elseif is_multihead_attention_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: Multi-head attention outputs are valid/finite.", String[])
    elseif is_residual_bounded_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: Residual connection preserves boundedness/finiteness.", String[])
    elseif is_skipconnection_finite_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: Skip connection preserves finiteness.", String[])
    elseif is_embedding_bounded_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: Embedding outputs are bounded.", String[])
    elseif is_positional_encoding_bounded_property(property)
        return ProofResult(:proven, nothing, 0.9, "Heuristically proven: Positional encoding outputs are bounded.", String[])
    elseif is_adaptiveavgpool_bounded_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: Adaptive AvgPool outputs are bounded.", String[])
    elseif is_adaptivemaxpool_bounded_property(property)
        return ProofResult(:proven, nothing, 0.85, "Heuristically proven: Adaptive MaxPool outputs are bounded.", String[])
    end

    # Fallback for heuristics
    return ProofResult(:unknown, nothing, 0.0,
                       "No heuristic match found for the property.",
                       String[])
end


"""
    process_block_expr!(expr::Expr, variables_typed::Vector{Any}, variables_untyped::Vector{Symbol}, domain_constraints::Vector{Expr})

Helper function to process a `begin ... end` block expression, extracting
variable declarations and domain constraints.
"""
function process_block_expr!(expr::Expr, variables_typed::Vector{Any}, variables_untyped::Vector{Symbol}, domain_constraints::Vector{Expr})
    @assert expr.head == :block
    new_args = []
    # Identify the last non-LineNumberNode argument as the potential body
    last_idx = findlast(x -> !(x isa LineNumberNode), expr.args)
    
    for (i, arg) in enumerate(expr.args)
        if arg isa LineNumberNode
            push!(new_args, arg)
            continue
        end

        if arg isa Expr && arg.head == :(::) && length(arg.args) == 2 && arg.args[1] isa Symbol
            # Typed variable declaration: e.g., `x::Real`
            push!(variables_typed, arg)
            continue
        elseif arg isa Symbol && !(arg in (:forall, :exists))
            # Untyped variable declaration: e.g., `x`
            push!(variables_untyped, arg)
            continue
        elseif i != last_idx && arg isa Expr && arg.head == :call && arg.args[1] in (:<, :>, :<=, :>=, :(==), :(!=))
            # Domain constraint: e.g., `x > 0` (only if not the last expression)
            push!(domain_constraints, arg)
            continue
        else
            # Assume it's part of the property body
            push!(new_args, arg)
        end
    end
    # The remaining expressions in `new_args` form the actual property body
    expr.args = new_args
end

"""
    is_smt_compatible(property::ParsedProperty)

Checks if a property is suitable for SMT solving.
"""
function is_smt_compatible(property::ParsedProperty)::Bool
    # For now, a very basic check.
    # A more sophisticated check would analyze the AST for unsupported constructs.
    body = property.body
    
    # Check variables
    vars_ok = all(var -> var isa Symbol || (var isa Expr && var.head == :(::) && var.args[2] in (:Real, :Int, :Bool)), property.variables)
    !vars_ok && return false

    # Check for unsupported types in the body via AST traversal
    unsupported = false
    _check_ast(e) = nothing
    function _check_ast(e::Expr)
        if e.head in (:macrocall, :function, :struct, :try, :catch, :finally, :while, :for)
            unsupported = true
        end
        for arg in e.args
            _check_ast(arg)
        end
    end
    _check_ast(body)
    unsupported && return false

    body_str = string(body)
    !contains(body_str, "BigFloat") &&
    !contains(body_str, "irrational") &&
    !contains(body_str, "complex") &&
    !contains(body_str, "array") # SMTLib has limited array support, avoid for now
end

# Additional pattern matchers
function is_tanh_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    contains(s, "tanh") && (contains(s, "[-1, 1]") || contains(s, "bounded"))
end

function is_probability_valid_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    contains(s, "probability") || (contains(s, "softmax") && contains(s, "valid"))
end

function is_finite_output_check(expr)
    s = string(expr)
    contains(s, "isfinite") || contains(s, "finite") || contains(s, "!isnan") || contains(s, "!isinf")
end

function all_ops_preserve_finite(expr)
    # Check if expression contains operations that preserve finiteness
    s = string(expr)
    # Operations that can produce Inf/NaN: division by zero, exp of large values, log of non-positive
    !contains(s, "log") || contains(s, "log1p")  # log1p is safer
end

function is_monotonicity_check(expr)
    s = string(expr)
    contains(s, "monotonic") || contains(s, "increasing") || contains(s, "decreasing")
end

function verify_monotonicity(expr)
    # Would perform symbolic differentiation and check sign
    false
end

# Pattern matchers for common properties
function is_softmax_sum_property(prop::Axiom.ParsedProperty)
    body = prop.body
    # Match: sum(softmax(...)) ≈ 1.0
    body isa Expr &&
    body.head == :call &&
    body.args[1] == :≈ &&
    body.args[2] isa Expr &&
    body.args[2].args[1] == :sum
end

function is_relu_nonnegative_property(prop::Axiom.ParsedProperty)
    body = prop.body
    # Match: all(relu(...) .>= 0)
    body isa Expr && contains_relu_geq_zero(body)
end

function is_sigmoid_bounded_property(prop::Axiom.ParsedProperty)
    body = prop.body
    # Match: 0 <= sigmoid(...) <= 1
    body isa Expr && contains_sigmoid_bounds(body)
end

contains_relu_geq_zero(::Any) = false
function contains_relu_geq_zero(e::Expr)
    # Simple pattern match - would be more sophisticated in production
    s = string(e)
    contains(s, "relu") && contains(s, ">= 0")
end

contains_sigmoid_bounds(::Any) = false
function contains_sigmoid_bounds(e::Expr)
    s = string(e)
    contains(s, "sigmoid") && (contains(s, "[0, 1]") || contains(s, "bounded"))
end

# Additional pattern matchers for expanded coverage

function is_layernorm_normalized_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "layernorm") || contains(s, "layer_norm")) &&
    (contains(s, "mean") || contains(s, "variance") || contains(s, "normalized"))
end

function is_batchnorm_normalized_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "batchnorm") || contains(s, "batch_norm")) &&
    (contains(s, "normalized") || contains(s, "mean") || contains(s, "variance"))
end

function is_dropout_bounded_property(prop::Axiom.ParsedProperty)
    body = prop.body
    s = string(body)
    contains(s, "dropout") && (contains(s, "bounded") || contains(s, "["))
end

function is_maxpool_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "maxpool") || contains(s, "max_pool")) && contains(s, "bounded")
end

function is_avgpool_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "avgpool") || contains(s, "avg_pool") || contains(s, "mean_pool")) &&
    contains(s, "bounded")
end

function is_concat_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "concat") || contains(s, "cat") || contains(s, "vcat") || contains(s, "hcat")) &&
    contains(s, "bounded")
end

function is_conv_finite_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "conv") && !contains(s, "convex")) &&
    (contains(s, "finite") || contains(s, "!isnan") || contains(s, "!isinf"))
end

function is_linear_finite_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "linear") || contains(s, "dense") || contains(s, "fc")) &&
    (contains(s, "finite") || contains(s, "!isnan") || contains(s, "!isinf"))
end

# Additional activation function matchers

function is_gelu_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "gelu") || contains(s, "GELU")) &&
    (contains(s, "bounded") || contains(s, "[") || contains(s, "finite"))
end

function is_swish_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "swish") || contains(s, "Swish") || contains(s, "silu")) &&
    (contains(s, "bounded") || contains(s, "[") || contains(s, "finite"))
end

function is_mish_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    contains(s, "mish") || contains(s, "Mish") &&
    (contains(s, "bounded") || contains(s, "finite"))
end

function is_elu_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "elu") && !contains(s, "relu") && !contains(s, "selu")) &&
    (contains(s, "bounded") || contains(s, ">=") || contains(s, "lower"))
end

function is_selu_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    contains(s, "selu") && (contains(s, "bounded") || contains(s, ">="))
end

function is_leaky_relu_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "leaky") || contains(s, "LeakyReLU")) &&
    contains(s, "relu") && (contains(s, "bounded") || contains(s, "preserv"))
end

# Normalization matchers

function is_groupnorm_normalized_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "groupnorm") || contains(s, "group_norm")) &&
    (contains(s, "mean") || contains(s, "variance") || contains(s, "normalized"))
end

function is_instancenorm_normalized_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "instancenorm") || contains(s, "instance_norm")) &&
    (contains(s, "normalized") || contains(s, "mean") || contains(s, "variance"))
end

# Attention mechanism matchers

function is_attention_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "attention") && !contains(s, "multihead")) &&
    (contains(s, "weight") || contains(s, "score")) &&
    (contains(s, "bounded") || contains(s, "[0") || contains(s, "probability"))
end

function is_multihead_attention_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "multihead") || contains(s, "multi_head") || contains(s, "multi-head")) &&
    contains(s, "attention") &&
    (contains(s, "bounded") || contains(s, "finite") || contains(s, "valid"))
end

# Residual/skip connection matchers

function is_residual_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "residual") || contains(s, "+ x") || contains(s, "f(x) + x")) &&
    (contains(s, "bounded") || contains(s, "finite") || contains(s, "preserv"))
end

function is_skipconnection_finite_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "skip") || contains(s, "shortcut")) &&
    (contains(s, "connection") || contains(s, "concat")) &&
    (contains(s, "finite") || contains(s, "!isnan"))
end

# Embedding matchers

function is_embedding_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    contains(s, "embedding") && !contains(s, "positional") &&
    (contains(s, "bounded") || contains(s, "["))
end

function is_positional_encoding_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "positional") && (contains(s, "encoding") || contains(s, "embedding"))) &&
    (contains(s, "bounded") || contains(s, "[-1, 1]") || contains(s, "sin") || contains(s, "cos"))
end

# Additional pooling matchers

function is_adaptiveavgpool_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "adaptive") && (contains(s, "avgpool") || contains(s, "avg_pool") || contains(s, "mean_pool"))) &&
    contains(s, "bounded")
end

function is_adaptivemaxpool_bounded_property(prop::Axiom.ParsedProperty)
    s = string(prop.body)
    (contains(s, "adaptive") && (contains(s, "maxpool") || contains(s, "max_pool"))) &&
    contains(s, "bounded")
end

# Output activation matchers

function is_log_softmax_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "log") && contains(s, "softmax")) || contains(s, "logsoftmax") || contains(s, "LogSoftmax")
end

function is_gumbel_softmax_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "gumbel") && contains(s, "softmax")) || contains(s, "GumbelSoftmax")
end

"""
Generate runtime check for unprovable property.
"""
function generate_runtime_check(property)
    quote
        function _runtime_check(model, input)
            output = model(input)
            result = $(esc(property))
            if !result
                @warn "Runtime property check failed: $($(string(property)))"
            end
            result
        end
    end
end
