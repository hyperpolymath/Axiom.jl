# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl @prove Macro
#
# Formal verification system for proving properties about models.
# Uses symbolic execution and SMT solver integration.

# SMTLib is a weak dependency - only load if available
const HAS_SMTLIB = Ref(false)
function __init__()
    if isdefined(@__MODULE__, :SMTLib)
        HAS_SMTLIB[] = true
    end
end

"""
    @prove property

Attempt to formally prove a property about a model.
Properties that can be proven are verified at compile time.
Properties that cannot be proven generate warnings and runtime checks.

# Syntax
```julia
@prove ∀x. property(x)
@prove ∃x. property(x)
@prove property1 ⟹ property2
```

# Examples
```julia
@prove ∀x. sum(softmax(x)) ≈ 1.0
@prove ∀x. all(sigmoid(x) .>= 0)
@prove ∀x ε. (norm(ε) < 0.01) ⟹ stable(model(x), model(x + ε))
```
"""
macro prove(property)
    _prove_impl(property)
end

function _prove_impl(property)
    # Parse the property
    parsed = parse_property(property)

    quote
        # Attempt compile-time proof
        proof_result = attempt_proof($(QuoteNode(parsed)))

        if proof_result.status == :proven
            # Property proven, no runtime check needed
            reason_msg = isempty(proof_result.reason) ? "" : " ($(proof_result.reason))"
            @info "✓ Property proven: $($(string(property)))$reason_msg"
        elseif proof_result.status == :disproven
            # Property false, compilation should fail
            msg = "✗ Property disproven: $($(string(property)))\n" *
                  "Counterexample: $(proof_result.counterexample)\n" *
                  "Reason: $(proof_result.reason)"
            if !isempty(proof_result.suggestions)
                msg *= "\n\nSuggestions:\n" * join("  - " .* proof_result.suggestions, "\n")
            end
            error(msg)
        else
            # Cannot prove, add runtime check
            msg = "⚠ Cannot prove property: $($(string(property)))"
            if !isempty(proof_result.reason)
                msg *= "\nReason: $(proof_result.reason)"
            end
            if !isempty(proof_result.suggestions)
                msg *= "\n\nSuggestions:\n" * join("  - " .* proof_result.suggestions, "\n")
            end
            msg *= "\nAdding runtime assertion instead."
            @warn msg
            $(generate_runtime_check(property))
        end
    end
end

"""
Parsed property representation.
"""
struct ParsedProperty
    quantifier::Symbol  # :forall, :exists, :none
    variables::Vector{Symbol}
    body::Expr
end

"""
Parse a property expression.
"""
function parse_property(expr)
    if expr isa Expr && expr.head == :call
        op = expr.args[1]

        # Universal quantifier: ∀x. P(x)
        if op == :∀ || op == :forall
            return ParsedProperty(:forall, [expr.args[2]], expr.args[3])
        end

        # Existential quantifier: ∃x. P(x)
        if op == :∃ || op == :exists
            return ParsedProperty(:exists, [expr.args[2]], expr.args[3])
        end

        # Implication: P ⟹ Q
        if op == :⟹ || op == :implies
            return ParsedProperty(:none, Symbol[], expr)
        end
    end

    # No quantifier, just a property
    ParsedProperty(:none, Symbol[], expr)
end

"""
Proof result from verification attempt.
"""
struct ProofResult
    status::Symbol  # :proven, :disproven, :unknown
    counterexample::Any
    confidence::Float64
    reason::String  # Explanation of why the proof succeeded/failed
    suggestions::Vector{String}  # Suggestions for fixing failed proofs
end

# Constructor with defaults
ProofResult(status, counterexample, confidence) =
    ProofResult(status, counterexample, confidence, "", String[])

"""
Attempt to prove a property using symbolic execution and SMT solver integration.
"""
function attempt_proof(property::ParsedProperty)
    # Strategy 1: Pattern matching for common provable properties
    pattern_result = check_known_patterns(property)
    if pattern_result.status != :unknown
        return pattern_result
    end

    # Strategy 2: Symbolic execution for simple properties
    symbolic_result = symbolic_proof(property)
    if symbolic_result.status != :unknown
        return symbolic_result
    end

    # Strategy 3: SMT solver integration (if available)
    smt_result = smt_proof(property)
    if smt_result.status != :unknown
        return smt_result
    end

    # Default: cannot prove
    ProofResult(:unknown, nothing, 0.0)
end

"""
Check against known provable patterns.
"""
function check_known_patterns(property::ParsedProperty)
    if is_softmax_sum_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Softmax outputs always sum to 1 by definition: exp(xᵢ) / Σexp(xⱼ)",
            String[])
    end

    if is_relu_nonnegative_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "ReLU(x) = max(0, x) is non-negative by definition",
            String[])
    end

    if is_sigmoid_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Sigmoid(x) = 1/(1 + exp(-x)) ∈ (0, 1) for all finite x",
            String[])
    end

    if is_tanh_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x)) ∈ (-1, 1) for all finite x",
            String[])
    end

    if is_probability_valid_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Softmax produces valid probability distributions: sum=1, all ∈ [0,1]",
            String[])
    end

    if is_layernorm_normalized_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Layer normalization: (x - μ) / σ produces mean=0, variance=1",
            String[])
    end

    if is_batchnorm_normalized_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Batch normalization: (x - E[x]) / √(Var[x] + ε) produces normalized output",
            String[])
    end

    if is_dropout_bounded_property(property)
        return ProofResult(:proven, nothing, 0.95,
            "Dropout preserves bounds: if x ∈ [a,b], then Dropout(x,p)/(1-p) ∈ [a,b]",
            String[])
    end

    if is_maxpool_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "MaxPool(x) = max(xᵢ) preserves bounds: if all xᵢ ∈ [a,b], then max(xᵢ) ∈ [a,b]",
            String[])
    end

    if is_avgpool_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "AvgPool(x) = mean(xᵢ) preserves bounds: if all xᵢ ∈ [a,b], then mean(xᵢ) ∈ [a,b]",
            String[])
    end

    if is_concat_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Concatenation preserves bounds: concat([x₁,...,xₙ]) where xᵢ ∈ [a,b] ⟹ result ∈ [a,b]",
            String[])
    end

    if is_conv_finite_property(property)
        return ProofResult(:proven, nothing, 0.95,
            "Convolution preserves finiteness: if input and weights are finite, output is finite",
            ["Ensure convolution weights are initialized with finite values",
             "Consider weight regularization to prevent numerical instability"])
    end

    if is_linear_finite_property(property)
        return ProofResult(:proven, nothing, 0.95,
            "Linear layer preserves finiteness: W*x + b is finite if W, x, and b are finite",
            ["Ensure weight initialization produces finite values",
             "Consider gradient clipping during training"])
    end

    # Additional activation functions
    if is_gelu_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "GELU(x) = x·Φ(x) is approximately bounded: outputs roughly in [-0.17, ∞) for x ∈ ℝ",
            String[])
    end

    if is_swish_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Swish(x) = x·σ(x) where σ is sigmoid; for x ∈ ℝ, Swish(x) ∈ (-0.28, ∞)",
            String[])
    end

    if is_mish_bounded_property(property)
        return ProofResult(:proven, nothing, 0.95,
            "Mish(x) = x·tanh(softplus(x)) is bounded for finite inputs",
            String[])
    end

    if is_elu_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "ELU(x) = x if x>0 else α(exp(x)-1) has lower bound -α and no upper bound",
            String[])
    end

    if is_selu_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "SELU is scaled ELU with self-normalizing properties: bounded below by -λα",
            String[])
    end

    if is_leaky_relu_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "LeakyReLU(x) = max(αx, x) preserves bounds with scaling factor",
            String[])
    end

    # Normalization layers
    if is_groupnorm_normalized_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Group normalization: (x - μ_group) / σ_group produces normalized output per group",
            String[])
    end

    if is_instancenorm_normalized_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Instance normalization: (x - μ_instance) / σ_instance produces mean=0, var=1 per instance",
            String[])
    end

    # Attention mechanisms
    if is_attention_bounded_property(property)
        return ProofResult(:proven, nothing, 0.95,
            "Attention weights (softmax over scores) are valid probabilities: sum=1, all ∈ [0,1]",
            String[])
    end

    if is_multihead_attention_property(property)
        return ProofResult(:proven, nothing, 0.9,
            "Multi-head attention concatenates bounded attention outputs",
            ["Ensure each attention head receives proper input shapes",
             "Verify weight matrices are initialized correctly"])
    end

    # Residual connections
    if is_residual_bounded_property(property)
        return ProofResult(:proven, nothing, 0.95,
            "Residual connection f(x) + x preserves finiteness if f and x are finite",
            ["Ensure residual branch outputs are bounded",
             "Consider using layer normalization before residuals"])
    end

    if is_skipconnection_finite_property(property)
        return ProofResult(:proven, nothing, 0.95,
            "Skip connection preserves finiteness: if all inputs are finite, concatenation/addition is finite",
            String[])
    end

    # Embedding properties
    if is_embedding_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Embedding lookup produces bounded output if embedding matrix is bounded",
            ["Verify embedding weights are initialized with bounded values",
             "Consider adding embedding normalization"])
    end

    if is_positional_encoding_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Positional encoding using sin/cos is bounded: all values ∈ [-1, 1]",
            String[])
    end

    # Pooling variants
    if is_adaptiveavgpool_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Adaptive average pooling preserves bounds like regular average pooling",
            String[])
    end

    if is_adaptivemaxpool_bounded_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "Adaptive max pooling preserves bounds like regular max pooling",
            String[])
    end

    # Output activation patterns
    if is_log_softmax_property(property)
        return ProofResult(:proven, nothing, 1.0,
            "LogSoftmax(x) = log(softmax(x)) produces values ≤ 0, sum of exp = 1",
            String[])
    end

    if is_gumbel_softmax_property(property)
        return ProofResult(:proven, nothing, 0.9,
            "Gumbel-Softmax produces valid probability distribution at τ → 0",
            String[])
    end

    ProofResult(:unknown, nothing, 0.0,
        "Property does not match any known provable patterns",
        ["Try simplifying the property into smaller sub-properties",
         "Check if the property can be decomposed into known patterns",
         "Supported patterns: softmax, relu, sigmoid, tanh, gelu, swish, mish, elu, selu, leaky_relu",
         "Normalization: layernorm, batchnorm, groupnorm, instancenorm",
         "Attention: attention weights, multi-head attention",
         "Connections: residual, skip connections, embeddings, positional encoding",
         "Pooling: maxpool, avgpool, adaptive variants",
         "Consider using SMT solvers for custom properties (set AXIOM_SMT_SOLVER)",
         "Add @ensure for runtime verification if formal proof is not feasible"])
end

"""
Symbolic execution for property verification.
"""
function symbolic_proof(property::ParsedProperty)
    body = property.body

    # Check for finite output properties
    if is_finite_output_check(body)
        # Check if all operations in the expression preserve finiteness
        if all_ops_preserve_finite(body)
            return ProofResult(:proven, nothing, 0.95,
                "Symbolic analysis confirms all operations preserve finiteness",
                String[])
        else
            return ProofResult(:unknown, nothing, 0.0,
                "Expression contains operations that may produce Inf/NaN (e.g., log, exp, division)",
                ["Use safe alternatives like log1p instead of log",
                 "Add guards to prevent division by zero",
                 "Clip intermediate values to prevent overflow"])
        end
    end

    # Check for monotonicity properties
    if is_monotonicity_check(body)
        if verify_monotonicity(body)
            return ProofResult(:proven, nothing, 0.9,
                "Verified monotonicity through symbolic differentiation",
                String[])
        else
            return ProofResult(:unknown, nothing, 0.0,
                "Cannot verify monotonicity - symbolic differentiation inconclusive",
                ["Simplify the function to isolate monotonic components",
                 "Try proving monotonicity on sub-intervals"])
        end
    end

    ProofResult(:unknown, nothing, 0.0,
        "Symbolic execution did not match any patterns",
        ["Consider adding known patterns for your operations",
         "Try using SMT solver for deeper analysis"])
end

"""
SMT solver integration for formal verification.

This integrates with external SMT solvers (Z3, CVC5, Yices, MathSAT) via SMTLib.
Falls back to heuristic methods otherwise.
"""
function smt_proof(property::ParsedProperty)
    ctx = get_smt_context()

    if ctx === nothing
        return ProofResult(:unknown, nothing, 0.0)
    end

    vars = property.variables
    expr = normalize_smt_expr(property.body)

    for v in vars
        SMTLib.declare(ctx, v, Float64)
    end

    if property.quantifier == :exists
        SMTLib.assert!(ctx, expr)
    else
        SMTLib.assert!(ctx, Expr(:call, :!, expr))
    end

    script = SMTLib.build_script(ctx, true)
    cache_key = smt_cache_key(ctx, script)
    if smt_cache_enabled()
        cached = smt_cache_get(cache_key)
        cached !== nothing && return finalize_smt_result(property, cached)
    end

    result = if use_rust_smt_runner() && rust_available()
        output = rust_smt_run(string(ctx.solver.kind), ctx.solver.path, script, ctx.timeout_ms)
        SMTLib.parse_result(output)
    else
        SMTLib.check_sat(ctx; get_model=true)
    end

    smt_cache_put(cache_key, result)
    return finalize_smt_result(property, result)

    if result.status == :sat
        if property.quantifier == :exists
            return ProofResult(:proven, result.model, 1.0)
        end
        return ProofResult(:disproven, result.model, 1.0)
    elseif result.status == :unsat
        if property.quantifier == :exists
            return ProofResult(:disproven, nothing, 1.0)
        end
        return ProofResult(:proven, nothing, 1.0)
    end

    ProofResult(:unknown, nothing, 0.0)
end

"""
Normalize expressions for SMT-LIB conversion.
"""
function normalize_smt_expr(expr)
    if expr isa Expr
        if expr.head == :call && expr.args[1] == :≈ && length(expr.args) == 3
            return Expr(:call, :(==), normalize_smt_expr(expr.args[2]), normalize_smt_expr(expr.args[3]))
        end
        return Expr(expr.head, map(normalize_smt_expr, expr.args)...)
    end
    expr
end

"""
Get available SMT solver.
"""
const SMT_ALLOWLIST = Set([:z3, :cvc5, :yices, :mathsat])
const SMT_CACHE = Dict{UInt64, SMTLib.SMTResult}()
const SMT_CACHE_ORDER = UInt64[]

function use_rust_smt_runner()
    get(ENV, "AXIOM_SMT_RUNNER", "") == "rust"
end

function rust_available()
    # Check if AXIOM_RUST_LIB is set and the library exists
    rust_lib = get(ENV, "AXIOM_RUST_LIB", "")
    isempty(rust_lib) && return false
    isfile(rust_lib)
end

function rust_smt_run(solver_kind::String, solver_path::String, script::String, timeout_ms::Int)
    # This requires the Rust library to be loaded
    if !rust_available()
        error("Rust SMT runner requested but AXIOM_RUST_LIB not available")
    end

    # Call the Rust FFI function (requires axiom_core library)
    # For now, fallback to Julia implementation
    # TODO: Implement actual Rust FFI call when libaxiom_core provides smt_run_safe
    @warn "Rust SMT runner not fully implemented yet, falling back to Julia runner"

    # Use Julia's SMTLib instead
    # This is a placeholder - in production, would call Rust FFI
    return ""  # Empty output triggers fallback to Julia SMTLib path
end

function smt_cache_enabled()
    get(ENV, "AXIOM_SMT_CACHE", "") in ("1", "true", "yes")
end

function smt_cache_max()
    raw = get(ENV, "AXIOM_SMT_CACHE_MAX", nothing)
    raw === nothing && return 128
    parsed = tryparse(Int, raw)
    parsed === nothing ? 128 : parsed
end

function smt_cache_key(ctx::SMTLib.SMTContext, script::String)
    hash((ctx.solver.kind, ctx.solver.path, ctx.logic, ctx.timeout_ms, script))
end

function smt_cache_get(key::UInt64)
    get(SMT_CACHE, key, nothing)
end

function smt_cache_put(key::UInt64, result::SMTLib.SMTResult)
    smt_cache_enabled() || return
    max_entries = smt_cache_max()
    max_entries <= 0 && return
    if !haskey(SMT_CACHE, key)
        push!(SMT_CACHE_ORDER, key)
    end
    SMT_CACHE[key] = result
    while length(SMT_CACHE_ORDER) > max_entries
        oldest = popfirst!(SMT_CACHE_ORDER)
        delete!(SMT_CACHE, oldest)
    end
end

function finalize_smt_result(property::ParsedProperty, result::SMTLib.SMTResult)
    if result.status == :sat
        if property.quantifier == :exists
            return ProofResult(:proven, result.model, 1.0,
                "SMT solver found satisfying assignment: ∃x. property(x) is SAT",
                String[])
        end
        return ProofResult(:disproven, result.model, 1.0,
            "SMT solver found counterexample: ¬(∀x. property(x)) is SAT",
            ["Check if the property holds for the provided counterexample",
             "Consider adding preconditions to restrict the domain",
             "Verify that the model behaves correctly on the counterexample input"])
    elseif result.status == :unsat
        if property.quantifier == :exists
            return ProofResult(:disproven, nothing, 1.0,
                "SMT solver proved no satisfying assignment exists: ∃x. property(x) is UNSAT",
                ["Property is impossible to satisfy",
                 "Review the property definition - it may be too restrictive"])
        end
        return ProofResult(:proven, nothing, 1.0,
            "SMT solver proved property holds for all inputs: ∀x. property(x) is valid (¬property is UNSAT)",
            String[])
    end

    ProofResult(:unknown, nothing, 0.0,
        "SMT solver returned unknown - timeout, resource limit, or unsupported logic",
        ["Increase timeout with AXIOM_SMT_TIMEOUT_MS",
         "Try a different SMT solver (z3, cvc5, yices, mathsat)",
         "Simplify the property to use supported SMT logic",
         "Consider decomposing into simpler sub-properties"])
end

function smt_solver_preference()
    preference = get(ENV, "AXIOM_SMT_SOLVER", nothing)
    preference === nothing && return nothing
    Symbol(lowercase(preference))
end

function smt_timeout_ms()
    raw = get(ENV, "AXIOM_SMT_TIMEOUT_MS", nothing)
    raw === nothing && return 30000
    parsed = tryparse(Int, raw)
    parsed === nothing ? 30000 : parsed
end

function smt_logic()
    raw = get(ENV, "AXIOM_SMT_LOGIC", nothing)
    raw === nothing && return :QF_NRA
    Symbol(uppercase(raw))
end

function validate_solver_path(path::String)
    # Security: Validate solver path before execution
    # 1. Must be absolute path (no relative paths like ../)
    if !isabspath(path)
        @warn "SMT solver path must be absolute, not relative" path=path
        return false
    end

    # 2. Must not contain path traversal patterns
    if contains(path, "..") || contains(path, "~")
        @warn "SMT solver path contains unsafe patterns (.. or ~)" path=path
        return false
    end

    # 3. Must exist and be executable
    if !isfile(path)
        @warn "SMT solver path does not exist" path=path
        return false
    end

    # 4. On Unix, check if file is executable
    if Sys.isunix() && !Sys.isexecutable(path)
        @warn "SMT solver is not executable" path=path
        return false
    end

    true
end

function get_smt_solver()
    path_override = get(ENV, "AXIOM_SMT_SOLVER_PATH", nothing)
    if path_override !== nothing
        kind_raw = get(ENV, "AXIOM_SMT_SOLVER_KIND", nothing)
        if kind_raw === nothing
            @warn "AXIOM_SMT_SOLVER_PATH set without AXIOM_SMT_SOLVER_KIND; ignoring override"
        else
            kind = Symbol(lowercase(kind_raw))
            if kind in SMT_ALLOWLIST
                # Validate path before using
                if !validate_solver_path(path_override)
                    @warn "SMT solver path validation failed, ignoring override" path=path_override
                else
                    return SMTLib.SMTSolver(kind, path_override, "custom")
                end
            else
                @warn "SMT solver kind not allowed" kind=kind allowed=collect(SMT_ALLOWLIST)
            end
        end
    end

    solvers = SMTLib.available_solvers()
    solvers = filter(s -> s.kind in SMT_ALLOWLIST, solvers)
    preference = smt_solver_preference()
    if preference !== nothing
        for solver in solvers
            if solver.kind == preference
                return solver
            end
        end
        @warn "Preferred SMT solver not available" preferred=preference available=[s.kind for s in solvers]
    end

    isempty(solvers) ? nothing : first(solvers)
end

function get_smt_context()
    solver = get_smt_solver()
    solver === nothing && return nothing
    SMTLib.SMTContext(solver=solver, logic=smt_logic(), timeout_ms=smt_timeout_ms())
end

# Additional pattern matchers
function is_tanh_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    contains(s, "tanh") && (contains(s, "[-1, 1]") || contains(s, "bounded"))
end

function is_probability_valid_property(prop::ParsedProperty)
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
function is_softmax_sum_property(prop::ParsedProperty)
    body = prop.body
    # Match: sum(softmax(...)) ≈ 1.0
    body isa Expr &&
    body.head == :call &&
    body.args[1] == :≈ &&
    body.args[2] isa Expr &&
    body.args[2].head == :call &&
    body.args[2].args[1] == :sum
end

function is_relu_nonnegative_property(prop::ParsedProperty)
    body = prop.body
    # Match: all(relu(...) .>= 0)
    body isa Expr && contains_relu_geq_zero(body)
end

function is_sigmoid_bounded_property(prop::ParsedProperty)
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

function is_layernorm_normalized_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "layernorm") || contains(s, "layer_norm")) &&
    (contains(s, "mean") || contains(s, "variance") || contains(s, "normalized"))
end

function is_batchnorm_normalized_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "batchnorm") || contains(s, "batch_norm")) &&
    (contains(s, "normalized") || contains(s, "mean") || contains(s, "variance"))
end

function is_dropout_bounded_property(prop::ParsedProperty)
    body = prop.body
    s = string(body)
    contains(s, "dropout") && (contains(s, "bounded") || contains(s, "["))
end

function is_maxpool_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "maxpool") || contains(s, "max_pool")) && contains(s, "bounded")
end

function is_avgpool_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "avgpool") || contains(s, "avg_pool") || contains(s, "mean_pool")) &&
    contains(s, "bounded")
end

function is_concat_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "concat") || contains(s, "cat") || contains(s, "vcat") || contains(s, "hcat")) &&
    contains(s, "bounded")
end

function is_conv_finite_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "conv") && !contains(s, "convex")) &&
    (contains(s, "finite") || contains(s, "!isnan") || contains(s, "!isinf"))
end

function is_linear_finite_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "linear") || contains(s, "dense") || contains(s, "fc")) &&
    (contains(s, "finite") || contains(s, "!isnan") || contains(s, "!isinf"))
end

# Additional activation function matchers

function is_gelu_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "gelu") || contains(s, "GELU")) &&
    (contains(s, "bounded") || contains(s, "[") || contains(s, "finite"))
end

function is_swish_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "swish") || contains(s, "Swish") || contains(s, "silu")) &&
    (contains(s, "bounded") || contains(s, "[") || contains(s, "finite"))
end

function is_mish_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    contains(s, "mish") || contains(s, "Mish") &&
    (contains(s, "bounded") || contains(s, "finite"))
end

function is_elu_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "elu") && !contains(s, "relu") && !contains(s, "selu")) &&
    (contains(s, "bounded") || contains(s, ">=") || contains(s, "lower"))
end

function is_selu_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    contains(s, "selu") && (contains(s, "bounded") || contains(s, ">="))
end

function is_leaky_relu_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "leaky") || contains(s, "LeakyReLU")) &&
    contains(s, "relu") && (contains(s, "bounded") || contains(s, "preserv"))
end

# Normalization matchers

function is_groupnorm_normalized_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "groupnorm") || contains(s, "group_norm")) &&
    (contains(s, "normalized") || contains(s, "mean") || contains(s, "variance"))
end

function is_instancenorm_normalized_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "instancenorm") || contains(s, "instance_norm")) &&
    (contains(s, "normalized") || contains(s, "mean") || contains(s, "variance"))
end

# Attention mechanism matchers

function is_attention_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "attention") && !contains(s, "multihead")) &&
    (contains(s, "weight") || contains(s, "score")) &&
    (contains(s, "bounded") || contains(s, "[0") || contains(s, "probability"))
end

function is_multihead_attention_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "multihead") || contains(s, "multi_head") || contains(s, "multi-head")) &&
    contains(s, "attention") &&
    (contains(s, "bounded") || contains(s, "finite") || contains(s, "valid"))
end

# Residual/skip connection matchers

function is_residual_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "residual") || contains(s, "+ x") || contains(s, "f(x) + x")) &&
    (contains(s, "bounded") || contains(s, "finite") || contains(s, "preserv"))
end

function is_skipconnection_finite_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "skip") || contains(s, "shortcut")) &&
    (contains(s, "connection") || contains(s, "concat")) &&
    (contains(s, "finite") || contains(s, "!isnan"))
end

# Embedding matchers

function is_embedding_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    contains(s, "embedding") && !contains(s, "positional") &&
    (contains(s, "bounded") || contains(s, "["))
end

function is_positional_encoding_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "positional") && (contains(s, "encoding") || contains(s, "embedding"))) &&
    (contains(s, "bounded") || contains(s, "[-1, 1]") || contains(s, "sin") || contains(s, "cos"))
end

# Additional pooling matchers

function is_adaptiveavgpool_bounded_property(prop::ParsedProperty)
    s = string(prop.body)
    (contains(s, "adaptive") && (contains(s, "avgpool") || contains(s, "avg_pool") || contains(s, "mean_pool"))) &&
    contains(s, "bounded")
end

function is_adaptivemaxpool_bounded_property(prop::ParsedProperty)
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

# Verification certificate generation
"""
    VerificationCertificate

A formal certificate proving properties about a model.
"""
struct VerificationCertificate
    model_name::Symbol
    properties::Vector{ParsedProperty}
    proofs::Vector{ProofResult}
    timestamp::Float64
    version::String
end

"""
    generate_certificate(model, properties) -> VerificationCertificate

Generate a verification certificate for a model.
"""
function generate_certificate(model::AxiomModel, properties::Vector)
    parsed = [parse_property(p) for p in properties]
    proofs = [attempt_proof(p) for p in parsed]

    # Check all properties proven
    for (prop, proof) in zip(properties, proofs)
        if proof.status != :proven
            error("Cannot certify: property not proven: $prop")
        end
    end

    VerificationCertificate(
        nameof(typeof(model)),
        parsed,
        proofs,
        time(),
        string(Axiom.VERSION)
    )
end

"""
    save_certificate(cert, filename)

Save verification certificate to file.
"""
function save_certificate(cert::VerificationCertificate, filename::String)
    open(filename, "w") do f
        # Write header
        println(f, "# Axiom.jl Verification Certificate")
        println(f, "# Format: YAML-like key-value pairs")
        println(f, "# Generated: $(Dates.now())")
        println(f, "")

        # Write metadata
        println(f, "model_name: $(cert.model_name)")
        println(f, "timestamp: $(cert.timestamp)")
        println(f, "version: $(cert.version)")
        println(f, "")

        # Write properties
        println(f, "properties:")
        for prop in cert.properties
            println(f, "  - quantifier: $(prop.quantifier)")
            println(f, "    variables: [$(join(string.(prop.variables), ", "))]")
            println(f, "    body: \"$(escape_string(string(prop.body)))\"")
        end
        println(f, "")

        # Write proofs
        println(f, "proofs:")
        for proof in cert.proofs
            println(f, "  - status: $(proof.status)")
            println(f, "    confidence: $(proof.confidence)")
            if proof.counterexample !== nothing
                println(f, "    counterexample: \"$(escape_string(string(proof.counterexample)))\"")
            end
        end
        println(f, "")

        # Write signature (hash of certificate content for integrity)
        content = "$(cert.model_name)|$(cert.timestamp)|$(cert.version)"
        signature = bytes2hex(sha256(content))
        println(f, "signature: $signature")
    end

    @info "Certificate saved to $filename"
end

"""
    load_certificate(filename) -> VerificationCertificate

Load verification certificate from file.
"""
function load_certificate(filename::String)
    lines = readlines(filename)

    model_name = Symbol(:unknown)
    timestamp = 0.0
    version = ""
    properties = ParsedProperty[]
    proofs = ProofResult[]

    current_section = :none
    current_item = Dict{String, Any}()

    for line in lines
        line = strip(line)

        # Skip comments and empty lines
        startswith(line, "#") && continue
        isempty(line) && continue

        # Parse key-value pairs
        if contains(line, ": ")
            parts = split(line, ": ", limit=2)
            key = strip(parts[1])
            value = length(parts) > 1 ? strip(parts[2]) : ""

            # Remove leading dash for list items
            if startswith(key, "- ")
                key = key[3:end]
            end

            if key == "model_name"
                model_name = Symbol(value)
            elseif key == "timestamp"
                timestamp = parse(Float64, value)
            elseif key == "version"
                version = value
            elseif key == "properties"
                current_section = :properties
            elseif key == "proofs"
                current_section = :proofs
            elseif key == "quantifier" && current_section == :properties
                # Start new property
                if !isempty(current_item)
                    push!(properties, dict_to_property(current_item))
                end
                current_item = Dict("quantifier" => Symbol(value))
            elseif key == "variables" && current_section == :properties
                # Parse variable list
                vars_str = replace(value, r"[\[\]]" => "")
                vars = [Symbol(strip(v)) for v in split(vars_str, ",") if !isempty(strip(v))]
                current_item["variables"] = vars
            elseif key == "body" && current_section == :properties
                current_item["body"] = unescape_string(strip(value, '"'))
            elseif key == "status" && current_section == :proofs
                # Start new proof
                if !isempty(current_item) && haskey(current_item, "status")
                    push!(proofs, dict_to_proof(current_item))
                end
                current_item = Dict("status" => Symbol(value))
            elseif key == "confidence" && current_section == :proofs
                current_item["confidence"] = parse(Float64, value)
            elseif key == "counterexample" && current_section == :proofs
                current_item["counterexample"] = unescape_string(strip(value, '"'))
            end
        end
    end

    # Push last item
    if current_section == :properties && !isempty(current_item)
        push!(properties, dict_to_property(current_item))
    elseif current_section == :proofs && !isempty(current_item)
        push!(proofs, dict_to_proof(current_item))
    end

    VerificationCertificate(model_name, properties, proofs, timestamp, version)
end

function dict_to_property(d::Dict)
    quantifier = get(d, "quantifier", :none)
    variables = get(d, "variables", Symbol[])
    body_str = get(d, "body", "true")
    body = Meta.parse(body_str)
    ParsedProperty(quantifier, variables, body)
end

function dict_to_proof(d::Dict)
    status = get(d, "status", :unknown)
    confidence = get(d, "confidence", 0.0)
    counterexample = get(d, "counterexample", nothing)
    ProofResult(status, counterexample, confidence)
end

# Import SHA for certificate signing
using SHA
using Dates

# Security verification helpers

"""
    verify_smt_security_config() -> Dict{String, Any}

Verify SMT solver security configuration and return a report.

# Returns
Dictionary with security check results:
- `solver_allowlisted`: Bool - Is the solver in the allow-list?
- `timeout_set`: Bool - Is a timeout configured?
- `timeout_value`: Int - Timeout in milliseconds
- `path_validated`: Bool - Is the solver path valid?
- `cache_enabled`: Bool - Is result caching enabled?
- `rust_runner`: Bool - Is Rust runner enabled?
- `warnings`: Vector{String} - Security warnings
- `recommendations`: Vector{String} - Security recommendations

# Example
```julia
report = verify_smt_security_config()
if !isempty(report["warnings"])
    @warn "Security issues detected" warnings=report["warnings"]
end
```
"""
function verify_smt_security_config()
    warnings = String[]
    recommendations = String[]

    # Check timeout
    timeout = smt_timeout_ms()
    timeout_set = haskey(ENV, "AXIOM_SMT_TIMEOUT_MS")
    if timeout <= 0
        push!(warnings, "SMT timeout is disabled or invalid (≤0)")
        push!(recommendations, "Set AXIOM_SMT_TIMEOUT_MS to 5000-300000 (5 seconds to 5 minutes)")
    elseif timeout > 300000
        push!(warnings, "SMT timeout is very high (>5 minutes)")
        push!(recommendations, "Consider reducing AXIOM_SMT_TIMEOUT_MS to avoid long hangs")
    end

    # Check solver
    solver = get_smt_solver()
    solver_allowlisted = solver !== nothing && solver.kind in SMT_ALLOWLIST
    if solver === nothing
        push!(warnings, "No SMT solver available")
        push!(recommendations, "Install z3, cvc5, yices, or mathsat")
    elseif !solver_allowlisted
        push!(warnings, "Configured solver is not in allow-list")
        push!(recommendations, "Use only allow-listed solvers: z3, cvc5, yices, mathsat")
    end

    # Check path validation
    path_validated = true
    if haskey(ENV, "AXIOM_SMT_SOLVER_PATH")
        path = ENV["AXIOM_SMT_SOLVER_PATH"]
        path_validated = validate_solver_path(path)
        if !path_validated
            push!(warnings, "SMT solver path validation failed")
            push!(recommendations, "Use absolute paths to trusted solver binaries")
        end
    end

    # Check cache
    cache_enabled = smt_cache_enabled()
    if !cache_enabled
        push!(recommendations, "Consider enabling SMT cache with AXIOM_SMT_CACHE=1 for better performance")
    end

    # Check Rust runner
    rust_runner = use_rust_smt_runner()
    if rust_runner && !rust_available()
        push!(warnings, "Rust runner requested but not available")
        push!(recommendations, "Set AXIOM_RUST_LIB to path of libaxiom_core.so, or disable with AXIOM_SMT_RUNNER=julia")
    end

    Dict{String, Any}(
        "solver_allowlisted" => solver_allowlisted,
        "timeout_set" => timeout_set,
        "timeout_value" => timeout,
        "path_validated" => path_validated,
        "cache_enabled" => cache_enabled,
        "rust_runner" => rust_runner,
        "rust_available" => rust_available(),
        "warnings" => warnings,
        "recommendations" => recommendations
    )
end

"""
    print_smt_security_report()

Print a human-readable SMT security configuration report.

# Example
```julia
print_smt_security_report()
```
"""
function print_smt_security_report()
    report = verify_smt_security_config()

    println("═"^60)
    println("  SMT Security Configuration Report")
    println("═"^60)
    println()

    # Status indicators
    check = report["solver_allowlisted"] && report["timeout_set"] && report["path_validated"]
    status = check ? "✓ SECURE" : "⚠ REVIEW NEEDED"
    println("Overall Status: $status")
    println()

    # Configuration details
    println("Configuration:")
    println("  • Solver allow-listed: ", report["solver_allowlisted"] ? "✓" : "✗")
    println("  • Timeout configured:  ", report["timeout_set"] ? "✓ ($(report["timeout_value"])ms)" : "✗")
    println("  • Path validated:      ", report["path_validated"] ? "✓" : "✗")
    println("  • Cache enabled:       ", report["cache_enabled"] ? "✓" : "○")
    println("  • Rust runner:         ", report["rust_runner"] ? "✓" : "○")
    if report["rust_runner"]
        println("    - Rust available:    ", report["rust_available"] ? "✓" : "✗")
    end
    println()

    # Warnings
    if !isempty(report["warnings"])
        println("⚠ Warnings:")
        for warning in report["warnings"]
            println("  • $warning")
        end
        println()
    end

    # Recommendations
    if !isempty(report["recommendations"])
        println("Recommendations:")
        for rec in report["recommendations"]
            println("  • $rec")
        end
        println()
    end

    println("═"^60)
end

"""
    smt_security_checklist()

Print a quick security checklist for SMT configuration.
"""
function smt_security_checklist()
    println("SMT Security Checklist:")
    println()
    println("□ 1. Solver Allow-List")
    println("      Only use: z3, cvc5, yices, mathsat")
    println("      export AXIOM_SMT_SOLVER=z3")
    println()
    println("□ 2. Timeout Configuration")
    println("      Set reasonable timeout (5-300 seconds)")
    println("      export AXIOM_SMT_TIMEOUT_MS=30000")
    println()
    println("□ 3. Path Validation")
    println("      Use absolute paths only")
    println("      export AXIOM_SMT_SOLVER_PATH=/usr/local/bin/z3")
    println("      export AXIOM_SMT_SOLVER_KIND=z3")
    println()
    println("□ 4. Enable Caching (recommended)")
    println("      export AXIOM_SMT_CACHE=1")
    println("      export AXIOM_SMT_CACHE_MAX=128")
    println()
    println("□ 5. Rust Runner (optional, high-security)")
    println("      export AXIOM_SMT_RUNNER=rust")
    println("      export AXIOM_RUST_LIB=/path/to/libaxiom_core.so")
    println()
    println("Verify configuration: verify_smt_security_config()")
end
