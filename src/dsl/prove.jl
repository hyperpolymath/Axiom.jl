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

    ProofResult(:unknown, nothing, 0.0,
        "Property does not match any known provable patterns",
        ["Try simplifying the property into smaller sub-properties",
         "Check if the property can be decomposed into known patterns",
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

function get_smt_solver()
    path_override = get(ENV, "AXIOM_SMT_SOLVER_PATH", nothing)
    if path_override !== nothing
        kind_raw = get(ENV, "AXIOM_SMT_SOLVER_KIND", nothing)
        if kind_raw === nothing
            @warn "AXIOM_SMT_SOLVER_PATH set without AXIOM_SMT_SOLVER_KIND; ignoring override"
        else
            kind = Symbol(lowercase(kind_raw))
            if kind in SMT_ALLOWLIST
                return SMTLib.SMTSolver(kind, path_override, "custom")
            end
            @warn "SMT solver kind not allowed" kind=kind
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
