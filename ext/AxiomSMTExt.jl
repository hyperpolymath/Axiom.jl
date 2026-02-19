module AxiomSMTExt

using Axiom
using SMTLib
using SHA
using Dates

function __init__()
    # This ensures @prove is exported from the main Axiom module when SMTLib is loaded
    Base.eval(Axiom, :(export @prove))
end

"""
SMT solver integration for formal verification.

This integrates with external SMT solvers (Z3, CVC5, Yices, MathSAT) via SMTLib.
Falls back to heuristic methods otherwise.
"""
function Axiom.smt_proof(property::Axiom.ParsedProperty)
    ctx = get_smt_context()

    if ctx === nothing
        return Axiom.ProofResult(:unknown, nothing, 0.0)
    end

    vars = property.variables
    expr = normalize_smt_expr(property.body)

    for v in vars
        # Strip Julia type annotations if present
        name = v isa Expr && v.head == :(::) ? v.args[1] : v
        if name isa Symbol
            SMTLib.declare(ctx, name, Float64)
        end
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

    result = if use_rust_smt_runner() && Axiom.rust_available()
        output = Axiom.rust_smt_run(string(ctx.solver.kind), ctx.solver.path, script, ctx.timeout_ms)
        SMTLib.parse_result(output)
    else
        SMTLib.check_sat(ctx; get_model=true)
    end

    smt_cache_put(cache_key, result)
    return finalize_smt_result(property, result)
end

"""
Normalize expressions for SMT-LIB conversion.
"""
function normalize_smt_expr(expr)
    if expr isa Expr
        if expr.head == :block
            # Handle blocks by taking the last non-LineNumberNode
            args = filter(x -> !(x isa LineNumberNode), expr.args)
            if isempty(args)
                return :true
            elseif length(args) == 1
                return normalize_smt_expr(args[1])
            else
                # If multiple expressions remain, AND them together
                return Expr(:call, :&&, map(normalize_smt_expr, args)...)
            end
        end
        if expr.head == :(::) && length(expr.args) == 2
            # Strip type annotation: x::Real -> x
            return normalize_smt_expr(expr.args[1])
        end
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

function finalize_smt_result(property::Axiom.ParsedProperty, result::SMTLib.SMTResult)
    if result.status == :sat
        if property.quantifier == :exists
            return Axiom.ProofResult(:proven, result.model, 1.0,
                "SMT solver found satisfying assignment: ∃x. property(x) is SAT",
                String[])
        end
        return Axiom.ProofResult(:disproven, result.model, 1.0,
            "SMT solver found counterexample: ¬(∀x. property(x)) is SAT",
            ["Check if the property holds for the provided counterexample",
             "Consider adding preconditions to restrict the domain",
             "Verify that the model behaves correctly on the counterexample input"])
    elseif result.status == :unsat
        if property.quantifier == :exists
            return Axiom.ProofResult(:disproven, nothing, 1.0,
                "SMT solver proved no satisfying assignment exists: ∃x. property(x) is UNSAT",
                ["Property is impossible to satisfy",
                 "Review the property definition - it may be too restrictive"])
        end
        return Axiom.ProofResult(:proven, nothing, 1.0,
            "SMT solver proved property holds for all inputs: ∀x. property(x) is valid (¬property is UNSAT)",
            String[])
    end

    Axiom.ProofResult(:unknown, nothing, 0.0,
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
                @warn "SMTLib solver kind not allowed" kind=kind allowed=collect(SMT_ALLOWLIST)
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

end # module AxiomSMTExt