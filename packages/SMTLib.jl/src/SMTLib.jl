# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    SMTLib.jl

A lightweight Julia interface to SMT solvers (Z3, CVC5, etc.) via SMT-LIB2 format.

# Features
- Auto-detection of installed SMT solvers
- Julia expression to SMT-LIB2 conversion
- Support for multiple logics (QF_LIA, QF_LRA, QF_NRA, QF_BV, etc.)
- Model parsing and counterexample extraction
- Timeout support

# Example
```julia
using SMTLib

# Check satisfiability
result = @smt begin
    x::Int
    y::Int
    x + y == 10
    x > 0
    y > 0
end

if result.status == :sat
    println("x = ", result.model[:x])
    println("y = ", result.model[:y])
end
```
"""
module SMTLib

export SMTSolver, SMTResult, SMTContext
export @smt, check_sat, get_model, push!, pop!
export declare, assert!, reset!
export find_solver, available_solvers
export to_smtlib, from_smtlib

# Supported SMT-LIB logics
const LOGICS = [
    :QF_LIA,    # Quantifier-free linear integer arithmetic
    :QF_LRA,    # Quantifier-free linear real arithmetic
    :QF_NIA,    # Quantifier-free nonlinear integer arithmetic
    :QF_NRA,    # Quantifier-free nonlinear real arithmetic
    :QF_BV,     # Quantifier-free bitvectors
    :QF_AUFLIA, # Quantifier-free arrays, uninterpreted functions, linear integer arithmetic
    :LIA,       # Linear integer arithmetic
    :LRA,       # Linear real arithmetic
    :AUFLIRA,   # Arrays, uninterpreted functions, linear arithmetic
    :ALL,       # All supported theories
]

"""
    SMTSolver

Wrapper for an external SMT solver.
"""
struct SMTSolver
    kind::Symbol      # :z3, :cvc5, :yices, :mathsat
    path::String      # Path to solver executable
    version::String   # Solver version string
end

function Base.show(io::IO, s::SMTSolver)
    print(io, "SMTSolver(:$(s.kind), \"$(s.path)\")")
end

"""
    SMTResult

Result from an SMT query.
"""
struct SMTResult
    status::Symbol                      # :sat, :unsat, :unknown, :timeout
    model::Dict{Symbol, Any}            # Variable assignments (if sat)
    unsat_core::Vector{Symbol}          # Unsat core (if unsat and requested)
    statistics::Dict{String, Any}       # Solver statistics
    raw_output::String                  # Raw solver output
end

function Base.show(io::IO, r::SMTResult)
    print(io, "SMTResult(:$(r.status)")
    if r.status == :sat && !isempty(r.model)
        print(io, ", model=$(r.model)")
    end
    print(io, ")")
end

"""
    SMTContext

Stateful context for incremental SMT solving.
"""
mutable struct SMTContext
    solver::SMTSolver
    logic::Symbol
    declarations::Vector{String}
    assertions::Vector{String}
    timeout_ms::Int
end

function SMTContext(; solver::Union{SMTSolver, Nothing}=nothing,
                     logic::Symbol=:QF_LRA,
                     timeout_ms::Int=30000)
    s = solver === nothing ? find_solver() : solver
    if s === nothing
        error("No SMT solver found. Install z3 or cvc5.")
    end
    SMTContext(s, logic, String[], String[], timeout_ms)
end

# ============================================================================
# Solver Discovery
# ============================================================================

"""
    find_solver(preference=nothing) -> Union{SMTSolver, Nothing}

Find an available SMT solver on the system.
Optionally specify a preference (:z3, :cvc5, :yices, :mathsat).
"""
function find_solver(preference::Union{Symbol, Nothing}=nothing)
    solvers = available_solvers()

    if isempty(solvers)
        return nothing
    end

    if preference !== nothing
        for s in solvers
            if s.kind == preference
                return s
            end
        end
    end

    # Return first available
    first(solvers)
end

"""
    available_solvers() -> Vector{SMTSolver}

List all available SMT solvers on the system.
"""
function available_solvers()
    solvers = SMTSolver[]

    # Check for Z3
    z3_path = Sys.which("z3")
    if z3_path !== nothing
        version = try
            strip(read(`$z3_path --version`, String))
        catch
            "unknown"
        end
        push!(solvers, SMTSolver(:z3, z3_path, version))
    end

    # Check for CVC5
    cvc5_path = Sys.which("cvc5")
    if cvc5_path !== nothing
        version = try
            strip(read(`$cvc5_path --version`, String))
        catch
            "unknown"
        end
        push!(solvers, SMTSolver(:cvc5, cvc5_path, version))
    end

    # Check for Yices
    yices_path = Sys.which("yices-smt2")
    if yices_path !== nothing
        version = try
            strip(read(`$yices_path --version`, String))
        catch
            "unknown"
        end
        push!(solvers, SMTSolver(:yices, yices_path, version))
    end

    # Check for MathSAT
    mathsat_path = Sys.which("mathsat")
    if mathsat_path !== nothing
        push!(solvers, SMTSolver(:mathsat, mathsat_path, "unknown"))
    end

    solvers
end

# ============================================================================
# SMT-LIB Generation
# ============================================================================

"""
    to_smtlib(expr) -> String

Convert a Julia expression to SMT-LIB2 format.
"""
function to_smtlib(expr)
    if expr isa Symbol
        return string(expr)
    elseif expr isa Bool
        return expr ? "true" : "false"
    elseif expr isa Integer
        return expr < 0 ? "(- $(abs(expr)))" : string(expr)
    elseif expr isa AbstractFloat
        return expr < 0 ? "(- $(abs(expr)))" : string(Float64(expr))
    elseif expr isa Expr
        return expr_to_smtlib(expr)
    else
        return string(expr)
    end
end

function expr_to_smtlib(expr::Expr)
    if expr.head == :call
        op = expr.args[1]
        args = expr.args[2:end]

        smt_op = julia_op_to_smt(op)
        smt_args = join([to_smtlib(a) for a in args], " ")

        return "($smt_op $smt_args)"
    elseif expr.head == :&&
        args = [to_smtlib(a) for a in expr.args]
        return "(and $(join(args, " ")))"
    elseif expr.head == :||
        args = [to_smtlib(a) for a in expr.args]
        return "(or $(join(args, " ")))"
    elseif expr.head == :comparison
        # Handle chained comparisons: a < b < c
        return handle_chained_comparison(expr)
    elseif expr.head == :if || expr.head == :elseif
        cond = to_smtlib(expr.args[1])
        then_branch = to_smtlib(expr.args[2])
        else_branch = length(expr.args) > 2 ? to_smtlib(expr.args[3]) : "false"
        return "(ite $cond $then_branch $else_branch)"
    elseif expr.head == :let
        # Handle let bindings
        return handle_let(expr)
    end

    # Fallback
    string(expr)
end

function handle_chained_comparison(expr::Expr)
    # a < b < c becomes (and (< a b) (< b c))
    parts = String[]
    for i in 1:2:length(expr.args)-2
        left = to_smtlib(expr.args[i])
        op = julia_op_to_smt(expr.args[i+1])
        right = to_smtlib(expr.args[i+2])
        push!(parts, "($op $left $right)")
    end

    if length(parts) == 1
        return parts[1]
    else
        return "(and $(join(parts, " ")))"
    end
end

function handle_let(expr::Expr)
    # Simple let handling
    bindings = expr.args[1]
    body = expr.args[2]

    smt_bindings = String[]
    for binding in bindings.args
        var = binding.args[1]
        val = to_smtlib(binding.args[2])
        push!(smt_bindings, "($(var) $val)")
    end

    "(let ($(join(smt_bindings, " "))) $(to_smtlib(body)))"
end

"""
Map Julia operators to SMT-LIB operators.
"""
function julia_op_to_smt(op)
    op_map = Dict(
        # Arithmetic
        :+ => "+",
        :- => "-",
        :* => "*",
        :/ => "/",
        :div => "div",
        :mod => "mod",
        :rem => "rem",
        :abs => "abs",

        # Comparison
        :(==) => "=",
        :!= => "distinct",
        :≠ => "distinct",
        :< => "<",
        :> => ">",
        :<= => "<=",
        :>= => ">=",
        :≤ => "<=",
        :≥ => ">=",

        # Logical
        :! => "not",
        :¬ => "not",
        :&& => "and",
        :|| => "or",
        :∧ => "and",
        :∨ => "or",
        :⟹ => "=>",
        :implies => "=>",
        :iff => "=",
        :⟺ => "=",
        :xor => "xor",

        # Quantifiers (for quantified logics)
        :∀ => "forall",
        :∃ => "exists",
        :forall => "forall",
        :exists => "exists",

        # Array operations
        :select => "select",
        :store => "store",

        # Bitvector operations
        :bvadd => "bvadd",
        :bvsub => "bvsub",
        :bvmul => "bvmul",
        :bvand => "bvand",
        :bvor => "bvor",
        :bvxor => "bvxor",
        :bvnot => "bvnot",
        :bvshl => "bvshl",
        :bvlshr => "bvlshr",
        :bvashr => "bvashr",

        # Math functions (for nonlinear arithmetic)
        :^ => "^",
        :sqrt => "sqrt",
        :exp => "exp",
        :log => "log",
        :sin => "sin",
        :cos => "cos",
        :tan => "tan",
    )

    get(op_map, op, string(op))
end

# ============================================================================
# Type Declarations
# ============================================================================

"""
    smt_type(::Type) -> String

Get SMT-LIB type string for Julia type.
"""
smt_type(::Type{Int}) = "Int"
smt_type(::Type{Int64}) = "Int"
smt_type(::Type{Int32}) = "Int"
smt_type(::Type{Bool}) = "Bool"
smt_type(::Type{Float64}) = "Real"
smt_type(::Type{Float32}) = "Real"
smt_type(::Type{<:AbstractFloat}) = "Real"
smt_type(::Type{<:Integer}) = "Int"

# Bitvector types
struct BitVec{N} end
smt_type(::Type{BitVec{N}}) where N = "(_ BitVec $N)"

# Array types
struct SMTArray{K, V} end
smt_type(::Type{SMTArray{K, V}}) where {K, V} = "(Array $(smt_type(K)) $(smt_type(V)))"

# ============================================================================
# Context Operations
# ============================================================================

"""
    declare(ctx::SMTContext, name::Symbol, type)

Declare a variable in the context.
"""
function declare(ctx::SMTContext, name::Symbol, type)
    decl = "(declare-const $name $(smt_type(type)))"
    push!(ctx.declarations, decl)
    nothing
end

"""
    assert!(ctx::SMTContext, expr)

Add an assertion to the context.
"""
function assert!(ctx::SMTContext, expr)
    smt_expr = to_smtlib(expr)
    push!(ctx.assertions, "(assert $smt_expr)")
    nothing
end

"""
    reset!(ctx::SMTContext)

Reset the context, clearing all declarations and assertions.
"""
function reset!(ctx::SMTContext)
    empty!(ctx.declarations)
    empty!(ctx.assertions)
    nothing
end

"""
    check_sat(ctx::SMTContext) -> SMTResult

Check satisfiability of current context.
"""
function check_sat(ctx::SMTContext; get_model::Bool=true)
    # Build SMT-LIB script
    script = build_script(ctx, get_model)

    # Run solver
    run_solver(ctx.solver, script, ctx.timeout_ms)
end

function build_script(ctx::SMTContext, get_model::Bool)
    lines = String[]

    # Set logic
    push!(lines, "(set-logic $(ctx.logic))")

    # Set options
    push!(lines, "(set-option :produce-models true)")

    # Declarations
    append!(lines, ctx.declarations)

    # Assertions
    append!(lines, ctx.assertions)

    # Check sat
    push!(lines, "(check-sat)")

    # Get model if requested
    if get_model
        push!(lines, "(get-model)")
    end

    join(lines, "\n")
end

# ============================================================================
# Solver Execution
# ============================================================================

"""
    run_solver(solver::SMTSolver, script::String, timeout_ms::Int) -> SMTResult

Execute SMT solver with given script.
"""
function run_solver(solver::SMTSolver, script::String, timeout_ms::Int)
    # Write script to temp file
    temp_file = tempname() * ".smt2"
    write(temp_file, script)

    try
        # Build command with timeout
        cmd = build_solver_command(solver, temp_file, timeout_ms)

        # Run solver
        output = try
            read(cmd, String)
        catch e
            if e isa ProcessFailedException
                # Some solvers return non-zero on unsat
                String(e.procs[1].cmd.exec[end])
            else
                return SMTResult(:error, Dict{Symbol, Any}(), Symbol[],
                               Dict{String, Any}(), "Error: $e")
            end
        end

        # Parse result
        parse_result(output)
    finally
        rm(temp_file, force=true)
    end
end

function build_solver_command(solver::SMTSolver, input_file::String, timeout_ms::Int)
    timeout_sec = timeout_ms ÷ 1000

    if solver.kind == :z3
        `$(solver.path) -T:$timeout_sec $input_file`
    elseif solver.kind == :cvc5
        `$(solver.path) --tlimit=$timeout_ms $input_file`
    elseif solver.kind == :yices
        `$(solver.path) --timeout=$timeout_sec $input_file`
    else
        `$(solver.path) $input_file`
    end
end

"""
    parse_result(output::String) -> SMTResult

Parse SMT solver output.
"""
function parse_result(output::String)
    lines = split(strip(output), '\n')

    # Determine status
    status = :unknown
    for line in lines
        line = strip(line)
        if line == "sat"
            status = :sat
            break
        elseif line == "unsat"
            status = :unsat
            break
        elseif line == "unknown"
            status = :unknown
            break
        elseif startswith(line, "timeout")
            status = :timeout
            break
        end
    end

    # Parse model if sat
    model = Dict{Symbol, Any}()
    if status == :sat
        model = parse_model(output)
    end

    SMTResult(status, model, Symbol[], Dict{String, Any}(), output)
end

"""
    parse_model(output::String) -> Dict{Symbol, Any}

Parse model from solver output.
"""
function parse_model(output::String)
    model = Dict{Symbol, Any}()

    # Simple regex-based parsing for common formats
    # Pattern: (define-fun name () Type value)
    pattern = r"\(define-fun\s+(\w+)\s+\(\)\s+\w+\s+(.+?)\)"

    for m in eachmatch(pattern, output)
        name = Symbol(m.captures[1])
        value_str = strip(m.captures[2])

        # Parse value
        value = parse_smt_value(value_str)
        model[name] = value
    end

    model
end

"""
    parse_smt_value(s::String) -> Any

Parse an SMT-LIB value.
"""
function parse_smt_value(s::String)
    s = strip(s)

    # Boolean
    if s == "true"
        return true
    elseif s == "false"
        return false
    end

    # Integer
    int_match = match(r"^(-?\d+)$", s)
    if int_match !== nothing
        return parse(Int, int_match.captures[1])
    end

    # Negative integer: (- N)
    neg_match = match(r"^\(-\s*(\d+)\)$", s)
    if neg_match !== nothing
        return -parse(Int, neg_match.captures[1])
    end

    # Real/Rational: (/ N D)
    rat_match = match(r"^\(/\s*(-?\d+)\s+(\d+)\)$", s)
    if rat_match !== nothing
        num = parse(Int, rat_match.captures[1])
        den = parse(Int, rat_match.captures[2])
        return num // den
    end

    # Decimal
    dec_match = match(r"^(-?\d+\.\d+)$", s)
    if dec_match !== nothing
        return parse(Float64, dec_match.captures[1])
    end

    # Bitvector: #bNNNN or #xHHHH
    if startswith(s, "#b")
        return parse(Int, s[3:end], base=2)
    elseif startswith(s, "#x")
        return parse(Int, s[3:end], base=16)
    end

    # Return as string if unparseable
    s
end

# ============================================================================
# Convenience Macro
# ============================================================================

"""
    @smt [solver=...] [logic=...] [timeout=...] begin ... end

Convenience macro for SMT queries.

# Example
```julia
result = @smt begin
    x::Int
    y::Int
    x + y == 10
    x > 0
    y > 0
end
```
"""
macro smt(args...)
    # Parse options and body
    opts = Dict{Symbol, Any}()
    body = nothing

    for arg in args
        if arg isa Expr && arg.head == :(=)
            opts[arg.args[1]] = arg.args[2]
        elseif arg isa Expr && arg.head == :block
            body = arg
        end
    end

    if body === nothing
        error("@smt requires a begin...end block")
    end

    # Generate code
    solver_expr = get(opts, :solver, nothing)
    logic_expr = get(opts, :logic, :(:QF_LIA))
    timeout_expr = get(opts, :timeout, 30000)

    quote
        ctx = SMTContext(logic=$(esc(logic_expr)), timeout_ms=$(esc(timeout_expr)))
        $(generate_smt_body(body, :ctx))
        check_sat(ctx)
    end
end

function generate_smt_body(body::Expr, ctx_sym::Symbol)
    stmts = Expr[]

    for stmt in body.args
        stmt isa LineNumberNode && continue

        if stmt isa Expr && stmt.head == :(::)
            # Variable declaration: x::Int
            var = stmt.args[1]
            typ = stmt.args[2]
            push!(stmts, :(declare($ctx_sym, $(QuoteNode(var)), $typ)))
        elseif stmt isa Expr
            # Assertion
            push!(stmts, :(assert!($ctx_sym, $(QuoteNode(stmt)))))
        end
    end

    Expr(:block, stmts...)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    prove(expr; kwargs...) -> Bool

Attempt to prove an expression is always true.
Returns true if proven, false if counterexample found or unknown.
"""
function prove(expr; solver=nothing, logic=:QF_LIA, timeout_ms=30000)
    # To prove P, show that ¬P is unsatisfiable
    ctx = SMTContext(solver=solver, logic=logic, timeout_ms=timeout_ms)

    # Extract free variables from expression
    vars = extract_variables(expr)
    for (name, typ) in vars
        declare(ctx, name, typ)
    end

    # Assert negation
    neg_expr = Expr(:call, :!, expr)
    assert!(ctx, neg_expr)

    result = check_sat(ctx)

    result.status == :unsat
end

"""
Extract free variables from expression (heuristic).
"""
function extract_variables(expr)
    # Simple heuristic - would need type inference in practice
    vars = Dict{Symbol, Type}()
    _extract_vars!(vars, expr)
    vars
end

function _extract_vars!(vars::Dict, expr)
    if expr isa Symbol && !haskey(vars, expr) && !is_operator(expr)
        vars[expr] = Int  # Default to Int
    elseif expr isa Expr
        for arg in expr.args
            _extract_vars!(vars, arg)
        end
    end
end

function is_operator(s::Symbol)
    s in [:+, :-, :*, :/, :div, :mod, :==, :!=, :<, :>, :<=, :>=,
          :!, :&&, :||, :true, :false, :∧, :∨, :¬, :⟹, :⟺]
end

end # module
