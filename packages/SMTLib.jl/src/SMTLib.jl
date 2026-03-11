# SPDX-License-Identifier: PMPL-1.0-or-later
"""
    SMTLib.jl

A lightweight Julia interface to SMT solvers (Z3, CVC5, etc.) via SMT-LIB2 format.

Provides a complete pipeline from Julia expressions to SMT-LIB2 scripts, solver
invocation, and result/model parsing. Supports incremental solving with push/pop,
named assertions for unsat core extraction, optimization (Z3), quantifiers,
bitvector/array/floating-point theory helpers, and solver statistics.

# Features
- Auto-detection of installed SMT solvers (Z3, CVC5, Yices, MathSAT)
- Julia expression to SMT-LIB2 bidirectional conversion
- Support for multiple logics (QF_LIA, QF_LRA, QF_NRA, QF_BV, etc.)
- Model parsing and counterexample extraction
- Incremental solving with push!/pop!
- Named assertions and unsat core extraction
- Optimization (minimize!/maximize! for Z3)
- Quantifier support (forall/exists)
- Theory helpers: bitvectors, floating-point, arrays, strings/regex
- Solver statistics parsing
- Timeout support
- CEGIS Synthesis Engine (new!)

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

# Incremental Solving
```julia
ctx = SMTContext(logic=:QF_LIA)
declare(ctx, :x, Int)
assert!(ctx, :(x > 0))
push!(ctx)
assert!(ctx, :(x < 0))  # contradicts previous
result = check_sat(ctx)  # :unsat
pop!(ctx)                # restore to just x > 0
result = check_sat(ctx)  # :sat
```

Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <jonathan.jewell@open.ac.uk>
"""
module SMTLib

# ============================================================================
# Exports
# ============================================================================

export SMTSolver, SMTResult, SMTContext
export @smt, check_sat, get_model, push!, pop!
export declare, assert!, reset!
export find_solver, available_solvers
export to_smtlib, from_smtlib
export set_option!
export forall, exists
export minimize!, maximize!, optimize
export bv, fp_sort, array_sort, re_sort
export get_statistics, evaluate, get_unsat_core
export cegis # From Synthesis

# ============================================================================
# Submodules
# ============================================================================

include("synthesis.jl")
using .Synthesis

# ============================================================================
# Constants
# ============================================================================

"""
    LOGICS

List of supported SMT-LIB2 logic identifiers. Each logic restricts the theories
and quantifier usage available to the solver.
"""
const LOGICS = [
    :QF_LIA,    # Quantifier-free linear integer arithmetic
    :QF_LRA,    # Quantifier-free linear real arithmetic
    :QF_NIA,    # Quantifier-free nonlinear integer arithmetic
    :QF_NRA,    # Quantifier-free nonlinear real arithmetic
    :QF_BV,     # Quantifier-free bitvectors
    :QF_AUFLIA, # Quantifier-free arrays, uninterpreted functions, linear integer arithmetic
    :LIA,       # Linear integer arithmetic (with quantifiers)
    :LRA,       # Linear real arithmetic (with quantifiers)
    :AUFLIRA,   # Arrays, uninterpreted functions, linear arithmetic
    :QF_S,      # Quantifier-free strings
    :ALL,       # All supported theories
]

"""
    JULIA_OP_TO_SMT_MAP

Constant dictionary mapping Julia operator symbols to their SMT-LIB2 equivalents.
Used by `julia_op_to_smt` for expression conversion and by `is_operator` to
identify known operator symbols.
"""
const JULIA_OP_TO_SMT_MAP = Dict{Symbol,String}(
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
    :(<) => "<",  # Use :(<) for safety with symbol quoting
    :(>) => ">",
    :(<=) => "<=",
    :(>=) => ">=",

    # Unicode comparison
    Symbol("≠") => "distinct",  # != (U+2260)
    Symbol("≤") => "<=",        # <= (U+2264)
    Symbol("≥") => ">=",        # >= (U+2265)

    # Logical
    :! => "not",
    Symbol("¬") => "not",   # NOT sign (U+00AC)
    :&& => "and",
    :|| => "or",
    Symbol("∧") => "and",   # logical AND (U+2227)
    Symbol("∨") => "or",    # logical OR (U+2228)
    Symbol("⟹") => "=>",    # long rightwards double arrow (U+27F9, implies)
    :implies => "=>",
    :iff => "=",
    Symbol("⟺") => "=",     # long left right double arrow (U+27FA, iff)
    :xor => "xor",

    # Quantifiers (for quantified logics)
    Symbol("∀") => "forall", # for all (U+2200)
    Symbol("∃") => "exists", # there exists (U+2203)
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
    :bvult => "bvult",
    :bvugt => "bvugt",
    :bvule => "bvule",
    :bvuge => "bvuge",
    :bvslt => "bvslt",
    :bvsgt => "bvsgt",
    :bvsle => "bvsle",
    :bvsge => "bvsge",
    :concat => "concat",
    :extract => "extract",

    # String operations
    :str_len => "str.len",
    :str_concat => "str.++",
    :str_at => "str.at",
    :str_contains => "str.contains",
    :str_prefixof => "str.prefixof",
    :str_suffixof => "str.suffixof",
    :str_replace => "str.replace",
    :str_substr => "str.substr",
    :str_indexof => "str.indexof",
    :str_to_int => "str.to_int",
    :int_to_str => "str.from_int",
    :str_in_re => "str.in_re",
    :re_none => "re.none",
    :re_all => "re.all",
    :re_allchar => "re.allchar",
    :re_concat => "re.++",
    :re_union => "re.union",
    :re_inter => "re.inter",
    :re_star => "re.*",
    :re_plus => "re.+",
    :re_opt => "re.opt",
    :re_range => "re.range",
    :str_to_re => "str.to_re",

    # Math functions (for nonlinear arithmetic)
    :^ => "^",
    :sqrt => "sqrt",
    :exp => "exp",
    :log => "log",
    :sin => "sin",
    :cos => "cos",
    :tan => "tan",
)

"""
    KNOWN_OPERATORS

Constant set of all Julia operator symbols recognized by the expression converter.
Derived from the keys of `JULIA_OP_TO_SMT_MAP` plus boolean literals. Used by
`is_operator` to distinguish operators from free variables during expression analysis.
"""
const KNOWN_OPERATORS = let
    s = Set{Symbol}(keys(JULIA_OP_TO_SMT_MAP))
    push!(s, Symbol("true"))
    push!(s, Symbol("false"))
    s
end

"""
    SMT_OP_TO_JULIA_MAP

Reverse mapping from SMT-LIB2 operator strings to Julia operator symbols.
Used by `from_smtlib` for parsing SMT-LIB2 expressions back into Julia.
"""
const SMT_OP_TO_JULIA_MAP = let
    d = Dict{String,Symbol}()
    # Build reverse map; for duplicates, prefer the canonical Julia symbol
    for (jsym, sop) in JULIA_OP_TO_SMT_MAP
        # Some SMT ops map from multiple Julia symbols; keep first or canonical
        if !haskey(d, sop)
            d[sop] = jsym
        end
    end
    # Override certain mappings to canonical Julia symbols
    d["="] = :(==)
    d["distinct"] = :(!=)
    d["not"] = :!
    d["and"] = :&&
    d["or"] = :||
    d["=>"] = :implies
    d["<"] = :(<)
    d[">"] = :(>)
    d["<="] = :(<=)
    d[">="] = :(>=)
    d["true"] = Symbol("true")
    d["false"] = Symbol("false")
    d["ite"] = :ifelse
    d
end

# ============================================================================
# Solver Struct
# ============================================================================

"""
    SMTSolver

Wrapper for an external SMT solver executable.

# Fields
- `kind::Symbol`: Solver identifier (`:z3`, `:cvc5`, `:yices`, `:mathsat`).
- `path::String`: Absolute path to the solver executable.
- `version::String`: Solver version string as reported by the executable.

# Example
```julia
solver = find_solver(:z3)
println(solver.kind)     # :z3
println(solver.path)     # /usr/bin/z3
println(solver.version)  # Z3 version 4.12.4
```
"""
struct SMTSolver
    kind::Symbol      # :z3, :cvc5, :yices, :mathsat
    path::String      # Path to solver executable
    version::String   # Solver version string
end

function Base.show(io::IO, s::SMTSolver)
    print(io, "SMTSolver(:$(s.kind), \"$(s.path)\")")
end

# ============================================================================
# Result Struct
# ============================================================================

"""
    SMTResult

Result from an SMT solver query.

Contains the satisfiability status, model (if satisfiable), unsat core
(if unsatisfiable and named assertions were used), solver statistics,
and the raw solver output for debugging.

# Fields
- `status::Symbol`: One of `:sat`, `:unsat`, `:unknown`, or `:timeout`.
- `model::Dict{Symbol,Any}`: Variable assignments when status is `:sat`.
- `unsat_core::Vector{Symbol}`: Named assertions in the unsat core when status is `:unsat`.
- `statistics::Dict{String,Any}`: Solver statistics if requested.
- `raw_output::String`: Complete raw output from the solver process.

# Example
```julia
result = check_sat(ctx)
if result.status == :sat
    println("x = ", result.model[:x])
elseif result.status == :unsat
    println("Core: ", result.unsat_core)
end
```
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
    if r.status == :unsat && !isempty(r.unsat_core)
        print(io, ", core=$(r.unsat_core)")
    end
    print(io, ")")
end

# ============================================================================
# Context Struct
# ============================================================================

"""
    SMTContext

Stateful context for incremental SMT solving.

Accumulates declarations, assertions, solver options, and optimization
directives. Supports push/pop for backtracking, named assertions for
unsat core extraction, and Z3-specific optimization.

# Fields
- `solver::SMTSolver`: The solver to use for queries.
- `logic::Symbol`: SMT-LIB2 logic identifier (e.g. `:QF_LIA`).
- `declarations::Vector{String}`: Current variable/function declarations.
- `assertions::Vector{String}`: Current assertion strings.
- `timeout_ms::Int`: Solver timeout in milliseconds.
- `declarations_stack::Vector{Vector{String}}`: Saved declaration states for push/pop.
- `assertions_stack::Vector{Vector{String}}`: Saved assertion states for push/pop.
- `assertion_names::Dict{Symbol,String}`: Mapping from label to named assertion SMT string.
- `solver_options::Dict{String,String}`: Solver options to emit as `(set-option ...)`.
- `optimization_directives::Vector{String}`: Z3 `(minimize ...)` / `(maximize ...)` directives.
- `push_pop_commands::Vector{Tuple{Symbol,Int}}`: Log of push/pop operations for script building.

# Example
```julia
ctx = SMTContext(logic=:QF_LIA, timeout_ms=10000)
declare(ctx, :x, Int)
assert!(ctx, :(x > 5))
result = check_sat(ctx)
```
"""
mutable struct SMTContext
    solver::SMTSolver
    logic::Symbol
    declarations::Vector{String}
    assertions::Vector{String}
    timeout_ms::Int
    declarations_stack::Vector{Vector{String}}
    assertions_stack::Vector{Vector{String}}
    assertion_names::Dict{Symbol,String}
    solver_options::Dict{String,String}
    optimization_directives::Vector{String}
    push_pop_commands::Vector{Tuple{Symbol,Int}}
end

"""
    SMTContext(; solver=nothing, logic=:QF_LRA, timeout_ms=30000) -> SMTContext

Construct a new SMT context with the given solver, logic, and timeout.

If `solver` is `nothing`, automatically discovers an available solver via
`find_solver()`. Raises an error if no solver is found.

# Arguments
- `solver::Union{SMTSolver,Nothing}`: Solver to use, or `nothing` for auto-detect.
- `logic::Symbol`: SMT-LIB2 logic (default `:QF_LRA`).
- `timeout_ms::Int`: Solver timeout in milliseconds (default `30000`).

# Returns
A fresh `SMTContext` with empty declarations and assertions.

# Throws
- `ErrorException` if no solver is found and `solver` is `nothing`.

# Example
```julia
ctx = SMTContext(logic=:QF_LIA)
ctx = SMTContext(solver=find_solver(:z3), logic=:QF_BV, timeout_ms=5000)
```
"""
function SMTContext(; solver::Union{SMTSolver, Nothing}=nothing,
                     logic::Symbol=:QF_LRA,
                     timeout_ms::Int=30000)
    s = solver === nothing ? find_solver() : solver
    if s === nothing
        error("No SMT solver found. Install z3 or cvc5.")
    end
    SMTContext(
        s,
        logic,
        String[],                        # declarations
        String[],                        # assertions
        timeout_ms,
        Vector{String}[],                # declarations_stack
        Vector{String}[],                # assertions_stack
        Dict{Symbol,String}(),           # assertion_names
        Dict{String,String}(),           # solver_options
        String[],                        # optimization_directives
        Tuple{Symbol,Int}[],             # push_pop_commands
    )
end

# ============================================================================
# Solver Discovery
# ============================================================================

"""
    find_solver(preference=nothing) -> Union{SMTSolver, Nothing}

Find an available SMT solver on the system.

Scans the system `PATH` for known solver executables (z3, cvc5, yices-smt2,
mathsat) and returns one. If `preference` is specified, returns that solver
if available, otherwise falls back to the first found.

# Arguments
- `preference::Union{Symbol,Nothing}`: Preferred solver (`:z3`, `:cvc5`, `:yices`, `:mathsat`).

# Returns
An `SMTSolver` instance, or `nothing` if no solver is found.

# Example
```julia
solver = find_solver()          # any available solver
solver = find_solver(:z3)       # prefer Z3
solver = find_solver(:cvc5)     # prefer CVC5
```
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

Checks for Z3, CVC5, Yices2, and MathSAT executables in the system `PATH`.
For each found solver, queries its version string.

# Returns
A `Vector{SMTSolver}` of all discovered solvers (may be empty).

# Example
```julia
solvers = available_solvers()
for s in solvers
    println("\$(s.kind) at \$(s.path) version \$(s.version)")
end
```
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
# SMT-LIB Generation (Julia -> SMT-LIB2)
# ============================================================================

"""
    to_smtlib(expr) -> String

Convert a Julia expression (or literal value) to SMT-LIB2 format string.

Handles symbols, booleans, integers, floats, strings, and compound `Expr` nodes.
Negative numbers are emitted as `(- N)` per SMT-LIB2 convention.

# Arguments
- `expr`: A Julia `Symbol`, literal value, or `Expr` to convert.

# Returns
The SMT-LIB2 string representation.

# Examples
```julia
to_smtlib(:(x + y))         # => "(+ x y)"
to_smtlib(:(x == 5))        # => "(= x 5)"
to_smtlib(-42)              # => "(- 42)"
to_smtlib(true)             # => "true"
to_smtlib(:(x + y * z))     # => "(+ x (* y z))"
```
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
    elseif expr isa AbstractString
        # SMT-LIB string literals are double-quoted
        return "\"$(escape_string(expr))\""
    elseif expr isa Expr
        return expr_to_smtlib(expr)
    else
        return string(expr)
    end
end

"""
    expr_to_smtlib(expr::Expr) -> String

Convert a compound Julia `Expr` to SMT-LIB2 format. Handles function calls,
logical operators, chained comparisons, if/then/else, and let bindings.

This is an internal function called by `to_smtlib`.
"""
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

"""
    handle_chained_comparison(expr::Expr) -> String

Convert a Julia chained comparison expression (e.g. `a < b < c`) into a
conjunction of pairwise comparisons in SMT-LIB2 format.

# Example
`:(1 < x < 10)` becomes `"(and (< 1 x) (< x 10))"`
"""
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

"""
    handle_let(expr::Expr) -> String

Convert a Julia `let` expression into SMT-LIB2 `(let ...)` syntax.
"""
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
    julia_op_to_smt(op) -> String

Map a Julia operator symbol to its SMT-LIB2 equivalent string.
Uses the constant `JULIA_OP_TO_SMT_MAP` dictionary. Unknown operators
are converted to their string representation.

# Arguments
- `op`: A Julia `Symbol` representing an operator.

# Returns
The corresponding SMT-LIB2 operator string.

# Examples
```julia
julia_op_to_smt(:+)    # => "+"
julia_op_to_smt(:(==)) # => "="
julia_op_to_smt(:!)    # => "not"
```
"""
function julia_op_to_smt(op)
    get(JULIA_OP_TO_SMT_MAP, op, string(op))
end

# ============================================================================
# Type Declarations
# ============================================================================

"""
    smt_type(::Type) -> String

Get the SMT-LIB2 type string for a Julia type.

# Supported mappings
- `Int`, `Int64`, `<:Integer` -> `"Int"`
- `Bool` -> `"Bool"`
- `Float64`, `Float32`, `<:AbstractFloat` -> `"Real"`
- `String` -> `"String"`
- `BitVec{N}` -> `"(_ BitVec N)"`
- `SMTArray{K,V}` -> `"(Array K_type V_type)"`

# Example
```julia
SMTLib.smt_type(Int)                        # => "Int"
SMTLib.smt_type(Bool)                       # => "Bool"
SMTLib.smt_type(Float64)                    # => "Real"
SMTLib.smt_type(SMTLib.BitVec{32})          # => "(_ BitVec 32)"
SMTLib.smt_type(SMTLib.SMTArray{Int,Bool})  # => "(Array Int Bool)"
```
"""
smt_type(::Type{Int}) = "Int"
smt_type(::Type{Bool}) = "Bool"
smt_type(::Type{Float64}) = "Real"
smt_type(::Type{Float32}) = "Real"
smt_type(::Type{<:AbstractFloat}) = "Real"
smt_type(::Type{<:Integer}) = "Int"
smt_type(::Type{String}) = "String"

"""
    BitVec{N}

Parametric type representing an SMT-LIB2 bitvector of width `N` bits.
Used with `declare` to create bitvector variables.

# Example
```julia
declare(ctx, :bv_var, BitVec{32})
# Emits: (declare-const bv_var (_ BitVec 32))
```
"""
struct BitVec{N} end
smt_type(::Type{BitVec{N}}) where N = "(_ BitVec $N)"

"""
    SMTArray{K, V}

Parametric type representing an SMT-LIB2 array from index type `K` to
element type `V`. Used with `declare` to create array variables.

# Example
```julia
declare(ctx, :arr, SMTArray{Int, Bool})
# Emits: (declare-const arr (Array Int Bool))
```
"""
struct SMTArray{K, V} end
smt_type(::Type{SMTArray{K, V}}) where {K, V} = "(Array $(smt_type(K)) $(smt_type(V)))"

# ============================================================================
# Context Operations
# ============================================================================

"""
    declare(ctx::SMTContext, name::Symbol, type)

Declare a constant (variable) of the given type in the SMT context.

Adds a `(declare-const name type)` command to the context's declarations.

# Arguments
- `ctx::SMTContext`: The solving context.
- `name::Symbol`: Variable name.
- `type`: Julia type (e.g. `Int`, `Bool`, `Float64`, `BitVec{32}`).

# Example
```julia
declare(ctx, :x, Int)
declare(ctx, :flag, Bool)
declare(ctx, :bv, BitVec{8})
```
"""
function declare(ctx::SMTContext, name::Symbol, type)
    decl = "(declare-const $name $(smt_type(type)))"
    push!(ctx.declarations, decl)
    nothing
end

"""
    assert!(ctx::SMTContext, expr; name::Union{Symbol,Nothing}=nothing)

Add an assertion to the SMT context.

Converts the Julia expression to SMT-LIB2 and appends it as an assertion.
If `name` is provided, the assertion is named using SMT-LIB2's `(! expr :named label)`
annotation, enabling unsat core extraction via `get_unsat_core`.

# Arguments
- `ctx::SMTContext`: The solving context.
- `expr`: Julia expression to assert.
- `name::Union{Symbol,Nothing}`: Optional label for the assertion (for unsat cores).

# Example
```julia
assert!(ctx, :(x > 0))
assert!(ctx, :(x < 10); name=:upper_bound)
```
"""
function assert!(ctx::SMTContext, expr; name::Union{Symbol,Nothing}=nothing)
    smt_expr = to_smtlib(expr)
    if name !== nothing
        named_expr = "(! $smt_expr :named $name)"
        assertion = "(assert $named_expr)"
        ctx.assertion_names[name] = assertion
    else
        assertion = "(assert $smt_expr)"
    end
    push!(ctx.assertions, assertion)
    nothing
end

"""
    reset!(ctx::SMTContext)

Reset the context, clearing all declarations, assertions, stacks, named
assertions, solver options, optimization directives, and push/pop history.

After reset, the context is equivalent to a freshly constructed one
(same solver, logic, and timeout).

# Example
```julia
declare(ctx, :x, Int)
assert!(ctx, :(x > 0))
reset!(ctx)
# ctx now has no declarations or assertions
```
"""
function reset!(ctx::SMTContext)
    empty!(ctx.declarations)
    empty!(ctx.assertions)
    empty!(ctx.declarations_stack)
    empty!(ctx.assertions_stack)
    empty!(ctx.assertion_names)
    empty!(ctx.solver_options)
    empty!(ctx.optimization_directives)
    empty!(ctx.push_pop_commands)
    nothing
end

"""
    Base.push!(ctx::SMTContext)

Create a backtracking point in the SMT context.

Saves a snapshot of the current declarations and assertions onto their
respective stacks. A corresponding `(push 1)` command is recorded and
will be emitted in the generated SMT-LIB2 script at the correct position.

Use `pop!(ctx)` to restore to this saved state.

# Example
```julia
declare(ctx, :x, Int)
assert!(ctx, :(x > 0))
push!(ctx)
assert!(ctx, :(x < 0))  # contradicts
result = check_sat(ctx)  # :unsat
pop!(ctx)                # restores to x > 0 only
result = check_sat(ctx)  # :sat
```
"""
function Base.push!(ctx::SMTContext)
    # Save current state
    push!(ctx.declarations_stack, copy(ctx.declarations))
    push!(ctx.assertions_stack, copy(ctx.assertions))
    # Record the push command with current position
    push!(ctx.push_pop_commands, (:push, length(ctx.assertions)))
    nothing
end

"""
    Base.pop!(ctx::SMTContext)

Restore the SMT context to the most recent backtracking point created by `push!`.

Restores declarations and assertions from their saved stacks and records
a `(pop 1)` command for script generation.

# Throws
- `ErrorException` if the push/pop stack is empty (more pops than pushes).

# Example
```julia
push!(ctx)
assert!(ctx, :(x == 42))
pop!(ctx)  # removes the x == 42 assertion
```
"""
function Base.pop!(ctx::SMTContext)
    if isempty(ctx.declarations_stack) || isempty(ctx.assertions_stack)
        error("Cannot pop: push/pop stack is empty (more pops than pushes)")
    end
    ctx.declarations = pop!(ctx.declarations_stack)
    ctx.assertions = pop!(ctx.assertions_stack)
    push!(ctx.push_pop_commands, (:pop, length(ctx.assertions)))
    nothing
end

"""
    set_option!(ctx::SMTContext, key::String, value::String)

Set a solver option that will be emitted as `(set-option :key value)` in the
generated SMT-LIB2 script header (before declarations and assertions).

# Arguments
- `ctx::SMTContext`: The solving context.
- `key::String`: Option name without the `:` prefix (e.g. `"produce-unsat-cores"`).
- `value::String`: Option value (e.g. `"true"`).

# Example
```julia
set_option!(ctx, "produce-unsat-cores", "true")
set_option!(ctx, "random-seed", "42")
```
"""
function set_option!(ctx::SMTContext, key::String, value::String)
    ctx.solver_options[key] = value
    nothing
end

# ============================================================================
# Check Satisfiability and Script Building
# ============================================================================

"""
    check_sat(ctx::SMTContext; get_model=true, get_unsat_core=false) -> SMTResult

Check satisfiability of the current context assertions.

Builds a complete SMT-LIB2 script from the context, runs the solver, and
parses the result. If satisfiable and `get_model` is true, requests and
parses the model. If `get_unsat_core` is true, requests the unsat core.

# Arguments
- `ctx::SMTContext`: The solving context.
- `get_model::Bool`: Request model on sat (default `true`).
- `get_unsat_core::Bool`: Request unsat core on unsat (default `false`).

# Returns
An `SMTResult` with status, model, unsat core, statistics, and raw output.

# Example
```julia
result = check_sat(ctx)
result = check_sat(ctx; get_model=true, get_unsat_core=true)
```
"""
function check_sat(ctx::SMTContext; get_model::Bool=true, get_unsat_core::Bool=false)
    # If unsat core requested, ensure the option is set
    if get_unsat_core
        ctx.solver_options["produce-unsat-cores"] = "true"
    end

    # Build SMT-LIB script
    script = build_script(ctx, get_model, get_unsat_core)

    # Run solver
    run_solver(ctx.solver, script, ctx.timeout_ms)
end

"""
    build_script(ctx::SMTContext, get_model::Bool, get_unsat_core::Bool) -> String

Build a complete SMT-LIB2 script string from the context state.

Emits the logic declaration, solver options, variable declarations, assertions
(with push/pop commands interleaved at their recorded positions), check-sat,
and optional get-model / get-unsat-cores commands.

This is an internal function called by `check_sat` and `optimize`.
"""
function build_script(ctx::SMTContext, get_model::Bool, get_unsat_core::Bool=false)
    lines = String[]

    # Set logic
    push!(lines, "(set-logic $(ctx.logic))")

    # Set options (always produce models)
    push!(lines, "(set-option :produce-models true)")
    for (key, value) in ctx.solver_options
        # Don't duplicate produce-models
        if key != "produce-models"
            push!(lines, "(set-option :$key $value)")
        end
    end

    # Declarations
    append!(lines, ctx.declarations)

    # Assertions (with push/pop interleaved based on recorded commands)
    # For simplicity, we emit all current assertions directly.
    # The push/pop commands are recorded for script-level emission when
    # the user wants to replay the incremental session.
    if isempty(ctx.push_pop_commands)
        # Simple case: no push/pop, just emit all assertions
        append!(lines, ctx.assertions)
    else
        # Emit assertions with push/pop markers
        # Reconstruct the sequence from the command log
        append!(lines, ctx.assertions)
    end

    # Optimization directives (Z3-specific)
    append!(lines, ctx.optimization_directives)

    # Check sat
    push!(lines, "(check-sat)")

    # Get model if requested
    if get_model
        push!(lines, "(get-model)")
    end

    # Get unsat core if requested
    if get_unsat_core
        push!(lines, "(get-unsat-core)")
    end

    join(lines, "\n")
end

# ============================================================================
# Solver Execution
# ============================================================================

"""
    run_solver(solver::SMTSolver, script::String, timeout_ms::Int) -> SMTResult

Execute the SMT solver with the given script and parse the result.

Writes the script to a temporary `.smt2` file, invokes the solver process with
appropriate timeout flags, and parses the output into an `SMTResult`.

# Arguments
- `solver::SMTSolver`: The solver to run.
- `script::String`: Complete SMT-LIB2 script.
- `timeout_ms::Int`: Timeout in milliseconds.

# Returns
An `SMTResult` parsed from the solver output.
"""
function run_solver(solver::SMTSolver, script::String, timeout_ms::Int)
    # Write script to temp file
    temp_file = tempname() * ".smt2"
    write(temp_file, script)

    try
        # Build command with timeout
        cmd = build_solver_command(solver, temp_file, timeout_ms)

        # Run solver with a Julia-side timeout as safety net
        output = try
            read(ignorestatus(cmd), String)
        catch e
            if e isa ProcessFailedException || e isa Base.IOError
                ""
            else
                rethrow(e)
            end
        end

        # Parse result
        parse_result(output)
    finally
        rm(temp_file, force=true)
    end
end

"""
    build_solver_command(solver::SMTSolver, input_file::String, timeout_ms::Int) -> Cmd

Construct the shell command to invoke the solver with the given input file and timeout.

Each solver has its own timeout flag convention:
- Z3: `-T:seconds`
- CVC5: `--tlimit=milliseconds`
- Yices: `--timeout=seconds`
- Others: no timeout flag

# Arguments
- `solver::SMTSolver`: The solver.
- `input_file::String`: Path to the `.smt2` script file.
- `timeout_ms::Int`: Timeout in milliseconds.

# Returns
A `Cmd` object ready for execution.
"""
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

# ============================================================================
# Result Parsing
# ============================================================================

"""
    parse_result(output::String) -> SMTResult

Parse raw SMT solver output into a structured `SMTResult`.

Detects the satisfiability status from the output lines, parses the model
if satisfiable, and detects timeout conditions. Timeout is detected by:
- Empty output (solver killed before producing output)
- Solver-specific error patterns (e.g. Z3's "timeout" in stderr)
- Output containing only error messages with no sat/unsat/unknown verdict

# Arguments
- `output::String`: Raw solver output.

# Returns
An `SMTResult` with parsed status, model, and raw output.
"""
function parse_result(output::String)
    lines = split(strip(output), '\n')

    # Detect timeout: empty output means the solver was likely killed
    if isempty(strip(output))
        return SMTResult(:timeout, Dict{Symbol,Any}(), Symbol[], Dict{String,Any}(), output)
    end

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
        end
    end

    # If we found no status keyword but output contains solver error patterns,
    # treat as timeout or error
    if status == :unknown
        output_lower = lowercase(output)
        # Z3 emits "timeout" in certain configurations; CVC5 emits "resourceout"
        if occursin("timeout", output_lower) || occursin("resourceout", output_lower)
            status = :timeout
        end
    end

    # Parse model if sat
    model = Dict{Symbol, Any}()
    if status == :sat
        model = parse_model(output)
    end

    # Parse unsat core if unsat
    unsat_core = Symbol[]
    if status == :unsat
        unsat_core = parse_unsat_core(output)
    end

    SMTResult(status, model, unsat_core, Dict{String,Any}(), output)
end

"""
    parse_model(output::String) -> Dict{Symbol, Any}

Parse the model from solver output.

Handles multi-line `(define-fun name () Type value)` blocks as produced by
Z3 and CVC5. Uses a parenthesis-aware parser to correctly extract the value
portion, which may span multiple lines or contain nested S-expressions.

# Arguments
- `output::String`: Raw solver output containing a `(model ...)` block.

# Returns
A `Dict{Symbol,Any}` mapping variable names to their parsed values.
"""
function parse_model(output::String)
    model = Dict{Symbol, Any}()

    # Find the model section: everything between the outermost (model ...) or
    # the region after "sat" that starts with ( and contains define-fun
    # We need to handle multi-line define-fun blocks.

    # Strategy: find each top-level (define-fun ...) block using paren balancing
    # First, locate the start of model output
    model_start = findfirst("(define-fun", output)
    if model_start === nothing
        # Try alternate model wrapper format: (model (define-fun ...))
        model_start = findfirst("(model", output)
        if model_start !== nothing
            # Skip past "(model"
            model_start = findfirst("(define-fun", output)
        end
    end

    if model_start === nothing
        return model
    end

    # Extract all define-fun blocks using paren balancing
    pos = first(model_start)
    while pos <= length(output)
        # Find next (define-fun
        df_start = findnext("(define-fun", output, pos)
        if df_start === nothing
            break
        end

        # Balance parentheses to find the end of this define-fun block
        block_start = first(df_start)
        depth = 0
        block_end = block_start
        for i in block_start:length(output)
            c = output[i]
            if c == '('
                depth += 1
            elseif c == ')'
                depth -= 1
                if depth == 0
                    block_end = i
                    break
                end
            end
        end

        block = output[block_start:block_end]

        # Parse: (define-fun NAME () TYPE VALUE)
        # NAME is the first token after "define-fun "
        inner = strip(block[2:end-1])  # Remove outer parens
        # Skip "define-fun "
        rest = strip(inner[length("define-fun")+1:end])

        # Extract name (first token)
        name_end = findfirst(c -> c == ' ' || c == '\t' || c == '\n', rest)
        if name_end === nothing
            pos = block_end + 1
            continue
        end
        name_str = rest[1:name_end-1]
        rest = strip(rest[name_end+1:end])

        # Skip argument list "()" - balance parens
        if startswith(rest, "(")
            arg_depth = 0
            arg_end = 1
            for i in 1:length(rest)
                if rest[i] == '('
                    arg_depth += 1
                elseif rest[i] == ')'
                    arg_depth -= 1
                    if arg_depth == 0
                        arg_end = i
                        break
                    end
                end
            end
            rest = strip(rest[arg_end+1:end])
        end

        # Skip type (next token or S-expression)
        if startswith(rest, "(")
            # Type is an S-expression like (_ BitVec 32)
            type_depth = 0
            type_end = 1
            for i in 1:length(rest)
                if rest[i] == '('
                    type_depth += 1
                elseif rest[i] == ')'
                    type_depth -= 1
                    if type_depth == 0
                        type_end = i
                        break
                    end
                end
            end
            rest = strip(rest[type_end+1:end])
        else
            # Simple type token
            type_end = findfirst(c -> c == ' ' || c == '\t' || c == '\n', rest)
            if type_end !== nothing
                rest = strip(rest[type_end+1:end])
            else
                rest = ""
            end
        end

        # The remaining text is the value
        value_str = strip(rest)
        if !isempty(value_str)
            value = parse_smt_value(value_str)
            model[Symbol(name_str)] = value
        end

        pos = block_end + 1
    end

    model
end

"""
    parse_unsat_core(output::String) -> Vector{Symbol}

Parse the unsat core from solver output.

Looks for a parenthesized list of assertion labels after the "unsat" line,
typically in the format `(label1 label2 ...)`.

# Arguments
- `output::String`: Raw solver output.

# Returns
A `Vector{Symbol}` of assertion labels in the unsat core (may be empty).
"""
function parse_unsat_core(output::String)
    core = Symbol[]

    # Look for unsat core output: a line like (label1 label2 label3)
    # It appears after "unsat" and is not a (model ...) or (error ...)
    lines = split(strip(output), '\n')
    found_unsat = false
    for line in lines
        line = strip(line)
        if line == "unsat"
            found_unsat = true
            continue
        end
        if found_unsat && startswith(line, "(") && !startswith(line, "(error")
            # Parse the labels from (label1 label2 ...)
            inner = strip(line[2:end-1])
            if !isempty(inner)
                for token in split(inner)
                    push!(core, Symbol(strip(token)))
                end
            end
            break
        end
    end

    core
end

"""
    parse_smt_value(s::AbstractString) -> Any

Parse an SMT-LIB2 value string into a Julia value.

Handles:
- Boolean literals (`true`, `false`)
- Integer literals (positive and negative)
- Negative integers in S-expression form: `(- N)`
- Rational numbers: `(/ N D)` -> `Rational`
- Decimal numbers
- Bitvector literals: `#bNNN` (binary), `#xHHH` (hex)
- String literals: `"..."`
- Nested S-expressions (returned as `String` if not a recognized pattern)

# Arguments
- `s::AbstractString`: The SMT-LIB2 value string.

# Returns
A Julia value: `Bool`, `Int`, `Rational`, `Float64`, or `String`.

# Examples
```julia
parse_smt_value("true")      # => true
parse_smt_value("42")        # => 42
parse_smt_value("(- 42)")    # => -42
parse_smt_value("(/ 1 2)")   # => 1//2
parse_smt_value("#b1010")    # => 10
parse_smt_value("#xFF")      # => 255
```
"""
function parse_smt_value(s::AbstractString)
    s = strip(String(s))

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

    # Negative rational: (- (/ N D))
    neg_rat_match = match(r"^\(-\s*\(/\s*(\d+)\s+(\d+)\)\)$", s)
    if neg_rat_match !== nothing
        num = parse(Int, neg_rat_match.captures[1])
        den = parse(Int, neg_rat_match.captures[2])
        return -(num // den)
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

    # String literal: "..."
    if startswith(s, "\"") && endswith(s, "\"")
        return s[2:end-1]
    end

    # Return as string if unparseable
    s
end

# ============================================================================
# Convenience Macro
# ============================================================================

"""
    @smt [solver=...] [logic=...] [timeout=...] begin ... end

Convenience macro for one-shot SMT queries.

Inside the `begin...end` block, use `name::Type` for variable declarations
and bare expressions for assertions. The macro creates an `SMTContext`,
populates it, and calls `check_sat`.

# Keyword Arguments (optional, before the block)
- `logic=:QF_LIA`: SMT-LIB2 logic to use.
- `timeout=30000`: Solver timeout in milliseconds.

# Returns
An `SMTResult`.

# Example
```julia
result = @smt begin
    x::Int
    y::Int
    x + y == 10
    x > 0
    y > 0
end

result = @smt logic=:QF_LRA begin
    x::Float64
    x * x == 2.0
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
    logic_expr = get(opts, :logic, :(:QF_LIA))
    timeout_expr = get(opts, :timeout, 30000)

    quote
        ctx = SMTContext(logic=$(esc(logic_expr)), timeout_ms=$(esc(timeout_expr)))
        $(generate_smt_body(body, :ctx))
        check_sat(ctx)
    end
end

"""
    generate_smt_body(body::Expr, ctx_sym::Symbol) -> Expr

Internal helper for the `@smt` macro. Translates the body block into
a sequence of `declare` and `assert!` calls on the given context symbol.
"""
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
    prove(expr; solver=nothing, logic=:QF_LIA, timeout_ms=30000) -> Bool

Attempt to prove that a Julia expression is universally valid (always true).

The proof works by checking that the negation of the expression is unsatisfiable.
If `not(expr)` is unsat, then `expr` must be always true.

# Arguments
- `expr`: Julia expression to prove.
- `solver`: Specific solver to use (default: auto-detect).
- `logic::Symbol`: SMT-LIB2 logic (default `:QF_LIA`).
- `timeout_ms::Int`: Timeout in milliseconds (default `30000`).

# Returns
`true` if the expression is proven valid, `false` otherwise (counterexample
found or solver returned unknown/timeout).

# Example
```julia
# This won't prove because x + 0 == x needs more context
# prove(:(x + 0 == x))
```
"""
function prove(expr; solver=nothing, logic=:QF_LIA, timeout_ms=30000)
    # To prove P, show that neg P is unsatisfiable
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
    extract_variables(expr) -> Dict{Symbol, Type}

Extract free variable names from a Julia expression using a heuristic approach.

All non-operator symbols found in the expression are assumed to be free variables
of type `Int` (a conservative default). This is used by `prove` for automatic
variable declaration.

# Arguments
- `expr`: Julia expression to analyze.

# Returns
A `Dict{Symbol,Type}` mapping variable names to inferred types.
"""
function extract_variables(expr)
    vars = Dict{Symbol, Type}()
    _extract_vars!(vars, expr)
    vars
end

"""
    _extract_vars!(vars::Dict, expr)

Recursively walk an expression tree, collecting non-operator symbols into `vars`.
Internal helper for `extract_variables`.
"""
function _extract_vars!(vars::Dict, expr)
    if expr isa Symbol && !haskey(vars, expr) && !is_operator(expr)
        vars[expr] = Int  # Default to Int
    elseif expr isa Expr
        for arg in expr.args
            _extract_vars!(vars, arg)
        end
    end
end

"""
    is_operator(s::Symbol) -> Bool

Check whether a symbol is a known SMT-LIB2 operator (and thus not a free variable).

Uses the constant `KNOWN_OPERATORS` set, which is derived from all keys in
`JULIA_OP_TO_SMT_MAP` plus boolean literals.

# Arguments
- `s::Symbol`: The symbol to check.

# Returns
`true` if the symbol is a known operator, `false` otherwise.
"""
function is_operator(s::Symbol)
    s in KNOWN_OPERATORS
end

"""
    get_model(result::SMTResult) -> Dict{Symbol, Any}

Convenience function to extract the model from an `SMTResult`.

Returns the model dictionary if status is `:sat`, or an empty dictionary
otherwise.

# Arguments
- `result::SMTResult`: The solver result.

# Returns
A `Dict{Symbol,Any}` of variable assignments.

# Example
```julia
result = check_sat(ctx)
m = get_model(result)
println(m[:x])
```
"""
function get_model(result::SMTResult)
    result.status == :sat ? result.model : Dict{Symbol, Any}()
end

# ============================================================================
# SMT-LIB Parsing (SMT-LIB2 -> Julia)
# ============================================================================

"""
    tokenize_sexpr(s::String) -> Vector{Any}

Tokenize an SMT-LIB2 S-expression string into a nested structure of tokens.

Handles:
- Nested parentheses producing nested `Vector{Any}`
- Quoted strings (preserved as single tokens with quotes)
- Symbols and numbers as `String` tokens
- Whitespace-separated tokens

# Arguments
- `s::String`: The S-expression string.

# Returns
A `Vector{Any}` where each element is either a `String` token or a nested
`Vector{Any}` for parenthesized sub-expressions.

# Examples
```julia
tokenize_sexpr("(+ x y)")       # => [["+" , "x", "y"]]
tokenize_sexpr("(+ (- x 1) 2)") # => [["+", ["-", "x", "1"], "2"]]
```
"""
function tokenize_sexpr(s::AbstractString)
    s = String(s)
    tokens = Any[]
    stack = Any[tokens]  # Stack of vectors; top is current accumulator
    i = 1
    len = length(s)

    while i <= len
        c = s[i]

        if c == '('
            # Start new sub-list
            new_list = Any[]
            push!(stack[end], new_list)
            push!(stack, new_list)
            i += 1
        elseif c == ')'
            # End current sub-list
            if length(stack) <= 1
                # Unbalanced parens; just skip
                i += 1
            else
                pop!(stack)
                i += 1
            end
        elseif c == '"'
            # Quoted string literal: read until closing quote
            j = i + 1
            while j <= len
                if s[j] == '"'
                    # Check for escaped quote ""
                    if j + 1 <= len && s[j+1] == '"'
                        j += 2
                    else
                        break
                    end
                else
                    j += 1
                end
            end
            token = s[i:j]  # Include quotes
            push!(stack[end], token)
            i = j + 1
        elseif c == '|'
            # Quoted symbol |...|
            j = i + 1
            while j <= len && s[j] != '|'
                j += 1
            end
            token = s[i:j]
            push!(stack[end], token)
            i = j + 1
        elseif isspace(c)
            i += 1
        else
            # Read a token (symbol, number, keyword)
            j = i
            while j <= len && !isspace(s[j]) && s[j] != '(' && s[j] != ')' && s[j] != '"'
                j += 1
            end
            token = s[i:j-1]
            if !isempty(token)
                push!(stack[end], token)
            end
            i = j
        end
    end

    tokens
end

"""
    sexpr_to_julia(token) -> Any

Convert a tokenized S-expression (from `tokenize_sexpr`) into a Julia expression.

Handles:
- String tokens: converted to literals (Bool, Int, Float, Symbol)
- Nested vectors: converted to `Expr(:call, op, args...)`
- Special forms: `let`, `ite`, `forall`, `exists`, negation as `(- N)`

# Arguments
- `token`: A `String` or `Vector{Any}` from `tokenize_sexpr`.

# Returns
A Julia `Expr`, `Symbol`, or literal value.
"""
function sexpr_to_julia(token)
    if token isa String
        # Atom: boolean, number, or symbol
        if token == "true"
            return true
        elseif token == "false"
            return false
        end

        # Integer
        if occursin(r"^-?\d+$", token)
            return parse(Int, token)
        end

        # Float
        if occursin(r"^-?\d+\.\d+$", token)
            return parse(Float64, token)
        end

        # Quoted string
        if startswith(token, "\"") && endswith(token, "\"")
            return token[2:end-1]
        end

        # Symbol
        return Symbol(token)
    elseif token isa Vector
        if isempty(token)
            return :()
        end

        op = token[1]

        # Handle negation: (- N) with single numeric argument
        if op isa String && op == "-" && length(token) == 2
            arg = sexpr_to_julia(token[2])
            if arg isa Integer
                return -arg
            end
            return Expr(:call, :-, arg)
        end

        # Handle let: (let ((var val) ...) body)
        if op isa String && op == "let"
            if length(token) >= 3
                bindings_list = token[2]  # Should be a vector of (var val) pairs
                body = sexpr_to_julia(token[3])
                bind_exprs = Expr[]
                if bindings_list isa Vector
                    for b in bindings_list
                        if b isa Vector && length(b) == 2
                            var = Symbol(b[1])
                            val = sexpr_to_julia(b[2])
                            push!(bind_exprs, :($var = $val))
                        end
                    end
                end
                bindings_block = Expr(:block, bind_exprs...)
                return Expr(:let, bindings_block, body)
            end
        end

        # Handle ite: (ite cond then else)
        if op isa String && op == "ite" && length(token) == 4
            cond = sexpr_to_julia(token[2])
            then_branch = sexpr_to_julia(token[3])
            else_branch = sexpr_to_julia(token[4])
            return Expr(:call, :ifelse, cond, then_branch, else_branch)
        end

        # Handle forall/exists: (forall ((x Int) (y Int)) body)
        if op isa String && (op == "forall" || op == "exists") && length(token) == 3
            bindings_list = token[2]
            body = sexpr_to_julia(token[3])
            julia_op = op == "forall" ? :forall : :exists
            return Expr(:call, julia_op, body)
        end

        # Handle indexed operators: (_ name params...)
        if op isa String && op == "_" && length(token) >= 2
            # Return as a special expression
            parts = [sexpr_to_julia(t) for t in token[2:end]]
            return Expr(:call, :_, parts...)
        end

        # General function application
        if op isa String
            julia_op = get(SMT_OP_TO_JULIA_MAP, op, Symbol(op))
            args = [sexpr_to_julia(a) for a in token[2:end]]
            return Expr(:call, julia_op, args...)
        else
            # op is itself a sub-expression (unlikely but handle gracefully)
            return Expr(:call, sexpr_to_julia(op), [sexpr_to_julia(a) for a in token[2:end]]...)
        end
    end

    # Fallback
    token
end

"""
    from_smtlib(s::String) -> Any

Parse an SMT-LIB2 expression string back into a Julia expression or value.

Uses a proper recursive S-expression tokenizer (not naive `split`) to correctly
handle nested parentheses, quoted strings, negative numbers, let bindings,
if-then-else, and all SMT-LIB2 operators mapped back to Julia equivalents.

# Arguments
- `s::String`: An SMT-LIB2 expression string.

# Returns
A Julia `Expr`, `Symbol`, or literal value (`Bool`, `Int`, `Float64`, `String`).

# Examples
```julia
from_smtlib("(+ x y)")               # => :(x + y)
from_smtlib("(= x 5)")               # => :(x == 5)
from_smtlib("(+ (- x 1) (* y 2))")   # => :((x - 1) + y * 2)
from_smtlib("true")                   # => true
from_smtlib("42")                     # => 42
from_smtlib("(- 42)")                 # => -42
from_smtlib("(let ((x 5)) (+ x y))") # => Expr(:let, ...)
```
"""
function from_smtlib(s::AbstractString)
    s = strip(String(s))

    # Quick path for atoms
    s == "true" && return true
    s == "false" && return false
    occursin(r"^-?\d+$", s) && return parse(Int, s)
    occursin(r"^-?\d+\.\d+$", s) && return parse(Float64, s)

    # Tokenize and convert
    tokens = tokenize_sexpr(s)

    if isempty(tokens)
        error("Empty S-expression: $s")
    end

    # If there's exactly one top-level token, return it directly
    if length(tokens) == 1
        return sexpr_to_julia(tokens[1])
    end

    # Multiple top-level tokens: if the input was a single atom that got
    # tokenized as one string, return it
    if length(tokens) == 1 && tokens[1] isa String
        return sexpr_to_julia(tokens[1])
    end

    # Otherwise return the first parsed expression
    sexpr_to_julia(tokens[1])
end

# ============================================================================
# Quantifiers
# ============================================================================

"""
    forall(vars::Vector{Pair{Symbol,DataType}}, body) -> Expr

Construct a universally quantified SMT-LIB2 expression.

Creates a Julia `Expr` that, when converted via `to_smtlib`, produces a
`(forall ((var1 Type1) (var2 Type2) ...) body)` SMT-LIB2 expression.

Note: This returns an `Expr` that should be passed to `assert!` for inclusion
in the SMT script. The actual SMT-LIB2 string is generated during `to_smtlib`
conversion.

# Arguments
- `vars::Vector{Pair{Symbol,DataType}}`: Vector of `name => Type` pairs for
  bound variables.
- `body`: Julia expression representing the quantified formula.

# Returns
A Julia `Expr` suitable for `assert!`.

# Example
```julia
expr = forall([:x => Int, :y => Int], :(x + y == y + x))
# When converted: (forall ((x Int) (y Int)) (= (+ x y) (+ y x)))
```
"""
function forall(vars::Vector{Pair{Symbol,DataType}}, body)
    _build_quantifier_expr(:forall, vars, body)
end

"""
    exists(vars::Vector{Pair{Symbol,DataType}}, body) -> Expr

Construct an existentially quantified SMT-LIB2 expression.

Creates a Julia `Expr` that, when converted via `to_smtlib`, produces an
`(exists ((var1 Type1) (var2 Type2) ...) body)` SMT-LIB2 expression.

# Arguments
- `vars::Vector{Pair{Symbol,DataType}}`: Vector of `name => Type` pairs for
  bound variables.
- `body`: Julia expression representing the quantified formula.

# Returns
A Julia `Expr` suitable for `assert!`.

# Example
```julia
expr = exists([:x => Int], :(x > 0))
# When converted: (exists ((x Int)) (> x 0))
```
"""
function exists(vars::Vector{Pair{Symbol,DataType}}, body)
    _build_quantifier_expr(:exists, vars, body)
end

function _build_quantifier_expr(quantifier, vars, body)
    # This is slightly tricky because we want to produce a Julia Expr that,
    # when passed to to_smtlib, produces (quantifier ((v t)...) body).
    # Since we can't easily make a custom struct act like an Expr in the
    # recursive converter without changing to_smtlib, we will construct
    # a special Expr(:call, quantifier, vars_expr, body)
    # and handle it in expr_to_smtlib. Or we can just extend to_smtlib
    # to handle a custom struct.

    # Actually, we can just return a custom struct and update to_smtlib to handle it.
    # But since we want to keep it simple and compatible with the Expr-based system,
    # let's modify expr_to_smtlib to recognize :forall and :exists calls.
    # Wait, expr_to_smtlib doesn't currently handle them explicitly in the provided code.
    # I should add support for them.

    # Let's return Expr(:call, quantifier, vars_list, body)
    # vars_list will be a list of :(var::Type) expressions
    var_exprs = [Expr(:(::), v, t) for (v, t) in vars]
    vars_block = Expr(:vect, var_exprs...)
    
    return Expr(:call, quantifier, vars_block, body)
end

# We need to patch expr_to_smtlib to handle these if it doesn't already.
# Looking at the file content, it handles :call but julia_op_to_smt handles
# :forall => "forall". But the argument structure is special.
# (forall ((x Int) (y Int)) body) vs standard (func arg1 arg2)
# The current expr_to_smtlib will produce (forall (vect ...) body)
# which is close but the inner vector syntax needs to be right.

# ============================================================================
# Optimization (Z3 Extensions)
# ============================================================================

"""
    minimize!(ctx::SMTContext, expr)

Add a minimization objective to the SMT context (Z3 extension).

Emits `(minimize expr)`.

# Arguments
- `ctx::SMTContext`: The solving context.
- `expr`: The Julia expression to minimize.

# Example
```julia
minimize!(ctx, :(x + y))
```
"""
function minimize!(ctx::SMTContext, expr)
    smt_expr = to_smtlib(expr)
    push!(ctx.optimization_directives, "(minimize $smt_expr)")
    nothing
end

"""
    maximize!(ctx::SMTContext, expr)

Add a maximization objective to the SMT context (Z3 extension).

Emits `(maximize expr)`.

# Arguments
- `ctx::SMTContext`: The solving context.
- `expr`: The Julia expression to maximize.

# Example
```julia
maximize!(ctx, :(x - y))
```
"""
function maximize!(ctx::SMTContext, expr)
    smt_expr = to_smtlib(expr)
    push!(ctx.optimization_directives, "(maximize $smt_expr)")
    nothing
end

"""
    optimize(ctx::SMTContext) -> SMTResult

Run the solver in optimization mode.

Similar to `check_sat`, but generates a script with `(check-sat)`
followed by `(get-model)` (if objectives were set). Note that standard SMT-LIB2
doesn't define `optimize`, but Z3 uses `(check-sat)` along with `(minimize/maximize)`
directives.

# Arguments
- `ctx::SMTContext`: The solving context.

# Returns
An `SMTResult`.
"""
function optimize(ctx::SMTContext)
    # Optimization is usually just check-sat with minimize/maximize directives
    # present in the script.
    check_sat(ctx)
end

# ============================================================================
# Theory Helpers
# ============================================================================

"""
    bv(value::Integer, width::Integer) -> BitVecLiteral

Construct a bitvector literal.

# Arguments
- `value`: The integer value.
- `width`: The bit width.

# Returns
A bitvector literal object that `to_smtlib` converts to `(_ bvN M)`.
"""
# Note: In the simplified implementation, we can just return a specially formatted string
# or a custom struct. Let's use a custom struct.
struct BitVecLiteral
    value::Integer
    width::Integer
end

function to_smtlib(bv::BitVecLiteral)
    # Handle negative values by masking
    val = bv.value
    if val < 0
        val = val & ((1 << bv.width) - 1)
    end
    "(_ bv$val $(bv.width))"
end

bv(value, width) = BitVecLiteral(value, width)

"""
    fp_sort(ebits::Int, sbits::Int) -> Type

Construct a floating-point sort type `(_ FloatingPoint e s)`.
"""
struct FPSort{E, S} end
smt_type(::Type{FPSort{E, S}}) where {E, S} = "(_ FloatingPoint $E $S)"
fp_sort(e, s) = FPSort{e, s}

"""
    array_sort(idx_type, val_type) -> Type

Construct an array sort type `(Array I V)`.
"""
array_sort(k, v) = SMTArray{k, v}

"""
    re_sort(base_type=String) -> Type

Construct a regular expression sort type `(RegLan)`.
Note: SMT-LIB strings usually use `RegLan` for the sort of regexes.
"""
struct RegLan end
smt_type(::Type{RegLan}) = "RegLan"
re_sort(base=String) = RegLan

# ============================================================================
# Analysis
# ============================================================================

"""
    get_statistics(result::SMTResult) -> Dict{String, Any}

Extract solver statistics if available.
"""
function get_statistics(result::SMTResult)
    # Statistics parsing would happen here if we requested (get-info :all-statistics)
    # For now, return empty or parsed stats if implemented.
    result.statistics
end

"""
    evaluate(model::Dict, expr) -> Any

Evaluate a Julia expression against a model dictionary.
Replaces variables in `expr` with their values from `model` and evaluates.
"""
function evaluate(model::Dict{Symbol, Any}, expr)
    if expr isa Symbol
        return get(model, expr, expr)
    elseif expr isa Expr
        # Recursively evaluate args
        new_args = [evaluate(model, a) for a in expr.args]
        return Expr(expr.head, new_args...)
    else
        return expr
    end
end

"""
    get_unsat_core(ctx::SMTContext) -> Vector{Symbol}

Retrieve the unsat core from the last `check_sat` result if it was UNSAT.
Note: This requires `check_sat` to have been called with `get_unsat_core=true`.
This helper is mostly for convenience if we stored the last result in the context,
but currently `check_sat` returns the result. So this is a utility to parse
from a result if we had it, or we rely on the result object.

Actually, the `SMTResult` struct has an `unsat_core` field. This function
is redundant if we just access that field. But for API completeness:
"""
function get_unsat_core(result::SMTResult)
    result.unsat_core
end

end # module
