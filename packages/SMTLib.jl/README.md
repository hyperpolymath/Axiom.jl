# SMTLib.jl

A lightweight Julia interface to SMT solvers via SMT-LIB2 format.

> **Standalone Package**: This package is designed to be published independently.
> Currently bundled with [Axiom.jl](https://github.com/hyperpolymath/axiom.jl) at `packages/SMTLib.jl`.
> To publish separately, copy this directory to a new repository.

## Features

- **Auto-detection** of installed SMT solvers (Z3, CVC5, Yices, MathSAT)
- **Julia expression to SMT-LIB2** conversion
- **Multiple logics**: QF_LIA, QF_LRA, QF_NRA, QF_BV, arrays, and more
- **Model parsing** and counterexample extraction
- **Timeout support**
- **Incremental solving** with push/pop semantics
- **Zero dependencies** - pure Julia

## Installation

```julia
using Pkg

# From Axiom.jl monorepo
Pkg.develop(path="packages/SMTLib.jl")

# Or from standalone repo (when published)
# Pkg.add(url="https://github.com/hyperpolymath/SMTLib.jl")
```

### Prerequisites

Install at least one SMT solver:

```bash
# Z3 (recommended)
brew install z3          # macOS
apt install z3           # Ubuntu/Debian
pacman -S z3             # Arch

# CVC5
brew install cvc5
apt install cvc5
```

## Quick Start

```julia
using SMTLib

# Simple satisfiability check
ctx = SMTContext(logic=:QF_LIA)

declare(ctx, :x, Int)
declare(ctx, :y, Int)

assert!(ctx, :(x + y == 10))
assert!(ctx, :(x > 0))
assert!(ctx, :(y > 0))

result = check_sat(ctx)

if result.status == :sat
    println("x = ", result.model[:x])
    println("y = ", result.model[:y])
end
```

### Using the @smt Macro

```julia
result = @smt logic=:QF_LIA begin
    x::Int
    y::Int
    x + y == 10
    x > 0
    y > 0
end

println(result.status)  # :sat
println(result.model)   # Dict(:x => 5, :y => 5)
```

### Proving Properties

```julia
# Prove that x^2 >= 0 for all real x
# (Returns true if proven)
proven = prove(:(x * x >= 0), logic=:QF_NRA)
```

## Supported Logics

| Logic | Description |
|-------|-------------|
| `QF_LIA` | Quantifier-free linear integer arithmetic |
| `QF_LRA` | Quantifier-free linear real arithmetic |
| `QF_NIA` | Quantifier-free nonlinear integer arithmetic |
| `QF_NRA` | Quantifier-free nonlinear real arithmetic |
| `QF_BV` | Quantifier-free bitvectors |
| `QF_AUFLIA` | Arrays, uninterpreted functions, linear integer arithmetic |
| `ALL` | All supported theories |

## API Reference

### Solver Discovery

```julia
available_solvers()        # List all detected solvers
find_solver()              # Get first available solver
find_solver(:z3)           # Get specific solver
```

### Context Management

```julia
ctx = SMTContext(logic=:QF_LIA, timeout_ms=30000)
declare(ctx, :x, Int)      # Declare variable
assert!(ctx, expr)         # Add assertion
check_sat(ctx)             # Check satisfiability
reset!(ctx)                # Clear context
```

### SMT-LIB Conversion

```julia
to_smtlib(:(x + y == 10))  # "(= (+ x y) 10)"
```

### Types

```julia
# Built-in types
Int, Float64, Bool

# Bitvectors
BitVec{32}   # 32-bit bitvector

# Arrays
SMTArray{Int, Int}  # Array from Int to Int
```

## Examples

### Sudoku Solver

```julia
function solve_sudoku(grid)
    ctx = SMTContext(logic=:QF_LIA)

    # Variables: cell[i,j] for 1 ≤ i,j ≤ 9
    for i in 1:9, j in 1:9
        declare(ctx, Symbol("c_$(i)_$(j)"), Int)
        assert!(ctx, :(1 <= $(Symbol("c_$(i)_$(j)")) <= 9))
    end

    # Row constraints, column constraints, box constraints...
    # ... (implementation details)

    result = check_sat(ctx)
    # Extract solution from result.model
end
```

### Verification

```julia
# Verify array bounds check is sufficient
ctx = SMTContext(logic=:QF_LIA)

declare(ctx, :i, Int)
declare(ctx, :n, Int)

# Precondition: 0 <= i < n
assert!(ctx, :(i >= 0))
assert!(ctx, :(i < n))
assert!(ctx, :(n > 0))

# Try to prove out-of-bounds is impossible
# (check if i < 0 || i >= n is satisfiable under preconditions)
assert!(ctx, :(i < 0 || i >= n))

result = check_sat(ctx)
@assert result.status == :unsat  # Proven safe!
```

## License

MIT License - see LICENSE file.

## Acknowledgments

Extracted from [Axiom.jl](https://github.com/hyperpolymath/axiom.jl), a provably correct ML framework.
