# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using SMTLib

@testset "SMTLib.jl" begin

    @testset "Solver Discovery" begin
        solvers = available_solvers()
        @test solvers isa Vector{SMTSolver}
        # Note: May be empty if no solvers installed
    end

    @testset "SMT-LIB Generation" begin
        # Basic expressions
        @test to_smtlib(:(x + y)) == "(+ x y)"
        @test to_smtlib(:(x - y)) == "(- x y)"
        @test to_smtlib(:(x * y)) == "(* x y)"
        @test to_smtlib(:(x / y)) == "(/ x y)"

        # Comparisons
        @test to_smtlib(:(x == y)) == "(= x y)"
        @test to_smtlib(:(x < y)) == "(< x y)"
        @test to_smtlib(:(x <= y)) == "(<= x y)"
        @test to_smtlib(:(x > y)) == "(> x y)"
        @test to_smtlib(:(x >= y)) == "(>= x y)"

        # Logical operators
        @test to_smtlib(:(!x)) == "(not x)"
        @test to_smtlib(:(x && y)) == "(and x y)"
        @test to_smtlib(:(x || y)) == "(or x y)"

        # Literals
        @test to_smtlib(true) == "true"
        @test to_smtlib(false) == "false"
        @test to_smtlib(42) == "42"
        @test to_smtlib(-42) == "(- 42)"
        @test to_smtlib(3.14) == "3.14"

        # Nested expressions
        @test to_smtlib(:(x + y * z)) == "(+ x (* y z))"
        @test to_smtlib(:((x + y) * z)) == "(* (+ x y) z)"
    end

    @testset "Type Mapping" begin
        @test SMTLib.smt_type(Int) == "Int"
        @test SMTLib.smt_type(Int64) == "Int"
        @test SMTLib.smt_type(Bool) == "Bool"
        @test SMTLib.smt_type(Float64) == "Real"
        @test SMTLib.smt_type(SMTLib.BitVec{32}) == "(_ BitVec 32)"
        @test SMTLib.smt_type(SMTLib.SMTArray{Int, Int}) == "(Array Int Int)"
    end

    @testset "Value Parsing" begin
        @test SMTLib.parse_smt_value("true") == true
        @test SMTLib.parse_smt_value("false") == false
        @test SMTLib.parse_smt_value("42") == 42
        @test SMTLib.parse_smt_value("(- 42)") == -42
        @test SMTLib.parse_smt_value("(/ 1 2)") == 1//2
        @test SMTLib.parse_smt_value("#b1010") == 10
        @test SMTLib.parse_smt_value("#xFF") == 255
    end

    @testset "Context Operations" begin
        solver = find_solver()
        if solver !== nothing
            ctx = SMTContext(solver=solver, logic=:QF_LIA)

            declare(ctx, :x, Int)
            declare(ctx, :y, Int)
            assert!(ctx, :(x + y == 10))
            assert!(ctx, :(x > 0))
            assert!(ctx, :(y > 0))

            result = check_sat(ctx)
            @test result.status == :sat
            @test haskey(result.model, :x)
            @test haskey(result.model, :y)

            if haskey(result.model, :x) && haskey(result.model, :y)
                x_val = result.model[:x]
                y_val = result.model[:y]
                if x_val isa Number && y_val isa Number
                    @test x_val + y_val == 10
                    @test x_val > 0
                    @test y_val > 0
                end
            end
        else
            @warn "No SMT solver found, skipping solver tests"
        end
    end

    @testset "Unsatisfiable" begin
        solver = find_solver()
        if solver !== nothing
            ctx = SMTContext(solver=solver, logic=:QF_LIA)

            declare(ctx, :x, Int)
            assert!(ctx, :(x > 0))
            assert!(ctx, :(x < 0))

            result = check_sat(ctx)
            @test result.status == :unsat
        end
    end

    @testset "Real Arithmetic" begin
        solver = find_solver()
        if solver !== nothing
            ctx = SMTContext(solver=solver, logic=:QF_LRA)

            declare(ctx, :x, Float64)
            declare(ctx, :y, Float64)
            assert!(ctx, :(x + y == 1.0))
            assert!(ctx, :(x - y == 0.5))

            result = check_sat(ctx)
            @test result.status == :sat
        end
    end

end
