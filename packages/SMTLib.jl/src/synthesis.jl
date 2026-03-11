# SPDX-License-Identifier: PMPL-1.0-or-later
module Synthesis

using ..SMTLib

export cegis, synthesize_constant

"""
    cegis(spec_func, input_vars; solver=nothing, max_iters=100)

Counter-Example Guided Inductive Synthesis (CEGIS) loop.

Attempts to find a constant value `C` such that `spec_func(C, input)` holds for all inputs.
This is a simplified 0-th order synthesis (finding magic constants).

# Arguments
- `spec_func`: A function `f(C, input_val)` that returns a boolean expression.
               `C` is the constant to synthesize, `input_val` is a test input.
- `input_vars`: A list of variable names (Symbols) and their types (e.g. `[:x => Int]`) representing the input space.

# Example
```julia
# Synthesize C such that x + C > 10 for x > 5
# spec: (C, x) -> (x > 5) => (x + C > 10)
result = cegis((c, x) -> (x > 5) ==> (x + c > 10), [:x => Int])
```
"""
function cegis(spec_func, input_vars; solver=nothing, logic=:QF_LIA, max_iters=100)
    # 1. Setup Synthesis Context (find C)
    synth_ctx = SMTContext(solver=solver, logic=logic)
    declare(synth_ctx, :C, Int) # Currently supports only single Int constant

    # 2. Setup Verification Context (find counterexample x given C)
    verify_ctx = SMTContext(solver=solver, logic=logic)
    declare(verify_ctx, :C, Int)
    for (name, type) in input_vars
        declare(verify_ctx, name, type)
    end

    # Inputs collected so far
    inputs = [] 

    for i in 1:max_iters
        # --- SYNTHESIS STEP ---
        push!(synth_ctx)
        # Add constraints for all known inputs
        for input_val in inputs
            # spec_func(C, input_val) must be true
            # Note: spec_func must return an Expr that uses :C and literal input_val
            # We need to construct the expression where 'x' is replaced by its value
            # Simplified: assuming single input variable for now
            # Actually, spec_func should probably take Expr/Symbol for C and *values* for x
            
            # Let's assume spec_func returns an Expr like :(x > 5 ==> x + C > 10)
            # We need to substitute the *value* of inputs into this expression?
            # Or simpler: The user provides a generator function.
            
            # Let's simplify: spec_func(C_sym, input_vals...) returns Expr
            constraint = spec_func(:C, input_val...)
            assert!(synth_ctx, constraint)
        end

        res_synth = check_sat(synth_ctx)
        if res_synth.status != :sat
            return (:unsat, "Could not synthesize a candidate")
        end
        
        candidate_C = res_synth.model[:C]
        pop!(synth_ctx)

        # --- VERIFICATION STEP ---
        push!(verify_ctx)
        # Assert C is the candidate
        assert!(verify_ctx, :(C == $candidate_C))
        
        # Assert NOT (spec holds for all x) <==> EXISTS x such that NOT spec
        # We assume spec_func returns P(C, x). We check SAT( !P(C, x) )
        # If SAT, we found a counterexample.
        
        # We need to pass the *symbol* :x to spec_func here, not a value
        input_syms = [n for (n, t) in input_vars]
        spec_expr = spec_func(:C, input_syms...)
        neg_spec = Expr(:call, :!, spec_expr)
        
        assert!(verify_ctx, neg_spec)
        
        res_verify = check_sat(verify_ctx)
        
        if res_verify.status == :unsat
            # No counterexample exists -> Candidate is valid!
            return (:success, candidate_C)
        elseif res_verify.status == :sat
            # Counterexample found
            # Extract values for input vars
            ce_vals = [res_verify.model[n] for (n, t) in input_vars]
            push!(inputs, ce_vals)
            pop!(verify_ctx)
        else
            return (:error, "Verification failed or timed out")
        end
    end

    return (:timeout, "Max iterations reached")
end

end # module
