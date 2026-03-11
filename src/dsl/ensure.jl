# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl @ensure Macro
#
# Runtime assertion system with compile-time optimization.
# Ensures are checked at runtime but can be optimized away in release builds.

"""
    @ensure condition [message]

Assert that a condition holds. Unlike @assert, @ensure:
- Is tied to the Axiom verification system
- Can be formally verified in some cases
- Provides better error messages for ML contexts

# Examples
```julia
@ensure sum(probabilities) ≈ 1.0 "Probabilities must sum to 1"
@ensure all(x -> x >= 0, outputs) "Outputs must be non-negative"
@ensure !any(isnan, gradients) "Gradients contain NaN"
```
"""
macro ensure(condition)
    _ensure_impl(condition, nothing)
end

macro ensure(condition, message)
    _ensure_impl(condition, message)
end

function _ensure_impl(condition, message)
    cond_str = string(condition)
    msg = message === nothing ? "Ensure failed: $cond_str" : message

    quote
        if !$(esc(condition))
            throw(EnsureViolation(
                $(QuoteNode(condition)),
                $msg,
                nothing
            ))
        end
        nothing
    end
end

"""
Exception for @ensure violations.
"""
struct EnsureViolation <: Exception
    condition::Any
    message::String
    value::Any  # The actual value that violated the condition
end

function Base.showerror(io::IO, e::EnsureViolation)
    println(io, "Ensure Violation: $(e.message)")
    println(io, "  Condition: $(e.condition)")
    if e.value !== nothing
        println(io, "  Actual value: $(e.value)")
    end
    println(io)
    println(io, "This indicates a bug in the model or input data.")
end

# Common ML-specific ensure patterns
"""
    ensure_valid_probabilities(x)

Ensure tensor represents valid probability distribution.
"""
function ensure_valid_probabilities(x)
    @ensure all(x .>= 0) "Probabilities must be non-negative"
    @ensure all(x .<= 1) "Probabilities must be <= 1"

    # Check sum ≈ 1 along last dimension
    sums = sum(x, dims=ndims(x))
    @ensure all(isapprox.(sums, 1.0, atol=1e-5)) "Probabilities must sum to 1"

    true
end

"""
    ensure_no_nan(x, name="tensor")

Ensure tensor contains no NaN values.
"""
function ensure_no_nan(x, name="tensor")
    if any(isnan, x)
        throw(EnsureViolation(
            :(any(isnan, $name)),
            "Found NaN values in $name",
            count(isnan, x)
        ))
    end
    true
end

"""
    ensure_no_inf(x, name="tensor")

Ensure tensor contains no Inf values.
"""
function ensure_no_inf(x, name="tensor")
    if any(isinf, x)
        throw(EnsureViolation(
            :(any(isinf, $name)),
            "Found Inf values in $name",
            count(isinf, x)
        ))
    end
    true
end

"""
    ensure_finite(x, name="tensor")

Ensure tensor contains only finite values (no NaN or Inf).
"""
function ensure_finite(x, name="tensor")
    ensure_no_nan(x, name)
    ensure_no_inf(x, name)
    true
end

"""
    ensure_bounded(x, low, high, name="tensor")

Ensure tensor values are within bounds.
"""
function ensure_bounded(x, low, high, name="tensor")
    if any(x .< low) || any(x .> high)
        throw(EnsureViolation(
            :($low <= $name <= $high),
            "Values in $name must be in [$low, $high]",
            (minimum(x), maximum(x))
        ))
    end
    true
end

"""
    ensure_shape(x, expected_shape)

Ensure tensor has expected shape.
"""
function ensure_shape(x, expected_shape)
    actual = size(x)
    for (i, (exp, act)) in enumerate(zip(expected_shape, actual))
        if exp !== :any && exp != act
            throw(EnsureViolation(
                :(size(x) == $expected_shape),
                "Shape mismatch at dimension $i: expected $exp, got $act",
                actual
            ))
        end
    end
    true
end

# Gradient-specific ensures
"""
    ensure_gradient_finite(grads, name="gradients")

Ensure all gradients are finite (useful for debugging training).
"""
function ensure_gradient_finite(grads, name="gradients")
    for (param_name, grad) in pairs(grads)
        ensure_finite(grad, "$name.$param_name")
    end
    true
end

"""
    ensure_gradient_not_exploding(grads, threshold=1e6)

Ensure gradients haven't exploded beyond threshold.
"""
function ensure_gradient_not_exploding(grads, threshold=1e6)
    for (param_name, grad) in pairs(grads)
        max_val = maximum(abs, grad)
        if max_val > threshold
            throw(EnsureViolation(
                :(max_gradient < $threshold),
                "Gradient explosion detected in $param_name: max value = $max_val",
                max_val
            ))
        end
    end
    true
end
