# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Automatic Differentiation
#
# Production-grade reverse-mode AD using Zygote.jl backend

using Zygote

"""
    gradient(f, x...)

Compute gradient of scalar function f with respect to inputs x.
Uses Zygote.jl backend for production-grade automatic differentiation.

# Examples
```julia
# Scalar gradient
g = Axiom.gradient(x -> x^2 + 3x, 2.0)
# g = (7.0,)

# Multiple inputs
g = Axiom.gradient((x, y) -> x^2 + x*y, 2.0, 3.0)
# g = (7.0, 2.0)
```
"""
function gradient(f, x...)
    return Zygote.gradient(f, x...)
end

"""
    jacobian(f, x)

Compute Jacobian matrix of vector-valued function f at point x.
Uses Zygote.jl backend.

# Examples
```julia
J = Axiom.jacobian(x -> [x[1]^2, x[1]*x[2]], [3.0, 4.0])
# Returns 2×2 Jacobian matrix
```
"""
function jacobian(f, x)
    return Zygote.jacobian(f, x)[1]
end

"""
    pullback(f, x...)

Compute pullback of function f at inputs x.
Returns (y, back) where y is the forward pass result and back is the pullback function.

# Examples
```julia
y, back = Axiom.pullback(x -> x^2, 3.0)
# y = 9.0
# back(1.0) returns gradient
```
"""
function pullback(f, x...)
    return Zygote.pullback(f, x...)
end

"""
    @no_grad expr

Execute expression without tracking gradients.
Useful for inference or when gradients are not needed.

# Examples
```julia
@no_grad begin
    y = model(x)
    loss = criterion(y, target)
end
```
"""
macro no_grad(expr)
    quote
        Zygote.@ignore $(esc(expr))
    end
end

# Gradient utilities for parameter updates
"""
    zero_grad!(params)

Reset gradients for a collection of parameters to nothing.
"""
function zero_grad!(params)
    for p in params
        if applicable(setfield!, p, :grad, nothing)
            setfield!(p, :grad, nothing)
        end
    end
end

"""
    clip_grad_norm!(params, max_norm)

Clip gradients by global norm to prevent exploding gradients.
"""
function clip_grad_norm!(params, max_norm)
    total_norm = 0.0
    for p in params
        if hasfield(typeof(p), :grad) && !isnothing(getfield(p, :grad))
            g = getfield(p, :grad)
            total_norm += sum(abs2, g)
        end
    end
    total_norm = sqrt(total_norm)

    if total_norm > max_norm
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in params
            if hasfield(typeof(p), :grad) && !isnothing(getfield(p, :grad))
                g = getfield(p, :grad)
                setfield!(p, :grad, g * clip_coef)
            end
        end
    end

    return total_norm
end

# Export main functions
export gradient, jacobian, pullback, @no_grad, zero_grad!, clip_grad_norm!
