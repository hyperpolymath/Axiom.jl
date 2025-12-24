# Axiom.jl Automatic Differentiation
#
# Reverse-mode AD for gradient computation.
# This is a minimal implementation - production would use Zygote.jl or Enzyme.jl

"""
    Gradient

Wrapper type for tracked tensors in autograd.
"""
mutable struct Gradient{T, S}
    value::S
    grad::Union{S, Nothing}
    backward_fn::Union{Function, Nothing}
    requires_grad::Bool
    parents::Vector{Gradient}
end

function Gradient(value; requires_grad::Bool=true)
    Gradient{eltype(value), typeof(value)}(
        value, nothing, nothing, requires_grad, Gradient[]
    )
end

# Access value
Base.getindex(g::Gradient) = g.value
value(g::Gradient) = g.value
grad(g::Gradient) = g.grad

# Math operations that track gradients
function Base.:+(a::Gradient, b::Gradient)
    result = Gradient(a.value + b.value)
    result.parents = [a, b]
    result.backward_fn = function(grad)
        if a.requires_grad
            a.grad = a.grad === nothing ? grad : a.grad + grad
        end
        if b.requires_grad
            b.grad = b.grad === nothing ? grad : b.grad + grad
        end
    end
    result
end

function Base.:*(a::Gradient, b::Gradient)
    result = Gradient(a.value * b.value)
    result.parents = [a, b]
    result.backward_fn = function(grad)
        if a.requires_grad
            g = grad * b.value'
            a.grad = a.grad === nothing ? g : a.grad + g
        end
        if b.requires_grad
            g = a.value' * grad
            b.grad = b.grad === nothing ? g : b.grad + g
        end
    end
    result
end

function Base.:-(a::Gradient, b::Gradient)
    result = Gradient(a.value - b.value)
    result.parents = [a, b]
    result.backward_fn = function(grad)
        if a.requires_grad
            a.grad = a.grad === nothing ? grad : a.grad + grad
        end
        if b.requires_grad
            g = -grad
            b.grad = b.grad === nothing ? g : b.grad + g
        end
    end
    result
end

# Broadcast operations
Base.broadcasted(::typeof(+), a::Gradient, b) = Gradient(a.value .+ b)
Base.broadcasted(::typeof(*), a::Gradient, b) = Gradient(a.value .* b)

"""
    backward!(loss::Gradient)

Compute gradients via backpropagation.
"""
function backward!(loss::Gradient)
    # Initialize gradient of loss to 1
    if loss.grad === nothing
        loss.grad = ones(eltype(loss.value), size(loss.value))
    end

    # Topological sort
    visited = Set{Gradient}()
    order = Gradient[]

    function visit(node)
        if node in visited
            return
        end
        push!(visited, node)
        for parent in node.parents
            visit(parent)
        end
        push!(order, node)
    end

    visit(loss)

    # Backward pass in reverse topological order
    for node in reverse(order)
        if node.backward_fn !== nothing
            node.backward_fn(node.grad)
        end
    end
end

"""
    zero_grad!(g::Gradient)

Reset gradient to nothing.
"""
function zero_grad!(g::Gradient)
    g.grad = nothing
end

# No-grad context
struct NoGradContext end

"""
    @no_grad expr

Execute expression without tracking gradients.
"""
macro no_grad(expr)
    quote
        # In production, this would disable gradient tracking
        $(esc(expr))
    end
end

# Gradient computation utilities
"""
    gradient(f, x...)

Compute gradient of f with respect to x.
"""
function gradient(f, x...)
    # Wrap inputs
    wrapped = [Gradient(xi) for xi in x]

    # Forward pass
    y = f(wrapped...)

    # Backward pass
    backward!(y)

    # Return gradients
    [w.grad for w in wrapped]
end

"""
    jacobian(f, x)

Compute Jacobian matrix.
"""
function jacobian(f, x)
    n = length(x)
    y = f(x)
    m = length(y)

    J = zeros(eltype(x), m, n)

    for i in 1:m
        # Compute row i of Jacobian
        g = Gradient(x)
        out = f(g)
        out.grad = zeros(eltype(x), m)
        out.grad[i] = 1.0
        backward!(out)
        J[i, :] = vec(g.grad)
    end

    J
end
