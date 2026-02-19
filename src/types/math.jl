# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Tensor Mathematics
#
# Element-wise operations and matrix algebra for Axiom tensor types

# 1. Element-wise addition
function Base.:+(a::Tensor{T, N, S}, b::Tensor{T, N, S}) where {T, N, S}
    return Tensor{T, N, S}(a.data + b.data)
end

# 2. Element-wise subtraction
function Base.:-(a::Tensor{T, N, S}, b::Tensor{T, N, S}) where {T, N, S}
    return Tensor{T, N, S}(a.data - b.data)
end

# 3. Scalar multiplication
function Base.:*(a::Tensor{T, N, S}, b::Number) where {T, N, S}
    return Tensor{T, N, S}(a.data .* b)
end

function Base.:*(a::Number, b::Tensor{T, N, S}) where {T, N, S}
    return Tensor{T, N, S}(a .* b.data)
end

# 4. Matrix multiplication (2D only)
function Base.:*(a::Tensor{T, 2, S1}, b::Tensor{T, 2, S2}) where {T, S1, S2}
    # In a real implementation, we would check Shape compatibility at compile time here.
    return Tensor(a.data * b.data)
end

# 5. Reductions
function Base.sum(t::Tensor)
    return sum(t.data)
end

# 6. Transpose
function Base.adjoint(t::Tensor{T, 2, S}) where {T, S}
    return Tensor(collect(t.data'))
end

# 7. Broadcasters/Forwarding for activations
(t::Tensor)(x) = t.data(x) # Handle case where layer itself is called (though usually layers wrap tensors)
