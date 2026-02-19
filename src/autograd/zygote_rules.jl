# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Zygote Rules
#
# Custom adjoints for Axiom tensor types to ensure efficient and correct AD

using Zygote
using ChainRulesCore

# 1. Iterate rule (needed for sum, etc. if not using direct sum adjoint)
# Although we will define a direct sum adjoint, providing iterate is good for generic code.
function Base.iterate(t::AbstractTensor, state...)
    return iterate(t.data, state...)
end

# 2. Tensor constructor adjoint
Zygote.@adjoint function Tensor{T, N, Shape}(data::Array{T, N}) where {T, N, Shape}
    return Tensor{T, N, Shape}(data), Δ -> (Δ.data,)
end

Zygote.@adjoint function Tensor(data::Array{T, N}) where {T, N}
    return Tensor(data), Δ -> (Δ.data,)
end

# 3. Basic math operations
Zygote.@adjoint function Base.:+(a::Tensor{T, N, S}, b::Tensor{T, N, S}) where {T, N, S}
    return a + b, Δ -> (Δ, Δ)
end

Zygote.@adjoint function Base.:-(a::Tensor{T, N, S}, b::Tensor{T, N, S}) where {T, N, S}
    return a - b, Δ -> (Δ, -Δ)
end

Zygote.@adjoint function Base.:*(a::Tensor{T, 2, S1}, b::Tensor{T, 2, S2}) where {T, S1, S2}
    return a * b, Δ -> (Δ * b', a' * Δ)
end

# 4. Reduction operations
Zygote.@adjoint function Base.sum(t::Tensor{T, N, S}) where {T, N, S}
    return sum(t.data), Δ -> (ones_like(t) * Δ,)
end

# 5. Getindex adjoint
Zygote.@adjoint function Base.getindex(t::Tensor{T, N, S}, i...) where {T, N, S}
    y = t.data[i...]
    function getindex_pullback(Δ)
        adj_data = zeros(T, size(t.data)...)
        adj_data[i...] .= Δ
        return (Tensor{T, N, S}(adj_data), map(_ -> nothing, i)...)
    end
    return y, getindex_pullback
end
