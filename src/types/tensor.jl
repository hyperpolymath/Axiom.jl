# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Tensor Type System
#
# Provides compile-time shape checking through parametric types.
# The shape is encoded in the type parameter, enabling the Julia
# compiler to catch shape mismatches before runtime.

"""
    AbstractTensor{T, N}

Base type for all tensor types in Axiom.jl.
- `T`: Element type (Float32, Float64, etc.)
- `N`: Number of dimensions
"""
abstract type AbstractTensor{T, N} end

"""
    Tensor{T, N, Shape} <: AbstractTensor{T, N}

A tensor with compile-time verified shape.

# Type Parameters
- `T`: Element type
- `N`: Number of dimensions
- `Shape`: Tuple of dimension sizes (compile-time constant)

# Examples
```julia
# 2D tensor (matrix) of Float32, shape (28, 28)
x :: Tensor{Float32, 2, (28, 28)}

# 4D tensor (batch of images), shape (batch, height, width, channels)
images :: Tensor{Float32, 4, (32, 224, 224, 3)}

# Dynamic batch dimension
batch :: Tensor{Float32, 2, (:, 784)}  # : means dynamic
```
"""
struct Tensor{T, N, Shape} <: AbstractTensor{T, N}
    data::Array{T, N}

    function Tensor{T, N, Shape}(data::Array{T, N}) where {T, N, Shape}
        # Verify shape at construction time
        expected = collect(Shape)
        actual = size(data)

        for (i, (exp, act)) in enumerate(zip(expected, actual))
            if exp !== :dynamic && exp != act
                throw(DimensionMismatch(
                    "Tensor shape mismatch at dimension $i: expected $exp, got $act"
                ))
            end
        end

        new{T, N, Shape}(data)
    end
end

# Convenience constructors
Tensor(data::Array{T, N}) where {T, N} = Tensor{T, N, Tuple(size(data))}(data)

function Tensor{T, N, Shape}() where {T, N, Shape}
    dims = [s === :dynamic ? 1 : s for s in Shape]
    Tensor{T, N, Shape}(zeros(T, dims...))
end

"""
    DynamicTensor{T, N}

A tensor with runtime-determined shape. Use when shape is not known at compile time.
"""
struct DynamicTensor{T, N} <: AbstractTensor{T, N}
    data::Array{T, N}
end

DynamicTensor(data::Array{T, N}) where {T, N} = DynamicTensor{T, N}(data)

# Shape query functions
Base.size(t::Tensor{T, N, Shape}) where {T, N, Shape} = Shape
Base.size(t::DynamicTensor) = size(t.data)
Base.length(t::AbstractTensor) = prod(size(t))
Base.ndims(::Tensor{T, N}) where {T, N} = N
Base.ndims(::DynamicTensor{T, N}) where {T, N} = N
Base.eltype(::AbstractTensor{T}) where T = T

# Data access
Base.getindex(t::Tensor, args...) = getindex(t.data, args...)
Base.setindex!(t::Tensor, v, args...) = setindex!(t.data, v, args...)
Base.getindex(t::DynamicTensor, args...) = getindex(t.data, args...)
Base.setindex!(t::DynamicTensor, v, args...) = setindex!(t.data, v, args...)

# Conversion
Base.Array(t::Tensor) = t.data
Base.Array(t::DynamicTensor) = t.data

"""
    to_dynamic(t::Tensor) -> DynamicTensor

Convert a statically-shaped tensor to dynamic shape.
"""
to_dynamic(t::Tensor{T, N}) where {T, N} = DynamicTensor(t.data)

"""
    to_static(t::DynamicTensor, Shape) -> Tensor

Convert a dynamically-shaped tensor to static shape.
Throws if shapes don't match.
"""
function to_static(t::DynamicTensor{T, N}, ::Type{Tensor{T, N, Shape}}) where {T, N, Shape}
    Tensor{T, N, Shape}(t.data)
end

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", t::Tensor{T, N, Shape}) where {T, N, Shape}
    print(io, "Tensor{$T, $N, $Shape}")
    if length(t) <= 10
        print(io, ":\n")
        show(io, MIME"text/plain"(), t.data)
    else
        print(io, " with $(length(t)) elements")
    end
end

function Base.show(io::IO, t::Tensor{T, N, Shape}) where {T, N, Shape}
    print(io, "Tensor{$T, $N, $Shape}(...)")
end

# Tensor creation utilities
"""
    zeros_like(t::Tensor) -> Tensor

Create a tensor of zeros with the same shape and type.
"""
zeros_like(::Tensor{T, N, Shape}) where {T, N, Shape} = Tensor{T, N, Shape}(zeros(T, Shape...))

"""
    ones_like(t::Tensor) -> Tensor

Create a tensor of ones with the same shape and type.
"""
ones_like(::Tensor{T, N, Shape}) where {T, N, Shape} = Tensor{T, N, Shape}(ones(T, Shape...))

"""
    randn_like(t::Tensor) -> Tensor

Create a tensor of random normal values with the same shape and type.
"""
randn_like(::Tensor{T, N, Shape}) where {T, N, Shape} = Tensor{T, N, Shape}(randn(T, Shape...))

# Named tensor creation
"""
    axiom_zeros(T, dims...) -> Tensor

Create a zero tensor with specified type and dimensions.
"""
axiom_zeros(::Type{T}, dims::Int...) where T = Tensor(zeros(T, dims...))
axiom_zeros(dims::Int...) = axiom_zeros(Float32, dims...)

"""
    axiom_ones(T, dims...) -> Tensor

Create a ones tensor with specified type and dimensions.
"""
axiom_ones(::Type{T}, dims::Int...) where T = Tensor(ones(T, dims...))
axiom_ones(dims::Int...) = axiom_ones(Float32, dims...)

"""
    axiom_randn(T, dims...) -> Tensor

Create a random normal tensor with specified type and dimensions.
"""
axiom_randn(::Type{T}, dims::Int...) where T = Tensor(randn(T, dims...))
axiom_randn(dims::Int...) = axiom_randn(Float32, dims...)
