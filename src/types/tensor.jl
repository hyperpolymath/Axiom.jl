# SPDX-License-Identifier: PMPL-1.0-or-later
#
# Axiom.jl Tensor Type System: Foundations for Provably Correct Machine Learning
# ===========================================================================
#
# This file defines the core tensor types for Axiom.jl, designed to bring
# robust type-system-level guarantees to machine learning development in Julia.
# A central feature is **compile-time shape checking**, which leverages Julia's
# powerful parametric type system to embed tensor dimensions directly into the
# type signature.
#
# Rationale and Benefits of Compile-Time Shape Checking:
# -----------------------------------------------------
# Traditional ML frameworks often encounter runtime errors due to mismatched
# tensor shapes (e.g., trying to multiply matrices of incompatible dimensions).
# These errors can be hard to debug, especially in complex models. By encoding
# shape information in the type system, Axiom.jl enables the Julia compiler
# to:
# 1.  **Catch Dimension Mismatches at Compile-Time**: Errors are flagged before
#     the code even runs, saving significant development and debugging time.
# 2.  **Improve Code Reliability**: Ensures that model architectures adhere to
#     their specifications, leading to more robust and trustworthy ML systems.
# 3.  **Enhance Readability**: Type signatures convey rich information about
#     the expected data flow and tensor transformations.
#
# Distinction between `Tensor` and `DynamicTensor`:
# -----------------------------------------------
# -   `Tensor{T, N, Shape}`: This is the primary tensor type for static shape
#     checking. The `Shape` parameter is a `Tuple` of integers representing
#     fixed dimension sizes. It can also include the special symbol `:dynamic`
#     for dimensions whose size is only known at runtime (e.g., batch size).
#     This provides a flexible balance between static guarantees and dynamic
#     flexibility.
# -   `DynamicTensor{T, N}`: Used when the entire shape (or multiple dimensions)
#     cannot be determined at compile-time or when shape validation is deferred
#     to runtime. It sacrifices compile-time safety for greater flexibility.
#     Conversions between `Tensor` and `DynamicTensor` are provided to manage
#     these trade-offs.
#
# By providing these distinct tensor types and compile-time verification,
# Axiom.jl aims to reduce common ML programming errors and enable more formally
# verifiable neural network architectures.
#
#

"""
    AbstractTensor{T, N}

An abstract base type for all tensor implementations within Axiom.jl.
This type serves as the root of the tensor type hierarchy, providing a common
interface and enabling polymorphism across different concrete tensor types
(e.g., `Tensor` for static shapes, `DynamicTensor` for dynamic shapes).

Type Parameters:
- `T`: The element type of the tensor (e.g., `Float32`, `Float64`, `Int32`).
       This specifies the data type of the individual values stored within the tensor.
- `N`: The number of dimensions (rank) of the tensor. This is an integer
       representing how many indices are required to access an element (e.g.,
       `2` for a matrix, `3` for an image, `4` for a batch of images).
"""
abstract type AbstractTensor{T, N} end

"""
    Tensor{T, N, Shape} <: AbstractTensor{T, N}

A core tensor type in Axiom.jl designed for **compile-time shape verification**.
The `Shape` of the tensor is encoded directly into its type parameters, allowing
Julia's type system to catch dimension mismatches and other structural errors
before runtime. This significantly enhances the reliability and formal
verifiability of machine learning models.

Type Parameters:
- `T`: The element type of the tensor (e.g., `Float32`, `Float64`).
- `N`: The number of dimensions (rank) of the tensor.
- `Shape`: A `Tuple` of `Int`s or `Symbol`s representing the sizes of each
           dimension. This `Tuple` is a compile-time constant.
    - If a dimension size is an `Int` (e.g., `28`), it means that dimension
      has a fixed size.
    - If a dimension size is the `Symbol` `:dynamic` (e.g., `:(:dynamic)`),
      it indicates that this particular dimension's size can vary at runtime.
      This is especially useful for batch dimensions, where the batch size
      might not be fixed at compile time.

Fields:
- `data::Array{T, N}`: The underlying Julia `Array` that stores the tensor's data.

Benefits:
- **Early Error Detection**: Prevents common ML bugs related to incorrect tensor
  shapes.
- **Improved Code Clarity**: Type signatures provide clear expectations about
  data dimensions.
- **Facilitates Formal Verification**: Enables proving properties about data flow.

# Examples
```julia
# 2D tensor (matrix) of Float32, shape (28, 28)
x_static :: Tensor{Float32, 2, Tuple{28, 28}}

# 4D tensor (batch of images), shape (batch, height, width, channels)
images_static :: Tensor{Float32, 4, Tuple{32, 224, 224, 3}}

# Tensor with a dynamic batch dimension (size known only at runtime)
batch_dynamic :: Tensor{Float32, 2, Tuple{typeof(:dynamic), 784}} # Note `typeof(:dynamic)` for type parameter
```
"""
struct Tensor{T, N, Shape} <: AbstractTensor{T, N}
    data::Array{T, N}

    """
        Tensor{T, N, Shape}(data::Array{T, N}) where {T, N, Shape}

    Primary constructor for `Tensor`. It performs a rigorous shape verification
    at construction time, ensuring that the dimensions of the provided `data`
    array match the `Shape` type parameter.

    Arguments:
    - `data`: A Julia `Array{T, N}` whose dimensions will be checked against `Shape`.

    Throws:
    - `DimensionMismatch`: If any fixed dimension in `Shape` does not match
                           the corresponding dimension in `size(data)`. Dynamic
                           dimensions (`:dynamic`) are not checked.
    """
    function Tensor{T, N, Shape}(data::Array{T, N}) where {T, N, Shape}
        expected = collect(Shape) # Convert tuple of types to vector of values for iteration
        actual = size(data)

        # Ensure the rank (N) of the data matches the expected number of dimensions in Shape
        if length(expected) != length(actual)
            throw(DimensionMismatch(
                "Tensor rank mismatch: expected $(length(expected)) dimensions, got $(length(actual))"
            ))
        end

        for (i, (exp_dim, act_dim)) in enumerate(zip(expected, actual))
            # If the expected dimension is not dynamic and does not match the actual dimension
            if exp_dim !== :dynamic && exp_dim != act_dim
                throw(DimensionMismatch(
                    "Tensor shape mismatch at dimension $i: expected $exp_dim, got $act_dim"
                ))
            end
        end

        new{T, N, Shape}(data)
    end
end

"""
    Tensor(data::Array{T, N}) where {T, N}

Convenience constructor for `Tensor` that infers the `Shape` type parameter
directly from the dimensions of the provided `data` array at runtime.
This creates a fully static `Tensor` where all dimensions are fixed.

Arguments:
- `data`: A Julia `Array{T, N}`.

Returns:
- A `Tensor{T, N, Tuple{Dim1, Dim2, ...}}` where `Dim`s are the sizes of `data`.
"""
Tensor(data::Array{T, N}) where {T, N} = Tensor{T, N, Tuple(size(data))}(data)

"""
    Tensor{T, N, Shape}() where {T, N, Shape}

Convenience constructor for creating an uninitialized `Tensor` (filled with zeros)
with a specified static `Shape` and element type `T`.

If `Shape` includes `:dynamic` dimensions, those dimensions are initialized
with a size of `1` (or any arbitrary non-zero size) to allow the `zeros` function
to create the array. Runtime shape validation should then be performed when
actual data is assigned.

Returns:
- A `Tensor{T, N, Shape}` instance with data initialized to zeros.
"""
function Tensor{T, N, Shape}() where {T, N, Shape}
    # For dynamic dimensions, use an arbitrary size (e.g., 1) for initialization
    # The actual size will be determined when real data is loaded.
    dims = [s === :dynamic ? 1 : s for s in Shape]
    # Check if all dims are integers, otherwise throw (e.g. if Shape contains typeof(:dynamic))
    if !all(isa.(dims, Integer))
        error("Cannot create uninitialized Tensor with dynamic dimensions without providing actual sizes. Consider using Tensor(data) or providing full static Shape.")
    end
    Tensor{T, N, Shape}(zeros(T, dims...))
end

"""
    DynamicTensor{T, N} <: AbstractTensor{T, N}

A tensor type designed for scenarios where the full shape of the tensor is
determined exclusively at runtime and cannot be known or partially specified
at compile-time. `DynamicTensor` sacrifices the compile-time shape verification
offered by `Tensor` for maximum flexibility.

Use `DynamicTensor` when:
-   Dealing with highly variable input sizes where no compile-time guarantees
    are feasible (e.g., variable length sequences).
-   Interfacing with external libraries or data sources that do not provide
    shape information upfront.
-   Prototyping or during early development stages where strict shape
    constraints are temporarily undesirable.

Trade-offs:
-   **No Compile-Time Shape Checking**: Shape mismatches will result in runtime
    errors, similar to standard Julia `Array` operations.
-   **Flexibility**: Allows for arbitrary shape changes during execution without
    type instability issues related to `Shape` parameters.

Fields:
- `data::Array{T, N}`: The underlying Julia `Array` that stores the tensor's data.

# Examples
```julia
# A 2D tensor where both dimensions are dynamic
x_dynamic :: DynamicTensor{Float32, 2}

# Instantiate with an array
data = rand(Float32, 10, 20)
dyn_tensor = DynamicTensor(data)
```
"""
struct DynamicTensor{T, N} <: AbstractTensor{T, N}
    data::Array{T, N}
end



# Shape query functions
"""
    Base.size(t::Tensor{T, N, Shape}) where {T, N, Shape}

Returns the compile-time `Shape` tuple for a `Tensor`. Note that if `Shape`
contains `:dynamic` symbols, those symbols will be returned directly in the
tuple, representing an unknown dimension size.
"""
Base.size(t::Tensor{T, N, Shape}) where {T, N, Shape} = Shape

"""
    Base.size(t::DynamicTensor)

Returns the runtime `size` of the underlying `data` array for a `DynamicTensor`.
"""
Base.size(t::DynamicTensor) = size(t.data)
"""
    Base.length(t::AbstractTensor)

Returns the total number of elements in an `AbstractTensor`. This is computed
as the product of its dimension sizes. Note that if a `Tensor` has `:dynamic`
dimensions, this operation will behave as if the dynamic dimension has size 1,
or will rely on the runtime size of the underlying `data` for `DynamicTensor`.
"""
Base.length(t::AbstractTensor) = prod(size(t))

"""
    Base.ndims(::Tensor{T, N}) where {T, N}
    Base.ndims(::DynamicTensor{T, N}) where {T, N}

Returns the number of dimensions (rank) of a `Tensor` or `DynamicTensor`.
This value is directly extracted from the `N` type parameter.
"""
Base.ndims(::Tensor{T, N}) where {T, N} = N
Base.ndims(::DynamicTensor{T, N}) where {T, N} = N

"""
    Base.eltype(::AbstractTensor{T}) where T

Returns the element type of an `AbstractTensor`. This value is directly
extracted from the `T` type parameter.
"""
Base.eltype(::AbstractTensor{T}) where T = T

# Data access
"""
    Base.getindex(t::Tensor, args...)
    Base.getindex(t::DynamicTensor, args...)

Provides direct element access to the underlying `data` array of a `Tensor`
or `DynamicTensor`. This allows using standard Julia array indexing syntax
(e.g., `t[1, 2]`, `t[begin, :]`) on `AbstractTensor` objects.
"""
Base.getindex(t::Tensor, args...) = getindex(t.data, args...)
Base.getindex(t::DynamicTensor, args...) = getindex(t.data, args...)

"""
    Base.setindex!(t::Tensor, v, args...)
    Base.setindex!(t::DynamicTensor, v, args...)

Provides direct element assignment to the underlying `data` array of a `Tensor`
or `DynamicTensor`. This allows using standard Julia array assignment syntax
(e.g., `t[1, 2] = val`) on `AbstractTensor` objects.
"""
Base.setindex!(t::Tensor, v, args...) = setindex!(t.data, v, args...)
Base.setindex!(t::DynamicTensor, v, args...) = setindex!(t.data, v, args...)

# Conversion
"""
    Base.Array(t::Tensor)
    Base.Array(t::DynamicTensor)

Converts a `Tensor` or `DynamicTensor` back to its underlying native Julia
`Array`. This is useful when interfacing with functions that expect standard
Julia arrays.
"""
Base.Array(t::Tensor) = t.data
Base.Array(t::DynamicTensor) = t.data

"""
    to_dynamic(t::Tensor{T, N}) -> DynamicTensor{T, N}

Converts a statically-shaped `Tensor` to a `DynamicTensor`. This operation
discards the compile-time shape information, allowing the resulting tensor
to be used in contexts where shape variability is required or expected.

Arguments:
- `t`: The `Tensor` instance to convert.

Returns:
- A `DynamicTensor{T, N}` instance containing the same underlying data as `t`.

# Example
```julia
static_tensor = Tensor(rand(Float32, 2, 3))
dynamic_tensor = to_dynamic(static_tensor)
# dynamic_tensor now has runtime-determined shape, no compile-time shape constraint
```
"""
to_dynamic(t::Tensor{T, N}) where {T, N} = DynamicTensor(t.data)

"""
    to_static(t::DynamicTensor{T, N}, ::Type{Tensor{T, N, Shape}}) -> Tensor{T, N, Shape}

Converts a dynamically-shaped `DynamicTensor` to a statically-shaped `Tensor`.
This operation re-introduces compile-time shape verification by checking the
`DynamicTensor`'s current runtime shape against the `Shape` type parameter
of the target `Tensor` type.

Arguments:
- `t`: The `DynamicTensor` instance to convert.
- `::Type{Tensor{T, N, Shape}}`: A type literal specifying the desired static shape.
                                   The `T`, `N`, and `Shape` parameters must match
                                   the actual data in `t` (or `Shape` can contain `:dynamic`
                                   for matching dimensions).

Returns:
- A `Tensor{T, N, Shape}` instance containing the same underlying data as `t`,
  but now with compile-time shape guarantees.

Throws:
- `DimensionMismatch`: If the runtime shape of `t` does not conform to the
                       fixed dimensions specified in the `Shape` type parameter.

# Example
```julia
dynamic_tensor = DynamicTensor(rand(Float32, 2, 3))
# Convert to a static tensor with known shape (2, 3)
static_tensor = to_static(dynamic_tensor, Tensor{Float32, 2, Tuple{2, 3}})

# Note: Attempting to convert to a mismatched shape will throw DimensionMismatch
```
"""
function to_static(t::DynamicTensor{T, N}, ::Type{Tensor{T, N, Shape}}) where {T, N, Shape}
    Tensor{T, N, Shape}(t.data) # The Tensor constructor will perform the shape check
end

# Pretty printing
"""
    Base.show(io::IO, ::MIME"text/plain", t::Tensor{T, N, Shape})

Provides a detailed, multi-line string representation of a `Tensor` for rich
display contexts (e.g., in the Julia REPL or notebooks).
- For tensors with 10 or fewer elements, the full content of the underlying
  `data` array is shown.
- For larger tensors, a compact summary indicating the type, dimensions, and
  number of elements is displayed.
"""
function Base.show(io::IO, ::MIME"text/plain", t::Tensor{T, N, Shape}) where {T, N, Shape}
    print(io, "Tensor{$T, $N, $Shape}")
    if length(t) <= 10
        print(io, ":\n")
        show(io, MIME"text/plain"(), t.data)
    else
        print(io, " with $(length(t)) elements")
    end
end

"""
    Base.show(io::IO, t::Tensor{T, N, Shape})

Provides a compact, single-line string representation of a `Tensor` for
less verbose display contexts.
"""
function Base.show(io::IO, t::Tensor{T, N, Shape}) where {T, N, Shape}
    print(io, "Tensor{$T, $N, $Shape}(...)")
end

# Tensor creation utilities
# Tensor creation utilities
"""
    zeros_like(t::Tensor) -> Tensor

Creates a new `Tensor` filled with zeros, inheriting the element type (`T`),
number of dimensions (`N`), and compile-time shape (`Shape`) from an existing `Tensor`.
This is a convenient way to create a companion tensor with identical structural
properties for operations like gradients or accumulation.

Arguments:
- `t`: The reference `Tensor` whose properties (`T`, `N`, `Shape`) will be
       used for the new zero tensor.

Returns:
- A new `Tensor{T, N, Shape}` instance with all elements initialized to `zero(T)`.

# Note on `:dynamic` dimensions:
If the `Shape` of the reference `Tensor` contains `:dynamic` symbols, this
function will effectively treat those dimensions as having a size of `1` (or
the actual runtime size if the underlying `data` has a non-`1` dimension).
This is because `zeros(T, Shape...)` requires concrete integer dimensions.
Users should be aware that such newly created tensors will have specific,
potentially small, sizes for originally dynamic dimensions.

# Example
```julia
static_tensor = Tensor{Float32, 2, Tuple{2, 2}}(rand(Float32, 2, 2))
zeros_tensor = zeros_like(static_tensor) # Tensor{Float32, 2, Tuple{2, 2}} with zeros

dynamic_batch_tensor = Tensor{Float32, 2, Tuple{typeof(:dynamic), 784}}(rand(Float32, 32, 784))
zeros_dynamic_batch = zeros_like(dynamic_batch_tensor) # Tensor{Float32, 2, Tuple{typeof(:dynamic), 784}} with zeros, where dynamic dim is size 32
```
"""
zeros_like(t::Tensor{T, N, Shape}) where {T, N, Shape} = Tensor{T, N, Shape}(zeros(T, size(t.data)...)) # Use size(t.data) for dynamic dimensions

"""
    ones_like(t::Tensor) -> Tensor

Creates a new `Tensor` filled with ones, inheriting the element type (`T`),
number of dimensions (`N`), and compile-time shape (`Shape`) from an existing `Tensor`.

Arguments:
- `t`: The reference `Tensor` whose properties (`T`, `N`, `Shape`) will be
       used for the new ones tensor.

Returns:
- A new `Tensor{T, N, Shape}` instance with all elements initialized to `one(T)`.

# Note on `:dynamic` dimensions:
Similar to `zeros_like`, if the `Shape` contains `:dynamic` symbols, this
function will treat those dimensions as having the actual runtime size of the
`t.data` array.

# Example
```julia
static_tensor = Tensor{Float32, 2, Tuple{2, 2}}(rand(Float32, 2, 2))
ones_tensor = ones_like(static_tensor) # Tensor{Float32, 2, Tuple{2, 2}} with ones
```
"""
ones_like(t::Tensor{T, N, Shape}) where {T, N, Shape} = Tensor{T, N, Shape}(ones(T, size(t.data)...)) # Use size(t.data) for dynamic dimensions

"""
    randn_like(t::Tensor) -> Tensor

Creates a new `Tensor` filled with random values drawn from a standard normal
distribution, inheriting the element type (`T`), number of dimensions (`N`),
and compile-time shape (`Shape`) from an existing `Tensor`.

Arguments:
- `t`: The reference `Tensor` whose properties (`T`, `N`, `Shape`) will be
       used for the new random tensor.

Returns:
- A new `Tensor{T, N, Shape}` instance with elements sampled from a standard
  normal distribution.

# Note on `:dynamic` dimensions:
Similar to `zeros_like`, if the `Shape` contains `:dynamic` symbols, this
function will treat those dimensions as having the actual runtime size of the
`t.data` array.

# Example
```julia
static_tensor = Tensor{Float32, 2, Tuple{2, 2}}(rand(Float32, 2, 2))
randn_tensor = randn_like(static_tensor) # Tensor{Float32, 2, Tuple{2, 2}} with random normal values
```
"""
randn_like(t::Tensor{T, N, Shape}) where {T, N, Shape} = Tensor{T, N, Shape}(randn(T, size(t.data)...)) # Use size(t.data) for dynamic dimensions

# Named tensor creation
"""
    axiom_zeros(T::Type, dims::Int...) -> Tensor
    axiom_zeros(dims::Int...) -> Tensor{Float32}

Creates a new `Tensor` instance with all elements initialized to zero.
This is a convenient factory function for directly constructing statically-shaped
tensors without needing to manually create an underlying `Array` first.

Arguments:
- `T::Type`: The element type for the tensor (e.g., `Float32`, `Float64`).
- `dims::Int...`: A variadic argument representing the fixed integer sizes
                  of each dimension for the new tensor.

Returns:
- A `Tensor{T, N, Tuple{dims...}}` where `N` is `length(dims)`.
  If `T` is not specified, it defaults to `Float32`.

# Example
```julia
# Create a 2x3 tensor of Float32 zeros
t1 = axiom_zeros(Float32, 2, 3) # Tensor{Float32, 2, Tuple{2, 3}}

# Create a 4-dimensional tensor of default Float32 zeros (e.g., batch of images)
t2 = axiom_zeros(32, 64, 64, 3) # Tensor{Float32, 4, Tuple{32, 64, 64, 3}}
```
"""
axiom_zeros(::Type{T}, dims::Int...) where T = Tensor(zeros(T, dims...))
axiom_zeros(dims::Int...) = axiom_zeros(Float32, dims...)

"""
    axiom_ones(T::Type, dims::Int...) -> Tensor
    axiom_ones(dims::Int...) -> Tensor{Float32}

Creates a new `Tensor` instance with all elements initialized to one.
This is a convenient factory function for directly constructing statically-shaped
tensors.

Arguments:
- `T::Type`: The element type for the tensor.
- `dims::Int...`: A variadic argument representing the fixed integer sizes
                  of each dimension.

Returns:
- A `Tensor{T, N, Tuple{dims...}}` where `N` is `length(dims)`.
  If `T` is not specified, it defaults to `Float32`.

# Example
```julia
# Create a 2x3 tensor of Float64 ones
t1 = axiom_ones(Float64, 2, 3) # Tensor{Float64, 2, Tuple{2, 3}}

# Create a 10x10 tensor of default Float32 ones
t2 = axiom_ones(10, 10) # Tensor{Float32, 2, Tuple{10, 10}}
```
"""
axiom_ones(::Type{T}, dims::Int...) where T = Tensor(ones(T, dims...))
axiom_ones(dims::Int...) = axiom_ones(Float32, dims...)

"""
    axiom_randn(T::Type, dims::Int...) -> Tensor
    axiom_randn(dims::Int...) -> Tensor{Float32}

Creates a new `Tensor` instance with elements initialized to random values
drawn from a standard normal distribution (mean 0, variance 1).
This is a convenient factory function for directly constructing statically-shaped
tensors.

Arguments:
- `T::Type`: The element type for the tensor.
- `dims::Int...`: A variadic argument representing the fixed integer sizes
                  of each dimension.

Returns:
- A `Tensor{T, N, Tuple{dims...}}` where `N` is `length(dims)`.
  If `T` is not specified, it defaults to `Float32`.

# Example
```julia
# Create a 2x2 tensor of Float32 random normal values
t1 = axiom_randn(Float32, 2, 2) # Tensor{Float32, 2, Tuple{2, 2}}

# Create a 5-dimensional tensor of default Float32 random values
t2 = axiom_randn(2, 2, 2, 2, 2) # Tensor{Float32, 5, Tuple{2, 2, 2, 2, 2}}
```
"""
axiom_randn(::Type{T}, dims::Int...) where T = Tensor(randn(T, dims...))
axiom_randn(dims::Int...) = axiom_randn(Float32, dims...)
