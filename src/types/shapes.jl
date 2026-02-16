# SPDX-License-Identifier: PMPL-1.0-or-later
#
# Axiom.jl Shape System: Compile-Time Shape Inference and Verification
# ===================================================================
#
# This file defines the core `Shape` representation and associated functions
# that enable Axiom.jl's powerful **compile-time shape checking and inference**.
# The `Shape` type is a cornerstone of the framework's "provably correct ML"
# philosophy, allowing developers to define and reason about tensor dimensions
# at the type level.
#
# Significance for Provably Correct ML:
# -----------------------------------
# By making tensor shapes part of the type system, Axiom.jl can:
# 1.  **Prevent Runtime Shape Errors**: Many common bugs in neural network
#     architectures stem from unexpected or incompatible tensor dimensions.
#     The `Shape` system ensures that operations are only applied to tensors
#     whose dimensions are mathematically consistent, detecting errors before
#     execution.
# 2.  **Facilitate Formal Verification**: Compile-time knowledge of shapes
#     is crucial for building formal proofs about the properties of ML models.
#     It guarantees structural integrity, simplifying the task of verifying
#     functional correctness and safety.
# 3.  **Enhance Code Reasoning**: Developers can reason about the flow of
#     data and the transformations applied by layers with greater precision,
#     improving both development efficiency and the clarity of model definitions.
#
# The functions within this module provide the logic for:
# -   Inferring the output shape of common neural network layers (e.g., `Dense`, `Conv2D`).
# -   Checking compatibility between shapes (e.g., for broadcasting).
# -   Defining and handling `ShapeMismatchError` for clear error reporting.
#
# This system works in tight conjunction with the `Tensor` type defined in
# `types/tensor.jl`, where `Shape` is used as a type parameter to enforce
# these compile-time guarantees.
#


"""
    Shape{dims}

Represents a tensor's shape at compile-time. This `struct` is used as a type
parameter (specifically, the `Shape` parameter) within the `Tensor` type
definition in `Axiom.jl/src/types/tensor.jl`. By embedding shape information
into the type system, Axiom.jl enables static analysis and compile-time
verification of tensor dimensions, preventing many common runtime errors.

Type Parameter:
- `dims`: A `Tuple` where each element specifies the size of a dimension.
    - An `Int` value (e.g., `28`) indicates a fixed-size dimension.
    - The `Symbol` `:dynamic` indicates a dimension whose size is
      determined at runtime (e.g., a batch dimension). When used as a type
      parameter, this would typically appear as `typeof(:dynamic)`.

# Examples (as type parameters to `Tensor`)
```julia
# A 3D shape (e.g., image: height, width, channels) with fixed dimensions
Tensor{Float32, 3, Shape{Tuple{28, 28, 1}}}

# A 3D shape where the first dimension is dynamic (e.g., dynamic batch size)
Tensor{Float32, 3, Shape{Tuple{typeof(:dynamic), 224, 224}}}

# A 2D shape where both dimensions are dynamic
Tensor{Float32, 2, Shape{Tuple{typeof(:dynamic), typeof(:dynamic)}}}
```
"""
struct Shape{dims} end

# Shape arithmetic for layer composition
"""
    DynamicDim

A constant `Symbol` (`:`) used to represent a dynamically sized dimension within
a `Shape` tuple. When `:dynamic` is present in a `Shape` type parameter (e.g.,
`Shape{Tuple{typeof(:dynamic), 28}}`), it indicates that the size of that
specific dimension is not fixed at compile-time and will be determined
at runtime. This is particularly useful for flexible batch dimensions.
"""
const DynamicDim = Symbol(":")


"""
    shape_after_dense(input_shape::Type{Shape{input}}, out_features::Int) -> Shape

Infers the output `Shape` of a dense (fully connected) layer given its `input_shape`
and the number of `out_features`. A dense layer typically transforms the last
dimension of its input.

Arguments:
- `input_shape::Type{Shape{input}}`: The compile-time `Shape` of the input tensor.
                                       Expected formats are:
                                       - `Shape{Tuple{num_features}}` for 1D input (e.g., a single feature vector).
                                       - `Shape{Tuple{batch_size, num_features}}` for 2D input (e.g., a batch of feature vectors).
- `out_features::Int`: The number of output features (neurons) in the dense layer.

Returns:
- A `Shape` type representing the inferred output dimensions.

Throws:
- `ErrorException`: If `input_shape` is neither 1D nor 2D.

# Details:
- If input is 1D `(num_features,)`, the output will be 1D `(out_features,)`.
- If input is 2D `(batch_size, num_features)`, the output will be 2D `(batch_size, out_features)`.
  The batch dimension (`batch_size`) is propagated as-is, including `:dynamic` if present.

# Examples
```julia
# 1D input: (784,) -> (128,)
output1 = shape_after_dense(Shape{Tuple{784}}, 128) # Shape{Tuple{128}}

# 2D input: (32, 784) -> (32, 128)
output2 = shape_after_dense(Shape{Tuple{32, 784}}, 128) # Shape{Tuple{32, 128}}

# 2D input with dynamic batch: (typeof(:dynamic), 784) -> (typeof(:dynamic), 128)
output3 = shape_after_dense(Shape{Tuple{typeof(:dynamic), 784}}, 128) # Shape{Tuple{typeof(:dynamic), 128}}

# Note: Invalid input rank (e.g., 3D input) will throw ErrorException
```
"""
function shape_after_dense(input_shape::Type{Shape{input}}, out_features::Int) where input
    if length(input) == 1
        return Shape{Tuple{out_features}}
    elseif length(input) == 2
        # (batch, features) -> (batch, out_features)
        batch = input[1]
        return Shape{Tuple{batch, out_features}}
    else
        error("Dense layer expects 1D or 2D input, got $(length(input))D")
    end
end

"""
    shape_after_conv2d(
        input_shape::Type{Shape{input}},
        out_channels::Int,
        kernel::Tuple{Int, Int},
        stride::Tuple{Int, Int} = (1, 1),
        padding::Tuple{Int, Int} = (0, 0)
    ) -> Shape

Infers the output `Shape` of a 2D convolutional layer given its `input_shape`
and convolutional parameters. Convolutional layers are fundamental for feature
extraction in image processing tasks.

Arguments:
- `input_shape::Type{Shape{input}}`: The compile-time `Shape` of the input tensor.
                                       Expected to be 4D: `Shape{Tuple{N, H, W, C}}`,
                                       where `N` is batch size, `H` is height,
                                       `W` is width, and `C` is input channels.
- `out_channels::Int`: The number of feature maps (output channels) produced by
                       the convolutional layer.
- `kernel::Tuple{Int, Int}`: A tuple `(kernel_height, kernel_width)` specifying
                             the dimensions of the convolutional filter.
- `stride::Tuple{Int, Int}`: A tuple `(stride_height, stride_width)` specifying
                             the step size of the filter across the input.
                             Defaults to `(1, 1)`.
- `padding::Tuple{Int, Int}`: A tuple `(padding_height, padding_width)` specifying
                              the amount of zero-padding applied to the input's
                              height and width dimensions. Defaults to `(0, 0)`.

Returns:
- A `Shape` type representing the inferred output dimensions: `(N_out, H_out, W_out, C_out)`.

Throws:
- `AssertionError`: If `input_shape` is not 4D.

# Details on Output Dimension Calculation:
The output height (`out_H`) and width (`out_W`) are calculated using the standard
convolution formula:
`out_H = floor((H + 2 * pH - kH) / sH) + 1`
`out_W = floor((W + 2 * pW - kW) / sW) + 1`
where `(kH, kW)` are kernel dimensions, `(sH, sW)` are stride dimensions, and
`(pH, pW)` are padding dimensions.

The batch dimension (`N`) is propagated as-is, including `:dynamic` if present.
The output channels (`C_out`) will be `out_channels`.

# Examples
```julia
# Input: (batch=1, H=28, W=28, C=1), Kernel=(3,3), Stride=(1,1), Padding=(0,0), Out_channels=32
output1 = shape_after_conv2d(Shape{Tuple{1, 28, 28, 1}}, 32, (3, 3))
# Expected output: Shape{Tuple{1, 26, 26, 32}}

# Input with dynamic batch: (typeof(:dynamic), H=32, W=32, C=3), Kernel=(5,5), Stride=(2,2), Padding=(0,0), Out_channels=64
output2 = shape_after_conv2d(Shape{Tuple{typeof(:dynamic), 32, 32, 3}}, 64, (5, 5), (2, 2))
# Expected output: Shape{Tuple{typeof(:dynamic), 14, 14, 64}}

# Input with padding
output3 = shape_after_conv2d(Shape{Tuple{1, 28, 28, 1}}, 32, (3, 3), (1,1), (1,1))
# Expected output: Shape{Tuple{1, 28, 28, 32}}
```
"""
function shape_after_conv2d(
    ::Type{Shape{input}},
    out_channels::Int,
    kernel::Tuple{Int, Int},
    stride::Tuple{Int, Int} = (1, 1),
    padding::Tuple{Int, Int} = (0, 0)
) where input
    @assert length(input) == 4 "Conv2D expects 4D input (N, H, W, C), got $(length(input))D"

    N, H, W, C = input
    kH, kW = kernel
    sH, sW = stride
    pH, pW = padding

    # Output dimensions using standard formula
    out_H = div(H + 2*pH - kH, sH) + 1
    out_W = div(W + 2*pW - kW, sW) + 1

    # Handle dynamic batch
    if N === :dynamic
        return Shape{Tuple{typeof(:dynamic), out_H, out_W, out_channels}}
    else
        return Shape{Tuple{N, out_H, out_W, out_channels}}
    end
end

"""
    shape_after_flatten(input_shape::Type{Shape{input}}) -> Shape

Infers the output `Shape` of a flatten layer. A flatten layer reshapes an
input tensor into a 2D tensor, typically `(batch_size, num_features)`,
where `num_features` is the product of all dimensions except the batch dimension.
This is commonly used to connect convolutional layers to dense layers.

Arguments:
- `input_shape::Type{Shape{input}}`: The compile-time `Shape` of the input tensor.
                                       This can be any `N`-dimensional shape.

Returns:
- A `Shape` type representing the inferred output dimensions: `(batch_size, num_features)`.

# Details:
- If `input_shape` is already 1D `(num_features,)`, it remains unchanged.
- If `input_shape` is `(batch_size, dim2, dim3, ...)`, the output becomes
  `(batch_size, dim2 * dim3 * ...)`.
- The batch dimension (`batch_size`) is propagated as-is, including `:dynamic` if present.
  If the input `Shape` is 1D, there is no batch dimension, and it is assumed to be `1` implicitly.

# Examples
```julia
# Input: (batch=1, H=28, W=28, C=1) -> (1, 784)
output1 = shape_after_flatten(Shape{Tuple{1, 28, 28, 1}}) # Shape{Tuple{1, 784}}

# Input with dynamic batch: (typeof(:dynamic), H=7, W=7, C=64) -> (typeof(:dynamic), 3136)
output2 = shape_after_flatten(Shape{Tuple{typeof(:dynamic), 7, 7, 64}}) # Shape{Tuple{typeof(:dynamic), 3136}}

# Input is already 1D: (100,) -> (100,)
output3 = shape_after_flatten(Shape{Tuple{100}}) # Shape{Tuple{100}}
```
"""
function shape_after_flatten(::Type{Shape{input}}) where input
    if length(input) == 1
        return Shape{input}  # Already flat
    end

    batch = input[1]
    features = prod(input[2:end])

    if batch === :dynamic
        return Shape{Tuple{typeof(:dynamic), features}}
    else
        return Shape{Tuple{batch, features}}
    end
end

"""
    shape_after_pool2d(
        input_shape::Type{Shape{input}},
        kernel::Tuple{Int, Int},
        stride::Tuple{Int, Int} = kernel
    ) -> Shape

Infers the output `Shape` of a 2D pooling layer (such as `MaxPool2D` or `AvgPool2D`)
given its `input_shape` and pooling parameters. Pooling layers reduce the spatial
dimensions (height and width) of the input, typically to downsample feature maps
and reduce computational complexity.

Arguments:
- `input_shape::Type{Shape{input}}`: The compile-time `Shape` of the input tensor.
                                       Expected to be 4D: `Shape{Tuple{N, H, W, C}}`,
                                       where `N` is batch size, `H` is height,
                                       `W` is width, and `C` is input channels.
- `kernel::Tuple{Int, Int}`: A tuple `(kernel_height, kernel_width)` specifying
                             the dimensions of the pooling window.
- `stride::Tuple{Int, Int}`: A tuple `(stride_height, stride_width)` specifying
                             the step size of the pooling window. If not provided,
                             it defaults to the `kernel` size (non-overlapping pools).

Returns:
- A `Shape` type representing the inferred output dimensions: `(N_out, H_out, W_out, C_out)`.

Throws:
- `AssertionError`: If `input_shape` is not 4D.

# Details on Output Dimension Calculation:
The output height (`out_H`) and width (`out_W`) are calculated using the standard
pooling formula (without explicit padding here, as pooling usually handles padding
implicitly or requires it to be pre-applied):
`out_H = floor((H - kH) / sH) + 1`
`out_W = floor((W - kW) / sW) + 1`
where `(kH, kW)` are kernel dimensions and `(sH, sW)` are stride dimensions.

The batch dimension (`N`) and channel dimension (`C`) are propagated as-is,
including `:dynamic` if present.

# Examples
```julia
# Input: (batch=1, H=28, W=28, C=32), Kernel=(2,2), Stride=(2,2)
output1 = shape_after_pool2d(Shape{Tuple{1, 28, 28, 32}}, (2, 2), (2, 2))
# Expected output: Shape{Tuple{1, 14, 14, 32}}

# Input with dynamic batch: (typeof(:dynamic), H=14, W=14, C=64), Kernel=(3,3), Stride=(1,1)
output2 = shape_after_pool2d(Shape{Tuple{typeof(:dynamic), 14, 14, 64}}, (3, 3), (1, 1))
# Expected output: Shape{Tuple{typeof(:dynamic), 12, 12, 64}}
```
"""
function shape_after_pool2d(
    ::Type{Shape{input}},
    kernel::Tuple{Int, Int},
    stride::Tuple{Int, Int} = kernel
) where input
    @assert length(input) == 4 "Pool2D expects 4D input (N, H, W, C), got $(length(input))D"

    N, H, W, C = input
    kH, kW = kernel
    sH, sW = stride

    out_H = div(H - kH, sH) + 1
    out_W = div(W - kW, sW) + 1

    if N === :dynamic
        return Shape{Tuple{typeof(:dynamic), out_H, out_W, C}}
    else
        return Shape{Tuple{N, out_H, out_W, C}}
    end
end

"""
    shape_after_global_pool(input_shape::Type{Shape{input}}) -> Shape

Infers the output `Shape` of a global pooling layer (such as `GlobalAvgPool` or `GlobalMaxPool`).
Global pooling layers reduce all spatial dimensions of an input tensor to `1`,
effectively performing pooling across the entire height and width of each feature map.
The output typically retains the batch and channel dimensions.

Arguments:
- `input_shape::Type{Shape{input}}`: The compile-time `Shape` of the input tensor.
                                       Expected to be at least 3D, with the first
                                       dimension being batch, and the last being channels.
                                       e.g., `Shape{Tuple{N, H, W, C}}` or `Shape{Tuple{N, D, H, W, C}}`.

Returns:
- A `Shape` type representing the inferred output dimensions: `(N_out, C_out)`.

Throws:
- `AssertionError`: If `input_shape` has fewer than 3 dimensions, as global pooling
                    requires spatial dimensions to reduce.

# Details:
- The first dimension of the input is treated as the batch dimension (`N`).
- The last dimension of the input is treated as the channel dimension (`C`).
- All intermediate spatial dimensions are effectively reduced to a single value
  by the global pooling operation.
- The batch dimension (`N`) is propagated as-is, including `:dynamic` if present.

# Examples
```julia
# Input: (batch=1, H=28, W=28, C=32) -> (1, 32)
output1 = shape_after_global_pool(Shape{Tuple{1, 28, 28, 32}}) # Shape{Tuple{1, 32}}

# Input with dynamic batch: (typeof(:dynamic), H=14, W=14, C=64) -> (typeof(:dynamic), 64)
output2 = shape_after_global_pool(Shape{Tuple{typeof(:dynamic), 14, 14, 64}}) # Shape{Tuple{typeof(:dynamic), 64}}

# Input: (batch=1, D=10, H=28, W=28, C=32) -> (1, 32) (assuming last is channels)
output3 = shape_after_global_pool(Shape{Tuple{1, 10, 28, 28, 32}}) # Shape{Tuple{1, 32}}
```
"""
function shape_after_global_pool(::Type{Shape{input}}) where input
    @assert length(input) >= 3 "GlobalPool expects at least 3D input (batch, spatial_dims..., channels)."

    batch = input[1]
    channels = input[end] # Assumes channels are last, common in Flux.jl

    if batch === :dynamic
        return Shape{Tuple{typeof(:dynamic), channels}}
    else
        return Shape{Tuple{batch, channels}}
    end
end

"""
    shapes_compatible(shape1::Type{Shape{s1}}, shape2::Type{Shape{s2}}) -> Bool

Checks if two compile-time `Shape` types are compatible for element-wise
operations (e.g., addition, subtraction, multiplication). Shape compatibility
is a fundamental concept for type-safe tensor operations in Axiom.jl, ensuring
that operations are only performed on tensors with valid dimensions.

Two shapes are considered compatible if:
1.  They have the same number of dimensions (rank).
2.  For every corresponding dimension, either their sizes match, or at least
    one of them is `:dynamic` (which acts as a wildcard).

Arguments:
- `shape1::Type{Shape{s1}}`: The first `Shape` type to compare.
- `shape2::Type{Shape{s2}}`: The second `Shape` type to compare.

Returns:
- `Bool`: `true` if the shapes are compatible, `false` otherwise.

# Examples
```julia
# Compatible: exact match
shapes_compatible(Shape{Tuple{2, 3}}, Shape{Tuple{2, 3}}) # true

# Compatible: dynamic dimension acts as wildcard
shapes_compatible(Shape{Tuple{typeof(:dynamic), 3}}, Shape{Tuple{2, 3}}) # true
shapes_compatible(Shape{Tuple{2, 3}}, Shape{Tuple{typeof(:dynamic), 3}}) # true
shapes_compatible(Shape{Tuple{typeof(:dynamic), 3}}, Shape{Tuple{typeof(:dynamic), 3}}) # true

# Incompatible: different ranks
shapes_compatible(Shape{Tuple{2, 3}}, Shape{Tuple{2, 3, 4}}) # false

# Incompatible: fixed dimensions mismatch
shapes_compatible(Shape{Tuple{2, 3}}, Shape{Tuple{2, 4}}) # false
```
"""
function shapes_compatible(::Type{Shape{s1}}, ::Type{Shape{s2}}) where {s1, s2}
    length(s1) == length(s2) || return false # Ranks must be the same

    for (d1, d2) in zip(s1, s2)
        # If either dimension is dynamic, they are compatible for that axis
        d1 === :dynamic && continue
        d2 === :dynamic && continue
        # Otherwise, fixed dimensions must match
        d1 == d2 || return false
    end

    return true
end

"""
    broadcast_shapes(shape1::Type{Shape{s1}}, shape2::Type{Shape{s2}}) -> Shape

Infers the resulting `Shape` when two tensors with compile-time `Shape` types
are broadcast together. This function follows Julia's standard broadcasting
rules, which allow operations between arrays of different sizes under certain
conditions (e.g., matching trailing dimensions, singleton dimensions expanding).

# Arguments
- `shape1::Type{Shape{s1}}`: The `Shape` type of the first tensor.
- `shape2::Type{Shape{s2}}`: The `Shape` type of the second tensor.

# Returns
- A `Shape` type representing the inferred broadcasted dimensions.

# Broadcasting Rules Applied:
The comparison of dimensions is performed from right to left (trailing dimensions first).
For each corresponding dimension:
- If dimensions are equal, that size is used.
- If one dimension is `1`, it is expanded to match the other.
- If one dimension is `:dynamic`, the result for that dimension is `:dynamic`.
- If dimensions are unequal and neither is `1` or `:dynamic`, an error is thrown.
- If one shape has fewer dimensions, it is effectively padded with leading `1`s.

# Throws
- `ErrorException`: If the shapes are not broadcastable according to Julia's rules.

# Examples
```julia
# Same shape
broadcast_shapes(Shape{Tuple{2, 3}}, Shape{Tuple{2, 3}}) # Shape{Tuple{2, 3}}

# Singleton dimension expansion
broadcast_shapes(Shape{Tuple{2, 3}}, Shape{Tuple{1, 3}}) # Shape{Tuple{2, 3}}
broadcast_shapes(Shape{Tuple{2, 3}}, Shape{Tuple{2, 1}}) # Shape{Tuple{2, 3}}

# Trailing singleton (Julia's rule: (3,) broadcasts with (2,3) to (2,3))
broadcast_shapes(Shape{Tuple{3}}, Shape{Tuple{2, 3}}) # Shape{Tuple{2, 3}}

# Dynamic dimension propagation
broadcast_shapes(Shape{Tuple{typeof(:dynamic), 3}}, Shape{Tuple{2, 3}}) # Shape{Tuple{typeof(:dynamic), 3}}
broadcast_shapes(Shape{Tuple{typeof(:dynamic), 1}}, Shape{Tuple{2, 3}}) # Shape{Tuple{2, 3}}
broadcast_shapes(Shape{Tuple{typeof(:dynamic), 3}}, Shape{Tuple{typeof(:dynamic), 3}}) # Shape{Tuple{typeof(:dynamic), 3}}

# Note: Incompatible fixed dimensions will throw ErrorException
```
"""
function broadcast_shapes(::Type{Shape{s1}}, ::Type{Shape{s2}}) where {s1, s2}
    n1, n2 = length(s1), length(s2)
    n = max(n1, n2)

    result = []
    for i in 1:n
        # Get dimensions from right to left, padding with 1 if shorter
        d1 = i <= n1 ? s1[n1 - i + 1] : 1
        d2 = i <= n2 ? s2[n2 - i + 1] : 1

        if d1 === :dynamic || d2 === :dynamic
            push!(result, :dynamic)
        elseif d1 == 1
            push!(result, d2)
        elseif d2 == 1
            push!(result, d1)
        elseif d1 == d2
            push!(result, d1)
        else
            error("Shapes not broadcastable: $s1 and $s2 at dimension $(n - i + 1) (from right). Got $d1 and $d2.")
        end
    end

    return Shape{Tuple(reverse(result))} # Reverse to get back to left-to-right order
end

# Shape error formatting
"""
    ShapeMismatchError <: Exception

A custom exception type thrown when a tensor shape validation fails. This
error indicates that an operation was attempted with tensors whose dimensions
are incompatible according to the compile-time shape checking rules of Axiom.jl.

Fields:
- `expected::Any`: The expected shape or dimension value. This can be a `Shape` type,
                   a `Tuple` of integers/symbols, or a single integer/symbol.
- `actual::Any`: The actual shape or dimension value encountered.
- `context::String`: A descriptive string indicating where the shape mismatch
                     occurred (e.g., "Dense layer input", "Conv2D output").
"""
struct ShapeMismatchError <: Exception
    expected::Any
    actual::Any
    context::String
end

"""
    Base.showerror(io::IO, e::ShapeMismatchError)

Custom `Base.showerror` method for `ShapeMismatchError`. This provides a
user-friendly and actionable error message in the console, guiding the
developer to identify and fix the tensor dimension incompatibility.
"""
function Base.showerror(io::IO, e::ShapeMismatchError)
    println(io, "Shape mismatch in $(e.context)")
    println(io, "  Expected: $(e.expected)")
    println(io, "  Got: $(e.actual)")
    println(io)
    println(io, "Tip: Check that your layer dimensions are compatible.")
end
