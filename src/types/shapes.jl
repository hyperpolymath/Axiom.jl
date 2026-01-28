# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Shape System
#
# Compile-time shape inference and verification.
# This is the core of Axiom.jl's type-safe tensor operations.

"""
    Shape{dims}

Compile-time shape representation.

# Examples
```julia
Shape{(28, 28, 1)}      # Image shape
Shape{(32, :, 784)}     # Batch with dynamic size
Shape{(:, :)}           # Fully dynamic 2D
```
"""
struct Shape{dims} end

# Shape arithmetic for layer composition
const DynamicDim = Symbol(":")

"""
    shape_after_dense(input_shape, out_features) -> Shape

Compute output shape after Dense layer.
"""
function shape_after_dense(::Type{Shape{input}}, out_features::Int) where input
    if length(input) == 1
        return Shape{(out_features,)}
    elseif length(input) == 2
        # (batch, features) -> (batch, out_features)
        batch = input[1]
        return Shape{(batch, out_features)}
    else
        error("Dense layer expects 1D or 2D input, got $(length(input))D")
    end
end

"""
    shape_after_conv2d(input_shape, out_channels, kernel, stride, padding) -> Shape

Compute output shape after Conv2D layer.
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

    # Output dimensions
    out_H = div(H + 2*pH - kH, sH) + 1
    out_W = div(W + 2*pW - kW, sW) + 1

    # Handle dynamic batch
    if N === :dynamic
        return Shape{(:dynamic, out_H, out_W, out_channels)}
    else
        return Shape{(N, out_H, out_W, out_channels)}
    end
end

"""
    shape_after_flatten(input_shape) -> Shape

Compute output shape after Flatten layer.
"""
function shape_after_flatten(::Type{Shape{input}}) where input
    if length(input) == 1
        return Shape{input}  # Already flat
    end

    batch = input[1]
    features = prod(input[2:end])

    if batch === :dynamic
        return Shape{(:dynamic, features)}
    else
        return Shape{(batch, features)}
    end
end

"""
    shape_after_pool2d(input_shape, kernel, stride) -> Shape

Compute output shape after MaxPool2D or AvgPool2D.
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
        return Shape{(:dynamic, out_H, out_W, C)}
    else
        return Shape{(N, out_H, out_W, C)}
    end
end

"""
    shape_after_global_pool(input_shape) -> Shape

Compute output shape after GlobalAvgPool or GlobalMaxPool.
"""
function shape_after_global_pool(::Type{Shape{input}}) where input
    @assert length(input) >= 3 "GlobalPool expects at least 3D input"

    batch = input[1]
    channels = input[end]

    if batch === :dynamic
        return Shape{(:dynamic, channels)}
    else
        return Shape{(batch, channels)}
    end
end

"""
    shapes_compatible(shape1, shape2) -> Bool

Check if two shapes are compatible for operations.
"""
function shapes_compatible(::Type{Shape{s1}}, ::Type{Shape{s2}}) where {s1, s2}
    length(s1) == length(s2) || return false

    for (d1, d2) in zip(s1, s2)
        # Dynamic dimensions are always compatible
        d1 === :dynamic && continue
        d2 === :dynamic && continue
        d1 == d2 || return false
    end

    return true
end

"""
    broadcast_shapes(shape1, shape2) -> Shape

Compute broadcast result shape.
"""
function broadcast_shapes(::Type{Shape{s1}}, ::Type{Shape{s2}}) where {s1, s2}
    n1, n2 = length(s1), length(s2)
    n = max(n1, n2)

    result = []
    for i in 1:n
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
            error("Shapes not broadcastable: $s1 and $s2")
        end
    end

    return Shape{Tuple(reverse(result))}
end

# Shape error formatting
struct ShapeMismatchError <: Exception
    expected::Any
    actual::Any
    context::String
end

function Base.showerror(io::IO, e::ShapeMismatchError)
    println(io, "Shape mismatch in $(e.context)")
    println(io, "  Expected: $(e.expected)")
    println(io, "  Got: $(e.actual)")
    println(io)
    println(io, "Tip: Check that your layer dimensions are compatible.")
end
