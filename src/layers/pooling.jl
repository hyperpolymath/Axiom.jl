# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Pooling Layers
#
# MaxPool, AvgPool, GlobalPool, AdaptivePool, Flatten

"""
    MaxPool(kernel_size; stride=kernel_size, padding=0)

Max pooling layer.
"""
struct MaxPool2d <: StatelessLayer
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
end

function MaxPool2d(
    kernel_size::Union{Int, Tuple{Int, Int}};
    stride::Union{Int, Tuple{Int, Int}, Nothing} = nothing,
    padding::Union{Int, Tuple{Int, Int}} = 0
)
    ks = kernel_size isa Int ? (kernel_size, kernel_size) : kernel_size
    st = stride === nothing ? ks : (stride isa Int ? (stride, stride) : stride)
    pd = padding isa Int ? (padding, padding) : padding

    MaxPool2d(ks, st, pd)
end

# Alias
MaxPool(args...; kwargs...) = MaxPool2d(args...; kwargs...)

function forward(mp::MaxPool2d, x::AbstractTensor)
    has_batch = ndims(x) == 4
    x_data = has_batch ? x.data : reshape(x.data, 1, size(x.data)...)

    N, H, W, C = size(x_data)
    kH, kW = mp.kernel_size
    sH, sW = mp.stride
    pH, pW = mp.padding

    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    # Pad input
    if pH > 0 || pW > 0
        x_padded = fill(typemin(eltype(x_data)), N, H + 2*pH, W + 2*pW, C)
        x_padded[:, pH+1:pH+H, pW+1:pW+W, :] = x_data
        x_data = x_padded
    end

    y = zeros(eltype(x_data), N, H_out, W_out, C)

    for n in 1:N
        for c in 1:C
            for i in 1:H_out
                for j in 1:W_out
                    h_start = (i - 1) * sH + 1
                    w_start = (j - 1) * sW + 1

                    patch = x_data[n, h_start:h_start+kH-1, w_start:w_start+kW-1, c]
                    y[n, i, j, c] = maximum(patch)
                end
            end
        end
    end

    Tensor(has_batch ? y : dropdims(y, dims=1))
end

function output_shape(mp::MaxPool2d, input_shape)
    H, W, C = input_shape[end-2:end]
    N = length(input_shape) == 4 ? input_shape[1] : nothing

    kH, kW = mp.kernel_size
    sH, sW = mp.stride
    pH, pW = mp.padding

    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if N !== nothing
        (N, H_out, W_out, C)
    else
        (H_out, W_out, C)
    end
end

"""
    AvgPool(kernel_size; stride=kernel_size, padding=0)

Average pooling layer.
"""
struct AvgPool2d <: StatelessLayer
    kernel_size::Tuple{Int, Int}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    count_include_pad::Bool
end

function AvgPool2d(
    kernel_size::Union{Int, Tuple{Int, Int}};
    stride::Union{Int, Tuple{Int, Int}, Nothing} = nothing,
    padding::Union{Int, Tuple{Int, Int}} = 0,
    count_include_pad::Bool = true
)
    ks = kernel_size isa Int ? (kernel_size, kernel_size) : kernel_size
    st = stride === nothing ? ks : (stride isa Int ? (stride, stride) : stride)
    pd = padding isa Int ? (padding, padding) : padding

    AvgPool2d(ks, st, pd, count_include_pad)
end

# Alias
AvgPool(args...; kwargs...) = AvgPool2d(args...; kwargs...)

function forward(ap::AvgPool2d, x::AbstractTensor)
    has_batch = ndims(x) == 4
    x_data = has_batch ? x.data : reshape(x.data, 1, size(x.data)...)

    N, H, W, C = size(x_data)
    kH, kW = ap.kernel_size
    sH, sW = ap.stride
    pH, pW = ap.padding

    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    # Pad input
    if pH > 0 || pW > 0
        x_padded = zeros(eltype(x_data), N, H + 2*pH, W + 2*pW, C)
        x_padded[:, pH+1:pH+H, pW+1:pW+W, :] = x_data
        x_data = x_padded
    end

    y = zeros(eltype(x_data), N, H_out, W_out, C)

    for n in 1:N
        for c in 1:C
            for i in 1:H_out
                for j in 1:W_out
                    h_start = (i - 1) * sH + 1
                    w_start = (j - 1) * sW + 1

                    patch = x_data[n, h_start:h_start+kH-1, w_start:w_start+kW-1, c]
                    y[n, i, j, c] = mean(patch)
                end
            end
        end
    end

    Tensor(has_batch ? y : dropdims(y, dims=1))
end

output_shape(ap::AvgPool2d, input_shape) = output_shape(MaxPool2d(ap.kernel_size, ap.stride, ap.padding), input_shape)

"""
    GlobalAvgPool()

Global average pooling - reduces spatial dimensions to 1.
"""
struct GlobalAvgPool <: StatelessLayer end

function forward(::GlobalAvgPool, x::AbstractTensor)
    # x: (N, H, W, C) -> (N, C) or (H, W, C) -> (C,)
    x_data = x.data
    spatial_dims = collect(2:ndims(x_data)-1)
    if isempty(spatial_dims)
        # 2D input (N, C) - just return as is
        return x
    end
    Tensor(dropdims(mean(x_data, dims=Tuple(spatial_dims)), dims=Tuple(spatial_dims)))
end

function output_shape(::GlobalAvgPool, input_shape)
    if length(input_shape) >= 3
        # (N, H, W, C) -> (N, C)
        (input_shape[1], input_shape[end])
    else
        # (H, W, C) -> (C,)
        (input_shape[end],)
    end
end

"""
    GlobalMaxPool()

Global max pooling - reduces spatial dimensions to 1.
"""
struct GlobalMaxPool <: StatelessLayer end

function forward(::GlobalMaxPool, x::AbstractTensor)
    x_data = x.data
    spatial_dims = collect(2:ndims(x_data)-1)
    if isempty(spatial_dims)
        return x
    end
    Tensor(dropdims(maximum(x_data, dims=Tuple(spatial_dims)), dims=Tuple(spatial_dims)))
end

output_shape(::GlobalMaxPool, input_shape) = output_shape(GlobalAvgPool(), input_shape)

"""
    AdaptiveAvgPool(output_size)

Adaptive average pooling - outputs fixed size regardless of input.
"""
struct AdaptiveAvgPool2d <: StatelessLayer
    output_size::Tuple{Int, Int}
end

AdaptiveAvgPool2d(size::Int) = AdaptiveAvgPool2d((size, size))

function forward(ap::AdaptiveAvgPool2d, x::AbstractTensor)
    has_batch = ndims(x) == 4
    x_data = has_batch ? x.data : reshape(x.data, 1, size(x.data)...)

    N, H_in, W_in, C = size(x_data)
    H_out, W_out = ap.output_size

    y = zeros(eltype(x_data), N, H_out, W_out, C)

    for i in 1:H_out
        for j in 1:W_out
            # Compute window
            h_start = floor(Int, (i - 1) * H_in / H_out) + 1
            h_end = ceil(Int, i * H_in / H_out)
            w_start = floor(Int, (j - 1) * W_in / W_out) + 1
            w_end = ceil(Int, j * W_in / W_out)

            y[:, i, j, :] = mean(x_data[:, h_start:h_end, w_start:w_end, :], dims=(2, 3))
        end
    end

    Tensor(has_batch ? y : dropdims(y, dims=1))
end

function output_shape(ap::AdaptiveAvgPool2d, input_shape)
    H_out, W_out = ap.output_size
    if length(input_shape) == 4
        (input_shape[1], H_out, W_out, input_shape[end])
    else
        (H_out, W_out, input_shape[end])
    end
end

"""
    Flatten(; start_dim=2)

Flatten layer - reshapes input to 2D (batch, features).
"""
struct Flatten <: StatelessLayer
    start_dim::Int
end

Flatten(; start_dim::Int=2) = Flatten(start_dim)


function forward(f::Flatten, x::AbstractTensor)
    if ndims(x) <= 2
        return x
    end

    # Keep batch dimension, flatten the rest
    x_data = x.data
    batch_size = size(x_data, 1)
    features = prod(size(x_data)[f.start_dim:end])

    Tensor(reshape(x_data, batch_size, features))
end

function output_shape(f::Flatten, input_shape)
    if length(input_shape) <= 1
        return input_shape
    end

    batch = input_shape[1]
    features = prod(input_shape[f.start_dim:end])

    (batch, features)
end

"""
    Reshape(shape)

Reshape layer with automatic batch dimension handling.
"""
struct Reshape{S} <: StatelessLayer
    shape::S
end

function forward(r::Reshape, x::AbstractTensor)
    x_data = x.data
    batch_size = size(x_data, 1)
    new_shape = (-1, r.shape...)
    Tensor(reshape(x_data, batch_size, r.shape...))
end

function output_shape(r::Reshape, input_shape)
    (input_shape[1], r.shape...)
end

"""
    Unsqueeze(dim)

Add a dimension at the specified position.
"""
struct Unsqueeze <: StatelessLayer
    dim::Int
end

function forward(u::Unsqueeze, x::AbstractTensor)
    x_data = x.data
    shape = collect(size(x_data))
    insert!(shape, u.dim, 1)
    Tensor(reshape(x_data, shape...))
end

function output_shape(u::Unsqueeze, input_shape)
    shape = collect(input_shape)
    insert!(shape, u.dim, 1)
    Tuple(shape)
end

"""
    Squeeze(dim=nothing)

Remove dimensions of size 1.
"""
struct Squeeze <: StatelessLayer
    dim::Union{Int, Nothing}
end

Squeeze() = Squeeze(nothing)

function forward(s::Squeeze, x::AbstractTensor)
    x_data = x.data
    if s.dim === nothing
        Tensor(dropdims(x_data, dims=Tuple(findall(==(1), size(x_data)))))
    else
        if size(x_data, s.dim) == 1
            Tensor(dropdims(x_data, dims=s.dim))
        else
            x
        end
    end
end
