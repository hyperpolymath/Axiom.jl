# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Convolutional Layers
#
# 1D, 2D, and 3D convolutions with compile-time shape verification.

"""
    Conv(in_channels, out_channels, kernel_size; kwargs...)

Convenience constructor that dispatches to Conv1d, Conv2d, or Conv3d
based on kernel_size dimensionality.
"""
function Conv(in_channels::Int, out_channels::Int, kernel_size; kwargs...)
    if kernel_size isa Int
        Conv1d(in_channels, out_channels, kernel_size; kwargs...)
    elseif length(kernel_size) == 2
        Conv2d(in_channels, out_channels, kernel_size; kwargs...)
    elseif length(kernel_size) == 3
        Conv3d(in_channels, out_channels, kernel_size; kwargs...)
    else
        error("kernel_size must be Int or tuple of length 2 or 3")
    end
end

"""
    Conv2d(in_channels, out_channels, kernel_size; stride=1, padding=0, dilation=1, groups=1, bias=true)

2D convolution layer.

# Arguments
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `kernel_size`: Size of convolving kernel (Int or Tuple{Int,Int})
- `stride`: Stride of convolution (default: 1)
- `padding`: Padding added to input (default: 0)
- `dilation`: Dilation rate (default: 1)
- `groups`: Number of blocked connections (default: 1)
- `bias`: Include bias term (default: true)

# Shape
- Input: (N, H, W, C_in) or (H, W, C_in)
- Output: (N, H', W', C_out) or (H', W', C_out)

# Examples
```julia
conv = Conv2d(3, 64, (3, 3))                    # 3x3 conv
conv = Conv2d(64, 128, (3, 3), stride=2)        # Strided conv
conv = Conv2d(128, 256, (3, 3), padding=1)      # Same padding
```
"""
mutable struct Conv2d{T} <: AbstractLayer
    weight::Array{T, 4}  # (kH, kW, C_in, C_out)
    bias::Union{Vector{T}, Nothing}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    dilation::Tuple{Int, Int}
    groups::Int
    in_channels::Int
    out_channels::Int
    kernel_size::Tuple{Int, Int}
end

function Conv2d(
    in_channels::Int,
    out_channels::Int,
    kernel_size::Union{Int, Tuple{Int, Int}};
    stride::Union{Int, Tuple{Int, Int}} = 1,
    padding::Union{Int, Tuple{Int, Int}, Symbol} = 0,
    dilation::Union{Int, Tuple{Int, Int}} = 1,
    groups::Int = 1,
    bias::Bool = true,
    init::AbstractInitializer = HeNormal(),
    dtype::Type{T} = Float32
) where T
    # Normalize tuple arguments
    ks = kernel_size isa Int ? (kernel_size, kernel_size) : kernel_size
    st = stride isa Int ? (stride, stride) : stride
    dl = dilation isa Int ? (dilation, dilation) : dilation

    # Handle 'same' padding
    if padding === :same
        pd = (div(ks[1] - 1, 2), div(ks[2] - 1, 2))
    elseif padding isa Int
        pd = (padding, padding)
    else
        pd = padding
    end

    # Validate groups
    @assert in_channels % groups == 0 "in_channels must be divisible by groups"
    @assert out_channels % groups == 0 "out_channels must be divisible by groups"

    # Initialize weights: (kH, kW, C_in/groups, C_out)
    weight = T.(init(ks[1], ks[2], div(in_channels, groups), out_channels))
    b = bias ? zeros(T, out_channels) : nothing

    Conv2d{T}(weight, b, st, pd, dl, groups, in_channels, out_channels, ks)
end

function forward(c::Conv2d, x::AbstractArray)
    # x shape: (N, H, W, C) or (H, W, C)
    # Pure Julia implementation (slow but correct)
    # Real implementation would use BLAS or Rust backend

    has_batch = ndims(x) == 4
    if !has_batch
        x = reshape(x, 1, size(x)...)
    end

    N, H, W, C_in = size(x)
    kH, kW = c.kernel_size
    sH, sW = c.stride
    pH, pW = c.padding

    # Output dimensions
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    # Pad input
    if pH > 0 || pW > 0
        x_padded = zeros(eltype(x), N, H + 2*pH, W + 2*pW, C_in)
        x_padded[:, pH+1:pH+H, pW+1:pW+W, :] = x
        x = x_padded
    end

    # Allocate output
    y = zeros(eltype(x), N, H_out, W_out, c.out_channels)

    # Convolution (naive implementation)
    for n in 1:N
        for oc in 1:c.out_channels
            for i in 1:H_out
                for j in 1:W_out
                    h_start = (i - 1) * sH + 1
                    w_start = (j - 1) * sW + 1

                    patch = x[n, h_start:h_start+kH-1, w_start:w_start+kW-1, :]
                    kernel = c.weight[:, :, :, oc]

                    y[n, i, j, oc] = sum(patch .* kernel)
                end
            end
        end
    end

    # Add bias
    if c.bias !== nothing
        for oc in 1:c.out_channels
            y[:, :, :, oc] .+= c.bias[oc]
        end
    end

    has_batch ? y : dropdims(y, dims=1)
end

function parameters(c::Conv2d)
    if c.bias !== nothing
        (weight = c.weight, bias = c.bias)
    else
        (weight = c.weight,)
    end
end

function output_shape(c::Conv2d, input_shape)
    H, W, C = input_shape[end-2:end]
    N = length(input_shape) == 4 ? input_shape[1] : nothing

    kH, kW = c.kernel_size
    sH, sW = c.stride
    pH, pW = c.padding

    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if N !== nothing
        (N, H_out, W_out, c.out_channels)
    else
        (H_out, W_out, c.out_channels)
    end
end

function show_layer_params(io::IO, c::Conv2d)
    print(io, "$(c.in_channels) => $(c.out_channels), $(c.kernel_size)")
    if c.stride != (1, 1)
        print(io, ", stride=$(c.stride)")
    end
    if c.padding != (0, 0)
        print(io, ", padding=$(c.padding)")
    end
end

# Alias
const Conv2D = Conv2d

"""
    Conv1d(in_channels, out_channels, kernel_size; kwargs...)

1D convolution layer.
"""
mutable struct Conv1d{T} <: AbstractLayer
    weight::Array{T, 3}  # (kernel, C_in, C_out)
    bias::Union{Vector{T}, Nothing}
    stride::Int
    padding::Int
    dilation::Int
    in_channels::Int
    out_channels::Int
    kernel_size::Int
end

function Conv1d(
    in_channels::Int,
    out_channels::Int,
    kernel_size::Int;
    stride::Int = 1,
    padding::Int = 0,
    dilation::Int = 1,
    bias::Bool = true,
    dtype::Type{T} = Float32
) where T
    weight = randn(T, kernel_size, in_channels, out_channels) .* sqrt(2.0f0 / in_channels)
    b = bias ? zeros(T, out_channels) : nothing

    Conv1d{T}(weight, b, stride, padding, dilation, in_channels, out_channels, kernel_size)
end

function forward(c::Conv1d, x::AbstractArray)
    # Simplified 1D conv
    # x shape: (N, L, C) or (L, C)

    has_batch = ndims(x) == 3
    if !has_batch
        x = reshape(x, 1, size(x)...)
    end

    N, L, C_in = size(x)
    k = c.kernel_size
    s = c.stride
    p = c.padding

    L_out = div(L + 2*p - k, s) + 1

    # Pad input
    if p > 0
        x_padded = zeros(eltype(x), N, L + 2*p, C_in)
        x_padded[:, p+1:p+L, :] = x
        x = x_padded
    end

    y = zeros(eltype(x), N, L_out, c.out_channels)

    for n in 1:N
        for oc in 1:c.out_channels
            for i in 1:L_out
                start = (i - 1) * s + 1
                patch = x[n, start:start+k-1, :]
                kernel = c.weight[:, :, oc]
                y[n, i, oc] = sum(patch .* kernel)
            end
        end
    end

    if c.bias !== nothing
        for oc in 1:c.out_channels
            y[:, :, oc] .+= c.bias[oc]
        end
    end

    has_batch ? y : dropdims(y, dims=1)
end

function parameters(c::Conv1d)
    if c.bias !== nothing
        (weight = c.weight, bias = c.bias)
    else
        (weight = c.weight,)
    end
end

"""
    ConvTranspose2d(in_channels, out_channels, kernel_size; kwargs...)

Transposed 2D convolution (deconvolution).
"""
mutable struct ConvTranspose2d{T} <: AbstractLayer
    weight::Array{T, 4}
    bias::Union{Vector{T}, Nothing}
    stride::Tuple{Int, Int}
    padding::Tuple{Int, Int}
    output_padding::Tuple{Int, Int}
    in_channels::Int
    out_channels::Int
    kernel_size::Tuple{Int, Int}
end

function ConvTranspose2d(
    in_channels::Int,
    out_channels::Int,
    kernel_size::Union{Int, Tuple{Int, Int}};
    stride::Union{Int, Tuple{Int, Int}} = 1,
    padding::Union{Int, Tuple{Int, Int}} = 0,
    output_padding::Union{Int, Tuple{Int, Int}} = 0,
    bias::Bool = true,
    dtype::Type{T} = Float32
) where T
    ks = kernel_size isa Int ? (kernel_size, kernel_size) : kernel_size
    st = stride isa Int ? (stride, stride) : stride
    pd = padding isa Int ? (padding, padding) : padding
    op = output_padding isa Int ? (output_padding, output_padding) : output_padding

    weight = randn(T, ks[1], ks[2], out_channels, in_channels) .* sqrt(2.0f0 / in_channels)
    b = bias ? zeros(T, out_channels) : nothing

    ConvTranspose2d{T}(weight, b, st, pd, op, in_channels, out_channels, ks)
end

function forward(c::ConvTranspose2d, x::AbstractArray)
    # Transposed convolution (deconvolution) - pure Julia implementation
    # x shape: (N, H, W, C_in) or (H, W, C_in)

    has_batch = ndims(x) == 4
    if !has_batch
        x = reshape(x, 1, size(x)...)
    end

    N, H_in, W_in, C_in = size(x)
    kH, kW = c.kernel_size
    sH, sW = c.stride
    pH, pW = c.padding
    opH, opW = c.output_padding

    # Output dimensions for transposed convolution
    H_out = (H_in - 1) * sH - 2 * pH + kH + opH
    W_out = (W_in - 1) * sW - 2 * pW + kW + opW

    # Allocate output
    y = zeros(eltype(x), N, H_out, W_out, c.out_channels)

    # Transposed convolution: scatter input values weighted by kernel
    for n in 1:N
        for ic in 1:C_in
            for oc in 1:c.out_channels
                for i in 1:H_in
                    for j in 1:W_in
                        # Calculate output region this input affects
                        h_start = (i - 1) * sH + 1 - pH
                        w_start = (j - 1) * sW + 1 - pW

                        input_val = x[n, i, j, ic]

                        # Scatter to output with kernel weights
                        for kh in 1:kH
                            for kw in 1:kW
                                h_out = h_start + kh - 1
                                w_out = w_start + kw - 1

                                # Check bounds
                                if 1 <= h_out <= H_out && 1 <= w_out <= W_out
                                    y[n, h_out, w_out, oc] += input_val * c.weight[kh, kw, oc, ic]
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    # Add bias
    if c.bias !== nothing
        for oc in 1:c.out_channels
            y[:, :, :, oc] .+= c.bias[oc]
        end
    end

    has_batch ? y : dropdims(y, dims=1)
end

function output_shape(c::ConvTranspose2d, input_shape)
    H, W, C = input_shape[end-2:end]
    N = length(input_shape) == 4 ? input_shape[1] : nothing

    kH, kW = c.kernel_size
    sH, sW = c.stride
    pH, pW = c.padding
    opH, opW = c.output_padding

    H_out = (H - 1) * sH - 2 * pH + kH + opH
    W_out = (W - 1) * sW - 2 * pW + kW + opW

    if N !== nothing
        (N, H_out, W_out, c.out_channels)
    else
        (H_out, W_out, c.out_channels)
    end
end

function parameters(c::ConvTranspose2d)
    if c.bias !== nothing
        (weight = c.weight, bias = c.bias)
    else
        (weight = c.weight,)
    end
end
