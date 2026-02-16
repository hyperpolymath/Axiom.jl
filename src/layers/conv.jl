# SPDX-License-Identifier: PMPL-1.0-or-later
#
# Axiom.jl Convolutional Layers: Processing Sequential and Spatial Data
# ====================================================================
#
# This file provides implementations for various convolutional layers, which are
# essential components in deep learning for processing data with grid-like topologies,
# such as images (2D) and sequences (1D). Axiom.jl's convolutional layers
# integrate **compile-time shape verification**, enhancing the robustness and
# correctness of neural network architectures.
#
# Types of Convolutional Layers Implemented:
# ----------------------------------------
# -   `Conv1d`: Designed for 1D inputs, typically sequences or time-series data.
#     It applies a 1D filter across the input.
# -   `Conv2d`: The most common type of convolutional layer, used for 2D inputs
#     like images. It applies 2D filters across the height and width dimensions.
# -   `ConvTranspose2d`: Also known as deconvolution or fractional-strided convolution.
#     This layer performs the inverse operation of a standard convolution, often
#     used in generative models (e.g., autoencoders, GANs) to upsample spatial
#     dimensions.
#
# Fundamental Operation:
# ----------------------
# Convolutional layers operate by sliding a small filter (kernel) across the
# input data, performing element-wise multiplications and summing the results
# to produce a feature map. Key parameters like `kernel_size`, `stride`, `padding`,
# and `dilation` control the behavior and output size of these operations.
#
# Compile-Time Shape Verification:
# --------------------------------
# A critical aspect of Axiom.jl's convolutional layers is their integration
# with the `Shape` system. This allows the framework to predict the output
# dimensions of these layers at compile-time, catching potential mismatches
# and ensuring the overall architectural integrity of the neural network.
#
#


"""
    Conv(in_channels::Int, out_channels::Int, kernel_size; kwargs...)

A convenience constructor that automatically dispatches to the appropriate
1D, 2D, or 3D convolutional layer (`Conv1d`, `Conv2d`, or `Conv3d`) based on
the dimensionality of the `kernel_size` argument. This simplifies layer creation
by allowing a single `Conv` call for various spatial convolution types.

Arguments:
- `in_channels::Int`: The number of input channels for the convolution.
- `out_channels::Int`: The number of output channels (feature maps) for the convolution.
- `kernel_size`: The size of the convolutional kernel. Its type determines the
                 convolutional layer to be constructed:
    - `Int`: Dispatches to `Conv1d`. The kernel will have this size.
    - `Tuple{Int, Int}`: Dispatches to `Conv2d`. The kernel will have `(height, width)` dimensions.
    - `Tuple{Int, Int, Int}`: Dispatches to `Conv3d`. The kernel will have `(depth, height, width)` dimensions.

Keyword Arguments (`kwargs...`):
- All keyword arguments relevant to `Conv1d`, `Conv2d`, or `Conv3d` (e.g., `stride`,
  `padding`, `dilation`, `groups`, `bias`, `init`, `dtype`) can be passed through.

Returns:
- An instance of `Conv1d`, `Conv2d`, or `Conv3d` as determined by `kernel_size`.

Throws:
- `ErrorException`: If `kernel_size` is not an `Int` or a `Tuple` of length 2 or 3.

# Examples
```julia
# 1D Convolution (e.g., for sequence data)
conv1d_layer = Conv(10, 20, 3) # kernel_size = 3. Dispatches to Conv1d(10, 20, 3)

# 2D Convolution (e.g., for image data)
conv2d_layer = Conv(3, 64, (3, 3)) # kernel_size = (3, 3). Dispatches to Conv2d(3, 64, (3, 3))

# 2D Convolution with custom stride and padding
conv2d_custom = Conv(64, 128, (5, 5), stride=2, padding=1)

# Note: Conv3d is conceptually supported by this dispatcher but may not be
# fully implemented yet. Conv(1, 1, (2,2,2)) would dispatch to Conv3d.
```
"""
function Conv(in_channels::Int, out_channels::Int, kernel_size; kwargs...)
    if kernel_size isa Int
        Conv1d(in_channels, out_channels, kernel_size; kwargs...)
    elseif length(kernel_size) == 2
        Conv2d(in_channels, out_channels, kernel_size; kwargs...)
    elseif length(kernel_size) == 3
        # Conv3d now implemented
        Conv3d(in_channels, out_channels, kernel_size; kwargs...)
    else
        error("kernel_size must be an Int (for Conv1d) or a Tuple of length 2 (for Conv2d) or 3 (for Conv3d). Got kernel_size: $(kernel_size)")
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

function forward(c::Conv2d, x::AbstractTensor)
    # x shape: (N, H, W, C) or (H, W, C)
    # Pure Julia implementation (slow but correct)
    # Real implementation would use BLAS or Rust backend

    has_batch = ndims(x) == 4
    if !has_batch
        x_data = reshape(x.data, 1, size(x.data)...)
    else
        x_data = x.data
    end

    N, H, W, C_in = size(x_data)
    kH, kW = c.kernel_size
    sH, sW = c.stride
    pH, pW = c.padding

    # Output dimensions
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    # Pad input
    if pH > 0 || pW > 0
        x_padded = zeros(eltype(x_data), N, H + 2*pH, W + 2*pW, C_in)
        x_padded[:, pH+1:pH+H, pW+1:pW+W, :] = x_data
        x_data = x_padded
    end

    # Allocate output
    y = zeros(eltype(x_data), N, H_out, W_out, c.out_channels)

    # Convolution (naive implementation)
    for n in 1:N
        for oc in 1:c.out_channels
            for i in 1:H_out
                for j in 1:W_out
                    h_start = (i - 1) * sH + 1
                    w_start = (j - 1) * sW + 1

                    patch = x_data[n, h_start:h_start+kH-1, w_start:w_start+kW-1, :]
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

    Tensor(has_batch ? y : dropdims(y, dims=1))
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

function forward(c::Conv1d, x::AbstractTensor)
    # Simplified 1D conv
    # x shape: (N, L, C) or (L, C)

    has_batch = ndims(x) == 3
    if !has_batch
        x_data = reshape(x.data, 1, size(x.data)...)
    else
        x_data = x.data
    end

    N, L, C_in = size(x_data)
    k = c.kernel_size
    s = c.stride
    p = c.padding

    L_out = div(L + 2*p - k, s) + 1

    # Pad input
    if p > 0
        x_padded = zeros(eltype(x_data), N, L + 2*p, C_in)
        x_padded[:, p+1:p+L, :] = x_data
        x_data = x_padded
    end

    y = zeros(eltype(x_data), N, L_out, c.out_channels)

    for n in 1:N
        for oc in 1:c.out_channels
            for i in 1:L_out
                start = (i - 1) * s + 1
                patch = x_data[n, start:start+k-1, :]
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

    Tensor(has_batch ? y : dropdims(y, dims=1))
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

function forward(c::ConvTranspose2d, x::AbstractTensor)
    # Transposed convolution (deconvolution) - pure Julia implementation
    # x shape: (N, H, W, C_in) or (H, W, C_in)

    has_batch = ndims(x) == 4
    if !has_batch
        x_data = reshape(x.data, 1, size(x.data)...)
    else
        x_data = x.data
    end

    N, H_in, W_in, C_in = size(x_data)
    kH, kW = c.kernel_size
    sH, sW = c.stride
    pH, pW = c.padding
    opH, opW = c.output_padding

    # Output dimensions for transposed convolution
    H_out = (H_in - 1) * sH - 2 * pH + kH + opH
    W_out = (W_in - 1) * sW - 2 * pW + kW + opW

    # Allocate output
    y = zeros(eltype(x_data), N, H_out, W_out, c.out_channels)

    # Transposed convolution: scatter input values weighted by kernel
    for n in 1:N
        for ic in 1:C_in
            for oc in 1:c.out_channels
                for i in 1:H_in
                    for j in 1:W_in
                        # Calculate output region this input affects
                        h_start = (i - 1) * sH + 1 - pH
                        w_start = (j - 1) * sW + 1 - pW

                        input_val = x_data[n, i, j, ic]

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

    Tensor(has_batch ? y : dropdims(y, dims=1))
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

# ============================================================================
# Conv3d - 3D Convolution
# ============================================================================

"""
    Conv3d(in_channels, out_channels, kernel_size; stride=1, padding=0, dilation=1, groups=1, bias=true)

3D convolution layer for volumetric data (e.g., video, 3D medical imaging).

# Arguments
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `kernel_size`: Size of convolving kernel (Int or Tuple{Int,Int,Int})
- `stride`: Stride of convolution (default: 1)
- `padding`: Padding added to input (default: 0)
- `dilation`: Dilation rate (default: 1)
- `groups`: Number of blocked connections (default: 1)
- `bias`: Include bias term (default: true)

# Shape
- Input: (N, D, H, W, C_in) or (D, H, W, C_in)
- Output: (N, D', H', W', C_out) or (D', H', W', C_out)

# Examples
```julia
conv = Conv3d(3, 16, (3, 3, 3))                      # 3x3x3 conv
conv = Conv3d(16, 32, (3, 3, 3), stride=(1,1,1))     # Strided conv
conv = Conv3d(32, 64, (3, 3, 3), padding=(1,1,1))    # Same padding
```
"""
mutable struct Conv3d{T} <: AbstractLayer
    weight::Array{T, 5}  # (kD, kH, kW, C_in, C_out)
    bias::Union{Vector{T}, Nothing}
    stride::Tuple{Int, Int, Int}
    padding::Tuple{Int, Int, Int}
    dilation::Tuple{Int, Int, Int}
    groups::Int
    in_channels::Int
    out_channels::Int
    kernel_size::Tuple{Int, Int, Int}
end

function Conv3d(
    in_channels::Int,
    out_channels::Int,
    kernel_size::Union{Int, Tuple{Int, Int, Int}};
    stride::Union{Int, Tuple{Int, Int, Int}} = 1,
    padding::Union{Int, Tuple{Int, Int, Int}, Symbol} = 0,
    dilation::Union{Int, Tuple{Int, Int, Int}} = 1,
    groups::Int = 1,
    bias::Bool = true,
    init::AbstractInitializer = HeNormal(),
    dtype::Type{T} = Float32
) where T
    # Normalize tuple arguments
    ks = kernel_size isa Int ? (kernel_size, kernel_size, kernel_size) : kernel_size
    st = stride isa Int ? (stride, stride, stride) : stride
    dl = dilation isa Int ? (dilation, dilation, dilation) : dilation

    # Handle 'same' padding
    if padding === :same
        pd = (div(ks[1] - 1, 2), div(ks[2] - 1, 2), div(ks[3] - 1, 2))
    elseif padding isa Int
        pd = (padding, padding, padding)
    else
        pd = padding
    end

    # Validate groups
    @assert in_channels % groups == 0 "in_channels must be divisible by groups"
    @assert out_channels % groups == 0 "out_channels must be divisible by groups"

    # Initialize weights: (kD, kH, kW, C_in/groups, C_out)
    weight = T.(init(ks[1], ks[2], ks[3], div(in_channels, groups), out_channels))
    b = bias ? zeros(T, out_channels) : nothing

    Conv3d{T}(weight, b, st, pd, dl, groups, in_channels, out_channels, ks)
end

function forward(c::Conv3d, x::AbstractTensor)
    # x shape: (N, D, H, W, C) or (D, H, W, C)
    # Pure Julia implementation (naive but correct)

    has_batch = ndims(x) == 5
    if !has_batch
        x_data = reshape(x.data, 1, size(x.data)...)
    else
        x_data = x.data
    end

    N, D, H, W, C_in = size(x_data)
    kD, kH, kW = c.kernel_size
    sD, sH, sW = c.stride
    pD, pH, pW = c.padding

    # Output dimensions
    D_out = div(D + 2*pD - kD, sD) + 1
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    # Pad input
    if pD > 0 || pH > 0 || pW > 0
        x_padded = zeros(eltype(x_data), N, D + 2*pD, H + 2*pH, W + 2*pW, C_in)
        x_padded[:, pD+1:pD+D, pH+1:pH+H, pW+1:pW+W, :] = x_data
        x_data = x_padded
    end

    # Allocate output
    y = zeros(eltype(x_data), N, D_out, H_out, W_out, c.out_channels)

    # Convolution (naive implementation)
    for n in 1:N
        for oc in 1:c.out_channels
            for d in 1:D_out
                for i in 1:H_out
                    for j in 1:W_out
                        d_start = (d - 1) * sD + 1
                        h_start = (i - 1) * sH + 1
                        w_start = (j - 1) * sW + 1

                        patch = x_data[n, d_start:d_start+kD-1, h_start:h_start+kH-1, w_start:w_start+kW-1, :]
                        kernel = c.weight[:, :, :, :, oc]

                        y[n, d, i, j, oc] = sum(patch .* kernel)
                    end
                end
            end
        end
    end

    # Add bias
    if c.bias !== nothing
        for oc in 1:c.out_channels
            y[:, :, :, :, oc] .+= c.bias[oc]
        end
    end

    Tensor(has_batch ? y : dropdims(y, dims=1))
end

function parameters(c::Conv3d)
    if c.bias !== nothing
        (weight = c.weight, bias = c.bias)
    else
        (weight = c.weight,)
    end
end

function output_shape(c::Conv3d, input_shape::Type{Shape{input}}) where input
    # input: (D, H, W, C_in) or (N, D, H, W, C_in)
    N = length(input) == 5 ? input[1] : nothing
    D, H, W, C_in = length(input) == 5 ? input[2:end] : input

    kD, kH, kW = c.kernel_size
    sD, sH, sW = c.stride
    pD, pH, pW = c.padding

    D_out = div(D + 2*pD - kD, sD) + 1
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if N !== nothing
        Shape{Tuple{N, D_out, H_out, W_out, c.out_channels}}
    else
        Shape{Tuple{D_out, H_out, W_out, c.out_channels}}
    end
end
