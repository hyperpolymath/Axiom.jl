# SPDX-License-Identifier: PMPL-1.0-or-later
# Zig Backend FFI Bindings
# High-performance kernels implemented in Zig with SIMD optimization

"""
    ZigBackend

Backend using Zig for compute-intensive operations.
Provides SIMD-optimized kernels with zero-cost abstractions.
"""
struct ZigBackend <: AbstractBackend end

const ZIG_LIB = joinpath(@__DIR__, "..", "..", "zig", "zig-out", "lib", "libaxiom_zig")

# Check if Zig library is available
function zig_available()
    lib_path = ZIG_LIB * (Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so")
    isfile(lib_path)
end

# ============================================================================
# Matrix Operations
# ============================================================================

"""
    zig_matmul!(C, A, B)

Matrix multiplication using Zig SIMD kernels.
C = A × B where A is (m×k) and B is (k×n).
"""
function zig_matmul!(C::Matrix{Float32}, A::Matrix{Float32}, B::Matrix{Float32})
    m, k = size(A)
    k2, n = size(B)
    if k != k2
        throw(ArgumentError("Inner dimensions must match: A is ($m×$k), B is ($k2×$n)"))
    end
    if size(C) != (m, n)
        throw(ArgumentError("Output size mismatch: expected ($m×$n), got $(size(C))"))
    end

    ccall((:matmul_tiled, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t, Csize_t),
          A, B, C, m, k, n)
    return C
end

"""
    zig_matvec!(y, A, x)

Matrix-vector multiplication: y = A × x
"""
function zig_matvec!(y::Vector{Float32}, A::Matrix{Float32}, x::Vector{Float32})
    m, n = size(A)
    if length(x) != n
        throw(ArgumentError("Vector size must match matrix columns: length(x)=$(length(x)), n=$n"))
    end
    if length(y) != m
        throw(ArgumentError("Output size must match matrix rows: length(y)=$(length(y)), m=$m"))
    end

    ccall((:matvec, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t),
          A, x, y, m, n)
    return y
end

"""
    zig_transpose!(B, A)

Matrix transpose: B = A'
"""
function zig_transpose!(B::Matrix{Float32}, A::Matrix{Float32})
    m, n = size(A)
    if size(B) != (n, m)
        throw(ArgumentError("Output size must be transposed: expected ($n×$m), got $(size(B))"))
    end

    ccall((:transpose, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t),
          A, B, m, n)
    return B
end

# ============================================================================
# Activation Functions
# ============================================================================

"""
    zig_relu!(output, input)

ReLU activation: max(0, x)
"""
function zig_relu!(output::Vector{Float32}, input::Vector{Float32})
    if length(output) != length(input)
        throw(ArgumentError("Output and input lengths must match: length(output)=$(length(output)), length(input)=$(length(input))"))
    end
    ccall((:relu, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t),
          input, output, length(input))
    return output
end

"""
    zig_relu_inplace!(data)

In-place ReLU activation.
"""
function zig_relu_inplace!(data::Vector{Float32})
    ccall((:relu_inplace, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Csize_t),
          data, length(data))
    return data
end

"""
    zig_leaky_relu!(output, input, alpha=0.01)

Leaky ReLU: x if x > 0, else alpha * x
"""
function zig_leaky_relu!(output::Vector{Float32}, input::Vector{Float32}, alpha::Float32=0.01f0)
    if length(output) != length(input)
        throw(ArgumentError("Output and input lengths must match: length(output)=$(length(output)), length(input)=$(length(input))"))
    end
    ccall((:leaky_relu, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t, Cfloat),
          input, output, length(input), alpha)
    return output
end

"""
    zig_gelu!(output, input)

GELU activation (Gaussian Error Linear Unit).
"""
function zig_gelu!(output::Vector{Float32}, input::Vector{Float32})
    if length(output) != length(input)
        throw(ArgumentError("Output and input lengths must match: length(output)=$(length(output)), length(input)=$(length(input))"))
    end
    ccall((:gelu, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t),
          input, output, length(input))
    return output
end

"""
    zig_gelu_fast!(output, input)

Fast GELU approximation using sigmoid.
"""
function zig_gelu_fast!(output::Vector{Float32}, input::Vector{Float32})
    if length(output) != length(input)
        throw(ArgumentError("Output and input lengths must match: length(output)=$(length(output)), length(input)=$(length(input))"))
    end
    ccall((:gelu_fast, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t),
          input, output, length(input))
    return output
end

"""
    zig_sigmoid!(output, input)

Sigmoid activation: 1 / (1 + exp(-x))
"""
function zig_sigmoid!(output::Vector{Float32}, input::Vector{Float32})
    if length(output) != length(input)
        throw(ArgumentError("Output and input lengths must match: length(output)=$(length(output)), length(input)=$(length(input))"))
    end
    ccall((:sigmoid, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t),
          input, output, length(input))
    return output
end

"""
    zig_tanh!(output, input)

Tanh activation.
"""
function zig_tanh!(output::Vector{Float32}, input::Vector{Float32})
    if length(output) != length(input)
        throw(ArgumentError("Output and input lengths must match: length(output)=$(length(output)), length(input)=$(length(input))"))
    end
    ccall((:tanh_activation, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t),
          input, output, length(input))
    return output
end

"""
    zig_swish!(output, input)

Swish/SiLU activation: x * sigmoid(x)
"""
function zig_swish!(output::Vector{Float32}, input::Vector{Float32})
    if length(output) != length(input)
        throw(ArgumentError("Output and input lengths must match: length(output)=$(length(output)), length(input)=$(length(input))"))
    end
    ccall((:swish, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t),
          input, output, length(input))
    return output
end

"""
    zig_softmax!(output, input)

Softmax activation (numerically stable).
"""
function zig_softmax!(output::Vector{Float32}, input::Vector{Float32})
    if length(output) != length(input)
        throw(ArgumentError("Output and input lengths must match: length(output)=$(length(output)), length(input)=$(length(input))"))
    end
    ccall((:softmax, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t),
          input, output, length(input))
    return output
end

"""
    zig_softmax_batched!(output, input, batch_size, num_classes)

Batched softmax activation.
"""
function zig_softmax_batched!(output::Matrix{Float32}, input::Matrix{Float32})
    batch_size, num_classes = size(input)
    if size(output) != (batch_size, num_classes)
        throw(ArgumentError("Output size must match input size: expected ($batch_size×$num_classes), got $(size(output))"))
    end
    ccall((:softmax_batched, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t),
          input, output, batch_size, num_classes)
    return output
end

# ============================================================================
# Convolution Operations
# ============================================================================

"""
    zig_conv2d!(output, input, weight, bias, params...)

2D Convolution with optional bias.
"""
function zig_conv2d!(
    output::Array{Float32,4},
    input::Array{Float32,4},
    weight::Array{Float32,4},
    bias::Union{Vector{Float32}, Nothing},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int}
)
    if stride[1] <= 0 || stride[2] <= 0
        throw(ArgumentError("Stride must be positive, got $stride"))
    end

    batch, h_in, w_in, c_in = size(input)
    kh, kw, _, c_out = size(weight)
    h_out = (h_in + 2*padding[1] - kh) ÷ stride[1] + 1
    w_out = (w_in + 2*padding[2] - kw) ÷ stride[2] + 1

    bias_ptr = isnothing(bias) ? C_NULL : pointer(bias)

    ccall((:conv2d, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Cvoid}, Ptr{Float32},
           Csize_t, Csize_t, Csize_t, Csize_t,
           Csize_t, Csize_t, Csize_t,
           Csize_t, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t),
          input, weight, bias_ptr, output,
          batch, h_in, w_in, c_in,
          h_out, w_out, c_out,
          kh, kw, stride[1], stride[2], padding[1], padding[2])
    return output
end

"""
    zig_conv1x1!(output, input, weight, bias)

Optimized 1x1 (pointwise) convolution.
"""
function zig_conv1x1!(
    output::Array{Float32,4},
    input::Array{Float32,4},
    weight::Matrix{Float32},
    bias::Union{Vector{Float32}, Nothing}
)
    batch, h, w, c_in = size(input)
    c_out = size(weight, 2)

    bias_ptr = isnothing(bias) ? C_NULL : pointer(bias)

    ccall((:conv1x1, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Cvoid}, Ptr{Float32},
           Csize_t, Csize_t, Csize_t, Csize_t, Csize_t),
          input, weight, bias_ptr, output,
          batch, h, w, c_in, c_out)
    return output
end

"""
    zig_depthwise_conv2d!(output, input, weight, bias, stride, padding)

Depthwise separable convolution.
"""
function zig_depthwise_conv2d!(
    output::Array{Float32,4},
    input::Array{Float32,4},
    weight::Array{Float32,3},
    bias::Union{Vector{Float32}, Nothing},
    stride::Tuple{Int,Int},
    padding::Tuple{Int,Int}
)
    if stride[1] <= 0 || stride[2] <= 0
        throw(ArgumentError("Stride must be positive, got $stride"))
    end

    batch, h_in, w_in, channels = size(input)
    kh, kw, _ = size(weight)
    h_out = (h_in + 2*padding[1] - kh) ÷ stride[1] + 1
    w_out = (w_in + 2*padding[2] - kw) ÷ stride[2] + 1

    bias_ptr = isnothing(bias) ? C_NULL : pointer(bias)

    ccall((:depthwise_conv2d, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Cvoid}, Ptr{Float32},
           Csize_t, Csize_t, Csize_t, Csize_t,
           Csize_t, Csize_t,
           Csize_t, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t),
          input, weight, bias_ptr, output,
          batch, h_in, w_in, channels,
          h_out, w_out,
          kh, kw, stride[1], stride[2], padding[1], padding[2])
    return output
end

# ============================================================================
# Pooling Operations
# ============================================================================

"""
    zig_maxpool2d!(output, input, kernel_size, stride)

2D Max Pooling.
"""
function zig_maxpool2d!(
    output::Array{Float32,4},
    input::Array{Float32,4},
    kernel_size::Tuple{Int,Int},
    stride::Tuple{Int,Int}
)
    batch, h_in, w_in, channels = size(input)
    kh, kw = kernel_size

    ccall((:maxpool2d, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32},
           Csize_t, Csize_t, Csize_t, Csize_t,
           Csize_t, Csize_t, Csize_t, Csize_t),
          input, output,
          batch, h_in, w_in, channels,
          kh, kw, stride[1], stride[2])
    return output
end

"""
    zig_avgpool2d!(output, input, kernel_size, stride)

2D Average Pooling.
"""
function zig_avgpool2d!(
    output::Array{Float32,4},
    input::Array{Float32,4},
    kernel_size::Tuple{Int,Int},
    stride::Tuple{Int,Int}
)
    batch, h_in, w_in, channels = size(input)
    kh, kw = kernel_size

    ccall((:avgpool2d, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32},
           Csize_t, Csize_t, Csize_t, Csize_t,
           Csize_t, Csize_t, Csize_t, Csize_t),
          input, output,
          batch, h_in, w_in, channels,
          kh, kw, stride[1], stride[2])
    return output
end

"""
    zig_global_avgpool2d!(output, input)

Global Average Pooling: reduces (N,H,W,C) to (N,C).
"""
function zig_global_avgpool2d!(output::Matrix{Float32}, input::Array{Float32,4})
    batch, h, w, channels = size(input)
    if size(output) != (batch, channels)
        throw(ArgumentError("Output size must be ($batch×$channels), got $(size(output))"))
    end

    ccall((:global_avgpool2d, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t, Csize_t, Csize_t),
          input, output, batch, h, w, channels)
    return output
end

"""
    zig_global_maxpool2d!(output, input)

Global Max Pooling: reduces (N,H,W,C) to (N,C).
"""
function zig_global_maxpool2d!(output::Matrix{Float32}, input::Array{Float32,4})
    batch, h, w, channels = size(input)
    if size(output) != (batch, channels)
        throw(ArgumentError("Output size must be ($batch×$channels), got $(size(output))"))
    end

    ccall((:global_maxpool2d, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t, Csize_t, Csize_t),
          input, output, batch, h, w, channels)
    return output
end

"""
    zig_adaptive_avgpool2d!(output, input)

Adaptive Average Pooling: produces fixed output size.
"""
function zig_adaptive_avgpool2d!(output::Array{Float32,4}, input::Array{Float32,4})
    batch, h_in, w_in, channels = size(input)
    _, h_out, w_out, _ = size(output)

    ccall((:adaptive_avgpool2d, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32},
           Csize_t, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t),
          input, output,
          batch, h_in, w_in, channels, h_out, w_out)
    return output
end

# ============================================================================
# Normalization Operations
# ============================================================================

"""
    zig_layernorm!(output, input, gamma, beta, eps)

Layer Normalization.
"""
function zig_layernorm!(
    output::Matrix{Float32},
    input::Matrix{Float32},
    gamma::Vector{Float32},
    beta::Vector{Float32},
    eps::Float32=1f-5
)
    batch_size, hidden_size = size(input)
    if size(output) != (batch_size, hidden_size)
        throw(ArgumentError("Output size must be ($batch_size×$hidden_size), got $(size(output))"))
    end
    if length(gamma) != hidden_size
        throw(ArgumentError("Gamma length must be $hidden_size, got $(length(gamma))"))
    end
    if length(beta) != hidden_size
        throw(ArgumentError("Beta length must be $hidden_size, got $(length(beta))"))
    end

    ccall((:layernorm, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Csize_t, Csize_t, Cfloat),
          input, output, gamma, beta, batch_size, hidden_size, eps)
    return output
end

"""
    zig_rmsnorm!(output, input, weight, eps)

RMS Normalization (used in LLaMA, etc.).
"""
function zig_rmsnorm!(
    output::Matrix{Float32},
    input::Matrix{Float32},
    weight::Vector{Float32},
    eps::Float32=1f-5
)
    batch_size, hidden_size = size(input)
    if size(output) != (batch_size, hidden_size)
        throw(ArgumentError("Output size must be ($batch_size×$hidden_size), got $(size(output))"))
    end
    if length(weight) != hidden_size
        throw(ArgumentError("Weight length must be $hidden_size, got $(length(weight))"))
    end

    ccall((:rmsnorm, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Csize_t, Csize_t, Cfloat),
          input, output, weight, batch_size, hidden_size, eps)
    return output
end

"""
    zig_batchnorm!(output, input, gamma, beta, running_mean, running_var, eps)

Batch Normalization (inference mode).
"""
function zig_batchnorm!(
    output::Matrix{Float32},
    input::Matrix{Float32},
    gamma::Vector{Float32},
    beta::Vector{Float32},
    running_mean::Vector{Float32},
    running_var::Vector{Float32},
    eps::Float32=1f-5
)
    batch_size, num_features = size(input)
    if size(output) != (batch_size, num_features)
        throw(ArgumentError("Output size must be ($batch_size×$num_features), got $(size(output))"))
    end

    ccall((:batchnorm, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t, Cfloat),
          input, output, gamma, beta, running_mean, running_var,
          batch_size, num_features, eps)
    return output
end

"""
    zig_instancenorm!(output, input, gamma, beta, eps)

Instance Normalization.
"""
function zig_instancenorm!(
    output::Array{Float32,4},
    input::Array{Float32,4},
    gamma::Union{Vector{Float32}, Nothing},
    beta::Union{Vector{Float32}, Nothing},
    eps::Float32=1f-5
)
    batch, h, w, channels = size(input)

    gamma_ptr = isnothing(gamma) ? C_NULL : pointer(gamma)
    beta_ptr = isnothing(beta) ? C_NULL : pointer(beta)

    ccall((:instancenorm, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Cvoid}, Ptr{Cvoid},
           Csize_t, Csize_t, Csize_t, Csize_t, Cfloat),
          input, output, gamma_ptr, beta_ptr,
          batch, h, w, channels, eps)
    return output
end

"""
    zig_groupnorm!(output, input, gamma, beta, num_groups, eps)

Group Normalization.
"""
function zig_groupnorm!(
    output::Array{Float32,4},
    input::Array{Float32,4},
    gamma::Vector{Float32},
    beta::Vector{Float32},
    num_groups::Int,
    eps::Float32=1f-5
)
    batch, h, w, channels = size(input)
    if channels % num_groups != 0
        throw(ArgumentError("Channels ($channels) must be divisible by num_groups ($num_groups)"))
    end

    ccall((:groupnorm, ZIG_LIB), Cvoid,
          (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
           Csize_t, Csize_t, Csize_t, Csize_t, Csize_t, Cfloat),
          input, output, gamma, beta,
          batch, h, w, channels, num_groups, eps)
    return output
end

# ============================================================================
# Backend Interface Implementation
# ============================================================================

function forward(::ZigBackend, layer::Dense, x::AbstractArray)
    if !zig_available()
        return forward(JuliaBackend(), layer, x)
    end

    # Convert to expected format
    input = Float32.(x)
    batch_size = size(input, 1)
    output = zeros(Float32, batch_size, layer.out_features)

    # Use Zig matmul
    zig_matmul!(output, input, Float32.(layer.weight'))

    # Add bias if present
    if layer.use_bias
        output .+= layer.bias'
    end

    return output
end

function forward(::ZigBackend, layer::Conv2D, x::AbstractArray)
    if !zig_available()
        return forward(JuliaBackend(), layer, x)
    end

    input = Float32.(x)
    batch, h_in, w_in, _ = size(input)
    h_out = (h_in + 2*layer.padding[1] - layer.kernel_size[1]) ÷ layer.stride[1] + 1
    w_out = (w_in + 2*layer.padding[2] - layer.kernel_size[2]) ÷ layer.stride[2] + 1

    output = zeros(Float32, batch, h_out, w_out, layer.out_channels)
    weight = Float32.(layer.weight)
    bias = layer.use_bias ? Float32.(layer.bias) : nothing

    zig_conv2d!(output, input, weight, bias, layer.stride, layer.padding)
    return output
end

# Export backend
export ZigBackend, zig_available
export zig_matmul!, zig_matvec!, zig_transpose!
export zig_relu!, zig_relu_inplace!, zig_leaky_relu!, zig_gelu!, zig_sigmoid!, zig_tanh!, zig_swish!, zig_softmax!
export zig_conv2d!, zig_conv1x1!, zig_depthwise_conv2d!
export zig_maxpool2d!, zig_avgpool2d!, zig_global_avgpool2d!, zig_global_maxpool2d!, zig_adaptive_avgpool2d!
export zig_layernorm!, zig_rmsnorm!, zig_batchnorm!, zig_instancenorm!, zig_groupnorm!