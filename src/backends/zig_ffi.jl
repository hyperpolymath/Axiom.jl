# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Zig FFI
#
# Interface to high-performance Zig backend.
# Primary native backend with SIMD vectorization and multi-threaded dispatch.

# Track if Zig backend is available
const _zig_available = Ref(false)
const _zig_lib = Ref{Ptr{Nothing}}(C_NULL)

"""
    init_zig_backend(lib_path::String)

Initialize the Zig backend from shared library.
"""
function init_zig_backend(lib_path::String)
    if !isfile(lib_path)
        error("Zig library not found: $lib_path")
    end

    try
        _zig_lib[] = Libdl.dlopen(lib_path)
        _zig_available[] = true
        @info "Zig backend initialized from $lib_path"

        # Call initialization function if exists
        init_fn = Libdl.dlsym(_zig_lib[], :axiom_zig_init; throw_error=false)
        if init_fn != C_NULL
            ccall(init_fn, Cvoid, ())
        end
    catch e
        @warn "Failed to load Zig backend: $e"
        _zig_available[] = false
    end
end

"""
    zig_available() -> Bool

Check if Zig backend is available.
"""
zig_available() = _zig_available[]

"""
    @zig_call func_name ret_type arg_types args...

Call a function in the Zig library.
"""
macro zig_call(func_name, ret_type, arg_types, args...)
    quote
        if !zig_available()
            error("Zig backend not available")
        end

        func_ptr = Libdl.dlsym(_zig_lib[], $(QuoteNode(func_name)))
        ccall(func_ptr, $(esc(ret_type)), $(esc(arg_types)), $(map(esc, args)...))
    end
end

# ============================================================================
# Row-major conversion helpers (Julia is column-major, Zig expects row-major)
# ============================================================================

"""Convert Julia column-major array to row-major flat vector for Zig FFI."""
function _to_row_major_vec(x::AbstractArray{Float32})
    if ndims(x) == 1
        return Vector{Float32}(x)
    end
    vec(permutedims(x, reverse(1:ndims(x))))
end

"""Convert row-major flat vector back to Julia column-major array."""
function _from_row_major_vec(v::Vector{Float32}, dims::Tuple)
    if length(dims) == 1
        return v
    end
    permutedims(reshape(v, reverse(dims)...), reverse(1:length(dims)))
end

# ============================================================================
# Matrix Operations
# ============================================================================

"""
Matrix multiplication via Zig.
"""
function backend_matmul(::ZigBackend, A::Matrix{Float32}, B::Matrix{Float32})
    if !zig_available()
        return A * B
    end

    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("Matrix dimensions must match"))

    A_row = _to_row_major_vec(A)
    B_row = _to_row_major_vec(B)
    C_row = Vector{Float32}(undef, m * n)

    @zig_call axiom_matmul Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t
    ) A_row B_row C_row m k n

    _from_row_major_vec(C_row, (m, n))
end

# ============================================================================
# Activation Functions
# ============================================================================

"""
ReLU via Zig.
"""
function backend_relu(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return relu(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_relu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Sigmoid via Zig.
"""
function backend_sigmoid(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return sigmoid(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_sigmoid Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
GELU via Zig.
"""
function backend_gelu(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return gelu(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_gelu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Tanh via Zig.
"""
function backend_tanh(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return tanh.(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_tanh Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Leaky ReLU via Zig.
"""
function backend_leaky_relu(::ZigBackend, x::Array{Float32}, alpha::Float32=0.01f0)
    if !zig_available()
        return leaky_relu(x, alpha)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_leaky_relu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Cfloat
    ) x y n alpha

    y
end

"""
ELU via Zig.
"""
function backend_elu(::ZigBackend, x::Array{Float32}, alpha::Float32=1.0f0)
    if !zig_available()
        return elu(x, alpha)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_elu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Cfloat
    ) x y n alpha

    y
end

"""
SELU via Zig.
"""
function backend_selu(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return selu(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_selu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Swish/SiLU via Zig.
"""
function backend_swish(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return swish(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_swish Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Mish via Zig.
"""
function backend_mish(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return mish(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_mish Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Hard Swish via Zig.
"""
function backend_hardswish(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return hardswish(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_hardswish Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Hard Sigmoid via Zig.
"""
function backend_hardsigmoid(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return hardsigmoid(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_hardsigmoid Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Softmax via Zig.
"""
function backend_softmax(::ZigBackend, x::Array{Float32}, dim::Int)
    if !zig_available()
        return softmax(x, dims=dim)
    end

    ndims(x) == 2 || return softmax(x, dims=dim)
    x_dims = size(x)
    x_row = _to_row_major_vec(x)
    y_row = Vector{Float32}(undef, length(x_row))
    batch_size = size(x, 1)
    num_classes = size(x, 2)

    @zig_call axiom_softmax Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t
    ) x_row y_row batch_size num_classes

    _from_row_major_vec(y_row, x_dims)
end

"""
Log Softmax via Zig.
"""
function backend_log_softmax(::ZigBackend, x::Array{Float32}, dim::Int)
    if !zig_available()
        return log_softmax(x, dims=dim)
    end

    ndims(x) == 2 || return log_softmax(x, dims=dim)
    x_dims = size(x)
    x_row = _to_row_major_vec(x)
    y_row = Vector{Float32}(undef, length(x_row))
    batch_size = size(x, 1)
    num_classes = size(x, 2)

    @zig_call axiom_log_softmax Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t
    ) x_row y_row batch_size num_classes

    _from_row_major_vec(y_row, x_dims)
end

"""
Softplus via Zig.
"""
function backend_softplus(::ZigBackend, x::Array{Float32})
    if !zig_available()
        return softplus(x)
    end

    y = similar(x)
    n = length(x)

    @zig_call axiom_softplus Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

# ============================================================================
# In-place Activation Functions (zero-allocation inference)
# ============================================================================

"""ReLU in-place via Zig."""
function backend_relu!(::ZigBackend, x::Array{Float32})
    zig_available() || return (x .= max.(zero(Float32), x))
    @zig_call axiom_relu_inplace Cvoid (Ptr{Float32}, Csize_t) x length(x)
    x
end

"""Sigmoid in-place via Zig."""
function backend_sigmoid!(::ZigBackend, x::Array{Float32})
    zig_available() || return (x .= 1 ./ (1 .+ exp.(-x)))
    @zig_call axiom_sigmoid_inplace Cvoid (Ptr{Float32}, Csize_t) x length(x)
    x
end

"""Tanh in-place via Zig."""
function backend_tanh!(::ZigBackend, x::Array{Float32})
    zig_available() || return (x .= tanh.(x))
    @zig_call axiom_tanh_inplace Cvoid (Ptr{Float32}, Csize_t) x length(x)
    x
end

"""GELU in-place via Zig."""
function backend_gelu!(::ZigBackend, x::Array{Float32})
    zig_available() || return (x .= gelu(x))
    @zig_call axiom_gelu_inplace Cvoid (Ptr{Float32}, Csize_t) x length(x)
    x
end

"""Swish in-place via Zig."""
function backend_swish!(::ZigBackend, x::Array{Float32})
    zig_available() || return (x .= swish(x))
    @zig_call axiom_swish_inplace Cvoid (Ptr{Float32}, Csize_t) x length(x)
    x
end

# ============================================================================
# Convolution
# ============================================================================

"""
Conv2D via Zig.
"""
function backend_conv2d(
    ::ZigBackend,
    input::Array{Float32, 4},
    weight::Array{Float32, 4},
    bias::Union{Vector{Float32}, Nothing},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int}
)
    if !zig_available()
        return backend_conv2d(JuliaBackend(), input, weight, bias, stride, padding)
    end

    N, H_in, W_in, C_in = size(input)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride
    pH, pW = padding

    H_out = div(H_in + 2*pH - kH, sH) + 1
    W_out = div(W_in + 2*pW - kW, sW) + 1

    input_row = _to_row_major_vec(input)
    weight_row = _to_row_major_vec(weight)
    output_row = Vector{Float32}(undef, N * H_out * W_out * C_out)

    bias_ptr = bias === nothing ? C_NULL : pointer(bias)

    @zig_call axiom_conv2d Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t, Csize_t,
        Csize_t, Csize_t, Csize_t,
        Csize_t, Csize_t,
        Csize_t, Csize_t,
        Csize_t, Csize_t
    ) input_row weight_row bias_ptr output_row N H_in W_in C_in H_out W_out C_out kH kW sH sW pH pW

    _from_row_major_vec(output_row, (N, H_out, W_out, C_out))
end

# ============================================================================
# Pooling
# ============================================================================

"""
MaxPool2D via Zig.
"""
function backend_maxpool2d(
    ::ZigBackend,
    input::Array{Float32, 4},
    kernel_size::Tuple{Int, Int},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int}
)
    if !zig_available()
        return backend_maxpool2d(JuliaBackend(), input, kernel_size, stride, padding)
    end

    N, H_in, W_in, C = size(input)
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    H_out = div(H_in + 2*pH - kH, sH) + 1
    W_out = div(W_in + 2*pW - kW, sW) + 1

    output = Array{Float32}(undef, N, H_out, W_out, C)

    @zig_call axiom_maxpool2d Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t, Csize_t,
        Csize_t, Csize_t,
        Csize_t, Csize_t
    ) input output N H_in W_in C kH kW sH sW

    output
end

"""
Global Average Pool2D via Zig.
"""
function backend_global_avgpool2d(::ZigBackend, input::Array{Float32, 4})
    if !zig_available()
        return backend_global_avgpool2d(JuliaBackend(), input)
    end

    N, H, W, C = size(input)
    output = Array{Float32}(undef, N, C)

    @zig_call axiom_global_avgpool2d Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t, Csize_t
    ) input output N H W C

    output
end

# ============================================================================
# Normalization
# ============================================================================

"""
Batch normalization via Zig.
"""
function backend_batchnorm(
    ::ZigBackend,
    x::Array{Float32},
    gamma::Vector{Float32},
    beta::Vector{Float32},
    running_mean::Vector{Float32},
    running_var::Vector{Float32},
    eps::Float32,
    training::Bool
)
    if !zig_available()
        return backend_batchnorm(JuliaBackend(), x, gamma, beta, running_mean, running_var, eps, training)
    end

    ndims(x) >= 2 || return backend_batchnorm(JuliaBackend(), x, gamma, beta, running_mean, running_var, eps, training)
    x_dims = Tuple(size(x))
    x_row = _to_row_major_vec(x)
    y_row = Vector{Float32}(undef, length(x_row))
    running_mean_buf = copy(running_mean)
    running_var_buf = copy(running_var)
    n_features = length(gamma)
    n_elements = length(x_row)

    @zig_call axiom_batchnorm Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Ptr{Float32}, Ptr{Float32},
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Cfloat, Cint
    ) x_row y_row gamma beta running_mean_buf running_var_buf n_elements n_features eps training

    _from_row_major_vec(y_row, x_dims)
end

"""
Layer normalization via Zig.
Converts to row-major (batch-contiguous) layout for Zig FFI.
"""
function backend_layernorm(
    ::ZigBackend,
    x::Array{Float32},
    gamma::Vector{Float32},
    beta::Vector{Float32},
    eps::Float32
)
    batch_size = size(x, 1)
    hidden_size = size(x, 2)

    if !zig_available()
        return backend_layernorm(JuliaBackend(), x, gamma, beta, (hidden_size,), eps)
    end

    x_dims = Tuple(size(x))
    x_row = _to_row_major_vec(x)
    y_row = Vector{Float32}(undef, length(x_row))

    @zig_call axiom_layernorm Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Cfloat
    ) x_row y_row gamma beta batch_size hidden_size eps

    _from_row_major_vec(y_row, x_dims)
end

"""
RMS normalization via Zig.
Converts to row-major (batch-contiguous) layout for Zig FFI.
"""
function backend_rmsnorm(
    ::ZigBackend,
    x::Array{Float32},
    weight::Vector{Float32},
    eps::Float32
)
    batch_size = size(x, 1)
    hidden_size = size(x, 2)

    if !zig_available()
        return backend_rmsnorm(JuliaBackend(), x, weight, eps)
    end

    x_dims = Tuple(size(x))

    x_row = _to_row_major_vec(x)
    y_row = Vector{Float32}(undef, length(x_row))

    @zig_call axiom_rmsnorm Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Cfloat
    ) x_row y_row weight batch_size hidden_size eps

    _from_row_major_vec(y_row, x_dims)
end

# ============================================================================
# Zig Backend Utilities
# ============================================================================

"""
Get Zig backend version info.
"""
function zig_backend_info()
    if !zig_available()
        return "Zig backend not available"
    end

    version_ptr = @zig_call axiom_zig_version Cstring ()
    version = unsafe_string(version_ptr)

    Dict(
        "version" => version,
        "lib_path" => _zig_lib[],
        "available" => true
    )
end
