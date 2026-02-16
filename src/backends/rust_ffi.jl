# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Rust FFI
#
# Interface to high-performance Rust backend.

# Track if Rust backend is available
const _rust_available = Ref(false)
const _rust_lib = Ref{Ptr{Nothing}}(C_NULL)

"""
    init_rust_backend(lib_path::String)

Initialize the Rust backend from shared library.
"""
function init_rust_backend(lib_path::String)
    if !isfile(lib_path)
        error("Rust library not found: $lib_path")
    end

    try
        _rust_lib[] = Libdl.dlopen(lib_path)
        _rust_available[] = true
        @info "Rust backend initialized from $lib_path"
    catch e
        @warn "Failed to load Rust backend: $e"
        _rust_available[] = false
    end
end

"""
    rust_available() -> Bool

Check if Rust backend is available.
"""
rust_available() = _rust_available[]

"""
    @rust_call func_name ret_type arg_types args...

Call a function in the Rust library.
"""
macro rust_call(func_name, ret_type, arg_types, args...)
    quote
        if !rust_available()
            error("Rust backend not available")
        end

        func_ptr = Libdl.dlsym(_rust_lib[], $(QuoteNode(func_name)))
        ccall(func_ptr, $(esc(ret_type)), $(esc(arg_types)), $(map(esc, args)...))
    end
end

@inline _reverse_perm(::Val{N}) where {N} = ntuple(i -> N - i + 1, N)

"""
    _to_row_major_vec(x::Array{Float32,N}) where N

Convert a Julia column-major array to a row-major flattened buffer for Rust FFI.
"""
function _to_row_major_vec(x::Array{Float32, N}) where {N}
    N == 1 && return copy(vec(x))
    perm = _reverse_perm(Val(N))
    vec(Array(permutedims(x, perm)))
end

"""
    _from_row_major_vec(data::Vector{Float32}, dims::NTuple{N, Int}) where N

Convert a row-major flattened Rust output buffer back to Julia column-major array.
"""
function _from_row_major_vec(data::Vector{Float32}, dims::NTuple{N, Int}) where {N}
    N == 1 && return copy(data)
    perm = _reverse_perm(Val(N))
    reshaped = reshape(data, reverse(dims))
    Array(permutedims(reshaped, perm))
end

"""
Run an SMT solver via the Rust backend runner.
"""
function rust_smt_run(kind::AbstractString, path::AbstractString, script::AbstractString, timeout_ms::Integer)
    if !rust_available()
        error("Rust backend not available")
    end

    ptr = @rust_call axiom_smt_run Ptr{UInt8} (Cstring, Cstring, Cstring, Cuint) kind path script UInt32(timeout_ms)
    if ptr == C_NULL
        return ""
    end

    output = unsafe_string(ptr)
    @rust_call axiom_smt_free Cvoid (Ptr{UInt8},) ptr
    output
end

# ============================================================================
# Rust Backend Operations
# ============================================================================

"""
Matrix multiplication via Rust.
"""
function backend_matmul(::RustBackend, A::Matrix{Float32}, B::Matrix{Float32})
    if !rust_available()
        # Fallback to Julia
        return A * B
    end

    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("Matrix dimensions must match: ($(size(A))) * ($(size(B)))"))

    A_row = _to_row_major_vec(A)
    B_row = _to_row_major_vec(B)
    C_row = Vector{Float32}(undef, m * n)

    # Call Rust function
    # axiom_matmul(A_ptr, B_ptr, C_ptr, m, k, n)
    @rust_call axiom_matmul Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t
    ) A_row B_row C_row m k n

    _from_row_major_vec(C_row, (m, n))
end

"""
ReLU via Rust.
"""
function backend_relu(::RustBackend, x::Array{Float32})
    if !rust_available()
        return relu(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_relu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Softmax via Rust.
"""
function backend_softmax(::RustBackend, x::Array{Float32}, dim::Int)
    if !rust_available()
        return softmax(x, dims=dim)
    end

    ndims(x) == 2 || return softmax(x, dims=dim)
    x_dims = (size(x, 1), size(x, 2))
    x_row = _to_row_major_vec(x)
    y_row = Vector{Float32}(undef, length(x_row))
    batch_size = size(x, 1)
    num_classes = size(x, 2)

    @rust_call axiom_softmax Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t
    ) x_row y_row batch_size num_classes

    _from_row_major_vec(y_row, x_dims)
end

"""
Conv2D via Rust.
"""
function backend_conv2d(
    ::RustBackend,
    input::Array{Float32, 4},
    weight::Array{Float32, 4},
    bias::Union{Vector{Float32}, Nothing},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int}
)
    if !rust_available()
        # Fallback to Julia
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

    @rust_call axiom_conv2d Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t, Csize_t,  # input dims
        Csize_t, Csize_t, Csize_t, Csize_t,  # kernel dims
        Csize_t, Csize_t,  # stride
        Csize_t, Csize_t   # padding
    ) input_row weight_row bias_ptr output_row N H_in W_in C_in kH kW C_in C_out sH sW pH pW

    _from_row_major_vec(output_row, (N, H_out, W_out, C_out))
end

"""
Sigmoid via Rust.
"""
function backend_sigmoid(::RustBackend, x::Array{Float32})
    if !rust_available()
        return sigmoid(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_sigmoid Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
GELU via Rust.
"""
function backend_gelu(::RustBackend, x::Array{Float32})
    if !rust_available()
        return gelu(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_gelu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Tanh via Rust.
"""
function backend_tanh(::RustBackend, x::Array{Float32})
    if !rust_available()
        return tanh.(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_tanh Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Leaky ReLU via Rust.
"""
function backend_leaky_relu(::RustBackend, x::Array{Float32}, alpha::Float32=0.01f0)
    if !rust_available()
        return leaky_relu(x, alpha)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_leaky_relu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Cfloat
    ) x y n alpha

    y
end

"""
ELU via Rust.
"""
function backend_elu(::RustBackend, x::Array{Float32}, alpha::Float32=1.0f0)
    if !rust_available()
        return elu(x, alpha)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_elu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Cfloat
    ) x y n alpha

    y
end

"""
SELU via Rust.
"""
function backend_selu(::RustBackend, x::Array{Float32})
    if !rust_available()
        return selu(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_selu Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Swish/SiLU via Rust.
"""
function backend_swish(::RustBackend, x::Array{Float32})
    if !rust_available()
        return swish(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_swish Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Mish via Rust.
"""
function backend_mish(::RustBackend, x::Array{Float32})
    if !rust_available()
        return mish(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_mish Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Hard Swish via Rust.
"""
function backend_hardswish(::RustBackend, x::Array{Float32})
    if !rust_available()
        return hardswish(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_hardswish Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Hard Sigmoid via Rust.
"""
function backend_hardsigmoid(::RustBackend, x::Array{Float32})
    if !rust_available()
        return hardsigmoid(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_hardsigmoid Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Log Softmax via Rust.
"""
function backend_log_softmax(::RustBackend, x::Array{Float32}, dim::Int)
    if !rust_available()
        return log_softmax(x, dims=dim)
    end

    y = similar(x)
    batch_size = size(x, 1)
    num_classes = size(x, 2)

    @rust_call axiom_log_softmax Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t
    ) x y batch_size num_classes

    y
end

"""
Softplus via Rust.
"""
function backend_softplus(::RustBackend, x::Array{Float32})
    if !rust_available()
        return softplus(x)
    end

    y = similar(x)
    n = length(x)

    @rust_call axiom_softplus Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t
    ) x y n

    y
end

"""
Batch normalization via Rust.
"""
function backend_batchnorm(
    ::RustBackend,
    x::Array{Float32},
    gamma::Vector{Float32},
    beta::Vector{Float32},
    running_mean::Vector{Float32},
    running_var::Vector{Float32},
    eps::Float32,
    training::Bool
)
    if !rust_available()
        return backend_batchnorm(JuliaBackend(), x, gamma, beta, running_mean, running_var, eps, training)
    end

    ndims(x) >= 2 || return backend_batchnorm(JuliaBackend(), x, gamma, beta, running_mean, running_var, eps, training)
    x_dims = Tuple(size(x))
    x_row = _to_row_major_vec(x)
    y_row = Vector{Float32}(undef, length(x_row))
    # Rust FFI currently takes mutable running stats pointers.
    # Pass copies to avoid unintended caller-side mutation.
    running_mean_buf = copy(running_mean)
    running_var_buf = copy(running_var)
    n_features = length(gamma)
    n_elements = length(x_row)

    @rust_call axiom_batchnorm Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Ptr{Float32}, Ptr{Float32},
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Cfloat, Cint
    ) x_row y_row gamma beta running_mean_buf running_var_buf n_elements n_features eps training

    _from_row_major_vec(y_row, x_dims)
end

"""
Layer normalization via Rust.
"""
function backend_layernorm(
    ::RustBackend,
    x::Array{Float32},
    gamma::Vector{Float32},
    beta::Vector{Float32},
    eps::Float32
)
    if !rust_available()
        return backend_layernorm(JuliaBackend(), x, gamma, beta, eps)
    end

    y = similar(x)
    batch_size = size(x, 1)
    hidden_size = size(x, 2)

    @rust_call axiom_layernorm Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Cfloat
    ) x y gamma beta batch_size hidden_size eps

    y
end

"""
RMS normalization via Rust.
"""
function backend_rmsnorm(
    ::RustBackend,
    x::Array{Float32},
    weight::Vector{Float32},
    eps::Float32
)
    if !rust_available()
        return backend_rmsnorm(JuliaBackend(), x, weight, eps)
    end

    y = similar(x)
    batch_size = size(x, 1)
    hidden_size = size(x, 2)

    @rust_call axiom_rmsnorm Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Cfloat
    ) x y weight batch_size hidden_size eps

    y
end

"""
MaxPool2D via Rust.
"""
function backend_maxpool2d(
    ::RustBackend,
    input::Array{Float32, 4},
    kernel_size::Tuple{Int, Int},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int}
)
    if !rust_available()
        return backend_maxpool2d(JuliaBackend(), input, kernel_size, stride, padding)
    end

    N, H_in, W_in, C = size(input)
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    H_out = div(H_in + 2*pH - kH, sH) + 1
    W_out = div(W_in + 2*pW - kW, sW) + 1

    output = Array{Float32}(undef, N, H_out, W_out, C)

    @rust_call axiom_maxpool2d Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t, Csize_t,  # input dims
        Csize_t, Csize_t,  # kernel size
        Csize_t, Csize_t,  # stride
        Csize_t, Csize_t   # padding
    ) input output N H_in W_in C kH kW sH sW pH pW

    output
end

"""
Global Average Pool2D via Rust.
"""
function backend_global_avgpool2d(::RustBackend, input::Array{Float32, 4})
    if !rust_available()
        return backend_global_avgpool2d(JuliaBackend(), input)
    end

    N, H, W, C = size(input)
    output = Array{Float32}(undef, N, C)

    @rust_call axiom_global_avgpool2d Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t, Csize_t
    ) input output N H W C

    output
end

# ============================================================================
# Rust Backend Utilities
# ============================================================================

"""
    benchmark_rust_vs_julia(op, args...; iterations=100)

Benchmark Rust vs Julia implementation.
"""
function benchmark_rust_vs_julia(op::Symbol, args...; iterations::Int=100)
    if !rust_available()
        @warn "Rust backend not available for benchmarking"
        return nothing
    end

    # Time Julia
    julia_time = @elapsed for _ in 1:iterations
        if op == :matmul
            backend_matmul(JuliaBackend(), args...)
        elseif op == :relu
            backend_relu(JuliaBackend(), args...)
        elseif op == :softmax
            backend_softmax(JuliaBackend(), args...)
        end
    end

    # Time Rust
    rust_time = @elapsed for _ in 1:iterations
        if op == :matmul
            backend_matmul(RustBackend(""), args...)
        elseif op == :relu
            backend_relu(RustBackend(""), args...)
        elseif op == :softmax
            backend_softmax(RustBackend(""), args...)
        end
    end

    speedup = julia_time / rust_time

    Dict(
        "julia_time" => julia_time / iterations,
        "rust_time" => rust_time / iterations,
        "speedup" => speedup
    )
end

"""
Get Rust backend version info.
"""
function rust_backend_info()
    if !rust_available()
        return "Rust backend not available"
    end

    # Get version string from Rust
    version_ptr = @rust_call axiom_version Cstring ()
    version = unsafe_string(version_ptr)

    Dict(
        "version" => version,
        "lib_path" => _rust_lib[],
        "available" => true
    )
end
