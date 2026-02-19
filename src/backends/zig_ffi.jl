# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Zig FFI
#
# Interface to high-performance Zig backend.

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
# Zig Backend Operations
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

    # Zig expects row-major, but let's assume our current FFI standard 
    # (Axiom.jl seems to use row-major for Rust, so likely for Zig too)
    A_row = _to_row_major_vec(A)
    B_row = _to_row_major_vec(B)
    C_row = Vector{Float32}(undef, m * n)

    @zig_call axiom_matmul Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t
    ) A_row B_row C_row m k n

    _from_row_major_vec(C_row, (m, n))
end

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

    # Zig axiom_conv2d signature:
    # (input, weight, bias, output, batch, h_in, w_in, c_in, h_out, w_out, c_out, kh, kw, sH, sW, pH, pW)
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
