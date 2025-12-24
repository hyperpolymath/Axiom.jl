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
    _, n = size(B)
    C = Matrix{Float32}(undef, m, n)

    # Call Rust function
    # axiom_matmul(A_ptr, B_ptr, C_ptr, m, k, n)
    @rust_call axiom_matmul Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t
    ) A B C m k n

    C
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

    y = similar(x)
    batch_size = size(x, 1)
    num_classes = size(x, 2)

    @rust_call axiom_softmax Cvoid (
        Ptr{Float32}, Ptr{Float32}, Csize_t, Csize_t
    ) x y batch_size num_classes

    y
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

    output = Array{Float32}(undef, N, H_out, W_out, C_out)

    bias_ptr = bias === nothing ? C_NULL : pointer(bias)

    @rust_call axiom_conv2d Cvoid (
        Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Csize_t, Csize_t,  # input dims
        Csize_t, Csize_t, Csize_t, Csize_t,  # kernel dims
        Csize_t, Csize_t,  # stride
        Csize_t, Csize_t   # padding
    ) input weight bias_ptr output N H_in W_in C_in kH kW C_in C_out sH sW pH pW

    output
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

    y = similar(x)
    n_features = length(gamma)
    n_elements = length(x)

    @rust_call axiom_batchnorm Cvoid (
        Ptr{Float32}, Ptr{Float32},
        Ptr{Float32}, Ptr{Float32},
        Ptr{Float32}, Ptr{Float32},
        Csize_t, Csize_t, Cfloat, Cint
    ) x y gamma beta running_mean running_var n_elements n_features eps training

    y
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
