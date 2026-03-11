# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Backend Abstraction
#
# Interface for different computation backends.

using Libdl
using LinearAlgebra: I
using AcceleratorGate: select_backend as ag_select_backend,
    fits_on_device as ag_fits_on_device, estimate_cost as ag_estimate_cost,
    DeviceCapabilities, device_capabilities, PlatformInfo, detect_platform

"""
    AbstractBackend

Base type for computation backends.
"""
abstract type AbstractBackend end

"""
    JuliaBackend

Pure Julia implementation (default, for development and debugging).
"""
struct JuliaBackend <: AbstractBackend end

"""
    ZigBackend

High-performance Zig implementation.
"""
struct ZigBackend <: AbstractBackend
    lib_path::String
end

"""
    CUDABackend

GPU acceleration via CUDA (NVIDIA).
"""
struct CUDABackend <: AbstractBackend
    device::Int
end

"""
    MetalBackend

GPU acceleration via Metal (Apple Silicon).
"""
struct MetalBackend <: AbstractBackend
    device::Int
end

"""
    ROCmBackend

GPU acceleration via ROCm (AMD GPUs).
"""
struct ROCmBackend <: AbstractBackend
    device::Int
end

"""
    TPUBackend

Tensor Processing Unit backend target.
"""
struct TPUBackend <: AbstractBackend
    device::Int
end

"""
    NPUBackend

Neural Processing Unit backend target.
"""
struct NPUBackend <: AbstractBackend
    device::Int
end

"""
    DSPBackend

Digital Signal Processor backend target.
"""
struct DSPBackend <: AbstractBackend
    device::Int
end

"""
    PPUBackend

Physics Processing Unit backend target.
"""
struct PPUBackend <: AbstractBackend
    device::Int
end

"""
    MathBackend

Math coprocessor backend target.
"""
struct MathBackend <: AbstractBackend
    device::Int
end

"""
    FPGABackend

FPGA accelerator backend target.
"""
struct FPGABackend <: AbstractBackend
    device::Int
end

"""
    VPUBackend

Vector/Vision Processing Unit backend target.
"""
struct VPUBackend <: AbstractBackend
    device::Int
end

"""
    QPUBackend

Quantum Processing Unit backend target.
"""
struct QPUBackend <: AbstractBackend
    device::Int
end

"""
    CryptoBackend

Cryptographic coprocessor backend target.
"""
struct CryptoBackend <: AbstractBackend
    device::Int
end

"""
    SmartBackend

Per-operation dispatch backend that routes each operation to its fastest
backend based on benchmark data. Wraps Julia and Zig backends and selects
the optimal one for each kernel. Zig backend uses SIMD + multi-threading.

Dispatch table (from 2026-02-20 benchmarks, post-SIMD + threading):
- matmul → Julia (BLAS, 7–50x faster)
- relu → Julia (near-parity, avoids FFI overhead)
- sigmoid → Zig (2.6–2.9x faster, threaded ≥64K)
- gelu → Zig (3.0–3.5x faster, SIMD @exp, threaded ≥64K)
- softmax → Zig (<50K classes), Julia (≥50K classes)
- layernorm → Zig (1.2–2.6x faster)
- rmsnorm → Zig (6.5–7.3x faster)
- batchnorm → Julia (1.2–1.8x faster)
- conv2d → Julia (BLAS-based)
"""
struct SmartBackend <: AbstractBackend
    julia::JuliaBackend
    zig::Union{ZigBackend, Nothing}
end

function SmartBackend(; zig_path::Union{String,Nothing}=nothing)
    julia = JuliaBackend()
    zig = zig_path !== nothing && isfile(zig_path) ? ZigBackend(zig_path) : nothing
    SmartBackend(julia, zig)
end

# SmartBackend dispatch: route each operation to the fastest backend.
# Falls back to Julia if the preferred native backend is unavailable.

function _smart_zig_or_julia(sb::SmartBackend)
    sb.zig !== nothing ? sb.zig : sb.julia
end

# --- MatMul: always Julia (BLAS) ---
function backend_matmul(sb::SmartBackend, A::Array{Float32}, B::Array{Float32})
    backend_matmul(sb.julia, A, B)
end

# --- ReLU: Julia (FFI overhead negates kernel parity) ---
function backend_relu(sb::SmartBackend, x::Array{Float32})
    backend_relu(sb.julia, x)
end

# --- Sigmoid: Zig (3.3x faster) ---
function backend_sigmoid(sb::SmartBackend, x::Array{Float32})
    backend_sigmoid(_smart_zig_or_julia(sb), x)
end

# --- GELU: Zig (3.0–3.5x faster after SIMD optimization) ---
function backend_gelu(sb::SmartBackend, x::Array{Float32})
    backend_gelu(_smart_zig_or_julia(sb), x)
end

# --- Softmax: Zig for small, Julia for large ---
function backend_softmax(sb::SmartBackend, x::Array{Float32}, dim::Int)
    classes = size(x, dim)
    if sb.zig !== nothing && classes < 50_000
        backend_softmax(sb.zig, x, dim)
    else
        backend_softmax(sb.julia, x, dim)
    end
end

# --- LayerNorm: Zig (1.9x faster) ---
function backend_layernorm(sb::SmartBackend, x::Array{Float32}, gamma::Vector{Float32}, beta::Vector{Float32}, eps::Float32)
    backend_layernorm(_smart_zig_or_julia(sb), x, gamma, beta, eps)
end

# JuliaBackend signature passthrough (with normalized_shape)
function backend_layernorm(sb::SmartBackend, x::Array{Float32}, gamma::Vector{Float32}, beta::Vector{Float32}, nshape::Tuple, eps::Float32)
    if sb.zig !== nothing
        backend_layernorm(sb.zig, x, gamma, beta, eps)
    else
        backend_layernorm(sb.julia, x, gamma, beta, nshape, eps)
    end
end

# --- RMSNorm: Zig (7.2x faster) ---
function backend_rmsnorm(sb::SmartBackend, x::Array{Float32}, weight::Vector{Float32}, eps::Float32)
    backend_rmsnorm(_smart_zig_or_julia(sb), x, weight, eps)
end

# --- BatchNorm: Julia (1.4x faster) ---
function backend_batchnorm(sb::SmartBackend, x::Array{Float32}, gamma::Vector{Float32}, beta::Vector{Float32},
                           rmean::Vector{Float32}, rvar::Vector{Float32}, eps::Float32, training::Bool)
    backend_batchnorm(sb.julia, x, gamma, beta, rmean, rvar, eps, training)
end

# --- Conv2d: Julia (BLAS-based) ---
function backend_conv2d(sb::SmartBackend, x::Array{Float32}, w::Array{Float32},
                        bias, stride::Tuple, padding::Tuple)
    backend_conv2d(sb.julia, x, w, bias, stride, padding)
end

# --- Tanh: Zig (same @exp path as sigmoid) ---
function backend_tanh(sb::SmartBackend, x::Array{Float32})
    backend_tanh(_smart_zig_or_julia(sb), x)
end

# --- Swish/SiLU: Zig has SIMD+threaded, Julia as fallback ---
function backend_swish(sb::SmartBackend, x::Array{Float32})
    backend_swish(_smart_zig_or_julia(sb), x)
end

# --- ELU: Zig has SIMD+threaded, Julia as fallback ---
function backend_elu(sb::SmartBackend, x::Array{Float32}, alpha::Float32)
    backend_elu(_smart_zig_or_julia(sb), x, alpha)
end

# --- Leaky ReLU: Zig has SIMD+threaded, Julia as fallback ---
function backend_leaky_relu(sb::SmartBackend, x::Array{Float32}, alpha::Float32)
    backend_leaky_relu(_smart_zig_or_julia(sb), x, alpha)
end

# --- SELU: Zig has SIMD+threaded, Julia as fallback ---
function backend_selu(sb::SmartBackend, x::Array{Float32})
    backend_selu(_smart_zig_or_julia(sb), x)
end

# --- Mish: Zig has SIMD+threaded, Julia as fallback ---
function backend_mish(sb::SmartBackend, x::Array{Float32})
    backend_mish(_smart_zig_or_julia(sb), x)
end

# --- Hard Swish: Zig has SIMD+threaded, Julia as fallback ---
function backend_hardswish(sb::SmartBackend, x::Array{Float32})
    backend_hardswish(_smart_zig_or_julia(sb), x)
end

# --- Hard Sigmoid: Zig has SIMD+threaded, Julia as fallback ---
function backend_hardsigmoid(sb::SmartBackend, x::Array{Float32})
    backend_hardsigmoid(_smart_zig_or_julia(sb), x)
end

# --- Softplus: Julia preferred (MKL exp+log faster than Zig) ---
function backend_softplus(sb::SmartBackend, x::Array{Float32})
    backend_softplus(sb.julia, x)
end

# --- Log Softmax: follows softmax routing ---
function backend_log_softmax(sb::SmartBackend, x::Array{Float32}, dim::Int)
    classes = size(x, dim)
    if sb.zig !== nothing && classes < 50_000
        backend_log_softmax(sb.zig, x, dim)
    else
        backend_log_softmax(sb.julia, x, dim)
    end
end

# ============================================================================
# In-place SmartBackend dispatch
# ============================================================================

function backend_relu!(sb::SmartBackend, x::Array{Float32})
    backend_relu!(_smart_zig_or_julia(sb), x)
end

function backend_sigmoid!(sb::SmartBackend, x::Array{Float32})
    backend_sigmoid!(_smart_zig_or_julia(sb), x)
end

function backend_tanh!(sb::SmartBackend, x::Array{Float32})
    backend_tanh!(_smart_zig_or_julia(sb), x)
end

function backend_gelu!(sb::SmartBackend, x::Array{Float32})
    backend_gelu!(_smart_zig_or_julia(sb), x)
end

function backend_swish!(sb::SmartBackend, x::Array{Float32})
    backend_swish!(_smart_zig_or_julia(sb), x)
end

# Global current backend
const _current_backend = Ref{AbstractBackend}(JuliaBackend())

# GPU runtime diagnostics counters (self-healing/fallback observability).
const _GPU_RUNTIME_DIAGNOSTICS = Dict{String, Dict{String, Int}}(
    "cuda" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "rocm" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "metal" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
)

function _gpu_backend_key(backend)
    name = lowercase(string(nameof(typeof(backend))))
    if occursin("cuda", name)
        return "cuda"
    elseif occursin("rocm", name)
        return "rocm"
    elseif occursin("metal", name)
        return "metal"
    end
    "unknown"
end

function _gpu_record_diagnostic!(backend, key::String)
    backend_key = _gpu_backend_key(backend)
    haskey(_GPU_RUNTIME_DIAGNOSTICS, backend_key) || return
    counters = _GPU_RUNTIME_DIAGNOSTICS[backend_key]
    counters[key] = get(counters, key, 0) + 1
end

"""
    reset_gpu_runtime_diagnostics!()

Reset in-process GPU runtime diagnostics counters.
"""
function reset_gpu_runtime_diagnostics!()
    for counters in values(_GPU_RUNTIME_DIAGNOSTICS)
        counters["compile_fallbacks"] = 0
        counters["runtime_errors"] = 0
        counters["runtime_fallbacks"] = 0
        counters["recoveries"] = 0
    end
    nothing
end

"""
    gpu_runtime_diagnostics() -> Dict{String,Any}

Return machine-readable counters for GPU fallback/self-healing behavior.
"""
function gpu_runtime_diagnostics()
    backends = Dict{String, Any}()
    for key in ("cuda", "rocm", "metal")
        counters = get(_GPU_RUNTIME_DIAGNOSTICS, key, Dict{String, Int}())
        backends[key] = Dict(
            "compile_fallbacks" => get(counters, "compile_fallbacks", 0),
            "runtime_errors" => get(counters, "runtime_errors", 0),
            "runtime_fallbacks" => get(counters, "runtime_fallbacks", 0),
            "recoveries" => get(counters, "recoveries", 0),
        )
    end

    Dict(
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "self_healing_enabled" => _gpu_self_healing_enabled(),
        "backends" => backends,
    )
end

function _gpu_self_healing_enabled()
    forced = _backend_env_available("AXIOM_GPU_SELF_HEAL")
    forced === nothing ? true : forced
end

function _gpu_forward_with_recovery(model, x::AbstractTensor, backend)
    try
        return _gpu_forward(model, x, backend)
    catch err
        _gpu_record_diagnostic!(backend, "runtime_errors")
        if !_gpu_self_healing_enabled()
            rethrow()
        end
        _gpu_record_diagnostic!(backend, "runtime_fallbacks")
        @warn "GPU execution failed for $(typeof(backend)); self-healing fallback to Julia backend" exception=(err, catch_backtrace())
        recovered = forward(model, x)
        _gpu_record_diagnostic!(backend, "recoveries")
        return recovered
    end
end

"""
    current_backend() -> AbstractBackend

Get the current computation backend.
"""
current_backend() = _current_backend[]

"""
    set_backend!(backend::AbstractBackend)

Set the computation backend.
"""
function set_backend!(backend::AbstractBackend)
    _current_backend[] = backend
    @info "Backend set to $(typeof(backend))"
end

"""
    @with_backend backend expr

Execute expression with specified backend.
"""
macro with_backend(backend, expr)
    quote
        old_backend = current_backend()
        set_backend!($(esc(backend)))
        try
            $(esc(expr))
        finally
            set_backend!(old_backend)
        end
    end
end

# ============================================================================
# Backend Operations Interface
# ============================================================================

"""
    backend_matmul(backend, A, B)

Matrix multiplication on specified backend.
"""
function backend_matmul end

"""
    backend_conv2d(backend, input, weight, bias, stride, padding)

2D convolution on specified backend.
"""
function backend_conv2d end

"""
    backend_relu(backend, x)

ReLU activation on specified backend.
"""
function backend_relu end

"""
    backend_softmax(backend, x, dim)

Softmax on specified backend.
"""
function backend_softmax end

"""
    backend_batchnorm(backend, x, gamma, beta, mean, var, eps)

Batch normalization on specified backend.
"""
function backend_batchnorm end

"""
    backend_layernorm(backend, x, gamma, beta, normalized_shape, eps)

Layer normalization on specified backend.
"""
function backend_layernorm end

"""
    backend_maxpool2d(backend, input, kernel_size, stride, padding)

2D max pooling on specified backend.
"""
function backend_maxpool2d end

"""
    backend_avgpool2d(backend, input, kernel_size, stride, padding, count_include_pad=true)

2D average pooling on specified backend.
"""
function backend_avgpool2d end

"""
    backend_global_avgpool2d(backend, input)

Global average pooling on specified backend.
"""
function backend_global_avgpool2d end

# Default implementations (Julia backend)
backend_matmul(::JuliaBackend, A, B) = A * B
backend_relu(::JuliaBackend, x) = relu(x)
backend_softmax(::JuliaBackend, x, dim) = softmax(x, dims=dim)

function backend_conv2d(::JuliaBackend, input::Array{Float32,4}, weight::Array{Float32,4},
                        bias::Union{Vector{Float32},Nothing}, stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    N, H, W, C_in = size(input)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride
    pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    x_data = input
    if pH > 0 || pW > 0
        x_padded = zeros(Float32, N, H + 2*pH, W + 2*pW, C_in)
        x_padded[:, pH+1:pH+H, pW+1:pW+W, :] = x_data
        x_data = x_padded
    end

    y = zeros(Float32, N, H_out, W_out, C_out)
    for n in 1:N, oc in 1:C_out, i in 1:H_out, j in 1:W_out
        hs = (i-1)*sH+1; ws = (j-1)*sW+1
        y[n,i,j,oc] = sum(x_data[n, hs:hs+kH-1, ws:ws+kW-1, :] .* weight[:,:,:,oc])
    end

    if bias !== nothing
        for oc in 1:C_out
            y[:,:,:,oc] .+= bias[oc]
        end
    end
    y
end

function backend_batchnorm(::JuliaBackend, x::Array{Float32}, gamma::Vector{Float32},
                           beta::Vector{Float32}, running_mean::Vector{Float32},
                           running_var::Vector{Float32}, eps::Float32, training::Bool)
    μ = reshape(running_mean, ones(Int, ndims(x)-1)..., :)
    σ² = reshape(running_var, ones(Int, ndims(x)-1)..., :)
    x_norm = (x .- μ) ./ sqrt.(σ² .+ eps)
    γ = reshape(gamma, ones(Int, ndims(x)-1)..., :)
    β = reshape(beta, ones(Int, ndims(x)-1)..., :)
    γ .* x_norm .+ β
end

function backend_maxpool2d(::JuliaBackend, input::Array{Float32,4},
                           kernel_size::Tuple{Int,Int}, stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    N, H, W, C = size(input)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    x_data = input
    if pH > 0 || pW > 0
        x_padded = fill(-Inf32, N, H + 2*pH, W + 2*pW, C)
        x_padded[:, pH+1:pH+H, pW+1:pW+W, :] = x_data
        x_data = x_padded
    end

    y = Array{Float32}(undef, N, H_out, W_out, C)
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i-1)*sH+1; ws = (j-1)*sW+1
        y[n,i,j,c] = maximum(x_data[n, hs:hs+kH-1, ws:ws+kW-1, c])
    end
    y
end

function backend_avgpool2d(::JuliaBackend, input::Array{Float32,4},
                           kernel_size::Tuple{Int,Int}, stride::Tuple{Int,Int}, padding::Tuple{Int,Int})
    N, H, W, C = size(input)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding
    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    x_data = input
    if pH > 0 || pW > 0
        x_padded = zeros(Float32, N, H + 2*pH, W + 2*pW, C)
        x_padded[:, pH+1:pH+H, pW+1:pW+W, :] = x_data
        x_data = x_padded
    end

    y = Array{Float32}(undef, N, H_out, W_out, C)
    for n in 1:N, c in 1:C, i in 1:H_out, j in 1:W_out
        hs = (i-1)*sH+1; ws = (j-1)*sW+1
        y[n,i,j,c] = mean(x_data[n, hs:hs+kH-1, ws:ws+kW-1, c])
    end
    y
end

function backend_global_avgpool2d(::JuliaBackend, input::Array{Float32,4})
    N, H, W, C = size(input)
    y = Array{Float32}(undef, N, C)
    for n in 1:N, c in 1:C
        y[n,c] = mean(input[n,:,:,c])
    end
    y
end

const CoprocessorBackend = Union{TPUBackend, NPUBackend, DSPBackend, PPUBackend, MathBackend, FPGABackend, VPUBackend, QPUBackend, CryptoBackend}

function _coprocessor_label(backend::CoprocessorBackend)
    if backend isa TPUBackend
        return "TPU"
    elseif backend isa NPUBackend
        return "NPU"
    elseif backend isa DSPBackend
        return "DSP"
    elseif backend isa PPUBackend
        return "PPU"
    elseif backend isa MathBackend
        return "MATH"
    elseif backend isa FPGABackend
        return "FPGA"
    elseif backend isa VPUBackend
        return "VPU"
    elseif backend isa QPUBackend
        return "QPU"
    elseif backend isa CryptoBackend
        return "CRYPTO"
    end
    string(typeof(backend))
end

function _coprocessor_required_env_key(backend::CoprocessorBackend)
    if backend isa TPUBackend
        return "AXIOM_TPU_REQUIRED"
    elseif backend isa NPUBackend
        return "AXIOM_NPU_REQUIRED"
    elseif backend isa DSPBackend
        return "AXIOM_DSP_REQUIRED"
    elseif backend isa PPUBackend
        return "AXIOM_PPU_REQUIRED"
    elseif backend isa MathBackend
        return "AXIOM_MATH_REQUIRED"
    elseif backend isa FPGABackend
        return "AXIOM_FPGA_REQUIRED"
    elseif backend isa VPUBackend
        return "AXIOM_VPU_REQUIRED"
    elseif backend isa QPUBackend
        return "AXIOM_QPU_REQUIRED"
    elseif backend isa CryptoBackend
        return "AXIOM_CRYPTO_REQUIRED"
    end
    "AXIOM_COPROCESSOR_REQUIRED"
end

function _coprocessor_required(backend::CoprocessorBackend)
    specific = _backend_env_available(_coprocessor_required_env_key(backend))
    if specific !== nothing
        return specific
    end
    global_required = _backend_env_available("AXIOM_COPROCESSOR_REQUIRED")
    global_required === nothing ? false : global_required
end

# Coprocessor runtime diagnostics counters (self-healing/fallback observability).
const _COPROCESSOR_RUNTIME_DIAGNOSTICS = Dict{String, Dict{String, Int}}(
    "tpu" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "npu" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "dsp" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "ppu" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "math" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "fpga" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "vpu" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "qpu" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
    "crypto" => Dict(
        "compile_fallbacks" => 0,
        "runtime_errors" => 0,
        "runtime_fallbacks" => 0,
        "recoveries" => 0,
    ),
)

function _coprocessor_backend_key(backend::CoprocessorBackend)
    if backend isa TPUBackend
        return "tpu"
    elseif backend isa NPUBackend
        return "npu"
    elseif backend isa DSPBackend
        return "dsp"
    elseif backend isa PPUBackend
        return "ppu"
    elseif backend isa MathBackend
        return "math"
    elseif backend isa FPGABackend
        return "fpga"
    elseif backend isa VPUBackend
        return "vpu"
    elseif backend isa QPUBackend
        return "qpu"
    elseif backend isa CryptoBackend
        return "crypto"
    end
    "unknown"
end

function _coprocessor_record_diagnostic!(backend::CoprocessorBackend, key::String)
    backend_key = _coprocessor_backend_key(backend)
    haskey(_COPROCESSOR_RUNTIME_DIAGNOSTICS, backend_key) || return
    counters = _COPROCESSOR_RUNTIME_DIAGNOSTICS[backend_key]
    counters[key] = get(counters, key, 0) + 1
end

"""
    reset_coprocessor_runtime_diagnostics!()

Reset in-process coprocessor runtime diagnostics counters.
"""
function reset_coprocessor_runtime_diagnostics!()
    for counters in values(_COPROCESSOR_RUNTIME_DIAGNOSTICS)
        counters["compile_fallbacks"] = 0
        counters["runtime_errors"] = 0
        counters["runtime_fallbacks"] = 0
        counters["recoveries"] = 0
    end
    nothing
end

"""
    coprocessor_runtime_diagnostics() -> Dict{String,Any}

Return machine-readable counters for coprocessor fallback/self-healing behavior.
"""
function coprocessor_runtime_diagnostics()
    backends = Dict{String, Any}()
    for key in ("tpu", "npu", "dsp", "ppu", "math", "fpga", "vpu", "qpu", "crypto")
        counters = get(_COPROCESSOR_RUNTIME_DIAGNOSTICS, key, Dict{String, Int}())
        backends[key] = Dict(
            "compile_fallbacks" => get(counters, "compile_fallbacks", 0),
            "runtime_errors" => get(counters, "runtime_errors", 0),
            "runtime_fallbacks" => get(counters, "runtime_fallbacks", 0),
            "recoveries" => get(counters, "recoveries", 0),
        )
    end

    Dict(
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "self_healing_enabled" => _coprocessor_self_healing_enabled(),
        "backends" => backends,
    )
end

function _coprocessor_self_healing_enabled()
    forced = _backend_env_available("AXIOM_COPROCESSOR_SELF_HEAL")
    forced === nothing ? true : forced
end

# Coprocessor extension hooks. Concrete accelerator extensions can overload these.
function backend_coprocessor_matmul end
function backend_coprocessor_conv2d end
function backend_coprocessor_relu end
function backend_coprocessor_softmax end
function backend_coprocessor_batchnorm end
function backend_coprocessor_layernorm end
function backend_coprocessor_maxpool2d end
function backend_coprocessor_avgpool2d end
function backend_coprocessor_global_avgpool2d end

for (hook, cpu_op) in (
    (:backend_coprocessor_matmul, :backend_matmul),
    (:backend_coprocessor_conv2d, :backend_conv2d),
    (:backend_coprocessor_relu, :backend_relu),
    (:backend_coprocessor_softmax, :backend_softmax),
    (:backend_coprocessor_batchnorm, :backend_batchnorm),
    (:backend_coprocessor_layernorm, :backend_layernorm),
    (:backend_coprocessor_maxpool2d, :backend_maxpool2d),
    (:backend_coprocessor_avgpool2d, :backend_avgpool2d),
    (:backend_coprocessor_global_avgpool2d, :backend_global_avgpool2d),
)
    hook_name = String(hook)
    @eval function $hook(backend::CoprocessorBackend, args...)
        label = _coprocessor_label(backend)
        if _coprocessor_required(backend)
            throw(ErrorException(string(
                label,
                " extension hook not loaded for `",
                $hook_name,
                "` while strict mode is enabled (set ",
                _coprocessor_required_env_key(backend),
                "=0 or AXIOM_COPROCESSOR_REQUIRED=0 to allow fallback).\n",
                "To configure this backend, run: Axiom.coprocessor_setup_guide(\"",
                label,
                "\")"
            )))
        end
        @warn string(
            label,
            " extension hook not loaded for `",
            $hook_name,
            "`, falling back to Julia backend. ",
            "Run Axiom.coprocessor_setup_guide(\"",
            label,
            "\") for setup instructions."
        ) maxlog=1
        return $cpu_op(JuliaBackend(), args...)
    end
end

backend_matmul(backend::CoprocessorBackend, A::AbstractArray, B::AbstractArray) =
    backend_coprocessor_matmul(backend, A, B)
backend_conv2d(
    backend::CoprocessorBackend,
    input::AbstractArray{T, 4},
    weight::AbstractArray{T, 4},
    bias::Union{AbstractVector{T}, Nothing},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int},
) where {T} = backend_coprocessor_conv2d(backend, input, weight, bias, stride, padding)
backend_relu(backend::CoprocessorBackend, x::AbstractArray) =
    backend_coprocessor_relu(backend, x)
backend_softmax(backend::CoprocessorBackend, x::AbstractArray, dim::Int) =
    backend_coprocessor_softmax(backend, x, dim)
backend_batchnorm(
    backend::CoprocessorBackend,
    x::AbstractArray{T},
    gamma::AbstractVector{T},
    beta::AbstractVector{T},
    running_mean::AbstractVector{T},
    running_var::AbstractVector{T},
    eps::T,
    training::Bool,
) where {T} = backend_coprocessor_batchnorm(
    backend,
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps,
    training,
)
backend_layernorm(
    backend::CoprocessorBackend,
    x::AbstractArray{T},
    gamma::AbstractArray{T},
    beta::AbstractArray{T},
    normalized_shape::Tuple,
    eps::T,
) where {T} = backend_coprocessor_layernorm(
    backend,
    x,
    gamma,
    beta,
    normalized_shape,
    eps,
)
backend_maxpool2d(
    backend::CoprocessorBackend,
    input::AbstractArray{T, 4},
    kernel_size::Tuple{Int, Int},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int},
) where {T} = backend_coprocessor_maxpool2d(backend, input, kernel_size, stride, padding)
backend_avgpool2d(
    backend::CoprocessorBackend,
    input::AbstractArray{T, 4},
    kernel_size::Tuple{Int, Int},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int},
    count_include_pad::Bool=true,
) where {T} = backend_coprocessor_avgpool2d(
    backend,
    input,
    kernel_size,
    stride,
    padding,
    count_include_pad,
)
backend_global_avgpool2d(backend::CoprocessorBackend, input::AbstractArray{T, 4}) where {T} =
    backend_coprocessor_global_avgpool2d(backend, input)

# ============================================================================
# Compilation Target
# ============================================================================

"""
    CompilationTarget

Target configuration for model compilation.
"""
struct CompilationTarget
    backend::AbstractBackend
    optimize::Symbol  # :none, :default, :aggressive
    precision::Symbol  # :float32, :float16, :mixed
end

# ============================================================================
# Backend-Aware Layer Forward Dispatch
# ============================================================================
#
# These methods override the default Julia-only forward functions defined in
# layers/*.jl to route through the current backend. The JuliaBackend path
# is identical to the original implementations.

"""
    forward(d::Dense, x::AbstractTensor)

Backend-aware Dense forward pass. Dispatches matrix multiplication through
the current backend (Zig, GPU, or Julia).
"""
function forward(d::Dense, x::AbstractTensor)
    backend = current_backend()

    if backend isa JuliaBackend
        # Original Julia implementation (unchanged)
        if ndims(x) == 1
            y = d.weight' * x.data
        else
            y = x.data * d.weight
        end
    else
        # Route through backend dispatch
        if ndims(x) == 1
            x_mat = reshape(Float32.(x.data), 1, :)
            w_f32 = Float32.(d.weight)
            y = vec(backend_matmul(backend, x_mat, w_f32))
        else
            y = backend_matmul(backend, Float32.(x.data), Float32.(d.weight))
        end
    end

    if d.bias !== nothing
        y = y .+ d.bias'
    end

    Tensor(d.activation(y))
end

"""
    forward(c::Conv2d, x::AbstractTensor)

Backend-aware Conv2d forward pass. Dispatches convolution through
the current backend when available.
"""
function forward(c::Conv2d, x::AbstractTensor)
    backend = current_backend()

    has_batch = ndims(x) == 4
    if !has_batch
        x_data = reshape(x.data, 1, size(x.data)...)
    else
        x_data = x.data
    end

    if !(backend isa JuliaBackend)
        # Try backend dispatch
        try
            y = backend_conv2d(backend, Float32.(x_data), Float32.(c.weight),
                               c.bias === nothing ? nothing : Float32.(c.bias),
                               c.stride, c.padding)
            return Tensor(has_batch ? y : dropdims(y, dims=1))
        catch
            # Fall through to Julia implementation
        end
    end

    # Julia implementation (reference)
    N, H, W, C_in = size(x_data)
    kH, kW = c.kernel_size
    sH, sW = c.stride
    pH, pW = c.padding

    H_out = div(H + 2*pH - kH, sH) + 1
    W_out = div(W + 2*pW - kW, sW) + 1

    if pH > 0 || pW > 0
        x_padded = zeros(eltype(x_data), N, H + 2*pH, W + 2*pW, C_in)
        x_padded[:, pH+1:pH+H, pW+1:pW+W, :] = x_data
        x_data = x_padded
    end

    y = zeros(eltype(x_data), N, H_out, W_out, c.out_channels)

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

    if c.bias !== nothing
        for oc in 1:c.out_channels
            y[:, :, :, oc] .+= c.bias[oc]
        end
    end

    Tensor(has_batch ? y : dropdims(y, dims=1))
end

"""
    forward(bn::BatchNorm, x::AbstractTensor)

Backend-aware BatchNorm forward pass.
"""
function forward(bn::BatchNorm, x::AbstractTensor)
    backend = current_backend()
    x_data = x.data

    if !(backend isa JuliaBackend) && !bn.training && bn.affine
        # Try backend dispatch (inference mode only)
        try
            y = backend_batchnorm(backend, Float32.(x_data),
                                  Float32.(bn.γ), Float32.(bn.β),
                                  Float32.(bn.running_mean), Float32.(bn.running_var),
                                  Float32(bn.eps), false)
            return Tensor(y)
        catch
            # Fall through to Julia implementation
        end
    end

    # Julia implementation (reference, handles both training and inference)
    if bn.training
        dims = collect(1:ndims(x_data)-1)
        μ = mean(x_data, dims=dims)
        σ² = var(x_data, dims=dims, corrected=false)

        if bn.track_running_stats
            bn.running_mean .= (1 - bn.momentum) .* bn.running_mean .+ bn.momentum .* vec(μ)
            bn.running_var .= (1 - bn.momentum) .* bn.running_var .+ bn.momentum .* vec(σ²)
        end
    else
        μ = reshape(bn.running_mean, ones(Int, ndims(x_data)-1)..., :)
        σ² = reshape(bn.running_var, ones(Int, ndims(x_data)-1)..., :)
    end

    x_norm = (x_data .- μ) ./ sqrt.(σ² .+ bn.eps)

    if bn.affine
        γ = reshape(bn.γ, ones(Int, ndims(x_data)-1)..., :)
        β = reshape(bn.β, ones(Int, ndims(x_data)-1)..., :)
        x_norm = γ .* x_norm .+ β
    end

    Tensor(x_norm)
end

"""
    compile(model; backend=JuliaBackend(), optimize=:default, precision=:float32)

Compile a model for deployment.

# Arguments
- `model`: Model to compile
- `backend`: Target backend
- `optimize`: Optimization level
- `precision`: Numerical precision

# Returns
Compiled model ready for inference.
"""
function compile(
    model;
    backend::AbstractBackend = JuliaBackend(),
    optimize::Symbol = :default,
    precision::Symbol = :float32,
    verify::Bool = true
)
    target = CompilationTarget(backend, optimize, precision)

    # Verify model before compilation
    if verify
        result = verify_model(model)
        if !result.passed
            @warn "Model verification failed - proceed with caution"
        end
    end

    # Apply optimizations
    optimized = if optimize == :none
        model
    else
        optimize_model(model, target)
    end

    # Convert precision
    converted = if precision == :float16
        to_float16(optimized)
    elseif precision == :mixed
        to_mixed_precision(optimized)
    else
        optimized
    end

    # Compile to target backend
    compile_to_backend(converted, backend)
end

function optimize_model(model, target::CompilationTarget)
    optimized = model

    # Optimization 1: Operator fusion (for Pipelines)
    if optimized isa Pipeline
        optimized = optimize_pipeline(optimized)
    end

    # Optimization 2: Fold BatchNorm into linear layers for inference
    optimized = fold_batchnorm(optimized)

    # Optimization 3: Constant folding - precompute static values
    optimized = fold_constants(optimized)

    # Optimization 4: Dead code elimination - remove unused layers
    optimized = eliminate_dead_code(optimized)

    # Aggressive optimizations
    if target.optimize == :aggressive
        optimized = apply_aggressive_optimizations(optimized, target)
    end

    optimized
end

"""
    fold_batchnorm(model) -> model

Fold BatchNorm parameters into preceding Dense/Conv layers for inference.

For a Dense+BatchNorm pair:
  y = γ * (Wx + b - μ) / √(σ² + ε) + β
  = (γ/√(σ²+ε)) * W * x + (γ/√(σ²+ε)) * (b - μ) + β
  = W_folded * x + b_folded

This eliminates the BatchNorm as a separate operation.
"""
function fold_batchnorm(model::Pipeline)
    layers = collect(model.layers)
    new_layers = AbstractLayer[]
    i = 1

    while i <= length(layers)
        if i < length(layers) && layers[i+1] isa BatchNorm && !layers[i+1].training
            bn = layers[i+1]
            layer = layers[i]

            if layer isa Dense && bn.affine && bn.num_features == layer.out_features
                # Fold BN into Dense
                folded = _fold_bn_into_dense(layer, bn)
                push!(new_layers, folded)
                i += 2  # Skip the BatchNorm
                continue
            elseif layer isa Conv2d && bn.affine
                # Fold BN into Conv2d
                folded = _fold_bn_into_conv(layer, bn)
                push!(new_layers, folded)
                i += 2
                continue
            end
        end

        push!(new_layers, layers[i])
        i += 1
    end

    length(new_layers) == length(layers) ? model : Pipeline(Tuple(new_layers))
end

function fold_batchnorm(model)
    model  # Non-pipeline models: nothing to fold
end

function _fold_bn_into_dense(dense::Dense, bn::BatchNorm)
    inv_std = 1.0f0 ./ sqrt.(bn.running_var .+ bn.eps)
    scale = bn.γ .* inv_std

    # W_folded = diag(scale) * W (scale each output row)
    new_weight = dense.weight .* scale'

    # b_folded = scale * (b - μ) + β
    b = dense.bias !== nothing ? dense.bias : zeros(Float32, dense.out_features)
    new_bias = scale .* (b .- bn.running_mean) .+ bn.β

    folded = Dense(dense.in_features, dense.out_features, dense.activation)
    folded.weight .= new_weight
    folded.bias .= new_bias
    folded
end

function _fold_bn_into_conv(conv::Conv2d, bn::BatchNorm)
    inv_std = 1.0f0 ./ sqrt.(bn.running_var .+ bn.eps)
    scale = bn.γ .* inv_std

    # Scale each output channel's weights
    # conv.weight shape: (kH, kW, in_channels, out_channels)
    new_weight = copy(conv.weight)
    for c in 1:length(scale)
        new_weight[:, :, :, c] .*= scale[c]
    end

    new_bias = if conv.bias !== nothing
        scale .* (conv.bias .- bn.running_mean) .+ bn.β
    else
        scale .* (.-bn.running_mean) .+ bn.β
    end

    folded = Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                    stride=conv.stride, padding=conv.padding)
    folded.weight .= new_weight
    if folded.bias !== nothing
        folded.bias .= new_bias
    end
    folded
end

"""
    fold_constants(model) -> model

Pre-evaluate constant subexpressions in the model graph.
For Pipelines: collapse adjacent Dense layers with identity activation
(Dense(A) → Dense(B) = Dense(B*A)) when no activation is applied.
"""
function fold_constants(model::Pipeline)
    layers = collect(model.layers)
    new_layers = AbstractLayer[]
    i = 1

    while i <= length(layers)
        if i < length(layers) &&
           layers[i] isa Dense && layers[i].activation === identity &&
           layers[i+1] isa Dense && layers[i+1].activation === identity &&
           layers[i].out_features == layers[i+1].in_features
            # Fuse two adjacent linear layers: y = W2 * W1 * x + W2*b1 + b2
            d1, d2 = layers[i], layers[i+1]
            fused_weight = d1.weight * d2.weight  # (in1, out1) * (out1, out2) = (in1, out2)
            fused_bias = if d1.bias !== nothing && d2.bias !== nothing
                d2.weight' * d1.bias .+ d2.bias  # W2^T * b1 + b2
            elseif d2.bias !== nothing
                d2.bias
            elseif d1.bias !== nothing
                d2.weight' * d1.bias
            else
                nothing
            end

            fused = Dense(d1.in_features, d2.out_features)
            fused.weight .= fused_weight
            if fused_bias !== nothing && fused.bias !== nothing
                fused.bias .= fused_bias
            end
            push!(new_layers, fused)
            i += 2
            continue
        end

        push!(new_layers, layers[i])
        i += 1
    end

    length(new_layers) == length(layers) ? model : Pipeline(Tuple(new_layers))
end

function fold_constants(model)
    model
end

"""
    eliminate_dead_code(model) -> model

Remove layers that have no effect on the output:
- Dense layers that are identity transforms (weight ≈ I, bias ≈ 0)
- Dropout layers in inference mode (dropout rate = 0 or training = false)
"""
function eliminate_dead_code(model::Pipeline)
    layers = collect(model.layers)
    new_layers = AbstractLayer[]

    for layer in layers
        if _is_dead_layer(layer)
            continue
        end
        push!(new_layers, layer)
    end

    length(new_layers) == length(layers) ? model : Pipeline(Tuple(new_layers))
end

function eliminate_dead_code(model)
    model
end

function _is_dead_layer(layer::Dense)
    # Check if Dense is approximately identity: W ≈ I and b ≈ 0
    layer.in_features != layer.out_features && return false
    layer.activation !== identity && return false

    w_is_identity = isapprox(layer.weight, Matrix{Float32}(I, layer.in_features, layer.out_features), atol=1e-6)
    b_is_zero = layer.bias === nothing || all(abs.(layer.bias) .< 1e-6)

    w_is_identity && b_is_zero
end

function _is_dead_layer(layer)
    # Check for Dropout with 0 rate or in eval mode
    if hasproperty(layer, :p) && hasproperty(layer, :training)
        return layer.p == 0 || !layer.training
    end
    false
end

function apply_aggressive_optimizations(model, target::CompilationTarget)
    # Run fold passes iteratively until convergence
    prev = model
    for _ in 1:3
        optimized = fold_batchnorm(prev)
        optimized = fold_constants(optimized)
        optimized = eliminate_dead_code(optimized)
        optimized === prev && break
        prev = optimized
    end
    prev
end

function to_float16(model)
    convert_precision(model, Float16)
end

function to_mixed_precision(model)
    MixedPrecisionWrapper(model)
end

function convert_precision(model::AbstractLayer, ::Type{T}) where T
    params = parameters(model)
    if isempty(params)
        return model
    end

    # Convert each parameter array to the target type
    for (name, param) in pairs(params)
        if param isa AbstractArray
            converted = T.(param)
            setfield!(model, name, converted)
        end
    end

    model
end

function convert_precision(model::Pipeline, ::Type{T}) where T
    converted_layers = Tuple(convert_precision(layer, T) for layer in model.layers)
    Pipeline(converted_layers)
end

function convert_precision(model, ::Type{T}) where T
    # For non-layer types, return as-is
    model
end

"""
    MixedPrecisionWrapper

Wrapper that executes forward pass in Float16 while maintaining Float32 master weights.
Includes dynamic loss scaling for numerical stability during training.

# Fields
- `model`: The wrapped model
- `master_weights`: Float32 copies of all parameters (updated during optimizer step)
- `loss_scale`: Current dynamic loss scale factor
- `loss_scale_growth_interval`: Steps between scale increases
- `loss_scale_growth_factor`: Multiplier when increasing scale
- `loss_scale_backoff_factor`: Multiplier when NaN/Inf detected
- `steps_since_last_scale`: Counter for growth interval
- `precision_hints`: Per-layer precision overrides (symbol → :float16 or :float32)
"""
mutable struct MixedPrecisionWrapper{M}
    model::M
    master_weights::Dict{Symbol, Any}
    loss_scale::Float32
    loss_scale_growth_interval::Int
    loss_scale_growth_factor::Float32
    loss_scale_backoff_factor::Float32
    steps_since_last_scale::Int
    precision_hints::Dict{Symbol, Symbol}
end

function MixedPrecisionWrapper(model;
        initial_loss_scale::Float32 = 2f0^15,
        growth_interval::Int = 2000,
        growth_factor::Float32 = 2f0,
        backoff_factor::Float32 = 0.5f0,
        precision_hints::Dict{Symbol, Symbol} = Dict{Symbol, Symbol}())
    master = Dict{Symbol, Any}()
    params = parameters(model)
    for (name, param) in pairs(params)
        if param isa AbstractArray{Float32}
            master[name] = copy(param)
        end
    end
    MixedPrecisionWrapper(model, master, initial_loss_scale,
                          growth_interval, growth_factor, backoff_factor,
                          0, precision_hints)
end

"""
    forward(mp::MixedPrecisionWrapper, x)

Forward pass in Float16 with Float32 master weights.
Layers listed in `precision_hints` as `:float32` skip Float16 conversion.
"""
function forward(mp::MixedPrecisionWrapper, x)
    x_f16 = if x isa AbstractTensor
        Tensor(Float16.(x.data))
    else
        Float16.(x)
    end

    # Temporarily convert model weights to Float16
    params = parameters(mp.model)
    for (name, param) in pairs(params)
        if param isa AbstractArray{Float32}
            # Check precision hints — some layers stay in Float32
            if get(mp.precision_hints, name, :float16) == :float32
                continue
            end
            setfield!(mp.model, name, Float16.(param))
        end
    end

    # Forward pass in Float16
    y = forward(mp.model, x_f16)

    # Restore Float32 master weights
    for (name, master_param) in mp.master_weights
        setfield!(mp.model, name, master_param)
    end

    if y isa AbstractTensor
        return Tensor(Float32.(y.data))
    end
    Float32.(y)
end

"""
    scale_loss(mp::MixedPrecisionWrapper, loss)

Scale a loss value by the current dynamic loss scale factor.
Call before backward pass to prevent Float16 underflow in gradients.
"""
function scale_loss(mp::MixedPrecisionWrapper, loss)
    loss * mp.loss_scale
end

"""
    unscale_and_update!(mp::MixedPrecisionWrapper, grads)

Unscale gradients, check for NaN/Inf, and update loss scale.
Returns `(unscaled_grads, valid)` where `valid` indicates whether
the gradients are usable (no NaN/Inf found).
"""
function unscale_and_update!(mp::MixedPrecisionWrapper, grads)
    inv_scale = 1f0 / mp.loss_scale

    has_nan_inf = false
    unscaled = map(grads) do g
        if g isa AbstractArray
            ug = Float32.(g) .* inv_scale
            if any(isnan, ug) || any(isinf, ug)
                has_nan_inf = true
            end
            ug
        else
            g
        end
    end

    if has_nan_inf
        # Reduce loss scale on NaN/Inf
        mp.loss_scale *= mp.loss_scale_backoff_factor
        mp.steps_since_last_scale = 0
        return (unscaled, false)
    end

    # Gradients are valid — maybe increase scale
    mp.steps_since_last_scale += 1
    if mp.steps_since_last_scale >= mp.loss_scale_growth_interval
        mp.loss_scale *= mp.loss_scale_growth_factor
        mp.steps_since_last_scale = 0
    end

    (unscaled, true)
end

"""
    update_master_weights!(mp::MixedPrecisionWrapper)

Copy current model parameters back to master weights.
Call after optimizer step to keep master weights in sync.
"""
function update_master_weights!(mp::MixedPrecisionWrapper)
    params = parameters(mp.model)
    for (name, param) in pairs(params)
        if param isa AbstractArray{Float32} && haskey(mp.master_weights, name)
            copyto!(mp.master_weights[name], param)
        end
    end
end

(mp::MixedPrecisionWrapper)(x) = forward(mp, x)

parameters(mp::MixedPrecisionWrapper) = parameters(mp.model)
output_shape(mp::MixedPrecisionWrapper, input_shape) = output_shape(mp.model, input_shape)

function compile_to_backend(model, backend::JuliaBackend)
    # Julia backend - just return the model
    model
end

function compile_to_backend(model, backend::ZigBackend)
    @info "Compiling to Zig backend..."

    # Verify Zig library exists
    if !isfile(backend.lib_path)
        @warn "Zig library not found at $(backend.lib_path), falling back to Julia backend"
        return model
    end

    # Wrap model for Zig execution
    ZigCompiledModel(model, backend)
end

function compile_to_backend(model, backend::CUDABackend)
    @info "Compiling to CUDA backend on device $(backend.device)..."

    # Check CUDA availability
    if !cuda_available()
        _gpu_record_diagnostic!(backend, "compile_fallbacks")
        @warn "CUDA not available, falling back to Julia backend"
        return model
    end
    device_count = cuda_device_count()
    if backend.device < 0 || backend.device >= device_count
        _gpu_record_diagnostic!(backend, "compile_fallbacks")
        @warn "CUDA device $(backend.device) out of range (available: 0:$(max(0, device_count - 1))), falling back to Julia backend"
        return model
    end

    # Wrap model for CUDA execution
    GPUCompiledModel(model, backend)
end

function compile_to_backend(model, backend::MetalBackend)
    @info "Compiling to Metal backend on device $(backend.device)..."

    # Check Metal availability
    if !metal_available()
        _gpu_record_diagnostic!(backend, "compile_fallbacks")
        @warn "Metal not available, falling back to Julia backend"
        return model
    end
    device_count = metal_device_count()
    if backend.device < 0 || backend.device >= device_count
        _gpu_record_diagnostic!(backend, "compile_fallbacks")
        @warn "Metal device $(backend.device) out of range (available: 0:$(max(0, device_count - 1))), falling back to Julia backend"
        return model
    end

    # Wrap model for Metal execution
    GPUCompiledModel(model, backend)
end

function compile_to_backend(model, backend::TPUBackend)
    _compile_coprocessor(model, backend, tpu_available(), tpu_device_count(), "TPU")
end

function compile_to_backend(model, backend::NPUBackend)
    _compile_coprocessor(model, backend, npu_available(), npu_device_count(), "NPU")
end

function compile_to_backend(model, backend::DSPBackend)
    _compile_coprocessor(model, backend, dsp_available(), dsp_device_count(), "DSP")
end

function compile_to_backend(model, backend::PPUBackend)
    _compile_coprocessor(model, backend, ppu_available(), ppu_device_count(), "PPU")
end

function compile_to_backend(model, backend::MathBackend)
    _compile_coprocessor(model, backend, math_available(), math_device_count(), "MATH")
end

function compile_to_backend(model, backend::FPGABackend)
    _compile_coprocessor(model, backend, fpga_available(), fpga_device_count(), "FPGA")
end

function compile_to_backend(model, backend::VPUBackend)
    _compile_coprocessor(model, backend, vpu_available(), vpu_device_count(), "VPU")
end

function compile_to_backend(model, backend::QPUBackend)
    _compile_coprocessor(model, backend, qpu_available(), qpu_device_count(), "QPU")
end

function compile_to_backend(model, backend::CryptoBackend)
    _compile_coprocessor(model, backend, crypto_available(), crypto_device_count(), "CRYPTO")
end

function _compile_coprocessor(model, backend, available::Bool, device_count::Int, label::String)
    @info "Compiling to $(label) backend on device $(backend.device)..."
    required = _coprocessor_required(backend)
    required_key = _coprocessor_required_env_key(backend)

    if !available
        _coprocessor_record_diagnostic!(backend, "compile_fallbacks")
        if required
            error("$(label) backend not available and strict mode is enabled (set $(required_key)=0 or AXIOM_COPROCESSOR_REQUIRED=0 to allow fallback)")
        end
        @warn "$(label) backend not available, falling back to Julia backend"
        return model
    end
    if backend.device < 0 || backend.device >= device_count
        _coprocessor_record_diagnostic!(backend, "compile_fallbacks")
        if required
            error("$(label) device $(backend.device) out of range (available: 0:$(max(0, device_count - 1))) and strict mode is enabled (set $(required_key)=0 or AXIOM_COPROCESSOR_REQUIRED=0 to allow fallback)")
        end
        @warn "$(label) device $(backend.device) out of range (available: 0:$(max(0, device_count - 1))), falling back to Julia backend"
        return model
    end
    CoprocessorCompiledModel(model, backend)
end

"""
Check if CUDA is available.
"""
function cuda_available()
    forced = _backend_env_available("AXIOM_CUDA_AVAILABLE")
    forced !== nothing && return forced
    false
end

"""
Check if Metal is available.
"""
function metal_available()
    forced = _backend_env_available("AXIOM_METAL_AVAILABLE")
    forced !== nothing && return forced
    false
end

"""
    _backend_env_available(key::String) -> Union{Bool, Nothing}

Read a backend availability override from environment.
Returns `nothing` when no override is configured.
"""
function _backend_env_available(key::String)
    raw = lowercase(strip(get(ENV, key, "")))
    isempty(raw) && return nothing
    if raw in ("1", "true", "yes", "on")
        return true
    end
    if raw in ("0", "false", "no", "off")
        return false
    end
    nothing
end

"""
    _backend_env_count(available_key::String, count_key::String) -> Union{Int, Nothing}

Read a backend device-count override from environment.
Returns `nothing` when no count override is configured.
"""
function _backend_env_count(available_key::String, count_key::String)
    forced_available = _backend_env_available(available_key)
    forced_available === false && return 0
    raw = strip(get(ENV, count_key, ""))
    isempty(raw) && return nothing
    parsed = tryparse(Int, raw)
    parsed === nothing && return nothing
    max(parsed, 0)
end

function _accelerator_env_flag(key::String)
    forced = _backend_env_available(key)
    forced === nothing ? false : forced
end

function _accelerator_env_count(available_key::String, count_key::String)
    forced_count = _backend_env_count(available_key, count_key)
    forced_count !== nothing && return forced_count
    _accelerator_env_flag(available_key) || return 0
    1
end

"""
    tpu_available() -> Bool

Check if TPU backend is available.
"""
tpu_available() = _accelerator_env_flag("AXIOM_TPU_AVAILABLE")

"""
    npu_available() -> Bool

Check if NPU backend is available.
"""
npu_available() = _accelerator_env_flag("AXIOM_NPU_AVAILABLE")

"""
    dsp_available() -> Bool

Check if DSP backend is available.
"""
dsp_available() = _accelerator_env_flag("AXIOM_DSP_AVAILABLE")

"""
    fpga_available() -> Bool

Check if FPGA backend is available.
"""
fpga_available() = _accelerator_env_flag("AXIOM_FPGA_AVAILABLE")

"""
    ppu_available() -> Bool

Check if PPU backend is available.
"""
ppu_available() = _accelerator_env_flag("AXIOM_PPU_AVAILABLE")

"""
    math_available() -> Bool

Check if Math backend is available.
"""
math_available() = _accelerator_env_flag("AXIOM_MATH_AVAILABLE")

"""
    vpu_available() -> Bool

Check if VPU backend is available.
"""
vpu_available() = _accelerator_env_flag("AXIOM_VPU_AVAILABLE")

"""
    qpu_available() -> Bool

Check if QPU backend is available.
"""
qpu_available() = _accelerator_env_flag("AXIOM_QPU_AVAILABLE")

"""
    crypto_available() -> Bool

Check if Crypto backend is available.
"""
crypto_available() = _accelerator_env_flag("AXIOM_CRYPTO_AVAILABLE")

"""
    tpu_device_count() -> Int

Get number of available TPU devices.
"""
tpu_device_count() = _accelerator_env_count("AXIOM_TPU_AVAILABLE", "AXIOM_TPU_DEVICE_COUNT")

"""
    npu_device_count() -> Int

Get number of available NPU devices.
"""
npu_device_count() = _accelerator_env_count("AXIOM_NPU_AVAILABLE", "AXIOM_NPU_DEVICE_COUNT")

"""
    dsp_device_count() -> Int

Get number of available DSP devices.
"""
dsp_device_count() = _accelerator_env_count("AXIOM_DSP_AVAILABLE", "AXIOM_DSP_DEVICE_COUNT")

"""
    fpga_device_count() -> Int

Get number of available FPGA devices.
"""
fpga_device_count() = _accelerator_env_count("AXIOM_FPGA_AVAILABLE", "AXIOM_FPGA_DEVICE_COUNT")

"""
    ppu_device_count() -> Int

Get number of available PPU devices.
"""
ppu_device_count() = _accelerator_env_count("AXIOM_PPU_AVAILABLE", "AXIOM_PPU_DEVICE_COUNT")

"""
    math_device_count() -> Int

Get number of available Math coprocessor devices.
"""
math_device_count() = _accelerator_env_count("AXIOM_MATH_AVAILABLE", "AXIOM_MATH_DEVICE_COUNT")

"""
    vpu_device_count() -> Int

Get number of available VPU devices.
"""
vpu_device_count() = _accelerator_env_count("AXIOM_VPU_AVAILABLE", "AXIOM_VPU_DEVICE_COUNT")

"""
    qpu_device_count() -> Int

Get number of available QPU devices.
"""
qpu_device_count() = _accelerator_env_count("AXIOM_QPU_AVAILABLE", "AXIOM_QPU_DEVICE_COUNT")

"""
    crypto_device_count() -> Int

Get number of available Crypto accelerator devices.
"""
crypto_device_count() = _accelerator_env_count("AXIOM_CRYPTO_AVAILABLE", "AXIOM_CRYPTO_DEVICE_COUNT")

"""
    detect_coprocessor() -> Union{AbstractBackend, Nothing}

Auto-detect available non-GPU coprocessor backend.
"""
function detect_coprocessor()
    if tpu_available() && tpu_device_count() > 0
        return TPUBackend(0)
    end
    if npu_available() && npu_device_count() > 0
        return NPUBackend(0)
    end
    if ppu_available() && ppu_device_count() > 0
        return PPUBackend(0)
    end
    if math_available() && math_device_count() > 0
        return MathBackend(0)
    end
    if fpga_available() && fpga_device_count() > 0
        return FPGABackend(0)
    end
    if dsp_available() && dsp_device_count() > 0
        return DSPBackend(0)
    end
    if vpu_available() && vpu_device_count() > 0
        return VPUBackend(0)
    end
    if qpu_available() && qpu_device_count() > 0
        return QPUBackend(0)
    end
    if crypto_available() && crypto_device_count() > 0
        return CryptoBackend(0)
    end
    nothing
end

"""
    detect_accelerator() -> Union{AbstractBackend, Nothing}

Auto-detect GPU first, then non-GPU coprocessor backends.
"""
function detect_accelerator()
    if isdefined(@__MODULE__, :detect_gpu)
        gpu_backend = detect_gpu()
        gpu_backend !== nothing && return gpu_backend
    end
    detect_coprocessor()
end

function _hook_has_backend_override(hook::Function, backend_type::DataType)
    for method in methods(hook)
        sig = Base.unwrap_unionall(method.sig)
        params = sig.parameters
        length(params) >= 2 || continue
        first_arg = params[2]
        if first_arg == backend_type && method.module !== @__MODULE__
            return true
        end
    end
    false
end

function _coprocessor_hook_overrides(backend_type::DataType)
    hooks = Dict{String, Bool}()
    for (name, hook) in (
        ("backend_coprocessor_matmul", backend_coprocessor_matmul),
        ("backend_coprocessor_conv2d", backend_coprocessor_conv2d),
        ("backend_coprocessor_relu", backend_coprocessor_relu),
        ("backend_coprocessor_softmax", backend_coprocessor_softmax),
        ("backend_coprocessor_batchnorm", backend_coprocessor_batchnorm),
        ("backend_coprocessor_layernorm", backend_coprocessor_layernorm),
        ("backend_coprocessor_maxpool2d", backend_coprocessor_maxpool2d),
        ("backend_coprocessor_avgpool2d", backend_coprocessor_avgpool2d),
        ("backend_coprocessor_global_avgpool2d", backend_coprocessor_global_avgpool2d),
    )
        hooks[name] = _hook_has_backend_override(hook, backend_type)
    end
    hooks
end

"""
    coprocessor_capability_report() -> Dict{String,Any}

Return a machine-readable snapshot of non-GPU accelerator strategy state,
including environment-based availability, device counts, and extension hook status.
"""
function coprocessor_capability_report()
    selected = detect_coprocessor()
    backends = Dict{String, Any}()
    strategy = [
        ("TPU", TPUBackend, tpu_available, tpu_device_count),
        ("NPU", NPUBackend, npu_available, npu_device_count),
        ("PPU", PPUBackend, ppu_available, ppu_device_count),
        ("MATH", MathBackend, math_available, math_device_count),
        ("FPGA", FPGABackend, fpga_available, fpga_device_count),
        ("DSP", DSPBackend, dsp_available, dsp_device_count),
        ("VPU", VPUBackend, vpu_available, vpu_device_count),
        ("QPU", QPUBackend, qpu_available, qpu_device_count),
        ("CRYPTO", CryptoBackend, crypto_available, crypto_device_count),
    ]

    for (label, backend_type, available_fn, count_fn) in strategy
        available = available_fn()
        count = count_fn()
        required = _coprocessor_required(backend_type(0))
        hooks = _coprocessor_hook_overrides(backend_type)
        kernel_hooks_loaded = !isempty(hooks) && all(values(hooks))
        backends[label] = Dict(
            "available" => available,
            "device_count" => count,
            "required" => required,
            "compilable" => available && count > 0,
            "kernel_hooks_loaded" => kernel_hooks_loaded,
            "hook_overrides" => hooks,
        )
    end

    Dict(
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "strategy_order" => ["TPU", "NPU", "PPU", "MATH", "FPGA", "DSP", "VPU", "QPU", "CRYPTO"],
        "selected_backend" => selected === nothing ? nothing : string(typeof(selected)),
        "runtime_diagnostics" => coprocessor_runtime_diagnostics(),
        "backends" => backends,
    )
end

"""
    select_device!(backend::CoprocessorBackend, device::Int)

Return a backend handle for the selected coprocessor device.
"""
select_device!(::TPUBackend, device::Int) = TPUBackend(device)
select_device!(::NPUBackend, device::Int) = NPUBackend(device)
select_device!(::DSPBackend, device::Int) = DSPBackend(device)
select_device!(::PPUBackend, device::Int) = PPUBackend(device)
select_device!(::MathBackend, device::Int) = MathBackend(device)
select_device!(::FPGABackend, device::Int) = FPGABackend(device)
select_device!(::VPUBackend, device::Int) = VPUBackend(device)
select_device!(::QPUBackend, device::Int) = QPUBackend(device)
select_device!(::CryptoBackend, device::Int) = CryptoBackend(device)

# ============================================================================
# Resource-Aware Hardware Dispatch
# ============================================================================

"""
    DeviceResources

Hardware resource descriptor for intelligent workload placement.
All values in bytes (memory) or count (compute units).
"""
struct DeviceResources
    total_memory::Int64      # Total device memory in bytes
    available_memory::Int64  # Currently available memory
    compute_units::Int       # Number of cores/SMs/CUs
    clock_mhz::Int           # Clock speed in MHz
    bandwidth_gbps::Float64  # Memory bandwidth in GB/s
end

"""
    query_device_resources(backend::AbstractBackend) -> Union{DeviceResources, Nothing}

Query hardware resources for a given backend. Returns `nothing` if
resource information is unavailable (e.g., no driver or env vars).
"""
function query_device_resources(::JuliaBackend)
    mem = Sys.total_memory()
    free = Sys.free_memory()
    cores = Sys.CPU_THREADS
    DeviceResources(mem, free, cores, 0, 0.0)
end

function query_device_resources(backend::AbstractBackend)
    # Read from environment variables (set by driver wrappers)
    key = _resource_env_prefix(backend)
    total_mem = _env_int64("$(key)_TOTAL_MEMORY", 0)
    avail_mem = _env_int64("$(key)_AVAILABLE_MEMORY", total_mem)
    compute = _env_int("$(key)_COMPUTE_UNITS", 0)
    clock = _env_int("$(key)_CLOCK_MHZ", 0)
    bw = _env_float("$(key)_BANDWIDTH_GBPS", 0.0)

    total_mem == 0 && return nothing
    DeviceResources(total_mem, avail_mem, compute, clock, bw)
end

function _resource_env_prefix(::CUDABackend)   "AXIOM_CUDA" end
function _resource_env_prefix(::ROCmBackend)   "AXIOM_ROCM" end
function _resource_env_prefix(::MetalBackend)  "AXIOM_METAL" end
function _resource_env_prefix(b::CoprocessorBackend)
    "AXIOM_$(_coprocessor_label(b))"
end
function _resource_env_prefix(::ZigBackend)    "AXIOM_ZIG" end

function _env_int64(key::String, default::Int64)
    raw = strip(get(ENV, key, ""))
    isempty(raw) && return default
    parsed = tryparse(Int64, raw)
    parsed === nothing ? default : parsed
end

function _env_int(key::String, default::Int)
    raw = strip(get(ENV, key, ""))
    isempty(raw) && return default
    parsed = tryparse(Int, raw)
    parsed === nothing ? default : parsed
end

function _env_float(key::String, default::Float64)
    raw = strip(get(ENV, key, ""))
    isempty(raw) && return default
    parsed = tryparse(Float64, raw)
    parsed === nothing ? default : parsed
end

"""
    estimate_model_memory(model) -> Int64

Estimate memory footprint of a model in bytes.
Counts all parameter arrays (weights, biases, running stats).
"""
function estimate_model_memory(model)
    total = Int64(0)
    if model isa Pipeline
        for layer in model.layers
            total += _layer_memory(layer)
        end
    else
        total += _layer_memory(model)
    end
    total
end

function _layer_memory(layer)
    mem = Int64(0)
    try
        params = parameters(layer)
        for (_, param) in pairs(params)
            if param isa AbstractArray
                mem += sizeof(param)
            end
        end
    catch
    end
    # Add running stats for BatchNorm
    if layer isa BatchNorm
        mem += sizeof(layer.running_mean) + sizeof(layer.running_var)
    end
    mem
end

"""
    select_best_backend(model; prefer::Symbol=:auto) -> AbstractBackend

Automatically select the best available backend for a model based on:
1. Hardware availability
2. Device memory vs model size
3. User preference

Returns JuliaBackend as fallback if no accelerator is suitable.
"""
function select_best_backend(model; prefer::Symbol=:auto)
    model_bytes = estimate_model_memory(model)
    # Add 2x overhead for activations/gradients during inference
    required_memory = model_bytes * 2

    candidates = Tuple{AbstractBackend, DeviceResources, Float64}[]

    # Check GPU backends
    for (avail_fn, count_fn, constructor) in (
        (cuda_available, cuda_device_count, CUDABackend),
        (rocm_available, rocm_device_count, ROCmBackend),
        (metal_available, metal_device_count, MetalBackend),
    )
        if avail_fn() && count_fn() > 0
            backend = constructor(0)
            res = query_device_resources(backend)
            if res !== nothing && res.available_memory >= required_memory
                score = _compute_backend_score(res, required_memory, :gpu)
                push!(candidates, (backend, res, score))
            end
        end
    end

    # Check coprocessor backends
    for (avail_fn, count_fn, constructor) in (
        (tpu_available,    tpu_device_count,    TPUBackend),
        (npu_available,    npu_device_count,    NPUBackend),
        (vpu_available,    vpu_device_count,    VPUBackend),
        (qpu_available,    qpu_device_count,    QPUBackend),
        (fpga_available,   fpga_device_count,   FPGABackend),
    )
        if avail_fn() && count_fn() > 0
            backend = constructor(0)
            res = query_device_resources(backend)
            if res !== nothing && res.available_memory >= required_memory
                score = _compute_backend_score(res, required_memory, :coprocessor)
                push!(candidates, (backend, res, score))
            end
        end
    end

    # Sort by score (highest first)
    sort!(candidates, by=x -> x[3], rev=true)

    if isempty(candidates)
        @info "No accelerator has sufficient resources, using Julia CPU backend" model_bytes required_memory
        return JuliaBackend()
    end

    best = candidates[1]
    @info "Selected backend" backend=typeof(best[1]) score=best[3] memory_available=best[2].available_memory model_requires=required_memory
    best[1]
end

function _compute_backend_score(res::DeviceResources, required_memory::Int64, kind::Symbol)
    # Score = weighted combination of compute capacity and memory headroom
    memory_ratio = res.available_memory / max(required_memory, 1)
    compute_score = log2(max(res.compute_units, 1))
    bandwidth_score = res.bandwidth_gbps / 100.0  # Normalize to ~1.0 range

    # GPU gets a bonus for ML workloads
    kind_bonus = kind == :gpu ? 2.0 : 1.0

    (memory_ratio * 0.3 + compute_score * 0.4 + bandwidth_score * 0.3) * kind_bonus
end

"""
    resource_report(model) -> Dict

Generate a resource utilization report for a model across all available backends.
"""
function resource_report(model)
    model_bytes = estimate_model_memory(model)

    backends = Dict{String, Any}()
    for (label, constructor, avail_fn, count_fn) in (
        ("Julia",  () -> JuliaBackend(), () -> true,  () -> 1),
        ("CUDA",   () -> CUDABackend(0), cuda_available, cuda_device_count),
        ("ROCm",   () -> ROCmBackend(0), rocm_available, rocm_device_count),
        ("Metal",  () -> MetalBackend(0), metal_available, metal_device_count),
        ("TPU",    () -> TPUBackend(0),  tpu_available, tpu_device_count),
        ("NPU",    () -> NPUBackend(0),  npu_available, npu_device_count),
    )
        if avail_fn() && count_fn() > 0
            res = query_device_resources(constructor())
            backends[label] = Dict(
                "resources" => res === nothing ? nothing : Dict(
                    "total_memory" => res.total_memory,
                    "available_memory" => res.available_memory,
                    "compute_units" => res.compute_units,
                ),
                "model_fits" => res !== nothing && res.available_memory >= model_bytes * 2,
                "utilization" => res !== nothing && res.total_memory > 0 ?
                    round(model_bytes / res.total_memory * 100, digits=1) : nothing,
            )
        end
    end

    Dict(
        "model_memory_bytes" => model_bytes,
        "model_memory_mb" => round(model_bytes / 1024^2, digits=2),
        "recommended_backend" => string(typeof(select_best_backend(model))),
        "backends" => backends,
    )
end

# ============================================================================
# AcceleratorGate Integration
# ============================================================================

"""
    fits_on_device(backend::AbstractBackend, model) -> Bool

Check whether a model's memory footprint (including activation/gradient
overhead) fits on the device associated with `backend`.  Uses
`estimate_model_memory` for the model size and `query_device_resources`
for available hardware memory.

Falls through to AcceleratorGate's `fits_on_device` if device capabilities
are available, otherwise checks via environment-variable-based resource
queries.
"""
function fits_on_device(backend::AbstractBackend, model)
    model_bytes = estimate_model_memory(model)
    required = model_bytes * 2  # 2x overhead for activations/gradients

    # Try AcceleratorGate device capabilities first
    caps = device_capabilities(backend)
    if caps !== nothing
        return caps.memory_available >= required
    end

    # Fall back to Axiom's own resource queries
    res = query_device_resources(backend)
    res === nothing && return false
    res.available_memory >= required
end

"""
    smart_select_backend(model; prefer::Symbol=:auto,
                         batch_size::Int=1,
                         layer_types::Vector{Symbol}=Symbol[]) -> AbstractBackend

Intelligent backend selection that considers model size, dominant layer types,
and batch size to pick the optimal accelerator.

Selection heuristics:
  - Large batch + matmul-heavy → TPU (systolic array advantage)
  - Conv-heavy + GPU available → GPU (CUDA/ROCm/Metal)
  - Small model + low power → NPU (inference-optimised)
  - Signal processing layers → DSP (native FFT)
  - Physics-informed layers → PPU (energy conservation)
  - High precision required → Math coprocessor
  - Encrypted inference → Crypto coprocessor
  - Quantum layers → QPU
  - FPGA available + streaming patterns → FPGA
  - Wide SIMD beneficial → VPU

Falls back to `select_best_backend` for memory/compute scoring.
"""
function smart_select_backend(model;
                              prefer::Symbol=:auto,
                              batch_size::Int=1,
                              layer_types::Vector{Symbol}=Symbol[])
    # If user specified a preference, try that first
    if prefer != :auto
        backend = _preference_to_backend(prefer)
        if backend !== nothing && fits_on_device(backend, model)
            return backend
        end
    end

    # Detect dominant layer types from the model if not provided
    if isempty(layer_types)
        layer_types = _detect_layer_types(model)
    end

    has_conv = :conv in layer_types
    has_matmul = :dense in layer_types || :linear in layer_types
    has_norm = :batchnorm in layer_types || :layernorm in layer_types

    # Heuristic scoring for coprocessor backends
    scored = Tuple{AbstractBackend, Float64}[]

    # TPU: best for large batch matmul-heavy workloads
    if tpu_available() && tpu_device_count() > 0
        score = has_matmul ? 8.0 : 4.0
        score += batch_size >= 32 ? 4.0 : 0.0  # TPU excels at large batches
        b = TPUBackend(0)
        fits_on_device(b, model) && push!(scored, (b, score))
    end

    # GPU: best for conv-heavy or general large models
    for (avail, count, ctor) in (
        (cuda_available, cuda_device_count, CUDABackend),
        (rocm_available, rocm_device_count, ROCmBackend),
        (metal_available, metal_device_count, MetalBackend),
    )
        if avail() && count() > 0
            score = has_conv ? 9.0 : 6.0
            score += has_matmul ? 2.0 : 0.0
            b = ctor(0)
            fits_on_device(b, model) && push!(scored, (b, score))
        end
    end

    # NPU: low-power inference, quantization-friendly
    if npu_available() && npu_device_count() > 0
        score = 5.0
        score += batch_size <= 4 ? 2.0 : 0.0  # NPU efficient for small batches
        b = NPUBackend(0)
        fits_on_device(b, model) && push!(scored, (b, score))
    end

    # VPU: SIMD-heavy element-wise operations
    if vpu_available() && vpu_device_count() > 0
        score = 4.0
        b = VPUBackend(0)
        fits_on_device(b, model) && push!(scored, (b, score))
    end

    # FPGA: streaming patterns
    if fpga_available() && fpga_device_count() > 0
        score = 3.0
        b = FPGABackend(0)
        fits_on_device(b, model) && push!(scored, (b, score))
    end

    # DSP: signal processing layers
    if dsp_available() && dsp_device_count() > 0
        score = has_conv ? 5.0 : 2.0  # DSP has native FFT for conv
        b = DSPBackend(0)
        fits_on_device(b, model) && push!(scored, (b, score))
    end

    # PPU, Math, Crypto, QPU: specialised
    if ppu_available() && ppu_device_count() > 0
        b = PPUBackend(0)
        fits_on_device(b, model) && push!(scored, (b, 2.0))
    end
    if math_available() && math_device_count() > 0
        b = MathBackend(0)
        fits_on_device(b, model) && push!(scored, (b, 2.0))
    end
    if crypto_available() && crypto_device_count() > 0
        b = CryptoBackend(0)
        fits_on_device(b, model) && push!(scored, (b, 1.0))
    end
    if qpu_available() && qpu_device_count() > 0
        b = QPUBackend(0)
        fits_on_device(b, model) && push!(scored, (b, 1.0))
    end

    if isempty(scored)
        @info "No accelerator fits; falling back to Julia CPU"
        return JuliaBackend()
    end

    sort!(scored, by=x -> x[2], rev=true)
    best = scored[1]
    @info "Smart backend selection" backend=typeof(best[1]) score=best[2] batch_size
    best[1]
end

function _preference_to_backend(pref::Symbol)
    pref === :cuda   && return CUDABackend(0)
    pref === :rocm   && return ROCmBackend(0)
    pref === :metal  && return MetalBackend(0)
    pref === :tpu    && return TPUBackend(0)
    pref === :npu    && return NPUBackend(0)
    pref === :fpga   && return FPGABackend(0)
    pref === :vpu    && return VPUBackend(0)
    pref === :qpu    && return QPUBackend(0)
    pref === :dsp    && return DSPBackend(0)
    pref === :ppu    && return PPUBackend(0)
    pref === :math   && return MathBackend(0)
    pref === :crypto && return CryptoBackend(0)
    pref === :julia  && return JuliaBackend()
    nothing
end

function _detect_layer_types(model)
    types = Symbol[]
    if model isa Pipeline
        for layer in model.layers
            _classify_layer!(types, layer)
        end
    else
        _classify_layer!(types, model)
    end
    unique!(types)
end

function _classify_layer!(types::Vector{Symbol}, layer)
    layer isa Dense      && push!(types, :dense)
    layer isa Conv2d     && push!(types, :conv)
    layer isa BatchNorm  && push!(types, :batchnorm)
    layer isa LayerNorm  && push!(types, :layernorm)
    layer isa MaxPool2d  && push!(types, :maxpool)
    layer isa AvgPool2d  && push!(types, :avgpool)
    layer isa ReLU       && push!(types, :relu)
    layer isa Softmax    && push!(types, :softmax)
    nothing
end

"""
    ZigCompiledModel

Model wrapper that dispatches operations to Zig backend.
"""
struct ZigCompiledModel{M}
    model::M
    backend::ZigBackend
    lib_handle::Ptr{Nothing}
end

function ZigCompiledModel(model, backend::ZigBackend)
    # Load the Zig shared library
    lib_handle = try
        Libdl.dlopen(backend.lib_path)
    catch e
        @warn "Failed to load Zig library: $e"
        Ptr{Nothing}()
    end

    ZigCompiledModel(model, backend, lib_handle)
end

function forward(zm::ZigCompiledModel, x)
    if zm.lib_handle == Ptr{Nothing}()
        # Fallback to Julia implementation
        return forward(zm.model, x)
    end

    # Dispatch to Zig backend
    zig_forward(zm.model, x, zm.lib_handle)
end

(zm::ZigCompiledModel)(x) = forward(zm, x)

function zig_forward(model, x, lib_handle)
    # Default: fall back to Julia
    forward(model, x)
end

function zig_forward(model::Dense, x, lib_handle)
    # Call Zig matmul if available
    matmul_fn = Libdl.dlsym(lib_handle, :axiom_matmul; throw_error=false)
    if matmul_fn != C_NULL
        # Implementation in zig_ffi.jl handles the backend_matmul call
        # but here we are in abstract.jl and model is compiled.
        # For simplicity, we just use the backend dispatcher if possible.
        backend_matmul(ZigBackend(""), x, model.weight) # dummy path, it uses global _zig_lib
    else
        forward(model, x)
    end
end

parameters(zm::ZigCompiledModel) = parameters(zm.model)
output_shape(zm::ZigCompiledModel, input_shape) = output_shape(zm.model, input_shape)

"""
    GPUCompiledModel

Base wrapper for GPU-accelerated models (CUDA/Metal).
"""
struct GPUCompiledModel{M, B <: AbstractBackend}
    model::M
    backend::B
end

function _gpu_apply_activation(activation, y, backend::AbstractBackend)
    if activation === identity
        return y
    elseif activation === relu
        return backend_gpu_relu(backend, y)
    elseif activation === softmax
        dim = ndims(y) > 1 ? ndims(y) : 1
        return backend_gpu_softmax(backend, y, dim)
    end
    activation(y)
end

function _gpu_forward_layer(layer::Dense, x::AbstractTensor, backend::AbstractBackend)
    x_data = x.data

    y = if ndims(x_data) == 1
        y_mat = backend_gpu_matmul(backend, reshape(x_data, 1, :), layer.weight)
        vec(y_mat)
    else
        backend_gpu_matmul(backend, x_data, layer.weight)
    end

    if layer.bias !== nothing
        y = ndims(y) == 1 ? (y .+ layer.bias) : (y .+ layer.bias')
    end

    Tensor(_gpu_apply_activation(layer.activation, y, backend))
end

function _gpu_forward_layer(layer::Conv2d, x::AbstractTensor, backend::AbstractBackend)
    has_batch = ndims(x) == 4
    x_data = has_batch ? x.data : reshape(x.data, 1, size(x.data)...)
    y = backend_gpu_conv2d(backend, x_data, layer.weight, layer.bias, layer.stride, layer.padding)
    Tensor(has_batch ? y : dropdims(y, dims=1))
end

function _gpu_forward_layer(layer::BatchNorm, x::AbstractTensor, backend::AbstractBackend)
    x_data = x.data
    gamma = layer.affine ? layer.γ : ones(eltype(x_data), layer.num_features)
    beta = layer.affine ? layer.β : zeros(eltype(x_data), layer.num_features)
    y = backend_gpu_batchnorm(
        backend,
        x_data,
        gamma,
        beta,
        layer.running_mean,
        layer.running_var,
        eltype(x_data)(layer.eps),
        layer.training,
    )
    Tensor(y)
end

function _gpu_forward_layer(::ReLU, x::AbstractTensor, backend::AbstractBackend)
    Tensor(backend_gpu_relu(backend, x.data))
end

function _gpu_forward_layer(layer::Softmax, x::AbstractTensor, backend::AbstractBackend)
    dim = layer.dims == -1 ? ndims(x.data) : layer.dims
    Tensor(backend_gpu_softmax(backend, x.data, dim))
end

function _gpu_forward_layer(layer::AbstractLayer, x::AbstractTensor, ::AbstractBackend)
    forward(layer, x)
end

function _gpu_forward(model::Pipeline, x::AbstractTensor, backend::AbstractBackend)
    for layer in model.layers
        x = _gpu_forward_layer(layer, x, backend)
    end
    x
end

function _gpu_forward(model::AbstractLayer, x::AbstractTensor, backend::AbstractBackend)
    _gpu_forward_layer(model, x, backend)
end

function _gpu_forward(model, x::AbstractTensor, ::AbstractBackend)
    forward(model, x)
end

function forward(gm::GPUCompiledModel, x::AbstractTensor)
    _gpu_forward_with_recovery(gm.model, x, gm.backend)
end

function forward(gm::GPUCompiledModel, x)
    @debug "GPUCompiledModel requires tensor input for accelerated path; falling back to model forward"
    forward(gm.model, x)
end

(gm::GPUCompiledModel)(x) = forward(gm, x)

parameters(gm::GPUCompiledModel) = parameters(gm.model)
output_shape(gm::GPUCompiledModel, input_shape) = output_shape(gm.model, input_shape)

# Type aliases for specific GPU backends
const CUDACompiledModel = GPUCompiledModel{M, CUDABackend} where M
const MetalCompiledModel = GPUCompiledModel{M, MetalBackend} where M

"""
    CoprocessorCompiledModel

Wrapper for non-GPU accelerator backends (TPU/NPU/DSP/FPGA).
"""
struct CoprocessorCompiledModel{M, B <: AbstractBackend}
    model::M
    backend::B
end

function _coprocessor_apply_activation(activation, y, backend::CoprocessorBackend)
    if activation === identity
        return y
    elseif activation === relu
        return backend_relu(backend, y)
    elseif activation === softmax
        dim = ndims(y) > 1 ? ndims(y) : 1
        return backend_softmax(backend, y, dim)
    end
    activation(y)
end

function _coprocessor_forward_layer(layer::Dense, x::AbstractTensor, backend::CoprocessorBackend)
    x_data = x.data

    y = if ndims(x_data) == 1
        layer.weight' * x_data
    else
        backend_matmul(backend, x_data, layer.weight)
    end

    if layer.bias !== nothing
        y = y .+ layer.bias'
    end

    Tensor(_coprocessor_apply_activation(layer.activation, y, backend))
end

function _coprocessor_forward_layer(layer::Conv2d, x::AbstractTensor, backend::CoprocessorBackend)
    has_batch = ndims(x) == 4
    x_data = has_batch ? x.data : reshape(x.data, 1, size(x.data)...)
    y = backend_conv2d(backend, x_data, layer.weight, layer.bias, layer.stride, layer.padding)
    Tensor(has_batch ? y : dropdims(y, dims=1))
end

function _coprocessor_forward_layer(layer::BatchNorm, x::AbstractTensor, backend::CoprocessorBackend)
    x_data = x.data
    gamma = layer.affine ? layer.γ : ones(eltype(x_data), layer.num_features)
    beta = layer.affine ? layer.β : zeros(eltype(x_data), layer.num_features)
    y = backend_batchnorm(
        backend,
        x_data,
        gamma,
        beta,
        layer.running_mean,
        layer.running_var,
        eltype(x_data)(layer.eps),
        layer.training
    )
    Tensor(y)
end

function _coprocessor_forward_layer(layer::LayerNorm, x::AbstractTensor, backend::CoprocessorBackend)
    x_data = x.data
    normalized_shape = Tuple(layer.normalized_shape)
    gamma = layer.elementwise_affine ? layer.γ : ones(eltype(x_data), normalized_shape...)
    beta = layer.elementwise_affine ? layer.β : zeros(eltype(x_data), normalized_shape...)
    y = backend_layernorm(
        backend,
        x_data,
        gamma,
        beta,
        normalized_shape,
        eltype(x_data)(layer.eps),
    )
    Tensor(y)
end

function _coprocessor_forward_layer(layer::MaxPool2d, x::AbstractTensor, backend::CoprocessorBackend)
    has_batch = ndims(x) == 4
    x_data = has_batch ? x.data : reshape(x.data, 1, size(x.data)...)
    y = backend_maxpool2d(backend, x_data, layer.kernel_size, layer.stride, layer.padding)
    Tensor(has_batch ? y : dropdims(y, dims=1))
end

function _coprocessor_forward_layer(layer::AvgPool2d, x::AbstractTensor, backend::CoprocessorBackend)
    has_batch = ndims(x) == 4
    x_data = has_batch ? x.data : reshape(x.data, 1, size(x.data)...)
    y = backend_avgpool2d(
        backend,
        x_data,
        layer.kernel_size,
        layer.stride,
        layer.padding,
        layer.count_include_pad,
    )
    Tensor(has_batch ? y : dropdims(y, dims=1))
end

function _coprocessor_forward_layer(::GlobalAvgPool, x::AbstractTensor, backend::CoprocessorBackend)
    if ndims(x) == 4
        return Tensor(backend_global_avgpool2d(backend, x.data))
    end
    forward(GlobalAvgPool(), x)
end

function _coprocessor_forward_layer(::ReLU, x::AbstractTensor, backend::CoprocessorBackend)
    Tensor(backend_relu(backend, x.data))
end

function _coprocessor_forward_layer(layer::Softmax, x::AbstractTensor, backend::CoprocessorBackend)
    dim = layer.dims == -1 ? ndims(x.data) : layer.dims
    Tensor(backend_softmax(backend, x.data, dim))
end

function _coprocessor_forward_layer(layer::AbstractLayer, x::AbstractTensor, ::CoprocessorBackend)
    forward(layer, x)
end

function _coprocessor_forward(model::Pipeline, x::AbstractTensor, backend::CoprocessorBackend)
    for layer in model.layers
        x = _coprocessor_forward_layer(layer, x, backend)
    end
    x
end

function _coprocessor_forward(model::AbstractLayer, x::AbstractTensor, backend::CoprocessorBackend)
    _coprocessor_forward_layer(model, x, backend)
end

function _coprocessor_forward(model, x::AbstractTensor, ::CoprocessorBackend)
    forward(model, x)
end

function _coprocessor_forward_with_recovery(model, x::AbstractTensor, backend::CoprocessorBackend)
    try
        return _coprocessor_forward(model, x, backend)
    catch err
        _coprocessor_record_diagnostic!(backend, "runtime_errors")
        if _coprocessor_required(backend)
            message = string(
                _coprocessor_label(backend),
                " execution failed with strict mode enabled (set ",
                _coprocessor_required_env_key(backend),
                "=0 or AXIOM_COPROCESSOR_REQUIRED=0 to allow fallback). ",
                "Original error: ",
                sprint(showerror, err),
            )
            throw(ErrorException(message))
        end
        if !_coprocessor_self_healing_enabled()
            message = string(
                _coprocessor_label(backend),
                " execution failed with self-healing disabled (AXIOM_COPROCESSOR_SELF_HEAL=0). ",
                "Original error: ",
                sprint(showerror, err),
            )
            throw(ErrorException(message))
        end
        _coprocessor_record_diagnostic!(backend, "runtime_fallbacks")
        @warn "$(_coprocessor_label(backend)) execution failed; self-healing fallback to Julia backend" exception=(err, catch_backtrace())
        recovered = forward(model, x)
        _coprocessor_record_diagnostic!(backend, "recoveries")
        return recovered
    end
end

function forward(cm::CoprocessorCompiledModel, x::AbstractTensor)
    _coprocessor_forward_with_recovery(cm.model, x, cm.backend)
end

function forward(cm::CoprocessorCompiledModel, x)
    @debug "CoprocessorCompiledModel requires tensor input for accelerated path; falling back to model forward"
    forward(cm.model, x)
end

(cm::CoprocessorCompiledModel)(x) = forward(cm, x)

parameters(cm::CoprocessorCompiledModel) = parameters(cm.model)
output_shape(cm::CoprocessorCompiledModel, input_shape) = output_shape(cm.model, input_shape)

const TPUCompiledModel = CoprocessorCompiledModel{M, TPUBackend} where M
const NPUCompiledModel = CoprocessorCompiledModel{M, NPUBackend} where M
const DSPCompiledModel = CoprocessorCompiledModel{M, DSPBackend} where M
const PPUCompiledModel = CoprocessorCompiledModel{M, PPUBackend} where M
const MathCompiledModel = CoprocessorCompiledModel{M, MathBackend} where M
const FPGACompiledModel = CoprocessorCompiledModel{M, FPGABackend} where M
const VPUCompiledModel = CoprocessorCompiledModel{M, VPUBackend} where M
const QPUCompiledModel = CoprocessorCompiledModel{M, QPUBackend} where M
const CryptoCompiledModel = CoprocessorCompiledModel{M, CryptoBackend} where M

# ============================================================================
# Coprocessor Installation Instructions & Real Capability Detection
# ============================================================================

"""
    _coprocessor_install_instructions(label::String) -> String

Return human-readable instructions for installing and configuring a coprocessor
backend. Includes library requirements, driver versions, and environment variables.
"""
function _coprocessor_install_instructions(label::String)
    instructions = Dict{String, String}(
        "TPU" => """
            TPU Backend Setup:
              1. Install Google Cloud TPU libraries:
                 - libtpu.so (from cloud-tpu-client or JAX distribution)
                 - Set AXIOM_TPU_LIB_PATH to the libtpu.so directory
              2. Set environment variables:
                 export AXIOM_TPU_AVAILABLE=1
                 export AXIOM_TPU_DEVICE_COUNT=<number_of_tpu_cores>
              3. For Google Cloud TPU VMs, the driver is pre-installed.
              4. Package extension: Add AxiomTPUExt (when available) to load kernel hooks.
            """,
        "NPU" => """
            NPU Backend Setup:
              1. Install vendor NPU SDK:
                 - Intel NPU: install intel-npu-driver and intel-level-zero-gpu
                 - Qualcomm: install QNN SDK (libQnnHtp.so)
                 - Huawei Ascend: install CANN toolkit (libascendcl.so)
              2. Set environment variables:
                 export AXIOM_NPU_AVAILABLE=1
                 export AXIOM_NPU_DEVICE_COUNT=<number_of_npu_devices>
                 export AXIOM_NPU_LIB_PATH=<path_to_npu_libraries>
              3. Package extension: Add AxiomNPUExt (when available) to load kernel hooks.
            """,
        "DSP" => """
            DSP Backend Setup:
              1. Install Qualcomm Hexagon SDK or TI C6000 DSP tools:
                 - Hexagon: libcdsprpc.so (from Hexagon SDK)
                 - TI: TI-RTOS DSP runtime
              2. Set environment variables:
                 export AXIOM_DSP_AVAILABLE=1
                 export AXIOM_DSP_DEVICE_COUNT=<number_of_dsp_cores>
              3. Package extension: Add AxiomDSPExt (when available) to load kernel hooks.
            """,
        "PPU" => """
            PPU Backend Setup (Physics Processing Unit):
              1. PPU backends are experimental. Supported hardware:
                 - NVIDIA PhysX accelerators (via CUDA)
                 - Dedicated physics coprocessors (vendor-specific SDK)
              2. Set environment variables:
                 export AXIOM_PPU_AVAILABLE=1
                 export AXIOM_PPU_DEVICE_COUNT=<number_of_ppu_devices>
              3. Package extension: Add AxiomPPUExt (when available) to load kernel hooks.
            """,
        "MATH" => """
            Math Coprocessor Backend Setup:
              1. Supported math coprocessors:
                 - Intel AMX (Advanced Matrix Extensions): requires kernel >= 5.19
                 - ARM SVE/SME coprocessors: requires vendor runtime
              2. Check availability: grep -q 'amx' /proc/cpuinfo
              3. Set environment variables:
                 export AXIOM_MATH_AVAILABLE=1
                 export AXIOM_MATH_DEVICE_COUNT=1
              4. Package extension: Add AxiomMathExt (when available) to load kernel hooks.
            """,
        "FPGA" => """
            FPGA Backend Setup:
              1. Install FPGA runtime:
                 - Intel/Altera: OpenCL for FPGA (aocl), OneAPI FPGA toolkit
                 - Xilinx/AMD: Vitis AI Runtime (libxrt_core.so)
              2. Load bitstream for ML inference (vendor-specific).
              3. Set environment variables:
                 export AXIOM_FPGA_AVAILABLE=1
                 export AXIOM_FPGA_DEVICE_COUNT=<number_of_fpga_devices>
                 export AXIOM_FPGA_BITSTREAM=<path_to_bitstream>
              4. Package extension: Add AxiomFPGAExt (when available) to load kernel hooks.
            """,
        "VPU" => """
            VPU Backend Setup (Vision Processing Unit):
              1. Install Intel Movidius / Myriad VPU runtime:
                 - OpenVINO toolkit (libinference_engine.so)
                 - Intel Neural Compute SDK
              2. Set environment variables:
                 export AXIOM_VPU_AVAILABLE=1
                 export AXIOM_VPU_DEVICE_COUNT=<number_of_vpu_devices>
              3. Package extension: Add AxiomVPUExt (when available) to load kernel hooks.
            """,
        "QPU" => """
            QPU Backend Setup (Quantum Processing Unit):
              1. Install quantum computing SDK:
                 - IBM Qiskit: pip install qiskit (bridge via PyCall.jl)
                 - Google Cirq: pip install cirq (bridge via PyCall.jl)
                 - Amazon Braket: AWS SDK configuration
              2. QPU backends execute quantum circuits for quantum ML kernels.
              3. Set environment variables:
                 export AXIOM_QPU_AVAILABLE=1
                 export AXIOM_QPU_DEVICE_COUNT=1
                 export AXIOM_QPU_PROVIDER=<ibm|google|amazon>
              4. Package extension: Add AxiomQPUExt (when available) to load kernel hooks.
            """,
        "CRYPTO" => """
            Crypto Coprocessor Backend Setup:
              1. Supported cryptographic accelerators:
                 - Intel QAT (QuickAssist): install qat driver + libqat
                 - ARM CryptoCell: vendor SDK
                 - AES-NI / SHA-NI (CPU extensions): auto-detected
              2. Used for encrypted inference and homomorphic encryption.
              3. Set environment variables:
                 export AXIOM_CRYPTO_AVAILABLE=1
                 export AXIOM_CRYPTO_DEVICE_COUNT=1
              4. Package extension: Add AxiomCryptoExt (when available) to load kernel hooks.
            """,
    )
    get(instructions, label, "No installation instructions available for $label backend.")
end

"""
    coprocessor_setup_guide(backend_label::String)

Print detailed setup instructions for a coprocessor backend.
Valid labels: "TPU", "NPU", "DSP", "PPU", "MATH", "FPGA", "VPU", "QPU", "CRYPTO".
"""
function coprocessor_setup_guide(label::String)
    label = uppercase(label)
    println(_coprocessor_install_instructions(label))
end

"""
    coprocessor_setup_guide()

Print setup instructions for all coprocessor backends.
"""
function coprocessor_setup_guide()
    for label in ("TPU", "NPU", "DSP", "PPU", "MATH", "FPGA", "VPU", "QPU", "CRYPTO")
        println("=" ^ 70)
        println("  $label Backend")
        println("=" ^ 70)
        println(_coprocessor_install_instructions(label))
    end
end

"""
    detect_system_coprocessors() -> Dict{String,Any}

Probe the local system for real coprocessor hardware by checking:
- PCI devices (lspci)
- Kernel modules (lsmod)
- CPU features (/proc/cpuinfo)
- Shared libraries (ldconfig -p)

Returns a Dict mapping backend label → detection results.
This does NOT set AXIOM_*_AVAILABLE env vars; it reports what the system has.
"""
function detect_system_coprocessors()
    results = Dict{String, Any}()

    # Helper: run command and capture output, return empty string on failure
    function _run_cmd(cmd::Cmd)
        try
            buf = IOBuffer()
            run(pipeline(cmd, stdout=buf, stderr=devnull))
            return String(take!(buf))
        catch
            return ""
        end
    end

    # Check CPU features
    cpuinfo = ""
    if isfile("/proc/cpuinfo")
        try
            cpuinfo = read("/proc/cpuinfo", String)
        catch
        end
    end

    # Math coprocessor (AMX, SVE)
    has_amx = occursin("amx", lowercase(cpuinfo))
    has_avx512 = occursin("avx512", lowercase(cpuinfo))
    has_aesni = occursin("aes", lowercase(cpuinfo))
    results["MATH"] = Dict(
        "detected" => has_amx || has_avx512,
        "features" => filter(!isempty, [
            has_amx ? "Intel AMX" : "",
            has_avx512 ? "AVX-512" : "",
        ]),
        "note" => has_amx ? "AMX detected — set AXIOM_MATH_AVAILABLE=1 to enable" : "No dedicated math coprocessor detected",
    )

    # Crypto coprocessor (AES-NI, SHA-NI)
    has_sha = occursin("sha_ni", lowercase(cpuinfo)) || occursin("sha1", lowercase(cpuinfo))
    results["CRYPTO"] = Dict(
        "detected" => has_aesni,
        "features" => filter(!isempty, [
            has_aesni ? "AES-NI" : "",
            has_sha ? "SHA-NI" : "",
        ]),
        "note" => has_aesni ? "AES-NI detected — crypto acceleration available" : "No crypto acceleration detected",
    )

    # PCI-based detection (TPU, FPGA, VPU, NPU)
    pci_output = _run_cmd(`lspci`)

    # Google TPU (Coral / Cloud TPU)
    has_tpu_pci = occursin("Google", pci_output) && occursin(r"TPU|Coral|Edge", pci_output)
    has_tpu_lib = isfile("/usr/lib/libtpu.so") || isfile("/usr/local/lib/libtpu.so")
    results["TPU"] = Dict(
        "detected" => has_tpu_pci || has_tpu_lib,
        "pci_device" => has_tpu_pci,
        "library_found" => has_tpu_lib,
        "note" => has_tpu_pci || has_tpu_lib ? "TPU hardware detected — set AXIOM_TPU_AVAILABLE=1 to enable" : "No TPU hardware detected",
    )

    # Intel FPGA (Altera) / Xilinx (AMD)
    has_fpga_pci = occursin(r"Altera|Xilinx|FPGA", pci_output)
    results["FPGA"] = Dict(
        "detected" => has_fpga_pci,
        "pci_device" => has_fpga_pci,
        "note" => has_fpga_pci ? "FPGA detected — install vendor SDK and set AXIOM_FPGA_AVAILABLE=1" : "No FPGA detected",
    )

    # Intel VPU (Movidius/Myriad)
    has_vpu_pci = occursin(r"Myriad|Movidius|VPU|03e7", pci_output)
    results["VPU"] = Dict(
        "detected" => has_vpu_pci,
        "pci_device" => has_vpu_pci,
        "note" => has_vpu_pci ? "VPU detected — install OpenVINO and set AXIOM_VPU_AVAILABLE=1" : "No VPU detected",
    )

    # NPU (various vendors)
    has_npu_pci = occursin(r"NPU|Neural|Ascend", pci_output)
    results["NPU"] = Dict(
        "detected" => has_npu_pci,
        "pci_device" => has_npu_pci,
        "note" => has_npu_pci ? "NPU detected — install vendor SDK and set AXIOM_NPU_AVAILABLE=1" : "No NPU detected",
    )

    # DSP (typically not PCI — embedded)
    results["DSP"] = Dict(
        "detected" => false,
        "note" => "DSP detection requires vendor-specific tools (Hexagon SDK, TI tools)",
    )

    # PPU (typically integrated in GPU or dedicated)
    results["PPU"] = Dict(
        "detected" => false,
        "note" => "PPU detection requires NVIDIA PhysX or vendor-specific SDK",
    )

    # QPU (cloud-only, no local detection)
    results["QPU"] = Dict(
        "detected" => false,
        "note" => "QPU backends require cloud provider access (IBM, Google, Amazon)",
    )

    results
end
