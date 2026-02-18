# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Backend Abstraction
#
# Interface for different computation backends.

using Libdl

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
    RustBackend

High-performance Rust implementation.
"""
struct RustBackend <: AbstractBackend
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
    CryptoBackend

Cryptographic coprocessor backend target.
"""
struct CryptoBackend <: AbstractBackend
    device::Int
end

"""
    FPGABackend

FPGA accelerator backend target.
"""
struct FPGABackend <: AbstractBackend
    device::Int
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

const CoprocessorBackend = Union{TPUBackend, NPUBackend, DSPBackend, PPUBackend, MathBackend, CryptoBackend, FPGABackend}

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
    elseif backend isa CryptoBackend
        return "CRYPTO"
    elseif backend isa FPGABackend
        return "FPGA"
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
    elseif backend isa CryptoBackend
        return "AXIOM_CRYPTO_REQUIRED"
    elseif backend isa FPGABackend
        return "AXIOM_FPGA_REQUIRED"
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
    "crypto" => Dict(
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
    elseif backend isa CryptoBackend
        return "crypto"
    elseif backend isa FPGABackend
        return "fpga"
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
    for key in ("tpu", "npu", "dsp", "ppu", "math", "crypto", "fpga")
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
        if _coprocessor_required(backend)
            throw(ErrorException(string(
                _coprocessor_label(backend),
                " extension hook not loaded for `",
                $hook_name,
                "` while strict mode is enabled (set ",
                _coprocessor_required_env_key(backend),
                "=0 or AXIOM_COPROCESSOR_REQUIRED=0 to allow fallback)."
            )))
        end
        @warn string(
            _coprocessor_label(backend),
            " extension hook not loaded for `",
            $hook_name,
            "`, falling back to Julia backend"
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

function fold_batchnorm(model)
    # For pipelines, the optimization is already handled in optimize_pipeline
    model
end

function fold_constants(model)
    # No-op for most models - constants are evaluated at definition time in Julia
    model
end

function eliminate_dead_code(model)
    # In Julia, unused code is typically not compiled anyway
    model
end

function apply_aggressive_optimizations(model, target::CompilationTarget)
    # Aggressive optimizations that may affect numerical precision
    # - Fuse more operations
    # - Use approximate math functions
    # - Enable auto-vectorization hints
    model
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
"""
struct MixedPrecisionWrapper{M}
    model::M
    # Master weights stored in Float32
    master_weights::Dict{Symbol, Any}
end

function MixedPrecisionWrapper(model)
    # Store copy of original Float32 weights
    master = Dict{Symbol, Any}()
    params = parameters(model)
    for (name, param) in pairs(params)
        if param isa AbstractArray{Float32}
            master[name] = copy(param)
        end
    end
    MixedPrecisionWrapper(model, master)
end

function forward(mp::MixedPrecisionWrapper, x)
    # Convert input to Float16 for forward pass.
    x_f16 = if x isa AbstractTensor
        Tensor(Float16.(x.data))
    else
        Float16.(x)
    end

    # Temporarily convert model weights to Float16
    params = parameters(mp.model)
    for (name, param) in pairs(params)
        if param isa AbstractArray{Float32}
            setfield!(mp.model, name, Float16.(param))
        end
    end

    # Forward pass in Float16
    y = forward(mp.model, x_f16)

    # Restore Float32 weights
    for (name, master_param) in mp.master_weights
        setfield!(mp.model, name, master_param)
    end

    # Return result as Float32, preserving tensor wrappers when available.
    if y isa AbstractTensor
        return Tensor(Float32.(y.data))
    end
    Float32.(y)
end

(mp::MixedPrecisionWrapper)(x) = forward(mp, x)

parameters(mp::MixedPrecisionWrapper) = parameters(mp.model)
output_shape(mp::MixedPrecisionWrapper, input_shape) = output_shape(mp.model, input_shape)

function compile_to_backend(model, backend::JuliaBackend)
    # Julia backend - just return the model
    model
end

function compile_to_backend(model, backend::RustBackend)
    @info "Compiling to Rust backend..."

    # Verify Rust library exists
    if !isfile(backend.lib_path)
        @warn "Rust library not found at $(backend.lib_path), falling back to Julia backend"
        return model
    end

    # Wrap model for Rust execution
    RustCompiledModel(model, backend)
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

function compile_to_backend(model, backend::CryptoBackend)
    _compile_coprocessor(model, backend, crypto_available(), crypto_device_count(), "CRYPTO")
end

function compile_to_backend(model, backend::FPGABackend)
    _compile_coprocessor(model, backend, fpga_available(), fpga_device_count(), "FPGA")
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
    crypto_available() -> Bool

Check if cryptographic coprocessor backend is available.
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
    crypto_device_count() -> Int

Get number of available cryptographic coprocessor devices.
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
    if crypto_available() && crypto_device_count() > 0
        return CryptoBackend(0)
    end
    if fpga_available() && fpga_device_count() > 0
        return FPGABackend(0)
    end
    if dsp_available() && dsp_device_count() > 0
        return DSPBackend(0)
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
        if first_arg == backend_type
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
including environment-based availability, device counts, and kernel hook status.
"""
function coprocessor_capability_report()
    selected = detect_coprocessor()
    backends = Dict{String, Any}()
    strategy = [
        ("TPU", TPUBackend, tpu_available, tpu_device_count),
        ("NPU", NPUBackend, npu_available, npu_device_count),
        ("PPU", PPUBackend, ppu_available, ppu_device_count),
        ("MATH", MathBackend, math_available, math_device_count),
        ("CRYPTO", CryptoBackend, crypto_available, crypto_device_count),
        ("FPGA", FPGABackend, fpga_available, fpga_device_count),
        ("DSP", DSPBackend, dsp_available, dsp_device_count),
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
        "strategy_order" => ["TPU", "NPU", "PPU", "MATH", "CRYPTO", "FPGA", "DSP"],
        "selected_backend" => selected === nothing ? nothing : string(typeof(selected)),
        "runtime_diagnostics" => coprocessor_runtime_diagnostics(),
        "backends" => backends,
    )
end

"""
    select_device!(backend::Union{TPUBackend,NPUBackend,DSPBackend,PPUBackend,MathBackend,CryptoBackend,FPGABackend}, device::Int)

Return a backend handle for the selected coprocessor device.
"""
select_device!(::TPUBackend, device::Int) = TPUBackend(device)
select_device!(::NPUBackend, device::Int) = NPUBackend(device)
select_device!(::DSPBackend, device::Int) = DSPBackend(device)
select_device!(::PPUBackend, device::Int) = PPUBackend(device)
select_device!(::MathBackend, device::Int) = MathBackend(device)
select_device!(::CryptoBackend, device::Int) = CryptoBackend(device)
select_device!(::FPGABackend, device::Int) = FPGABackend(device)

"""
    RustCompiledModel

Model wrapper that dispatches operations to Rust backend.
"""
struct RustCompiledModel{M}
    model::M
    backend::RustBackend
    lib_handle::Ptr{Nothing}
end

function RustCompiledModel(model, backend::RustBackend)
    # Load the Rust shared library
    lib_handle = try
        Libdl.dlopen(backend.lib_path)
    catch e
        @warn "Failed to load Rust library: $e"
        Ptr{Nothing}()
    end

    RustCompiledModel(model, backend, lib_handle)
end

function forward(rm::RustCompiledModel, x)
    if rm.lib_handle == Ptr{Nothing}()
        # Fallback to Julia implementation
        return forward(rm.model, x)
    end

    # Dispatch to Rust backend based on layer type
    rust_forward(rm.model, x, rm.lib_handle)
end

(rm::RustCompiledModel)(x) = forward(rm, x)

function rust_forward(model, x, lib_handle)
    # Default: fall back to Julia for unsupported layers
    forward(model, x)
end

function rust_forward(model::Dense, x, lib_handle)
    # Call Rust matmul if available
    matmul_fn = Libdl.dlsym(lib_handle, :axiom_matmul; throw_error=false)
    if matmul_fn != C_NULL
        # Would call: ccall(matmul_fn, ...)
        # For now, fall back to Julia
        forward(model, x)
    else
        forward(model, x)
    end
end

parameters(rm::RustCompiledModel) = parameters(rm.model)
output_shape(rm::RustCompiledModel, input_shape) = output_shape(rm.model, input_shape)

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

Wrapper for non-GPU accelerator backends (TPU/NPU/PPU/MATH/CRYPTO/DSP/FPGA).
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
const CryptoCompiledModel = CoprocessorCompiledModel{M, CryptoBackend} where M
const FPGACompiledModel = CoprocessorCompiledModel{M, FPGABackend} where M
