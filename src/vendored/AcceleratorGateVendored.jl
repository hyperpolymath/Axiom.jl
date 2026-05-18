# SPDX-License-Identifier: MPL-2.0
# (PMPL-1.0-or-later preferred; MPL-2.0 required for Julia ecosystem)
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# ============================================================================
# VENDORED CODE — DO NOT EDIT BEHAVIOUR
# ============================================================================
#
# This module is a faithful, verbatim vendoring of selected definitions from:
#
#     hyperpolymath/AcceleratorGate.jl
#     src/AcceleratorGate.jl
#
# The original AcceleratorGate.jl is licensed MPL-2.0 (see SPDX header above,
# copied unchanged from the upstream source). All copyright remains with the
# original author.
#
# WHY THIS EXISTS:
#   AcceleratorGate.jl is currently an UNREGISTERED package. Depending on it
#   directly blocks Axiom.jl from being registered in the Julia General
#   registry. To decouple Axiom.jl from the registry-readiness of an external
#   unregistered dependency, the *specific* functions Axiom.jl consumes are
#   vendored here as an internal module.
#
# WHAT IS VENDORED:
#   Axiom.jl's src/backends/abstract.jl imports exactly 7 symbols:
#     select_backend, fits_on_device, estimate_cost,
#     DeviceCapabilities, device_capabilities, PlatformInfo, detect_platform
#   plus every helper / constant / type they transitively require. All such
#   code below is copied REAL implementation from the upstream single-file
#   source — nothing is stubbed, faked, or invented. The only adaptation is
#   the removal of upstream code NOT in the transitive closure of these 7
#   symbols (the module otherwise compiles standalone with identical
#   behaviour for the vendored entry points).
#
# RECONCILIATION NOTE (backend type hierarchy):
#   AcceleratorGate's selection logic is typed against ITS OWN backend type
#   hierarchy (AcceleratorGate.AbstractBackend with JuliaBackend, CUDABackend,
#   etc.). Axiom.jl defines a SEPARATE, unrelated `Axiom.AbstractBackend`
#   hierarchy in src/backends/abstract.jl. These are deliberately NOT merged:
#   `select_backend`/`detect_gpu`/`detect_coprocessor` construct and return
#   AcceleratorGate's own backend structs, so merging the hierarchies would
#   change behaviour. This module therefore keeps AcceleratorGate's complete
#   backend hierarchy internal and self-contained, exactly as upstream. The
#   exported symbols behave identically to the upstream package.
#
# Upstream provenance: hyperpolymath/AcceleratorGate.jl src/AcceleratorGate.jl
# ============================================================================

module AcceleratorGateVendored

using Dates

export DeviceCapabilities, device_capabilities, fits_on_device,
       estimate_cost, select_backend,
       PlatformInfo, detect_platform

# ============================================================================
# Backend Type Hierarchy
# (verbatim from AcceleratorGate.jl — required transitively by select_backend)
# ============================================================================

abstract type AbstractBackend end

struct JuliaBackend <: AbstractBackend end
struct RustBackend <: AbstractBackend
    lib_path::String
end
struct ZigBackend <: AbstractBackend
    lib_path::String
end
struct CUDABackend <: AbstractBackend
    device::Int
end
struct ROCmBackend <: AbstractBackend
    device::Int
end
struct MetalBackend <: AbstractBackend
    device::Int
end
struct TPUBackend <: AbstractBackend
    device::Int
end
struct NPUBackend <: AbstractBackend
    device::Int
end
struct DSPBackend <: AbstractBackend
    device::Int
end
struct PPUBackend <: AbstractBackend
    device::Int
end
struct MathBackend <: AbstractBackend
    device::Int
end
struct FPGABackend <: AbstractBackend
    device::Int
end
struct VPUBackend <: AbstractBackend
    device::Int
end
struct QPUBackend <: AbstractBackend
    device::Int
end
struct CryptoBackend <: AbstractBackend
    device::Int
end

# ============================================================================
# Environment Helpers
# (verbatim — required by accelerator availability / count predicates)
# ============================================================================

function _backend_env_available(key::String)
    raw = lowercase(strip(get(ENV, key, "")))
    isempty(raw) && return nothing
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    nothing
end

function _backend_env_count(available_key::String, count_key::String)
    forced = _backend_env_available(available_key)
    forced === false && return 0
    raw = strip(get(ENV, count_key, ""))
    isempty(raw) && return nothing
    parsed = tryparse(Int, raw)
    parsed === nothing ? nothing : max(parsed, 0)
end

_accelerator_env_flag(key) = let f = _backend_env_available(key); f === nothing ? false : f end
function _accelerator_env_count(avail_key, count_key)
    c = _backend_env_count(avail_key, count_key)
    c !== nothing && return c
    _accelerator_env_flag(avail_key) || return 0
    1
end

# ============================================================================
# Accelerator Availability
# (verbatim — required by detect_gpu / detect_coprocessor / select_backend)
# ============================================================================

cuda_available()   = _accelerator_env_flag("AXIOM_CUDA_AVAILABLE")
rocm_available()   = _accelerator_env_flag("AXIOM_ROCM_AVAILABLE")
metal_available()  = _accelerator_env_flag("AXIOM_METAL_AVAILABLE")
tpu_available()    = _accelerator_env_flag("AXIOM_TPU_AVAILABLE")
npu_available()    = _accelerator_env_flag("AXIOM_NPU_AVAILABLE")
dsp_available()    = _accelerator_env_flag("AXIOM_DSP_AVAILABLE")
ppu_available()    = _accelerator_env_flag("AXIOM_PPU_AVAILABLE")
math_available()   = _accelerator_env_flag("AXIOM_MATH_AVAILABLE")
fpga_available()   = _accelerator_env_flag("AXIOM_FPGA_AVAILABLE")
vpu_available()    = _accelerator_env_flag("AXIOM_VPU_AVAILABLE")
qpu_available()    = _accelerator_env_flag("AXIOM_QPU_AVAILABLE")
crypto_available() = _accelerator_env_flag("AXIOM_CRYPTO_AVAILABLE")

cuda_device_count()   = _accelerator_env_count("AXIOM_CUDA_AVAILABLE", "AXIOM_CUDA_DEVICE_COUNT")
rocm_device_count()   = _accelerator_env_count("AXIOM_ROCM_AVAILABLE", "AXIOM_ROCM_DEVICE_COUNT")
metal_device_count()  = _accelerator_env_count("AXIOM_METAL_AVAILABLE", "AXIOM_METAL_DEVICE_COUNT")
tpu_device_count()    = _accelerator_env_count("AXIOM_TPU_AVAILABLE", "AXIOM_TPU_DEVICE_COUNT")
npu_device_count()    = _accelerator_env_count("AXIOM_NPU_AVAILABLE", "AXIOM_NPU_DEVICE_COUNT")
dsp_device_count()    = _accelerator_env_count("AXIOM_DSP_AVAILABLE", "AXIOM_DSP_DEVICE_COUNT")
ppu_device_count()    = _accelerator_env_count("AXIOM_PPU_AVAILABLE", "AXIOM_PPU_DEVICE_COUNT")
math_device_count()   = _accelerator_env_count("AXIOM_MATH_AVAILABLE", "AXIOM_MATH_DEVICE_COUNT")
fpga_device_count()   = _accelerator_env_count("AXIOM_FPGA_AVAILABLE", "AXIOM_FPGA_DEVICE_COUNT")
vpu_device_count()    = _accelerator_env_count("AXIOM_VPU_AVAILABLE", "AXIOM_VPU_DEVICE_COUNT")
qpu_device_count()    = _accelerator_env_count("AXIOM_QPU_AVAILABLE", "AXIOM_QPU_DEVICE_COUNT")
crypto_device_count() = _accelerator_env_count("AXIOM_CRYPTO_AVAILABLE", "AXIOM_CRYPTO_DEVICE_COUNT")

# ============================================================================
# Platform Detection
# (verbatim — PlatformInfo + detect_platform and their helpers)
# ============================================================================

"""
    PlatformInfo

Describes the host platform's operating system, CPU architecture, and
environment class (mobile, embedded, server). Used by `select_backend` and
`_arch_compatible` to make platform-aware dispatch decisions.
"""
struct PlatformInfo
    os::Symbol              # :linux, :macos, :windows, :freebsd, :openbsd, :minix, :android, :ios, :unknown
    arch::Symbol            # :x86_64, :aarch64, :arm, :riscv64, :powerpc64, :mips, :unknown
    is_mobile::Bool         # iOS or Android
    is_embedded::Bool       # MINIX, bare metal, RTOS-like environments
    is_server::Bool         # Detected server environment (many cores / large RAM)
    julia_version::VersionNumber
    word_size::Int          # 32 or 64
    endianness::Symbol      # :little or :big
end

"""
    detect_platform() -> PlatformInfo

Probe the current host and return a `PlatformInfo` describing the OS,
architecture, and environment class.
"""
function detect_platform()::PlatformInfo
    os = _detect_os()
    arch = Sys.ARCH
    is_mobile = os in (:android, :ios)
    is_embedded = os === :minix || _is_embedded_env()
    is_server = _is_server_env()
    endian = ENDIAN_BOM == 0x04030201 ? :little : :big
    PlatformInfo(os, arch, is_mobile, is_embedded, is_server,
                 VERSION, Sys.WORD_SIZE, endian)
end

"""
    _detect_os() -> Symbol

Determine the operating system, distinguishing Android from Linux and iOS
from macOS where possible.
"""
function _detect_os()::Symbol
    Sys.islinux()   && return _check_android() ? :android : :linux
    Sys.isapple()   && return _check_ios() ? :ios : :macos
    Sys.iswindows()  && return :windows
    Sys.isbsd()      && return :freebsd
    # Attempt uname for exotic OSes (MINIX, OpenBSD, etc.)
    try
        uname = lowercase(strip(read(`uname -s`, String)))
        contains(uname, "minix")   && return :minix
        contains(uname, "openbsd") && return :openbsd
    catch
        # uname unavailable (Windows, sandboxed, etc.) — already handled above
    end
    :unknown
end

"""
    _check_android() -> Bool

Heuristic: Android sets `ANDROID_ROOT` and ships `/system/app`.
"""
function _check_android()::Bool
    haskey(ENV, "ANDROID_ROOT") || isdir("/system/app")
end

"""
    _check_ios() -> Bool

Rough heuristic: Apple + aarch64 + no /usr/local (macOS Homebrew path) likely
indicates an iOS/iPadOS/Catalyst environment.
"""
function _check_ios()::Bool
    Sys.isapple() && Sys.ARCH === :aarch64 && !isdir("/usr/local")
end

"""
    _is_embedded_env() -> Bool

Heuristic for resource-constrained embedded targets: very few CPU threads
and less than 512 MiB total RAM.
"""
function _is_embedded_env()::Bool
    Sys.CPU_THREADS <= 2 && Sys.total_memory() < 512 * 1024 * 1024
end

"""
    _is_server_env() -> Bool

Heuristic for server-class hardware: 16+ CPU threads or 32+ GiB RAM.
"""
function _is_server_env()::Bool
    Sys.CPU_THREADS >= 16 || Sys.total_memory() > 32 * Int64(1024)^3
end

# ============================================================================
# Architecture-Aware Coprocessor Compatibility
# (verbatim — required by select_backend / detect_gpu / detect_coprocessor)
# ============================================================================

"""
    _arch_compatible(backend::AbstractBackend, arch::Symbol) -> Bool

Check whether `backend` is compatible with CPU architecture `arch`.
Some accelerators are only reachable on certain host architectures
(e.g. CUDA requires x86_64 or aarch64 for Jetson/Grace).
"""
function _arch_compatible(backend::AbstractBackend, arch::Symbol)::Bool
    # CUDA requires x86_64 or aarch64 (Jetson / Grace Hopper)
    backend isa CUDABackend  && return arch in (:x86_64, :aarch64)
    # Metal requires Apple Silicon (aarch64 on macOS)
    backend isa MetalBackend && return arch === :aarch64
    # ROCm requires x86_64
    backend isa ROCmBackend  && return arch === :x86_64
    # VPU / DSP are common on ARM SoCs (mobile, embedded)
    backend isa VPUBackend   && return arch in (:aarch64, :arm)
    backend isa DSPBackend   && return arch in (:aarch64, :arm, :x86_64)
    # NPU available on modern ARM SoCs and some Intel x86
    backend isa NPUBackend   && return arch in (:aarch64, :arm, :x86_64)
    # QPU is cloud-accessible from any architecture
    backend isa QPUBackend   && return true
    # FPGA typically via PCIe on x86 / ARM servers
    backend isa FPGABackend  && return arch in (:x86_64, :aarch64)
    # TPU — cloud-accessible or edge TPU (USB/PCIe) on x86/ARM
    backend isa TPUBackend   && return arch in (:x86_64, :aarch64)
    # Julia, Rust, Zig, Math, Crypto, PPU — always compatible
    true
end

# ============================================================================
# Auto-Detection
# (verbatim — detect_gpu / detect_coprocessor required by select_backend)
# ============================================================================

function detect_gpu()
    platform = detect_platform()
    if cuda_available() && cuda_device_count() > 0
        b = CUDABackend(0)
        _arch_compatible(b, platform.arch) && return b
    end
    if rocm_available() && rocm_device_count() > 0
        b = ROCmBackend(0)
        _arch_compatible(b, platform.arch) && return b
    end
    if metal_available() && metal_device_count() > 0
        b = MetalBackend(0)
        _arch_compatible(b, platform.arch) && return b
    end
    nothing
end

function detect_coprocessor()
    platform = detect_platform()
    for (avail_fn, count_fn, ctor) in (
        (tpu_available,    tpu_device_count,    TPUBackend),
        (npu_available,    npu_device_count,    NPUBackend),
        (vpu_available,    vpu_device_count,    VPUBackend),
        (qpu_available,    qpu_device_count,    QPUBackend),
        (ppu_available,    ppu_device_count,    PPUBackend),
        (math_available,   math_device_count,   MathBackend),
        (crypto_available, crypto_device_count, CryptoBackend),
        (fpga_available,   fpga_device_count,   FPGABackend),
        (dsp_available,    dsp_device_count,    DSPBackend),
    )
        if avail_fn() && count_fn() > 0
            b = ctor(0)
            _arch_compatible(b, platform.arch) && return b
        end
    end
    nothing
end

# ============================================================================
# Device Capabilities
# (verbatim — DeviceCapabilities + device_capabilities + fits_on_device)
# ============================================================================

"""
    DeviceCapabilities

Describes the hardware capabilities of a specific backend device, including
compute resources, memory, precision support, and vendor information.
Extensions should override `device_capabilities` to return populated instances.
"""
struct DeviceCapabilities
    backend::AbstractBackend
    compute_units::Int          # cores/SMs/CUs/slices
    clock_mhz::Int              # clock speed
    memory_bytes::Int64         # total device memory
    memory_available::Int64     # currently available memory
    max_workgroup_size::Int     # max threads per workgroup
    supports_f64::Bool          # double precision support
    supports_f16::Bool          # half precision support
    supports_int8::Bool         # int8 quantized ops
    vendor::String              # "NVIDIA", "AMD", "Apple", "Intel", "Google", "Qualcomm", etc.
    driver_version::String      # driver/SDK version
end

"""
    device_capabilities(b::AbstractBackend) -> Union{Nothing, DeviceCapabilities}

Query device capabilities for a given backend. Returns `nothing` by default;
backend extensions should override this with real hardware queries.
"""
function device_capabilities(b::AbstractBackend)::Union{Nothing, DeviceCapabilities}
    # Default: return nothing (extensions override)
    nothing
end

"""
    fits_on_device(b::AbstractBackend, required_memory::Int64) -> Bool

Check whether a workload requiring `required_memory` bytes can fit on the
device associated with backend `b`.
"""
function fits_on_device(b::AbstractBackend, required_memory::Int64)::Bool
    caps = device_capabilities(b)
    caps === nothing && return false
    caps.memory_available >= required_memory
end

"""
    estimate_cost(b::AbstractBackend, op::Symbol, data_size::Int) -> Float64

Estimate the relative cost of performing operation `op` on `data_size` elements
using backend `b`. Lower values are better. Returns `Inf` for backends that
have not registered cost models (extensions override with real estimates).
"""
function estimate_cost(b::AbstractBackend, op::Symbol, data_size::Int)::Float64
    # Default: return Inf for non-Julia backends (extensions override with real estimates)
    b isa JuliaBackend && return Float64(data_size)  # CPU cost ~ data size
    Inf
end

# ============================================================================
# Resource-Aware Backend Selection
# (verbatim — select_backend, the principal entry point)
# ============================================================================

"""
    select_backend(op, data_size; required_memory=0, prefer_precision=:f64, exclude=DataType[]) -> AbstractBackend

Auto-select the best backend for a given workload by considering available
hardware, memory requirements, precision needs, and estimated operation cost.
"""
function select_backend(op::Symbol, data_size::Int;
                        required_memory::Int64=Int64(0),
                        prefer_precision::Symbol=:f64,
                        exclude::Vector{DataType}=DataType[])::AbstractBackend
    platform = detect_platform()
    candidates = AbstractBackend[]

    # ---- Platform-aware candidate gathering --------------------------------

    if platform.is_mobile
        # Mobile: prefer low-power accelerators (NPU > VPU), avoid GPU for battery
        for (avail_fn, count_fn, ctor) in (
            (npu_available, npu_device_count, NPUBackend),
            (vpu_available, vpu_device_count, VPUBackend),
            (dsp_available, dsp_device_count, DSPBackend),
        )
            if avail_fn() && count_fn() > 0
                b = ctor(0)
                _arch_compatible(b, platform.arch) && !(typeof(b) in exclude) && push!(candidates, b)
            end
        end
    elseif platform.is_embedded
        # Embedded: prefer FPGA/DSP (low memory footprint), check strictly
        for (avail_fn, count_fn, ctor) in (
            (fpga_available, fpga_device_count, FPGABackend),
            (dsp_available,  dsp_device_count,  DSPBackend),
        )
            if avail_fn() && count_fn() > 0
                b = ctor(0)
                _arch_compatible(b, platform.arch) && !(typeof(b) in exclude) && push!(candidates, b)
            end
        end
    else
        # Desktop / Server: try GPU first for maximum throughput
        gpu = detect_gpu()
        gpu !== nothing && !(typeof(gpu) in exclude) && push!(candidates, gpu)

        # Then coprocessors (server environments get all of them)
        coproc = detect_coprocessor()
        coproc !== nothing && !(typeof(coproc) in exclude) && push!(candidates, coproc)
    end

    # Always include Julia fallback
    push!(candidates, JuliaBackend())

    # ---- Filter by architecture compatibility ------------------------------
    filter!(b -> _arch_compatible(b, platform.arch), candidates)

    # ---- Filter by memory requirement --------------------------------------
    if required_memory > 0
        filter!(b -> fits_on_device(b, required_memory), candidates)
    end

    # ---- Filter by precision requirement -----------------------------------
    if prefer_precision == :f64
        filter!(b -> begin
            caps = device_capabilities(b)
            caps === nothing || caps.supports_f64
        end, candidates)
    end

    # ---- Select lowest cost ------------------------------------------------
    best = JuliaBackend()
    best_cost = estimate_cost(best, op, data_size)
    for b in candidates
        c = estimate_cost(b, op, data_size)
        if c < best_cost
            best = b
            best_cost = c
        end
    end
    best
end

end # module AcceleratorGateVendored
