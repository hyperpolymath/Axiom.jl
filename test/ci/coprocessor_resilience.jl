# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Random
using Axiom

Random.seed!(0xA710)

function with_env(overrides::Dict{String, String}, f::Function)
    previous = Dict{String, Union{String, Nothing}}()
    for key in keys(overrides)
        previous[key] = get(ENV, key, nothing)
    end

    try
        for (key, value) in overrides
            ENV[key] = value
        end
        return f()
    finally
        for (key, value) in previous
            if value === nothing
                delete!(ENV, key)
            else
                ENV[key] = value
            end
        end
    end
end

with_env(f::Function, overrides::Dict{String, String}) = with_env(overrides, f)

const COPROCESSOR_FAILURE_MODE = Dict{DataType, Bool}(
    Axiom.TPUBackend => false,
    Axiom.NPUBackend => false,
    Axiom.DSPBackend => false,
    Axiom.PPUBackend => false,
    Axiom.MathBackend => false,
    Axiom.FPGABackend => false,
)

function backend_key(backend)
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
    end
    error("Unexpected backend type: $(typeof(backend))")
end

function Axiom.backend_coprocessor_matmul(
    backend::Union{Axiom.TPUBackend, Axiom.NPUBackend, Axiom.DSPBackend, Axiom.PPUBackend, Axiom.MathBackend, Axiom.FPGABackend},
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    if get(COPROCESSOR_FAILURE_MODE, typeof(backend), false)
        error("Injected coprocessor matmul failure for $(typeof(backend))")
    end
    A * B
end

function Axiom.backend_coprocessor_relu(
    backend::Union{Axiom.TPUBackend, Axiom.NPUBackend, Axiom.DSPBackend, Axiom.PPUBackend, Axiom.MathBackend, Axiom.FPGABackend},
    x::AbstractArray{Float32},
)
    max.(x, 0f0)
end

function Axiom.backend_coprocessor_softmax(
    backend::Union{Axiom.TPUBackend, Axiom.NPUBackend, Axiom.DSPBackend, Axiom.PPUBackend, Axiom.MathBackend, Axiom.FPGABackend},
    x::AbstractArray{Float32},
    dim::Int,
)
    Axiom.softmax(x, dims=dim)
end

function reset_coprocessor_failure_mode!()
    for backend_type in keys(COPROCESSOR_FAILURE_MODE)
        COPROCESSOR_FAILURE_MODE[backend_type] = false
    end
end

@testset "Coprocessor self-healing fallback and diagnostics" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 5, 6))
    cpu = model(x).data

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_NPU_AVAILABLE" => "1",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_PPU_AVAILABLE" => "1",
        "AXIOM_MATH_AVAILABLE" => "1",
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_NPU_DEVICE_COUNT" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_PPU_DEVICE_COUNT" => "1",
        "AXIOM_MATH_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
        "AXIOM_COPROCESSOR_SELF_HEAL" => "1",
    )) do
        for backend in (TPUBackend(0), NPUBackend(0), DSPBackend(0), PPUBackend(0), MathBackend(0), FPGABackend(0))
            reset_coprocessor_runtime_diagnostics!()
            reset_coprocessor_failure_mode!()
            COPROCESSOR_FAILURE_MODE[typeof(backend)] = true

            compiled = compile(model, backend=backend, verify=false, optimize=:none)
            @test compiled !== model
            y = compiled(x).data
            @test isapprox(y, cpu; atol=1f-5, rtol=1f-5)

            diag = coprocessor_runtime_diagnostics()["backends"][backend_key(backend)]
            @test diag["runtime_errors"] >= 1
            @test diag["runtime_fallbacks"] >= 1
            @test diag["recoveries"] >= 1
        end
    end

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_COPROCESSOR_SELF_HEAL" => "0",
    )) do
        reset_coprocessor_runtime_diagnostics!()
        reset_coprocessor_failure_mode!()
        COPROCESSOR_FAILURE_MODE[Axiom.TPUBackend] = true

        compiled = compile(model, backend=TPUBackend(0), verify=false, optimize=:none)
        @test compiled !== model

        caught = nothing
        try
            compiled(x)
        catch err
            caught = err
        end

        @test caught isa ErrorException
        @test occursin("AXIOM_COPROCESSOR_SELF_HEAL=0", sprint(showerror, caught))

        diag = coprocessor_runtime_diagnostics()["backends"]["tpu"]
        @test diag["runtime_errors"] >= 1
        @test diag["runtime_fallbacks"] == 0
        @test diag["recoveries"] == 0
    end
end

@testset "Coprocessor compile fallback diagnostics" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "0",
        "AXIOM_NPU_AVAILABLE" => "0",
        "AXIOM_DSP_AVAILABLE" => "0",
        "AXIOM_PPU_AVAILABLE" => "0",
        "AXIOM_MATH_AVAILABLE" => "0",
        "AXIOM_FPGA_AVAILABLE" => "0",
        "AXIOM_TPU_DEVICE_COUNT" => "0",
        "AXIOM_NPU_DEVICE_COUNT" => "0",
        "AXIOM_DSP_DEVICE_COUNT" => "0",
        "AXIOM_PPU_DEVICE_COUNT" => "0",
        "AXIOM_MATH_DEVICE_COUNT" => "0",
        "AXIOM_FPGA_DEVICE_COUNT" => "0",
    )) do
        reset_coprocessor_runtime_diagnostics!()

        @test compile(model, backend=TPUBackend(0), verify=false, optimize=:none) === model
        @test compile(model, backend=NPUBackend(0), verify=false, optimize=:none) === model
        @test compile(model, backend=DSPBackend(0), verify=false, optimize=:none) === model
        @test compile(model, backend=PPUBackend(0), verify=false, optimize=:none) === model
        @test compile(model, backend=MathBackend(0), verify=false, optimize=:none) === model
        @test compile(model, backend=FPGABackend(0), verify=false, optimize=:none) === model

        diag = coprocessor_runtime_diagnostics()["backends"]
        @test diag["tpu"]["compile_fallbacks"] == 1
        @test diag["npu"]["compile_fallbacks"] == 1
        @test diag["dsp"]["compile_fallbacks"] == 1
        @test diag["ppu"]["compile_fallbacks"] == 1
        @test diag["math"]["compile_fallbacks"] == 1
        @test diag["fpga"]["compile_fallbacks"] == 1
    end

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_NPU_AVAILABLE" => "1",
        "AXIOM_NPU_DEVICE_COUNT" => "1",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_PPU_AVAILABLE" => "1",
        "AXIOM_PPU_DEVICE_COUNT" => "1",
        "AXIOM_MATH_AVAILABLE" => "1",
        "AXIOM_MATH_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
    )) do
        reset_coprocessor_runtime_diagnostics!()

        @test compile(model, backend=TPUBackend(8), verify=false, optimize=:none) === model
        @test compile(model, backend=NPUBackend(8), verify=false, optimize=:none) === model
        @test compile(model, backend=DSPBackend(8), verify=false, optimize=:none) === model
        @test compile(model, backend=PPUBackend(8), verify=false, optimize=:none) === model
        @test compile(model, backend=MathBackend(8), verify=false, optimize=:none) === model
        @test compile(model, backend=FPGABackend(8), verify=false, optimize=:none) === model

        diag = coprocessor_runtime_diagnostics()["backends"]
        @test diag["tpu"]["compile_fallbacks"] == 1
        @test diag["npu"]["compile_fallbacks"] == 1
        @test diag["dsp"]["compile_fallbacks"] == 1
        @test diag["ppu"]["compile_fallbacks"] == 1
        @test diag["math"]["compile_fallbacks"] == 1
        @test diag["fpga"]["compile_fallbacks"] == 1
    end
end
