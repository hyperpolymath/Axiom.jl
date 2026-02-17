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

@testset "DSP strict mode compile gate" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )

    with_env(Dict(
        "AXIOM_DSP_AVAILABLE" => "0",
        "AXIOM_DSP_DEVICE_COUNT" => "0",
        "AXIOM_DSP_REQUIRED" => "1",
    )) do
        err = nothing
        try
            compile(model, backend = DSPBackend(0), verify = false, optimize = :none)
        catch caught
            err = caught
        end
        @test err isa ErrorException
        @test occursin("AXIOM_DSP_REQUIRED", sprint(showerror, err))
    end
end

@testset "DSP strict mode runtime hook gate" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 5, 6))

    with_env(Dict(
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_DSP_REQUIRED" => "1",
    )) do
        compiled = compile(model, backend = DSPBackend(0), verify = false, optimize = :none)
        @test compiled isa Axiom.CoprocessorCompiledModel

        err = nothing
        try
            compiled(x)
        catch caught
            err = caught
        end

        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("AXIOM_DSP_REQUIRED", msg)
        @test occursin("strict mode enabled", msg)
    end
end

# Install minimal DSP hook overrides to demonstrate strict-mode success path.
@eval begin
    function Axiom.backend_coprocessor_matmul(
        backend::Axiom.DSPBackend,
        A::AbstractMatrix{Float32},
        B::AbstractMatrix{Float32},
    )
        A * B
    end

    function Axiom.backend_coprocessor_relu(
        backend::Axiom.DSPBackend,
        x::AbstractArray{Float32},
    )
        max.(x, 0f0)
    end

    function Axiom.backend_coprocessor_softmax(
        backend::Axiom.DSPBackend,
        x::AbstractArray{Float32},
        dim::Int,
    )
        Axiom.softmax(x, dims = dim)
    end
end

@testset "DSP strict mode with hook overrides" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 5, 6))
    cpu = model(x).data

    with_env(Dict(
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_DSP_REQUIRED" => "1",
    )) do
        compiled = compile(model, backend = DSPBackend(0), verify = false, optimize = :none)
        @test compiled isa Axiom.CoprocessorCompiledModel

        y = compiled(x).data
        @test size(y) == size(cpu)
        @test all(isfinite, y)
        @test isapprox(y, cpu; atol = 1f-5, rtol = 1f-5)

        report = coprocessor_capability_report()
        dsp = report["backends"]["DSP"]
        @test dsp["required"] == true
        @test dsp["hook_overrides"]["backend_coprocessor_matmul"] == true
        @test dsp["hook_overrides"]["backend_coprocessor_relu"] == true
        @test dsp["hook_overrides"]["backend_coprocessor_softmax"] == true
    end
end
