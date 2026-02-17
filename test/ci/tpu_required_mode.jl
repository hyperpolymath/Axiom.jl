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

@testset "TPU strict mode compile gate" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "0",
        "AXIOM_TPU_DEVICE_COUNT" => "0",
        "AXIOM_TPU_REQUIRED" => "1",
    )) do
        err = nothing
        try
            compile(model, backend = TPUBackend(0), verify = false, optimize = :none)
        catch caught
            err = caught
        end
        @test err isa ErrorException
        @test occursin("AXIOM_TPU_REQUIRED", sprint(showerror, err))
    end
end

@testset "TPU strict mode runtime hook gate" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 5, 6))

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_TPU_REQUIRED" => "1",
    )) do
        compiled = compile(model, backend = TPUBackend(0), verify = false, optimize = :none)
        @test compiled isa Axiom.CoprocessorCompiledModel

        err = nothing
        try
            compiled(x)
        catch caught
            err = caught
        end

        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("AXIOM_TPU_REQUIRED", msg)
        @test occursin("strict mode enabled", msg)
    end
end

# Install minimal TPU hook overrides to demonstrate strict-mode success path.
@eval begin
    function Axiom.backend_coprocessor_matmul(
        backend::Axiom.TPUBackend,
        A::AbstractMatrix{Float32},
        B::AbstractMatrix{Float32},
    )
        A * B
    end

    function Axiom.backend_coprocessor_relu(
        backend::Axiom.TPUBackend,
        x::AbstractArray{Float32},
    )
        max.(x, 0f0)
    end

    function Axiom.backend_coprocessor_softmax(
        backend::Axiom.TPUBackend,
        x::AbstractArray{Float32},
        dim::Int,
    )
        Axiom.softmax(x, dims = dim)
    end
end

@testset "TPU strict mode with hook overrides" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 5, 6))
    cpu = model(x).data

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_TPU_REQUIRED" => "1",
    )) do
        compiled = compile(model, backend = TPUBackend(0), verify = false, optimize = :none)
        @test compiled isa Axiom.CoprocessorCompiledModel

        y = compiled(x).data
        @test size(y) == size(cpu)
        @test all(isfinite, y)
        @test isapprox(y, cpu; atol = 1f-5, rtol = 1f-5)

        report = coprocessor_capability_report()
        tpu = report["backends"]["TPU"]
        @test tpu["required"] == true
        @test tpu["hook_overrides"]["backend_coprocessor_matmul"] == true
        @test tpu["hook_overrides"]["backend_coprocessor_relu"] == true
        @test tpu["hook_overrides"]["backend_coprocessor_softmax"] == true
    end
end
