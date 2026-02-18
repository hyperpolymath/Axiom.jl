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

@testset "Math strict mode compile gate" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )

    with_env(Dict(
        "AXIOM_MATH_AVAILABLE" => "0",
        "AXIOM_MATH_DEVICE_COUNT" => "0",
        "AXIOM_MATH_REQUIRED" => "1",
    )) do
        err = nothing
        try
            compile(model, backend = MathBackend(0), verify = false, optimize = :none)
        catch caught
            err = caught
        end
        @test err isa ErrorException
        @test occursin("AXIOM_MATH_REQUIRED", sprint(showerror, err))
    end
end

@testset "Math strict mode runtime with built-in production kernels" begin
    dense_model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    dense_x = Tensor(randn(Float32, 5, 6))
    dense_cpu = dense_model(dense_x).data

    conv_model = Sequential(
        Conv2d(3, 4, (3, 3), padding = 1),
        BatchNorm(4),
        ReLU(),
        MaxPool2d((2, 2)),
        GlobalAvgPool(),
        Dense(4, 3),
        Softmax(),
    )
    conv_x = Tensor(randn(Float32, 2, 8, 8, 3))
    conv_cpu = conv_model(conv_x).data

    norm_model = Sequential(
        Dense(6, 6, identity),
        LayerNorm(6),
        ReLU(),
        Dense(6, 3),
        Softmax(),
    )
    norm_x = Tensor(randn(Float32, 4, 6))
    norm_cpu = norm_model(norm_x).data

    avgpool_model = Sequential(
        Conv2d(3, 4, (3, 3), padding = 1),
        AvgPool2d((2, 2)),
        GlobalAvgPool(),
        Dense(4, 3),
        Softmax(),
    )
    avgpool_x = Tensor(randn(Float32, 2, 8, 8, 3))
    avgpool_cpu = avgpool_model(avgpool_x).data

    with_env(Dict(
        "AXIOM_MATH_AVAILABLE" => "1",
        "AXIOM_MATH_DEVICE_COUNT" => "1",
        "AXIOM_MATH_REQUIRED" => "1",
    )) do
        reset_coprocessor_runtime_diagnostics!()

        dense_compiled = compile(dense_model, backend = MathBackend(0), verify = false, optimize = :none)
        @test dense_compiled isa Axiom.CoprocessorCompiledModel
        dense_y = dense_compiled(dense_x).data
        @test size(dense_y) == size(dense_cpu)
        @test all(isfinite, dense_y)
        @test isapprox(dense_y, dense_cpu; atol = 1f-5, rtol = 1f-5)
        @test all(isapprox.(sum(dense_y, dims = 2), 1.0f0, atol = 2f-4))

        conv_compiled = compile(conv_model, backend = MathBackend(0), verify = false, optimize = :none)
        @test conv_compiled isa Axiom.CoprocessorCompiledModel
        conv_y = conv_compiled(conv_x).data
        @test size(conv_y) == size(conv_cpu)
        @test all(isfinite, conv_y)
        @test isapprox(conv_y, conv_cpu; atol = 2f-4, rtol = 2f-4)
        @test all(isapprox.(sum(conv_y, dims = 2), 1.0f0, atol = 2f-4))

        norm_compiled = compile(norm_model, backend = MathBackend(0), verify = false, optimize = :none)
        @test norm_compiled isa Axiom.CoprocessorCompiledModel
        norm_y = norm_compiled(norm_x).data
        @test size(norm_y) == size(norm_cpu)
        @test all(isfinite, norm_y)
        @test isapprox(norm_y, norm_cpu; atol = 1f-4, rtol = 1f-4)
        @test all(isapprox.(sum(norm_y, dims = 2), 1.0f0, atol = 2f-4))

        avgpool_compiled = compile(avgpool_model, backend = MathBackend(0), verify = false, optimize = :none)
        @test avgpool_compiled isa Axiom.CoprocessorCompiledModel
        avgpool_y = avgpool_compiled(avgpool_x).data
        @test size(avgpool_y) == size(avgpool_cpu)
        @test all(isfinite, avgpool_y)
        @test isapprox(avgpool_y, avgpool_cpu; atol = 2f-4, rtol = 2f-4)
        @test all(isapprox.(sum(avgpool_y, dims = 2), 1.0f0, atol = 2f-4))

        report = coprocessor_capability_report()
        math = report["backends"]["MATH"]
        @test math["required"] == true
        @test math["kernel_hooks_loaded"] == true
        for hook in (
            "backend_coprocessor_matmul",
            "backend_coprocessor_conv2d",
            "backend_coprocessor_relu",
            "backend_coprocessor_softmax",
            "backend_coprocessor_batchnorm",
            "backend_coprocessor_layernorm",
            "backend_coprocessor_maxpool2d",
            "backend_coprocessor_avgpool2d",
            "backend_coprocessor_global_avgpool2d",
        )
            @test math["hook_overrides"][hook] == true
        end

        diag = coprocessor_runtime_diagnostics()["backends"]["math"]
        @test diag["compile_fallbacks"] == 0
        @test diag["runtime_errors"] == 0
        @test diag["runtime_fallbacks"] == 0
        @test diag["recoveries"] == 0
    end
end
