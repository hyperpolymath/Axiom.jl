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

@testset "GPU fallback behavior (no hardware/extension)" begin
    with_env(Dict(
        "AXIOM_CUDA_AVAILABLE" => "0",
        "AXIOM_ROCM_AVAILABLE" => "0",
        "AXIOM_METAL_AVAILABLE" => "0",
        "AXIOM_CUDA_DEVICE_COUNT" => "0",
        "AXIOM_ROCM_DEVICE_COUNT" => "0"
    )) do
        model = Sequential(
            Dense(6, 4, relu),
            Dense(4, 3),
            Softmax()
        )

        @test !cuda_available()
        @test !rocm_available()
        @test !metal_available()
        @test cuda_device_count() == 0
        @test rocm_device_count() == 0
        @test detect_gpu() === nothing

        @test compile(model, backend = CUDABackend(0), verify = false, optimize = :none) === model
        @test compile(model, backend = ROCmBackend(0), verify = false, optimize = :none) === model
        @test compile(model, backend = MetalBackend(0), verify = false, optimize = :none) === model

        A = randn(Float32, 7, 5)
        B = randn(Float32, 5, 3)
        cpu = Axiom.backend_matmul(JuliaBackend(), A, B)

        @test isapprox(Axiom.backend_gpu_matmul(CUDABackend(0), A, B), cpu; atol = 1f-5, rtol = 1f-5)
        @test isapprox(Axiom.backend_gpu_matmul(ROCmBackend(0), A, B), cpu; atol = 1f-5, rtol = 1f-5)
        @test isapprox(Axiom.backend_gpu_matmul(MetalBackend(0), A, B), cpu; atol = 1f-5, rtol = 1f-5)
    end
end
