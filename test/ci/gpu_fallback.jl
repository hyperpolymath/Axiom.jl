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

const GPU_TEST_HOOKS = Dict{DataType, Dict{Symbol, Int}}(
    Axiom.CUDABackend => Dict(:matmul => 0, :relu => 0, :softmax => 0),
    Axiom.ROCmBackend => Dict(:matmul => 0, :relu => 0, :softmax => 0),
    Axiom.MetalBackend => Dict(:matmul => 0, :relu => 0, :softmax => 0),
)

function reset_gpu_test_hooks!()
    for counters in values(GPU_TEST_HOOKS)
        counters[:matmul] = 0
        counters[:relu] = 0
        counters[:softmax] = 0
    end
end

function Axiom.backend_gpu_matmul(
    backend::Union{Axiom.CUDABackend, Axiom.ROCmBackend, Axiom.MetalBackend},
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    GPU_TEST_HOOKS[typeof(backend)][:matmul] += 1
    A * B
end

function Axiom.backend_gpu_relu(
    backend::Union{Axiom.CUDABackend, Axiom.ROCmBackend, Axiom.MetalBackend},
    x::AbstractArray{Float32},
)
    GPU_TEST_HOOKS[typeof(backend)][:relu] += 1
    max.(x, 0f0)
end

function Axiom.backend_gpu_softmax(
    backend::Union{Axiom.CUDABackend, Axiom.ROCmBackend, Axiom.MetalBackend},
    x::AbstractArray{Float32},
    dim::Int,
)
    GPU_TEST_HOOKS[typeof(backend)][:softmax] += 1
    Axiom.softmax(x, dims=dim)
end

@testset "GPU fallback behavior (no hardware/extension)" begin
    with_env(Dict(
        "AXIOM_CUDA_AVAILABLE" => "0",
        "AXIOM_ROCM_AVAILABLE" => "0",
        "AXIOM_METAL_AVAILABLE" => "0",
        "AXIOM_CUDA_DEVICE_COUNT" => "0",
        "AXIOM_ROCM_DEVICE_COUNT" => "0",
        "AXIOM_METAL_DEVICE_COUNT" => "0",
    )) do
        model = Sequential(
            Dense(6, 4, relu),
            Dense(4, 3),
            Softmax()
        )

        @test !cuda_available()
        @test !rocm_available()
        @test !metal_available()
        @test Axiom.cuda_device_count() == 0
        @test Axiom.rocm_device_count() == 0
        @test Axiom.metal_device_count() == 0
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

@testset "GPU compiled model uses extension hooks when available" begin
    with_env(Dict(
        "AXIOM_CUDA_AVAILABLE" => "1",
        "AXIOM_ROCM_AVAILABLE" => "1",
        "AXIOM_METAL_AVAILABLE" => "1",
        "AXIOM_CUDA_DEVICE_COUNT" => "1",
        "AXIOM_ROCM_DEVICE_COUNT" => "1",
        "AXIOM_METAL_DEVICE_COUNT" => "1",
    )) do
        model = Sequential(
            Dense(6, 4, relu),
            Dense(4, 3),
            Softmax()
        )
        x = Tensor(randn(Float32, 5, 6))
        cpu = model(x).data

        for backend in (CUDABackend(0), ROCmBackend(0), MetalBackend(0))
            reset_gpu_test_hooks!()
            compiled = compile(model, backend=backend, verify=false, optimize=:none)
            @test compiled !== model
            y = compiled(x).data
            @test size(y) == size(cpu)
            @test isapprox(y, cpu; atol = 1f-5, rtol = 1f-5)
            @test GPU_TEST_HOOKS[typeof(backend)][:matmul] >= 2
            @test GPU_TEST_HOOKS[typeof(backend)][:relu] >= 1
            @test GPU_TEST_HOOKS[typeof(backend)][:softmax] >= 1

            out_of_range = select_device!(backend, 9)
            @test compile(model, backend=out_of_range, verify=false, optimize=:none) === model
        end
    end
end
