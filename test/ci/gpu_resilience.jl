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

const GPU_FAILURE_MODE = Dict{DataType, Bool}(
    Axiom.CUDABackend => false,
    Axiom.ROCmBackend => false,
    Axiom.MetalBackend => false,
)

function backend_key(backend)
    if backend isa CUDABackend
        return "cuda"
    elseif backend isa ROCmBackend
        return "rocm"
    elseif backend isa MetalBackend
        return "metal"
    end
    error("Unexpected backend type: $(typeof(backend))")
end

function Axiom.backend_gpu_matmul(
    backend::Union{Axiom.CUDABackend, Axiom.ROCmBackend, Axiom.MetalBackend},
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    if get(GPU_FAILURE_MODE, typeof(backend), false)
        error("Injected GPU matmul failure for $(typeof(backend))")
    end
    A * B
end

function Axiom.backend_gpu_relu(
    backend::Union{Axiom.CUDABackend, Axiom.ROCmBackend, Axiom.MetalBackend},
    x::AbstractArray{Float32},
)
    max.(x, 0f0)
end

function Axiom.backend_gpu_softmax(
    backend::Union{Axiom.CUDABackend, Axiom.ROCmBackend, Axiom.MetalBackend},
    x::AbstractArray{Float32},
    dim::Int,
)
    Axiom.softmax(x, dims=dim)
end

function reset_gpu_failure_mode!()
    for backend_type in keys(GPU_FAILURE_MODE)
        GPU_FAILURE_MODE[backend_type] = false
    end
end

@testset "GPU self-healing fallback and diagnostics" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 5, 6))
    cpu = model(x).data

    with_env(Dict(
        "AXIOM_CUDA_AVAILABLE" => "1",
        "AXIOM_ROCM_AVAILABLE" => "1",
        "AXIOM_METAL_AVAILABLE" => "1",
        "AXIOM_CUDA_DEVICE_COUNT" => "1",
        "AXIOM_ROCM_DEVICE_COUNT" => "1",
        "AXIOM_METAL_DEVICE_COUNT" => "1",
        "AXIOM_GPU_SELF_HEAL" => "1",
    )) do
        for backend in (CUDABackend(0), ROCmBackend(0), MetalBackend(0))
            reset_gpu_runtime_diagnostics!()
            reset_gpu_failure_mode!()
            GPU_FAILURE_MODE[typeof(backend)] = true

            compiled = compile(model, backend = backend, verify = false, optimize = :none)
            @test compiled !== model
            y = compiled(x).data
            @test isapprox(y, cpu; atol = 1f-5, rtol = 1f-5)

            diag = gpu_runtime_diagnostics()["backends"][backend_key(backend)]
            @test diag["runtime_errors"] >= 1
            @test diag["runtime_fallbacks"] >= 1
            @test diag["recoveries"] >= 1
        end
    end

    with_env(Dict(
        "AXIOM_CUDA_AVAILABLE" => "1",
        "AXIOM_CUDA_DEVICE_COUNT" => "1",
        "AXIOM_GPU_SELF_HEAL" => "0",
    )) do
        reset_gpu_runtime_diagnostics!()
        reset_gpu_failure_mode!()
        GPU_FAILURE_MODE[Axiom.CUDABackend] = true

        compiled = compile(model, backend = CUDABackend(0), verify = false, optimize = :none)
        @test compiled !== model
        @test_throws ErrorException compiled(x)

        diag = gpu_runtime_diagnostics()["backends"]["cuda"]
        @test diag["runtime_errors"] >= 1
        @test diag["runtime_fallbacks"] == 0
        @test diag["recoveries"] == 0
    end
end

@testset "GPU capability report" begin
    with_env(Dict(
        "AXIOM_CUDA_AVAILABLE" => "1",
        "AXIOM_ROCM_AVAILABLE" => "0",
        "AXIOM_METAL_AVAILABLE" => "0",
        "AXIOM_CUDA_DEVICE_COUNT" => "2",
        "AXIOM_ROCM_DEVICE_COUNT" => "0",
        "AXIOM_METAL_DEVICE_COUNT" => "0",
    )) do
        report = gpu_capability_report()
        @test report["strategy_order"] == ["CUDA", "ROCm", "Metal"]
        @test occursin("CUDABackend", report["selected_backend"])
        cuda = report["backends"]["CUDA"]
        @test cuda["available"] == true
        @test cuda["device_count"] == 2
        @test cuda["compilable"] == true
        @test haskey(cuda["hook_overrides"], "backend_gpu_matmul")
    end
end
