# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Random
using Axiom

Random.seed!(0xA710)

function parse_bool_env(key::String, default::Bool)
    raw = lowercase(strip(get(ENV, key, default ? "1" : "0")))
    raw in ("1", "true", "yes", "on")
end

function fail_or_skip(msg::String, required::Bool)
    if required
        error(msg)
    else
        @info msg
        return true
    end
end

backend_name = lowercase(strip(get(ENV, "AXIOM_GPU_BACKEND", "")))
required = parse_bool_env("AXIOM_GPU_REQUIRED", true)

isempty(backend_name) && error("AXIOM_GPU_BACKEND must be one of: cuda, rocm, metal")

target_backend = nothing

if backend_name == "cuda"
    try
        @eval using CUDA
    catch err
        fail_or_skip("CUDA.jl unavailable: $err", required) && exit(0)
    end
    if !(cuda_available() && cuda_device_count() > 0)
        fail_or_skip("CUDA backend is not functional on this runner", required) && exit(0)
    end
    target_backend = CUDABackend(0)
elseif backend_name == "rocm"
    try
        @eval using AMDGPU
    catch err
        fail_or_skip("AMDGPU.jl unavailable: $err", required) && exit(0)
    end
    if !(rocm_available() && rocm_device_count() > 0)
        fail_or_skip("ROCm backend is not functional on this runner", required) && exit(0)
    end
    target_backend = ROCmBackend(0)
elseif backend_name == "metal"
    try
        @eval using Metal
    catch err
        fail_or_skip("Metal.jl unavailable: $err", required) && exit(0)
    end
    if !metal_available()
        fail_or_skip("Metal backend is not functional on this runner", required) && exit(0)
    end
    target_backend = MetalBackend(0)
else
    error("Unsupported AXIOM_GPU_BACKEND: $backend_name")
end

@testset "GPU hardware smoke ($backend_name)" begin
    model = Sequential(
        Dense(10, 6, relu),
        Dense(6, 3),
        Softmax()
    )
    x = Tensor(randn(Float32, 4, 10))

    cpu = model(x).data
    compiled = compile(model, backend = target_backend, verify = false, optimize = :none)
    acc = compiled(x).data

    @test size(acc) == size(cpu)
    @test all(isfinite, acc)
    @test all(isapprox.(sum(acc, dims = 2), 1.0f0, atol = 1f-4))
end
