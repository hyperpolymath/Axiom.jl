# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Random
using Axiom

Random.seed!(0xA710)

function parse_bool_env(key::String, default::Bool)
    raw = lowercase(strip(get(ENV, key, default ? "1" : "0")))
    raw in ("1", "true", "yes", "on")
end

function maybe_load_accelerator_backend()
    backend_name = lowercase(strip(get(ENV, "AXIOM_SMOKE_ACCELERATOR", "")))
    isempty(backend_name) && return nothing

    if backend_name == "rust"
        lib_path = get(ENV, "AXIOM_RUST_LIB", "")
        isempty(lib_path) && error("AXIOM_RUST_LIB is required when AXIOM_SMOKE_ACCELERATOR=rust")
        isfile(lib_path) || error("AXIOM_RUST_LIB does not exist: $lib_path")
        Axiom.init_rust_backend(lib_path)
        Axiom.rust_available() || error("Rust backend failed to initialize from $lib_path")
        return RustBackend(lib_path)
    elseif backend_name == "cuda"
        @eval using CUDA
        return CUDABackend(0)
    elseif backend_name == "rocm"
        @eval using AMDGPU
        return ROCmBackend(0)
    elseif backend_name == "metal"
        @eval using Metal
        return MetalBackend(0)
    end

    error("Unsupported AXIOM_SMOKE_ACCELERATOR value: $backend_name")
end

@testset "Runtime smoke (CPU)" begin
    model = Sequential(
        Dense(10, 5, relu),
        Dense(5, 3),
        Softmax()
    )
    x = Tensor(randn(Float32, 4, 10))
    y = model(x)
    @test size(y.data) == (4, 3)
    @test all(isfinite, y.data)
    @test all(isapprox.(sum(y.data, dims = 2), 1.0f0, atol = 1f-5))

    data = [(x, nothing)]
    result = verify(model, properties = [ValidProbabilities(), FiniteOutput(), NoNaN()], data = data)
    @test result.passed

    cert = generate_certificate(model, result, model_name = "runtime-smoke")
    cert_path = tempname() * ".cert"
    save_certificate(cert, cert_path)
    loaded = load_certificate(cert_path)
    @test verify_certificate(loaded)
    rm(cert_path)
end

accelerator = maybe_load_accelerator_backend()
if accelerator !== nothing
    required = parse_bool_env("AXIOM_SMOKE_ACCELERATOR_REQUIRED", true)

    @testset "Runtime smoke (accelerated)" begin
        model = Sequential(
            Dense(10, 6, relu),
            Dense(6, 3),
            Softmax()
        )
        x = Tensor(randn(Float32, 4, 10))
        compiled = compile(model, backend = accelerator, verify = false, optimize = :none)
        y = compiled(x)
        @test size(y.data) == (4, 3)
        @test all(isfinite, y.data)
        @test all(isapprox.(sum(y.data, dims = 2), 1.0f0, atol = 1f-4))
    end
elseif parse_bool_env("AXIOM_SMOKE_ACCELERATOR_REQUIRED", false)
    error("Accelerated runtime smoke required but no AXIOM_SMOKE_ACCELERATOR configured")
end
