# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Random
using Axiom

Random.seed!(0xA710)

const PARITY_TOLERANCES = Dict(
    :matmul => (atol = 1f-4, rtol = 1f-4),
    :dense => (atol = 1f-4, rtol = 1f-4),
    :conv2d => (atol = 2f-4, rtol = 2f-4),
    :normalization => (atol = 1f-4, rtol = 1f-4),
    :activations => (atol = 1f-5, rtol = 1f-5)
)

function require_rust_backend!()
    lib_path = get(ENV, "AXIOM_RUST_LIB", "")
    isempty(lib_path) && error("AXIOM_RUST_LIB is required for backend parity tests")
    isfile(lib_path) || error("AXIOM_RUST_LIB does not exist: $lib_path")
    Axiom.init_rust_backend(lib_path)
    Axiom.rust_available() || error("Rust backend failed to initialize from $lib_path")
    return lib_path
end

function assert_parity(got, expected, key::Symbol)
    tol = PARITY_TOLERANCES[key]
    @test isapprox(got, expected; atol = tol.atol, rtol = tol.rtol)
end

lib_path = require_rust_backend!()
rust = RustBackend(lib_path)

@testset "CPU vs Rust backend parity" begin
    @testset "matmul" begin
        A = randn(Float32, 16, 12)
        B = randn(Float32, 12, 7)
        cpu = Axiom.backend_matmul(JuliaBackend(), A, B)
        rs = Axiom.backend_matmul(rust, A, B)
        assert_parity(rs, cpu, :matmul)
    end

    @testset "activations" begin
        x = randn(Float32, 128)
        cpu_relu = Axiom.backend_relu(JuliaBackend(), x)
        rs_relu = Axiom.backend_relu(rust, x)
        assert_parity(rs_relu, cpu_relu, :activations)

        logits = randn(Float32, 8, 5)
        cpu_softmax = Axiom.backend_softmax(JuliaBackend(), logits, 2)
        rs_softmax = Axiom.backend_softmax(rust, logits, 2)
        assert_parity(rs_softmax, cpu_softmax, :activations)
    end

    @testset "dense" begin
        layer = Dense(12, 6, relu; dtype = Float32)
        x = randn(Float32, 10, 12)

        cpu = layer(Tensor(x)).data
        rs_linear = Axiom.backend_matmul(rust, x, layer.weight)
        if layer.bias !== nothing
            rs_linear .+= layer.bias'
        end
        rs = layer.activation(rs_linear)

        assert_parity(rs, cpu, :dense)
    end

    @testset "conv2d" begin
        input = randn(Float32, 2, 10, 10, 3)
        weight = randn(Float32, 3, 3, 3, 4)
        bias = randn(Float32, 4)
        stride = (1, 1)
        padding = (1, 1)

        cpu = Axiom.backend_conv2d(JuliaBackend(), input, weight, bias, stride, padding)
        rs = Axiom.backend_conv2d(rust, input, weight, bias, stride, padding)

        assert_parity(rs, cpu, :conv2d)
    end

    @testset "normalization" begin
        x = randn(Float32, 6, 8)
        gamma = rand(Float32, 8)
        beta = rand(Float32, 8)
        running_mean = randn(Float32, 8)
        running_var = rand(Float32, 8) .+ 1f0
        eps = 1f-5

        cpu = Axiom.backend_batchnorm(
            JuliaBackend(),
            x,
            gamma,
            beta,
            running_mean,
            running_var,
            eps,
            false
        )

        rs = Axiom.backend_batchnorm(
            rust,
            x,
            gamma,
            beta,
            running_mean,
            running_var,
            eps,
            false
        )

        assert_parity(rs, cpu, :normalization)
    end
end

println("Backend parity checks passed with tolerances: ", PARITY_TOLERANCES)
