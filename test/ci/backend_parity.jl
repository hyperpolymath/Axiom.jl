# SPDX-License-Identifier: PMPL-1.0-or-later
# Backend parity test: Julia vs Zig (SIMD + threaded)

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

function require_zig_backend!()
    lib_path = get(ENV, "AXIOM_ZIG_LIB", "")
    isempty(lib_path) && error("AXIOM_ZIG_LIB is required for backend parity tests")
    isfile(lib_path) || error("AXIOM_ZIG_LIB does not exist: $lib_path")
    Axiom.init_zig_backend(lib_path)
    Axiom.zig_available() || error("Zig backend failed to initialize from $lib_path")
    return lib_path
end

function assert_parity(got, expected, key::Symbol)
    tol = PARITY_TOLERANCES[key]
    @test isapprox(got, expected; atol = tol.atol, rtol = tol.rtol)
end

lib_path = require_zig_backend!()
zig = ZigBackend(lib_path)

@testset "CPU vs Zig backend parity" begin
    @testset "matmul" begin
        A = randn(Float32, 16, 12)
        B = randn(Float32, 12, 7)
        cpu = Axiom.backend_matmul(JuliaBackend(), A, B)
        zg = Axiom.backend_matmul(zig, A, B)
        assert_parity(zg, cpu, :matmul)
    end

    @testset "activations" begin
        x = randn(Float32, 128)
        cpu_relu = Axiom.backend_relu(JuliaBackend(), x)
        zg_relu = Axiom.backend_relu(zig, x)
        assert_parity(zg_relu, cpu_relu, :activations)

        logits = randn(Float32, 8, 5)
        cpu_softmax = Axiom.backend_softmax(JuliaBackend(), logits, 2)
        zg_softmax = Axiom.backend_softmax(zig, logits, 2)
        assert_parity(zg_softmax, cpu_softmax, :activations)
    end

    @testset "dense" begin
        layer = Dense(12, 6, relu; dtype = Float32)
        x = randn(Float32, 10, 12)

        cpu = layer(Tensor(x)).data
        zg_linear = Axiom.backend_matmul(zig, x, layer.weight)
        if layer.bias !== nothing
            zg_linear .+= layer.bias'
        end
        zg = layer.activation(zg_linear)

        assert_parity(zg, cpu, :dense)
    end

    @testset "conv2d" begin
        input = randn(Float32, 2, 10, 10, 3)
        weight = randn(Float32, 3, 3, 3, 4)
        bias = randn(Float32, 4)
        stride = (1, 1)
        padding = (1, 1)

        cpu = Axiom.backend_conv2d(JuliaBackend(), input, weight, bias, stride, padding)
        zg = Axiom.backend_conv2d(zig, input, weight, bias, stride, padding)

        assert_parity(zg, cpu, :conv2d)
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

        zg = Axiom.backend_batchnorm(
            zig,
            x,
            gamma,
            beta,
            running_mean,
            running_var,
            eps,
            false
        )

        assert_parity(zg, cpu, :normalization)
    end

    @testset "threaded activations (>64K)" begin
        # Test above THREAD_THRESHOLD to exercise multi-threaded path
        x = randn(Float32, 100_000)
        cpu_sig = Axiom.backend_sigmoid(JuliaBackend(), x)
        zg_sig = Axiom.backend_sigmoid(zig, x)
        assert_parity(zg_sig, cpu_sig, :activations)

        cpu_gelu = Axiom.backend_gelu(JuliaBackend(), x)
        zg_gelu = Axiom.backend_gelu(zig, x)
        assert_parity(zg_gelu, cpu_gelu, :activations)
    end
end

println("Backend parity checks passed with tolerances: ", PARITY_TOLERANCES)
