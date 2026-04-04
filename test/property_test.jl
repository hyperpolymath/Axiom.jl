# SPDX-License-Identifier: MPL-2.0
# (PMPL-1.0-or-later preferred; MPL-2.0 required for Julia ecosystem)
# Property-based invariant tests for Axiom.jl

using Test
using Axiom
using Statistics

@testset "Property-Based Tests" begin

    @testset "Invariant: softmax output sums to 1" begin
        for _ in 1:50
            n = rand(2:20)
            x = randn(Float32, rand(1:8), n)
            y = softmax(x)
            @test all(isapprox.(sum(y, dims=2), 1.0f0, atol=1f-5))
        end
    end

    @testset "Invariant: relu output is non-negative" begin
        for _ in 1:50
            x = randn(Float32, rand(10:100))
            @test all(relu(x) .>= 0)
        end
    end

    @testset "Invariant: sigmoid output in [0, 1]" begin
        for _ in 1:50
            x = randn(Float32, rand(10:100))
            y = sigmoid(x)
            @test all(0.0f0 .<= y .<= 1.0f0)
        end
    end

    @testset "Invariant: Dense layer preserves batch size" begin
        for _ in 1:50
            in_f  = rand(4:32)
            out_f = rand(2:16)
            batch = rand(1:16)
            layer = Dense(in_f, out_f)
            x = Tensor(randn(Float32, batch, in_f))
            y = layer(x)
            @test size(y, 1) == batch
            @test size(y, 2) == out_f
        end
    end

    @testset "Invariant: Sequential output probabilities valid" begin
        for _ in 1:30
            d1 = rand(8:32)
            d2 = rand(4:16)
            cls = rand(2:8)
            model = Sequential(Dense(d1, d2, relu), Dense(d2, cls), Softmax())
            x = Tensor(randn(Float32, rand(1:8), d1))
            y = model(x)
            @test all(y.data .>= 0)
            @test !any(isnan, y.data)
            @test all(isapprox.(sum(y.data, dims=2), 1.0f0, atol=1f-4))
        end
    end

end
