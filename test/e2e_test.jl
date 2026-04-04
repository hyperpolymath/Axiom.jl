# SPDX-License-Identifier: MPL-2.0
# (PMPL-1.0-or-later preferred; MPL-2.0 required for Julia ecosystem)
# E2E pipeline tests for Axiom.jl

using Test
using Axiom
using Statistics

@testset "E2E Pipeline Tests" begin

    @testset "Full MLP classification pipeline" begin
        # Build a full MLP, pass a batch through, verify output shape + validity
        model = Sequential(
            Dense(20, 64, relu),
            Dense(64, 32, relu),
            Dense(32, 5),
            Softmax()
        )
        x = Tensor(randn(Float32, 16, 20))
        y = model(x)

        @test size(y) == (16, 5)
        @test all(y.data .>= 0)
        @test !any(isnan, y.data)
        @test all(isapprox.(sum(y.data, dims=2), 1.0f0, atol=1f-4))
    end

    @testset "Full CNN feature extraction pipeline" begin
        # Conv → BN → Pool → Dense → Softmax
        model = Sequential(
            Conv2d(1, 8, (3, 3), padding=1),
            BatchNorm(8),
            ReLU(),
            MaxPool2d((2, 2)),
            GlobalAvgPool(),
            Dense(8, 4),
            Softmax()
        )
        x = Tensor(randn(Float32, 4, 16, 16, 1))
        y = model(x)

        @test size(y) == (4, 4)
        @test all(y.data .>= 0)
        @test all(isapprox.(sum(y.data, dims=2), 1.0f0, atol=1f-4))
    end

    @testset "Data loading and one-hot pipeline" begin
        # Simulate a small dataset, load it, encode labels, split, batch
        X = randn(Float32, 60, 10)
        y_labels = rand(1:3, 60)

        train_data, test_data = train_test_split((X, y_labels), test_ratio=0.2)
        @test size(train_data[1], 1) == 48
        @test size(test_data[1], 1) == 12

        onehot = one_hot(y_labels, 3)
        @test size(onehot) == (60, 3)
        @test all(sum(onehot, dims=2) .== 1.0f0)

        loader = DataLoader((X, y_labels), batch_size=16, shuffle=false)
        @test length(loader) == 4
        for (bx, _) in loader
            @test size(bx, 2) == 10
        end
    end

    @testset "Error handling: shape mismatches" begin
        layer = Dense(10, 5)
        x_bad = Tensor(randn(Float32, 4, 9))
        @test_throws DimensionMismatch layer(x_bad)

        @test_throws AssertionError Dense(-1, 10)
        @test_throws AssertionError Dense(10, 0)
    end

end
