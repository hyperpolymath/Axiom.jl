# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Test Suite

using Test
using Axiom
using LinearAlgebra
using Statistics

@testset "Axiom.jl" begin

    @testset "Tensor Types" begin
        # Test tensor creation
        t = axiom_zeros(Float32, 10, 5)
        @test size(t) == (10, 5)
        @test eltype(t) == Float32

        t = axiom_randn(Float32, 32, 784)
        @test size(t) == (32, 784)

        # Test dynamic tensor
        dt = DynamicTensor(randn(Float32, 16, 64))
        @test size(dt) == (16, 64)
    end

    @testset "Dense Layer" begin
        # Test Dense layer forward pass
        layer = Dense(784, 128)
        @test layer.in_features == 784
        @test layer.out_features == 128

        x = randn(Float32, 32, 784)
        y = layer(x)
        @test size(y) == (32, 128)

        # Test without bias
        layer_no_bias = Dense(784, 128, bias=false)
        @test layer_no_bias.bias === nothing

        # Test with activation
        layer_relu = Dense(784, 128, relu)
        y = layer_relu(x)
        @test all(y .>= 0)  # ReLU output

        # Test shape mismatch
        x_wrong_shape = randn(Float32, 32, 783) # Wrong number of features
        # Note: The actual error will be a MethodError because the matrix multiplication
        # inside the layer will fail, not an explicit DimensionMismatch from Axiom.
        # This is because the `@ensure` macro is not yet used to check input shapes.
        # Once it is, this test should be updated to `_throws DimensionMismatch`.
        @test_throws DimensionMismatch layer(x_wrong_shape)

        # Test invalid constructor arguments
        @test_throws ArgumentError Dense(-10, 128) # Negative in_features
        @test_throws ArgumentError Dense(784, -20) # Negative out_features
    end

    @testset "Conv2d Layer" begin
        layer = Conv2d(3, 64, (3, 3))
        @test layer.in_channels == 3
        @test layer.out_channels == 64
        @test layer.kernel_size == (3, 3)

        x = randn(Float32, 4, 32, 32, 3)  # (N, H, W, C)
        y = layer(x)
        @test size(y) == (4, 30, 30, 64)

        # Test with padding
        layer_padded = Conv2d(3, 64, (3, 3), padding=1)
        y = layer_padded(x)
        @test size(y) == (4, 32, 32, 64)

        # Test with stride
        layer_strided = Conv2d(3, 64, (3, 3), stride=2)
        y = layer_strided(x)
        @test size(y) == (4, 15, 15, 64)

        # Test shape mismatch
        x_wrong_channels = randn(Float32, 4, 32, 32, 4) # 4 channels instead of 3
        # Similar to Dense, this will likely throw a MethodError on matmul
        @test_throws DimensionMismatch layer(x_wrong_channels)
    end

    @testset "Activations" begin
        x = randn(Float32, 100)

        # ReLU
        y = relu(x)
        @test all(y .>= 0)
        @test sum(y .== 0) > 0  # Some values should be 0

        # Sigmoid
        y = sigmoid(x)
        @test all(0 .<= y .<= 1)

        # Softmax
        x = randn(Float32, 10, 5)
        y = softmax(x)
        sums = sum(y, dims=2)
        @test all(isapprox.(sums, 1.0, atol=1e-5))

        # GELU
        y = gelu(randn(Float32, 100))
        @test length(y) == 100

        # Test with empty input
        x_empty = Float32[]
        @test isempty(relu(x_empty))
        @test isempty(sigmoid(x_empty))
    end

    @testset "Normalization Layers" begin
        # BatchNorm
        bn = BatchNorm(64)
        x = randn(Float32, 32, 64)
        bn.training = true
        y = bn(x)
        @test size(y) == size(x)
        @test isapprox(mean(y), 0, atol=1e-5)
        @test isapprox(std(y), 1, atol=1e-1)

        # LayerNorm
        ln = LayerNorm(64)
        y = ln(x)
        @test size(y) == size(x)
        @test isapprox(mean(y), 0, atol=1e-5)
        @test isapprox(std(y), 1, atol=1e-1)
    end

    @testset "Pooling Layers" begin
        x = randn(Float32, 4, 32, 32, 64)

        # MaxPool
        mp = MaxPool2d((2, 2))
        y = mp(x)
        @test size(y) == (4, 16, 16, 64)

        # AvgPool
        ap = AvgPool2d((2, 2))
        y = ap(x)
        @test size(y) == (4, 16, 16, 64)

        # GlobalAvgPool
        gap = GlobalAvgPool()
        y = gap(x)
        @test size(y) == (4, 64)

        # Flatten
        fl = Flatten()
        x_flat = fl(x)
        @test size(x_flat) == (4, 32 * 32 * 64)
    end

    @testset "Pipeline/Sequential" begin
        # Build a simple network
        model = Sequential(
            Dense(784, 256, relu),
            Dense(256, 128, relu),
            Dense(128, 10),
            Softmax()
        )

        x = randn(Float32, 32, 784)
        y = model(x)

        @test size(y) == (32, 10)
        @test all(isapprox.(sum(y, dims=2), 1.0, atol=1e-5))

        # Test empty pipeline
        empty_model = Sequential()
        x_identity = randn(Float32, 5, 5)
        @test empty_model(x_identity) == x_identity

        # Test single layer pipeline
        single_layer_model = Sequential(Dense(10, 5))
        x_single = randn(Float32, 2, 10)
        y_single = single_layer_model(x_single)
        @test size(y_single) == (2, 5)
    end

    @testset "Optimizers" begin
        # SGD
        opt = SGD(lr=0.01f0)
        @test opt.lr == 0.01f0

        # Adam
        opt = Adam(lr=0.001f0)
        @test opt.lr == 0.001f0
        @test opt.beta1 == 0.9f0
        @test opt.beta2 == 0.999f0

        # AdamW
        opt = AdamW(lr=0.001f0, weight_decay=0.01f0)
        @test opt.weight_decay == 0.01f0
    end

    @testset "Loss Functions" begin
        pred = randn(Float32, 32, 10)
        target = randn(Float32, 32, 10)

        # MSE
        loss = mse_loss(pred, target)
        @test loss >= 0

        # Cross-entropy (with softmax pred)
        pred_softmax = softmax(randn(Float32, 32, 10))
        target_onehot = zeros(Float32, 32, 10)
        for i in 1:32
            target_onehot[i, rand(1:10)] = 1.0f0
        end
        loss = crossentropy(pred_softmax, target_onehot)
        @test loss >= 0

        # Binary cross-entropy
        pred_sigmoid = sigmoid(randn(Float32, 32, 1))
        target_binary = Float32.(rand(Bool, 32, 1))
        loss = binary_crossentropy(pred_sigmoid, target_binary)
        @test loss >= 0
    end

    @testset "Data Utilities" begin
        # DataLoader
        X = randn(Float32, 100, 10)
        y = rand(1:5, 100)

        loader = DataLoader((X, y), batch_size=32, shuffle=true)
        @test length(loader) == 4  # ceil(100/32)

        batch_count = 0
        for (bx, by) in loader
            batch_count += 1
            @test size(bx, 2) == 10
        end
        @test batch_count == 4

        # Train/test split
        train_data, test_data = train_test_split((X, y), test_ratio=0.2)
        @test size(train_data[1], 1) == 80
        @test size(test_data[1], 1) == 20

        # One-hot encoding
        labels = [1, 2, 3, 1, 2]
        onehot = one_hot(labels, 3)
        @test size(onehot) == (5, 3)
        @test onehot[1, 1] == 1.0f0
        @test onehot[2, 2] == 1.0f0
    end

    @testset "Verification" begin
        # Build model
        model = Sequential(
            Dense(10, 5, relu),
            Dense(5, 3),
            Softmax()
        )

        # Check output properties
        x = randn(Float32, 4, 10)
        y = model(x)

        # Probabilities should sum to 1
        sums = sum(y, dims=2)
        @test all(isapprox.(sums, 1.0, atol=1e-5))

        # All values should be non-negative
        @test all(y .>= 0)

        # No NaN
        @test !any(isnan, y)

        # Test property checking
        prop = ValidProbabilities()
        data = [(x, nothing)]
        @test check(prop, model, data)
    end

    @testset "Ensure Macro" begin
        x = [0.3f0, 0.3f0, 0.4f0]

        # Should not throw
        @ensure sum(x) ≈ 1.0 "Probabilities must sum to 1"

        # Should throw
        @test_throws EnsureViolation @ensure sum(x) ≈ 2.0 "Wrong sum"
    end

    @testset "SMT Rust Runner" begin
        if get(ENV, "AXIOM_SMT_RUNNER", "") != "rust"
            @test true # Skip if not enabled
        else
            # Test error case: runner enabled, but lib not set
            if !haskey(ENV, "AXIOM_RUST_LIB")
                 prop = Axiom.ParsedProperty(:exists, [:x], :(x > 0))
                 @test_throws ErrorException Axiom.smt_proof(prop)
            end

            # In a real CI, we would build the rust lib and run the tests
            # Here we just check that the code doesn't crash if the lib is not available
            if !Axiom.rust_available()
                @info "Skipping full Rust SMT runner test; backend not available"
                @test true
            else
                solver = Axiom.get_smt_solver()
                if solver === nothing
                    @info "Skipping Rust SMT runner test; solver not available"
                    @test true
                else
                    prop = Axiom.ParsedProperty(:exists, [:x], :(x > 0))
                    result = Axiom.smt_proof(prop)
                    # We can't know the result without the actual lib, so just check it doesn't error
                    @test result.status in (:proven, :unknown)
                end
            end
        end
    end

    @testset "SMT Cache" begin
        if get(ENV, "AXIOM_SMT_CACHE", "") in ("1", "true", "yes")
            solver = Axiom.get_smt_solver()
            if solver === nothing
                @info "Skipping SMT cache test; no solver available"
                @test true
            else
                prop = Axiom.ParsedProperty(:forall, [:x], :(x > 0))
                result1 = Axiom.smt_proof(prop)
                result2 = Axiom.smt_proof(prop)
                @test result1.status == result2.status
            end
        else
            @test true
        end
    end

    # Include proof serialization tests
    include("verification/serialization_tests.jl")

end

println("\nAll tests passed!")
