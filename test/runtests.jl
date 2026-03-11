# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Test Suite

using Test
using Axiom
using LinearAlgebra
using Statistics
using HTTP
using JSON

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

        x = Tensor(randn(Float32, 32, 784))
        y = layer(x)
        @test size(y) == (32, 128)

        # Test without bias
        layer_no_bias = Dense(784, 128, bias=false)
        @test layer_no_bias.bias === nothing

        # Test with activation
        layer_relu = Dense(784, 128, relu)
        y = layer_relu(x)
        @test all(y.data .>= 0)  # ReLU output

        # Test shape mismatch
        x_wrong_shape = Tensor(randn(Float32, 32, 783)) # Wrong number of features
        # Note: The actual error will be a MethodError because the matrix multiplication
        # inside the layer will fail, not an explicit DimensionMismatch from Axiom.
        # This is because the `@ensure` macro is not yet used to check input shapes.
        # Once it is, this test should be updated to `_throws DimensionMismatch`.
        @test_throws DimensionMismatch layer(x_wrong_shape)

        # Test invalid constructor arguments
        @test_throws AssertionError Dense(-10, 128) # Negative in_features
        @test_throws AssertionError Dense(784, -20) # Negative out_features
    end

    @testset "Conv2d Layer" begin
        layer = Conv2d(3, 64, (3, 3))
        @test layer.in_channels == 3
        @test layer.out_channels == 64
        @test layer.kernel_size == (3, 3)

        x = Tensor(randn(Float32, 4, 32, 32, 3))  # (N, H, W, C)
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
        x_wrong_channels = Tensor(randn(Float32, 4, 32, 32, 4)) # 4 channels instead of 3
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
        x = Tensor(randn(Float32, 32, 64))
        bn.training = true
        y = bn(x)
        @test size(y) == size(x)
        @test isapprox(mean(y.data), 0, atol=1e-5)
        @test isapprox(std(y.data), 1, atol=1e-1)

        # LayerNorm
        ln = LayerNorm(64)
        y = ln(x)
        @test size(y) == size(x)
        @test isapprox(mean(y.data), 0, atol=1e-5)
        @test isapprox(std(y.data), 1, atol=1e-1)
    end

    @testset "Pooling Layers" begin
        x = Tensor(randn(Float32, 4, 32, 32, 64))

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

        x = Tensor(randn(Float32, 32, 784))
        y = model(x)

        @test size(y) == (32, 10)
        @test all(isapprox.(sum(y.data, dims=2), 1.0, atol=1e-5))

        # Test empty pipeline
        empty_model = Sequential()
        x_identity = Tensor(randn(Float32, 5, 5))
        @test empty_model(x_identity) == x_identity

        # Test single layer pipeline
        single_layer_model = Sequential(Dense(10, 5))
        x_single = Tensor(randn(Float32, 2, 10))
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

    @testset "Coprocessor Backends" begin
        model = Sequential(
            Dense(10, 5, relu),
            Dense(5, 3),
            Softmax()
        )

        old_tpu_available = get(ENV, "AXIOM_TPU_AVAILABLE", nothing)
        old_tpu_count = get(ENV, "AXIOM_TPU_DEVICE_COUNT", nothing)

        try
            ENV["AXIOM_TPU_AVAILABLE"] = "1"
            ENV["AXIOM_TPU_DEVICE_COUNT"] = "2"

            @test tpu_available()
            @test tpu_device_count() == 2
            @test detect_coprocessor() == TPUBackend(0)

            compiled_ok = compile(model, backend=TPUBackend(1), verify=false)
            @test compiled_ok isa Axiom.CoprocessorCompiledModel

            compiled_bad = compile(model, backend=TPUBackend(4), verify=false)
            @test compiled_bad === model
        finally
            if old_tpu_available === nothing
                delete!(ENV, "AXIOM_TPU_AVAILABLE")
            else
                ENV["AXIOM_TPU_AVAILABLE"] = old_tpu_available
            end

            if old_tpu_count === nothing
                delete!(ENV, "AXIOM_TPU_DEVICE_COUNT")
            else
                ENV["AXIOM_TPU_DEVICE_COUNT"] = old_tpu_count
            end
        end

        @test select_device!(TPUBackend(0), 1) == TPUBackend(1)
        @test select_device!(NPUBackend(0), 2) == NPUBackend(2)
        @test select_device!(DSPBackend(0), 3) == DSPBackend(3)
        @test select_device!(FPGABackend(0), 4) == FPGABackend(4)

        accel = detect_accelerator()
        @test accel === nothing || accel isa AbstractBackend
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

    @testset "Serving APIs" begin
        model = Sequential(
            Dense(10, 5, relu),
            Dense(5, 3),
            Softmax()
        )
        input_batch = [Float32.(randn(10)), Float32.(randn(10))]

        function start_test_server(start_fn)
            for _ in 1:20
                port = rand(20_000:45_000)
                try
                    server = start_fn(port)
                    return server, port
                catch
                    # Retry with another port
                end
            end
            error("Unable to bind test server after multiple attempts")
        end

        function grpc_varint(value::Integer)
            u = value >= 0 ? UInt64(value) : reinterpret(UInt64, Int64(value))
            out = UInt8[]
            while u >= 0x80
                push!(out, UInt8((u & 0x7f) | 0x80))
                u >>= 7
            end
            push!(out, UInt8(u))
            out
        end

        function grpc_field_key(field::Integer, wire::Integer)
            grpc_varint((UInt64(field) << 3) | UInt64(wire))
        end

        function grpc_frame(payload::Vector{UInt8})
            n = length(payload)
            out = UInt8[0x00]
            push!(out, UInt8((n >> 24) & 0xff))
            push!(out, UInt8((n >> 16) & 0xff))
            push!(out, UInt8((n >> 8) & 0xff))
            push!(out, UInt8(n & 0xff))
            append!(out, payload)
            out
        end

        function grpc_unframe(body::Vector{UInt8})
            @test length(body) >= 5
            @test body[1] == 0x00
            n = (Int(body[2]) << 24) | (Int(body[3]) << 16) | (Int(body[4]) << 8) | Int(body[5])
            @test length(body) == 5 + n
            body[6:end]
        end

        function grpc_predict_request(values::Vector{Float32})
            packed = collect(reinterpret(UInt8, values))
            out = UInt8[]
            append!(out, grpc_field_key(1, 2))
            append!(out, grpc_varint(length(packed)))
            append!(out, packed)
            out
        end

        function grpc_read_varint(data::Vector{UInt8}, idx::Int)
            result = UInt64(0)
            shift = 0
            while true
                @test idx <= length(data)
                b = data[idx]
                idx += 1
                result |= UInt64(b & 0x7f) << shift
                if (b & 0x80) == 0
                    return result, idx
                end
                shift += 7
            end
        end

        function grpc_decode_floats_field1(payload::Vector{UInt8})
            vals = Float32[]
            idx = 1
            while idx <= length(payload)
                key, idx = grpc_read_varint(payload, idx)
                field = Int(key >> 3)
                wire = Int(key & 0x07)
                if field == 1 && wire == 2
                    block_len, idx = grpc_read_varint(payload, idx)
                    stop_idx = idx + Int(block_len) - 1
                    while idx <= stop_idx
                        u = UInt32(payload[idx]) |
                            (UInt32(payload[idx + 1]) << 8) |
                            (UInt32(payload[idx + 2]) << 16) |
                            (UInt32(payload[idx + 3]) << 24)
                        push!(vals, reinterpret(Float32, u))
                        idx += 4
                    end
                elseif wire == 2
                    block_len, idx = grpc_read_varint(payload, idx)
                    idx += Int(block_len)
                elseif wire == 5
                    idx += 4
                elseif wire == 0
                    _, idx = grpc_read_varint(payload, idx)
                else
                    error("Unsupported wire type in gRPC test decoder: $wire")
                end
            end
            vals
        end

        function grpc_decode_health_response(payload::Vector{UInt8})
            idx = 1
            while idx <= length(payload)
                key, idx = grpc_read_varint(payload, idx)
                field = Int(key >> 3)
                wire = Int(key & 0x07)
                if field == 1 && wire == 2
                    n, idx = grpc_read_varint(payload, idx)
                    return String(payload[idx:idx+Int(n)-1])
                elseif wire == 2
                    n, idx = grpc_read_varint(payload, idx)
                    idx += Int(n)
                elseif wire == 0
                    _, idx = grpc_read_varint(payload, idx)
                elseif wire == 5
                    idx += 4
                else
                    error("Unsupported wire type in gRPC test decoder: $wire")
                end
            end
            ""
        end

        rest_server, rest_port = start_test_server(port ->
            serve_rest(model; host="127.0.0.1", port=port, background=true)
        )
        try
            sleep(0.2)
            health_resp = HTTP.get("http://127.0.0.1:$(rest_port)/health")
            @test health_resp.status == 200
            health_body = JSON.parse(String(health_resp.body))
            @test health_body["status"] == "ok"

            predict_resp = HTTP.post(
                "http://127.0.0.1:$(rest_port)/predict",
                ["Content-Type" => "application/json"],
                JSON.json(Dict("input" => input_batch))
            )
            @test predict_resp.status == 200
            predict_body = JSON.parse(String(predict_resp.body))
            @test haskey(predict_body, "output")
            @test length(predict_body["output"]) == 2
        finally
            close(rest_server)
        end

        gql_server, gql_port = start_test_server(port ->
            serve_graphql(model; host="127.0.0.1", port=port, background=true)
        )
        try
            sleep(0.2)
            gql_health = HTTP.post(
                "http://127.0.0.1:$(gql_port)/graphql",
                ["Content-Type" => "application/json"],
                JSON.json(Dict("query" => "query Health { health }"))
            )
            @test gql_health.status == 200
            gql_health_body = JSON.parse(String(gql_health.body))
            @test gql_health_body["data"]["health"] == "ok"

            gql_predict = HTTP.post(
                "http://127.0.0.1:$(gql_port)/graphql",
                ["Content-Type" => "application/json"],
                JSON.json(Dict(
                    "query" => raw"query Predict($input: [[Float!]!]) { predict(input: $input) }",
                    "variables" => Dict("input" => input_batch)
                ))
            )
            @test gql_predict.status == 200
            gql_predict_body = JSON.parse(String(gql_predict.body))
            @test haskey(gql_predict_body["data"], "predict")
            @test length(gql_predict_body["data"]["predict"]) == 2
        finally
            close(gql_server)
        end

        proto_path = tempname() * ".proto"
        out_path = generate_grpc_proto(proto_path)
        @test out_path == proto_path
        @test isfile(proto_path)
        proto_content = read(proto_path, String)
        @test occursin("service AxiomInference", proto_content)
        @test occursin("rpc Predict", proto_content)

        grpc_pred = grpc_predict(model, input_batch)
        @test haskey(grpc_pred, "output")
        @test length(grpc_pred["output"]) == 2

        grpc_server, grpc_port = start_test_server(port ->
            serve_grpc(model; host="127.0.0.1", port=port, background=true)
        )
        try
            sleep(0.2)
            grpc_health_resp = HTTP.post(
                "http://127.0.0.1:$(grpc_port)/axiom.v1.AxiomInference/Health",
                ["Content-Type" => "application/grpc+json"],
                "{}"
            )
            @test grpc_health_resp.status == 200
            grpc_health_body = JSON.parse(String(grpc_health_resp.body))
            @test grpc_health_body["status"] == "SERVING"

            grpc_predict_resp = HTTP.post(
                "http://127.0.0.1:$(grpc_port)/axiom.v1.AxiomInference/Predict",
                ["Content-Type" => "application/grpc+json"],
                JSON.json(Dict("input" => input_batch))
            )
            @test grpc_predict_resp.status == 200
            grpc_predict_body = JSON.parse(String(grpc_predict_resp.body))
            @test haskey(grpc_predict_body, "output")
            @test length(grpc_predict_body["output"]) == 2

            grpc_health_bin = HTTP.post(
                "http://127.0.0.1:$(grpc_port)/axiom.v1.AxiomInference/Health",
                ["Content-Type" => "application/grpc"],
                grpc_frame(UInt8[])
            )
            @test grpc_health_bin.status == 200
            @test HTTP.header(grpc_health_bin, "grpc-status", "") == "0"
            health_status = grpc_decode_health_response(grpc_unframe(Vector{UInt8}(grpc_health_bin.body)))
            @test health_status == "SERVING"

            predict_input = Float32.(input_batch[1])
            grpc_predict_bin = HTTP.post(
                "http://127.0.0.1:$(grpc_port)/axiom.v1.AxiomInference/Predict",
                ["Content-Type" => "application/grpc"],
                grpc_frame(grpc_predict_request(predict_input))
            )
            @test grpc_predict_bin.status == 200
            @test HTTP.header(grpc_predict_bin, "grpc-status", "") == "0"
            binary_output = grpc_decode_floats_field1(grpc_unframe(Vector{UInt8}(grpc_predict_bin.body)))
            @test length(binary_output) == 3
            @test isapprox(sum(binary_output), 1.0f0; atol=1f-4)
        finally
            close(grpc_server)
        end

        grpc_stat = grpc_support_status()
        @test grpc_stat["proto_generation"] == true
        @test grpc_stat["inprocess_handlers"] == true
        @test grpc_stat["network_server"] == true
        @test grpc_stat["binary_wire"] == true

        rm(proto_path)
    end

    @testset "Interop APIs" begin
        spec = Dict(
            "format" => "axiom.pytorch.sequential.v1",
            "layers" => Any[
                Dict(
                    "type" => "Linear",
                    "in_features" => 3,
                    "out_features" => 2,
                    "weight" => Any[
                        Any[1.0, 2.0, 3.0],
                        Any[-1.0, 0.5, 0.25],
                    ],
                    "bias" => Any[0.1, -0.2]
                ),
                Dict("type" => "ReLU"),
                Dict(
                    "type" => "Linear",
                    "in_features" => 2,
                    "out_features" => 1,
                    "weight" => Any[
                        Any[0.5, -1.0],
                    ],
                    "bias" => Any[0.0]
                ),
            ],
        )

        spec_path = tempname() * ".json"
        open(spec_path, "w") do io
            write(io, JSON.json(spec))
        end

        imported = from_pytorch(spec_path)
        @test imported isa Axiom.Pipeline

        x = Tensor(Float32[
            1.0 2.0 3.0;
            0.0 -1.0 2.0;
        ])
        y = imported(x)
        @test size(y) == (2, 1)

        onnx_path = tempname() * ".onnx"
        out_path = to_onnx(imported, onnx_path; input_shape=(1, 3))
        @test out_path == onnx_path
        @test isfile(onnx_path)

        onnx_bytes = read(onnx_path)
        @test length(onnx_bytes) > 0
        @test findfirst(Vector{UInt8}(codeunits("Gemm")), onnx_bytes) !== nothing
        @test findfirst(Vector{UInt8}(codeunits("Relu")), onnx_bytes) !== nothing

        pt_path = tempname() * ".pth"
        write(pt_path, "placeholder")
        @test_throws ArgumentError from_pytorch(pt_path; bridge=false)

        bridge_script = tempname() * ".py"
        open(bridge_script, "w") do io
            write(io, """
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--strict", action="store_true")
p.add_argument("--no-strict", action="store_false", dest="strict")
args = p.parse_args()

spec = {
  "format": "axiom.pytorch.sequential.v1",
  "layers": [
    {
      "type": "Linear",
      "in_features": 3,
      "out_features": 2,
      "weight": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
      "bias": [0.0, 0.0]
    },
    {"type": "ReLU"}
  ]
}

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(spec, f)
""")
        end
        bridged = from_pytorch(
            pt_path;
            python_cmd="python3",
            bridge_script=bridge_script,
            strict=true
        )
        @test bridged isa Axiom.Pipeline

        cv_model = Sequential(
            Conv2d(3, 4, (3, 3), padding=1),
            BatchNorm(4),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(4, 8, (3, 3), padding=1),
            AvgPool2d((2, 2)),
            GlobalAvgPool(),
            Dense(8, 4),
            Softmax()
        )
        cv_onnx_path_a = tempname() * ".onnx"
        cv_onnx_path_b = tempname() * ".onnx"
        to_onnx(cv_model, cv_onnx_path_a; input_shape=(1, 16, 16, 3))
        to_onnx(cv_model, cv_onnx_path_b; input_shape=(1, 16, 16, 3))
        cv_onnx_a = read(cv_onnx_path_a)
        cv_onnx_b = read(cv_onnx_path_b)
        @test cv_onnx_a == cv_onnx_b
        @test findfirst(Vector{UInt8}(codeunits("Conv")), cv_onnx_a) !== nothing
        @test findfirst(Vector{UInt8}(codeunits("BatchNormalization")), cv_onnx_a) !== nothing
        @test findfirst(Vector{UInt8}(codeunits("MaxPool")), cv_onnx_a) !== nothing
        @test findfirst(Vector{UInt8}(codeunits("AveragePool")), cv_onnx_a) !== nothing
        @test findfirst(Vector{UInt8}(codeunits("GlobalAveragePool")), cv_onnx_a) !== nothing

        rm(spec_path)
        rm(onnx_path)
        rm(pt_path)
        rm(bridge_script)
        rm(cv_onnx_path_a)
        rm(cv_onnx_path_b)
    end

    @testset "Verification" begin
        # Build model
        model = Sequential(
            Dense(10, 5, relu),
            Dense(5, 3),
            Softmax()
        )

        # Check output properties
        x = Tensor(randn(Float32, 4, 10))
        y = model(x)

        # Probabilities should sum to 1
        sums = sum(y.data, dims=2)
        @test all(isapprox.(sums, 1.0, atol=1e-5))

        # All values should be non-negative
        @test all(y.data .>= 0)

        # No NaN
        @test !any(isnan, y.data)

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

    @testset "SMT Runner" begin
        solver = Axiom.get_smt_solver()
        if solver === nothing
            @info "Skipping SMT runner test; no solver available"
            @test true
        else
            prop = Axiom.ParsedProperty(:exists, [:x], :(x > 0))
            result = Axiom.smt_proof(prop)
            @test result.status in (:proven, :unknown)
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

    # Include invertible / reversible computing tests
    include("test_invertible.jl")

    # Include proof serialization tests
    include("verification/serialization_tests.jl")
    include("verification/proof_export_tests.jl")

end

println("\nAll tests passed!")
