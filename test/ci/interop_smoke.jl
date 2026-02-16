# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Random
using JSON
using Axiom

Random.seed!(0xA710)

@testset "Interop smoke (from_pytorch/to_onnx)" begin
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
