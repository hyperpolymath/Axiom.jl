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

@testset "Coprocessor strategy and fallback behavior" begin
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax()
    )
    x = Tensor(randn(Float32, 5, 6))
    cpu = model(x).data

    conv_model = Sequential(
        Conv2d(3, 4, (3, 3), padding = 1),
        BatchNorm(4),
        ReLU(),
        MaxPool2d((2, 2)),
        GlobalAvgPool(),
        Dense(4, 3),
        Softmax()
    )
    conv_x = Tensor(randn(Float32, 2, 8, 8, 3))
    conv_cpu = conv_model(conv_x).data

    norm_model = Sequential(
        Dense(6, 6, identity),
        LayerNorm(6),
        ReLU(),
        Dense(6, 3),
        Softmax()
    )
    norm_x = Tensor(randn(Float32, 4, 6))
    norm_cpu = norm_model(norm_x).data

    backends = [
        ("TPU", TPUBackend(0), "AXIOM_TPU_AVAILABLE", "AXIOM_TPU_DEVICE_COUNT"),
        ("NPU", NPUBackend(0), "AXIOM_NPU_AVAILABLE", "AXIOM_NPU_DEVICE_COUNT"),
        ("PPU", PPUBackend(0), "AXIOM_PPU_AVAILABLE", "AXIOM_PPU_DEVICE_COUNT"),
        ("MATH", MathBackend(0), "AXIOM_MATH_AVAILABLE", "AXIOM_MATH_DEVICE_COUNT"),
        ("DSP", DSPBackend(0), "AXIOM_DSP_AVAILABLE", "AXIOM_DSP_DEVICE_COUNT"),
        ("FPGA", FPGABackend(0), "AXIOM_FPGA_AVAILABLE", "AXIOM_FPGA_DEVICE_COUNT"),
    ]

    for (label, backend, available_key, count_key) in backends
        with_env(Dict(
            available_key => "0",
            count_key => "0",
        )) do
            @test compile(model, backend = backend, verify = false, optimize = :none) === model
        end

        with_env(Dict(
            available_key => "1",
            count_key => "2",
        )) do
            compiled = compile(model, backend = backend, verify = false, optimize = :none)
            @test compiled isa Axiom.CoprocessorCompiledModel
            y = compiled(x).data
            @test size(y) == size(cpu)
            @test isapprox(y, cpu; atol = 1f-5, rtol = 1f-5)
            @test all(isfinite, y)
            @test all(isapprox.(sum(y, dims = 2), 1.0f0, atol = 1f-4))

            conv_compiled = compile(conv_model, backend = backend, verify = false, optimize = :none)
            conv_y = conv_compiled(conv_x).data
            @test size(conv_y) == size(conv_cpu)
            @test isapprox(conv_y, conv_cpu; atol = 2f-4, rtol = 2f-4)
            @test all(isfinite, conv_y)
            @test all(isapprox.(sum(conv_y, dims = 2), 1.0f0, atol = 2f-4))

            norm_compiled = compile(norm_model, backend = backend, verify = false, optimize = :none)
            norm_y = norm_compiled(norm_x).data
            @test size(norm_y) == size(norm_cpu)
            @test isapprox(norm_y, norm_cpu; atol = 1f-4, rtol = 1f-4)
            @test all(isfinite, norm_y)
            @test all(isapprox.(sum(norm_y, dims = 2), 1.0f0, atol = 2f-4))
        end

        with_env(Dict(
            available_key => "1",
            count_key => "1",
        )) do
            out_of_range = select_device!(backend, 9)
            @test compile(model, backend = out_of_range, verify = false, optimize = :none) === model
        end
    end

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_NPU_AVAILABLE" => "1",
        "AXIOM_PPU_AVAILABLE" => "1",
        "AXIOM_MATH_AVAILABLE" => "1",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_NPU_DEVICE_COUNT" => "1",
        "AXIOM_PPU_DEVICE_COUNT" => "1",
        "AXIOM_MATH_DEVICE_COUNT" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
        "AXIOM_CUDA_AVAILABLE" => "0",
        "AXIOM_ROCM_AVAILABLE" => "0",
        "AXIOM_METAL_AVAILABLE" => "0",
        "AXIOM_CUDA_DEVICE_COUNT" => "0",
        "AXIOM_ROCM_DEVICE_COUNT" => "0",
    )) do
        @test detect_coprocessor() isa TPUBackend
        @test detect_accelerator() isa TPUBackend
    end

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "0",
        "AXIOM_NPU_AVAILABLE" => "1",
        "AXIOM_PPU_AVAILABLE" => "1",
        "AXIOM_MATH_AVAILABLE" => "1",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "0",
        "AXIOM_NPU_DEVICE_COUNT" => "1",
        "AXIOM_PPU_DEVICE_COUNT" => "1",
        "AXIOM_MATH_DEVICE_COUNT" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
    )) do
        @test detect_coprocessor() isa NPUBackend
    end

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "0",
        "AXIOM_NPU_AVAILABLE" => "0",
        "AXIOM_PPU_AVAILABLE" => "1",
        "AXIOM_MATH_AVAILABLE" => "1",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "0",
        "AXIOM_NPU_DEVICE_COUNT" => "0",
        "AXIOM_PPU_DEVICE_COUNT" => "1",
        "AXIOM_MATH_DEVICE_COUNT" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
    )) do
        @test detect_coprocessor() isa PPUBackend
    end

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "0",
        "AXIOM_NPU_AVAILABLE" => "0",
        "AXIOM_PPU_AVAILABLE" => "0",
        "AXIOM_MATH_AVAILABLE" => "1",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "0",
        "AXIOM_NPU_DEVICE_COUNT" => "0",
        "AXIOM_PPU_DEVICE_COUNT" => "0",
        "AXIOM_MATH_DEVICE_COUNT" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
    )) do
        @test detect_coprocessor() isa MathBackend
    end

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "0",
        "AXIOM_NPU_AVAILABLE" => "0",
        "AXIOM_PPU_AVAILABLE" => "0",
        "AXIOM_MATH_AVAILABLE" => "0",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "0",
        "AXIOM_NPU_DEVICE_COUNT" => "0",
        "AXIOM_PPU_DEVICE_COUNT" => "0",
        "AXIOM_MATH_DEVICE_COUNT" => "0",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
    )) do
        @test detect_coprocessor() isa FPGABackend
    end

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "0",
        "AXIOM_NPU_AVAILABLE" => "0",
        "AXIOM_PPU_AVAILABLE" => "0",
        "AXIOM_MATH_AVAILABLE" => "0",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_FPGA_AVAILABLE" => "0",
        "AXIOM_TPU_DEVICE_COUNT" => "0",
        "AXIOM_NPU_DEVICE_COUNT" => "0",
        "AXIOM_PPU_DEVICE_COUNT" => "0",
        "AXIOM_MATH_DEVICE_COUNT" => "0",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "0",
    )) do
        @test detect_coprocessor() isa DSPBackend
    end

    @testset "Coprocessor capability report" begin
        with_env(Dict(
            "AXIOM_TPU_AVAILABLE" => "1",
            "AXIOM_NPU_AVAILABLE" => "0",
            "AXIOM_PPU_AVAILABLE" => "0",
            "AXIOM_MATH_AVAILABLE" => "0",
            "AXIOM_DSP_AVAILABLE" => "0",
            "AXIOM_FPGA_AVAILABLE" => "0",
            "AXIOM_TPU_DEVICE_COUNT" => "2",
            "AXIOM_NPU_DEVICE_COUNT" => "0",
            "AXIOM_PPU_DEVICE_COUNT" => "0",
            "AXIOM_MATH_DEVICE_COUNT" => "0",
            "AXIOM_DSP_DEVICE_COUNT" => "0",
            "AXIOM_FPGA_DEVICE_COUNT" => "0",
        )) do
            report = coprocessor_capability_report()
            @test report["strategy_order"] == ["TPU", "NPU", "PPU", "MATH", "FPGA", "DSP"]
            @test occursin("TPUBackend", report["selected_backend"])
            @test haskey(report, "runtime_diagnostics")
            @test report["runtime_diagnostics"]["self_healing_enabled"] == true
            tpu = report["backends"]["TPU"]
            @test tpu["available"] == true
            @test tpu["device_count"] == 2
            @test tpu["required"] == false
            @test tpu["compilable"] == true
            @test haskey(tpu["hook_overrides"], "backend_coprocessor_matmul")
            @test tpu["hook_overrides"]["backend_coprocessor_matmul"] == false
            npu = report["backends"]["NPU"]
            @test npu["required"] == false
            ppu = report["backends"]["PPU"]
            @test ppu["required"] == false
            math = report["backends"]["MATH"]
            @test math["required"] == false
            dsp = report["backends"]["DSP"]
            @test dsp["required"] == false
        end
    end
end
