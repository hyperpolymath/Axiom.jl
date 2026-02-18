#!/usr/bin/env julia
# SPDX-License-Identifier: PMPL-1.0-or-later

using Random
using JSON
using Dates
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

function compile_required_probe(model)
    with_env(Dict(
        "AXIOM_CRYPTO_AVAILABLE" => "0",
        "AXIOM_CRYPTO_DEVICE_COUNT" => "0",
        "AXIOM_CRYPTO_REQUIRED" => "1",
    )) do
        err_message = nothing
        try
            compile(model, backend = CryptoBackend(0), verify = false, optimize = :none)
        catch err
            err_message = sprint(showerror, err)
        end

        Dict(
            "raised_error" => err_message !== nothing,
            "error_message" => err_message,
        )
    end
end

function run_compiled_probe(model, x, cpu)
    compile_ms = @elapsed compiled = compile(model, backend = CryptoBackend(0), verify = false, optimize = :none)
    infer_ms = @elapsed y = compiled(x).data

    Dict(
        "compiled_wrapper" => compiled isa Axiom.CoprocessorCompiledModel,
        "compile_ms" => round(compile_ms * 1000; digits = 3),
        "inference_ms" => round(infer_ms * 1000; digits = 3),
        "parity_ok" => isapprox(y, cpu; atol = 2f-4, rtol = 2f-4),
        "finite_ok" => all(isfinite, y),
        "probability_ok" => all(isapprox.(sum(y, dims = 2), 1.0f0, atol = 2f-4)),
    )
end

function strict_runtime_probe()
    dense_model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    dense_x = Tensor(randn(Float32, 6, 6))
    dense_cpu = dense_model(dense_x).data

    conv_model = Sequential(
        Conv2d(3, 4, (3, 3), padding = 1),
        BatchNorm(4),
        ReLU(),
        MaxPool2d((2, 2)),
        GlobalAvgPool(),
        Dense(4, 3),
        Softmax(),
    )
    conv_x = Tensor(randn(Float32, 2, 8, 8, 3))
    conv_cpu = conv_model(conv_x).data

    norm_model = Sequential(
        Dense(6, 6, identity),
        LayerNorm(6),
        ReLU(),
        Dense(6, 3),
        Softmax(),
    )
    norm_x = Tensor(randn(Float32, 4, 6))
    norm_cpu = norm_model(norm_x).data

    avgpool_model = Sequential(
        Conv2d(3, 4, (3, 3), padding = 1),
        AvgPool2d((2, 2)),
        GlobalAvgPool(),
        Dense(4, 3),
        Softmax(),
    )
    avgpool_x = Tensor(randn(Float32, 2, 8, 8, 3))
    avgpool_cpu = avgpool_model(avgpool_x).data

    with_env(Dict(
        "AXIOM_CRYPTO_AVAILABLE" => "1",
        "AXIOM_CRYPTO_DEVICE_COUNT" => "1",
        "AXIOM_CRYPTO_REQUIRED" => "1",
    )) do
        reset_coprocessor_runtime_diagnostics!()

        dense = run_compiled_probe(dense_model, dense_x, dense_cpu)
        conv = run_compiled_probe(conv_model, conv_x, conv_cpu)
        norm = run_compiled_probe(norm_model, norm_x, norm_cpu)
        avgpool = run_compiled_probe(avgpool_model, avgpool_x, avgpool_cpu)

        report = coprocessor_capability_report()
        crypto = report["backends"]["CRYPTO"]
        diagnostics = coprocessor_runtime_diagnostics()["backends"]["crypto"]

        Dict(
            "dense" => dense,
            "conv" => conv,
            "norm" => norm,
            "avgpool" => avgpool,
            "required" => crypto["required"],
            "kernel_hooks_loaded" => crypto["kernel_hooks_loaded"],
            "hook_overrides" => crypto["hook_overrides"],
            "runtime_diagnostics" => diagnostics,
        )
    end
end

function main()
    failures = String[]

    compile_probe = compile_required_probe(Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    ))
    (compile_probe["raised_error"] == true) || push!(failures, "Crypto strict compile probe did not raise an error")
    if compile_probe["error_message"] !== nothing
        occursin("AXIOM_CRYPTO_REQUIRED", compile_probe["error_message"]) ||
            push!(failures, "Crypto strict compile probe error message missing AXIOM_CRYPTO_REQUIRED hint")
    end

    strict_runtime = strict_runtime_probe()
    (strict_runtime["required"] == true) || push!(failures, "Crypto strict runtime probe did not report required=true")
    (strict_runtime["kernel_hooks_loaded"] == true) || push!(failures, "Crypto strict runtime probe did not report full hook coverage")
    for key in ("dense", "conv", "norm", "avgpool")
        probe = strict_runtime[key]
        (probe["compiled_wrapper"] == true) || push!(failures, "Crypto strict runtime probe for $key did not compile to coprocessor wrapper")
        (probe["parity_ok"] == true) || push!(failures, "Crypto strict runtime probe parity check failed for $key")
        (probe["finite_ok"] == true) || push!(failures, "Crypto strict runtime finite check failed for $key")
        (probe["probability_ok"] == true) || push!(failures, "Crypto strict runtime probability check failed for $key")
    end

    hooks = strict_runtime["hook_overrides"]
    for hook in (
        "backend_coprocessor_matmul",
        "backend_coprocessor_conv2d",
        "backend_coprocessor_relu",
        "backend_coprocessor_softmax",
        "backend_coprocessor_batchnorm",
        "backend_coprocessor_layernorm",
        "backend_coprocessor_maxpool2d",
        "backend_coprocessor_avgpool2d",
        "backend_coprocessor_global_avgpool2d",
    )
        Bool(hooks[hook]) || push!(failures, "Crypto hook override not detected for $hook")
    end

    diagnostics = strict_runtime["runtime_diagnostics"]
    (diagnostics["compile_fallbacks"] == 0) || push!(failures, "Crypto compile fallback counter should remain 0 in strict runtime probe")
    (diagnostics["runtime_errors"] == 0) || push!(failures, "Crypto runtime error counter should remain 0 in strict runtime probe")
    (diagnostics["runtime_fallbacks"] == 0) || push!(failures, "Crypto runtime fallback counter should remain 0 in strict runtime probe")
    (diagnostics["recoveries"] == 0) || push!(failures, "Crypto recovery counter should remain 0 in strict runtime probe")

    payload = Dict(
        "format" => "axiom-crypto-strict-evidence.v2",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "compile_required_probe" => compile_probe,
        "strict_runtime_probe" => strict_runtime,
        "capability_report" => coprocessor_capability_report(),
    )

    out_path = get(ENV, "AXIOM_CRYPTO_STRICT_EVIDENCE_PATH", joinpath(pwd(), "build", "crypto_strict_evidence.json"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, payload, 2)
    end

    println("crypto strict evidence written: $out_path")

    if !isempty(failures)
        println("crypto strict evidence checks failed:")
        for failure in failures
            println(" - $failure")
        end
        exit(1)
    end
end

main()
