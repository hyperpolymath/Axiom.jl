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
        "AXIOM_TPU_AVAILABLE" => "0",
        "AXIOM_TPU_DEVICE_COUNT" => "0",
        "AXIOM_TPU_REQUIRED" => "1",
    )) do
        err_message = nothing
        try
            compile(model, backend = TPUBackend(0), verify = false, optimize = :none)
        catch err
            err_message = sprint(showerror, err)
        end

        Dict(
            "raised_error" => err_message !== nothing,
            "error_message" => err_message,
        )
    end
end

function missing_hook_probe(model, x)
    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_TPU_REQUIRED" => "1",
    )) do
        compiled = compile(model, backend = TPUBackend(0), verify = false, optimize = :none)

        err_message = nothing
        try
            compiled(x)
        catch err
            err_message = sprint(showerror, err)
        end

        Dict(
            "compiled_wrapper" => compiled isa Axiom.CoprocessorCompiledModel,
            "raised_error" => err_message !== nothing,
            "error_message" => err_message,
        )
    end
end

function install_tpu_demo_hooks!()
    @eval begin
        function Axiom.backend_coprocessor_matmul(
            backend::Axiom.TPUBackend,
            A::AbstractMatrix{Float32},
            B::AbstractMatrix{Float32},
        )
            A * B
        end

        function Axiom.backend_coprocessor_relu(
            backend::Axiom.TPUBackend,
            x::AbstractArray{Float32},
        )
            max.(x, 0f0)
        end

        function Axiom.backend_coprocessor_softmax(
            backend::Axiom.TPUBackend,
            x::AbstractArray{Float32},
            dim::Int,
        )
            Axiom.softmax(x, dims = dim)
        end
    end
    nothing
end

function strict_success_probe(model, x, cpu)
    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_TPU_REQUIRED" => "1",
    )) do
        compile_ms = @elapsed compiled = compile(model, backend = TPUBackend(0), verify = false, optimize = :none)
        # Hooks are installed at runtime in this script, so run inference in latest world-age.
        infer_ms = @elapsed y = Base.invokelatest(compiled, x).data

        report = coprocessor_capability_report()
        tpu = report["backends"]["TPU"]

        Dict(
            "compiled_wrapper" => compiled isa Axiom.CoprocessorCompiledModel,
            "compile_ms" => round(compile_ms * 1000; digits = 3),
            "inference_ms" => round(infer_ms * 1000; digits = 3),
            "parity_ok" => isapprox(y, cpu; atol = 1f-5, rtol = 1f-5),
            "finite_ok" => all(isfinite, y),
            "probability_ok" => all(isapprox.(sum(y, dims = 2), 1.0f0, atol = 2f-4)),
            "required" => tpu["required"],
            "hook_overrides" => tpu["hook_overrides"],
        )
    end
end

function main()
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 6, 6))
    cpu = model(x).data

    failures = String[]

    compile_probe = compile_required_probe(model)
    (compile_probe["raised_error"] == true) || push!(failures, "TPU strict compile probe did not raise an error")
    if compile_probe["error_message"] !== nothing
        occursin("AXIOM_TPU_REQUIRED", compile_probe["error_message"]) ||
            push!(failures, "TPU strict compile probe error message missing AXIOM_TPU_REQUIRED hint")
    end

    missing_hook = missing_hook_probe(model, x)
    (missing_hook["compiled_wrapper"] == true) || push!(failures, "TPU strict missing-hook probe did not compile to coprocessor wrapper")
    (missing_hook["raised_error"] == true) || push!(failures, "TPU strict missing-hook probe did not raise an error")
    if missing_hook["error_message"] !== nothing
        occursin("strict mode enabled", missing_hook["error_message"]) ||
            push!(failures, "TPU strict missing-hook probe error message missing strict-mode wording")
    end

    install_tpu_demo_hooks!()
    strict_ok = strict_success_probe(model, x, cpu)

    (strict_ok["compiled_wrapper"] == true) || push!(failures, "TPU strict success probe did not compile to coprocessor wrapper")
    (strict_ok["required"] == true) || push!(failures, "TPU strict success probe did not report required=true")
    (strict_ok["parity_ok"] == true) || push!(failures, "TPU strict success probe parity check failed")
    (strict_ok["finite_ok"] == true) || push!(failures, "TPU strict success probe finite check failed")
    (strict_ok["probability_ok"] == true) || push!(failures, "TPU strict success probe probability check failed")

    hooks = strict_ok["hook_overrides"]
    Bool(hooks["backend_coprocessor_matmul"]) || push!(failures, "TPU hook override not detected for matmul")
    Bool(hooks["backend_coprocessor_relu"]) || push!(failures, "TPU hook override not detected for relu")
    Bool(hooks["backend_coprocessor_softmax"]) || push!(failures, "TPU hook override not detected for softmax")

    payload = Dict(
        "format" => "axiom-tpu-strict-evidence.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "compile_required_probe" => compile_probe,
        "missing_hook_probe" => missing_hook,
        "strict_success_probe" => strict_ok,
        "capability_report" => coprocessor_capability_report(),
    )

    out_path = get(ENV, "AXIOM_TPU_STRICT_EVIDENCE_PATH", joinpath(pwd(), "build", "tpu_strict_evidence.json"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, payload, 2)
    end

    println("tpu strict evidence written: $out_path")

    if !isempty(failures)
        println("tpu strict evidence checks failed:")
        for failure in failures
            println(" - $failure")
        end
        exit(1)
    end
end

main()
