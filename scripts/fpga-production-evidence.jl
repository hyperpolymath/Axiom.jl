#!/usr/bin/env julia
# SPDX-License-Identifier: PMPL-1.0-or-later

using Random
using JSON
using Dates
using Axiom

Random.seed!(0xA710)

function parse_bool_env(key::String, default::Bool)
    raw = lowercase(strip(get(ENV, key, default ? "1" : "0")))
    raw in ("1", "true", "yes", "on")
end

function parse_int_env(key::String, default::Int)
    raw = strip(get(ENV, key, ""))
    isempty(raw) && return default
    parsed = tryparse(Int, raw)
    parsed === nothing ? default : max(parsed, 1)
end

function parse_float_env(key::String, default::Float64)
    raw = strip(get(ENV, key, ""))
    isempty(raw) && return default
    parsed = tryparse(Float64, raw)
    parsed === nothing ? default : parsed
end

function percentile(sorted_values::Vector{Float64}, q::Float64)
    isempty(sorted_values) && return nothing
    idx = clamp(Int(ceil(q * length(sorted_values))), 1, length(sorted_values))
    sorted_values[idx]
end

function summarize_latency_ms(samples_ms::Vector{Float64})
    sorted = sort(samples_ms)
    Dict(
        "samples_ms" => round.(samples_ms; digits = 6),
        "min_ms" => round(minimum(samples_ms); digits = 3),
        "p50_ms" => round(percentile(sorted, 0.50); digits = 3),
        "p95_ms" => round(percentile(sorted, 0.95); digits = 3),
        "max_ms" => round(maximum(samples_ms); digits = 3),
    )
end

function benchmark_model(compiled, x; warmup::Int, iterations::Int)
    for _ in 1:warmup
        compiled(x)
    end

    samples_ms = Vector{Float64}(undef, iterations)
    for i in 1:iterations
        start_ns = time_ns()
        compiled(x)
        finish_ns = time_ns()
        samples_ms[i] = (finish_ns - start_ns) / 1_000_000
    end

    summarize_latency_ms(samples_ms)
end

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

function load_baseline(path::String)
    if !isfile(path)
        return Dict{String, Any}()
    end
    try
        JSON.parsefile(path)
    catch err
        @warn "Failed to parse fpga baseline file; ignoring baseline checks" path = path exception = (err, catch_backtrace())
        Dict{String, Any}()
    end
end

function baseline_p50_ms(baseline::Dict{String, Any})
    haskey(baseline, "fpga_p50_ms") || return nothing
    value = tryparse(Float64, string(baseline["fpga_p50_ms"]))
    value === nothing ? nothing : value
end

function main()
    required = parse_bool_env("AXIOM_FPGA_REQUIRED", true)
    enforce_regression = parse_bool_env("AXIOM_FPGA_BASELINE_ENFORCE", false)
    warmup = parse_int_env("AXIOM_FPGA_PERF_WARMUP", 3)
    iterations = parse_int_env("AXIOM_FPGA_PERF_ITERATIONS", 20)
    max_ratio = parse_float_env("AXIOM_FPGA_MAX_REGRESSION_RATIO", 1.20)
    baseline_path = get(ENV, "AXIOM_FPGA_BASELINE_PATH", joinpath(pwd(), "benchmark", "fpga_performance_baseline.json"))
    out_path = get(ENV, "AXIOM_FPGA_PRODUCTION_EVIDENCE_PATH", joinpath(pwd(), "build", "fpga_production_evidence.json"))

    model = Sequential(
        Dense(128, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 10),
        Softmax(),
    )
    x = Tensor(randn(Float32, 128, 128))
    cpu_reference = model(x).data
    cpu_perf = benchmark_model(model, x; warmup = warmup, iterations = iterations)

    failures = String[]
    baseline = load_baseline(baseline_path)

    result = with_env(Dict(
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_REQUIRED" => required ? "1" : "0",
    )) do
        reset_coprocessor_runtime_diagnostics!()
        compile_ms = @elapsed compiled = compile(model, backend = FPGABackend(0), verify = false, optimize = :none)
        infer_perf = benchmark_model(compiled, x; warmup = warmup, iterations = iterations)
        y = compiled(x).data

        report = coprocessor_capability_report()
        fpga = report["backends"]["FPGA"]
        diagnostics = coprocessor_runtime_diagnostics()["backends"]["fpga"]

        Dict(
            "compiled_wrapper" => compiled isa Axiom.CoprocessorCompiledModel,
            "compile_ms" => round(compile_ms * 1000; digits = 3),
            "inference" => infer_perf,
            "parity_ok" => isapprox(y, cpu_reference; atol = 2f-4, rtol = 2f-4),
            "finite_ok" => all(isfinite, y),
            "probability_ok" => all(isapprox.(sum(y, dims = 2), 1.0f0, atol = 2f-4)),
            "hook_coverage_ok" => Bool(fpga["kernel_hooks_loaded"]) && all(values(fpga["hook_overrides"])),
            "required" => Bool(fpga["required"]),
            "runtime_diagnostics" => diagnostics,
        )
    end

    baseline_p50 = baseline_p50_ms(baseline)
    baseline_report = Dict{String, Any}()
    if baseline_p50 === nothing
        baseline_report["present"] = false
        baseline_report["ratio_vs_baseline"] = nothing
        baseline_report["max_allowed_ratio"] = max_ratio
        baseline_report["regressed"] = false
    else
        p50_ms = Float64(result["inference"]["p50_ms"])
        ratio = p50_ms / baseline_p50
        regressed = ratio > max_ratio
        baseline_report["present"] = true
        baseline_report["baseline_p50_ms"] = round(baseline_p50; digits = 3)
        baseline_report["ratio_vs_baseline"] = round(ratio; digits = 3)
        baseline_report["max_allowed_ratio"] = max_ratio
        baseline_report["regressed"] = regressed
        if enforce_regression && regressed
            push!(failures, "FPGA regression ratio $(round(ratio; digits = 3)) exceeds $(round(max_ratio; digits = 3))")
        end
    end

    result["compiled_wrapper"] || push!(failures, "FPGA backend did not produce a coprocessor compiled wrapper")
    result["parity_ok"] || push!(failures, "FPGA backend parity mismatch against CPU reference")
    result["finite_ok"] || push!(failures, "FPGA backend produced non-finite outputs")
    result["probability_ok"] || push!(failures, "FPGA backend output probabilities are invalid")
    result["hook_coverage_ok"] || push!(failures, "FPGA backend hook coverage is incomplete")

    diagnostics = result["runtime_diagnostics"]
    Int(diagnostics["compile_fallbacks"]) == 0 || push!(failures, "FPGA compile fallback counter is non-zero")
    Int(diagnostics["runtime_errors"]) == 0 || push!(failures, "FPGA runtime error counter is non-zero")
    Int(diagnostics["runtime_fallbacks"]) == 0 || push!(failures, "FPGA runtime fallback counter is non-zero")

    payload = Dict(
        "format" => "axiom-fpga-production-evidence.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "config" => Dict(
            "required" => required,
            "enforce_regression" => enforce_regression,
            "max_regression_ratio" => max_ratio,
            "warmup_iterations" => warmup,
            "benchmark_iterations" => iterations,
            "baseline_path" => baseline_path,
        ),
        "cpu_reference" => cpu_perf,
        "fpga_capabilities" => coprocessor_capability_report()["backends"]["FPGA"],
        "fpga_backend_result" => result,
        "baseline" => baseline_report,
    )

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, payload, 2)
    end

    println("fpga production evidence written: $out_path")
    if !isempty(failures)
        println("fpga production evidence checks failed:")
        for failure in failures
            println(" - $failure")
        end
        exit(1)
    end
end

main()
