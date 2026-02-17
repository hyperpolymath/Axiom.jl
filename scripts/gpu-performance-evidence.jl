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

function backend_instance(name::String)
    if name == "cuda"
        return CUDABackend(0)
    elseif name == "rocm"
        return ROCmBackend(0)
    elseif name == "metal"
        return MetalBackend(0)
    end
    error("Unsupported GPU backend: $name")
end

function backend_state(name::String)
    if name == "cuda"
        return (available = cuda_available(), device_count = cuda_device_count())
    elseif name == "rocm"
        return (available = rocm_available(), device_count = rocm_device_count())
    elseif name == "metal"
        return (available = metal_available(), device_count = metal_device_count())
    end
    (available = false, device_count = 0)
end

function normalize_backend_targets(raw_target::String)
    target = lowercase(strip(raw_target))
    if target in ("", "all")
        return ["cuda", "rocm", "metal"]
    end
    target in ("cuda", "rocm", "metal") || error("AXIOM_GPU_BASELINE_BACKEND must be one of: all, cuda, rocm, metal")
    [target]
end

function load_baseline(path::String)
    if !isfile(path)
        return Dict{String, Any}()
    end
    try
        JSON.parsefile(path)
    catch err
        @warn "Failed to parse GPU baseline file; ignoring baseline checks" path = path exception = (err, catch_backtrace())
        Dict{String, Any}()
    end
end

function baseline_p50_ms(baseline::Dict{String, Any}, backend_name::String)
    backends = get(baseline, "backends", Dict{String, Any}())
    haskey(backends, backend_name) || return nothing
    entry = backends[backend_name]
    entry isa Dict || return nothing
    haskey(entry, "inference_p50_ms") || return nothing
    value = tryparse(Float64, string(entry["inference_p50_ms"]))
    value === nothing ? nothing : value
end

function main()
    target_raw = get(ENV, "AXIOM_GPU_BASELINE_BACKEND", get(ENV, "AXIOM_GPU_BACKEND", "all"))
    targets = normalize_backend_targets(target_raw)
    required = parse_bool_env("AXIOM_GPU_REQUIRED", false)
    enforce_regression = parse_bool_env("AXIOM_GPU_BASELINE_ENFORCE", false)
    warmup = parse_int_env("AXIOM_GPU_PERF_WARMUP", 3)
    iterations = parse_int_env("AXIOM_GPU_PERF_ITERATIONS", 20)
    max_ratio = parse_float_env("AXIOM_GPU_MAX_REGRESSION_RATIO", 1.20)
    baseline_path = get(ENV, "AXIOM_GPU_BASELINE_PATH", joinpath(pwd(), "benchmark", "gpu_performance_baseline.json"))
    out_path = get(ENV, "AXIOM_GPU_EVIDENCE_PATH", joinpath(pwd(), "build", "gpu_performance_evidence.json"))

    model = Sequential(
        Dense(128, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 10),
        Softmax(),
    )
    x = Tensor(randn(Float32, 128, 128))
    cpu_reference = model(x).data
    cpu_perf = benchmark_model(model, x; warmup = warmup, iterations = iterations)

    baseline = load_baseline(baseline_path)
    results = Any[]
    failures = String[]

    for name in targets
        backend = backend_instance(name)
        state = backend_state(name)
        compile_ms = @elapsed compiled = compile(model, backend = backend, verify = false, optimize = :none)
        compiled_wrapper = compiled !== model
        y = compiled(x).data

        parity_ok = isapprox(y, cpu_reference; atol = 2f-4, rtol = 2f-4)
        finite_ok = all(isfinite, y)

        entry = Dict(
            "backend" => name,
            "available" => state.available,
            "device_count" => state.device_count,
            "compiled_wrapper" => compiled_wrapper,
            "compile_ms" => round(compile_ms * 1000; digits = 3),
            "parity_ok" => parity_ok,
            "finite_ok" => finite_ok,
        )

        if compiled_wrapper
            perf = benchmark_model(compiled, x; warmup = warmup, iterations = iterations)
            p50_ms = Float64(perf["p50_ms"])
            cpu_p50_ms = Float64(cpu_perf["p50_ms"])
            speedup = p50_ms > 0 ? round(cpu_p50_ms / p50_ms; digits = 3) : Inf
            entry["inference"] = perf
            entry["speedup_vs_cpu_p50"] = speedup

            baseline_p50 = baseline_p50_ms(baseline, name)
            if baseline_p50 === nothing
                entry["baseline"] = Dict(
                    "present" => false,
                    "ratio_vs_baseline" => nothing,
                    "max_allowed_ratio" => max_ratio,
                    "regressed" => false,
                )
            else
                ratio = p50_ms / baseline_p50
                regressed = ratio > max_ratio
                entry["baseline"] = Dict(
                    "present" => true,
                    "baseline_p50_ms" => round(baseline_p50; digits = 3),
                    "ratio_vs_baseline" => round(ratio; digits = 3),
                    "max_allowed_ratio" => max_ratio,
                    "regressed" => regressed,
                )
                if enforce_regression && regressed
                    push!(failures, "$name regression ratio $(round(ratio; digits = 3)) exceeds $(round(max_ratio; digits = 3))")
                end
            end
        else
            entry["inference"] = Dict(
                "samples_ms" => Float64[],
                "min_ms" => nothing,
                "p50_ms" => nothing,
                "p95_ms" => nothing,
                "max_ms" => nothing,
            )
            entry["speedup_vs_cpu_p50"] = nothing
            entry["baseline"] = Dict(
                "present" => false,
                "ratio_vs_baseline" => nothing,
                "max_allowed_ratio" => max_ratio,
                "regressed" => false,
            )

            if required
                push!(failures, "$name backend requested but no compiled GPU wrapper was produced")
            end
        end

        parity_ok || push!(failures, "$name backend parity mismatch against CPU reference")
        finite_ok || push!(failures, "$name backend produced non-finite outputs")
        push!(results, entry)
    end

    payload = Dict(
        "format" => "axiom-gpu-performance-evidence.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "config" => Dict(
            "targets" => targets,
            "required" => required,
            "enforce_regression" => enforce_regression,
            "max_regression_ratio" => max_ratio,
            "warmup_iterations" => warmup,
            "benchmark_iterations" => iterations,
            "baseline_path" => baseline_path,
        ),
        "cpu_reference" => cpu_perf,
        "gpu_capabilities" => gpu_capability_report(),
        "backend_results" => results,
    )

    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, payload, 2)
    end

    println("gpu performance evidence written: $out_path")
    if !isempty(failures)
        println("gpu performance evidence checks failed:")
        for failure in failures
            println(" - $failure")
        end
        exit(1)
    end
end

main()
