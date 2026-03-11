#!/usr/bin/env julia
# SPDX-License-Identifier: PMPL-1.0-or-later

using Random
using JSON
using Dates
using Axiom

Random.seed!(0xA710)

function backend_probe_entry(model, x, cpu, backend)
    compile_ms = @elapsed compiled = compile(model, backend = backend, verify = false, optimize = :none)
    infer_ms = @elapsed y = compiled(x).data

    is_compiled_wrapper = compiled isa Axiom.CoprocessorCompiledModel
    finite_ok = all(isfinite, y)
    prob_ok = all(isapprox.(sum(y, dims = 2), 1.0f0, atol = 2f-4))
    parity_ok = isapprox(y, cpu; atol = 2f-4, rtol = 2f-4)

    Dict(
        "backend" => string(typeof(backend)),
        "device" => backend.device,
        "compiled_wrapper" => is_compiled_wrapper,
        "compile_ms" => round(compile_ms * 1000; digits = 3),
        "inference_ms" => round(infer_ms * 1000; digits = 3),
        "finite_ok" => finite_ok,
        "probability_ok" => prob_ok,
        "parity_ok" => parity_ok,
    )
end

function main()
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 8, 6))
    cpu = model(x).data

    report = coprocessor_capability_report()

    probes = Any[]
    for backend in (TPUBackend(0), NPUBackend(0), DSPBackend(0), PPUBackend(0), MathBackend(0), FPGABackend(0))
        push!(probes, backend_probe_entry(model, x, cpu, backend))
    end

    payload = Dict(
        "format" => "axiom-coprocessor-evidence.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "capability_report" => report,
        "probe_results" => probes,
    )

    out_path = get(ENV, "AXIOM_COPROCESSOR_EVIDENCE_PATH", joinpath(pwd(), "build", "coprocessor_evidence.json"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, payload, 2)
    end

    println("coprocessor evidence written: $out_path")
end

main()
