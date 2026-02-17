# SPDX-License-Identifier: PMPL-1.0-or-later

using Axiom
using JSON
using Dates

function avg_latency_ms(model, x; iterations::Int = 60)
    model(x)  # warmup
    elapsed = @elapsed begin
        for _ in 1:iterations
            model(x)
        end
    end
    (elapsed / iterations) * 1000
end

function run()
    model = Sequential(
        Dense(12, 16),
        ReLU(),
        Dense(16, 8),
        ReLU(),
        Dense(8, 4),
        Softmax(),
    )
    x = Tensor(randn(Float32, 32, 12))

    none_model = compile(model; backend = JuliaBackend(), optimize = :none, precision = :float32, verify = false)
    default_model = compile(model; backend = JuliaBackend(), optimize = :default, precision = :float32, verify = false)
    aggressive_model = compile(model; backend = JuliaBackend(), optimize = :aggressive, precision = :float32, verify = false)
    mixed_model = compile(model; backend = JuliaBackend(), optimize = :default, precision = :mixed, verify = false)

    y_none = none_model(x).data
    y_default = default_model(x).data
    y_aggressive = aggressive_model(x).data
    y_mixed = mixed_model(x).data

    drift_default = maximum(abs.(y_none .- y_default))
    drift_aggressive = maximum(abs.(y_none .- y_aggressive))
    drift_mixed = maximum(abs.(y_none .- y_mixed))

    evidence = Dict{String, Any}(
        "format" => "axiom-optimization-evidence.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "flags" => Dict(
            "none" => Dict("optimize" => "none", "precision" => "float32"),
            "default" => Dict("optimize" => "default", "precision" => "float32"),
            "aggressive" => Dict("optimize" => "aggressive", "precision" => "float32"),
            "mixed" => Dict("optimize" => "default", "precision" => "mixed"),
        ),
        "layer_counts" => Dict(
            "original" => length(model.layers),
            "default" => length(default_model.layers),
            "aggressive" => length(aggressive_model.layers),
        ),
        "latency_ms" => Dict(
            "none" => avg_latency_ms(none_model, x),
            "default" => avg_latency_ms(default_model, x),
            "aggressive" => avg_latency_ms(aggressive_model, x),
            "mixed" => avg_latency_ms(mixed_model, x),
        ),
        "drift_max_abs" => Dict(
            "default_vs_none" => drift_default,
            "aggressive_vs_none" => drift_aggressive,
            "mixed_vs_none" => drift_mixed,
        ),
    )

    out_path = get(ENV, "AXIOM_OPTIMIZATION_EVIDENCE_PATH", joinpath(pwd(), "build", "optimization_evidence.json"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, evidence, 2)
    end

    println("optimization evidence written: $out_path")
end

run()
