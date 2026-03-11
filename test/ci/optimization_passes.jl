# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Axiom

function avg_latency_ms(model, x; iterations::Int = 40)
    model(x)  # warmup
    elapsed = @elapsed begin
        for _ in 1:iterations
            model(x)
        end
    end
    (elapsed / iterations) * 1000
end

@testset "Optimization flags + mixed precision behavior" begin
    model = Sequential(
        Dense(12, 16),
        ReLU(),
        Dense(16, 8),
        ReLU(),
        Dense(8, 4),
        Softmax(),
    )

    x = Tensor(randn(Float32, 32, 12))

    compiled_none = compile(model; backend = JuliaBackend(), optimize = :none, precision = :float32, verify = false)
    compiled_default = compile(model; backend = JuliaBackend(), optimize = :default, precision = :float32, verify = false)
    compiled_aggressive = compile(model; backend = JuliaBackend(), optimize = :aggressive, precision = :float32, verify = false)
    compiled_mixed = compile(model; backend = JuliaBackend(), optimize = :default, precision = :mixed, verify = false)

    @test compiled_none isa Axiom.Pipeline
    @test compiled_default isa Axiom.Pipeline
    @test compiled_aggressive isa Axiom.Pipeline
    @test compiled_mixed isa Axiom.MixedPrecisionWrapper

    @test length(compiled_default.layers) <= length(model.layers)
    @test length(compiled_aggressive.layers) <= length(model.layers)

    y_none = compiled_none(x).data
    y_default = compiled_default(x).data
    y_aggressive = compiled_aggressive(x).data
    y_mixed = compiled_mixed(x).data

    @test size(y_none) == size(y_default) == size(y_aggressive) == size(y_mixed)
    @test all(isfinite, y_none)
    @test all(isfinite, y_mixed)

    drift_default = maximum(abs.(y_none .- y_default))
    drift_aggressive = maximum(abs.(y_none .- y_aggressive))
    drift_mixed = maximum(abs.(y_none .- y_mixed))

    @test drift_default <= 1f-4
    @test drift_aggressive <= 1f-4
    @test drift_mixed <= 5f-2

    lat_none = avg_latency_ms(compiled_none, x)
    lat_default = avg_latency_ms(compiled_default, x)
    lat_aggressive = avg_latency_ms(compiled_aggressive, x)
    lat_mixed = avg_latency_ms(compiled_mixed, x)

    @test lat_none > 0
    @test lat_default > 0
    @test lat_aggressive > 0
    @test lat_mixed > 0
end
