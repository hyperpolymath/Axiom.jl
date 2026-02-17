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

const COPROCESSOR_FAILURE_MODE = Dict{DataType, Bool}(
    Axiom.TPUBackend => false,
    Axiom.NPUBackend => false,
    Axiom.DSPBackend => false,
    Axiom.PPUBackend => false,
    Axiom.MathBackend => false,
    Axiom.FPGABackend => false,
)

function backend_key(backend)
    if backend isa TPUBackend
        return "tpu"
    elseif backend isa NPUBackend
        return "npu"
    elseif backend isa DSPBackend
        return "dsp"
    elseif backend isa PPUBackend
        return "ppu"
    elseif backend isa MathBackend
        return "math"
    elseif backend isa FPGABackend
        return "fpga"
    end
    error("Unexpected backend type: $(typeof(backend))")
end

function Axiom.backend_coprocessor_matmul(
    backend::Union{Axiom.TPUBackend, Axiom.NPUBackend, Axiom.DSPBackend, Axiom.PPUBackend, Axiom.MathBackend, Axiom.FPGABackend},
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    if get(COPROCESSOR_FAILURE_MODE, typeof(backend), false)
        error("Injected coprocessor matmul failure for $(typeof(backend))")
    end
    A * B
end

function Axiom.backend_coprocessor_relu(
    backend::Union{Axiom.TPUBackend, Axiom.NPUBackend, Axiom.DSPBackend, Axiom.PPUBackend, Axiom.MathBackend, Axiom.FPGABackend},
    x::AbstractArray{Float32},
)
    max.(x, 0f0)
end

function Axiom.backend_coprocessor_softmax(
    backend::Union{Axiom.TPUBackend, Axiom.NPUBackend, Axiom.DSPBackend, Axiom.PPUBackend, Axiom.MathBackend, Axiom.FPGABackend},
    x::AbstractArray{Float32},
    dim::Int,
)
    Axiom.softmax(x, dims=dim)
end

function reset_coprocessor_failure_mode!()
    for backend_type in keys(COPROCESSOR_FAILURE_MODE)
        COPROCESSOR_FAILURE_MODE[backend_type] = false
    end
end

function recovery_probe(model, x, cpu, backend)
    reset_coprocessor_runtime_diagnostics!()
    reset_coprocessor_failure_mode!()
    COPROCESSOR_FAILURE_MODE[typeof(backend)] = true

    compile_ms = @elapsed compiled = compile(model, backend=backend, verify=false, optimize=:none)
    y = compiled(x).data
    key = backend_key(backend)
    diag = coprocessor_runtime_diagnostics()["backends"][key]

    Dict(
        "backend" => key,
        "compiled_wrapper" => compiled isa Axiom.CoprocessorCompiledModel,
        "compile_ms" => round(compile_ms * 1000; digits=3),
        "parity_ok" => isapprox(y, cpu; atol=1f-5, rtol=1f-5),
        "finite_ok" => all(isfinite, y),
        "diagnostics" => diag,
    )
end

function disabled_self_heal_probe(model, x)
    reset_coprocessor_runtime_diagnostics!()
    reset_coprocessor_failure_mode!()
    COPROCESSOR_FAILURE_MODE[Axiom.TPUBackend] = true

    compiled = compile(model, backend=TPUBackend(0), verify=false, optimize=:none)

    err_message = nothing
    try
        compiled(x)
    catch err
        err_message = sprint(showerror, err)
    end

    Dict(
        "backend" => "tpu",
        "compiled_wrapper" => compiled isa Axiom.CoprocessorCompiledModel,
        "raised_error" => err_message !== nothing,
        "error_message" => err_message,
        "diagnostics" => coprocessor_runtime_diagnostics()["backends"]["tpu"],
    )
end

function main()
    model = Sequential(
        Dense(6, 4, relu),
        Dense(4, 3),
        Softmax(),
    )
    x = Tensor(randn(Float32, 5, 6))
    cpu = model(x).data

    failures = String[]
    recovery_results = Any[]

    with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_NPU_AVAILABLE" => "1",
        "AXIOM_DSP_AVAILABLE" => "1",
        "AXIOM_PPU_AVAILABLE" => "1",
        "AXIOM_MATH_AVAILABLE" => "1",
        "AXIOM_FPGA_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_NPU_DEVICE_COUNT" => "1",
        "AXIOM_DSP_DEVICE_COUNT" => "1",
        "AXIOM_PPU_DEVICE_COUNT" => "1",
        "AXIOM_MATH_DEVICE_COUNT" => "1",
        "AXIOM_FPGA_DEVICE_COUNT" => "1",
        "AXIOM_COPROCESSOR_SELF_HEAL" => "1",
    )) do
        for backend in (TPUBackend(0), NPUBackend(0), DSPBackend(0), PPUBackend(0), MathBackend(0), FPGABackend(0))
            result = recovery_probe(model, x, cpu, backend)
            push!(recovery_results, result)

            result["parity_ok"] || push!(failures, "$(result["backend"]) parity mismatch during self-healing recovery")
            result["finite_ok"] || push!(failures, "$(result["backend"]) produced non-finite values during self-healing recovery")

            diag = result["diagnostics"]
            Int(diag["runtime_errors"]) >= 1 || push!(failures, "$(result["backend"]) did not record runtime_errors")
            Int(diag["runtime_fallbacks"]) >= 1 || push!(failures, "$(result["backend"]) did not record runtime_fallbacks")
            Int(diag["recoveries"]) >= 1 || push!(failures, "$(result["backend"]) did not record recoveries")
        end
    end

    disabled_probe = with_env(Dict(
        "AXIOM_TPU_AVAILABLE" => "1",
        "AXIOM_TPU_DEVICE_COUNT" => "1",
        "AXIOM_COPROCESSOR_SELF_HEAL" => "0",
    )) do
        disabled_self_heal_probe(model, x)
    end

    disabled_probe["raised_error"] || push!(failures, "self-healing disabled path did not raise an error")
    if disabled_probe["error_message"] !== nothing
        occursin("AXIOM_COPROCESSOR_SELF_HEAL=0", disabled_probe["error_message"]) ||
            push!(failures, "disabled self-healing error message missing AXIOM_COPROCESSOR_SELF_HEAL hint")
    end

    disabled_diag = disabled_probe["diagnostics"]
    Int(disabled_diag["runtime_errors"]) >= 1 || push!(failures, "disabled self-healing path did not record runtime_errors")
    Int(disabled_diag["runtime_fallbacks"]) == 0 || push!(failures, "disabled self-healing path unexpectedly recorded runtime_fallbacks")
    Int(disabled_diag["recoveries"]) == 0 || push!(failures, "disabled self-healing path unexpectedly recorded recoveries")

    payload = Dict(
        "format" => "axiom-coprocessor-resilience-evidence.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "capability_report" => coprocessor_capability_report(),
        "recovery_results" => recovery_results,
        "disabled_self_heal_probe" => disabled_probe,
    )

    out_path = get(ENV, "AXIOM_COPROCESSOR_RESILIENCE_EVIDENCE_PATH", joinpath(pwd(), "build", "coprocessor_resilience_evidence.json"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, payload, 2)
    end

    println("coprocessor resilience evidence written: $out_path")

    if !isempty(failures)
        println("coprocessor resilience evidence checks failed:")
        for failure in failures
            println(" - $failure")
        end
        exit(1)
    end
end

main()
