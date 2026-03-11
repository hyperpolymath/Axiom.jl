# SPDX-License-Identifier: PMPL-1.0-or-later

using Axiom
using JSON

function run()
    reset_verification_telemetry!()

    model = Sequential(
        Dense(10, 6, relu),
        Dense(6, 3),
        Softmax(),
    )

    x = Tensor(randn(Float32, 4, 10))
    data = [(x, nothing)]

    result = verify(
        model;
        properties = [FiniteOutput(), Axiom.BoundedOutput(0.0f0, 0.1f0)],
        data = data,
        telemetry_source = "verification-telemetry-evidence",
    )

    payload = verification_result_telemetry(
        result;
        mode = "STANDARD",
        source = "verification-telemetry-evidence",
        tags = Dict{String, Any}("artifact" => "verification_telemetry"),
    )

    summary = verification_telemetry_report()

    evidence = Dict{String, Any}(
        "format" => "axiom-verification-telemetry-evidence.v1",
        "run" => payload,
        "summary" => summary,
    )

    out_path = get(ENV, "AXIOM_VERIFICATION_TELEMETRY_PATH", joinpath(pwd(), "build", "verification_telemetry_evidence.json"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, evidence, 2)
    end

    println("verification telemetry evidence written: $out_path")
end

run()
