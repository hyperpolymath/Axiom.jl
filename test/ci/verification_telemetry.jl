# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using Axiom

@testset "Verification telemetry reporting" begin
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
        telemetry_source = "ci-verification-telemetry",
    )

    @test !result.passed

    run_payload = verification_result_telemetry(
        result;
        mode = "STANDARD",
        source = "ci-verification-telemetry",
        tags = Dict{String, Any}("suite" => "verification_telemetry"),
    )
    @test run_payload["format"] == "axiom-verification-telemetry.v1"
    @test run_payload["properties_total"] == 2
    @test run_payload["properties_failed"] == 1
    @test run_payload["counterexamples_count"] >= 1

    summary = verification_telemetry_report()
    @test summary["format"] == "axiom-verification-telemetry-summary.v1"
    @test summary["runs"] == 1
    @test summary["failed"] == 1
    @test summary["property_failures"] == 1
    @test haskey(summary["by_property"], "FiniteOutput")
    @test haskey(summary["by_property"], "BoundedOutput")
    @test summary["last_run"]["source"] == "ci-verification-telemetry"
end
