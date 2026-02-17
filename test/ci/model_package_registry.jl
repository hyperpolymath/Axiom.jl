# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using JSON
using Axiom

@testset "Model package + registry workflow" begin
    model = Sequential(
        Dense(8, 4, relu),
        Dense(4, 2),
        Softmax(),
    )

    metadata = create_metadata(
        model;
        name = "ci-package-model",
        architecture = "Sequential",
        version = "0.2.0",
        task = "classification",
        authors = ["Axiom CI"],
        description = "Packaging workflow coverage",
        backend_compatibility = ["JuliaBackend", "RustBackend"],
    )

    @test verify_and_claim!(metadata, "FiniteOutput", "verified=true; source=ci") == true

    tmp = mktempdir()
    cert_path = joinpath(tmp, "ci.cert")
    write(cert_path, "dummy certificate for packaging test\n")

    bundle = export_model_package(
        model,
        metadata,
        tmp;
        base_name = "ci_bundle",
        certificate_path = cert_path,
        tags = ["ci", "could"],
        extra = Dict{String, Any}("pipeline" => "model-package-registry"),
    )

    @test isfile(bundle["manifest"])
    @test isfile(bundle["model"])
    @test isfile(bundle["metadata"])
    @test isfile(bundle["certificate"])

    manifest = JSON.parsefile(bundle["manifest"])
    @test manifest["format"] == MODEL_PACKAGE_FORMAT
    @test manifest["identity"]["name"] == "ci-package-model"
    @test manifest["verification"]["claim_count"] >= 1
    @test manifest["verification"]["verified_claim_count"] >= 1
    @test length(manifest["artifacts"]["model"]["sha256"]) == 64
    @test length(manifest["artifacts"]["metadata"]["sha256"]) == 64
    @test length(manifest["reproducibility"]["manifest_sha256"]) == 64

    registry_entry = build_registry_entry(
        bundle["manifest"];
        channel = "candidate",
        tags = ["ci", "registry"],
        uri = "https://example.invalid/ci_bundle.package.json",
    )
    @test registry_entry["format"] == MODEL_REGISTRY_ENTRY_FORMAT
    @test registry_entry["channel"] == "candidate"
    @test startswith(registry_entry["index_key"], "ci-package-model@")
    @test registry_entry["package"]["format"] == MODEL_PACKAGE_FORMAT

    entry_path = joinpath(tmp, "registry-entry.json")
    export_registry_entry(registry_entry, entry_path)
    @test isfile(entry_path)

    loaded_entry = JSON.parsefile(entry_path)
    @test loaded_entry["identity"]["version"] == "0.2.0"
    @test loaded_entry["package"]["manifest_file_sha256"] == registry_entry["package"]["manifest_file_sha256"]
end
