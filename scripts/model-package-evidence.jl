# SPDX-License-Identifier: PMPL-1.0-or-later

using Axiom
using JSON

function run()
    model = Sequential(
        Dense(8, 4, relu),
        Dense(4, 2),
        Softmax(),
    )

    metadata = create_metadata(
        model;
        name = "axiom-evidence-model",
        architecture = "Sequential",
        version = "0.1.0",
        task = "classification",
        authors = ["Axiom.jl"],
        description = "Packaging evidence artifact",
        backend_compatibility = ["JuliaBackend", "RustBackend"],
    )

    verify_and_claim!(metadata, "FiniteOutput", "verified=true; source=model-package-evidence")

    out_dir = get(ENV, "AXIOM_MODEL_PACKAGE_DIR", joinpath(pwd(), "build", "model_package"))
    mkpath(out_dir)
    cert_path = joinpath(out_dir, "evidence.cert")
    write(cert_path, "model package evidence certificate placeholder\n")

    package_paths = export_model_package(
        model,
        metadata,
        out_dir;
        base_name = "axiom_evidence_model",
        certificate_path = cert_path,
        tags = ["could", "packaging", "registry"],
        extra = Dict{String, Any}("evidence" => true),
    )

    registry_path = joinpath(out_dir, "axiom_evidence_model.registry.json")
    export_registry_entry(
        package_paths["manifest"],
        registry_path;
        channel = "evidence",
        tags = ["could", "registry"],
    )

    evidence = Dict{String, Any}(
        "format" => "axiom-model-package-evidence.v1",
        "manifest" => package_paths["manifest"],
        "model" => package_paths["model"],
        "metadata" => package_paths["metadata"],
        "certificate" => get(package_paths, "certificate", nothing),
        "registry_entry" => registry_path,
    )

    out_path = get(ENV, "AXIOM_MODEL_PACKAGE_EVIDENCE_PATH", joinpath(pwd(), "build", "model_package_evidence.json"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, evidence, 2)
    end

    println("model package evidence written: $out_path")
end

run()
