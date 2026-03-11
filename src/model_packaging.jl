# SPDX-License-Identifier: PMPL-1.0-or-later
# Model package + registry manifests
#
# Baseline packaging workflow for reusable model artifacts with deterministic
# hashes and machine-readable registry entries.

const MODEL_PACKAGE_FORMAT = "axiom-model-package.v1"
const MODEL_REGISTRY_ENTRY_FORMAT = "axiom-model-registry-entry.v1"

function _canonical_repr(value)
    if value isa AbstractDict
        keys_sorted = sort(collect(keys(value)); by = key -> String(key))
        parts = String[]
        for key in keys_sorted
            push!(parts, string(String(key), "=", _canonical_repr(value[key])))
        end
        return "{" * join(parts, ",") * "}"
    elseif value isa AbstractVector
        return "[" * join((_canonical_repr(item) for item in value), ",") * "]"
    elseif value isa Tuple
        return "[" * join((_canonical_repr(item) for item in value), ",") * "]"
    elseif value isa DateTime
        return Dates.format(value, "yyyy-mm-ddTHH:MM:SS.sss")
    elseif value === nothing
        return "nothing"
    end
    repr(value)
end

_sha256_text(text::AbstractString) = bytes2hex(SHA.sha256(String(text)))

function _sha256_file(path::String)
    ctx = SHA.SHA256_CTX()
    open(path, "r") do io
        while !eof(io)
            chunk = read(io, 1 << 16)
            isempty(chunk) || SHA.update!(ctx, chunk)
        end
    end
    bytes2hex(SHA.digest!(ctx))
end

function _artifact_entry(path::String, root::String)
    Dict(
        "path" => relpath(path, root),
        "sha256" => _sha256_file(path),
        "bytes" => filesize(path),
    )
end

function _claim_dict(claim::VerificationClaim)
    Dict(
        "property" => claim.property,
        "specification" => claim.specification,
        "verified" => claim.verified,
        "verifier" => claim.verifier,
        "timestamp" => Dates.format(claim.timestamp, "yyyy-mm-ddTHH:MM:SS.sss"),
        "certificate_path" => claim.certificate_path,
    )
end

"""
    model_package_manifest(model, metadata::ModelMetadata, output_dir::String;
                           base_name="", certificate_path=nothing, tags=String[],
                           extra=Dict{String,Any}()) -> Dict{String,Any}

Build a machine-readable package manifest for a model artifact bundle.
Use `export_model_package` to also materialize files on disk.
"""
function model_package_manifest(
    model,
    metadata::ModelMetadata,
    output_dir::String;
    base_name::String = "",
    certificate_path::Union{Nothing, String} = nothing,
    tags::Vector{String} = String[],
    extra::Dict{String, Any} = Dict{String, Any}(),
)
    mkpath(output_dir)

    raw_base = isempty(base_name) ? string(metadata.name, "-", metadata.version) : base_name
    safe_base = replace(raw_base, r"[^A-Za-z0-9_.-]+" => "_")

    model_path = joinpath(output_dir, "$safe_base.axiom")
    metadata_path = joinpath(output_dir, "$safe_base.metadata.json")

    save_model(model, model_path)
    save_metadata(metadata, metadata_path)

    certificate_entry = nothing
    if certificate_path !== nothing
        isfile(certificate_path) || throw(ArgumentError("certificate_path not found: $certificate_path"))
        cert_ext = splitext(certificate_path)[2]
        cert_name = isempty(cert_ext) ? "$safe_base.cert" : "$safe_base$cert_ext"
        cert_path = joinpath(output_dir, cert_name)
        cp(certificate_path, cert_path; force = true)
        certificate_entry = _artifact_entry(cert_path, output_dir)
    end

    claims = [_claim_dict(claim) for claim in metadata.verification_claims]
    verified_count = count(claim -> get(claim, "verified", false), claims)

    artifacts = Dict(
        "model" => _artifact_entry(model_path, output_dir),
        "metadata" => _artifact_entry(metadata_path, output_dir),
    )
    if certificate_entry !== nothing
        artifacts["certificate"] = certificate_entry
    end

    manifest = Dict{String, Any}(
        "format" => MODEL_PACKAGE_FORMAT,
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "identity" => Dict(
            "name" => metadata.name,
            "version" => metadata.version,
            "architecture" => metadata.architecture,
            "task" => metadata.task,
        ),
        "artifacts" => artifacts,
        "verification" => Dict(
            "claim_count" => length(claims),
            "verified_claim_count" => verified_count,
            "claims" => claims,
        ),
        "provenance" => Dict(
            "framework" => metadata.framework,
            "source" => metadata.source,
            "training_data" => metadata.training_data,
            "training_config" => metadata.training_config,
            "metadata_checksum" => metadata.checksum,
        ),
        "deployment" => Dict(
            "precision" => string(metadata.precision),
            "input_shape" => collect(metadata.input_shape),
            "output_shape" => collect(metadata.output_shape),
            "backend_compatibility" => metadata.backend_compatibility,
        ),
        "tags" => collect(tags),
        "extra" => Dict(extra),
    )

    manifest["reproducibility"] = Dict(
        "manifest_sha256" => _sha256_text(_canonical_repr(manifest)),
        "metadata_checksum" => metadata.checksum,
    )

    manifest
end

"""
    export_model_package(model, metadata::ModelMetadata, output_dir::String;
                         base_name="", certificate_path=nothing, tags=String[],
                         extra=Dict{String,Any}()) -> Dict{String,String}

Materialize a model package bundle (`.axiom`, metadata JSON, package manifest).
Returns file paths for generated artifacts.
"""
function export_model_package(
    model,
    metadata::ModelMetadata,
    output_dir::String;
    base_name::String = "",
    certificate_path::Union{Nothing, String} = nothing,
    tags::Vector{String} = String[],
    extra::Dict{String, Any} = Dict{String, Any}(),
)
    manifest = model_package_manifest(
        model,
        metadata,
        output_dir;
        base_name = base_name,
        certificate_path = certificate_path,
        tags = tags,
        extra = extra,
    )

    identity = manifest["identity"]
    raw_base = isempty(base_name) ? string(identity["name"], "-", identity["version"]) : base_name
    safe_base = replace(raw_base, r"[^A-Za-z0-9_.-]+" => "_")
    manifest_path = joinpath(output_dir, "$safe_base.package.json")

    open(manifest_path, "w") do io
        JSON.print(io, manifest, 2)
    end

    paths = Dict{String, String}(
        "manifest" => manifest_path,
        "model" => joinpath(output_dir, manifest["artifacts"]["model"]["path"]),
        "metadata" => joinpath(output_dir, manifest["artifacts"]["metadata"]["path"]),
    )
    if haskey(manifest["artifacts"], "certificate")
        paths["certificate"] = joinpath(output_dir, manifest["artifacts"]["certificate"]["path"])
    end

    paths
end

"""
    load_model_package_manifest(path::String) -> Dict{String,Any}

Read a package manifest from disk.
"""
load_model_package_manifest(path::String) = Dict{String, Any}(JSON.parsefile(path))

function _registry_entry_dict(
    manifest::Dict{String, Any};
    manifest_path::Union{Nothing, String} = nothing,
    channel::String = "stable",
    tags::Vector{String} = String[],
    uri::Union{Nothing, String} = nothing,
)
    identity = Dict{String, Any}(manifest["identity"])
    package_ref = Dict{String, Any}(
        "format" => manifest["format"],
        "manifest_sha256" => manifest["reproducibility"]["manifest_sha256"],
    )
    if manifest_path !== nothing
        package_ref["manifest_path"] = manifest_path
        package_ref["manifest_file_sha256"] = _sha256_file(manifest_path)
    end
    if uri !== nothing
        package_ref["uri"] = uri
    end

    Dict{String, Any}(
        "format" => MODEL_REGISTRY_ENTRY_FORMAT,
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "channel" => channel,
        "index_key" => string(identity["name"], "@", identity["version"]),
        "identity" => identity,
        "package" => package_ref,
        "verification" => Dict(
            "claim_count" => manifest["verification"]["claim_count"],
            "verified_claim_count" => manifest["verification"]["verified_claim_count"],
        ),
        "tags" => collect(tags),
    )
end

"""
    build_registry_entry(manifest::Dict{String,Any}; channel="stable", tags=String[], uri=nothing)
    build_registry_entry(manifest_path::String; channel="stable", tags=String[], uri=nothing)

Build a machine-readable registry entry for a packaged model artifact.
"""
function build_registry_entry(
    manifest::Dict{String, Any};
    channel::String = "stable",
    tags::Vector{String} = String[],
    uri::Union{Nothing, String} = nothing,
)
    _registry_entry_dict(manifest; manifest_path = nothing, channel = channel, tags = tags, uri = uri)
end

function build_registry_entry(
    manifest_path::String;
    channel::String = "stable",
    tags::Vector{String} = String[],
    uri::Union{Nothing, String} = nothing,
)
    manifest = load_model_package_manifest(manifest_path)
    _registry_entry_dict(
        manifest;
        manifest_path = manifest_path,
        channel = channel,
        tags = tags,
        uri = uri,
    )
end

"""
    export_registry_entry(entry::Dict{String,Any}, output_path::String) -> String
    export_registry_entry(manifest_path::String, output_path::String; channel="stable", tags=String[], uri=nothing) -> String

Write a registry entry JSON artifact.
"""
function export_registry_entry(entry::Dict{String, Any}, output_path::String)
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON.print(io, entry, 2)
    end
    output_path
end

function export_registry_entry(
    manifest_path::String,
    output_path::String;
    channel::String = "stable",
    tags::Vector{String} = String[],
    uri::Union{Nothing, String} = nothing,
)
    entry = build_registry_entry(manifest_path; channel = channel, tags = tags, uri = uri)
    export_registry_entry(entry, output_path)
end
