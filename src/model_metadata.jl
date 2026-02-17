# SPDX-License-Identifier: PMPL-1.0-or-later
# Model Package Metadata Schema
#
# Defines metadata structure for pretrained models including:
# - Model architecture and weights
# - Verification claims and certificates
# - Provenance and supply chain info
# - Performance benchmarks
#
# Refs: Issue #16 - Model package metadata

using Dates
using JSON
using SHA

"""
    VerificationClaim

A formal property that has been verified for the model.

# Fields
- `property::String` - Property description (e.g., "Lipschitz continuity")
- `specification::String` - Formal specification
- `verified::Bool` - Verification status
- `verifier::String` - Verification tool used
- `timestamp::DateTime` - When verified
- `certificate_path::Union{String,Nothing}` - Path to verification certificate
"""
struct VerificationClaim
    property::String
    specification::String
    verified::Bool
    verifier::String
    timestamp::DateTime
    certificate_path::Union{String,Nothing}
end

"""
    ModelMetadata

Metadata schema for packaged models with verification claims.

# Fields
- `name::String` - Model name (e.g., "ResNet50-ImageNet")
- `version::String` - Semantic version (e.g., "1.0.0")
- `architecture::String` - Architecture type (e.g., "ResNet", "Transformer")
- `task::String` - Task type (e.g., "image-classification", "text-generation")
- `framework::String` - Origin framework (e.g., "axiom", "pytorch", "tensorflow")
- `created::DateTime` - Creation timestamp
- `authors::Vector{String}` - Model authors/contributors
- `license::String` - License identifier (SPDX)
- `description::String` - Human-readable description

# Verification
- `verification_claims::Vector{VerificationClaim}` - Formal properties verified
- `verification_certificate::Union{String,Nothing}` - Path to certificate file

# Provenance
- `source::String` - Original source (URL, DOI, etc.)
- `training_data::String` - Dataset description
- `training_config::Dict{String,Any}` - Hyperparameters used
- `checksum::String` - SHA256 of weights file

# Performance
- `metrics::Dict{String,Float64}` - Performance metrics (accuracy, F1, etc.)
- `benchmarks::Dict{String,Any}` - Benchmark results

# Deployment
- `input_shape::Tuple` - Expected input shape
- `output_shape::Tuple` - Output shape
- `precision::Symbol` - Numerical precision (:float32, :float16, :mixed)
- `backend_compatibility::Vector{String}` - Compatible backends
"""
struct ModelMetadata
    # Identity
    name::String
    version::String
    architecture::String
    task::String
    framework::String
    created::DateTime
    authors::Vector{String}
    license::String
    description::String

    # Verification
    verification_claims::Vector{VerificationClaim}
    verification_certificate::Union{String,Nothing}

    # Provenance
    source::String
    training_data::String
    training_config::Dict{String,Any}
    checksum::String

    # Performance
    metrics::Dict{String,Float64}
    benchmarks::Dict{String,Any}

    # Deployment
    input_shape::Tuple
    output_shape::Tuple
    precision::Symbol
    backend_compatibility::Vector{String}
end
"""
    create_metadata(model; kwargs...) -> ModelMetadata

Create metadata for a model with sensible defaults.

# Required Arguments
- `model` - The model object
- `name::String` - Model name
- `architecture::String` - Architecture type

# Optional Arguments
- `version::String` - Version (default: "0.1.0")
- `task::String` - Task type (default: "general")
- `authors::Vector{String}` - Authors (default: empty)
- `license::String` - License (default: "PMPL-1.0-or-later")
- `description::String` - Description (default: "")
- `source::String` - Source (default: "local")
- `training_data::String` - Dataset (default: "unknown")
- `training_config::Dict` - Training config (default: empty)
- `metrics::Dict` - Performance metrics (default: empty)
- `backend_compatibility::Vector{String}` - Compatible backends (default: ["Julia"])

# Example
```julia
metadata = create_metadata(
    model,
    name="MyResNet",
    architecture="ResNet",
    task="image-classification",
    metrics=Dict("accuracy" => 0.92, "f1" => 0.91)
)
```
"""
function create_metadata(
    model;
    name::String,
    architecture::String,
    version::String = "0.1.0",
    task::String = "general",
    authors::Vector{String} = String[],
    license::String = "PMPL-1.0-or-later",
    description::String = "",
    source::String = "local",
    training_data::String = "unknown",
    training_config::Dict{String,Any} = Dict{String,Any}(),
    metrics::Dict{String,Float64} = Dict{String,Float64}(),
    backend_compatibility::Vector{String} = ["Julia"]
)
    # Compute checksum of model parameters
    params = parameters(model)
    checksum = compute_model_checksum(params)

    # Detect input/output shapes
    input_shape = input_shape_from_model(model)
    output_shape = output_shape_from_model(model)

    # Detect precision
    precision = detect_precision(params)

    ModelMetadata(
        name,
        version,
        architecture,
        task,
        "axiom",
        now(),
        authors,
        license,
        description,
        VerificationClaim[],  # Empty initially
        nothing,
        source,
        training_data,
        training_config,
        checksum,
        metrics,
        Dict{String,Any}(),  # Benchmarks empty initially
        input_shape,
        output_shape,
        precision,
        backend_compatibility
    )
end

"""
    add_verification_claim!(metadata::ModelMetadata, claim::VerificationClaim)

Add a verification claim to model metadata.
"""
function add_verification_claim!(metadata::ModelMetadata, claim::VerificationClaim)
    push!(metadata.verification_claims, claim)
end

"""
    verify_and_claim!(metadata::ModelMetadata, property::String, specification::String)

Record a verification claim in metadata.

Returns `true` only when an explicit `verified=true` marker is present in
`specification`. This function does not execute formal verification on its own.
"""
function verify_and_claim!(
    metadata::ModelMetadata,
    property::String,
    specification::String;
    verifier::String = "Axiom.jl @prove"
)
    # Conservative by default: never mark a claim verified without explicit evidence.
    verified = occursin(r"\bverified\s*=\s*true\b"i, specification)
    if !verified
        @warn "Claim recorded as unverified: provide external proof evidence in `specification` to mark verified=true" property verifier
    end

    claim = VerificationClaim(
        property,
        specification,
        verified,
        verifier,
        now(),
        nothing
    )

    add_verification_claim!(metadata, claim)

    return verified
end

"""
    save_metadata(metadata::ModelMetadata, path::String)

Save metadata to JSON file.
"""
function save_metadata(metadata::ModelMetadata, path::String)
    dict = Dict(
        "schema_version" => "1.0.0",
        "name" => metadata.name,
        "version" => metadata.version,
        "architecture" => metadata.architecture,
        "task" => metadata.task,
        "framework" => metadata.framework,
        "created" => string(metadata.created),
        "authors" => metadata.authors,
        "license" => metadata.license,
        "description" => metadata.description,
        "verification" => Dict(
            "claims" => [
                Dict(
                    "property" => claim.property,
                    "specification" => claim.specification,
                    "verified" => claim.verified,
                    "verifier" => claim.verifier,
                    "timestamp" => string(claim.timestamp),
                    "certificate_path" => claim.certificate_path
                ) for claim in metadata.verification_claims
            ],
            "certificate" => metadata.verification_certificate
        ),
        "provenance" => Dict(
            "source" => metadata.source,
            "training_data" => metadata.training_data,
            "training_config" => metadata.training_config,
            "checksum" => metadata.checksum
        ),
        "performance" => Dict(
            "metrics" => metadata.metrics,
            "benchmarks" => metadata.benchmarks
        ),
        "deployment" => Dict(
            "input_shape" => metadata.input_shape,
            "output_shape" => metadata.output_shape,
            "precision" => string(metadata.precision),
            "backend_compatibility" => metadata.backend_compatibility
        )
    )

    open(path, "w") do f
        JSON.print(f, dict, 2)
    end

    @info "Metadata saved to $path"
end

"""
    load_metadata(path::String) -> ModelMetadata

Load metadata from JSON file.
"""
function load_metadata(path::String)
    dict = JSON.parsefile(path)

    # Parse verification claims
    claims = [
        VerificationClaim(
            claim["property"],
            claim["specification"],
            claim["verified"],
            claim["verifier"],
            DateTime(claim["timestamp"]),
            get(claim, "certificate_path", nothing)
        ) for claim in dict["verification"]["claims"]
    ]

    ModelMetadata(
        dict["name"],
        dict["version"],
        dict["architecture"],
        dict["task"],
        dict["framework"],
        DateTime(dict["created"]),
        Vector{String}(dict["authors"]),
        dict["license"],
        dict["description"],
        claims,
        get(dict["verification"], "certificate", nothing),
        dict["provenance"]["source"],
        dict["provenance"]["training_data"],
        Dict{String,Any}(dict["provenance"]["training_config"]),
        dict["provenance"]["checksum"],
        Dict{String,Float64}(dict["performance"]["metrics"]),
        Dict{String,Any}(dict["performance"]["benchmarks"]),
        Tuple(dict["deployment"]["input_shape"]),
        Tuple(dict["deployment"]["output_shape"]),
        Symbol(dict["deployment"]["precision"]),
        Vector{String}(dict["deployment"]["backend_compatibility"])
    )
end

"""
    validate_metadata(metadata::ModelMetadata) -> Bool

Validate metadata completeness and consistency.
"""
function validate_metadata(metadata::ModelMetadata)
    issues = String[]

    # Required fields
    isempty(metadata.name) && push!(issues, "Name is required")
    isempty(metadata.version) && push!(issues, "Version is required")
    isempty(metadata.architecture) && push!(issues, "Architecture is required")

    # Version format (basic SemVer check)
    if !occursin(r"^\d+\.\d+\.\d+", metadata.version)
        push!(issues, "Version must be SemVer format (x.y.z)")
    end

    # License check
    if isempty(metadata.license)
        push!(issues, "License is required")
    end

    # Checksum validation
    if length(metadata.checksum) != 64  # SHA256 is 64 hex chars
        push!(issues, "Invalid checksum format (expected SHA256)")
    end

    # Report issues
    if !isempty(issues)
        @warn "Metadata validation failed" issues
        return false
    end

    @info "Metadata validation passed"
    return true
end

# ============================================================================
# Helper Functions
# ============================================================================

function _parameter_pairs(params)
    if params isa AbstractDict
        return collect(params)
    end
    collect(pairs(params))
end

function compute_model_checksum(params)
    # Concatenate all parameter arrays and hash
    hasher = SHA.SHA256_CTX()
    for (name, param) in sort(_parameter_pairs(params); by = entry -> string(entry[1]))
        if param isa AbstractArray
            SHA.update!(hasher, reinterpret(UInt8, vec(param)))
        end
    end
    bytes2hex(SHA.digest!(hasher))
end

function input_shape_from_model(model)
    # Try to infer from model structure by inspecting first layer
    if hasproperty(model, :in_features)
        return (model.in_features,)
    elseif hasproperty(model, :in_channels)
        # For Conv layers, input shape depends on image dimensions
        # Return channel count as first dimension
        return (model.in_channels,)
    elseif hasproperty(model, :layers) && !isempty(model.layers)
        # Sequential/pipeline - get input shape from first layer
        return input_shape_from_model(model.layers[1])
    end
    return ()
end

function output_shape_from_model(model)
    # Try to infer from model structure by inspecting last layer
    if hasproperty(model, :out_features)
        return (model.out_features,)
    elseif hasproperty(model, :out_channels)
        # For Conv layers, output shape depends on image dimensions
        # Return channel count
        return (model.out_channels,)
    elseif hasproperty(model, :layers) && !isempty(model.layers)
        # Sequential/pipeline - get output shape from last layer
        return output_shape_from_model(model.layers[end])
    end
    return ()
end

function detect_precision(params)
    # Check parameter types
    for (_, param) in pairs(params)
        if param isa AbstractArray
            T = eltype(param)
            if T <: Float16
                return :float16
            elseif T <: Float32
                return :float32
            elseif T <: Float64
                return :float64
            end
        end
    end
    :unknown
end
