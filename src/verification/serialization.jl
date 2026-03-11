# SPDX-License-Identifier: PMPL-1.0-or-later
# Proof Serialization
#
# Serialize and deserialize proof results for auditing, caching, and verification.

# Note: Dates, SHA, JSON are loaded in main Axiom module

# Import ProofResult from parent if prove.jl is loaded
# Otherwise define a minimal version for serialization
if !isdefined(@__MODULE__, :ProofResult)
    """
    Minimal ProofResult for serialization (used when prove.jl not loaded).
    """
    struct ProofResult
        status::Symbol
        counterexample::Union{Dict, String, Nothing}
        confidence::Float64
        details::String
        suggestions::Vector{String}
    end
end

"""
    ProofCertificate

A complete proof certificate including result, metadata, and SMT artifacts.
"""
struct ProofCertificate
    # Core proof result
    property::String
    status::Symbol  # :proven, :disproven, or :unknown
    counterexample::Union{Dict, String, Nothing}
    confidence::Float64
    details::String
    suggestions::Vector{String}

    # Metadata
    timestamp::String  # ISO 8601
    axiom_version::String
    julia_version::String
    hostname::String

    # Verification artifacts
    smt_query::Union{String, Nothing}  # SMT-LIB format query
    smt_output::Union{String, Nothing}  # Raw SMT solver output
    smt_solver::Union{String, Nothing}  # Solver name (z3, cvc5, etc.)

    # Audit trail
    proof_method::String  # "pattern", "symbolic", "smt", "unknown"
    execution_time_ms::Float64
    hash::String  # SHA256 of canonical representation
end

"""
    serialize_proof(result::ProofResult, property::String; metadata...) -> Dict

Serialize a proof result to a dictionary suitable for JSON export.

# Arguments
- `result::ProofResult`: The proof result to serialize
- `property::String`: The property expression that was proven
- `metadata...`: Optional metadata (smt_query, smt_output, smt_solver, proof_method, execution_time_ms)

# Returns
Dictionary with proof certificate data in a standardized format.

# Example
```julia
result = ProofResult(:proven, nothing, 1.0, "ReLU is non-negative", String[])
cert = serialize_proof(result, "∀x. relu(x) >= 0",
                       proof_method="pattern",
                       execution_time_ms=0.5)
```
"""
function serialize_proof(result::ProofResult, property::String;
                        smt_query::Union{String, Nothing}=nothing,
                        smt_output::Union{String, Nothing}=nothing,
                        smt_solver::Union{String, Nothing}=nothing,
                        proof_method::String="unknown",
                        execution_time_ms::Float64=0.0)

    # Generate certificate
    cert = ProofCertificate(
        property,
        result.status,
        result.counterexample,
        result.confidence,
        result.details,
        result.suggestions,
        iso8601_timestamp(),
        axiom_version_string(),
        string(VERSION),
        gethostname(),
        smt_query,
        smt_output,
        smt_solver,
        proof_method,
        execution_time_ms,
        compute_certificate_hash(property, result, smt_query, smt_output)
    )

    # Convert to dictionary
    return Dict(
        "version" => "1.0",
        "format" => "axiom-proof-certificate",
        "property" => cert.property,
        "result" => Dict(
            "status" => string(cert.status),
            "counterexample" => serialize_counterexample(cert.counterexample),
            "confidence" => cert.confidence,
            "details" => cert.details,
            "suggestions" => cert.suggestions
        ),
        "metadata" => Dict(
            "timestamp" => cert.timestamp,
            "axiom_version" => cert.axiom_version,
            "julia_version" => cert.julia_version,
            "hostname" => cert.hostname
        ),
        "artifacts" => Dict(
            "smt_query" => cert.smt_query,
            "smt_output" => cert.smt_output,
            "smt_solver" => cert.smt_solver,
            "proof_method" => cert.proof_method,
            "execution_time_ms" => cert.execution_time_ms
        ),
        "audit" => Dict(
            "hash" => cert.hash,
            "hash_algorithm" => "SHA256"
        )
    )
end

"""
    deserialize_proof(data::Dict) -> ProofCertificate

Deserialize a proof certificate from a dictionary.

# Arguments
- `data::Dict`: Dictionary containing serialized proof certificate

# Returns
`ProofCertificate` struct reconstructed from the data.

# Example
```julia
cert = deserialize_proof(JSON.parse(cert_json))
```
"""
function deserialize_proof(data::Union{Dict, AbstractDict})
    result_data = data["result"]
    metadata = data["metadata"]
    artifacts = data["artifacts"]
    audit = data["audit"]

    ProofCertificate(
        data["property"],
        Symbol(result_data["status"]),
        deserialize_counterexample(result_data["counterexample"]),
        result_data["confidence"],
        result_data["details"],
        result_data["suggestions"],
        metadata["timestamp"],
        metadata["axiom_version"],
        metadata["julia_version"],
        metadata["hostname"],
        artifacts["smt_query"],
        artifacts["smt_output"],
        artifacts["smt_solver"],
        artifacts["proof_method"],
        artifacts["execution_time_ms"],
        audit["hash"]
    )
end

"""
    export_proof_certificate(cert_dict::Dict, path::String)

Export a proof certificate to a file in JSON format.

# Arguments
- `cert_dict::Dict`: Serialized proof certificate
- `path::String`: Output file path (typically .json or .proof)

# Example
```julia
export_proof_certificate(cert, "proofs/relu_nonnegative.proof")
```
"""
function export_proof_certificate(cert_dict::Dict, path::String)
    # Ensure directory exists
    dir = dirname(path)
    !isempty(dir) && !isdir(dir) && mkpath(dir)

    # Write formatted JSON
    open(path, "w") do io
        JSON.print(io, cert_dict, 2)  # 2-space indent
    end

    @info "Proof certificate exported: $path"
end

"""
    import_proof_certificate(path::String) -> ProofCertificate

Import a proof certificate from a JSON file.

# Arguments
- `path::String`: Path to the proof certificate file

# Returns
`ProofCertificate` struct loaded from the file.

# Example
```julia
cert = import_proof_certificate("proofs/relu_nonnegative.proof")
```
"""
function import_proof_certificate(path::String)
    data = JSON.parsefile(path)
    return deserialize_proof(data)
end

"""
    verify_proof_certificate(cert::ProofCertificate) -> Bool

Verify the integrity of a proof certificate by recomputing its hash.

# Arguments
- `cert::ProofCertificate`: Certificate to verify

# Returns
`true` if the certificate hash matches recomputed hash, `false` otherwise.

# Example
```julia
cert = import_proof_certificate("proof.json")
if verify_proof_certificate(cert)
    println("Certificate valid")
else
    println("Certificate tampered or corrupted")
end
```
"""
function verify_proof_certificate(cert::ProofCertificate)
    # Reconstruct ProofResult for hash computation
    result = ProofResult(
        cert.status,
        cert.counterexample,
        cert.confidence,
        cert.details,
        cert.suggestions
    )

    recomputed_hash = compute_certificate_hash(
        cert.property,
        result,
        cert.smt_query,
        cert.smt_output
    )

    return recomputed_hash == cert.hash
end

# Helper functions

"""Serialize counterexample to JSON-compatible format."""
function serialize_counterexample(ce::Any)
    if ce === nothing
        return nothing
    elseif ce isa Dict
        return ce
    elseif ce isa NamedTuple
        return Dict(String(k) => v for (k, v) in pairs(ce))
    else
        # Fallback: convert to string
        return string(ce)
    end
end

"""Deserialize counterexample from JSON-compatible format."""
function deserialize_counterexample(ce::Any)
    if ce === nothing || ce == "nothing"
        return nothing
    elseif ce isa Dict
        return ce
    else
        return ce
    end
end

"""Get current timestamp in ISO 8601 format."""
function iso8601_timestamp()
    return Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ")
end

"""Get Axiom.jl version string."""
function axiom_version_string()
    # Try to get from Project.toml
    project_path = joinpath(@__DIR__, "..", "..", "Project.toml")
    if isfile(project_path)
        for line in eachline(project_path)
            m = match(r"^version\s*=\s*\"(.+)\"", line)
            if m !== nothing
                return m.captures[1]
            end
        end
    end
    return "unknown"
end

"""Compute SHA256 hash of certificate for integrity verification."""
function compute_certificate_hash(property::String, result::ProofResult,
                                   smt_query::Union{String, Nothing},
                                   smt_output::Union{String, Nothing})
    # Canonical representation for hashing
    canonical = string(
        "property:", property, "\n",
        "status:", result.status, "\n",
        "counterexample:", repr(result.counterexample), "\n",
        "confidence:", result.confidence, "\n",
        "details:", result.details, "\n",
        "smt_query:", something(smt_query, ""), "\n",
        "smt_output:", something(smt_output, ""), "\n"
    )

    return bytes2hex(sha256(canonical))
end

# Module exports
export ProofCertificate
export serialize_proof, deserialize_proof
export export_proof_certificate, import_proof_certificate
export verify_proof_certificate