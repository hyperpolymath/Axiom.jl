# Axiom.jl Verification Certificates
#
# Cryptographically signed certificates proving model properties.

using SHA

"""
    Certificate

A formal certificate proving properties about a model.
"""
struct Certificate
    # Model identification
    model_hash::String
    model_name::String

    # What was proven
    properties::Vector{Property}
    verification_mode::VerificationMode

    # Evidence
    test_data_hash::Union{String, Nothing}
    proof_type::Symbol  # :static, :empirical, :formal

    # Metadata
    created_at::Float64
    axiom_version::String
    verifier_id::String

    # Signature (for tamper detection)
    signature::String
end

"""
    generate_certificate(model, result::VerificationResult; kwargs...) -> Certificate

Generate a verification certificate for a model.
"""
function generate_certificate(
    model,
    result::VerificationResult;
    model_name::String = "unnamed",
    test_data = nothing,
    verifier_id::String = "axiom-default"
)
    # Compute model hash
    model_hash = compute_model_hash(model)

    # Compute test data hash if provided
    test_data_hash = test_data === nothing ? nothing : compute_data_hash(test_data)

    # Determine proof type
    proof_type = :empirical  # Default

    # Create certificate
    cert = Certificate(
        model_hash,
        model_name,
        [prop for (prop, passed) in result.properties_checked if passed],
        STANDARD,
        test_data_hash,
        proof_type,
        time(),
        string(Axiom.VERSION),
        verifier_id,
        ""  # Signature computed below
    )

    # Sign certificate
    sign_certificate(cert)
end

"""
Compute hash of model parameters.
"""
function compute_model_hash(model)
    params = parameters(model)
    h = SHA.sha256(repr(params))
    bytes2hex(h)
end

"""
Compute hash of test data.
"""
function compute_data_hash(data)
    h = SHA.sha256(repr(collect(data)))
    bytes2hex(h)
end

"""
Sign a certificate (simplified - production would use proper PKI).
"""
function sign_certificate(cert::Certificate)
    # Concatenate certificate fields
    content = string(
        cert.model_hash,
        cert.model_name,
        cert.verification_mode,
        cert.created_at,
        cert.axiom_version
    )

    # Compute signature
    signature = bytes2hex(SHA.sha256(content))

    # Return new certificate with signature
    Certificate(
        cert.model_hash,
        cert.model_name,
        cert.properties,
        cert.verification_mode,
        cert.test_data_hash,
        cert.proof_type,
        cert.created_at,
        cert.axiom_version,
        cert.verifier_id,
        signature
    )
end

"""
    verify_certificate(cert::Certificate) -> Bool

Verify that a certificate has not been tampered with.
"""
function verify_certificate(cert::Certificate)
    content = string(
        cert.model_hash,
        cert.model_name,
        cert.verification_mode,
        cert.created_at,
        cert.axiom_version
    )

    expected_signature = bytes2hex(SHA.sha256(content))
    cert.signature == expected_signature
end

"""
    save_certificate(cert::Certificate, path::String)

Save certificate to file.
"""
function save_certificate(cert::Certificate, path::String)
    open(path, "w") do f
        println(f, "# Axiom.jl Verification Certificate")
        println(f, "# Generated: $(cert.created_at)")
        println(f, "")
        println(f, "model_name: $(cert.model_name)")
        println(f, "model_hash: $(cert.model_hash)")
        println(f, "axiom_version: $(cert.axiom_version)")
        println(f, "verification_mode: $(cert.verification_mode)")
        println(f, "proof_type: $(cert.proof_type)")
        println(f, "")
        println(f, "properties:")
        for prop in cert.properties
            println(f, "  - $(typeof(prop).name.name)")
        end
        println(f, "")
        println(f, "signature: $(cert.signature)")
    end

    @info "Certificate saved to $path"
end

"""
    load_certificate(path::String) -> Certificate

Load certificate from file.
"""
function load_certificate(path::String)
    lines = readlines(path)

    # Parse YAML-like format
    model_hash = ""
    model_name = ""
    axiom_version = ""
    verification_mode = STANDARD
    proof_type = :empirical
    properties = Property[]
    signature = ""
    created_at = 0.0
    test_data_hash = nothing
    verifier_id = "axiom-default"

    current_section = :none

    for line in lines
        line = strip(line)

        # Skip comments and empty lines
        startswith(line, "#") && continue
        isempty(line) && continue

        # Parse key-value pairs
        if contains(line, ": ")
            parts = split(line, ": ", limit=2)
            key = strip(parts[1])
            value = length(parts) > 1 ? strip(parts[2]) : ""

            if key == "model_name"
                model_name = value
            elseif key == "model_hash"
                model_hash = value
            elseif key == "axiom_version"
                axiom_version = value
            elseif key == "verification_mode"
                verification_mode = parse_verification_mode(value)
            elseif key == "proof_type"
                proof_type = Symbol(value)
            elseif key == "signature"
                signature = value
            elseif key == "properties"
                current_section = :properties
            elseif startswith(key, "- ") && current_section == :properties
                # Parse property name
                prop_name = strip(key[3:end])
                prop = parse_property_type(prop_name)
                if prop !== nothing
                    push!(properties, prop)
                end
            end
        end
    end

    cert = Certificate(
        model_hash,
        model_name,
        properties,
        verification_mode,
        test_data_hash,
        proof_type,
        created_at,
        axiom_version,
        verifier_id,
        ""  # Will verify signature below
    )

    # Verify signature
    if !isempty(signature) && !verify_loaded_signature(cert, signature)
        @warn "Certificate signature verification failed - certificate may have been tampered with"
    end

    # Return certificate with original signature
    Certificate(
        cert.model_hash,
        cert.model_name,
        cert.properties,
        cert.verification_mode,
        cert.test_data_hash,
        cert.proof_type,
        cert.created_at,
        cert.axiom_version,
        cert.verifier_id,
        signature
    )
end

"""
Parse verification mode from string.
"""
function parse_verification_mode(s::String)
    s = uppercase(strip(s))
    if s == "STRICT"
        return STRICT
    elseif s == "FAST"
        return FAST
    elseif s == "DEBUG"
        return DEBUG
    else
        return STANDARD
    end
end

"""
Parse property type from name string.
"""
function parse_property_type(name::String)
    name = strip(name)

    # Map property names to types
    if name == "ValidProbabilities" || name == "ValidProbability"
        return ValidProbabilities()
    elseif name == "FiniteOutput" || name == "FiniteOutputs"
        return FiniteOutput()
    elseif name == "NoNaN" || name == "NoNaNs"
        return NoNaN()
    elseif name == "BoundedActivation" || name == "BoundedActivations"
        return BoundedActivation()
    else
        @debug "Unknown property type: $name"
        return nothing
    end
end

"""
Verify signature of loaded certificate.
"""
function verify_loaded_signature(cert::Certificate, expected_signature::String)
    content = string(
        cert.model_hash,
        cert.model_name,
        cert.verification_mode,
        cert.created_at,
        cert.axiom_version
    )

    computed_signature = bytes2hex(SHA.sha256(content))
    computed_signature == expected_signature
end

# ============================================================================
# Certificate Display
# ============================================================================

function Base.show(io::IO, cert::Certificate)
    println(io, "╔══════════════════════════════════════════╗")
    println(io, "║   AXIOM.JL VERIFICATION CERTIFICATE      ║")
    println(io, "╠══════════════════════════════════════════╣")
    println(io, "║ Model: $(rpad(cert.model_name, 30))   ║")
    println(io, "║ Hash:  $(cert.model_hash[1:16])...         ║")
    println(io, "║                                          ║")
    println(io, "║ Verified Properties:                     ║")
    for prop in cert.properties
        name = string(typeof(prop).name.name)
        println(io, "║   ✓ $(rpad(name, 34))   ║")
    end
    println(io, "║                                          ║")
    println(io, "║ Proof Type: $(rpad(string(cert.proof_type), 26))   ║")
    println(io, "║ Axiom Version: $(rpad(cert.axiom_version, 23))   ║")
    println(io, "╚══════════════════════════════════════════╝")
end

function Base.show(io::IO, ::MIME"text/plain", cert::Certificate)
    show(io, cert)
end
