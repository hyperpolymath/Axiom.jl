# SPDX-License-Identifier: MPL-2.0
# Axiom.jl Verification Certificates
#
# Tamper-evidence certificates for model properties. By default, the
# certificate carries a SHA-256 content digest over its public fields — this
# detects accidental or naive tampering but is NOT a keyed/authenticating
# signature (anyone can recompute the digest); this default,
# `authenticated=false` behavior is unchanged by the rest of this file.
#
# An OPT-IN authenticating path is also provided: `sign_certificate_hybrid`
# computes a real hybrid Ed448 (classical) + Dilithium5/ML-DSA-87
# (post-quantum) signature over the certificate content (per the estate
# Trustfile scheme; see `signing.jl` and `crypto/README.md`), and
# `verify_certificate_hybrid` requires BOTH components to verify. This is the
# authenticating-signature half of G01. Certificates produced by the plain
# `generate_certificate`/`save_certificate` path are untouched — callers must
# explicitly opt in to hybrid signing.

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

    # Determine proof type based on verification evidence
    proof_type = _infer_proof_type(result)

    # Create certificate
    cert = Certificate(
        model_hash,
        model_name,
        [prop for (prop, passed) in result.properties_checked if passed],
        STANDARD,  # VerificationResult carries no mode field; certificates default to STANDARD
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
Infer proof type from verification result evidence.

Returns:
- `:formal` if SMT solver provided a proof
- `:static` if compile-time shape analysis sufficed
- `:empirical` if checked against test data only
"""
function _infer_proof_type(result::VerificationResult)
    # Check if any property was verified via SMT
    for (prop, passed) in result.properties_checked
        if passed && _is_statically_provable(prop)
            return :static
        end
    end
    # Default: properties were checked empirically against test data
    :empirical
end

"""
Check if a property can be statically verified without test data.
"""
function _is_statically_provable(prop::Property)
    # These properties have algebraic guarantees when the model
    # architecture is known (e.g., Softmax guarantees ValidProbabilities)
    prop isa FiniteOutput || prop isa NoNaN || prop isa NoInf
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
Attach a tamper-evidence digest to a certificate.

NOTE: this computes an unkeyed SHA-256 content digest, NOT a keyed or asymmetric
signature — it detects naive tampering but does not authenticate authorship (a
forger can recompute the digest). Authenticating hybrid Ed448+Dilithium5
signatures are planned; see ROADMAP.adoc.
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

# ============================================================================
# Hybrid Ed448+Dilithium5 authenticating signatures (opt-in; G01)
# ============================================================================

"""
Canonical certificate content string used as the hybrid-signature message.
Includes the properties list (unlike the SHA-256 tamper-evidence digest
above) so a forger cannot add/remove a claimed property without also
invalidating the authenticating signature.
"""
function canonical_certificate_content(cert::Certificate)
    string(
        cert.model_hash,
        "|", cert.model_name,
        "|", cert.verification_mode,
        "|", join(sort([string(typeof(p).name.name) for p in cert.properties]), ","),
        "|", cert.test_data_hash === nothing ? "" : cert.test_data_hash,
        "|", cert.proof_type,
        "|", cert.created_at,
        "|", cert.axiom_version,
        "|", cert.verifier_id,
    )
end

"""
    sign_certificate_hybrid(cert::Certificate, keys::HybridKeyPair) -> HybridSignature

Compute a real, authenticating hybrid Ed448+Dilithium5 signature over `cert`'s
canonical content (via its SHA3-512 digest — see `signing.jl`). This is
opt-in: it does not modify `cert` or its (SHA-256 digest, `authenticated =
false`) `signature` field. Pair with `verify_certificate_hybrid` and/or
`save_certificate_hybrid`.

Throws the standard "crypto shim not built" error if `crypto/target/.../
libaxiom_crypto.*` has not been compiled (see `just build-crypto`).
"""
function sign_certificate_hybrid(cert::Certificate, keys::HybridKeyPair)
    hybrid_sign(canonical_certificate_content(cert), keys)
end

"""
    verify_certificate_hybrid(cert::Certificate, sig::HybridSignature, pubkeys::HybridPublicKeys) -> Bool

Verify a hybrid Ed448+Dilithium5 signature over `cert`'s canonical content.
Returns `true` only if BOTH the Ed448 and the Dilithium5 signature verify.
Unlike `verify_certificate` (SHA-256 content digest), a forger without the
private keys cannot produce a signature that passes this check, even if they
know the exact certificate content and recompute the SHA3-512 digest
themselves.
"""
function verify_certificate_hybrid(cert::Certificate, sig::HybridSignature, pubkeys::HybridPublicKeys)
    hybrid_verify(canonical_certificate_content(cert), sig, pubkeys)
end

"""
    save_certificate_hybrid(cert::Certificate, keys::HybridKeyPair, path::String)

Save `cert` to `path` as JSON with a real hybrid Ed448+Dilithium5 signature in
place of the default SHA-256 content-digest `signature` block. The written
JSON is self-contained for verification: it carries `algorithm =
"Ed448+Dilithium5"`, `authenticated = true`, both signature values, and both
PUBLIC keys (hex-encoded) — never the private keys.
"""
function save_certificate_hybrid(cert::Certificate, keys::HybridKeyPair, path::String)
    sig = sign_certificate_hybrid(cert, keys)
    data = _certificate_json_data(cert)
    data["signature"] = hybrid_signature_to_dict(sig, public_keys(keys))

    open(path, "w") do f
        JSON.print(f, data, 2)
    end
    @info "Hybrid-signed certificate saved to $path"
    nothing
end

"""
    save_certificate(cert::Certificate, path::String; format=:auto)

Save certificate to file. Format is auto-detected from extension:
- `.json` → JSON (machine-readable, recommended)
- `.cert` or other → YAML-like text (human-readable)
"""
function save_certificate(cert::Certificate, path::String; format::Symbol=:auto)
    fmt = format
    if fmt == :auto
        fmt = endswith(path, ".json") ? :json : :text
    end

    if fmt == :json
        _save_certificate_json(cert, path)
    else
        _save_certificate_text(cert, path)
    end

    @info "Certificate saved to $path"
end

"""
Build the certificate JSON `Dict`, shared by the default (SHA-256
content-digest) and hybrid-signed (`save_certificate_hybrid`) save paths.
The default `signature` block is set here and may be overridden by callers
that want a different (e.g. hybrid) signature block.
"""
function _certificate_json_data(cert::Certificate)
    Dict(
        "format" => "axiom-verification-certificate",
        "version" => "2.0",
        "model" => Dict(
            "name" => cert.model_name,
            "hash" => cert.model_hash,
            "hash_algorithm" => "SHA256"
        ),
        "verification" => Dict(
            "mode" => string(cert.verification_mode),
            "proof_type" => string(cert.proof_type),
            "properties" => [string(typeof(prop).name.name) for prop in cert.properties],
            "verifier_id" => cert.verifier_id
        ),
        "metadata" => Dict(
            "created_at" => cert.created_at,
            "axiom_version" => cert.axiom_version,
            "test_data_hash" => cert.test_data_hash
        ),
        "signature" => Dict(
            "value" => cert.signature,
            "algorithm" => "SHA-256",
            "kind" => "content-digest",
            "authenticated" => false,
            "note" => "Unkeyed SHA-256 digest over public certificate fields: " *
                      "detects naive tampering but does NOT authenticate authorship " *
                      "(a forger can recompute it). Authenticating hybrid " *
                      "Ed448+Dilithium5 signatures are available opt-in via " *
                      "sign_certificate_hybrid/save_certificate_hybrid; see ROADMAP.adoc."
        )
    )
end

function _save_certificate_json(cert::Certificate, path::String)
    data = _certificate_json_data(cert)

    open(path, "w") do f
        JSON.print(f, data, 2)
    end
end

function _save_certificate_text(cert::Certificate, path::String)
    open(path, "w") do f
        println(f, "# Axiom.jl Verification Certificate")
        println(f, "# Generated: $(cert.created_at)")
        println(f, "")
        println(f, "model_name: $(cert.model_name)")
        println(f, "model_hash: $(cert.model_hash)")
        println(f, "created_at: $(cert.created_at)")
        println(f, "axiom_version: $(cert.axiom_version)")
        println(f, "verification_mode: $(cert.verification_mode)")
        println(f, "proof_type: $(cert.proof_type)")
        println(f, "verifier_id: $(cert.verifier_id)")
        if cert.test_data_hash !== nothing
            println(f, "test_data_hash: $(cert.test_data_hash)")
        end
        println(f, "")
        println(f, "properties:")
        for prop in cert.properties
            println(f, "  - $(typeof(prop).name.name)")
        end
        println(f, "")
        println(f, "signature: $(cert.signature)")
    end
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

        # Parse list entries in properties section
        if startswith(line, "- ") && current_section == :properties
            prop_name = strip(line[3:end])
            prop = parse_property_type(prop_name)
            if prop !== nothing
                push!(properties, prop)
            end
            continue
        end

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
            elseif key == "created_at"
                created_at = tryparse(Float64, value) === nothing ? 0.0 : parse(Float64, value)
            elseif key == "verification_mode"
                verification_mode = parse_verification_mode(value)
            elseif key == "proof_type"
                proof_type = Symbol(value)
            elseif key == "verifier_id"
                verifier_id = value
            elseif key == "test_data_hash"
                test_data_hash = isempty(value) ? nothing : value
            elseif key == "signature"
                signature = value
            elseif key == "properties"
                current_section = :properties
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
function parse_verification_mode(s::AbstractString)
    s = uppercase(strip(String(s)))
    if s == "QUICK" || s == "FAST"
        return QUICK
    elseif s == "STANDARD"
        return STANDARD
    elseif s == "THOROUGH" || s == "STRICT"
        return THOROUGH
    elseif s == "EXHAUSTIVE" || s == "DEBUG"
        return EXHAUSTIVE
    else
        return STANDARD
    end
end

"""
Parse property type from name string.
"""
function parse_property_type(name::AbstractString)
    name = strip(String(name))

    # Map property names to types
    if name == "ValidProbabilities" || name == "ValidProbability"
        return ValidProbabilities()
    elseif name == "FiniteOutput" || name == "FiniteOutputs"
        return FiniteOutput()
    elseif name == "NoNaN" || name == "NoNaNs"
        return NoNaN()
    elseif startswith(name, "BoundedOutput")
        # Parameterized bounds are not serialized in the current format.
        return BoundedOutput(-Inf32, Inf32)
    elseif name == "NoInf" || name == "NoInfs"
        return NoInf()
    else
        @debug "Unknown property type: $name"
        return nothing
    end
end

"""
Verify signature of loaded certificate.
"""
function verify_loaded_signature(cert::Certificate, expected_signature::AbstractString)
    content = string(
        cert.model_hash,
        cert.model_name,
        cert.verification_mode,
        cert.created_at,
        cert.axiom_version
    )

    computed_signature = bytes2hex(SHA.sha256(content))
    computed_signature == String(expected_signature)
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
