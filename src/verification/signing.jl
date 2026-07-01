# SPDX-License-Identifier: MPL-2.0
# Axiom.jl Hybrid Post-Quantum Certificate Signing
#
# Authenticating hybrid Ed448 (classical) + Dilithium5/ML-DSA-87 (post-quantum)
# signatures over certificate content, per the estate Trustfile scheme. This
# closes the authenticating-signature half of G01: the *default* certificate
# produced by `generate_certificate` remains a SHA-256 content digest with
# `authenticated=false` (see certificates.jl) — this module adds an *opt-in*
# real-signature path on top of it, it does not change the default.
#
# Both Ed448 and Dilithium5 are vetted, non-hand-rolled primitives:
#   - Ed448 via the system libcrypto (OpenSSL 3.x), through the Rust `openssl`
#     crate.
#   - Dilithium5 (ML-DSA-87, FIPS 204) via the Rust `pqcrypto-dilithium` crate
#     (PQClean reference implementation).
# The Rust shim lives in `crypto/` (crate `axiom_crypto`) and is compiled to a
# cdylib loaded here via Libdl. See `crypto/README.md` for the full C-ABI
# reference.
#
# NEVER commit a private key. Real signing keys are generated out-of-band
# (HSM / offline signing process per ROADMAP.adoc); only the resulting PUBLIC
# keys are ever embedded in a certificate. `generate_hybrid_keypair()` below
# is for tests/examples and returns keys held only in memory.

using Libdl
using SHA

# ============================================================================
# Library loading
# ============================================================================

const _AXIOM_CRYPTO_LIB_CANDIDATES = [
    joinpath(@__DIR__, "..", "..", "crypto", "target", "release", "libaxiom_crypto.so"),
    joinpath(@__DIR__, "..", "..", "crypto", "target", "release", "libaxiom_crypto.dylib"),
    joinpath(@__DIR__, "..", "..", "crypto", "target", "release", "libaxiom_crypto.dll"),
    joinpath(@__DIR__, "..", "..", "crypto", "target", "debug", "libaxiom_crypto.so"),
    joinpath(@__DIR__, "..", "..", "crypto", "target", "debug", "libaxiom_crypto.dylib"),
    joinpath(@__DIR__, "..", "..", "crypto", "target", "debug", "libaxiom_crypto.dll"),
]

const _crypto_lib = Ref{Ptr{Nothing}}(C_NULL)
const _crypto_available = Ref(false)

"""
Locate the built `axiom_crypto` cdylib, if any. Returns the path or `nothing`.
Honors `AXIOM_CRYPTO_LIB` env var override (matches the `AXIOM_ZIG_LIB`
convention used by the Zig backend).
"""
function _find_crypto_lib()
    override = get(ENV, "AXIOM_CRYPTO_LIB", "")
    if !isempty(override) && isfile(override)
        return override
    end
    for candidate in _AXIOM_CRYPTO_LIB_CANDIDATES
        if isfile(candidate)
            return candidate
        end
    end
    return nothing
end

function _ensure_crypto_loaded!()
    _crypto_available[] && return true

    lib_path = _find_crypto_lib()
    if lib_path === nothing
        return false
    end

    try
        _crypto_lib[] = Libdl.dlopen(lib_path)
        _crypto_available[] = true
        return true
    catch e
        @warn "Failed to load axiom_crypto shim from $lib_path: $e"
        return false
    end
end

"""
    crypto_shim_available() -> Bool

Whether the compiled `axiom_crypto` cdylib (hybrid Ed448+Dilithium5 signing)
is available in this process.
"""
crypto_shim_available() = _ensure_crypto_loaded!()

function _crypto_sym(name::Symbol)
    _ensure_crypto_loaded!() || _crypto_not_built_error()
    Libdl.dlsym(_crypto_lib[], name)
end

function _crypto_not_built_error()
    error(
        "crypto shim not built — run `just build-crypto` (or `cd crypto && " *
        "cargo build --release`) to build the axiom_crypto cdylib before " *
        "using hybrid_sign / hybrid_verify / generate_hybrid_keypair. The " *
        "rest of Axiom.jl (including the default digest-only certificate " *
        "path) works without it."
    )
end

# ============================================================================
# Status codes (mirror crypto/src/lib.rs)
# ============================================================================

const _AXIOM_CRYPTO_OK = Int32(0)

# ============================================================================
# Fixed sizes (mirror the Rust *_len()/_maxlen() getters; validated at load
# time against the actual shim in `_crypto_sizes_consistent()` below so a
# stale/mismatched build fails loudly instead of silently corrupting buffers).
# ============================================================================

const ED448_PUBLIC_KEY_LEN = 57
const ED448_SECRET_KEY_LEN = 57
const ED448_SIGNATURE_LEN = 114
const DILITHIUM5_PUBLIC_KEY_LEN = 2592
const DILITHIUM5_SECRET_KEY_LEN = 4896
const DILITHIUM5_SIGNATURE_MAXLEN = 4627

function _checked_len(sym::Symbol, expected::Int)
    fn = _crypto_sym(sym)
    got = Int(ccall(fn, Csize_t, ()))
    if got != expected
        error(
            "axiom_crypto ABI mismatch: $sym returned $got, Julia side expects " *
            "$expected. The compiled crypto/target library does not match this " *
            "version of signing.jl — rebuild with `just build-crypto`."
        )
    end
    got
end

"""
Validate that the loaded shim's advertised buffer sizes match what this
Julia module expects, so a stale build fails loudly rather than corrupting
memory. Called lazily by every public entry point below.
"""
function _validate_crypto_abi!()
    _checked_len(:axiom_crypto_ed448_public_key_len, ED448_PUBLIC_KEY_LEN)
    _checked_len(:axiom_crypto_ed448_secret_key_len, ED448_SECRET_KEY_LEN)
    _checked_len(:axiom_crypto_ed448_signature_len, ED448_SIGNATURE_LEN)
    _checked_len(:axiom_crypto_dilithium5_public_key_len, DILITHIUM5_PUBLIC_KEY_LEN)
    _checked_len(:axiom_crypto_dilithium5_secret_key_len, DILITHIUM5_SECRET_KEY_LEN)
    _checked_len(:axiom_crypto_dilithium5_signature_maxlen, DILITHIUM5_SIGNATURE_MAXLEN)
    nothing
end

# ============================================================================
# Key containers
# ============================================================================

"""
    HybridKeyPair

An in-memory Ed448 + Dilithium5 keypair. Only ever produced by
`generate_hybrid_keypair()` (tests/examples) or reconstructed from
out-of-band–generated raw key bytes; never deserialized from a certificate
(certificates only ever carry public keys).
"""
struct HybridKeyPair
    ed448_public::Vector{UInt8}
    ed448_secret::Vector{UInt8}
    dilithium5_public::Vector{UInt8}
    dilithium5_secret::Vector{UInt8}
end

"""
    HybridPublicKeys

The public half of a `HybridKeyPair` — safe to embed in a certificate.
"""
struct HybridPublicKeys
    ed448_public::Vector{UInt8}
    dilithium5_public::Vector{UInt8}
end

public_keys(kp::HybridKeyPair) = HybridPublicKeys(kp.ed448_public, kp.dilithium5_public)

"""
    HybridSignature

A hybrid Ed448 + Dilithium5 signature over the same message. Both components
must verify for `hybrid_verify` to return `true`.
"""
struct HybridSignature
    ed448::Vector{UInt8}
    dilithium5::Vector{UInt8}
end

# ============================================================================
# Keypair generation
# ============================================================================

"""
    generate_hybrid_keypair() -> HybridKeyPair

Generate a fresh in-memory Ed448 + Dilithium5 keypair. For tests and
examples. Real certificate-signing keys are generated and held out-of-band
(HSM / offline signing process); this function is not that pathway — it
exists so tests can exercise `hybrid_sign`/`hybrid_verify` without a
pre-provisioned key.
"""
function generate_hybrid_keypair()
    _validate_crypto_abi!()

    ed_pub = Vector{UInt8}(undef, ED448_PUBLIC_KEY_LEN)
    ed_sec = Vector{UInt8}(undef, ED448_SECRET_KEY_LEN)
    rc = ccall(
        _crypto_sym(:axiom_crypto_ed448_keypair), Int32,
        (Ptr{UInt8}, Ptr{UInt8}), ed_pub, ed_sec
    )
    rc == _AXIOM_CRYPTO_OK || error("axiom_crypto_ed448_keypair failed (code $rc)")

    dl_pub = Vector{UInt8}(undef, DILITHIUM5_PUBLIC_KEY_LEN)
    dl_sec = Vector{UInt8}(undef, DILITHIUM5_SECRET_KEY_LEN)
    rc = ccall(
        _crypto_sym(:axiom_crypto_dilithium5_keypair), Int32,
        (Ptr{UInt8}, Ptr{UInt8}), dl_pub, dl_sec
    )
    rc == _AXIOM_CRYPTO_OK || error("axiom_crypto_dilithium5_keypair failed (code $rc)")

    HybridKeyPair(ed_pub, ed_sec, dl_pub, dl_sec)
end

# ============================================================================
# Content digest (estate hashing standard: SHA3-512)
# ============================================================================

"""
    certificate_content_digest(content::AbstractString) -> Vector{UInt8}

Compute the SHA3-512 digest of certificate content, per the estate hashing
standard. This is the message the hybrid signature is computed over (either
directly, or the hex-encoded digest string — `hybrid_sign` accepts raw
content and does this internally).
"""
certificate_content_digest(content::AbstractString) = SHA.sha3_512(String(content))

# ============================================================================
# Sign / verify
# ============================================================================

"""
    hybrid_sign(content::AbstractString, keys::HybridKeyPair) -> HybridSignature

Sign `content` with both Ed448 and Dilithium5. The message actually signed is
the SHA3-512 digest of `content` (not `content` itself), so signature size is
independent of certificate size and matches `certificate_content_digest`.
"""
function hybrid_sign(content::AbstractString, keys::HybridKeyPair)
    _validate_crypto_abi!()
    digest = certificate_content_digest(content)
    _hybrid_sign_digest(digest, keys)
end

function _hybrid_sign_digest(digest::Vector{UInt8}, keys::HybridKeyPair)
    length(keys.ed448_secret) == ED448_SECRET_KEY_LEN ||
        error("malformed Ed448 secret key (expected $ED448_SECRET_KEY_LEN bytes)")
    length(keys.dilithium5_secret) == DILITHIUM5_SECRET_KEY_LEN ||
        error("malformed Dilithium5 secret key (expected $DILITHIUM5_SECRET_KEY_LEN bytes)")

    ed_sig = Vector{UInt8}(undef, ED448_SIGNATURE_LEN)
    rc = ccall(
        _crypto_sym(:axiom_crypto_ed448_sign), Int32,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Ptr{UInt8}),
        digest, length(digest), keys.ed448_secret, ed_sig
    )
    rc == _AXIOM_CRYPTO_OK || error("axiom_crypto_ed448_sign failed (code $rc)")

    dl_sig = Vector{UInt8}(undef, DILITHIUM5_SIGNATURE_MAXLEN)
    sig_len = Ref{Csize_t}(0)
    rc = ccall(
        _crypto_sym(:axiom_crypto_dilithium5_sign), Int32,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Ptr{UInt8}, Ptr{Csize_t}),
        digest, length(digest), keys.dilithium5_secret, dl_sig, sig_len
    )
    rc == _AXIOM_CRYPTO_OK || error("axiom_crypto_dilithium5_sign failed (code $rc)")
    resize!(dl_sig, Int(sig_len[]))

    HybridSignature(ed_sig, dl_sig)
end

"""
    hybrid_verify(content::AbstractString, sig::HybridSignature, pubkeys::HybridPublicKeys) -> Bool

Verify a hybrid signature over `content`. Returns `true` only if **both**
the Ed448 and the Dilithium5 signature verify against the SHA3-512 digest of
`content`. Any malformed input (wrong-length keys/signature) is treated as
"does not verify" rather than throwing, so callers can use this directly as
a boolean gate on untrusted certificate data.
"""
function hybrid_verify(content::AbstractString, sig::HybridSignature, pubkeys::HybridPublicKeys)
    _validate_crypto_abi!()
    digest = certificate_content_digest(content)
    _hybrid_verify_digest(digest, sig, pubkeys)
end

function _hybrid_verify_digest(digest::Vector{UInt8}, sig::HybridSignature, pubkeys::HybridPublicKeys)
    if length(pubkeys.ed448_public) != ED448_PUBLIC_KEY_LEN ||
       length(pubkeys.dilithium5_public) != DILITHIUM5_PUBLIC_KEY_LEN ||
       length(sig.ed448) != ED448_SIGNATURE_LEN ||
       isempty(sig.dilithium5) || length(sig.dilithium5) > DILITHIUM5_SIGNATURE_MAXLEN
        return false
    end

    ed_ok = ccall(
        _crypto_sym(:axiom_crypto_ed448_verify), Int32,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Ptr{UInt8}),
        digest, length(digest), sig.ed448, pubkeys.ed448_public
    )
    ed_ok == 1 || return false

    dl_ok = ccall(
        _crypto_sym(:axiom_crypto_dilithium5_verify), Int32,
        (Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, Ptr{UInt8}),
        digest, length(digest), sig.dilithium5, length(sig.dilithium5), pubkeys.dilithium5_public
    )
    dl_ok == 1
end

# ============================================================================
# Hex encoding helpers (for JSON embedding)
# ============================================================================

hex_encode(bytes::AbstractVector{UInt8}) = bytes2hex(bytes)
hex_decode(s::AbstractString) = hex2bytes(String(s))

"""
    hybrid_signature_to_dict(sig::HybridSignature, pubkeys::HybridPublicKeys) -> Dict

Build the JSON-serializable `signature` block for a signed certificate:
`algorithm="Ed448+Dilithium5"`, `authenticated=true`, both signature values
and both public keys (hex-encoded), so verification is self-contained
(no external key lookup required).
"""
function hybrid_signature_to_dict(sig::HybridSignature, pubkeys::HybridPublicKeys)
    Dict(
        "algorithm" => "Ed448+Dilithium5",
        "kind" => "hybrid-signature",
        "authenticated" => true,
        "digest_algorithm" => "SHA3-512",
        "ed448" => Dict(
            "signature" => hex_encode(sig.ed448),
            "public_key" => hex_encode(pubkeys.ed448_public),
        ),
        "dilithium5" => Dict(
            "signature" => hex_encode(sig.dilithium5),
            "public_key" => hex_encode(pubkeys.dilithium5_public),
        ),
        "note" => "Hybrid Ed448 (classical, RFC 8032) + Dilithium5/ML-DSA-87 " *
                  "(post-quantum, FIPS 204) signature over the SHA3-512 digest " *
                  "of the certificate content. Both components must verify.",
    )
end

"""
    hybrid_signature_from_dict(d) -> Union{Tuple{HybridSignature,HybridPublicKeys}, Nothing}

Parse the `signature` block produced by `hybrid_signature_to_dict` back into
a `(HybridSignature, HybridPublicKeys)` pair, or `nothing` if `d` does not
look like a hybrid-signature block (e.g. it's the legacy content-digest
block).
"""
function hybrid_signature_from_dict(d)
    (d isa AbstractDict) || return nothing
    get(d, "algorithm", nothing) == "Ed448+Dilithium5" || return nothing
    haskey(d, "ed448") && haskey(d, "dilithium5") || return nothing

    ed = d["ed448"]
    dl = d["dilithium5"]
    sig = HybridSignature(hex_decode(ed["signature"]), hex_decode(dl["signature"]))
    pubkeys = HybridPublicKeys(hex_decode(ed["public_key"]), hex_decode(dl["public_key"]))
    (sig, pubkeys)
end
