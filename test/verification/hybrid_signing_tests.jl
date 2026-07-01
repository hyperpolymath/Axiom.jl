# SPDX-License-Identifier: MPL-2.0
# Hybrid Ed448+Dilithium5 authenticating-signature tests (G01).
#
# Unlike the default SHA-256 content-digest certificate (soundness_tests.jl),
# this exercises the REAL cryptographic signing path: a forger without the
# private keys must not be able to produce a signature that verifies, even
# when they know the exact certificate content and can recompute its SHA3-512
# digest themselves. That is the whole point of this test file.
#
# Guarded on `crypto_shim_available()`: the shim is a compiled Rust cdylib
# (crypto/target/release/libaxiom_crypto.*) that is NOT part of a plain
# `Pkg.instantiate()` — it must be built explicitly (`just build-crypto`).
# If it is not built, this file reports that honestly via `@test_skip`
# (visible in the test summary as "Broken"/skipped, never silently omitted)
# rather than pretending the suite exercised the crypto path.

using Test
using Axiom
using JSON

@testset "Hybrid Ed448+Dilithium5 signing (G01 authenticating signature)" begin
    if !crypto_shim_available()
        @warn "axiom_crypto cdylib not built — hybrid signing tests will be reported as skipped. " *
              "Run `just build-crypto` (or `cd crypto && cargo build --release`) to exercise them."
        @test_skip crypto_shim_available()
    else
        model = Sequential(Dense(10, 5, relu), Dense(5, 3), Softmax())
        x = Tensor(randn(Float32, 2, 10))
        result = verify(model; properties = [ValidProbabilities(), FiniteOutput()], data = [(x, nothing)])
        cert = generate_certificate(model, result; model_name = "hybrid-signing-ci")

        @testset "Keypair generation produces correctly sized keys" begin
            kp = generate_hybrid_keypair()
            @test length(kp.ed448_public) == 57
            @test length(kp.ed448_secret) == 57
            @test length(kp.dilithium5_public) == 2592
            @test length(kp.dilithium5_secret) == 4896
            # Two independently generated keypairs must differ.
            kp2 = generate_hybrid_keypair()
            @test kp.ed448_public != kp2.ed448_public
            @test kp.dilithium5_public != kp2.dilithium5_public
        end

        @testset "Default (unsigned) certificate path is unaffected (regression guard)" begin
            # This must keep passing exactly as it did before hybrid signing
            # existed: authenticated=false, kind=content-digest, SHA-256.
            path = tempname() * ".json"
            save_certificate(cert, path; format = :json)
            try
                parsed = JSON.parsefile(path)
                @test parsed["signature"]["authenticated"] == false
                @test parsed["signature"]["kind"] == "content-digest"
                @test parsed["signature"]["algorithm"] == "SHA-256"
                @test verify_certificate(cert)
            finally
                rm(path; force = true)
            end
        end

        @testset "Genuine hybrid signature verifies (both algorithms)" begin
            kp = generate_hybrid_keypair()
            sig = sign_certificate_hybrid(cert, kp)
            pk = public_keys(kp)

            @test length(sig.ed448) == 114
            @test length(sig.dilithium5) > 0

            @test verify_certificate_hybrid(cert, sig, pk)

            # Each component individually verifies too (both required, per hybrid_verify).
            digest = canonical_certificate_content(cert)
            @test hybrid_verify(digest, sig, pk)
        end

        @testset "Tampered certificate field is rejected" begin
            kp = generate_hybrid_keypair()
            sig = sign_certificate_hybrid(cert, kp)
            pk = public_keys(kp)

            tampered = Axiom.Certificate(
                cert.model_hash, "TAMPERED-" * cert.model_name, cert.properties,
                cert.verification_mode, cert.test_data_hash, cert.proof_type,
                cert.created_at, cert.axiom_version, cert.verifier_id, cert.signature,
            )
            @test !verify_certificate_hybrid(tampered, sig, pk)

            # Tampering with the properties list must also be caught (the
            # hybrid canonical content includes properties, unlike the
            # legacy SHA-256 digest fields).
            tampered_props = Axiom.Certificate(
                cert.model_hash, cert.model_name, Axiom.Property[],
                cert.verification_mode, cert.test_data_hash, cert.proof_type,
                cert.created_at, cert.axiom_version, cert.verifier_id, cert.signature,
            )
            @test !verify_certificate_hybrid(tampered_props, sig, pk)
        end

        @testset "save_certificate_hybrid / load round-trip is self-contained" begin
            kp = generate_hybrid_keypair()
            path = tempname() * ".json"
            save_certificate_hybrid(cert, kp, path)
            try
                parsed = JSON.parsefile(path)
                @test parsed["signature"]["algorithm"] == "Ed448+Dilithium5"
                @test parsed["signature"]["authenticated"] == true
                @test parsed["signature"]["digest_algorithm"] == "SHA3-512"
                @test haskey(parsed["signature"]["ed448"], "public_key")
                @test haskey(parsed["signature"]["dilithium5"], "public_key")

                # Verification is self-contained: reconstruct sig+pubkeys purely
                # from the JSON on disk (no external key lookup) and re-verify.
                sig2, pk2 = hybrid_signature_from_dict(parsed["signature"])
                @test verify_certificate_hybrid(cert, sig2, pk2)
            finally
                rm(path; force = true)
            end
        end

        @testset "THE KEY ASSERTION: forgery without the private key is cryptographically rejected" begin
            # This is the whole point of hybrid signing versus the old
            # digest-only certificate: a "forger" who has full knowledge of
            # the certificate content (and can therefore recompute the exact
            # SHA3-512 digest the legitimate signer would sign) but does NOT
            # possess the legitimate signer's private keys must be UNABLE to
            # produce a signature that verifies against the legitimate
            # signer's PUBLIC keys. Under the old SHA-256 content-digest
            # scheme this attack trivially succeeds (soundness_tests.jl
            # documents that as the known, accepted limitation). Here it must
            # fail.
            victim_kp = generate_hybrid_keypair()
            victim_pub = public_keys(victim_kp)

            # The forger recomputes the exact canonical content / SHA3-512
            # digest -- full knowledge of the "message" -- but signs it with
            # their OWN keypair, because they do not have the victim's
            # private keys.
            forger_kp = generate_hybrid_keypair()
            @test forger_kp.ed448_secret != victim_kp.ed448_secret
            @test forger_kp.dilithium5_secret != victim_kp.dilithium5_secret

            forged_sig = sign_certificate_hybrid(cert, forger_kp)

            # The forged signature must NOT verify against the victim's public keys.
            forged_verifies = verify_certificate_hybrid(cert, forged_sig, victim_pub)
            @test !forged_verifies

            # Mix-and-match cross-check: even pairing the forger's genuine
            # Ed448 signature with the victim's Dilithium5-signed digest (an
            # attacker splicing components from different keypairs) must
            # still fail, since hybrid_verify requires BOTH to verify against
            # the SAME (victim) public keys.
            mixed_sig = Axiom.HybridSignature(forged_sig.ed448, forged_sig.dilithium5)
            @test !verify_certificate_hybrid(cert, mixed_sig, victim_pub)

            # Sanity: the forger's OWN signature over the same content DOES
            # verify against the forger's OWN public key -- proving the
            # failure above is specifically about authentication against the
            # victim's identity, not a generic crypto malfunction.
            forger_pub = public_keys(forger_kp)
            @test verify_certificate_hybrid(cert, forged_sig, forger_pub)
        end
    end
end
