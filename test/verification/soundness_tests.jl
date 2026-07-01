# SPDX-License-Identifier: MPL-2.0
# Soundness regression tests — guard against reintroducing the P1 soundness holes:
#   G02: vacuous NoNaN static "proof" (has_safe_operations == true)
#   G01: certificate falsely labelling an unkeyed digest as "SHA256-HMAC"
# These run inside the main suite so the holes cannot silently reopen.

using Test
using Axiom
using JSON

@testset "Soundness (P1 holes)" begin
    model = Sequential(Dense(10, 5, relu), Dense(5, 3), Softmax())

    @testset "NoNaN is not a vacuous static proof (G02)" begin
        # Structure alone cannot soundly prove NoNaN (a finite-input guarantee is
        # also required), so the static path must return :unknown, never :proven.
        @test Axiom.try_static_verify(NoNaN(), model) === :unknown
        # The unconditional `true` is gone; the honest predicate makes no claim.
        @test Axiom.has_safe_operations(model) === false
        # Regression: genuinely-sound static proofs still work.
        @test Axiom.try_static_verify(ValidProbabilities(), model) === :proven
        # NoNaN still verifies empirically on well-behaved data (fallback path).
        x = Tensor(randn(Float32, 4, 10))
        res = verify(model; properties = [NoNaN()], data = [(x, nothing)])
        @test res.passed
    end

    @testset "Certificate does not falsely claim a keyed signature (G01)" begin
        x = Tensor(randn(Float32, 2, 10))
        result = verify(model; properties = [ValidProbabilities(), FiniteOutput()], data = [(x, nothing)])
        cert = generate_certificate(model, result; model_name = "soundness-ci")

        path = tempname() * ".json"
        save_certificate(cert, path; format = :json)
        try
            content = read(path, String)
            # The false "SHA256-HMAC" label (unkeyed digest sold as HMAC) is gone.
            @test !occursin("SHA256-HMAC", content)
            parsed = JSON.parsefile(path)
            @test parsed["signature"]["authenticated"] == false
            @test parsed["signature"]["kind"] == "content-digest"
        finally
            rm(path; force = true)
        end

        # Tamper-evidence: an altered digest is rejected. (A forger who recomputes
        # the digest still passes — the documented limit of a content digest;
        # authenticating Ed448+Dilithium5 signatures are tracked in ROADMAP.adoc.)
        @test verify_certificate(cert)
        forged = Axiom.Certificate(cert.model_hash, cert.model_name, cert.properties,
            cert.verification_mode, cert.test_data_hash, cert.proof_type,
            cert.created_at, cert.axiom_version, cert.verifier_id, "deadbeef")
        @test !verify_certificate(forged)
    end
end
