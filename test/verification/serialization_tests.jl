# SPDX-License-Identifier: PMPL-1.0-or-later
# Tests for proof serialization

using Test
using Axiom
using Axiom: ProofResult
using JSON
using Dates

@testset "Proof Serialization" begin
    @testset "Serialize proven result" begin
        result = ProofResult(:proven, nothing, 1.0,
                            "ReLU is non-negative by definition",
                            String[])

        cert_dict = serialize_proof(result, "∀x. relu(x) >= 0",
                                   proof_method="pattern",
                                   execution_time_ms=0.5)

        @test cert_dict["version"] == "1.0"
        @test cert_dict["format"] == "axiom-proof-certificate"
        @test cert_dict["property"] == "∀x. relu(x) >= 0"
        @test cert_dict["result"]["status"] == "proven"
        @test cert_dict["result"]["confidence"] == 1.0
        @test cert_dict["result"]["details"] == "ReLU is non-negative by definition"
        @test cert_dict["artifacts"]["proof_method"] == "pattern"
        @test cert_dict["artifacts"]["execution_time_ms"] == 0.5
        @test haskey(cert_dict["audit"], "hash")
        @test cert_dict["audit"]["hash_algorithm"] == "SHA256"
    end

    @testset "Serialize disproven result with counterexample" begin
        result = ProofResult(:disproven, Dict("x" => -1.0), 1.0,
                            "Found counterexample",
                            ["Try constraining input domain", "Add precondition"])

        cert_dict = serialize_proof(result, "∀x. exp(x) < 1",
                                   proof_method="smt",
                                   execution_time_ms=42.3,
                                   smt_solver="z3")

        @test cert_dict["result"]["status"] == "disproven"
        @test cert_dict["result"]["counterexample"] == Dict("x" => -1.0)
        @test length(cert_dict["result"]["suggestions"]) == 2
        @test cert_dict["artifacts"]["smt_solver"] == "z3"
    end

    @testset "Serialize unknown result" begin
        result = ProofResult(:unknown, nothing, 0.3,
                            "Property too complex for SMT solver",
                            ["Try simplifying property", "Consider runtime checks"])

        cert_dict = serialize_proof(result, "∀x. complex_function(x) > 0",
                                   proof_method="smt",
                                   execution_time_ms=5000.0)

        @test cert_dict["result"]["status"] == "unknown"
        @test cert_dict["result"]["confidence"] == 0.3
        @test cert_dict["artifacts"]["execution_time_ms"] == 5000.0
    end

    @testset "Serialize/deserialize round-trip" begin
        result = ProofResult(:proven, nothing, 1.0,
                            "Softmax outputs sum to 1",
                            String[])

        cert_dict = serialize_proof(result, "∀x. sum(softmax(x)) == 1.0",
                                   proof_method="pattern",
                                   smt_query="(assert (= (+ ...) 1.0))",
                                   smt_output="unsat")

        # Serialize to JSON string and back
        json_str = JSON.json(cert_dict)
        parsed = JSON.parse(json_str)

        # Deserialize
        cert = deserialize_proof(parsed)

        @test cert.property == "∀x. sum(softmax(x)) == 1.0"
        @test cert.status == :proven
        @test cert.confidence == 1.0
        @test cert.details == "Softmax outputs sum to 1"
        @test cert.proof_method == "pattern"
        @test cert.smt_query == "(assert (= (+ ...) 1.0))"
        @test cert.smt_output == "unsat"
        @test !isempty(cert.hash)
    end

    @testset "Certificate export/import" begin
        result = ProofResult(:proven, nothing, 1.0,
                            "Sigmoid is bounded",
                            String[])

        cert_dict = serialize_proof(result, "∀x. 0 < sigmoid(x) < 1",
                                   proof_method="pattern")

        # Export to temp file
        tmpfile = tempname() * ".proof"
        export_proof_certificate(cert_dict, tmpfile)
        @test isfile(tmpfile)

        # Import and verify
        cert = import_proof_certificate(tmpfile)
        @test cert.property == "∀x. 0 < sigmoid(x) < 1"
        @test cert.status == :proven

        # Cleanup
        rm(tmpfile)
    end

    @testset "Certificate integrity verification" begin
        result = ProofResult(:proven, nothing, 1.0,
                            "Test property",
                            String[])

        cert_dict = serialize_proof(result, "test_property",
                                   proof_method="test")

        cert = deserialize_proof(cert_dict)

        # Verify valid certificate
        @test verify_proof_certificate(cert)

        # Tamper with certificate
        tampered = ProofCertificate(
            "tampered_property",  # Changed property
            cert.status,
            cert.counterexample,
            cert.confidence,
            cert.details,
            cert.suggestions,
            cert.timestamp,
            cert.axiom_version,
            cert.julia_version,
            cert.hostname,
            cert.smt_query,
            cert.smt_output,
            cert.smt_solver,
            cert.proof_method,
            cert.execution_time_ms,
            cert.hash  # Old hash - doesn't match
        )

        # Verification should fail
        @test !verify_proof_certificate(tampered)
    end

    @testset "Metadata generation" begin
        result = ProofResult(:proven, nothing, 1.0, "", String[])

        cert_dict = serialize_proof(result, "test", proof_method="test")

        # Check metadata fields exist
        @test haskey(cert_dict["metadata"], "timestamp")
        @test haskey(cert_dict["metadata"], "axiom_version")
        @test haskey(cert_dict["metadata"], "julia_version")
        @test haskey(cert_dict["metadata"], "hostname")

        # Verify timestamp format (ISO 8601)
        ts = cert_dict["metadata"]["timestamp"]
        @test occursin(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts)

        # Julia version should be non-empty
        @test !isempty(cert_dict["metadata"]["julia_version"])
    end

    @testset "Counterexample serialization" begin
        # Dict counterexample
        result1 = ProofResult(:disproven, Dict("x" => 1.0, "y" => -2.0), 1.0, "", String[])
        cert1 = serialize_proof(result1, "test1", proof_method="test")
        @test cert1["result"]["counterexample"] == Dict("x" => 1.0, "y" => -2.0)

        # NamedTuple counterexample
        result2 = ProofResult(:disproven, (x=1.0, y=-2.0), 1.0, "", String[])
        cert2 = serialize_proof(result2, "test2", proof_method="test")
        @test cert2["result"]["counterexample"]["x"] == 1.0
        @test cert2["result"]["counterexample"]["y"] == -2.0

        # String counterexample
        result3 = ProofResult(:disproven, "invalid input", 1.0, "", String[])
        cert3 = serialize_proof(result3, "test3", proof_method="test")
        @test cert3["result"]["counterexample"] == "invalid input"

        # Nothing counterexample
        result4 = ProofResult(:proven, nothing, 1.0, "", String[])
        cert4 = serialize_proof(result4, "test4", proof_method="test")
        @test cert4["result"]["counterexample"] === nothing
    end
end
