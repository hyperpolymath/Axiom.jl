# SPDX-License-Identifier: PMPL-1.0-or-later
# Tests for proof-assistant export bundle and obligation metadata.

using Test
using JSON
using Axiom
using Axiom: ProofResult

@testset "Proof Assistant Export Bundle" begin
    result = ProofResult(
        :proven,
        nothing,
        1.0,
        "Softmax outputs are normalized",
        String[]
    )

    cert_dict = serialize_proof(
        result,
        "∀x. sum(softmax(x)) == 1.0";
        proof_method = "pattern",
        execution_time_ms = 0.7
    )
    cert = deserialize_proof(cert_dict)

    bundle_dir = mktempdir()
    bundle = export_proof_bundle(cert, bundle_dir; base_name = "softmax_norm")

    @test haskey(bundle, "manifest")
    @test haskey(bundle, "assistants")
    @test isfile(bundle["manifest"])

    assistant_paths = bundle["assistants"]
    @test isfile(assistant_paths["lean"])
    @test isfile(assistant_paths["coq"])
    @test isfile(assistant_paths["isabelle"])

    manifest = JSON.parsefile(bundle["manifest"])
    @test manifest["format"] == "axiom-proof-assistant-bundle.v1"
    @test manifest["certificate_hash"] == cert.hash
    @test manifest["proof_status"] == "proven"
    @test length(manifest["obligations"]) == 1
    @test manifest["obligations"][1]["status"] == "proven_by_certificate"

    lean_src = read(assistant_paths["lean"], String)
    @test occursin("AXIOM_CERTIFICATE_HASH: $(cert.hash)", lean_src)
    @test occursin("axiom_certificate_witness", lean_src)
    @test occursin("PROOF OBLIGATION", lean_src)

    coq_src = read(assistant_paths["coq"], String)
    @test occursin("AXIOM_CERTIFICATE_HASH: $(cert.hash)", coq_src)
    @test occursin("axiom_certificate_witness", coq_src)
    @test occursin("Admitted.", coq_src)

    imported_lean = import_lean_certificate(assistant_paths["lean"])
    imported_coq = import_coq_certificate(assistant_paths["coq"])

    @test imported_lean.status == :unknown
    @test imported_coq.status == :unknown
    @test occursin("Unresolved obligations:", imported_lean.details)
    @test occursin("Unresolved obligations:", imported_coq.details)

    completed_lean = joinpath(bundle_dir, "completed.lean")
    write(
        completed_lean,
        """
        -- AXIOM_CERTIFICATE_HASH: deadbeef
        -- AXIOM_OBLIGATION_ID: feedface
        theorem completed : True := by
          trivial
        """
    )
    completed_cert = import_lean_certificate(completed_lean)
    @test completed_cert.status == :proven

    expected_id = manifest["obligations"][1]["id"]
    report = proof_assistant_obligation_report(
        assistant_paths["coq"],
        :coq;
        expected_certificate_hash = cert.hash,
        expected_obligation_id = expected_id,
    )
    @test report["status"] == "incomplete"
    @test report["hash_matches"] == true
    @test report["obligation_matches"] == true

    isabelle_report = proof_assistant_obligation_report(
        assistant_paths["isabelle"],
        :isabelle;
        expected_certificate_hash = cert.hash,
        expected_obligation_id = expected_id,
    )
    @test isabelle_report["status"] == "incomplete"
    @test isabelle_report["hash_matches"] == true
    @test isabelle_report["obligation_matches"] == true

    completed_bundle_lean = joinpath(bundle_dir, "softmax_norm.lean")
    write(
        completed_bundle_lean,
        """
        -- AXIOM_CERTIFICATE_HASH: $(cert.hash)
        -- AXIOM_PROOF_STATUS: proven
        -- AXIOM_PROOF_METHOD: pattern
        -- AXIOM_OBLIGATION_ID: $(expected_id)

        theorem completed_bundle : True := by
          trivial
        """
    )
    reconciled = reconcile_proof_bundle(bundle["manifest"])
    @test reconciled["obligations"][1]["assistant_reports"]["lean"]["status"] == "complete"
    @test reconciled["obligations"][1]["status"] == "interactive_required"
end
