# SPDX-License-Identifier: MPL-2.0
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
        theorem completed : forall x, relu x >= 0 := by
          intro x
          exact relu_nonneg x
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

        theorem completed_bundle : forall x, sum(softmax(x)) == 1.0 := by
          intro x
          exact softmax_sum_one x
        """
    )
    reconciled = reconcile_proof_bundle(bundle["manifest"])
    @test reconciled["obligations"][1]["assistant_reports"]["lean"]["status"] == "complete"
    @test reconciled["obligations"][1]["status"] == "interactive_required"
end

@testset "Proof Assistant Import Rejects Vacuous/Empty Proofs (G03)" begin
    # An empty file (or a file with only certificate-hash/obligation-id
    # metadata comments and no theorem at all) must NOT be reported as
    # verified merely because it contains no sorry/Admitted/oops tokens.
    empty_dir = mktempdir()

    empty_lean = joinpath(empty_dir, "empty.lean")
    write(empty_lean, "")
    empty_cert = import_lean_certificate(empty_lean)
    @test empty_cert.status != :proven
    empty_report = proof_assistant_obligation_report(empty_lean, :lean)
    @test empty_report["status"] == "incomplete" || empty_report["status"] == "missing_metadata"
    @test empty_report["has_statement"] == false

    metadata_only_lean = joinpath(empty_dir, "metadata_only.lean")
    write(
        metadata_only_lean,
        """
        -- AXIOM_CERTIFICATE_HASH: deadbeef
        -- AXIOM_OBLIGATION_ID: feedface
        """
    )
    metadata_only_cert = import_lean_certificate(metadata_only_lean)
    @test metadata_only_cert.status != :proven
    metadata_only_report = proof_assistant_obligation_report(metadata_only_lean, :lean)
    @test metadata_only_report["status"] == "incomplete"
    @test metadata_only_report["has_statement"] == false

    # A trivial/vacuous theorem (`theorem foo : True := trivial`) contains no
    # sorry/Admitted/oops escape token, but it also proves nothing about the
    # claimed property -- it must be rejected exactly like the empty case.
    trivial_lean = joinpath(empty_dir, "trivial.lean")
    write(
        trivial_lean,
        """
        -- AXIOM_CERTIFICATE_HASH: deadbeef
        -- AXIOM_OBLIGATION_ID: feedface
        theorem foo : True := trivial
        """
    )
    trivial_cert = import_lean_certificate(trivial_lean)
    @test trivial_cert.status != :proven
    trivial_report = proof_assistant_obligation_report(trivial_lean, :lean)
    @test trivial_report["status"] == "incomplete"
    @test trivial_report["has_statement"] == false
    @test trivial_report["unresolved"] == 0  # confirms rejection is NOT just token-scan

    trivial_coq = joinpath(empty_dir, "trivial.v")
    write(
        trivial_coq,
        """
        (* AXIOM_CERTIFICATE_HASH: deadbeef *)
        (* AXIOM_OBLIGATION_ID: feedface *)
        Theorem foo : True.
        Proof.
          exact I.
        Qed.
        """
    )
    trivial_coq_cert = import_coq_certificate(trivial_coq)
    @test trivial_coq_cert.status != :proven
    trivial_coq_report = proof_assistant_obligation_report(trivial_coq, :coq)
    @test trivial_coq_report["status"] == "incomplete"
    @test trivial_coq_report["has_statement"] == false

    trivial_thy = joinpath(empty_dir, "trivial.thy")
    write(
        trivial_thy,
        """
        (* AXIOM_CERTIFICATE_HASH: deadbeef *)
        (* AXIOM_OBLIGATION_ID: feedface *)
        lemma foo: "True"
          by simp
        """
    )
    trivial_isabelle_cert = import_isabelle_certificate(trivial_thy)
    @test trivial_isabelle_cert.status != :proven
    trivial_isabelle_report = proof_assistant_obligation_report(trivial_thy, :isabelle)
    @test trivial_isabelle_report["status"] == "incomplete"
    @test trivial_isabelle_report["has_statement"] == false

    # Sanity check: a real (non-vacuous) property theorem with no escape
    # tokens IS accepted -- the fix must not reject genuine proofs.
    real_lean = joinpath(empty_dir, "real.lean")
    write(
        real_lean,
        """
        -- AXIOM_CERTIFICATE_HASH: deadbeef
        -- AXIOM_OBLIGATION_ID: feedface
        theorem axiom_property_relu_nonneg : forall x, relu(x) >= 0 := by
          intro x
          exact relu_nonneg x
        """
    )
    real_cert = import_lean_certificate(real_lean)
    @test real_cert.status == :proven
    real_report = proof_assistant_obligation_report(real_lean, :lean)
    @test real_report["status"] == "complete"
    @test real_report["has_statement"] == true
end
