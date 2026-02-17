# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using JSON
using Axiom
using Axiom: ProofResult

function sample_certificate()
    result = ProofResult(
        :proven,
        nothing,
        1.0,
        "Proof-bundle CI reconciliation check",
        String[],
    )

    cert_dict = serialize_proof(
        result,
        "∀x. sum(softmax(x)) == 1.0";
        proof_method = "pattern",
        execution_time_ms = 0.25,
    )

    deserialize_proof(cert_dict)
end

@testset "Proof assistant bundle reconciliation CI" begin
    cert = sample_certificate()
    bundle_dir = mktempdir()
    bundle = export_proof_bundle(cert, bundle_dir; base_name = "ci_bundle")
    manifest = JSON.parsefile(bundle["manifest"])

    expected_hash = cert.hash
    expected_id = manifest["obligations"][1]["id"]

    # Generated artifacts should all be unresolved placeholders initially.
    for (assistant_name, path) in bundle["assistants"]
        report = proof_assistant_obligation_report(
            path,
            Symbol(assistant_name);
            expected_certificate_hash = expected_hash,
            expected_obligation_id = expected_id,
        )
        @test report["status"] == "incomplete"
        @test report["complete"] == false
    end

    initial = reconcile_proof_bundle(bundle["manifest"])
    @test initial["obligations"][1]["status"] == "interactive_required"

    write(
        bundle["assistants"]["lean"],
        """
        -- AXIOM_CERTIFICATE_HASH: $(expected_hash)
        -- AXIOM_PROOF_STATUS: proven
        -- AXIOM_PROOF_METHOD: pattern
        -- AXIOM_OBLIGATION_ID: $(expected_id)

        theorem ci_bundle_complete : True := by
          trivial
        """,
    )

    write(
        bundle["assistants"]["coq"],
        """
        (* AXIOM_CERTIFICATE_HASH: $(expected_hash) *)
        (* AXIOM_PROOF_STATUS: proven *)
        (* AXIOM_PROOF_METHOD: pattern *)
        (* AXIOM_OBLIGATION_ID: $(expected_id) *)

        Theorem ci_bundle_complete : True.
        Proof.
          exact I.
        Qed.
        """,
    )

    write(
        bundle["assistants"]["isabelle"],
        """
        theory CIBundle
          imports Main
        begin

        (* AXIOM_CERTIFICATE_HASH: $(expected_hash) *)
        (* AXIOM_PROOF_STATUS: proven *)
        (* AXIOM_PROOF_METHOD: pattern *)
        (* AXIOM_OBLIGATION_ID: $(expected_id) *)

        theorem ci_bundle_complete: "True"
          by simp

        end
        """,
    )

    reconciled = reconcile_proof_bundle(bundle["manifest"])
    reports = reconciled["obligations"][1]["assistant_reports"]

    @test reconciled["obligations"][1]["status"] == "assistant_completed"
    @test reconciled["obligations"][1]["assistant_unresolved_total"] == 0
    @test reports["lean"]["status"] == "complete"
    @test reports["coq"]["status"] == "complete"
    @test reports["isabelle"]["status"] == "complete"
end
