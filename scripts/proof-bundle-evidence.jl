#!/usr/bin/env julia
# SPDX-License-Identifier: PMPL-1.0-or-later

using JSON
using Dates
using Axiom
using Axiom: ProofResult

function sample_certificate()
    result = ProofResult(
        :proven,
        nothing,
        1.0,
        "Proof-bundle evidence generation",
        String[],
    )

    cert_dict = serialize_proof(
        result,
        "∀x. sum(softmax(x)) == 1.0";
        proof_method = "pattern",
        execution_time_ms = 0.3,
    )

    deserialize_proof(cert_dict)
end

function main()
    cert = sample_certificate()
    bundle_dir = mktempdir()
    bundle = export_proof_bundle(cert, bundle_dir; base_name = "proof_bundle_evidence")
    manifest = JSON.parsefile(bundle["manifest"])

    expected_hash = cert.hash
    expected_id = manifest["obligations"][1]["id"]

    initial_reports = Dict{String, Any}()
    for (assistant_name, path) in bundle["assistants"]
        initial_reports[assistant_name] = proof_assistant_obligation_report(
            path,
            Symbol(assistant_name);
            expected_certificate_hash = expected_hash,
            expected_obligation_id = expected_id,
        )
    end

    initial = reconcile_proof_bundle(bundle["manifest"]; persist = false)

    write(
        bundle["assistants"]["lean"],
        """
        -- AXIOM_CERTIFICATE_HASH: $(expected_hash)
        -- AXIOM_PROOF_STATUS: proven
        -- AXIOM_PROOF_METHOD: pattern
        -- AXIOM_OBLIGATION_ID: $(expected_id)

        theorem proof_bundle_complete : True := by
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

        Theorem proof_bundle_complete : True.
        Proof.
          exact I.
        Qed.
        """,
    )

    write(
        bundle["assistants"]["isabelle"],
        """
        theory ProofBundleEvidence
          imports Main
        begin

        (* AXIOM_CERTIFICATE_HASH: $(expected_hash) *)
        (* AXIOM_PROOF_STATUS: proven *)
        (* AXIOM_PROOF_METHOD: pattern *)
        (* AXIOM_OBLIGATION_ID: $(expected_id) *)

        theorem proof_bundle_complete: "True"
          by simp

        end
        """,
    )

    reconciled = reconcile_proof_bundle(bundle["manifest"]; persist = false)

    payload = Dict(
        "format" => "axiom-proof-bundle-evidence.v1",
        "generated_at" => Dates.format(now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "manifest_path" => bundle["manifest"],
        "initial_obligation_status" => initial["obligations"][1]["status"],
        "final_obligation_status" => reconciled["obligations"][1]["status"],
        "initial_reports" => initial_reports,
        "final_reports" => reconciled["obligations"][1]["assistant_reports"],
        "initial_unresolved_total" => initial["obligations"][1]["assistant_unresolved_total"],
        "final_unresolved_total" => reconciled["obligations"][1]["assistant_unresolved_total"],
    )

    out_path = get(ENV, "AXIOM_PROOF_EVIDENCE_PATH", joinpath(pwd(), "build", "proof_bundle_evidence.json"))
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        JSON.print(io, payload, 2)
    end

    println("proof bundle evidence written: $out_path")

    if payload["initial_obligation_status"] != "interactive_required"
        error("Unexpected initial obligation status: $(payload["initial_obligation_status"])\n")
    end

    if payload["final_obligation_status"] != "assistant_completed"
        error("Unexpected final obligation status: $(payload["final_obligation_status"])\n")
    end

    if payload["final_unresolved_total"] != 0
        error("Final unresolved obligation count must be zero\n")
    end
end

main()
