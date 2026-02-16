# SPDX-License-Identifier: PMPL-1.0-or-later

using Test
using JSON
using Axiom
using Axiom: ProofResult

@testset "Proof certificate integrity" begin
    result = ProofResult(:proven, nothing, 1.0, "CI integrity check", String[])
    cert_dict = serialize_proof(
        result,
        "∀x. relu(x) >= 0";
        proof_method = "ci",
        execution_time_ms = 1.0
    )

    tmp_dir = mktempdir()
    cert_a = joinpath(tmp_dir, "cert-a.proof")
    cert_b = joinpath(tmp_dir, "cert-b.proof")

    export_proof_certificate(cert_dict, cert_a)
    export_proof_certificate(cert_dict, cert_b)

    # Deterministic export: same input dict must yield identical bytes.
    @test read(cert_a, String) == read(cert_b, String)

    imported = import_proof_certificate(cert_a)
    @test verify_proof_certificate(imported)

    tampered_data = JSON.parsefile(cert_a)
    tampered_data["property"] = "∀x. relu(x) > 100"
    tampered_path = joinpath(tmp_dir, "tampered.proof")
    open(tampered_path, "w") do io
        JSON.print(io, tampered_data, 2)
    end

    tampered_cert = import_proof_certificate(tampered_path)
    @test !verify_proof_certificate(tampered_cert)
end

@testset "Model certificate integrity" begin
    model = Sequential(Dense(10, 5, relu), Dense(5, 3), Softmax())
    x = Tensor(randn(Float32, 2, 10))
    result = verify(model, properties = [ValidProbabilities(), FiniteOutput(), NoNaN()], data = [(x, nothing)])
    @test result.passed

    cert = generate_certificate(model, result, model_name = "ci-certificate")
    cert_path = tempname() * ".cert"
    save_certificate(cert, cert_path)

    loaded = load_certificate(cert_path)
    @test verify_certificate(loaded)

    content = read(cert_path, String)
    tampered_content = replace(content, r"signature:\s*[0-9a-f]+" => "signature: deadbeef")
    tampered_path = tempname() * ".cert"
    write(tampered_path, tampered_content)

    tampered = load_certificate(tampered_path)
    @test !verify_certificate(tampered)

    rm(cert_path)
    rm(tampered_path)
end
