# SPDX-License-Identifier: MPL-2.0
# Offline HuggingFace-integration tests (G06).
#
# Exercises the network-free core of `Axiom.HuggingFaceCompat`: architecture
# detection, model building from a config Dict, SafeTensors dtype mapping, and
# the pure verification report. The hub-fetch path
# (`from_pretrained`/`get_model_info`/`download_file`) needs network and is
# intentionally NOT exercised here — documented in the module header.

using Test
using Axiom

const HF = Axiom.HuggingFaceCompat

@testset "HuggingFace integration (offline)" begin

    @testset "detect_architecture" begin
        @test HF.detect_architecture(Dict("model_type" => "bert")) == "bert"
        @test HF.detect_architecture(Dict("architectures" => ["GPT2LMHeadModel"])) == "GPT2LMHeadModel"
        @test HF.detect_architecture(Dict("num_hidden_layers" => 12,
                                          "num_attention_heads" => 12)) == "transformer"
        @test HF.detect_architecture(Dict{String,Any}()) == "unknown"
    end

    @testset "build_model_from_config" begin
        bert_cfg = Dict("hidden_size" => 128, "num_hidden_layers" => 2,
                        "num_attention_heads" => 4, "intermediate_size" => 256,
                        "vocab_size" => 1000)
        bert = HF.build_model_from_config(bert_cfg, "bert")
        @test bert isa Axiom.Pipeline
        # 1 embedding + 2*(2 LayerNorm + 4 Dense + 2 Dense) + 1 pooler = 18
        @test length(bert.layers) == 18

        gpt2_cfg = Dict("n_embd" => 64, "n_layer" => 2, "vocab_size" => 500,
                        "n_positions" => 128)
        gpt2 = HF.build_model_from_config(gpt2_cfg, "gpt2")
        @test gpt2 isa Axiom.Pipeline
        # 1 embedding + 2*(2 LayerNorm + 3 Dense) + final LayerNorm + head = 13
        @test length(gpt2.layers) == 13

        # Unknown architecture falls back to the generic (BERT-shaped) builder.
        gen = HF.build_model_from_config(bert_cfg, "totally-unknown-arch")
        @test gen isa Axiom.Pipeline
    end

    @testset "SafeTensors dtype mapping" begin
        @test HF._safetensors_dtype("F32") === Float32
        @test HF._safetensors_dtype("F16") === Float16
        @test HF._safetensors_dtype("F64") === Float64
        @test HF._safetensors_dtype("I32") === Int32
        @test HF._safetensors_dtype("I64") === Int64
        @test HF._safetensors_dtype("BF16") === nothing   # BFloat16 not native in Julia
        @test HF._safetensors_dtype("NOPE") === nothing   # unsupported dtype
    end

    @testset "verify_imported_model (pure, no network)" begin
        bert = HF.build_model_from_config(
            Dict("hidden_size" => 32, "num_hidden_layers" => 1,
                 "num_attention_heads" => 2, "intermediate_size" => 64,
                 "vocab_size" => 100), "bert")
        report = HF.verify_imported_model(bert, "bert")
        @test haskey(report, "passed")
        @test haskey(report, "warnings")
        @test haskey(report, "failures")
        @test isempty(report["failures"])
        # Secure default: remote-code execution is reported disabled unless the
        # AXIOM_HF_TRUST_REMOTE_CODE env opt-in is set.
        @test any(p -> occursin("Remote code execution disabled", p), report["passed"])
    end

    @testset "public API surface exists" begin
        @test isdefined(HF, :from_pretrained)
        @test isdefined(HF, :load_tokenizer)
    end
end
