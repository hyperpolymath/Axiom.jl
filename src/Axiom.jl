# SPDX-FileCopyrightText: 2025 Axiom.jl Contributors
# SPDX-License-Identifier: PMPL-1.0-or-later

"""
    Axiom

Axiom.jl: Provably Correct Machine Learning

Axiom.jl is a cutting-edge Julia framework designed to bridge the gap between
high-performance machine learning and rigorous formal verification. In an era
where AI systems are deployed in safety-critical and high-stakes applications,
ensuring their correctness, reliability, and trustworthiness is paramount.
Axiom.jl addresses this challenge by integrating formal methods directly into
the ML development lifecycle.

Vision & Philosophy:
--------------------
The core philosophy of Axiom.jl is "Provably Correct Machine Learning."
This means moving beyond empirical testing and statistical guarantees to
provide mathematical assurances about the behavior and properties of ML models.
By leveraging techniques from formal verification and dependent type theory,
Axiom.jl enables developers to build AI systems with a higher degree of
confidence and accountability.

Key Features:
-------------
-   **Compile-time Shape Verification**: Catch tensor dimension mismatches and
    other structural errors at compile-time, significantly reducing runtime
    bugs and improving development efficiency.
-   **Formal Property Guarantees**: Define and verify formal properties of
    ML models (e.g., robustness, safety invariants, fairness criteria) using
    specialized DSLs and integration with proof assistants.
-   **Multi-Backend Support**: Seamlessly integrate with high-performance
    Rust backend via Foreign Function Interface (FFI),
    allowing for optimized computation without sacrificing verification
    capabilities. GPU acceleration available through CUDA/ROCm/Metal extensions.
-   **Model Interoperability**: Facilitate the import and export of models
    from popular frameworks like PyTorch and ONNX, enabling formal
    verification workflows for existing models.
-   **Comprehensive ML Toolkit**: Provides a full suite of ML functionalities,
    including a flexible tensor type system, a rich library of neural network
    layers, automatic differentiation, various optimizers and loss functions,
    and robust training infrastructure.
-   **Verification Ecosystem**: Offers tools for property specification, checker
    integration, certificate generation, and serialization, forming a complete
    workflow for provable ML.

Motivation & Impact:
--------------------
Traditional ML development relies heavily on extensive testing, which can only
demonstrate the presence of bugs, not their absence. Formal methods, in contrast,
can prove the absence of certain classes of errors. Axiom.jl aims to empower
researchers and engineers to build ML models that are not only powerful but also
demonstrably correct, opening new avenues for trustworthy AI in domains such
as autonomous systems, medical diagnostics, and financial modeling.

This module serves as the main entry point for the Axiom.jl framework,
orchestrating the loading of various components and providing key exports
for user convenience.
"""
module Axiom


using LinearAlgebra
using Random
using Statistics
using Dates
using HTTP
using SHA
using JSON

# Core type system
include("types/tensor.jl")
include("types/shapes.jl")

# Layer abstractions
include("layers/abstract.jl")
include("layers/dense.jl")
include("layers/conv.jl")
include("layers/activations.jl")
include("layers/normalization.jl")
include("layers/pooling.jl")

# DSL macros
include("dsl/axiom_macro.jl")
include("dsl/ensure.jl")
# `@prove` degrades gracefully when SMT extension is unavailable.
include("dsl/prove.jl")
include("dsl/pipeline.jl")

# Automatic differentiation
include("autograd/gradient.jl")
include("autograd/tape.jl")

# Training infrastructure
include("training/optimizers.jl")
include("training/loss.jl")
include("training/train.jl")

# Verification system
include("verification/properties.jl")
include("verification/checker.jl")
include("verification/certificates.jl")
include("verification/serialization.jl")
include("proof_export.jl")  # Issue #19 - Proof assistant integration

# Backend abstraction
include("backends/abstract.jl")
include("backends/julia_backend.jl")
include("backends/gpu_hooks.jl")  # GPU backend interface (issue #12)

# Rust FFI (loaded conditionally)
include("backends/rust_ffi.jl")

# Model metadata and packaging
include("model_metadata.jl")

# Utilities
include("utils/initialization.jl")
include("utils/data.jl")

# API serving
include("serving/api.jl")

# Integrations
include("integrations/interop.jl")

# Re-exports for user convenience
export @axiom, @ensure

# Tensor types and creation
export Tensor, DynamicTensor, Shape, DynamicShape
export axiom_zeros, axiom_ones, axiom_randn
export zeros_like, ones_like, randn_like
export to_dynamic, to_static

# Layers
export Dense, Conv, Conv2d, Conv2D, BatchNorm, LayerNorm, Dropout
export MaxPool2d, AvgPool2d, GlobalAvgPool, Flatten

# Activation functions (lowercase)
export relu, sigmoid, tanh, softmax, gelu, leaky_relu

# Activation layers (capitalized)
export ReLU, Sigmoid, Tanh, Softmax, GELU, LeakyReLU

# Model containers
export Sequential, Chain, Residual

# Optimizers
export Adam, SGD, RMSprop, AdamW

# Loss functions
export mse_loss, crossentropy, binary_crossentropy

# Training
export train!, compile, verify

# Backends
export AbstractBackend, JuliaBackend, RustBackend
export CUDABackend, ROCmBackend, MetalBackend
export TPUBackend, NPUBackend, DSPBackend, FPGABackend
export current_backend, set_backend!, @with_backend
export detect_gpu, detect_coprocessor, detect_accelerator
export cuda_available, rocm_available, metal_available
export cuda_device_count, rocm_device_count, metal_device_count
export tpu_available, npu_available, dsp_available, fpga_available
export tpu_device_count, npu_device_count, dsp_device_count, fpga_device_count
export select_device!, gpu_capability_report, coprocessor_capability_report
export gpu_runtime_diagnostics, reset_gpu_runtime_diagnostics!

# Data utilities
export DataLoader, train_test_split, one_hot
export make_moons, make_blobs

# API serving
export serve_rest, serve_graphql, graphql_execute
export serve_grpc, generate_grpc_proto, grpc_predict, grpc_health, grpc_support_status

# Verification
export ValidProbabilities, FiniteOutput, NoNaN, NoInf, check
export EnsureViolation
export ProofCertificate, serialize_proof, deserialize_proof
export export_proof_certificate, import_proof_certificate, verify_proof_certificate
export VerificationResult, generate_certificate, save_certificate, load_certificate, verify_certificate

# Interop
export from_pytorch, to_onnx
export load_pytorch, export_onnx

# Model metadata and packaging (issues #15, #16)
export ModelMetadata, VerificationClaim
export create_metadata, save_metadata, load_metadata, validate_metadata
export add_verification_claim!, verify_and_claim!

# Proof assistant integration (issue #19)
export export_lean, export_coq, export_isabelle
export proof_obligation_manifest, export_proof_bundle
export proof_assistant_obligation_report, reconcile_proof_bundle
export import_lean_certificate, import_coq_certificate, import_isabelle_certificate

# Version info
const VERSION = v"1.0.0"

function __init__()
    # Check for Rust backend availability
    if haskey(ENV, "AXIOM_RUST_LIB")
        try
            init_rust_backend(ENV["AXIOM_RUST_LIB"])
        catch e
            @warn "Rust backend not available, using pure Julia" exception=e
        end
    end
end

end # module Axiom
