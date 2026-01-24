# SPDX-FileCopyrightText: 2025 Axiom.jl Contributors
# SPDX-License-Identifier: MIT

# Axiom.jl - Provably Correct Machine Learning
#
# A Julia ML framework with:
# - Compile-time shape verification
# - Formal property guarantees
# - Rust/Zig performance backends
# - PyTorch model import

module Axiom

using LinearAlgebra
using Random
using Statistics
using Dates
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
# TODO: prove.jl requires SMTLib (weak dependency) - move to extension
# include("dsl/prove.jl")
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

# Backend abstraction
include("backends/abstract.jl")
include("backends/julia_backend.jl")
include("backends/gpu_hooks.jl")  # GPU backend interface (issue #12)

# Rust FFI (loaded conditionally)
include("backends/rust_ffi.jl")

# Utilities
include("utils/initialization.jl")
include("utils/data.jl")

# Integrations
include("integrations/huggingface.jl")

# Re-exports for user convenience
export @axiom, @ensure  # TODO: @prove requires SMTLib extension

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

# Data utilities
export DataLoader, train_test_split, one_hot

# Verification
export ValidProbabilities, FiniteOutput, check
export EnsureViolation
export ProofCertificate, serialize_proof, deserialize_proof
export export_proof_certificate, import_proof_certificate, verify_proof_certificate

# Interop
export from_pytorch, to_onnx

# Version info
const VERSION = v"0.1.0"

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
