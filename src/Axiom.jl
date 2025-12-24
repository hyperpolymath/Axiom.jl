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

# Backend abstraction
include("backends/abstract.jl")
include("backends/julia_backend.jl")

# Rust FFI (loaded conditionally)
include("backends/rust_ffi.jl")

# Utilities
include("utils/initialization.jl")
include("utils/data.jl")

# Re-exports for user convenience
export @axiom, @ensure, @prove
export Tensor, Shape, DynamicShape
export Dense, Conv, Conv2D, BatchNorm, LayerNorm, Dropout
export ReLU, Sigmoid, Tanh, Softmax, GELU, LeakyReLU
export MaxPool, AvgPool, GlobalAvgPool, Flatten
export Sequential, Chain, Residual
export Adam, SGD, RMSprop, AdamW
export mse_loss, crossentropy, binary_crossentropy
export train!, compile, verify
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
