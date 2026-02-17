# SPDX-License-Identifier: PMPL-1.0-or-later
# Template skeleton for a Crypto backend extension module.
#
# Copy this into your Crypto integration package and replace CPU fallbacks with
# real Crypto kernel calls.

module AxiomCryptoExtSkeleton

using Axiom

# Minimal hook set for Dense + ReLU + Softmax pipelines.
function Axiom.backend_coprocessor_matmul(
    backend::Axiom.CryptoBackend,
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    # Replace with real Crypto matmul kernel dispatch.
    A * B
end

function Axiom.backend_coprocessor_relu(
    backend::Axiom.CryptoBackend,
    x::AbstractArray{Float32},
)
    # Replace with real Crypto elementwise activation kernel.
    max.(x, 0f0)
end

function Axiom.backend_coprocessor_softmax(
    backend::Axiom.CryptoBackend,
    x::AbstractArray{Float32},
    dim::Int,
)
    # Replace with real Crypto softmax kernel.
    Axiom.softmax(x, dims = dim)
end

# Optional: implement additional hooks as kernel coverage expands.
# - backend_coprocessor_conv2d
# - backend_coprocessor_batchnorm
# - backend_coprocessor_layernorm
# - backend_coprocessor_maxpool2d
# - backend_coprocessor_avgpool2d
# - backend_coprocessor_global_avgpool2d

end # module AxiomCryptoExtSkeleton
