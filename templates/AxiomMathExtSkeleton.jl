# SPDX-License-Identifier: PMPL-1.0-or-later
# Template skeleton for a Math backend extension module.
#
# Copy this into your Math integration package and replace CPU fallbacks with
# real Math kernel calls.

module AxiomMathExtSkeleton

using Axiom

# Minimal hook set for Dense + ReLU + Softmax pipelines.
function Axiom.backend_coprocessor_matmul(
    backend::Axiom.MathBackend,
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    # Replace with real Math matmul kernel dispatch.
    A * B
end

function Axiom.backend_coprocessor_relu(
    backend::Axiom.MathBackend,
    x::AbstractArray{Float32},
)
    # Replace with real Math elementwise activation kernel.
    max.(x, 0f0)
end

function Axiom.backend_coprocessor_softmax(
    backend::Axiom.MathBackend,
    x::AbstractArray{Float32},
    dim::Int,
)
    # Replace with real Math softmax kernel.
    Axiom.softmax(x, dims = dim)
end

# Optional: implement additional hooks as kernel coverage expands.
# - backend_coprocessor_conv2d
# - backend_coprocessor_batchnorm
# - backend_coprocessor_layernorm
# - backend_coprocessor_maxpool2d
# - backend_coprocessor_avgpool2d
# - backend_coprocessor_global_avgpool2d

end # module AxiomMathExtSkeleton
