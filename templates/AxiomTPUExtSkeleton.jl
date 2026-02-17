# SPDX-License-Identifier: PMPL-1.0-or-later
# Template skeleton for a TPU backend extension module.
#
# Copy this into your TPU integration package and replace CPU fallbacks with
# real TPU kernel calls.

module AxiomTPUExtSkeleton

using Axiom

# Minimal hook set for Dense + ReLU + Softmax pipelines.
function Axiom.backend_coprocessor_matmul(
    backend::Axiom.TPUBackend,
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    # Replace with real TPU matmul kernel dispatch.
    A * B
end

function Axiom.backend_coprocessor_relu(
    backend::Axiom.TPUBackend,
    x::AbstractArray{Float32},
)
    # Replace with real TPU elementwise activation kernel.
    max.(x, 0f0)
end

function Axiom.backend_coprocessor_softmax(
    backend::Axiom.TPUBackend,
    x::AbstractArray{Float32},
    dim::Int,
)
    # Replace with real TPU softmax kernel.
    Axiom.softmax(x, dims = dim)
end

# Optional: implement additional hooks as kernel coverage expands.
# - backend_coprocessor_conv2d
# - backend_coprocessor_batchnorm
# - backend_coprocessor_layernorm
# - backend_coprocessor_maxpool2d
# - backend_coprocessor_avgpool2d
# - backend_coprocessor_global_avgpool2d

end # module AxiomTPUExtSkeleton
