# SPDX-License-Identifier: PMPL-1.0-or-later
# Template skeleton for a DSP backend extension module.
#
# Copy this into your DSP integration package and replace CPU fallbacks with
# real DSP kernel calls.

module AxiomDSPExtSkeleton

using Axiom

# Minimal hook set for Dense + ReLU + Softmax pipelines.
function Axiom.backend_coprocessor_matmul(
    backend::Axiom.DSPBackend,
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    # Replace with real DSP matmul kernel dispatch.
    A * B
end

function Axiom.backend_coprocessor_relu(
    backend::Axiom.DSPBackend,
    x::AbstractArray{Float32},
)
    # Replace with real DSP elementwise activation kernel.
    max.(x, 0f0)
end

function Axiom.backend_coprocessor_softmax(
    backend::Axiom.DSPBackend,
    x::AbstractArray{Float32},
    dim::Int,
)
    # Replace with real DSP softmax kernel.
    Axiom.softmax(x, dims = dim)
end

# Optional: implement additional hooks as kernel coverage expands.
# - backend_coprocessor_conv2d
# - backend_coprocessor_batchnorm
# - backend_coprocessor_layernorm
# - backend_coprocessor_maxpool2d
# - backend_coprocessor_avgpool2d
# - backend_coprocessor_global_avgpool2d

end # module AxiomDSPExtSkeleton
