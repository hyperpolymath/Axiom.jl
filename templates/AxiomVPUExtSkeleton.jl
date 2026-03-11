# SPDX-License-Identifier: PMPL-1.0-or-later
# Template skeleton for a VPU (Vision Processing Unit) backend extension module.
#
# Copy this into your VPU integration package and replace CPU fallbacks with
# real VPU kernel calls (e.g., Intel Movidius, Hailo).

module AxiomVPUExtSkeleton

using Axiom

# Minimal hook set for Dense + ReLU + Softmax pipelines.
function Axiom.backend_coprocessor_matmul(
    backend::Axiom.VPUBackend,
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    # Replace with real VPU matmul kernel dispatch.
    A * B
end

function Axiom.backend_coprocessor_relu(
    backend::Axiom.VPUBackend,
    x::AbstractArray{Float32},
)
    # Replace with real VPU elementwise activation kernel.
    max.(x, 0f0)
end

function Axiom.backend_coprocessor_softmax(
    backend::Axiom.VPUBackend,
    x::AbstractArray{Float32},
    dim::Int,
)
    # Replace with real VPU softmax kernel.
    Axiom.softmax(x, dims = dim)
end

# Optional: implement additional hooks as kernel coverage expands.
# - backend_coprocessor_conv2d
# - backend_coprocessor_batchnorm
# - backend_coprocessor_layernorm
# - backend_coprocessor_maxpool2d
# - backend_coprocessor_avgpool2d
# - backend_coprocessor_global_avgpool2d

end # module AxiomVPUExtSkeleton
