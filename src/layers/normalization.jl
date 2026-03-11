# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Normalization Layers
#
# BatchNorm, LayerNorm, InstanceNorm, GroupNorm

"""
    BatchNorm(num_features; momentum=0.1, eps=1e-5, affine=true, track_running_stats=true)

Batch Normalization layer.

# Arguments
- `num_features`: Number of features/channels
- `momentum`: Momentum for running statistics
- `eps`: Small constant for numerical stability
- `affine`: Whether to include learnable affine parameters
- `track_running_stats`: Whether to track running mean/variance

# Reference
Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training"
"""
mutable struct BatchNorm{T} <: AbstractLayer
    γ::Union{Vector{T}, Nothing}  # Scale
    β::Union{Vector{T}, Nothing}  # Shift
    running_mean::Vector{T}
    running_var::Vector{T}
    momentum::Float32
    eps::Float32
    affine::Bool
    track_running_stats::Bool
    training::Bool
    num_features::Int
end

function BatchNorm(
    num_features::Int;
    momentum = 0.1f0,
    eps = Float32(1e-5),
    affine::Bool = true,
    track_running_stats::Bool = true,
    dtype::Type{T} = Float32
) where T
    γ = affine ? ones(T, num_features) : nothing
    β = affine ? zeros(T, num_features) : nothing
    running_mean = zeros(T, num_features)
    running_var = ones(T, num_features)

    BatchNorm{T}(γ, β, running_mean, running_var, momentum, eps,
                 affine, track_running_stats, true, num_features)
end

# forward(bn::BatchNorm, x::AbstractTensor) is defined in backends/abstract.jl
# with backend-aware dispatch (routes through Zig/GPU when active).

function parameters(bn::BatchNorm)
    if bn.affine
        (γ = bn.γ, β = bn.β)
    else
        NamedTuple()
    end
end

output_shape(::BatchNorm, input_shape) = input_shape

function show_layer_params(io::IO, bn::BatchNorm)
    print(io, bn.num_features)
    if !bn.affine
        print(io, ", affine=false")
    end
end

"""
    LayerNorm(normalized_shape; eps=1e-5, elementwise_affine=true)

Layer Normalization layer.

# Arguments
- `normalized_shape`: Shape of the input to normalize (without batch dimension)
- `eps`: Small constant for numerical stability
- `elementwise_affine`: Whether to include learnable affine parameters

# Reference
Ba et al., "Layer Normalization"
"""
mutable struct LayerNorm{T, S} <: AbstractLayer
    γ::Union{Array{T}, Nothing}
    β::Union{Array{T}, Nothing}
    normalized_shape::S
    eps::Float32
    elementwise_affine::Bool
end

function LayerNorm(
    normalized_shape;
    eps::Float32 = Float32(1e-5),
    elementwise_affine::Bool = true,
    dtype::Type{T} = Float32
) where T
    shape = normalized_shape isa Int ? (normalized_shape,) : Tuple(normalized_shape)

    γ = elementwise_affine ? ones(T, shape...) : nothing
    β = elementwise_affine ? zeros(T, shape...) : nothing

    LayerNorm{T, typeof(shape)}(γ, β, shape, eps, elementwise_affine)
end

function forward(ln::LayerNorm, x::AbstractTensor)
    backend = current_backend()
    x_data = x.data

    # Try native backend dispatch for 2D inputs with affine params
    if !(backend isa JuliaBackend) && ln.elementwise_affine && ndims(x_data) == 2
        try
            gamma = Float32.(vec(ln.γ))
            beta = Float32.(vec(ln.β))
            if backend isa SmartBackend
                y = backend_layernorm(backend, Float32.(x_data), gamma, beta, ln.normalized_shape, Float32(ln.eps))
            else
                y = backend_layernorm(backend, Float32.(x_data), gamma, beta, Float32(ln.eps))
            end
            return Tensor(y)
        catch
            # Fall through to Julia implementation
        end
    end

    # Julia implementation (reference, handles arbitrary dimensions)
    n_dims = length(ln.normalized_shape)
    norm_dims = collect(ndims(x_data)-n_dims+1:ndims(x_data))

    μ = mean(x_data, dims=norm_dims)
    σ² = var(x_data, dims=norm_dims, corrected=false)

    x_norm = (x_data .- μ) ./ sqrt.(σ² .+ ln.eps)

    if ln.elementwise_affine
        leading_ones = ntuple(_ -> 1, ndims(x_data) - n_dims)
        reshape_size = (leading_ones..., ln.normalized_shape...)
        γ_reshaped = reshape(ln.γ, reshape_size)
        β_reshaped = reshape(ln.β, reshape_size)
        x_norm = γ_reshaped .* x_norm .+ β_reshaped
    end

    Tensor(x_norm)
end

function parameters(ln::LayerNorm)
    if ln.elementwise_affine
        (γ = ln.γ, β = ln.β)
    else
        NamedTuple()
    end
end

output_shape(::LayerNorm, input_shape) = input_shape

"""
    InstanceNorm(num_features; eps=1e-5, affine=false)

Instance Normalization layer.
Normalizes each sample independently across spatial dimensions.

# Reference
Ulyanov et al., "Instance Normalization: The Missing Ingredient for Fast Stylization"
"""
mutable struct InstanceNorm{T} <: AbstractLayer
    γ::Union{Vector{T}, Nothing}
    β::Union{Vector{T}, Nothing}
    eps::Float32
    affine::Bool
    num_features::Int
end

function InstanceNorm(
    num_features::Int;
    eps::Float32 = Float32(1e-5),
    affine::Bool = false,
    dtype::Type{T} = Float32
) where T
    γ = affine ? ones(T, num_features) : nothing
    β = affine ? zeros(T, num_features) : nothing

    InstanceNorm{T}(γ, β, eps, affine, num_features)
end

function forward(inst::InstanceNorm, x::AbstractTensor)
    # x shape: (N, H, W, C) or similar
    # Normalize over spatial dimensions (not batch or channel)
    x_data = x.data
    spatial_dims = collect(2:ndims(x_data)-1)

    μ = mean(x_data, dims=spatial_dims)
    σ² = var(x_data, dims=spatial_dims, corrected=false)

    x_norm = (x_data .- μ) ./ sqrt.(σ² .+ inst.eps)

    if inst.affine
        γ = reshape(inst.γ, ones(Int, ndims(x_data)-1)..., :)
        β = reshape(inst.β, ones(Int, ndims(x_data)-1)..., :)
        x_norm = γ .* x_norm .+ β
    end

    Tensor(x_norm)
end

function parameters(inst::InstanceNorm)
    if inst.affine
        (γ = inst.γ, β = inst.β)
    else
        NamedTuple()
    end
end

output_shape(::InstanceNorm, input_shape) = input_shape

"""
    GroupNorm(num_groups, num_channels; eps=1e-5, affine=true)

Group Normalization layer.
Divides channels into groups and normalizes within each group.

# Reference
Wu & He, "Group Normalization"
"""
mutable struct GroupNorm{T} <: AbstractLayer
    γ::Union{Vector{T}, Nothing}
    β::Union{Vector{T}, Nothing}
    num_groups::Int
    num_channels::Int
    eps::Float32
    affine::Bool
end

function GroupNorm(
    num_groups::Int,
    num_channels::Int;
    eps::Float32 = Float32(1e-5),
    affine::Bool = true,
    dtype::Type{T} = Float32
) where T
    @assert num_channels % num_groups == 0 "num_channels must be divisible by num_groups"

    γ = affine ? ones(T, num_channels) : nothing
    β = affine ? zeros(T, num_channels) : nothing

    GroupNorm{T}(γ, β, num_groups, num_channels, eps, affine)
end

function forward(gn::GroupNorm, x::AbstractTensor)
    # Simplified implementation
    # x shape: (N, ..., C)
    x_data = x.data
    N = size(x_data, 1)
    C = size(x_data, ndims(x_data))
    group_size = C ÷ gn.num_groups

    # Reshape to separate groups
    original_shape = size(x_data)
    spatial = original_shape[2:end-1]

    # Normalize per group
    y = similar(x_data)
    for g in 1:gn.num_groups
        c_start = (g - 1) * group_size + 1
        c_end = g * group_size

        group_data = selectdim(x_data, ndims(x_data), c_start:c_end)

        μ = mean(group_data)
        σ² = var(group_data, corrected=false)

        normalized = (group_data .- μ) ./ sqrt(σ² + gn.eps)

        # This is a simplified version - proper implementation would be more efficient
        for (i, c) in enumerate(c_start:c_end)
            selectdim(y, ndims(y), c:c) .= selectdim(normalized, ndims(normalized), i:i)
        end
    end

    if gn.affine
        γ = reshape(gn.γ, ones(Int, ndims(x_data)-1)..., :)
        β = reshape(gn.β, ones(Int, ndims(x_data)-1)..., :)
        y = γ .* y .+ β
    end

    Tensor(y)
end

function parameters(gn::GroupNorm)
    if gn.affine
        (γ = gn.γ, β = gn.β)
    else
        NamedTuple()
    end
end

output_shape(::GroupNorm, input_shape) = input_shape

"""
    RMSNorm(dim; eps=1e-6)

Root Mean Square Normalization (used in LLaMA and other models).
"""
mutable struct RMSNorm{T} <: AbstractLayer
    weight::Vector{T}
    eps::Float32
end

function RMSNorm(dim::Int; eps::Float32=Float32(1e-6), dtype::Type{T}=Float32) where T
    RMSNorm{T}(ones(T, dim), eps)
end

function forward(rn::RMSNorm, x::AbstractTensor)
    backend = current_backend()
    x_data = x.data

    # Try native backend dispatch for 2D inputs
    if !(backend isa JuliaBackend) && ndims(x_data) == 2
        try
            weight = Float32.(rn.weight)
            y = backend_rmsnorm(backend, Float32.(x_data), weight, Float32(rn.eps))
            return Tensor(y)
        catch
            # Fall through to Julia implementation
        end
    end

    # Julia implementation (reference)
    rms = sqrt.(mean(x_data .^ 2, dims=ndims(x_data)) .+ rn.eps)
    x_norm = x_data ./ rms
    Tensor(x_norm .* rn.weight')
end

parameters(rn::RMSNorm) = (weight = rn.weight,)
output_shape(::RMSNorm, input_shape) = input_shape
