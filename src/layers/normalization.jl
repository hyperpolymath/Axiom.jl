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
    momentum::Float32 = 0.1f0,
    eps::Float32 = 1e-5f0,
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

function forward(bn::BatchNorm, x::AbstractArray)
    # x shape: (N, ..., C) where C is the feature dimension

    if bn.training
        # Use batch statistics
        dims = collect(1:ndims(x)-1)  # All dims except channels
        μ = mean(x, dims=dims)
        σ² = var(x, dims=dims, corrected=false)

        # Update running statistics
        if bn.track_running_stats
            bn.running_mean .= (1 - bn.momentum) .* bn.running_mean .+ bn.momentum .* vec(μ)
            bn.running_var .= (1 - bn.momentum) .* bn.running_var .+ bn.momentum .* vec(σ²)
        end
    else
        # Use running statistics
        μ = reshape(bn.running_mean, ones(Int, ndims(x)-1)..., :)
        σ² = reshape(bn.running_var, ones(Int, ndims(x)-1)..., :)
    end

    # Normalize
    x_norm = (x .- μ) ./ sqrt.(σ² .+ bn.eps)

    # Scale and shift
    if bn.affine
        γ = reshape(bn.γ, ones(Int, ndims(x)-1)..., :)
        β = reshape(bn.β, ones(Int, ndims(x)-1)..., :)
        x_norm = γ .* x_norm .+ β
    end

    x_norm
end

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
    eps::Float32 = 1e-5f0,
    elementwise_affine::Bool = true,
    dtype::Type{T} = Float32
) where T
    shape = normalized_shape isa Int ? (normalized_shape,) : Tuple(normalized_shape)

    γ = elementwise_affine ? ones(T, shape...) : nothing
    β = elementwise_affine ? zeros(T, shape...) : nothing

    LayerNorm{T, typeof(shape)}(γ, β, shape, eps, elementwise_affine)
end

function forward(ln::LayerNorm, x::AbstractArray)
    # Normalize over the last n dimensions matching normalized_shape
    n_dims = length(ln.normalized_shape)
    norm_dims = collect(ndims(x)-n_dims+1:ndims(x))

    μ = mean(x, dims=norm_dims)
    σ² = var(x, dims=norm_dims, corrected=false)

    x_norm = (x .- μ) ./ sqrt.(σ² .+ ln.eps)

    if ln.elementwise_affine
        x_norm = ln.γ .* x_norm .+ ln.β
    end

    x_norm
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
    eps::Float32 = 1e-5f0,
    affine::Bool = false,
    dtype::Type{T} = Float32
) where T
    γ = affine ? ones(T, num_features) : nothing
    β = affine ? zeros(T, num_features) : nothing

    InstanceNorm{T}(γ, β, eps, affine, num_features)
end

function forward(inst::InstanceNorm, x::AbstractArray)
    # x shape: (N, H, W, C) or similar
    # Normalize over spatial dimensions (not batch or channel)
    spatial_dims = collect(2:ndims(x)-1)

    μ = mean(x, dims=spatial_dims)
    σ² = var(x, dims=spatial_dims, corrected=false)

    x_norm = (x .- μ) ./ sqrt.(σ² .+ inst.eps)

    if inst.affine
        γ = reshape(inst.γ, ones(Int, ndims(x)-1)..., :)
        β = reshape(inst.β, ones(Int, ndims(x)-1)..., :)
        x_norm = γ .* x_norm .+ β
    end

    x_norm
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
    eps::Float32 = 1e-5f0,
    affine::Bool = true,
    dtype::Type{T} = Float32
) where T
    @assert num_channels % num_groups == 0 "num_channels must be divisible by num_groups"

    γ = affine ? ones(T, num_channels) : nothing
    β = affine ? zeros(T, num_channels) : nothing

    GroupNorm{T}(γ, β, num_groups, num_channels, eps, affine)
end

function forward(gn::GroupNorm, x::AbstractArray)
    # Simplified implementation
    # x shape: (N, ..., C)
    N = size(x, 1)
    C = size(x, ndims(x))
    group_size = C ÷ gn.num_groups

    # Reshape to separate groups
    original_shape = size(x)
    spatial = original_shape[2:end-1]

    # Normalize per group
    y = similar(x)
    for g in 1:gn.num_groups
        c_start = (g - 1) * group_size + 1
        c_end = g * group_size

        group_data = selectdim(x, ndims(x), c_start:c_end)

        μ = mean(group_data)
        σ² = var(group_data, corrected=false)

        normalized = (group_data .- μ) ./ sqrt(σ² + gn.eps)

        # This is a simplified version - proper implementation would be more efficient
        for (i, c) in enumerate(c_start:c_end)
            selectdim(y, ndims(y), c:c) .= selectdim(normalized, ndims(normalized), i:i)
        end
    end

    if gn.affine
        γ = reshape(gn.γ, ones(Int, ndims(x)-1)..., :)
        β = reshape(gn.β, ones(Int, ndims(x)-1)..., :)
        y = γ .* y .+ β
    end

    y
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

function RMSNorm(dim::Int; eps::Float32=1e-6f0, dtype::Type{T}=Float32) where T
    RMSNorm{T}(ones(T, dim), eps)
end

function forward(rn::RMSNorm, x::AbstractArray)
    rms = sqrt.(mean(x .^ 2, dims=ndims(x)) .+ rn.eps)
    x_norm = x ./ rms
    x_norm .* rn.weight'
end

parameters(rn::RMSNorm) = (weight = rn.weight,)
output_shape(::RMSNorm, input_shape) = input_shape
