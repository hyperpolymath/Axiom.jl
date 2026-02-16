# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Julia Backend
#
# Pure Julia implementations of all operations.
# This is the reference implementation for correctness testing.

# ============================================================================
# Matrix Operations
# ============================================================================

function backend_matmul(::JuliaBackend, A::AbstractArray, B::AbstractArray)
    A * B
end

function backend_transpose(::JuliaBackend, A::AbstractArray)
    permutedims(A, reverse(1:ndims(A)))
end

# ============================================================================
# Convolution
# ============================================================================

function backend_conv2d(
    ::JuliaBackend,
    input::AbstractArray{T, 4},
    weight::AbstractArray{T, 4},
    bias::Union{AbstractVector{T}, Nothing},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int}
) where T
    N, H_in, W_in, C_in = size(input)
    kH, kW, _, C_out = size(weight)
    sH, sW = stride
    pH, pW = padding

    H_out = div(H_in + 2*pH - kH, sH) + 1
    W_out = div(W_in + 2*pW - kW, sW) + 1

    # Pad input
    if pH > 0 || pW > 0
        padded = zeros(T, N, H_in + 2*pH, W_in + 2*pW, C_in)
        padded[:, pH+1:pH+H_in, pW+1:pW+W_in, :] = input
        input = padded
    end

    output = zeros(T, N, H_out, W_out, C_out)

    # Naive implementation (im2col would be faster)
    @inbounds for n in 1:N
        for oc in 1:C_out
            for i in 1:H_out
                for j in 1:W_out
                    h_start = (i - 1) * sH + 1
                    w_start = (j - 1) * sW + 1

                    val = zero(T)
                    for kh in 1:kH
                        for kw in 1:kW
                            for ic in 1:C_in
                                val += input[n, h_start+kh-1, w_start+kw-1, ic] * weight[kh, kw, ic, oc]
                            end
                        end
                    end
                    output[n, i, j, oc] = val
                end
            end
        end
    end

    # Add bias
    if bias !== nothing
        for oc in 1:C_out
            output[:, :, :, oc] .+= bias[oc]
        end
    end

    output
end

# ============================================================================
# Activations
# ============================================================================

function backend_relu(::JuliaBackend, x::AbstractArray{T}) where T
    max.(zero(T), x)
end

function backend_leaky_relu(::JuliaBackend, x::AbstractArray{T}, alpha::T) where T
    ifelse.(x .> zero(T), x, alpha .* x)
end

function backend_sigmoid(::JuliaBackend, x::AbstractArray{T}) where T
    one(T) ./ (one(T) .+ exp.(-x))
end

function backend_tanh(::JuliaBackend, x::AbstractArray)
    tanh.(x)
end

function backend_softmax(::JuliaBackend, x::AbstractArray{T}, dim::Int) where T
    x_max = maximum(x, dims=dim)
    exp_x = exp.(x .- x_max)
    exp_x ./ sum(exp_x, dims=dim)
end

function backend_gelu(::JuliaBackend, x::AbstractArray{T}) where T
    c = T(sqrt(2 / π))
    T(0.5) .* x .* (one(T) .+ tanh.(c .* (x .+ T(0.044715) .* x .^ 3)))
end

# ============================================================================
# Normalization
# ============================================================================

function backend_batchnorm(
    ::JuliaBackend,
    x::AbstractArray{T},
    gamma::AbstractVector{T},
    beta::AbstractVector{T},
    running_mean::AbstractVector{T},
    running_var::AbstractVector{T},
    eps::T,
    training::Bool
) where T
    if training
        dims = collect(1:ndims(x)-1)
        μ = mean(x, dims=dims)
        σ² = var(x, dims=dims, corrected=false)
    else
        μ = reshape(running_mean, ones(Int, ndims(x)-1)..., :)
        σ² = reshape(running_var, ones(Int, ndims(x)-1)..., :)
    end

    x_norm = (x .- μ) ./ sqrt.(σ² .+ eps)

    γ = reshape(gamma, ones(Int, ndims(x)-1)..., :)
    β = reshape(beta, ones(Int, ndims(x)-1)..., :)

    γ .* x_norm .+ β
end

function backend_layernorm(
    ::JuliaBackend,
    x::AbstractArray{T},
    gamma::AbstractArray{T},
    beta::AbstractArray{T},
    normalized_shape::Tuple,
    eps::T
) where T
    nd = ndims(x)
    n_dims = length(normalized_shape)
    if n_dims == 0 || n_dims > nd
        throw(DimensionMismatch("normalized_shape=$(normalized_shape) is incompatible with input shape $(size(x))"))
    end
    norm_dims = collect(nd - n_dims + 1:nd)

    μ = mean(x, dims=norm_dims)
    σ² = var(x, dims=norm_dims, corrected=false)

    x_norm = (x .- μ) ./ sqrt.(σ² .+ eps)

    affine_shape = ntuple(i -> i <= nd - n_dims ? 1 : normalized_shape[i - (nd - n_dims)], nd)
    γ = reshape(gamma, affine_shape)
    β = reshape(beta, affine_shape)

    γ .* x_norm .+ β
end

# ============================================================================
# Pooling
# ============================================================================

function backend_maxpool2d(
    ::JuliaBackend,
    input::AbstractArray{T, 4},
    kernel_size::Tuple{Int, Int},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int}
) where T
    N, H_in, W_in, C = size(input)
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    H_out = div(H_in + 2*pH - kH, sH) + 1
    W_out = div(W_in + 2*pW - kW, sW) + 1

    # Pad with -Inf
    if pH > 0 || pW > 0
        padded = fill(typemin(T), N, H_in + 2*pH, W_in + 2*pW, C)
        padded[:, pH+1:pH+H_in, pW+1:pW+W_in, :] = input
        input = padded
    end

    output = zeros(T, N, H_out, W_out, C)

    @inbounds for n in 1:N
        for c in 1:C
            for i in 1:H_out
                for j in 1:W_out
                    h_start = (i - 1) * sH + 1
                    w_start = (j - 1) * sW + 1

                    max_val = typemin(T)
                    for kh in 1:kH
                        for kw in 1:kW
                            val = input[n, h_start+kh-1, w_start+kw-1, c]
                            max_val = max(max_val, val)
                        end
                    end
                    output[n, i, j, c] = max_val
                end
            end
        end
    end

    output
end

function backend_avgpool2d(
    ::JuliaBackend,
    input::AbstractArray{T, 4},
    kernel_size::Tuple{Int, Int},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int},
    count_include_pad::Bool=true,
) where T
    N, H_in, W_in, C = size(input)
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    H_out = div(H_in + 2*pH - kH, sH) + 1
    W_out = div(W_in + 2*pW - kW, sW) + 1

    # Pad with zeros for out-of-bounds regions
    if pH > 0 || pW > 0
        padded = zeros(T, N, H_in + 2*pH, W_in + 2*pW, C)
        padded[:, pH+1:pH+H_in, pW+1:pW+W_in, :] = input
        input = padded
    end

    output = zeros(T, N, H_out, W_out, C)

    @inbounds for n in 1:N
        for c in 1:C
            for i in 1:H_out
                for j in 1:W_out
                    h_start = (i - 1) * sH + 1
                    w_start = (j - 1) * sW + 1

                    sum_val = zero(T)
                    valid_count = 0
                    for kh in 1:kH
                        for kw in 1:kW
                            h_idx = h_start + kh - 1
                            w_idx = w_start + kw - 1
                            if h_idx >= pH + 1 && h_idx <= pH + H_in &&
                               w_idx >= pW + 1 && w_idx <= pW + W_in
                                valid_count += 1
                            end
                            sum_val += input[n, h_idx, w_idx, c]
                        end
                    end
                    divisor = count_include_pad ? (kH * kW) : max(valid_count, 1)
                    output[n, i, j, c] = sum_val / T(divisor)
                end
            end
        end
    end

    output
end

function backend_global_avgpool2d(::JuliaBackend, input::AbstractArray{T, 4}) where T
    N, H, W, C = size(input)
    output = zeros(T, N, C)
    scale = T(H * W)

    @inbounds for n in 1:N
        for c in 1:C
            sum_val = zero(T)
            for i in 1:H
                for j in 1:W
                    sum_val += input[n, i, j, c]
                end
            end
            output[n, c] = sum_val / scale
        end
    end

    output
end

# ============================================================================
# Utility Functions
# ============================================================================

function backend_flatten(::JuliaBackend, x::AbstractArray, start_dim::Int)
    batch_size = size(x, 1)
    features = prod(size(x)[start_dim:end])
    reshape(x, batch_size, features)
end

function backend_dropout(::JuliaBackend, x::AbstractArray{T}, p::T, training::Bool) where T
    if !training || p == zero(T)
        return x
    end

    mask = rand(T, size(x)...) .> p
    x .* mask ./ (one(T) - p)
end
