# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl PPU Coprocessor Backend
#
# Built-in production kernels for PPUBackend. These methods are intentionally
# defined in-tree so strict mode does not depend on runtime extension injection.

# ============================================================================
# Matrix Operations
# ============================================================================

function _ppu_as_strided_matrix(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    A isa StridedMatrix{T} ? A : Matrix{T}(A)
end

function _ppu_backend_matmul(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T<:AbstractFloat}
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("matmul size mismatch: $(size(A)) x $(size(B))"))
    A_mat = _ppu_as_strided_matrix(A)
    B_mat = _ppu_as_strided_matrix(B)
    C = Matrix{T}(undef, size(A_mat, 1), size(B_mat, 2))
    mul!(C, A_mat, B_mat)
    C
end

function backend_coprocessor_matmul(
    ::PPUBackend,
    A::AbstractMatrix{Float32},
    B::AbstractMatrix{Float32},
)
    _ppu_backend_matmul(A, B)
end

function backend_coprocessor_matmul(
    ::PPUBackend,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T<:AbstractFloat}
    _ppu_backend_matmul(A, B)
end

# ============================================================================
# Activations
# ============================================================================

function backend_coprocessor_relu(
    ::PPUBackend,
    x::AbstractArray{T},
) where {T<:AbstractFloat}
    _ppu_backend_relu(x)
end

function backend_coprocessor_relu(
    ::PPUBackend,
    x::AbstractArray{Float32},
)
    _ppu_backend_relu(x)
end

function _ppu_backend_relu(
    x::AbstractArray{T},
) where {T<:AbstractFloat}
    y = similar(x)
    z = zero(T)
    @inbounds @simd for idx in eachindex(x, y)
        v = x[idx]
        y[idx] = ifelse(v > z, v, z)
    end
    y
end

function _ppu_softmax_dim2(x::AbstractMatrix{T}) where {T<:AbstractFloat}
    rows, cols = size(x)
    y = similar(x)
    cols == 0 && return y

    @inbounds for i in 1:rows
        row_max = x[i, 1]
        for j in 2:cols
            v = x[i, j]
            row_max = ifelse(v > row_max, v, row_max)
        end

        row_sum = zero(T)
        for j in 1:cols
            e = exp(x[i, j] - row_max)
            y[i, j] = e
            row_sum += e
        end

        inv_sum = inv(row_sum)
        for j in 1:cols
            y[i, j] *= inv_sum
        end
    end

    y
end

function _ppu_softmax_dim1(x::AbstractMatrix{T}) where {T<:AbstractFloat}
    rows, cols = size(x)
    y = similar(x)
    rows == 0 && return y

    @inbounds for j in 1:cols
        col_max = x[1, j]
        for i in 2:rows
            v = x[i, j]
            col_max = ifelse(v > col_max, v, col_max)
        end

        col_sum = zero(T)
        for i in 1:rows
            e = exp(x[i, j] - col_max)
            y[i, j] = e
            col_sum += e
        end

        inv_sum = inv(col_sum)
        for i in 1:rows
            y[i, j] *= inv_sum
        end
    end

    y
end

function backend_coprocessor_softmax(
    ::PPUBackend,
    x::AbstractArray{T},
    dim::Int,
) where {T<:AbstractFloat}
    _ppu_backend_softmax(x, dim)
end

function backend_coprocessor_softmax(
    ::PPUBackend,
    x::AbstractArray{Float32},
    dim::Int,
)
    _ppu_backend_softmax(x, dim)
end

function _ppu_backend_softmax(
    x::AbstractArray{T},
    dim::Int,
) where {T<:AbstractFloat}
    if x isa AbstractMatrix{T}
        dim == 2 && return _ppu_softmax_dim2(x)
        dim == 1 && return _ppu_softmax_dim1(x)
    end
    backend_softmax(JuliaBackend(), x, dim)
end

# ============================================================================
# Convolution + Normalization + Pooling
# ============================================================================

function backend_coprocessor_conv2d(
    ::PPUBackend,
    input::AbstractArray{T, 4},
    weight::AbstractArray{T, 4},
    bias::Union{AbstractVector{T}, Nothing},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int},
) where {T<:AbstractFloat}
    backend_conv2d(JuliaBackend(), input, weight, bias, stride, padding)
end

function backend_coprocessor_batchnorm(
    ::PPUBackend,
    x::AbstractArray{T},
    gamma::AbstractVector{T},
    beta::AbstractVector{T},
    running_mean::AbstractVector{T},
    running_var::AbstractVector{T},
    eps::T,
    training::Bool,
) where {T<:AbstractFloat}
    backend_batchnorm(JuliaBackend(), x, gamma, beta, running_mean, running_var, eps, training)
end

function backend_coprocessor_layernorm(
    ::PPUBackend,
    x::AbstractArray{T},
    gamma::AbstractArray{T},
    beta::AbstractArray{T},
    normalized_shape::Tuple,
    eps::T,
) where {T<:AbstractFloat}
    backend_layernorm(JuliaBackend(), x, gamma, beta, normalized_shape, eps)
end

function backend_coprocessor_maxpool2d(
    ::PPUBackend,
    input::AbstractArray{T, 4},
    kernel_size::Tuple{Int, Int},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int},
) where {T<:AbstractFloat}
    backend_maxpool2d(JuliaBackend(), input, kernel_size, stride, padding)
end

function backend_coprocessor_avgpool2d(
    ::PPUBackend,
    input::AbstractArray{T, 4},
    kernel_size::Tuple{Int, Int},
    stride::Tuple{Int, Int},
    padding::Tuple{Int, Int},
    count_include_pad::Bool=true,
) where {T<:AbstractFloat}
    backend_avgpool2d(JuliaBackend(), input, kernel_size, stride, padding, count_include_pad)
end

function backend_coprocessor_global_avgpool2d(
    ::PPUBackend,
    input::AbstractArray{T, 4},
) where {T<:AbstractFloat}
    backend_global_avgpool2d(JuliaBackend(), input)
end
