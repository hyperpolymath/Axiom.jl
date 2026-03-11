# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Weight Initialization Utilities
#
# Various initialization schemes for neural network weights.

"""
    kaiming_uniform!(tensor, a=0, mode=:fan_in, nonlinearity=:leaky_relu)

Initialize tensor with Kaiming uniform distribution.
"""
function kaiming_uniform!(tensor::AbstractArray{T}, a::Real=0;
                          mode::Symbol=:fan_in,
                          nonlinearity::Symbol=:leaky_relu) where T
    fan_in, fan_out = _calculate_fan(tensor)
    fan = mode == :fan_in ? fan_in : fan_out

    gain = _calculate_gain(nonlinearity, a)
    std = gain / sqrt(fan)
    bound = sqrt(3.0) * std

    tensor .= T.((rand(size(tensor)...) .- 0.5) .* 2 .* bound)
    tensor
end

"""
    kaiming_normal!(tensor, a=0, mode=:fan_in, nonlinearity=:leaky_relu)

Initialize tensor with Kaiming normal distribution.
"""
function kaiming_normal!(tensor::AbstractArray{T}, a::Real=0;
                         mode::Symbol=:fan_in,
                         nonlinearity::Symbol=:leaky_relu) where T
    fan_in, fan_out = _calculate_fan(tensor)
    fan = mode == :fan_in ? fan_in : fan_out

    gain = _calculate_gain(nonlinearity, a)
    std = gain / sqrt(fan)

    tensor .= T.(randn(size(tensor)...) .* std)
    tensor
end

"""
    xavier_uniform!(tensor, gain=1.0)

Initialize tensor with Xavier uniform distribution.
"""
function xavier_uniform!(tensor::AbstractArray{T}, gain::Real=1.0) where T
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * sqrt(2.0 / (fan_in + fan_out))
    bound = sqrt(3.0) * std

    tensor .= T.((rand(size(tensor)...) .- 0.5) .* 2 .* bound)
    tensor
end

"""
    xavier_normal!(tensor, gain=1.0)

Initialize tensor with Xavier normal distribution.
"""
function xavier_normal!(tensor::AbstractArray{T}, gain::Real=1.0) where T
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * sqrt(2.0 / (fan_in + fan_out))

    tensor .= T.(randn(size(tensor)...) .* std)
    tensor
end

"""
    orthogonal!(tensor, gain=1.0)

Initialize tensor with orthogonal initialization.
"""
function orthogonal!(tensor::AbstractArray{T}, gain::Real=1.0) where T
    rows, cols = size(tensor, 1), prod(size(tensor)[2:end])
    flat_shape = (rows, cols)

    if rows < cols
        Q = qr(randn(T, cols, rows)).Q
        Q = Matrix(Q)'
    else
        Q = qr(randn(T, rows, cols)).Q
        Q = Matrix(Q)
    end

    tensor .= T.(reshape(gain .* Q, size(tensor)...))
    tensor
end

"""
    sparse!(tensor, sparsity, std=0.01)

Initialize tensor with sparse initialization.
"""
function sparse!(tensor::AbstractArray{T}, sparsity::Real;
                 std::Real=0.01) where T
    tensor .= T.(randn(size(tensor)...) .* std)

    # Set random entries to zero
    n_zeros = round(Int, length(tensor) * sparsity)
    indices = randperm(length(tensor))[1:n_zeros]

    for idx in indices
        tensor[idx] = zero(T)
    end

    tensor
end

"""
    zeros!(tensor)

Initialize tensor with zeros.
"""
function zeros!(tensor::AbstractArray{T}) where T
    tensor .= zero(T)
    tensor
end

"""
    ones!(tensor)

Initialize tensor with ones.
"""
function ones!(tensor::AbstractArray{T}) where T
    tensor .= one(T)
    tensor
end

"""
    constant!(tensor, value)

Initialize tensor with constant value.
"""
function constant!(tensor::AbstractArray{T}, value::Real) where T
    tensor .= T(value)
    tensor
end

# Helper functions
"""
    _calculate_fan(tensor) -> (fan_in, fan_out)

Calculate fan-in and fan-out for a tensor, used for initialization schemes.
"""
function _calculate_fan(tensor::AbstractArray)
    dimensions = ndims(tensor)
    if dimensions < 2
        error("Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions")
    end

    if dimensions == 2
        fan_in = size(tensor, 2)
        fan_out = size(tensor, 1)
    else
        num_input_maps = size(tensor, ndims(tensor) - 1)
        num_output_maps = size(tensor, ndims(tensor))
        receptive_field_size = prod(size(tensor)[1:ndims(tensor)-2])

        fan_in = num_input_maps * receptive_field_size
        fan_out = num_output_maps * receptive_field_size
    end

    return (fan_in, fan_out)
end

"""
    _calculate_gain(nonlinearity, param=0) -> Float64

Calculate the recommended gain value for a given nonlinearity function,
used in Kaiming and Xavier initialization.
"""
function _calculate_gain(nonlinearity::Symbol, param::Real=0)
    if nonlinearity == :linear || nonlinearity == :sigmoid
        return 1.0
    elseif nonlinearity == :tanh
        return 5.0 / 3
    elseif nonlinearity == :relu
        return sqrt(2.0)
    elseif nonlinearity == :leaky_relu
        return sqrt(2.0 / (1 + param^2))
    elseif nonlinearity == :selu
        return 3.0 / 4
    else
        error("Unknown nonlinearity: $nonlinearity")
    end
end
