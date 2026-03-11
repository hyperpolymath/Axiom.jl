# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Data Loading Utilities
#
# Data loading and batching utilities.

"""
    DataLoader(data; batch_size=32, shuffle=false)

Iterator that yields batches of data.

# Arguments
- `data`: Tuple of (features, labels) arrays
- `batch_size`: Number of samples per batch
- `shuffle`: Whether to shuffle data each epoch

# Example
```julia
X = randn(Float32, 1000, 784)
y = rand(1:10, 1000)

loader = DataLoader((X, y), batch_size=32, shuffle=true)

for (batch_x, batch_y) in loader
    # Process batch
end
```
"""
struct DataLoader{T}
    data::T
    batch_size::Int
    shuffle::Bool
    n_samples::Int
    indices::Vector{Int}
end

function DataLoader(data::Tuple; batch_size::Int=32, shuffle::Bool=false)
    n_samples = size(data[1], 1)
    indices = collect(1:n_samples)
    DataLoader(data, batch_size, shuffle, n_samples, indices)
end

function Base.iterate(dl::DataLoader, state=1)
    if state > dl.n_samples
        # Reset indices for next epoch
        if dl.shuffle
            shuffle!(dl.indices)
        end
        return nothing
    end

    batch_end = min(state + dl.batch_size - 1, dl.n_samples)
    batch_indices = dl.indices[state:batch_end]

    # Extract batch for each array in data tuple
    batch = Tuple(
        ndims(arr) == 1 ? arr[batch_indices] : arr[batch_indices, :]
        for arr in dl.data
    )

    return (batch, batch_end + 1)
end

Base.length(dl::DataLoader) = ceil(Int, dl.n_samples / dl.batch_size)

function Base.eltype(dl::DataLoader{T}) where T
    Tuple{typeof.([arr[1:1, :] for arr in dl.data])...}
end

"""
    train_test_split(data; test_ratio=0.2, shuffle=true)

Split data into training and test sets.

# Arguments
- `data`: Tuple of arrays
- `test_ratio`: Fraction of data for test set
- `shuffle`: Whether to shuffle before splitting

# Returns
(train_data, test_data)
"""
function train_test_split(data::Tuple; test_ratio::Float64=0.2, shuffle::Bool=true)
    n_samples = size(data[1], 1)
    n_test = round(Int, n_samples * test_ratio)
    n_train = n_samples - n_test

    indices = shuffle ? randperm(n_samples) : collect(1:n_samples)
    train_indices = indices[1:n_train]
    test_indices = indices[n_train+1:end]

    train_data = Tuple(
        ndims(arr) == 1 ? arr[train_indices] : arr[train_indices, :]
        for arr in data
    )

    test_data = Tuple(
        ndims(arr) == 1 ? arr[test_indices] : arr[test_indices, :]
        for arr in data
    )

    return (train_data, test_data)
end

"""
    normalize(X; dims=1)

Normalize features to zero mean and unit variance.
"""
function normalize(X::AbstractArray; dims::Int=1)
    μ = mean(X, dims=dims)
    σ = std(X, dims=dims)

    # Avoid division by zero
    σ = max.(σ, 1e-8)

    (X .- μ) ./ σ
end

"""
    one_hot(y, num_classes)

Convert integer labels to one-hot encoding.

# Arguments
- `y`: Vector of integer labels (1-indexed)
- `num_classes`: Number of classes

# Returns
Matrix of size (length(y), num_classes)
"""
function one_hot(y::AbstractVector{<:Integer}, num_classes::Int)
    n = length(y)
    onehot = zeros(Float32, n, num_classes)

    for (i, label) in enumerate(y)
        onehot[i, label] = 1.0f0
    end

    onehot
end

"""
    to_categorical(y)

Convert one-hot encoded matrix to integer labels.
"""
function to_categorical(y::AbstractMatrix)
    [argmax(y[i, :]) for i in 1:size(y, 1)]
end

"""
    shuffle_data(data)

Shuffle data tuple along first dimension.
"""
function shuffle_data(data::Tuple)
    n_samples = size(data[1], 1)
    indices = randperm(n_samples)

    Tuple(
        ndims(arr) == 1 ? arr[indices] : arr[indices, :]
        for arr in data
    )
end

"""
    batch_data(data, batch_size)

Split data into batches (as a vector of tuples).
"""
function batch_data(data::Tuple, batch_size::Int)
    n_samples = size(data[1], 1)
    n_batches = ceil(Int, n_samples / batch_size)

    batches = []
    for i in 1:n_batches
        start_idx = (i - 1) * batch_size + 1
        end_idx = min(i * batch_size, n_samples)

        batch = Tuple(
            ndims(arr) == 1 ? arr[start_idx:end_idx] : arr[start_idx:end_idx, :]
            for arr in data
        )
        push!(batches, batch)
    end

    batches
end

# Simple datasets for testing
"""
    make_moons(n_samples=1000; noise=0.1)

Generate two interleaved half circles.
"""
function make_moons(n_samples::Int=1000; noise::Float64=0.1)
    n_samples_out = div(n_samples, 2)
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = cos.(LinRange(0, π, n_samples_out))
    outer_circ_y = sin.(LinRange(0, π, n_samples_out))
    inner_circ_x = 1 .- cos.(LinRange(0, π, n_samples_in))
    inner_circ_y = 1 .- sin.(LinRange(0, π, n_samples_in)) .- 0.5

    X = Float32.([
        vcat(outer_circ_x, inner_circ_x) vcat(outer_circ_y, inner_circ_y)
    ])
    X .+= Float32.(randn(size(X)) .* noise)

    y = vcat(ones(Int, n_samples_out), 2 * ones(Int, n_samples_in))

    (X, y)
end

"""
    make_blobs(n_samples=1000, n_features=2, centers=3)

Generate isotropic Gaussian blobs.
"""
function make_blobs(n_samples::Int=1000, n_features::Int=2, centers::Int=3)
    n_per_center = div(n_samples, centers)

    X = Float32[]
    y = Int[]

    for c in 1:centers
        center = randn(Float32, n_features) .* 5
        samples = randn(Float32, n_per_center, n_features) .+ center'
        X = isempty(X) ? samples : vcat(X, samples)
        append!(y, fill(c, n_per_center))
    end

    (X, y)
end
