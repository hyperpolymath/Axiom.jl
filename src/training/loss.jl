# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Loss Functions
#
# Standard loss functions for neural network training.

"""
    mse_loss(pred, target; reduction=:mean)

Mean Squared Error loss.

# Arguments
- `pred`: Predictions
- `target`: Ground truth
- `reduction`: :mean, :sum, or :none
"""
_loss_data(x) = x isa AbstractTensor ? x.data : x

function mse_loss(pred, target; reduction::Symbol=:mean)
    pred_data = _loss_data(pred)
    target_data = _loss_data(target)
    diff = (pred_data .- target_data) .^ 2

    if reduction == :mean
        mean(diff)
    elseif reduction == :sum
        sum(diff)
    else
        diff
    end
end

"""
    l1_loss(pred, target; reduction=:mean)

L1 (Mean Absolute Error) loss.
"""
function l1_loss(pred, target; reduction::Symbol=:mean)
    pred_data = _loss_data(pred)
    target_data = _loss_data(target)
    diff = abs.(pred_data .- target_data)

    if reduction == :mean
        mean(diff)
    elseif reduction == :sum
        sum(diff)
    else
        diff
    end
end

"""
    smooth_l1_loss(pred, target; beta=1.0, reduction=:mean)

Smooth L1 loss (Huber loss).
"""
function smooth_l1_loss(pred, target; beta::Float32=1.0f0, reduction::Symbol=:mean)
    pred_data = _loss_data(pred)
    target_data = _loss_data(target)
    diff = abs.(pred_data .- target_data)
    loss = ifelse.(diff .< beta,
                   0.5f0 .* diff .^ 2 ./ beta,
                   diff .- 0.5f0 .* beta)

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

"""
    crossentropy(pred, target; reduction=:mean)

Cross-entropy loss for classification.

# Arguments
- `pred`: Predicted probabilities (after softmax)
- `target`: Ground truth labels (one-hot or indices)
- `reduction`: :mean, :sum, or :none
"""
function crossentropy(pred, target; reduction::Symbol=:mean, eps::Float32=Float32(1e-7))
    pred_data = _loss_data(pred)
    target_data = _loss_data(target)

    # Clamp predictions for numerical stability
    pred_clamped = clamp.(pred_data, eps, 1.0f0 - eps)

    if target_data isa AbstractVector{<:Integer}
        # Index format - select correct class probability
        batch_size = size(pred_data, 1)
        loss = [-log(pred_clamped[i, target_data[i]]) for i in 1:batch_size]
    else
        # One-hot format
        loss = -sum(target_data .* log.(pred_clamped), dims=2)
    end

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

"""
    binary_crossentropy(pred, target; reduction=:mean)

Binary cross-entropy loss.
"""
function binary_crossentropy(pred, target; reduction::Symbol=:mean, eps::Float32=Float32(1e-7))
    pred_data = _loss_data(pred)
    target_data = _loss_data(target)
    pred_clamped = clamp.(pred_data, eps, 1.0f0 - eps)
    loss = -(target_data .* log.(pred_clamped) .+ (1 .- target_data) .* log.(1 .- pred_clamped))

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

"""
    nll_loss(pred, target; reduction=:mean)

Negative Log Likelihood loss (expects log probabilities).
"""
function nll_loss(pred, target; reduction::Symbol=:mean)
    pred_data = _loss_data(pred)
    target_data = _loss_data(target)

    if target_data isa AbstractVector{<:Integer}
        batch_size = size(pred_data, 1)
        loss = [-pred_data[i, target_data[i]] for i in 1:batch_size]
    else
        loss = -sum(target_data .* pred_data, dims=2)
    end

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

"""
    kl_divergence(p, q; reduction=:mean)

KL Divergence: KL(p || q) = sum(p * log(p / q))
"""
function kl_divergence(p, q; reduction::Symbol=:mean, eps::Float32=Float32(1e-7))
    p_data = _loss_data(p)
    q_data = _loss_data(q)
    p_clamped = clamp.(p_data, eps, 1.0f0)
    q_clamped = clamp.(q_data, eps, 1.0f0)

    loss = sum(p_clamped .* log.(p_clamped ./ q_clamped), dims=2)

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

"""
    hinge_loss(pred, target; reduction=:mean)

Hinge loss for SVMs and margin-based classification.
"""
function hinge_loss(pred, target; reduction::Symbol=:mean)
    pred_data = _loss_data(pred)
    target_data = _loss_data(target)
    # target should be -1 or 1
    loss = max.(0, 1 .- target_data .* pred_data)

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

"""
    focal_loss(pred, target; alpha=0.25, gamma=2.0, reduction=:mean)

Focal Loss for handling class imbalance.
"""
function focal_loss(pred, target; alpha::Float32=0.25f0, gamma::Float32=2.0f0,
                    reduction::Symbol=:mean, eps::Float32=Float32(1e-7))
    pred_data = _loss_data(pred)
    target_data = _loss_data(target)
    pred_clamped = clamp.(pred_data, eps, 1.0f0 - eps)

    # Binary case
    ce = -target_data .* log.(pred_clamped) .- (1 .- target_data) .* log.(1 .- pred_clamped)
    p_t = target_data .* pred_clamped .+ (1 .- target_data) .* (1 .- pred_clamped)
    alpha_t = target_data .* alpha .+ (1 .- target_data) .* (1 - alpha)

    loss = alpha_t .* ((1 .- p_t) .^ gamma) .* ce

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

"""
    contrastive_loss(embeddings1, embeddings2, labels; margin=1.0, reduction=:mean)

Contrastive loss for siamese networks.
"""
function contrastive_loss(emb1, emb2, labels; margin::Float32=1.0f0, reduction::Symbol=:mean)
    emb1_data = _loss_data(emb1)
    emb2_data = _loss_data(emb2)
    labels_data = _loss_data(labels)
    # labels: 1 for similar, 0 for dissimilar
    distances = sqrt.(sum((emb1_data .- emb2_data) .^ 2, dims=2) .+ Float32(1e-7))

    # Similar pairs: minimize distance
    # Dissimilar pairs: maximize distance up to margin
    loss = labels_data .* distances .^ 2 .+ (1 .- labels_data) .* max.(0, margin .- distances) .^ 2

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

"""
    triplet_loss(anchor, positive, negative; margin=0.2, reduction=:mean)

Triplet loss for metric learning.
"""
function triplet_loss(anchor, positive, negative; margin::Float32=0.2f0, reduction::Symbol=:mean)
    anchor_data = _loss_data(anchor)
    positive_data = _loss_data(positive)
    negative_data = _loss_data(negative)
    pos_dist = sqrt.(sum((anchor_data .- positive_data) .^ 2, dims=2) .+ Float32(1e-7))
    neg_dist = sqrt.(sum((anchor_data .- negative_data) .^ 2, dims=2) .+ Float32(1e-7))

    loss = max.(0, pos_dist .- neg_dist .+ margin)

    if reduction == :mean
        mean(loss)
    elseif reduction == :sum
        sum(loss)
    else
        loss
    end
end

# ============================================================================
# Loss function wrappers for @axiom
# ============================================================================

"""
    Loss

Wrapper type that stores loss function with its parameters.
"""
struct Loss{F, P}
    fn::F
    params::P
end

Loss(fn::Function; kwargs...) = Loss(fn, kwargs)

(loss::Loss)(pred, target) = loss.fn(pred, target; loss.params...)

# Pre-configured losses
const MSELoss = Loss(mse_loss)
const L1Loss = Loss(l1_loss)
const CrossEntropyLoss = Loss(crossentropy)
const BCELoss = Loss(binary_crossentropy)
const NLLLoss = Loss(nll_loss)
