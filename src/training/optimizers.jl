# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Optimizers
#
# Standard optimization algorithms for neural network training.

"""
    AbstractOptimizer

Base type for all optimizers.
"""
abstract type AbstractOptimizer end

"""
    step!(optimizer, params, grads)

Perform a single optimization step, updating `params` in-place based on
the computed `grads`. Each optimizer implements its own logic for this
function.
"""
function step! end

"""
    zero_grad!(params)

Reset all gradients to zero.
"""
function zero_grad!(params::NamedTuple)
    for (name, param) in pairs(params)
        if param isa Gradient
            param.grad = nothing
        end
    end
end

# ============================================================================
# SGD
# ============================================================================

"""
    SGD(params; lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=false)

Stochastic Gradient Descent with momentum.

# Arguments
- `lr`: Learning rate
- `momentum`: Momentum factor (default: 0)
- `weight_decay`: L2 regularization factor (default: 0)
- `nesterov`: Use Nesterov momentum (default: false)
"""
mutable struct SGD <: AbstractOptimizer
    lr::Float32
    momentum::Float32
    weight_decay::Float32
    nesterov::Bool
    velocity::Dict{UInt64, Any}
end

function SGD(; lr::Float32=0.01f0, momentum::Float32=0.0f0,
             weight_decay::Float32=0.0f0, nesterov::Bool=false)
    SGD(lr, momentum, weight_decay, nesterov, Dict())
end

function step!(opt::SGD, params::NamedTuple, grads::NamedTuple)
    for name in keys(params)
        param = params[name]
        grad = grads[name]

        grad === nothing && continue

        # Weight decay
        if opt.weight_decay > 0
            grad = grad + opt.weight_decay * param
        end

        # Momentum
        if opt.momentum > 0
            id = objectid(param)
            if !haskey(opt.velocity, id)
                opt.velocity[id] = zero(param)
            end
            v = opt.velocity[id]
            v .= opt.momentum .* v .+ grad

            if opt.nesterov
                grad = grad + opt.momentum * v
            else
                grad = v
            end
        end

        # Update
        param .-= opt.lr .* grad
    end
end

# ============================================================================
# Adam
# ============================================================================

"""
    Adam(; lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

Adam optimizer.

# Arguments
- `lr`: Learning rate
- `betas`: Coefficients for computing running averages
- `eps`: Term for numerical stability
- `weight_decay`: L2 regularization factor
"""
mutable struct Adam <: AbstractOptimizer
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    weight_decay::Float32
    step_count::Int
    m::Dict{UInt64, Any}  # First moment
    v::Dict{UInt64, Any}  # Second moment
end

function Adam(; lr::Float32=0.001f0, betas::Tuple{Float32, Float32}=(0.9f0, 0.999f0),
              eps::Float32=Float32(1e-8), weight_decay::Float32=0.0f0)
    Adam(lr, betas[1], betas[2], eps, weight_decay, 0, Dict(), Dict())
end

function step!(opt::Adam, params::NamedTuple, grads::NamedTuple)
    opt.step_count += 1
    t = opt.step_count

    for name in keys(params)
        param = params[name]
        grad = grads[name]

        grad === nothing && continue

        id = objectid(param)

        # Initialize moments
        if !haskey(opt.m, id)
            opt.m[id] = zero(param)
            opt.v[id] = zero(param)
        end

        m = opt.m[id]
        v = opt.v[id]

        # Weight decay
        if opt.weight_decay > 0
            grad = grad + opt.weight_decay * param
        end

        # Update moments
        m .= opt.beta1 .* m .+ (1 - opt.beta1) .* grad
        v .= opt.beta2 .* v .+ (1 - opt.beta2) .* grad .^ 2

        # Bias correction
        m_hat = m ./ (1 - opt.beta1^t)
        v_hat = v ./ (1 - opt.beta2^t)

        # Update parameters
        param .-= opt.lr .* m_hat ./ (sqrt.(v_hat) .+ opt.eps)
    end
end

# ============================================================================
# AdamW
# ============================================================================

"""
    AdamW(; lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

AdamW optimizer (Adam with decoupled weight decay).
"""
mutable struct AdamW <: AbstractOptimizer
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    weight_decay::Float32
    step_count::Int
    m::Dict{UInt64, Any}
    v::Dict{UInt64, Any}
end

function AdamW(; lr::Float32=0.001f0, betas::Tuple{Float32, Float32}=(0.9f0, 0.999f0),
               eps::Float32=Float32(1e-8), weight_decay::Float32=0.01f0)
    AdamW(lr, betas[1], betas[2], eps, weight_decay, 0, Dict(), Dict())
end

function step!(opt::AdamW, params::NamedTuple, grads::NamedTuple)
    opt.step_count += 1
    t = opt.step_count

    for name in keys(params)
        param = params[name]
        grad = grads[name]

        grad === nothing && continue

        id = objectid(param)

        if !haskey(opt.m, id)
            opt.m[id] = zero(param)
            opt.v[id] = zero(param)
        end

        m = opt.m[id]
        v = opt.v[id]

        # Update moments (without weight decay in gradient)
        m .= opt.beta1 .* m .+ (1 - opt.beta1) .* grad
        v .= opt.beta2 .* v .+ (1 - opt.beta2) .* grad .^ 2

        # Bias correction
        m_hat = m ./ (1 - opt.beta1^t)
        v_hat = v ./ (1 - opt.beta2^t)

        # Update with decoupled weight decay
        param .-= opt.lr .* (m_hat ./ (sqrt.(v_hat) .+ opt.eps) .+ opt.weight_decay .* param)
    end
end

# ============================================================================
# RMSprop
# ============================================================================

"""
    RMSprop(; lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0)

RMSprop optimizer.
"""
mutable struct RMSprop <: AbstractOptimizer
    lr::Float32
    alpha::Float32
    eps::Float32
    weight_decay::Float32
    momentum::Float32
    v::Dict{UInt64, Any}
    buffer::Dict{UInt64, Any}
end

function RMSprop(; lr::Float32=0.01f0, alpha::Float32=0.99f0,
                 eps::Float32=Float32(1e-8), weight_decay::Float32=0.0f0,
                 momentum::Float32=0.0f0)
    RMSprop(lr, alpha, eps, weight_decay, momentum, Dict(), Dict())
end

function step!(opt::RMSprop, params::NamedTuple, grads::NamedTuple)
    for name in keys(params)
        param = params[name]
        grad = grads[name]

        grad === nothing && continue

        id = objectid(param)

        if !haskey(opt.v, id)
            opt.v[id] = zero(param)
            if opt.momentum > 0
                opt.buffer[id] = zero(param)
            end
        end

        v = opt.v[id]

        # Weight decay
        if opt.weight_decay > 0
            grad = grad + opt.weight_decay * param
        end

        # Update running average
        v .= opt.alpha .* v .+ (1 - opt.alpha) .* grad .^ 2

        if opt.momentum > 0
            buffer = opt.buffer[id]
            buffer .= opt.momentum .* buffer .+ grad ./ (sqrt.(v) .+ opt.eps)
            param .-= opt.lr .* buffer
        else
            param .-= opt.lr .* grad ./ (sqrt.(v) .+ opt.eps)
        end
    end
end

# ============================================================================
# Learning Rate Schedulers
# ============================================================================

"""
    AbstractScheduler

Base type for learning rate schedulers.
"""
abstract type AbstractScheduler end

"""
    get_lr(scheduler, epoch)

Get learning rate for given epoch.
"""
function get_lr end

"""
    StepLR(optimizer, step_size, gamma=0.1)

Decays learning rate by gamma every step_size epochs.
"""
mutable struct StepLR <: AbstractScheduler
    optimizer::AbstractOptimizer
    step_size::Int
    gamma::Float32
    base_lr::Float32
end

function StepLR(optimizer::AbstractOptimizer, step_size::Int; gamma::Float32=0.1f0)
    StepLR(optimizer, step_size, gamma, optimizer.lr)
end

function step!(scheduler::StepLR, epoch::Int)
    factor = scheduler.gamma ^ div(epoch, scheduler.step_size)
    scheduler.optimizer.lr = scheduler.base_lr * factor
end

"""
    CosineAnnealingLR(optimizer, T_max, eta_min=0)

Cosine annealing learning rate.
"""
mutable struct CosineAnnealingLR <: AbstractScheduler
    optimizer::AbstractOptimizer
    T_max::Int
    eta_min::Float32
    base_lr::Float32
end

function CosineAnnealingLR(optimizer::AbstractOptimizer, T_max::Int; eta_min::Float32=0.0f0)
    CosineAnnealingLR(optimizer, T_max, eta_min, optimizer.lr)
end

function step!(scheduler::CosineAnnealingLR, epoch::Int)
    t = epoch % scheduler.T_max
    scheduler.optimizer.lr = scheduler.eta_min +
        (scheduler.base_lr - scheduler.eta_min) * (1 + cos(π * t / scheduler.T_max)) / 2
end

"""
    WarmupScheduler(optimizer, warmup_epochs, after_scheduler)

Linear warmup followed by another scheduler.
"""
mutable struct WarmupScheduler <: AbstractScheduler
    optimizer::AbstractOptimizer
    warmup_epochs::Int
    after_scheduler::AbstractScheduler
    base_lr::Float32
end

function WarmupScheduler(optimizer::AbstractOptimizer, warmup_epochs::Int, after_scheduler::AbstractScheduler)
    WarmupScheduler(optimizer, warmup_epochs, after_scheduler, optimizer.lr)
end

function step!(scheduler::WarmupScheduler, epoch::Int)
    if epoch < scheduler.warmup_epochs
        # Linear warmup
        scheduler.optimizer.lr = scheduler.base_lr * (epoch + 1) / scheduler.warmup_epochs
    else
        # Use after scheduler
        step!(scheduler.after_scheduler, epoch - scheduler.warmup_epochs)
    end
end
