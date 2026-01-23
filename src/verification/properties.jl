# Axiom.jl Verifiable Properties
#
# Standard properties that can be verified about ML models.

"""
    Property

Abstract base type for verifiable properties.
"""
abstract type Property end

"""
    check(property, model, data) -> Bool

Check if a model satisfies a property on given data.
"""
function check end

"""
    prove(property, model) -> ProofResult

Attempt to formally prove a property about a model.
"""
function prove end

# ============================================================================
# Output Properties
# ============================================================================

"""
    ValidProbabilities()

Output represents valid probability distribution.
- sum ≈ 1.0
- all values ∈ [0, 1]
"""
struct ValidProbabilities <: Property end

function check(::ValidProbabilities, model, data)
    for (x, _) in data
        output = model(x)

        # Check sum
        sums = sum(output, dims=ndims(output))
        if !all(isapprox.(sums, 1.0, atol=1e-5))
            return false
        end

        # Check bounds
        if any(output .< 0) || any(output .> 1)
            return false
        end
    end
    true
end

"""
    BoundedOutput(low, high)

Output values are bounded.
"""
struct BoundedOutput{T} <: Property
    low::T
    high::T
end

function check(prop::BoundedOutput, model, data)
    for (x, _) in data
        output = model(x)
        if any(output .< prop.low) || any(output .> prop.high)
            return false
        end
    end
    true
end

"""
    NoNaN()

Output contains no NaN values.
"""
struct NoNaN <: Property end

function check(::NoNaN, model, data)
    for (x, _) in data
        output = model(x)
        if any(isnan, output)
            return false
        end
    end
    true
end

"""
    NoInf()

Output contains no Inf values.
"""
struct NoInf <: Property end

function check(::NoInf, model, data)
    for (x, _) in data
        output = model(x)
        if any(isinf, output)
            return false
        end
    end
    true
end

"""
    FiniteOutput()

Output contains only finite values (no NaN or Inf).
"""
struct FiniteOutput <: Property end

function check(::FiniteOutput, model, data)
    check(NoNaN(), model, data) && check(NoInf(), model, data)
end

# ============================================================================
# Robustness Properties
# ============================================================================

"""
    LocalLipschitz(epsilon, delta)

Local Lipschitz continuity: ∀x. ||x - x'|| < ε ⟹ ||f(x) - f(x')|| < δ
"""
struct LocalLipschitz{T} <: Property
    epsilon::T
    delta::T
end

function check(prop::LocalLipschitz, model, data; n_perturbations::Int=10)
    for (x, _) in data
        base_output = model(x)

        for _ in 1:n_perturbations
            # Random perturbation within epsilon ball
            perturbation = randn(size(x)...) .* prop.epsilon
            perturbed_x = x .+ perturbation

            perturbed_output = model(perturbed_x)

            # Check output difference
            diff = maximum(abs.(base_output .- perturbed_output))
            if diff > prop.delta
                return false
            end
        end
    end
    true
end

"""
    AdversarialRobust(epsilon, attack)

Model is robust to adversarial perturbations of size epsilon.
"""
struct AdversarialRobust{T} <: Property
    epsilon::T
    attack::Symbol  # :fgsm, :pgd, etc.
end

AdversarialRobust(epsilon) = AdversarialRobust(epsilon, :fgsm)

function check(prop::AdversarialRobust, model, data)
    for (x, y) in data
        # Get original prediction
        original_pred = argmax(model(x), dims=2)

        # Generate adversarial example
        adv_x = generate_adversarial(model, x, y, prop.epsilon, prop.attack)

        # Check if prediction changed
        adv_pred = argmax(model(adv_x), dims=2)

        if original_pred != adv_pred
            return false
        end
    end
    true
end

"""
Generate adversarial example using specified attack.
"""
function generate_adversarial(model, x, y, epsilon, attack::Symbol)
    if attack == :fgsm
        return fgsm_attack(model, x, y, epsilon)
    elseif attack == :pgd
        return pgd_attack(model, x, y, epsilon)
    else
        error("Unknown attack: $attack")
    end
end

"""
Fast Gradient Sign Method attack.
"""
function fgsm_attack(model, x, y, epsilon)
    # Compute gradient of loss w.r.t. input
    grad = compute_input_gradient(model, x, y)

    # Sign of gradient
    sign_grad = sign.(grad)

    # Perturb in direction of sign
    x .+ epsilon .* sign_grad
end

"""
Projected Gradient Descent attack.
"""
function pgd_attack(model, x, y, epsilon; steps::Int=20, step_size=nothing)
    step_size = step_size === nothing ? epsilon / steps * 2 : step_size

    adv_x = copy(x)
    original_x = copy(x)

    for _ in 1:steps
        grad = compute_input_gradient(model, adv_x, y)
        adv_x = adv_x .+ step_size .* sign.(grad)

        # Project back to epsilon ball
        perturbation = adv_x .- original_x
        perturbation = clamp.(perturbation, -epsilon, epsilon)
        adv_x = original_x .+ perturbation
    end

    adv_x
end

"""
Compute gradient of loss with respect to input.
"""
function compute_input_gradient(model, x, y)
    # Simplified numerical gradient
    eps = 1e-5f0
    grad = similar(x)

    for i in eachindex(x)
        x_plus = copy(x)
        x_plus[i] += eps

        x_minus = copy(x)
        x_minus[i] -= eps

        loss_plus = crossentropy(model(x_plus), y)
        loss_minus = crossentropy(model(x_minus), y)

        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    end

    grad
end

# ============================================================================
# Fairness Properties
# ============================================================================

"""
    DemographicParity(sensitive_attr, threshold)

Predictions should not depend on sensitive attribute.
"""
struct DemographicParity <: Property
    sensitive_attr::Int  # Index of sensitive attribute
    threshold::Float32
end

function check(prop::DemographicParity, model, data)
    # Group predictions by sensitive attribute
    group_0_preds = Float64[]
    group_1_preds = Float64[]

    for (x, _) in data
        pred = mean(model(x))  # Average prediction

        for i in 1:size(x, 1)
            if x[i, prop.sensitive_attr] == 0
                push!(group_0_preds, pred)
            else
                push!(group_1_preds, pred)
            end
        end
    end

    # Check disparity
    disparity = abs(mean(group_0_preds) - mean(group_1_preds))
    disparity < prop.threshold
end

"""
    EqualizedOdds(sensitive_attr, threshold)

True positive and false positive rates should be equal across groups.
"""
struct EqualizedOdds <: Property
    sensitive_attr::Int
    threshold::Float32
end

# ============================================================================
# Property Combinators
# ============================================================================

"""
    AllOf(properties...)

All properties must hold.
"""
struct AllOf <: Property
    properties::Vector{Property}
end

AllOf(props...) = AllOf(collect(props))

function check(prop::AllOf, model, data)
    all(check(p, model, data) for p in prop.properties)
end

"""
    AnyOf(properties...)

At least one property must hold.
"""
struct AnyOf <: Property
    properties::Vector{Property}
end

AnyOf(props...) = AnyOf(collect(props))

function check(prop::AnyOf, model, data)
    any(check(p, model, data) for p in prop.properties)
end

# ============================================================================
# Standard Property Sets
# ============================================================================

"""
Standard properties for classification models.
"""
const CLASSIFICATION_PROPERTIES = AllOf(
    ValidProbabilities(),
    FiniteOutput()
)

"""
Standard properties for regression models.
"""
const REGRESSION_PROPERTIES = FiniteOutput()

"""
Safety-critical properties.
"""
const SAFETY_CRITICAL_PROPERTIES = AllOf(
    FiniteOutput(),
    LocalLipschitz(0.01f0, 0.1f0)
)
