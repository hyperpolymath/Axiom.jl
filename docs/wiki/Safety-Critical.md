<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# Safety-Critical Applications

> Deploying verified machine learning in regulated environments

## Overview

Axiom.jl was designed from the ground up for safety-critical deployments. While other frameworks treat verification as an afterthought, Axiom.jl makes it foundational.

**Key capabilities:**
- Compile-time proof generation
- Runtime invariant checking
- Auditable verification certificates
- Deterministic inference
- Regulatory compliance support

## Regulatory Landscape

### Medical Devices (FDA, EU MDR)

| Requirement | Axiom.jl Feature |
|-------------|------------------|
| Software verification | `@prove` compile-time proofs |
| Risk management | Property-based verification |
| Traceability | Verification certificates |
| Reproducibility | Deterministic backends |
| Change control | Certificate versioning |

### Autonomous Vehicles (ISO 26262)

| ASIL Level | Axiom.jl Approach |
|------------|-------------------|
| ASIL-A | `@ensure` runtime checks |
| ASIL-B | `@prove` + runtime monitoring |
| ASIL-C | Full verification + redundancy |
| ASIL-D | Formal proofs + hardware checks |

### Aerospace (DO-178C)

| DAL Level | Verification Method |
|-----------|---------------------|
| DAL-E | Basic testing |
| DAL-D | `@ensure` assertions |
| DAL-C | Property verification |
| DAL-B | `@prove` formal methods |
| DAL-A | Complete formal verification |

## Verification Levels

### Level 1: Runtime Assertions

The foundation - catch errors before they cause harm:

```julia
@axiom function medical_classifier(x)
    @ensure size(x, 1) == 224 "Image must be 224×224"
    @ensure size(x, 2) == 224 "Image must be 224×224"
    @ensure all(0 .≤ x .≤ 1) "Pixel values must be normalized"

    result = model(x)

    @ensure length(result) == num_conditions "Output must cover all conditions"
    @ensure sum(result) ≈ 1.0 "Probabilities must sum to 1"
    @ensure maximum(result) > confidence_threshold "Must have confident prediction"

    return result
end
```

### Level 2: Property Verification

Prove properties about the model:

```julia
# Define properties
struct BoundedOutputs <: VerificationProperty
    min_val::Float32
    max_val::Float32
end

struct MonotonicInRegion <: VerificationProperty
    region_bounds::Tuple{Float32, Float32}
end

# Verify at model creation
model = @axiom verified=true begin
    Dense(10 => 32, activation=relu)
    Dense(32 => 1, activation=sigmoid)
end

# Prove outputs are in [0, 1]
@prove BoundedOutputs(0.0f0, 1.0f0) model

# Prove monotonicity in a region
@prove MonotonicInRegion((0.0f0, 1.0f0)) model
```

### Level 3: Formal Verification

Mathematical proofs of correctness:

```julia
# Define formal specification
spec = @specification begin
    # Input constraints
    ∀x: input_domain(x) → bounded(x, -10, 10)

    # Output constraints
    ∀x: input_domain(x) → output(model, x) ∈ [0, 1]

    # Robustness
    ∀x, δ: |δ| < ε → |output(model, x + δ) - output(model, x)| < δ_out
end

# Generate proof
proof = verify_specification(model, spec)

# Export for auditors
save_proof(proof, "model_formal_proof.v")  # Coq format
save_proof(proof, "model_formal_proof.smt")  # SMT-LIB format
```

## Verification Certificates

### Certificate Structure

```julia
struct VerificationCertificate
    # Identity
    model_hash::Vector{UInt8}      # SHA-256 of model weights
    certificate_id::UUID
    timestamp::DateTime

    # Verification results
    properties_verified::Vector{Property}
    proofs::Vector{Proof}

    # Environment
    axiom_version::VersionNumber
    backend_version::String
    platform::String

    # Signatures
    verifier_signature::Vector{UInt8}
end
```

### Generating Certificates

```julia
using Axiom.Verification

# Define what to verify
properties = [
    BoundedOutputProperty(-1.0, 1.0),
    LipschitzProperty(10.0),
    InputRobustness(0.01),
]

# Run verification
cert = generate_certificate(model, properties)

# Export
export_certificate(cert, "cert.json")      # JSON for APIs
export_certificate(cert, "cert.pdf")       # PDF for auditors
export_certificate(cert, "cert.xml")       # XML for regulatory systems
```

### Certificate Validation

```julia
# Load and validate certificate
cert = load_certificate("model_v1.cert")

# Verify certificate authenticity
is_valid = validate_certificate(cert)

# Check if model matches certificate
model_matches = verify_model_hash(model, cert)

# Check if properties still hold (rerun verification)
properties_hold = reverify(model, cert.properties_verified)

if is_valid && model_matches && properties_hold
    println("Model deployment approved")
else
    println("Model failed verification")
end
```

## Deterministic Inference

### Why Determinism Matters

In safety-critical systems, the same input must always produce the same output:

```julia
# Enable deterministic mode
Axiom.set_deterministic!(true)

# Now all operations are reproducible
x = rand(Float32, 784)
y1 = forward(model, x)
y2 = forward(model, x)

@assert y1 == y2  # Guaranteed equal
```

### Determinism Settings

```julia
# Global determinism
Axiom.config[:deterministic] = true

# Disable parallel execution (can introduce non-determinism)
Axiom.config[:parallel] = false

# Fix random seeds
Axiom.config[:seed] = 42

# Disable fast math (can change results)
Axiom.config[:fast_math] = false
```

### Backend Determinism

| Backend | Deterministic by Default | Notes |
|---------|-------------------------|-------|
| Julia | Yes | Single-threaded |
| Zig | Configurable | Multi-threaded dispatch can be non-deterministic |
| Coprocessor targets (TPU/NPU/DSP/FPGA) | Strategy-level | Fallback-first; production kernels are backend-specific roadmap work |

## Redundancy Patterns

### N-Version Programming

Run multiple implementations and vote:

```julia
struct RedundantModel
    implementations::Vector{Model}
    voting_strategy::VotingStrategy
end

function forward(rm::RedundantModel, x)
    outputs = [forward(impl, x) for impl in rm.implementations]

    # Voting strategies:
    # - Majority: Most common output
    # - Average: Mean of outputs
    # - Strict: All must agree

    return vote(rm.voting_strategy, outputs)
end

# Create redundant model with different backends
model = RedundantModel([
    @axiom backend=JuliaBackend() begin ... end,
    @axiom backend=ZigBackend() begin ... end,
    @axiom backend=TPUBackend(0) begin ... end,
], MajorityVote())
```

### Watchdog Monitoring

```julia
struct WatchdogMonitor
    timeout::Float64
    fallback::Function
end

function monitored_inference(monitor::WatchdogMonitor, model, x)
    result = Ref{Any}(nothing)
    completed = Ref(false)

    # Inference task
    task = @async begin
        result[] = forward(model, x)
        completed[] = true
    end

    # Watchdog
    deadline = time() + monitor.timeout
    while time() < deadline && !completed[]
        sleep(0.001)
    end

    if completed[]
        return result[]
    else
        # Timeout - use fallback
        @warn "Inference timeout, using fallback"
        return monitor.fallback(x)
    end
end
```

### Self-Checking Pairs

```julia
struct SelfCheckingPair
    forward_model::Model
    checker_model::Model
    tolerance::Float32
end

function safe_forward(scp::SelfCheckingPair, x)
    # Forward pass
    y_forward = forward(scp.forward_model, x)

    # Check pass (could be inverse or verifier)
    y_check = forward(scp.checker_model, x)

    # Verify consistency
    if !isapprox(y_forward, y_check, atol=scp.tolerance)
        error("Self-check failed: forward and checker disagree")
    end

    return y_forward
end
```

## Error Handling

### Graceful Degradation

```julia
struct SafeModel
    primary::Model
    fallback::Model
    error_handler::Function
end

function forward(sm::SafeModel, x)
    try
        result = forward(sm.primary, x)

        # Sanity checks
        if any(isnan, result) || any(isinf, result)
            throw(NumericalError("NaN/Inf in output"))
        end

        return result
    catch e
        # Log error
        @error "Primary model failed" exception=e

        # Attempt fallback
        return forward(sm.fallback, x)
    end
end
```

### Error Classification

```julia
abstract type InferenceError <: Exception end

struct InputValidationError <: InferenceError
    message::String
    input_shape::Tuple
    expected_shape::Tuple
end

struct NumericalError <: InferenceError
    message::String
    location::String
end

struct TimeoutError <: InferenceError
    message::String
    timeout_ms::Float64
    elapsed_ms::Float64
end

struct VerificationError <: InferenceError
    message::String
    property::VerificationProperty
    actual_value::Any
end
```

## Audit Trail

### Logging

```julia
struct AuditLogger
    stream::IO
    log_inputs::Bool
    log_outputs::Bool
    log_timing::Bool
end

function log_inference(logger::AuditLogger, model, x, y, elapsed)
    entry = Dict(
        "timestamp" => now(),
        "model_hash" => bytes2hex(sha256(model)),
        "elapsed_ms" => elapsed * 1000,
    )

    if logger.log_inputs
        entry["input_hash"] = bytes2hex(sha256(x))
        entry["input_shape"] = size(x)
    end

    if logger.log_outputs
        entry["output"] = y
    end

    println(logger.stream, JSON.json(entry))
end
```

### Compliance Reports

```julia
function generate_compliance_report(model, test_suite)
    report = ComplianceReport()

    # Model identification
    report.model_id = model.id
    report.model_version = model.version
    report.model_hash = sha256(model)

    # Verification status
    report.certificate = load_certificate(model.certificate_path)
    report.certificate_valid = validate_certificate(report.certificate)

    # Test results
    for test in test_suite
        result = run_test(model, test)
        push!(report.test_results, result)
    end

    # Generate summary
    report.pass_rate = count(r -> r.passed, report.test_results) / length(report.test_results)
    report.recommendation = report.pass_rate >= 0.99 ? :approve : :reject

    return report
end
```

## Best Practices

### 1. Defense in Depth

```julia
@axiom function ultra_safe_inference(x)
    # Layer 1: Input validation
    @ensure valid_input(x) "Input validation failed"

    # Layer 2: Bounds checking
    @ensure all(-100 .≤ x .≤ 100) "Input out of bounds"

    # Layer 3: Model inference with timeout
    result = with_timeout(1.0) do
        forward(model, x)
    end

    # Layer 4: Output validation
    @ensure valid_output(result) "Output validation failed"

    # Layer 5: Sanity check
    @ensure reasonable_output(result) "Output unreasonable"

    return result
end
```

### 2. Fail-Safe Defaults

```julia
const SAFE_DEFAULT = zeros(Float32, 10)  # Neutral prediction

function safe_predict(model, x)
    try
        return forward(model, x)
    catch
        @warn "Inference failed, returning safe default"
        return SAFE_DEFAULT
    end
end
```

### 3. Continuous Monitoring

```julia
struct RuntimeMonitor
    metrics::Dict{String, Float64}
    alerts::Vector{Alert}
end

function monitor!(mon::RuntimeMonitor, model, x, y)
    # Track statistics
    mon.metrics["inference_count"] = get(mon.metrics, "inference_count", 0) + 1
    mon.metrics["avg_confidence"] = (
        get(mon.metrics, "avg_confidence", 0) * (mon.metrics["inference_count"] - 1)
        + maximum(y)
    ) / mon.metrics["inference_count"]

    # Check for anomalies
    if maximum(y) < 0.5
        push!(mon.alerts, Alert(:low_confidence, y))
    end

    if any(isnan, y) || any(isinf, y)
        push!(mon.alerts, Alert(:numerical_error, y))
    end
end
```

## Certification Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Development                             │
│  1. Design model architecture                                    │
│  2. Train model                                                  │
│  3. Define verification properties                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Verification                                  │
│  4. Run @prove for formal properties                            │
│  5. Generate verification certificate                            │
│  6. Document evidence                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Validation                                    │
│  7. Execute test suite                                          │
│  8. Generate compliance report                                   │
│  9. Independent review                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Deployment                                    │
│  10. Deploy with runtime monitoring                             │
│  11. Enable audit logging                                        │
│  12. Set up alerting                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Maintenance                                   │
│  13. Monitor production metrics                                  │
│  14. Respond to alerts                                           │
│  15. Re-certify on changes                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

*Next: [Framework Comparison](Framework-Comparison.md) to see how Axiom.jl compares to PyTorch and TensorFlow*
