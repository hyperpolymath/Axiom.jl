# Verification System

> *"Trust, but verify. Actually, just verify."*

---

## Overview

Axiom.jl's verification system lets you **prove properties** about your models. This isn't just testing - it's mathematical certainty.

```julia
@axiom Classifier begin
    # ...

    # These aren't just assertions - they're GUARANTEES
    @ensure sum(output) ≈ 1.0      # Runtime check
    @prove ∀x. output(x) ∈ [0, 1]  # Mathematical proof
end
```

---

## The Verification Pyramid

```
                    ╱╲
                   ╱  ╲
                  ╱    ╲
                 ╱ @prove╲        <- Mathematical proofs
                ╱________╲
               ╱          ╲
              ╱  @ensure   ╲     <- Runtime assertions
             ╱______________╲
            ╱                ╲
           ╱  Shape Checking  ╲  <- Compile-time types
          ╱____________________╲
```

Each level provides stronger guarantees:

| Level | Catches | When | Cost |
|-------|---------|------|------|
| Shape Checking | Type errors | Compile time | Free |
| @ensure | Invariant violations | Runtime | Small |
| @prove | Logic errors | Compile time | Variable |

---

## @ensure: Runtime Assertions

### Basic Usage

```julia
@axiom Model begin
    # ...

    @ensure condition "error message"
end
```

### Common Patterns

```julia
@axiom SafeClassifier begin
    input :: Image(224, 224, 3)
    output :: Probabilities(1000)

    # ... layers ...

    # Probability constraints
    @ensure sum(output) ≈ 1.0 "Probabilities must sum to 1"
    @ensure all(output .>= 0) "Probabilities must be non-negative"
    @ensure all(output .<= 1) "Probabilities must be <= 1"

    # Numerical stability
    @ensure !any(isnan, output) "Output contains NaN"
    @ensure !any(isinf, output) "Output contains Inf"

    # Confidence bounds
    @ensure maximum(output) >= 0.1 "Prediction too uncertain"
end
```

### Built-in Ensure Functions

```julia
# Instead of writing manual checks...
@ensure sum(output) ≈ 1.0
@ensure all(output .>= 0)

# Use the built-in:
@ensure valid_probabilities(output)

# Other built-ins:
@ensure no_nan(output)
@ensure no_inf(output)
@ensure finite(output)
@ensure bounded(output, 0, 1)
@ensure normalized(output)  # L2 norm = 1
```

### Conditional Ensures

```julia
@axiom Model begin
    # Only check during training
    @ensure training || valid_probabilities(output)

    # Check gradient bounds during training
    @ensure !training || gradient_bounded(grads, 10.0)
end
```

---

## @prove: Formal Proofs

### What Can Be Proven?

The `@prove` macro attempts to **mathematically prove** properties about your model.

```julia
@axiom Model begin
    # ...

    # These are PROVEN, not just tested
    @prove ∀x. sum(softmax(x)) == 1.0
    @prove ∀x. all(sigmoid(x) .∈ [0, 1])
    @prove ∀x. relu(x) >= 0
end
```

### How It Works

1. **Pattern Matching**: Known properties of functions (e.g., softmax always sums to 1)
2. **Symbolic Execution**: Trace computation symbolically
3. **SMT Solvers**: Use Z3/CVC5 for complex properties
4. **Fallback**: If unprovable, becomes runtime assertion

### SMT Solver Configuration

Axiom uses the bundled `packages/SMTLib.jl` adapter to talk to external solvers
(`z3`, `cvc5`, `yices`, `mathsat`) when available. You can tune behavior with
environment variables:

- `AXIOM_SMT_SOLVER` (e.g., `z3`, `cvc5`)
- `AXIOM_SMT_SOLVER_PATH` + `AXIOM_SMT_SOLVER_KIND`
- `AXIOM_SMT_TIMEOUT_MS` (default: 30000)
- `AXIOM_SMT_LOGIC` (default: `QF_NRA`)
- `AXIOM_SMT_RUNNER=rust` to execute the solver via the Rust backend runner
- `AXIOM_SMT_CACHE=1` to enable SMT result caching
- `AXIOM_SMT_CACHE_MAX` to cap cache entries (default: 128)

### Rust Runner Example

```bash
export AXIOM_SMT_RUNNER=rust
export AXIOM_RUST_LIB=/path/to/libaxiom_core.so
export AXIOM_SMT_SOLVER=z3
```

```julia
@prove ∃x. x > 0
```

### Proof Status

```julia
@axiom Model begin
    @prove ∀x. sum(softmax(x)) == 1.0  # Proven by definition
end

# Output during compilation:
# ✓ Property proven: ∀x. sum(softmax(x)) == 1.0
#   Proof: By definition of softmax
```

```julia
@axiom Model begin
    @prove ∀x. custom_function(x) > 0  # Can't prove
end

# Output during compilation:
# ⚠ Cannot prove property: ∀x. custom_function(x) > 0
#   Adding runtime assertion instead.
#   Consider: Provide proof hints or simplify property
```

### Proof Syntax

```julia
# Universal quantification
@prove ∀x. property(x)

# Existential quantification
@prove ∃x. property(x)

# Implication
@prove condition ⟹ consequence

# Bounded quantification
@prove ∀x ∈ [0, 1]. property(x)

# Multiple variables
@prove ∀x y. property(x, y)

# Epsilon-delta (robustness)
@prove ∀x ε. (norm(ε) < δ) ⟹ close(f(x), f(x + ε))
```

### Robustness Proofs

```julia
@axiom RobustClassifier begin
    # ...

    # Local Lipschitz continuity
    @prove ∀x ε. (norm(ε) < 0.01) ⟹ (norm(f(x + ε) - f(x)) < 0.1)

    # Adversarial robustness
    @prove ∀x ε. (norm(ε) < 0.03) ⟹ (argmax(f(x)) == argmax(f(x + ε)))
end
```

---

## verify() Function

For post-hoc verification of existing models:

```julia
using Axiom

# Load a model (from PyTorch, ONNX, or Axiom)
model = from_pytorch("model.pth")

# Verify properties
result = verify(model,
    properties = [
        ValidProbabilities(),
        FiniteOutput(),
        NoNaN(),
        LocalLipschitz(0.01, 0.1)
    ],
    data = test_loader
)

println(result)
```

Output:
```
Verification Result: ✓ PASSED

Properties checked: 4
  ✓ ValidProbabilities
  ✓ FiniteOutput
  ✓ NoNaN
  ✓ LocalLipschitz(ε=0.01, δ=0.1)

Runtime: 2.34s
```

### Verification Modes

```julia
# Quick check (basic properties)
verify(model, mode=QUICK)

# Standard (default)
verify(model, mode=STANDARD)

# Thorough (extensive testing)
verify(model, mode=THOROUGH)

# Exhaustive (for safety-critical)
verify(model, mode=EXHAUSTIVE)
```

### Custom Properties

```julia
# Define custom property
struct MyProperty <: Property
    threshold::Float32
end

function check(prop::MyProperty, model, data)
    for (x, _) in data
        output = model(x)
        if maximum(output) < prop.threshold
            return false
        end
    end
    return true
end

# Use it
result = verify(model, properties=[MyProperty(0.5)])
```

---

## Verification Certificates

For regulatory compliance, generate formal certificates:

```julia
# Verify model
result = verify(model,
    properties = SAFETY_CRITICAL_PROPERTIES,
    data = test_data
)

# Generate certificate
if result.passed
    cert = generate_certificate(model, result,
        model_name = "MedicalDiagnosisAI",
        verifier_id = "FDA-Submission-2024"
    )

    # Display certificate
    println(cert)

    # Save for submission
    save_certificate(cert, "fda_certificate.cert")
end
```

Output:
```
╔══════════════════════════════════════════╗
║   AXIOM.JL VERIFICATION CERTIFICATE      ║
╠══════════════════════════════════════════╣
║ Model: MedicalDiagnosisAI                ║
║ Hash:  a3f2c9d8e1b4...                   ║
║                                          ║
║ Verified Properties:                     ║
║   ✓ ValidProbabilities                   ║
║   ✓ FiniteOutput                         ║
║   ✓ NoNaN                                ║
║   ✓ LocalLipschitz                       ║
║   ✓ AdversarialRobust                    ║
║                                          ║
║ Proof Type: empirical + static           ║
║ Axiom Version: 0.1.0                     ║
╚══════════════════════════════════════════╝
```

---

## Property Reference

### Output Properties

| Property | Description | Provable? |
|----------|-------------|-----------|
| `ValidProbabilities()` | sum=1, all ∈ [0,1] | If ends with Softmax |
| `BoundedOutput(lo, hi)` | all ∈ [lo, hi] | If ends with bounded activation |
| `FiniteOutput()` | No NaN or Inf | Usually |
| `NoNaN()` | No NaN values | Usually |
| `NoInf()` | No Inf values | Usually |

### Robustness Properties

| Property | Description | Provable? |
|----------|-------------|-----------|
| `LocalLipschitz(ε, δ)` | \|f(x+ε) - f(x)\| < δ | Sometimes |
| `AdversarialRobust(ε)` | Prediction stable under perturbation | Sometimes |

### Fairness Properties

| Property | Description | Provable? |
|----------|-------------|-----------|
| `DemographicParity(attr, threshold)` | Equal prediction rates | Empirical |
| `EqualizedOdds(attr, threshold)` | Equal TPR/FPR | Empirical |

---

## Best Practices

### 1. Start with @ensure, Graduate to @prove

```julia
# Start here: runtime checks
@ensure valid_probabilities(output)

# Then try: formal proofs
@prove ∀x. valid_probabilities(output(x))
```

### 2. Layer Your Guarantees

```julia
@axiom Model begin
    # Level 1: Basic sanity
    @ensure finite(output)

    # Level 2: Domain constraints
    @ensure valid_probabilities(output)

    # Level 3: Safety requirements
    @ensure confidence(output) >= 0.7

    # Level 4: Formal properties
    @prove ∀x. bounded(output(x), 0, 1)
end
```

### 3. Verify Before Deployment

```julia
# Always verify with production-like data
result = verify(model,
    properties = PRODUCTION_REQUIREMENTS,
    data = production_validation_set,
    mode = EXHAUSTIVE
)

if !result.passed
    error("Model failed verification - DO NOT DEPLOY")
end
```

### 4. Generate Certificates

```julia
# For audit trail
cert = generate_certificate(model, result)
save_certificate(cert, "deployment_$(today()).cert")
```

---

## Next Steps

- [Formal Proofs Deep Dive](Formal-Proofs.md) - Advanced proof techniques
- [Safety-Critical Applications](Safety-Critical.md) - Medical, automotive, etc.
- [Custom Properties](Custom-Properties.md) - Define your own
- [Verification API](../api/verification.md) - Complete reference
