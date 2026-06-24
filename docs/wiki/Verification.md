<!-- SPDX-License-Identifier: CC-BY-SA-4.0 -->
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
- `AXIOM_SMT_RUNNER=zig` to execute the solver via the Zig backend runner
- `AXIOM_SMT_CACHE=1` to enable SMT result caching
- `AXIOM_SMT_CACHE_MAX` to cap cache entries (default: 128)

### Execution Modes

Axiom supports three SMT execution modes:

| Mode | Environment Variable | Security | Performance | Use Case |
|------|---------------------|----------|-------------|----------|
| **Julia (default)** | None | Basic | Fast | Development |
| **Containerized** | `AXIOM_SMT_RUNNER=container` | **High** | Medium | Production |
| **Zig** | `AXIOM_SMT_RUNNER=zig` | Medium | Fast | Embedded |

**Recommendation:** Use `container` mode for production deployments requiring maximum security.

### Julia Mode (Default)

By default, Axiom uses the Julia SMT path (SMTLib) and does not require Rust or containers.

```julia
@prove ∃x. x > 0
```

**Security:** Basic (no isolation, trusts solver binary)

### Containerized SMT Execution (Recommended for Production)

For maximum security, run SMT solvers in isolated containers using svalinn/vordr.

**Benefits:**
- Process isolation (PID, network, mount namespaces)
- Resource limits (2GB RAM, 2 CPU cores)
- Reproducible solver builds via Guix
- Supply chain verification

**Setup:**

```bash
# Build the container (one-time setup)
cd verified-container-spec/examples/axiom-smt-runner
podman build -t axiom-smt-runner:latest -f Containerfile .

# Or use pre-built image
podman pull ghcr.io/hyperpolymath/axiom-smt-runner:latest
```

**Usage:**

```bash
# Enable containerized execution
export AXIOM_SMT_RUNNER=container
export AXIOM_SMT_CONTAINER_IMAGE=axiom-smt-runner:latest
export AXIOM_CONTAINER_RUNTIME=podman  # or 'svalinn' or 'docker'
```

```julia
using Axiom

# SMT solvers now run in isolated containers automatically
@prove ∀x. x > 0 ⟹ (x + 1) > 0
```

**With Svalinn/Vordr (Advanced):**

```bash
# Use Svalinn for attestation verification
export AXIOM_CONTAINER_RUNTIME=svalinn
export AXIOM_SMT_POLICY=/path/to/svalinn-policy.json
```

### Optional Zig Runner Example

The Zig runner is optional and only used when explicitly enabled.

```bash
export AXIOM_SMT_RUNNER=zig
export AXIOM_ZIG_LIB=/path/to/libaxiom_zig.so
export AXIOM_SMT_SOLVER=z3
```

```julia
@prove ∃x. x > 0
```

### Cache Example (Julia-only)

```bash
export AXIOM_SMT_CACHE=1
export AXIOM_SMT_CACHE_MAX=64
```

```julia
@prove ∀x. x > 0 ⟹ (x + 1) > 0
@prove ∀x. x > 0 ⟹ (x + 1) > 0  # second call hits cache
```

---

## SMT Runner Security Hardening

### Quick Start Security Setup

**Essential 3-step setup for secure SMT verification:**

```bash
# 1. Set timeout (REQUIRED - prevents infinite loops)
export AXIOM_SMT_TIMEOUT_MS=30000

# 2. Choose allow-listed solver (REQUIRED)
export AXIOM_SMT_SOLVER=z3  # or cvc5, yices, mathsat

# 3. Enable caching (RECOMMENDED - reduces solver invocations)
export AXIOM_SMT_CACHE=1
```

**Verify your configuration:**

```julia
using Axiom

# Print security report
print_smt_security_report()

# Or get programmatic report
report = verify_smt_security_config()
```

### Security Checklist

When using external SMT solvers for formal verification, follow these security best practices:

#### 1. Solver Allow-List ✓

**What**: Only use vetted, well-known SMT solvers.

**Why**: Prevents execution of arbitrary binaries disguised as SMT solvers.

**Implementation**: Axiom enforces an allow-list of trusted solvers:

```julia
# In prove.jl:
const SMT_ALLOWLIST = Set([:z3, :cvc5, :yices, :mathsat])
```

**User Action**:
- ✓ Only install SMT solvers from official sources
- ✓ Verify checksums/signatures of downloaded binaries
- ✗ Never use custom or unknown solvers without code review

#### 2. Timeout Configuration ✓

**What**: Always set a timeout to prevent infinite loops.

**Why**: Malicious or malformed SMT queries can hang solvers indefinitely.

**Implementation**:

```bash
# Set timeout (default: 30 seconds)
export AXIOM_SMT_TIMEOUT_MS=30000

# For quick proofs
export AXIOM_SMT_TIMEOUT_MS=5000

# For complex proofs (max recommended: 5 minutes)
export AXIOM_SMT_TIMEOUT_MS=300000
```

**Best Practices**:
- ✓ Start with 5-10 seconds for simple properties
- ✓ Increase incrementally if needed
- ✗ Never disable timeouts (set to 0)
- ✗ Don't exceed 5 minutes without good reason

#### 3. Path Validation ✓

**What**: Validate solver binary paths before execution.

**Why**: Prevents path traversal attacks and execution of unintended binaries.

**Implementation**:

```bash
# Explicit solver path (use absolute paths)
export AXIOM_SMT_SOLVER_PATH=/usr/local/bin/z3
export AXIOM_SMT_SOLVER_KIND=z3

# OR: Let Axiom auto-detect (safer)
unset AXIOM_SMT_SOLVER_PATH
```

**User Action**:
- ✓ Use absolute paths, not relative paths
- ✓ Verify the binary is the expected solver: `z3 --version`
- ✗ Never use paths from untrusted sources
- ✗ Don't use paths containing `..`, `~`, or shell expansions

#### 4. Result Caching ✓

**What**: Cache SMT solver results to avoid redundant solver invocations.

**Why**: Reduces solver execution frequency, minimizing attack surface.

**Implementation**:

```bash
# Enable caching (disabled by default)
export AXIOM_SMT_CACHE=1

# Limit cache size (default: 128 entries)
export AXIOM_SMT_CACHE_MAX=128
```

**Security Benefits**:
- Fewer solver executions = smaller attack window
- Cached results are deterministic and pre-validated
- Cache key includes solver path and script, preventing poisoning

**Trade-offs**:
- ✓ Faster verification on repeated properties
- ✓ Reduced solver invocations
- ✗ Uses additional memory (bounded by CACHE_MAX)

#### 5. Optional Zig Runner 🔒

**What**: Execute SMT solvers through a Zig subprocess manager.

**Why**: Provides additional sandboxing and resource limits.

**Implementation**:

```bash
# Enable Zig runner (optional, requires Zig backend)
export AXIOM_SMT_RUNNER=zig
export AXIOM_ZIG_LIB=/path/to/libaxiom_zig.so

# Configure Zig runner
export AXIOM_ZIG_SANDBOX=strict  # Future: seccomp, namespaces
```

**Security Benefits**:
- Zig memory safety prevents buffer overflows in runner code
- Future: Process isolation, resource limits, seccomp filters
- Centralized auditing of solver invocations

**When to Use**:
- ✓ High-security environments
- ✓ Untrusted input properties
- ✓ Safety-critical applications (medical, automotive)
- ✗ Development/testing (adds overhead)

---

### Security Verification Checklist

Before deploying verification in production, use the built-in security verification tools:

```julia
using Axiom

# Method 1: Print full security report
print_smt_security_report()

# Method 2: Get programmatic report
report = verify_smt_security_config()
if !isempty(report["warnings"])
    @error "Security warnings detected" warnings=report["warnings"]
end

# Method 3: Print quick checklist
smt_security_checklist()
```

**Manual verification steps:**

```bash
# 1. Verify solver is allow-listed
julia -e 'using Axiom; @show Axiom.SMT_ALLOWLIST'

# 2. Check timeout is set
echo $AXIOM_SMT_TIMEOUT_MS  # Should be 5000-300000

# 3. Verify solver binary
which z3
z3 --version

# 4. Test timeout works
export AXIOM_SMT_TIMEOUT_MS=1000
julia -e '@prove ∀x. very_complex_property(x)'  # Should timeout

# 5. Enable caching in production
export AXIOM_SMT_CACHE=1
```

---

### Threat Model & Mitigations

| Threat | Mitigation | Status |
|--------|------------|--------|
| **Malicious solver binary** | Allow-list only trusted solvers | ✓ Enforced |
| **Infinite loop in solver** | Mandatory timeouts | ✓ Enforced |
| **Path traversal attack** | Absolute path validation | ✓ Enforced |
| **Resource exhaustion** | Timeout + cache limits | ✓ Enforced |
| **Cache poisoning** | Hash includes solver + script | ✓ Enforced |
| **Sandbox escape** | Zig runner isolation | 🚧 Planned |
| **Supply chain attack** | Checksum verification | ⚠ User responsibility |

---

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

# Load a model (from PyTorch descriptor, ONNX export pipeline, or Axiom)
model = from_pytorch("model.pytorch.json")

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
