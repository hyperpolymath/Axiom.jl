<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# Advanced SMT Properties

> *"With great expressiveness comes great solver complexity."*

---

## Overview

Axiom.jl supports **advanced SMT properties** including quantifiers, non-linear arithmetic, and complex invariants. These enable proving sophisticated properties about models that go beyond pattern matching.

**Trade-offs:**
- ✅ **Pro**: Can express richer properties (e.g., "for all inputs, output is bounded")
- ⚠️ **Con**: SMT solvers may timeout or return "unknown" for complex properties
- ⚠️ **Con**: Non-linear arithmetic is undecidable in general

---

## Quantifiers

### Universal Quantification (∀)

Prove a property holds **for all** values in a domain.

```julia
@prove ∀x. relu(x) >= 0  # ReLU is always non-negative

@prove ∀x ∈ ℝ. sigmoid(x) ∈ (0, 1)  # Sigmoid is bounded

@prove ∀input. sum(softmax(input)) == 1.0  # Softmax sums to 1
```

**When to use:**
- Activation function properties (bounds, monotonicity)
- Probability distribution constraints
- Output invariants

**Solver support:**
- ✅ z3: Full support for linear quantifiers
- ✅ cvc5: Good support with heuristics
- ⚠️ yices: Limited quantifier support
- ✗ mathsat: No quantifier support in free version

### Existential Quantification (∃)

Prove there **exists** at least one value satisfying a property.

```julia
@prove ∃x. relu(x) > 0  # ReLU can be positive

@prove ∃ε. norm(model(x) - model(x + ε)) > δ  # Adversarial example exists

@prove ∃x. prediction(x) == target  # Solution exists
```

**When to use:**
- Adversarial robustness (finding counterexamples)
- Satisfiability checking
- Optimization feasibility

**Note**: Existential quantifiers often convert to satisfiability (SAT) problems, which SMT solvers handle efficiently.

---

## Quantifier Patterns

### Nested Quantifiers

```julia
# For all inputs, there exists an ε that doesn't change the prediction
@prove ∀x. ∃ε. (norm(ε) < 0.01) ⟹ (argmax(model(x)) == argmax(model(x + ε)))

# Lipschitz continuity
@prove ∀x y. norm(model(x) - model(y)) <= L * norm(x - y)
```

**Complexity warning:**
- Nested quantifiers increase solver complexity exponentially
- Consider splitting into multiple simpler properties
- Use pattern matching triggers when possible

### Bounded Quantification

Restrict quantifier domain to improve solver performance:

```julia
# Bounded domain
@prove ∀x ∈ [0, 1]. sigmoid(x) >= 0.5

# Conditional properties
@prove ∀x. (x > 0) ⟹ (log(x) < x)

# Finite sets
@prove ∀class ∈ {0, 1, ..., 9}. probability[class] >= 0
```

---

## Non-Linear Arithmetic

### Supported Operations

SMT solvers have varying support for non-linear operations:

| Operation | z3 | cvc5 | yices | Notes |
|-----------|----|----|-------|-------|
| Multiplication (x * y) | ✅ | ✅ | Partial | Avoid nested multiplication |
| Division (x / y) | ✅ | ✅ | ✗ | Check y ≠ 0 |
| Exponentiation (x^n) | Partial | Partial | ✗ | Integer n only |
| Logarithm (log x) | ✗ | ✗ | ✗ | Use approximations |
| Trigonometric (sin, cos) | ✗ | ✗ | ✗ | Not directly supported |

### Linear Approximations

For non-linear operations, use **piecewise linear approximations**:

```julia
# Instead of:
@prove ∀x. exp(x) > 0  # Non-linear, may timeout

# Use Taylor approximation:
@prove ∀x ∈ [-1, 1]. taylor_exp(x, degree=3) > 0  # Linear constraints

# Or interval arithmetic:
@prove ∀x ∈ [a, b]. exp_lower_bound(x) > 0
```

### Polynomial Invariants

Polynomial constraints are decidable for bounded degrees:

```julia
# Quadratic invariant
@prove ∀x y. (x^2 + y^2 <= 1) ⟹ (model_output(x, y) ∈ [0, 1])

# Degree-2 Lyapunov function
@prove ∀state. V(state) >= 0 ∧ V(0) == 0
```

**Guidelines:**
- Keep polynomial degree ≤ 2 for reliability
- Degree 3-4 may work but expect longer solve times
- Degree ≥ 5 often leads to "unknown" results

---

## Complex Invariants

### Robustness Properties

```julia
# Adversarial robustness (L∞ ball)
@prove ∀x ε. (norm(ε, Inf) < δ) ⟹
            (argmax(model(x)) == argmax(model(x + ε)))

# Certified defense radius
@prove ∀x. certified_radius(model, x) >= ε_min
```

### Distributional Properties

```julia
# Output distribution constraints
@prove ∀input. let output = model(input)
    sum(output) ≈ 1.0 ∧
    all(output .>= 0) ∧
    entropy(output) >= min_entropy
end
```

### Compositional Invariants

```julia
# Property preservation through layers
@prove ∀x. let
    h1 = layer1(x)
    h2 = layer2(h1)
    h3 = layer3(h2)
in
    (bounded(h1, 0, 1) ∧ bounded(h2, 0, 1)) ⟹ bounded(h3, 0, 1)
end
```

---

## Solver Strategy Selection

Different solvers excel at different property types:

### z3 (Recommended for Most Cases)

```bash
export AXIOM_SMT_SOLVER_KIND=z3
```

**Best for:**
- Universal quantifiers with linear arithmetic
- Bit-vector reasoning
- Complex boolean combinations

**Limitations:**
- Non-linear arithmetic can be slow
- May produce "unknown" for nested quantifiers

### cvc5 (Alternative for Quantifiers)

```bash
export AXIOM_SMT_SOLVER_KIND=cvc5
```

**Best for:**
- Quantifier instantiation heuristics
- String constraints
- Algebraic datatypes

**Limitations:**
- Less mature than z3
- Fewer optimization heuristics

### yices (Fast for Linear Properties)

```bash
export AXIOM_SMT_SOLVER_KIND=yices
```

**Best for:**
- Linear arithmetic (very fast)
- SAT problems
- Large conjunctions of linear constraints

**Limitations:**
- No quantifier support
- No non-linear arithmetic

---

## Tooling Guidance

### When to Use Advanced Properties

**Use advanced properties when:**
1. ✅ Simple pattern matching is insufficient
2. ✅ Property requires reasoning about all inputs
3. ✅ You need formal certification (safety-critical systems)
4. ✅ Property is linear or low-degree polynomial

**Avoid advanced properties when:**
1. ✗ Property involves transcendental functions (sin, exp, log)
2. ✗ High-degree polynomials (≥ 5)
3. ✗ Deeply nested quantifiers (∀∃∀∃...)
4. ✗ Solver consistently returns "unknown"

### Fallback Strategies

If SMT solver fails, try these alternatives:

```julia
# 1. Simplify property
@prove ∀x ∈ [0, 1]. relu(x) >= 0  # Restrict domain

# 2. Split into cases
@prove ∀x. (x < 0) ⟹ (relu(x) == 0)
@prove ∀x. (x >= 0) ⟹ (relu(x) == x)

# 3. Use empirical verification
@prove_empirically ∀x. complex_property(x)  # Test on samples

# 4. Manual proof certificate
@assume ∀x. property(x)  # Document external proof
```

### Performance Optimization

**Techniques to speed up advanced SMT queries:**

1. **Domain restriction**: Bound quantifier ranges
2. **Incremental solving**: Break into subproblems
3. **Pattern triggers**: Guide quantifier instantiation
4. **Preprocessing**: Simplify expressions before SMT
5. **Parallelization**: Run multiple solvers simultaneously

```julia
# Bad: Unbounded domain
@prove ∀x. complex_property(x)  # May timeout

# Good: Bounded domain
@prove ∀x ∈ [-100, 100]. complex_property(x)  # Faster

# Better: Multiple small ranges
@prove ∀x ∈ [-10, -1]. complex_property(x)
@prove ∀x ∈ [-1, 1]. complex_property(x)
@prove ∀x ∈ [1, 10]. complex_property(x)
```

---

## Examples

### Example 1: ReLU Monotonicity

```julia
# Property: ReLU is monotonically increasing
@prove ∀x y. (x <= y) ⟹ (relu(x) <= relu(y))
```

**Solver result**: ✓ Proven (z3, 0.3s)

### Example 2: Softmax Lipschitz Continuity

```julia
# Property: Softmax is Lipschitz continuous
@prove ∀x y. norm(softmax(x) - softmax(y), 2) <= norm(x - y, 2)
```

**Solver result**: ⚠️ Unknown (non-linear norm, timeout after 30s)
**Fallback**: Use empirical testing or manual proof

### Example 3: Adversarial Robustness

```julia
# Property: Model is robust to small perturbations
@prove ∀x ε. (norm(ε, Inf) < 0.01) ⟹
            (argmax(classifier(x)) == argmax(classifier(x + ε)))
```

**Solver result**: ⚠️ Unknown for general neural networks
**Alternative**: Use specialized tools (e.g., CROWN, α,β-CROWN)

### Example 4: Bounded Activation

```julia
# Property: Sigmoid output is strictly bounded
@prove ∀x ∈ ℝ. 0 < sigmoid(x) < 1
```

**Solver result**: ✓ Proven (z3, 1.2s) using domain theory

---

## Debugging Advanced Properties

### Common Issues

**Issue 1: Solver returns "unknown"**

```julia
# Check: Is property too complex?
@prove ∀x. sin(x^3 + log(x)) > 0  # Too complex

# Fix: Simplify or use approximations
@prove ∀x ∈ [1, 10]. x^3 > 0  # Simpler
```

**Issue 2: Solver times out**

```julia
# Check: Is domain too large?
@prove ∀x ∈ [-10^6, 10^6]. property(x)  # Huge range

# Fix: Restrict domain
@prove ∀x ∈ [-10, 10]. property(x)  # Reasonable range
```

**Issue 3: Counterexample found unexpectedly**

```julia
# Check: Is property actually false?
@prove ∀x. x^2 >= 0  # True
@prove ∀x. x^2 > 0   # False! (counterexample: x=0)

# Fix: Correct the property
@prove ∀x. (x ≠ 0) ⟹ (x^2 > 0)  # True
```

### Verbose Mode

Enable verbose logging to debug SMT queries:

```bash
export AXIOM_SMT_VERBOSE=1
export AXIOM_SMT_SAVE_QUERIES=/tmp/axiom_smt_queries/
```

This saves SMT-LIB files for manual inspection:

```bash
ls /tmp/axiom_smt_queries/
# property_0001.smt2
# property_0002.smt2

# Manually run solver
z3 /tmp/axiom_smt_queries/property_0001.smt2
```

---

## Further Reading

- **SMT-LIB Standard**: http://smtlib.cs.uiowa.edu/
- **z3 Guide**: https://microsoft.github.io/z3guide/
- **cvc5 Documentation**: https://cvc5.github.io/
- **Verified Deep Learning**: Survey of formal verification techniques
- **Neural Network Verification**: CROWN, DeepPoly, Marabou tools

---

## Summary

| Feature | Complexity | Solver Support | Use Case |
|---------|------------|----------------|----------|
| Linear quantifiers | Medium | ✅ Good | Activation bounds, probability constraints |
| Non-linear arithmetic | High | ⚠️ Limited | Polynomial invariants (degree ≤ 2) |
| Nested quantifiers | Very High | ⚠️ Poor | Avoid if possible, split into simpler properties |
| Transcendental functions | Undecidable | ✗ None | Use approximations or specialized tools |

**Recommendation**: Start with simple linear properties. Gradually add complexity only when needed, and always test solver performance on representative examples.
