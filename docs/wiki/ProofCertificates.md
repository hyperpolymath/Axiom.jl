<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# Proof Certificates

> *"Trust, but verify."* - Ronald Reagan

---

## Overview

Axiom.jl generates **cryptographically signed proof certificates** for all verified properties. These certificates provide:

- ✅ **Audit trail**: Complete record of what was proven and how
- ✅ **Reproducibility**: Full SMT query/output for independent verification
- ✅ **Integrity**: SHA256 hash prevents tampering
- ✅ **Portability**: JSON format works with any tooling

Proof certificates are essential for:
- Regulatory compliance (FDA, FAA, automotive safety)
- Bug bounty programs (prove vulnerability claims)
- Research publications (reproducible results)
- MLOps pipelines (CI/CD integration)

---

## Quick Start

### Export a Proof Certificate

```julia
using Axiom

# Verify a property
result = @prove ∀x. relu(x) >= 0

# Export certificate
cert_dict = serialize_proof(
    result,
    "∀x. relu(x) >= 0",
    proof_method="pattern",
    execution_time_ms=0.5
)

export_proof_certificate(cert_dict, "proofs/relu_nonnegative.proof")
```

**Result**: Certificate saved to `proofs/relu_nonnegative.proof`

### Verify a Certificate

```julia
# Load certificate
cert = import_proof_certificate("proofs/relu_nonnegative.proof")

# Verify integrity
if verify_proof_certificate(cert)
    println("✅ Certificate is valid")
else
    println("❌ Certificate has been tampered with")
end
```

---

## Certificate Format

Proof certificates are JSON files with this structure:

```json
{
  "version": "1.0",
  "format": "axiom-proof-certificate",
  "property": "∀x. relu(x) >= 0",
  "result": {
    "status": "proven",
    "counterexample": null,
    "confidence": 1.0,
    "reason": "ReLU is non-negative by definition",
    "suggestions": []
  },
  "metadata": {
    "timestamp": "2025-01-23T12:00:00.000Z",
    "axiom_version": "0.1.0",
    "julia_version": "1.9.0",
    "hostname": "build-server-01"
  },
  "artifacts": {
    "smt_query": "(assert (forall ((x Real)) (>= (relu x) 0)))\n(check-sat)",
    "smt_output": "sat\n...",
    "smt_solver": "z3",
    "proof_method": "symbolic",
    "execution_time_ms": 250.3
  },
  "audit": {
    "hash": "a1b2c3d4e5f6...",
    "hash_algorithm": "SHA256"
  }
}
```

**Key Fields:**

| Field | Description |
|-------|-------------|
| `property` | Mathematical property that was proven |
| `result.status` | `:proven`, `:disproven`, or `:unknown` |
| `result.counterexample` | Example input that violates property (if disproven) |
| `result.confidence` | 0.0-1.0 (1.0 = formal proof, <1.0 = empirical) |
| `artifacts.smt_query` | Full SMT-LIB query sent to solver |
| `artifacts.smt_output` | Raw solver response |
| `artifacts.proof_method` | `"pattern"`, `"symbolic"`, `"smt"`, or `"empirical"` |
| `audit.hash` | SHA256 of canonical certificate representation |

---

## Proof Methods

Axiom.jl uses multiple proof strategies:

### 1. Pattern Matching (`"pattern"`)

**How it works**: Recognizes common patterns (e.g., ReLU is non-negative)
**Confidence**: 1.0 (formal proof)
**Speed**: Instant
**Example**:

```julia
@prove ∀x. relu(x) >= 0  # Pattern: ReLU definition
```

### 2. Symbolic Execution (`"symbolic"`)

**How it works**: Traces code paths symbolically
**Confidence**: 1.0 (formal proof)
**Speed**: Fast (< 1s)
**Example**:

```julia
@prove ∀x ∈ [0, 1]. sigmoid(x) >= 0.5  # Symbolic evaluation
```

### 3. SMT Solving (`"smt"`)

**How it works**: Encodes property as SMT-LIB, calls z3/cvc5/yices
**Confidence**: 1.0 (formal proof if solver returns `sat`)
**Speed**: Slow (1s-30s)
**Example**:

```julia
@prove ∀x y. (x <= y) ⟹ (relu(x) <= relu(y))  # Monotonicity requires SMT
```

### 4. Empirical Testing (`"empirical"`)

**How it works**: Tests property on 10,000 random samples
**Confidence**: 0.9999 (not a proof, high confidence)
**Speed**: Medium (0.1s-1s)
**Example**:

```julia
@prove_empirically ∀x. complex_model(x) ∈ [0, 1]  # Too complex for SMT
```

---

## CI/CD Integration

Axiom.jl includes a GitHub Actions workflow to verify certificates in CI:

### Workflow: `verify-certificates.yml`

**Triggers:**
- PR with changes to `proofs/` directory
- Push to `main` branch
- Manual workflow dispatch

**Jobs:**

1. **Verify Certificates**: Loads all `.proof` and `.json` files in `proofs/`, verifies SHA256 hashes
2. **Check Freshness**: Warns if certificates are >90 days old
3. **Coverage Report**: Compares certified proofs vs. documented properties

**Usage:**

```bash
# Add certificate to repo
mkdir -p proofs
julia --project=. -e '
  using Axiom
  result = @prove ∀x. relu(x) >= 0
  cert = serialize_proof(result, "∀x. relu(x) >= 0", proof_method="pattern")
  export_proof_certificate(cert, "proofs/relu_nonnegative.proof")
'

# Commit and push
git add proofs/relu_nonnegative.proof
git commit -m "proof: add ReLU non-negativity certificate"
git push

# CI automatically verifies certificate
```

**Output Example:**

```
Verifying proofs/relu_nonnegative.proof... ✅ VALID
Verifying proofs/sigmoid_bounded.proof... ✅ VALID

============================================================
Verification Summary:
  ✅ Passed: 2
  ❌ Failed: 0
============================================================

All certificates verified successfully!
```

---

## Security & Tamper Detection

Proof certificates use **SHA256 hashing** to detect tampering:

### How Integrity Verification Works

1. **Certificate Generation**:
   - Compute canonical string: `property + status + counterexample + ...`
   - Hash with SHA256: `hash = sha256(canonical)`
   - Store in `audit.hash` field

2. **Verification**:
   - Recompute canonical string from certificate
   - Recompute hash
   - Compare: `recomputed_hash == stored_hash`

3. **Tamper Detection**:
   - If hashes don't match: certificate was modified
   - CI fails with error message

### What is Protected

✅ **Protected** (included in hash):
- Property expression
- Proof status (proven/disproven/unknown)
- Counterexample (if any)
- Confidence level
- Reason string
- SMT query and output

❌ **Not Protected** (metadata only):
- Timestamp
- Hostname
- Axiom version
- Julia version

**Rationale**: Metadata fields don't affect proof validity, so they're excluded from hash to allow re-runs on different machines.

### Example: Detect Tampering

```julia
# Load certificate
cert = import_proof_certificate("proofs/relu_nonnegative.proof")

# Manually edit file (simulate tampering)
data = JSON.parsefile("proofs/relu_nonnegative.proof")
data["result"]["status"] = "disproven"  # Change proven → disproven
open("proofs/relu_nonnegative.proof", "w") do f
    JSON.print(f, data, 2)
end

# Verify - will fail
cert = import_proof_certificate("proofs/relu_nonnegative.proof")
verify_proof_certificate(cert)  # Returns false - hash mismatch!
```

---

## Best Practices

### 1. Organize Certificates by Property Type

```
proofs/
├── activation-functions/
│   ├── relu_nonnegative.proof
│   ├── sigmoid_bounded.proof
│   └── softmax_sum_to_one.proof
├── robustness/
│   ├── adversarial_l2_norm.proof
│   └── certified_radius.proof
└── numerical-stability/
    ├── no_nan_outputs.proof
    └── finite_gradients.proof
```

### 2. Name Files Descriptively

**Good**:
- `relu_nonnegative.proof`
- `sigmoid_output_range_0_to_1.proof`
- `adversarial_robustness_l_inf_epsilon_0.01.proof`

**Bad**:
- `proof1.proof`
- `test.json`
- `cert.proof`

### 3. Commit Certificates to Git

Proof certificates are small (< 10 KB) and belong in version control:

```bash
git add proofs/
git commit -m "proof: add certificates for activation function properties"
```

**Benefits**:
- Track proof history over time
- Tie proofs to code versions
- CI verifies certificates on every PR

### 4. Regenerate Stale Certificates

Certificates older than 90 days may not reflect current code. CI will warn:

```
⚠️  Warning: 1 certificate(s) are >90 days old:
  - proofs/relu_nonnegative.proof
```

**Fix**: Re-run verification and export new certificate:

```julia
result = @prove ∀x. relu(x) >= 0
cert = serialize_proof(result, "∀x. relu(x) >= 0", proof_method="pattern")
export_proof_certificate(cert, "proofs/relu_nonnegative.proof")
```

### 5. Include SMT Artifacts

Always pass `smt_query` and `smt_output` when available:

```julia
# Good: Full SMT artifacts
cert = serialize_proof(
    result,
    "∀x. relu(x) >= 0",
    smt_query="(assert (forall ((x Real)) (>= (relu x) 0)))\n(check-sat)",
    smt_output="sat\n(model\n  ...\n)",
    smt_solver="z3",
    proof_method="smt",
    execution_time_ms=250.3
)

# Bad: Missing artifacts
cert = serialize_proof(result, "∀x. relu(x) >= 0")  # No SMT data
```

**Why**: SMT artifacts enable independent verification by re-running solver.

---

## Advanced Usage

### Batch Export All Proofs

```julia
using Axiom

# Define properties to prove
properties = [
    "∀x. relu(x) >= 0",
    "∀x ∈ ℝ. sigmoid(x) ∈ (0, 1)",
    "∀input. sum(softmax(input)) == 1.0"
]

# Prove and export each
for prop in properties
    result = @prove eval(Meta.parse(prop))

    # Generate filename from property
    filename = replace(prop, r"[^a-zA-Z0-9]" => "_") * ".proof"
    filepath = "proofs/" * filename

    cert = serialize_proof(result, prop, proof_method="pattern")
    export_proof_certificate(cert, filepath)

    println("✅ Exported: $filepath")
end
```

### Verify All Certificates

```julia
using Axiom

# Find all certificates
cert_files = String[]
for (root, dirs, files) in walkdir("proofs")
    for file in files
        if endswith(file, ".proof") || endswith(file, ".json")
            push!(cert_files, joinpath(root, file))
        end
    end
end

# Verify each
for cert_file in cert_files
    cert = import_proof_certificate(cert_file)

    if verify_proof_certificate(cert)
        println("✅ $cert_file - VALID")
    else
        println("❌ $cert_file - INVALID")
    end
end
```

### Certificate Diff Tool

Compare two versions of a certificate:

```julia
using Axiom
using JSON

function diff_certificates(path1::String, path2::String)
    cert1 = import_proof_certificate(path1)
    cert2 = import_proof_certificate(path2)

    println("Comparing:")
    println("  1: $path1")
    println("  2: $path2")
    println()

    # Compare key fields
    fields = [
        :property, :status, :confidence, :reason,
        :proof_method, :execution_time_ms, :axiom_version
    ]

    for field in fields
        val1 = getfield(cert1, field)
        val2 = getfield(cert2, field)

        if val1 != val2
            println("❌ $field differs:")
            println("    1: $val1")
            println("    2: $val2")
        else
            println("✅ $field: same")
        end
    end
end

# Usage
diff_certificates("proofs/relu_old.proof", "proofs/relu_new.proof")
```

---

## Troubleshooting

### Issue 1: Certificate Import Fails

**Error**: `ERROR: JSON parse error`

**Cause**: Malformed JSON file

**Fix**: Validate JSON:

```bash
julia -e 'using JSON; JSON.parsefile("proofs/cert.proof")'
```

### Issue 2: Hash Verification Fails

**Error**: `Certificate has been tampered with`

**Cause**: File was modified after generation

**Fix**: Regenerate certificate from source:

```julia
result = @prove ∀x. relu(x) >= 0
cert = serialize_proof(result, "∀x. relu(x) >= 0", proof_method="pattern")
export_proof_certificate(cert, "proofs/relu_nonnegative.proof")
```

### Issue 3: CI Verification Times Out

**Error**: `Verification took >5 minutes`

**Cause**: Too many large certificates

**Fix**: Reduce certificate size or split into multiple jobs:

```yaml
# In verify-certificates.yml
- name: Verify certificates
  timeout-minutes: 10  # Increase timeout
```

---

## API Reference

### `serialize_proof`

```julia
serialize_proof(
    result::ProofResult,
    property::String;
    smt_query::Union{String, Nothing} = nothing,
    smt_output::Union{String, Nothing} = nothing,
    smt_solver::Union{String, Nothing} = nothing,
    proof_method::String = "unknown",
    execution_time_ms::Float64 = 0.0
) -> Dict
```

**Parameters:**
- `result`: ProofResult from `@prove`
- `property`: Property expression string
- `smt_query`: Optional SMT-LIB query
- `smt_output`: Optional solver output
- `smt_solver`: Solver name (z3, cvc5, yices)
- `proof_method`: "pattern", "symbolic", "smt", "empirical"
- `execution_time_ms`: Proof time in milliseconds

**Returns:** Dictionary suitable for JSON export

### `export_proof_certificate`

```julia
export_proof_certificate(cert_dict::Dict, path::String)
```

**Parameters:**
- `cert_dict`: Serialized certificate from `serialize_proof`
- `path`: Output file path (`.proof` or `.json`)

**Side Effects:** Creates file at `path`, creates parent directory if needed

### `import_proof_certificate`

```julia
import_proof_certificate(path::String) -> ProofCertificate
```

**Parameters:**
- `path`: Path to certificate file

**Returns:** `ProofCertificate` struct

**Throws:** Error if file doesn't exist or JSON is malformed

### `verify_proof_certificate`

```julia
verify_proof_certificate(cert::ProofCertificate) -> Bool
```

**Parameters:**
- `cert`: Certificate loaded with `import_proof_certificate`

**Returns:** `true` if hash matches, `false` if tampered

---

## Further Reading

- **Serialization Internals**: `src/verification/serialization.jl`
- **CI Workflow**: `.github/workflows/verify-certificates.yml`
- **SMT Properties**: `docs/wiki/AdvancedSMT.md`
- **Verification System**: `docs/wiki/Verification.md`

---

## Summary

| Feature | Status |
|---------|--------|
| JSON export | ✅ Implemented |
| SHA256 integrity | ✅ Implemented |
| SMT artifact capture | ✅ Implemented |
| CI verification | ✅ Implemented |
| Freshness checks | ✅ Implemented |
| Coverage reporting | ✅ Implemented |

**Recommendation**: Always export proof certificates for production models. Store in `proofs/` directory and commit to Git. CI will verify integrity automatically.
