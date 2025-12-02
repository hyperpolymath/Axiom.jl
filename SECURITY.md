# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues via:

1. **Email**: security@axiom-jl.org (preferred)
2. **GitLab Security Advisory**: Use the confidential issue feature
3. **PGP Encrypted Email**: See `.well-known/security.txt` for our PGP key

### What to Include

Please include the following in your report:

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested fixes (optional)

### Response Timeline

| Action | Timeline |
|--------|----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 7 days |
| Status update | Every 14 days |
| Fix release | Depends on severity |

### Severity Levels

| Severity | Description | Target Fix Time |
|----------|-------------|-----------------|
| Critical | Remote code execution, data breach | 24-72 hours |
| High | Privilege escalation, significant data exposure | 7 days |
| Medium | Limited data exposure, DoS | 30 days |
| Low | Minor issues, hardening | 90 days |

## Security Measures

### Code Security

- **Memory Safety**: Rust and Zig backends provide memory safety guarantees
- **Type Safety**: Julia's type system prevents many classes of bugs
- **Formal Verification**: `@prove` macro enables mathematical correctness proofs
- **Runtime Checks**: `@ensure` macro validates invariants at runtime

### Supply Chain Security

- **Dependency Auditing**: Regular `cargo audit` and `cargo deny` checks
- **SBOM Generation**: Software Bill of Materials available
- **Reproducible Builds**: Nix flake ensures reproducibility
- **Signed Releases**: All releases are signed with GPG

### Development Practices

- **Code Review**: All changes require review
- **CI/CD Security**: Automated security scanning in pipeline
- **Secrets Management**: No secrets in repository
- **SPDX Headers**: All source files have license headers

## Security Features for Users

### Verification System

Axiom.jl provides built-in security features for ML models:

```julia
# Runtime bounds checking
@ensure all(0 .≤ output .≤ 1) "Output must be valid probabilities"

# Formal verification
@prove BoundedOutputs(0.0, 1.0) model

# Verification certificates
cert = generate_certificate(model, properties)
```

### Deterministic Inference

For safety-critical applications:

```julia
Axiom.set_deterministic!(true)  # Reproducible results
```

### Input Validation

```julia
@ensure valid_input(x) "Input validation failed"
@ensure no_nan(x) "Input contains NaN values"
```

## Security Advisories

Security advisories are published at:

- GitHub Security Advisories
- `.well-known/security.txt`
- Mailing list (security-announce@axiom-jl.org)

## Acknowledgments

We thank the following security researchers for responsible disclosure:

*No vulnerabilities reported yet.*

## Contact

- Security Team: security@axiom-jl.org
- PGP Key: See `.well-known/security.txt`
- Response Team: See `MAINTAINERS.md`

---

*This security policy follows [RFC 9116](https://www.rfc-editor.org/rfc/rfc9116) and RSR security standards.*
