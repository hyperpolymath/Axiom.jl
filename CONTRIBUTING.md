<!-- SPDX-License-Identifier: MPL-2.0 -->
# Contributing

Thank you for your interest in contributing to Axiom.jl.

## How to Contribute

We welcome contributions in many forms:

- **Code:** Improving the layers, verification, serving, or interop code
- **Documentation:** Enhancing docs (see `EXPLAINME.adoc` for a claim-by-claim accounting)
- **Testing:** Adding tests, especially for the known gaps noted in `EXPLAINME.adoc`
- **Bug reports:** Filing clear, reproducible issues

## Getting Started

1. **Read `EXPLAINME.adoc`:** It documents, with file paths, what is implemented and what is known-broken.
2. **Environment:** Julia 1.10+. `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
3. **Tests:** `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Development Workflow

### Branch Naming

```
docs/short-description       # Documentation
test/what-added              # Test additions
feat/short-description       # New features
fix/issue-number-description # Bug fixes
refactor/what-changed        # Code improvements
security/what-fixed          # Security fixes
```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`, `security`

## Reporting Bugs

Before reporting:
1. Search existing issues
2. Check if it's already fixed in `main`

When reporting, include:
- Clear, descriptive title
- Environment details (OS, versions, toolchain)
- Steps to reproduce
- Expected vs actual behaviour

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).
