# Contributing to Axiom.jl

Thank you for your interest in contributing to Axiom.jl! This document provides guidelines for contributing.

## Ways to Contribute

### 1. Report Bugs

Found a bug? Please open an issue with:
- Julia version (`julia --version`)
- Axiom.jl version
- Minimal reproducing example
- Expected vs actual behavior

### 2. Suggest Features

Have an idea? Open an issue with:
- Use case description
- Proposed API
- Why existing solutions don't work

### 3. Improve Documentation

Documentation improvements are always welcome:
- Fix typos
- Clarify explanations
- Add examples
- Translate to other languages

### 4. Write Code

#### Setting Up Development Environment

```bash
# Clone the repo
git clone https://github.com/Hyperpolymath/Axiom.jl.git
cd Axiom.jl

# Start Julia with the project
julia --project=.

# Install dependencies
using Pkg
Pkg.instantiate()

# Run tests
Pkg.test()
```

#### For Rust Backend Development

```bash
cd rust
cargo build
cargo test
```

#### Code Style

**Julia:**
- Use 4-space indentation
- Follow Julia conventions from the [style guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- Document public functions with docstrings

**Rust:**
- Use `rustfmt` for formatting
- Follow Rust API guidelines
- Document public items

#### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'`
5. Commit with clear message: `git commit -m "Add: my feature"`
6. Push: `git push origin feature/my-feature`
7. Open a Pull Request

### Pull Request Guidelines

- One feature/fix per PR
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow existing code style

## Development Areas

### High Priority

- [ ] GPU backend (CUDA, Metal)
- [ ] More layer implementations
- [ ] SMT solver integration for @prove
- [ ] Hugging Face model import
- [ ] Performance benchmarks

### Good First Issues

Look for issues tagged `good-first-issue` on GitHub.

## Questions?

- [GitHub Discussions](https://github.com/Hyperpolymath/Axiom.jl/discussions)
- [Discord](https://discord.gg/axiomjl)

## Code of Conduct

Be respectful and constructive. We're all here to build something great together.
