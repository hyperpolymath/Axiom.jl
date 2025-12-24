# Changelog

All notable changes to Axiom.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Zig backend with SIMD-optimized operations
- Comprehensive documentation (Architecture, Performance, Safety-Critical, etc.)
- RSR compliance infrastructure (flake.nix, Justfile, security policies)
- Attention mechanisms for transformer models
- Flash attention implementation
- Rotary position embeddings

### Changed
- Updated Home.md with complete documentation map
- Enhanced verification system documentation

### Security
- Added SECURITY.md with vulnerability disclosure process
- Added .well-known/security.txt (RFC 9116 compliant)

## [0.1.0] - 2025-01-01

### Added
- Initial release of Axiom.jl framework
- Julia DSL with `@axiom` macro for model definition
- Compile-time shape verification via parametric `Tensor{T, Shape}` type
- Runtime verification with `@ensure` macro
- Formal verification with `@prove` macro
- Verification certificates for regulatory compliance

#### Layers
- Dense (fully connected)
- Conv2D, Conv1D (convolution)
- MaxPool2D, AvgPool2D (pooling)
- GlobalAvgPool2D, GlobalMaxPool2D
- BatchNorm, LayerNorm, GroupNorm, InstanceNorm (normalization)
- Dropout
- Flatten, Reshape
- Sequential container

#### Activations
- ReLU, ReLU6, LeakyReLU, ELU, SELU
- GELU, Swish/SiLU, Mish
- Sigmoid, Tanh, Softmax
- HardSigmoid, HardSwish
- Softplus

#### Training
- Adam, AdamW, SGD, RMSprop optimizers
- CrossEntropyLoss, BinaryCrossEntropyLoss
- MSELoss, L1Loss, HuberLoss

#### Backends
- JuliaBackend: Pure Julia implementation
- RustBackend: High-performance parallel backend using Rayon
- ZigBackend: SIMD-optimized backend with minimal binary size

#### Compatibility
- PyTorch model import
- ONNX export/import

#### Documentation
- Complete wiki documentation
- API reference
- Tutorials (beginner to advanced)
- Migration guide from PyTorch

### Security
- Memory-safe backends (Rust, Zig)
- Type-safe tensor operations
- Verification system for safety-critical applications

---

## Release Notes Format

### Types of Changes

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

### Version Numbering

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

[Unreleased]: https://github.com/Hyperpolymath/Axiom.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Hyperpolymath/Axiom.jl/releases/tag/v0.1.0
