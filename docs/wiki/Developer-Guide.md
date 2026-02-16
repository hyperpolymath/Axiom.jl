# Developer Guide

This guide defines the development workflow and release gates for Axiom.jl.

## Prerequisites

- Julia `1.10+`
- Git
- Optional for backend work:
  - Rust toolchain
  - CUDA / ROCm / Metal runtimes

## Local Setup

```bash
cd Axiom.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Build and Test

Run the full baseline pipeline before merging:

```bash
julia --project=. -e 'using Pkg; Pkg.build(); Pkg.precompile(); Pkg.test()'
```

## Runtime Smoke Tests

Run quick runtime checks after unit tests:

```bash
julia --project=. -e 'using Axiom; model=Sequential(Dense(10,5,relu),Dense(5,3),Softmax()); x=Tensor(randn(Float32,2,10)); y=model(x); @assert size(y.data)==(2,3); result=verify(model, properties=[ValidProbabilities(), FiniteOutput()], data=[(x,nothing)]); @assert result.passed'
julia --project=. examples/mnist.jl
```

## Quality Gates

Use these checks to keep production paths clean:

```bash
rg -n "OPEN_ITEM|FIX_ITEM|XXX|HACK" src test ext docs/wiki examples
```

If a marker is required (for templates or roadmap planning), keep it out of production code paths (`src`, `ext`, `test`) and explain it in review notes.

## Documentation Expectations

- User-facing behavior changes require updates in:
  - `docs/wiki/User-Guide.md`
  - `docs/wiki/Home.md`
- Developer workflow changes require updates in:
  - `docs/wiki/Developer-Guide.md`
- Claims in README/wiki must match tested behavior.

## Release Checklist

1. `Pkg.build`, `Pkg.precompile`, and `Pkg.test` pass on a clean environment.
2. Runtime smoke checks pass.
3. No unresolved production work-marker markers in `src`, `ext`, or `test`.
4. README/wiki claims are aligned with actual implementation status.
5. Version metadata is consistent (`Project.toml` and `Axiom.VERSION`).
