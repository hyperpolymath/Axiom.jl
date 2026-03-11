<!-- SPDX-License-Identifier: PMPL-1.0-or-later -->
# Performance Tuning and Benchmarking

Axiom.jl provides comprehensive benchmarking infrastructure to track performance across Julia and Zig backends (plus optional GPU accelerator paths where available).

## Running Benchmarks

### Julia Benchmarks

```bash
cd Axiom.jl
julia --project=benchmark benchmark/benchmarks.jl
```

Results are saved to `benchmark/results_TIMESTAMP.json`.

### Zig Benchmarks

```bash
cd zig
zig build test -Doptimize=ReleaseFast
```

The Zig backend (.so artifact) is benchmarked via Julia FFI using `benchmark/benchmarks.jl` with `AXIOM_ZIG_LIB` set.

## Regression Tracking

Benchmarks run automatically in CI. The workflow:
1. Runs Julia benchmarks (and Zig FFI benchmarks if .so is built)
2. Compares against baseline
3. Fails if >10% regression detected
4. Updates baseline on main branch

## See Also

- [GPU Backends](GPU-Backends.md)
- [Issue #13](https://github.com/hyperpolymath/Axiom.jl/issues/13)
