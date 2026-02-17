# Performance Tuning and Benchmarking

Axiom.jl provides comprehensive benchmarking infrastructure to track performance across Julia and Rust backends (plus optional accelerator paths where available).

## Running Benchmarks

### Julia Benchmarks

```bash
cd Axiom.jl
julia --project=benchmark benchmark/benchmarks.jl
```

Results are saved to `benchmark/results_TIMESTAMP.json`.

### Rust Benchmarks

```bash
cd rust
cargo bench --bench backend_benchmarks
```

Results are saved to `rust/target/criterion/`.

## Regression Tracking

Benchmarks run automatically in CI. The workflow:
1. Runs Julia and Rust benchmarks
2. Compares against baseline
3. Fails if >10% regression detected  
4. Updates baseline on main branch

## See Also

- [GPU Backends](GPU-Backends.md)
- [Issue #13](https://github.com/hyperpolymath/Axiom.jl/issues/13)
