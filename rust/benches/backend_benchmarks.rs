// SPDX-License-Identifier: PMPL-1.0-or-later
// Rust Backend Benchmarks for Axiom.jl
//
// Mirrors Julia benchmarks in benchmark/benchmarks.jl
// Run with: cargo bench
//
// Refs: Issue #13 - Benchmarks and regression baselines

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Import Axiom Rust backend functions
// (These would normally come from the axiom crate)
mod ops {
    pub fn matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    #[inline(always)]
    pub fn relu(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }

    #[inline(always)]
    pub fn gelu(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|&v| {
                0.5 * v
                    * (1.0
                        + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
            })
            .collect()
    }

    #[inline(always)]
    pub fn swish(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
    }

    pub fn softmax(x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|&v| v / sum).collect()
    }

    pub fn batchnorm(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mean: &[f32],
        var: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        x.iter()
            .zip(gamma.iter())
            .zip(beta.iter())
            .zip(mean.iter())
            .zip(var.iter())
            .map(|((((x, g), b), m), v)| g * (x - m) / (v + eps).sqrt() + b)
            .collect()
    }
}

fn matmul_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [64, 256, 1024].iter() {
        let a: Vec<f32> = (0..size * size).map(|_| rand::random()).collect();
        let b: Vec<f32> = (0..size * size).map(|_| rand::random()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bencher, &size| {
            bencher.iter(|| ops::matmul(black_box(&a), black_box(&b), size, size, size));
        });
    }

    group.finish();
}

fn activation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    for size in [1000, 10000, 100000].iter() {
        let x: Vec<f32> = (0..*size)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();

        group.bench_with_input(BenchmarkId::new("relu", size), size, |bencher, _| {
            bencher.iter(|| ops::relu(black_box(&x)));
        });

        group.bench_with_input(BenchmarkId::new("gelu", size), size, |bencher, _| {
            bencher.iter(|| ops::gelu(black_box(&x)));
        });

        group.bench_with_input(BenchmarkId::new("swish", size), size, |bencher, _| {
            bencher.iter(|| ops::swish(black_box(&x)));
        });

        group.bench_with_input(BenchmarkId::new("softmax", size), size, |bencher, _| {
            bencher.iter(|| ops::softmax(black_box(&x)));
        });
    }

    group.finish();
}

fn normalization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    for (batch, features) in [(32, 128), (64, 256), (128, 512)].iter() {
        let size = batch * features;
        let x: Vec<f32> = (0..size).map(|_| rand::random()).collect();
        let gamma: Vec<f32> = vec![1.0; *features];
        let beta: Vec<f32> = vec![0.0; *features];
        let mean: Vec<f32> = vec![0.0; *features];
        let var: Vec<f32> = vec![1.0; *features];
        let eps = 1e-5f32;

        group.bench_with_input(
            BenchmarkId::new("batchnorm", format!("{}x{}", batch, features)),
            &(batch, features),
            |bencher, _| {
                bencher.iter(|| {
                    ops::batchnorm(
                        black_box(&x),
                        black_box(&gamma),
                        black_box(&beta),
                        black_box(&mean),
                        black_box(&var),
                        black_box(eps),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    matmul_benchmarks,
    activation_benchmarks,
    normalization_benchmarks
);
criterion_main!(benches);
