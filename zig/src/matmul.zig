// SPDX-License-Identifier: PMPL-1.0-or-later
//! Matrix Multiplication Operations
//!
//! Provides highly optimized matrix multiplication using:
//! - Tiled algorithms for cache efficiency
//! - SIMD vectorization
//! - Loop unrolling

const std = @import("std");

/// SIMD vector size (AVX = 8 floats)
const VEC_SIZE = 8;
const Vec = @Vector(VEC_SIZE, f32);

/// Tile size for cache-efficient matrix multiplication
/// Chosen to fit L1 cache: 64 * 64 * 4 bytes = 16KB
const TILE_SIZE = 64;

/// Naive matrix multiplication (for small matrices or verification)
pub fn matmul_naive(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    // Zero output
    @memset(c, 0);

    var i: usize = 0;
    while (i < m) : (i += 1) {
        var kk: usize = 0;
        while (kk < k) : (kk += 1) {
            const a_ik = a[i * k + kk];
            var j: usize = 0;
            while (j < n) : (j += 1) {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
}

/// Tiled matrix multiplication with SIMD
/// C[m,n] = A[m,k] @ B[k,n]
pub fn matmul_tiled(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    // Zero output
    @memset(c, 0);

    // For small matrices, use naive algorithm
    if (m * n < 1024) {
        matmul_naive(a, b, c, m, k, n);
        return;
    }

    // Tiled multiplication
    var i_tile: usize = 0;
    while (i_tile < m) : (i_tile += TILE_SIZE) {
        const i_end = @min(i_tile + TILE_SIZE, m);

        var k_tile: usize = 0;
        while (k_tile < k) : (k_tile += TILE_SIZE) {
            const k_end = @min(k_tile + TILE_SIZE, k);

            var j_tile: usize = 0;
            while (j_tile < n) : (j_tile += TILE_SIZE) {
                const j_end = @min(j_tile + TILE_SIZE, n);

                // Process tile with SIMD
                matmul_tile_simd(a, b, c, m, k, n, i_tile, i_end, k_tile, k_end, j_tile, j_end);
            }
        }
    }
}

/// Process a single tile with SIMD
fn matmul_tile_simd(
    a: []const f32,
    b: []const f32,
    c: []f32,
    _: usize, // m (unused but kept for clarity)
    k: usize,
    n: usize,
    i_start: usize,
    i_end: usize,
    k_start: usize,
    k_end: usize,
    j_start: usize,
    j_end: usize,
) void {
    var i = i_start;
    while (i < i_end) : (i += 1) {
        var kk = k_start;
        while (kk < k_end) : (kk += 1) {
            const a_ik = a[i * k + kk];
            const a_vec: Vec = @splat(a_ik);

            // SIMD inner loop
            var j = j_start;
            while (j + VEC_SIZE <= j_end) : (j += VEC_SIZE) {
                const b_vec: Vec = b[kk * n + j ..][0..VEC_SIZE].*;
                const c_vec: Vec = c[i * n + j ..][0..VEC_SIZE].*;
                c[i * n + j ..][0..VEC_SIZE].* = c_vec + a_vec * b_vec;
            }

            // Scalar remainder
            while (j < j_end) : (j += 1) {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
}

/// Matrix-vector multiplication: y = A @ x
pub fn matvec(
    a: []const f32,
    x: []const f32,
    y: []f32,
    m: usize,
    n: usize,
) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var sum: f32 = 0;
        const row = a[i * n ..][0..n];

        // SIMD dot product
        var j: usize = 0;
        var acc: Vec = @splat(0);
        while (j + VEC_SIZE <= n) : (j += VEC_SIZE) {
            const a_vec: Vec = row[j..][0..VEC_SIZE].*;
            const x_vec: Vec = x[j..][0..VEC_SIZE].*;
            acc += a_vec * x_vec;
        }

        // Reduce SIMD accumulator
        sum = @reduce(.Add, acc);

        // Scalar remainder
        while (j < n) : (j += 1) {
            sum += row[j] * x[j];
        }

        y[i] = sum;
    }
}

/// Outer product: C = a * b^T
pub fn outer(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        const a_i = a[i];
        const a_vec: Vec = @splat(a_i);

        var j: usize = 0;
        while (j + VEC_SIZE <= n) : (j += VEC_SIZE) {
            const b_vec: Vec = b[j..][0..VEC_SIZE].*;
            c[i * n + j ..][0..VEC_SIZE].* = a_vec * b_vec;
        }

        while (j < n) : (j += 1) {
            c[i * n + j] = a_i * b[j];
        }
    }
}

/// Transpose matrix in-place (square matrices only)
pub fn transpose_inplace(a: []f32, n: usize) void {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var j = i + 1;
        while (j < n) : (j += 1) {
            const tmp = a[i * n + j];
            a[i * n + j] = a[j * n + i];
            a[j * n + i] = tmp;
        }
    }
}

/// Transpose matrix (general)
pub fn transpose(
    a: []const f32,
    b: []f32,
    m: usize,
    n: usize,
) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            b[j * m + i] = a[i * n + j];
        }
    }
}
