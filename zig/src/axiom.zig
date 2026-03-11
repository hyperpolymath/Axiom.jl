// SPDX-License-Identifier: PMPL-1.0-or-later
//! Axiom.jl Zig Backend — Sole Native Backend
//!
//! High-performance, minimal-footprint backend for Axiom.jl.
//! Provides SIMD-optimized neural network operations with multi-threaded
//! dispatch and zero-overhead C FFI.
//!
//! Features:
//! - 8-wide f32 SIMD vectorization for all activations
//! - Multi-threaded dispatch (4 threads, 64K threshold for element-wise ops)
//! - Batch-parallel threading for softmax/layernorm/rmsnorm
//! - In-place activation variants (zero-allocation inference)
//! - 36 FFI exports, ~400KB compiled .so

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;

pub const matmul = @import("matmul.zig");
pub const activations = @import("activations.zig");
pub const conv = @import("conv.zig");
pub const pool = @import("pool.zig");
pub const norm = @import("norm.zig");
pub const attention = @import("attention.zig");
pub const threading = @import("threading.zig");

// ============================================================================
// Version & Initialization
// ============================================================================

pub const VERSION = "0.1.0";

export fn axiom_zig_version() [*:0]const u8 {
    return "Axiom.jl Zig Backend v" ++ VERSION;
}

export fn axiom_zig_init() void {
    // Initialize thread pool, allocators, etc.
    std.log.info("Axiom Zig Backend initialized", .{});
}

// ============================================================================
// Matrix Operations (FFI Exports)
// ============================================================================

/// Matrix multiplication: C = A @ B
/// Uses tiled algorithm with SIMD for cache efficiency
export fn axiom_matmul(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    c_ptr: [*]f32,
    m: usize,
    k: usize,
    n: usize,
) void {
    const a = a_ptr[0 .. m * k];
    const b = b_ptr[0 .. k * n];
    const c = c_ptr[0 .. m * n];

    matmul.matmul_tiled(a, b, c, m, k, n);
}

/// Batched matrix multiplication
export fn axiom_bmm(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    c_ptr: [*]f32,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) void {
    const mat_size_a = m * k;
    const mat_size_b = k * n;
    const mat_size_c = m * n;

    var i: usize = 0;
    while (i < batch) : (i += 1) {
        const a = a_ptr[i * mat_size_a ..][0..mat_size_a];
        const b = b_ptr[i * mat_size_b ..][0..mat_size_b];
        const c = c_ptr[i * mat_size_c ..][0..mat_size_c];

        matmul.matmul_tiled(a, b, c, m, k, n);
    }
}

// ============================================================================
// Activation Functions (FFI Exports)
// ============================================================================

export fn axiom_relu(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_relu(x_ptr, y_ptr, n);
}

export fn axiom_relu_inplace(x_ptr: [*]f32, n: usize) void {
    activations.relu_inplace(x_ptr[0..n]);
}

export fn axiom_sigmoid_inplace(x_ptr: [*]f32, n: usize) void {
    activations.sigmoid_inplace(x_ptr[0..n]);
}

export fn axiom_tanh_inplace(x_ptr: [*]f32, n: usize) void {
    activations.tanh_inplace(x_ptr[0..n]);
}

export fn axiom_gelu_inplace(x_ptr: [*]f32, n: usize) void {
    activations.gelu_inplace(x_ptr[0..n]);
}

export fn axiom_swish_inplace(x_ptr: [*]f32, n: usize) void {
    activations.swish_inplace(x_ptr[0..n]);
}

export fn axiom_gelu(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_gelu(x_ptr, y_ptr, n);
}

export fn axiom_sigmoid(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_sigmoid(x_ptr, y_ptr, n);
}

export fn axiom_softmax(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    batch_size: usize,
    num_classes: usize,
) void {
    threading.parallel_softmax_batched(x_ptr, y_ptr, batch_size, num_classes);
}

export fn axiom_swish(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_swish(x_ptr, y_ptr, n);
}

export fn axiom_tanh(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_tanh(x_ptr, y_ptr, n);
}

export fn axiom_leaky_relu(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize, alpha: f32) void {
    threading.parallel_leaky_relu(x_ptr, y_ptr, n, alpha);
}

export fn axiom_elu(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize, alpha: f32) void {
    threading.parallel_elu(x_ptr, y_ptr, n, alpha);
}

export fn axiom_selu(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_selu(x_ptr, y_ptr, n);
}

export fn axiom_mish(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_mish(x_ptr, y_ptr, n);
}

export fn axiom_hardswish(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_hard_swish(x_ptr, y_ptr, n);
}

export fn axiom_hardsigmoid(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_hard_sigmoid(x_ptr, y_ptr, n);
}

export fn axiom_log_softmax(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    batch_size: usize,
    num_classes: usize,
) void {
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const offset = b * num_classes;
        activations.log_softmax(x_ptr[offset..][0..num_classes], y_ptr[offset..][0..num_classes]);
    }
}

export fn axiom_softplus(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    threading.parallel_softplus(x_ptr, y_ptr, n);
}

// ============================================================================
// Convolution (FFI Exports)
// ============================================================================

export fn axiom_conv2d(
    input_ptr: [*]const f32,
    weight_ptr: [*]const f32,
    bias_ptr: ?[*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    h_out: usize,
    w_out: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    conv.conv2d(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch,
        h_in,
        w_in,
        c_in,
        h_out,
        w_out,
        c_out,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    );
}

// ============================================================================
// Pooling (FFI Exports)
// ============================================================================

export fn axiom_maxpool2d(
    input_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    h_in: usize,
    w_in: usize,
    channels: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    pool.maxpool2d(
        input_ptr,
        output_ptr,
        batch,
        h_in,
        w_in,
        channels,
        kh,
        kw,
        stride_h,
        stride_w,
    );
}

export fn axiom_avgpool2d(
    input_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    h_in: usize,
    w_in: usize,
    channels: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    pool.avgpool2d(
        input_ptr,
        output_ptr,
        batch,
        h_in,
        w_in,
        channels,
        kh,
        kw,
        stride_h,
        stride_w,
    );
}

export fn axiom_global_avgpool2d(
    input_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    h: usize,
    w: usize,
    channels: usize,
) void {
    pool.global_avgpool2d(input_ptr, output_ptr, batch, h, w, channels);
}

// ============================================================================
// Normalization (FFI Exports)
// ============================================================================

export fn axiom_layernorm(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    gamma_ptr: [*]const f32,
    beta_ptr: [*]const f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) void {
    threading.parallel_layernorm(x_ptr, y_ptr, gamma_ptr, beta_ptr, batch_size, hidden_size, eps);
}

export fn axiom_rmsnorm(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    weight_ptr: [*]const f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) void {
    threading.parallel_rmsnorm(x_ptr, y_ptr, weight_ptr, batch_size, hidden_size, eps);
}

export fn axiom_batchnorm(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    gamma_ptr: [*]const f32,
    beta_ptr: [*]const f32,
    running_mean_ptr: [*]const f32,
    running_var_ptr: [*]const f32,
    n_elements: usize,
    n_features: usize,
    eps: f32,
    training: i32,
) void {
    // Zig backend only supports inference mode
    _ = training;
    const batch_size = n_elements / n_features;
    norm.batchnorm(x_ptr, y_ptr, gamma_ptr, beta_ptr, running_mean_ptr, running_var_ptr, batch_size, n_features, eps);
}

// ============================================================================
// Attention (FFI Exports)
// ============================================================================

export fn axiom_scaled_dot_product_attention(
    q_ptr: [*]const f32,
    k_ptr: [*]const f32,
    v_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    seq_len: usize,
    head_dim: usize,
    mask_ptr: ?[*]const f32,
) void {
    attention.scaled_dot_product_attention(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        batch,
        seq_len,
        head_dim,
        mask_ptr,
    );
}

export fn axiom_flash_attention(
    q_ptr: [*]const f32,
    k_ptr: [*]const f32,
    v_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) void {
    attention.flash_attention(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        batch,
        seq_len,
        head_dim,
        block_size,
    );
}

export fn axiom_rotary_embedding(
    x_ptr: [*]f32,
    seq_len: usize,
    head_dim: usize,
    base: f32,
) void {
    attention.apply_rotary_embedding(x_ptr, seq_len, head_dim, base);
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Element-wise addition with SIMD
export fn axiom_add(a_ptr: [*]const f32, b_ptr: [*]const f32, c_ptr: [*]f32, n: usize) void {
    const Vec = @Vector(8, f32);
    const vec_len = n / 8;

    var i: usize = 0;
    while (i < vec_len) : (i += 1) {
        const offset = i * 8;
        const a_vec: Vec = a_ptr[offset..][0..8].*;
        const b_vec: Vec = b_ptr[offset..][0..8].*;
        c_ptr[offset..][0..8].* = a_vec + b_vec;
    }

    // Handle remainder
    var j = vec_len * 8;
    while (j < n) : (j += 1) {
        c_ptr[j] = a_ptr[j] + b_ptr[j];
    }
}

/// Element-wise multiplication with SIMD
export fn axiom_mul(a_ptr: [*]const f32, b_ptr: [*]const f32, c_ptr: [*]f32, n: usize) void {
    const Vec = @Vector(8, f32);
    const vec_len = n / 8;

    var i: usize = 0;
    while (i < vec_len) : (i += 1) {
        const offset = i * 8;
        const a_vec: Vec = a_ptr[offset..][0..8].*;
        const b_vec: Vec = b_ptr[offset..][0..8].*;
        c_ptr[offset..][0..8].* = a_vec * b_vec;
    }

    // Handle remainder
    var j = vec_len * 8;
    while (j < n) : (j += 1) {
        c_ptr[j] = a_ptr[j] * b_ptr[j];
    }
}

/// Fill array with scalar
export fn axiom_fill(ptr: [*]f32, n: usize, value: f32) void {
    const Vec = @Vector(8, f32);
    const val_vec: Vec = @splat(value);
    const vec_len = n / 8;

    var i: usize = 0;
    while (i < vec_len) : (i += 1) {
        ptr[i * 8 ..][0..8].* = val_vec;
    }

    var j = vec_len * 8;
    while (j < n) : (j += 1) {
        ptr[j] = value;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "relu" {
    var input = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var output: [5]f32 = undefined;

    activations.relu(&input, &output);

    try testing.expectEqual(@as(f32, 0.0), output[0]);
    try testing.expectEqual(@as(f32, 0.0), output[1]);
    try testing.expectEqual(@as(f32, 0.0), output[2]);
    try testing.expectEqual(@as(f32, 1.0), output[3]);
    try testing.expectEqual(@as(f32, 2.0), output[4]);
}

test "softmax sums to 1" {
    var input = [_]f32{ 1.0, 2.0, 3.0 };
    var output: [3]f32 = undefined;

    activations.softmax(&input, &output);

    var sum: f32 = 0;
    for (output) |v| {
        sum += v;
    }

    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "matmul identity" {
    // 2x2 identity test
    const a = [_]f32{ 1, 0, 0, 1 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c: [4]f32 = undefined;

    matmul.matmul_naive(&a, &b, &c, 2, 2, 2);

    try testing.expectEqual(@as(f32, 5), c[0]);
    try testing.expectEqual(@as(f32, 6), c[1]);
    try testing.expectEqual(@as(f32, 7), c[2]);
    try testing.expectEqual(@as(f32, 8), c[3]);
}
