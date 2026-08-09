// SPDX-License-Identifier: MPL-2.0
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

pub const AXIOM_STATUS_OK: u32 = 0;
pub const AXIOM_STATUS_NULL_POINTER: u32 = 1;
pub const AXIOM_STATUS_NON_FINITE_INPUT: u32 = 2;
pub const AXIOM_STATUS_ALIASING: u32 = 3;
pub const AXIOM_STATUS_INVALID_DIMENSION: u32 = 4;
pub const AXIOM_STATUS_NON_FINITE_RESULT: u32 = 5;

const AddressRange = struct {
    start: usize,
    end: usize,

    fn overlaps(self: AddressRange, other: AddressRange) bool {
        return self.start < other.end and other.start < self.end;
    }
};

fn f32Range(pointer: anytype, len: usize) ?AddressRange {
    if (len == 0) return .{ .start = 0, .end = 0 };
    const resolved = pointer orelse return null;
    const start = @intFromPtr(resolved);
    const bytes = std.math.mul(usize, len, @sizeOf(f32)) catch return null;
    const end = std.math.add(usize, start, bytes) catch return null;
    return .{ .start = start, .end = end };
}

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

fn matmulCellFinite(a: []const f32, b: []const f32, row: usize, column: usize, k: usize, n: usize) bool {
    var sum: f32 = 0.0;
    for (0..k) |inner| {
        sum += a[row * k + inner] * b[inner * n + column];
        if (!std.math.isFinite(sum)) return false;
    }
    return true;
}

/// Checked row-major matrix multiplication. Dimensions derive every buffer
/// length; inputs and all prospective cells are validated before output is
/// changed. The legacy void export remains available for proven callers.
export fn axiom_matmul_checked(
    a_ptr: ?[*]const f32,
    b_ptr: ?[*]const f32,
    c_ptr: ?[*]f32,
    m: usize,
    k: usize,
    n: usize,
) u32 {
    const a_len = std.math.mul(usize, m, k) catch return AXIOM_STATUS_INVALID_DIMENSION;
    const b_len = std.math.mul(usize, k, n) catch return AXIOM_STATUS_INVALID_DIMENSION;
    const c_len = std.math.mul(usize, m, n) catch return AXIOM_STATUS_INVALID_DIMENSION;
    if ((a_len != 0 and a_ptr == null) or (b_len != 0 and b_ptr == null) or (c_len != 0 and c_ptr == null))
        return AXIOM_STATUS_NULL_POINTER;
    const a_range = f32Range(a_ptr, a_len) orelse return AXIOM_STATUS_INVALID_DIMENSION;
    const b_range = f32Range(b_ptr, b_len) orelse return AXIOM_STATUS_INVALID_DIMENSION;
    const c_range = f32Range(c_ptr, c_len) orelse return AXIOM_STATUS_INVALID_DIMENSION;
    if (c_range.overlaps(a_range) or c_range.overlaps(b_range)) return AXIOM_STATUS_ALIASING;
    const a = if (a_len == 0) &[_]f32{} else a_ptr.?[0..a_len];
    const b = if (b_len == 0) &[_]f32{} else b_ptr.?[0..b_len];
    for (a) |value| if (!std.math.isFinite(value)) return AXIOM_STATUS_NON_FINITE_INPUT;
    for (b) |value| if (!std.math.isFinite(value)) return AXIOM_STATUS_NON_FINITE_INPUT;
    for (0..m) |row| for (0..n) |column| {
        if (!matmulCellFinite(a, b, row, column, k, n)) return AXIOM_STATUS_NON_FINITE_RESULT;
    };
    if (c_len != 0) matmul.matmul_tiled(a, b, c_ptr.?[0..c_len], m, k, n);
    return AXIOM_STATUS_OK;
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

/// Checked, output-atomic ReLU for FFI consumers that cannot prove pointer and
/// numeric preconditions in their type system. Legacy `axiom_relu` remains for
/// compatibility with existing Julia callers.
export fn axiom_relu_checked(x_ptr: ?[*]const f32, y_ptr: ?[*]f32, n: usize) u32 {
    if (n == 0) return AXIOM_STATUS_OK;
    const input_range = f32Range(x_ptr, n) orelse return AXIOM_STATUS_NULL_POINTER;
    const output_range = f32Range(y_ptr, n) orelse return AXIOM_STATUS_NULL_POINTER;
    if (input_range.overlaps(output_range)) return AXIOM_STATUS_ALIASING;
    const input = x_ptr.?[0..n];
    for (input) |value| if (!std.math.isFinite(value)) return AXIOM_STATUS_NON_FINITE_INPUT;
    activations.relu(input, y_ptr.?[0..n]);
    return AXIOM_STATUS_OK;
}

export fn axiom_relu6(x_ptr: [*]const f32, y_ptr: [*]f32, n: usize) void {
    activations.relu6(x_ptr[0..n], y_ptr[0..n]);
}

export fn axiom_relu6_checked(x_ptr: ?[*]const f32, y_ptr: ?[*]f32, n: usize) u32 {
    if (n == 0) return AXIOM_STATUS_OK;
    const input_range = f32Range(x_ptr, n) orelse return AXIOM_STATUS_NULL_POINTER;
    const output_range = f32Range(y_ptr, n) orelse return AXIOM_STATUS_NULL_POINTER;
    if (input_range.overlaps(output_range)) return AXIOM_STATUS_ALIASING;
    const input = x_ptr.?[0..n];
    for (input) |value| if (!std.math.isFinite(value)) return AXIOM_STATUS_NON_FINITE_INPUT;
    activations.relu6(input, y_ptr.?[0..n]);
    return AXIOM_STATUS_OK;
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
    if (seq_len == 0 or seq_len > 64 or head_dim == 0) return;
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

export fn axiom_scaled_dot_product_attention_checked(
    q_ptr: ?[*]const f32,
    k_ptr: ?[*]const f32,
    v_ptr: ?[*]const f32,
    output_ptr: ?[*]f32,
    batch: usize,
    seq_len: usize,
    head_dim: usize,
    mask_ptr: ?[*]const f32,
) u32 {
    if (seq_len == 0 or seq_len > 64 or head_dim == 0) return AXIOM_STATUS_INVALID_DIMENSION;
    if (batch != 0 and (q_ptr == null or k_ptr == null or v_ptr == null or output_ptr == null))
        return AXIOM_STATUS_NULL_POINTER;
    if (batch == 0) return AXIOM_STATUS_OK;
    attention.scaled_dot_product_attention(q_ptr.?, k_ptr.?, v_ptr.?, output_ptr.?, batch, seq_len, head_dim, mask_ptr);
    return AXIOM_STATUS_OK;
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
    if (seq_len == 0 or seq_len > 4096 or head_dim == 0 or block_size == 0 or block_size > 64) return;
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

export fn axiom_flash_attention_checked(
    q_ptr: ?[*]const f32,
    k_ptr: ?[*]const f32,
    v_ptr: ?[*]const f32,
    output_ptr: ?[*]f32,
    batch: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) u32 {
    if (seq_len == 0 or seq_len > 4096 or head_dim == 0 or block_size == 0 or block_size > 64)
        return AXIOM_STATUS_INVALID_DIMENSION;
    if (batch != 0 and (q_ptr == null or k_ptr == null or v_ptr == null or output_ptr == null))
        return AXIOM_STATUS_NULL_POINTER;
    if (batch == 0) return AXIOM_STATUS_OK;
    attention.flash_attention(q_ptr.?, k_ptr.?, v_ptr.?, output_ptr.?, batch, seq_len, head_dim, block_size);
    return AXIOM_STATUS_OK;
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

fn validateBinaryBuffers(a_ptr: ?[*]const f32, b_ptr: ?[*]const f32, c_ptr: ?[*]f32, n: usize) u32 {
    if (n == 0) return AXIOM_STATUS_OK;
    const a_range = f32Range(a_ptr, n) orelse return AXIOM_STATUS_NULL_POINTER;
    const b_range = f32Range(b_ptr, n) orelse return AXIOM_STATUS_NULL_POINTER;
    const c_range = f32Range(c_ptr, n) orelse return AXIOM_STATUS_NULL_POINTER;
    if (c_range.overlaps(a_range) or c_range.overlaps(b_range)) return AXIOM_STATUS_ALIASING;
    return AXIOM_STATUS_OK;
}

export fn axiom_add_checked(a_ptr: ?[*]const f32, b_ptr: ?[*]const f32, c_ptr: ?[*]f32, n: usize) u32 {
    const status = validateBinaryBuffers(a_ptr, b_ptr, c_ptr, n);
    if (status != AXIOM_STATUS_OK or n == 0) return status;
    const a = a_ptr.?[0..n];
    const b = b_ptr.?[0..n];
    for (a, b) |left, right| {
        if (!std.math.isFinite(left) or !std.math.isFinite(right)) return AXIOM_STATUS_NON_FINITE_INPUT;
        if (!std.math.isFinite(left + right)) return AXIOM_STATUS_NON_FINITE_RESULT;
    }
    axiom_add(a_ptr.?, b_ptr.?, c_ptr.?, n);
    return AXIOM_STATUS_OK;
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

export fn axiom_mul_checked(a_ptr: ?[*]const f32, b_ptr: ?[*]const f32, c_ptr: ?[*]f32, n: usize) u32 {
    const status = validateBinaryBuffers(a_ptr, b_ptr, c_ptr, n);
    if (status != AXIOM_STATUS_OK or n == 0) return status;
    const a = a_ptr.?[0..n];
    const b = b_ptr.?[0..n];
    for (a, b) |left, right| {
        if (!std.math.isFinite(left) or !std.math.isFinite(right)) return AXIOM_STATUS_NON_FINITE_INPUT;
        if (!std.math.isFinite(left * right)) return AXIOM_STATUS_NON_FINITE_RESULT;
    }
    axiom_mul(a_ptr.?, b_ptr.?, c_ptr.?, n);
    return AXIOM_STATUS_OK;
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

test "checked relu family is finite, non-aliasing, and output atomic" {
    const input = [_]f32{ -2.0, -0.0, 2.5, 9.0 };
    var relu_output = [_]f32{ 91.0, 91.0, 91.0, 91.0 };
    try testing.expectEqual(AXIOM_STATUS_OK, axiom_relu_checked(&input, &relu_output, input.len));
    try testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 2.5, 9.0 }, &relu_output);

    var relu6_output = [_]f32{ 92.0, 92.0, 92.0, 92.0 };
    try testing.expectEqual(AXIOM_STATUS_OK, axiom_relu6_checked(&input, &relu6_output, input.len));
    try testing.expectEqualSlices(f32, &[_]f32{ 0.0, 0.0, 2.5, 6.0 }, &relu6_output);

    const invalid = [_]f32{ 1.0, std.math.nan(f32), 3.0 };
    var untouched = [_]f32{ 7.0, 8.0, 9.0 };
    try testing.expectEqual(AXIOM_STATUS_NON_FINITE_INPUT, axiom_relu_checked(&invalid, &untouched, invalid.len));
    try testing.expectEqualSlices(f32, &[_]f32{ 7.0, 8.0, 9.0 }, &untouched);

    var aliased = [_]f32{ -1.0, 2.0 };
    try testing.expectEqual(AXIOM_STATUS_ALIASING, axiom_relu_checked(&aliased, &aliased, aliased.len));
    try testing.expectEqualSlices(f32, &[_]f32{ -1.0, 2.0 }, &aliased);
}

test "checked attention rejects dimensions beyond fixed scratch capacity" {
    const input = [_]f32{1.0};
    var output = [_]f32{77.0};
    try testing.expectEqual(
        AXIOM_STATUS_INVALID_DIMENSION,
        axiom_scaled_dot_product_attention_checked(&input, &input, &input, &output, 1, 65, 1, null),
    );
    try testing.expectEqual(@as(f32, 77.0), output[0]);
    try testing.expectEqual(
        AXIOM_STATUS_INVALID_DIMENSION,
        axiom_flash_attention_checked(&input, &input, &input, &output, 1, 1, 1, 65),
    );
    try testing.expectEqual(@as(f32, 77.0), output[0]);
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

test "checked matmul and binary operations are output atomic" {
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var matrix = [_]f32{91.0} ** 4;
    try testing.expectEqual(AXIOM_STATUS_OK, axiom_matmul_checked(&a, &b, &matrix, 2, 3, 2));
    try testing.expectEqualSlices(f32, &[_]f32{ 58, 64, 139, 154 }, &matrix);

    const huge = [_]f32{ std.math.floatMax(f32), std.math.floatMax(f32) };
    const factors = [_]f32{ 2.0, 1.0, 2.0, 2.0 };
    var untouched_matrix = [_]f32{ 71.0, 72.0 };
    try testing.expectEqual(
        AXIOM_STATUS_NON_FINITE_RESULT,
        axiom_matmul_checked(&huge, &factors, &untouched_matrix, 1, 2, 2),
    );
    try testing.expectEqualSlices(f32, &[_]f32{ 71.0, 72.0 }, &untouched_matrix);

    const left = [_]f32{ 1.5, -2.0, 4.0 };
    const right = [_]f32{ 2.0, 3.0, -0.5 };
    var output = [_]f32{91.0} ** 3;
    try testing.expectEqual(AXIOM_STATUS_OK, axiom_add_checked(&left, &right, &output, 3));
    try testing.expectEqualSlices(f32, &[_]f32{ 3.5, 1.0, 3.5 }, &output);
    try testing.expectEqual(AXIOM_STATUS_OK, axiom_mul_checked(&left, &right, &output, 3));
    try testing.expectEqualSlices(f32, &[_]f32{ 3.0, -6.0, -2.0 }, &output);

    var overflow_output = [_]f32{81.0};
    try testing.expectEqual(
        AXIOM_STATUS_NON_FINITE_RESULT,
        axiom_mul_checked(&[_]f32{std.math.floatMax(f32)}, &[_]f32{2.0}, &overflow_output, 1),
    );
    try testing.expectEqual(@as(f32, 81.0), overflow_output[0]);
}
