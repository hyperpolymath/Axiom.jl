// SPDX-License-Identifier: PMPL-1.0-or-later
//! Threading support for Axiom.jl Zig backend
//!
//! Provides parallel dispatch for element-wise operations on large arrays.
//! Below THREAD_THRESHOLD, operations run single-threaded with SIMD.
//! Above THREAD_THRESHOLD, work is split across CPU threads.

const std = @import("std");
const activations = @import("activations.zig");
const norm = @import("norm.zig");

/// Minimum elements before spawning threads (64K)
pub const THREAD_THRESHOLD: usize = 65536;

/// Maximum worker threads (main thread also does work)
const MAX_WORKERS: usize = 3; // + main thread = 4 total

// ============================================================================
// Thread context types for each activation signature
// ============================================================================

const UnaryCtx = struct {
    input: [*]const f32,
    output: [*]f32,
    start: usize,
    len: usize,
};

const UnaryWithAlphaCtx = struct {
    input: [*]const f32,
    output: [*]f32,
    start: usize,
    len: usize,
    alpha: f32,
};

// ============================================================================
// Worker functions (called by spawned threads)
// ============================================================================

fn worker_sigmoid(ctx: UnaryCtx) void {
    activations.sigmoid(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_gelu(ctx: UnaryCtx) void {
    activations.gelu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_tanh(ctx: UnaryCtx) void {
    activations.tanh_activation(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_relu(ctx: UnaryCtx) void {
    activations.relu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_swish(ctx: UnaryCtx) void {
    activations.swish(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_mish(ctx: UnaryCtx) void {
    activations.mish(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_selu(ctx: UnaryCtx) void {
    activations.selu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_hard_sigmoid(ctx: UnaryCtx) void {
    activations.hard_sigmoid(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_hard_swish(ctx: UnaryCtx) void {
    activations.hard_swish(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_softplus(ctx: UnaryCtx) void {
    activations.softplus(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_leaky_relu(ctx: UnaryWithAlphaCtx) void {
    activations.leaky_relu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len], ctx.alpha);
}

fn worker_elu(ctx: UnaryWithAlphaCtx) void {
    activations.elu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len], ctx.alpha);
}

// ============================================================================
// Parallel dispatch functions (called from axiom.zig FFI exports)
// ============================================================================

/// Parallel sigmoid for large arrays
pub fn parallel_sigmoid(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.sigmoid(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_sigmoid, input, output, n);
}

/// Parallel GELU for large arrays
pub fn parallel_gelu(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.gelu(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_gelu, input, output, n);
}

/// Parallel tanh for large arrays
pub fn parallel_tanh(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.tanh_activation(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_tanh, input, output, n);
}

/// Parallel relu for large arrays
pub fn parallel_relu(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.relu(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_relu, input, output, n);
}

/// Parallel swish for large arrays
pub fn parallel_swish(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.swish(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_swish, input, output, n);
}

/// Parallel mish for large arrays
pub fn parallel_mish(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.mish(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_mish, input, output, n);
}

/// Parallel selu for large arrays
pub fn parallel_selu(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.selu(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_selu, input, output, n);
}

/// Parallel hard_sigmoid for large arrays
pub fn parallel_hard_sigmoid(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.hard_sigmoid(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_hard_sigmoid, input, output, n);
}

/// Parallel hard_swish for large arrays
pub fn parallel_hard_swish(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.hard_swish(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_hard_swish, input, output, n);
}

/// Parallel softplus for large arrays
pub fn parallel_softplus(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.softplus(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_softplus, input, output, n);
}

/// Parallel leaky_relu for large arrays
pub fn parallel_leaky_relu(input: [*]const f32, output: [*]f32, n: usize, alpha: f32) void {
    if (n < THREAD_THRESHOLD) {
        activations.leaky_relu(input[0..n], output[0..n], alpha);
        return;
    }
    parallel_unary_alpha(worker_leaky_relu, input, output, n, alpha);
}

/// Parallel elu for large arrays
pub fn parallel_elu(input: [*]const f32, output: [*]f32, n: usize, alpha: f32) void {
    if (n < THREAD_THRESHOLD) {
        activations.elu(input[0..n], output[0..n], alpha);
        return;
    }
    parallel_unary_alpha(worker_elu, input, output, n, alpha);
}

// ============================================================================
// Internal: generic parallel dispatch
// ============================================================================

fn parallel_unary(
    comptime worker_fn: fn (UnaryCtx) void,
    input: [*]const f32,
    output: [*]f32,
    n: usize,
) void {
    const num_threads = @min(MAX_WORKERS + 1, (n + THREAD_THRESHOLD - 1) / THREAD_THRESHOLD);
    const chunk = (n + num_threads - 1) / num_threads;

    var handles: [MAX_WORKERS]?std.Thread = .{null} ** MAX_WORKERS;

    // Spawn worker threads for all but last chunk
    for (0..num_threads - 1) |t| {
        const start = t * chunk;
        const end = @min(start + chunk, n);
        handles[t] = std.Thread.spawn(.{}, worker_fn, .{UnaryCtx{
            .input = input,
            .output = output,
            .start = start,
            .len = end - start,
        }}) catch null;
    }

    // Main thread does last chunk
    const last_start = (num_threads - 1) * chunk;
    const last_len = n - last_start;
    worker_fn(UnaryCtx{
        .input = input,
        .output = output,
        .start = last_start,
        .len = last_len,
    });

    // Join workers
    for (&handles) |*maybe_handle| {
        if (maybe_handle.*) |handle| {
            handle.join();
            maybe_handle.* = null;
        }
    }
}

fn parallel_unary_alpha(
    comptime worker_fn: fn (UnaryWithAlphaCtx) void,
    input: [*]const f32,
    output: [*]f32,
    n: usize,
    alpha: f32,
) void {
    const num_threads = @min(MAX_WORKERS + 1, (n + THREAD_THRESHOLD - 1) / THREAD_THRESHOLD);
    const chunk = (n + num_threads - 1) / num_threads;

    var handles: [MAX_WORKERS]?std.Thread = .{null} ** MAX_WORKERS;

    for (0..num_threads - 1) |t| {
        const start = t * chunk;
        const end = @min(start + chunk, n);
        handles[t] = std.Thread.spawn(.{}, worker_fn, .{UnaryWithAlphaCtx{
            .input = input,
            .output = output,
            .start = start,
            .len = end - start,
            .alpha = alpha,
        }}) catch null;
    }

    const last_start = (num_threads - 1) * chunk;
    const last_len = n - last_start;
    worker_fn(UnaryWithAlphaCtx{
        .input = input,
        .output = output,
        .start = last_start,
        .len = last_len,
        .alpha = alpha,
    });

    for (&handles) |*maybe_handle| {
        if (maybe_handle.*) |handle| {
            handle.join();
            maybe_handle.* = null;
        }
    }
}

// ============================================================================
// Batch-parallel dispatch for normalization/softmax operations
// ============================================================================

/// Minimum total elements before batch-parallel threading kicks in
const BATCH_THREAD_THRESHOLD: usize = 8192;

/// Context for batch-parallel softmax
const SoftmaxBatchCtx = struct {
    input: [*]const f32,
    output: [*]f32,
    batch_start: usize,
    batch_count: usize,
    num_classes: usize,
};

fn worker_softmax_batch(ctx: SoftmaxBatchCtx) void {
    const offset = ctx.batch_start * ctx.num_classes;
    activations.softmax_batched(
        ctx.input + offset,
        ctx.output + offset,
        ctx.batch_count,
        ctx.num_classes,
    );
}

/// Parallel softmax over batch dimension
pub fn parallel_softmax_batched(
    input_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch_size: usize,
    num_classes: usize,
) void {
    const total = batch_size * num_classes;
    if (total < BATCH_THREAD_THRESHOLD or batch_size < 4) {
        activations.softmax_batched(input_ptr, output_ptr, batch_size, num_classes);
        return;
    }
    parallel_batch(SoftmaxBatchCtx, worker_softmax_batch, input_ptr, output_ptr, batch_size, num_classes, null, null, null, 0);
}

/// Context for batch-parallel layernorm
const LayernormBatchCtx = struct {
    input: [*]const f32,
    output: [*]f32,
    gamma: [*]const f32,
    beta: [*]const f32,
    batch_start: usize,
    batch_count: usize,
    hidden_size: usize,
    eps: f32,
};

fn worker_layernorm_batch(ctx: LayernormBatchCtx) void {
    const offset = ctx.batch_start * ctx.hidden_size;
    norm.layernorm(
        ctx.input + offset,
        ctx.output + offset,
        ctx.gamma,
        ctx.beta,
        ctx.batch_count,
        ctx.hidden_size,
        ctx.eps,
    );
}

/// Parallel layernorm over batch dimension
pub fn parallel_layernorm(
    input_ptr: [*]const f32,
    output_ptr: [*]f32,
    gamma_ptr: [*]const f32,
    beta_ptr: [*]const f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) void {
    const total = batch_size * hidden_size;
    if (total < BATCH_THREAD_THRESHOLD or batch_size < 4) {
        norm.layernorm(input_ptr, output_ptr, gamma_ptr, beta_ptr, batch_size, hidden_size, eps);
        return;
    }

    const num_threads = @min(MAX_WORKERS + 1, batch_size);
    const chunk = (batch_size + num_threads - 1) / num_threads;
    var handles: [MAX_WORKERS]?std.Thread = .{null} ** MAX_WORKERS;

    for (0..num_threads - 1) |t| {
        const b_start = t * chunk;
        const b_count = @min(chunk, batch_size - b_start);
        handles[t] = std.Thread.spawn(.{}, worker_layernorm_batch, .{LayernormBatchCtx{
            .input = input_ptr,
            .output = output_ptr,
            .gamma = gamma_ptr,
            .beta = beta_ptr,
            .batch_start = b_start,
            .batch_count = b_count,
            .hidden_size = hidden_size,
            .eps = eps,
        }}) catch null;
    }

    // Main thread does last chunk
    const last_start = (num_threads - 1) * chunk;
    const last_count = batch_size - last_start;
    worker_layernorm_batch(LayernormBatchCtx{
        .input = input_ptr,
        .output = output_ptr,
        .gamma = gamma_ptr,
        .beta = beta_ptr,
        .batch_start = last_start,
        .batch_count = last_count,
        .hidden_size = hidden_size,
        .eps = eps,
    });

    for (&handles) |*maybe_handle| {
        if (maybe_handle.*) |handle| {
            handle.join();
            maybe_handle.* = null;
        }
    }
}

/// Context for batch-parallel rmsnorm
const RmsnormBatchCtx = struct {
    input: [*]const f32,
    output: [*]f32,
    weight: [*]const f32,
    batch_start: usize,
    batch_count: usize,
    hidden_size: usize,
    eps: f32,
};

fn worker_rmsnorm_batch(ctx: RmsnormBatchCtx) void {
    const offset = ctx.batch_start * ctx.hidden_size;
    norm.rmsnorm(
        ctx.input + offset,
        ctx.output + offset,
        ctx.weight,
        ctx.batch_count,
        ctx.hidden_size,
        ctx.eps,
    );
}

/// Parallel rmsnorm over batch dimension
pub fn parallel_rmsnorm(
    input_ptr: [*]const f32,
    output_ptr: [*]f32,
    weight_ptr: [*]const f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) void {
    const total = batch_size * hidden_size;
    if (total < BATCH_THREAD_THRESHOLD or batch_size < 4) {
        norm.rmsnorm(input_ptr, output_ptr, weight_ptr, batch_size, hidden_size, eps);
        return;
    }

    const num_threads = @min(MAX_WORKERS + 1, batch_size);
    const chunk = (batch_size + num_threads - 1) / num_threads;
    var handles: [MAX_WORKERS]?std.Thread = .{null} ** MAX_WORKERS;

    for (0..num_threads - 1) |t| {
        const b_start = t * chunk;
        const b_count = @min(chunk, batch_size - b_start);
        handles[t] = std.Thread.spawn(.{}, worker_rmsnorm_batch, .{RmsnormBatchCtx{
            .input = input_ptr,
            .output = output_ptr,
            .weight = weight_ptr,
            .batch_start = b_start,
            .batch_count = b_count,
            .hidden_size = hidden_size,
            .eps = eps,
        }}) catch null;
    }

    const last_start = (num_threads - 1) * chunk;
    const last_count = batch_size - last_start;
    worker_rmsnorm_batch(RmsnormBatchCtx{
        .input = input_ptr,
        .output = output_ptr,
        .weight = weight_ptr,
        .batch_start = last_start,
        .batch_count = last_count,
        .hidden_size = hidden_size,
        .eps = eps,
    });

    for (&handles) |*maybe_handle| {
        if (maybe_handle.*) |handle| {
            handle.join();
            maybe_handle.* = null;
        }
    }
}

// Generic batch parallel dispatcher (used by softmax)
fn parallel_batch(
    comptime Ctx: type,
    comptime worker_fn: fn (Ctx) void,
    input: [*]const f32,
    output: [*]f32,
    batch_size: usize,
    feature_size: usize,
    _gamma: ?[*]const f32,
    _beta: ?[*]const f32,
    _weight: ?[*]const f32,
    _eps: f32,
) void {
    _ = _gamma;
    _ = _beta;
    _ = _weight;
    _ = _eps;

    const num_threads = @min(MAX_WORKERS + 1, batch_size);
    const chunk = (batch_size + num_threads - 1) / num_threads;
    var handles: [MAX_WORKERS]?std.Thread = .{null} ** MAX_WORKERS;

    for (0..num_threads - 1) |t| {
        const b_start = t * chunk;
        const b_count = @min(chunk, batch_size - b_start);
        handles[t] = std.Thread.spawn(.{}, worker_fn, .{Ctx{
            .input = input,
            .output = output,
            .batch_start = b_start,
            .batch_count = b_count,
            .num_classes = feature_size,
        }}) catch null;
    }

    const last_start = (num_threads - 1) * chunk;
    const last_count = batch_size - last_start;
    worker_fn(Ctx{
        .input = input,
        .output = output,
        .batch_start = last_start,
        .batch_count = last_count,
        .num_classes = feature_size,
    });

    for (&handles) |*maybe_handle| {
        if (maybe_handle.*) |handle| {
            handle.join();
            maybe_handle.* = null;
        }
    }
}
