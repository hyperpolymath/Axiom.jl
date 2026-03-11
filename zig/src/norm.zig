// SPDX-License-Identifier: PMPL-1.0-or-later
//! Normalization Operations
//!
//! Layer normalization, RMS normalization, and batch normalization.

const std = @import("std");
const math = std.math;

const VEC_SIZE = 8;
const Vec = @Vector(VEC_SIZE, f32);

/// Layer Normalization
/// Normalizes over the last dimension (hidden_size)
/// SIMD-optimized mean, variance, and normalization.
pub fn layernorm(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    gamma_ptr: [*]const f32,
    beta_ptr: [*]const f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) void {
    const hidden_f: f32 = @floatFromInt(hidden_size);

    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const x = x_ptr[b * hidden_size ..][0..hidden_size];
        var y = y_ptr[b * hidden_size ..][0..hidden_size];

        // Compute mean with SIMD
        var sum_acc: Vec = @splat(0);
        var i: usize = 0;
        while (i + VEC_SIZE <= hidden_size) : (i += VEC_SIZE) {
            const x_vec: Vec = x[i..][0..VEC_SIZE].*;
            sum_acc += x_vec;
        }
        var sum: f32 = @reduce(.Add, sum_acc);
        while (i < hidden_size) : (i += 1) {
            sum += x[i];
        }
        const mean = sum / hidden_f;

        // Compute variance with SIMD
        const mean_vec: Vec = @splat(mean);
        var var_acc: Vec = @splat(0);
        i = 0;
        while (i + VEC_SIZE <= hidden_size) : (i += VEC_SIZE) {
            const x_vec: Vec = x[i..][0..VEC_SIZE].*;
            const diff = x_vec - mean_vec;
            var_acc += diff * diff;
        }
        var var_sum: f32 = @reduce(.Add, var_acc);
        while (i < hidden_size) : (i += 1) {
            const diff = x[i] - mean;
            var_sum += diff * diff;
        }
        const inv_std = 1.0 / @sqrt(var_sum / hidden_f + eps);

        // Normalize and scale with SIMD
        const inv_std_vec: Vec = @splat(inv_std);
        i = 0;
        while (i + VEC_SIZE <= hidden_size) : (i += VEC_SIZE) {
            const x_vec: Vec = x[i..][0..VEC_SIZE].*;
            const g_vec: Vec = gamma_ptr[i..][0..VEC_SIZE].*;
            const b_vec: Vec = beta_ptr[i..][0..VEC_SIZE].*;
            const normalized = (x_vec - mean_vec) * inv_std_vec;
            y[i..][0..VEC_SIZE].* = g_vec * normalized + b_vec;
        }
        while (i < hidden_size) : (i += 1) {
            const normalized = (x[i] - mean) * inv_std;
            y[i] = gamma_ptr[i] * normalized + beta_ptr[i];
        }
    }
}

/// RMS Normalization (used in LLaMA, etc.)
/// x_norm = x / sqrt(mean(x^2) + eps) * weight
pub fn rmsnorm(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    weight_ptr: [*]const f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) void {
    const hidden_f: f32 = @floatFromInt(hidden_size);

    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const x = x_ptr[b * hidden_size ..][0..hidden_size];
        var y = y_ptr[b * hidden_size ..][0..hidden_size];

        // Compute mean of squares with SIMD
        var sum_sq: f32 = 0;
        var i: usize = 0;

        var acc: Vec = @splat(0);
        while (i + VEC_SIZE <= hidden_size) : (i += VEC_SIZE) {
            const x_vec: Vec = x[i..][0..VEC_SIZE].*;
            acc += x_vec * x_vec;
        }
        sum_sq = @reduce(.Add, acc);

        // Remainder
        while (i < hidden_size) : (i += 1) {
            sum_sq += x[i] * x[i];
        }

        const rms = @sqrt(sum_sq / hidden_f + eps);
        const inv_rms = 1.0 / rms;

        // Scale with SIMD
        const inv_rms_vec: Vec = @splat(inv_rms);
        i = 0;
        while (i + VEC_SIZE <= hidden_size) : (i += VEC_SIZE) {
            const x_vec: Vec = x[i..][0..VEC_SIZE].*;
            const w_vec: Vec = weight_ptr[i..][0..VEC_SIZE].*;
            y[i..][0..VEC_SIZE].* = x_vec * inv_rms_vec * w_vec;
        }

        // Remainder
        while (i < hidden_size) : (i += 1) {
            y[i] = x[i] * inv_rms * weight_ptr[i];
        }
    }
}

/// Batch Normalization
/// For inference mode (uses running statistics)
pub fn batchnorm(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    gamma_ptr: [*]const f32,
    beta_ptr: [*]const f32,
    running_mean_ptr: [*]const f32,
    running_var_ptr: [*]const f32,
    batch_size: usize,
    num_features: usize,
    eps: f32,
) void {
    // Precompute inv_std for each feature
    var inv_std: [4096]f32 = undefined;
    var i: usize = 0;
    while (i < num_features) : (i += 1) {
        inv_std[i] = 1.0 / @sqrt(running_var_ptr[i] + eps);
    }

    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const x = x_ptr[b * num_features ..][0..num_features];
        var y = y_ptr[b * num_features ..][0..num_features];

        var f: usize = 0;
        while (f < num_features) : (f += 1) {
            const normalized = (x[f] - running_mean_ptr[f]) * inv_std[f];
            y[f] = gamma_ptr[f] * normalized + beta_ptr[f];
        }
    }
}

/// Instance Normalization
pub fn instancenorm(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    gamma_ptr: ?[*]const f32,
    beta_ptr: ?[*]const f32,
    batch: usize,
    h: usize,
    w: usize,
    channels: usize,
    eps: f32,
) void {
    const spatial_size = h * w;
    const spatial_f: f32 = @floatFromInt(spatial_size);

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        var c: usize = 0;
        while (c < channels) : (c += 1) {
            // Compute mean over spatial dimensions
            var sum: f32 = 0;
            var s: usize = 0;
            while (s < spatial_size) : (s += 1) {
                const idx = n * spatial_size * channels + s * channels + c;
                sum += x_ptr[idx];
            }
            const mean = sum / spatial_f;

            // Compute variance
            var var_sum: f32 = 0;
            s = 0;
            while (s < spatial_size) : (s += 1) {
                const idx = n * spatial_size * channels + s * channels + c;
                const diff = x_ptr[idx] - mean;
                var_sum += diff * diff;
            }
            const variance = var_sum / spatial_f;
            const inv_std = 1.0 / @sqrt(variance + eps);

            // Get affine parameters
            const gamma = if (gamma_ptr) |g| g[c] else 1.0;
            const beta = if (beta_ptr) |b| b[c] else 0.0;

            // Normalize and scale
            s = 0;
            while (s < spatial_size) : (s += 1) {
                const idx = n * spatial_size * channels + s * channels + c;
                const normalized = (x_ptr[idx] - mean) * inv_std;
                y_ptr[idx] = gamma * normalized + beta;
            }
        }
    }
}

/// Group Normalization
pub fn groupnorm(
    x_ptr: [*]const f32,
    y_ptr: [*]f32,
    gamma_ptr: [*]const f32,
    beta_ptr: [*]const f32,
    batch: usize,
    h: usize,
    w: usize,
    channels: usize,
    num_groups: usize,
    eps: f32,
) void {
    const spatial_size = h * w;
    const channels_per_group = channels / num_groups;
    const group_size = spatial_size * channels_per_group;
    const group_size_f: f32 = @floatFromInt(group_size);

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        var g: usize = 0;
        while (g < num_groups) : (g += 1) {
            const c_start = g * channels_per_group;

            // Compute mean over group
            var sum: f32 = 0;
            var s: usize = 0;
            while (s < spatial_size) : (s += 1) {
                var c = c_start;
                while (c < c_start + channels_per_group) : (c += 1) {
                    const idx = n * spatial_size * channels + s * channels + c;
                    sum += x_ptr[idx];
                }
            }
            const mean = sum / group_size_f;

            // Compute variance
            var var_sum: f32 = 0;
            s = 0;
            while (s < spatial_size) : (s += 1) {
                var c = c_start;
                while (c < c_start + channels_per_group) : (c += 1) {
                    const idx = n * spatial_size * channels + s * channels + c;
                    const diff = x_ptr[idx] - mean;
                    var_sum += diff * diff;
                }
            }
            const variance = var_sum / group_size_f;
            const inv_std = 1.0 / @sqrt(variance + eps);

            // Normalize and scale
            s = 0;
            while (s < spatial_size) : (s += 1) {
                var c = c_start;
                while (c < c_start + channels_per_group) : (c += 1) {
                    const idx = n * spatial_size * channels + s * channels + c;
                    const normalized = (x_ptr[idx] - mean) * inv_std;
                    y_ptr[idx] = gamma_ptr[c] * normalized + beta_ptr[c];
                }
            }
        }
    }
}
