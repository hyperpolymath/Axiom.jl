// SPDX-License-Identifier: PMPL-1.0-or-later
//! Pooling Operations
//!
//! Max pooling, average pooling, and global pooling.

const std = @import("std");
const math = std.math;

/// 2D Max Pooling
pub fn maxpool2d(
    input: [*]const f32,
    output: [*]f32,
    batch: usize,
    h_in: usize,
    w_in: usize,
    channels: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    const h_out = (h_in - kh) / stride_h + 1;
    const w_out = (w_in - kw) / stride_w + 1;
    const input_hw = h_in * w_in;
    const output_hw = h_out * w_out;

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        const input_batch = input + n * input_hw * channels;
        const output_batch = output + n * output_hw * channels;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            var oh: usize = 0;
            while (oh < h_out) : (oh += 1) {
                var ow: usize = 0;
                while (ow < w_out) : (ow += 1) {
                    var max_val: f32 = -math.floatMax(f32);

                    var ki: usize = 0;
                    while (ki < kh) : (ki += 1) {
                        var kj: usize = 0;
                        while (kj < kw) : (kj += 1) {
                            const ih = oh * stride_h + ki;
                            const iw = ow * stride_w + kj;
                            const in_idx = ih * w_in * channels + iw * channels + c;
                            const val = input_batch[in_idx];
                            if (val > max_val) max_val = val;
                        }
                    }

                    const out_idx = oh * w_out * channels + ow * channels + c;
                    output_batch[out_idx] = max_val;
                }
            }
        }
    }
}

/// 2D Average Pooling
pub fn avgpool2d(
    input: [*]const f32,
    output: [*]f32,
    batch: usize,
    h_in: usize,
    w_in: usize,
    channels: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
) void {
    const h_out = (h_in - kh) / stride_h + 1;
    const w_out = (w_in - kw) / stride_w + 1;
    const input_hw = h_in * w_in;
    const output_hw = h_out * w_out;
    const kernel_size: f32 = @floatFromInt(kh * kw);

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        const input_batch = input + n * input_hw * channels;
        const output_batch = output + n * output_hw * channels;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            var oh: usize = 0;
            while (oh < h_out) : (oh += 1) {
                var ow: usize = 0;
                while (ow < w_out) : (ow += 1) {
                    var sum: f32 = 0;

                    var ki: usize = 0;
                    while (ki < kh) : (ki += 1) {
                        var kj: usize = 0;
                        while (kj < kw) : (kj += 1) {
                            const ih = oh * stride_h + ki;
                            const iw = ow * stride_w + kj;
                            const in_idx = ih * w_in * channels + iw * channels + c;
                            sum += input_batch[in_idx];
                        }
                    }

                    const out_idx = oh * w_out * channels + ow * channels + c;
                    output_batch[out_idx] = sum / kernel_size;
                }
            }
        }
    }
}

/// Global Average Pooling
/// Reduces (N, H, W, C) to (N, C)
pub fn global_avgpool2d(
    input: [*]const f32,
    output: [*]f32,
    batch: usize,
    h: usize,
    w: usize,
    channels: usize,
) void {
    const spatial_size = h * w;
    const spatial_size_f: f32 = @floatFromInt(spatial_size);

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        const input_batch = input + n * spatial_size * channels;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            var sum: f32 = 0;

            var s: usize = 0;
            while (s < spatial_size) : (s += 1) {
                sum += input_batch[s * channels + c];
            }

            output[n * channels + c] = sum / spatial_size_f;
        }
    }
}

/// Global Max Pooling
/// Reduces (N, H, W, C) to (N, C)
pub fn global_maxpool2d(
    input: [*]const f32,
    output: [*]f32,
    batch: usize,
    h: usize,
    w: usize,
    channels: usize,
) void {
    const spatial_size = h * w;

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        const input_batch = input + n * spatial_size * channels;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            var max_val: f32 = -math.floatMax(f32);

            var s: usize = 0;
            while (s < spatial_size) : (s += 1) {
                const val = input_batch[s * channels + c];
                if (val > max_val) max_val = val;
            }

            output[n * channels + c] = max_val;
        }
    }
}

/// Adaptive Average Pooling
/// Produces fixed output size regardless of input
pub fn adaptive_avgpool2d(
    input: [*]const f32,
    output: [*]f32,
    batch: usize,
    h_in: usize,
    w_in: usize,
    channels: usize,
    h_out: usize,
    w_out: usize,
) void {
    const input_hw = h_in * w_in;
    const output_hw = h_out * w_out;

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        const input_batch = input + n * input_hw * channels;
        const output_batch = output + n * output_hw * channels;

        var c: usize = 0;
        while (c < channels) : (c += 1) {
            var oh: usize = 0;
            while (oh < h_out) : (oh += 1) {
                var ow: usize = 0;
                while (ow < w_out) : (ow += 1) {
                    // Compute window bounds
                    const h_start = oh * h_in / h_out;
                    const h_end = (oh + 1) * h_in / h_out;
                    const w_start = ow * w_in / w_out;
                    const w_end = (ow + 1) * w_in / w_out;

                    var sum: f32 = 0;
                    var count: usize = 0;

                    var ih = h_start;
                    while (ih < h_end) : (ih += 1) {
                        var iw = w_start;
                        while (iw < w_end) : (iw += 1) {
                            sum += input_batch[ih * w_in * channels + iw * channels + c];
                            count += 1;
                        }
                    }

                    const count_f: f32 = @floatFromInt(count);
                    output_batch[oh * w_out * channels + ow * channels + c] = sum / count_f;
                }
            }
        }
    }
}
