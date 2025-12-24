//! Convolution Operations
//!
//! Optimized 2D convolution with SIMD.

const std = @import("std");

const VEC_SIZE = 8;
const Vec = @Vector(VEC_SIZE, f32);

/// 2D Convolution
/// Input: (N, H, W, C_in), Weight: (kH, kW, C_in, C_out), Output: (N, H_out, W_out, C_out)
pub fn conv2d(
    input: [*]const f32,
    weight: [*]const f32,
    bias: ?[*]const f32,
    output: [*]f32,
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
    const input_hw = h_in * w_in;
    const output_hw = h_out * w_out;
    const kernel_size = kh * kw * c_in;

    // Process each batch
    var n: usize = 0;
    while (n < batch) : (n += 1) {
        const input_batch = input + n * input_hw * c_in;
        const output_batch = output + n * output_hw * c_out;

        // Process each output channel
        var oc: usize = 0;
        while (oc < c_out) : (oc += 1) {
            const weight_oc = weight + oc * kernel_size;
            const bias_val: f32 = if (bias) |b| b[oc] else 0;

            // Process each output position
            var oh: usize = 0;
            while (oh < h_out) : (oh += 1) {
                var ow: usize = 0;
                while (ow < w_out) : (ow += 1) {
                    var sum: f32 = bias_val;

                    // Convolution kernel
                    var ki: usize = 0;
                    while (ki < kh) : (ki += 1) {
                        const ih = oh * stride_h + ki;
                        if (ih < pad_h or ih >= h_in + pad_h) continue;
                        const ih_actual = ih - pad_h;

                        var kj: usize = 0;
                        while (kj < kw) : (kj += 1) {
                            const iw = ow * stride_w + kj;
                            if (iw < pad_w or iw >= w_in + pad_w) continue;
                            const iw_actual = iw - pad_w;

                            // Dot product over input channels
                            var ic: usize = 0;
                            while (ic < c_in) : (ic += 1) {
                                const in_idx = ih_actual * w_in * c_in + iw_actual * c_in + ic;
                                const w_idx = ki * kw * c_in + kj * c_in + ic;
                                sum += input_batch[in_idx] * weight_oc[w_idx];
                            }
                        }
                    }

                    const out_idx = oh * w_out * c_out + ow * c_out + oc;
                    output_batch[out_idx] = sum;
                }
            }
        }
    }
}

/// 1x1 Convolution (pointwise) - optimized special case
pub fn conv1x1(
    input: [*]const f32,
    weight: [*]const f32,
    bias: ?[*]const f32,
    output: [*]f32,
    batch: usize,
    h: usize,
    w: usize,
    c_in: usize,
    c_out: usize,
) void {
    const hw = h * w;

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        const input_batch = input + n * hw * c_in;
        const output_batch = output + n * hw * c_out;

        // For each spatial position
        var pos: usize = 0;
        while (pos < hw) : (pos += 1) {
            const input_pos = input_batch + pos * c_in;

            // For each output channel
            var oc: usize = 0;
            while (oc < c_out) : (oc += 1) {
                const weight_oc = weight + oc * c_in;
                var sum: f32 = if (bias) |b| b[oc] else 0;

                // Dot product with SIMD
                var ic: usize = 0;
                var acc: Vec = @splat(0);
                while (ic + VEC_SIZE <= c_in) : (ic += VEC_SIZE) {
                    const in_vec: Vec = input_pos[ic..][0..VEC_SIZE].*;
                    const w_vec: Vec = weight_oc[ic..][0..VEC_SIZE].*;
                    acc += in_vec * w_vec;
                }
                sum += @reduce(.Add, acc);

                // Remainder
                while (ic < c_in) : (ic += 1) {
                    sum += input_pos[ic] * weight_oc[ic];
                }

                output_batch[pos * c_out + oc] = sum;
            }
        }
    }
}

/// Depthwise convolution
pub fn depthwise_conv2d(
    input: [*]const f32,
    weight: [*]const f32,
    bias: ?[*]const f32,
    output: [*]f32,
    batch: usize,
    h_in: usize,
    w_in: usize,
    channels: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    const input_hw = h_in * w_in;
    const output_hw = h_out * w_out;
    const kernel_size = kh * kw;

    var n: usize = 0;
    while (n < batch) : (n += 1) {
        const input_batch = input + n * input_hw * channels;
        const output_batch = output + n * output_hw * channels;

        // Process each channel independently
        var c: usize = 0;
        while (c < channels) : (c += 1) {
            const weight_c = weight + c * kernel_size;
            const bias_val: f32 = if (bias) |b| b[c] else 0;

            var oh: usize = 0;
            while (oh < h_out) : (oh += 1) {
                var ow: usize = 0;
                while (ow < w_out) : (ow += 1) {
                    var sum: f32 = bias_val;

                    var ki: usize = 0;
                    while (ki < kh) : (ki += 1) {
                        const ih = oh * stride_h + ki;
                        if (ih < pad_h or ih >= h_in + pad_h) continue;
                        const ih_actual = ih - pad_h;

                        var kj: usize = 0;
                        while (kj < kw) : (kj += 1) {
                            const iw = ow * stride_w + kj;
                            if (iw < pad_w or iw >= w_in + pad_w) continue;
                            const iw_actual = iw - pad_w;

                            const in_idx = ih_actual * w_in * channels + iw_actual * channels + c;
                            const w_idx = ki * kw + kj;
                            sum += input_batch[in_idx] * weight_c[w_idx];
                        }
                    }

                    const out_idx = oh * w_out * channels + ow * channels + c;
                    output_batch[out_idx] = sum;
                }
            }
        }
    }
}

/// im2col transformation for efficient convolution via GEMM
pub fn im2col(
    input: [*]const f32,
    output: [*]f32,
    batch: usize,
    h_in: usize,
    w_in: usize,
    c_in: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) void {
    const col_height = h_out * w_out;
    const col_width = c_in * kh * kw;
    _ = batch;

    var oh: usize = 0;
    while (oh < h_out) : (oh += 1) {
        var ow: usize = 0;
        while (ow < w_out) : (ow += 1) {
            const row_idx = oh * w_out + ow;

            var ki: usize = 0;
            while (ki < kh) : (ki += 1) {
                var kj: usize = 0;
                while (kj < kw) : (kj += 1) {
                    const ih = oh * stride_h + ki;
                    const iw = ow * stride_w + kj;

                    var c: usize = 0;
                    while (c < c_in) : (c += 1) {
                        const col_idx = ki * kw * c_in + kj * c_in + c;

                        if (ih >= pad_h and ih < h_in + pad_h and
                            iw >= pad_w and iw < w_in + pad_w)
                        {
                            const ih_actual = ih - pad_h;
                            const iw_actual = iw - pad_w;
                            const in_idx = ih_actual * w_in * c_in + iw_actual * c_in + c;
                            output[row_idx * col_width + col_idx] = input[in_idx];
                        } else {
                            output[row_idx * col_width + col_idx] = 0;
                        }
                    }
                }
            }
        }
    }
    _ = col_height;
}
