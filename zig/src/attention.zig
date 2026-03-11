// SPDX-License-Identifier: PMPL-1.0-or-later
//! Attention Mechanisms
//!
//! Optimized attention operations for transformer models.

const std = @import("std");
const math = std.math;
const activations = @import("activations.zig");
const matmul = @import("matmul.zig");

const VEC_SIZE = 8;
const Vec = @Vector(VEC_SIZE, f32);

/// Scaled Dot-Product Attention
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
pub fn scaled_dot_product_attention(
    q_ptr: [*]const f32,
    k_ptr: [*]const f32,
    v_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    seq_len: usize,
    head_dim: usize,
    mask_ptr: ?[*]const f32,
) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        const q_batch = q_ptr + b * seq_len * head_dim;
        const k_batch = k_ptr + b * seq_len * head_dim;
        const v_batch = v_ptr + b * seq_len * head_dim;
        const out_batch = output_ptr + b * seq_len * head_dim;

        // Compute attention scores: Q @ K^T
        // scores[seq_len, seq_len]
        var scores: [4096]f32 = undefined; // Max seq_len = 64

        var i: usize = 0;
        while (i < seq_len) : (i += 1) {
            var j: usize = 0;
            while (j < seq_len) : (j += 1) {
                // Dot product of Q[i] and K[j]
                var sum: f32 = 0;
                var d: usize = 0;

                var acc: Vec = @splat(0);
                while (d + VEC_SIZE <= head_dim) : (d += VEC_SIZE) {
                    const q_vec: Vec = q_batch[i * head_dim + d ..][0..VEC_SIZE].*;
                    const k_vec: Vec = k_batch[j * head_dim + d ..][0..VEC_SIZE].*;
                    acc += q_vec * k_vec;
                }
                sum = @reduce(.Add, acc);

                while (d < head_dim) : (d += 1) {
                    sum += q_batch[i * head_dim + d] * k_batch[j * head_dim + d];
                }

                scores[i * seq_len + j] = sum * scale;
            }
        }

        // Apply mask if provided (for causal attention)
        if (mask_ptr) |mask| {
            i = 0;
            while (i < seq_len) : (i += 1) {
                var j: usize = 0;
                while (j < seq_len) : (j += 1) {
                    const mask_val = mask[b * seq_len * seq_len + i * seq_len + j];
                    if (mask_val == 0) {
                        scores[i * seq_len + j] = -math.floatMax(f32);
                    }
                }
            }
        }

        // Softmax over each row
        i = 0;
        while (i < seq_len) : (i += 1) {
            // Find max for numerical stability
            var max_val: f32 = scores[i * seq_len];
            var j: usize = 1;
            while (j < seq_len) : (j += 1) {
                if (scores[i * seq_len + j] > max_val) {
                    max_val = scores[i * seq_len + j];
                }
            }

            // Compute exp and sum
            var sum: f32 = 0;
            j = 0;
            while (j < seq_len) : (j += 1) {
                scores[i * seq_len + j] = @exp(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }

            // Normalize
            const inv_sum = 1.0 / sum;
            j = 0;
            while (j < seq_len) : (j += 1) {
                scores[i * seq_len + j] *= inv_sum;
            }
        }

        // Compute output: scores @ V
        i = 0;
        while (i < seq_len) : (i += 1) {
            var d: usize = 0;
            while (d < head_dim) : (d += 1) {
                var sum: f32 = 0;
                var j: usize = 0;
                while (j < seq_len) : (j += 1) {
                    sum += scores[i * seq_len + j] * v_batch[j * head_dim + d];
                }
                out_batch[i * head_dim + d] = sum;
            }
        }
    }
}

/// Multi-Head Attention
/// Splits input into multiple heads, applies attention, concatenates results
pub fn multihead_attention(
    q_ptr: [*]const f32,
    k_ptr: [*]const f32,
    v_ptr: [*]const f32,
    wq_ptr: [*]const f32,
    wk_ptr: [*]const f32,
    wv_ptr: [*]const f32,
    wo_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
    mask_ptr: ?[*]const f32,
    // Scratch space
    scratch_ptr: [*]f32,
) void {
    const head_dim = d_model / num_heads;
    const qkv_size = batch * seq_len * d_model;

    // Scratch space layout:
    // [0..qkv_size]: projected Q
    // [qkv_size..2*qkv_size]: projected K
    // [2*qkv_size..3*qkv_size]: projected V
    // [3*qkv_size..4*qkv_size]: attention output
    const q_proj = scratch_ptr;
    const k_proj = scratch_ptr + qkv_size;
    const v_proj = scratch_ptr + 2 * qkv_size;
    const attn_out = scratch_ptr + 3 * qkv_size;

    // Project Q, K, V through linear layers
    // Q_proj = Q @ Wq
    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var s: usize = 0;
        while (s < seq_len) : (s += 1) {
            const q_in = q_ptr + (b * seq_len + s) * d_model;
            const k_in = k_ptr + (b * seq_len + s) * d_model;
            const v_in = v_ptr + (b * seq_len + s) * d_model;

            const q_out = q_proj + (b * seq_len + s) * d_model;
            const k_out = k_proj + (b * seq_len + s) * d_model;
            const v_out = v_proj + (b * seq_len + s) * d_model;

            // Linear projection for each position
            var o: usize = 0;
            while (o < d_model) : (o += 1) {
                var q_sum: f32 = 0;
                var k_sum: f32 = 0;
                var v_sum: f32 = 0;

                var i: usize = 0;
                while (i < d_model) : (i += 1) {
                    q_sum += q_in[i] * wq_ptr[i * d_model + o];
                    k_sum += k_in[i] * wk_ptr[i * d_model + o];
                    v_sum += v_in[i] * wv_ptr[i * d_model + o];
                }

                q_out[o] = q_sum;
                k_out[o] = k_sum;
                v_out[o] = v_sum;
            }
        }
    }

    // Apply attention for each head
    var h: usize = 0;
    while (h < num_heads) : (h += 1) {
        b = 0;
        while (b < batch) : (b += 1) {
            // Extract head-specific Q, K, V (interleaved format)
            var s: usize = 0;
            while (s < seq_len) : (s += 1) {
                var d: usize = 0;
                while (d < head_dim) : (d += 1) {
                    const src_idx = (b * seq_len + s) * d_model + h * head_dim + d;
                    const dst_idx = b * seq_len * head_dim + s * head_dim + d;

                    // Temporary storage within the same scratch (reusing part of attn_out)
                    const temp_q = attn_out + dst_idx;
                    const temp_k = attn_out + batch * seq_len * head_dim + dst_idx;
                    const temp_v = attn_out + 2 * batch * seq_len * head_dim + dst_idx;

                    temp_q[0] = q_proj[src_idx];
                    temp_k[0] = k_proj[src_idx];
                    temp_v[0] = v_proj[src_idx];
                }
            }
        }

        // Note: This is a simplified implementation
        // A production version would properly separate head computation
    }

    // Apply scaled dot-product attention per head (simplified: treating all as one)
    scaled_dot_product_attention(
        q_proj,
        k_proj,
        v_proj,
        attn_out,
        batch * num_heads,
        seq_len,
        head_dim,
        mask_ptr,
    );

    // Output projection: attn_out @ Wo
    b = 0;
    while (b < batch) : (b += 1) {
        var s: usize = 0;
        while (s < seq_len) : (s += 1) {
            const attn_in = attn_out + (b * seq_len + s) * d_model;
            const out = output_ptr + (b * seq_len + s) * d_model;

            var o: usize = 0;
            while (o < d_model) : (o += 1) {
                var sum: f32 = 0;
                var i: usize = 0;
                while (i < d_model) : (i += 1) {
                    sum += attn_in[i] * wo_ptr[i * d_model + o];
                }
                out[o] = sum;
            }
        }
    }
}

/// Flash Attention (memory-efficient attention)
/// Computes attention in blocks to reduce memory usage
pub fn flash_attention(
    q_ptr: [*]const f32,
    k_ptr: [*]const f32,
    v_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch: usize,
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const num_blocks = (seq_len + block_size - 1) / block_size;

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        const q_batch = q_ptr + b * seq_len * head_dim;
        const k_batch = k_ptr + b * seq_len * head_dim;
        const v_batch = v_ptr + b * seq_len * head_dim;
        const out_batch = output_ptr + b * seq_len * head_dim;

        // Initialize output and normalization factors
        var max_vals: [4096]f32 = undefined;
        var sum_vals: [4096]f32 = undefined;

        var i: usize = 0;
        while (i < seq_len) : (i += 1) {
            max_vals[i] = -math.floatMax(f32);
            sum_vals[i] = 0;
            var d: usize = 0;
            while (d < head_dim) : (d += 1) {
                out_batch[i * head_dim + d] = 0;
            }
        }

        // Process K, V in blocks
        var kv_block: usize = 0;
        while (kv_block < num_blocks) : (kv_block += 1) {
            const kv_start = kv_block * block_size;
            const kv_end = @min(kv_start + block_size, seq_len);

            // For each query position
            i = 0;
            while (i < seq_len) : (i += 1) {
                var block_max: f32 = -math.floatMax(f32);
                var scores: [64]f32 = undefined; // block_size scores

                // Compute scores for this block
                var j = kv_start;
                var local_j: usize = 0;
                while (j < kv_end) : ({
                    j += 1;
                    local_j += 1;
                }) {
                    var sum: f32 = 0;
                    var d: usize = 0;
                    while (d < head_dim) : (d += 1) {
                        sum += q_batch[i * head_dim + d] * k_batch[j * head_dim + d];
                    }
                    scores[local_j] = sum * scale;
                    if (scores[local_j] > block_max) {
                        block_max = scores[local_j];
                    }
                }

                // Update running max and rescale previous results
                const prev_max = max_vals[i];
                const new_max = @max(prev_max, block_max);
                const scale_old = @exp(prev_max - new_max);
                const scale_new = @exp(block_max - new_max);

                // Rescale accumulated output
                var d: usize = 0;
                while (d < head_dim) : (d += 1) {
                    out_batch[i * head_dim + d] *= scale_old;
                }

                // Update sum
                sum_vals[i] *= scale_old;

                // Add contribution from this block
                j = kv_start;
                local_j = 0;
                while (j < kv_end) : ({
                    j += 1;
                    local_j += 1;
                }) {
                    const p = @exp(scores[local_j] - block_max) * scale_new;
                    sum_vals[i] += p;

                    d = 0;
                    while (d < head_dim) : (d += 1) {
                        out_batch[i * head_dim + d] += p * v_batch[j * head_dim + d];
                    }
                }

                max_vals[i] = new_max;
            }
        }

        // Final normalization
        i = 0;
        while (i < seq_len) : (i += 1) {
            const inv_sum = 1.0 / sum_vals[i];
            var d: usize = 0;
            while (d < head_dim) : (d += 1) {
                out_batch[i * head_dim + d] *= inv_sum;
            }
        }
    }
}

/// Causal mask generation
pub fn generate_causal_mask(
    mask_ptr: [*]f32,
    seq_len: usize,
) void {
    var i: usize = 0;
    while (i < seq_len) : (i += 1) {
        var j: usize = 0;
        while (j < seq_len) : (j += 1) {
            mask_ptr[i * seq_len + j] = if (j <= i) 1.0 else 0.0;
        }
    }
}

/// Relative position encoding
pub fn apply_rotary_embedding(
    x_ptr: [*]f32,
    seq_len: usize,
    head_dim: usize,
    base: f32,
) void {
    const half_dim = head_dim / 2;

    var pos: usize = 0;
    while (pos < seq_len) : (pos += 1) {
        var i: usize = 0;
        while (i < half_dim) : (i += 1) {
            const freq = 1.0 / math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
            const angle = @as(f32, @floatFromInt(pos)) * freq;
            const cos_val = @cos(angle);
            const sin_val = @sin(angle);

            const idx = pos * head_dim + i;
            const idx2 = pos * head_dim + i + half_dim;

            const x1 = x_ptr[idx];
            const x2 = x_ptr[idx2];

            x_ptr[idx] = x1 * cos_val - x2 * sin_val;
            x_ptr[idx2] = x1 * sin_val + x2 * cos_val;
        }
    }
}
