// SPDX-License-Identifier: MPL-2.0
// Reproduction: batch_size == 5 underflow in threading.parallel_layernorm /
// parallel_rmsnorm / parallel_batch (softmax).
//
// Condition to reach the parallel path: total >= 8192 and batch_size >= 4.
// With batch_size = 5: num_threads = min(4, 5) = 4; chunk = ceil(5/4) = 2;
// last_start = (4-1)*2 = 6 > 5  =>  `batch_size - last_start` wraps (usize).
//
// Build/run against the repository's zig/src tree:
//   zig test -OReleaseSafe --dep axiom -Mroot=repro_underflow.zig -Maxiom=../../../../zig/src/axiom.zig
const std = @import("std");
const axiom = @import("axiom");

test "layernorm batch_size=5 hidden=2048 reaches usize underflow" {
    const B: usize = 5;
    const H: usize = 2048; // 5*2048 = 10240 >= BATCH_THREAD_THRESHOLD (8192)
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, B * H);
    defer alloc.free(x);
    const y = try alloc.alloc(f32, B * H);
    defer alloc.free(y);
    const gamma = try alloc.alloc(f32, H);
    defer alloc.free(gamma);
    const beta = try alloc.alloc(f32, H);
    defer alloc.free(beta);
    for (x, 0..) |*v, i| v.* = @floatFromInt(i % 17);
    @memset(gamma, 1.0);
    @memset(beta, 0.0);
    @memset(y, 0.0);
    // In ReleaseSafe/Debug this panics with "integer overflow" (or out-of-bounds).
    // In ReleaseFast it silently reads/writes out of bounds.
    axiom.threading.parallel_layernorm(x.ptr, y.ptr, gamma.ptr, beta.ptr, B, H, 1e-5);
}

test "softmax batch_size=5 classes=2048 reaches usize underflow" {
    const B: usize = 5;
    const C: usize = 2048;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, B * C);
    defer alloc.free(x);
    const y = try alloc.alloc(f32, B * C);
    defer alloc.free(y);
    for (x, 0..) |*v, i| v.* = @floatFromInt(i % 13);
    axiom.threading.parallel_softmax_batched(x.ptr, y.ptr, B, C);
}
