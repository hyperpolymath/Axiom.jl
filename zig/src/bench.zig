//! Benchmarks for Axiom Zig Backend
//!
//! Run with: zig build bench

const std = @import("std");
const time = std.time;

const matmul = @import("matmul.zig");
const activations = @import("activations.zig");
const conv = @import("conv.zig");
const norm = @import("norm.zig");
const attention = @import("attention.zig");

const Allocator = std.mem.Allocator;

fn bench(comptime name: []const u8, comptime f: anytype, args: anytype, iterations: usize) void {
    // Warmup
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        @call(.auto, f, args);
    }

    const start = time.nanoTimestamp();

    i = 0;
    while (i < iterations) : (i += 1) {
        @call(.auto, f, args);
    }

    const end = time.nanoTimestamp();
    const elapsed_ns = @as(u64, @intCast(end - start));
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;

    std.debug.print("{s}: {d:.2}ms total, {d:.2}μs/iter ({d} iters)\n", .{ name, elapsed_ms, per_iter_us, iterations });
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n=== Axiom Zig Backend Benchmarks ===\n\n", .{});

    // Matrix multiplication benchmarks
    try benchMatmul(allocator);

    // Activation function benchmarks
    try benchActivations(allocator);

    // Convolution benchmarks
    try benchConv(allocator);

    // Normalization benchmarks
    try benchNorm(allocator);

    // Attention benchmarks
    try benchAttention(allocator);

    std.debug.print("\n=== Benchmarks Complete ===\n", .{});
}

fn benchMatmul(allocator: Allocator) !void {
    std.debug.print("--- Matrix Multiplication ---\n", .{});

    // Small matrix (64x64)
    {
        const m = 64;
        const k = 64;
        const n = 64;
        const a = try allocator.alloc(f32, m * k);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, k * n);
        defer allocator.free(b);
        const c = try allocator.alloc(f32, m * n);
        defer allocator.free(c);

        for (a) |*v| v.* = 1.0;
        for (b) |*v| v.* = 1.0;

        bench("matmul 64x64", matmul.matmul_tiled, .{ a, b, c, m, k, n }, 10000);
    }

    // Medium matrix (256x256)
    {
        const m = 256;
        const k = 256;
        const n = 256;
        const a = try allocator.alloc(f32, m * k);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, k * n);
        defer allocator.free(b);
        const c = try allocator.alloc(f32, m * n);
        defer allocator.free(c);

        for (a) |*v| v.* = 1.0;
        for (b) |*v| v.* = 1.0;

        bench("matmul 256x256", matmul.matmul_tiled, .{ a, b, c, m, k, n }, 1000);
    }

    // Large matrix (1024x1024)
    {
        const m = 1024;
        const k = 1024;
        const n = 1024;
        const a = try allocator.alloc(f32, m * k);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, k * n);
        defer allocator.free(b);
        const c = try allocator.alloc(f32, m * n);
        defer allocator.free(c);

        for (a) |*v| v.* = 1.0;
        for (b) |*v| v.* = 1.0;

        bench("matmul 1024x1024", matmul.matmul_tiled, .{ a, b, c, m, k, n }, 100);
    }

    std.debug.print("\n", .{});
}

fn benchActivations(allocator: Allocator) !void {
    std.debug.print("--- Activation Functions ---\n", .{});

    const n = 1024 * 1024; // 1M elements
    const input = try allocator.alloc(f32, n);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, n);
    defer allocator.free(output);

    for (input, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i % 100)) / 50.0 - 1.0;
    }

    bench("relu 1M", activations.relu, .{ input, output }, 1000);
    bench("gelu 1M", activations.gelu, .{ input, output }, 100);
    bench("sigmoid 1M", activations.sigmoid, .{ input, output }, 100);
    bench("swish 1M", activations.swish, .{ input, output }, 100);

    std.debug.print("\n", .{});
}

fn benchConv(allocator: Allocator) !void {
    std.debug.print("--- Convolution ---\n", .{});

    // NHWC format: (1, 32, 32, 64) with 3x3 kernel to 64 channels
    const batch = 1;
    const h_in = 32;
    const w_in = 32;
    const c_in = 64;
    const c_out = 64;
    const kh = 3;
    const kw = 3;
    const h_out = 30;
    const w_out = 30;

    const input = try allocator.alloc(f32, batch * h_in * w_in * c_in);
    defer allocator.free(input);
    const weight = try allocator.alloc(f32, kh * kw * c_in * c_out);
    defer allocator.free(weight);
    const output = try allocator.alloc(f32, batch * h_out * w_out * c_out);
    defer allocator.free(output);

    for (input) |*v| v.* = 1.0;
    for (weight) |*v| v.* = 0.1;

    bench("conv2d 32x32x64 -> 30x30x64", conv.conv2d, .{
        input.ptr, weight.ptr, null, output.ptr, batch, h_in, w_in, c_in, h_out, w_out, c_out, kh, kw, 1, 1, 0, 0,
    }, 100);

    std.debug.print("\n", .{});
}

fn benchNorm(allocator: Allocator) !void {
    std.debug.print("--- Normalization ---\n", .{});

    const batch = 32;
    const hidden = 768;

    const input = try allocator.alloc(f32, batch * hidden);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, batch * hidden);
    defer allocator.free(output);
    const gamma = try allocator.alloc(f32, hidden);
    defer allocator.free(gamma);
    const beta = try allocator.alloc(f32, hidden);
    defer allocator.free(beta);

    for (input) |*v| v.* = 1.0;
    for (gamma) |*v| v.* = 1.0;
    for (beta) |*v| v.* = 0.0;

    bench("layernorm 32x768", norm.layernorm, .{
        input.ptr, output.ptr, gamma.ptr, beta.ptr, batch, hidden, 1e-5,
    }, 10000);

    bench("rmsnorm 32x768", norm.rmsnorm, .{
        input.ptr, output.ptr, gamma.ptr, batch, hidden, 1e-5,
    }, 10000);

    std.debug.print("\n", .{});
}

fn benchAttention(allocator: Allocator) !void {
    std.debug.print("--- Attention ---\n", .{});

    const batch = 1;
    const seq_len = 64;
    const head_dim = 64;

    const q = try allocator.alloc(f32, batch * seq_len * head_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, batch * seq_len * head_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, batch * seq_len * head_dim);
    defer allocator.free(v);
    const output = try allocator.alloc(f32, batch * seq_len * head_dim);
    defer allocator.free(output);

    for (q) |*x| x.* = 0.1;
    for (k) |*x| x.* = 0.1;
    for (v) |*x| x.* = 0.1;

    bench("sdpa 64x64", attention.scaled_dot_product_attention, .{
        q.ptr, k.ptr, v.ptr, output.ptr, batch, seq_len, head_dim, null,
    }, 1000);

    bench("flash_attn 64x64 (block=16)", attention.flash_attention, .{
        q.ptr, k.ptr, v.ptr, output.ptr, batch, seq_len, head_dim, 16,
    }, 1000);

    std.debug.print("\n", .{});
}
