// SPDX-License-Identifier: MPL-2.0
// Does the chunking underflow in parallel_layernorm (threading.zig:392) write out of bounds
// in ReleaseFast? Output buffer is followed by a canary region; rows are checked against a
// scalar reference. Run:
//   zig test --dep axiom -Mroot=repro_underflow_canary.zig -Maxiom=../../../../zig/src/threading.zig -OReleaseFast
//   zig test --dep axiom -Mroot=repro_underflow_canary.zig -Maxiom=../../../../zig/src/threading.zig -OReleaseSafe   (panics)
const std = @import("std");
const threading = @import("axiom");

test "parallel_layernorm B=5 H=2048: canary after output" {
    const B: usize = 5;
    const H: usize = 2048;
    const n = B * H;
    const canary_len: usize = 4 * H;
    var x: [n]f32 = undefined;
    var y: [n + 4 * 2048]f32 = undefined; // output + canary
    var gamma: [H]f32 = undefined;
    var beta: [H]f32 = undefined;
    for (0..n) |i| x[i] = @floatFromInt((i * 7919) % 101);
    for (0..H) |i| {
        gamma[i] = 1.0;
        beta[i] = 0.0;
    }
    for (0..n + canary_len) |i| y[i] = 12345.0;
    threading.parallel_layernorm(&x, &y, &gamma, &beta, B, H, 1e-5);
    // rows
    var bad_rows: usize = 0;
    for (0..B) |b| {
        var mean: f64 = 0;
        for (0..H) |i| mean += x[b * H + i];
        mean /= @floatFromInt(H);
        var v: f64 = 0;
        for (0..H) |i| {
            const d = x[b * H + i] - mean;
            v += d * d;
        }
        v /= @floatFromInt(H);
        const inv = 1.0 / @sqrt(v + 1e-5);
        var row_bad = false;
        for (0..H) |i| {
            const ref: f32 = @floatCast((x[b * H + i] - mean) * inv);
            if (@abs(y[b * H + i] - ref) > 1e-3) row_bad = true;
        }
        if (row_bad) bad_rows += 1;
    }
    var canary_hits: usize = 0;
    for (n..n + canary_len) |i| {
        if (y[i] != 12345.0) canary_hits += 1;
    }
    std.debug.print("\nbad_rows={d} canary_overwritten={d}/{d}\n", .{ bad_rows, canary_hits, canary_len });
    try std.testing.expectEqual(@as(usize, 0), bad_rows);
    try std.testing.expectEqual(@as(usize, 0), canary_hits);
}
