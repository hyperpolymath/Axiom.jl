// SPDX-License-Identifier: MPL-2.0
// Reproduction: norm.batchnorm uses a fixed `[4096]f32` stack scratch buffer
// indexed by `num_features` with no bound check.  num_features > 4096 writes
// past the stack array.  In ReleaseSafe/Debug this is an index-out-of-bounds
// panic; through the shipped ReleaseFast .so it is silently wrong (4097..8191)
// and SIGSEGV from 8192 features (see batchnorm_probe.py).
// Run: zig test -OReleaseSafe --dep axiom -Mroot=repro_batchnorm.zig -Maxiom=../../../../zig/src/axiom.zig
const std = @import("std");
const axiom = @import("axiom");

test "batchnorm num_features=4097 overflows fixed stack scratch" {
    const B: usize = 2;
    const F: usize = 4097;
    const alloc = std.testing.allocator;
    const x = try alloc.alloc(f32, B * F);
    defer alloc.free(x);
    const y = try alloc.alloc(f32, B * F);
    defer alloc.free(y);
    const gamma = try alloc.alloc(f32, F);
    defer alloc.free(gamma);
    const beta = try alloc.alloc(f32, F);
    defer alloc.free(beta);
    const rmean = try alloc.alloc(f32, F);
    defer alloc.free(rmean);
    const rvar = try alloc.alloc(f32, F);
    defer alloc.free(rvar);
    @memset(x, 1.0);
    @memset(gamma, 1.0);
    @memset(beta, 0.0);
    @memset(rmean, 0.0);
    @memset(rvar, 1.0);
    axiom.norm.batchnorm(x.ptr, y.ptr, gamma.ptr, beta.ptr, rmean.ptr, rvar.ptr, B, F, 1e-5);
}
