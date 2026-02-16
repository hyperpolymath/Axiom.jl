// Axiom.jl Zig FFI integration tests.
// SPDX-License-Identifier: PMPL-1.0-or-later

const std = @import("std");
const testing = std.testing;

const Handle = ?*anyopaque;

extern fn axiom_init() Handle;
extern fn axiom_free(Handle) void;
extern fn axiom_process(Handle, u32) c_int;
extern fn axiom_get_string(Handle) ?[*:0]const u8;
extern fn axiom_free_string(?[*:0]const u8) void;
extern fn axiom_last_error() ?[*:0]const u8;
extern fn axiom_version() [*:0]const u8;
extern fn axiom_build_info() [*:0]const u8;
extern fn axiom_process_array(Handle, ?[*]const u8, u32) c_int;
extern fn axiom_register_callback(Handle, u64) c_int;
extern fn axiom_invoke_callback(Handle, u64, u32, ?*u32) c_int;
extern fn axiom_is_initialized(Handle) u32;

fn mulAddCallback(ctx: u64, input: u32) callconv(.c) u32 {
    return @as(u32, @truncate(ctx)) * 2 + input;
}

test "init/free lifecycle" {
    const h = axiom_init() orelse return error.InitFailed;
    try testing.expectEqual(@as(u32, 1), axiom_is_initialized(h));
    axiom_free(h);
    try testing.expectEqual(@as(u32, 0), axiom_is_initialized(h));
}

test "process works and null handle reports error" {
    const h = axiom_init() orelse return error.InitFailed;
    defer axiom_free(h);

    try testing.expectEqual(@as(c_int, 0), axiom_process(h, 42));
    try testing.expectEqual(@as(c_int, 4), axiom_process(null, 42));
    try testing.expect(axiom_last_error() != null);
}

test "get string and free string" {
    const h = axiom_init() orelse return error.InitFailed;
    defer axiom_free(h);

    _ = axiom_process(h, 123);
    const msg = axiom_get_string(h) orelse return error.MissingString;
    defer axiom_free_string(msg);

    const txt = std.mem.span(msg);
    try testing.expect(std.mem.indexOf(u8, txt, "123") != null);
}

test "process array updates internal state" {
    const h = axiom_init() orelse return error.InitFailed;
    defer axiom_free(h);

    const buf = [_]u8{ 1, 2, 3, 4 };
    try testing.expectEqual(@as(c_int, 0), axiom_process_array(h, &buf, buf.len));
    const msg = axiom_get_string(h) orelse return error.MissingString;
    defer axiom_free_string(msg);
    const txt = std.mem.span(msg);
    try testing.expect(std.mem.indexOf(u8, txt, "10") != null); // checksum 1+2+3+4
}

test "register and invoke callback" {
    const h = axiom_init() orelse return error.InitFailed;
    defer axiom_free(h);

    const cb_ptr = @as(u64, @intFromPtr(&mulAddCallback));
    try testing.expectEqual(@as(c_int, 0), axiom_register_callback(h, cb_ptr));

    var out: u32 = 0;
    try testing.expectEqual(@as(c_int, 0), axiom_invoke_callback(h, 7, 5, &out));
    try testing.expectEqual(@as(u32, 19), out); // 7*2 + 5
}

test "version/build info strings are present" {
    const ver = std.mem.span(axiom_version());
    const build = std.mem.span(axiom_build_info());
    try testing.expect(ver.len > 0);
    try testing.expect(build.len > 0);
}
