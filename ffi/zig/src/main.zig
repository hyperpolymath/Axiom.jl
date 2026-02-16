// Axiom.jl Zig FFI implementation.
//
// This implements the C symbols declared in `src/Abi/Foreign.idr`.
// The design here is a concrete bidirectional scaffold:
// - Host -> runtime calls (`axiom_process`, `axiom_process_array`, ...)
// - Runtime -> host callbacks (`axiom_register_callback`, `axiom_invoke_callback`)
//
// SPDX-License-Identifier: PMPL-1.0-or-later

const std = @import("std");
const builtin = @import("builtin");

const VERSION: [:0]const u8 = "0.1.0";
const BUILD_INFO: [:0]const u8 = "Axiom Zig FFI built with Zig " ++ builtin.zig_version_string;
const HANDLE_MAGIC: u64 = 0x4158494f4d5f4646; // "AXIOM_FF"

pub const Result = enum(c_int) {
    ok = 0,
    @"error" = 1,
    invalid_param = 2,
    out_of_memory = 3,
    null_pointer = 4,
};

pub const Callback = *const fn (u64, u32) callconv(.c) u32;

const AxiomHandle = struct {
    magic: u64 = HANDLE_MAGIC,
    initialized: bool = true,
    last_input: u32 = 0,
    callback: ?Callback = null,
    callback_ctx: u64 = 0,
};

threadlocal var last_error_buf: [256:0]u8 = [_:0]u8{0} ** 256;

fn setError(msg: []const u8) void {
    const n = @min(msg.len, last_error_buf.len - 1);
    std.mem.copyForwards(u8, last_error_buf[0..n], msg[0..n]);
    last_error_buf[n] = 0;
}

fn clearError() void {
    last_error_buf[0] = 0;
}

fn handleOrResult(handle: ?*AxiomHandle) ?*AxiomHandle {
    const h = handle orelse {
        setError("Null handle");
        return null;
    };
    if (h.magic != HANDLE_MAGIC) {
        setError("Invalid handle");
        return null;
    }
    return h;
}

export fn axiom_init() ?*AxiomHandle {
    const h = std.heap.c_allocator.create(AxiomHandle) catch {
        setError("Failed to allocate handle");
        return null;
    };
    h.* = .{};
    clearError();
    return h;
}

export fn axiom_free(handle: ?*AxiomHandle) void {
    const h = handle orelse return;
    if (h.magic != HANDLE_MAGIC) {
        return;
    }
    // Intentionally idempotent for scaffold safety (avoid double-free crashes in FFI tests).
    // A production version should add explicit ownership tracking and deterministic cleanup.
    h.initialized = false;
    h.callback = null;
    clearError();
}

export fn axiom_process(handle: ?*AxiomHandle, input: u32) c_int {
    const h = handleOrResult(handle) orelse return @intFromEnum(Result.null_pointer);
    if (!h.initialized) {
        setError("Handle not initialized");
        return @intFromEnum(Result.@"error");
    }

    h.last_input = input;
    if (h.callback) |cb| {
        _ = cb(h.callback_ctx, input);
    }

    clearError();
    return @intFromEnum(Result.ok);
}

export fn axiom_get_string(handle: ?*AxiomHandle) ?[*:0]const u8 {
    const h = handleOrResult(handle) orelse return null;
    if (!h.initialized) {
        setError("Handle not initialized");
        return null;
    }

    var tmp: [128]u8 = undefined;
    const msg = std.fmt.bufPrint(&tmp, "axiom-last-input={d}", .{h.last_input}) catch {
        setError("Failed to format string");
        return null;
    };

    const out = std.heap.c_allocator.dupeZ(u8, msg) catch {
        setError("Failed to allocate string");
        return null;
    };

    clearError();
    return out.ptr;
}

export fn axiom_free_string(str: ?[*:0]const u8) void {
    const s = str orelse return;
    const span = std.mem.span(s);
    std.heap.c_allocator.free(@constCast(span));
}

export fn axiom_process_array(handle: ?*AxiomHandle, buffer: ?[*]const u8, len: u32) c_int {
    const h = handleOrResult(handle) orelse return @intFromEnum(Result.null_pointer);
    if (!h.initialized) {
        setError("Handle not initialized");
        return @intFromEnum(Result.@"error");
    }
    const buf = buffer orelse {
        setError("Null buffer");
        return @intFromEnum(Result.null_pointer);
    };

    const slice = buf[0..len];
    var checksum: u32 = 0;
    for (slice) |b| checksum +%= @as(u32, b);
    h.last_input = checksum;

    if (h.callback) |cb| {
        _ = cb(h.callback_ctx, checksum);
    }

    clearError();
    return @intFromEnum(Result.ok);
}

export fn axiom_last_error() ?[*:0]const u8 {
    if (last_error_buf[0] == 0) {
        return null;
    }
    return @as([*:0]const u8, @ptrCast(&last_error_buf));
}

export fn axiom_version() [*:0]const u8 {
    return VERSION.ptr;
}

export fn axiom_build_info() [*:0]const u8 {
    return BUILD_INFO.ptr;
}

export fn axiom_register_callback(handle: ?*AxiomHandle, callback_ptr: u64) c_int {
    const h = handleOrResult(handle) orelse return @intFromEnum(Result.null_pointer);
    if (!h.initialized) {
        setError("Handle not initialized");
        return @intFromEnum(Result.@"error");
    }
    if (callback_ptr == 0) {
        setError("Null callback pointer");
        return @intFromEnum(Result.null_pointer);
    }

    const cb: Callback = @ptrFromInt(callback_ptr);
    h.callback = cb;
    h.callback_ctx = @as(u64, @intFromPtr(h));
    clearError();
    return @intFromEnum(Result.ok);
}

export fn axiom_invoke_callback(handle: ?*AxiomHandle, ctx: u64, input: u32, out_ptr: ?*u32) c_int {
    const h = handleOrResult(handle) orelse return @intFromEnum(Result.null_pointer);
    if (!h.initialized) {
        setError("Handle not initialized");
        return @intFromEnum(Result.@"error");
    }
    const out = out_ptr orelse {
        setError("Null output pointer");
        return @intFromEnum(Result.null_pointer);
    };
    const cb = h.callback orelse {
        setError("No callback registered");
        return @intFromEnum(Result.invalid_param);
    };

    out.* = cb(ctx, input);
    clearError();
    return @intFromEnum(Result.ok);
}

export fn axiom_is_initialized(handle: ?*AxiomHandle) u32 {
    const h = handle orelse return 0;
    if (h.magic != HANDLE_MAGIC) return 0;
    return if (h.initialized) 1 else 0;
}

fn addCallback(ctx: u64, input: u32) callconv(.c) u32 {
    return @as(u32, @truncate(ctx)) + input;
}

test "lifecycle" {
    const h = axiom_init() orelse return error.InitFailed;
    try std.testing.expectEqual(@as(u32, 1), axiom_is_initialized(h));
    axiom_free(h);
    try std.testing.expectEqual(@as(u32, 0), axiom_is_initialized(h));
}

test "null handle process yields null_pointer" {
    try std.testing.expectEqual(@as(c_int, @intFromEnum(Result.null_pointer)), axiom_process(null, 7));
    try std.testing.expect(axiom_last_error() != null);
}

test "string round-trip" {
    const h = axiom_init() orelse return error.InitFailed;
    defer axiom_free(h);
    _ = axiom_process(h, 42);
    const str = axiom_get_string(h) orelse return error.NoString;
    defer axiom_free_string(str);
    const s = std.mem.span(str);
    try std.testing.expect(std.mem.indexOf(u8, s, "42") != null);
}

test "callback registration and invocation" {
    const h = axiom_init() orelse return error.InitFailed;
    defer axiom_free(h);

    const cb_ptr = @as(u64, @intFromPtr(&addCallback));
    try std.testing.expectEqual(@as(c_int, @intFromEnum(Result.ok)), axiom_register_callback(h, cb_ptr));

    var out: u32 = 0;
    try std.testing.expectEqual(
        @as(c_int, @intFromEnum(Result.ok)),
        axiom_invoke_callback(h, 5, 10, &out),
    );
    try std.testing.expectEqual(@as(u32, 15), out);
}
