// Axiom.jl Zig FFI Build Configuration
// SPDX-License-Identifier: PMPL-1.0-or-later

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "axiom",
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    const lib_static = b.addLibrary(.{
        .linkage = .static,
        .name = "axiom",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    b.installArtifact(lib);
    b.installArtifact(lib_static);

    const header = b.addInstallHeaderFile(
        b.path("include/axiom.h"),
        "axiom.h",
    );
    b.getInstallStep().dependOn(&header.step);

    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("test/integration_test.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    integration_tests.linkLibrary(lib_static);
    const run_integration_tests = b.addRunArtifact(integration_tests);

    const test_step = b.step("test", "Run unit and integration tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_integration_tests.step);

    const integration_step = b.step("test-integration", "Run integration tests only");
    integration_step.dependOn(&run_integration_tests.step);
}
