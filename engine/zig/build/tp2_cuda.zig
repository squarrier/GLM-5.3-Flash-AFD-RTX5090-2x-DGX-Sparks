//! The two-rank layer's test tool (Mia's patch 0006, lifted family-agnostic into zig/src/cuda): tf-tp2-gather for -Dtarget, and its CPU side in zig build test.
const std = @import("std");

/// `zig build tp2-gather`: tf-tp2-gather over `cuda` (the runtime, fn_roce's fatbin embedded when built with kernels), outside `install`.
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-tp2-gather", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/tp2_gather.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} }) });
    b.step("tp2-gather", "tf-tp2-gather: NCCL on both rails against the one-shot RoCE all-gather between two GPUs, 16 KiB to 32 MiB a rank").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// zig build test: tf-tp2-gather's CPU side (sizes, expected words, statistics, gates, flags, rails from a sysfs tree), the tool itself compiled for the host.
pub fn hostTests(b: *std.Build, step: *std.Build.Step, cuda: *std.Build.Module) void {
    for ([_][]const u8{ "zig/tests/cuda/tp2/cases.zig", "zig/tests/cuda/tp2/rails.zig", "zig/tests/cuda/tp2/roce_cpu.zig", "zig/tests/cuda/tp2_gather.zig" }) |path| {
        const m = b.createModule(.{ .root_source_file = b.path(path), .target = b.graph.host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    }
}
