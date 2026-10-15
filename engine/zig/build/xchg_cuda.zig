//! The three-node exchange's lab tool (qualification_12f4270a) and its CPU tests: tf-xchg for -Dtarget, and the protocol over fake peers in zig build test; both carry fn_xchg.cu's CPU build (xchg_emu.cpp).
const std = @import("std");

fn module(b: *std.Build, path: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, observe: bool) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path(path), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    m.addCSourceFile(.{ .file = b.path(if (observe) "zig/tests/cuda/xchg/trace_emu.cpp" else "zig/tests/cuda/xchg/xchg_emu.cpp"), .flags = if (observe) &.{ "-std=c++20", "-O2", "-ffp-contract=off", "-DTF_ping_pong_fixture_TRACE=1" } else &.{ "-std=c++20", "-O2", "-ffp-contract=off" } });
    return m;
}

/// `zig build xchg`: tf-xchg over `cuda` (fn_xchg's fatbin embedded when built with kernels), outside `install`.
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-xchg", .root_module = module(b, "zig/tests/cuda/tf_xchg.zig", target, optimize, cuda, false) });
    b.step("xchg", "tf-xchg: the per-MoE-layer exchange between an attention GPU and two expert halves; exact data, round trips, silent and dead experts").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

/// zig build test: the protocol against fake peers (exact counts, bounded waits, silent and dead experts), the handshake's
/// window (qualification_1fee6d78: window.zig), and the tool's CPU side.
pub fn hostTests(b: *std.Build, step: *std.Build.Step, cuda: *std.Build.Module) void {
    for ([_][]const u8{ "zig/tests/cuda/xchg/protocol.zig", "zig/tests/cuda/xchg/window.zig", "zig/tests/cuda/tf_xchg.zig" }) |path| {
        step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, path, b.graph.host, .debug, cuda, std.mem.eql(u8, path, "zig/tests/cuda/xchg/protocol.zig")) })).step);
    }
}
