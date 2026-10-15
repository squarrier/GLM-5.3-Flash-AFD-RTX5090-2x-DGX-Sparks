//! ping_pong_fixture diagnostic-only build; no changes to the canonical product build graph.
//! Sources: Ash Hart's TensorFold and inherited private host-test wiring; notices unchanged.
//! Assisted-by: GPT-6.1 Sol (OpenAI Codex; planning and execution), via Hermes.
const std = @import("std");
const glm_build = @import("zig/build/glm_cuda.zig");
pub fn build(b: *std.Build) void {
    const opt: std.builtin.OptimizeMode = if (b.option(bool, "safe", "ReleaseSafe") orelse false) .safe else .debug;
    const host = b.graph.host;
    const options = b.addOptions(); options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = host, .optimize = opt, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addCSourceFile(.{ .file = b.path("zig/src/cuda/roce_proxy_stub.c"), .flags = &.{"-std=gnu11", "-O2"} });
    const stagger = b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = host, .optimize = opt });
    cuda.addImport("stagger", stagger);
    const filter = b.option([]const u8, "filter", "Exact diagnostic test name substring");
    const roots = [_][]const u8{ "zig/tests/cuda/xchg/fake.zig", "zig/tests/cuda/xchg/protocol.zig", "zig/tests/cuda/afd_attn_fail.zig" };
    const names = [_][]const u8{ "cursor", "protocol", "attn" };
    for (roots, names, 0..) |path, name, i| {
        const root = b.createModule(.{ .root_source_file = b.path(path), .target = host, .optimize = opt, .link_libc = true, .link_libcpp = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
        root.addCSourceFile(.{ .file = b.path("zig/tests/cuda/xchg/trace_emu.cpp"), .flags = &.{"-std=c++20", "-O2", "-ffp-contract=off", "-DTF_ping_pong_fixture_TRACE=1"} });
        if (i == 2) {
            const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = host, .optimize = opt, .link_libc = true });
            const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = host, .optimize = opt, .link_libc = true });
            core.addImport("tokenizer", tokenizer);
            const glm = glm_build.module(b, host, opt, cuda, core); root.addImport("glm_cuda", glm);
            root.addAnonymousImport("forward_exl3_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_forward_exl3.json") });
            const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = opt, .link_libc = true });
            glm.addImport("lanes", lanes); root.addImport("lanes", lanes);
            root.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = opt, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
            root.addImport("glm_lanes", b.createModule(.{ .root_source_file = b.path("zig/src/native/glm_cuda_lanes.zig"), .target = host, .optimize = opt, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
            const data = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/grammar_tests_data.zig"), .target = host, .optimize = opt, .link_libc = true });
            data.addImport("glm_cuda", glm); data.addImport("lanes", lanes); root.addImport("grammar_tests_data", data);
            root.addImport("grammar_tests_c_sync", b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/grammar_tests_c_rendezvous.zig"), .target = host, .optimize = opt, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
        }
        const t = b.addTest(.{ .root_module = root, .filters = if (filter) |f| &.{f} else &.{} });
        const run = b.addRunArtifact(t); run.has_side_effects = true;
        b.step(name, "ping_pong_fixture causal fixture only, not qualification").dependOn(&run.step);
        b.step(b.fmt("install-{s}", .{name}), "Install diagnostic binary").dependOn(&b.addInstallArtifact(t, .{ .dest_sub_path = b.fmt("ping_pong_fixture-{s}", .{name}) }).step);
    }
}
