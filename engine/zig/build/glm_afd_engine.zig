//! qualification_0fa2bee9's G-fail in `zig build test` (and `zig build test-afd-attn` alone): the AFD attention engine's steps over
//! native RoCE exchange with a silent and a dead expert, three nodes in one process on fn_xchg's CPU build (xchg_emu.cpp), the
//! experts running qualification_cec3c04c's loop, and the wire audit end to end. Kept out of the shared build files, as
//! glm_afd_expert.zig: build.zig calls `hostTests` once; no kernel images (the CPU tests launch nothing).

const std = @import("std");
const glm_build = @import("glm_cuda.zig");

/// The runtime module without kernel images, as zig/build/glm_fwd_tool.zig's `runtime`.
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    const verbs = target.result.os.tag == .linux and if (std.Io.Dir.accessAbsolute(b.graph.io, "/usr/include/infiniband/verbs.h", .{})) |_| true else |_| false;
    cuda.addCSourceFile(.{ .file = b.path(if (verbs) "zig/src/cuda/roce_proxy.c" else "zig/src/cuda/roce_proxy_stub.c"), .flags = &.{ "-std=gnu11", "-O2", "-idirafter", "/usr/include" } });
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// G-fail and the audit: `zig build test`, or `zig build test-afd-attn` alone.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug);
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = host, .optimize = .debug, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const glm = glm_build.module(b, host, .debug, cuda, core);
    const root = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/afd_attn_fail.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{ .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda } },
    });
    root.addCSourceFile(.{ .file = b.path("zig/tests/cuda/xchg/trace_emu.cpp"), .flags = &.{ "-std=c++20", "-O2", "-ffp-contract=off", "-DTF_ping_pong_fixture_TRACE=1" } });
    root.addAnonymousImport("forward_exl3_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_forward_exl3.json") });
    // qualification_4be15bbf: the served stream's gates (afd_serve_fail.zig): the lane core, the engine seam and qualification_qualification_e91d6b91's lane backend
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    glm.addImport("lanes", lanes);
    root.addImport("lanes", lanes);
    root.addImport("engine_api", b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
    root.addImport("glm_lanes", b.createModule(.{ .root_source_file = b.path("zig/src/native/glm_cuda_lanes.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
    const data = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/grammar_tests_data.zig"), .target = host, .optimize = .debug, .link_libc = true });
    data.addImport("glm_cuda", glm);
    data.addImport("lanes", lanes);
    root.addImport("grammar_tests_data", data);
    // grammar_tests's AFD tests (afd_serve_lanes.zig, via afd_serve_fail.zig) also need its rendezvous (grammar_tests_tests.zig's `grammar_tests_c_sync`)
    root.addImport("grammar_tests_c_sync", b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/grammar_tests_c_rendezvous.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    step.dependOn(&run.step);
    b.step("test-afd-attn", "qualification_0fa2bee9: the AFD attention engine's G-fail and wire audit over native RoCE exchange (three nodes, CPU)").dependOn(&run.step);
}
