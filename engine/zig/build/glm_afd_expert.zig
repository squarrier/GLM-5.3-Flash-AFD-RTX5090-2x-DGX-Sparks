//! qualification_cec3c04c's expert role in `zig build test`: zig/tests/cuda/afd_expert_loop.zig (G-loop: python 2.15's per-forward loop
//! over native RoCE exchange, three nodes in one process on fn_xchg's CPU build) and the `expert` command's body
//! (zig/tests/cuda/afd_expert.zig, which qualification_955b7191's tf-glm-afd calls) compiled for the host. Kept out of the shared build
//! files: build.zig calls `hostTests` after them; the runtime and family modules are built as glm_fwd_tool.zig builds
//! them (no kernel images: the CPU tests launch nothing).

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

/// G-loop and the command's flags and generator: `zig build test`, or `zig build test-afd-expert` alone.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug);
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = host, .optimize = .debug, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const glm = glm_build.module(b, host, .debug, cuda, core);
    const root = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/afd_expert_loop.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{ .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda } },
    });
    root.addCSourceFile(.{ .file = b.path("zig/tests/cuda/xchg/xchg_emu.cpp"), .flags = &.{ "-std=c++20", "-O2", "-ffp-contract=off" } });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    step.dependOn(&run.step);
    b.step("test-afd-expert", "qualification_cec3c04c: the AFD expert role's loop over native RoCE exchange (three nodes, CPU) and its command").dependOn(&run.step);
}
