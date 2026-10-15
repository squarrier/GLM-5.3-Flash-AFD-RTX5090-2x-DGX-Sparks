//! GLM-5.3-Flash's Triton half of the build: the wrappers' host tests against the Python wrappers' launches and the
//! CPU-built kernel set's spec (no GPU, no cubins). Kept out of zig/build/cuda.zig, which qualification_qualification_44ba554e-loader, qualification_qualification_44ba554e-kernels and
//! qualification_qualification_44ba554e-transport also edit: build.zig calls `hostTests` after cuda.zig's.

const std = @import("std");

/// The runtime module as zig/build/cuda.zig's `runtime` builds it host-only (no kernel images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = .debug, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = .debug }));
    return cuda;
}

/// zig/glm_triton_host.zig's tests (cuda_triton.zig and tests/cuda/glm/triton_set.zig) join `zig build test`;
/// `zig build test-glm-triton` runs them alone.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/glm_triton_host.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "cuda", .module = runtime(b, host) }},
    }) });
    const run = b.addRunArtifact(tests);
    step.dependOn(&run.step);
    b.step("test-glm-triton", "GLM's Triton wrappers against the Python wrappers' launches and the kernel set (CPU)").dependOn(&run.step);
}
