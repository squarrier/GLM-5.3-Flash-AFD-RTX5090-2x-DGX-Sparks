//! GLM-5.3-Flash's CUDA half of the build: its module, host tests (one against the Metal loader) and `tf-glm-load`.

const std = @import("std");

/// The family on CUDA (zig/src/families/glm/cuda.zig) over the runtime and the backend-neutral core.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    const glm = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm.addImport("cuda", cuda);
    glm.addImport("core", core);
    glm.addImport("lanes", b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return glm;
}

/// The module's tests, and weights.zig's loader over host memory (zig/tests/cuda/fake_metal.zig) against our split.
pub fn hostTests(b: *std.Build, step: *std.Build.Step, glm: *std.Build.Module) void {
    const host = b.graph.host;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);
    const metal = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/fake_metal.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const check = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/glm_cuda_host.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "metal", .module = metal }},
    }) });
    const run = b.addRunArtifact(check);
    step.dependOn(&run.step);
    // TF_GLM_CHECKPOINT=DIR adds the Metal loader's dry plan of a real checkpoint (headers only; a sparse copy serves)
    const real = b.addRunArtifact(check);
    real.has_side_effects = true; // the variable is not an input the cache sees
    b.step("test-glm-split", "GLM's CUDA split against the Metal loader (TF_GLM_CHECKPOINT: a real checkpoint's headers too)").dependOn(&real.step);
    // tf-exl3-check built for the host without kernels: `zig build test` compiles the tool's GPU half too
    step.dependOn(&exl3Check(b, host, .debug, glm, glm.import_table.get("cuda").?, glm.import_table.get("core").?).step);
    // qualification_b3fba1da: and runs its tests (the prompt windows' bench parts, glm_exl3_prompt.zig)
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = exl3Check(b, host, .debug, glm, glm.import_table.get("cuda").?, glm.import_table.get("core").?).root_module })).step);
    // qualification_1b9bf461: tf-g53-check likewise
    step.dependOn(&g53Check(b, host, .debug, glm, glm.import_table.get("cuda").?, glm.import_table.get("core").?).step);
}

/// `tf-glm-load`: a rank's plan from a checkpoint's headers, its reads timed, or its weights loaded onto the GPU; and
/// `tf-exl3-check`: one MoE layer's EXL3 kernels on Mia's checkpoint against exl3_check.py's references.
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, glm: *std.Build.Module, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const exe = b.addExecutable(.{ .name = "tf-glm-load", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_load.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
    b.installArtifact(exe);
    b.step("tf-glm-load", "GLM-5.3-Flash's TP rank: plan, timed reads, or a load onto the GPU").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const check = exl3Check(b, target, optimize, glm, cuda, core);
    b.installArtifact(check);
    b.step("tf-exl3-check", "One MoE layer's EXL3 kernels on Mia's checkpoint, byte for byte against python-0.6").dependOn(&b.addInstallArtifact(check, .{}).step);
    const g53c = g53Check(b, target, optimize, glm, cuda, core);
    b.installArtifact(g53c);
    b.step("tf-g53-check", "qualification_1b9bf461: g53 + KT3R on one MoE layer's half, byte for byte against python 2.15's g53_check.py").dependOn(&b.addInstallArtifact(g53c, .{}).step);
}

/// `tf-g53-check` (qualification_1b9bf461): the halves' g53_rank port against python 2.15's own kernel (tools/zig/g53_check.py).
fn g53Check(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, glm: *std.Build.Module, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "tf-g53-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_g53_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
}

fn exl3Check(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, glm: *std.Build.Module, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "tf-exl3-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_exl3_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    }) });
}
