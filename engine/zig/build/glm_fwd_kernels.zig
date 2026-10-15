//! GLM-5.3-Flash's forward kernels (qualification_qualification_44ba554e.5 lane A): GLM's instances of the shared affine kernels as their own fatbins
//! (copies by zig/tests/cuda/copies.py, built with python-0.6's extension flags) and tf-glm-mm, which checks them on one
//! GPU byte for byte against python-0.6. Kept out of zig/build/cuda.zig, the merges' hotspot: build.zig calls `build`.

const std = @import("std");
const glm_build = @import("glm_cuda.zig");

/// The copies in zig/kernels/cuda (kernels/qmm.py's tensorfold_qmm_v5, cuda/experts.py's tensorfold_experts_v7), then
/// GLM's own glm_select (qualification_qualification_973313f6: select_tokens' torch ops after its kernels, one launch; not a copy, no SASS to equal).
pub const names = [_][]const u8{ "glm_qmm_group", "glm_qmm_prefill", "glm_experts", "glm_experts_prefill", "glm_select" };

/// torch.utils.cpp_extension's nvcc flags (torch 2.13, as zig/build/cuda.zig passes them), then both extensions' -O3.
const flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
    "-O3",
};

/// -Dnvcc, -Dsm and -Dfatbins as zig/build/cuda.zig declares them on Linux; an option is declared once, so read here.
fn given(b: *std.Build, name: []const u8) ?[]const u8 {
    return switch (b.user_input_options.get(name) orelse return null) {
        .scalar => |s| s,
        else => null,
    };
}

/// nvcc -fatbin with torch's flags, -O3 and one -gencode per SASS target, as python-0.6's build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, name: []const u8, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&flags);
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s},code=sm_{s}", .{ sm, sm }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{name})));
    return out;
}

/// The four images: built by -Dnvcc for -Dsm (default 121), read from -Dfatbins, or none (a host build).
fn images(b: *std.Build) ?[names.len]std.Build.LazyPath {
    var out: [names.len]std.Build.LazyPath = undefined;
    if (given(b, "fatbins")) |dir| {
        for (names, &out) |n, *o| o.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{n}) }));
        return out;
    }
    const nvcc = given(b, "nvcc") orelse return null;
    const run = b.addSystemCommand(&.{ nvcc, "--version" });
    run.has_side_effects = true; // the compiler's version text is an input of every fatbin
    const version = run.captureStdOut(.{});
    for (names, &out) |n, *o| o.* = fatbin(b, nvcc, version, n, given(b, "sm") orelse "121");
    return out;
}

/// `glm_mm_images`: each image as `pub const <name>: []const u8`, 16-byte aligned; empty slices in a host build.
fn imagesModule(b: *std.Build, imgs: ?[names.len]std.Build.LazyPath) *std.Build.Module {
    const wf = b.addWriteFiles();
    var src: []const u8 = "//! GLM's affine fatbins (zig/build/glm_fwd_kernels.zig), embedded.\n";
    for (names, 0..) |n, i| {
        if (imgs) |im| {
            _ = wf.addCopyFile(im[i], b.fmt("{s}.fatbin", .{n}));
            src = b.fmt("{s}const {s}_bytes align(16) = @embedFile(\"{s}.fatbin\").*;\npub const {s}: []const u8 = &{s}_bytes;\n", .{ src, n, n, n, n });
        } else src = b.fmt("{s}pub const {s}: []const u8 = &.{{}};\n", .{ src, n });
    }
    return b.createModule(.{ .root_source_file = wf.add("glm_mm_images.zig", src) });
}

/// The CUDA runtime with no images of zig/build/cuda.zig's list (only GLM's four are needed here).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// tf-glm-mm's root module over the bindings (zig/src/families/glm/cuda_kernels_mm.zig) and the images.
fn toolModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, imgs: *std.Build.Module) *std.Build.Module {
    const cuda = runtime(b, target, optimize);
    const mm = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_kernels_mm.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    return b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_mm.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_mm", .module = mm }, .{ .name = "glm_mm_images", .module = imgs } },
    });
}

/// `zig build glm-mm-fatbins` (the four images into fatbin/) and `glm-mm` (tf-glm-mm) on Linux, outside `install`;
/// `zig build test` compiles tf-glm-mm for the host and runs its tests (the bindings' run in the glm module's), and
/// glm_select.cu on the CPU emulator against python's select_tokens tail (zig/glm_select_host.zig).
pub fn build(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = toolModule(b, host, .debug, imagesModule(b, null)) })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = layerModule(b, host, .debug, imagesModule(b, null)) })).step);
    const sel = b.createModule(.{ .root_source_file = b.path("zig/glm_select_host.zig"), .target = host, .optimize = .debug, .link_libc = true, .link_libcpp = true });
    sel.addCSourceFile(.{ .file = b.path("zig/tests/cuda/glm_select_emu.cpp"), .flags = &.{ "-std=c++20", "-O2", "-fno-exceptions" } });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = sel })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = coverModule(b, host) })).step);
    if (target.result.os.tag != .linux) return;
    const imgs = images(b);
    const fat = b.step("glm-mm-fatbins", "GLM-5.3-Flash's affine kernel copies as fatbins (-Dnvcc, -Dsm), into fatbin/");
    if (imgs) |im| for (names, im) |n, file| fat.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{n})).step);
    const exe = b.addExecutable(.{ .name = "tf-glm-mm", .root_module = toolModule(b, target, optimize, imagesModule(b, imgs)) });
    b.step("glm-mm", "tf-glm-mm: GLM-5.3-Flash's affine kernel copies on the GPU, byte for byte against python-0.6").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const layer = b.addExecutable(.{ .name = "tf-glm-layer", .root_module = layerModule(b, target, optimize, imagesModule(b, imgs)) });
    b.step("glm-layer", "tf-glm-layer: G-layer-1 on the MoE layers' MLP halves, one rank, against qualification_6607f886's oracle (qualification_qualification_736508ad)").dependOn(&b.addInstallArtifact(layer, .{}).step);
}

/// tf-glm-layer's root module (zig/tests/cuda/glm_layer.zig): the GLM family over a runtime with the RoCE stub (no
/// comm: one rank), the affine images embedded as tf-glm-mm's.
fn layerModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, imgs: *std.Build.Module) *std.Build.Module {
    const fam = family(b, target, optimize);
    return b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_layer.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = fam.cuda }, .{ .name = "glm_cuda", .module = fam.glm }, .{ .name = "glm_mm_images", .module = imgs } },
    });
}

/// The GLM family over a runtime with the RoCE stub (no comm: one rank).
fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) struct { cuda: *std.Build.Module, glm: *std.Build.Module } {
    const cuda = runtime(b, target, optimize);
    cuda.addCSourceFile(.{ .file = b.path("zig/src/cuda/roce_proxy_stub.c"), .flags = &.{ "-std=gnu11", "-O2", "-idirafter", "/usr/include" } });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    return .{ .cuda = cuda, .glm = glm_build.module(b, target, optimize, cuda, core) };
}

/// qualification_qualification_ea348a94's G-cover (zig/tests/cuda/glm_dsa_cover.zig): the DSA block's launches on the long prompt against kernels.json.
fn coverModule(b: *std.Build, host: std.Build.ResolvedTarget) *std.Build.Module {
    const fam = family(b, host, .debug);
    const m = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_dsa_cover.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = fam.cuda }, .{ .name = "glm_cuda", .module = fam.glm } },
    });
    m.addAnonymousImport("dsa_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_dsa.json") });
    m.addAnonymousImport("dsa_fixture_exl3", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_dsa_exl3.json") });
    return m;
}
