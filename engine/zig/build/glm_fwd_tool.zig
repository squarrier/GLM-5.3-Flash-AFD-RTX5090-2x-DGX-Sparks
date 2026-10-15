//! GLM-5.3-Flash's forward tool and the pack pass's host checks (qualification_qualification_44ba554e.5 lane B): `tf-glm-forward` on Linux targets, and
//! zig/glm_pack_host.zig (glm_pack.cu on a CPU emulator against the host reference and python-0.6's fixtures) in
//! `zig build test`. build.zig calls `hostTests`, and zig/build/cuda.zig calls `tool` over its runtime (as xchg's tool);
//! glm_pack's fatbin follows the same -Dnvcc, -Dsm and -Dfatbins options as the images the run verbs embed (KDA, the
//! torch ops), read here without declaring them twice.

const std = @import("std");
const glm_build = @import("glm_cuda.zig");

/// torch.utils.cpp_extension's nvcc flags, as zig/build/cuda.zig passes them to every fatbin.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// A -D option zig/build/cuda.zig declares (nvcc, fatbins, sm), as the user gave it.
fn given(b: *std.Build, name: []const u8) ?[]const u8 {
    const v = b.user_input_options.get(name) orelse return null;
    return switch (v) {
        .scalar => |s| s,
        else => null,
    };
}

/// The host tests' runtime module, without kernel images, as zig/build/glm_triton.zig builds it.
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

/// The runtime, the core and the GLM family over them: the host tests' (`family`), or zig/build/cuda.zig's for the tool.
pub const Mods = struct { cuda: *std.Build.Module, core: *std.Build.Module, glm: *std.Build.Module };

/// The GLM family module over its own runtime and core (zig/build/glm_cuda.zig's `module`).
fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Mods {
    const cuda = runtime(b, target, optimize);
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    return .{ .cuda = cuda, .core = core, .glm = glm_build.module(b, target, optimize, cuda, core) };
}

/// glm_pack.cu and python-0.6's experts_pack.cu on the CPU (zig/tests/cuda/glm_pack_emu.cpp), linked into `m`.
fn emulator(b: *std.Build, m: *std.Build.Module) void {
    m.addCSourceFile(.{ .file = b.path("zig/tests/cuda/glm_pack_emu.cpp"), .flags = &.{ "-std=c++20", "-O2", "-fno-exceptions" } });
}

/// zig/glm_pack_host.zig's and zig/glm_fwd_host.zig's tests join `zig build test`; `zig build test-glm-pack` runs them alone.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const mods = family(b, host, .debug);
    const alone = b.step("test-glm-pack", "GLM's pack pass and the tool's EXL3 launches: glm_pack.cu emulated, qualification_765330ce's fixtures (CPU)");
    for ([_][]const u8{ "zig/glm_pack_host.zig", "zig/glm_fwd_host.zig" }) |path| {
        const root = b.createModule(.{
            .root_source_file = b.path(path),
            .target = host,
            .optimize = .debug,
            .link_libc = true,
            .link_libcpp = true,
            .imports = &.{ .{ .name = "glm_cuda", .module = mods.glm }, .{ .name = "cuda", .module = mods.cuda } },
        });
        emulator(b, root);
        const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
        step.dependOn(&run.step);
        alone.dependOn(&run.step);
    }
}

/// An image tf-glm-forward embeds: its name, its source under zig/kernels/cuda, zig/build/cuda.zig's flags for it, and
/// whose it is: the tool's own, lane A's matmuls or DSA select, qualification_765330ce's EXL3 (those three are in qualification_qualification_44ba554e-int, not qualification_qualification_44ba554e-fwd-b),
/// or the nucleus draw's (qualification_45c10bcf; optional: a set without them still runs greedy).
const Image = struct { name: []const u8, src: []const u8, flags: []const []const u8, part: Part = .tool };
const Part = enum { tool, mm, dsa, exl3, nucleus };

/// The torch-op replacements' flags, as zig/build/cuda.zig's (no contraction, no flush to zero).
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

/// glm_pack (check-weights), then what the run verbs launch beyond the GLM module: KDA and the sampler's torch ops.
const images = [_]Image{
    .{ .name = "glm_pack", .src = "glm_pack", .flags = &.{"-O3"} },
    .{ .name = "glm_kda", .src = "glm_kda", .flags = &.{ "-O3", "--fmad=false" } },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    // the TP nucleus draw's (glm/cuda_nucleus.zig): scale, mass and integer row sums; f64 row maxima
    .{ .name = "torch_nucleus", .src = "torch_ops/nucleus", .flags = torch_ops, .part = .nucleus },
    .{ .name = "torch_argmax_f64", .src = "torch_ops/argmax_f64", .flags = torch_ops, .part = .nucleus },
    // lane A's affine kernels and DSA select, as zig/build/glm_fwd_kernels.zig builds them (torch's flags, -O3)
    .{ .name = "glm_qmm_group", .src = "glm_qmm_group", .flags = &.{"-O3"}, .part = .mm },
    .{ .name = "glm_qmm_prefill", .src = "glm_qmm_prefill", .flags = &.{"-O3"}, .part = .mm },
    .{ .name = "glm_experts", .src = "glm_experts", .flags = &.{"-O3"}, .part = .mm },
    .{ .name = "glm_experts_prefill", .src = "glm_experts_prefill", .flags = &.{"-O3"}, .part = .mm },
    .{ .name = "glm_select", .src = "glm_select", .flags = &.{"-O3"}, .part = .dsa },
    // qualification_765330ce's EXL3 images, as zig/build/cuda.zig's kernel list builds them (the extensions' extra_cuda_cflags)
    .{ .name = "exl3_experts", .src = "exl3/experts", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    .{ .name = "exl3_grouped_cb0", .src = "exl3/grouped_cb0", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    .{ .name = "exl3_grouped_cb1", .src = "exl3/grouped_cb1", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    .{ .name = "exl3_grouped_cb2", .src = "exl3/grouped_cb2", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    .{ .name = "exl3_linear", .src = "exl3/linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" }, .part = .exl3 },
    .{ .name = "glm_exl3", .src = "glm_exl3", .flags = &.{"-O3"}, .part = .exl3 },
    // Mia's 0128 on qualification_765330ce's expert copies (exl3/l2d/, zig/build/cuda.zig's entries): run only with TF_GLM_EXL3_L2D/_PDL
    .{ .name = "exl3_l2d_experts", .src = "exl3/l2d/experts", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    .{ .name = "exl3_l2d_grouped_cb0", .src = "exl3/l2d/grouped_cb0", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    .{ .name = "exl3_l2d_grouped_cb1", .src = "exl3/l2d/grouped_cb1", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    .{ .name = "exl3_l2d_grouped_cb2", .src = "exl3/l2d/grouped_cb2", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    // X1 on the same copies (exl3/x1/, zig/build/cuda.zig's entries): run only with TF_GLM_EXL3_X1=1
    .{ .name = "exl3_x1_experts", .src = "exl3/x1/experts", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
    .{ .name = "exl3_x1_grouped_cb1", .src = "exl3/x1/grouped_cb1", .flags = &.{ "-O3", "-lineinfo" }, .part = .exl3 },
};

/// Whether this tree has an image's source (lane A's and qualification_765330ce's are in qualification_qualification_44ba554e-int, not in qualification_qualification_44ba554e-fwd-b).
fn present(b: *std.Build, im: Image) bool {
    const sub = b.fmt("zig/kernels/cuda/{s}.cu", .{im.src});
    if (b.root.root_dir.handle.access(b.graph.io, b.pathJoin(&.{ b.root.sub_path, sub }), .{})) |_| return true else |_| return false;
}

/// Whether every image of `part` is embedded.
fn all(part: Part, paths: []const ?std.Build.LazyPath) bool {
    for (images, paths) |im, p| if (im.part == part and p == null) return false;
    return true;
}

/// nvcc -fatbin of an image with torch's flags, its own, and one -gencode per SASS target, as zig/build/cuda.zig's.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, im: Image, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(im.flags);
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s},code=sm_{s}", .{ sm, sm }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{im.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{im.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{im.src})));
    return out;
}

/// `tf-glm-forward` (zig/tests/cuda/glm_forward.zig) over `mods` (zig/build/cuda.zig's runtime, with its images under the
/// same options): with -Dnvcc or -Dfatbins it embeds the images above (check-weights packs on the GPU; first, teacher and
/// dump run); without, only its CPU modes run (--dry, --host).
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, mods: Mods) void {
    const nvcc = given(b, "nvcc");
    const version: ?std.Build.LazyPath = if (nvcc) |n| blk: {
        const v = b.addSystemCommand(&.{ n, "--version" });
        v.has_side_effects = true;
        break :blk v.captureStdOut(.{});
    } else null;
    var paths: [images.len]?std.Build.LazyPath = @splat(null);
    for (images, &paths) |im, *p| {
        if (im.part != .tool and !present(b, im)) continue;
        if (given(b, "fatbins")) |dir| {
            const f = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{im.name}) });
            if (std.Io.Dir.accessAbsolute(b.graph.io, f, .{})) |_| {
                p.* = .{ .cwd_relative = f };
            } else |_| {} // a set without ours: nvcc or none
        }
        if (p.* == null) if (nvcc) |n| {
            p.* = fatbin(b, n, version.?, im, given(b, "sm") orelse "121");
        };
    }
    var run_images = true;
    for (images[1..], paths[1..]) |im, p| run_images = run_images and (im.part != .tool or p != null);
    const options = b.addOptions();
    options.addOption(bool, "glm_pack", paths[0] != null);
    options.addOption(bool, "run_images", run_images);
    options.addOption(bool, "mm_images", all(.mm, &paths));
    options.addOption(bool, "dsa_image", all(.dsa, &paths));
    options.addOption(bool, "exl3_images", all(.exl3, &paths));
    options.addOption(bool, "nucleus_images", all(.nucleus, &paths));
    const root = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_forward.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{ .{ .name = "glm_cuda", .module = mods.glm }, .{ .name = "cuda", .module = mods.cuda }, .{ .name = "core", .module = mods.core } },
    });
    root.addOptions("glm_forward_options", options);
    for (images, paths) |im, p| if (p) |path| root.addAnonymousImport(b.fmt("{s}_fatbin", .{im.name}), .{ .root_source_file = path });
    emulator(b, root);
    const exe = b.addExecutable(.{ .name = "tf-glm-forward", .root_module = root });
    b.installArtifact(exe);
    const step = b.step("tf-glm-forward", "GLM-5.3-Flash's forward tool: check-weights, and first, teacher and dump on two ranks");
    step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    for (images, paths) |im, p| if (p) |path| step.dependOn(&b.addInstallFile(path, b.fmt("fatbin/{s}.fatbin", .{im.name})).step);
}
