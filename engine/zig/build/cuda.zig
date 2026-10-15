//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, the runtime, Nemotron, the CLI.

const std = @import("std");
const glm_build = @import("glm_cuda.zig");
const glm_afd_build = @import("glm_afd.zig");
const glm_fwd_tool_build = @import("glm_fwd_tool.zig");
const tp2_build = @import("tp2_cuda.zig");
const xchg_build = @import("xchg_cuda.zig");
const glm_serve_build = @import("glm_serve.zig");

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, arch_specific: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

const kernels = [_]Kernel{
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's layouts and the serial feed
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    // GLM's TP nucleus draw (glm/cuda_nucleus.zig): f64 scaling, fixed-point masses, integer row sums; f64 row maxima
    .{ .name = "torch_nucleus", .src = "torch_ops/nucleus", .flags = torch_ops },
    .{ .name = "torch_argmax_f64", .src = "torch_ops/argmax_f64", .flags = torch_ops },
    // GLM-5.3-Flash (glm5_next): copies of its extensions' device code (zig/tests/cuda/copies.py), the same flags
    .{ .name = "glm_kda", .flags = &.{ "-O3", "--fmad=false" } }, // glm5_next/cuda/kda.py, tensorfold_glm_kda_v2
    // Mia's patch 0001 (MiaAI-Lab, Apache-2.0): the one-shot RoCE all-gather's GPU half (fn_roce.cu from her 0002; cuda/roce.zig), b12x's RoCEnante protocol
    .{ .name = "fn_roce", .flags = &.{"-O3"} },
    // EXL3 (cuda/exl3, glm5_next/cuda/exl3.cu): copies of the extensions' device code (zig/tests/cuda/exl3_copies.py)
    .{ .name = "exl3_experts", .src = "exl3/experts", .flags = &.{ "-O3", "-lineinfo" } }, // tensorfold_exl3_experts_v1
    .{ .name = "exl3_grouped_cb0", .src = "exl3/grouped_cb0", .flags = &.{ "-O3", "-lineinfo" } }, // its experts_cb0-2.cu
    .{ .name = "exl3_grouped_cb1", .src = "exl3/grouped_cb1", .flags = &.{ "-O3", "-lineinfo" } },
    .{ .name = "exl3_grouped_cb2", .src = "exl3/grouped_cb2", .flags = &.{ "-O3", "-lineinfo" } },
    .{ .name = "exl3_linear", .src = "exl3/linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tensorfold_exl3_linear_v3
    .{ .name = "glm_exl3", .flags = &.{"-O3"} }, // glm5_next/cuda/exl3_mm.py, tensorfold_glm_exl3_v1
    // Mia's 0128 (GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold a687724, Apache-2.0) on those copies: exl3/l2d/, the experts flags
    .{ .name = "exl3_l2d_experts", .src = "exl3/l2d/experts", .flags = &.{ "-O3", "-lineinfo" } },
    .{ .name = "exl3_l2d_grouped_cb0", .src = "exl3/l2d/grouped_cb0", .flags = &.{ "-O3", "-lineinfo" } },
    .{ .name = "exl3_l2d_grouped_cb1", .src = "exl3/l2d/grouped_cb1", .flags = &.{ "-O3", "-lineinfo" } },
    .{ .name = "exl3_l2d_grouped_cb2", .src = "exl3/l2d/grouped_cb2", .flags = &.{ "-O3", "-lineinfo" } },
    // X1, python 2.15's prompt switches (IMP2's adaptation of Mia's 0001, 0020 and 0009, Apache-2.0) on those copies: exl3/x1/
    .{ .name = "exl3_x1_experts", .src = "exl3/x1/experts", .flags = &.{ "-O3", "-lineinfo" } },
    .{ .name = "exl3_x1_grouped_cb1", .src = "exl3/x1/grouped_cb1", .flags = &.{ "-O3", "-lineinfo" } },
    // SEQ on X1's kernel (qualification_b3fba1da; Mia's TF_GLM_EXL3_SEQ design, recipe patch 0004, Apache-2.0): exl3/x1s/
    .{ .name = "exl3_x1s_grouped_cb1", .src = "exl3/x1s/grouped_cb1", .flags = &.{ "-O3", "-lineinfo" } },
    // qualification_8724e161 (qualification_7af419c1's qualification_babb0a2d): bertholomus's grouped_mma and work list (bertholomus/TensorFold 90b68e0, Apache-2.0) on X1's helpers: exl3/mma/
    .{ .name = "exl3_mma_cb1", .src = "exl3/mma/grouped_cb1", .flags = &.{ "-O3", "-lineinfo" } },
    // qualification_1b9bf461: glm53f-afd's large-M expert kernels (Hugh Madden, MIT) as python 2.15 ports them (g53_rank.cu, patch 0034),
    // its device code unchanged in g53/ (zig/tests/cuda/g53_copies.py); g53_experts.py's NVCC, glm53f-rank's build.rs flags
    .{ .name = "g53_rank", .src = "g53/rank", .flags = &.{ "-O3", "--fmad=false", "--ftz=false", "--prec-div=true" } },
    // K4 (qualification_e4ad04cd; MiaAI-Lab's EXL3 prompt kernel, recipe patches 0004 / 0009 / 0020, Apache-2.0, as python 2.15 ports it): exl3/k4/
    .{ .name = "exl3_k4", .src = "exl3/k4/experts_prompt", .flags = &.{ "-O3", "-lineinfo" } },
    // qualification_12f4270a: the three-node exchange's GPU half (cuda/xchg), on fn_roce's design with bounded, non-trapping waits; no contraction for the lab's stand-in experts
    .{ .name = "fn_xchg", .flags = &.{ "-O3", "--fmad=false" } },
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// core/stagger.zig, the segment schedule the Metal and CUDA runners share, as its own module (no imports).
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    options.addOption([]const u8, "source_id", @import("source_id.zig").get(b)); // rank_fingerprint: -Dsource_id, else unset
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    // the RoCE all-gather's proxy (Mia's patch 0006): C over libibverbs' header, the library opened at run time; the stub refuses every open without the header or off Linux
    const verbs = target.result.os.tag == .linux and if (std.Io.Dir.accessAbsolute(b.graph.io, "/usr/include/infiniband/verbs.h", .{})) |_| true else |_| false;
    cuda.addCSourceFile(.{ .file = b.path(if (verbs) "zig/src/cuda/roce_proxy.c" else "zig/src/cuda/roce_proxy_stub.c"), .flags = &.{ "-std=gnu11", "-O2", "-idirafter", "/usr/include" } });
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (kernels, images) |k, image| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, tokenizer: *std.Build.Module, glm: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    const glm = glm_build.module(b, target, optimize, cuda, core);
    glm.addImport("lanes", lanes);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .tokenizer = tokenizer, .glm = glm };
}

/// GLM-5.3-Flash's CUDA kernel bindings (zig/src/families/glm/cuda_kernels.zig) over the runtime.
fn glmKernels(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_kernels.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (121; later 120,89)") orelse "121";
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("lanes", mods.lanes);
    b.installArtifact(b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner }));
    // GLM-5.3-Flash's kernel copies against python-0.6's oracle (zig/tests/cuda/glm/box/kda.sh): its own step, not in `install`
    const glm_test = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_kernels.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_kernels", .module = glmKernels(b, target, optimize, cuda) } } });
    b.step("glm-kernels", "tf-glm-kernels: GLM-5.3-Flash's kernel copies on the GPU, byte for byte against python-0.6").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-kernels", .root_module = glm_test }), .{}).step);
    glm_build.tool(b, target, optimize, mods.glm, cuda, mods.core);
    tp2_build.tool(b, target, optimize, cuda);
    xchg_build.tool(b, target, optimize, cuda);
    // the GLM lab tools over this runtime too: every image above with -Dnvcc or -Dfatbins (fn_xchg, fn_roce, EXL3), none without
    glm_fwd_tool_build.tool(b, target, optimize, .{ .cuda = cuda, .core = mods.core, .glm = mods.glm });
    glm_afd_build.tool(b, target, optimize, .{ .cuda = cuda, .glm = mods.glm });
    _ = nativeServer(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.glm, mods.tokenizer, build_options, true);
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, glm: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    // the CUDA GLM's serve open loads a rank with tf-glm-forward's loader (zig/build/glm_serve.zig)
    const glm_tool = glm_serve_build.tool(b, target, optimize, glm, cuda, glm.import_table.get("core").?);
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "nemotron", .module = nemotron }, .{ .name = "glm_cuda", .module = glm }, .{ .name = "glm_tool", .module = glm_tool } },
    });
    const data = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/grammar_tests_data.zig"), .target = target, .optimize = optimize, .link_libc = true });
    data.addImport("glm_cuda", glm);
    data.addImport("lanes", lanes);
    mod.addImport("grammar_tests_data", data);
    return .{ .api = api, .engines = mod };
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, glm: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, lanes, nemotron, glm);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = m.engines }} }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin").dependOn(&install.step);
    return exe;
}

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const native = engines(b, host, .debug, cuda, mods.lanes, mods.nemotron, mods.glm).engines;
    // grammar_tests's TP tests in native/cuda.zig also need its rendezvous (grammar_tests_tests.zig's `grammar_tests_c_sync`; engines() gives grammar_tests_data)
    native.addImport("grammar_tests_c_sync", b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/grammar_tests_c_rendezvous.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} }));
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, native, stagger(b, host, .debug), glmKernels(b, host, .debug, cuda) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    // rank_fingerprint: the native module's TF_GLM_FINGERPRINT tests alone (glm_cuda.zig's switch read, the rank's identity step);
    // native/cuda.zig reaches glm_cuda.zig's tests through its family-list test, so that one is kept
    b.step("test-rank_fingerprint_native", "rank_fingerprint: the CUDA native module's TF_GLM_FINGERPRINT tests alone (one binary)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = native, .filters = &.{ "rank_fingerprint", "every registered family is listed for capabilities" } })).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
    glm_build.hostTests(b, step, mods.glm);
    tp2_build.hostTests(b, step, cuda);
    xchg_build.hostTests(b, step, cuda);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.lanes, mods.nemotron, mods.glm, mods.tokenizer, build_options, false);
}

/// Validate the complete named input set, including images unused by today's server.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const check = b.addSystemCommand(&.{ "sh", "-c", "for file do [ -f \"$file\" ] && [ -s \"$file\" ] || { echo \"missing or empty CUDA fatbin: $file\" >&2; exit 1; }; done", "check-fatbins" });
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}
