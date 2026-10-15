//! rank_fingerprint: TF_GLM_FINGERPRINT's CPU tests. `register` (build.zig) puts its four test roots in `zig build test` and adds
//! test-rank_fingerprint (the four, the CUDA runtime's tests with fingerprint.zig and proto.zig's, cuda_ranks.zig's own) and
//! test-rank_fingerprint_off (the two recorders alone); -Drankfp-safe builds them ReleaseSafe. `standalone` is rank_fingerprint_build.zig's
//! graph (`focus`, `off`; -Dsafe): the recorders are written against the base's API, so that file, this one and the two
//! recorders run unchanged on the base tree too (the off transcripts are compared base against head).
const std = @import("std");

/// The host tests' runtime without kernel images (zig/build/glm_afd.zig's form; no -Dsource_id: src `unset`).
fn runtime(b: *std.Build, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const host = b.graph.host;
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = host, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    const verbs = host.result.os.tag == .linux and if (std.Io.Dir.accessAbsolute(b.graph.io, "/usr/include/infiniband/verbs.h", .{})) |_| true else |_| false;
    cuda.addCSourceFile(.{ .file = b.path(if (verbs) "zig/src/cuda/roce_proxy.c" else "zig/src/cuda/roce_proxy_stub.c"), .flags = &.{ "-std=gnu11", "-O2", "-idirafter", "/usr/include" } });
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = host, .optimize = optimize }));
    return cuda;
}

const Roots = struct {
    cuda: *std.Build.Module,
    ranks: *std.Build.Module,

    fn init(b: *std.Build, optimize: std.builtin.OptimizeMode) Roots {
        const cuda = runtime(b, optimize);
        return .{ .cuda = cuda, .ranks = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda_ranks.zig"), .target = b.graph.host, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} }) };
    }

    /// A test root over the runtime (and cuda_ranks.zig as glm_ranks), run with its output kept in the step's log.
    fn run(r: Roots, b: *std.Build, path: []const u8, optimize: std.builtin.OptimizeMode) *std.Build.Step {
        const m = b.createModule(.{ .root_source_file = b.path(path), .target = b.graph.host, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = r.cuda }, .{ .name = "glm_ranks", .module = r.ranks } } });
        const x = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        x.has_side_effects = true;
        return &x.step;
    }

    fn runModule(b: *std.Build, m: *std.Build.Module) *std.Build.Step {
        const x = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        x.has_side_effects = true;
        return &x.step;
    }
};

/// The two recorders (base API only): the AFD hello's frames and the pair's bring-up, TF_GLM_FINGERPRINT unset.
const off_roots = [_][]const u8{ "zig/tests/cuda/xchg/recorder.zig", "zig/tests/cuda/rank_fingerprint_pair_off.zig" };
/// The switch on: the AFD hello (three nodes over loopback) and the pair (two ranks over a local link).
const on_roots = [_][]const u8{ "zig/tests/cuda/xchg/identity.zig", "zig/tests/cuda/rank_fingerprint_pair.zig" };

pub fn register(b: *std.Build, test_step: *std.Build.Step) void {
    const opt: std.builtin.OptimizeMode = if (b.option(bool, "rank_fingerprint_safe", "rank_fingerprint's CPU tests in ReleaseSafe") orelse false) .safe else .debug;
    const r = Roots.init(b, opt);
    const all = b.step("test-rank_fingerprint", "rank_fingerprint: TF_GLM_FINGERPRINT off (the recorders = the base) and on (refusals naming the field, cross-role accept, unset-src WARN), the runtime's and cuda_ranks.zig's tests");
    const off = b.step("test-rank_fingerprint_off", "rank_fingerprint: the AFD hello's and the pair's recorders with TF_GLM_FINGERPRINT unset (base API only)");
    for (off_roots) |p| {
        off.dependOn(r.run(b, p, opt));
        all.dependOn(r.run(b, p, opt));
    }
    for (on_roots) |p| all.dependOn(r.run(b, p, opt));
    all.dependOn(Roots.runModule(b, r.cuda));
    all.dependOn(Roots.runModule(b, r.ranks));
    // `zig build test`: the four roots in Debug (the runtime's and cuda_ranks.zig's tests already run there)
    const d = Roots.init(b, .debug);
    for (off_roots ++ on_roots) |p| test_step.dependOn(d.run(b, p, .debug));
}

/// rank_fingerprint_build.zig: `focus` (test-rank_fingerprint's set) and `off` (the recorders), -Dsafe for ReleaseSafe.
pub fn standalone(b: *std.Build) void {
    const opt: std.builtin.OptimizeMode = if (b.option(bool, "safe", "ReleaseSafe") orelse false) .safe else .debug;
    const r = Roots.init(b, opt);
    const off = b.step("off", "the recorders, TF_GLM_FINGERPRINT unset (base API only: runs on the base tree too)");
    for (off_roots) |p| off.dependOn(r.run(b, p, opt));
    const focus = b.step("focus", "rank_fingerprint's tests: the recorders, the switch on, the runtime's and cuda_ranks.zig's tests");
    for (off_roots ++ on_roots) |p| focus.dependOn(r.run(b, p, opt));
    focus.dependOn(Roots.runModule(b, r.cuda));
    focus.dependOn(Roots.runModule(b, r.ranks));
}
