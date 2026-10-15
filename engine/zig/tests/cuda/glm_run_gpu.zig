//! tf-glm-forward's run verbs on a GPU (glm_run.zig): the rank's bring-up in python GlmEngine's order over
//! glm/cuda_ranks.zig (the link, NCCL and RoCE, a barrier; the settings and the run's request agreed; the weights
//! packed as check-weights does; ready and a barrier; the Engine on its device memory), the kernels the forward launches
//! (GLM's Triton cubins from --triton, the KDA and torch-op images this tool embeds), then the verb, armed on the
//! watchdog: a lost peer or a call past TF_TP_TIMEOUT_S exits 3.
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");
const options = @import("glm_forward_options");
const tool = @import("glm_forward.zig");
const verbs = @import("glm_run.zig").Run;
const widen = @import("glm_run.zig").widen;

const Args = verbs.ArgsT;
const Tokens = verbs.TokensT;

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// The images the forward needs beyond the runtime's (a host-only build of the tool embeds none of them).
const img = struct {
    const kda: []const u8 = if (options.run_images) &Blob("glm_kda_fatbin").bytes else &.{};
    const pointwise: []const u8 = if (options.run_images) &Blob("torch_pointwise_fatbin").bytes else &.{};
    const topk: []const u8 = if (options.run_images) &Blob("torch_topk_fatbin").bytes else &.{};
    const indexing: []const u8 = if (options.run_images) &Blob("torch_indexing_fatbin").bytes else &.{};
    const movement: []const u8 = if (options.run_images) &Blob("torch_movement_fatbin").bytes else &.{};
    const nucleus: []const u8 = if (options.nucleus_images) &Blob("torch_nucleus_fatbin").bytes else &.{};
    const argmax_f64: []const u8 = if (options.nucleus_images) &Blob("torch_argmax_f64_fatbin").bytes else &.{};
    const pack: []const u8 = if (options.glm_pack) &Blob("glm_pack_fatbin").bytes else &.{};
    const qmm_group: []const u8 = if (options.mm_images) &Blob("glm_qmm_group_fatbin").bytes else &.{};
    const qmm_prefill: []const u8 = if (options.mm_images) &Blob("glm_qmm_prefill_fatbin").bytes else &.{};
    const experts: []const u8 = if (options.mm_images) &Blob("glm_experts_fatbin").bytes else &.{};
    const experts_prefill: []const u8 = if (options.mm_images) &Blob("glm_experts_prefill_fatbin").bytes else &.{};
    const select: []const u8 = if (options.dsa_image) &Blob("glm_select_fatbin").bytes else &.{};
    const exl3_experts: []const u8 = if (options.exl3_images) &Blob("exl3_experts_fatbin").bytes else &.{};
    const exl3_grouped_cb0: []const u8 = if (options.exl3_images) &Blob("exl3_grouped_cb0_fatbin").bytes else &.{};
    const exl3_grouped_cb1: []const u8 = if (options.exl3_images) &Blob("exl3_grouped_cb1_fatbin").bytes else &.{};
    const exl3_grouped_cb2: []const u8 = if (options.exl3_images) &Blob("exl3_grouped_cb2_fatbin").bytes else &.{};
    const exl3_linear: []const u8 = if (options.exl3_images) &Blob("exl3_linear_fatbin").bytes else &.{};
    const glm_exl3: []const u8 = if (options.exl3_images) &Blob("glm_exl3_fatbin").bytes else &.{};
    const exl3_l2d_experts: []const u8 = if (options.exl3_images) &Blob("exl3_l2d_experts_fatbin").bytes else &.{};
    const exl3_l2d_grouped_cb0: []const u8 = if (options.exl3_images) &Blob("exl3_l2d_grouped_cb0_fatbin").bytes else &.{};
    const exl3_l2d_grouped_cb1: []const u8 = if (options.exl3_images) &Blob("exl3_l2d_grouped_cb1_fatbin").bytes else &.{};
    const exl3_l2d_grouped_cb2: []const u8 = if (options.exl3_images) &Blob("exl3_l2d_grouped_cb2_fatbin").bytes else &.{};
    const exl3_x1_experts: []const u8 = if (options.exl3_images) &Blob("exl3_x1_experts_fatbin").bytes else &.{};
    const exl3_x1_grouped_cb1: []const u8 = if (options.exl3_images) &Blob("exl3_x1_grouped_cb1_fatbin").bytes else &.{};
};

const Kda = glm.kda_kernels.Kda;

/// kda_kernels.Kda from this tool's glm_kda image (the runtime's is empty in the tool's build).
fn kdaLoad(d: *const cuda.Driver) !Kda {
    if (img.kda.len == 0) return error.NoKdaImage;
    var m = try cuda.Module.load(d, img.kda);
    errdefer m.unload();
    const s = glm.kda_kernels.sym;
    return .{ .module = m, .chain_fn = try m.function(s.chain), .replay_fn = try m.function(s.replay), .replay_layers_fn = try m.function(s.replay_layers), .prep_fn = try m.function(s.prep), .step_fn = try m.function(s.step), .out_fn = try m.function(s.out) };
}

/// Lane A's affine matmuls and experts (cuda_kernels_mm.Mm) where the tree has them (qualification_qualification_44ba554e-int): fwd.Kernels.mm's type.
const has_mm = @hasField(glm.fwd.Kernels, "mm");
const Mm = if (has_mm) @typeInfo(@FieldType(glm.fwd.Kernels, "mm")).pointer.child else void;

/// Mm from this tool's four images of lane A's (none in qualification_qualification_44ba554e-fwd-b, whose matmuls are still error.KernelNotPorted).
fn mmLoad(ctx: *const cuda.Context) !Mm {
    if (comptime has_mm) {
        return Mm.load(ctx, .{ .qmm_group = img.qmm_group, .qmm_prefill = img.qmm_prefill, .experts = img.experts, .experts_prefill = img.experts_prefill });
    } else return {};
}

/// qualification_765330ce's six EXL3 images (qualification_qualification_44ba554e-int) as its kernels.Set, from this tool's copies (the tool's runtime embeds none); with
/// 0128's switches on, its four exl3_l2d images too.
const has_exl3 = @hasField(glm.fwd.Kernels, "exl3");
const X3 = if (has_exl3) glm.exl3.kernels.Set else void;

fn exl3Load(d: *const cuda.Driver, sd: Sched) !X3 {
    if (comptime has_exl3) {
        if (img.glm_exl3.len == 0) return error.NoExl3Images;
        var s: X3 = .{};
        errdefer s.unload();
        inline for (glm.exl3.kernels.images_all) |im| s.modules[@backingInt(im)] = try cuda.Module.load(d, @field(img, @tagName(im)));
        if (sd.on()) inline for (glm.exl3.kernels.images_l2d) |im| {
            s.modules[@backingInt(im)] = try cuda.Module.load(d, @field(img, @tagName(im)));
        };
        return s;
    } else return error.NoExl3Kernels;
}

/// Mia's 0128 in the EXL3 experts (TF_GLM_EXL3_L2D, TF_GLM_EXL3_PDL), read once; a bad value refuses the run, and
/// the settings are said when a switch is on (unset: no line, today's kernels).
const Sched = glm.exl3.experts.Sched;

pub fn exl3Sched(rank: u32) !Sched {
    const s = Sched.fromEnv() catch |e| {
        std.debug.print("FAIL rank {d}: TF_GLM_EXL3_L2D takes 0..64 and TF_GLM_EXL3_PDL 0 or 1 ({t})\n", .{ rank, e });
        return e;
    };
    if (s.on()) std.debug.print("INFO rank {d}: EXL3 experts: Mia's 0128 kernels (exl3_l2d), L2 prefetch {d} steps ahead, programmatic launch {s}\n", .{ rank, s.l2d, if (s.pdl) "on" else "off" });
    return s;
}

/// Lane A's DSA select (glm_select, qualification_qualification_44ba554e-int): fwd.Kernels.dsa's type, from this tool's image.
const has_dsa = @hasField(glm.fwd.Kernels, "dsa");
const Dsa = if (has_dsa) @typeInfo(@typeInfo(@FieldType(glm.fwd.Kernels, "dsa")).optional.child).pointer.child else void;

fn dsaLoad(d: *const cuda.Driver) !?Dsa {
    if (comptime has_dsa) {
        if (img.select.len == 0) return null;
        return try Dsa.load(d, img.select);
    } else return null;
}

/// The extension kernels: KDA's, lane A's Mm and DSA select, and qualification_765330ce's EXL3 set with its shared-memory opt-in, each
/// where the tree has it (an MLX rank loads no EXL3 image; a missing select stops at the first sparse row, said).
fn kernels(kda: *const Kda, mm: *const Mm, x3: ?*const X3, optin: u32, dsa: ?*const Dsa, sd: Sched) glm.fwd.Kernels {
    var k: glm.fwd.Kernels = if (comptime has_mm) .{ .kda = kda, .mm = mm } else .{ .kda = kda };
    if (comptime has_exl3) {
        k.exl3 = x3;
        k.exl3_optin = optin;
        k.exl3_sched = sd;
    }
    if (comptime has_dsa) k.dsa = dsa;
    return k;
}

/// Stand-ins for what run() loads, so a --dry run builds run()'s set without a driver; nothing dereferences them.
const stand = struct {
    var kda: Kda = undefined;
    var mm: Mm = undefined;
    var x3: X3 = undefined;
    var dsa: Dsa = undefined;
};

/// Whether this build embeds the run verbs' images (a host build embeds none: its GPU verbs refuse, its --dry records).
pub const images = options.run_images;

/// The fwd.Kernels run() passes a rank, built from this build's images at stand-ins (--dry): run()'s refusal where it
/// would refuse (no KDA, lane A's or EXL3 images), and a null .dsa where glm_select's image is missing.
pub fn drySet(exl3: bool, sd: Sched) !glm.fwd.Kernels {
    if (img.kda.len == 0) return error.NoKdaImage;
    if (comptime has_mm) if (!options.mm_images) return error.NotFound; // Mm.load's
    const x3: ?*const X3 = if (!exl3) null else if (comptime !has_exl3) return error.NoExl3Kernels else if (img.glm_exl3.len == 0) return error.NoExl3Images else &stand.x3;
    if (exl3 and sd.on() and img.exl3_l2d_experts.len == 0) return error.NoExl3Images; // exl3Load's, with 0128 on
    return kernels(&stand.kda, &stand.mm, x3, 0, if (img.select.len == 0) null else &stand.dsa, sd);
}

/// A set without .dsa (a tree with lane A's DSA block): its decode stops with NoDsaKernels at the first sparse row.
pub fn lacksDsa(k: glm.fwd.Kernels) bool {
    if (comptime has_dsa) return k.dsa == null;
    return false; // no DSA block in this tree: its DSA layers stop with NotPorted, which --dry says
}

/// The set's fields that are there, for the log.
pub fn describe(k: glm.fwd.Kernels, buf: []u8) ![]const u8 {
    const x3 = if (comptime has_exl3) k.exl3 != null else false;
    const dsa = if (comptime has_dsa) k.dsa != null else false;
    return std.fmt.bufPrint(buf, "kda{s}{s}{s}", .{ if (has_mm) ", mm" else "", if (x3) ", exl3" else "", if (dsa) ", dsa" else "" });
}

/// The bring-up's steps (glm.ranks.bringUp calls them in python's order) and what they leave.
const Up = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    a: Args,
    t: Tokens,
    d: *const cuda.Driver,
    pair: glm.ranks.Pair,
    ix: ?glm.index.Index = null,
    r: ?tool.Rank = null,
    run: ?tool.Run = null,
    pk: ?glm.pack.Kernels = null,
    stream: ?cuda.Stream = null,
    w: ?glm.views.Weights = null,
    e: ?*glm.engine.Engine = null,
    dev: ?glm.engine.Device = null,
    /// the load's time: the reads, the pack and the binding (qualification_e84ee1bf's TIME line)
    load_ns: u64 = 0,

    pub fn comm(u: *Up) !void {
        const c = std.mem.lastIndexOfScalar(u8, u.a.master, ':') orelse return error.Master;
        const at = try std.Io.net.IpAddress.parse(u.a.master[0..c], try std.fmt.parseInt(u16, u.a.master[c + 1 ..], 10));
        try u.pair.openLink(at, 600_000);
        try u.pair.openComm();
        try u.pair.startWatch();
    }

    pub fn barrier(u: *Up, label: []const u8) !void {
        try u.pair.barrier(label, 0);
    }

    pub fn settings(u: *Up) !void {
        var buf: [256]u8 = undefined;
        const req = try verbs.requestFn(u.a, u.t, &buf);
        try u.pair.agree(.{ .capacity = u.a.capacity, .long_context = u.a.long_context, .prefill_rows = u.a.prefill_rows, .mtp = u.a.mtp, .run = req });
    }

    /// weights.load for the rank: check-weights' read and pack onto the GPU, the packed buffers kept and bound.
    pub fn load(u: *Up) !void {
        const t0 = nowNs();
        defer u.load_ns = nowNs() - t0;
        u.ix = try glm.index.open(u.gpa, u.a.model);
        const ta: tool.Args = .{ .dir = u.a.model, .rank = u.a.rank, .ranks = u.a.ranks, .layers = u.a.layers, .keep = true, .mtp_layer = u.a.mtp_layer };
        u.r = try tool.plans(u.gpa, ta, &u.ix.?);
        const r = &u.r.?;
        if (!u.a.mtp) r.m.mtp_layers = 0;
        const image = if (u.a.fatbin) |p| try tool.slurp(u.gpa, p, 64 << 20) else img.pack;
        if (image.len == 0) return error.NoPackImage;
        u.pk = try glm.pack.Kernels.load(u.d, image);
        u.stream = try cuda.Stream.init(u.d, true);
        const rows = try u.gpa.alloc(tool.Row, r.pk.items.items.len);
        @memset(rows, .{});
        u.run = .{ .gpa = u.gpa, .a = ta, .ix = &u.ix.?, .r = r, .rows = rows, .stored = try u.gpa.alloc(?tool.Run.Mem, r.split.groups().len), .driver = u.d, .stream = u.stream, .kernels = &u.pk.? };
        @memset(u.run.?.stored, null);
        try u.run.?.all();
        u.w = try glm.views.rank(u.gpa, &r.m);
        const bound = try glm.pack.bind(&u.w.?, &r.pk);
        std.debug.print("INFO rank {d}: {d} views bound to the packed rank\n", .{ u.a.rank, bound });
    }

    pub fn ready(u: *Up, label: []const u8) !void {
        try u.pair.barrier(label, glm.ranks.ready_s);
    }

    pub fn engine(u: *Up) !void {
        const r = &u.r.?;
        u.e = try glm.engine.Engine.create(u.gpa, &r.m, &u.w.?, u.a.capacity, 8, u.a.prefill_rows, u.a.long_context);
        if (u.a.sampling()) |s| { // the nucleus rows before attach: they live in the Device's block
            u.e.?.sampling = s;
            try u.e.?.useSampling(u.gpa);
        }
        u.dev = try glm.engine.Device.attach(u.e.?, u.d);
        try @import("glm_run.zig").logBytes(u.e.?, u.a.rank, "");
    }

    fn deinit(u: *Up) void {
        u.pair.close();
        if (u.dev) |*x| x.detach();
        if (u.e) |x| x.destroy(u.gpa);
        if (u.w) |*x| x.deinit(u.gpa);
        if (u.run) |*x| {
            for (x.packed_groups.items) |*y| x.free(y);
            x.packed_groups.deinit(u.gpa);
            for (x.stored) |*y| if (y.*) |*z| x.free(z);
            u.gpa.free(x.stored);
            u.gpa.free(x.rows);
        }
        if (u.stream) |*x| x.deinit();
        if (u.pk) |*x| x.deinit();
        if (u.r) |*x| x.deinit(u.gpa);
        if (u.ix) |*x| x.deinit();
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// bench's runs re-arm the watchdog (the verb is armed once, for TF_TP_TIMEOUT_S).
fn rearm(ctx: *anyopaque) void {
    const p: *glm.ranks.Pair = @ptrCast(@alignCast(ctx));
    p.arm(0);
}

pub fn run(init: std.process.Init, a: Args, t: Tokens) !u8 {
    const gpa = init.gpa;
    const sd = exl3Sched(a.rank) catch return 2; // before the bring-up: a bad switch costs no load
    cuda.watch.installStop();
    var driver = cuda.Driver.open() catch |e| {
        std.debug.print("FAIL rank {d}: no CUDA driver ({t})\n", .{ a.rank, e });
        return 2;
    };
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var u: Up = .{ .gpa = gpa, .io = init.io, .a = a, .t = t, .d = &driver, .pair = .{ .gpa = gpa, .io = init.io, .rank = a.rank } };
    defer u.deinit();
    const up0 = nowNs();
    glm.ranks.bringUp(&u) catch |e| {
        std.debug.print("FAIL rank {d}: bring-up: {t}\n", .{ a.rank, e });
        return u.pair.lost(e);
    };
    if (u.run) |x| std.debug.print("TIME rank {d}: bring-up {d:.0} ms; the load {d:.0} ms (reads {d:.0} ms, packing {d:.0} ms; stored groups at most {d:.2} GiB at once); MemAvailable {d:.2} GiB\n", .{ a.rank, ms(nowNs() - up0), ms(u.load_ns), ms(x.read_ns), ms(x.pack_ns), @as(f64, @floatFromInt(x.peak_stored)) / (1 << 30), tool.memAvailableGib() });
    const e = u.e.?;
    const m = &u.r.?.m;
    var set = try cuda.aot.Set.load(gpa, init.io, &driver, ctx.device, a.triton orelse return error.NoTritonDir);
    defer set.deinit();
    var kda = try kdaLoad(&driver);
    defer kda.unload();
    var mm = try mmLoad(&ctx);
    defer if (comptime has_mm) mm.unload();
    var x3: ?X3 = if (m.quant == .exl3) try exl3Load(&driver, sd) else null;
    defer if (comptime has_exl3) if (x3) |*s| s.unload();
    const optin: u32 = if (comptime has_exl3) @intCast(try ctx.attribute(.max_shared_memory_per_block_optin)) else 0;
    var dsa = try dsaLoad(&driver);
    defer if (comptime has_dsa) if (dsa) |*k| k.unload();
    e.chip = .{ .major = @intCast(try ctx.attribute(.compute_capability_major)), .minor = @intCast(try ctx.attribute(.compute_capability_minor)), .sms = @intCast(try ctx.attribute(.multiprocessor_count)) };
    var sk = try glm.sample.Kernels.loadImages(&driver, .{ img.pointwise, img.topk, img.indexing, img.movement });
    defer sk.unload();
    const wide = a.prefill_rows * m.ll() * 128;
    var q = try cuda.DeviceBuffer.alloc(&driver, wide * 4);
    defer q.free();
    var y = try cuda.DeviceBuffer.alloc(&driver, wide * 2);
    defer y.free();
    var f = glm.fwd.Fwd.launching(m, .{ .stream = u.stream.?, .set = &set, .kernels = kernels(&kda, &mm, if (x3) |*s| s else null, optin, if (dsa) |*k| k else null, sd), .comm = &u.pair.comm.?, .q_tmp = q.ptr, .y_tmp = y.ptr });
    e.smp.k = &sk;
    try glm.sample.prepare(gpa, &f, &e.smp);
    var nk: ?glm.nucleus.Kernels = if (e.nuc != null) try glm.nucleus.Kernels.loadImages(&driver, &sk, .{ img.nucleus, img.argmax_f64 }) else null;
    defer if (nk) |*x| x.unload();
    if (nk) |*x| e.nuc.?.k = x;
    var rec: verbs.RecorderT = .{ .gpa = gpa, .out = a.out, .stream = u.stream };
    defer rec.entries.deinit(gpa);
    u.pair.arm(0);
    const res: anyerror!usize = switch (a.verb) {
        .first => widen(verbs.firstFn(e, &f, a, t, gpa)),
        .teacher => verbs.teacherFn(e, &f, a, t, .{ .stream = u.stream }, &rec, gpa),
        .dump => widen(verbs.dumpFn(e, &f, a, t, &rec, gpa)),
        .bench => @import("glm_bench.zig").run(e, &f, a, t, .{ .stream = u.stream }, .{ .ctx = &u.pair, .call = rearm }, gpa),
    };
    u.pair.disarm();
    const n = res catch |err| {
        std.debug.print("FAIL rank {d}: {t}: {t}\n", .{ a.rank, a.verb, err });
        return u.pair.lost(err);
    };
    try u.stream.?.synchronize();
    std.debug.print("PASS rank {d}: {t} done ({d}) under {s}\n", .{ a.rank, a.verb, n, a.out });
    return 0;
}
