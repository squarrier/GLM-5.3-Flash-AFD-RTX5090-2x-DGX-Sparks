//! One TP rank of the CUDA GLM inside `tensorfold-native serve` (qualification_qualification_44ba554e.8 day 2), brought up as tf-glm-forward's
//! (zig/tests/cuda/glm_run_gpu.zig, lane B's) does it, in python GlmEngine's order (glm/cuda_ranks.zig's bringUp): the
//! control link and NCCL over it, a barrier; the settings both ranks share, then each rank's admission (its own plan's
//! bytes against its own MemAvailable, glm_cuda_admit.zig) agreed; the rank's weights by tf-glm-forward's loader (the
//! split plan, each stored group read, packed and freed, the views bound); ready and a barrier; the Engine on its
//! device block. Then what the forward launches: the Triton set (TENSORFOLD_CUDA_KERNELS) and the 17 extension images
//! tf-glm-forward embeds, read from TF_GLM_FATBINS (qualification_e84ee1bf's pinned files, each sha256 logged), and the GPU Fwd.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const tool = @import("glm_tool");
const admit = @import("glm_cuda_admit.zig");
const Allocator = std.mem.Allocator;

/// The images tf-glm-forward embeds (zig/build/glm_fwd_tool.zig's list), each `<name>.fatbin` in TF_GLM_FATBINS.
pub const image_names = [_][]const u8{ "glm_pack", "glm_kda", "torch_pointwise", "torch_topk", "torch_indexing", "torch_movement", "glm_qmm_group", "glm_qmm_prefill", "glm_experts", "glm_experts_prefill", "glm_select", "exl3_experts", "exl3_grouped_cb0", "exl3_grouped_cb1", "exl3_grouped_cb2", "exl3_linear", "glm_exl3" };

/// What a rank serves with: the settings both ranks agree, and where its files are.
pub const Options = struct {
    dir: [:0]const u8,
    rank: u32,
    master: []const u8,
    capacity: usize,
    prefill_rows: usize,
    long_context: bool = true,
    triton: []const u8,
    fatbins: []const u8,
    floor_gib: f64 = admit.floor_gib,
    /// drafts on (qualification_qualification_44ba554e.8 day 3): the MTP layer loaded, and the policy both ranks agree (glm_drafts.parse's spec)
    drafts: bool = false,
    draft_spec: []const u8 = "0",
    /// TF_GLM_EXL3_L2D / TF_GLM_EXL3_PDL (qualification_96d80965), from the open's settings (glm_cuda.envFrom): off by default
    exl3_sched: glm.exl3.experts.Sched = .{},
    /// rank_fingerprint: TF_GLM_FINGERPRINT=1 (glm_cuda.envFrom): the settings end ` fp=1` and the identities are agreed before the
    /// load (glm.ranks.bringUp's identity step); off by default
    fingerprint: bool = false,
};

const Mm = @typeInfo(@FieldType(glm.fwd.Kernels, "mm")).pointer.child;
const Dsa = @typeInfo(@typeInfo(@FieldType(glm.fwd.Kernels, "dsa")).optional.child).pointer.child;
const X3 = glm.exl3.kernels.Set;

pub const Rank = struct {
    gpa: Allocator,
    io: std.Io,
    o: Options,
    d: *const cuda.Driver,
    ctx: *const cuda.Context,
    pair: glm.ranks.Pair,
    images: [image_names.len]?[]align(64) u8 = @splat(null),
    /// Mia's 0128 (the open's exl3_sched, set by init); its four exl3_l2d images, read only when a switch is on
    sched: glm.exl3.experts.Sched = .{},
    l2d: [glm.exl3.kernels.images_l2d.len]?[]align(64) u8 = @splat(null),
    ix: ?glm.index.Index = null,
    r: ?tool.Rank = null,
    w: ?glm.views.Weights = null,
    e: ?*glm.engine.Engine = null,
    run: ?tool.Run = null,
    pk: ?glm.pack.Kernels = null,
    stream: ?cuda.Stream = null,
    dev: ?glm.engine.Device = null,
    set: ?cuda.aot.Set = null,
    kda: ?glm.kda_kernels.Kda = null,
    mm: ?Mm = null,
    x3: ?X3 = null,
    dsa: ?Dsa = null,
    sk: ?glm.sample.Kernels = null,
    q: ?cuda.DeviceBuffer = null,
    y: ?cuda.DeviceBuffer = null,
    f: glm.fwd.Fwd = undefined,
    serve: glm.serve.Serve = undefined,
    forecast: ?admit.Forecast = null,
    /// a refusal's words (the admission), for the open's problem
    problem: []const u8 = "",
    load_ns: u64 = 0,
    up_ns: u64 = 0,

    /// The images read (each sha256 logged) and the Triton set's aot.json found, before any peer is met. With Mia's 0128
    /// switches on (the open's settings, refused in words there when malformed: glm_cuda.envFrom), one INFO line and
    /// exl3_l2d's four images beside the 17; unset, nothing more is read.
    pub fn init(r: *Rank) !void {
        r.sched = r.o.exl3_sched;
        if (r.sched.on()) std.debug.print("INFO rank {d}: EXL3 experts: Mia's 0128 kernels (exl3_l2d), L2 prefetch {d} steps ahead, programmatic launch {s}\n", .{ r.o.rank, r.sched.l2d, if (r.sched.pdl) "on" else "off" });
        for (image_names, &r.images) |name, *im| im.* = try r.read(name);
        if (r.sched.on()) for (glm.exl3.kernels.images_l2d, &r.l2d) |im, *b| {
            b.* = try r.read(@tagName(im));
        };
        var buf: [4096]u8 = undefined;
        std.Io.Dir.cwd().access(r.io, try std.fmt.bufPrint(&buf, "{s}/aot.json", .{r.o.triton}), .{}) catch return error.NoTritonSet;
    }

    /// `<name>.fatbin` from TF_GLM_FATBINS, its size and sha256 logged.
    fn read(r: *Rank, name: []const u8) ![]align(64) u8 {
        var buf: [4096]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/{s}.fatbin", .{ r.o.fatbins, name });
        const bytes = try slurpAligned(r.gpa, path);
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &sha, .{});
        std.debug.print("INFO rank {d}: image {s}.fatbin {d} B sha256 {s}\n", .{ r.o.rank, name, bytes.len, std.fmt.bytesToHex(sha, .lower) });
        return bytes;
    }

    fn image(r: *const Rank, name: []const u8) []const u8 {
        for (image_names, r.images) |n, im| if (std.mem.eql(u8, n, name)) return im.?;
        unreachable;
    }

    pub fn comm(r: *Rank) !void {
        const c = std.mem.lastIndexOfScalar(u8, r.o.master, ':') orelse return error.Master;
        const at = try std.Io.net.IpAddress.parse(r.o.master[0..c], try std.fmt.parseInt(u16, r.o.master[c + 1 ..], 10));
        try r.pair.openLink(at, 600_000);
        // NCCL carries every gather, as in tf-glm-forward's build (its runtime has no kernel images for comm.zig's own
        // RoCE all-gather or the L2 prefetch); the rails are NCCL's (NCCL_IB_HCA and the rest, the kit's ZL_ROCE=1)
        var opt = cuda.comm.Options.fromEnv();
        opt.roce = false;
        opt.l2_prefetch = false;
        r.pair.comm = try cuda.comm.Comm.init(r.gpa, r.pair.link.?, r.rank(), 2, opt);
        try r.pair.startWatch();
    }

    fn rank(r: *const Rank) u32 {
        return r.o.rank;
    }

    pub fn barrier(r: *Rank, label: []const u8) !void {
        try r.pair.barrier(label, 0);
    }

    /// python's `mine` agreed (the MTP layer with drafts on, the run `serve` with its warm-up and the draft policy),
    /// then the admission.
    pub fn settings(r: *Rank) !void {
        var runbuf: [96]u8 = undefined;
        const run = if (r.o.drafts) try std.fmt.bufPrint(&runbuf, "serve warm=96x16 drafts={s}", .{r.o.draft_spec}) else "serve warm=96x16";
        try r.pair.agree(.{ .capacity = r.o.capacity, .long_context = r.o.long_context, .prefill_rows = r.o.prefill_rows, .mtp = r.o.drafts, .run = run, .fp = r.o.fingerprint });
        try r.plan();
        const f = try admit.forecast(&r.r.?, r.e.?, r.o.rank, r.o.prefill_rows, r.gpa);
        r.forecast = f;
        const avail = tool.memAvailableGib();
        var buf: [1024]u8 = undefined;
        std.debug.print("FORECAST {s}\n", .{try admit.describe(f, avail, r.o.floor_gib, &buf)});
        const mine = admit.fits(f, avail, r.o.floor_gib);
        // both verdicts over the link: a refusal on either rank stops both cleanly (no lost peer, no exit 3)
        const w = &r.pair.watch.?;
        w.idle.store(true, .release);
        r.pair.link.?.agree(r.gpa, r.o.rank, if (mine) "admitted" else "refused") catch |e| {
            w.quiet();
            if (e != error.RanksDisagree) return e;
            r.problem = if (mine) "the other rank refused its admission (its log has the numbers)" else try admit.refusal(r.gpa, f, avail, r.o.floor_gib) orelse "";
            return error.Refused;
        };
        if (!mine) {
            w.quiet();
            r.problem = try admit.refusal(r.gpa, f, avail, r.o.floor_gib) orelse "";
            return error.Refused;
        }
        w.idle.store(false, .release);
    }

    /// rank_fingerprint: TF_GLM_FINGERPRINT=1 on this rank (glm.ranks.bringUp takes the identity step only then).
    pub fn fingerprint(r: *const Rank) bool {
        return r.o.fingerprint;
    }

    /// rank_fingerprint: this rank's identity agreed with the other's before the load: `tp-rank` on its arch, the checkpoint's
    /// config.json and name, the binary, the fatbins `init` read (and the Triton set's aot.json), -Dsource_id.
    pub fn identity(r: *Rank) !void {
        const fp = cuda.fingerprint;
        var k: fp.Kernels = .{};
        for (image_names, r.images) |name, im| if (im) |b| try k.add(name, b);
        for (glm.exl3.kernels.images_l2d, r.l2d) |im, b| if (b) |x| try k.add(@tagName(im), x);
        try k.addTriton(r.o.triton);
        const l = try fp.local("tp-rank", try r.ctx.capability(), r.o.dir, &k);
        var buf: [fp.line_max]u8 = undefined;
        const line = try l.identity().line(&buf);
        std.debug.print("INFO rank {d}: " ++ fp.name ++ "=1: identity {s}\n", .{ r.o.rank, line });
        try r.pair.agreeIdentity(line);
    }

    /// The rank's plans and its Engine's shapes (no device memory yet): what the admission sizes. With drafts on the
    /// MTP layer is planned and the Engine holds its buffers (python's Weights.mtp); off, it is never read.
    pub fn plan(r: *Rank) !void {
        r.ix = try glm.index.open(r.gpa, r.o.dir);
        r.r = try tool.plans(r.gpa, r.toolArgs(), &r.ix.?);
        if (!r.o.drafts) r.r.?.m.mtp_layers = 0; // drafts off: the MTP head is never read (qualification_e84ee1bf QUESTION 1)
        r.w = try glm.views.rank(r.gpa, &r.r.?.m);
        r.e = try glm.engine.Engine.create(r.gpa, &r.r.?.m, &r.w.?, r.o.capacity, 8, r.o.prefill_rows, r.o.long_context);
    }

    fn toolArgs(r: *const Rank) tool.Args {
        return .{ .dir = r.o.dir, .rank = r.o.rank, .ranks = 2, .keep = true, .mtp_layer = r.o.drafts };
    }

    /// weights.load for the rank, as tf-glm-forward's: read, packed onto the GPU, the packed buffers kept and bound.
    pub fn load(r: *Rank) !void {
        const t0 = nowNs();
        defer r.load_ns = nowNs() - t0;
        const p = &r.r.?;
        r.pk = try glm.pack.Kernels.load(r.d, r.image("glm_pack"));
        r.stream = try cuda.Stream.init(r.d, true);
        const rows = try r.gpa.alloc(tool.Row, p.pk.items.items.len);
        @memset(rows, .{});
        r.run = .{ .gpa = r.gpa, .a = r.toolArgs(), .ix = &r.ix.?, .r = p, .rows = rows, .stored = try r.gpa.alloc(?tool.Run.Mem, p.split.groups().len), .driver = r.d, .stream = r.stream, .kernels = &r.pk.? };
        @memset(r.run.?.stored, null);
        try r.run.?.all();
        const bound = try glm.pack.bind(&r.w.?, &p.pk);
        std.debug.print("INFO rank {d}: {d} views bound to the packed rank\n", .{ r.o.rank, bound });
    }

    pub fn ready(r: *Rank, label: []const u8) !void {
        try r.pair.barrier(label, glm.ranks.ready_s);
    }

    pub fn engine(r: *Rank) !void {
        r.dev = try glm.engine.Device.attach(r.e.?, r.d);
        const b = try r.e.?.deviceBytes();
        std.debug.print("INFO rank {d}: the Engine's device block {d:.1} MiB at capacity {d}: the prompt Buffers ({d} rows) {d:.1} MiB, the DSA scratch {d:.1} MiB\n", .{ r.o.rank, mib(b.total), r.o.capacity, r.o.prefill_rows, mib(b.pbuf), mib(b.dsa) });
    }

    /// The bring-up (python's order), then the kernels and the GPU Fwd the serve calls run on.
    pub fn up(r: *Rank) !void {
        const t0 = nowNs();
        try glm.ranks.bringUp(r);
        r.up_ns = nowNs() - t0;
        const x = r.run.?;
        std.debug.print("TIME rank {d}: bring-up {d:.0} ms; the load {d:.0} ms (reads {d:.0} ms, packing {d:.0} ms; stored groups at most {d:.2} GiB at once); MemAvailable {d:.2} GiB\n", .{ r.o.rank, ms(r.up_ns), ms(r.load_ns), ms(x.read_ns), ms(x.pack_ns), @as(f64, @floatFromInt(x.peak_stored)) / (1 << 30), tool.memAvailableGib() });
        try r.kernels();
    }

    fn kernels(r: *Rank) !void {
        const e = r.e.?;
        const m = &r.r.?.m;
        r.set = try cuda.aot.Set.load(r.gpa, r.io, r.d, r.ctx.device, r.o.triton);
        r.kda = try kdaLoad(r.d, r.image("glm_kda"));
        r.mm = try Mm.load(r.ctx, .{ .qmm_group = r.image("glm_qmm_group"), .qmm_prefill = r.image("glm_qmm_prefill"), .experts = r.image("glm_experts"), .experts_prefill = r.image("glm_experts_prefill") });
        if (m.quant == .exl3) {
            var s: X3 = .{};
            errdefer s.unload();
            inline for (glm.exl3.kernels.images_all) |im| s.modules[@backingInt(im)] = try cuda.Module.load(r.d, r.image(@tagName(im)));
            if (r.sched.on()) inline for (glm.exl3.kernels.images_l2d, r.l2d) |im, b| {
                s.modules[@backingInt(im)] = try cuda.Module.load(r.d, b.?); // 0128's, only with a switch on
            };
            r.x3 = s;
        }
        r.dsa = try Dsa.load(r.d, r.image("glm_select"));
        const optin: u32 = @intCast(try r.ctx.attribute(.max_shared_memory_per_block_optin));
        e.chip = .{ .major = @intCast(try r.ctx.attribute(.compute_capability_major)), .minor = @intCast(try r.ctx.attribute(.compute_capability_minor)), .sms = @intCast(try r.ctx.attribute(.multiprocessor_count)) };
        r.sk = try glm.sample.Kernels.loadImages(r.d, .{ r.image("torch_pointwise"), r.image("torch_topk"), r.image("torch_indexing"), r.image("torch_movement") });
        const wide = r.o.prefill_rows * m.ll() * 128;
        r.q = try cuda.DeviceBuffer.alloc(r.d, wide * 4);
        r.y = try cuda.DeviceBuffer.alloc(r.d, wide * 2);
        var k: glm.fwd.Kernels = .{ .kda = &r.kda.?, .mm = &r.mm.?, .dsa = &r.dsa.? };
        k.exl3 = if (r.x3) |*s| s else null;
        k.exl3_optin = optin;
        k.exl3_sched = r.sched;
        r.f = glm.fwd.Fwd.launching(m, .{ .stream = r.stream.?, .set = &r.set.?, .kernels = k, .comm = &r.pair.comm.?, .q_tmp = r.q.?.ptr, .y_tmp = r.y.?.ptr });
        e.smp.k = &r.sk.?;
        try glm.sample.prepare(r.gpa, &r.f, &e.smp);
        r.serve = .{ .e = e, .f = &r.f, .absorb = r.o.drafts };
    }

    /// The device and host memory the rank holds, freed (the pair is closed by its owner first).
    pub fn deinit(r: *Rank) void {
        if (r.q) |*x| x.free();
        if (r.y) |*x| x.free();
        if (r.sk) |*x| x.unload();
        if (r.dsa) |*x| x.unload();
        if (r.x3) |*x| x.unload();
        if (r.mm) |*x| x.unload();
        if (r.kda) |*x| x.unload();
        if (r.set) |*x| x.deinit();
        if (r.dev) |*x| x.detach();
        if (r.e) |x| x.destroy(r.gpa);
        if (r.w) |*x| x.deinit(r.gpa);
        if (r.run) |*x| {
            for (x.packed_groups.items) |*y| x.free(y);
            x.packed_groups.deinit(r.gpa);
            for (x.stored) |*y| if (y.*) |*z| x.free(z);
            r.gpa.free(x.stored);
            r.gpa.free(x.rows);
        }
        if (r.stream) |*x| x.deinit();
        if (r.pk) |*x| x.deinit();
        if (r.r) |*x| x.deinit(r.gpa);
        if (r.ix) |*x| x.deinit();
        for (r.images) |im| if (im) |x| r.gpa.free(x);
        for (r.l2d) |im| if (im) |x| r.gpa.free(x);
    }
};

/// kda_kernels.Kda from the glm_kda image (tf-glm-forward's kdaLoad).
fn kdaLoad(d: *const cuda.Driver, bytes: []const u8) !glm.kda_kernels.Kda {
    var m = try cuda.Module.load(d, bytes);
    errdefer m.unload();
    const s = glm.kda_kernels.sym;
    return .{ .module = m, .chain_fn = try m.function(s.chain), .replay_fn = try m.function(s.replay), .replay_layers_fn = try m.function(s.replay_layers), .prep_fn = try m.function(s.prep), .step_fn = try m.function(s.step), .out_fn = try m.function(s.out) };
}

/// A file's bytes at a 64-byte boundary (a fatbin's header wants its alignment), at most 256 MiB.
fn slurpAligned(gpa: Allocator, path: []const u8) ![]align(64) u8 {
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) {
        std.log.err("cannot open {s}", .{path});
        return error.FileNotFound;
    }
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end <= 0 or end > 256 << 20) return error.BadImageSize;
    const out = try gpa.alignedAlloc(u8, .fromByteUnits(64), @intCast(end));
    errdefer gpa.free(out);
    var done: usize = 0;
    while (done < out.len) {
        const n = std.c.pread(fd, out.ptr + done, out.len - done, @intCast(done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
    return out;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn mib(n: usize) f64 {
    return @as(f64, @floatFromInt(n)) / (1024.0 * 1024.0);
}
