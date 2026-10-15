//! `tf-exl3-check CKPT REF [--dry] [--listing FILE]`: the Zig half of the EXL3 kernel check against exl3_check.py's REF.
//! It plans the manifest's TP2 rank, reads only layer L's experts group, writes Plan.tables into it, uploads and binds
//! it, then replays each case through the bindings and compares the outputs' bytes with python-0.6's. --dry does all
//! but CUDA (a header-only copy of the checkpoint serves). PASS/FAIL lines on stdout; exit 1 on any FAIL.
//! TF_GLM_EXL3_L2D / TF_GLM_EXL3_PDL take Mia's 0128 kernels (qualification_6052b66d); --bench N times GLM's routed call (BENCH lines).
//! TF_GLM_EXL3_X1 / TF_GLM_EXL3_SEQ take X1 and SEQ in the routed cases of 64+ rows (qualification_b3fba1da, glm_exl3_prompt.zig).

const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const x3 = glm.exl3;
const chk = x3.check;
/// qualification_3e1f1105: --lists and the PGROUP bench (TF_GLM_EXL3_PGROUP, bertholomus's parallel grouping)
const lg = @import("glm_exl3_lists.zig");
/// qualification_8724e161: --ident and the MMA bench (TF_GLM_EXL3_MMA, bertholomus's grouped_mma)
const mt = @import("glm_exl3_mma.zig");
const ex = x3.experts;
const gm = x3.glm;
const lin = x3.linear;
const pr = @import("glm_exl3_prompt.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Obj = std.json.ObjectMap;
const n_kinds = @typeInfo(chk.Kind).@"enum".field_names.len;
const k4c = @import("glm_exl3_check_k4.zig"); // qualification_e4ad04cd

const usage =
    \\usage: tf-exl3-check CKPT REF [--dry] [--listing FILE] [--bench N] [--l2d-kernels]
    \\  CKPT     Mia's EXL3 checkpoint folder (078455ff; --dry: a header-only copy serves)
    \\  REF      exl3_check.py's output folder: manifest.json, in/, out/
    \\  --dry    everything but CUDA: plan, read, tables, layout hashes, each case's shapes and launches
    \\  --listing FILE   each case's arrays and scratch bytes as exl3_check.py's dry.txt lists them
    \\  --bench N        after each routed-bf16 case of 1-16 rows and of 64, 128, 594, 1,024 and 2,048 rows: its routed
    \\                   call N times, then gate/up with its epilogue, gate/up alone and down alone (PDL off), medians
    \\                   in us; with SEQ on, X1's launches and SEQ's alternate run by run (BENCH-AB: their ratio);
    \\                   the outputs are compared again after
    \\  --l2d-kernels    0128's kernels with TF_GLM_EXL3_L2D and TF_GLM_EXL3_PDL unset or 0 too (a bench arm)
    \\  env TF_GLM_EXL3_L2D (0..64) and TF_GLM_EXL3_PDL (0 or 1): Mia's 0128 in the routed calls (default off)
    \\  env TF_GLM_EXL3_X1 (0 or 1, its four 0 or 1) and TF_GLM_EXL3_SEQ (0 or 1, only with X1): the halves' prompt switches
    \\  --lists          group's and groupPar's lists byte for byte at R 64-2,049 (qualification_3e1f1105's G-lists)
    \\  env TF_GLM_EXL3_PGROUP (0 or 1): the parallel grouping in the routed cases of 64+ rows; with --bench N,
    \\                   its arms at 1,024 and 2,048 rows (TF_GLM_EXL3_X1 and its four: X1 in that routed call)
    \\  --ident          X1's routed y against MMA's (and SEQ with MMA) byte for byte at R 64-2,048 (qualification_8724e161's G-ident)
    \\  env TF_GLM_EXL3_MMA (0 or 1, only with X1 and PGROUP): bertholomus's grouped_mma in the routed cases of 64+
    \\                   rows; with --bench N, its arms (X1, + SEQ, + PGROUP, + both, + PGROUP + MMA, + all three) at
    \\                   64-2,048 rows instead of the BENCH lines above
    \\  env TF_GLM_HALVES_KERNEL=k4: K4's table instead (glm_exl3_check_k4.zig; REF from exl3_check.py with
    \\                   TF_EXL3_CHECK_K4=1 on python 2.15's tree); --bench N then times every K4 case against today's
    \\
;

const Args = struct { ckpt: []const u8 = "", ref: []const u8 = "", dry: bool = false, listing: ?[]const u8 = null, bench: u32 = 0, x: bool = false, lists: bool = false, ident: bool = false };

/// One case's buffers: device ones (or --dry stand-ins), and the host bytes an upload reads until the stream syncs.
pub const Slot = struct { name: []const u8, addr: u64, bytes: u64, buf: ?cuda.DeviceBuffer = null, host: ?[]u8 = null };

pub const Bufs = struct {
    s: [20]Slot = undefined,
    n: usize = 0,

    pub fn get(b: *const Bufs, name: []const u8) ?*const Slot {
        for (b.s[0..b.n]) |*x| if (std.mem.eql(u8, x.name, name)) return x;
        return null;
    }

    pub fn at(b: *const Bufs, name: []const u8) u64 {
        return if (b.get(name)) |x| x.addr else 0;
    }
};

pub const Dev = struct { d: *const cuda.Driver, ctx: *const cuda.Context, stream: cuda.Stream, set: x3.kernels.Set, sh: ex.Shared };

pub const Tool = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    w: *std.Io.Writer,
    a: Args,
    man: Obj = undefined,
    layer: u32 = 0,
    rank: u32 = 0,
    facts: chk.Facts = undefined,
    plan: x3.split.Plan = undefined,
    e: x3.split.Experts = undefined,
    bound: ex.Layer = undefined,
    glm_ex: ?gm.Experts = null,
    base: u64 = 1 << 40, // the group's device address (--dry: a stand-in)
    next: u64 = 1 << 41, // --dry's next stand-in address
    strips: [2][2]u64 = .{ .{ 0, 0 }, .{ 0, 0 } }, // gate and down, experts 0 and 7: the re-laid-out words
    dev: ?*Dev = null,
    fails: u32 = 0,
    launches: usize = 0,
    sched: ex.Sched = .{},
    x1: pr.X1 = .{},
    suh_shared: bool = false, // qualification_3e1f1105
    mma_cases: u32 = 0, // qualification_8724e161: the cases whose launches hold exl3_mma's kernels
    k4: ?k4c.State = null, // qualification_e4ad04cd: TF_GLM_HALVES_KERNEL=k4, K4's table (glm_exl3_check_k4.zig)

    pub fn out(t: *Tool, comptime fmt: []const u8, args: anytype) void {
        t.w.print(fmt ++ "\n", args) catch {};
        t.w.flush() catch {};
    }

    pub fn fail(t: *Tool, comptime fmt: []const u8, args: anytype) void {
        t.fails += 1;
        t.out("FAIL " ++ fmt, args);
    }
};

fn field(o: Obj, name: []const u8) ?std.json.Value {
    return o.get(name);
}

pub fn int(o: Obj, name: []const u8) i64 {
    const v = o.get(name) orelse return -1;
    return switch (v) {
        .integer => |i| i,
        .bool => |b| @intFromBool(b),
        else => -1,
    };
}

pub fn str(o: Obj, name: []const u8) []const u8 {
    const v = o.get(name) orelse return "";
    return if (v == .string) v.string else "";
}

pub fn hex(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var a: Args = .{};
    var pos: usize = 0;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (std.mem.eql(u8, k, "--dry")) {
            a.dry = true;
        } else if (std.mem.eql(u8, k, "--listing") and i + 1 < argv.len) {
            i += 1;
            a.listing = argv[i];
        } else if (std.mem.eql(u8, k, "--bench") and i + 1 < argv.len) {
            i += 1;
            a.bench = std.fmt.parseInt(u32, argv[i], 10) catch 0;
            if (a.bench == 0) pos = 9;
        } else if (std.mem.eql(u8, k, "--l2d-kernels")) {
            a.x = true;
        } else if (std.mem.eql(u8, k, "--lists")) {
            a.lists = true;
        } else if (std.mem.eql(u8, k, "--ident")) {
            a.ident = true;
        } else if (pos < 2 and k.len > 0 and k[0] != '-') {
            if (pos == 0) a.ckpt = k else a.ref = k;
            pos += 1;
        } else pos = 9;
    }
    if (pos != 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var sd = ex.Sched.fromEnv() catch {
        std.debug.print("FAIL tf-exl3-check: TF_GLM_EXL3_L2D takes 0..64 and TF_GLM_EXL3_PDL 0 or 1\n", .{});
        return 2;
    };
    sd.x = a.x;
    const p = ex.Prompt.fromEnv() catch {
        std.debug.print("FAIL tf-exl3-check: TF_GLM_EXL3_X1 takes 0 or 1 (with it 1 its four 0 or 1); TF_GLM_EXL3_PGROUP 0 or 1; TF_GLM_EXL3_SEQ 0 or 1, and 1 only with X1; TF_GLM_EXL3_MMA 0 or 1, and 1 only with X1 and PGROUP\n", .{});
        return 2;
    };
    const k4on = k4c.mode() catch {
        std.debug.print("FAIL tf-exl3-check: TF_GLM_HALVES_KERNEL takes k4 here (or 0, empty, unset)\n", .{});
        return 2;
    };
    var buf: [64 << 10]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buf);
    var t: Tool = .{ .gpa = init.gpa, .io = init.io, .w = &stdout.interface, .a = a, .sched = sd, .x1 = .{ .p = p }, .k4 = if (k4on) .{} else null };
    defer t.x1.deinit();
    if (p.pgroup) t.out("INFO tf-exl3-check: EXL3 experts: PGROUP (bertholomus's group_count/group_place) in the routed cases of {d}+ rows; X1 {s} in its bench", .{ ex.prompt_rows, if (p.on()) "on" else "off" });
    if (p.mma) t.out("INFO tf-exl3-check: EXL3 experts: MMA (bertholomus's grouped_mma, exl3_mma) in the routed cases of {d}+ rows over PGROUP's lists: gate/up {s}, down grouped_mma", .{ ex.prompt_rows, if (p.seq) "SEQ's kernel" else "grouped_mma, the splits folded" });
    if (sd.on()) t.out("INFO tf-exl3-check: EXL3 experts: Mia's 0128 kernels (exl3_l2d), l2d {d}, pdl {d}", .{ sd.l2d, @intFromBool(sd.pdl) });
    var arena: std.heap.ArenaAllocator = .init(init.gpa);
    defer arena.deinit();
    run(&t, arena.allocator()) catch |e| {
        t.fail("tf-exl3-check: {t} (no verdict for the cases after this)", .{e});
        return 1;
    };
    return if (t.fails == 0) 0 else 1;
}

fn readAll(t: *Tool, path: []const u8, most: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(t.io, path, t.gpa, .limited(most));
}

pub fn refFile(t: *Tool, rel: []const u8, most: usize) ![]u8 {
    const p = try std.fs.path.join(t.gpa, &.{ t.a.ref, rel });
    defer t.gpa.free(p);
    return readAll(t, p, most);
}

fn run(t: *Tool, ar: std.mem.Allocator) !void {
    const gpa = t.gpa;
    const text = try refFile(t, "manifest.json", 1 << 28);
    defer gpa.free(text);
    t.man = (try std.json.parseFromSliceLeaky(std.json.Value, ar, text, .{})).object;
    t.layer = @intCast(int(t.man, "layer"));
    t.rank = @intCast(int(t.man, "rank"));
    if (int(t.man, "dry") == 1 and !t.a.dry) return error.DryManifestHasNoOutputs;
    const cpath = try std.fs.path.join(gpa, &.{ t.a.ckpt, "config.json" });
    defer gpa.free(cpath);
    const cjson = try readAll(t, cpath, 1 << 22);
    defer gpa.free(cjson);
    var c = try glm.config.parse(gpa, cjson);
    try glm.config.subset(&c, @min(t.layer + 1, c.layers));
    var ix = try glm.index.open(gpa, t.a.ckpt);
    defer ix.deinit();
    t.plan = try x3.split.plan(gpa, &c, &ix.names, t.rank, 2);
    defer t.plan.deinit();
    t.e = t.plan.experts[t.layer] orelse return error.NoExpertsInLayer;
    const g = t.e.gate.trellis.g;
    const group = try gpa.alloc(u8, t.plan.groups.items[g]);
    var group_held = true;
    defer if (group_held) gpa.free(group);
    @memset(group, 0);
    try readGroup(t, &ix, g, group);
    var driver: cuda.Driver = undefined;
    var ctx: cuda.Context = undefined;
    var dv: Dev = undefined;
    if (!t.a.dry) {
        driver = try cuda.Driver.open();
        ctx = try cuda.Context.init(&driver, 0);
        dv = .{ .d = &driver, .ctx = &ctx, .stream = try cuda.Stream.init(&driver, true), .set = try x3.kernels.Set.load(&driver), .sh = .{ .optin = @intCast(try ctx.attribute(.max_shared_memory_per_block_optin)) } };
        if (t.sched.on()) try dv.set.loadL2d(&driver);
        if (t.x1.p.on()) try dv.set.loadX1(&driver); // qualification_3e1f1105's bench: X1's routed call
        if (t.x1.p.mma or t.a.ident) { // qualification_8724e161: MMA's image, and SEQ's for --ident's and --bench's SEQ arms
            try dv.set.loadX1s(&driver);
            try dv.set.loadMma(&driver);
        }
        if (t.k4 != null) { // qualification_e4ad04cd: K4's image, and X1's for its rotation a row (rot_rows)
            try dv.set.loadX1(&driver);
            try dv.set.loadK4(&driver);
        }
        t.dev = &dv;
        var name: [256]u8 = undefined;
        t.out("RESULT device: {s}, sm_{d}, opt-in shared {d} B a block", .{ try ctx.name(&name), try ctx.capability(), dv.sh.optin });
    }
    defer if (t.dev) |d| { // after every device buffer below is freed
        d.set.unload();
        d.stream.deinit();
        ctx.deinit();
        driver.close();
    };
    var gbuf: ?cuda.DeviceBuffer = null;
    defer if (gbuf) |*b| b.free();
    var sbufs: [4]?cuda.DeviceBuffer = @splat(null);
    defer for (&sbufs) |*s| if (s.*) |*b| b.free();
    if (t.dev) |d| {
        gbuf = try cuda.DeviceBuffer.alloc(d.d, group.len);
        t.base = gbuf.?.ptr;
    }
    const E = t.e.count;
    t.plan.tables(t.layer, t.base, group[t.e.tables.off..][0 .. 36 * E]);
    t.bound = t.plan.bind(t.layer, t.base).?;
    t.glm_ex = t.plan.bindGlm(t.layer, t.base);
    t.facts = .{ .layer = t.bound, .gate = .{ t.e.gate.k, t.e.gate.n }, .down = .{ t.e.down.k, t.e.down.n }, .limit = c.swiglu_limit };
    try layout(t, group);
    try t.x1.setup(if (t.dev) |d| d.d else null, if (t.dev) |d| &d.set else null, group, t.e, 2048, chk.slots, &t.next); // qualification_b3fba1da
    if (t.x1.p.on()) t.out("INFO tf-exl3-check: EXL3 experts: X1 on routed windows of {d}+ rows: {s}", .{ ex.prompt_rows, t.x1.note[0..t.x1.note_len] });
    t.suh_shared = try lg.sharedSuh(t, group);
    if (t.dev) |d| {
        try gbuf.?.upload(0, group); // pageable cuMemcpyHtoD: the context sync below orders it before any launch
        try d.ctx.synchronize();
    }
    try stripsUp(t, group, &sbufs);
    defer if (t.k4) |*s| if (s.suh_buf) |*b| b.free();
    if (t.k4 != null) try k4c.setup(t, group); // qualification_e4ad04cd: the shared suh test on the group's bytes, the row uploaded
    gpa.free(group);
    group_held = false;
    if (t.k4 != null) try k4c.cases(t) else try cases(t);
    if (t.x1.p.mma) t.out("INFO tf-exl3-check: EXL3 experts: MMA in {d} cases (their launches hold exl3_mma's work list and grouped_mma)", .{t.mma_cases});
    try lg.after(t);
    try mt.after(t);
}

/// Layer L's experts group alone, read as the loader reads it (O_DIRECT, io_uring), into host memory.
fn readGroup(t: *Tool, ix: *glm.index.Index, g: u32, group: []u8) !void {
    var copies: std.ArrayList(glm.split.Copy) = .empty;
    defer copies.deinit(t.gpa);
    for (t.plan.copies.items) |cp| if (cp.g == g) try copies.append(t.gpa, cp);
    var s = try glm.read.schedule(t.gpa, copies.items, .{});
    defer s.deinit();
    const views = try t.gpa.alloc([]u8, t.plan.groups.items.len);
    defer t.gpa.free(views);
    for (views) |*v| v.* = &.{};
    views[g] = group;
    var sink: glm.read.HostSink = .{ .groups = views };
    const st = try glm.read.run(t.gpa, ix.shards, &s, sink.sink(), .{});
    t.out("RESULT read: layer {d} rank {d}/2 experts group {d}: {d} copies in {d} reads, {d} B kept of {d} B read in {d:.2} s ({d:.3} GB/s, direct {}, ring {}); group {d} B", .{ t.layer, t.rank, g, copies.items.len, st.reads, st.kept, st.bytes, @as(f64, @floatFromInt(st.ns)) / 1e9, st.gbPerS(), st.direct, st.ring, group.len });
}

/// The group's bytes against python-0.6's: each expert's trellis, the stacked scales, the width tables.
fn layout(t: *Tool, group: []const u8) !void {
    const lf = (field(t.man, "layer_facts") orelse return error.NoLayerFacts).object;
    const E = t.e.count;
    var n: u32 = 0;
    var bad: u32 = 0;
    const projs = [_]struct { []const u8, x3.split.Proj }{ .{ "gate", t.e.gate }, .{ "up", t.e.up }, .{ "down", t.e.down } };
    for (projs, 0..) |p, j| {
        const want = (lf.get("trellis_sha256").?.object.get(p[0]) orelse return error.NoLayerFacts).array.items;
        if (want.len != E) return error.ExpertCount;
        for (0..E) |x| { // through the tables Plan.tables wrote: the pointer less the base, the width
            const ptr = std.mem.readInt(u64, group[t.e.tables.off + (j * E + x) * 8 ..][0..8], .little);
            const k2 = std.mem.readInt(u32, group[t.e.tables.off + 24 * E + (j * E + x) * 4 ..][0..4], .little);
            const len = @as(u64, p[1].k) * p[1].n * k2 / 16;
            n += 1;
            if (ptr < t.base or ptr - t.base + len > group.len) {
                bad += 1;
                t.fail("layout {s} expert {d}: the pointer table's {x} (+{d} B) is outside the group", .{ p[0], x, ptr, len });
                continue;
            }
            const h = hex(group[ptr - t.base ..][0..len]);
            if (!std.mem.eql(u8, &h, want[x].string)) {
                bad += 1;
                if (bad <= 5) t.fail("layout {s} expert {d}: trellis sha256 {s}, python-0.6's {s}", .{ p[0], x, &h, want[x].string });
            }
        }
        const k2 = hex(group[t.e.tables.off + 24 * E + j * 4 * E ..][0 .. 4 * E]);
        n += 1;
        if (!std.mem.eql(u8, &k2, str(lf.get("k2_sha256").?.object, p[0]))) {
            bad += 1;
            t.fail("layout {s}: the width table differs from python-0.6's {s}_k2", .{ p[0], p[0] });
        }
    }
    const scales = [_]struct { []const u8, x3.split.Proj, bool }{ .{ "suh_g", t.e.gate, true }, .{ "suh_u", t.e.up, true }, .{ "svh_g", t.e.gate, false }, .{ "svh_u", t.e.up, false }, .{ "suh_d", t.e.down, true }, .{ "svh_d", t.e.down, false } };
    for (scales) |s| {
        const r = if (s[2]) s[1].suh else s[1].svh;
        const h = hex(group[r.off..][0 .. @as(u64, E) * (if (s[2]) s[1].k else s[1].n) * 2]);
        n += 1;
        if (!std.mem.eql(u8, &h, str(lf.get("scales_sha256").?.object, s[0]))) {
            bad += 1;
            t.fail("layout {s}: the stacked scales differ from python-0.6's", .{s[0]});
        }
    }
    if (bad == 0) t.out("PASS layout: {d} of {d} ({d} expert trellises, 3 width tables, 6 scale stacks) byte-equal to python-0.6's", .{ n, n, 3 * E }) else t.out("FAIL layout: {d} of {d} byte-equal to python-0.6's", .{ n - bad, n });
}

/// linear.py's strips of gate and down, experts 0 and 7, re-laid out on the host and uploaded.
fn stripsUp(t: *Tool, group: []const u8, bufs: *[4]?cuda.DeviceBuffer) !void {
    for ([_]x3.split.Proj{ t.e.gate, t.e.down }, 0..) |p, i| for (chk.experts_checked, 0..) |x, j| {
        const len = @as(u64, p.k) * p.n * p.k2s[x] / 16;
        if (t.dev) |d| {
            const host = try t.gpa.alloc(u8, len);
            defer t.gpa.free(host);
            chk.strips(host, group[p.trellis.off + p.offsets[x] ..][0..len], p.k, p.n, p.k2s[x]);
            bufs[i * 2 + j] = try cuda.DeviceBuffer.alloc(d.d, len);
            try bufs[i * 2 + j].?.upload(0, host);
            t.strips[i][j] = bufs[i * 2 + j].?.ptr;
        } else {
            t.strips[i][j] = t.next;
            t.next += std.mem.alignForward(u64, len, 256);
        }
    };
    if (t.dev) |d| try d.ctx.synchronize();
}

pub fn trellisBytes(p: x3.split.Proj) u64 {
    return @as(u64, p.k) * p.n * p.k2s[0] / 16;
}

pub fn place(t: *Tool, b: *Bufs, name: []const u8, bytes: u64, how: enum { zero, minus1, upload }, host: ?[]u8) !void {
    if (b.get(name) != null) return;
    var s: Slot = .{ .name = name, .addr = t.next, .bytes = bytes, .host = host };
    if (t.dev) |d| {
        s.buf = try cuda.DeviceBuffer.alloc(d.d, bytes);
        s.addr = s.buf.?.ptr;
        switch (how) { // on the kernels' stream, so each lands before the launches
            .zero => try s.buf.?.fill8(0, d.stream.handle),
            .minus1 => try s.buf.?.fill32(0xffff_ffff, d.stream.handle),
            .upload => try s.buf.?.uploadAsync(0, host.?, d.stream.handle),
        }
    } else t.next += std.mem.alignForward(u64, bytes, 256) + 256;
    b.s[b.n] = s;
    b.n += 1;
}

pub fn release(t: *Tool, b: *Bufs) void {
    for (b.s[0..b.n]) |*s| {
        if (s.buf) |*x| x.free();
        if (s.host) |h| t.gpa.free(h);
    }
    b.n = 0;
}

pub fn findArr(list: std.json.Value, name: []const u8) ?Obj {
    for (list.array.items) |v| if (std.mem.eql(u8, str(v.object, "name"), name)) return v.object;
    return null;
}

/// The manifest's record of an array against the bindings' (dtype, shape, bytes): a mismatch's reason, or null.
pub fn sameArr(m: Obj, a: chk.Arr) ?[]const u8 {
    if (!std.mem.eql(u8, str(m, "dtype"), @tagName(a.dtype))) return "dtype";
    const sh = (m.get("shape") orelse return "shape").array.items;
    if (sh.len != a.dims) return "rank";
    for (sh, a.shape[0..a.dims]) |v, d| if (v.integer != d) return "shape";
    if (int(m, "bytes") != a.bytes()) return "bytes";
    return null;
}

/// The manifest's case against the table's and the bindings' spec: ids, call, arrays, scratch bytes.
fn agree(t: *Tool, mc: Obj, c: chk.Case, s: chk.Spec, id: []const u8) bool {
    const call = mc.get("call").?.object;
    var why: ?[]const u8 = null;
    if (!std.mem.eql(u8, str(mc, "kind"), @tagName(c.kind))) why = "kind";
    if (chk.isRouted(c.kind) or c.kind == .@"glm-mm") {
        if (int(call, "R") != c.rows or int(call, "rows") != c.rows) why = "rows";
        if (int(call, "limit_f32_bits") != @as(u32, @bitCast(s.limit))) why = "limit";
    }
    if (chk.isRouted(c.kind) and (!std.mem.eql(u8, str(call, "act"), @tagName(s.act)) or !std.mem.eql(u8, str(call, "x_dtype"), @tagName(s.x_dtype)) or int(call, "weighted") != @intFromBool(c.kind == .@"routed-wts"))) why = "call";
    if (chk.isRouted(c.kind)) { // the tiles experts.Scratch chose (nt, warps, K splits, prefetch) against the bindings'
        const tl = t.facts.tiles() catch return false;
        for ([_]struct { []const u8, ex.Tile }{ .{ "cfg_gu", tl[0] }, .{ "cfg_d", tl[1] } }) |p| {
            const v = if (call.get(p[0])) |x| x.array.items else &.{};
            if (v.len != 4 or v[0].integer != p[1].nt or v[1].integer != p[1].warps or v[2].integer != p[1].sk or v[3].integer != p[1].pf) why = "tile config";
        }
    }
    if (c.kind == .linear) {
        const sp = call.get("split").?.array.items;
        if (int(call, "M") != c.rows or sp[0].integer != s.split[0] or sp[1].integer != s.split[1]) why = "linear call";
    }
    const ins = mc.get("inputs").?;
    const outs = mc.get("outputs").?;
    if (ins.array.items.len != s.n_in or outs.array.items.len != s.n_out) why = "array count";
    for (s.inputs()) |a| why = why orelse if (findArr(ins, a.name)) |m| sameArr(m, a) else "inputs";
    for (s.outputs()) |a| why = why orelse if (findArr(outs, a.name)) |m| sameArr(m, a) else "outputs";
    const sc = mc.get("scratch").?.object;
    if (sc.count() != s.n_scratch) why = why orelse "scratch count";
    for (s.scratches()) |x| if (int(sc, x.name) != x.bytes) {
        why = why orelse "scratch bytes";
    };
    if (why) |r| {
        t.fail("{t} {s}: the manifest's {s} differs from the case table and the Zig bindings", .{ c.kind, id, r });
        return false;
    }
    return true;
}

fn cases(t: *Tool) !void {
    const want = try chk.table(t.gpa);
    defer t.gpa.free(want);
    const got = t.man.get("cases").?.array.items;
    var lst: std.Io.Writer.Allocating = .init(t.gpa);
    defer lst.deinit();
    try chk.head(&lst.writer, t.layer, t.rank, t.facts, .{ trellisBytes(t.e.gate), trellisBytes(t.e.up), trellisBytes(t.e.down) });
    var pass: [n_kinds]u32 = @splat(0);
    var total: [n_kinds]u32 = @splat(0);
    if (got.len != want.len) t.fail("cases: the manifest has {d}, the case table {d}", .{ got.len, want.len });
    for (want, 0..) |c, i| {
        var idb: [96]u8 = undefined;
        const id = c.id(&idb);
        total[@backingInt(c.kind)] += 1;
        const s = try chk.spec(c, t.facts);
        try chk.line(&lst.writer, id, c.kind, s);
        if (i >= got.len or !std.mem.eql(u8, str(got[i].object, "id"), id)) {
            t.fail("{t} {s}: not the manifest's case {d}", .{ c.kind, id, i });
            continue;
        }
        if (!agree(t, got[i].object, c, s, id)) continue;
        if (try one(t, got[i].object, c, s, id)) pass[@backingInt(c.kind)] += 1;
    }
    if (t.a.listing) |path| try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = lst.written() });
    for (0..n_kinds) |k| if (total[k] > 0) {
        const kind: chk.Kind = @fromBackingInt(@intCast(k));
        const ok = pass[k] == total[k];
        if (t.a.dry) {
            t.out("{s} dry {t}: {d} of {d} cases agree with the manifest (shapes, dtypes, scratch bytes, inputs' sha256); launches built", .{ if (ok) "PASS" else "FAIL", kind, pass[k], total[k] });
        } else t.out("{s} {t}: {d} of {d} cases byte-equal to python-0.6", .{ if (ok) "PASS" else "FAIL", kind, pass[k], total[k] });
    };
    t.out("RESULT {s}: {d} cases, {d} launches", .{ if (t.a.dry) "dry" else "check", want.len, t.launches });
}

/// One case: its buffers placed (inputs uploaded and checked, outputs zeroed, members -1), its launches built and,
/// on a GPU, run on one stream after a context sync; then each output's bytes against python-0.6's.
fn one(t: *Tool, mc: Obj, c: chk.Case, s: chk.Spec, id: []const u8) !bool {
    var b: Bufs = .{};
    defer release(t, &b);
    for (s.inputs()) |a| {
        const m = findArr(mc.get("inputs").?, a.name).?;
        const host = try refFile(t, str(m, "file"), 1 << 31);
        if (host.len != a.bytes() or !std.mem.eql(u8, &hex(host), str(m, "sha256"))) {
            t.gpa.free(host);
            t.fail("{t} {s}: input {s} is not the bytes the manifest lists", .{ c.kind, id, a.name });
            return false;
        }
        try place(t, &b, a.name, a.bytes(), .upload, host);
    }
    for (s.outputs()) |a| try place(t, &b, a.name, a.bytes(), .zero, null);
    for (s.scratches()) |x| try place(t, &b, x.name, x.bytes, if (std.mem.eql(u8, x.name, "members")) .minus1 else .zero, null);
    if (t.x1.p.pgroup and chk.isRouted(c.kind)) try place(t, &b, "pg_counts", 4 * @as(u64, t.bound.count), .zero, null); // qualification_3e1f1105
    if (t.x1.p.mma and chk.isRouted(c.kind)) try place(t, &b, "mma_work", 4 * @as(u64, ex.mma.workInts(c.rows, chk.slots, t.bound.count)), .zero, null); // qualification_8724e161
    var ls: x3.kernels.Launches = .{};
    build(t, &ls, &b, c, s, t.x1.p) catch |e| {
        t.fail("{t} {s}: the bindings refused the case ({t})", .{ c.kind, id, e });
        return false;
    };
    t.launches += ls.len;
    t.mma_cases += @intFromBool(for (ls.slice()) |l| {
        if (l.image == .exl3_mma_cb1) break true;
    } else false);
    const d = t.dev orelse return true;
    try d.ctx.synchronize();
    try d.set.run(d.stream, &ls);
    try d.stream.synchronize();
    for (ls.slice()) |l| if (l.max_dynamic) |v| {
        d.sh.max_dynamic = v; // the raised limit stays raised, as it does in python-0.6's process
    };
    var ok = true;
    for (s.outputs()) |a| ok = try compare(t, mc, b.get(a.name).?, a, c.kind, id) and ok;
    if (t.a.bench > 0 and !t.x1.p.mma and c.kind == .@"routed-bf16" and pr.benched(c.rows)) { // MMA: BENCH-MMA's arms
        var x1only = t.x1.p; // with SEQ: X1's launches as the other arm, alternating
        x1only.seq = false;
        var ls0: x3.kernels.Launches = .{};
        try build(t, &ls0, &b, c, s, x1only);
        const two = t.x1.p.seq and c.rows >= ex.prompt_rows;
        const arms: []const *const x3.kernels.Launches = if (two) &.{ &ls0, &ls } else &.{&ls};
        const labels: []const []const u8 = if (two) &.{ pr.X1.label(x1only, t.sched), pr.X1.label(t.x1.p, t.sched) } else &.{pr.X1.label(if (c.rows >= ex.prompt_rows) t.x1.p else .{}, t.sched)};
        try pr.bench(t.gpa, t.w, d.d, d.stream, d.set, arms, labels, c.rows, t.a.bench, t.sched);
        for (s.outputs()) |a| ok = try compare(t, mc, b.get(a.name).?, a, c.kind, id) and ok; // the same bytes after
    }
    return ok;
}

fn build(t: *Tool, ls: *x3.kernels.Launches, b: *const Bufs, c: chk.Case, s: chk.Spec, px: ex.Prompt) !void {
    const D = t.bound.dims;
    switch (c.kind) {
        .@"routed-bf16", .@"routed-f32", .@"routed-wts", .@"routed-fp16", .@"routed-hot" => {
            const tl = try t.facts.tiles();
            const sc: ex.Scratch = .{ .xg = b.at("xg"), .xu = b.at("xu"), .xd = b.at("xd"), .z = b.at("z"), .y = b.at("y"), .ids = b.at("ids"), .count = b.at("count"), .members = b.at("members"), .rows = c.rows, .slots = chk.slots, .experts = t.bound.count, .cfg_gu = tl[0], .cfg_d = tl[1], .counts = b.at("pg_counts"), .work = b.at("mma_work") };
            const sh: ex.Shared = if (t.dev) |d| d.sh else .{ .optin = 101376 };
            const wts: ?u64 = if (c.kind == .@"routed-wts") b.at("wts") else null;
            if (px.on()) return ex.routedX1(ls, b.at("x"), D, s.x_dtype == .bf16, b.at("pick"), wts, t.bound, sc, b.at("out"), c.rows, s.limit, s.act, true, sh, t.sched, px, t.x1.ord, t.x1.shared); // qualification_b3fba1da
            try ex.routedSched(ls, b.at("x"), D, s.x_dtype == .bf16, b.at("pick"), wts, t.bound, sc, b.at("out"), c.rows, s.limit, s.act, true, sh, t.sched);
        },
        .combine => ex.combine(ls, b.at("y"), b.at("wts"), b.at("out"), c.rows, D, chk.slots),
        .@"glm-mm" => {
            const g = t.glm_ex orelse return error.NoGlmBinding;
            const plan: gm.Plan = .{ .items = b.at("items"), .counts = b.at("counts"), .members = b.at("members"), .experts = t.bound.count + 1 };
            try gm.routed(ls, b.at("x"), D, b.at("pick"), plan, g, .{ .xg = b.at("xg"), .xu = b.at("xu"), .xd = b.at("xd"), .z = b.at("z"), .rows = c.rows, .slots = chk.slots }, b.at("y"), c.rows, s.limit);
        },
        .linear, .unpack, .dequant => {
            const p = switch (c.proj) {
                .gate => t.e.gate,
                .up => t.e.up,
                .down => t.e.down,
            };
            const x = c.expert;
            const words = t.base + p.trellis.off + p.offsets[x];
            if (c.kind == .dequant) return ex.dequant(ls, words, b.at("out"), p.k, p.n, p.k2s[x], t.e.cb);
            const xi: usize = if (x == chk.experts_checked[0]) 0 else 1;
            const l: lin.Layer = .{ .words = if (c.layout == .strips) t.strips[if (c.proj == .down) 1 else 0][xi] else words, .suh = t.base + p.suh.off + @as(u64, x) * p.k * 2, .svh = t.base + p.svh.off + @as(u64, x) * p.n * 2, .bias = null, .counters = b.at("counters"), .k = p.k, .n = p.n, .k2 = p.k2s[x], .cb = t.e.cb, .layout = c.layout };
            if (c.kind == .unpack) return lin.unpack(ls, l, b.at("w"));
            const yd: lin.DType = if (c.out == .bf16) .bf16 else .f32;
            try lin.call(ls, l, b.at("x"), .bf16, c.rows, b.at("y"), yd, b.at("xh"), b.at("z"));
        },
    }
}

test {
    _ = pr; // the prompt windows' bench parts (zig build test runs the tool's tests)
}

/// An output's bytes against python-0.6's file (or its sha256 when the reference kept hashes only).
pub fn compare(t: *Tool, mc: Obj, slot: *const Slot, a: chk.Arr, kind: anytype, id: []const u8) !bool {
    const m = findArr(mc.get("outputs").?, a.name).?;
    const got = try t.gpa.alloc(u8, a.bytes());
    defer t.gpa.free(got);
    try slot.buf.?.download(0, got);
    const file = str(m, "file");
    if (file.len == 0) {
        if (std.mem.eql(u8, &hex(got), str(m, "sha256"))) return true;
        t.fail("{t} {s}: {s} sha256 {s}, python-0.6's {s} (no reference file to locate the first byte)", .{ kind, id, a.name, &hex(got), str(m, "sha256") });
        return false;
    }
    const want = try refFile(t, file, 1 << 31);
    defer t.gpa.free(want);
    if (want.len != got.len) {
        t.fail("{t} {s}: {s} is {d} bytes, python-0.6's {d}", .{ kind, id, a.name, got.len, want.len });
        return false;
    }
    const first = std.mem.indexOfDiff(u8, got, want) orelse return true;
    var words: usize = 0;
    var i: usize = 0;
    while (i + 4 <= got.len) : (i += 4) {
        if (!std.mem.eql(u8, got[i..][0..4], want[i..][0..4])) words += 1;
    }
    const w0 = first / 4 * 4;
    t.fail("{t} {s}: {s} {t} first difference at byte {d} (word {x} here, {x} in python-0.6's), {d} of {d} words differ", .{ kind, id, a.name, a.dtype, first, got[w0..][0..@min(4, got.len - w0)], want[w0..][0..@min(4, want.len - w0)], words, got.len / 4 });
    return false;
}
