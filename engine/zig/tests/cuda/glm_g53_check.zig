//! `tf-g53-check CKPT REF [--dry] [--bench N]` (qualification_1b9bf461's G-bits): the Zig g53 + KT3R port against python TF-AFD 2.15's
//! own kernel on one MoE layer's TP2 half. REF is tools/zig/g53_check.py's folder: python 2.15's g53_experts.routed at
//! tuned_cfg(R, KT3, tier 1,536) on seeded rows and routes, at the R list of 2.15's BENCHMARKS §Quality. This tool
//! plans the manifest's rank, reads only layer L's experts group (as tf-exl3-check does), runs cuda_g53's launches on
//! the same inputs and compares every output byte; each window's configuration and the layer's scope and shared-suh
//! test must be python's too. --dry: all but CUDA (a header-only checkpoint copy serves). PASS/FAIL lines; exit 1 on
//! any FAIL. --bench N times each window's call N times (BENCH lines, and BENCH-AB against python's --bench).
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const x3 = glm.exl3;
const ae = glm.afd_expert;
const g53 = ae.g53;
const Sha256 = std.crypto.hash.sha2.Sha256;

const usage =
    \\usage: tf-g53-check CKPT REF [--dry] [--bench N]
    \\  CKPT     Mia's EXL3 checkpoint folder (078455ff; --dry: a header-only copy serves)
    \\  REF      tools/zig/g53_check.py's folder: manifest.json, in/, out/
    \\  --dry    everything but CUDA: plan, read, scope, shared-suh test, each window's configuration and launches
    \\  --bench N  each window's call N times after its check: median, p10, p90 in us (and against python's)
    \\
;

const Arr = struct { file: []const u8, dtype: []const u8, shape: []const u64, sha256: []const u8 };
const OutJ = struct { file: []const u8, sha256: []const u8 };
const CaseJ = struct { rows: u32, cfg: g53.Cfg, out: ?OutJ = null, fault: ?i64 = null, bench_us: ?[3]f64 = null };
const Manifest = struct {
    layer: u32,
    rank: u32,
    dry: u32 = 0,
    limit: ?f64 = null,
    shared_suh: ?bool = null,
    inputs: struct { x: Arr, pick: Arr, wts: Arr },
    cases: []const CaseJ,
};

const Tool = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    w: *std.Io.Writer,
    ckpt: []const u8 = "",
    ref: []const u8 = "",
    dry: bool = false,
    bench: u32 = 0,
    fails: u32 = 0,

    fn out(t: *Tool, comptime fmt: []const u8, args: anytype) void {
        t.w.print(fmt ++ "\n", args) catch {};
        t.w.flush() catch {};
    }

    fn fail(t: *Tool, comptime fmt: []const u8, args: anytype) void {
        t.fails += 1;
        t.out("FAIL " ++ fmt, args);
    }

    fn read(t: *Tool, dir: []const u8, rel: []const u8, most: usize) ![]u8 {
        const p = try std.fs.path.join(t.gpa, &.{ dir, rel });
        defer t.gpa.free(p);
        return std.Io.Dir.cwd().readFileAlloc(t.io, p, t.gpa, .limited(most));
    }
};

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var buf: [64 << 10]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buf);
    var t: Tool = .{ .gpa = init.gpa, .io = init.io, .w = &stdout.interface };
    var pos: usize = 0;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (std.mem.eql(u8, k, "--dry")) {
            t.dry = true;
        } else if (std.mem.eql(u8, k, "--bench") and i + 1 < argv.len) {
            i += 1;
            t.bench = std.fmt.parseInt(u32, argv[i], 10) catch 0;
            if (t.bench == 0) pos = 9;
        } else if (pos < 2 and k.len > 0 and k[0] != '-') {
            if (pos == 0) t.ckpt = k else t.ref = k;
            pos += 1;
        } else pos = 9;
    }
    if (pos != 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    run(&t) catch |e| {
        t.fail("tf-g53-check: {t} (no verdict for the windows after this)", .{e});
        return 1;
    };
    return if (t.fails == 0) 0 else 1;
}

fn hex(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}

/// Layer L's experts group alone, read as the loader reads it (tf-exl3-check's readGroup).
fn readGroup(t: *Tool, plan: *const x3.split.Plan, ix: *glm.index.Index, g: u32, group: []u8) !void {
    var copies: std.ArrayList(glm.split.Copy) = .empty;
    defer copies.deinit(t.gpa);
    for (plan.copies.items) |cp| if (cp.g == g) try copies.append(t.gpa, cp);
    var s = try glm.read.schedule(t.gpa, copies.items, .{});
    defer s.deinit();
    const views = try t.gpa.alloc([]u8, plan.groups.items.len);
    defer t.gpa.free(views);
    for (views) |*v| v.* = &.{};
    views[g] = group;
    var sink: glm.read.HostSink = .{ .groups = views };
    const st = try glm.read.run(t.gpa, ix.shards, &s, sink.sink(), .{});
    t.out("RESULT read: experts group {d}: {d} copies, {d} B kept of {d} B read in {d:.2} s", .{ g, copies.items.len, st.kept, st.bytes, @as(f64, @floatFromInt(st.ns)) / 1e9 });
}

const Dev = struct { d: cuda.Driver, ctx: cuda.Context, stream: cuda.Stream, set: g53.Set };

fn run(t: *Tool) !void {
    const gpa = t.gpa;
    const mtext = try t.read(t.ref, "manifest.json", 1 << 24);
    defer gpa.free(mtext);
    const parsed = try std.json.parseFromSlice(Manifest, gpa, mtext, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    if (m.dry == 1 and !t.dry) return error.DryManifestHasNoOutputs;
    const cjson = try t.read(t.ckpt, "config.json", 1 << 22);
    defer gpa.free(cjson);
    var c = try glm.config.parse(gpa, cjson);
    try glm.config.subset(&c, @min(m.layer + 1, c.layers));
    if (m.limit) |lim| if (@as(f32, @floatCast(lim)) != c.swiglu_limit) t.fail("limit: python's {d}, config.json's {d}", .{ lim, c.swiglu_limit });
    var ix = try glm.index.open(gpa, t.ckpt);
    defer ix.deinit();
    var plan = try x3.split.plan(gpa, &c, &ix.names, m.rank, 2);
    defer plan.deinit();
    const e = plan.experts[m.layer] orelse return error.NoExpertsInLayer;
    const gi = e.gate.trellis.g;
    const group = try gpa.alloc(u8, plan.groups.items[gi]);
    defer gpa.free(group);
    @memset(group, 0);
    try readGroup(t, &plan, &ix, gi, group);

    var dev: ?*Dev = null;
    defer if (dev) |d| {
        d.set.unload();
        d.stream.deinit();
        d.ctx.deinit();
        d.d.close();
        gpa.destroy(d);
    };
    if (!t.dry) {
        const d = try gpa.create(Dev);
        errdefer gpa.destroy(d);
        d.d = try cuda.Driver.open();
        errdefer d.d.close();
        d.ctx = try cuda.Context.init(&d.d, 0);
        errdefer d.ctx.deinit();
        d.stream = try cuda.Stream.init(&d.d, true);
        errdefer d.stream.deinit();
        const sms: u32 = @intCast(try d.ctx.attribute(.multiprocessor_count));
        d.set = try g53.Set.load(&d.d, cuda.kernels.g53_rank, sms);
        dev = d;
        var name: [256]u8 = undefined;
        t.out("RESULT device: {s}, sm_{d}, {d} SMs", .{ try d.ctx.name(&name), try d.ctx.capability(), sms });
    }
    var bufs: [10]?cuda.DeviceBuffer = @splat(null);
    defer for (&bufs) |*b| if (b.*) |*x| x.free();
    var next: u64 = 1 << 40; // --dry's stand-in addresses
    const alloc = struct {
        fn f(dv: ?*Dev, slot: *?cuda.DeviceBuffer, nx: *u64, n: usize) !u64 {
            if (dv) |d| {
                slot.* = try cuda.DeviceBuffer.alloc(&d.d, n);
                return slot.*.?.ptr;
            }
            const p = nx.*;
            nx.* += std.mem.alignForward(u64, n, 1 << 20);
            return p;
        }
    }.f;

    // the group: its tables, the layer bound at its address, scope and shared-suh test on the host bytes
    const base = try alloc(dev, &bufs[0], &next, group.len);
    const E = e.count;
    plan.tables(m.layer, base, group[e.tables.off..][0 .. 36 * E]);
    const layer = plan.bind(m.layer, base).?;
    if (g53.layerReady(layer, group[e.tables.off..][0 .. 36 * E])) |why| return t.fail("scope: layer {d} rank {d}: {s} (python's layer_ready takes it)", .{ m.layer, m.rank, why });
    const K = layer.dims;
    const sg = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, group[e.gate.suh.off..][0 .. @as(usize, E) * K * 2]));
    const su = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, group[e.up.suh.off..][0 .. @as(usize, E) * K * 2]));
    const hs = try gpa.alloc(u16, 2 * @as(usize, E) * K);
    defer gpa.free(hs);
    for (sg, 0..) |v, j| hs[j] = v;
    for (su, 0..) |v, j| hs[E * K + j] = v;
    const shared = ae.sharedSuh(hs[0 .. E * K], hs[E * K ..], K);
    if (m.shared_suh) |want| if (want != shared) t.fail("shared suh: python's {}, here {}", .{ want, shared });
    t.out("RESULT layer {d} rank {d}/2: in scope, shared gate/up suh {} (rowrot {s})", .{ m.layer, m.rank, shared, if (shared) "on" else "off" });

    // the inputs (python's bytes, their sha256 checked), out, the plan block, the large buffers, the shared suh row
    var rmax: u32 = 0;
    var cmax: g53.Cfg = try g53.defaultCfg(1);
    for (m.cases) |cs| {
        rmax = @max(rmax, cs.rows);
        cmax.sk = @max(cmax.sk, cs.cfg.sk);
        cmax.skd = @max(cmax.skd, cs.cfg.skd);
    }
    if (rmax == 0 or rmax > g53.MAX_ROWS) return error.Rows;
    if (m.inputs.x.shape.len != 2 or m.inputs.x.shape[0] < rmax or m.inputs.x.shape[1] != g53.H) return error.InputShape;
    var at: [3]u64 = undefined;
    const ins = [_]Arr{ m.inputs.x, m.inputs.pick, m.inputs.wts };
    var S: u32 = 0;
    for (ins, 0..) |a, j| {
        const bytes = try t.read(t.ref, a.file, 1 << 28);
        defer gpa.free(bytes);
        if (!std.mem.eql(u8, &hex(bytes), a.sha256)) return t.fail("input {s}: not the bytes the manifest lists", .{a.file});
        if (j > 0) S = @intCast(a.shape[1]);
        at[j] = try alloc(dev, &bufs[1 + j], &next, bytes.len);
        if (dev != null) try bufs[1 + j].?.upload(0, bytes);
    }
    const out = try alloc(dev, &bufs[4], &next, @as(usize, rmax) * g53.H * 4);
    const pb = try alloc(dev, &bufs[5], &next, g53.Plan.bytes(rmax));
    if (dev != null) try bufs[5].?.fill8(0, null); // python's torch.zeros
    cmax.rot = 2;
    const need = g53.bigNeeds(rmax, cmax);
    const bg: g53.Big = .{ .z = try alloc(dev, &bufs[6], &next, need.z), .xd = try alloc(dev, &bufs[7], &next, need.xd), .zd = try alloc(dev, &bufs[8], &next, need.zd), .xh = try alloc(dev, &bufs[9], &next, need.xh), .z_bytes = need.z, .xd_bytes = need.xd, .zd_bytes = need.zd, .xh_bytes = need.xh };
    var suh_row: ?u64 = null;
    var srow: ?cuda.DeviceBuffer = null;
    defer if (srow) |*b| b.free();
    if (shared) {
        if (dev) |d| {
            srow = try cuda.DeviceBuffer.alloc(&d.d, K * 2);
            try srow.?.upload(0, std.mem.sliceAsBytes(hs[0..K]));
            suh_row = srow.?.ptr;
        } else suh_row = next;
    }
    if (dev != null) {
        try bufs[0].?.upload(0, group);
        try dev.?.ctx.synchronize();
    }
    const ready = [_]bool{true};
    const suhs = [_]?u64{suh_row};
    const node: g53.Node = .{ .set = if (dev) |d| &d.set else undefined, .plan = g53.Plan.place(rmax, pb), .kt3r = true, .max_rows = rmax, .ready = &ready, .suh = &suhs };

    var pass: u32 = 0;
    for (m.cases) |cs| {
        const R = cs.rows;
        const cfg = try g53.tunedCfg(R, true);
        if (!std.meta.eql(cfg, cs.cfg)) {
            t.fail("g53 r{d}: configuration {any}, python's {any}", .{ R, cfg, cs.cfg });
            continue;
        }
        var ls: g53.Launches = .{};
        node.launches(&ls, layer, 0, at[0], g53.H, at[1], at[2], S, out, R, c.swiglu_limit, bg) catch |err| {
            t.fail("g53 r{d}: the binding refused the window ({t})", .{ R, err });
            continue;
        };
        const d = dev orelse {
            t.out("DRY g53 r{d}: mt {d} gw {d} nt {d} l2 {d} rot {d}: {d} launches", .{ R, cfg.mt, cfg.gw, cfg.nt, cfg.l2, if (suh_row != null) cfg.rot else 1, ls.len });
            for (ls.slice()) |l| t.out("DRY g53 r{d}:   {s} block {d} grid {d} items {d}", .{ R, l.symbol, l.cfg.block.x, l.cfg.grid.x, l.items });
            pass += 1;
            continue;
        };
        try bufs[4].?.fill8(0xff, null); // NaN in every word: an unwritten one differs from python's
        try d.set.run(&d.d, d.stream, &ls);
        try d.stream.synchronize();
        var meta: [4]i32 = undefined;
        try bufs[5].?.download(ls.meta - pb, std.mem.asBytes(&meta));
        if (meta[1] != 0 or (cs.fault orelse 0) != 0) t.fail("g53 r{d}: fault bits {d} here, python's {d}", .{ R, meta[1], cs.fault orelse -1 });
        const got = try gpa.alloc(u8, @as(usize, R) * g53.H * 4);
        defer gpa.free(got);
        try bufs[4].?.download(0, got);
        const o = cs.out orelse return error.NoReference;
        const want = try t.read(t.ref, o.file, 1 << 28);
        defer gpa.free(want);
        if (want.len != got.len or !std.mem.eql(u8, &hex(want), o.sha256)) {
            t.fail("g53 r{d}: python's out file is not the manifest's", .{R});
            continue;
        }
        if (std.mem.eql(u8, got, want)) {
            pass += 1;
            t.out("EQUAL g53 r{d}: out [{d}, 4096] fp32 byte-equal to python 2.15's (sha256 {s}); mt {d} gw {d} nt {d} l2 {d}", .{ R, R, o.sha256[0..16], cfg.mt, cfg.gw, cfg.nt, cfg.l2 });
        } else {
            const gw_ = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, got));
            const ww = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, want));
            var first: usize = 0;
            var n: usize = 0;
            for (gw_, ww, 0..) |a, b, j| if (a != b) {
                if (n == 0) first = j;
                n += 1;
            };
            t.fail("g53 r{d}: {d} of {d} words differ, the first at row {d} column {d} ({x} here, {x} python's)", .{ R, n, gw_.len, first / g53.H, first % g53.H, gw_[first], ww[first] });
        }
        if (t.bench > 0) try bench(t, d, &ls, R, cs.bench_us);
    }
    const kind = if (t.dry) "DRY" else "byte-equal to python 2.15's";
    if (t.fails == 0) t.out("PASS g53: {d} of {d} windows {s} (layer {d} rank {d}, KT3R from {d} rows)", .{ pass, m.cases.len, kind, m.layer, m.rank, g53.kt3_tier }) else t.out("FAIL g53: {d} of {d} windows {s}, {d} failures", .{ pass, m.cases.len, kind, t.fails });
}

/// --bench N: the window's call N times back to back, an event between runs (the meta clear included, as served).
fn bench(t: *Tool, d: *Dev, ls: *const g53.Launches, R: u32, py: ?[3]f64) !void {
    const n = t.bench;
    const ev = try t.gpa.alloc(cuda.Event, n + 1);
    defer t.gpa.free(ev);
    var made: usize = 0;
    defer for (ev[0..made]) |*x| x.deinit();
    while (made < ev.len) : (made += 1) ev[made] = try cuda.Event.init(&d.d, true);
    const us = try t.gpa.alloc(f64, n);
    defer t.gpa.free(us);
    for (0..n) |j| {
        try ev[j].record(d.stream);
        try d.set.run(&d.d, d.stream, ls);
    }
    try ev[n].record(d.stream);
    try ev[n].synchronize();
    for (us, 0..) |*u, j| u.* = @as(f64, try cuda.Event.elapsedMs(ev[j], ev[j + 1])) * 1000;
    std.mem.sort(f64, us, {}, std.sort.asc(f64));
    t.out("BENCH g53 r{d}: median {d:.1} us, p10 {d:.1}, p90 {d:.1} ({d} runs)", .{ R, us[n / 2], us[n / 10], us[n - 1 - n / 10], n });
    if (py) |p| t.out("BENCH-AB g53 r{d}: zig / python median {d:.3} ({d:.1} / {d:.1} us)", .{ R, us[n / 2] / p[0], us[n / 2], p[0] });
}

test {
    _ = &main;
}
