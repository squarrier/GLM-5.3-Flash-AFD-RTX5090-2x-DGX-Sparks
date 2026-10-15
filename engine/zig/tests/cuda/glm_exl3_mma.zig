//! tf-exl3-check's qualification_8724e161 part: TF_GLM_EXL3_MMA, bertholomus's grouped_mma over a work list (Albert Lee, BertholomusAI;
//! bertholomus/TensorFold deepseek-v41-tp2 90b68e0, Apache-2.0) against X1's grouped launches. --ident (G-ident): the
//! routed call's per-slot outputs y, X1 against MMA (and SEQ with MMA), byte for byte over the whole buffer, at R 64,
//! 65, 128, 594, 1,024, 1,900 and 2,048, on the reference's routing (routed-bf16 2,048's x and picks, the first R
//! rows), its hot routing (routed-hot 2,048's) and a skewed synthetic one; with --dry the launches are built only.
//! --bench N with the switch on (BENCH-MMA): the routed call at 64-2,048 rows and gate/up's and down's grouped launches
//! alone at 1,024 and 2,048, arms x1, seq (X1 + SEQ), pgroup, pgroup+seq, pgroup+mma and pgroup+seq+mma, round robin
//! run by run in one process; the arms' y compared after.

const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const x3 = glm.exl3;
const ex = x3.experts;
const mma = ex.mma;
const pg = x3.pgroup;
const Launches = x3.kernels.Launches;
const Sha256 = std.crypto.hash.sha2.Sha256;

const rows_ident = [_]u32{ 64, 65, 128, 594, 1024, 1900, 2048 };
const rows_bench = [_]u32{ 64, 128, 594, 1024, 2048 };
const slots: u32 = 9;
const sentinel: u8 = 0x5a; // y before each arm: the shared slot's rows, which no arm writes, must match too

/// The reference case `id`'s input `name`, checked against the manifest's sha256 (glm_exl3_lists.zig's rule).
fn refInput(t: anytype, id: []const u8, name: []const u8) ![]u8 {
    for (t.man.get("cases").?.array.items) |c| {
        const o = c.object;
        const cid = o.get("id") orelse continue;
        if (!std.mem.eql(u8, cid.string, id)) continue;
        for (o.get("inputs").?.array.items) |a| {
            if (!std.mem.eql(u8, a.object.get("name").?.string, name)) continue;
            const p = try std.fs.path.join(t.gpa, &.{ t.a.ref, a.object.get("file").?.string });
            defer t.gpa.free(p);
            const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, p, t.gpa, .limited(1 << 31));
            var h: [32]u8 = undefined;
            Sha256.hash(bytes, &h, .{});
            if (std.mem.eql(u8, &std.fmt.bytesToHex(h, .lower), a.object.get("sha256").?.string)) return bytes;
            t.gpa.free(bytes);
            return error.RefInputSha;
        }
    }
    return error.NoRefInput;
}

/// After the case table: --ident, then --bench's MMA arms when the switch is on.
pub fn after(t: anytype) !void {
    if (t.a.ident) {
        if (t.x1.p.on()) try ident(t) else t.fail("ident: --ident needs TF_GLM_EXL3_X1=1 (MMA replaces X1's grouped launches)", .{});
    }
    if (t.a.bench > 0 and t.x1.p.mma and t.dev != null) try bench(t);
}

/// An arm of the routed call: the env's X1 switches with PGROUP, SEQ and MMA as named.
const Arm = struct { label: []const u8, pgroup: bool, seq: bool, mma: bool };
const arms_ident = [_]Arm{ .{ .label = "x1", .pgroup = false, .seq = false, .mma = false }, .{ .label = "mma", .pgroup = true, .seq = false, .mma = true }, .{ .label = "seq+mma", .pgroup = true, .seq = true, .mma = true } };
const arms_bench = [_]Arm{ .{ .label = "x1", .pgroup = false, .seq = false, .mma = false }, .{ .label = "seq", .pgroup = false, .seq = true, .mma = false }, .{ .label = "pgroup", .pgroup = true, .seq = false, .mma = false }, .{ .label = "pgroup+seq", .pgroup = true, .seq = true, .mma = false }, .{ .label = "pgroup+mma", .pgroup = true, .seq = false, .mma = true }, .{ .label = "pgroup+seq+mma", .pgroup = true, .seq = true, .mma = true } };

/// One window's device buffers (R rows): x, pick, the scratch, PGROUP's counts, MMA's work list and X1's order list.
const Window = struct {
    R: u32,
    bufs: [13]?cuda.DeviceBuffer = @splat(null),
    y_bytes: usize = 0,

    const names = [_][]const u8{ "x", "pick", "xg", "xu", "xd", "z", "y", "ids", "count", "members", "counts", "work", "order" };

    fn init(t: anytype, R: u32, x: []const u8, pick: []const u8) !Window {
        const d = t.dev.?;
        const E = t.bound.count;
        const tl = try t.facts.tiles();
        const sz = ex.Scratch.bytes(R, slots, t.bound, tl[0], tl[1]);
        const lens = [_]usize{ x.len, pick.len, sz.xg, sz.xg, sz.xd, sz.z, sz.y, sz.ids, 4, sz.members, 4 * @as(usize, E), 4 * @as(usize, mma.workInts(R, slots, E)), 4 * @as(usize, ex.orderSlots(R, slots, E)) };
        var w: Window = .{ .R = R, .y_bytes = sz.y };
        errdefer w.deinit();
        for (lens, 0..) |len, i| w.bufs[i] = try cuda.DeviceBuffer.alloc(d.d, len);
        try w.bufs[0].?.upload(0, x);
        try w.bufs[1].?.upload(0, pick);
        return w;
    }

    fn deinit(w: *Window) void {
        for (&w.bufs) |*b| if (b.*) |*x| {
            x.free();
            b.* = null;
        };
    }

    fn at(w: *const Window, i: usize) u64 {
        return w.bufs[i].?.ptr;
    }

    /// The routed call of arm `a` (y per slot, no weights), as the halves make it (routedX1, do_group).
    fn call(w: *const Window, t: anytype, a: Arm, ls: *Launches) !void {
        const E = t.bound.count;
        const tl = try t.facts.tiles();
        var p = t.x1.p;
        p.pgroup = a.pgroup;
        p.seq = a.seq;
        p.mma = a.mma;
        const s: ex.Scratch = .{ .xg = w.at(2), .xu = w.at(3), .xd = w.at(4), .z = w.at(5), .y = w.at(6), .ids = w.at(7), .count = w.at(8), .members = w.at(9), .rows = w.R, .slots = slots, .experts = E, .cfg_gu = tl[0], .cfg_d = tl[1], .counts = if (a.pgroup) w.at(10) else 0, .work = if (a.mma) w.at(11) else 0 };
        const suh: ?u64 = if (t.suh_shared) t.bound.suh_g else null; // python's shared_suh: expert 0's row, the same bytes
        try ex.routedX1(ls, w.at(0), t.bound.dims, true, w.at(1), null, t.bound, s, 0, w.R, t.facts.limit, .bf16, true, t.dev.?.sh, t.sched, p, w.at(12), suh);
    }

    /// Arm `a` once from a fresh y (sentinel bytes), its y's sha256 hex.
    fn run(w: *const Window, t: anytype, a: Arm, y: []u8) ![64]u8 {
        const d = t.dev.?;
        var ls: Launches = .{};
        try w.call(t, a, &ls);
        try w.bufs[6].?.fill8(sentinel, null);
        try d.ctx.synchronize();
        try d.set.run(d.stream, &ls);
        try d.stream.synchronize();
        try w.bufs[6].?.download(0, y);
        var h: [32]u8 = undefined;
        Sha256.hash(y, &h, .{});
        return std.fmt.bytesToHex(h, .lower);
    }
};

/// The launches of an arm with --dry's stand-in addresses: built, and MMA's arms holding exl3_mma's kernels.
fn dryCall(t: anytype, a: Arm, R: u32) !bool {
    const E = t.bound.count;
    const tl = try t.facts.tiles();
    var p = t.x1.p;
    p.pgroup = a.pgroup;
    p.seq = a.seq;
    p.mma = a.mma;
    const s: ex.Scratch = .{ .xg = 2 << 40, .xu = 3 << 40, .xd = 4 << 40, .z = 5 << 40, .y = 6 << 40, .ids = 7 << 40, .count = 8 << 40, .members = 9 << 40, .rows = R, .slots = slots, .experts = E, .cfg_gu = tl[0], .cfg_d = tl[1], .counts = if (a.pgroup) 10 << 40 else 0, .work = if (a.mma) 11 << 40 else 0 };
    var ls: Launches = .{};
    try ex.routedX1(&ls, 1 << 40, t.bound.dims, true, 1 << 41, null, t.bound, s, 0, R, t.facts.limit, .bf16, true, .{ .optin = 101376 }, t.sched, p, 12 << 40, if (t.suh_shared) t.bound.suh_g else null);
    var n: usize = 0;
    for (ls.slice()) |l| n += @intFromBool(l.image == .exl3_mma_cb1);
    return n == @as(usize, if (!a.mma) 0 else if (a.seq) 2 else 3);
}

/// G-ident: every routing x R, each MMA arm's y = X1's y, word for word over the whole buffer.
fn ident(t: anytype) !void {
    const E = t.bound.count;
    var pass: u32 = 0;
    var total: u32 = 0;
    var src: [2][2][]u8 = undefined; // [ref, hot] x [x, pick]
    var got: usize = 0;
    defer for (src[0..got]) |s| {
        t.gpa.free(s[0]);
        t.gpa.free(s[1]);
    };
    for ([_][]const u8{ "routed-bf16-2048", "routed-hot-2048" }, &src) |id, *s| {
        s[0] = try refInput(t, id, "x");
        s[1] = refInput(t, id, "pick") catch |e| {
            t.gpa.free(s[0]);
            return e;
        };
        got += 1;
    }
    const D = t.bound.dims;
    for ([_][]const u8{ "ref", "hot", "skewed" }, 0..) |name, k| for (rows_ident) |R| {
        total += 1;
        if (t.dev == null) {
            var ok = true;
            for (arms_ident) |a| ok = (try dryCall(t, a, R)) and ok;
            if (ok) pass += 1 else t.fail("dry ident {s} r{d}: an arm's launches are not MMA's", .{ name, R });
            continue;
        }
        const x = src[if (k == 1) 1 else 0][0][0 .. 2 * @as(usize, R) * D];
        var pick: []const u8 = src[if (k == 1) 1 else 0][1][0 .. 4 * @as(usize, R) * slots];
        var syn: ?[]i32 = null;
        defer if (syn) |s| t.gpa.free(s);
        if (k == 2) {
            syn = try pg.synthetic(t.gpa, .skewed, R, @intCast(E), 0x3b + R);
            pick = std.mem.sliceAsBytes(syn.?);
        }
        var w = try Window.init(t, R, x, pick);
        defer w.deinit();
        const y0 = try t.gpa.alloc(u8, w.y_bytes);
        defer t.gpa.free(y0);
        const y1 = try t.gpa.alloc(u8, w.y_bytes);
        defer t.gpa.free(y1);
        const h0 = try w.run(t, arms_ident[0], y0);
        var ok = true;
        for (arms_ident[1..]) |a| {
            const h = try w.run(t, a, y1);
            if (std.mem.indexOfDiff(u8, y0, y1)) |i| {
                ok = false;
                t.fail("ident {s} r{d}: {s}'s y differs from x1's at byte {d} of {d} (sha256 {s} against {s})", .{ name, R, a.label, i, y0.len, h[0..16], h0[0..16] });
            }
        }
        if (!ok) continue;
        pass += 1;
        t.out("PASS ident {s} r{d}: y {d} B, x1 = mma = seq+mma byte for byte (sha256 {s})", .{ name, R, y0.len, h0[0..16] });
    };
    t.out("RESULT ident: {d} of {d} {s}", .{ pass, total, if (t.dev == null) "built (dry: the launches, no GPU)" else "byte-equal on the GPU, X1 against MMA and SEQ with MMA" });
}

fn median(us: []f64) [3]f64 {
    std.mem.sort(f64, us, {}, std.sort.asc(f64));
    const n = us.len;
    return .{ us[n / 2], us[n / 10], us[n - 1 - n / 10] };
}

/// gate/up's (first) or down's (second) grouped launch of a routed call, alone (PDL off): X1's, SEQ's or MMA's image.
fn grouped(ls: *const Launches, second: bool) !Launches {
    var n: usize = 0;
    for (ls.slice()) |l| {
        const g = switch (l.image) {
            .exl3_x1_grouped_cb1, .exl3_x1s_grouped_cb1 => true,
            .exl3_mma_cb1 => !std.mem.eql(u8, l.symbol, x3.kernels.sym.work_list),
            else => false,
        };
        if (!g) continue;
        if (n == @intFromBool(second)) {
            var out: Launches = .{};
            var x = l;
            x.cfg.pdl = false;
            out.push(x);
            return out;
        }
        n += 1;
    }
    return error.NoGroupedLaunches;
}

/// BENCH-MMA: each part, the arms round robin run by run (N runs each), between chained events; medians, ratios to x1.
/// Every R takes the first R rows of routed-bf16 2,048's x and picks (the reference has no 128- or 594-row case).
fn bench(t: anytype) !void {
    const d = t.dev.?;
    const n = t.a.bench;
    const na = arms_bench.len;
    const x_all = try refInput(t, "routed-bf16-2048", "x");
    defer t.gpa.free(x_all);
    const p_all = try refInput(t, "routed-bf16-2048", "pick");
    defer t.gpa.free(p_all);
    for (rows_bench) |R| {
        const xh = x_all[0 .. 2 * @as(usize, R) * t.bound.dims];
        const ph = p_all[0 .. 4 * @as(usize, R) * slots];
        var w = try Window.init(t, R, xh, ph);
        defer w.deinit();
        var calls: [na]Launches = undefined;
        for (arms_bench, &calls) |a, *ls| {
            ls.* = .{};
            try w.call(t, a, ls);
        }
        try d.ctx.synchronize();
        for (0..3) |_| for (calls) |ls| try d.set.run(d.stream, &ls); // warm-up
        try d.stream.synchronize();
        const parts: []const []const u8 = if (R >= 1024) &.{ "routed", "gate_up", "down" } else &.{"routed"};
        const ev = try t.gpa.alloc(cuda.Event, na * n + 1);
        defer t.gpa.free(ev);
        var made: usize = 0;
        defer for (ev[0..made]) |*e| e.deinit();
        while (made < ev.len) : (made += 1) ev[made] = try cuda.Event.init(d.d, true);
        const us = try t.gpa.alloc(f64, na * n);
        defer t.gpa.free(us);
        const one = try t.gpa.alloc(f64, n);
        defer t.gpa.free(one);
        for (parts, 0..) |part, pi| {
            var lists: [na]Launches = undefined;
            for (calls, &lists) |c, *l| l.* = if (pi == 0) c else try grouped(&c, pi == 2);
            try ev[0].record(d.stream);
            for (0..na * n) |r| {
                try d.set.run(d.stream, &lists[r % na]);
                try ev[r + 1].record(d.stream);
            }
            try ev[na * n].synchronize();
            for (us, 0..) |*u, r| u.* = @as(f64, try cuda.Event.elapsedMs(ev[r], ev[r + 1])) * 1000;
            var med0: f64 = 0;
            for (arms_bench, 0..) |a, ai| {
                for (one, 0..) |*o, i| o.* = us[i * na + ai];
                const m = median(one);
                if (ai == 0) med0 = m[0];
                t.out("BENCH-MMA r{d} {s} {s}: median {d:.1} us, p10 {d:.1}, p90 {d:.1}, /x1 {d:.4} ({d} runs, round robin of {d}; l2d {d}, pdl {d}, rowrot {s})", .{ R, part, a.label, m[0], m[1], m[2], m[0] / med0, n, na, t.sched.l2d, @intFromBool(t.sched.pdl), if (t.suh_shared) "on" else "off" });
            }
        }
        const y0 = try t.gpa.alloc(u8, w.y_bytes);
        defer t.gpa.free(y0);
        const y1 = try t.gpa.alloc(u8, w.y_bytes);
        defer t.gpa.free(y1);
        const h0 = try w.run(t, arms_bench[0], y0);
        var same = true;
        for (arms_bench[1..]) |a| {
            _ = try w.run(t, a, y1);
            if (std.mem.indexOfDiff(u8, y0, y1)) |i| {
                same = false;
                t.fail("bench-mma r{d}: {s}'s y differs from x1's at byte {d}", .{ R, a.label, i });
            }
        }
        if (same) t.out("PASS bench-mma r{d}: the {d} arms' y byte-equal after the runs (sha256 {s})", .{ R, na, h0[0..16] });
    }
}
