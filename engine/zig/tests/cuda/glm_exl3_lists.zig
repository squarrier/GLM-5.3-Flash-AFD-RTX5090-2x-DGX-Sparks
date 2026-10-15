//! tf-exl3-check's qualification_3e1f1105 part: TF_GLM_EXL3_PGROUP, the parallel grouping of bertholomus (Albert Lee, BertholomusAI;
//! bertholomus/TensorFold glm-dsa-tp4 dbe0773, Apache-2.0) against the one-block grouping of Ash Hart's TensorFold.
//! --lists (G-lists): `group`'s and `groupPar`'s uids, ucount and members byte for byte, and both against the CPU rule
//! (cuda_exl3_pgroup.zig), at R 64, 65, 512, 1,024, 1,900, 2,048 and 2,049, on the reference's routing (routed-bf16
//! 2,048's picks, the first R rows; 2,049 adds its row 0 again), its hot routing (routed-hot 2,048's) and a skewed
//! synthetic one; with --dry the launches run on host memory. --bench N with the switch on: at 1,024 and 2,048 rows,
//! the grouping alone and the routed call (X1 as the env sets it), off and on alternating run by run (BENCH-PG lines),
//! and the routed outputs of both arms compared after.

const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const x3 = glm.exl3;
const ex = x3.experts;
const pg = x3.pgroup;
const Launches = x3.kernels.Launches;
const Sha256 = std.crypto.hash.sha2.Sha256;

const rows_lists = [_]u32{ 64, 65, 512, 1024, 1900, 2048, 2049 };
const rows_bench = [_]u32{ 1024, 2048 };
const slots: u32 = 9;
const sentinel: u8 = 0x5a; // every word of every list buffer before the launch: untouched words must match too

/// The reference case `id`'s input `name`, checked against the manifest's sha256.
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

fn ints(gpa: std.mem.Allocator, bytes: []const u8) ![]i32 {
    const out = try gpa.alloc(i32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * 4]);
    return out;
}

/// python's `shared_suh` test (the halves' sharedSuh) on the layer's gate and up suh in the host group: X1's rowrot.
pub fn sharedSuh(t: anytype, group: []const u8) !bool {
    const n = @as(usize, t.bound.count) * t.bound.dims;
    const h = try t.gpa.alloc(u16, 2 * n);
    defer t.gpa.free(h);
    @memcpy(std.mem.sliceAsBytes(h[0..n]), group[@intCast(t.bound.suh_g - t.base)..][0 .. 2 * n]);
    @memcpy(std.mem.sliceAsBytes(h[n..]), group[@intCast(t.bound.suh_u - t.base)..][0 .. 2 * n]);
    return glm.afd_expert.sharedSuh(h[0..n], h[n..], t.bound.dims);
}

/// After the case table: --lists, then --bench's PGROUP arms when the switch is on.
pub fn after(t: anytype) !void {
    if (t.a.lists) try lists(t);
    if (t.a.bench > 0 and t.x1.p.pgroup and t.dev != null) try bench(t);
}

const Out = struct {
    ids: []i32,
    count: []i32,
    members: []i32,

    fn init(gpa: std.mem.Allocator, R: usize, E: usize) !Out {
        const maxu = @min(R * slots, E);
        const o: Out = .{ .ids = try gpa.alloc(i32, maxu), .count = try gpa.alloc(i32, 1), .members = try gpa.alloc(i32, maxu * R) };
        for ([_][]i32{ o.ids, o.count, o.members }) |w| @memset(std.mem.sliceAsBytes(w), sentinel);
        return o;
    }

    fn deinit(o: Out, gpa: std.mem.Allocator) void {
        gpa.free(o.ids);
        gpa.free(o.count);
        gpa.free(o.members);
    }

    /// The first differing word of the three lists, or null.
    fn diff(a: Out, b: Out) ?struct { []const u8, usize } {
        for ([_][]const u8{ "uids", "ucount", "members" }, [_][]i32{ a.ids, a.count, a.members }, [_][]i32{ b.ids, b.count, b.members }) |name, x, y| {
            if (std.mem.indexOfDiff(i32, x, y)) |i| return .{ name, i };
        }
        return null;
    }
};

/// One grouping (group or groupPar) of `pick` into fresh sentinel-filled lists: on the GPU, or with --dry on host
/// memory through the CPU's run of each launch (cuda_exl3_pgroup.zig).
fn grouped(t: anytype, pick: []i32, R: u32, par: bool) !Out {
    const E = t.bound.count;
    const out = try Out.init(t.gpa, R, E);
    errdefer out.deinit(t.gpa);
    const counts = try t.gpa.alloc(i32, E);
    defer t.gpa.free(counts);
    var ls: Launches = .{};
    const at = [_]u64{ 1 << 40, 2 << 40, 3 << 40, 4 << 40, 5 << 40 }; // --dry's stand-ins
    const d = t.dev orelse {
        if (par) try ex.groupPar(&ls, at[0], at[4], at[1], at[2], at[3], R, slots, E, R) else try ex.group(&ls, at[0], at[1], at[2], at[3], R, slots, E, R, .{ .optin = 101376 });
        const m: pg.Mem = .{ .at = &at, .words = &.{ pick, out.ids, out.count, out.members, counts } };
        for (ls.slice()) |l| try pg.run(l, m);
        return out;
    };
    var bufs: [5]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for ([_][]i32{ pick, out.ids, out.count, out.members, counts }) |w| {
        bufs[made] = try cuda.DeviceBuffer.alloc(d.d, 4 * w.len);
        made += 1;
    }
    try bufs[0].upload(0, std.mem.sliceAsBytes(pick));
    for (bufs[1..]) |b| try b.fill8(sentinel, null);
    if (par) try ex.groupPar(&ls, bufs[0].ptr, bufs[4].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, R, slots, E, R) else try ex.group(&ls, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, R, slots, E, R, d.sh);
    try d.ctx.synchronize();
    try d.set.run(d.stream, &ls);
    try d.stream.synchronize();
    for (ls.slice()) |l| if (l.max_dynamic) |v| {
        d.sh.max_dynamic = v; // the raised limit stays raised, as the case table keeps it
    };
    for ([_][]i32{ out.ids, out.count, out.members }, bufs[1..4]) |w, b| try b.download(0, std.mem.sliceAsBytes(w));
    return out;
}

/// G-lists: every routing x R, group = groupPar = the rule, word for word over the whole buffers.
fn lists(t: anytype) !void {
    const E = t.bound.count;
    const src = [_][]const u8{ "routed-bf16-2048", "routed-hot-2048" };
    var refs: [2][]i32 = undefined;
    for (src, &refs) |id, *r| {
        const b = try refInput(t, id, "pick");
        defer t.gpa.free(b);
        r.* = try ints(t.gpa, b);
    }
    defer for (refs) |r| t.gpa.free(r);
    var pass: u32 = 0;
    var total: u32 = 0;
    for ([_][]const u8{ "ref", "hot", "skewed" }, 0..) |name, k| for (rows_lists) |R| {
        const pick = try t.gpa.alloc(i32, R * slots);
        defer t.gpa.free(pick);
        if (k < 2) {
            const r = refs[k];
            const n = @min(pick.len, r.len);
            @memcpy(pick[0..n], r[0..n]);
            if (n < pick.len) @memcpy(pick[n..], r[0 .. pick.len - n]); // 2,049: its row 0 again
        } else {
            const s = try pg.synthetic(t.gpa, .skewed, R, @intCast(E), 0x2a + R);
            defer t.gpa.free(s);
            @memcpy(pick, s);
        }
        const want = try Out.init(t.gpa, R, E);
        defer want.deinit(t.gpa);
        want.count[0] = @intCast(pg.rule(pick, slots, E, R, want.ids, want.members));
        const one = try grouped(t, pick, R, false);
        defer one.deinit(t.gpa);
        const par = try grouped(t, pick, R, true);
        defer par.deinit(t.gpa);
        var busiest: usize = 0;
        for (0..E) |e| busiest = @max(busiest, std.mem.count(i32, pick, &.{@intCast(e)}));
        total += 1;
        const bad = par.diff(one) orelse one.diff(want);
        if (bad) |b| {
            t.fail("lists {s} r{d}: {s} word {d} differs ({s})", .{ name, R, b[0], b[1], if (par.diff(one) != null) "groupPar against group" else "group against the rule" });
            continue;
        }
        pass += 1;
        t.out("PASS {s}lists {s} r{d}: uids {d} B, ucount 4 B, members {d} B: group = groupPar = the rule (ucount {d}, busiest {d} members)", .{ if (t.dev == null) "dry " else "", name, R, 4 * one.ids.len, 4 * one.members.len, want.count[0], busiest });
    };
    t.out("RESULT lists: {d} of {d} byte-equal ({s})", .{ pass, total, if (t.dev == null) "on host memory, the launches as built" else "on the GPU" });
}

fn median(us: []f64) [3]f64 {
    std.mem.sort(f64, us, {}, std.sort.asc(f64));
    const n = us.len;
    return .{ us[n / 2], us[n / 10], us[n - 1 - n / 10] };
}

/// `a` and `b` N times each, alternating run by run (a first on even runs), between chained events.
fn timeAB(t: anytype, d: anytype, a: *const Launches, b: *const Launches, ua: []f64, ub: []f64) !void {
    const n = ua.len;
    const ev = try t.gpa.alloc(cuda.Event, 2 * n + 1);
    defer t.gpa.free(ev);
    var made: usize = 0;
    defer for (ev[0..made]) |*e| e.deinit();
    while (made < ev.len) : (made += 1) ev[made] = try cuda.Event.init(d.d, true);
    try ev[0].record(d.stream);
    for (0..2 * n) |k| {
        const a_now = (k % 2 == 0) == ((k / 2) % 2 == 0);
        try d.set.run(d.stream, if (a_now) a else b);
        try ev[k + 1].record(d.stream);
    }
    try ev[2 * n].synchronize();
    for (0..2 * n) |k| {
        const us = @as(f64, try cuda.Event.elapsedMs(ev[k], ev[k + 1])) * 1000;
        if ((k % 2 == 0) == ((k / 2) % 2 == 0)) ua[k / 2] = us else ub[k / 2] = us;
    }
}

/// --bench N, the switch on: the grouping alone and the routed call, off against on, at 1,024 and 2,048 rows.
fn bench(t: anytype) !void {
    const d = t.dev.?;
    const n = t.a.bench;
    const E = t.bound.count;
    const tl = try t.facts.tiles();
    for (rows_bench) |R| {
        var idb: [32]u8 = undefined;
        const id = try std.fmt.bufPrint(&idb, "routed-bf16-{d}", .{R});
        const xh = try refInput(t, id, "x");
        defer t.gpa.free(xh);
        const ph = try refInput(t, id, "pick");
        defer t.gpa.free(ph);
        const sz = ex.Scratch.bytes(R, slots, t.bound, tl[0], tl[1]);
        const lens = [_]usize{ xh.len, ph.len, sz.xg, sz.xg, sz.xd, sz.z, sz.y, sz.ids, 4, sz.members, 4 * E, 4 * ex.orderSlots(R, slots, E) };
        var bufs: [lens.len]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        defer for (bufs[0..made]) |*b| b.free();
        for (lens) |len| {
            bufs[made] = try cuda.DeviceBuffer.alloc(d.d, len);
            made += 1;
        }
        try bufs[0].upload(0, xh);
        try bufs[1].upload(0, ph);
        const off: ex.Scratch = .{ .xg = bufs[2].ptr, .xu = bufs[3].ptr, .xd = bufs[4].ptr, .z = bufs[5].ptr, .y = bufs[6].ptr, .ids = bufs[7].ptr, .count = bufs[8].ptr, .members = bufs[9].ptr, .rows = R, .slots = slots, .experts = E, .cfg_gu = tl[0], .cfg_d = tl[1] };
        var on = off;
        on.counts = bufs[10].ptr;
        const suh: ?u64 = if (t.suh_shared) t.bound.suh_g else null; // python's shared_suh: expert 0's row, the same bytes
        var calls: [2]Launches = .{ .{}, .{} };
        for ([_]ex.Scratch{ off, on }, &calls) |s, *ls| try ex.routedX1(ls, bufs[0].ptr, t.bound.dims, true, bufs[1].ptr, null, t.bound, s, 0, R, t.facts.limit, .bf16, true, d.sh, t.sched, t.x1.p, bufs[11].ptr, suh);
        var groups: [2]Launches = .{ .{}, .{} };
        try ex.group(&groups[0], bufs[1].ptr, off.ids, off.count, off.members, R, slots, E, R, d.sh);
        try ex.groupPar(&groups[1], bufs[1].ptr, on.counts, on.ids, on.count, on.members, R, slots, E, R);
        try d.ctx.synchronize();
        for (0..3) |_| for (calls ++ groups) |ls| try d.set.run(d.stream, &ls); // warm-up, the attribute raised once
        try d.stream.synchronize();
        const ua = try t.gpa.alloc(f64, n);
        defer t.gpa.free(ua);
        const ub = try t.gpa.alloc(f64, n);
        defer t.gpa.free(ub);
        for ([_][]const u8{ "group", "routed" }, [_]*const [2]Launches{ &groups, &calls }) |part, ab| {
            try timeAB(t, d, &ab[0], &ab[1], ua, ub);
            const a = median(ua);
            const b = median(ub);
            t.out("BENCH-PG r{d} {s}: off median {d:.1} us (p10 {d:.1}, p90 {d:.1}), on median {d:.1} us (p10 {d:.1}, p90 {d:.1}), on/off {d:.3} ({d} runs each, alternating; x1 {s}, l2d {d}, pdl {d}, rowrot {s})", .{ R, part, a[0], a[1], a[2], b[0], b[1], b[2], b[0] / a[0], n, if (t.x1.p.on()) "on" else "off", t.sched.l2d, @intFromBool(t.sched.pdl), if (suh != null) "on" else "off" });
        }
        var sums: [2][64]u8 = undefined;
        const y = try t.gpa.alloc(u8, sz.y);
        defer t.gpa.free(y);
        for (calls, &sums) |ls, *s| {
            try d.set.run(d.stream, &ls);
            try d.stream.synchronize();
            try bufs[6].download(0, y);
            var h: [32]u8 = undefined;
            Sha256.hash(y, &h, .{});
            s.* = std.fmt.bytesToHex(h, .lower);
        }
        if (std.mem.eql(u8, &sums[0], &sums[1])) {
            t.out("PASS bench-pg r{d}: the routed call's y off and on byte-equal after the runs (sha256 {s})", .{ R, sums[0][0..16] });
        } else t.fail("bench-pg r{d}: the routed call's y differs off ({s}) and on ({s})", .{ R, sums[0][0..16], sums[1][0..16] });
    }
}
