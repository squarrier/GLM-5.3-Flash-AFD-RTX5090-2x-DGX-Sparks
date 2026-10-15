//! G-lanes (qualification_9e49592e): the expert halves' side of the prompt chunk lanes (TF_AFD_LANES=N, evidence/qualification_0f991257/WIRE.md §13) over
//! G-loop's harness (afd_expert_loop.zig: three nodes in one process, fn_xchg's kernels on CPU threads, tf-xchg's lab
//! half as each half's compute) on settings of N slots a source. The attention here sends §13's order by hand, as
//! python 0026's lanes send it with N in flight: every lane's first request, then each request's replies taken in the
//! wire's order and that lane's next request sent after them; a window's rows come from its chunk alone (`seed`), so
//! the same chunk sent solo or in a lane carries the same bytes. The halves' half of G-trace-lanes (each window served
//! with its own rows, in the wire's order, at 8,000 tokens too), and the refusals: lanes not agreed, an old peer, out
//! of order, more lanes than agreed, a group stopped early, a dead half failing the request (#214).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const loop = @import("afd_expert_loop.zig");
const pp = @import("afd_expert_pairs.zig");
const ae = glm.afd_expert;
const xchg = cuda.xchg;
const proto = xchg.proto;
const Harness = loop.Harness;
const Win = pp.Win;

const gpa = std.testing.allocator;
const testing = std.testing;
const layers = loop.layers;
const ms = loop.ms;

/// WIRE.md §13: lane `k`'s request at position `pos` of the group `g` (lane 0's fseq its first; word 8 the next
/// lane's rows, the last lane's lane 0's; word 10 the lanes).
fn laneHeader(g: []const Win, k: usize, pos: usize) proto.Header {
    return .{ .fseq = g[k].fseq, .rows = @intCast(g[k].rows), .layer = @intCast(pos), .mode = .lanes, .layers = layers, .extra = .{ @intCast(k), @intCast(g[(k + 1) % g.len].rows), g[0].fseq, @intCast(g.len) } };
}

/// Request i of the group in the wire's order (lane i % n at position i / n), or `bad`'s header at its index.
fn header(g: []const Win, i: usize, bad: ?pp.Bad) proto.Header {
    if (bad) |x| if (x.at == i) return x.hd;
    return laneHeader(g, i % g.len, i / g.len);
}

/// A lane group as the attention sends it with N = g.len in flight: every lane's first request, then each request's
/// replies taken in the wire's order and the next request sent after them; requests [0, upto) only (then the attention
/// stops); `bad` replaces request `bad.at`'s header. Stops at a failure.
fn sendLanes(h: *Harness, g: []const Win, w6: i32, upto: usize, bad: ?pp.Bad) !u64 {
    const total = @min(upto, g.len * layers);
    var wrong: u64 = 0;
    var sent: usize = 0;
    while (sent < @min(total, g.len)) : (sent += 1) try pp.send(h, header(g, sent, bad), w6, g[sent % g.len].seed);
    for (0..total) |i| {
        wrong += (try pp.collect(h, header(g, i, bad), w6, g[i % g.len].seed)) orelse return wrong;
        if (sent < total) {
            try pp.send(h, header(g, sent, bad), w6, g[sent % g.len].seed);
            sent += 1;
        }
    }
    return wrong;
}

/// The record a group leaves: every lane's position p in lane order, then p + 1 (the wire's order).
fn wantLanes(out: *std.ArrayList([4]usize), g: []const Win) !void {
    for (0..layers) |pos| for (g, 0..) |w, k| try out.append(gpa, .{ pos, w.rows, k, @intCast(w.fseq) });
}

/// qualification_47b9bf3a: a lane group whose absorbs are its lanes' extra layer (the harness's 42 positions: the backbone's 41 and the
/// MTP layer's last), sent as sendLanes does with lane k's word 6 `w6[k]`: 42 (word 5) for an absorbing lane, 41 for
/// the last lane when `short` (the prompt's last chunk: its absorb after the group), whose MTP request is not sent;
/// requests [0, upto) only.
fn sendLanesMtp(h: *Harness, g: []const Win, w6: []const i32, short: bool, upto: usize) !u64 {
    const total = @min(upto, g.len * layers - @intFromBool(short));
    var wrong: u64 = 0;
    var sent: usize = 0;
    while (sent < @min(total, g.len)) : (sent += 1) try pp.send(h, header(g, sent, null), w6[sent % g.len], g[sent % g.len].seed);
    for (0..total) |i| {
        wrong += (try pp.collect(h, header(g, i, null), w6[i % g.len], g[i % g.len].seed)) orelse return wrong;
        if (sent < total) {
            try pp.send(h, header(g, sent, null), w6[sent % g.len], g[sent % g.len].seed);
            sent += 1;
        }
    }
    return wrong;
}

const backbone = layers - 1; // TF_AFD_MTP=1's form: the backbone's positions; the last is the MTP layer's

/// One run's records: each expert's (position, rows, slot, fseq) and (request, reply) hashes, in compute order.
const Rec = struct { seen: [][4]usize, sums: [][2]u64 };

/// Chunks `rows` (seeds 100-103) served serially (each a slice [0, 41) then its absorb, the slice [41, 1), as qualification_21e21bc0's
/// attention sends them) or, `n` > 0, in lanes of `n` with the absorbs in-lane (four lanes: one group, the last chunk
/// short and its absorb after; three: a group, then chunk 3 alone and its absorb), on a fresh harness of `wire` rows.
fn recordMtp(n: u8, rows: [4]usize, wire: usize) ![2]Rec {
    var h: Harness = .{ .io = std.testing.io, .pairs = n > 0, .inflight = @max(n, 1), .lanes = n, .rows = wire };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    var ch: [4]Win = undefined;
    for (&ch, rows, 0..) |*w, r, k| w.* = .{ .fseq = @intCast(k + 1), .rows = r, .seed = @intCast(100 + k) };
    var wrong: u64 = 0;
    if (n == 0) {
        for (ch, 0..) |c, k| {
            const f: i32 = @intCast(2 * k + 1);
            wrong += try pp.sendSolo(&h, .{ .fseq = f, .rows = c.rows, .seed = c.seed }, .slice, 0, backbone, backbone);
            wrong += try pp.sendSolo(&h, .{ .fseq = f + 1, .rows = c.rows, .seed = c.seed }, .slice, backbone, 1, 1);
        }
    } else {
        const w6 = [_]i32{ layers, layers, layers, if (n == 4) backbone else layers };
        wrong += try sendLanesMtp(&h, ch[0..n], w6[0..n], n == 4, std.math.maxInt(usize));
        if (n == 3) wrong += try pp.sendSolo(&h, .{ .fseq = 4, .rows = ch[3].rows, .seed = ch[3].seed }, .slice, 0, backbone, backbone);
        wrong += try pp.sendSolo(&h, .{ .fseq = 5, .rows = ch[3].rows, .seed = ch[3].seed }, .slice, backbone, 1, 1); // chunk 3's absorb
    }
    loop.settle();
    const failed = h.attn().failure();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?xchg.Failure, null), failed);
    for (h.ends) |e| try testing.expectEqual(ae.End.stopped, e);
    var out: [2]Rec = undefined;
    for (&out, &h.experts) |*o, *x| o.* = .{ .seen = try gpa.dupe([4]usize, x.seen.items), .sums = try gpa.dupe([2]u64, x.sums.items) };
    return out;
}

fn freeRec(run: [2]Rec) void {
    for (run) |r| {
        gpa.free(r.seen);
        gpa.free(r.sums);
    }
}

/// Where chunk c's position `pos` was computed in a lanes run of `n` (recordMtp's order).
fn laneIndex(n: usize, c: usize, pos: usize) usize {
    if (c < n and (c + 1 < n or n == 3)) return pos * n + c; // in-lane: every position in the wire's order
    if (n == 4) return if (pos < backbone) pos * n + c else n * layers - 1; // the short last lane, then its absorb
    return n * layers + pos; // three lanes: chunk 3 alone, then its absorb
}

/// G-trace-lanes' halves' half with the MTP absorbs in-lane: serial against lanes of `n`, by chunk and position.
fn traceMtp(n: u8, rows: [4]usize, wire: usize) !void {
    const s = try recordMtp(0, rows, wire);
    defer freeRec(s);
    const l = try recordMtp(n, rows, wire);
    defer freeRec(l);
    for (s, l) |x, y| {
        try testing.expectEqual(4 * layers, x.seen.len);
        try testing.expectEqual(4 * layers, y.seen.len);
        for (0..4) |c| for (0..layers) |pos| {
            const si = c * layers + pos;
            const li = laneIndex(n, c, pos);
            try testing.expectEqual([2]usize{ pos, rows[c] }, [2]usize{ x.seen[si][0], x.seen[si][1] });
            try testing.expectEqual([2]usize{ pos, rows[c] }, [2]usize{ y.seen[li][0], y.seen[li][1] });
            try testing.expectEqual(x.sums[si], y.sums[li]);
        };
    }
    std.debug.print("G-trace-lanes halves N={d} (MTP absorbs in-lane): chunks {any} on a {d}-row wire, 42 positions each, both halves: requests and replies in lanes = solo, by chunk and position\n", .{ n, rows, wire });
}

test "qualification_47b9bf3a: G-trace-lanes, the halves' half with the MTP absorbs as each lane's extra layer (word 6 = 42 = word 5; the prompt's last chunk one short, its absorb after the group), scaled and at 8,000 tokens in 2,048-row chunks, four lanes and three: every request and reply equal the serial slices' by chunk and position" {
    for ([_]u8{ 4, 3 }) |n| {
        try traceMtp(n, .{ 32, 32, 32, 29 }, loop.max_rows);
        try traceMtp(n, .{ 2048, 2048, 2048, 1856 }, ae.prefill_rows);
    }
}

test "qualification_47b9bf3a refused: a short lane that is not the last (lane 1's word 6 one less than lane 0's = word 5), and a short last lane in a group without the MTP position (lane 0's word 6 = 41, not word 5): WrongPartner at that lane's first request; nothing posted for it" {
    const cases = [_]struct { w6: [3]i32, at: u64 }{ .{ .w6 = .{ layers, backbone, backbone }, .at = 1 }, .{ .w6 = .{ backbone, backbone, backbone - 1 }, .at = 2 } };
    for (cases) |case| {
        var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = 3, .lanes = 3 };
        try h.start(20 * std.time.ns_per_s);
        defer h.free();
        const wrong = try pp.sendSolo(&h, .{ .fseq = 1, .rows = 8, .seed = 1 }, .eager, 0, layers, 0);
        _ = try sendLanesMtp(&h, &g3, &case.w6, false, std.math.maxInt(usize));
        _ = try pp.sendSolo(&h, .{ .fseq = 5, .rows = 8, .seed = 5 }, .eager, 0, layers, 0); // fails at once if a half is gone
        loop.settle();
        const failed = h.attn().failure();
        h.stop();
        try testing.expectEqual(@as(u64, 0), wrong);
        try testing.expectEqual(@as(?proto.Why, .dead), pp.why(failed));
        for (h.ends, h.stats, h.posted) |e, s, p| {
            try testing.expectEqual(ae.End{ .breach = error.WrongPartner }, e);
            try testing.expectEqual(@as(u64, 1 + 1 + case.at), s.turns);
            try testing.expect(p <= layers + case.at); // never a reply for the refused request
        }
    }
}

fn wantSolo(out: *std.ArrayList([4]usize), w: Win) !void {
    for (0..layers) |pos| try out.append(gpa, .{ pos, w.rows, 0, @intCast(w.fseq) });
}

test "G-lanes: decode windows, solo chunks, a group of four (one lane at decode rows), a group of three and a pair in one run on lanes=4's settings: a CPU turn for each window's first request; each window's own rows in the wire's order (lane 0's position p, lane 1's, ...); replies exact; counts exact; a stop exits 0" {
    var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = 4, .lanes = 4 };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const d1: Win = .{ .fseq = 1, .rows = 1, .seed = 1 };
    const g4 = [_]Win{ .{ .fseq = 2, .rows = 32, .seed = 2 }, .{ .fseq = 3, .rows = 17, .seed = 3 }, .{ .fseq = 4, .rows = 9, .seed = 4 }, .{ .fseq = 5, .rows = 3, .seed = 5 } };
    const c1: Win = .{ .fseq = 6, .rows = 32, .seed = 6 };
    const g3b = [_]Win{ .{ .fseq = 7, .rows = 32, .seed = 7 }, .{ .fseq = 8, .rows = 32, .seed = 8 }, .{ .fseq = 9, .rows = 5, .seed = 9 } };
    const a: Win = .{ .fseq = 10, .rows = 9, .seed = 10 };
    const b: Win = .{ .fseq = 11, .rows = 3, .seed = 11 };
    const d2: Win = .{ .fseq = 12, .rows = 2, .seed = 12 };
    var wrong: u64 = 0;
    wrong += try pp.sendSolo(&h, d1, .eager, 0, layers, 0);
    wrong += try sendLanes(&h, &g4, 0, std.math.maxInt(usize), null);
    wrong += try pp.sendSolo(&h, c1, .chunk, 0, layers, 0);
    wrong += try sendLanes(&h, &g3b, 0, std.math.maxInt(usize), null);
    wrong += try pp.sendPair2(&h, a, b, layers, 0, 2 * layers); // a remainder of two: a pair's group (mode 7)
    wrong += try pp.sendSolo(&h, d2, .eager, 0, layers, 0);
    loop.settle();
    const c = h.counts();
    const failed = h.attn().failure();
    var tb: [256]u8 = undefined;
    const text = try h.attn().cfg.settings.text(&tb);
    h.stop();
    const n = 12 * layers;
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?xchg.Failure, null), failed);
    try testing.expect(std.mem.endsWith(u8, text, " pairs=1 inflight=4 lanes=4"));
    try testing.expectEqual(@as(u32, n), c.rung);
    try testing.expectEqual([2]u64{ n, n }, c.requests);
    try testing.expectEqual([2]u64{ n, n }, h.posted);
    var want: std.ArrayList([4]usize) = .empty;
    defer want.deinit(gpa);
    try wantSolo(&want, d1);
    try wantLanes(&want, &g4);
    try wantSolo(&want, c1);
    try wantLanes(&want, &g3b);
    for (0..layers) |pos| {
        try want.append(gpa, .{ pos, a.rows, 0, @intCast(a.fseq) });
        try want.append(gpa, .{ pos, b.rows, 1, @intCast(b.fseq) });
    }
    try wantSolo(&want, d2);
    for (h.ends, h.stats, h.experts) |e, s, x| {
        try testing.expectEqual(ae.End.stopped, e);
        try testing.expectEqual(ae.Stats{ .turns = 12, .forwards = 12, .layers = n, .rows = layers * (1 + 32 + 17 + 9 + 3 + 32 + 32 + 32 + 5 + 9 + 3 + 2) }, s);
        try testing.expectEqualSlices([4]usize, want.items, x.seen.items);
    }
}

test "qualification_9e49592e: G-trace-lanes, the halves' half: at 8,000 tokens in 2,048-row chunks (2,048 x 3 + 1,856), four lanes in one group and three (a group, then 1,856 alone): every reply right, each window served with its own rows in the wire's order" {
    for ([_]u8{ 4, 3 }) |n| {
        var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = n, .lanes = n, .rows = ae.prefill_rows };
        try h.start(20 * std.time.ns_per_s);
        defer h.free();
        const all = [_]Win{ .{ .fseq = 1, .rows = 2048, .seed = 1 }, .{ .fseq = 2, .rows = 2048, .seed = 2 }, .{ .fseq = 3, .rows = 2048, .seed = 3 }, .{ .fseq = 4, .rows = 1856, .seed = 4 } };
        var wrong = try sendLanes(&h, all[0..n], 0, std.math.maxInt(usize), null);
        if (n == 3) wrong += try pp.sendSolo(&h, all[3], .chunk, 0, layers, 0);
        loop.settle();
        const failed = h.attn().failure();
        h.stop();
        try testing.expectEqual(@as(u64, 0), wrong);
        try testing.expectEqual(@as(?xchg.Failure, null), failed);
        var want: std.ArrayList([4]usize) = .empty;
        defer want.deinit(gpa);
        try wantLanes(&want, all[0..n]);
        if (n == 3) try wantSolo(&want, all[3]);
        for (h.ends, h.experts) |e, x| {
            try testing.expectEqual(ae.End.stopped, e);
            try testing.expectEqualSlices([4]usize, want.items, x.seen.items);
        }
        std.debug.print("G-trace-lanes N={d} (halves, 8,000 tokens in 2,048-row chunks): {d} requests each half, every reply right, the wire's order\n", .{ n, want.items.len });
    }
}

test "qualification_9e49592e refused: lanes not agreed (TF_AFD_LANES unset on every node): a mode-9 first header is WrongMode; nothing is posted for it; the attention fails (dead)" {
    var h: Harness = .{ .io = std.testing.io, .pairs = true };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const wrong = try pp.sendSolo(&h, .{ .fseq = 1, .rows = 32, .seed = 1 }, .chunk, 0, layers, 0);
    const g = [_]Win{ .{ .fseq = 2, .rows = 32, .seed = 2 }, .{ .fseq = 3, .rows = 17, .seed = 3 }, .{ .fseq = 4, .rows = 9, .seed = 4 } };
    _ = try sendLanes(&h, &g, 0, 1, null); // lane 0's first only: one in flight on these settings
    loop.settle();
    const failed = h.attn().failure();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?proto.Why, .dead), pp.why(failed));
    try testing.expectEqual([2]u64{ layers, layers }, h.posted);
    for (h.ends, h.stats) |e, s| {
        try testing.expectEqual(ae.End{ .breach = error.WrongMode }, e);
        try testing.expectEqual(@as(u8, 3), e.code());
        try testing.expectEqual(ae.Stats{ .turns = 2, .forwards = 1, .layers = layers, .rows = 32 * layers }, s);
    }
}

test "qualification_9e49592e refused: an old peer: the attention with lanes=4 and halves with the pairs alone, the reverse, and lanes=3 against lanes=4, disagree at the handshake (SettingsDiffer) before any request" {
    for ([_][2]u8{ .{ 4, 0 }, .{ 0, 4 }, .{ 3, 4 } }) |ab| {
        var h: Harness = .{ .io = std.testing.io, .pairs = true, .lanes = ab[1], .inflight = @max(ab[1], 1), .attn_lanes = ab[0], .attn_inflight = @max(ab[0], 1) };
        const got = h.start(20 * std.time.ns_per_s);
        if (got) |_| {
            h.free();
            return error.TestExpectedSettingsDiffer;
        } else |_| {}
        for (h.nodes[0].links) |l| if (l) |x| x.close();
        h.nodes[0].links = .{ null, null };
        h.server.close();
        for (h.threads) |t| if (t) |x| x.join();
        for (1..3) |i| h.nodes[i].deinit();
        h.nodes[0].deinit();
        h.bufs.deinit(h.engine(), gpa);
        for (h.regions) |r| gpa.free(r);
        for (h.status) |s| gpa.free(s);
        try testing.expectError(error.SettingsDiffer, got);
        var differ: usize = 0;
        for (&h.experts, h.ends) |*x, e| {
            try testing.expect(e == .err);
            differ += @intFromBool(e.err == error.SettingsDiffer);
            try testing.expectEqual(@as(usize, 0), x.seen.items.len);
            x.deinit();
        }
        try testing.expect(differ >= 1);
    }
}

const g3 = [_]Win{ .{ .fseq = 2, .rows = 32, .seed = 2 }, .{ .fseq = 3, .rows = 17, .seed = 3 }, .{ .fseq = 4, .rows = 9, .seed = 4 } };

/// A run on lanes=3's settings whose group of three is refused at request `bad.at`: the breach, the replies posted
/// for the group, the group's turns.
fn refused(bad: pp.Bad, want: ae.Breach, posted: u64, turns: u64) !void {
    var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = 3, .lanes = 3 };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const wrong = try pp.sendSolo(&h, .{ .fseq = 1, .rows = 8, .seed = 1 }, .eager, 0, layers, 0);
    _ = try sendLanes(&h, &g3, 0, if (bad.hd.extra[3] > 3) 1 else std.math.maxInt(usize), bad);
    _ = try pp.sendSolo(&h, .{ .fseq = 5, .rows = 8, .seed = 5 }, .eager, 0, layers, 0); // fails at once if a half is gone
    loop.settle();
    const failed = h.attn().failure();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?proto.Why, .dead), pp.why(failed));
    // with three in flight nothing waits for the replies before the refused request: the half may exit 3 while its
    // proxy still holds some of them (WrongPairEnd, after the group's sync: the refused one's too)
    const late: u64 = posted;
    for (h.posted) |p| {
        if (p > layers + posted or p + late < layers + posted) {
            std.debug.print("posted {any}: want {d} each (or {d} fewer)\n", .{ h.posted, layers + posted, late });
            return error.TestUnexpectedPosted;
        }
    }
    for (h.ends, h.stats) |e, s| {
        try testing.expectEqual(ae.End{ .breach = want }, e);
        try testing.expectEqual(@as(u8, 3), e.code());
        try testing.expectEqual(@as(u64, 1 + turns), s.turns);
        try testing.expectEqual(@as(u64, 1), s.forwards);
    }
}

test "qualification_9e49592e refused: out of order: lane 1's first before lane 0's, or a group of four on lanes=3's settings (WrongPair); a new group's first where lane 1's belongs, a solo chunk where lane 2's belongs, lane 2's rows not what lane 1 announced, the last lane's word 8 not lane 0's rows (WrongPartner); the group's tail from another window (WrongPairEnd)" {
    try refused(.{ .at = 0, .hd = laneHeader(&g3, 1, 0) }, error.WrongPair, 0, 1);
    var four = laneHeader(&g3, 0, 0);
    four.extra[3] = 4; // more lanes than the settings agreed
    try refused(.{ .at = 0, .hd = four }, error.WrongPair, 0, 1);
    var again = laneHeader(&g3, 0, 0);
    again.fseq = 3; // a new group opened where lane 1's first belongs
    again.extra[2] = 3;
    try refused(.{ .at = 1, .hd = again }, error.WrongPartner, 1, 2);
    const chunk: proto.Header = .{ .fseq = 4, .rows = 9, .layer = 0, .mode = .chunk, .layers = layers };
    try refused(.{ .at = 2, .hd = chunk }, error.WrongPartner, 2, 3);
    var rows = laneHeader(&g3, 2, 0);
    rows.rows = 16; // not the rows lane 1 announced
    try refused(.{ .at = 2, .hd = rows }, error.WrongPartner, 2, 3);
    var wrap = laneHeader(&g3, 2, 0);
    wrap.extra[1] = 31; // the last lane's word 8 must be lane 0's rows
    try refused(.{ .at = 2, .hd = wrap }, error.WrongPartner, 2, 3);
    var last = laneHeader(&g3, 2, layers - 1);
    last.layer = 0; // a request of another window's start, served as the group's last
    try refused(.{ .at = 3 * layers - 1, .hd = last }, error.WrongPairEnd, 3 * layers, 3);
}

test "qualification_9e49592e: a group stopped early with three in flight (requests 0..6 sent), then the attention gone: each half's next take gives up (300 ms), it exits 3, nothing is posted after; never a hang" {
    var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = 3, .lanes = 3 };
    try h.start(300 * ms);
    defer h.free();
    const wrong = try sendLanes(&h, &g3, 0, 7, null);
    loop.settle();
    h.attnGone();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual([2]u64{ 7, 7 }, h.posted);
    for (h.ends, h.stats, h.experts) |e, s, x| {
        try testing.expect(e == .failed); // the take of request 7 gave up; every later one returned at once
        try testing.expectEqual(proto.Why.gave_up, e.failed.why);
        try testing.expectEqual(@as(u8, 3), e.code());
        try testing.expectEqual(@as(u64, 3), s.turns); // each lane's first
        try testing.expectEqual(@as(u64, 0), s.forwards);
        try testing.expectEqual(3 * layers, x.seen.items.len); // the loop ran the group's computes, posting none after 7
    }
}

test "qualification_9e49592e #214: --fault exit at lane 1's first request inside a group of three (three in flight): the request before it answered; the attention fails on it (dead), never a hang; the other half exits 3" {
    // request 44 = lane 1's first (a solo forward's 42 first, lane 0's 43): expert 0 answers 43, then dies once 44 lands
    var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = 3, .lanes = 3, .fault = .{ .{ .what = .exit, .at = layers + 2 }, null } };
    try h.start(300 * ms);
    defer h.free();
    const wrong = try pp.sendSolo(&h, .{ .fseq = 1, .rows = 4, .seed = 1 }, .eager, 0, layers, 0);
    _ = try sendLanes(&h, &g3, 0, std.math.maxInt(usize), null);
    loop.settle();
    const failed = h.attn().failure();
    h.attnGone();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?proto.Why, .dead), pp.why(failed));
    const at = pp.where(failed) orelse return error.TestExpectedFailure;
    try testing.expectEqual(@as(u32, 0), at[0]);
    try testing.expect(at[1] >= layers + 2 and at[1] <= layers + 4); // the doorbell when the death was seen: 44, or past it (45, 46 in flight)
    try testing.expectEqual(ae.End.die, h.ends[0]);
    try testing.expectEqual(ae.Stats{ .turns = 2, .forwards = 1, .layers = layers + 1, .rows = 4 * layers + 32 }, h.stats[0]);
    try testing.expectEqual(@as(u64, layers + 1), h.posted[0]); // nothing from expert 0 for lane 1's first
    try testing.expect(h.posted[1] >= layers + 2 and h.posted[1] <= layers + 4); // expert 1 answered what had landed
    try testing.expectEqual(@as(u8, 3), h.ends[1].code());
}

test "the halves' TF_AFD_LANES (qualification_9e49592e): unset or 0 off; 3 or 4 with TF_AFD_PAIRS=1 and TF_AFD_INFLIGHT unset; anything else refuses the run (BadAfdLanes)" {
    const P = ae.pairs;
    try testing.expectEqual(@as(u8, 0), try P.parseLanes(null, false, false));
    try testing.expectEqual(@as(u8, 0), try P.parseLanes("0", true, true));
    try testing.expectEqual(@as(u8, 3), try P.parseLanes("3", true, false));
    try testing.expectEqual(@as(u8, 4), try P.parseLanes("4", true, false));
    try testing.expectError(error.BadAfdLanes, P.parseLanes("4", false, false)); // lanes only with the pairs
    try testing.expectError(error.BadAfdLanes, P.parseLanes("3", true, true)); // the lanes set the requests in flight
    for ([_][]const u8{ "1", "2", "5", "true", "" }) |v| try testing.expectError(error.BadAfdLanes, P.parseLanes(v, true, false));
}
