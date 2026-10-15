//! G-pairs (qualification_c2d4ee2e): the expert halves' side of the 3-node prefill pairs (TF_AFD_PAIRS=1, evidence/qualification_0f991257/WIRE.md) over
//! G-loop's harness (afd_expert_loop.zig: three nodes in one process, fn_xchg's kernels on CPU threads, tf-xchg's lab
//! half as each half's compute). The attention here sends WIRE.md §4's order by hand, A_0, B_0, A_1, B_1, ..., each
//! request's two replies taken before the next goes; a window's rows come from its chunk alone (`seed`), so the same
//! chunk sent solo or paired carries the same bytes. The halves' half of G-trace-pairs, the interleave from the record,
//! and the refusals: pairs not agreed, an old peer, out of order, a pair stopped early, a dead or silent half.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const loop = @import("afd_expert_loop.zig");
const lab = @import("xchg/lab.zig");
const ae = glm.afd_expert;
const xchg = cuda.xchg;
const proto = xchg.proto;
const Harness = loop.Harness;

const gpa = std.testing.allocator;
const testing = std.testing;
const layers = loop.layers;
const ms = loop.ms;

/// A prompt chunk's window: its forward number, rows, and the chunk its rows are made from.
pub const Win = struct { fseq: i32, rows: usize, seed: i32 };

/// One request from the attention (header `hd`, word 6 `w6`; its rows the chunk `seed`'s), both replies taken and
/// checked against the stand-in: the wrong values, or null once the exchange failed.
fn exchange(h: *Harness, hd: proto.Header, w6: i32, seed: i32) !?u64 {
    try send(h, hd, w6, seed);
    return collect(h, hd, w6, seed);
}

/// The request's bytes (header `hd`, word 6 `w6`, the chunk `seed`'s rows) into the attention's request buffer.
fn fill(h: *Harness, hd: proto.Header, w6: i32, seed: i32) void {
    var f = hd;
    f.fseq = seed;
    lab.fillRequest(loop.layout, loop.experts, f, h.bufs.req.host);
    hd.write(h.bufs.req.host[0..proto.HDR_BYTES]);
    std.mem.writeInt(i32, h.bufs.req.host[24..28], w6, .little);
}

/// The request posted (into its send slot: the buffer is free again at once).
pub fn send(h: *Harness, hd: proto.Header, w6: i32, seed: i32) !void {
    fill(h, hd, w6, seed);
    try h.engine().post(h.attn(), h.bufs.req, proto.wireBytes(loop.layout.req(@intCast(hd.rows))));
}

/// The next request's two replies taken and checked against the stand-in over the request they answer (refilled from
/// its header and seed, so another request may have gone since): the wrong values, or null once the exchange failed.
pub fn collect(h: *Harness, hd: proto.Header, w6: i32, seed: i32) !?u64 {
    const R: usize = @intCast(hd.rows);
    const rb = loop.layout.rep(R, false);
    try h.engine().take(h.attn(), &h.bufs.rep, &.{ rb, rb });
    if (h.attn().failure() != null) return null;
    fill(h, hd, w6, seed);
    var wrong: u64 = 0;
    for (0..2) |k| {
        const hh = lab.half(loop.experts, @intCast(k));
        lab.expertRef(loop.layout, h.bufs.req.host, R, hh.lo, hh.hi, false, h.bufs.ref);
        wrong += lab.wrongValues(h.bufs.rep[k].host[0..rb], h.bufs.ref[0..rb], 4);
    }
    return wrong;
}

/// WIRE.md §3: window `slot`'s request at position `pos` of the pair (a, b).
fn pairHeader(a: Win, b: Win, slot: u32, pos: usize) proto.Header {
    const w = if (slot == 0) a else b;
    const o = if (slot == 0) b else a;
    return .{ .fseq = w.fseq, .rows = @intCast(w.rows), .layer = @intCast(pos), .mode = .pair, .layers = layers, .extra = .{ @intCast(slot), @intCast(o.rows), a.fseq, 0 } };
}

pub const Bad = struct { at: usize, hd: proto.Header };

/// A pair as WIRE.md §4 sends it: A_0, B_0, A_1, ... over `count` positions (word 6 `w6`), requests [0, upto) only
/// (then the attention stops); `bad` replaces request `bad.at`'s header. Stops at a failure.
fn sendPair(h: *Harness, a: Win, b: Win, count: usize, w6: i32, upto: usize, bad: ?Bad) !u64 {
    var wrong: u64 = 0;
    for (0..@min(upto, 2 * count)) |k| {
        const slot: u32 = @intCast(k % 2);
        var hd = pairHeader(a, b, slot, k / 2);
        if (bad) |x| {
            if (x.at == k) hd = x.hd;
        }
        wrong += (try exchange(h, hd, w6, if (slot == 0) a.seed else b.seed)) orelse return wrong;
    }
    return wrong;
}

/// qualification_c533375f: a pair with two requests in flight (WIRE.md §12): A_0 and B_0 posted together, then each request's replies
/// taken only once the request after it has gone (A_{p+1} posted before B_p's take), as the attention sends them with
/// TF_AFD_INFLIGHT=2; requests [0, upto) only. Stops at a failure.
pub fn sendPair2(h: *Harness, a: Win, b: Win, count: usize, w6: i32, upto: usize) !u64 {
    const n = @min(upto, 2 * count);
    if (n == 0) return 0;
    var wrong: u64 = 0;
    try send(h, pairHeader(a, b, 0, 0), w6, a.seed);
    for (0..n) |k| {
        if (k + 1 < n) {
            const s: u32 = @intCast((k + 1) % 2);
            try send(h, pairHeader(a, b, s, (k + 1) / 2), w6, if (s == 0) a.seed else b.seed);
        }
        const s: u32 = @intCast(k % 2);
        wrong += (try collect(h, pairHeader(a, b, s, k / 2), w6, if (s == 0) a.seed else b.seed)) orelse return wrong;
    }
    return wrong;
}

/// A solo forward of mode `mode` over positions [first, first + count) (word 6 `w6`, a slice's count).
pub fn sendSolo(h: *Harness, w: Win, mode: proto.Mode, first: usize, count: usize, w6: i32) !u64 {
    var wrong: u64 = 0;
    for (first..first + count) |pos| {
        const hd: proto.Header = .{ .fseq = w.fseq, .rows = @intCast(w.rows), .layer = @intCast(pos), .mode = mode, .layers = layers };
        wrong += (try exchange(h, hd, w6, w.seed)) orelse return wrong;
    }
    return wrong;
}

/// The record a solo forward leaves: (position, rows, slot 0, fseq) for each position.
fn wantSolo(out: *std.ArrayList([4]usize), w: Win, first: usize, count: usize) !void {
    for (first..first + count) |pos| try out.append(gpa, .{ pos, w.rows, 0, @intCast(w.fseq) });
}

/// The record a pair leaves: A_p then B_p, every position (the wire's order).
fn wantPair(out: *std.ArrayList([4]usize), a: Win, b: Win, count: usize) !void {
    for (0..count) |pos| {
        try out.append(gpa, .{ pos, a.rows, 0, @intCast(a.fseq) });
        try out.append(gpa, .{ pos, b.rows, 1, @intCast(b.fseq) });
    }
}

test "G-pairs: decode windows, solo chunks and pairs in one run; a CPU turn for each window's first request; each window's own rows in the wire's order (A_0, B_0, A_1, ...); counts exact; a stop exits 0" {
    var h: Harness = .{ .io = std.testing.io, .pairs = true };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const d1: Win = .{ .fseq = 1, .rows = 1, .seed = 1 };
    const a1: Win = .{ .fseq = 2, .rows = 32, .seed = 2 };
    const b1: Win = .{ .fseq = 3, .rows = 17, .seed = 3 };
    const c1: Win = .{ .fseq = 4, .rows = 32, .seed = 4 };
    const a2: Win = .{ .fseq = 5, .rows = 9, .seed = 5 };
    const b2: Win = .{ .fseq = 6, .rows = 3, .seed = 6 }; // B at decode rows: the halves' decode scratch
    const d2: Win = .{ .fseq = 7, .rows = 2, .seed = 7 };
    var wrong: u64 = 0;
    wrong += try sendSolo(&h, d1, .eager, 0, layers, 0);
    wrong += try sendPair(&h, a1, b1, layers, 0, 2 * layers, null);
    wrong += try sendSolo(&h, c1, .chunk, 0, layers, 0);
    wrong += try sendPair(&h, a2, b2, layers, 0, 2 * layers, null);
    wrong += try sendSolo(&h, d2, .eager, 0, layers, 0);
    loop.settle();
    const c = h.counts();
    const failed = h.attn().failure();
    h.stop();
    const n = 7 * layers;
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?xchg.Failure, null), failed);
    try testing.expectEqual(@as(u32, n), c.rung);
    try testing.expectEqual([2]u64{ n, n }, c.requests);
    try testing.expectEqual([2]u64{ n, n }, h.posted);
    var want: std.ArrayList([4]usize) = .empty;
    defer want.deinit(gpa);
    try wantSolo(&want, d1, 0, layers);
    try wantPair(&want, a1, b1, layers);
    try wantSolo(&want, c1, 0, layers);
    try wantPair(&want, a2, b2, layers);
    try wantSolo(&want, d2, 0, layers);
    for (h.ends, h.stats, h.experts) |e, s, x| {
        try testing.expectEqual(ae.End.stopped, e);
        try testing.expectEqual(@as(u8, 0), e.code());
        // a CPU turn for each window's first request: 3 solo forwards and 2 pairs' 2 windows
        try testing.expectEqual(ae.Stats{ .turns = 7, .forwards = 7, .layers = n, .rows = layers * (1 + 32 + 17 + 32 + 9 + 3 + 2) }, s);
        try testing.expectEqualSlices([4]usize, want.items, x.seen.items);
    }
}

/// One run's records: each expert's (position, rows, slot, fseq) and (request, reply) hashes, in compute order.
const Rec = struct { seen: [][4]usize, sums: [][2]u64 };

/// The chunks of a prompt: 4 windows of `rows` (their bytes from seeds 100-103); forward numbers set by the sender.
const Chunks = [4]Win;

/// Run serial (TF_AFD_PAIRS unset) or paired chunks `c` on a fresh harness of `wire` rows (qualification_c533375f: `two`, the pairs with
/// two requests in flight and TF_AFD_INFLIGHT=2's settings); each expert's records.
fn record(pairs: bool, wire: usize, c: Chunks, two: bool) ![2]Rec {
    var h: Harness = .{ .io = std.testing.io, .pairs = pairs, .rows = wire, .inflight = if (two) 2 else 1 };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const wrong = if (pairs) try paired(&h, c, two) else try serial(&h, c);
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

fn chunks(rows: [4]usize) Chunks {
    var c: Chunks = undefined;
    for (&c, rows, 0..) |*w, r, k| w.* = .{ .fseq = 0, .rows = r, .seed = @intCast(100 + k) };
    return c;
}

const backbone = layers - 1; // TF_AFD_MTP=1's form: the backbone's positions; the last is the MTP layer's

fn with(w: Win, fseq: i32) Win {
    return .{ .fseq = fseq, .rows = w.rows, .seed = w.seed };
}

/// Serial (TF_AFD_PAIRS unset): each chunk a slice [0, 41) (word 6 41, as qualification_21e21bc0's attention sends it), then its MTP
/// absorb, the slice [41, 1); chunk k's forward numbers 2k + 1 and 2k + 2.
fn serial(h: *Harness, ch: Chunks) !u64 {
    var wrong: u64 = 0;
    for (ch, 0..) |c, k| {
        const f: i32 = @intCast(2 * k + 1);
        wrong += try sendSolo(h, with(c, f), .slice, 0, backbone, backbone);
        wrong += try sendSolo(h, with(c, f + 1), .slice, backbone, 1, 1);
    }
    return wrong;
}

/// Paired (TF_AFD_PAIRS=1): chunks (0, 1) and (2, 3) as pairs over the backbone (word 6 41), then each chunk's MTP
/// absorb in chunk order (WIRE.md §4); a pair's forward numbers f, f + 1, its absorbs f + 2, f + 3. `two`: each pair
/// with two requests in flight (qualification_c533375f).
fn paired(h: *Harness, ch: Chunks, two: bool) !u64 {
    var wrong: u64 = 0;
    for (0..2) |q| {
        const f: i32 = @intCast(4 * q + 1);
        const a = with(ch[2 * q], f);
        const b = with(ch[2 * q + 1], f + 1);
        wrong += if (two) try sendPair2(h, a, b, backbone, backbone, 2 * backbone) else try sendPair(h, a, b, backbone, backbone, 2 * backbone, null);
        wrong += try sendSolo(h, with(a, f + 2), .slice, backbone, 1, 1);
        wrong += try sendSolo(h, with(b, f + 3), .slice, backbone, 1, 1);
    }
    return wrong;
}

/// G-trace-pairs' halves' half over chunks `rows` on a wire of `wire` rows: serial against paired (qualification_c533375f: `two`, with
/// two requests in flight), by chunk and position.
fn tracePairs(rows: [4]usize, wire: usize, two: bool) !void {
    const ch = chunks(rows);
    const s = try record(false, wire, ch, false);
    defer freeRec(s); // freed on every path: a failing paired run leaks nothing
    const p = try record(true, wire, ch, two);
    defer freeRec(p);
    for (s, p) |x, y| {
        try testing.expectEqual(4 * layers, x.seen.len);
        try testing.expectEqual(4 * layers, y.seen.len);
        // the paired record: each pair's 2 x 41 computes alternate A_p, B_p, then A's absorb, B's
        for (0..2) |q| {
            const base = q * 2 * layers;
            for (0..2 * backbone) |k| {
                const r = y.seen[base + k];
                try testing.expectEqual([3]usize{ k / 2, k % 2, 4 * q + 1 + k % 2 }, [3]usize{ r[0], r[2], r[3] });
                try testing.expectEqual(ch[2 * q + k % 2].rows, r[1]);
            }
            try testing.expectEqual([4]usize{ backbone, ch[2 * q].rows, 0, 4 * q + 3 }, y.seen[base + 2 * backbone]);
            try testing.expectEqual([4]usize{ backbone, ch[2 * q + 1].rows, 0, 4 * q + 4 }, y.seen[base + 2 * backbone + 1]);
        }
        // each chunk's 42 computes (41 backbone positions, the absorb), by position: rows, request bytes, reply bytes
        for (0..4) |c| {
            const q = c / 2;
            const odd = c % 2;
            for (0..layers) |pos| {
                const si = c * layers + pos; // serial: chunk c's slice, then its absorb
                const pi = if (pos < backbone) q * 2 * layers + 2 * pos + odd else q * 2 * layers + 2 * backbone + odd;
                try testing.expectEqual([2]usize{ pos, ch[c].rows }, [2]usize{ x.seen[si][0], x.seen[si][1] });
                try testing.expectEqual([2]usize{ pos, ch[c].rows }, [2]usize{ y.seen[pi][0], y.seen[pi][1] });
                try testing.expectEqual(x.sums[si], y.sums[pi]);
            }
        }
    }
    std.debug.print("G-trace-pairs halves{s}: chunks {any} on a {d}-row wire, 42 positions each, both halves: requests and replies paired = solo (pair order A_p, B_p, then the absorbs)\n", .{ if (two) " (two in flight)" else "", rows, wire });
}

test "G-trace-pairs, the halves' half (scaled): 4 chunks (32, 32, 32, 29 rows) with TF_AFD_MTP's slices; each chunk's requests past the header and its replies, paired, equal the solo chunk's at every position; the interleave from the record" {
    try tracePairs(.{ 32, 32, 32, 29 }, loop.max_rows, false);
}

test "G-trace-pairs, the halves' half at 8,000 tokens in 2,048-row chunks (2,048, 2,048, 2,048, 1,856), as attn serve chunks them: paired = solo, by chunk and position" {
    try tracePairs(.{ 2048, 2048, 2048, 1856 }, ae.prefill_rows, false);
}

test "qualification_c533375f: G-trace-pairs, the halves' half with two requests in flight (TF_AFD_INFLIGHT=2: B_0 with A_0, A_{p+1} before B_p's take), scaled and at 8,000 tokens in 2,048-row chunks: the halves serve them in the wire's order, paired = solo by chunk and position" {
    try tracePairs(.{ 32, 32, 32, 29 }, loop.max_rows, true);
    try tracePairs(.{ 2048, 2048, 2048, 1856 }, ae.prefill_rows, true);
}

test "qualification_c533375f: G-pairs with two in flight: decode windows, solo chunks and two-in-flight pairs (one B at decode rows) in one run on inflight=2's settings; replies exact, counts exact, a CPU turn for each window's first request, a stop exits 0" {
    var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = 2 };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const d1: Win = .{ .fseq = 1, .rows = 1, .seed = 1 };
    const a1: Win = .{ .fseq = 2, .rows = 32, .seed = 2 };
    const b1: Win = .{ .fseq = 3, .rows = 17, .seed = 3 };
    const c1: Win = .{ .fseq = 4, .rows = 32, .seed = 4 };
    const a2: Win = .{ .fseq = 5, .rows = 9, .seed = 5 };
    const b2: Win = .{ .fseq = 6, .rows = 3, .seed = 6 }; // B at decode rows: the halves' decode scratch
    const d2: Win = .{ .fseq = 7, .rows = 2, .seed = 7 };
    var wrong: u64 = 0;
    wrong += try sendSolo(&h, d1, .eager, 0, layers, 0);
    wrong += try sendPair2(&h, a1, b1, layers, 0, 2 * layers);
    wrong += try sendSolo(&h, c1, .chunk, 0, layers, 0);
    wrong += try sendPair2(&h, a2, b2, layers, 0, 2 * layers);
    wrong += try sendSolo(&h, d2, .eager, 0, layers, 0);
    loop.settle();
    const c = h.counts();
    const failed = h.attn().failure();
    var tb: [256]u8 = undefined;
    const text = try h.attn().cfg.settings.text(&tb);
    h.stop();
    const n = 7 * layers;
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?xchg.Failure, null), failed);
    try testing.expect(std.mem.endsWith(u8, text, " pairs=1 inflight=2"));
    try testing.expectEqual(@as(u32, n), c.rung);
    try testing.expectEqual([2]u64{ n, n }, c.requests);
    try testing.expectEqual([2]u64{ n, n }, h.posted);
    for (h.ends, h.stats) |e, s| {
        try testing.expectEqual(ae.End.stopped, e);
        try testing.expectEqual(ae.Stats{ .turns = 7, .forwards = 7, .layers = n, .rows = layers * (1 + 32 + 17 + 32 + 9 + 3 + 2) }, s);
    }
}

test "qualification_c533375f refused: an old peer for two in flight: the attention with inflight=2 and halves with pairs alone, and the reverse, disagree at the handshake (SettingsDiffer) before any request" {
    for ([_][2]u8{ .{ 2, 1 }, .{ 1, 2 } }) |ab| {
        var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = ab[1], .attn_inflight = ab[0] };
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

/// The attention's failure: why it failed, and at which peer and request (null when it did not fail: a test failure
/// with the values printed, never a crash of the test runner).
pub fn why(f: ?xchg.Failure) ?proto.Why {
    return if (f) |x| x.why else null;
}

pub fn where(f: ?xchg.Failure) ?[2]u32 {
    return if (f) |x| .{ x.peer, x.seq } else null;
}

test "refused: pairs not agreed (TF_AFD_PAIRS unset on the halves): a mode-7 first header is WrongMode; nothing is posted for it; the attention fails (dead)" {
    var h: Harness = .{ .io = std.testing.io };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const wrong = try sendSolo(&h, .{ .fseq = 1, .rows = 32, .seed = 1 }, .chunk, 0, layers, 0);
    _ = try sendPair(&h, .{ .fseq = 2, .rows = 32, .seed = 2 }, .{ .fseq = 3, .rows = 17, .seed = 3 }, layers, 0, 2 * layers, null);
    loop.settle();
    const failed = h.attn().failure();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?proto.Why, .dead), why(failed));
    try testing.expectEqual([2]u64{ layers, layers }, h.posted);
    for (h.ends, h.stats) |e, s| {
        try testing.expectEqual(ae.End{ .breach = error.WrongMode }, e);
        try testing.expectEqual(@as(u8, 3), e.code());
        try testing.expectEqual(ae.Stats{ .turns = 2, .forwards = 1, .layers = layers, .rows = 32 * layers }, s);
    }
}

test "refused: an old peer: the attention with pairs and halves without, and the reverse, disagree at the handshake (SettingsDiffer) before any request" {
    for ([_][2]bool{ .{ true, false }, .{ false, true } }) |ab| {
        var h: Harness = .{ .io = std.testing.io, .pairs = ab[1], .attn_pairs = ab[0] };
        const got = h.start(20 * std.time.ns_per_s);
        if (got) |_| { // the texts agreed (a node without the hunk's " pairs=1"): everything down, then the test fails
            h.free();
            return error.TestExpectedSettingsDiffer;
        } else |_| {}
        // the attention refused at the first hello: close what is up (links, then the listener), then the halves' ends
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
            try testing.expect(e == .err); // no forward ran: the role ends in AFD FAILED <why>, exit 1
            differ += @intFromBool(e.err == error.SettingsDiffer);
            try testing.expectEqual(@as(usize, 0), x.seen.items.len);
            x.deinit();
        }
        try testing.expect(differ >= 1); // the one the attention answered saw the texts differ; the other found it gone
    }
}

/// A run whose pair is refused at request `at` with header `bad`: the breach, the replies posted, the turns.
fn refused(bad: Bad, want: ae.Breach, posted: u64, turns: u64) !void {
    var h: Harness = .{ .io = std.testing.io, .pairs = true };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const a: Win = .{ .fseq = 2, .rows = 32, .seed = 2 };
    const b: Win = .{ .fseq = 3, .rows = 17, .seed = 3 };
    const wrong = try sendSolo(&h, .{ .fseq = 1, .rows = 8, .seed = 1 }, .eager, 0, layers, 0);
    _ = try sendPair(&h, a, b, layers, 0, 2 * layers, bad);
    _ = try sendSolo(&h, .{ .fseq = 4, .rows = 8, .seed = 4 }, .eager, 0, layers, 0); // fails at once if a half is gone
    loop.settle();
    const failed = h.attn().failure();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual(@as(?proto.Why, .dead), why(failed));
    // WrongPairEnd is found after the pair's sync: the refused request's reply was posted and may still be in the
    // half's proxy when it exits 3 (the attention then fails dead at that request, not at the next one)
    const late: u64 = if (want == error.WrongPairEnd) 1 else 0;
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

test "refused: out of order: B's first request before A's (WrongPair); a second slot 0, another pair's first or a solo chunk where B_0 belongs (WrongPartner); A_0 answered, nothing after" {
    const a: Win = .{ .fseq = 2, .rows = 32, .seed = 2 };
    const b: Win = .{ .fseq = 3, .rows = 17, .seed = 3 };
    try refused(.{ .at = 0, .hd = pairHeader(a, b, 1, 0) }, error.WrongPair, 0, 1);
    var again = pairHeader(a, b, 0, 0);
    again.fseq = 3; // a new pair opened where B_0 belongs: the first one stopped after A_0
    again.extra[2] = 3;
    try refused(.{ .at = 1, .hd = again }, error.WrongPartner, 1, 2);
    const chunk: proto.Header = .{ .fseq = 3, .rows = 17, .layer = 0, .mode = .chunk, .layers = layers };
    try refused(.{ .at = 1, .hd = chunk }, error.WrongPartner, 1, 2);
    var rows = pairHeader(a, b, 1, 0);
    rows.rows = 16; // not the rows A announced
    try refused(.{ .at = 1, .hd = rows }, error.WrongPartner, 1, 2);
}

test "refused: B's last request is not B's last (the pair's tail from another forward): WrongPairEnd after the pair's sync; nothing is answered after it; the attention fails (dead)" {
    const a: Win = .{ .fseq = 2, .rows = 32, .seed = 2 };
    const b: Win = .{ .fseq = 3, .rows = 17, .seed = 3 };
    var last = pairHeader(a, b, 1, layers - 1);
    last.layer = 0; // a request of another window's start, served as the pair's last
    try refused(.{ .at = 2 * layers - 1, .hd = last }, error.WrongPairEnd, 2 * layers, 2);
}

test "a pair stopped early, then the attention gone: each half's next take gives up (300 ms), it exits 3, nothing is posted after; never a hang" {
    var h: Harness = .{ .io = std.testing.io, .pairs = true };
    try h.start(300 * ms);
    defer h.free();
    const wrong = try sendPair(&h, .{ .fseq = 1, .rows = 32, .seed = 1 }, .{ .fseq = 2, .rows = 17, .seed = 2 }, layers, 0, 7, null); // A_0 .. A_3
    loop.settle();
    h.attnGone();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual([2]u64{ 7, 7 }, h.posted);
    for (h.ends, h.stats, h.experts) |e, s, x| {
        try testing.expect(e == .failed); // the take of B_3 gave up; every later one returned at once
        try testing.expectEqual(proto.Why.gave_up, e.failed.why);
        try testing.expectEqual(@as(u8, 3), e.code());
        try testing.expectEqual(@as(u64, 2), s.turns); // A_0's and B_0's
        try testing.expectEqual(@as(u64, 0), s.forwards);
        try testing.expectEqual(2 * layers, x.seen.items.len); // the loop ran the pair's 84 computes, posting none after 7
    }
}

test "qualification_c533375f: a pair stopped early with two in flight (A_0 .. A_3 sent, each before the request ahead of it was taken), then the attention gone: each half's next take gives up (300 ms), it exits 3, nothing is posted after; never a hang" {
    var h: Harness = .{ .io = std.testing.io, .pairs = true, .inflight = 2 };
    try h.start(300 * ms);
    defer h.free();
    const wrong = try sendPair2(&h, .{ .fseq = 1, .rows = 32, .seed = 1 }, .{ .fseq = 2, .rows = 17, .seed = 2 }, layers, 0, 7); // A_0 .. A_3
    loop.settle();
    h.attnGone();
    h.stop();
    try testing.expectEqual(@as(u64, 0), wrong);
    try testing.expectEqual([2]u64{ 7, 7 }, h.posted);
    for (h.ends, h.stats, h.experts) |e, s, x| {
        try testing.expect(e == .failed); // the take of B_3 gave up; every later one returned at once
        try testing.expectEqual(proto.Why.gave_up, e.failed.why);
        try testing.expectEqual(@as(u8, 3), e.code());
        try testing.expectEqual(@as(u64, 2), s.turns);
        try testing.expectEqual(@as(u64, 0), s.forwards);
        try testing.expectEqual(2 * layers, x.seen.items.len);
    }
}

test "--fault exit at B_0 and silent at B_3 inside a pair: the requests before it answered, counted as the attention sends them; the attention fails on it (dead, silent); the other half exits 3" {
    {
        // request 44 = the pair's B_0 (a solo forward's 42 first): expert 0 answers A_0 (43), then dies once 44 lands
        var h: Harness = .{ .io = std.testing.io, .pairs = true, .fault = .{ .{ .what = .exit, .at = layers + 2 }, null } };
        try h.start(300 * ms);
        defer h.free();
        const wrong = try sendSolo(&h, .{ .fseq = 1, .rows = 4, .seed = 1 }, .eager, 0, layers, 0);
        _ = try sendPair(&h, .{ .fseq = 2, .rows = 32, .seed = 2 }, .{ .fseq = 3, .rows = 17, .seed = 3 }, layers, 0, 2 * layers, null);
        loop.settle();
        const failed = h.attn().failure();
        h.attnGone();
        h.stop();
        try testing.expectEqual(@as(u64, 0), wrong);
        try testing.expectEqual(@as(?proto.Why, .dead), why(failed));
        try testing.expectEqual(@as(?[2]u32, .{ 0, layers + 2 }), where(failed));
        try testing.expectEqual(ae.End.die, h.ends[0]);
        try testing.expectEqual(ae.Stats{ .turns = 2, .forwards = 1, .layers = layers + 1, .rows = 4 * layers + 32 }, h.stats[0]);
        try testing.expectEqual([2]u64{ layers + 1, layers + 2 }, h.posted); // nothing from expert 0 for B_0
        try testing.expectEqual(@as(u8, 3), h.ends[1].code());
    }
    {
        // the pair's 8th request (B_3, seq 8): expert 1 answers 7, then nothing, its link and heartbeat up
        var h: Harness = .{ .io = std.testing.io, .pairs = true, .fault = .{ null, .{ .what = .silent, .at = 8 } }, .silent_ns = 300 * ms };
        try h.start(2 * std.time.ns_per_s);
        defer h.free();
        const wrong = try sendPair(&h, .{ .fseq = 1, .rows = 32, .seed = 1 }, .{ .fseq = 2, .rows = 17, .seed = 2 }, layers, 0, 2 * layers, null);
        const failed = h.attn().failure();
        loop.settle();
        h.attnGone();
        h.stop();
        try testing.expectEqual(@as(u64, 0), wrong);
        try testing.expectEqual(@as(?proto.Why, .silent), why(failed));
        try testing.expectEqual(@as(?[2]u32, .{ 1, 8 }), where(failed));
        try testing.expect(h.ends[1] == .lost); // idle until the attention's EOF
        try testing.expectEqual(ae.Stats{ .turns = 2, .forwards = 0, .layers = 7, .rows = 4 * 32 + 3 * 17 }, h.stats[1]);
        try testing.expectEqual([2]u64{ 8, 7 }, h.posted);
        try testing.expectEqual(@as(u8, 3), h.ends[0].code());
    }
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "the halves' TF_AFD_PAIRS: unset or 0 off, 1 on, anything else refuses the run (BadAfdPairs)" {
    defer _ = unsetenv("TF_AFD_PAIRS");
    _ = unsetenv("TF_AFD_PAIRS");
    try testing.expect(!try ae.pairs.fromEnv());
    _ = setenv("TF_AFD_PAIRS", "0", 1);
    try testing.expect(!try ae.pairs.fromEnv());
    _ = setenv("TF_AFD_PAIRS", "1", 1);
    try testing.expect(try ae.pairs.fromEnv());
    _ = setenv("TF_AFD_PAIRS", "2", 1);
    try testing.expectError(error.BadAfdPairs, ae.pairs.fromEnv());
}
