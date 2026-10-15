//! qualification_0f991257: the attention's pair scheduler (cuda_afd_pairs.zig, TF_AFD_PAIRS=1) against the halves' pair loop (qualification_c2d4ee2e's
//! afd_expert.Opts.pairs) on afd_attn_fail.zig's three nodes in one process: the attention's forward recorded, its
//! exchanges real (fn_xchg's kernels on CPU threads), each half qualification_cec3c04c's loop over the lab's stand-in. Run where the tree
//! has both cards (the merge of qualification_2484d3f6 and qualification_90cdec31); a tree without the halves' pairs skips them.
//!   - G-trace-pairs, the 3-node form: a 4-chunk prompt (two pairs: 64 + 64, 64 + 40) and 3 decode steps, against the
//!     same run unpaired: each chunk's launches, its requests past the header and both halves' replies equal, by window
//!     and position; every reply exact; the audit clean with each window checked as a forward of its own;
//!   - a dead half and a silent half inside a pair: the prefill fails on the request (never a hang), nothing is sent
//!     after, every later call refused at once with nothing launched or sent.
//!   - qualification_c533375f: both again with two requests in flight a link (TF_AFD_INFLIGHT=2 on all three nodes), and G-cross.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const fail = @import("afd_attn_fail.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const ae = glm.afd_expert;
const ag = glm.afd_engine;
const rec = glm.afd_pairs_rec;
const testing = std.testing;
const gpa = testing.allocator;
const ms = std.time.ns_per_ms;
const layers = fail.layers;

/// The halves serve pairs in this tree (qualification_c2d4ee2e's loop); otherwise the gates skip.
const served = @hasField(ae.Opts, "pairs");

fn skip() error{SkipZigTest} {
    std.debug.print("SKIP qualification_0f991257 3-node pairs: this tree's halves serve no pairs (afd_expert.Opts.pairs is qualification_c2d4ee2e's, branch qualification_90cdec31)\n", .{});
    return error.SkipZigTest;
}

fn prompt(comptime n: usize) [n]u32 {
    var v: [n]u32 = undefined;
    for (&v, 0..) |*x, i| x.* = @intCast(300 + (7 * i) % 500);
    return v;
}

/// The paired prefill's launches chunk by chunk against the solo one's (the solo run cut at its commits; State.reset's
/// and the first token's launches too); the pairs' only additions their position fills.
fn sameLaunches(solo: *fail.Harness, paired: *fail.Harness, log: *const rec.Log, chunks: usize) !void {
    const ex = solo.t.events.items;
    const ey = paired.t.events.items;
    const reset = log.marks.items[0].at;
    try testing.expectEqualStrings(&rec.digest(&solo.space, ex[0..reset]), &rec.digest(&paired.space, ey[0..reset]));
    var ranges: [16][2]usize = undefined;
    try testing.expectEqual(chunks, rec.soloChunks(&solo.space, ex, reset, &ranges));
    for (ranges[0..chunks], 0..) |r, k| {
        const got = log.chunkDigest(&paired.space, k);
        try testing.expectEqual(r[1] - r[0], got.n);
        try testing.expectEqualStrings(&rec.digest(&solo.space, ex[r[0]..r[1]]), &got.hex);
    }
    const first = log.marks.items[log.marks.items.len - 1];
    try testing.expectEqualStrings(&rec.digest(&solo.space, ex[ranges[chunks - 1][1]..]), &rec.digest(&paired.space, ey[first.at..]));
    try testing.expectEqual(ex.len + log.own(), ey.len);
}

test "G-trace-pairs, the attention's 3-node form: 232 tokens in 64-row chunks (pairs 64 + 64, 64 + 40) and 3 decode steps: each chunk's launches, requests past the header and both halves' replies equal the unpaired run's; every reply exact; the audit clean, a window a forward" {
    if (!served) return skip();
    try gTrace(1);
}

test "qualification_c533375f: G-trace-pairs' 3-node form with two requests in flight (TF_AFD_INFLIGHT=2 on all three nodes): the same launches, messages and replies as the unpaired run's; at most two in flight; the audit clean; G-cross" {
    if (!served) return skip();
    try gTrace(2);
}

/// G-trace-pairs' 3-node form with `inflight` requests in flight a link (the unpaired run against the paired one).
fn gTrace(inflight: u8) !void {
    var solo: fail.Harness = .{ .io = testing.io, .fault = fail.no_fault };
    try solo.start();
    defer solo.free();
    var paired: fail.Harness = .{ .io = testing.io, .fault = fail.no_fault, .pairs = true, .inflight = inflight };
    try paired.start();
    defer paired.free();
    var ms_solo: std.ArrayList(fail.Msg) = .empty;
    defer ms_solo.deinit(gpa);
    var ms_pair: std.ArrayList(fail.Msg) = .empty;
    defer ms_pair.deinit(gpa);
    solo.ah.msgs = &ms_solo;
    paired.ah.msgs = &ms_pair;
    var log: rec.Log = .{ .t = &paired.t, .gpa = gpa };
    defer log.deinit();
    paired.pp.hook = log.hook();
    const p = prompt(232);
    var fs = solo.fwd();
    var ts = try solo.at.prefill(&fs, &p);
    try fs.flush();
    var fp = paired.fwd();
    var tp = try paired.at.prefill(&fp, &p);
    try fp.flush();
    try testing.expectEqual(ts, tp);
    try sameLaunches(&solo, &paired, &log, 4);
    const sy_b = if (inflight > 1) "pair.sy" else "pbuf.sy";
    try testing.expectEqual(@as(?usize, null), log.crossed(&paired.space, .{ "pbuf.sy", sy_b })); // qualification_c533375f's G-cross
    const launches = .{ solo.t.events.items.len, paired.t.events.items.len }; // the prefills'
    paired.pp.hook = null;
    for (0..3) |_| {
        fs = solo.fwd();
        ts = try solo.at.step(&fs, ts);
        fp = paired.fwd();
        tp = try paired.at.step(&fp, tp);
        try testing.expectEqual(ts, tp);
    }
    fail.settle();
    const cs = solo.counts();
    const cp = paired.counts();
    const failed = .{ solo.attn().failure(), paired.attn().failure() };
    solo.stop();
    paired.stop();
    try testing.expectEqual(@as(?xchg.Failure, null), failed[0]);
    try testing.expectEqual(@as(?xchg.Failure, null), failed[1]);
    try testing.expectEqual(@as(u64, 0), paired.ah.wrong);
    try testing.expectEqual(@as(u32, 7 * layers), cp.rung);
    try testing.expectEqual(cs, cp);
    try testing.expectEqual(solo.posted, paired.posted);
    for (paired.ends, paired.stats, solo.stats) |e, s, s0| {
        try testing.expectEqual(ae.End.stopped, e);
        try testing.expectEqual(s0.layers, s.layers);
        try testing.expectEqual(s0.rows, s.rows);
    }
    try testing.expectEqual(solo.e.st.pos, paired.e.st.pos);
    try testing.expectEqual(@as(u64, 2), paired.pp.runs); // (64, 64) and (64, 40)
    // the messages: by window and position, the paired run's equal the unpaired run's (one request and two replies each)
    try testing.expectEqual(@as(usize, 3 * 7 * layers), ms_pair.items.len);
    try testing.expectEqual(ms_solo.items.len, ms_pair.items.len);
    std.mem.sort(fail.Msg, ms_solo.items, {}, fail.Msg.less);
    std.mem.sort(fail.Msg, ms_pair.items, {}, fail.Msg.less);
    for (ms_solo.items, ms_pair.items) |a, b| {
        try testing.expectEqual(a.fseq, b.fseq);
        try testing.expectEqual(a.pos, b.pos);
        try testing.expectEqual(a.what, b.what);
        try testing.expectEqualSlices(u8, &a.sha, &b.sha);
    }
    var buf: [256]u8 = undefined;
    const line = paired.win.line(&buf);
    var tb: [256]u8 = undefined;
    const text = try paired.attn().cfg.settings.text(&tb);
    std.debug.print("G-trace-pairs (3 nodes, {d} in flight): the prefill {d} launches unpaired, {d} paired (2 pairs' position fills); {d} messages equal by window and position; {s}; settings \"{s}\"\n", .{ inflight, launches[0], launches[1], ms_pair.items.len, line, text });
    try testing.expect(std.mem.startsWith(u8, line, "afd-check: window complete, 7 forwards, 0 mismatches"));
    try testing.expect(std.mem.endsWith(u8, text, if (inflight > 1) " pairs=1 inflight=2" else " pairs=1"));
    try testing.expectEqual(@as(usize, inflight), paired.ah.most); // two in flight, never three (one: the pairs as qualification_0f991257)
    try testing.expectEqual(@as(usize, 1), solo.ah.most);
}

/// After the failure: `n` decode steps and a prompt, each refused at once with nothing launched or sent.
fn refusedAfter(h: *fail.Harness, n: usize) !void {
    const before = h.counts();
    const p = prompt(232);
    for (0..n + 1) |i| {
        var f = h.fwd();
        const t0 = h.attn().nowNs();
        const r = if (i < n) h.at.step(&f, 7) else h.at.prefill(&f, &p);
        try testing.expectError(error.ExpertFailed, r);
        try f.flush();
        try testing.expectEqual(@as(usize, 0), h.t.events.items.len);
        try testing.expect(h.attn().nowNs() - t0 < 50 * ms);
    }
    try testing.expectEqual(before, h.counts());
}

test "a dead half inside a pair (expert 0 exits at request 2, B_0) and a silent one (expert 1 from request 4, B_1): the prefill fails on that request before its first token, never a hang; nothing is sent after; every later call refused at once" {
    if (!served) return skip();
    const p = prompt(232);
    { // dead at B_0: requests A_0 and B_0 sent, A_1 never (it follows B_0's replies)
        var h: fail.Harness = .{ .io = testing.io, .fault = .{ .{ .what = .exit, .at = 2 }, null }, .pairs = true };
        try h.start();
        defer h.free();
        var f = h.fwd();
        const t0 = h.attn().nowNs();
        try testing.expectError(error.ExpertFailed, h.at.prefill(&f, &p));
        const ns = h.attn().nowNs() - t0;
        try f.flush();
        for (h.t.events.items) |e| try testing.expect(!std.mem.eql(u8, e.name, "torch.topk")); // no first token drawn
        const failed = h.at.failure().?;
        try refusedAfter(&h, 2);
        fail.settle();
        const c = h.counts();
        var buf: [128]u8 = undefined;
        std.debug.print("pairs: dead half: {s} in {d} ms\n", .{ ag.describe(failed, &buf), ns / ms });
        h.stop();
        try testing.expectEqual(xchg.Failure{ .why = .dead, .peer = 0, .seq = 2, .gave_up = 0, .breach = 0 }, failed);
        try testing.expect(ns < 1000 * ms);
        try testing.expectEqual(@as(usize, 0), h.e.st.pos); // the pair committed nothing
        try testing.expectEqual(@as(u32, 2), c.rung);
        try testing.expectEqual([2]u64{ 2, 2 }, c.requests);
        try testing.expectEqual(ae.End.die, h.ends[0]);
        try testing.expectEqual([2]u64{ 1, 2 }, h.posted); // expert 0 answered A_0; expert 1 both
        try testing.expectEqual(@as(u64, 0), h.ah.wrong);
    }
    { // silent from B_1: A_0, B_0, A_1 answered by both, B_1 by expert 0 only; the watcher fails it at ~300 ms
        var h: fail.Harness = .{ .io = testing.io, .fault = .{ null, .{ .what = .silent, .at = 4 } }, .pairs = true };
        try h.start();
        defer h.free();
        var f = h.fwd();
        const t0 = h.attn().nowNs();
        try testing.expectError(error.ExpertFailed, h.at.prefill(&f, &p));
        const ns = h.attn().nowNs() - t0;
        try f.flush();
        const failed = h.at.failure().?;
        try refusedAfter(&h, 2);
        fail.settle();
        const c = h.counts();
        var buf: [128]u8 = undefined;
        std.debug.print("pairs: silent half: {s} in {d} ms\n", .{ ag.describe(failed, &buf), ns / ms });
        h.stop();
        try testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 4, .gave_up = 0, .breach = 0 }, failed);
        try testing.expect(ns >= 300 * ms and ns < 2000 * ms);
        try testing.expectEqual(@as(usize, 0), h.e.st.pos);
        try testing.expectEqual(@as(u32, 4), c.rung);
        try testing.expectEqual([2]u64{ 4, 4 }, c.requests);
        try testing.expectEqual([2]u64{ 4, 3 }, h.posted);
        try testing.expectEqual(ae.End.stopped, h.ends[1]); // silent until the attention's stop frame
        try testing.expectEqual(@as(u64, 0), h.ah.wrong);
    }
}

test "qualification_c533375f: a dead half and a silent one with two in flight (TF_AFD_INFLIGHT=2): expert 0 exits at B_0 (A_1 may already be sent: it goes once A_0 is taken), expert 1 silent from B_1; the prefill fails on that request before its first token, never a hang; nothing is sent after; every later call refused at once" {
    if (!served) return skip();
    const p = prompt(232);
    { // dead at B_0: A_0 and B_0 sent together, A_1 after A_0's take unless the watcher saw the death first
        var h: fail.Harness = .{ .io = testing.io, .fault = .{ .{ .what = .exit, .at = 2 }, null }, .pairs = true, .inflight = 2 };
        try h.start();
        defer h.free();
        var f = h.fwd();
        const t0 = h.attn().nowNs();
        try testing.expectError(error.ExpertFailed, h.at.prefill(&f, &p));
        const ns = h.attn().nowNs() - t0;
        try f.flush();
        for (h.t.events.items) |e| try testing.expect(!std.mem.eql(u8, e.name, "torch.topk"));
        const failed = h.at.failure().?;
        try refusedAfter(&h, 2);
        fail.settle();
        const c = h.counts();
        var buf: [128]u8 = undefined;
        std.debug.print("pairs, two in flight: dead half: {s} in {d} ms; {d} rung, {d} most in flight\n", .{ ag.describe(failed, &buf), ns / ms, c.rung, h.ah.most });
        h.stop();
        try testing.expectEqual(proto.Why.dead, failed.why);
        try testing.expectEqual(@as(u32, 0), failed.peer);
        try testing.expect(c.rung == 2 or c.rung == 3); // nothing after A_1 either way
        try testing.expect(failed.seq >= 2 and failed.seq <= c.rung);
        try testing.expect(ns < 1000 * ms);
        try testing.expectEqual(@as(usize, 0), h.e.st.pos);
        try testing.expectEqual([2]u64{ c.rung, c.rung }, c.requests);
        try testing.expectEqual(ae.End.die, h.ends[0]);
        try testing.expectEqual(@as(u64, 1), h.posted[0]); // A_0 only
        try testing.expect(h.posted[1] >= 2 and h.posted[1] <= c.rung);
        try testing.expectEqual(@as(usize, 2), h.ah.most); // B_0 went with A_0
        try testing.expectEqual(@as(u64, 0), h.ah.wrong);
    }
    { // silent from B_1 (the pair's last request): the doorbell stays at 4, the watcher fails it at ~300 ms
        var h: fail.Harness = .{ .io = testing.io, .fault = .{ null, .{ .what = .silent, .at = 4 } }, .pairs = true, .inflight = 2 };
        try h.start();
        defer h.free();
        var f = h.fwd();
        const t0 = h.attn().nowNs();
        try testing.expectError(error.ExpertFailed, h.at.prefill(&f, &p));
        const ns = h.attn().nowNs() - t0;
        try f.flush();
        const failed = h.at.failure().?;
        try refusedAfter(&h, 2);
        fail.settle();
        const c = h.counts();
        var buf: [128]u8 = undefined;
        std.debug.print("pairs, two in flight: silent half: {s} in {d} ms\n", .{ ag.describe(failed, &buf), ns / ms });
        h.stop();
        try testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 4, .gave_up = 0, .breach = 0 }, failed);
        try testing.expect(ns >= 300 * ms and ns < 2000 * ms);
        try testing.expectEqual(@as(usize, 0), h.e.st.pos);
        try testing.expectEqual(@as(u32, 4), c.rung);
        try testing.expectEqual([2]u64{ 4, 4 }, c.requests);
        try testing.expectEqual([2]u64{ 4, 3 }, h.posted);
        try testing.expectEqual(ae.End.stopped, h.ends[1]);
        try testing.expectEqual(@as(usize, 2), h.ah.most);
        try testing.expectEqual(@as(u64, 0), h.ah.wrong);
    }
}
