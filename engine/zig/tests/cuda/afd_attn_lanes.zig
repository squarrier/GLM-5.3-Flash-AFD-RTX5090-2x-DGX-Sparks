//! qualification_9e49592e: the attention's lanes (cuda_afd_lanes.zig, TF_AFD_LANES=N) against the halves' lane groups (afd_expert's
//! Opts.lanes) on afd_attn_fail.zig's three nodes in one process: the attention's forward recorded, its exchanges real
//! (fn_xchg's kernels on CPU threads over a region of N slots a source), each half qualification_cec3c04c's loop over the lab's stand-in.
//!   - G-trace-lanes, the 3-node form: a 4-chunk prompt (232 tokens: 64 x 3 + 40) and 3 decode steps with four lanes
//!     (one group of four) and three (a group of three, then 40 alone), against the same run without lanes: the tokens,
//!     every request past its header and both halves' replies equal by window and position (each chunk's launches:
//!     cuda_afd_lanes_test.zig's G-trace-lanes over the stand-in exchange); every reply exact; the audit clean with
//!     each window checked as a forward of its own; at most N in flight;
//!   - a dead half and a silent one inside a group of three: the prefill fails on that request before its first token
//!     (#214: never a hang), nothing is sent after, every later call refused at once with nothing launched or sent.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const fail = @import("afd_attn_fail.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const ae = glm.afd_expert;
const ag = glm.afd_engine;
const testing = std.testing;
const gpa = testing.allocator;
const ms = std.time.ns_per_ms;
const layers = fail.layers;

fn prompt(comptime n: usize) [n]u32 {
    var v: [n]u32 = undefined;
    for (&v, 0..) |*x, i| x.* = @intCast(300 + (7 * i) % 500);
    return v;
}

fn gTrace(n: u8) !void {
    var solo: fail.Harness = .{ .io = testing.io, .fault = fail.no_fault, .identity = 1 };
    try solo.start();
    defer solo.free();
    var lane: fail.Harness = .{ .io = testing.io, .fault = fail.no_fault, .pairs = true, .lanes = n, .identity = n };
    try lane.start();
    defer lane.free();
    var ms_solo: std.ArrayList(fail.Msg) = .empty;
    defer ms_solo.deinit(gpa);
    var ms_lane: std.ArrayList(fail.Msg) = .empty;
    defer ms_lane.deinit(gpa);
    solo.ah.msgs = &ms_solo;
    lane.ah.msgs = &ms_lane;
    const p = prompt(232);
    var fs = solo.fwd();
    var ts = try solo.at.prefill(&fs, &p);
    try fs.flush();
    var fl = lane.fwd();
    var tl = try lane.at.prefill(&fl, &p);
    try fl.flush();
    try testing.expectEqual(ts, tl);
    const launches = .{ solo.t.events.items.len, lane.t.events.items.len }; // the prefills'
    for (0..3) |_| {
        fs = solo.fwd();
        ts = try solo.at.step(&fs, ts);
        fl = lane.fwd();
        tl = try lane.at.step(&fl, tl);
        try testing.expectEqual(ts, tl);
    }
    fail.settle();
    const cs = solo.counts();
    const cl = lane.counts();
    const failed = .{ solo.attn().failure(), lane.attn().failure() };
    solo.stop();
    lane.stop();
    try testing.expectEqual(@as(?xchg.Failure, null), failed[0]);
    try testing.expectEqual(@as(?xchg.Failure, null), failed[1]);
    try testing.expectEqual(@as(u64, 0), lane.ah.wrong);
    try testing.expectEqual(@as(u32, 7 * layers), cl.rung);
    try testing.expectEqual(cs, cl);
    try testing.expectEqual(solo.posted, lane.posted);
    for (lane.ends, lane.stats, solo.stats) |e, s, s0| {
        try testing.expectEqual(ae.End.stopped, e);
        try testing.expectEqual(s0.layers, s.layers);
        try testing.expectEqual(s0.rows, s.rows);
        try testing.expectEqual(s0.forwards, s.forwards);
    }
    try testing.expectEqual(solo.e.st.pos, lane.e.st.pos);
    try testing.expectEqual(@as(u64, 1), lane.ln.runs);
    // the messages: by window and position, the lanes' run's equal the run's without (one request and two replies each)
    try testing.expectEqual(@as(usize, 3 * 7 * layers), ms_lane.items.len);
    try testing.expectEqual(ms_solo.items.len, ms_lane.items.len);
    std.mem.sort(fail.Msg, ms_solo.items, {}, fail.Msg.less);
    std.mem.sort(fail.Msg, ms_lane.items, {}, fail.Msg.less);
    for (ms_solo.items, ms_lane.items) |a, b| {
        try testing.expectEqual(a.fseq, b.fseq);
        try testing.expectEqual(a.pos, b.pos);
        try testing.expectEqual(a.what, b.what);
        try testing.expectEqualSlices(u8, &a.sha, &b.sha);
    }
    var buf: [256]u8 = undefined;
    const line = lane.win.line(&buf);
    var tb: [256]u8 = undefined;
    const text = try lane.attn().cfg.settings.text(&tb);
    std.debug.print("G-trace-lanes (3 nodes, N={d}): the prefill {d} launches without lanes, {d} in lanes ({d} position fills); {d} messages equal by window and position; {s}; settings \"{s}\"\n", .{ n, launches[0], launches[1], launches[1] - launches[0], ms_lane.items.len, line, text });
    try testing.expect(std.mem.startsWith(u8, line, "afd-check: window complete, 7 forwards, 0 mismatches"));
    var want: [64]u8 = undefined;
    try testing.expect(std.mem.endsWith(u8, text, try std.fmt.bufPrint(&want, " pairs=1 inflight={d} lanes={d}", .{ n, n })));
    try testing.expectEqual(@as(usize, n), lane.ah.most); // N in flight, never N + 1
    try testing.expectEqual(@as(usize, 1), solo.ah.most);
    try testing.expectEqual(@as(usize, n - 1), launches[1] - launches[0]); // lanes 1..N-1's position fills (a lone chunk adds none)
}

test "qualification_9e49592e: G-trace-lanes, the attention's 3-node form: 232 tokens in 64-row chunks and 3 decode steps, four lanes (one group) and three (a group, then 40 alone) against the run without lanes: tokens, requests past the header and both halves' replies equal by window and position; every reply exact; the audit clean, a window a forward; at most N in flight" {
    try gTrace(4);
    try gTrace(3);
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

test "qualification_9e49592e #214: a dead half inside a group of three (expert 0 exits at request 2, lane 1's first; the three firsts in flight) and a silent one (expert 1 from request 6, the group's last): the prefill fails on that request before its first token, never a hang; nothing is sent after; every later call refused at once" {
    const p = prompt(232);
    { // dead at lane 1's first: requests 1-3 sent together, lane 0's second after its take unless the watcher saw the death first
        var h: fail.Harness = .{ .io = testing.io, .fault = .{ .{ .what = .exit, .at = 2 }, null }, .pairs = true, .lanes = 3 };
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
        std.debug.print("lanes: dead half: {s} in {d} ms\n", .{ ag.describe(failed, &buf), ns / ms });
        h.stop();
        try testing.expectEqual(proto.Why.dead, failed.why);
        try testing.expectEqual(@as(u32, 0), failed.peer);
        try testing.expect(c.rung == 3 or c.rung == 4); // nothing after lane 0's second either way
        try testing.expect(failed.seq >= 2 and failed.seq <= c.rung);
        try testing.expect(ns < 1000 * ms);
        try testing.expectEqual(@as(usize, 0), h.e.st.pos); // the group committed nothing
        try testing.expectEqual([2]u64{ c.rung, c.rung }, c.requests);
        try testing.expectEqual(ae.End.die, h.ends[0]);
        try testing.expectEqual(@as(u64, 1), h.posted[0]); // lane 0's first only
        try testing.expect(h.posted[1] >= 2 and h.posted[1] <= c.rung);
        try testing.expectEqual(@as(usize, 3), h.ah.most); // the three firsts together
        try testing.expectEqual(@as(u64, 0), h.ah.wrong);
    }
    { // silent from request 6 (lane 2's second, the group's last): the doorbell stays at 6, the watcher fails it at ~300 ms
        var h: fail.Harness = .{ .io = testing.io, .fault = .{ null, .{ .what = .silent, .at = 6 } }, .pairs = true, .lanes = 3 };
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
        std.debug.print("lanes: silent half: {s} in {d} ms\n", .{ ag.describe(failed, &buf), ns / ms });
        h.stop();
        try testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 6, .gave_up = 0, .breach = 0 }, failed);
        try testing.expect(ns >= 300 * ms and ns < 2000 * ms);
        try testing.expectEqual(@as(usize, 0), h.e.st.pos);
        try testing.expectEqual(@as(u32, 6), c.rung);
        try testing.expectEqual([2]u64{ 6, 6 }, c.requests);
        try testing.expectEqual([2]u64{ 6, 5 }, h.posted);
        try testing.expectEqual(ae.End.stopped, h.ends[1]); // silent until the attention's stop frame
        try testing.expectEqual(@as(usize, 3), h.ah.most);
        try testing.expectEqual(@as(u64, 0), h.ah.wrong);
    }
}
