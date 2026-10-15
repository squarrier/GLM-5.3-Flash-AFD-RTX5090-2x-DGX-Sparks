//! qualification_c533375f's checks of the prompt chunk pairs with two requests in flight (TF_AFD_INFLIGHT=2, evidence/qualification_0f991257/WIRE.md §12)
//! on the AFD attention node on a CPU, over qualification_0f991257's recorder and stand-in exchange (cuda_afd_pairs_test.zig's Rec): the
//! same requests and words as one in flight, each window's next request posted before the other window's take (two in
//! flight, never three), each chunk's launches equal the solo prefill's (B's shared-expert rows its own), the order read
//! back from the record, G-cross (what crosses a step is each window's own; the gate fails on a shared buffer), a pair
//! right behind a pair (its B_0 after A_0's take), and the refusals (a pair stopped early, a failed exchange).
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("cuda_fwd.zig");
const rec = @import("cuda_afd_pairs_rec.zig");
const pt = @import("cuda_afd_pairs_test.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const testing = std.testing;
const gpa = testing.allocator;
const Rec = pt.Rec;
const M = pt.M;
const head = pt.head;

/// One pair of the cut model (dense layers 0-2, MoE layers 3 and 4: positions 0 and 1) with two in flight: B_0 with
/// A_0, then each window's next request before the other's take (35 steps, the absorbs and commits included).
const two = [_]M{
    .{ .who = .a, .step = .stage, .pos = 0 },   .{ .who = .a, .step = .front, .pos = 0 },   .{ .who = .a, .step = .front, .pos = 1 },
    .{ .who = .a, .step = .front, .pos = 2 },   .{ .who = .a, .step = .front, .pos = 3 },   .{ .who = .a, .step = .route, .pos = 0 },
    .{ .who = .a, .step = .request, .pos = 0 }, .{ .who = .pair, .step = .position, .pos = 0 }, .{ .who = .b, .step = .stage, .pos = 0 },
    .{ .who = .b, .step = .front, .pos = 0 },   .{ .who = .b, .step = .front, .pos = 1 },   .{ .who = .b, .step = .front, .pos = 2 },
    .{ .who = .b, .step = .front, .pos = 3 },   .{ .who = .b, .step = .route, .pos = 0 },   .{ .who = .b, .step = .request, .pos = 0 },
    .{ .who = .a, .step = .take, .pos = 0 },    .{ .who = .a, .step = .post, .pos = 0 },    .{ .who = .a, .step = .front, .pos = 4 },
    .{ .who = .a, .step = .route, .pos = 1 },   .{ .who = .a, .step = .request, .pos = 1 }, .{ .who = .b, .step = .take, .pos = 0 },
    .{ .who = .b, .step = .post, .pos = 0 },    .{ .who = .b, .step = .front, .pos = 4 },   .{ .who = .b, .step = .route, .pos = 1 },
    .{ .who = .b, .step = .request, .pos = 1 }, .{ .who = .a, .step = .take, .pos = 1 },    .{ .who = .a, .step = .post, .pos = 1 },
    .{ .who = .a, .step = .tail, .pos = 0 },    .{ .who = .b, .step = .take, .pos = 1 },    .{ .who = .b, .step = .post, .pos = 1 },
    .{ .who = .b, .step = .tail, .pos = 0 },    .{ .who = .a, .step = .absorb, .pos = 0 },  .{ .who = .a, .step = .commit, .pos = 0 },
    .{ .who = .b, .step = .absorb, .pos = 0 },  .{ .who = .b, .step = .commit, .pos = 0 },
};

/// The same pair right behind a pair (nothing between on the wire): B_0 after A_0's take, then two in flight.
const behind = [_]M{
    .{ .who = .a, .step = .stage, .pos = 0 },   .{ .who = .a, .step = .front, .pos = 0 },   .{ .who = .a, .step = .front, .pos = 1 },
    .{ .who = .a, .step = .front, .pos = 2 },   .{ .who = .a, .step = .front, .pos = 3 },   .{ .who = .a, .step = .route, .pos = 0 },
    .{ .who = .a, .step = .request, .pos = 0 }, .{ .who = .pair, .step = .position, .pos = 0 }, .{ .who = .b, .step = .stage, .pos = 0 },
    .{ .who = .b, .step = .front, .pos = 0 },   .{ .who = .b, .step = .front, .pos = 1 },   .{ .who = .b, .step = .front, .pos = 2 },
    .{ .who = .b, .step = .front, .pos = 3 },   .{ .who = .b, .step = .route, .pos = 0 },   .{ .who = .a, .step = .take, .pos = 0 },
    .{ .who = .b, .step = .request, .pos = 0 }, .{ .who = .a, .step = .post, .pos = 0 },    .{ .who = .a, .step = .front, .pos = 4 },
    .{ .who = .a, .step = .route, .pos = 1 },   .{ .who = .a, .step = .request, .pos = 1 }, .{ .who = .b, .step = .take, .pos = 0 },
    .{ .who = .b, .step = .post, .pos = 0 },    .{ .who = .b, .step = .front, .pos = 4 },   .{ .who = .b, .step = .route, .pos = 1 },
    .{ .who = .b, .step = .request, .pos = 1 }, .{ .who = .a, .step = .take, .pos = 1 },    .{ .who = .a, .step = .post, .pos = 1 },
    .{ .who = .a, .step = .tail, .pos = 0 },    .{ .who = .b, .step = .take, .pos = 1 },    .{ .who = .b, .step = .post, .pos = 1 },
    .{ .who = .b, .step = .tail, .pos = 0 },    .{ .who = .a, .step = .absorb, .pos = 0 },  .{ .who = .a, .step = .commit, .pos = 0 },
    .{ .who = .b, .step = .absorb, .pos = 0 },  .{ .who = .b, .step = .commit, .pos = 0 },
};

/// The solo prefill (x) and the paired one with two in flight (y), as qualification_0f991257's G-trace-pairs runs them.
fn twoInFlight(x: *Rec, y: *Rec, mtp: bool, capacity: usize, rows: usize, prompt: []const u32, chunks: usize, log: *rec.Log) !void {
    try x.initWith(mtp, capacity, rows);
    errdefer x.deinit();
    try y.initWith(mtp, capacity, rows);
    errdefer y.deinit();
    y.inflight = 2;
    try pt.traceEqual(x, y, prompt, chunks, log);
    try testing.expectEqual(@as(usize, 2), y.s.most); // two in flight, never three
    try testing.expectEqual(@as(usize, 1), x.s.most);
}

/// The same headers as the solo run's, by forward and position, once mode and the pair words are set aside.
fn sameRequests(x: *const Rec, y: *const Rec) !void {
    try testing.expectEqual(x.s.posts, y.s.posts);
    for (x.s.hdrs[0..x.s.posts]) |h| {
        var found = false;
        for (y.s.hdrs[0..y.s.posts]) |g| found = found or (g.fseq == h.fseq and g.layer == h.layer and g.rows == h.rows and g.layers == h.layers);
        try testing.expect(found);
    }
}

test "qualification_c533375f two in flight: 158 tokens in 64-row chunks (a pair, then 30 alone): one in flight's requests and words; B_0 with A_0, A_1 before B_0's take; each chunk's launches equal the solo prefill's; the order and G-cross from the record" {
    var x: Rec = undefined;
    var y: Rec = undefined;
    var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
    defer log.deinit();
    const prompt = pt.ids(158);
    try twoInFlight(&x, &y, false, 2560, 64, &prompt, 3, &log);
    defer x.deinit();
    defer y.deinit();
    try pt.interleaveAs(&log, &x, &y, &two, 0);
    const want = [_]proto.Header{
        head(1, 64, 0, .pair, 2, .{ 0, 64, 1 }), head(2, 64, 0, .pair, 2, .{ 1, 64, 1 }), head(1, 64, 1, .pair, 2, .{ 0, 64, 1 }),
        head(2, 64, 1, .pair, 2, .{ 1, 64, 1 }), head(3, 30, 0, .chunk, 2, .{ 0, 0, 0 }), head(3, 30, 1, .chunk, 2, .{ 0, 0, 0 }),
    };
    try testing.expectEqual(want.len, y.s.posts);
    for (want, y.s.hdrs[0..want.len]) |w, g| try testing.expectEqual(w, g);
    // the takes before each post: B_0 with A_0 (0, 0), A_1 after one take, B_1 after two; the lone chunk as today
    try testing.expectEqualSlices(usize, &.{ 0, 0, 1, 2, 4, 5 }, y.s.before[0..want.len]);
    try testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3, 4, 5 }, x.s.before[0..want.len]);
    try sameRequests(&x, &y);
    try testing.expectEqual(@as(usize, 158), y.e.st.pos);
    try testing.expectEqual(@as(u64, 1), y.p.runs);
}

test "qualification_c533375f two in flight: two pairs right behind each other (64 + 64, 64 + 17 = 209 tokens, no MTP absorbs between): the second pair's B_0 waits for its A_0's take (the halves read the first pair's tail), then two in flight; launches chunk by chunk the solo's" {
    var x: Rec = undefined;
    var y: Rec = undefined;
    var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
    defer log.deinit();
    const prompt = pt.ids(209);
    try twoInFlight(&x, &y, false, 2560, 64, &prompt, 4, &log);
    defer x.deinit();
    defer y.deinit();
    try pt.interleaveAs(&log, &x, &y, &two, 0);
    try pt.interleaveAs(&log, &x, &y, &behind, two.len);
    try testing.expectEqual(@as(usize, 8), y.s.posts);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 1, 2, 4, 5, 5, 6 }, y.s.before[0..8]);
    try testing.expectEqual(head(3, 64, 0, .pair, 2, .{ 0, 17, 3 }), y.s.hdrs[4]);
    try testing.expectEqual(head(4, 17, 0, .pair, 2, .{ 1, 64, 3 }), y.s.hdrs[5]);
    try testing.expectEqual(head(3, 64, 1, .pair, 2, .{ 0, 17, 3 }), y.s.hdrs[6]);
    try testing.expectEqual(head(4, 17, 1, .pair, 2, .{ 1, 64, 3 }), y.s.hdrs[7]);
    try sameRequests(&x, &y);
    try testing.expectEqual(@as(u64, 2), y.p.runs);
}

test "qualification_c533375f two in flight with the MTP head (TF_AFD_MTP=1): 81 tokens (64 + 17): the absorbs after the pair in chunk order, word 6 the count; each chunk's launches (its absorb included) the solo's" {
    var x: Rec = undefined;
    var y: Rec = undefined;
    var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
    defer log.deinit();
    const prompt = pt.ids(81);
    try twoInFlight(&x, &y, true, 2560, 64, &prompt, 2, &log);
    defer x.deinit();
    defer y.deinit();
    try pt.interleaveAs(&log, &x, &y, &two, 0);
    const want = [_]proto.Header{
        head(1, 64, 0, .pair, 3, .{ 0, 17, 1 }), head(2, 17, 0, .pair, 3, .{ 1, 64, 1 }), head(1, 64, 1, .pair, 3, .{ 0, 17, 1 }),
        head(2, 17, 1, .pair, 3, .{ 1, 64, 1 }), head(3, 64, 2, .slice, 3, .{ 0, 0, 0 }), head(4, 16, 2, .slice, 3, .{ 0, 0, 0 }),
    };
    try testing.expectEqual(want.len, y.s.posts);
    for (want, y.s.hdrs[0..want.len]) |w, g| try testing.expectEqual(w, g);
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 2, 1, 1 }, y.s.counts[0..want.len]);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 1, 2, 4, 5 }, y.s.before[0..want.len]);
    try testing.expectEqual(@as(usize, 80), y.e.st.mtp_len);
}

test "qualification_c533375f G-trace-pairs with two in flight at 8,000 tokens in 2,048-row chunks (2,048 + 2,048, 2,048 + 1,856; capacity 8,256, the long context past the dense limit) with TF_AFD_MTP's absorbs: each chunk's launches equal the solo prefill's; the wire's words; both pairs two in flight (the absorbs between them)" {
    var x: Rec = undefined;
    var y: Rec = undefined;
    var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
    defer log.deinit();
    const prompt = try gpa.alloc(u32, 8000);
    defer gpa.free(prompt);
    for (prompt, 0..) |*v, i| v.* = @intCast(1000 + (7 * i) % 5000);
    try twoInFlight(&x, &y, true, 8256, 2048, prompt, 4, &log);
    defer x.deinit();
    defer y.deinit();
    try pt.interleaveAs(&log, &x, &y, &two, 0);
    try pt.interleaveAs(&log, &x, &y, &two, two.len); // the absorbs between the pairs: the second's B_0 with its A_0 too
    const want = [_]proto.Header{
        head(1, 2048, 0, .pair, 3, .{ 0, 2048, 1 }), head(2, 2048, 0, .pair, 3, .{ 1, 2048, 1 }), head(1, 2048, 1, .pair, 3, .{ 0, 2048, 1 }),
        head(2, 2048, 1, .pair, 3, .{ 1, 2048, 1 }), head(3, 2048, 2, .slice, 3, .{ 0, 0, 0 }),   head(4, 2048, 2, .slice, 3, .{ 0, 0, 0 }),
        head(5, 2048, 0, .pair, 3, .{ 0, 1856, 5 }), head(6, 1856, 0, .pair, 3, .{ 1, 2048, 5 }), head(5, 2048, 1, .pair, 3, .{ 0, 1856, 5 }),
        head(6, 1856, 1, .pair, 3, .{ 1, 2048, 5 }), head(7, 2048, 2, .slice, 3, .{ 0, 0, 0 }),   head(8, 1855, 2, .slice, 3, .{ 0, 0, 0 }),
    };
    try testing.expectEqual(want.len, y.s.posts);
    for (want, y.s.hdrs[0..want.len]) |w, g| try testing.expectEqual(w, g);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 1, 2, 4, 5, 6, 6, 7, 8, 10, 11 }, y.s.before[0..want.len]);
    try testing.expectEqual(@as(usize, 8000), y.e.st.pos);
    try testing.expectEqual(@as(usize, 7999), y.e.st.mtp_len);
    try testing.expectEqual(@as(u64, 2), y.p.runs);
    std.debug.print("G-trace-pairs two in flight (attention, 8,000 tokens in 2,048-row chunks, MTP): {d} launches solo, {d} paired; 4 chunks equal; 12 requests, at most 2 in flight; G-cross clean\n", .{ x.t.events.items.len, y.t.events.items.len });
}

/// A paired run of `prompt` with the hook's log, then G-cross over it (null: clean).
fn crossedRun(y: *Rec, log: *rec.Log, prompt: []const u32) !?usize {
    y.p.hook = log.hook();
    var f = fwd.Fwd.recording(&y.m, &y.t);
    _ = try y.at.prefill(&f, prompt);
    try f.flush();
    const sy_b = if (y.p.inflight > 1) "pair.sy" else "pbuf.sy";
    return log.crossed(&y.space, .{ "pbuf.sy", sy_b });
}

test "qualification_c533375f G-cross fails on a shared buffer: two in flight with B's shared-expert rows A's (A's next request rewrites them before B's take), and one in flight with B's streams A's; clean with the copies" {
    const prompt = pt.ids(158);
    { // two in flight, B's sy pointed at A's: the gate names B's request inside A's request-to-take window
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        y.inflight = 2;
        try y.start(true);
        y.p.c.sy = y.e.pbuf.sy;
        var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
        defer log.deinit();
        try testing.expect(try crossedRun(&y, &log, &prompt) != null);
    }
    { // one in flight, B's streams pointed at A's
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        try y.start(true);
        y.p.c.x = y.e.pbuf.x;
        var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
        defer log.deinit();
        try testing.expect(try crossedRun(&y, &log, &prompt) != null);
    }
    for ([_]u8{ 1, 2 }) |n| { // the copies as made: clean
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        y.inflight = n;
        try y.start(true);
        var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
        defer log.deinit();
        try testing.expectEqual(@as(?usize, null), try crossedRun(&y, &log, &prompt));
    }
}

test "qualification_c533375f refused with two in flight: not agreed (before any launch); a pair stopped early (a post's error at A_1, with A_0 and B_0 in flight) dies at the next begin, nothing more sent; a failed exchange (at B_0's take, A_1 already sent) fails the prefill before its first token, later calls refused at once" {
    const prompt = pt.ids(158);
    { // not agreed: refused before State.reset's launches, nothing posted
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        y.inflight = 2;
        try y.start(true);
        y.p.agreed = false;
        var f = fwd.Fwd.recording(&y.m, &y.t);
        try testing.expectError(error.NotPorted, y.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.t.events.items.len);
        try testing.expectEqual(@as(usize, 0), y.s.posts);
    }
    { // stopped between its exchanges: the next begin dies (stopped), every later call refuses, nothing more sent
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        y.inflight = 2;
        try y.start(true);
        y.s.err_post = 3;
        var f = fwd.Fwd.recording(&y.m, &y.t);
        try testing.expectError(error.HostPost, y.at.prefill(&f, &prompt));
        try testing.expectEqual(@as(usize, 2), y.s.posts); // A_0 and B_0, both in flight
        try testing.expectEqual(@as(usize, 1), y.s.takes); // A_0's
        try testing.expectError(error.OutOfStep, y.at.step(&f, 7));
        try testing.expectEqual(xchg.proto.Why.stopped, y.at.failure().?.why);
        y.t.reset();
        try testing.expectError(error.ExpertFailed, y.at.step(&f, 7));
        try testing.expectError(error.ExpertFailed, y.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.t.events.items.len);
        try testing.expectEqual(@as(usize, 2), y.s.posts);
    }
    { // a half fails at B_0's take (A_1 already posted: seq 3): the prefill fails at its check, no first token drawn
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        y.inflight = 2;
        try y.start(true);
        y.s.fail_take = 2;
        var f = fwd.Fwd.recording(&y.m, &y.t);
        try testing.expectError(error.ExpertFailed, y.at.prefill(&f, &prompt));
        try f.flush();
        for (y.t.events.items) |e| try testing.expect(!std.mem.eql(u8, e.name, "torch.topk"));
        try testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 3, .gave_up = 0, .breach = 0 }, y.at.failure().?);
        const posts = y.s.posts;
        y.t.reset();
        try testing.expectError(error.ExpertFailed, y.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.t.events.items.len);
        try testing.expectEqual(posts, y.s.posts);
    }
}
