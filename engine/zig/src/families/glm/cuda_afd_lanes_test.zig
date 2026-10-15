//! qualification_9e49592e's checks of the prompt chunk lanes (TF_AFD_LANES=N, evidence/qualification_0f991257/WIRE.md §13) on the AFD attention node on a
//! CPU, over qualification_0f991257's recorder and stand-in exchange (cuda_afd_pairs_test.zig's Rec, its node over a region of N slots a
//! source): a group's requests in the wire's order (lane 0's position p, lane 1's, ..., then p + 1) with their words, up
//! to N in flight and never N + 1; each chunk's launches equal the solo prefill's (G-trace-lanes, at 8,000 tokens too);
//! python 0026's order read back from the record; G-cross over N lanes; a group right behind a group (its request
//! N - 1 after its first take); a remainder of two as a pair's group, a lone last chunk alone; and the refusals (not
//! agreed, a group stopped early, a failed exchange). qualification_47b9bf3a's MTP absorbs in-lane: cuda_afd_lanes_mtp_test.zig.
const std = @import("std");
const cuda = @import("cuda");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const pairs = @import("cuda_afd_pairs.zig");
const lanes = @import("cuda_afd_lanes.zig");
const rec = @import("cuda_afd_pairs_rec.zig");
const pt = @import("cuda_afd_pairs_test.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const testing = std.testing;
const gpa = testing.allocator;
const Rec = pt.Rec;

/// pt.Rec with TF_AFD_LANES=N: its node's settings carry pairs=1 inflight=N lanes=N (a region of N slots a source), the
/// lanes agreed at `start` and their copies placed last (every other buffer keeps the solo run's address).
pub const LRec = struct {
    r: Rec,
    l: lanes.Lanes,
    n: u8,

    pub fn init(y: *LRec, n: u8, mtp: bool, capacity: usize, rows: usize) !void {
        y.n = n;
        try y.r.initWith(mtp, capacity, rows);
        const settings: proto.Settings = .{ .layout = y.r.a.layout, .max_rows = rows, .ports = 1, .pairs = true, .inflight = n, .lanes = n };
        const region = gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(settings.region().total())) catch |err| {
            y.r.deinit();
            return err;
        };
        @memset(region, 0);
        y.r.node.deinit();
        gpa.free(y.r.region);
        y.r.region = region;
        const mem: xchg.Memory = .{ .region = region, .region_dev = @intFromPtr(region.ptr), .status = y.r.status, .status_dev = @intFromPtr(y.r.status.ptr) };
        y.r.node = try xchg.Node.init(gpa, testing.io, .attn, .{ .settings = settings, .warn_failures = true }, mem);
        try testing.expectEqual(@as(u32, n), y.r.node.reg.slots);
    }

    pub fn start(y: *LRec) !void {
        const x = &y.r;
        try x.start(false);
        x.a.paired = true;
        x.a.inflight = y.n;
        x.a.lanes = y.n;
        y.l = try lanes.Lanes.init(&x.at);
        y.l.agreed = true;
        try y.l.walk(&x.space, trace.placeAll);
        x.at.lanes = &y.l;
    }

    pub fn deinit(y: *LRec) void {
        y.r.deinit();
    }
};

/// The lanes' hook as a test's sink (rec.Log's for N lanes): every step (who, lane, what, position) in order, where
/// its launches start, and the chunk it belongs to (a group's lanes, then a lone chunk, in prompt order; a later
/// lane's position fill and the first token belong to none).
pub const Log = struct {
    t: *trace.Trace,
    marks: std.ArrayList(Mark) = .empty,
    base: usize = 0,
    seen: usize = 0,

    const Mark = struct { who: lanes.Who, k: u8, step: pairs.Step, pos: usize, at: usize, chunk: ?usize };

    pub fn deinit(l: *Log) void {
        l.marks.deinit(gpa);
    }

    fn hook(l: *Log) lanes.Hook {
        return .{ .ctx = l, .at = at };
    }

    fn at(ctx: *anyopaque, who: lanes.Who, k: u8, step: pairs.Step, pos: usize) anyerror!void {
        const l: *Log = @ptrCast(@alignCast(ctx));
        try l.t.flush();
        if (who == .lane and k == 0 and step == .stage) l.base = l.seen;
        const chunk: ?usize = switch (who) {
            .lane => l.base + k,
            .solo => l.seen,
            .group => null,
        };
        if (chunk) |c| l.seen = @max(l.seen, c + 1);
        try l.marks.append(gpa, .{ .who = who, .k = k, .step = step, .pos = pos, .at = l.t.events.items.len, .chunk = chunk });
    }

    /// Mark i's launches: from its start to the next mark's.
    fn span(l: *const Log, i: usize) []const trace.Event {
        const ev = l.t.events.items;
        const m = l.marks.items;
        return ev[m[i].at..if (i + 1 < m.len) m[i + 1].at else ev.len];
    }

    /// Chunk k's launches, in order (shallow copies; the caller frees the slice).
    fn chunkEvents(l: *const Log, k: usize) ![]trace.Event {
        var out: std.ArrayList(trace.Event) = .empty;
        errdefer out.deinit(gpa);
        for (l.marks.items, 0..) |m, i| if (m.chunk == k) try out.appendSlice(gpa, l.span(i));
        return out.toOwnedSlice(gpa);
    }

    /// The launches of the groups' own steps (each later lane's position fill).
    fn own(l: *const Log) usize {
        var n: usize = 0;
        for (l.marks.items, 0..) |m, i| if (m.who == .group and m.step == .position) {
            n += l.span(i).len;
        };
        return n;
    }

    /// G-cross over `n` lanes (rec.Log.crossed's two rules, qualification_c533375f): no launch of a lane's steps (stage to tail) names
    /// another lane's crossing buffer (lane 0's the prompt Buffers' and the State's; lane k's its copies), and between a
    /// lane's request and its take no other lane's launch names that lane's shared-expert rows. Gives the first launch
    /// that does (its index in the record), or null.
    fn crossed(l: *const Log, space: *const trace.Space, n: usize) ?usize {
        const marks = l.marks.items;
        for (marks, 0..) |m, i| {
            if (m.who != .lane or m.step == .absorb or m.step == .commit) continue;
            for (0..n) |j| {
                if (j == m.k) continue;
                for (rec.crossing) |c| {
                    var b: [32]u8 = undefined;
                    const name = laneName(&b, j, c);
                    for (l.span(i), m.at..) |e, x| if (rec.names(space, e, name)) return x;
                }
            }
        }
        for (marks, 0..) |m, i| {
            if (m.who != .lane or m.step != .request) continue;
            var j = i + 1;
            while (j < marks.len and !(marks[j].who == .lane and marks[j].k == m.k and marks[j].step == .take and marks[j].pos == m.pos)) : (j += 1) {}
            var b: [32]u8 = undefined;
            const sy = laneName(&b, m.k, .{ "pbuf.sy", "pair.sy" });
            for (marks[i + 1 .. j], i + 1..) |o, oi| {
                if (o.who != .lane or o.k == m.k) continue;
                for (l.span(oi), o.at..) |e, x| if (rec.names(space, e, sy)) return x;
            }
        }
        return null;
    }
};

/// Lane k's buffer for a crossing pair (rec.crossing's: lane 0's original, else `lane{k}.` and the copy's field).
fn laneName(buf: *[32]u8, k: usize, c: [2][]const u8) []const u8 {
    if (k == 0) return c[0];
    return std.fmt.bufPrint(buf, "lane{d}.{s}", .{ k, c[1]["pair.".len..] }) catch unreachable;
}

/// The solo prefill (x) and the lanes' (y) of `prompt` in two fresh recorders, as pt.traceEqual for the pairs: every
/// chunk's launches (a group's lanes by the hook, the solo run's cut at its commits) digest equal by name, State.reset's
/// and the first token's too, and the lanes add only each later lane's position fills (`own` of them: its pos_dev's,
/// and qualification_47b9bf3a's mtp_pos_dev's for a lane whose absorb is its extra layer); G-cross clean. `absorb` false: the request
/// does not absorb into the MTP head (drafts off), on both runs.
fn traceEqual(x: *Rec, y: *LRec, prompt: []const u32, chunks: usize, own: usize, log: *Log, absorb: bool) !void {
    try x.start(false);
    x.at.absorb = x.at.absorb and absorb;
    var fx = fwd.Fwd.recording(&x.m, &x.t);
    const tok_x = try x.at.prefill(&fx, prompt);
    try fx.flush();
    try y.start();
    y.r.at.absorb = y.r.at.absorb and absorb;
    y.l.hook = log.hook();
    var fy = fwd.Fwd.recording(&y.r.m, &y.r.t);
    const tok_y = try y.r.at.prefill(&fy, prompt);
    try fy.flush();
    try testing.expectEqual(tok_x, tok_y);
    const ex = x.t.events.items;
    const ey = y.r.t.events.items;
    const reset = log.marks.items[0].at; // State.reset's launches: the lanes' run's before its first step
    try testing.expect(reset > 0 and reset <= ex.len);
    try testing.expectEqualStrings(&rec.digest(&x.space, ex[0..reset]), &rec.digest(&y.r.space, ey[0..reset]));
    var ranges: [16][2]usize = undefined;
    try testing.expectEqual(chunks, rec.soloChunks(&x.space, ex, reset, &ranges));
    for (ranges[0..chunks], 0..) |r, k| {
        const want = rec.digest(&x.space, ex[r[0]..r[1]]);
        const evs = try log.chunkEvents(k);
        defer gpa.free(evs);
        const got = rec.digest(&y.r.space, evs);
        if (r[1] - r[0] != evs.len or !std.mem.eql(u8, &want, &got)) std.debug.print("chunk {d}: {d} launches solo, {d} in lanes\n", .{ k, r[1] - r[0], evs.len });
        try testing.expectEqual(r[1] - r[0], evs.len);
        try testing.expectEqualStrings(&want, &got);
    }
    const first = log.marks.items[log.marks.items.len - 1];
    try testing.expect(first.who == .group and first.step == .first);
    try testing.expectEqualStrings(&rec.digest(&x.space, ex[ranges[chunks - 1][1]..]), &rec.digest(&y.r.space, ey[first.at..])); // the first token
    try testing.expectEqual(own, log.own()); // the later lanes' position fills, nothing else added
    try testing.expectEqual(ex.len + own, ey.len);
    try testing.expectEqual(x.e.st.pos, y.r.e.st.pos);
    try testing.expectEqual(x.e.st.parity, y.r.e.st.parity);
    try testing.expectEqual(x.e.st.mtp_len, y.r.e.st.mtp_len);
    if (log.crossed(&y.r.space, y.n)) |k| {
        std.debug.print("G-cross: launch {d} ({s}) names a buffer of another lane's\n", .{ k, ey[k].name });
        return error.TestCrossed;
    }
}

/// A run's two recorders (x solo, y in lanes of `n`) at `capacity` with `rows`-row chunks, traced equal.
pub fn run(x: *Rec, y: *LRec, n: u8, mtp: bool, capacity: usize, rows: usize, prompt: []const u32, chunks: usize, own: usize, log: *Log) !void {
    return runWith(x, y, n, mtp, capacity, rows, prompt, chunks, own, log, true);
}

/// `run`, with `absorb` false the request not absorbing into the MTP head (traceEqual's).
pub fn runWith(x: *Rec, y: *LRec, n: u8, mtp: bool, capacity: usize, rows: usize, prompt: []const u32, chunks: usize, own: usize, log: *Log, absorb: bool) !void {
    try x.initWith(mtp, capacity, rows);
    errdefer x.deinit();
    try y.init(n, mtp, capacity, rows);
    errdefer y.deinit();
    try traceEqual(x, y, prompt, chunks, own, log, absorb);
    try testing.expect(y.r.s.most <= n); // never more than N in flight
    try testing.expectEqual(@as(usize, 1), x.s.most);
}

pub fn lh(fseq: i32, rows: i32, layer: i32, mode: proto.Mode, layers: i32, extra: [4]i32) proto.Header {
    return .{ .fseq = fseq, .rows = rows, .layer = layer, .mode = mode, .layers = layers, .extra = extra };
}

/// A step of the record: who, which lane, what, where.
const M = struct { who: lanes.Who, k: u8, step: pairs.Step, pos: usize };

fn L(k: u8, step: pairs.Step, pos: usize) M {
    return .{ .who = .lane, .k = k, .step = step, .pos = pos };
}

/// One group of three on the cut model (dense layers 0-2, MoE layers 3 and 4: positions 0 and 1), python 0026's
/// three rules: each lane's attention to its first exchange and the exchange out (lane order), then each take
/// followed by that lane's hc_post, its attention and its next exchange (three in flight), then the tails in lane
/// order, then the absorbs and commits in chunk order.
const three = [_]M{
    L(0, .stage, 0),                                        L(0, .front, 0),    L(0, .front, 1),   L(0, .front, 2),   L(0, .front, 3), L(0, .route, 0), L(0, .request, 0),
    .{ .who = .group, .k = 1, .step = .position, .pos = 0 }, L(1, .stage, 0),    L(1, .front, 0),   L(1, .front, 1),   L(1, .front, 2), L(1, .front, 3), L(1, .route, 0),
    L(1, .request, 0),                                      .{ .who = .group, .k = 2, .step = .position, .pos = 0 }, L(2, .stage, 0), L(2, .front, 0), L(2, .front, 1), L(2, .front, 2),
    L(2, .front, 3),                                        L(2, .route, 0),    L(2, .request, 0), L(0, .take, 0),    L(0, .post, 0),  L(0, .front, 4), L(0, .route, 1),
    L(0, .request, 1),                                      L(1, .take, 0),     L(1, .post, 0),    L(1, .front, 4),   L(1, .route, 1), L(1, .request, 1), L(2, .take, 0),
    L(2, .post, 0),                                         L(2, .front, 4),    L(2, .route, 1),   L(2, .request, 1), L(0, .take, 1),  L(0, .post, 1),  L(0, .tail, 0),
    L(1, .take, 1),                                         L(1, .post, 1),     L(1, .tail, 0),    L(2, .take, 1),    L(2, .post, 1),  L(2, .tail, 0),  L(0, .absorb, 0),
    L(0, .commit, 0),                                       L(1, .absorb, 0),   L(1, .commit, 0),  L(2, .absorb, 0),  L(2, .commit, 0),
};

fn orderIs(log: *const Log, want: []const M, from: usize) !void {
    for (want, log.marks.items[from..][0..want.len]) |w, got| {
        try testing.expectEqual(w.who, got.who);
        try testing.expectEqual(w.k, got.k);
        try testing.expectEqual(w.step, got.step);
        try testing.expectEqual(w.pos, got.pos);
    }
}

test "qualification_9e49592e lanes of three: 222 tokens in 64-row chunks (a group of three, then 30 alone): the wire's order and words (lane 0's position p, lane 1's, lane 2's), three in flight and never four; each chunk's launches equal the solo prefill's; python 0026's order from the record; G-cross clean" {
    var x: Rec = undefined;
    var y: LRec = undefined;
    var log: Log = .{ .t = &y.r.t };
    defer log.deinit();
    const prompt = pt.ids(222);
    try run(&x, &y, 3, false, 2560, 64, &prompt, 4, 2, &log);
    defer x.deinit();
    defer y.deinit();
    try orderIs(&log, &three, 0);
    const want = [_]proto.Header{
        lh(1, 64, 0, .lanes, 2, .{ 0, 64, 1, 3 }), lh(2, 64, 0, .lanes, 2, .{ 1, 64, 1, 3 }), lh(3, 64, 0, .lanes, 2, .{ 2, 64, 1, 3 }),
        lh(1, 64, 1, .lanes, 2, .{ 0, 64, 1, 3 }), lh(2, 64, 1, .lanes, 2, .{ 1, 64, 1, 3 }), lh(3, 64, 1, .lanes, 2, .{ 2, 64, 1, 3 }),
        lh(4, 30, 0, .chunk, 2, .{ 0, 0, 0, 0 }),  lh(4, 30, 1, .chunk, 2, .{ 0, 0, 0, 0 }),
    };
    try testing.expectEqual(want.len, y.r.s.posts);
    for (want, y.r.s.hdrs[0..want.len]) |w, g| try testing.expectEqual(w, g);
    // the takes before each post: the three firsts together, each lane's next after its own take; the lone chunk's
    // one at a time as today
    try testing.expectEqualSlices(usize, &.{ 0, 0, 0, 1, 2, 3, 6, 7 }, y.r.s.before[0..want.len]);
    try testing.expectEqual(@as(usize, 3), y.r.s.most);
    try testing.expectEqual(@as(usize, 222), y.r.e.st.pos);
    try testing.expectEqual(@as(u64, 1), y.l.runs);
    try testing.expectEqual(@as(u64, 4), y.r.at.forwards);
}

test "qualification_9e49592e lanes of four: 512 tokens (two groups of four, right behind each other: no MTP absorbs between): the second group's request 3 (lane 3's first, in the slot of the first group's tail) waits for its first take; launches chunk by chunk the solo's" {
    var x: Rec = undefined;
    var y: LRec = undefined;
    var log: Log = .{ .t = &y.r.t };
    defer log.deinit();
    const prompt = pt.ids(512);
    try run(&x, &y, 4, false, 2560, 64, &prompt, 8, 6, &log);
    defer x.deinit();
    defer y.deinit();
    try testing.expectEqual(@as(usize, 16), y.r.s.posts);
    try testing.expectEqualSlices(usize, &.{ 0, 0, 0, 0, 1, 2, 3, 4, 8, 8, 8, 9, 9, 10, 11, 12 }, y.r.s.before[0..16]);
    for (0..2) |g| for (0..2) |p| for (0..4) |k| {
        const fseq: i32 = @intCast(4 * g + k + 1);
        try testing.expectEqual(lh(fseq, 64, @intCast(p), .lanes, 2, .{ @intCast(k), 64, @intCast(4 * g + 1), 4 }), y.r.s.hdrs[8 * g + 4 * p + k]);
    };
    try testing.expectEqual(@as(usize, 4), y.r.s.most);
    try testing.expectEqual(@as(u64, 2), y.l.runs);
}

test "qualification_9e49592e lanes of four over 6 chunks (4 + 2: the remainder a pair's group, mode 7 with qualification_0f991257's words) and over 5 (4 + a lone 30-row chunk, mode 3): launches chunk by chunk the solo's" {
    {
        var x: Rec = undefined;
        var y: LRec = undefined;
        var log: Log = .{ .t = &y.r.t };
        defer log.deinit();
        const prompt = pt.ids(350);
        try run(&x, &y, 4, false, 2560, 64, &prompt, 6, 4, &log);
        defer x.deinit();
        defer y.deinit();
        try testing.expectEqual(@as(usize, 12), y.r.s.posts);
        const pair = [_]proto.Header{ lh(5, 64, 0, .pair, 2, .{ 0, 30, 5, 0 }), lh(6, 30, 0, .pair, 2, .{ 1, 64, 5, 0 }), lh(5, 64, 1, .pair, 2, .{ 0, 30, 5, 0 }), lh(6, 30, 1, .pair, 2, .{ 1, 64, 5, 0 }) };
        for (pair, y.r.s.hdrs[8..12]) |w, g| try testing.expectEqual(w, g);
        try testing.expectEqualSlices(usize, &.{ 8, 8, 9, 10 }, y.r.s.before[8..12]);
        try testing.expectEqual(@as(u64, 2), y.l.runs);
    }
    {
        var x: Rec = undefined;
        var y: LRec = undefined;
        var log: Log = .{ .t = &y.r.t };
        defer log.deinit();
        const prompt = pt.ids(286);
        try run(&x, &y, 4, false, 2560, 64, &prompt, 5, 3, &log);
        defer x.deinit();
        defer y.deinit();
        try testing.expectEqual(@as(usize, 10), y.r.s.posts);
        try testing.expectEqual(lh(5, 30, 0, .chunk, 2, .{ 0, 0, 0, 0 }), y.r.s.hdrs[8]);
        try testing.expectEqual(lh(5, 30, 1, .chunk, 2, .{ 0, 0, 0, 0 }), y.r.s.hdrs[9]);
        try testing.expectEqual(@as(u64, 1), y.l.runs);
    }
}

/// A lanes run of `prompt` with the hook's log, then G-cross over it (null: clean).
fn crossedRun(y: *LRec, log: *Log, prompt: []const u32) !?usize {
    y.l.hook = log.hook();
    var f = fwd.Fwd.recording(&y.r.m, &y.r.t);
    _ = try y.r.at.prefill(&f, prompt);
    try f.flush();
    return log.crossed(&y.r.space, y.n);
}

test "qualification_9e49592e G-cross fails on a shared buffer: lane 2's shared-expert rows lane 1's (lane 1's take reads them after lane 2's request), lane 1's streams lane 0's; clean with the copies" {
    const prompt = pt.ids(222);
    for (0..3) |case| {
        var y: LRec = undefined;
        try y.init(3, false, 2560, 64);
        defer y.deinit();
        try y.start();
        switch (case) {
            0 => y.l.c[1].sy = y.l.c[0].sy,
            1 => y.l.c[0].x = y.r.e.pbuf.x,
            else => {},
        }
        var log: Log = .{ .t = &y.r.t };
        defer log.deinit();
        const got = try crossedRun(&y, &log, &prompt);
        if (case < 2) try testing.expect(got != null) else try testing.expectEqual(@as(?usize, null), got);
    }
}

test "qualification_9e49592e lanes refused: not agreed (before any launch); a group stopped early (a post's error at lane 0's position 1, the three firsts in flight) dies at the next begin, nothing more sent; a failed exchange (at lane 1's first take, lane 0's second sent) fails the prefill before its first token, later calls refused at once" {
    const prompt = pt.ids(222);
    { // not agreed: refused before State.reset's launches, nothing posted
        var y: LRec = undefined;
        try y.init(3, false, 2560, 64);
        defer y.deinit();
        try y.start();
        y.l.agreed = false;
        var f = fwd.Fwd.recording(&y.r.m, &y.r.t);
        try testing.expectError(error.NotPorted, y.r.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.r.t.events.items.len);
        try testing.expectEqual(@as(usize, 0), y.r.s.posts);
    }
    { // stopped between its exchanges: the next begin dies (stopped), every later call refuses, nothing more sent
        var y: LRec = undefined;
        try y.init(3, false, 2560, 64);
        defer y.deinit();
        try y.start();
        y.r.s.err_post = 4;
        var f = fwd.Fwd.recording(&y.r.m, &y.r.t);
        try testing.expectError(error.HostPost, y.r.at.prefill(&f, &prompt));
        try testing.expectEqual(@as(usize, 3), y.r.s.posts); // the three firsts, in flight
        try testing.expectEqual(@as(usize, 1), y.r.s.takes); // lane 0's
        try testing.expectError(error.OutOfStep, y.r.at.step(&f, 7));
        try testing.expectEqual(xchg.proto.Why.stopped, y.r.at.failure().?.why);
        y.r.t.reset();
        try testing.expectError(error.ExpertFailed, y.r.at.step(&f, 7));
        try testing.expectError(error.ExpertFailed, y.r.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.r.t.events.items.len);
        try testing.expectEqual(@as(usize, 3), y.r.s.posts);
    }
    { // a half fails at lane 1's first take (lane 0's second already posted: seq 4): the prefill fails at its check
        var y: LRec = undefined;
        try y.init(3, false, 2560, 64);
        defer y.deinit();
        try y.start();
        y.r.s.fail_take = 2;
        var f = fwd.Fwd.recording(&y.r.m, &y.r.t);
        try testing.expectError(error.ExpertFailed, y.r.at.prefill(&f, &prompt));
        try f.flush();
        for (y.r.t.events.items) |e| try testing.expect(!std.mem.eql(u8, e.name, "torch.topk"));
        try testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 4, .gave_up = 0, .breach = 0 }, y.r.at.failure().?);
        const posts = y.r.s.posts;
        y.r.t.reset();
        try testing.expectError(error.ExpertFailed, y.r.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.r.t.events.items.len);
        try testing.expectEqual(posts, y.r.s.posts);
    }
}

/// Every placed buffer's address, in walk order.
const Places = struct {
    ptr: [64]u64 = undefined,
    n: usize = 0,

    fn visit(p: *Places, path: []const u8, b: *views.Buf) anyerror!void {
        _ = path;
        p.ptr[p.n] = b.ptr;
        p.n += 1;
    }
};

test "qualification_9e49592e: lanes 1..N-1's copies (each the flight's: x, post, comb, fnormed, pos_dev, hdr, sy; qualification_47b9bf3a: with the MTP head, mx and mtp_pos_dev too) keep their places when sized after placing; N - 1 times a pair's B copies with two in flight (and the MTP layer's two)" {
    for ([_]u8{ 3, 4 }) |n| for ([_]bool{ false, true }) |mtp| {
        var y: LRec = undefined;
        try y.init(n, mtp, 2560, 64);
        defer y.deinit();
        try y.start();
        var before: Places = .{};
        try y.l.walk(&before, Places.visit);
        const got = try y.l.bytes();
        var after: Places = .{};
        try y.l.walk(&after, Places.visit);
        try testing.expectEqual(@as(usize, if (mtp) 9 else 7) * (n - 1), before.n);
        for (before.ptr[0..before.n]) |p| try testing.expect(p != 0);
        try testing.expectEqualSlices(u64, before.ptr[0..before.n], after.ptr[0..after.n]);
        var c = pairs.Copies.init(&y.r.e.pbuf, &y.r.e.st, &y.r.a);
        var one: bufs.Arena = .{};
        try views.walk(&c, views.Path.init("pair"), &one, bufs.Arena.visit);
        const extra: usize = if (mtp) std.mem.alignForward(usize, y.r.e.pbuf.mx.bytes(), 256) + 256 else 0; // mx, mtp_pos_dev
        try testing.expectEqual((one.used + extra) * (n - 1), got);
    };
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "qualification_9e49592e: TF_AFD_LANES: unset or 0 off, 3 or 4 on with TF_AFD_PAIRS=1 and TF_AFD_INFLIGHT unset, anything else BadAfdLanes" {
    defer _ = unsetenv("TF_AFD_LANES");
    defer _ = unsetenv("TF_AFD_INFLIGHT");
    _ = unsetenv("TF_AFD_INFLIGHT");
    _ = unsetenv("TF_AFD_LANES");
    try std.testing.expectEqual(@as(u8, 0), try lanes.fromEnv(true));
    try std.testing.expectEqual(@as(u8, 0), try lanes.fromEnv(false));
    _ = setenv("TF_AFD_LANES", "0", 1);
    try std.testing.expectEqual(@as(u8, 0), try lanes.fromEnv(false));
    for ([_][*:0]const u8{ "3", "4" }, [_]u8{ 3, 4 }) |v, n| {
        _ = setenv("TF_AFD_LANES", v, 1);
        try std.testing.expectEqual(n, try lanes.fromEnv(true));
        try std.testing.expectError(error.BadAfdLanes, lanes.fromEnv(false)); // lanes only with the pairs
        _ = setenv("TF_AFD_INFLIGHT", "2", 1);
        try std.testing.expectError(error.BadAfdLanes, lanes.fromEnv(true)); // the lanes set the requests in flight
        _ = unsetenv("TF_AFD_INFLIGHT");
    }
    for ([_][*:0]const u8{ "1", "2", "5", "true", "" }) |v| {
        _ = setenv("TF_AFD_LANES", v, 1);
        try std.testing.expectError(error.BadAfdLanes, lanes.fromEnv(true));
    }
}

test "qualification_9e49592e: the lanes' turn keeps the wire's order (lane 0's position p, lane 1's, ..., then p + 1), at most N in flight, takes in order; a group right behind a group holds request slots - 1 for the first take" {
    var t: lanes.Turn = .{ .n = 3, .count = 2, .most = 3, .slots = 3 };
    try std.testing.expectError(error.OutOfStep, t.request(1, 0)); // lane 1 before lane 0
    try t.request(0, 0);
    try t.request(1, 0);
    try t.request(2, 0); // three in flight
    try std.testing.expect(!t.may()); // a fourth waits
    try std.testing.expectError(error.OutOfStep, t.request(0, 1));
    try std.testing.expectEqual(@as(?usize, 0), t.oldest());
    try std.testing.expectError(error.OutOfStep, t.take(1, 0)); // lane 1's take before lane 0's
    try t.take(0, 0);
    try std.testing.expectError(error.OutOfStep, t.request(1, 1)); // lane 1's position 1 before lane 0's
    try t.request(0, 1);
    try t.take(1, 0);
    try t.take(2, 0);
    try t.request(1, 1);
    try t.request(2, 1);
    try std.testing.expect(!t.may()); // past the group
    try t.take(0, 1);
    try t.take(1, 1);
    try std.testing.expect(!t.done());
    try t.take(2, 1);
    try std.testing.expect(t.done());
    try std.testing.expectEqual(@as(?usize, null), t.oldest());
    try std.testing.expectError(error.OutOfStep, t.take(0, 2)); // nothing in flight
    // right behind a group, four lanes over four slots: request 3 (lane 3's first) reuses the slot the halves may
    // still read the last group's tail in, so it waits for request 0's take
    var b: lanes.Turn = .{ .n = 4, .count = 2, .most = 4, .slots = 4, .behind = true };
    try b.request(0, 0);
    try b.request(1, 0);
    try b.request(2, 0);
    try std.testing.expect(!b.may());
    try std.testing.expectError(error.OutOfStep, b.request(3, 0));
    try b.take(0, 0);
    try b.request(3, 0);
    try b.request(0, 1); // four in flight once the first take is in
    try std.testing.expect(!b.may());
    // three lanes over four slots (a remainder group): request 3 is lane 0's position 1, after its first take anyway
    var c: lanes.Turn = .{ .n = 3, .count = 2, .most = 4, .slots = 4, .behind = true };
    try c.request(0, 0);
    try c.request(1, 0);
    try c.request(2, 0);
    try std.testing.expect(!c.may());
    try c.take(0, 0);
    try c.request(0, 1);
}
