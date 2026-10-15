//! qualification_0f991257's checks of prompt chunk pairs on the AFD attention node on a CPU, the launches recorded and the exchange a
//! stand-in (cuda_afd_engine_test.zig's way), on the EXL3 fixture cut to layers 0-4 (layer 3 DSA and layer 4 KDA are
//! the MoE layers) in 64-row chunks: the wire's order and header words (evidence/qualification_0f991257/WIRE.md §3-§4); each chunk's
//! launches equal the solo prefill's same chunk (G-trace-pairs' form); IMP5's interleave read back from the record; the
//! MTP absorbs after the pair in chunk order; the refusals (pairs not agreed, a pair stopped early, a failed exchange).
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const eng = @import("cuda_engine.zig");
const afd = @import("cuda_afd_attn.zig");
const ae = @import("cuda_afd_engine.zig");
const pairs = @import("cuda_afd_pairs.zig");
const rec = @import("cuda_afd_pairs_rec.zig");
const lt = @import("cuda_layer_test.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const testing = std.testing;
const gpa = testing.allocator;
const J = std.json.Value;

/// The attention node recorded (cuda_afd_engine_test.zig's Rec; with `mtp` cuda_afd_drafts_test.zig's MTP head), and
/// an attention Node (never attended) for its status words.
pub const Rec = struct {
    parsed: std.json.Parsed(J),
    m: md.Model,
    w: views.Weights,
    e: *eng.Engine,
    a: afd.Afd,
    space: trace.Space,
    t: trace.Trace,
    region: []align(4096) u8,
    status: []u32,
    node: xchg.Node,
    s: Stub = .{},
    at: ae.Attn = undefined,
    p: pairs.Pairs = undefined,
    /// qualification_c533375f: the requests a pair keeps in flight (TF_AFD_INFLIGHT; the provider's at `start`)
    inflight: u8 = 1,

    pub fn init(x: *Rec, mtp: bool) !void {
        return x.initWith(mtp, 2560, 64);
    }

    /// At `capacity` with prompt chunks and wire rows of `rows` (the long context past the dense limit, as attn serve).
    pub fn initWith(x: *Rec, mtp: bool, capacity: usize, rows: usize) !void {
        x.s = .{};
        x.inflight = 1; // the caller's `undefined` Rec: one in flight unless a test sets two before `start`
        x.parsed = try std.json.parseFromSlice(J, gpa, lt.fixture_exl3, .{});
        x.m = try md.Model.parse(gpa, x.parsed.value.object.get("config_json").?.string, 1, 0);
        const mtp_layers = x.m.mtp_layers;
        try x.m.cut(5);
        x.m.mtp_layers = if (mtp) mtp_layers else 0;
        x.w = try views.rank(gpa, &x.m);
        x.e = try eng.Engine.create(gpa, &x.m, &x.w, capacity, 8, rows, capacity != 2560 and capacity > x.m.denseLimit());
        x.a = try afd.Afd.initWith(&x.m, rows, mtp);
        x.space = .{ .gpa = gpa };
        try views.walk(&x.w, views.Path.init("w"), &x.space, trace.placeAll);
        try x.e.walk(&x.space, trace.placeAll);
        try x.e.walkTmp(&x.space, trace.placeAll);
        x.e.tmp.space = &x.space;
        try x.space.place("b.afd_recv", &x.a.afd_recv);
        x.t = trace.Trace.init(gpa);
        const settings: proto.Settings = .{ .layout = x.a.layout, .max_rows = rows, .ports = 1, .pairs = true };
        const total = (proto.Region{ .slot = settings.slotBytes() }).total();
        x.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(total));
        @memset(x.region, 0);
        x.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(x.status, 0);
        const mem: xchg.Memory = .{ .region = x.region, .region_dev = @intFromPtr(x.region.ptr), .status = x.status, .status_dev = @intFromPtr(x.status.ptr) };
        x.node = try xchg.Node.init(gpa, testing.io, .attn, .{ .settings = settings, .warn_failures = true }, mem);
    }

    /// The engine over the stand-in exchange; with `paired`, the pairs agreed (the copies placed last, so every other
    /// buffer keeps the solo run's address).
    pub fn start(x: *Rec, paired: bool) !void {
        x.a.wire = .{ .node = &x.node, .host = x.s.host() };
        x.at = try ae.Attn.init(x.e, &x.a);
        x.at.absorb = x.e.mbuf != null;
        if (!paired) return;
        x.a.inflight = x.inflight;
        x.p = pairs.Pairs.init(&x.at);
        x.p.agreed = true;
        try x.p.walk(&x.space, trace.placeAll);
        x.at.pairs = &x.p;
    }

    pub fn deinit(x: *Rec) void {
        x.node.deinit();
        gpa.free(x.region);
        gpa.free(x.status);
        x.t.deinit();
        x.space.deinit();
        x.e.destroy(gpa);
        x.w.deinit(gpa);
        x.parsed.deinit();
    }
};

/// The exchange's stand-in: each post's header and word 6 kept, the takes counted; the exchange fails at take
/// `fail_take`; post `err_post` errs (a host exchange's error between two requests of a pair). qualification_c533375f: the takes done
/// before each post (`before`), and the most requests in flight at once (`most`).
pub const Stub = struct {
    hdrs: [64]proto.Header = undefined,
    counts: [64]i32 = undefined,
    before: [64]usize = undefined,
    posts: usize = 0,
    takes: usize = 0,
    most: usize = 0,
    fail_take: ?usize = null,
    err_post: ?usize = null,

    fn host(s: *Stub) afd.Host {
        return .{ .ctx = s, .post = post, .take = take };
    }

    fn post(ctx: *anyopaque, node: *xchg.Node, hdr: *const [proto.HDR_BYTES]u8, pos: usize, R: usize) anyerror!void {
        _ = node;
        const s: *Stub = @ptrCast(@alignCast(ctx));
        if (s.err_post != null and s.err_post.? == s.posts + 1) return error.HostPost;
        const h = try proto.Header.read(hdr);
        if (h.layer != pos or h.rows != R) return error.HeaderNotPos;
        s.hdrs[s.posts] = h;
        s.counts[s.posts] = std.mem.readInt(i32, hdr[24..28], .little);
        s.before[s.posts] = s.takes;
        s.posts += 1;
        s.most = @max(s.most, s.posts - s.takes);
    }

    fn take(ctx: *anyopaque, node: *xchg.Node, pos: usize, R: usize) anyerror!void {
        _ = .{ pos, R };
        const s: *Stub = @ptrCast(@alignCast(ctx));
        s.takes += 1;
        if (s.fail_take != null and s.fail_take.? == s.takes) node.fail(.{ .why = .silent, .peer = 1, .seq = @intCast(s.posts) });
    }
};

pub fn ids(comptime n: usize) [n]u32 {
    var v: [n]u32 = undefined;
    for (&v, 0..) |*x, i| x.* = @intCast(1000 + i);
    return v;
}

pub fn head(fseq: i32, rows: i32, layer: i32, mode: proto.Mode, layers: i32, extra: [3]i32) proto.Header {
    return .{ .fseq = fseq, .rows = rows, .layer = layer, .mode = mode, .layers = layers, .extra = .{ extra[0], extra[1], extra[2], 0 } };
}

/// The first launch where two runs part (by name), printed with its neighbours: a failing gate's diagnosis.
fn diverge(sx: *const trace.Space, ex: []const trace.Event, sy: *const trace.Space, ey: []const trace.Event) void {
    const n = @min(ex.len, ey.len);
    var i: usize = 0;
    while (i < n and std.mem.eql(u8, &rec.digest(sx, ex[i .. i + 1]), &rec.digest(sy, ey[i .. i + 1]))) i += 1;
    std.debug.print("first divergence at {d} of {d} (solo) / {d} (paired)\n", .{ i, ex.len, ey.len });
    const lo = if (i >= 2) i - 2 else 0;
    for (lo..@min(i + 3, ex.len)) |j| std.debug.print("  solo   {d}: {s} {s}\n", .{ j, @tagName(ex[j].op), ex[j].name });
    for (lo..@min(i + 3, ey.len)) |j| std.debug.print("  paired {d}: {s} {s}\n", .{ j, @tagName(ey[j].op), ey[j].name });
}

/// The solo prefill and the paired one of `prompt` in two fresh recorders: every chunk's launches (a pair's A and B by
/// the hook, the solo run's cut at its commits) digest equal by name, State.reset's and the first token's launches
/// too, and the paired run adds only each pair's position fill; gives the paired run's step log for the caller.
pub fn traceEqual(x: *Rec, y: *Rec, prompt: []const u32, chunks: usize, log: *rec.Log) !void {
    try x.start(false);
    var fx = fwd.Fwd.recording(&x.m, &x.t);
    const tok_x = try x.at.prefill(&fx, prompt);
    try fx.flush();
    try y.start(true);
    y.p.hook = log.hook();
    var fy = fwd.Fwd.recording(&y.m, &y.t);
    const tok_y = try y.at.prefill(&fy, prompt);
    try fy.flush();
    try testing.expectEqual(tok_x, tok_y);
    const ex = x.t.events.items;
    const ey = y.t.events.items;
    const reset = log.marks.items[0].at; // State.reset's launches: the paired run's before its first step
    try testing.expect(reset > 0 and reset <= ex.len);
    try testing.expectEqualStrings(&rec.digest(&x.space, ex[0..reset]), &rec.digest(&y.space, ey[0..reset]));
    var ranges: [16][2]usize = undefined;
    try testing.expectEqual(chunks, rec.soloChunks(&x.space, ex, reset, &ranges));
    for (ranges[0..chunks], 0..) |r, k| {
        const want = rec.digest(&x.space, ex[r[0]..r[1]]);
        const got = log.chunkDigest(&y.space, k);
        if (r[1] - r[0] != got.n or !std.mem.eql(u8, &want, &got.hex)) {
            const evs = try log.chunkEvents(gpa, k);
            defer gpa.free(evs);
            std.debug.print("chunk {d}: ", .{k});
            diverge(&x.space, ex[r[0]..r[1]], &y.space, evs);
        }
        try testing.expectEqual(r[1] - r[0], got.n);
        try testing.expectEqualStrings(&want, &got.hex);
    }
    const first = log.marks.items[log.marks.items.len - 1];
    try testing.expect(first.who == .pair and first.step == .first);
    try testing.expectEqualStrings(&rec.digest(&x.space, ex[ranges[chunks - 1][1]..]), &rec.digest(&y.space, ey[first.at..])); // the first token
    try testing.expectEqual(chunks / 2, log.own()); // one position fill a pair, nothing else added
    try testing.expectEqual(ex.len + log.own(), ey.len);
    try testing.expectEqual(x.e.st.pos, y.e.st.pos);
    try testing.expectEqual(x.e.st.parity, y.e.st.parity);
    try testing.expectEqual(x.e.st.mtp_len, y.e.st.mtp_len);
    try noCross(log, y);
}

/// G-cross (qualification_c533375f, rec.Log.crossed): what crosses a step is each window's own, from the paired run's record (B's
/// shared-expert rows its own with two in flight, A's buffer with one).
pub fn noCross(log: *const rec.Log, y: *const Rec) !void {
    const sy_b = if (y.p.inflight > 1) "pair.sy" else "pbuf.sy";
    if (log.crossed(&y.space, .{ "pbuf.sy", sy_b })) |k| {
        std.debug.print("G-cross: launch {d} ({s}) names a buffer of the other window's\n", .{ k, y.t.events.items[k].name });
        return error.TestCrossed;
    }
}

pub const M = struct { who: pairs.Who, step: pairs.Step, pos: usize };

/// IMP5's order for one pair of the cut model (dense layers 0-2, MoE layers 3 and 4: positions 0 and 1), as WIRE.md §4.
const imp5 = [_]M{
    .{ .who = .a, .step = .stage, .pos = 0 },   .{ .who = .a, .step = .front, .pos = 0 },   .{ .who = .a, .step = .front, .pos = 1 },
    .{ .who = .a, .step = .front, .pos = 2 },   .{ .who = .a, .step = .front, .pos = 3 },   .{ .who = .a, .step = .route, .pos = 0 },
    .{ .who = .a, .step = .request, .pos = 0 }, .{ .who = .pair, .step = .position, .pos = 0 }, .{ .who = .b, .step = .stage, .pos = 0 },
    .{ .who = .b, .step = .front, .pos = 0 },   .{ .who = .b, .step = .front, .pos = 1 },   .{ .who = .b, .step = .front, .pos = 2 },
    .{ .who = .b, .step = .front, .pos = 3 },   .{ .who = .b, .step = .route, .pos = 0 },   .{ .who = .a, .step = .take, .pos = 0 },
    .{ .who = .b, .step = .request, .pos = 0 }, .{ .who = .a, .step = .post, .pos = 0 },    .{ .who = .a, .step = .front, .pos = 4 },
    .{ .who = .a, .step = .route, .pos = 1 },   .{ .who = .b, .step = .take, .pos = 0 },    .{ .who = .a, .step = .request, .pos = 1 },
    .{ .who = .b, .step = .post, .pos = 0 },    .{ .who = .b, .step = .front, .pos = 4 },   .{ .who = .b, .step = .route, .pos = 1 },
    .{ .who = .a, .step = .take, .pos = 1 },    .{ .who = .b, .step = .request, .pos = 1 }, .{ .who = .a, .step = .post, .pos = 1 },
    .{ .who = .a, .step = .tail, .pos = 0 },    .{ .who = .b, .step = .take, .pos = 1 },    .{ .who = .b, .step = .post, .pos = 1 },
    .{ .who = .b, .step = .tail, .pos = 0 },    .{ .who = .a, .step = .absorb, .pos = 0 },  .{ .who = .a, .step = .commit, .pos = 0 },
    .{ .who = .b, .step = .absorb, .pos = 0 },  .{ .who = .b, .step = .commit, .pos = 0 },
};

/// Each exchange's gap: the launches between a request (afd.exchange) and its own take (afd.collect: the oldest
/// request's, qualification_c533375f's two in flight included), in order.
fn gaps(events: []const trace.Event, out: []usize) usize {
    var n: usize = 0;
    var sent: [4]usize = undefined;
    var oldest: usize = 0;
    var tail: usize = 0;
    for (events, 0..) |e, i| {
        if (std.mem.eql(u8, e.name, "afd.exchange")) {
            sent[tail % 4] = i;
            tail += 1;
        } else if (std.mem.eql(u8, e.name, "afd.collect")) {
            if (oldest < tail) {
                if (n < out.len) {
                    out[n] = i - sent[oldest % 4];
                    n += 1;
                }
                oldest += 1;
            }
        }
    }
    return n;
}

/// The record read back: the steps in IMP5's order (`want`, from mark `from`), and in each of the pair's exchanges more
/// launches between the request and its take than a solo chunk's (its shared expert alone): the other chunk's
/// attention, enqueued before the GPU's wait for the reply.
fn interleave(log: *const rec.Log, x: *const Rec, y: *const Rec) !void {
    return interleaveAs(log, x, y, &imp5, 0);
}

pub fn interleaveAs(log: *const rec.Log, x: *const Rec, y: *const Rec, want: []const M, from: usize) !void {
    for (want, log.marks.items[from..][0..want.len]) |w, got| {
        try testing.expectEqual(w.who, got.who);
        try testing.expectEqual(w.step, got.step);
        try testing.expectEqual(w.pos, got.pos);
    }
    var gs: [16]usize = undefined;
    var gp: [16]usize = undefined;
    try testing.expect(gaps(x.t.events.items, &gs) >= 1);
    try testing.expect(gaps(y.t.events.items, &gp) >= 4);
    for (gp[0..4]) |g| try testing.expect(g > gs[0]);
}

test "pairs: 158 tokens in 64-row chunks (a pair, then 30 alone): the wire's order and words, each chunk's launches equal the solo prefill's, IMP5's interleave from the record" {
    var x: Rec = undefined;
    try x.init(false);
    defer x.deinit();
    var y: Rec = undefined;
    try y.init(false);
    defer y.deinit();
    var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
    defer log.deinit();
    const prompt = ids(158);
    try traceEqual(&x, &y, &prompt, 3, &log);
    try interleave(&log, &x, &y);
    // the wire: A0 B0 A1 B1 (mode 7, words 7-9), then the odd last chunk as today (mode 3)
    const want = [_]proto.Header{
        head(1, 64, 0, .pair, 2, .{ 0, 64, 1 }), head(2, 64, 0, .pair, 2, .{ 1, 64, 1 }), head(1, 64, 1, .pair, 2, .{ 0, 64, 1 }),
        head(2, 64, 1, .pair, 2, .{ 1, 64, 1 }), head(3, 30, 0, .chunk, 2, .{ 0, 0, 0 }), head(3, 30, 1, .chunk, 2, .{ 0, 0, 0 }),
    };
    try testing.expectEqual(want.len, y.s.posts);
    for (want, y.s.hdrs[0..want.len], y.s.counts[0..want.len]) |w, g, c| {
        try testing.expectEqual(w, g);
        try testing.expectEqual(@as(i32, 0), c); // word 6: 0 without the MTP head
    }
    // the same requests as the solo run's, by forward and position, once mode and the pair words are set aside
    try testing.expectEqual(x.s.posts, y.s.posts);
    for (x.s.hdrs[0..x.s.posts]) |h| {
        var found = false;
        for (y.s.hdrs[0..y.s.posts]) |g| found = found or (g.fseq == h.fseq and g.layer == h.layer and g.rows == h.rows and g.layers == h.layers);
        try testing.expect(found);
    }
    try testing.expectEqual(@as(usize, 158), y.e.st.pos);
    try testing.expectEqual(@as(u64, 1), y.p.runs);
}

test "pairs: two pairs (64 + 64, 64 + 17 = 209 tokens), B short: the same launches chunk by chunk, B's rows in A's word 8" {
    var x: Rec = undefined;
    try x.init(false);
    defer x.deinit();
    var y: Rec = undefined;
    try y.init(false);
    defer y.deinit();
    var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
    defer log.deinit();
    const prompt = ids(209);
    try traceEqual(&x, &y, &prompt, 4, &log);
    try testing.expectEqual(@as(usize, 8), y.s.posts);
    try testing.expectEqual(head(3, 64, 0, .pair, 2, .{ 0, 17, 3 }), y.s.hdrs[4]);
    try testing.expectEqual(head(4, 17, 0, .pair, 2, .{ 1, 64, 3 }), y.s.hdrs[5]);
    try testing.expectEqual(head(4, 17, 1, .pair, 2, .{ 1, 64, 3 }), y.s.hdrs[7]);
    try testing.expectEqual(@as(u64, 2), y.p.runs);
}

test "pairs with the MTP head (TF_AFD_MTP=1): the absorbs after the pair in chunk order (mode 5 at the MTP position, seq s+3 and s+4), word 6 the count, each chunk's launches (its absorb included) the solo's" {
    var x: Rec = undefined;
    try x.init(true);
    defer x.deinit();
    var y: Rec = undefined;
    try y.init(true);
    defer y.deinit();
    var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
    defer log.deinit();
    const prompt = ids(81);
    try traceEqual(&x, &y, &prompt, 2, &log);
    try interleave(&log, &x, &y);
    const want = [_]proto.Header{
        head(1, 64, 0, .pair, 3, .{ 0, 17, 1 }), head(2, 17, 0, .pair, 3, .{ 1, 64, 1 }), head(1, 64, 1, .pair, 3, .{ 0, 17, 1 }),
        head(2, 17, 1, .pair, 3, .{ 1, 64, 1 }), head(3, 64, 2, .slice, 3, .{ 0, 0, 0 }), head(4, 16, 2, .slice, 3, .{ 0, 0, 0 }),
    };
    try testing.expectEqual(want.len, y.s.posts);
    for (want, y.s.hdrs[0..want.len]) |w, g| try testing.expectEqual(w, g);
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 2, 1, 1 }, y.s.counts[0..want.len]);
    try testing.expectEqual(@as(usize, 80), y.e.st.mtp_len);
}

test "G-trace-pairs at 8,000 tokens in 2,048-row chunks (2,048 + 2,048, 2,048 + 1,856, as attn serve chunks them; capacity 8,256, the long context past the dense limit) with TF_AFD_MTP's absorbs: each chunk's launches equal the solo prefill's; the wire's words in IMP5's order" {
    var x: Rec = undefined;
    try x.initWith(true, 8256, 2048);
    defer x.deinit();
    var y: Rec = undefined;
    try y.initWith(true, 8256, 2048);
    defer y.deinit();
    var log: rec.Log = .{ .t = &y.t, .gpa = gpa };
    defer log.deinit();
    const prompt = try gpa.alloc(u32, 8000);
    defer gpa.free(prompt);
    for (prompt, 0..) |*v, i| v.* = @intCast(1000 + (7 * i) % 5000);
    try traceEqual(&x, &y, prompt, 4, &log);
    try interleave(&log, &x, &y);
    const want = [_]proto.Header{
        head(1, 2048, 0, .pair, 3, .{ 0, 2048, 1 }), head(2, 2048, 0, .pair, 3, .{ 1, 2048, 1 }), head(1, 2048, 1, .pair, 3, .{ 0, 2048, 1 }),
        head(2, 2048, 1, .pair, 3, .{ 1, 2048, 1 }), head(3, 2048, 2, .slice, 3, .{ 0, 0, 0 }),   head(4, 2048, 2, .slice, 3, .{ 0, 0, 0 }),
        head(5, 2048, 0, .pair, 3, .{ 0, 1856, 5 }), head(6, 1856, 0, .pair, 3, .{ 1, 2048, 5 }), head(5, 2048, 1, .pair, 3, .{ 0, 1856, 5 }),
        head(6, 1856, 1, .pair, 3, .{ 1, 2048, 5 }), head(7, 2048, 2, .slice, 3, .{ 0, 0, 0 }),   head(8, 1855, 2, .slice, 3, .{ 0, 0, 0 }),
    };
    try testing.expectEqual(want.len, y.s.posts);
    for (want, y.s.hdrs[0..want.len]) |w, g| try testing.expectEqual(w, g);
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1 }, y.s.counts[0..want.len]);
    try testing.expectEqual(@as(usize, 8000), y.e.st.pos);
    try testing.expectEqual(@as(usize, 7999), y.e.st.mtp_len);
    try testing.expectEqual(@as(u64, 2), y.p.runs);
    std.debug.print("G-trace-pairs (attention, 8,000 tokens in 2,048-row chunks, MTP): {d} launches solo, {d} paired; 4 chunks equal; 12 requests in IMP5's order\n", .{ x.t.events.items.len, y.t.events.items.len });
}

test "pairs refused: not agreed (before any launch); a pair stopped early (a post's error at A[1]) dies at the next begin, nothing more sent; a failed exchange fails the prefill before its first token, later calls refused at once" {
    const prompt = ids(158);
    { // the halves did not agree pairs: refused before State.reset's launches, nothing posted
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        try y.start(true);
        y.p.agreed = false;
        var f = fwd.Fwd.recording(&y.m, &y.t);
        try testing.expectError(error.NotPorted, y.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.t.events.items.len);
        try testing.expectEqual(@as(usize, 0), y.s.posts);
    }
    { // a pair stopped between its exchanges: the next begin dies (stopped), every later call refuses, nothing more sent
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        try y.start(true);
        y.s.err_post = 3;
        var f = fwd.Fwd.recording(&y.m, &y.t);
        try testing.expectError(error.HostPost, y.at.prefill(&f, &prompt));
        try testing.expectEqual(@as(usize, 2), y.s.posts); // A0 and B0
        try testing.expectError(error.OutOfStep, y.at.step(&f, 7));
        try testing.expectEqual(xchg.proto.Why.stopped, y.at.failure().?.why);
        y.t.reset();
        try testing.expectError(error.ExpertFailed, y.at.step(&f, 7));
        try testing.expectError(error.ExpertFailed, y.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.t.events.items.len);
        try testing.expectEqual(@as(usize, 2), y.s.posts);
    }
    { // a half fails at B[0]'s take: the prefill fails at its check, no first token drawn; the next refuses at once
        var y: Rec = undefined;
        try y.init(false);
        defer y.deinit();
        try y.start(true);
        y.s.fail_take = 2;
        var f = fwd.Fwd.recording(&y.m, &y.t);
        try testing.expectError(error.ExpertFailed, y.at.prefill(&f, &prompt));
        try f.flush();
        for (y.t.events.items) |e| try testing.expect(!std.mem.eql(u8, e.name, "torch.topk"));
        try testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 2, .gave_up = 0, .breach = 0 }, y.at.failure().?);
        const posts = y.s.posts;
        y.t.reset();
        try testing.expectError(error.ExpertFailed, y.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.t.events.items.len);
        try testing.expectEqual(posts, y.s.posts);
    }
}

/// Each copy's address in walk order (B's copies: x, post, comb, fnormed, pos_dev, hdr, and sy with two in flight).
const Places = struct {
    ptr: [16]u64 = @splat(0),
    n: usize = 0,

    fn visit(s: *Places, path: []const u8, b: *views.Buf) anyerror!void {
        _ = path;
        s.ptr[s.n] = b.ptr;
        s.n += 1;
    }
};

// qualification_c533375f (qualification_242abc20's segfault: attention-host's attn first with TF_AFD_PAIRS=1 died in cuMemcpyHtoDAsync at B's first header copy):
// `bytes` after the copies are placed (attn's INFO line runs it after `attach`, the --dry path after the Space's
// placing) leaves every copy where it was, and gives the bytes a placing walk takes; one in flight and two.
test "qualification_c533375f: B's copies keep their places when sized after placing" {
    for ([_]u8{ 1, 2 }) |inflight| {
        var x: Rec = undefined;
        try x.init(true);
        defer x.deinit();
        x.inflight = inflight;
        try x.start(true);
        var before: Places = .{};
        try x.p.walk(&before, Places.visit);
        const n = try x.p.bytes();
        var after: Places = .{};
        try x.p.walk(&after, Places.visit);
        try testing.expect(before.n >= 6);
        for (before.ptr[0..before.n]) |p| try testing.expect(p != 0);
        try testing.expectEqualSlices(u64, before.ptr[0..before.n], after.ptr[0..after.n]);
        var c = x.p.c;
        var placing: @import("cuda_buffers.zig").Arena = .{ .base = 1 << 44 };
        try views.walk(&c, views.Path.init("pair"), &placing, @import("cuda_buffers.zig").Arena.visit);
        try testing.expectEqual(placing.used, n);
    }
}
