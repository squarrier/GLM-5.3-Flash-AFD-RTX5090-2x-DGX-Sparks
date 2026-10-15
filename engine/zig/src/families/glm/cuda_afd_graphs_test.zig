//! qualification_f16009a8's checks of the graphs by key on a CPU: the control flow on a stand-in driver (captured once, replayed after, the
//! least recently used freed past the cap, a failed capture run eagerly from its restored state and its key left eager,
//! captures stopped after max_failures), the recorder's check (a key's forwards must record the same events), and
//! G-trace on the AFD attention node (the EXL3 fixture, the launches recorded, the exchange a stand-in): greedy decodes
//! across the dense chunks' boundary and the dense limit, drafted decodes (f3, auto, an empty MTP cache), every key's
//! later forwards recording exactly its first's, the whole run's events and tokens equal to the run without graphs.
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const eng = @import("cuda_engine.zig");
const afd = @import("cuda_afd_attn.zig");
const ae = @import("cuda_afd_engine.zig");
const ex = @import("cuda_afd_expert.zig");
const drafts = @import("cuda_afd_drafts.zig");
const graphs = @import("cuda_afd_graphs.zig");
const lt = @import("cuda_layer_test.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const testing = std.testing;
const J = std.json.Value;

// -- the control flow on a stand-in driver ----------------------------------------------------------------------------

/// The driver's graph calls stood in: a capture in progress, the handles made, launched and freed, the free memory.
const Fake = struct {
    capturing: bool = false,
    begins: usize = 0,
    fail_begin: bool = false,
    fail_end: bool = false,
    next: usize = 1,
    launched: [64]usize = undefined,
    nl: usize = 0,
    freed: usize = 0,
    free_mem: usize = 1 << 34,
    per_graph: usize = 3 << 20,

    fn ops(f: *Fake) graphs.Ops {
        return .{ .ctx = f, .begin = begin, .end = end, .launch = launch, .free = free, .free_bytes = freeBytes };
    }
    fn of(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }
    fn begin(ctx: *anyopaque) anyerror!void {
        const f = of(ctx);
        f.begins += 1;
        if (f.fail_begin) return error.CaptureRefused;
        f.capturing = true;
    }
    fn end(ctx: *anyopaque, keep: bool) anyerror!?graphs.Made {
        const f = of(ctx);
        f.capturing = false;
        if (f.fail_end) return error.CaptureInvalidated;
        if (!keep) return null;
        f.free_mem -= f.per_graph;
        f.next += 1;
        return .{ .h = f.next - 1, .nodes = 7 };
    }
    fn launch(ctx: *anyopaque, h: usize) anyerror!void {
        const f = of(ctx);
        f.launched[f.nl] = h;
        f.nl += 1;
    }
    fn free(ctx: *anyopaque, h: usize) void {
        _ = h;
        const f = of(ctx);
        f.freed += 1;
        f.free_mem += f.per_graph;
    }
    fn freeBytes(ctx: *anyopaque) usize {
        return of(ctx).free_mem;
    }
};

/// A forward: its launches counted as captured (inside a capture) or eager; `pos` its host state (0 before, 3 after).
const Counter = struct {
    fake: *Fake,
    eager: usize = 0,
    captured: usize = 0,
    pos: usize = 0,
    fail_in_capture: bool = false,

    pub fn launch(b: *Counter) !void {
        b.pos = 1; // half way: a failed capture leaves it here
        if (b.fake.capturing) {
            if (b.fail_in_capture) return error.StreamCaptureUnsupported;
            b.captured += 1;
        } else b.eager += 1;
        b.pos = 3;
    }
    pub fn restore(b: *Counter) void {
        b.pos = 0;
    }
    pub fn replayed(b: *Counter) void {
        b.pos = 3;
    }
    pub fn state(b: *const Counter) struct { pos: usize } {
        return .{ .pos = b.pos };
    }
};

fn key(rows: usize) graphs.Key {
    return .{ .kind = .backbone, .rows = rows, .count = 2 };
}

test "a key captured once and launched, replayed after; another key its own graph; the pinned bytes counted, freed at the end" {
    var fake: Fake = .{};
    var g = graphs.Graphs.init(testing.allocator, fake.ops(), "test");
    g.verbose = false;
    defer g.deinit();
    var b: Counter = .{ .fake = &fake };
    try testing.expectEqual(graphs.How.captured, try g.run(null, key(1), &b));
    try testing.expectEqual(@as(usize, 3), b.pos);
    b.pos = 0;
    try testing.expectEqual(graphs.How.replayed, try g.run(null, key(1), &b));
    try testing.expectEqual(@as(usize, 3), b.pos); // the host state a replay leaves is the launches'
    try testing.expectEqual(graphs.How.captured, try g.run(null, key(4), &b));
    try testing.expectEqual(graphs.How.replayed, try g.run(null, key(1), &b));
    try testing.expectEqualSlices(usize, &.{ 1, 1, 2, 1 }, fake.launched[0..fake.nl]);
    try testing.expectEqual(@as(usize, 2), b.captured);
    try testing.expectEqual(@as(usize, 0), b.eager);
    try testing.expectEqual(graphs.Stats{ .captures = 2, .replays = 2, .nodes = 14, .pinned = 6 << 20, .kept = 2 }, g.stats);
    var lb: [320]u8 = undefined;
    try testing.expectEqualStrings("graphs test: 2 captured, 2 replays, 0 eager, 0 failed captures, 0 evicted; 2 graphs kept (14 nodes, 6291456 B pinned)", g.line(&lb));
    g.deinit();
    try testing.expectEqual(@as(usize, 2), fake.freed);
    g = graphs.Graphs.init(testing.allocator, fake.ops(), "test");
}

test "past the cap the least recently used graph is freed first; its key captured again when it comes back" {
    var fake: Fake = .{};
    var g = graphs.Graphs.init(testing.allocator, fake.ops(), "test");
    g.verbose = false;
    g.max = 2;
    defer g.deinit();
    var b: Counter = .{ .fake = &fake };
    _ = try g.run(null, key(1), &b); // graph 1
    _ = try g.run(null, key(2), &b); // graph 2
    _ = try g.run(null, key(1), &b); // 1 used last: 2 is the oldest
    try testing.expectEqual(graphs.How.captured, try g.run(null, key(3), &b)); // graph 3 in 2's place
    try testing.expectEqual(@as(usize, 1), fake.freed);
    try testing.expectEqual(graphs.How.replayed, try g.run(null, key(1), &b));
    try testing.expectEqual(graphs.How.captured, try g.run(null, key(2), &b)); // graph 4 in 3's place
    try testing.expectEqual(@as(usize, 2), g.stats.evicted);
    try testing.expectEqual(@as(usize, 2), g.stats.kept);
    try testing.expectEqual(@as(usize, 6 << 20), g.stats.pinned);
    try testing.expectEqualSlices(usize, &.{ 1, 2, 1, 3, 1, 4 }, fake.launched[0..fake.nl]);
}

test "a failed capture (refused, a launch inside it, its end) runs the forward eagerly from its restored state; the key stays eager; captures stop after max_failures" {
    var fake: Fake = .{};
    var g = graphs.Graphs.init(testing.allocator, fake.ops(), "test");
    g.verbose = false;
    defer g.deinit();
    var b: Counter = .{ .fake = &fake, .fail_in_capture = true };
    try testing.expectEqual(graphs.How.eager, try g.run(null, key(1), &b)); // the launch inside the capture fails
    try testing.expectEqual(@as(usize, 3), b.pos);
    try testing.expectEqual(@as(usize, 1), b.eager);
    try testing.expect(!fake.capturing); // the capture was ended
    try testing.expectEqual(graphs.How.eager, try g.run(null, key(1), &b)); // no second try for that key
    try testing.expectEqual(@as(usize, 1), fake.begins);
    try testing.expectEqual(@as(usize, 2), b.eager);
    b.fail_in_capture = false;
    fake.fail_begin = true;
    try testing.expectEqual(graphs.How.eager, try g.run(null, key(2), &b)); // the capture refused at its start
    fake.fail_begin = false;
    fake.fail_end = true;
    try testing.expectEqual(graphs.How.eager, try g.run(null, key(3), &b)); // the capture's end fails
    try testing.expectEqual(@as(usize, 4), b.eager);
    try testing.expect(!g.off);
    try testing.expectEqual(graphs.How.eager, try g.run(null, key(4), &b)); // the fourth failure: captures stop
    try testing.expect(g.off);
    fake.fail_end = false;
    const begins = fake.begins;
    try testing.expectEqual(graphs.How.eager, try g.run(null, key(5), &b));
    try testing.expectEqual(begins, fake.begins);
    try testing.expectEqual(@as(usize, 0), fake.nl); // no graph launched, all eager
    try testing.expectEqual(graphs.Stats{ .eager = 6, .failures = 4 }, g.stats);
    // a body's own error is the forward's: eager again from the restored state, it fails there too
    var g2 = graphs.Graphs.init(testing.allocator, fake.ops(), "test");
    defer g2.deinit();
    g2.verbose = false;
    var bad: Bad = .{};
    try testing.expectError(error.Rows, g2.run(null, key(1), &bad));
    try testing.expectEqual(@as(usize, 2), bad.tries);
}

const Bad = struct {
    tries: usize = 0,
    pub fn launch(b: *Bad) !void {
        b.tries += 1;
        return error.Rows;
    }
    pub fn restore(_: *Bad) void {}
    pub fn replayed(_: *Bad) void {}
    pub fn state(_: *const Bad) struct {} {
        return .{};
    }
};

// -- the recorder's check ---------------------------------------------------------------------------------------------

/// A forward recorded: one event naming its rows and a value outside the key (`leak`), the host state its launches set.
const Recorded = struct {
    t: *trace.Trace,
    rows: usize,
    leak: i64 = 0,
    pos: usize = 0,
    bad_replay: bool = false,

    pub fn launch(b: *Recorded) !void {
        try b.t.call(.ext, "forward", &.{ .{ .int = @intCast(b.rows) }, .{ .int = b.leak } });
        b.pos = 3;
    }
    pub fn restore(b: *Recorded) void {
        b.pos = 0;
    }
    pub fn replayed(b: *Recorded) void {
        b.pos = if (b.bad_replay) 2 else 3;
    }
    pub fn state(b: *const Recorded) struct { pos: usize } {
        return .{ .pos = b.pos };
    }
};

test "under a recorder each forward runs eagerly; a key's later forward must record its first's events, and its host state replay as a graph's" {
    var t = trace.Trace.init(testing.allocator);
    defer t.deinit();
    var g = graphs.Graphs.init(testing.allocator, null, "test");
    defer g.deinit();
    var b: Recorded = .{ .t = &t, .rows = 1 };
    try testing.expectEqual(graphs.How.captured, try fresh(&g, &t, key(1), &b));
    try testing.expectEqual(graphs.How.replayed, try fresh(&g, &t, key(1), &b));
    try testing.expectEqual(@as(usize, 2), t.events.items.len); // both recorded: the recorder's forwards are eager
    t.reset(); // the first forward's events are the graphs' own copy
    try testing.expectEqual(graphs.How.replayed, try fresh(&g, &t, key(1), &b));
    b.leak = 9; // a value the key does not carry changed what the launches read
    try testing.expectError(error.GraphMismatch, fresh(&g, &t, key(1), &b));
    b.rows = 4;
    try testing.expectEqual(graphs.How.captured, try fresh(&g, &t, key(4), &b));
    b.bad_replay = true;
    try testing.expectError(error.GraphReplayed, fresh(&g, &t, key(4), &b));
    try testing.expectEqual(@as(usize, 2), g.stats.captures);
    try testing.expectEqual(@as(usize, 2), g.stats.replays);
}

/// A forward begun (its host state at its start, as Afd.begin leaves the provider's), then run.
fn fresh(g: *graphs.Graphs, t: *trace.Trace, k: graphs.Key, b: *Recorded) !graphs.How {
    b.pos = 0;
    return g.run(t, k, b);
}

// -- the expert halves --------------------------------------------------------------------------------------------------

/// An expert half recorded: each take, compute and post one event (compute's with a count outside the key, `leak`).
const RecExpert = struct {
    graphs: ?*graphs.Graphs = null,
    rec: ?*trace.Trace,
    timed: ?bool = false,
    leak: bool = false,
    computes: i64 = 0,

    pub fn take(e: *RecExpert, n: *const xchg.Node, R: usize) !void {
        _ = n;
        try e.rec.?.call(.ext, "xchg.take", &.{.{ .int = @intCast(R) }});
    }
    pub fn compute(e: *RecExpert, pos: usize, f: ex.Forward) !void {
        e.computes += 1;
        try e.rec.?.call(.ext, "expert.compute", &.{ .{ .int = @intCast(pos) }, .{ .int = @intCast(f.rows) }, .{ .int = if (e.leak) e.computes else 0 } });
    }
    pub fn post(e: *RecExpert, n: *const xchg.Node, R: usize) !void {
        _ = n;
        try e.rec.?.call(.ext, "xchg.post", &.{.{ .int = @intCast(R) }});
    }
};

test "the halves' forwards: a decode window or an MTP slice one graph by (rows, slice); prompt chunks, a fault's cut forward and timed halves eager" {
    var t = trace.Trace.init(testing.allocator);
    defer t.deinit();
    var g = graphs.Graphs.init(testing.allocator, null, "expert 0");
    defer g.deinit();
    var e: RecExpert = .{ .graphs = &g, .rec = &t };
    var node: xchg.Node = undefined; // the stand-in never reads it
    const dec: ex.Forward = .{ .fseq = 1, .rows = 1, .mode = .eager, .first = 0, .count = 3 };
    try graphs.layers(&e, &node, dec, 3);
    try graphs.layers(&e, &node, ex.Forward{ .fseq = 4, .rows = 1, .mode = .eager, .first = 0, .count = 3 }, 3);
    try graphs.layers(&e, &node, ex.Forward{ .fseq = 7, .rows = 1, .mode = .slice, .first = 3, .count = 1 }, 1);
    try graphs.layers(&e, &node, ex.Forward{ .fseq = 8, .rows = 4, .mode = .slice, .first = 0, .count = 3 }, 3);
    try testing.expectEqual(graphs.Stats{ .captures = 3, .replays = 1 }, g.stats);
    try testing.expectEqual(@as(usize, 3 * 3 * 2 + 3 + 3 * 3), t.events.items.len);
    try graphs.layers(&e, &node, ex.Forward{ .fseq = 11, .rows = 64, .mode = .chunk, .first = 0, .count = 3 }, 3); // a prompt chunk
    try graphs.layers(&e, &node, ex.Forward{ .fseq = 14, .rows = 1, .mode = .eager, .first = 0, .count = 3 }, 2); // a fault at its third
    e.timed = true;
    try graphs.layers(&e, &node, dec, 3); // TF_AFD_TIME's clock: its events stay out of graphs
    try testing.expectEqual(graphs.Stats{ .captures = 3, .replays = 1 }, g.stats);
    try testing.expectEqual(@as(usize, 3 * 3 * 2 + 3 + 3 * 3 + 3 * 3 + 2 * 3 + 3 * 3), t.events.items.len);
    e.timed = false;
    e.leak = true; // a value outside the key: the recorder's check refuses the replay
    try testing.expectError(error.GraphMismatch, graphs.layers(&e, &node, dec, 3));
    var cpu: struct { // a stand-in without graphs (afd_expert_loop's CpuExpert's shape): the loop as before
        n: usize = 0,
        pub fn take(c: *@This(), _: *const xchg.Node, _: usize) !void {
            c.n += 1;
        }
        pub fn compute(c: *@This(), _: usize, _: ex.Forward) !void {
            c.n += 1;
        }
        pub fn post(c: *@This(), _: *const xchg.Node, _: usize) !void {
            c.n += 1;
        }
    } = .{};
    try graphs.layers(&cpu, &node, dec, 3);
    try testing.expectEqual(@as(usize, 9), cpu.n);
}

test "TF_AFD_GRAPHS: 1 on, unset or 0 off, anything else refused" {
    var kb: [32]u8 = undefined; // the process's own value (a gate's run sets it for every test), put back after
    const prior = saved("TF_AFD_GRAPHS", &kb);
    defer _ = if (prior) |p| setenv("TF_AFD_GRAPHS", p, 1) else unsetenv("TF_AFD_GRAPHS");
    _ = unsetenv("TF_AFD_GRAPHS");
    try testing.expect(!try graphs.fromEnv());
    _ = setenv("TF_AFD_GRAPHS", "1", 1);
    try testing.expect(try graphs.fromEnv());
    _ = setenv("TF_AFD_GRAPHS", "0", 1);
    try testing.expect(!try graphs.fromEnv());
    _ = setenv("TF_AFD_GRAPHS", "true", 1);
    try testing.expectError(error.BadAfdGraphs, graphs.fromEnv());
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// The process's own `name` copied into `buf` (a gate's run may set it for every test), to put it back after.
fn saved(name: [*:0]const u8, buf: *[32]u8) ?[:0]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    if (v.len >= buf.len) return null;
    @memcpy(buf[0..v.len], v);
    buf[v.len] = 0;
    return buf[0..v.len :0];
}

// -- G-trace on the attention node ------------------------------------------------------------------------------------

/// The attention node recorded at long context (the indexer past the dense limit): the EXL3 fixture at world 1 cut to
/// `layers`, with or without its MTP layer, B3's Engine at capacity 2,560 in chunks of 64, the provider, a node.
const Rec = struct {
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

    fn init(x: *Rec, gpa: std.mem.Allocator, layers: usize, chunk: usize, with_mtp: bool) !void {
        x.parsed = try std.json.parseFromSlice(J, gpa, lt.fixture_exl3, .{});
        x.m = try md.Model.parse(gpa, x.parsed.value.object.get("config_json").?.string, 1, 0);
        const mtp_layers = x.m.mtp_layers;
        try x.m.cut(layers);
        x.m.mtp_layers = if (with_mtp) mtp_layers else 0;
        x.w = try views.rank(gpa, &x.m);
        x.e = try eng.Engine.create(gpa, &x.m, &x.w, 2560, 8, chunk, true);
        x.a = try afd.Afd.initWith(&x.m, chunk, with_mtp);
        x.space = .{ .gpa = gpa };
        try views.walk(&x.w, views.Path.init("w"), &x.space, trace.placeAll);
        try x.e.walk(&x.space, trace.placeAll);
        try x.e.walkTmp(&x.space, trace.placeAll);
        x.e.tmp.space = &x.space;
        try x.space.place("b.afd_recv", &x.a.afd_recv);
        x.t = trace.Trace.init(gpa);
        const settings: proto.Settings = .{ .layout = x.a.layout, .max_rows = chunk, .ports = 1 };
        const total = (proto.Region{ .slot = settings.slotBytes() }).total();
        x.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(total));
        @memset(x.region, 0);
        x.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(x.status, 0);
        const mem: xchg.Memory = .{ .region = x.region, .region_dev = @intFromPtr(x.region.ptr), .status = x.status, .status_dev = @intFromPtr(x.status.ptr) };
        x.node = try xchg.Node.init(gpa, testing.io, .attn, .{ .settings = settings, .warn_failures = true }, mem);
    }

    fn deinit(x: *Rec, gpa: std.mem.Allocator) void {
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

/// The exchange's stand-in: every post's header and every take hashed in order.
const Stub = struct {
    h: std.hash.Wyhash = .init(0),
    posts: usize = 0,
    takes: usize = 0,

    fn host(s: *Stub) afd.Host {
        return .{ .ctx = s, .post = post, .take = take };
    }
    fn post(ctx: *anyopaque, node: *xchg.Node, hdr: *const [proto.HDR_BYTES]u8, pos: usize, R: usize) anyerror!void {
        _ = node;
        const s: *Stub = @ptrCast(@alignCast(ctx));
        s.h.update(hdr);
        s.h.update(std.mem.asBytes(&[2]usize{ pos, R }));
        s.posts += 1;
    }
    fn take(ctx: *anyopaque, node: *xchg.Node, pos: usize, R: usize) anyerror!void {
        _ = node;
        const s: *Stub = @ptrCast(@alignCast(ctx));
        s.h.update(std.mem.asBytes(&[3]usize{ 7, pos, R }));
        s.takes += 1;
    }
};

/// Draws: 5, but 6 at every fifth verify row from the second (a rejection), at probability 0.9.
const Picks = struct {
    e: *const eng.Engine,
    vrows: usize = 0,

    fn pick(ctx: *anyopaque, logits: trace.T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        _ = at;
        const p: *Picks = @ptrCast(@alignCast(ctx));
        const verify = logits.ptr == p.e.buf.logits.ptr;
        for (out, 0..) |*o, i| o.* = 5 + @as(u32, @intFromBool(verify and (p.vrows + i) % 5 == 1));
        if (verify) p.vrows += out.len;
        if (probs) |x| @memset(x, 0.9);
    }
};

fn ramp(gpa: std.mem.Allocator, n: usize) ![]u32 {
    const ids = try gpa.alloc(u32, n);
    for (ids, 0..) |*v, i| v.* = @intCast(100 + (13 * i) % 900);
    return ids;
}

/// One node's session: greedy decodes across the dense chunks' 512-key boundary, across the dense limit and again
/// from a short prompt; with the MTP head, drafted decodes: f3 after 300 tokens, auto at the dense limit (verify windows
/// across it), f3 with the head's cache empty (its first step's row zeroed). The tokens go to `toks`.
fn session(gpa: std.mem.Allocator, x: *Rec, at: *ae.Attn, f: *fwd.Fwd, toks: *std.ArrayList(u32)) !void {
    const dl = x.m.denseLimit();
    var out: [32]u32 = undefined;
    for ([_]usize{ 500, dl - 6, 10 }, [_]usize{ 30, 12, 10 }) |n, k| {
        const ids = try ramp(gpa, n);
        defer gpa.free(ids);
        const first = try at.prefill(f, ids);
        try toks.appendSlice(gpa, try at.decode(f, first, k, &out, false));
    }
    if (x.m.mtp_layers == 0) return;
    for ([_]usize{ 300, dl - 5, 40 }, [_]bool{ true, true, false }, 0..) |n, absorb, i| {
        const ids = try ramp(gpa, n);
        defer gpa.free(ids);
        at.absorb = absorb;
        const first = try at.prefill(f, ids);
        var pol: drafts.DepthPolicy = if (i == 1) drafts.DepthPolicy.auto(true) else .{ .most = 3, .fixed = true };
        const d = if (i == 1) try drafts.auto(at, f, first, 20, &out, &pol, false, null) else try drafts.mtp(at, f, first, 20, &out, &pol, false, null);
        try toks.appendSlice(gpa, d.tokens);
    }
}

const Run = struct { toks: std.ArrayList(u32) = .empty, posts: usize = 0, hash: u64 = 0, pos: usize = 0, stats: graphs.Stats = .{}, keys: []graphs.Key = &.{} };

fn runSession(gpa: std.mem.Allocator, x: *Rec, with_graphs: bool) !Run {
    var s: Stub = .{};
    x.a.wire = .{ .node = &x.node, .host = s.host() };
    var at = try ae.Attn.init(x.e, &x.a);
    var picks: Picks = .{ .e = x.e };
    x.t.picker = .{ .ctx = &picks, .pick = Picks.pick };
    var f = fwd.Fwd.recording(&x.m, &x.t);
    var g = graphs.Graphs.init(gpa, null, "attn");
    defer g.deinit();
    g.max = 1 << 20; // every key kept: each later forward compared with its key's first
    if (with_graphs) at.graphs = &g;
    var r: Run = .{};
    try session(gpa, x, &at, &f, &r.toks);
    try f.flush();
    r.posts = s.posts;
    r.hash = s.h.final();
    r.pos = x.e.st.pos;
    r.stats = g.stats;
    r.keys = try gpa.alloc(graphs.Key, g.entries.items.len);
    for (g.entries.items, r.keys) |e, *k| k.* = e.key;
    return r;
}

fn gtrace(with_mtp: bool) !void {
    const gpa = testing.allocator;
    var runs: [2]Run = undefined;
    var xs: [2]Rec = undefined;
    for (0..2) |i| try xs[i].init(gpa, 5, 64, with_mtp);
    defer for (&xs) |*x| x.deinit(gpa);
    for (0..2) |i| runs[i] = try runSession(gpa, &xs[i], i == 1);
    defer for (&runs) |*r| {
        r.toks.deinit(gpa);
        gpa.free(r.keys);
    };
    // the run with the graphs launched what the run without launched, event for event, and drew the same tokens
    try testing.expect(graphs.eventsEql(xs[0].t.events.items, xs[1].t.events.items));
    try testing.expectEqualSlices(u32, runs[0].toks.items, runs[1].toks.items);
    try testing.expectEqual(runs[0].posts, runs[1].posts);
    try testing.expectEqual(runs[0].hash, runs[1].hash);
    try testing.expectEqual(runs[0].pos, runs[1].pos);
    const st = runs[1].stats;
    try testing.expect(st.replays > st.captures and st.captures > 0);
    // the keys seen: both parities, 1 and 2 dense chunks, the sparse pools past the dense limit, MTP steps, wide windows
    var seen: [8]bool = @splat(false);
    for (runs[1].keys) |k| {
        if (k.kind == .backbone) seen[k.parity] = true;
        if (k.kind == .backbone and k.dense and k.nch == 1) seen[2] = true;
        if (k.kind == .backbone and k.dense and k.nch == 2) seen[3] = true;
        if (k.kind == .backbone and !k.dense and k.sparse and k.np > 0) seen[4] = true;
        if (k.kind == .mtp) seen[5] = true;
        if (k.kind == .backbone and k.rows > 1 and k.dense and k.sparse) seen[6] = true; // a verify window across the limit
        if (k.kind == .mtp and k.zero_first) seen[7] = true;
    }
    const want: usize = if (with_mtp) 8 else 5;
    for (seen[0..want], 0..) |s, i| if (!s) {
        std.debug.print("G-trace: key class {d} not seen\n", .{i});
        return error.KeyNotSeen;
    };
    std.debug.print("G-trace (MTP {}): {d} forwards' compute through the graphs: {d} keys captured, {d} replays each equal to its key's first, {d} events and {d} tokens equal to the run without graphs\n", .{ with_mtp, st.captures + st.replays, st.captures, st.replays, xs[1].t.events.items.len, runs[1].toks.items.len });
}

test "G-trace: greedy decodes across 512 keys and the dense limit, drafted decodes (f3, auto, an empty MTP cache): every replay launches its key's first forward's launches; the run equals the run without graphs" {
    try gtrace(false);
    try gtrace(true);
}
