//! qualification_f16009a8: CUDA graphs for the AFD decode forwards (TF_AFD_GRAPHS=1 on a node; unset or 0: every launch eager, as before).
//! A forward's launches are captured the first time its key is seen and replayed as one graph launch after that. The
//! key is what the launches read on the host beyond the buffers' fixed addresses (tpurtell/cuteafd PR #22 by Hugh
//! Madden: decode graphs keyed by what their launches read): the rows, the exchange's slice, the KDA states' parity,
//! the DSA layers' dense chunks or sparse pools, the MTP head's zeroed first row. A failed capture leaves its key eager
//! and the forward runs eagerly (Mia's Qwen dual repo v1.1, f8287ef: a failed capture never ends the server); the
//! device memory each graph pins is measured as the free memory its instantiation took. Under a recorder (a CPU test,
//! --dry) every forward is recorded eagerly and a key seen before must record exactly its first forward's events
//! (G-trace: a replay launches what eager launches), its host state restored and replayed as a graph's would be.
const std = @import("std");
const cuda = @import("cuda");
const trace = @import("cuda_trace.zig");
const clk = @import("cuda_afd_clock.zig");

const xchg = cuda.xchg;
const aot = cuda.aot;
const Trace = trace.Trace;
const Event = trace.Event;
const Val = trace.Val;

/// The widest window graphed: decode.max_rows (a verify window of 7 drafts); prompt chunks stay eager.
pub const max_rows = 8;
/// Graphs kept at once; past it the least recently used is freed first.
pub const max_graphs = 64;
/// Failed captures after which a node stops capturing (every forward eager from then on).
pub const max_failures = 4;

/// TF_AFD_GRAPHS: 1 graphs this node's decode forwards; unset or 0 leaves every launch eager.
pub fn fromEnv() error{BadAfdGraphs}!bool {
    const v = std.mem.span(std.c.getenv("TF_AFD_GRAPHS") orelse return false);
    if (std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    std.debug.print("INFO afd: TF_AFD_GRAPHS takes 1 or 0 (or unset), not \"{s}\"\n", .{v});
    return error.BadAfdGraphs;
}

pub const Kind = enum(u8) { backbone, mtp, expert };

/// What a forward's launches read on the host: two forwards of one key launch the same kernels with the same arguments.
pub const Key = struct {
    kind: Kind,
    rows: usize,
    /// the exchange's slice of MoE positions [first, first + count)
    first: usize = 0,
    count: usize = 0,
    /// the backbone's KDA states (State.parity: which half each layer reads)
    parity: u1 = 0,
    /// the MTP head at position 0 (its embedding row zeroed)
    zero_first: bool = false,
    /// the DSA layers' host choices (cuda_dsa.Shape)
    dense: bool = false,
    nch: usize = 0,
    sparse: bool = false,
    np: usize = 0,

    pub fn line(k: Key, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "{t} rows {d} slice [{d}, {d}) parity {d} zero_first {d} dense {d} nch {d} sparse {d} np {d}", .{ k.kind, k.rows, k.first, k.first + k.count, k.parity, @intFromBool(k.zero_first), @intFromBool(k.dense), k.nch, @intFromBool(k.sparse), k.np }) catch "?";
    }
};

/// A capture's graph: its executable's handle and its nodes.
pub const Made = struct { h: usize, nodes: usize };

/// The driver's graph calls on a forward's stream (`Cuda`), or a CPU test's stand-in.
pub const Ops = struct {
    ctx: *anyopaque,
    begin: *const fn (ctx: *anyopaque) anyerror!void,
    /// the capture ended: with `keep` its graph instantiated and uploaded, else discarded (null)
    end: *const fn (ctx: *anyopaque, keep: bool) anyerror!?Made,
    launch: *const fn (ctx: *anyopaque, h: usize) anyerror!void,
    free: *const fn (ctx: *anyopaque, h: usize) void,
    /// the device's free bytes (cuMemGetInfo): around an instantiation, what its graph pins
    free_bytes: *const fn (ctx: *anyopaque) usize,
};

/// Ops on `stream`: a relaxed capture (other threads' driver calls go on), instantiated and uploaded at its end.
pub const Cuda = struct {
    stream: cuda.Stream,

    pub fn ops(c: *Cuda) Ops {
        return .{ .ctx = c, .begin = begin, .end = end, .launch = launch, .free = free, .free_bytes = freeBytes };
    }

    fn of(ctx: *anyopaque) *Cuda {
        return @ptrCast(@alignCast(ctx));
    }

    fn exec(c: *const Cuda, h: usize) cuda.graph.Exec {
        return .{ .d = c.stream.d, .handle = @ptrFromInt(h) };
    }

    fn begin(ctx: *anyopaque) anyerror!void {
        try cuda.graph.beginCapture(of(ctx).stream, .relaxed);
    }

    fn end(ctx: *anyopaque, keep: bool) anyerror!?Made {
        const s = of(ctx).stream;
        var g = try cuda.graph.endCapture(s);
        defer g.deinit();
        if (!keep) return null;
        const nodes = try g.nodeCount();
        var e = try g.instantiate();
        errdefer e.deinit();
        try e.upload(s);
        return .{ .h = @intFromPtr(e.handle), .nodes = nodes };
    }

    fn launch(ctx: *anyopaque, h: usize) anyerror!void {
        const c = of(ctx);
        try c.exec(h).launchOn(c.stream);
    }

    fn free(ctx: *anyopaque, h: usize) void {
        var e = of(ctx).exec(h);
        e.deinit();
    }

    fn freeBytes(ctx: *anyopaque) usize {
        const d = of(ctx).stream.d;
        var f: usize = 0;
        var t: usize = 0;
        if (d.api.cuMemGetInfo_v2(&f, &t) != cuda.abi.success) return 0;
        return f;
    }
};

/// How a forward ran.
pub const How = enum { eager, captured, replayed };

pub const Stats = struct { captures: usize = 0, replays: usize = 0, eager: usize = 0, failures: usize = 0, evicted: usize = 0, nodes: usize = 0, pinned: usize = 0, kept: usize = 0 };

const Entry = struct {
    key: Key,
    /// the graph's executable (null: its capture failed, so the key runs eagerly)
    h: ?usize = null,
    nodes: usize = 0,
    pinned: usize = 0,
    used: u64 = 0,
    /// under a recorder: its first forward's events
    rec: ?std.heap.ArenaAllocator = null,
    events: []const Event = &.{},
};

/// A node's graphs by key. `run` takes a body: `launch` (the forward's launches on the stream), `restore` (its host
/// state as before them), `replayed` (its host state as they leave it, after a replay) and `state` (that state).
pub const Graphs = struct {
    gpa: std.mem.Allocator,
    /// the driver's calls (null: recorders only)
    ops: ?Ops,
    /// the node in the log's lines ("attn", "expert 0")
    who: []const u8,
    entries: std.ArrayList(Entry) = .empty,
    clock: u64 = 0,
    max: usize = max_graphs,
    /// captures stopped after max_failures failed: every new key runs eagerly
    off: bool = false,
    stats: Stats = .{},
    /// a line a capture (the lab's G-mem reads them)
    verbose: bool = true,

    pub fn init(gpa: std.mem.Allocator, ops: ?Ops, who: []const u8) Graphs {
        return .{ .gpa = gpa, .ops = ops, .who = who };
    }

    pub fn deinit(g: *Graphs) void {
        for (g.entries.items) |*e| g.drop(e);
        g.entries.deinit(g.gpa);
        g.* = undefined;
    }

    fn drop(g: *Graphs, e: *Entry) void {
        if (e.h) |h| if (g.ops) |o| o.free(o.ctx, h);
        if (e.rec) |*a| a.deinit();
        g.stats.pinned -|= e.pinned;
        g.stats.nodes -|= e.nodes;
        if (e.h != null) g.stats.kept -|= 1;
        e.* = .{ .key = e.key };
    }

    fn find(g: *Graphs, key: Key) ?*Entry {
        for (g.entries.items) |*e| if (std.meta.eql(e.key, key)) return e;
        return null;
    }

    /// A new entry for `key`: appended, or past `max` the least recently used one freed and reused.
    fn slot(g: *Graphs, key: Key) !*Entry {
        if (g.entries.items.len < g.max) {
            try g.entries.append(g.gpa, .{ .key = key, .used = g.clock });
            return &g.entries.items[g.entries.items.len - 1];
        }
        var lru = &g.entries.items[0];
        for (g.entries.items[1..]) |*e| if (e.used < lru.used) {
            lru = e;
        };
        g.drop(lru);
        g.stats.evicted += 1;
        lru.* = .{ .key = key, .used = g.clock };
        return lru;
    }

    /// The forward `body` of `key`: replayed from its graph, captured on the key's first sight (then launched), or
    /// eager (its capture failed, or captures stopped). `rec`: a recorder's forward (eager, checked against the first).
    pub fn run(g: *Graphs, rec: ?*Trace, key: Key, body: anytype) !How {
        if (rec) |t| return g.record(t, key, body);
        const o = g.ops orelse {
            try body.launch();
            return .eager;
        };
        g.clock += 1;
        if (g.find(key)) |e| {
            e.used = g.clock;
            if (e.h) |h| {
                try o.launch(o.ctx, h);
                body.replayed();
                g.stats.replays += 1;
                return .replayed;
            }
        } else if (!g.off) return g.capture(o, key, body);
        try body.launch();
        g.stats.eager += 1;
        return .eager;
    }

    fn capture(g: *Graphs, o: Ops, key: Key, body: anytype) !How {
        const e = try g.slot(key);
        const before = o.free_bytes(o.ctx);
        const t0 = nowNs();
        const made = make(o, body) catch |err| {
            body.restore(); // nothing ran: the eager forward starts from the state the capture started from
            g.stats.failures += 1;
            if (g.stats.failures >= max_failures) g.off = true;
            var kb: [256]u8 = undefined;
            std.debug.print("WARN graphs {s}: the capture of {s} failed ({s}): that key runs eagerly{s}\n", .{ g.who, key.line(&kb), @errorName(err), if (g.off) "; captures stopped, every new key eager" else "" });
            try body.launch();
            g.stats.eager += 1;
            return .eager;
        };
        e.h = made.h;
        e.nodes = made.nodes;
        e.pinned = before -| o.free_bytes(o.ctx);
        g.stats.captures += 1;
        g.stats.kept += 1;
        g.stats.nodes += made.nodes;
        g.stats.pinned += e.pinned;
        if (g.verbose) {
            var kb: [256]u8 = undefined;
            std.debug.print("INFO graphs {s}: captured {s}: {d} nodes in {d:.1} ms, {d} B pinned ({d} graphs, {d} B pinned in all)\n", .{ g.who, key.line(&kb), made.nodes, @as(f64, @floatFromInt(nowNs() - t0)) / 1e6, e.pinned, g.stats.kept, g.stats.pinned });
        }
        try o.launch(o.ctx, made.h);
        body.replayed();
        return .captured;
    }

    /// The capture: begun, the body's launches recorded (nothing runs), ended; a failure anywhere ends it, discarded.
    fn make(o: Ops, body: anytype) !Made {
        try o.begin(o.ctx);
        body.launch() catch |err| {
            _ = o.end(o.ctx, false) catch {};
            return err;
        };
        return (try o.end(o.ctx, true)) orelse error.NoGraph;
    }

    /// A recorder's forward: launched (recorded) eagerly; its host state restored and replayed as a graph's would be,
    /// each checked; a key seen before must have recorded exactly the same events (else GraphMismatch).
    fn record(g: *Graphs, t: *Trace, key: Key, body: anytype) !How {
        try t.flush();
        const at = t.events.items.len;
        const pre = body.state();
        try body.launch();
        try t.flush();
        const got = t.events.items[at..];
        const post = body.state();
        body.restore();
        if (!std.meta.eql(pre, body.state())) return error.GraphRestore;
        body.replayed();
        if (!std.meta.eql(post, body.state())) return error.GraphReplayed;
        g.clock += 1;
        if (g.find(key)) |e| {
            e.used = g.clock;
            if (!eventsEql(e.events, got)) {
                var kb: [256]u8 = undefined;
                std.debug.print("ERROR graphs {s}: {s}: {d} events recorded, not its first forward's {d}\n", .{ g.who, key.line(&kb), got.len, e.events.len });
                return error.GraphMismatch;
            }
            g.stats.replays += 1;
            return .replayed;
        }
        const e = try g.slot(key);
        e.rec = std.heap.ArenaAllocator.init(g.gpa);
        const ar = &e.rec.?;
        e.events = try copyEvents(ar.allocator(), got);
        g.stats.captures += 1;
        return .captured;
    }

    /// The log's summary line.
    pub fn line(g: *const Graphs, buf: []u8) []const u8 {
        const s = g.stats;
        return std.fmt.bufPrint(buf, "graphs {s}: {d} captured, {d} replays, {d} eager, {d} failed captures{s}, {d} evicted; {d} graphs kept ({d} nodes, {d} B pinned)", .{ g.who, s.captures, s.replays, s.eager, s.failures, if (g.off) " (captures stopped)" else "", s.evicted, s.kept, s.nodes, s.pinned }) catch "graphs: ?";
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

// -- the expert halves ------------------------------------------------------------------------------------------------

/// cuda_afd_expert.forwards' loop over `run` of a forward's positions: one graph when `e` carries graphs and the whole
/// forward is a decode window or an MTP slice of at most max_rows rows, its TF_AFD_TIME clock off (`e.rec`: a CPU
/// stand-in's recorder); else each position's take, compute and post, as before.
pub fn layers(e: anytype, n: *const xchg.Node, f: anytype, run: usize) !void {
    const E = @TypeOf(e.*);
    if (comptime @hasField(E, "graphs")) if (e.graphs) |g| if (run == f.count and f.rows <= max_rows and (f.mode == .eager or f.mode == .slice) and !timed(e)) {
        var body: Positions(@TypeOf(e), @TypeOf(f)) = .{ .e = e, .n = n, .f = f };
        const rec: ?*Trace = if (comptime @hasField(E, "rec")) e.rec else null;
        _ = try g.run(rec, .{ .kind = .expert, .rows = f.rows, .first = f.first, .count = f.count }, &body);
        return;
    };
    for (f.first..f.first + run) |pos| {
        try e.take(n, f.rows);
        try e.compute(pos, f);
        try e.post(n, f.rows);
    }
}

/// The expert's TF_AFD_TIME clock (Device.timed, read at its first compute): its events stay out of graphs.
fn timed(e: anytype) bool {
    if (comptime !@hasField(@TypeOf(e.*), "timed")) return false;
    return e.timed orelse clk.fromEnv();
}

/// An expert half's forward as a graph body: each position's take, compute and post; no host state changes.
fn Positions(comptime E: type, comptime F: type) type {
    return struct {
        e: E,
        n: *const xchg.Node,
        f: F,

        pub fn launch(b: *@This()) !void {
            for (b.f.first..b.f.first + b.f.count) |pos| {
                try b.e.take(b.n, b.f.rows);
                try b.e.compute(pos, b.f);
                try b.e.post(b.n, b.f.rows);
            }
        }
        pub fn restore(_: *@This()) void {}
        pub fn replayed(_: *@This()) void {}
        pub fn state(_: *const @This()) struct {} {
            return .{};
        }
    };
}

// -- the events a recorder compares -----------------------------------------------------------------------------------

/// Two recorded forwards launch the same: every event's op, name, grid, arguments, constexprs and values equal.
pub fn eventsEql(a: []const Event, b: []const Event) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (!eventEql(x, y)) return false;
    return true;
}

fn eventEql(x: Event, y: Event) bool {
    if (x.op != y.op or !std.mem.eql(u8, x.name, y.name) or !std.mem.eql(u32, &x.grid, &y.grid)) return false;
    if (x.args.len != y.args.len or x.consts.len != y.consts.len or x.vals.len != y.vals.len) return false;
    for (x.args, y.args) |p, q| if (!std.mem.eql(u8, p.name, q.name) or !argEql(p.value, q.value)) return false;
    for (x.consts, y.consts) |p, q| if (!std.mem.eql(u8, p.name, q.name) or !std.meta.eql(p.int, q.int) or bits(p.f32) != bits(q.f32)) return false;
    for (x.vals, y.vals) |p, q| if (!valEql(p, q)) return false;
    return true;
}

fn bits(v: ?f32) ?u32 {
    return if (v) |x| @as(u32, @bitCast(x)) else null;
}

fn argEql(p: aot.Arg.Value, q: aot.Arg.Value) bool {
    if (std.meta.activeTag(p) != std.meta.activeTag(q)) return false;
    return switch (p) {
        .ptr => |a| a.addr == q.ptr.addr and std.mem.eql(u8, a.ty, q.ptr.ty),
        .i32 => |a| a == q.i32,
        .f32 => |a| @as(u32, @bitCast(a)) == @as(u32, @bitCast(q.f32)),
        .u64 => |a| a == q.u64,
    };
}

fn valEql(p: Val, q: Val) bool {
    if (std.meta.activeTag(p) != std.meta.activeTag(q)) return false;
    return switch (p) {
        .t => |a| std.meta.eql(a, q.t),
        .list => |a| a.len == q.list.len and for (a, q.list) |u, v| {
            if (!valEql(u, v)) break false;
        } else true,
        .int => |a| a == q.int,
        .f32 => |a| a == q.f32,
        .bool => |a| a == q.bool,
        .none => true,
    };
}

/// `events` and everything they point at, in `a`.
pub fn copyEvents(a: std.mem.Allocator, events: []const Event) ![]const Event {
    const out = try a.alloc(Event, events.len);
    for (events, out) |e, *o| {
        const args = try a.alloc(aot.Arg, e.args.len);
        for (e.args, args) |x, *y| y.* = .{ .name = try a.dupe(u8, x.name), .value = switch (x.value) {
            .ptr => |p| .{ .ptr = .{ .addr = p.addr, .ty = try a.dupe(u8, p.ty) } },
            else => x.value,
        } };
        const consts = try a.alloc(aot.Const, e.consts.len);
        for (e.consts, consts) |x, *y| y.* = .{ .name = try a.dupe(u8, x.name), .int = x.int, .f32 = x.f32 };
        o.* = .{ .op = e.op, .name = try a.dupe(u8, e.name), .grid = e.grid, .args = args, .consts = consts, .vals = try copyVals(a, e.vals) };
    }
    return out;
}

fn copyVals(a: std.mem.Allocator, vals: []const Val) ![]const Val {
    const out = try a.alloc(Val, vals.len);
    for (vals, out) |v, *o| o.* = switch (v) {
        .list => |l| .{ .list = try copyVals(a, l) },
        else => v,
    };
    return out;
}
