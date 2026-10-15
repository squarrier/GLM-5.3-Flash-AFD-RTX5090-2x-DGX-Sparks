//! prefill_interleave on the recorded real device Users (no CUDA numerics; launches, operands and wire headers only): a prompt's
//! served pass a step at a time (`Users.prefillStep`: plain chunks, qualification_0f991257 pairs, qualification_9e49592e lanes of 3 and 4) is the whole pass
//! (`Users.prefill`) launch for launch on State0 and State1; with another user's decode rounds between its steps, the
//! prompt's launches are its solo pass's (plain chunks launch for launch; pairs and lanes chunk for chunk, each chunk's
//! launches in order: between two groups a decode forward ends the `behind` overlap, so only where a group's requests
//! stand among the kinds differs) and each round's its solo round's (temporaries compared by python's name, its
//! counter aside), the wire alternates prompt and decode forwards, at the 512 and 2048 boundaries; a release between steps
//! clears the pass and the slot's next generation starts from its reset; a failed step is sticky.
//! Fixture scaffolding follows afd_users_device_test.zig (Ash Hart's cuda_afd_engine_test.zig) and qualification_9e49592e's lanes recorder.
//! Claude Opus 5.5 (planning and execution), via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const schedule = glm.afd_users_round;
const t = std.testing;
const gpa = t.allocator;
const proto = cuda.xchg.proto;
const Key = schedule.Key;

const Path = enum { plain, pairs, lanes3, lanes4 };
const rows = 64;

/// The recorded attention node with its users (count 2 or 4) on `path`, over a stand-in exchange keeping every header.
const FRec = struct {
    parsed: std.json.Parsed(std.json.Value),
    m: glm.model.Model,
    w: glm.views.Weights,
    e: *glm.engine.Engine,
    a: glm.afd_attn.Afd,
    at: glm.afd_engine.Attn,
    space: glm.trace.Space,
    trace: glm.trace.Trace,
    f: glm.fwd.Fwd,
    users: glm.afd_users.Users,
    l: glm.afd_lanes.Lanes = undefined,
    p: glm.afd_pairs.Pairs = undefined,
    region: []align(4096) u8,
    status: []u32,
    node: cuda.xchg.Node,
    posts: usize = 0,
    takes: usize = 0,
    fail_take: ?usize = null,
    hdrs: [1024]proto.Header = undefined,

    fn init(r: *FRec, path: Path, n: u8) !void {
        r.posts = 0;
        r.takes = 0;
        r.fail_take = null;
        r.parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
        r.m = try glm.model.Model.parse(gpa, r.parsed.value.object.get("config_json").?.string, 1, 0);
        try r.m.cut(5);
        r.m.mtp_layers = 0;
        r.w = try glm.views.rank(gpa, &r.m);
        r.e = try glm.engine.Engine.create(gpa, &r.m, &r.w, 2560, 8, rows, true);
        r.a = try glm.afd_attn.Afd.init(&r.m, rows);
        r.space = .{ .gpa = gpa };
        try glm.views.walk(&r.w, glm.views.Path.init("w"), &r.space, glm.trace.placeAll);
        try r.e.walk(&r.space, glm.trace.placeAll);
        try r.e.walkTmp(&r.space, glm.trace.placeAll);
        r.e.tmp.space = &r.space;
        try r.space.place("b.afd_recv", &r.a.afd_recv);
        r.trace = glm.trace.Trace.init(gpa);
        r.f = glm.fwd.Fwd.recording(&r.m, &r.trace);
        const k: u8 = switch (path) {
            .lanes3 => 3,
            .lanes4 => 4,
            else => 1,
        };
        const settings: proto.Settings = switch (path) {
            .plain => .{ .layout = r.a.layout, .max_rows = rows, .ports = 1 },
            .pairs => .{ .layout = r.a.layout, .max_rows = rows, .ports = 1, .pairs = true },
            .lanes3, .lanes4 => .{ .layout = r.a.layout, .max_rows = rows, .ports = 1, .pairs = true, .inflight = k, .lanes = k },
        };
        r.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(settings.region().total()));
        @memset(r.region, 0);
        r.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(r.status, 0);
        r.node = try cuda.xchg.Node.init(gpa, t.io, .attn, .{ .settings = settings, .warn_failures = true }, .{ .region = r.region, .region_dev = @intFromPtr(r.region.ptr), .status = r.status, .status_dev = @intFromPtr(r.status.ptr) });
        r.a.wire = .{ .node = &r.node, .host = .{ .ctx = r, .post = post, .take = take } };
        r.at = try glm.afd_engine.Attn.init(r.e, &r.a);
        switch (path) {
            .plain => {},
            .pairs => {
                r.a.inflight = 1;
                r.p = glm.afd_pairs.Pairs.init(&r.at);
                r.p.agreed = true;
                try r.p.walk(&r.space, glm.trace.placeAll);
                r.at.pairs = &r.p;
            },
            .lanes3, .lanes4 => {
                r.a.paired = true;
                r.a.inflight = k;
                r.a.lanes = k;
                r.l = try glm.afd_lanes.Lanes.init(&r.at);
                r.l.agreed = true;
                try r.l.walk(&r.space, glm.trace.placeAll);
                r.at.lanes = &r.l;
            },
        }
        r.users = try glm.afd_users.Users.init(gpa, &r.at, &r.f, n);
        try r.users.walk(&r.space, glm.trace.placeAll);
    }

    fn deinit(r: *FRec) void {
        r.node.deinit();
        gpa.free(r.region);
        gpa.free(r.status);
        r.users.deinit(gpa);
        r.trace.deinit();
        r.space.deinit();
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.parsed.deinit();
    }

    fn post(p: *anyopaque, _: *cuda.xchg.Node, h: *const [proto.HDR_BYTES]u8, pos: usize, n: usize) !void {
        const r: *FRec = @ptrCast(@alignCast(p));
        const hdr = try proto.Header.read(h);
        try t.expectEqual(@as(i32, @intCast(n)), hdr.rows);
        try t.expectEqual(@as(i32, @intCast(pos)), hdr.layer);
        if (r.posts < r.hdrs.len) r.hdrs[r.posts] = hdr;
        r.posts += 1;
    }

    fn take(p: *anyopaque, node: *cuda.xchg.Node, _: usize, _: usize) !void {
        const r: *FRec = @ptrCast(@alignCast(p));
        r.takes += 1;
        if (r.fail_take == r.takes) node.fail(.{ .why = .silent, .peer = 1, .seq = @intCast(r.posts) });
    }

    /// The launches recorded so far.
    fn mark(r: *FRec) !usize {
        try r.f.flush();
        return r.trace.events.items.len;
    }

    fn state(r: *FRec, slot: u8) *glm.buffers.State {
        return if (slot == 0) &r.e.st else &r.users.states[slot - 1];
    }

    /// A decoder at `pos` on `slot` (generation 1, its State's position set, as the concurrent_users recorders do).
    fn decoder(r: *FRec, slot: u8, pos: usize) void {
        r.users.generation[slot] = 1;
        r.state(slot).pos = pos;
        r.state(slot).parity = @intCast(pos % 2);
    }
};

fn ids(n: usize) ![]u32 {
    const v = try gpa.alloc(u32, n);
    for (v, 0..) |*x, i| x.* = @intCast(1000 + i % 37);
    return v;
}

/// A buffer's name with python's temporaries' counters dropped (`tmp.clone.3` → `tmp.clone`, `tmp.topk.2.values` →
/// `tmp.topk.values`): an interleaved run makes the same temporaries in another order of ops; on a GPU naming does nothing.
fn normName(path: []const u8, buf: []u8) []const u8 {
    if (!std.mem.startsWith(u8, path, "tmp.")) return path;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, path, '.');
    while (it.next()) |part| {
        const digits = part.len > 0 and for (part) |c| {
            if (c < '0' or c > '9') break false;
        } else true;
        if (digits) continue;
        if (n > 0) {
            buf[n] = '.';
            n += 1;
        }
        @memcpy(buf[n..][0..part.len], part);
        n += part.len;
    }
    return buf[0..n];
}

fn expectAddr(sa: *const glm.trace.Space, x: u64, sb: *const glm.trace.Space, y: u64) !void {
    const p = sa.name(x);
    const q = sb.name(y);
    try t.expectEqual(p == null, q == null);
    if (p == null) return t.expectEqual(x, y);
    var bp: [128]u8 = undefined;
    var bq: [128]u8 = undefined;
    try t.expectEqualStrings(normName(p.?.path, &bp), normName(q.?.path, &bq));
    try t.expectEqual(p.?.off, q.?.off);
}

fn expectVal(sa: *const glm.trace.Space, x: glm.trace.Val, sb: *const glm.trace.Space, y: glm.trace.Val) !void {
    try t.expectEqual(std.meta.activeTag(x), std.meta.activeTag(y));
    switch (x) {
        .t => |v| {
            try expectAddr(sa, v.ptr, sb, y.t.ptr);
            try t.expectEqual(v.dtype, y.t.dtype);
            try t.expectEqual(v.dims, y.t.dims);
            try t.expectEqual(v.shape, y.t.shape);
            try t.expectEqual(v.stride, y.t.stride);
        },
        .list => |l| {
            try t.expectEqual(l.len, y.list.len);
            for (l, y.list) |u, v| try expectVal(sa, u, sb, v);
        },
        else => try t.expectEqualDeep(x, y),
    }
}

/// serve_rec.expectSame's comparison (op, name, grid, arguments by buffer name, constexprs, values) with temporaries by
/// `normName`.
fn expectSameNorm(sa: *const glm.trace.Space, a: []const glm.trace.Event, sb: *const glm.trace.Space, b: []const glm.trace.Event) !void {
    try t.expectEqual(a.len, b.len);
    for (a, b, 0..) |x, y, i| {
        errdefer std.debug.print("launch {d}: {s} against {s}\n", .{ i, x.name, y.name });
        try t.expectEqual(x.op, y.op);
        try t.expectEqualStrings(x.name, y.name);
        try t.expectEqual(x.grid, y.grid);
        try t.expectEqual(x.args.len, y.args.len);
        for (x.args, y.args) |u, v| {
            try t.expectEqualStrings(u.name, v.name);
            try t.expectEqual(std.meta.activeTag(u.value), std.meta.activeTag(v.value));
            switch (u.value) {
                .ptr => |p| {
                    try t.expectEqualStrings(p.ty, v.value.ptr.ty);
                    try expectAddr(sa, p.addr, sb, v.value.ptr.addr);
                },
                else => try t.expectEqualDeep(u.value, v.value),
            }
        }
        try t.expectEqualDeep(x.consts, y.consts);
        try t.expectEqual(x.vals.len, y.vals.len);
        for (x.vals, y.vals) |u, v| try expectVal(sa, u, sb, v);
    }
}

/// The same launches with the same operands, each kind of launch (by name) in the same order; where kinds interleave
/// may differ (the pairs' and lanes' `behind` rule, WIRE.md §12-13: a pair or group right behind a pair or group sends its
/// next request after the first take instead of before it; with another user's forward between them it is not behind).
/// Gives how many launches stand elsewhere, and their names in `moved`.
fn expectSameByName(sa: *const glm.trace.Space, a: []const glm.trace.Event, sb: *const glm.trace.Space, b: []const glm.trace.Event, moved: *std.ArrayList([]const u8)) !usize {
    try t.expectEqual(a.len, b.len);
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    var differ: usize = 0;
    for (a, b) |x, y| {
        if (!std.mem.eql(u8, x.name, y.name)) {
            differ += 1;
            for ([_][]const u8{ x.name, y.name }) |nm| {
                for (moved.items) |m| {
                    if (std.mem.eql(u8, m, nm)) break;
                } else try moved.append(gpa, nm);
            }
        }
        for (names.items) |n| {
            if (std.mem.eql(u8, n, x.name)) break;
        } else try names.append(gpa, x.name);
    }
    var xa: std.ArrayList(glm.trace.Event) = .empty;
    defer xa.deinit(gpa);
    var yb: std.ArrayList(glm.trace.Event) = .empty;
    defer yb.deinit(gpa);
    for (names.items) |n| {
        xa.clearRetainingCapacity();
        yb.clearRetainingCapacity();
        for (a) |x| if (std.mem.eql(u8, x.name, n)) try xa.append(gpa, x);
        for (b) |y| if (std.mem.eql(u8, y.name, n)) try yb.append(gpa, y);
        errdefer std.debug.print("launches named {s}: {d} against {d}\n", .{ n, xa.items.len, yb.items.len });
        try expectSameNorm(sa, xa.items, sb, yb.items);
    }
    return differ;
}

const Case = struct { path: Path, len: usize, steps: usize };

/// The lanes' and the pairs' hooks as one sink (cuda_afd_lanes_test.zig's and cuda_afd_pairs_rec.zig's Logs): every
/// step's mark, where its launches start, and the chunk it belongs to (a group's lanes or a pair's A and B, then a lone
/// chunk, in prompt order; the group's or pair's own steps, its position fills and the first token, belong to none). An
/// `end` mark closes each prompt step, so a decode round between two steps belongs to no chunk.
const LLog = struct {
    t: *glm.trace.Trace,
    marks: std.ArrayList(Mark) = .empty,
    base: usize = 0,
    seen: usize = 0,

    const Mark = struct { end: bool, at: usize, chunk: ?usize };

    fn deinit(l: *LLog) void {
        l.marks.deinit(gpa);
    }

    fn lanesHook(l: *LLog) glm.afd_lanes.Hook {
        return .{ .ctx = l, .at = atLanes };
    }

    fn pairsHook(l: *LLog) glm.afd_pairs.Hook {
        return .{ .ctx = l, .at = atPairs };
    }

    fn put(l: *LLog, chunk: ?usize, closes: bool) !void {
        try l.t.flush();
        if (chunk) |c| l.seen = @max(l.seen, c + 1);
        try l.marks.append(gpa, .{ .end = closes, .at = l.t.events.items.len, .chunk = chunk });
    }

    fn atLanes(ctx: *anyopaque, who: glm.afd_lanes.Who, k: u8, step: glm.afd_pairs.Step, _: usize) anyerror!void {
        const l: *LLog = @ptrCast(@alignCast(ctx));
        if (who == .lane and k == 0 and step == .stage) l.base = l.seen;
        try l.put(switch (who) {
            .lane => l.base + k,
            .solo => l.seen,
            .group => null,
        }, false);
    }

    fn atPairs(ctx: *anyopaque, who: glm.afd_pairs.Who, step: glm.afd_pairs.Step, _: usize) anyerror!void {
        const l: *LLog = @ptrCast(@alignCast(ctx));
        if (who == .a and step == .stage) l.base = l.seen;
        try l.put(switch (who) {
            .a => l.base,
            .b => l.base + 1,
            .solo => l.seen,
            .pair => null,
        }, false);
    }

    fn end(l: *LLog) !void {
        try l.put(null, true);
    }

    /// The launches of chunk `k` (or, null, of the groups' or pairs' own steps), in order.
    fn of(l: *const LLog, k: ?usize, out: *std.ArrayList(glm.trace.Event)) !void {
        out.clearRetainingCapacity();
        const ev = l.t.events.items;
        const m = l.marks.items;
        for (m, 0..) |mk, i| {
            if (mk.end) continue;
            if (k) |c| {
                if (mk.chunk != c) continue;
            } else if (mk.chunk != null) continue;
            try out.appendSlice(gpa, ev[mk.at..if (i + 1 < m.len) m[i + 1].at else ev.len]);
        }
    }
};

/// The served cut at 64-row chunks: plain one chunk a step; pairs a pair (64 + the rest up to 64) or a lone last chunk;
/// lanes a group of up to N chunks, a remainder of two a pair's group, a lone last chunk alone.
const cases = [_]Case{
    .{ .path = .plain, .len = 1, .steps = 1 },    .{ .path = .plain, .len = 64, .steps = 1 },   .{ .path = .plain, .len = 65, .steps = 2 },
    .{ .path = .plain, .len = 200, .steps = 4 },  .{ .path = .pairs, .len = 65, .steps = 1 },   .{ .path = .pairs, .len = 129, .steps = 2 },
    .{ .path = .pairs, .len = 300, .steps = 3 },  .{ .path = .lanes3, .len = 222, .steps = 2 }, .{ .path = .lanes3, .len = 384, .steps = 2 },
    .{ .path = .lanes4, .len = 129, .steps = 1 }, .{ .path = .lanes4, .len = 286, .steps = 2 }, .{ .path = .lanes4, .len = 600, .steps = 3 },
};

test "prefill_interleave device: Users.prefillStep is Users.prefill launch for launch on State0 and State1 (plain chunks, qualification_0f991257 pairs, qualification_9e49592e lanes of 3 and 4 at 64-row chunks): the same first token, headers, positions and parities" {
    for (cases) |c| for ([_]u8{ 0, 1 }) |slot| {
        const prompt = try ids(c.len);
        defer gpa.free(prompt);
        var x: FRec = undefined;
        try x.init(c.path, 2);
        defer x.deinit();
        var y: FRec = undefined;
        try y.init(c.path, 2);
        defer y.deinit();
        const key: Key = .{ .slot = slot, .generation = 1 };
        const tok_x = try x.users.prefill(key, prompt);
        var steps: usize = 0;
        const tok_y = while (true) {
            steps += 1;
            if (try y.users.prefillStep(key, prompt)) |tok| break tok;
            try t.expect(y.users.fill[slot] != null);
        };
        try t.expectEqual(c.steps, steps);
        try t.expectEqual(tok_x, tok_y);
        try t.expect(y.users.fill[slot] == null);
        try x.f.flush();
        try y.f.flush();
        try glm.serve_rec.expectSame(&x.space, x.trace.events.items, &y.space, y.trace.events.items);
        try t.expectEqual(x.posts, y.posts);
        for (x.hdrs[0..x.posts], y.hdrs[0..y.posts]) |hx, hy| try t.expectEqual(hx, hy);
        try t.expectEqual(x.state(slot).pos, y.state(slot).pos);
        try t.expectEqual(c.len, y.state(slot).pos);
        try t.expectEqual(x.state(slot).parity, y.state(slot).parity);
        try t.expectEqual(@as(u64, 1), y.users.prefills);
        try t.expectEqual(x.at.forwards, y.at.forwards);
    };
    std.debug.print("prefill_interleave device: {d} cases x State0/State1: stepwise = whole, launch for launch, headers and positions\n", .{cases.len});
}

const Inter = struct { path: Path, len: usize, steps: usize, dec: u8, fill: u8, pos: usize };

/// A decoder's rounds between the prompt's steps (one round between every two, as 1:1) against the serial order (the
/// decoder's rounds, then the whole pass) on another recorder: the prompt's launches, each round's, the wire's headers.
fn interleaved(c: Inter) !void {
    const prompt = try ids(c.len);
    defer gpa.free(prompt);
    var x: FRec = undefined; // serial
    try x.init(c.path, 2);
    defer x.deinit();
    var y: FRec = undefined; // interleaved
    try y.init(c.path, 2);
    defer y.deinit();
    x.decoder(c.dec, c.pos);
    y.decoder(c.dec, c.pos);
    var lx: LLog = .{ .t = &x.trace };
    defer lx.deinit();
    var ly: LLog = .{ .t = &y.trace };
    defer ly.deinit();
    switch (c.path) {
        .plain => {},
        .pairs => {
            x.p.hook = lx.pairsHook();
            y.p.hook = ly.pairsHook();
        },
        .lanes3, .lanes4 => {
            x.l.hook = lx.lanesHook();
            y.l.hook = ly.lanesHook();
        },
    }
    const dk: Key = .{ .slot = c.dec, .generation = 1 };
    const fk: Key = .{ .slot = c.fill, .generation = 1 };
    const R = c.steps - 1;
    var xr: [16][2]usize = undefined;
    var yr: [16][2]usize = undefined;
    var xo: [16]u32 = undefined;
    var yo: [16]u32 = undefined;
    var token: u32 = 10;
    for (0..R) |i| {
        xr[i][0] = try x.mark();
        var out: [1]u32 = undefined;
        try x.users.round(&.{.{ .key = dk, .token = token, .position = c.pos + i }}, &out);
        xr[i][1] = try x.mark();
        xo[i] = out[0];
        token = out[0] +% 1;
    }
    const xp0 = try x.mark();
    const tok_x = try x.users.prefill(fk, prompt);
    const xp1 = try x.mark();
    try lx.end();
    var yp: std.ArrayList(glm.trace.Event) = .empty; // the prompt's launches across its steps
    defer yp.deinit(gpa);
    var modes: std.ArrayList(proto.Mode) = .empty;
    defer modes.deinit(gpa);
    token = 10;
    var tok_y: ?u32 = null;
    var steps: usize = 0;
    while (tok_y == null) {
        const s0 = try y.mark();
        const p0 = y.posts;
        tok_y = try y.users.prefillStep(fk, prompt);
        try yp.appendSlice(gpa, y.trace.events.items[s0..try y.mark()]);
        try ly.end();
        for (y.hdrs[p0..y.posts]) |h| try t.expect(h.mode != .eager); // a prompt step's forwards
        steps += 1;
        if (tok_y != null) break;
        const i = steps - 1;
        const r0 = y.posts;
        yr[i][0] = try y.mark();
        var out: [1]u32 = undefined;
        try y.users.round(&.{.{ .key = dk, .token = token, .position = c.pos + i }}, &out);
        yr[i][1] = try y.mark();
        yo[i] = out[0];
        token = out[0] +% 1;
        try t.expectEqual(@as(usize, 2), y.posts - r0); // the round's two MoE layers, R1 eager
        for (y.hdrs[r0..y.posts]) |h| {
            try t.expectEqual(proto.Mode.eager, h.mode);
            try t.expectEqual(@as(i32, 1), h.rows);
        }
    }
    try t.expectEqual(c.steps, steps);
    try t.expectEqual(tok_x, tok_y.?);
    // the prompt: plain chunks launch for launch; pairs and lanes each kind of launch in order (the behind rule's place
    // of a request aside: in the serial order a group follows a group, interleaved a decode forward stands between)
    var moved: std.ArrayList([]const u8) = .empty;
    defer moved.deinit(gpa);
    const differ = try expectSameByName(&x.space, x.trace.events.items[xp0..xp1], &y.space, yp.items, &moved);
    if (c.path == .plain) {
        try t.expectEqual(@as(usize, 0), differ);
        try expectSameNorm(&x.space, x.trace.events.items[xp0..xp1], &y.space, yp.items);
    }
    if (c.path != .plain) { // qualification_0f991257/qualification_9e49592e's identity: every chunk's launches the serial pass's same chunk, in order
        try t.expectEqual(lx.seen, ly.seen);
        try t.expect(lx.seen >= 2);
        var ex: std.ArrayList(glm.trace.Event) = .empty;
        defer ex.deinit(gpa);
        var ey: std.ArrayList(glm.trace.Event) = .empty;
        defer ey.deinit(gpa);
        for (0..lx.seen) |k| {
            try lx.of(k, &ex);
            try ly.of(k, &ey);
            try t.expect(ex.items.len > 0);
            errdefer std.debug.print("chunk {d}: {d} launches serial, {d} interleaved\n", .{ k, ex.items.len, ey.items.len });
            try expectSameNorm(&x.space, ex.items, &y.space, ey.items);
        }
        try lx.of(null, &ex); // the position fills and the first token, in order
        try ly.of(null, &ey);
        try expectSameNorm(&x.space, ex.items, &y.space, ey.items);
        try expectSameNorm(&x.space, x.trace.events.items[xp0..lx.marks.items[0].at], &y.space, y.trace.events.items[0..ly.marks.items[0].at]); // State.reset's
    }
    var mb: [256]u8 = undefined;
    var mn: usize = 0;
    for (moved.items) |nm| {
        if (mn + nm.len + 1 > mb.len) break;
        @memcpy(mb[mn..][0..nm.len], nm);
        mb[mn + nm.len] = ' ';
        mn += nm.len + 1;
    }
    if (differ > 0) std.debug.print("prefill_interleave device {s} {d} dec{d}@{d}: chunk for chunk the serial pass's; {d} of {d} prompt launches stand elsewhere across kinds (the behind rule's flow): {s}\n", .{ @tagName(c.path), c.len, c.dec, c.pos, differ, yp.items.len, mb[0..mn] });
    for (0..R) |i| {
        try expectSameNorm(&x.space, x.trace.events.items[xr[i][0]..xr[i][1]], &y.space, y.trace.events.items[yr[i][0]..yr[i][1]]);
        try t.expectEqual(xo[i], yo[i]);
    }
    try t.expectEqual(x.posts, y.posts);
    try t.expectEqual(c.pos + R, y.state(c.dec).pos);
    try t.expectEqual(x.state(c.dec).parity, y.state(c.dec).parity);
    try t.expectEqual(c.len, y.state(c.fill).pos);
    try t.expectEqual(x.state(c.fill).parity, y.state(c.fill).parity);
    try t.expectEqual(x.users.rounds, y.users.rounds);
}

test "prefill_interleave device: a decoder's rounds between another user's prompt steps (served lanes of 4, and plain chunks): the prompt's launches are its solo pass's, each round's its solo round's, the wire alternates prompt and R1 decode forwards; State0/State1 at the 512 and 2048 boundaries" {
    const shapes = [_][3]usize{ .{ @intFromEnum(Path.lanes4), 1100, 5 }, .{ @intFromEnum(Path.plain), 200, 4 }, .{ @intFromEnum(Path.pairs), 300, 3 } };
    var n: usize = 0;
    for (shapes) |sh| for ([_][2]u8{ .{ 0, 1 }, .{ 1, 0 } }) |slots| for ([_]usize{ 511, 2047 }) |pos| {
        try interleaved(.{ .path = @enumFromInt(sh[0]), .len = sh[1], .steps = sh[2], .dec = slots[0], .fill = slots[1], .pos = pos });
        n += 1;
    };
    std.debug.print("prefill_interleave device: {d} interleaved cases (lanes4/plain/pairs x decoder State0/State1 x 511/2047): prompt = solo pass, rounds = solo rounds, headers alternate\n", .{n});
}

test "prefill_interleave device: a release between steps clears the pass; the slot's next generation starts from its reset and is a fresh whole pass; the decoder's rounds are unaffected" {
    const first = try ids(600);
    defer gpa.free(first);
    const next = try ids(286);
    defer gpa.free(next);
    for ([_][2]u8{ .{ 0, 1 }, .{ 1, 0 } }) |slots| {
        var x: FRec = undefined; // a fresh recorder: the next generation's whole pass
        try x.init(.lanes4, 2);
        defer x.deinit();
        var y: FRec = undefined;
        try y.init(.lanes4, 2);
        defer y.deinit();
        y.decoder(slots[0], 511);
        x.decoder(slots[0], 511);
        const dk: Key = .{ .slot = slots[0], .generation = 1 };
        try t.expect(try y.users.prefillStep(.{ .slot = slots[1], .generation = 1 }, first) == null);
        var out: [1]u32 = undefined;
        try y.users.round(&.{.{ .key = dk, .token = 10, .position = 511 }}, &out);
        try t.expect(try y.users.prefillStep(.{ .slot = slots[1], .generation = 1 }, first) == null);
        y.users.release(.{ .slot = slots[1], .generation = 1 }); // cancelled between its second and third steps
        try t.expect(y.users.fill[slots[1]] == null);
        try t.expectError(error.StaleGeneration, y.users.prefillStep(.{ .slot = slots[1], .generation = 1 }, first)); // the old generation never resumes
        try y.users.round(&.{.{ .key = dk, .token = 11, .position = 512 }}, &out);
        const s0 = try y.mark();
        var steps: usize = 0;
        const tok_y = while (true) {
            steps += 1;
            if (try y.users.prefillStep(.{ .slot = slots[1], .generation = 2 }, next)) |tok| break tok;
        };
        const s1 = try y.mark();
        const x0 = try x.mark();
        const tok_x = try x.users.prefill(.{ .slot = slots[1], .generation = 1 }, next);
        try expectSameNorm(&x.space, x.trace.events.items[x0..try x.mark()], &y.space, y.trace.events.items[s0..s1]);
        try t.expectEqual(tok_x, tok_y);
        try t.expectEqual(@as(usize, 2), steps);
        try t.expectEqual(@as(usize, 286), y.state(slots[1]).pos);
        try t.expectEqual(@as(usize, 513), y.state(slots[0]).pos);
        try t.expectEqual(@as(u64, 2), y.users.prefills);
    }
}

test "prefill_interleave device: a failed exchange inside a prompt step is sticky: at the latest the next call (the decoder's round, a later step, a new generation's step) is refused with nothing more sent, as a whole pass's failure is before its first token" {
    const prompt = try ids(600);
    defer gpa.free(prompt);
    var y: FRec = undefined;
    try y.init(.lanes4, 2);
    defer y.deinit();
    y.decoder(0, 511);
    const dk: Key = .{ .slot = 0, .generation = 1 };
    const fk: Key = .{ .slot = 1, .generation = 1 };
    try t.expect(try y.users.prefillStep(fk, prompt) == null);
    var out: [1]u32 = undefined;
    try y.users.round(&.{.{ .key = dk, .token = 10, .position = 511 }}, &out);
    y.fail_take = y.takes + 3; // inside the next group's first exchanges
    if (y.users.prefillStep(fk, prompt)) |got| {
        try t.expect(got == null); // a step that is not the last has no sync of its own: the failure shows at the next call
    } else |err| try t.expectEqual(error.ExpertFailed, err);
    try t.expect(y.at.failure() != null);
    try y.f.flush();
    const posts = y.posts;
    const events = y.trace.events.items.len;
    try t.expectError(error.ExpertFailed, y.users.round(&.{.{ .key = dk, .token = 11, .position = 512 }}, &out));
    try t.expectError(error.ExpertFailed, y.users.prefillStep(fk, prompt));
    y.users.release(fk);
    try t.expectError(error.ExpertFailed, y.users.prefillStep(.{ .slot = 1, .generation = 2 }, prompt));
    try y.f.flush();
    try t.expectEqual(posts, y.posts);
    try t.expectEqual(events, y.trace.events.items.len);
    try t.expectEqual(@as(usize, 512), y.state(0).pos);
}
