//! qualification_qualification_44ba554e.8 day 3's CPU gates for drafts in the served stream (glm_cuda_drafts.zig): two recording TP ranks of the prompt
//! fixture's model with its MTP head (rank 0 and rank 1 of world 2) over an in-memory link, and a toy target answering
//! every sample call (cuda_decode_test's Toy: the token at each position by a hash, a draft right unless another hash
//! says not). Rank 0 decodes each lone stream by qualification_qualification_86ea5170's loop for the policy (glm_cuda_drafts.Lone) with every operation
//! mirrored; rank 1 follows in a thread. A drafted request, a cancel inside a verify window, a second request and one
//! past the drafted reach (handed to the lane core's one-row rounds) give the same launches on both ranks; each rank's
//! are tf-glm-forward `--drafts`'s for the same ids and policy (decode.prefill with the head, glm_drafts.runWith), and
//! the replies are its tokens.
const std = @import("std");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const tool = @import("glm_tool");
const gl = @import("glm_cuda_lanes.zig");
const gp = @import("glm_cuda_pair.zig");
const gd = @import("glm_cuda_drafts.zig");
const rec = glm.serve_rec;
const trace = glm.trace;
const decode = glm.decode;
const Serve = glm.serve.Serve;
const Pair = glm.ranks.Pair;
const E = glm.engine.Engine;
const Steps = gp.Armed(Serve);

const testing = std.testing;
const gpa = testing.allocator;

const Kind = enum { first, main, draft };

/// Which rows a sample call reads (cuda_decode_test.kindOf): prefill's clone, a verify window's, or the MTP head's.
fn kindOf(e: *const E, logits: trace.T) Kind {
    if (logits.ptr == e.buf.logits.ptr) return .main;
    if (e.mbuf) |*mb| if (logits.ptr == mb.logits.ptr) return .draft;
    return .first;
}

/// The absolute position a sample call's row answers (cuda_decode_test.position).
fn position(e: *const E, kind: Kind, r: usize) usize {
    return switch (kind) {
        .first => e.st.pos,
        .main => e.st.pos + 1 + r,
        .draft => e.st.pos + 1 + e.st.mtp_drafted,
    };
}

/// The toy target: never an eos (each reply runs to its length); `cancel_at` verify windows raise `cancelled`.
const Toy = struct {
    e: *const E,
    seed: u64,
    miss: u64,
    mains: usize = 0,
    cancel_at: ?usize = null,
    cancelled: bool = false,

    fn hash(t: *const Toy, p: usize, salt: u64) u64 {
        var h = std.hash.Wyhash.init(t.seed ^ salt);
        h.update(std.mem.asBytes(&p));
        return h.final();
    }

    fn truth(t: *const Toy, p: usize) u32 {
        const v: u32 = @intCast(t.hash(p, 1) % 150000);
        return if (std.mem.indexOfScalar(u32, t.e.m.eos[0..t.e.m.eos_n], v) != null) v + 1 else v;
    }

    fn pick(ctx: *anyopaque, logits: trace.T, _: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        const t: *Toy = @ptrCast(@alignCast(ctx));
        const kind = kindOf(t.e, logits);
        if (kind == .main) {
            t.mains += 1;
            if (t.cancel_at) |n| if (t.mains == n) {
                t.cancelled = true;
            };
        }
        for (out, 0..) |*o, r| {
            const p = position(t.e, kind, r);
            o.* = t.truth(p);
            if (kind == .draft and t.hash(p, 2) % 1000 < t.miss) o.* +%= 1 + @as(u32, @intCast(p % 5));
            if (probs) |ps| ps[r] = 0.05 + @as(f64, @floatFromInt(t.hash(p, 3) % 950)) / 1000.0;
        }
    }

    fn isCancelled(ctx: *anyopaque) bool {
        const t: *Toy = @ptrCast(@alignCast(ctx));
        return t.cancelled;
    }
};

/// A recording rank with the MTP head: a Serve that absorbs at prefill, its toy target answering its samples.
const Side = struct {
    r: *rec.Rank,
    f: glm.fwd.Fwd,
    s: Serve,
    toy: Toy,

    fn init(rank: usize) !*Side {
        const x = try gpa.create(Side);
        errdefer gpa.destroy(x);
        x.r = try rec.Rank.initWith(gpa, rank, null, true);
        x.f = x.r.recording();
        x.s = .{ .e = x.r.e, .f = &x.f, .absorb = true };
        x.toy = .{ .e = x.r.e, .seed = 17, .miss = 300 };
        x.r.t.picker = .{ .ctx = &x.toy, .pick = Toy.pick };
        return x;
    }

    fn deinit(x: *Side) void {
        x.r.deinit(gpa);
        gpa.destroy(x);
    }

    fn events(x: *Side) ![]const trace.Event {
        try x.f.flush();
        return x.r.t.events.items;
    }
};

fn followThread(side: *Side, pair: *Pair, d: gd.Drafts, done: *gp.Followed, failed: *?anyerror) void {
    gp.follow(Serve, &side.s, pair, gpa, done, d) catch |e| {
        failed.* = e;
        pair.link.?.close();
        pair.link = null;
    };
}

fn ids(comptime n: usize, base: u32, stride: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*p, i| p.* = base + stride * @as(u32, @intCast(i));
    return out;
}

const Served = struct { tokens: [64]u32 = undefined, n: usize = 0, reason: lanes.stream.Reason = .length, rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, handed: bool = false };

fn noop(_: *anyopaque) void {}

/// One request as the lane host runs a lone greedy drafted stream: Lone.run, then the lane core's rounds when it hands
/// the stream over (past the drafted reach); `cancel_at` verify windows: the request is cancelled inside that window.
fn serveOne(lone: *gd.Lone(Steps), toy: *Toy, prompt: []const u32, max_new: u32, cancel_at: ?usize) !Served {
    toy.mains = 0;
    toy.cancel_at = cancel_at;
    toy.cancelled = false;
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = max_new, .eos = &.{154_820}, .drafts = true, .cancel_check = .{ .ptr = toy, .check = Toy.isCancelled } });
    defer s.deinit(gpa);
    var out: Served = .{};
    var ctx: u8 = 0;
    out.handed = lone.run(&s, &ctx, noop) catch |e| blk: {
        if (e != error.Cancelled) return e;
        s.reason = .cancelled;
        break :blk false;
    };
    if (out.handed) {
        var cfg = try lanes.Config.init(gpa, lone.back.facts(), 1, 0);
        defer cfg.deinit(gpa);
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, lone.back.backend(), clock.clock());
        defer core.deinit();
        try core.adopt(&s);
        while (core.activeCount() > 0) try core.step();
    }
    const em = s.emitted();
    @memcpy(out.tokens[0..em.len], em);
    out.n = em.len;
    out.reason = s.reason;
    out.rounds = s.rounds;
    out.drafted = s.drafted;
    out.accepted = s.accepted;
    return out;
}

/// tf-glm-forward `--drafts`'s calls for one request on a lone rank: decode.prefill with the head, then runWith's
/// loop for `count` tokens (stopped after `rounds` rounds when set: the served stream's cancel), or `steps` one-row
/// steps (a request past the drafted reach).
const Stop = struct {
    rounds: usize,
    seen: usize = 0,
    fn before(_: *anyopaque, _: decode.Hook.Op, _: []const u32) anyerror!void {}
    fn kept(p: *anyopaque, _: []const u32, _: usize, _: usize) anyerror!bool {
        const x: *Stop = @ptrCast(@alignCast(p));
        x.seen += 1;
        return x.seen >= x.rounds;
    }
};

fn toolRun(side: *Side, d: gd.Drafts, prompt: []const u32, count: usize, rounds: ?usize, steps: ?usize, out: []u32) ![]const u32 {
    side.toy.cancel_at = null;
    const first = try side.s.prefill(prompt);
    defer side.s.release();
    if (steps) |n| {
        out[0] = first;
        for (1..n + 1) |k| out[k] = try side.s.step(out[k - 1]);
        return out[0 .. n + 1];
    }
    var stop: Stop = .{ .rounds = rounds orelse std.math.maxInt(usize) };
    const hook: decode.Hook = .{ .ctx = &stop, .before = Stop.before, .kept = Stop.kept };
    const r = try tool.drafts.runWith(side.s.e, side.s.f, d, first, count, out, null, if (rounds != null) &hook else null);
    return r.tokens;
}

fn scenario(spec: []const u8) !void {
    var problem: []const u8 = "";
    const d = (try gd.parse(gpa, spec, &problem)).?;
    const r0 = try Side.init(0);
    defer r0.deinit();
    const r1 = try Side.init(1);
    defer r1.deinit();
    const links = try gp.memoryLinks();
    var p0: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 0, .link = links[0] };
    var p1: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 1, .link = links[1] };
    var done: gp.Followed = .{};
    var failed: ?anyerror = null;
    const t = try std.Thread.spawn(.{}, followThread, .{ r1, &p1, d, &done, &failed });
    var sender: gp.Sender = .{ .gpa = gpa, .pair = &p0 };
    var armed: Steps = .{ .steps = &r0.s, .pair = &p0 };
    const cap = r0.r.e.st.capacity;
    var b: gl.Lanes(Steps) = .{ .steps = &armed, .mirror = sender.mirror(), .capacity = cap, .reach = cap };
    var lone: gd.Lone(Steps) = .{ .gpa = gpa, .back = &b, .d = d, .reach = cap };
    const first = ids(37, 1000, 7);
    const cancelled = ids(20, 3000, 11);
    const second = ids(9, 5000, 3);
    const far = ids(12, 7000, 5);
    // the scenario, then rank 0's stop (rank 1 acks and its follower returns), before any check can leave the test
    const a = serveOne(&lone, &r0.toy, &first, 24, null); // drafted to its length
    const c = serveOne(&lone, &r0.toy, &cancelled, 40, 3); // cancelled inside its third verify window
    const e2 = serveOne(&lone, &r0.toy, &second, 6, null);
    lone.reach = far.len + 4; // past the drafted reach: the absorbing prompt pass, then the lane core's one-row rounds
    const h = serveOne(&lone, &r0.toy, &far, 5, null);
    gp.stop(&p0);
    t.join();
    gp.closeFollower(&p1);
    try testing.expectEqual(@as(?anyerror, null), failed);
    const ra = try a;
    const rc = try c;
    const re = try e2;
    const rh = try h;
    try testing.expectEqual(@as(usize, 24), ra.n);
    try testing.expectEqual(lanes.stream.Reason.length, ra.reason);
    try testing.expectEqual(lanes.stream.Reason.cancelled, rc.reason);
    try testing.expect(rc.n > 1 and rc.n < 40);
    try testing.expectEqual(@as(usize, 6), re.n);
    try testing.expect(rh.handed and rh.n == 5 and rh.reason == .length);
    // the replies are the target's tokens (drafted == serial); the counters landed in the streams
    for ([_]Served{ ra, rc, re, rh }, [_][]const u32{ &first, &cancelled, &second, &far }) |x, p| {
        for (x.tokens[0..x.n], 0..) |tok, i| try testing.expectEqual(r0.toy.truth(p.len + i), tok);
    }
    try testing.expect(ra.drafted > 0 and ra.accepted > 0 and ra.accepted <= ra.drafted and ra.rounds + ra.accepted >= 23);
    try testing.expectEqual(@as(u64, 3), rc.rounds);
    try testing.expectEqual(@as(u64, 0), rh.drafted);
    // rank 1 heard every frame: a begin and a release a request, the drafted decodes and their operations
    try testing.expectEqual(done.rounds, sender.sent);
    try testing.expectEqual(@as(u64, 4), done.begins);
    try testing.expectEqual(@as(u64, 4), done.releases);
    try testing.expectEqual(@as(u64, 3), done.decodes);
    try testing.expectEqual(@as(u64, rh.n - 1), done.steps);
    try testing.expectEqual(ra.rounds + rc.rounds + re.rounds, done.windows);
    try testing.expectEqual(done.windows, done.keeps);
    try testing.expect(done.drafts > 0);
    // both ranks made the same launches, in lockstep (only rank-own scalars may differ)
    const e0 = try r0.events();
    const e1 = try r1.events();
    const l = try rec.expectLockstep(&r0.r.space, e0, &r1.r.space, e1);
    std.debug.print("drafts {s}: two ranks, {d} launches each, in lockstep; {d} scalar values differ (the ranks' own) in:", .{ spec, e0.len, l.differ });
    for (l.names[0..l.n]) |x| std.debug.print(" {s}", .{x});
    std.debug.print("; rounds {d}/{d}/{d}, drafted {d}, accepted {d}; rank 1: {d} frames ({d} draft, {d} window, {d} keep)\n", .{ ra.rounds, rc.rounds, re.rounds, ra.drafted + rc.drafted + re.drafted, ra.accepted + rc.accepted + re.accepted, done.rounds, done.drafts, done.windows, done.keeps });
    for (l.names[0..l.n]) |x| try testing.expectEqualStrings("torch.add", x);
    // each rank's launches are tf-glm-forward --drafts's for the same ids and policy, and its replies the tool's tokens
    for ([_]*Side{ r0, r1 }, [_]usize{ 0, 1 }) |side, rank| {
        const ref = try Side.init(rank);
        defer ref.deinit();
        var out: [64]u32 = undefined;
        try testing.expectEqualSlices(u32, ra.tokens[0..ra.n], try toolRun(ref, d, &first, 24, null, null, &out));
        try testing.expectEqualSlices(u32, rc.tokens[0..rc.n], (try toolRun(ref, d, &cancelled, 40, 3, null, &out))[0..rc.n]);
        try testing.expectEqualSlices(u32, re.tokens[0..re.n], try toolRun(ref, d, &second, 6, null, null, &out));
        try testing.expectEqualSlices(u32, rh.tokens[0..rh.n], try toolRun(ref, d, &far, 0, null, rh.n - 1, &out));
        try rec.expectSame(&ref.r.space, try ref.events(), &side.r.space, try side.events());
    }
}

test "drafts auto in the served stream: two recording ranks, a drafted request, a cancel inside a verify window, a second request and one past the drafted reach; each rank's launches tf-glm-forward --drafts auto's" {
    try scenario("auto");
}

test "drafts a (python's adaptive mtp_decode) and c3:0.35 in the served stream: the same, on the mtp loop" {
    try scenario("a");
    try scenario("c3:0.35");
}

test "drafts on: the warm-up runs qualification_qualification_86ea5170's loop on both ranks, recorded, its tokens the drafts-off warm-up's" {
    var problem: []const u8 = "";
    const d = (try gd.parse(gpa, "auto", &problem)).?;
    const r0 = try Side.init(0);
    defer r0.deinit();
    const r1 = try Side.init(1);
    defer r1.deinit();
    var p0: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 0 };
    var p1: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 1 };
    var w0: [gl.warm_tokens]u32 = undefined;
    var w1: [gl.warm_tokens]u32 = undefined;
    const x0 = try gd.warmUp(Serve, &r0.s, &p0, d, &w0);
    _ = try gd.warmUp(Serve, &r1.s, &p1, d, &w1);
    try testing.expectEqualSlices(u32, &w0, &w1);
    try testing.expect(x0.drafted > 0 and !r0.s.open);
    _ = try rec.expectLockstep(&r0.r.space, try r0.events(), &r1.r.space, try r1.events());
    // the drafts-off warm-up on the same target gives the same tokens (greedy: drafts change the launches only)
    const off = try Side.init(0);
    defer off.deinit();
    off.s.absorb = false;
    var w: [gl.warm_tokens]u32 = undefined;
    try gl.warmUp(Serve, &off.s, &w);
    try testing.expectEqualSlices(u32, &w, &w0);
}

test "a drafted decode's frames out of step: rank 1 refuses a frame whose words are not its own, and a lost rank 0 is an error, never a hang" {
    var problem: []const u8 = "";
    const d = (try gd.parse(gpa, "f3", &problem)).?;
    const r1 = try Side.init(1);
    defer r1.deinit();
    const ids5 = ids(5, 100, 1);
    _ = try r1.s.prefill(&ids5);
    const links = try gp.memoryLinks();
    var p1: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 1, .link = links[1] };
    var done: gp.Followed = .{};
    var p0: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 0, .link = links[0] };
    var s0: gp.Sender = .{ .gpa = gpa, .pair = &p0 };
    const m = s0.mirror();
    try m.tell(.decode, &.{ 7, 9 });
    try m.tell(.draft, &.{ 999, 3, 7 }); // a position this rank is not at
    try testing.expectError(error.MirrorOutOfStep, gp.follow(Serve, &r1.s, &p1, gpa, &done, d));
    try m.tell(.decode, &.{ 7, 9 });
    links[0].close(); // rank 0 gone inside the decode: the read fails (a real pair's watchdog exits 3 first)
    try testing.expectError(error.PeerClosed, gp.follow(Serve, &r1.s, &p1, gpa, &done, d));
    links[1].close();
}
