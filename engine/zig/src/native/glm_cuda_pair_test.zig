//! qualification_qualification_44ba554e.8 day 2's CPU gates for the CUDA GLM's pair (glm_cuda_pair.zig, glm_cuda_lanes.zig): two recording TP ranks of
//! the prompt fixture's model (rank 0 and rank 1 of world 2) over an in-memory link (a UNIX socket pair). Rank 0 serves
//! through the lane core with its mirror on the link, rank 1 follows in a thread. A served request, a cancel and a
//! second request give the same launches on both ranks, each rank's the launches a lone rank of its own makes for the
//! same calls; the warm-up runs on both ranks, recorded; rank 0's stop ends rank 1's follower with its ack.
const std = @import("std");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const gl = @import("glm_cuda_lanes.zig");
const gp = @import("glm_cuda_pair.zig");
const rec = glm.serve_rec;
const Serve = glm.serve.Serve;
const Pair = glm.ranks.Pair;

const testing = std.testing;
const gpa = testing.allocator;

/// A recording rank's serve calls: a Serve on the rank's own recording Fwd.
const Side = struct {
    r: *rec.Rank,
    f: glm.fwd.Fwd,
    s: Serve,

    fn init(rank: usize) !*Side {
        const x = try gpa.create(Side);
        errdefer gpa.destroy(x);
        x.r = try rec.Rank.init(gpa, rank, null);
        x.f = x.r.recording();
        x.s = .{ .e = x.r.e, .f = &x.f };
        return x;
    }

    fn deinit(x: *Side) void {
        x.r.deinit(gpa);
        gpa.destroy(x);
    }

    fn events(x: *Side) ![]const glm.trace.Event {
        try x.f.flush();
        return x.r.t.events.items;
    }
};

fn followThread(side: *Side, pair: *Pair, done: *gp.Followed, failed: *?anyerror) void {
    gp.follow(Serve, &side.s, pair, gpa, done, .{}) catch |e| {
        failed.* = e;
        pair.link.?.close(); // rank 0's stop then fails instead of waiting for an ack
        pair.link = null;
    };
}

/// A request through the lane core over rank 0's backend: its emitted tokens, the rounds it ran, and how it ended;
/// `cancel` set: the stream is cancelled after that many rounds (the core releases it between rounds).
fn request(b: *gl.Lanes(gp.Armed(Serve)), prompt: []const u32, max_new: u32, cancel: ?usize) !struct { emitted: usize, rounds: usize, reason: lanes.stream.Reason } {
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = max_new, .eos = &.{154_820}, .drafts = false });
    defer s.deinit(gpa);
    try core.addStream(&s);
    var rounds: usize = 0;
    while (core.activeCount() > 0) : (rounds += 1) {
        if (cancel) |n| if (rounds == n) {
            core.discard(&s);
            break;
        };
        try core.step();
    }
    return .{ .emitted = s.emitted().len, .rounds = rounds, .reason = s.reason };
}

fn ids(comptime n: usize, base: u32, stride: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*p, i| p.* = base + stride * @as(u32, @intCast(i));
    return out;
}

test "two recording ranks over an in-memory link: a served request, a cancel and a second request make the same launches on both ranks, each a lone rank's" {
    const r0 = try Side.init(0);
    defer r0.deinit();
    const r1 = try Side.init(1);
    defer r1.deinit();
    const links = try gp.memoryLinks();
    var p0: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 0, .link = links[0] };
    var p1: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 1, .link = links[1] };
    var done: gp.Followed = .{};
    var failed: ?anyerror = null;
    const t = try std.Thread.spawn(.{}, followThread, .{ r1, &p1, &done, &failed });
    var sender: gp.Sender = .{ .gpa = gpa, .pair = &p0 };
    var armed: gp.Armed(Serve) = .{ .steps = &r0.s, .pair = &p0 };
    var b: gl.Lanes(gp.Armed(Serve)) = .{ .steps = &armed, .mirror = sender.mirror(), .capacity = r0.r.e.st.capacity, .reach = r0.r.e.st.capacity };
    const first = ids(37, 1000, 7);
    const cancelled = ids(20, 3000, 11);
    const second = ids(9, 5000, 3);
    // the scenario, then rank 0's stop (rank 1 acks and its follower returns), before any check can leave the test
    const a = request(&b, &first, 5, null); // served to its length
    const c = request(&b, &cancelled, 40, 3); // cancelled after 3 rounds: the core releases it between rounds
    const d = request(&b, &second, 3, null);
    gp.stop(&p0);
    t.join();
    gp.closeFollower(&p1);
    try testing.expectEqual(@as(?anyerror, null), failed);
    const ra = try a;
    const rc = try c;
    const rd = try d;
    try testing.expectEqual(@as(usize, 5), ra.emitted);
    try testing.expectEqual(lanes.stream.Reason.length, ra.reason);
    try testing.expectEqual(lanes.stream.Reason.cancelled, rc.reason);
    try testing.expect(rc.emitted < 40);
    try testing.expectEqual(@as(usize, 3), rd.emitted);
    const steps = [_]usize{ ra.emitted - 1, rc.emitted - 1, rd.emitted - 1 };
    // rank 1 heard a begin, its steps and a release a request, in order
    try testing.expectEqual(@as(u64, 3), done.begins);
    try testing.expectEqual(@as(u64, steps[0] + steps[1] + steps[2]), done.steps);
    try testing.expectEqual(@as(u64, 3), done.releases);
    try testing.expectEqual(done.rounds, sender.sent);
    // both ranks made the same launches, in lockstep (only rank-own scalars may differ)
    const e0 = try r0.events();
    const e1 = try r1.events();
    try testing.expect(e0.len > 0);
    const l = try rec.expectLockstep(&r0.r.space, e0, &r1.r.space, e1);
    std.debug.print("two ranks: {d} launches each, in lockstep; {d} scalar values differ (the ranks' own); steps {any}; in:", .{ e0.len, l.differ, steps });
    for (l.names[0..l.n]) |x| std.debug.print(" {s}", .{x});
    std.debug.print("\n", .{});
    // the ranks differ only in their own vocab offset: the sampler's `ids + vocab_offset` (torch.add), once a forward
    try testing.expectEqual(3 + steps[0] + steps[1] + steps[2], l.differ);
    try testing.expectEqual(@as(usize, 1), l.n);
    try testing.expectEqualStrings("torch.add", l.names[0]);
    // each rank's launches are a lone rank's for the same calls: prefill, steps fed the token drawn (0 when recorded)
    for ([_]*Side{ r0, r1 }, [_]usize{ 0, 1 }) |side, rank| {
        const lone = try Side.init(rank);
        defer lone.deinit();
        for ([_][]const u32{ &first, &cancelled, &second }, steps) |p, n| {
            var tok = try lone.s.prefill(p);
            for (0..n) |_| tok = try lone.s.step(tok);
            lone.s.release();
        }
        try rec.expectSame(&lone.r.space, try lone.events(), &side.r.space, try side.events());
    }
}

test "the warm-up runs on both ranks, recorded: 96 prompt tokens and 16 greedy tokens, decode.prefill's and serial_decode's launches" {
    const r0 = try Side.init(0);
    defer r0.deinit();
    const r1 = try Side.init(1);
    defer r1.deinit();
    var w0: [gl.warm_tokens]u32 = undefined;
    var w1: [gl.warm_tokens]u32 = undefined;
    try gl.warmUp(Serve, &r0.s, &w0);
    try gl.warmUp(Serve, &r1.s, &w1);
    try testing.expectEqualSlices(u32, &w0, &w1);
    try testing.expectEqual(@as(usize, gl.warm_prompt + gl.warm_tokens - 1), r0.r.e.st.pos);
    try testing.expect(!r0.s.open);
    const e0 = try r0.events();
    const lw = try rec.expectLockstep(&r0.r.space, e0, &r1.r.space, try r1.events());
    try testing.expectEqual(@as(usize, gl.warm_tokens), lw.differ); // the vocab offset, once a forward
    try testing.expectEqualStrings("torch.add", lw.names[0]);
    // the tool's way on a fresh rank: decode.prefill of the same 96 ids, then serialDecode's 16 tokens
    const tool = try Side.init(0);
    defer tool.deinit();
    const p = ids(gl.warm_prompt, 1000, 1);
    const first = try tool.r.e.prefill(&tool.f, &p, false);
    var out: [gl.warm_tokens]u32 = undefined;
    const toks = try tool.r.e.serialDecode(&tool.f, first, gl.warm_tokens, &out, false);
    try testing.expectEqualSlices(u32, toks, &w0);
    try rec.expectSame(&tool.r.space, try tool.events(), &r0.r.space, e0);
    std.debug.print("warm-up: {d} launches a rank, recorded, equal to the tool's prefill + serial_decode\n", .{e0.len});
}

test "a round frame carries its op and words; a malformed one is refused" {
    const bytes = try gp.encode(gpa, .begin, &.{ 7, 8, 154_819 });
    defer gpa.free(bytes);
    try testing.expectEqual(@as(usize, 16), bytes.len);
    const r = try gp.decode(gpa, bytes);
    defer gpa.free(r.words);
    try testing.expectEqual(gl.Op.begin, r.op);
    try testing.expectEqualSlices(u32, &.{ 7, 8, 154_819 }, r.words);
    try testing.expectError(error.MirrorFrame, gp.decode(gpa, bytes[0..6]));
    try testing.expectError(error.MirrorFrame, gp.decode(gpa, &.{ 9, 0, 0, 0 }));
}

test "rank 1's follower stops on an unexpected frame, and a lost rank 0 is an error, never a hang" {
    const r1 = try Side.init(1);
    defer r1.deinit();
    const links = try gp.memoryLinks();
    var p1: Pair = .{ .gpa = gpa, .io = testing.io, .rank = 1, .link = links[1] };
    var done: gp.Followed = .{};
    try links[0].send(.settings, "late");
    try testing.expectError(error.UnexpectedFrame, gp.follow(Serve, &r1.s, &p1, gpa, &done, .{}));
    links[0].close(); // rank 0 gone with no stop: the read fails (a real pair's watchdog exits 3 first)
    try testing.expectError(error.PeerClosed, gp.follow(Serve, &r1.s, &p1, gpa, &done, .{}));
    links[1].close();
}
