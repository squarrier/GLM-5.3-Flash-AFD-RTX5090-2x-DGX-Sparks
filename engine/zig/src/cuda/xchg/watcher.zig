//! The attention rank's failure rule (qualification_3df82c97) as a pure step over what its watcher sees: a dead or failed link at once, a stale heartbeat, a request whose reply has not landed in `silent_ns`.
//! qualification_qualification_a42e4dd8's 10 s host rule (ep_wire.zig `Service.run`) for the pingpong; the watcher thread (node.zig) applies the verdict to the pinned status, which every GPU wait reads.
const std = @import("std");
const proto = @import("proto.zig");

pub const Rules = struct {
    /// a posted request whose reply has not landed this long fails the exchange (a silent or hung expert)
    silent_ns: u64 = 10 * std.time.ns_per_s,
    /// an expert with no heartbeat this long fails it (a hung expert while the attention is idle)
    beat_stale_ns: u64 = 10 * std.time.ns_per_s,
};

/// One expert as one round sees it.
pub const Seen = struct {
    /// its reply to the last request the GPU rang has landed (every port's flag)
    landed: bool = true,
    /// its control link closed or erred: the process is gone
    link_dead: bool = false,
    /// its pair's proxy failed (a write completion error)
    proxy_failed: bool = false,
    /// when its last heartbeat arrived
    beat_ns: u64 = 0,

    pub fn markDead(s: *Seen) void {
        s.link_dead = true;
    }
};

pub const Verdict = struct { why: proto.Why, peer: u32, seq: u32 };

pub const Rule = struct {
    rules: Rules = .{},
    /// the last request the GPU rang, and when the watcher first saw it
    posted: u32 = 0,
    posted_ns: u64 = 0,

    /// One round at `now` with the doorbell's seq: a verdict to record, else null; a dead link, then a failed proxy, then stale beats, then silence.
    pub fn step(r: *Rule, now: u64, doorbell: u32, seen: [proto.EXPERTS]Seen) ?Verdict {
        if (doorbell != r.posted) {
            r.posted = doorbell;
            r.posted_ns = now;
        }
        for (seen, 0..) |s, k| if (s.link_dead) return .{ .why = .dead, .peer = @intCast(k), .seq = r.posted };
        for (seen, 0..) |s, k| if (s.proxy_failed) return .{ .why = .proxy, .peer = @intCast(k), .seq = r.posted };
        for (seen, 0..) |s, k| if (now -| s.beat_ns > r.rules.beat_stale_ns) return .{ .why = .beats, .peer = @intCast(k), .seq = r.posted };
        if (r.posted == 0) return null;
        for (seen, 0..) |s, k| if (!s.landed and now -| r.posted_ns > r.rules.silent_ns) return .{ .why = .silent, .peer = @intCast(k), .seq = r.posted };
        return null;
    }
};

const s_ns = std.time.ns_per_s;

test "idle with fresh beats never fails, whatever the time" {
    var r: Rule = .{};
    const fresh = [_]Seen{ .{ .beat_ns = 100 * s_ns }, .{ .beat_ns = 100 * s_ns } };
    try std.testing.expectEqual(@as(?Verdict, null), r.step(100 * s_ns, 0, fresh));
    try std.testing.expectEqual(@as(?Verdict, null), r.step(105 * s_ns, 0, fresh));
}

test "a reply missing 10 s after its post fails that expert at 10 s plus 1 ns, not before; a landed one never" {
    var r: Rule = .{};
    const t0 = 1000 * s_ns;
    var seen = [_]Seen{ .{ .landed = false, .beat_ns = t0 }, .{ .landed = true, .beat_ns = t0 } };
    try std.testing.expectEqual(@as(?Verdict, null), r.step(t0, 5, seen));
    seen[0].beat_ns = t0 + 10 * s_ns;
    seen[1].beat_ns = t0 + 10 * s_ns;
    try std.testing.expectEqual(@as(?Verdict, null), r.step(t0 + 10 * s_ns, 5, seen));
    try std.testing.expectEqual(Verdict{ .why = .silent, .peer = 0, .seq = 5 }, r.step(t0 + 10 * s_ns + 1, 5, seen).?);
    seen[0].landed = true;
    try std.testing.expectEqual(@as(?Verdict, null), r.step(t0 + 10 * s_ns + 2, 5, seen));
}

test "each new post restarts the clock: silence counts from the latest request" {
    var r: Rule = .{ .rules = .{ .silent_ns = 50, .beat_stale_ns = 1 << 60 } };
    const quiet = [_]Seen{ .{ .landed = false }, .{ .landed = false } };
    try std.testing.expectEqual(@as(?Verdict, null), r.step(0, 7, quiet));
    try std.testing.expectEqual(@as(?Verdict, null), r.step(40, 8, quiet));
    try std.testing.expectEqual(@as(?Verdict, null), r.step(90, 8, quiet));
    try std.testing.expectEqual(Verdict{ .why = .silent, .peer = 0, .seq = 8 }, r.step(91, 8, quiet).?);
}

test "a dead link fails at once and before silence; a failed proxy and stale beats name their expert" {
    var r: Rule = .{};
    var seen = [_]Seen{ .{ .landed = false, .beat_ns = 0 }, .{ .landed = false, .link_dead = true, .beat_ns = 0 } };
    try std.testing.expectEqual(Verdict{ .why = .dead, .peer = 1, .seq = 3 }, r.step(1, 3, seen).?);
    seen[1].link_dead = false;
    seen[0].proxy_failed = true;
    try std.testing.expectEqual(Verdict{ .why = .proxy, .peer = 0, .seq = 3 }, r.step(2, 3, seen).?);
    seen[0].proxy_failed = false;
    seen[0].landed = true;
    seen[1].landed = true;
    seen[1].beat_ns = 5 * s_ns;
    try std.testing.expectEqual(Verdict{ .why = .beats, .peer = 0, .seq = 3 }, r.step(10 * s_ns + 1, 3, seen).?);
    seen[0].beat_ns = 5 * s_ns;
    try std.testing.expectEqual(@as(?Verdict, null), r.step(15 * s_ns, 3, seen));
    try std.testing.expectEqual(Verdict{ .why = .beats, .peer = 0, .seq = 3 }, r.step(15 * s_ns + 1, 3, seen).?);
}
