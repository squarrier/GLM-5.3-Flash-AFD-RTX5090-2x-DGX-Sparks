//! rank_fingerprint: TF_GLM_FINGERPRINT=1 on the two-Spark pair on the CPU, in cuda_ranks.zig's "two ranks over a local link" form:
//! the bring-up takes the identity step between `settings` and `load` only with the switch on; the settings text ends
//! ` fp=1`, so a rank without the switch disagrees at today's check (RanksDisagree on both, no identity sent); with it on
//! both ranks, the identities agree, or both ranks refuse (RanksDisagree) naming the field and both values: a different
//! config, a different ckpt, a different exe, different kernels, a set but different src (one test each); src unset on
//! a rank is accepted with one WARN naming it.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("glm_ranks");
const fp = cuda.fingerprint;
const net = std.Io.net;

const gpa = std.testing.allocator;

/// A 64-character stand-in for a digest: `s` (two characters) 32 times.
fn hex(comptime s: *const [2]u8) [64]u8 {
    var out: [64]u8 = undefined;
    for (0..32) |i| @memcpy(out[2 * i ..][0..2], s);
    return out;
}

const c1 = &hex("c1");
const c2 = &hex("c2");
const e1 = &hex("e1");
const e2 = &hex("e2");
const k7 = &hex("e7");
const k8 = &hex("e8");

/// Both GB10 ranks of the lab's pair: one role and arch, one checkpoint, one binary, one kernel set, one source.
const rank_id: fp.Identity = .{ .role = "tp-rank", .arch = "aarch64/sm_121", .config = c1, .ckpt = "mia-exl3-4bpw-tensorfold-078455ff", .exe = e1, .kernels = k7, .src = "b5804353" };

test "rank_fingerprint-PAIR: the bring-up takes the identity step between settings and load only with the switch on" {
    const Rec = struct {
        on: bool,
        steps: [9][]const u8 = undefined,
        n: usize = 0,
        fn add(r: *@This(), s: []const u8) void {
            r.steps[r.n] = s;
            r.n += 1;
        }
        pub fn comm(r: *@This()) !void {
            r.add("open_comm");
        }
        pub fn barrier(r: *@This(), label: []const u8) !void {
            r.add(if (std.mem.eql(u8, label, "comm")) "barrier" else "barrier (loaded)");
        }
        pub fn settings(r: *@This()) !void {
            r.add("settings");
        }
        pub fn fingerprint(r: *const @This()) bool {
            return r.on;
        }
        pub fn identity(r: *@This()) !void {
            r.add("identity");
        }
        pub fn load(r: *@This()) !void {
            r.add("load");
        }
        pub fn ready(r: *@This(), label: []const u8) !void {
            r.add(label);
        }
        pub fn engine(r: *@This()) !void {
            r.add("Engine");
        }
    };
    var on: Rec = .{ .on = true };
    try ranks.bringUp(&on);
    try std.testing.expectEqualDeep(@as([]const []const u8, &.{ "open_comm", "barrier", "settings", "identity", "load", "loading", "barrier (loaded)", "Engine" }), on.steps[0..on.n]);
    var off: Rec = .{ .on = false };
    try ranks.bringUp(&off);
    try std.testing.expectEqualDeep(@as([]const []const u8, &.{ "open_comm", "barrier", "settings", "load", "loading", "barrier (loaded)", "Engine" }), off.steps[0..off.n]);
}

test "rank_fingerprint-PAIR: the settings text ends fp=1 with the switch on (after run=), and is today's without it" {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    const today: ranks.Settings = .{ .capacity = 2560, .long_context = true, .prefill_rows = 2048, .mtp = true, .run = "first" };
    var on = today;
    on.fp = true;
    try std.testing.expectEqualStrings("drafter=0 capacity=2560 long_context=1 serial_only=0 latent=1 prefill_rows=2048 mtp=1 ring=0 run=first", try today.text(&a));
    try std.testing.expectEqualStrings("drafter=0 capacity=2560 long_context=1 serial_only=0 latent=1 prefill_rows=2048 mtp=1 ring=0 run=first fp=1", try on.text(&b));
    var small: [104]u8 = undefined; // room for today's text, not the suffix
    try std.testing.expectError(error.NoSpaceLeft, on.text(&small));
}

/// Two ranks over a local link: each agrees the settings (with the switch as given), then, when that agreed, its
/// identity line; how each ended and what each concluded.
const Two = struct {
    ends: [2]?anyerror = .{ null, null },
    verdicts: [2]?fp.Verdict = .{ null, null },
    lines: [2][512]u8 = undefined,

    fn settings(on: bool) ranks.Settings {
        return .{ .capacity = 262_144 + 2_560, .long_context = true, .prefill_rows = 2048, .mtp = true, .run = "serve warm=96x16 drafts=3", .fp = on };
    }

    fn rank(t: *Two, p: *ranks.Pair, on: bool, line: []const u8) void {
        const i = p.rank;
        p.agree(settings(on)) catch |e| {
            t.ends[i] = e;
            return;
        };
        p.agreeIdentity(line) catch |e| {
            t.ends[i] = e;
        };
        t.verdicts[i] = p.fp_verdict;
    }

    fn run(t: *Two, on: [2]bool, ids: [2]fp.Identity) !void {
        const l0 = try ids[0].line(&t.lines[0]);
        const l1 = try ids[1].line(&t.lines[1]);
        var p0: ranks.Pair = .{ .gpa = gpa, .io = std.testing.io, .rank = 0 };
        p0.server = try cuda.link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
        const port = p0.server.?.port();
        const Peer = struct {
            fn run(tt: *Two, at: u16, sw: bool, line: []const u8) void {
                var p1: ranks.Pair = .{ .gpa = gpa, .io = std.testing.io, .rank = 1 };
                p1.openLink(net.IpAddress.parse("127.0.0.1", at) catch unreachable, 5000) catch |e| {
                    tt.ends[1] = e;
                    return;
                };
                tt.rank(&p1, sw, line);
                p1.close(); // rank 0's stop, this rank's ack
            }
        };
        const th = try std.Thread.spawn(.{}, Peer.run, .{ t, port, on[1], l1 });
        if (p0.server.?.accept(5000)) |l| {
            p0.link = l;
            t.rank(&p0, on[0], l0);
        } else |e| t.ends[0] = e;
        p0.close();
        th.join();
    }

    fn print(t: *const Two, case: []const u8) void {
        for (0..2) |i| {
            const how: []const u8 = if (t.ends[i]) |e| @errorName(e) else "agreed";
            if (t.verdicts[i]) |v| {
                std.debug.print("rank_fingerprint-PAIR {s} rank-{d}: {s}; refused={} why=\"{s}\" warn=\"{s}\"\n", .{ case, i, how, v.refused, v.why(), v.warning() orelse "" });
            } else std.debug.print("rank_fingerprint-PAIR {s} rank-{d}: {s}; no identity judged\n", .{ case, i, how });
        }
    }

    fn refusedBoth(t: *const Two, field: fp.Field, a: []const u8, b: []const u8) !void {
        var wb: [64]u8 = undefined;
        const named = try std.fmt.bufPrint(&wb, fp.name ++ ": {t} differs: ", .{field});
        for (0..2) |i| {
            try std.testing.expectEqual(@as(?anyerror, error.RanksDisagree), t.ends[i]);
            const v = t.verdicts[i] orelse return error.NoVerdict;
            try std.testing.expect(v.refused and v.differs(field));
            try std.testing.expect(std.mem.indexOf(u8, v.why(), named) != null);
            try std.testing.expect(std.mem.indexOf(u8, v.why(), a) != null and std.mem.indexOf(u8, v.why(), b) != null);
        }
        try std.testing.expectEqualStrings(t.verdicts[0].?.why(), t.verdicts[1].?.why());
    }
};

test "rank_fingerprint-PAIR: the switch on, both ranks the lab's: agreed on both, no WARN" {
    var t: Two = .{};
    try t.run(.{ true, true }, .{ rank_id, rank_id });
    t.print("agree");
    for (0..2) |i| {
        try std.testing.expectEqual(@as(?anyerror, null), t.ends[i]);
        try std.testing.expect(t.verdicts[i] != null and !t.verdicts[i].?.refused and t.verdicts[i].?.warning() == null);
    }
}

test "rank_fingerprint-PAIR: a different config on rank 1: both ranks refuse (RanksDisagree), config named with both values" {
    var r1 = rank_id;
    r1.config = c2;
    var t: Two = .{};
    try t.run(.{ true, true }, .{ rank_id, r1 });
    t.print("config");
    try t.refusedBoth(.config, "rank-0 " ++ c1, "rank-1 " ++ c2);
}

test "rank_fingerprint-PAIR: a different ckpt on rank 1: both ranks refuse (RanksDisagree), ckpt named with both values" {
    var r1 = rank_id;
    r1.ckpt = "mia-exl3-4bpw-tensorfold-1234abcd";
    var t: Two = .{};
    try t.run(.{ true, true }, .{ rank_id, r1 });
    t.print("ckpt");
    try t.refusedBoth(.ckpt, "rank-0 mia-exl3-4bpw-tensorfold-078455ff", "rank-1 mia-exl3-4bpw-tensorfold-1234abcd");
}

test "rank_fingerprint-PAIR: a different exe on rank 0 (same role and arch): both ranks refuse (RanksDisagree), exe named with both values" {
    var r0 = rank_id;
    r0.exe = e2;
    var t: Two = .{};
    try t.run(.{ true, true }, .{ r0, rank_id });
    t.print("exe");
    try t.refusedBoth(.exe, "rank-0 " ++ e2, "rank-1 " ++ e1);
    try std.testing.expect(std.mem.indexOf(u8, t.verdicts[0].?.why(), "(both role tp-rank, arch aarch64/sm_121)") != null);
}

test "rank_fingerprint-PAIR: different kernels on rank 1 (same role and arch): both ranks refuse (RanksDisagree), kernels named with both values" {
    var r1 = rank_id;
    r1.kernels = k8;
    var t: Two = .{};
    try t.run(.{ true, true }, .{ rank_id, r1 });
    t.print("kernels");
    try t.refusedBoth(.kernels, "rank-0 " ++ k7, "rank-1 " ++ k8);
}

test "rank_fingerprint-PAIR: src set on both ranks but different: both ranks refuse (RanksDisagree), src named with both values" {
    var r1 = rank_id;
    r1.src = "deadbeef";
    var t: Two = .{};
    try t.run(.{ true, true }, .{ rank_id, r1 });
    t.print("src");
    try t.refusedBoth(.src, "rank-0 b5804353", "rank-1 deadbeef");
}

test "rank_fingerprint-PAIR: src unset on rank 1: agreed on both, one WARN naming rank-1, src not compared" {
    var r1 = rank_id;
    r1.src = fp.unset;
    var t: Two = .{};
    try t.run(.{ true, true }, .{ rank_id, r1 });
    t.print("src-unset");
    for (0..2) |i| {
        try std.testing.expectEqual(@as(?anyerror, null), t.ends[i]);
        try std.testing.expect(!t.verdicts[i].?.refused);
        try std.testing.expectEqualStrings(fp.name ++ ": src unset on rank-1 (built without -Dsource_id): src not compared", t.verdicts[i].?.warning().?);
    }
}

test "rank_fingerprint-PAIR: rank 1 without the switch: today's check refuses (RanksDisagree on both), no identity is sent" {
    var t: Two = .{};
    try t.run(.{ true, false }, .{ rank_id, rank_id });
    t.print("no-switch");
    for (0..2) |i| {
        try std.testing.expectEqual(@as(?anyerror, error.RanksDisagree), t.ends[i]);
        try std.testing.expect(t.verdicts[i] == null);
    }
}
