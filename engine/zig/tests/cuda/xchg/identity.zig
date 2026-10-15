//! rank_fingerprint: TF_GLM_FINGERPRINT=1 at the AFD hello on the CPU, in window.zig's form: three nodes in one process, real
//! control links on loopback, fake proxies between the regions. The switch on everywhere: the identities agree (the
//! attention's exe and kernels differ from the halves': another role and arch, accepted), or every node refuses
//! (error.SettingsDiffer) naming the field and both values: a different config, a different ckpt, the halves' exe, the
//! halves' kernels, a set but different src (one test each); src unset on some nodes is accepted with one WARN naming
//! them; a node without the switch disagrees at today's check (no identity is sent); an expert whose own line comes back
//! altered refuses (recorder.zig's scripted attention), and the on-wire frames of both sides are recorded.
const std = @import("std");
const cuda = @import("cuda");
const xchg = cuda.xchg;
const proto = xchg.proto;
const fp = cuda.fingerprint;
const fake = @import("fake.zig");
const rec = @import("recorder.zig");
const net = std.Io.net;

const gpa = std.testing.allocator;
const ms = std.time.ns_per_ms;

/// A 64-character stand-in for a digest: `s` (two characters) 32 times.
fn hex(comptime s: *const [2]u8) [64]u8 {
    var out: [64]u8 = undefined;
    for (0..32) |i| @memcpy(out[2 * i ..][0..2], s);
    return out;
}

const c1 = &hex("c1");
const c2 = &hex("c2");
const a1 = &hex("a1");
const a7 = &hex("a7");
const e1 = &hex("e1");
const e2 = &hex("e2");
const k7 = &hex("e7");
const k8 = &hex("e8");

/// The lab's three (evidence/qualification_df46b359): the attention on attention-host, both halves on GB10s, one checkpoint and one source.
const ckpt = "mia-exl3-4bpw-tensorfold-078455ff";
const attn_id: fp.Identity = .{ .role = "attention", .arch = "x86_64/sm_120", .config = c1, .ckpt = ckpt, .exe = a1, .kernels = a7, .src = "b5804353" };
const expert_id: fp.Identity = .{ .role = "expert", .arch = "aarch64/sm_121", .config = c1, .ckpt = ckpt, .exe = e1, .kernels = k7, .src = "b5804353" };
const lab = [3]?fp.Identity{ attn_id, expert_id, expert_id };

fn sleepMs(t: u64) void {
    const ts: std.c.timespec = .{ .sec = @intCast(t / 1000), .nsec = @intCast((t % 1000) * ms) };
    _ = std.c.nanosleep(&ts, null);
}

/// How an expert's start ended: joined (then stopped by the attention's stop frame), or its join's error.
const End = struct { joined: bool = false, stopped: bool = false, err: ?anyerror = null };

const Three = struct {
    regions: [3][]align(4096) u8 = undefined,
    status: [3][]u32 = undefined,
    nodes: [3]xchg.Node = undefined,
    openers: [3]fake.Opener = undefined,
    threads: [2]?std.Thread = .{ null, null },
    ends: [2]End = .{ .{}, .{} },
    attend: ?anyerror = null,
    lines: [3][512]u8 = undefined,
    texts: [3][256]u8 = undefined,
    text_lens: [3]usize = undefined,
    verdicts: [3]?fp.Verdict = .{ null, null, null },
    /// expert 1's wait before its join (ms): expert 0 is accepted first
    delay1_ms: u64 = 0,

    /// A node's config: the switch on with `line` (null: the switch unset, today's config).
    fn config(line: ?[]const u8) xchg.Config {
        return .{ .settings = .{ .layout = rec.layout, .max_rows = rec.max_rows, .ports = 1, .fp = line != null }, .beat_ns = 50 * ms, .setup_timeout_ms = 10_000, .warn_failures = true, .identity = line };
    }

    /// The three up and down: node i with `ids[i]` (null: without the switch); every result kept, then all freed.
    fn run(t: *Three, ids: [3]?fp.Identity) !void {
        var lines: [3]?[]const u8 = .{ null, null, null };
        for (ids, 0..) |id, i| if (id) |x| {
            lines[i] = try x.line(&t.lines[i]);
        };
        const total = config(lines[0]).settings.region().total();
        for (0..3) |i| {
            t.regions[i] = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(total));
            @memset(t.regions[i], 0);
            t.status[i] = try gpa.alloc(u32, proto.STATUS_WORDS);
            @memset(t.status[i], 0);
            t.openers[i] = .{ .gpa = gpa };
            const role: xchg.Role = if (i == 0) .attn else .{ .expert = @intCast(i - 1) };
            const mem: xchg.Memory = .{ .region = t.regions[i], .region_dev = @intFromPtr(t.regions[i].ptr), .status = t.status[i], .status_dev = @intFromPtr(t.status[i].ptr) };
            t.nodes[i] = try xchg.Node.init(gpa, std.testing.io, role, config(lines[i]), mem);
            t.text_lens[i] = (try t.nodes[i].cfg.settings.text(&t.texts[i])).len;
        }
        const server = try cuda.link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
        const master = try net.IpAddress.parse("127.0.0.1", server.port());
        for (0..2) |k| t.threads[k] = try std.Thread.spawn(.{}, expertMain, .{ t, k, master });
        if (t.nodes[0].attend(server, t.openers[0].opener())) |_| t.nodes[0].stopExperts() else |e| t.attend = e;
        t.nodes[0].deinit(); // the attention is down (or gave up): its links close, as when its process exits
        server.close();
        for (t.threads) |th| if (th) |x| x.join();
        for (&t.nodes, 0..) |*n, i| {
            n.deinit();
            t.verdicts[i] = n.fp_verdict;
        }
        for (t.regions) |r| gpa.free(r);
        for (t.status) |s| gpa.free(s);
    }

    fn expertMain(t: *Three, k: usize, master: net.IpAddress) void {
        if (k == 1 and t.delay1_ms > 0) sleepMs(t.delay1_ms);
        const n = &t.nodes[k + 1];
        n.join(master, t.openers[k + 1].opener()) catch |e| {
            t.ends[k].err = e;
            return;
        };
        t.ends[k].joined = true;
        t.ends[k].stopped = n.idle() == .stopped;
    }

    fn text(t: *const Three, i: usize) []const u8 {
        return t.texts[i][0..t.text_lens[i]];
    }

    fn label(i: usize) []const u8 {
        return ([3][]const u8{ "attention", "expert-0", "expert-1" })[i];
    }

    /// One line a node for the evidence: how its start ended and what it concluded.
    fn print(t: *const Three, case: []const u8) void {
        for (0..3) |i| {
            const how: []const u8 = if (i == 0) (if (t.attend) |e| @errorName(e) else "attended") else if (t.ends[i - 1].err) |e| @errorName(e) else if (t.ends[i - 1].stopped) "joined, stopped" else "joined";
            if (t.verdicts[i]) |v| {
                std.debug.print("rank_fingerprint-AFD {s} {s}: {s}; refused={} why=\"{s}\" warn=\"{s}\"\n", .{ case, label(i), how, v.refused, v.why(), v.warning() orelse "" });
            } else std.debug.print("rank_fingerprint-AFD {s} {s}: {s}; no identity judged\n", .{ case, label(i), how });
        }
    }

    /// Every node up and stopped cleanly; each judged the table alike, not refused.
    fn acceptedAll(t: *const Three) !void {
        try std.testing.expectEqual(@as(?anyerror, null), t.attend);
        for (t.ends) |e| try std.testing.expect(e.joined and e.stopped and e.err == null);
        for (t.verdicts) |v| try std.testing.expect(v != null and !v.?.refused);
        for (0..3) |i| try std.testing.expect(std.mem.endsWith(u8, t.text(i), " fp=1"));
    }

    /// Every node refused with SettingsDiffer, each naming `field` with both values, all three the same words.
    fn refusedAll(t: *const Three, field: fp.Field, a: []const u8, b: []const u8) !void {
        try std.testing.expectEqual(@as(?anyerror, error.SettingsDiffer), t.attend);
        for (t.ends) |e| try std.testing.expectEqual(@as(?anyerror, error.SettingsDiffer), e.err);
        var wb: [64]u8 = undefined;
        const named = try std.fmt.bufPrint(&wb, fp.name ++ ": {t} differs: ", .{field});
        for (t.verdicts) |v| {
            const x = v orelse return error.NoVerdict;
            try std.testing.expect(x.refused and x.differs(field));
            try std.testing.expect(std.mem.indexOf(u8, x.why(), named) != null);
            try std.testing.expect(std.mem.indexOf(u8, x.why(), a) != null and std.mem.indexOf(u8, x.why(), b) != null);
            try std.testing.expectEqualStrings(t.verdicts[0].?.why(), x.why());
        }
    }
};

test "rank_fingerprint-AFD: the switch on, the lab's three: accepted on every node; the attention's exe and kernels differ from the halves' (another role and arch) and no WARN" {
    var t: Three = .{};
    try t.run(lab);
    t.print("lab");
    try t.acceptedAll();
    for (t.verdicts) |v| {
        try std.testing.expect(!v.?.differs(.exe) and !v.?.differs(.kernels));
        try std.testing.expect(v.?.warning() == null);
    }
    try std.testing.expect(!std.mem.eql(u8, attn_id.exe, expert_id.exe) and !std.mem.eql(u8, attn_id.kernels, expert_id.kernels));
}

test "rank_fingerprint-AFD: a different config on expert 1: every node refuses (SettingsDiffer), config named with both values" {
    var e_1 = expert_id;
    e_1.config = c2;
    var t: Three = .{};
    try t.run(.{ attn_id, expert_id, e_1 });
    t.print("config");
    try t.refusedAll(.config, "attention " ++ c1, "expert-1 " ++ c2);
}

test "rank_fingerprint-AFD: a different ckpt on expert 0: every node refuses (SettingsDiffer), ckpt named with both values" {
    var e0 = expert_id;
    e0.ckpt = "mia-exl3-4bpw-tensorfold-1234abcd";
    var t: Three = .{};
    try t.run(.{ attn_id, e0, expert_id });
    t.print("ckpt");
    try t.refusedAll(.ckpt, "attention " ++ ckpt, "expert-0 mia-exl3-4bpw-tensorfold-1234abcd");
}

test "rank_fingerprint-AFD: the halves' exe differs (same role and arch): every node refuses (SettingsDiffer), exe named with both values" {
    var e_1 = expert_id;
    e_1.exe = e2;
    var t: Three = .{};
    try t.run(.{ attn_id, expert_id, e_1 });
    t.print("exe");
    try t.refusedAll(.exe, "expert-0 " ++ e1, "expert-1 " ++ e2);
    try std.testing.expect(std.mem.indexOf(u8, t.verdicts[0].?.why(), "(both role expert, arch aarch64/sm_121)") != null);
}

test "rank_fingerprint-AFD: the halves' kernels differ (same role and arch): every node refuses (SettingsDiffer), kernels named with both values" {
    var e0 = expert_id;
    e0.kernels = k8;
    var t: Three = .{};
    try t.run(.{ attn_id, e0, expert_id });
    t.print("kernels");
    try t.refusedAll(.kernels, "expert-0 " ++ k8, "expert-1 " ++ k7);
}

test "rank_fingerprint-AFD: src set on every node but different on expert 1: every node refuses (SettingsDiffer), src named with both values" {
    var e_1 = expert_id;
    e_1.src = "deadbeef";
    var t: Three = .{};
    try t.run(.{ attn_id, expert_id, e_1 });
    t.print("src");
    try t.refusedAll(.src, "attention b5804353", "expert-1 deadbeef");
}

test "rank_fingerprint-AFD: src unset on the attention and expert 1: accepted, one WARN naming them on every node, src not compared" {
    var a = attn_id;
    a.src = fp.unset;
    var e_1 = expert_id;
    e_1.src = fp.unset;
    var e0 = expert_id;
    e0.src = "deadbeef"; // would differ from any other set src: not compared while some are unset
    var t: Three = .{};
    try t.run(.{ a, e0, e_1 });
    t.print("src-unset");
    try t.acceptedAll();
    for (t.verdicts) |v| try std.testing.expectEqualStrings(fp.name ++ ": src unset on attention, expert-1 (built without -Dsource_id): src not compared", v.?.warning().?);
}

test "rank_fingerprint-AFD: expert 1 without the switch: today's check refuses it (SettingsDiffer on the attention and expert 1), no identity is judged anywhere" {
    var t: Three = .{ .delay1_ms = 300 }; // expert 0 is accepted first; then expert 1's hello lacks fp=1
    try t.run(.{ attn_id, expert_id, null });
    t.print("no-switch");
    try std.testing.expectEqual(@as(?anyerror, error.SettingsDiffer), t.attend);
    try std.testing.expectEqual(@as(?anyerror, error.SettingsDiffer), t.ends[1].err);
    try std.testing.expect(t.ends[0].err != null and !t.ends[0].joined); // the attention gave up before the table
    for (t.verdicts) |v| try std.testing.expect(v == null);
    try std.testing.expect(!std.mem.endsWith(u8, t.text(2), " fp=1"));
}

const Seen = struct {
    v: ?fp.Verdict = null,
    fn hook(ctx: *anyopaque, n: *xchg.Node) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        s.v = n.fp_verdict;
    }
};

test "rank_fingerprint-AFD: an expert whose own line comes back altered in the table refuses (SettingsDiffer), naming it; its frames: hello with fp=1, its line, nothing more" {
    var lb: [3][512]u8 = undefined;
    const mine = try expert_id.line(&lb[0]);
    var seen: Seen = .{};
    var altered = expert_id;
    altered.exe = e2;
    var t: rec.Transcript = .{};
    const cfg = Three.config(mine);
    const got = rec.expertSide(cfg, 1, .{ .attention = try attn_id.line(&lb[1]), .expert_other = mine, .self_override = try altered.line(&lb[2]), .hook = .{ .ctx = &seen, .f = Seen.hook } }, &t);
    t.print("expert-side-altered");
    try std.testing.expectError(error.SettingsDiffer, got);
    const v = seen.v orelse return error.NoVerdict;
    std.debug.print("rank_fingerprint-AFD altered expert-1: refused={} why=\"{s}\"\n", .{ v.refused, v.why() });
    try std.testing.expect(v.refused and std.mem.indexOf(u8, v.why(), "expert-1's line came back as") != null);
    try std.testing.expectEqualSlices(u32, &.{ rec.hello, rec.ident }, t.seq());
}

test "rank_fingerprint-AFD: the switch on, the wire: expert 0 sends hello (fp=1), its line, its blob, ready; the attention sends ack, ack, the table twice, its blob twice, ready twice, stop twice" {
    var lb: [3][512]u8 = undefined;
    const lines = [3][]const u8{ try attn_id.line(&lb[0]), try expert_id.line(&lb[1]), try expert_id.line(&lb[2]) };
    var seen: Seen = .{};
    var te: rec.Transcript = .{};
    var cfg = rec.configOff(true);
    cfg.settings.fp = true;
    cfg.identity = lines[1];
    try rec.expertSide(cfg, 0, .{ .attention = lines[0], .expert_other = lines[2], .hook = .{ .ctx = &seen, .f = Seen.hook } }, &te);
    te.print("expert-side-on");
    try std.testing.expectEqualSlices(u32, &.{ rec.hello, rec.ident, rec.blob, rec.ready }, te.seq());
    try std.testing.expect(seen.v != null and !seen.v.?.refused);
    var ta: rec.Transcript = .{};
    var acfg = rec.configOff(false);
    acfg.settings.fp = true;
    acfg.identity = lines[0];
    try rec.attentionSide(acfg, .{ lines[1], lines[2] }, &ta);
    ta.print("attention-side-on");
    try std.testing.expectEqualSlices(u32, &.{ rec.ack, rec.ack, rec.ident, rec.ident, rec.blob, rec.blob, rec.ready, rec.ready, rec.stop, rec.stop }, ta.seq());
    try std.testing.expect(std.mem.indexOf(u8, ta.text(), "text=\"attention role=attention arch=x86_64/sm_120 ") != null);
}
