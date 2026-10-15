//! rank_fingerprint: the two-Spark pair's bring-up recorded with TF_GLM_FINGERPRINT unset. The steps a rank takes (glm.ranks.bringUp
//! over a CPU rank that has the identity step, so the head's switch decides whether it runs) and every frame that rank
//! sends over the control link: a real Pair (the link, its barriers and the settings agreement, no NCCL or GPU) against
//! a scripted peer, each rank in turn. Written against the base's API only (b5804353): the same file runs on the base
//! and on rank_fingerprint, and with the switch unset the transcripts must be the same bytes.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("glm_ranks");
const link = cuda.link;
const net = std.Io.net;

const gpa = std.testing.allocator;
/// the lab's settings (qualification_cef1fec3's two-Spark deployment: 262,144 tokens of window and its slack, drafts on)
const lab_settings: ranks.Settings = .{ .capacity = 262_144 + 2_560, .long_context = true, .prefill_rows = 2048, .mtp = true, .run = "serve warm=96x16 drafts=3" };

extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// The switch as a rank's open reads it: on at "1" only.
fn switchOn() bool {
    const v = std.c.getenv("TF_GLM_FINGERPRINT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// One CPU rank in GlmEngine's order: the Pair's link for real; the load and the Engine recorded only.
const CpuRank = struct {
    pair: ranks.Pair,
    master: net.IpAddress,
    server: ?link.Server = null,
    steps: [12][]const u8 = undefined,
    n: usize = 0,

    fn add(r: *CpuRank, s: []const u8) void {
        r.steps[r.n] = s;
        r.n += 1;
    }

    pub fn comm(r: *CpuRank) !void {
        r.add("open_comm");
        if (r.pair.rank == 0) {
            r.pair.server = r.server;
            r.pair.link = try r.pair.server.?.accept(10_000);
        } else try r.pair.openLink(r.master, 10_000);
    }

    pub fn barrier(r: *CpuRank, label: []const u8) !void {
        r.add(if (std.mem.eql(u8, label, "comm")) "barrier" else "barrier (loaded)");
        try r.pair.barrier(label, 0);
    }

    pub fn settings(r: *CpuRank) !void {
        r.add("settings");
        try r.pair.agree(lab_settings);
    }

    pub fn fingerprint(r: *const CpuRank) bool {
        _ = r;
        return switchOn();
    }

    pub fn identity(r: *CpuRank) !void {
        r.add("identity");
    }

    pub fn load(r: *CpuRank) !void {
        r.add("load");
    }

    pub fn ready(r: *CpuRank, label: []const u8) !void {
        r.add(label);
        try r.pair.barrier(label, 0);
    }

    pub fn engine(r: *CpuRank) !void {
        r.add("Engine");
    }
};

/// The frames the real rank sent and the steps it took, one line each.
const Transcript = struct {
    buf: [4096]u8 = undefined,
    len: usize = 0,

    fn line(t: *Transcript, comptime fmt: []const u8, args: anytype) !void {
        const s = try std.fmt.bufPrint(t.buf[t.len..], fmt ++ "\n", args);
        t.len += s.len;
    }

    fn text(t: *const Transcript) []const u8 {
        return t.buf[0..t.len];
    }

    fn print(t: *const Transcript, side: []const u8) void {
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(t.text(), &d, .{});
        std.debug.print("rank_fingerprint-REC {s} sha256={s}\n{s}", .{ side, &std.fmt.bytesToHex(d, .lower), t.text() });
    }

    /// One frame from the real rank: its tag, length and bytes as hex.
    fn frame(t: *Transcript, f: link.Frame) !void {
        try t.line("rank->peer tag={d} len={d} bytes={x}", .{ @backingInt(f.tag), f.bytes.len, f.bytes });
    }
};

/// The labels and settings both ranks agree in bring-up order, as link.agree hashes them.
fn agreed(i: usize, buf: []u8) ![]const u8 {
    return switch (i) {
        0 => "comm",
        1 => try lab_settings.text(buf),
        2 => "loading",
        3 => "loaded",
        else => unreachable,
    };
}

/// The scripted peer of a real rank `real`: for each agreement in order it answers as link.agree does (rank 0 first),
/// with the digest the real rank must send; then the end (Pair.close: rank 0's stop, rank 1's ack).
fn scriptPeer(l: link.Link, real: u32, t: *Transcript) !void {
    for (0..4) |i| {
        var b: [512]u8 = undefined;
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(try agreed(i, &b), &d, .{});
        if (real == 1) try l.send(.settings, &d);
        const f = try l.recv(gpa);
        defer f.deinit(gpa);
        try t.frame(f);
        if (real == 0) try l.send(.settings, &d);
    }
    if (real == 0) {
        const f = try l.recv(gpa); // the stop
        defer f.deinit(gpa);
        try t.frame(f);
        try l.send(.ack, "");
    } else {
        try l.send(.stop, "");
        const f = try l.recv(gpa); // the ack
        defer f.deinit(gpa);
        try t.frame(f);
    }
}

fn record(real: u32, t: *Transcript) !void {
    const server = try link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    const master = try net.IpAddress.parse("127.0.0.1", server.port());
    var r: CpuRank = .{ .pair = .{ .gpa = gpa, .io = std.testing.io, .rank = real }, .master = master };
    const Peer = struct {
        fn run(rk: u32, at: net.IpAddress, tt: *Transcript, end: *?anyerror) void {
            const l = link.Link.join(std.testing.io, at, 10_000) catch |e| {
                end.* = e;
                return;
            };
            defer l.close();
            scriptPeer(l, rk, tt) catch |e| {
                end.* = e;
            };
        }
        fn serve(sv: link.Server, rk: u32, tt: *Transcript, end: *?anyerror) void {
            defer sv.close();
            const l = sv.accept(10_000) catch |e| {
                end.* = e;
                return;
            };
            defer l.close();
            scriptPeer(l, rk, tt) catch |e| {
                end.* = e;
            };
        }
    };
    var peer_end: ?anyerror = null;
    var frames: Transcript = .{};
    const th = if (real == 0) blk: {
        r.server = server;
        break :blk try std.Thread.spawn(.{}, Peer.run, .{ real, master, &frames, &peer_end });
    } else try std.Thread.spawn(.{}, Peer.serve, .{ server, real, &frames, &peer_end });
    const up = ranks.bringUp(&r);
    r.pair.close(); // the end both ranks make (rank 0's stop, rank 1's ack); closes rank 0's listener too
    th.join();
    try up;
    if (peer_end) |e| return e;
    try t.line("rank {d} steps:", .{real});
    for (r.steps[0..r.n]) |s| try t.line("  {s}", .{s});
    try t.line("{s}", .{std.mem.trimEnd(u8, frames.text(), "\n")});
}

test "rank_fingerprint-REC: TF_GLM_FINGERPRINT unset: the pair's bring-up steps and each rank's frames (the base's sequence and bytes)" {
    _ = unsetenv("TF_GLM_FINGERPRINT");
    var t: Transcript = .{};
    try record(0, &t);
    try record(1, &t);
    t.print("pair-off");
    const want_steps = "  open_comm\n  barrier\n  settings\n  load\n  loading\n  barrier (loaded)\n  Engine\n";
    try std.testing.expect(std.mem.indexOf(u8, t.text(), "rank 0 steps:\n" ++ want_steps) != null);
    try std.testing.expect(std.mem.indexOf(u8, t.text(), "rank 1 steps:\n" ++ want_steps) != null);
    try std.testing.expect(std.mem.indexOf(u8, t.text(), "identity") == null);
    try std.testing.expectEqual(@as(usize, 2 * (7 + 1 + 5)), std.mem.count(u8, t.text(), "\n")); // steps + header + 5 frames, a rank
}
