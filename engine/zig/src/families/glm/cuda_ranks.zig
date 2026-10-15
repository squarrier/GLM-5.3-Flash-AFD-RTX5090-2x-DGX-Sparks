//! Two TP ranks in lockstep (qualification_qualification_44ba554e.5 B3), in python-0.6 GlmEngine's bring-up order (glm5_next/cuda/engine.py, ed78d6f):
//! the communicator (open_comm: here the control link, then NCCL and RoCE over it, cuda/comm.zig) and a barrier; the
//! settings both ranks must share (python's `mine`) agreed; the rank's weights; `ready("loading")` and a barrier; then
//! the Engine. Both ranks then make the same calls (the run's request is part of the agreed settings), each armed on
//! the watchdog (cuda/watch.zig): a lost peer, or a call past TF_TP_TIMEOUT_S, ends the process at once with exit 3.
const std = @import("std");
const cuda = @import("cuda");
const net = std.Io.net;
const Allocator = std.mem.Allocator;
const Watch = cuda.watch.Watch;

/// python's `mine` (what both ranks must share, in its order), then `run`: the verb and the request's digest.
pub const Settings = struct {
    drafter: bool = false,
    capacity: usize,
    long_context: bool,
    serial_only: bool = false,
    latent: bool = true,
    prefill_rows: usize,
    mtp: bool,
    ring: usize = 0,
    run: []const u8 = "",
    /// rank_fingerprint: TF_GLM_FINGERPRINT=1 on this rank: the text ends ` fp=1`, so a rank without it disagrees; off, today's text
    fp: bool = false,

    pub fn text(s: Settings, buf: []u8) ![]const u8 {
        const t = try std.fmt.bufPrint(buf, "drafter={d} capacity={d} long_context={d} serial_only={d} latent={d} prefill_rows={d} mtp={d} ring={d} run={s}", .{
            @intFromBool(s.drafter), s.capacity,     @intFromBool(s.long_context), @intFromBool(s.serial_only),
            @intFromBool(s.latent),  s.prefill_rows, @intFromBool(s.mtp),          s.ring,
            s.run,
        });
        if (!s.fp) return t;
        const fp = " fp=1";
        if (t.len + fp.len > buf.len) return error.NoSpaceLeft;
        @memcpy(buf[t.len..][0..fp.len], fp);
        return buf[0 .. t.len + fp.len];
    }
};

/// GlmEngine.__init__'s bring-up in python's order, each step `h`'s (a GPU rank's Pair and loader, or a recorder). rank_fingerprint:
/// with TF_GLM_FINGERPRINT=1 (`h.fingerprint()`, where `h` has the step) the ranks' identities are agreed before the load.
pub fn bringUp(h: anytype) !void {
    try h.comm();
    try h.barrier("comm");
    try h.settings();
    if (comptime hasIdentity(@TypeOf(h))) if (h.fingerprint()) try h.identity();
    try h.load();
    try h.ready("loading");
    try h.barrier("loaded");
    try h.engine();
}

/// rank_fingerprint: whether a bring-up's `h` has the identity step (the GPU rank, a recorder that records it).
fn hasIdentity(comptime H: type) bool {
    const T = switch (@typeInfo(H)) {
        .pointer => |p| p.child,
        else => H,
    };
    return @hasDecl(T, "identity");
}

/// How long python's comm.ready waits for a rank still loading before it names it (seconds).
pub const ready_s = 3600;

/// One rank's side of the pair: the control link, the communicator over it, and the watchdog.
pub const Pair = struct {
    gpa: Allocator,
    io: std.Io,
    rank: u32,
    server: ?cuda.link.Server = null,
    link: ?cuda.link.Link = null,
    comm: ?cuda.comm.Comm = null,
    watch: ?Watch = null,
    /// rank_fingerprint (TF_GLM_FINGERPRINT=1): what this rank concluded from both ranks' identity lines (null without the switch)
    fp_verdict: ?cuda.fingerprint.Verdict = null,

    /// The control link: rank 0 serves `master` and accepts rank 1, which joins (retrying while rank 0 starts).
    pub fn openLink(p: *Pair, master: net.IpAddress, timeout_ms: u32) !void {
        if (p.rank == 0) {
            p.server = try cuda.link.Server.open(master);
            p.link = try p.server.?.accept(@intCast(timeout_ms));
        } else p.link = try cuda.link.Link.join(p.io, master, timeout_ms);
    }

    /// open_comm's collectives: NCCL on every rail (its id over the link), RoCE where both ranks open it.
    pub fn openComm(p: *Pair) !void {
        p.comm = try cuda.comm.Comm.init(p.gpa, p.link.?, p.rank, 2, cuda.comm.Options.fromEnv());
    }

    /// The watchdog on the link (tf-tp2-gather's): from here a lost peer exits 3.
    pub fn startWatch(p: *Pair) !void {
        p.watch = .{ .io = p.io, .fd = p.link.?.fd, .rank = p.rank };
        try p.watch.?.start();
    }

    /// A call both ranks make, armed for `seconds` (the watch's TF_TP_TIMEOUT_S at least).
    pub fn arm(p: *Pair, seconds: u64) void {
        if (p.watch) |*w| w.arm(@max(seconds, w.base_s));
    }

    pub fn disarm(p: *Pair) void {
        if (p.watch) |*w| w.disarm();
    }

    /// A call's error: with the other rank gone, the watchdog's exit 3 (whichever rank saw it first); else `e`.
    pub fn lost(p: *Pair, e: anyerror) anyerror {
        if (p.watch) |*w| if (Watch.isLoss(e) or w.peerGone(1000)) w.fail("the other rank is gone (this rank's link or collective failed)");
        return e;
    }

    /// comm.barrier and comm.ready(label): both ranks reach `label` (a wrong label is error.RanksDisagree), armed.
    pub fn barrier(p: *Pair, label: []const u8, seconds: u64) !void {
        p.arm(seconds);
        defer p.disarm();
        p.link.?.agree(p.gpa, p.rank, label) catch |e| return p.lost(e);
    }

    /// Both ranks run with the same settings, or neither starts (error.RanksDisagree on both).
    pub fn agree(p: *Pair, s: Settings) !void {
        var buf: [512]u8 = undefined;
        const text = try s.text(&buf);
        p.link.?.agree(p.gpa, p.rank, text) catch |e| {
            if (e == error.RanksDisagree) std.log.warn("rank {d}: the ranks' settings differ; this rank: {s}", .{ p.rank, text });
            return p.lost(e);
        };
    }

    /// rank_fingerprint (TF_GLM_FINGERPRINT=1, after `agree`): this rank's identity line swapped for the other's (rank 0 first, as
    /// link.agree), then the same judgement on both ranks (fingerprint.judge over rank-0 and rank-1): a mismatch is
    /// error.RanksDisagree on both, its words naming the field and both values.
    pub fn agreeIdentity(p: *Pair, mine: []const u8) !void {
        const l = p.link.?;
        const fp = cuda.fingerprint;
        if (p.rank == 0) l.send(fp.tag, mine) catch |e| return p.lost(e);
        const theirs = l.expect(p.gpa, fp.tag) catch |e| return p.lost(e);
        defer p.gpa.free(theirs);
        if (p.rank != 0) l.send(fp.tag, mine) catch |e| return p.lost(e);
        var tb: [fp.table_max]u8 = undefined;
        const table = try fp.tableOf(&tb, &.{ .{ "rank-0", if (p.rank == 0) mine else theirs }, .{ "rank-1", if (p.rank == 0) theirs else mine } });
        p.fp_verdict = fp.verdictOf(table, null);
        const v = &p.fp_verdict.?;
        if (v.warning()) |w| std.log.warn("rank {d}: {s}", .{ p.rank, w });
        if (!v.refused) return;
        std.log.warn("rank {d}: {s}", .{ p.rank, v.why() });
        return p.lost(error.RanksDisagree);
    }

    /// The end (tf-tp2-gather's): both watches quiet, rank 0 says stop and rank 1 acknowledges, then each closes.
    pub fn close(p: *Pair) void {
        if (p.watch) |*w| w.quiet();
        if (p.link) |l| {
            if (p.rank == 0) {
                if (l.send(.stop, "")) |_| {
                    if (l.expect(p.gpa, .ack)) |b| p.gpa.free(b) else |_| {}
                } else |_| {}
            } else if (l.expect(p.gpa, .stop)) |b| {
                p.gpa.free(b);
                l.send(.ack, "") catch {};
            } else |_| {}
        }
        if (p.watch) |*w| w.deinit();
        if (p.comm) |*c| c.deinit();
        if (p.link) |l| l.close();
        if (p.server) |s| s.close();
        p.* = .{ .gpa = p.gpa, .io = p.io, .rank = p.rank };
    }
};

test "GlmEngine's bring-up order" {
    const Rec = struct {
        steps: [8][]const u8 = undefined,
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
    var r: Rec = .{};
    try bringUp(&r);
    try std.testing.expectEqualDeep(@as([]const []const u8, &.{ "open_comm", "barrier", "settings", "load", "loading", "barrier (loaded)", "Engine" }), r.steps[0..r.n]);
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("drafter=0 capacity=2560 long_context=1 serial_only=0 latent=1 prefill_rows=2048 mtp=1 ring=0 run=first", try (Settings{ .capacity = 2560, .long_context = true, .prefill_rows = 2048, .mtp = true, .run = "first" }).text(&buf));
}

const Lost = struct {
    var why: ?[]const u8 = null;
    fn record(_: *Watch, reason: []const u8) void {
        why = reason;
    }
};

test "two ranks over a local link: barriers and settings in step, different settings refused, a lost peer is the watchdog's" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var p0: Pair = .{ .gpa = gpa, .io = io, .rank = 0 };
    p0.server = try cuda.link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    const port = p0.server.?.port();
    const Peer = struct {
        fn run(io2: std.Io, at: u16, ok: *bool) void {
            var p1: Pair = .{ .gpa = std.testing.allocator, .io = io2, .rank = 1 };
            p1.openLink(net.IpAddress.parse("127.0.0.1", at) catch return, 5000) catch return;
            p1.barrier("comm", 0) catch return;
            p1.agree(.{ .capacity = 2560, .long_context = true, .prefill_rows = 2048, .mtp = true, .run = "first" }) catch return;
            const refused = if (p1.agree(.{ .capacity = 4096, .long_context = true, .prefill_rows = 2048, .mtp = true })) |_| false else |e| e == error.RanksDisagree;
            p1.link.?.close(); // rank 1 dies mid-run
            p1.link = null;
            ok.* = refused;
        }
    };
    var ok = false;
    const t = try std.Thread.spawn(.{}, Peer.run, .{ io, port, &ok });
    p0.link = try p0.server.?.accept(5000);
    try p0.barrier("comm", 0);
    try p0.agree(.{ .capacity = 2560, .long_context = true, .prefill_rows = 2048, .mtp = true, .run = "first" });
    try std.testing.expectError(error.RanksDisagree, p0.agree(.{ .capacity = 2560, .long_context = true, .prefill_rows = 2048, .mtp = true }));
    t.join();
    try std.testing.expect(ok);
    p0.watch = .{ .io = io, .fd = p0.link.?.fd, .rank = 0, .lost = Lost.record };
    if (p0.barrier("loading", ready_s)) |_| return error.NoLoss else |e| try std.testing.expect(Watch.isLoss(e));
    try std.testing.expect(Lost.why != null);
    p0.watch = null;
    p0.close();
}
