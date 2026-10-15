//! Each rank's watchdog: the other rank is lost when the control link closes or errs, or a call passes its deadline; the process then exits at once with code 3.
//! Source: Mia's patch 0006, MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, zig/src/families/flashnext/cuda_mirror.zig (Watch, keepalive; Apache-2.0, Copyright 2026 MiaAI-Lab).
//! installStop is from her patch 0007, zig/src/families/flashnext/cuda_native.zig. Changed here: family-agnostic, no comm field (0006 never reads it), Linux-only keepalive, fail, isLoss, peerGone.
//! Keepalive (5 s idle, 3 probes 2 s apart) makes a dead link err within ~11 s; TF_TP_TIMEOUT_S (300 s by default) bounds every armed call.
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

/// The exit code of a rank whose peer is lost (the watchdog's _exit).
pub const lost_exit_code: u8 = 3;

pub const Watch = struct {
    io: std.Io,
    fd: posix.socket_t,
    rank: u32,
    /// a call's deadline at least, in seconds (TF_TP_TIMEOUT_S, 300 by default: a long prompt's chunks)
    base_s: u64 = 300,
    deadline_ms: std.atomic.Value(i64) = .init(0),
    done: std.atomic.Value(bool) = .init(false),
    /// rank 1 waits on the link with no stream open: the link closing then is rank 0's stop, not a loss
    idle: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// what a lost peer does: log and exit 3; the tests record the reason instead
    lost: *const fn (w: *Watch, why: []const u8) void = abortAndExit,

    pub fn start(w: *Watch) !void {
        keepalive(w.fd);
        if (std.c.getenv("TF_TP_TIMEOUT_S")) |v| w.base_s = std.fmt.parseInt(u64, std.mem.span(v), 10) catch w.base_s;
        w.thread = try std.Thread.spawn(.{}, run, .{w});
    }

    /// A clean stop is under way: the other rank may close the link now.
    pub fn quiet(w: *Watch) void {
        w.done.store(true, .release);
    }

    pub fn deinit(w: *Watch) void {
        w.done.store(true, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
    }

    fn now(w: *const Watch) i64 {
        return @intCast(@divTrunc(std.Io.Timestamp.now(w.io, .awake).toNanoseconds(), std.time.ns_per_ms));
    }

    /// A call starts that must end within `seconds`.
    pub fn arm(w: *Watch, seconds: u64) void {
        w.deadline_ms.store(w.now() + @as(i64, @intCast(seconds * 1000)), .release);
    }

    pub fn disarm(w: *Watch) void {
        w.deadline_ms.store(0, .release);
    }

    /// The caller saw the other rank go (a link read or write failed): the watchdog's own exit, as 0006's Agreement.fatal does.
    pub fn fail(w: *Watch, why: []const u8) void {
        w.lost(w, why);
    }

    /// A link error that means the other rank is gone, for `fail`.
    pub fn isLoss(e: anyerror) bool {
        return e == error.PeerClosed or e == error.LinkReadFailed or e == error.LinkWriteFailed;
    }

    /// True when the other rank closed the link or it erred, waiting up to `ms` for the close to show (another layer's error may come first).
    pub fn peerGone(w: *const Watch, ms: i32) bool {
        const gone: i16 = 0x2000 | posix.POLL.HUP | posix.POLL.ERR | posix.POLL.NVAL;
        var fds = [_]posix.pollfd{.{ .fd = w.fd, .events = 0x2000, .revents = 0 }};
        const n = posix.poll(&fds, ms) catch return false;
        return n > 0 and fds[0].revents & gone != 0;
    }

    fn run(w: *Watch) void {
        const rdhup: i16 = 0x2000; // POLLRDHUP: the peer closed its end
        const gone = rdhup | posix.POLL.HUP | posix.POLL.ERR | posix.POLL.NVAL;
        while (!w.done.load(.acquire)) {
            var fds = [_]posix.pollfd{.{ .fd = w.fd, .events = rdhup | posix.POLL.IN, .revents = 0 }};
            const n = posix.poll(&fds, 200) catch 0;
            // readable data alone is a frame for the loop: wait for it to be read, not for the peer
            if (n > 0 and fds[0].revents & gone == 0) std.Io.sleep(w.io, .fromMilliseconds(50), .awake) catch {};
            if (n > 0 and fds[0].revents & gone != 0) {
                // with bytes still unread (rank 0's stop) the loop gets 2 s to read them and quiet the watch
                var grace: u32 = 0;
                const unread = fds[0].revents & posix.POLL.IN != 0;
                while (unread and grace < 20 and !w.done.load(.acquire)) : (grace += 1) std.Io.sleep(w.io, .fromMilliseconds(100), .awake) catch {};
                if (w.done.load(.acquire)) return;
                if (w.idle.load(.acquire)) {
                    // the follow loop reads the close and returns; give it a moment, then leave cleanly anyway
                    var settle: u32 = 0;
                    while (settle < 20 and !w.done.load(.acquire)) : (settle += 1) std.Io.sleep(w.io, .fromMilliseconds(100), .awake) catch {};
                    if (w.done.load(.acquire)) return;
                    std.log.info("rank {d}: the other rank closed the link while no stream was open: stopping", .{w.rank});
                    std.c._exit(0);
                }
                return w.lost(w, "the other rank closed the control link (it exited, or the link is down)");
            }
            const d = w.deadline_ms.load(.acquire);
            if (d != 0 and w.now() > d) return w.lost(w, "a call passed its deadline (TF_TP_TIMEOUT_S): the other rank stalled");
        }
    }

    /// Log and leave at once; not ncclCommAbort first (it segfaulted with a collective in flight) and no atexit handler that could wait on the GPU.
    fn abortAndExit(w: *Watch, why: []const u8) void {
        std.log.err("rank {d}: {s}; exiting", .{ w.rank, why });
        std.c._exit(lost_exit_code);
    }
};

/// TCP keepalive on the control link: a link that goes down errs within ~11 s (5 s idle, 3 probes 2 s apart); Linux's option numbers.
pub fn keepalive(fd: posix.socket_t) void {
    if (builtin.os.tag != .linux) return;
    const on: c_int = 1;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, std.mem.asBytes(&on)) catch {};
    for ([_]struct { u32, c_int }{ .{ 4, 5 }, .{ 5, 2 }, .{ 6, 3 } }) |o| { // TCP_KEEPIDLE, KEEPINTVL, KEEPCNT
        posix.setsockopt(fd, posix.IPPROTO.TCP, o[0], std.mem.asBytes(&o[1])) catch {};
    }
}

/// A rank stops on SIGTERM or SIGINT, also as a container's PID 1 (where a default action is ignored): it exits 0 at once and the other rank's watchdog follows.
pub fn installStop() void {
    const stop: posix.Sigaction = .{ .handler = .{ .handler = onStop }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.TERM, &stop, null);
    posix.sigaction(.INT, &stop, null);
}

fn onStop(_: posix.SIG) callconv(.c) void {
    const line = "tensorfold: stop signal: this rank exits, the other follows\n";
    _ = std.c.write(2, line, line.len);
    std.c._exit(0);
}

const Lost = struct {
    var why: ?[]const u8 = null;
    fn record(_: *Watch, reason: []const u8) void {
        why = reason;
    }
};

test "the watchdog sees the other rank close the link, and a call past its deadline" {
    if (builtin.os.tag != .linux) return error.SkipZigTest; // POLLRDHUP is Linux's
    const link = @import("link.zig");
    const net = std.Io.net;
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const peer = try link.Link.join(io, try net.IpAddress.parse("127.0.0.1", server.port()), 5000);
    const l = try server.accept(5000);
    defer l.close();
    Lost.why = null;
    var w: Watch = .{ .io = io, .fd = l.fd, .rank = 0, .lost = Lost.record };
    try w.start();
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
    try std.testing.expect(Lost.why == null); // an open, quiet link is fine
    peer.close();
    var waited: u32 = 0;
    while (Lost.why == null and waited < 50) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    w.deinit();
    try std.testing.expect(std.mem.indexOf(u8, Lost.why orelse "", "closed") != null);

    const peer2 = try link.Link.join(io, try net.IpAddress.parse("127.0.0.1", server.port()), 5000);
    defer peer2.close();
    const l2 = try server.accept(5000);
    defer l2.close();
    Lost.why = null;
    var w2: Watch = .{ .io = io, .fd = l2.fd, .rank = 1, .lost = Lost.record, .base_s = 0 };
    try w2.start();
    w2.base_s = 0; // start() may read TF_TP_TIMEOUT_S
    w2.arm(0);
    waited = 0;
    while (Lost.why == null and waited < 50) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    w2.deinit();
    try std.testing.expect(std.mem.indexOf(u8, Lost.why orelse "", "deadline") != null);
}

test "a quiet watch lets the other rank close the link" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const link = @import("link.zig");
    const net = std.Io.net;
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const peer = try link.Link.join(io, try net.IpAddress.parse("127.0.0.1", server.port()), 5000);
    const l = try server.accept(5000);
    defer l.close();
    Lost.why = null;
    var w: Watch = .{ .io = io, .fd = l.fd, .rank = 0, .lost = Lost.record };
    try w.start();
    w.arm(60);
    w.quiet();
    peer.close();
    std.Io.sleep(io, .fromMilliseconds(500), .awake) catch {};
    w.deinit();
    try std.testing.expect(Lost.why == null);
}

test "peerGone: an open link is not gone, a closed one is; fail is the watchdog's exit" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const link = @import("link.zig");
    const net = std.Io.net;
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const peer = try link.Link.join(io, try net.IpAddress.parse("127.0.0.1", server.port()), 5000);
    const l = try server.accept(5000);
    defer l.close();
    Lost.why = null;
    var w: Watch = .{ .io = io, .fd = l.fd, .rank = 0, .lost = Lost.record };
    try std.testing.expect(!w.peerGone(50));
    try peer.send(.round, "unread");
    try std.testing.expect(!w.peerGone(50)); // a frame waiting is not a close
    peer.close();
    try std.testing.expect(w.peerGone(1000));
    try std.testing.expect(Watch.isLoss(error.PeerClosed) and !Watch.isLoss(error.NcclFailed));
    w.fail("test");
    try std.testing.expectEqualStrings("test", Lost.why.?);
}

test "keepalive: 5 s idle, probes 2 s apart, 3 of them (~11 s to a dead link)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const fd = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    try std.testing.expect(posix.errno(fd) == .SUCCESS);
    defer _ = posix.system.close(@intCast(fd));
    keepalive(@intCast(fd));
    const Opt = struct {
        fn get(s: posix.socket_t, level: i32, name: u32) !c_int {
            var v: c_int = -1;
            var len: posix.socklen_t = @sizeOf(c_int);
            if (posix.errno(posix.system.getsockopt(s, level, name, std.mem.asBytes(&v), &len)) != .SUCCESS) return error.GetFailed;
            return v;
        }
    };
    try std.testing.expect(try Opt.get(@intCast(fd), posix.SOL.SOCKET, posix.SO.KEEPALIVE) != 0);
    try std.testing.expectEqual(@as(c_int, 5), try Opt.get(@intCast(fd), posix.IPPROTO.TCP, 4));
    try std.testing.expectEqual(@as(c_int, 2), try Opt.get(@intCast(fd), posix.IPPROTO.TCP, 5));
    try std.testing.expectEqual(@as(c_int, 3), try Opt.get(@intCast(fd), posix.IPPROTO.TCP, 6));
}
