//! tf-tp2-gather link: the control link and the watchdog without a GPU, round trips under an armed deadline, so the failure path runs on any host.
//! --sigterm-at T raises SIGTERM on this rank (it exits 0); --stall-at T stops answering with the link open; the other rank must exit 3 either way.
const std = @import("std");
const cuda = @import("cuda");
const link_mod = cuda.link;
const Watch = cuda.watch.Watch;

pub const Options = struct {
    rank: u32,
    master: std.Io.net.IpAddress,
    hold_s: f64 = 10,
    sigterm_at: ?f64 = null,
    stall_at: ?f64 = null,
    link_timeout_ms: u32 = 600_000,
};

fn seconds(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9;
}

/// --sigterm-at and --stall-at, checked before each trip.
fn inject(io: std.Io, o: Options, t0: i96, trips: u64) void {
    if (o.sigterm_at) |t| if (seconds(io, t0) >= t) {
        std.debug.print("INFO rank {d}: SIGTERM to itself after {d} trips\n", .{ o.rank, trips });
        _ = std.c.raise(.TERM);
    };
    if (o.stall_at) |t| if (seconds(io, t0) >= t) {
        std.debug.print("INFO rank {d}: stalling with the link open after {d} trips\n", .{ o.rank, trips });
        while (true) std.Io.sleep(io, .fromMilliseconds(1000), .awake) catch {};
    };
}

/// Rank 0 sends a frame every 10 ms for `hold_s`, rank 1 answers each until the stop; every wait is armed with the deadline (TF_TP_TIMEOUT_S).
pub fn run(gpa: std.mem.Allocator, io: std.Io, o: Options) !void {
    cuda.watch.installStop();
    var server: ?link_mod.Server = null;
    defer if (server) |s| s.close();
    const link = if (o.rank == 0) blk: {
        server = try link_mod.Server.open(o.master);
        break :blk try server.?.accept(@intCast(o.link_timeout_ms));
    } else try link_mod.Link.join(io, o.master, o.link_timeout_ms);
    defer link.close();
    try link.agree(gpa, o.rank, "tf-tp2-gather link v1");
    var w: Watch = .{ .io = io, .fd = link.fd, .rank = o.rank };
    try w.start();
    // on an error return too: the watch stops before the link closes, so the close is not taken for a loss
    defer w.deinit();
    std.debug.print("INFO rank {d}: link up, watchdog on (deadline {d} s, keepalive ~11 s)\n", .{ o.rank, w.base_s });
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var trips: u64 = 0;
    var word: [8]u8 = undefined;
    if (o.rank == 0) {
        while (seconds(io, t0) < o.hold_s) : (trips += 1) {
            inject(io, o, t0, trips);
            std.mem.writeInt(u64, &word, trips, .little);
            w.arm(w.base_s);
            link.send(.round, &word) catch |e| return lost(&w, e);
            gpa.free(link.expect(gpa, .ack) catch |e| return lost(&w, e));
            w.disarm();
            std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
        }
        // a clean end: quiet first, so rank 1 closing after its last answer is no loss
        w.quiet();
        try link.send(.stop, "");
        gpa.free(try link.expect(gpa, .ack));
    } else {
        while (true) : (trips += 1) {
            inject(io, o, t0, trips);
            w.arm(w.base_s);
            const f = link.recv(gpa) catch |e| return lost(&w, e);
            defer f.deinit(gpa);
            w.disarm();
            if (f.tag == .stop) {
                w.quiet();
                try link.send(.ack, "");
                break;
            }
            if (f.tag != .round) return error.UnexpectedFrame;
            link.send(.ack, f.bytes) catch |e| return lost(&w, e);
        }
    }
    w.deinit();
    std.debug.print("PASS rank {d}: {d} round trips, clean stop\n", .{ o.rank, trips });
}

/// A link error on the main thread: the other rank is gone, so the watchdog's exit (3), else the error itself.
fn lost(w: *Watch, e: anyerror) anyerror {
    if (Watch.isLoss(e)) w.fail("the other rank closed the control link (it exited, or the link is down)");
    return e;
}
