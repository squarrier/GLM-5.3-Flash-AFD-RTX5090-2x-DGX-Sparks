//! qualification_1fee6d78: the handshake's window (Config.setup_timeout_ms, which TF_AFD_SETUP_MS sets per process) on the CPU, in
//! protocol.zig's form: three nodes in one process, real control links on loopback, fake proxies between the regions.
//! An expert that joins after a short window has passed fails the start as today (PeerTimeout); a longer window set
//! through Config takes the same late join, with the three settings texts unchanged; and with no switch, experts that
//! start before the attention listens retry their connect within their own window (Link.join), the other route.
const std = @import("std");
const cuda = @import("cuda");
const xchg = cuda.xchg;
const proto = xchg.proto;
const fake = @import("fake.zig");
const net = std.Io.net;
const posix = std.posix;
const Threaded = std.Io.Threaded;

const gpa = std.testing.allocator;
const layout: proto.Layout = .{ .d = 64, .s = 9, .bf16 = true };
const max_rows = 32;
const ms = std.time.ns_per_ms;
/// the short window and the late expert's delay (ms): the delay is well past the window
const short_ms: u32 = 1_000;
const late_ms: u64 = 3_000;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Sleeps until `t` ms after `from` (a signal cannot shorten it).
fn sleepUntil(from: u64, t: u64) void {
    while (nowNs() < from + t * ms) {
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 1_000_000 };
        _ = std.c.nanosleep(&ts, null);
    }
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
    /// each expert's wait before its join starts (ms)
    delays: [2]u64 = .{ 0, 0 },
    t0: u64 = 0,

    /// A node's config with the window `window_ms` (null: Config's own, today's 120,000 ms).
    fn config(window_ms: ?u32) xchg.Config {
        return xchg.setup.with(.{ .settings = .{ .layout = layout, .max_rows = max_rows, .ports = 1 }, .beat_ns = 50 * ms, .warn_failures = true }, window_ms);
    }

    /// The attention with `attn`, both experts with `expert`, each over its own region.
    fn init(t: *Three, attn: xchg.Config, expert: xchg.Config) !void {
        const total = (proto.Region{ .slot = attn.settings.slotBytes() }).total();
        for (0..3) |i| {
            t.regions[i] = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(total));
            @memset(t.regions[i], 0);
            t.status[i] = try gpa.alloc(u32, proto.STATUS_WORDS);
            @memset(t.status[i], 0);
            t.openers[i] = .{ .gpa = gpa };
            const role: xchg.Role = if (i == 0) .attn else .{ .expert = @intCast(i - 1) };
            const mem: xchg.Memory = .{ .region = t.regions[i], .region_dev = @intFromPtr(t.regions[i].ptr), .status = t.status[i], .status_dev = @intFromPtr(t.status[i].ptr) };
            t.nodes[i] = try xchg.Node.init(gpa, std.testing.io, role, if (i == 0) attn else expert, mem);
        }
    }

    /// Both experts' threads from now (t0): each waits its delay, joins `master`, then idles until the stop frame.
    fn spawn(t: *Three, master: net.IpAddress) !void {
        t.t0 = nowNs();
        for (0..2) |k| t.threads[k] = try std.Thread.spawn(.{}, expertMain, .{ t, k, master });
    }

    fn expertMain(t: *Three, k: usize, master: net.IpAddress) void {
        sleepUntil(t.t0, t.delays[k]);
        const n = &t.nodes[k + 1];
        n.join(master, t.openers[k + 1].opener()) catch |e| {
            t.ends[k].err = e;
            return;
        };
        t.ends[k].joined = true;
        t.ends[k].stopped = n.idle() == .stopped;
    }

    /// After the attention is down and its listener closed: the experts' threads joined, every node down, freed.
    fn deinit(t: *Three) void {
        for (t.threads) |th| if (th) |x| x.join();
        for (&t.nodes) |*n| n.deinit();
        for (t.regions) |r| gpa.free(r);
        for (t.status) |s| gpa.free(s);
    }

    /// Each node's settings text equals `want` (the window is not in it).
    fn textsAre(t: *Three, want: []const u8) !void {
        var b: [256]u8 = undefined;
        for (&t.nodes) |*n| try std.testing.expectEqualStrings(want, try n.cfg.settings.text(&b));
    }
};

fn loopback(port: u16) !net.IpAddress {
    return net.IpAddress.parse("127.0.0.1", port);
}

/// A TCP socket bound to a free loopback port and not listening: a connect to it is refused (the port stays held).
fn boundNotListening() !cuda.link.Server {
    const address = try loopback(0);
    var storage: Threaded.PosixAddress = undefined;
    const len = Threaded.addressToPosix(&address, &storage);
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM | posix.SOCK.CLOEXEC, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    errdefer _ = posix.system.close(fd);
    if (posix.errno(posix.system.bind(fd, &storage.any, len)) != .SUCCESS) return error.BindFailed;
    return .{ .fd = fd };
}

test "qualification_1fee6d78: an expert that joins after a short window has passed fails the start as today (the attention's PeerTimeout); the late expert never gets in" {
    var t: Three = .{ .delays = .{ 0, late_ms } };
    try t.init(Three.config(short_ms), Three.config(short_ms));
    const server = try cuda.link.Server.open(try loopback(0));
    try t.spawn(try loopback(server.port()));
    const got = t.nodes[0].attend(server, t.openers[0].opener());
    const took = nowNs() - t.t0;
    t.nodes[0].deinit(); // the attention gives up: its link to expert 0 closes, as when its process exits
    sleepUntil(t.t0, late_ms + 1_000); // the late expert has connected into the backlog by now
    server.close(); // then the listener: the late expert's pending connection is reset
    t.deinit();
    std.debug.print("qualification_1fee6d78 WINDOW short {d} ms, expert 1 at {d} ms: attend {s} after {d} ms; expert 0 joined {}, {s}; expert 1 joined {}, {s}\n", .{ short_ms, late_ms, if (got) |_| "ok" else |e| @errorName(e), took / ms, t.ends[0].joined, if (t.ends[0].err) |e| @errorName(e) else "-", t.ends[1].joined, if (t.ends[1].err) |e| @errorName(e) else "-" });
    try std.testing.expectError(error.PeerTimeout, got);
    try std.testing.expect(took >= (short_ms - 50) * ms and took < late_ms * ms);
    for (t.ends) |e| try std.testing.expect(!e.joined and e.err != null);
}

test "qualification_1fee6d78: a longer window set through Config (TF_AFD_SETUP_MS's 10000, on all three) takes the same late expert; the three settings texts are unchanged" {
    var b: [256]u8 = undefined;
    const want = try Three.config(null).settings.text(&b);
    const long = try xchg.setup.parse("10000");
    var t: Three = .{ .delays = .{ 0, late_ms } };
    try t.init(Three.config(long), Three.config(long));
    const server = try cuda.link.Server.open(try loopback(0));
    try t.spawn(try loopback(server.port()));
    const got = t.nodes[0].attend(server, t.openers[0].opener());
    const took = nowNs() - t.t0;
    if (got) |_| t.nodes[0].stopExperts() else |_| {}
    const texts = t.textsAre(want);
    const windows = [3]u32{ t.nodes[0].cfg.setup_timeout_ms, t.nodes[1].cfg.setup_timeout_ms, t.nodes[2].cfg.setup_timeout_ms };
    t.nodes[0].deinit();
    server.close();
    t.deinit();
    std.debug.print("qualification_1fee6d78 WINDOW long {d} ms, expert 1 at {d} ms: attend {s} after {d} ms; experts joined {} {}, stopped {} {}; settings \"{s}\" on all three\n", .{ long.?, late_ms, if (got) |_| "ok" else |e| @errorName(e), took / ms, t.ends[0].joined, t.ends[1].joined, t.ends[0].stopped, t.ends[1].stopped, want });
    try got;
    try texts;
    try std.testing.expectEqual([3]u32{ 10_000, 10_000, 10_000 }, windows);
    try std.testing.expect(took >= (late_ms - 50) * ms); // later than the short window would have waited
    for (t.ends) |e| try std.testing.expect(e.joined and e.stopped and e.err == null);
}

test "qualification_1fee6d78: the other route, no switch: experts that start 1.5 s before the attention listens retry their connect within their own 120,000 ms (Link.join) and are attended" {
    var t: Three = .{};
    try t.init(Three.config(null), Three.config(null));
    const server = try boundNotListening(); // the attention's port before its process starts: a connect is refused
    try t.spawn(try loopback(server.port()));
    sleepUntil(t.t0, 1_500);
    const listening = posix.errno(posix.system.listen(server.fd, 4)) == .SUCCESS; // the attention starts: its listener first
    const got = if (listening) t.nodes[0].attend(server, t.openers[0].opener()) else error.ListenFailed;
    const took = nowNs() - t.t0;
    if (got) |_| t.nodes[0].stopExperts() else |_| {}
    var b: [256]u8 = undefined;
    const texts = t.textsAre(try Three.config(null).settings.text(&b));
    const window = t.nodes[1].cfg.setup_timeout_ms;
    t.nodes[0].deinit();
    server.close();
    t.deinit();
    std.debug.print("qualification_1fee6d78 WINDOW unset ({d} ms), the attention listening 1500 ms after both experts started: attend {s} after {d} ms; experts joined {} {}\n", .{ window, if (got) |_| "ok" else |e| @errorName(e), took / ms, t.ends[0].joined, t.ends[1].joined });
    try got;
    try texts;
    try std.testing.expectEqual(@as(u32, 120_000), window);
    try std.testing.expect(took >= 1_450 * ms);
    for (t.ends) |e| try std.testing.expect(e.joined and e.stopped and e.err == null);
}
