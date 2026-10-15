//! A stand-in for roce_proxy.c between in-process regions (tests): it polls its node's doorbell and, for each new message in order, copies the send slot into the peer's receive slot and then writes the peer's flags, as the proxy's post does; with faults.
const std = @import("std");
const cuda = @import("cuda");
const xchg = cuda.xchg;
const proto = xchg.proto;
const trace = @import("trace.zig");

// Test synchronization at cursor capture, not a timer or an expert-ready flag.
pub const CursorGate = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    polled: std.atomic.Value(bool) = .init(false),
    cursor: std.atomic.Value(u32) = .init(0),
};

pub const Faults = struct {
    /// messages past this seq are never sent: a silent peer (its process, link and heartbeat live on)
    silent_after: ?u32 = null,
    /// this seq's flags are written one ahead: a protocol breach
    breach_at: ?u32 = null,
    /// at this seq the proxy reports a failed write and stops
    fail_at: ?u32 = null,
};

pub const Fake = struct {
    gpa: std.mem.Allocator,
    region: []u8,
    rank: u32,
    reg: proto.Region,
    ports: u32,
    faults: Faults,
    /// qualification_17c70e37: the settings' stripes (Node.pairUp sets them, as for roce_proxy.c), and a pause after each stripe so that
    /// a take sees a message land stripe by stripe
    stripes: u32 = 0,
    stripe_min: u64 = proto.STRIPE_MIN,
    stripe_pause_ns: u64 = 0,
    peer: ?[*]u8 = null,
    running: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    sent: std.atomic.Value(u64) = .init(0),
    thread: ?std.Thread = null,
    initialized: std.atomic.Value(bool) = .init(false),
    cursor_gate: ?*CursorGate = null,
    observer: trace.Writer = .{},
    request_header: bool = false,

    const vt: xchg.Proxy.VTable = .{ .blob = blob, .connect = connect, .start = start, .failure = failure, .ops = ops, .close = close, .stripes = setStripes };

    fn self(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }

    fn setStripes(ctx: *anyopaque, k: u32, min: u64) anyerror!void {
        const f = self(ctx);
        if (f.thread != null or k > proto.STRIPES_MAX) return error.Invalid;
        f.stripes = k;
        f.stripe_min = min;
    }

    fn blob(ctx: *anyopaque, out: []u8) anyerror!usize {
        const f = self(ctx);
        std.mem.writeInt(u64, out[0..8], @intFromPtr(f.region.ptr), .little);
        std.mem.writeInt(u64, out[8..16], f.reg.slot, .little);
        return 16;
    }

    fn connect(ctx: *anyopaque, peer: []const u8) anyerror!void {
        const f = self(ctx);
        if (peer.len != 16 or std.mem.readInt(u64, peer[8..16], .little) != f.reg.slot) return error.BadRoceBlob;
        f.peer = @ptrFromInt(std.mem.readInt(u64, peer[0..8], .little));
    }

    fn start(ctx: *anyopaque) anyerror!void {
        const f = self(ctx);
        f.initialized.store(false, .release);
        f.running.store(true, .release);
        f.thread = try std.Thread.spawn(.{}, run, .{f});
        // Ownership passes to the publisher only after this proxy captured its initial cursor.
        while (!f.initialized.load(.acquire)) std.atomic.spinLoopHint();
    }

    fn word(base: [*]u8, off: u64) *u32 {
        return @ptrCast(@alignCast(base + off));
    }

    fn run(f: *Fake) void {
        defer f.observer.put(46, .{f.sent.load(.acquire), @intFromBool(f.failed.load(.acquire)), 0, 0, 0, 0, 0, 0});
        const ctrl = f.reg.ctrl();
        if (f.cursor_gate) |g| {
            g.entered.store(true, .release);
            while (!g.release.load(.acquire)) std.atomic.spinLoopHint();
        }
        var posted: u32 = @atomicLoad(u32, word(f.region.ptr, ctrl), .acquire);
        f.observer.put(40, .{posted, f.rank, 0, 0, 0, 0, 0, 0});
        if (f.cursor_gate) |g| g.cursor.store(posted, .release);
        f.initialized.store(true, .release);
        while (f.running.load(.acquire)) {
            const seq = @atomicLoad(u32, word(f.region.ptr, ctrl), .acquire);
            if (f.cursor_gate) |g| g.polled.store(true, .release);
            if (seq == posted) {
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
                _ = std.c.nanosleep(&ts, null);
                continue;
            }
            var s = posted +% 1;
            while (true) : (s +%= 1) {
                if (!f.post(s)) return;
                posted = s;
                if (s == seq) break;
            }
        }
    }

    /// Message `s`, as roce_proxy.c's post: the bytes the doorbell names, then each port's flag; qualification_17c70e37: in stripes as
    /// post_striped sends them (each port's stripe j, then its word), a pause after each stripe.
    fn post(f: *Fake, s: u32) bool {
        f.observer.seq = s; f.observer.slot = f.reg.slotOf(s);
        f.observer.put(41, .{s, f.rank, f.sent.load(.acquire), 0, 0, 0, 0, 0});
        if (f.faults.fail_at) |at| if (s == at) {
            f.failed.store(true, .release);
            f.observer.put(44, .{s, f.rank, 0, 0, 0, 0, 0, 0});
            return false;
        };
        if (f.faults.silent_after) |after| if (s > after) {
            f.observer.put(43, .{s, f.rank, 0, 0, 0, 0, 0, 0});
            return true;
        };
        const slot = f.reg.slotOf(s);
        const bytes = @atomicLoad(u32, word(f.region.ptr, f.reg.ctrl() + 16 + 4 * @as(u64, slot)), .acquire);
        const from: usize = @intCast(f.reg.send(slot));
        const to: usize = @intCast(f.reg.recv(f.rank, slot));
        const value = if (f.faults.breach_at != null and f.faults.breach_at.? == s) s +% 1 else s;
        // Only this proxy owns this published send slot until it acknowledges consumption.
        if (f.request_header and bytes >= proto.HDR_BYTES) {
            if (proto.Header.read(f.region[from..][0..proto.HDR_BYTES])) |hdr| {
                f.observer.fseq = hdr.fseq; f.observer.layer = hdr.layer; f.observer.rows = hdr.rows;
            } else |_| {}
        }
        defer f.observer.put(42, .{bytes, value, f.sent.load(.acquire), f.rank, 0, 0, 0, 0});
        const n = proto.stripesOf(f.stripes, f.stripe_min, bytes);
        if (n > 1) {
            const units: u64 = bytes / 16;
            for (0..n) |j| {
                for (0..f.ports) |p| {
                    const sh = proto.portShare(units, f.ports, @intCast(p));
                    const st = proto.stripeUnits(sh[1], n, @intCast(j));
                    const lo: usize = @intCast(16 * (sh[0] + st[0]));
                    const len: usize = @intCast(16 * (st[1] - st[0]));
                    @memcpy(f.peer.?[to + lo ..][0..len], f.region[from + lo ..][0..len]);
                    if (j + 1 == n) _ = f.sent.fetchAdd(@intFromBool(p == 0), .acq_rel); // counted before the last flag
                    @atomicStore(u32, word(f.peer.?, f.reg.flag(f.rank, slot, @intCast(p)) + 4 * @as(u64, proto.stripeWord(n, @intCast(j)))), value, .release);
                }
                if (f.stripe_pause_ns > 0 and j + 1 < n) {
                    const ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(f.stripe_pause_ns) };
                    _ = std.c.nanosleep(&ts, null);
                }
            }
            return true;
        }
        @memcpy(f.peer.?[to..][0..bytes], f.region[from..][0..bytes]);
        _ = f.sent.fetchAdd(1, .acq_rel); // counted before the flag, so a peer that saw the flag sees the count
        for (0..f.ports) |p| @atomicStore(u32, word(f.peer.?, f.reg.flag(f.rank, slot, @intCast(p))), value, .release);
        return true;
    }

    fn failure(ctx: *anyopaque) ?[]const u8 {
        return if (self(ctx).failed.load(.acquire)) "fake proxy: RDMA write failed (injected)" else null;
    }

    fn ops(ctx: *anyopaque) u64 {
        return self(ctx).sent.load(.acquire);
    }

    fn close(ctx: *anyopaque) void {
        const f = self(ctx);
        f.running.store(false, .release);
        if (f.thread) |t| t.join();
        f.gpa.destroy(f);
    }
};

/// Opens Fakes with each pair rank's faults (the attention is rank 1 with expert 0 and rank 0 with expert 1).
pub const Opener = struct {
    gpa: std.mem.Allocator,
    ports: u32 = 1,
    faults: [2]Faults = .{ .{}, .{} },
    /// qualification_17c70e37: each fake's pause after a stripe
    stripe_pause_ns: u64 = 0,
    recorder: ?*trace.Recorder = null,
    owner: u32 = 0,

    pub fn opener(o: *Opener) xchg.Opener {
        return .{ .ctx = o, .open = open };
    }

    fn open(ctx: ?*anyopaque, rank: u32, region: []u8, reg: proto.Region) anyerror!xchg.Proxy {
        const o: *Opener = @ptrCast(@alignCast(ctx.?));
        const f = try o.gpa.create(Fake);
        f.* = .{ .gpa = o.gpa, .region = region, .rank = rank, .reg = reg, .ports = o.ports, .faults = o.faults[rank], .stripe_pause_ns = o.stripe_pause_ns, .request_header = o.owner == 0 };
        if (o.recorder) |r| f.observer = r.writer(if (o.owner == 0) 4-rank else 4+o.owner);
        return .{ .ctx = f, .vt = &Fake.vt };
    }
};

const Starter = struct {
    f: *Fake,
    returned: std.atomic.Value(bool) = .init(false),
    fn run(s: *Starter) void {
        Fake.start(s.f) catch @panic("proxy start failed");
        s.returned.store(true, .release);
    }
};
test "ping_pong_fixture cursor acknowledgement is per proxy and publication after return is delivered exactly once" {
    const gpa = std.testing.allocator;
    const reg: proto.Region = .{ .slot = 4096 };
    var src: [2][]align(4096) u8 = undefined;
    var dst: [2][]align(4096) u8 = undefined;
    var gates: [2]CursorGate = .{ .{}, .{} };
    var fs: [2]*Fake = undefined;
    var starters: [2]Starter = undefined;
    var starts: [2]std.Thread = undefined;
    for (0..2) |k| {
        src[k] = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(reg.total()));
        dst[k] = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(reg.total()));
        @memset(src[k], 0); @memset(dst[k], 0);
        fs[k] = try gpa.create(Fake);
        fs[k].* = .{ .gpa = gpa, .region = src[k], .rank = 0, .reg = reg, .ports = 1, .faults = .{}, .peer = dst[k].ptr, .cursor_gate = &gates[k] };
        starters[k] = .{ .f = fs[k] };
        starts[k] = try std.Thread.spawn(.{}, Starter.run, .{&starters[k]});
    }
    for (&gates) |*g| while (!g.entered.load(.acquire)) std.atomic.spinLoopHint();
    gates[0].release.store(true, .release);
    while (!starters[0].returned.load(.acquire)) std.atomic.spinLoopHint();
    const first_initialized = fs[0].initialized.load(.acquire);
    const second_not_returned = !starters[1].returned.load(.acquire) and !fs[1].initialized.load(.acquire);
    gates[1].release.store(true, .release);
    for (starts) |t| t.join();
    defer for (0..2) |k| { Fake.close(fs[k]); gpa.free(src[k]); gpa.free(dst[k]); };
    for (0..2) |k| {
        @memset(src[k][@intCast(reg.send(reg.slotOf(1)))..][0..16], @intCast(77+k));
        @atomicStore(u32, Fake.word(src[k].ptr, reg.ctrl()+16+4*reg.slotOf(1)), 16, .release);
        @atomicStore(u32, Fake.word(src[k].ptr, reg.ctrl()), 1, .release);
    }
    const deadline = trace.now() + 5 * std.time.ns_per_s;
    for (0..2) |k| {
        while (@atomicLoad(u32, Fake.word(dst[k].ptr, reg.flag(0, reg.slotOf(1), 0)), .acquire) != 1 and trace.now() < deadline) std.atomic.spinLoopHint();
        fs[k].running.store(false, .release); fs[k].thread.?.join(); fs[k].thread = null;
        std.debug.print("ping_pong_fixture_CURSOR_AFTER proxy={d} initial_cursor={d} published=1 sent={d} peer_flag={d}\n", .{k, gates[k].cursor.load(.acquire), fs[k].sent.load(.acquire), @atomicLoad(u32, Fake.word(dst[k].ptr, reg.flag(0, reg.slotOf(1), 0)), .acquire)});
        try std.testing.expectEqual(@as(u32, 0), gates[k].cursor.load(.acquire));
        try std.testing.expectEqual(@as(u64, 1), fs[k].sent.load(.acquire));
        try std.testing.expectEqual(@as(u32, 1), @atomicLoad(u32, Fake.word(dst[k].ptr, reg.flag(0, reg.slotOf(1), 0)), .acquire));
        try std.testing.expect(std.mem.allEqual(u8, dst[k][@intCast(reg.recv(0, reg.slotOf(1)))..][0..16], @intCast(77+k)));
    }
    try std.testing.expect(first_initialized and second_not_returned);
}
