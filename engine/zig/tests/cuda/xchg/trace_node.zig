//! ping_pong_fixture test-owned atomic observer; never reads Node.rule or claims to be Rule.step's input.
const std = @import("std");
const cuda = @import("cuda");
const trace = @import("trace.zig");
pub const Monitor = struct {
    node: *cuda.xchg.Node,
    writer: trace.Writer,
    stop: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    pub fn start(m: *Monitor) !void { m.thread = try std.Thread.spawn(.{}, run, .{m}); }
    pub fn join(m: *Monitor) void {
        m.stop.store(true, .release);
        if (m.thread) |t| t.join();
        m.thread = null;
    }
    fn run(m: *Monitor) void {
        var old: [8]u64 = @splat(std.math.maxInt(u64));
        var beats: [8]u64 = old;
        while (!m.stop.load(.acquire)) {
            const n = m.node;
            var data: [8]u64 = @splat(0);
            for (0..5) |i| data[i] = @atomicLoad(u32, &n.mem.status[i], .acquire);
            const seq = n.doorbell();
            data[5] = seq;
            for (0..2) |k| data[6] |= @as(u64, @intFromBool(seq == 0 or n.landedFrom(n.src(@intCast(k)), seq))) << @intCast(k);
            data[7] = @intFromBool(n.cfg.run_watcher);
            m.writer.seq = seq; m.writer.slot = n.reg.slotOf(seq);
            if (!std.mem.eql(u64, &old, &data)) { m.writer.put(70, data); old = data; }
            const live: [8]u64 = .{n.beats[0].load(.acquire), n.beats[1].load(.acquire), n.proxies[0].?.ops(), n.proxies[1].?.ops(), seq, data[6], 0, 0};
            if (!std.mem.eql(u64, &beats, &live)) { m.writer.put(71, live); beats = live; }
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 200_000 };
            _ = std.c.nanosleep(&ts, null);
        }
        m.writer.put(72, @splat(0));
    }
};
