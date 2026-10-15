//! grammar_mask_2: qualification_2acf601e's device-gate FAIL on CPU. The MOCK device sees a sink's row as the forward's non-blocking stream
//! sees a copy made off it: at once when the copy beat the next draw, else only after a fence the forward records. With
//! row 5's copy late (the 5090's schedule) an unfenced gate draws row 4's winner; a fenced one passes every schedule.
const std = @import("std");
const glm = @import("glm_cuda");
const fx = @import("grammar_mask_fixture.zig");
const md = glm.sample.mask_dev;
const a = std.testing.allocator;

/// sha256 of the gate's 128 uploads (seed 0x4752_4132, V 154880) as evidence/grammar_mask_2/diag/diag_gate64.zig replays them.
const uploads_sha256 = "732060d8712a40f8fe07421f0b02a24ea1bae5f8f249517b74324e60e7471540";

/// The MOCK device's gate logits: `seen` is what its draws read, `sent` the sink's last row until that copy lands.
pub const LateRow = struct {
    t: *glm.trace.Trace,
    logits: u64,
    schedule: []const usize,
    seen: []f32,
    sent: []f32,
    tmp: []f32,
    pending: bool = false,
    mark: usize = 0,
    puts: usize = 0,
    stale_draws: usize = 0,
    fenced: usize = 0,
    digest: std.crypto.hash.sha2.Sha256 = .init(.{}),

    /// `schedule`: the puts (1-based) whose copy has not landed by the next draw unless the forward fenced it.
    pub fn init(t: *glm.trace.Trace, logits: u64, width: usize, schedule: []const usize) !LateRow {
        const seen = try a.alloc(f32, width);
        errdefer a.free(seen);
        @memset(seen, 0); // a fresh buffer
        const sent = try a.alloc(f32, width);
        errdefer a.free(sent);
        return .{ .t = t, .logits = logits, .schedule = schedule, .seen = seen, .sent = sent, .tmp = try a.alloc(f32, width) };
    }
    pub fn deinit(g: *LateRow) void {
        a.free(g.seen);
        a.free(g.sent);
        a.free(g.tmp);
    }
    fn of(p: *anyopaque) *LateRow {
        return @ptrCast(@alignCast(p));
    }
    pub fn sink(g: *LateRow) md.Sink {
        return .{ .ctx = g, .put = put };
    }
    fn put(p: *anyopaque, bits: []const u16) !void {
        const g = of(p);
        g.puts += 1;
        g.digest.update(std.mem.sliceAsBytes(bits));
        for (bits, g.sent) |b, *x| x.* = @bitCast(@as(u32, b) << 16);
        g.pending = true;
        g.mark = g.t.events.items.len;
        if (std.mem.indexOfScalar(usize, g.schedule, g.puts) == null) g.land();
    }
    fn land(g: *LateRow) void {
        @memcpy(g.seen, g.sent);
        g.pending = false;
    }
    /// Before a draw: a fence recorded since the put lands its row; otherwise this draw reads the buffer as it was.
    fn sync(g: *LateRow) void {
        if (!g.pending) return;
        for (g.t.events.items[g.mark..]) |e| {
            const into = std.mem.eql(u8, e.name, "torch.copy_") and e.vals.len > 0 and e.vals[0] == .t and e.vals[0].t.ptr == g.logits;
            if (into or std.mem.eql(u8, e.name, "torch.cuda.synchronize")) {
                g.fenced += 1;
                return g.land();
            }
        }
        g.stale_draws += 1;
    }
    /// The late copy lands just after the draw that missed it.
    fn after(g: *LateRow) void {
        if (g.pending) g.land();
    }
    pub fn picker(g: *LateRow) glm.trace.Picker {
        return .{ .ctx = g, .pick = plain };
    }
    fn plain(p: *anyopaque, _: glm.trace.T, _: ?[]const u64, out: []u32, _: ?[]f64) !void {
        const g = of(p);
        g.sync();
        var order: [1]usize = undefined;
        glm.sample.topkOrder(g.seen, 1, &order);
        @memset(out, @intCast(order[0]));
        g.after();
    }
    pub fn emulator(g: *LateRow) md.Emulator {
        return .{ .ctx = g, .candidate = device };
    }
    fn device(p: *anyopaque, x: *const md.Dev, _: glm.trace.T) !md.Candidate {
        const g = of(p);
        g.sync();
        @memcpy(g.tmp, g.seen);
        md.maskRow(g.tmp, x.host[0..x.words]);
        var order: [1]usize = undefined;
        glm.sample.topkOrder(g.tmp, 1, &order);
        g.after();
        return .{ .id = @intCast(order[0]), .bits = @bitCast(g.tmp[order[0]]) };
    }
};

const every_put = blk: {
    var x: [128]usize = undefined;
    for (&x, 1..) |*v, k| v.* = k;
    break :blk x;
};

/// The 64-row gate under three landing schedules (qualification_2acf601e's put 9 late, every put late, none): 64/64, no stale draw.
pub fn late(comptime Bound: type) !void {
    const schedules = [_]struct { name: []const u8, puts: []const usize }{
        .{ .name = "qualification_2acf601e (put 9 late)", .puts = &.{9} },
        .{ .name = "every put late", .puts = &every_put },
        .{ .name = "none late", .puts = &.{} },
    };
    for (schedules) |s| {
        var c: fx.Chain(Bound) = .{};
        try c.init(true);
        defer c.deinit();
        const V = c.rec.m.vocab;
        var lb = glm.views.Buf.of(.bf16, &.{ 1, V });
        try c.rec.space.place("gate.logits", &lb);
        var g = try LateRow.init(&c.rec.t, lb.ptr, V, s.puts);
        defer g.deinit();
        c.rec.t.picker = g.picker();
        c.dev.emulate = g.emulator();
        const rows = 64;
        const r = try md.gate(&c.dev, &c.f, &c.rec.e.smp, a, glm.trace.T.whole(lb), g.sink(), rows, 0x4752_4132);
        try std.testing.expect(r.rows == rows and r.identity == rows and r.forbidden == rows and r.single == rows and r.all_inf == rows);
        try std.testing.expectEqual(@as(usize, 2 * rows), g.puts);
        try std.testing.expectEqual(@as(usize, 0), g.stale_draws);
        try std.testing.expectEqual(s.puts.len, g.fenced);
        var sum: [32]u8 = undefined;
        g.digest.final(&sum);
        const hex = std.fmt.bytesToHex(sum, .lower);
        try std.testing.expectEqualStrings(uploads_sha256, &hex);
        std.debug.print("grammar_mask_2 gate (MOCK device, {s}): {d} rows, identity {d}, forbidden {d}, single {d}, all-inf {d}; late copies fenced {d}, stale draws {d}; uploads sha256 {s}\n", .{ s.name, r.rows, r.identity, r.forbidden, r.single, r.all_inf, g.fenced, g.stale_draws, hex[0..16] });
    }
}
