//! The expert half's TF_AFD_TIME clock (qualification_e5a7fc3b): where a 3-node step's time goes on the expert side. Each MoE position's
//! work is timed on the GPU with an event pair, from the request landed (the stream past the take's wait) to the reply
//! ready (the combine done, before the post); a forward's pairs are read after its one sync and summed. A TIME line
//! every `every` forwards gives the medians so far, decode windows (up to 8 rows) and prompt chunks apart: the experts'
//! share, against the attention's wait for both replies (tf-glm-afd attn bench's profile). Off unless TF_AFD_TIME=1, and
//! then the loop's launches are unchanged (events only).
const std = @import("std");
const cuda = @import("cuda");

/// TF_AFD_TIME=1 turns the clock on; anything else (or unset) leaves it off.
pub fn fromEnv() bool {
    const v = std.c.getenv("TF_AFD_TIME") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// A forward as the clock sees it: its MoE positions and its rows; `slot` a prefill pair's window (qualification_c2d4ee2e: B's events
/// are its own, so a pair is two forwards here), 0 for every other forward.
pub const Span = struct { first: usize, count: usize, rows: usize, slot: u32 = 0 };

/// The event pair of position `pos` of window `slot` among `layers` positions (slot 0: the position itself).
pub fn at(slot: u32, layers: usize, pos: usize) usize {
    return slot * layers + pos;
}

/// The median and the p90 (nearest rank) of `xs`, sorted in place (bench's rule).
pub fn stats(xs: []f64) [2]f64 {
    if (xs.len == 0) return .{ 0, 0 };
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const mid = if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
    const r = @min(xs.len - 1, (xs.len * 9 + 9) / 10 - 1);
    return .{ mid, xs[r] };
}

/// The forwards' summed milliseconds, decode windows and prompt chunks apart, and the TIME line over them.
pub const Tally = struct {
    gpa: std.mem.Allocator,
    decode: std.ArrayList(f64) = .empty,
    prompt: std.ArrayList(f64) = .empty,
    layers: u64 = 0,
    rows: u64 = 0,

    pub fn deinit(t: *Tally) void {
        t.decode.deinit(t.gpa);
        t.prompt.deinit(t.gpa);
    }

    /// One forward of `n` positions and `rows` rows took `ms` of GPU time over its positions.
    pub fn add(t: *Tally, ms: f64, n: usize, rows: usize) !void {
        try (if (rows <= 8) &t.decode else &t.prompt).append(t.gpa, ms);
        t.layers += n;
        t.rows += rows;
    }

    pub fn forwards(t: *const Tally) usize {
        return t.decode.items.len + t.prompt.items.len;
    }

    /// `TIME expert H: N forwards (...): decode windows median M ms, p90 P ms a forward (M / L ms a MoE layer); prompt
    /// chunks median ...`, over every forward so far.
    pub fn line(t: *const Tally, h: u32, layers: usize, buf: []u8) ![]const u8 {
        var tmp: [2][]f64 = undefined;
        var st: [2][2]f64 = undefined;
        for ([_][]const f64{ t.decode.items, t.prompt.items }, 0..) |xs, i| {
            tmp[i] = try t.gpa.dupe(f64, xs);
            st[i] = stats(tmp[i]);
        }
        defer for (tmp) |x| t.gpa.free(x);
        const per = if (layers > 0) st[0][0] / @as(f64, @floatFromInt(layers)) else 0;
        return std.fmt.bufPrint(buf, "TIME expert {d}: {d} forwards ({d} decode windows, {d} prompt chunks; {d} MoE layers, {d} rows in all): the half's experts from the request landed to the reply ready, decode windows median {d:.3} ms, p90 {d:.3} ms a forward ({d:.4} ms a MoE layer at {d} layers); prompt chunks median {d:.3} ms, p90 {d:.3} ms a forward", .{ h, t.forwards(), t.decode.items.len, t.prompt.items.len, t.layers, t.rows, st[0][0], st[0][1], per, layers, st[1][0], st[1][1] });
    }
};

/// The device side: an event pair a MoE position, the forward recorded last, the tally.
pub const Clock = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    h: u32,
    layers: usize,
    every: usize = 128,
    pairs: std.ArrayList([2]cuda.Event) = .empty,
    pending: [cuda.xchg.proto.LANES_MAX]?Span = @splat(null), // a forward's span; a prefill pair's two windows' (qualification_c2d4ee2e), a lane group's (qualification_9e49592e)
    tally: Tally,

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, h: u32, layers: usize) !*Clock {
        const c = try gpa.create(Clock);
        c.* = .{ .gpa = gpa, .d = d, .h = h, .layers = layers, .tally = .{ .gpa = gpa } };
        return c;
    }

    pub fn destroy(c: *Clock) void {
        for (c.pairs.items) |*p| for (p) |*e| e.deinit();
        c.pairs.deinit(c.gpa);
        c.tally.deinit();
        c.gpa.destroy(c);
    }

    /// Before position `pos`'s launches (`at`'s index for a pair's window B): the stream reaches this event once the
    /// take's wait has passed.
    pub fn begin(c: *Clock, pos: usize, stream: cuda.Stream) !void {
        while (c.pairs.items.len <= pos) try c.pairs.append(c.gpa, .{ try cuda.Event.init(c.d, true), try cuda.Event.init(c.d, true) });
        try c.pairs.items[pos][0].record(stream);
    }

    /// After position `pos`'s combine, before its post.
    pub fn end(c: *Clock, pos: usize, span: Span, stream: cuda.Stream) !void {
        try c.pairs.items[pos][1].record(stream);
        c.pending[span.slot] = span;
    }

    /// After the forward's (or the pair's) one sync: each window's pairs read and summed into the tally; a TIME line
    /// every `every` forwards.
    pub fn fold(c: *Clock) !void {
        for (&c.pending) |*pend| {
            const p = pend.* orelse continue;
            pend.* = null;
            var ms: f64 = 0;
            for (p.first..p.first + p.count) |pos| {
                const k = at(p.slot, c.layers, pos);
                ms += try cuda.Event.elapsedMs(c.pairs.items[k][0], c.pairs.items[k][1]);
            }
            try c.tally.add(ms, p.count, p.rows);
            if (c.tally.forwards() % c.every == 0) c.report();
        }
    }

    pub fn report(c: *Clock) void {
        var buf: [640]u8 = undefined;
        const l = c.tally.line(c.h, c.layers, &buf) catch return;
        std.debug.print("{s}\n", .{l});
    }
};

test "the clock's switch, its tally and its line (decode windows and prompt chunks apart)" {
    try std.testing.expect(!fromEnv() or std.mem.eql(u8, std.mem.span(std.c.getenv("TF_AFD_TIME").?), "1"));
    var t: Tally = .{ .gpa = std.testing.allocator };
    defer t.deinit();
    for (0..9) |i| try t.add(@as(f64, @floatFromInt(10 + i)), 42, 1); // 10..18 ms: median 14, p90 18
    try t.add(400, 42, 2048);
    try std.testing.expectEqual(@as(usize, 10), t.forwards());
    var buf: [640]u8 = undefined;
    const l = try t.line(1, 42, &buf);
    try std.testing.expect(std.mem.startsWith(u8, l, "TIME expert 1: 10 forwards (9 decode windows, 1 prompt chunks; 420 MoE layers, 2057 rows in all)"));
    try std.testing.expect(std.mem.indexOf(u8, l, "decode windows median 14.000 ms, p90 18.000 ms a forward (0.3333 ms a MoE layer at 42 layers)") != null);
    try std.testing.expect(std.mem.indexOf(u8, l, "prompt chunks median 400.000 ms") != null);
    var xs = [_]f64{ 3, 1, 2, 4 };
    try std.testing.expectEqual([2]f64{ 2.5, 4 }, stats(&xs));
    // qualification_c2d4ee2e: a pair's window B times events of its own, past window A's (slot 0: today's index)
    try std.testing.expectEqual([3]usize{ 5, 47, 83 }, [3]usize{ at(0, 42, 5), at(1, 42, 5), at(1, 42, 41) });
}
