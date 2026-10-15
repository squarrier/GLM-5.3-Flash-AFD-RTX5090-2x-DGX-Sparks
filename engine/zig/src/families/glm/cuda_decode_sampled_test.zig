//! qualification_qualification_44ba554e.7 day 2's G-D2-equal: sampled drafted == sampled serial for the same seed. python's keyed draws (exact_sampling:
//! a Gumbel-max over uniform(seed, position, id)) keep a draft exactly when serial sampling would have drawn it, so a
//! verify window commits the tokens a serial run draws. Every draw here is qualification_45c10bcf's host draw (nucleus_draw.drawRow, Mia's
//! union cover) on a toy row of candidates a position: the target's row at absolute position p, and the head's row
//! there, the target's or (at `miss` per mille of positions) another. A draft's probability is its share of the mass,
//! the confidence floor's input, as python's nucleus_rows gives it. The loops run on a CPU recording, as day 1's toy.
const std = @import("std");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const nd = @import("nucleus_draw.zig");
const decode = @import("cuda_decode.zig");
const dt = @import("cuda_decode_test.zig");

const testing = std.testing;

/// The toy: each rank sends `count` candidates a row, three drawn from a hash of (p, rank, slot) and a last one far
/// below (mass 0), so a nucleus always ends inside the candidates; at `eos_at` the target's row puts the eos far above.
const Keyed = struct {
    e: *const dt.E,
    s: nd.Sampling,
    miss: u64,
    eos_at: ?usize,
    sh: nd.Shards,

    const count = 4;
    const stride = 1 + 1 + 3 * count;

    fn hash(p: usize, salt: u64) u64 {
        var h = std.hash.Wyhash.init(0x5eed ^ salt);
        h.update(std.mem.asBytes(&p));
        return h.final();
    }

    /// The draw at absolute position p from the target's row, or from the head's (`head`).
    fn draw(k: *const Keyed, p: usize, head: bool) !nd.Drawn {
        const m = k.e.m;
        const vl = m.vl();
        const wrong: u64 = if (head and hash(p, 2) % 1000 < k.miss) 100 else 0;
        const eos: ?u32 = if (!head and k.eos_at == p) m.eos[0] else null;
        var v: [nd.max_world][count]f64 = undefined;
        var ids: [nd.max_world][count]i64 = undefined;
        var top = -std.math.inf(f64);
        for (0..m.world) |r| for (0..count) |j| {
            const logit: f64 = if (j == count - 1) -60.0 else @as(f64, @floatFromInt(hash(p, 16 * r + j + wrong) % 4000)) / 1000.0;
            v[r][j] = logit / k.s.temperature;
            ids[r][j] = @intCast(r * vl + j * (vl / count) + hash(p, 50 + 16 * r + j) % (vl / count));
            if (eos) |x| if (ids[r][j] == x) {
                ids[r][j] -= 1;
            };
        };
        if (eos) |x| {
            const r = x / vl;
            v[r][0] = 50.0;
            ids[r][0] = x;
        }
        for (0..m.world) |r| for (0..count) |j| {
            top = @max(top, v[r][j]);
        };
        var words: [nd.max_world * stride]u64 = undefined;
        for (0..m.world) |r| {
            const b = r * stride;
            var sum: u64 = 0;
            words[b] = 0;
            for (0..count) |j| {
                const mass: i64 = @intFromFloat(@floor(@exp(v[r][j] - top) * 1099511627776.0)); // MASS = 2**40
                words[b + 2 + j] = @bitCast(v[r][j]);
                words[b + 2 + count + j] = @bitCast(ids[r][j]);
                words[b + 2 + 2 * count + j] = @bitCast(mass);
                sum += @intCast(mass);
            }
            words[b + 1] = sum;
        }
        const pk: nd.Packet = .{ .words = words[0 .. m.world * stride], .world = m.world, .rows = 1, .count = count };
        var work: [nd.max_world * count]nd.Cand = undefined;
        return (try nd.drawRow(pk, k.sh, 0, p, k.s, &work)) orelse error.NotCovered;
    }

    /// The draw at the positions the Engine passes its nucleus draw (python's: checked against the state's).
    fn pick(ctx: *anyopaque, logits: trace.T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        const k: *const Keyed = @ptrCast(@alignCast(ctx));
        const kind = dt.kindOf(k.e, logits);
        const positions = at orelse return error.NotSampled;
        for (out, positions, 0..) |*o, p, r| {
            try testing.expectEqual(@as(u64, dt.position(k.e, kind, r)), p);
            const d = try k.draw(@intCast(p), kind == .draft);
            o.* = d.token;
            if (probs) |ps| ps[r] = d.share;
        }
    }
};

test "G-D2-equal: sampled drafted == sampled serial for one seed, through qualification_45c10bcf's keyed draw at every depth and policy" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, dt.fixture_mlx, .{});
    defer parsed.deinit();
    const r = try dt.Rank.init(gpa, parsed.value);
    defer r.deinit(gpa);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2 };
    const samplings = [_]nd.Sampling{
        .{ .seed = 1234, .temperature = 1.0, .top_p = 0.95 },
        .{ .seed = 99, .temperature = 0.7, .top_p = 0.9, .min_p = 0.05 },
        .{ .seed = 7, .temperature = 1.3, .top_p = 1.0, .min_p = 0.1 }, // no top_p cut: min_p alone
    };
    const policies = [_]decode.DepthPolicy{ .{ .most = 1, .fixed = true }, .{ .most = 3, .fixed = true }, .{ .most = 7, .fixed = true }, .{}, decode.DepthPolicy.auto(true), .{ .most = 7, .fixed = true, .confidence = 0.2 }, decode.DepthPolicy.auto(false) };
    var runs: usize = 0;
    var drafted: usize = 0;
    var accepted: usize = 0;
    for (samplings) |s| for ([_]u64{ 0, 300, 1000 }) |miss| for ([_]usize{ 1, 6, 19 }) |count| for ([_]?usize{ null, prompt.len + 4 }) |eos_at| {
        var k: Keyed = .{ .e = r.e, .s = s, .miss = miss, .eos_at = eos_at, .sh = nd.Shards.uniform(r.m.world, r.m.vl()) };
        t.picker = .{ .ctx = &k, .pick = Keyed.pick };
        defer t.picker = null;
        r.e.sampling = s; // every draw takes the nucleus path; the recording's picker draws it
        defer r.e.sampling = null;
        const stop = eos_at != null;
        var serial_out: [32]u32 = undefined;
        r.fresh(&t);
        var f = fwd.Fwd.recording(&r.m, &t);
        const first = try r.e.prefill(&f, &prompt, true);
        const ser = try decode.serial(r.e, &f, first, count, &serial_out, stop);
        for (ser.tokens, 0..) |x, i| try testing.expectEqual((try k.draw(prompt.len + i, false)).token, x); // keyed at p
        for (policies) |p0| for ([_]bool{ false, true }) |auto| {
            errdefer std.debug.print("seed {d} miss {d} count {d} eos {?d} auto {} policy {any}\n", .{ s.seed, miss, count, eos_at, auto, p0 });
            var policy = p0;
            var out: [32]u32 = undefined;
            r.fresh(&t);
            f = fwd.Fwd.recording(&r.m, &t);
            try testing.expectEqual(first, try r.e.prefill(&f, &prompt, true));
            const d = if (auto) try r.e.autoDecode(&f, first, count, &out, &policy, stop) else try r.e.mtpDecode(&f, first, count, &out, &policy, stop);
            try testing.expectEqualSlices(u32, ser.tokens, d.tokens);
            try testing.expect(d.accepted <= d.drafted and d.rounds <= ser.rounds);
            drafted += d.drafted;
            accepted += d.accepted;
            runs += 1;
        };
    };
    try testing.expectEqual(@as(usize, 3 * 3 * 3 * 2 * 7 * 2), runs);
    std.debug.print("G-D2-equal: {d} sampled runs equal to serial; drafts {d}, kept {d}\n", .{ runs, drafted, accepted });
    try testing.expect(accepted > 0 and accepted < drafted); // drafts kept and drafts rejected, both
}
