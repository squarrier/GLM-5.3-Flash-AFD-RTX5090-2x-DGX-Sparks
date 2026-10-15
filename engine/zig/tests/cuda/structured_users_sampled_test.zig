//! structured_users: grammar-constrained sampling (TF_GLM_SAMPLING=1 with TF_AFD_GRAMMAR_MASK=1). The masked adapter masks the
//! widened row before nucleus_rows' temperature, maxima, masses and cuts (cuda_mask_dev.Dev.sampled), so a disallowed
//! token has no mass and cannot be drawn; the drawn token is checked eligible on the host (cuda_mask_dev.drawn).
//! (1) The host draw (nucleus_draw.levels, python-0.6's _draw with Mia's union cover) over MOCK device rows (the
//! kernel's CPU reference maskRow over the words, then the f64 scaling, maxima and fixed-point masses): a fixed-seed
//! sweep of 12,000 draws on a 4,099-token vocabulary and 480 on GLM's 154,880, never a disallowed token, while the same
//! keyed draws unmasked do pick disallowed ones. (2) The recorded launches of a masked sampled draw and its refusals.
//! (3) A sampled structured stream through the actual AFD Bound chain (recorded), keyed at the plain sampled positions.
//! Mia (@MiaAI_lab) authored the nucleus recipe and its union cover; Ash Hart the grammar_mask kernel and lane core.
//! Claude Opus 5.5 via Hermes.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const fx = @import("grammar_mask_fixture.zig");
const nd = glm.nucleus_draw;
const md = glm.sample.mask_dev;
const mask_cpu = glm.sample.mask_cpu;
const t = std.testing;
const gpa = t.allocator;

fn bf16(v: f32) f32 {
    const b: u32 = @bitCast(v);
    return @bitCast((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000);
}

fn set(mask: []u32, id: usize) void {
    mask[id / 32] |= @as(u32, 1) << @as(u5, @intCast(id % 32));
}

fn clear(mask: []u32, id: usize) void {
    mask[id / 32] &= ~(@as(u32, 1) << @as(u5, @intCast(id % 32)));
}

/// One device row as nucleus_rows' prefix leaves it (MOCK, CPU): `x * (1/t)` in f64, the row's maximum,
/// `floor(exp(scaled - top) * 2^40)` (a -inf column: 0), the shard's sum; and nucleus_draw.levels' source over it (one
/// rank: its top `count` candidates by value, ties to the lower id).
const Rows = struct {
    V: usize,
    scaled: []f64,
    mass: []i64,
    order: []u32,
    buf: []u64,
    sum: u64 = 0,
    gathers: usize = 0,

    fn init(V: usize) !Rows {
        const scaled = try gpa.alloc(f64, V);
        errdefer gpa.free(scaled);
        const mass = try gpa.alloc(i64, V);
        errdefer gpa.free(mass);
        const order = try gpa.alloc(u32, V);
        errdefer gpa.free(order);
        return .{ .V = V, .scaled = scaled, .mass = mass, .order = order, .buf = try gpa.alloc(u64, nd.Packet.stride(1, V)) };
    }
    fn deinit(x: *Rows) void {
        gpa.free(x.scaled);
        gpa.free(x.mass);
        gpa.free(x.order);
        gpa.free(x.buf);
    }
    fn fill(x: *Rows, row: []const f32, temperature: f64) void {
        const inv = 1.0 / temperature;
        var top = -std.math.inf(f64);
        for (row, x.scaled) |v, *s| {
            s.* = @as(f64, v) * inv;
            top = @max(top, s.*);
        }
        x.sum = 0;
        for (x.scaled, x.mass) |s, *m| {
            const d = s - top;
            m.* = if (d > -std.math.inf(f64)) @intFromFloat(@floor(@exp(d) * 0x1p40)) else 0;
            x.sum += @intCast(m.*);
        }
        for (x.order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, x.order, x.scaled, struct {
            fn less(s: []f64, a: u32, b: u32) bool {
                return s[a] > s[b] or (s[a] == s[b] and a < b);
            }
        }.less);
    }
    pub fn gather(x: *Rows, count: usize) !nd.Packet {
        x.gathers += 1;
        const c = @min(count, x.V);
        const w = x.buf[0..nd.Packet.stride(1, c)];
        w[0] = 0;
        w[1] = x.sum;
        for (x.order[0..c], 0..) |id, j| {
            w[2 + j] = @bitCast(x.scaled[id]);
            w[2 + c + j] = id;
            w[2 + 2 * c + j] = @bitCast(x.mass[id]);
        }
        return .{ .words = w, .world = 1, .rows = 1, .count = c };
    }
    pub fn again(_: *Rows) !nd.Packet {
        return error.UnexpectedAgain; // MOCK rows never carry a bad id
    }
};

extern "c" fn log(x: f64) f64; // libm's, as nucleus_draw's (python's math.log)

/// REFERENCE, independent of nucleus_draw.levels and of any -inf: the keyed draw over the ALLOWED tokens alone, as the
/// card's rule reads (the mask before the sampling): their `x / t`, masses against their own maximum, then top_k, top_p
/// (python's exact need) and min_p over those alone, then the keyed Gumbel argmax (python-0.6's _draw). Its token and
/// share are what a masked sampled draw must give. `init` once a row, `draw` a position.
const Allowed = struct {
    c: []nd.Cand,
    s: nd.Sampling,
    keep: usize = 0,
    total: u64 = 0,

    fn init(row: []const f32, mask: []const u32, s: nd.Sampling, c: []nd.Cand) Allowed {
        const inv = 1.0 / s.temperature;
        var top = -std.math.inf(f64);
        for (row, 0..) |v, i| if (mask_cpu.eligible(mask, @intCast(i))) {
            top = @max(top, @as(f64, v) * inv);
        };
        var n: usize = 0;
        for (row, 0..) |v, i| if (mask_cpu.eligible(mask, @intCast(i))) {
            const x = @as(f64, v) * inv;
            c[n] = .{ .v = x, .id = @intCast(i), .m = @intFromFloat(@floor(@exp(x - top) * 0x1p40)) };
            n += 1;
        };
        std.mem.sort(nd.Cand, c[0..n], {}, struct {
            fn before(_: void, a: nd.Cand, b: nd.Cand) bool {
                return a.v > b.v or (a.v == b.v and a.id < b.id);
            }
        }.before);
        const limit = if (s.top_k > 0) @min(n, @as(usize, s.top_k)) else n;
        var total: u64 = 0;
        for (c[0..limit]) |x| total += @intCast(x.m);
        var keep = limit;
        if (s.top_p > 0.0 and s.top_p < 1.0) {
            const need = nd.need(s.top_p, total);
            var acc: u64 = 0;
            for (c[0..limit], 0..) |x, i| {
                acc += @intCast(x.m);
                if (acc >= need) {
                    keep = i + 1;
                    break;
                }
            }
        }
        if (s.min_p > 0.0) {
            const floor = top + s.minLog();
            var below: usize = 0;
            for (c[0..n]) |x| below += @intFromBool(x.v >= floor);
            keep = @min(keep, below);
        }
        return .{ .c = c, .s = s, .keep = keep, .total = total };
    }

    fn draw(x: *const Allowed, position: u64) nd.Drawn {
        var best: usize = 0;
        var best_score: f64 = 0;
        for (x.c[0..x.keep], 0..) |y, j| {
            const score = y.v - log(-log(nd.uniform(x.s.seed, position, @bitCast(y.id))));
            if (j == 0 or score > best_score) {
                best = j;
                best_score = score;
            }
        }
        return .{ .token = @intCast(x.c[best].id), .share = @as(f64, @floatFromInt(x.c[best].m)) / @as(f64, @floatFromInt(x.total)) };
    }
};

const Kind = enum { single, sparse, half, tail, below_top };

/// A seeded row: bf16 normals, a peaked head of 24 tokens far above; and its mask of `kind`, the plain argmax always
/// disallowed (so every row's most likely token is outside the grammar).
fn makeRow(r: std.Random, row: []f32, mask: []u32, kind: Kind) !void {
    const V = row.len;
    for (row) |*v| v.* = bf16(@floatCast(r.floatNorm(f64) * 2.0));
    for (0..24) |_| {
        const i = r.uintLessThan(usize, V);
        row[i] = bf16(row[i] + 9.0);
    }
    var top: usize = 0;
    for (row, 0..) |v, i| if (v > row[top]) {
        top = i;
    };
    @memset(mask, 0);
    switch (kind) {
        .single => set(mask, (top + 1 + r.uintLessThan(usize, V - 1)) % V),
        .sparse => for (0..V) |i| if (r.uintLessThan(u32, 100) == 0 and row[i] < 5.0) set(mask, i),
        .half => for (0..V) |i| if (r.boolean()) set(mask, i),
        .tail => {
            var lows: [8]usize = undefined;
            for (&lows, 0..) |*l, k| l.* = k;
            for (8..V) |i| {
                var hi: usize = 0;
                for (lows, 0..) |l, k| if (row[l] > row[lows[hi]]) {
                    hi = k;
                };
                if (row[i] < row[lows[hi]]) lows[hi] = i;
            }
            for (lows) |l| set(mask, l);
        },
        .below_top => {
            for (0..V) |i| set(mask, i);
            var cut: [32]usize = undefined;
            for (&cut, 0..) |*c, k| c.* = k;
            for (32..V) |i| {
                var lo: usize = 0;
                for (cut, 0..) |c, k| if (row[c] < row[cut[lo]]) {
                    lo = k;
                };
                if (row[i] > row[cut[lo]]) cut[lo] = i;
            }
            for (cut) |c| clear(mask, c);
        },
    }
    clear(mask, top);
    if (V % 32 != 0) mask[mask.len - 1] &= (@as(u32, 1) << @as(u5, @intCast(V % 32))) - 1;
    if (std.mem.allEqual(u32, mask, 0)) set(mask, (top + 1) % V);
}

const Tally = struct { draws: usize = 0, disallowed: usize = 0, reference_differs: usize = 0, unmasked_disallowed: usize = 0, top_k: usize = 0, gathers: [3]usize = @splat(0) };

/// `rows` seeded rows of width V, `positions` keyed draws each (sampling settings cycling over the row index): every
/// masked draw checked by the product's own host check (cuda_mask_dev.drawn) and against the words directly.
fn sweepOf(V: usize, rows: usize, positions: usize, seed: u64, tally: *Tally) !void {
    const row = try gpa.alloc(f32, V);
    defer gpa.free(row);
    const masked = try gpa.alloc(f32, V);
    defer gpa.free(masked);
    const mask = try gpa.alloc(u32, (V + 31) / 32);
    defer gpa.free(mask);
    const work = try gpa.alloc(nd.Cand, V);
    defer gpa.free(work);
    var mrows = try Rows.init(V);
    defer mrows.deinit();
    var prows = try Rows.init(V);
    defer prows.deinit();
    const ref_work = try gpa.alloc(nd.Cand, V);
    defer gpa.free(ref_work);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const temps = [_]f64{ 0.3, 0.8, 1.0, 1.7 };
    const top_ps = [_]f64{ 0.5, 0.9, 0.95, 1.0, 1.0 };
    const min_ps = [_]f64{ 0.0, 0.0, 0.05, 0.1 };
    const top_ks = [_]u32{ 0, 0, 5, 40, 0, 20 };
    const kinds = [_]Kind{ .single, .sparse, .half, .tail, .below_top };
    const sh = nd.Shards.uniform(1, V);
    for (0..rows) |i| {
        try makeRow(r, row, mask, kinds[i % kinds.len]);
        @memcpy(masked, row);
        md.maskRow(masked, mask); // the kernel's CPU reference over the staged words
        const s: nd.Sampling = .{ .seed = seed +% i, .temperature = temps[i % temps.len], .top_k = top_ks[(i / 4) % top_ks.len], .top_p = top_ps[(i / 2) % top_ps.len], .min_p = min_ps[(i / 3) % min_ps.len] };
        tally.top_k += @intFromBool(s.top_k != 0);
        mrows.fill(masked, s.temperature);
        prows.fill(row, s.temperature);
        const allowed = Allowed.init(row, mask, s, ref_work);
        for (0..positions) |k| {
            const pos = [_]u64{1000 + 7 * k};
            var out: [1]nd.Drawn = undefined;
            const g = try nd.levels(&mrows, sh, &pos, s, work, &out);
            tally.gathers[g - 1] += 1;
            tally.draws += 1;
            if (md.drawn(V, mask, out[0].token)) |_| {} else |_| tally.disallowed += 1;
            if (!mask_cpu.eligible(mask, out[0].token)) tally.disallowed += 1;
            const ref = allowed.draw(pos[0]);
            tally.reference_differs += @intFromBool(ref.token != out[0].token or ref.share != out[0].share);
            var plain: [1]nd.Drawn = undefined;
            _ = try nd.levels(&prows, sh, &pos, s, work, &plain);
            tally.unmasked_disallowed += @intFromBool(!mask_cpu.eligible(mask, plain[0].token));
        }
    }
}

pub fn sweep() !void {
    var tally: Tally = .{};
    sweepOf(4099, 125, 96, 0x4752_4133, &tally) catch |err| {
        std.debug.print("structured_users sampled sweep: error.{s} after {d} draws\n", .{ @errorName(err), tally.draws });
        return err;
    };
    try t.expectEqual(@as(usize, 12_000), tally.draws);
    try t.expectEqual(@as(usize, 0), tally.disallowed);
    try t.expectEqual(@as(usize, 0), tally.reference_differs); // the allowed tokens' own draw, token and share
    try t.expect(tally.unmasked_disallowed > 1000); // the same keyed draws unmasked: the mask is what keeps them out
    try t.expect(tally.gathers[0] > 0 and tally.gathers[1] > 0); // NUCLEUS-covered draws and whole-shard draws both ran (V <= WIDER: whole shards are the second gather)
    try t.expect(tally.top_k > 0);
    std.debug.print("structured_users sampled sweep: V 4099, {d} masked draws ({d} rows with top_k), {d} disallowed, {d} differ from the allowed-only reference draw; unmasked the same keyed draws pick {d} disallowed; gathers (1, 2, 3) {any}\n", .{ tally.draws, tally.top_k, tally.disallowed, tally.reference_differs, tally.unmasked_disallowed, tally.gathers });
}

pub fn glmVocab() !void {
    var tally: Tally = .{};
    sweepOf(154_880, 12, 40, 0x474c_4d35, &tally) catch |err| {
        std.debug.print("structured_users sampled GLM vocabulary: error.{s} after {d} draws\n", .{ @errorName(err), tally.draws });
        return err;
    };
    try t.expectEqual(@as(usize, 480), tally.draws);
    try t.expectEqual(@as(usize, 0), tally.disallowed);
    try t.expectEqual(@as(usize, 0), tally.reference_differs);
    try t.expect(tally.unmasked_disallowed > 0);
    std.debug.print("structured_users sampled GLM vocabulary: V 154880, {d} masked draws ({d} rows with top_k), {d} disallowed, {d} differ from the allowed-only reference draw; unmasked {d} disallowed; gathers (1, 2, 3) {any}\n", .{ tally.draws, tally.top_k, tally.disallowed, tally.reference_differs, tally.unmasked_disallowed, tally.gathers });
}

/// The recorder's picker for sampled draws: the positions it was asked at and a scripted token (`from_dev`: the lowest
/// eligible id of the adapter's STAGED words, as a draw on a masked row lands).
const Answer = struct {
    token: u32 = 0,
    from_dev: ?*const md.Dev = null,
    positions: [16]u64 = @splat(0),
    asked: usize = 0,
    fn picker(x: *Answer) glm.trace.Picker {
        return .{ .ctx = x, .pick = pick };
    }
    fn pick(p: *anyopaque, _: glm.trace.T, positions: ?[]const u64, out: []u32, probs: ?[]f64) !void {
        const x: *Answer = @ptrCast(@alignCast(p));
        if (positions) |ps| for (ps) |q| {
            if (x.asked < x.positions.len) x.positions[x.asked] = q;
            x.asked += 1;
        };
        var token = x.token;
        if (x.from_dev) |d| {
            for (d.host[0..d.words], 0..) |w, i| if (w != 0) {
                token = @intCast(i * 32 + @ctz(w));
                break;
            };
        }
        @memset(out, token);
        if (probs) |q| @memset(q, 0);
    }
};

fn names(tr: *glm.trace.Trace, out: *std.ArrayList([]const u8)) !void {
    out.clearRetainingCapacity();
    for (tr.events.items) |e| try out.append(gpa, e.name);
}

pub fn recorded() !void {
    var m = try glm.model.Model.from(&glm.Config{}, .{}, 1, 0);
    m.layers = 3;
    m.mtp_layers = 0;
    var w = try glm.views.rank(gpa, &m);
    defer w.deinit(gpa);
    const e = try glm.engine.Engine.create(gpa, &m, &w, 2560, 8, 64, false);
    defer e.destroy(gpa);
    var tr = glm.trace.Trace.init(gpa);
    defer tr.deinit();
    var f = glm.fwd.Fwd.recording(&m, &tr);
    try e.useSampling(gpa);
    var space: glm.trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try e.walk(&space, glm.trace.placeAll);
    try e.walkTmp(&space, glm.trace.placeAll);
    e.tmp.space = &space;
    var dev = try md.Dev.recorded(gpa, &m, &space);
    defer dev.close();
    e.smp.masked_dev = &dev;
    var ans: Answer = .{};
    tr.picker = ans.picker();
    e.sampling = .{ .seed = 9, .temperature = 0.8, .top_p = 0.9 };
    const V = m.vl();
    const logits = glm.trace.T.of(e.buf.logits, 0, &.{ 1, V });
    const mask = try gpa.alloc(u32, (V + 31) / 32);
    defer gpa.free(mask);
    @memset(mask, 0);
    set(mask, 77);
    set(mask, 1234);
    var seen: std.ArrayList([]const u8) = .empty;
    defer seen.deinit(gpa);
    var plain: std.ArrayList([]const u8) = .empty;
    defer plain.deinit(gpa);
    var out: [1]u32 = .{0};
    // the plain sampled draw's launches (world 1: no gathers between ranks)
    tr.reset();
    try e.sampleRows(&f, logits, 100, &out, null);
    try f.flush();
    try names(&tr, &plain);
    // masked: the staging and grammar_mask between the widening and the temperature, the rest unchanged
    ans.token = 1234;
    tr.reset();
    try e.sampleMasked(&f, logits, 100, &out, mask);
    try f.flush();
    try names(&tr, &seen);
    const inserted = [_][]const u8{ "grammar.staged.synchronize", "grammar.host.write", "torch.copy_", "grammar.staged.record", "torch.fill_", "grammar.mask", "grammar.invalid.cpu" };
    try t.expectEqualStrings("torch.float", plain.items[0]);
    try t.expectEqualStrings("nucleus.scale", plain.items[1]);
    try t.expectEqual(plain.items.len + inserted.len, seen.items.len);
    try t.expectEqualStrings("torch.float", seen.items[0]);
    for (inserted, seen.items[1..][0..inserted.len]) |want, got| try t.expectEqualStrings(want, got);
    for (plain.items[1..], seen.items[1 + inserted.len ..]) |want, got| try t.expectEqualStrings(want, got);
    try t.expectEqual(@as(u32, 1234), out[0]);
    try t.expectEqual(@as(u64, 100), ans.positions[ans.asked - 1]);
    try t.expectEqual(@as(u64, 1), dev.stats.sampled);
    // every token allowed, or no mask: nucleus_rows itself (the same events as the plain draw)
    for ([_]bool{ true, false }) |all| {
        const words = try gpa.alloc(u32, if (all) mask.len else 0);
        defer gpa.free(words);
        @memset(words, 0xffff_ffff);
        if (all and V % 32 != 0) words[words.len - 1] = (@as(u32, 1) << @as(u5, @intCast(V % 32))) - 1;
        tr.reset();
        try e.sampleMasked(&f, logits, 100, &out, words);
        try f.flush();
        try names(&tr, &seen);
        try t.expectEqual(plain.items.len, seen.items.len);
        for (plain.items, seen.items) |want, got| try t.expectEqualStrings(want, got);
    }
    try t.expectEqual(@as(u64, 1), dev.stats.identity);
    try t.expectEqual(@as(u64, 1), dev.stats.plain);
    // an empty mask (a dead end) is refused before any launch
    @memset(mask, 0);
    tr.reset();
    try t.expectError(error.EmptyMask, e.sampleMasked(&f, logits, 100, &out, mask));
    try f.flush();
    try t.expectEqual(@as(usize, 0), tr.events.items.len);
    // a draw outside the words (a device/host disagreement) is refused, never returned, and poisons the adapter
    set(mask, 77);
    ans.token = 5;
    out[0] = 0;
    try t.expectError(error.MaskInconsistent, e.sampleMasked(&f, logits, 101, &out, mask));
    try t.expect(dev.poisoned);
    try t.expectError(error.ControlPoison, e.sampleMasked(&f, logits, 102, &out, mask));
    // the host check itself
    try t.expectError(error.MaskInconsistent, md.drawn(V, mask, 5));
    try t.expectError(error.BadTokenId, md.drawn(V, mask, @intCast(V)));
    try t.expectEqual(@as(u32, 77), try md.drawn(V, mask, 77));
    std.debug.print("structured_users sampled recorded: {d} plain launches, the masked draw {d} (grammar_mask between torch.float and nucleus.scale); all-allowed and unmasked equal plain; empty refused with no launch; a disallowed answer refused (MaskInconsistent) and poisons\n", .{ plain.items.len, plain.items.len + inserted.len });
}

/// A sampled structured stream (TF_GLM_SAMPLING=1's Bound) through the actual AFD chain on the recorder: its first token
/// and each step drawn among its script's tokens, keyed at the plain sampled path's positions (the prompt's length,
/// then each fed token's next); the matcher closed once.
pub fn chain(comptime Bound: type) !void {
    chainOf(Bound) catch |err| {
        std.debug.print("structured_users sampled chain: error.{s}\n", .{@errorName(err)});
        return err;
    };
}

fn check(what: []const u8, ok: bool) !void {
    if (ok) return;
    std.debug.print("structured_users sampled chain: not {s}\n", .{what});
    return error.TestUnexpectedResult;
}

fn chainOf(comptime Bound: type) !void {
    const C = fx.Chain(Bound);
    var c: C = undefined;
    try c.init(true);
    defer c.deinit();
    try c.rec.e.useSampling(gpa);
    c.rec.e.sampled_top_k = true; // as afd_attn_gpu.zig's run with TF_GLM_SAMPLING=1: top_k through the nucleus
    try glm.views.walk(&c.rec.e.nuc.?.s, glm.views.Path.init("tmp.nucleus"), &c.rec.space, glm.trace.placeAll);
    c.bound.sampled = true;
    var ans: Answer = .{ .from_dev = &c.dev };
    c.rec.t.picker = ans.picker();
    var script: fx.Script = .{ .steps = &.{ &.{ 300, 7 }, &.{ 8, 900 }, &.{9} } };
    script.init(c.rec.m.vocab);
    var ids: [21]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i);
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &ids, .max_new = 8, .drafts = false, .sampling = .{ .seed = 11, .temperature = 0.7, .top_k = 20, .top_p = 0.9 }, .rules = &script.rules });
    defer s.deinit(gpa);
    try c.drive(&s);
    try t.expectEqualSlices(u32, &.{ 7, 8, 9 }, s.emitted());
    try t.expectEqual(lanes.stream.Reason.stop, s.reason);
    try t.expectEqual(@as(usize, 3), ans.asked);
    try t.expectEqualSlices(u64, &.{ 21, 22, 23 }, ans.positions[0..3]);
    try t.expectEqual(@as(u64, 3), c.dev.stats.sampled);
    try t.expectEqual(@as(u64, 0), c.dev.stats.masked); // no greedy masked draw: every draw sampled
    try check("released", c.back.owner == null);
    try check("closed", !c.serve.open);
    try check("without deferred logits", c.rec.at.deferred_logits == null);
    try check("without a first pending", !c.back.first_pending);
    // the plain sampled path's own positions for the same prompt
    var plain_ans: Answer = .{ .token = 40 };
    c.rec.t.picker = plain_ans.picker();
    var p = try lanes.Stream.init(gpa, .{ .id = "p", .prompt = &ids, .max_new = 3, .drafts = false, .sampling = .{ .seed = 11, .temperature = 0.7, .top_k = 20, .top_p = 0.9 } });
    defer p.deinit(gpa);
    try c.drive(&p);
    try t.expectEqual(@as(usize, 3), plain_ans.asked);
    try t.expectEqualSlices(u64, ans.positions[0..3], plain_ans.positions[0..3]);
    std.debug.print("structured_users sampled chain: structured sampled {any} (stop) at positions {any} = the plain sampled path's; {d} sampled masked draws\n", .{ s.emitted(), ans.positions[0..3], c.dev.stats.sampled });
}
