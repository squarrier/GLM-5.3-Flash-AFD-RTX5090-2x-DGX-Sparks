//! qualification_45c10bcf's CPU gate: Zig's host draw (nucleus_draw.zig, Mia's union cover always on) on python's fixtures
//! (tools/zig/glm_nucleus_fixtures.py: python-0.6 ed78d6f's `_draw`, unpatched and with patch 0034), each case's
//! logits and masses remade here bit for bit (splitmix64, bf16 rounding, `texp`); every rank's top candidates are
//! taken as the GPU's topk may take them (ties to the higher id, in no order) and packed as cuda_nucleus.zig packs
//! them. Tokens and shares must be the stock draw's, and the gathers the union path's. Then 0092's checks, and the
//! sampled Engine's launches recorded on a CPU.
const std = @import("std");
const nd = @import("nucleus_draw.zig");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const nucleus = @import("cuda_nucleus.zig");
const sample = @import("cuda_sample.zig");
const eng = @import("cuda_engine.zig");
const ct = @import("cuda_compute_test.zig");

const testing = std.testing;
const J = std.json.Value;
const get = ct.get;
const int = ct.int;
const fixture = @embedFile("fixtures_cuda_nucleus.json");
extern "c" fn exp(x: f64) f64;

// Pearson gates use qualification_45c10bcf oracle logits, but an independent full-row softmax reference.
test "sampling_tests chi-square: 10k draws on qualification_45c10bcf oracle row at three temperature/top-p settings" {
    const gpa = testing.allocator;
    const doc = try std.json.parseFromSlice(J, gpa, fixture, .{});
    defer doc.deinit();
    var c = get(doc.value, "cases").array.items[0];
    const V = int(get(c, "vocab"));
    const settings = [_][2]f64{ .{ 0.6, 0.6 }, .{ 1.0, 0.8 }, .{ 1.3, 0.9 } };
    for (settings, 0..) |setting, row| {
        try c.object.put(gpa, "temperature", .{ .float = setting[0] });
        const all = try Rows.make(gpa, c);
        defer all.deinit(gpa);
        const data: Rows = .{ .x = all.x[row * V ..][0..V], .scaled = all.scaled[row * V ..][0..V], .mass = all.mass[row * V ..][0..V] };
        const sh = nd.Shards.uniform(1, V);
        var emu = try Emu.init(gpa, sh, &data, 1, V);
        defer emu.deinit();
        const work = try gpa.alloc(nd.Cand, V);
        defer gpa.free(work);
        var p = try emu.gather(@min(nd.nucleus, V));
        const sample_params: nd.Sampling = .{ .seed = 0, .temperature = setting[0], .top_p = setting[1] };
        if ((try nd.drawRow(p, sh, 0, 37, sample_params, work)) == null) p = try emu.gather(V);
        const order = try gpa.alloc(u32, V);
        defer gpa.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, ByValue{ .scaled = data.scaled[0..V] }, struct {
            fn less(ctx: ByValue, a: u32, b: u32) bool {
                return ctx.scaled[a] > ctx.scaled[b] or (ctx.scaled[a] == ctx.scaled[b] and a < b);
            }
        }.less);
        const probs = try gpa.alloc(f64, V);
        defer gpa.free(probs);
        var total: f64 = 0;
        for (probs, 0..) |*pr, id| {
            pr.* = exp(data.scaled[id] - data.scaled[order[0]]);
            total += pr.*;
        }
        var mass: f64 = 0;
        var keep: usize = 0;
        while (keep < V and mass < setting[1] * total) : (keep += 1) mass += probs[order[keep]];
        const wanted = try gpa.alloc(f64, V);
        defer gpa.free(wanted);
        @memset(wanted, 0);
        for (order[0..keep]) |id| wanted[id] = probs[id] / mass;
        const counts = try gpa.alloc(u32, V);
        defer gpa.free(counts);
        @memset(counts, 0);
        for (0..10000) |seed| {
            var params = sample_params;
            params.seed = seed;
            const got = (try nd.drawRow(p, sh, 0, 37, params, work)).?;
            const repeat = (try nd.drawRow(p, sh, 0, 37, params, work)).?;
            try testing.expectEqual(got.token, repeat.token);
            try testing.expect(wanted[got.token] > 0);
            counts[got.token] += 1;
        }
        var chi: f64 = 0;
        var bins: usize = 0;
        var tail_e: f64 = 0;
        var tail_o: u64 = 0;
        for (wanted, counts) |pr, count| {
            const e = pr * 10000;
            if (e >= 5) {
                const delta = @as(f64, @floatFromInt(count)) - e;
                chi += delta * delta / e;
                bins += 1;
            } else {
                tail_e += e;
                tail_o += count;
            }
        }
        if (tail_e > 0) {
            try testing.expect(tail_e >= 5);
            const delta = @as(f64, @floatFromInt(tail_o)) - tail_e;
            chi += delta * delta / tail_e;
            bins += 1;
        }
        // A conservative deterministic bound; exact survival probabilities are checked by the evidence script.
        const df: f64 = @floatFromInt(bins - 1);
        try testing.expect(chi < df + 6 * @sqrt(2 * df) + 20);
        std.debug.print("sampling_tests_CHI {{\"temperature\":{d},\"top_p\":{d},\"draws\":10000,\"oracle_case\":0,\"row\":{d},\"candidates\":{d},\"keep\":{d},\"chi2\":{d},\"df\":{d}}}\n", .{ setting[0], setting[1], row, p.count, keep, chi, bins - 1 });
    }
}

// -- the fixture's rows, as glm_nucleus_fixtures.py makes them ---------------------------------------------------

const golden: u64 = 0x9E3779B97F4A7C15;
const step: u64 = 0xD1B54A32D192ED03;
const ln2_hi: f64 = 0x1.62e42fee00000p-1;
const ln2_lo: f64 = 0x1.a39ef35793c76p-33;
const inv_ln2: f64 = 0x1.71547652b82fep+0;
const taylor = [12]f64{ 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p-1, 0x1.5555555555555p-3, 0x1.5555555555555p-5, 0x1.1111111111111p-7, 0x1.6c16c16c16c17p-10, 0x1.a01a01a01a01ap-13, 0x1.a01a01a01a01ap-16, 0x1.71de3a556c734p-19, 0x1.27e4fb7789f5cp-22, 0x1.ae64567f544e4p-26 };
const cutoff: f64 = -45.0;

fn words(seed: u64, stream: u64, out: []u64) void {
    const key = nd.mix(seed *% golden +% stream);
    for (out, 0..) |*w, j| w.* = nd.mix(key +% @as(u64, j) *% step);
}

const Keyed = struct { w: u64, id: u32 };

fn byWord(_: void, a: Keyed, b: Keyed) bool {
    return a.w < b.w or (a.w == b.w and a.id < b.id);
}

/// pick's order: the vocabulary by (word, id); its first k are pick(seed, stream, k).
fn pick(gpa: std.mem.Allocator, seed: u64, stream: u64, vocab: usize) ![]u32 {
    const w = try gpa.alloc(u64, vocab);
    defer gpa.free(w);
    words(seed, stream, w);
    const k = try gpa.alloc(Keyed, vocab);
    defer gpa.free(k);
    for (k, w, 0..) |*x, y, i| x.* = .{ .w = y, .id = @intCast(i) };
    std.mem.sort(Keyed, k, {}, byWord);
    const out = try gpa.alloc(u32, vocab);
    for (out, k) |*o, x| o.* = x.id;
    return out;
}

fn lg(n: u64) f64 {
    const e: u6 = @intCast(63 - @clz(n));
    return @as(f64, @floatFromInt(e)) + @as(f64, @floatFromInt(((n << 5) >> e) & 31)) / 32.0;
}

fn texp(d: f64) f64 {
    const k = @floor(d * inv_ln2 + 0.5);
    const r = (d - k * ln2_hi) - k * ln2_lo;
    var p = taylor[11];
    var i: usize = 11;
    while (i > 0) {
        i -= 1;
        p = p * r + taylor[i];
    }
    return std.math.ldexp(p, @as(i32, @intFromFloat(k)));
}

/// A case's rows: bf16 logits as fp32, then f64 scaled rows and fixed-point masses (texp for torch.exp).
const Rows = struct {
    x: []f32,
    scaled: []f64,
    mass: []i64,

    fn make(gpa: std.mem.Allocator, c: J) !Rows {
        const seed: u64 = @intCast(get(c, "seed").integer);
        const kind = get(c, "kind").string;
        const R = int(get(c, "rows"));
        const V = int(get(c, "vocab"));
        const x = try gpa.alloc(f32, R * V);
        errdefer gpa.free(x);
        const z = try gpa.alloc(f64, V);
        defer gpa.free(z);
        const w = try gpa.alloc(u64, V);
        defer gpa.free(w);
        const row = try gpa.alloc(f64, V);
        defer gpa.free(row);
        for (0..R) |r| {
            words(seed, 10 + r, w);
            for (z, w) |*o, y| {
                var f: i64 = -131070;
                inline for (.{ 0, 16, 32, 48 }) |s| f += @intCast((y >> s) & 0xFFFF);
                o.* = @as(f64, @floatFromInt(f)) * 0x1p-15;
            }
            try fill(gpa, row, z, kind, seed, r);
            for (row, x[r * V ..][0..V]) |v, *o| o.* = bf16(@floatCast(v));
        }
        const t: f64 = 1.0 / @max(num(get(c, "temperature")), 1e-6);
        const scaled = try gpa.alloc(f64, R * V);
        errdefer gpa.free(scaled);
        const mass = try gpa.alloc(i64, R * V);
        for (0..R) |r| {
            const s = scaled[r * V ..][0..V];
            for (s, x[r * V ..][0..V]) |*o, v| o.* = @as(f64, v) * t;
            var top = -std.math.inf(f64);
            for (s) |v| top = @max(top, v);
            for (s, mass[r * V ..][0..V]) |v, *m| {
                const d = v - top;
                m.* = if (d < cutoff) 0 else @intFromFloat(@floor(texp(d) * 0x1p40));
            }
        }
        return .{ .x = x, .scaled = scaled, .mass = mass };
    }

    fn deinit(r: Rows, gpa: std.mem.Allocator) void {
        gpa.free(r.x);
        gpa.free(r.scaled);
        gpa.free(r.mass);
    }

    /// sha256 of the fp32 rows then the masses (little-endian), its first 8 bytes in hex: the fixture's `digest`.
    fn digest(r: Rows) [16]u8 {
        var h: std.crypto.hash.sha2.Sha256 = .init(.{});
        h.update(std.mem.sliceAsBytes(r.x));
        h.update(std.mem.sliceAsBytes(r.mass));
        var d: [32]u8 = undefined;
        h.final(&d);
        return std.fmt.bytesToHex(d[0..8].*, .lower);
    }
};

fn fill(gpa: std.mem.Allocator, row: []f64, z: []const f64, kind: []const u8, seed: u64, r: usize) !void {
    const V = row.len;
    const eq = std.mem.eql;
    if (eq(u8, kind, "peaked")) {
        @memcpy(row, z);
        for (row[0..40]) |*v| v.* += 12.0;
    } else if (eq(u8, kind, "normal")) {
        for (row, z) |*v, y| v.* = 4.0 * y;
    } else if (eq(u8, kind, "flat")) {
        for (row, z) |*v, y| v.* = 0.25 * y;
    } else if (eq(u8, kind, "gflat")) {
        for (row, z) |*v, y| v.* = y * 0x1p-10;
    } else if (eq(u8, kind, "tied") or eq(u8, kind, "zipf")) {
        @memset(row, 0.0);
        const tied = eq(u8, kind, "tied");
        const order = try pick(gpa, seed, if (tied) 99 else 300 + r, V);
        defer gpa.free(order);
        if (tied) {
            for (order[0..8]) |i| row[i] = 2.0;
        } else for (order, 1..) |i, n| {
            row[i] = -0.75 * lg(n);
        }
    } else {
        const plus: struct { stream: u64, k: usize, add: f64 } = if (eq(u8, kind, "broad")) .{ .stream = 200 + r, .k = 4000, .add = 7.0 } else if (eq(u8, kind, "gpeaked")) .{ .stream = 98, .k = 5, .add = 14.0 } else if (eq(u8, kind, "gbroad")) .{ .stream = 400 + r, .k = 4000, .add = 9.0 } else return error.Kind;
        @memcpy(row, z);
        const order = try pick(gpa, seed, plus.stream, V);
        defer gpa.free(order);
        for (order[0..plus.k]) |i| row[i] += plus.add;
    }
}

/// round to nearest even at bf16's 8 bits (finite values)
fn bf16(v: f32) f32 {
    const b: u32 = @bitCast(v);
    return @bitCast((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000);
}

fn num(v: J) f64 {
    return switch (v) {
        .float => |x| x,
        .integer => |i| @floatFromInt(i),
        else => std.debug.panic("fixture: not a number", .{}),
    };
}

// -- the ranks' packets, as the GPU may make them ------------------------------------------------------------------

const ByValue = struct {
    scaled: []const f64,
    fn high(ctx: ByValue, a: u32, b: u32) bool { // value down, ties to the higher id: not python's tie order
        return ctx.scaled[a] > ctx.scaled[b] or (ctx.scaled[a] == ctx.scaled[b] and a > b);
    }
};

/// nucleus_draw.levels' source over a case's rows: each rank's shard ordered once a row; a level's packet holds the
/// top `count` of each, written last to first, padding (-inf, -1, 0) past a narrower shard's width.
const Emu = struct {
    gpa: std.mem.Allocator,
    sh: nd.Shards,
    rows: usize,
    vocab: usize,
    data: *const Rows,
    order: []u32,
    sums: []u64,
    buf: []u64,
    gathers: usize = 0,
    /// 0092's tests: an id this far outside its shard in the first copy (and in the second too with `stuck`), or a
    /// rank's header set (a kernel saw a bad index: every copy shows it)
    corrupt: i64 = 0,
    stuck: bool = false,
    bad_header: bool = false,
    agains: usize = 0,
    last: nd.Packet = undefined,

    fn init(gpa: std.mem.Allocator, sh: nd.Shards, data: *const Rows, rows: usize, vocab: usize) !Emu {
        const order = try gpa.alloc(u32, rows * vocab);
        errdefer gpa.free(order);
        const sums = try gpa.alloc(u64, sh.world * rows);
        errdefer gpa.free(sums);
        for (0..rows) |r| for (0..sh.world) |k| {
            const lo = sh.cut[k];
            const part = order[r * vocab + lo ..][0..sh.width(k)];
            for (part, 0..) |*o, j| o.* = @intCast(lo + j);
            std.mem.sort(u32, part, ByValue{ .scaled = data.scaled[r * vocab ..][0..vocab] }, ByValue.high);
            var s: u64 = 0;
            for (data.mass[r * vocab + lo ..][0..sh.width(k)]) |m| s += @intCast(m);
            sums[k * rows + r] = s;
        };
        const buf = try gpa.alloc(u64, sh.world * nd.Packet.stride(rows, sh.widest()));
        return .{ .gpa = gpa, .sh = sh, .rows = rows, .vocab = vocab, .data = data, .order = order, .sums = sums, .buf = buf };
    }

    fn deinit(e: *Emu) void {
        e.gpa.free(e.order);
        e.gpa.free(e.sums);
        e.gpa.free(e.buf);
    }

    pub fn gather(e: *Emu, count: usize) !nd.Packet {
        e.gathers += 1;
        const R = e.rows;
        const stride = nd.Packet.stride(R, count);
        for (0..e.sh.world) |k| {
            const w = e.buf[k * stride ..][0..stride];
            w[0] = 0;
            const real = @min(count, e.sh.width(k));
            for (0..R) |r| {
                w[1 + r] = e.sums[k * R + r];
                const top = e.order[r * e.vocab + e.sh.cut[k] ..][0..real];
                for (0..count) |j| {
                    const at = 1 + R + r * count + j;
                    if (j < real) {
                        const i = top[real - 1 - j];
                        w[at] = @bitCast(e.data.scaled[r * e.vocab + i]);
                        w[at + R * count] = i;
                        w[at + 2 * R * count] = @bitCast(e.data.mass[r * e.vocab + i]);
                    } else {
                        w[at] = @bitCast(-std.math.inf(f64));
                        w[at + R * count] = @bitCast(@as(i64, -1));
                        w[at + 2 * R * count] = 0;
                    }
                }
            }
        }
        e.last = .{ .words = e.buf[0 .. e.sh.world * stride], .world = e.sh.world, .rows = R, .count = count };
        if (e.corrupt != 0) e.flip(e.corrupt);
        if (e.bad_header) e.buf[stride] = 1; // rank 1's
        return e.last;
    }

    /// the first rank's first id moved by `by` (a float's bits read as an id, in 0092's issue)
    fn flip(e: *Emu, by: i64) void {
        const at = 1 + e.rows;
        const p = &e.buf[at + e.rows * e.last.count];
        p.* = @bitCast(@as(i64, @bitCast(p.*)) + by);
    }

    pub fn again(e: *Emu) !nd.Packet {
        e.agains += 1;
        if (e.corrupt != 0 and !e.stuck) e.flip(-e.corrupt);
        return e.last;
    }
};

fn positionsOf(c: J, out: []u64) []u64 {
    const xs = get(c, "positions").array.items;
    for (xs, 0..) |x, i| out[i] = @intCast(x.integer);
    return out[0..xs.len];
}

fn samplingOf(c: J) nd.Sampling {
    return .{ .seed = @intCast(get(c, "sampling_seed").integer), .temperature = num(get(c, "temperature")), .top_k = 0, .top_p = num(get(c, "top_p")), .min_p = num(get(c, "min_p")) };
}

fn shardsOf(c: J) nd.Shards {
    var cuts: [nd.max_world + 1]u64 = undefined;
    const xs = get(c, "cuts").array.items;
    for (xs, 0..) |x, i| cuts[i] = @intCast(x.integer);
    return nd.Shards.of(cuts[0..xs.len]);
}

test "the host draw is python-0.6's stock _draw on every fixture row; the gathers are 0034's" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-nucleus-draw/1", get(fx, "schema").string);
    try testing.expectEqual(@as(usize, nd.nucleus), int(get(fx, "nucleus")));
    try testing.expectEqual(@as(usize, nd.wider), int(get(fx, "wider")));
    var tally = [3]usize{ 0, 0, 0 };
    var draws: usize = 0;
    const cases = get(fx, "cases").array.items;
    for (cases, 0..) |c, ci| {
        errdefer std.debug.print("case {d} ({s})\n", .{ ci, get(c, "kind").string });
        const data = try Rows.make(gpa, c);
        defer data.deinit(gpa);
        try testing.expectEqualStrings(get(c, "digest").string, &data.digest());
        const sh = shardsOf(c);
        const R = int(get(c, "rows"));
        var emu = try Emu.init(gpa, sh, &data, R, int(get(c, "vocab")));
        defer emu.deinit();
        const work = try gpa.alloc(nd.Cand, int(get(c, "vocab")));
        defer gpa.free(work);
        var pos: [8]u64 = undefined;
        var out: [8]nd.Drawn = undefined;
        const n = try nd.levels(&emu, sh, positionsOf(c, &pos), samplingOf(c), work, out[0..R]);
        try testing.expectEqual(int(get(c, "gathers_union")), n);
        try testing.expectEqual(n, emu.gathers);
        for (out[0..R], get(c, "tokens").array.items, get(c, "shares").array.items) |d, t, s| {
            try testing.expectEqual(@as(u32, @intCast(t.integer)), d.token);
            try testing.expectEqual(@as(u64, @intCast(s.integer)), @as(u64, @bitCast(d.share)));
        }
        tally[n - 1] += 1;
        draws += R;
    }
    const want = get(get(fx, "gathers"), "union").array.items;
    for (tally, want) |t, w| try testing.expectEqual(int(w), t);
    try testing.expectEqual(@as(usize, 152), cases.len);
    try testing.expectEqual(@as(usize, 444), draws);
    std.debug.print("nucleus fixture: {d} cases, {d} draws equal to python-0.6's stock _draw (tokens, share bits); gathers (1, 2, 3) {any}, 0034's\n", .{ cases.len, draws, tally });
}

test "0092: a bad id is copied again once after a device sync and the draw goes on; still bad, the request fails" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture, .{});
    defer parsed.deinit();
    const c = get(parsed.value, "cases").array.items[0];
    const data = try Rows.make(gpa, c);
    defer data.deinit(gpa);
    const sh = shardsOf(c);
    const R = int(get(c, "rows"));
    const V = int(get(c, "vocab"));
    const s = samplingOf(c);
    const work = try gpa.alloc(nd.Cand, V);
    defer gpa.free(work);
    var pos: [8]u64 = undefined;
    const positions = positionsOf(c, &pos);
    var want: [8]nd.Drawn = undefined;
    var got: [8]nd.Drawn = undefined;
    var clean = try Emu.init(gpa, sh, &data, R, V);
    defer clean.deinit();
    const n = try nd.levels(&clean, sh, positions, s, work, want[0..R]);
    try testing.expectEqual(@as(usize, 0), clean.agains);
    // far outside the vocabulary, negative, and inside the next rank's shard (a valid id, from the wrong rank)
    for ([_]i64{ 1 << 33, -(1 << 40), @intCast(sh.cut[1]) }) |by| {
        var bad = try Emu.init(gpa, sh, &data, R, V);
        defer bad.deinit();
        bad.corrupt = by;
        try testing.expectEqual(n, try nd.levels(&bad, sh, positions, s, work, got[0..R]));
        try testing.expectEqual(n, bad.agains); // each level's first copy bad, its second good
        for (want[0..R], got[0..R]) |a, b| try testing.expectEqual(a, b);
        bad.stuck = true;
        bad.agains = 0;
        try testing.expectError(error.BadTokenId, nd.levels(&bad, sh, positions, s, work, got[0..R]));
        try testing.expectEqual(@as(usize, 1), bad.agains);
    }
    var hdr = try Emu.init(gpa, sh, &data, R, V);
    defer hdr.deinit();
    hdr.bad_header = true;
    try testing.expectError(error.BadTokenId, nd.levels(&hdr, sh, positions, s, work, got[0..R]));
    try testing.expectEqual(@as(usize, 1), hdr.agains);
}

fn bits(x: f64) u64 {
    return @bitCast(x);
}

test "0092's check: ids inside the sending rank's shard, padding only past its real candidates, no negative mass or sum" {
    // two ranks, shards [0, 3) and [3, 5), one row of 3 candidates a rank: rank 1 sends its 2 and a padding slot
    const sh = nd.Shards.of(&.{ 0, 3, 5 });
    const S = comptime nd.Packet.stride(1, 3);
    const none: u64 = @bitCast(@as(i64, -1));
    const good = [2 * S]u64{
        0, 10, bits(1.0), bits(0.5), bits(0.25),               0, 1, 2,    5, 3, 2,
        0, 7,  bits(2.0), bits(1.5), bits(-std.math.inf(f64)), 3, 4, none, 4, 3, 0,
    };
    try testing.expect(nd.check(.{ .words = &good, .world = 2, .rows = 1, .count = 3 }, sh));
    try testing.expect(!nd.check(.{ .words = good[0..S], .world = 1, .rows = 1, .count = 3 }, sh));
    const edits = [_]struct { at: usize, v: u64 }{
        .{ .at = 0, .v = 1 }, // rank 0's header: a kernel saw a bad index
        .{ .at = 7, .v = 3 }, // rank 0 sends rank 1's id
        .{ .at = S + 7, .v = 4 }, // a real id in rank 1's padding slot
        .{ .at = S + 5, .v = none }, // padding among rank 1's real candidates
        .{ .at = 8, .v = none }, // a negative mass
        .{ .at = S + 1, .v = none }, // a negative shard sum
    };
    for (edits) |x| {
        var w = good;
        w[x.at] = x.v;
        try testing.expect(!nd.check(.{ .words = &w, .world = 2, .rows = 1, .count = 3 }, sh));
    }
}

test "0092 on the greedy path's gathered pairs: every id inside the sending rank's vocabulary rows" {
    const v: u32 = @bitCast(@as(f32, 1.0));
    const neg: u32 = @bitCast(@as(i32, -1));
    var two = [_]u32{ v, v, 1, 9, v, v, 10, 19 }; // [world 2, R 1, 2k]: k values, then k ids, a rank of 10 rows
    try testing.expect(sample.idsInShards(2, 10, &two, 1, 2));
    for ([_][2]u32{ .{ 7, 20 }, .{ 2, 10 }, .{ 2, neg }, .{ 6, 9 } }) |x| {
        var w = two;
        w[x[0]] = x[1];
        try testing.expect(!sample.idsInShards(2, 10, &w, 1, 2));
    }
    var one = [_]u32{ v, v, v, v, 0, 5, 9, 3 }; // one rank: values [R 2, k 2], then ids [2, 2]
    try testing.expect(sample.idsInShards(1, 10, &one, 2, 2));
    one[6] = 10;
    try testing.expect(!sample.idsInShards(1, 10, &one, 2, 2));
}

const E = eng.EngineOf(ct.Marks);

test "a sampled Engine records nucleus_rows' launches on a TP2 rank; temperature 0 stays greedy; top_k is refused" {
    const gpa = testing.allocator;
    var m = try md.Model.from(&@import("config.zig").Config{}, .{}, 2, 0);
    m.layers = 3;
    m.mtp_layers = 0;
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    const e = try E.create(gpa, &m, &w, 2560, 8, 64, false);
    defer e.destroy(gpa);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var f = fwd.Fwd.recording(&m, &t);
    const V = m.vl();
    var out: [9]u32 = @splat(7);
    e.sampling = .{ .seed = 5, .temperature = 1.0, .top_p = 0.95 };
    try testing.expectError(error.NoNucleus, e.sampleRows(&f, trace.T.of(e.buf.logits, 0, &.{ 4, V }), 100, out[0..4], null));
    try e.useSampling(gpa);
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try e.walk(&space, trace.placeAll);
    try e.walkTmp(&space, trace.placeAll);
    e.tmp.space = &space;
    t.reset();
    try e.sampleRows(&f, trace.T.of(e.buf.logits, 0, &.{ 4, V }), 100, out[0..4], null); // a verify window of 4 rows
    try f.flush();
    const want = [_][]const u8{ "torch.float", "nucleus.scale", "nucleus.max", "all_gather", "nucleus.flip", "nucleus.max", "nucleus.mass", "nucleus.sum", "torch.topk", "torch.add", "nucleus.take", "nucleus.take", "nucleus.header", "all_gather", "torch.cpu" };
    try testing.expectEqual(want.len, t.events.items.len);
    for (want, t.events.items) |a, b| try testing.expectEqualStrings(a, b.name);
    const ev = t.events.items;
    try testing.expectEqual(@as(i64, nd.nucleus), ev[8].vals[1].int); // the first level: NUCLEUS candidates
    try testing.expectEqual(2 * nd.Packet.stride(4, nd.nucleus), ev[13].vals[0].t.shape[0]); // one packed gather
    try testing.expectEqual(2 * 4, ev[3].vals[0].t.shape[0]); // the ranks' maxima
    for (out[0..4]) |x| try testing.expectEqual(@as(u32, 0), x); // recording draws nothing: the trace answers 0 ...
    for (out[4..]) |x| try testing.expectEqual(@as(u32, 7), x);
    try testing.expectEqual(@as(u64, 0), e.stats.rows); // ... and counts nothing
    t.reset();
    try e.sampleRows(&f, trace.T.of(e.buf.logits, 0, &.{ 1, V }), 100, out[0..1], null); // one row: no transpose
    try f.flush();
    for (t.events.items) |x| try testing.expect(!std.mem.eql(u8, x.name, "nucleus.flip"));
    try testing.expectError(error.SampleShape, e.sampleRows(&f, trace.T.of(e.buf.logits, 0, &.{ 9, V }), 100, out[0..9], null));
    e.sampling.?.temperature = 0.0;
    t.reset();
    try e.sampleRows(&f, trace.T.of(e.buf.logits, 0, &.{ 4, V }), 100, out[0..4], null);
    try f.flush();
    try testing.expect(t.events.items.len > 0);
    for (t.events.items) |x| try testing.expect(!std.mem.startsWith(u8, x.name, "nucleus."));
    e.sampling = .{ .seed = 5, .temperature = 1.0, .top_k = 50 };
    try testing.expectError(error.TopKNotPorted, e.sampleRows(&f, trace.T.of(e.buf.logits, 0, &.{ 1, V }), 100, out[0..1], null));
}
