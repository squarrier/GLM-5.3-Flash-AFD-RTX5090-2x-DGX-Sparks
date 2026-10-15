//! python-0.6's TP nucleus draw on the host (cuda/sampling.py `_draw`, ed78d6f) with Mia's union cover always on (patch
//! 0034, MiaAI-Lab recipe 549fdc5): each rank's gathered candidates (value, id, fixed-point mass) and shard sum; a row is
//! covered when the ranks' candidates merged (value down, then id) reach top_p of the mass above every partial rank's
//! last one, so no token left out can be in the nucleus and the draw is whole shards' draw. `levels` takes NUCLEUS
//! candidates, then WIDER, then whole shards, as nucleus_rows does with 0034's flag on; `check` is Mia's 0092 rule.
const std = @import("std");

extern "c" fn log(x: f64) f64;

/// python's NUCLEUS: candidates a rank sends first; 0034's WIDER: then, when a shard is wider than it.
pub const nucleus = 1024;
pub const wider = 16 * nucleus;
/// ranks a packet may hold
pub const max_world = 8;

/// python's exact_sampling.Sampling (core/lanes/sampling.zig's, which the GLM module does not import).
pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: u32 = 0,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,

    /// ln(min_p) (libm's, as python's math.log), -inf when off.
    pub fn minLog(s: Sampling) f64 {
        return if (s.min_p > 0.0) log(s.min_p) else -std.math.inf(f64);
    }
};

/// splitmix64's finalizer and the keyed uniform (0, 1] of (seed, position, id), Ash Hart's core/lanes/sampling.zig.
pub fn mix(v: u64) u64 {
    var x = v;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

/// math.ceil(Fraction(top_p) * total) exactly, for 0 < top_p < 1: top_p = m * 2^-e, m < 2^53, in 128 bits.
pub fn need(top_p: f64, total: u64) u64 {
    const bits: u64 = @bitCast(top_p);
    const field = (bits >> 52) & 0x7FF;
    const frac = bits & ((@as(u64, 1) << 52) - 1);
    const m: u128 = if (field == 0) frac else frac | (@as(u64, 1) << 52);
    const e: u32 = if (field == 0) 1074 else @intCast(1075 - field);
    const p = m * total;
    if (p == 0) return 0;
    if (e >= 128) return 1;
    const q = p >> @intCast(e);
    return @intCast(if (p & ((@as(u128, 1) << @intCast(e)) - 1) != 0) q + 1 else q);
}

/// The ranks' vocabulary shards: rank k holds ids [cut[k], cut[k + 1]).
pub const Shards = struct {
    world: usize,
    cut: [max_world + 1]u64 = @splat(0),

    pub fn uniform(world: usize, w: u64) Shards {
        var s: Shards = .{ .world = world };
        for (0..world + 1) |k| s.cut[k] = k * w;
        return s;
    }
    pub fn of(cuts: []const u64) Shards {
        var s: Shards = .{ .world = cuts.len - 1 };
        @memcpy(s.cut[0..cuts.len], cuts);
        return s;
    }
    pub fn width(s: Shards, k: usize) u64 {
        return s.cut[k + 1] - s.cut[k];
    }
    pub fn widest(s: Shards) u64 {
        var w: u64 = 0;
        for (0..s.world) |k| w = @max(w, s.width(k));
        return w;
    }
};

/// The gathered rows of every rank, 64-bit words [world][stride]: a header (0, or a rank's kernels saw a bad index),
/// the shard's mass sum a row, then [rows][count] values (f64 bits), ids (-1: padding) and masses.
pub const Packet = struct {
    words: []const u64,
    world: usize,
    rows: usize,
    count: usize,

    pub fn stride(rows: usize, count: usize) usize {
        return 1 + rows + 3 * rows * count;
    }
    fn at(p: Packet, k: usize, part: usize, r: usize, j: usize) u64 {
        return p.words[k * stride(p.rows, p.count) + 1 + p.rows + (part * p.rows + r) * p.count + j];
    }
    pub fn header(p: Packet, k: usize) u64 {
        return p.words[k * stride(p.rows, p.count)];
    }
    pub fn sum(p: Packet, k: usize, r: usize) u64 {
        return p.words[k * stride(p.rows, p.count) + 1 + r];
    }
    pub fn value(p: Packet, k: usize, r: usize, j: usize) f64 {
        return @bitCast(p.at(k, 0, r, j));
    }
    pub fn id(p: Packet, k: usize, r: usize, j: usize) i64 {
        return @bitCast(p.at(k, 1, r, j));
    }
    pub fn mass(p: Packet, k: usize, r: usize, j: usize) i64 {
        return @bitCast(p.at(k, 2, r, j));
    }
};

/// 0092's rule (Mia's, for the drafter's candidates): every gathered id inside its rank's shard, padding (-1) only past
/// the rank's real candidates, every mass and sum non-negative, and no rank's header set.
pub fn check(p: Packet, sh: Shards) bool {
    if (p.world != sh.world) return false;
    for (0..p.world) |k| {
        if (p.header(k) != 0) return false;
        const real = @min(p.count, sh.width(k));
        for (0..p.rows) |r| {
            if (@as(i64, @bitCast(p.sum(k, r))) < 0) return false;
            for (0..p.count) |j| {
                const i = p.id(k, r, j);
                if (j >= real) {
                    if (i != -1) return false;
                } else if (i < sh.cut[k] or i >= sh.cut[k + 1] or p.mass(k, r, j) < 0) return false;
            }
        }
    }
    return true;
}

pub const Cand = struct { v: f64, id: i64, m: i64 };

/// np.lexsort((ids, -values)): value down (-0 equal to +0), then id up.
fn before(_: void, a: Cand, b: Cand) bool {
    return a.v > b.v or (a.v == b.v and a.id < b.id);
}

pub const Drawn = struct { token: u32, share: f64 };

/// `_draw` for row r with 0034's cover: its (token, share of the mass), or null when its nucleus runs past the
/// candidates. `work` holds a row's candidates of every rank.
pub fn drawRow(p: Packet, sh: Shards, r: usize, position: u64, s: Sampling, work: []Cand) !?Drawn {
    const cut = s.top_p > 0.0 and s.top_p < 1.0;
    var total: u64 = 0;
    var top = -std.math.inf(f64);
    var n: usize = 0;
    for (0..p.world) |k| {
        total = std.math.add(u64, total, p.sum(k, r)) catch return error.NucleusMass;
        for (0..p.count) |j| {
            const v = p.value(k, r, j);
            if (v > top or std.math.isNan(v)) top = v; // vals[:, r, :].max(), padding's -inf included
            if (p.id(k, r, j) < 0) continue;
            if (n == work.len) return error.NucleusWork;
            work[n] = .{ .v = v, .id = p.id(k, r, j), .m = p.mass(k, r, j) };
            n += 1;
        }
    }
    if (total == 0 or n == 0) return error.NucleusMass;
    const floor = top + s.minLog(); // the min_p cut (-inf when off)
    const c = work[0..n];
    std.mem.sort(Cand, c, {}, before);
    for (c[1..], c[0 .. n - 1]) |x, y| if (x.id == y.id) return error.NucleusDuplicate;
    var limit = n;
    if (s.top_k > 0) {
        const wanted = @min(@as(usize, s.top_k), sh.cut[sh.world]);
        if (n < wanted) return null;
        limit = wanted;
        for (0..p.world) |k| {
            if (p.count >= sh.width(k)) continue;
            if (!(c[limit - 1].v > lastOf(p, k, r))) return null;
        }
        total = 0;
        for (c[0..limit]) |x| total = std.math.add(u64, total, @intCast(x.m)) catch return error.NucleusMass;
        if (total == 0) return error.NucleusMass;
    }
    var keep = limit;
    if (cut) {
        const nd = need(s.top_p, total);
        var acc: u64 = 0;
        var stop: ?usize = null;
        for (c[0..limit], 0..) |x, i| {
            acc +|= @intCast(x.m);
            if (acc >= nd) {
                stop = i;
                break;
            }
        }
        for (0..p.world) |k| { // _union_covers: above the last candidate of every rank that sent part of its shard
            if (@min(p.count, sh.width(k)) == sh.width(k)) continue;
            const i = stop orelse return null;
            if (!(c[i].v > lastOf(p, k, r))) return null;
        }
        if (stop) |i| keep = i + 1;
    } else if (s.top_k == 0) {
        for (0..p.world) |k| { // the stock rule: min_p alone, a candidate below its cut
            if (@min(p.count, sh.width(k)) == sh.width(k)) continue;
            if (!(s.min_p > 0.0 and anyBelow(p, k, r, floor))) return null;
        }
    }
    if (s.min_p > 0.0) {
        var below: usize = 0;
        for (c) |x| below += @intFromBool(x.v >= floor);
        keep = @min(keep, below);
    }
    if (keep == 0) return error.NucleusEmpty;
    var best: usize = 0;
    var best_score: f64 = 0;
    for (c[0..keep], 0..) |x, j| {
        const score = x.v - log(-log(uniform(s.seed, position, @bitCast(x.id))));
        if (j == 0 or (!std.math.isNan(best_score) and (std.math.isNan(score) or score > best_score))) {
            best = j;
            best_score = score;
        }
    }
    const m: f64 = @floatFromInt(c[best].m);
    return .{ .token = @intCast(c[best].id), .share = m / @as(f64, @floatFromInt(total)) };
}

/// A rank's last candidate in row r: its least real value.
fn lastOf(p: Packet, k: usize, r: usize) f64 {
    var v = std.math.inf(f64);
    for (0..p.count) |j| if (p.id(k, r, j) >= 0) {
        const x = p.value(k, r, j);
        if (x < v or std.math.isNan(x)) v = x;
    };
    return v;
}

fn anyBelow(p: Packet, k: usize, r: usize, floor: f64) bool {
    for (0..p.count) |j| if (p.id(k, r, j) >= 0 and p.value(k, r, j) < floor) return true;
    return false;
}

/// `_draw`: every row's draw, or false (nothing kept) when some row's nucleus runs past the candidates.
pub fn draw(p: Packet, sh: Shards, positions: []const u64, s: Sampling, work: []Cand, out: []Drawn) !bool {
    if (positions.len != p.rows or out.len < p.rows) return error.NucleusRows;
    for (0..p.rows) |r| out[r] = (try drawRow(p, sh, r, positions[r], s, work)) orelse return false;
    return true;
}

/// Per request: sampled calls by the gathers they took (1: NUCLEUS covered; 2: WIDER, or whole shards when no shard is
/// wider; 3: whole shards), and the rows drawn.
pub const Stats = struct {
    gathers: [3]u64 = @splat(0),
    rows: u64 = 0,

    pub fn add(s: *Stats, gathers: usize, rows: usize) void {
        s.gathers[gathers - 1] += 1;
        s.rows += rows;
    }
};

/// nucleus_rows with 0034's flag on over `src` (`gather(count)`: every rank's top `count` candidates; `again()`: the
/// last gather's rows copied again after a device sync): NUCLEUS, then WIDER when a shard is wider, then whole shards.
/// Each gather passes `check` or is copied once again, then the request fails. Gives the gathers taken (1-3).
pub fn levels(src: anytype, sh: Shards, positions: []const u64, s: Sampling, work: []Cand, out: []Drawn) !usize {
    const widest = sh.widest();
    const counts = [3]u64{ nucleus, wider, widest };
    var n: usize = 0;
    for (counts, 0..) |count, i| {
        if (i == 1 and widest <= wider) continue;
        var p = try src.gather(@intCast(@min(count, widest)));
        n += 1;
        if (!check(p, sh)) {
            p = try src.again();
            if (!check(p, sh)) return error.BadTokenId;
        }
        if (try draw(p, sh, positions, s, work, out)) return n;
    }
    return error.NucleusNotCovered;
}

test "need is math.ceil(Fraction(top_p) * total)" {
    // python: math.ceil(Fraction(top_p) * total) for each pair
    try std.testing.expectEqual(@as(u64, 1044536046388), need(0.95, 1 << 40));
    try std.testing.expectEqual(@as(u64, 600002), need(0.6, 1000003));
    try std.testing.expectEqual(@as(u64, 4565569158243126206), need(0.99, (1 << 62) + 12345));
    try std.testing.expectEqual(@as(u64, 161777742864449529), need(0.95, 154880 << 40));
    try std.testing.expectEqual(@as(u64, 1), need(0x1p-1074, 3));
    try std.testing.expectEqual(@as(u64, 0), need(0.5, 0));
    try std.testing.expectEqual(@as(u64, 5), need(0.5, 10));
}

test "uniform is python's keyed uniform (exact_sampling.uniform)" {
    // python: uniform(12345, 1000, [7, 154879])
    try std.testing.expectEqual(@as(u64, 0x3fe71ce56472c90c), @as(u64, @bitCast(uniform(12345, 1000, 7))));
    try std.testing.expectEqual(@as(u64, 0x3f91ceeff08577d0), @as(u64, @bitCast(uniform(12345, 1000, 154879))));
}
