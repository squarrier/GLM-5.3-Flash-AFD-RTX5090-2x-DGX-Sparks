//! tf-tp2-gather's CPU side: the size table (16 KiB to 32 MiB a rank), Mia's reference figures, the expected words and the timing statistics.
const std = @import("std");

/// GLM-5.3-Flash's hidden width: a TP2 layer gathers fp32 partials of [rows, 4096], 16 KiB a row a rank.
pub const glm_width: usize = 4096;
/// Flash Next's hidden width, the shape of Mia's measured gathers (10 KiB a row a rank).
pub const mia_width: usize = 2560;

pub const Case = struct { name: []const u8, rows: usize, width: usize };

pub fn bytesOf(c: Case) usize {
    return c.rows * c.width * 4;
}

/// The card's sizes: GLM rows 1 to 2,048 (16 KiB to 32 MiB a rank), RoCE's 512 KiB slot at 32 rows.
pub const glm_cases = [_]Case{
    .{ .name = "glm-1", .rows = 1, .width = glm_width },
    .{ .name = "glm-2", .rows = 2, .width = glm_width },
    .{ .name = "glm-4", .rows = 4, .width = glm_width },
    .{ .name = "glm-8", .rows = 8, .width = glm_width },
    .{ .name = "glm-16", .rows = 16, .width = glm_width },
    .{ .name = "glm-32", .rows = 32, .width = glm_width },
    .{ .name = "glm-64", .rows = 64, .width = glm_width },
    .{ .name = "glm-128", .rows = 128, .width = glm_width },
    .{ .name = "glm-256", .rows = 256, .width = glm_width },
    .{ .name = "glm-512", .rows = 512, .width = glm_width },
    .{ .name = "glm-1024", .rows = 1024, .width = glm_width },
    .{ .name = "glm-2048", .rows = 2048, .width = glm_width },
};

/// Mia's measured shapes (Flash Next, nodes.sh:209-212): a decode row, 16 rows, a 2,048-row prompt chunk (21 MB a rank); plus 4-byte-unit shapes.
pub const mia_cases = [_]Case{
    .{ .name = "mia-decode-1", .rows = 1, .width = mia_width },
    .{ .name = "mia-decode-16", .rows = 16, .width = mia_width },
    .{ .name = "mia-prompt-2048", .rows = 2048, .width = mia_width },
    .{ .name = "odd-7x65", .rows = 7, .width = 65 },
    .{ .name = "agree-1", .rows = 1, .width = 1 },
};

/// Mia's figures to compare with, in microseconds, and where they come from.
pub const Figure = struct { case: []const u8, us: f64, what: []const u8 };
pub const mia_figures = [_]Figure{
    .{ .case = "mia-decode-1", .us = 19.5, .what = "a decode row, fp32 all-gather on two rails (nodes.sh:211)" },
    .{ .case = "mia-decode-16", .us = 34.0, .what = "16 rows (nodes.sh:211)" },
    .{ .case = "mia-prompt-2048", .us = 1180.0, .what = "a 2,048-row prompt chunk, 21 MB a rank (nodes.sh:211-212)" },
    .{ .case = "glm-2048", .us = 1900.0, .what = "a 32 MB all-gather on two rails, 3.5 ms on one (nodes.sh:209-210): 32 MiB a rank, at her 21 MB chunk's rate" },
};

pub fn figureFor(name: []const u8) ?Figure {
    for (mia_figures) |f| if (std.mem.eql(u8, f.case, name)) return f;
    return null;
}

/// The cases `set` names: glm (the card's 12 sizes), mia (her shapes), or all.
pub fn select(set: []const u8) ![]const Case {
    if (std.mem.eql(u8, set, "glm")) return &glm_cases;
    if (std.mem.eql(u8, set, "mia")) return &mia_cases;
    if (std.mem.eql(u8, set, "all")) return &all_cases;
    return error.UnknownSet;
}

const all_cases = glm_cases ++ mia_cases;

/// The largest per-rank shard of `cases`, for buffers allocated once.
pub fn mostBytes(cases: []const Case) usize {
    var m: usize = 0;
    for (cases) |c| m = @max(m, bytesOf(c));
    return m;
}

fn splitmix(x: u64) u64 {
    var z = x +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/// Rank `r`'s words for one seed: first the fp32 values a careless copy changes (NaN payloads, signed zeros, denormals, infinities), then a stream of its own (Mia's comm_test fill).
pub fn fill(words: []u32, seed: u64, r: u32) void {
    const special = [_]u32{ 0x7fc0_0001, 0xffc1_2345, 0x7f80_0001, 0x8000_0000, 0x0000_0000, 0x0000_0001, 0x807f_ffff, 0x7f80_0000, 0xff80_0000, 0x7f7f_ffff };
    for (words, 0..) |*w, i| {
        w.* = if (i < special.len) special[i] else @truncate(splitmix(seed ^ (@as(u64, r) << 56) ^ @as(u64, i)));
    }
    // the ranks' blocks differ everywhere, so a swap or a duplicated block shows: rank 1 holds the specials in reverse order
    if (r == 1) std.mem.reverse(u32, words[0..@min(special.len, words.len)]);
}

/// The first differing word of `got` against `want`, and how many differ.
pub const Diff = struct { count: usize, first: usize };

pub fn diff(got: []const u32, want: []const u32) ?Diff {
    if (std.mem.eql(u32, got, want)) return null;
    var d: Diff = .{ .count = 0, .first = 0 };
    for (got, want, 0..) |g, w, i| if (g != w) {
        if (d.count == 0) d.first = i;
        d.count += 1;
    };
    return d;
}

pub const Stats = struct { p50: f64, p90: f64, min: f64, mean: f64 };

/// Percentiles of `v` (sorted in place): nearest rank, as Mia's comm_test reports them.
pub fn stats(v: []f64) Stats {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (v) |x| sum += x;
    const at = struct {
        fn q(s: []const f64, p: f64) f64 {
            const i: usize = @intFromFloat(@floor(p * @as(f64, @floatFromInt(s.len - 1)) + 0.5));
            return s[i];
        }
    }.q;
    return .{ .p50 = at(v, 0.5), .p90 = at(v, 0.9), .min = v[0], .mean = sum / @as(f64, @floatFromInt(v.len)) };
}

/// nccl-tests' bus bandwidth of a two-rank all-gather: the bytes each rank sends over the time, in GB/s.
pub fn busbw(bytes_per_rank: usize, us: f64) f64 {
    return @as(f64, @floatFromInt(bytes_per_rank)) / 1e3 / us;
}

/// The text both ranks agree on before they gather (Link.agree): the same cases, counts and options on both.
pub fn settingsText(buf: []u8, set: []const u8, iters: usize, warmup: usize, roce: bool) ![]const u8 {
    return std.fmt.bufPrint(buf, "tf-tp2-gather v1 set={s} iters={d} warmup={d} roce={}", .{ set, iters, warmup, roce });
}

/// run's and link's options after RANK MASTER PORT.
pub const Flags = struct {
    set: []const u8 = "all",
    iters: usize = 500,
    warmup: usize = 50,
    /// the rails NCCL_IB_HCA must name (and NCCL_MIN/MAX_NCHANNELS 4, as Mia's launcher sets them); null: not checked
    rails: ?usize = null,
    /// after the table, gather (decode rows over RoCE, 4 MiB over NCCL) for this long: the failure tests' load
    soak_s: f64 = 0,
    /// link: how long the round trips run
    hold_s: f64 = 10,
    sigterm_at: ?f64 = null,
    stall_at: ?f64 = null,
    /// roce-cpu: rank 1 rings a bad doorbell at its 10th gather (size: 8 bytes; skip: three sequences ahead), which its proxy must refuse
    fault: Fault = .none,

    pub const Fault = enum { none, size, skip };

    pub fn parse(args: []const []const u8) !Flags {
        var f: Flags = .{};
        var i: usize = 0;
        while (i < args.len) : (i += 2) {
            if (i + 1 >= args.len) return error.Usage;
            const k = args[i];
            const v = args[i + 1];
            if (std.mem.eql(u8, k, "--set")) {
                _ = select(v) catch return error.Usage;
                f.set = v;
            } else if (std.mem.eql(u8, k, "--iters")) {
                f.iters = std.fmt.parseInt(usize, v, 10) catch return error.Usage;
            } else if (std.mem.eql(u8, k, "--warmup")) {
                f.warmup = std.fmt.parseInt(usize, v, 10) catch return error.Usage;
            } else if (std.mem.eql(u8, k, "--rails")) {
                f.rails = std.fmt.parseInt(usize, v, 10) catch return error.Usage;
            } else if (std.mem.eql(u8, k, "--soak")) {
                f.soak_s = std.fmt.parseFloat(f64, v) catch return error.Usage;
            } else if (std.mem.eql(u8, k, "--hold")) {
                f.hold_s = std.fmt.parseFloat(f64, v) catch return error.Usage;
            } else if (std.mem.eql(u8, k, "--sigterm-at")) {
                f.sigterm_at = std.fmt.parseFloat(f64, v) catch return error.Usage;
            } else if (std.mem.eql(u8, k, "--stall-at")) {
                f.stall_at = std.fmt.parseFloat(f64, v) catch return error.Usage;
            } else if (std.mem.eql(u8, k, "--fault")) {
                f.fault = std.meta.stringToEnum(Fault, v) orelse return error.Usage;
            } else return error.Usage;
        }
        if (f.iters == 0) return error.Usage;
        return f;
    }
};

/// run's start gates on one rank, from its environment: null when they pass, else the reason (both ranks agree on the outcome before they gather).
pub fn gates(lookup: *const fn ([:0]const u8) ?[]const u8, rails: ?usize, proxy_built: bool) ?[]const u8 {
    const roce = if (lookup("TF_TP_ROCE")) |v| !std.mem.eql(u8, v, "0") else true;
    if (roce and !proxy_built) return "RoCE is on but this binary has the proxy stub (built without libibverbs' header)";
    if (roce and lookup("NCCL_IB_HCA") == null) return "RoCE is on but NCCL_IB_HCA is not set";
    const want = rails orelse return null;
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, std.mem.trimStart(u8, lookup("NCCL_IB_HCA") orelse "", "="), ',');
    while (it.next()) |_| n += 1;
    if (n != want) return "NCCL_IB_HCA does not name the rails --rails asks for";
    for ([_][:0]const u8{ "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS" }) |name| {
        if (!std.mem.eql(u8, lookup(name) orelse "", "4")) return "NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS must be 4 (Mia's launcher)";
    }
    return null;
}

test "gates: the proxy, the rails and 4 channels" {
    const Env = struct {
        var pairs: []const [2][]const u8 = &.{};
        fn get(name: [:0]const u8) ?[]const u8 {
            for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
            return null;
        }
    };
    Env.pairs = &.{ .{ "NCCL_IB_HCA", "=rocep1s0f0,roceP2p1s0f0" }, .{ "NCCL_MIN_NCHANNELS", "4" }, .{ "NCCL_MAX_NCHANNELS", "4" } };
    try std.testing.expect(gates(Env.get, 2, true) == null);
    try std.testing.expect(gates(Env.get, 1, true) != null);
    try std.testing.expect(gates(Env.get, 2, false) != null);
    Env.pairs = &.{ .{ "TF_TP_ROCE", "0" }, .{ "NCCL_IB_HCA", "rocep1s0f0" }, .{ "NCCL_MIN_NCHANNELS", "4" }, .{ "NCCL_MAX_NCHANNELS", "2" } };
    try std.testing.expect(gates(Env.get, null, false) == null); // NCCL alone: no proxy needed
    try std.testing.expect(gates(Env.get, 1, false) != null); // the channels
    Env.pairs = &.{};
    try std.testing.expect(gates(Env.get, null, true) != null); // RoCE on by default needs NCCL_IB_HCA
}

test "flags: defaults, every option, and refusals" {
    const d = try Flags.parse(&.{});
    try std.testing.expectEqualStrings("all", d.set);
    try std.testing.expectEqual(@as(usize, 500), d.iters);
    try std.testing.expect(d.rails == null and d.sigterm_at == null and d.soak_s == 0);
    const f = try Flags.parse(&.{ "--set", "glm", "--iters", "200", "--warmup", "20", "--rails", "2", "--soak", "30", "--sigterm-at", "12.5", "--stall-at", "9" });
    try std.testing.expectEqualStrings("glm", f.set);
    try std.testing.expectEqual(@as(usize, 200), f.iters);
    try std.testing.expectEqual(@as(?usize, 2), f.rails);
    try std.testing.expectEqual(@as(?f64, 12.5), f.sigterm_at);
    try std.testing.expectEqual(Flags.Fault.none, f.fault);
    try std.testing.expectEqual(Flags.Fault.skip, (try Flags.parse(&.{ "--fault", "skip" })).fault);
    try std.testing.expectError(error.Usage, Flags.parse(&.{ "--fault", "late" }));
    try std.testing.expectError(error.Usage, Flags.parse(&.{"--iters"}));
    try std.testing.expectError(error.Usage, Flags.parse(&.{ "--iters", "0" }));
    try std.testing.expectError(error.Usage, Flags.parse(&.{ "--set", "qwen" }));
    try std.testing.expectError(error.Usage, Flags.parse(&.{ "--bogus", "1" }));
}

test "the card's sizes run from 16 KiB to 32 MiB a rank, RoCE's default slot at 32 rows" {
    try std.testing.expectEqual(@as(usize, 16 << 10), bytesOf(glm_cases[0]));
    try std.testing.expectEqual(@as(usize, 32 << 20), bytesOf(glm_cases[glm_cases.len - 1]));
    try std.testing.expectEqual(@as(usize, 512 << 10), bytesOf(glm_cases[5]));
    for (glm_cases[1..], glm_cases[0 .. glm_cases.len - 1]) |c, p| try std.testing.expectEqual(2 * bytesOf(p), bytesOf(c));
    try std.testing.expectEqual(@as(usize, 32 << 20), mostBytes(try select("all")));
    try std.testing.expectError(error.UnknownSet, select("qwen"));
}

test "Mia's shapes: 10 KiB a decode row, 160 KiB for 16 rows, 21 MB a prompt chunk" {
    try std.testing.expectEqual(@as(usize, 10240), bytesOf(mia_cases[0]));
    try std.testing.expectEqual(@as(usize, 163840), bytesOf(mia_cases[1]));
    try std.testing.expectEqual(@as(usize, 20971520), bytesOf(mia_cases[2]));
    try std.testing.expect(bytesOf(mia_cases[3]) % 16 != 0); // the kernel's 4-byte form
    for (mia_figures) |f| {
        var found = false;
        for (try select("all")) |c| found = found or std.mem.eql(u8, c.name, f.case);
        try std.testing.expect(found);
    }
    try std.testing.expectEqual(@as(f64, 19.5), figureFor("mia-decode-1").?.us);
    try std.testing.expect(figureFor("glm-1024") == null and figureFor("glm-2048").?.us == 1900); // her 32 MB in 1.9 ms is 32 MiB a rank
}

test "fill: the ranks' blocks differ everywhere, the specials lead, a seed changes every drawn word" {
    var a: [64]u32 = undefined;
    var b: [64]u32 = undefined;
    var c: [64]u32 = undefined;
    fill(&a, 1, 0);
    fill(&b, 1, 1);
    fill(&c, 2, 0);
    try std.testing.expectEqual(@as(u32, 0x7fc0_0001), a[0]);
    try std.testing.expectEqual(@as(u32, 0x7f7f_ffff), b[0]);
    for (a, b) |x, y| try std.testing.expect(x != y or x == 0);
    for (a[10..], c[10..]) |x, y| try std.testing.expect(x != y);
    try std.testing.expectEqual(@as(?Diff, null), diff(&a, &a));
    a[17] ^= 1;
    a[40] ^= 1;
    const d = diff(&a, &c).?;
    try std.testing.expectEqual(@as(usize, 10), d.first); // the specials match, the drawn words do not
    var e = a;
    e[17] ^= 1;
    try std.testing.expectEqual(@as(usize, 17), diff(&e, &a).?.first);
}

test "stats: nearest-rank percentiles; busbw in GB/s" {
    var v = [_]f64{ 5, 1, 4, 2, 3, 10, 6, 7, 8, 9 };
    const s = stats(&v);
    try std.testing.expectEqual(@as(f64, 1), s.min);
    try std.testing.expectEqual(@as(f64, 6), s.p50);
    try std.testing.expectEqual(@as(f64, 9), s.p90);
    try std.testing.expectEqual(@as(f64, 5.5), s.mean);
    try std.testing.expectApproxEqAbs(@as(f64, 17.66), busbw(32 << 20, 1900), 0.01);
}

test "both ranks agree on the same settings text" {
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    try std.testing.expectEqualStrings(try settingsText(&b1, "all", 500, 50, true), try settingsText(&b2, "all", 500, 50, true));
    try std.testing.expect(!std.mem.eql(u8, try settingsText(&b1, "all", 500, 50, true), try settingsText(&b2, "glm", 500, 50, true)));
}
