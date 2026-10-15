//! Mia's EXL3 GLM-5.3-Flash checkpoint (078455ff: the routed experts EXL3 4-bit mcg, every other tensor BF16) for one TP
//! rank, as python-0.6 split.py and weights.py read it: split.py's rule for each language-model tensor (rows, columns,
//! the second axis, or whole), the head's vocabulary rows as weights.py keeps them, and the routed experts stacked per
//! projection for the EXL3 kernels (experts.prepare): gate/up keep their tile columns and svh, down its tile rows and
//! suh (TP2: each rank half of every routed expert's rows). The codebook markers are checked by name, never read.
const std = @import("std");
const cfg = @import("config.zig");
const split = @import("cuda_split.zig");
const ex = @import("cuda_exl3_experts.zig");
const gm = @import("cuda_exl3_glm.zig");

const Ref = split.Ref;
const Src = split.Src;

/// split.py's split kinds: whole, the leading axis, the second axis of a 2-D tensor, the second axis of any.
pub const Rule = enum { rep, row, col, dim1 };

const prefix = "model.language_model.";

fn expertProj(name: []const u8) ?struct { proj: []const u8, part: []const u8 } {
    const at = std.mem.indexOf(u8, name, ".mlp.experts.") orelse return null;
    var i = at + ".mlp.experts.".len;
    const d0 = i;
    while (i < name.len and std.ascii.isDigit(name[i])) i += 1;
    if (i == d0 or i >= name.len or name[i] != '.') return null;
    const rest = name[i + 1 ..];
    const us = std.mem.indexOf(u8, rest, "_proj.") orelse return null;
    return .{ .proj = rest[0..us], .part = rest[us + "_proj.".len ..] };
}

fn has(name: []const u8, s: []const u8) bool {
    return std.mem.indexOf(u8, name, s) != null;
}

fn anyOf(name: []const u8, comptime fmt: []const u8, alts: []const []const u8) bool {
    for (alts) |a| {
        var buf: [96]u8 = undefined;
        if (has(name, std.fmt.bufPrint(&buf, fmt, .{a}) catch unreachable)) return true;
    }
    return false;
}

fn endsAny(name: []const u8, comptime fmt: []const u8, alts: []const []const u8) bool {
    for (alts) |a| {
        var buf: [96]u8 = undefined;
        if (std.mem.endsWith(u8, name, std.fmt.bufPrint(&buf, fmt, .{a}) catch unreachable)) return true;
    }
    return false;
}

/// split.py's rule(): its EXL3 expert table first, then exactly one of its ROW, COL and REP patterns must match.
pub fn rule(name: []const u8) error{NoRule}!Rule {
    if (expertProj(name)) |e| {
        const parts = [_][]const u8{ "trellis", "suh", "svh", "mcg" };
        for (parts) |p| if (std.mem.eql(u8, e.part, p)) {
            if (std.mem.eql(u8, p, "mcg")) return .rep;
            const down = std.mem.eql(u8, e.proj, "down");
            if (!down and !std.mem.eql(u8, e.proj, "gate") and !std.mem.eql(u8, e.proj, "up")) break;
            if (std.mem.eql(u8, p, "trellis")) return if (down) .row else .dim1;
            if (std.mem.eql(u8, p, "suh")) return if (down) .row else .rep;
            return if (down) .rep else .row; // svh
        };
    }
    const e = expertProj(name);
    const row = (e != null and (std.mem.eql(u8, e.?.proj, "gate") or std.mem.eql(u8, e.?.proj, "up"))) or
        anyOf(name, ".mlp.shared_experts.{s}_proj.", &.{ "gate", "up" }) or anyOf(name, ".mlp.{s}_proj.", &.{ "gate", "up" }) or
        anyOf(name, ".self_attn.{s}_proj.", &.{ "q", "k", "v", "f_b", "g_b", "b", "q_b", "kv_b" }) or
        anyOf(name, ".self_attn.{s}_conv1d.", &.{ "q", "k", "v" }) or endsAny(name, ".self_attn.{s}", &.{ "A_log", "dt_bias" });
    const col = (e != null and std.mem.eql(u8, e.?.proj, "down")) or has(name, ".mlp.shared_experts.down_proj.") or
        has(name, ".mlp.down_proj.") or has(name, ".self_attn.o_proj.");
    const rep = std.mem.startsWith(u8, name, "lm_head.") or has(name, "embed_tokens.") or
        std.mem.eql(u8, name, prefix ++ "norm.weight") or std.mem.endsWith(u8, name, "_layernorm.weight") or
        endsAny(name, ".hc_{s}", &.{ "attn_fn", "attn_base", "attn_scale", "ffn_fn", "ffn_base", "ffn_scale" }) or
        endsAny(name, ".mlp.gate.{s}", &.{ "weight", "e_score_correction_bias" }) or has(name, ".self_attn.indexer.") or
        anyOf(name, ".self_attn.{s}.", &.{ "q_a_proj", "kv_a_proj_with_mqa", "f_a_proj", "g_a_proj" }) or
        std.mem.endsWith(u8, name, ".self_attn.o_norm.weight") or has(name, ".eh_proj.") or
        endsAny(name, ".{s}.weight", &.{ "enorm", "hnorm" }) or std.mem.endsWith(u8, name, ".shared_head.norm.weight");
    const hits = @as(u8, @intFromBool(row)) + @intFromBool(col) + @intFromBool(rep);
    if (hits != 1) return error.NoRule;
    return if (row) .row else if (col) .col else .rep;
}

fn bytesOf(d: split.DType) u64 {
    return switch (d) {
        .u32, .f32, .i32 => 4,
        .bf16, .f16, .i16 => 2,
        .other => 0,
    };
}

fn count(s: Src) u64 {
    var n: u64 = 1;
    for (s.shape[0..s.rank]) |d| n *= d;
    return n;
}

/// One projection of a layer's routed experts on this rank, stacked: trellis words, suh [E, k], svh [E, n].
pub const Proj = struct {
    trellis: Ref,
    suh: Ref,
    svh: Ref,
    k: u32, // this rank's input rows
    n: u32, // this rank's output columns
    k2: [2]u32, // (min, max) half-bits a value over the experts
    offsets: []u64, // each expert's trellis, bytes past `trellis`
    k2s: []u8, // each expert's half-bits a value
};

/// A layer's routed experts: the three projections and the pointer and width tables experts.prepare builds
/// (int64 gate, up, down pointers [E], then int32 gate, up, down widths [E]), written after the load (`tables`).
pub const Experts = struct { gate: Proj, up: Proj, down: Proj, tables: Ref, count: u32, cb: ex.Codebook };

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    gpa: std.mem.Allocator,
    rank: u32,
    ranks: u32,
    groups: std.ArrayList(u64) = .empty,
    copies: std.ArrayList(split.Copy) = .empty,
    used: std.StringHashMapUnmanaged(void) = .empty, // names read, and the markers checked by name
    tensors: std.StringHashMapUnmanaged(Ref) = .empty, // every tensor but the routed experts: this rank's part
    experts: [cfg.max_layers]?Experts = @splat(null),
    markers: u32 = 0,

    pub fn deinit(p: *Plan) void {
        p.groups.deinit(p.gpa);
        p.copies.deinit(p.gpa);
        p.used.deinit(p.gpa);
        p.tensors.deinit(p.gpa);
        p.arena.deinit();
    }

    pub fn bytes(p: *const Plan) u64 {
        var n: u64 = 0;
        for (p.groups.items) |g| n += g;
        return n;
    }

    pub fn sourceBytes(p: *const Plan) u64 {
        var n: u64 = 0;
        for (p.copies.items) |c| n += c.len;
        return n;
    }

    /// Layer i's experts bound for the kernels, its group at device address `base`.
    pub fn bind(p: *const Plan, i: usize, base: u64) ?ex.Layer {
        const e = p.experts[i] orelse return null;
        const t = base + e.tables.off;
        const E: u64 = e.count;
        return .{ .gate_ptr = t, .up_ptr = t + 8 * E, .down_ptr = t + 16 * E, .gate_k2 = t + 24 * E, .up_k2 = t + 28 * E, .down_k2 = t + 32 * E, .suh_g = base + e.gate.suh.off, .suh_u = base + e.up.suh.off, .svh_g = base + e.gate.svh.off, .svh_u = base + e.up.svh.off, .suh_d = base + e.down.suh.off, .svh_d = base + e.down.svh.off, .count = e.count, .dims = e.gate.k, .width = e.gate.n, .cb = e.cb, .k2_gu = .{ @min(e.gate.k2[0], e.up.k2[0]), @max(e.gate.k2[1], e.up.k2[1]) }, .k2_d = e.down.k2 };
    }

    /// Layer i's experts as GLM's exl3_mm reads them (4-bit mcg only: each projection's trellises back to back).
    pub fn bindGlm(p: *const Plan, i: usize, base: u64) ?gm.Experts {
        const e = p.experts[i] orelse return null;
        if (e.cb != .mcg) return null;
        for ([_]Proj{ e.gate, e.up, e.down }) |pr| {
            if (pr.k2[0] != 8 or pr.k2[1] != 8) return null;
            for (pr.offsets, 0..) |o, x| if (o != x * @as(u64, pr.k) * pr.n / 2) return null;
        }
        return .{ .gt = base + e.gate.trellis.off, .ut = base + e.up.trellis.off, .dt = base + e.down.trellis.off, .suh_g = base + e.gate.suh.off, .suh_u = base + e.up.suh.off, .svh_g = base + e.gate.svh.off, .svh_u = base + e.up.svh.off, .suh_d = base + e.down.suh.off, .svh_d = base + e.down.svh.off, .count = e.count, .width = e.gate.n, .dims = e.gate.k };
    }

    /// Layer i's tables as the kernels read them, its group at device address `base` (36 E bytes).
    pub fn tables(p: *const Plan, i: usize, base: u64, out: []u8) void {
        const e = p.experts[i].?;
        const E = e.count;
        for ([_]Proj{ e.gate, e.up, e.down }, 0..) |pr, j| for (0..E) |x| {
            std.mem.writeInt(u64, out[(j * E + x) * 8 ..][0..8], base + pr.trellis.off + pr.offsets[x], .little);
            std.mem.writeInt(u32, out[24 * E + (j * E + x) * 4 ..][0..4], pr.k2s[x], .little);
        };
    }
};

const Planner = struct {
    p: *Plan,
    names: *const split.Names,
    at: u64 = 0,

    fn begin(l: *Planner) !void {
        try l.p.groups.append(l.p.gpa, 0);
        l.at = 0;
    }

    fn take(l: *Planner, len: u64) Ref {
        const g: u32 = @intCast(l.p.groups.items.len - 1);
        const r: Ref = .{ .g = g, .off = l.at };
        l.at = std.mem.alignForward(u64, l.at + len, split.alignment);
        l.p.groups.items[g] = l.at;
        return r;
    }

    fn named(l: *Planner, name: []const u8) !struct { key: []const u8, s: Src } {
        const key = l.names.getKey(name) orelse {
            std.log.err("glm exl3: the checkpoint has no tensor {s}", .{name});
            return error.MissingTensor;
        };
        try l.p.used.put(l.p.gpa, key, {});
        return .{ .key = key, .s = l.names.get(key).? };
    }

    /// This rank's part of a tensor stored [d0, d1, ...] by `r`, as split.py's RankReader reads and cuts it.
    fn part(l: *Planner, s: Src, r: Rule, dst: Ref) !u64 {
        const size = bytesOf(s.dtype);
        const ranks = l.p.ranks;
        const rank = l.p.rank;
        if (size == 0 or s.len != size * count(s)) return error.UnexpectedTensor;
        switch (r) {
            .rep => {
                try l.p.copies.append(l.p.gpa, .{ .shard = s.shard, .off = s.off, .len = s.len, .g = dst.g, .dst = dst.off });
                return s.len;
            },
            .row => {
                if (s.shape[0] % ranks != 0) return error.BadSplit;
                const per = s.len / s.shape[0] * (s.shape[0] / ranks);
                try l.p.copies.append(l.p.gpa, .{ .shard = s.shard, .off = s.off + rank * per, .len = per, .g = dst.g, .dst = dst.off });
                return per;
            },
            .col, .dim1 => {
                if (s.rank < 2 or s.shape[1] % ranks != 0 or (r == .col and s.rank != 2)) return error.BadSplit;
                const inner = s.len / (s.shape[0] * s.shape[1]);
                const chunk = s.shape[1] / ranks * inner;
                try l.p.copies.append(l.p.gpa, .{ .shard = s.shard, .off = s.off, .len = s.len, .g = dst.g, .dst = dst.off, .row = @intCast(s.shape[1] * inner), .at = @intCast(rank * chunk), .width = @intCast(chunk) });
                return s.len / s.shape[1] * (s.shape[1] / ranks);
            },
        }
    }

    fn kept(l: *const Planner, s: Src, r: Rule) u64 {
        return switch (r) {
            .rep => s.len,
            .row, .col, .dim1 => s.len / l.p.ranks,
        };
    }

    /// A tensor by split.py's rule into the current group.
    fn generic(l: *Planner, name: []const u8) !void {
        const t = try l.named(name);
        const r = rule(name) catch |err| {
            std.log.err("glm exl3: {s}: split.py's rule is ambiguous or missing", .{name});
            return err;
        };
        const dst = l.take(l.kept(t.s, r));
        _ = try l.part(t.s, r, dst);
        try l.p.tensors.put(l.p.gpa, t.key, dst);
    }

    /// The head's vocabulary rows [rank V / ranks, (rank + 1) V / ranks), as weights.py keeps them of the whole head.
    fn head(l: *Planner, name: []const u8) !void {
        const t = try l.named(name);
        const dst = l.take(t.s.len / l.p.ranks);
        _ = try l.part(t.s, .row, dst);
        try l.p.tensors.put(l.p.gpa, t.key, dst);
    }

    fn src(l: *Planner, i: usize, e: usize, which: []const u8, comp: []const u8) !Src {
        var buf: [160]u8 = undefined;
        return (try l.named(try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{ i, e, which, comp }))).s;
    }

    /// One projection of layer i's E routed experts, stored trellis int16 [K/16, N/16, 8 k2], suh f16 [K], svh f16 [N].
    fn proj(l: *Planner, i: usize, E: u32, name: []const u8, K: u32, N: u32, down: bool) !Proj {
        const a = l.p.arena.allocator();
        const ranks = l.p.ranks;
        const k = if (down) K / ranks else K;
        const n = if (down) N else N / ranks;
        var out: Proj = .{ .trellis = undefined, .suh = undefined, .svh = undefined, .k = k, .n = n, .k2 = .{ 16, 2 }, .offsets = try a.alloc(u64, E), .k2s = try a.alloc(u8, E) };
        var total: u64 = 0;
        for (0..E) |e| {
            const t = try l.src(i, e, name, "trellis");
            const w = t.shape[2];
            if (t.dtype != .i16 or t.rank != 3 or t.shape[0] * 16 != K or t.shape[1] * 16 != N or w % 8 != 0 or w / 8 < 2 or w / 8 > 16) {
                std.log.err("glm exl3: layer {d} expert {d} {s}_proj.trellis is {t} {any}; the kernels read int16 [{d}, {d}, 8 k2]", .{ i, e, name, t.dtype, t.shape[0..t.rank], K / 16, N / 16 });
                return error.UnexpectedTensor;
            }
            out.k2s[e] = @intCast(w / 8);
            out.k2 = .{ @min(out.k2[0], out.k2s[e]), @max(out.k2[1], out.k2s[e]) };
            out.offsets[e] = total;
            total = std.mem.alignForward(u64, total + t.len / ranks, 16);
        }
        out.trellis = l.take(total);
        for (0..E) |e| _ = try l.part(try l.src(i, e, name, "trellis"), if (down) .row else .dim1, .{ .g = out.trellis.g, .off = out.trellis.off + out.offsets[e] });
        out.suh = l.take(@as(u64, E) * k * 2);
        out.svh = l.take(@as(u64, E) * n * 2);
        for (0..E) |e| {
            const suh = try l.src(i, e, name, "suh");
            const svh = try l.src(i, e, name, "svh");
            if (suh.dtype != .f16 or suh.rank != 1 or suh.shape[0] != K or svh.dtype != .f16 or svh.rank != 1 or svh.shape[0] != N) return error.UnexpectedTensor;
            _ = try l.part(suh, if (down) .row else .rep, .{ .g = out.suh.g, .off = out.suh.off + e * k * 2 });
            _ = try l.part(svh, if (down) .rep else .row, .{ .g = out.svh.g, .off = out.svh.off + e * n * 2 });
        }
        return out;
    }

    /// The codebook from the marker tensors' names (format.py's rule: mul1, else mcg, else 3inst), every expert alike.
    fn codebook(l: *Planner, i: usize, E: u32) !ex.Codebook {
        var cb: ?ex.Codebook = null;
        for (0..E) |e| for ([_][]const u8{ "gate", "up", "down" }) |pr| {
            var got: ex.Codebook = .@"3inst";
            for ([_]struct { []const u8, ex.Codebook }{ .{ "mcg", .mcg }, .{ "mul1", .mul1 } }) |m| {
                var buf: [160]u8 = undefined;
                const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{ i, e, pr, m[0] });
                if (l.names.getKey(name)) |key| {
                    const s = l.names.get(key).?;
                    if (s.dtype != .i32 or s.len != 4) return error.UnexpectedTensor;
                    try l.p.used.put(l.p.gpa, key, {});
                    l.p.markers += 1;
                    got = m[1];
                }
            }
            if (cb != null and cb.? != got) return error.MixedCodebooks;
            cb = got;
        };
        return cb.?;
    }

    fn experts(l: *Planner, c: *const cfg.Config, i: usize) !Experts {
        const E = c.experts;
        try l.begin();
        const gate = try l.proj(i, E, "gate", c.hidden, c.moe_inter, false);
        const up = try l.proj(i, E, "up", c.hidden, c.moe_inter, false);
        const down = try l.proj(i, E, "down", c.moe_inter, c.hidden, true);
        return .{ .gate = gate, .up = up, .down = down, .tables = l.take(36 * @as(u64, E)), .count = E, .cb = try l.codebook(i, E) };
    }
};

fn layerOf(name: []const u8) ?usize {
    const p = prefix ++ "layers.";
    if (!std.mem.startsWith(u8, name, p)) return null;
    const rest = name[p.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(usize, rest[0..dot], 10) catch null;
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Rank `rank` of `ranks`: the head's group (embedding, head rows, final norm), then per layer of the first `c.run` and
/// the MTP layer (when stored) a group of its other tensors by split.py's rules and, on MoE layers, one of its experts.
pub fn plan(gpa: std.mem.Allocator, c: *const cfg.Config, names: *const split.Names, rank: u32, ranks: u32) !Plan {
    var p: Plan = .{ .arena = .init(gpa), .gpa = gpa, .rank = rank, .ranks = ranks };
    errdefer p.deinit();
    var l: Planner = .{ .p = &p, .names = names };
    const a = p.arena.allocator();
    var by_layer: [cfg.max_layers]std.ArrayList([]const u8) = @splat(.empty);
    var it = names.keyIterator();
    while (it.next()) |k| if (layerOf(k.*)) |i| {
        if (i < cfg.max_layers and expertProj(k.*) == null) try by_layer[i].append(a, k.*);
    };
    try l.begin();
    try l.generic(prefix ++ "embed_tokens.weight");
    try l.head("lm_head.weight");
    try l.generic(prefix ++ "norm.weight");
    var mtp_name: [96]u8 = undefined;
    const has_mtp = c.mtp > 0 and names.contains(try std.fmt.bufPrint(&mtp_name, prefix ++ "layers.{d}.eh_proj.weight", .{c.layers}));
    for (0..c.layers + 1) |i| {
        if (i >= c.run and !(i == c.layers and has_mtp)) continue;
        const ns = by_layer[i].items;
        std.mem.sort([]const u8, ns, {}, lessThan);
        try l.begin();
        for (ns) |n| try l.generic(n);
        if (c.isMoe(@intCast(i))) p.experts[i] = try l.experts(c, i);
    }
    return p;
}

test "split.py's rules for GLM-5.3-Flash's tensor names" {
    const want = [_]struct { []const u8, Rule }{
        .{ prefix ++ "layers.3.mlp.experts.7.gate_proj.trellis", .dim1 },    .{ prefix ++ "layers.3.mlp.experts.7.up_proj.suh", .rep },
        .{ prefix ++ "layers.3.mlp.experts.7.up_proj.svh", .row },           .{ prefix ++ "layers.3.mlp.experts.7.down_proj.trellis", .row },
        .{ prefix ++ "layers.3.mlp.experts.7.down_proj.suh", .row },         .{ prefix ++ "layers.3.mlp.experts.7.down_proj.svh", .rep },
        .{ prefix ++ "layers.3.mlp.experts.7.down_proj.mcg", .rep },         .{ prefix ++ "layers.0.self_attn.q_proj.weight", .row },
        .{ prefix ++ "layers.0.self_attn.b_proj.weight", .row },             .{ prefix ++ "layers.0.self_attn.f_a_proj.weight", .rep },
        .{ prefix ++ "layers.0.self_attn.A_log", .row },                     .{ prefix ++ "layers.0.self_attn.o_proj.weight", .col },
        .{ prefix ++ "layers.3.self_attn.indexer.wq_b.weight", .rep },       .{ prefix ++ "layers.3.self_attn.kv_b_proj.weight", .row },
        .{ prefix ++ "layers.3.mlp.shared_experts.down_proj.weight", .col }, .{ prefix ++ "layers.3.mlp.gate.weight", .rep },
        .{ prefix ++ "layers.1.mlp.gate_proj.weight", .row },                .{ prefix ++ "layers.45.eh_proj.weight", .rep },
        .{ "lm_head.weight", .rep },                                         .{ prefix ++ "layers.7.hc_ffn_fn", .rep },
    };
    for (want) |w| try std.testing.expectEqual(w[1], try rule(w[0]));
    try std.testing.expectError(error.NoRule, rule(prefix ++ "layers.0.self_attn.mystery.weight"));
}
