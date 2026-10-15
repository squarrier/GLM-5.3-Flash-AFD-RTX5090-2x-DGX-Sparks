//! GLM-5.3-Flash's rank of the MLX checkpoint for CUDA: weights.zig's split and checks, each tensor as stored, unpacked.
const std = @import("std");
const cfg = @import("config.zig");

/// TP2's slices: weights.zig's (the shared expert and the MTP layer's experts whole) or python-0.6 split.py's (by rows too).
pub const Split = enum {
    metal,
    python,

    pub fn parse(s: []const u8) ?Split {
        return std.meta.stringToEnum(Split, s);
    }
};

pub const DType = enum { u32, bf16, f32, f16, i16, i32, other };

/// A tensor's bytes in one shard, from its safetensors header.
pub const Src = struct { shard: u32, off: u64, len: u64, dtype: DType, shape: [4]usize = @splat(1), rank: u8 };

/// The checkpoint's language-model tensors by name (core's index fills it; the reader holds the shard paths).
pub const Names = std.StringHashMapUnmanaged(Src);

/// Shard bytes [off, off + len) into group `g` at `dst`; `row` > 0: each stored `row`-byte row keeps [at, at + width).
pub const Copy = struct {
    shard: u32,
    off: u64,
    len: u64,
    g: u32,
    dst: u64,
    row: u32 = 0,
    at: u32 = 0,
    width: u32 = 0,

    /// The bytes it lands in its group.
    pub fn kept(c: Copy) u64 {
        return if (c.row == 0) c.len else c.len / c.row * c.width;
    }
};

/// A tensor's place: group `g` (one allocation), `off` bytes in.
pub const Ref = struct { g: u32, off: u64 };

/// A 4-bit affine matrix in groups of 64 as MLX stores it: w [n, k/8] u32, s and b [n, k/64] bf16 (stacked: [e, n, ...]).
pub const Q4 = struct {
    w: Ref,
    s: Ref,
    b: Ref,
    n: u32,
    k: u32,

    pub fn wBytes(n: usize, k: usize) usize {
        return n * k / 2;
    }

    pub fn sBytes(n: usize, k: usize) usize {
        return n * (k / 64) * 2;
    }
};

pub const Hc = struct { fnp: Ref, base: Ref, scale: Ref }; // fn [24, 4D] bf16 as stored (Metal packs it; CUDA reads it so)

pub const Kda = struct { in_proj: Q4, f_b: Q4, g_b: Q4, o_proj: Q4, conv: Ref, a_log: Ref, dt_bias: Ref, o_norm: Ref }; // conv [q | k | v][channel][tap] bf16

pub const Mla = struct { x_proj: Q4, qr_proj: Q4, kv_b: Q4, o_proj: Q4, q_norm: Ref, kv_norm: Ref, k_norm_w: Ref, k_norm_b: Ref, ape: Ref, igate: Ref };

pub const Dense = struct { gate_up: Q4, down: Q4 };

pub const Moe = struct { router: Ref, bias: Ref, gate: Q4, up: Q4, down: Q4, sh_gate_up: Q4, sh_down: Q4, experts: u32 };

pub const Layer = struct {
    hc: ?[2]Hc, // attention's and the MLP's hyper-connections (none on the MTP layer)
    in_norm: Ref,
    post_norm: Ref,
    attn: union(cfg.Kind) { kda: Kda, mla: Mla },
    mlp: union(enum) { dense: Dense, moe: Moe },
};

pub const Mtp = struct { eh_proj: Q4, enorm: Ref, hnorm: Ref, norm: Ref };

/// One rank's tensors: where each lands (groups of bytes) and the copies that fill them.
pub const Plan = struct {
    gpa: std.mem.Allocator,
    layers: [cfg.max_layers]Layer = undefined, // the backbone's, then the MTP layer at index c.layers
    mtp: ?Mtp = null,
    embed: Q4 = undefined,
    head: Q4 = undefined,
    norm: Ref = undefined,
    groups: std.ArrayList(u64) = .empty, // each group's bytes: the embedding and head, then per layer its dense part and its experts
    copies: std.ArrayList(Copy) = .empty,
    used: std.StringHashMapUnmanaged(void) = .empty, // the names read (keys borrowed from the index)

    pub fn deinit(p: *Plan) void {
        p.groups.deinit(p.gpa);
        p.copies.deinit(p.gpa);
        p.used.deinit(p.gpa);
    }

    /// Bytes of every group: the rank's weights as the loader leaves them.
    pub fn bytes(p: *const Plan) u64 {
        var n: u64 = 0;
        for (p.groups.items) |g| n += g;
        return n;
    }

    /// Bytes the copies take from the shards (a sliced tensor counts whole).
    pub fn sourceBytes(p: *const Plan) u64 {
        var n: u64 = 0;
        for (p.copies.items) |c| n += c.len;
        return n;
    }
};

const prefix = "model.language_model.";
/// Tensors start on this boundary, as weights.zig's `take`.
pub const alignment = 256;

const Planner = struct {
    gpa: std.mem.Allocator,
    c: *const cfg.Config,
    names: *const Names,
    p: *Plan,
    how: Split,
    at: u64 = 0, // the current group's next free byte

    /// Layer i's routed experts whole on every rank: the Metal pair's MTP layer (its drafts need no exchange there).
    fn whole(l: *const Planner, i: usize) bool {
        return i == l.c.layers and l.how == .metal;
    }

    fn src(l: *Planner, comptime fmt: []const u8, args: anytype) !Src {
        var name: [200]u8 = undefined;
        return l.named(try std.fmt.bufPrint(&name, fmt, args));
    }

    fn named(l: *Planner, full: []const u8) !Src {
        const key = l.names.getKey(full) orelse {
            std.log.err("glm: the checkpoint has no tensor {s}", .{full});
            return error.MissingTensor;
        };
        try l.p.used.put(l.gpa, key, {});
        return l.names.get(key).?;
    }

    fn expect(s: Src, name: []const u8, dtype: DType, shape: []const usize) !void {
        if (s.dtype == dtype and s.rank == shape.len and std.mem.eql(usize, s.shape[0..s.rank], shape)) return;
        std.log.err("glm: {s} is {t} {any}; the kernels read {t} {any}", .{ name, s.dtype, s.shape[0..s.rank], dtype, shape });
        return error.UnexpectedTensor;
    }

    /// A new group that the next `take`s fill.
    fn begin(l: *Planner) !void {
        try l.p.groups.append(l.gpa, 0);
        l.at = 0;
    }

    fn take(l: *Planner, len: usize) Ref {
        const g: u32 = @intCast(l.p.groups.items.len - 1);
        const r: Ref = .{ .g = g, .off = l.at };
        l.at = std.mem.alignForward(u64, l.at + len, alignment);
        l.p.groups.items[g] = l.at;
        return r;
    }

    fn copy(l: *Planner, s: Src, off: u64, len: u64, dst: Ref) !void {
        if (off + len > s.len) return error.BadSplit;
        try l.p.copies.append(l.gpa, .{ .shard = s.shard, .off = s.off + off, .len = len, .g = dst.g, .dst = dst.off });
    }

    /// A tensor as stored (bf16 norms, fp32 biases, the bf16 router and mixes), checked.
    fn plain(l: *Planner, comptime fmt: []const u8, args: anytype, dtype: DType, shape: []const usize) !Ref {
        var name: [200]u8 = undefined;
        const full = try std.fmt.bufPrint(&name, fmt, args);
        const s = try l.named(full);
        try expect(s, full, dtype, shape);
        const r = l.take(s.len);
        try l.copy(s, 0, s.len, r);
        return r;
    }

    /// 4-bit projections that read one input, stacked by rows in order; each part checked [n_i, k].
    fn q4(l: *Planner, k: usize, comptime fmt: []const u8, parts: []const []const u8, ns: []const usize, args: anytype) !Q4 {
        const zeros: [8]usize = @splat(0);
        return l.q4Rows(k, fmt, parts, zeros[0..parts.len], ns, ns, args);
    }

    /// 4-bit rows [lo, lo + n) of each part stored [all, k], stacked (weights.zig's, without its tile padding).
    fn q4Rows(l: *Planner, k: usize, comptime fmt: []const u8, parts: []const []const u8, los: []const usize, ns: []const usize, alls: []const usize, args: anytype) !Q4 {
        var n: usize = 0;
        for (ns) |x| n += x;
        const out: Q4 = .{ .w = l.take(Q4.wBytes(n, k)), .s = l.take(Q4.sBytes(n, k)), .b = l.take(Q4.sBytes(n, k)), .n = @intCast(n), .k = @intCast(k) };
        var row: usize = 0;
        for (parts, los, ns, alls) |p, lo, rows, all| {
            const comps = [_][]const u8{ "weight", "scales", "biases" };
            const dsts = [_]Ref{ at(out.w, Q4.wBytes(row, k)), at(out.s, Q4.sBytes(row, k)), at(out.b, Q4.sBytes(row, k)) };
            for (comps, dsts, 0..) |comp, dst, j| {
                var name: [200]u8 = undefined;
                const base = try std.fmt.bufPrint(&name, fmt, args);
                var full: [240]u8 = undefined;
                const nm = try std.fmt.bufPrint(&full, "{s}{s}.{s}", .{ base, p, comp });
                const s = try l.named(nm);
                try expect(s, nm, if (j == 0) .u32 else .bf16, &.{ all, if (j == 0) k / 8 else k / 64 });
                const row_bytes: usize = if (j == 0) k / 2 else k / 64 * 2;
                try l.copy(s, lo * row_bytes, rows * row_bytes, dst);
            }
            row += rows;
        }
        return out;
    }

    /// One 4-bit matrix stored [n, k_all]: its input columns [lo, lo + k), each stored row read whole and sliced.
    fn q4Cols(l: *Planner, comptime fmt: []const u8, args: anytype, n: usize, k_all: usize, lo: usize, k: usize) !Q4 {
        const out: Q4 = .{ .w = l.take(Q4.wBytes(n, k)), .s = l.take(Q4.sBytes(n, k)), .b = l.take(Q4.sBytes(n, k)), .n = @intCast(n), .k = @intCast(k) };
        const comps = [_][]const u8{ "weight", "scales", "biases" };
        for (comps, [_]Ref{ out.w, out.s, out.b }, 0..) |comp, dst, j| {
            var name: [200]u8 = undefined;
            const base = try std.fmt.bufPrint(&name, fmt, args);
            var full: [240]u8 = undefined;
            const nm = try std.fmt.bufPrint(&full, "{s}.{s}", .{ base, comp });
            const s = try l.named(nm);
            try expect(s, nm, if (j == 0) .u32 else .bf16, &.{ n, if (j == 0) k_all / 8 else k_all / 64 });
            try l.cols(s, if (j == 0) k_all / 2 else k_all / 64 * 2, if (j == 0) lo / 2 else lo / 64 * 2, if (j == 0) k / 2 else k / 64 * 2, dst);
        }
        return out;
    }

    /// Each `row`-byte stored row's bytes [first, first + width) of the whole tensor, rows in order.
    fn cols(l: *Planner, s: Src, row: usize, first: usize, width: usize, dst: Ref) !void {
        if (s.len % row != 0 or first + width > row) return error.BadSplit;
        try l.p.copies.append(l.gpa, .{ .shard = s.shard, .off = s.off, .len = s.len, .g = dst.g, .dst = dst.off, .row = @intCast(row), .at = @intCast(first), .width = @intCast(width) });
    }

    /// Elements [lo, lo + n) of a vector stored with `all` elements of `size` bytes (TP2: one rank's heads).
    fn plainRange(l: *Planner, comptime fmt: []const u8, args: anytype, dtype: DType, all: usize, lo: usize, n: usize, size: usize) !Ref {
        var name: [200]u8 = undefined;
        const full = try std.fmt.bufPrint(&name, fmt, args);
        const s = try l.named(full);
        try expect(s, full, dtype, &.{all});
        const r = l.take(n * size);
        try l.copy(s, lo * size, n * size, r);
        return r;
    }

    /// This rank's routed experts' `proj` stacked [own, n, k]; by rows, gate/up keep rows `inter` and down keeps input columns `inter`.
    fn experts(l: *Planner, i: usize, proj: []const u8, n_all: usize, k_all: usize, part: enum { whole, rows, cols }) !Q4 {
        const every = l.whole(i); // the Metal pair's MTP layer: every expert whole on every rank, as weights.zig loads it
        const lo = if (every) 0 else l.c.own[0];
        const e = (if (every) l.c.experts else l.c.own[1]) - lo;
        const in_lo: usize = l.c.inter[0];
        const in_n: usize = l.c.inter[1] - l.c.inter[0];
        const n = if (part == .rows) in_n else n_all;
        const k = if (part == .cols) in_n else k_all;
        const out: Q4 = .{ .w = l.take(e * Q4.wBytes(n, k)), .s = l.take(e * Q4.sBytes(n, k)), .b = l.take(e * Q4.sBytes(n, k)), .n = @intCast(n), .k = @intCast(k) };
        for (0..e) |x| {
            const comps = [_][]const u8{ "weight", "scales", "biases" };
            const dsts = [_]Ref{ at(out.w, x * Q4.wBytes(n, k)), at(out.s, x * Q4.sBytes(n, k)), at(out.b, x * Q4.sBytes(n, k)) };
            for (comps, dsts, 0..) |comp, dst, j| {
                const s = try l.src(prefix ++ "layers.{d}.mlp.experts.{d}.{s}.{s}", .{ i, lo + x, proj, comp });
                var name: [200]u8 = undefined;
                try expect(s, try std.fmt.bufPrint(&name, "experts.{d}.{d}.{s}.{s}", .{ i, lo + x, proj, comp }), if (j == 0) .u32 else .bf16, &.{ n_all, if (j == 0) k_all / 8 else k_all / 64 });
                const row_all: usize = if (j == 0) k_all / 2 else k_all / 64 * 2; // a stored row's bytes
                switch (part) {
                    .whole => try l.copy(s, 0, s.len, dst),
                    .rows => try l.copy(s, in_lo * row_all, in_n * row_all, dst),
                    .cols => try l.cols(s, row_all, if (j == 0) in_lo / 2 else in_lo / 64 * 2, if (j == 0) in_n / 2 else in_n / 64 * 2, dst),
                }
            }
        }
        return out;
    }

    fn hc(l: *Planner, i: usize, site: []const u8) !Hc {
        const D: usize = l.c.hidden;
        return .{
            .fnp = try l.plain(prefix ++ "layers.{d}.hc_{s}_fn", .{ i, site }, .bf16, &.{ 24, 4 * D }),
            .base = try l.plain(prefix ++ "layers.{d}.hc_{s}_base", .{ i, site }, .f32, &.{24}),
            .scale = try l.plain(prefix ++ "layers.{d}.hc_{s}_scale", .{ i, site }, .f32, &.{3}),
        };
    }

    fn kda(l: *Planner, i: usize) !Kda {
        const c = l.c;
        const D: usize = c.hidden;
        const W: usize = c.kdaWidth(); // this rank's heads' channels
        const H: usize = c.kda_heads;
        const dk: usize = c.kda_dim;
        const h0: usize = c.tp_rank * H; // TP2: this rank's first head (0 on one GPU)
        const ch0 = h0 * dk;
        const Wa = W * c.tp; // every head's channels, as stored
        const Ha = H * c.tp;
        const a = prefix ++ "layers.{d}.self_attn.";
        const row: usize = c.conv * 2; // a channel's taps
        const conv = l.take(3 * W * row);
        for ([_][]const u8{ "q", "k", "v" }, 0..) |x, part| {
            var name: [200]u8 = undefined;
            const full = try std.fmt.bufPrint(&name, a ++ "{s}_conv1d.weight", .{ i, x });
            const s = try l.named(full);
            try expect(s, full, .bf16, &.{ Wa, 1, c.conv });
            try l.copy(s, ch0 * row, W * row, at(conv, part * W * row));
        }
        return .{
            .in_proj = try l.q4Rows(D, a, &.{ "q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj" }, &.{ ch0, ch0, ch0, 0, 0, h0 }, &.{ W, W, W, dk, dk, H }, &.{ Wa, Wa, Wa, dk, dk, Ha }, .{i}),
            .f_b = try l.q4Rows(dk, a, &.{"f_b_proj"}, &.{ch0}, &.{W}, &.{Wa}, .{i}),
            .g_b = try l.q4Rows(dk, a, &.{"g_b_proj"}, &.{ch0}, &.{W}, &.{Wa}, .{i}),
            .o_proj = if (c.tp == 1) try l.q4(W, a, &.{"o_proj"}, &.{D}, .{i}) else try l.q4Cols(a ++ "o_proj", .{i}, D, Wa, ch0, W),
            .conv = conv,
            .a_log = try l.plainRange(a ++ "A_log", .{i}, .f32, Ha, h0, H, 4),
            .dt_bias = try l.plainRange(a ++ "dt_bias", .{i}, .f32, Wa, ch0, W, 4),
            .o_norm = try l.plain(a ++ "o_norm.weight", .{i}, .bf16, &.{dk}),
        };
    }

    fn mla(l: *Planner, i: usize) !Mla {
        const c = l.c;
        const D: usize = c.hidden;
        const a = prefix ++ "layers.{d}.self_attn.";
        const H: usize = c.mla_heads; // this rank's heads
        const h0: usize = c.tp_rank * H; // TP2: its first head (0 on one GPU)
        const Ha = H * c.tp; // every head, as stored
        return .{
            .x_proj = try l.q4Rows(D, a, &.{ "q_a_proj", "kv_a_proj_with_mqa", "indexer.wk", "indexer.weights_proj" }, &.{ 0, 0, 0, 0 }, &.{ c.q_lora, c.kv_lora, c.i_dim, c.i_heads }, &.{ c.q_lora, c.kv_lora, c.i_dim, c.i_heads }, .{i}),
            .qr_proj = try l.q4Rows(c.q_lora, a, &.{ "q_b_proj", "indexer.wq_b" }, &.{ h0 * c.nope, 0 }, &.{ H * c.nope, c.i_heads * c.i_dim }, &.{ Ha * c.nope, c.i_heads * c.i_dim }, .{i}),
            .kv_b = try l.q4Rows(c.kv_lora, a, &.{"kv_b_proj"}, &.{h0 * (c.nope + c.v_dim)}, &.{H * (c.nope + c.v_dim)}, &.{Ha * (c.nope + c.v_dim)}, .{i}),
            .o_proj = if (c.tp == 1) try l.q4(H * c.v_dim, a, &.{"o_proj"}, &.{D}, .{i}) else try l.q4Cols(a ++ "o_proj", .{i}, D, Ha * c.v_dim, h0 * c.v_dim, H * c.v_dim),
            .q_norm = try l.plain(a ++ "q_a_layernorm.weight", .{i}, .bf16, &.{c.q_lora}),
            .kv_norm = try l.plain(a ++ "kv_a_layernorm.weight", .{i}, .bf16, &.{c.kv_lora}),
            .k_norm_w = try l.plain(a ++ "indexer.k_norm.weight", .{i}, .bf16, &.{c.i_dim}),
            .k_norm_b = try l.plain(a ++ "indexer.k_norm.bias", .{i}, .bf16, &.{c.i_dim}),
            .ape = try l.plain(a ++ "indexer.index_kpool_compress_ape", .{i}, .bf16, &.{ c.kpool, c.i_dim }),
            .igate = try l.plain(a ++ "indexer.index_kpool_compress_gate", .{i}, .bf16, &.{ c.i_dim, D }),
        };
    }

    fn layer(l: *Planner, i: usize) !Layer {
        const c = l.c;
        const D: usize = c.hidden;
        const plain_layer = i == c.layers; // the MTP block: no hyper-connections
        try l.begin();
        var out: Layer = .{
            .hc = if (plain_layer) null else .{ try l.hc(i, "attn"), try l.hc(i, "ffn") },
            .in_norm = try l.plain(prefix ++ "layers.{d}.input_layernorm.weight", .{i}, .bf16, &.{D}),
            .post_norm = try l.plain(prefix ++ "layers.{d}.post_attention_layernorm.weight", .{i}, .bf16, &.{D}),
            .attn = switch (c.kind(@intCast(i))) {
                .kda => .{ .kda = try l.kda(i) },
                .mla => .{ .mla = try l.mla(i) },
            },
            .mlp = undefined,
        };
        if (plain_layer) { // the MTP head's own tensors, in the layer's group (before the experts begin theirs)
            const p = prefix ++ "layers.{d}.";
            l.p.mtp = .{
                .eh_proj = try l.q4(2 * D, p, &.{"eh_proj"}, &.{D}, .{i}),
                .enorm = try l.plain(p ++ "enorm.weight", .{i}, .bf16, &.{D}),
                .hnorm = try l.plain(p ++ "hnorm.weight", .{i}, .bf16, &.{D}),
                .norm = try l.plain(p ++ "shared_head.norm.weight", .{i}, .bf16, &.{D}),
            };
        }
        const m = prefix ++ "layers.{d}.mlp.";
        if (c.isMoe(@intCast(i))) {
            const in_lo: usize = c.inter[0];
            const in_n: usize = c.inter[1] - c.inter[0];
            const sh_rows = l.how == .python and c.byRows(); // split.py: the shared expert by rows like the routed ones
            var moe: Moe = .{
                .router = try l.plain(m ++ "gate.weight", .{i}, .bf16, &.{ c.experts, D }),
                .bias = try l.plain(m ++ "gate.e_score_correction_bias", .{i}, .f32, &.{c.experts}),
                .sh_gate_up = if (sh_rows) try l.q4Rows(D, m ++ "shared_experts.", &.{ "gate_proj", "up_proj" }, &.{ in_lo, in_lo }, &.{ in_n, in_n }, &.{ c.moe_inter, c.moe_inter }, .{i}) else try l.q4(D, m ++ "shared_experts.", &.{ "gate_proj", "up_proj" }, &.{ c.moe_inter, c.moe_inter }, .{i}),
                .sh_down = if (sh_rows) try l.q4Cols(m ++ "shared_experts.down_proj", .{i}, D, c.moe_inter, in_lo, in_n) else try l.q4(c.moe_inter, m ++ "shared_experts.", &.{"down_proj"}, &.{D}, .{i}),
                .gate = undefined,
                .up = undefined,
                .down = undefined,
                .experts = if (l.whole(i)) c.experts else c.own[1] - c.own[0],
            };
            try l.begin();
            const by_rows = c.byRows() and !l.whole(i);
            moe.gate = try l.experts(i, "gate_proj", c.moe_inter, D, if (by_rows) .rows else .whole);
            moe.up = try l.experts(i, "up_proj", c.moe_inter, D, if (by_rows) .rows else .whole);
            moe.down = try l.experts(i, "down_proj", D, c.moe_inter, if (by_rows) .cols else .whole);
            out.mlp = .{ .moe = moe };
        } else {
            const n: usize = c.dense_inter; // this rank's rows; TP2: rank r's half [r n, (r + 1) n)
            const lo = c.tp_rank * n;
            out.mlp = .{ .dense = .{
                .gate_up = try l.q4Rows(D, m, &.{ "gate_proj", "up_proj" }, &.{ lo, lo }, &.{ n, n }, &.{ n * c.tp, n * c.tp }, .{i}),
                .down = if (c.tp == 1) try l.q4(n, m, &.{"down_proj"}, &.{D}, .{i}) else try l.q4Cols(m ++ "down_proj", .{i}, D, n * c.tp, lo, n),
            } };
        }
        return out;
    }
};

fn at(r: Ref, bytes: usize) Ref {
    return .{ .g = r.g, .off = r.off + bytes };
}

/// The first `c.run` layers, the MTP layer (when stored) and the head in `how`'s slices: every name, dtype and shape checked.
pub fn plan(gpa: std.mem.Allocator, c: *const cfg.Config, names: *const Names, how: Split) !Plan {
    var p: Plan = .{ .gpa = gpa };
    errdefer p.deinit();
    var l: Planner = .{ .gpa = gpa, .c = c, .names = names, .p = &p, .how = how };
    const D: usize = c.hidden;
    const V: usize = c.vocab;
    try l.begin();
    p.embed = try l.q4(D, prefix ++ "embed_tokens", &.{""}, &.{V}, .{});
    const vp = c.vocabPart(); // TP2: this rank's half of the head's rows
    p.head = try l.q4Rows(D, "lm_head", &.{""}, &.{vp[0]}, &.{vp[1] - vp[0]}, &.{V}, .{});
    p.norm = try l.plain(prefix ++ "norm.weight", .{}, .bf16, &.{D});
    for (0..c.run) |i| p.layers[i] = try l.layer(i);
    var mtp_name: [96]u8 = undefined;
    const has_mtp = c.mtp > 0 and names.contains(try std.fmt.bufPrint(&mtp_name, prefix ++ "layers.{d}.eh_proj.weight", .{c.layers}));
    if (has_mtp) p.layers[c.layers] = try l.layer(c.layers);
    return p;
}

/// The engine's TP2 config for `rank`, as engine.zig sets it for the Mac pair: experts by rows, heads, dense rows and head by halves.
pub fn rankConfig(c: cfg.Config, rank: u32, ranks: u32) !cfg.Config {
    var out = c;
    if (ranks == 1) return out;
    try cfg.splitRows(&out, rank, ranks);
    try cfg.splitHeads(&out, rank, ranks);
    return out;
}

test "copies keep whole bytes, or each stored row's slice" {
    const whole: Copy = .{ .shard = 0, .off = 0, .len = 4096, .g = 0, .dst = 0 };
    try std.testing.expectEqual(@as(u64, 4096), whole.kept());
    const half: Copy = .{ .shard = 0, .off = 0, .len = 4096, .g = 0, .dst = 0, .row = 1024, .at = 512, .width = 512 };
    try std.testing.expectEqual(@as(u64, 2048), half.kept());
}

test "TP2's rank configs split the experts by rows and the heads, as engine.zig does" {
    const c0 = try rankConfig(.{}, 0, 2);
    const c1 = try rankConfig(.{}, 1, 2);
    try std.testing.expectEqual([2]u32{ 0, 1024 }, c0.inter);
    try std.testing.expectEqual([2]u32{ 1024, 2048 }, c1.inter);
    try std.testing.expectEqual(@as(u32, 32), c1.kda_heads);
    try std.testing.expectEqual(@as(u32, 32), c1.mla_heads);
    try std.testing.expectEqual(@as(u32, 6144), c1.dense_inter);
    try std.testing.expectEqual(@as(u32, 1), c1.tp_rank);
    try std.testing.expect(c0.byRows());
}
