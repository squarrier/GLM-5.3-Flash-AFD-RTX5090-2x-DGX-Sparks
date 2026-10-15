//! GLM-5.3-Flash as python-0.6's CUDA engine reads it (glm5_next/cuda/weights.py Config, ed78d6f): config.zig's checked
//! parse of config.json, the quantization and norm_topk_prob it leaves out, a TP rank's shares, and the layers a run loads.
const std = @import("std");
const cfg = @import("config.zig");

/// weights.py's `quant`: "mlx" (affine 4-bit everywhere) or "exl3" (EXL3 routed experts, BF16 elsewhere).
pub const Quant = enum { mlx, exl3 };
/// Config.kinds: KDA (linear attention) or DSA (MLA with the indexer).
pub const Kind = enum { kda, dsa };
/// Config.mlp_kinds
pub const MlpKind = enum { dense, moe };

pub const max_layers = cfg.max_layers;

/// python-0.6's Config, field for field (names as there), plus the rank this process runs of `world`.
pub const Model = struct {
    hidden: usize,
    layers: usize, // backbone layers loaded (Config cut to the first n, as a layer check does)
    vocab: usize,
    eps: f32,
    heads: usize,
    q_lora: usize,
    kv_lora: usize,
    qk_dim: usize,
    v_dim: usize,
    lin_heads: usize,
    lin_dim: usize,
    conv: usize,
    lower: f32,
    experts: usize,
    top_k: usize,
    moe_width: usize,
    shared_width: usize,
    dense_width: usize,
    routed_scale: f32,
    norm_topk: bool,
    streams: usize,
    hc_iters: usize,
    hc_eps: f32,
    index_heads: usize,
    index_dim: usize,
    index_topk: usize,
    kpool: usize,
    limit: f32,
    kinds: [max_layers]Kind,
    mlp_kinds: [max_layers]MlpKind,
    eos: [3]u32,
    eos_n: usize,
    mtp_layers: usize,
    group_size: usize,
    bits: usize,
    quant: Quant,
    world: usize = 1,
    rank: usize = 0,

    /// Config.dense_limit: the largest context whose top-k keeps every visible key.
    pub fn denseLimit(m: *const Model) usize {
        return m.index_topk + m.kpool - 1;
    }
    /// A rank's DSA heads (c.heads // w.world)
    pub fn hl(m: *const Model) usize {
        return m.heads / m.world;
    }
    /// A rank's KDA heads (c.lin_heads // w.world)
    pub fn ll(m: *const Model) usize {
        return m.lin_heads / m.world;
    }
    /// A rank's dense MLP, routed expert and shared expert widths, and vocabulary rows
    pub fn dl(m: *const Model) usize {
        return m.dense_width / m.world;
    }
    pub fn ml(m: *const Model) usize {
        return m.moe_width / m.world;
    }
    pub fn sl(m: *const Model) usize {
        return m.shared_width / m.world;
    }
    pub fn vl(m: *const Model) usize {
        return m.vocab / m.world;
    }
    /// KDAW.proj's rows: q | k | v (a rank's heads), f_a and g_a whole, b (a rank's heads)
    pub fn kdaWidth(m: *const Model) usize {
        return 3 * m.ll() * m.lin_dim + 2 * m.lin_dim + m.ll();
    }
    /// Layers of a kind among the first `n` (State's per-kind arenas)
    pub fn count(m: *const Model, k: Kind, n: usize) usize {
        var c: usize = 0;
        for (m.kinds[0..n]) |x| c += @intFromBool(x == k);
        return c;
    }
    /// State.kda_index / dsa_index: layer i's place among the loaded layers of its kind
    pub fn kindIndex(m: *const Model, i: usize) usize {
        return m.count(m.kinds[i], i);
    }

    /// From config.zig's parse (the whole model, unsplit) and the fields it does not read.
    pub fn from(c: *const cfg.Config, x: Extra, world: usize, rank: usize) !Model {
        if (world == 0 or rank >= world or c.mla_heads % world != 0 or c.kda_heads % world != 0) return error.BadRankSplit;
        var m: Model = .{
            .hidden = c.hidden, .layers = c.run, .vocab = c.vocab, .eps = c.eps, .heads = c.mla_heads,
            .q_lora = c.q_lora, .kv_lora = c.kv_lora, .qk_dim = c.nope, .v_dim = c.v_dim, .lin_heads = c.kda_heads,
            .lin_dim = c.kda_dim, .conv = c.conv, .lower = c.lower_bound, .experts = c.experts, .top_k = c.topk,
            .moe_width = c.moe_inter, .shared_width = c.moe_inter * x.shared_experts, .dense_width = c.dense_inter,
            .routed_scale = c.routed_scale, .norm_topk = x.norm_topk, .streams = c.hc, .hc_iters = c.sinkhorn,
            .hc_eps = c.hc_eps, .index_heads = c.i_heads, .index_dim = c.i_dim, .index_topk = c.i_topk,
            .kpool = c.kpool, .limit = c.swiglu_limit, .kinds = undefined, .mlp_kinds = undefined, .eos = c.eos,
            .eos_n = c.eos_n, .mtp_layers = c.mtp, .group_size = x.group_size, .bits = x.bits, .quant = x.quant,
            .world = world, .rank = rank,
        };
        for (0..max_layers) |i| {
            m.kinds[i] = if (c.mla.isSet(i)) .dsa else .kda;
            m.mlp_kinds[i] = if (i < c.dense_layers) .dense else .moe;
        }
        return m;
    }

    /// The first `n` backbone layers (python's Config cut there, MTP off): a check that loads only those.
    pub fn cut(m: *Model, n: usize) !void {
        if (n == 0 or n > m.layers) return error.BadLayerCount;
        m.layers = n;
        m.mtp_layers = 0;
    }

    /// config.json's text: config.zig's checked parse, then the quantization and the MoE fields it leaves out.
    pub fn parse(gpa: std.mem.Allocator, json: []const u8, world: usize, rank: usize) !Model {
        const c = try cfg.parse(gpa, json);
        return from(&c, try Extra.parse(gpa, json), world, rank);
    }
};

/// What weights.py's Config.read takes beyond config.zig: quantization (bits, group size, method), norm_topk_prob,
/// n_shared_experts; and its refusals (a float32 activation dtype, mixed-bit encodes, other methods).
pub const Extra = struct {
    bits: usize = 4,
    group_size: usize = 64,
    quant: Quant = .mlx,
    norm_topk: bool = true,
    shared_experts: usize = 1,

    pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Extra {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        if (root.get("tensorfold_activation_dtype")) |v| switch (v) {
            .null => {},
            .string => |s| if (!std.mem.eql(u8, s, "bfloat16")) return error.ActivationDtype,
            else => return error.ActivationDtype,
        };
        const t = (root.get("text_config") orelse parsed.value).object;
        var x: Extra = .{};
        if (t.get("norm_topk_prob")) |v| x.norm_topk = v == .bool and v.bool;
        if (t.get("n_shared_experts")) |v| x.shared_experts = @intCast(v.integer);
        const q = root.get("quantization") orelse root.get("quantization_config") orelse return x;
        if (q != .object) return x;
        if (q.object.get("group_size")) |v| x.group_size = @intCast(v.integer);
        if (q.object.get("bits")) |v| x.bits = switch (v) {
            .integer => |i| @intCast(i),
            .string => |s| std.fmt.parseInt(usize, s, 10) catch return error.MixedBits, // bits_of: "mixed_k34..." refused
            else => return error.MixedBits,
        };
        if (q.object.get("quant_method")) |v| {
            if (v != .string) return error.UnsupportedQuant;
            x.quant = if (std.ascii.eqlIgnoreCase(v.string, "mlx")) .mlx else if (std.ascii.eqlIgnoreCase(v.string, "exl3")) .exl3 else return error.UnsupportedQuant;
        }
        return x;
    }
};

test "the shipped shape: rank shares and the dense limit" {
    var m = try Model.from(&cfg.Config{}, .{}, 2, 1);
    try std.testing.expectEqual(@as(usize, 32), m.hl());
    try std.testing.expectEqual(@as(usize, 12576), m.kdaWidth());
    try std.testing.expectEqual(@as(usize, 6144), m.dl());
    try std.testing.expectEqual(@as(usize, 77440), m.vl());
    try std.testing.expectEqual(@as(usize, 2051), m.denseLimit());
    try std.testing.expectEqual(Kind.dsa, m.kinds[3]);
    try std.testing.expectEqual(@as(usize, 10), m.kindIndex(43));
    try m.cut(3);
    try std.testing.expectEqual(@as(usize, 3), m.count(.kda, m.layers));
    try std.testing.expectError(error.BadLayerCount, m.cut(4));
    try std.testing.expectError(error.BadRankSplit, Model.from(&cfg.Config{}, .{}, 2, 2));
}

test "Extra: python-0.6's refusals" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.MixedBits, Extra.parse(gpa, "{\"quantization\":{\"bits\":\"mixed_k34_per_tensor\"}}"));
    try std.testing.expectError(error.ActivationDtype, Extra.parse(gpa, "{\"tensorfold_activation_dtype\":\"float32\"}"));
    try std.testing.expectError(error.UnsupportedQuant, Extra.parse(gpa, "{\"quantization\":{\"quant_method\":\"gptq\"}}"));
    const e = try Extra.parse(gpa, "{\"quantization_config\":{\"quant_method\":\"EXL3\",\"bits\":\"4\"},\"text_config\":{\"norm_topk_prob\":false}}");
    try std.testing.expectEqual(Quant.exl3, e.quant);
    try std.testing.expectEqual(@as(usize, 4), e.bits);
    try std.testing.expect(!e.norm_topk);
}
