//! The pack pass of an EXL3 rank (qualification_qualification_7c75d777 PLAN §3, qualification_qualification_cec53599): qualification_765330ce's split of Mia's 078455ff (cuda_exl3_split.zig, called as
//! it is) into python-0.6's load-time tensors (weights.py `load` with exl3): make_b16 of each BF16 part, stack_b16
//! and torch.cat of several, kv_b's key and value rows by head and latent.AbsorbW's [H, rows, latent] copies of them,
//! experts.prepare's tables and suh/svh stacks with each trellis left where the split put it (python keeps each as its
//! own tensor, `keep[i]`: gate's, up's, then down's), and qmm.quantize4's draft head. Every buffer carries python's
//! attribute path, in its walk's order, as qualification_2b1188e0's weights.json names it (the 18 `*_ptr` tables hold addresses).
const std = @import("std");
const cfg = @import("config.zig");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const split = @import("cuda_split.zig");
const xs = @import("cuda.zig").exl3.split; // qualification_765330ce's cuda_exl3_split.zig, by the module: never loaded where it is missing
const pack = @import("cuda_pack.zig");
const xo = pack.x_ops;

const Ref = split.Ref;
const Buf = views.Buf;
const pre = "model.language_model.";

const Builder = struct {
    a: std.mem.Allocator,
    p: *pack.Plan,
    m: *const md.Model,
    x: *const xs.Plan,

    fn sub(b: *Builder, path: []const u8, leaf: []const u8) ![]const u8 {
        return std.fmt.allocPrint(b.a, "{s}.{s}", .{ path, leaf });
    }

    /// The split's part of checkpoint tensor `name` (a full name, or `pre` ++ a layer's `layers.{i}.` ++ `leaf`).
    fn ref(b: *Builder, i: ?usize, leaf: []const u8) !Ref {
        var buf: [192]u8 = undefined;
        const name = if (i) |n| try std.fmt.bufPrint(&buf, pre ++ "layers.{d}.{s}", .{ n, leaf }) else leaf;
        return b.x.tensors.get(name) orelse {
            std.log.err("glm pack exl3: the split holds no {s}", .{name});
            return error.MissingTensor;
        };
    }

    fn item(b: *Builder, path: []const u8, leaf: []const u8, buf: Buf, op: pack.Op, x: ?xo.Op) !void {
        const full = if (leaf.len == 0) try b.a.dupe(u8, path) else try b.sub(path, leaf);
        const home: u32 = if (x) |o| xo.home(o) else switch (op) {
            .copy => |r| r.g,
            .rows => |r| r.src.g,
            else => return error.NotAnExl3Op,
        };
        try b.p.items.append(b.a, .{ .path = full, .buf = buf, .op = op, .home = home, .x = x });
    }

    /// make_b16 / t(): a stored tensor as it is.
    fn copy(b: *Builder, path: []const u8, leaf: []const u8, i: ?usize, name: []const u8, dtype: views.DType, shape: []const usize) !void {
        try b.item(path, leaf, Buf.of(dtype, shape), .{ .copy = try b.ref(i, name) }, null);
    }

    /// stack_b16 / torch.cat: layer i's stored tensors `names` back to back, `rows[j]` rows of `k` bf16 each.
    fn cat(b: *Builder, path: []const u8, leaf: []const u8, i: usize, names: []const []const u8, rows: []const usize, k: usize) !void {
        var c: xo.Cat = .{ .n = @intCast(names.len) };
        var n: usize = 0;
        for (names, rows, 0..) |name, r, j| {
            c.srcs[j] = try b.ref(i, name);
            c.lens[j] = r * k * 2;
            n += r;
        }
        try b.item(path, leaf, Buf.of(.bf16, &.{ n, k }), .{ .copy = c.srcs[0] }, .{ .cat = c });
    }

    /// Stored bytes left in place (experts.prepare's stacks and trellises), `off` bytes past the split's ref.
    fn alias(b: *Builder, path: []const u8, leaf: []const u8, r: Ref, off: u64, dtype: views.DType, shape: []const usize) !void {
        const at: Ref = .{ .g = r.g, .off = r.off + off };
        try b.item(path, leaf, Buf.of(dtype, shape), .{ .copy = at }, .{ .alias = at });
    }

    fn layer(b: *Builder, path: []const u8, i: usize, mtp: bool) !void {
        const m = b.m;
        const D = m.hidden;
        if (!mtp) for ([_][]const u8{ "attn", "ffn" }) |site| {
            const p = try std.fmt.allocPrint(b.a, "{s}.{s}_hc", .{ path, site });
            var n: [32]u8 = undefined;
            try b.copy(p, "fn", i, try std.fmt.bufPrint(&n, "hc_{s}_fn", .{site}), .bf16, &.{ 24, m.streams * D });
            try b.copy(p, "base", i, try std.fmt.bufPrint(&n, "hc_{s}_base", .{site}), .f32, &.{24});
            try b.copy(p, "scale", i, try std.fmt.bufPrint(&n, "hc_{s}_scale", .{site}), .f32, &.{3});
        };
        try b.copy(path, "in_norm", i, "input_layernorm.weight", .bf16, &.{D});
        try b.copy(path, "post_norm", i, "post_attention_layernorm.weight", .bf16, &.{D});
        if (mtp or m.kinds[i] == .dsa) try b.dsa(try b.sub(path, "dsa"), i) else try b.kda(try b.sub(path, "kda"), i);
        if (mtp or m.mlp_kinds[i] == .moe) return b.moe(try b.sub(path, "moe"), i);
        const p = try b.sub(path, "mlp");
        const dl = m.dl();
        try b.cat(p, "gu.weight", i, &.{ "mlp.gate_proj.weight", "mlp.up_proj.weight" }, &.{ dl, dl }, D);
        try b.copy(p, "down.weight", i, "mlp.down_proj.weight", .bf16, &.{ D, dl });
    }

    /// weights.py kda(i): KDAW with BF16 projections.
    fn kda(b: *Builder, p: []const u8, i: usize) !void {
        const m = b.m;
        const L = m.ll();
        const W = L * m.lin_dim;
        const a = "self_attn.";
        try b.cat(p, "proj.weight", i, &.{ a ++ "q_proj.weight", a ++ "k_proj.weight", a ++ "v_proj.weight", a ++ "f_a_proj.weight", a ++ "g_a_proj.weight", a ++ "b_proj.weight" }, &.{ W, W, W, m.lin_dim, m.lin_dim, L }, m.hidden);
        try b.copy(p, "fb.weight", i, a ++ "f_b_proj.weight", .bf16, &.{ W, m.lin_dim });
        try b.copy(p, "gb.weight", i, a ++ "g_b_proj.weight", .bf16, &.{ W, m.lin_dim });
        try b.cat(p, "conv", i, &.{ a ++ "q_conv1d.weight", a ++ "k_conv1d.weight", a ++ "v_conv1d.weight" }, &.{ W, W, W }, m.conv);
        try b.copy(p, "a_log", i, a ++ "A_log", .f32, &.{L});
        try b.copy(p, "dt_bias", i, a ++ "dt_bias", .f32, &.{W});
        try b.copy(p, "norm", i, a ++ "o_norm.weight", .bf16, &.{m.lin_dim});
        try b.copy(p, "o.weight", i, a ++ "o_proj.weight", .bf16, &.{ m.hidden, W });
    }

    /// weights.py dsa(i): DSAW and IndexW with BF16 projections, kv_b's rows by head, latent.AbsorbW.
    fn dsa(b: *Builder, p: []const u8, i: usize) !void {
        const m = b.m;
        const H = m.hl();
        const D = m.hidden;
        const per = m.qk_dim + m.v_dim;
        const keys: pack.Rows = .{ .block = m.qk_dim, .stride = per };
        const vals: pack.Rows = .{ .first = m.qk_dim, .block = m.v_dim, .stride = per };
        const a = "self_attn.";
        try b.cat(p, "proj.weight", i, &.{ a ++ "q_a_proj.weight", a ++ "kv_a_proj_with_mqa.weight" }, &.{ m.q_lora, m.kv_lora }, D);
        try b.copy(p, "q_norm", i, a ++ "q_a_layernorm.weight", .bf16, &.{m.q_lora});
        try b.copy(p, "kv_norm", i, a ++ "kv_a_layernorm.weight", .bf16, &.{m.kv_lora});
        try b.copy(p, "q_b.weight", i, a ++ "q_b_proj.weight", .bf16, &.{ H * m.qk_dim, m.q_lora });
        const kv = try b.ref(i, a ++ "kv_b_proj.weight");
        const row = m.kv_lora * 2;
        try b.item(p, "kv_k.weight", Buf.of(.bf16, &.{ H * m.qk_dim, m.kv_lora }), .{ .rows = .{ .src = kv, .n = H * m.qk_dim, .bytes = row, .rows = keys } }, null);
        try b.item(p, "kv_v.weight", Buf.of(.bf16, &.{ H * m.v_dim, m.kv_lora }), .{ .rows = .{ .src = kv, .n = H * m.v_dim, .bytes = row, .rows = vals } }, null);
        try b.copy(p, "o.weight", i, a ++ "o_proj.weight", .bf16, &.{ D, H * m.v_dim });
        const ix = try b.sub(p, "index");
        try b.cat(ix, "kw.weight", i, &.{ a ++ "indexer.wk.weight", a ++ "indexer.weights_proj.weight" }, &.{ m.index_dim, m.index_heads }, D);
        try b.copy(ix, "qb.weight", i, a ++ "indexer.wq_b.weight", .bf16, &.{ m.index_heads * m.index_dim, m.q_lora });
        try b.copy(ix, "ln_w", i, a ++ "indexer.k_norm.weight", .bf16, &.{m.index_dim});
        try b.copy(ix, "ln_b", i, a ++ "indexer.k_norm.bias", .bf16, &.{m.index_dim});
        try b.copy(ix, "gate", i, a ++ "indexer.index_kpool_compress_gate", .bf16, &.{ m.index_dim, D });
        try b.copy(ix, "ape", i, a ++ "indexer.index_kpool_compress_ape", .bf16, &.{ m.kpool, m.index_dim });
        // latent.AbsorbW.from_rows(w[krows].float(), w[vrows].float(), HL): the same rows, bf16, [H, rows, latent]
        const ab = try b.sub(p, "absorb");
        try b.item(ab, "wk", Buf.of(.bf16, &.{ H, m.qk_dim, m.kv_lora }), .{ .rows = .{ .src = kv, .n = H * m.qk_dim, .bytes = row, .rows = keys } }, null);
        try b.item(ab, "wv", Buf.of(.bf16, &.{ H, m.v_dim, m.kv_lora }), .{ .rows = .{ .src = kv, .n = H * m.v_dim, .bytes = row, .rows = vals } }, null);
    }

    /// weights.py moe(i) with exl3: the router, its bias, experts.prepare's Exl3RoutedExperts over qualification_765330ce's stacks
    /// (tables, suh/svh, each trellis in place), then the shared expert as a BF16 MLP.
    fn moe(b: *Builder, p: []const u8, i: usize) !void {
        const m = b.m;
        const D = m.hidden;
        const E = m.experts;
        try b.copy(p, "router", i, "mlp.gate.weight", .bf16, &.{ E, D });
        try b.copy(p, "bias", i, "mlp.gate.e_score_correction_bias", .f32, &.{E});
        const e = b.x.experts[i] orelse return error.NoExperts;
        if (e.count != E) return error.ExpertCount;
        // one group holds the layer's tables, stacks and trellises: Plan.tables writes the trellises' addresses from
        // the tables' group base (as qualification_765330ce's bind does)
        for ([_]xs.Proj{ e.gate, e.up, e.down }) |pr| if (pr.trellis.g != e.tables.g or pr.suh.g != e.tables.g or pr.svh.g != e.tables.g)
            return error.ExpertsSpanGroups;
        const ex = try b.sub(p, "experts");
        const t = e.tables;
        try b.item(ex, "gate_ptr", Buf.of(.i64, &.{E}), .{ .copy = t }, .{ .tables = .{ .dst = t, .layer = @intCast(i), .bytes = @intCast(36 * E), .ctx = b.x, .write = writeTables } });
        for ([_][]const u8{ "up_ptr", "down_ptr" }, 1..) |leaf, j| try b.alias(ex, leaf, t, 8 * E * j, .i64, &.{E});
        for ([_][]const u8{ "gate_k2", "up_k2", "down_k2" }, 0..) |leaf, j| try b.alias(ex, leaf, t, 24 * E + 4 * E * j, .i32, &.{E});
        try b.alias(ex, "suh_g", e.gate.suh, 0, .f16, &.{ E, e.gate.k });
        try b.alias(ex, "suh_u", e.up.suh, 0, .f16, &.{ E, e.up.k });
        try b.alias(ex, "svh_g", e.gate.svh, 0, .f16, &.{ E, e.gate.n });
        try b.alias(ex, "svh_u", e.up.svh, 0, .f16, &.{ E, e.up.n });
        try b.alias(ex, "suh_d", e.down.suh, 0, .f16, &.{ E, e.down.k });
        try b.alias(ex, "svh_d", e.down.svh, 0, .f16, &.{ E, e.down.n });
        for ([_]xs.Proj{ e.gate, e.up, e.down }, 0..) |pr, j| for (0..E) |x| {
            var leaf: [32]u8 = undefined;
            const name = try std.fmt.bufPrint(&leaf, "keep[{d}]", .{j * E + x});
            try b.alias(ex, name, pr.trellis, pr.offsets[x], .i16, &.{ pr.k / 16, pr.n / 16, 8 * @as(usize, pr.k2s[x]) });
        };
        const sh = try b.sub(p, "shared");
        const sl = m.sl();
        try b.cat(sh, "gu.weight", i, &.{ "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight" }, &.{ sl, sl }, D);
        try b.copy(sh, "down.weight", i, "mlp.shared_experts.down_proj.weight", .bf16, &.{ D, sl });
    }
};

/// qualification_765330ce's Plan.tables for the pack's table write: layer i's pointers and widths, its group at `base`.
fn writeTables(ctx: *const anyopaque, i: usize, base: u64, out: []u8) void {
    const x: *const xs.Plan = @ptrCast(@alignCast(ctx));
    x.tables(i, base, out);
}

/// Every buffer python-0.6's weights.load leaves for this EXL3 rank, from qualification_765330ce's split `x` of the config `c` (its
/// first `c.run` layers and, when stored, the MTP layer `c.layers`): the embedding, the layers, the norm, the head's
/// rows, `Weights.mtp`, and the draft head (python builds it whether MTP is on or not). `x` must outlive the plan.
pub fn plan(gpa: std.mem.Allocator, m: *const md.Model, c: *const cfg.Config, x: *const xs.Plan) !pack.Plan {
    if (m.quant != .exl3) return error.NotExl3;
    var out: pack.Plan = .{ .arena = .init(gpa), .groups = x.groups.items.len };
    errdefer out.deinit();
    var b: Builder = .{ .a = out.arena.allocator(), .p = &out, .m = m, .x = x };
    const D = m.hidden;
    try b.copy("w.embed", "", null, pre ++ "embed_tokens.weight", .bf16, &.{ m.vocab, D });
    for (0..m.layers) |i| try b.layer(try std.fmt.allocPrint(b.a, "w.layers[{d}]", .{i}), i, false);
    try b.copy("w.norm", "", null, pre ++ "norm.weight", .bf16, &.{D});
    try b.copy("w.head", "weight", null, "lm_head.weight", .bf16, &.{ m.vl(), D });
    const L = c.layers;
    if (m.mtp_layers > 0 and x.experts[L] != null) {
        try b.copy("w.mtp", "enorm", L, "enorm.weight", .bf16, &.{D});
        try b.copy("w.mtp", "hnorm", L, "hnorm.weight", .bf16, &.{D});
        try b.copy("w.mtp", "eh.weight", L, "eh_proj.weight", .bf16, &.{ D, 2 * D });
        try b.copy("w.mtp", "norm", L, "shared_head.norm.weight", .bf16, &.{D});
        try b.layer("w.mtp.layer", L, true);
    }
    // quantize4(head.weight): the head's stored rows are the bytes make_b16 copied
    const head = try b.ref(null, "lm_head.weight");
    const q = views.Q4.of(m.vl(), D);
    for ([_]Buf{ q.weight, q.scales, q.biases }, [_][]const u8{ "weight", "scales", "biases" }, 0..) |buf, leaf, j| {
        try b.item("w.draft_head", leaf, buf, .{ .copy = head }, .{ .quant = .{ .src = head, .n = m.vl(), .k = D, .part = @enumFromInt(j) } });
    }
    return out;
}
