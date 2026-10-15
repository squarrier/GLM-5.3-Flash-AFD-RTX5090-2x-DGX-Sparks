//! python-0.6's weight dataclasses (glm5_next/cuda/weights.py, qmm.py; ed78d6f) as device buffers: every tensor in
//! python's layout (dtype, shape, contiguous) and named by its attribute path (`w.layers.0.kda.proj.weight`), so the
//! forward's launches, the pack pass and check-weights agree with python's. `rank` gives a rank's dataclasses with
//! their shapes; `walk` visits each buffer with its path (the arena, the digests and the tests use it).
const std = @import("std");
const md = @import("cuda_model.zig");
const Model = md.Model;

pub const DType = enum {
    bf16,
    f32,
    f16,
    i32,
    i64,
    i16,
    u8,
    f8e4m3, // patch 0002's make_f8 bytes (torch.float8_e4m3fn)

    pub fn size(d: DType) usize {
        return switch (d) {
            .bf16, .f16, .i16 => 2,
            .f32, .i32 => 4,
            .i64 => 8,
            .u8, .f8e4m3 => 1,
        };
    }
};

/// A contiguous tensor: its device address (0 until an arena places it), dtype and shape.
pub const Buf = struct {
    ptr: u64 = 0,
    dtype: DType,
    dims: u8,
    shape: [5]usize = @splat(1),
    host: bool = false, // pinned host memory (Buffers.ids_host), never a kernel argument

    pub fn of(dtype: DType, shape: []const usize) Buf {
        var b: Buf = .{ .dtype = dtype, .dims = @intCast(shape.len) };
        @memcpy(b.shape[0..shape.len], shape);
        return b;
    }
    pub fn numel(b: Buf) usize {
        var n: usize = 1;
        for (b.shape[0..b.dims]) |s| n *= s;
        return n;
    }
    pub fn bytes(b: Buf) usize {
        return b.numel() * b.dtype.size();
    }
    /// The elements of one index of the first dimension.
    pub fn inner(b: Buf) usize {
        return b.numel() / b.shape[0];
    }
};

/// qmm.Q4 as kernels/qmm.py `pack` leaves it: int32 words [npad/64, k/64, 8, 32, 2], bf16 scales and biases [k/64, npad].
pub const Q4 = struct {
    weight: Buf,
    scales: Buf,
    biases: Buf,
    n: usize,
    k: usize,

    pub fn of(n: usize, k: usize) Q4 {
        const npad = (n + 127) / 128 * 128;
        return .{ .weight = Buf.of(.i32, &.{ npad / 64, k / 64, 8, 32, 2 }), .scales = Buf.of(.bf16, &.{ k / 64, npad }), .biases = Buf.of(.bf16, &.{ k / 64, npad }), .n = n, .k = k };
    }
};

/// qmm.B16: [n, k] bf16 as an EXL3 checkpoint stores every non-expert weight.
pub const B16 = struct {
    weight: Buf,
    n: usize,
    k: usize,

    pub fn of(n: usize, k: usize) B16 {
        return .{ .weight = Buf.of(.bf16, &.{ n, k }), .n = n, .k = k };
    }
};

/// patch 0002's qmm.F8 (the dense-q4 form's head, qualification_qualification_ebed70b7): e4m3 [n, k] and fp32 scales [n, k/128], W ~ q * scale.
pub const F8 = struct {
    weight: Buf,
    scale: Buf,
    n: usize,
    k: usize,

    pub fn of(n: usize, k: usize) F8 {
        return .{ .weight = Buf.of(.f8e4m3, &.{ n, k }), .scale = Buf.of(.f32, &.{ n, k / 128 }), .n = n, .k = k };
    }
};

/// The weight-format seam (qualification_qualification_7c75d777 PLAN §3): a projection the forward's `mm` takes in either format.
pub const Linear = union(enum) {
    q4: Q4,
    b16: B16,
    f8: F8, // patch 0002's FP8 head (TF_GLM_F8=head with TF_GLM_DENSE=q4; cuda_pack_f8.adapt makes it, never `of`)

    pub fn of(m: *const Model, outs: usize, ins: usize) Linear {
        return if (m.quant == .exl3) .{ .b16 = B16.of(outs, ins) } else .{ .q4 = Q4.of(outs, ins) };
    }
    pub fn n(l: Linear) usize {
        return switch (l) {
            inline else => |x| x.n,
        };
    }
    pub fn k(l: Linear) usize {
        return switch (l) {
            inline else => |x| x.k,
        };
    }
};

/// HCW: the mixing projection fn [24, S*D] bf16, base [24] and scale [3] fp32.
pub const Hc = struct { @"fn": Buf, base: Buf, scale: Buf };

/// KDAW: proj = [q | k | v | f_a | g_a | b], conv [3 HL 128, taps] bf16, a_log [HL] and dt_bias [HL 128] fp32.
pub const Kda = struct {
    proj: Linear,
    fb: Linear,
    gb: Linear,
    conv: Buf,
    a_log: Buf,
    dt_bias: Buf,
    norm: Buf,
    o: Linear,
    heads: usize,

    pub fn faOff(x: *const Kda) usize {
        return 3 * x.heads * 128;
    }
    pub fn gaOff(x: *const Kda) usize {
        return x.faOff() + 128;
    }
    pub fn bOff(x: *const Kda) usize {
        return x.gaOff() + 128;
    }
};

/// IndexW: the DSA indexer, replicated (keys and head weights stacked; queries; key LayerNorm; pool gate and bias).
pub const Index = struct { kw: Linear, qb: Linear, ln_w: Buf, ln_b: Buf, gate: Buf, ape: Buf };

/// latent.AbsorbQ4 (the checkpoint's 4-bit kv_b rows as stored) or AbsorbW (BF16 blocks); lane A fills the fields.
pub const Absorb = union(enum) {
    q4: struct { kw: Buf, ks: Buf, kb: Buf, vw: Buf, vs: Buf, vb: Buf },
    b16: struct { wk: Buf, wv: Buf },
};

/// DSAW: proj = [q_a | kv_a], kv_k / kv_v the key and value rows of kv_b for the rank's heads.
pub const Dsa = struct {
    proj: Linear,
    q_norm: Buf,
    kv_norm: Buf,
    q_b: Linear,
    kv_k: Linear,
    kv_v: Linear,
    o: Linear,
    heads: usize,
    index: ?Index = null,
    absorb: ?Absorb = null,
};

/// MLPW: gu = [gate | up] (a rank's rows of each), down.
pub const Mlp = struct { gu: Linear, down: Linear, width: usize };

/// MoEW's experts: cuda/experts.Experts (E + 1 with the shared expert last) or cuda/exl3/experts.Exl3RoutedExperts.
pub const Experts = union(enum) {
    q4: ExpertsQ4,
    exl3: Exl3Experts,
};

/// cuda/experts.Experts: experts.pack's blocks, up [E + 1, NI/32, D/gs, 2 (gate, up), block] and down [E + 1, D/32,
/// NI/gs, 1, block] int32 (block = 32 NTW gs/32 words, then 8 NTW of scales and biases), and its scalar fields.
pub const ExpertsQ4 = struct {
    up: Buf,
    down: Buf,
    count: usize,
    gs: usize,
    width: usize,
    dims: usize,
    limit: f32,
};

/// Exl3RoutedExperts' tensors (gate, up, down trellis pointers and half-bit widths a expert, suh/svh stacked fp16,
/// trellis_bytes on the host) and fields; its per-expert trellises (`keep`) are qualification_qualification_44ba554e-exl3's storage, not listed here.
pub const Exl3Experts = struct {
    gate_ptr: Buf,
    up_ptr: Buf,
    down_ptr: Buf,
    gate_k2: Buf,
    up_k2: Buf,
    down_k2: Buf,
    suh_g: Buf,
    suh_u: Buf,
    svh_g: Buf,
    svh_u: Buf,
    suh_d: Buf,
    svh_d: Buf,
    trellis_bytes: Buf,
    count: usize,
    dims: usize,
    width: usize,
    cb: u8 = 1, // codebook_id("mcg"): weights.py prepares every layer with "mcg"
    k2_gu: [2]u32, // (min, max) half-bits a value over gate and up, as prepare reads the trellises
    k2_d: [2]u32,

    /// E experts of D -> I at `bits` a value (a uniform checkpoint: every trellis [K/16, N/16, 8 * 2 bits]).
    pub fn of(e: usize, d: usize, i: usize, bits: usize) Exl3Experts {
        const p = Buf.of(.i64, &.{e});
        const k2 = Buf.of(.i32, &.{e});
        var tb = Buf.of(.i64, &.{e});
        tb.host = true; // torch.tensor(...) with no device: a host tensor even on a GPU
        const half: u32 = @intCast(2 * bits);
        return .{ .gate_ptr = p, .up_ptr = p, .down_ptr = p, .gate_k2 = k2, .up_k2 = k2, .down_k2 = k2, .suh_g = Buf.of(.f16, &.{ e, d }), .suh_u = Buf.of(.f16, &.{ e, d }), .svh_g = Buf.of(.f16, &.{ e, i }), .svh_u = Buf.of(.f16, &.{ e, i }), .suh_d = Buf.of(.f16, &.{ e, i }), .svh_d = Buf.of(.f16, &.{ e, d }), .trellis_bytes = tb, .count = e, .dims = d, .width = i, .k2_gu = .{ half, half }, .k2_d = .{ half, half } };
    }
};

pub const Moe = struct { router: Buf, bias: Buf, experts: Experts, shared: ?Mlp = null };

pub const Layer = struct {
    index: usize,
    kind: md.Kind,
    attn_hc: ?Hc,
    ffn_hc: ?Hc,
    in_norm: Buf,
    post_norm: Buf,
    kda: ?Kda = null,
    dsa: ?Dsa = null,
    mlp: ?Mlp = null,
    moe: ?Moe = null,
};

/// glue.embed's table: MLX rows as stored (words, scales, biases) or [V, D] bf16.
pub const Table = union(enum) { q4: [3]Buf, b16: Buf };

/// MTPW: the MTP head (the checkpoint's layers.<num_hidden_layers>.): its input norms, eh [D, 2D] over [embedding |
/// hidden], shared_head.norm, and its DSA + MoE layer with a plain residual (no hyper-connections).
pub const Mtp = struct { enorm: Buf, hnorm: Buf, eh: Linear, norm: Buf, layer: Layer };

/// Weights: the rank's dataclasses and the MTP head (lane B's). EXL3's BF16 head has a 4-bit copy for drafts (quantize4).
pub const Weights = struct {
    embed: Table,
    layers: []Layer,
    norm: Buf,
    head: Linear,
    mtp: ?Mtp = null,
    draft_head: ?Q4 = null,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
        w.* = undefined;
    }
};

fn hc(m: *const Model) Hc {
    return .{ .@"fn" = Buf.of(.bf16, &.{ 24, m.streams * m.hidden }), .base = Buf.of(.f32, &.{24}), .scale = Buf.of(.f32, &.{3}) };
}

fn kda(m: *const Model) Kda {
    const ll = m.ll();
    return .{
        .proj = Linear.of(m, m.kdaWidth(), m.hidden),
        .fb = Linear.of(m, ll * m.lin_dim, m.lin_dim),
        .gb = Linear.of(m, ll * m.lin_dim, m.lin_dim),
        .conv = Buf.of(.bf16, &.{ 3 * ll * m.lin_dim, m.conv }),
        .a_log = Buf.of(.f32, &.{ll}),
        .dt_bias = Buf.of(.f32, &.{ll * m.lin_dim}),
        .norm = Buf.of(.bf16, &.{m.lin_dim}),
        .o = Linear.of(m, m.hidden, ll * m.lin_dim),
        .heads = ll,
    };
}

fn mlp(m: *const Model, width: usize) Mlp {
    return .{ .gu = Linear.of(m, 2 * width, m.hidden), .down = Linear.of(m, m.hidden, width), .width = width };
}

/// weights.py `dsa(i)`: [q_a | kv_a], the norms, q_b, kv_b's key and value rows of the rank's heads, o, IndexW (keys
/// with head weights stacked, queries, LayerNorm, pool gate and bias) and latent's absorb (AbsorbQ4 or AbsorbW).
fn dsa(m: *const Model) Dsa {
    const h = m.hl();
    const lw = m.kv_lora;
    const absorb: Absorb = if (m.quant == .exl3) .{ .b16 = .{ .wk = Buf.of(.bf16, &.{ h, m.qk_dim, lw }), .wv = Buf.of(.bf16, &.{ h, m.v_dim, lw }) } } else .{ .q4 = .{
        .kw = Buf.of(.i32, &.{ h, m.qk_dim, lw / 8 }), .ks = Buf.of(.bf16, &.{ h, m.qk_dim, lw / 64 }), .kb = Buf.of(.bf16, &.{ h, m.qk_dim, lw / 64 }),
        .vw = Buf.of(.i32, &.{ h, m.v_dim, lw / 8 }),  .vs = Buf.of(.bf16, &.{ h, m.v_dim, lw / 64 }),  .vb = Buf.of(.bf16, &.{ h, m.v_dim, lw / 64 }),
    } };
    return .{
        .proj = Linear.of(m, m.q_lora + lw, m.hidden),
        .q_norm = Buf.of(.bf16, &.{m.q_lora}),
        .kv_norm = Buf.of(.bf16, &.{lw}),
        .q_b = Linear.of(m, h * m.qk_dim, m.q_lora),
        .kv_k = Linear.of(m, h * m.qk_dim, lw),
        .kv_v = Linear.of(m, h * m.v_dim, lw),
        .o = Linear.of(m, m.hidden, h * m.v_dim),
        .heads = h,
        .index = .{
            .kw = Linear.of(m, m.index_dim + m.index_heads, m.hidden),
            .qb = Linear.of(m, m.index_heads * m.index_dim, m.q_lora),
            .ln_w = Buf.of(.bf16, &.{m.index_dim}),
            .ln_b = Buf.of(.bf16, &.{m.index_dim}),
            .gate = Buf.of(.bf16, &.{ m.index_dim, m.hidden }),
            .ape = Buf.of(.bf16, &.{ m.kpool, m.index_dim }),
        },
        .absorb = absorb,
    };
}

/// weights.py `moe(i)`: the router and its bias, then experts.make's packed [gate, up] and down with the shared
/// expert as expert E (MLX; experts.pack's blocks for B1's pack pass), or EXL3's routed experts and the shared
/// expert as a BF16 MLP.
pub fn moe(m: *const Model) Moe {
    const e = m.experts;
    const d = m.hidden;
    const w = m.ml();
    const router = Buf.of(.bf16, &.{ e, d });
    const bias = Buf.of(.f32, &.{e});
    if (m.quant == .exl3) return .{ .router = router, .bias = bias, .experts = .{ .exl3 = Exl3Experts.of(e, d, w, m.bits) }, .shared = mlp(m, m.sl()) };
    const gs = m.group_size;
    const block = 32 * 4 * (gs / 32) + 8 * 4; // experts.py pack: NTW n8 tiles a warp, words then scales and biases
    return .{ .router = router, .bias = bias, .experts = .{ .q4 = .{
        .up = Buf.of(.i32, &.{ e + 1, w / 32, d / gs, 2, block }),
        .down = Buf.of(.i32, &.{ e + 1, d / 32, w / gs, 1, block }),
        .count = e + 1,
        .gs = gs,
        .width = w,
        .dims = d,
        .limit = m.limit, // experts.make(..., limit=cfg.limit)
    } } };
}

/// Layer i's dataclass for the model's rank: KDA or DSA attention, then the dense MLP or the MoE.
pub fn layer(m: *const Model, i: usize) !Layer {
    var l: Layer = .{ .index = i, .kind = m.kinds[i], .attn_hc = hc(m), .ffn_hc = hc(m), .in_norm = Buf.of(.bf16, &.{m.hidden}), .post_norm = Buf.of(.bf16, &.{m.hidden}) };
    switch (m.kinds[i]) {
        .kda => l.kda = kda(m),
        .dsa => l.dsa = dsa(m),
    }
    switch (m.mlp_kinds[i]) {
        .dense => l.mlp = mlp(m, m.dl()),
        .moe => l.moe = moe(m),
    }
    return l;
}

/// The rank's weights: the embedding (stored), every loaded layer, the final norm, the head's vocabulary rows, and the
/// MTP head when the model keeps it.
pub fn rank(gpa: std.mem.Allocator, m: *const Model) !Weights {
    const layers = try gpa.alloc(Layer, m.layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, i| l.* = try layer(m, i);
    const embed: Table = if (m.quant == .exl3) .{ .b16 = Buf.of(.bf16, &.{ m.vocab, m.hidden }) } else .{ .q4 = .{
        Buf.of(.i32, &.{ m.vocab, m.hidden / 8 }), Buf.of(.bf16, &.{ m.vocab, m.hidden / 64 }), Buf.of(.bf16, &.{ m.vocab, m.hidden / 64 }),
    } };
    const D = m.hidden;
    return .{
        .embed = embed,
        .layers = layers,
        .norm = Buf.of(.bf16, &.{m.hidden}),
        .head = Linear.of(m, m.vl(), m.hidden),
        .mtp = if (m.mtp_layers == 0) null else .{ .enorm = Buf.of(.bf16, &.{D}), .hnorm = Buf.of(.bf16, &.{D}), .eh = Linear.of(m, D, 2 * D), .norm = Buf.of(.bf16, &.{D}), .layer = try mtpLayer(m) },
        .draft_head = if (m.quant == .exl3) Q4.of(m.vl(), m.hidden) else null, // python builds it with MTP off too
    };
}

/// weights.load's layer(cfg.layers, plain=True): the MTP head's layer, DSA and MoE (lane A's arms), no hyper-connections.
pub fn mtpLayer(m: *const Model) !Layer {
    if (m.layers >= md.max_layers) return error.TooManyLayers; // the MTP layer's index is one past the backbone's
    var p = m.*;
    p.kinds[m.layers] = .dsa;
    p.mlp_kinds[m.layers] = .moe;
    var l = try layer(&p, m.layers);
    l.attn_hc = null;
    l.ffn_hc = null;
    return l;
}

/// A path being built ("w.layers.3.kda.proj.weight"): python's attribute names and list indices, dot-joined.
pub const Path = struct {
    buf: [192]u8 = undefined,
    len: usize = 0,

    pub fn init(root: []const u8) Path {
        var p: Path = .{};
        p.push(root);
        return p;
    }
    pub fn push(p: *Path, s: []const u8) void {
        if (p.len != 0) p.add(".");
        p.add(s);
    }
    pub fn pushIndex(p: *Path, i: usize) void {
        var tmp: [24]u8 = undefined;
        p.push(std.fmt.bufPrint(&tmp, "{d}", .{i}) catch unreachable);
    }
    fn add(p: *Path, s: []const u8) void {
        @memcpy(p.buf[p.len..][0..s.len], s);
        p.len += s.len;
    }
    pub fn str(p: *const Path) []const u8 {
        return p.buf[0..p.len];
    }
};

/// Visits every Buf reachable from `v` (structs by field name, optionals when set, a union's live payload in place,
/// slices and arrays by index) with its path; `visit(ctx, path, *Buf)`.
pub fn walk(v: anytype, path: Path, ctx: anytype, comptime visit: fn (@TypeOf(ctx), []const u8, *Buf) anyerror!void) anyerror!void {
    const T = @typeInfo(@TypeOf(v)).pointer.child;
    if (T == Buf) return visit(ctx, path.str(), v);
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names) |name| {
            var p = path;
            p.push(name);
            try walk(&@field(v, name), p, ctx, visit);
        },
        .optional => if (v.*) |*x| try walk(x, path, ctx, visit),
        .@"union" => switch (v.*) {
            inline else => |*x| try walk(x, path, ctx, visit),
        },
        .array => for (v, 0..) |*x, i| {
            var p = path;
            p.pushIndex(i);
            try walk(x, p, ctx, visit);
        },
        .pointer => |q| if (q.size == .slice) for (v.*, 0..) |*x, i| {
            var p = path;
            p.pushIndex(i);
            try walk(x, p, ctx, visit);
        },
        else => {},
    }
}

test "a TP2 rank's KDA layer and dense MLP in python's packed shapes" {
    var m = try Model.from(&@import("config.zig").Config{}, .{}, 2, 0);
    const l = try layer(&m, 0);
    const k = l.kda.?;
    try std.testing.expectEqual([5]usize{ 198, 64, 8, 32, 2 }, k.proj.q4.weight.shape);
    try std.testing.expectEqual([5]usize{ 64, 12672, 1, 1, 1 }, k.proj.q4.scales.shape);
    try std.testing.expectEqual(@as(usize, 12544), k.bOff());
    try std.testing.expectEqual(@as(usize, 2), k.fb.q4.scales.shape[0]);
    try std.testing.expectEqual([5]usize{ 64, 96, 8, 32, 2 }, l.mlp.?.down.q4.weight.shape);
    try std.testing.expect(l.dsa == null and (try layer(&m, 3)).kda == null);
}

test "a TP2 rank's DSA layer and MoE in python's shapes (weights.load of layers 0-3, both formats)" {
    // evidence/qualification_qualification_abcce4b2/scripts/layer3_shapes.py: python-0.6's own load of each header mirror, w.layers.3's buffers
    var m = try Model.from(&@import("config.zig").Config{}, .{}, 2, 0);
    const l = try layer(&m, 3);
    const d = l.dsa.?;
    try std.testing.expectEqual([5]usize{ 32, 64, 8, 32, 2 }, d.proj.q4.weight.shape);
    try std.testing.expectEqual([5]usize{ 128, 24, 8, 32, 2 }, d.q_b.q4.weight.shape);
    try std.testing.expectEqual([5]usize{ 128, 8, 8, 32, 2 }, d.kv_v.q4.weight.shape);
    try std.testing.expectEqual([5]usize{ 64, 128, 8, 32, 2 }, d.o.q4.weight.shape);
    try std.testing.expectEqual([5]usize{ 4, 64, 8, 32, 2 }, d.index.?.kw.q4.weight.shape);
    try std.testing.expectEqual([5]usize{ 64, 24, 8, 32, 2 }, d.index.?.qb.q4.weight.shape);
    try std.testing.expectEqual([5]usize{ 4, 128, 1, 1, 1 }, d.index.?.ape.shape);
    try std.testing.expectEqual([5]usize{ 32, 256, 64, 1, 1 }, d.absorb.?.q4.vw.shape);
    try std.testing.expectEqual([5]usize{ 32, 256, 8, 1, 1 }, d.absorb.?.q4.ks.shape);
    const x = l.moe.?;
    try std.testing.expectEqual([5]usize{ 288, 4096, 1, 1, 1 }, x.router.shape);
    try std.testing.expectEqual([5]usize{ 289, 32, 64, 2, 288 }, x.experts.q4.up.shape);
    try std.testing.expectEqual([5]usize{ 289, 128, 16, 1, 288 }, x.experts.q4.down.shape);
    try std.testing.expect(x.shared == null and l.mlp == null and l.kda == null);
    m.quant = .exl3;
    const e = try layer(&m, 3);
    try std.testing.expectEqual([5]usize{ 2048, 4096, 1, 1, 1 }, e.dsa.?.proj.b16.weight.shape);
    try std.testing.expectEqual([5]usize{ 160, 4096, 1, 1, 1 }, e.dsa.?.index.?.kw.b16.weight.shape);
    try std.testing.expectEqual([5]usize{ 32, 256, 512, 1, 1 }, e.dsa.?.absorb.?.b16.wk.shape);
    const ex = e.moe.?.experts.exl3;
    try std.testing.expectEqual([5]usize{ 288, 1024, 1, 1, 1 }, ex.suh_d.shape);
    try std.testing.expectEqual([5]usize{ 288, 4096, 1, 1, 1 }, ex.svh_d.shape);
    try std.testing.expect(ex.gate_ptr.dtype == .i64 and ex.up_k2.dtype == .i32 and ex.trellis_bytes.host);
    try std.testing.expectEqual([5]usize{ 2048, 4096, 1, 1, 1 }, e.moe.?.shared.?.gu.b16.weight.shape);
    try std.testing.expectEqual([5]usize{ 4096, 1024, 1, 1, 1 }, e.moe.?.shared.?.down.b16.weight.shape);
}
