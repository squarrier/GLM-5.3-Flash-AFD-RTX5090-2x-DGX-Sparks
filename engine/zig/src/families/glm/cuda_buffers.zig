//! python-0.6's Buffers and State (glm5_next/cuda/forward.py, ed78d6f) field for field: a decode window's or prompt
//! chunk's scratch, and one sequence's committed caches, each tensor a views.Buf of python's dtype and shape for the
//! model's rank. Names are python's attributes, so `views.walk` gives `b.normed`, `st.scratch_set.out`, `st.kc.0`.
//! EXL3's expert scratch is cuda/exl3/experts.py's Scratch, whose `y` is `b.ey` (python's ey is a view of it).
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const Model = md.Model;
const Buf = views.Buf;

/// geometry.MLA_PROMPT_ATT_ROWS: a prompt chunk's rows a dense latent attention call takes
pub const prompt_att_rows = 512;
/// glue.HC_BLOCKS
pub const hc_blocks = 16;
/// cuda/experts.py: TILE (a decode item's pairs) and SMALL (pairs the one-block plan takes)
pub const expert_tile = 16;
pub const expert_small = 1024;
/// latent.CHUNK
pub const chunk = 512;

/// latent.chunks_for
pub fn chunksFor(length: usize) usize {
    return (length + chunk - 1) / chunk;
}

/// latent.LatentScratch: chunk partials for `part_rows` rows, absorbed queries and latent outputs for `rows`.
pub const LatentScratch = struct { po: Buf, pm: Buf, pl: Buf, qa: Buf, ol: Buf, dummy: Buf };
/// kda.KDAScratch (a prompt chunk's) and KDAScratchSet (every layer's window, State.scratch_set)
pub const KdaScratch = struct { out: Buf, k: Buf, v: Buf, g: Buf, b: Buf };
/// cuda/experts.Plan
pub const Plan = struct { members: Buf, items: Buf, counts: Buf, rank: Buf, hist: Buf };

/// cuda/exl3/experts.py's tile settings (n tiles, warps, K splits, prefetch): GLM_GATEUP, GLM_DOWN, then the fallbacks
const glm_gateup = [4]usize{ 8, 4, 4, 1 };
const glm_down = [4]usize{ 8, 4, 1, 1 };
const exl3_fallbacks = [_][4]usize{ .{ 8, 4, 2, 1 }, .{ 8, 4, 1, 1 }, .{ 4, 4, 2, 2 }, .{ 4, 4, 1, 2 } };

fn fits(c: [4]usize, K: usize, N: usize) bool {
    return K % (16 * c[2] * c[1]) == 0 and N % (16 * c[0]) == 0;
}

/// cuda/exl3/experts.default_config: GLM's setting for a K -> N projection where it divides, else the first fallback that does.
pub fn exl3Config(K: usize, N: usize, gateup: bool) ![4]usize {
    const first = if (gateup) glm_gateup else glm_down;
    if (fits(first, K, N)) return first;
    for (exl3_fallbacks) |c| if (fits(c, K, N)) return c;
    return error.NoTileSetting; // python: ValueError
}

/// cuda/exl3/experts.Scratch for `rows` rows of `slots` slots (its `y` is Buffers.ey).
pub const Exl3Scratch = struct { xg: Buf, xu: Buf, xd: Buf, z: Buf, ids: Buf, count: Buf, members_buf: Buf };

fn exl3Scratch(m: *const Model, rows: usize, slots: usize) !Exl3Scratch {
    const D = m.hidden;
    const I = m.ml();
    const P = rows * slots;
    const gu = try exl3Config(D, I, true);
    const dn = try exl3Config(I, D, false);
    const maxu = @min(P, m.experts);
    return .{
        .xg = Buf.of(.f16, &.{ P, D }),
        .xu = Buf.of(.f16, &.{ P, D }),
        .xd = Buf.of(.f16, &.{ P, I }),
        .z = Buf.of(.f32, &.{@max(2 * gu[2] * I, dn[2] * D) * P}),
        .ids = Buf.of(.i32, &.{maxu}),
        .count = Buf.of(.i32, &.{1}),
        .members_buf = Buf.of(.i32, &.{maxu * rows}),
    };
}

pub const Buffers = struct {
    /// What `initWith` leaves out. The AFD attention node runs no routed experts, so it needs no EXL3 expert scratch
    /// (python 2.15 engine.py:527-530, `exl3_scratch=afd is None`); the default leaves nothing out.
    pub const Opt = struct { exl3_scratch: bool = true };

    rows: usize,
    prefill: bool,
    ids: Buf,
    ids_host: Buf,
    lat_s: LatentScratch,
    kproj: ?Buf = null,
    kscratch: ?KdaScratch = null,
    hin: Buf,
    x: Buf,
    normed: Buf,
    xs: Buf,
    post: Buf,
    comb: Buf,
    hcpart: Buf,
    ka: Buf,
    kg: Buf,
    xs_fa: Buf,
    xs_ga: Buf,
    kxs: Buf,
    dp: Buf,
    qr: Buf,
    xs_qr: Buf,
    lat: Buf,
    xs_lat: Buf,
    q: Buf,
    kn: Buf,
    vn: Buf,
    xs_ao: Buf,
    ikr: Buf,
    igr: Buf,
    qi: Buf,
    gu: Buf,
    act: Buf,
    xs_act: Buf,
    mlog: Buf,
    pick: Buf,
    wts: Buf,
    eact: Buf,
    ey: Buf,
    plan: ?Plan,
    exl3: ?Exl3Scratch = null,
    sgu: ?Buf = null,
    sact: ?Buf = null,
    sxs: ?Buf = null,
    sy: ?Buf = null,
    part: Buf,
    gath: Buf,
    sk: Buf,
    hidden: Buf,
    fnormed: Buf,
    fxs: Buf,
    logits: Buf,
    me: Buf,
    mcat: Buf,
    mxs: Buf,
    mx: Buf,
    /// mtp.py's zero_first: this MTP step starts at position 0 (its embedding row is zeroed)
    zero_first: bool = false,
    /// python's `staged`: the ids' host-to-device copy, so the pinned rows are not rewritten under it (a GPU's)
    staged: ?cuda.Event = null,

    /// Buffers(w, rows, capacity, prefill=...) for the model's rank (latent DSA, python's default).
    pub fn init(m: *const Model, rows: usize, capacity: usize, prefill: bool) !Buffers {
        return initWith(m, rows, capacity, prefill, .{});
    }

    /// `init` with what `opt` leaves out (the default leaves nothing out: the same buffers as `init`).
    pub fn initWith(m: *const Model, rows: usize, capacity: usize, prefill: bool, opt: Opt) !Buffers {
        const D = m.hidden;
        const S = m.streams;
        const HL = m.hl();
        const LL = m.ll();
        const slots = m.top_k + 1;
        const ml = m.ml();
        const dl = m.dl();
        const exl3 = m.quant == .exl3;
        const part_rows = @min(rows, if (prefill) prompt_att_rows else rows);
        const nch = chunksFor(@min(capacity, 2560) + rows);
        const pairs = rows * slots;
        const items = @min(pairs, m.experts + 1) + pairs / expert_tile;
        const wide = pairs > expert_small;
        var b: Buffers = .{
            .rows = rows,
            .prefill = prefill,
            .ids = Buf.of(.i32, &.{rows}),
            .ids_host = .{ .dtype = .i32, .dims = 1, .shape = .{ rows, 1, 1, 1, 1 }, .host = true },
            .lat_s = .{
                .po = Buf.of(.f32, &.{nch * part_rows * HL * m.kv_lora}),
                .pm = Buf.of(.f32, &.{nch * part_rows * HL}),
                .pl = Buf.of(.f32, &.{nch * part_rows * HL}),
                .qa = Buf.of(.bf16, &.{ rows, HL, m.kv_lora }),
                .ol = Buf.of(.bf16, &.{ rows, HL, m.kv_lora }),
                .dummy = Buf.of(.i32, &.{1}),
            },
            .hin = Buf.of(.bf16, &.{ rows, D }),
            .x = Buf.of(.bf16, &.{ rows, S * D }),
            .normed = Buf.of(.bf16, &.{ rows, D }),
            .xs = Buf.of(.f32, &.{ rows, D / 64 }),
            .post = Buf.of(.f32, &.{ rows, S }),
            .comb = Buf.of(.f32, &.{ rows, S * S }),
            .hcpart = Buf.of(.f32, &.{ rows, hc_blocks, 32 }),
            .ka = Buf.of(.bf16, &.{ rows, LL * 128 }),
            .kg = Buf.of(.bf16, &.{ rows, LL * 128 }),
            .xs_fa = Buf.of(.f32, &.{ rows, 2 }),
            .xs_ga = Buf.of(.f32, &.{ rows, 2 }),
            .kxs = Buf.of(.f32, &.{ rows, LL * 128 / 64 }),
            .dp = Buf.of(.bf16, &.{ rows, m.q_lora + m.kv_lora }),
            .qr = Buf.of(.bf16, &.{ rows, m.q_lora }),
            .xs_qr = Buf.of(.f32, &.{ rows, m.q_lora / 64 }),
            .lat = Buf.of(.bf16, &.{ rows, m.kv_lora }),
            .xs_lat = Buf.of(.f32, &.{ rows, m.kv_lora / 64 }),
            .q = Buf.of(.bf16, &.{ rows, HL, m.qk_dim }),
            .kn = Buf.of(.bf16, &.{ rows, HL, m.qk_dim }),
            .vn = Buf.of(.bf16, &.{ rows, HL, m.v_dim }),
            .xs_ao = Buf.of(.f32, &.{ rows, HL * m.v_dim / 64 }),
            .ikr = Buf.of(.bf16, &.{ rows, m.index_dim + m.index_heads }),
            .igr = Buf.of(.f32, &.{ rows, m.index_dim }),
            .qi = Buf.of(.bf16, &.{ rows, m.index_heads * m.index_dim }),
            .gu = Buf.of(.bf16, &.{ rows, 2 * dl }),
            .act = Buf.of(.bf16, &.{ rows, dl }),
            .xs_act = Buf.of(.f32, &.{ rows, dl / 64 }),
            .mlog = Buf.of(.f32, &.{ rows, m.experts }),
            .pick = Buf.of(.i32, &.{ rows, slots }),
            .wts = Buf.of(.f32, &.{ rows, slots }),
            .eact = Buf.of(.bf16, &.{ rows * slots, ml }),
            .ey = Buf.of(if (prefill and !exl3) .bf16 else .f32, &.{ rows, slots, D }), // EXL3: exl3.y, fp32
            .plan = if (exl3) null else .{
                .members = Buf.of(.i32, &.{pairs}),
                .items = Buf.of(.i32, &.{ items, 3 }),
                .counts = Buf.of(.i32, &.{2}),
                .rank = Buf.of(.i32, &.{if (wide) pairs else 1}),
                .hist = Buf.of(.i32, &.{if (wide) (pairs + 1023) / 1024 * (m.experts + 1) else 1}),
            },
            .part = Buf.of(.f32, &.{ rows, D }),
            .gath = Buf.of(.f32, &.{m.world * rows * D}),
            .sk = Buf.of(.f32, &.{if (prefill and !exl3) 1 else 8 * rows * 16384}),
            .hidden = Buf.of(.bf16, &.{ rows, D }),
            .fnormed = Buf.of(.bf16, &.{ rows, D }),
            .fxs = Buf.of(.f32, &.{ rows, D / 64 }),
            .logits = Buf.of(.bf16, &.{ if (prefill) 1 else rows, m.vl() }),
            .me = Buf.of(.bf16, &.{ rows, D }),
            .mcat = Buf.of(.bf16, &.{ rows, 2 * D }),
            .mxs = Buf.of(.f32, &.{ rows, 2 * D / 64 }),
            .mx = Buf.of(.bf16, &.{ rows, D }),
        };
        if (exl3) { // EXL3 routed experts, and the shared expert as a BF16 MLP
            const sl = m.sl();
            if (opt.exl3_scratch) b.exl3 = try exl3Scratch(m, rows, slots);
            b.sgu = Buf.of(.bf16, &.{ rows, 2 * sl });
            b.sact = Buf.of(.bf16, &.{ rows, sl });
            b.sxs = Buf.of(.f32, &.{ rows, sl / 64 });
            b.sy = Buf.of(.f32, &.{ rows, D });
        }
        if (prefill) {
            b.kproj = Buf.of(.bf16, &.{ 1, rows, if (m.count(.kda, m.layers) > 0) m.kdaWidth() else 0 });
            b.kscratch = kdaScratch(rows, LL, null);
        }
        return b;
    }
};

fn kdaScratch(rows: usize, heads: usize, layers: ?usize) KdaScratch {
    const n = layers orelse 0;
    const lead: []const usize = if (layers != null) &.{n} else &.{};
    return .{
        .out = shaped(.bf16, lead, &.{ rows, heads * 128 }),
        .k = shaped(.f32, lead, &.{ rows, heads, 128 }),
        .v = shaped(.bf16, lead, &.{ rows, heads, 128 }),
        .g = shaped(.f32, lead, &.{ rows, heads, 128 }),
        .b = shaped(.f32, lead, &.{ rows, heads }),
    };
}

fn shaped(dtype: views.DType, lead: []const usize, rest: []const usize) Buf {
    var s: [5]usize = undefined;
    @memcpy(s[0..lead.len], lead);
    @memcpy(s[lead.len..][0..rest.len], rest);
    return Buf.of(dtype, s[0 .. lead.len + rest.len]);
}

/// One device block for a set of buffers (python allocates each alone): a buffer at each 256-byte boundary, walk order;
/// a pass at base 0 sizes the block, a second at its address places the buffers. Pinned host buffers stay out.
pub const Arena = struct {
    base: u64 = 0,
    used: usize = 0,

    pub fn visit(a: *Arena, path: []const u8, b: *Buf) anyerror!void {
        _ = path;
        if (b.host) return;
        b.ptr = a.base + a.used;
        a.used = std.mem.alignForward(usize, a.used + b.bytes(), 256);
    }
};

/// DSA indexer caches of one layer (long contexts): keys, gates, pool keys
pub const IndexCache = [3]Buf;

pub const State = struct {
    capacity: usize,
    pos: usize = 0,
    /// MTP cache rows written (python's mtp_len; mtp_pos_dev holds it on the device)
    mtp_len: usize = 0,
    /// MTP rows a chained draft wrote past the kept ones (python's mtp_drafted; decode.absorb trims them)
    mtp_drafted: usize = 0,
    /// each KDA layer's live half of `rec` (python's `cur`, all equal between commits)
    parity: u1 = 0,
    pos_dev: Buf,
    mtp_pos_dev: Buf,
    conv: Buf,
    rec: Buf,
    proj: Buf,
    scratch_set: KdaScratch,
    kc: []Buf,
    mtp_kc: ?Buf = null,
    index: ?[]IndexCache = null,

    /// State(w, capacity, rows) for the model's rank: latent DSA caches; the MTP cache when the model keeps the MTP layer.
    pub fn init(gpa: std.mem.Allocator, m: *const Model, capacity: usize, rows: usize, long_context: bool) !State {
        const n = m.count(.kda, m.layers);
        const ndsa = m.count(.dsa, m.layers);
        const LL = m.ll();
        const width = if (n > 0) m.kdaWidth() else 0;
        const kc = try gpa.alloc(Buf, ndsa);
        errdefer gpa.free(kc);
        for (kc) |*c| c.* = Buf.of(.bf16, &.{ capacity, m.kv_lora });
        var st: State = .{
            .capacity = capacity,
            .pos_dev = Buf.of(.i32, &.{1}),
            .mtp_pos_dev = Buf.of(.i32, &.{1}),
            .conv = Buf.of(.bf16, &.{ n, m.conv - 1, 3 * LL * 128 }),
            .rec = Buf.of(.f32, &.{ 2, n, LL, 128, 128 }),
            .proj = Buf.of(.bf16, &.{ n, rows, width }),
            .scratch_set = kdaScratch(rows, LL, n),
            .kc = kc,
            .mtp_kc = if (m.mtp_layers > 0) Buf.of(.bf16, &.{ capacity, m.kv_lora }) else null,
        };
        if (long_context) {
            const idx = try gpa.alloc(IndexCache, ndsa + @intFromBool(m.mtp_layers > 0));
            for (idx) |*c| c.* = .{ Buf.of(.bf16, &.{ capacity, m.index_dim }), Buf.of(.bf16, &.{ capacity, m.index_dim }), Buf.of(.bf16, &.{ capacity / 4 + 2, m.index_dim }) };
            st.index = idx;
        }
        return st;
    }

    pub fn deinit(st: *State, gpa: std.mem.Allocator) void {
        gpa.free(st.kc);
        if (st.index) |x| gpa.free(x);
        st.* = undefined;
    }
};

test "a decode window's buffers in python's shapes (TP2 rank, 8 rows, capacity 2560)" {
    const m = try Model.from(&@import("config.zig").Config{}, .{}, 2, 0);
    const b = try Buffers.init(&m, 8, 2560, false);
    try std.testing.expectEqual(@as(usize, 786432), b.lat_s.po.numel());
    try std.testing.expectEqual([5]usize{ 76, 3, 1, 1, 1 }, b.plan.?.items.shape);
    try std.testing.expectEqual(@as(usize, 65536), b.gath.numel());
    try std.testing.expectEqual(@as(usize, 1048576), b.sk.numel());
    try std.testing.expect(b.exl3 == null and b.sgu == null);
    const p = try Buffers.init(&m, 2048, 2560, true);
    try std.testing.expectEqual(@as(usize, 1), p.sk.numel());
    try std.testing.expectEqual(@as(usize, 1), p.logits.shape[0]);
}

test "EXL3: default_config's tile settings and the expert scratch" {
    try std.testing.expectEqual([4]usize{ 8, 4, 4, 1 }, try exl3Config(4096, 1024, true));
    try std.testing.expectEqual([4]usize{ 8, 4, 1, 1 }, try exl3Config(1024, 4096, false));
    try std.testing.expectEqual([4]usize{ 8, 4, 2, 1 }, try exl3Config(128, 1024, true));
    try std.testing.expectError(error.NoTileSetting, exl3Config(4096, 40, true));
    var m = try Model.from(&@import("config.zig").Config{}, .{ .quant = .exl3 }, 2, 0);
    const b = try Buffers.init(&m, 8, 2560, false);
    try std.testing.expectEqual(@as(usize, 589824), b.exl3.?.z.numel());
    try std.testing.expect(b.plan == null and b.ey.dtype == .f32 and b.sxs.?.shape[1] == 16);
    m.experts = 50; // fewer experts than pairs: ids and members by experts
    try std.testing.expectEqual(@as(usize, 50), (try Buffers.init(&m, 8, 2560, false)).exl3.?.ids.numel());
}

/// Every buffer of `b` as views.walk names it (path, dtype, shape, host) and the total bytes, digested; `skip` leaves
/// out the paths that start with it.
fn layoutDigest(b: *Buffers, skip: ?[]const u8) ![32]u8 {
    const Ctx = struct {
        h: std.crypto.hash.sha2.Sha256 = .init(.{}),
        bytes: usize = 0,
        skip: ?[]const u8,
        fn visit(c: *@This(), path: []const u8, x: *Buf) anyerror!void {
            if (c.skip) |s| if (std.mem.startsWith(u8, path, s)) return;
            var line: [256]u8 = undefined;
            c.h.update(try std.fmt.bufPrint(&line, "{s} {t} {any} {}\n", .{ path, x.dtype, x.shape[0..x.dims], x.host }));
            c.bytes += x.bytes();
        }
    };
    var c: Ctx = .{ .skip = skip };
    try views.walk(b, views.Path.init("b"), &c, Ctx.visit);
    var tail: [32]u8 = undefined;
    c.h.update(try std.fmt.bufPrint(&tail, "{d}\n", .{c.bytes}));
    return c.h.finalResult();
}

/// The default layouts (MLX and EXL3, a decode window and a prompt chunk), digested on qualification_qualification_44ba554e-int 4128a9c's `init`.
const default_layouts = [_]struct { quant: md.Quant, rows: usize, prefill: bool, hex: []const u8 }{
    .{ .quant = .mlx, .rows = 8, .prefill = false, .hex = "1eff8e718ad29c6ce58f275a60c685608958028bb69e6511790106dd73d57719" },
    .{ .quant = .mlx, .rows = 2048, .prefill = true, .hex = "d9c98a693bf430510ba330bf97b294abf5c47579fcfd3f50e78db370406a6b2b" },
    .{ .quant = .exl3, .rows = 8, .prefill = false, .hex = "4a31df3bb270c9c1832b4fed70f1a9d617a848584853412510f49e299c1555df" },
    .{ .quant = .exl3, .rows = 2048, .prefill = true, .hex = "4157cebf32fefa9da744c33a189c21198b8b7f1ac1c6cb31fddbd4633142be28" },
};

test "Buffers: the default layout is qualification_qualification_44ba554e-int's byte for byte; attention-only drops the EXL3 expert scratch alone" {
    for (default_layouts) |want| {
        const m = try Model.from(&@import("config.zig").Config{}, .{ .quant = want.quant }, 2, 0);
        var b = try Buffers.init(&m, want.rows, 2560, want.prefill);
        var d = try Buffers.initWith(&m, want.rows, 2560, want.prefill, .{});
        var a = try Buffers.initWith(&m, want.rows, 2560, want.prefill, .{ .exl3_scratch = false });
        const hex = std.fmt.bytesToHex(try layoutDigest(&b, null), .lower);
        if (!std.mem.eql(u8, &hex, want.hex)) std.debug.print("layout {t} rows {d}: {s}\n", .{ want.quant, want.rows, &hex });
        try std.testing.expectEqualStrings(want.hex, &hex);
        try std.testing.expectEqualStrings(want.hex, &std.fmt.bytesToHex(try layoutDigest(&d, null), .lower));
        try std.testing.expect(a.exl3 == null and (want.quant == .mlx) == (b.exl3 == null));
        try std.testing.expectEqual(try layoutDigest(&b, "b.exl3."), try layoutDigest(&a, null));
    }
}
