//! python-0.6's prompt chunks in kda_block (glm5_next/cuda/forward.py, ed78d6f): a chunk's rows project into b.kproj on
//! the shared prefill matmul, kda.chain takes its wide path from 64 rows (kda._wide_scratch), and the layer commits at
//! once (its conv window shifts by the chunk; st.cur flips in cuda_compute.compute). An interior cut (forward.Cut, for
//! a kept prefix) runs the chain twice: the first rows into the cut's state (their output cloned, the cut's window
//! shifted), then the rest from there, joined by `torch.cat`. The temporaries are python's (cuda_tmp.zig). A chunk's
//! DSA layers (lane A's block, A4) take the engine's Scratch for sparse partials past b.lat_s (python's torch.empty).
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const dsa = @import("cuda_dsa.zig");
const kern = @import("cuda_kernels.zig");
const cc = @import("cuda_compute.zig");
const Tmp = @import("cuda_tmp.zig").Tmp;
const Fwd = @import("cuda_fwd.zig").Fwd;
const Kernels = @import("cuda_fwd.zig").Kernels;

const T = trace.T;
const Buf = views.Buf;
const Buffers = bufs.Buffers;
const State = bufs.State;

/// forward.Cut: the recurrent state (like st.rec[0]) and conv windows (st.conv's clone) at row `point` of a chunk.
pub const Cut = struct { point: usize, rec: Buf, conv: Buf };

/// lane A's dsa_kernels.Scratch (A4) read off fwd.Kernels.dsa's type; void where the tree's DSA block is A3's stub.
pub const DsaScratch = blk: {
    if (!@hasField(Kernels, "dsa") or !@hasDecl(dsa, "blockWith") or !@hasDecl(layer, "ffnHalf")) break :blk void;
    const K = @typeInfo(@typeInfo(@FieldType(Kernels, "dsa")).optional.child).pointer.child;
    if (!@hasField(K, "scratch")) break :blk void;
    break :blk @typeInfo(@typeInfo(@FieldType(K, "scratch")).optional.child).pointer.child;
};

/// A prompt chunk's context: its cut, and python's temporaries (a cut's `first` and `cat`, kda._wide_scratch's q, y;
/// latent.sparse_attention's partials past b.lat_s in `dsa`, which the engine reserves: dsaScratch).
pub const Prompt = struct {
    tmp: *Tmp,
    cut: ?*const Cut = null,
    first: Buf,
    cat: Buf,
    wide: usize = 0,
    q: Buf = .{ .dtype = .f32, .dims = 1 },
    y: Buf = .{ .dtype = .bf16, .dims = 1 },
    dsa: ?DsaScratch = null,

    /// For prefill Buffers of `rows` rows on the model's rank.
    pub fn init(m: *const md.Model, tmp: *Tmp, rows: usize) Prompt {
        const ch = m.ll() * 128;
        return .{ .tmp = tmp, .first = Buf.of(.bf16, &.{ rows, ch }), .cat = Buf.of(.bf16, &.{ rows, ch }) };
    }

    /// python's `kda._tmp.clear()` (a fixture case starts without the wide scratch)
    pub fn reset(p: *Prompt) void {
        p.wide = 0;
    }

    /// kda._wide_scratch(rows): on a CPU python's q and y, remade (and so renamed) for a longer window.
    fn wideScratch(p: *Prompt, f: *const Fwd, rows: usize, heads: usize) !?[2]T {
        if (f.rec == null) return null; // the Gpu's q_tmp and y_tmp
        if (p.wide < rows) {
            p.wide = rows;
            p.q = Buf.of(.f32, &.{ rows, heads, 128 });
            p.y = Buf.of(.bf16, &.{ rows, heads, 128 });
            try p.tmp.made("empty", &p.q);
            try p.tmp.made("empty", &p.y);
        }
        return .{ T.whole(p.q), T.whole(p.y) };
    }
};

/// The engine's Scratch for prompt chunks of pb.rows rows (python's capacity plan: geometry.mla_chunk_scratch's last
/// term): a long context past the dense limit with a DSA layer, and sparse partials b.lat_s cannot hold; else null.
pub fn dsaScratch(m: *const md.Model, pb: *const Buffers, capacity: usize, long_context: bool) ?DsaScratch {
    if (comptime DsaScratch == void) return null else {
        if (!long_context or capacity <= m.denseLimit() or (m.count(.dsa, m.layers) == 0 and m.mtp_layers == 0)) return null;
        const s = DsaScratch.init(m.hl(), pb.lat_s.qa.shape[2], pb.rows); // a chunk's rows, as dsa.temps asks
        return if (s.po.numel() <= pb.lat_s.po.numel() and s.pm.numel() <= pb.lat_s.pm.numel()) null else s;
    }
}

/// The Scratch a prompt chunk's DSA block takes (opt.prompt's), else null.
fn scratchOf(opt: layer.Opt) ?*const DsaScratch {
    if (comptime DsaScratch == void) return null else {
        const p = opt.prompt orelse return null;
        return if (p.dsa) |*s| s else null;
    }
}

/// forward.dsa_block (lane A's), passing a prompt chunk's Scratch to partialWith; without one, dsa.block itself.
pub fn dsaBlock(f: *Fwd, l: *const views.Layer, st: *const State, b: *const Buffers, R: usize, opt: layer.Opt) !u64 {
    if (comptime DsaScratch == void) return dsa.block(f, l, st, b, R, opt) else {
        const x = scratchOf(opt) orelse return dsa.block(f, l, st, b, R, opt);
        return dsa.blockWith(f, l, st, b, R, opt, x);
    }
}

/// layer.layerForward, a prompt chunk's DSA layer with dsaBlock: hc_pre and hc_post as cuda_layer.zig's, then its FFN
/// half (layer.ffnHalf). Every other layer or window is layer.layerForward itself.
pub fn layerForward(f: *Fwd, l: *const views.Layer, st: *const State, b: *const Buffers, R: usize, opt: layer.Opt) !void {
    if (comptime DsaScratch == void) return layer.layerForward(f, l, st, b, R, opt) else {
        if (l.kind != .dsa or scratchOf(opt) == null) return layer.layerForward(f, l, st, b, R, opt);
        const m = f.m;
        if (R == 0 or R > b.rows) return error.Rows;
        const h = l.attn_hc orelse return error.PlainLayer;
        try f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.in_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
        const g = try dsaBlock(f, l, st, b, R, opt);
        try f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
        try layer.ffnHalf(f, l, b, R, opt);
    }
}

/// kda.chain over the chunk's rows [r0, r0 + n) from window `conv` and state `in` into `out`, on b.kscratch.
fn chain(f: *Fwd, p: *Prompt, k: views.Kda, b: *const Buffers, r0: usize, n: usize, conv: T, in: T, out: T) !void {
    const kproj = b.kproj.?;
    const s = b.kscratch.?;
    const width = k.proj.n();
    const ch = k.heads * 128;
    const wide = if (n >= kern.wide_rows) try p.wideScratch(f, n, k.heads) else null;
    try f.kdaChain(.{
        .p = T.rows2(kproj, r0 * width * 2, n, width, width),
        .b_off = k.bOff(),
        .a = T.rows2(b.ka, r0 * ch * 2, n, ch, ch),
        .g = T.rows2(b.kg, r0 * ch * 2, n, ch, ch),
        .conv_state = conv,
        .conv_w = T.whole(k.conv),
        .state_in = in,
        .a_log = T.whole(k.a_log),
        .dt_bias = T.whole(k.dt_bias),
        .norm_w = T.whole(k.norm),
        .eps = f.m.eps,
        .lower = f.m.lower,
        .rows = n,
        .out = T.whole(s.out),
        .state_out = out,
        .k_save = T.whole(s.k),
        .v_save = T.whole(s.v),
        .g_save = T.whole(s.g),
        .b_save = T.whole(s.b),
        .q_tmp = if (wide) |w| w[0] else null,
        .y_tmp = if (wide) |w| w[1] else null,
    });
}

/// forward.kda_block for a prompt chunk of R rows (layer.kdaBlock's prefill form), with opt.prompt's cut if any.
pub fn kdaBlock(f: *Fwd, l: *const views.Layer, st: *const State, b: *const Buffers, R: usize, opt: layer.Opt) !u64 {
    const m = f.m;
    const k = l.kda orelse return error.NotKda;
    const p = opt.prompt orelse return error.NoPrompt;
    const kproj = b.kproj orelse return error.NotPrefill;
    const s = b.kscratch orelse return error.NotPrefill;
    if (R > kproj.shape[1]) return error.Rows;
    const li = m.kindIndex(l.index);
    const width = k.proj.n();
    const ch = k.heads * 128;
    const C = st.conv.shape[2];
    try qmm.mm(f, true, layer.rowsOf(b.normed, R), k.proj, layer.rowsOf(b.xs, R), T.rows2(kproj, 0, R, width, width), false, b.sk, opt.chip);
    try qmm.mm(f, true, T.rows2(kproj, k.faOff() * 2, R, 128, width), k.fb, null, layer.rowsOf(b.ka, R), false, b.sk, opt.chip);
    try qmm.mm(f, true, T.rows2(kproj, k.gaOff() * 2, R, 128, width), k.gb, null, layer.rowsOf(b.kg, R), false, b.sk, opt.chip);
    const n = st.rec.shape[1];
    const state = k.heads * 128 * 128 * 4;
    const cur: usize = st.parity;
    const conv = T.of(st.conv, li * st.conv.inner() * 2, st.conv.shape[1..3]);
    const in = T.of(st.rec, (cur * n + li) * state, &.{ k.heads, 128, 128 });
    const out = T.of(st.rec, ((1 - cur) * n + li) * state, &.{ k.heads, 128, 128 });
    const proj_layer = kproj.shape[1] * kproj.shape[2];
    var y = T.rows2(s.out, 0, R, ch, ch);
    if (p.cut) |c| {
        const at = c.point;
        if (at == 0 or at >= R) return error.Cut;
        const crec = T.of(c.rec, li * state, &.{ k.heads, 128, 128 });
        const cconv = T.of(c.conv, li * c.conv.inner() * 2, c.conv.shape[1..3]);
        try chain(f, p, k, b, 0, at, conv, in, crec);
        const first = try cc.clone(f, p.tmp, T.rows2(s.out, 0, at, ch, ch), &p.first);
        try f.tri.shiftConv(cconv.ptr, kproj.ptr, at, 1, 3 * C, proj_layer, width, C);
        try chain(f, p, k, b, at, R - at, cconv, crec, out);
        y = try cc.cat(f, p.tmp, first, T.rows2(s.out, 0, R - at, ch, ch), &p.cat);
    } else try chain(f, p, k, b, 0, R, conv, in, out);
    try f.tri.shiftConv(conv.ptr, kproj.ptr, R, 1, 3 * C, proj_layer, width, C); // the layer commits the chunk now
    return layer.outProj(f, b, y, k.o, null, R, opt.chip);
}
