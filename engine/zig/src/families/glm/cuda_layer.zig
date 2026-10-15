//! python-0.6's layer_forward for a TP rank (glm5_next/cuda/forward.py, ed78d6f): the attention hyper-connection, the
//! KDA block (here) or the DSA block (cuda_dsa.zig), the MLP hyper-connection, the dense MLP (here) or the MoE
//! (cuda_moe.zig); every projection through cuda_qmm.mm and every rank partial through `gather`, in python's order on
//! python's buffers (cuda_buffers.zig), so the launches are python's (qualification_qualification_7c75d777 PLAN §4 checks them on a CPU). The method is
//! Mia's (MiaAI-Lab, @MiaAI_lab) flashnext/cuda_forward.zig, MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429.
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const fwd = @import("cuda_fwd.zig");
const dsa = @import("cuda_dsa.zig");
const moe = @import("cuda_moe.zig");

const T = trace.T;
const Fwd = fwd.Fwd;
const Buffers = bufs.Buffers;
const State = bufs.State;

/// layer_forward's options: python's nch, host_pos and sparse_np; a prompt chunk's cut and temporaries (cuda_prompt.zig).
pub const Opt = struct { nch: ?usize = null, host_pos: ?usize = null, sparse_np: ?usize = null, chip: qmm.Chip = .{}, prompt: ?*@import("cuda_prompt.zig").Prompt = null };

/// Rows [0, R) of a [rows, cols] buffer, contiguous.
pub fn rowsOf(b: views.Buf, R: usize) T {
    return T.rows2(b, 0, R, b.shape[1], b.shape[1]);
}

/// forward.gather: every rank's fp32 partial b.part[:R] in rank order ([world, R, D] at the returned address).
pub fn gather(f: *Fwd, b: *const Buffers, R: usize) !u64 {
    const d = b.part.shape[1];
    if (f.m.world == 1) return b.part.ptr;
    try f.allGather(T.of(b.part, 0, &.{R * d}), T.of(b.gath, 0, &.{f.m.world * R * d}));
    return b.gath.ptr;
}

/// forward.out_proj: the projection into b.part[:R] (fp32), then the gather.
pub fn outProj(f: *Fwd, b: *const Buffers, x: T, q: views.Linear, xs: ?T, R: usize, chip: qmm.Chip) !u64 {
    try qmm.mm(f, b.prefill, x, q, xs, rowsOf(b.part, R), true, b.sk, chip);
    return gather(f, b, R);
}

/// qmm.group_sums(x, out): each row's 64-input group sums.
fn groupSums(f: *Fwd, x: T, out: views.Buf, R: usize) !T {
    try f.tri.groupSums(x.ptr, x.stride[0], out.ptr, R, x.shape[1]);
    return rowsOf(out, R);
}

/// forward.kda_block, a decode window (no cut): the projections, kda.chain into the layer's scratch, out_proj.
pub fn kdaBlock(f: *Fwd, l: *const views.Layer, st: *const State, b: *const Buffers, R: usize, opt: Opt) !u64 {
    const m = f.m;
    const k = l.kda orelse return error.NotKda;
    if (b.prefill) return @import("cuda_prompt.zig").kdaBlock(f, l, st, b, R, opt); // prompt chunks and cuts (B3)
    const li = m.kindIndex(l.index);
    const width = k.proj.n();
    const rows = st.proj.shape[1];
    const p = T.rows2(st.proj, li * rows * width * 2, R, width, width);
    try qmm.mm(f, false, p_in(b, R), k.proj, rowsOf(b.xs, R), p, false, b.sk, opt.chip);
    const fa = T.rows2(st.proj, p.ptr - st.proj.ptr + k.faOff() * 2, R, 128, width);
    const ga = T.rows2(st.proj, p.ptr - st.proj.ptr + k.gaOff() * 2, R, 128, width);
    try qmm.mm(f, false, fa, k.fb, try groupSums(f, fa, b.xs_fa, R), rowsOf(b.ka, R), false, b.sk, opt.chip);
    try qmm.mm(f, false, ga, k.gb, try groupSums(f, ga, b.xs_ga, R), rowsOf(b.kg, R), false, b.sk, opt.chip);
    const cur: usize = st.parity;
    const ll = k.heads;
    const n = st.rec.shape[1];
    const state = ll * 128 * 128 * 4;
    const s = st.scratch_set;
    const ch = ll * 128;
    try f.kdaChain(.{
        .p = p,
        .b_off = k.bOff(),
        .a = rowsOf(b.ka, R),
        .g = rowsOf(b.kg, R),
        .conv_state = T.of(st.conv, li * st.conv.inner() * 2, st.conv.shape[1..3]),
        .conv_w = T.whole(k.conv),
        .state_in = T.of(st.rec, (cur * n + li) * state, &.{ ll, 128, 128 }),
        .a_log = T.whole(k.a_log),
        .dt_bias = T.whole(k.dt_bias),
        .norm_w = T.whole(k.norm),
        .eps = m.eps,
        .lower = m.lower,
        .rows = R,
        .out = T.of(s.out, li * rows * ch * 2, &.{ rows, ch }),
        .state_out = T.of(st.rec, ((1 - cur) * n + li) * state, &.{ ll, 128, 128 }),
        .k_save = T.of(s.k, li * rows * ch * 4, &.{ rows, ll, 128 }),
        .v_save = T.of(s.v, li * rows * ch * 2, &.{ rows, ll, 128 }),
        .g_save = T.of(s.g, li * rows * ch * 4, &.{ rows, ll, 128 }),
        .b_save = T.of(s.b, li * rows * ll * 4, &.{ rows, ll }),
    });
    const out = T.rows2(s.out, li * rows * ch * 2, R, ch, ch);
    return outProj(f, b, out, k.o, try groupSums(f, out, b.kxs, R), R, opt.chip);
}

/// b.normed[:R], the block's input
fn p_in(b: *const Buffers, R: usize) T {
    return rowsOf(b.normed, R);
}

/// forward.mlp_block: [gate | up], SwiGLU (with its group sums), down into the rank's partial.
pub fn mlpBlock(f: *Fwd, l: *const views.Layer, b: *const Buffers, R: usize, opt: Opt) !u64 {
    const x = l.mlp orelse return error.NotDense;
    try qmm.mm(f, b.prefill, rowsOf(b.normed, R), x.gu, rowsOf(b.xs, R), rowsOf(b.gu, R), false, b.sk, opt.chip);
    try f.tri.swiglu(b.gu.ptr, b.act.ptr, b.xs_act.ptr, f.m.limit, R, b.act.shape[1]);
    return outProj(f, b, rowsOf(b.act, R), x.down, rowsOf(b.xs_act, R), R, opt.chip);
}

/// glue.hc_pre with the layer's `h` and norm into b.normed, b.xs, b.post, b.comb.
fn hcPre(f: *Fwd, h: views.Hc, norm: views.Buf, b: *const Buffers, R: usize) !void {
    const m = f.m;
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
}

/// glue.hc_post(x, x, g, post, comb): the block's gathered partials mixed back into the streams.
fn hcPost(f: *Fwd, g: u64, b: *const Buffers, R: usize) !void {
    const m = f.m;
    try f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
}

/// forward.layer_forward for R rows of a window (prompt chunks and cuts: lane B3).
pub fn layerForward(f: *Fwd, l: *const views.Layer, st: *const State, b: *const Buffers, R: usize, opt: Opt) !void {
    if (R == 0 or R > b.rows) return error.Rows;
    try hcPre(f, l.attn_hc orelse return error.PlainLayer, l.in_norm, b, R);
    const g = switch (l.kind) {
        .kda => try kdaBlock(f, l, st, b, R, opt),
        .dsa => try dsa.block(f, l, st, b, R, opt),
    };
    try hcPost(f, g, b, R);
    try ffnHalf(f, l, b, R, opt);
}

/// layer_forward's FFN half: hc_pre with ffn_hc, the dense MLP or the MoE (cuda_moe.block, or f.moe's provider:
/// python 2.15's AFD branch), then hc_post over the block's planes (the AFD's three: both expert halves, the shared).
pub fn ffnHalf(f: *Fwd, l: *const views.Layer, b: *const Buffers, R: usize, opt: Opt) !void {
    try hcPre(f, l.ffn_hc orelse return error.PlainLayer, l.post_norm, b, R);
    const w = f.m.world;
    const p: fwd.Planes = if (l.mlp != null) .{ .g = try mlpBlock(f, l, b, R, opt), .world = w } else if (f.moe) |x| try x.block(x.ctx, f, l, b, R, opt) else .{ .g = try moe.block(f, l, b, R, opt), .world = w };
    try f.tri.hcPost(b.x.ptr, b.x.ptr, p.g, b.post.ptr, b.comb.ptr, p.world, R, f.m.hidden, f.m.streams);
}
