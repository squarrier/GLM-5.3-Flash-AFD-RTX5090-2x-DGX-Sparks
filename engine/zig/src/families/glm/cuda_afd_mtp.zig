//! The MTP head on the AFD attention node (qualification_21e21bc0; python 2.15 refuses TF_GLM_MTP=1 under AFD, engine.py:499-502, so
//! there is no python twin): python-0.6's mtp_compute as cuda_mtp.zig's computeWith runs it (lane A's DSA block, the
//! draft head), but the MTP layer's MoE goes through f.moe, the AFD provider begun for the MTP slice (Afd.beginMtp:
//! the expert halves' last position, the MTP layer's routed experts on the same expert path as layers 3-44), and its
//! three planes (both halves' partials, then the local shared expert) are summed into the residual, as hc_post sums a
//! backbone layer's. The switch is TF_AFD_MTP=1 on the attention and on both halves; a mismatch is a header breach.
//! qualification_47b9bf3a: `compute` is `front`, the exchange and `back`, so a prompt lane runs the MTP layer as its extra layer
//! (cuda_afd_lanes_mtp.zig) with the same launches.
const std = @import("std");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const prompt = @import("cuda_prompt.zig");
const cc = @import("cuda_compute.zig");
const mtp = @import("cuda_mtp.zig");
const fwd = @import("cuda_fwd.zig");
const Fwd = fwd.Fwd;

const T = trace.T;
const Buffers = bufs.Buffers;
const State = bufs.State;

/// TF_AFD_MTP: 1 serves the MTP layer's routed experts on both halves and loads the MTP head on the attention; unset
/// or 0 leaves both roles as they were.
pub fn fromEnv() !bool {
    const v = std.mem.span(std.c.getenv("TF_AFD_MTP") orelse return false);
    if (std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    std.log.err("TF_AFD_MTP is 1, 0 or unset, not {s}", .{v});
    return error.BadAfdMtp;
}

/// mtp_compute under AFD: computeWith(LaneA)'s launches, the MTP layer's MoE through the provider and its planes
/// summed by the residual add; the logits of the last row (every row unless `last_only`) in b.logits.
pub fn compute(f: *Fwd, w: *const views.Weights, st: *const State, b: *const Buffers, n: usize, opt: layer.Opt, last_only: bool) !T {
    const x = w.mtp orelse return error.NoMtp;
    const p = f.moe orelse return error.NoProvider;
    if (n == 0 or n > b.rows) return error.Rows;
    try front(f, w, st, b, n, opt);
    const planes = try p.block(p.ctx, f, &x.layer, b, n, opt); // the exchange: Afd.block at the MTP slice's position
    return back(f, w, b, n, planes, last_only, opt.chip);
}

/// qualification_47b9bf3a: `compute`'s launches up to the MTP layer's exchange (the embedding, both norms, eh_proj, the DSA block and
/// the post-attention norm into b.normed / b.xs; b.mx, the MTP residual, crosses the exchange).
pub fn front(f: *Fwd, w: *const views.Weights, st: *const State, b: *const Buffers, n: usize, opt: layer.Opt) !void {
    const m = f.m;
    const D = m.hidden;
    const x = w.mtp orelse return error.NoMtp;
    if (n == 0 or n > b.rows) return error.Rows;
    try f.tri.embed(b.ids.ptr, cc.table(w.embed), b.me.ptr, n, D, 1);
    if (b.zero_first) try cc.zero(f, T.of(b.me, 0, &.{D}));
    try f.tri.rmsnorm(b.me.ptr, D, x.enorm.ptr, b.mcat.ptr, 2 * D, null, m.eps, n, D);
    try f.tri.rmsnorm(b.hin.ptr, D, x.hnorm.ptr, b.mcat.ptr + D * 2, 2 * D, null, m.eps, n, D);
    const cat = layer.rowsOf(b.mcat, n);
    var sums: ?T = null;
    if (!b.prefill) {
        try f.tri.groupSums(cat.ptr, cat.stride[0], b.mxs.ptr, n, 2 * D);
        sums = layer.rowsOf(b.mxs, n);
    }
    try qmm.mm(f, b.prefill, cat, x.eh, sums, layer.rowsOf(b.mx, n), false, b.sk, opt.chip);
    const l = &x.layer;
    try f.tri.rmsnorm(b.mx.ptr, D, l.in_norm.ptr, b.normed.ptr, D, b.xs.ptr, m.eps, n, D);
    const g = try prompt.dsaBlock(f, l, st, b, n, opt);
    try f.tri.residualAdd(b.mx.ptr, b.mx.ptr, g, m.world, n, D);
    try f.tri.rmsnorm(b.mx.ptr, D, l.post_norm.ptr, b.normed.ptr, D, b.xs.ptr, m.eps, n, D);
}

/// qualification_47b9bf3a: `compute`'s launches after the exchange: the MTP layer's planes summed into b.mx, the final norm and the
/// draft head on the last row (every row unless `last_only`).
pub fn back(f: *Fwd, w: *const views.Weights, b: *const Buffers, n: usize, planes: fwd.Planes, last_only: bool, chip: qmm.Chip) !T {
    const m = f.m;
    const D = m.hidden;
    const x = w.mtp orelse return error.NoMtp;
    // the world-1 Triton set has _residual_add at WORLD 1 only (hc_post takes three planes, this kernel does not): each
    // plane added in turn, plane 0 first (both halves' partials, then the shared expert); drafts only propose
    for (0..planes.world) |i| try f.tri.residualAdd(b.mx.ptr, b.mx.ptr, planes.g + i * n * D * 4, 1, n, D);
    const lo = if (last_only) n - 1 else 0;
    const k = n - lo;
    try f.tri.rmsnorm(b.mx.ptr + lo * D * 2, D, x.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, m.eps, k, D);
    const head: views.Linear = if (w.draft_head) |q| .{ .q4 = q } else w.head; // the head's logits only draft
    const out = T.rows2(b.logits, 0, k, w.head.n(), b.logits.shape[1]);
    try qmm.mm(f, b.prefill, layer.rowsOf(b.fnormed, k), head, layer.rowsOf(b.fxs, k), out, false, b.sk, chip);
    return out;
}

/// mtp.mtp_forward under AFD: stage, then `compute` at the MTP cache's length (the caller advances it, setMtpLen); `p`,
/// a prompt chunk's context on prefill's absorb (its DSA Scratch).
pub fn forward(f: *Fwd, w: *const views.Weights, st: *const State, b: *Buffers, next: []const u32, hidden: T, chip: qmm.Chip, p: ?*prompt.Prompt) !T {
    const n = try mtp.stage(f, st, b, next, hidden);
    return compute(f, w, st, b, n, options(st, n, chip, p), true);
}

/// The MTP step's options for `n` staged rows: the chunks through the MTP cache's length and that length (qualification_f16009a8's key).
pub fn options(st: *const State, n: usize, chip: qmm.Chip, p: ?*prompt.Prompt) layer.Opt {
    const nch = (st.mtp_len + n + cc.chunk - 1) / cc.chunk;
    return .{ .nch = nch, .host_pos = st.mtp_len, .chip = chip, .prompt = p };
}

/// The MTP head's options for `n` rows at the MTP cache's length (`forward`'s).
pub fn optAt(st: *const State, n: usize, chip: qmm.Chip, p: ?*prompt.Prompt) layer.Opt {
    const nch = (st.mtp_len + n + cc.chunk - 1) / cc.chunk;
    return .{ .nch = nch, .host_pos = st.mtp_len, .chip = chip, .prompt = p };
}

/// qualification_47b9bf3a: `forward` to the MTP layer's exchange (a prompt lane's extra layer, cuda_afd_lanes_mtp.zig): stage, then
/// `front`; the options its exchange and `back` take.
pub fn forwardFront(f: *Fwd, w: *const views.Weights, st: *const State, b: *Buffers, next: []const u32, hidden: T, chip: qmm.Chip, p: ?*prompt.Prompt) !layer.Opt {
    if (w.mtp == null) return error.NoMtp;
    const n = try mtp.stage(f, st, b, next, hidden);
    const opt = optAt(st, n, chip, p);
    try front(f, w, st, b, n, opt);
    return opt;
}
