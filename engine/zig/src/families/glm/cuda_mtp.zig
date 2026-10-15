//! python-0.6's MTP head (glm5_next/cuda/mtp.py, ed78d6f): mtp_stage puts the next tokens and the main rows' hidden
//! (final-normed) into the MTP buffers; mtp_compute embeds the tokens (a step from position 0 zeroes its first row),
//! norms both inputs, projects [embedding | hidden] through eh, runs the MTP layer (lane A's DSA and MoE blocks around a
//! plain residual) and gives the head's logits (EXL3: the 4-bit draft head). Launches as cuda_compute.zig's: python's.
const std = @import("std");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const prompt = @import("cuda_prompt.zig");
const moe = @import("cuda_moe.zig");
const cc = @import("cuda_compute.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;
const Buffers = bufs.Buffers;
const State = bufs.State;

/// The MTP layer's blocks: lane A's (cuda_dsa.zig picks the MTP caches for the plain layer, as mtp.py passes them),
/// the DSA block with the engine's Scratch on a prompt chunk's absorb (cuda_prompt.dsaBlock).
pub const LaneA = struct {
    pub const dsaBlock = prompt.dsaBlock;
    pub const moeBlock = moe.block;
};

/// mtp.mtp_stage: the next tokens into b.ids and the main rows' hidden into b.hin (unless it is b.hin); gives n.
pub fn stage(f: *Fwd, st: *const State, b: *Buffers, next: []const u32, hidden: T) !usize {
    const n = next.len;
    if (n == 0 or n > b.rows or hidden.shape[0] != n) return error.Rows;
    b.zero_first = st.mtp_len == 0;
    try cc.checkRoom(f.m, st, n, st.mtp_len);
    try cc.stageIds(f, b, next);
    if (hidden.ptr != b.hin.ptr) try cc.copyDtoD(f, layer.rowsOf(b.hin, n), hidden);
    try cc.stagedRecord(f, b);
    return n;
}

/// mtp.mtp_compute (decode windows; prompt chunks: lane B3): the logits' view of the last row (every row unless
/// `last_only`) in b.logits.
pub fn compute(f: *Fwd, w: *const views.Weights, st: *const State, b: *const Buffers, n: usize, opt: layer.Opt, last_only: bool) !T {
    return computeWith(LaneA, f, w, st, b, n, opt, last_only);
}

/// mtp_compute with the MTP layer's blocks from `B` (LaneA; a test's markers).
pub fn computeWith(comptime B: type, f: *Fwd, w: *const views.Weights, st: *const State, b: *const Buffers, n: usize, opt: layer.Opt, last_only: bool) !T {
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
    const g = try B.dsaBlock(f, l, st, b, n, opt);
    try f.tri.residualAdd(b.mx.ptr, b.mx.ptr, g, m.world, n, D);
    try f.tri.rmsnorm(b.mx.ptr, D, l.post_norm.ptr, b.normed.ptr, D, b.xs.ptr, m.eps, n, D);
    const g2 = try B.moeBlock(f, l, b, n, opt);
    try f.tri.residualAdd(b.mx.ptr, b.mx.ptr, g2, m.world, n, D);
    const lo = if (last_only) n - 1 else 0;
    const k = n - lo;
    try f.tri.rmsnorm(b.mx.ptr + lo * D * 2, D, x.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, m.eps, k, D);
    const head: views.Linear = if (w.draft_head) |q| .{ .q4 = q } else w.head; // the head's logits only draft
    const out = T.rows2(b.logits, 0, k, w.head.n(), b.logits.shape[1]);
    try qmm.mm(f, b.prefill, layer.rowsOf(b.fnormed, k), head, layer.rowsOf(b.fxs, k), out, false, b.sk, opt.chip);
    return out;
}

/// mtp.mtp_forward: stage, then compute at the MTP cache's length (the caller advances it, cuda_compute.setMtpLen).
pub fn forward(f: *Fwd, w: *const views.Weights, st: *const State, b: *Buffers, next: []const u32, hidden: T, chip: qmm.Chip, last_only: bool) !T {
    const n = try stage(f, st, b, next, hidden);
    const nch = (st.mtp_len + n + cc.chunk - 1) / cc.chunk;
    return compute(f, w, st, b, n, .{ .nch = nch, .host_pos = st.mtp_len, .chip = chip }, last_only);
}
