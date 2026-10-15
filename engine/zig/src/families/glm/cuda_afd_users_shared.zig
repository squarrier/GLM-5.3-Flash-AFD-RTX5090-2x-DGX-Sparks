//! shared_forward (TF_AFD_USERS_SHARED=1, stage 1): the users round as one forward over every user's row. The KDA and DSA
//! blocks split into a row-by-row half (once over all rows) and a per-sequence half (each user's chain or attention at
//! R = 1 on its own State, at its own position), as @Bizuayeu split them for shared verify windows
//! (github.com/Bizuayeu/TensorFold pr/glm53-flash-cuda, Apache-2.0: 75131b66 "a round's verify windows share one
//! forward, each stream's state on its own", 146e113f cuda_shared.zig's mixer, d3cc052c slots). Reimplemented over this
//! tree's q4/EXL3 kernels and buffers from his design; no code of his is copied. Each half below calls exactly the
//! launches cuda_layer.zig's kdaBlock and cuda_dsa.zig's partialWith make, with the same arguments for a row.
//! Ash Hart authored TensorFold/MCDMA; Mia (@MiaAI_lab) authored the EXL3 recipe; this file: Claude Opus 5.5
//! (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const dsa = @import("cuda_dsa.zig");
const dk = @import("cuda_dsa_kernels.zig");
const tri = @import("cuda_triton.zig");
const cc = @import("cuda_compute.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;
const Buf = views.Buf;

/// The users' two stages: every row's KDA in-projection (each copied to its user's st.proj), and the chains' outputs
/// (row i by row i's user) for one out-projection.
pub const Stages = struct {
    proj: Buf,
    out: Buf,

    pub fn init(m: *const md.Model, rows: usize) Stages {
        return .{ .proj = Buf.of(.bf16, &.{ rows, m.kdaWidth() }), .out = Buf.of(.bf16, &.{ rows, m.ll() * 128 }) };
    }
};

fn rowsOf(x: Buf, R: usize) T {
    return layer.rowsOf(x, R);
}

/// qmm.group_sums into `out`'s first R rows (cuda_layer.zig's groupSums).
fn groupSums(f: *Fwd, x: T, out: Buf, R: usize) !T {
    try f.tri.groupSums(x.ptr, x.stride[0], out.ptr, R, x.shape[1]);
    return rowsOf(out, R);
}

/// Row `i` of a row-major buffer as a one-row buffer.
pub fn rowOf(x: Buf, i: usize) Buf {
    var y = x;
    y.ptr += i * x.inner() * x.dtype.size();
    y.shape[0] = 1;
    return y;
}

/// A per-sequence call's Buffers: `b` at R = 1 with the rows it reads or writes moved to row `i`; every scratch at its
/// base, as the users' R = 1 view (cuda_afd_users.zig Saved.view) leaves it.
pub fn rowView(b: bufs.Buffers, i: usize) bufs.Buffers {
    var v = b;
    v.rows = 1;
    v.lat = rowOf(b.lat, i);
    v.ikr = rowOf(b.ikr, i);
    v.igr = rowOf(b.igr, i);
    v.qi = rowOf(b.qi, i);
    v.ka = rowOf(b.ka, i);
    v.kg = rowOf(b.kg, i);
    v.lat_s.qa = rowOf(b.lat_s.qa, i);
    v.lat_s.ol = rowOf(b.lat_s.ol, i);
    return v;
}

// -- KDA (cuda_layer.zig kdaBlock, decode) --------------------------------------------------------------------------

/// kdaBlock's projections over `n` rows: the q4 in-projection into the stage, then f_a and g_a (their group sums and
/// q4 projections) into b.ka and b.kg.
pub fn kdaIn(f: *Fwd, l: *const views.Layer, b: *const bufs.Buffers, s: Stages, n: usize, chip: qmm.Chip) !void {
    const k = l.kda orelse return error.NotKda;
    const width = k.proj.n();
    if (n == 0 or n > s.proj.shape[0] or width != s.proj.shape[1]) return error.Rows;
    const p = T.rows2(s.proj, 0, n, width, width);
    try qmm.mm(f, false, rowsOf(b.normed, n), k.proj, rowsOf(b.xs, n), p, false, b.sk, chip);
    const fa = T.rows2(s.proj, k.faOff() * 2, n, 128, width);
    const ga = T.rows2(s.proj, k.gaOff() * 2, n, 128, width);
    try qmm.mm(f, false, fa, k.fb, try groupSums(f, fa, b.xs_fa, n), rowsOf(b.ka, n), false, b.sk, chip);
    try qmm.mm(f, false, ga, k.gb, try groupSums(f, ga, b.xs_ga, n), rowsOf(b.kg, n), false, b.sk, chip);
}

/// kdaBlock's chain for row `i` on its user's State: its projected row into its own st.proj (the commit's conv shift
/// reads it there), the chain at R = 1 from its own conv window and recurrent state (its saves on its State), the
/// output into the stage's row i.
pub fn kdaSeq(f: *Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, s: Stages, i: usize) !void {
    const m = f.m;
    const k = l.kda orelse return error.NotKda;
    const li = m.kindIndex(l.index);
    const width = k.proj.n();
    const rows = st.proj.shape[1];
    const p = T.rows2(st.proj, li * rows * width * 2, 1, width, width);
    try cc.copyDtoD(f, p, T.rows2(s.proj, i * width * 2, 1, width, width));
    const cur: usize = st.parity;
    const ll = k.heads;
    const n = st.rec.shape[1];
    const state = ll * 128 * 128 * 4;
    const ss = st.scratch_set;
    const ch = ll * 128;
    try f.kdaChain(.{
        .p = p,
        .b_off = k.bOff(),
        .a = T.rows2(b.ka, i * ch * 2, 1, ch, ch),
        .g = T.rows2(b.kg, i * ch * 2, 1, ch, ch),
        .conv_state = T.of(st.conv, li * st.conv.inner() * 2, st.conv.shape[1..3]),
        .conv_w = T.whole(k.conv),
        .state_in = T.of(st.rec, (cur * n + li) * state, &.{ ll, 128, 128 }),
        .a_log = T.whole(k.a_log),
        .dt_bias = T.whole(k.dt_bias),
        .norm_w = T.whole(k.norm),
        .eps = m.eps,
        .lower = m.lower,
        .rows = 1,
        .out = T.of(s.out, i * ch * 2, &.{ 1, ch }),
        .state_out = T.of(st.rec, ((1 - cur) * n + li) * state, &.{ ll, 128, 128 }),
        .k_save = T.of(ss.k, li * rows * ch * 4, &.{ rows, ll, 128 }),
        .v_save = T.of(ss.v, li * rows * ch * 2, &.{ rows, ll, 128 }),
        .g_save = T.of(ss.g, li * rows * ch * 4, &.{ rows, ll, 128 }),
        .b_save = T.of(ss.b, li * rows * ll * 4, &.{ rows, ll }),
    });
}

/// kdaBlock's out-projection over `n` rows: the chains' outputs' group sums and the q4 o into b.part[:n] (gathered).
pub fn kdaOut(f: *Fwd, l: *const views.Layer, b: *const bufs.Buffers, s: Stages, n: usize, chip: qmm.Chip) !u64 {
    const k = l.kda orelse return error.NotKda;
    const ch = k.heads * 128;
    const out = T.rows2(s.out, 0, n, ch, ch);
    return layer.outProj(f, b, out, k.o, try groupSums(f, out, b.kxs, n), n, chip);
}

// -- DSA (cuda_dsa.zig partialWith, decode) -------------------------------------------------------------------------

/// partialWith's row-by-row launches over `n` rows: the q4 a.proj, both norms, q_b, the indexer's keys and pool gate
/// (`indexed`: the States hold index caches), absorb_q, and the indexer's queries (`sparse`: some row's attention runs
/// past the dense limit).
pub fn dsaIn(f: *Fwd, l: *const views.Layer, b: *const bufs.Buffers, n: usize, indexed: bool, sparse: bool, chip: qmm.Chip) !void {
    const m = f.m;
    const a = l.dsa orelse return error.NotDsa;
    if (n == 0 or n > b.rows) return error.Rows;
    const HL = a.heads;
    const qk = m.qk_dim;
    const dw = b.dp.shape[1];
    try qmm.mm(f, b.prefill, rowsOf(b.normed, n), a.proj, rowsOf(b.xs, n), rowsOf(b.dp, n), false, b.sk, chip);
    try f.tri.rmsnorm(b.dp.ptr, dw, a.q_norm.ptr, b.qr.ptr, m.q_lora, b.xs_qr.ptr, m.eps, n, m.q_lora);
    try f.tri.rmsnorm(b.dp.ptr + m.q_lora * 2, dw, a.kv_norm.ptr, b.lat.ptr, m.kv_lora, b.xs_lat.ptr, m.eps, n, dw - m.q_lora);
    try qmm.mm(f, b.prefill, rowsOf(b.qr, n), a.q_b, rowsOf(b.xs_qr, n), T.rows2(b.q, 0, n, HL * qk, HL * qk), false, b.sk, chip);
    const ab = dsa.absorbOf(a.absorb orelse return error.NotPorted);
    if (indexed) { // indexUpdate's projections (its per-sequence update is dsaSeq's)
        const ix = a.index orelse return error.NoIndexer;
        const normed = rowsOf(b.normed, n);
        try qmm.mm(f, b.prefill, normed, ix.kw, rowsOf(b.xs, n), rowsOf(b.ikr, n), false, b.sk, chip);
        const ne = ix.gate.shape[0];
        if (b.gath.numel() < tri.router_ks * n * ne) return error.RouterScratch;
        try f.tri.router(normed.ptr, normed.stride[0], ix.gate.ptr, b.igr.ptr, b.igr.shape[1], b.gath.ptr, n, ix.gate.shape[1], ne);
    }
    try f.tri.absorbQ(b.q.ptr, ab.w, b.lat_s.qa.ptr, n, HL, qk, ab.lw);
    if (sparse) {
        const ix = a.index orelse return error.NoIndexer;
        try qmm.mm(f, b.prefill, rowsOf(b.qr, n), ix.qb, rowsOf(b.xs_qr, n), rowsOf(b.qi, n), false, b.sk, chip);
    }
}

/// What dsaIn needs from row `v` (a rowView) on `st`: its index caches, and whether its attention runs sparse.
pub const Need = struct { indexed: bool, sparse: bool };

pub fn dsaNeed(m: *const md.Model, l: *const views.Layer, st: *const bufs.State, v: *const bufs.Buffers, opt: layer.Opt) !Need {
    const c = try dsa.caches(m, l, st);
    const sh = try dsa.shape(m, l, st, v, 1, opt);
    return .{ .indexed = c.index != null, .sparse = sh.sparse };
}

/// partialWith's per-sequence launches for row view `v` (R = 1) on its user's State at its own position: the latent
/// write, the indexer update, the dense chunks or the select blocks, tail and sparse attention, into v's lat_s.ol.
pub fn dsaSeq(f: *Fwd, l: *const views.Layer, st: *const bufs.State, v: *const bufs.Buffers, opt: layer.Opt, x: ?*const dk.Scratch) !void {
    const m = f.m;
    const a = l.dsa orelse return error.NotDsa;
    const R: usize = 1;
    if (v.rows != R) return error.Rows;
    const c = try dsa.caches(m, l, st);
    const HL = a.heads;
    const qk = m.qk_dim;
    const ab = dsa.absorbOf(a.absorb orelse return error.NotPorted);
    try f.tri.latentWrite(v.lat.ptr, v.lat.shape[1], c.lc.ptr, c.pos.ptr, R, c.lc.shape[1]);
    const sh = try dsa.shape(m, l, st, v, R, opt);
    if (c.index) |ic| { // indexUpdate's last launch on this row's keys and gates
        const ix = a.index orelse return error.NoIndexer;
        try f.tri.indexUpdate(v.ikr.ptr, v.ikr.shape[1], v.igr.ptr, .{ .ln_w = ix.ln_w.ptr, .ln_b = ix.ln_b.ptr, .ape = ix.ape.ptr }, .{ .ik = ic[0].ptr, .ig = ic[1].ptr, .pk = ic[2].ptr }, c.pos.ptr, R, m.index_dim);
    }
    const qa = v.lat_s.qa;
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(qk))));
    const ol = v.lat_s.ol;
    const s = v.lat_s;
    if (sh.dense) // dense_attention: one block of one row at its position (part_rows of a one-row view is 1)
        try f.tri.latentAttention(qa.ptr, c.lc.ptr, c.pos.ptr, .{ .po = s.po.ptr, .pm = s.pm.ptr, .pl = s.pl.ptr, .aux = s.dummy.ptr }, ol.ptr, R, HL, ab.lw, sh.nch, tri.headBlock(R), scale);
    if (sh.sparse) {
        const ic = c.index.?;
        const tmp = try dsa.temps(m, v, R, sh.np, x);
        try dsa.selectBlocks(f, v, c.pos, ic[2].ptr, tmp, R, sh.np);
        try dsa.selectTail(f, tmp.cat orelse tmp.pools, c.pos, R, tmp.tokens, tmp.counts);
        try f.tri.latentSparse(qa.ptr, c.lc.ptr, tmp.tokens.ptr, tmp.counts.ptr, ol.ptr, tmp.part, R, HL, ab.lw, dk.width, scale);
    }
}

/// partialWith's last row-by-row launches over `n` rows: expand_v, the group sums and the q4 o into b.part[:n]
/// (gathered, blockWith's).
pub fn dsaOut(f: *Fwd, l: *const views.Layer, b: *const bufs.Buffers, n: usize, chip: qmm.Chip) !u64 {
    const a = l.dsa orelse return error.NotDsa;
    const HL = a.heads;
    const ab = dsa.absorbOf(a.absorb orelse return error.NotPorted);
    try f.tri.expandV(b.lat_s.ol.ptr, ab.w, b.vn.ptr, n, HL, ab.v, ab.lw);
    const o = T.rows2(b.vn, 0, n, HL * ab.v, HL * ab.v);
    try f.tri.groupSums(o.ptr, o.stride[0], b.xs_ao.ptr, n, o.shape[1]);
    try qmm.mm(f, b.prefill, o, a.o, rowsOf(b.xs_ao, n), rowsOf(b.part, n), true, b.sk, chip);
    return layer.gather(f, b, n);
}

test "shared_forward row view: rows moved to row i, scratch at its base, one row" {
    const m = try md.Model.from(&@import("config.zig").Config{}, .{}, 1, 0);
    var b = try bufs.Buffers.init(&m, 4, 2560, false);
    b.lat.ptr = 0x100000;
    b.lat_s.qa.ptr = 0x200000;
    b.lat_s.po.ptr = 0x300000;
    b.ka.ptr = 0x400000;
    const v = rowView(b, 3);
    try std.testing.expectEqual(@as(usize, 1), v.rows);
    try std.testing.expectEqual(b.lat.ptr + 3 * m.kv_lora * 2, v.lat.ptr);
    try std.testing.expectEqual(b.lat_s.qa.ptr + 3 * m.hl() * m.kv_lora * 2, v.lat_s.qa.ptr);
    try std.testing.expectEqual(b.ka.ptr + 3 * m.ll() * 128 * 2, v.ka.ptr);
    try std.testing.expectEqual(b.lat_s.po.ptr, v.lat_s.po.ptr);
    try std.testing.expectEqual(@as(usize, 1), v.lat.shape[0]);
    try std.testing.expectEqual(b.lat.shape[1], v.lat.shape[1]);
    const s = Stages.init(&m, 4);
    try std.testing.expectEqual(@as(usize, 4 * m.kdaWidth() * 2), s.proj.bytes());
    try std.testing.expectEqual(@as(usize, 4 * m.ll() * 128 * 2), s.out.bytes());
    std.debug.print("shared_forward row view: row 3 of lat/qa/ka moved, scratch at its base, one row; stages {d} + {d} B at 4 rows\n", .{ s.proj.bytes(), s.out.bytes() });
}
