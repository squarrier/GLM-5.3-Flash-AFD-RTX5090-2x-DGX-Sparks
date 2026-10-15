//! The DSA block (python-0.6 ed78d6f forward.dsa_block on the latent cache: _dsa_latent, dense_attention, with
//! latent.py's and sparse.py's host parts): q_a/kv_a, the norms, q_b, the latent write, the indexer update, absorb,
//! dense chunks, past the dense limit the top-512 pools and latent sparse attention, expand, out_proj; python's
//! launches in python's order (qualification_qualification_7c75d777 PLAN §4). The MTP layer (index m.layers) takes mtp.py's caches (qualification_qualification_3a5e8e9a ANSWER).
//! Python's temporaries (torch.empty in the wrappers) live in Buffers free inside the block (`Temps`). Prompt chunks
//! (A4): dense blocks of 512 rows and select blocks of 512 rows, each at its first row's position (a torch add,
//! `pos_dev + r0`), the select blocks' pools joined by torch.cat, and sparse partials past b.lat_s in dk.Scratch.
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const tri = @import("cuda_triton.zig");
const dk = @import("cuda_dsa_kernels.zig");
const layer = @import("cuda_layer.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;
const Opt = layer.Opt;

const T = trace.T;
const Buf = views.Buf;

/// The blocks a chunk may take (decode.PREFILL_ROWS of 2,048 rows: 4; TENSORFOLD_PREFILL_ROWS' most, 16,384: 32).
pub const max_blocks = 32;

/// forward.dsa_block for R rows: the rank's attention partial gathered ([world, R, D] fp32 at the returned address).
pub fn block(f: *Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: Opt) !u64 {
    return blockWith(f, l, st, b, R, opt, scratchOf(f));
}

/// `block` with `x` for the sparse partials b.lat_s cannot hold (a trace's own Scratch; on a GPU Kernels.dsa's).
pub fn blockWith(f: *Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: Opt, x: ?*const dk.Scratch) !u64 {
    try partialWith(f, l, st, b, R, opt, x);
    return layer.gather(f, b, R);
}

/// The engine's Scratch, reserved beside glm_select's image (null: none, or a trace). shared_forward: pub (its per-user part).
pub fn scratchOf(f: *const Fwd) ?*const dk.Scratch {
    const g = f.gpu orelse return null;
    const k = g.kernels.dsa orelse return null;
    return k.scratch;
}

/// The caches a DSA layer reads: State.kc[i], pos_dev and index[i] (i: the layer's place among the DSA layers), or for
/// the MTP layer (mtp.py) st.mtp_kc, st.mtp_pos_dev and the indexer's last cache.
pub const Caches = struct { lc: Buf, pos: Buf, index: ?bufs.IndexCache };

pub fn caches(m: *const md.Model, l: *const views.Layer, st: *const bufs.State) !Caches {
    if (l.index == m.layers) return .{
        .lc = st.mtp_kc orelse return error.NoMtpCache,
        .pos = st.mtp_pos_dev,
        .index = if (st.index) |x| x[x.len - 1] else null,
    };
    if (l.index > m.layers or m.kinds[l.index] != .dsa) return error.NotDsa;
    const di = m.kindIndex(l.index); // python's st.dsa_index[layer.index]
    if (di >= st.kc.len) return error.NotDsa;
    return .{ .lc = st.kc[di], .pos = st.pos_dev, .index = if (st.index) |x| x[di] else null };
}

/// Where python's temporaries of one call sit: the router's K slices in b.gath (until the gather), the pool scores
/// and sparse attention's partials in b.lat_s (after the dense merge; partials past it in the Scratch), the select
/// blocks' pools, their cat and the counts in b.kn (unused on the latent path), the tokens in b.gath; each apart from
/// every other argument of the launches that take it. `pools` holds every block's rows; `cat` is null for one block.
pub const Temps = struct { scores: u64, pools: T, cat: ?T, tokens: T, counts: T, part: tri.Partials };

pub fn temps(m: *const md.Model, b: *const bufs.Buffers, R: usize, np: usize, x: ?*const dk.Scratch) !Temps {
    const H = m.hl();
    const lw = b.lat_s.qa.shape[2];
    const nch = tri.chunksFor(dk.width);
    const B = @min(R, tri.select_rows);
    const pool_bytes = R * dk.topk_pools * 8;
    const kn = pool_bytes * @as(usize, if (R > B) 2 else 1) + R * 4;
    if (B * np * 4 > b.lat_s.po.bytes() or kn > b.kn.bytes() or R * dk.width * 4 > b.gath.bytes()) return error.DsaScratch;
    var pools = T.of(b.kn, 0, &.{ R, dk.topk_pools });
    pools.dtype = .i64;
    var cat: ?T = null;
    if (R > B) {
        cat = pools;
        cat.?.ptr += pool_bytes;
    }
    var counts = T.of(b.kn, kn - R * 4, &.{R});
    counts.dtype = .i32;
    var tokens = T.of(b.gath, 0, &.{ R, dk.width });
    tokens.dtype = .i32;
    const s = b.lat_s;
    const part: tri.Partials = if (nch * R * H * lw <= s.po.numel() and nch * R * H <= s.pm.numel())
        .{ .po = s.po.ptr, .pm = s.pm.ptr, .pl = s.pl.ptr }
    else blk: { // python's torch.empty of [nch * R * H * lw] fp32: 640 MiB at 2,048 rows
        const y = x orelse return error.NoDsaScratch;
        if (nch * R * H * lw > y.po.numel() or nch * R * H > y.pm.numel()) return error.DsaScratch;
        break :blk .{ .po = y.po.ptr, .pm = y.pm.ptr, .pl = y.pl.ptr };
    };
    return .{ .scores = s.po.ptr, .pools = pools, .cat = cat, .tokens = tokens, .counts = counts, .part = part };
}

/// latent.AbsorbQ4 or AbsorbW as the Triton wrappers take them, with the latent width and value rows (a.lw, a.v_dim).
pub fn absorbOf(a: views.Absorb) struct { w: tri.Absorb, lw: usize, v: usize } {
    return switch (a) {
        .q4 => |x| .{ .w = .{ .q4 = .{ .kw = x.kw.ptr, .ks = x.ks.ptr, .kb = x.kb.ptr, .vw = x.vw.ptr, .vs = x.vs.ptr, .vb = x.vb.ptr } }, .lw = x.kw.shape[2] * 8, .v = x.vw.shape[1] },
        .b16 => |x| .{ .w = .{ .b16 = .{ .wk = x.wk.ptr, .wv = x.wv.ptr } }, .lw = x.wk.shape[2], .v = x.wv.shape[1] },
    };
}

/// What a DSA block's launches read on the host beyond its buffers (qualification_f16009a8's graph key): whether the dense chunks run and
/// how many (`nch`), whether sparse rows run and over how many pools (`np`).
pub const Shape = struct { dense: bool = false, nch: usize = 0, sparse: bool = false, np: usize = 0 };

/// The rows a dense attention block takes (a prompt chunk's blocks of prompt_att_rows, a decode window's all).
fn partRows(b: *const bufs.Buffers) usize {
    return @min(b.rows, if (b.prefill) bufs.prompt_att_rows else b.rows);
}

/// partialWith's host choices for R rows at opt's position: dense chunks before the dense limit, the pools past it.
pub fn shape(m: *const md.Model, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: Opt) !Shape {
    const a = l.dsa orelse return error.NotDsa;
    const c = try caches(m, l, st);
    const dl = m.denseLimit();
    const all_sparse = opt.sparse_np != null or (opt.host_pos != null and opt.host_pos.? >= dl);
    var s: Shape = .{ .dense = !all_sparse, .sparse = c.index != null and (all_sparse or (opt.host_pos != null and opt.host_pos.? + R - 1 >= dl)) };
    if (s.dense) {
        const s_nch = b.lat_s.pm.numel() / (partRows(b) * a.heads);
        s.nch = @min(opt.nch orelse s_nch, s_nch);
    }
    if (s.sparse) s.np = opt.sparse_np orelse tri.poolBucket(opt.host_pos.?, R, c.index.?[2].shape[0] - 2);
    return s;
}

/// dsa_block up to its gather: b.part[:R] (fp32) = the rank's heads' attention through o.
pub fn partial(f: *Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: Opt) !void {
    return partialWith(f, l, st, b, R, opt, scratchOf(f));
}

/// `partial` with `x` (blockWith's).
pub fn partialWith(f: *Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: Opt, x: ?*const dk.Scratch) !void {
    const m = f.m;
    const a = l.dsa orelse return error.NotDsa;
    if (R == 0 or R > b.rows) return error.Rows;
    const c = try caches(m, l, st);
    const HL = a.heads;
    const qk = m.qk_dim;
    const dw = b.dp.shape[1];
    try qmm.mm(f, b.prefill, rowsOf(b.normed, R), a.proj, rowsOf(b.xs, R), rowsOf(b.dp, R), false, b.sk, opt.chip);
    try f.tri.rmsnorm(b.dp.ptr, dw, a.q_norm.ptr, b.qr.ptr, m.q_lora, b.xs_qr.ptr, m.eps, R, m.q_lora);
    try f.tri.rmsnorm(b.dp.ptr + m.q_lora * 2, dw, a.kv_norm.ptr, b.lat.ptr, m.kv_lora, b.xs_lat.ptr, m.eps, R, dw - m.q_lora);
    try qmm.mm(f, b.prefill, rowsOf(b.qr, R), a.q_b, rowsOf(b.xs_qr, R), T.rows2(b.q, 0, R, HL * qk, HL * qk), false, b.sk, opt.chip);
    const ab = absorbOf(a.absorb orelse return error.NotPorted); // TF_GLM_LATENT=0's per-head caches: not ported
    // _dsa_latent
    try f.tri.latentWrite(b.lat.ptr, b.lat.shape[1], c.lc.ptr, c.pos.ptr, R, c.lc.shape[1]);
    const sh = try shape(m, l, st, b, R, opt);
    if (c.index) |ic| try indexUpdate(f, a, ic, c.pos, b, R, opt);
    const qa = b.lat_s.qa;
    try f.tri.absorbQ(b.q.ptr, ab.w, qa.ptr, R, HL, qk, ab.lw);
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(qk))));
    const ol = b.lat_s.ol;
    const s = b.lat_s;
    if (sh.dense) { // dense_attention: blocks of part_rows rows, each at its first row's position
        const part_rows = partRows(b);
        const nch = sh.nch;
        const hb = tri.headBlock(R); // latent.head_block of all R rows, every block's
        const row = HL * ab.lw * 2;
        var r0: usize = 0;
        while (r0 < R) : (r0 += part_rows) {
            const at = if (r0 == 0) c.pos.ptr else try posAdd(f, c.pos, r0, try atSlot(b, R, r0 / part_rows));
            try f.tri.latentAttention(qa.ptr + r0 * row, c.lc.ptr, at, .{ .po = s.po.ptr, .pm = s.pm.ptr, .pl = s.pl.ptr, .aux = s.dummy.ptr }, ol.ptr + r0 * row, @min(part_rows, R - r0), HL, ab.lw, nch, hb, scale);
        }
    }
    if (sh.sparse) {
        const ix = a.index orelse return error.NoIndexer;
        const ic = c.index.?;
        try qmm.mm(f, b.prefill, rowsOf(b.qr, R), ix.qb, rowsOf(b.xs_qr, R), rowsOf(b.qi, R), false, b.sk, opt.chip);
        const np = sh.np;
        const tmp = try temps(m, b, R, np, x);
        try selectBlocks(f, b, c.pos, ic[2].ptr, tmp, R, np);
        try selectTail(f, tmp.cat orelse tmp.pools, c.pos, R, tmp.tokens, tmp.counts);
        try f.tri.latentSparse(qa.ptr, c.lc.ptr, tmp.tokens.ptr, tmp.counts.ptr, ol.ptr, tmp.part, R, HL, ab.lw, dk.width, scale);
    }
    try f.tri.expandV(ol.ptr, ab.w, b.vn.ptr, R, HL, ab.v, ab.lw);
    const o = T.rows2(b.vn, 0, R, HL * ab.v, HL * ab.v);
    try f.tri.groupSums(o.ptr, o.stride[0], b.xs_ao.ptr, R, o.shape[1]);
    try qmm.mm(f, b.prefill, o, a.o, rowsOf(b.xs_ao, R), rowsOf(b.part, R), true, b.sk, opt.chip);
}

/// The indexer's update (long contexts): keys and head weights, the pool gate (glue.router, its K slices in b.gath),
/// then sparse.index_update's keys, gates and the pools the window completes.
fn indexUpdate(f: *Fwd, a: views.Dsa, ic: bufs.IndexCache, pos: Buf, b: *const bufs.Buffers, R: usize, opt: Opt) !void {
    const ix = a.index orelse return error.NoIndexer;
    const normed = rowsOf(b.normed, R);
    try qmm.mm(f, b.prefill, normed, ix.kw, rowsOf(b.xs, R), rowsOf(b.ikr, R), false, b.sk, opt.chip);
    const ne = ix.gate.shape[0];
    if (b.gath.numel() < tri.router_ks * R * ne) return error.RouterScratch;
    try f.tri.router(normed.ptr, normed.stride[0], ix.gate.ptr, b.igr.ptr, b.igr.shape[1], b.gath.ptr, R, ix.gate.shape[1], ne);
    try f.tri.indexUpdate(b.ikr.ptr, b.ikr.shape[1], b.igr.ptr, .{ .ln_w = ix.ln_w.ptr, .ln_b = ix.ln_b.ptr, .ape = ix.ape.ptr }, .{ .ik = ic[0].ptr, .ig = ic[1].ptr, .pk = ic[2].ptr }, pos.ptr, R, f.m.index_dim);
}

/// select_tokens' blocks of SELECT_ROWS rows over the window's np pools (sparse.py 196-204): each block's scores and
/// top pools (python's own tensor a block, here its rows of tmp.pools) from its first row's position, then their cat.
pub fn selectBlocks(f: *Fwd, b: *const bufs.Buffers, pos: Buf, pk: u64, tmp: Temps, R: usize, np: usize) !void {
    const m = f.m;
    const heads = b.ikr.shape[1] - m.index_dim; // python: wts = ikr[:, index_dim:], H = wts.shape[1]
    const dim = b.qi.shape[1] / heads;
    const B = @min(R, tri.select_rows);
    var blocks: [max_blocks]T = undefined;
    var n: usize = 0;
    var a: usize = 0;
    while (a < R) : (a += B) {
        if (n == max_blocks) return error.DsaScratch;
        const at = if (a == 0) pos.ptr else try posAdd(f, pos, a, try atSlot(b, R, max_blocks + n));
        blocks[n] = tmp.pools;
        blocks[n].ptr += a * dk.topk_pools * 8;
        blocks[n].shape[0] = @min(B, R - a);
        try f.tri.selectTokens(b.qi.ptr + a * heads * dim * 2, b.ikr.ptr + (a * b.ikr.shape[1] + m.index_dim) * 2, b.ikr.shape[1], pk, tmp.scores, blocks[n].ptr, &.{at}, blocks[n].shape[0], np, heads, dim);
        n += 1;
    }
    if (tmp.cat) |out| try catPools(f, blocks[0..n], out);
}

/// The i-th block position's int32 in b.gath past a call's tokens (dense blocks from 1, select blocks from max_blocks):
/// python's `pos_dev + r0`, a new tensor a block. Slots are 16 bytes apart: Triton specializes a pointer on its 16-byte
/// divisibility, and torch's allocations always have it (the G-cover, zig/tests/cuda/glm_dsa_cover.zig, holds this).
fn atSlot(b: *const bufs.Buffers, R: usize, i: usize) !T {
    const off = std.mem.alignForward(usize, R * dk.width * 4, 256) + i * 16;
    if (off + 4 > b.gath.bytes()) return error.DsaScratch;
    var t = T.of(b.gath, off, &.{1});
    t.dtype = .i32;
    return t;
}

/// `pos_dev + n` (int32 [1] plus a python int: a torch add on the device) into `out`, glm_select's tf_glm_pos_add on a
/// GPU; gives out's address.
fn posAdd(f: *Fwd, pos: Buf, n: usize, out: T) !u64 {
    if (f.rec) |t| {
        try t.call(.ext, "torch.add", &.{ .{ .t = T.whole(pos) }, .{ .int = @intCast(n) }, .{ .t = out } });
        return out.ptr;
    }
    const g = f.gpu orelse return error.NoGpu;
    const k = g.kernels.dsa orelse return error.NoDsaKernels;
    try k.posAdd(g.stream, pos.ptr, @intCast(n), out.ptr);
    return out.ptr;
}

/// sparse.py 204, `torch.cat(blocks)`: the select blocks' pools into one tensor, a copy of each block on a GPU.
fn catPools(f: *Fwd, blocks: []const T, out: T) !void {
    if (f.rec) |t| {
        var vals: [max_blocks]trace.Val = undefined;
        for (blocks, 0..) |x, i| vals[i] = .{ .t = x };
        return t.call(.ext, "torch.cat", &.{ .{ .list = vals[0..blocks.len] }, .{ .t = out } });
    }
    const g = f.gpu orelse return error.NoGpu;
    const d = g.stream.d;
    var at: u64 = 0;
    for (blocks) |x| {
        const n = x.shape[0] * x.shape[1] * 8;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(out.ptr + at, x.ptr, n, g.stream.handle), "cuMemcpyDtoDAsync");
        at += n;
    }
}

/// select_tokens' torch ops after its kernels, one event (tools/zig/glm_dsa_fixtures.py checks python's ops and folds
/// them alike), or glm_select.cu's one launch.
pub fn selectTail(f: *Fwd, pools: T, pos: Buf, R: usize, tokens: T, counts: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.select_tail", &.{ .{ .t = pools }, .{ .t = T.whole(pos) }, .{ .int = @intCast(R) }, .{ .t = tokens }, .{ .t = counts } });
    const g = f.gpu orelse return error.NoGpu;
    const k = g.kernels.dsa orelse return error.NoDsaKernels;
    try k.selectTail(g.stream, pools.ptr, pos.ptr, tokens.ptr, counts.ptr, R);
}

fn rowsOf(x: Buf, R: usize) T {
    return layer.rowsOf(x, R);
}
