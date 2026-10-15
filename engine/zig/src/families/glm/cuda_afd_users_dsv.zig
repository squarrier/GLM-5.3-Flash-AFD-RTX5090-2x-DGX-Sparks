//! shared_verify (TF_AFD_USERS_SHARED_DRAFTS=1, default off): drafted shared verify on the AFD users path, shared_forward stage 2.
//! Every stream's verify window (its pending row, then up to three held or copied drafts) in ONE forward of at most
//! eight rows: each layer's row-by-row half once over every row, each window's per-sequence half (its KDA chain or its
//! DSA attention over its own rows) on its own State at its own position, one exchange of all rows a MoE layer. Then
//! acceptance, rollback and MTP drafting per stream, each on its own State, as the lane core's shared round asks
//! (verify, a draft request naming the kept rows, keep). Design: @Bizuayeu's shared verify windows
//! (github.com/Bizuayeu/TensorFold pr/glm53-flash-cuda 75131b66 "a round's verify windows share one forward, each
//! stream's state on its own", 146e113f, d3cc052c; Apache-2.0), reimplemented over this tree's q4/EXL3 kernels from
//! his design and shared_forward's split (cuda_afd_users_shared.zig); no code of his is copied (evidence/shared_verify/DESIGN.md).
//! Ash Hart authored TensorFold and its lane core; Mia (@MiaAI_lab) authored the EXL3 recipe and the MTP drafter; this
//! file: Claude Opus 5.5 (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const dsa = @import("cuda_dsa.zig");
const dk = @import("cuda_dsa_kernels.zig");
const tri = @import("cuda_triton.zig");
const cc = @import("cuda_compute.zig");
const ad = @import("cuda_afd_drafts.zig");
const sampling = @import("cuda_sample.zig");
const sharing = @import("cuda_afd_users_shared.zig");
const drafting = @import("cuda_afd_users_drafts.zig");
const scheduling = @import("cuda_afd_users_round.zig");
const Users = @import("cuda_afd_users.zig").Users;
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;
const Buf = views.Buf;
const Key = scheduling.Key;
const Row = scheduling.Row;

/// A drafted shared forward's rows at most: at <= 8 rows every kernel runs the variant today's decode windows run (the
/// q4 group GEMM's tile 7, the halves' decode scratch, the decode Buffers' and every slot State's window rows).
pub const max_rows: usize = 8;
/// A stream's window at most: its pending row and three drafts (the widest drafted == undrafted window measured).
pub const max_width: usize = 4;
pub const max_drafts: usize = max_width - 1;

// -- the switch ------------------------------------------------------------------------------------------------------

/// TF_AFD_USERS_SHARED_DRAFTS: unset or 0 off; 1 on; anything else refused.
pub fn parse(value: ?[]const u8) error{BadAfdUsersSharedDrafts}!bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadAfdUsersSharedDrafts;
}

/// The switch against the boot: on needs TF_AFD_USERS_SHARED=1 (`shared`) and TF_AFD_USERS_DRAFTS=1 (`drafts`, which
/// itself needs TF_AFD_USERS 2 or 4, `--drafts POLICY` and TF_AFD_MTP=1).
pub fn sharedDraftsFor(value: ?[]const u8, shared: bool, drafts: bool) error{ BadAfdUsersSharedDrafts, UsersSharedDraftsNeedsShared, UsersSharedDraftsNeedsDrafts }!bool {
    const on = try parse(value);
    if (on and !shared) return error.UsersSharedDraftsNeedsShared;
    if (on and !drafts) return error.UsersSharedDraftsNeedsDrafts;
    return on;
}

pub fn fromEnv() error{BadAfdUsersSharedDrafts}!bool {
    return parse(if (std.c.getenv("TF_AFD_USERS_SHARED_DRAFTS")) |v| std.mem.span(v) else null);
}

// -- a round ---------------------------------------------------------------------------------------------------------

/// A stream's verify window: its first row (the pending token, its key, its cache length, its draw: sampled and
/// masked rows are never drafted), then its drafts, each drawn greedy at row.position + 1 + j.
pub const Window = struct {
    row: Row,
    drafts: []const u32 = &.{},

    pub fn rows(w: Window) usize {
        return 1 + w.drafts.len;
    }
};

/// The rows of a round's windows; refused past the cap, a window past its width, sampled or masked drafted rows.
pub fn total(wins: []const Window) !usize {
    var n: usize = 0;
    for (wins) |w| {
        if (w.rows() > max_width) return error.WindowRows;
        if (w.drafts.len > 0 and (w.row.mask.len != 0 or w.row.sampling != null)) return error.UsersDraftsUnsupported;
        n += w.rows();
    }
    if (wins.len == 0 or wins.len > 4 or n > max_rows) return error.Rows;
    return n;
}

/// The device order of a drafted shared round (Users' Dev below; the tests' stand-ins): one forward over every window's
/// rows, synchronised and checked before any draw; every row of a live window drawn (a cancelled window draws
/// nothing); a one-row window's row committed now (concurrent_users's), a wider window's rows left for the stream's draft request
/// or keep (`Dev.settle`); a failure anywhere commits nothing after it.
pub fn run(comptime Device: type, d: *Device, wins: []const Window, out: []u32) !void {
    const R = try total(wins);
    if (out.len != R) return error.Rows;
    @memset(out, 0);
    try d.alive();
    try d.dsvBegin(wins, R);
    try d.dsvOpen(wins, R);
    for (0..d.layerCount()) |li| {
        try d.alive();
        try d.dsvIn(wins, R, li);
        var r0: usize = 0;
        for (wins) |w| {
            try d.alive();
            try d.dsvSeq(w, r0, li);
            r0 += w.rows();
        }
        try d.alive();
        try d.dsvOut(R, li);
    }
    try d.alive();
    try d.dsvTail(wins, R);
    try d.settle(); // no logits are read and no State committed until every launch synchronised and checked
    var r0: usize = 0;
    for (wins) |w| {
        defer r0 += w.rows();
        if (w.row.cancelled()) continue;
        try d.alive();
        try d.dsvDraw(w, r0, out[r0 .. r0 + w.rows()]);
    }
    try d.alive();
    r0 = 0;
    for (wins) |w| {
        defer r0 += w.rows();
        if (w.row.cancelled() or w.rows() != 1) continue;
        try d.alive();
        try d.dsvCommit(w.row, 1, 1);
    }
    try d.settle(); // queued position/conv writes finish before scratch reuse or retirement
}

// -- the per-sequence half over a window's rows ----------------------------------------------------------------------

/// A window's per-sequence call's Buffers: `b` at R = w with the rows it reads or writes moved to row `r0` (shared_forward's
/// rowView at w rows: partRows of the view is w, so a window of w <= 8 rows is one dense block at its position, as the
/// single-user decode window's on the 8-row Buffers); every scratch at its base.
pub fn segView(b: bufs.Buffers, r0: usize, w: usize) bufs.Buffers {
    var v = b;
    v.rows = w;
    v.lat = rowsOfBuf(b.lat, r0, w);
    v.ikr = rowsOfBuf(b.ikr, r0, w);
    v.igr = rowsOfBuf(b.igr, r0, w);
    v.qi = rowsOfBuf(b.qi, r0, w);
    v.ka = rowsOfBuf(b.ka, r0, w);
    v.kg = rowsOfBuf(b.kg, r0, w);
    v.lat_s.qa = rowsOfBuf(b.lat_s.qa, r0, w);
    v.lat_s.ol = rowsOfBuf(b.lat_s.ol, r0, w);
    return v;
}

/// Rows [r0, r0 + n) of a row-major buffer as a buffer of n rows.
pub fn rowsOfBuf(x: Buf, r0: usize, n: usize) Buf {
    var y = x;
    y.ptr += r0 * x.inner() * x.dtype.size();
    y.shape[0] = n;
    return y;
}

/// kdaBlock's chain for a window of `w` rows at row `r0` on its user's State: its projected rows into its own st.proj
/// rows [0, w) (the commit's conv shift and replay read them there), the chain at R = w from its conv window and
/// recurrent state (the saves for w rows on its State), the outputs into the stage's rows [r0, r0 + w).
pub fn kdaSeqW(f: *Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, s: sharing.Stages, r0: usize, w: usize) !void {
    const m = f.m;
    const k = l.kda orelse return error.NotKda;
    const li = m.kindIndex(l.index);
    const width = k.proj.n();
    const rows = st.proj.shape[1];
    if (w == 0 or w > rows or r0 + w > s.proj.shape[0]) return error.Rows;
    const p = T.rows2(st.proj, li * rows * width * 2, w, width, width);
    try cc.copyDtoD(f, p, T.rows2(s.proj, r0 * width * 2, w, width, width));
    const cur: usize = st.parity;
    const ll = k.heads;
    const n = st.rec.shape[1];
    const state = ll * 128 * 128 * 4;
    const ss = st.scratch_set;
    const ch = ll * 128;
    try f.kdaChain(.{
        .p = p,
        .b_off = k.bOff(),
        .a = T.rows2(b.ka, r0 * ch * 2, w, ch, ch),
        .g = T.rows2(b.kg, r0 * ch * 2, w, ch, ch),
        .conv_state = T.of(st.conv, li * st.conv.inner() * 2, st.conv.shape[1..3]),
        .conv_w = T.whole(k.conv),
        .state_in = T.of(st.rec, (cur * n + li) * state, &.{ ll, 128, 128 }),
        .a_log = T.whole(k.a_log),
        .dt_bias = T.whole(k.dt_bias),
        .norm_w = T.whole(k.norm),
        .eps = m.eps,
        .lower = m.lower,
        .rows = w,
        .out = T.of(s.out, r0 * ch * 2, &.{ w, ch }),
        .state_out = T.of(st.rec, ((1 - cur) * n + li) * state, &.{ ll, 128, 128 }),
        .k_save = T.of(ss.k, li * rows * ch * 4, &.{ rows, ll, 128 }),
        .v_save = T.of(ss.v, li * rows * ch * 2, &.{ rows, ll, 128 }),
        .g_save = T.of(ss.g, li * rows * ch * 4, &.{ rows, ll, 128 }),
        .b_save = T.of(ss.b, li * rows * ll * 4, &.{ rows, ll }),
    });
}

/// What dsaIn needs from a window view `v` (segView) on `st`: its index caches, and whether its attention runs sparse.
pub fn dsaNeedW(m: anytype, l: *const views.Layer, st: *const bufs.State, v: *const bufs.Buffers, opt: layer.Opt) !sharing.Need {
    const c = try dsa.caches(m, l, st);
    const sh = try dsa.shape(m, l, st, v, v.rows, opt);
    return .{ .indexed = c.index != null, .sparse = sh.sparse };
}

/// partialWith's per-sequence launches for a window view `v` (R = v.rows, its rows moved to the window's first) on its
/// user's State at its own position: the latent write of its rows, the indexer update, the dense block (part_rows of
/// the view is R: one block at the State's position, as the single-user window's) or the select blocks, tail and sparse
/// attention, into v's lat_s.ol rows.
pub fn dsaSeqW(f: *Fwd, l: *const views.Layer, st: *const bufs.State, v: *const bufs.Buffers, opt: layer.Opt, x: ?*const dk.Scratch) !void {
    const m = f.m;
    const a = l.dsa orelse return error.NotDsa;
    const R: usize = v.rows;
    if (R == 0 or R > max_rows) return error.Rows;
    const c = try dsa.caches(m, l, st);
    const HL = a.heads;
    const qk = m.qk_dim;
    const ab = dsa.absorbOf(a.absorb orelse return error.NotPorted);
    try f.tri.latentWrite(v.lat.ptr, v.lat.shape[1], c.lc.ptr, c.pos.ptr, R, c.lc.shape[1]);
    const sh = try dsa.shape(m, l, st, v, R, opt);
    if (c.index) |ic| { // indexUpdate's last launch on the window's keys and gates
        const ix = a.index orelse return error.NoIndexer;
        try f.tri.indexUpdate(v.ikr.ptr, v.ikr.shape[1], v.igr.ptr, .{ .ln_w = ix.ln_w.ptr, .ln_b = ix.ln_b.ptr, .ape = ix.ape.ptr }, .{ .ik = ic[0].ptr, .ig = ic[1].ptr, .pk = ic[2].ptr }, c.pos.ptr, R, m.index_dim);
    }
    const qa = v.lat_s.qa;
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(qk))));
    const ol = v.lat_s.ol;
    const s = v.lat_s;
    if (sh.dense) // dense_attention: one block of the window's R rows at its first row's position (part_rows = R)
        try f.tri.latentAttention(qa.ptr, c.lc.ptr, c.pos.ptr, .{ .po = s.po.ptr, .pm = s.pm.ptr, .pl = s.pl.ptr, .aux = s.dummy.ptr }, ol.ptr, R, HL, ab.lw, sh.nch, tri.headBlock(R), scale);
    if (sh.sparse) {
        const ic = c.index.?;
        const tmp = try dsa.temps(m, v, R, sh.np, x);
        try dsa.selectBlocks(f, v, c.pos, ic[2].ptr, tmp, R, sh.np);
        try dsa.selectTail(f, tmp.cat orelse tmp.pools, c.pos, R, tmp.tokens, tmp.counts);
        try f.tri.latentSparse(qa.ptr, c.lc.ptr, tmp.tokens.ptr, tmp.counts.ptr, ol.ptr, tmp.part, R, HL, ab.lw, dk.width, scale);
    }
}

// -- the users' device half ------------------------------------------------------------------------------------------

/// A slot's drafted-shared state on the device side: the backlog index its last window's final-normed rows were
/// copied to (`base`, `copied` rows, counted only at its draft request), the rows of its last window still to settle
/// (`pending`: a wider window's, until its settle), and `in_head`: the backlog's row 0 is the
/// stream's start pair (its last kept row, its pending token), ALREADY absorbed into its MTP cache with
/// `mtp_drafted` chained drafts after it (a lone resume un-absorbs it first: `unabsorb`).
pub const Slot = struct { base: u16 = 0, copied: u8 = 0, pending: u8 = 0, in_head: bool = false };

/// Counters for the lab and the tests.
pub const Counts = struct { rounds: u64 = 0, rows: u64 = 0, windows: u64 = 0, drafted_rows: u64 = 0, settles: u64 = 0, replays: u64 = 0, draft_requests: u64 = 0, absorbed: u64 = 0, drafts: u64 = 0, unabsorbed: u64 = 0 };

pub const Slots = struct {
    slot: [4]Slot = @splat(.{}),
    counts: Counts = .{},
};

fn stateOf(u: *Users, key: Key) !*bufs.State {
    if (key.slot >= u.count or u.generation[key.slot] != key.generation) return error.StaleGeneration;
    return if (key.slot == 0) &u.at.e.st else &u.states[key.slot - 1];
}

fn optW(u: *Users, st: *const bufs.State, w: usize) layer.Opt {
    return .{ .nch = cc.chunksFor(st, w), .host_pos = st.pos, .chip = u.at.e.chip };
}

/// The backlog a window's rows join: a tracked slot's (its pairs end at its pending token), else none.
fn backlogOf(u: *Users, slot: u8) ?*drafting.Backlog {
    const d = if (u.drafts) |*x| x else return null;
    const b = &d.backlog[slot];
    if (b.track == .off or b.head) return null;
    return b;
}

/// The Users as `run`'s Device: shared_forward's device half (cuda_afd_users.zig sharedOpen..sharedTail) over windows.
pub const Dev = struct {
    u: *Users,

    pub fn alive(d: *Dev) !void {
        try d.u.at.alive();
    }

    pub fn layerCount(d: *const Dev) usize {
        return d.u.at.e.w.layers.len;
    }

    pub fn settle(d: *Dev) !void {
        try d.u.settle();
    }

    /// Users.begin's for R rows of `wins.len` streams: the last forward's check, the Afd's provider, one exchange
    /// header of R rows a MoE layer (eager: the decode path's route).
    pub fn dsvBegin(d: *Dev, wins: []const Window, R: usize) !void {
        const u = d.u;
        u.together = false;
        try u.at.finish(u.f);
        u.f.moe = u.at.a.provider();
        try u.at.a.begin(R, .eager, if (u.f.gpu) |g| g.stream else null);
        u.at.forwards += 1;
        std.log.info("concurrent_users decode rows={d} users={d} round={d}", .{ R, wins.len, u.rounds });
        var r0: usize = 0;
        for (wins, 0..) |w, i| {
            const st = try stateOf(u, w.row.key);
            std.log.info("shared_verify window={d} slot={d} generation={d} position={d} parity={d} row0={d} rows={d}", .{ i, w.row.key.slot, w.row.key.generation, w.row.position, st.parity, r0, w.rows() });
            r0 += w.rows();
        }
    }

    /// sharedOpen's for R rows: every window's ids (pending, drafts) staged at once and embedded into the Buffers' rows.
    pub fn dsvOpen(d: *Dev, wins: []const Window, R: usize) !void {
        const u = d.u;
        const e = u.at.e;
        const b = &e.buf;
        if (R == 0 or R > max_rows or R > b.rows) return error.Rows;
        var ids: [max_rows]u32 = undefined;
        var k: usize = 0;
        for (wins) |w| {
            ids[k] = w.row.token;
            @memcpy(ids[k + 1 .. k + w.rows()], w.drafts);
            k += w.rows();
        }
        u.together = true; // the draws read b.logits' rows (Users.logitsOf)
        try cc.stageIds(u.f, b, ids[0..R]);
        try cc.stagedRecord(u.f, b);
        try u.f.tri.embed(b.ids.ptr, cc.table(e.w.embed), b.x.ptr, R, e.m.hidden, e.m.streams);
    }

    /// sharedIn's for R rows: layer `li`'s attention hc_pre over every row, then the block's row-by-row half (the KDA
    /// projections, or the DSA projections, norms, absorb and the indexer projections some window needs).
    pub fn dsvIn(d: *Dev, wins: []const Window, R: usize, li: usize) !void {
        const u = d.u;
        const e = u.at.e;
        const m = e.m;
        const b = &e.buf;
        const l = &e.w.layers[li];
        const h = l.attn_hc orelse return error.PlainLayer;
        try u.f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.in_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
        switch (l.kind) {
            .kda => try sharing.kdaIn(u.f, l, b, u.stages.?, R, e.chip),
            .dsa => {
                var need: ?sharing.Need = null;
                var sparse = false;
                var r0: usize = 0;
                for (wins) |w| {
                    const st = try stateOf(u, w.row.key);
                    const v = segView(b.*, r0, w.rows());
                    const x = try dsaNeedW(m, l, st, &v, optW(u, st, w.rows()));
                    if (need) |first| if (first.indexed != x.indexed) return error.UsersMixedIndex;
                    need = x;
                    sparse = sparse or x.sparse;
                    r0 += w.rows();
                }
                try sharing.dsaIn(u.f, l, b, R, need.?.indexed, sparse, e.chip);
            },
        }
    }

    /// Window `w`'s per-sequence half of layer `li` (rows [r0, r0 + w.rows())) on its own State at its own position.
    pub fn dsvSeq(d: *Dev, w: Window, r0: usize, li: usize) !void {
        const u = d.u;
        const e = u.at.e;
        const l = &e.w.layers[li];
        const st = try stateOf(u, w.row.key);
        switch (l.kind) {
            .kda => try kdaSeqW(u.f, l, st, &e.buf, u.stages.?, r0, w.rows()),
            .dsa => {
                const v = segView(e.buf, r0, w.rows());
                try dsaSeqW(u.f, l, st, &v, optW(u, st, w.rows()), dsa.scratchOf(u.f));
            },
        }
    }

    /// sharedOut's for R rows: the block out-projection and hc_post over every row, then the FFN half over every row
    /// (the dense MLP, or the MoE's hc_pre, ONE route + request + take of all R rows to both halves, hc_post).
    pub fn dsvOut(d: *Dev, R: usize, li: usize) !void {
        const u = d.u;
        const e = u.at.e;
        const m = e.m;
        const b = &e.buf;
        const l = &e.w.layers[li];
        const g = switch (l.kind) {
            .kda => try sharing.kdaOut(u.f, l, b, u.stages.?, R, e.chip),
            .dsa => try sharing.dsaOut(u.f, l, b, R, e.chip),
        };
        try u.f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
        try layer.ffnHalf(u.f, l, b, R, .{ .chip = e.chip }); // the MoE's block: f.moe, the Afd's provider (dsvBegin)
        if (u.isMoe(li)) {
            u.exchanges += 1;
            std.log.info("concurrent_users exchange round={d} layer={d} rows={d}", .{ u.rounds, li, R });
        }
    }

    /// sharedTail's for R rows (the streams' mean, the final norm, the head into the decode logits' rows), then each
    /// tracked slot's window rows (final-normed) copied to its backlog's end on the forward's stream, counted only at
    /// its draft request (user_drafts's grow, a window's rows instead of one).
    pub fn dsvTail(d: *Dev, wins: []const Window, R: usize) !void {
        const u = d.u;
        const e = u.at.e;
        const b = &e.buf;
        const D = e.m.hidden;
        try u.f.tri.streamMean(b.x.ptr, b.hidden.ptr, R, D, e.m.streams);
        try u.f.tri.rmsnorm(b.hidden.ptr, D, e.w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, e.m.eps, R, D);
        try qmm.mm(u.f, false, cc.rows(b.fnormed, 0, R), e.w.head, cc.rows(b.fxs, 0, R), cc.rows(b.logits, 0, R), false, b.sk, e.chip);
        const v = &u.dsvs.?;
        var r0: usize = 0;
        for (wins) |w| {
            defer r0 += w.rows();
            const sl = &v.slot[w.row.key.slot];
            sl.copied = 0;
            sl.pending = if (w.row.cancelled() or w.rows() == 1) 0 else @intCast(w.rows()); // settled by settle
            if (w.row.cancelled()) continue;
            const k = backlogOf(u, w.row.key.slot) orelse continue;
            if (k.n + w.rows() > k.capacity()) return error.BacklogFull;
            try cc.copyDtoD(u.f, cc.rows(k.hidden, k.n, w.rows()), cc.rows(b.fnormed, r0, w.rows()));
            sl.base = @intCast(k.n);
            sl.copied = @intCast(w.rows());
        }
        const c = &v.counts;
        c.rounds += 1;
        c.rows += R;
        c.windows += wins.len;
        for (wins) |w| c.drafted_rows += w.drafts.len;
        u.rounds += 1;
    }

    /// Window `w`'s draws (rows [r0, r0 + rows)): a sampled or masked one-row window as the users round draws it
    /// (Users.sample: user_sampling's keyed draw, structured_users's mask); every other row the greedy draw of its own logits row.
    pub fn dsvDraw(d: *Dev, w: Window, r0: usize, out: []u32) !void {
        const u = d.u;
        if (w.row.sampling != null or w.row.mask.len != 0) {
            if (w.drafts.len != 0) return error.UsersDraftsUnsupported;
            out[0] = try u.sample(w.row, r0);
            return;
        }
        for (out, 0..) |*o, j| {
            var token: [1]u32 = .{0};
            try sampling.greedy(u.f, &u.at.e.smp, cc.rows(u.at.e.buf.logits, r0 + j, 1), &token);
            o.* = token[0];
        }
    }

    /// forward.commit of `keep` of a window's `rows` on the row's State (the KDA replay of the kept rows from its own
    /// saves when keep < rows; the parity flip; the conv shift from its st.proj rows; pos += keep).
    pub fn dsvCommit(d: *Dev, row: Row, rows: usize, keep: usize) !void {
        const u = d.u;
        const st = try stateOf(u, row.key);
        u.observeRound(.commit_before, row, 0);
        try cc.commit(u.f, st, &u.at.e.buf, rows, keep);
        u.observeRound(.commit_after, row, 1);
        u.dsvs.?.counts.settles += 1;
        if (keep < rows) u.dsvs.?.counts.replays += 1;
    }
};

/// Users.roundDrafted's body: the round (run) on the bound context; any failure latches the attention side dead. A
/// window whose stream's last window is not settled yet is refused before any launch.
pub fn round(u: *Users, wins: []const Window, out: []u32) !void {
    const v = if (u.dsvs) |*x| x else return error.UsersSharedDraftsOff;
    if (u.stages == null or u.drafts == null) return error.UsersSharedDraftsOff;
    for (wins) |w| {
        const st = try stateOf(u, w.row.key);
        if (st.pos != w.row.position) return error.PositionMismatch;
        if (v.slot[w.row.key.slot].pending != 0) return error.Unsettled;
        try cc.checkRoom(u.at.e.m, st, w.rows(), null);
    }
    errdefer u.at.a.dead = true; // an interrupted header/exchange window is not safe to reuse
    var dev: Dev = .{ .u = u };
    try run(Dev, &dev, wins, out);
}

/// Users.settleDrafted's body: `keep` of the last window's `rows` stay (a drafted window's rows wait for this); named
/// once, with that window's rows.
pub fn settle(u: *Users, key: Key, rows: usize, keep: usize) !void {
    if (rows < 2 or rows > max_width or keep < 1 or keep > rows) return error.KeepOutOfStep;
    const v = if (u.dsvs) |*x| x else return error.UsersSharedDraftsOff;
    const st = try stateOf(u, key);
    if (v.slot[key.slot].pending != rows) return error.KeepOutOfStep;
    errdefer u.at.a.dead = true;
    var dev: Dev = .{ .u = u };
    try dev.dsvCommit(.{ .key = key, .token = 0, .position = st.pos }, rows, keep);
    v.slot[key.slot].pending = 0;
    try u.settle();
}

/// Users.draftDrafted's body, on the slot's State (swapped into the Engine for it, as user_drafts's drafted loop is):
/// the pairs its MTP cache has not absorbed (the backlog's, then with `verified` the last window's first
/// `follow.len` rows with `follow` as their next tokens), absorbed (the chained drafts of the last request trimmed
/// first), then `out.len` drafts chained at `position` (no floors: exactly out.len). The last pair stays as row 0 of
/// the backlog, in head (absorbed), for the next request or a lone resume. `follow` without `verified`: the prompt's
/// first token, which the backlog's seed pair must end at.
pub fn draft(u: *Users, key: Key, follow: []const u32, verified: bool, position: usize, out: []u32) !void {
    const v = if (u.dsvs) |*x| x else return error.UsersSharedDraftsOff;
    const d = if (u.drafts) |*x| x else return error.UsersDraftsOff;
    if (key.slot >= u.count or u.generation[key.slot] != key.generation) return error.StaleGeneration;
    if (follow.len == 0 or out.len > max_drafts) return error.DraftOutOfStep;
    const b = &d.backlog[key.slot];
    const sl = &v.slot[key.slot];
    if (b.track == .off or b.head) return error.NotTracked;
    if (sl.in_head and b.n != 1) return error.BacklogOutOfStep;
    const start: usize = @intFromBool(sl.in_head);
    if (verified) {
        if (sl.copied == 0 or follow.len > sl.copied or sl.base != b.n) return error.DraftOutOfStep;
        @memcpy(b.tokens[b.n .. b.n + follow.len], follow);
        b.n += follow.len;
    } else {
        if (sl.in_head or b.n == 0 or follow.len != 1 or b.tokens[b.n - 1] != follow[0]) return error.DraftOutOfStep;
    }
    sl.copied = 0;
    const st = try stateOf(u, key);
    if (st.pos + 1 != position) return error.PositionMismatch;
    const n = b.n - start;
    if (n == 0) return error.DraftOutOfStep;
    const e = u.at.e;
    if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
    defer if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
    errdefer {
        u.at.a.dead = true;
        b.clear();
        sl.* = .{};
    }
    const hidden = cc.rows(b.hidden, start, n);
    var steps: usize = 0;
    if (out.len == 0) {
        steps = try ad.absorbRows(u.at, u.f, hidden, b.tokens[start..b.n]);
    } else {
        const got = try ad.draft(u.at, u.f, hidden, b.tokens[start..b.n], position, out.len, .{}, out, null, &steps);
        if (got.len != out.len) return error.DraftRows;
    }
    if (b.n > 1) { // the last pair to the front: the stream's start pair, absorbed
        try cc.copyDtoD(u.f, cc.rows(b.hidden, 0, 1), cc.rows(b.hidden, b.n - 1, 1));
        b.tokens[0] = b.tokens[b.n - 1];
        b.n = 1;
    }
    sl.in_head = true;
    v.counts.draft_requests += 1;
    v.counts.absorbed += n;
    v.counts.drafts += out.len;
}

/// Inside user_drafts's drafted loop (Users.drafted, the slot's State in the Engine), before its start: a slot whose start
/// pair is in head (absorbed by a drafted shared round, its drafts chained after it) un-absorbs it, so the loop starts
/// from exactly the single-user loop's start pair (the pair and the drafts trimmed from the MTP cache).
pub fn unabsorb(u: *Users, slot: u8) !void {
    const v = if (u.dsvs) |*x| x else return;
    const sl = &v.slot[slot];
    if (!sl.in_head) return;
    const st = &u.at.e.st;
    const k = st.mtp_drafted + 1;
    if (st.mtp_len < k) return error.MtpOutOfStep;
    try cc.setMtpLen(u.f, st, st.mtp_len - k);
    st.mtp_drafted = 0;
    sl.in_head = false;
    v.counts.unabsorbed += 1;
}

/// The stages at the cap's rows (shared_forward's two, 8 rows instead of TF_AFD_USERS's) for the forecast.
pub fn stageBytes(m: anytype, on: bool) u64 {
    if (!on) return 0;
    const s = sharing.Stages.init(m, max_rows);
    return std.mem.alignForward(u64, s.proj.bytes(), 256) + std.mem.alignForward(u64, s.out.bytes(), 256);
}

test "shared_verify switch: unset and 0 off, 1 on; on needs TF_AFD_USERS_SHARED=1 and TF_AFD_USERS_DRAFTS=1; anything else refused" {
    const t = std.testing;
    try t.expectEqual(false, try parse(null));
    try t.expectEqual(false, try parse("0"));
    try t.expectEqual(true, try parse("1"));
    for ([_][]const u8{ "", "2", "00", "01", "true", "on", " 1", "1 ", "-1", "yes", "1:1", "shared" }) |x| try t.expectError(error.BadAfdUsersSharedDrafts, parse(x));
    try t.expect(try sharedDraftsFor("1", true, true));
    try t.expect(!try sharedDraftsFor(null, false, false));
    try t.expect(!try sharedDraftsFor("0", false, true));
    try t.expectError(error.UsersSharedDraftsNeedsShared, sharedDraftsFor("1", false, true));
    try t.expectError(error.UsersSharedDraftsNeedsDrafts, sharedDraftsFor("1", true, false));
    try t.expectError(error.BadAfdUsersSharedDrafts, sharedDraftsFor("2", true, true));
}

test "shared_verify window rows: the cap, the width, undrafted sampled and masked rows" {
    const t = std.testing;
    const k: Key = .{ .slot = 0, .generation = 1 };
    const d3 = [_]u32{ 1, 2, 3 };
    const four = [_]Window{ .{ .row = .{ .key = k, .token = 1, .position = 0 }, .drafts = &d3 }, .{ .row = .{ .key = k, .token = 1, .position = 0 }, .drafts = &d3 } };
    try t.expectEqual(@as(usize, 8), try total(&four));
    const nine = [_]Window{ four[0], four[1], .{ .row = .{ .key = k, .token = 1, .position = 0 } } };
    try t.expectError(error.Rows, total(&nine));
    const d4 = [_]u32{ 1, 2, 3, 4 };
    try t.expectError(error.WindowRows, total(&.{.{ .row = .{ .key = k, .token = 1, .position = 0 }, .drafts = &d4 }}));
    const mask = [_]u32{1};
    try t.expectError(error.UsersDraftsUnsupported, total(&.{.{ .row = .{ .key = k, .token = 1, .position = 0, .mask = &mask }, .drafts = d3[0..1] }}));
    try t.expectError(error.UsersDraftsUnsupported, total(&.{.{ .row = .{ .key = k, .token = 1, .position = 0, .sampling = .{ .seed = 1, .temperature = 0.7 } }, .drafts = d3[0..1] }}));
    try t.expectEqual(@as(usize, 1), try total(&.{.{ .row = .{ .key = k, .token = 1, .position = 0, .mask = &mask } }}));
    try t.expectError(error.Rows, total(&.{}));
}

test "shared_verify window view: rows moved to the window's first, w rows, scratch at its base; w = 1 is shared_forward's row view" {
    const m = try @import("cuda_model.zig").Model.from(&@import("config.zig").Config{}, .{}, 1, 0);
    var b = try bufs.Buffers.init(&m, 8, 2560, false);
    b.lat.ptr = 0x100000;
    b.lat_s.qa.ptr = 0x200000;
    b.lat_s.po.ptr = 0x300000;
    b.ka.ptr = 0x400000;
    b.kg.ptr = 0x500000;
    const v = segView(b, 3, 4);
    try std.testing.expectEqual(@as(usize, 4), v.rows);
    try std.testing.expectEqual(b.lat.ptr + 3 * m.kv_lora * 2, v.lat.ptr);
    try std.testing.expectEqual(@as(usize, 4), v.lat.shape[0]);
    try std.testing.expectEqual(b.lat_s.qa.ptr + 3 * m.hl() * m.kv_lora * 2, v.lat_s.qa.ptr);
    try std.testing.expectEqual(b.ka.ptr + 3 * m.ll() * 128 * 2, v.ka.ptr);
    try std.testing.expectEqual(b.kg.ptr + 3 * m.ll() * 128 * 2, v.kg.ptr);
    try std.testing.expectEqual(b.lat_s.po.ptr, v.lat_s.po.ptr);
    const one = segView(b, 5, 1);
    const ms = sharing.rowView(b, 5);
    inline for (.{ "lat", "ikr", "igr", "qi", "ka", "kg" }) |name| {
        try std.testing.expectEqual(@field(ms, name).ptr, @field(one, name).ptr);
        try std.testing.expectEqualSlices(usize, @field(ms, name).shape[0..@field(ms, name).dims], @field(one, name).shape[0..@field(one, name).dims]);
    }
    try std.testing.expectEqual(ms.lat_s.qa.ptr, one.lat_s.qa.ptr);
    try std.testing.expectEqual(ms.lat_s.ol.ptr, one.lat_s.ol.ptr);
    try std.testing.expectEqual(ms.rows, one.rows);
    const s = sharing.Stages.init(&m, max_rows);
    try std.testing.expectEqual(stageBytes(&m, true), std.mem.alignForward(u64, s.proj.bytes(), 256) + std.mem.alignForward(u64, s.out.bytes(), 256));
    try std.testing.expectEqual(@as(u64, 0), stageBytes(&m, false));
    std.debug.print("shared_verify window view: rows 3..6 of lat/qa/ka/kg moved, 4 rows, scratch at its base; w = 1 = shared_forward's row view; stages at 8 rows {d} B\n", .{stageBytes(&m, true)});
}
