//! V3 batched MTP port on native TensorFold state/cache/quantized kernel interfaces.
//! TensorFold/MTP: Ash Hart; weights/drafter: Mia's AI Lab. V3 extension: GPT-6.1 Sol via Hermes.
//! Helios (GPT-6.1 Sol/OpenAI Codex via Hermes) authored these minimal current-native adapters.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");
const dsa = @import("cuda_dsa_block.zig");
const latent = @import("cuda_latent.zig");
const q4 = @import("cuda_q4.zig");
pub const Request = struct { slot: u32, follow: []const u32, from_prompt: bool, depth: usize, out: []u32, chances: ?[]f64 = null };
pub const Testing = if (@import("builtin").is_test) struct {
    pub var forward: ?*const fn (*engine.Engine, []const u32, []const fw.Seg) anyerror!void = null;
    pub var draw: ?*const fn (*engine.Engine, u32, usize, usize, *u32, *f64) anyerror!void = null;
} else struct {};
const Work = struct { q: Request, stash_index: usize, cursor: usize = 0, tokens: [engine.max_drafts]u32 = undefined, probs: [engine.max_drafts]f64 = undefined };
fn wave(e: *engine.Engine, ids: []const u32, segs: []const fw.Seg) !void {
    for (segs) |g| try g.st.readyMtp(e.stream, g.rows);
    var recorded = false;
    if (comptime @import("builtin").is_test) if (Testing.forward) |f| {
        try f(e, ids, segs);
        recorded = true;
    };
    if (!recorded) try kernel(e, ids, segs);
    for (segs) |g| try g.st.setMtpLen(e.stream, g.st.mtp_len + g.rows);
}
fn draw(e: *engine.Engine, w: *Work, index: usize, row: usize) !void {
    try e.bind(w.q.slot);
    var token: u32 = undefined;
    var probability: f64 = 1;
    var recorded = false;
    if (comptime @import("builtin").is_test) if (Testing.draw) |f| {
        try f(e, w.q.slot, row, index, &token, &probability);
        recorded = true;
    };
    if (!recorded) {
        const logits = e.bm.logits + row * e.share.vocab.len() * 2;
        if (w.q.chances != null and !e.structured and e.cur.sampling == null) {
            const p = try e.greedyChance(logits);
            token = p.token;
            probability = p.chance;
        } else try e.draw(logits, 1, e.cur.st.pos + 1 + index, 0, @as(*[1]u32, @ptrCast(&token)));
    }
    w.tokens[index] = token;
    w.probs[index] = probability;
    // Preserve each user's last hidden row before the next wave reuses the MTP workspace.
    const stash = e.bsp.?.hin + w.stash_index * e.c.hidden * 2;
    try e.copy(stash, e.bm.fnormed + row * e.c.hidden * 2, e.c.hidden * 2);
}
/// Absorb waves, then one native shared MTP pass per depth. Target positions never change here.
pub fn run(e: *engine.Engine, requests: []const Request) !void {
    if (e.routed) |p| try p.check(p.ctx);
    if (!e.mtp or e.bsp == null) return error.NoDraftHead;
    if (requests.len == 0 or requests.len > engine.max_shared) return error.TooManyStreams;
    var work: [engine.max_shared]Work = undefined;
    var depth: usize = 0;
    for (requests, 0..) |q, i| {
        if (q.slot >= e.seqs.items.len or e.seqs.items[q.slot] == null) return error.UnknownSlot;
        const st = e.seqs.items[q.slot].?;
        if (q.follow.len == 0 or q.follow.len > fw.window_rows or q.depth > engine.max_drafts or q.out.len < q.depth or st.in_head > st.st.mtp_len) return error.BadWindow;
        if (q.from_prompt and q.follow.len != 1) return error.BadWindow;
        if (q.chances) |c| if (c.len < q.depth) return error.BadWindow;
        for (requests[0..i]) |prior| if (prior.slot == q.slot) return error.SlotTwice;
        if (st.st.mtp_len - st.in_head + q.follow.len + (q.depth -| 1) > st.st.capacity) return error.PastCapacity;
        work[i] = .{ .q = q, .stash_index = i };
        depth = @max(depth, q.depth);
    }
    for (requests) |q| {
        try e.bind(q.slot);
        try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len - e.cur.in_head);
        e.cur.in_head = 0;
    }
    const ws = work[0..requests.len];
    var ids: [engine.shared_rows]u32 = undefined;
    var segs: [engine.max_shared]fw.Seg = undefined;
    var owner: [engine.max_shared]usize = undefined;
    while (true) {
        var R: usize = 0;
        var N: usize = 0;
        for (ws, 0..) |*w, i| {
            if (w.cursor == w.q.follow.len or R == ids.len) continue;
            try e.bind(w.q.slot);
            const n = @min(w.q.follow.len - w.cursor, ids.len - R);
            segs[N] = .{ .st = &e.cur.st, .row0 = R, .rows = n };
            owner[N] = i;
            N += 1;
            @memcpy(ids[R..][0..n], w.q.follow[w.cursor..][0..n]);
            const hidden = if (w.q.from_prompt) e.cur.st.prompt_hidden else e.cur.rows_at + w.cursor * e.c.hidden * 2;
            try e.copy(e.bm.hin + R * e.c.hidden * 2, hidden, n * e.c.hidden * 2);
            w.cursor += n;
            R += n;
        }
        if (R == 0) break;
        try wave(e, ids[0..R], segs[0..N]);
        for (segs[0..N], owner[0..N]) |g, i| if (ws[i].cursor == ws[i].q.follow.len and ws[i].q.depth > 0) try draw(e, &ws[i], 0, g.row0 + g.rows - 1);
    }
    var j: usize = 1;
    while (j < depth) : (j += 1) {
        var R: usize = 0;
        for (ws, 0..) |*w, i| {
            if (w.q.depth <= j) continue;
            try e.bind(w.q.slot);
            ids[R] = w.tokens[j - 1];
            segs[R] = .{ .st = &e.cur.st, .row0 = R, .rows = 1 };
            owner[R] = i;
            try e.copy(e.bm.hin + R * e.c.hidden * 2, e.bsp.?.hin + w.stash_index * e.c.hidden * 2, e.c.hidden * 2);
            R += 1;
        }
        try wave(e, ids[0..R], segs[0..R]);
        for (owner[0..R], 0..) |i, row| {
            try e.bind(ws[i].q.slot);
            e.cur.in_head += 1;
            try draw(e, &ws[i], j, row);
        }
    }
    try e.stream.synchronize();
    try e.checkGathers();
    for (ws) |w| {
        @memcpy(w.q.out[0..w.q.depth], w.tokens[0..w.q.depth]);
        if (w.q.chances) |p| @memcpy(p[0..w.q.depth], w.probs[0..w.q.depth]);
    }
}
/// Same native projections/FP32 boundaries; shared row-local work, each sequence's causal MTP attention.
fn kernel(e: *engine.Engine, ids: []const u32, segs: []const fw.Seg) !void {
    const R = ids.len;
    const b = &e.bm;
    const c = &e.c;
    const D = c.hidden;
    const m = e.w.mtp orelse return error.NoMtpHead;
    var one: [1]fw.Rank = undefined;
    var f = e.view(b, &one);
    one[0].st = segs[0].st;
    one[0].segs = segs;
    const set = f.set();
    try cuda.copy_source.enqueue(e.d, b.ids, std.mem.sliceAsBytes(ids), e.stream);
    try glue.embedB16(set, e.stream, b.ids, e.w.embed, b.me, R, D, 1);
    for (segs) |g| if (g.st.mtp_len == 0) try e.d.check(e.d.api.cuMemsetD8Async(b.me + g.row0 * D * 2, 0, D * 2, e.stream.handle), "MTP position0 embedding");
    try glue.rmsnorm(set, e.stream, b.me, D, m.enorm, b.mcat, 2 * D, null, c.eps, R, D);
    try glue.rmsnorm(set, e.stream, b.hin, D, m.hnorm, b.mcat + D * 2, 2 * D, null, c.eps, R, D);
    try b16.matmul(set, e.stream, b.mcat, 2 * D, m.eh, b.mx, b.kda.mm_part, R, D, 2 * D, false);
    try glue.rmsnormBy(set, e.stream, b.mx, D, m.layer.in_norm, b.normed, D, b.xs, c.eps, R, D);
    const layer = m.layer.mixer.mla;
    var sparse_any = false;
    for (segs) |g| {
        sparse_any = sparse_any or dsa.sparseRows(layer, g.st.mtp_len, g.rows);
    }
    try dsa.prepare(set, e.stream, layer, b.dsa, b.normed, D, R, sparse_any);
    for (segs) |g| try dsa.attend(.{ .set = set, .ops = f.m.ops() }, e.stream, layer, g.st.mtp_caches, b.dsa, g.row0, g.rows, g.st.mtp_pos_dev, g.st.mtp_len, @min(latent.chunksFor(g.st.mtp_len + g.rows), b.nch), b.part_rows);
    try latent.expandV(set, e.stream, b.dsa.ol, layer.wv, b.dsa.vn, R, layer.heads, layer.dv, layer.lw);
    try dsa.fill(set, e.stream, layer, b.dsa, 0, R, b.gathered);
    try f.resolve(R);
    try glue.residualAdd(set, e.stream, b.mx, b.mx, b.gathered, R, D, e.w.share.world);
    try glue.rmsnormBy(set, e.stream, b.mx, D, m.layer.post_norm, b.normed, D, b.xs, c.eps, R, D);
    try f.ffn(one[0], m.layer, R, b.gathered);
    try f.resolve(R);
    if (b.moe.hybrid) |h| {
        try h.check();
        if (h.result != .remote or h.result.remote.rows != R) return error.NoRemoteResult;
        try glue.residualAdd(set, e.stream, b.mx, b.mx, h.epilogue, R, D, 3);
    } else try glue.residualAdd(set, e.stream, b.mx, b.mx, b.gathered, R, D, e.w.share.world);
    try glue.rmsnorm(set, e.stream, b.mx, D, m.norm, b.fnormed, D, b.fxs, c.eps, R, D);
    // Native affine-Q4 tile2 accepts at most16 rows, while an MTP absorb wave may hold32.
    // Keep the full shared forward/state wave; slice only its final row-independent draft head.
    var head_at: usize = 0;
    while (head_at < R) {
        const count = @min(@as(usize, 16), R - head_at);
        try q4.matmul(&f.m.q4, e.stream, b.fnormed + head_at * D * 2, D, b.fxs + head_at * (D / 64) * 4, m.draft, b.logits + head_at * m.draft.n * 2, count);
        head_at += count;
    }
}
