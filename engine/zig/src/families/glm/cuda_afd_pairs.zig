//! qualification_0f991257: prompt chunk pairs on the AFD attention node (TF_AFD_PAIRS=1; evidence/qualification_0f991257/WIRE.md). Two consecutive prompt
//! chunks A and B run interleaved by layer on the one stream: the 5090 runs B's layer-L attention while the halves run
//! A's layer-L experts, and A's layer L+1 while they run B's layer L (IMP5's order, ours: evidence/IMP5/DESIGN.md §1).
//! Each chunk keeps its rows, kernels and exchanges; only their order changes, so the bits are the solo chunks'.
//! - The split: Afd's route, request and take (cuda_afd_attn.zig), and each MoE layer's front here (its hc_pre, its
//!   attention block, hc_post and the FFN's hc_pre); a chunk's layer loop resumes at each stop (`advance`).
//! - What crosses a stop gets a second copy for B (`Copies`): the residual streams x, hc_post's post and comb, the
//!   final-normed rows its MTP absorb reads, its position (pos_dev) and its header table. The rest is shared: between
//!   two steps of a chunk the other chunk writes only what that chunk has consumed (IMP5 §1's argument).
//! - Positions: B sees the stream at A's end (its own pos_dev) and its KDA layers the other half of the recurrent state;
//!   the commits go A then B; the MTP absorbs run in chunk order after the pair (WIRE.md §4).
//! qualification_c533375f (TF_AFD_INFLIGHT=2 with the pairs, WIRE.md §12): two requests in flight a link. Each chunk's next request goes
//! before the other chunk's take (B_0 with A_0; A_{p+1} before B_p's take), so the halves find the other window's
//! request landed when they finish one and the wire hides under their compute; B's shared-expert rows (sy) then cross
//! a step too and get B's copy.
//! Prior art: ByteDance's MegaScale-Infer ping-pong and SGLang's two-batch overlap; the exchanges in flight are Hugh
//! Madden's (@dangerm00se, glm53f-afd, MIT); TensorFold, its GLM engine and MCDMA are Ash Hart's (@ashxhart).
const std = @import("std");
const cuda = @import("cuda");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const prompt = @import("cuda_prompt.zig");
const cc = @import("cuda_compute.zig");
const fwd = @import("cuda_fwd.zig");
const afd = @import("cuda_afd_attn.zig");
const ae = @import("cuda_afd_engine.zig");
const structured = @import("cuda_afd_structured.zig");

const proto = cuda.xchg.proto;
const Fwd = fwd.Fwd;
const Buf = views.Buf;
const T = trace.T;

/// TF_AFD_PAIRS: 1 pairs a prompt's chunks (all three roles must agree: the settings' `pairs`); unset or 0 leaves the
/// role as it was; anything else is refused before the load (the halves' twin: cuda_afd_halves.zig, qualification_c2d4ee2e).
pub fn fromEnv() error{BadAfdPairs}!bool {
    const v = std.mem.span(std.c.getenv("TF_AFD_PAIRS") orelse return false);
    if (std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    std.debug.print("INFO attn: TF_AFD_PAIRS takes 1 or 0 (or unset), not \"{s}\"\n", .{v});
    return error.BadAfdPairs;
}

/// TF_AFD_INFLIGHT (qualification_c533375f): 2 keeps two requests in flight a link inside a pair, and needs TF_AFD_PAIRS=1 on the node
/// (`pairs`); unset or 1 is one in flight, as the pairs were; anything else is refused before the load (the halves'
/// twin: cuda_afd_halves.zig).
pub fn inflightFromEnv(pairs: bool) error{BadAfdInflight}!u8 {
    const v = std.mem.span(std.c.getenv("TF_AFD_INFLIGHT") orelse return 1);
    if (std.mem.eql(u8, v, "1")) return 1;
    if (std.mem.eql(u8, v, "2") and pairs) return 2;
    if (std.mem.eql(u8, v, "2")) std.debug.print("INFO attn: TF_AFD_INFLIGHT=2 needs TF_AFD_PAIRS=1 on every node\n", .{}) else std.debug.print("INFO attn: TF_AFD_INFLIGHT takes 1 or 2 (or unset), not \"{s}\"\n", .{v});
    return error.BadAfdInflight;
}

/// B's second copies, in the prompt Buffers' shapes (and the State's pos_dev, the provider's header table); with two
/// requests in flight (qualification_c533375f) B's shared-expert rows too, which A's next request would otherwise rewrite before B's take.
/// qualification_47b9bf3a: a prompt lane's MTP layer (cuda_afd_lanes_mtp.zig) adds the MTP residual and its position; null for a pair.
pub const Copies = struct {
    x: Buf,
    post: Buf,
    comb: Buf,
    fnormed: Buf,
    pos_dev: Buf,
    hdr: Buf,
    sy: ?Buf = null,
    mx: ?Buf = null,
    mtp_pos_dev: ?Buf = null,

    pub fn init(pb: *const bufs.Buffers, st: *const bufs.State, a: *const afd.Afd) Copies {
        const sy: ?Buf = if (a.inflight > 1) if (pb.sy) |s| fresh(s) else null else null;
        return .{ .x = fresh(pb.x), .post = fresh(pb.post), .comb = fresh(pb.comb), .fnormed = fresh(pb.fnormed), .pos_dev = fresh(st.pos_dev), .hdr = fresh(a.w.hdr), .sy = sy };
    }

    pub fn fresh(b: Buf) Buf {
        return Buf.of(b.dtype, b.shape[0..b.dims]);
    }
};

/// What a scheduler step is, for `Hook` (a test attributes each launch to its chunk, and reads IMP5's order back).
pub const Step = enum { stage, front, route, request, take, post, tail, absorb, commit, position, first };
/// `solo`: a prompt's odd last chunk (Attn.chunk); `pair`: the pair's own launch (B's position) and the first token
pub const Who = enum { a, b, solo, pair };
pub const Hook = struct { ctx: *anyopaque, at: *const fn (ctx: *anyopaque, who: Who, step: Step, pos: usize) anyerror!void };

/// IMP5's order on the wire (WIRE.md §4): A[0] B[0] A[1] B[1] …, the requests taken in that order, at most `most` in
/// flight: 1 (each taken before the next is posted), or 2 with TF_AFD_INFLIGHT=2 (WIRE.md §12); any other step is
/// refused (python's `_busy`).
pub const Turn = struct {
    count: usize,
    most: usize = 1,
    next: usize = 0,
    taken: usize = 0,

    pub fn request(t: *Turn, slot: u1, pos: usize) !void {
        if (t.next - t.taken >= t.most or t.next >= 2 * t.count or t.next != 2 * pos + slot) return error.OutOfStep;
        t.next += 1;
    }

    pub fn take(t: *Turn, slot: u1, pos: usize) !void {
        if (t.taken == t.next or t.taken != 2 * pos + slot) return error.OutOfStep;
        t.taken += 1;
    }

    pub fn done(t: Turn) bool {
        return t.taken == t.next and t.next == 2 * t.count;
    }
};

/// One window's provider state (Afd's fields of the forward in flight), kept while the other window's steps run.
const Window = struct { seq: u32, cur: usize, first: usize, count: usize, pos: usize, pair: afd.Pair };

/// The pairs of a node: B's copies on the device (and B's audit ring), whether the halves agreed, the test hook.
pub const Pairs = struct {
    c: Copies,
    /// the settings both halves agreed carry `pairs` (Node.attend passed with it): a pair is refused otherwise
    agreed: bool = false,
    dev: ?cuda.DeviceBuffer = null,
    /// TF_AFD_CHECK's ring for B's windows (A's is the provider's own), as Afd.audit makes it
    ring: ?afd.Ring = null,
    hook: ?Hook = null,
    /// pairs run (for the log)
    runs: u64 = 0,
    /// qualification_c533375f: the requests a pair keeps in flight a link (the provider's `inflight` at init: 1, or 2)
    inflight: u8 = 1,
    /// qualification_c533375f: B's seq of the last pair while no other forward has begun since (the provider's seq still there): the
    /// halves may still be reading that pair's last header in B's receive slot, so the next pair's B_0 waits for A_0's
    /// take (WIRE.md §12)
    last: ?u32 = null,

    /// For the attention node `x` (the copies unplaced: `attach`, or a test's Space by `walk`).
    pub fn init(x: *const ae.Attn) Pairs {
        return .{ .c = Copies.init(&x.e.pbuf, &x.e.st, x.a), .inflight = x.a.inflight };
    }

    /// The copies by path (`pair.x`, … `pair.hdr`), for a Space or an arena.
    pub fn walk(p: *Pairs, ctx: anytype, comptime visit: fn (@TypeOf(ctx), []const u8, *Buf) anyerror!void) !void {
        try views.walk(&p.c, views.Path.init("pair"), ctx, visit);
    }

    /// The device bytes `attach` takes (each copy at a 256-byte boundary): the forecast's share. qualification_c533375f (qualification_242abc20's segfault):
    /// sized on a copy of the copies, because Arena.visit places what it walks: walking the copies themselves after
    /// `attach` (attn's INFO line; the --dry path after the Space's placing) put every one of B's buffers back at its
    /// offset from 0, and B's first header copy (cuMemcpyHtoDAsync in Run.headers) wrote to that offset on attention-host.
    pub fn bytes(p: *const Pairs) !usize {
        var c = p.c;
        var a: bufs.Arena = .{};
        try views.walk(&c, views.Path.init("pair"), &a, bufs.Arena.visit);
        return a.used;
    }

    /// One device block for the copies, zeroed; on `d`, before the first prompt.
    pub fn attach(p: *Pairs, d: *const cuda.Driver) !void {
        if (p.dev != null) return error.Attached;
        var dev = try cuda.DeviceBuffer.alloc(d, try p.bytes());
        errdefer dev.free();
        try dev.fill8(0, null);
        var a: bufs.Arena = .{ .base = dev.ptr };
        try p.walk(&a, bufs.Arena.visit);
        p.dev = dev;
    }

    /// TF_AFD_CHECK on a GPU: B's pinned ring, as Afd.audit's for A (forwards of up to `rows` rows).
    pub fn audit(p: *Pairs, d: *const cuda.Driver, a: *const afd.Afd, rows: usize) !void {
        if (p.ring != null) return error.Attached;
        var r = afd.Ring.of(a.layout, rows, null);
        const host = try cuda.HostBuffer.alloc(d, r.total(a.positions()));
        r.bytes = host.bytes;
        r.owned = host;
        p.ring = r;
    }

    pub fn detach(p: *Pairs) void {
        if (p.ring) |*r| if (r.owned) |*h| h.free();
        if (p.dev) |*d| d.free();
        p.ring = null;
        p.dev = null;
    }

    /// decode.prefill under AFD in pairs: chunks (0, 1), (2, 3), … as pairs, an odd last chunk alone (Attn.chunk),
    /// then the sync and the check before the first token, as Attn.prefill. Refused before any launch unless the halves
    /// agreed pairs.
    pub fn prefill(p: *Pairs, x: *ae.Attn, f: *Fwd, ids: []const u32) !u32 {
        if (!p.agreed) return error.NotPorted;
        if (ids.len == 0) return error.EmptyPrompt;
        try x.alive(); // before State.reset's launches
        const e = x.e;
        if (x.structured_control != null) try structured.reset(x, f, ids) else try cc.reset(f, &e.st);
        const n = e.prefill_rows;
        var last: ?T = null;
        var start: usize = 0;
        while (start < ids.len) {
            if (ids.len - start > n) {
                const rb = @min(n, ids.len - start - n);
                var r: Run = .{ .p = p, .x = x, .f = f, .ids = ids, .start = start };
                last = try r.pair(n, rb);
                start += n + rb;
            } else {
                if (p.hook) |h| try h.at(h.ctx, .solo, .stage, 0);
                last = try x.chunk(f, ids, start);
                start = ids.len;
            }
        }
        if (p.hook) |h| try h.at(h.ctx, .pair, .first, 0);
        return x.firstToken(f, last.?);
    }

    /// prefill_interleave: `prefill` a step at a time: its preamble with the first, one pair (or the lone last chunk) a call from
    /// `start`, the first token with the last (null while steps remain); the same calls in the same order as the whole pass.
    pub fn step(p: *Pairs, x: *ae.Attn, f: *Fwd, ids: []const u32, start: *usize, first: bool) !?u32 {
        const e = x.e;
        if (first) {
            if (!p.agreed) return error.NotPorted;
            if (ids.len == 0) return error.EmptyPrompt;
            try x.alive(); // before State.reset's launches
            if (x.structured_control != null) try structured.reset(x, f, ids) else try cc.reset(f, &e.st);
        }
        if (start.* >= ids.len) return error.PromptStepPast;
        const n = e.prefill_rows;
        var last: T = undefined;
        if (ids.len - start.* > n) {
            const rb = @min(n, ids.len - start.* - n);
            var r: Run = .{ .p = p, .x = x, .f = f, .ids = ids, .start = start.* };
            last = try r.pair(n, rb);
            start.* += n + rb;
        } else {
            if (p.hook) |h| try h.at(h.ctx, .solo, .stage, 0);
            last = try x.chunk(f, ids, start.*);
            start.* = ids.len;
        }
        if (start.* < ids.len) return null;
        if (p.hook) |h| try h.at(h.ctx, .pair, .first, 0);
        return try x.firstToken(f, last);
    }
};

/// One chunk of a pair: its rows, its views of the prompt Buffers and the State, its options, its window and its
/// resumable layer loop (`li`, the next layer; `moe`, the MoE layer whose exchange is next).
const Chunk = struct {
    who: Who,
    start: usize,
    R: usize,
    b: bufs.Buffers,
    st: bufs.State,
    opt: layer.Opt = .{},
    w: Window,
    li: usize = 0,
    moe: ?*const views.Layer = null,
    planes: fwd.Planes = undefined,
    last: ?T = null,
};

/// One pair's run: the two chunks, the wire's turn, the provider's own header table and ring (A's).
const Run = struct {
    p: *Pairs,
    x: *ae.Attn,
    f: *Fwd,
    ids: []const u32,
    start: usize,
    turn: Turn = .{ .count = 0 },
    ca: Chunk = undefined,
    cb: Chunk = undefined,
    own_hdr: Buf = undefined,
    own_ring: ?afd.Ring = null,
    begun: bool = false,
    gate: structured.Gate = .{},

    fn hook(r: *Run, who: Who, step: Step, pos: usize) !void {
        try structured.action(r, who, step, pos);
        if (r.p.hook) |h| try h.at(h.ctx, who, step, pos);
    }

    /// Window `w` into the provider (its header table and audit ring by its slot: A's are the provider's own).
    fn load(r: *Run, w: Window) void {
        const a = r.x.a;
        a.seq = w.seq;
        a.cur = w.cur;
        a.mode = .chunk;
        a.first = w.first;
        a.count = w.count;
        a.pos = w.pos;
        a.pair = w.pair;
        a.w.hdr = if (w.pair.slot == 0) r.own_hdr else r.p.c.hdr;
        a.ring = if (w.pair.slot == 0) r.own_ring else r.p.ring;
    }

    /// The provider back to its own buffers with window `w` (complete or not), out of pair mode.
    pub fn restore(r: *Run, w: Window) void {
        r.load(w);
        const a = r.x.a;
        a.pair = null;
        a.w.hdr = r.own_hdr;
        a.ring = r.own_ring;
    }

    /// The pair's begin (WIRE.md §3): refused before any launch as Afd.beginAt refuses (a failed exchange, a dead node,
    /// a forward stopped between its exchanges dies here), and without the halves' agreement; then A takes seq + 1 and
    /// B seq + 2, and each window's header table is written (one blocking copy each, as beginAt's).
    fn begin(r: *Run, ra: usize, rb: usize) ![2]Window {
        const x = r.x;
        try x.finish(r.f);
        r.f.moe = x.a.provider();
        const ws = r.beginChecked(ra, rb) catch |err| {
            if (err == error.ExpertFailed or err == error.OutOfStep) _ = x.failure();
            return err;
        };
        x.forwards += 2;
        return ws;
    }

    fn beginChecked(r: *Run, ra: usize, rb: usize) ![2]Window {
        const a = r.x.a;
        if (a.wire) |w| try w.node.check();
        if (a.dead) return error.ExpertFailed;
        if (a.pos != a.first and a.pos != a.first + a.count) {
            a.dead = true;
            if (a.wire) |w| w.node.fail(.{ .why = .stopped, .peer = 0, .seq = w.node.doorbell() });
            return error.OutOfStep;
        }
        if (!r.p.agreed) return error.NotPorted;
        if (ra == 0 or ra > a.rows or rb == 0 or rb > a.rows or a.layers == 0) return error.Rows;
        const s = a.seq;
        r.own_hdr = a.w.hdr;
        r.own_ring = a.ring;
        r.turn = .{ .count = a.layers, .most = r.p.inflight };
        const ws = [2]Window{
            .{ .seq = s +% 1, .cur = ra, .first = 0, .count = a.layers, .pos = 0, .pair = .{ .slot = 0, .other = rb, .first = s +% 1 } },
            .{ .seq = s +% 2, .cur = rb, .first = 0, .count = a.layers, .pos = 0, .pair = .{ .slot = 1, .other = ra, .first = s +% 1 } },
        };
        errdefer r.restore(ws[0]); // nothing posted yet: A's window, not begun
        for (ws) |w| {
            r.load(w);
            try r.headers();
        }
        r.begun = true;
        return ws;
    }

    /// The loaded window's header rows into its table on the device (Afd.beginAt's tail; nothing on a CPU).
    fn headers(r: *Run) !void {
        const a = r.x.a;
        const h = a.host orelse return;
        const st = (r.f.gpu orelse return error.NoStream).stream;
        const n = a.positions() * proto.HDR_BYTES;
        a.headers(h.bytes[0..n]);
        try st.d.check(st.d.api.cuMemcpyHtoDAsync_v2(a.w.hdr.ptr, h.bytes.ptr, n, st.handle), "cuMemcpyHtoDAsync");
        try st.synchronize(); // python's hdr_tab.copy_ is blocking: the next rewrite must not race it
    }

    /// The pair's two chunks: A on the prompt Buffers and the State, B on its views (the copies, A's end, the other half
    /// of the KDA states); then IMP5's order (WIRE.md §4), the MTP absorbs and the commits in chunk order.
    fn pair(r: *Run, ra: usize, rb: usize) !T { return structured.pair(r, ra, rb); }
    pub fn pairRun(r: *Run, ra: usize, rb: usize) !T {
        const x = r.x;
        const e = x.e;
        const behind = r.p.last != null and r.p.last.? == x.a.seq; // the wire's last forward: a pair's (WIRE.md §12)
        const sy_b: ?Buf = if (r.p.inflight > 1) (r.p.c.sy orelse return error.NoExl3Scratch) else null; // qualification_c533375f: B's own
        const ws = try r.begin(ra, rb);
        errdefer if (x.structured_control == null) r.stopOrdinary();
        r.ca = .{ .who = .a, .start = r.start, .R = ra, .b = e.pbuf, .st = e.st, .w = ws[0] };
        r.cb = .{ .who = .b, .start = r.start + ra, .R = rb, .b = e.pbuf, .st = e.st, .w = ws[1] };
        const c = &r.p.c;
        r.cb.b.x = c.x;
        r.cb.b.post = c.post;
        r.cb.b.comb = c.comb;
        r.cb.b.fnormed = c.fnormed;
        if (sy_b) |s| r.cb.b.sy = s; // two in flight: A's next request (its shared expert) runs before B's take
        r.cb.st.pos = e.st.pos + ra;
        r.cb.st.pos_dev = c.pos_dev;
        if (e.st.rec.shape[1] > 0) r.cb.st.parity = 1 - e.st.parity; // A's prompt chunk commits its KDA layers as it runs
        // A to its first exchange, then B's attention while it flies
        try r.open(&r.ca);
        _ = try r.advance(&r.ca);
        try r.route(&r.ca);
        try r.request(&r.ca);
        try r.hook(.pair, .position, 0);
        try cc.fill(r.f, T.whole(c.pos_dev), @intCast(r.cb.st.pos)); // B's pos_dev: A's end (A commits after the pair)
        try r.open(&r.cb);
        _ = try r.advance(&r.cb);
        try r.route(&r.cb);
        if (r.turn.most > 1) try r.flight(behind) else try r.single();
        try r.tail(&r.cb);
        if (!r.turn.done() or r.ca.moe != null or r.cb.moe != null) return error.OutOfStep;
        r.restore(r.cb.w);
        r.begun = false;
        r.p.last = r.cb.w.seq; // until another forward begins (the absorbs, a lone chunk, decode)
        try r.audited();
        r.p.runs += 1;
        for ([_]*Chunk{ &r.ca, &r.cb }) |k| {
            if (e.st.rec.shape[1] > 0) e.st.parity = 1 - e.st.parity; // the chunk's KDA layers committed: cc.compute's flip
            try r.hook(k.who, .absorb, 0);
            try x.absorbChunk(r.f, &k.b, r.ids, k.start, k.R);
            try r.hook(k.who, .commit, 0);
            try cc.commit(r.f, &e.st, &e.pbuf, k.R, k.R);
        }
        return r.cb.last.?;
    }

    /// One request in flight (WIRE.md §4): each take, then the other window's request, then the taken window's rest.
    fn single(r: *Run) !void {
        while (true) {
            try r.take(&r.ca);
            try r.request(&r.cb);
            try r.post(&r.ca);
            const more = try r.advance(&r.ca);
            if (more) try r.route(&r.ca) else try r.tail(&r.ca);
            try r.take(&r.cb);
            if (more) try r.request(&r.ca);
            try r.post(&r.cb);
            if (!try r.advance(&r.cb)) break;
            try r.route(&r.cb);
        }
    }

    /// qualification_c533375f, two requests in flight (WIRE.md §12): B_0 goes with A_0, then each window's next request before the
    /// other's take, so the halves always have the other window's request landed. `behind` (a pair's tail was the
    /// wire's last forward): B_0 goes after A_0's take instead, once the halves have read that tail.
    fn flight(r: *Run, behind: bool) !void {
        if (!behind) try r.request(&r.cb);
        try r.take(&r.ca);
        if (behind) try r.request(&r.cb);
        while (true) {
            try r.post(&r.ca);
            if (try r.advance(&r.ca)) {
                try r.route(&r.ca);
                try r.request(&r.ca);
            } else try r.tail(&r.ca);
            try r.take(&r.cb);
            try r.post(&r.cb);
            if (!try r.advance(&r.cb)) break;
            try r.route(&r.cb);
            try r.request(&r.cb);
            try r.take(&r.ca);
        }
    }

    /// A pair that stops before its 2·count exchanges (an error in a step): the provider keeps the incomplete window, so
    /// the next begin dies (Node.fail(.stopped), python's die) and nothing more is sent (WIRE.md §6). A window not yet
    /// begun on the wire (pos 0) is not a stop.
    pub fn stopOrdinary(r: *Run) void {
        if (!r.begun) return;
        r.begun = false;
        const done_a = r.ca.w.pos == r.ca.w.count;
        r.restore(if (done_a) r.cb.w else r.ca.w);
    }

    /// TF_AFD_CHECK: each window hashed as a forward of its own (its seq, its ring), after the pair's one sync.
    fn audited(r: *Run) !void {
        const x = r.x;
        if (x.check == null) return;
        for ([_]Window{ r.ca.w, r.cb.w }) |w| {
            r.load(w);
            x.pending = x.a.ring != null;
            const done = x.finish(r.f);
            r.restore(r.cb.w);
            try done;
        }
    }

    /// The chunk's ids staged (B's against its own position) and its embedding: forward.compute's first launch.
    fn open(r: *Run, c: *Chunk) !void {
        const e = r.x.e;
        const m = e.m;
        try r.hook(c.who, .stage, 0);
        const ids = r.ids[c.start..][0..c.R];
        c.R = try cc.stage(r.f, &c.st, &c.b, ids);
        c.opt = .{ .nch = cc.chunksFor(&c.st, c.R), .host_pos = c.st.pos, .chip = e.chip, .prompt = &e.prompt };
        try r.f.tri.embed(c.b.ids.ptr, cc.table(e.w.embed), c.b.x.ptr, c.R, m.hidden, m.streams);
    }

    /// The resumable layer loop: chunk c's dense layers whole (prompt.layerForward), up to the next MoE layer's front,
    /// where it stops (true); false once every layer ran.
    fn advance(r: *Run, c: *Chunk) !bool {
        const e = r.x.e;
        c.moe = null;
        while (c.li < e.w.layers.len) {
            const l = &e.w.layers[c.li];
            c.li += 1;
            try r.hook(c.who, .front, c.li - 1);
            if (l.mlp != null) {
                try prompt.layerForward(r.f, l, &c.st, &c.b, c.R, c.opt);
                continue;
            }
            try front(r.f, l, &c.st, &c.b, c.R, c.opt);
            c.moe = l;
            return true;
        }
        return false;
    }

    fn route(r: *Run, c: *Chunk) !void {
        try r.hook(c.who, .route, c.w.pos);
        r.load(c.w);
        try r.x.a.route(r.f, c.moe.?, &c.b, c.R);
    }

    fn request(r: *Run, c: *Chunk) !void {
        try r.hook(c.who, .request, c.w.pos);
        try r.turn.request(c.w.pair.slot, c.w.pos);
        r.load(c.w);
        try r.x.a.request(r.f, c.moe.?, &c.b, c.R, c.opt);
        c.w.pos = r.x.a.pos;
    }

    pub fn take(r: *Run, c: *Chunk) !void {
        try r.hook(c.who, .take, c.w.pos - 1);
        try r.turn.take(c.w.pair.slot, c.w.pos - 1);
        r.load(c.w);
        c.planes = try r.x.a.take(r.f, &c.b, c.R);
    }

    /// layer_forward's last launch for a MoE layer: hc_post of the three planes into the chunk's streams.
    fn post(r: *Run, c: *Chunk) !void {
        const m = r.x.e.m;
        try r.hook(c.who, .post, c.w.pos - 1);
        try r.f.tri.hcPost(c.b.x.ptr, c.b.x.ptr, c.planes.g, c.b.post.ptr, c.b.comb.ptr, c.planes.world, c.R, m.hidden, m.streams);
    }

    /// forward.compute's tail (the streams' mean, the final norm, the head on the last row) and prefill's two clones.
    fn tail(r: *Run, c: *Chunk) !void {
        const e = r.x.e;
        const m = e.m;
        const D = m.hidden;
        const b = &c.b;
        try r.hook(c.who, .tail, 0);
        try r.f.tri.streamMean(b.x.ptr, b.hidden.ptr, c.R, D, m.streams);
        try r.f.tri.rmsnorm(b.hidden.ptr, D, e.w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, m.eps, c.R, D);
        const out = layer.rowsOf(b.logits, 1);
        try qmm.mm(r.f, b.prefill, cc.rows(b.fnormed, c.R - 1, 1), e.w.head, cc.rows(b.fxs, c.R - 1, 1), out, false, b.sk, c.opt.chip);
        c.last = try cc.clone(r.f, &e.tmp, out, &e.last);
        _ = try cc.clone(r.f, &e.tmp, cc.rows(b.fnormed, c.R - 1, 1), &e.last_hidden);
    }
};

/// A MoE layer's launches up to its exchange (cuda_layer.layerForward and cuda_prompt.layerForward without ffnHalf's
/// block and last hc_post): hc_pre, the KDA or DSA block (a prompt chunk's: the engine's Scratch), hc_post, then the
/// FFN's hc_pre into b.normed, b.xs, b.post and b.comb. qualification_9e49592e's lanes run it too.
pub fn front(f: *Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: layer.Opt) !void {
    const m = f.m;
    if (R == 0 or R > b.rows) return error.Rows;
    const h = l.attn_hc orelse return error.PlainLayer;
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.in_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
    const g = switch (l.kind) {
        .kda => try layer.kdaBlock(f, l, st, b, R, opt),
        .dsa => try prompt.dsaBlock(f, l, st, b, R, opt),
    };
    try f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
    const k = l.ffn_hc orelse return error.PlainLayer;
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = k.@"fn".ptr, .base = k.base.ptr, .scale = k.scale.ptr }, l.post_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "TF_AFD_PAIRS: unset or 0 off, 1 on, anything else BadAfdPairs; the turn refuses a second request in flight and every step out of IMP5's order" {
    defer _ = unsetenv("TF_AFD_PAIRS");
    _ = unsetenv("TF_AFD_PAIRS");
    try std.testing.expect(!try fromEnv());
    _ = setenv("TF_AFD_PAIRS", "0", 1);
    try std.testing.expect(!try fromEnv());
    _ = setenv("TF_AFD_PAIRS", "1", 1);
    try std.testing.expect(try fromEnv());
    _ = setenv("TF_AFD_PAIRS", "true", 1);
    try std.testing.expectError(error.BadAfdPairs, fromEnv());
    var t: Turn = .{ .count = 2 };
    try std.testing.expectError(error.OutOfStep, t.take(0, 0)); // nothing in flight
    try std.testing.expectError(error.OutOfStep, t.request(1, 0)); // B before A
    try t.request(0, 0);
    try std.testing.expectError(error.OutOfStep, t.request(1, 0)); // a second request in flight
    try std.testing.expectError(error.OutOfStep, t.take(1, 0)); // the wrong window
    try t.take(0, 0);
    try std.testing.expectError(error.OutOfStep, t.request(0, 1)); // A[1] before B[0]
    try t.request(1, 0);
    try t.take(1, 0);
    try t.request(0, 1);
    try t.take(0, 1);
    try std.testing.expect(!t.done());
    try t.request(1, 1);
    try t.take(1, 1);
    try std.testing.expect(t.done());
    try std.testing.expectError(error.OutOfStep, t.request(0, 2)); // past the pair
}

test "qualification_c533375f: TF_AFD_INFLIGHT: unset or 1 one in flight, 2 two (with TF_AFD_PAIRS=1 only), anything else BadAfdInflight; the turn keeps two in flight in IMP5's order and refuses a third, a take out of order and B ahead of A" {
    defer _ = unsetenv("TF_AFD_INFLIGHT");
    _ = unsetenv("TF_AFD_INFLIGHT");
    try std.testing.expectEqual(@as(u8, 1), try inflightFromEnv(true));
    try std.testing.expectEqual(@as(u8, 1), try inflightFromEnv(false));
    _ = setenv("TF_AFD_INFLIGHT", "1", 1);
    try std.testing.expectEqual(@as(u8, 1), try inflightFromEnv(false));
    _ = setenv("TF_AFD_INFLIGHT", "2", 1);
    try std.testing.expectEqual(@as(u8, 2), try inflightFromEnv(true));
    try std.testing.expectError(error.BadAfdInflight, inflightFromEnv(false)); // two in flight only inside a pair
    for ([_][*:0]const u8{ "0", "3", "true", "" }) |v| {
        _ = setenv("TF_AFD_INFLIGHT", v, 1);
        try std.testing.expectError(error.BadAfdInflight, inflightFromEnv(true));
    }
    var t: Turn = .{ .count = 2, .most = 2 };
    try std.testing.expectError(error.OutOfStep, t.request(1, 0)); // B before A
    try t.request(0, 0);
    try t.request(1, 0); // B_0 with A_0: two in flight
    try std.testing.expectError(error.OutOfStep, t.request(0, 1)); // a third
    try std.testing.expectError(error.OutOfStep, t.take(1, 0)); // B_0's take before A_0's
    try t.take(0, 0);
    try std.testing.expectError(error.OutOfStep, t.request(1, 1)); // B_1 before A_1
    try t.request(0, 1); // A_1 before B_0's take
    try std.testing.expectError(error.OutOfStep, t.take(0, 1)); // A_1's take before B_0's
    try t.take(1, 0);
    try t.request(1, 1);
    try std.testing.expect(!t.done());
    try t.take(0, 1);
    try t.take(1, 1);
    try std.testing.expect(t.done());
    try std.testing.expectError(error.OutOfStep, t.take(1, 1)); // nothing in flight
    try std.testing.expectError(error.OutOfStep, t.request(0, 2)); // past the pair
}
