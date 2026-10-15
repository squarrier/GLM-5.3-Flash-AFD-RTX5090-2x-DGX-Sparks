//! qualification_9e49592e: prompt chunk lanes on the AFD attention node (TF_AFD_LANES=N, N = 3 or 4, with TF_AFD_PAIRS=1;
//! evidence/qualification_0f991257/WIRE.md §13). Up to N consecutive prompt chunks run interleaved by layer on the one stream, with up to
//! N requests in flight a link: qualification_0f991257's pair scheduler (cuda_afd_pairs.zig) generalised to N lanes, in the order python
//! 2.15's 0026 lanes run (prefill_pairs._forward_lanes), three rules in turn:
//! 1. the next exchange goes out when its lane stopped before it and fewer than N are in flight (lane order within each
//!    MoE layer: lane 0's, lane 1's, ..., then lane 0's next layer, which the halves follow);
//! 2. else a lane runs its attention to its next exchange (or its tail) when it needs no reply and the lane before it
//!    already ran that layer's attention, tried from the lane whose exchange goes out next;
//! 3. else the oldest exchange in flight is taken. So every take has every other lane's attention it can run queued
//!    before it (IMP5's order, extended), and the halves find the next request landed when they reply.
//! Each chunk keeps its rows, kernels and exchanges; only their order changes, so the bits are the solo chunks'. Lane 0
//! runs on the prompt Buffers and the State; lanes 1..N-1 on their own crossing buffers (cuda_afd_pairs.Copies: x,
//! post / comb, fnormed, pos_dev, the header table, and sy, which crosses a step with more than one in flight). The
//! routed rows (normed, xs, pick, wts) and afd_recv stay shared: a lane past its route sends before any other lane's
//! attention runs (rule 1 first; a lane held behind a group waits only while every other lane is in flight), and a
//! lane's take is followed by its hc_post before the next take (rule 3 runs only when no lane can run). The MTP absorbs
//! of a pair's group, and of the prompt's last chunk, run after the group in chunk order, as the pairs'; qualification_47b9bf3a: every
//! other chunk's absorb is its lane's extra layer (cuda_afd_lanes_mtp.zig). The lanes and the exchanges in flight are
//! Hugh Madden's (@dangerm00se, github.com/hughmadden; glm53f-afd run_lanes, MIT); python 2.15's 0026 / 0027 are our
//! recipe's code written to them; the pair scheduler is IMP5's and qualification_0f991257's; TensorFold, its GLM engine and MCDMA are Ash
//! Hart's (@ashxhart).
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
const pairs = @import("cuda_afd_pairs.zig");
const lmtp = @import("cuda_afd_lanes_mtp.zig");
const structured = @import("cuda_afd_structured.zig");

const proto = cuda.xchg.proto;
const Fwd = fwd.Fwd;
const Buf = views.Buf;
const T = trace.T;
const Step = pairs.Step;

/// TF_AFD_LANES: 3 or 4 runs a prompt's chunks in lanes of up to N with N requests in flight a link (all three roles
/// must agree: the settings' lanes=N); it needs TF_AFD_PAIRS=1 (`paired`) and TF_AFD_INFLIGHT unset (the lanes set
/// it); unset or 0 leaves the role as it was; anything else is refused before the load (the halves' twin:
/// cuda_afd_halves.lanesFromEnv).
pub fn fromEnv(paired: bool) error{BadAfdLanes}!u8 {
    const v = std.mem.span(std.c.getenv("TF_AFD_LANES") orelse return 0);
    if (std.mem.eql(u8, v, "0")) return 0;
    const n: u8 = if (std.mem.eql(u8, v, "3")) 3 else if (std.mem.eql(u8, v, "4")) 4 else {
        std.debug.print("INFO attn: TF_AFD_LANES takes 3, 4 or 0 (or unset), not \"{s}\"\n", .{v});
        return error.BadAfdLanes;
    };
    if (!paired) {
        std.debug.print("INFO attn: TF_AFD_LANES={s} needs TF_AFD_PAIRS=1 on every node\n", .{v});
        return error.BadAfdLanes;
    }
    if (std.c.getenv("TF_AFD_INFLIGHT") != null) {
        std.debug.print("INFO attn: TF_AFD_LANES={s} keeps {s} requests in flight a link: leave TF_AFD_INFLIGHT unset\n", .{ v, v });
        return error.BadAfdLanes;
    }
    return n;
}

/// Who a scheduler step is for, for `Hook`: lane k of a group, a prompt's lone last chunk, the group's own launches
/// (a lane's position fill) and the first token.
pub const Who = enum { lane, solo, group };
pub const Hook = struct { ctx: *anyopaque, at: *const fn (ctx: *anyopaque, who: Who, k: u8, step: Step, pos: usize) anyerror!void };

/// A group's order on the wire (WIRE.md §13): lane 0's position p, lane 1's, ..., lane n-1's, then p + 1; taken in that
/// order; at most `most` in flight. `behind`: the group right behind a group (nothing else on the wire since): the
/// halves may still read that group's last header in the slot request `slots - 1` takes, so it waits for the first take.
/// `short` (qualification_47b9bf3a): the last lane has one position fewer (the MTP layer: its absorb runs after the group).
pub const Turn = struct {
    n: usize,
    count: usize,
    most: usize,
    slots: usize = proto.SLOTS,
    behind: bool = false,
    short: usize = 0,
    next: usize = 0,
    taken: usize = 0,

    /// The group's requests on each link.
    pub fn total(t: Turn) usize {
        return t.n * t.count - t.short;
    }

    /// Whether the next request may go now.
    pub fn may(t: Turn) bool {
        if (t.next >= t.total() or t.next - t.taken >= t.most) return false;
        return !(t.behind and t.taken == 0 and t.next + 1 >= t.slots);
    }

    pub fn request(t: *Turn, k: usize, pos: usize) !void {
        if (!t.may() or t.next != pos * t.n + k) return error.OutOfStep;
        t.next += 1;
    }

    pub fn take(t: *Turn, k: usize, pos: usize) !void {
        if (t.taken == t.next or t.taken != pos * t.n + k) return error.OutOfStep;
        t.taken += 1;
    }

    /// The lane whose request is the oldest in flight, or null.
    pub fn oldest(t: Turn) ?usize {
        return if (t.taken < t.next) t.taken % t.n else null;
    }

    pub fn done(t: Turn) bool {
        return t.taken == t.next and t.next == t.total();
    }
};

/// The lanes of a node: lanes 1..N-1's copies on the device (and their audit rings), whether the halves agreed, the
/// test hook.
pub const Lanes = struct {
    /// TF_AFD_LANES' N: a group's most lanes, and the requests in flight a link
    n: u8,
    /// lanes 1..N-1's copies (lane 0 runs on the prompt Buffers and the State)
    c: [proto.LANES_MAX - 1]pairs.Copies,
    /// the settings both halves agreed carry lanes=N (Node.attend passed with it): a group is refused otherwise
    agreed: bool = false,
    dev: ?cuda.DeviceBuffer = null,
    /// TF_AFD_CHECK's rings for lanes 1..N-1 (lane 0's is the provider's own), as Afd.audit makes it
    rings: [proto.LANES_MAX - 1]?afd.Ring = @splat(null),
    hook: ?Hook = null,
    /// groups run (for the log)
    runs: u64 = 0,
    /// the last group's last seq while no other forward has begun since (the provider's seq still there)
    last: ?u32 = null,

    /// For the attention node `x` (its provider's `lanes` N; the copies unplaced: `attach`, or a test's Space by `walk`;
    /// with the provider's MTP position, each lane's MTP residual and position too: cuda_afd_lanes_mtp.copies).
    pub fn init(x: *const ae.Attn) !Lanes {
        const n = x.a.lanes;
        if (n < 3 or n > proto.LANES_MAX or x.a.inflight != n) return error.BadAfdLanes;
        var l: Lanes = .{ .n = n, .c = undefined };
        for (&l.c) |*c| c.* = lmtp.copies(&x.e.pbuf, &x.e.st, x.a);
        return l;
    }

    fn path(k: usize) views.Path {
        var buf: [8]u8 = undefined;
        return views.Path.init(std.fmt.bufPrint(&buf, "lane{d}", .{k}) catch unreachable);
    }

    /// The copies by path (`lane1.x`, … `laneK.sy`), for a Space or an arena.
    pub fn walk(l: *Lanes, ctx: anytype, comptime visit: fn (@TypeOf(ctx), []const u8, *Buf) anyerror!void) !void {
        for (l.c[0 .. l.n - 1], 1..) |*c, k| try views.walk(c, path(k), ctx, visit);
    }

    /// The device bytes `attach` takes, sized on a copy of the copies (qualification_c533375f's rule: Arena.visit places what it walks).
    pub fn bytes(l: *const Lanes) !usize {
        var c = l.c;
        var a: bufs.Arena = .{};
        for (c[0 .. l.n - 1], 1..) |*x, k| try views.walk(x, path(k), &a, bufs.Arena.visit);
        return a.used;
    }

    /// One device block for the copies, zeroed; on `d`, before the first prompt.
    pub fn attach(l: *Lanes, d: *const cuda.Driver) !void {
        if (l.dev != null) return error.Attached;
        var dev = try cuda.DeviceBuffer.alloc(d, try l.bytes());
        errdefer dev.free();
        try dev.fill8(0, null);
        var a: bufs.Arena = .{ .base = dev.ptr };
        try l.walk(&a, bufs.Arena.visit);
        l.dev = dev;
    }

    /// TF_AFD_CHECK on a GPU: lanes 1..N-1's pinned rings, as Afd.audit's for lane 0 (forwards of up to `rows` rows).
    pub fn audit(l: *Lanes, d: *const cuda.Driver, a: *const afd.Afd, rows: usize) !void {
        for (l.rings[0 .. l.n - 1]) |*ring| {
            if (ring.* != null) return error.Attached;
            var r = afd.Ring.of(a.layout, rows, null);
            const host = try cuda.HostBuffer.alloc(d, r.total(a.positions()));
            r.bytes = host.bytes;
            r.owned = host;
            ring.* = r;
        }
    }

    pub fn detach(l: *Lanes) void {
        for (&l.rings) |*ring| {
            if (ring.*) |*r| if (r.owned) |*h| h.free();
            ring.* = null;
        }
        if (l.dev) |*d| d.free();
        l.dev = null;
    }

    /// decode.prefill under AFD in lanes: groups of up to N consecutive chunks (a remainder of two runs as a pair's
    /// group, mode 7, as python's begin_pair), a lone last chunk alone (Attn.chunk), then the sync and the check before
    /// the first token, as Attn.prefill. Refused before any launch unless the halves agreed the lanes.
    pub fn prefill(l: *Lanes, x: *ae.Attn, f: *Fwd, ids: []const u32) !u32 {
        if (!l.agreed) return error.NotPorted;
        if (ids.len == 0) return error.EmptyPrompt;
        try x.alive(); // before State.reset's launches
        const e = x.e;
        if (x.structured_control != null) try structured.reset(x, f, ids) else try cc.reset(f, &e.st);
        const n = e.prefill_rows;
        var last: ?T = null;
        var start: usize = 0;
        while (start < ids.len) {
            const left = (ids.len - start + n - 1) / n;
            if (left >= 2) {
                var rows: [proto.LANES_MAX]usize = undefined;
                const k = @min(l.n, left);
                var s = start;
                for (rows[0..k]) |*r| {
                    r.* = @min(n, ids.len - s);
                    s += r.*;
                }
                var r: Run = .{ .l = l, .x = x, .f = f, .ids = ids, .start = start };
                last = try r.group(rows[0..k]);
                start = s;
            } else {
                if (l.hook) |h| try h.at(h.ctx, .solo, 0, .stage, 0);
                last = try x.chunk(f, ids, start);
                start = ids.len;
            }
        }
        if (l.hook) |h| try h.at(h.ctx, .group, 0, .first, 0);
        return x.firstToken(f, last.?);
    }

    /// prefill_interleave: `prefill` a step at a time: its preamble with the first, one group (or the lone last chunk) a call from
    /// `start`, the first token with the last (null while steps remain); the same calls in the same order as the whole pass.
    pub fn step(l: *Lanes, x: *ae.Attn, f: *Fwd, ids: []const u32, start: *usize, first: bool) !?u32 {
        const e = x.e;
        if (first) {
            if (!l.agreed) return error.NotPorted;
            if (ids.len == 0) return error.EmptyPrompt;
            try x.alive(); // before State.reset's launches
            if (x.structured_control != null) try structured.reset(x, f, ids) else try cc.reset(f, &e.st);
        }
        if (start.* >= ids.len) return error.PromptStepPast;
        const n = e.prefill_rows;
        var last: T = undefined;
        const left = (ids.len - start.* + n - 1) / n;
        if (left >= 2) {
            var rows: [proto.LANES_MAX]usize = undefined;
            const k = @min(l.n, left);
            var s = start.*;
            for (rows[0..k]) |*r| {
                r.* = @min(n, ids.len - s);
                s += r.*;
            }
            var r: Run = .{ .l = l, .x = x, .f = f, .ids = ids, .start = start.* };
            last = try r.group(rows[0..k]);
            start.* = s;
        } else {
            if (l.hook) |h| try h.at(h.ctx, .solo, 0, .stage, 0);
            last = try x.chunk(f, ids, start.*);
            start.* = ids.len;
        }
        if (start.* < ids.len) return null;
        if (l.hook) |h| try h.at(h.ctx, .group, 0, .first, 0);
        return try x.firstToken(f, last);
    }
};

/// One window's provider state (Afd's fields of the forward in flight), kept while the other lanes' steps run: its
/// lane `slot`, and its pair words (a group of two) or lane words (three or more).
const Window = struct { seq: u32, cur: usize, first: usize, count: usize, pos: usize, slot: u8, pair: ?afd.Pair, lane: ?afd.Lane };

/// Where a lane's loop stopped (python's _steps): not started, before its exchange (routed), waiting for its reply,
/// its reply taken, or ended (its tail run).
const At = enum { idle, send, wait, done, end };

/// One lane of a group: its rows, its views of the prompt Buffers and the State, its options, its window, its
/// resumable layer loop (`li`, the next layer; `moe`, the MoE layer whose exchange is next) and where it stopped.
/// qualification_47b9bf3a: `absorb`, the MTP layer is its extra layer (its window one position longer); `mtp`, that layer is under way.
const Chunk = struct {
    k: u8,
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
    at: At = .idle,
    /// MoE layers whose attention this lane ran (python's sent[k])
    sent: usize = 0,
    absorb: bool = false,
    mtp: bool = false,
    mopt: layer.Opt = .{},
};

/// One group's run: its lanes, the wire's turn, the provider's own header table and ring (lane 0's).
const Run = struct {
    l: *Lanes,
    x: *ae.Attn,
    f: *Fwd,
    ids: []const u32,
    start: usize,
    n: usize = 0,
    turn: Turn = .{ .n = 1, .count = 0, .most = 1 },
    ch: [proto.LANES_MAX]Chunk = undefined,
    own_hdr: Buf = undefined,
    own_ring: ?afd.Ring = null,
    begun: bool = false,
    gate: structured.Gate = .{},

    fn hook(r: *Run, who: Who, k: u8, step: Step, pos: usize) !void {
        try structured.laneAction(r, k, step, pos);
        if (r.l.hook) |h| try h.at(h.ctx, who, k, step, pos);
    }

    /// Window `w` into the provider (its header table and audit ring by its lane: lane 0's are the provider's own).
    fn load(r: *Run, w: Window) void {
        const a = r.x.a;
        a.seq = w.seq;
        a.cur = w.cur;
        a.mode = .chunk;
        a.first = w.first;
        a.count = w.count;
        a.pos = w.pos;
        a.pair = w.pair;
        a.lane = w.lane;
        a.w.hdr = if (w.slot == 0) r.own_hdr else r.l.c[w.slot - 1].hdr;
        a.ring = if (w.slot == 0) r.own_ring else r.l.rings[w.slot - 1];
    }

    /// The provider back to its own buffers with window `w` (complete or not), out of the group.
    pub fn restore(r: *Run, w: Window) void {
        r.load(w);
        const a = r.x.a;
        a.pair = null;
        a.lane = null;
        a.w.hdr = r.own_hdr;
        a.ring = r.own_ring;
    }

    /// The group's begin (WIRE.md §13): refused before any launch as Afd.beginAt refuses (a failed exchange, a dead
    /// node, a forward stopped between its exchanges dies here), and without the halves' agreement; then lane k takes
    /// seq + 1 + k, and each lane's header table is written (one blocking copy each, as beginAt's).
    fn begin(r: *Run, rows: []const usize) !void {
        const x = r.x;
        try x.finish(r.f);
        r.f.moe = x.a.provider();
        r.beginChecked(rows) catch |err| {
            if (err == error.ExpertFailed or err == error.OutOfStep) _ = x.failure();
            return err;
        };
        x.forwards += rows.len;
    }

    fn beginChecked(r: *Run, rows: []const usize) !void {
        const a = r.x.a;
        if (a.wire) |w| try w.node.check();
        if (a.dead) return error.ExpertFailed;
        if (a.pos != a.first and a.pos != a.first + a.count) {
            a.dead = true;
            if (a.wire) |w| w.node.fail(.{ .why = .stopped, .peer = 0, .seq = w.node.doorbell() });
            return error.OutOfStep;
        }
        if (!r.l.agreed) return error.NotPorted;
        if (rows.len < 2 or rows.len > r.l.n or a.layers == 0) return error.Rows;
        for (rows) |R| if (R == 0 or R > a.rows) return error.Rows;
        const s = a.seq;
        const slots: usize = if (a.wire) |w| w.node.reg.slots else proto.SLOTS;
        r.own_hdr = a.w.hdr;
        r.own_ring = a.ring;
        r.n = rows.len;
        var in: [proto.LANES_MAX]bool = undefined; // qualification_47b9bf3a: the lanes whose MTP absorb is their extra layer
        var from = r.start;
        for (rows, 0..) |R, k| {
            in[k] = lmtp.inLane(r.x, rows.len, r.ids.len, from, R);
            from += R;
        }
        r.turn = .{ .n = rows.len, .count = a.layers + @intFromBool(in[0]), .most = @min(r.l.n, slots), .slots = slots, .behind = r.l.last != null and r.l.last.? == s, .short = @intFromBool(in[0] and !in[rows.len - 1]) };
        for (rows, 0..) |R, k| {
            const next = rows[(k + 1) % rows.len];
            const first = s +% 1;
            r.ch[k].w = .{
                .seq = s +% @as(u32, @intCast(k + 1)), .cur = R, .first = 0, .count = a.layers + @intFromBool(in[k]), .pos = 0, .slot = @intCast(k),
                .pair = if (rows.len == 2) .{ .slot = @intCast(k), .other = next, .first = first } else null,
                .lane = if (rows.len > 2) .{ .slot = @intCast(k), .next = next, .first = first, .n = @intCast(rows.len) } else null,
            };
        }
        errdefer r.restore(r.ch[0].w); // nothing posted yet: lane 0's window, not begun
        for (r.ch[0..rows.len]) |*c| {
            r.load(c.w);
            try r.headers();
        }
        r.begun = true;
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

    /// The group's lanes: lane 0 on the prompt Buffers and the State, lane k on its views (its copies, the end of the
    /// lanes before it, the KDA states' other half on odd lanes; qualification_47b9bf3a: its MTP residual and the MTP length the lanes
    /// before it reach); then the three rules (the module's doc), and in chunk order each chunk's MTP length or absorb
    /// (cuda_afd_lanes_mtp.settle) and its commit.
    fn group(r: *Run, rows: []const usize) !T { return structured.group(r, rows); }
    pub fn groupRun(r: *Run, rows: []const usize) !T {
        const x = r.x;
        const e = x.e;
        try r.begin(rows);
        errdefer if (x.structured_control == null) r.stopOrdinary();
        var start = r.start;
        for (r.ch[0..r.n], 0..) |*c, k| {
            const w = c.w;
            c.* = .{ .k = @intCast(k), .start = start, .R = rows[k], .b = e.pbuf, .st = e.st, .w = w, .absorb = w.count > x.a.layers };
            start += rows[k];
            if (k == 0) continue;
            const cp = &r.l.c[k - 1];
            c.b.x = cp.x;
            c.b.post = cp.post;
            c.b.comb = cp.comb;
            c.b.fnormed = cp.fnormed;
            c.b.sy = cp.sy orelse return error.NoExl3Scratch; // N in flight: a later lane's request runs before its take
            c.st.pos = r.ch[k - 1].st.pos + rows[k - 1];
            c.st.pos_dev = cp.pos_dev;
            if (e.st.rec.shape[1] > 0 and k % 2 == 1) c.st.parity = 1 - e.st.parity; // the lane before commits its KDA layers as it runs
            if (cp.mx) |mx| c.b.mx = mx; // qualification_47b9bf3a: the MTP residual crosses the MTP layer's exchange
            if (cp.mtp_pos_dev) |d| c.st.mtp_pos_dev = d;
            c.st.mtp_len = r.ch[k - 1].st.mtp_len + rows[k - 1]; // an in-lane lane's are all in-lane before it
        }
        try r.lanes();
        if (!r.turn.done()) return error.OutOfStep;
        for (r.ch[0..r.n]) |*c| if (c.at != .end or c.moe != null) return error.OutOfStep;
        const end = r.ch[r.n - 1].w;
        r.restore(end);
        r.begun = false;
        r.l.last = end.seq; // until another forward begins (the absorbs, a lone chunk, decode)
        try r.audited();
        r.l.runs += 1;
        for (r.ch[0..r.n]) |*c| {
            if (e.st.rec.shape[1] > 0) e.st.parity = 1 - e.st.parity; // the chunk's KDA layers committed: cc.compute's flip
            try r.hook(.lane, c.k, .absorb, 0);
            try lmtp.settle(r.f, x, c.absorb, &c.b, r.ids, c.start, c.R);
            try r.hook(.lane, c.k, .commit, 0);
            try cc.commit(r.f, &e.st, &e.pbuf, c.R, c.R);
        }
        return r.ch[r.n - 1].last.?;
    }

    /// python's _forward_lanes over the group's lanes: rule 1, else rule 2, else rule 3, until every lane ended. With N
    /// in flight a lane's exchange goes out as soon as its attention ran: the routed rows (normed, xs, pick, wts) stay
    /// shared, because no other lane's step runs while one waits to send (only a take, behind a group: `Turn.behind`);
    /// afd_recv stays shared, because a lane past its take is the only lane that can run (python's argument).
    fn lanes(r: *Run) !void {
        var turn: usize = 0;
        while (true) {
            var live = false;
            for (r.ch[0..r.n]) |c| live = live or c.at != .end;
            if (!live) return;
            const t = &r.ch[turn];
            if (t.at == .send and r.turn.may()) { // 1. the next exchange out, in the wire's order
                try r.request(t);
                t.at = .wait;
                turn = (turn + 1) % r.n;
                continue;
            }
            for (r.ch[0..r.n], 0..) |c, k| if (c.at == .send and k != turn) return error.OutOfStep;
            if (t.at != .send) { // 2. a lane's attention to its next exchange, or its tail
                if (r.readyFrom(turn)) |k| {
                    try r.run(&r.ch[k]);
                    continue;
                }
            }
            const k = r.turn.oldest() orelse return error.OutOfStep; // 3. the oldest exchange in flight taken
            try r.take(&r.ch[k]);
            r.ch[k].at = .done;
        }
    }

    /// Lane k's next phase may run: it needs no reply (not started, or its reply taken), and the lane before it already
    /// ran that phase (it ran more MoE layers' attention, or ended).
    fn ready(r: *const Run, k: usize) bool {
        const c = &r.ch[k];
        if (c.at != .idle and c.at != .done) return false;
        if (k == 0) return true;
        const p = &r.ch[k - 1];
        return p.at == .end or p.sent > c.sent;
    }

    fn readyFrom(r: *const Run, turn: usize) ?usize {
        for (0..r.n) |i| {
            const k = (turn + i) % r.n;
            if (r.ready(k)) return k;
        }
        return null;
    }

    /// A lane's attention: its open (or the hc_post of its reply), its layers to the next MoE layer's route (then it
    /// stops before the exchange), or to its end and its tail. qualification_47b9bf3a: an absorbing lane's tail goes on to the MTP layer's
    /// route (its extra exchange, at position `layers`), and that layer's reply ends it (cuda_afd_lanes_mtp.back).
    fn run(r: *Run, c: *Chunk) !void {
        if (c.mtp) {
            try r.hook(.lane, c.k, .post, c.w.pos - 1);
            try lmtp.back(r.f, r.x, &c.b, c.R, c.planes, c.mopt);
            c.moe = null;
            c.at = .end;
            return;
        }
        if (c.at == .idle) try r.open(c) else try r.post(c);
        if (try r.advance(c)) {
            try r.route(c);
            c.sent += 1;
            c.at = .send;
            return;
        }
        try r.tail(c);
        c.at = .end;
        if (!c.absorb) return;
        try r.hook(.lane, c.k, .front, r.x.e.w.layers.len);
        c.mopt = try lmtp.front(r.f, r.x, &c.st, &c.b, r.ids, c.start, c.R);
        c.moe = &r.x.e.w.mtp.?.layer;
        c.mtp = true;
        try r.route(c);
        c.sent += 1;
        c.at = .send;
    }

    /// A group that stops before its exchanges are done (an error in a step): the provider keeps the first incomplete
    /// window in the wire's order (lane order: it has made its first request whenever any lane has, unless nothing was
    /// sent), so the next begin dies (Node.fail(.stopped), python's die) and nothing more is sent (WIRE.md §6, §13).
    pub fn stopOrdinary(r: *Run) void {
        if (!r.begun) return;
        r.begun = false;
        for (r.ch[0..r.n]) |c| if (c.w.pos < c.w.count) {
            r.restore(c.w);
            if (c.w.pos == 0 and r.turn.next > 0) { // a later lane not yet on the wire while lane 0 is (count 1 only)
                r.x.a.dead = true;
                if (r.x.a.wire) |w| w.node.fail(.{ .why = .stopped, .peer = 0, .seq = w.node.doorbell() });
            }
            return;
        };
        r.restore(r.ch[r.n - 1].w);
    }

    /// TF_AFD_CHECK: each window hashed as a forward of its own (its seq, its ring), after the group's one sync.
    fn audited(r: *Run) !void {
        const x = r.x;
        if (x.check == null) return;
        const end = r.ch[r.n - 1].w;
        for (r.ch[0..r.n]) |c| {
            r.load(c.w);
            x.pending = x.a.ring != null;
            const done = x.finish(r.f);
            r.restore(end);
            try done;
        }
    }

    /// The lane's position (lane k > 0: the end of the lanes before it, on its own pos_dev; qualification_47b9bf3a: an absorbing lane's MTP
    /// length on its own mtp_pos_dev), its ids staged and its embedding: forward.compute's first launch.
    fn open(r: *Run, c: *Chunk) !void {
        const e = r.x.e;
        const m = e.m;
        if (c.k > 0) {
            try r.hook(.group, c.k, .position, 0);
            try cc.fill(r.f, T.whole(c.st.pos_dev), @intCast(c.st.pos));
            if (c.absorb) try lmtp.position(r.f, &c.st);
        }
        try r.hook(.lane, c.k, .stage, 0);
        const ids = r.ids[c.start..][0..c.R];
        c.R = try cc.stage(r.f, &c.st, &c.b, ids);
        c.opt = .{ .nch = cc.chunksFor(&c.st, c.R), .host_pos = c.st.pos, .chip = e.chip, .prompt = &e.prompt };
        try r.f.tri.embed(c.b.ids.ptr, cc.table(e.w.embed), c.b.x.ptr, c.R, m.hidden, m.streams);
    }

    /// The resumable layer loop: the lane's dense layers whole (prompt.layerForward), up to the next MoE layer's front,
    /// where it stops (true); false once every layer ran.
    fn advance(r: *Run, c: *Chunk) !bool {
        const e = r.x.e;
        c.moe = null;
        while (c.li < e.w.layers.len) {
            const l = &e.w.layers[c.li];
            c.li += 1;
            try r.hook(.lane, c.k, .front, c.li - 1);
            if (l.mlp != null) {
                try prompt.layerForward(r.f, l, &c.st, &c.b, c.R, c.opt);
                continue;
            }
            try pairs.front(r.f, l, &c.st, &c.b, c.R, c.opt);
            c.moe = l;
            return true;
        }
        return false;
    }

    fn route(r: *Run, c: *Chunk) !void {
        try r.hook(.lane, c.k, .route, c.w.pos);
        r.load(c.w);
        try r.x.a.route(r.f, c.moe.?, &c.b, c.R);
    }

    fn request(r: *Run, c: *Chunk) !void {
        try r.hook(.lane, c.k, .request, c.w.pos);
        try r.turn.request(c.k, c.w.pos);
        r.load(c.w);
        try r.x.a.request(r.f, c.moe.?, &c.b, c.R, if (c.mtp) c.mopt else c.opt); // qualification_47b9bf3a: the MTP layer's own options
        c.w.pos = r.x.a.pos;
    }

    pub fn take(r: *Run, c: *Chunk) !void {
        try r.hook(.lane, c.k, .take, c.w.pos - 1);
        try r.turn.take(c.k, c.w.pos - 1);
        r.load(c.w);
        c.planes = try r.x.a.take(r.f, &c.b, c.R);
    }

    /// layer_forward's last launch for a MoE layer: hc_post of the three planes into the lane's streams.
    fn post(r: *Run, c: *Chunk) !void {
        const m = r.x.e.m;
        try r.hook(.lane, c.k, .post, c.w.pos - 1);
        try r.f.tri.hcPost(c.b.x.ptr, c.b.x.ptr, c.planes.g, c.b.post.ptr, c.b.comb.ptr, c.planes.world, c.R, m.hidden, m.streams);
    }

    /// forward.compute's tail (the streams' mean, the final norm, the head on the last row) and prefill's two clones.
    fn tail(r: *Run, c: *Chunk) !void {
        const e = r.x.e;
        const m = e.m;
        const D = m.hidden;
        const b = &c.b;
        try r.hook(.lane, c.k, .tail, 0);
        try r.f.tri.streamMean(b.x.ptr, b.hidden.ptr, c.R, D, m.streams);
        try r.f.tri.rmsnorm(b.hidden.ptr, D, e.w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, m.eps, c.R, D);
        const out = layer.rowsOf(b.logits, 1);
        try qmm.mm(r.f, b.prefill, cc.rows(b.fnormed, c.R - 1, 1), e.w.head, cc.rows(b.fxs, c.R - 1, 1), out, false, b.sk, c.opt.chip);
        c.last = try cc.clone(r.f, &e.tmp, out, &e.last);
        _ = try cc.clone(r.f, &e.tmp, cc.rows(b.fnormed, c.R - 1, 1), &e.last_hidden);
    }
};
