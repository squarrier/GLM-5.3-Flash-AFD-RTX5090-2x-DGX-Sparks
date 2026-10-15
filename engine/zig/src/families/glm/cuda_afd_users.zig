//! concurrent_users independent States and layer-major decode on the existing CUDA/AFD stream, kernels and expert format.
//! Ash Hart authored TensorFold/MCDMA; Mia (@MiaAI_lab) authored EXL3 recipes/exactness; inherited kernel credits apply.
//! shared_forward (TF_AFD_USERS_SHARED=1): the shared round's device half (sharedOpen..sharedTail), after @Bizuayeu's shared
//! verify windows (github.com/Bizuayeu/TensorFold pr/glm53-flash-cuda 75131b66, Apache-2.0; reimplemented, not copied).
//! draft_shared (both switches): user_drafts's drafts beside shared_forward's shared rounds, each tracked row's backlog row copied at its
//! own tail or from its row of the shared head; draft_shared's edits: Claude Opus 5.5 (Anthropic; planning and execution), via
//! Hermes.
const std = @import("std");
const cuda = @import("cuda");
const bufs = @import("cuda_buffers.zig");
const views = @import("cuda_views.zig");
const cc = @import("cuda_compute.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const prompt = @import("cuda_prompt.zig");
const pairs = @import("cuda_afd_pairs.zig");
const sampling = @import("cuda_sample.zig");
const ae = @import("cuda_afd_engine.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;
const trace = @import("cuda_trace.zig");
const scheduling = @import("cuda_afd_users_round.zig");
const pooling = @import("cuda_afd_pool.zig");
const nucleus = @import("cuda_nucleus.zig");
const drafting = @import("cuda_afd_users_drafts.zig"); // user_drafts: a lone user's MTP drafts (TF_AFD_USERS_DRAFTS)
const sharing = @import("cuda_afd_users_shared.zig"); // shared_forward: one forward over every row (TF_AFD_USERS_SHARED)
const dsv = @import("cuda_afd_users_dsv.zig"); // shared_verify: drafted shared verify (TF_AFD_USERS_SHARED_DRAFTS)
const pp = @import("cuda_afd_users_pp.zig"); // ping_pong: independent micro-batches, default off
const dsa = @import("cuda_dsa.zig");
const Key = scheduling.Key;
const Row = scheduling.Row;
const Buf = views.Buf;
const T = trace.T;

fn rowBuffer(b: Buf, rows: usize) Buf {
    var shape = b.shape;
    shape[0] = rows;
    return Buf.of(b.dtype, shape[0..b.dims]);
}

/// Only fields which cross another user's attention or a combined exchange are private; other decode scratch is shared.
pub const Saved = struct {
    x: Buf,
    normed: Buf,
    xs: Buf,
    post: Buf,
    comb: Buf,
    logits: Buf,

    fn init(b: *const bufs.Buffers) Saved {
        return .{ .x = rowBuffer(b.x, 1), .normed = rowBuffer(b.normed, 1), .xs = rowBuffer(b.xs, 1), .post = rowBuffer(b.post, 1), .comb = rowBuffer(b.comb, 1), .logits = rowBuffer(b.logits, 1) };
    }

    fn view(s: Saved, base: bufs.Buffers) bufs.Buffers {
        var b = base;
        b.rows = 1;
        b.x = s.x;
        b.normed = s.normed;
        b.xs = s.xs;
        b.post = s.post;
        b.comb = s.comb;
        b.logits = s.logits;
        return b;
    }
};

pub const Bytes = struct { states: u64, rows: u64, total: u64 };
const Sizer = struct {
    used: u64 = 0,
    fn visit(s: *Sizer, _: []const u8, b: *Buf) anyerror!void {
        if (!b.host) s.used = std.mem.alignForward(u64, s.used + b.bytes(), 256);
    }
};

/// Extra allocation occurs ONLY on the opted-in path. Slot zero is the ORIGINAL Engine State; no Engine is cloned.
pub const Users = struct {
    at: *ae.Attn,
    f: *Fwd,
    ctx: ?*const cuda.Context = null,
    count: u8,
    states: [3]bufs.State = undefined,
    saved: [4]Saved = undefined,
    normed: Buf,
    xs: Buf,
    planes: Buf,
    generation: [4]u64 = @splat(0),
    dev: ?cuda.DeviceBuffer = null,
    prefills: u64 = 0,
    rounds: u64 = 0,
    exchanges: u64 = 0,
    observations: bool = false,
    observer: ?struct { ctx: *anyopaque, note: *const fn (*anyopaque, scheduling.Observation) void } = null,
    /// shared_pool (TF_AFD_POOL_TOKENS): every slot's capacity-sized caches from one pool, a region per live request; null:
    /// each slot's own State caches (concurrent_users).
    pool: ?*pooling.Pool = null,
    /// prefill_interleave: a slot's served prompt pass under way a step at a time (the next row and the prompt's length)
    fill: [4]?Fill = @splat(null),
    /// user_sampling (TF_AFD_USERS_SAMPLING=1): each live slot's request sampling (null: greedy), for its first token's draw;
    /// a round's row carries its own (Row.sampling). Never set with the switch off.
    sampling: [4]?nucleus.Sampling = @splat(null),
    /// user_drafts (TF_AFD_USERS_DRAFTS=1): the drafted loop's policy and each slot's carried MTP state; null: off.
    drafts: ?Drafts = null,
    /// shared_forward (TF_AFD_USERS_SHARED=1): the shared round's two stages; null: each row's own forward (concurrent_users's).
    stages: ?sharing.Stages = null,
    /// shared_forward: this round's forward was the shared one (its logits are b.logits' rows); set by sharedOpen, reset by begin.
    together: bool = false,
    /// shared_verify (TF_AFD_USERS_SHARED_DRAFTS=1): each slot's drafted-shared state (cuda_afd_users_dsv.zig); null: off.
    dsvs: ?dsv.Slots = null,
    /// ping_pong: B's stop-crossing copies and audit ring; null keeps the existing order and allocation.
    ping_pong: ?pp.State = null,
    bd: ?@import("cuda_afd_users_bd.zig").Stash = null, // batched_drafts: per-slot MTP hidden, only with the switch on

    pub const Fill = struct { start: usize, len: usize };
    pub const Drafts = struct { policy: drafting.Policy, backlog: [4]drafting.Backlog };
    /// user_drafts: the users backend's view of a drafted loop (native/glm_cuda_users.zig's lone driver)
    pub const DraftHook = drafting.Hook;
    pub const DraftRan = drafting.Ran;

    pub fn init(gpa: std.mem.Allocator, at: *ae.Attn, f: *Fwd, count: u8) !Users {
        if (count != 2 and count != 4) return error.BadAfdUsers;
        if (at.graphs != null or at.absorb or at.e.tap != null) return error.UsersModeUnsupported;
        if (at.e.buf.rows < count or at.a.rows < count) return error.Rows;
        var u: Users = .{ .at = at, .f = f, .count = count, .normed = rowBuffer(at.e.buf.normed, count), .xs = rowBuffer(at.e.buf.xs, count), .planes = Buf.of(.f32, &.{ 3, at.e.m.hidden }) };
        var made: usize = 0;
        errdefer for (u.states[0..made]) |*st| st.deinit(gpa);
        for (u.states[0 .. count - 1]) |*st| {
            st.* = try bufs.State.init(gpa, at.e.m, at.e.st.capacity, at.e.buf.rows, at.e.st.index != null);
            made += 1;
        }
        for (u.saved[0..count]) |*s| s.* = Saved.init(&at.e.buf);
        u.observations = if (std.c.getenv("TF_AFD_USERS_TRACE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        return u;
    }

    pub fn deinit(u: *Users, gpa: std.mem.Allocator) void {
        if (u.ping_pong) |*p| p.detach();
        if (u.dev) |*d| d.free();
        for (u.states[0 .. u.count - 1]) |*st| st.deinit(gpa);
        u.* = undefined;
    }

    pub fn walk(u: *Users, context: anytype, comptime visit: fn (@TypeOf(context), []const u8, *Buf) anyerror!void) !void {
        for (u.states[0 .. u.count - 1], 1..) |*st, i| {
            var name: [32]u8 = undefined;
            try views.walk(st, views.Path.init(try std.fmt.bufPrint(&name, "users.state{d}", .{i})), context, visit);
        }
        for (u.saved[0..u.count], 0..) |*s, i| {
            var name: [32]u8 = undefined;
            try views.walk(s, views.Path.init(try std.fmt.bufPrint(&name, "users.row{d}", .{i})), context, visit);
        }
        try visit(context, "users.normed", &u.normed);
        try visit(context, "users.xs", &u.xs);
        try visit(context, "users.planes", &u.planes);
        if (u.drafts) |*d| for (d.backlog[0..u.count], 0..) |*b, i| { // user_drafts: after every other buffer (unset: none)
            var name: [32]u8 = undefined;
            try visit(context, try std.fmt.bufPrint(&name, "users.backlog{d}", .{i}), &b.hidden);
        };
        if (u.stages) |*s| { // shared_forward: after every other buffer (unset: none)
            try visit(context, "users.shared_proj", &s.proj);
            try visit(context, "users.shared_out", &s.out);
        }
        if (u.ping_pong) |*p| try views.walk(&p.c, views.Path.init("users.ping"), context, visit);
        if (u.bd) |*b| try visit(context, "users.mtp_hidden", &b.hidden);
    }

    pub fn bytes(u: *Users) !Bytes {
        var st: Sizer = .{};
        for (u.states[0 .. u.count - 1]) |*s| try views.walk(s, views.Path.init("st"), &st, Sizer.visit);
        var all: Sizer = .{};
        try u.walk(&all, Sizer.visit);
        return .{ .states = st.used, .rows = all.used - st.used, .total = all.used };
    }

    pub fn attach(u: *Users, d: *const cuda.Driver) !void {
        if (u.dev != null) return error.Attached;
        var dev = try cuda.DeviceBuffer.alloc(d, @intCast((try u.bytes()).total));
        errdefer dev.free();
        try dev.fill8(0, null);
        var arena: bufs.Arena = .{ .base = dev.ptr };
        try u.walk(&arena, bufs.Arena.visit);
        if (u.ping_pong) |*p| if (u.at.a.ring != null) try p.audit(d, u.at.a);
        u.dev = dev;
    }

    fn bind(u: *Users) !void {
        if (u.ctx) |ctx| try ctx.makeCurrent();
    }

    /// shared_pool, before attach: the extra States keep their per-request buffers only; every slot's caches (slot 0's: the
    /// Engine's, detached before its attach) come from `p`'s regions.
    pub fn usePool(u: *Users, p: *pooling.Pool) !void {
        if (u.dev != null) return error.Attached;
        for (u.states[0 .. u.count - 1]) |*st| pooling.detachTokens(st);
        u.pool = p;
    }

    fn slotState(u: *Users, slot: u8) *bufs.State {
        return if (slot == 0) &u.at.e.st else &u.states[slot - 1];
    }

    /// user_drafts (TF_AFD_USERS_DRAFTS=1), before attach: the drafted loop's policy and a backlog of `cap` pairs a slot
    /// (bf16 rows in the users' block); the model's MTP head and its decode buffers must be loaded (TF_AFD_MTP=1).
    /// draft_shared: with shared_forward's stages too (either order): a lone user drafts, the rounds are the shared forward, each
    /// tracked slot's backlog grows from its own row of the shared head (sharedTail).
    pub fn useDrafts(u: *Users, policy: drafting.Policy, cap: usize) !void {
        if (u.dev != null) return error.Attached;
        if (u.at.e.mbuf == null or u.at.e.st.mtp_kc == null or !u.at.a.mtp) return error.NoMtp;
        if (cap < @import("cuda_decode.zig").max_rows) return error.BacklogRows; // a yielding round's kept rows fit
        var d: Drafts = .{ .policy = policy, .backlog = undefined };
        for (&d.backlog) |*b| b.* = drafting.Backlog.init(u.at.e.m.hidden, cap);
        u.drafts = d;
    }

    /// shared_forward (TF_AFD_USERS_SHARED=1), before attach: every round one forward over every row (sharedOpen..sharedTail),
    /// its two stages in the users' block after every other buffer. The rounds stay undrafted; draft_shared: with user_drafts's
    /// drafts (either order) a lone user drafts by its own loop between them.
    pub fn useShared(u: *Users) !void {
        if (u.dev != null) return error.Attached;
        u.stages = sharing.Stages.init(u.at.e.m, u.count);
    }

    /// shared_verify (TF_AFD_USERS_SHARED_DRAFTS=1), before attach and after useDrafts: every stream's verify window in one
    /// forward of at most dsv.max_rows rows (shared_forward's stages at those rows, in the users' block after every other
    /// buffer), each drafting stream's MTP drafts on its own State (cuda_afd_users_dsv.zig).
    pub fn useSharedDrafts(u: *Users) !void {
        if (u.dev != null) return error.Attached;
        if (u.drafts == null) return error.UsersDraftsOff;
        if (u.at.e.buf.rows < dsv.max_rows or u.at.a.rows < dsv.max_rows) return error.Rows;
        u.stages = sharing.Stages.init(u.at.e.m, dsv.max_rows);
        u.dsvs = .{};
    }

    /// batched_drafts: the existing MTP buffers, with a private hidden row per slot.
    pub fn useBatchedDrafts(u: *Users) !void {
        if (u.dev != null) return error.Attached;
        if (u.dsvs == null) return error.UsersSharedDraftsOff;
        if (u.at.e.mbuf == null or u.at.e.mbuf.?.rows < @import("cuda_afd_users_bd.zig").max_rows) return error.Rows;
        u.bd = @import("cuda_afd_users_bd.zig").Stash.init(u.at.e.m, u.count);
    }

    pub fn draftsBatched(u: *Users, requests: []const @import("cuda_afd_users_bd.zig").Request) !void {
        try u.bind();
        try u.at.alive();
        try @import("cuda_afd_users_bd.zig").run(u, requests);
    }

    /// shared_verify: one forward over every window's rows; out takes every row's draw (window by window). A one-row window's
    /// row is committed; a wider window's rows wait for settleDrafted (the stream's draft request or keep).
    pub fn usePingPong(u: *Users) !void {
        if (u.dev != null) return error.Attached;
        if (u.dsvs == null) return error.PingPongNeedsSharedDrafts;
        _ = try pp.parse("1", true, u.at.a.paired, u.at.a.inflight, u.at.a.lanes);
        if (u.at.structured_control != null) return error.PingPongControlUnsupported;
        u.ping_pong = .{ .c = try pp.Copies.init(&u.at.e.buf, u.at.a) };
    }

    pub fn roundDrafted(u: *Users, wins: []const dsv.Window, out: []u32) !void {
        try u.bind();
        try u.at.alive();
        if (u.ping_pong != null) return @import("cuda_afd_users_pp_device.zig").round(u, wins, out);
        try dsv.round(u, wins, out);
    }

    /// shared_verify: `keep` of the last window's `rows` stay on `key`'s State (the rest rolled back).
    pub fn settleDrafted(u: *Users, key: Key, rows: usize, keep: usize) !void {
        try u.bind();
        try u.at.alive();
        try dsv.settle(u, key, rows, keep);
    }

    /// shared_verify: `key`'s MTP head absorbs its unabsorbed pairs (with `verified`, the last window's kept rows and `follow`)
    /// and chains out.len drafts at `at_position` into `out` (cuda_afd_users_dsv.draft).
    pub fn draftDrafted(u: *Users, key: Key, follow: []const u32, verified: bool, at_position: usize, out: []u32) !void {
        try u.bind();
        try u.at.alive();
        try dsv.draft(u, key, follow, verified, at_position, out);
    }

    /// user_drafts: how `slot`'s next prompt pass treats its MTP head (drafting.Track); nothing without drafts.
    pub fn track(u: *Users, slot: u8, how: drafting.Track) void {
        const d = if (u.drafts) |*x| x else return;
        if (slot >= u.count) return;
        d.backlog[slot].clear();
        d.backlog[slot].track = how;
    }

    /// user_drafts: `key`'s slot carries its MTP state to its pending token (a resume can start from it).
    pub fn carried(u: *Users, key: Key) bool {
        const d = if (u.drafts) |*x| x else return false;
        if (key.slot >= u.count or u.generation[key.slot] != key.generation) return false;
        return d.backlog[key.slot].carried();
    }

    /// user_drafts: the pairs `key`'s backlog holds now (tests and the lab's counters).
    pub fn backlogLen(u: *Users, key: Key) usize {
        const d = if (u.drafts) |*x| x else return 0;
        return if (key.slot < u.count) d.backlog[key.slot].n else 0;
    }

    /// user_drafts: the drafted loop for `key` on its own State (swapped into the Engine for the loop, as prefillAs does for
    /// a prompt pass), from its start pair; `hook` hears each round's kept tokens and may end it (and `carry`).
    pub fn drafted(u: *Users, key: Key, pending: u32, count: usize, out: []u32, hook: *const drafting.Hook) !drafting.Ran {
        const d = if (u.drafts) |*x| x else return error.UsersDraftsOff;
        try u.bind();
        try u.at.alive();
        if (key.slot >= u.count or u.generation[key.slot] != key.generation) return error.StaleGeneration;
        if (u.fill[key.slot] != null) return error.FillingDrafted;
        const e = u.at.e;
        if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
        defer if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
        errdefer d.backlog[key.slot].clear();
        try dsv.unabsorb(u, key.slot); // shared_verify: a start pair a drafted shared round absorbed back out first (off: none)
        try drafting.start(u.at, u.f, &d.backlog[key.slot], pending);
        return drafting.run(u.at, u.f, d.policy, pending, count, out, hook);
    }

    /// user_drafts, inside `drafted`'s hook: the round's kept rows and tokens become `key`'s backlog (the loop yields).
    pub fn carry(u: *Users, key: Key, kept: []const u32) !void {
        const d = if (u.drafts) |*x| x else return error.UsersDraftsOff;
        if (key.slot >= u.count or u.generation[key.slot] != key.generation) return error.StaleGeneration;
        try drafting.carry(u.at, u.f, &d.backlog[key.slot], kept);
    }

    /// user_drafts: before a shared round, a slot whose backlog is full absorbs all its pairs but the last (its own State).
    fn flushFull(u: *Users, rows: []const Row) !void {
        const d = if (u.drafts) |*x| x else return;
        for (rows) |row| {
            const b = &d.backlog[row.key.slot];
            if (b.track == .off or b.n < b.capacity()) continue;
            const e = u.at.e;
            if (row.key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[row.key.slot - 1]);
            defer if (row.key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[row.key.slot - 1]);
            try drafting.absorbHead(u.at, u.f, b);
        }
    }

    /// user_drafts: the backlog a shared round grows for this row's slot (a tracked slot whose pairs end at its pending).
    fn grown(u: *Users, row: Row) ?*drafting.Backlog {
        const d = if (u.drafts) |*x| x else return null;
        const b = &d.backlog[row.key.slot];
        if (b.track == .off) return null;
        if (b.head) { // a lone opening that never drafted cannot carry: untracked from here, never resumed (never stale)
            b.clear();
            return null;
        }
        return b;
    }

    /// shared_pool: the region of the request about to prefill in `slot` (its prompt, its reply, one row), first fit or after
    /// compaction (the moved regions' rows copied on the forward's stream, their views re-pointed), before the slot
    /// changes owner; nothing without a pool. A full pool (PoolFull) or a request past it (ContextFull) changes nothing.
    pub fn reserve(u: *Users, slot: u8, prompt_len: usize, max_new: usize) !void {
        const p = u.pool orelse return;
        if (slot >= u.count) return error.SlotOutOfStep;
        try u.at.alive(); // a failed attention side moves nothing
        var moves: [pooling.max_slots]pooling.Move = undefined;
        const got = try p.reserve(slot, pooling.need(prompt_len, max_new), &moves);
        if (got.moved > 0) {
            try u.bind();
            for (moves[0..got.moved]) |mv| {
                try p.copyMove(u.f, mv);
                p.view(p.regions[mv.slot].?, u.slotState(mv.slot));
            }
        }
        p.view(got.region, u.slotState(slot));
    }

    fn state(u: *Users, key: Key) *bufs.State {
        std.debug.assert(key.slot < u.count and u.generation[key.slot] == key.generation);
        return if (key.slot == 0) &u.at.e.st else &u.states[key.slot - 1];
    }

    pub fn position(u: *Users, key: Key) usize {
        return u.state(key).pos;
    }

    /// Existing PAIRS/LANES remain within one user's serial prefill: their Run builds State views from e.st each time.
    pub fn prefill(u: *Users, key: Key, ids: []const u32) !u32 {
        return u.prefillAs(key, ids, false);
    }

    /// structured_users: a structured user's prompt pass with its first draw deferred (Attn.prefillToLogits): the last row's
    /// logits wait for firstMasked, which draws among its grammar's tokens.
    pub fn prefillDeferred(u: *Users, key: Key, ids: []const u32) !void {
        _ = try u.prefillAs(key, ids, true);
    }

    fn prefillAs(u: *Users, key: Key, ids: []const u32, deferred: bool) !u32 {
        try u.bind();
        try u.at.alive();
        if (key.slot >= u.count or key.generation <= u.generation[key.slot]) return error.StaleGeneration;
        u.generation[key.slot] = key.generation;
        const row: Row = .{ .key = key, .token = 0, .position = 0 };
        u.observeRound(.prefill_before, row, 0);
        const token = blk: {
            const e = u.at.e;
            if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
            defer if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
            const restore = u.useSampling(key.slot); // user_sampling: a sampled request's first draw (Attn.firstToken) is its own
            defer e.sampling = restore;
            u.prefills += 1;
            if (deferred) {
                try u.at.prefillToLogits(u.f, ids);
                break :blk 0;
            }
            const how = u.tracked(key.slot); // user_drafts: a slot that may draft absorbs its prompt (qualification_21e21bc0's prefill)
            if (how == .off) break :blk try u.at.prefill(u.f, ids);
            u.at.absorb = true;
            defer u.at.absorb = false;
            const first = try u.at.prefill(u.f, ids);
            try u.seeded(key.slot, first);
            break :blk first;
        };
        u.observeRound(.prefill_after, row, 0);
        return token;
    }

    /// user_drafts: how `slot`'s prompt pass treats the MTP head (off without drafts).
    fn tracked(u: *Users, slot: u8) drafting.Track {
        const d = if (u.drafts) |*x| x else return .off;
        return d.backlog[slot].track;
    }

    /// user_drafts: a tracked prompt pass ended with `first`: a `seed` slot's first pair from the Engine's last_hidden; a
    /// `lone` slot's stays there for the drafted loop that follows at once (the single-user path's calls).
    fn seeded(u: *Users, slot: u8, first: u32) !void {
        const b = &u.drafts.?.backlog[slot];
        if (b.track == .lone) {
            b.n = 0;
            b.head = true;
            return;
        }
        try drafting.seed(u.at, u.f, b, first);
    }

    /// structured_users: the deferred first token among `mask`'s tokens (empty: the plain draw), on the slot's own State.
    pub fn firstMasked(u: *Users, key: Key, mask: []const u32) !u32 {
        try u.bind();
        try u.at.alive();
        if (key.slot >= u.count or u.generation[key.slot] != key.generation) return error.StaleGeneration;
        const e = u.at.e;
        if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
        defer if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
        const restore = u.useSampling(key.slot); // user_sampling: Attn.select's masked draw at the slot's own position
        defer e.sampling = restore;
        return u.at.pickFirst(u.f, mask);
    }

    /// user_sampling (TF_AFD_USERS_SAMPLING=1): `slot`'s request sampling for its first token's draw, set by the native Users
    /// at the claim (null: greedy); `release` clears it.
    pub fn configureSampling(u: *Users, slot: u8, s: ?nucleus.Sampling) void {
        if (slot < u.count) u.sampling[slot] = s;
    }

    /// user_sampling: the Engine's sampling for one call on `slot`'s behalf — its request's, as the single-user entry's
    /// configureSampling sets it for a request — returning the previous value (null in users mode) for the caller to
    /// restore. A greedy slot (null) leaves the Engine as it is: today's greedy draw.
    fn useSampling(u: *Users, slot: u8) ?nucleus.Sampling {
        const saved = u.at.e.sampling;
        if (slot < u.count) if (u.sampling[slot]) |s| {
            u.at.e.sampling = s;
        };
        return saved;
    }

    /// structured_users: the masked adapter is installed on this sampler (TF_AFD_GRAMMAR_MASK=1).
    pub fn checkMasked(u: *Users) !void {
        try sampling.maskedReady(u.f, &u.at.e.smp);
    }

    pub fn maskedWidth(u: *const Users) usize {
        return u.at.e.m.vocab;
    }

    pub fn maskedWords(u: *const Users) usize {
        return (u.at.e.m.vocab + 31) / 32;
    }

    /// structured_users: a row's mask refused before the round's forward when malformed or empty (no launch), as stepMasked's.
    pub fn maskCheck(u: *Users, mask: []const u32) !void {
        try sampling.mask_cpu.shape(u.at.e.m.vocab, mask);
        try sampling.maskedReady(u.f, &u.at.e.smp);
        try sampling.maskedPrecheck(&u.at.e.smp, mask);
    }

    /// prefill_interleave: one step of the served prompt pass for `key` (the first claims the generation and resets the State, as
    /// `prefill`); the first token with the last step, else null. The slot's State is swapped in for the step only.
    pub fn prefillStep(u: *Users, key: Key, ids: []const u32) !?u32 {
        try u.bind();
        try u.at.alive();
        if (key.slot >= u.count) return error.StaleGeneration;
        if (u.at.structured_control != null) return error.UsersModeUnsupported; // the users serve never installs one
        const row: Row = .{ .key = key, .token = 0, .position = 0 };
        const first = u.fill[key.slot] == null;
        if (first) {
            if (key.generation <= u.generation[key.slot]) return error.StaleGeneration;
            u.generation[key.slot] = key.generation;
            u.observeRound(.prefill_before, row, 0);
            u.fill[key.slot] = .{ .start = 0, .len = ids.len };
        } else if (key.generation != u.generation[key.slot] or ids.len != u.fill[key.slot].?.len) return error.StaleGeneration;
        const token = blk: {
            const e = u.at.e;
            if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
            defer if (key.slot != 0) std.mem.swap(bufs.State, &e.st, &u.states[key.slot - 1]);
            const restore = u.useSampling(key.slot); // user_sampling: the last step's first draw is the request's own
            defer e.sampling = restore;
            if (first) u.prefills += 1;
            const how = u.tracked(key.slot); // user_drafts: each step of a tracked slot's pass absorbs, seeded at the last
            if (how == .off) break :blk try u.served(ids, &u.fill[key.slot].?.start, first);
            u.at.absorb = true;
            defer u.at.absorb = false;
            const got = try u.served(ids, &u.fill[key.slot].?.start, first);
            if (got) |t| try u.seeded(key.slot, t);
            break :blk got;
        };
        const t = token orelse return null;
        u.fill[key.slot] = null;
        u.observeRound(.prefill_after, row, 0);
        return t;
    }

    /// prefill_interleave: the next step of the path `Attn.prefill` takes (qualification_9e49592e lanes, qualification_0f991257 pairs or plain chunks), from `start`.
    fn served(u: *Users, ids: []const u32, start: *usize, first: bool) !?u32 {
        const x = u.at;
        if (x.lanes) |l| if (ids.len > x.e.prefill_rows and x.e.tap == null) return l.step(x, u.f, ids, start, first);
        if (x.pairs) |p| if (ids.len > x.e.prefill_rows and x.e.tap == null) return p.step(x, u.f, ids, start, first);
        return x.promptStep(u.f, ids, start, first);
    }

    /// Scalar snapshots only; state is the physical State slot, never a retained borrowed pointer.
    pub fn observeRound(u: *Users, event: @FieldType(scheduling.Observation, "event"), row: Row, commits: usize) void {
        if (!u.observations) return;
        const st = u.state(row.key);
        var state_id: u8 = 255;
        if (st == &u.at.e.st) {
            state_id = 0;
        } else {
            for (u.states[0 .. u.count - 1], 1..) |*candidate, i| {
                if (st == candidate) state_id = @intCast(i);
            }
        }
        const value: scheduling.Observation = .{ .event = event, .slot = row.key.slot, .generation = row.key.generation, .state = state_id, .position = st.pos, .parity = st.parity, .round = u.rounds, .commits = commits };
        if (u.observer) |o| o.note(o.ctx, value) else std.log.info("concurrent_users OWN event={s} slot={d} generation={d} state={d} position={d} parity={d} round={d} commits={d}", .{ @tagName(value.event), value.slot, value.generation, value.state, value.position, value.parity, value.round, value.commits });
    }

    pub fn observeRelease(u: *Users, key: Key, reason: []const u8) void {
        const event: @FieldType(scheduling.Observation, "event") = if (std.mem.eql(u8, reason, "cancelled")) .cancel_retired else if (std.mem.eql(u8, reason, "error")) .failure_retired else return;
        u.observeRound(event, .{ .key = key, .token = 0, .position = 0 }, 0);
    }

    pub fn release(u: *Users, key: Key) void {
        u.observeRound(.retire, .{ .key = key, .token = 0, .position = 0 }, 0);
        u.at.deferred_logits = null; // structured_users: a structured owner's deferred first-token logits never outlive it
        if (key.slot < u.count) u.fill[key.slot] = null; // prefill_interleave: a pass under way ends with its owner
        if (key.slot < u.count) u.sampling[key.slot] = null; // user_sampling: a request's sampling never outlives it
        if (u.pool) |p| _ = p.release(key.slot); // shared_pool: exactly its region's blocks back to the pool
        if (u.drafts) |*d| if (key.slot < u.count) d.backlog[key.slot].clear(); // user_drafts: its carried MTP state too
        if (u.dsvs) |*v| if (key.slot < u.count) { // shared_verify: its in-head start pair and copied rows too
            v.slot[key.slot] = .{};
        };
    } // State is reset only on the next generation's prefill, never another user

    pub fn round(u: *Users, rows: []const Row, out: []u32) !void {
        try u.bind();
        try u.at.alive();
        var seen: [4]bool = @splat(false);
        for (rows) |row| {
            if (row.key.slot >= u.count or u.generation[row.key.slot] != row.key.generation or seen[row.key.slot]) return error.StaleGeneration;
            seen[row.key.slot] = true;
            if (u.position(row.key) != row.position) return error.PositionMismatch;
            try cc.checkRoom(u.at.e.m, u.state(row.key), 1, null);
        }
        errdefer u.at.a.dead = true; // interrupted header/exchange windows are not safe to reuse
        try scheduling.run(Users, u, rows, out);
        u.rounds += 1;
        try u.flushFull(rows); // user_drafts: a full backlog absorbed (its own MTP steps) before the next round adds a row
    }

    pub fn alive(u: *Users) !void {
        try u.at.alive();
    }

    pub fn begin(u: *Users, rows: []const Row) !void {
        u.together = false; // shared_forward: sharedOpen sets it for a shared round
        try u.at.finish(u.f);
        u.f.moe = u.at.a.provider();
        try u.at.a.begin(rows.len, .eager, if (u.f.gpu) |g| g.stream else null);
        u.at.forwards += 1;
        std.log.info("concurrent_users decode rows={d} users={d} round={d}", .{ rows.len, rows.len, u.rounds });
        for (rows, 0..) |row, i| std.log.info("concurrent_users row={d} slot={d} generation={d} position={d} parity={d}", .{ i, row.key.slot, row.key.generation, row.position, u.state(row.key).parity });
    }

    fn view(u: *const Users, row: Row) bufs.Buffers {
        return u.saved[row.key.slot].view(u.at.e.buf);
    }

    fn opt(u: *Users, row: Row) layer.Opt {
        const st = u.state(row.key);
        return .{ .nch = cc.chunksFor(st, 1), .host_pos = st.pos, .chip = u.at.e.chip };
    }

    pub fn open(u: *Users, row: Row, _: usize) !void {
        const e = u.at.e;
        const b = u.view(row);
        _ = try cc.stage(u.f, u.state(row.key), &b, &.{row.token});
        try u.f.tri.embed(b.ids.ptr, cc.table(e.w.embed), b.x.ptr, 1, e.m.hidden, e.m.streams);
    }

    pub fn layerCount(u: *const Users) usize {
        return u.at.e.w.layers.len;
    }

    pub fn isMoe(u: *const Users, li: usize) bool {
        return u.at.e.w.layers[li].mlp == null;
    }

    pub fn front(u: *Users, row: Row, _: usize, li: usize) !void {
        const b = u.view(row);
        const l = &u.at.e.w.layers[li];
        if (u.isMoe(li)) try pairs.front(u.f, l, u.state(row.key), &b, 1, u.opt(row)) else try prompt.layerForward(u.f, l, u.state(row.key), &b, 1, u.opt(row));
    }

    pub fn gather(u: *Users, row: Row, i: usize) !void {
        const b = u.saved[row.key.slot];
        try cc.copyDtoD(u.f, cc.rows(u.normed, i, 1), cc.rows(b.normed, 0, 1));
        try cc.copyDtoD(u.f, cc.rows(u.xs, i, 1), cc.rows(b.xs, 0, 1));
    }

    pub fn exchange(u: *Users, li: usize, n: usize) !void {
        var b = u.at.e.buf; // shared R<=4 routing/shared-expert scratch; no user attention ever runs at R=C
        b.normed = u.normed;
        b.xs = u.xs;
        const l = &u.at.e.w.layers[li];
        try u.at.a.route(u.f, l, &b, n);
        try u.at.a.request(u.f, l, &b, n, .{ .chip = u.at.e.chip });
        _ = try u.at.a.take(u.f, &b, n);
        u.exchanges += 1;
        std.log.info("concurrent_users exchange round={d} layer={d} rows={d}", .{ u.rounds, li, n });
    }

    /// Original expert0, expert1, shared planes; gather one row of EACH plane before the original R1 hcPost.
    pub fn scatter(u: *Users, row: Row, i: usize, n: usize) !void {
        const e = u.at.e;
        const b = u.view(row);
        for (0..3) |plane| {
            const src = T.of(u.at.a.afd_recv, 4 * e.m.hidden * (plane * n + i), &.{ 1, e.m.hidden });
            try cc.copyDtoD(u.f, T.of(u.planes, 4 * e.m.hidden * plane, &.{ 1, e.m.hidden }), src);
        }
        try u.f.tri.hcPost(b.x.ptr, b.x.ptr, u.planes.ptr, b.post.ptr, b.comb.ptr, 3, 1, e.m.hidden, e.m.streams);
    }

    pub fn tail(u: *Users, row: Row, _: usize) !void {
        const e = u.at.e;
        const b = u.view(row);
        const D = e.m.hidden;
        try u.f.tri.streamMean(b.x.ptr, b.hidden.ptr, 1, D, e.m.streams);
        try u.f.tri.rmsnorm(b.hidden.ptr, D, e.w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, e.m.eps, 1, D);
        try qmm.mm(u.f, false, cc.rows(b.fnormed, 0, 1), e.w.head, cc.rows(b.fxs, 0, 1), cc.rows(b.logits, 0, 1), false, b.sk, e.chip);
        // user_drafts (draft_shared's fix): the row's final-normed hidden into its slot's backlog now, on the forward's stream,
        // before the next row's tail reuses the shared fnormed scratch (user_drafts DESIGN.md's rule); counted at its commit
        if (u.grown(row)) |k| try drafting.grow(u.f, k, b.fnormed);
    }

    pub fn settle(u: *Users) !void {
        try ae.Attn.settle(u.f, u.at);
        u.at.launched();
        try u.at.finish(u.f);
    }

    pub fn sample(u: *Users, row: Row, i: usize) !u32 {
        var token: [1]u32 = .{0};
        const logits = u.logitsOf(row, i); // shared_forward: a shared round's row i of b.logits; else the slot's own (as before)
        // user_sampling (TF_AFD_USERS_SAMPLING=1): a sampled row draws exactly as the single-user step does (Attn.step,
        // stepMasked): the Engine's sampling set to its own request's for this one draw, qualification_45c10bcf's keyed draw at its drawn
        // token's position (row.position + 1), after its own grammar mask when structured; never keyed by slot or round
        if (row.sampling) |s| if (s.temperature > 0) {
            const e = u.at.e;
            const restore = e.sampling;
            e.sampling = s;
            defer e.sampling = restore;
            if (row.mask.len != 0) try e.sampleMasked(u.f, logits, row.position + 1, &token, row.mask) else try e.sampleRows(u.f, logits, row.position + 1, &token, null);
            return token[0];
        };
        // structured_users: a structured row draws among its own grammar's tokens (the masked adapter); a plain row as before
        if (row.mask.len != 0) {
            try sampling.greedyMasked(u.f, &u.at.e.smp, logits, &token, row.mask);
            return token[0];
        }
        try sampling.greedy(u.f, &u.at.e.smp, logits, &token);
        if (u.grown(row)) |b| drafting.drawn(b, token[0]); // user_drafts: the pair's token (counted at its commit)
        return token[0];
    }

    /// The logits row `row` draws from: a shared round's row `i` of the Engine's decode logits (shared_forward), else the
    /// slot's own saved row (each row's own forward).
    fn logitsOf(u: *Users, row: Row, i: usize) T {
        return if (u.together) cc.rows(u.at.e.buf.logits, i, 1) else cc.rows(u.saved[row.key.slot].logits, 0, 1);
    }

    pub fn commit(u: *Users, row: Row, _: usize) !void {
        const b = u.view(row);
        u.observeRound(.commit_before, row, 0);
        try cc.commit(u.f, u.state(row.key), &b, 1, 1);
        // user_drafts: the row's pair (its hidden copied at its tail or the shared head's, its token at the draw) counted
        if (u.grown(row)) |k| drafting.committed(k);
        u.observeRound(.commit_after, row, 1);
    }

    // -- shared_forward (TF_AFD_USERS_SHARED=1): the shared round's device half (cuda_afd_users_round.forwardShared) --------

    /// shared_forward: every round of this backend is one forward over every row (useShared ran).
    pub fn shared(u: *const Users) bool {
        return u.stages != null;
    }

    /// shared_forward: the round's ids staged at once (forward.stage's copies) and embedded into the decode Buffers' rows.
    pub fn sharedOpen(u: *Users, rows: []const Row) !void {
        const e = u.at.e;
        const b = &e.buf;
        if (rows.len == 0 or rows.len > 4 or rows.len > b.rows) return error.Rows;
        var ids: [4]u32 = undefined;
        for (rows, 0..) |row, i| ids[i] = row.token;
        u.together = true;
        try cc.stageIds(u.f, b, ids[0..rows.len]);
        try cc.stagedRecord(u.f, b);
        try u.f.tri.embed(b.ids.ptr, cc.table(e.w.embed), b.x.ptr, rows.len, e.m.hidden, e.m.streams);
    }

    /// shared_forward: layer `li`'s attention hc_pre over every row, then its block's row-by-row half (the KDA projections, or
    /// the DSA projections, norms, absorb and the indexer's projections its rows need).
    pub fn sharedIn(u: *Users, rows: []const Row, li: usize) !void {
        const e = u.at.e;
        const m = e.m;
        const b = &e.buf;
        const l = &e.w.layers[li];
        const n = rows.len;
        const h = l.attn_hc orelse return error.PlainLayer;
        try u.f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.in_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, n, m.hidden, m.streams);
        switch (l.kind) {
            .kda => try sharing.kdaIn(u.f, l, b, u.stages.?, n, e.chip),
            .dsa => {
                var need: ?sharing.Need = null;
                var sparse = false;
                for (rows, 0..) |row, i| {
                    const v = sharing.rowView(b.*, i);
                    const x = try sharing.dsaNeed(m, l, u.state(row.key), &v, u.opt(row));
                    if (need) |first| if (first.indexed != x.indexed) return error.UsersMixedIndex;
                    need = x;
                    sparse = sparse or x.sparse;
                }
                try sharing.dsaIn(u.f, l, b, n, need.?.indexed, sparse, e.chip);
            },
        }
    }

    /// shared_forward: row `i`'s per-sequence half of layer `li` on its own State at its own position (R = 1): its KDA chain,
    /// or its latent write, indexer update and attention.
    pub fn sharedSeq(u: *Users, row: Row, i: usize, li: usize) !void {
        const e = u.at.e;
        const l = &e.w.layers[li];
        const st = u.state(row.key);
        switch (l.kind) {
            .kda => try sharing.kdaSeq(u.f, l, st, &e.buf, u.stages.?, i),
            .dsa => {
                const v = sharing.rowView(e.buf, i);
                try sharing.dsaSeq(u.f, l, st, &v, u.opt(row), dsa.scratchOf(u.f));
            },
        }
    }

    /// shared_forward: layer `li`'s block out-projection and attention hc_post over every row, then its FFN half over every
    /// row: the dense MLP, or the MoE's hc_pre, one R = C route + request + take (the router and shared expert here, the
    /// routed experts on both halves) and hc_post over its three planes.
    pub fn sharedOut(u: *Users, rows: []const Row, li: usize) !void {
        const e = u.at.e;
        const m = e.m;
        const b = &e.buf;
        const l = &e.w.layers[li];
        const n = rows.len;
        const g = switch (l.kind) {
            .kda => try sharing.kdaOut(u.f, l, b, u.stages.?, n, e.chip),
            .dsa => try sharing.dsaOut(u.f, l, b, n, e.chip),
        };
        try u.f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, n, m.hidden, m.streams);
        try layer.ffnHalf(u.f, l, b, n, .{ .chip = e.chip }); // the MoE's block: f.moe, the Afd's provider (begin)
        if (u.isMoe(li)) {
            u.exchanges += 1;
            std.log.info("concurrent_users exchange round={d} layer={d} rows={d}", .{ u.rounds, li, n });
        }
    }

    /// shared_forward: the streams' mean, the final norm and the head over every row into the decode logits' rows.
    /// draft_shared (with user_drafts's drafts): each tracked row's final-normed hidden (row i of the head's) into its own slot's
    /// backlog, on the forward's stream before anything reuses fnormed; its token at the draw, counted at its commit.
    pub fn sharedTail(u: *Users, rows: []const Row) !void {
        const e = u.at.e;
        const b = &e.buf;
        const n = rows.len;
        const D = e.m.hidden;
        try u.f.tri.streamMean(b.x.ptr, b.hidden.ptr, n, D, e.m.streams);
        try u.f.tri.rmsnorm(b.hidden.ptr, D, e.w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, e.m.eps, n, D);
        try qmm.mm(u.f, false, cc.rows(b.fnormed, 0, n), e.w.head, cc.rows(b.fxs, 0, n), cc.rows(b.logits, 0, n), false, b.sk, e.chip);
        for (rows, 0..) |row, i| if (u.grown(row)) |k| try drafting.grow(u.f, k, sharing.rowOf(b.fnormed, i));
    }
};

test {
    _ = &Users.attach;
    _ = &Users.prefill;
    _ = &Users.prefillStep;
    _ = &Users.round;
    _ = &Users.configureSampling;
}
