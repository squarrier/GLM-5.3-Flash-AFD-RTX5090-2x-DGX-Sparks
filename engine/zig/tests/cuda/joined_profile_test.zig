//! joined_profile (local integration): the final head's combined identity CPU tests (step test-joined_profile, filter "joined_profile"): four_user_profile's
//! shared pool (TF_AFD_POOL_TOKENS) and interleaved prompts (TF_AFD_USERS_INTERLEAVE), structured_users's structured output
//! (TF_AFD_GRAMMAR_SERVE), user_sampling's sampled users (TF_AFD_USERS_SAMPLING=1: each row its own request's keyed draw) and
//! user_drafts's lone drafts (TF_AFD_USERS_DRAFTS=1, mode A) together at TF_AFD_USERS=4, through the real LaneHost, lane
//! core and users backend (native/glm_cuda_users.zig) with its lone driver. The Steps stand-in joins four_user_profile's (the real
//! pool's accounting, stepped passes, a FAKE schema script whose draws take the lowest id its mask allows), user_drafts's
//! (an MTP head that refuses a stale or out-of-step carried state; drafted rounds of mtpWith's shape) and user_sampling's (a
//! slot's sampling set at its claim; a sampled row draws lanes.fake.next keyed by its own request's seed at its drawn
//! token's position, never by slot, row or round). It refuses a drafted loop or a tracked slot for a sampled stream
//! and a row whose sampling is not its slot's. References: a greedy reply is lanes.fake.next of its history (the
//! undrafted single-user reply); a seeded sampled reply is the same keyed draw of its history (boot S's, where qualification_45c10bcf's
//! keyed draw keeps sampled drafted = sampled serial). No CUDA numerics; hardware equality is qualification_5ff9a097's.
//! Ash Hart authored TensorFold and its lane core (the lone driver and its yield hand-over); four_user_profile, shared_pool, prefill_interleave,
//! structured_users, user_drafts and user_sampling are Claude Opus 5.5 via Hermes, as their files say; this file: Claude Opus 5.5
//! (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const glm = @import("glm_cuda");
const users = @import("glm_users");
const P = glm.afd_pool;
const scheduling = glm.afd_users_round;
const Key = scheduling.Key;
const Row = scheduling.Row;
const t = std.testing;
const gpa = t.allocator;
const io = t.io;
const eq = t.expectEqual;

/// The served capacity at the 262,144 window (concurrent_users's), the backend's ContextFull bound.
const cap: usize = 262208;
/// The stand-in's mask width (a schema row's draws); lanes.fake.next's plain draws stay below 97.
const width: usize = 2048;
const words = (width + 31) / 32;

/// FAKE matcher (structured_users's Script, four_user_profile's copy): step i allows `steps[i]`; terminated once every step is accepted.
const Script = struct {
    rules: lanes.grammar.Rules = undefined,
    steps: []const []const u32,
    accepted: usize = 0,
    closes: usize = 0,
    fn init(x: *Script, w: usize) void {
        x.rules = .{ .ptr = x, .width = w, .vtable = &.{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close } };
    }
    fn of(p: *anyopaque) *Script {
        return @ptrCast(@alignCast(p));
    }
    fn fill(p: *anyopaque, out: []u32) !void {
        const x = of(p);
        @memset(out, 0);
        if (x.accepted < x.steps.len) for (x.steps[x.accepted]) |tok| {
            out[tok / 32] |= @as(u32, 1) << @as(u5, @intCast(tok % 32));
        };
    }
    fn accept(p: *anyopaque, token: u32) !bool {
        const x = of(p);
        if (x.accepted >= x.steps.len or std.mem.indexOfScalar(u32, x.steps[x.accepted], token) == null) return false;
        x.accepted += 1;
        return true;
    }
    fn rollback(p: *anyopaque, n: usize) !void {
        const x = of(p);
        if (n > x.accepted) return error.BadRollback;
        x.accepted -= n;
    }
    fn terminated(p: *anyopaque) bool {
        const x = of(p);
        return x.accepted >= x.steps.len;
    }
    fn close(p: *anyopaque) void {
        of(p).closes += 1;
    }
};

/// The lowest id a mask allows (the stand-in's masked draw, as structured_users's recorded adapter's emulator).
fn lowest(mask: []const u32) u32 {
    for (mask, 0..) |w, i| if (w != 0) return @intCast(i * 32 + @ctz(w));
    return 0;
}

/// Holds the engine thread inside its next drafted round (or shared round) once armed, until opened.
const Gate = struct {
    armed: std.atomic.Value(bool) = .init(false),
    entered: std.atomic.Value(bool) = .init(false),
    open: std.atomic.Value(bool) = .init(false),
    fn at(g: *Gate) void {
        if (!g.armed.load(.acquire)) return;
        g.armed.store(false, .release);
        g.entered.store(true, .release);
        for (0..10_000) |_| {
            if (g.open.load(.acquire)) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
    fn waitEntered(g: *Gate) !void {
        for (0..10_000) |_| {
            if (g.entered.load(.acquire)) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.WaitTimeout;
    }
};

const Track = enum { off, seed, lone };

/// A slot: its generation, its cache (history), a pass under way (prefill_interleave), its MTP head (user_drafts's stand-in: the
/// rows its cache absorbed, the pairs it owes ending at `last`, `head` for a lone opening) and its request's sampling
/// (user_sampling: set at the claim, cleared at release).
const Slot = struct {
    generation: u64 = 0,
    history: std.ArrayList(u32) = .empty,
    step: ?usize = null,
    track: Track = .off,
    mtp: usize = 0,
    owed: usize = 0,
    last: u32 = 0,
    head: bool = false,
    sampling: ?scheduling.Sampling = null,
};

/// The draw after `history` (its length the drawn token's position): greedy, or keyed by the request's seed.
fn draw(history: []const u32, s: ?scheduling.Sampling) u32 {
    const keyed: ?lanes.Sampling = if (s) |x| .{ .seed = x.seed, .temperature = 1.0 } else null;
    return lanes.fake.next(history, keyed, history.len);
}

fn sameSampling(a: ?scheduling.Sampling, b: ?scheduling.Sampling) bool {
    if (a == null or b == null) return a == null and b == null;
    return a.?.seed == b.?.seed and a.?.temperature == b.?.temperature and a.?.top_k == b.?.top_k and a.?.top_p == b.?.top_p and a.?.min_p == b.?.min_p;
}

const Steps = struct {
    slots: [4]Slot = @splat(.{}),
    chunks: usize = 3,
    pool: ?*P.Pool = null,
    draft_gate: ?*Gate = null,
    depth: usize = 3,
    /// pairs a slot's backlog holds before a flush (the family's 256; small here so a shared phase flushes)
    backlog: usize = 8,
    tracks: usize = 0,
    drafted_rounds: usize = 0,
    proposed: usize = 0,
    refused_drafts: usize = 0,
    loops: usize = 0,
    resumed_loops: usize = 0,
    carries: usize = 0,
    flushes: usize = 0,
    stale: usize = 0,
    rounds: usize = 0,
    max_rows: usize = 0,
    passes: usize = 0,
    whole: usize = 0,
    deferred: usize = 0,
    refused: usize = 0,
    masked_draws: usize = 0,
    sampled_draws: usize = 0,
    configured: usize = 0,
    configured_sampled: usize = 0,
    /// a sampled row in a round with another row (user_sampling's keyed draw beside other users)
    sampled_shared: usize = 0,

    pub const DraftHook = struct {
        ctx: *anyopaque,
        kept: *const fn (ctx: *anyopaque, tokens: []const u32, drafts: usize, keep: usize) anyerror!bool,
    };
    pub const DraftRan = struct { rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, tokens: usize = 0, ns: u64 = 0 };

    fn deinit(s: *Steps) void {
        for (&s.slots) |*sl| sl.history.deinit(gpa);
    }

    /// shared_pool (Users.reserve): the request's region before its slot changes owner; a full pool refuses it unchanged.
    pub fn reserve(s: *Steps, slot: u8, prompt_len: usize, max_new: usize) !void {
        const p = s.pool orelse return;
        var moves: [P.max_slots]P.Move = undefined;
        _ = p.reserve(slot, P.need(prompt_len, max_new), &moves) catch |err| {
            s.refused += 1;
            return err;
        };
    }

    /// user_sampling (Users.configureSampling): the slot's request sampling at its claim (null: greedy).
    pub fn configureSampling(s: *Steps, slot: u8, sampling: ?scheduling.Sampling) void {
        s.slots[slot].sampling = sampling;
        s.configured += 1;
        if (sampling != null) s.configured_sampled += 1;
    }

    fn claim(s: *Steps, key: Key, ids: []const u32) !void {
        const sl = &s.slots[key.slot];
        if (key.generation <= sl.generation) return error.StaleGeneration;
        sl.generation = key.generation;
        sl.history.clearRetainingCapacity();
        try sl.history.appendSlice(gpa, ids);
    }

    /// A tracked pass absorbed every prompt row but the last (qualification_21e21bc0's prefill absorb); its start pair ends at `first`.
    fn passEnd(s: *Steps, slot: u8, first: u32) !void {
        const sl = &s.slots[slot];
        if (sl.track == .off) return;
        if (sl.sampling != null) return error.SampledTracked; // user_drafts's rule: a sampled stream never drafts
        sl.mtp = sl.history.items.len - 1;
        sl.owed = 1;
        sl.last = first;
        sl.head = sl.track == .lone;
    }

    pub fn prefill(s: *Steps, key: Key, ids: []const u32) !u32 {
        try s.claim(key, ids);
        s.whole += 1;
        const sl = &s.slots[key.slot];
        const first = draw(sl.history.items, sl.sampling);
        try s.passEnd(key.slot, first);
        return first;
    }

    /// structured_users: the prompt pass with its first draw deferred to firstMasked.
    pub fn prefillDeferred(s: *Steps, key: Key, ids: []const u32) !void {
        try s.claim(key, ids);
        s.deferred += 1;
    }

    pub fn firstMasked(s: *Steps, key: Key, mask: []const u32) !u32 {
        const sl = &s.slots[key.slot];
        if (mask.len == 0) return draw(sl.history.items, sl.sampling);
        s.masked_draws += 1;
        return lowest(mask);
    }

    pub fn checkMasked(_: *Steps) !void {}

    pub fn maskedWidth(_: *const Steps) usize {
        return width;
    }

    pub fn maskedWords(_: *const Steps) usize {
        return words;
    }

    pub fn maskCheck(_: *Steps, mask: []const u32) !void {
        if (mask.len != words) return error.MaskShape;
        for (mask) |w| if (w != 0) return;
        return error.EmptyMask;
    }

    /// prefill_interleave: one step of `chunks`; the first claims the generation, the last gives the first token.
    pub fn prefillStep(s: *Steps, key: Key, ids: []const u32) !?u32 {
        const sl = &s.slots[key.slot];
        if (sl.step == null) {
            if (key.generation <= sl.generation) return error.StaleGeneration;
            sl.generation = key.generation;
            sl.history.clearRetainingCapacity();
            sl.step = 0;
            s.passes += 1;
        } else if (key.generation != sl.generation) return error.StaleGeneration;
        const k = sl.step.?;
        sl.step = k + 1;
        if (k + 1 < s.chunks) return null;
        sl.step = null;
        try sl.history.appendSlice(gpa, ids);
        const first = draw(sl.history.items, sl.sampling);
        try s.passEnd(key.slot, first);
        return first;
    }

    pub fn position(s: *Steps, key: Key) usize {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        return s.slots[key.slot].history.items.len;
    }

    /// The slot's pass, its carried MTP state and its sampling end with its owner; (shared_pool) its blocks back.
    pub fn release(s: *Steps, key: Key) void {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        const sl = &s.slots[key.slot];
        sl.step = null;
        sl.track = .off;
        sl.owed = 0;
        sl.head = false;
        sl.sampling = null;
        if (s.pool) |p| _ = p.release(key.slot);
    }

    pub fn round(s: *Steps, rows: []const Row, out: []u32) !void {
        for (rows) |row| {
            const sl = &s.slots[row.key.slot];
            if (sl.generation != row.key.generation) return error.StaleGeneration;
            if (sl.history.items.len != row.position) return error.PositionMismatch;
            if (sl.step != null) return error.FillingInRound;
            if (!sameSampling(row.sampling, sl.sampling)) return error.RowSamplingNotTheSlots;
            if (row.sampling != null and sl.track != .off) return error.SampledTracked;
        }
        s.rounds += 1;
        s.max_rows = @max(s.max_rows, rows.len);
        for (rows, out) |row, *o| {
            const sl = &s.slots[row.key.slot];
            try sl.history.append(gpa, row.token);
            if (row.mask.len != 0) {
                s.masked_draws += 1;
                o.* = lowest(row.mask);
            } else o.* = draw(sl.history.items, row.sampling);
            if (row.sampling != null) {
                s.sampled_draws += 1;
                if (rows.len > 1) s.sampled_shared += 1;
            }
            if (sl.track == .off) continue;
            if (sl.head) { // a lone opening that never drafted: untracked from here (the family's `grown`)
                sl.track = .off;
                sl.owed = 0;
                sl.head = false;
                continue;
            }
            if (sl.last != row.token) return error.BacklogOutOfStep; // the owed pairs end at the row's token
            sl.owed += 1;
            sl.last = o.*;
        }
        for (rows) |row| { // the family's flushFull: a full backlog absorbs all its pairs but the last
            const sl = &s.slots[row.key.slot];
            if (sl.track == .off or sl.owed < s.backlog) continue;
            sl.mtp += sl.owed - 1;
            sl.owed = 1;
            s.flushes += 1;
        }
    }

    // ---- user_drafts's drafting calls (cuda_afd_users.zig's track, carried, drafted, carry) ----

    pub fn track(s: *Steps, slot: u8, how: Track) void {
        const sl = &s.slots[slot];
        sl.track = how;
        sl.owed = 0;
        sl.head = false;
        if (how != .off) s.tracks += 1;
    }

    pub fn carried(s: *Steps, key: Key) bool {
        const sl = &s.slots[key.slot];
        return sl.generation == key.generation and sl.track != .off and sl.owed >= 1 and !sl.head;
    }

    pub fn carry(s: *Steps, key: Key, kept: []const u32) !void {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (kept.len != sl.owed or kept[kept.len - 1] != sl.last) return error.CarryOutOfStep;
        s.carries += 1;
    }

    /// mtpWith's shape (user_drafts's stand-in), refused for a sampled slot (user_drafts's rule kept in the join).
    pub fn drafted(s: *Steps, key: Key, pending: u32, count: usize, out: []u32, hook: *const DraftHook) !DraftRan {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (sl.sampling != null) return error.SampledDrafted;
        if (sl.track == .off) return error.NotTracked;
        if (count == 0 or count > out.len) return error.Count;
        if (sl.owed == 0 or sl.last != pending or sl.mtp + sl.owed != sl.history.items.len) {
            s.stale += 1;
            return error.StaleMtp;
        }
        s.loops += 1;
        if (!sl.head) s.resumed_loops += 1;
        sl.mtp += sl.owed - 1;
        sl.owed = 1;
        sl.head = false;
        var r: DraftRan = .{};
        out[0] = pending;
        var n: usize = 1;
        var cur = pending;
        var h: std.ArrayList(u32) = .empty;
        defer h.deinit(gpa);
        while (n < count) {
            if (s.draft_gate) |g| g.at();
            s.drafted_rounds += 1;
            sl.mtp += sl.owed; // the draft's absorb
            sl.owed = 0;
            const depth = @min(s.depth, count - n);
            var drafts: [8]u32 = undefined;
            h.clearRetainingCapacity();
            try h.appendSlice(gpa, sl.history.items);
            try h.append(gpa, cur);
            for (0..depth) |j| { // the head's guess of each next token; every fifth position's is refused
                const right = lanes.fake.next(h.items, null, h.items.len);
                drafts[j] = if (h.items.len % 5 == 0) (right + 1) % lanes.fake.vocab else right;
                try h.append(gpa, drafts[j]);
            }
            var sampled: [9]u32 = undefined; // the verify: row j's draw after [cur, d1..dj]
            h.clearRetainingCapacity();
            try h.appendSlice(gpa, sl.history.items);
            try h.append(gpa, cur);
            var keep: usize = 1;
            for (0..depth + 1) |j| {
                sampled[j] = lanes.fake.next(h.items, null, h.items.len);
                if (j == depth or sampled[j] != drafts[j]) break;
                keep += 1;
                try h.append(gpa, drafts[j]);
            }
            s.proposed += depth;
            s.refused_drafts += depth - (keep - 1);
            try sl.history.append(gpa, cur);
            try sl.history.appendSlice(gpa, drafts[0 .. keep - 1]);
            sl.owed = keep;
            sl.last = sampled[keep - 1];
            r.rounds += 1;
            r.drafted += depth;
            r.accepted += keep - 1;
            const take = @min(keep, count - n);
            @memcpy(out[n .. n + take], sampled[0..take]);
            n += take;
            cur = sampled[keep - 1];
            if (try hook.kept(hook.ctx, sampled[0..take], depth, keep)) break;
        }
        r.tokens = n;
        return r;
    }
};

const Back = users.Users(Steps);

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    drafted: u64 = 0,
    message: [96]u8 = @splat(0),

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |x| b.tokens.appendSlice(gpa, x) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.drafted = f.stats.drafted;
                const n = @min(f.message.len, b.message.len);
                @memcpy(b.message[0..n], f.message[0..n]);
            },
            else => {},
        }
    }

    fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }

    fn wait(b: *Box) !api.Reason {
        for (0..30_000) |_| {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.WaitTimeout;
    }

    fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
};

/// The reference reply: each token the draw of the prompt and every token before it, greedy (the undrafted reply)
/// or keyed by `sampling`'s seed (boot S's seeded reply).
fn expectRef(prompt: []const u32, sampling: ?lanes.Sampling, got: []const u32) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, prompt);
    for (got) |x| {
        try eq(lanes.fake.next(history.items, sampling, history.items.len), x);
        try history.append(gpa, x);
    }
}

const Opts = struct { drafts: bool = true, sampled: bool = true, interleave: bool = true, pool: ?*P.Pool = null, draft_reach: usize = cap };

/// The users backend over Steps at TF_AFD_USERS=n, the lane core and the real LaneHost as afd_attn_users.run installs
/// them: shared_pool's admission with a pool, prefill_interleave's interleave (1:1), structured_users's structured hook (each schema request
/// the next script), user_sampling's switch (`sampled`), user_drafts's switch (`drafts`: the lone driver installed).
const Served = struct {
    steps: Steps = .{},
    back: Back = undefined,
    cfg: lanes.Config = undefined,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: api.LaneHost = undefined,
    /// holds the engine thread in a drafted round (owned here, so it outlives every engine call)
    gate: Gate = .{},
    scripts: []Script = &.{},
    given: usize = 0,
    up: bool = false,

    fn init(x: *Served, n: u8, o: Opts, scripts: []Script) !void {
        x.steps.pool = o.pool;
        x.scripts = scripts;
        x.back = .{ .steps = &x.steps, .count = n, .capacity = cap, .reach = cap, .interleave = o.interleave, .sampled = o.sampled, .drafts = o.drafts, .draft_reach = o.draft_reach };
        x.cfg = try lanes.Config.init(gpa, x.back.facts(), 1, 0);
        x.core = lanes.Engine.init(gpa, &x.cfg, x.back.backend(), x.clock.clock());
        const admit: ?api.Admit = if (o.pool) |p| .{ .ctx = p, .fits = P.Pool.admitFits } else null;
        x.host = api.LaneHost.init(gpa, io, &x.core, .{ .lanes = n, .context_window = 262144, .drafts = if (o.drafts) "auto" else "off", .greedy_users = true, .users_sampled = o.sampled, .users_drafts = o.drafts, .structures = true, .admit = admit });
        x.host.structured = .{ .ctx = x, .rules = rules };
        if (o.drafts) x.host.lone = x.back.lone(gpa);
        if (o.interleave) x.host.fill_policy = .{ .rounds = 1, .steps = 1 };
    }

    fn start(x: *Served) !void {
        try x.host.start();
        x.up = true;
    }

    fn stop(x: *Served) void {
        if (x.up) x.host.stop();
        x.up = false;
    }

    fn deinit(x: *Served) void {
        x.stop();
        x.core.deinit();
        x.cfg.deinit(gpa);
        x.steps.deinit();
    }

    fn owners(x: *const Served) usize {
        var k: usize = 0;
        for (x.back.slots[0..x.back.count]) |sl| k += @intFromBool(sl.owner != null);
        return k;
    }

    fn rules(ctx: *anyopaque, _: *const api.Request) anyerror!*lanes.grammar.Rules {
        const x: *Served = @ptrCast(@alignCast(ctx));
        if (x.given >= x.scripts.len) return error.NoScript;
        const s = &x.scripts[x.given];
        x.given += 1;
        s.init(width);
        return &s.rules;
    }
};

fn promptOf(comptime n: usize, base: u32) [n]u32 {
    var v: [n]u32 = undefined;
    for (&v, 0..) |*x, i| x.* = base + @as(u32, @intCast(i % 23));
    return v;
}

const pa = promptOf(40, 3);
const pb = promptOf(300, 8);
const pc = promptOf(120, 2);
const pd = promptOf(50, 5);
const json: api.Structure = .{ .kind = .json };
/// The schema request's script: each step's lowest allowed id is 11 + step (the distractors are higher ids).
const schema = [_][]const u32{ &.{ 600, 11 }, &.{12}, &.{ 13, 900 }, &.{14}, &.{ 15, 700 }, &.{16}, &.{17}, &.{ 18, 1000 }, &.{19}, &.{20}, &.{ 21, 650 }, &.{22} };
const schema_reply = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };
const s7: lanes.Sampling = .{ .seed = 7, .temperature = 0.8, .top_k = 0, .top_p = 0.9 };
const s23: lanes.Sampling = .{ .seed = 23, .temperature = 0.6, .top_k = 40, .top_p = 1.0 };

/// The four requests of boot M's mix, as the users server builds them with both switches on (`drafts` unless the
/// request said draft:false; a sampled request's draft:true is admitted and never drafted): a greedy reply that drafts
/// alone, a seeded sampled reply, a sampled schema reply, a greedy reply.
const four = [_]api.Request{
    .{ .prompt = &pa, .max_tokens = 120, .drafts = true },
    .{ .prompt = &pb, .max_tokens = 40, .drafts = true, .sampling = s7 },
    .{ .prompt = &pd, .max_tokens = 20, .drafts = true, .sampling = s23, .structure = json },
    .{ .prompt = &pc, .max_tokens = 30, .drafts = true },
};

fn fixtureModel() !struct { parsed: std.json.Parsed(std.json.Value), m: glm.model.Model } {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    errdefer parsed.deinit();
    var m = try glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
    try m.cut(5);
    m.mtp_layers = 0;
    return .{ .parsed = parsed, .m = m };
}

/// The pool blocks the four requests' regions take (four_user_profile's sizing: exactly them).
fn fourBlocks() usize {
    var blocks: usize = 0;
    for (four) |r| blocks += P.blocksFor(P.need(r.prompt.len, r.max_tokens));
    return blocks;
}

const Reply = struct {
    buf: [256]u32 = undefined,
    n: usize = 0,
    reason: ?api.Reason = null,
    fn got(r: *const Reply) []const u32 {
        return r.buf[0..r.n];
    }
};

const Run = struct {
    out: [4]Reply = @splat(.{}),
    loops: usize = 0,
    resumed_loops: usize = 0,
    drafted_rounds: usize = 0,
    stale: usize = 0,
    rounds: usize = 0,
    max_rows: usize = 0,
    passes: usize = 0,
    sampled_draws: usize = 0,
    sampled_shared: usize = 0,
    configured_sampled: usize = 0,
    masked_draws: usize = 0,
    closes: usize = 0,
    runs: u64 = 0,
    resumes: u64 = 0,
    yields: u64 = 0,
    pool_back: bool = false,
    owners: usize = 0,
    failed: bool = false,
};

/// A fresh TF_AFD_USERS=4 served path with the pool on (exactly the four's regions): `which` submitted before the
/// host starts (arriving together; one index: its solo run).
fn serve(o: Opts, which: []const usize, rs: []const api.Request) !Run {
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    var pool = try P.Pool.init(gpa, &fx.m, fourBlocks() * P.block, 8, true, cap);
    defer pool.deinit(gpa);
    var boxes: [4]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var scripts = [_]Script{.{ .steps = &schema }};
    var x: Served = .{}; // after the pool and the boxes: its deinit (the host's stop) runs before theirs
    var opts = o;
    opts.pool = &pool;
    try x.init(4, opts, &scripts);
    defer x.deinit();
    const e = x.host.engine();
    for (which) |i| try e.submit(@intCast(i + 1), &rs[i], boxes[i].sink());
    try x.start();
    var run: Run = .{};
    for (which) |i| {
        run.out[i].reason = try boxes[i].wait();
        const got = boxes[i].tokens.items;
        @memcpy(run.out[i].buf[0..got.len], got);
        run.out[i].n = got.len;
    }
    x.stop();
    run.loops = x.steps.loops;
    run.resumed_loops = x.steps.resumed_loops;
    run.drafted_rounds = x.steps.drafted_rounds;
    run.stale = x.steps.stale;
    run.rounds = x.steps.rounds;
    run.max_rows = x.steps.max_rows;
    run.passes = x.steps.passes;
    run.sampled_draws = x.steps.sampled_draws;
    run.sampled_shared = x.steps.sampled_shared;
    run.configured_sampled = x.steps.configured_sampled;
    run.masked_draws = x.steps.masked_draws;
    run.closes = scripts[0].closes;
    run.runs = x.back.counts.runs;
    run.resumes = x.back.counts.resumes;
    run.yields = x.back.counts.yields;
    run.pool_back = pool.freeBlocks() == pool.blocks;
    run.owners = x.owners();
    run.failed = x.back.failed != null;
    return run;
}

const envcall = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};
const harness = @import("env_harness.zig"); // FJPANIC: std's startup environment kept whole

/// An environment variable's value restored at the end of a test (four_user_profile's Saved).
const SavedEnv = struct {
    name: [*:0]const u8,
    old: ?[:0]u8,
    fn take(name: [*:0]const u8) !SavedEnv {
        try harness.detach(); // FJPANIC: libc's environ off std's startup array before the test unsets anything
        return .{ .name = name, .old = if (std.c.getenv(name)) |p| try gpa.dupeSentinel(u8, std.mem.span(p), 0) else null };
    }
    fn put(s: SavedEnv) void {
        if (s.old) |v| {
            _ = envcall.setenv(s.name, v, 1);
            gpa.free(v);
        } else _ = envcall.unsetenv(s.name);
    }
};

// ---- the switches together ---------------------------------------------------------------------------------------

test "joined_profile env harness (FJPANIC): SavedEnv moves libc's environ off std's startup array first, so unsetting a name the process started with leaves std's startup environment whole; put() restores the name" {
    var buf: [256]u8 = undefined;
    const name = harness.startupName(&buf) orelse return error.SkipZigTest; // no startup environment: nothing to shift
    {
        const s = try SavedEnv.take(name.ptr);
        defer s.put();
        try t.expect(harness.detached()); // checked before anything is unset, so a failure leaves std's slice whole
        _ = envcall.unsetenv(name.ptr);
        try t.expect(std.c.getenv(name.ptr) == null);
        try t.expect(harness.startupWhole());
    }
    try t.expect(std.c.getenv(name.ptr) != null);
}

test "joined_profile switches: plan() reads TF_AFD_USERS_SAMPLING and TF_AFD_USERS_DRAFTS together at TF_AFD_USERS=4 with the pool and the interleave; either unset leaves the other's plan; each refusal stays its own parent's" {
    const names = [_][*:0]const u8{ "TF_AFD_USERS", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE", "TF_AFD_MTP", "TF_GLM_SAMPLING", "TF_AFD_USERS_SAMPLING", "TF_AFD_USERS_DRAFTS" };
    var saved: [names.len]SavedEnv = undefined;
    for (names, &saved) |n, *s| s.* = try SavedEnv.take(n);
    defer for (saved) |s| s.put();
    for (names) |n| _ = envcall.unsetenv(n);
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const auto = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    const off = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "0", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    _ = envcall.setenv("TF_AFD_USERS", "4", 1);
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "720896", 1);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "1", 1);
    _ = envcall.setenv("TF_AFD_MTP", "1", 1);
    _ = envcall.setenv("TF_GLM_SAMPLING", "1", 1);
    // both unset: four_user_profile's plan (and drafts refused as before)
    const p0 = try served.plan(a, off, cfg);
    try eq(false, p0.sampled);
    try eq(false, p0.users_drafts);
    try t.expectError(error.UsersDraftsUnsupported, served.plan(a, auto, cfg));
    // boot M: both on, with the pool and the interleave
    _ = envcall.setenv("TF_AFD_USERS_SAMPLING", "1", 1);
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    const m = try served.plan(a, auto, cfg);
    try eq(@as(u8, 4), m.users);
    try eq(@as(usize, 720896), m.pool);
    try t.expect(m.interleave != null);
    try eq(true, m.sampled);
    try eq(true, m.users_drafts);
    try t.expect(m.drafts.on());
    // either unset leaves the other's plan
    _ = envcall.unsetenv("TF_AFD_USERS_SAMPLING");
    const d = try served.plan(a, auto, cfg);
    try eq(false, d.sampled);
    try eq(true, d.users_drafts);
    _ = envcall.setenv("TF_AFD_USERS_SAMPLING", "1", 1);
    _ = envcall.unsetenv("TF_AFD_USERS_DRAFTS");
    const s = try served.plan(a, off, cfg);
    try eq(true, s.sampled);
    try eq(false, s.users_drafts);
    try t.expectError(error.UsersDraftsUnsupported, served.plan(a, auto, cfg)); // user_drafts's switch unset: refused
    // each refusal its own parent's
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    try t.expectError(error.UsersDraftsNeedPolicy, served.plan(a, off, cfg));
    _ = envcall.setenv("TF_AFD_USERS_SAMPLING", "2", 1);
    try t.expectError(error.BadAfdUsersSampling, served.plan(a, auto, cfg));
    _ = envcall.setenv("TF_AFD_USERS_SAMPLING", "1", 1);
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "yes", 1);
    try t.expectError(error.BadAfdUsersDrafts, served.plan(a, auto, cfg));
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    _ = envcall.unsetenv("TF_GLM_SAMPLING");
    try t.expectError(error.UsersSamplingNeedsGlmSampling, served.plan(a, auto, cfg));
}

// ---- boot M's mix: identity ------------------------------------------------------------------------------------------

test "joined_profile USERS=4 with pool + interleave + structured + sampling + drafts: a lone greedy reply drafted = its undrafted bytes; every switch off gives the same bytes" {
    const on = try serve(.{}, &.{0}, &four);
    const r = on.out[0];
    try eq(api.Reason.length, r.reason.?);
    try eq(@as(usize, 120), r.n);
    try expectRef(&pa, null, r.got());
    try t.expect(on.loops >= 1 and on.drafted_rounds > 0 and on.drafted_rounds < 120); // drafted: fewer verifies than tokens
    try eq(@as(u64, 1), on.runs);
    try eq(@as(usize, 0), on.stale);
    try t.expect(on.pool_back);
    try eq(@as(usize, 0), on.owners);
    const greedy_off = [_]api.Request{ .{ .prompt = &pa, .max_tokens = 120, .drafts = false }, four[1], four[2], four[3] };
    const off = try serve(.{ .drafts = false, .sampled = false }, &.{0}, &greedy_off);
    try t.expectEqualSlices(u32, r.got(), off.out[0].got());
    try eq(@as(usize, 0), off.loops);
    try eq(@as(usize, 119), off.rounds);
}

test "joined_profile USERS=4 boot M's mix: four different requests together = each one alone (greedy drafting, seeded sampled, sampled schema, greedy); schema reply valid; interleaved = serial; the sampled ones never drafted; the pool all back" {
    var solo: [4]Run = undefined;
    for (0..4) |i| solo[i] = try serve(.{}, &[_]usize{i}, &four);
    const all = [_]usize{ 0, 1, 2, 3 };
    const together = try serve(.{}, &all, &four);
    const serial = try serve(.{ .interleave = false }, &all, &four);
    for (0..4) |i| {
        try t.expectEqualSlices(u32, solo[i].out[i].got(), together.out[i].got());
        try t.expectEqualSlices(u32, together.out[i].got(), serial.out[i].got());
        try eq(solo[i].out[i].reason.?, together.out[i].reason.?);
    }
    // each reply its reference: greedy = the undrafted reply; sampled = boot S's keyed reply; the schema's script
    try expectRef(&pa, null, together.out[0].got());
    try eq(@as(usize, 120), together.out[0].n);
    try expectRef(&pb, s7, together.out[1].got());
    try eq(@as(usize, 40), together.out[1].n);
    try t.expectEqualSlices(u32, &schema_reply, together.out[2].got());
    try eq(api.Reason.stop, together.out[2].reason.?);
    try eq(@as(usize, 1), together.closes);
    try expectRef(&pc, null, together.out[3].got());
    try eq(@as(usize, 30), together.out[3].n);
    // the join's mechanics: shared rounds with sampled rows beside others; prompts a step at a time; drafts only
    // for greedy streams (the stand-in refuses a sampled one); the last greedy left drafted again
    try t.expect(together.max_rows >= 2);
    try t.expect(together.sampled_shared > 0);
    try eq(@as(usize, 2), together.configured_sampled);
    try t.expect(together.passes >= 3); // prefill_interleave: the prompts the lone driver did not take went in steps
    try t.expect(together.resumes >= 1);
    try eq(@as(usize, 0), together.stale);
    try t.expect(together.pool_back and serial.pool_back);
    try eq(@as(usize, 0), together.owners);
    try t.expect(!together.failed and !serial.failed);
    // alone, a sampled request is never drafted: no lone run, every round one row
    try eq(@as(u64, 0), solo[1].runs);
    try eq(@as(usize, 0), solo[1].loops);
    try eq(@as(usize, 1), solo[1].max_rows);
    try eq(@as(u64, 0), solo[2].runs);
}

test "joined_profile a lone greedy stream drafted -> shared -> drafted beside a seeded sampled arrival (both switches on, USERS=4, pool + interleave): both replies = their references" {
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    var pool = try P.Pool.init(gpa, &fx.m, fourBlocks() * P.block, 8, true, cap);
    defer pool.deinit(gpa);
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    var x: Served = .{};
    defer x.deinit();
    try x.init(4, .{ .pool = &pool }, &.{});
    x.steps.draft_gate = &x.gate;
    x.gate.armed.store(true, .release);
    const e = x.host.engine();
    const ra: api.Request = .{ .prompt = &pa, .max_tokens = 160, .drafts = true };
    const rb: api.Request = .{ .prompt = &pb, .max_tokens = 40, .drafts = true, .sampling = s7 };
    try e.submit(1, &ra, ba.sink());
    try x.start();
    try x.gate.waitEntered(); // A drafts alone, held inside its first drafted round
    try e.submit(2, &rb, bb.sink()); // queued at once: the round's hook asks and hands A over
    x.gate.open.store(true, .release);
    _ = try bb.wait();
    _ = try ba.wait();
    x.stop();
    try expectRef(&pa, null, ba.tokens.items);
    try eq(@as(usize, 160), ba.tokens.items.len);
    try expectRef(&pb, s7, bb.tokens.items);
    try eq(@as(usize, 40), bb.tokens.items.len);
    try t.expect(x.back.counts.yields >= 1); // A handed over at B's arrival
    try t.expect(x.steps.carries >= 1);
    try eq(@as(usize, 2), x.steps.max_rows); // A and the sampled B shared undrafted rounds
    try t.expect(x.steps.sampled_shared > 0);
    try t.expect(x.back.counts.resumes >= 1); // B's end: A back to its drafted rounds
    try t.expect(x.steps.resumed_loops >= 1);
    try eq(@as(usize, 0), x.steps.stale);
    try eq(@as(usize, 0), x.owners());
    try eq(pool.blocks, pool.freeBlocks());
    try eq(@as(?anyerror, null), x.back.failed);
}

// ---- the seeded sampled reply --------------------------------------------------------------------------------------

/// A greedy request drafting alone (held in its first drafted round), then the seeded sampled `rs` arriving.
fn switched(ra: *const api.Request, rs: *const api.Request, a_box: *Box, s_box: *Box, x: *Served, pool: *P.Pool) !void {
    try x.init(4, .{ .pool = pool }, &.{});
    x.steps.draft_gate = &x.gate;
    x.gate.armed.store(true, .release);
    const e = x.host.engine();
    try e.submit(1, ra, a_box.sink());
    try x.start();
    try x.gate.waitEntered();
    try e.submit(2, rs, s_box.sink());
    x.gate.open.store(true, .release);
    _ = try s_box.wait();
    _ = try a_box.wait();
    x.stop();
}

test "joined_profile a seeded sampled reply at USERS=4 (both switches on) is byte-identical alone, beside three users and through a lone drafted <-> shared undrafted switch, and = boot S's keyed reply" {
    const rs: api.Request = .{ .prompt = &pc, .max_tokens = 60, .drafts = true, .sampling = s7 };
    const mix = [_]api.Request{ four[0], rs, four[2], four[3] };
    // alone: never drafted, every round one row
    const alone = try serve(.{}, &.{1}, &mix);
    try expectRef(&pc, s7, alone.out[1].got());
    try eq(@as(usize, 60), alone.out[1].n);
    try eq(@as(u64, 0), alone.runs);
    try eq(@as(usize, 0), alone.loops);
    try eq(@as(usize, 1), alone.max_rows);
    // beside three users (a greedy one drafting, a sampled schema, a greedy one)
    const beside = try serve(.{}, &.{ 0, 1, 2, 3 }, &mix);
    try t.expectEqualSlices(u32, alone.out[1].got(), beside.out[1].got());
    try t.expect(beside.max_rows >= 2 and beside.sampled_shared > 0);
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    // through a switch, the sampled reply the shorter: the greedy one drafted -> shared -> drafted again
    {
        var pool = try P.Pool.init(gpa, &fx.m, fourBlocks() * P.block, 8, true, cap);
        defer pool.deinit(gpa);
        var ab: Box = .{};
        defer ab.deinit();
        var sb: Box = .{};
        defer sb.deinit();
        var x: Served = .{};
        defer x.deinit();
        const long: api.Request = .{ .prompt = &pa, .max_tokens = 160, .drafts = true };
        try switched(&long, &rs, &ab, &sb, &x, &pool);
        try t.expectEqualSlices(u32, alone.out[1].got(), sb.tokens.items);
        try expectRef(&pa, null, ab.tokens.items);
        try t.expect(x.back.counts.yields >= 1 and x.back.counts.resumes >= 1);
        try eq(@as(usize, 0), x.steps.stale);
        try eq(pool.blocks, pool.freeBlocks());
    }
    // through a switch, the greedy reply the shorter: the sampled one goes on alone, undrafted, never taken back
    {
        var pool = try P.Pool.init(gpa, &fx.m, fourBlocks() * P.block, 8, true, cap);
        defer pool.deinit(gpa);
        var ab: Box = .{};
        defer ab.deinit();
        var sb: Box = .{};
        defer sb.deinit();
        var x: Served = .{};
        defer x.deinit();
        const short: api.Request = .{ .prompt = &pa, .max_tokens = 24, .drafts = true };
        try switched(&short, &rs, &ab, &sb, &x, &pool);
        try t.expectEqualSlices(u32, alone.out[1].got(), sb.tokens.items);
        try expectRef(&pa, null, ab.tokens.items);
        try eq(@as(usize, 24), ab.tokens.items.len);
        try t.expect(x.back.counts.yields >= 1);
        try eq(@as(u64, 0), x.back.counts.resumes); // a sampled stream left alone stays in the lane core's rounds
        try eq(@as(usize, 0), x.steps.stale);
        try eq(pool.blocks, pool.freeBlocks());
    }
}

// ---- each switch keeps its own default-off bytes ---------------------------------------------------------------

test "joined_profile each switch alone keeps its parent's behaviour at USERS=4 (pool + interleave + structured): both unset and drafts-only refuse a sampled request by name (GreedyOnly); sampling-only serves it = boot S; the greedy and schema bytes equal across all four switch settings" {
    const greedy = [_]api.Request{
        .{ .prompt = &pa, .max_tokens = 120, .drafts = false },
        .{ .prompt = &pb, .max_tokens = 40, .drafts = false },
        .{ .prompt = &pd, .max_tokens = 20, .drafts = false, .structure = json },
        .{ .prompt = &pc, .max_tokens = 30, .drafts = false },
    };
    const drafted = [_]api.Request{ four[0], .{ .prompt = &pb, .max_tokens = 40, .drafts = true }, .{ .prompt = &pd, .max_tokens = 20, .drafts = true, .structure = json }, four[3] };
    const all = [_]usize{ 0, 1, 2, 3 };
    const none = try serve(.{ .drafts = false, .sampled = false }, &all, &greedy);
    const samp_only = try serve(.{ .drafts = false }, &all, &greedy);
    const drafts_only = try serve(.{ .sampled = false }, &all, &drafted);
    const both = try serve(.{}, &all, &drafted);
    for (0..4) |i| {
        try t.expectEqualSlices(u32, none.out[i].got(), samp_only.out[i].got());
        try t.expectEqualSlices(u32, none.out[i].got(), drafts_only.out[i].got());
        try t.expectEqualSlices(u32, none.out[i].got(), both.out[i].got());
    }
    try eq(@as(usize, 0), none.loops + samp_only.loops);
    try t.expect(drafts_only.loops >= 1 and both.loops >= 1);
    // a sampled request: refused by name unless TF_AFD_USERS_SAMPLING=1 (user_sampling), then = boot S
    const sampled = [_]api.Request{ .{ .prompt = &pb, .max_tokens = 40, .drafts = false, .sampling = s7 }, four[1], four[2], four[3] };
    inline for (.{ Opts{ .drafts = false, .sampled = false }, Opts{ .sampled = false } }) |o| {
        const fx = try fixtureModel();
        defer fx.parsed.deinit();
        var pool = try P.Pool.init(gpa, &fx.m, fourBlocks() * P.block, 8, true, cap);
        defer pool.deinit(gpa);
        var box: Box = .{};
        defer box.deinit();
        var x: Served = .{};
        defer x.deinit();
        var opts = o;
        opts.pool = &pool;
        try x.init(4, opts, &.{});
        const req = if (o.drafts) four[1] else sampled[0];
        try x.host.engine().submit(1, &req, box.sink());
        try x.start();
        try eq(api.Reason.failed, try box.wait());
        try t.expect(std.mem.indexOf(u8, &box.message, "GreedyOnly") != null);
        x.stop();
        try eq(@as(usize, 0), x.steps.loops);
        try eq(pool.blocks, pool.freeBlocks());
    }
    const served_s = try serve(.{ .drafts = false }, &.{0}, &sampled);
    try expectRef(&pb, s7, served_s.out[0].got());
    try eq(@as(usize, 40), served_s.out[0].n);
}
