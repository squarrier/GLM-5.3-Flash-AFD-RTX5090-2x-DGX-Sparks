//! four_user_profile: the USERS=4 join's combined CPU tests (step test-four_user_profile, filter "four_user_profile"): shared_pool's shared cache pool
//! (TF_AFD_POOL_TOKENS), prefill_interleave's interleaved prefill (TF_AFD_USERS_INTERLEAVE) and structured_users's structured output
//! (TF_AFD_GRAMMAR_SERVE) together on the users backend (native/glm_cuda_users.zig) through the real lane core and
//! LaneHost, TF_AFD_USERS=4. The Steps stand-in's tokens depend on each slot's whole history (lanes.fake.next, as
//! prefill_interleave's afd_users_fill_test.zig), its cache regions come from the real pool's accounting (cuda_afd_pool.zig's
//! Pool: reserve, release, admitFits; no device), and a schema request's matcher is a FAKE script (structured_users's
//! structured_users_users_test.zig form) whose draws take the lowest id its mask allows. No CUDA numerics.
//! Ash Hart authored TensorFold and its lane core (#623 by @BlivionIaG, merged by him); the pool follows TensorFold
//! v1.0.7's pool and Mia's GLM recipe patch 0074 (ezoushen); shared_pool, prefill_interleave and structured_users are Claude Opus 5.5 via
//! Hermes, as their files say; this file Claude Opus 5.5 (Anthropic) via Hermes.
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

/// The stand-in's mask width (a schema row's draws); lanes.fake.next's plain draws stay below 97.
const width: usize = 2048;
const words = (width + 31) / 32;
/// The served capacity at the 262,144 window (concurrent_users's), the backend's ContextFull bound.
const cap: usize = 262208;

/// FAKE matcher (structured_users's Script): step i allows `steps[i]`; terminated once every step is accepted; counts its closes.
const Script = struct {
    rules: lanes.grammar.Rules = undefined,
    steps: []const []const u32,
    accepted: usize = 0,
    fills: usize = 0,
    closes: usize = 0,
    fn init(x: *Script, w: usize) void {
        x.rules = .{ .ptr = x, .width = w, .vtable = &.{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close } };
    }
    fn of(p: *anyopaque) *Script {
        return @ptrCast(@alignCast(p));
    }
    fn fill(p: *anyopaque, out: []u32) !void {
        const x = of(p);
        x.fills += 1;
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

/// Holds the engine thread inside its next round once armed, until opened (requests submitted meanwhile arrive together).
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

const Slot = struct { generation: u64 = 0, history: std.ArrayList(u32) = .empty, step: ?usize = null };
const Kind = enum { reserve, whole, deferred, step, round, release };
/// What Steps did, in order, with every slot's position at that moment (the recorder).
const Ev = struct { kind: Kind, slot: u8, generation: u64, k: usize = 0, positions: [4]usize };

/// prefill_interleave's Steps stand-in with shared_pool's `reserve` (the real pool's accounting) and structured_users's masked calls.
const Steps = struct {
    slots: [4]Slot = @splat(.{}),
    chunks: usize = 6,
    log: std.ArrayList(Ev) = .empty,
    pool: ?*P.Pool = null,
    gate: ?*Gate = null,
    refused: usize = 0,
    whole: usize = 0,
    deferred: usize = 0,
    passes: usize = 0,
    rounds: usize = 0,
    max_rows: usize = 0,
    masked_draws: usize = 0,

    fn deinit(s: *Steps) void {
        for (&s.slots) |*sl| sl.history.deinit(gpa);
        s.log.deinit(gpa);
    }

    fn note(s: *Steps, kind: Kind, slot: u8, generation: u64, k: usize) void {
        var p: [4]usize = undefined;
        for (s.slots, &p) |sl, *v| v.* = sl.history.items.len;
        s.log.append(gpa, .{ .kind = kind, .slot = slot, .generation = generation, .k = k, .positions = p }) catch {};
    }

    /// shared_pool (Users.reserve): the request's region before its slot changes owner; a full pool refuses it unchanged.
    pub fn reserve(s: *Steps, slot: u8, prompt_len: usize, max_new: usize) !void {
        const p = s.pool orelse return;
        var moves: [P.max_slots]P.Move = undefined;
        _ = p.reserve(slot, P.need(prompt_len, max_new), &moves) catch |err| {
            s.refused += 1;
            return err;
        };
        s.note(.reserve, slot, 0, P.blocksFor(P.need(prompt_len, max_new)));
    }

    fn claim(s: *Steps, key: Key, ids: []const u32) !void {
        const sl = &s.slots[key.slot];
        if (key.generation <= sl.generation) return error.StaleGeneration;
        sl.generation = key.generation;
        sl.history.clearRetainingCapacity();
        try sl.history.appendSlice(gpa, ids);
    }

    pub fn prefill(s: *Steps, key: Key, ids: []const u32) !u32 {
        try s.claim(key, ids);
        s.whole += 1;
        s.note(.whole, key.slot, key.generation, 0);
        const h = s.slots[key.slot].history.items;
        return lanes.fake.next(h, null, h.len);
    }

    /// structured_users: the prompt pass with its first draw deferred to firstMasked.
    pub fn prefillDeferred(s: *Steps, key: Key, ids: []const u32) !void {
        try s.claim(key, ids);
        s.deferred += 1;
        s.note(.deferred, key.slot, key.generation, 0);
    }

    pub fn firstMasked(s: *Steps, key: Key, mask: []const u32) !u32 {
        const h = s.slots[key.slot].history.items;
        if (mask.len == 0) return lanes.fake.next(h, null, h.len);
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
        s.note(.step, key.slot, key.generation, k);
        sl.step = k + 1;
        if (k + 1 < s.chunks) return null;
        sl.step = null;
        try sl.history.appendSlice(gpa, ids);
        return lanes.fake.next(sl.history.items, null, sl.history.items.len);
    }

    pub fn position(s: *Steps, key: Key) usize {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        return s.slots[key.slot].history.items.len;
    }

    /// The slot's pass ends and (shared_pool) exactly its region's blocks go back to the pool.
    pub fn release(s: *Steps, key: Key) void {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        s.slots[key.slot].step = null;
        if (s.pool) |p| _ = p.release(key.slot);
        s.note(.release, key.slot, key.generation, 0);
    }

    pub fn round(s: *Steps, rows: []const Row, out: []u32) !void {
        for (rows) |row| {
            const sl = &s.slots[row.key.slot];
            if (sl.generation != row.key.generation) return error.StaleGeneration;
            if (sl.history.items.len != row.position) return error.PositionMismatch;
            if (sl.step != null) return error.FillingInRound; // a filling slot never takes part in a round
        }
        if (s.gate) |g| g.at();
        s.rounds += 1;
        s.max_rows = @max(s.max_rows, rows.len);
        for (rows, out) |row, *o| {
            const sl = &s.slots[row.key.slot];
            try sl.history.append(gpa, row.token);
            if (row.mask.len != 0) {
                s.masked_draws += 1;
                o.* = lowest(row.mask);
            } else o.* = lanes.fake.next(sl.history.items, null, sl.history.items.len);
        }
        s.note(.round, rows[0].key.slot, rows[0].key.generation, rows.len);
    }
};

/// Steps whose rounds between two steps of one pass, over every pass in the record.
fn gapsWithRounds(s: *const Steps) usize {
    var n: usize = 0;
    for (s.log.items, 0..) |ev, i| {
        if (ev.kind != .step or ev.k == 0) continue;
        var j = i;
        var between = false;
        while (j > 0) {
            j -= 1;
            const p = s.log.items[j];
            if (p.kind == .step and p.slot == ev.slot and p.generation == ev.generation) break;
            if (p.kind == .round) between = true;
        }
        if (between) n += 1;
    }
    return n;
}

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |x| b.tokens.appendSlice(gpa, x) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }

    fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }

    fn count(b: *Box) usize {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.tokens.items.len;
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

    fn waitTokens(b: *Box, n: usize) !void {
        for (0..30_000) |_| {
            if (b.count() >= n) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.WaitTimeout;
    }

    fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
};

fn expectSolo(prompt: []const u32, got: []const u32) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, prompt);
    for (got) |x| {
        try eq(lanes.fake.next(history.items, null, history.items.len), x);
        try history.append(gpa, x);
    }
}

const Back = users.Users(Steps);

/// The users backend over Steps at TF_AFD_USERS=n, the lane core and the real LaneHost: shared_pool's admission hook with a
/// pool, prefill_interleave's interleave (1:1) when on, structured_users's structured hook (each schema request takes the next script).
const Served = struct {
    steps: Steps = .{},
    back: Back = undefined,
    cfg: lanes.Config = undefined,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: api.LaneHost = undefined,
    scripts: []Script = &.{},
    given: usize = 0,
    up: bool = false,

    fn init(x: *Served, n: u8, interleave: bool, pool: ?*P.Pool, scripts: []Script) !void {
        x.steps.pool = pool;
        x.scripts = scripts;
        x.back = .{ .steps = &x.steps, .count = n, .capacity = cap, .reach = cap, .interleave = interleave };
        x.cfg = try lanes.Config.init(gpa, x.back.facts(), 1, 0);
        x.core = lanes.Engine.init(gpa, &x.cfg, x.back.backend(), x.clock.clock());
        const admit: ?api.Admit = if (pool) |p| .{ .ctx = p, .fits = P.Pool.admitFits } else null;
        x.host = api.LaneHost.init(gpa, io, &x.core, .{ .lanes = n, .context_window = 262144, .drafts = "off", .greedy_users = true, .structures = true, .admit = admit });
        x.host.structured = .{ .ctx = x, .rules = rules };
        x.host.fill_policy = .{ .rounds = 1, .steps = 1 };
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

    fn rules(ctx: *anyopaque, _: *const api.Request) anyerror!*lanes.grammar.Rules {
        const x: *Served = @ptrCast(@alignCast(ctx));
        if (x.given >= x.scripts.len) return error.NoScript;
        const s = &x.scripts[x.given];
        x.given += 1;
        s.init(width);
        return &s.rules;
    }
};

fn fixtureModel(cut: ?usize, mtp: usize) !struct { parsed: std.json.Parsed(std.json.Value), m: glm.model.Model } {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    errdefer parsed.deinit();
    var m = try glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
    if (cut) |n| try m.cut(n);
    m.mtp_layers = @intCast(mtp);
    return .{ .parsed = parsed, .m = m };
}

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
/// Three plain users and one schema request (TF_AFD_USERS=4).
const four = [_]api.Request{
    .{ .prompt = &pa, .max_tokens = 120, .drafts = false },
    .{ .prompt = &pb, .max_tokens = 40, .drafts = false },
    .{ .prompt = &pc, .max_tokens = 30, .drafts = false },
    .{ .prompt = &pd, .max_tokens = 20, .drafts = false, .structure = json },
};

const Reply = struct {
    buf: [256]u32 = undefined,
    n: usize = 0,
    reason: ?api.Reason = null,
    fn got(r: *const Reply) []const u32 {
        return r.buf[0..r.n];
    }
};

const Stats = struct {
    failed: bool = false,
    pool_back: bool = false,
    reserves: u64 = 0,
    reserved: u64 = 0,
    released: u64 = 0,
    refused: usize = 0,
    whole: usize = 0,
    deferred: usize = 0,
    passes: usize = 0,
    rounds: usize = 0,
    max_rows: usize = 0,
    masked_draws: usize = 0,
    gaps: usize = 0,
    closes: usize = 0,
};

/// A fresh served TF_AFD_USERS=4 path with the pool on (exactly the four requests' regions) and the interleave on or
/// off: the four submitted before the host starts (arriving together), or only `only` (its solo run).
fn serve(interleave: bool, only: ?usize, out: *[4]Reply, st: *Stats) !void {
    const fx = try fixtureModel(5, 0);
    defer fx.parsed.deinit();
    var blocks: usize = 0;
    for (four) |r| blocks += P.blocksFor(P.need(r.prompt.len, r.max_tokens));
    var pool = try P.Pool.init(gpa, &fx.m, blocks * P.block, 8, true, cap);
    defer pool.deinit(gpa);
    var boxes: [4]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var scripts = [_]Script{.{ .steps = &schema }};
    var x: Served = .{}; // after the pool and the boxes: its deinit (the host's stop) runs before theirs
    try x.init(4, interleave, &pool, &scripts);
    defer x.deinit();
    const e = x.host.engine();
    for (&four, 0..) |*r, i| if (only == null or only.? == i) try e.submit(@intCast(i + 1), r, boxes[i].sink());
    try x.start();
    for (0..four.len) |i| if (only == null or only.? == i) {
        out[i].reason = try boxes[i].wait();
        const got = boxes[i].tokens.items;
        @memcpy(out[i].buf[0..got.len], got);
        out[i].n = got.len;
    };
    x.stop();
    st.* = .{
        .failed = x.back.failed != null,
        .pool_back = pool.freeBlocks() == pool.blocks,
        .reserves = pool.reserves,
        .reserved = pool.reserved_blocks,
        .released = pool.released_blocks,
        .refused = x.steps.refused,
        .whole = x.steps.whole,
        .deferred = x.steps.deferred,
        .passes = x.steps.passes,
        .rounds = x.steps.rounds,
        .max_rows = x.steps.max_rows,
        .masked_draws = x.steps.masked_draws,
        .gaps = gapsWithRounds(&x.steps),
        .closes = scripts[0].closes,
    };
}

const envcall = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

/// An environment variable's value restored at the end of a test.
const Saved = struct {
    name: [*:0]const u8,
    old: ?[:0]u8,
    fn take(name: [*:0]const u8) !Saved {
        return .{ .name = name, .old = if (std.c.getenv(name)) |p| try gpa.dupeSentinel(u8, std.mem.span(p), 0) else null };
    }
    fn put(s: Saved) void {
        if (s.old) |v| {
            _ = envcall.setenv(s.name, v, 1);
            gpa.free(v);
        } else _ = envcall.unsetenv(s.name);
    }
};

test "four_user_profile switches: TF_AFD_USERS=4 with TF_AFD_POOL_TOKENS and TF_AFD_USERS_INTERLEAVE read together by plan(); either unset leaves the other's plan; each refusal stays its own parent's" {
    const I = scheduling.Interleave;
    const users_env = try Saved.take("TF_AFD_USERS");
    defer users_env.put();
    const pool_env = try Saved.take("TF_AFD_POOL_TOKENS");
    defer pool_env.put();
    const il_env = try Saved.take("TF_AFD_USERS_INTERLEAVE");
    defer il_env.put();
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const flags = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--no-drafts", "--drafts", "0", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    _ = envcall.setenv("TF_AFD_USERS", "4", 1);
    _ = envcall.unsetenv("TF_AFD_POOL_TOKENS");
    _ = envcall.unsetenv("TF_AFD_USERS_INTERLEAVE");
    const off = try served.plan(a, flags, cfg); // both unset: concurrent_users's USERS=4 plan
    try eq(@as(usize, 0), off.pool);
    try eq(@as(?I, null), off.interleave);
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "781568", 1);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "1", 1);
    const both = try served.plan(a, flags, cfg);
    try eq(@as(usize, 781_568), both.pool);
    try eq(@as(?I, .{ .rounds = 1, .steps = 1 }), both.interleave);
    try eq(@as(u8, 4), both.users);
    try eq(@as(usize, 262_208), both.capacity);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "8:1", 1);
    try eq(@as(?I, .{ .rounds = 8, .steps = 1 }), (try served.plan(a, flags, cfg)).interleave);
    _ = envcall.unsetenv("TF_AFD_POOL_TOKENS");
    const il_only = try served.plan(a, flags, cfg);
    try eq(@as(usize, 0), il_only.pool);
    try eq(@as(?I, .{ .rounds = 8, .steps = 1 }), il_only.interleave);
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "781568", 1);
    _ = envcall.unsetenv("TF_AFD_USERS_INTERLEAVE");
    const pool_only = try served.plan(a, flags, cfg);
    try eq(@as(usize, 781_568), pool_only.pool);
    try eq(@as(?I, null), pool_only.interleave);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "1", 1);
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "262144", 1);
    try t.expectError(error.PoolTooSmall, served.plan(a, flags, cfg)); // shared_pool's refusal with the interleave on
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "781568", 1);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "yes", 1);
    try t.expectError(error.BadAfdUsersInterleave, served.plan(a, flags, cfg)); // prefill_interleave's refusal with the pool on
}

test "four_user_profile pool + interleave: a request queued beside a prompt that has not taken its region yet waits for that region (the prompt's first step); a pool that cannot hold it then waits, never refused, and runs once a region returns" {
    const fx = try fixtureModel(5, 0);
    defer fx.parsed.deinit();
    const live: api.Request = .{ .prompt = &pa, .max_tokens = 100000, .drafts = false };
    const ba = P.blocksFor(P.need(pa.len, live.max_tokens));
    const bb = P.blocksFor(P.need(pb.len, four[1].max_tokens));
    var pool = try P.Pool.init(gpa, &fx.m, (ba + bb) * P.block, 8, true, cap); // the live user's and B's regions, no more
    defer pool.deinit(gpa);
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var gate: Gate = .{};
    defer gate.open.store(true, .release);
    var x: Served = .{}; // after the pool, the boxes and the gate: its deinit (the host's stop) runs before theirs
    try x.init(4, true, &pool, &.{});
    defer x.deinit();
    x.steps.gate = &gate;
    try x.start();
    const e = x.host.engine();
    try e.submit(1, &live, a.sink());
    try a.waitTokens(3);
    gate.armed.store(true, .release); // B and C arrive together while the live user's round runs
    try gate.waitEntered();
    try e.submit(2, &four[1], b.sink());
    try e.submit(3, &four[2], c.sink());
    gate.open.store(true, .release);
    try eq(api.Reason.length, try b.wait());
    try eq(api.Reason.length, try c.wait()); // not refused: it waited for room
    e.cancel(1);
    try eq(api.Reason.cancelled, try a.wait());
    x.stop();
    try expectSolo(&pb, b.tokens.items);
    try expectSolo(&pc, c.tokens.items);
    try expectSolo(&pa, a.tokens.items);
    try eq(@as(usize, 40), b.tokens.items.len);
    try eq(@as(usize, 30), c.tokens.items.len);
    try t.expect(x.back.failed == null);
    try eq(@as(usize, 0), x.steps.refused);
    try eq(@as(u64, 3), pool.reserves);
    try eq(pool.blocks, pool.freeBlocks());
    try eq(pool.reserved_blocks, pool.released_blocks);
    // C's region was taken only after B's came back: the third reservation follows B's release
    var reserves: usize = 0;
    var third: ?usize = null;
    var b_released: ?usize = null;
    for (x.steps.log.items, 0..) |ev, i| {
        if (ev.kind == .reserve) {
            reserves += 1;
            if (reserves == 3) third = i;
        }
        if (ev.kind == .release and b_released == null and ev.slot == 1 and ev.generation == 1) b_released = i;
    }
    try t.expect(third != null and b_released != null and b_released.? < third.?);
    std.debug.print("four_user_profile pool + interleave admission: pool {d} blocks = live {d} + B {d}; C (1 block) waited for B's first step, then for room; reserves {d}, refused {d}, reserved {d} = released {d}; B 40 and C 30 tokens = solo\n", .{ pool.blocks, ba, bb, pool.reserves, x.steps.refused, pool.reserved_blocks, pool.released_blocks });
}

test "four_user_profile USERS=4 pool + interleave + a schema request: four arriving together through the real LaneHost; each reply is its solo reply (4-way solo = together) and the interleaved run's replies are the serial run's" {
    var vt: Steps = .{};
    defer vt.deinit();
    var on: Back = .{ .steps = &vt, .count = 4, .capacity = cap, .reach = cap, .interleave = true };
    try t.expect(on.backend().vtable.prefill_step != null); // prefill_interleave's stepped vtable ...
    try t.expect(on.backend().vtable.first_masked != null); // ... carries structured_users's masked first (the four_user_profile join)
    var off: Back = .{ .steps = &vt, .count = 4, .capacity = cap, .reach = cap };
    try t.expect(off.backend().vtable.prefill_step == null);
    try t.expect(off.backend().vtable.first_masked != null);

    var together: [4]Reply = @splat(.{});
    var st: Stats = .{};
    try serve(true, null, &together, &st);
    for (together[0..3], four[0..3]) |r, q| {
        try eq(api.Reason.length, r.reason.?);
        try eq(@as(usize, q.max_tokens), r.n);
        try expectSolo(q.prompt, r.got());
    }
    try eq(api.Reason.stop, together[3].reason.?); // the schema's end
    try t.expectEqualSlices(u32, &schema_reply, together[3].got());
    try t.expect(!st.failed);
    try t.expect(st.pool_back);
    try eq(@as(u64, 4), st.reserves);
    try eq(st.reserved, st.released);
    try eq(@as(usize, 0), st.refused);
    try eq(@as(usize, 3), st.passes); // the plain prompts went in a step at a time
    try eq(@as(usize, 0), st.whole);
    try eq(@as(usize, 1), st.deferred); // the schema request's whole pass, its first draw under its mask
    try eq(@as(usize, 10), st.gaps); // B's and C's prompts: a live round between every two of their six steps
    try eq(@as(usize, 4), st.max_rows); // a round with all four users
    try eq(@as(usize, 12), st.masked_draws);
    try eq(@as(usize, 1), st.closes);

    var serial: [4]Reply = @splat(.{});
    var ss: Stats = .{};
    try serve(false, null, &serial, &ss);
    try eq(@as(usize, 0), ss.passes);
    try eq(@as(usize, 3), ss.whole);
    try eq(@as(usize, 1), ss.deferred);
    try eq(@as(usize, 0), ss.gaps);
    try eq(@as(usize, 4), ss.max_rows);
    try t.expect(!ss.failed and ss.pool_back and ss.refused == 0);
    for (together, serial) |x, y| { // interleaved = serial
        try eq(x.reason, y.reason);
        try t.expectEqualSlices(u32, x.got(), y.got());
    }

    for (0..4) |i| { // 4-way solo = together
        var solo: [4]Reply = @splat(.{});
        var so: Stats = .{};
        try serve(true, i, &solo, &so);
        try eq(together[i].reason, solo[i].reason);
        try t.expectEqualSlices(u32, together[i].got(), solo[i].got());
        try eq(@as(usize, 1), so.max_rows);
        try t.expect(!so.failed and so.pool_back and so.refused == 0);
    }
    std.debug.print("four_user_profile USERS=4 pool + interleave + schema: together {d}/{d}/{d}/{d} tokens (length x3, stop), = serial, = solo x4; passes {d}, gaps {d}, max rows {d}, masked draws {d}, reserves {d} reserved {d} = released {d}\n", .{ together[0].n, together[1].n, together[2].n, together[3].n, st.passes, st.gaps, st.max_rows, st.masked_draws, st.reserves, st.reserved, st.released });
}
