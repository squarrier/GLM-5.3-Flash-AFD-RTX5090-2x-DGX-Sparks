//! user_drafts (TF_AFD_USERS_DRAFTS=1, mode A) CPU tests (step test-user_drafts, filter "user_drafts"): MTP drafts with
//! TF_AFD_USERS 2 or 4 through the real LaneHost, lane core and users backend (native/glm_cuda_users.zig) with its lone
//! driver, over a Steps stand-in whose tokens depend on each slot's whole history (lanes.fake.next, four_user_profile's form) and
//! whose drafted rounds take cuda_afd_drafts.mtpWith's shape: drafts proposed (every fifth position's refused), a
//! verify that keeps through the first draft it did not draw, the kept tokens to the hook. The stand-in's MTP head
//! counts the rows its cache absorbed and the pairs it owes, and refuses a drafted loop whose carried state is stale
//! (StaleMtp) or a carry out of step. No CUDA numerics; hardware equality is the lab's.
//! Ash Hart authored TensorFold and its lane core (the lone driver and its yield hand-over); four_user_profile, shared_pool, prefill_interleave
//! and structured_users are Claude Opus 5.5 via Hermes, as their files say; this file: Claude Opus 5.5 (Anthropic; planning and
//! execution), via Hermes.
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
const envcall = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

/// The served capacity at the 262,144 window (concurrent_users's), the backend's ContextFull bound.
const cap: usize = 262208;

/// Holds the engine thread inside its next drafted round once armed, until opened (four_user_profile's Gate).
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

/// A slot: its generation, its cache (history), a pass under way (prefill_interleave), and its MTP head: the rows its cache
/// absorbed (`mtp`), the pairs it owes (`owed`: the last ends at the pending token `last`), `head` while the start pair
/// is a lone opening's (the Engine's last_hidden in the family).
const Slot = struct {
    generation: u64 = 0,
    history: std.ArrayList(u32) = .empty,
    step: ?usize = null,
    track: Track = .off,
    mtp: usize = 0,
    owed: usize = 0,
    last: u32 = 0,
    head: bool = false,
};

/// four_user_profile's Steps (whole and stepped prompt passes, shared_pool's reserve, rounds) with the drafting calls the lone driver
/// makes (track, carried, drafted, carry) over a stand-in MTP head.
const Steps = struct {
    slots: [4]Slot = @splat(.{}),
    chunks: usize = 3,
    pool: ?*P.Pool = null,
    gate: ?*Gate = null,
    depth: usize = 3,
    /// pairs a slot's backlog holds before a flush (the family's 256; small here so a shared phase flushes)
    backlog: usize = 8,
    /// a drafted round fails (an exchange) once this many drafted rounds ran
    fail_after: ?usize = null,
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
    refused: usize = 0,

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

    fn claim(s: *Steps, key: Key, ids: []const u32) !void {
        const sl = &s.slots[key.slot];
        if (key.generation <= sl.generation) return error.StaleGeneration;
        sl.generation = key.generation;
        sl.history.clearRetainingCapacity();
        try sl.history.appendSlice(gpa, ids);
    }

    /// A tracked pass absorbed every prompt row but the last (qualification_21e21bc0's prefill absorb); its start pair ends at `first`.
    fn passEnd(s: *Steps, slot: u8, first: u32) void {
        const sl = &s.slots[slot];
        if (sl.track == .off) return;
        sl.mtp = sl.history.items.len - 1;
        sl.owed = 1;
        sl.last = first;
        sl.head = sl.track == .lone;
    }

    pub fn prefill(s: *Steps, key: Key, ids: []const u32) !u32 {
        try s.claim(key, ids);
        s.whole += 1;
        const h = s.slots[key.slot].history.items;
        const first = lanes.fake.next(h, null, h.len);
        s.passEnd(key.slot, first);
        return first;
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
        const first = lanes.fake.next(sl.history.items, null, sl.history.items.len);
        s.passEnd(key.slot, first);
        return first;
    }

    pub fn position(s: *Steps, key: Key) usize {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        return s.slots[key.slot].history.items.len;
    }

    /// The slot's pass and its carried MTP state end with its owner; (shared_pool) its region's blocks back to the pool.
    pub fn release(s: *Steps, key: Key) void {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        const sl = &s.slots[key.slot];
        sl.step = null;
        sl.track = .off;
        sl.owed = 0;
        sl.head = false;
        if (s.pool) |p| _ = p.release(key.slot);
    }

    pub fn round(s: *Steps, rows: []const Row, out: []u32) !void {
        for (rows) |row| {
            const sl = &s.slots[row.key.slot];
            if (sl.generation != row.key.generation) return error.StaleGeneration;
            if (sl.history.items.len != row.position) return error.PositionMismatch;
            if (sl.step != null) return error.FillingInRound;
        }
        s.rounds += 1;
        s.max_rows = @max(s.max_rows, rows.len);
        for (rows, out) |row, *o| {
            const sl = &s.slots[row.key.slot];
            try sl.history.append(gpa, row.token);
            o.* = lanes.fake.next(sl.history.items, null, sl.history.items.len);
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

    // ---- the drafting calls (cuda_afd_users.zig's track, carried, drafted, carry) ----

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

    /// Inside the loop's hook: the round's kept rows become the pairs owed (they must be exactly the last verify's).
    pub fn carry(s: *Steps, key: Key, kept: []const u32) !void {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (kept.len != sl.owed or kept[kept.len - 1] != sl.last) return error.CarryOutOfStep;
        s.carries += 1;
    }

    /// mtpWith's shape: the start pair (a resume absorbs its owed pairs but the last first), then rounds of drafts
    /// (absorbing what is owed) and a verify of [pending, drafts] kept through the first refused draft.
    pub fn drafted(s: *Steps, key: Key, pending: u32, count: usize, out: []u32, hook: *const DraftHook) !DraftRan {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (sl.track == .off) return error.NotTracked;
        if (count == 0 or count > out.len) return error.Count;
        // never stale: the owed pairs end at the pending token and cover exactly the cache rows the head lacks
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
            if (s.gate) |g| g.at();
            if (s.fail_after) |k| if (s.drafted_rounds >= k) return error.InjectedExchangeFailure;
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
            try sl.history.append(gpa, cur); // the kept rows committed: [cur, d1 .. d(keep-1)]
            try sl.history.appendSlice(gpa, drafts[0 .. keep - 1]);
            sl.owed = keep; // the kept rows' pairs, owed to the next draft (or carried)
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

    fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
};

/// The reply the undrafted path gives: each token lanes.fake.next of the prompt and every token before it.
fn expectSolo(prompt: []const u32, got: []const u32) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, prompt);
    for (got) |x| {
        try eq(lanes.fake.next(history.items, null, history.items.len), x);
        try history.append(gpa, x);
    }
}

const Opts = struct { drafts: bool = true, interleave: bool = false, pool: ?*P.Pool = null, draft_reach: usize = cap };

/// The users backend over Steps at TF_AFD_USERS=n, the lane core and the real LaneHost with the switch on (the lone
/// driver installed, as afd_attn_users.run installs it) or off (as today).
const Served = struct {
    steps: Steps = .{},
    back: Back = undefined,
    cfg: lanes.Config = undefined,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: api.LaneHost = undefined,
    up: bool = false,

    fn init(x: *Served, n: u8, o: Opts) !void {
        x.steps.pool = o.pool;
        x.back = .{ .steps = &x.steps, .count = n, .capacity = cap, .reach = cap, .interleave = o.interleave, .drafts = o.drafts, .draft_reach = o.draft_reach };
        x.cfg = try lanes.Config.init(gpa, x.back.facts(), 1, 0);
        x.core = lanes.Engine.init(gpa, &x.cfg, x.back.backend(), x.clock.clock());
        const admit: ?api.Admit = if (o.pool) |p| .{ .ctx = p, .fits = P.Pool.admitFits } else null;
        x.host = api.LaneHost.init(gpa, io, &x.core, .{ .lanes = n, .context_window = 262144, .drafts = if (o.drafts) "auto" else "off", .greedy_users = true, .users_drafts = o.drafts, .admit = admit });
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

/// A request as the users server builds it with the switch on: greedy, `drafts` unless the request said draft:false.
fn req(prompt: []const u32, max: u32) api.Request {
    return .{ .prompt = prompt, .max_tokens = max, .drafts = true };
}

// ---- the switch ------------------------------------------------------------------------------------------------------

test "user_drafts switch: TF_AFD_USERS_DRAFTS unset or 0 off, 1 on; plan refuses it without users, without a policy, a bad value" {
    try eq(false, try scheduling.parseDrafts(null));
    try eq(false, try scheduling.parseDrafts("0"));
    try eq(true, try scheduling.parseDrafts("1"));
    for ([_][]const u8{ "", "2", "00", "01", "true", "on", " 1", "1 ", "-1" }) |v| try t.expectError(error.BadAfdUsersDrafts, scheduling.parseDrafts(v));
    const names = [_][*:0]const u8{ "TF_AFD_USERS", "TF_AFD_USERS_DRAFTS", "TF_AFD_MTP", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE" };
    var old: [names.len]?[:0]u8 = @splat(null);
    for (names, &old) |n, *o| o.* = if (std.c.getenv(n)) |v| try gpa.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        for (names, old) |n, o| {
            if (o) |v| {
                _ = envcall.setenv(n, v, 1);
                gpa.free(v);
            } else _ = envcall.unsetenv(n);
        }
    }
    for (names) |n| _ = envcall.unsetenv(n);
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const auto = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    const off = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "0", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    _ = envcall.setenv("TF_AFD_USERS", "2", 1);
    _ = envcall.setenv("TF_AFD_MTP", "1", 1);
    try t.expectError(error.UsersDraftsUnsupported, served.plan(a, auto, cfg)); // unset: refused as before
    try eq(false, (try served.plan(a, off, cfg)).users_drafts);
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "0", 1);
    try t.expectError(error.UsersDraftsUnsupported, served.plan(a, auto, cfg));
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    const p = try served.plan(a, auto, cfg);
    try eq(true, p.users_drafts);
    try eq(@as(u8, 2), p.users);
    try t.expect(p.drafts.on());
    try t.expectError(error.UsersDraftsNeedPolicy, served.plan(a, off, cfg));
    _ = envcall.unsetenv("TF_AFD_MTP");
    try t.expectError(error.DraftsNeedTfAfdMtp, served.plan(a, auto, cfg)); // the policy needs the MTP head, as today
    _ = envcall.setenv("TF_AFD_MTP", "1", 1);
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "yes", 1);
    try t.expectError(error.BadAfdUsersDrafts, served.plan(a, auto, cfg));
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    _ = envcall.unsetenv("TF_AFD_USERS");
    const one = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144" });
    try t.expectError(error.UsersDraftsNeedsUsers, served.plan(a, one, cfg));
    _ = envcall.setenv("TF_AFD_USERS", "4", 1);
    const four = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    try eq(true, (try served.plan(a, four, cfg)).users_drafts);
}

// ---- a lone user -------------------------------------------------------------------------------------------------------

/// One request alone on a fresh served path (switch on or off); its reply, its stats, the stand-in's counters.
fn alone(o: Opts, r: api.Request, box: *Box, x: *Served) !void {
    try x.init(2, o);
    const e = x.host.engine();
    try e.submit(1, &r, box.sink());
    try x.start();
    _ = try box.wait();
    x.stop();
}

test "user_drafts lone user: its drafted reply = the undrafted reply byte for byte, refused drafts included; switch off = today's path" {
    var on_box: Box = .{};
    defer on_box.deinit();
    var on: Served = .{};
    defer on.deinit();
    try alone(.{}, req(&pa, 120), &on_box, &on);
    try eq(api.Reason.length, on_box.done.?);
    try eq(@as(usize, 120), on_box.tokens.items.len);
    try expectSolo(&pa, on_box.tokens.items);
    try t.expect(on.steps.drafted_rounds > 0 and on.steps.refused_drafts > 0 and on.steps.proposed > on.steps.refused_drafts);
    try t.expect(on.steps.drafted_rounds < 120); // drafts kept: fewer verify forwards than tokens
    try eq(@as(u64, 1), on.back.counts.runs);
    try eq(@as(u64, 0), on.back.counts.resumes);
    try eq(@as(u64, 0), on.back.counts.yields);
    try eq(@as(usize, 0), on.steps.rounds); // no shared round: the whole reply in the lone driver
    try eq(@as(usize, 0), on.steps.stale);
    try t.expect(on_box.drafted > 0);
    try eq(@as(usize, 0), on.owners());
    try eq(@as(?anyerror, null), on.back.failed);
    // the switch off: no driver, no track, every round shared (one row): the same bytes
    var off_box: Box = .{};
    defer off_box.deinit();
    var off: Served = .{};
    defer off.deinit();
    try alone(.{ .drafts = false }, .{ .prompt = &pa, .max_tokens = 120, .drafts = false }, &off_box, &off);
    try t.expectEqualSlices(u32, on_box.tokens.items, off_box.tokens.items);
    try eq(@as(usize, 0), off.steps.tracks);
    try eq(@as(usize, 0), off.steps.drafted_rounds);
    try eq(@as(usize, 119), off.steps.rounds);
    try eq(@as(u64, 0), off_box.drafted);
    try t.expect(off.host.lone == null);
}

test "user_drafts past the drafted reach: the lone driver hands the stream over after its prompt pass, untracked, never resumed; reply = solo" {
    var box: Box = .{};
    defer box.deinit();
    var x: Served = .{};
    defer x.deinit();
    try alone(.{ .draft_reach = 100 }, req(&pa, 120), &box, &x);
    try expectSolo(&pa, box.tokens.items);
    try eq(@as(usize, 120), box.tokens.items.len);
    try eq(@as(usize, 0), x.steps.drafted_rounds);
    try eq(@as(usize, 0), x.steps.tracks);
    try eq(@as(u64, 1), x.back.counts.yields);
    try eq(@as(u64, 0), x.back.counts.resumes);
    try eq(@as(usize, 119), x.steps.rounds);
}

// ---- lone -> shared -> lone ------------------------------------------------------------------------------------------

test "user_drafts lone -> shared -> lone mid-reply: an arrival hands it over with its kept rows carried; at the other's end it drafts again; both = solo" {
    var gate: Gate = .{};
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    var x: Served = .{};
    defer x.deinit();
    try x.init(2, .{});
    x.steps.gate = &gate;
    gate.armed.store(true, .release);
    const e = x.host.engine();
    const ra = req(&pa, 160);
    const rb = req(&pb, 40);
    try e.submit(1, &ra, ba.sink());
    try x.start();
    try gate.waitEntered(); // A drafts alone, held inside its first drafted round
    try e.submit(2, &rb, bb.sink()); // queued at once: the round's hook asks and hands A over
    gate.open.store(true, .release);
    _ = try bb.wait();
    _ = try ba.wait();
    x.stop();
    try expectSolo(&pa, ba.tokens.items);
    try expectSolo(&pb, bb.tokens.items);
    try eq(@as(usize, 160), ba.tokens.items.len);
    try eq(@as(usize, 40), bb.tokens.items.len);
    try t.expect(x.back.counts.yields >= 1); // A handed over at B's arrival
    try t.expect(x.steps.carries >= 1); // with its last verify's kept rows
    try t.expect(x.steps.max_rows == 2); // A and B shared undrafted rounds
    try t.expect(x.back.counts.resumes >= 1); // B's end: A back to its drafted rounds
    try t.expect(x.steps.resumed_loops >= 1);
    try t.expect(x.steps.flushes >= 1); // the shared phase outgrew a backlog: absorbed, nothing dropped
    try eq(@as(usize, 0), x.steps.stale);
    try eq(@as(usize, 0), x.owners());
    try eq(@as(?anyerror, null), x.back.failed);
}

// ---- concurrent users ------------------------------------------------------------------------------------------------

/// `rs` submitted before the host starts (arriving together) on a fresh served path; each reply.
fn together(n: u8, o: Opts, rs: []const api.Request, boxes: []Box, x: *Served) !void {
    try x.init(n, o);
    const e = x.host.engine();
    for (rs, 0..) |*r, i| try e.submit(@intCast(i + 1), r, boxes[i].sink());
    try x.start();
    for (boxes) |*b| _ = try b.wait();
    x.stop();
}

test "user_drafts C2 on different prompts: each reply = its solo reply; shared rounds undrafted, the last left drafts again" {
    const rs = [_]api.Request{ req(&pa, 90), req(&pc, 30) };
    var boxes: [2]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var x: Served = .{};
    defer x.deinit();
    try together(2, .{}, &rs, &boxes, &x);
    for (rs, boxes) |r, b| {
        try expectSolo(r.prompt, b.tokens.items);
        try eq(@as(usize, r.max_tokens), b.tokens.items.len);
    }
    try eq(@as(usize, 2), x.steps.max_rows);
    try t.expect(x.back.counts.resumes >= 1);
    try t.expect(x.steps.drafted_rounds > 0);
    try eq(@as(usize, 0), x.steps.stale);
    try eq(@as(usize, 0), x.owners());
}

test "user_drafts USERS=4 with pool + interleave: four replies = solo; the pool's regions all back; the last left drafts again" {
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    const rs = [_]api.Request{ req(&pa, 120), req(&pb, 40), req(&pc, 30), req(&pd, 20) };
    var blocks: usize = 0;
    for (rs) |r| blocks += P.blocksFor(P.need(r.prompt.len, r.max_tokens));
    var pool = try P.Pool.init(gpa, &fx.m, blocks * P.block, 8, true, cap);
    defer pool.deinit(gpa);
    var boxes: [4]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var x: Served = .{};
    defer x.deinit();
    try together(4, .{ .interleave = true, .pool = &pool }, &rs, &boxes, &x);
    for (rs, boxes) |r, b| {
        try expectSolo(r.prompt, b.tokens.items);
        try eq(@as(usize, r.max_tokens), b.tokens.items.len);
    }
    try t.expect(x.steps.passes >= 4); // prefill_interleave: every prompt went in a step at a time
    try t.expect(x.steps.max_rows >= 2);
    try t.expect(x.back.counts.resumes >= 1);
    try eq(@as(usize, 0), x.steps.stale);
    try eq(pool.blocks, pool.freeBlocks());
    try eq(@as(usize, 0), x.owners());
    // the switch off on the same four: the same bytes
    var off_boxes: [4]Box = @splat(.{});
    defer for (&off_boxes) |*b| b.deinit();
    var pool2 = try P.Pool.init(gpa, &fx.m, blocks * P.block, 8, true, cap);
    defer pool2.deinit(gpa);
    var y: Served = .{};
    defer y.deinit();
    const off = [_]api.Request{ .{ .prompt = &pa, .max_tokens = 120, .drafts = false }, .{ .prompt = &pb, .max_tokens = 40, .drafts = false }, .{ .prompt = &pc, .max_tokens = 30, .drafts = false }, .{ .prompt = &pd, .max_tokens = 20, .drafts = false } };
    try together(4, .{ .drafts = false, .interleave = true, .pool = &pool2 }, &off, &off_boxes, &y);
    for (boxes, off_boxes) |b, ob| try t.expectEqualSlices(u32, b.tokens.items, ob.tokens.items);
    try eq(@as(usize, 0), y.steps.tracks);
}

fn fixtureModel() !struct { parsed: std.json.Parsed(std.json.Value), m: glm.model.Model } {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    errdefer parsed.deinit();
    var m = try glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
    try m.cut(5);
    m.mtp_layers = 0;
    return .{ .parsed = parsed, .m = m };
}

// ---- cancel and failure while drafting -------------------------------------------------------------------------------

test "user_drafts cancel while drafting: the request ends cancelled, its slot released; the next request = solo" {
    var gate: Gate = .{};
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    var x: Served = .{};
    defer x.deinit();
    try x.init(2, .{});
    x.steps.gate = &gate;
    gate.armed.store(true, .release);
    const e = x.host.engine();
    const ra = req(&pa, 160);
    try e.submit(1, &ra, ba.sink());
    try x.start();
    try gate.waitEntered();
    e.cancel(1);
    gate.open.store(true, .release);
    try eq(api.Reason.cancelled, try ba.wait());
    try t.expect(ba.tokens.items.len < 160);
    try expectSolo(&pa, ba.tokens.items); // what it sent before the cancel is its solo reply's start
    const rb = req(&pc, 30);
    try e.submit(2, &rb, bb.sink());
    try eq(api.Reason.length, try bb.wait());
    x.stop();
    try expectSolo(&pc, bb.tokens.items);
    try eq(@as(usize, 0), x.owners());
    try eq(@as(?anyerror, null), x.back.failed);
    try eq(@as(usize, 0), x.steps.stale);
    for (x.steps.slots) |sl| try eq(Track.off, sl.track);
}

test "user_drafts failure while drafting: the request fails in words, the backend latches it, the slot released; a later request refused" {
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    var x: Served = .{};
    defer x.deinit();
    try x.init(2, .{});
    x.steps.fail_after = 2;
    const e = x.host.engine();
    const ra = req(&pa, 160);
    try e.submit(1, &ra, ba.sink());
    try x.start();
    try eq(api.Reason.failed, try ba.wait());
    try t.expect(std.mem.indexOf(u8, &ba.message, "InjectedExchangeFailure") != null);
    try expectSolo(&pa, ba.tokens.items);
    try eq(@as(?anyerror, error.InjectedExchangeFailure), x.back.failed);
    try eq(@as(usize, 0), x.owners());
    const rb = req(&pc, 30);
    try e.submit(2, &rb, bb.sink());
    try eq(api.Reason.failed, try bb.wait()); // refused before any device call, as after a failed shared round
    x.stop();
    try eq(@as(usize, 0), bb.tokens.items.len);
    try eq(@as(usize, 0), x.owners());
}

// ---- TF_GLM_SAMPLING=1 -------------------------------------------------------------------------------------------------

test "user_drafts TF_GLM_SAMPLING=1: the single-user rule; greedy requests draft as unset, a sampled request is never drafted" {
    const old = if (std.c.getenv("TF_GLM_SAMPLING")) |v| try gpa.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer if (old) |v| {
        _ = envcall.setenv("TF_GLM_SAMPLING", v, 1);
        gpa.free(v);
    } else {
        _ = envcall.unsetenv("TF_GLM_SAMPLING");
    };
    _ = envcall.setenv("TF_GLM_SAMPLING", "1", 1);
    var b1: Box = .{};
    defer b1.deinit();
    var x: Served = .{};
    defer x.deinit();
    try alone(.{}, req(&pa, 120), &b1, &x);
    try expectSolo(&pa, b1.tokens.items);
    try t.expect(x.steps.drafted_rounds > 0);
    try t.expect(!x.back.lone(gpa).sampled); // the driver never takes a sampled stream (api.Lone.sampled)
    var b2: Box = .{};
    defer b2.deinit();
    var y: Served = .{};
    defer y.deinit();
    var sampled = req(&pa, 20);
    sampled.sampling = .{ .temperature = 0.7, .seed = 7 };
    try alone(.{}, sampled, &b2, &y);
    try eq(api.Reason.failed, b2.done.?); // the users backend refuses it (GreedyOnly) without drafting
    try t.expect(std.mem.indexOf(u8, &b2.message, "GreedyOnly") != null);
    try eq(@as(usize, 0), y.steps.drafted_rounds);
    try eq(@as(u64, 0), y.back.counts.runs);
}

// ---- recorded real users (glm.afd_users.Users with the MTP head; launches recorded, the exchange a stand-in) ----------

const cuda = @import("cuda");
const proto = cuda.xchg.proto;

/// The attention node recorded with its MTP head (cuda_afd_drafts_test.zig's rig from the tests' fixture, cut to 5
/// layers, the MTP layer kept; TF_AFD_MTP's provider), each exchange's header kept as (layer, rows).
const MRec = struct {
    parsed: std.json.Parsed(std.json.Value),
    m: glm.model.Model,
    w: glm.views.Weights,
    e: *glm.engine.Engine,
    a: glm.afd_attn.Afd,
    at: glm.afd_engine.Attn,
    space: glm.trace.Space,
    tr: glm.trace.Trace,
    f: glm.fwd.Fwd,
    region: []align(4096) u8,
    status: []u32,
    node: cuda.xchg.Node,
    posts: usize = 0,
    hdrs: [1024][2]i32 = undefined,

    fn init(r: *MRec) !void {
        r.posts = 0;
        r.parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
        r.m = try glm.model.Model.parse(gpa, r.parsed.value.object.get("config_json").?.string, 1, 0);
        const mtp = r.m.mtp_layers;
        try r.m.cut(5);
        r.m.mtp_layers = if (mtp > 0) mtp else 1;
        r.w = try glm.views.rank(gpa, &r.m);
        r.e = try glm.engine.Engine.create(gpa, &r.m, &r.w, 2560, 8, 64, false);
        r.a = try glm.afd_attn.Afd.initWith(&r.m, 64, true);
        r.space = .{ .gpa = gpa };
        try glm.views.walk(&r.w, glm.views.Path.init("w"), &r.space, glm.trace.placeAll);
        try r.e.walk(&r.space, glm.trace.placeAll);
        try r.e.walkTmp(&r.space, glm.trace.placeAll);
        r.e.tmp.space = &r.space;
        try r.space.place("b.afd_recv", &r.a.afd_recv);
        r.tr = glm.trace.Trace.init(gpa);
        const settings: proto.Settings = .{ .layout = r.a.layout, .max_rows = 8, .ports = 1 };
        r.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast((proto.Region{ .slot = settings.slotBytes() }).total()));
        @memset(r.region, 0);
        r.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(r.status, 0);
        r.node = try cuda.xchg.Node.init(gpa, io, .attn, .{ .settings = settings, .warn_failures = true }, .{ .region = r.region, .region_dev = @intFromPtr(r.region.ptr), .status = r.status, .status_dev = @intFromPtr(r.status.ptr) });
        r.a.wire = .{ .node = &r.node, .host = .{ .ctx = r, .post = post, .take = take } };
        r.f = glm.fwd.Fwd.recording(&r.m, &r.tr);
        r.at = try glm.afd_engine.Attn.init(r.e, &r.a);
    }

    fn deinit(r: *MRec) void {
        r.node.deinit();
        gpa.free(r.region);
        gpa.free(r.status);
        r.tr.deinit();
        r.space.deinit();
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.parsed.deinit();
    }

    fn post(p: *anyopaque, _: *cuda.xchg.Node, h: *const [proto.HDR_BYTES]u8, pos: usize, n: usize) !void {
        const r: *MRec = @ptrCast(@alignCast(p));
        const hdr = try proto.Header.read(h);
        if (hdr.layer != @as(i32, @intCast(pos)) or hdr.rows != @as(i32, @intCast(n))) return error.HeaderNotPos;
        if (r.posts < r.hdrs.len) r.hdrs[r.posts] = .{ hdr.layer, hdr.rows };
        r.posts += 1;
    }

    fn take(_: *anyopaque, _: *cuda.xchg.Node, _: usize, _: usize) !void {}
};

/// Every draw answered with token 5 at probability 1 (each draft is drawn again by the verify: all kept).
const Picks = struct {
    n: usize = 0,
    fn pick(ctx: *anyopaque, logits: glm.trace.T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        _ = .{ logits, at };
        const p: *Picks = @ptrCast(@alignCast(ctx));
        @memset(out, 5);
        if (probs) |x| @memset(x, 1.0);
        p.n += 1;
    }
};

const UHook = glm.afd_users.Users.DraftHook;

/// The lone driver's hook in small: after `yield_after` rounds the loop yields with the round's kept rows carried.
const Yield = struct {
    u: *glm.afd_users.Users,
    key: Key,
    yield_after: usize = 1,
    rounds: usize = 0,
    fn kept(p: *anyopaque, tokens: []const u32, _: usize, keep: usize) anyerror!bool {
        const y: *Yield = @ptrCast(@alignCast(p));
        y.rounds += 1;
        if (y.rounds < y.yield_after or tokens.len != keep) return false;
        try y.u.carry(y.key, tokens);
        return true;
    }
};

/// The rows the slot's MTP head lacks: its State's rows less what its cache absorbed (held draft rows aside).
fn lacking(st: anytype) usize {
    return st.pos - (st.mtp_len - st.mtp_drafted);
}

test "user_drafts recorded real users: a tracked slot's carried MTP state stays in step through its prompt pass, shared rounds, a yielding drafted loop, flushes and a resume" {
    var r: MRec = undefined;
    try r.init();
    defer r.deinit();
    var picks: Picks = .{};
    r.tr.picker = .{ .ctx = &picks, .pick = Picks.pick };
    var u = try glm.afd_users.Users.init(gpa, &r.at, &r.f, 2);
    defer u.deinit(gpa);
    try t.expectError(error.BacklogRows, u.useDrafts(.{ .depth = .{ .most = 3, .fixed = true } }, 7));
    try u.useDrafts(.{ .depth = .{ .most = 3, .fixed = true } }, 8); // 8 pairs: a shared phase flushes
    try u.walk(&r.space, glm.trace.placeAll);
    var ids: [10]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + i);
    const k0: Key = .{ .slot = 0, .generation = 1 };
    const k1: Key = .{ .slot = 1, .generation = 1 };
    const st1 = &u.states[0];
    u.track(1, .seed);
    var p1 = try u.prefill(k1, &ids);
    try eq(@as(usize, 10), st1.pos);
    try eq(@as(usize, 9), st1.mtp_len); // the prompt absorbed but its last row (qualification_21e21bc0's prefill)
    try eq(@as(usize, 1), u.backlogLen(k1)); // that row and the first token: the start pair
    try t.expect(u.carried(k1));
    var p0 = try u.prefill(k0, &ids); // slot 0 untracked: its pass as today, nothing absorbed
    try eq(@as(usize, 0), r.e.st.mtp_len);
    try t.expect(!u.carried(k0));
    try eq(@as(usize, 1), lacking(st1));
    for (0..3) |_| { // shared rounds: each adds the row's pair to slot 1's backlog only
        var out: [2]u32 = undefined;
        const rows = [_]Row{ .{ .key = k0, .token = p0, .position = r.e.st.pos }, .{ .key = k1, .token = p1, .position = st1.pos } };
        try u.round(&rows, &out);
        p0 = out[0];
        p1 = out[1];
    }
    try eq(@as(usize, 13), st1.pos);
    try eq(@as(usize, 4), u.backlogLen(k1));
    try eq(lacking(st1), u.backlogLen(k1));
    try eq(@as(usize, 0), r.e.st.mtp_len);
    // a drafted loop on slot 1 that yields after its first round: its pairs but the last absorbed first, the round's
    // kept rows carried
    var y: Yield = .{ .u = &u, .key = k1 };
    const hook: UHook = .{ .ctx = &y, .kept = Yield.kept };
    var out: [9]u32 = undefined;
    const ran = try u.drafted(k1, p1, 9, &out, &hook);
    try eq(@as(u64, 1), ran.rounds);
    try eq(@as(u64, 3), ran.drafted);
    try eq(@as(u64, 3), ran.accepted);
    try eq(@as(usize, 17), st1.pos);
    try eq(@as(usize, 4), u.backlogLen(k1));
    try eq(lacking(st1), u.backlogLen(k1));
    try t.expect(u.carried(k1));
    p1 = out[ran.tokens - 1];
    for (0..5) |_| { // the shared phase again: a full backlog (8 pairs) absorbed after its round, the last kept
        var o: [2]u32 = undefined;
        const rows = [_]Row{ .{ .key = k0, .token = p0, .position = r.e.st.pos }, .{ .key = k1, .token = p1, .position = st1.pos } };
        try u.round(&rows, &o);
        p0 = o[0];
        p1 = o[1];
        try eq(lacking(st1), u.backlogLen(k1));
    }
    try eq(@as(usize, 22), st1.pos);
    try eq(@as(usize, 2), u.backlogLen(k1));
    try eq(@as(usize, 0), st1.mtp_drafted); // the held draft rows trimmed by the flush's absorb
    // the resume: the rest absorbed, the loop runs to its count
    var none: Yield = .{ .u = &u, .key = k1, .yield_after = std.math.maxInt(usize) };
    const hook2: UHook = .{ .ctx = &none, .kept = Yield.kept };
    var out2: [5]u32 = undefined;
    const ran2 = try u.drafted(k1, p1, 5, &out2, &hook2);
    try eq(@as(usize, 5), ran2.tokens);
    try eq(@as(usize, 26), st1.pos);
    // a stale start refused before any launch: the pending token out of step with the carried pairs
    try u.carry(k1, out2[1..5]);
    const posts = r.posts;
    try t.expectError(error.PendingOutOfStep, u.drafted(k1, out2[4] +% 1, 3, &out, &hook2));
    try eq(posts, r.posts);
    try t.expect(!u.carried(k1)); // a failed start leaves nothing to resume
    u.release(k1);
    try eq(@as(usize, 0), u.backlogLen(k1));
    try t.expect(!u.carried(k1));
}

test "user_drafts forecast: the backlog bytes the users forecast adds to its rows are exactly what useDrafts adds to Users.bytes(); unset adds none" {
    var r: MRec = undefined;
    try r.init();
    defer r.deinit();
    const served_users = @import("afd_attn_users.zig");
    for ([_]u8{ 2, 4 }) |n| {
        var u = try glm.afd_users.Users.init(gpa, &r.at, &r.f, n);
        defer u.deinit(gpa);
        const before = try u.bytes();
        try u.useDrafts(.{}, glm.afd_users_drafts.rows);
        const after = try u.bytes();
        try eq(before.states, after.states);
        var x = before;
        served_users.withDrafts(&x, &r.m, n, true);
        try eq(after.rows, x.rows);
        try eq(after.total, x.total);
        try eq(@as(u64, n) * glm.afd_users_drafts.rows * r.m.hidden * 2, served_users.draftBytes(&r.m, n, true));
        var y = before;
        served_users.withDrafts(&y, &r.m, n, false);
        try eq(before, y);
    }
}

test "user_drafts recorded real users: a lone user's prompt pass and drafted loop make the single-user drafted request's exchanges, tokens and MTP cache" {
    var ids: [10]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + i);
    // the users path: slot 0 tracked `lone`, its loop from the Engine's last_hidden (the single-user path's calls)
    var r: MRec = undefined;
    try r.init();
    defer r.deinit();
    var picks: Picks = .{};
    r.tr.picker = .{ .ctx = &picks, .pick = Picks.pick };
    var u = try glm.afd_users.Users.init(gpa, &r.at, &r.f, 2);
    defer u.deinit(gpa);
    try u.useDrafts(.{ .depth = .{ .most = 3, .fixed = true } }, glm.afd_users_drafts.rows);
    try u.walk(&r.space, glm.trace.placeAll);
    const k0: Key = .{ .slot = 0, .generation = 1 };
    u.track(0, .lone);
    const first = try u.prefill(k0, &ids);
    try t.expect(!u.carried(k0)); // its start pair is the Engine's last_hidden: no resume from it
    var none: Yield = .{ .u = &u, .key = k0, .yield_after = std.math.maxInt(usize) };
    const hook: UHook = .{ .ctx = &none, .kept = Yield.kept };
    var out: [9]u32 = undefined;
    const ran = try u.drafted(k0, first, 9, &out, &hook);
    // the single-user path on a fresh node: qualification_21e21bc0's absorbing prefill and cuda_afd_drafts.mtp
    var s: MRec = undefined;
    try s.init();
    defer s.deinit();
    var picks2: Picks = .{};
    s.tr.picker = .{ .ctx = &picks2, .pick = Picks.pick };
    s.at.absorb = true;
    const first2 = try s.at.prefill(&s.f, &ids);
    var pol: glm.afd_drafts.DepthPolicy = .{ .most = 3, .fixed = true };
    var out2: [9]u32 = undefined;
    const d = try glm.afd_drafts.mtp(&s.at, &s.f, first2, 9, &out2, &pol, false, null);
    try eq(first2, first);
    try t.expectEqualSlices(u32, d.tokens, out[0..ran.tokens]);
    try eq(@as(u64, d.rounds), ran.rounds);
    try eq(@as(u64, d.drafted), ran.drafted);
    try eq(@as(u64, d.accepted), ran.accepted);
    try eq(s.posts, r.posts);
    try t.expectEqualSlices([2]i32, s.hdrs[0..@min(s.posts, s.hdrs.len)], r.hdrs[0..@min(r.posts, r.hdrs.len)]);
    try eq(s.e.st.pos, r.e.st.pos);
    try eq(s.e.st.mtp_len, r.e.st.mtp_len);
    try eq(s.e.st.mtp_drafted, r.e.st.mtp_drafted);
    try eq(picks2.n, picks.n);
}
