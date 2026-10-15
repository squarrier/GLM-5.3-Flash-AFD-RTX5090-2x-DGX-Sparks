//! shared_verify (TF_AFD_USERS_SHARED_DRAFTS=1): drafted shared verify, token level. The real LaneHost, lane core and users
//! backend (native/glm_cuda_users.zig) over a history-dependent stand-in Steps whose drafted shared rounds, settles and
//! MTP draft requests keep the family's rules (cuda_afd_users_dsv.zig): every reply = its solo reply byte for byte
//! (greedy drafted = undrafted), solo = together, lone <-> drafted shared transitions, per-stream acceptance and
//! rollback, cancel and failure inside a drafted round, the drafted reach, sampled streams (user_sampling's rule), the switch
//! through plan() and default off. The recorded real-Users launch tests are shared_verify_rec_test.zig. Design: @Bizuayeu's
//! shared verify windows (Bizuayeu/TensorFold pr/glm53-flash-cuda 75131b66, Apache-2.0; reimplemented, not copied).
//! Harness after user_drafts's user_drafts_test.zig. Claude Opus 5.5 (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const glm = @import("glm_cuda");
const users = @import("glm_users");
const P = glm.afd_pool;
const scheduling = glm.afd_users_round;
const dsv = glm.afd_users_dsv;
const pp = glm.afd_users_pp;
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
const harness = @import("env_harness.zig"); // FJPANIC: std's startup environment kept whole

/// Saves each name's value before a test changes it (restoreEnv puts them back).
fn saveEnv(names: []const [*:0]const u8, old: []?[:0]u8) !void {
    try harness.detach(); // FJPANIC: libc's environ off std's startup array before the test unsets anything
    for (names, old) |n, *o| o.* = if (std.c.getenv(n)) |v| try gpa.dupeSentinel(u8, std.mem.span(v), 0) else null;
}

/// Restores what saveEnv saved: the old value, or unset when the name was absent.
fn restoreEnv(names: []const [*:0]const u8, old: []const ?[:0]u8) void {
    for (names, old) |n, o| {
        if (o) |v| {
            _ = envcall.setenv(n, v, 1);
            gpa.free(v);
        } else _ = envcall.unsetenv(n);
    }
}

test {
    _ = @import("shared_verify_rec_test.zig"); // shared_verify: the recorded real-Users launch tests
}

/// The served capacity at the 262,144 window (concurrent_users's), the backend's ContextFull bound.
const cap: usize = 262208;
const vocab = lanes.fake.vocab;

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

/// The stand-in's draw of a row: the greedy next token of its history, or a sampled row's keyed draw (its own seed and
/// the drawn token's position, never its slot or round: user_sampling's rule).
fn draw(h: []const u32, s: ?scheduling.Sampling, position: usize) u32 {
    const g = lanes.fake.next(h, null, h.len);
    const x = s orelse return g;
    return @intCast((g + (x.seed +% position *% 31) % 5) % vocab);
}

/// The stand-in head's guess of each next token after `h`: right, but every fifth position's is refused.
fn guess(h: []const u32) u32 {
    const right = lanes.fake.next(h, null, h.len);
    return if (h.len % 5 == 0) (right + 1) % vocab else right;
}

/// A slot: its generation, its cache (history), a pass under way, its sampling, and its MTP head: the positions its
/// cache absorbed (`mtp`), the pairs it owes (`owed`, ending at the pending token `last`), `head` while the start pair
/// is a lone opening's, `inhead` while a drafted shared round's draft request absorbed it (shared_verify), the last wider
/// window's rows until settled, and the window rows a tracked slot's draft request may count.
const Slot = struct {
    generation: u64 = 0,
    history: std.ArrayList(u32) = .empty,
    step: ?usize = null,
    sampling: ?scheduling.Sampling = null,
    track: Track = .off,
    mtp: usize = 0,
    owed: usize = 0,
    last: u32 = 0,
    head: bool = false,
    inhead: bool = false,
    window: [dsv.max_width]u32 = @splat(0),
    rows: usize = 0,
    copied: usize = 0,
};

/// user_drafts's stand-in Steps (prompt passes whole and stepped, shared_pool's reserve, its lone drafted loop, carry and the
/// per-row round) with shared_verify's calls: roundDrafted, settleDrafted and draftDrafted, under the family's rules.
const Steps = struct {
    slots: [4]Slot = @splat(.{}),
    ping_pong: bool = false,
    pipe_rounds: usize = 0,
    chunks: usize = 3,
    pool: ?*P.Pool = null,
    gate: ?*Gate = null,
    /// arm `gate` at the first drafted shared round of two windows once this many drafted rounds ran
    arm_at: ?usize = null,
    depth: usize = 3,
    fail_after: ?usize = null,
    tracks: usize = 0,
    loops: usize = 0,
    resumed_loops: usize = 0,
    unabsorbed: usize = 0,
    carries: usize = 0,
    stale: usize = 0,
    undrafted_rounds: usize = 0,
    dsv_rounds: usize = 0,
    max_rows: usize = 0,
    max_width: usize = 0,
    wide: usize = 0,
    settles: usize = 0,
    replays: usize = 0,
    requests: usize = 0,
    proposed: usize = 0,
    sampled_rows: usize = 0,
    passes: usize = 0,
    whole: usize = 0,
    refused: usize = 0,
    lone_rounds: usize = 0,
    batches: usize = 0,
    draft_gate: ?*Gate = null,
    fail_draft: bool = false,

    pub const DraftHook = struct {
        ctx: *anyopaque,
        kept: *const fn (ctx: *anyopaque, tokens: []const u32, drafts: usize, keep: usize) anyerror!bool,
    };
    pub const DraftRan = struct { rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, tokens: usize = 0, ns: u64 = 0 };

    fn deinit(s: *Steps) void {
        for (&s.slots) |*sl| sl.history.deinit(gpa);
    }

    pub fn reserve(s: *Steps, slot: u8, prompt_len: usize, max_new: usize) !void {
        const p = s.pool orelse return;
        var moves: [P.max_slots]P.Move = undefined;
        _ = p.reserve(slot, P.need(prompt_len, max_new), &moves) catch |err| {
            s.refused += 1;
            return err;
        };
    }

    pub fn configureSampling(s: *Steps, slot: u8, x: ?scheduling.Sampling) void {
        s.slots[slot].sampling = x;
    }

    fn passEnd(s: *Steps, slot: u8, first: u32) void {
        const sl = &s.slots[slot];
        if (sl.track == .off) return;
        sl.mtp = sl.history.items.len - 1;
        sl.owed = 1;
        sl.last = first;
        sl.head = sl.track == .lone;
        sl.inhead = false;
    }

    pub fn prefill(s: *Steps, key: Key, ids: []const u32) !u32 {
        const sl = &s.slots[key.slot];
        if (key.generation <= sl.generation) return error.StaleGeneration;
        sl.generation = key.generation;
        sl.history.clearRetainingCapacity();
        try sl.history.appendSlice(gpa, ids);
        s.whole += 1;
        const first = draw(sl.history.items, sl.sampling, sl.history.items.len);
        s.passEnd(key.slot, first);
        return first;
    }

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
        const first = draw(sl.history.items, sl.sampling, sl.history.items.len);
        s.passEnd(key.slot, first);
        return first;
    }

    pub fn position(s: *Steps, key: Key) usize {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        return s.slots[key.slot].history.items.len;
    }

    pub fn release(s: *Steps, key: Key) void {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        const sl = &s.slots[key.slot];
        sl.step = null;
        sl.track = .off;
        sl.owed = 0;
        sl.head = false;
        sl.inhead = false;
        sl.rows = 0;
        sl.copied = 0;
        sl.sampling = null;
        if (s.pool) |p| _ = p.release(key.slot);
    }

    /// user_drafts's undrafted per-row round (the switch off): one row a stream, a tracked slot's pair owed.
    pub fn round(s: *Steps, rows: []const Row, out: []u32) !void {
        s.undrafted_rounds += 1;
        s.max_rows = @max(s.max_rows, rows.len);
        for (rows, out) |row, *o| {
            const sl = &s.slots[row.key.slot];
            if (sl.generation != row.key.generation) return error.StaleGeneration;
            if (sl.history.items.len != row.position) return error.PositionMismatch;
            try sl.history.append(gpa, row.token);
            o.* = draw(sl.history.items, row.sampling, sl.history.items.len);
            if (sl.track == .off) continue;
            if (sl.head) {
                sl.track = .off;
                sl.owed = 0;
                sl.head = false;
                continue;
            }
            if (sl.last != row.token) return error.BacklogOutOfStep;
            sl.owed += 1;
            sl.last = o.*;
        }
    }

    // ---- shared_verify: the drafted shared round, its settles and draft requests (cuda_afd_users_dsv.zig's rules) ----

    pub fn roundDrafted(s: *Steps, wins: []const dsv.Window, out: []u32) !void {
        if (!s.ping_pong or wins.len == 1) return s.roundSerial(wins, out);
        var device: Pipe = .{ .s = s };
        try pp.run(Pipe, &device, wins, out);
        s.pipe_rounds += 1;
    }

    const Pipe = struct {
        s: *Steps,
        turn: glm.afd_pairs.Turn = .{ .count = 3, .most = 2 },
        old_gate: ?*Gate = null,
        old_failure: ?usize = null,
        pub fn alive(_: *Pipe) !void {}
        pub fn fail(_: *Pipe) void {}
        pub fn begin(d: *Pipe, _: *const pp.Plan) !bool {
            d.old_gate = d.s.gate;
            d.old_failure = d.s.fail_after;
            return false;
        }
        pub fn open(_: *Pipe, _: u1, _: []const dsv.Window, _: usize) !void {}
        pub fn advance(_: *Pipe, _: u1, _: []const dsv.Window, _: usize, li: *usize) !?usize {
            if (li.* == 3) return null;
            const at = li.*;
            li.* += 1;
            return at;
        }
        pub fn issue(d: *Pipe, slot: u1, _: usize, li: usize) !void {
            try d.turn.request(slot, li);
            if (d.turn.next == 1) {
                if (d.old_gate) |g| {
                    if (d.s.arm_at) |k| if (d.s.dsv_rounds >= k) {
                        g.armed.store(true, .release);
                        d.s.arm_at = null;
                    };
                    g.at();
                }
                if (d.old_failure) |k| if (d.s.dsv_rounds >= k) return error.InjectedExchangeFailure;
            }
        }
        pub fn complete(d: *Pipe, slot: u1, _: usize, li: usize) !void {
            try d.turn.take(slot, li);
        }
        pub fn tailDraw(d: *Pipe, _: u1, wins: []const dsv.Window, _: usize, out: []u32) !void {
            // All simulated exchanges already completed; reuse shared_verify's history/MTP bookkeeping, not its scheduler.
            const gate = d.s.gate;
            const failure = d.s.fail_after;
            d.s.gate = null;
            d.s.fail_after = null;
            defer {
                d.s.gate = gate;
                d.s.fail_after = failure;
            }
            try d.s.roundSerial(wins, out);
        }
        pub fn commit(_: *Pipe, _: dsv.Window) !void {}
        pub fn end(d: *Pipe) !void {
            if (!d.turn.done()) return error.OutOfStep;
        }
    };

    fn roundSerial(s: *Steps, wins: []const dsv.Window, out: []u32) !void {
        const total = try dsv.total(wins);
        if (out.len != total) return error.Rows;
        for (wins) |w| {
            const sl = &s.slots[w.row.key.slot];
            if (sl.generation != w.row.key.generation) return error.StaleGeneration;
            if (sl.history.items.len != w.row.position) return error.PositionMismatch;
            if (sl.step != null) return error.FillingInRound;
            if (sl.rows != 0) return error.Unsettled;
        }
        if (s.gate) |g| {
            if (s.arm_at) |k| if (s.dsv_rounds >= k and wins.len == 2) {
                g.armed.store(true, .release);
                s.arm_at = null;
            };
            g.at();
        }
        if (s.fail_after) |k| if (s.dsv_rounds >= k) return error.InjectedExchangeFailure;
        s.dsv_rounds += 1;
        s.max_rows = @max(s.max_rows, total);
        var h: std.ArrayList(u32) = .empty;
        defer h.deinit(gpa);
        var r0: usize = 0;
        for (wins) |w| {
            defer r0 += w.rows();
            const sl = &s.slots[w.row.key.slot];
            s.max_width = @max(s.max_width, w.rows());
            if (w.row.cancelled()) {
                @memset(out[r0 .. r0 + w.rows()], 0);
                sl.copied = 0;
                continue;
            }
            h.clearRetainingCapacity();
            try h.appendSlice(gpa, sl.history.items);
            try h.append(gpa, w.row.token);
            for (0..w.rows()) |j| { // row j's draw after [history, pending, d1 .. dj]
                out[r0 + j] = draw(h.items, w.row.sampling, w.row.position + 1 + j);
                if (j < w.drafts.len) try h.append(gpa, w.drafts[j]);
            }
            if (w.row.sampling != null) s.sampled_rows += 1;
            sl.copied = if (sl.track != .off and !sl.head) w.rows() else 0;
            if (w.rows() == 1) {
                try sl.history.append(gpa, w.row.token); // a one-row window's row committed in the round
            } else {
                s.wide += 1;
                sl.window[0] = w.row.token;
                @memcpy(sl.window[1..w.rows()], w.drafts);
                sl.rows = w.rows();
            }
        }
    }

    pub fn settleDrafted(s: *Steps, key: Key, rows: usize, keep: usize) !void {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (sl.rows != rows or keep == 0 or keep > rows) return error.KeepOutOfStep;
        try sl.history.appendSlice(gpa, sl.window[0..keep]);
        sl.rows = 0;
        s.settles += 1;
        if (keep < rows) s.replays += 1;
    }

    pub fn draftDrafted(s: *Steps, key: Key, follow: []const u32, verified: bool, position_: usize, out: []u32) !void {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (sl.track == .off or sl.head) return error.NotTracked;
        if (sl.rows != 0) return error.Unsettled;
        const L = sl.history.items.len;
        if (position_ != L + 1 or follow.len == 0) return error.PositionMismatch;
        if (verified) {
            const k = follow.len;
            if (sl.copied == 0 or k > sl.copied or k > L) return error.DraftOutOfStep;
            for (0..k) |j| { // each kept row's next token: the greedy draw after it
                if (follow[j] != lanes.fake.next(sl.history.items[0 .. L - k + j + 1], null, L - k + j + 1)) return error.FollowOutOfStep;
            }
            // never stale: the head absorbed exactly the pairs before the kept rows, and its last pair (absorbed in
            // head, or owed) ends at the window's pending token
            const before = if (sl.inhead) sl.mtp else sl.mtp + sl.owed;
            if (before != L - k or sl.last != sl.history.items[L - k]) {
                s.stale += 1;
                return error.StaleMtp;
            }
        } else if (sl.inhead or sl.owed != 1 or sl.mtp + 1 != L or sl.last != follow[0]) {
            s.stale += 1;
            return error.StaleMtp;
        }
        sl.mtp = L;
        sl.owed = 0;
        sl.inhead = true;
        sl.last = follow[follow.len - 1];
        sl.copied = 0;
        var h: std.ArrayList(u32) = .empty;
        defer h.deinit(gpa);
        try h.appendSlice(gpa, sl.history.items);
        try h.append(gpa, sl.last);
        for (out) |*o| {
            o.* = guess(h.items);
            try h.append(gpa, o.*);
        }
        s.requests += 1;
        s.proposed += out.len;
    }

    pub fn draftsBatched(s: *Steps, requests: []const glm.afd_users_bd.Request) !void {
        s.batches += 1;
        if (s.draft_gate) |g| g.at();
        if (s.fail_draft) return error.ExpertGone;
        for (requests) |r| {
            if (r.row.cancelled()) continue;
            try s.draftDrafted(r.row.key, r.follow, r.verified, r.position, r.out);
        }
    }

    // ---- user_drafts's lone drafting calls (cuda_afd_users.zig's track, carried, drafted, carry) ----

    pub fn track(s: *Steps, slot: u8, how: Track) void {
        const sl = &s.slots[slot];
        sl.track = how;
        sl.owed = 0;
        sl.head = false;
        sl.inhead = false;
        if (how != .off) s.tracks += 1;
    }

    pub fn carried(s: *Steps, key: Key) bool {
        const sl = &s.slots[key.slot];
        return sl.generation == key.generation and sl.track != .off and (sl.owed >= 1 or sl.inhead) and !sl.head;
    }

    pub fn carry(s: *Steps, key: Key, kept: []const u32) !void {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (kept.len != sl.owed or kept[kept.len - 1] != sl.last) return error.CarryOutOfStep;
        s.carries += 1;
    }

    /// mtpWith's shape; a start pair a drafted shared round absorbed in head is un-absorbed first (the family's
    /// dsv.unabsorb inside Users.drafted).
    pub fn drafted(s: *Steps, key: Key, pending: u32, count: usize, out: []u32, hook: *const DraftHook) !DraftRan {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (sl.track == .off) return error.NotTracked;
        if (count == 0 or count > out.len) return error.Count;
        if (sl.inhead) {
            sl.mtp -= 1;
            sl.owed = 1;
            sl.inhead = false;
            s.unabsorbed += 1;
        }
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
            s.lone_rounds += 1;
            sl.mtp += sl.owed;
            sl.owed = 0;
            const depth = @min(s.depth, count - n);
            var drafts: [8]u32 = undefined;
            h.clearRetainingCapacity();
            try h.appendSlice(gpa, sl.history.items);
            try h.append(gpa, cur);
            for (0..depth) |j| {
                drafts[j] = guess(h.items);
                try h.append(gpa, drafts[j]);
            }
            var sampled: [9]u32 = undefined;
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
    accepted: u64 = 0,
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
                b.accepted = f.stats.accepted;
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

/// The reply the undrafted path gives: each token the draw after the prompt and every token before it (a sampled
/// request's keyed by its seed and the token's position).
fn expectSolo(prompt: []const u32, got: []const u32, s: ?scheduling.Sampling) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, prompt);
    for (got) |x| {
        try eq(draw(history.items, s, history.items.len), x);
        try history.append(gpa, x);
    }
}

const Opts = struct { ping_pong: bool = false, on: bool = true, interleave: bool = false, pool: ?*P.Pool = null, draft_reach: usize = cap, sampled: bool = false, batched: bool = false };

/// The users backend over Steps at TF_AFD_USERS=n with user_drafts's lone driver, the lane core and the real LaneHost:
/// shared_verify's switch on (the lane core drafts, Config at 4 rows and 3 drafts) or off (user_drafts's path, as today).
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
        x.steps.ping_pong = o.ping_pong;
        x.back = .{ .steps = &x.steps, .count = n, .capacity = cap, .reach = cap, .interleave = o.interleave, .drafts = true, .draft_reach = o.draft_reach, .shared_drafts = o.on, .sampled = o.sampled, .ping_pong = o.ping_pong, .batched_drafts = o.batched };
        x.cfg = try lanes.Config.init(gpa, x.back.facts(), x.back.windowRows(), x.back.windowRows() - 1);
        x.core = lanes.Engine.init(gpa, &x.cfg, x.back.backend(), x.clock.clock());
        const admit: ?api.Admit = if (o.pool) |p| .{ .ctx = p, .fits = P.Pool.admitFits } else null;
        x.host = api.LaneHost.init(gpa, io, &x.core, .{ .lanes = n, .context_window = 262144, .drafts = "auto", .greedy_users = true, .users_drafts = true, .users_sampled = o.sampled, .admit = admit });
        x.host.lone = x.back.lone(gpa);
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

fn req(prompt: []const u32, max: u32) api.Request {
    return .{ .prompt = prompt, .max_tokens = max, .drafts = true };
}

/// `rs` submitted before the host starts (arriving together) on a fresh served path; each reply.
fn together(n: u8, o: Opts, rs: []const api.Request, boxes: []Box, x: *Served) !void {
    try x.init(n, o);
    const e = x.host.engine();
    for (rs, 0..) |*r, i| try e.submit(@intCast(i + 1), r, boxes[i].sink());
    try x.start();
    for (boxes) |*b| _ = try b.wait();
    x.stop();
}

/// Every reply = its solo reply, its full length; nothing stale, no undrafted per-row round, every slot released.
fn expectServed(rs: []const api.Request, boxes: []const Box, x: *const Served) !void {
    for (rs, boxes) |r, b| {
        try expectSolo(r.prompt, b.tokens.items, null);
        try eq(@as(usize, r.max_tokens), b.tokens.items.len);
        try eq(api.Reason.length, b.done.?);
    }
    try eq(@as(usize, 0), x.steps.stale);
    try eq(@as(usize, 0), x.owners());
    try eq(@as(?anyerror, null), x.back.failed);
    try t.expect(x.steps.max_rows <= dsv.max_rows and x.steps.max_width <= dsv.max_width);
}

pub fn checkPingPongReplies(n: u8, sampled: bool) !void {
    return checkPingPongComposition(n, sampled, false);
}

pub fn checkPingPongComposition(n: u8, sampled: bool, batched: bool) !void {
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    var rs = [_]api.Request{ req(&pa, 120), req(&pb, 70), req(&pc, 50), req(&pd, 40) };
    const sample: lanes.sampling.Sampling = .{ .seed = 19, .temperature = 0.7 };
    if (sampled) rs[1].sampling = sample;
    var blocks: usize = 0;
    for (rs[0..n]) |r| blocks += P.blocksFor(P.need(r.prompt.len, r.max_tokens));
    var pool = try P.Pool.init(gpa, &fx.m, blocks * P.block, 8, true, cap);
    defer pool.deinit(gpa);
    var boxes: [4]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var x: Served = .{};
    defer x.deinit();
    try together(n, .{ .ping_pong = true, .interleave = true, .pool = &pool, .sampled = sampled, .batched = batched }, rs[0..n], boxes[0..n], &x);
    for (rs[0..n], boxes[0..n], 0..) |r, b, i| {
        try expectSolo(r.prompt, b.tokens.items, if (sampled and i == 1) users.drawOf(sample) else null);
        try eq(@as(usize, r.max_tokens), b.tokens.items.len);
        try eq(api.Reason.length, b.done.?);
    }
    try t.expect(x.steps.pipe_rounds > 0 and x.steps.max_rows <= 8 and x.steps.replays > 0);
    try eq(@as(usize, 0), x.owners());
    try eq(@as(usize, 0), x.steps.stale);
    try eq(pool.blocks, pool.freeBlocks());
    if (sampled) try eq(@as(u64, 0), boxes[1].drafted);
    var pool_off = try P.Pool.init(gpa, &fx.m, blocks * P.block, 8, true, cap);
    defer pool_off.deinit(gpa);
    var y: Served = .{};
    defer y.deinit();
    var off: [4]Box = @splat(.{});
    defer for (&off) |*b| b.deinit();
    try together(n, .{ .on = !sampled, .interleave = true, .pool = &pool_off, .sampled = sampled }, rs[0..n], off[0..n], &y);
    for (boxes[0..n], off[0..n]) |a, b| try t.expectEqualSlices(u32, a.tokens.items, b.tokens.items);
    try eq(pool_off.blocks, pool_off.freeBlocks());
    std.debug.print("ping_pong LaneHost C{d} sampled={}: {d}/{d} on=off=solo; pipeline rounds={d}, micro rows<={d}, rollback={d}, pool restored\n", .{ n, sampled, n, n, x.steps.pipe_rounds, x.steps.max_rows, x.steps.replays });
}

pub fn checkPingPongTransition(cancel_: bool) !void {
    var gate: Gate = .{};
    var x: Served = .{};
    defer x.deinit();
    try x.init(2, .{ .ping_pong = true });
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    const e = x.host.engine();
    const ra = req(&pa, 160);
    const rb = req(&pc, if (cancel_) 150 else 40);
    x.steps.gate = &gate;
    if (cancel_) {
        x.steps.arm_at = 4;
        try e.submit(1, &ra, ba.sink());
        try e.submit(2, &rb, bb.sink());
    } else {
        gate.armed.store(true, .release);
        try e.submit(1, &ra, ba.sink());
    }
    try x.start();
    defer gate.open.store(true, .release);
    try gate.waitEntered();
    if (cancel_) e.cancel(2) else try e.submit(2, &rb, bb.sink());
    gate.open.store(true, .release);
    const why = try bb.wait();
    _ = try ba.wait();
    x.stop();
    try expectSolo(&pa, ba.tokens.items, null);
    try expectSolo(&pc, bb.tokens.items, null);
    try eq(if (cancel_) api.Reason.cancelled else api.Reason.length, why);
    try eq(@as(usize, 0), x.owners());
    try eq(@as(usize, 0), x.steps.stale);
    try t.expect(x.steps.pipe_rounds > 0);
    if (!cancel_) try t.expect(x.back.counts.yields > 0 and x.back.counts.resumes > 0 and x.steps.unabsorbed > 0);
}

pub fn checkPingPongFailure() !void {
    var x: Served = .{};
    defer x.deinit();
    try x.init(2, .{ .ping_pong = true });
    x.steps.fail_after = 4;
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    const ra = req(&pa, 100);
    const rb = req(&pc, 100);
    const e = x.host.engine();
    try e.submit(1, &ra, ba.sink());
    try e.submit(2, &rb, bb.sink());
    try x.start();
    try eq(api.Reason.failed, try ba.wait());
    try eq(api.Reason.failed, try bb.wait());
    x.stop();
    try t.expect(x.back.failed != null);
    try eq(@as(usize, 0), x.owners());
}

// ---- concurrent drafted users -----------------------------------------------------------------------------------------

test "shared_verify C2 drafted shared rounds: each reply = its solo reply; windows wider than one row, drafts refused and rolled back per stream; the switch off gives the same bytes" {
    const rs = [_]api.Request{ req(&pa, 90), req(&pc, 60) };
    var boxes: [2]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var x: Served = .{};
    defer x.deinit();
    try together(2, .{}, &rs, &boxes, &x);
    try expectServed(&rs, &boxes, &x);
    try t.expect(x.steps.dsv_rounds > 0 and x.steps.wide > 0); // drafted windows in shared rounds
    try t.expect(x.steps.max_rows >= 3); // more than one row a stream in one forward
    try t.expect(x.steps.replays > 0); // a refused draft rolled back on its own stream
    try t.expect(x.steps.settles > x.steps.replays); // and fully kept windows too
    try t.expect(x.steps.requests > 0 and x.steps.proposed > 0);
    try eq(@as(usize, 0), x.steps.undrafted_rounds); // every core round drafted shared (shared_verify's forward)
    try t.expect(x.back.dsv_counts.wide > 0 and x.back.dsv_counts.most_rows == x.steps.max_rows);
    try t.expect(boxes[0].drafted > 0 and boxes[0].accepted > 0); // the lane core's per-stream acceptance counted
    // the switch off on the same two: user_drafts's path (shared rounds undrafted), the same bytes
    var off_boxes: [2]Box = @splat(.{});
    defer for (&off_boxes) |*b| b.deinit();
    var y: Served = .{};
    defer y.deinit();
    try together(2, .{ .on = false }, &rs, &off_boxes, &y);
    for (boxes, off_boxes) |b, ob| try t.expectEqualSlices(u32, b.tokens.items, ob.tokens.items);
    try eq(@as(usize, 0), y.steps.dsv_rounds);
    try eq(@as(usize, 0), y.steps.wide);
    try t.expect(y.steps.undrafted_rounds > 0);
    std.debug.print("shared_verify C2: 2/2 replies = solo = switch off; {d} drafted shared rounds ({d} wider windows, at most {d} rows), {d} settles ({d} rolled back), {d} draft requests\n", .{ x.steps.dsv_rounds, x.steps.wide, x.steps.max_rows, x.steps.settles, x.steps.replays, x.steps.requests });
}

fn fixtureModel() !struct { parsed: std.json.Parsed(std.json.Value), m: glm.model.Model } {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    errdefer parsed.deinit();
    var m = try glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
    try m.cut(5);
    m.mtp_layers = 0;
    return .{ .parsed = parsed, .m = m };
}

test "shared_verify USERS=4 with pool + interleave: four drafted replies = solo; at most 8 rows a round; the pool's regions all back; the switch off gives the same bytes" {
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    const rs = [_]api.Request{ req(&pa, 120), req(&pb, 40), req(&pc, 70), req(&pd, 50) };
    var blocks: usize = 0;
    for (rs) |r| blocks += P.blocksFor(P.need(r.prompt.len, r.max_tokens));
    var pool = try P.Pool.init(gpa, &fx.m, blocks * P.block, 8, true, cap);
    defer pool.deinit(gpa);
    var boxes: [4]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var x: Served = .{};
    defer x.deinit();
    try together(4, .{ .interleave = true, .pool = &pool }, &rs, &boxes, &x);
    try expectServed(&rs, &boxes, &x);
    try t.expect(x.steps.passes >= 4); // prefill_interleave: every prompt went in a step at a time
    try t.expect(x.steps.wide > 0 and x.steps.max_rows >= 5);
    try t.expect(x.steps.replays > 0);
    try eq(pool.blocks, pool.freeBlocks());
    var off_boxes: [4]Box = @splat(.{});
    defer for (&off_boxes) |*b| b.deinit();
    var pool2 = try P.Pool.init(gpa, &fx.m, blocks * P.block, 8, true, cap);
    defer pool2.deinit(gpa);
    var y: Served = .{};
    defer y.deinit();
    try together(4, .{ .on = false, .interleave = true, .pool = &pool2 }, &rs, &off_boxes, &y);
    for (boxes, off_boxes) |b, ob| try t.expectEqualSlices(u32, b.tokens.items, ob.tokens.items);
    try eq(pool2.blocks, pool2.freeBlocks());
    std.debug.print("shared_verify USERS=4 pool + interleave: 4/4 replies = solo = switch off; {d} drafted shared rounds, at most {d} rows ({d} a window), {d} rolled back; pool blocks all back\n", .{ x.steps.dsv_rounds, x.steps.max_rows, x.steps.max_width, x.steps.replays });
}

// ---- transitions ---------------------------------------------------------------------------------------------------------

test "shared_verify lone -> drafted shared -> lone mid-reply: the arrival hands A over with its kept rows carried, A drafts beside B, at B's end A drafts alone from its own start pair; both = solo" {
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
    try e.submit(2, &rb, bb.sink());
    gate.open.store(true, .release);
    _ = try bb.wait();
    _ = try ba.wait();
    x.stop();
    try expectServed(&.{ ra, rb }, &.{ ba, bb }, &x);
    try t.expect(x.back.counts.yields >= 1); // A handed over at B's arrival
    try t.expect(x.steps.carries >= 1); // with its last lone round's kept rows
    try t.expect(x.steps.wide >= 1); // A and B drafted in shared rounds
    try t.expect(x.back.counts.resumes >= 1); // B's end: A back to its lone drafted rounds
    try t.expect(x.steps.unabsorbed >= 1); // from its own start pair (un-absorbed)
    try t.expect(x.steps.resumed_loops >= 1);
    std.debug.print("shared_verify lone -> shared -> lone: 2/2 = solo; {d} yields, {d} carries, {d} drafted shared rounds ({d} wider windows), {d} resumes from an un-absorbed start pair\n", .{ x.back.counts.yields, x.steps.carries, x.steps.dsv_rounds, x.steps.wide, x.steps.unabsorbed });
}

test "shared_verify past the drafted reach: that stream keeps one-row windows beside a drafted one; both = solo" {
    const rs = [_]api.Request{ req(&pa, 90), req(&pb, 60) };
    var boxes: [2]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var x: Served = .{};
    defer x.deinit();
    try together(2, .{ .draft_reach = 200 }, &rs, &boxes, &x); // pb's prompt alone is past it
    try expectServed(&rs, &boxes, &x);
    try t.expect(x.steps.wide > 0); // A drafted
    try t.expect(x.steps.max_rows <= dsv.max_width + 1); // B never more than its pending row
}

// ---- sampled beside drafted ------------------------------------------------------------------------------------------------

test "shared_verify sampled stream beside drafted ones: never drafted, every row its own keyed draw (user_sampling's rule); each reply = its solo" {
    const s: lanes.sampling.Sampling = .{ .seed = 11, .temperature = 0.7 };
    const rs = [_]api.Request{ req(&pa, 80), .{ .prompt = &pc, .max_tokens = 50, .drafts = true, .sampling = s } };
    var boxes: [2]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var x: Served = .{};
    defer x.deinit();
    try together(2, .{ .sampled = true }, &rs, &boxes, &x);
    try expectSolo(&pa, boxes[0].tokens.items, null);
    try expectSolo(&pc, boxes[1].tokens.items, users.drawOf(s));
    try eq(@as(usize, 80), boxes[0].tokens.items.len);
    try eq(@as(usize, 50), boxes[1].tokens.items.len);
    try t.expect(x.steps.wide > 0); // A drafted
    try t.expect(x.steps.sampled_rows >= 49); // B's rows all sampled, one row each (dsv.total refuses a drafted one)
    try eq(@as(u64, 0), boxes[1].drafted);
    try eq(@as(usize, 0), x.steps.stale);
    try eq(@as(usize, 0), x.owners());
}

// ---- cancel and failure inside a drafted shared round ----------------------------------------------------------------------

test "shared_verify cancel inside a drafted shared round: that request ends cancelled, its slot released, nothing kept after it; the other = solo; a later request = solo" {
    var gate: Gate = .{};
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    var x: Served = .{};
    defer x.deinit();
    try x.init(2, .{});
    const e = x.host.engine();
    const ra = req(&pa, 150);
    const rb = req(&pc, 150);
    try e.submit(1, &ra, ba.sink());
    try e.submit(2, &rb, bb.sink());
    x.steps.gate = &gate;
    x.steps.arm_at = 4; // held inside a drafted shared round of both, a few rounds in
    try x.start();
    try gate.waitEntered();
    e.cancel(2);
    gate.open.store(true, .release);
    try eq(api.Reason.cancelled, try bb.wait());
    _ = try ba.wait();
    try expectSolo(&pa, ba.tokens.items, null);
    try expectSolo(&pc, bb.tokens.items, null); // what B got before its cancel is its solo prefix
    try t.expect(bb.tokens.items.len < 150);
    try eq(@as(?anyerror, null), x.back.failed);
    var bc: Box = .{};
    defer bc.deinit();
    const rc = req(&pd, 40);
    try e.submit(3, &rc, bc.sink());
    _ = try bc.wait();
    x.stop();
    try expectSolo(&pd, bc.tokens.items, null);
    try eq(@as(usize, 40), bc.tokens.items.len);
    try eq(@as(usize, 0), x.owners());
    try eq(@as(usize, 0), x.steps.stale);
}

test "shared_verify failure inside a drafted shared round: the requests fail in words, the backend latches it, every slot released; a later request refused" {
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    var x: Served = .{};
    defer x.deinit();
    try x.init(2, .{});
    x.steps.fail_after = 6;
    const e = x.host.engine();
    const ra = req(&pa, 150);
    const rb = req(&pc, 150);
    try e.submit(1, &ra, ba.sink());
    try e.submit(2, &rb, bb.sink());
    try x.start();
    try eq(api.Reason.failed, try ba.wait());
    try eq(api.Reason.failed, try bb.wait());
    try t.expect(std.mem.indexOf(u8, &ba.message, "InjectedExchangeFailure") != null);
    try eq(@as(?anyerror, error.InjectedExchangeFailure), x.back.failed);
    try expectSolo(&pa, ba.tokens.items, null); // what each got before the failure is its solo prefix
    try expectSolo(&pc, bb.tokens.items, null);
    try eq(@as(usize, 0), x.owners());
    var bc: Box = .{};
    defer bc.deinit();
    const rc = req(&pd, 20);
    try e.submit(3, &rc, bc.sink());
    try eq(api.Reason.failed, try bc.wait()); // refused before any device call
    x.stop();
    try eq(@as(usize, 0), bc.tokens.items.len);
    try eq(@as(usize, 0), x.owners());
}

// ---- the switch -------------------------------------------------------------------------------------------------------------

test "shared_verify env harness (FJPANIC): saveEnv moves libc's environ off std's startup array first, so unsetting a name the process started with leaves std's startup environment whole; restoreEnv puts the name back" {
    var buf: [256]u8 = undefined;
    const name = harness.startupName(&buf) orelse return error.SkipZigTest; // no startup environment: nothing to shift
    const names = [_][*:0]const u8{name.ptr};
    {
        var old: [names.len]?[:0]u8 = @splat(null);
        try saveEnv(&names, &old);
        defer restoreEnv(&names, &old);
        try t.expect(harness.detached()); // checked before anything is unset, so a failure leaves std's slice whole
        _ = envcall.unsetenv(name.ptr);
        try t.expect(std.c.getenv(name.ptr) == null);
        try t.expect(harness.startupWhole());
    }
    try t.expect(std.c.getenv(name.ptr) != null);
}

test "shared_verify switch through plan(): unset and 0 off, 1 on with TF_AFD_USERS_SHARED=1 and TF_AFD_USERS_DRAFTS=1 at users 2 and 4; refusals by name; the others' plans unchanged" {
    const names = [_][*:0]const u8{ "TF_AFD_USERS", "TF_AFD_USERS_DRAFTS", "TF_AFD_MTP", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE", "TF_AFD_USERS_SHARED", "TF_AFD_USERS_SHARED_DRAFTS", "TF_AFD_USERS_SAMPLING" };
    var old: [names.len]?[:0]u8 = @splat(null);
    try saveEnv(&names, &old);
    defer restoreEnv(&names, &old);
    for (names) |n| _ = envcall.unsetenv(n);
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const auto2 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    const auto4 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    const off2 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "0", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    _ = envcall.setenv("TF_AFD_USERS", "2", 1);
    _ = envcall.setenv("TF_AFD_MTP", "1", 1);
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    // unset: user_drafts's plan as before; shared + drafts draft_shared's composition as before (a lone user drafts, the shared
    // rounds undrafted at TF_AFD_USERS's rows)
    var p = try served.plan(a, auto2, cfg);
    try t.expect(p.users_drafts and !p.users_shared and !p.users_shared_drafts);
    try eq(@as(u8, 2), p.stageRows());
    _ = envcall.setenv("TF_AFD_USERS_SHARED", "1", 1);
    p = try served.plan(a, auto2, cfg);
    try t.expect(p.users_drafts and p.users_shared and !p.users_shared_drafts);
    try eq(@as(u8, 2), p.stageRows());
    _ = envcall.setenv("TF_AFD_USERS_SHARED_DRAFTS", "0", 1);
    p = try served.plan(a, auto2, cfg);
    try t.expect(p.users_drafts and p.users_shared and !p.users_shared_drafts);
    // on: both switches with it, at users 2 and 4
    _ = envcall.setenv("TF_AFD_USERS_SHARED_DRAFTS", "1", 1);
    p = try served.plan(a, auto2, cfg);
    try t.expect(p.users_drafts and p.users_shared and p.users_shared_drafts);
    try eq(@as(u8, 8), p.stageRows());
    _ = envcall.setenv("TF_AFD_USERS", "4", 1);
    p = try served.plan(a, auto4, cfg);
    try t.expect(p.users_shared_drafts and p.users == 4);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "1", 1);
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "655360", 1); // shared_pool: a whole number of 256-token blocks
    p = try served.plan(a, auto4, cfg);
    try t.expect(p.users_shared_drafts and p.pool == 655360 and p.interleave != null);
    _ = envcall.unsetenv("TF_AFD_USERS_INTERLEAVE");
    _ = envcall.unsetenv("TF_AFD_POOL_TOKENS");
    _ = envcall.setenv("TF_AFD_USERS", "2", 1);
    // refusals, each its own
    _ = envcall.unsetenv("TF_AFD_USERS_SHARED");
    try t.expectError(error.UsersSharedDraftsNeedsShared, served.plan(a, auto2, cfg));
    _ = envcall.setenv("TF_AFD_USERS_SHARED", "1", 1);
    _ = envcall.unsetenv("TF_AFD_USERS_DRAFTS");
    try t.expectError(error.UsersSharedDraftsNeedsDrafts, served.plan(a, off2, cfg));
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    try t.expectError(error.UsersDraftsNeedPolicy, served.plan(a, off2, cfg)); // user_drafts's own: a policy
    for ([_][*:0]const u8{ "2", "on", "true", " 1", "01", "" }) |v| {
        _ = envcall.setenv("TF_AFD_USERS_SHARED_DRAFTS", v, 1);
        try t.expectError(error.BadAfdUsersSharedDrafts, served.plan(a, auto2, cfg));
    }
    _ = envcall.unsetenv("TF_AFD_USERS_SHARED_DRAFTS");
    _ = envcall.unsetenv("TF_AFD_USERS_SHARED");
    p = try served.plan(a, auto2, cfg); // and back: user_drafts's plan
    try t.expect(p.users_drafts and !p.users_shared and !p.users_shared_drafts);
    std.debug.print("shared_verify plan: unset/0 = the plan as before; 1 with shared + drafts at users 2 and 4 (and pool + interleave); needs-shared, needs-drafts and 6 spellings refused\n", .{});
}

// ---- default off -------------------------------------------------------------------------------------------------------------

test "shared_verify default off: the backend's facts, window rows and lone driver as before; a drafted stream refused; on: 4 rows, 3 drafts, 8 rows a round" {
    var steps: Steps = .{};
    defer steps.deinit();
    var off: Back = .{ .steps = &steps, .count = 4, .capacity = cap, .reach = cap, .drafts = true, .draft_reach = cap };
    const f0 = off.facts();
    try eq(@as(u32, 1), f0.exact_width);
    try eq(@as(u32, 4), f0.batch_rows);
    try t.expect(f0.single_row_streams and !f0.mtp and !f0.speculate);
    try eq(@as(u32, 1), off.windowRows());
    const l0 = off.lone(gpa);
    try t.expect(l0.core_undrafted and !l0.core_drafts);
    var cfg0 = try lanes.Config.init(gpa, f0, off.windowRows(), off.windowRows() - 1);
    defer cfg0.deinit(gpa);
    try t.expect(!cfg0.family_mtp and cfg0.family_width == 1 and cfg0.single_row_streams and cfg0.family_streams);
    var s = try lanes.Stream.init(gpa, .{ .id = "x", .prompt = &pa, .max_new = 8, .drafts = true });
    defer s.deinit(gpa);
    try t.expectError(error.UsersDraftsUnsupported, off.backend().prefill(&s)); // as today
    var on: Back = .{ .steps = &steps, .count = 4, .capacity = cap, .reach = cap, .drafts = true, .draft_reach = cap, .shared_drafts = true };
    const f1 = on.facts();
    try t.expect(f1.mtp and f1.speculate and !f1.speculate_early and f1.single_row_streams and f1.draft_streams);
    try eq(@as(u32, 4), f1.exact_width);
    try eq(@as(u32, 8), f1.batch_rows);
    try eq(@as(u32, 4), f1.max_streams);
    try eq(@as(u32, 4), on.windowRows());
    try t.expect(on.lone(gpa).core_drafts);
    var cfg1 = try lanes.Config.init(gpa, f1, on.windowRows(), on.windowRows() - 1);
    defer cfg1.deinit(gpa);
    try t.expect(cfg1.family_mtp and cfg1.family_width == 4 and cfg1.most_drafts == 3 and cfg1.batch_rows == 8 and cfg1.single_row_streams);
    std.debug.print("shared_verify default off: facts exact_width 1, batch_rows 4, Config 1 row / 0 drafts, a drafted stream refused; on: 4 rows, 3 drafts, 8 rows a round\n", .{});
}

/// batched_drafts reuses the shared_verify host fixture; its actual shared MTP launches are tested separately.
pub fn checkBatchedHost() !void {
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    const rs = [_]api.Request{ req(&pa, 120), req(&pb, 40), req(&pc, 70), req(&pd, 50) };
    for ([_]u8{ 2, 4 }) |n| {
        var pool = try P.Pool.init(gpa, &fx.m, 16 * P.block, 8, true, cap);
        defer pool.deinit(gpa);
        var a: Served = .{};
        defer a.deinit();
        var ab: [4]Box = @splat(.{});
        defer for (&ab) |*b| b.deinit();
        try together(n, .{ .batched = true, .pool = &pool, .interleave = true }, rs[0..n], ab[0..n], &a);
        try expectServed(rs[0..n], ab[0..n], &a);
        try t.expect(a.steps.batches > 0 and a.steps.replays > 0);
        try eq(pool.blocks, pool.freeBlocks());
        var b: Served = .{};
        defer b.deinit();
        var bb: [4]Box = @splat(.{});
        defer for (&bb) |*x| x.deinit();
        try together(n, .{ .interleave = true }, rs[0..n], bb[0..n], &b);
        var c: Served = .{};
        defer c.deinit();
        var cb: [4]Box = @splat(.{});
        defer for (&cb) |*x| x.deinit();
        try together(n, .{ .on = false, .interleave = true }, rs[0..n], cb[0..n], &c);
        for (0..n) |i| {
            try t.expectEqualSlices(u32, bb[i].tokens.items, ab[i].tokens.items);
            try t.expectEqualSlices(u32, cb[i].tokens.items, ab[i].tokens.items);
        }
        std.debug.print("batched_drafts host C{d}: drafted = undrafted {d}/{d}; solo = together {d}/{d}; per-stream path exact; batches {d}, rollbacks {d}, pool all released\n", .{ n, n, n, n, n, a.steps.batches, a.steps.replays });
    }
}

pub fn checkBatchedTransition() !void {
    var gate: Gate = .{};
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    var x: Served = .{};
    defer x.deinit();
    try x.init(2, .{ .batched = true });
    x.steps.gate = &gate;
    gate.armed.store(true, .release);
    const e = x.host.engine();
    const ra = req(&pa, 160);
    const rb = req(&pb, 40);
    try e.submit(1, &ra, ba.sink());
    try x.start();
    try gate.waitEntered();
    try e.submit(2, &rb, bb.sink());
    gate.open.store(true, .release);
    _ = try bb.wait();
    _ = try ba.wait();
    x.stop();
    try expectServed(&.{ ra, rb }, &.{ ba, bb }, &x);
    try t.expect(x.steps.batches > 0 and x.back.counts.yields > 0 and x.back.counts.resumes > 0 and x.steps.unabsorbed > 0);
    std.debug.print("batched_drafts transitions: solo = together 2/2; lone -> batched shared -> lone, start pair unabsorbed\n", .{});
}

pub fn checkBatchedCancelFailure() !void {
    for ([_]bool{ false, true }) |fail| {
        var gate: Gate = .{};
        var ba: Box = .{};
        defer ba.deinit();
        var bb: Box = .{};
        defer bb.deinit();
        var x: Served = .{};
        defer x.deinit();
        try x.init(2, .{ .batched = true });
        x.steps.draft_gate = &gate;
        gate.armed.store(true, .release);
        const e = x.host.engine();
        const ra = req(&pa, 150);
        const rb = req(&pc, 150);
        try e.submit(1, &ra, ba.sink());
        try e.submit(2, &rb, bb.sink());
        try x.start();
        try gate.waitEntered();
        if (fail) x.steps.fail_draft = true else e.cancel(2);
        gate.open.store(true, .release);
        try eq(if (fail) api.Reason.failed else api.Reason.cancelled, try bb.wait());
        try eq(if (fail) api.Reason.failed else api.Reason.length, try ba.wait());
        x.stop();
        try eq(@as(usize, 0), x.owners());
        try expectSolo(&pa, ba.tokens.items, null);
        try expectSolo(&pc, bb.tokens.items, null);
        if (fail) try eq(@as(?anyerror, error.ExpertGone), x.back.failed) else try eq(@as(?anyerror, null), x.back.failed);
    }
    std.debug.print("batched_drafts host mid-draft cancel/failure: prefixes exact, slots released, failure sticky\n", .{});
}
