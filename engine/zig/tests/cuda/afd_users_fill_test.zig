//! prefill_interleave: the users backend (native/glm_cuda_users.zig) filling a prompt a step of the served pass at a time
//! (TF_AFD_USERS_INTERLEAVE), through the real lane core and LaneHost, over a Steps stand-in whose tokens depend on each
//! slot's whole history (lanes.fake.next): off hands out today's vtable; on, a live user commits between every two steps
//! of another user's prompt and both replies equal their solo replies; the slot's generation is claimed at the first
//! step; a cancel between steps releases the slot and the next request takes its next generation; a failed step is
//! sticky for every owner, as a failed whole prefill; two prompts arriving together fill oldest first.
//! Claude Opus 5.5 (planning and execution), via Hermes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const glm = @import("glm_cuda");
const users = @import("glm_users");
const scheduling = glm.afd_users_round;
const Key = scheduling.Key;
const Row = scheduling.Row;
const gpa = std.testing.allocator;
const io = std.testing.io;
const eq = std.testing.expectEqual;

/// One user's slot: its generation, its history (the prompt once filled, then every fed token), its pass's next step.
const Slot = struct { generation: u64 = 0, history: std.ArrayList(u32) = .empty, step: ?usize = null };

/// What Steps did, in order, with every slot's position at that moment (the recorder).
const Ev = struct { kind: enum { whole, step, round, release }, slot: u8, generation: u64, k: usize = 0, positions: [4]usize };

pub const Steps = struct {
    slots: [4]Slot = @splat(.{}),
    chunks: usize = 6,
    log: std.ArrayList(Ev) = .empty,
    failed: bool = false,
    fail_step: ?usize = null,
    whole: usize = 0,
    passes: usize = 0,
    rounds: usize = 0,
    hook: ?struct { ctx: *anyopaque, at: *const fn (*anyopaque, Key, usize) void } = null,

    fn deinit(s: *Steps) void {
        for (&s.slots) |*sl| sl.history.deinit(gpa);
        s.log.deinit(gpa);
    }

    fn note(s: *Steps, kind: @FieldType(Ev, "kind"), key: Key, k: usize) void {
        var p: [4]usize = undefined;
        for (s.slots, &p) |sl, *v| v.* = sl.history.items.len;
        s.log.append(gpa, .{ .kind = kind, .slot = key.slot, .generation = key.generation, .k = k, .positions = p }) catch {};
    }

    fn alive(s: *Steps) !void {
        if (s.failed) return error.ExpertFailed;
    }

    pub fn prefill(s: *Steps, key: Key, ids: []const u32) !u32 {
        try s.alive();
        const sl = &s.slots[key.slot];
        if (key.generation <= sl.generation) return error.StaleGeneration;
        sl.generation = key.generation;
        sl.history.clearRetainingCapacity();
        try sl.history.appendSlice(gpa, ids);
        s.whole += 1;
        s.note(.whole, key, 0);
        return lanes.fake.next(sl.history.items, null, sl.history.items.len);
    }

    pub fn prefillStep(s: *Steps, key: Key, ids: []const u32) !?u32 {
        try s.alive();
        const sl = &s.slots[key.slot];
        if (sl.step == null) {
            if (key.generation <= sl.generation) return error.StaleGeneration;
            sl.generation = key.generation;
            sl.history.clearRetainingCapacity();
            sl.step = 0;
            s.passes += 1;
        } else if (key.generation != sl.generation) return error.StaleGeneration;
        const k = sl.step.?;
        s.note(.step, key, k);
        if (s.hook) |h| h.at(h.ctx, key, k);
        if (s.fail_step == k) {
            s.failed = true; // as a failed exchange: every owner fails
            return error.ExpertFailed;
        }
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

    pub fn release(s: *Steps, key: Key) void {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        s.slots[key.slot].step = null;
        s.note(.release, key, 0);
    }

    pub fn round(s: *Steps, rows: []const Row, out: []u32) !void {
        try s.alive();
        for (rows) |row| {
            const sl = &s.slots[row.key.slot];
            if (sl.generation != row.key.generation) return error.StaleGeneration;
            if (sl.history.items.len != row.position) return error.PositionMismatch;
            if (sl.step != null) return error.FillingInRound; // a filling slot never takes part in a round
        }
        s.rounds += 1;
        for (rows, out) |row, *o| {
            const sl = &s.slots[row.key.slot];
            try sl.history.append(gpa, row.token);
            o.* = lanes.fake.next(sl.history.items, null, sl.history.items.len);
        }
        s.note(.round, rows[0].key, rows.len);
    }
};

/// The same Steps without `prefillStep`: today's device, which never fills.
const Whole = struct {
    pub fn prefill(_: *Whole, _: Key, _: []const u32) !u32 {
        return 0;
    }
    pub fn position(_: *Whole, _: Key) usize {
        return 0;
    }
    pub fn release(_: *Whole, _: Key) void {}
    pub fn round(_: *Whole, _: []const Row, _: []u32) !void {}
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    message: []const u8 = "",

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.message = f.message;
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
        for (0..20000) |_| {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.WaitTimeout;
    }

    fn waitTokens(b: *Box, n: usize) !void {
        for (0..20000) |_| {
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
    for (got) |t| {
        try eq(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

const Back = users.Users(Steps);

/// The users backend over Steps, the lane core and the real LaneHost at R:C, interleave on.
const Served = struct {
    steps: Steps = .{},
    back: Back = undefined,
    cfg: lanes.Config = undefined,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: api.LaneHost = undefined,

    fn init(x: *Served, n: u8, policy: api.LaneHost.FillPolicy) !void {
        x.back = .{ .steps = &x.steps, .count = n, .capacity = 262208, .reach = 262208, .interleave = true };
        x.cfg = try lanes.Config.init(gpa, x.back.facts(), 1, 0);
        x.core = lanes.Engine.init(gpa, &x.cfg, x.back.backend(), x.clock.clock());
        x.host = api.LaneHost.init(gpa, io, &x.core, .{ .lanes = n, .context_window = 262144, .drafts = "off", .greedy_users = true });
        x.host.fill_policy = policy;
        try x.host.start();
    }

    fn deinit(x: *Served) void {
        x.host.stop();
        x.core.deinit();
        x.cfg.deinit(gpa);
        x.steps.deinit();
    }

    /// The steps of (slot, generation)'s pass in the record, in order.
    fn stepsOf(x: *const Served, slot: u8, generation: u64, out: []Ev) []Ev {
        var k: usize = 0;
        for (x.steps.log.items) |e| if (e.kind == .step and e.slot == slot and e.generation == generation) {
            out[k] = e;
            k += 1;
        };
        return out[0..k];
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
const ra: api.Request = .{ .prompt = &pa, .max_tokens = 100000, .drafts = false };
const rb30: api.Request = .{ .prompt = &pb, .max_tokens = 30, .drafts = false };
const rb40: api.Request = .{ .prompt = &pb, .max_tokens = 40, .drafts = false };
const rc30: api.Request = .{ .prompt = &pc, .max_tokens = 30, .drafts = false };

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

test "prefill_interleave switch: TF_AFD_USERS_INTERLEAVE unset or 0 off, 1 and canonical R:C (1..64) on, malformed or without TF_AFD_USERS refused, all before weights" {
    const I = scheduling.Interleave;
    try eq(@as(?I, null), try scheduling.parseInterleave(null));
    try eq(@as(?I, null), try scheduling.parseInterleave("0"));
    try eq(@as(?I, .{ .rounds = 1, .steps = 1 }), try scheduling.parseInterleave("1"));
    try eq(@as(?I, .{ .rounds = 1, .steps = 1 }), try scheduling.parseInterleave("1:1"));
    try eq(@as(?I, .{ .rounds = 8, .steps = 1 }), try scheduling.parseInterleave("8:1"));
    try eq(@as(?I, .{ .rounds = 1, .steps = 4 }), try scheduling.parseInterleave("1:4"));
    try eq(@as(?I, .{ .rounds = 64, .steps = 64 }), try scheduling.parseInterleave("64:64"));
    for ([_][]const u8{ "", "00", "2", "true", "on", " 1", "1 ", "1:", ":1", "0:1", "1:0", "01:1", "1:01", "65:1", "1:65", "1:1:1", "-1:1", "1:+1", "1.0", "100:1" }) |v|
        try std.testing.expectError(error.BadAfdUsersInterleave, scheduling.parseInterleave(v));
    const users_env = try Saved.take("TF_AFD_USERS");
    defer users_env.put();
    const il_env = try Saved.take("TF_AFD_USERS_INTERLEAVE");
    defer il_env.put();
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const flags = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--no-drafts", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    _ = envcall.setenv("TF_AFD_USERS", "2", 1);
    _ = envcall.unsetenv("TF_AFD_USERS_INTERLEAVE");
    try eq(@as(?I, null), (try served.plan(a, flags, cfg)).interleave); // unset: today's plan
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "0", 1);
    try eq(@as(?I, null), (try served.plan(a, flags, cfg)).interleave);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "1", 1);
    const p = try served.plan(a, flags, cfg);
    try eq(@as(?I, .{ .rounds = 1, .steps = 1 }), p.interleave);
    try eq(@as(u8, 2), p.users);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "8:1", 1);
    try eq(@as(?I, .{ .rounds = 8, .steps = 1 }), (try served.plan(a, flags, cfg)).interleave);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "yes", 1);
    try std.testing.expectError(error.BadAfdUsersInterleave, served.plan(a, flags, cfg));
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "1", 1);
    _ = envcall.unsetenv("TF_AFD_USERS");
    try std.testing.expectError(error.UsersInterleaveNeedsUsers, served.plan(a, try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read" }), cfg));
}

test "prefill_interleave users: TF_AFD_USERS_INTERLEAVE off hands out today's vtable (no prefill_step), on adds it; a device without prefillStep never fills" {
    var steps: Steps = .{};
    defer steps.deinit();
    var off: Back = .{ .steps = &steps, .count = 2, .capacity = 4096, .reach = 4096 };
    try std.testing.expect(off.backend().vtable.prefill_step == null);
    var on: Back = .{ .steps = &steps, .count = 2, .capacity = 4096, .reach = 4096, .interleave = true };
    try std.testing.expect(on.backend().vtable.prefill_step != null);
    var whole: Whole = .{};
    var never: users.Users(Whole) = .{ .steps = &whole, .count = 2, .capacity = 4096, .reach = 4096, .interleave = true };
    try std.testing.expect(never.backend().vtable.prefill_step == null);
    // off, the lane core runs today's whole pass through the same backend
    var cfg = try lanes.Config.init(gpa, off.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, off.backend(), clock.clock());
    defer core.deinit();
    try std.testing.expect(!core.fills());
    var s = try lanes.Stream.init(gpa, .{ .id = "w", .prompt = &pa, .max_new = 5, .drafts = false });
    defer s.deinit(gpa);
    try core.addStream(&s);
    while (core.activeCount() > 0) try core.step();
    try expectSolo(&pa, s.emitted());
    try eq(@as(usize, 1), steps.whole);
    try eq(@as(usize, 0), steps.passes);
}

test "prefill_interleave users C2 through the real LaneHost at 1:1: the live user commits between every two steps of the other user's prompt; both replies are their solo replies; the slot's generation is claimed at the first step" {
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var x: Served = .{}; // after the boxes: its deinit (the host's stop) runs before theirs
    try x.init(2, .{});
    defer x.deinit();
    const e = x.host.engine();
    try e.submit(1, &ra, a.sink());
    try a.waitTokens(3);
    try e.submit(2, &rb30, b.sink());
    try eq(api.Reason.length, try b.wait());
    try expectSolo(&pb, b.tokens.items);
    e.cancel(1);
    try eq(api.Reason.cancelled, try a.wait());
    try expectSolo(&pa, a.tokens.items);
    var out: [16]Ev = undefined;
    const bs = x.stepsOf(1, 1, &out);
    try eq(@as(usize, 6), bs.len);
    for (bs, 0..) |s, k| {
        try eq(k, s.k);
        if (k > 0) try eq(@as(usize, 1), s.positions[0] - bs[k - 1].positions[0]); // one round of slot 0 between steps
    }
    try eq(@as(usize, 0), x.steps.whole); // every pass stepped, none whole
    try eq(@as(usize, 2), x.steps.passes);
    std.debug.print("prefill_interleave users C2 1:1: slot 0's position at slot 1's 6 steps {d} {d} {d} {d} {d} {d}\n", .{ bs[0].positions[0], bs[1].positions[0], bs[2].positions[0], bs[3].positions[0], bs[4].positions[0], bs[5].positions[0] });
}

const Cancel = struct {
    engine: api.Engine,
    id: api.Id,
    slot: u8,
    at: usize,
    gate: ?*std.atomic.Value(bool) = null,
    entered: ?*std.atomic.Value(bool) = null,
    fn hookAt(ctx: *anyopaque, key: Key, k: usize) void {
        const c: *Cancel = @ptrCast(@alignCast(ctx));
        if (key.slot == c.slot and k == 0) if (c.gate) |g| {
            c.entered.?.store(true, .release);
            for (0..5000) |_| {
                if (g.load(.acquire)) break;
                std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            }
        };
        if (key.slot == c.slot and k == c.at) c.engine.cancel(c.id);
    }
};

test "prefill_interleave users: a prompt cancelled between steps releases its slot at once; the next request takes the slot's next generation; its reply and the live user's are their solo replies" {
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var x: Served = .{}; // after the boxes: its deinit (the host's stop) runs before theirs
    try x.init(2, .{});
    defer x.deinit();
    const e = x.host.engine();
    var cancel: Cancel = .{ .engine = e, .id = 2, .slot = 1, .at = 2 };
    x.steps.hook = .{ .ctx = &cancel, .at = Cancel.hookAt };
    try e.submit(1, &ra, a.sink());
    try a.waitTokens(3);
    try e.submit(2, &rb30, b.sink());
    try eq(api.Reason.cancelled, try b.wait());
    try eq(@as(usize, 0), b.tokens.items.len);
    cancel.at = 1000;
    try e.submit(3, &rc30, c.sink());
    try eq(api.Reason.length, try c.wait());
    try expectSolo(&pc, c.tokens.items);
    e.cancel(1);
    try eq(api.Reason.cancelled, try a.wait());
    try expectSolo(&pa, a.tokens.items);
    var out: [16]Ev = undefined;
    try eq(@as(usize, 3), x.stepsOf(1, 1, &out).len); // steps 0..2, none after the cancel
    try eq(@as(usize, 6), x.stepsOf(1, 2, &out).len); // the next request: the slot's next generation, a whole stepped pass
    var released = false;
    var after = false;
    for (x.steps.log.items) |ev| {
        if (ev.kind == .release and ev.slot == 1 and ev.generation == 1) released = true;
        if (ev.kind == .step and ev.slot == 1 and ev.generation == 1 and released) after = true;
    }
    try std.testing.expect(released and !after);
}

test "prefill_interleave users: a failed step is sticky for every owner, as a failed whole prefill: the prompt and the live user fail, and a later request is refused before any Steps call" {
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var d: Box = .{};
    defer d.deinit();
    var x: Served = .{}; // after the boxes: its deinit (the host's stop) runs before theirs
    try x.init(2, .{});
    defer x.deinit();
    const e = x.host.engine();
    try e.submit(1, &ra, a.sink());
    try a.waitTokens(3);
    x.steps.fail_step = 2;
    try e.submit(2, &rb30, b.sink());
    try eq(api.Reason.failed, try b.wait());
    try eq(api.Reason.failed, try a.wait());
    const before = x.steps.log.items.len;
    try e.submit(4, &rc30, d.sink());
    try eq(api.Reason.failed, try d.wait());
    try eq(before, x.steps.log.items.len); // refused before Steps
    try eq(@as(usize, 0), d.tokens.items.len);
}

test "prefill_interleave users C4: two prompts arriving together fill oldest first, a step a round, while the live users decode; replies are their solo replies" {
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var x: Served = .{}; // after the boxes: its deinit (the host's stop) runs before theirs
    try x.init(4, .{});
    defer x.deinit();
    const e = x.host.engine();
    var open: std.atomic.Value(bool) = .init(false);
    var entered: std.atomic.Value(bool) = .init(false);
    defer open.store(true, .release);
    var gate: Cancel = .{ .engine = e, .id = 0, .slot = 1, .at = 1000, .gate = &open, .entered = &entered };
    x.steps.hook = .{ .ctx = &gate, .at = Cancel.hookAt };
    try e.submit(1, &ra, a.sink());
    try a.waitTokens(3);
    try e.submit(2, &rb40, b.sink());
    for (0..5000) |_| {
        if (entered.load(.acquire)) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(entered.load(.acquire));
    try e.submit(3, &rc30, c.sink());
    open.store(true, .release);
    try eq(api.Reason.length, try b.wait());
    try eq(api.Reason.length, try c.wait());
    try expectSolo(&pb, b.tokens.items);
    try expectSolo(&pc, c.tokens.items);
    e.cancel(1);
    try eq(api.Reason.cancelled, try a.wait());
    try expectSolo(&pa, a.tokens.items);
    // the record: slot 1's six steps, then slot 2's six, slot 0 a round between every two; slot 1 decodes while 2 fills
    var seq: [16]Ev = undefined;
    var n: usize = 0;
    for (x.steps.log.items) |ev| if (ev.kind == .step and ev.slot != 0) {
        seq[n] = ev;
        n += 1;
    };
    try eq(@as(usize, 12), n);
    for (seq[0..n], 0..) |ev, k| {
        try eq(@as(u8, if (k < 6) 1 else 2), ev.slot);
        if (k > 0) try eq(@as(usize, 1), ev.positions[0] - seq[k - 1].positions[0]);
    }
    try std.testing.expect(seq[11].positions[1] > seq[6].positions[1]);
}
