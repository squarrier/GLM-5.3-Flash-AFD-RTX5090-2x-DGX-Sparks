//! prefill_interleave through the real LaneHost: with a backend that fills a step a call (TensorFold #623's `prefill_step`,
//! @BlivionIaG's lane-core slice merged by Ash Hart), a new prompt goes in a step at a time between the live streams'
//! rounds, oldest first, at the host's fill policy (R rounds, then C steps); the recorder (the fake target's prompt
//! hook) shows the alternation; replies equal their solo replies; a cancel or a failed step ends that request alone and
//! the next request is correct. Without `prefill_step` (today's backends, the default) the whole pass runs inside one
//! admission and every live stream waits for it. Claude Opus 5.5 (planning and execution), via Hermes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

const gpa = std.testing.allocator;
const io = std.testing.io;
const eq = std.testing.expectEqual;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?Reason = null,
    message: []const u8 = "",
    target: ?*lanes.fake.Fake = null,
    lanes_at_end: usize = 0, // the target's lanes when this request finished (on the host's thread)

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| {
                b.done = f.reason;
                b.message = f.message;
                if (b.target) |x| b.lanes_at_end = x.lanes.count();
            },
            else => {},
        }
    }

    fn count(b: *Box) usize {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.tokens.items.len;
    }

    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }

    fn waitTokens(b: *Box, n: usize) void {
        while (b.count() < n) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }

    fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
};

/// The prompt hook's record: which prompt (its first token), which step, and every box's tokens at that moment.
const Rec = struct {
    marks: [256]Mark = undefined,
    n: usize = 0,
    boxes: [3]*Box,
    core: ?*lanes.Engine = null, // its steps: the rounds of the live streams run so far
    cancel: ?struct { engine: api.Engine, id: Id, first: u32, at: usize } = null,
    /// held at this prompt's step 0 (on the host's thread) until the test opens it: later arrivals queue behind it
    gate: ?struct { first: u32, entered: *std.atomic.Value(bool), open: *std.atomic.Value(bool) } = null,

    const Mark = struct { first: u32, step: usize, seen: [3]usize, rounds: u64 };

    fn call(ctx: *anyopaque, s: *lanes.Stream, step: usize) void {
        const r: *Rec = @ptrCast(@alignCast(ctx));
        var seen: [3]usize = undefined;
        for (r.boxes, &seen) |b, *v| v.* = b.count();
        if (r.n < r.marks.len) r.marks[r.n] = .{ .first = s.prompt()[0], .step = step, .seen = seen, .rounds = if (r.core) |c| c.steps else 0 };
        r.n += 1;
        if (r.cancel) |c| if (c.first == s.prompt()[0] and c.at == step) c.engine.cancel(c.id);
        if (r.gate) |g| if (g.first == s.prompt()[0] and step == 0) {
            g.entered.store(true, .release);
            for (0..5000) |_| {
                if (g.open.load(.acquire)) break;
                std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
            }
        };
    }

    /// The steps of the prompt starting with `first`, in order.
    fn of(r: *const Rec, first: u32, out: []Mark) []Mark {
        var k: usize = 0;
        for (r.marks[0..@min(r.n, r.marks.len)]) |m| if (m.first == first) {
            out[k] = m;
            k += 1;
        };
        return out[0..k];
    }
};

/// The fake target's greedy (or keyed) reply to `prompt`, `n` tokens: what a request alone gets.
fn solo(prompt: []const u32, sampling: ?lanes.Sampling, n: usize) ![]u32 {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, prompt);
    const out = try gpa.alloc(u32, n);
    for (out) |*t| {
        t.* = lanes.fake.next(history.items, sampling, history.items.len);
        try history.append(gpa, t.*);
    }
    return out;
}

fn expectSolo(prompt: []const u32, sampling: ?lanes.Sampling, got: []const u32) !void {
    const want = try solo(prompt, sampling, got.len);
    defer gpa.free(want);
    try std.testing.expectEqualSlices(u32, want, got);
}

const pa = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
const pb = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
const pc = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8 };
const keyed: lanes.Sampling = .{ .seed = 11, .temperature = 0.7, .top_k = 5 };

const Fixture = struct {
    cfg: lanes.Config = undefined,
    target: lanes.fake.Fake = undefined,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: LaneHost = undefined,

    fn init(x: *Fixture, stepped: bool, chunks: usize, n: u32, policy: LaneHost.FillPolicy) !void {
        x.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        x.target = .{ .gpa = gpa, .prefill_chunks = chunks };
        x.clock = .{};
        x.core = lanes.Engine.init(gpa, &x.cfg, if (stepped) x.target.stepped() else x.target.backend(), x.clock.clock());
        x.host = LaneHost.init(gpa, io, &x.core, .{ .lanes = n });
        x.host.fill_policy = policy;
    }

    fn deinit(x: *Fixture) void {
        x.core.deinit();
        x.target.deinit();
        x.cfg.deinit(gpa);
    }
};

/// A live greedy decoder (a), then a second prompt (b, 8 steps) while it decodes; the record of b's steps.
fn alternation(stepped: bool, policy: LaneHost.FillPolicy, b_sampling: ?lanes.Sampling, marks: []Rec.Mark) !usize {
    var x: Fixture = .{};
    try x.init(stepped, 8, 2, policy);
    defer x.deinit();
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var none: Box = .{};
    defer none.deinit();
    var rec: Rec = .{ .boxes = .{ &a, &b, &none }, .core = &x.core };
    x.target.prefill_hook = Rec.call;
    x.target.prefill_hook_ctx = &rec;
    try x.host.start();
    defer x.host.stop();
    const e = x.host.engine();
    const long: Request = .{ .prompt = &pa, .max_tokens = 100000, .drafts = false };
    const second: Request = .{ .prompt = &pb, .max_tokens = 24, .drafts = false, .sampling = b_sampling };
    try e.submit(1, &long, .{ .ctx = &a, .event = Box.event });
    a.waitTokens(3);
    try e.submit(2, &second, .{ .ctx = &b, .event = Box.event });
    try eq(Reason.length, b.wait());
    try eq(@as(usize, 24), b.tokens.items.len);
    try expectSolo(&pb, b_sampling, b.tokens.items);
    e.cancel(1);
    try eq(Reason.cancelled, a.wait());
    try expectSolo(&pa, null, a.tokens.items);
    const got = rec.of(pb[0], marks);
    return got.len;
}

test "prefill_interleave a lane host keeps a live stream committing while another prompt fills a step a round (1:1, 2:1, 1:2): the recorder shows the alternation, and both replies are their solo replies" {
    for ([_]LaneHost.FillPolicy{ .{ .rounds = 1, .steps = 1 }, .{ .rounds = 2, .steps = 1 }, .{ .rounds = 1, .steps = 2 } }, [_]?lanes.Sampling{ null, keyed, null }) |policy, sampling| {
        var marks: [64]Rec.Mark = undefined;
        const n = try alternation(true, policy, sampling, &marks);
        try eq(@as(usize, 8), n); // the prompt's eight steps, each its own call
        var seen: [8]usize = undefined;
        for (marks[0..n], 0..) |m, k| {
            try eq(k, m.step);
            seen[k] = m.seen[0];
            if (k == 0) continue;
            const turn = k % policy.steps == 0; // a new fill turn: the live stream's `rounds` rounds came first
            try eq(@as(u64, if (turn) policy.rounds else 0), m.rounds - marks[k - 1].rounds);
            try eq(@as(usize, if (turn) policy.rounds else 0), m.seen[0] - marks[k - 1].seen[0]); // one token a round, alone
            try eq(@as(usize, 0), m.seen[1]); // the filling request has no token before its last step
        }
        std.debug.print("prefill_interleave host {d}:{d}: the decoder's tokens at the prompt's 8 steps {any}\n", .{ policy.rounds, policy.steps, seen });
    }
}

test "prefill_interleave default off: a backend without prefill_step runs the whole pass inside one admission, and the live stream gets no round while it prefills (today's stall)" {
    var marks: [64]Rec.Mark = undefined;
    const n = try alternation(false, .{}, null, &marks);
    try eq(@as(usize, 8), n);
    for (marks[1..n]) |m| try eq(marks[0].seen[0], m.seen[0]);
    std.debug.print("prefill_interleave host off: the decoder's tokens at the prompt's 8 chunks all {d}\n", .{marks[0].seen[0]});
}

test "prefill_interleave a prompt cancelled at a step finishes cancelled with its lane released; the live stream goes on and the next request is its solo self" {
    var x: Fixture = .{};
    try x.init(true, 8, 2, .{});
    defer x.deinit();
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{ .target = &x.target };
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var rec: Rec = .{ .boxes = .{ &a, &b, &c } };
    x.target.prefill_hook = Rec.call;
    x.target.prefill_hook_ctx = &rec;
    try x.host.start();
    defer x.host.stop();
    const e = x.host.engine();
    rec.cancel = .{ .engine = e, .id = 2, .first = pb[0], .at = 3 };
    const long: Request = .{ .prompt = &pa, .max_tokens = 100000, .drafts = false };
    try e.submit(1, &long, .{ .ctx = &a, .event = Box.event });
    a.waitTokens(3);
    const second: Request = .{ .prompt = &pb, .max_tokens = 24, .drafts = false };
    try e.submit(2, &second, .{ .ctx = &b, .event = Box.event });
    try eq(Reason.cancelled, b.wait());
    try eq(@as(usize, 0), b.tokens.items.len);
    try eq(@as(usize, 1), b.lanes_at_end); // only the live stream's lane is held once it ended
    var steps: [64]Rec.Mark = undefined;
    try eq(@as(usize, 4), rec.of(pb[0], &steps).len); // steps 0..3, none after the cancel
    const third: Request = .{ .prompt = &pc, .max_tokens = 24, .drafts = false };
    try e.submit(3, &third, .{ .ctx = &c, .event = Box.event });
    try eq(Reason.length, c.wait());
    try expectSolo(&pc, null, c.tokens.items);
    try eq(@as(usize, 8), rec.of(pc[0], &steps).len);
    e.cancel(1);
    try eq(Reason.cancelled, a.wait());
    try expectSolo(&pa, null, a.tokens.items);
}

test "prefill_interleave a failed prompt step fails that request alone, in the backend's words; the live stream goes on and the next request is its solo self" {
    var x: Fixture = .{};
    try x.init(true, 8, 2, .{});
    defer x.deinit();
    const Words = struct {
        fn text(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return if (err == error.PrefillStepFailed) "the prompt step failed" else null;
        }
    };
    x.host.explain = .{ .text = Words.text };
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{ .target = &x.target };
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var rec: Rec = .{ .boxes = .{ &a, &b, &c } };
    x.target.prefill_hook = Rec.call;
    x.target.prefill_hook_ctx = &rec;
    const long: Request = .{ .prompt = &pa, .max_tokens = 100000, .drafts = false };
    x.target.step_fail = null; // the decoder's own pass goes in whole steps
    try x.host.start();
    defer x.host.stop();
    const e = x.host.engine();
    try e.submit(1, &long, .{ .ctx = &a, .event = Box.event });
    a.waitTokens(3);
    x.target.step_fail = 2; // read on the host's thread at the next prompt's steps (the decoder's pass is done)
    const second: Request = .{ .prompt = &pb, .max_tokens = 24, .drafts = false };
    try e.submit(2, &second, .{ .ctx = &b, .event = Box.event });
    try eq(Reason.failed, b.wait());
    try std.testing.expectEqualStrings("the prompt step failed", b.message);
    try eq(@as(usize, 1), b.lanes_at_end);
    const before = a.count();
    a.waitTokens(before + 3); // the live stream still decodes
    x.target.step_fail = null;
    const third: Request = .{ .prompt = &pc, .max_tokens = 24, .drafts = false };
    try e.submit(3, &third, .{ .ctx = &c, .event = Box.event });
    try eq(Reason.length, c.wait());
    try expectSolo(&pc, null, c.tokens.items);
    e.cancel(1);
    try eq(Reason.cancelled, a.wait());
    try expectSolo(&pa, null, a.tokens.items);
}

test "prefill_interleave two prompts arriving together fill oldest first, a step a round; the first decodes while the second fills; a queued filling request cancelled before its first step ends at once; replies are their solo replies" {
    var x: Fixture = .{};
    try x.init(true, 6, 4, .{});
    defer x.deinit();
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var d: Box = .{ .target = &x.target };
    defer d.deinit();
    var rec: Rec = .{ .boxes = .{ &a, &b, &c }, .core = &x.core };
    x.target.prefill_hook = Rec.call;
    x.target.prefill_hook_ctx = &rec;
    try x.host.start();
    defer x.host.stop();
    const e = x.host.engine();
    const pd = [_]u32{ 6, 6, 6, 1, 2, 3 };
    var entered: std.atomic.Value(bool) = .init(false);
    var open: std.atomic.Value(bool) = .init(false);
    defer open.store(true, .release);
    rec.gate = .{ .first = pb[0], .entered = &entered, .open = &open }; // b held at its step 0 while c and d arrive
    rec.cancel = .{ .engine = e, .id = 4, .first = pb[0], .at = 1 }; // d waits behind b and c, cancelled at b's step 1
    const long: Request = .{ .prompt = &pa, .max_tokens = 100000, .drafts = false };
    try e.submit(1, &long, .{ .ctx = &a, .event = Box.event });
    a.waitTokens(3);
    const rb: Request = .{ .prompt = &pb, .max_tokens = 40, .drafts = false };
    const rc: Request = .{ .prompt = &pc, .max_tokens = 24, .drafts = false, .sampling = keyed };
    const rd: Request = .{ .prompt = &pd, .max_tokens = 24, .drafts = false };
    try e.submit(2, &rb, .{ .ctx = &b, .event = Box.event });
    for (0..5000) |_| {
        if (entered.load(.acquire)) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(entered.load(.acquire));
    try e.submit(3, &rc, .{ .ctx = &c, .event = Box.event });
    try e.submit(4, &rd, .{ .ctx = &d, .event = Box.event });
    open.store(true, .release);
    try eq(Reason.length, b.wait());
    try eq(Reason.length, c.wait());
    try eq(Reason.cancelled, d.wait());
    try eq(@as(usize, 0), d.tokens.items.len);
    try expectSolo(&pb, null, b.tokens.items);
    try expectSolo(&pc, keyed, c.tokens.items);
    var bm: [64]Rec.Mark = undefined;
    var cm: [64]Rec.Mark = undefined;
    var dm: [64]Rec.Mark = undefined;
    const bs = rec.of(pb[0], &bm);
    const cs = rec.of(pc[0], &cm);
    try eq(@as(usize, 6), bs.len);
    try eq(@as(usize, 6), cs.len);
    try eq(@as(usize, 0), rec.of(pd[0], &dm).len); // never stepped
    // oldest first: every step of b before c's first, and the live stream's round between every two steps
    const all = rec.marks[0..rec.n];
    var i: usize = 0;
    while (all[i].first != pb[0]) i += 1;
    errdefer for (all, 0..) |m, j| std.debug.print("mark {d}: prompt {d} step {d} decoder {d} first-prompt {d} rounds {d}\n", .{ j, m.first, m.step, m.seen[0], m.seen[1], m.rounds });
    for (0..12) |k| {
        try eq(if (k < 6) pb[0] else pc[0], all[i + k].first);
        if (k == 0) continue;
        try eq(@as(u64, 1), all[i + k].rounds - all[i + k - 1].rounds); // one round of the live streams between every two steps
        try std.testing.expect(all[i + k].seen[0] > all[i + k - 1].seen[0]); // the decoder committed in it
    }
    for (all[i..][0..12]) |m| std.debug.print("prefill_interleave host two prompts: prompt {d} step {d} rounds {d} decoder {d} first {d}\n", .{ m.first, m.step, m.rounds, m.seen[0], m.seen[1] });
    try std.testing.expect(cs[5].seen[1] > cs[0].seen[1]); // b decodes while c fills
    e.cancel(1);
    try eq(Reason.cancelled, a.wait());
    try expectSolo(&pa, null, a.tokens.items);
}
