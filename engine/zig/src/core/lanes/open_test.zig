//! prefill_interleave: a prompt pass a step at a time between the live streams' rounds (TensorFold #623's `prefill_step` and
//! `fillStream`, @BlivionIaG's lane-core slice merged by Ash Hart, ported onto this head) on the fake target, whose
//! tokens depend on all fed history: a stepped pass opens as the whole pass does (the same reply, greedy and sampled,
//! drafts on and off); a live stream's rounds run between another prompt's steps and both replies equal their solo
//! replies; a cancel or a failure between steps releases the lane and the next stream is correct; a backend without
//! `prefill_step` never fills. Claude Opus 5.5 (planning and execution), via Hermes.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;
const eq = std.testing.expectEqual;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
};

/// engine_test.zig's model: drafted rounds on, so a step between rounds meets the drafts' held state too.
fn model() !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

const Flag = struct {
    on: bool = false,
    fn check(ctx: *anyopaque) bool {
        const f: *Flag = @ptrCast(@alignCast(ctx));
        return f.on;
    }
};

const Streams = struct {
    s: []sm.Stream,
    p: []SuffixLookup,
    flags: []Flag,

    fn init(cases: []const Case) !Streams {
        const x: Streams = .{ .s = try gpa.alloc(sm.Stream, cases.len), .p = try gpa.alloc(SuffixLookup, cases.len), .flags = try gpa.alloc(Flag, cases.len) };
        for (cases, x.s, x.p, x.flags) |c, *s, *p, *fl| {
            fl.* = .{};
            p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
            s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .cancel_check = .{ .ptr = fl, .check = Flag.check } }); // no EOS: every reply runs to max_new
        }
        return x;
    }

    fn deinit(x: Streams) void {
        for (x.s, x.p) |*s, *p| {
            s.deinit(gpa);
            p.deinit();
        }
        gpa.free(x.s);
        gpa.free(x.p);
        gpa.free(x.flags);
    }
};

/// One case alone, its whole pass (`addStream`) on the plain backend, stepped to the end: the reference reply.
fn solo(c: Case) ![]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const x = try Streams.init(&.{c});
    defer x.deinit();
    try engine.addStream(&x.s[0]);
    while (engine.activeCount() > 0) try engine.step();
    return gpa.dupe(u32, x.s[0].emitted());
}

const p1 = [_]u32{ 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4 };
const p2 = [_]u32{ 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p3 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5, 3 };
const keyed: Sampling = .{ .seed = 7, .temperature = 0.8, .top_k = 5 };

test "prefill_interleave fillStream opens a stream as addStream does: the same reply a step at a time, greedy and sampled, drafts on and off, 1 to 9 steps" {
    const cases = [_]Case{ .{ .prompt = &p1 }, .{ .prompt = &p2, .drafts = false }, .{ .prompt = &p3, .sampling = keyed }, .{ .prompt = &p1, .sampling = keyed, .drafts = false, .max_new = 25 } };
    for (cases) |c| for ([_]usize{ 1, 2, 5, 9 }) |chunks| {
        const want = try solo(c);
        defer gpa.free(want);
        var cfg = try model();
        defer cfg.deinit(gpa);
        var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = chunks };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.stepped(), clock.clock());
        defer engine.deinit();
        try std.testing.expect(engine.fills());
        const x = try Streams.init(&.{c});
        defer x.deinit();
        var calls: usize = 0;
        while (true) {
            calls += 1;
            const done = try engine.fillStream(&x.s[0], calls == 1);
            try eq(calls == chunks, done);
            try eq(@as(usize, if (done) 1 else 0), engine.activeCount());
            if (done) break;
        }
        try eq(chunks, target.prefill_count);
        try eq(@as(usize, 1), x.s[0].emitted().len); // opened: its first token, as after addStream
        while (engine.activeCount() > 0) try engine.step();
        try std.testing.expectEqualSlices(u32, want, x.s[0].emitted());
    };
}

/// What ran, in order: a prompt step (its stream and step) or a round (the live streams, the decoder's tokens after it).
const Mark = struct { kind: enum { step, round }, stream: usize = 0, step: usize = 0, live: usize = 0, tokens: usize = 0 };

/// cases[0] opened whole and live; cases[1..] then fill oldest first as LaneHost's fill turns run them: a turn of `steps`
/// steps (the next prompt's from where the last one ended), then `rounds` rounds of the live streams (none when nothing
/// is live); then every stream to its end. The replies, and the marks.
fn interleaved(cases: []const Case, chunks: usize, rounds: usize, steps: usize, marks: *std.ArrayList(Mark)) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = chunks };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer engine.deinit();
    const x = try Streams.init(cases);
    defer x.deinit();
    try engine.addStream(&x.s[0]);
    var next: usize = 1; // the oldest filling stream
    var k: usize = 0; // its next step
    while (next < cases.len) {
        for (0..steps) |_| {
            if (next == cases.len) break;
            const done = try engine.fillStream(&x.s[next], k == 0);
            try marks.append(gpa, .{ .kind = .step, .stream = next, .step = k });
            k += 1;
            if (done) {
                next += 1;
                k = 0;
            }
        }
        if (next == cases.len) break;
        for (0..rounds) |_| if (engine.activeCount() > 0) {
            try engine.step();
            try marks.append(gpa, .{ .kind = .round, .live = engine.activeCount(), .tokens = x.s[0].emitted().len });
        };
    }
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, x.s) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn freeAll(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

test "prefill_interleave a live stream's rounds run between another prompt's steps: it commits between every two steps, and both replies equal their solo replies (1:1, 2:1, 1:2; greedy and sampled)" {
    const sets = [_][2]Case{
        .{ .{ .prompt = &p1, .max_new = 120, .drafts = false }, .{ .prompt = &p3 } },
        .{ .{ .prompt = &p2, .max_new = 120, .sampling = keyed }, .{ .prompt = &p1, .sampling = keyed, .drafts = false } },
    };
    for (sets) |cases| for ([_][2]usize{ .{ 1, 1 }, .{ 2, 1 }, .{ 1, 2 } }) |policy| {
        var marks: std.ArrayList(Mark) = .empty;
        defer marks.deinit(gpa);
        const got = try interleaved(&cases, 6, policy[0], policy[1], &marks);
        defer freeAll(got);
        for (cases, got) |c, g| {
            const want = try solo(c);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, g);
        }
        // the recorder: steps 0..5 of stream 1 in order, `rounds` rounds of the decoder between turns, each round a token
        var steps: usize = 0;
        var last: usize = 1; // the decoder's first token, from its whole pass
        var since: usize = 0;
        for (marks.items) |m| switch (m.kind) {
            .step => {
                try eq(@as(usize, 1), m.stream);
                try eq(steps, m.step);
                if (steps > 0 and steps % policy[1] == 0) try eq(policy[0], since);
                if (steps % policy[1] != 0) try eq(@as(usize, 0), since);
                since = 0;
                steps += 1;
            },
            .round => {
                try eq(@as(usize, 1), m.live); // the filling stream takes no part until it opens
                try std.testing.expect(m.tokens > last); // the decoder committed in every round between steps
                last = m.tokens;
                since += 1;
            },
        };
        try eq(@as(usize, 6), steps);
    };
}

test "prefill_interleave two prompts arriving together fill oldest first, a step a round; the first joins the rounds while the second fills; all three replies equal their solo replies" {
    const cases = [_]Case{ .{ .prompt = &p2, .max_new = 200, .drafts = false }, .{ .prompt = &p1, .max_new = 30 }, .{ .prompt = &p3, .max_new = 30, .sampling = keyed } };
    var marks: std.ArrayList(Mark) = .empty;
    defer marks.deinit(gpa);
    const got = try interleaved(&cases, 4, 1, 1, &marks);
    defer freeAll(got);
    for (cases, got) |c, g| {
        const want = try solo(c);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, g);
    }
    var order: std.ArrayList(usize) = .empty;
    defer order.deinit(gpa);
    var live_while_second: usize = 0;
    for (marks.items) |m| switch (m.kind) {
        .step => try order.append(gpa, m.stream),
        .round => if (order.items.len > 4) {
            live_while_second = @max(live_while_second, m.live);
        },
    };
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1, 2, 2, 2, 2 }, order.items);
    try eq(@as(usize, 2), live_while_second); // the decoder and the first prompt's stream decode while the second fills
}

test "prefill_interleave a stream cancelled between steps: fillStream gives Cancelled and its lane is released; the live stream and the next stream are their solo selves" {
    const cases = [_]Case{ .{ .prompt = &p1, .max_new = 80, .drafts = false }, .{ .prompt = &p2 }, .{ .prompt = &p3 } };
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 5 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer engine.deinit();
    const x = try Streams.init(&cases);
    defer x.deinit();
    try engine.addStream(&x.s[0]);
    try std.testing.expect(!try engine.fillStream(&x.s[1], true));
    try engine.step();
    try std.testing.expect(!try engine.fillStream(&x.s[1], false));
    try engine.step();
    x.flags[1].on = true; // cancelled between its second and third steps
    try std.testing.expectError(error.Cancelled, engine.fillStream(&x.s[1], false));
    try eq(@as(usize, 1), target.lanes.count()); // only the decoder's lane is held
    try eq(@as(usize, 1), engine.activeCount());
    try engine.step();
    var k: usize = 0;
    while (!try engine.fillStream(&x.s[2], k == 0)) : (k += 1) try engine.step();
    while (engine.activeCount() > 0) try engine.step();
    for ([_]usize{ 0, 2 }) |i| {
        const want = try solo(cases[i]);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, x.s[i].emitted());
    }
    try eq(@as(usize, 0), x.s[1].emitted().len);
    try eq(@as(usize, 0), target.lanes.count());
}

test "prefill_interleave a failed step returns its error with the stream out of the rounds; the host's discard releases the lane; the next stream is its solo self" {
    const cases = [_]Case{ .{ .prompt = &p1, .max_new = 80, .drafts = false }, .{ .prompt = &p2 }, .{ .prompt = &p3, .sampling = keyed } };
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 5, .step_fail = 2 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer engine.deinit();
    const x = try Streams.init(&cases);
    defer x.deinit();
    try engine.addStream(&x.s[0]);
    try std.testing.expect(!try engine.fillStream(&x.s[1], true));
    try engine.step();
    try std.testing.expect(!try engine.fillStream(&x.s[1], false));
    try engine.step();
    try std.testing.expectError(error.PrefillStepFailed, engine.fillStream(&x.s[1], false));
    try eq(@as(usize, 1), engine.activeCount());
    engine.discard(&x.s[1]); // what LaneHost.drop does
    try eq(@as(usize, 1), target.lanes.count());
    target.step_fail = null;
    var k: usize = 0;
    while (!try engine.fillStream(&x.s[2], k == 0)) : (k += 1) try engine.step();
    while (engine.activeCount() > 0) try engine.step();
    for ([_]usize{ 0, 2 }) |i| {
        const want = try solo(cases[i]);
        defer gpa.free(want);
        try std.testing.expectEqualSlices(u32, want, x.s[i].emitted());
    }
    try eq(@as(usize, 0), target.lanes.count());
}

test "prefill_interleave a backend without prefill_step never fills: fills() false, fillStream refused before any call" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    try std.testing.expect(!engine.fills());
    const x = try Streams.init(&.{.{ .prompt = &p1 }});
    defer x.deinit();
    try std.testing.expectError(error.NoPrefillSteps, engine.fillStream(&x.s[0], true));
    try eq(@as(usize, 0), target.prefill_count);
    try eq(@as(usize, 0), target.lanes.count());
    try eq(@as(usize, 0), engine.activeCount());
}
