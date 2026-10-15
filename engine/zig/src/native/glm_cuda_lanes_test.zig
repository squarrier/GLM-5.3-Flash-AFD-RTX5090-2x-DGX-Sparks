//! The CUDA GLM's lane backend (glm_cuda_lanes.zig) under the lane core, over a scripted forward: one stream's calls,
//! the words rank 1 is sent and its replay, and the refusals, on CPU.
const std = @import("std");
const lanes = @import("lanes");
const gl = @import("glm_cuda_lanes.zig");

const testing = std.testing;
const gpa = testing.allocator;

/// A scripted forward with the Steps calls: the first token and each step's are `reply`'s in turn; every call logged.
const Script = struct {
    reply: []const u32,
    len: usize = 0,
    at: usize = 0,
    fail_prefill: bool = false,
    calls: std.ArrayList(u32) = .empty, // begin: 0xB000_0000 + ids, step: the token, release: 0xE000_0000

    fn deinit(s: *Script) void {
        s.calls.deinit(gpa);
    }

    pub fn prefill(s: *Script, ids: []const u32) !u32 {
        try s.calls.append(gpa, 0xB000_0000 + @as(u32, @intCast(ids.len)));
        try s.calls.appendSlice(gpa, ids);
        if (s.fail_prefill) return error.Injected;
        s.len = ids.len;
        s.at = 0;
        return s.reply[0];
    }

    pub fn step(s: *Script, token: u32) !u32 {
        try s.calls.append(gpa, token);
        s.len += 1;
        s.at += 1;
        return s.reply[@min(s.at, s.reply.len - 1)];
    }

    pub fn release(s: *Script) void {
        s.calls.append(gpa, 0xE000_0000) catch {};
        s.len = 0;
    }
};

/// What rank 1 is sent, in order.
const Recorder = struct {
    ops: std.ArrayList(gl.Op) = .empty,
    words: std.ArrayList([]u32) = .empty,

    fn deinit(r: *Recorder) void {
        for (r.words.items) |w| gpa.free(w);
        r.words.deinit(gpa);
        r.ops.deinit(gpa);
    }

    fn mirror(r: *Recorder) gl.Mirror {
        return .{ .ctx = r, .send = send };
    }

    fn send(ctx: ?*anyopaque, op: gl.Op, words: []const u32) anyerror!void {
        const r: *Recorder = @ptrCast(@alignCast(ctx.?));
        try r.ops.append(gpa, op);
        try r.words.append(gpa, try gpa.dupe(u32, words));
    }
};

const Backend = gl.Lanes(Script);

/// A stream's emitted tokens and finish, run through the lane core over `b`.
fn run(b: *Backend, prompt: []const u32, max_new: u32, sampling: ?lanes.Sampling) !struct { emitted: []u32, reason: lanes.stream.Reason } {
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = max_new, .eos = &.{99}, .sampling = sampling, .drafts = false });
    defer s.deinit(gpa);
    try core.addStream(&s);
    while (core.activeCount() > 0) try core.step();
    return .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason };
}

test "one stream: prefill, then one step a round to eos, each call's words sent first, and rank 1's replay makes the same calls" {
    var script: Script = .{ .reply = &.{ 10, 11, 12, 99 } };
    defer script.deinit();
    var rec: Recorder = .{};
    defer rec.deinit();
    var b: Backend = .{ .steps = &script, .mirror = rec.mirror(), .capacity = 64, .reach = 64 };
    const got = try run(&b, &.{ 5, 6, 7 }, 16, null);
    defer gpa.free(got.emitted);
    try testing.expectEqualSlices(u32, &.{ 10, 11, 12, 99 }, got.emitted);
    try testing.expectEqual(lanes.stream.Reason.stop, got.reason);
    try testing.expectEqualSlices(gl.Op, &.{ .begin, .step, .step, .step, .release }, rec.ops.items);
    try testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, rec.words.items[0]);
    for ([_]u32{ 10, 11, 12 }, rec.words.items[1..4]) |t, w| try testing.expectEqualSlices(u32, &.{t}, w);
    try testing.expectEqual(@as(?*lanes.Stream, null), b.owner);
    // rank 1: the same words on its own forward make the same calls
    var follower: Script = .{ .reply = &.{ 10, 11, 12, 99 } };
    defer follower.deinit();
    for (rec.ops.items, rec.words.items) |op, w| try gl.replay(Script, &follower, op, w);
    try testing.expectEqualSlices(u32, script.calls.items, follower.calls.items);
    // the slot is free for the next request, which starts afresh
    const again = try run(&b, &.{ 8, 9 }, 2, null);
    defer gpa.free(again.emitted);
    try testing.expectEqualSlices(u32, &.{ 10, 11 }, again.emitted);
    try testing.expectEqual(lanes.stream.Reason.length, again.reason);
    try testing.expectEqualSlices(gl.Op, &.{ .begin, .step, .release }, rec.ops.items[5..]);
}

test "a sampled request, an empty one and ones past the capacity or the dense reach are refused before rank 1 hears of them" {
    var script: Script = .{ .reply = &.{ 10, 99 } };
    defer script.deinit();
    var rec: Recorder = .{};
    defer rec.deinit();
    var b: Backend = .{ .steps = &script, .mirror = rec.mirror(), .capacity = 64, .reach = 32 };
    try testing.expectError(error.GreedyOnly, run(&b, &.{ 5, 6 }, 4, .{ .seed = 1, .temperature = 0.7 }));
    try testing.expectError(error.ContextFull, run(&b, &.{ 5, 6 }, 63, null));
    try testing.expectError(error.PastDenseLimit, run(&b, &.{ 5, 6 }, 31, null));
    try testing.expectEqual(@as(usize, 0), rec.ops.items.len);
    try testing.expectEqual(@as(usize, 0), script.calls.items.len);
    // temperature 0 is greedy: served
    const got = try run(&b, &.{ 5, 6 }, 4, .{ .seed = 1, .temperature = 0 });
    defer gpa.free(got.emitted);
    try testing.expectEqualSlices(u32, &.{ 10, 99 }, got.emitted);
    try testing.expectEqualStrings("the native CUDA GLM-5.3-Flash engine decodes greedily only: send temperature 0 (or serve with --temperature 0)", gl.explain(null, error.GreedyOnly).?);
}

test "a failed prefill frees the slot on both ranks" {
    var script: Script = .{ .reply = &.{ 10, 99 }, .fail_prefill = true };
    defer script.deinit();
    var rec: Recorder = .{};
    defer rec.deinit();
    var b: Backend = .{ .steps = &script, .mirror = rec.mirror(), .capacity = 64, .reach = 64 };
    try testing.expectError(error.Injected, run(&b, &.{ 5, 6, 7 }, 4, null));
    try testing.expectEqual(@as(?*lanes.Stream, null), b.owner);
    try testing.expectEqualSlices(gl.Op, &.{ .begin, .release }, rec.ops.items);
    script.fail_prefill = false;
    const got = try run(&b, &.{ 5, 6, 7 }, 4, null);
    defer gpa.free(got.emitted);
    try testing.expectEqualSlices(u32, &.{ 10, 99 }, got.emitted);
}

test "a window the one-stream backend never makes is refused" {
    var script: Script = .{ .reply = &.{10} };
    defer script.deinit();
    var b: Backend = .{ .steps = &script, .capacity = 64, .reach = 64 };
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2 }, .max_new = 4, .drafts = false });
    defer s.deinit(gpa);
    const be = b.backend();
    try be.prefill(&s);
    try testing.expectError(error.NoFreeSlot, be.prefill(&s));
    try testing.expectEqual(@as(u64, 10), try be.first(&s, 2));
    var sampled: [2]u32 = undefined;
    var drafts: [1]u32 = undefined;
    var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &drafts }};
    const two = [_]lanes.backend.Window{.{ .stream = &s, .pending = 10, .held = 0, .tokens = &.{11}, .parents = null, .positions = &.{ 3, 4 } }};
    try testing.expectError(error.DraftsUnsupported, be.verify(&two, &out));
    const late = [_]lanes.backend.Window{.{ .stream = &s, .pending = 10, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{4} }};
    try testing.expectError(error.PositionMismatch, be.verify(&late, &out));
    try testing.expectError(error.Unsupported, be.queue(&s, .{ .value = 10 }, 3));
    be.release(&s);
    try testing.expectEqual(@as(?*lanes.Stream, null), b.owner);
}
