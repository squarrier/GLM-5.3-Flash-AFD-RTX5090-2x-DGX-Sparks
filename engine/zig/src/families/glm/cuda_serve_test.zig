//! qualification_qualification_44ba554e.8 day 1's check of the serve calls (cuda_serve.zig): a served request's prefill and steps make, launch for
//! launch, what decode.prefill and Engine.serialDecode make for the same prompt (tf-glm-forward's `first`, then
//! python's serial_decode, which qualification_66fa03c8/qualification_fefbc9d8 checked on two Sparks), on a TP2 rank cut as the prompt fixture's.
const std = @import("std");
const serve = @import("cuda_serve.zig");
const rec = @import("cuda_serve_rec.zig");

const testing = std.testing;
const expectSame = rec.expectSame;

test "a served request makes decode.prefill's and serial_decode's launches, one step a call; a second request starts afresh" {
    const gpa = testing.allocator;
    const tool = try rec.Rank.init(gpa, null, null);
    defer tool.deinit(gpa);
    const served = try rec.Rank.init(gpa, null, null);
    defer served.deinit(gpa);
    var prompt: [37]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(1000 + 7 * i);
    // the tool's way: prefill, then serialDecode's 5 tokens (the first and 4 steps), no eos stop
    var f = tool.recording();
    const first = try tool.e.prefill(&f, &prompt, false);
    var out: [5]u32 = undefined;
    const toks = try tool.e.serialDecode(&f, first, 5, &out, false);
    try f.flush();
    // the server's way: the same prompt, then one step a round, each fed the last token drawn
    var g = served.recording();
    var s: serve.Serve = .{ .e = served.e, .f = &g };
    var got: [5]u32 = undefined;
    got[0] = try s.prefill(&prompt);
    for (1..5) |k| got[k] = try s.step(got[k - 1]);
    try g.flush();
    try testing.expectEqualSlices(u32, toks, &got);
    try expectSame(&tool.space, tool.t.events.items, &served.space, served.t.events.items);
    try testing.expectEqual(prompt.len + 4, served.e.st.pos);
    try testing.expectEqual(prompt.len + 4, s.len);
    try testing.expect(served.t.events.items.len > 0);
    // a second request on the same Engine: State.reset, then the same launches as a fresh prefill
    s.release();
    try testing.expectError(error.NoSequence, s.step(0));
    served.clear();
    _ = try s.prefill(prompt[0..9]);
    try g.flush();
    try testing.expectEqual(@as(usize, 9), served.e.st.pos);
    const fresh = try rec.Rank.init(gpa, null, null);
    defer fresh.deinit(gpa);
    var h = fresh.recording();
    _ = try fresh.e.prefill(&h, prompt[0..9], false);
    try h.flush();
    try expectSame(&fresh.space, fresh.t.events.items, &served.space, served.t.events.items);
    try testing.expectError(error.EmptyPrompt, s.prefill(&.{}));
}

test "the serve calls refuse a sequence past the Engine's capacity" {
    const gpa = testing.allocator;
    const r = try rec.Rank.init(gpa, null, null);
    defer r.deinit(gpa);
    var f = r.recording();
    var s: serve.Serve = .{ .e = r.e, .f = &f };
    const cap = r.e.st.capacity;
    const long = try gpa.alloc(u32, cap);
    defer gpa.free(long);
    @memset(long, 1000);
    try testing.expectError(error.PastCapacity, s.prefill(long));
    try testing.expect(!s.open);
}
