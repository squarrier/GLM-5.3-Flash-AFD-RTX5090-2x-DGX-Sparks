//! Independent fake shared held/branch/forced row-mask oracle; no GLM head claim.
const std = @import("std");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const a = std.testing.allocator;
const Tree = struct {
    rules: lanes.grammar.Rules = undefined,
    prefix: [2]u32 = undefined,
    len: usize = 0,
    accepts: usize = 0,
    rollbacks: usize = 0,
    closes: usize = 0,
    fn init(t: *Tree) void {
        t.rules = .{ .ptr = t, .width = 64, .vtable = &.{ .fill = fill, .accept = accept, .rollback = undo, .terminated = ended, .close = close } };
    }
    fn self(p: *anyopaque) *Tree { return @ptrCast(@alignCast(p)); }
    fn fill(p: *anyopaque, words: []u32) !void {
        const t = self(p);
        @memset(words, 0);
        if (t.len == 0) words[0] = (1 << 7) | (1 << 9) else words[1] = if (t.prefix[0] == 7) (1 << 8) else (1 << 9);
    }
    fn accept(p: *anyopaque, token: u32) !bool {
        const t = self(p);
        const allowed = if (t.len == 0) token == 7 or token == 9 else if (t.len == 1) token == (if (t.prefix[0] == 7) @as(u32, 40) else 41) else false;
        if (!allowed) return false;
        t.prefix[t.len] = token;
        t.len += 1;
        t.accepts += 1;
        return true;
    }
    fn undo(p: *anyopaque, n: usize) !void {
        const t = self(p);
        if (n > t.len) return error.BadRollback;
        t.len -= n;
        t.rollbacks += n;
    }
    fn ended(p: *anyopaque) bool { return self(p).len == 2; }
    fn close(p: *anyopaque) void { self(p).closes += 1; }
};
test "grammar_tests_r N7 shared held branch rejected and terminated descendants exact full words parent positions rollback" {
    var t = Tree{}; t.init(); defer t.rules.close();
    // Known held token ids are fixture inputs; unknown held readback remains unsupported, not inferred.
    var plan = try lanes.grammar_window.build(a, &t.rules, &.{ 7, 99, 40, 9, 41, 12 }, &.{ -1, 0, 0, 1, 0, 4, 2 }, &.{ 10, 11, 11, 12, 11, 12, 12 });
    defer plan.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 4 }, plan.rows.items);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 0 }, plan.parents.items);
    try std.testing.expectEqualSlices(u64, &.{ 10, 11, 11 }, plan.positions.items);
    try std.testing.expectEqualSlices(u32, &.{ (1 << 7) | (1 << 9), 0, 0, 1 << 8, 0, 1 << 9 }, plan.masks.items);
    try std.testing.expectEqual(@as(usize, 0), t.len);
    try std.testing.expectEqual(t.accepts, t.rollbacks);
    try std.testing.expect(!t.rules.poisoned);
    t.rules.close(); t.rules.close();
    try std.testing.expectEqual(@as(usize, 1), t.closes);
}
test "grammar_tests_r N7 forced close prompt ignored marker activates next token and terminated row pruned" {
    var rules = data.Fake{}; rules.init(64);
    rules.rules.active = false; rules.rules.after = 19;
    var s = try lanes.Stream.init(a, .{ .id = "forced-boundary-oracle", .prompt = &.{ 7, 19, 8 }, .max_new = 9, .eos = &.{7}, .rules = &rules.rules, .think_close = &.{ 18, 19, 20 }, .think_end = 19 });
    defer s.deinit(a);
    try std.testing.expect(!rules.rules.active and rules.accepted == 0);
    try std.testing.expectEqualSlices(u32, &.{ 18, 19 }, s.think_close);
    _ = try s.commit(a, &.{ 91, 18, 19 });
    try std.testing.expect(rules.rules.active and rules.accepted == 0);
    var plan = try lanes.grammar_window.build(a, &rules.rules, &.{ 7, 8 }, &.{ -1, 0, 1 }, &.{ 3, 4, 5 });
    defer plan.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, plan.rows.items);
    try std.testing.expectEqualSlices(u32, &.{ 1 << 7, 0, 1 << 8, 0 }, plan.masks.items);
    try std.testing.expectEqual(@as(usize, 0), rules.accepted);
    _ = try s.commit(a, &.{7});
    try std.testing.expect(!s.finished); // EOS list cannot supersede fake grammar termination.
    _ = try s.commit(a, &.{8});
    try std.testing.expect(s.finished and rules.rules.terminated());
    try std.testing.expectEqualSlices(u32, &.{ 91, 18, 19, 7, 8 }, s.emitted());
}
