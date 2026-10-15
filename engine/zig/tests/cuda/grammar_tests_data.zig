//! grammar_tests independent adversarial CPU values and counted fake Rules, not real grammar.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const a = std.testing.allocator;
// Test-only observations; the original assertions remain the runtime oracle.
pub fn witness(comptime id: []const u8, expected: usize, observed: usize) void {
    std.debug.print("WITNESS {s} expected={d} observed={d}\n", .{ id, expected, observed });
}
pub const Data = struct {
    ranks: []glm.sample.mask_cpu.Rank,
    values: []f32,
    ids: []i64,
    cpu: glm.sample.mask_cpu.Cpu,
    pub fn init(width: usize, world: usize) !Data {
        const values = try a.alloc(f32, width);
        errdefer a.free(values);
        const ids = try a.alloc(i64, width);
        errdefer a.free(ids);
        const ranks = try a.alloc(glm.sample.mask_cpu.Rank, world);
        errdefer a.free(ranks);
        @memset(values, -1000);
        for (ids, 0..) |*id, i| id.* = @intCast(i);
        std.mem.swap(i64, &ids[0], &ids[7]); // mask bits MUST be indexed by mapped global id
        for (ranks, 0..) |*rank, r| rank.* = .{ .values = values[r * (width / world) ..][0 .. width / world], .ids = ids[r * (width / world) ..][0 .. width / world] };
        return .{ .ranks = ranks, .values = values, .ids = ids, .cpu = .{ .gpa = a, .width = width, .ranks = ranks } };
    }
    pub fn value(d: *Data, token: u32, v: f32) void {
        const column = std.mem.indexOfScalar(i64, d.ids, token).?;
        d.values[column] = v;
    }
    pub fn set(d: *Data) void {
        @memset(d.values, -1000);
        d.value(2, 100);
        d.value(7, 80);
        d.value(8, 70);
    }
    pub fn picker(d: *Data) glm.trace.Picker { return .{ .ctx = d, .pick = plain }; }
    fn plain(ptr: *anyopaque, _: glm.trace.T, _: ?[]const u64, out: []u32, _: ?[]f64) !void {
        const d: *Data = @ptrCast(@alignCast(ptr));
        // The fixture's independent full-row numeric argmax; no candidate top-k or candidate sampler call.
        var best: usize = 0;
        for (d.values, d.ids, 0..) |v, id, i| {
            if (v > d.values[best] or (v == d.values[best] and id < d.ids[best])) best = i;
        }
        @memset(out, @as(u32, @intCast(d.ids[best])));
    }
    pub fn deinit(d: *Data) void {
        a.free(d.ranks);
        a.free(d.values);
        a.free(d.ids);
    }
};
pub const Fake = struct {
    rules: lanes.grammar.Rules = undefined,
    tokens: [2]u32 = .{ 7, 8 },
    accepted: usize = 0,
    fills: usize = 0,
    closes: usize = 0,
    fail_fill: bool = false,
    fail_fill_at: ?usize = null,
    fail_rollback: bool = false,
    all: bool = false,
    empty: bool = false,
    pub fn init(x: *Fake, width: usize) void {
        x.rules = .{ .ptr = x, .width = width, .vtable = &.{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close } };
    }
    fn from(p: *anyopaque) *Fake { return @ptrCast(@alignCast(p)); }
    fn fill(p: *anyopaque, words: []u32) !void {
        const x = from(p);
        x.fills += 1;
        if (x.fail_fill or x.fail_fill_at == x.fills) return error.MaskFillFailed;
        @memset(words, if (x.all) std.math.maxInt(u32) else 0);
        if (!x.all and !x.empty and x.accepted < x.tokens.len) {
            const token = x.tokens[x.accepted];
            words[token / 32] |= @as(u32, 1) << @as(u5, @intCast(token % 32));
        }
    }
    fn accept(p: *anyopaque, token: u32) !bool {
        const x = from(p);
        if (!x.all and (x.accepted >= x.tokens.len or x.tokens[x.accepted] != token)) return false;
        x.accepted += 1;
        return true;
    }
    fn rollback(p: *anyopaque, n: usize) !void {
        const x = from(p);
        if (x.fail_rollback) return error.RollbackFailed;
        if (n > x.accepted) return error.BadRollback;
        x.accepted -= n;
    }
    fn terminated(p: *anyopaque) bool { return from(p).accepted >= 2; }
    fn close(p: *anyopaque) void { from(p).closes += 1; }
};
pub fn count(t: *glm.trace.Trace, name: []const u8) usize {
    var n: usize = 0;
    for (t.events.items) |e| if (std.mem.eql(u8, e.name, name)) { n += 1; };
    return n;
}
/// Sweep callers replace every structured request and CPU temporary allocator.
/// Even when admission/step fails, Rules must follow exactly the emitted prefix.
pub fn allocationTransaction(gpa: std.mem.Allocator, backend: lanes.backend.Backend, cfg: *lanes.Config, rules: *Fake) !void {
    var s = try lanes.Stream.init(gpa, .{ .id = "allocation-failure", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &rules.rules });
    defer s.deinit(gpa);
    var clock = lanes.fake.FixedClock{};
    var core = lanes.Engine.init(gpa, cfg, backend, clock.clock());
    defer core.deinit();
    const result = drive(&core, &s);
    try std.testing.expectEqual(rules.accepted, s.emitted().len);
    const expected = [_]u32{ 7, 8 };
    try std.testing.expectEqualSlices(u32, expected[0..s.emitted().len], s.emitted());
    return result;
}
fn drive(core: *lanes.Engine, s: *lanes.Stream) !void {
    try core.addStream(s);
    while (!s.finished) try core.step();
}
pub const PollCancel = struct {
    calls: usize = 0,
    pub fn check(ptr: *anyopaque) bool {
        const x: *PollCancel = @ptrCast(@alignCast(ptr));
        x.calls += 1;
        return x.calls >= 2;
    }
};
pub fn stopFirst(_: *anyopaque, emitted: []const u32) bool { return emitted.len >= 1; }
pub fn selection(backend: lanes.backend.Backend, s: *lanes.Stream) !void {
    const hook = backend.vtable.first_masked orelse return error.IncompleteMaskedAdapter;
    try backend.prefill(s);
    defer backend.release(s);
    const first = try s.rules.?.next(a); defer a.free(first);
    const handle = try hook(backend.ptr, s, s.prompt_len, first);
    // These selection assertions execute BEFORE Stream's defensive accept/output check.
    if (try backend.read(handle) != 7) {
        std.debug.print("FirstAllowedWinnerRuntimeViolation: allowed7 must win before accept\n", .{});
        return error.FirstAllowedWinnerRuntimeViolation;
    }
    _ = try s.commit(a, &.{7});
    s.cache_len = s.prompt_len;
    s.pending = 7;
    const second = try s.rules.?.next(a); defer a.free(second);
    var sampled: [1]u32 = undefined;
    var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
    const window = [_]lanes.backend.Window{.{ .stream = s, .pending = 7, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{s.cache_len + 1}, .masks = second }};
    try backend.verify(&window, &out);
    if (sampled[0] != 8) {
        std.debug.print("SecondAllowedWinnerRuntimeViolation: allowed8 must win before accept\n", .{});
        return error.SecondAllowedWinnerRuntimeViolation;
    }
    _ = try s.commit(a, &.{8});
}
pub fn postCloseSelection(backend: lanes.backend.Backend, s: *lanes.Stream) !void {
    const hook = backend.vtable.first_masked orelse return error.IncompleteMaskedAdapter;
    try backend.prefill(s);
    defer backend.release(s);
    const before = try s.rules.?.next(a); defer a.free(before);
    try std.testing.expectEqual(@as(usize, 0), before.len);
    const first = try hook(backend.ptr, s, s.prompt_len, before);
    const observed_first = try backend.read(first);
    witness("FirstAllowedWinnerRuntimeViolation.N1-cached", 19, observed_first);
    try std.testing.expectEqual(@as(u32, 19), observed_first);
    _ = try s.commit(a, &.{19});
    try std.testing.expect(s.rules.?.active);
    s.cache_len = s.prompt_len;
    s.pending = 19;
    inline for ([_]u32{ 7, 8 }) |expected| {
        const mask = try s.rules.?.next(a); defer a.free(mask);
        var sampled: [1]u32 = undefined;
        var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
        const w = [_]lanes.backend.Window{.{ .stream = s, .pending = s.pending.?, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{s.cache_len + 1}, .masks = mask }};
        try backend.verify(&w, &out);
        witness("PostCloseMaskedSelectionRuntimeViolation.N2-dropped-verify-mask", expected, sampled[0]);
        // Oracle before the defensive follow/commit; post-hoc rejection is not this detector.
        if (sampled[0] != expected) {
            std.debug.print("PostCloseMaskedSelectionRuntimeViolation: independent allowed {d}, sampled {d}\n", .{ expected, sampled[0] });
            return error.PostCloseMaskedSelectionRuntimeViolation;
        }
        _ = try s.commit(a, &.{expected});
        s.pending = expected;
        s.cache_len += 1;
    }
}
pub fn malformedWindows(backend: lanes.backend.Backend, s: *lanes.Stream) !void {
    const allocator = std.testing.allocator;
    const mask = try s.rules.?.next(allocator); defer allocator.free(mask);
    const positions = [_]u64{s.cache_len + 1};
    const wrong_positions = [_]u64{ s.cache_len + 1, s.cache_len + 2 };
    inline for (0..10) |i| {
        var sampled = [_]u32{ std.math.maxInt(u32), std.math.maxInt(u32) };
        var output = [_]lanes.backend.Verified{.{ .sampled = sampled[0..1], .drafts = &.{} }};
        var w = lanes.backend.Window{ .stream = s, .pending = s.pending.?, .held = 0, .tokens = &.{}, .parents = null, .positions = &positions, .masks = mask };
        const expected: anyerror = switch (i) {
            0 => blk: { w.positions = &wrong_positions; break :blk error.PositionMismatch; },
            1 => blk: { w.positions = &.{}; break :blk error.PositionMismatch; },
            2 => blk: { w.held = 1; break :blk error.DraftsUnsupported; },
            3 => blk: { w.tokens = &.{9}; break :blk error.DraftsUnsupported; },
            4 => blk: { w.parents = &.{}; break :blk error.DraftsUnsupported; },
            5 => blk: { w.early = true; break :blk error.DraftsUnsupported; },
            6 => blk: { w.masks = &.{1}; break :blk error.MaskShape; },
            7 => blk: { output[0].sampled = &.{}; break :blk error.MaskShape; },
            8 => blk: { output[0].sampled = &sampled; break :blk error.MaskShape; },
            9 => blk: { output[0].drafts = sampled[1..2]; break :blk error.MaskShape; },
            else => unreachable,
        };
        try std.testing.expectError(expected, backend.verify(&.{w}, &output));
        try std.testing.expectEqual(std.math.maxInt(u32), sampled[0]);
        try std.testing.expectEqual(std.math.maxInt(u32), sampled[1]);
        try std.testing.expectEqual(@as(usize, 1), s.emitted().len);
    }
}
