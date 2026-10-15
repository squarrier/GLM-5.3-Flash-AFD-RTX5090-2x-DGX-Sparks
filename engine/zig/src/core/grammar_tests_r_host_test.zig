//! grammar_tests_r actual private LaneHost guard/driver witness; fake CPU head only.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const a = std.testing.allocator;
const Driver = struct {
    loaded: bool = true,
    warmed: bool = false,
    runs: usize = 0,
    absorbed: usize = 0,
    drafted: usize = 0,
    fn run(ptr: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) !bool {
        const d: *Driver = @ptrCast(@alignCast(ptr));
        if (!d.loaded or !d.warmed) return error.HeadNotWarmed;
        d.runs += 1;
        d.absorbed += 1;
        d.drafted += 1;
        s.drafted += 1;
        _ = try s.commit(a, &.{ 7, 8 });
        hooks.committed(hooks.ctx);
        return false;
    }
};
const Sink = struct {
    tokens: usize = 0,
    finished: usize = 0,
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const s: *Sink = @ptrCast(@alignCast(ptr));
        switch (e.*) {
            .tokens => |ts| s.tokens += ts.len,
            .finished => s.finished += 1,
            else => {},
        }
    }
};
fn noop(_: *anyopaque) void {}
fn noYield(_: *anyopaque) bool { return false; }
pub fn run(comptime Host: type, comptime Job: type, comptime fits: anytype, comptime runLone: anytype) !void {
    var cfg = try lanes.Config.init(a, .{}, 1, 0);
    defer cfg.deinit(a);
    var target = lanes.fake.Fake{ .gpa = a };
    defer target.deinit();
    var clock = lanes.fake.FixedClock{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = Host.init(a, std.testing.io, &core, .{ .lanes = 1 });
    defer host.stop();
    var driver = Driver{};
    driver.warmed = true;
    var warm = try lanes.Stream.init(a, .{ .id = "cpu-head-warmup", .prompt = &.{ 3, 4 }, .max_new = 2, .drafts = true });
    defer warm.deinit(a);
    _ = try Driver.run(&driver, &warm, .{ .ctx = &host, .committed = noop, .yield = noYield });
    try std.testing.expectEqual(@as(usize, 1), driver.runs);
    const warm_runs = driver.runs;
    const warm_absorb = driver.absorbed;
    const warm_drafts = driver.drafted;
    host.lone = .{ .ctx = &driver, .run = Driver.run };
    var sink = Sink{};
    const structured = api.Request{ .prompt = &.{ 3, 4 }, .max_tokens = 2, .drafts = true, .loop_guard = false, .structure = .{ .kind = .json, .text = "" } };
    var job = Job{ .host = &host, .id = 1, .request = &structured, .sink = .{ .ctx = &sink, .event = Sink.event } };
    job.stream = try lanes.Stream.init(a, .{ .id = "internal-structured-guard", .prompt = structured.prompt, .max_new = 2, .drafts = true });
    defer job.stream.deinit(a);
    try host.admitted.append(a, &job);
    defer host.admitted.clearRetainingCapacity();
    try std.testing.expect(structured.structure != null and structured.drafts);
    try std.testing.expectEqual(@as(usize, 0), core.activeCount());
    try std.testing.expectEqual(@as(usize, 1), host.admitted.items.len);
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len + host.cancels.items.len);
    // Execute the actual guard with ALL other eligibility predicates satisfied.
    if (fits(&host, &job)) _ = try host.lone.?.run(host.lone.?.ctx, &job.stream, .{ .ctx = &host, .committed = noop, .yield = noYield });
    if (driver.runs != warm_runs or driver.absorbed != warm_absorb or driver.drafted != warm_drafts) {
        std.debug.print("StructuredLoneGuardRuntimeViolation: actual lone guard dispatched structured CPU driver\n", .{});
        return error.StructuredLoneGuardRuntimeViolation;
    }
    try std.testing.expectEqual(@as(usize, 0), sink.tokens);
    host.admitted.clearRetainingCapacity();
    const plain = api.Request{ .prompt = &.{ 3, 4 }, .max_tokens = 2, .drafts = true, .loop_guard = false };
    const next = try a.create(Job);
    next.* = .{ .host = &host, .id = 2, .request = &plain, .sink = .{ .ctx = &sink, .event = Sink.event }, .started = true };
    next.proposer = try lanes.SuffixLookup.init(a, .{ .min_match = 4 });
    next.stream = try lanes.Stream.init(a, .{ .id = "plain-reuse-warmed-head", .prompt = plain.prompt, .max_new = 2, .drafts = true });
    try host.admitted.append(a, next);
    if (!fits(&host, next)) return error.PlainLoneEligibilityLost;
    try std.testing.expect(runLone(&host, next, std.Io.Clock.awake.now(std.testing.io).toNanoseconds()));
    try std.testing.expectEqual(warm_runs + 1, driver.runs);
    try std.testing.expectEqual(warm_absorb + 1, driver.absorbed);
    try std.testing.expectEqual(warm_drafts + 1, driver.drafted);
    try std.testing.expectEqual(@as(usize, 2), sink.tokens);
    try std.testing.expectEqual(@as(usize, 1), sink.finished);
    try std.testing.expect(host.lone_job == null and host.admitted.items.len == 0);
    std.debug.print("grammar_tests_r actual lone: eligible loaded/warmed; structured runs/absorb/draft=0; plain runs/absorb/draft=1; guard unchanged\n", .{});
}
