//! CPU fake context entry point; never opens a CUDA driver or device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_cuda_lanes.zig");
const pair = @import("glm_cuda_pair.zig");
const data = @import("grammar_tests_data");
fn notReady(_: cuda.abi.Context) callconv(.c) cuda.abi.Result { return cuda.abi.error_not_ready; }
pub fn tp(comptime Host: type, comptime Gpu: type, comptime wrap: fn (*Host) lanes.backend.Backend) !void {
    const a = std.testing.allocator;
    const r = try glm.serve_rec.Rank.init(a, 0, null); defer r.deinit(a);
    var f = r.recording();
    var serve = glm.serve.Serve{ .e = r.e, .f = &f };
    var world = glm.ranks.Pair{ .gpa = a, .io = std.testing.io, .rank = 0 };
    var armed = pair.Armed(glm.serve.Serve){ .steps = &serve, .pair = &world };
    var adapter = gl.Lanes(@TypeOf(armed)){ .steps = &armed, .capacity = 2560, .reach = 2560 };
    var gpu: Gpu = undefined;
    gpu.driver.api.cuCtxSetCurrent = notReady;
    gpu.ctx = .{ .d = &gpu.driver, .device = 0, .handle = null };
    var h = Host{ .gpa = a, .gpu = &gpu, .family = &adapter, .release = noop, .inner = adapter.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    const b = wrap(&h);
    var rules = data.Fake{}; rules.init(r.m.vocab);
    var s = try lanes.Stream.init(a, .{ .id = "bind-failure", .prompt = &.{ 3, 4 }, .max_new = 2, .rules = &rules.rules });
    defer s.deinit(a);
    const result = b.prefill(&s);
    const not_ready = if (result) |_| false else |err| err == error.NotReady;
    data.witness("BindBeforeLaunchRuntimeViolation.N1-bind-continue", 1, @intFromBool(not_ready));
    try std.testing.expectError(error.NotReady, result);
    try std.testing.expectError(error.NotReady, b.vtable.first_masked.?(b.ptr, &s, 2, &.{}));
    try std.testing.expectError(error.NotReady, b.verify(&.{.{ .stream = &s, .pending = 7, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{3} }}, &.{}));
    b.release(&s);
    try std.testing.expectEqual(@as(usize, 0), r.t.events.items.len);
    try std.testing.expect(adapter.owner == null and !serve.open);
    h.gpu = null;
    var plain = try lanes.Stream.init(a, .{ .id = "plain-after-bind-failure", .prompt = &.{ 3, 4 }, .max_new = 1, .drafts = false });
    defer plain.deinit(a);
    try b.prefill(&plain);
    try std.testing.expectEqual(@as(u64, 0), try b.first(&plain, 2));
    b.release(&plain);
    try std.testing.expect(adapter.owner == null and !serve.open);
}
fn noop(_: *anyopaque) void {}
