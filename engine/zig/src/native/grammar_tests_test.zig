//! grammar_tests constructor red gate: execute the real Lanes vtable, never a hook registry.
const std = @import("std");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const pair = @import("glm_cuda_pair.zig");
const gl = @import("glm_cuda_lanes.zig");

pub fn tp(comptime Host: type, comptime wrap: fn (*Host) lanes.backend.Backend) !void {
    const r = try glm.serve_rec.Rank.init(std.testing.allocator, null, null);
    defer r.deinit(std.testing.allocator);
    var f = r.recording();
    var s: glm.serve.Serve = .{ .e = r.e, .f = &f };
    var p: glm.ranks.Pair = .{ .gpa = std.testing.allocator, .io = std.testing.io, .rank = 0 };
    var armed: pair.Armed(glm.serve.Serve) = .{ .steps = &s, .pair = &p };
    var backend: gl.Lanes(@TypeOf(armed)) = .{ .steps = &armed, .capacity = 2560, .reach = 2560 };
    var h: Host = .{ .gpa = std.testing.allocator, .gpu = null, .family = &backend, .release = noRelease, .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    const wrapped = wrap(&h);
    @import("grammar_tests_data").witness("MissingWrapperRuntimeViolation.N1-wrapper", 1, @intFromBool(wrapped.vtable.first_masked != null));
    try std.testing.expect(wrapped.vtable.first_masked != null);
}
fn noRelease(_: *anyopaque) void {}
