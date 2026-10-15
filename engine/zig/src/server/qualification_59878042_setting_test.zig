//! Invalid config refuses before allocation/text/engine/clock/listener; no valid serve startup is invoked.
const std = @import("std");
const a = std.testing.allocator;
fn forbiddenNow(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp { @panic("invalid setting reached clock"); }
fn forbiddenRandom(_: ?*anyopaque, _: []u8) void { @panic("invalid setting reached ids"); }
test "qualification_4533b989 invalid serve and direct Config exit before allocation engine text and clock" {
    var vtable = std.testing.io.vtable.*; vtable.now = forbiddenNow; vtable.random = forbiddenRandom;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };
    for ([_][]const u8{ "", "bad", "nan", "inf", "-inf", "-1", "0.01", "300.01" }) |setting| {
        var env = std.process.Environ.Map.init(a); defer env.deinit();
        try env.put("TF_SSE_KEEPALIVE_SECONDS", setting);
        var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
        const status = @import("serve.zig").run(fail.allocator(), io, .{}, .{ .engine = undefined, .text = undefined, .served = "m", .environ = &env });
        try std.testing.expectEqual(@as(u8, 1), status);
        try std.testing.expectEqual(@as(usize, 0), fail.alloc_index);
        try std.testing.expect(!fail.has_induced_failure);
    }
    for ([_]i64{ -1, 1, 999_999_999, 300_000_000_001 }) |ns| {
        var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
        try std.testing.expectError(error.Setting, @import("server.zig").Server.init(fail.allocator(), io, undefined, undefined, .{ .served_name = "m", .model_ids = &.{"m"}, .sse_keepalive_ns = ns }, null));
        try std.testing.expectEqual(@as(usize, 0), fail.alloc_index);
        try std.testing.expect(!fail.has_induced_failure);
    }
}
