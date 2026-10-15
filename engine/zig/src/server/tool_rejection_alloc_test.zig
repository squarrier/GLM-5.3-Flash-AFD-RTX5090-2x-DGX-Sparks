//! qualification_214a6e4c allocation identity: observation is stack-only and cannot add parser allocations.
const std = @import("std");
const parser = @import("tool_parse.zig");
const json = @import("json.zig");
const rejection = @import("tool_rejection.zig");
const Ignore = struct {
    fn record(_: *anyopaque, _: rejection.Name) void {}
};
test "qualification_214a6e4c off and enabled diagnostics add zero parser allocations" {
    var tool_arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer tool_arena.deinit();
    const tools = (try json.parse(tool_arena.allocator(), "[{\"function\":{\"name\":\"Lookup\"}}]")).ok.array;
    const text = "<tool_call>{\"name\":\"missing\",\"arguments\":{\"q\":\"ARGUMENT_SECRET\"}}</tool_call>";
    var totals: [3][2]usize = undefined;
    var dummy: u8 = 0;
    for (0..3) |i| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        var counter: std.testing.FailingAllocator = .init(arena.allocator(), .{});
        const a = counter.allocator();
        if (i == 0) _ = try parser.parse(a, text, tools, null) else _ = try parser.parseObserved(a, text, tools, null, if (i == 1) null else .{ .ctx = &dummy, .record = Ignore.record });
        totals[i] = .{ counter.allocations, counter.allocated_bytes };
    }
    try std.testing.expectEqualSlices(usize, &totals[0], &totals[1]);
    try std.testing.expectEqualSlices(usize, &totals[0], &totals[2]);
}
