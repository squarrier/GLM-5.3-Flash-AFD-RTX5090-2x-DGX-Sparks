//! Direct native streamed-prefix checks; final diagnostics must never observe probes.
const std = @import("std");
const json = @import("json.zig");
const tools_parse = @import("tool_parse.zig");
const stream = @import("tool_stream.zig");
const rejection = @import("tool_rejection.zig");
const Counter = struct {
    count: usize = 0,
    fn record(ctx: *anyopaque, _: rejection.Name) void {
        const c: *Counter = @ptrCast(@alignCast(ctx));
        c.count += 1;
    }
};
test "qualification_214a6e4c every UTF8 delimiter split and repeated prefix remains final-only" {
    const texts = [_][]const u8{
        "<tool_call>missingé🙂<arg_key>q</arg_key><arg_value>ARGUMENT_SECRET</arg_value></tool_call>",
        "<tool_call><function=Lookup><parameter=q>é🙂</parameter></function></tool_call><tool_call>missing</tool_call>",
        "Quoted <tool_call> then <tool_call>missing</tool_call>",
        "Quoted \"<tool_call>missing</tool_call>\"",
    };
    for (texts, 0..) |text, ti| for (0..text.len + 1) |cut| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = (try json.parse(a, "[{\"function\":{\"name\":\"Lookup\"}}]")).ok.array;
        var s = try stream.Streamer.init(a, tools);
        var out: std.ArrayList(json.Value) = .empty;
        var c: Counter = .{};
        try s.feed(text[0..cut], &out);
        const prefix_count = out.items.len;
        try s.feed(text[0..cut], &out);
        try std.testing.expectEqual(prefix_count, out.items.len);
        _ = try tools_parse.closeCall(a, text[0..cut], tools);
        _ = try tools_parse.parse(a, text[0..cut], tools, null);
        try std.testing.expectEqual(@as(usize, 0), c.count);
        try s.feed(text, &out);
        const final_count = out.items.len;
        try s.feed(text, &out);
        try std.testing.expectEqual(final_count, out.items.len);
        _ = try tools_parse.parseObserved(a, text, tools, null, .{ .ctx = &c, .record = Counter.record });
        try std.testing.expectEqual(@as(usize, if (ti == 2) 0 else 1), c.count);
        var names: usize = 0;
        for (out.items) |delta| for (delta.get("tool_calls").?.array) |call| {
            if (call.get("function").?.has("name")) names += 1;
        };
        try std.testing.expectEqual(@as(usize, if (ti == 1) 1 else 0), names);
    };
}
