//! qualification_214a6e4c: finalized native parser diagnostics; policy and wire remain control.
const std = @import("std");
const json = @import("json.zig");
const parser = @import("tool_parse.zig");
const rejection = @import("tool_rejection.zig");

test {
    _ = @import("tool_rejection_stream_test.zig");
    _ = @import("tool_rejection_alloc_test.zig");
}

const Capture = struct {
    names: [16]rejection.Name = undefined,
    len: usize = 0,
    fn record(ctx: *anyopaque, name: rejection.Name) void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.names[c.len] = name;
        c.len += 1;
    }
    fn sink(c: *Capture) rejection.Sink {
        return .{ .ctx = c, .record = record };
    }
};

const Case = struct { text: []const u8, count: usize, single: ?usize = null, tools: bool = true };
test "qualification_214a6e4c finalized parsed-but-unoffered decisions only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"Lookup\"}}]")).ok.array;
    const cases = [_]Case{
        .{ .text = "<tool_call>LOOKUP</tool_call>", .count = 0 },
        .{ .text = "<tool_call>functions__Lookup</tool_call>", .count = 1 },
        .{ .text = "<tool_call>missing</tool_call>", .count = 1 },
        .{ .text = "<tool_call>{bad}</tool_call>", .count = 0 },
        .{ .text = "<tool_call>{\"name\":\"missing\",\"arguments\":[]}</tool_call>", .count = 0 },
        .{ .text = "<tool_call>missing</tool_call>", .count = 0, .tools = false },
        .{ .text = "<tool_call>missing", .count = 0 },
        .{ .text = "{\"name\":\"missing\",\"arguments\":{}}", .count = 0 },
        .{ .text = "<tool_call>missing</tool_call><tool_call>Lookup</tool_call>", .count = 1 },
        .{ .text = "<tool_call>Lookup</tool_call><tool_call>missing</tool_call>", .count = 0, .single = 1 },
        .{ .text = "<tool_call>missing</tool_call><tool_call>Lookup</tool_call>", .count = 1, .single = 1 },
        .{ .text = "<tool_call>{\"name\":\"missing\",\"arguments\":{\"v\":[{\"n\":NaN}]}}</tool_call>", .count = 0, .single = 1 },
        .{ .text = "<tool_call>{\"name\":\"missing\",\"arguments\":{\"v\":[{\"n\":3}]}}</tool_call>", .count = 1, .single = 1 },
        .{ .text = "Mention <tool_call> then <tool_call>missing</tool_call>", .count = 0 },
        .{ .text = "Quoted \"<tool_call>missing</tool_call>\"", .count = 1 },
    };
    for (cases) |c| {
        var capture: Capture = .{};
        const active = if (c.tools) tools else &.{};
        const off = try parser.parse(a, c.text, active, c.single);
        const on = try parser.parseObserved(a, c.text, active, c.single, capture.sink());
        try std.testing.expectEqualStrings(off.content, on.content);
        try std.testing.expectEqual(off.calls != null, on.calls != null);
        if (on.calls) |calls| {
            try std.testing.expectEqualStrings("Lookup", calls[0].get("function").?.strField("name").?);
            try std.testing.expectEqualStrings(off.calls.?[0].get("function").?.strField("arguments").?, calls[0].get("function").?.strField("arguments").?);
        }
        try std.testing.expectEqual(c.count, capture.len);
    }
}

test "qualification_214a6e4c owned bounded escaped UTF8 name no argument lifetime" {
    var capture: Capture = .{};
    {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const tools = (try json.parse(a, "[{\"function\":{\"name\":\"Lookup\"}}]")).ok.array;
        _ = try parser.parseObserved(a, "<tool_call>{\"name\":\"unknown\\\"\\n\\u001b\",\"arguments\":{\"secret\":\"ARGUMENT_SECRET\"}}</tool_call>", tools, null, capture.sink());
    }
    try std.testing.expectEqual(@as(usize, 1), capture.len);
    try std.testing.expectEqualStrings("\"unknown\\\"\\n\\u001b\"", capture.names[0].slice());
    const samples = [_][]const u8{ "\"\\\n\r\t\x00", "é🙂日本語", "\xff", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa🙂" };
    for (samples) |s| {
        const name = rejection.Name.init(s);
        try std.testing.expect(name.slice().len <= 128);
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const v = (try json.parse(arena.allocator(), name.slice())).ok;
        try std.testing.expect(v == .string);
        try std.testing.expect(std.unicode.utf8ValidateSlice(v.string));
    }
}

test "qualification_214a6e4c strict default-off modes" {
    try std.testing.expect(!try rejection.mode(null));
    try std.testing.expect(!try rejection.mode("0"));
    try std.testing.expect(try rejection.mode("1"));
    for ([_][]const u8{ "", "2", "01", "true", " 1", "1\n", "-1", "NaN" }) |s| try std.testing.expectError(error.InvalidToolRejectionLog, rejection.mode(s));
}
