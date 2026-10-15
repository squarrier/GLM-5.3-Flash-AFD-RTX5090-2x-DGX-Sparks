//! tool_api_fixes (scratch only, never committed): @Bizuayeu's server items on pr/glm53-flash-cuda 84e9caa1 as CPU fixtures
//! against this tree. Inputs and expected values are his commits' own test cases (b9528625, 557011805, be17a101).
//! Each item is its own test so one failing item does not hide another; every case prints an tool_api_fixes-CASE line.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const tool_parse = @import("tool_parse.zig");
const responses_translate = @import("responses_translate.zig");

fn toolsOf(a: std.mem.Allocator, text: []const u8) ![]const json.Value {
    return (try json.parse(a, text)).ok.array;
}

/// The arguments string of the call `parse` returns at `index` (last when null), or null when it returned no call.
fn argsOf(r: tool_parse.Parsed, index: ?usize) ?[]const u8 {
    const calls = r.calls orelse return null;
    if (calls.len == 0) return null;
    const c = calls[index orelse calls.len - 1];
    const f = c.get("function") orelse return null;
    const args = f.get("arguments") orelse return null;
    return if (args == .string) args.string else null;
}

fn nameOf(r: tool_parse.Parsed) ?[]const u8 {
    const calls = r.calls orelse return null;
    if (calls.len == 0) return null;
    const f = calls[calls.len - 1].get("function") orelse return null;
    const n = f.get("name") orelse return null;
    return if (n == .string) n.string else null;
}

fn report(item: []const u8, case: usize, ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("tool_api_fixes-CASE {s} {d} {s} ", .{ item, case, if (ok) "PASS" else "FAIL" });
    std.debug.print(fmt ++ "\n", args);
}

test "tool_api_fixes item 1 (b9528625): a GLM call whose <arg_key> the model wrote as whitespace parses" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try toolsOf(a, "[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"parameters\":{\"properties\":{\"filePath\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}}]");
    // his "a repaired GLM call parses, in both modes and once the end token closed it": before his fix the first came
    // back as text and the second lost its second argument
    const cases = [_][2][]const u8{
        .{ "<tool_call>read_file filePath</arg_key><arg_value>/srv/afd</arg_value></tool_call>", "{\"filePath\":\"/srv/afd\"}" },
        .{ "<tool_call>read_file<arg_key>filePath</arg_key><arg_value>/a</arg_value>\nlimit</arg_key><arg_value>3</arg_value></tool_call>", "{\"filePath\":\"/a\",\"limit\":3}" },
    };
    var failed: usize = 0;
    var n: usize = 0;
    for (cases) |c| for ([_]?usize{ null, 1 }) |max_calls| {
        n += 1;
        const r = try tool_parse.parse(a, c[0], tools, max_calls);
        const got = argsOf(r, null);
        const ok = got != null and std.mem.eql(u8, got.?, c[1]) and std.mem.eql(u8, r.content, "");
        if (!ok) failed += 1;
        report("1", n, ok, "max_calls={?d} want args {s} content \"\"; got args {?s} content {s}", .{ max_calls, c[1], got, r.content });
    };
    n += 1;
    const open = "<tool_call>read_file filePath</arg_key><arg_value>/x</arg_value>";
    const close = try tool_parse.closeCall(a, open, tools);
    const ok_close = std.mem.eql(u8, close, "</tool_call>");
    if (!ok_close) failed += 1;
    report("1", n, ok_close, "closeCall(open, repaired) want \"</tool_call>\" got \"{s}\"", .{close});
    std.debug.print("tool_api_fixes-ITEM 1 b9528625 {s} ({d}/{d} cases failed)\n", .{ if (failed == 0) "PASS" else "FAIL", failed, n });
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "tool_api_fixes item 2 (557011805): a GLM value the end token left open is closed, as the 2.x line closes it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const lookup = try toolsOf(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]");
    const tools = try toolsOf(a,
        \\[{"type":"function","function":{"name":"read_file","parameters":{"properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}}},
        \\ {"type":"function","function":{"name":"run","parameters":{"properties":{"command":{"type":"string"}}}}},
        \\ {"type":"function","function":{"name":"now","parameters":{"properties":{}}}}]
    );
    var failed: usize = 0;
    var n: usize = 0;
    { // his added row of "a call the end token left open closes when it parses whole"
        n += 1;
        const text = "Trying.<tool_call>lookup<arg_key>n</arg_key><arg_value>7";
        const close = try tool_parse.closeCall(a, text, lookup);
        var ok = std.mem.eql(u8, close, "</arg_value></tool_call>");
        var got_args: ?[]const u8 = null;
        var content: []const u8 = "";
        if (ok) {
            const r = try tool_parse.parse(a, try std.mem.concat(a, u8, &.{ text, close }), lookup, null);
            got_args = argsOf(r, null);
            content = r.content;
            ok = got_args != null and std.mem.eql(u8, got_args.?, "{\"n\":7}") and std.mem.eql(u8, content, "Trying.");
        }
        if (!ok) failed += 1;
        report("2", n, ok, "closeCall(\"{s}\") want \"</arg_value></tool_call>\" got \"{s}\" (args {?s}, content {s})", .{ text, close, got_args, content });
    }
    // 2.x's close_glm_call cases (his "a GLM call the end token left open closes as the 2.x line closes it")
    const closes = [_][3][]const u8{
        .{ "<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>", "</tool_call>", "read_file" },
        .{ "<tool_call>run<arg_key>command</arg_key><arg_value>ls -la", "</arg_value></tool_call>", "run" },
        .{ "Let me look:<tool_call>now", "</tool_call>", "now" },
        .{ "<tool_call>now</tool_call>\n<tool_call>read_file<arg_key>path</arg_key><arg_value>x</arg_value>\n", "</tool_call>", "read_file" },
    };
    for (closes) |c| {
        n += 1;
        const close = try tool_parse.closeCall(a, c[0], tools);
        var ok = std.mem.eql(u8, close, c[1]);
        var name: ?[]const u8 = null;
        if (ok) {
            const r = try tool_parse.parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
            name = nameOf(r);
            ok = name != null and std.mem.eql(u8, name.?, c[2]);
        }
        if (!ok) failed += 1;
        report("2", n, ok, "closeCall want \"{s}\" got \"{s}\" (tool {?s}) for {s}", .{ c[1], close, name, c[0] });
    }
    // what must stay open (he keeps these as text too); a regression guard both ways
    const left_open = [_][]const u8{
        "no call at all",
        "<tool_call>now</tool_call>",
        "<tool_call>launch<arg_key>x</arg_key><arg_value>1",
        "<tool_call>run<arg_key>comm",
        "<tool_call>run<arg_key>command</arg_key>",
        "<tool_call>run command=ls",
        "<tool_call>",
        "<tool_call>now</think>The answer.",
        "<tool_call>run<arg_key>command</arg_key><arg_value>ls</arg_value> and then",
    };
    for (left_open) |text| {
        n += 1;
        const close = try tool_parse.closeCall(a, text, tools);
        const ok = close.len == 0;
        if (!ok) failed += 1;
        report("2", n, ok, "left open: want \"\" got \"{s}\" for {s}", .{ close, text });
    }
    std.debug.print("tool_api_fixes-ITEM 2 557011805 {s} ({d}/{d} cases failed)\n", .{ if (failed == 0) "PASS" else "FAIL", failed, n });
    try std.testing.expectEqual(@as(usize, 0), failed);
}

/// A Responses request with no stored conversation to read (stream_preflight_test.zig's).
const NoHistory = struct {
    pub fn conversation(_: NoHistory, cx: *errors.Cx, _: json.Value) errors.Refused![]json.Value {
        return cx.refuse("no stored responses in this test");
    }
};

test "tool_api_fixes item 3 (be17a101): Responses accepts OpenAI's include values and ignores them, refuses others" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var failed: usize = 0;
    var n: usize = 0;
    // accepted (his responses_include_test.zig's list, sent as a request body's include)
    for ([_][]const u8{ "null", "[]", "[\"reasoning.encrypted_content\"]", "[\"message.output_text.logprobs\", \"web_search_call.results\"]" }) |inc| {
        n += 1;
        var cx: errors.Cx = .{ .a = a };
        const body = (try json.parse(a, try std.mem.concat(a, u8, &.{ "{\"input\":\"x\",\"include\":", inc, "}" }))).ok;
        const refused = if (responses_translate.translate(&cx, body, NoHistory{})) |_| false else |e| e == error.Refused;
        const ok = !refused;
        if (!ok) failed += 1;
        report("3", n, ok, "include {s}: want accepted; got {s} {s}", .{ inc, if (refused) "refused:" else "accepted", cx.message });
    }
    // refused by name (an unknown value, or not a list of strings)
    for ([_][]const u8{ "[\"reasoning.bogus\"]", "\"reasoning.encrypted_content\"", "[1]" }) |inc| {
        n += 1;
        var cx: errors.Cx = .{ .a = a };
        const body = (try json.parse(a, try std.mem.concat(a, u8, &.{ "{\"input\":\"x\",\"include\":", inc, "}" }))).ok;
        const refused = if (responses_translate.translate(&cx, body, NoHistory{})) |_| false else |e| e == error.Refused;
        const ok = refused and std.mem.indexOf(u8, cx.message, "include") != null;
        if (!ok) failed += 1;
        report("3", n, ok, "include {s}: want refused naming include; got {s} {s}", .{ inc, if (refused) "refused:" else "accepted", cx.message });
    }
    std.debug.print("tool_api_fixes-ITEM 3 be17a101 {s} ({d}/{d} cases failed)\n", .{ if (failed == 0) "PASS" else "FAIL", failed, n });
    try std.testing.expectEqual(@as(usize, 0), failed);
}
