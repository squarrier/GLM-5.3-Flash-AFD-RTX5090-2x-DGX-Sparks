//! prose_filter_852a actual routes, inherited memory transport; exact authors/source/licences are retained in NOTICE.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const model = @import("model_text.zig");
const json = @import("json.zig");
const http = @import("http_conn.zig");
const routes = @import("routes.zig");
const log = @import("log.zig");
const Value = json.Value;
const a = std.testing.allocator;
const call = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"a\"}}</tool_call>";
const prose = " \nBefore.\n" ++ call ++ "\nAfter. \n";
const retained = "Before.\n\nAfter.";
const args = "{\"q\":\"a\"}";
const paths = [_][]const u8{ "/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/completions" };
var random_counter: u8 = 0;
fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
    return .{ .nanoseconds = 10 * std.time.ns_per_s };
}
fn random(_: ?*anyopaque, bytes: []u8) void {
    random_counter +%= 1;
    @memset(bytes, random_counter);
}
const Text = struct {
    fn text(t: *Text) model.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } };
    }
    fn encode(_: *anyopaque, g: std.mem.Allocator, bytes: []const u8, _: bool) model.Error![]u32 {
        const ids = try g.alloc(u32, bytes.len);
        for (bytes, ids) |b, *id| id.* = b;
        return ids;
    }
    fn decode(_: *anyopaque, g: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const bytes = try g.alloc(u8, ids.len);
        for (ids, bytes) |id, *b| b.* = @intCast(id);
        return bytes;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, g: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(g, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 256;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{0};
    }
    fn render(_: *anyopaque, g: std.mem.Allocator, messages: Value, _: model.RenderOptions, _: *[]const u8) model.Error![]u8 {
        return json.stringify(g, messages, .{});
    }
    fn template(_: *anyopaque) []const u8 {
        return "<tool_call>";
    }
};
const Engine = struct {
    chunks: []const []const u8,
    prompt: []u32 = &.{},
    submits: usize = 0,
    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 262144, .name = "prose_filter_852a CPU" };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        a.free(e.prompt);
        e.prompt = a.dupe(u32, request.prompt) catch return error.Busy;
        e.submits += 1;
        sink.event(sink.ctx, id, &.{ .prefilled = 3 });
        for (e.chunks) |chunk| {
            const tokens = a.alloc(u32, chunk.len) catch return error.Busy;
            for (chunk, tokens) |b, *t| t.* = b;
            sink.event(sink.ctx, id, &.{ .tokens = tokens });
            @memset(tokens, 0xee);
            a.free(tokens);
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .stats = .{ .prefill_seconds = 0.25, .rounds = 2, .drafted = 4, .accepted = 3 } } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {
        @panic("unexpected cancel in finite fixture");
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};
const Sink = struct {
    wire: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        s.wire.appendSlice(a, bytes) catch return error.Closed;
    }
    fn gone(_: *anyopaque) bool {
        return false;
    }
};
const Result = struct { status: u16, wire: []u8, value: Value };
fn parse(g: std.mem.Allocator, text: []const u8) !Value {
    return (try json.parse(g, text)).ok;
}
fn exchange(srv: *server.Server, g: std.mem.Allocator, path: []const u8, request_body: []const u8, label: []const u8) !Result {
    var sink: Sink = .{};
    defer sink.wire.deinit(a);
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(g, "{d}", .{request_body.len}) }};
    var c: http.Conn = .{ .fd = -1, .peer = "CPU memory", .gpa = g, .buf = try g.dupe(u8, request_body), .end = request_body.len, .method = "POST", .version = "HTTP/1.1", .path = path, .headers = &headers, .quiet_log = true, .test_transport = .{ .ctx = &sink, .write_all = Sink.write, .peer_gone = Sink.gone } };
    routes.dispatch(srv, &c, g);
    try std.testing.expect(!c.broken);
    const wire = try g.dupe(u8, sink.wire.items);
    std.debug.print("prose_filter_852a-WIRE {s} {s} {x}\n", .{ label, path, wire });
    const split = std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.MissingWire;
    const status = try std.fmt.parseInt(u16, wire[9..12], 10);
    const payload = wire[split + 4 ..];
    return .{ .status = status, .wire = payload, .value = if (std.mem.startsWith(u8, payload, "data:") or std.mem.startsWith(u8, payload, "event:")) .null else try parse(g, payload) };
}
fn body(g: std.mem.Allocator, path: []const u8, stream: bool, single: bool, offered: bool) ![]const u8 {
    const msg = std.mem.eql(u8, path, paths[1]);
    const resp = std.mem.eql(u8, path, paths[2]);
    const legacy = std.mem.eql(u8, path, paths[3]);
    const tools = if (!offered or legacy) "" else if (msg) ",\"tools\":[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\"}}]" else if (resp) ",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}]" else ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    const policy = if (single) ",\"parallel_tool_calls\":false" else "";
    const input = if (resp) "\"input\":\"x\",\"max_output_tokens\":2048,\"store\":true" else if (legacy) "\"prompt\":\"x\",\"max_tokens\":2048" else "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2048";
    const usage = if (stream and !resp and !msg) ",\"stream_options\":{\"include_usage\":true}" else "";
    return std.fmt.allocPrint(g, "{{\"model\":\"m\",{s},\"stream\":{}{s}{s}{s}}}", .{ input, stream, tools, policy, usage });
}
fn init(eng: *Engine, text: *Text, io: std.Io) !*server.Server {
    return server.Server.init(a, io, eng.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = eng.chunks.len > 0 and std.mem.startsWith(u8, eng.chunks[0], "<think>"), .use_drafts = false, .keep_warm_s = 0 }, null);
}
fn setupIo(vt: *std.Io.VTable) std.Io {
    vt.* = std.testing.io.vtable.*;
    vt.now = now;
    vt.random = random;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = vt };
    log.init(io, true);
    random_counter = 0;
    return io;
}
fn message(v: Value) Value {
    return v.get("choices").?.array[0].get("message").?;
}
fn checkCall(v: Value, name: []const u8, arguments: []const u8) !void {
    try std.testing.expectEqualStrings(name, v.get("function").?.strField("name").?);
    try std.testing.expectEqualStrings(arguments, v.get("function").?.strField("arguments").?);
    try std.testing.expect(std.mem.startsWith(u8, v.strField("id").?, "call_"));
}
fn checkUsage(v: Value, length: usize) !void {
    try std.testing.expectEqual(@as(i64, @intCast(length)), v.get("usage").?.get("completion_tokens").?.int64().?);
    try std.testing.expectEqualStrings("exact", v.strField("exact_mode").?);
    try std.testing.expectEqualStrings("prose_filter_852a CPU", v.get("tensorfold").?.strField("engine").?);
    try std.testing.expect(v.get("tensorfold").?.get("token_sha").?.string.len == 12);
    try std.testing.expectEqual(@as(i64, 3), v.get("speculative").?.get("accepted").?.int64().?);
}
fn runOne(g: std.mem.Allocator, chunks: []const []const u8, path: []const u8, stream: bool, single: bool, offered: bool, label: []const u8) !Result {
    var vt: std.Io.VTable = undefined;
    const io = setupIo(&vt);
    defer log.init(std.testing.io, true);
    var eng: Engine = .{ .chunks = chunks };
    defer a.free(eng.prompt);
    var text: Text = .{};
    const srv = try init(&eng, &text, io);
    defer srv.deinit();
    random_counter = 0;
    const result = try exchange(srv, g, path, try body(g, path, stream, single, offered), label);
    try std.testing.expectEqual(@as(u16, 200), result.status);
    try std.testing.expectEqual(@as(usize, 1), eng.submits);
    return result;
}

test "prose_filter_852a baseline actual whole chat drops already parsed before after prose" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const v = (try runOne(arena.allocator(), &.{prose}, paths[0], false, false, true, "before-after")).value;
    const m = message(v);
    try checkCall(m.get("tool_calls").?.array[0], "lookup", args);
    try checkUsage(v, prose.len);
    try std.testing.expectEqualStrings("tool_calls", v.get("choices").?.array[0].strField("finish_reason").?);
    const content = m.get("content").?;
    try std.testing.expect(content == .string);
    try std.testing.expectEqualStrings(retained, content.string);
}
test "prose_filter_852a reasoning and first allowed call retain prose without changing tokens extras or arguments" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const raw = "<think>plan</think>Before.\n" ++ call ++ "\n" ++ call ++ "\nAfter.";
    const v = (try runOne(arena.allocator(), &.{raw}, paths[0], false, true, true, "single-reason")).value;
    const m = message(v);
    try std.testing.expectEqual(@as(usize, 1), m.get("tool_calls").?.array.len);
    try checkCall(m.get("tool_calls").?.array[0], "lookup", args);
    try std.testing.expectEqualStrings("plan", m.strField("reasoning_content").?);
    try checkUsage(v, raw.len);
    try std.testing.expect(m.get("content").? == .string);
    try std.testing.expectEqualStrings("Before.\n\n\nAfter.", m.strField("content").?);
}
test "prose_filter_852a whole Messages keeps text plus same tool use id input and stop reason" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const v = (try runOne(arena.allocator(), &.{prose}, paths[1], false, false, true, "messages-prose")).value;
    const blocks = v.get("content").?.array;
    try std.testing.expectEqualStrings("tool_use", v.strField("stop_reason").?);
    try std.testing.expectEqual(@as(usize, 2), blocks.len);
    try std.testing.expectEqualStrings("text", blocks[0].strField("type").?);
    try std.testing.expectEqualStrings(retained, blocks[0].strField("text").?);
    try std.testing.expectEqualStrings("lookup", blocks[1].strField("name").?);
    try std.testing.expectEqualStrings("a", blocks[1].get("input").?.strField("q").?);
    try std.testing.expect(std.mem.startsWith(u8, blocks[1].strField("id").?, "call_"));
}
test "prose_filter_852a whole Responses keeps text and calls in stored GET and previous response history exactly once" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    var vt: std.Io.VTable = undefined;
    const io = setupIo(&vt);
    defer log.init(std.testing.io, true);
    var eng: Engine = .{ .chunks = &.{prose} };
    defer a.free(eng.prompt);
    var text: Text = .{};
    const srv = try init(&eng, &text, io);
    defer srv.deinit();
    random_counter = 0;
    const result = try exchange(srv, g, paths[2], try body(g, paths[2], false, false, true), "responses-prose");
    const v = result.value;
    try std.testing.expectEqual(@as(u16, 200), result.status);
    const output = v.get("output").?.array;
    const rid = v.strField("id").?;
    const stored = (try srv.store.get(io, g, rid)).?;
    try std.testing.expectEqualStrings(try json.stringify(g, v, .{}), try json.stringify(g, stored, .{}));
    var sink: Sink = .{};
    defer sink.wire.deinit(a);
    var conn: http.Conn = .{ .fd = -1, .peer = "CPU GET", .gpa = g, .buf = &.{}, .version = "HTTP/1.1", .quiet_log = true, .test_transport = .{ .ctx = &sink, .write_all = Sink.write, .peer_gone = Sink.gone } };
    @import("responses.zig").get(srv, &conn, g, rid);
    std.debug.print("prose_filter_852a-WIRE stored-get /v1/responses {x}\n", .{sink.wire.items});
    const get_split = std.mem.indexOf(u8, sink.wire.items, "\r\n\r\n").?;
    const got = try parse(g, sink.wire.items[get_split + 4 ..]);
    try std.testing.expectEqualStrings(try json.stringify(g, v, .{}), try json.stringify(g, got, .{}));
    const fc = output[output.len - 1];
    const cid = fc.strField("call_id").?;
    try std.testing.expectEqualStrings("lookup", fc.strField("name").?);
    try std.testing.expectEqualStrings(args, fc.strField("arguments").?);
    try std.testing.expectEqualStrings("completed", fc.strField("status").?);
    eng.chunks = &.{"ok"};
    const follow = try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"previous_response_id\":\"{s}\",\"input\":[{{\"type\":\"function_call_output\",\"call_id\":\"{s}\",\"output\":\"done\"}}],\"max_output_tokens\":2048}}", .{ rid, cid });
    const follow_result = try exchange(srv, g, paths[2], follow, "history-follow");
    try std.testing.expectEqual(@as(u16, 200), follow_result.status);
    const rendered = try Text.decode(undefined, g, eng.prompt);
    std.debug.print("prose_filter_852a-HISTORY {x}\n", .{rendered});
    const history = try parse(g, rendered);
    var text_count: usize = 0;
    var call_count: usize = 0;
    for (history.array) |m| {
        if (m.strField("content")) |content| if (std.mem.eql(u8, content, retained)) {
            text_count += 1;
        };
        if (m.get("tool_calls")) |calls| for (calls.array) |c| {
            try std.testing.expectEqualStrings(cid, c.strField("id").?);
            try std.testing.expectEqualStrings("lookup", c.get("function").?.strField("name").?);
            try std.testing.expectEqualStrings("a", c.get("function").?.get("arguments").?.strField("q").?);
            call_count += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 1), call_count);
    try std.testing.expectEqual(@as(usize, 2), output.len);
    try std.testing.expectEqualStrings(retained, output[0].get("content").?.array[0].strField("text").?);
    try std.testing.expectEqual(@as(usize, 1), text_count);
}
test "prose_filter_852a tool only whitespace only no call empty ordinary legacy and errors remain unchanged" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    for ([_][]const u8{ call, " \n\t" ++ call ++ " \n\t" }) |raw| for (paths[0..3]) |path| {
        const v = (try runOne(g, &.{raw}, path, false, false, true, "tool-only")).value;
        if (std.mem.eql(u8, path, paths[0])) {
            try std.testing.expect(message(v).get("content").? == .null);
            try checkCall(message(v).get("tool_calls").?.array[0], "lookup", args);
        } else if (std.mem.eql(u8, path, paths[1])) {
            try std.testing.expectEqual(@as(usize, 1), v.get("content").?.array.len);
            try std.testing.expectEqualStrings("tool_use", v.get("content").?.array[0].strField("type").?);
        } else {
            try std.testing.expectEqual(@as(usize, 1), v.get("output").?.array.len);
            try std.testing.expectEqualStrings("function_call", v.get("output").?.array[0].strField("type").?);
        }
    };
    for ([_][]const u8{ "", "ordinary text\n" }) |raw| for ([_][]const u8{ paths[0], paths[3] }) |path| {
        const v = (try runOne(g, &.{raw}, path, false, false, false, "no-call")).value;
        const content = if (std.mem.eql(u8, path, paths[0])) message(v).strField("content").? else v.get("choices").?.array[0].strField("text").?;
        try std.testing.expectEqualStrings(raw, content);
        try checkUsage(v, raw.len);
    };
    var vt: std.Io.VTable = undefined;
    const io = setupIo(&vt);
    defer log.init(std.testing.io, true);
    var eng: Engine = .{ .chunks = &.{"unused"} };
    defer a.free(eng.prompt);
    var text: Text = .{};
    const srv = try init(&eng, &text, io);
    defer srv.deinit();
    for (paths) |path| {
        const v = try exchange(srv, g, path, "{\"model\":\"m\",\"parallel_tool_calls\":\"bad\"}", "error");
        try std.testing.expectEqual(@as(u16, 400), v.status);
    }
    try std.testing.expectEqual(@as(usize, 0), eng.submits);
}
test "prose_filter_852a policy and streaming inventory preserves offered unknown malformed unclosed spaced nested and call order" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    const cases = [_][]const u8{
        prose,
        "Before.\n" ++ call ++ "\n" ++ call ++ "\nAfter.",
        "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>",
        "<tool_call>not a call</tool_call>",
        "<tool_call>not a call",
        "<tool_call>{\"name\":\"  LOOKUP  \",\"arguments\":{\"q\":\"a\"}}</tool_call>",
        "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":{\"nested\":[1,true,{\"v\":\"x\"}]}}}</tool_call>",
        "plain no call",
    };
    for (cases, 0..) |raw, i| for ([_]bool{ false, true }) |single| for (paths) |path| for ([_]bool{ false, true }) |stream| {
        const label = try std.fmt.allocPrint(g, "inventory-{d}-{}-{}", .{ i, single, stream });
        _ = try runOne(g, &.{ raw[0 .. raw.len / 2], raw[raw.len / 2 ..] }, path, stream, single, true, label);
    };
    for ([_]bool{ false, true }) |single| {
        const v = (try runOne(g, &.{call ++ call}, paths[0], false, single, true, "order")).value;
        const calls = message(v).get("tool_calls").?.array;
        try std.testing.expectEqual(@as(usize, if (single) 1 else 2), calls.len);
        for (calls) |c| try checkCall(c, "lookup", args);
        if (!single) try std.testing.expect(!std.mem.eql(u8, calls[0].strField("id").?, calls[1].strField("id").?));
    }
}
test "prose_filter_852a stream before after content args finish usage DONE no duplicate" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    const r = try runOne(g, &.{ "Before.\n", "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":", "\"a\"}}</tool_call>", "\nAfter." }, paths[0], true, false, true, "stream-prose");
    var lines = std.mem.splitSequence(u8, r.wire, "\n");
    var content: std.ArrayList(u8) = .empty;
    var arguments: std.ArrayList(u8) = .empty;
    var call_ids: usize = 0;
    var finish: usize = 0;
    var usage: usize = 0;
    var done: usize = 0;
    while (lines.next()) |line| {
        if (std.mem.eql(u8, line, "data: [DONE]")) {
            done += 1;
            continue;
        }
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const v = try parse(g, line[6..]);
        const choices = v.get("choices").?.array;
        if (choices.len == 0) {
            try std.testing.expectEqual(@as(i64, @intCast("Before.\n".len + call.len + "\nAfter.".len)), v.get("usage").?.get("completion_tokens").?.int64().?);
            usage += 1;
            continue;
        }
        const c = choices[0];
        if (c.strField("finish_reason")) |f| {
            try std.testing.expectEqualStrings("tool_calls", f);
            finish += 1;
        }
        const d = c.get("delta").?;
        if (d.strField("content")) |t| try content.appendSlice(g, t);
        if (d.get("tool_calls")) |calls| for (calls.array) |tc| {
            if (tc.strField("id") != null) call_ids += 1;
            if (tc.get("function")) |fun| if (fun.strField("arguments")) |t| try arguments.appendSlice(g, t);
        };
    }
    try std.testing.expectEqualStrings(retained, content.items);
    try std.testing.expectEqualStrings(args, arguments.items);
    try std.testing.expectEqual(@as(usize, 1), call_ids);
    try std.testing.expectEqual(@as(usize, 1), finish);
    try std.testing.expectEqual(@as(usize, 1), usage);
    try std.testing.expectEqual(@as(usize, 1), done);
}
