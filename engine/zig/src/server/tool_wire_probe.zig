//! qualification_214a6e4c wire probe: real whole/stream/translator routes over POSIX pipes, never sockets.
//! Fixture engine synchronously delivers borrowed byte-token chunks then finished.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const responses = @import("responses.zig");
const anthropic = @import("anthropic.zig");
const server = @import("server.zig");
const serve = @import("serve.zig");
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

const Text = struct {
    fn text(t: *Text) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |b, *id| id.* = b;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *b| b.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, s: []const u8) ?u32 {
        return if (std.mem.eql(u8, s, "<tool_call>")) 1000 else null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 1024;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{0};
    }
    fn render(_: *anyopaque, a: Allocator, messages: Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return json.stringify(a, messages, .{});
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "<tool_call>";
    }
};

const Eng = struct {
    a: Allocator,
    chunks: []Value,
    reason: api.Reason,
    submitted: usize = 0,
    fn engine(e: *Eng) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 262144, .name = "CPU-fixture", .call_gates = false };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Eng = @ptrCast(@alignCast(ctx));
        e.submitted += 1;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.chunks) |chunk| {
            const length = if (chunk == .string) chunk.string.len else chunk.array.len;
            const tokens = e.a.alloc(u32, length) catch return error.Busy;
            if (chunk == .string) {
                for (chunk.string, tokens) |b, *t| t.* = b;
            } else {
                for (chunk.array, tokens) |b, *t| t.* = @intCast(b.int64().?);
            }
            sink.event(sink.ctx, id, &.{ .tokens = tokens });
            @memset(tokens, 0xee); // the server must have copied the borrowed callback slice
            e.a.free(tokens);
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = e.reason, .stats = .{ .prefill_seconds = 0.25 } } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn print(bytes: []const u8) void {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = std.posix.system.write(2, bytes[at..].ptr, bytes.len - at);
        if (std.posix.errno(n) != .SUCCESS or n <= 0) return;
        at += @intCast(n);
    }
}

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const a = init.arena.allocator();
    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, argv[1], a, .limited(1 << 20));
    const fixture = (try json.parse(a, raw)).ok;
    var text: Text = .{};
    var eng: Eng = .{ .a = init.gpa, .chunks = fixture.get("chunks").?.array, .reason = if (std.mem.eql(u8, fixture.strField("reason").?, "length")) .length else .stop };
    @import("log.zig").init(init.io, false);
    if (std.mem.eql(u8, fixture.strField("route").?, "invalid")) {
        const rc = serve.run(init.gpa, init.io, .{ .model = "CPU", .host = "127.0.0.1", .port = 0, .keep_warm = 0 }, .{ .engine = eng.engine(), .text = text.text(), .served = "m", .environ = init.environ_map });
        print("INVALID_RC ");
        print(try std.fmt.allocPrint(a, "{d}\n", .{rc}));
        return if (rc == 2) 0 else 1;
    }
    var config: server.Config = .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .default_max_tokens = 4096, .keep_warm_s = 0 };
    // The frozen-base probe omits this field in its saved runner copy only.
    config.tool_rejection_log = @import("tool_rejection.zig").mode(init.environ_map.get("TF_TOOL_REJECTION_LOG")) catch return 2;
    const srv = try server.Server.init(init.gpa, init.io, eng.engine(), text.text(), config, null);
    defer srv.deinit();
    // stdout is a parent-owned pipe: partial reply bytes survive an inherited parser panic.
    var conn = try Conn.init(a, 1, "CPU-pipe");
    defer conn.deinit();
    conn.version = "HTTP/1.1";
    conn.method = "POST";
    conn.quiet_log = true;
    const body = try json.stringify(a, fixture.get("body").?, .{});
    @memcpy(conn.buf[0..body.len], body);
    conn.end = body.len;
    const headers = try a.alloc(@import("http_conn.zig").Header, 1);
    headers[0] = .{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) };
    conn.headers = headers;
    const route = fixture.strField("route").?;
    conn.path = route;
    if (std.mem.eql(u8, route, "/v1/responses")) responses.post(srv, &conn, a) else if (std.mem.eql(u8, route, "/v1/messages")) anthropic.post(srv, &conn, a) else openai.post(srv, &conn, a, std.mem.eql(u8, route, "/v1/chat/completions"));
    print("SUBMITTED ");
    print(try std.fmt.allocPrint(a, "{d}\n", .{eng.submitted}));
    return 0;
}
