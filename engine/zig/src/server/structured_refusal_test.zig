//! qualification_3b492693 test-only refusal/coverage proof; GPT-6.1 Sol via Hermes; native TensorFold by Ash Hart.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const grammar = @import("grammar.zig");
const openai = @import("openai.zig");
const model_text = @import("model_text.zig");
const Conn = @import("http_conn.zig").Conn;
const Server = @import("server.zig").Server;
const Config = @import("server.zig").Config;
const lanes = @import("lanes");
const gl = @import("glm_lanes");
const a = std.testing.allocator;

pub const ByteText = struct {
    pub fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, g: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try g.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, g: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try g.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 { return null; }
    fn tokenString(_: *anyopaque, g: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 { return std.fmt.allocPrint(g, "{d}", .{id}); }
    fn vocabSize(_: *anyopaque) u32 { return 256; }
    fn eosIds(_: *anyopaque) []const u32 { return &.{}; }
    fn render(_: *anyopaque, g: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 { return g.dupe(u8, "hi"); }
    fn templateSource(_: *anyopaque) []const u8 { return ""; }
};

pub const Capture = struct {
    submits: usize = 0,
    admits: usize = 0,
    slots: usize = 0,
    forwards: usize = 0,
    tokens: usize = 0,
    callbacks: usize = 0,
    drafts: bool = false,
    prompt: [2]u32 = .{0, 0},
    max_tokens: u32 = 0,
    pub fn engine(c: *@This()) api.Engine {
        return .{ .ctx = c, .vtable = &.{ .info = info, .submit = submit, .admit = admit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info { return .{ .context_window = 32 }; }
    fn admit(ctx: *anyopaque, _: bool) bool { const c: *Capture = @ptrCast(@alignCast(ctx)); c.admits += 1; return true; }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.submits += 1; c.slots += 1; c.forwards += 1;
        c.drafts = r.drafts; c.max_tokens = r.max_tokens;
        std.debug.assert(r.structure == null and r.call == null and r.sampling == null);
        std.debug.assert(r.prompt.len == 2 and r.eos.len == 0 and r.chunks.len == 0 and r.think_budget == 0);
        @memcpy(&c.prompt, r.prompt);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 }); c.callbacks += 1;
        sink.event(sink.ctx, id, &.{ .tokens = &.{'o', 'k'} }); c.callbacks += 1; c.tokens += 2;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } }); c.callbacks += 1;
        c.slots -= 1;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{}; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
};

const Reply = struct {
    status: u16 = 0,
    payload: json.Value = .null,
    opened: usize = 0,
    events: usize = 0,
    fn out(r: *@This()) openai.Out { return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = reply } }; }
    fn open(ctx: *anyopaque) error{Closed}!void { const r: *Reply = @ptrCast(@alignCast(ctx)); r.opened += 1; }
    fn event(ctx: *anyopaque, _: ?json.Value) error{Closed}!void { const r: *Reply = @ptrCast(@alignCast(ctx)); r.events += 1; }
    fn reply(ctx: *anyopaque, code: u16, value: json.Value) void { const r: *Reply = @ptrCast(@alignCast(ctx)); r.status = code; r.payload = value; }
};

const unsupported = "structured output (response_format and the guided_* fields) is not supported by this engine yet";
const Case = struct { field: []const u8, kind: ?grammar.Kind = null, message: []const u8 = unsupported };
const valid = [_]Case{
    .{ .field = "\"response_format\":{\"type\":\"json_object\"}", .kind = .json },
    .{ .field = "\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"reply\",\"schema\":{\"type\":\"object\"}}}", .kind = .json_schema },
    .{ .field = "\"guided_regex\":\"[a-z]+\"", .kind = .regex },
    .{ .field = "\"guided_choice\":[\"yes\",\"no\"]", .kind = .choice },
    .{ .field = "\"guided_grammar\":\"root ::= \\\"yes\\\"\"", .kind = .grammar },
};
const malformed = [_]Case{
    .{ .field = "\"response_format\":[]", .message = "response_format must be an object such as {\"type\": \"json_object\"}" },
    .{ .field = "\"response_format\":{\"type\":\"json_schema\"}", .message = "response_format json_schema needs json_schema.schema (a JSON schema object)" },
    .{ .field = "\"guided_json\":false", .message = "guided_json must be a JSON schema object" },
    .{ .field = "\"guided_json\":\"{\"", .message = "guided_json is not valid JSON: Expecting property name enclosed in double quotes" },
    .{ .field = "\"guided_regex\":\"\"", .message = "guided_regex must be a non-empty string" },
    .{ .field = "\"guided_choice\":[]", .message = "guided_choice must be a non-empty list of non-empty strings" },
    .{ .field = "\"guided_grammar\":2", .message = "guided_grammar must be a non-empty string" },
    .{ .field = "\"structured_outputs\":[]", .message = "structured_outputs must be an object such as {\"json\": {...}}" },
    .{ .field = "\"structured_outputs\":{\"bad\":true}", .message = "structured_outputs bad is not supported: use json, json_object, regex, choice or grammar" },
};
const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}],\"tool_choice\":\"required\"";

fn transact(srv: *Server, c: *Capture, case: Case, chat: bool, stream: bool, required: bool, drafts: bool) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const g = arena.allocator();
    const prompt = if (chat) "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]" else "\"prompt\":\"hi\"";
    const raw = (try json.parse(g, try std.fmt.allocPrint(g, "{{{s},\"model\":\"served-alias\",\"max_tokens\":2,\"temperature\":0,\"stream\":{s},{s}{s}}}", .{prompt, if (stream) "true" else "false", case.field, if (required) tools else ""}))).ok;
    if (case.kind) |k| {
        var cx: errors.Cx = .{ .a = g };
        const s = (try grammar.spec(&cx, raw)).?;
        try std.testing.expectEqual(k, s.kind);
    }
    // Negative fd is ignored by poll; no socket is created or read by this fixture.
    var conn = try Conn.init(a, -1, "cpu-only");
    defer conn.deinit();
    const before = c.*;
    var r: Reply = .{};
    openai.run(srv, g, r.out(), .{ .conn = &conn }, chat, raw);
    try std.testing.expectEqual(@as(u16, 400), r.status);
    var expected_cx: errors.Cx = .{ .a = g };
    const expected = if (required) try std.fmt.allocPrint(g, "{s} cannot be combined with tool_choice \"required\" or a named function: send one", .{(try grammar.spec(&expected_cx, raw)).?.field}) else case.message;
    try std.testing.expectEqualStrings(expected, r.payload.get("error").?.get("message").?.string);
    try std.testing.expectEqual(before.submits, c.submits);
    try std.testing.expectEqual(before.admits, c.admits);
    try std.testing.expectEqual(before.forwards, c.forwards);
    try std.testing.expectEqual(before.tokens, c.tokens);
    try std.testing.expectEqual(before.callbacks, c.callbacks);
    try std.testing.expectEqual(@as(usize, 0), c.slots);
    try std.testing.expectEqual(@as(usize, 0), r.opened);
    try std.testing.expectEqual(@as(usize, 0), r.events);
    try std.testing.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
    const plain = (try json.parse(g, try std.fmt.allocPrint(g, "{{{s},\"max_tokens\":2,\"temperature\":0}}", .{prompt}))).ok;
    var next: Reply = .{};
    openai.run(srv, g, next.out(), .{ .conn = &conn }, chat, plain);
    try std.testing.expectEqual(@as(u16, 200), next.status);
    try std.testing.expectEqualStrings("ok", if (chat) next.payload.get("choices").?.array[0].get("message").?.get("content").?.string else next.payload.get("choices").?.array[0].get("text").?.string);
    try std.testing.expectEqual(before.submits + 1, c.submits);
    // Native whole replies submit directly; only streaming replies call chat.admit.
    try std.testing.expectEqual(before.admits, c.admits);
    try std.testing.expectEqual(before.forwards + 1, c.forwards);
    try std.testing.expectEqual(before.tokens + 2, c.tokens);
    try std.testing.expectEqual(before.callbacks + 3, c.callbacks);
    try std.testing.expectEqualSlices(u32, &.{'h', 'i'}, &c.prompt);
    try std.testing.expectEqual(@as(u32, 2), c.max_tokens);
    try std.testing.expectEqual(drafts, c.drafts);
    try std.testing.expectEqual(@as(usize, 0), c.slots);
}

test "qualification_fb84ff48 actual native refusal transactions and next plain request" {
    var kinds: [5]usize = @splat(0);
    var count: usize = 0;
    for ([_]bool{false, true}) |drafts| {
        var text: ByteText = .{};
        var c: Capture = .{};
        const srv = try Server.init(a, std.testing.io, c.engine(), text.text(), .{ .served_name = "served-alias", .model_ids = &.{"served-alias"}, .enable_thinking = false, .use_drafts = drafts, .keep_warm_s = 0 }, null);
        defer srv.deinit();
        try std.testing.expect(!srv.info.structures);
        for ([_]bool{false, true}) |chat| for ([_]bool{false, true}) |stream| {
            for (valid) |case| {
                try transact(srv, &c, case, chat, stream, false, drafts); count += 1;
                kinds[@intFromEnum(case.kind.?)] += 1;
                try transact(srv, &c, case, chat, stream, true, drafts); count += 1;
            }
            for (malformed) |case| { try transact(srv, &c, case, chat, stream, false, drafts); count += 1; }
        };
    }
    for (kinds) |n| try std.testing.expect(n > 0);
    try std.testing.expectEqual(@as(usize, 152), count);
    std.debug.print("qualification_fb84ff48 refusal transactions={d}; next_plain={d}; kinds={any}; no sockets/device\n", .{count, count, kinds});
}

// These declarations are CPU constructor doubles, not the native GPU Host/Armed/Bound types.
const Descriptor = struct {
    structures: bool = false,
    hooks: [22]bool = @splat(false),
    wrapper: bool = false,
    model_dir: []const u8 = "",
    alias: []const u8 = "served-alias",
};
fn audit(d: Descriptor) !void {
    if (!d.structures) return;
    if (!d.wrapper) return error.MissingWrapper;
    for (d.hooks) |hook| if (!hook) return error.MissingHook;
    if (!std.mem.eql(u8, d.model_dir, "/fixture/model")) return error.WrongModelDir;
    if (std.mem.eql(u8, d.model_dir, d.alias)) return error.AliasIsDirectory;
}
const mock_complete: Descriptor = .{ .structures = true, .hooks = @splat(true), .wrapper = true, .model_dir = "/fixture/model" };

test "qualification_fb84ff48 N0 mock audit rejects every missing hook and bad directory" {
    try audit(mock_complete);
    var n: usize = 0;
    for (0..22) |h| {
        var d = mock_complete; d.hooks[h] = false;
        try std.testing.expectError(error.MissingHook, audit(d)); n += 1;
    }
    var d = mock_complete; d.wrapper = false;
    try std.testing.expectError(error.MissingWrapper, audit(d)); n += 1;
    d.hooks[4] = false;
    try std.testing.expectError(error.MissingWrapper, audit(d)); n += 1;
    for ([_][]const u8{"", "/wrong/model", "served-alias"}) |dir| {
        d = mock_complete; d.model_dir = dir;
        try std.testing.expectError(error.WrongModelDir, audit(d)); n += 1;
    }
    const product: Descriptor = .{ .structures = (api.Info{}).structures };
    try audit(product);
    try std.testing.expect(!product.structures and product.model_dir.len == 0);
    if (@hasField(Config, "model_dir")) try std.testing.expectEqualStrings("", (Config{ .served_name = "m", .model_ids = &.{"m"} }).model_dir);
    if (@hasField(api.Structure, "compiler")) try std.testing.expect((api.Structure{ .kind = .json }).compiler == null);
    try std.testing.expectEqual(@as(usize, 27), n);
    std.debug.print("qualification_fb84ff48 N0 missing-hook/path assertions={d}; baseline product unsupported\n", .{n});
}

const MockSteps = struct {
    calls: usize = 0,
    len: usize = 0,
    pub fn prefill(s: *@This(), ids: []const u32) !u32 { s.calls += 1; s.len = ids.len; return 'o'; }
    pub fn step(s: *@This(), _: u32) !u32 { s.calls += 1; s.len += 1; return 'k'; }
    pub fn release(s: *@This()) void { s.len = 0; }
};
fn WrapperDouble(comptime Steps: type, comptime side: u8) type {
    return struct {
        inner: *Steps,
        len: usize = 0,
        binds: usize = 0,
        pub fn prefill(s: *@This(), ids: []const u32) !u32 { s.binds += 1; defer s.len = s.inner.len; return s.inner.prefill(ids); }
        pub fn step(s: *@This(), t: u32) !u32 { s.binds += 1; defer s.len = s.inner.len; return s.inner.step(t); }
        pub fn release(s: *@This()) void { s.inner.release(); s.len = s.inner.len; }
        pub const constructor_side = side;
    };
}

test "qualification_fb84ff48 counted TP and AFD constructor double closure" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    var counts: [2]usize = @splat(0);
    inline for (.{0, 1}) |side| {
        var steps: MockSteps = .{};
        var wrapped: WrapperDouble(MockSteps, side) = .{ .inner = &steps };
        var actual: gl.Lanes(@TypeOf(wrapped)) = .{ .steps = &wrapped, .capacity = 32, .reach = 32 };
        const be = actual.backend();
        if (@hasField(lanes.backend.Backend.VTable, "first_masked")) try std.testing.expect(be.vtable.first_masked == null);
        var stream = try lanes.Stream.init(a, .{ .id = "constructor", .prompt = &.{'h', 'i'}, .max_new = 2, .drafts = false });
        defer stream.deinit(a);
        try be.prefill(&stream);
        try std.testing.expectEqual(@as(u32, 'o'), try be.read(try be.first(&stream, 2)));
        var sampled: [1]u32 = undefined;
        var results = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
        const window: lanes.backend.Window = .{ .stream = &stream, .pending = 'o', .held = 0, .tokens = &.{}, .parents = null, .positions = &.{3} };
        try be.verify(&.{window}, &results);
        try std.testing.expectEqual(@as(u32, 'k'), sampled[0]);
        try std.testing.expectEqual(@as(usize, 2), wrapped.binds);
        try std.testing.expectEqual(@as(usize, 2), steps.calls);
        try std.testing.expectEqual(@as(u32, 1), actual.facts().exact_width);
        be.release(&stream);
        try std.testing.expect(actual.owner == null and wrapped.len == 0);
        counts[side] = steps.calls;
    }
    try std.testing.expectEqualSlices(usize, &.{2, 2}, &counts);
    std.debug.print("qualification_fb84ff48 actual Lanes over constructor doubles TP={d} AFD={d}; actual Host/Armed/Bound/device NOT RUN\n", .{counts[0], counts[1]});
}
