//! literal_boundary (local integration): Mia GLM #158's App.run cases through the actual routes (routes.dispatch over the in-memory test
//! transport, prose_filter_852a's harness), a fake engine whose every token is one character (or one GLM tag), CPU only.
//! Reimplemented from MiaAI-Lab GLM PR #158 (@rwl4, 029a852f) tests/tool_envelope_contract.py; no code copied.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const model = @import("model_text.zig");
const json = @import("json.zig");
const http = @import("http_conn.zig");
const routes = @import("routes.zig");
const log = @import("log.zig");
const reply_text = @import("reply_text.zig");
const u = @import("literal_boundary_units_test.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const gpa = std.testing.allocator;

/// GLM-5.3's single-token tags; in `.special` mode the fake engine emits each as one token, as the real tokenizer does.
const specials = [_][]const u8{ "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "</think>" };
const special_base: u32 = 0x200000;
const Mode = enum { chars, special };

var random_counter: u8 = 0;
fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
    return .{ .nanoseconds = 10 * std.time.ns_per_s };
}
fn random(_: ?*anyopaque, bytes: []u8) void {
    random_counter +%= 1;
    @memset(bytes, random_counter);
}

fn nextCp(s: []const u8, i: usize) struct { cp: u32, len: usize } {
    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .cp = s[i], .len = 1 };
    if (i + n > s.len) return .{ .cp = s[i], .len = 1 };
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return .{ .cp = s[i], .len = 1 };
    return .{ .cp = cp, .len = n };
}

/// The reply as the fake engine's token ids: one per character, or GLM tags as one token each.
fn tokenize(a: Allocator, s: []const u8, mode: Mode) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var i: usize = 0;
    outer: while (i < s.len) {
        if (mode == .special) for (specials, 0..) |tag, k| if (std.mem.startsWith(u8, s[i..], tag)) {
            try out.append(a, special_base + @as(u32, @intCast(k)));
            i += tag.len;
            continue :outer;
        };
        const c = nextCp(s, i);
        try out.append(a, c.cp);
        i += c.len;
    }
    return out.items;
}

const Text = struct {
    fn text(t: *Text) model.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } };
    }
    fn encode(_: *anyopaque, g: Allocator, s: []const u8, _: bool) model.Error![]u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < s.len) {
            const c = nextCp(s, i);
            try out.append(g, c.cp);
            i += c.len;
        }
        return out.items;
    }
    fn decode(_: *anyopaque, g: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| {
            if (id >= special_base and id < special_base + specials.len) {
                try out.appendSlice(g, specials[id - special_base]);
                continue;
            }
            var buf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(@intCast(id), &buf) catch {
                try out.appendSlice(g, "\u{fffd}");
                continue;
            };
            try out.appendSlice(g, buf[0..n]);
        }
        return out.items;
    }
    fn tokenId(_: *anyopaque, s: []const u8) ?u32 {
        for (specials, 0..) |tag, k| if (std.mem.eql(u8, s, tag)) return special_base + @as(u32, @intCast(k));
        return null;
    }
    fn tokenString(_: *anyopaque, g: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(g, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return special_base + specials.len;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{0};
    }
    fn render(_: *anyopaque, g: Allocator, messages: Value, _: model.RenderOptions, _: *[]const u8) model.Error![]u8 {
        return json.stringify(g, messages, .{});
    }
    fn template(_: *anyopaque) []const u8 {
        return "<tool_call>";
    }
};

const Engine = struct {
    tokens: []const u32,
    width: usize,
    ended: bool,
    submits: usize = 0,
    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 262144, .name = "literal_boundary CPU" };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        e.submits += 1;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var all: std.ArrayList(u32) = .empty;
        defer all.deinit(gpa);
        all.appendSlice(gpa, e.tokens) catch return error.Busy;
        if (e.ended) all.append(gpa, 0) catch return error.Busy;
        var i: usize = 0;
        while (i < all.items.len) : (i += e.width) {
            const piece = all.items[i..@min(all.items.len, i + e.width)];
            const tokens = gpa.dupe(u32, piece) catch return error.Busy;
            sink.event(sink.ctx, id, &.{ .tokens = tokens });
            @memset(tokens, 0xee); // the server must have copied the borrowed slice
            gpa.free(tokens);
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (e.ended) .stop else .length, .stats = .{ .prefill_seconds = 0.01 } } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
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
        s.wire.appendSlice(gpa, bytes) catch return error.Closed;
    }
    fn gone(_: *anyopaque) bool {
        return false;
    }
};

const Route = enum { chat, messages, responses };
const Run = struct { reply: []const u8, mode: Mode = .chars, width: usize = 1, thinking: bool = false, stream: bool = false, single: bool = false, ended: bool = true, route: Route = .chat, tools: []const u8 = u.write_tools };

/// What a client sees: finish reason, the calls with their arguments, the content (streamed: every content delta joined).
const Seen = struct { status: u16, finish: []const u8 = "", calls: [][]const u8 = &.{}, names: [][]const u8 = &.{}, content: []const u8 = "", reasoning: []const u8 = "", completion_tokens: i64 = -1 };

fn body(g: Allocator, r: Run) ![]const u8 {
    const t = try u.tools(g, r.tools);
    const f = t[0].get("function").?;
    const name = f.strField("name").?;
    const params = try json.stringify(g, f.get("parameters").?, .{});
    const policy = if (r.single) ",\"parallel_tool_calls\":false" else "";
    return switch (r.route) {
        .chat => std.fmt.allocPrint(g, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"CPU synthetic fixture.\"}}],\"tools\":{s},\"stream\":{},\"max_tokens\":65536{s}}}", .{ r.tools, r.stream, policy }),
        .messages => std.fmt.allocPrint(g, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"CPU synthetic fixture.\"}}],\"tools\":[{{\"name\":\"{s}\",\"input_schema\":{s}}}],\"stream\":{},\"max_tokens\":65536{s}}}", .{ name, params, r.stream, if (r.thinking) ",\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":4096}" else "" }),
        .responses => std.fmt.allocPrint(g, "{{\"model\":\"m\",\"input\":\"CPU synthetic fixture.\",\"tools\":[{{\"type\":\"function\",\"name\":\"{s}\",\"parameters\":{s}}}],\"stream\":{},\"max_output_tokens\":65536}}", .{ name, params, r.stream }),
    };
}

fn exchange(g: Allocator, reply: []const u8, r: Run) !Seen {
    var vt: std.Io.VTable = std.testing.io.vtable.*;
    vt.now = now;
    vt.random = random;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vt };
    log.init(io, true);
    defer log.init(std.testing.io, true);
    random_counter = 0;
    var eng: Engine = .{ .tokens = try tokenize(g, reply, r.mode), .width = r.width, .ended = r.ended };
    var text: Text = .{};
    const srv = try server.Server.init(gpa, io, eng.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = r.thinking, .use_drafts = false, .keep_warm_s = 0 }, null);
    defer srv.deinit();
    const req = try body(g, r);
    const path = switch (r.route) {
        .chat => "/v1/chat/completions",
        .messages => "/v1/messages",
        .responses => "/v1/responses",
    };
    var sink: Sink = .{};
    defer sink.wire.deinit(gpa);
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(g, "{d}", .{req.len}) }};
    var c: http.Conn = .{ .fd = -1, .peer = "CPU memory", .gpa = g, .buf = try g.dupe(u8, req), .end = req.len, .method = "POST", .version = "HTTP/1.1", .path = path, .headers = &headers, .quiet_log = true, .test_transport = .{ .ctx = &sink, .write_all = Sink.write, .peer_gone = Sink.gone } };
    routes.dispatch(srv, &c, g);
    const wire = try g.dupe(u8, sink.wire.items);
    const split = std.mem.indexOf(u8, wire, "\r\n\r\n") orelse return error.MissingWire;
    const status = try std.fmt.parseInt(u16, wire[9..12], 10);
    const payload = wire[split + 4 ..];
    if (r.stream) return streamed(g, status, payload, r.route);
    return whole(g, status, (try json.parse(g, payload)).ok, r.route);
}

fn whole(g: Allocator, status: u16, v: Value, route: Route) !Seen {
    var seen: Seen = .{ .status = status };
    var calls: std.ArrayList([]const u8) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    switch (route) {
        .chat => {
            const ch = v.get("choices").?.array[0];
            seen.finish = ch.strField("finish_reason") orelse "";
            const m = ch.get("message").?;
            seen.content = m.strField("content") orelse "";
            seen.reasoning = m.strField("reasoning_content") orelse "";
            if (m.get("tool_calls")) |tc| for (tc.array) |one| {
                try calls.append(g, one.get("function").?.strField("arguments").?);
                try names.append(g, one.get("function").?.strField("name").?);
            };
            seen.completion_tokens = v.get("usage").?.get("completion_tokens").?.int64().?;
        },
        .messages => {
            seen.finish = v.strField("stop_reason") orelse "";
            var content: std.ArrayList(u8) = .empty;
            for (v.get("content").?.array) |b| {
                const kind = b.strField("type").?;
                if (std.mem.eql(u8, kind, "text")) try content.appendSlice(g, b.strField("text").?);
                if (std.mem.eql(u8, kind, "tool_use")) {
                    try calls.append(g, try json.stringify(g, b.get("input").?, .{ .ascii = false, .compact = true }));
                    try names.append(g, b.strField("name").?);
                }
            }
            seen.content = content.items;
        },
        .responses => {
            seen.finish = v.strField("status") orelse "";
            var content: std.ArrayList(u8) = .empty;
            for (v.get("output").?.array) |item| {
                const kind = item.strField("type").?;
                if (std.mem.eql(u8, kind, "message")) for (item.get("content").?.array) |part| {
                    if (part.strField("text")) |t| try content.appendSlice(g, t);
                };
                if (std.mem.eql(u8, kind, "function_call")) {
                    try calls.append(g, item.strField("arguments").?);
                    try names.append(g, item.strField("name").?);
                }
            }
            seen.content = content.items;
        },
    }
    seen.calls = calls.items;
    seen.names = names.items;
    return seen;
}

fn streamed(g: Allocator, status: u16, payload: []const u8, route: Route) !Seen {
    var seen: Seen = .{ .status = status };
    if (route != .chat) return error.StreamedTranslatorNotCovered;
    var content: std.ArrayList(u8) = .empty;
    var reasoning: std.ArrayList(u8) = .empty;
    var args: std.ArrayList(std.ArrayList(u8)) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.splitSequence(u8, payload, "\n");
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ") or std.mem.eql(u8, line, "data: [DONE]")) continue;
        const v = (try json.parse(g, line[6..])).ok;
        if (v.get("usage")) |use| if (use == .object) {
            seen.completion_tokens = use.get("completion_tokens").?.int64().?;
        };
        const choices = (v.get("choices") orelse continue).array;
        if (choices.len == 0) continue;
        const ch = choices[0];
        if (ch.strField("finish_reason")) |f| seen.finish = f;
        const d = ch.get("delta") orelse continue;
        if (d.strField("content")) |t| try content.appendSlice(g, t);
        if (d.strField("reasoning_content")) |t| try reasoning.appendSlice(g, t);
        if (d.get("tool_calls")) |tc| for (tc.array) |one| {
            const index: usize = @intCast(one.get("index").?.int64().?);
            if (one.strField("id") != null) {
                while (args.items.len <= index) try args.append(g, .empty);
                try names.append(g, if (one.get("function")) |f| f.strField("name") orelse "" else "");
            }
            if (index < args.items.len) if (one.get("function")) |f| if (f.strField("arguments")) |t| try args.items[index].appendSlice(g, t);
        };
    }
    const calls = try g.alloc([]const u8, args.items.len);
    for (args.items, calls) |x, *slot| slot.* = x.items;
    seen.calls = calls;
    seen.names = names.items;
    seen.content = content.items;
    seen.reasoning = reasoning.items;
    return seen;
}

fn argOk(g: Allocator, arguments: []const u8, key: []const u8, want: []const u8) !bool {
    const v = switch (try json.parse(g, arguments)) {
        .ok => |x| x,
        .err => return false,
    };
    const got = v.strField(key) orelse return false;
    return std.mem.eql(u8, got, want);
}

fn describe(g: Allocator, r: Run, s: Seen) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(g, "status={d} finish={s} calls={d}", .{ s.status, s.finish, s.calls.len });
    for (s.calls) |c| try out.print(g, " args={s}", .{u.clip(c, 120)});
    try out.print(g, " content={s} tokens={d} mode={s} width={d}", .{ u.clip(s.content, 120), s.completion_tokens, @tagName(r.mode), r.width });
    return out.items;
}

fn label(g: Allocator, prefix: []const u8, r: Run) ![]const u8 {
    return std.fmt.allocPrint(g, "{s}-{s}-w{d}-t{d}-s{d}-p{d}-{s}", .{ prefix, @tagName(r.mode), r.width, @intFromBool(r.thinking), @intFromBool(r.stream), @intFromBool(r.single), @tagName(r.route) });
}

test "literal_boundary routes: Mia #158 App.run matrix and extras through the actual routes" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const g = arena.allocator();
    var rec: u.Rec = .{ .a = g, .suite = "routes" };
    const whole_width: usize = 1 << 20;
    // A. one call per value: fragmented widths x thinking x stream x one-call policy x tokenizer form (Mia's first loop)
    for (u.values, 0..) |v, vi| {
        const raw = try u.call(g, v);
        for ([_]Mode{ .chars, .special }) |mode| for ([_]usize{ 1, 2, 3, 7, 17, whole_width }) |width| for ([_]bool{ false, true }) |thinking| for ([_]bool{ false, true }) |stream| for ([_]bool{ false, true }) |single| {
            const r: Run = .{ .reply = "", .mode = mode, .width = width, .thinking = thinking, .stream = stream, .single = single };
            const full = try std.mem.concat(g, u8, &.{ if (thinking) "brief thought</think>" else "", "Before. ", raw, " After." });
            const s = try exchange(g, full, r);
            const content_ok = if (stream) std.mem.eql(u8, reply_text.pyStrip(s.content), "Before.  After.") else std.mem.eql(u8, s.content, "Before.  After.");
            const pass = s.status == 200 and std.mem.eql(u8, s.finish, "tool_calls") and s.calls.len == 1 and (try argOk(g, s.calls[0], "content", v)) and content_ok;
            rec.check(try label(g, try std.fmt.allocPrint(g, "A-v{d}", .{vi}), r), if (stream) "stream" else "whole", "chat route: chat.zig:413/530 hide + openai.zig:308-322 attachCalls", "boundary", pass, try describe(g, r, s));
        };
    }
    // B. two calls, the first holding a literal closer (Mia: two calls, no content)
    for ([_]usize{ 1, 3, 17, whole_width }) |width| for ([_]bool{ false, true }) |stream| {
        const r: Run = .{ .reply = "", .width = width, .stream = stream };
        const raw = try std.mem.concat(g, u8, &.{ try u.call(g, "a </tool_call> b"), "\n", try u.call(g, "second") });
        const s = try exchange(g, raw, r);
        const pass = s.calls.len == 2 and (try argOk(g, s.calls[0], "content", "a </tool_call> b")) and (try argOk(g, s.calls[1], "content", "second")) and reply_text.pyStrip(s.content).len == 0;
        rec.check(try label(g, "B-two", r), if (stream) "stream" else "whole", "chat route", "boundary", pass, try describe(g, r, s));
    };
    // C. truncated (no end token, finish length): no call, no content (Mia's two cuts)
    {
        const cut1 = try u.call(g, "a </tool_call> b");
        const cuts = [_][]const u8{ cut1[0 .. cut1.len - "</arg_value></tool_call>".len], "<tool_call>write<arg_key>content</arg_key><arg_value>cut" };
        for (cuts, 0..) |cut, ci| for ([_]usize{ 1, 3, 17, whole_width }) |width| for ([_]bool{ false, true }) |stream| {
            const r: Run = .{ .reply = "", .width = width, .stream = stream, .ended = false };
            const s = try exchange(g, cut, r);
            const no_call = s.calls.len == 0 and std.mem.eql(u8, s.finish, "length");
            const no_content = s.content.len == 0;
            // cut0 holds the literal closer: its no-call half is the boundary bug (below); a cut call's raw text coming back as
            // content on finish=length is the pre-existing truncation contract (openai.zig:529-530), as for cut1
            rec.check(try label(g, try std.fmt.allocPrint(g, "C-cut{d}", .{ci}), r), if (stream) "stream" else "whole", "chat route: openai.zig:308-322, :529-530", "contract", no_call and no_content, try describe(g, r, s));
            if (ci == 0) rec.check(try label(g, "C-cut0-nocall", r), if (stream) "stream" else "whole", "tool_parse.zig:55-58 envelopes", "boundary", no_call, try describe(g, r, s));
        };
    }
    // E. a call ending an unclosed think block, including Mia's long-call commit path (THINK_CALL_HOLD 1024 chars)
    {
        var long: std.ArrayList(u8) = .empty;
        for (0..200) |_| try long.appendSlice(g, "long payload ");
        try long.appendSlice(g, "</tool_call> end");
        for ([_][]const u8{ "before </tool_call> after", long.items }, 0..) |v, vi| for ([_]usize{ 1, 7, 64, whole_width }) |width| for ([_]bool{ false, true }) |single| for ([_]bool{ false, true }) |stream| {
            const r: Run = .{ .reply = "", .width = width, .thinking = true, .stream = stream, .single = single };
            const s = try exchange(g, try std.mem.concat(g, u8, &.{ "thought ", try u.call(g, v) }), r);
            const pass = s.calls.len == 1 and (try argOk(g, s.calls[0], "content", v)) and s.content.len == 0;
            rec.check(try label(g, try std.fmt.allocPrint(g, "E-think{d}", .{vi}), r), if (stream) "stream" else "whole", "chat.zig:512-520 closeCall + splitThinking, then attachCalls", "boundary", pass, try describe(g, r, s));
        };
    }
    // G. sparkglm RC4.1's case: one argument value far over 1,024 tokens (no literal closer, then with one)
    {
        var file: std.ArrayList(u8) = .empty;
        for (0..300) |i| try file.print(g, "line {d:0>4} of a long generated source file\n", .{i});
        var file_close: std.ArrayList(u8) = .empty;
        try file_close.appendSlice(g, file.items);
        try file_close.appendSlice(g, "const close = \"</tool_call>\";\n");
        for ([_][]const u8{ file.items, file_close.items }, 0..) |v, vi| for ([_]usize{ 1, 64, whole_width }) |width| for ([_]bool{ false, true }) |stream| {
            const r: Run = .{ .reply = "", .width = width, .stream = stream };
            const s = try exchange(g, try u.call(g, v), r);
            const pass = std.mem.eql(u8, s.finish, "tool_calls") and s.calls.len == 1 and (try argOk(g, s.calls[0], "content", v)) and s.completion_tokens > 1024 and reply_text.pyStrip(s.content).len == 0;
            rec.check(try label(g, try std.fmt.allocPrint(g, "G-long{d}", .{vi}), r), if (stream) "stream" else "whole", "no body cap on the server; chat.zig:384 max_tokens only", if (vi == 0) "long" else "boundary", pass, try describe(g, r, s));
        };
    }
    // H. extras: a two-argument write_file, a literal </think> in a value, and the Messages/Responses translations
    {
        const wf = "<tool_call>write_file<arg_key>path</arg_key><arg_value>zig/src/server/x.zig</arg_value><arg_key>content</arg_key><arg_value>const close = \"</tool_call>\";\n</arg_value></tool_call>";
        for ([_]usize{ 1, whole_width }) |width| for ([_]bool{ false, true }) |stream| {
            const r: Run = .{ .reply = "", .width = width, .stream = stream, .tools = u.file_tools };
            const s = try exchange(g, wf, r);
            const pass = s.calls.len == 1 and (try argOk(g, s.calls[0], "path", "zig/src/server/x.zig")) and (try argOk(g, s.calls[0], "content", "const close = \"</tool_call>\";\n")) and reply_text.pyStrip(s.content).len == 0;
            rec.check(try label(g, "H-writefile", r), if (stream) "stream" else "whole", "tool_parse.zig:390 glmCall drops the cut value; reply_text.zig:127 leak", "boundary", pass, try describe(g, r, s));
        };
        for ([_]bool{ false, true }) |stream| {
            const r: Run = .{ .reply = "", .thinking = true, .stream = stream };
            const v = "x </think> y";
            const s = try exchange(g, try std.mem.concat(g, u8, &.{ "thought ", try u.call(g, v) }), r);
            const pass = s.calls.len == 1 and (try argOk(g, s.calls[0], "content", v)) and s.content.len == 0;
            rec.check(try label(g, "H-thinkclose-unclosed", r), if (stream) "stream" else "whole", "reply_text.zig:76 splitThinking finds </think> inside the value", "extra", pass, try describe(g, r, s));
        }
        for ([_]Route{ .messages, .responses }) |route| for ([_]bool{ false, true }) |thinking| {
            const r: Run = .{ .reply = "", .thinking = thinking, .route = route };
            const v = u.values[1];
            const full = try std.mem.concat(g, u8, &.{ if (thinking) "brief thought</think>" else "", "Before. ", try u.call(g, v), " After." });
            const s = try exchange(g, full, r);
            const pass = s.calls.len == 1 and (try argOk(g, s.calls[0], "content", v)) and std.mem.eql(u8, reply_text.pyStrip(s.content), "Before.  After.");
            rec.check(try label(g, "H-translate", r), "whole", "anthropic.zig / responses.zig translate the chat reply", "boundary", pass, try describe(g, r, s));
        };
    }
    try rec.done();
}
