//! Actual four-route CPU rich wire controls; scripted time/transport, not kernel or hardware qualification.
//! GPT-6.1 Sol via Hermes; inherited Ash/Mia/stevehansen/BadAd84/JayLeaton credits preserved.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const model = @import("model_text.zig");
const json = @import("json.zig");
const http = @import("http_conn.zig");
const routes = @import("routes.zig");
const log = @import("log.zig");
const a = std.testing.allocator;
const candidate = @hasField(server.Config, "sse_keepalive_ns");
const paths = [_][]const u8{ "/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/v1/messages", "/messages", "/v1/responses", "/responses" };
const Scenario = enum { literal, tools, reasoning, utf8, hidden, backlog_stop, backlog_length, backlog_eos, prepare_cancel, before_submit, background_cancel, role_fault, final_fault, callback_fault };
var active: ?*Fixture = null;
var time_ns: i96 = 0;
var clock_reads: usize = 0;
var random_reads: usize = 0;
fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp { clock_reads += 1; return .{ .nanoseconds = time_ns }; }
fn random(_: ?*anyopaque, bytes: []u8) void { random_reads += 1; @memset(bytes, 0x12); }
fn wait(ctx: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
    if (timeout == .deadline) { time_ns = @max(time_ns, timeout.deadline.raw.toNanoseconds()); return; }
    return std.testing.io.vtable.futexWait(ctx, ptr, expected, timeout);
}
fn sleep(_: ?*anyopaque, timeout: std.Io.Timeout) std.Io.Cancelable!void {
    switch (timeout) {
        .duration => |d| time_ns += d.raw.toNanoseconds(),
        .deadline => |d| time_ns = @max(time_ns, d.raw.toNanoseconds()),
        .none => @panic("unbounded fixture sleep"),
    }
}
const Text = struct {
    fn text(t: *Text) model.Text { return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } }; }
    fn encode(_: *anyopaque, g: std.mem.Allocator, bytes: []const u8, _: bool) model.Error![]u32 {
        if (active) |f| { if (f.armed and f.scenario == .prepare_cancel) f.prepared_cancel = true; }
        const ids = try g.alloc(u32, bytes.len); for (bytes, ids) |b, *id| id.* = b; return ids;
    }
    fn decode(_: *anyopaque, g: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const bytes = try g.alloc(u8, ids.len); for (ids, bytes) |id, *b| b.* = @intCast(id);
        // Text.decode's native contract replaces a trailing split character, never emits invalid UTF-8.
        if (bytes.len > 0 and bytes[bytes.len - 1] == 0xc3) return std.mem.concat(g, u8, &.{ bytes[0 .. bytes.len - 1], "\u{fffd}" });
        return bytes;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 { return null; }
    fn tokenString(_: *anyopaque, g: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 { return std.fmt.allocPrint(g, "{d}", .{id}); }
    fn vocab(_: *anyopaque) u32 { return 256; }
    fn eos(_: *anyopaque) []const u32 { return &.{0}; }
    fn render(_: *anyopaque, g: std.mem.Allocator, _: json.Value, _: model.RenderOptions, _: *[]const u8) model.Error![]u8 {
        if (active) |f| { if (f.armed and f.scenario == .prepare_cancel) f.prepared_cancel = true; }
        return g.dupe(u8, "prompt");
    }
    fn template(_: *anyopaque) []const u8 { return ""; }
};
const Fixture = struct {
    scenario: Scenario,
    period: i64,
    armed: bool = false,
    conn: ?*http.Conn = null,
    sink: ?api.Sink = null,
    request: ?*const api.Request = null,
    id: api.Id = 0,
    wire: std.ArrayList(u8) = .empty,
    comments: usize = 0,
    writes: usize = 0,
    cancel_count: usize = 0,
    stage: usize = 0,
    submit_count: usize = 0,
    finished: bool = false,
    opened: bool = false,
    prepared_cancel: bool = false,
    disconnected: bool = false,
    cancel_bytes: usize = 0,
    fault: bool = false,
    prefill_sent: bool = false,
    no_data_before_hidden_comment: bool = false,
    prefix_wire_len: usize = 0,
    fn self(ctx: *anyopaque) *Fixture { return @ptrCast(@alignCast(ctx)); }
    fn check(f: *Fixture) void {
        if (comptime candidate) if (f.period == 0) std.debug.assert(f.conn.?.stream_activity == null);
    }
    fn engine(f: *Fixture) api.Engine { return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } }; }
    fn info(_: *anyopaque) api.Info { return .{ .context_window = 1024 }; }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{}; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
    fn finish(f: *Fixture, reason: api.Reason) void {
        if (f.finished) return;
        f.finished = true;
        const sink = f.sink.?;
        std.debug.assert(f.request != null and f.request.?.prompt.len > 0);
        f.request = null; f.sink = null;
        sink.event(sink.ctx, f.id, &.{ .finished = .{ .reason = reason, .stats = .{ .prefill_seconds = 0.5 } } });
    }
    fn tokens(f: *Fixture, text: []const u8) void {
        var buf: [256]u32 = undefined; std.debug.assert(text.len <= buf.len);
        for (text, buf[0..text.len]) |b, *id| id.* = b;
        f.sink.?.event(f.sink.?.ctx, f.id, &.{ .tokens = buf[0..text.len] });
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f = self(ctx); f.check(); f.id = id; f.sink = sink; f.request = request; f.submit_count += 1;
        if (f.scenario == .backlog_stop or f.scenario == .backlog_length or f.scenario == .backlog_eos) {
            time_ns = 4 * std.time.ns_per_s;
            sink.event(sink.ctx, id, &.{ .prefilled = 0 });
            f.tokens("ok");
            if (f.scenario == .backlog_eos) sink.event(sink.ctx, id, &.{ .tokens = &.{0} });
            f.finish(if (f.scenario == .backlog_length) .length else .stop);
        }
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const f = self(ctx); f.check(); f.cancel_count += 1;
        if (!f.disconnected) { f.disconnected = true; f.cancel_bytes = f.wire.items.len; }
        f.finish(.cancelled);
    }
    fn gone(ctx: *anyopaque) bool {
        const f = self(ctx); f.check();
        if (f.prepared_cancel) return true;
        if (f.scenario == .before_submit and f.opened) return true;
        if (f.scenario == .background_cancel and f.opened and time_ns >= 1250 * std.time.ns_per_ms) {
            f.disconnected = true; f.cancel_bytes = f.wire.items.len; return true;
        }
        if (f.submit_count == 0 or f.finished) return false;
        f.stage += 1;
        if (f.stage == 1) {
            time_ns = 900 * std.time.ns_per_ms;
            f.sink.?.event(f.sink.?.ctx, f.id, &.{ .prefilled = 0 }); f.prefill_sent = true;
            if (f.scenario == .utf8) { f.prefix_wire_len = f.wire.items.len; f.tokens("\xc3"); }
            if (f.scenario == .hidden) { f.prefix_wire_len = f.wire.items.len; f.tokens("<tool"); }
        } else if (f.stage == 2) {
            time_ns = std.time.ns_per_s;
        } else if (f.stage == 3) {
            time_ns = 2 * std.time.ns_per_s;
            switch (f.scenario) {
                .tools => f.tokens("<tool_call>\n<function=f>\n<parameter=q>\nvalue\n</parameter>\n</function>\n</tool_call>"),
                .reasoning => f.tokens("<think>reason</think>answer"),
                .utf8 => f.tokens("\xa9"),
                .hidden => f.tokens("_call>\n<function=f>\n<parameter=q>\nv\n</parameter>\n</function>\n</tool_call>"),
                else => f.tokens("literal : keepalive\n\n content"),
            }
            // Callback-fault retains the engine borrow until the failed DATA invokes native cancel.
            if (f.scenario != .callback_fault) f.finish(.stop);
        } else if (f.stage >= 6 and !f.finished) f.finish(.stop);
        return false;
    }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const f = self(ctx); f.check(); f.writes += 1;
        std.debug.assert(!f.disconnected);
        const is_header = std.mem.startsWith(u8, bytes, "HTTP/");
        if (is_header) f.opened = true;
        const fail = (f.scenario == .role_fault and !is_header and f.submit_count == 0) or
            (f.scenario == .final_fault and (std.mem.indexOf(u8, bytes, "\"finish_reason\": \"stop\"") != null or std.mem.indexOf(u8, bytes, "\"finish_reason\": \"length\"") != null or std.mem.indexOf(u8, bytes, "event: message_stop") != null or std.mem.indexOf(u8, bytes, "event: response.completed") != null)) or
            (f.scenario == .callback_fault and f.submit_count > 0 and std.mem.indexOf(u8, bytes, "literal") != null);
        if (fail) { f.fault = true; return error.Closed; }
        f.wire.appendSlice(a, bytes) catch return error.Closed;
    }
    fn commentNow(_: ?*anyopaque) i96 { return time_ns; }
    fn poll(_: ?*anyopaque, fd: std.posix.socket_t, ms: i32) http.PollResult { std.debug.assert(fd == -1 and ms > 0 and ms <= 50); return .out; }
    fn send(ctx: ?*anyopaque, fd: std.posix.socket_t, bytes: []const u8) http.SendResult {
        const f = self(ctx.?); std.debug.assert(fd == -1 and !f.disconnected and f.period > 0);
        if ((f.scenario == .utf8 or (f.scenario == .hidden and !std.mem.endsWith(u8, f.conn.?.path, "/completions")) or (f.scenario == .hidden and std.mem.endsWith(u8, f.conn.?.path, "/chat/completions"))) and f.comments == 0) {
            std.debug.assert(f.wire.items.len == f.prefix_wire_len);
            f.no_data_before_hidden_comment = true;
        }
        f.comments += 1; f.wire.appendSlice(a, bytes) catch @panic("allocation"); return .{ .sent = bytes.len };
    }
};
fn run(path: []const u8, scenario: Scenario, stream: bool, separate: bool, period: i64, label: []const u8) !void {
    time_ns = 0; clock_reads = 0; random_reads = 0;
    var vt = std.testing.io.vtable.*; vt.now = now; vt.random = random; vt.futexWait = wait; vt.sleep = sleep;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vt };
    log.init(io, true); defer log.init(std.testing.io, true);
    var f: Fixture = .{ .scenario = scenario, .period = period }; defer f.wire.deinit(a);
    std.debug.assert(active == null); active = &f; defer active = null;
    var text: Text = .{};
    var config: server.Config = .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = scenario == .reasoning, .use_drafts = false, .keep_warm_s = 0 };
    if (comptime candidate) if (!std.mem.eql(u8, label, "unset")) { config.sse_keepalive_ns = period; };
    const srv = try server.Server.init(a, io, f.engine(), text.text(), config, null); defer srv.deinit();
    if (scenario == .background_cancel) srv.preparing.store(1, .release);
    defer if (scenario == .background_cancel) srv.preparing.store(0, .release);
    var arena = std.heap.ArenaAllocator.init(a); defer arena.deinit(); const g = arena.allocator();
    const chat_route = std.mem.endsWith(u8, path, "/chat/completions");
    const messages = std.mem.endsWith(u8, path, "/messages");
    const responses = std.mem.endsWith(u8, path, "/responses");
    const bg = scenario == .background_cancel;
    const tool = scenario == .tools or scenario == .hidden;
    const extra = if (!tool) "" else if (messages) ",\"tools\":[{\"name\":\"f\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}]" else if (responses) ",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}]" else ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}}]";
    const usage = if (separate) ",\"stream_options\":{\"include_usage\":true}" else "";
    const background = if (!bg) "" else if (messages) ",\"system\":\"write title\"" else ",\"priority\":\"background\"";
    const thinking = if (scenario == .reasoning and messages) ",\"thinking\":{\"type\":\"adaptive\"}" else "";
    const body = if (responses) try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"input\":\"x\",\"stream\":{},\"max_output_tokens\":200{s}{s}{s}}}", .{stream, extra, background, thinking}) else if (messages or chat_route) try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":{},\"max_tokens\":200{s}{s}{s}{s}}}", .{stream, extra, if (chat_route) usage else "", background, thinking}) else try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"prompt\":\"x\",\"stream\":{},\"max_tokens\":200{s}{s}{s}}}", .{stream, extra, usage, background});
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(g, "{d}", .{body.len}) }};
    var c: http.Conn = .{ .fd = -1, .peer = "CPU fixture", .gpa = g, .buf = try g.dupe(u8, body), .end = body.len, .method = "POST", .version = "HTTP/1.1", .path = path, .headers = &headers, .quiet_log = true, .test_transport = .{ .ctx = &f, .write_all = Fixture.write, .peer_gone = Fixture.gone } };
    f.conn = &c; f.armed = true;
    if (comptime candidate) c.test_transport.?.comment_ops = .{ .ctx = &f, .now = Fixture.commentNow, .poll = Fixture.poll, .send = Fixture.send };
    const Observation = @import("sse_test_observe.zig").Test;
    var probe: @import("qualification_59878042_private_probe.zig").Probe = .{ .enabled = period > 0, .streamed = stream };
    var hook: Observation.Hook = .{ .ctx = &probe, .record = @import("qualification_59878042_private_probe.zig").Probe.record };
    const previous = Observation.active; Observation.active = &hook;
    defer Observation.active = previous;
    routes.dispatch(srv, &c, g);
    try probe.check();
    std.debug.print("qualification_013d0d4c setting={s} route={s} scenario={s} stream={} usage={} starts={d} stops={d} session_events={d} sinks={d} generates={d} mutex={d}\n", .{label, path, @tagName(scenario), stream, separate, probe.starts, probe.stops, probe.session_events, probe.sinks, probe.generates, probe.mutex_checks});
    if (comptime candidate) try std.testing.expect(c.stream_activity == null);
    try std.testing.expect(f.request == null and f.sink == null);
    try std.testing.expectEqual(@as(i64, if (bg) 1 else 0), srv.preparing.load(.acquire));
    if (period == 0 or !stream) try std.testing.expectEqual(@as(usize, 0), f.comments);
    if (scenario == .prepare_cancel or scenario == .before_submit or scenario == .background_cancel) {
        try std.testing.expectEqual(@as(usize, 0), f.submit_count);
        try std.testing.expectEqual(@as(usize, 0), f.cancel_count);
        if (scenario != .background_cancel) try std.testing.expectEqual(@as(usize, 0), f.comments);
        if (scenario == .background_cancel and stream and period > 0) try std.testing.expectEqual(@as(usize, 1), f.comments);
    }
    if (stream and period > 0 and (scenario == .utf8 or (scenario == .hidden and (messages or chat_route or responses)))) try std.testing.expect(f.no_data_before_hidden_comment);
    if (scenario == .backlog_stop or scenario == .backlog_length or scenario == .backlog_eos) try std.testing.expectEqual(@as(usize, 0), f.comments);
    if (scenario == .final_fault and stream) {
        try std.testing.expect(f.fault and c.broken and f.finished);
        try std.testing.expectEqual(@as(usize, 1), f.submit_count);
    }
    if (scenario == .callback_fault and stream) {
        try std.testing.expect(f.fault and c.broken and f.finished);
        try std.testing.expectEqual(@as(usize, 1), f.cancel_count);
        try std.testing.expectEqual(f.cancel_bytes, f.wire.items.len);
    }
    std.debug.print("qualification_cd8779fb {s} {s} {s} {} {} clock={d} random={d} submit={d} cancel={d} comments={d} fault={}\n", .{label, path, @tagName(scenario), stream, separate, clock_reads, random_reads, f.submit_count, f.cancel_count, f.comments, f.fault});
    std.debug.print("qualification_68e90858 {s} {s} {s} {} {} {x}\n", .{label, path, @tagName(scenario), stream, separate, f.wire.items});
}
test "qualification_4533b989 CPU actual rich route native independent control inventory" {
    for (paths) |path| for ([_]Scenario{ .literal, .tools, .reasoning, .utf8, .hidden, .backlog_stop, .backlog_length, .backlog_eos }) |scenario| for ([_]bool{ false, true }) |stream| for ([_]bool{ false, true }) |separate| {
        try run(path, scenario, stream, separate, 0, "unset");
        if (comptime candidate) {
            try run(path, scenario, stream, separate, 0, "zero");
            try run(path, scenario, stream, separate, std.time.ns_per_s, "on");
        }
    };
}
test "qualification_4533b989 CPU actual prepare before submit background and native write failures" {
    for (paths) |path| for ([_]Scenario{ .prepare_cancel, .before_submit, .background_cancel, .role_fault, .final_fault, .callback_fault }) |scenario| {
        try run(path, scenario, true, false, 0, "unset");
        if (comptime candidate) {
            try run(path, scenario, true, false, 0, "zero");
            try run(path, scenario, true, false, std.time.ns_per_s, "on");
        }
    };
}
