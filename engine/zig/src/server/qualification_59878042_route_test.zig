//! Actual routes/translator/HTTP bytes with fixed Io and an explicitly scripted CPU engine (not LaneHost).
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const model_text = @import("model_text.zig");
const json = @import("json.zig");
const http = @import("http_conn.zig");
const routes = @import("routes.zig");
const log = @import("log.zig");
const a = std.testing.allocator;
const candidate = @hasField(server.Config, "sse_keepalive_ns");
var clock_ns: i96 = 0;
var clock_reads: usize = 0;
var random_calls: usize = 0;
fn fixedNow(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp { clock_reads += 1; return .{ .nanoseconds = clock_ns }; }
fn fixedRandom(_: ?*anyopaque, bytes: []u8) void { random_calls += 1; @memset(bytes, 0x12); }
fn fixedWait(ctx: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
    switch (timeout) {
        .deadline => |d| clock_ns = @max(clock_ns, d.raw.toNanoseconds()),
        else => return std.testing.io.vtable.futexWait(ctx, ptr, expected, timeout),
    }
}
const Text = struct {
    fn text(t: *Text) model_text.Text { return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } }; }
    fn encode(_: *anyopaque, g: std.mem.Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try g.alloc(u32, s.len); for (s, ids) |b, *p| p.* = b; return ids;
    }
    fn decode(_: *anyopaque, g: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const bytes = try g.alloc(u8, ids.len); for (ids, bytes) |id, *p| p.* = @intCast(id); return bytes;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 { return null; }
    fn tokenString(_: *anyopaque, g: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 { return std.fmt.allocPrint(g, "{d}", .{id}); }
    fn vocab(_: *anyopaque) u32 { return 256; }
    fn eos(_: *anyopaque) []const u32 { return &.{}; }
    fn render(_: *anyopaque, g: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 { return g.dupe(u8, "prompt"); }
    fn template(_: *anyopaque) []const u8 { return ""; }
};
var broken_prefix: usize = 3;
pub const Mode = enum { success, busy, failed, cancelled, closed_open, broken_comment, finished_backlog, context, unsupported, invalid_body };
const Fixture = struct {
    wire: std.ArrayList(u8) = .empty,
    sink: ?api.Sink = null,
    id: api.Id = 0,
    stages: usize = 0,
    comments: usize = 0,
    cancelled: usize = 0,
    done: bool = false,
    submitted: bool = false,
    mode: Mode,
    fn self(ctx: ?*anyopaque) *Fixture { return @ptrCast(@alignCast(ctx.?)); }
    fn engine(f: *Fixture) api.Engine { return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .admit = admit } }; }
    fn info(_: *anyopaque) api.Info { return .{ .context_window = 128 }; }
    fn admit(ctx: *anyopaque, _: bool) bool { return self(ctx).mode != .busy; }
    fn finish(f: *Fixture, reason: api.Reason) void {
        if (f.done) return;
        f.done = true;
        if (f.sink) |s| s.event(s.ctx, f.id, &.{ .finished = .{ .reason = reason, .message = if (reason == .failed) "scripted failure" else "", .stats = .{ .prefill_seconds = 2 } } });
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f = self(ctx); f.id = id; f.sink = sink; f.submitted = true;
        if (f.mode == .finished_backlog) {
            clock_ns = 6 * std.time.ns_per_s;
            sink.event(sink.ctx, id, &.{ .prefilled = 0 });
            sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
            f.finish(.stop);
        }
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void { const f = self(ctx); f.cancelled += 1; f.finish(.cancelled); }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{}; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
    fn gone(ctx: *anyopaque) bool {
        const f = self(ctx);
        if (!f.submitted or f.done) return false;
        f.stages += 1; clock_ns = @as(i96, @intCast(f.stages)) * 2 * std.time.ns_per_s;
        if (f.stages >= 3) {
            if (f.mode == .cancelled) return true;
            const s = f.sink.?;
            s.event(s.ctx, f.id, &.{ .prefilled = 0 });
            if (f.mode != .failed) s.event(s.ctx, f.id, &.{ .tokens = &.{ 'o', 'k' } });
            f.finish(if (f.mode == .failed) .failed else .stop);
        }
        return false;
    }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const f = self(ctx);
        if (f.mode == .closed_open) return error.Closed;
        f.wire.appendSlice(a, bytes) catch return error.Closed;
    }
    fn now(_: ?*anyopaque) i96 { return clock_ns; }
    fn poll(_: ?*anyopaque, fd: std.posix.socket_t, ms: i32) http.PollResult {
        std.debug.assert(fd == -1 and ms > 0 and ms <= 50); return .out;
    }
    fn send(ctx: ?*anyopaque, fd: std.posix.socket_t, bytes: []const u8) http.SendResult {
        std.debug.assert(fd == -1);
        const f = self(ctx);
        if (f.mode == .broken_comment) {
            if (f.comments > 0 or broken_prefix == 0) return .failed;
            f.comments += 1; f.wire.appendSlice(a, bytes[0..broken_prefix]) catch @panic("allocation"); return .{ .sent = broken_prefix };
        }
        f.comments += 1; f.wire.appendSlice(a, bytes) catch @panic("allocation"); return .{ .sent = bytes.len };
    }
};
pub const paths = [_][]const u8{ "/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/v1/messages", "/messages", "/v1/responses", "/responses" };
pub fn run(path: []const u8, mode: Mode, stream: bool, period: i64, label: []const u8) ![]u8 {
    clock_ns = 0; clock_reads = 0; random_calls = 0;
    var vtable = std.testing.io.vtable.*; vtable.now = fixedNow; vtable.random = fixedRandom; vtable.futexWait = fixedWait;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };
    log.init(io, true); defer log.init(std.testing.io, true);
    var f: Fixture = .{ .mode = mode }; defer f.wire.deinit(a);
    var text: Text = .{};
    var config: server.Config = .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .keep_warm_s = 0 };
    if (comptime candidate) config.sse_keepalive_ns = period;
    const srv = try server.Server.init(a, io, f.engine(), text.text(), config, null); defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(a); defer arena.deinit(); const g = arena.allocator();
    const tokens: usize = if (mode == .context) 999 else 8;
    const extra = if (mode != .unsupported) "" else if (std.mem.endsWith(u8, path, "/messages"))
        ",\"tools\":[{\"name\":\"f\",\"input_schema\":{\"type\":\"object\"}}],\"tool_choice\":{\"type\":\"any\"}"
    else if (std.mem.endsWith(u8, path, "/responses"))
        ",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}],\"tool_choice\":\"required\""
    else if (!std.mem.endsWith(u8, path, "/chat/completions")) ",\"response_format\":{\"type\":\"json_object\"}"
    else ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}],\"tool_choice\":\"required\"";
    const body = if (mode == .invalid_body) "{}" else if (std.mem.endsWith(u8, path, "/responses"))
        try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"input\":\"x\",\"stream\":{},\"max_output_tokens\":{d}{s}}}", .{stream, tokens, extra})
    else if (std.mem.endsWith(u8, path, "/messages") or std.mem.endsWith(u8, path, "/chat/completions"))
        try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":{},\"max_tokens\":{d}{s}}}", .{stream, tokens, extra})
    else try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"prompt\":\"x\",\"stream\":{},\"max_tokens\":{d}{s}}}", .{stream, tokens, extra});
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(g, "{d}", .{body.len}) }};
    var c: http.Conn = .{ .fd = -1, .peer = "fixture", .gpa = g, .buf = try g.dupe(u8, body), .end = body.len, .method = "POST", .version = "HTTP/1.1", .path = path, .headers = &headers, .quiet_log = true, .test_transport = .{ .ctx = &f, .write_all = Fixture.write, .peer_gone = Fixture.gone } };
    if (comptime candidate) c.test_transport.?.comment_ops = .{ .ctx = &f, .now = Fixture.now, .poll = Fixture.poll, .send = Fixture.send };
    const Observation = @import("sse_test_observe.zig").Test;
    const Private = @import("qualification_59878042_private_probe.zig");
    var probe: Private.Probe = .{ .enabled = period > 0, .streamed = stream, .upstream = Observation.active };
    var hook: Observation.Hook = .{ .ctx = &probe, .record = Private.Probe.record };
    const previous = Observation.active; Observation.active = &hook;
    defer Observation.active = previous;
    routes.dispatch(srv, &c, g);
    try probe.check();
    if (mode == .broken_comment and period > 0) {
        try std.testing.expect(probe.at(.session).? < probe.at(.drain_begin).?);
        // Last session transition is the failed-comment immediate stop, before drain.
        try std.testing.expectEqual(@as(usize, 1), probe.stops);
        var last_session: usize = 0;
        for (probe.tags[0..probe.n], 0..) |tag, i| if (tag == .session) { last_session = i; };
        try std.testing.expect(last_session < probe.at(.drain_begin).?);
        std.debug.print("qualification_826b01bf route={s} prefix={d} stop_before_drain=true drain_finished=true no_tick_in_drain=true\n", .{path, broken_prefix});
    }
    std.debug.print("qualification_b6c5136c setting={s} route={s} mode={s} stream={} starts={d} stops={d} session_events={d} sinks={d} generates={d}\n", .{label, path, @tagName(mode), stream, probe.starts, probe.stops, probe.session_events, probe.sinks, probe.generates});
    if (comptime candidate) try std.testing.expect(c.stream_activity == null);
    try std.testing.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
    if (!stream or period == 0 or mode == .busy or mode == .closed_open or mode == .context or mode == .unsupported or mode == .invalid_body or mode == .finished_backlog) try std.testing.expectEqual(@as(usize, 0), f.comments);
    if (mode == .cancelled or (mode == .broken_comment and period > 0)) try std.testing.expectEqual(@as(usize, 1), f.cancelled);
    if (stream and period > 0 and (mode == .success or mode == .failed)) try std.testing.expect(f.comments >= 2);
    if (mode == .broken_comment and period > 0) {
        try std.testing.expect(c.broken and c.close); try std.testing.expect(std.mem.endsWith(u8, f.wire.items, ": keepalive\n\n"[0..broken_prefix]));
        for ([_][]const u8{ "[DONE]", "message_stop", "response.completed", "response.incomplete", "response.failed" }) |terminal| try std.testing.expect(std.mem.indexOf(u8, f.wire.items, terminal) == null);
        try std.testing.expect(f.done);
    }
    std.debug.print("qualification_af802d02 {s} {s} {s} {} clock_reads={d} random_calls={d}\n", .{ label, path, @tagName(mode), stream, clock_reads, random_calls });
    std.debug.print("qualification_b8a152e4 {s} {s} {s} {} {x}\n", .{ label, path, @tagName(mode), stream, f.wire.items });
    return a.dupe(u8, f.wire.items);
}
fn withoutComments(bytes: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty; errdefer out.deinit(a);
    var it = std.mem.splitSequence(u8, bytes, ": keepalive\n\n");
    while (it.next()) |piece| try out.appendSlice(a, piece);
    return out.toOwnedSlice(a);
}
test "qualification_4533b989 actual four routes aliases fixed bytes stream nonstream negatives and terminals" {
    for (paths) |path| for ([_]bool{ false, true }) |stream| for ([_]Mode{ .success, .busy, .failed, .cancelled, .closed_open, .finished_backlog, .context, .unsupported, .invalid_body }) |mode| {
        const base = try run(path, mode, stream, 0, "unset"); defer a.free(base);
        if (comptime candidate) {
            const zero = try run(path, mode, stream, 0, "zero"); defer a.free(zero); try std.testing.expectEqualStrings(base, zero);
            const on = try run(path, mode, stream, std.time.ns_per_s, "on"); defer a.free(on);
            const stripped = try withoutComments(on); defer a.free(stripped); try std.testing.expectEqualStrings(base, stripped);
        }
    };
}
test "qualification_4533b989 actual all routes truncated comment cancels drains no terminal" {
    broken_prefix = 3;
    if (comptime candidate) for (paths) |path| { const bytes = try run(path, .broken_comment, true, std.time.ns_per_s, "broken"); a.free(bytes); };
}
test "qualification_4533b989 CPU actual four routes aliases every truncated comment prefix cancels and drains" {
    defer broken_prefix = 3;
    if (comptime candidate) for (0..13) |boundary| {
        broken_prefix = boundary;
        var buf: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&buf, "broken{d}", .{boundary});
        for (paths) |path| { const bytes = try run(path, .broken_comment, true, std.time.ns_per_s, label); a.free(bytes); }
    };
}
