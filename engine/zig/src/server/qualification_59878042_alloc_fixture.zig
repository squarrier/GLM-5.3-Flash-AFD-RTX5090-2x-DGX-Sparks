//! N09 fixture-only logical allocators and authentic owner/callsite drives.
//! GPT-6.1 Sol (OpenAI Codex), via Hermes; inherited credits unchanged.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server.zig");
const model = @import("model_text.zig");
const json = @import("json.zig");
const http = @import("http_conn.zig");
const openai = @import("openai.zig");
const chat = @import("chat.zig");
const routes = @import("routes.zig");
const log = @import("log.zig");
const Observe = @import("sse_test_observe.zig").Test;
const a = std.testing.allocator;
const Allocator = std.mem.Allocator;
pub const Stage = enum { a1, a2, a3, a4, a5, g1 };
pub const Kind = enum { plain, reasoning, tools };
pub const Fault = enum { none, role, startup, content, final, usage };
pub const paths = [_][]const u8{ "/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/v1/messages", "/messages", "/v1/responses", "/responses" };
const Operation = enum { alloc, resize, remap };
const Attempt = struct {
    stage: Stage, operation: Operation, bytes: usize, ret_addr: usize,
    address: usize = 0, fired: bool = false, success: bool = false,
    site: []const u8,
    frames: [16]usize = undefined,
    nframes: usize = 0,
};
threadlocal var emitted_stacks: [16384]u64 = undefined;
threadlocal var emitted_count: usize = 0;

const Block = struct { address: usize, bytes: usize, live: bool = true };
const Free = struct { address: usize, bytes: usize };

/// A delegates to the request arena *per logical vtable operation*. G delegates
/// to the ordinary allocator used solely for the isolated native Mailbox.
/// All ledgers have fixture-owned fixed storage, outside either failure scope.
const Trace = struct {
    backing: Allocator,
    stage: Stage,
    enabled: bool = false,
    strict: bool = false,
    fail_at: ?usize = null,
    fired: ?usize = null,
    attempts: [4096]Attempt = undefined,
    n: usize = 0,
    blocks: [1024]Block = undefined,
    nb: usize = 0,
    freed: [1024]Free = undefined,
    nf: usize = 0,
    live: usize = 0,
    allocated_bytes: usize = 0,
    freed_bytes: usize = 0,
    fn allocator(t: *Trace) Allocator { return .{ .ptr = t, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } }; }
    fn self(ctx: *anyopaque) *Trace { return @ptrCast(@alignCast(ctx)); }
    fn owner(stage: Stage) []const u8 { return switch (stage) {
        .a1 => "openai.Run.stream:roleDelta/chat.deltaOf/Run.chunk",
        .a2 => "chat.Generation.commit:collected/fresh/Incremental/Shown/splitThinking/hideInto/Streamer",
        .a3 => "chat.Generation.finish:decode/closeCall/thinking/runtime/spec/telemetry",
        .a4 => "openai.Run.stream:stopped-tail/chunk/extras/usage",
        .a5 => "native.Translator.Wire.open/event:Reply/sse.stringify/frame",
        .g1 => "chat.Mailbox.onEvent",
    }; }
    fn begin(t: *Trace, op: Operation, bytes: usize, ret_addr: usize) ?usize {
        if (!t.enabled) return null;
        std.debug.assert(t.n < t.attempts.len);
        const i = t.n; t.n += 1;
        const should_fail = t.fail_at != null and t.fail_at.? == i;
        t.attempts[i] = .{ .stage = t.stage, .operation = op, .bytes = bytes, .ret_addr = ret_addr, .fired = should_fail, .site = owner(t.stage) };
        if (t.fail_at == null) t.attempts[i].nframes = std.debug.captureCurrentStackTrace(.{}, &t.attempts[i].frames).return_addresses.len;
        if (should_fail) { std.debug.assert(t.fired == null); t.fired = i; }
        return i;
    }
    fn fail(t: *Trace, i: ?usize) bool { return if (i) |index| t.attempts[index].fired else false; }
    fn add(t: *Trace, address: usize, bytes: usize) void {
        if (!t.strict or !t.enabled) return;
        std.debug.assert(t.nb < t.blocks.len);
        for (t.blocks[0..t.nb]) |b| std.debug.assert(!b.live or b.address != address);
        t.blocks[t.nb] = .{ .address = address, .bytes = bytes }; t.nb += 1; t.live += 1; t.allocated_bytes += bytes;
    }
    fn block(t: *Trace, address: usize) *Block {
        for (t.blocks[0..t.nb]) |*b| if (b.live and b.address == address) return b;
        @panic("free/remap after borrow or duplicate/unowned free");
    }
    fn resized(t: *Trace, old: usize, address: usize, bytes: usize) void {
        if (!t.strict or !t.enabled) return;
        const b = t.block(old);
        t.allocated_bytes = t.allocated_bytes - b.bytes + bytes;
        b.address = address; b.bytes = bytes;
    }
    fn alloc(ctx: *anyopaque, bytes: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const t = self(ctx); const i = t.begin(.alloc, bytes, ret_addr);
        if (t.fail(i)) return null;
        const p = t.backing.rawAlloc(bytes, alignment, ret_addr) orelse return null;
        if (i) |index| { t.attempts[index].address = @intFromPtr(p); t.attempts[index].success = true; }
        t.add(@intFromPtr(p), bytes); return p;
    }
    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, bytes: usize, ret_addr: usize) bool {
        const t = self(ctx); const i = t.begin(.resize, bytes, ret_addr);
        if (t.fail(i)) return false;
        const ok = t.backing.rawResize(memory, alignment, bytes, ret_addr);
        if (i) |index| { t.attempts[index].address = @intFromPtr(memory.ptr); t.attempts[index].success = ok; }
        if (ok) t.resized(@intFromPtr(memory.ptr), @intFromPtr(memory.ptr), bytes);
        return ok;
    }
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, bytes: usize, ret_addr: usize) ?[*]u8 {
        const t = self(ctx); const i = t.begin(.remap, bytes, ret_addr);
        if (t.fail(i)) return null;
        const p = t.backing.rawRemap(memory, alignment, bytes, ret_addr) orelse return null;
        if (i) |index| { t.attempts[index].address = @intFromPtr(p); t.attempts[index].success = true; }
        t.resized(@intFromPtr(memory.ptr), @intFromPtr(p), bytes); return p;
    }
    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const t = self(ctx);
        if (t.strict and t.enabled) {
            const b = t.block(@intFromPtr(memory.ptr)); std.debug.assert(b.bytes == memory.len);
            b.live = false; t.live -= 1;
            std.debug.assert(t.nf < t.freed.len);
            t.freed[t.nf] = .{ .address = @intFromPtr(memory.ptr), .bytes = memory.len }; t.nf += 1; t.freed_bytes += memory.len;
        }
        t.backing.rawFree(memory, alignment, ret_addr);
    }
    fn inventory(t: *Trace, label: []const u8) void {
        for (t.attempts[0..t.n], 0..) |*attempt, i| {
            const key = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(attempt.frames[0..attempt.nframes]));
            if (std.mem.indexOfScalar(u64, emitted_stacks[0..emitted_count], key) == null) {
                std.debug.assert(emitted_count < emitted_stacks.len);
                emitted_stacks[emitted_count] = key; emitted_count += 1;
                std.debug.print("qualification_6c18cccc key={x}\n", .{key});
                std.debug.dumpStackTrace(&.{ .return_addresses = attempt.frames[0..attempt.nframes], .skipped = .unknown });
                std.debug.print("qualification_bed773d4 key={x}\n", .{key});
            }
            std.debug.print("qualification_fb21131a scenario={s} stage={s} index={d} op={s} bytes={d} ret_addr={x} success={} fired={} site={s}\n", .{ label, @tagName(attempt.stage), i, @tagName(attempt.operation), attempt.bytes, attempt.ret_addr, attempt.success, attempt.fired, attempt.site });
            std.debug.print("qualification_7d22183e scenario={s} stage={s} index={d} key={x} ret_addr={x} frames=", .{label, @tagName(attempt.stage), i, key, attempt.ret_addr});
            for (attempt.frames[0..attempt.nframes]) |frame| std.debug.print("{x},", .{frame});
            std.debug.print("\n", .{});
        }
    }
};

pub const Result = struct { attempts: usize, fired: ?usize, generated: bool = false, stopped: bool = false, live: usize = 0, frees: usize = 0, signatures: [4096]u64 = undefined };
fn result(t: *const Trace) Result {
    var r: Result = .{ .attempts = t.n, .fired = t.fired, .live = t.live, .frees = t.nf };
    for (t.attempts[0..t.n], 0..) |attempt, i| {
        var hash = std.hash.Wyhash.init(0);
        const op: u8 = @intFromEnum(attempt.operation);
        hash.update(std.mem.asBytes(&op));
        hash.update(std.mem.asBytes(&attempt.bytes));
        hash.update(std.mem.asBytes(&attempt.ret_addr));
        r.signatures[i] = hash.final();
    }
    return r;
}
fn samePrefix(base: *const Result, replay: *const Result, count: usize) !void {
    try std.testing.expect(base.attempts >= count and replay.attempts >= count);
    try std.testing.expectEqualSlices(u64, base.signatures[0..count], replay.signatures[0..count]);
}
pub const Counts = struct {
    baselines: usize = 0, triggered: usize = 0, sentinels: usize = 0, zero: usize = 0,
    pub fn print(c: Counts, label: []const u8) void { std.debug.print("qualification_29a263d8 family={s} baselines={d} triggered={d} sentinels={d} zero={d}\n", .{ label, c.baselines, c.triggered, c.sentinels, c.zero }); }
};

threadlocal var time_ns: i96 = 0;
fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp { return .{ .nanoseconds = time_ns }; }
fn random(_: ?*anyopaque, bytes: []u8) void { @memset(bytes, 0x12); }
fn wait(ctx: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
    if (timeout == .deadline) { time_ns = @max(time_ns, timeout.deadline.raw.toNanoseconds()); return; }
    return std.testing.io.vtable.futexWait(ctx, ptr, expected, timeout);
}
const Text = struct {
    fn text(t: *Text) model.Text { return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } }; }
    fn encode(_: *anyopaque, g: Allocator, bytes: []const u8, _: bool) model.Error![]u32 { const ids = try g.alloc(u32, bytes.len); for (bytes, ids) |b, *id| id.* = b; return ids; }
    fn decode(_: *anyopaque, g: Allocator, ids: []const u32) Allocator.Error![]u8 { const bytes = try g.alloc(u8, ids.len); for (ids, bytes) |id, *b| b.* = @intCast(id); return bytes; }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 { return null; }
    fn tokenString(_: *anyopaque, g: Allocator, id: u32) Allocator.Error![]u8 { return std.fmt.allocPrint(g, "{d}", .{id}); }
    fn vocab(_: *anyopaque) u32 { return 256; }
    fn eos(_: *anyopaque) []const u32 { return &.{0}; }
    fn render(_: *anyopaque, g: Allocator, _: json.Value, _: model.RenderOptions, _: *[]const u8) model.Error![]u8 { return g.dupe(u8, "prompt"); }
    fn template(_: *anyopaque) []const u8 { return ""; }
};

const Driver = struct {
    wire: std.ArrayList(u8) = .empty,
    path: []const u8 = "/v1/chat/completions",
    fault: Fault = .none,
    failed_writes: usize = 0,
    writes: usize = 0,
    comments: usize = 0,
    submitted: usize = 0,
    cancelled: usize = 0,
    finished: bool = false,
    sink: ?api.Sink = null,
    request: ?*const api.Request = null,
    id: api.Id = 0,
    turns: usize = 0,
    trace: ?*Trace = null,
    tags: [2048]std.meta.Tag(Observe.Event) = undefined,
    n: usize = 0,
    starts: usize = 0,
    stops: usize = 0,
    generated: bool = false,
    stop_at: ?usize = null,
    box_begin_at: ?usize = null,
    box_end_at: ?usize = null,
    request_end_at: ?usize = null,
    generated_at: ?usize = null,
    run_exit_at: ?usize = null,
    enabled: bool = false,
    native_campaign: bool = false,
    terminal: api.Reason = .stop,
    prose: []const u8 = "hello",
    eos_token: bool = false,
    gtrace: ?*Trace = null,
    refusal: ?api.SubmitError = null,
    fn self(ctx: *anyopaque) *Driver { return @ptrCast(@alignCast(ctx)); }
    fn record(ctx: *anyopaque, event: Observe.Event) void {
        const d = self(ctx); std.debug.assert(d.n < d.tags.len);
        const index = d.n; d.tags[d.n] = std.meta.activeTag(event); d.n += 1;
        switch (event) {
            .run_open => if (d.trace) |t| { if (t.stage == .a1) t.enabled = true; if (t.stage == .a5) t.enabled = false; },
            .translated => |e| if (d.trace) |t| { if (t.stage == .a5 and d.native_campaign) t.enabled = e.phase == .enter; },
            .generate_enter => {
                if (d.trace) |t| {
                    if (t.stage == .a1) t.enabled = false;
                    if (d.native_campaign and t.stage == .a2) t.enabled = true;
                }
                if (d.gtrace) |t| t.enabled = true;
            },
            .generated => { d.generated = true; d.generated_at = index; if (d.trace) |t| { if (d.native_campaign and (t.stage == .a2 or t.stage == .a3)) t.enabled = false; if (t.stage == .a4 and !d.enabled) t.enabled = true; } },
            .session => |s| switch (s.phase) {
                .start => { std.debug.assert(d.enabled and s.active and s.attached); d.starts += 1; },
                .stop => { std.debug.assert(!s.active and !s.attached); d.stops += 1; d.stop_at = index; if (d.trace) |t| { if (t.stage == .a4) t.enabled = true; } },
                else => {},
            },
            .box_deinit => |b| {
                std.debug.assert((b.finished or d.refusal != null) and d.request == null and d.sink == null);
                if (b.phase == .begin) d.box_begin_at = index else {
                    d.box_end_at = index;
                    if (d.gtrace) |t| { std.debug.assert(t.live == 0 and t.allocated_bytes == t.freed_bytes); t.enabled = false; }
                }
            },
            .request_end => d.request_end_at = index,
            .run_exit => { d.run_exit_at = index; if (d.trace) |t| t.enabled = false; },
            .drain_end => |e| std.debug.assert(e.finished),
            else => {},
        }
    }
    fn engine(d: *Driver) api.Engine { return .{ .ctx = d, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } }; }
    fn info(_: *anyopaque) api.Info { return .{ .context_window = 4096 }; }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void { out.* = .{}; }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const d = self(ctx); std.debug.assert(d.request == null);
        if (d.refusal) |e| return e;
        d.submitted += 1; d.request = request; d.sink = sink; d.id = id;
    }
    fn finish(d: *Driver, reason: api.Reason) void {
        if (d.finished) return;
        std.debug.assert(d.request != null and d.request.?.prompt.len > 0);
        const s = d.sink.?; d.request = null; d.sink = null; d.finished = true;
        if (d.native_campaign) if (d.trace) |t| if (t.stage == .a3) { t.enabled = true; };
        const stats: api.Stats = if (d.native_campaign) .{ .prefill_widths = &.{ 2, 3 }, .prefill_raised = &.{ true, false }, .telemetry_json = "{\"rounds\":1}", .prefill_seconds = 0.1, .loop_period = 1, .rounds = 2, .drafted = 3, .accepted = 2 } else .{};
        s.event(s.ctx, d.id, &.{ .finished = .{ .reason = reason, .stats = stats, .message = if (d.native_campaign and reason == .failed) "native fixture failure" else "" } });
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void { const d = self(ctx); d.cancelled += 1; if (d.request != null) d.finish(.cancelled); }
    fn gone(ctx: *anyopaque) bool {
        const d = self(ctx); if (d.request == null) return false;
        std.debug.assert(d.turns < 12); d.turns += 1;
        time_ns += 100 * std.time.ns_per_ms;
        if (d.turns == 1) {
            const s = d.sink.?;
            if (d.native_campaign) s.event(s.ctx, d.id, &.{ .prefilled = 1 });
            var tokens: [256]u32 = undefined;
            for (d.prose, tokens[0..d.prose.len]) |b, *id| id.* = b;
            s.event(s.ctx, d.id, &.{ .tokens = tokens[0..d.prose.len] });
            if (d.eos_token) s.event(s.ctx, d.id, &.{ .tokens = &.{0} });
        }
        if (d.turns == 3) d.finish(d.terminal);
        return false;
    }
    fn translated(d: *Driver) bool { return std.mem.endsWith(u8, d.path, "/messages") or std.mem.endsWith(u8, d.path, "/responses"); }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const d = self(ctx); d.writes += 1;
        const header = std.mem.startsWith(u8, bytes, "HTTP/");
        const role = std.mem.startsWith(u8, bytes, "data: ") and std.mem.indexOf(u8, bytes, "\"role\": \"assistant\"") != null and std.mem.indexOf(u8, bytes, "\"delta\"") != null;
        const startup = std.mem.startsWith(u8, bytes, "event: message_start\n") or std.mem.startsWith(u8, bytes, "event: response.created\n");
        const terminal = std.mem.indexOf(u8, bytes, "\"finish_reason\": \"stop\"") != null or std.mem.indexOf(u8, bytes, "\"finish_reason\": \"length\"") != null or std.mem.startsWith(u8, bytes, "event: message_stop\n") or std.mem.startsWith(u8, bytes, "event: response.completed\n");
        const hit = switch (d.fault) {
            .none => false,
            .role => role,
            .startup => startup,
            .content => d.submitted > 0 and !header and !terminal,
            .final => terminal,
            .usage => std.mem.indexOf(u8, bytes, "\"usage\"") != null and (std.mem.indexOf(u8, bytes, "\"choices\": []") != null or std.mem.startsWith(u8, bytes, "event: message_delta\n") or std.mem.startsWith(u8, bytes, "event: response.completed\n")),
        };
        if (hit) { d.failed_writes += 1; return error.Closed; }
        d.wire.appendSlice(a, bytes) catch return error.Closed;
        if (header and d.native_campaign) if (d.trace) |t| if (t.stage == .a5) { t.enabled = true; };
    }
    fn commentNow(_: ?*anyopaque) i96 { return time_ns; }
    fn poll(_: ?*anyopaque, fd: std.posix.socket_t, ms: i32) http.PollResult { std.debug.assert(fd == -1 and ms > 0 and ms <= 50); return .out; }
    fn send(ctx: ?*anyopaque, fd: std.posix.socket_t, bytes: []const u8) http.SendResult { std.debug.assert(fd == -1); const d = self(ctx.?); d.comments += 1; d.wire.appendSlice(a, bytes) catch @panic("ledger allocation"); return .{ .sent = bytes.len }; }
    fn conn(d: *Driver, g: Allocator) http.Conn { return .{ .fd = -1, .peer = "allocator-fixture", .gpa = g, .buf = &.{}, .path = d.path, .version = "HTTP/1.1", .quiet_log = true, .test_transport = .{ .ctx = d, .write_all = write, .peer_gone = gone, .comment_ops = .{ .ctx = d, .now = commentNow, .poll = poll, .send = send } } }; }
    fn verify(d: *Driver, srv: *server.Server, connection: *http.Conn) !void {
        try std.testing.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
        try std.testing.expectEqual(d.starts, d.stops);
        try std.testing.expect(connection.stream_activity == null and d.request == null and d.sink == null);
        try std.testing.expect(d.run_exit_at != null);
        if (d.submitted > 0) {
            try std.testing.expect(d.finished and d.generated and d.box_begin_at != null and d.box_end_at != null);
            try std.testing.expect(d.box_begin_at.? < d.box_end_at.? and d.box_end_at.? < d.request_end_at.? and d.request_end_at.? < d.generated_at.?);
            if (d.stop_at) |stop| try std.testing.expect(d.generated_at.? < stop and stop < d.run_exit_at.?);
        } else if (d.refusal != null and d.generated) {
            try std.testing.expect(d.box_begin_at != null and d.box_end_at != null and d.request_end_at != null and d.generated);
            try std.testing.expect(d.box_end_at.? < d.request_end_at.? and d.request_end_at.? < d.generated_at.?);
            try std.testing.expectEqual(@as(usize, 0), d.cancelled);
        } else try std.testing.expect(d.box_begin_at == null and d.request_end_at == null and !d.generated);
    }
};
fn ioFor(vt: *std.Io.VTable) std.Io { vt.* = std.testing.io.vtable.*; vt.now = now; vt.random = random; vt.futexWait = wait; return .{ .userdata = std.testing.io.userdata, .vtable = vt }; }
fn config(period: i64) server.Config { return .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false, .keep_warm_s = 0, .sse_keepalive_ns = period }; }
fn parsed(g: Allocator, bytes: []const u8) !json.Value { return (try json.parse(g, bytes)).ok; }

pub fn runInventory(is_chat: bool, usage: bool, period: i64, stage: Stage, fail_at: ?usize) !Result {
    time_ns = 0;
    var vt: std.Io.VTable = undefined; const io = ioFor(&vt);
    log.init(io, true); defer log.init(std.testing.io, true);
    var d: Driver = .{ .enabled = period > 0 }; defer d.wire.deinit(a);
    var text: Text = .{};
    const srv = try server.Server.init(a, io, d.engine(), text.text(), config(period), null); defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer { arena.deinit(); std.debug.print("qualification_d1381028 boundary=arena_deinit native=true\n", .{}); }
    const g = arena.allocator();
    const bootstrap_reserve = try g.alloc(u8, 1024 * 1024); g.free(bootstrap_reserve);
    var trace: Trace = .{ .backing = g, .stage = stage, .fail_at = fail_at }; d.trace = &trace;
    var hook: Observe.Hook = .{ .ctx = &d, .record = Driver.record }; const saved = Observe.active; Observe.active = &hook; defer Observe.active = saved;
    var conn = d.conn(g); defer { conn.deinit(); std.debug.print("qualification_d1381028 boundary=conn_deinit native=true\n", .{}); }
    const raw = try parsed(g, if (is_chat) (if (usage) "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":8,\"stream_options\":{\"include_usage\":true}}" else "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":8}") else (if (usage) "{\"prompt\":\"x\",\"stream\":true,\"max_tokens\":8,\"stream_options\":{\"include_usage\":true}}" else "{\"prompt\":\"x\",\"stream\":true,\"max_tokens\":8}"));
    try openai.TestFixture.direct(srv, g, trace.allocator(), &conn, is_chat, raw);
    try d.verify(srv, &conn);
    if (fail_at == null) trace.inventory(if (is_chat) (if (usage) "chat-separate" else "chat-attached") else (if (usage) "completion-separate" else "completion-attached"));
    std.debug.print("qualification_a797d8f5 family=run stage={s} chat={} usage={} period={d} fail_index={?d} attempts={d} fired={?d} submit={d} start={d} stop={d} generated={} arena_deinit=after-return\n", .{ @tagName(stage), is_chat, usage, period, fail_at, trace.n, trace.fired, d.submitted, d.starts, d.stops, d.generated });
    var r = result(&trace); r.generated = d.generated; r.stopped = d.starts == d.stops; return r;
}
pub fn sweepRun(is_chat: bool, usage: bool, period: i64, stage: Stage, counts: *Counts) !void {
    const base = try runInventory(is_chat, usage, period, stage, null); counts.baselines += 1;
    if (base.attempts == 0) counts.zero += 1;
    for (0..base.attempts) |i| { const r = try runInventory(is_chat, usage, period, stage, i); try std.testing.expectEqual(@as(?usize, i), r.fired); try samePrefix(&base, &r, i + 1); counts.triggered += 1; }
    const sentinel = try runInventory(is_chat, usage, period, stage, base.attempts);
    try std.testing.expect(sentinel.fired == null and sentinel.generated and sentinel.attempts == base.attempts); try samePrefix(&base, &sentinel, base.attempts); counts.sentinels += 1;
}

const GenDrive = struct {
    trace: *Trace,
    selected: Stage,
    calls: usize = 0,
    fn checkpoint(ctx: *anyopaque, phase: chat.TestFixture.Phase) void {
        const d: *GenDrive = @ptrCast(@alignCast(ctx));
        d.trace.enabled = (phase == .commit and d.selected == .a2) or (phase == .finish and d.selected == .a3);
    }
    fn call(ctx: *anyopaque, _: json.Value) error{Closed}!void { const d: *GenDrive = @ptrCast(@alignCast(ctx)); d.calls += 1; }
};
pub fn generationInventory(kind: Kind, stage: Stage, fail_at: ?usize) !Result {
    time_ns = 0;
    var vt: std.Io.VTable = undefined; const io = ioFor(&vt);
    log.init(io, true); defer log.init(std.testing.io, true);
    var engine: Driver = .{}; defer engine.wire.deinit(a);
    var text: Text = .{};
    const srv = try server.Server.init(a, io, engine.engine(), text.text(), config(0), null); defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer { arena.deinit(); std.debug.print("qualification_d1381028 boundary=arena_deinit native=true\n", .{}); }
    const g = arena.allocator();
    const bootstrap_reserve = try g.alloc(u8, 1024 * 1024); g.free(bootstrap_reserve);
    var trace: Trace = .{ .backing = g, .stage = stage, .fail_at = fail_at };
    var d: GenDrive = .{ .trace = &trace, .selected = stage };
    const prose = switch (kind) { .plain => "hello", .reasoning => "<think>reason</think>answer", .tools => "prose <tool_call>\n<function=f>\n<parameter=q>\nv\n</parameter>\n</function>\n</tool_call>" };
    const tokens = try g.alloc(u32, prose.len); for (prose, tokens) |b, *id| id.* = b;
    const tools = if (kind == .tools) (try parsed(g, "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}}]")).array else &.{};
    const saved = Observe.active; Observe.active = null; defer Observe.active = saved;
    var escaped = false;
    if (chat.TestFixture.generation(srv, trace.allocator(), .{ .ctx = &d, .call = GenDrive.call }, tokens, kind == .reasoning, tools, &d, GenDrive.checkpoint)) |_| {} else |e| { try std.testing.expect(e == error.OutOfMemory); escaped = true; }
    if (fail_at == null) trace.inventory(@tagName(kind));
    std.debug.print("qualification_a797d8f5 family=generation kind={s} stage={s} fail_index={?d} attempts={d} fired={?d} escaped={} sink_calls={d} engine_fixture=already-finished arena_deinit=after-return\n", .{ @tagName(kind), @tagName(stage), fail_at, trace.n, trace.fired, escaped, d.calls });
    return result(&trace);
}
pub fn sweepGeneration(kind: Kind, stage: Stage, counts: *Counts) !void {
    const base = try generationInventory(kind, stage, null); counts.baselines += 1;
    try std.testing.expect(base.attempts > 0);
    for (0..base.attempts) |i| { const r = try generationInventory(kind, stage, i); try std.testing.expectEqual(@as(?usize, i), r.fired); try samePrefix(&base, &r, i + 1); counts.triggered += 1; }
    const sentinel = try generationInventory(kind, stage, base.attempts);
    try std.testing.expect(sentinel.fired == null and sentinel.attempts == base.attempts); try samePrefix(&base, &sentinel, base.attempts); counts.sentinels += 1;
}

const BoxDrive = struct {
    trace: *Trace,
    expected: [6]Free = undefined,
    n: usize = 0,
    begin: bool = false,
    end: bool = false,
    prior_frees: usize = 0,
    fn snapshot(ctx: *anyopaque, buffers: []const chat.TestFixture.Buffer) void {
        const d: *BoxDrive = @ptrCast(@alignCast(ctx));
        std.debug.assert(buffers.len == 2 or buffers.len == 6);
        d.n = 0; d.prior_frees = d.trace.nf;
        for (buffers) |b| if (b.bytes > 0) {
            d.expected[d.n] = .{ .address = b.address, .bytes = b.bytes }; d.n += 1;
            for (d.trace.attempts[0..d.trace.n]) |*attempt| if (attempt.address == b.address and attempt.success and std.mem.eql(u8, attempt.site, Trace.owner(.g1))) { attempt.site = b.site; };
        };
    }
    fn record(ctx: *anyopaque, event: Observe.Event) void {
        const d: *BoxDrive = @ptrCast(@alignCast(ctx));
        if (event == .box_deinit) {
            std.debug.assert(event.box_deinit.finished);
            if (event.box_deinit.phase == .begin) { std.debug.assert(!d.begin and !d.end and d.trace.nf == d.prior_frees); d.begin = true; }
            else { std.debug.assert(d.begin and !d.end and d.trace.live == 0); d.end = true; }
        }
    }
};
pub fn mailboxInventory(fail_at: ?usize) !Result {
    return mailboxInventoryRounds(fail_at, 1);
}
fn mailboxInventoryRounds(fail_at: ?usize, rounds: usize) !Result {
    var vt: std.Io.VTable = undefined; const io = ioFor(&vt);
    var engine: Driver = .{}; defer engine.wire.deinit(a);
    var text: Text = .{};
    const srv = try server.Server.init(a, io, engine.engine(), text.text(), config(0), null); defer srv.deinit();
    var trace: Trace = .{ .backing = a, .stage = .g1, .enabled = true, .strict = true, .fail_at = fail_at };
    var d: BoxDrive = .{ .trace = &trace };
    // The ordinary server backing allocator is temporarily isolated to Mailbox only.
    const saved_gpa = srv.gpa; srv.gpa = trace.allocator(); defer srv.gpa = saved_gpa;
    var hook: Observe.Hook = .{ .ctx = &d, .record = BoxDrive.record }; const saved = Observe.active; Observe.active = &hook; defer Observe.active = saved;
    chat.TestFixture.mailbox(srv, rounds, &d, BoxDrive.snapshot);
    try std.testing.expect(d.begin and d.end and trace.live == 0 and trace.nf - d.prior_frees == d.n and trace.allocated_bytes == trace.freed_bytes);
    for (trace.freed[d.prior_frees..trace.nf], d.expected[0..d.n], 0..) |freed, expected, i| {
        try std.testing.expectEqual(expected.address, freed.address); try std.testing.expectEqual(expected.bytes, freed.bytes);
        std.debug.print("qualification_971d955d stage=g1 rounds={d} fail_index={?d} teardown_order={d} address={x} bytes={d}\n", .{rounds, fail_at, i, freed.address, freed.bytes});
    }
    if (fail_at == null) trace.inventory("six-nonempty-native-mailbox-copies");
    std.debug.print("qualification_a797d8f5 family=mailbox rounds={d} fail_index={?d} attempts={d} fired={?d} frees={d} interim_frees={d} teardown_frees={d} live={d} bytes_allocated={d} bytes_freed={d} native_finished=true\n", .{rounds, fail_at, trace.n, trace.fired, trace.nf, d.prior_frees, d.n, trace.live, trace.allocated_bytes, trace.freed_bytes });
    return result(&trace);
}
pub fn sweepMailbox(counts: *Counts) !void {
    for ([_]usize{ 1, 40 }) |rounds| {
        const base = try mailboxInventoryRounds(null, rounds); counts.baselines += 1;
        for (0..base.attempts) |i| { const r = try mailboxInventoryRounds(i, rounds); try std.testing.expectEqual(@as(?usize, i), r.fired); try samePrefix(&base, &r, i + 1); counts.triggered += 1; }
        const sentinel = try mailboxInventoryRounds(base.attempts, rounds);
        try std.testing.expect(sentinel.fired == null and sentinel.attempts == base.attempts and sentinel.frees >= 6); try samePrefix(&base, &sentinel, base.attempts); counts.sentinels += 1;
    }
}

pub fn faultCell(path: []const u8, period: i64, setting: usize, fault: Fault, tools: bool) !void {
    return faultCase(path, period, setting, fault, tools, false);
}
pub fn faultUsageCell(path: []const u8, period: i64, setting: usize, fault: Fault, tools: bool) !void {
    return faultCase(path, period, setting, fault, tools, true);
}
fn faultCase(path: []const u8, period: i64, setting: usize, fault: Fault, tools: bool, usage: bool) !void {
    time_ns = 0;
    var vt: std.Io.VTable = undefined; const io = ioFor(&vt);
    log.init(io, true); defer log.init(std.testing.io, true);
    var d: Driver = .{ .path = path, .fault = fault, .enabled = period > 0 }; defer d.wire.deinit(a);
    var text: Text = .{};
    const srv = try server.Server.init(a, io, d.engine(), text.text(), config(period), null); defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer { arena.deinit(); std.debug.print("qualification_d1381028 boundary=arena_deinit native=true\n", .{}); }
    const g = arena.allocator();
    const bootstrap_reserve = try g.alloc(u8, 1024 * 1024); g.free(bootstrap_reserve);
    const messages_route = std.mem.endsWith(u8, path, "/messages");
    const responses_route = std.mem.endsWith(u8, path, "/responses");
    const is_chat = std.mem.endsWith(u8, path, "/chat/completions");
    const tool = if (!tools) "" else if (messages_route) ",\"tools\":[{\"name\":\"f\",\"input_schema\":{\"type\":\"object\"}}]" else if (responses_route) ",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]" else ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]";
    const body0 = if (responses_route) try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"input\":\"x\",\"stream\":true,\"store\":false,\"max_output_tokens\":8{s}}}", .{tool}) else if (messages_route or is_chat) try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":true,\"max_tokens\":8{s}}}", .{tool}) else try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"prompt\":\"x\",\"stream\":true,\"max_tokens\":8{s}}}", .{tool});
    const body = if (usage and !d.translated()) try std.fmt.allocPrint(g, "{s},\"stream_options\":{{\"include_usage\":true}}}}", .{body0[0..body0.len-1]}) else body0;
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(g, "{d}", .{body.len}) }};
    var conn = d.conn(g); conn.buf = try g.dupe(u8, body); conn.end = body.len; conn.method = "POST"; conn.headers = &headers;
    defer { conn.deinit(); std.debug.print("qualification_d1381028 boundary=conn_deinit native=true\n", .{}); }
    var hook: Observe.Hook = .{ .ctx = &d, .record = Driver.record }; const saved = Observe.active; Observe.active = &hook; defer Observe.active = saved;
    routes.dispatch(srv, &conn, g);
    try d.verify(srv, &conn);
    const applicable = switch (fault) { .none => false, .role => is_chat, .startup => d.translated(), .content, .final => true, .usage => usage };
    try std.testing.expectEqual(@as(usize, if (applicable) 1 else 0), d.failed_writes);
    const pre_submit = (fault == .role and is_chat and !tools) or (fault == .startup and d.translated());
    try std.testing.expectEqual(@as(usize, if (pre_submit) 0 else 1), d.submitted);
    try std.testing.expectEqual(@as(usize, if (period > 0 and !(fault == .startup and d.translated())) 1 else 0), d.starts);
    const cancelled = fault == .content or (fault == .role and is_chat and tools);
    try std.testing.expectEqual(@as(usize, if (cancelled) 1 else 0), d.cancelled);
    if (!applicable) try std.testing.expect(d.generated and d.finished and d.failed_writes == 0);
    if (fault == .role and !is_chat) try std.testing.expect(d.generated and d.submitted == 1);
    std.debug.print("qualification_69dfece8 path={s} setting={s} fault={s} tools={} usage={} usage_route={s} applicability={s} failed_writes={d} submit={d} cancel={d} start={d} stop={d} box_end={?d} request_end={?d} generated={?d} stopped_at={?d} run_exit={?d} conn_arena_deinit=after-return\n", .{path, if (setting == 0) "UNSET" else if (setting == 1) "ZERO" else "ENABLED", @tagName(fault), tools, usage, if (d.translated()) "fixed-translator-terminal" else if (usage) "separate-empty-choices" else "attached", if (applicable) "TRIGGERED" else if (fault == .role and d.translated()) "SWALLOWED-NOT-TRIGGERED" else "NOT-APPLICABLE", d.failed_writes, d.submitted, d.cancelled, d.starts, d.stops, d.box_end_at, d.request_end_at, d.generated_at, d.stop_at, d.run_exit_at});
}

pub fn wrapperControl() !void {
    var arena = std.heap.ArenaAllocator.init(a); defer arena.deinit();
    var t: Trace = .{ .backing = arena.allocator(), .stage = .a2, .enabled = true };
    const g = t.allocator();
    const empty = try g.alloc(u8, 0); g.free(empty); try std.testing.expectEqual(@as(usize, 0), t.n);
    var memory = try g.alloc(u8, 16);
    if (g.rawResize(memory, .of(u8), 8, @returnAddress())) memory = memory.ptr[0..8];
    if (g.rawRemap(memory, .of(u8), 4, @returnAddress())) |p| memory = p[0..4];
    g.free(memory);
    try std.testing.expectEqual(@as(usize, 3), t.n);
    try std.testing.expect(t.attempts[0].operation == .alloc and t.attempts[1].operation == .resize and t.attempts[2].operation == .remap);
    t.inventory("allocator-vtable-positive-control");
}

pub const Translator = enum { messages, responses };
pub const TranslatedKind = enum { startup, role, empty_choices, empty_content, content, reasoning, tools, terminal, failed, incomplete, pending };
const Translation = struct {
    trace: *Trace,
    driver: *Driver,
    conn: *http.Conn,
    srv: *server.Server,
    kind: TranslatedKind,
    payload: json.Value,
    content: json.Value,
    terminal: json.Value,
    period: i64,
    entered: bool = false,
    escaped: bool = false,
    fn drive(out: openai.Out) anyerror!void {
        const t = active_translation.?;
        t.entered = true;
        if (t.kind == .startup) { t.trace.enabled = false; return; }
        var session: @import("sse_keepalive.zig").Session = .{ .conn = t.conn, .io = t.srv.io, .period_ns = t.period };
        if (t.period > 0) session.start();
        defer session.stop();
        if (t.kind == .terminal or t.kind == .incomplete or t.kind == .failed) try out.vt.event(out.ctx, t.content);
        if (t.kind == .pending) try out.vt.event(out.ctx, t.payload);
        const before = t.driver.writes;
        const bytes = t.driver.wire.items.len;
        const deadline = session.next_ns;
        time_ns = 700 * std.time.ns_per_ms;
        t.trace.enabled = true;
        defer t.trace.enabled = false;
        try out.vt.event(out.ctx, if (t.kind == .terminal or t.kind == .incomplete) t.terminal else if (t.kind == .pending) t.content else t.payload);
        if (t.kind == .terminal or t.kind == .incomplete or t.kind == .pending) try out.vt.event(out.ctx, null);
        if (t.kind == .role or t.kind == .empty_choices or t.kind == .empty_content) {
            try std.testing.expectEqual(@as(usize, 0), t.trace.n);
            try std.testing.expectEqual(before, t.driver.writes);
            try std.testing.expectEqual(bytes, t.driver.wire.items.len);
            try std.testing.expectEqual(deadline, session.next_ns);
        } else try std.testing.expect(t.driver.writes > before);
    }
};
threadlocal var active_translation: ?*Translation = null;

pub fn translatedInventory(route: Translator, period: i64, kind: TranslatedKind, fail_at: ?usize) !Result {
    time_ns = 0;
    var vt: std.Io.VTable = undefined; const io = ioFor(&vt);
    log.init(io, true); defer log.init(std.testing.io, true);
    var d: Driver = .{ .enabled = period > 0 }; defer d.wire.deinit(a);
    var text: Text = .{};
    const srv = try server.Server.init(a, io, d.engine(), text.text(), config(period), null); defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer { arena.deinit(); std.debug.print("qualification_d1381028 boundary=arena_deinit native=true\n", .{}); }
    const g = arena.allocator();
    const bootstrap_reserve = try g.alloc(u8, 1024 * 1024); g.free(bootstrap_reserve);
    // The excluded bootstrap reserve above stabilizes native arena slab boundaries.
    // A still sees every logical alloc/resize/remap, never just backing slabs.
    var trace: Trace = .{ .backing = g, .stage = .a5, .fail_at = fail_at };
    var conn = d.conn(g); defer { conn.deinit(); std.debug.print("qualification_d1381028 boundary=conn_deinit native=true\n", .{}); }
    const content = try parsed(g, "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}");
    const terminal = try parsed(g, if (kind == .incomplete) "{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":5,\"total_tokens\":11}}" else "{\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":6,\"completion_tokens\":5,\"total_tokens\":11}}");
    const payload = switch (kind) {
        .startup, .content, .terminal, .incomplete => content,
        .failed => try parsed(g, "{\"error\":{\"message\":\"failed fixture\",\"type\":\"server_error\"}}"),
        .pending => try parsed(g, "{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_f\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"q\\\":\"}}]},\"finish_reason\":null}]}"),
        .role => try parsed(g, "{\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}"),
        .empty_choices => try parsed(g, "{\"choices\":[]}"),
        .empty_content => try parsed(g, "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":null}]}"),
        .reasoning => try parsed(g, "{\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"reason\"},\"finish_reason\":null}]}"),
        .tools => try parsed(g, "{\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_f\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"q\\\":\\\"v\\\"}\"}}]},\"finish_reason\":null}]}"),
    };
    var t: Translation = .{ .trace = &trace, .driver = &d, .conn = &conn, .srv = srv, .kind = kind, .payload = payload, .content = content, .terminal = terminal, .period = period };
    const saved_translation = active_translation; active_translation = &t; defer active_translation = saved_translation;
    var hook: Observe.Hook = .{ .ctx = &d, .record = Driver.record }; const saved = Observe.active; Observe.active = &hook; defer Observe.active = saved;
    // Startup begins AFTER header/fixture Reply initialization, not a role event.
    // Test-only transport enables the logical wrapper at that real boundary.
    const saved_transport = conn.test_transport.?;
    if (kind == .startup) conn.test_transport.?.write_all = translatedStartupWrite;
    const call: *const fn (*server.Server, *http.Conn, Allocator, *const fn (openai.Out) anyerror!void) anyerror!void = if (route == .messages) @import("anthropic.zig").TestFixture.withWire else @import("responses.zig").TestFixture.withWire;
    call(srv, &conn, trace.allocator(), Translation.drive) catch |e| {
        try std.testing.expect(e == error.Closed and trace.fired != null);
        t.escaped = true;
    };
    conn.test_transport = saved_transport;
    try std.testing.expectEqual(d.starts, d.stops);
    try std.testing.expect(conn.stream_activity == null);
    if (kind == .startup) try std.testing.expectEqual(@as(usize, 0), d.starts)
    else try std.testing.expectEqual(@as(usize, if (period > 0) 1 else 0), d.starts);
    if (fail_at == null) trace.inventory(@tagName(kind));
    std.debug.print("qualification_a797d8f5 family=translated route={s} kind={s} stage=a5 period={d} fail_index={?d} attempts={d} fired={?d} escaped={} entered={} start={d} stop={d} physical_writes={d} failed_transport_writes={d} generation_fixture=not-invoked arena_deinit=after-owner-return\n", .{ @tagName(route), @tagName(kind), period, fail_at, trace.n, trace.fired, t.escaped, t.entered, d.starts, d.stops, d.writes, d.failed_writes });
    var r = result(&trace); r.stopped = d.starts == d.stops; return r;
}
fn translatedStartupWrite(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
    try Driver.write(ctx, bytes);
    if (std.mem.startsWith(u8, bytes, "HTTP/")) active_translation.?.trace.enabled = true;
}
pub const NativeKind = enum { plain, reasoning, tools, failed, length, cancelled, eos, stop_string, submit_busy, submit_closed, unfinished_tool_eos, fallback_tools, policy_tail };
/// Actual routes.dispatch -> Run.stream -> generate.loop/finish, with unfinished
/// engine borrowing. Bootstrap/plan allocations are disabled, not claimed swept.
pub fn nativeInventory(path: []const u8, period: i64, setting: usize, usage: bool, kind: NativeKind, stage: Stage, fail_at: ?usize) !Result {
    time_ns = 0;
    var vt: std.Io.VTable = undefined; const io = ioFor(&vt);
    log.init(io, true); defer log.init(std.testing.io, true);
    var d: Driver = .{ .path = path, .enabled = period > 0, .native_campaign = true,
        .terminal = switch (kind) { .failed => .failed, .length => .length, .cancelled => .cancelled, else => .stop },
        .prose = switch (kind) {
            .reasoning => "<think>reason</think>answer",
            .tools => "prose <tool_call>\n<function=f>\n<parameter=q>\nv\n</parameter>\n</function>\n</tool_call>",
            .unfinished_tool_eos => "<tool_call>{\"name\":\"f\",\"arguments\":{\"q\":\"v\"}",
            .fallback_tools => "{\"name\":\"f\",\"arguments\":{\"q\":\"v\"}}",
            .policy_tail => "hello <too",
            else => "hello",
        }, .eos_token = kind == .eos or kind == .unfinished_tool_eos, .refusal = switch (kind) { .submit_busy => error.Busy, .submit_closed => error.Closed, else => null } };
    defer d.wire.deinit(a);
    var text: Text = .{};
    var conf = config(period); conf.enable_thinking = kind == .reasoning;
    const srv = try server.Server.init(a, io, d.engine(), text.text(), conf, null); defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(a); defer arena.deinit();
    const g = arena.allocator();
    const reserve = try g.alloc(u8, 1024 * 1024); g.free(reserve);
    var trace: Trace = .{ .backing = g, .stage = stage, .fail_at = fail_at };
    var gtrace: Trace = .{ .backing = a, .stage = .g1, .fail_at = if (stage == .g1) fail_at else null, .strict = true };
    d.trace = if (stage == .g1) null else &trace;
    d.gtrace = &gtrace;
    srv.gpa = gtrace.allocator(); defer srv.gpa = a;
    const messages_route = std.mem.endsWith(u8, path, "/messages");
    const responses_route = std.mem.endsWith(u8, path, "/responses");
    const chat_route = std.mem.endsWith(u8, path, "/chat/completions");
    const tool_kind = kind == .tools or kind == .unfinished_tool_eos or kind == .fallback_tools or kind == .policy_tail;
    const tools = if (!tool_kind) "" else if (messages_route) ",\"tools\":[{\"name\":\"f\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}]" else if (responses_route) ",\"tools\":[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}]" else ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}}]";
    const use = if (usage) ",\"stream_options\":{\"include_usage\":true}" else "";
    const stopping = if (kind == .stop_string) ",\"stop\":[\"lo\"]" else if (kind == .policy_tail) ",\"parallel_tool_calls\":false" else "";
    const thinking = if (kind == .reasoning and messages_route) ",\"thinking\":{\"type\":\"adaptive\"}" else "";
    const body = if (responses_route) try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"input\":\"x\",\"stream\":true,\"store\":false,\"max_output_tokens\":200{s}{s}}}", .{tools, stopping}) else if (messages_route or chat_route) try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"stream\":true,\"max_tokens\":200{s}{s}{s}{s}}}", .{tools, if (chat_route) use else "", stopping, thinking}) else try std.fmt.allocPrint(g, "{{\"model\":\"m\",\"prompt\":\"x\",\"stream\":true,\"max_tokens\":200{s}{s}{s}}}", .{tools, use, stopping});
    var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(g, "{d}", .{body.len}) }};
    var conn = d.conn(g); conn.buf = try g.dupe(u8, body); conn.end = body.len; conn.method = "POST"; conn.headers = &headers;
    defer conn.deinit();
    var hook: Observe.Hook = .{ .ctx = &d, .record = Driver.record }; const saved = Observe.active; Observe.active = &hook; defer Observe.active = saved;
    routes.dispatch(srv, &conn, trace.allocator());
    try d.verify(srv, &conn);
    const selected = if (stage == .g1) &gtrace else &trace;
    try std.testing.expect(gtrace.live == 0 and gtrace.allocated_bytes == gtrace.freed_bytes);
    if (fail_at == null) {
        var labelbuf: [256]u8 = undefined;
        const label = try std.fmt.bufPrint(&labelbuf, "native-{s}-{s}-setting{d}-usage{}", .{path, @tagName(kind), setting, usage});
        selected.inventory(label);
    }
    std.debug.print("qualification_8e3a7f14 path={s} setting={d} usage={} kind={s} stage={s} fail_index={?d} attempts={d} fired={?d} cancel={d} box_freed_bytes={d} box_live={d} generated={}\n", .{path, setting, usage, @tagName(kind), @tagName(stage), fail_at, selected.n, selected.fired, d.cancelled, gtrace.freed_bytes, gtrace.live, d.generated});
    var r = result(selected); r.generated = d.generated; r.stopped = d.starts == d.stops; return r;
}
pub fn sweepNative(path: []const u8, period: i64, setting: usize, usage: bool, kind: NativeKind, stage: Stage, counts: *Counts) !void {
    const base = try nativeInventory(path, period, setting, usage, kind, stage, null); counts.baselines += 1;
    if (base.attempts == 0) counts.zero += 1;
    for (0..base.attempts) |i| {
        const r = try nativeInventory(path, period, setting, usage, kind, stage, i);
        try std.testing.expectEqual(@as(?usize, i), r.fired); try samePrefix(&base, &r, i + 1); counts.triggered += 1;
    }
    const sentinel = try nativeInventory(path, period, setting, usage, kind, stage, base.attempts);
    try std.testing.expect(sentinel.fired == null and sentinel.attempts == base.attempts); try samePrefix(&base, &sentinel, base.attempts); counts.sentinels += 1;
}

pub fn sweepTranslated(route: Translator, period: i64, kind: TranslatedKind, counts: *Counts) !void {
    const base = try translatedInventory(route, period, kind, null); counts.baselines += 1;
    if (kind == .role or kind == .empty_choices or kind == .empty_content) {
        try std.testing.expectEqual(@as(usize, 0), base.attempts); counts.zero += 1;
    } else try std.testing.expect(base.attempts > 0);
    for (0..base.attempts) |i| { const r = try translatedInventory(route, period, kind, i); try std.testing.expectEqual(@as(?usize, i), r.fired); try samePrefix(&base, &r, i + 1); counts.triggered += 1; }
    const sentinel = try translatedInventory(route, period, kind, base.attempts);
    try std.testing.expect(sentinel.fired == null and sentinel.attempts == base.attempts); try samePrefix(&base, &sentinel, base.attempts); counts.sentinels += 1;
}
