//! Real LaneHost/lanes.fake prefill and accepted-queue witnesses through all actual routes; no sockets/GPU.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const http = @import("http_conn.zig");
const server = @import("server.zig");
const model = @import("model_text.zig");
const json = @import("json.zig");
const routes = @import("routes.zig");
const log = @import("log.zig");
const a = std.testing.allocator;
// Fixture gates use native awake time, never the synthetic request clock.
fn awaitFlag(flag: *const std.atomic.Value(bool)) void {
    const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 10 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() >= deadline) @panic("qualification_4533b989 CPU fixture barrier timed out");
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch @panic("qualification_4533b989 CPU fixture sleep cancelled");
    }
}
var time_ns: std.atomic.Value(i64) = .init(0);
fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp { return .{ .nanoseconds = time_ns.load(.acquire) }; }
fn random(_: ?*anyopaque, bytes: []u8) void { @memset(bytes, 0x12); }
fn wait(ctx: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
    switch (timeout) {
        .deadline => |d| {
            _ = time_ns.fetchMax(@intCast(d.raw.toNanoseconds()), .acq_rel);
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        },
        else => return std.testing.io.vtable.futexWait(ctx, ptr, expected, timeout),
    }
}
const Text = struct {
    fn text(t: *Text) model.Text { return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } }; }
    fn encode(_: *anyopaque, g: std.mem.Allocator, s: []const u8, _: bool) model.Error![]u32 { const ids = try g.alloc(u32, s.len); for (s, ids) |b, *p| p.* = b; return ids; }
    fn decode(_: *anyopaque, g: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 { const bytes = try g.alloc(u8, ids.len); for (ids, bytes) |id, *p| p.* = @intCast(id); return bytes; }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 { return null; }
    fn tokenString(_: *anyopaque, g: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 { return std.fmt.allocPrint(g, "{d}", .{id}); }
    fn vocab(_: *anyopaque) u32 { return 256; }
    fn eos(_: *anyopaque) []const u32 { return &.{}; }
    fn render(_: *anyopaque, g: std.mem.Allocator, _: json.Value, _: model.RenderOptions, _: *[]const u8) model.Error![]u8 { return g.dupe(u8, "prompt"); }
    fn template(_: *anyopaque) []const u8 { return ""; }
};
const Phase = enum { normal, lone, queue };
const Gate = struct {
    host: *api.LaneHost,
    phase: Phase,
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    request_thread: std.Thread.Id,
    engine_thread: std.atomic.Value(std.Thread.Id) = .init(0),
    submitted: bool = false,
    request_id: api.Id = 0,
    sink: ?api.Sink = null,
    comments: usize = 0,
    stages: usize = 0,
    cancels: usize = 0,
    data: std.ArrayList(u8) = .empty,
    fn self(ctx: ?*anyopaque) *Gate { return @ptrCast(@alignCast(ctx.?)); }
    fn prefill(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
        const gate = self(ctx);
        gate.engine_thread.store(std.Thread.getCurrentId(), .release);
        gate.entered.store(true, .release);
        awaitFlag(&gate.release);
    }
    fn engine(g: *Gate) api.Engine { return .{ .ctx = g, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .admit = admit } }; }
    fn info(ctx: *anyopaque) api.Info { return self(ctx).host.engine().info(); }
    fn admit(ctx: *anyopaque, bg: bool) bool { return self(ctx).host.engine().admits(bg); }
    fn submit(ctx: *anyopaque, id: api.Id, req: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const g = self(ctx); g.submitted = true; g.request_id = id; g.sink = sink;
        try g.host.engine().submit(id, req, sink);
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const g = self(ctx); g.cancels += 1; g.host.engine().cancel(id);
        if (g.phase == .queue) g.host.engine().cancel(999);
        g.release.store(true, .release);
    }
    fn status(ctx: *anyopaque, out: *api.Status, tokens: []u32) void { self(ctx).host.engine().status(out, tokens); }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
    fn gone(ctx: *anyopaque) bool {
        const g = self(ctx);
        std.debug.assert(std.Thread.getCurrentId() == g.request_thread);
        if (!g.submitted or !g.entered.load(.acquire)) return false;
        g.stages += 1;
        time_ns.store(@as(i64, @intCast(g.stages)) * 2 * std.time.ns_per_s, .release);
        return g.stages >= 3;
    }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void { self(ctx).data.appendSlice(a, bytes) catch return error.Closed; }
    fn commentNow(_: ?*anyopaque) i96 { return time_ns.load(.acquire); }
    fn poll(_: ?*anyopaque, fd: std.posix.socket_t, ms: i32) http.PollResult { std.debug.assert(fd == -1 and ms > 0 and ms <= 50); return .out; }
    fn send(ctx: ?*anyopaque, fd: std.posix.socket_t, bytes: []const u8) http.SendResult {
        const g = self(ctx);
        std.debug.assert(fd == -1 and std.Thread.getCurrentId() == g.request_thread);
        std.debug.assert(g.engine_thread.load(.acquire) != g.request_thread);
        std.debug.assert(g.entered.load(.acquire) and !g.release.load(.acquire));
        std.debug.assert(g.host.mutex.tryLock());
        g.host.mutex.unlock(std.testing.io);
        g.comments += 1;
        g.data.appendSlice(a, bytes) catch @panic("allocation");
        return .{ .sent = bytes.len };
    }
};
const Lone = struct {
    be: lanes.backend.Backend,
    fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
        const l: *Lone = @ptrCast(@alignCast(ctx));
        _ = try l.be.opening(a, s);
        return error.GatedExit;
    }
};
fn discard(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
test "qualification_4533b989 actual LaneHost queue normal and lone prefill comments and cancellation all four routes" {
    for ([_][]const u8{ "/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses" }) |path| for ([_]Phase{ .normal, .lone, .queue }) |phase| for ([_]i64{ 0, std.time.ns_per_s }) |period| {
        time_ns.store(0, .release);
        var vtable = std.testing.io.vtable.*; vtable.now = now; vtable.random = random; vtable.futexWait = wait;
        const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };
        log.init(io, true); defer log.init(std.testing.io, true);
        var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7); defer cfg.deinit(a);
        var fake: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 1 }; defer fake.deinit();
        var fixed: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, fake.backend(), fixed.clock()); defer core.deinit();
        var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
        var gate: Gate = .{ .host = &host, .phase = phase, .request_thread = std.Thread.getCurrentId() }; defer gate.data.deinit(a);
        fake.prefill_hook = Gate.prefill; fake.prefill_hook_ctx = &gate;
        var lone: Lone = .{ .be = fake.backend() };
        if (phase == .lone) host.lone = .{ .ctx = &lone, .run = Lone.run };
        try host.start(); defer host.stop();
        var dummy: u8 = 0;
        const blocker: api.Request = .{ .prompt = &.{ 3, 1, 4 }, .max_tokens = 1, .drafts = false };
        if (phase == .queue) {
            try host.engine().submit(999, &blocker, .{ .ctx = &dummy, .event = discard });
            awaitFlag(&gate.entered);
        }
        var text: Text = .{};
        const srv = try server.Server.init(a, io, gate.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .keep_warm_s = 0, .use_drafts = phase == .lone, .enable_thinking = false, .sse_keepalive_ns = period }, null); defer srv.deinit();
        var arena = std.heap.ArenaAllocator.init(a);
        defer { arena.deinit(); std.debug.print("qualification_0974903b route={s} phase={s} period={d} boundary=arena_deinit after_route_return=true\n", .{path, @tagName(phase), period}); }
        const g = arena.allocator();
        const body = if (std.mem.endsWith(u8, path, "/responses")) "{\"model\":\"m\",\"input\":\"x\",\"stream\":true,\"max_output_tokens\":8}" else if (std.mem.endsWith(u8, path, "/messages") or std.mem.endsWith(u8, path, "/chat/completions")) "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":8}" else "{\"model\":\"m\",\"prompt\":\"x\",\"stream\":true,\"max_tokens\":8}";
        var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(g, "{d}", .{body.len}) }};
        var c: http.Conn = .{ .fd = -1, .peer = "fixture", .gpa = g, .buf = try g.dupe(u8, body), .end = body.len, .method = "POST", .version = "HTTP/1.1", .path = path, .headers = &headers, .quiet_log = true, .test_transport = .{ .ctx = &gate, .write_all = Gate.write, .peer_gone = Gate.gone, .comment_ops = .{ .ctx = &gate, .now = Gate.commentNow, .poll = Gate.poll, .send = Gate.send } } };
        routes.dispatch(srv, &c, g);
        try std.testing.expect(gate.submitted and gate.entered.load(.acquire) and gate.release.load(.acquire));
        try std.testing.expectEqual(@as(usize, 1), gate.cancels);
        try std.testing.expectEqual(@as(usize, if (period > 0) 2 else 0), gate.comments);
        try std.testing.expect(c.stream_activity == null);
        try std.testing.expect(std.mem.indexOf(u8, gate.data.items, "[DONE]") == null);
        try std.testing.expect(std.mem.indexOf(u8, gate.data.items, "message_stop") == null);
        try std.testing.expect(std.mem.indexOf(u8, gate.data.items, "response.completed") == null);
        try std.testing.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
        std.debug.print("qualification_c492c4ea {s} {s} period={d} comments={d} cancels={d} engine-thread-different=true\n", .{path, @tagName(phase), period, gate.comments, gate.cancels});
    };
}

// The existing Backend VTable decorates Fake.verify/release only; all math still runs in lanes.fake.
const HeldPhase = enum { normal, lone, queue, decode };
var active_held: ?*Held = null;
const Held = struct {
    host: *api.LaneHost,
    backend: lanes.backend.Backend,
    phase: HeldPhase,
    period: i64,
    request_thread: std.Thread.Id,
    case_id: usize = 0,
    finished_case_id: ?usize = null,
    finished_empty_lengths: [4]usize = undefined,
    borrow_snapshot: bool = false,
    mailbox_ledger: ?*@import("qualification_59878042_private_probe.zig").Ledger = null,
    engine_thread: std.atomic.Value(std.Thread.Id) = .init(0),
    entered: std.atomic.Value(bool) = .init(false),
    held_at: std.atomic.Value(i64) = .init(0),
    release_work: std.atomic.Value(bool) = .init(false),
    allow_finished: std.atomic.Value(bool) = .init(false),
    cancelled: std.atomic.Value(bool) = .init(false),
    finished: std.atomic.Value(bool) = .init(false),
    release_probe: std.atomic.Value(bool) = .init(false),
    probe_sent: std.atomic.Value(bool) = .init(false),
    submitted: std.atomic.Value(bool) = .init(false),
    returned: std.atomic.Value(bool) = .init(false),
    prefilled: std.atomic.Value(bool) = .init(false),
    engine_tokens: std.atomic.Value(usize) = .init(0),
    drain_waits: std.atomic.Value(usize) = .init(0),
    borrowed: ?*const api.Request = null,
    sink: ?api.Sink = null,
    prompt_copy: [256]u32 = undefined,
    prompt_len: usize = 0,
    id: api.Id = 0,
    mailbox_epoch: ?*const u32 = null,
    first_drain_epoch: ?u32 = null,
    disconnect: bool = false,
    cancel_count: usize = 0,
    cancellation_bytes: usize = 0,
    cancellation_comments: usize = 0,
    writes_after_cancel: usize = 0,
    last_wire_ns: i64 = 0,
    comments: usize = 0,
    cadence: [8]i64 = undefined,
    bytes: std.ArrayList(u8) = .empty,

    fn self(ctx: *anyopaque) *Held { return @ptrCast(@alignCast(ctx)); }
    fn engine(h: *Held) api.Engine { return .{ .ctx = h, .vtable = &.{ .info = info, .admit = admit, .submit = submit, .cancel = cancel, .status = status, .memory = memory } }; }
    fn info(ctx: *anyopaque) api.Info { return self(ctx).host.engine().info(); }
    fn admit(ctx: *anyopaque, bg: bool) bool { return self(ctx).host.engine().admits(bg); }
    fn status(ctx: *anyopaque, out: *api.Status, tokens: []u32) void { self(ctx).host.engine().status(out, tokens); }
    fn memory(_: *anyopaque, _: bool) ?api.Memory { return null; }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h = self(ctx);
        std.debug.assert(std.Thread.getCurrentId() == h.request_thread);
        std.debug.assert(request.prompt.len <= h.prompt_copy.len and request.max_tokens == 8);
        h.borrowed = request;
        h.sink = sink;
        h.id = id;
        h.prompt_len = request.prompt.len;
        @memcpy(h.prompt_copy[0..h.prompt_len], request.prompt);
        try h.host.engine().submit(id, request, .{ .ctx = h, .event = event });
        h.submitted.store(true, .release);
    }
    fn record(ctx: *anyopaque, e: @import("sse_test_observe.zig").Test.Event) void {
        const h = self(ctx);
        if (e == .idle and !h.borrow_snapshot) {
            h.checkBorrow();
            std.debug.assert(h.mailbox_ledger.?.teardown_frees == 0);
            std.debug.print("qualification_64db54ab case={d} request={x} prompt={x} prompt_len={d} sink={x} finished=false mailbox_frees_before_finish=0\n", .{h.case_id, @intFromPtr(h.borrowed.?), @intFromPtr(h.borrowed.?.prompt.ptr), h.prompt_len, @intFromPtr(h.sink.?.ctx)});
            h.borrow_snapshot = true;
        }
        if (e == .drain_end or e == .box_deinit or e == .request_end or e == .generated) {
            std.debug.assert(h.finished.load(.acquire) and h.borrowed == null and h.sink == null);
            std.debug.assert(h.allow_finished.load(.acquire) and h.drain_waits.load(.acquire) >= 2);
        }
    }
    fn checkBorrow(h: *Held) void {
        std.debug.assert(!h.returned.load(.acquire) and !h.finished.load(.acquire));
        const request = h.borrowed.?;
        std.debug.assert(request.max_tokens == 8);
        std.debug.assert(std.mem.eql(u32, request.prompt, h.prompt_copy[0..h.prompt_len]));
        std.debug.assert(h.sink != null);
    }
    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const h = self(ctx);
        std.debug.assert(std.Thread.getCurrentId() == h.engine_thread.load(.acquire));
        h.checkBorrow();
        const sink = h.sink.?;
        switch (e.*) {
            .prefilled => h.prefilled.store(true, .release),
            .tokens => |tokens| { _ = h.engine_tokens.fetchAdd(tokens.len, .acq_rel); },
            .finished => |f| {
                std.debug.assert(f.reason == .cancelled);
                std.debug.assert(h.allow_finished.load(.acquire) and h.drain_waits.load(.acquire) >= 2);
                std.debug.assert(f.stats.prefill_widths.len == 0);
                std.debug.assert(f.stats.prefill_raised.len == 0);
                std.debug.assert(f.stats.telemetry_json.len == 0);
                std.debug.assert(f.message.len == 0);
                std.debug.assert(h.finished_case_id == null);
                h.finished_empty_lengths = .{ f.stats.prefill_widths.len, f.stats.prefill_raised.len, f.stats.telemetry_json.len, f.message.len };
                h.finished_case_id = h.case_id;
                h.borrowed = null;
                h.sink = null;
                h.finished.store(true, .release);
            },
        }
        // Never wait in an api.Sink callback; the real Mailbox receives the actual engine event.
        sink.event(sink.ctx, id, e);
    }
    fn hold(h: *Held) void {
        h.engine_thread.store(std.Thread.getCurrentId(), .release);
        h.held_at.store(time_ns.load(.acquire), .release);
        h.entered.store(true, .release);
        awaitFlag(&h.release_work);
    }
    fn prefill(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
        const h = self(ctx);
        h.engine_thread.store(std.Thread.getCurrentId(), .release);
        if (h.phase != .decode) h.hold();
    }
    fn verify(ctx: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        const h = active_held.?;
        std.debug.assert(ctx == h.backend.ptr);
        // Opening never verifies: this is the genuine familyRound inside LaneHost's core.step.
        if (h.phase == .decode and !h.entered.load(.acquire)) h.hold();
        return h.backend.verify(windows, out);
    }
    fn release(ctx: *anyopaque, stream: *lanes.Stream) void {
        const h = active_held.?;
        std.debug.assert(ctx == h.backend.ptr);
        if (h.cancelled.load(.acquire) and !h.release_probe.swap(true, .acq_rel)) {
            h.checkBorrow();
            std.debug.assert(h.drain_waits.load(.acquire) >= 1);
            // Labelled CPU nonterminal wake: copy into the real mailbox while the request is draining.
            h.probe_sent.store(true, .release);
            const e: api.Event = .{ .tokens = &.{88} };
            const sink = h.sink.?;
            sink.event(sink.ctx, h.id, &e);
            // Only a second native drain wait with a changed epoch permits the engine's finished.
            awaitFlag(&h.allow_finished);
            h.checkBorrow();
        }
        h.backend.release(stream);
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        std.debug.assert(h.disconnect and !h.finished.load(.acquire));
        h.cancel_count += 1;
        h.cancelled.store(true, .release);
        h.host.engine().cancel(id);
        if (h.phase == .queue) h.host.engine().cancel(999);
        // Do not release here: cancellation must reach the real request's drain first.
    }
    fn gone(ctx: *anyopaque) bool {
        const h = self(ctx);
        std.debug.assert(std.Thread.getCurrentId() == h.request_thread);
        if (!h.submitted.load(.acquire) or !h.entered.load(.acquire)) return false;
        if (time_ns.load(.acquire) < h.held_at.load(.acquire) + 4 * std.time.ns_per_s) return false;
        if (!h.disconnect) {
            h.disconnect = true;
            h.cancellation_bytes = h.bytes.items.len;
            h.cancellation_comments = h.comments;
        }
        return true;
    }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const h = self(ctx);
        std.debug.assert(std.Thread.getCurrentId() == h.request_thread);
        if (h.disconnect) h.writes_after_cancel += 1;
        h.bytes.appendSlice(a, bytes) catch return error.Closed;
        h.last_wire_ns = time_ns.load(.acquire);
    }
    fn send(ctx: ?*anyopaque, fd: std.posix.socket_t, bytes: []const u8) http.SendResult {
        const h = self(ctx.?);
        std.debug.assert(fd == -1 and std.Thread.getCurrentId() == h.request_thread);
        std.debug.assert(h.engine_thread.load(.acquire) != h.request_thread);
        std.debug.assert(h.submitted.load(.acquire) and h.entered.load(.acquire));
        std.debug.assert(!h.disconnect and !h.release_work.load(.acquire));
        std.debug.assert(std.mem.eql(u8, bytes, ": keepalive\n\n"));
        std.debug.assert(h.host.mutex.tryLock());
        h.host.mutex.unlock(std.testing.io);
        if (h.phase == .decode) std.debug.assert(h.prefilled.load(.acquire) and h.engine_tokens.load(.acquire) > 0);
        if (h.phase == .queue) {
            var status_: api.Status = .{};
            h.host.engine().status(&status_, &.{});
            std.debug.assert(status_.running == 1 and status_.waiting == 1);
        }
        const at = time_ns.load(.acquire);
        const gap = at - h.last_wire_ns;
        std.debug.assert(gap >= h.period and gap <= h.period + 50 * std.time.ns_per_ms);
        std.debug.assert(h.comments < h.cadence.len);
        h.cadence[h.comments] = gap;
        h.comments += 1;
        h.last_wire_ns = at;
        h.bytes.appendSlice(a, bytes) catch @panic("allocation");
        return .{ .sent = bytes.len };
    }
};

fn heldWait(ctx: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
    if (active_held) |h| if (std.Thread.getCurrentId() == h.request_thread and h.submitted.load(.acquire)) {
        if (timeout == .deadline) {
            h.mailbox_epoch = ptr;
            // Establish the engine barrier before advancing the scripted healthy-client clock.
            awaitFlag(&h.entered);
        }
    };
    // The synthetic clock moves by real Generation.loop deadlines, not clock-call counts.
    return wait(ctx, ptr, expected, timeout);
}
fn heldDrainWait(ctx: ?*anyopaque, ptr: *const u32, expected: u32) void {
    if (active_held) |h| if (std.Thread.getCurrentId() == h.request_thread and h.cancelled.load(.acquire) and !h.finished.load(.acquire) and h.mailbox_epoch == ptr) {
        // No borrowed-object reads once finished is allowed to race with this waiter.
        if (!h.allow_finished.load(.acquire)) h.checkBorrow();
        _ = h.drain_waits.fetchAdd(1, .acq_rel);
        if (h.first_drain_epoch == null) {
            h.first_drain_epoch = expected;
            h.release_work.store(true, .release);
        } else if (h.probe_sent.load(.acquire) and h.first_drain_epoch.? != expected) {
            h.allow_finished.store(true, .release);
        }
    };
    // Bound the fixture's native futex wait without changing production Generation.drain.
    const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 10 * std.time.ns_per_s;
    while (@atomicLoad(u32, ptr, .acquire) == expected) {
        if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() >= deadline) @panic("qualification_4533b989 CPU native drain wait timed out");
        std.testing.io.vtable.futexWait(ctx, ptr, expected, .{ .deadline = .{ .raw = .{ .nanoseconds = deadline }, .clock = .awake } }) catch @panic("qualification_4533b989 CPU native drain wait cancelled");
    }
}

const Scope = struct { id: usize, conn_done: bool = false, arena_done: bool = false };

test "qualification_4533b989 CPU actual held decode and LaneHost borrow survives nonterminal cancellation drain all routes" {
    var case_serial: usize = 0;
    for ([_][]const u8{ "/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses" }) |path| for ([_]HeldPhase{ .normal, .lone, .queue, .decode }) |phase| for ([_]i64{ 0, 1025 * std.time.ns_per_ms }) |period| {
        time_ns.store(0, .release);
        var vt = std.testing.io.vtable.*;
        vt.now = now; vt.random = random; vt.futexWait = heldWait; vt.futexWaitUncancelable = heldDrainWait;
        const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vt };
        log.init(io, true); defer log.init(std.testing.io, true);
        var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = false, .hidden_rows = true }, 8, 7); defer cfg.deinit(a);
        var fake: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 1 }; defer fake.deinit();
        const backend = fake.backend();
        var bvt = backend.vtable.*; bvt.verify = Held.verify; bvt.release = Held.release;
        var fixed: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(a, &cfg, .{ .ptr = backend.ptr, .vtable = &bvt }, fixed.clock()); defer core.deinit();
        var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
        var h: Held = .{ .host = &host, .backend = backend, .phase = phase, .period = period, .request_thread = std.Thread.getCurrentId() };
        defer h.bytes.deinit(a);
        std.debug.assert(active_held == null); active_held = &h; defer active_held = null;
        fake.prefill_hook = Held.prefill; fake.prefill_hook_ctx = &h;
        var lone: Lone = .{ .be = .{ .ptr = backend.ptr, .vtable = &bvt } };
        if (phase == .lone) host.lone = .{ .ctx = &lone, .run = Lone.run };
        try host.start(); defer host.stop();
        var dummy: u8 = 0;
        const blocker: api.Request = .{ .prompt = &.{ 3, 1, 4 }, .max_tokens = 1, .drafts = false };
        if (phase == .queue) {
            try host.engine().submit(999, &blocker, .{ .ctx = &dummy, .event = discard });
            awaitFlag(&h.entered);
        }
        var text: Text = .{};
        const srv = try server.Server.init(a, io, h.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .keep_warm_s = 0, .use_drafts = phase == .lone, .enable_thinking = false, .sse_keepalive_ns = period }, null); defer srv.deinit();
        var scope: Scope = .{ .id = case_serial };
        case_serial += 1;
        defer std.debug.assert(scope.conn_done and scope.arena_done);
        var arena = std.heap.ArenaAllocator.init(a);
        defer {
            std.debug.assert(scope.conn_done and !scope.arena_done);
            arena.deinit(); scope.arena_done = true;
            std.debug.print("qualification_8c8feba8 case={d} route={s} phase={s} period={d} boundary=arena_deinit conn_before_arena=true after_route_return=true\n", .{scope.id, path, @tagName(phase), period});
        }
        const g = arena.allocator();
        const body = if (std.mem.endsWith(u8, path, "/responses")) "{\"model\":\"m\",\"input\":\"x\",\"stream\":true,\"max_output_tokens\":8}" else if (std.mem.endsWith(u8, path, "/messages") or std.mem.endsWith(u8, path, "/chat/completions")) "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":8}" else "{\"model\":\"m\",\"prompt\":\"x\",\"stream\":true,\"max_tokens\":8}";
        var headers = [_]http.Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(g, "{d}", .{body.len}) }};
        var c: http.Conn = .{ .fd = -1, .peer = "CPU held fixture", .gpa = g, .buf = try g.dupe(u8, body), .end = body.len, .method = "POST", .version = "HTTP/1.1", .path = path, .headers = &headers, .quiet_log = true, .test_transport = .{ .ctx = &h, .write_all = Held.write, .peer_gone = Held.gone, .comment_ops = .{ .ctx = &h, .now = Gate.commentNow, .poll = Gate.poll, .send = Held.send } } };
        defer {
            std.debug.assert(h.returned.load(.acquire) and c.stream_activity == null and !scope.conn_done and !scope.arena_done);
            c.deinit(); scope.conn_done = true;
            std.debug.print("qualification_8c8feba8 case={d} route={s} phase={s} period={d} boundary=conn_deinit after_route_return=true\n", .{scope.id, path, @tagName(phase), period});
        }
        const Observation = @import("sse_test_observe.zig").Test;
        const Private = @import("qualification_59878042_private_probe.zig");
        var ledger: Private.Ledger = .{ .backing = a };
        h.case_id = scope.id;
        h.mailbox_ledger = &ledger;
        var borrow_hook: Observation.Hook = .{ .ctx = &h, .record = Held.record };
        var probe: Private.Probe = .{ .enabled = period > 0, .streamed = true, .gpa = &ledger, .upstream = &borrow_hook };
        var hook: Observation.Hook = .{ .ctx = &probe, .record = Private.Probe.record };
        const previous = Observation.active; Observation.active = &hook;
        defer Observation.active = previous;
        srv.gpa = ledger.allocator();
        defer srv.gpa = a;
        routes.dispatch(srv, &c, g);
        try probe.check();
        ledger.printFrees(scope.id);
        try std.testing.expectEqual(@as(usize, 2), ledger.teardown_frees);
        try std.testing.expect(h.borrow_snapshot);
        try std.testing.expect(h.finished.load(.acquire) and h.borrowed == null and h.sink == null);
        try std.testing.expectEqual(@as(?usize, scope.id), h.finished_case_id);
        for ([_][]const u8{ "prefill_widths", "prefill_raised", "telemetry_json", "message" }, h.finished_empty_lengths) |buffer, len| {
            try std.testing.expectEqual(@as(usize, 0), len);
            std.debug.print("qualification_895f5c6f case={d} route={s} phase={s} period={d} buffer={s} len={d} allocation=NOT-APPLICABLE free=NOT-APPLICABLE after_dispatch=true\n", .{scope.id, path, @tagName(phase), period, buffer, len});
        }
        std.debug.print("qualification_ffcb6c3e case={d} finished_before_drain_end=true borrow_released_before_box=true frees_before_request_end=true request_end_before_generated=true\n", .{scope.id});
        try std.testing.expect(probe.mutex_checks > 0 and probe.boxes == 1 and ledger.teardown_frees > 0);
        try std.testing.expect(probe.at(.drain_begin).? < probe.at(.drain_end).?);
        try std.testing.expect(probe.at(.generated).? < probe.at(.run_exit).?);
        if (period > 0) try std.testing.expectEqual(@as(usize, 1), probe.starts);
        std.debug.print("qualification_31a98361 route={s} phase={s} period={d} mutex={d} starts={d} stops={d} boxes={d} allocated_bytes={d} freed_bytes={d} teardown_frees={d} drain_finished=true request_end_after_box=true\n", .{path, @tagName(phase), period, probe.mutex_checks, probe.starts, probe.stops, probe.boxes, ledger.allocated, ledger.freed, ledger.teardown_frees});
        h.returned.store(true, .release);
        try std.testing.expect(h.finished.load(.acquire) and h.release_probe.load(.acquire));
        try std.testing.expect(h.drain_waits.load(.acquire) >= 2 and h.allow_finished.load(.acquire));
        try std.testing.expectEqual(@as(usize, 1), h.cancel_count);
        try std.testing.expect(h.borrowed == null and h.sink == null);
        try std.testing.expectEqual(@as(usize, 0), h.writes_after_cancel);
        try std.testing.expectEqual(h.cancellation_bytes, h.bytes.items.len);
        try std.testing.expectEqual(h.cancellation_comments, h.comments);
        if (period == 0) try std.testing.expectEqual(@as(usize, 0), h.comments) else try std.testing.expect(h.comments >= 3);
        try std.testing.expect(c.stream_activity == null);
        for ([_][]const u8{ "[DONE]", "message_stop", "response.completed", "response.incomplete", "response.failed" }) |terminal| try std.testing.expect(std.mem.indexOf(u8, h.bytes.items, terminal) == null);
        try std.testing.expectEqual(@as(i64, 0), srv.preparing.load(.acquire));
        if (phase == .decode) try std.testing.expect(h.prefilled.load(.acquire) and h.engine_tokens.load(.acquire) > 0);
        std.debug.print("qualification_3c7f19b8 CPU synthetic-clock/transport {s} {s} period={d} comments={d} cadence-ns={any} drain-waits={d} nonterminal-wake=true no-data-after-cancel=true distinct-engine-thread=true\n", .{ path, @tagName(phase), period, h.comments, h.cadence[0..h.comments], h.drain_waits.load(.acquire) });
    };
}
