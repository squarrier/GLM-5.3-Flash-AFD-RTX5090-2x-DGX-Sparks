//! Frozen N05/N07: authentic private Wire/Reply, socket-free transport and real Session.
//! GPT-6.1 Sol (OpenAI Codex), via Hermes; inherited attribution remains unchanged.
const std = @import("std");
const json = @import("json.zig");
const chat = @import("chat.zig");
const openai = @import("openai.zig");
const http = @import("http_conn.zig");
const keep = @import("sse_keepalive.zig");
const Server = @import("server.zig").Server;
const log = @import("log.zig");
const Observation = @import("sse_test_observe.zig").Test;
const a = std.testing.allocator;
const Route = Observation.Route;
const Input = enum { role, empty_choices, empty_content };
const Control = enum { timer, pending_call, postfinal };
const ms = std.time.ns_per_ms;
const comment = ": keepalive\n\n";

const Frame = struct { offset: usize, len: usize };
// Counts logical alloc/resize/remap attempts, not Arena slab allocations; no failure campaign.
const Logical = struct {
    inner: std.mem.Allocator,
    attempts: usize = 0,
    fn allocator(l: *Logical) std.mem.Allocator {
        return .{ .ptr = l, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const l: *Logical = @ptrCast(@alignCast(ctx)); l.attempts += 1;
        return l.inner.rawAlloc(len, alignment, ra);
    }
    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) bool {
        const l: *Logical = @ptrCast(@alignCast(ctx)); l.attempts += 1;
        return l.inner.rawResize(memory, alignment, len, ra);
    }
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
        const l: *Logical = @ptrCast(@alignCast(ctx)); l.attempts += 1;
        return l.inner.rawRemap(memory, alignment, len, ra);
    }
    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const l: *Logical = @ptrCast(@alignCast(ctx)); l.inner.rawFree(memory, alignment, ra);
    }
};
const Cell = struct {
    route: Route,
    input: Input,
    period: i64,
    control: Control = .timer,
    time: i96 = 0,
    io: std.Io = undefined,
    conn: *http.Conn = undefined,
    session: ?*keep.Session = null,
    logical: *Logical = undefined,
    payload: json.Value = undefined,
    content: json.Value = undefined,
    call_start: json.Value = undefined,
    queued_text: json.Value = undefined,
    call_end: json.Value = undefined,
    terminal: json.Value = undefined,
    empty: json.Value = undefined,
    wire: std.ArrayList(u8) = .empty,
    frames: [32]Frame = undefined,
    frames_n: usize = 0,
    writes: usize = 0,
    comments: usize = 0,
    saved_activity: usize = 0,
    ledger: [256]Observation.Event = undefined,
    ledger_n: usize = 0,
    arm_noop: bool = false,
    fail_frame: ?usize = null,
    prefix: usize = 0,
    content_write_base: usize = 0,
    content_active: bool = false,
    faults: usize = 0,
    failed_next: i96 = 0,
    failed_wrote: usize = 0,

    fn self(ctx: ?*anyopaque) *Cell { return @ptrCast(@alignCast(ctx.?)); }
    fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp { return .{ .nanoseconds = active.?.time }; }
    fn random(_: ?*anyopaque, bytes: []u8) void { @memset(bytes, 0x12); }
    fn commentNow(ctx: ?*anyopaque) i96 { return self(ctx).time; }
    fn poll(_: ?*anyopaque, fd: std.posix.socket_t, timeout_ms: i32) http.PollResult {
        std.debug.assert(fd == -1 and timeout_ms > 0 and timeout_ms <= 50);
        return .out;
    }
    fn send(ctx: ?*anyopaque, fd: std.posix.socket_t, bytes: []const u8) http.SendResult {
        std.debug.assert(fd == -1);
        const c = self(ctx);
        std.debug.assert(std.mem.eql(u8, bytes, comment));
        c.comments += 1;
        c.wire.appendSlice(a, bytes) catch @panic("fixture allocation");
        return .{ .sent = bytes.len };
    }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const c = self(ctx);
        const at = c.writes; c.writes += 1;
        if (c.arm_noop or (c.content_active and c.fail_frame != null and at == c.content_write_base + c.fail_frame.?)) {
            std.debug.assert(c.prefix < bytes.len);
            c.faults += 1;
            c.failed_next = if (c.session) |s| s.next_ns else 0;
            c.failed_wrote = c.countSession(.wrote);
            c.wire.appendSlice(a, bytes[0..c.prefix]) catch return error.Closed;
            return error.Closed;
        }
        std.debug.assert(c.frames_n < c.frames.len);
        c.frames[c.frames_n] = .{ .offset = c.wire.items.len, .len = bytes.len }; c.frames_n += 1;
        c.wire.appendSlice(a, bytes) catch return error.Closed;
    }
    fn gone(_: *anyopaque) bool { return false; }
    fn savedWrote(ctx: *anyopaque) void { self(ctx).saved_activity += 1; }
    fn record(ctx: *anyopaque, event: Observation.Event) void {
        const c = self(ctx);
        switch (event) { .session, .translated => {}, else => @panic("unexpected private-fixture observation") }
        std.debug.assert(c.ledger_n < c.ledger.len);
        c.ledger[c.ledger_n] = event; c.ledger_n += 1;
    }
    fn countSession(c: *const Cell, phase: Observation.SessionPhase) usize {
        var n: usize = 0;
        for (c.ledger[0..c.ledger_n]) |e| switch (e) { .session => |s| { if (s.phase == phase) n += 1; }, else => {} };
        return n;
    }
    fn pair(c: *Cell, start: usize, unchanged: bool, terminal: bool) !void {
        try std.testing.expect(c.ledger_n >= start + 2);
        const enter = c.ledger[start].translated;
        const exit = c.ledger[c.ledger_n - 1].translated;
        try std.testing.expectEqual(c.route, enter.route);
        try std.testing.expectEqual(c.route, exit.route);
        try std.testing.expectEqual(Observation.Phase.enter, enter.phase);
        try std.testing.expectEqual(Observation.Phase.exit, exit.phase);
        try std.testing.expectEqual(terminal, enter.terminal);
        try std.testing.expectEqual(terminal, exit.terminal);
        if (unchanged) {
            try std.testing.expectEqual(start + 2, c.ledger_n);
            try std.testing.expectEqual(enter.sequence_or_index, exit.sequence_or_index);
        }
    }
};
threadlocal var active: ?*Cell = null;

fn choice(g: std.mem.Allocator, delta: json.Value, finish: json.Value) !json.Value {
    const ch = try json.newObject(g);
    try ch.put(g, "index", .{ .int = "0" });
    try ch.put(g, "delta", delta);
    try ch.put(g, "finish_reason", finish);
    const choices = try g.alloc(json.Value, 1); choices[0] = .{ .object = ch };
    const p = try json.newObject(g); try p.put(g, "choices", .{ .array = choices });
    return .{ .object = p };
}
fn parsed(g: std.mem.Allocator, text: []const u8) !json.Value { return (try json.parse(g, text)).ok; }

fn noOp(c: *Cell, out: openai.Out, payload: json.Value, s: *keep.Session, terminal: bool) !void {
    const bytes = c.wire.items.len;
    const writes = c.writes;
    const activity = c.saved_activity;
    const wrote = c.countSession(.wrote);
    const next = s.next_ns;
    const ledger = c.ledger_n;
    const attempts = c.logical.attempts;
    try out.vt.event(out.ctx, payload);
    try c.pair(ledger, true, terminal);
    try std.testing.expectEqual(bytes, c.wire.items.len);
    try std.testing.expectEqual(writes, c.writes);
    try std.testing.expectEqual(activity, c.saved_activity);
    try std.testing.expectEqual(wrote, c.countSession(.wrote));
    try std.testing.expectEqual(next, s.next_ns);
    try std.testing.expectEqual(attempts, c.logical.attempts);
    try std.testing.expectEqual(@as(usize, 0), c.faults);
}

fn drive(out: openai.Out) anyerror!void {
    const c = active.?;
    // withWire has already completed actual startup events. No startup write resets this Session.
    try std.testing.expect(c.writes > 0 and c.saved_activity > 0);
    try std.testing.expectEqual(@as(usize, 0), c.ledger_n);
    var s: keep.Session = .{ .conn = c.conn, .io = c.io, .period_ns = c.period };
    c.session = &s; defer c.session = null;
    if (c.period > 0) s.start();
    defer s.stop();
    try std.testing.expectEqual(@as(usize, if (c.period > 0) 1 else 0), c.countSession(.start));
    try std.testing.expectEqual(@as(i96, if (c.period > 0) 1000 * ms else 0), s.next_ns);
    if (c.period > 0) try std.testing.expect(c.conn.stream_activity.?.ctx == @as(*anyopaque, @ptrCast(&s)));

    if (c.control == .pending_call) {
        c.time = 200 * ms; try out.vt.event(out.ctx, c.call_start);
        const before = c.writes;
        c.time = 400 * ms; try out.vt.event(out.ctx, c.queued_text);
        if (c.route == .messages) try std.testing.expectEqual(before, c.writes);
        c.time = 700 * ms; try noOp(c, out, c.empty, &s, false);
        const bytes = c.wire.items.len;
        const wrote = c.countSession(.wrote);
        c.time = 1700 * ms; try out.vt.event(out.ctx, c.call_end);
        try std.testing.expect(c.wire.items.len > bytes);
        if (c.period > 0) {
            try std.testing.expect(c.countSession(.wrote) > wrote);
            try std.testing.expectEqual(@as(i96, 2700 * ms), s.next_ns);
        }
        if (c.route == .messages) try std.testing.expect(std.mem.indexOf(u8, c.wire.items[bytes..], "queued") != null);
        return;
    }
    if (c.control == .postfinal) {
        c.time = 400 * ms;
        if (c.route == .messages) { try out.vt.event(out.ctx, c.terminal); try out.vt.event(out.ctx, null); } else try out.vt.event(out.ctx, c.terminal);
        s.stop();
        try std.testing.expect(c.conn.stream_activity.?.ctx == @as(*anyopaque, @ptrCast(c)));
        c.time = 700 * ms; try noOp(c, out, c.empty, &s, true);
        const bytes = c.wire.items.len; c.time = 1000 * ms; try s.tick();
        try std.testing.expectEqual(bytes, c.wire.items.len);
        try std.testing.expectEqual(@as(usize, 0), c.comments);
        return;
    }

    c.time = 700 * ms;
    try noOp(c, out, c.payload, &s, false);
    c.arm_noop = false;
    const before_tick = c.wire.items.len;
    c.time = 1000 * ms; try s.tick();
    try std.testing.expectEqual(@as(usize, if (c.period > 0) 1 else 0), c.comments);
    if (c.period > 0) {
        try std.testing.expectEqualStrings(comment, c.wire.items[before_tick..]);
        try std.testing.expectEqual(@as(i96, 2000 * ms), s.next_ns);
        try std.testing.expectEqual(@as(usize, 1), c.countSession(.comment));
    } else try std.testing.expectEqual(before_tick, c.wire.items.len);

    c.time = 1700 * ms;
    c.content_write_base = c.writes;
    c.content_active = true;
    const before_content = c.wire.items.len;
    const attempts = c.logical.attempts;
    const ledger = c.ledger_n;
    if (c.fail_frame) |frame| {
        try std.testing.expectError(error.Closed, out.vt.event(out.ctx, c.content));
        try c.pair(ledger, false, false);
        try std.testing.expectEqual(@as(usize, 1), c.faults);
        try std.testing.expect(c.conn.broken and c.conn.close);
        try std.testing.expectEqual(c.content_write_base + frame + 1, c.writes);
        try std.testing.expectEqual(c.failed_next, s.next_ns);
        try std.testing.expectEqual(c.failed_wrote, c.countSession(.wrote));
        try std.testing.expectEqual(@as(usize, if (c.period > 0) frame else 0), c.countSession(.wrote));
        // No Generation exists in this owner fixture: cancellation/drain remains a route-owned witness.
        return;
    }
    try out.vt.event(out.ctx, c.content);
    try std.testing.expect(c.logical.attempts > attempts);
    try c.pair(ledger, false, false);
    try std.testing.expect(c.wire.items.len > before_content);
    try std.testing.expect(std.mem.indexOf(u8, c.wire.items[before_content..], "hello : keepalive") != null);
    const completed = c.writes - c.content_write_base;
    try std.testing.expectEqual(@as(usize, if (c.period > 0) completed else 0), c.countSession(.wrote));
    try std.testing.expectEqual(@as(i96, if (c.period > 0) 2700 * ms else 0), s.next_ns);
    const before = c.wire.items.len;
    c.time = 2000 * ms; try s.tick();
    c.time = 2699 * ms; try s.tick();
    try std.testing.expectEqual(before, c.wire.items.len);
    c.time = 2700 * ms; try s.tick();
    try std.testing.expectEqual(@as(usize, if (c.period > 0) 2 else 0), c.comments);
    if (c.period > 0) try std.testing.expectEqualStrings(comment, c.wire.items[before..]);
}

fn run(c: *Cell) !void {
    var arena: std.heap.ArenaAllocator = .init(a); defer arena.deinit();
    const g = arena.allocator();
    // All driving JSON is bootstrapped before actual Wire.open and before Session.start.
    c.payload = switch (c.input) {
        .role => try choice(g, try chat.deltaOf(g, "role", "assistant"), .null),
        .empty_choices => try parsed(g, "{\"choices\":[]}"),
        .empty_content => try choice(g, try chat.deltaOf(g, "content", ""), .null),
    };
    c.content = try choice(g, try chat.deltaOf(g, "content", "hello : keepalive"), .null);
    c.call_start = try parsed(g, "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"function\":{\"name\":\"f\",\"arguments\":\"{\"}}]},\"finish_reason\":null}]}");
    c.queued_text = try choice(g, try chat.deltaOf(g, "content", "queued"), .null);
    c.call_end = try parsed(g, "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]},\"finish_reason\":null}]}");
    c.terminal = try choice(g, try chat.deltaOf(g, "content", "done"), .{ .string = "stop" });
    c.empty = try parsed(g, "{\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}");
    // Timer/pending no-op cells exclude usage; postfinal deliberately includes separate usage.
    if (c.control != .postfinal) c.empty = try parsed(g, "{\"choices\":[]}");
    var vtable = std.testing.io.vtable.*; vtable.now = Cell.now; vtable.random = Cell.random;
    c.io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };
    const saved = active; active = c; defer active = saved;
    log.init(c.io, true); defer log.init(std.testing.io, true);
    // Private Wire/Reply uses only srv.io/config; no engine, model, request, store or service is run.
    var srv: Server = .{ .gpa = a, .io = c.io, .engine = undefined, .info = .{ .context_window = 128 }, .text = undefined,
        .config = .{ .served_name = "m", .model_ids = &.{"m"}, .keep_warm_s = 0, .sse_keepalive_ns = c.period },
        .keys = null, .metrics = .{ .gpa = a }, .store = .{ .gpa = a }, .arena = .init(a) };
    defer srv.arena.deinit(); defer srv.store.deinit();
    var conn: http.Conn = .{ .fd = -1, .buf = &.{}, .peer = "", .gpa = a, .quiet_log = true,
        .test_transport = .{ .ctx = c, .write_all = Cell.write, .peer_gone = Cell.gone,
            .comment_ops = .{ .ctx = c, .now = Cell.commentNow, .poll = Cell.poll, .send = Cell.send } },
        .stream_activity = .{ .ctx = c, .wrote = Cell.savedWrote } };
    defer conn.out.deinit(a);
    c.conn = &conn;
    var hook: Observation.Hook = .{ .ctx = c, .record = Cell.record };
    const old_hook = Observation.active; Observation.active = &hook; defer Observation.active = old_hook;
    var logical: Logical = .{ .inner = g }; c.logical = &logical;
    // Armed no-op faults are installed only after real startup succeeds.
    const arm = c.arm_noop; c.arm_noop = false;
    armed_after_open = arm; defer armed_after_open = false;
    switch (c.route) {
        .messages => try @import("anthropic.zig").TestFixture.withWire(&srv, &conn, logical.allocator(), driveArmed),
        .responses => try @import("responses.zig").TestFixture.withWire(&srv, &conn, logical.allocator(), driveArmed),
    }
    try std.testing.expect(conn.stream_activity.?.ctx == @as(*anyopaque, @ptrCast(c)));
    try std.testing.expectEqual(@as(usize, if (c.period > 0) 1 else 0), c.countSession(.stop));
}
threadlocal var armed_after_open: bool = false;
fn driveArmed(out: openai.Out) anyerror!void { active.?.arm_noop = armed_after_open; try drive(out); }

test "qualification_d615eebd N05 N07 private translators preterminal no-op timer and actual content reset" {
    var cells: usize = 0;
    for ([_]Route{ .messages, .responses }) |route| for ([_]Input{ .role, .empty_choices, .empty_content }) |input| for ([_]i64{ 0, std.time.ns_per_s }) |period| for ([_]bool{ false, true }) |armed| {
        var c: Cell = .{ .route = route, .input = input, .period = period, .arm_noop = armed };
        defer c.wire.deinit(a);
        try run(&c); cells += 1;
        std.debug.print("qualification_337b0f0b noop route={s} input={s} period={d} armed_not_triggered={} logical_noop_attempts=0 physical_content_frames={d}\n", .{ @tagName(route), @tagName(input), period, armed, c.writes - c.content_write_base });
    };
    try std.testing.expectEqual(@as(usize, 24), cells);
    std.debug.print("qualification_337b0f0b noop_timer_cells={d}\n", .{cells});
}

test "qualification_d615eebd N05 N07 private translators every content-event partial prefix has no failed-write reset" {
    var cells: usize = 0;
    var expected_cells: usize = 0;
    for ([_]Route{ .messages, .responses }) |route| for ([_]Input{ .role, .empty_choices, .empty_content }) |input| for ([_]i64{ 0, std.time.ns_per_s }) |period| {
        var baseline: Cell = .{ .route = route, .input = input, .period = period };
        defer baseline.wire.deinit(a); try run(&baseline);
        const frames = baseline.frames[baseline.content_write_base..baseline.frames_n];
        try std.testing.expect(frames.len > 0);
        for (frames, 0..) |frame, index| {
            expected_cells += frame.len;
            for (0..frame.len) |prefix| {
                var c: Cell = .{ .route = route, .input = input, .period = period, .fail_frame = index, .prefix = prefix };
                defer c.wire.deinit(a); try run(&c);
                // Exact raw byte prefix: only complete preceding frames and the failed partial frame.
                try std.testing.expectEqualStrings(baseline.wire.items[0..frame.offset + prefix], c.wire.items);
                cells += 1;
            }
            std.debug.print("qualification_337b0f0b fault_frame route={s} input={s} period={d} frame={d} prefixes={d}\n", .{ @tagName(route), @tagName(input), period, index, frame.len });
        }
        std.debug.print("qualification_337b0f0b applicable_fault route={s} input={s} period={d} content_frames={d}\n", .{ @tagName(route), @tagName(input), period, frames.len });
    };
    try std.testing.expect(cells > 0);
    try std.testing.expectEqual(expected_cells, cells);
    std.debug.print("qualification_337b0f0b applicable_content_partial_prefix_cells={d} owner_fixture_no_generation=true\n", .{cells});
}

test "qualification_d615eebd N05 N07 pending-call and postfinal empty-choice controls remain distinct" {
    var cells: usize = 0;
    for ([_]Route{ .messages, .responses }) |route| for ([_]i64{ 0, std.time.ns_per_s }) |period| for ([_]Control{ .pending_call, .postfinal }) |control| {
        var c: Cell = .{ .route = route, .input = .empty_choices, .period = period, .control = control };
        defer c.wire.deinit(a); try run(&c); cells += 1;
        std.debug.print("qualification_337b0f0b stateful route={s} period={d} control={s}\n", .{ @tagName(route), period, @tagName(control) });
    };
    try std.testing.expectEqual(@as(usize, 8), cells);
    std.debug.print("qualification_337b0f0b distinct_stateful_terminal_cells={d}\n", .{cells});
}
