//! Native socket-free transport and config tests; actual production algorithm, injected syscall results.
const std = @import("std");
const http = @import("http_conn.zig");
const sse = @import("sse.zig");
const keep = @import("sse_keepalive.zig");
const a = std.testing.allocator;
const Ops = struct {
    time: i96 = 0,
    elapsed: i96 = 0,
    step: usize = 1,
    polls: usize = 0,
    sends: usize = 0,
    written: std.ArrayList(u8) = .empty,
    poll_result: http.PollResult = .out,
    send_result: ?http.SendResult = null,
    activity: usize = 0,
    fail_data: bool = false,
    fn self(ctx: ?*anyopaque) *Ops { return @ptrCast(@alignCast(ctx.?)); }
    fn now(ctx: ?*anyopaque) i96 { return self(ctx).time; }
    fn poll(ctx: ?*anyopaque, fd: std.posix.socket_t, ms: i32) http.PollResult {
        std.debug.assert(fd == -1 and ms > 0 and ms <= 50);
        const o = self(ctx); o.polls += 1; o.time += o.elapsed;
        return o.poll_result;
    }
    fn send(ctx: ?*anyopaque, fd: std.posix.socket_t, bytes: []const u8) http.SendResult {
        std.debug.assert(fd == -1);
        const o = self(ctx); o.sends += 1;
        if (o.send_result) |r| return r;
        const n = @min(o.step, bytes.len);
        o.written.appendSlice(a, bytes[0..n]) catch @panic("allocation");
        return .{ .sent = n };
    }
    fn ops(o: *Ops) http.CommentIo { return .{ .ctx = o, .now = now, .poll = poll, .send = send }; }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const o = self(ctx);
        if (o.fail_data) return error.Closed;
        o.written.appendSlice(a, bytes) catch return error.Closed;
    }
    fn gone(_: *anyopaque) bool { return false; }
    fn wrote(ctx: *anyopaque) void { self(ctx).activity += 1; }
};
fn conn(o: *Ops) http.Conn { return .{ .fd = -1, .buf = &.{}, .peer = "", .gpa = a, .test_transport = .{ .ctx = o, .write_all = Ops.write, .peer_gone = Ops.gone, .comment_ops = o.ops() }, .stream_activity = .{ .ctx = o, .wrote = Ops.wrote } }; }

test "qualification_4533b989 actual comment partial boundary serialization and observer" {
    for (1..14) |step| {
        var o: Ops = .{ .step = step }; defer o.written.deinit(a);
        var c = conn(&o);
        var arena = std.heap.ArenaAllocator.init(a); defer arena.deinit();
        try sse.data(&c, arena.allocator(), .{ .string = ": keepalive content" });
        try sse.comment(&c, std.testing.io);
        try sse.event(&c, arena.allocator(), "typed", .null);
        try sse.done(&c);
        try std.testing.expectEqual(@as(usize, 3), o.activity);
        try std.testing.expectEqualStrings("data: \": keepalive content\"\n\n: keepalive\n\nevent: typed\ndata: null\n\ndata: [DONE]\n\n", o.written.items);
        try std.testing.expect(!c.broken);
    }
}
test "qualification_4533b989 actual bounded frozen clock retry cap" {
    for ([_]http.SendResult{ .again, .interrupted }) |r| {
        var o: Ops = .{ .send_result = r }; defer o.written.deinit(a);
        var c = conn(&o);
        try std.testing.expectError(error.Closed, c.writeCommentWith(o.ops()));
        try std.testing.expectEqual(@as(usize, 64), o.polls);
        try std.testing.expectEqual(@as(usize, 64), o.sends);
        try std.testing.expect(c.broken and c.close);
    }
}
test "qualification_4533b989 actual absolute deadline no progress extension" {
    var o: Ops = .{ .elapsed = 50 * std.time.ns_per_ms }; defer o.written.deinit(a);
    var c = conn(&o);
    try std.testing.expectError(error.Closed, c.writeCommentWith(o.ops()));
    try std.testing.expectEqual(@as(usize, 5), o.polls);
    try std.testing.expectEqual(@as(usize, 4), o.sends);
    try std.testing.expectEqual(@as(i96, 250 * std.time.ns_per_ms), o.time);
    try std.testing.expectEqualStrings(": ke", o.written.items);
    try std.testing.expectError(error.Closed, sse.done(&c));
    try std.testing.expectEqual(@as(usize, 0), o.activity);
}
test "qualification_4533b989 actual failed zero oversized send and failed interrupted timeout poll" {
    for ([_]http.SendResult{ .failed, .{ .sent = 0 }, .{ .sent = 14 } }) |r| {
        var o: Ops = .{ .send_result = r }; defer o.written.deinit(a);
        var c = conn(&o);
        try std.testing.expectError(error.Closed, c.writeCommentWith(o.ops()));
        try std.testing.expectEqual(@as(usize, 1), o.sends);
    }
    for ([_]http.PollResult{ .failed, .interrupted, .timeout }) |r| {
        var o: Ops = .{ .poll_result = r }; defer o.written.deinit(a);
        var c = conn(&o);
        try std.testing.expectError(error.Closed, c.writeCommentWith(o.ops()));
        try std.testing.expectEqual(@as(usize, 0), o.sends);
        try std.testing.expectEqual(@as(usize, if (r == .failed) 1 else 64), o.polls);
    }
}
test "qualification_4533b989 failed data never reports completed activity" {
    var o: Ops = .{ .fail_data = true }; defer o.written.deinit(a);
    var c = conn(&o);
    try std.testing.expectError(error.Closed, sse.done(&c));
    try std.testing.expectEqual(@as(usize, 0), o.activity);
}
var active_ops: ?*Ops = null;
fn fixedNow(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp { return .{ .nanoseconds = active_ops.?.time }; }
test "qualification_4533b989 actual Session wire reset cadence stop observer restoration and failure" {
    var o: Ops = .{ .step = 13 }; defer o.written.deinit(a);
    active_ops = &o; defer active_ops = null;
    var vtable = std.testing.io.vtable.*; vtable.now = fixedNow;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };
    var c = conn(&o);
    var session: keep.Session = .{ .conn = &c, .io = io, .period_ns = std.time.ns_per_s };
    session.start();
    o.time = 700 * std.time.ns_per_ms;
    var arena = std.heap.ArenaAllocator.init(a); defer arena.deinit();
    try sse.event(&c, arena.allocator(), "typed", .null);
    o.time = 1500 * std.time.ns_per_ms; try session.tick();
    try std.testing.expectEqual(@as(usize, 0), o.sends);
    o.time = 1700 * std.time.ns_per_ms; try session.tick();
    try std.testing.expectEqual(@as(usize, 1), o.sends);
    try session.tick(); try std.testing.expectEqual(@as(usize, 1), o.sends);
    o.time = 100 * std.time.ns_per_s; try session.tick();
    try std.testing.expectEqual(@as(usize, 2), o.sends);
    session.stop(); session.stop();
    try std.testing.expect(c.stream_activity.?.ctx == @as(*anyopaque, @ptrCast(&o)));
    o.time = 200 * std.time.ns_per_s; try session.tick();
    try std.testing.expectEqual(@as(usize, 2), o.sends);
    try sse.done(&c); try std.testing.expectEqual(@as(usize, 1), o.activity);
    session.start();
    o.send_result = .failed; o.time += std.time.ns_per_s;
    try std.testing.expectError(error.Closed, session.tick());
    try std.testing.expect(!session.active and c.stream_activity.?.ctx == @as(*anyopaque, @ptrCast(&o)));
    const n = o.sends; try session.tick(); try std.testing.expectEqual(n, o.sends);
}
test { _ = @import("qualification_59878042_native_alloc_test.zig"); _ = @import("qualification_59878042_translated_test.zig"); _ = @import("qualification_59878042_allocator_test.zig"); _ = @import("qualification_59878042_observe_test.zig"); _ = keep; _ = @import("qualification_59878042_route_test.zig"); _ = @import("qualification_59878042_setting_test.zig"); _ = @import("qualification_59878042_lane_test.zig"); _ = @import("qualification_59878042_boundary_test.zig"); _ = @import("qualification_59878042_rich_route_test.zig"); }
