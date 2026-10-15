//! qualification_4533b989 N07/N10 socket-free witnesses through actual SSE and raw comment wrappers.
//! GPT-6.1 Sol (OpenAI Codex), via Hermes; inherited TensorFold/SSE credits unchanged.
const std = @import("std");
const http = @import("http_conn.zig");
const sse = @import("sse.zig");
const keep = @import("sse_keepalive.zig");
const a = std.testing.allocator;
const frame = ": keepalive\n\n";
const Tag = enum { data, comment, event, done };
const Poll = struct {
    rc: c_int = 1,
    revents: i16 = std.posix.POLL.OUT,
    err: std.posix.E = .SUCCESS,
    advance_ns: i96 = 0,
};
const Send = struct {
    rc: ?isize = null,
    err: std.posix.E = .SUCCESS,
    advance_ns: i96 = 0,
};

const Fixture = struct {
    time: i96 = 0,
    polls: usize = 0,
    sends: usize = 0,
    writes: usize = 0,
    activity: usize = 0,
    notified_len: usize = 0,
    inside_write: bool = false,
    comment_offset: usize = 0,
    poll_plan: []const Poll = &.{},
    send_plan: []const Send = &.{},
    poll_default: Poll = .{},
    send_default: Send = .{},
    send_chunk: usize = frame.len,
    data_chunk: usize = 1,
    data_fail_at: ?usize = null,
    wire: std.ArrayList(u8) = .empty,
    tags: std.ArrayList(Tag) = .empty,
    timeouts: std.ArrayList(i32) = .empty,

    fn deinit(f: *Fixture) void {
        f.wire.deinit(a);
        f.tags.deinit(a);
        f.timeouts.deinit(a);
    }
    fn self(ctx: *anyopaque) *Fixture {
        return @ptrCast(@alignCast(ctx));
    }
    fn append(f: *Fixture, bytes: []const u8, tag: Tag) void {
        f.wire.appendSlice(a, bytes) catch @panic("fixture allocation");
        for (bytes) |_| f.tags.append(a, tag) catch @panic("fixture allocation");
    }
    fn poll(ctx: *anyopaque, fds: [*]std.posix.pollfd, count: usize, ms: i32) c_int {
        const f = self(ctx);
        std.debug.assert(count == 1 and fds[0].fd == -1);
        std.debug.assert(fds[0].events == std.posix.POLL.OUT and fds[0].revents == 0);
        std.debug.assert(ms > 0 and ms <= 50 and !f.inside_write);
        f.timeouts.append(a, ms) catch @panic("fixture allocation");
        const step = if (f.polls < f.poll_plan.len) f.poll_plan[f.polls] else f.poll_default;
        f.polls += 1;
        f.time += step.advance_ns;
        fds[0].revents = step.revents;
        std.posix.system._errno().* = @intCast(@backingInt(step.err));
        return step.rc;
    }
    fn sendto(ctx: *anyopaque, fd: std.posix.socket_t, bytes: [*]const u8, len: usize, flags: u32, dest: ?*const std.posix.sockaddr, dest_len: std.posix.socklen_t) isize {
        const f = self(ctx);
        std.debug.assert(fd == -1 and flags == std.posix.MSG.DONTWAIT);
        std.debug.assert(dest == null and dest_len == 0 and !f.inside_write);
        std.debug.assert(std.mem.eql(u8, frame[f.comment_offset..], bytes[0..len]));
        const step = if (f.sends < f.send_plan.len) f.send_plan[f.sends] else f.send_default;
        f.sends += 1;
        f.time += step.advance_ns;
        const rc = step.rc orelse @as(isize, @intCast(@min(f.send_chunk, len)));
        if (rc > 0 and rc <= @as(isize, @intCast(len))) {
            const n: usize = @intCast(rc);
            f.append(bytes[0..n], .comment);
            f.comment_offset += n;
        }
        std.posix.system._errno().* = @intCast(@backingInt(step.err));
        return rc;
    }
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const f = self(ctx);
        std.debug.assert(!f.inside_write);
        f.inside_write = true;
        defer f.inside_write = false;
        f.writes += 1;
        const old_activity = f.activity;
        const tag: Tag = if (std.mem.startsWith(u8, bytes, "event: ")) .event else if (std.mem.eql(u8, bytes, "data: [DONE]\n\n")) .done else .data;
        var offset: usize = 0;
        while (offset < bytes.len) {
            std.debug.assert(f.activity == old_activity);
            if (f.data_fail_at) |boundary| if (offset == boundary) return error.Closed;
            var n = @min(f.data_chunk, bytes.len - offset);
            if (f.data_fail_at) |boundary| n = @min(n, boundary - offset);
            std.debug.assert(n > 0);
            f.append(bytes[offset .. offset + n], tag);
            offset += n;
        }
        std.debug.assert(f.activity == old_activity);
    }
    fn gone(_: *anyopaque) bool {
        @panic("boundary witness must not probe a peer");
    }
    fn wrote(ctx: *anyopaque) void {
        const f = self(ctx);
        std.debug.assert(!f.inside_write);
        f.activity += 1;
        f.notified_len = f.wire.items.len;
    }
    fn conn(f: *Fixture) http.Conn {
        return .{
            .fd = -1,
            .buf = &.{},
            .peer = "socket-free-boundary",
            .gpa = a,
            .close = false,
            .test_transport = .{
                .ctx = f,
                .write_all = write,
                .peer_gone = gone,
                .raw_comment_syscalls = .{ .ctx = f, .poll = poll, .sendto = sendto },
            },
            .stream_activity = .{ .ctx = f, .wrote = wrote },
        };
    }
};

// Preserve testing Io userdata; only the synchronous clock consults the active fixture.
var active: ?*Fixture = null;
fn now(_: ?*anyopaque, _: std.Io.Clock) std.Io.Timestamp {
    return .{ .nanoseconds = active.?.time };
}
fn fixtureIo(f: *Fixture, vtable: *std.Io.VTable) std.Io {
    std.debug.assert(active == null);
    active = f;
    vtable.* = std.testing.io.vtable.*;
    vtable.now = now;
    return .{ .userdata = std.testing.io.userdata, .vtable = vtable };
}
fn expectTags(f: *const Fixture, start: usize, end: usize, tag: Tag) !void {
    try std.testing.expectEqual(f.wire.items.len, f.tags.items.len);
    for (f.tags.items[start..end]) |actual| try std.testing.expectEqual(tag, actual);
}

const data_frame = "data: \": keepalive content\"\n\n";
const event_frame = "event: typed\ndata: null\n\n";
const done_frame = "data: [DONE]\n\n";
fn emit(tag: Tag, c: *http.Conn, allocator: std.mem.Allocator) !void {
    switch (tag) {
        .data => try sse.data(c, allocator, .{ .string = ": keepalive content" }),
        .event => try sse.event(c, allocator, "typed", .null),
        .done => try sse.done(c),
        .comment => unreachable,
    }
}

test "qualification_4533b989 N07 raw success splits 0..12 serialize DATA comment typed event DONE" {
    try std.testing.expectEqual(@as(usize, 13), frame.len);
    for (0..frame.len) |boundary| {
        const plan = [_]Send{.{ .rc = @intCast(boundary) }};
        var f: Fixture = .{ .send_plan = if (boundary == 0) &.{} else &plan };
        defer f.deinit();
        var vtable: std.Io.VTable = undefined;
        const io = fixtureIo(&f, &vtable);
        defer active = null;
        var c = f.conn();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        try emit(.data, &c, arena.allocator());
        try std.testing.expectEqual(@as(usize, 1), f.activity);
        try std.testing.expectEqual(data_frame.len, f.notified_len);
        try sse.comment(&c, io);
        try std.testing.expectEqual(@as(usize, 1), f.activity);
        try std.testing.expectEqual(data_frame.len, f.notified_len);
        try emit(.event, &c, arena.allocator());
        try emit(.done, &c, arena.allocator());
        try std.testing.expectEqualStrings(data_frame ++ frame ++ event_frame ++ done_frame, f.wire.items);
        try expectTags(&f, 0, data_frame.len, .data);
        try expectTags(&f, data_frame.len, data_frame.len + frame.len, .comment);
        try expectTags(&f, data_frame.len + frame.len, data_frame.len + frame.len + event_frame.len, .event);
        try expectTags(&f, data_frame.len + frame.len + event_frame.len, f.wire.items.len, .done);
        try std.testing.expectEqual(@as(usize, 3), f.activity);
        try std.testing.expectEqual(f.wire.items.len, f.notified_len);
        try std.testing.expectEqual(@as(usize, if (boundary == 0) 1 else 2), f.sends);
        try std.testing.expectEqual(f.sends, f.polls);
        try std.testing.expect(!c.broken and !c.close);
    }
}

test "qualification_4533b989 N07 N10 EPIPE at every prefix 0..12 stops Session and forbids tails" {
    for (0..frame.len) |boundary| {
        const plan = [_]Send{ .{ .rc = @intCast(boundary) }, .{ .rc = -1, .err = .PIPE } };
        var f: Fixture = .{ .send_plan = if (boundary == 0) plan[1..] else &plan };
        defer f.deinit();
        var vtable: std.Io.VTable = undefined;
        const io = fixtureIo(&f, &vtable);
        defer active = null;
        var c = f.conn();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        try emit(.data, &c, arena.allocator());
        var session: keep.Session = .{ .conn = &c, .io = io, .period_ns = std.time.ns_per_s };
        session.start();
        defer session.stop();
        f.time = std.time.ns_per_s;
        try std.testing.expectError(error.Closed, session.tick());
        try std.testing.expect(!session.active and c.broken and c.close);
        try std.testing.expect(c.stream_activity.?.ctx == @as(*anyopaque, @ptrCast(&f)));
        try std.testing.expect(c.stream_activity.?.wrote == Fixture.wrote);
        try std.testing.expectEqualStrings(data_frame, f.wire.items[0..data_frame.len]);
        try std.testing.expectEqualStrings(frame[0..boundary], f.wire.items[data_frame.len..]);
        try expectTags(&f, data_frame.len, f.wire.items.len, .comment);
        const polls = f.polls;
        const sends = f.sends;
        const writes = f.writes;
        for ([_]Tag{ .data, .event, .done }) |tag| try std.testing.expectError(error.Closed, emit(tag, &c, arena.allocator()));
        try session.tick();
        try std.testing.expectEqual(polls, f.polls);
        try std.testing.expectEqual(sends, f.sends);
        try std.testing.expectEqual(writes, f.writes);
        try std.testing.expectEqual(@as(usize, 1), f.activity);
        try std.testing.expectEqual(data_frame.len, f.notified_len);
        try std.testing.expectEqual(data_frame.len + boundary, f.wire.items.len);
    }
}

test "qualification_4533b989 N07 DATA typed event DONE observer only after every complete write" {
    for ([_]Tag{ .data, .event, .done }) |tag| {
        const expected: []const u8 = switch (tag) {
            .data => data_frame,
            .event => event_frame,
            .done => done_frame,
            .comment => unreachable,
        };
        for (0..expected.len + 1) |boundary| {
            var f: Fixture = .{ .data_fail_at = if (boundary < expected.len) boundary else null };
            defer f.deinit();
            var c = f.conn();
            var arena = std.heap.ArenaAllocator.init(a);
            defer arena.deinit();
            if (boundary < expected.len) {
                try std.testing.expectError(error.Closed, emit(tag, &c, arena.allocator()));
                try std.testing.expectEqual(@as(usize, 0), f.activity);
                try std.testing.expect(c.broken and c.close);
            } else {
                try emit(tag, &c, arena.allocator());
                try std.testing.expectEqual(@as(usize, 1), f.activity);
                try std.testing.expectEqual(expected.len, f.notified_len);
                try std.testing.expect(!c.broken);
            }
            try std.testing.expectEqualStrings(expected[0..boundary], f.wire.items);
            try expectTags(&f, 0, boundary, tag);
            try std.testing.expectEqual(@as(usize, 0), f.polls);
            try std.testing.expectEqual(@as(usize, 0), f.sends);
        }
    }
}

test "qualification_4533b989 N10 raw poll ERR HUP NVAL take precedence over OUT and errno fails" {
    for ([_]i16{ std.posix.POLL.ERR, std.posix.POLL.HUP, std.posix.POLL.NVAL }) |bit| {
        for ([_]i16{ 0, std.posix.POLL.OUT }) |out| {
            var f: Fixture = .{ .poll_default = .{ .revents = bit | out } };
            defer f.deinit();
            var vtable: std.Io.VTable = undefined;
            const io = fixtureIo(&f, &vtable);
            defer active = null;
            var c = f.conn();
            try std.testing.expectError(error.Closed, sse.comment(&c, io));
            try std.testing.expectEqual(@as(usize, 1), f.polls);
            try std.testing.expectEqual(@as(usize, 0), f.sends);
            try std.testing.expect(c.broken and c.close);
        }
    }
    var f: Fixture = .{ .poll_default = .{ .rc = -1, .err = .BADF } };
    defer f.deinit();
    var vtable: std.Io.VTable = undefined;
    const io = fixtureIo(&f, &vtable);
    defer active = null;
    var c = f.conn();
    try std.testing.expectError(error.Closed, sse.comment(&c, io));
    try std.testing.expectEqual(@as(usize, 1), f.polls);
    try std.testing.expectEqual(@as(usize, 0), f.sends);
}

test "qualification_4533b989 N10 raw poll zero absent OUT and EINTR consume frozen 64 attempt cap" {
    for ([_]Poll{
        .{ .rc = 0, .revents = 0 },
        .{ .revents = 0 },
        .{ .revents = std.posix.POLL.IN },
        .{ .rc = -1, .err = .INTR },
    }) |step| {
        var f: Fixture = .{ .poll_default = step };
        defer f.deinit();
        var vtable: std.Io.VTable = undefined;
        const io = fixtureIo(&f, &vtable);
        defer active = null;
        var c = f.conn();
        try std.testing.expectError(error.Closed, sse.comment(&c, io));
        try std.testing.expectEqual(@as(usize, 64), f.polls);
        try std.testing.expectEqual(@as(usize, 0), f.sends);
        try std.testing.expectEqual(@as(i96, 0), f.time);
        try std.testing.expect(c.broken and c.close);
    }
}

test "qualification_4533b989 N10 raw poll timeout EINTR then OUT and send AGAIN EINTR partial succeed" {
    const polls = [_]Poll{ .{ .rc = 0, .revents = 0 }, .{ .rc = -1, .err = .INTR } };
    const sends = [_]Send{ .{ .rc = -1, .err = .AGAIN }, .{ .rc = -1, .err = .INTR }, .{ .rc = 3 } };
    var f: Fixture = .{ .poll_plan = &polls, .send_plan = &sends };
    defer f.deinit();
    var vtable: std.Io.VTable = undefined;
    const io = fixtureIo(&f, &vtable);
    defer active = null;
    var c = f.conn();
    try sse.comment(&c, io);
    try std.testing.expectEqualStrings(frame, f.wire.items);
    try std.testing.expectEqual(@as(usize, 6), f.polls);
    try std.testing.expectEqual(@as(usize, 4), f.sends);
    try std.testing.expectEqual(@as(usize, 0), f.activity);
    try std.testing.expect(!c.broken);
}

test "qualification_4533b989 N10 raw send zero oversized EPIPE and other errno fail immediately" {
    for ([_]Send{ .{ .rc = 0 }, .{ .rc = 14 }, .{ .rc = -1, .err = .PIPE }, .{ .rc = -1, .err = .IO } }) |step| {
        var f: Fixture = .{ .send_default = step };
        defer f.deinit();
        var vtable: std.Io.VTable = undefined;
        const io = fixtureIo(&f, &vtable);
        defer active = null;
        var c = f.conn();
        try std.testing.expectError(error.Closed, sse.comment(&c, io));
        try std.testing.expectEqual(@as(usize, 1), f.polls);
        try std.testing.expectEqual(@as(usize, 1), f.sends);
        try std.testing.expectEqual(@as(usize, 0), f.wire.items.len);
        try std.testing.expect(c.broken and c.close);
    }
}

test "qualification_4533b989 N10 raw send frozen AGAIN EINTR cap includes partial progress" {
    for ([_]std.posix.E{ .AGAIN, .INTR }) |err| {
        const initial = [_]Send{.{ .rc = 1 }};
        var f: Fixture = .{ .send_plan = &initial, .send_default = .{ .rc = -1, .err = err } };
        defer f.deinit();
        var vtable: std.Io.VTable = undefined;
        const io = fixtureIo(&f, &vtable);
        defer active = null;
        var c = f.conn();
        try std.testing.expectError(error.Closed, sse.comment(&c, io));
        try std.testing.expectEqual(@as(usize, 64), f.polls);
        try std.testing.expectEqual(@as(usize, 64), f.sends);
        try std.testing.expectEqualStrings(":", f.wire.items);
        try std.testing.expectEqual(@as(i96, 0), f.time);
        try std.testing.expect(c.broken and c.close);
    }
}

test "qualification_4533b989 N10 raw deadline 250ms not extended by progress or poll retries" {
    for ([_]Poll{
        .{ .advance_ns = 50 * std.time.ns_per_ms },
        .{ .rc = 0, .revents = 0, .advance_ns = 50 * std.time.ns_per_ms },
        .{ .rc = -1, .err = .INTR, .advance_ns = 50 * std.time.ns_per_ms },
    }) |step| {
        var f: Fixture = .{ .poll_default = step, .send_chunk = 1 };
        defer f.deinit();
        var vtable: std.Io.VTable = undefined;
        const io = fixtureIo(&f, &vtable);
        defer active = null;
        var c = f.conn();
        try std.testing.expectError(error.Closed, sse.comment(&c, io));
        try std.testing.expectEqual(@as(usize, 5), f.polls);
        try std.testing.expectEqual(@as(usize, if (step.rc == 1) 4 else 0), f.sends);
        try std.testing.expectEqual(@as(i96, 250 * std.time.ns_per_ms), f.time);
        try std.testing.expectEqualStrings(if (step.rc == 1) ": ke" else "", f.wire.items);
        try std.testing.expect(c.broken and c.close);
    }
}

test "qualification_4533b989 N10 raw remaining poll budget rounds up and checks deadline before send" {
    const polls = [_]Poll{ .{ .advance_ns = 249_500_000 }, .{ .advance_ns = 500_000 } };
    var f: Fixture = .{ .poll_plan = &polls, .send_chunk = 1 };
    defer f.deinit();
    var vtable: std.Io.VTable = undefined;
    const io = fixtureIo(&f, &vtable);
    defer active = null;
    var c = f.conn();
    try std.testing.expectError(error.Closed, sse.comment(&c, io));
    try std.testing.expectEqualSlices(i32, &.{ 50, 1 }, f.timeouts.items);
    try std.testing.expectEqual(@as(usize, 1), f.sends);
    try std.testing.expectEqualStrings(":", f.wire.items);
    try std.testing.expectEqual(@as(i96, 250_000_000), f.time);
}

test "qualification_4533b989 N10 raw send progress and interruptions never extend absolute deadline" {
    for ([_]std.posix.E{ .SUCCESS, .AGAIN, .INTR }) |err| {
        var f: Fixture = .{ .send_chunk = 1, .send_default = .{ .rc = if (err == .SUCCESS) null else -1, .err = err, .advance_ns = 50 * std.time.ns_per_ms } };
        defer f.deinit();
        var vtable: std.Io.VTable = undefined;
        const io = fixtureIo(&f, &vtable);
        defer active = null;
        var c = f.conn();
        try std.testing.expectError(error.Closed, sse.comment(&c, io));
        try std.testing.expectEqual(@as(usize, 5), f.polls);
        try std.testing.expectEqual(@as(usize, 5), f.sends);
        try std.testing.expectEqualStrings(if (err == .SUCCESS) ": kee" else "", f.wire.items);
        try std.testing.expectEqual(@as(i96, 250_000_000), f.time);
        try std.testing.expect(c.broken and c.close);
    }
}

test "qualification_4533b989 N10 Session refresh after raw comment completion no data notification or burst" {
    var f: Fixture = .{ .send_default = .{ .advance_ns = 40 * std.time.ns_per_ms } };
    defer f.deinit();
    var vtable: std.Io.VTable = undefined;
    const io = fixtureIo(&f, &vtable);
    defer active = null;
    var c = f.conn();
    var session: keep.Session = .{ .conn = &c, .io = io, .period_ns = std.time.ns_per_s };
    session.start();
    defer session.stop();
    f.time = std.time.ns_per_s;
    try session.tick();
    try std.testing.expectEqual(@as(i96, 2_040_000_000), session.next_ns);
    try std.testing.expectEqual(@as(usize, 0), f.activity);
    try session.tick();
    try std.testing.expectEqual(@as(usize, 1), f.sends);
    f.time = 100 * std.time.ns_per_s;
    f.comment_offset = 0;
    try session.tick();
    try std.testing.expectEqual(@as(usize, 2), f.sends);
    try std.testing.expectEqualStrings(frame ++ frame, f.wire.items);
    try std.testing.expectEqual(@as(i96, 101_040_000_000), session.next_ns);
    session.stop();
    try std.testing.expect(c.stream_activity.?.wrote == Fixture.wrote);
}
