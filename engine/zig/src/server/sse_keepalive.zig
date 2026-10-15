//! Request-thread SSE comments; no producer, engine callback, allocation or borrowed payload.
const std = @import("std");
const http = @import("http_conn.zig");
const sse = @import("sse.zig");

pub fn secondsNs(raw: ?[]const u8) error{Setting}!i64 {
    const text = std.mem.trim(u8, raw orelse return 0, " \t\r\n");
    if (text.len == 0) return error.Setting;
    const seconds = std.fmt.parseFloat(f64, text) catch return error.Setting;
    if (!std.math.isFinite(seconds) or seconds < 0 or seconds > 300 or (seconds > 0 and seconds < 1)) return error.Setting;
    if (seconds == 0) return 0;
    return @intFromFloat(@ceil(seconds * std.time.ns_per_s));
}

pub fn validateNs(ns: i64) error{Setting}!void {
    if (ns < 0 or ns > 300 * std.time.ns_per_s or (ns > 0 and ns < std.time.ns_per_s)) return error.Setting;
}

pub const Session = struct {
    conn: *http.Conn,
    io: std.Io,
    period_ns: i64,
    next_ns: i96 = 0,
    active: bool = false,
    saved_activity: ?http.StreamActivity = null,

    pub fn start(s: *Session) void {
        std.debug.assert(s.period_ns > 0 and !s.active);
        s.saved_activity = s.conn.stream_activity;
        s.active = true;
        s.conn.stream_activity = .{ .ctx = s, .wrote = wrote };
        s.next_ns = std.Io.Clock.awake.now(s.io).toNanoseconds() + s.period_ns;
        if (comptime @import("builtin").is_test) s.observe(.start);
    }

    fn wrote(ctx: *anyopaque) void {
        const s: *Session = @ptrCast(@alignCast(ctx));
        if (s.active) s.next_ns = std.Io.Clock.awake.now(s.io).toNanoseconds() + s.period_ns;
        if (comptime @import("builtin").is_test) s.observe(.wrote);
    }

    fn observe(s: *Session, phase: if (@import("builtin").is_test) @import("sse_test_observe.zig").Test.SessionPhase else void) void {
        if (comptime @import("builtin").is_test) @import("sse_test_observe.zig").Test.record(.{ .session = .{ .phase = phase, .active = s.active, .next_ns = s.next_ns, .attached = if (s.conn.stream_activity) |h| h.ctx == @as(*anyopaque, @ptrCast(s)) else false } });
    }

    pub fn tick(s: *Session) error{Closed}!void {
        if (!s.active) return;
        if (std.Io.Clock.awake.now(s.io).toNanoseconds() < s.next_ns) return;
        sse.comment(s.conn, s.io) catch {
            s.stop();
            return error.Closed;
        };
        s.next_ns = std.Io.Clock.awake.now(s.io).toNanoseconds() + s.period_ns;
        if (comptime @import("builtin").is_test) s.observe(.comment);
    }

    pub fn stop(s: *Session) void {
        if (!s.active) return;
        s.active = false;
        s.conn.stream_activity = s.saved_activity;
        if (comptime @import("builtin").is_test) s.observe(.stop);
    }
};

test "qualification_4533b989 parser finite default-off range" {
    try std.testing.expectEqual(@as(i64, 0), try secondsNs(null));
    for ([_][]const u8{ "0", "-0", " 0\n" }) |v| try std.testing.expectEqual(@as(i64, 0), try secondsNs(v));
    try std.testing.expectEqual(@as(i64, 1_500_000_000), try secondsNs("1.5"));
    try std.testing.expectEqual(@as(i64, 1_000_000_000), try secondsNs("1e0"));
    try std.testing.expectEqual(@as(i64, 300_000_000_000), try secondsNs("300"));
    for ([_][]const u8{ "", " ", "oops", "nan", "inf", "-inf", "-1", "0.5", "300.1" }) |v| try std.testing.expectError(error.Setting, secondsNs(v));
    for ([_]i64{ -1, 1, 999_999_999, 300_000_000_001 }) |v| try std.testing.expectError(error.Setting, validateNs(v));
    for ([_]i64{ 0, 1_000_000_000, 300_000_000_000 }) |v| try validateNs(v);
}
