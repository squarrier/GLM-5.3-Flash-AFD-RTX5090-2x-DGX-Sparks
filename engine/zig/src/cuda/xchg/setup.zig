//! TF_AFD_SETUP_MS (qualification_1fee6d78): how long this node's handshake waits for the others (Config.setup_timeout_ms), per process.
//! The attention accepts each expert for that long after its load; an expert retries its connect for that long after
//! its own. Unset keeps Config's 120,000 ms; the window is not in the settings text, so the nodes need not agree on it.
const std = @import("std");
const Config = @import("node.zig").Config;

pub const name = "TF_AFD_SETUP_MS";
/// The window's bounds in ms, both taken: 10 s to 30 min.
pub const min_ms: u32 = 10_000;
pub const max_ms: u32 = 1_800_000;

/// The variable's value: unset (null) keeps Config's window; decimal digits from min_ms to max_ms set it; anything else
/// (empty, a sign, a unit, a space, a separator, junk, out of range) is refused.
pub fn parse(v: ?[]const u8) error{BadAfdSetupMs}!?u32 {
    const s = v orelse return null;
    if (s.len == 0) return error.BadAfdSetupMs;
    for (s) |c| if (!std.ascii.isDigit(c)) return error.BadAfdSetupMs;
    const ms = std.fmt.parseInt(u32, s, 10) catch return error.BadAfdSetupMs;
    if (ms < min_ms or ms > max_ms) return error.BadAfdSetupMs;
    return ms;
}

/// `c` with the window `ms`; null returns `c` as it is. The settings are untouched, so their text is the same either way.
pub fn with(c: Config, ms: ?u32) Config {
    var x = c;
    if (ms) |m| x.setup_timeout_ms = m;
    return x;
}

/// The INFO line a node prints when the window is set (`role` names the node, e.g. "attn" or "afd expert 0"); null unset.
pub fn line(buf: []u8, role: []const u8, ms: ?u32) ?[]const u8 {
    const m = ms orelse return null;
    return std.fmt.bufPrint(buf, "INFO {s}: " ++ name ++ "={d}: the handshake waits {d} ms", .{ role, m, m }) catch null;
}

/// TF_AFD_SETUP_MS read once by a role before anything loads: set, its INFO line; refused, one line naming the variable
/// and error.BadAfdSetupMs (the run then ends `AFD FAILED BadAfdSetupMs`).
pub fn fromEnv(role: []const u8) error{BadAfdSetupMs}!?u32 {
    const v: ?[]const u8 = if (std.c.getenv(name)) |p| std.mem.span(p) else null;
    const ms = parse(v) catch |e| {
        std.debug.print("INFO {s}: " ++ name ++ " takes a whole number of milliseconds from {d} to {d} (or unset), not \"{s}\"\n", .{ role, min_ms, max_ms, v.? });
        return e;
    };
    var b: [192]u8 = undefined;
    if (line(&b, role, ms)) |l| std.debug.print("{s}\n", .{l});
    return ms;
}

extern "c" fn setenv(n: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(n: [*:0]const u8) c_int;

const testing = std.testing;

test "qualification_1fee6d78: TF_AFD_SETUP_MS's parse: unset keeps the window; 10,000 to 1,800,000 ms set it; empty, a sign, a unit, junk or out of range refused" {
    try testing.expectEqual(@as(?u32, null), try parse(null));
    for ([_]struct { []const u8, u32 }{ .{ "10000", 10_000 }, .{ "1800000", 1_800_000 }, .{ "120000", 120_000 }, .{ "300000", 300_000 }, .{ "10001", 10_001 }, .{ "1799999", 1_799_999 }, .{ "0300000", 300_000 } }) |c| {
        try testing.expectEqual(@as(?u32, c[1]), try parse(c[0]));
    }
    const bad = [_][]const u8{ "", "9999", "1800001", "0", "1", "4294967296", "99999999999999999999", "+300000", "-300000", "300000ms", "300s", "300 000", "300,000", "300_000", " 300000", "300000 ", "3e5", "0x493e0", "300000.0", "abc", "on" };
    for (bad) |v| try testing.expectError(error.BadAfdSetupMs, parse(v));
}

test "qualification_1fee6d78: the window through Config: unset is Config's 120,000 ms, set it is the value; the settings and their text are untouched" {
    const qualification_81840057 = "tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1";
    const forms = [_]struct { @import("proto.zig").Settings, []const u8 }{
        .{ .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 }, qualification_81840057 },
        .{ .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1, .pairs = true, .inflight = 2 }, qualification_81840057 ++ " pairs=1 inflight=2" },
        .{ .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1, .pairs = true, .inflight = 2, .seq = true }, qualification_81840057 ++ " pairs=1 inflight=2 seq=1" },
        .{ .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1, .pairs = true, .inflight = 2, .halves = "k4" }, qualification_81840057 ++ " pairs=1 inflight=2 halves=k4" },
    };
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    for (forms) |f| {
        const c: Config = .{ .settings = f[0] };
        try testing.expectEqual(@as(u32, 120_000), c.setup_timeout_ms);
        try testing.expectEqual(c, with(c, null));
        for ([_]?u32{ null, min_ms, 300_000, max_ms }) |w| {
            const x = with(c, w);
            try testing.expectEqual(w orelse 120_000, x.setup_timeout_ms);
            try testing.expectEqualStrings(f[1], try x.settings.text(&a));
            try testing.expectEqualStrings(try c.settings.text(&b), try x.settings.text(&a));
            var y = x;
            y.setup_timeout_ms = c.setup_timeout_ms;
            try testing.expectEqual(c, y); // nothing else moved
        }
    }
}

test "qualification_1fee6d78: the INFO line only when TF_AFD_SETUP_MS is set; fromEnv reads unset, a value and a refusal" {
    var b: [192]u8 = undefined;
    try testing.expect(line(&b, "attn", null) == null);
    try testing.expectEqualStrings("INFO attn: TF_AFD_SETUP_MS=300000: the handshake waits 300000 ms", line(&b, "attn", 300_000).?);
    try testing.expectEqualStrings("INFO afd expert 1: TF_AFD_SETUP_MS=10000: the handshake waits 10000 ms", line(&b, "afd expert 1", 10_000).?);
    defer _ = unsetenv(name);
    _ = unsetenv(name);
    try testing.expectEqual(@as(?u32, null), try fromEnv("attn"));
    _ = setenv(name, "300000", 1);
    try testing.expectEqual(@as(?u32, 300_000), try fromEnv("attn"));
    for ([_][*:0]const u8{ "", "300s", "-1", "9999" }) |v| {
        _ = setenv(name, v, 1);
        try testing.expectError(error.BadAfdSetupMs, fromEnv("attn"));
    }
}
