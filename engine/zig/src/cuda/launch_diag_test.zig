//! qualification_b95086f6 CPU-only tests; GPT-6.1 Sol via Hermes. Never opens a real CUDA driver.
const std = @import("std");
const L = @import("launch.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const extended = @hasDecl(L, "describe");
pub const std_options: std.Options = .{ .logFn = log };
var log_buf: [4096]u8 = undefined;
var log_used: usize = 0;
export fn qualification_0effdff1_capture_log(p: [*]const u8, n: usize) callconv(.c) void {
    @memcpy(log_buf[log_used..][0..n], p[0..n]);
    log_used += n;
}
pub fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    _ = level;
    _ = scope;
    var w = std.Io.Writer.fixed(log_buf[log_used..]);
    w.print(fmt ++ "\n", args) catch unreachable;
    log_used += w.buffered().len;
}
var call_buf: [2048]u8 = undefined;
var call_used: usize = 0;
var result: abi.Result = 0;
var error_mode: u8 = 0;
fn record(comptime fmt: []const u8, args: anytype) void {
    var w = std.Io.Writer.fixed(call_buf[call_used..]);
    w.print(fmt, args) catch unreachable;
    call_used += w.buffered().len;
}
fn params(p: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) void {
    record("extra={};", .{extra == null});
    if (p) |a| {
        record("args={x},{d};", .{ @as(*align(1) const u64, @ptrCast(a[0].?)).*, @as(*align(1) const i32, @ptrCast(a[1].?)).* });
    } else record("args=null;", .{});
}
fn plain(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: abi.Stream, p: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
    record("plain;f={x};g={d},{d},{d};b={d},{d},{d};shared={d};s={x};", .{ @intFromPtr(f), gx, gy, gz, bx, by, bz, shared, @intFromPtr(stream) });
    params(p, extra);
    return result;
}
fn ex(c: *const abi.LaunchConfig, f: abi.Function, p: ?[*]?*anyopaque, extra: ?[*]?*anyopaque) callconv(.c) abi.Result {
    record("ex;f={x};g={d},{d},{d};b={d},{d},{d};shared={d};s={x};n={d};", .{ @intFromPtr(f), c.grid_x, c.grid_y, c.grid_z, c.block_x, c.block_y, c.block_z, c.shared_bytes, @intFromPtr(c.stream), c.num_attrs });
    for (c.attrs.?[0..c.num_attrs]) |a| record("attr={x};", .{std.fmt.bytesToHex(std.mem.asBytes(&a), .lower)});
    params(p, extra);
    return result;
}
fn errorName(r: abi.Result, out: *?[*:0]const u8) callconv(.c) abi.Result {
    record("errorName={d};", .{r});
    out.* = if (error_mode == 2) null else "MOCK_ERROR";
    return if (error_mode == 1) 1 else 0;
}
fn errorText(r: abi.Result, out: *?[*:0]const u8) callconv(.c) abi.Result {
    record("errorText={d};", .{r});
    out.* = if (error_mode == 2) null else "mock text";
    return if (error_mode == 1) 1 else 0;
}
fn enabled() bool {
    const p = std.c.getenv("TF_CUDA_LAUNCH_DIAG") orelse return false;
    return std.mem.eql(u8, std.mem.span(p), "1");
}
const cfgs = [_]L.Config{
    .{ .grid = .{ .x = 70000, .y = 2, .z = 3 }, .block = .{ .x = 128, .y = 2 }, .shared = 49152 },
    .{ .grid = .{ .x = 4 }, .block = .{ .x = 64 }, .pdl = true },
    .{ .grid = .{ .x = 6 }, .block = .{ .x = 32 }, .cooperative = true },
    .{ .grid = .{ .x = 8 }, .block = .{ .x = 64 }, .cluster = .{ .x = 2, .y = 3, .z = 4 } },
    .{ .grid = .{ .x = 10 }, .block = .{ .x = 128 }, .shared = 123, .cluster = .{ .x = 2 }, .cluster_spread = true, .pdl = true, .cooperative = true },
    .{ .grid = .{}, .block = .{}, .cluster_spread = true },
};
fn expected(r: abi.Result) ?Error {
    return switch (r) {
        0 => null,
        2 => error.OutOfDeviceMemory,
        500 => error.NotFound,
        600 => error.NotReady,
        else => error.CudaFailed,
    };
}
fn runCase(ci: usize, r: abi.Result, empty: bool, mode: u8, fixture: []const u8) !void {
    var path_buf: [256]u8 = undefined;
    var path_writer = std.Io.Writer.fixed(&path_buf);
    try path_writer.print("/fixtures/{s}.so\x00", .{fixture});
    const path = path_buf[0 .. path_writer.buffered().len - 1 :0];
    var lib = try std.DynLib.open(path);
    defer lib.close();
    const count = lib.lookup(*const fn () callconv(.c) c_int, "mock_count").?;
    const reset = lib.lookup(*const fn () callconv(.c) void, "mock_reset").?;
    reset();
    var api: abi.Api = undefined;
    api.cuLaunchKernel = plain;
    api.cuLaunchKernelEx = ex;
    api.cuGetErrorName = errorName;
    api.cuGetErrorString = errorText;
    const d: Driver = .{ .lib = lib, .api = api };
    var args: L.Args = .{};
    if (!empty) {
        args.add(@as(u64, 0x1122334455667788));
        args.add(@as(i32, -3));
    }
    result = r;
    error_mode = mode;
    log_used = 0;
    call_used = 0;
    const actual: ?Error = if (L.launch(.{ .d = &d, .handle = @ptrFromInt(0x1234) }, cfgs[ci], .{ .d = &d, .handle = @ptrFromInt(0x5678) }, &args)) |_| null else |e| e;
    try std.testing.expectEqual(expected(r), actual);
    const control = std.c.getenv("qualification_93e79013_LAUNCH_CONTROL") != null;
    const diagnostics = !control and enabled() and r != 0;
    try std.testing.expectEqual(@as(c_int, if (diagnostics and !std.mem.eql(u8, fixture, "absent")) 1 else 0), count());
    if (diagnostics) {
        try std.testing.expect(std.mem.indexOf(u8, log_buf[0..log_used], "kernel ") != null);
        try std.testing.expect(std.mem.indexOf(u8, log_buf[0..log_used], "grid ") != null);
        try std.testing.expect(log_used <= 450);
        if (std.mem.eql(u8, fixture, "absent") or std.mem.eql(u8, fixture, "failed") or std.mem.eql(u8, fixture, "null")) try std.testing.expect(std.mem.indexOf(u8, log_buf[0..log_used], "kernel ?, grid ") != null);
        if (std.mem.eql(u8, fixture, "long")) try std.testing.expect(std.mem.indexOf(u8, log_buf[0..log_used], "XXXX?XXX") != null);
    } else if (r == 0 or r == 600) try std.testing.expectEqual(@as(usize, 0), log_used);
    var call_hex: [4096]u8 = undefined;
    var log_hex: [8192]u8 = undefined;
    const digits = "0123456789abcdef";
    for (call_buf[0..call_used], 0..) |b, i| {
        call_hex[i * 2] = digits[b >> 4];
        call_hex[i * 2 + 1] = digits[b & 15];
    }
    for (log_buf[0..log_used], 0..) |b, i| {
        log_hex[i * 2] = digits[b >> 4];
        log_hex[i * 2 + 1] = digits[b & 15];
    }
    std.debug.print("\nCASE|{d}|{d}|{}|{d}|{s}|{s}|{s}|{s}\n", .{ ci, r, empty, mode, fixture, if (actual) |e| @errorName(e) else "OK", call_hex[0 .. call_used * 2], log_hex[0 .. log_used * 2] });
}
test "plain and extended result operand attribute and error-log matrix" {
    for (cfgs, 0..) |_, ci| for ([_]abi.Result{ 0, 1, 2, 3, 4, 500, 600, 700, 719, 12345 }) |r| for ([_]bool{ false, true }) |empty| for (0..3) |mode| try runCase(ci, r, empty, @intCast(mode), "name");
}
test "optional absent failed null and long names" {
    for ([_][]const u8{ "absent", "failed", "null", "long" }) |fixture| for ([_]usize{ 0, 4 }) |ci| for ([_]abi.Result{ 0, 2, 600 }) |r| try runCase(ci, r, false, 0, fixture);
}
test "validation refuses before every launch" {
    var api: abi.Api = undefined;
    api.cuLaunchKernel = plain;
    api.cuLaunchKernelEx = ex;
    const d: Driver = .{ .lib = undefined, .api = api };
    for ([_]L.Config{
        .{ .grid = .{ .x = 0 }, .block = .{} },
        .{ .grid = .{}, .block = .{ .y = 0 }, .pdl = true },
        .{ .grid = .{}, .block = .{ .x = 1025 }, .cooperative = true },
    }) |cfg| {
        var a: L.Args = .{};
        call_used = 0;
        log_used = 0;
        try std.testing.expectError(error.Invalid, L.launch(.{ .d = &d, .handle = null }, cfg, .{ .d = &d, .handle = null }, &a));
        try std.testing.expectEqual(@as(usize, 0), call_used);
        try std.testing.expectEqual(@as(usize, 0), log_used);
    }
}
test "diagnostics bounded and complete even with long symbols or short buffers" {
    if (comptime !extended) return;
    var name: [8192]u8 = @splat('X');
    name[4] = '\n';
    var buf: [320]u8 = undefined;
    const text = L.describe(&buf, &name, cfgs[4]);
    try std.testing.expect(text.len <= buf.len);
    for ([_][]const u8{ "grid 10x1x1", "block 128x1x1", "shared 123 B", "cluster 2x1x1", "pdl", "cooperative", "spread" }) |part| try std.testing.expect(std.mem.indexOf(u8, text, part) != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, text, '\n') == null);
    for (0..33) |n| {
        @memset(&buf, 0xa5);
        try std.testing.expect(L.describe(buf[0..n], &name, cfgs[4]).len <= n);
        for (buf[n..]) |b| try std.testing.expectEqual(@as(u8, 0xa5), b);
    }
}
test "startup optional symbol absent and missing mandatory symbol preserve behavior" {
    log_used = 0;
    var d = try Driver.openPath("/fixtures/driver-absent.so");
    defer d.close();
    const count = d.lib.lookup(*const fn () callconv(.c) c_int, "mock_init_count").?;
    try std.testing.expectEqual(@as(c_int, 1), count());
    try std.testing.expectEqual(@as(usize, 0), log_used);
    try std.testing.expectError(error.MissingSymbol, Driver.openPath("/fixtures/name.so"));
    try std.testing.expect(std.mem.indexOf(u8, log_buf[0..log_used], "has no cuInit") != null);
    std.debug.print("STARTUP|success-init1|missing-cuInit|{d}\n", .{log_used});
}

test "direct Driver.check keeps original error classes" {
    var api: abi.Api = undefined;
    api.cuGetErrorName = errorName;
    api.cuGetErrorString = errorText;
    const d: Driver = .{ .lib = undefined, .api = api };
    for ([_]abi.Result{ 0, 1, 2, 4, 500, 600, 700, 12345 }) |r| {
        call_used = 0;
        log_used = 0;
        error_mode = 0;
        const actual: ?Error = if (d.check(r, "direct")) |_| null else |e| e;
        try std.testing.expectEqual(expected(r), actual);
    }
}

test "ABI layouts and mandatory driver table stay frozen" {
    std.debug.print("ABI|Driver={d},{d}|Api={d},{d}|Config={d},{d}|Args={d},{d}|LaunchConfig={d},{d}|Attribute={d},{d}\n", .{ @sizeOf(Driver), @alignOf(Driver), @sizeOf(abi.Api), @alignOf(abi.Api), @sizeOf(L.Config), @alignOf(L.Config), @sizeOf(L.Args), @alignOf(L.Args), @sizeOf(abi.LaunchConfig), @alignOf(abi.LaunchConfig), @sizeOf(abi.LaunchAttribute), @alignOf(abi.LaunchAttribute) });
    try std.testing.expectEqual(@as(usize, 69), @typeInfo(abi.Api).@"struct".field_names.len);
}

test "new diagnostic API exists (tests-first red witness)" {
    try std.testing.expect(extended or std.c.getenv("qualification_93e79013_LAUNCH_CONTROL") != null);
}
