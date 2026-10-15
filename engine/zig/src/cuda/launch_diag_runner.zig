//! CPU-only diagnostic log-capture runner; expected Driver.check logs are asserted, not discarded.
const std = @import("std");
const builtin = @import("builtin");
pub const std_options: std.Options = .{ .logFn = captureLog };
extern fn qualification_0effdff1_capture_log(p: [*]const u8, n: usize) callconv(.c) void;
fn captureLog(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime fmt: []const u8, args: anytype) void {
    _ = level;
    _ = scope;
    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    w.print(fmt ++ "\n", args) catch unreachable;
    qualification_0effdff1_capture_log(w.buffered().ptr, w.buffered().len);
}
pub fn main(init: std.process.Init.Minimal) void {
    std.testing.environ = init.environ;
    std.testing.allocator_instance = .init(std.heap.page_allocator, .{});
    std.testing.io_instance = .init(std.testing.allocator, .{});
    defer std.testing.io_instance.deinit();
    var passed: usize = 0;
    var failed: usize = 0;
    var skipped: usize = 0;
    for (builtin.test_functions) |t| {
        if (t.func()) |_| {
            passed += 1;
            std.debug.print("TEST|{s}|PASS\n", .{t.name});
        } else |e| {
            if (e == error.SkipZigTest) skipped += 1 else failed += 1;
            std.debug.print("TEST|{s}|{s}\n", .{ t.name, @errorName(e) });
            if (@errorReturnTrace()) |trace| std.debug.dumpErrorReturnTrace(trace);
        }
    }
    const leaks = std.testing.allocator_instance.deinit();
    std.debug.print("TOTAL|{d}|{d}|{d}|leaks={d}\n", .{ passed, failed, skipped, leaks });
    if (failed != 0 or skipped != 0 or leaks != 0) std.process.exit(1);
}
