//! Actual request-loop allocation cross-products, not a replacement engine host.
const std = @import("std");
const f = @import("qualification_59878042_alloc_fixture.zig");
test "qualification_d615eebd positive actual native loop allocator ownership inventory" {
    for (f.paths) |path| for ([_]f.Stage{ .a1, .a2, .a3, .a4, .g1 }) |stage| {
        const r = try f.nativeInventory(path, std.time.ns_per_s, 2, true, .plain, stage, null);
        try std.testing.expect(r.generated and r.stopped);
        if (stage != .a1 or !std.mem.endsWith(u8, path, "/completions") or std.mem.endsWith(u8, path, "/chat/completions")) try std.testing.expect(r.attempts > 0);
    };
}
test "qualification_d615eebd native unfinished loop role terminal tools allocator every index cross-products" {
    var c: f.Counts = .{};
    for (f.paths) |path| for ([_]i64{ 0, 0, std.time.ns_per_s }, 0..) |period, setting| for ([_]bool{ false, true }) |usage| for ([_]f.NativeKind{ .plain, .reasoning, .tools, .failed, .length, .cancelled, .eos, .stop_string, .submit_busy, .submit_closed }) |kind| for ([_]f.Stage{ .a1, .a2, .a3, .a4, .g1 }) |stage| {
        try f.sweepNative(path, period, setting, usage, kind, stage, &c);
    };
    try std.testing.expect(c.triggered > 0 and c.sentinels == c.baselines and c.zero > 0);
    c.print("native-request");
}

test "qualification_c40d64eb original unfinished tool EOS and fallback policy tail allocation witnesses" {
    var c: f.Counts = .{};
    for (f.paths) |path| for ([_]i64{ 0, 0, std.time.ns_per_s }, 0..) |period, setting| for ([_]bool{ false, true }) |usage| {
        try f.sweepNative(path, period, setting, usage, .unfinished_tool_eos, .a3, &c);
        try f.sweepNative(path, period, setting, usage, .fallback_tools, .a4, &c);
        try f.sweepNative(path, period, setting, usage, .policy_tail, .a4, &c);
    };
    c.print("original-missing-tool-branches");
}

test "qualification_c40d64eb N09 actual translator aliases settings usage and original native scenarios" {
    var c: f.Counts = .{};
    for (f.paths) |path| {
        if (!std.mem.endsWith(u8, path, "/messages") and !std.mem.endsWith(u8, path, "/responses")) continue;
        for ([_]i64{ 0, 0, std.time.ns_per_s }, 0..) |period, setting| for ([_]bool{ false, true }) |usage| for ([_]f.NativeKind{ .plain, .reasoning, .tools, .failed, .length, .cancelled, .eos, .stop_string, .submit_busy, .submit_closed }) |kind| {
            try f.sweepNative(path, period, setting, usage, kind, .a5, &c);
        };
    }
    c.print("native-translator-crossproduct");
}
