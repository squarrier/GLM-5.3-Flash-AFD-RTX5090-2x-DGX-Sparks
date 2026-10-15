//! qualification_17c70e37: TF_AFD_STRIPES in the AFD roles (the switch is the wire's alone): the attention's --dry record as unset's, and
//! both roles refusing a bad value before anything loads. Kept apart so afd_attn.zig and afd_expert.zig do not grow.
const std = @import("std");
const cuda = @import("cuda");
const attn = @import("afd_attn.zig");
const expert = @import("afd_expert.zig");
const cli = @import("tf_glm_afd.zig");
const proto = cuda.xchg.proto;
const testing = std.testing;
const Allocator = std.mem.Allocator;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

// TF_AFD_STRIPES as it was before a test, put back after it: the identity runs set it for the whole test binary
fn saved(buf: *[32:0]u8) ?[:0]const u8 {
    const v = std.mem.span(std.c.getenv("TF_AFD_STRIPES") orelse return null);
    if (v.len >= buf.len) return null;
    @memcpy(buf[0..v.len], v);
    buf[v.len] = 0;
    return buf[0..v.len :0];
}
fn restore(v: ?[:0]const u8) void {
    _ = if (v) |x| setenv("TF_AFD_STRIPES", x.ptr, 1) else unsetenv("TF_AFD_STRIPES");
}

// afd_attn.zig's tmpPath and fixtureConfig, copied (they are private there)
fn tmpPath(a: Allocator, tmp: testing.TmpDir) ![]u8 {
    const cwd = try std.process.currentPathAlloc(testing.io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

fn fixtureConfig(a: Allocator) ![]const u8 {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("afd_attn_fixture"), .{});
    return root.object.get("config_json").?.string;
}

test "qualification_17c70e37: attn first --dry --audit with TF_AFD_STRIPES=8 and =8/4096: its first token, forwards, exchanges, launches and records as unset's (the switch is the wire's alone); 1, 17, 8x and 8/100 refused before anything loads" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try tmpPath(a, tmp);
    const prompt = try a.alloc(u32, 600);
    for (prompt, 0..) |*p, i| p.* = @intCast((7 * i) % 1000);
    const f: cli.Flags = .{ .role = .attn, .verb = .first, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = "-", .capacity = 6152, .audit = true };
    var sb: [32:0]u8 = undefined;
    const was = saved(&sb); // taken now: a defer's expression runs at the end
    defer restore(was);
    _ = unsetenv("TF_AFD_STRIPES");
    const off = try attn.dry(testing.allocator, f, &.{ .name = "B", .prompt = prompt }, try fixtureConfig(a));
    defer testing.allocator.free(off.outcome.ok);
    for ([_][:0]const u8{ "8", "8/4096" }) |v| {
        _ = setenv("TF_AFD_STRIPES", v, 1);
        const on = try attn.dry(testing.allocator, f, &.{ .name = "B", .prompt = prompt }, try fixtureConfig(a));
        defer testing.allocator.free(on.outcome.ok);
        try testing.expectEqualStrings(off.outcome.ok, on.outcome.ok);
        try testing.expectEqual(off.forwards, on.forwards);
        try testing.expectEqual(off.exchanges, on.exchanges);
        try testing.expectEqual(on.exchanges, on.collects);
        try testing.expectEqual(off.launches, on.launches);
        try testing.expectEqual(off.records, on.records);
    }
    for ([_][:0]const u8{ "1", "17", "8x", "8/100" }) |v| {
        _ = setenv("TF_AFD_STRIPES", v, 1);
        try testing.expectError(error.BadAfdStripes, attn.dry(testing.allocator, f, &.{ .name = "B", .prompt = prompt }, try fixtureConfig(a)));
    }
}

test "qualification_17c70e37: TF_AFD_STRIPES in the expert: unset or 0 off; K (2-16) or K/MIN the stripes its settings carry; 1, 17, 8x, 8/100 or 0/4096 refuse the run before anything loads (exit 1)" {
    var sb: [32:0]u8 = undefined;
    const was = saved(&sb); // taken now: a defer's expression runs at the end
    defer restore(was);
    _ = unsetenv("TF_AFD_STRIPES");
    try testing.expectEqual(proto.Stripes{}, try proto.stripesFromEnv("afd expert"));
    for ([_]struct { [:0]const u8, proto.Stripes }{ .{ "0", .{} }, .{ "2", .{ .k = 2 } }, .{ "8", .{ .k = 8 } }, .{ "16", .{ .k = 16 } }, .{ "8/4096", .{ .k = 8, .min = 4096 } } }) |c| {
        _ = setenv("TF_AFD_STRIPES", c[0], 1);
        try testing.expectEqual(c[1], try proto.stripesFromEnv("afd expert"));
    }
    for ([_][:0]const u8{ "1", "17", "8x", "8/100", "0/4096" }) |v| {
        _ = setenv("TF_AFD_STRIPES", v, 1);
        try testing.expectError(error.BadAfdStripes, proto.stripesFromEnv("afd expert"));
        // the command refuses before the load: a model path that does not exist is never opened
        try testing.expectEqual(@as(u8, 1), expert.command(testing.allocator, testing.io, &.{ "--half", "0", "--model", "/nonexistent-qualification_17c70e37", "--triton", "/t", "--master", "127.0.0.1:1" }));
    }
}
