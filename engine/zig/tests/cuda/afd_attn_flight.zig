//! qualification_c533375f: tf-glm-afd attn first --dry with TF_AFD_INFLIGHT=2 (two requests in flight a link inside a pair,
//! evidence/qualification_0f991257/WIRE.md §12) on qualification_0f991257's G-dry-long prompt: the same first token, forwards, exchanges, launches and audit
//! records as TF_AFD_PAIRS=1 alone (the same launches, B's next request earlier), and the switch's refusals before
//! anything loads (TF_AFD_INFLIGHT=2 without the pairs; any value but 1 or 2).
const std = @import("std");
const cli = @import("tf_glm_afd.zig");
const attn = @import("afd_attn.zig");

const testing = std.testing;
const J = std.json.Value;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "qualification_c533375f: attn first --dry --audit with TF_AFD_PAIRS=1 TF_AFD_INFLIGHT=2 on G-dry-long's prompt (a pair of 2,048 + 2,048, then 904 alone): its first token, forwards, exchanges, launches and records as the pairs' alone; TF_AFD_INFLIGHT=2 without the pairs, and 3, refused before anything loads" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try std.process.currentPathAlloc(testing.io, a); // std.testing.tmpDir parents the dir at <cwd>/.zig-cache/tmp
    const out = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    const config = (try std.json.parseFromSliceLeaky(J, a, @embedFile("afd_attn_fixture"), .{})).object.get("config_json").?.string;
    const prompt = try a.alloc(u32, 5000);
    for (prompt, 0..) |*p, i| p.* = @intCast((7 * i) % 1000);
    const f: cli.Flags = .{ .role = .attn, .verb = .first, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = "-", .capacity = 6152, .audit = true };
    const t: attn.Tokens = .{ .name = "B", .prompt = prompt };
    defer _ = unsetenv("TF_AFD_PAIRS");
    defer _ = unsetenv("TF_AFD_INFLIGHT");
    const off = try attn.dry(testing.allocator, f, &t, config);
    defer testing.allocator.free(off.outcome.ok);
    _ = setenv("TF_AFD_PAIRS", "1", 1);
    const one = try attn.dry(testing.allocator, f, &t, config);
    defer testing.allocator.free(one.outcome.ok);
    _ = setenv("TF_AFD_INFLIGHT", "2", 1);
    const two = try attn.dry(testing.allocator, f, &t, config);
    defer testing.allocator.free(two.outcome.ok);
    try testing.expectEqualStrings(off.outcome.ok, two.outcome.ok);
    try testing.expectEqualStrings(one.outcome.ok, two.outcome.ok);
    try testing.expectEqual(one.forwards, two.forwards);
    try testing.expectEqual(@as(usize, 6), two.exchanges);
    try testing.expectEqual(two.exchanges, two.collects);
    try testing.expectEqual(one.launches, two.launches); // the pairs' launches (B's position fill the one more), reordered
    try testing.expectEqual(off.launches + 1, two.launches);
    try testing.expectEqual(one.records, two.records);
    _ = setenv("TF_AFD_INFLIGHT", "1", 1); // one in flight, as unset
    const unit = try attn.dry(testing.allocator, f, &t, config);
    defer testing.allocator.free(unit.outcome.ok);
    try testing.expectEqual(one.launches, unit.launches);
    for ([_][*:0]const u8{ "3", "0", "", "two" }) |v| {
        _ = setenv("TF_AFD_INFLIGHT", v, 1);
        try testing.expectError(error.BadAfdInflight, attn.dry(testing.allocator, f, &t, config));
    }
    _ = unsetenv("TF_AFD_PAIRS");
    _ = setenv("TF_AFD_INFLIGHT", "2", 1); // two in flight only inside a pair
    try testing.expectError(error.BadAfdInflight, attn.dry(testing.allocator, f, &t, config));
}
