//! glm_select.cu (qualification_qualification_973313f6: select_tokens' tail, one launch) on the CPU emulator (zig/tests/cuda/glm_select_emu.cpp, one
//! call a CUDA thread) against python-0.6's own torch ops on drawn pools (src/families/glm/fixtures_cuda_select.json,
//! tools/zig/glm_dsa_fixtures.py): the kernel's tokens and counts, launched with cuda_dsa_kernels' grid, must be python's.
const std = @import("std");

extern fn tf_glm_select_emu(grid: *const [3]c_uint, block: *const [3]c_uint, pools: [*]const i64, pos: *const i32, tokens: [*]i32, counts: [*]i32, rows: u32) void;
extern fn tf_glm_pos_add_emu(grid: *const [3]c_uint, block: *const [3]c_uint, pos: *const i32, n: i32, out: *i32) void;

const fixture = @embedFile("src/families/glm/fixtures_cuda_select.json");
/// cuda_dsa_kernels: a row's tokens, threads a block, and its grid (rows x cdiv(width, threads))
const width = 512 * 4 + 3;
const threads = 256;

fn splitmix(s: *u64) u64 {
    s.* +%= 0x9E3779B97F4A7C15;
    var z = s.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

test "glm_select.cu, emulated, gives python-0.6's select_tokens tail on drawn pools (both sides of the dense limit)" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, fixture, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("glm-select-tail/1", parsed.value.object.get("schema").?.string);
    var rows_seen: usize = 0;
    for (parsed.value.object.get("cases").?.array.items) |c| {
        const o = c.object;
        const rows: usize = @intCast(o.get("rows").?.integer);
        const np: u64 = @intCast(o.get("np").?.integer);
        var seed: u64 = @intCast(o.get("seed").?.integer);
        const pools = try gpa.alloc(i64, rows * 512);
        defer gpa.free(pools);
        for (pools) |*p| p.* = @intCast(splitmix(&seed) % np);
        const tokens = try gpa.alloc(i32, rows * width);
        defer gpa.free(tokens);
        @memset(tokens, 0x5a5a5a5a); // every token must be written
        var counts: [16]i32 = @splat(0x5a5a5a5a);
        const pos: i32 = @intCast(o.get("pos").?.integer);
        tf_glm_select_emu(&.{ @intCast(rows), (width + threads - 1) / threads, 1 }, &.{ threads, 1, 1 }, pools.ptr, &pos, tokens.ptr, &counts, @intCast(rows));
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(tokens), &sha, .{});
        try std.testing.expectEqualStrings(o.get("sha256").?.string, &std.fmt.bytesToHex(sha, .lower));
        for (o.get("counts").?.array.items, 0..) |n, r| try std.testing.expectEqual(n.integer, counts[r]);
        try std.testing.expectEqual(@as(i32, 0x5a5a5a5a), counts[rows]); // nothing past the rows
        rows_seen += rows;
    }
    try std.testing.expectEqual(@as(usize, 54), rows_seen);
}

test "glm_select.cu's tf_glm_pos_add, emulated: torch's int32 `pos_dev + n` of the blocks' first rows (qualification_qualification_ea348a94)" {
    // (pos, n): the long prompt's dense and select blocks (2,048-row chunks at 0 and 2,048, the tail at 4,096)
    for ([_][2]i32{ .{ 0, 512 }, .{ 0, 1536 }, .{ 2048, 512 }, .{ 2048, 1536 }, .{ 4096, 512 } }) |c| {
        var out: i32 = 0x5a5a5a5a;
        tf_glm_pos_add_emu(&.{ 1, 1, 1 }, &.{ 1, 1, 1 }, &c[0], c[1], &out);
        try std.testing.expectEqual(c[0] + c[1], out);
    }
    var out: i32 = 0;
    const pos: i32 = std.math.maxInt(i32);
    tf_glm_pos_add_emu(&.{ 2, 1, 1 }, &.{ 32, 1, 1 }, &pos, 1, &out); // one thread writes; int32 wraps
    try std.testing.expectEqual(std.math.minInt(i32), out);
}
