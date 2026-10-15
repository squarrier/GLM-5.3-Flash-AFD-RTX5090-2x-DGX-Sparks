//! mtp_fp8: patch 0002's stack_f8 (make_f8 of torch.cat's rows) as cuda_pack_x's f8cat op makes it for the MTP layer's
//! stacked projections (TF_GLM_MTP_FP8=1), on a CPU through glm_pack.cu's f8 kernels emulated (glm_pack_emu.cpp): each
//! part's rows made where the stack puts them equal make_f8 over the parts copied back to back, bit for bit, bytes and
//! scales. pack_f8.zig ties make_f8 (the f8 op) to python's own, so f8cat is python's stack_f8.
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const pack = glm.pack;
const xo = pack.x_ops;

extern fn tf_glm_pack_emu(kernel: c_int, grid: *const [3]c_uint, block: *const [3]c_uint, params: [*]?*anyopaque, sizes: [*]const u16, count: c_uint) c_int;

fn emu(kern: pack.Kern, grid: [3]u32, block: [3]u32, args: *cuda.Args) anyerror!void {
    if (tf_glm_pack_emu(@backingInt(kern), &grid, &block, args.pointers().?, &args.sizes, @intCast(args.count)) != 0) return error.EmulatorRefused;
}

/// splitmix64 of seed + (i + 1) * golden, then pack_f8.zig's bf16 weight rule (sign, exponent 114..123, 5 more one
/// time in 61, 7 mantissa bits), with a zero block and a huge one in the first part
fn weight(seed: u64, i: u64, k: usize) u16 {
    var z = seed +% (i + 1) *% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    const r = z ^ (z >> 31);
    if (i < 128) return 0; // row 0's first block all zeros: the 1e-30 floor
    if (i / k == 1 and i % k < 128) return @intCast(((r & 1) << 15) | (250 << 7) | ((r >> 16) & 0x7f));
    var e: u64 = 123 - ((r >> 1) % 10);
    if ((r >> 8) % 61 == 0) e += 5;
    return @intCast(((r & 1) << 15) | (e << 7) | ((r >> 16) & 0x7f));
}

fn make(f: *glm.fwd.Fwd, buf: glm.views.Buf, x: xo.Op, stored: []const u64, out: []u8) !void {
    @memset(out, 0xA5); // every byte must be written
    var b = buf;
    b.ptr = @intFromPtr(out.ptr);
    const it: pack.Item = .{ .path = "w.mtp.layer.dsa.proj.weight", .buf = b, .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0, .x = x };
    try pack.make(f, .{ .host = emu }, &it, stored);
}

test "stack_f8 (f8cat): parts made where the stack puts them equal make_f8 of the parts back to back, bytes and scales" {
    const gpa = std.testing.allocator;
    var m = try glm.model.Model.from(&glm.Config{}, .{}, 2, 0);
    var f = pack.hostFwd(&m);
    const k = 512;
    const parts = [_]usize{ 8, 5, 3 }; // rows of each stacked part (q_a, kv_a's form; three to cover the general case)
    var n: usize = 0;
    for (parts) |r| n += r;
    // stored group 0: the parts in reverse order with gaps (the split's places); group 1: the same rows back to back
    const g0 = try gpa.alignedAlloc(u8, .@"16", 3 * 4096 + n * k * 2);
    defer gpa.free(g0);
    const g1 = try gpa.alignedAlloc(u8, .@"16", n * k * 2);
    defer gpa.free(g1);
    @memset(g0, 0x5A);
    var c: xo.Cat = .{ .n = parts.len };
    var at: usize = 0;
    var row: usize = 0;
    var j: usize = parts.len;
    while (j > 0) : (j -= 1) at += parts[j - 1] * k * 2 + 4096; // the last part first in group 0
    for (parts, 0..) |r, p| {
        at -= r * k * 2 + 4096;
        c.srcs[p] = .{ .g = 0, .off = at };
        c.lens[p] = r * k * 2;
        const dst: []u16 = @alignCast(std.mem.bytesAsSlice(u16, g0[at .. at + r * k * 2]));
        const cat: []u16 = @alignCast(std.mem.bytesAsSlice(u16, g1[row * k * 2 .. (row + r) * k * 2]));
        for (dst, cat, 0..) |*d, *e, i| {
            d.* = weight(7, row * k + i, k);
            e.* = d.*;
        }
        row += r;
    }
    const stored = [_]u64{ @intFromPtr(g0.ptr), @intFromPtr(g1.ptr) };
    const f8 = glm.views.F8.of(n, k);
    for ([_]glm.views.Buf{ f8.weight, f8.scale }, 0..) |buf, pi| {
        const a = try gpa.alignedAlloc(u8, .@"16", buf.bytes());
        defer gpa.free(a);
        const b = try gpa.alignedAlloc(u8, .@"16", buf.bytes());
        defer gpa.free(b);
        const weight_part = pi == 0;
        try make(&f, buf, .{ .f8cat = .{ .cat = c, .k = k, .part = if (weight_part) .weight else .scale } }, &stored, a);
        try make(&f, buf, .{ .f8 = .{ .src = .{ .g = 1, .off = 0 }, .n = n, .k = k, .part = if (weight_part) .weight else .scale } }, &stored, b);
        try std.testing.expectEqualSlices(u8, b, a);
        try std.testing.expect(std.mem.indexOfNone(u8, a, &.{0xA5}) != null);
        // one part alone (eh, q_b, o, qb, down: a single stored tensor) is make_f8 of its rows
        var one: xo.Cat = .{ .n = 1 };
        one.srcs[0] = .{ .g = 1, .off = 0 };
        one.lens[0] = n * k * 2;
        try make(&f, buf, .{ .f8cat = .{ .cat = one, .k = k, .part = if (weight_part) .weight else .scale } }, &stored, a);
        try std.testing.expectEqualSlices(u8, b, a);
    }
    // a part that is not whole rows is refused, never made short
    var bad = c;
    bad.lens[1] -= 2;
    const out = try gpa.alignedAlloc(u8, .@"16", f8.scale.bytes());
    defer gpa.free(out);
    try std.testing.expectError(error.CatBytes, make(&f, f8.scale, .{ .f8cat = .{ .cat = bad, .k = k, .part = .scale } }, &stored, out));
}
