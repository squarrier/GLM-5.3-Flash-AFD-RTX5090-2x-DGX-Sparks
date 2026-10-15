//! patch 0002's make_f8 (qualification_qualification_ebed70b7: the dense-q4 form's FP8 head) on a CPU: glm_pack.cu's f8 kernels emulated (one call a
//! CUDA thread, glm_pack_emu.cpp) through cuda_pack_x's f8 op, against python's make_f8 run as torch runs it on the GPU
//! (q4f8.py, the vendored 0002, with torch's CUDA division by a CPU scalar: evidence/qualification_qualification_ebed70b7/f8/f8pack_fixture.py,
//! fixtures_pack_f8.json): the bytes and scales bit-equal by sha256, over inputs both sides draw with one integer rule.
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const pack = glm.pack;
const Buf = glm.views.Buf;
const Sha256 = std.crypto.hash.sha2.Sha256;
const fixture = @embedFile("fixtures_pack_f8.json");

extern fn tf_glm_pack_emu(kernel: c_int, grid: *const [3]c_uint, block: *const [3]c_uint, params: [*]?*anyopaque, sizes: [*]const u16, count: c_uint) c_int;

fn emu(kern: pack.Kern, grid: [3]u32, block: [3]u32, args: *cuda.Args) anyerror!void {
    if (tf_glm_pack_emu(@backingInt(kern), &grid, &block, args.pointers().?, &args.sizes, @intCast(args.count)) != 0) return error.EmulatorRefused;
}

/// splitmix64 of seed + (i + 1) * golden: the fixture script's draw, element by element
fn draw(seed: u64, i: u64) u64 {
    var z = seed +% (i + 1) *% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// A bf16 weight: sign, exponent 114..123 (5 more one time in 61), 7 random mantissa bits
fn weight(r: u64) u16 {
    var e: u64 = 123 - ((r >> 1) % 10);
    if ((r >> 8) % 61 == 0) e += 5;
    return @intCast(((r & 1) << 15) | (e << 7) | ((r >> 16) & 0x7f));
}

/// The fixture's edge rows (the "edges" case): zero and negative-zero blocks, bf16 subnormals (the 1e-30 floor), one
/// large value over small ones (e4m3 subnormals and zeros), a block of +-1.0 (+-448 exactly), huge magnitudes.
fn edge(row: usize, col: usize, r: u64, v: u16) u16 {
    const b = col / 128;
    const sign: u16 = @intCast((r & 1) << 15);
    return switch (row * 4 + b) {
        0 => 0x0000,
        1 => 0x8000,
        2 => sign | @as(u16, @intCast((r >> 16) & 0x7f)),
        3 => if (col % 128 == 5) 0x4040 else sign | @as(u16, @intCast((100 << 7) | ((r >> 16) & 0x7f))),
        4 => if (col % 128 == 0) 0x3f80 else sign | @as(u16, @intCast(((100 + (r >> 20) % 19) << 7) | ((r >> 16) & 0x7f))),
        5 => sign | 0x3f80,
        6 => sign | @as(u16, @intCast((250 << 7) | ((r >> 16) & 0x7f))),
        else => v,
    };
}

fn hex(out: *[64]u8, data: []const u8) []const u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(data, &d, .{});
    out.* = std.fmt.bytesToHex(d, .lower);
    return out;
}

test "make_f8: glm_pack.cu's f8 kernels equal python's make_f8 (torch's GPU division) bit for bit" {
    const gpa = std.testing.allocator;
    const root = try std.json.parseFromSlice(std.json.Value, gpa, fixture, .{});
    defer root.deinit();
    var m = try glm.model.Model.from(&glm.Config{}, .{}, 2, 0);
    var f = pack.hostFwd(&m);
    var cases: usize = 0;
    for (root.value.object.get("cases").?.array.items) |cv| {
        const c = cv.object;
        const n: usize = @intCast(c.get("n").?.integer);
        const k: usize = @intCast(c.get("k").?.integer);
        const seed: u64 = @intCast(c.get("seed").?.integer);
        const edges = c.get("edges").?.bool;
        const w = try gpa.alignedAlloc(u16, .@"16", n * k);
        defer gpa.free(w);
        for (w, 0..) |*x, i| {
            const r = draw(seed, i);
            x.* = if (edges) edge(i / k, i % k, r, weight(r)) else weight(r);
        }
        var h: [64]u8 = undefined;
        errdefer std.debug.print("case {s}\n", .{c.get("name").?.string});
        try std.testing.expectEqualStrings(c.get("input_sha256").?.string, hex(&h, std.mem.sliceAsBytes(w)));
        const stored = [_]u64{@intFromPtr(w.ptr)};
        const f8 = glm.views.F8.of(n, k);
        for ([_]Buf{ f8.weight, f8.scale }, [_][]const u8{ "q_sha256", "scale_sha256" }, 0..) |buf, key, j| {
            const out = try gpa.alignedAlloc(u8, .@"16", buf.bytes());
            defer gpa.free(out);
            @memset(out, 0xA5); // every byte must be written
            var b = buf;
            b.ptr = @intFromPtr(out.ptr);
            const it: pack.Item = .{ .path = "w.head", .buf = b, .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0, .x = .{ .f8 = .{ .src = .{ .g = 0, .off = 0 }, .n = n, .k = k, .part = if (j == 0) .weight else .scale } } };
            try pack.make(&f, .{ .host = emu }, &it, &stored);
            try std.testing.expectEqualStrings(c.get(key).?.string, hex(&h, out));
            // the negative control: torch's CPU make_f8 divides by 448 exactly, and its scales differ in some blocks
            const cpu = c.get("cpu_div").?.object;
            if (j == 1 and cpu.get("scales_differ").?.integer > 0) try std.testing.expect(!std.mem.eql(u8, cpu.get("scale_sha256").?.string, hex(&h, out)));
        }
        cases += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), cases);
}
