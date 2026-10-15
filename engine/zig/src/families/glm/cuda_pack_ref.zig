//! The pack pass's host reference (G-layout, qualification_qualification_7c75d777 PLAN §4): python-0.6's load-time layouts computed on the CPU as python
//! computes them (kernels/qmm.py `pack`'s nibble picks, the inverse of cuda/experts.py `unpack`, torch's row picks), not
//! as glm_pack.cu does, so the kernels (through the emulator) and python's fixtures are each checked against it.
const std = @import("std");
const pack = @import("cuda_pack.zig");

/// The G-layout cases' inputs, as tools/zig/glm_forward_fixtures.py `pack` draws them: splitmix64 from the case's
/// seed, a u32 word from each draw's high half, a 16-bit scale or bias from its top quarter, row-major.
pub const Draw = struct {
    s: u64,

    pub fn next(d: *Draw) u64 {
        d.s +%= 0x9E3779B97F4A7C15;
        var z = d.s;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }
    pub fn words(d: *Draw, out: []u8) void {
        for (0..out.len / 4) |i| std.mem.writeInt(u32, out[4 * i ..][0..4], @intCast(d.next() >> 32), .little);
    }
    pub fn halves(d: *Draw, out: []u8) void {
        for (0..out.len / 2) |i| std.mem.writeInt(u16, out[2 * i ..][0..2], @intCast(d.next() >> 48), .little);
    }
};

/// kernels/qmm.py OFFSETS: nibble p of a lane's word holds input 32 v + 2 (lane % 4) + OFFSETS[p] of its group
const offsets = [8]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };

fn word32(b: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, b[4 * i ..][0..4], .little);
}

fn half16(b: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, b[2 * i ..][0..2], .little);
}

/// `pack(words, scales, biases, 64)`'s words: python's (cols, kg, gs) nibbles, (T, j, r, kg, gs) picked at OFFSETS,
/// permuted to (T, kg, j, r, c, V): out [npad/64][kg][8][32][2], zero columns past n.
pub fn words(out: []u32, src: []const u8, n: usize, k: usize, rows: pack.Rows) void {
    const k8 = k / 8;
    const kg = k / 64;
    const npad = (n + 127) / 128 * 128;
    for (0..npad / 64) |t| for (0..kg) |g| for (0..8) |j| for (0..8) |r| for (0..4) |c| for (0..2) |v| {
        const col = t * 64 + j * 8 + r;
        var w: u32 = 0;
        if (col < n) {
            const row = rows.at(col);
            for (offsets, 0..) |o, p| {
                const in = g * 64 + 32 * v + 2 * c + o;
                const nib = (word32(src, row * k8 + in / 8) >> @intCast(4 * (in % 8))) & 0xF;
                w |= nib << @intCast(4 * p);
            }
        }
        out[(((t * kg + g) * 8 + j) * 32 + r * 4 + c) * 2 + v] = w;
    };
}

/// `pack`'s `major`: scales or biases [n, kg] -> [kg, npad], zero columns past n.
pub fn major(out: []u16, src: []const u8, n: usize, k: usize, rows: pack.Rows) void {
    const kg = k / 64;
    const npad = (n + 127) / 128 * 128;
    for (0..kg) |g| for (0..npad) |c| {
        out[g * npad + c] = if (c < n) half16(src, rows.at(c) * kg + g) else 0;
    };
}

/// `w[rows]` of a stored matrix with `bytes`-byte rows, contiguous (AbsorbQ4's per-head reshape keeps the bytes).
pub fn gather(out: []u8, src: []const u8, n: usize, bytes: usize, rows: pack.Rows) void {
    for (0..n) |c| @memcpy(out[c * bytes ..][0..bytes], src[rows.at(c) * bytes ..][0..bytes]);
}

/// experts.py `unpack` read backwards: block [E][n/32][kg][block] of `pack`, here written as projection `m` of `mm`
/// in `make`'s stack, for experts [first, first + e) of the buffer; `w` [e][n][k/8], `s`, `b` [e][n][k/64].
pub fn experts(out: []u32, w: []const u8, s: []const u8, b: []const u8, first: usize, e: usize, n: usize, k: usize, m: usize, mm: usize) void {
    const k8 = k / 8;
    const kg = k / 64;
    const nb = n / 32;
    for (0..e) |x| for (0..nb) |bb| for (0..kg) |g| {
        const blk = out[((((first + x) * nb + bb) * kg + g) * mm + m) * 288 ..][0..288];
        // unpack: words [:256] as (a=2, l=32, q'=4) permuted to (a, q', l) = (t=4, j=2, r=8, q=4) -> row t*8+r, col q*2+j
        for (0..4) |t| for (0..2) |j| for (0..8) |r| for (0..4) |q| {
            const f = ((t * 2 + j) * 8 + r) * 4 + q;
            const pos = ((f / 128) * 32 + f % 32) * 4 + (f / 32) % 4;
            const in = word32(w, (x * n + bb * 32 + t * 8 + r) * k8 + g * 8 + q * 2 + j);
            var word: u32 = 0;
            for (0..8) |slot| word |= ((in >> @intCast(4 * (2 * (slot % 4) + slot / 4))) & 0xF) << @intCast(4 * slot);
            blk[pos] = word;
        };
        // unpack: [256:] as bf16 (p'=4, i=2 scales|biases, t=4, h=2) -> row t*8 + p'*2 + h of scales or biases
        for (0..4) |p| for (0..2) |i| for (0..4) |t| {
            const src = if (i == 0) s else b;
            const row = x * n + bb * 32 + t * 8 + p * 2;
            const lo: u32 = half16(src, row * kg + g);
            const hi: u32 = half16(src, (row + 1) * kg + g);
            blk[256 + (p * 2 + i) * 4 + t] = lo | hi << 16;
        };
    };
}

fn region(stored: []const []const u8, r: @import("cuda_split.zig").Ref) []const u8 {
    return stored[r.g][@intCast(r.off)..];
}

/// The bytes python leaves for `it`, from the stored groups' host bytes.
pub fn item(out: []u8, it: *const pack.Item, stored: []const []const u8) void {
    const len = it.buf.bytes();
    switch (it.op) {
        .copy => |r| @memcpy(out[0..len], region(stored, r)[0..len]),
        .words => |x| words(std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(out[0..len]))), region(stored, x.src), x.n, x.k, x.rows),
        .major => |x| major(std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(out[0..len]))), region(stored, x.src), x.n, x.k, x.rows),
        .rows => |x| gather(out[0..len], region(stored, x.src), x.n, x.bytes, x.rows),
        .experts => |x| {
            const all = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(out[0..len])));
            for (0..x.mm) |m| {
                const r = x.routed[m];
                const sh = x.shared[m];
                experts(all, region(stored, r.w), region(stored, r.s), region(stored, r.b), 0, x.e, x.n, x.k, m, x.mm);
                experts(all, region(stored, sh.w), region(stored, sh.s), region(stored, sh.b), x.e, 1, x.n, x.k, m, x.mm);
            }
        },
    }
}

test "pack's words: a column's nibbles land where kernels/qmm.py's OFFSETS put them, pad columns zero" {
    // one column of 64 inputs (k = 64): word w holds inputs 8w .. 8w + 7, input i = i % 16
    var src: [8 * 4]u8 = undefined;
    for (0..8) |w| {
        var x: u32 = 0;
        for (0..8) |i| x |= @as(u32, @intCast((8 * w + i) % 16)) << @intCast(4 * i);
        std.mem.writeInt(u32, src[4 * w ..][0..4], x, .little);
    }
    var out: [2 * 1 * 8 * 32 * 2]u32 = undefined;
    words(&out, &src, 1, 64, .{});
    // column 0 is (j 0, r 0); lane c = 0 word v = 0 holds inputs 0, 8, 16, 24, 1, 9, 17, 25 (mod 16: 0 8 0 8 1 9 1 9)
    try std.testing.expectEqual(@as(u32, 0x91918080), out[0]);
    for (out[2 * 4 ..]) |x| try std.testing.expectEqual(@as(u32, 0), x);
}
