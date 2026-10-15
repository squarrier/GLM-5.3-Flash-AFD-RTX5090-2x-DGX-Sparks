//! TensorFold's EXL3 linear layers on CUDA (python-0.6 cuda/exl3/linear.py and linear.cu): y = x @ W + bias for 1-128
//! rows, any codebook and width, a row's bits its own. Each function is a C++ wrapper's launch for the exl3_linear
//! fatbin; `plan`, `strides` and `call` are linear.py's. Format after ExLlamaV3 (turboderp, MIT); the module by Ash
//! Hart and the TensorFold contributors.

const std = @import("std");
const cuda = @import("cuda");
const k = @import("cuda_exl3_kernels.zig");

const Launches = k.Launches;
pub const Codebook = k.Codebook;

/// linear.cu's io dtypes (dtype_of): fp16, bf16, fp32.
pub const DType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

/// The trellis words' order on the device: per 128-column block in k order ("strips"), or as stored.
pub const Layout = enum { strips, stored };

/// (K splits, warps a program) for a K x N layer (linear.py plan): the shape's alone, one reduction for every row count.
pub fn plan(K: u32, N: u32) [2]u32 {
    const blocks = 192;
    const min_tiles = 8;
    const kt = K / 16;
    const nb = N / 128;
    var sk: u32 = 1;
    var wk: u32 = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ sk, wk };
}

/// One Exl3Linear on the device: trellis words, fp16 scales, optional fp16 bias, split-K counters int32 [8 N / 128].
pub const Layer = struct {
    words: u64,
    suh: u64,
    svh: u64,
    bias: ?u64,
    counters: u64,
    k: u32,
    n: u32,
    k2: u32, // half-bits a value
    cb: Codebook,
    layout: Layout = .strips,
    split: ?[2]u32 = null, // plan(k, n) when null

    /// (words between k tiles, words between 128-column blocks).
    pub fn strides(l: Layer) [2]i64 {
        const tw: i64 = 4 * @as(i64, l.k2);
        return switch (l.layout) {
            .strips => .{ 8 * tw, @as(i64, l.k / 16) * 8 * tw },
            .stored => .{ @as(i64, l.n / 16) * tw, 8 * tw },
        };
    }

    pub fn sk(l: Layer) [2]u32 {
        return l.split orelse plan(l.k, l.n);
    }
};

fn int(x: anytype) c_int {
    return @intCast(x);
}

fn widthOk(k2: u32, cb: Codebook) bool {
    for (k.linear_widths) |p| if (p[0] == k2 and p[1] == @backingInt(cb)) return true;
    return false;
}

/// exl3_rot_in_cuda: rot_in_kernel<<<(ceil(K / 512), M), 128>>>, xh = fp16((x * suh) @ H / sqrt(128)).
pub fn rotIn(out: *Launches, x: u64, x_dtype: DType, suh: u64, xh: u64, M: u32, K: u32) error{Invalid}!void {
    if (K % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@backingInt(x_dtype));
    a.add(suh);
    a.add(xh);
    a.add(int(K));
    out.push(.{ .image = .exl3_linear, .symbol = k.sym.linear_rot_in, .cfg = .{ .grid = .{ .x = (K / 128 + 3) / 4, .y = M }, .block = .{ .x = 128 } }, .args = a });
}

/// exl3_linear_cuda: linear_kernel<K2, CB, WK><<<(N / 128, SK), WK * 32, WK * min(M, 8) * 512>>>.
pub fn linear(out: *Launches, xh: u64, l: Layer, y: u64, y_dtype: DType, z: ?u64, M: u32) error{Invalid}!void {
    const s = l.sk();
    const SK = s[0];
    const WK = s[1];
    if (WK != 2 and WK != 4 and WK != 8) return error.Invalid;
    if ((l.k / 16) % (SK * WK) != 0) return error.Invalid;
    if (SK > 1 and z == null) return error.Invalid;
    if (M < 1 or M > 128 or l.k % 128 != 0 or l.n % 128 != 0 or !widthOk(l.k2, l.cb)) return error.Invalid;
    const smem: u32 = WK * @min(M, 8) * 128 * 4;
    const st = l.strides();
    var a: cuda.Args = .{};
    a.add(xh);
    a.add(l.words);
    a.add(st[0]);
    a.add(st[1]);
    a.add(l.svh);
    a.add(l.bias orelse 0);
    a.add(y);
    a.add(@backingInt(y_dtype));
    a.add(z orelse 0);
    a.add(l.counters);
    for ([_]u32{ M, l.k, l.n, SK }) |v| a.add(int(v));
    out.push(.{ .image = .exl3_linear, .symbol = k.sym.linear(l.k2, l.cb, WK), .cfg = .{ .grid = .{ .x = l.n / 128, .y = SK }, .block = .{ .x = WK * 32 }, .shared = smem }, .args = a, .max_dynamic = if (smem > 48 * 1024) smem else null });
}

/// exl3_unpack_cuda: unpack_kernel<K2, CB><<<(N / 16, K / 16), 32>>>, W_q [K, N] fp16 from either layout.
pub fn unpack(out: *Launches, l: Layer, w: u64) error{Invalid}!void {
    if (!widthOk(l.k2, l.cb)) return error.Invalid;
    const st = l.strides();
    var a: cuda.Args = .{};
    a.add(l.words);
    a.add(w);
    a.add(int(l.n));
    a.add(st[0]);
    a.add(st[1]);
    out.push(.{ .image = .exl3_linear, .symbol = k.sym.unpack(l.k2, l.cb), .cfg = .{ .grid = .{ .x = l.n / 16, .y = l.k / 16 }, .block = .{ .x = 32 } }, .args = a });
}

/// Exl3Linear.__call__: rotate the input into xh, then the layer into y (z: SK > 1's fp32 partials).
pub fn call(out: *Launches, l: Layer, x: u64, x_dtype: DType, M: u32, y: u64, y_dtype: DType, xh: u64, z: ?u64) error{Invalid}!void {
    try rotIn(out, x, x_dtype, l.suh, xh, M, l.k);
    try linear(out, xh, l, y, y_dtype, if (l.sk()[0] > 1) z else null, M);
}

test "plan: the shape's own splits" {
    try std.testing.expectEqual([2]u32{ 1, 8 }, plan(4096, 12288));
    try std.testing.expectEqual([2]u32{ 8, 4 }, plan(4096, 4096));
    try std.testing.expectEqual([2]u32{ 8, 4 }, plan(4096, 1024));
    try std.testing.expectEqual([2]u32{ 1, 4 }, plan(128, 128));
}

test "strides by layout" {
    const l: Layer = .{ .words = 0, .suh = 0, .svh = 0, .bias = null, .counters = 0, .k = 4096, .n = 2048, .k2 = 8, .cb = .mcg };
    try std.testing.expectEqual([2]i64{ 256, 256 * 256 }, l.strides());
    var s = l;
    s.layout = .stored;
    try std.testing.expectEqual([2]i64{ 128 * 32, 256 }, s.strides());
}
