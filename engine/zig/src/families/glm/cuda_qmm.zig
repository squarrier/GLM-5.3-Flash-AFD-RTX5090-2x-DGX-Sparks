//! GLM-5.3-Flash's projections (python-0.6 ed78d6f: forward.mm, glm5_next/cuda/qmm.py matmul, kernels/qmm.py matmul
//! and prefill_matmul): K slices fixed by the weight's shape (SHAPE_SK, then SPLIT_TARGET), the grouped kernel's tile
//! by chip, and the one call python makes for a Linear of either format (the seam of qualification_qualification_7c75d777 PLAN §3).
const std = @import("std");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;

/// qmm.SPLIT_TARGET: K slices grow until column tiles times slices reach this
pub const split_target = 192;

/// qmm.SHAPE_SK: per-shape K slices ("n x k" -> slices), which fix every row's arithmetic alike
const shape_sk = [_][3]usize{
    .{ 12576, 4096, 4 }, .{ 4096, 4096, 2 }, .{ 2048, 4096, 4 }, .{ 8192, 1536, 4 }, .{ 8192, 512, 4 }, .{ 4096, 8192, 8 },
};

/// glm5_next qmm.split_k: the shape's table entry, else slices of 64-input groups until 192 programs (at most 8).
pub fn splitK(n: usize, k: usize) usize {
    for (shape_sk) |e| if (e[0] == n and e[1] == k) return e[2];
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// qmm_tiles.group_tile: 0 (the kernel's own pick) except on wide SM 12.0 parts (the RTX PRO 6000) past 16 rows.
pub fn groupTile(m: usize, major: u32, minor: u32, sms: u32) usize {
    if (major != 12 or minor != 0 or sms < 96 or m <= 16) return 0;
    return if (m <= 32) 12 else if (m <= 64) 11 else if (m <= 96) 10 else 11;
}

/// A rank's chip: GB10 (sm_121, 48 SMs) unless a launch path sets another.
pub const Chip = struct { major: u32 = 12, minor: u32 = 1, sms: u32 = 48 };

/// forward.mm: x [R, K] bf16 (rows `x.stride[0]` apart) times q.T into `out` [R, n] (bf16, or fp32 with `f32_out`).
/// Decode windows: q4 through qmm_group with the row's group sums `xs` (python passes them on every decode path);
/// prompt chunks: q4 through qmm_prefill; b16 through `_bmm` with `sk` (Buffers.sk) as its K-slice scratch; f8 (0002's
/// make_f8 head) through `_fmm` the same way.
pub fn mm(f: *Fwd, prefill: bool, x: T, q: views.Linear, xs: ?T, out: T, f32_out: bool, sk: views.Buf, chip: Chip) !void {
    const R = x.shape[0];
    switch (q) {
        .q4 => |w| {
            const ws = [3]T{ T.whole(w.weight), T.whole(w.scales), T.whole(w.biases) };
            if (x.shape[1] != w.k or x.stride[1] != 1 or x.dtype != .bf16) return error.MatmulInput;
            if (out.shape[0] != R or out.shape[1] != w.n or out.stride[0] != w.n) return error.MatmulOutput;
            // kernels/qmm.py clones an input whose rows are not 16-byte aligned; GLM's never are, so none is made
            if (x.stride[1] != 1 or (R > 1 and x.stride[0] % 8 != 0) or x.ptr % 16 != 0) return error.MatmulCopy;
            if (prefill) return f.qmmPrefill(x, ws, out, w.n, f32_out, 0);
            const sums = xs orelse return error.GroupSumsMissing; // python's group_sums(x) would allocate here
            return f.qmmGroup(x, sums, ws, out, w.n, splitK(w.n, w.k), f32_out, groupTile(R, chip.major, chip.minor, chip.sms));
        },
        .b16 => |w| {
            if (x.shape[1] != w.k or x.stride[1] != 1 or x.dtype != .bf16 or w.k % 64 != 0) return error.MatmulInput;
            const slices = @import("cuda_triton.zig").b16SplitK(w.n, w.k);
            if (slices > 1 and sk.numel() < slices * R * w.n) return error.SplitScratch; // python would allocate
            try f.tri.matmulB16(x.ptr, x.stride[0], w.weight.ptr, out.ptr, f32_out, sk.ptr, R, w.n, w.k);
        },
        .f8 => |w| { // patch 0002's _matmul_f8: its checks, then _fmm (and _reduce past one K slice)
            if (x.shape[1] != w.k or x.stride[1] != 1 or x.dtype != .bf16 or w.k % 128 != 0) return error.MatmulInput;
            if (out.shape[0] != R or out.shape[1] != w.n or out.stride[0] != w.n) return error.MatmulOutput;
            const slices = @import("cuda_triton.zig").b16SplitK(w.n, w.k);
            if (slices > 1 and sk.numel() < slices * R * w.n) return error.SplitScratch; // python would allocate
            try f.tri.matmulF8(x.ptr, x.stride[0], w.weight.ptr, w.scale.ptr, out.ptr, f32_out, sk.ptr, R, w.n, w.k);
        },
    }
}

test "split_k: the table first, then the shape's rule" {
    try std.testing.expectEqual(@as(usize, 4), splitK(12576, 4096));
    try std.testing.expectEqual(@as(usize, 2), splitK(4096, 4096));
    try std.testing.expectEqual(@as(usize, 1), splitK(4096, 128));
    try std.testing.expectEqual(@as(usize, 1), splitK(12288, 4096));
    try std.testing.expectEqual(@as(usize, 4), splitK(4096, 6144));
    try std.testing.expectEqual(@as(usize, 0), groupTile(8, 12, 1, 48));
    try std.testing.expectEqual(@as(usize, 11), groupTile(64, 12, 0, 188));
}
