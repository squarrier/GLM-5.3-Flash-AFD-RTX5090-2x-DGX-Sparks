//! GLM-5.3-Flash's own EXL3 expert kernels on CUDA (python-0.6 glm5_next/cuda/exl3.cu and exl3_mm.py): the grouped
//! 4-bit mcg trellis GEMV over `experts.route`'s items and the Hadamard rotations around it. Each function is a C++
//! wrapper's launch for the glm_exl3 fatbin; `routed` is exl3_mm.routed. Format after ExLlamaV3 (turboderp, MIT).

const std = @import("std");
const cuda = @import("cuda");
const k = @import("cuda_exl3_kernels.zig");

const Launches = k.Launches;

/// exl3_mm's tile counts, warps and K splits, fixed across row counts so each pair's bits stay its own.
pub const gateup_cfg = [3]u32{ 8, 4, 4 };
pub const down_cfg = [3]u32{ 8, 4, 1 };
/// tensorfold.cuda.experts.TILE: pairs an item holds.
pub const tile = 16;

/// experts.max_items: an item per used expert, plus one per `tile` pairs past its first.
pub fn maxItems(pairs: u32, experts: u32) u32 {
    return @min(pairs, experts) + pairs / tile;
}

/// One layer's routed experts on this rank (Exl3Experts): stacked 4-bit trellis words and scales.
pub const Experts = struct {
    gt: u64, // int32 [E, D/16, NI/16, 32]
    ut: u64,
    dt: u64, // int32 [E, NI/16, D/16, 32]
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, NI]
    svh_u: u64,
    suh_d: u64, // fp16 [E, NI]
    svh_d: u64, // fp16 [E, D]
    count: u32, // E (the trellis' first dim)
    width: u32, // NI
    dims: u32, // D
};

/// exl3_mm.Scratch for `rows` rows of `slots` slots (the last slot the shared expert's).
pub const Scratch = struct {
    xg: u64,
    xu: u64,
    xd: u64,
    z: u64,
    rows: u32,
    slots: u32,

    /// Byte sizes of each buffer, as Scratch.__init__ allocates them (xu as xg; z holds both configs' K splits).
    pub fn bytes(rows: u32, slots: u32, dims: u32, width: u32) struct { xg: usize, xd: usize, z: usize } {
        const P: usize = @as(usize, rows) * slots;
        const sk: usize = @max(gateup_cfg[2], down_cfg[2]);
        return .{ .xg = 2 * P * dims, .xd = 2 * P * width, .z = 4 * 2 * sk * P * @max(width, dims) };
    }
};

/// experts.route's plan: items (expert, first, count) int32, counts [items, distinct], members int32.
pub const Plan = struct { items: u64, counts: u64, members: u64, experts: u32 };

fn int(x: anytype) c_int {
    return @intCast(x);
}

/// exl3_grouped_cuda: grouped_kernel<NT, W><<<(max_items, N / (16 nt), mats * SK), W * 32>>>.
pub fn grouped(out: *Launches, x0: u64, x1: u64, t0: u64, t1: u64, p: Plan, z: u64, mats: u32, K: u32, N: u32, P: u32, SK: u32, max_items: u32, nt: u32, warps: u32, E: u32) error{Invalid}!void {
    if (K % (16 * SK * warps) != 0 or N % (16 * nt) != 0) return error.Invalid;
    const name = k.sym.glmGrouped(nt, warps);
    if (name.len == 0) return error.Invalid;
    var a: cuda.Args = .{};
    for ([_]u64{ x0, x1, t0, t1, p.items, p.counts, p.members, z }) |v| a.add(v);
    for ([_]u32{ K, N, P, SK, E }) |v| a.add(int(v));
    out.push(.{ .image = .glm_exl3, .symbol = name, .cfg = .{ .grid = .{ .x = max_items, .y = N / (16 * nt), .z = mats * SK }, .block = .{ .x = warps * 32 } }, .args = a });
}

/// exl3_rot_in_cuda: <<<(rows * slots, K / 128, 2), 32>>>; the last slot (the shared expert) is skipped.
pub fn rotIn(out: *Launches, x: u64, x_stride: u32, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, rows: u32, K: u32, slots: u32) error{Invalid}!void {
    if (K % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(int(x_stride));
    for ([_]u64{ pick, suh0, suh1, out0, out1 }) |v| a.add(v);
    a.add(int(K));
    a.add(int(slots));
    out.push(.{ .image = .glm_exl3, .symbol = k.sym.glm_rot_in, .cfg = .{ .grid = .{ .x = rows * slots, .y = K / 128, .z = 2 }, .block = .{ .x = 32 } }, .args = a });
}

/// exl3_gateup_epilogue_cuda: <<<(rows * slots, N / 128), 32>>>.
pub fn gateupEpilogue(out: *Launches, z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, rows: u32, P: u32, N: u32, SK: u32, slots: u32, limit: f32) error{Invalid}!void {
    if (N % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    for ([_]u64{ z, pick, svh_g, svh_u, suh_d, xd }) |v| a.add(v);
    for ([_]u32{ P, N, SK, slots }) |v| a.add(int(v));
    a.add(limit);
    out.push(.{ .image = .glm_exl3, .symbol = k.sym.glm_gateup_epilogue, .cfg = .{ .grid = .{ .x = rows * slots, .y = N / 128 }, .block = .{ .x = 32 } }, .args = a });
}

/// exl3_down_epilogue_cuda: <<<(rows * slots, D / 128), 32>>>.
pub fn downEpilogue(out: *Launches, z: u64, pick: u64, svh_d: u64, y: u64, rows: u32, P: u32, D: u32, SK: u32, slots: u32) error{Invalid}!void {
    if (D % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    for ([_]u64{ z, pick, svh_d, y }) |v| a.add(v);
    for ([_]u32{ P, D, SK, slots }) |v| a.add(int(v));
    out.push(.{ .image = .glm_exl3, .symbol = k.sym.glm_down_epilogue, .cfg = .{ .grid = .{ .x = rows * slots, .y = D / 128 }, .block = .{ .x = 32 } }, .args = a });
}

/// exl3_mm.routed: Y[row * slots + slot] (fp32) for rows 0..R-1's routed slots. P counts the scratch's rows, as Python's.
pub fn routed(out: *Launches, x: u64, x_stride: u32, pick: u64, p: Plan, ex: Experts, s: Scratch, y: u64, R: u32, limit: f32) error{Invalid}!void {
    const slots = s.slots;
    const P = s.rows * s.slots;
    const D = ex.dims;
    const NI = ex.width;
    const items = maxItems(R * slots, p.experts);
    try rotIn(out, x, x_stride, pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots);
    try grouped(out, s.xg, s.xu, ex.gt, ex.ut, p, s.z, 2, D, NI, P, gateup_cfg[2], items, gateup_cfg[0], gateup_cfg[1], ex.count);
    try gateupEpilogue(out, s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, NI, gateup_cfg[2], slots, limit);
    try grouped(out, s.xd, s.xd, ex.dt, ex.dt, p, s.z, 1, NI, D, P, down_cfg[2], items, down_cfg[0], down_cfg[1], ex.count);
    try downEpilogue(out, s.z, pick, ex.svh_d, y, R, P, D, down_cfg[2], slots);
}

test "max_items: an item per expert used plus one per 16 pairs" {
    try std.testing.expectEqual(@as(u32, 9 + 0), maxItems(9, 289));
    try std.testing.expectEqual(@as(u32, 289 + 1152), maxItems(2048 * 9, 289));
}

test "scratch bytes as exl3_mm.Scratch allocates them (a TP2 rank, 2,048 rows)" {
    const b = Scratch.bytes(2048, 9, 4096, 1024);
    try std.testing.expectEqual(@as(usize, 150994944), b.xg);
    try std.testing.expectEqual(@as(usize, 37748736), b.xd);
    try std.testing.expectEqual(@as(usize, 2415919104), b.z);
}
