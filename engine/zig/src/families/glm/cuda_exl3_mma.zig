//! qualification_8724e161 (qualification_7af419c1's qualification_babb0a2d): the halves' prompt windows through bertholomus's grouped_mma (exl3/mma/; Albert Lee,
//! BertholomusAI, Apache-2.0, his commits co-authored by Claude Opus 5.5): a device-built work list of (expert place,
//! 64-row member block) pairs (deepseek-v41-tp2 90b68e0), gate/up with the K splits folded and down (folded only where
//! it has splits, as GLM's experts.py:250-254), each weight tile decoded once for 64 member rows, in the launch order of
//! deepseek-v41-zig ce20a545:zig/src/families/dsv41/exl3_experts.zig:192-270. Every output keeps X1's fp32 chain: the
//! same SK x WK mma chains from zero over the same k tiles, warps then splits summed in order from 0.

const std = @import("std");
const cuda = @import("cuda");
const k = @import("cuda_exl3_kernels.zig");
const ex3 = @import("cuda_exl3_experts.zig");

const Launches = k.Launches;

/// experts_grouped.cuh's MMA_ROWS, MMA_COLS and MMA_KB; experts.cu's LIST_THREADS.
pub const rows = 64;
pub const cols = 64;
pub const kb = 4;
pub const list_threads = 1024;

/// experts.py _list_len: a work list's pairs for P member slots over at most maxu places in groups of `rows`.
pub fn listLen(P: u32, maxu: u32, group: u32) u32 {
    return (P + maxu * (group - 1)) / group;
}

/// The work list's int32 count for a scratch of `R` rows x `slots` over E experts (pairs x 2).
pub fn workInts(R: u32, slots: u32, E: u32) u32 {
    return 2 * listLen(R * slots, @min(R * slots, E), rows);
}

/// experts.py _mma_chains at MMA_KB: the K -> N projection's SK x WK chains split into steps of 4 k tiles, and N into
/// 64-column blocks (the kernel's instances: 4 k tiles a step, no 2 or 6).
pub fn chains(K: u32, N: u32, t: ex3.Tile) bool {
    if (K % 16 != 0 or N % cols != 0 or t.sk == 0 or t.warps == 0) return false;
    const kt = K / 16;
    if (kt % (t.warps * t.sk) != 0) return false;
    return (kt / (t.warps * t.sk)) % kb == 0;
}

/// Whether a layer's windows take the MMA path: X1's scope (promptReady: mcg, K2 8..8, GLM's tiles) and both
/// projections' chains (gate/up D -> I, down I -> D).
pub fn ready(ex: ex3.Layer, s: ex3.Scratch) bool {
    return ex3.promptReady(ex, s) and chains(ex.dims, ex.width, s.cfg_gu) and chains(ex.width, ex.dims, s.cfg_d);
}

fn int(x: anytype) c_int {
    return @intCast(x);
}

/// exl3x_work_list_cuda: work_list_kernel<<<1, 1024>>>(counts, uids, ucount, maxu, work, 64, n, null, 64, 0): one list
/// (both projections take 64-row blocks), n = listLen(P, maxu, 64) pairs, sentinels past the used places' groups.
pub fn workList(out: *Launches, counts: u64, uids: u64, ucount: u64, maxu: u32, work: u64, n: u32) error{Invalid}!void {
    if (counts == 0 or work == 0 or n == 0) return error.Invalid;
    var a: cuda.Args = .{};
    for ([_]u64{ counts, uids, ucount }) |v| a.add(v);
    a.add(int(maxu));
    a.add(work);
    a.add(int(rows));
    a.add(int(n));
    a.add(@as(u64, 0));
    a.add(int(rows));
    a.add(int(0));
    out.push(.{ .image = .exl3_mma_cb1, .symbol = k.sym.work_list, .cfg = .{ .grid = .{}, .block = .{ .x = list_threads } }, .args = a });
}

/// grouped_mma_launch<1>: grouped_mma_kernel<1, 8, 8, FOLD, 4, XROW><<<(1, N / 64, n mats), 256>>>(x0, x1, tp0, tp1,
/// k2_0, k2_1, uids, ucount, members, z, K, N, P, SK, WK, maxm, slots, n, work): Z [mats, P, N], one plane a matrix
/// (FOLD: the SK splits summed in order from 0; without, SK must be 1). WK is the window's warps.
pub fn grouped(out: *Launches, g: ex3.Grouped, fold: bool, xrow: bool, work: u64, n: u32) error{Invalid}!void {
    const symbol = k.sym.groupedMma(fold, xrow);
    if (symbol.len == 0 or work == 0 or n == 0) return error.Invalid;
    if (g.cb != .mcg or g.lo != 8 or g.hi != 8 or (!fold and g.SK != 1)) return error.Invalid;
    if (!chains(g.K, g.N, .{ .nt = g.nt, .warps = g.warps, .sk = g.SK, .pf = g.pf })) return error.Invalid;
    if (@as(u64, n) * g.mats > 65535) return error.Invalid; // grid z
    var a: cuda.Args = .{};
    for ([_]u64{ g.x0, g.x1, g.tp0, g.tp1, g.k2_0, g.k2_1, g.uids, g.ucount, g.members, g.z }) |v| a.add(v);
    for ([_]u32{ g.K, g.N, g.P, g.SK, g.warps, g.maxm, g.slots, n }) |v| a.add(int(v));
    a.add(work);
    out.push(.{ .image = .exl3_mma_cb1, .symbol = symbol, .cfg = .{ .grid = .{ .x = 1, .y = g.N / cols, .z = n * g.mats }, .block = .{ .x = 256 } }, .args = a });
}

/// routedX1's window (R >= prompt_rows, inside `ready`, an X1 switch in effect) with TF_GLM_EXL3_MMA=1, after PGROUP's
/// grouping (ids, count, members and `counts`, int32 [E]): the work list into `work` (int32 [workInts(rows, slots,
/// E)]), one rotation a row (XROW, with `shared_suh`) or rot_in, gate/up through grouped_mma with the splits folded
/// (with p.seq SEQ's kernel instead, X1's order list `ord` as routedX1 makes it), its epilogue at SK 1, down through
/// grouped_mma (folded where it has splits), then the down epilogue at SK 1 or, with weights, the fused combine.
pub fn routed(out: *Launches, x: u64, x_stride: u32, x_bf16: bool, pick: u64, wts: ?u64, ex: ex3.Layer, s: ex3.Scratch, o: u64, R: u32, limit: f32, act: ex3.Act, p: ex3.Prompt, ord: u64, shared_suh: ?u64, counts: u64, work: u64) error{Invalid}!void {
    const D = ex.dims;
    const I = ex.width;
    const E = ex.count;
    const slots = s.slots;
    const P = R * slots;
    if (R > s.rows or R < ex3.prompt_rows or !ready(ex, s)) return error.Invalid;
    const maxu = @min(P, s.experts);
    const n = listLen(P, maxu, rows);
    const xrow = p.xrow and shared_suh != null; // python: rowrot = rowrot and ex.shared_suh is not None
    try workList(out, counts, s.ids, s.count, maxu, work, n);
    var cap: u32 = 0;
    var obuf: u64 = 0;
    if (p.seq and p.order) { // SEQ's gate/up keeps X1's items' order (routedX1's rule)
        cap = ex3.orderSlots(R, slots, E);
        if (!(p.nfirst and cap > 65535)) {
            if (ord == 0) return error.Invalid;
            obuf = ord;
            ex3.order(out, s.ids, s.count, s.members, obuf, R, cap, E);
        }
    }
    if (xrow) try ex3.rotRows(out, x, x_stride, x_bf16, shared_suh.?, s.xg, R, D) else try ex3.rotIn(out, x, x_stride, x_bf16, pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E);
    var g: ex3.Grouped = .{ .x0 = s.xg, .x1 = if (xrow) s.xg else s.xu, .tp0 = ex.gate_ptr, .tp1 = ex.up_ptr, .k2_0 = ex.gate_k2, .k2_1 = ex.up_k2, .uids = s.ids, .ucount = s.count, .members = s.members, .z = s.z, .mats = 2, .K = D, .N = I, .P = P, .SK = s.cfg_gu.sk, .slots = slots, .maxm = R, .nexp_max = maxu, .cb = ex.cb, .nt = s.cfg_gu.nt, .warps = s.cfg_gu.warps, .pf = s.cfg_gu.pf, .lo = ex.k2_gu[0], .hi = ex.k2_gu[1] };
    if (p.seq) try ex3.groupedX1(out, g, .{ .nfirst = p.nfirst, .pa = p.pa, .xrow = xrow, .seq = true }, obuf, cap) else try grouped(out, g, true, xrow, work, n);
    try ex3.gateupEpilogue(out, s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, 1, slots, E, limit, act);
    g = .{ .x0 = s.xd, .x1 = s.xd, .tp0 = ex.down_ptr, .tp1 = ex.down_ptr, .k2_0 = ex.down_k2, .k2_1 = ex.down_k2, .uids = s.ids, .ucount = s.count, .members = s.members, .z = s.z, .mats = 1, .K = I, .N = D, .P = P, .SK = s.cfg_d.sk, .slots = slots, .maxm = R, .nexp_max = maxu, .cb = ex.cb, .nt = s.cfg_d.nt, .warps = s.cfg_d.warps, .pf = s.cfg_d.pf, .lo = ex.k2_d[0], .hi = ex.k2_d[1] };
    try grouped(out, g, s.cfg_d.sk > 1, false, work, n);
    if (wts) |w| return ex3.downCombine(out, s.z, pick, ex.svh_d, s.y, w, o, R, P, D, 1, slots, E);
    try ex3.downEpilogue(out, s.z, pick, ex.svh_d, s.y, R, P, D, 1, slots, E);
}

test {
    _ = @import("cuda_exl3_mma_test.zig");
}
