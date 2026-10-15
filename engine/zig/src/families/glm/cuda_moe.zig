//! The MoE block for cuda_layer.layerForward (python-0.6 forward.moe_block, ed78d6f): the router and top-k select,
//! then MLX 4-bit experts (cuda/experts.py's route, gate_up and down: cuda_experts.zig) or EXL3's routed experts
//! (exl3_generic.routed, i.e. cuda/exl3/experts.routed: qualification_qualification_44ba554e-exl3's bindings) with the shared expert as a BF16 MLP, the
//! weighted combine into the rank's partial, and the gather; python's launches in python's order (qualification_qualification_7c75d777 PLAN §3-§4).
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const tri = @import("cuda_triton.zig");
const experts = @import("cuda_experts.zig");
const exl3 = @import("cuda_exl3.zig");
const ex3 = exl3.experts;
const layer = @import("cuda_layer.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;
const Opt = layer.Opt;

const T = trace.T;
const Val = trace.Val;

/// forward.moe_block for R rows: the rank's MoE partial gathered ([world, R, D] fp32 at the returned address).
pub fn block(f: *Fwd, l: *const views.Layer, b: *const bufs.Buffers, R: usize, opt: Opt) !u64 {
    try partial(f, l.moe orelse return error.NotMoe, b, R, opt);
    return layer.gather(f, b, R);
}

/// moe_block up to its gather: b.part[:R] (fp32) = the routed experts weighted by b.wts, the shared expert last.
pub fn partial(f: *Fwd, x: views.Moe, b: *const bufs.Buffers, R: usize, opt: Opt) !void {
    const m = f.m;
    if (R == 0 or R > b.rows) return error.Rows;
    const slots = m.top_k + 1;
    const normed = layer.rowsOf(b.normed, R);
    try router(f, normed, x.router, b, R);
    try f.tri.select(b.mlog.ptr, x.bias.ptr, b.pick.ptr, b.wts.ptr, R, m.experts, m.top_k, m.routed_scale, m.norm_topk);
    switch (x.experts) {
        .q4 => |e| { // the shared expert is expert E of the grouped set
            if (x.shared != null) return error.SharedExpert;
            const c = try q4Calls(m, e, b, R);
            try experts.route(f, c.route);
            try experts.run(f, c.gate_up);
            try experts.run(f, c.down);
        },
        .exl3 => |e| { // the routed slots through the trellis kernels, the shared expert through BF16 matmuls
            const s = x.shared orelse return error.NoSharedExpert;
            const sgu = b.sgu orelse return error.NoExl3Scratch;
            const sact = b.sact orelse return error.NoExl3Scratch;
            const sxs = b.sxs orelse return error.NoExl3Scratch;
            const sy = b.sy orelse return error.NoExl3Scratch;
            try routedExl3(f, normed, e, b, R);
            try qmm.mm(f, b.prefill, normed, s.gu, layer.rowsOf(b.xs, R), layer.rowsOf(sgu, R), false, b.sk, opt.chip);
            try f.tri.swiglu(sgu.ptr, sact.ptr, sxs.ptr, m.limit, R, s.width);
            try qmm.mm(f, b.prefill, layer.rowsOf(sact, R), s.down, layer.rowsOf(sxs, R), layer.rowsOf(sy, R), true, b.sk, opt.chip);
            try sharedSlot(f, b, sy, R);
        },
    }
    try f.tri.combine(b.ey.ptr, b.ey.dtype == .f32, b.wts.ptr, b.part.ptr, R, m.hidden, slots);
}

/// The MLX experts' calls for R rows: experts.route on b.pick[:R], gate_up of b.normed[:R] into b.eact, down into
/// b.ey viewed [rows * slots, D] (`partial` records or launches these; the launch-half test reads them too).
pub const Q4Calls = struct { route: experts.Route, gate_up: experts.Call, down: experts.Call };

pub fn q4Calls(m: *const md.Model, e: views.ExpertsQ4, b: *const bufs.Buffers, R: usize) !Q4Calls {
    const slots = m.top_k + 1;
    const plan = b.plan orelse return error.NoPlan;
    const t = experts.planTile(b);
    return .{
        .route = .{ .picks = T.rows2(b.pick, 0, R, slots, slots), .experts = e.count, .tile = t, .plan = plan },
        .gate_up = experts.gateUpCall(layer.rowsOf(b.normed, R), e, plan, slots, t, b.eact, R, b.prefill),
        .down = experts.downCall(b.eact, e, plan, slots, t, T.of(b.ey, 0, &.{ b.rows * slots, m.hidden }), R, b.prefill),
    };
}

/// glue.router(b.normed[:R], w, b.mlog[:R]); python's K-slice partials are a torch.empty of each call, here b.gath
/// (free inside the block: the attention's gather is consumed, the MoE's not yet written).
fn router(f: *Fwd, x: T, w: views.Buf, b: *const bufs.Buffers, R: usize) !void {
    const ne = w.shape[0];
    if (b.gath.numel() < tri.router_ks * R * ne) return error.RouterScratch;
    try f.tri.router(x.ptr, x.stride[0], w.ptr, b.mlog.ptr, ne, b.gath.ptr, R, w.shape[1], ne);
}

fn int(x: usize) Val {
    return .{ .int = @intCast(x) };
}

fn whole(b: views.Buf) Val {
    return .{ .t = T.whole(b) };
}

/// exl3_generic.routed(b.normed[:R], b.pick, experts, b.exl3, R, limit): cuda/exl3/experts.routed with no weights and
/// GLM's bf16 SwiGLU (ACT_BF16), its six extension calls on python's tensors (Scratch.window(R): ids[:maxu], members
/// [maxu, R]; its y is b.ey), or qualification_qualification_44ba554e-exl3's launches of the same calls.
fn routedExl3(f: *Fwd, x: T, e: views.Exl3Experts, b: *const bufs.Buffers, R: usize) !void {
    const m = f.m;
    const s = b.exl3 orelse return error.NoExl3Scratch;
    const slots = m.top_k + 1;
    const D = e.dims;
    const I = e.width;
    const E = e.count;
    const P = R * slots;
    const maxu = @min(P, E);
    const gu = try ex3.defaultConfig(@intCast(D), @intCast(I), true);
    const dn = try ex3.defaultConfig(@intCast(I), @intCast(D), false);
    const pick = T.whole(b.pick);
    const ids = T.of(s.ids, 0, &.{maxu});
    const members = T.of(s.members_buf, 0, &.{ maxu, R });
    const y = T.of(b.ey, 0, &.{ b.rows * slots, D });
    if (f.rec) |r| {
        try r.call(.ext, "exl3_experts.group", &.{ .{ .t = pick }, .{ .t = ids }, whole(s.count), .{ .t = members }, int(R), int(slots), int(E) });
        try r.call(.ext, "exl3_experts.rot_in", &.{ .{ .t = x }, int(x.stride[0]), .{ .t = pick }, whole(e.suh_g), whole(e.suh_u), whole(s.xg), whole(s.xu), int(R), int(D), int(slots), int(E) });
        try r.call(.ext, "exl3_experts.grouped", &.{
            whole(s.xg),    whole(s.xu),        whole(e.gate_ptr), whole(e.up_ptr), whole(e.gate_k2),   whole(e.up_k2),
            .{ .t = ids },  whole(s.count),     .{ .t = members }, whole(s.z),      int(2),             int(D),
            int(I),         int(P),             int(gu.sk),        int(slots),      int(e.cb),          int(gu.nt),
            int(gu.warps),  int(gu.pf),         int(e.k2_gu[0]),   int(e.k2_gu[1]),
        });
        try r.call(.ext, "exl3_experts.gateup_epilogue", &.{ whole(s.z), .{ .t = pick }, whole(e.svh_g), whole(e.svh_u), whole(e.suh_d), whole(s.xd), int(R), int(P), int(I), int(gu.sk), int(slots), int(E), Val.float(m.limit), int(0) }); // act_mode ACT_BF16
        try r.call(.ext, "exl3_experts.grouped", &.{
            whole(s.xd),    whole(s.xd),        whole(e.down_ptr), whole(e.down_ptr), whole(e.down_k2), whole(e.down_k2),
            .{ .t = ids },  whole(s.count),     .{ .t = members }, whole(s.z),        int(1),           int(I),
            int(D),         int(P),             int(dn.sk),        int(slots),        int(e.cb),        int(dn.nt),
            int(dn.warps),  int(dn.pf),         int(e.k2_d[0]),    int(e.k2_d[1]),
        });
        return r.call(.ext, "exl3_experts.down_epilogue", &.{ whole(s.z), .{ .t = pick }, whole(e.svh_d), .{ .t = y }, int(R), int(P), int(D), int(dn.sk), int(slots), int(E) });
    }
    const g = f.gpu orelse return error.NoGpu;
    const set = g.kernels.exl3 orelse return error.NoExl3Kernels;
    var ls: exl3.kernels.Launches = .{};
    try exl3LaunchesSched(&ls, x, e, b, R, m.limit, .{ .optin = g.kernels.exl3_optin }, g.kernels.exl3_sched);
    try set.run(g.stream, &ls);
}

/// The launches of exl3_generic.routed's calls (qualification_qualification_44ba554e-exl3's experts.routed sequence) on the views' and buffers' addresses.
pub fn exl3Launches(ls: *exl3.kernels.Launches, x: T, e: views.Exl3Experts, b: *const bufs.Buffers, R: usize, limit: f32, sh: ex3.Shared) !void {
    return exl3LaunchesSched(ls, x, e, b, R, limit, sh, .{});
}

/// exl3Launches with 0128's switches (TF_GLM_EXL3_L2D, TF_GLM_EXL3_PDL; off: the same launches).
pub fn exl3LaunchesSched(ls: *exl3.kernels.Launches, x: T, e: views.Exl3Experts, b: *const bufs.Buffers, R: usize, limit: f32, sh: ex3.Shared, sd: ex3.Sched) !void {
    const s = b.exl3 orelse return error.NoExl3Scratch;
    const slots: u32 = @intCast(b.pick.shape[1]);
    const gu = try ex3.defaultConfig(@intCast(e.dims), @intCast(e.width), true);
    const dn = try ex3.defaultConfig(@intCast(e.width), @intCast(e.dims), false);
    const l: ex3.Layer = .{
        .gate_ptr = e.gate_ptr.ptr, .up_ptr = e.up_ptr.ptr, .down_ptr = e.down_ptr.ptr, .gate_k2 = e.gate_k2.ptr,
        .up_k2 = e.up_k2.ptr, .down_k2 = e.down_k2.ptr, .suh_g = e.suh_g.ptr, .suh_u = e.suh_u.ptr, .svh_g = e.svh_g.ptr,
        .svh_u = e.svh_u.ptr, .suh_d = e.suh_d.ptr, .svh_d = e.svh_d.ptr, .count = @intCast(e.count), .dims = @intCast(e.dims),
        .width = @intCast(e.width), .cb = @enumFromInt(e.cb), .k2_gu = e.k2_gu, .k2_d = e.k2_d,
    };
    const sc: ex3.Scratch = .{
        .xg = s.xg.ptr, .xu = s.xu.ptr, .xd = s.xd.ptr, .z = s.z.ptr, .y = b.ey.ptr, .ids = s.ids.ptr, .count = s.count.ptr,
        .members = s.members_buf.ptr, .rows = @intCast(b.rows), .slots = slots, .experts = @intCast(e.count), .cfg_gu = gu, .cfg_d = dn,
    };
    try ex3.routedSched(ls, x.ptr, @intCast(x.stride[0]), true, b.pick.ptr, null, l, sc, 0, @intCast(R), limit, .bf16, true, sh, sd);
}

/// b.ey[:R, top_k].copy_(b.sy[:R]) (EXL3): torch's copy of each row's shared expert into its last slot. Python's
/// trace has no event for a torch copy, so recording adds none; a GPU copies R rows of D fp32 on the stream in one
/// strided driver copy (lane B's cuda.copy2D, qualification_qualification_cec53599; it was one driver copy a row).
fn sharedSlot(f: *Fwd, b: *const bufs.Buffers, sy: views.Buf, R: usize) !void {
    if (f.rec != null) return;
    const g = f.gpu orelse return error.NoGpu;
    if (b.ey.dtype != .f32 or sy.dtype != .f32) return error.SharedSlotType;
    const c = slotCopy(b.ey.ptr, sy.ptr, f.m.hidden, f.m.top_k, R);
    try cuda.copy2D(g.stream.d, c.dst, c.dst_pitch, c.src, c.src_pitch, c.width, c.height, g.stream.handle);
}

/// sharedSlot's copy as cuda.copy2D's arguments: R rows of D fp32, row r from sy + r * D to slot top_k of ey's row r.
const Slot = struct { dst: u64, dst_pitch: usize, src: u64, src_pitch: usize, width: usize, height: usize };

fn slotCopy(ey: u64, sy: u64, d: usize, top_k: usize, R: usize) Slot {
    return .{ .dst = ey + top_k * d * 4, .dst_pitch = (top_k + 1) * d * 4, .src = sy, .src_pitch = d * 4, .width = d * 4, .height = R };
}

test "sharedSlot's 2D copy moves the bytes its per-row copies moved (cuMemcpy2D's rows on host arrays)" {
    for ([_][3]usize{ .{ 1, 8, 8 }, .{ 37, 16, 8 }, .{ 64, 24, 4 } }) |c| {
        const R = c[0];
        const d = c[1];
        const k = c[2];
        const n = R * (k + 1) * d * 4;
        const ey_rows = try std.testing.allocator.alloc(u8, n);
        defer std.testing.allocator.free(ey_rows);
        const ey_2d = try std.testing.allocator.alloc(u8, n);
        defer std.testing.allocator.free(ey_2d);
        const sy = try std.testing.allocator.alloc(u8, R * d * 4);
        defer std.testing.allocator.free(sy);
        for (sy, 0..) |*x, i| x.* = @truncate(i * 7 + 1);
        @memset(ey_rows, 0xa5);
        @memset(ey_2d, 0xa5);
        for (0..R) |r| @memcpy(ey_rows[((r * (k + 1) + k) * d) * 4 ..][0 .. d * 4], sy[r * d * 4 ..][0 .. d * 4]); // A2's copies
        const s = slotCopy(0, 0, d, k, R); // addresses from 0: offsets into the host arrays
        for (0..s.height) |r| @memcpy(ey_2d[s.dst + r * s.dst_pitch ..][0..s.width], sy[s.src + r * s.src_pitch ..][0..s.width]);
        try std.testing.expectEqualSlices(u8, ey_rows, ey_2d);
    }
}
