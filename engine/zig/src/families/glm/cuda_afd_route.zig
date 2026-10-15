//! The halves' routed dispatch (qualification_c863b626, the join of qualification_1b9bf461's g53kt3r and qualification_e4ad04cd's k4 into qualification_qualification_44ba554e-int): which launches a
//! window takes on an expert half, as Device.compute runs them. g53kt3r (Hugh Madden's glm53f-rank kernels with KT3R's
//! schedule) when it takes the window, then K4 (Mia's EXL3 prompt kernel), then X1's launches with their switches (SEQ,
//! PGROUP and MMA live there), else today's. Both halves kernels take every window whose mode is not eager (python
//! 2.15's rule, experts_node.py:224) at a position they serve, so SEQ, PGROUP and MMA never apply to a window either
//! takes; they stay on X1's windows (eager ones, and positions the kernel does not serve).
const std = @import("std");
const cuda = @import("cuda");
const ae = @import("cuda_afd_expert.zig");
const ex3 = @import("cuda_exl3_experts.zig");
const proto = cuda.xchg.proto;

pub const Route = enum { g53, k4, x1, today };

/// The route of forward `f`'s window at MoE position `pos`: TF_GLM_HALVES_KERNEL's g53 node or K4 state (null: not
/// that value), then X1's switches (off: today's launches).
pub fn route(g53n: ?*const ae.g53.Node, k4: ?*const ae.afdk4.On, x1: ex3.Prompt, pos: usize, f: ae.Forward) Route {
    if (g53n) |k| if (k.takes(pos, f.rows, f.mode != .eager)) return .g53;
    if (k4) |k| if (k.takes(pos, f.mode)) return .k4;
    return if (x1.on()) .x1 else .today;
}

test "qualification_c863b626: the halves' route: g53kt3r, then k4, take every window whose mode is not eager at a position they serve, whatever X1's switches (SEQ, PGROUP, MMA); eager windows and other positions keep X1's or today's, as with the switch unset" {
    const modes = [_]proto.Mode{ .eager, .chunk, .slice, .pair, .lanes };
    const x1 = try ex3.Prompt.parse("1", .{ null, null, null, null }); // ZL_XENV's X1, its four on
    var seq = x1;
    seq.seq = true; // qualification_969df694's f-s line
    var fm = seq; // f-m in qualification_946d4b77's seq form: SEQ + PGROUP + MMA
    fm.pgroup = true;
    fm.mma = true;
    var fold = x1; // qualification_946d4b77's fold form: PGROUP + MMA, SEQ unset
    fold.pgroup = true;
    fold.mma = true;
    const forms = [_]ex3.Prompt{ .{}, x1, seq, fm, fold };
    const ready = [_]bool{ true, false, true };
    const suh = [_]?u64{ null, null, null };
    const node: ae.g53.Node = .{ .set = undefined, .plan = undefined, .max_rows = 2048, .ready = &ready, .suh = &suh };
    const on: ae.afdk4.On = .{ .opts = .{}, .bufs = undefined, .ready = &ready, .x1b = undefined, .xrow = &.{} };
    var n: [4]usize = .{ 0, 0, 0, 0 };
    for (forms) |p| for (modes) |m| for (0..ready.len) |pos| for ([_]usize{ 1, 63, 64, 2048, 2049 }) |R| {
        const f: ae.Forward = .{ .fseq = 1, .rows = R, .mode = m, .first = 0, .count = 3 };
        const unset: Route = if (p.on()) .x1 else .today; // 614cdf7's dispatch
        try std.testing.expectEqual(unset, route(null, null, p, pos, f));
        const g = route(&node, null, p, pos, f);
        const k = route(null, &on, p, pos, f);
        const taken = m != .eager and ready[pos];
        try std.testing.expectEqual(if (taken and R <= 2048) Route.g53 else unset, g);
        try std.testing.expectEqual(if (taken) Route.k4 else unset, k);
        n[@intFromEnum(g)] += 1;
        n[@intFromEnum(k)] += 1;
    };
    // 5 forms x 4 prompt modes x 2 served positions: g53 at 4 of 5 row counts, k4 at all 5
    try std.testing.expectEqual(@as(usize, 5 * 4 * 2 * 4), n[@intFromEnum(Route.g53)]);
    try std.testing.expectEqual(@as(usize, 5 * 4 * 2 * 5), n[@intFromEnum(Route.k4)]);
}
