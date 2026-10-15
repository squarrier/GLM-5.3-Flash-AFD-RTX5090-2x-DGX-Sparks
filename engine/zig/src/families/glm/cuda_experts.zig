//! python-0.6's cuda/experts.py (ed78d6f) for GLM's MoE on MLX 4-bit experts: max_items, route (experts.cpp plan),
//! gate_up and down (experts.cpp run on a decode window's plan, prefill on a prompt chunk's). Each is the extension
//! call the Python wrapper makes, recorded with python's arguments (the CPU trace) or launched through the
//! glm_experts / glm_experts_prefill images (cuda_kernels_mm.zig, qualification_qualification_abcce4b2 A1). The module and kernels are Ash Hart's.
const std = @import("std");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
/// The affine kernels' bindings (tf-glm-layer loads Mm through here).
pub const mm = @import("cuda_kernels_mm.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;
const Val = trace.Val;

/// experts.py COLS: NTW n8 tiles a warp, 8 columns each.
pub const cols = 32;
/// experts.py TILE (a decode item's pairs) and PREFILL_TILE (a prompt plan's, `route`'s default tile).
pub const tile = bufs.expert_tile;
pub const prefill_tile = 64;

/// experts.max_items: an item per used expert, plus one per `t` pairs past its first.
pub fn maxItems(pairs: usize, experts: usize, t: usize) usize {
    return @min(pairs, experts) + pairs / t;
}

/// plan.tile as `route` leaves it: TILE on a decode window's plan, the route default PREFILL_TILE on a prompt chunk's.
pub fn planTile(b: *const bufs.Buffers) usize {
    return if (b.prefill) prefill_tile else tile;
}

fn int(x: usize) Val {
    return .{ .int = @intCast(x) };
}

/// experts.route(picks, plan): the picks [R, slots], the plan's experts (E + 1) and tile, and its scratch.
pub const Route = struct { picks: T, experts: usize, tile: usize, plan: bufs.Plan };

/// `_ext().plan(picks, R * slots, experts, tile, members, items, counts, rank, hist)`, recorded or launched.
pub fn route(f: *Fwd, r: Route) !void {
    const p = r.plan;
    if (f.rec) |rec| return rec.call(.ext, "experts.plan", &.{
        .{ .t = r.picks },          int(pairCount(r)),              int(r.experts),
        int(r.tile),                .{ .t = T.whole(p.members) }, .{ .t = T.whole(p.items) },
        .{ .t = T.whole(p.counts) }, .{ .t = T.whole(p.rank) },    .{ .t = T.whole(p.hist) },
    });
    const g = f.gpu orelse return error.NoGpu;
    try g.kernels.mm.plan(g.stream, r.picks.ptr, pairCount(r), r.experts, r.tile, mmPlan(p));
}

fn pairCount(r: Route) usize {
    return r.picks.shape[0] * r.picks.shape[1];
}

fn mmPlan(p: bufs.Plan) mm.Plan {
    return .{ .members = p.members.ptr, .items = p.items.ptr, .counts = p.counts.ptr, .rank = p.rank.ptr, .hist = p.hist.ptr };
}

/// The launch half of `route`: the launches Mm.plan makes for it.
pub fn planLaunches(r: Route) !mm.PlanLaunches {
    return mm.planLaunches(r.picks.ptr, pairCount(r), r.experts, r.tile, mmPlan(r.plan));
}

/// One experts.cpp run (decode) or prefill (prompt) call: gs, epi, x, slots, w, kg, nb, items, counts, members, out,
/// n, limit and the bound (run: max units; prefill: max items), as experts.gate_up and experts.down pass them.
pub const Call = struct { prefill: bool, gs: usize, epi: u8, x: T, slots: usize, w: T, kg: usize, nb: usize, plan: bufs.Plan, out: T, n: usize, limit: f32, bound: usize };

/// The call recorded as python makes it, or launched as experts.cpp makes it.
pub fn run(f: *Fwd, c: Call) !void {
    if (f.rec) |r| return r.call(.ext, if (c.prefill) "experts.prefill" else "experts.run", &.{
        int(c.gs),                       int(c.epi),                        .{ .t = c.x },
        int(c.slots),                    .{ .t = c.w },                     int(c.kg),
        int(c.nb),                       .{ .t = T.whole(c.plan.items) },   .{ .t = T.whole(c.plan.counts) },
        .{ .t = T.whole(c.plan.members) }, .{ .t = c.out },                 int(c.n),
        Val.float(c.limit),              int(c.bound),
    });
    const g = f.gpu orelse return error.NoGpu;
    if (c.prefill) return g.kernels.mm.expertsPrefill(g.stream, launchArgs(c));
    try g.kernels.mm.experts(g.stream, launchArgs(c));
}

/// experts.cpp's arguments from python's (cuda_kernels_mm_calls.zig reads a recorded call the same way).
pub fn launchArgs(c: Call) mm.Experts {
    return .{ .gs = c.gs, .epi = c.epi, .x = c.x.ptr, .x_stride = c.x.stride[0], .slots = c.slots, .w = c.w.ptr, .kg = c.kg, .nb = c.nb, .items = c.plan.items.ptr, .counts = c.plan.counts.ptr, .members = c.plan.members.ptr, .out = c.out.ptr, .n = c.n, .limit = c.limit, .bound = c.bound };
}

/// The launch half of `run`: the launch Mm.experts (on `chip`, `per_sm` blocks an SM for gate/up and down) or
/// Mm.expertsPrefill makes for the call; null when python launches nothing.
pub fn launchOf(c: Call, chip: mm.Chip, per_sm: [2]usize) !?mm.Launch {
    const e = launchArgs(c);
    if (c.prefill) return mm.prefillExpertsLaunch(e);
    return mm.runLaunch(e, per_sm[if (e.epi == 2) 0 else 1], chip.sms);
}

/// experts.gate_up(x, ex, plan, out, rows): x [R, D] bf16 -> out [R * slots, NI] bf16, each pair's activation
/// (SwiGLU, epilogue 2, when up holds gate and up); `t` the plan's tile, `ex.count` its experts (E + 1).
pub fn gateUpCall(x: T, ex: views.ExpertsQ4, plan: bufs.Plan, slots: usize, t: usize, out: views.Buf, R: usize, prefill: bool) Call {
    const items = maxItems(R * slots, ex.count, t);
    const nb = ex.width / cols;
    const epi: u8 = if (ex.up.shape[3] == 2) 2 else 1;
    return .{ .prefill = prefill, .gs = ex.gs, .epi = epi, .x = x, .slots = slots, .w = T.whole(ex.up), .kg = ex.dims / ex.gs, .nb = nb, .plan = plan, .out = T.whole(out), .n = ex.width, .limit = ex.limit, .bound = if (prefill) items else items * nb };
}

/// experts.down(act, ex, plan, out, rows): act [R * slots, NI] bf16 -> out [R * slots, D] per pair: fp32, or bf16
/// on a prompt chunk's plan when `out` is bf16 (epilogue 3).
pub fn downCall(act: views.Buf, ex: views.ExpertsQ4, plan: bufs.Plan, slots: usize, t: usize, out: T, R: usize, prefill: bool) Call {
    const items = maxItems(R * slots, ex.count, t);
    const nb = ex.dims / cols;
    const epi: u8 = if (prefill and out.dtype == .bf16) 3 else 0;
    return .{ .prefill = prefill, .gs = ex.gs, .epi = epi, .x = T.whole(act), .slots = 0, .w = T.whole(ex.down), .kg = ex.width / ex.gs, .nb = nb, .plan = plan, .out = out, .n = ex.dims, .limit = 0, .bound = if (prefill) items else items * nb };
}

test "max_items and the plan's tile as experts.py sizes them" {
    try std.testing.expectEqual(@as(usize, 19), maxItems(18, 289, tile));
    try std.testing.expectEqual(@as(usize, 289 + 288), maxItems(2048 * 9, 289, prefill_tile));
    try std.testing.expectEqual(@as(usize, 76), maxItems(72, 289, tile)); // Buffers' plan.items rows at 8 rows
}
