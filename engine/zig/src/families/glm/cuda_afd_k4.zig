//! K4 on the expert halves (qualification_e4ad04cd; TF_GLM_HALVES_KERNEL=k4): every window of a prompt forward (python 2.15's rule,
//! experts_node.py:224: `node.prompt = mode != MODE_EAGER`) takes MiaAI-Lab's EXL3 prompt kernel (cuda_exl3_k4.zig) at
//! a position where prompt_kernel_ready holds; eager windows and other positions keep today's launches (X1's under
//! TF_GLM_EXL3_X1=1). Its buffers (python's Scratch.prompt_bufs: the plan's items and counts and her order list, a set
//! a scratch) are a block of their own, so with the switch unset nothing is placed and every address is today's.

const std = @import("std");
const cuda = @import("cuda");
const ex3 = @import("cuda_exl3_experts.zig");
const k3 = @import("cuda_exl3_kernels.zig");
const k4 = @import("cuda_exl3_k4.zig");
const xs = @import("cuda_exl3_split.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const ae = @import("cuda_afd_expert.zig");
const trace = @import("cuda_trace.zig");
const tri = @import("cuda_triton.zig");
const proto = cuda.xchg.proto;
const Buf = views.Buf;
const T = trace.T;
const Val = trace.Val;

/// python's `node.prompt` for a forward of mode `m`: every mode but eager (chunk, slice, pair, lanes).
pub fn prompt(m: proto.Mode) bool {
    return m != .eager;
}

/// K4's block: each scratch's plan buffers, sized as python's _prompt_buffers sizes them (the scratch's widest window
/// at the smaller pass): items int32 [3 cap], counts int32 [2], order int32 [cap].
pub const Bufs = struct {
    decode_items: Buf,
    decode_counts: Buf,
    decode_order: Buf,
    pref_items: Buf,
    pref_counts: Buf,
    pref_order: Buf,

    pub fn init(b: *const ae.Bufs, experts: usize) Bufs {
        const cd = cap(&b.decode, experts);
        const cp = cap(&b.pref, experts);
        return .{
            .decode_items = Buf.of(.i32, &.{3 * cd}), .decode_counts = Buf.of(.i32, &.{2}), .decode_order = Buf.of(.i32, &.{cd}),
            .pref_items = Buf.of(.i32, &.{3 * cp}), .pref_counts = Buf.of(.i32, &.{2}), .pref_order = Buf.of(.i32, &.{cp}),
        };
    }

    fn cap(s: *const ae.Scratch, experts: usize) usize {
        return k4.promptItems(@intCast(s.rows), @intCast(s.slots), @intCast(experts), k4.passes[0]);
    }

    /// Scratch `s`'s set (python's s.prompt_bufs) as the bindings take it.
    pub fn of(x: *const Bufs, b: *const ae.Bufs, s: *const ae.Scratch) k4.Bufs {
        const d = s == &b.decode;
        const items = if (d) x.decode_items else x.pref_items;
        return .{ .items = items.ptr, .counts = (if (d) x.decode_counts else x.pref_counts).ptr, .order = (if (d) x.decode_order else x.pref_order).ptr, .cap = @intCast(items.shape[0] / 3) };
    }
};

/// K4's block (an allocation of its own): its bytes, the buffers' addresses from `base`.
pub fn place(x: *Bufs, base: u64) !usize {
    var a: bufs.Arena = .{ .base = base };
    try views.walk(x, views.Path.init("k4"), &a, bufs.Arena.visit);
    return a.used;
}

/// python's nc_aligned as the kernels need it: every expert's gate, up and down trellis of a position on a 16-byte
/// boundary, its group at device address `base` (the cp.async stages read 16 bytes at a time).
pub fn aligned(e: xs.Experts, base: u64) bool {
    for ([_]xs.Proj{ e.gate, e.up, e.down }) |pr| for (pr.offsets) |o| if ((base + pr.trellis.off + o) % 16 != 0) return false;
    return true;
}

/// The halves' K4: its switches (read once with TF_GLM_HALVES_KERNEL), its block, each position's prompt_kernel_ready
/// (the layer bound at its group, the scratches' slots, the trellises' alignment), and X1's block for the shared suh
/// rows its rowrot reads (python's `shx = rowrot and ex.shared_suh is not None`; `xrow` each position's test).
pub const On = struct {
    opts: k4.Opts,
    bufs: *const Bufs,
    ready: []const bool,
    x1b: *const ae.X1Bufs,
    xrow: []const bool,

    /// Whether the window of a forward of mode `m` at position `pos` takes K4.
    pub fn takes(o: *const On, pos: usize, m: proto.Mode) bool {
        return prompt(m) and o.ready[pos];
    }

    /// Position `pos`'s shared suh row for rowrot (null: rowrot off, or the layer has none: rot_in).
    pub fn suh(o: *const On, pos: usize) ?u64 {
        return if (o.opts.xrow and o.xrow[pos]) o.x1b.suh(pos) else null;
    }
};

/// Position `pos`'s prompt_kernel_ready on the half (its group at `base`, `slots` the scratches').
pub fn ready(x: *const ae.Half, pos: usize, base: u64, slots: usize) bool {
    return k4.ready(x.bind(pos, base), @intCast(slots), aligned(x.experts(pos), base));
}

/// One compute's routed launches under K4 (launchesX1's twin): python 2.15's _prompt_routed over the landing's views
/// into scratch `s`, y [P, D] for the caller's shared-slot zero and combine, as today's.
pub fn launches(ls: *k3.Launches, v: ae.Views, l: ex3.Layer, s: *const ae.Scratch, R: usize, limit: f32, sh: ex3.Shared, o: *const On, b: *const ae.Bufs, pos: usize) !void {
    if (R == 0 or R > s.rows) return error.Rows;
    try k4.routed(ls, v.normed, l.dims, true, v.pick, l, s.bound(l.count), @intCast(R), limit, true, sh, o.opts, o.bufs.of(b, s), o.suh(pos));
}

/// K4's buffers as a record names them (python's): each scratch's prompt buffers (b.<scratch>.prompt_bufs.0-2: items,
/// counts, order) and the layer's shared suh (w.layers.<l>.moe.experts.shared_suh; null: none).
pub const Rec = struct { bufs: *const Bufs, shared_suh: ?Buf };

/// python ExpertNode.compute(layer, R, x) of a prompt window under TF_GLM_EXL3_PROMPT=1, recorded (glm-forward-trace/1):
/// exl3_generic.routed's group, her rotation (rot_rows for a shared suh under rowrot, else rot_in) and exl3_experts.prompt
/// (experts.py:376-405, its 25 arguments), the shared slot's zero_ (a torch op: no event), and glue.combine of y.
pub fn record(t: *trace.Trace, e: *const views.Exl3Experts, b: *const ae.Bufs, v: ae.Views, R: usize, limit: f32, o: k4.Opts, r: Rec) !void {
    const s = b.scratch(R);
    if (R == 0 or R > s.rows) return error.Rows;
    const slots = s.slots;
    const D = e.dims;
    const E = e.count;
    const maxu = @min(R * slots, E);
    const x = at(v.normed, .bf16, &.{ R, D });
    const pick: Val = .{ .t = at(v.pick, .i32, &.{ R, slots }) };
    const ids: Val = .{ .t = T.of(s.ids, 0, &.{maxu}) };
    const members: Val = .{ .t = T.of(s.members_buf, 0, &.{ maxu, R }) };
    try t.call(.ext, "exl3_experts.group", &.{ pick, ids, whole(s.count), members, int(R), int(slots), int(E) });
    const shx = o.xrow and r.shared_suh != null; // python: shx = rowrot and ex.shared_suh is not None
    if (shx)
        try t.call(.ext, "exl3_experts.rot_rows", &.{ .{ .t = x }, int(x.stride[0]), whole(r.shared_suh.?), whole(s.xg), int(R), int(D) })
    else
        try t.call(.ext, "exl3_experts.rot_in", &.{ .{ .t = x }, int(x.stride[0]), pick, whole(e.suh_g), whole(e.suh_u), whole(s.xg), whole(s.xu), int(R), int(D), int(slots), int(E) });
    const d = s == &b.decode;
    const items = if (d) r.bufs.decode_items else r.bufs.pref_items;
    const counts = if (d) r.bufs.decode_counts else r.bufs.pref_counts;
    const ord: Val = if (o.order) whole(if (d) r.bufs.decode_order else r.bufs.pref_order) else .none;
    try t.call(.ext, "exl3_experts.prompt", &.{
        whole(s.xg),      if (shx) whole(s.xg) else whole(s.xu), whole(e.gate_ptr), whole(e.up_ptr),  whole(e.down_ptr), ids,    whole(s.count), members,
        whole(items),     whole(counts),                         ord,               whole(e.svh_g),    whole(e.svh_u),    whole(e.suh_d), whole(e.svh_d), whole(s.xd),
        whole(s.y),       int(D),                                int(e.width),      int(E),            int(k4.promptItems(@intCast(R), @intCast(slots), @intCast(E), o.pass)),
        Val.float(limit), int(o.pass),                           int(slots),        .{ .bool = shx },
    });
    const tr: tri.Tri = .{ .set = null, .rec = &t.tri_rec };
    try tr.combine(s.y.ptr, true, v.wts, b.routed.ptr, R, D, slots);
    try t.flush();
}

fn at(ptr: u64, dtype: views.DType, shape: []const usize) T {
    var b = Buf.of(dtype, shape);
    b.ptr = ptr;
    return T.whole(b);
}

fn int(x: usize) Val {
    return .{ .int = @intCast(x) };
}

fn whole(b: Buf) Val {
    return .{ .t = T.whole(b) };
}

test {
    _ = @import("cuda_afd_k4_test.zig");
}
