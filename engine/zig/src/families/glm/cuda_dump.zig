//! qualification_6607f886's recorder taps on the forward (evidence/qualification_6607f886/lab/qualification_6607f886_capture.py: layer_hook, block_hook, compute_hook), for
//! tf-glm-forward's `dump` and `teacher`: compute and layer_forward as cuda_compute.zig and cuda_layer.zig run them,
//! launch for launch (the test below holds them equal), with each block's gathered partials ([world, R, D] fp32: `L03.dsa`,
//! `L03.moe`, ...), b.x after each layer (`L03.x`), then `hidden` and `fnormed` handed to a Tap as python records them.
const std = @import("std");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const prompt = @import("cuda_prompt.zig");
const moe = @import("cuda_moe.zig");
const cc = @import("cuda_compute.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;
const Buffers = bufs.Buffers;
const State = bufs.State;

/// Where a recorded tensor goes: `name` (python's), the view, the window's first position and rows, and L.x's kind.
pub const Tap = struct {
    ctx: *anyopaque,
    tensor: *const fn (ctx: *anyopaque, name: []const u8, x: T, pos0: usize, rows: usize, kind: ?[]const u8) anyerror!void,

    fn put(t: Tap, comptime fmt: []const u8, args: anytype, x: T, pos0: usize, rows: usize, kind: ?[]const u8) !void {
        var buf: [32]u8 = undefined;
        try t.tensor(t.ctx, try std.fmt.bufPrint(&buf, fmt, args), x, pos0, rows, kind);
    }
};

/// A block's gathered partials at `g`: [world, R, D] fp32.
fn partials(f: *const Fwd, g: u64, R: usize, world: usize) T {
    const n = world * R * f.m.hidden;
    return T.of(.{ .ptr = g, .dtype = .f32, .dims = 1, .shape = .{ n, 1, 1, 1, 1 } }, 0, &.{ world, R, f.m.hidden });
}

/// layer.layerForward with the recorder's taps: the attention and MLP blocks' partials, then b.x.
pub fn layerForward(f: *Fwd, l: *const views.Layer, st: *const State, b: *const Buffers, R: usize, opt: layer.Opt, tap: Tap) !void {
    const m = f.m;
    if (R == 0 or R > b.rows) return error.Rows;
    const out: @import("cuda_triton.zig").HcOut = .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr };
    const h = l.attn_hc orelse return error.PlainLayer;
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.in_norm.ptr, out, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
    const g = switch (l.kind) {
        .kda => try layer.kdaBlock(f, l, st, b, R, opt),
        .dsa => try prompt.dsaBlock(f, l, st, b, R, opt), // a prompt chunk's: the engine's Scratch
    };
    try tap.put("L{d:0>2}.{t}", .{ l.index, l.kind }, partials(f, g, R, m.world), st.pos, R, null);
    try f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
    const h2 = l.ffn_hc orelse return error.PlainLayer;
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = h2.@"fn".ptr, .base = h2.base.ptr, .scale = h2.scale.ptr }, l.post_norm.ptr, out, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
    const dense = l.mlp != null;
    const planes: @import("cuda_fwd.zig").Planes = if (dense) .{ .g = try layer.mlpBlock(f, l, b, R, opt), .world = m.world }
        else if (f.moe) |provider| try provider.block(provider.ctx, f, l, b, R, opt)
        else .{ .g = try moe.block(f, l, b, R, opt), .world = m.world };
    try tap.put("L{d:0>2}.{s}", .{ l.index, if (dense) "mlp" else "moe" }, partials(f, planes.g, R, planes.world), st.pos, R, null);
    try f.tri.hcPost(b.x.ptr, b.x.ptr, planes.g, b.post.ptr, b.comb.ptr, planes.world, R, m.hidden, m.streams);
    try tap.put("L{d:0>2}.x", .{l.index}, layer.rowsOf(b.x, R), st.pos, R, @tagName(l.kind));
}

/// cc.compute with the recorder's taps (every layer's, then hidden and, with logits, fnormed).
pub fn compute(f: *Fwd, w: *const views.Weights, st: *State, b: *const Buffers, R: usize, opt: layer.Opt, logits: bool, tap: Tap) !?T {
    const m = f.m;
    const D = m.hidden;
    if (R == 0 or R > b.rows) return error.Rows;
    if (opt.prompt) |p| if (p.cut) |c| if (!b.prefill or c.point == 0 or c.point >= R) return error.Cut;
    try f.tri.embed(b.ids.ptr, cc.table(w.embed), b.x.ptr, R, D, m.streams);
    for (w.layers) |*l| try layerForward(f, l, st, b, R, opt, tap);
    if (b.prefill and st.rec.shape[1] > 0) st.parity = 1 - st.parity;
    try f.tri.streamMean(b.x.ptr, b.hidden.ptr, R, D, m.streams);
    try tap.put("hidden", .{}, layer.rowsOf(b.hidden, R), st.pos, R, null);
    if (!logits) return null;
    try f.tri.rmsnorm(b.hidden.ptr, D, w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, m.eps, R, D);
    try tap.put("fnormed", .{}, layer.rowsOf(b.fnormed, R), st.pos, R, null);
    const first = if (b.prefill) R - 1 else 0;
    const out = layer.rowsOf(b.logits, R - first);
    try qmm.mm(f, b.prefill, cc.rows(b.fnormed, first, R - first), w.head, cc.rows(b.fxs, first, R - first), out, false, b.sk, opt.chip);
    return out;
}
