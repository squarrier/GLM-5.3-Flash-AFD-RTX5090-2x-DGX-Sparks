//! python-0.6's decode step for a TP rank (glm5_next/cuda/forward.py, ed78d6f: check_room, stage, compute, commit,
//! _shift_conv, State.set_pos): the window's ids through the pinned host rows, the embedding, every layer
//! (cuda_layer.layerForward), the streams' mean, the final norm and the head on the rank's vocabulary rows; commit keeps
//! the window's first rows (kda.replay_layers when it drops some) and shifts the conv windows. Its torch ops are calls of
//! *Fwd as the extension calls are: recorded as python's `torch.<op>` (qualification_qualification_7c75d777 PLAN §4), or run on the rank's stream.
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const tri = @import("cuda_triton.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const prompt = @import("cuda_prompt.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;
const Tmp = @import("cuda_tmp.zig").Tmp;

const T = trace.T;
const Val = trace.Val;
const Buffers = bufs.Buffers;
const State = bufs.State;

/// attention.CHUNK: keys a chunk program
pub const chunk = 512;

/// forward.check_room: the window fits under the dense limit (without the indexer's long contexts) and the capacity.
pub fn checkRoom(m: *const md.Model, st: *const State, R: usize, pos: ?usize) !void {
    const p = pos orelse st.pos;
    if (p + R > m.denseLimit() and st.index == null) return error.PastDenseLimit;
    if (p + R > st.capacity) return error.PastCapacity;
}

/// forward.chunks_for: 512-key chunks through the window's last row.
pub fn chunksFor(st: *const State, R: usize) usize {
    return (st.pos + R + chunk - 1) / chunk;
}

/// A contiguous tensor's bytes.
pub fn bytes(x: T) usize {
    var n: usize = x.dtype.size();
    for (x.shape[0..x.dims]) |s| n *= s;
    return n;
}

/// Rows [first, first + n) of a [rows, cols] buffer.
pub fn rows(b: views.Buf, first: usize, n: usize) T {
    const cols = b.shape[1];
    return T.rows2(b, first * cols * b.dtype.size(), n, cols, cols);
}

/// glue.embed's table from the weights' (MLX rows as stored, or BF16).
pub fn table(e: views.Table) tri.Table {
    return switch (e) {
        .q4 => |x| .{ .q4 = .{ .w = x[0].ptr, .s = x[1].ptr, .b = x[2].ptr } },
        .b16 => |x| .{ .b16 = x.ptr },
    };
}

// -- torch ops: each recorded as python's call, or run on the rank's stream ---------------------------------------

const Dev = struct { d: *const cuda.Driver, s: cuda.abi.Stream };

fn dev(f: *const Fwd) !Dev {
    const g = f.gpu orelse return error.NoGpu;
    return .{ .d = g.stream.d, .s = g.stream.handle };
}

/// Buffers.staged.synchronize(): the last staged copy has left the pinned rows.
pub fn stagedSync(f: *Fwd, b: *const Buffers) !void {
    if (f.rec) |t| return t.call(.ext, "staged.synchronize", &.{});
    const e = b.staged orelse return error.NoStagedEvent;
    try e.synchronize();
}

/// Buffers.staged.record() on the rank's stream.
pub fn stagedRecord(f: *Fwd, b: *const Buffers) !void {
    if (f.rec) |t| return t.call(.ext, "staged.record", &.{});
    const e = b.staged orelse return error.NoStagedEvent;
    try e.record((f.gpu orelse return error.NoGpu).stream);
}

/// `b.ids_host[:R].numpy()[:] = tokens`: the window's ids into the pinned host rows.
pub fn hostWrite(f: *Fwd, b: *const Buffers, tokens: []const u32) !void {
    const dst = T.of(b.ids_host, 0, &.{tokens.len});
    if (f.rec) |t| {
        const vals = try t.arena.allocator().alloc(Val, tokens.len);
        for (tokens, vals) |x, *v| v.* = .{ .int = x };
        return t.call(.ext, "host.write", &.{ .{ .t = dst }, .{ .list = vals } });
    }
    if (!b.ids_host.host or b.ids_host.ptr == 0) return error.NoHostRows;
    const out: [*]i32 = @ptrFromInt(b.ids_host.ptr);
    for (tokens, 0..) |x, i| out[i] = @intCast(x);
}

/// `dst.copy_(src, non_blocking=True)`: pinned host rows to the device, on the rank's stream.
pub fn copyHtoD(f: *Fwd, dst: T, src: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.copy_", &.{ .{ .t = dst }, .{ .t = src }, .{ .bool = true } });
    const g = try dev(f);
    try g.d.check(g.d.api.cuMemcpyHtoDAsync_v2(dst.ptr, @ptrFromInt(src.ptr), bytes(dst), g.s), "cuMemcpyHtoDAsync");
}

/// `dst.copy_(src)` between device rows (mtp_stage's hidden), on the rank's stream.
pub fn copyDtoD(f: *Fwd, dst: T, src: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.copy_", &.{ .{ .t = dst }, .{ .t = src } });
    const g = try dev(f);
    try g.d.check(g.d.api.cuMemcpyDtoDAsync_v2(dst.ptr, src.ptr, bytes(dst), g.s), "cuMemcpyDtoDAsync");
}

/// `x.fill_(value)` of int32 elements (State.set_pos, set_mtp_len).
pub fn fill(f: *Fwd, x: T, value: i64) !void {
    if (f.rec) |t| return t.call(.ext, "torch.fill_", &.{ .{ .t = x }, .{ .int = value } });
    if (x.dtype != .i32) return error.FillDtype;
    const g = try dev(f);
    const word: u32 = @bitCast(@as(i32, @intCast(value)));
    try g.d.check(g.d.api.cuMemsetD32Async(x.ptr, word, bytes(x) / 4, g.s), "cuMemsetD32Async");
}

/// `x.zero_()` (mtp_compute's position-0 embedding row).
pub fn zero(f: *Fwd, x: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.zero_", &.{.{ .t = x }});
    const g = try dev(f);
    try g.d.check(g.d.api.cuMemsetD8Async(x.ptr, 0, bytes(x), g.s), "cuMemsetD8Async");
}

/// `x.cpu()`: device bytes into the host's `out` (pinned), the stream drained as torch's copy is.
pub fn toHost(f: *Fwd, x: T, out: []u8) !void {
    if (f.rec) |t| return t.call(.ext, "torch.cpu", &.{.{ .t = x }});
    if (out.len < bytes(x)) return error.HostRows;
    const g = try dev(f);
    try g.d.check(g.d.api.cuMemcpyDtoHAsync_v2(out.ptr, x.ptr, bytes(x), g.s), "cuMemcpyDtoHAsync");
    try g.d.check(g.d.api.cuStreamSynchronize(g.s), "cuStreamSynchronize");
}

fn contiguous(x: T) bool {
    var s: usize = 1;
    var i: usize = x.dims;
    while (i > 0) {
        i -= 1;
        if (x.shape[i] != 1 and x.stride[i] != s) return false;
        s *= x.shape[i];
    }
    return true;
}

/// `x.clone()`: a new tensor (python's `tmp.clone`) in `dst` (device rows of its size on a GPU); gives its view.
pub fn clone(f: *Fwd, tmp: *Tmp, x: T, dst: *views.Buf) !T {
    if (!contiguous(x)) return error.NotContiguous;
    try tmp.made("clone", dst);
    const out = T.of(dst.*, 0, x.shape[0..x.dims]);
    if (f.rec) |t| {
        try t.call(.ext, "torch.clone", &.{ .{ .t = x }, .{ .t = out } });
        return out;
    }
    if (bytes(out) > dst.bytes()) return error.TmpRows;
    const g = try dev(f);
    try g.d.check(g.d.api.cuMemcpyDtoDAsync_v2(out.ptr, x.ptr, bytes(out), g.s), "cuMemcpyDtoDAsync");
    return out;
}

/// `torch.cat((a, b))` of two row blocks into a new tensor (python's `tmp.cat`) in `dst`; gives its view.
pub fn cat(f: *Fwd, tmp: *Tmp, a: T, b: T, dst: *views.Buf) !T {
    if (!contiguous(a) or !contiguous(b) or a.dims != b.dims) return error.NotContiguous;
    try tmp.made("cat", dst);
    var shape = a.shape;
    shape[0] += b.shape[0];
    const out = T.of(dst.*, 0, shape[0..a.dims]);
    if (f.rec) |t| {
        try t.call(.ext, "torch.cat", &.{ .{ .list = &.{ .{ .t = a }, .{ .t = b } } }, .{ .t = out } });
        return out;
    }
    if (bytes(out) > dst.bytes()) return error.TmpRows;
    const g = try dev(f);
    try g.d.check(g.d.api.cuMemcpyDtoDAsync_v2(out.ptr, a.ptr, bytes(a), g.s), "cuMemcpyDtoDAsync");
    try g.d.check(g.d.api.cuMemcpyDtoDAsync_v2(out.ptr + bytes(a), b.ptr, bytes(b), g.s), "cuMemcpyDtoDAsync");
    return out;
}

// -- the step -----------------------------------------------------------------------------------------------------

/// stage's and mtp_stage's first half: wait for the pinned rows, write the ids, copy them to b.ids[:R].
pub fn stageIds(f: *Fwd, b: *const Buffers, tokens: []const u32) !void {
    try stagedSync(f, b);
    try hostWrite(f, b, tokens);
    try copyHtoD(f, T.of(b.ids, 0, &.{tokens.len}), T.of(b.ids_host, 0, &.{tokens.len}));
}

/// forward.stage: the window's ids into b.ids (checked against the room left); gives R.
pub fn stage(f: *Fwd, st: *const State, b: *const Buffers, tokens: []const u32) !usize {
    const R = tokens.len;
    if (R == 0 or R > b.rows) return error.Rows;
    try checkRoom(f.m, st, R, null);
    try stageIds(f, b, tokens);
    try stagedRecord(f, b);
    return R;
}

/// forward.compute for a window or a prompt chunk (opt.prompt: its cut and temporaries): the embedding, every layer,
/// the streams' mean and, with `logits`, the final norm and the head into b.logits (a prompt chunk's last row only);
/// gives the logits' view. A prompt chunk's KDA layers commit as they run (python's st.cur[li], flipped once here).
pub fn compute(f: *Fwd, w: *const views.Weights, st: *State, b: *const Buffers, R: usize, opt: layer.Opt, logits: bool) !?T {
    const m = f.m;
    const D = m.hidden;
    if (R == 0 or R > b.rows) return error.Rows;
    if (opt.prompt) |p| if (p.cut) |c| if (!b.prefill or c.point == 0 or c.point >= R) return error.Cut;
    try f.tri.embed(b.ids.ptr, table(w.embed), b.x.ptr, R, D, m.streams);
    for (w.layers) |*l| try prompt.layerForward(f, l, st, b, R, opt); // a prompt chunk's DSA layer: the engine's Scratch
    if (b.prefill and st.rec.shape[1] > 0) st.parity = 1 - st.parity;
    try f.tri.streamMean(b.x.ptr, b.hidden.ptr, R, D, m.streams);
    if (!logits) return null;
    try f.tri.rmsnorm(b.hidden.ptr, D, w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, m.eps, R, D);
    const first = if (b.prefill) R - 1 else 0; // the head reads a prompt chunk's last row (fnormed keeps all for the MTP)
    const out = layer.rowsOf(b.logits, R - first);
    try qmm.mm(f, b.prefill, rows(b.fnormed, first, R - first), w.head, rows(b.fxs, first, R - first), out, false, b.sk, opt.chip);
    return out;
}

/// forward.forward: stage, then compute with the window's chunks at the host position; gives the logits' view.
pub fn forward(f: *Fwd, w: *const views.Weights, st: *State, b: *const Buffers, tokens: []const u32, chip: qmm.Chip) !T {
    const R = try stage(f, st, b, tokens);
    return (try compute(f, w, st, b, R, .{ .nch = chunksFor(st, R), .host_pos = st.pos, .chip = chip }, true)).?;
}

/// kda.replay_layers(st.rec[cur], st.scratch_set, keep, st.rec[1 - cur]): every KDA layer's state after the kept rows.
fn replayLayers(f: *Fwd, st: *const State, cur: u1, keep: usize) !void {
    const n = st.rec.shape[1];
    const H = st.rec.shape[2];
    const s = st.scratch_set;
    const window = s.k.shape[1];
    const half = n * H * 128 * 128 * 4;
    const in = T.of(st.rec, @as(usize, cur) * half, &.{ n, H, 128, 128 });
    const out = T.of(st.rec, @as(usize, 1 - cur) * half, &.{ n, H, 128, 128 });
    if (f.rec) |t| return t.call(.ext, "kda.replay_layers", &.{
        .{ .t = in },                              .{ .int = @intCast(H * 128 * 128) }, .{ .t = T.whole(s.k) },
        .{ .t = T.whole(s.v) },                    .{ .t = T.whole(s.g) },              .{ .t = T.whole(s.b) },
        .{ .int = @intCast(window * H * 128) },    .{ .int = @intCast(window * H) },    .{ .int = @intCast(n) },
        .{ .int = @intCast(H) },                   .{ .int = @intCast(keep) },          .{ .t = out },
    });
    const g = f.gpu orelse return error.NoGpu;
    try g.kernels.kda.replayLayers(g.stream, .{
        .state_in = in.ptr, .state_stride = H * 128 * 128, .k_save = s.k.ptr, .v_save = s.v.ptr, .g_save = s.g.ptr,
        .b_save = s.b.ptr, .kv_stride = window * H * 128, .b_stride = window * H, .layers = n, .heads = H,
        .rows = keep, .state_out = out.ptr,
    });
}

/// forward._shift_conv(st.conv, st.proj, keep): each KDA layer's window takes rows keep .. keep + 2 of [window; rows].
fn shiftConv(f: *Fwd, st: *const State, keep: usize) !void {
    if (st.conv.shape[1] != 3) return error.ConvTaps; // python's kernel is written for 4-tap convolutions
    const C = st.conv.shape[2];
    try f.tri.shiftConv(st.conv.ptr, st.proj.ptr, keep, st.conv.shape[0], 3 * C, st.proj.shape[1] * st.proj.shape[2], st.proj.shape[2], C);
}

/// State.set_pos: the host position and pos_dev.
pub fn setPos(f: *Fwd, st: *State, pos: usize) !void {
    st.pos = pos;
    try fill(f, T.whole(st.pos_dev), @intCast(pos));
}

/// State.set_mtp_len: the MTP cache's rows and mtp_pos_dev.
pub fn setMtpLen(f: *Fwd, st: *State, n: usize) !void {
    st.mtp_len = n;
    try fill(f, T.whole(st.mtp_pos_dev), @intCast(n));
}

/// State.reset: zeroed conv windows and recurrent states, both positions 0 (decode.prefill starts a prompt here).
pub fn reset(f: *Fwd, st: *State) !void {
    try zero(f, T.whole(st.conv));
    try zero(f, T.whole(st.rec));
    st.parity = 0;
    try setPos(f, st, 0);
    try setMtpLen(f, st, 0);
    st.mtp_drafted = 0;
}

/// forward.commit: keep the window's first `keep` rows (all of a prompt chunk's, whose KDA layers committed already).
pub fn commit(f: *Fwd, st: *State, b: *const Buffers, R: usize, keep: usize) !void {
    if (keep < 1 or keep > R or (b.prefill and keep != R)) return error.Keep;
    const n = if (b.prefill) 0 else st.rec.shape[1];
    if (n > 0) {
        const cur = st.parity;
        if (keep < R) try replayLayers(f, st, cur, keep);
        st.parity = 1 - cur;
        try shiftConv(f, st, keep);
    }
    try setPos(f, st, st.pos + keep);
}

test "check_room: the dense limit without the indexer, then the capacity" {
    const gpa = std.testing.allocator;
    const m = try md.Model.from(&@import("config.zig").Config{}, .{}, 2, 0);
    var st = try State.init(gpa, &m, 2560, 8, false);
    defer st.deinit(gpa);
    try checkRoom(&m, &st, 8, 2043);
    try std.testing.expectError(error.PastDenseLimit, checkRoom(&m, &st, 8, 2044));
    var long = try State.init(gpa, &m, 2560, 8, true);
    defer long.deinit(gpa);
    try checkRoom(&m, &long, 8, 2552);
    try std.testing.expectError(error.PastCapacity, checkRoom(&m, &long, 8, 2553));
    st.pos = 511;
    try std.testing.expectEqual(@as(usize, 1), chunksFor(&st, 1));
    try std.testing.expectEqual(@as(usize, 2), chunksFor(&st, 2));
}
