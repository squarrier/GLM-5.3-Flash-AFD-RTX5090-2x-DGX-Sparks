//! The local AI adapter adapts V3's four-lane layer-major pipeline to current TensorFold state/commit APIs.
//! Hugh Madden authored the original lanes; Ash Hart authored TensorFold/MCDMA.
//! Native index rings hold4096 rows: use four1024-row lanes per full native4096-row
//! step, not four2048-row lanes which would overrun that ring. Pair tails stay native.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const pair_mod = @import("cuda_prompt_pair.zig");
const snapshots = @import("cuda_snapshots.zig");
pub const Testing = if (@import("builtin").is_test) struct {
    pub var prefill_views: ?*const fn ([]const *const fw.Forward, []const usize) anyerror!void = null;
} else struct {};
pub fn prefill(e: *engine.Engine, part: []const u32, start: usize, len: usize, cancel: ?engine.Cancel, marks: ?engine.Marks, split: bool) !?u32 {
    const rows = pair_mod.laneRows();
    const total = 4 * rows[0];
    if (e.w.role != .attention or len - start < total or part.len != @min(len - start, total + 1) or e.drafter != null or e.prompt_logprobs != null) return error.BadChunk;
    const p = if (e.pair) |*p| p else return error.BadChunk;
    if (p.extra[0] == null or p.extra[1] == null or !pair_mod.ringsAgree(e.cur.st.base, e.cur.st.ring_rows, start, fw.prefill_rows)) return error.BadChunk;
    if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    e.gate(cancel) catch |err| {
        if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
        return err;
    };
    const buffers = [_]*fw.Buffers{ &e.bp, &p.bp, &p.extra[0].?.bp, &p.extra[1].?.bp };
    const pos = [_]u64{ 0, p.pos.ptr, p.extra[0].?.pos.ptr, p.extra[1].?.pos.ptr };
    const st = &e.cur.st;
    try st.ready(e.stream, total);
    st.pending = rows[0];
    st.prompt = true;
    var states: [3]@import("cuda_state.zig").State = undefined;
    var ranks: [4][1]fw.Rank = undefined;
    var forwards: [4]fw.Forward = undefined;
    var ptrs: [4]*const fw.Forward = undefined;
    const ids: []const i32 = @ptrCast(part);
    for (buffers, rows, 0..) |b, n, i| {
        forwards[i] = e.view(b, &ranks[i]);
        forwards[i].prof = null;
        forwards[i].index_split = split;
        ptrs[i] = &forwards[i];
        if (i > 0) {
            states[i - 1] = st.*;
            states[i - 1].pos_dev = pos[i];
            try states[i - 1].setPos(e.stream, st.pos + i * rows[0]);
            ranks[i][0].st = &states[i - 1];
        }
        try cuda.copy_source.enqueue(e.d, b.ids, std.mem.sliceAsBytes(ids[i * rows[0] ..][0..n]), e.stream);
    }
    if(comptime @import("builtin").is_test) if(Testing.prefill_views) |observe| try observe(&ptrs,&rows);
    try fw.Forward.lanesBody(&ptrs, &rows);
    p.steps += 1;
    p.lane_steps += 1;
    e.index_splits += @intFromBool(split);
    // Preserve native prompt/MTP absorb and commits in exact chunk order. Only the
    // accepted/committed prefix changes the bound user's position and recurrent state.
    var first: ?u32 = null;
    for (buffers, rows, 0..) |b, n, i| {
        st.pending = n;
        st.prompt = true;
        if (i == 3 and start + total == len) {
            var token: [1]u32 = undefined;
            try e.draw(b.logits, 1, len, 0, &token);
            first = token[0];
            try e.holdPrompt(b.fnormed + (n - 1) * e.c.hidden * 2);
        }
        if (e.mtp) {
            const offset = i * rows[0] + 1;
            const end = @min(part.len, offset + n);
            if (end > offset) {
                try forwards[0].mtp(ids[offset..end], &.{b.fnormed});
                try st.setMtpLen(e.stream, st.mtp_len + end - offset);
            }
        }
        try forwards[0].commit(n);
    }
    return first;
}
