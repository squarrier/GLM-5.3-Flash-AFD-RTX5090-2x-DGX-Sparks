//! V3 two-flight shared-user port onto current TensorFold state, masking and draw interfaces.
//! TensorFold/MTP: Ash Hart. Shared-window design: @Bizuayeu. AFD: Hugh Madden and T.J. Purtell.
//! Helios (GPT-6.1 Sol/OpenAI Codex via Hermes) authored these minimal native adapters.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
pub const Testing = if (@import("builtin").is_test) struct {
    pub var before: ?*const fn (*const [2]*const fw.Forward, [2]usize) anyerror!void = null;
    pub var draw: ?*const fn (*engine.Engine, engine.Part) anyerror!void = null;
} else struct {};
const Group = struct { ids: [engine.shared_rows]u32 = undefined, segs: [engine.max_shared]fw.Seg = undefined, index: [engine.max_shared]usize = undefined, count: usize = 0, rows: usize = 0 };
/// Validate every member before staging/device/state mutation. No width/context/user cap reduction.
pub fn verify(e: *engine.Engine, parts: []const engine.Part) !void {
    if (e.routed) |p| try p.check(p.ctx);
    if (parts.len < 2 or parts.len > engine.max_shared) return error.WindowTooWide;
    const a = if (e.bs) |*b| b else return error.NoSharedWindows;
    const b = if (e.bsp) |*b| b else return error.NoSharedWindows;
    if (a.ids == b.ids) return error.BadPair;
    var total: usize = 0;
    for (parts, 0..) |p, i| {
        if (p.ids.len == 0 or p.ids.len > fw.window_rows or p.out.len < p.ids.len or p.ids.len > engine.shared_rows - total) return error.WindowTooWide;
        if (p.slot >= e.seqs.items.len or e.seqs.items[p.slot] == null) return error.UnknownSlot;
        for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
        total += p.ids.len;
    }
    var gs: [2]Group = .{ .{}, .{} };
    for (parts, 0..) |p, i| {
        const j: usize = if (gs[0].rows <= gs[1].rows) 0 else 1;
        const g = &gs[j];
        const q = e.seqs.items[p.slot].?;
        g.segs[g.count] = .{ .st = &q.st, .row0 = g.rows, .rows = p.ids.len };
        g.index[g.count] = i;
        g.count += 1;
        @memcpy(g.ids[g.rows..][0..p.ids.len], p.ids);
        g.rows += p.ids.len;
    }
    const buffers = [_]*fw.Buffers{ a, b };
    for (gs, buffers) |g, buf| if (g.rows == 0 or g.rows > buf.rows) return error.WindowTooWide;
    var ranks: [2][1]fw.Rank = undefined;
    var fs: [2]fw.Forward = undefined;
    var ptrs: [2]*const fw.Forward = undefined;
    var rows: [2]usize = undefined;
    for (&gs, buffers, &ranks, &fs, &ptrs, &rows) |*g, buf, *rank, *f, *ptr, *R| {
        if (g.rows == 0 or g.rows > buf.rows) return error.WindowTooWide;
        f.* = e.view(buf, rank);
        rank[0].st = g.segs[0].st;
        rank[0].segs = g.segs[0..g.count];
        f.pieces = null;
        f.prof = null;
        f.taps = null;
        ptr.* = f;
        R.* = g.rows;
        try cuda.copy_source.enqueue(e.d, buf.ids, std.mem.sliceAsBytes(g.ids[0..g.rows]), e.stream);
    }
    if (comptime @import("builtin").is_test) if (Testing.before) |observe| try observe(&ptrs, rows);
    try @import("cuda_forward_pair.zig").pingPong(&ptrs, rows);
    // Completion of a network plane is not CUDA consumption. No host draw or prefix
    // publication until BOTH complete forwards' stream work is proved and checked.
    try e.stream.synchronize();
    try e.checkGathers();
    for (&gs, buffers) |*g, buf| for (g.index[0..g.count], g.segs[0..g.count]) |original, seg| {
        const p = parts[original];
        try e.bind(p.slot);
        e.cur.rows_at = buf.fnormed + seg.row0 * e.c.hidden * 2;
        if (comptime @import("builtin").is_test) if (Testing.draw) |draw| {
            try draw(e, p);
            continue;
        };
        try e.draw(buf.logits + seg.row0 * e.share.vocab.len() * 2, p.ids.len, seg.st.pos + 1, p.bits, p.out);
    };
}
