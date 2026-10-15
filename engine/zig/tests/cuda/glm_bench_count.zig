//! bench's launch counts on --dry (qualification_358eae44): a prefill's launches a chunk and a decode's a step, printed as LAUNCHES lines
//! and kept for the record's `launches`. On a GPU nothing here runs (`launches` is 0 and prefillOf is e.prefill).
const std = @import("std");
const glm = @import("glm_cuda");

const Engine = glm.engine.Engine;
const Fwd = glm.fwd.Fwd;
const T = glm.trace.T;

/// --dry: the launches recorded so far (the trace's events, and the Triton launches not yet moved into it); 0 on a GPU.
pub fn launches(f: *Fwd) usize {
    const tr = f.rec orelse return 0;
    return tr.events.items.len + tr.tri_rec.n;
}

/// --dry: a prefill's count at each chunk's `fnormed` tap (glm/cuda_dump.zig's last tap before the chunk's logits; the
/// taps launch nothing, and this one only counts).
const Chunks = struct {
    f: *Fwd,
    gpa: std.mem.Allocator,
    at: std.ArrayList(usize) = .empty,

    fn tensor(ctx: *anyopaque, name: []const u8, x: T, pos0: usize, rows: usize, k: ?[]const u8) anyerror!void {
        _ = .{ x, pos0, rows, k };
        const c: *Chunks = @ptrCast(@alignCast(ctx));
        if (std.mem.eql(u8, name, "fnormed")) try c.at.append(c.gpa, launches(c.f));
    }
};

/// e.prefill, with the MTP absorb when `mtp` (bench --drafts); with `cj` (--dry, the warm-up), its launches a chunk
/// too: chunk k's run from chunk k-1's fnormed tap (the prefill's start for the first) to its own; the rest come after
/// the last.
pub fn prefillOf(e: *Engine, f: *Fwd, ids: []const u32, mtp: bool, cj: ?*std.ArrayList(u8), name: []const u8, R: u32, gpa: std.mem.Allocator) !u32 {
    const j = cj orelse return e.prefill(f, ids, mtp);
    var c: Chunks = .{ .f = f, .gpa = gpa };
    defer c.at.deinit(gpa);
    const n0 = launches(f);
    e.tap = .{ .ctx = &c, .tensor = Chunks.tensor };
    defer e.tap = null;
    const tok = try e.prefill(f, ids, mtp);
    const n1 = launches(f);
    const per = try gpa.alloc(usize, c.at.items.len);
    defer gpa.free(per);
    var prev = n0;
    for (c.at.items, per) |x, *y| {
        y.* = x - prev;
        prev = x;
    }
    var buf: [768]u8 = undefined;
    std.debug.print("LAUNCHES rank {d}: prefill of {d} tokens ({s}) in {d} chunks of up to {d} rows: {d} launches; a chunk: {s}; after the last chunk's fnormed {d}\n", .{ R, ids.len, name, per.len, e.prefill_rows, n1 - n0, runsText(per, &buf), n1 - prev });
    try j.print(gpa, "{s}{{\"prefill\": \"{s}\", \"tokens\": {d}, \"chunk_rows\": {d}, \"total\": {d}, \"after_last\": {d}, \"chunks\": ", .{ if (j.items.len == 0) "" else ", ", name, ids.len, e.prefill_rows, n1 - n0, n1 - prev });
    try countsJson(j, gpa, per);
    try j.append(gpa, '}');
    return tok;
}

/// --dry: the warm-up decode's launches a step.
pub fn stepCounts(j: *std.ArrayList(u8), gpa: std.mem.Allocator, R: u32, counts: []const usize, name: []const u8, from: usize) !void {
    var total: usize = 0;
    for (counts) |x| total += x;
    var buf: [768]u8 = undefined;
    std.debug.print("LAUNCHES rank {d}: decode of {d} steps from position {d} ({s}): {d} launches; a step: {s}\n", .{ R, counts.len, from, name, total, runsText(counts, &buf) });
    try j.print(gpa, "{s}{{\"decode\": \"{s}\", \"from\": {d}, \"steps\": {d}, \"total\": {d}, \"a_step\": ", .{ if (j.items.len == 0) "" else ", ", name, from, counts.len, total });
    try countsJson(j, gpa, counts);
    try j.append(gpa, '}');
}

/// `xs` as runs of equal values ("v x n, ..."), cut where `buf` ends.
pub fn runsText(xs: []const usize, buf: []u8) []const u8 {
    var w: usize = 0;
    var i: usize = 0;
    while (i < xs.len) {
        var k = i;
        while (k < xs.len and xs[k] == xs[i]) k += 1;
        const s = std.fmt.bufPrint(buf[w..], "{s}{d} x {d}", .{ if (i == 0) "" else ", ", xs[i], k - i }) catch return buf[0..w];
        w += s.len;
        i = k;
    }
    return buf[0..w];
}

pub fn countsJson(js: *std.ArrayList(u8), gpa: std.mem.Allocator, xs: []const usize) !void {
    try js.append(gpa, '[');
    for (xs, 0..) |x, i| try js.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", x });
    try js.append(gpa, ']');
}
