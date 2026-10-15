//! bench's rows=R+... (qualification_qualification_4ab1ea8f, qualification_1bae3bb5 §7 item 4): the R-row logits check. Is a verify window's logits row the serial
//! step's, bit for bit? On the one load, after the forms' runs, the decode prompt's greedy chain runs serially (bench's
//! step: a 1-row Engine.forward, greedy, commit), each step's logits row kept on the host; then, for each size R, the
//! same chain again in teacher-forced verify windows: Engine.forward of R chain tokens (a round whose drafts are all
//! the target's, so all R rows are kept) and commit of R rows. Each window row is compared with the serial step at the
//! same position: bitwise, the largest |difference| and the shard's top 2. OUT/rows/rows-rankR.json holds the counts,
//! every differing step and each step's serial top 2; the bf16 rows of the steps that differ (up to 64 a size) and of
//! steps 0-3 and every 64th go to OUT/rows/R<R>/<k>.{serial,verify}.bf16 (k: the step, its input token at position
//! prompt + k). On --dry the rows are zeros (no GPU): the launches run and every row compares equal.
const std = @import("std");
const glm = @import("glm_cuda");
const bench = @import("glm_bench.zig");
const runv = @import("glm_run.zig");

const Engine = glm.engine.Engine;
const Fwd = glm.fwd.Fwd;
const T = glm.trace.T;
const Host = runv.Run.HostT;
const Args = runv.Run.ArgsT;

/// rows written a size beyond steps 0-3 and every 64th
const max_written = 64;

/// Row r of a [R, V/world] logits view.
fn rowOf(x: T, r: usize) T {
    var y = x;
    y.ptr += r * x.stride[0] * x.dtype.size();
    y.shape[0] = 1;
    return y;
}

/// A logits row on the host as the rank holds it: bf16 bits (Host.logitsRow widens them exactly; --dry: zeros).
fn bf16Row(h: Host, gpa: std.mem.Allocator, x: T, out: []u16) !void {
    const w = try h.logitsRow(gpa, x);
    defer gpa.free(w);
    for (w, out) |v, *b| b.* = @truncate(@as(u32, @bitCast(v)) >> 16);
}

fn value(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The shard's top 2 (ids with the rank's offset added; the first index wins a tie, as torch's argmax).
const Top2 = struct {
    id: [2]usize = .{ 0, 0 },
    v: [2]f32 = .{ -std.math.inf(f32), -std.math.inf(f32) },

    fn of(row: []const u16, off: usize) Top2 {
        var t: Top2 = .{};
        for (row, 0..) |b, i| {
            const v = value(b);
            if (v > t.v[0]) {
                const old = t;
                t = .{ .id = .{ i + off, old.id[0] }, .v = .{ v, old.v[0] } };
            } else if (v > t.v[1]) {
                t.id[1] = i + off;
                t.v[1] = v;
            }
        }
        return t;
    }

    fn json(t: Top2, js: *std.ArrayList(u8), gpa: std.mem.Allocator) !void {
        try js.print(gpa, "[{d}, {d}, {d}, {d}]", .{ t.id[0], t.v[0], t.id[1], t.v[1] });
    }
};

fn maxDiff(a: []const u16, b: []const u16) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(value(x) - value(y)));
    return m;
}

fn writeRow(dir: []const u8, k: usize, kind: []const u8, row: []const u16) !void {
    var pb: [1024]u8 = undefined;
    try bench.writeFile(try std.fmt.bufPrint(&pb, "{s}/{d}.{s}.bf16", .{ dir, k, kind }), std.mem.sliceAsBytes(row));
}

/// The check, after the forms' runs: the spec's rows=R+... (each 1..max_rows) over its decode prompt and steps.
pub fn run(e: *Engine, f: *Fwd, a: Args, spec: bench.Spec, h: Host, rearm: bench.Rearm, gpa: std.mem.Allocator) !void {
    var p = try bench.Prompts.open(gpa, a.prompt);
    defer p.deinit();
    const dp = try p.ids(spec.decode);
    defer gpa.free(dp);
    const out, const R, const steps, const sizes = .{ a.out, a.rank, spec.steps, spec.rows };
    const vl = e.m.vl();
    const off = e.m.rank * vl;
    const chain = try gpa.alloc(u32, steps + 1);
    defer gpa.free(chain);
    const serial = try gpa.alloc(u16, steps * vl);
    defer gpa.free(serial);
    const tops = try gpa.alloc(Top2, steps);
    defer gpa.free(tops);
    const row = try gpa.alloc(u16, vl);
    defer gpa.free(row);
    rearm.now();
    chain[0] = try e.prefill(f, dp, false);
    for (0..steps) |k| {
        const lg = try bench.step(e, f, chain[k], &chain[k + 1]);
        const s = serial[k * vl ..][0..vl];
        try bf16Row(h, gpa, rowOf(lg, 0), s);
        tops[k] = Top2.of(s, off);
    }
    var base: [1024]u8 = undefined;
    const dir = try std.fmt.bufPrint(&base, "{s}/rows", .{out});
    try bench.mkdirs(dir);
    var js: std.ArrayList(u8) = .empty;
    defer js.deinit(gpa);
    try js.print(gpa, "{{\"rank\": {d}, \"dry\": {}, \"from\": {d}, \"steps\": {d}, \"vl\": {d}, \"chain\": [", .{ R, f.gpu == null, dp.len, steps, vl });
    for (chain, 0..) |x, i| try js.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", x });
    try js.appendSlice(gpa, "], \"serial_top2\": [");
    for (tops, 0..) |t, i| {
        if (i > 0) try js.appendSlice(gpa, ", ");
        try t.json(&js, gpa);
    }
    try js.appendSlice(gpa, "], \"sizes\": [");
    var it = std.mem.tokenizeScalar(u8, sizes, '+');
    var first = true;
    while (it.next()) |sz| : (first = false) {
        const n = try std.fmt.parseInt(usize, sz, 10);
        if (n == 0 or n > glm.decode.max_rows) return error.RowsSize;
        var sub: [1024]u8 = undefined;
        const rdir = try std.fmt.bufPrint(&sub, "{s}/R{d}", .{ dir, n });
        try bench.mkdirs(rdir);
        rearm.now();
        const first_tok = try e.prefill(f, dp, false);
        var forwards: usize = 0;
        var equal: usize = 0;
        var top_equal: usize = 0;
        var written: usize = 0;
        var largest: f32 = 0;
        var diffs: std.ArrayList(u8) = .empty;
        defer diffs.deinit(gpa);
        var wl: std.ArrayList(u8) = .empty;
        defer wl.deinit(gpa);
        var k: usize = 0;
        while (k < steps) {
            const m = @min(n, steps - k);
            const lg = try e.forward(f, chain[k .. k + m]);
            forwards += 1;
            for (0..m) |r| {
                const s = serial[(k + r) * vl ..][0..vl];
                try bf16Row(h, gpa, rowOf(lg, r), row);
                const same = std.mem.eql(u16, row, s);
                const d = maxDiff(row, s);
                const vt = Top2.of(row, off);
                equal += @intFromBool(same);
                top_equal += @intFromBool(vt.id[0] == tops[k + r].id[0]);
                largest = @max(largest, d);
                if (!same) {
                    try diffs.print(gpa, "{s}{{\"k\": {d}, \"row\": {d}, \"max_abs_diff\": {d}, \"serial\": ", .{ if (diffs.items.len == 0) "" else ", ", k + r, r, d });
                    try tops[k + r].json(&diffs, gpa);
                    try diffs.appendSlice(gpa, ", \"verify\": ");
                    try vt.json(&diffs, gpa);
                    try diffs.append(gpa, '}');
                }
                const kk = k + r;
                if ((!same and written < max_written) or kk < 4 or kk % 64 == 0) {
                    try writeRow(rdir, kk, "serial", s);
                    try writeRow(rdir, kk, "verify", row);
                    try wl.print(gpa, "{s}{d}", .{ if (wl.items.len == 0) "" else ", ", kk });
                    written += @intFromBool(!same);
                }
            }
            try glm.compute.commit(f, &e.st, &e.buf, m, m);
            k += m;
        }
        std.debug.print("ROWS rank {d}: R {d}: {d} verify forwards over the chain's {d} steps from position {d}: {d} rows bitwise equal to the serial step's, the largest |difference| {d}, the shard's argmax equal in {d}; first token {d} (serial {d})\n", .{ R, n, forwards, steps, dp.len, equal, largest, top_equal, first_tok, chain[0] });
        try js.print(gpa, "{s}{{\"rows\": {d}, \"forwards\": {d}, \"compared\": {d}, \"bitwise_equal\": {d}, \"argmax_equal\": {d}, \"max_abs_diff\": {d}, \"first\": {d}, \"diff\": [{s}], \"written\": [{s}]}}", .{ if (first) "" else ", ", n, forwards, steps, equal, top_equal, largest, first_tok, diffs.items, wl.items });
    }
    try js.appendSlice(gpa, "]}\n");
    var pb: [1024]u8 = undefined;
    try bench.writeFile(try std.fmt.bufPrint(&pb, "{s}/rows-rank{d}.json", .{ dir, R }), js.items);
}
