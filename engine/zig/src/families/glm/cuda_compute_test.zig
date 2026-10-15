//! qualification_qualification_44ba554e.5 B2's launch-trace check of a whole decode step (qualification_qualification_7c75d777 PLAN §4): python-0.6's stage, compute, greedy sample_rows
//! and commit, and mtp_stage + mtp_compute, recorded on a CPU by tools/zig/glm_step_fixtures.py over a checkpoint's
//! headers (MLX 4-bit; EXL3, BF16 dense) for a TP2 rank cut to layers 0-2 with the MTP head (layers 3+ and the MTP
//! layer's blocks are lane A's). The Zig rank places every buffer at a stand-in address and must make python's launches.
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const layer = @import("cuda_layer.zig");
const cc = @import("cuda_compute.zig");
const sample = @import("cuda_sample.zig");
const mtp = @import("cuda_mtp.zig");
const lt = @import("cuda_layer_test.zig");

const testing = std.testing;
const J = std.json.Value;
const fixture_mlx = @embedFile("fixtures_cuda_step.json");
const fixture_exl3 = @embedFile("fixtures_cuda_step_exl3.json");
const fixture_choose = @embedFile("fixtures_cuda_sample.json");
const fixture_topk = @embedFile("fixtures_cuda_topk.json");

/// python's buffers whose views are lane A's (the MTP layer's DSA and MoE): left to lane A's gates, on both sides.
const lane_a = [_][]const u8{ "w.mtp.layer.dsa.", "w.mtp.layer.moe." };

pub fn get(o: J, key: []const u8) J {
    return o.object.get(key) orelse std.debug.panic("fixture: no key {s}", .{key});
}

pub fn int(v: J) usize {
    return @intCast(v.integer);
}

/// The rank of a step fixture: its model (python's Config, then cut with the MTP head kept), weights, both Buffers
/// (the main window's and the MTP head's, python's `mbuf`), the state, the sampler's scratch, and their stand-in space.
const Rank = struct {
    m: md.Model,
    w: views.Weights,
    b: bufs.Buffers,
    mb: bufs.Buffers,
    st: bufs.State,
    smp: sample.Sampler,
    space: trace.Space,

    fn init(gpa: std.mem.Allocator, fx: J) !*Rank {
        const r = try gpa.create(Rank);
        errdefer gpa.destroy(r);
        r.m = try md.Model.parse(gpa, get(fx, "config_json").string, int(get(fx, "world")), int(get(fx, "rank")));
        try lt.expectConfig(&r.m, get(fx, "config"));
        const mtp_layers = r.m.mtp_layers;
        try r.m.cut(int(get(fx, "layers"))); // qualification_6607f886's config copy: the first n layers ...
        if (get(fx, "mtp").bool) r.m.mtp_layers = mtp_layers; // ... and the checkpoint's MTP head, as the run loaded it
        const rows = int(get(fx, "max_rows"));
        const capacity = int(get(fx, "capacity"));
        r.w = try views.rank(gpa, &r.m);
        errdefer r.w.deinit(gpa);
        r.b = try bufs.Buffers.init(&r.m, rows, capacity, false);
        r.mb = try bufs.Buffers.init(&r.m, rows, capacity, false);
        r.st = try bufs.State.init(gpa, &r.m, capacity, rows, false);
        errdefer r.st.deinit(gpa);
        r.smp = .{ .s = sample.Scratch.init(&r.m, rows) };
        r.space = .{ .gpa = gpa };
        errdefer r.space.deinit();
        try views.walk(&r.w, views.Path.init("w"), &r.space, trace.placeAll);
        try views.walk(&r.b, views.Path.init("b"), &r.space, trace.placeAll);
        try views.walk(&r.mb, views.Path.init("mbuf"), &r.space, trace.placeAll);
        try views.walk(&r.st, views.Path.init("st"), &r.space, trace.placeAll);
        try views.walk(&r.smp.s, views.Path.init("tmp"), &r.space, trace.placeAll);
        return r;
    }

    fn deinit(r: *Rank, gpa: std.mem.Allocator) void {
        r.w.deinit(gpa);
        r.st.deinit(gpa);
        r.space.deinit();
        gpa.destroy(r);
    }
};

fn laneA(path: []const u8) bool {
    for (lane_a) |p| if (std.mem.startsWith(u8, path, p)) return true;
    return false;
}

/// python's buffers are ours with dtype, shape and bytes; lane A's views (the MTP layer's DSA and MoE) are left to lane
/// A's gates on both sides (its names follow its binder: absorb.kw, EXL3's pointer tables); ours beyond python's are
/// otherwise only the sampler's scratch (`tmp.`, python's temporaries). Gives how many of python's are lane A's.
pub fn expectBuffers(space: *const trace.Space, want: J) !usize {
    var it = want.object.iterator();
    var skipped: usize = 0;
    while (it.next()) |kv| {
        errdefer std.debug.print("buffer {s}\n", .{kv.key_ptr.*});
        if (laneA(kv.key_ptr.*)) {
            skipped += 1;
            continue;
        }
        const e = for (space.entries.items) |e| {
            if (std.mem.eql(u8, e.path, kv.key_ptr.*)) break e;
        } else return error.MissingBuffer;
        try testing.expectEqualStrings(get(kv.value_ptr.*, "dtype").string, @tagName(e.buf.dtype));
        try testing.expect(get(kv.value_ptr.*, "owner").bool);
        const shape = get(kv.value_ptr.*, "shape").array.items;
        try testing.expectEqual(shape.len, e.buf.dims);
        for (shape, 0..) |s, i| try testing.expectEqual(int(s), e.buf.shape[i]);
        try testing.expectEqual(int(get(kv.value_ptr.*, "nbytes")), e.buf.bytes());
    }
    for (space.entries.items) |e| if (want.object.get(e.path) == null and !std.mem.startsWith(u8, e.path, "tmp.") and !laneA(e.path)) {
        std.debug.print("buffer {s}: not python's\n", .{e.path});
        return error.ExtraBuffer;
    };
    return skipped;
}

/// The MTP layer's blocks as the fixture's stubs: a mark each, the gathered partials at b.gath.
pub const Marks = struct {
    pub fn dsaBlock(f: *fwd.Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: layer.Opt) !u64 {
        _ = .{ l, st, R, opt };
        try f.rec.?.call(.ext, "mark.dsa_block", &.{});
        return b.gath.ptr;
    }
    pub fn moeBlock(f: *fwd.Fwd, l: *const views.Layer, b: *const bufs.Buffers, R: usize, opt: layer.Opt) !u64 {
        _ = .{ l, R, opt };
        try f.rec.?.call(.ext, "mark.moe_block", &.{});
        return b.gath.ptr;
    }
};

fn tokensOf(c: J, buf: []u32) []u32 {
    const xs = get(c, "tokens").array.items;
    for (xs, 0..) |x, i| buf[i] = @intCast(x.integer);
    return buf[0..xs.len];
}

/// One step fixture: its Config, buffers, and every case's launches; gives (steps, MTP cases, lane A's buffers).
pub fn expectSteps(text: []const u8) ![3]usize {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, text, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-forward-trace/1", get(fx, "schema").string);
    const r = try Rank.init(gpa, fx);
    defer r.deinit(gpa);
    try testing.expectEqual(int(get(fx, "vocab_offset")), r.m.rank * r.m.vl());
    const skipped = try expectBuffers(&r.space, get(fx, "buffers"));
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var count = [3]usize{ 0, 0, skipped };
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        const c = kv.value_ptr.*;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        t.reset();
        var f = fwd.Fwd.recording(&r.m, &t);
        var buf: [8]u32 = undefined;
        const tokens = tokensOf(c, &buf);
        if (std.mem.eql(u8, get(c, "kind").string, "step")) {
            r.st.pos = int(get(c, "pos"));
            r.st.parity = @intCast(int(get(c, "parity")));
            const R = try cc.stage(&f, &r.st, &r.b, tokens);
            try testing.expectEqual(int(get(c, "nch")), cc.chunksFor(&r.st, R));
            const logits = (try cc.compute(&f, &r.w, &r.st, &r.b, R, .{ .nch = cc.chunksFor(&r.st, R), .host_pos = r.st.pos }, true)).?;
            var out: [8]u32 = undefined;
            try sample.greedy(&f, &r.smp, logits, out[0..R]);
            const pos = r.st.pos;
            try cc.commit(&f, &r.st, &r.b, R, int(get(c, "keep")));
            try testing.expectEqual(pos + int(get(c, "keep")), r.st.pos);
            count[0] += 1;
        } else {
            r.st.mtp_len = int(get(c, "mtp_len"));
            const n = try mtp.stage(&f, &r.st, &r.mb, tokens, layer.rowsOf(r.b.fnormed, tokens.len));
            try testing.expectEqual(r.st.mtp_len == 0, r.mb.zero_first);
            const nch = (r.st.mtp_len + n + cc.chunk - 1) / cc.chunk;
            try testing.expectEqual(int(get(c, "nch")), nch);
            _ = try mtp.computeWith(Marks, &f, &r.w, &r.st, &r.mb, n, .{ .nch = nch, .host_pos = r.st.mtp_len }, true);
            count[1] += 1;
        }
        try f.flush();
        try lt.expectEvents(&r.space, get(c, "events").array.items, t.events.items);
    }
    return count;
}

test "MLX 4-bit: a TP2 rank's decode steps (stage, compute, greedy sample, commit) and MTP head are python-0.6's launches" {
    const n = try expectSteps(fixture_mlx);
    try testing.expectEqual(@as(usize, 6), n[0]);
    try testing.expectEqual(@as(usize, 2), n[1]);
    try testing.expect(n[2] > 0); // the MTP layer's DSA and MoE views: lane A's
}

test "EXL3 (BF16 dense, 4-bit draft head): the same steps and MTP head are python-0.6's launches" {
    const n = try expectSteps(fixture_exl3);
    try testing.expectEqual(@as(usize, 6), n[0]);
    try testing.expectEqual(@as(usize, 2), n[1]);
}

test "the MTP layer reaches lane A's DSA block (NotPorted until A3); one rank's sampler gathers nothing" {
    const gpa = testing.allocator;
    var m = try md.Model.from(&@import("config.zig").Config{}, .{}, 2, 0);
    m.layers = 3;
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    try testing.expect(w.mtp.?.layer.attn_hc == null and w.mtp.?.layer.kind == .dsa and w.mtp.?.layer.index == 3);
    var st = try bufs.State.init(gpa, &m, 2560, 8, false);
    defer st.deinit(gpa);
    var b = try bufs.Buffers.init(&m, 8, 2560, false);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var f = fwd.Fwd.recording(&m, &t);
    _ = try mtp.stage(&f, &st, &b, &.{ 5, 6 }, layer.rowsOf(b.fnormed, 2));
    if (mtp.compute(&f, &w, &st, &b, 2, .{}, true)) |_| {} else |e| try testing.expectEqualStrings("NotPorted", @errorName(e));
    try testing.expectError(error.Keep, cc.commit(&f, &st, &b, 2, 3));
    const nine: [9]u32 = @splat(1);
    try testing.expectError(error.Rows, cc.stage(&f, &st, &b, &nine));
    var one = try md.Model.from(&@import("config.zig").Config{}, .{}, 1, 0);
    one.layers = 3;
    var s1: sample.Sampler = .{ .s = sample.Scratch.init(&one, 4) };
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try views.walk(&s1.s, views.Path.init("tmp"), &space, trace.placeAll);
    t.reset();
    var f1 = fwd.Fwd.recording(&one, &t);
    var got: [2]u32 = .{ 7, 9 };
    try sample.greedy(&f1, &s1, trace.T.of(views.Buf.of(.bf16, &.{ 4, one.vl() }), 0, &.{ 2, one.vl() }), &got);
    try testing.expectEqual([2]u32{ 0, 0 }, got); // recorded: token 0, not what the caller's memory held
    try f1.flush();
    var names: [8][]const u8 = undefined;
    for (t.events.items, 0..) |e, i| names[i] = e.name;
    try testing.expectEqualDeep(@as([]const []const u8, &.{ "torch.float", "torch.topk", "torch.add", "torch.to", "torch.cpu", "torch.cpu" }), names[0..t.events.items.len]);
}

test "the greedy host choice is np.lexsort's on python's drawn rows (ties, signed zeros, infinities, NaNs)" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture_choose, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("glm-sample-choose/1", get(parsed.value, "schema").string);
    var cases: usize = 0;
    var ties: usize = 0;
    for (get(parsed.value, "cases").array.items) |c| {
        const vals = get(c, "values").array.items;
        const toks = get(c, "tokens").array.items;
        for (get(c, "chosen").array.items, vals, toks) |want, vr, tr| {
            var v: [16]f32 = undefined;
            var k: [16]i64 = undefined;
            for (vr.array.items, tr.array.items, 0..) |x, y, i| {
                v[i] = @bitCast(@as(u32, @intCast(x.integer)));
                k[i] = y.integer;
            }
            try testing.expectEqual(@as(u32, @intCast(want.integer)), sample.choose(v[0..vr.array.items.len], k[0..vr.array.items.len]));
        }
        // qualification_2b1188e0's long teacher step 25: 4663 and 83 tied at 7.65625, python's serial reply took 83 (the lower id)
        if (c.object.get("name")) |n| if (std.mem.startsWith(u8, n.string, "sample_tie_pair")) {
            try testing.expectEqual(@as(i64, 83), get(c, "chosen").array.items[0].integer);
            ties += 1;
        };
        cases += 1;
    }
    try testing.expectEqual(@as(usize, 66), cases);
    try testing.expectEqual(@as(usize, 2), ties);
}

fn bits(v: J) f32 {
    return @bitCast(@as(u32, @intCast(v.integer)));
}

test "topkOrder lists the top 5 python recorded on CUDA (qualification_6607f886, qualification_2b1188e0: ties and tied top-1s), topk 1 + choose its token" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture_topk, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("glm-topk-order/1", get(parsed.value, "schema").string);
    const row = try gpa.alloc(f32, int(get(parsed.value, "columns")));
    defer gpa.free(row);
    var n: [5]usize = @splat(0); // rows, rows with a tie, tied top-1 rows, tied pairs listed higher id first, qualification_2b1188e0's s25
    for (get(parsed.value, "cases").array.items) |c| {
        var vals: [2]f32 = undefined;
        var toks: [2]i64 = undefined;
        for (get(c, "ranks").array.items, 0..) |r, i| {
            @memset(row, -std.math.inf(f32)); // the columns below the fifth value cannot change topk's answer
            for (get(r, "entries").array.items) |e| row[int(e.array.items[0])] = bits(e.array.items[1]);
            const off = int(get(r, "vocab_offset"));
            const want = get(r, "top5").array.items;
            var got: [5]usize = undefined;
            sample.topkOrder(row, 5, &got);
            for (want, got) |w, g| {
                try testing.expectEqual(int(w.array.items[0]), g + off);
                try testing.expectEqual(bits(w.array.items[1]), row[g]);
            }
            var tied = false;
            for (want[0..4], want[1..]) |a, b| if (a.array.items[1].integer == b.array.items[1].integer) {
                tied = true;
                n[3] += @intFromBool(a.array.items[0].integer > b.array.items[0].integer);
            };
            n[1] += @intFromBool(tied);
            n[2] += @intFromBool(want[0].array.items[1].integer == want[1].array.items[1].integer);
            var one: [1]usize = undefined;
            sample.topkOrder(row, 1, &one); // sample_rows' torch.topk(logits, 1) on each rank, then the host's lexsort
            vals[i] = row[one[0]];
            toks[i] = @intCast(one[0] + off);
            n[0] += 1;
        }
        try testing.expectEqual(@as(u32, @intCast(get(c, "next_serial").integer)), sample.choose(&vals, &toks));
        if (std.mem.eql(u8, get(c, "name").string, "qualification_62397b5b")) { // a tied top-1: topk 5 lists 4663 first, greedy 83
            const t = get(get(c, "ranks").array.items[0], "top5").array.items;
            try testing.expectEqual([3]i64{ 4663, 83, 83 }, [3]i64{ t[0].array.items[0].integer, t[1].array.items[0].integer, toks[0] });
            n[4] += 1;
        }
    }
    try testing.expectEqualSlices(usize, &.{ 384, 90, 5, 89, 1 }, &n);
}
