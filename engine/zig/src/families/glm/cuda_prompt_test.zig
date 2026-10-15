//! qualification_qualification_44ba554e.5 B3's launch-trace check of prompt chunks (qualification_qualification_7c75d777 PLAN §4): python-0.6's decode.prefill of a 64-row prompt and of a
//! 2,085-row one (a 2,048-row chunk, then a 37-row tail; MTP absorb, commit, the greedy first token), and one chunk cut
//! at an interior row (forward.Cut; 100 rows at 40, 2,048 at 1,000), recorded on a CPU by tools/zig/glm_step_fixtures.py
//! --prompt over a checkpoint's headers (MLX 4-bit; EXL3, BF16 dense) for a TP2 rank cut to layers 0-2 with the MTP head
//! (layer 3 is DSA, lane A's A3; the MTP layer's blocks are `mark.*` events). The Zig rank's Engine makes the launches.
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const cc = @import("cuda_compute.zig");
const prompt = @import("cuda_prompt.zig");
const eng = @import("cuda_engine.zig");
const layer = @import("cuda_layer.zig");
const bufs = @import("cuda_buffers.zig");
const mtp = @import("cuda_mtp.zig");
const Tmp = @import("cuda_tmp.zig").Tmp;
const lt = @import("cuda_layer_test.zig");
const ct = @import("cuda_compute_test.zig");

const testing = std.testing;
const J = std.json.Value;
const get = ct.get;
const int = ct.int;
const E = eng.EngineOf(ct.Marks);
const fixture_mlx = @embedFile("fixtures_cuda_prompt.json");
const fixture_exl3 = @embedFile("fixtures_cuda_prompt_exl3.json");
const fixture_r1 = @embedFile("fixtures_cuda_step_r1.json");

/// The rank of a prompt fixture: its model (cut, with the MTP head kept), weights, python's decode.Engine (buf, pbuf,
/// mbuf, st, the sampler) and their stand-in space.
const Rank = struct {
    m: md.Model,
    w: views.Weights,
    e: *E,
    space: trace.Space,

    fn init(gpa: std.mem.Allocator, fx: J) !*Rank {
        const r = try gpa.create(Rank);
        errdefer gpa.destroy(r);
        r.m = try md.Model.parse(gpa, get(fx, "config_json").string, int(get(fx, "world")), int(get(fx, "rank")));
        try lt.expectConfig(&r.m, get(fx, "config"));
        const mtp_layers = r.m.mtp_layers;
        try r.m.cut(int(get(fx, "layers")));
        if (get(fx, "mtp").bool) r.m.mtp_layers = mtp_layers;
        r.w = try views.rank(gpa, &r.m);
        errdefer r.w.deinit(gpa);
        r.e = try E.create(gpa, &r.m, &r.w, int(get(fx, "capacity")), int(get(fx, "max_rows")), int(get(fx, "prefill_rows")), get(fx, "long_context").bool);
        errdefer r.e.destroy(gpa);
        r.space = .{ .gpa = gpa };
        errdefer r.space.deinit();
        try views.walk(&r.w, views.Path.init("w"), &r.space, trace.placeAll);
        try r.e.walk(&r.space, trace.placeAll);
        try r.e.walkTmp(&r.space, trace.placeAll);
        r.e.tmp.space = &r.space;
        return r;
    }

    fn deinit(r: *Rank, gpa: std.mem.Allocator) void {
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.space.deinit();
        gpa.destroy(r);
    }
};

/// One chunk at `pos` cut at `point`, as glm_step_fixtures.chunk_case runs it: the Cut (python's empty_like state and
/// the conv windows' clone), stage, compute with the cut, commit.
fn cutChunk(f: *fwd.Fwd, e: *E, tokens: []const u32, point: usize) !usize {
    const st = &e.st;
    var cut: prompt.Cut = .{ .point = point, .rec = views.Buf.of(.f32, st.rec.shape[1..5]), .conv = views.Buf.of(.bf16, st.conv.shape[0..3]) };
    try e.tmp.made("empty_like", &cut.rec);
    _ = try cc.clone(f, &e.tmp, trace.T.whole(st.conv), &cut.conv);
    e.prompt.cut = &cut;
    defer e.prompt.cut = null;
    const R = try cc.stage(f, st, &e.pbuf, tokens);
    const nch = cc.chunksFor(st, R);
    _ = try cc.compute(f, e.w, st, &e.pbuf, R, .{ .nch = nch, .host_pos = st.pos, .prompt = &e.prompt }, true);
    try cc.commit(f, st, &e.pbuf, R, R);
    return nch;
}

/// One prompt fixture: python's Config, buffers, and every case's launches; gives (prefills, cut chunks).
fn expectPrompts(text: []const u8, taps: ?*Names) ![2]usize {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, text, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-forward-trace/1", get(fx, "schema").string);
    const r = try Rank.init(gpa, fx);
    defer r.deinit(gpa);
    _ = try ct.expectBuffers(&r.space, get(fx, "buffers"));
    if (taps) |n| r.e.tap = .{ .ctx = n, .tensor = Names.tensor };
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var count = [2]usize{ 0, 0 };
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        const c = kv.value_ptr.*;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        t.reset();
        r.e.tmp.reset(); // python names each case's temporaries afresh ...
        r.e.prompt.reset(); // ... and starts it without kda's wide scratch
        var f = fwd.Fwd.recording(&r.m, &t);
        const xs = get(c, "tokens").array.items;
        const tokens = try gpa.alloc(u32, xs.len);
        defer gpa.free(tokens);
        for (xs, tokens) |x, *y| y.* = @intCast(x.integer);
        if (std.mem.eql(u8, get(c, "kind").string, "prefill")) {
            _ = try r.e.prefill(&f, tokens, true);
            try testing.expectEqual(tokens.len, r.e.st.pos);
            try testing.expectEqual(tokens.len - 1, r.e.st.mtp_len);
            try testing.expectEqual(get(c, "chunks").array.items.len, (tokens.len + r.e.prefill_rows - 1) / r.e.prefill_rows);
            count[0] += 1;
        } else {
            r.e.st.pos = int(get(c, "pos")); // a fresh engine at `pos` (python's set_pos comes before the log)
            r.e.st.parity = 0;
            try testing.expectEqual(int(get(c, "nch")), try cutChunk(&f, r.e, tokens, int(get(c, "point"))));
            count[1] += 1;
        }
        try f.flush();
        try lt.expectEvents(&r.space, get(c, "events").array.items, t.events.items);
    }
    return count;
}

test "MLX 4-bit: a TP2 rank's prefill (64 rows; 2,048 + 37) and cut chunks are python-0.6's launches" {
    const n = try expectPrompts(fixture_mlx, null);
    try testing.expectEqual([2]usize{ 2, 2 }, n);
}

test "EXL3 (BF16 dense, 4-bit draft head): the same prefills and cut chunks are python-0.6's launches" {
    const n = try expectPrompts(fixture_exl3, null);
    try testing.expectEqual([2]usize{ 2, 2 }, n);
}

/// The recorder's taps in qualification_6607f886's order (each chunk: L00.kda, L00.mlp, L00.x, ..., hidden, fnormed) and shapes.
const Names = struct {
    n: usize = 0,
    const want = [_][]const u8{ "L00.kda", "L00.mlp", "L00.x", "L01.kda", "L01.mlp", "L01.x", "L02.kda", "L02.mlp", "L02.x", "hidden", "fnormed" };

    fn tensor(ctx: *anyopaque, name: []const u8, x: trace.T, pos0: usize, rows: usize, kind: ?[]const u8) anyerror!void {
        const s: *Names = @ptrCast(@alignCast(ctx));
        const i = s.n % want.len;
        try testing.expectEqualStrings(want[i], name);
        const block = i < 9 and i % 3 != 2;
        try testing.expectEqual(@as(u8, if (block) 3 else 2), x.dims);
        if (block) try testing.expectEqual([3]usize{ 2, rows, 4096 }, x.shape[0..3].*) else try testing.expectEqual(rows, x.shape[0]);
        try testing.expectEqual(i < 9 and i % 3 == 2, kind != null);
        try testing.expect(pos0 == 0 or pos0 == 2048);
        s.n += 1;
    }
};

test "dump's taps (qualification_6607f886's recorder) keep the prefills' launches python's, in qualification_6607f886's order" {
    var names: Names = .{};
    const n = try expectPrompts(fixture_mlx, &names);
    try testing.expectEqual([2]usize{ 2, 2 }, n);
    try testing.expectEqual(@as(usize, 11 * 3), names.n); // prefill-64, then -2085's two chunks (cut chunks call compute)
}

test "rank 1 of 2: the decode steps' head and sampler on the rank's vocabulary rows (vocab offset 77,440)" {
    const n = try ct.expectSteps(fixture_r1);
    try testing.expectEqual(@as(usize, 6), n[0]);
    try testing.expectEqual(@as(usize, 2), n[1]);
}

test "a cut must lie inside a prefill chunk; a decode window has no prompt form" {
    const gpa = testing.allocator;
    var m = try md.Model.from(&@import("config.zig").Config{}, .{}, 2, 0);
    m.layers = 3;
    m.mtp_layers = 0;
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    const e = try E.create(gpa, &m, &w, 2560, 8, 64, false);
    defer e.destroy(gpa);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var f = fwd.Fwd.recording(&m, &t);
    var cut: prompt.Cut = .{ .point = 4, .rec = e.st.rec, .conv = e.st.conv };
    e.prompt.cut = &cut;
    try testing.expectError(error.Cut, cc.compute(&f, e.w, &e.st, &e.pbuf, 4, .{ .prompt = &e.prompt }, true));
    try testing.expectError(error.Cut, cc.compute(&f, e.w, &e.st, &e.buf, 8, .{ .prompt = &e.prompt }, true));
    e.prompt.cut = null;
    try testing.expectError(error.NoPrompt, cc.compute(&f, e.w, &e.st, &e.pbuf, 4, .{}, true));
    try testing.expectError(error.EmptyPrompt, e.prefill(&f, &.{}, true));
    try testing.expectError(error.NoMtp, e.mtpStep(&f, &.{1}, cc.rows(e.buf.fnormed, 0, 1)));
}

/// qualification_qualification_ea348a94's DSA prompt fixture (lane A's A4), where the tree has its DSA block (not A3's stub).
const fixture_dsa = if (prompt.DsaScratch == void) "" else @embedFile("fixtures_cuda_dsa.json");

test "B7: 2,048-row chunks past the dense limit take the engine's DSA Scratch on lane B's path: qualification_qualification_ea348a94's launches" {
    if (comptime prompt.DsaScratch == void) return error.SkipZigTest else {
        const gpa = testing.allocator;
        const parsed = try std.json.parseFromSlice(J, gpa, fixture_dsa, .{});
        defer parsed.deinit();
        const fx = parsed.value;
        var r = try lt.Rank.init(gpa, fx);
        defer r.deinit(gpa);
        var long = try lt.Extra.init(gpa, &r, &r.m, get(fx, "long_state"), "st.", true);
        defer long.deinit(gpa);
        var mm = r.m; // the MTP load keeps the MTP layer: its cache and the indexer's last
        mm.mtp_layers = @intCast(int(get(get(fx, "config"), "mtp_layers")));
        var xm = try lt.Extra.init(gpa, &r, &mm, get(fx, "mtp_state"), "st.", true);
        defer xm.deinit(gpa);
        var tmp: Tmp = .{};
        var t = trace.Trace.init(gpa);
        defer t.deinit();
        var binds: usize = 0; // python's torch.empty partials bound in the Scratch
        for ([_][]const u8{ "l3-prompt-long-r2048-p2048", "mtp-dsa-prompt-long-r2048-m2048" }) |name| {
            errdefer std.debug.print("case {s}\n", .{name});
            const c = get(get(fx, "cases"), name);
            const is_mtp = c.object.get("block") != null;
            const x = if (is_mtp) &xm else &long;
            const pb = if (x.pb) |*p| p else return error.NoPromptBuffers;
            var p = prompt.Prompt.init(&r.m, &tmp, pb.rows);
            p.dsa = prompt.dsaScratch(&r.m, pb, x.st.capacity, true) orelse return error.NoScratch;
            try testing.expectEqual(@as(usize, 673_710_080), p.dsa.?.bytes()); // 642.5 MiB: 32 heads, 2,048 rows
            try views.walk(&p.dsa, views.Path.init("dsa"), &x.space, trace.placeAll);
            x.st.pos = @intCast(int(get(c, "pos")));
            x.st.parity = @intCast(int(get(c, "parity")));
            if (is_mtp) x.st.mtp_len = x.st.pos;
            const rows: usize = @intCast(int(get(c, "rows")));
            const bare: layer.Opt = .{ .nch = @intCast(int(get(c, "nch"))), .host_pos = x.st.pos };
            var opt = bare;
            opt.prompt = &p;
            t.reset();
            var f = fwd.Fwd.recording(&r.m, &t);
            if (is_mtp and r.mtp == null) return error.NoMtpLayer;
            const l: *const views.Layer = if (is_mtp) &r.mtp.? else &r.w.layers[3];
            if (is_mtp) _ = try mtp.LaneA.dsaBlock(&f, l, &x.st, pb, rows, opt) else try prompt.layerForward(&f, l, &x.st, pb, rows, opt);
            try f.flush();
            var ck: lt.Check = .{ .space = &r.space, .extra = &x.space, .tmp = c.object.get("tmp") };
            try lt.expectCase(&ck, get(c, "events").array.items, t.events.items);
            for (ck.bound) |b| if (b) |y| {
                binds += @intFromBool(std.mem.startsWith(u8, y.path, "dsa."));
            };
            // without a prompt chunk's Scratch: lane A's own stop (the negative control)
            t.reset();
            f = fwd.Fwd.recording(&r.m, &t);
            if (is_mtp) try testing.expectError(error.NoDsaScratch, mtp.LaneA.dsaBlock(&f, l, &x.st, pb, rows, bare)) else try testing.expectError(error.NoDsaScratch, prompt.layerForward(&f, l, &x.st, pb, rows, bare));
            // none where no chunk needs one: at the dense limit, without a long context, or 512-row chunks
            try testing.expect(prompt.dsaScratch(&r.m, pb, r.m.denseLimit(), true) == null);
            try testing.expect(prompt.dsaScratch(&r.m, pb, x.st.capacity, false) == null);
            const small = try bufs.Buffers.init(&r.m, 512, x.st.capacity, true);
            try testing.expect(prompt.dsaScratch(&r.m, &small, x.st.capacity, true) == null);
        }
        try testing.expectEqual(@as(usize, 6), binds); // po, pm and pl in each case
    }
}
