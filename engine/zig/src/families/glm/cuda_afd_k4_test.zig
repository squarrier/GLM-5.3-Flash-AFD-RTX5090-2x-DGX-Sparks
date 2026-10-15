//! qualification_e4ad04cd: K4 on the halves (cuda_afd_k4.zig) on the CPU: python 2.15's window rule, its prompt buffers as a block of
//! their own, prompt_kernel_ready on python-0.6's split fixture (078455ff's layer 3), and the compute's routed launches
//! under K4 equal to cuda_exl3_k4.routed's on the landing's views and the window's scratch.

const std = @import("std");
const cuda = @import("cuda");
const ex3 = @import("cuda_exl3_experts.zig");
const k3 = @import("cuda_exl3_kernels.zig");
const k4 = @import("cuda_exl3_k4.zig");
const ae = @import("cuda_afd_expert.zig");
const aet = @import("cuda_afd_expert_test.zig");
const afdk4 = @import("cuda_afd_k4.zig");
const proto = cuda.xchg.proto;
const testing = std.testing;

const glm: proto.Layout = .{ .d = 4096, .s = 9 };

test "python 2.15's node.prompt: every forward mode but eager (chunk, slice, pair, lanes, and any other)" {
    try testing.expect(!afdk4.prompt(.eager));
    for ([_]proto.Mode{ .chunk, .slice, .pair, .lanes, @enumFromInt(11), @enumFromInt(0) }) |m| try testing.expect(afdk4.prompt(m));
}

test "K4's block: python's _prompt_buffers for each scratch (items [3 cap], counts [2], order [cap] at pass 64), placed apart" {
    var b = try ae.Bufs.init(4096, 9, 1024, 256, 64, 2048);
    const x = afdk4.Bufs.init(&b, 256);
    try testing.expectEqual(@as(usize, 3 * k4.promptItems(64, 9, 256, 64)), x.decode_items.shape[0]);
    try testing.expectEqual(@as(usize, 266), x.decode_order.shape[0]); // 257 + 576 / 64
    try testing.expectEqual(@as(usize, 545), x.pref_order.shape[0]); // 257 + 18,432 / 64
    try testing.expectEqual(@as(usize, 2), x.pref_counts.shape[0]);
    var placed = x;
    const n = try afdk4.place(&placed, 0);
    try testing.expect(n >= 4 * (3 * 266 + 2 + 266 + 3 * 545 + 2 + 545) and n < 4 * (3 * 266 + 2 + 266 + 3 * 545 + 2 + 545) + 6 * 256);
    _ = try afdk4.place(&placed, 1 << 40);
    var seen: [6]u64 = undefined;
    for ([_]views_Buf{ placed.decode_items, placed.decode_counts, placed.decode_order, placed.pref_items, placed.pref_counts, placed.pref_order }, 0..) |v, i| {
        try testing.expect(v.ptr >= 1 << 40 and v.ptr + v.bytes() <= (1 << 40) + n);
        seen[i] = v.ptr;
    }
    for (seen[0..5], seen[1..]) |a, c| try testing.expect(a < c);
    const d = placed.of(&b, &b.decode);
    try testing.expectEqual(k4.Bufs{ .items = placed.decode_items.ptr, .counts = placed.decode_counts.ptr, .order = placed.decode_order.ptr, .cap = 266 }, d);
    const p = placed.of(&b, b.scratch(2048));
    try testing.expectEqual(k4.Bufs{ .items = placed.pref_items.ptr, .counts = placed.pref_counts.ptr, .order = placed.pref_order.ptr, .cap = 545 }, p);
    // the scratch K4 takes at R rows holds python's cap at either pass
    for ([_]u32{ 1, 7, 64, 65, 594, 2048 }) |R| for (k4.passes) |pass| {
        const s = b.scratch(R);
        try testing.expect(k4.promptItems(R, 9, 256, pass) <= placed.of(&b, s).cap);
    };
}

const views_Buf = @import("cuda_views.zig").Buf;
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const lt = @import("cuda_layer_test.zig");
const J = std.json.Value;
const fixture_k4 = @embedFile("fixtures_cuda_afd_k4.json");

fn get(o: J, key: []const u8) J {
    return o.object.get(key) orelse std.debug.panic("fixture: no key {s}", .{key});
}

fn envVal(env: J, key: []const u8) ?[]const u8 {
    return if (env.object.get(key)) |v| v.string else null;
}

/// cuda_afd_expert_test's `bound`: the layer as the launches take it from the record's views.
fn bound(e: views.Exl3Experts) ex3.Layer {
    return .{ .gate_ptr = e.gate_ptr.ptr, .up_ptr = e.up_ptr.ptr, .down_ptr = e.down_ptr.ptr, .gate_k2 = e.gate_k2.ptr, .up_k2 = e.up_k2.ptr, .down_k2 = e.down_k2.ptr, .suh_g = e.suh_g.ptr, .suh_u = e.suh_u.ptr, .svh_g = e.svh_g.ptr, .svh_u = e.svh_u.ptr, .suh_d = e.suh_d.ptr, .svh_d = e.svh_d.ptr, .count = @intCast(e.count), .dims = @intCast(e.dims), .width = @intCast(e.width), .cb = @enumFromInt(e.cb), .k2_gu = e.k2_gu, .k2_d = e.k2_d };
}

/// The routed launches of K4 against the recorded prompt call (zig's record, python's by the G-trace check): group,
/// the rotation, the plan, her order, gate/up and down carry exl3_experts.prompt's buffers and values where
/// exl3x_prompt_cuda and prompt_launch pass them.
fn expectLaunches(ev: []const trace.Event, ls: []const k3.Launch, R: usize) !void {
    const g = ev[0].vals; // group(pick, ids, count, members, R, slots, E)
    const r = ev[1].vals;
    const p = ev[2].vals; // prompt's 25
    const ptr = struct {
        fn of(v: trace.Val) u64 {
            return v.t.ptr;
        }
        fn num(v: trace.Val) u64 {
            return @intCast(v.int);
        }
    };
    for (0..4) |i| try testing.expectEqual(@as(?u64, ptr.of(g[i])), ls[0].args.integer(i));
    for (4..7) |i| try testing.expectEqual(@as(?u64, ptr.num(g[i])), ls[0].args.integer(i));
    try testing.expectEqual(@as(?u64, R), ls[0].args.integer(7)); // maxm: the members' row stride
    if (std.mem.eql(u8, ev[1].name, "exl3_experts.rot_rows")) { // (x, stride, shared_suh, xg, R, D)
        try testing.expectEqualStrings(k3.sym.x1_rot_rows_bf16, ls[1].symbol);
        for ([_][2]usize{ .{ 0, 0 }, .{ 2, 3 }, .{ 3, 5 } }) |m| try testing.expectEqual(@as(?u64, ptr.of(r[m[0]])), ls[1].args.integer(m[1]));
        try testing.expectEqual(@as(?u64, ptr.num(r[1])), ls[1].args.integer(1));
        try testing.expectEqual(@as(?u64, ptr.num(r[5])), ls[1].args.integer(7));
        try testing.expectEqual(@as(u32, @intCast(R)), ls[1].cfg.grid.x);
    } else { // rot_in(x, stride, pick, suh_g, suh_u, xg, xu, R, D, slots, E)
        try testing.expectEqualStrings(k3.sym.rot_in_bf16, ls[1].symbol);
        for ([_]usize{ 0, 2, 3, 4, 5, 6 }) |i| try testing.expectEqual(@as(?u64, ptr.of(r[i])), ls[1].args.integer(i));
        for ([_][2]usize{ .{ 1, 1 }, .{ 8, 7 }, .{ 9, 8 }, .{ 10, 9 } }) |m| try testing.expectEqual(@as(?u64, ptr.num(r[m[0]])), ls[1].args.integer(m[1]));
    }
    const pass = ptr.num(p[22]);
    const cap = ptr.num(p[20]);
    const plan = ls[2]; // prompt_plan_kernel(uids, ucount, members, members.size(1), pass, max_items, E, items, counts)
    try testing.expectEqualStrings(k3.sym.k4_plan, plan.symbol);
    const want_plan = [9]u64{ ptr.of(p[5]), ptr.of(p[6]), ptr.of(p[7]), p[7].t.shape[1], pass, cap, ptr.num(p[19]), ptr.of(p[8]), ptr.of(p[9]) };
    for (want_plan, 0..) |w, i| try testing.expectEqual(@as(?u64, w), plan.args.integer(i));
    var i: usize = 3;
    const ord: u64 = if (p[10] == .none) 0 else ptr.of(p[10]);
    if (ord != 0) { // prompt_order_kernel(items, counts, order, max_items, E)
        try testing.expectEqualStrings(k3.sym.k4_order, ls[3].symbol);
        for ([5]u64{ ptr.of(p[8]), ptr.of(p[9]), ord, cap, ptr.num(p[19]) }, 0..) |w, j| try testing.expectEqual(@as(?u64, w), ls[3].args.integer(j));
        i = 4;
    }
    try testing.expectEqual(i + 2, ls.len);
    const gu = ls[i]; // prompt_gateup_kernel<PM, NW, SHX>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, D, N, E, limit, slots, order)
    try testing.expectEqualStrings(k3.sym.k4Gateup(@intCast(pass), p[24].bool), gu.symbol);
    const want_gu = [17]u64{ ptr.of(p[0]), ptr.of(p[1]), ptr.of(p[2]), ptr.of(p[3]), ptr.of(p[8]), ptr.of(p[9]), ptr.of(p[7]), ptr.of(p[11]), ptr.of(p[12]), ptr.of(p[13]), ptr.of(p[15]), ptr.num(p[17]), ptr.num(p[18]), ptr.num(p[19]), p[21].f32, ptr.num(p[23]), ord };
    for (want_gu, 0..) |w, j| try testing.expectEqual(@as(?u64, w), gu.args.integer(j));
    try testing.expectEqual([3]u32{ @intCast(ptr.num(p[18]) / 128), @intCast(cap), 1 }, [3]u32{ gu.cfg.grid.x, gu.cfg.grid.y, gu.cfg.grid.z });
    try testing.expectEqual(@as(u32, @intCast(pass / 8 * 32)), gu.cfg.block.x);
    try testing.expectEqual(k4.smem(@intCast(pass), if (p[24].bool) 1 else 2), gu.cfg.shared);
    const dn = ls[i + 1]; // prompt_down_kernel<PM, NW>(xd, Td, items, counts, members, svh_d, y, N, D, E, slots, order)
    try testing.expectEqualStrings(k3.sym.k4Down(@intCast(pass)), dn.symbol);
    const want_dn = [12]u64{ ptr.of(p[15]), ptr.of(p[4]), ptr.of(p[8]), ptr.of(p[9]), ptr.of(p[7]), ptr.of(p[14]), ptr.of(p[16]), ptr.num(p[18]), ptr.num(p[17]), ptr.num(p[19]), ptr.num(p[23]), ord };
    for (want_dn, 0..) |w, j| try testing.expectEqual(@as(?u64, w), dn.args.integer(j));
    try testing.expectEqual([3]u32{ @intCast(ptr.num(p[17]) / 256), @intCast(cap), 1 }, [3]u32{ dn.cfg.grid.x, dn.cfg.grid.y, dn.cfg.grid.z });
    try testing.expectEqual(k4.smem(@intCast(pass), 2), dn.cfg.shared);
}

test "G-trace-E with K4 (qualification_e4ad04cd): python 2.15's prompt windows under TF_GLM_EXL3_PROMPT=1 (1-2,048 rows, both halves, layers 3 and 44, pass 128, order and rowrot off) equal call for call; K4's launches carry the prompt call's buffers and values" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture_k4, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-forward-trace/1", get(fx, "schema").string);
    try testing.expectEqualStrings("tools/zig/glm_afd_k4_fixtures.py", get(fx, "generator").string);
    var n = try aet.Node.init(gpa, fx);
    defer n.space.deinit();
    const c = get(fx, "config");
    const D: usize = @intCast(get(c, "hidden").integer);
    const E: usize = @intCast(get(c, "experts").integer);
    // K4's buffers under python 2.15's names (each scratch's prompt_bufs) and each layer's shared suh (its clone, [D])
    var kb = afdk4.Bufs.init(&n.b, E);
    const names = [6][]const u8{ "b.decode.prompt_bufs.0", "b.decode.prompt_bufs.1", "b.decode.prompt_bufs.2", "b.pref.prompt_bufs.0", "b.pref.prompt_bufs.1", "b.pref.prompt_bufs.2" };
    for (names, [6]*views.Buf{ &kb.decode_items, &kb.decode_counts, &kb.decode_order, &kb.pref_items, &kb.pref_counts, &kb.pref_order }) |name, b| try n.space.place(name, b);
    var suh: [2]views.Buf = undefined;
    for (&suh, n.layers) |*s, l| {
        s.* = views.Buf.of(.f16, &.{D});
        var name: [64]u8 = undefined;
        try n.space.place(try std.fmt.bufPrint(&name, "w.layers.{d}.moe.experts.shared_suh", .{l}), s);
    }
    // K4's six buffers are python's (dtype, shape, bytes)
    for (names) |name| {
        const want = get(get(fx, "buffers"), name);
        const e = for (n.space.entries.items) |e| {
            if (std.mem.eql(u8, e.path, name)) break e;
        } else return error.MissingBuffer;
        try testing.expectEqualStrings(get(want, "dtype").string, @tagName(e.buf.dtype));
        const shape = get(want, "shape").array.items;
        try testing.expectEqual(shape.len, e.buf.dims);
        for (shape, 0..) |s, i| try testing.expectEqual(s.integer, @as(i64, @intCast(e.buf.shape[i])));
        try testing.expectEqual(get(want, "nbytes").integer, @as(i64, @intCast(e.buf.bytes())));
    }
    const limit: f32 = @bitCast(@as(u32, @intCast(get(c, "limit").integer)));
    const sh: ex3.Shared = .{ .optin = 101376 };
    var cases: usize = 0;
    var shx: usize = 0;
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        const cs = kv.value_ptr.*;
        const R: usize = @intCast(get(cs, "rows").integer);
        const li: u32 = @intCast(get(cs, "layer").integer);
        const k: usize = if (n.layers[0] == li) 0 else 1;
        const env = get(cs, "env");
        try testing.expectEqualStrings("1", envVal(env, "TF_GLM_EXL3_PROMPT").?);
        const o = try k4.Opts.parse(envVal(env, "TF_GLM_EXL3_PASS"), envVal(env, "TF_GLM_EXL3_ORDER"), envVal(env, "TF_GLM_EXL3_ROWROT"));
        const e = n.experts(li);
        // python's prompt_kernel_ready on the layer, as the half's `ready` tests it (the trellises aligned there)
        var key: [16]u8 = undefined;
        try testing.expectEqual(get(get(fx, "ready"), try std.fmt.bufPrint(&key, "h{d}-l{d}", .{ get(cs, "half").integer, li })).bool, k4.ready(bound(e.*), @intCast(n.b.decode.slots), true));
        var t = trace.Trace.init(gpa);
        defer t.deinit();
        const v = ae.landed(glm, n.landing.buf.ptr, R);
        try afdk4.record(&t, e, &n.b, v, R, limit, o, .{ .bufs = &kb, .shared_suh = suh[k] });
        const want = get(cs, "events").array.items;
        try lt.expectEvents(&n.space, want, t.events.items);
        try testing.expectEqual(want.len, t.events.items.len);
        var ls: k3.Launches = .{};
        const s = n.b.scratch(R);
        try k4.routed(&ls, v.normed, @intCast(D), true, v.pick, bound(e.*), s.bound(@intCast(E)), @intCast(R), limit, true, sh, o, kb.of(&n.b, s), if (o.xrow) suh[k].ptr else null);
        try expectLaunches(t.events.items, ls.slice(), R);
        shx += @intFromBool(t.events.items[2].vals[24].bool);
        cases += 1;
    }
    try testing.expectEqual(@as(usize, 44), cases); // 2 halves x 2 layers x (5 rows + 3 switches x 2 rows)
    try testing.expectEqual(@as(usize, 36), shx); // rowrot off: rot_in and the gate/up kernel without SHX
}

test "prompt_kernel_ready on python-0.6's split fixture (078455ff layer 3, experts 0-3): 4-bit mcg at 4,096 x 1,024, its trellises 16-byte aligned" {
    var f = try aet.Split.load();
    defer f.arena.deinit();
    for ([_]u32{ 0, 1 }) |h| {
        var x = try ae.half(testing.allocator, &f.c, &f.names, h);
        defer x.deinit();
        const l = x.bind(0, 1 << 40);
        try testing.expectEqual(ex3.Codebook.mcg, l.cb);
        try testing.expectEqual([2]u32{ 8, 8 }, l.k2_gu);
        try testing.expectEqual([2]u32{ 8, 8 }, l.k2_d);
        try testing.expect(afdk4.aligned(x.experts(0), 1 << 40));
        try testing.expect(afdk4.ready(&x, 0, 1 << 40, 9));
        try testing.expect(!afdk4.aligned(x.experts(0), (1 << 40) + 8)); // a group off 16 bytes: python's nc_aligned false
        try testing.expect(!afdk4.ready(&x, 0, (1 << 40) + 8, 9));
        try testing.expect(!afdk4.ready(&x, 0, 1 << 40, 33)); // group()'s member codes hold 32 slots
    }
}

/// An On over `x1b`'s shared suh rows for two positions (the first sharing one suh), every position ready but `not`.
fn on(opts: k4.Opts, kb: *const afdk4.Bufs, x1b: *const ae.X1Bufs, ready: []const bool) afdk4.On {
    return .{ .opts = opts, .bufs = kb, .ready = ready, .x1b = x1b, .xrow = &.{ true, false } };
}

test "the window's routed launches under K4 are cuda_exl3_k4.routed's on the landing's views and the window's scratch; takes and suh as python's" {
    var b = try ae.Bufs.init(4096, 9, 1024, 256, 64, 2048);
    var landing = ae.Landing.init(glm, 2048);
    _ = try ae.place(&b, &landing, 1 << 32);
    var kb = afdk4.Bufs.init(&b, 256);
    _ = try afdk4.place(&kb, 1 << 36);
    var x1b = ae.X1Bufs.init(&b, 256, 2);
    _ = try ae.placeX1(&x1b, 1 << 38);
    const l: ex3.Layer = .{ .gate_ptr = 0x1000, .up_ptr = 0x2000, .down_ptr = 0x3000, .gate_k2 = 0x4000, .up_k2 = 0x5000, .down_k2 = 0x6000, .suh_g = 0x7000, .suh_u = 0x8000, .svh_g = 0x9000, .svh_u = 0xa000, .suh_d = 0xb000, .svh_d = 0xc000, .count = 256, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
    const sh: ex3.Shared = .{ .optin = 101376 };
    const ready = [_]bool{ true, false };
    for ([_]k4.Opts{ .{}, .{ .pass = 128 }, .{ .order = false }, .{ .xrow = false } }) |o| {
        const q = on(o, &kb, &x1b, &ready);
        try testing.expect(q.takes(0, .chunk) and q.takes(0, .pair) and q.takes(0, .slice) and q.takes(0, .lanes));
        try testing.expect(!q.takes(0, .eager) and !q.takes(1, .chunk));
        try testing.expectEqual(if (o.xrow) @as(?u64, x1b.suh(0)) else null, q.suh(0));
        try testing.expectEqual(@as(?u64, null), q.suh(1)); // position 1's gate and up suh are not one vector: rot_in
        for ([_]usize{ 1, 7, 64, 594, 2048 }) |R| for ([_]usize{ 0, 1 }) |pos| {
            const v = ae.landed(glm, landing.buf.ptr, R);
            const s = b.scratch(R);
            var got: k3.Launches = .{};
            try afdk4.launches(&got, v, l, s, R, 7.0, sh, &q, &b, pos);
            var want: k3.Launches = .{};
            try k4.routed(&want, v.normed, 4096, true, v.pick, l, s.bound(256), @intCast(R), 7.0, true, sh, o, kb.of(&b, s), if (o.xrow and pos == 0) x1b.suh(0) else null);
            try testing.expectEqual(want.slice().len, got.slice().len);
            for (want.slice(), got.slice()) |a, c| {
                try testing.expectEqualStrings(a.symbol, c.symbol);
                try testing.expectEqual(a.cfg, c.cfg);
                try testing.expectEqual(a.max_dynamic, c.max_dynamic);
                try testing.expectEqualSlices(u8, a.args.storage[0..a.args.used], c.args.storage[0..c.args.used]);
            }
            // the plan's buffers are the window's scratch's: decode up to 64 rows, prompt past them
            const plan = got.slice()[2];
            try testing.expectEqual(@as(?u64, if (R <= 64) kb.decode_items.ptr else kb.pref_items.ptr), plan.args.integer(7));
            try testing.expectEqual(@as(?u64, s.ids.ptr), plan.args.integer(0));
        };
        var none: k3.Launches = .{};
        try testing.expectError(error.Rows, afdk4.launches(&none, ae.landed(glm, landing.buf.ptr, 1), l, &b.pref, 2049, 7.0, sh, &q, &b, 0));
        try testing.expectError(error.Rows, afdk4.launches(&none, ae.landed(glm, landing.buf.ptr, 1), l, &b.decode, 0, 7.0, sh, &q, &b, 0));
    }
}
