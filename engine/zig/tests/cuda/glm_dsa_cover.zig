//! qualification_qualification_ea348a94's G-cover for the DSA block on qualification_6607f886's long prompt: the Zig DSA block's Triton launches, recorded on a CPU at
//! stand-in addresses, against GLM-5.3-Flash's kernel set (glm/kernels.json, tools/zig/glm_aot.py's spec, from which
//! `glm_aot.py build` makes the cubins) by aot.zig's own rule: each launch finds exactly one variant, as aot.Set.find
//! will at run time. Rank 0 of 2, MLX and EXL3 (fixtures_cuda_dsa*.json's configs, cut to 4 layers): layer 3 on A4's
//! prompt chunks (2,048 rows from 0 and 2,048, 904 from 4,096; 37 from 2,048; 1,537 from 0), its sparse decode rows
//! (the long prompt's teacher steps, 1 row from 5,000 and 5,047; a window of 8), and the MTP layer on the absorbed
//! chunks (2,048 from 0 and 2,048, 903 from 4,096; 36 from 2,048).
const std = @import("std");
const aot = @import("cuda").aot;
const glm = @import("glm_cuda");
const md = glm.model;
const views = glm.views;
const bufs = glm.buffers;
const trace = glm.trace;
const fwd = glm.fwd;
const dk = glm.dsa_kernels;

const testing = std.testing;
const spec_text = @embedFile("glm/kernels.json");
const fixtures = [_][]const u8{ @embedFile("dsa_fixture"), @embedFile("dsa_fixture_exl3") };

/// The spec's entries as aot.json rows (qualification_955b7191's afd_cover.zig's reading of kernels.json).
fn entries(arena: std.mem.Allocator) ![]aot.Spec {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, spec_text, .{});
    const kernels = root.object.get("kernels").?.array.items;
    const out = try arena.alloc(aot.Spec, kernels.len);
    for (kernels, out) |k, *e| {
        const o = k.object;
        const sig = o.get("signature").?.object;
        const attrs = o.get("attrs").?.object;
        var params: std.ArrayList(aot.SpecParam) = .empty;
        for (o.get("params").?.array.items) |pv| {
            const ty = sig.get(pv.string).?.string;
            if (std.mem.eql(u8, ty, "constexpr")) continue;
            const div = if (attrs.get(pv.string)) |a| a.array.items.len > 0 else false;
            try params.append(arena, .{ .name = pv.string, .type = ty, .div16 = div, .nospec = false });
        }
        var consts: std.json.ArrayHashMap(aot.SpecConst) = .{};
        var it = o.get("constexprs").?.object.iterator();
        while (it.next()) |kv| {
            const v = kv.value_ptr.object;
            var c: aot.SpecConst = .{};
            if (v.get("bool")) |bv| {
                c.int = @intFromBool(bv.bool);
            } else if (v.get("int")) |x| {
                c.int = x.integer;
            } else if (v.get("fp64_bits")) |bits| {
                const x: f64 = @bitCast(try std.fmt.parseInt(u64, bits.string[2..], 16));
                c.f32 = @bitCast(@as(f32, @floatCast(x)));
            }
            try consts.map.put(arena, kv.key_ptr.*, c);
        }
        const name = o.get("name").?.string;
        e.* = .{ .@"fn" = name, .hash = "", .name = name, .num_warps = 0, .params = params.items, .consts = consts };
    }
    return out;
}

/// Launches by how many variants take them, the kernels launched, and the misses by kernel and constexprs.
const Tally = struct {
    launches: usize = 0,
    one: usize = 0,
    none: usize = 0,
    many: usize = 0,
    kernels: std.StringArrayHashMapUnmanaged(usize) = .empty,
    misses: std.StringArrayHashMapUnmanaged(usize) = .empty,

    fn add(t: *Tally, arena: std.mem.Allocator, set: []const aot.Spec, e: trace.Event) !void {
        t.launches += 1;
        const k = try t.kernels.getOrPut(arena, e.name);
        if (!k.found_existing) k.value_ptr.* = 0;
        k.value_ptr.* += 1;
        var hits: usize = 0;
        for (set) |s| hits += @intFromBool(std.mem.eql(u8, s.@"fn", e.name) and aot.accepts(s, e.args, e.consts));
        switch (hits) {
            0 => {
                t.none += 1;
                var key: []const u8 = e.name;
                for (e.consts) |c| if (c.int) |v| {
                    key = try std.fmt.allocPrint(arena, "{s} {s}={d}", .{ key, c.name, v });
                };
                const g = try t.misses.getOrPut(arena, key);
                if (!g.found_existing) g.value_ptr.* = 0;
                g.value_ptr.* += 1;
            },
            1 => t.one += 1,
            else => t.many += 1,
        }
    }

    fn print(t: *const Tally, what: []const u8) void {
        std.debug.print("{s}: {d} Triton launches; one variant {d}, none {d}, several {d}; kernels:", .{ what, t.launches, t.one, t.none, t.many });
        var it = t.kernels.iterator();
        while (it.next()) |kv| std.debug.print(" {s} x{d}", .{ kv.key_ptr.*, kv.value_ptr.* });
        std.debug.print("\n", .{});
        var m = t.misses.iterator();
        while (m.next()) |kv| std.debug.print("  no variant: {s} (x{d})\n", .{ kv.key_ptr.*, kv.value_ptr.* });
    }
};

/// One DSA block call recorded, its Triton launches tallied.
fn call(arena: std.mem.Allocator, set: []const aot.Spec, t: *Tally, m: *const md.Model, l: *const views.Layer, st: *bufs.State, b: *const bufs.Buffers, x: *const dk.Scratch, R: usize, pos: usize) !void {
    var tr = trace.Trace.init(testing.allocator);
    defer tr.deinit();
    var f = fwd.Fwd.recording(m, &tr);
    st.pos = pos;
    try glm.dsa.partialWith(&f, l, st, b, R, .{ .nch = (pos + R + 511) / 512, .host_pos = pos }, x);
    try f.flush();
    for (tr.events.items) |e| if (e.op == .triton) try t.add(arena, set, e);
}

/// Layer 3's and the MTP layer's DSA block on the long prompt's shapes, one fixture's model.
fn record(arena: std.mem.Allocator, set: []const aot.Spec, text: []const u8, t: *Tally) !void {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const fx = parsed.value.object;
    var m = try md.Model.parse(gpa, fx.get("config_json").?.string, 2, 0);
    try m.cut(@intCast(fx.get("layers").?.integer));
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    var mtp = try views.mtpLayer(&m);
    var mm = m; // the MTP load keeps the MTP layer: its cache beside the indexer's
    mm.mtp_layers = @intCast(fx.get("config").?.object.get("mtp_layers").?.integer);
    var pb = try bufs.Buffers.init(&m, 2048, 6144, true);
    var db = try bufs.Buffers.init(&m, 8, 6144, false);
    var st = try bufs.State.init(gpa, &m, 6144, 8, true);
    defer st.deinit(gpa);
    var mst = try bufs.State.init(gpa, &mm, 6144, 8, true);
    defer mst.deinit(gpa);
    var x = dk.Scratch.init(m.hl(), pb.lat_s.qa.shape[2], 2048);
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try views.walk(&w, views.Path.init("w"), &space, trace.placeAll);
    try views.walk(&mtp, views.Path.init("w.mtp.layer"), &space, trace.placeAll);
    try views.walk(&pb, views.Path.init("pb"), &space, trace.placeAll);
    try views.walk(&db, views.Path.init("b"), &space, trace.placeAll);
    try views.walk(&st, views.Path.init("st"), &space, trace.placeAll);
    try views.walk(&mst, views.Path.init("mst"), &space, trace.placeAll);
    try views.walk(&x, views.Path.init("dsa"), &space, trace.placeAll);
    for ([_][2]usize{ .{ 2048, 0 }, .{ 37, 2048 }, .{ 2048, 2048 }, .{ 904, 4096 }, .{ 1537, 0 } }) |c| try call(arena, set, t, &m, &w.layers[3], &st, &pb, &x, c[0], c[1]);
    for ([_][2]usize{ .{ 1, 5000 }, .{ 1, 5047 }, .{ 8, 5000 } }) |c| try call(arena, set, t, &m, &w.layers[3], &st, &db, &x, c[0], c[1]);
    for ([_][2]usize{ .{ 2048, 0 }, .{ 36, 2048 }, .{ 2048, 2048 }, .{ 903, 4096 } }) |c| {
        mst.mtp_len = c[1];
        try call(arena, set, t, &mm, &mtp, &mst, &pb, &x, c[0], c[1]);
    }
}

test "G-cover: every Triton launch of the DSA block on the long prompt's chunks and steps finds exactly one variant (A4)" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const set = try entries(arena);
    for (fixtures, [_][]const u8{ "MLX 4-bit", "EXL3" }) |text, what| {
        var t: Tally = .{};
        try record(arena, set, text, &t);
        t.print(what);
        try testing.expect(t.launches > 0);
        try testing.expectEqual(t.launches, t.one);
        for ([_][]const u8{ "_dense_chunks", "_merge", "_scores", "_select_rows", "_sparse_chunks" }) |k| try testing.expect(t.kernels.contains(k));
    }
}
