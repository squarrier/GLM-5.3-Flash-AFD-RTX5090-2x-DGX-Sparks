//! patch 0002's `_fmm` (Mia's v1.10 FP8 head; qualification_qualification_ebed70b7) on a CPU, no GPU: Tri.matmulF8's launches against python's own
//! (q4f8.py's _matmul_f8 at the head's TP2 shape, recorded by evidence/qualification_qualification_ebed70b7/f8/f8aot.py: fixtures_triton_f8.json), and
//! each of them against the _fmm set's spec (kernels_f8.json, glm_aot.py's steps for sm_121a): exactly one variant
//! by aot.zig's rule; every captured variant serves one. The spec's cubins join ZK4's set121 as set121f8.
const std = @import("std");
const aot = @import("cuda").aot;
const tri = @import("../../../src/families/glm/cuda_triton.zig");

const testing = std.testing;
const fixture_text = @embedFile("fixtures_triton_f8.json");
const spec_text = @embedFile("kernels_f8.json");

/// A rank's lm_head: 154,880 / 2 vocabulary rows over hidden 4,096
const head_n = 77440;
const head_k = 4096;

fn addr(i: u64) u64 {
    return 0x7f0000000000 + i * 0x10000000;
}

/// One launch as aot.zig's `find` sees it, from a fixture entry.
const Launch = struct { name: []const u8, args: [8]aot.Arg = undefined, na: usize = 0, consts: [8]aot.Const = undefined, nc: usize = 0 };

fn launchOf(o: std.json.ObjectMap) Launch {
    var l: Launch = .{ .name = o.get("fn").?.string };
    for (o.get("args").?.array.items) |a| {
        const ao = a.object;
        const n = ao.get("name").?.string;
        const ty = ao.get("type").?.string;
        l.args[l.na] = if (ty[0] == '*') aot.ptr(n, ty, if (ao.get("align16").?.bool) addr(0) else addr(0) + 8) else aot.int(n, @intCast(ao.get("int").?.integer));
        l.na += 1;
    }
    var it = o.get("consts").?.object.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* != .null) {
        l.consts[l.nc] = aot.ci(kv.key_ptr.*, kv.value_ptr.object.get("int").?.integer);
        l.nc += 1;
    };
    return l;
}

/// The spec's entries as aot.json rows (glm_aot.py's aot_row and zig_const; triton_set.zig's reading).
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
            const c: aot.SpecConst = if (v.get("bool")) |b| .{ .int = @intFromBool(b.bool) } else .{ .int = v.get("int").?.integer };
            try consts.map.put(arena, kv.key_ptr.*, c);
        }
        const name = o.get("name").?.string;
        e.* = .{ .@"fn" = name, .hash = if (o.get("rule") != null) "" else o.get("hash").?.string, .name = name, .num_warps = 0, .params = params.items, .consts = consts };
    }
    return out;
}

fn hits(set: []const aot.Spec, l: Launch, used: ?[]usize) usize {
    var n: usize = 0;
    for (set, 0..) |e, i| if (std.mem.eql(u8, e.@"fn", l.name) and aot.accepts(e, l.args[0..l.na], l.consts[0..l.nc])) {
        n += 1;
        if (used) |u| u[i] += 1;
    };
    return n;
}

test "matmulF8 launches what python's _matmul_f8 launches at the head's 1-8, 16, 512 and 1,024 rows" {
    const root = try std.json.parseFromSlice(std.json.Value, testing.allocator, fixture_text, .{});
    defer root.deinit();
    var it = root.value.object.get("cases").?.object.iterator();
    var cases: usize = 0;
    while (it.next()) |c| {
        const rows = try std.fmt.parseInt(usize, c.key_ptr.*["fmm/head/r".len..], 10);
        var rec: tri.Recorder = .{};
        const t: tri.Tri = .{ .set = null, .rec = &rec };
        try t.matmulF8(addr(1), head_k, addr(2), addr(3), addr(4), false, addr(5), rows, head_n, head_k);
        const want = c.value_ptr.array.items;
        try testing.expectEqual(want.len, rec.n);
        for (want, rec.launches[0..rec.n]) |w, got| {
            const l = launchOf(w.object);
            errdefer std.debug.print("case {s}\n", .{c.key_ptr.*});
            try testing.expectEqualStrings(l.name, got.name);
            for (w.object.get("grid").?.array.items, got.grid) |g, x| try testing.expectEqual(g.integer, @as(i64, x));
            try testing.expectEqual(l.na, got.nargs);
            for (l.args[0..l.na], got.args[0..got.nargs]) |a, b| {
                try testing.expectEqualStrings(a.name, b.name);
                switch (a.value) {
                    .ptr => |x| try testing.expectEqualStrings(x.ty, b.value.ptr.ty),
                    .i32 => |x| try testing.expectEqual(x, b.value.i32),
                    else => return error.UnexpectedArg,
                }
            }
            try testing.expectEqual(l.nc, got.nconsts);
            for (l.consts[0..l.nc], got.consts[0..got.nconsts]) |a, b| {
                try testing.expectEqualStrings(a.name, b.name);
                try testing.expectEqual(a.int.?, b.int.?);
            }
        }
        cases += 1;
    }
    try testing.expectEqual(@as(usize, 11), cases);
}

test "each of python's _fmm launches finds exactly one variant of the _fmm set; every captured variant serves one" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const set = try entries(arena);
    const used = try arena.alloc(usize, set.len);
    @memset(used, 0);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, fixture_text, .{});
    var it = root.object.get("cases").?.object.iterator();
    var launches: usize = 0;
    while (it.next()) |c| for (c.value_ptr.array.items) |w| {
        const l = launchOf(w.object);
        if (hits(set, l, used) != 1) {
            std.debug.print("case {s}: {d} variants\n", .{ c.key_ptr.*, hits(set, l, null) });
            return error.NotOneVariant;
        }
        launches += 1;
    };
    try testing.expectEqual(@as(usize, 11), launches);
    try testing.expectEqual(@as(usize, 9), set.len); // BM 16: M 1, div16, plain; BM 32, 64, 128: div16, plain
    for (set, used) |e, n| if (e.hash.len > 0 and n == 0) return error.DeadVariant;
    // negative control: B16's 64 columns at a 16-row bucket is no variant of this set
    var bad = launchOf((try std.json.parseFromSliceLeaky(std.json.Value, arena, fixture_text, .{})).object.get("cases").?.object.get("fmm/head/r1").?.array.items[0].object);
    for (bad.consts[0..bad.nc]) |*k| if (std.mem.eql(u8, k.name, "BLOCK_N")) {
        k.int = 64;
    };
    try testing.expectEqual(@as(usize, 0), hits(set, bad, null));
}
