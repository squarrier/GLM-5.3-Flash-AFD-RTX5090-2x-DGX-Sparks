//! qualification_qualification_abcce4b2 A1's launch check: tools/zig/glm_mm_fixtures.py ran python-0.6's own qmm and experts wrappers (the Python ones and
//! their C++, the CUDA runtime stood in) on a CPU for GLM-5.3-Flash's calls, the slice-1 forward trace's included. Each
//! launch python made (kernel, grid, block, dynamic shared bytes, cluster, PDL, and every argument's bytes at the offset
//! and size its cubin's EIATTR_KPARAM_INFO gives) must be the launch the bindings make from the same call; where python's
//! kernel is not one of GLM's instances (another chip, a wider window) the bindings must refuse with NotBuilt. The
//! forward's launch halves (cuda_fwd.zig qmmGroup, qmmPrefill) must make the same launches from the same tensors.
//! qualification_18588626: tools/zig/glm_mm_w1_fixtures.py's qmm_group calls at the 3-node attention's world-1 shapes (every shape its
//! kits' --dry records, tools/zig/glm_mm_w1_shapes.json) on the RTX 5090 and a GB10 (fixtures_cuda_mm_w1.json).
const std = @import("std");
const mm = @import("cuda_kernels_mm.zig");
const fwd = @import("cuda_fwd.zig");
const trace = @import("cuda_trace.zig");

const calls = mm.calls;
const testing = std.testing;
const J = std.json.Value;
const fixture = @embedFile("fixtures_cuda_mm.json");
const fixture_w1 = @embedFile("fixtures_cuda_mm_w1.json");

fn get(o: J, k: []const u8) J {
    return o.object.get(k) orelse std.debug.panic("fixture: no key {s}", .{k});
}

fn int(v: J) i64 {
    return v.integer;
}

/// The fixture's stand-in addresses (one 2^36-byte region a storage, the view's offset into it).
const StandIn = struct {
    pub fn addr(_: StandIn, t: J) !u64 {
        return @intCast(int(get(t, "addr")));
    }
};

fn dims(want: J, got: anytype) !void {
    const w = want.array.items;
    try testing.expectEqual(int(w[0]), @as(i64, got.x));
    try testing.expectEqual(int(w[1]), @as(i64, got.y));
    try testing.expectEqual(int(w[2]), @as(i64, got.z));
}

/// One launch against python's: kernel, configuration, and each argument's offset, size and bytes.
fn expectLaunch(want: J, got: mm.Launch, kparams: J) !void {
    const k = calls.key(got.kernel.symbol());
    try testing.expectEqualStrings(get(want, "kernel").string, k);
    try dims(get(want, "grid"), got.cfg.grid);
    try dims(get(want, "block"), got.cfg.block);
    try testing.expectEqual(int(get(want, "smem")), got.cfg.shared);
    const no_cluster: @TypeOf(got.cfg.grid) = .{ .x = 0, .y = 0, .z = 0 };
    try dims(get(want, "cluster"), got.cfg.cluster orelse no_cluster);
    try testing.expectEqual(int(get(want, "pdl")) != 0, got.cfg.pdl);
    const args = get(want, "args").array.items;
    const params = get(kparams, k).array.items;
    try testing.expectEqual(params.len, got.args.count);
    try testing.expectEqual(args.len, got.args.count);
    for (args, params, 0..) |a, p, i| {
        const off = got.args.offsets[i];
        const size = got.args.sizes[i];
        try testing.expectEqual(int(p.array.items[0]), off);
        try testing.expectEqual(int(p.array.items[1]), size);
        var hex: [2 * mm.max_arg_bytes]u8 = undefined;
        for (got.args.storage[off..][0..size], 0..) |byte, j| {
            hex[2 * j] = "0123456789abcdef"[byte >> 4];
            hex[2 * j + 1] = "0123456789abcdef"[byte & 15];
        }
        try testing.expectEqualStrings(a.string, hex[0 .. 2 * size]);
    }
}

fn sameLaunch(a: mm.Launch, b: mm.Launch) !void {
    try testing.expectEqual(a.kernel, b.kernel);
    try testing.expectEqualDeep(a.cfg, b.cfg);
    try testing.expectEqual(a.args.count, b.args.count);
    try testing.expectEqualSlices(u16, a.args.offsets[0..a.args.count], b.args.offsets[0..b.args.count]);
    try testing.expectEqualSlices(u16, a.args.sizes[0..a.args.count], b.args.sizes[0..b.args.count]);
    try testing.expectEqualSlices(u8, a.args.storage[0..a.args.used], b.args.storage[0..b.args.used]);
}

fn tensor(v: J) !trace.T {
    const t = try calls.tensor(v, StandIn{});
    var out: trace.T = .{ .ptr = t.ptr, .dtype = .bf16, .dims = @intCast(t.dims) };
    out.shape = t.shape;
    out.stride = t.stride;
    return out;
}

fn only(v: J) J {
    return get(v, "list").array.items[0];
}

/// The forward's own launch half for a qmm call, from trace tensors (the dtype is not read).
fn forwardLaunch(call: J, chip: mm.Chip) !?mm.Launch {
    const f = get(call, "fn").string;
    const a = get(call, "args").array.items;
    const n = struct {
        fn of(v: J) usize {
            return @intCast(int(get(v, "int")));
        }
    }.of;
    if (std.mem.eql(u8, f, "qmm_group")) {
        const w = [3]trace.T{ try tensor(only(a[2])), try tensor(only(a[3])), try tensor(only(a[4])) };
        return try fwd.qmmGroupLaunch(chip, try tensor(a[0]), try tensor(a[1]), w, try tensor(only(a[5])), n(only(a[6])), n(only(a[7])), get(a[8], "bool").bool, n(a[9]));
    }
    if (std.mem.eql(u8, f, "qmm_prefill")) return try fwd.qmmPrefillLaunch(try tensor(a[0]), .{ try tensor(a[1]), try tensor(a[2]), try tensor(a[3]) }, try tensor(a[4]), n(a[5]), get(a[7], "bool").bool, n(a[8]));
    return null;
}

/// What a fixture's comparison found: calls, launches equal to python's, calls refused (NotBuilt), qmm calls whose
/// forward launch half made the same launch, raised limits equal to Mm.load's, and launches by kernel.
const Counts = struct { calls: usize = 0, equal: usize = 0, refused: usize = 0, forward: usize = 0, raised: usize = 0, by_kernel: [mm.Kernel.count]usize = @splat(0) };

/// Every call of a glm-mm-launches/1 fixture: python's launches equal to the bindings' (and the forward's launch half's),
/// or refused with NotBuilt where python's kernel is not one of GLM's instances; its raised limits are Mm.load's, and
/// with `every_limit` each raised kernel of Mm.load is among them.
fn compareAll(text: []const u8, every_limit: bool) !Counts {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, text, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-mm-launches/1", get(fx, "schema").string);
    const kparams = get(fx, "kparams");
    const occ: usize = @intCast(int(get(fx, "occupancy")));
    var n: Counts = .{};
    for (get(fx, "cases").array.items) |c| {
        const id = get(c, "id").string;
        errdefer std.debug.print("fixture case {s}\n", .{id});
        const ch = get(c, "chip").array.items;
        const chip: mm.Chip = .{ .major = @intCast(int(ch[0])), .minor = @intCast(int(ch[1])), .sms = @intCast(int(ch[2])) };
        for (get(c, "calls").array.items) |call| {
            n.calls += 1;
            try testing.expect(call.object.get("error") == null);
            const want = if (call.object.get("launches")) |l| l.array.items else &.{};
            var all_ours = true;
            for (want) |w| all_ours = all_ours and calls.kernelOf(get(w, "kernel").string) != null;
            if (!all_ours) {
                try testing.expectError(error.NotBuilt, calls.launches(call, chip, .{ occ, occ }, StandIn{}));
                n.refused += 1;
                continue;
            }
            const got = try calls.launches(call, chip, .{ occ, occ }, StandIn{});
            try testing.expectEqual(want.len, got.n);
            for (want, got.l[0..got.n]) |w, l| {
                try expectLaunch(w, l, kparams);
                n.by_kernel[@backingInt(l.kernel)] += 1;
            }
            n.equal += want.len;
            if (try forwardLaunch(call, chip)) |l| {
                try sameLaunch(got.l[0], l);
                n.forward += 1;
            }
        }
    }
    // the limits the wrappers raise, and the decode experts' occupancy queries, are the ones Mm.load makes
    for (get(fx, "attrs").array.items) |a| {
        const it = a.array.items;
        const k = calls.kernelOf(it[0].string) orelse continue;
        try testing.expectEqual(@as(i64, 8), int(it[1])); // cudaFuncAttributeMaxDynamicSharedMemorySize
        try testing.expectEqual(int(it[2]), k.raised());
        n.raised += 1;
    }
    if (every_limit) for (std.enums.values(mm.Kernel)) |k| {
        if (k.raised() == 0) continue;
        var seen = false;
        for (get(fx, "attrs").array.items) |a| seen = seen or std.mem.eql(u8, a.array.items[0].string, calls.key(k.symbol()));
        try testing.expect(seen);
    };
    for (get(fx, "cases").array.items) |c| for (get(c, "calls").array.items) |call| {
        const qs = call.object.get("occupancy_queries") orelse continue;
        for (qs.array.items) |q| {
            const it = q.array.items;
            const k = calls.kernelOf(it[0].string) orelse return error.TestUnexpectedResult;
            try testing.expect(k == .gate_up or k == .down);
            try testing.expectEqual(@as(i64, 128), int(it[1]));
            try testing.expectEqual(@as(i64, 0), int(it[2]));
        }
    };
    return n;
}

test "every launch of python-0.6's qmm and experts wrappers for GLM's calls is the bindings' launch" {
    const n = try compareAll(fixture, true);
    // qualification_18588626: the RTX 5090's 1-, 8- and 16-row windows (tiles 7 and 8) are launches now; its 17-row ones (tile 12) refuse
    try testing.expect(n.equal > 200 and n.refused > 0 and n.forward > 150 and n.raised == 10);
    try testing.expectEqual(@as(usize, 4), n.by_kernel[@backingInt(mm.Kernel.group7_bf16)] + n.by_kernel[@backingInt(mm.Kernel.group7_f32)]);
    try testing.expectEqual(@as(usize, 2), n.by_kernel[@backingInt(mm.Kernel.group8_bf16)] + n.by_kernel[@backingInt(mm.Kernel.group8_f32)]);
    std.debug.print("glm mm fixture: {d} calls; {d} launches equal to python-0.6's ({d} qmm calls through the forward's launch halves too), {d} calls refused (NotBuilt), {d} raised limits\n", .{ n.calls, n.equal, n.forward, n.refused, n.raised });
}

test "qualification_18588626: python-0.6's qmm_group launches at the 3-node attention's world-1 shapes (decode, verify windows, the draft head; 9-16 rows) on the RTX 5090 and a GB10 are the bindings'" {
    const n = try compareAll(fixture_w1, false);
    try testing.expectEqual(@as(usize, 0), n.refused);
    try testing.expectEqual(n.calls, n.equal);
    try testing.expectEqual(n.calls, n.forward);
    try testing.expectEqual(@as(usize, 6), n.raised); // tiles 2, 7 and 8, bf16 and fp32 out
    var tile7: usize = 0;
    var tile8: usize = 0;
    var tile2: usize = 0;
    for (std.enums.values(mm.Kernel), n.by_kernel) |k, c| switch (k) {
        .group7_bf16, .group7_f32 => tile7 += c,
        .group8_bf16, .group8_f32 => tile8 += c,
        .group_bf16, .group_f32 => tile2 += c,
        else => try testing.expectEqual(@as(usize, 0), c),
    };
    try testing.expect(tile7 > 0 and tile8 > 0 and tile2 == tile7 + tile8); // each shape on both chips
    try testing.expectEqual(4 * tile8, tile7); // each projection at the 8 windows of 1-8 rows and at 9 and 16
    std.debug.print("glm mm fixture (world 1, qualification_18588626): {d} calls, {d} launches equal to python-0.6's (the forward's launch halves too): RTX 5090 tile 7 {d}, tile 8 {d}; GB10 tile 2 {d}; {d} raised limits\n", .{ n.calls, n.equal, tile7, tile8, tile2, n.raised });
}
