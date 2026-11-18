// Modified for V4 by Scott Quarrier with Helios: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! cuda_g53.zig's CPU tests: the arguments' layout and every launch against the cubins' tables (g53_kernels.json), the
//! configuration per window size (default_cfg, KT3R's tier), the plan's buffers and the binding's refusals.

const std = @import("std");
const g = @import("cuda_g53.zig");
const ex3 = @import("cuda_v3_g53_types.zig");
const testing = std.testing;

test "Args is g53_rank.cu's struct: 272 bytes, every field where the C++ compiler puts it" {
    try testing.expectEqual(@as(usize, 272), @sizeOf(g.Args));
    const at = [_]struct { []const u8, usize }{ .{ "x", 72 }, .{ "xs", 80 }, .{ "pick", 88 }, .{ "ps", 104 }, .{ "ws", 108 }, .{ "groups", 112 }, .{ "fault", 208 }, .{ "P", 216 }, .{ "nch", 244 }, .{ "l2hint", 248 }, .{ "limit", 252 }, .{ "xh", 256 }, .{ "gorder", 264 } };
    inline for (at) |f| try testing.expectEqual(f[1], @offsetOf(g.Args, f[0]));
}

test "default_cfg and KT3R's tier at every window of 1..4,096 rows: one set of bit fields, KT3 from 1,536 rows only" {
    for (1..g.MAX_ROWS + 1) |r| {
        const R: u32 = @intCast(r);
        const d = try g.defaultCfg(R);
        const k = try g.tunedCfg(R, true);
        try testing.expectEqual(d, try g.tunedCfg(R, false));
        for ([_]g.Cfg{ d, k }) |c| {
            try testing.expect(g.cfgOk(c));
            try testing.expectEqual([3]u32{ 1, 1, 0 }, [3]u32{ c.sk, c.skd, c.swiglu }); // the bit fields: every window
            try testing.expectEqual([4]u32{ 2, 2, 2, 1 }, [4]u32{ c.fuse, c.discard, c.rot, c.gord });
        }
        const want: [4]u32 = if (R <= 2048) .{ 2, 8, 2, 2 } else .{ 4, 16, 4, 1 };
        try testing.expectEqual(want, [4]u32{ d.mt, d.gw, d.nt, d.l2 });
        const kt: [4]u32 = if (R >= g.kt3_tier) .{ 4, 16, 4, 1 } else want;
        try testing.expectEqual(kt, [4]u32{ k.mt, k.gw, k.nt, k.l2 });
    }
    try testing.expectError(error.Rows, g.defaultCfg(0));
    try testing.expectError(error.Rows, g.tunedCfg(4097, true));
}

test "max_groups: glm53f-rank's bound (python's values)" {
    try testing.expectEqual(@as(u32, 8), g.maxGroups(1, 2));
    try testing.expectEqual(@as(u32, 800), g.maxGroups(2048, 2));
    try testing.expectEqual(@as(u32, 544), g.maxGroups(2048, 4));
    try testing.expectEqual(@as(u32, 1312), g.maxGroups(4096, 2));
    try testing.expectEqual(@as(u32, 304), g.maxGroups(64, 2)); // 512 routes: 16 full groups + 288
    for (1..g.MAX_ROWS + 1) |r| try testing.expect(g.maxGroups(@intCast(r), 4) <= g.maxGroups(@intCast(r), 2));
}

test "the plan block: python's Scratch buffers, 256-byte aligned, not overlapping" {
    for ([_]u32{ 1, 7, 2048, 4096 }) |R| {
        const p = g.Plan.place(R, 1 << 20);
        const s = g.Plan.sizes(R);
        const at = [_]u64{ p.groups, p.meta, p.pair_row, p.pair_route, p.inverse, p.gcnt, p.rcnt, p.gorder };
        for (at, 0..) |a, i| {
            try testing.expect(a % 256 == 0);
            if (i > 0) try testing.expect(a >= at[i - 1] + s[i - 1]);
        }
        try testing.expectEqual(@as(u64, (1 << 20) + g.Plan.bytes(R)), at[7] + s[7]);
        try testing.expectEqual(@as(usize, 16 * g.maxGroups(R, 2)), s[0]);
        try testing.expectEqual(@as(usize, 4 * 32 * R), s[6]);
    }
}

const layer: ex3.Layer = .{ .gate_ptr = 0x1000, .up_ptr = 0x2000, .down_ptr = 0x3000, .gate_k2 = 0x4000, .up_k2 = 0x5000, .down_k2 = 0x6000, .suh_g = 0x7000, .suh_u = 0x8000, .svh_g = 0x9000, .svh_u = 0xa000, .suh_d = 0xb000, .svh_d = 0xc000, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };

fn call(R: u32, suh: ?u64) g.Call {
    const n = g.bigNeeds(g.MAX_ROWS, .{ .mt = 2, .sk = 1, .skd = 1, .nt = 2, .gw = 8, .l2 = 2, .fuse = 2, .discard = 2, .swiglu = 0, .rot = 2, .gord = 1 });
    return .{ .layer = layer, .shared_suh = suh, .x = 0x10000, .xs = 4096, .pick = 0x20000, .ps = 9, .wts = 0x30000, .ws = 9, .out = 0x40000, .rows = R, .limit = 10.0, .plan = g.Plan.place(g.MAX_ROWS, 0x100000), .big = .{ .z = 0x1000000, .xd = 0x2000000, .zd = 0x3000000, .xh = 0x4000000, .z_bytes = n.z, .xd_bytes = n.xd, .zd_bytes = n.zd, .xh_bytes = n.xh } };
}

fn argsOf(l: g.Launch) g.Args {
    return std.mem.bytesToValue(g.Args, l.args.storage[0..@sizeOf(g.Args)]);
}

test "g53_ffn_launch's sequence: plan, rotation with a shared suh, gate/up, down; KT3 at 1,536+ rows; the unfused kernels with fuse 1" {
    for ([_]u32{ 1, 64, 1535, 1536, 2048, 4096 }) |R| for ([_]bool{ false, true }) |kt3r| for ([_]?u64{ null, 0x50000 }) |suh| {
        var ls: g.Launches = .{};
        const c = try g.tunedCfg(R, kt3r);
        try g.ffn(&ls, call(R, suh), c);
        const rot = suh != null;
        try testing.expectEqual(@as(usize, if (rot) 4 else 3), ls.len);
        const s = ls.slice();
        try testing.expectEqual(call(R, suh).plan.meta, ls.meta);
        try testing.expectEqualStrings(g.sym.plan, s[0].symbol);
        try testing.expectEqual(@as(?u64, R), s[0].args.integer(4));
        try testing.expectEqual(@as(?u64, 16 * c.mt), s[0].args.integer(5));
        const gu = s[if (rot) 2 else 1];
        const dn = s[if (rot) 3 else 2];
        if (rot) {
            try testing.expectEqualStrings(g.sym.rotrows, s[1].symbol);
            try testing.expectEqual(@as(u32, (R * 32 + 7) / 8), s[1].cfg.grid.x);
            try testing.expectEqual(@as(?u64, 0x50000), s[1].args.integer(1));
        }
        try testing.expectEqualStrings(g.sym.gateup(c.mt, c.gw, rot), gu.symbol);
        try testing.expectEqualStrings(g.sym.down(c.mt, c.nt), dn.symbol);
        try testing.expectEqual(c.gw * 32, gu.cfg.block.x);
        try testing.expectEqual(@as(u32, 256), dn.cfg.block.x);
        const mg = g.maxGroups(R, c.mt);
        try testing.expectEqual(4 * mg, gu.items);
        try testing.expectEqual(mg * (32 / c.nt), dn.items);
        const a = argsOf(gu);
        try testing.expectEqual(a, argsOf(dn));
        try testing.expectEqual([9]c_int{ @intCast(8 * R), @intCast(R), 1, 1, 1, 1, 1, @intCast(32 / c.nt), @intFromBool(c.l2 == 2) }, [9]c_int{ a.P, a.rows, a.sk, a.skd, a.bf16, a.fused, a.discard, a.nch, a.l2hint });
        try testing.expectEqual(if (rot) call(R, suh).big.xh else 0, a.xh);
        try testing.expectEqual(@as(u64, 0), a.gorder);
        try testing.expectEqual(ls.meta + 8, a.work);
        try testing.expectEqual(ls.meta + 4, a.fault);
        try testing.expectEqual(@as(f32, 10.0), a.limit);
    };
    var ls: g.Launches = .{};
    var c = try g.defaultCfg(7);
    c.fuse = 1;
    c.gord = 2;
    c.swiglu = 1;
    try g.ffn(&ls, call(7, 0x50000), c);
    var names: [7][]const u8 = undefined;
    for (ls.slice(), 0..) |l, i| names[i] = l.symbol;
    try testing.expectEqual(@as(usize, 7), ls.len);
    for ([_][]const u8{ g.sym.plan, g.sym.order, g.sym.rotrows, g.sym.gateup(2, 8, true), g.sym.epilogue, g.sym.down(2, 2), g.sym.reduce }, names) |w, n| try testing.expectEqualStrings(w, n);
    try testing.expectEqual(@as(u32, 7 * 8 * 8 / 4), ls.items[4].cfg.grid.x);
    try testing.expectEqual(@as(u32, 7 * 8), ls.items[6].cfg.grid.x);
    const a = argsOf(ls.items[3]);
    try testing.expectEqual([3]c_int{ 0, 0, 0 }, [3]c_int{ a.bf16, a.fused, a.discard });
    try testing.expect(a.gorder != 0);
}

test "the binding's refusals: rows, configuration, the layer's shape, the strides, alignments and buffer sizes" {
    const c = try g.defaultCfg(64);
    var ls: g.Launches = .{};
    try g.ffn(&ls, call(64, null), c);
    const bad = [_]struct { g.Call, g.Cfg }{
        .{ call(0, null), c },
        .{ call(4097, null), c },
        .{ blk: {
            var x = call(64, null);
            x.xs = 4094;
            break :blk x;
        }, c },
        .{ blk: {
            var x = call(64, null);
            x.x += 2;
            break :blk x;
        }, c },
        .{ blk: {
            var x = call(64, null);
            x.big.z += 64;
            break :blk x;
        }, c },
        .{ blk: {
            var x = call(64, null);
            x.big.zd_bytes = g.bigNeeds(64, c).zd - 4;
            break :blk x;
        }, c },
        .{ blk: {
            var x = call(64, null);
            x.plan.rows = 63;
            break :blk x;
        }, c },
        .{ blk: {
            var x = call(64, null);
            x.layer.width = 2048;
            break :blk x;
        }, c },
        .{ blk: {
            var x = call(64, null);
            x.ps = 7;
            break :blk x;
        }, c },
        .{ call(64, null), blk: {
            var y = c;
            y.sk = 3;
            break :blk y;
        } },
        .{ call(64, null), blk: {
            var y = c;
            y.mt = 8;
            break :blk y;
        } },
        .{ call(64, null), blk: {
            var y = c;
            y.skd = 8;
            break :blk y;
        } },
    };
    for (bad) |b| {
        var l2: g.Launches = .{};
        try testing.expectError(error.Invalid, g.ffn(&l2, b[0], b[1]));
    }
    var tb: [36 * 288]u8 = @splat(0);
    for (0..3 * 288) |i| std.mem.writeInt(u64, tb[i * 8 ..][0..8], 0x100000 + 1024 * i, .little);
    try testing.expectEqual(@as(?[]const u8, null), g.layerReady(layer, &tb));
    try testing.expect(g.layerReady(layer, tb[0 .. 24 * 288 - 1]) != null); // no whole table
    std.mem.writeInt(u64, tb[(2 * 288 + 5) * 8 ..][0..8], 0x100002, .little); // one down trellis 2 bytes off
    try testing.expect(g.layerReady(layer, &tb) != null);
    std.mem.writeInt(u64, tb[(2 * 288 + 5) * 8 ..][0..8], 0x100004, .little);
    try testing.expectEqual(@as(?[]const u8, null), g.layerReady(layer, &tb));
    var l3 = layer;
    l3.k2_d = .{ 6, 8 };
    try testing.expect(g.layerReady(l3, &tb) != null);
    l3 = layer;
    l3.cb = .@"3inst";
    try testing.expect(g.layerReady(l3, &tb) != null);
    l3 = layer;
    l3.svh_u += 4;
    try testing.expect(g.layerReady(l3, &tb) != null);
}

test "launch_persistent's grid: SMs x occupancy, at most the items, at least one block" {
    try testing.expectEqual(@as(u32, 96), g.persistentGrid(48, 2, 4000));
    try testing.expectEqual(@as(u32, 37), g.persistentGrid(48, 2, 37));
    try testing.expectEqual(@as(u32, 48), g.persistentGrid(48, 0, 4000));
    try testing.expectEqual(@as(u32, 1), g.persistentGrid(48, 1, 0));
}

/// g53_kernels.json (tools/zig/exl3_kernels.py over the sm_121 and sm_120 cubins, which agree).
const Entry = struct { symbol: []const u8, params: []const [2]u32, shared: u32 };
const Table = struct { g53_rank: []const Entry };

fn entry(t: Table, symbol: []const u8) ?Entry {
    for (t.g53_rank) |e| if (std.mem.eql(u8, e.symbol, symbol)) return e;
    return null;
}

test "every launch's arguments sit where the cubins' parameter tables put them, and every one of the 19 kernels is launched" {
    const parsed = try std.json.parseFromSlice(Table, testing.allocator, @embedFile("g53_kernels.json"), .{});
    defer parsed.deinit();
    const t = parsed.value;
    try testing.expectEqual(@as(usize, 19), t.g53_rank.len);
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(testing.allocator);
    for ([_]u32{ 1, 1536, 2048, 4096 }) |R| for ([_]u32{ 1, 2 }) |fuse| for ([_]u32{ 1, 2 }) |gord| for ([_]?u64{ null, 0x50000 }) |suh| for ([_]u32{ 2, 4 }) |mt| for ([_]u32{ 8, 16 }) |gw| for ([_]u32{ 1, 2, 4 }) |nt| {
        var c = try g.defaultCfg(R);
        c.fuse = fuse;
        c.gord = gord;
        c.mt = mt;
        c.gw = gw;
        c.nt = nt;
        var ls: g.Launches = .{};
        try g.ffn(&ls, call(R, suh), c);
        for (ls.slice()) |l| {
            const e = entry(t, l.symbol) orelse {
                std.debug.print("{s} is not in the cubins\n", .{l.symbol});
                return error.UnknownSymbol;
            };
            try testing.expectEqual(e.params.len, l.args.count);
            for (e.params, 0..) |p, i| {
                try testing.expectEqual(p[0], l.args.offsets[i]);
                try testing.expectEqual(p[1], l.args.sizes[i]);
            }
            try seen.put(testing.allocator, l.symbol, {});
        }
    };
    try testing.expectEqual(t.g53_rank.len, seen.count());
    for (g.gateups) |x| try testing.expect(entry(t, x.name) != null);
    for (g.downs) |x| try testing.expect(entry(t, x.name) != null);
    for (t.g53_rank) |e| try testing.expect(e.shared <= 48 * 1024); // static only: launched with no dynamic shared memory
}
