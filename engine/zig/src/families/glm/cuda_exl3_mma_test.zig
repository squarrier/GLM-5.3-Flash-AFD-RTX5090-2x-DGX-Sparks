//! qualification_8724e161's MMA in the bindings (qualification_7af419c1's qualification_babb0a2d; bertholomus's grouped_mma and work list, 90b68e0, Apache-2.0): the switch and
//! its refusals, the work list's length and its cover of every member, the MMA path's launches against X1's (the
//! grouping, rotation and epilogues X1's; gate/up and down through exl3/mma/), the instances, and the chains that keep
//! X1's bits.

const std = @import("std");
const ex3 = @import("cuda_exl3_experts.zig");
const mma = @import("cuda_exl3_mma.zig");
const k = @import("cuda_exl3_kernels.zig");
const testing = std.testing;

const layer: ex3.Layer = .{ .gate_ptr = 0x1000, .up_ptr = 0x2000, .down_ptr = 0x3000, .gate_k2 = 0x4000, .up_k2 = 0x5000, .down_k2 = 0x6000, .suh_g = 0x7000, .suh_u = 0x8000, .svh_g = 0x9000, .svh_u = 0xa000, .suh_d = 0xb000, .svh_d = 0xc000, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
const sh: ex3.Shared = .{ .optin = 101376 };
const counts: u64 = 0xc0000;
const work: u64 = 0xd0000;

fn scratch(rows: u32) ex3.Scratch {
    return .{ .xg = 0x10000, .xu = 0x20000, .xd = 0x30000, .z = 0x40000, .y = 0x50000, .ids = 0x60000, .count = 0x70000, .members = 0x80000, .rows = rows, .slots = 9, .experts = 288, .cfg_gu = ex3.glm_gateup, .cfg_d = ex3.glm_down };
}

fn argBytes(l: k.Launch) []const u8 {
    return l.args.storage[0..l.args.used];
}

fn expectSame(a: k.Launch, b: k.Launch) !void {
    try testing.expectEqual(a.image, b.image);
    try testing.expectEqualStrings(a.symbol, b.symbol);
    try testing.expectEqual(a.cfg, b.cfg);
    try testing.expectEqual(a.max_dynamic, b.max_dynamic);
    try testing.expectEqualSlices(u8, argBytes(a), argBytes(b));
}

fn prompt(sel: u32, seq: bool, on: bool) ex3.Prompt {
    return .{ .nfirst = sel & 8 != 0, .order = sel & 4 != 0, .pa = sel & 2 != 0, .xrow = sel & 1 != 0, .seq = seq, .mma = on };
}

/// routedX1 for X1's launches; the MMA path as routedX1 enters it (its grouping, then mma.routed).
fn routedX1(ls: *k.Launches, R: u32, wts: ?u64, p: ex3.Prompt, shared: ?u64) !void {
    try ex3.routedX1(ls, 0xd000, 4096, true, 0xe000, wts, layer, scratch(2048), 0xa0000, R, 7.0, .bf16, true, sh, .{}, p, 0x90000, shared);
}

fn routedMma(ls: *k.Launches, R: u32, wts: ?u64, p: ex3.Prompt, shared: ?u64) !void {
    try mma.routed(ls, 0xd000, 4096, true, 0xe000, wts, layer, scratch(2048), 0xa0000, R, 7.0, .bf16, p, 0x90000, shared, counts, work);
}

test "TF_GLM_EXL3_MMA: unset, empty or 0 off; 1 on with X1 in effect and PGROUP; 1 without either, and any other value, refused (BadExl3Mma)" {
    const x1 = try ex3.Prompt.parse("1", .{ null, null, null, null });
    for ([_]?[]const u8{ null, "", "0" }) |v| for ([_]bool{ false, true }) |pg| {
        try testing.expectEqual(x1, try ex3.Prompt.parseMma(x1, v, pg));
        try testing.expectEqual(ex3.Prompt{}, try ex3.Prompt.parseMma(.{}, v, pg));
    };
    var on = x1;
    on.mma = true;
    try testing.expectEqual(on, try ex3.Prompt.parseMma(x1, "1", true));
    try testing.expectError(error.BadExl3Mma, ex3.Prompt.parseMma(x1, "1", false)); // PGROUP off: no counts for the list
    try testing.expectError(error.BadExl3Mma, ex3.Prompt.parseMma(.{}, "1", true)); // X1 unset
    try testing.expectError(error.BadExl3Mma, ex3.Prompt.parseMma(try ex3.Prompt.parse("1", .{ "0", "0", "0", "0" }), "1", true));
    for ([_][]const u8{ "2", "yes", "on", " 1", "01", "true", "-1" }) |bad| {
        try testing.expectError(error.BadExl3Mma, ex3.Prompt.parseMma(x1, bad, true));
        try testing.expectError(error.BadExl3Mma, ex3.Prompt.parseMma(.{}, bad, false));
    }
    var both = try ex3.Prompt.parseSeq(x1, "1"); // SEQ with MMA: gate/up SEQ's kernel, down grouped_mma
    both = try ex3.Prompt.parseMma(both, "1", true);
    try testing.expect(both.seq and both.mma);
}

test "the shape rule: both of GLM's half projections take MMA's 4-tile steps (gate/up 16-tile chains, down 16); shapes and tiles outside it do not" {
    try testing.expect(mma.chains(4096, 1024, ex3.glm_gateup)); // 256 k tiles / (4 warps x 4 splits) = 16
    try testing.expect(mma.chains(1024, 4096, ex3.glm_down)); // 64 / (4 x 1) = 16
    try testing.expect(mma.ready(layer, scratch(2048)));
    try testing.expect(!mma.chains(1152, 5120, ex3.glm_down)); // a TP2 DeepSeek-V4.1 rank's down: 18-tile chains
    try testing.expect(!mma.chains(4096, 1000, ex3.glm_gateup)); // N not in 64-column blocks
    try testing.expect(!mma.chains(4096 + 8, 1024, ex3.glm_gateup));
    var odd = layer;
    odd.k2_gu = .{ 2, 8 }; // outside X1's scope (promptReady)
    try testing.expect(!mma.ready(odd, scratch(2048)));
}

test "the work list's length: python's _list_len at the half's shapes, and no routing has more (expert, 64-row block) pairs" {
    try testing.expectEqual(@as(u32, (2048 * 9 + 288 * 63) / 64), mma.listLen(2048 * 9, 288, 64));
    try testing.expectEqual(@as(u32, 571), mma.listLen(2048 * 9, 288, 64));
    try testing.expectEqual(@as(u32, (64 * 9 + 288 * 63) / 64), mma.listLen(64 * 9, 288, 64));
    try testing.expectEqual(@as(u32, 2 * 571), mma.workInts(2048, 9, 288));
    var rng = std.Random.DefaultPrng.init(0x2b);
    const r = rng.random();
    for ([_]u32{ 64, 65, 128, 594, 1024, 1900, 2048 }) |R| for (0..4) |trial| {
        var c: [288]u32 = @splat(0);
        for (0..R) |_| { // each row: 8 distinct routed experts (skewed on odd trials) and the shared slot
            var used: [288]bool = @splat(false);
            var n: usize = 0;
            while (n < 8) {
                const e = if (trial % 2 == 1) r.uintLessThan(u32, 24) else r.uintLessThan(u32, 287);
                if (used[e]) continue;
                used[e] = true;
                c[e] += 1;
                n += 1;
            }
            c[287] += 1;
        }
        var pairs: u32 = 0;
        for (c) |x| pairs += (x + 63) / 64;
        try testing.expect(pairs <= mma.listLen(R * 9, @min(R * 9, 288), 64));
    };
}

// The member lists as group_kernel and PGROUP's group_place write them (ascending experts, ascending slots, code
// (i / slots) * 32 + i % slots, -1 to maxm = R), work_list_kernel's pairs, and grouped_mma's rows: every member slot
// of every used expert covered by exactly one program row, no program row past an expert's members.
test "the work list and grouped_mma's rows cover every member slot once, on balanced and skewed routing at 64-2,048 rows" {
    const E = 288;
    const slots = 9;
    var rng = std.Random.DefaultPrng.init(0x1b);
    const r = rng.random();
    for ([_]u32{ 64, 65, 128, 594, 1024, 1900, 2048 }) |R| for (0..2) |skew| {
        const P = R * slots;
        const pick = try testing.allocator.alloc(i32, P);
        defer testing.allocator.free(pick);
        for (0..R) |row| {
            var used: [E]bool = @splat(false);
            for (0..slots - 1) |s| {
                var e: u32 = undefined;
                while (true) {
                    e = if (skew == 1) r.uintLessThan(u32, 20) else r.uintLessThan(u32, E - 1);
                    if (!used[e]) break;
                }
                used[e] = true;
                pick[row * slots + s] = @intCast(e);
            }
            pick[row * slots + slots - 1] = E - 1; // the shared expert has every row
        }
        var cnt: [E]u32 = @splat(0);
        for (pick) |e| cnt[@intCast(e)] += 1;
        const maxu = @min(P, E);
        var uids: [E]u32 = undefined;
        var nu: u32 = 0;
        for (0..E) |e| if (cnt[e] > 0) {
            uids[nu] = @intCast(e);
            nu += 1;
        };
        const members = try testing.allocator.alloc(i32, maxu * R);
        defer testing.allocator.free(members);
        @memset(members, -1);
        for (0..nu) |u| {
            var j: usize = 0;
            for (pick, 0..) |e, i| if (e == uids[u]) {
                members[u * R + j] = @intCast((i / slots) * 32 + i % slots);
                j += 1;
            };
        }
        const n = mma.listLen(P, maxu, mma.rows); // work_list_kernel: place order, group order, then sentinels
        const list = try testing.allocator.alloc([2]u32, n);
        defer testing.allocator.free(list);
        var q: u32 = 0;
        for (0..nu) |u| for (0..(cnt[uids[u]] + 63) / 64) |j| {
            list[q] = .{ @intCast(u), @intCast(j) };
            q += 1;
        };
        try testing.expect(q <= n);
        for (q..n) |i| list[i] = .{ 0x7fffffff, 0 };
        const seen = try testing.allocator.alloc(u8, P);
        defer testing.allocator.free(seen);
        @memset(seen, 0);
        for (list) |pair| { // grouped_mma_kernel's head: u >= ucount returns; rows_sh from members, -1 past maxm
            const u = pair[0];
            if (u >= nu) continue;
            for (0..mma.rows) |i| {
                const m = pair[1] * mma.rows + i;
                const code: i32 = if (m < R) members[u * R + m] else -1;
                if (code < 0) continue;
                const p: usize = @as(usize, @intCast(code >> 5)) * slots + @as(usize, @intCast(code & 31));
                try testing.expectEqual(@as(i32, @intCast(uids[u])), pick[p]);
                seen[p] += 1;
            }
        }
        for (seen) |v| try testing.expectEqual(@as(u8, 1), v);
    };
}

test "MMA against X1, every selector, 64-2,048 rows: the work list first; rotation and epilogues X1's (gate/up's at SK 1); gate/up and down grouped_mma over the list" {
    for ([_]u32{ 64, 65, 128, 594, 1024, 1900, 2048 }) |R| for (1..16) |s| for ([_]?u64{ null, 0xf000 }) |shared| for ([_]?u64{ null, 0xb0000 }) |wts| {
        const sel: u32 = @intCast(s);
        if (sel == 1 and shared == null) continue; // rowrot alone without a shared suh: X1 is not in effect (routedX1 falls back)
        var x: k.Launches = .{};
        var q: k.Launches = .{};
        var m: k.Launches = .{};
        try routedX1(&x, R, wts, prompt(sel, false, false), shared);
        try routedX1(&q, R, wts, prompt(sel, true, false), shared); // SEQ: gate/up's epilogue at SK 1
        try routedMma(&m, R, wts, prompt(sel, false, true), shared);
        const xs = x.slice();
        const ms = m.slice();
        const P = R * 9;
        const n = mma.listLen(P, 288, 64);
        const xrow = sel & 1 != 0 and shared != null;
        const order = sel & 4 != 0 and !(sel & 8 != 0 and ex3.orderSlots(R, 9, 288) > 65535);
        const x0: usize = @as(usize, 1) + @intFromBool(order); // X1: group [, order], rotation, gate/up, epilogue, down, epilogue
        try testing.expectEqual(@as(usize, 6), ms.len);
        try testing.expectEqual(xs.len - x0 + 1, ms.len);
        try testing.expectEqual(k.Image.exl3_mma_cb1, ms[0].image); // the work list: one block of 1,024, n pairs
        try testing.expectEqualStrings(k.sym.work_list, ms[0].symbol);
        try testing.expectEqual(@as(u32, 1024), ms[0].cfg.block.x);
        try testing.expectEqual(@as(?u64, n), ms[0].args.integer(6));
        try expectSame(xs[x0], ms[1]); // rotation
        try testing.expectEqualStrings(k.sym.groupedMma(true, xrow), ms[2].symbol);
        try testing.expectEqual(@as(u32, 1), ms[2].cfg.grid.x);
        try testing.expectEqual(@as(u32, 1024 / 64), ms[2].cfg.grid.y);
        try testing.expectEqual(n * 2, ms[2].cfg.grid.z);
        try testing.expectEqual(@as(u32, 256), ms[2].cfg.block.x);
        try testing.expectEqual(@as(?u64, 4), ms[2].args.integer(13)); // SK: the window's 4 splits, folded in the program
        try testing.expectEqual(@as(?u64, 4), ms[2].args.integer(14)); // WK: the window's 4 warps
        try testing.expectEqual(@as(?u64, R), ms[2].args.integer(15)); // maxm: the member lists' stride R
        try testing.expectEqual(@as(?u64, work), ms[2].args.integer(18));
        try expectSame(q.slice()[q.len - 3], ms[3]); // gate/up's epilogue: SEQ's, at SK 1
        try testing.expectEqualStrings(k.sym.groupedMma(false, false), ms[4].symbol); // down: one split, written as is
        try testing.expectEqual(@as(u32, 4096 / 64), ms[4].cfg.grid.y);
        try testing.expectEqual(n, ms[4].cfg.grid.z);
        try testing.expectEqual(@as(?u64, 1), ms[4].args.integer(13));
        try expectSame(xs[xs.len - 1], ms[5]); // down's epilogue (or combine): X1's (its SK is GLM's 1)
    };
}

test "SEQ with MMA: gate/up SEQ's launch (X1's order list where X1 makes it), down grouped_mma, the rest MMA's" {
    for ([_]u32{ 64, 1024, 2048 }) |R| for (1..16) |s| {
        const sel: u32 = @intCast(s);
        if (sel == 1) continue;
        var q: k.Launches = .{};
        var m: k.Launches = .{};
        try routedX1(&q, R, null, prompt(sel, true, false), 0xf000);
        try routedMma(&m, R, null, prompt(sel, true, true), 0xf000);
        const qs = q.slice();
        const ms = m.slice();
        try testing.expectEqual(qs.len, ms.len); // the group launch's place holds the work list
        try testing.expectEqualStrings(k.sym.work_list, ms[0].symbol);
        for (1..qs.len) |i| {
            if (i == qs.len - 2) {
                try testing.expectEqualStrings(k.sym.groupedMma(false, false), ms[i].symbol);
                continue;
            }
            try expectSame(qs[i], ms[i]); // order, rotation, SEQ's gate/up, its epilogue, down's epilogue
        }
    };
}

test "routed refuses windows MMA does not take: under 64 rows, past the scratch, outside its shapes; grouped refuses what exl3/mma/ does not build" {
    var ls: k.Launches = .{};
    try testing.expectError(error.Invalid, routedMma(&ls, 63, null, prompt(15, false, true), 0xf000));
    try testing.expectError(error.Invalid, mma.routed(&ls, 0xd000, 4096, true, 0xe000, null, layer, scratch(1024), 0xa0000, 2048, 7.0, .bf16, prompt(15, false, true), 0x90000, 0xf000, counts, work));
    var odd = layer;
    odd.k2_d = .{ 6, 8 };
    try testing.expectError(error.Invalid, mma.routed(&ls, 0xd000, 4096, true, 0xe000, null, odd, scratch(2048), 0xa0000, 1024, 7.0, .bf16, prompt(15, false, true), 0x90000, 0xf000, counts, work));
    try testing.expectError(error.Invalid, mma.routed(&ls, 0xd000, 4096, true, 0xe000, null, layer, scratch(2048), 0xa0000, 1024, 7.0, .bf16, prompt(15, false, true), 0x90000, 0xf000, 0, work)); // no counts
    const g: ex3.Grouped = .{ .x0 = 1, .x1 = 2, .tp0 = 3, .tp1 = 4, .k2_0 = 5, .k2_1 = 6, .uids = 7, .ucount = 8, .members = 9, .z = 10, .mats = 2, .K = 4096, .N = 1024, .P = 18432, .SK = 4, .slots = 9, .maxm = 2048, .nexp_max = 288, .cb = .mcg, .nt = 8, .warps = 4, .pf = 1, .lo = 8, .hi = 8 };
    ls = .{};
    try mma.grouped(&ls, g, true, true, work, 571);
    try testing.expectError(error.Invalid, mma.grouped(&ls, g, false, false, work, 571)); // 4 splits unfolded: one plane holds one
    try testing.expectError(error.Invalid, mma.grouped(&ls, g, true, true, work, 0));
    try testing.expectError(error.Invalid, mma.grouped(&ls, g, true, true, 0, 571));
    try testing.expectError(error.Invalid, mma.grouped(&ls, g, true, true, work, 32768)); // grid z past 65,535
    var h = g;
    h.cb = .@"3inst";
    try testing.expectError(error.Invalid, mma.grouped(&ls, h, true, false, work, 571));
    h = g;
    h.lo = 2;
    try testing.expectError(error.Invalid, mma.grouped(&ls, h, true, false, work, 571));
    h = g;
    h.K = 3584; // 224 k tiles: 14-tile chains, not in 4-tile steps
    try testing.expectError(error.Invalid, mma.grouped(&ls, h, true, false, work, 571));
    try testing.expectEqualStrings("", k.sym.groupedMma(false, true));
}

test "MMA's instances: the work list and three grouped_mma_kernel<1, 8, 8, FOLD, 4, XROW>, exactly the exl3_mma image's" {
    var names: std.ArrayList([:0]const u8) = .empty;
    defer names.deinit(testing.allocator);
    try k.all(.exl3_mma_cb1, &names, testing.allocator);
    try testing.expectEqual(@as(usize, 4), names.items.len);
    try testing.expectEqualStrings("_ZN15tf_exl3_experts16work_list_kernelEPKiS1_S1_iPiiiS2_ii", names.items[0]);
    try testing.expectEqualStrings("_ZN8tf_exl3x18grouped_mma_kernelILi1ELi8ELi8ELb1ELi4ELb1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiiiS7_", k.sym.groupedMma(true, true));
    try testing.expectEqualStrings("_ZN8tf_exl3x18grouped_mma_kernelILi1ELi8ELi8ELb1ELi4ELb0EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiiiS7_", k.sym.groupedMma(true, false));
    try testing.expectEqualStrings("_ZN8tf_exl3x18grouped_mma_kernelILi1ELi8ELi8ELb0ELi4ELb0EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiiiS7_", k.sym.groupedMma(false, false));
}

// grouped_kernel's chain of a k tile (X1: split * per_split + warp * per_warp) and grouped_mma's (the program's chains
// in order, per_range = KT / (SK WK) tiles each, steps of 4): the same chain for every k tile, no step across two.
test "the chains: every k tile in the same (split, warp) chain both ways at GLM's half shapes, and MMA's 4-tile steps never straddle a chain" {
    for ([_]struct { K: u32, t: ex3.Tile }{ .{ .K = 4096, .t = ex3.glm_gateup }, .{ .K = 1024, .t = ex3.glm_down } }) |c| {
        const KT = c.K / 16;
        const per_split = KT / c.t.sk;
        const per_warp = per_split / c.t.warps;
        const per_range = KT / (c.t.sk * c.t.warps);
        try testing.expectEqual(per_warp, per_range);
        try testing.expect(per_range % mma.kb == 0);
        for (0..KT) |kt| {
            const split = kt / per_split;
            const warp = (kt % per_split) / per_warp;
            const chain = kt / per_range;
            try testing.expectEqual(split * c.t.warps + warp, chain);
            try testing.expectEqual(chain, (kt - kt % mma.kb) / per_range); // the step's first tile is in the same chain
        }
    }
}

/// One output's fp32 chain both ways: X1's (each split's warp sum stored, the epilogue's sum from 0.f in split order)
/// and grouped_mma's (sacc = w0, then + w1, w2, w3; FOLD: tot = 0 + split sums in order; the epilogue's 0.f + tot at SK
/// 1; without FOLD, SK 1: tot = the split's sum).
fn sums(red: [4][4]f32, fold: bool) [2]u32 {
    const splits: usize = if (fold) 4 else 1;
    var x1: f32 = 0.0;
    for (red[0..splits]) |r| {
        var s = r[0];
        for (r[1..]) |w| s += w;
        x1 += s;
    }
    var tot: f32 = 0.0;
    for (red[0..splits]) |r| {
        var sacc: f32 = r[0];
        for (r[1..]) |w| sacc = sacc + w;
        tot = if (fold) tot + sacc else sacc;
    }
    const m: f32 = 0.0 + tot;
    return .{ @bitCast(x1), @bitCast(m) };
}

test "the bits: grouped_mma's sums equal X1's on 2^20 draws and the specials, folded (gate/up) and one split (down)" {
    var rng = std.Random.DefaultPrng.init(0x3a);
    const r = rng.random();
    const specials = [_]u32{ 0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x3f800000, 0xbf800000 };
    var n: usize = 0;
    while (n < 1 << 20) : (n += 1) {
        var red: [4][4]f32 = undefined;
        for (&red) |*row| for (row) |*v| {
            const u = r.int(u32);
            v.* = switch (u % 8) {
                0 => @bitCast(specials[(u >> 8) % specials.len]),
                1 => -0.0,
                2 => @bitCast(u & 0x80ffffff),
                else => @bitCast((u & 0x8fffffff) | 0x30000000),
            };
        };
        if (n % 3 == 0) red[1][0] = -red[0][0];
        for ([_]bool{ true, false }) |fold| {
            const c = sums(red, fold);
            const nan = (c[0] & 0x7fffffff) > 0x7f800000;
            if (nan) try testing.expect((c[1] & 0x7fffffff) > 0x7f800000) else try testing.expectEqual(c[0], c[1]);
        }
    }
}
