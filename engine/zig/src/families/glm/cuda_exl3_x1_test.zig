//! qualification_ae31a4aa's X1 in the bindings (python 2.15's prompt switches, cuda/exl3/experts.py routed's nfirst, prefetch, order and
//! rowrot; IMP2's adaptation of Mia's 0001, 0020 and 0009 onto the universal kernel): the switch, order_slots, and
//! routedX1's launches against routedSched's (switch off, under 64 rows, outside prompt_ready: byte for byte) and against
//! prompt_grouped_launch's formula (on).

const std = @import("std");
const ex3 = @import("cuda_exl3_experts.zig");
const k = @import("cuda_exl3_kernels.zig");
const testing = std.testing;

const layer: ex3.Layer = .{ .gate_ptr = 0x1000, .up_ptr = 0x2000, .down_ptr = 0x3000, .gate_k2 = 0x4000, .up_k2 = 0x5000, .down_k2 = 0x6000, .suh_g = 0x7000, .suh_u = 0x8000, .svh_g = 0x9000, .svh_u = 0xa000, .suh_d = 0xb000, .svh_d = 0xc000, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
const sh: ex3.Shared = .{ .optin = 101376 }; // GB10's and the 5090's opt-in shared bytes
const x_in: u64 = 0xd000;
const pick: u64 = 0xe000;
const ord_buf: u64 = 0x90000;
const suh: u64 = 0xf000;
const out_o: u64 = 0xa0000;

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

fn expectAllSame(a: []const k.Launch, b: []const k.Launch) !void {
    try testing.expectEqual(a.len, b.len);
    for (a, b) |x, y| try expectSame(x, y);
}

fn sched(ls: *k.Launches, l: ex3.Layer, s: ex3.Scratch, R: u32, wts: ?u64, sd: ex3.Sched) !void {
    try ex3.routedSched(ls, x_in, 4096, true, pick, wts, l, s, out_o, R, 7.0, .bf16, true, sh, sd);
}

fn x1(ls: *k.Launches, l: ex3.Layer, s: ex3.Scratch, R: u32, wts: ?u64, sd: ex3.Sched, p: ex3.Prompt, shared: ?u64) !void {
    try ex3.routedX1(ls, x_in, 4096, true, pick, wts, l, s, out_o, R, 7.0, .bf16, true, sh, sd, p, ord_buf, shared);
}

fn prompt(sel: u32) ex3.Prompt {
    return .{ .nfirst = sel & 8 != 0, .order = sel & 4 != 0, .pa = sel & 2 != 0, .xrow = sel & 1 != 0 };
}

test "TF_GLM_EXL3_X1: unset, empty or 0 reads nothing; 1 turns python's four on, each =0 off; anything but 0/1 refused" {
    const none = [4]?[]const u8{ null, null, null, null };
    for ([_]?[]const u8{ null, "", "0" }) |v| {
        try testing.expect(!(try ex3.Prompt.parse(v, none)).on());
        try testing.expect(!(try ex3.Prompt.parse(v, .{ "yes", "2", "x", "1" })).on()); // not read while X1 is off
    }
    try testing.expectEqual(ex3.Prompt{ .nfirst = true, .pa = true, .order = true, .xrow = true }, try ex3.Prompt.parse("1", none));
    try testing.expectEqual(ex3.Prompt{ .nfirst = true, .pa = true, .order = true, .xrow = true }, try ex3.Prompt.parse("1", .{ "1", "", "1", "" }));
    try testing.expectEqual(ex3.Prompt{ .nfirst = false, .pa = true, .order = true, .xrow = true }, try ex3.Prompt.parse("1", .{ "0", null, null, null }));
    try testing.expectEqual(ex3.Prompt{ .nfirst = true, .pa = false, .order = true, .xrow = true }, try ex3.Prompt.parse("1", .{ null, "0", null, null }));
    try testing.expectEqual(ex3.Prompt{ .nfirst = true, .pa = true, .order = false, .xrow = true }, try ex3.Prompt.parse("1", .{ null, null, "0", null }));
    try testing.expectEqual(ex3.Prompt{ .nfirst = true, .pa = true, .order = true, .xrow = false }, try ex3.Prompt.parse("1", .{ null, null, null, "0" }));
    try testing.expect(!(try ex3.Prompt.parse("1", .{ "0", "0", "0", "0" })).on()); // all four off: today's launches
    for ([_][]const u8{ "2", "yes", " 1", "1 ", "true", "-1" }) |bad| {
        try testing.expectError(error.BadExl3X1, ex3.Prompt.parse(bad, none));
        for (0..4) |i| {
            var each = none;
            each[i] = bad;
            try testing.expectError(error.BadExl3X1, ex3.Prompt.parse("1", each));
        }
    }
}

test "order_slots: python's Scratch.order sizes (b.pref.order 1,422 at 2,048 rows, b.decode.order 72 at 8) and its bound" {
    try testing.expectEqual(@as(u32, 1422), ex3.orderSlots(2048, 9, 288)); // fixtures_cuda_afd_expert.json's buffers
    try testing.expectEqual(@as(u32, 72), ex3.orderSlots(8, 9, 288));
    try testing.expectEqual(@as(u32, 846), ex3.orderSlots(1024, 9, 288));
    try testing.expectEqual(@as(u32, 604), ex3.orderSlots(594, 9, 288));
    try testing.expectEqual(@as(u32, 306), ex3.orderSlots(64, 9, 288));
    try testing.expectEqual(@as(u32, 9), ex3.orderSlots(1, 9, 288));
    var R: u32 = 1;
    while (R <= 2048) : (R += 1) { // at least 1, at most every (expert, 16-member tile), never shrinking with R
        const c = ex3.orderSlots(R, 9, 288);
        const maxu: u32 = @min(R * 9, 288);
        try testing.expect(c >= 1 and c <= maxu * ((R + 15) / 16) and c <= (R * 9 + 15 * maxu) / 16);
        if (R > 1) try testing.expect(c >= ex3.orderSlots(R - 1, 9, 288));
    }
}

test "X1 off, on under 64 rows, or a layer outside prompt_ready: routedSched's launches byte for byte" {
    const rows = [_]u32{ 1, 2, 8, 9, 16, 63, 64, 65, 128, 594, 1024, 2047, 2048 };
    const scheds = [_]ex3.Sched{ .{}, .{ .pdl = true }, .{ .l2d = 4, .pdl = true }, .{ .x = true } };
    var l3 = layer; // a 3-bit layer: k2 6
    l3.k2_gu = .{ 6, 6 };
    l3.k2_d = .{ 6, 6 };
    var l2 = layer; // another codebook
    l2.cb = .mul1;
    for (rows) |R| for (scheds) |sd| for ([_]?u64{ null, 0xb0000 }) |wts| {
        var want: k.Launches = .{};
        var got: k.Launches = .{};
        try sched(&want, layer, scratch(2048), R, wts, sd);
        try x1(&got, layer, scratch(2048), R, wts, sd, .{}, suh);
        try expectAllSame(want.slice(), got.slice());
        for (1..16) |sel| {
            if (R >= ex3.prompt_rows) break;
            got = .{};
            try x1(&got, layer, scratch(2048), R, wts, sd, prompt(@intCast(sel)), suh);
            try expectAllSame(want.slice(), got.slice());
        }
        for ([_]ex3.Layer{ l3, l2 }) |l| {
            want = .{};
            got = .{};
            try sched(&want, l, scratch(2048), R, wts, sd);
            try x1(&got, l, scratch(2048), R, wts, sd, prompt(15), suh);
            try expectAllSame(want.slice(), got.slice());
        }
        var s2 = scratch(2048); // gate/up at 2 tiles in flight
        s2.cfg_gu.pf = 2;
        want = .{};
        got = .{};
        try sched(&want, layer, s2, R, wts, sd);
        try x1(&got, layer, s2, R, wts, sd, prompt(15), suh);
        try expectAllSame(want.slice(), got.slice());
    };
}

/// The launch python's prompt_grouped_launch<1> makes for `g` (today's `grouped` launch `stock`) with the switches.
fn expectPromptGrouped(stock: k.Launch, got: k.Launch, nfirst: bool, ord: bool, pa: bool, xrow: bool, cap: u32, mats: u32, sk: u32, N: u32, R: u32) !void {
    const sel: u32 = (if (nfirst) @as(u32, 8) else 0) | (if (ord) @as(u32, 4) else 0) | (if (pa) @as(u32, 2) else 0) | (if (xrow) @as(u32, 1) else 0);
    if (sel == 0) return expectSame(stock, got);
    try testing.expectEqual(k.Image.exl3_x1_grouped_cb1, got.image);
    try testing.expectEqualStrings(k.sym.groupedX1(sel), got.symbol);
    try testing.expectEqualStrings(k.sym.grouped(.mcg, 8, 4, 1, 8, 8), stock.symbol);
    const MT = (R + 15) / 16;
    const items: u32 = if (ord) cap else @min(R * 9, 288);
    const z = if (ord) mats * sk else mats * sk * MT;
    const nbs = N / 128;
    const grid = if (nfirst) [3]u32{ nbs, items, z } else [3]u32{ items, nbs, z };
    try testing.expectEqual(grid, [3]u32{ got.cfg.grid.x, got.cfg.grid.y, got.cfg.grid.z });
    try testing.expectEqual(@as(u32, 128), got.cfg.block.x);
    try testing.expectEqual(@as(u32, 0), got.cfg.shared);
    try testing.expect(!got.cfg.pdl and got.max_dynamic == null);
    // today's 16 arguments (x1 = xg with rowrot: gate and up read the one rotated row), then the order list
    try testing.expectEqual(stock.args.count + 1, got.args.count);
    for (0..stock.args.count) |i| {
        const want = if (i == 1 and xrow) stock.args.integer(0) else stock.args.integer(i);
        try testing.expectEqual(want, got.args.integer(i));
    }
    try testing.expectEqual(@as(?u64, if (ord) ord_buf else 0), got.args.integer(stock.args.count));
}

test "X1 on at 64+ rows: python's routed order (group, order, rot_rows, gate/up, epilogue, down, epilogue) and prompt_grouped_launch's grids" {
    for ([_]u32{ 64, 65, 128, 594, 1024, 2048 }) |R| for (1..16) |sel| for ([_]?u64{ suh, null }) |shared| for ([_]?u64{ null, 0xb0000 }) |wts| {
        const p = prompt(@intCast(sel));
        var off: k.Launches = .{};
        var on: k.Launches = .{};
        try sched(&off, layer, scratch(2048), R, wts, .{});
        try x1(&on, layer, scratch(2048), R, wts, .{ .l2d = 4, .pdl = true }, p, shared); // 0128's switches do not reach X1's windows
        const o = off.slice();
        const g = on.slice();
        const cap = ex3.orderSlots(R, 9, 288);
        const xrow = p.xrow and shared != null;
        if (!(p.nfirst or p.pa or p.order or xrow)) { // rowrot alone on a layer without a shared suh: no switch, 0128's
            var want: k.Launches = .{};
            try sched(&want, layer, scratch(2048), R, wts, .{ .l2d = 4, .pdl = true });
            try expectAllSame(want.slice(), g);
            continue;
        }
        try testing.expectEqual(o.len + @intFromBool(p.order), g.len);
        try expectSame(o[0], g[0]); // group
        var i: usize = 1;
        if (p.order) { // prompt_order_kernel<<<1, 1024>>>(ids, count, members, order, R, cap, E)
            try testing.expectEqual(k.Image.exl3_x1_experts, g[1].image);
            try testing.expectEqualStrings(k.sym.x1_order, g[1].symbol);
            try testing.expectEqual([6]u32{ 1, 1, 1, 1024, 1, 1 }, [6]u32{ g[1].cfg.grid.x, g[1].cfg.grid.y, g[1].cfg.grid.z, g[1].cfg.block.x, g[1].cfg.block.y, g[1].cfg.block.z });
            for ([_]u64{ 0x60000, 0x70000, 0x80000, ord_buf, R, cap, 288 }, 0..) |v, a| try testing.expectEqual(@as(?u64, v), g[1].args.integer(a));
            try testing.expectEqual(@as(usize, 7), g[1].args.count);
            i = 2;
        }
        if (xrow) { // rot_in_kernel<bf16, true><<<(R, D / 128, 1), 32>>>(x, stride, null, suh, suh, xg, xg, D, 1, 1)
            try testing.expectEqual(k.Image.exl3_x1_experts, g[i].image);
            try testing.expectEqualStrings(k.sym.x1_rot_rows_bf16, g[i].symbol);
            try testing.expectEqual([4]u32{ R, 32, 1, 32 }, [4]u32{ g[i].cfg.grid.x, g[i].cfg.grid.y, g[i].cfg.grid.z, g[i].cfg.block.x });
            for ([_]u64{ x_in, 4096, 0, suh, suh, 0x10000, 0x10000, 4096, 1, 1 }, 0..) |v, a| try testing.expectEqual(@as(?u64, v), g[i].args.integer(a));
        } else try expectSame(o[1], g[i]); // today's rot_in
        try expectPromptGrouped(o[2], g[i + 1], p.nfirst, p.order, p.pa, xrow, cap, 2, 4, 1024, R);
        try expectSame(o[3], g[i + 2]); // the gate/up epilogue: today's (not 0128's)
        try expectPromptGrouped(o[4], g[i + 3], p.nfirst, p.order, p.pa, false, cap, 1, 1, 4096, R); // down: no rowrot
        try expectSame(o[5], g[i + 4]); // down epilogue, or the fused one with the combine
    };
}

test "X1: NFIRST with order slots past 65,535 drops the order list (python's grid-y guard); without NFIRST it stays" {
    const R: u32 = 120_000; // cap 67,770; z without the list 2 x 4 x 7,500
    const cap = ex3.orderSlots(R, 9, 288);
    try testing.expectEqual(@as(u32, 67_770), cap);
    var ls: k.Launches = .{};
    try ex3.routedX1(&ls, x_in, 4096, true, pick, null, layer, scratch(R), out_o, R, 7.0, .bf16, false, sh, .{}, prompt(15), ord_buf, suh);
    try testing.expectEqual(@as(usize, 5), ls.len); // rot_rows, gate/up, epilogue, down, epilogue: no group, no order
    try testing.expectEqualStrings(k.sym.x1_rot_rows_bf16, ls.items[0].symbol);
    try testing.expectEqualStrings(k.sym.groupedX1(8 | 2 | 1), ls.items[1].symbol);
    try testing.expectEqual(@as(?u64, 0), ls.items[1].args.integer(16));
    try testing.expectEqual([3]u32{ 8, 288, 2 * 4 * 7500 }, [3]u32{ ls.items[1].cfg.grid.x, ls.items[1].cfg.grid.y, ls.items[1].cfg.grid.z });
    ls = .{};
    try ex3.routedX1(&ls, x_in, 4096, true, pick, null, layer, scratch(R), out_o, R, 7.0, .bf16, false, sh, .{}, prompt(4 | 2 | 1), ord_buf, suh);
    try testing.expectEqualStrings(k.sym.x1_order, ls.items[0].symbol);
    try testing.expectEqualStrings(k.sym.groupedX1(4 | 2 | 1), ls.items[2].symbol);
    try testing.expectEqual([3]u32{ cap, 8, 2 * 4 }, [3]u32{ ls.items[2].cfg.grid.x, ls.items[2].cfg.grid.y, ls.items[2].cfg.grid.z });
    // the order switch without a list to write: refused, never launched past it
    ls = .{};
    try testing.expectError(error.Invalid, ex3.routedX1(&ls, x_in, 4096, true, pick, null, layer, scratch(2048), out_o, 2048, 7.0, .bf16, true, sh, .{}, prompt(4), 0, suh));
}

test "X1's instances: the 15 switch selectors' symbols are the image's, in prompt_grouped_launch's order" {
    var names: std.ArrayList([:0]const u8) = .empty;
    defer names.deinit(testing.allocator);
    try k.all(.exl3_x1_grouped_cb1, &names, testing.allocator);
    try testing.expectEqual(@as(usize, 15), names.items.len);
    for (1..16) |sel| try testing.expectEqualStrings(names.items[sel - 1], k.sym.groupedX1(@intCast(sel)));
    try testing.expectEqualStrings("", k.sym.groupedX1(0));
    try testing.expectEqualStrings("", k.sym.groupedX1(16));
    // python's selector bits: NFIRST 8, ORDER 4, PA 2, XROW 1 -> grouped_kernel<1, 8, 4, 1, 8, 8, N, O, P, X>
    try testing.expect(std.mem.indexOf(u8, k.sym.groupedX1(8), "Lb1ELb0ELb0ELb0E") != null);
    try testing.expect(std.mem.indexOf(u8, k.sym.groupedX1(1), "Lb0ELb0ELb0ELb1E") != null);
    try testing.expect(std.mem.indexOf(u8, k.sym.groupedX1(6), "Lb0ELb1ELb1ELb0E") != null);
}
