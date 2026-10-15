//! qualification_e4ad04cd: K4's launches against python 2.15's host code (cuda/exl3/experts_prompt.cu's exl3x_prompt_cuda and
//! prompt_launch; experts.py's _prompt_routed, prompt_items, prompt_kernel_ready), at GLM's shapes on a TP2 half.

const std = @import("std");
const testing = std.testing;
const k = @import("cuda_exl3_kernels.zig");
const ex3 = @import("cuda_exl3_experts.zig");
const k4 = @import("cuda_exl3_k4.zig");

const layer: ex3.Layer = .{ .gate_ptr = 0x1000, .up_ptr = 0x2000, .down_ptr = 0x3000, .gate_k2 = 0x4000, .up_k2 = 0x5000, .down_k2 = 0x6000, .suh_g = 0x7000, .suh_u = 0x8000, .svh_g = 0x9000, .svh_u = 0xa000, .suh_d = 0xb000, .svh_d = 0xc000, .count = 256, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
const scratch: ex3.Scratch = .{ .xg = 0x10000, .xu = 0x20000, .xd = 0x30000, .z = 0x40000, .y = 0x50000, .ids = 0x60000, .count = 0x70000, .members = 0x80000, .rows = 2048, .slots = 9, .experts = 256, .cfg_gu = ex3.glm_gateup, .cfg_d = ex3.glm_down };
const bufs: k4.Bufs = .{ .items = 0x90000, .counts = 0xa0000, .order = 0xb0000, .cap = k4.promptItems(2048, 9, 256, 64) };
const sh: ex3.Shared = .{ .optin = 101376 };

fn argBytes(l: k.Launch) []const u8 {
    return l.args.storage[0..l.args.used];
}

test "prompt_smem: python's constexpr at both passes, without and with shared rows" {
    try testing.expectEqual(@as(u32, 43008), k4.smem(64, 2));
    try testing.expectEqual(@as(u32, 33792), k4.smem(64, 1)); // the gate/up epilogue's half is the larger
    try testing.expectEqual(@as(u32, 73728), k4.smem(128, 2));
    try testing.expectEqual(@as(u32, 67584), k4.smem(128, 1));
    try testing.expectEqual(@as(u32, 8), k4.warps(64));
    try testing.expectEqual(@as(u32, 16), k4.warps(128));
}

test "prompt_items: an item a used expert and the shared one, plus one a pass of pairs past its first" {
    try testing.expectEqual(@as(u32, 545), k4.promptItems(2048, 9, 256, 64)); // 257 + 18,432 / 64
    try testing.expectEqual(@as(u32, 401), k4.promptItems(2048, 9, 256, 128));
    try testing.expectEqual(@as(u32, 9), k4.promptItems(1, 9, 256, 64));
    try testing.expectEqual(@as(u32, 63), k4.promptItems(7, 9, 256, 64));
    try testing.expectEqual(@as(u32, 266), k4.promptItems(64, 9, 256, 64));
    try testing.expectEqual(@as(u32, 545), bufs.cap);
}

test "prompt_kernel_ready: 4-bit mcg, D in 256 columns, the width in 128, 1-32 slots, 16-byte aligned trellises" {
    try testing.expect(k4.ready(layer, 9, true));
    try testing.expect(!k4.ready(layer, 9, false));
    try testing.expect(!k4.ready(layer, 33, true));
    try testing.expect(!k4.ready(layer, 0, true));
    var l = layer;
    l.dims = 4096 + 128;
    try testing.expect(!k4.ready(l, 9, true));
    l = layer;
    l.width = 1024 + 64;
    try testing.expect(!k4.ready(l, 9, true));
    l = layer;
    l.cb = .@"3inst";
    try testing.expect(!k4.ready(l, 9, true));
    l = layer;
    l.k2_d = .{ 6, 8 };
    try testing.expect(!k4.ready(l, 9, true));
}

test "prompt_switches: TF_GLM_EXL3_PASS unset or blank 64, 64 or 128 (stripped); ORDER and ROWROT on unless 0; anything else refused" {
    try testing.expectEqual(@as(u32, 64), try k4.Opts.parsePass(null));
    try testing.expectEqual(@as(u32, 64), try k4.Opts.parsePass(""));
    try testing.expectEqual(@as(u32, 64), try k4.Opts.parsePass("64"));
    try testing.expectEqual(@as(u32, 64), try k4.Opts.parsePass(" 64\n"));
    try testing.expectEqual(@as(u32, 64), try k4.Opts.parsePass("  "));
    try testing.expectEqual(@as(u32, 128), try k4.Opts.parsePass("128"));
    for ([_][]const u8{ "32", "256", "0", "1", "128x", "0x40", "+64", "64.0" }) |v| try testing.expectError(error.BadExl3Prompt, k4.Opts.parsePass(v));
    try testing.expectEqual(k4.Opts{}, try k4.Opts.parse(null, null, null));
    try testing.expectEqual(k4.Opts{ .pass = 128, .order = false, .xrow = true }, try k4.Opts.parse("128", "0", ""));
    try testing.expectEqual(k4.Opts{ .pass = 64, .order = true, .xrow = false }, try k4.Opts.parse(null, "1", "0"));
    for ([_][]const u8{ "2", "yes", "on", " 1", "01", "true", "-1" }) |v| {
        try testing.expectError(error.BadExl3Prompt, k4.Opts.parse(null, v, null));
        try testing.expectError(error.BadExl3Prompt, k4.Opts.parse(null, null, v));
    }
}

/// The routed launches for R rows at (pass, order, xrow), the layer with a shared suh or not.
fn routed(ls: *k.Launches, R: u32, o: k4.Opts, shared_suh: ?u64) !void {
    try k4.routed(ls, 0xd000, 4096, true, 0xe000, layer, scratch, R, 7.0, true, sh, o, bufs, shared_suh);
}

test "_prompt_routed's sequence: group, the rotation, the plan, her order, gate/up and down, each with python's grid, block, shared bytes and arguments" {
    for ([_]u32{ 1, 7, 64, 594, 2048 }) |R| for ([_]u32{ 64, 128 }) |pass| for ([_]bool{ true, false }) |ord| for ([_]?u64{ null, 0xf000 }) |suh| {
        var ls: k.Launches = .{};
        try routed(&ls, R, .{ .pass = pass, .order = ord, .xrow = true }, suh);
        const g = ls.slice();
        try testing.expectEqual(@as(usize, if (ord) 6 else 5), g.len);
        const cap = k4.promptItems(R, 9, 256, pass);
        const shx = suh != null;
        // group and the rotation: the routed path's own launches (ex3's)
        var want: k.Launches = .{};
        try ex3.group(&want, 0xe000, scratch.ids, scratch.count, scratch.members, R, 9, 256, R, sh);
        if (shx) try ex3.rotRows(&want, 0xd000, 4096, true, 0xf000, scratch.xg, R, 4096) else try ex3.rotIn(&want, 0xd000, 4096, true, 0xe000, layer.suh_g, layer.suh_u, scratch.xg, scratch.xu, R, 4096, 9, 256);
        for (want.slice(), g[0..2]) |a, b| {
            try testing.expectEqual(a.image, b.image);
            try testing.expectEqualStrings(a.symbol, b.symbol);
            try testing.expectEqual(a.cfg, b.cfg);
            try testing.expectEqual(a.max_dynamic, b.max_dynamic);
            try testing.expectEqualSlices(u8, argBytes(a), argBytes(b));
        }
        // the plan: <<<1, 1024>>>(uids, ucount, members, maxm = R, pass, cap, E, items, counts)
        try testing.expectEqualStrings(k.sym.k4_plan, g[2].symbol);
        try testing.expectEqual(k.Image.exl3_k4, g[2].image);
        try testing.expectEqual(@as(u32, 1024), g[2].cfg.block.x);
        try testing.expectEqual(@as(u32, 1), g[2].cfg.grid.x);
        try testing.expectEqual(@as(?u64, scratch.ids), g[2].args.integer(0));
        try testing.expectEqual(@as(?u64, R), g[2].args.integer(3));
        try testing.expectEqual(@as(?u64, pass), g[2].args.integer(4));
        try testing.expectEqual(@as(?u64, cap), g[2].args.integer(5));
        try testing.expectEqual(@as(?u64, bufs.counts), g[2].args.integer(8));
        var i: usize = 3;
        if (ord) { // her item order: <<<1, 1024>>>(items, counts, order, cap, E)
            try testing.expectEqualStrings(k.sym.k4_order, g[3].symbol);
            try testing.expectEqual(@as(?u64, bufs.order), g[3].args.integer(2));
            try testing.expectEqual(@as(?u64, cap), g[3].args.integer(3));
            i = 4;
        }
        const gu = g[i];
        const sm_gu = k4.smem(pass, if (shx) 1 else 2);
        try testing.expectEqualStrings(k.sym.k4Gateup(pass, shx), gu.symbol);
        try testing.expectEqual([3]u32{ 1024 / 128, cap, 1 }, [3]u32{ gu.cfg.grid.x, gu.cfg.grid.y, gu.cfg.grid.z });
        try testing.expectEqual(pass / 8 * 32, gu.cfg.block.x);
        try testing.expectEqual(sm_gu, gu.cfg.shared);
        try testing.expectEqual(@as(?u32, sm_gu), gu.max_dynamic);
        try testing.expectEqual(@as(usize, 17), gu.args.count);
        try testing.expectEqual(@as(?u64, scratch.xg), gu.args.integer(0));
        try testing.expectEqual(@as(?u64, if (shx) scratch.xg else scratch.xu), gu.args.integer(1));
        try testing.expectEqual(@as(?u64, scratch.xd), gu.args.integer(10));
        try testing.expectEqual(@as(?u64, 4096), gu.args.integer(11));
        try testing.expectEqual(@as(?u64, 1024), gu.args.integer(12));
        try testing.expectEqual(@as(?u64, if (ord) bufs.order else 0), gu.args.integer(16));
        const dn = g[i + 1];
        try testing.expectEqualStrings(k.sym.k4Down(pass), dn.symbol);
        try testing.expectEqual([3]u32{ 4096 / 256, cap, 1 }, [3]u32{ dn.cfg.grid.x, dn.cfg.grid.y, dn.cfg.grid.z });
        try testing.expectEqual(k4.smem(pass, 2), dn.cfg.shared);
        try testing.expectEqual(@as(usize, 12), dn.args.count);
        try testing.expectEqual(@as(?u64, scratch.xd), dn.args.integer(0));
        try testing.expectEqual(@as(?u64, layer.down_ptr), dn.args.integer(1));
        try testing.expectEqual(@as(?u64, scratch.y), dn.args.integer(6));
        try testing.expectEqual(@as(?u64, 1024), dn.args.integer(7)); // K = the expert width
        try testing.expectEqual(@as(?u64, 4096), dn.args.integer(8));
        try testing.expectEqual(@as(?u64, if (ord) bufs.order else 0), dn.args.integer(11));
    };
}

test "rowrot without a shared suh takes rot_in and the gate/up kernel without SHX, as python's shx does" {
    var ls: k.Launches = .{};
    try routed(&ls, 64, .{ .xrow = true }, null);
    try testing.expectEqualStrings(k.sym.rot_in_bf16, ls.slice()[1].symbol);
    try testing.expectEqualStrings(k.sym.k4Gateup(64, false), ls.slice()[4].symbol);
    ls = .{};
    try routed(&ls, 64, .{ .xrow = false }, 0xf000);
    try testing.expectEqualStrings(k.sym.rot_in_bf16, ls.slice()[1].symbol);
    try testing.expectEqualStrings(k.sym.k4Gateup(64, false), ls.slice()[4].symbol);
    ls = .{};
    try routed(&ls, 64, .{ .xrow = true }, 0xf000);
    try testing.expectEqualStrings(k.sym.x1_rot_rows_bf16, ls.slice()[1].symbol);
    try testing.expectEqualStrings(k.sym.k4Gateup(64, true), ls.slice()[4].symbol);
}

test "the refusals: a pass of neither 64 nor 128, rows past the scratch, buffers short of the window's items, shared rows with two inputs" {
    var ls: k.Launches = .{};
    try testing.expectError(error.Invalid, routed(&ls, 64, .{ .pass = 96 }, null));
    ls = .{};
    try testing.expectError(error.Invalid, routed(&ls, 2049, .{}, null));
    ls = .{};
    try testing.expectError(error.Invalid, routed(&ls, 0, .{}, null));
    ls = .{};
    var small = bufs;
    small.cap = k4.promptItems(2048, 9, 256, 64) - 1;
    try testing.expectError(error.Invalid, k4.routed(&ls, 0xd000, 4096, true, 0xe000, layer, scratch, 2048, 7.0, true, sh, .{}, small, null));
    ls = .{};
    try testing.expectError(error.Invalid, k4.gateup(&ls, .{ .xg = 1, .xu = 2, .tg = 3, .tu = 4, .items = 5, .counts = 6, .members = 7, .svh_g = 8, .svh_u = 9, .suh_d = 10, .xd = 11, .D = 4096, .N = 1024, .E = 256, .limit = 7.0, .slots = 9, .ord = 0, .cap = 9, .pass = 64, .shx = true }));
    try testing.expectError(error.Invalid, k4.down(&ls, .{ .xd = 1, .td = 2, .items = 3, .counts = 4, .members = 5, .svh_d = 6, .y = 7, .N = 1024, .D = 4096 + 128, .E = 256, .slots = 9, .ord = 0, .cap = 9, .pass = 64 }));
}
