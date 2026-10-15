//! qualification_b3fba1da's SEQ in the bindings (Mia's TF_GLM_EXL3_SEQ design, MiaAI-Lab recipe patch 0004, Apache-2.0, on X1's grouped
//! kernel): the switch and its refusals, routedX1's launches with SEQ against X1's (only gate/up's grouped launch and its
//! epilogue's SK change), SEQ's instances, and the fp32 argument that keeps every output's bits.

const std = @import("std");
const ex3 = @import("cuda_exl3_experts.zig");
const k = @import("cuda_exl3_kernels.zig");
const testing = std.testing;

const layer: ex3.Layer = .{ .gate_ptr = 0x1000, .up_ptr = 0x2000, .down_ptr = 0x3000, .gate_k2 = 0x4000, .up_k2 = 0x5000, .down_k2 = 0x6000, .suh_g = 0x7000, .suh_u = 0x8000, .svh_g = 0x9000, .svh_u = 0xa000, .suh_d = 0xb000, .svh_d = 0xc000, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
const sh: ex3.Shared = .{ .optin = 101376 };

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

fn routed(ls: *k.Launches, l: ex3.Layer, s: ex3.Scratch, R: u32, wts: ?u64, p: ex3.Prompt, shared: ?u64) !void {
    try ex3.routedX1(ls, 0xd000, 4096, true, 0xe000, wts, l, s, 0xa0000, R, 7.0, .bf16, true, sh, .{}, p, 0x90000, shared);
}

fn prompt(sel: u32, seq: bool) ex3.Prompt {
    return .{ .nfirst = sel & 8 != 0, .order = sel & 4 != 0, .pa = sel & 2 != 0, .xrow = sel & 1 != 0, .seq = seq };
}

test "TF_GLM_EXL3_SEQ: unset, empty or 0 off; 1 on with X1 in effect; 1 without X1 (or with X1's four off) and any other value refused (BadExl3Seq)" {
    const x1 = try ex3.Prompt.parse("1", .{ null, null, null, null });
    for ([_]?[]const u8{ null, "", "0" }) |v| {
        try testing.expectEqual(x1, try ex3.Prompt.parseSeq(x1, v));
        try testing.expectEqual(ex3.Prompt{}, try ex3.Prompt.parseSeq(.{}, v)); // X1 off: SEQ unset or 0 is fine
    }
    var on = x1;
    on.seq = true;
    try testing.expectEqual(on, try ex3.Prompt.parseSeq(x1, "1"));
    try testing.expectError(error.BadExl3Seq, ex3.Prompt.parseSeq(.{}, "1")); // X1 unset
    try testing.expectError(error.BadExl3Seq, ex3.Prompt.parseSeq(try ex3.Prompt.parse("1", .{ "0", "0", "0", "0" }), "1")); // X1's four off
    for ([_][]const u8{ "2", "yes", "on", " 1", "01", "true", "-1" }) |bad| {
        try testing.expectError(error.BadExl3Seq, ex3.Prompt.parseSeq(x1, bad));
        try testing.expectError(error.BadExl3Seq, ex3.Prompt.parseSeq(.{}, bad));
    }
    try testing.expect(!on.seq or on.on()); // SEQ is a form of X1: never on alone
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "TF_GLM_EXL3_SEQ from the environment: SEQ=1 without TF_GLM_EXL3_X1=1 refuses the run; with it, X1 plus SEQ" {
    const names = [_][:0]const u8{ "TF_GLM_EXL3_X1", "TF_GLM_EXL3_NFIRST", "TF_GLM_EXL3_PF", "TF_GLM_EXL3_ORDER", "TF_GLM_EXL3_ROWROT", "TF_GLM_EXL3_SEQ" };
    defer for (names) |n| {
        _ = unsetenv(n);
    };
    for (names) |n| _ = unsetenv(n);
    try testing.expectEqual(ex3.Prompt{}, try ex3.Prompt.fromEnv());
    _ = setenv("TF_GLM_EXL3_SEQ", "1", 1);
    try testing.expectError(error.BadExl3Seq, ex3.Prompt.fromEnv());
    _ = setenv("TF_GLM_EXL3_X1", "1", 1);
    try testing.expectEqual(ex3.Prompt{ .nfirst = true, .pa = true, .order = true, .xrow = true, .seq = true }, try ex3.Prompt.fromEnv());
    _ = setenv("TF_GLM_EXL3_SEQ", "2", 1);
    try testing.expectError(error.BadExl3Seq, ex3.Prompt.fromEnv());
    _ = setenv("TF_GLM_EXL3_SEQ", "0", 1);
    try testing.expectEqual(ex3.Prompt{ .nfirst = true, .pa = true, .order = true, .xrow = true }, try ex3.Prompt.fromEnv());
}

test "SEQ against X1, every selector, 64-2,048 rows: only gate/up's grouped launch (exl3_x1s, z / SK, the same arguments) and its epilogue's SK change" {
    for ([_]u32{ 64, 65, 128, 594, 1024, 1900, 2048 }) |R| for (1..16) |s| for ([_]?u64{ null, 0xf000 }) |shared| for ([_]?u64{ null, 0xb0000 }) |wts| {
        const sel: u32 = @intCast(s);
        var a: k.Launches = .{};
        var b: k.Launches = .{};
        try routed(&a, layer, scratch(2048), R, wts, prompt(sel, false), shared);
        try routed(&b, layer, scratch(2048), R, wts, prompt(sel, true), shared);
        try testing.expectEqual(a.len, b.len);
        if (prompt(sel, false).xrow and shared == null and sel == 1) { // rowrot alone without a shared suh: X1 is not in effect
            for (a.slice(), b.slice()) |x, y| try expectSame(x, y);
            continue;
        }
        var gu: ?usize = null;
        for (a.slice(), b.slice(), 0..) |x, y, i| {
            if (gu == null and x.image == .exl3_x1_grouped_cb1) { // gate/up: SEQ's instance of the same selector
                gu = i;
                try testing.expectEqual(k.Image.exl3_x1s_grouped_cb1, y.image);
                var xsel: u32 = 0;
                for (1..16) |t| if (std.mem.eql(u8, x.symbol, k.sym.groupedX1(@intCast(t)))) {
                    xsel = @intCast(t);
                };
                try testing.expectEqualStrings(k.sym.groupedX1Seq(xsel), y.symbol);
                try testing.expectEqual(x.cfg.grid.x, y.cfg.grid.x);
                try testing.expectEqual(x.cfg.grid.y, y.cfg.grid.y);
                try testing.expectEqual(x.cfg.grid.z, y.cfg.grid.z * ex3.glm_gateup.sk);
                try testing.expectEqual(x.cfg.block, y.cfg.block);
                try testing.expectEqual(x.cfg.shared, y.cfg.shared);
                try testing.expectEqualSlices(u8, argBytes(x), argBytes(y)); // the kernel still takes SK: it loops the splits
                continue;
            }
            if (gu != null and i == gu.? + 1) { // its epilogue: the same launch at SK 1
                try testing.expectEqual(@as(?u64, ex3.glm_gateup.sk), x.args.integer(8));
                try testing.expectEqual(@as(?u64, 1), y.args.integer(8));
                var want: k.Launches = .{};
                try ex3.gateupEpilogue(&want, 0x40000, 0xe000, layer.svh_g, layer.svh_u, layer.suh_d, 0x30000, R, R * 9, 1024, 1, 9, 288, 7.0, .bf16);
                try expectSame(want.items[0], y);
                continue;
            }
            try expectSame(x, y); // group, order, rotation, down and its epilogue (or combine): X1's
        }
        try testing.expect(gu != null);
    };
}

test "SEQ under 64 rows or outside prompt_ready: X1's launches byte for byte (SEQ never runs there)" {
    for ([_]u32{ 1, 2, 8, 16, 63 }) |R| {
        var a: k.Launches = .{};
        var b: k.Launches = .{};
        try routed(&a, layer, scratch(2048), R, null, prompt(15, false), 0xf000);
        try routed(&b, layer, scratch(2048), R, null, prompt(15, true), 0xf000);
        try testing.expectEqual(a.len, b.len);
        for (a.slice(), b.slice()) |x, y| try expectSame(x, y);
    }
    var odd = layer;
    odd.k2_gu = .{ 2, 8 };
    var a: k.Launches = .{};
    var b: k.Launches = .{};
    try routed(&a, odd, scratch(2048), 1024, null, prompt(15, false), 0xf000);
    try routed(&b, odd, scratch(2048), 1024, null, prompt(15, true), 0xf000);
    for (a.slice(), b.slice()) |x, y| try expectSame(x, y);
}

test "SEQ's instances: 15 symbols, X1's in selector order with grouped_kernel renamed grouped_seq_kernel, exactly the exl3_x1s image's" {
    var names: std.ArrayList([:0]const u8) = .empty;
    defer names.deinit(testing.allocator);
    try k.all(.exl3_x1s_grouped_cb1, &names, testing.allocator);
    try testing.expectEqual(@as(usize, 15), names.items.len);
    for (1..16) |s| {
        const x = k.sym.groupedX1(@intCast(s));
        const y = k.sym.groupedX1Seq(@intCast(s));
        try testing.expectEqualStrings(names.items[s - 1], y);
        const at = std.mem.indexOf(u8, x, "14grouped_kernel").?;
        try testing.expectEqualStrings(x[0..at], y[0..at]);
        try testing.expectEqualStrings("18grouped_seq_kernel", y[at .. at + 20]);
        try testing.expectEqualStrings(x[at + 16 ..], y[at + 20 ..]);
    }
    try testing.expectEqualStrings("", k.sym.groupedX1Seq(0));
    try testing.expectEqualStrings("", k.sym.groupedX1Seq(16));
}

/// One output's fp32 chain both ways: today's (each split's warp sum stored, then the epilogue's sum from 0.f in split
/// order) and SEQ's (the same sums added from 0.f in the program, then the epilogue's 0.f + that at SK 1).
fn chains(red: [4][4]f32) [2]u32 {
    var planes: [4]f32 = undefined;
    for (red, 0..) |r, s| planes[s] = ((r[0] + r[1]) + r[2]) + r[3];
    var today: f32 = 0.0;
    for (planes) |p| today += p;
    var tot: f32 = 0.0;
    for (red) |r| tot += ((r[0] + r[1]) + r[2]) + r[3];
    const seq: f32 = 0.0 + tot;
    return .{ @bitCast(today), @bitCast(seq) };
}

test "the bits: an RN sum from +0 is never -0, so the epilogue's 0.f + total is total: SEQ's chain equals today's on 2^20 draws and the specials" {
    var rng = std.Random.DefaultPrng.init(0x5e9);
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
                2 => @bitCast(u & 0x80ffffff), // denormals and tiny normals of either sign
                else => @bitCast((u & 0x8fffffff) | 0x30000000), // ordinary magnitudes
            };
        };
        if (n % 3 == 0) red[1][0] = -red[0][0]; // exact cancellations to zero
        const c = chains(red);
        const nan = (c[0] & 0x7fffffff) > 0x7f800000;
        if (nan) try testing.expect((c[1] & 0x7fffffff) > 0x7f800000) else try testing.expectEqual(c[0], c[1]);
        try testing.expect(c[1] != 0x80000000); // never -0
    }
}
