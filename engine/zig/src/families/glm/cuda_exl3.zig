//! TensorFold's EXL3 on CUDA for GLM-5.3-Flash: the kernels (copies of python-0.6's cuda/exl3 module and GLM's exl3.cu,
//! built as the exl3_* and glm_exl3 fatbins) with each wrapper's launch, and the reading of Mia's EXL3 checkpoint.
//!
//! The EXL3 format and its kernels follow ExLlamaV3 by turboderp (github.com/turboderp-org/exllamav3, MIT); TensorFold's
//! EXL3 module is by Ash Hart (github.com/ashhart) and the TensorFold contributors (python-0.6, ed78d6f).

const std = @import("std");

pub const kernels = @import("cuda_exl3_kernels.zig");
pub const experts = @import("cuda_exl3_experts.zig");
pub const linear = @import("cuda_exl3_linear.zig");
pub const glm = @import("cuda_exl3_glm.zig");
pub const split = @import("cuda_exl3_split.zig");
pub const check = @import("cuda_exl3_check.zig");
/// qualification_3e1f1105: TF_GLM_EXL3_PGROUP's CPU side (bertholomus's parallel grouping): the placement rule and the launches run on host memory
pub const pgroup = @import("cuda_exl3_pgroup.zig");
/// K4 (qualification_e4ad04cd, TF_GLM_HALVES_KERNEL=k4): MiaAI-Lab's EXL3 prompt kernel for prompt windows, python 2.15's port
pub const k4 = @import("cuda_exl3_k4.zig");
/// tf-exl3-check's K4 table (qualification_e4ad04cd)
pub const check_k4 = @import("cuda_exl3_check_k4.zig");

const Launch = kernels.Launch;
const Launches = kernels.Launches;

test {
    _ = kernels;
    _ = experts;
    _ = linear;
    _ = glm;
    _ = split;
    _ = check;
    _ = pgroup;
    _ = k4;
    _ = check_k4;
    _ = @import("cuda_exl3_test.zig");
    _ = @import("cuda_exl3_split_test.zig");
}

const Table = struct {
    arena: std.heap.ArenaAllocator,
    by_image: std.json.ObjectMap,

    fn load() !Table {
        var t: Table = .{ .arena = .init(std.testing.allocator), .by_image = undefined };
        const v = try std.json.parseFromSliceLeaky(std.json.Value, t.arena.allocator(), @embedFile("exl3_kernels.json"), .{});
        t.by_image = v.object;
        return t;
    }

    fn deinit(t: *Table) void {
        t.arena.deinit();
    }

    /// The cubins' entry for `symbol` in `image`: its parameters (offset, size) and static shared bytes.
    fn entry(t: *const Table, image: kernels.Image, symbol: []const u8) ?std.json.ObjectMap {
        for (t.by_image.get(@tagName(image)).?.array.items) |e| {
            if (std.mem.eql(u8, e.object.get("symbol").?.string, symbol)) return e.object;
        }
        return null;
    }
};

/// One launch of every kernel the bindings can pick, at shapes each wrapper accepts.
fn everyLaunch(out: *std.ArrayList(Launch), gpa: std.mem.Allocator) !void {
    var ls: Launches = .{};
    const sh: experts.Shared = .{ .optin = 101376 };
    try experts.group(&ls, 1, 2, 3, 4, 1536, 9, 288, 1536, sh); // past 48 KiB: the attribute is raised first
    try experts.rotIn(&ls, 1, 4096, true, 2, 3, 4, 5, 6, 1, 4096, 9, 288);
    try experts.rotIn(&ls, 1, 4096, false, 2, 3, 4, 5, 6, 1, 4096, 9, 288);
    try experts.gateupEpilogue(&ls, 1, 2, 3, 4, 5, 6, 1, 9, 1024, 4, 9, 288, 7.0, .bf16);
    try experts.downEpilogue(&ls, 1, 2, 3, 4, 1, 9, 4096, 1, 9, 288);
    experts.combine(&ls, 1, 2, 3, 1, 4096, 9);
    try experts.downCombine(&ls, 1, 2, 3, 4, 5, 6, 1, 9, 4096, 1, 9, 288);
    try experts.gateupEpilogueSched(&ls, 1, 2, 3, 4, 5, 6, 1, 9, 1024, 4, 9, 288, 7.0, .bf16, .{ .l2d = 4, .pdl = true }); // 0128's
    try out.appendSlice(gpa, ls.slice());
    ls = .{}; // qualification_3e1f1105: the parallel grouping, group_count then group_place
    try experts.groupPar(&ls, 1, 7, 2, 3, 4, 2048, 9, 288, 2048);
    try out.appendSlice(gpa, ls.slice());
    ls = .{}; // X1's own kernels (qualification_ae31a4aa): the items' order, one rotation a row, and the 15 switch instances
    experts.order(&ls, 1, 2, 3, 4, 2048, 1422, 288);
    try experts.rotRows(&ls, 1, 4096, true, 2, 3, 2048, 4096);
    try experts.rotRows(&ls, 1, 4096, false, 2, 3, 2048, 4096);
    try out.appendSlice(gpa, ls.slice());
    for (1..16) |sel| {
        ls = .{};
        const p: experts.Prompt = .{ .nfirst = sel & 8 != 0, .pa = sel & 2 != 0, .xrow = sel & 1 != 0 };
        try experts.groupedX1(&ls, .{ .x0 = 1, .x1 = 2, .tp0 = 3, .tp1 = 4, .k2_0 = 5, .k2_1 = 6, .uids = 7, .ucount = 8, .members = 9, .z = 10, .mats = 2, .K = 4096, .N = 1024, .P = 18432, .SK = 4, .slots = 9, .maxm = 2048, .nexp_max = 288, .cb = .mcg, .nt = 8, .warps = 4, .pf = 1, .lo = 8, .hi = 8 }, p, if (sel & 4 != 0) 11 else 0, 1422);
        var q = p; // qualification_b3fba1da: SEQ's instance of the same selector
        q.seq = true;
        try experts.groupedX1(&ls, .{ .x0 = 1, .x1 = 2, .tp0 = 3, .tp1 = 4, .k2_0 = 5, .k2_1 = 6, .uids = 7, .ucount = 8, .members = 9, .z = 10, .mats = 2, .K = 4096, .N = 1024, .P = 18432, .SK = 4, .slots = 9, .maxm = 2048, .nexp_max = 288, .cb = .mcg, .nt = 8, .warps = 4, .pf = 1, .lo = 8, .hi = 8 }, q, if (sel & 4 != 0) 11 else 0, 1422);
        try out.appendSlice(gpa, ls.slice());
    }
    ls = .{}; // qualification_8724e161: MMA's work list and grouped_mma's three instances (gate/up folded, with and without rowrot; down)
    try experts.mma.workList(&ls, 12, 7, 8, 288, 13, 571);
    const gm: experts.Grouped = .{ .x0 = 1, .x1 = 2, .tp0 = 3, .tp1 = 4, .k2_0 = 5, .k2_1 = 6, .uids = 7, .ucount = 8, .members = 9, .z = 10, .mats = 2, .K = 4096, .N = 1024, .P = 18432, .SK = 4, .slots = 9, .maxm = 2048, .nexp_max = 288, .cb = .mcg, .nt = 8, .warps = 4, .pf = 1, .lo = 8, .hi = 8 };
    try experts.mma.grouped(&ls, gm, true, false, 13, 571);
    try experts.mma.grouped(&ls, gm, true, true, 13, 571);
    var gd = gm;
    gd.mats = 1;
    gd.K = 1024;
    gd.N = 4096;
    gd.SK = 1;
    try experts.mma.grouped(&ls, gd, false, false, 13, 571);
    try out.appendSlice(gpa, ls.slice());
    ls = .{}; // K4's own kernels (qualification_e4ad04cd): the plan, her order, gate/up at both passes with and without SHX, down at both
    k4.plan(&ls, 1, 2, 3, 2048, 64, 545, 256, 4, 5);
    k4.order(&ls, 4, 5, 6, 545, 256);
    try out.appendSlice(gpa, ls.slice());
    for (k4.passes) |pass| {
        ls = .{};
        for ([_]bool{ false, true }) |shx| try k4.gateup(&ls, .{ .xg = 1, .xu = if (shx) 1 else 2, .tg = 3, .tu = 4, .items = 5, .counts = 6, .members = 7, .svh_g = 8, .svh_u = 9, .suh_d = 10, .xd = 11, .D = 4096, .N = 1024, .E = 256, .limit = 7.0, .slots = 9, .ord = 12, .cap = 545, .pass = pass, .shx = shx });
        try k4.down(&ls, .{ .xd = 1, .td = 2, .items = 3, .counts = 4, .members = 5, .svh_d = 6, .y = 7, .N = 1024, .D = 4096, .E = 256, .slots = 9, .ord = 8, .cap = 545, .pass = pass });
        try out.appendSlice(gpa, ls.slice());
    }
    for ([_]kernels.Codebook{ .@"3inst", .mcg, .mul1 }) |cb| {
        for (kernels.grouped_tiles) |t| for (kernels.grouped_ranges) |r| {
            ls = .{};
            try experts.grouped(&ls, .{ .x0 = 1, .x1 = 2, .tp0 = 3, .tp1 = 4, .k2_0 = 5, .k2_1 = 6, .uids = 7, .ucount = 8, .members = 9, .z = 10, .mats = 2, .K = 4096, .N = 1024, .P = 9, .SK = 1, .slots = 9, .maxm = 1, .nexp_max = 9, .cb = cb, .nt = t[0], .warps = t[1], .pf = t[2], .lo = r[0], .hi = r[1] });
            try out.appendSlice(gpa, ls.slice());
        };
        for (kernels.grouped_tiles_l2d) |t| for (kernels.grouped_ranges) |r| { // 0128's, l2d last
            ls = .{};
            try experts.grouped(&ls, .{ .x0 = 1, .x1 = 2, .tp0 = 3, .tp1 = 4, .k2_0 = 5, .k2_1 = 6, .uids = 7, .ucount = 8, .members = 9, .z = 10, .mats = 2, .K = 4096, .N = 1024, .P = 9, .SK = 1, .slots = 9, .maxm = 1, .nexp_max = 9, .cb = cb, .nt = t[0], .warps = t[1], .pf = t[2], .lo = r[0], .hi = r[1], .sched = .{ .l2d = 4 } });
            try out.appendSlice(gpa, ls.slice());
        };
        for (2..17) |k2| {
            ls = .{};
            try experts.dequant(&ls, 1, 2, 4096, 1024, @intCast(k2), cb);
            try out.appendSlice(gpa, ls.slice());
        }
    }
    ls = .{};
    try linear.rotIn(&ls, 1, .bf16, 2, 3, 1, 4096);
    try out.appendSlice(gpa, ls.slice());
    for (kernels.linear_widths) |p| {
        for (kernels.linear_warps) |wk| {
            ls = .{};
            const l: linear.Layer = .{ .words = 1, .suh = 2, .svh = 3, .bias = null, .counters = 4, .k = 4096, .n = 1024, .k2 = p[0], .cb = @fromBackingInt(@intCast(p[1])), .split = .{ 1, wk } };
            try linear.linear(&ls, 5, l, 6, .bf16, null, 1);
            try linear.unpack(&ls, l, 7);
            try out.appendSlice(gpa, ls.slice());
        }
    }
    for (kernels.glm_tiles) |t| {
        ls = .{};
        try glm.grouped(&ls, 1, 2, 3, 4, .{ .items = 5, .counts = 6, .members = 7, .experts = 288 }, 8, 2, 4096, 1024, 9, 1, 9, t[0], t[1], 288);
        try out.appendSlice(gpa, ls.slice());
    }
    ls = .{};
    try glm.rotIn(&ls, 1, 4096, 2, 3, 4, 5, 6, 1, 4096, 9);
    try glm.gateupEpilogue(&ls, 1, 2, 3, 4, 5, 6, 1, 9, 1024, 4, 9, 7.0);
    try glm.downEpilogue(&ls, 1, 2, 3, 4, 1, 9, 4096, 1, 9);
    try out.appendSlice(gpa, ls.slice());
}

test "the bindings name exactly the kernels each fatbin holds" {
    var t = try Table.load();
    defer t.deinit();
    const a = t.arena.allocator();
    inline for (kernels.images_all ++ kernels.images_l2d ++ kernels.images_x1 ++ kernels.images_x1s ++ kernels.images_mma ++ kernels.images_k4) |img| {
        var ours: std.ArrayList([:0]const u8) = .empty;
        try kernels.all(img, &ours, a);
        const cubin = t.by_image.get(@tagName(img)).?.array.items;
        try std.testing.expectEqual(cubin.len, ours.items.len);
        for (ours.items) |s| if (t.entry(img, s) == null) {
            std.debug.print("{t}: {s} is not in the cubins\n", .{ img, s });
            return error.UnknownSymbol;
        };
    }
}

test "every launch's arguments sit where the cubins' parameter tables put them, and every kernel is launched" {
    var t = try Table.load();
    defer t.deinit();
    const a = t.arena.allocator();
    var ls: std.ArrayList(Launch) = .empty;
    try everyLaunch(&ls, a);
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    for (ls.items) |l| {
        const e = t.entry(l.image, l.symbol) orelse {
            std.debug.print("{t}: {s} is not in the cubins\n", .{ l.image, l.symbol });
            return error.UnknownSymbol;
        };
        const ps = e.get("params").?.array.items;
        try std.testing.expectEqual(ps.len, l.args.count);
        for (ps, 0..) |p, i| {
            try std.testing.expectEqual(p.array.items[0].integer, l.args.offsets[i]);
            try std.testing.expectEqual(p.array.items[1].integer, l.args.sizes[i]);
        }
        try seen.put(a, try std.fmt.allocPrint(a, "{t}/{s}", .{ l.image, l.symbol }), {});
    }
    var total: usize = 0;
    var it = t.by_image.iterator();
    while (it.next()) |img| for (img.value_ptr.array.items) |e| {
        // exl3/l2d/'s images: only 0128's own kernels launch from them (the others are today's SASS: sass-check.sh)
        const s = e.object.get("symbol").?.string;
        if (std.mem.startsWith(u8, img.key_ptr.*, "exl3_l2d_") and !std.mem.startsWith(u8, s, "_ZN8tf_exl3x14grouped_kernel") and !std.mem.eql(u8, s, kernels.sym.gateup_epilogue)) continue;
        // exl3/x1/'s images: only X1's own kernels launch from them (its stock kernels launch from today's images)
        const x1_own = std.mem.startsWith(u8, s, "_ZN8tf_exl3x14grouped_kernel") or std.mem.eql(u8, s, kernels.sym.x1_order) or std.mem.eql(u8, s, kernels.sym.x1_rot_rows_bf16) or std.mem.eql(u8, s, kernels.sym.x1_rot_rows_f16);
        if (std.mem.startsWith(u8, img.key_ptr.*, "exl3_x1_") and !x1_own) continue;
        total += 1;
    };
    try std.testing.expectEqual(total, seen.count());
    const g = t.entry(.exl3_experts, kernels.sym.group).?;
    try std.testing.expectEqual(@as(i64, experts.group_static), g.get("shared").?.integer);
    try std.testing.expectEqual(@as(?u32, 101376 - experts.group_static), ls.items[0].max_dynamic);
}
