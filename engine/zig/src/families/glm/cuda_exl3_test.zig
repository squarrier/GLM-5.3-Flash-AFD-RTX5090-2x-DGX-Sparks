//! The EXL3 bindings against python-0.6's own wrappers: fixtures_cuda_exl3.json (tools/zig/exl3_fixtures.py) holds each
//! case's inputs and the launches python-0.6's Python and C++ wrappers made for each step, recorded on a CPU. Every step
//! is replayed through the Zig bindings and compared launch by launch: kernel, grid, block, shared bytes, the raised
//! shared-memory limit and each argument's bytes.
const std = @import("std");
const k = @import("cuda_exl3_kernels.zig");
const ex = @import("cuda_exl3_experts.zig");
const lin = @import("cuda_exl3_linear.zig");
const gm = @import("cuda_exl3_glm.zig");

const Obj = std.json.ObjectMap;

fn num(o: Obj, name: []const u8) u64 {
    return @intCast(o.get(name).?.integer);
}

fn n32(o: Obj, name: []const u8) u32 {
    return @intCast(o.get(name).?.integer);
}

fn opt(o: Obj, name: []const u8) ?u64 {
    const v = o.get(name) orelse return null;
    return if (v == .null) null else @intCast(v.integer);
}

fn pair(o: Obj, name: []const u8) [2]u32 {
    const a = o.get(name).?.array.items;
    return .{ @intCast(a[0].integer), @intCast(a[1].integer) };
}

fn tile(o: Obj, name: []const u8) ex.Tile {
    const a = o.get(name).?.array.items;
    return .{ .nt = @intCast(a[0].integer), .warps = @intCast(a[1].integer), .sk = @intCast(a[2].integer), .pf = @intCast(a[3].integer) };
}

fn f32bits(o: Obj, name: []const u8) f32 {
    return @bitCast(@as(u32, @intCast(o.get(name).?.integer)));
}

/// torch dtype codes as the recorder writes them (Half 5, Float 6, BFloat16 8).
fn dtype(code: u64) lin.DType {
    return switch (code) {
        5 => .f16,
        6 => .f32,
        8 => .bf16,
        else => unreachable,
    };
}

fn expectLaunches(id: []const u8, step: usize, want: []const std.json.Value, got: []const k.Launch) !void {
    errdefer std.debug.print("case {s}, step {d}\n", .{ id, step });
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| {
        const o = w.object;
        try std.testing.expectEqualStrings(o.get("image").?.string, @tagName(g.image));
        try std.testing.expectEqualStrings(o.get("symbol").?.string, g.symbol);
        const grid = o.get("grid").?.array.items;
        const block = o.get("block").?.array.items;
        try std.testing.expectEqual([3]i64{ grid[0].integer, grid[1].integer, grid[2].integer }, [3]i64{ g.cfg.grid.x, g.cfg.grid.y, g.cfg.grid.z });
        try std.testing.expectEqual([3]i64{ block[0].integer, block[1].integer, block[2].integer }, [3]i64{ g.cfg.block.x, g.cfg.block.y, g.cfg.block.z });
        try std.testing.expectEqual(num(o, "shared"), g.cfg.shared);
        try std.testing.expectEqual(opt(o, "max_dynamic"), if (g.max_dynamic) |v| @as(u64, v) else null);
        try std.testing.expectEqual((opt(o, "pdl") orelse 0) == 1, g.cfg.pdl); // 0128's launch_ex attribute; python-0.6 sets none
        const ps = o.get("params").?.array.items;
        try std.testing.expectEqual(ps.len, g.args.count);
        for (ps, 0..) |p, i| {
            var hex: [32]u8 = undefined;
            const bytes = g.args.storage[g.args.offsets[i]..][0..g.args.sizes[i]];
            try std.testing.expectEqualStrings(p.string, std.fmt.bufPrint(&hex, "{x}", .{bytes}) catch unreachable);
        }
    }
}

fn routed(c: Obj, steps: []const std.json.Value, id: []const u8) !usize {
    const in = c.get("inputs").?.object;
    const l = in.get("layer").?.object;
    const s = in.get("scratch").?.object;
    const layer: ex.Layer = .{ .gate_ptr = num(l, "gate_ptr"), .up_ptr = num(l, "up_ptr"), .down_ptr = num(l, "down_ptr"), .gate_k2 = num(l, "gate_k2"), .up_k2 = num(l, "up_k2"), .down_k2 = num(l, "down_k2"), .suh_g = num(l, "suh_g"), .suh_u = num(l, "suh_u"), .svh_g = num(l, "svh_g"), .svh_u = num(l, "svh_u"), .suh_d = num(l, "suh_d"), .svh_d = num(l, "svh_d"), .count = n32(l, "count"), .dims = n32(l, "dims"), .width = n32(l, "width"), .cb = @fromBackingInt(@intCast(num(l, "cb"))), .k2_gu = pair(l, "k2_gu"), .k2_d = pair(l, "k2_d") };
    const scratch: ex.Scratch = .{ .xg = num(s, "xg"), .xu = num(s, "xu"), .xd = num(s, "xd"), .z = num(s, "z"), .y = num(s, "y"), .ids = num(s, "ids"), .count = num(s, "count"), .members = num(s, "members"), .rows = n32(s, "rows"), .slots = n32(s, "slots"), .experts = n32(s, "experts"), .cfg_gu = tile(s, "cfg_gu"), .cfg_d = tile(s, "cfg_d") };
    var sh: ex.Shared = .{ .optin = n32(in, "optin") }; // one process: a raised limit stays raised
    // 0128's fixtures: the oracle's TF_EXL3X_L2D/_PDL, always 0128's kernels (l2d 0 without PDL too: Sched.x)
    const sd: ex.Sched = if (in.get("sched")) |v| .{ .l2d = n32(v.object, "l2d"), .pdl = num(v.object, "pdl") == 1, .x = true } else .{};
    var n: usize = 0;
    for (steps, 0..) |st, i| {
        const a = st.object.get("args").?.object;
        var ls: k.Launches = .{};
        try ex.routedSched(&ls, num(in, "x"), n32(in, "x_stride"), in.get("x_bf16").?.bool, num(in, "pick"), opt(in, "wts"), layer, scratch, opt(in, "out") orelse 0, n32(a, "R"), f32bits(in, "limit"), @fromBackingInt(@intCast(num(in, "act"))), a.get("group").?.bool, sh, sd);
        try expectLaunches(id, i, st.object.get("launches").?.array.items, ls.slice());
        for (ls.slice()) |x| if (x.max_dynamic) |v| {
            sh.max_dynamic = v;
        };
        n += ls.len;
    }
    return n;
}

fn linear(c: Obj, steps: []const std.json.Value, id: []const u8) !usize {
    const in = c.get("inputs").?.object;
    var layer: lin.Layer = .{ .words = num(in, "words"), .suh = num(in, "suh"), .svh = num(in, "svh"), .bias = opt(in, "bias"), .counters = num(in, "counters"), .k = n32(in, "k"), .n = n32(in, "n"), .k2 = n32(in, "k2"), .cb = @fromBackingInt(@intCast(num(in, "cb"))), .layout = if (std.mem.eql(u8, in.get("layout").?.string, "stored")) .stored else .strips, .split = pair(in, "split") };
    var n: usize = 0;
    for (steps, 0..) |st, i| {
        const a = st.object.get("args").?.object;
        var ls: k.Launches = .{};
        if (opt(a, "unpack")) |w| {
            try lin.unpack(&ls, layer, w);
        } else {
            if (a.get("split") != null) layer.split = pair(a, "split");
            try lin.call(&ls, layer, num(a, "x"), dtype(num(a, "x_dtype")), n32(a, "M"), num(a, "y"), dtype(num(a, "y_dtype")), num(a, "xh"), num(a, "z"));
        }
        try expectLaunches(id, i, st.object.get("launches").?.array.items, ls.slice());
        n += ls.len;
    }
    return n;
}

fn glmRouted(c: Obj, steps: []const std.json.Value, id: []const u8) !usize {
    const in = c.get("inputs").?.object;
    const e = in.get("experts").?.object;
    const p = in.get("plan").?.object;
    const s = in.get("scratch").?.object;
    const experts: gm.Experts = .{ .gt = num(e, "gt"), .ut = num(e, "ut"), .dt = num(e, "dt"), .suh_g = num(e, "suh_g"), .suh_u = num(e, "suh_u"), .svh_g = num(e, "svh_g"), .svh_u = num(e, "svh_u"), .suh_d = num(e, "suh_d"), .svh_d = num(e, "svh_d"), .count = n32(e, "count"), .width = n32(e, "width"), .dims = n32(e, "dims") };
    const plan: gm.Plan = .{ .items = num(p, "items"), .counts = num(p, "counts"), .members = num(p, "members"), .experts = n32(p, "experts") };
    const scratch: gm.Scratch = .{ .xg = num(s, "xg"), .xu = num(s, "xu"), .xd = num(s, "xd"), .z = num(s, "z"), .rows = n32(s, "rows"), .slots = n32(s, "slots") };
    var n: usize = 0;
    for (steps, 0..) |st, i| {
        const a = st.object.get("args").?.object;
        var ls: k.Launches = .{};
        try gm.routed(&ls, num(in, "x"), n32(in, "x_stride"), num(in, "pick"), plan, experts, scratch, num(in, "y"), n32(a, "R"), f32bits(in, "limit"));
        try expectLaunches(id, i, st.object.get("launches").?.array.items, ls.slice());
        n += ls.len;
    }
    return n;
}

fn single(kind: []const u8, c: Obj, steps: []const std.json.Value, id: []const u8) !usize {
    const in = c.get("inputs").?.object;
    var ls: k.Launches = .{};
    if (std.mem.eql(u8, kind, "dequant")) {
        try ex.dequant(&ls, num(in, "trellis"), num(in, "out"), n32(in, "K"), n32(in, "N"), n32(in, "k2"), @fromBackingInt(@intCast(num(in, "cb"))));
    } else if (std.mem.eql(u8, kind, "combine")) {
        ex.combine(&ls, num(in, "y"), num(in, "wts"), num(in, "out"), n32(in, "rows"), n32(in, "D"), n32(in, "slots"));
    } else if (std.mem.eql(u8, kind, "glm_grouped")) {
        const plan: gm.Plan = .{ .items = num(in, "items"), .counts = num(in, "counts"), .members = num(in, "members"), .experts = 0 };
        try gm.grouped(&ls, num(in, "x0"), num(in, "x1"), num(in, "t0"), num(in, "t1"), plan, num(in, "z"), n32(in, "mats"), n32(in, "K"), n32(in, "N"), n32(in, "P"), n32(in, "SK"), n32(in, "max_items"), n32(in, "nt"), n32(in, "warps"), n32(in, "E"));
    } else return error.UnknownCase;
    try std.testing.expectEqual(@as(usize, 1), steps.len);
    try expectLaunches(id, 0, steps[0].object.get("launches").?.array.items, ls.slice());
    return ls.len;
}

test "every EXL3 launch python-0.6's wrappers made, replayed through the Zig bindings, comes out the same" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const fx = (try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), @embedFile("fixtures_cuda_exl3.json"), .{})).object;
    var total: usize = 0;
    var kinds: std.StringHashMapUnmanaged(usize) = .empty;
    for (fx.get("cases").?.array.items) |cv| {
        const c = cv.object;
        const id = c.get("id").?.string;
        const kind = c.get("kind").?.string;
        const steps = c.get("steps").?.array.items;
        const n = if (std.mem.eql(u8, kind, "routed"))
            try routed(c, steps, id)
        else if (std.mem.eql(u8, kind, "linear"))
            try linear(c, steps, id)
        else if (std.mem.eql(u8, kind, "glm_routed"))
            try glmRouted(c, steps, id)
        else
            try single(kind, c, steps, id);
        total += n;
        const e = try kinds.getOrPut(arena.allocator(), kind);
        e.value_ptr.* = (if (e.found_existing) e.value_ptr.* else 0) + 1;
    }
    try std.testing.expectEqual(num(fx, "launches"), total);
    try std.testing.expectEqual(@as(usize, 6), kinds.count()); // routed, dequant, combine, linear, glm_routed, glm_grouped
}

test "every expert launch python-0.6.6 + Mia's 0128 made at each l2d and PDL setting, replayed with the same switches, comes out the same" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const fx = (try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), @embedFile("fixtures_cuda_exl3_l2d.json"), .{})).object;
    var total: usize = 0;
    var pdl: usize = 0;
    for (fx.get("cases").?.array.items) |cv| {
        const c = cv.object;
        try std.testing.expectEqualStrings("routed", c.get("kind").?.string);
        try std.testing.expect(c.get("inputs").?.object.get("sched") != null);
        total += try routed(c, c.get("steps").?.array.items, c.get("id").?.string);
        for (c.get("steps").?.array.items) |st| for (st.object.get("launches").?.array.items) |l| {
            if ((opt(l.object, "pdl") orelse 0) == 1) pdl += 1;
        };
    }
    try std.testing.expectEqual(num(fx, "launches"), total);
    try std.testing.expect(pdl > 0 and pdl < total); // both settings are in it
}
