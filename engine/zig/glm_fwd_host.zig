//! tf-glm-forward's host checks of the EXL3 MoE path (qualification_qualification_cec53599): cuda_moe.exl3Launches, the launches the tool runs through
//! fwd.Kernels.exl3, over a TP2 EXL3 rank's views and buffers (python's shapes: a decode window of 8 rows, a prompt
//! chunk of 2048) placed at qualification_765330ce's fixture addresses, equal python-0.6's own launches (fixtures_cuda_exl3.json's GLM
//! cases glm-tp2-decode and glm-tp2-prompt) step by step: kernel, grid, block, shared bytes and every argument's bytes.
const std = @import("std");
const glm = @import("glm_cuda");

/// qualification_765330ce's bindings and fixtures where the tree has them (qualification_qualification_44ba554e-int; not qualification_qualification_44ba554e-fwd-b, whose EXL3 ranks stop at the pack)
const has_exl3 = @hasDecl(glm, "exl3");
const fixture = if (has_exl3) @embedFile("src/families/glm/fixtures_cuda_exl3.json") else "";

const Obj = std.json.ObjectMap;

fn num(o: Obj, name: []const u8) u64 {
    return @intCast(o.get(name).?.integer);
}

/// The fixture's GLM cases this file replays: the tool's two window sizes.
const cases = [_]struct { id: []const u8, rows: usize, prefill: bool }{
    .{ .id = "glm-tp2-decode", .rows = 8, .prefill = false },
    .{ .id = "glm-tp2-prompt", .rows = 2048, .prefill = true },
};

fn expectLaunches(want: []const std.json.Value, got: anytype) !void {
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
        const raised: ?u64 = if (o.get("max_dynamic").? == .null) null else num(o, "max_dynamic");
        try std.testing.expectEqual(raised, if (g.max_dynamic) |v| @as(u64, v) else null);
        const ps = o.get("params").?.array.items;
        try std.testing.expectEqual(ps.len, g.args.count);
        for (ps, 0..) |p, i| {
            var hex: [32]u8 = undefined;
            const bytes = g.args.storage[g.args.offsets[i]..][0..g.args.sizes[i]];
            try std.testing.expectEqualStrings(p.string, try std.fmt.bufPrint(&hex, "{x}", .{bytes}));
        }
    }
}

/// One case: the rank's views and buffers at the case's addresses, then each step's R through exl3Launches.
fn replay(c: Obj, rows: usize, prefill: bool) !usize {
    const in = c.get("inputs").?.object;
    const l = in.get("layer").?.object;
    const s = in.get("scratch").?.object;
    const m = try glm.model.Model.from(&glm.config.Config{}, .{ .quant = .exl3 }, 2, 0);
    var e = glm.views.moe(&m).experts.exl3;
    var b = try glm.buffers.Buffers.init(&m, rows, 2560, prefill);
    // the forward's own facts first: python's rank shapes, as the fixture's case has them
    try std.testing.expectEqual(num(l, "count"), e.count);
    try std.testing.expectEqual(num(l, "dims"), e.dims);
    try std.testing.expectEqual(num(l, "width"), e.width);
    try std.testing.expectEqual(num(l, "cb"), e.cb);
    try std.testing.expectEqual(num(s, "rows"), b.rows);
    try std.testing.expectEqual(num(s, "slots"), b.pick.shape[1]);
    try std.testing.expect(in.get("x_bf16").?.bool and in.get("wts").? == .null and in.get("out").? == .null and num(in, "act") == 0);
    inline for (.{ "gate_ptr", "up_ptr", "down_ptr", "gate_k2", "up_k2", "down_k2", "suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d" }) |f|
        @field(e, f).ptr = num(l, f);
    const x3 = &b.exl3.?;
    inline for (.{ "xg", "xu", "xd", "z", "ids", "count" }) |f| @field(x3, f).ptr = num(s, f);
    x3.members_buf.ptr = num(s, "members");
    b.ey.ptr = num(s, "y");
    b.pick.ptr = num(in, "pick");
    b.normed.ptr = num(in, "x");
    const x = glm.trace.T.whole(b.normed);
    try std.testing.expectEqual(num(in, "x_stride"), x.stride[0]);
    // python's limit for these cases is the fixture's (+inf); the forward passes swiglu_limit, which G-trace checks
    const limit: f32 = @bitCast(@as(u32, @intCast(num(in, "limit"))));
    // one process: a raised shared-memory limit stays raised (the forward passes a fresh one a call, so it raises the
    // limit again where python would not: an attribute set, no launch differs)
    var sh: glm.exl3.experts.Shared = .{ .optin = @intCast(num(in, "optin")) };
    var n: usize = 0;
    for (c.get("steps").?.array.items) |st| {
        const a = st.object.get("args").?.object;
        try std.testing.expect(a.get("group").?.bool);
        var ls: glm.exl3.kernels.Launches = .{};
        try glm.moe.exl3Launches(&ls, x, e, &b, num(a, "R"), limit, sh);
        expectLaunches(st.object.get("launches").?.array.items, ls.slice()) catch |err| {
            std.debug.print("case {s}, R {d}: the forward's launches differ from python's\n", .{ c.get("id").?.string, num(a, "R") });
            return err;
        };
        for (ls.slice()) |one| if (one.max_dynamic) |v| {
            sh.max_dynamic = v;
        };
        n += ls.len;
    }
    return n;
}

test "the forward's EXL3 routed experts launch as python-0.6 does (qualification_765330ce's GLM cases, TP2 decode and prompt windows)" {
    if (comptime !has_exl3) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const fx = (try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), fixture, .{})).object;
    var launches: usize = 0;
    var steps: usize = 0;
    for (cases) |want| {
        const c = for (fx.get("cases").?.array.items) |c| {
            if (std.mem.eql(u8, c.object.get("id").?.string, want.id)) break c.object;
        } else return error.NoCase;
        launches += try replay(c, want.rows, want.prefill);
        steps += c.get("steps").?.array.items.len;
    }
    try std.testing.expectEqual(@as(usize, 20), steps);
    try std.testing.expectEqual(@as(usize, 120), launches);
}
