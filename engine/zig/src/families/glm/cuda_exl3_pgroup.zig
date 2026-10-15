//! qualification_3e1f1105: TF_GLM_EXL3_PGROUP's CPU side. bertholomus (Albert Lee, BertholomusAI) wrote the parallel grouping
//! (bertholomus/TensorFold glm-dsa-tp4 dbe0773, Apache-2.0); the one-block grouping it stands in for is Ash Hart's
//! TensorFold (python-0.6). `rule` is the placement both keep: the distinct experts below E in ascending id order
//! (uids, ucount), each one's members in ascending pick order as (i / slots) * 32 + i % slots, -1 after the last up to
//! maxm. `run` carries out one launch of group_kernel, group_count_kernel or group_place_kernel on host memory as its
//! kernel computes it, every argument read where the launch packed it, so the tests hold both launchers to the rule.

const std = @import("std");
const ex = @import("cuda_exl3_experts.zig");
const k = @import("cuda_exl3_kernels.zig");

fn enc(i: usize, slots: usize) i32 {
    return @intCast((i / slots) * 32 + i % slots);
}

/// The rule on `pick` (rows of `slots` ids): writes uids[0..ucount] and members[0..ucount * maxm]; returns ucount.
pub fn rule(pick: []const i32, slots: u32, E: u32, maxm: u32, uids: []i32, members: []i32) u32 {
    var u: usize = 0;
    for (0..E) |e| {
        var j: usize = 0;
        for (pick, 0..) |p, i| if (p == @as(i32, @intCast(e))) {
            if (j < maxm) members[u * maxm + j] = enc(i, slots);
            j += 1;
        };
        if (j == 0) continue;
        uids[u] = @intCast(e);
        while (j < maxm) : (j += 1) members[u * maxm + j] = -1;
        u += 1;
    }
    return @intCast(u);
}

/// Host memory standing in for device buffers: int32 words from each address.
pub const Mem = struct {
    at: []const u64,
    words: []const []i32,

    pub fn get(m: Mem, addr: u64, n: usize) error{Unmapped}![]i32 {
        for (m.at, m.words) |a, w| if (addr >= a and (addr - a) % 4 == 0 and (addr - a) / 4 + n <= w.len) return w[@intCast((addr - a) / 4)..][0..n];
        return error.Unmapped;
    }
};

fn arg(l: k.Launch, i: usize) error{Layout}!u64 {
    return l.args.integer(i) orelse error.Layout;
}

fn hits(pick: []const i32, e: usize) usize {
    var c: usize = 0;
    for (pick) |p| c += @intFromBool(p == @as(i32, @intCast(e)));
    return c;
}

/// One grouping launch on host memory: group_kernel (threads of 4 experts, below 4,096, placed by their scan),
/// group_count_kernel (a block an expert) or group_place_kernel (a block an expert: its place from the counts before
/// it, block 0 writes ucount, its members by in-order 256-wide scans, then -1). Any other launch: error.NotGrouping.
pub fn run(l: k.Launch, m: Mem) !void {
    const sym = l.symbol;
    if (std.mem.eql(u8, sym, k.sym.group)) {
        if (l.args.count != 8 or l.cfg.grid.x != 1 or l.cfg.block.x != 1024) return error.Layout;
        const R: usize = @intCast(try arg(l, 4));
        const slots: usize = @intCast(try arg(l, 5));
        const E: usize = @intCast(try arg(l, 6));
        const maxm: usize = @intCast(try arg(l, 7));
        if (l.cfg.shared != 4 * R * slots) return error.Layout;
        const pick = try m.get(try arg(l, 0), R * slots);
        var place: usize = 0;
        for (0..@min(E, 1024 * 4)) |e| {
            if (hits(pick, e) == 0) continue;
            (try m.get(try arg(l, 1) + 4 * place, 1))[0] = @intCast(e);
            const row = try m.get(try arg(l, 3) + 4 * place * maxm, maxm);
            var j: usize = 0;
            for (pick, 0..) |p, i| if (j < maxm and p == @as(i32, @intCast(e))) {
                row[j] = enc(i, slots);
                j += 1;
            };
            for (row[j..]) |*x| x.* = -1;
            place += 1;
        }
        (try m.get(try arg(l, 2), 1))[0] = @intCast(place);
        return;
    }
    const E: usize = l.cfg.grid.x;
    const count = std.mem.eql(u8, sym, k.sym.group_count);
    if (!count and !std.mem.eql(u8, sym, k.sym.group_place)) return error.NotGrouping;
    if (l.cfg.block.x != ex.place_threads or l.cfg.grid.y != 1 or l.cfg.grid.z != 1 or l.cfg.shared != 0) return error.Layout;
    if (count) {
        if (l.args.count != 3) return error.Layout;
        const pick = try m.get(try arg(l, 0), @intCast(try arg(l, 2)));
        const counts = try m.get(try arg(l, 1), E);
        for (counts, 0..) |*c, e| c.* = @intCast(hits(pick, e));
        return;
    }
    if (l.args.count != 9 or try arg(l, 7) != E) return error.Layout;
    const n: usize = @intCast(try arg(l, 5));
    const slots: usize = @intCast(try arg(l, 6));
    const maxm: usize = @intCast(try arg(l, 8));
    const pick = try m.get(try arg(l, 0), n);
    const counts = try m.get(try arg(l, 1), E);
    for (0..E) |e| {
        var before: usize = 0;
        for (counts[0..e]) |c| before += @intFromBool(c > 0);
        if (e == 0) {
            var used: usize = 0;
            for (counts) |c| used += @intFromBool(c > 0);
            (try m.get(try arg(l, 3), 1))[0] = @intCast(used);
        }
        if (counts[e] == 0) continue;
        (try m.get(try arg(l, 2) + 4 * before, 1))[0] = @intCast(e);
        const row = try m.get(try arg(l, 4) + 4 * before * maxm, maxm);
        var base: usize = 0;
        var start: usize = 0;
        while (start < n) : (start += ex.place_threads) {
            var prior: usize = 0; // hits of the threads before this one in the chunk: the kernel's scan
            for (0..ex.place_threads) |t| {
                const i = start + t;
                const hit = i < n and pick[i] == @as(i32, @intCast(e));
                if (hit and base + prior < maxm) row[base + prior] = enc(i, slots);
                prior += @intFromBool(hit);
            }
            base += prior;
        }
        if (base < maxm) {
            for (row[base..]) |*x| x.* = -1;
        }
    }
}

// ---- tests: the rule by hand, both launchers on host memory, the launch census, the refusals ----

const testing = std.testing;
const E_glm = 288;
const slots_glm = 9;

const layer: ex.Layer = .{ .gate_ptr = 0x1000, .up_ptr = 0x2000, .down_ptr = 0x3000, .gate_k2 = 0x4000, .up_k2 = 0x5000, .down_k2 = 0x6000, .suh_g = 0x7000, .suh_u = 0x8000, .svh_g = 0x9000, .svh_u = 0xa000, .suh_d = 0xb000, .svh_d = 0xc000, .count = E_glm, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
const at = struct {
    const pick: u64 = 1 << 32;
    const ids: u64 = 2 << 32;
    const count: u64 = 3 << 32;
    const members: u64 = 4 << 32;
    const counts: u64 = 5 << 32;
    const ord: u64 = 6 << 32;
    const suh: u64 = 7 << 32;
};

fn scratch(rows: u32, counts: u64) ex.Scratch {
    return .{ .xg = 0x10000, .xu = 0x20000, .xd = 0x30000, .z = 0x40000, .y = 0x50000, .ids = at.ids, .count = at.count, .members = at.members, .rows = rows, .slots = slots_glm, .experts = E_glm, .cfg_gu = ex.glm_gateup, .cfg_d = ex.glm_down, .counts = counts };
}

const all_x1: ex.Prompt = .{ .nfirst = true, .pa = true, .order = true, .xrow = true };

/// routedSched (X1 off, 0128 with PDL) or routedX1 (ZL_XENV's X1) over the stand-in scratch, PGROUP on (counts) or off.
fn launches(ls: *k.Launches, x1: bool, R: u32, rows: u32, pg: bool) !void {
    const s = scratch(rows, if (pg) at.counts else 0);
    const sh: ex.Shared = .{ .optin = 101376 };
    if (x1) return ex.routedX1(ls, 0xd000, 4096, true, at.pick, null, layer, s, 0, R, 7.0, .bf16, true, sh, .{ .pdl = true }, all_x1, at.ord, at.suh);
    return ex.routedSched(ls, 0xd000, 4096, true, at.pick, null, layer, s, 0, R, 7.0, .bf16, true, sh, .{ .pdl = true });
}

const Lists = struct {
    ids: []i32,
    count: []i32,
    members: []i32,
    counts: []i32,

    fn init(a: std.mem.Allocator, R: usize) !Lists {
        const maxu = @min(R * slots_glm, E_glm);
        const l: Lists = .{ .ids = try a.alloc(i32, maxu), .count = try a.alloc(i32, 1), .members = try a.alloc(i32, maxu * R), .counts = try a.alloc(i32, E_glm) };
        for ([_][]i32{ l.ids, l.count, l.members, l.counts }) |w| @memset(w, 0x5a5a5a5a); // untouched words must match too
        return l;
    }
};

/// The grouping launches of one routed call run on host memory: the lists they leave.
fn grouped(a: std.mem.Allocator, pick: []i32, x1: bool, R: u32, rows: u32, pg: bool) !Lists {
    var ls: k.Launches = .{};
    try launches(&ls, x1, R, rows, pg);
    const out = try Lists.init(a, R);
    const m: Mem = .{ .at = &.{ at.pick, at.ids, at.count, at.members, at.counts }, .words = &.{ pick, out.ids, out.count, out.members, out.counts } };
    var n: usize = 0;
    for (ls.slice()) |l| {
        run(l, m) catch |e| switch (e) {
            error.NotGrouping => continue,
            else => return e,
        };
        n += 1;
    }
    try testing.expectEqual(@as(usize, if (pg and R >= ex.prompt_rows) 2 else 1), n);
    return out;
}

pub const Routing = enum { uniform, hot, skewed };

/// python's routed cases' picks (8 distinct experts a row, then slot 8 = E), all rows the same 8 (routed-hot's), or
/// low ids far more often (a few rows with E - 1 first, one shared slot -1): R rows of 9 slots, experts below E.
pub fn synthetic(a: std.mem.Allocator, r: Routing, R: usize, E: i32, seed: u64) ![]i32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const p = try a.alloc(i32, R * slots_glm);
    for (0..R) |row| {
        const s = p[row * slots_glm ..][0..slots_glm];
        s[slots_glm - 1] = if (r == .skewed and row == 3) -1 else E;
        var j: usize = 0;
        while (j < slots_glm - 1) {
            const e: i32 = switch (r) {
                .uniform => rnd.intRangeLessThan(i32, 0, E),
                .hot => @intCast(17 + 31 * j),
                .skewed => if (j == 0 and row % 7 == 0) E - 1 else @intFromFloat(@floor(@as(f64, @floatFromInt(E)) * std.math.pow(f64, rnd.float(f64), 4))),
            };
            if (std.mem.indexOfScalar(i32, s[0..j], e) != null) continue;
            s[j] = e;
            j += 1;
        }
    }
    return p;
}

test "the placement rule by hand: ascending ids, members in pick order as row * 32 + slot, -1 to maxm, picks >= E or < 0 skipped" {
    const pick = [_]i32{ 4, 1, 5, 1, 2, -1, 4, 0, 1 }; // 3 rows of 3 slots, E 5: the 5 and the -1 are skipped
    var uids: [5]i32 = @splat(7);
    var members: [5 * 3]i32 = @splat(7);
    try testing.expectEqual(@as(u32, 4), rule(&pick, 3, 5, 3, &uids, &members));
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 4, 7 }, &uids);
    try testing.expectEqualSlices(i32, &.{ 65, -1, -1, 1, 32, 66, 33, -1, -1, 0, 64, -1, 7, 7, 7 }, &members);
    @memset(&members, 7); // maxm 1: the first member only, nothing padded
    try testing.expectEqual(@as(u32, 4), rule(&pick, 3, 5, 1, &uids, &members));
    try testing.expectEqualSlices(i32, &.{ 65, 1, 33, 0, 7 }, members[0..5]);
}

test "both launchers' grouping launches on host memory: groupPar's lists = group's = the rule's, word for word, R 64-2,049" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Routing{ .uniform, .hot, .skewed }) |r| for ([_]u32{ 64, 65, 512, 1024, 1900, 2048, 2049 }) |R| {
        const pick = try synthetic(a, r, R, E_glm, 0x2a + R);
        for ([_]bool{ false, true }) |x1| {
            if (x1 and R != 64 and R != 2049) continue; // the launchers share `grouping`: X1 at the ends
            const off = try grouped(a, pick, x1, R, 2049, false);
            const on = try grouped(a, pick, x1, R, 2049, true);
            const want = try Lists.init(a, R);
            want.count[0] = @intCast(rule(pick, slots_glm, E_glm, R, want.ids, want.members));
            for ([_]Lists{ off, on }) |got| {
                try testing.expectEqualSlices(i32, want.ids, got.ids);
                try testing.expectEqualSlices(i32, want.count, got.count);
                try testing.expectEqualSlices(i32, want.members, got.members);
            }
            for (on.counts, 0..) |c, e| try testing.expectEqual(@as(i32, @intCast(hits(pick, e))), c);
            if (r == .hot) try testing.expectEqual(@as(i32, 8), on.count[0]); // eight experts of R members each
        }
    };
}

test "the census: with counts set, only the grouping changes (group -> group_count + group_place at 64+ rows)" {
    for ([_]bool{ false, true }) |x1| for (1..2049) |r| {
        const R: u32 = @intCast(r);
        var off: k.Launches = .{};
        var on: k.Launches = .{};
        try launches(&off, x1, R, 2048, false);
        try launches(&on, x1, R, 2048, true);
        try testing.expectEqualStrings(k.sym.group, off.items[0].symbol);
        const par = R >= ex.prompt_rows;
        try testing.expectEqual(off.len + @intFromBool(par), on.len);
        if (par) {
            try testing.expectEqualStrings(k.sym.group_count, on.items[0].symbol);
            try testing.expectEqualStrings(k.sym.group_place, on.items[1].symbol);
            try testing.expectEqual(@as(?u64, R * slots_glm), on.items[0].args.integer(2));
            for ([_]u64{ at.pick, at.counts, at.ids, at.count, at.members, R * slots_glm, slots_glm, E_glm, R }, 0..) |v, i| try testing.expectEqual(@as(?u64, v), on.items[1].args.integer(i));
        }
        const skip: usize = @intFromBool(par);
        for (off.slice(), 0..) |p, i| {
            const q = on.items[i + if (i == 0) 0 else skip];
            if (i == 0 and par) continue;
            try testing.expectEqual(p.image, q.image);
            try testing.expectEqualStrings(p.symbol, q.symbol);
            try testing.expectEqual(p.cfg, q.cfg);
            try testing.expectEqual(p.max_dynamic, q.max_dynamic);
            try testing.expectEqualSlices(u8, p.args.storage[0..p.args.used], q.args.storage[0..q.args.used]);
        }
    };
}

test "the shared-memory wall: group refuses R 2,812 at 9 slots; groupPar takes R 4,096; its own refusals" {
    var ls: k.Launches = .{};
    try testing.expectError(error.Invalid, ex.group(&ls, 1, 2, 3, 4, 2812, 9, 288, 2812, .{ .optin = 101376 }));
    try ex.group(&ls, 1, 2, 3, 4, 2784, 9, 288, 2784, .{ .optin = 101376 });
    ls = .{};
    try launches(&ls, false, 4096, 4096, true);
    try testing.expectEqualStrings(k.sym.group_count, ls.items[0].symbol);
    try testing.expectError(error.Invalid, launches(&ls, false, 4096, 4096, false));
    ls = .{};
    try testing.expectError(error.Invalid, ex.groupPar(&ls, 1, 2, 3, 4, 5, 64, 33, 288, 64)); // more than 32 slots
    try testing.expectError(error.Invalid, ex.groupPar(&ls, 1, 2, 3, 4, 5, 64, 9, 0, 64));
    try testing.expectError(error.Invalid, ex.groupPar(&ls, 1, 0, 3, 4, 5, 64, 9, 288, 64)); // no counts buffer
    try testing.expectEqual(@as(usize, 0), ls.len);
}

test "TF_GLM_EXL3_PGROUP: unset, empty and 0 off, 1 on, with or without X1; anything else refused" {
    const x1 = try ex.Prompt.parse("1", .{ null, null, null, null });
    for ([_]ex.Prompt{ .{}, x1 }) |p| {
        for ([_]?[]const u8{ null, "", "0" }) |v| try testing.expectEqual(p, try ex.Prompt.parsePGroup(p, v));
        const on = try ex.Prompt.parsePGroup(p, "1");
        try testing.expect(on.pgroup);
        try testing.expectEqual(p.on(), on.on()); // X1's switches untouched
        for ([_][]const u8{ "2", "on", "yes", "true", " 1", "1 ", "01", "-1", "x" }) |v| try testing.expectError(error.BadExl3PGroup, ex.Prompt.parsePGroup(p, v));
    }
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "Prompt.fromEnv as the halves and tf-exl3-check read it once: PGROUP with X1 or without; X1's refusal first, then PGROUP's" {
    const names = [_][:0]const u8{ "TF_GLM_EXL3_X1", "TF_GLM_EXL3_NFIRST", "TF_GLM_EXL3_PF", "TF_GLM_EXL3_ORDER", "TF_GLM_EXL3_ROWROT", "TF_GLM_EXL3_PGROUP" };
    defer for (names) |n| {
        _ = unsetenv(n);
    };
    for (names) |n| _ = unsetenv(n);
    try testing.expectEqual(ex.Prompt{}, try ex.Prompt.fromEnv());
    _ = setenv("TF_GLM_EXL3_PGROUP", "1", 1);
    try testing.expectEqual(ex.Prompt{ .pgroup = true }, try ex.Prompt.fromEnv());
    _ = setenv("TF_GLM_EXL3_X1", "1", 1);
    try testing.expectEqual(ex.Prompt{ .nfirst = true, .pa = true, .order = true, .xrow = true, .pgroup = true }, try ex.Prompt.fromEnv());
    for ([_][:0]const u8{ "2", "on", "yes" }) |v| {
        _ = setenv("TF_GLM_EXL3_PGROUP", v, 1);
        try testing.expectError(error.BadExl3PGroup, ex.Prompt.fromEnv());
    }
    _ = setenv("TF_GLM_EXL3_X1", "2", 1);
    try testing.expectError(error.BadExl3X1, ex.Prompt.fromEnv());
    _ = setenv("TF_GLM_EXL3_X1", "0", 1);
    _ = setenv("TF_GLM_EXL3_PGROUP", "0", 1);
    try testing.expectEqual(ex.Prompt{}, try ex.Prompt.fromEnv());
}
