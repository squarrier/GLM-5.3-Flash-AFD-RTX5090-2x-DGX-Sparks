//! shared_pool CPU tests (step test-shared_pool): TF_AFD_POOL_TOKENS's switch and plan, the pool's blocks, first fit, release and
//! compaction, the 262,144-token cap's bytes from the header shapes, the backend's reserve (a full pool refuses one
//! request, never all), release on finish, cancel and failure, the lane host's admission over recorded real users,
//! and recorded real CUDA launches over pooled regions: the same launches as today's own States relative to each
//! region's base, together = solo at any placement, and the compaction's copies. Recorder only: no CUDA numerics.
//! Fixture scaffolding follows Ash Hart's cuda_afd_engine_test.zig (through concurrent_users's afd_users_device_test.zig); the
//! pool follows Ash Hart's TensorFold v1.0.7 glm/pool.zig and Mia's GLM recipe patch 0074 (by ezoushen); NOTICE applies.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const glm = @import("glm_cuda");
const users = @import("glm_users");
const dev = @import("afd_users_device_test.zig");
const P = glm.afd_pool;
const schedule = glm.afd_users_round;
const Space = glm.trace.Space;
const Buf = glm.views.Buf;
const t = std.testing;
const gpa = t.allocator;
const eq = t.expectEqual;
const envcall = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

// ---- the switch, the blocks, placement ------------------------------------------------------------------------------

test "shared_pool switch: TF_AFD_POOL_TOKENS unset or 0 off, else a decimal multiple of 256; plan refuses in words" {
    try eq(@as(usize, 0), try P.parse(null));
    try eq(@as(usize, 0), try P.parse("0"));
    try eq(@as(usize, 781_568), try P.parse("781568"));
    try eq(@as(usize, 256), try P.parse("256"));
    for ([_][]const u8{ "", "00", "0781568", "781567", "+781568", " 781568", "781568 ", "7.8e5", "0x100", "-256", "12345678901" }) |v|
        try t.expectError(error.BadAfdPool, P.parse(v));
    const names = [_][*:0]const u8{ "TF_AFD_USERS", "TF_AFD_POOL_TOKENS" };
    var old: [2]?[:0]u8 = .{ null, null };
    for (names, &old) |n, *o| o.* = if (std.c.getenv(n)) |v| try gpa.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        for (names, old) |n, o| {
            if (o) |v| {
                _ = envcall.setenv(n, v, 1);
                gpa.free(v);
            } else _ = envcall.unsetenv(n);
        }
    }
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const flags = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--no-drafts", "--drafts", "0", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    _ = envcall.unsetenv("TF_AFD_POOL_TOKENS");
    _ = envcall.setenv("TF_AFD_USERS", "4", 1);
    try eq(@as(usize, 0), (try served.plan(a, flags, cfg)).pool); // unset: off, concurrent_users as before
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "781568", 1);
    const p = try served.plan(a, flags, cfg);
    try eq(@as(usize, 781_568), p.pool);
    try eq(@as(u8, 4), p.users);
    try eq(@as(usize, 262_208), p.capacity);
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "262400", 1); // exactly one full-window region: 1,025 blocks
    try eq(@as(usize, 262_400), (try served.plan(a, flags, cfg)).pool);
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "262144", 1);
    try t.expectError(error.PoolTooSmall, served.plan(a, flags, cfg));
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "262145", 1);
    try t.expectError(error.BadAfdPool, served.plan(a, flags, cfg));
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "781568", 1);
    _ = envcall.unsetenv("TF_AFD_USERS");
    const one = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--context", "262144" });
    try t.expectError(error.PoolNeedsUsers, served.plan(a, one, cfg));
}

test "shared_pool blocks: a region of n blocks holds n*256-8 tokens; a request takes the fewest that hold it" {
    try eq(@as(usize, 248), P.tokensOf(1));
    try eq(@as(usize, 262_392), P.tokensOf(1025));
    try eq(@as(usize, 1025), P.blocksFor(P.need(262_143, 1))); // a full 262,144 window: prompt + reply + 1
    try eq(@as(usize, 1025), P.blocksFor(P.need(0, 262_144)));
    for ([_]usize{ 1, 247, 248, 249, 504, 505, 2_561, 262_145 }) |n| {
        const b = P.blocksFor(n);
        try t.expect(P.tokensOf(b) >= n);
        try t.expect(b == 1 or P.tokensOf(b - 1) < n);
    }
}

fn fixtureModel(cut: ?usize, mtp: usize) !struct { parsed: std.json.Parsed(std.json.Value), m: glm.model.Model } {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    errdefer parsed.deinit();
    var m = try glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
    if (cut) |n| try m.cut(n);
    m.mtp_layers = @intCast(mtp);
    return .{ .parsed = parsed, .m = m };
}

test "shared_pool first fit, exact release, PoolFull and ContextFull before any change, compaction moves each region once" {
    const fx = try fixtureModel(5, 0);
    defer fx.parsed.deinit();
    var p = try P.Pool.init(gpa, &fx.m, 12 * P.block, 8, true, 2560);
    defer p.deinit(gpa);
    var mv: [P.max_slots]P.Move = undefined;
    try eq(P.Region{ .start = 0, .blocks = 3 }, (try p.reserve(0, P.tokensOf(3), &mv)).region);
    try eq(P.Region{ .start = 3, .blocks = 3 }, (try p.reserve(1, P.tokensOf(3), &mv)).region);
    try eq(P.Region{ .start = 6, .blocks = 3 }, (try p.reserve(2, P.tokensOf(2) + 1, &mv)).region);
    try t.expectError(error.SlotOutOfStep, p.reserve(2, 1, &mv));
    const before = p.regions;
    try t.expectError(error.PoolFull, p.reserve(3, P.tokensOf(4), &mv));
    try t.expectError(error.ContextFull, p.reserve(3, P.tokensOf(12) + 1, &mv));
    try eq(before, p.regions); // refusals change nothing
    try eq(@as(u64, 0), p.compactions);
    try eq(@as(u32, 3), p.freeBlocks());
    try eq(@as(u32, 3), p.release(1)); // exactly its blocks
    try eq(@as(u32, 0), p.release(1));
    try eq(@as(u32, 6), p.freeBlocks());
    try t.expect(p.fits(P.tokensOf(6)) and !p.fits(P.tokensOf(7)));
    try t.expect(p.fits(P.tokensOf(12) + 1)); // never fits: admitted, so its prefill refuses it in words
    const got = try p.reserve(3, P.tokensOf(5), &mv); // 6 free blocks in two runs of 3: compaction
    try eq(@as(usize, 1), got.moved);
    try eq(P.Move{ .slot = 2, .from = 6, .to = 3, .blocks = 3 }, mv[0]);
    try eq(P.Region{ .start = 6, .blocks = 5 }, got.region);
    try eq(P.Region{ .start = 3, .blocks = 3 }, p.regions[2].?);
    try eq(@as(u32, 1), p.freeBlocks());
    for ([_]u8{ 0, 2, 3 }) |s| _ = p.release(s);
    try eq(p.blocks, p.freeBlocks());
    try eq(p.reserved_blocks, p.released_blocks);
    std.debug.print("shared_pool placement: first fit, exact release, refusals unchanged, 1 compaction moved 1 region (6->3)\n", .{});
}

test "shared_pool views: a region's buffers are State.init(capacity)'s shapes at its first rows, inside it; clamped to the Engine's" {
    const fx = try fixtureModel(5, 0);
    defer fx.parsed.deinit();
    const m = &fx.m;
    var p = try P.Pool.init(gpa, m, 30 * P.block, 8, true, 2560);
    defer p.deinit(gpa);
    var at: u64 = 1 << 36;
    const Place = struct {
        fn visit(a: *u64, _: []const u8, b: *Buf) anyerror!void {
            b.ptr = a.*;
            a.* += 1 << 36;
        }
    };
    try p.walk(&at, Place.visit);
    var st = try glm.buffers.State.init(gpa, m, 2560, 8, true);
    defer st.deinit(gpa);
    const whole = try P.tokenBytes(&st);
    P.detachTokens(&st);
    try eq(@as(u64, 0), try P.tokenBytes(&st));
    for ([_]P.Region{ .{ .start = 27, .blocks = 3 }, .{ .start = 19, .blocks = 11 }, .{ .start = 0, .blocks = 30 } }) |r| {
        p.view(r, &st);
        const cap = @min(P.tokensOf(r.blocks), 2560);
        try eq(cap, st.capacity);
        var want = try glm.buffers.State.init(gpa, m, cap, 8, true);
        defer want.deinit(gpa);
        for (st.kc, want.kc, p.base.kc) |v, w, b| {
            try eq(w.shape, v.shape);
            try eq(b.ptr + @as(u64, r.start) * P.block * b.inner() * 2, v.ptr);
            try t.expect(v.ptr + v.bytes() <= b.ptr + @as(u64, r.start + r.blocks) * P.block * b.inner() * 2);
        }
        for (st.index.?, want.index.?, p.base.index.?) |v, w, b| for (0..3) |j| {
            const per: u64 = if (j == 2) P.block / 4 else P.block;
            try eq(w[j].shape, v[j].shape);
            try eq(b[j].ptr + @as(u64, r.start) * per * b[j].inner() * 2, v[j].ptr);
            try t.expect(v[j].ptr + v[j].bytes() <= b[j].ptr + @min(b[j].bytes(), @as(u64, r.start + r.blocks) * per * b[j].inner() * 2));
        };
        try eq(try P.tokenBytes(&want), try P.tokenBytes(&st));
    }
    try eq(whole, try P.tokenBytes(&st)); // the 30-block region's view is the Engine's own State: capacity 2560
}

// ---- the 262,144-token cap: bytes from the header shapes, N = 4 admission at the pool size --------------------------

test "shared_pool N=4 at the 262,144 cap: pooled bytes = scripts/bytes.py from the shapes; two full windows beside two short" {
    const fx = try fixtureModel(null, 1);
    defer fx.parsed.deinit();
    const served = @import("afd_attn_users.zig");
    const q = try served.pooled(gpa, &fx.m, 262_208, 2048, 4, 781_568);
    // analysis/bytes.json: per-request 330,578,432 x 3, rows(4) 1,488,896, slot 0's caches 5,034,399,744, pool 15,005,958,144
    try eq(@as(u64, 3 * 330_578_432), q.states);
    try eq(@as(u64, 1_488_896), q.rows);
    try eq(@as(u64, 5_034_399_744), q.detached);
    try eq(@as(u64, 15_005_958_144), q.pool);
    const base: u64 = 21_074_526_720; // concurrent_users's dry on the header mirrors: users=1 total at capacity 262,208
    const total = base - q.detached + q.total;
    try eq(@as(u64, 32_039_309_312), total);
    const card: u64 = 32_607 << 20;
    const headroom: u64 = 2 << 30;
    try t.expect(total + headroom <= card);
    try eq(@as(usize, 781_568), try served.largestPool(gpa, &fx.m, 262_208, card - headroom - (total - q.pool)));
    const c4 = try served.extra(gpa, &fx.m, 262_208, 2048, 4);
    try eq(@as(u64, 37_170_950_144), base + c4.total); // today's C4: CAPACITY-LIMITED
    var p = try P.Pool.init(gpa, &fx.m, 781_568, 8, true, 262_208);
    defer p.deinit(gpa);
    var mv: [P.max_slots]P.Move = undefined;
    try eq(@as(u32, 3053), p.blocks);
    _ = try p.reserve(0, P.need(262_143, 1), &mv); // a full 262,144-token request
    _ = try p.reserve(1, P.need(4_000, 4_096), &mv);
    _ = try p.reserve(2, P.need(30_000, 4_096), &mv);
    try t.expect(P.Pool.admitFits(&p, 200_000, 62_144)); // a second full window fits beside them
    _ = try p.reserve(3, P.need(200_000, 62_144), &mv);
    try eq(@as(usize, 262_208), p.capacityOf(p.regions[0].?));
    _ = p.release(1);
    try t.expect(!P.Pool.admitFits(&p, 262_143, 1)); // a third full window waits
    try t.expect(P.Pool.admitFits(&p, 100_000, 4_096));
    _ = p.release(3);
    try t.expect(P.Pool.admitFits(&p, 262_143, 1)); // until one leaves
    var one = try P.Pool.init(gpa, &fx.m, 781_568, 8, true, 262_208);
    defer one.deinit(gpa);
    _ = try one.reserve(0, P.need(262_143, 1), &mv);
    const left = one.freeBlocks() / 3; // one full window beside three of equal size
    try eq(@as(usize, 173_048), P.tokensOf(left));
    std.debug.print("shared_pool N=4 full262144 MTP_loaded=1: pool 781568 tokens = {d} B, per-request states {d} B, rows {d} B, slot 0 detached {d} B; total {d} <= {d} - 2 GiB (today C4 {d}); 2 full windows + 2 short fit, a 3rd waits; beside one full: three of {d}\n", .{ q.pool, q.states, q.rows, q.detached, total, card, base + c4.total, P.tokensOf(left) });
}

// ---- recorded real users over pooled regions ------------------------------------------------------------------------

const positions = [_]usize{ 511, 2047, 513, 2048 };

fn rows(n: u8) [4]schedule.Row {
    var out: [4]schedule.Row = undefined;
    for (0..n) |i| out[i] = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = @intCast(10 + i), .position = positions[i] };
    return out;
}

/// concurrent_users's recorded users with the pool: slot 0's caches detached as afd_attn_gpu does before the Engine's attach, the
/// extra States' by usePool, the pool's buffers placed in the Space after everything else.
const Pooled = struct {
    r: dev.Rec = undefined,
    pool: P.Pool = undefined,
    fn init(x: *Pooled, n: u8, prefill_rows: usize, max_rows: u32, blocks: u32) !void {
        try x.r.initRows(n, prefill_rows, max_rows);
        errdefer x.r.deinit();
        P.detachTokens(&x.r.e.st);
        x.pool = try P.Pool.init(gpa, &x.r.m, @as(usize, blocks) * P.block, x.r.e.buf.rows, true, x.r.e.st.capacity);
        errdefer x.pool.deinit(gpa);
        try x.pool.walk(&x.r.space, glm.trace.placeAll);
        try x.r.users.usePool(&x.pool);
    }
    fn deinit(x: *Pooled) void {
        x.pool.deinit(gpa);
        x.r.deinit();
    }
    fn slotState(x: *Pooled, slot: usize) *glm.buffers.State {
        return if (slot == 0) &x.r.e.st else &x.r.users.states[slot - 1];
    }
};

fn addAll(s: *Space, path: []const u8, b: *Buf) anyerror!void {
    try s.add(path, b.*);
}

/// A Space naming each slot's State buffers as today's own States are named ("st", "users.stateN") at their current
/// addresses (a region's view: offsets from the region's base; with a pool, live regions only), then everything `s` names.
fn regionNames(s: *const Space, e: *glm.engine.Engine, u: *glm.afd_users.Users, pool: ?*const P.Pool) !Space {
    var out: Space = .{ .gpa = gpa };
    errdefer out.deinit();
    if (pool == null or pool.?.regions[0] != null) try glm.views.walk(&e.st, glm.views.Path.init("st"), &out, addAll);
    for (u.states[0 .. u.count - 1], 1..) |*st, i| {
        if (pool) |p| if (p.regions[i] == null) continue; // a released slot's stale view names nothing
        var name: [32]u8 = undefined;
        try glm.views.walk(st, glm.views.Path.init(try std.fmt.bufPrint(&name, "users.state{d}", .{i})), &out, addAll);
    }
    for (s.entries.items) |x| try out.add(x.path, x.buf);
    return out;
}

test "shared_pool recorded C4 round: pooled regions launch exactly today's own States, relative to each region's base" {
    var a: Pooled = .{};
    try a.init(4, 64, 8, 44);
    defer a.deinit();
    for (0..4) |i| try a.r.users.reserve(@intCast(i), 2560, 0); // 11 blocks each: views at the Engine's capacity
    for (0..4) |i| try eq(@as(usize, 2560), a.slotState(i).capacity);
    var b: dev.Rec = undefined;
    try b.initRows(4, 64, 8);
    defer b.deinit();
    const rs = rows(4);
    var ao: [4]u32 = undefined;
    var bo: [4]u32 = undefined;
    var ca: Capture = .{ .u = &a.r.users, .trace = &a.r.trace };
    var cb: Capture = .{ .u = &b.users, .trace = &b.trace };
    try schedule.run(Capture, &ca, rs[0..4], &ao); // Capture flushes around each front (64-launch recorder)
    try schedule.run(Capture, &cb, rs[0..4], &bo);
    try a.r.f.flush();
    try b.f.flush();
    var names = try regionNames(&a.r.space, a.r.e, &a.r.users, &a.pool);
    defer names.deinit();
    try glm.serve_rec.expectSame(&names, a.r.trace.events.items, &b.space, b.trace.events.items);
    var pooled_ops: usize = 0; // launches that read or write the pool (the attention caches)
    for (a.r.trace.events.items) |ev| for (ev.args) |arg| switch (arg.value) {
        .ptr => |q| if (a.r.space.name(q.addr)) |nm| if (std.mem.startsWith(u8, nm.path, "pool.")) {
            pooled_ops += 1;
        },
        else => {},
    };
    try t.expect(pooled_ops > 0);
    try eq(a.r.posts, b.posts);
    for (rs[0..4]) |row| try eq(row.position + 1, a.r.users.position(row.key));
    std.debug.print("shared_pool recorded C4 pooled=own States: {d} launches identical by name/offset/grid/scalar, {d} pool operands; positions 511/2047/513/2048\n", .{ a.r.trace.events.items.len, pooled_ops });
}

// Delegates to the real device methods, capturing each user's front launches (concurrent_users's Capture).
const Capture = struct {
    u: *glm.afd_users.Users,
    trace: *glm.trace.Trace,
    starts: [4][5]usize = undefined,
    ends: [4][5]usize = undefined,
    pub fn alive(c: *Capture) !void {
        try c.u.alive();
    }
    pub fn begin(c: *Capture, rs: []const schedule.Row) !void {
        try c.u.begin(rs);
    }
    pub fn open(c: *Capture, r: schedule.Row, i: usize) !void {
        try c.u.open(r, i);
    }
    pub fn layerCount(c: *Capture) usize {
        return c.u.layerCount();
    }
    pub fn isMoe(c: *Capture, li: usize) bool {
        return c.u.isMoe(li);
    }
    pub fn front(c: *Capture, r: schedule.Row, i: usize, li: usize) !void {
        try c.u.f.flush();
        c.starts[r.key.slot][li] = c.trace.events.items.len;
        try c.u.front(r, i, li);
        try c.u.f.flush();
        c.ends[r.key.slot][li] = c.trace.events.items.len;
    }
    pub fn gather(c: *Capture, r: schedule.Row, i: usize) !void {
        try c.u.gather(r, i);
    }
    pub fn exchange(c: *Capture, li: usize, n: usize) !void {
        try c.u.exchange(li, n);
    }
    pub fn scatter(c: *Capture, r: schedule.Row, i: usize, n: usize) !void {
        try c.u.scatter(r, i, n);
    }
    pub fn tail(c: *Capture, r: schedule.Row, i: usize) !void {
        try c.u.tail(r, i);
    }
    pub fn settle(c: *Capture) !void {
        try c.u.settle();
    }
    pub fn sample(c: *Capture, r: schedule.Row, i: usize) !u32 {
        return c.u.sample(r, i);
    }
    pub fn commit(c: *Capture, r: schedule.Row, i: usize) !void {
        try c.u.commit(r, i);
    }
};

test "shared_pool recorded C4 small regions: each user's fronts together = alone at block 0, relative to its region's base" {
    // regions of 3, 9, 3, 9 blocks (capacity 760 / 2296: under the Engine's 2560), after a 2-block gap
    const need = [_]usize{ 516, 2052, 518, 2053 };
    var a: Pooled = .{};
    try a.init(4, 64, 8, 2 + 24);
    defer a.deinit();
    var mv: [P.max_slots]P.Move = undefined;
    _ = try a.pool.reserve(3, P.tokensOf(2), &mv); // a gap the others sit after (slot 3 re-reserved below)
    for (0..3) |i| try a.r.users.reserve(@intCast(i), need[i] - 1, 0);
    _ = a.pool.release(3);
    try a.r.users.reserve(3, need[3] - 1, 0);
    const want = [_]P.Region{ .{ .start = 2, .blocks = 3 }, .{ .start = 5, .blocks = 9 }, .{ .start = 14, .blocks = 3 }, .{ .start = 17, .blocks = 9 } };
    for (want, 0..) |w, i| try eq(w, a.pool.regions[i].?);
    const rs = rows(4);
    var out: [4]u32 = undefined;
    var ca: Capture = .{ .u = &a.r.users, .trace = &a.r.trace };
    try schedule.run(Capture, &ca, rs[0..4], out[0..4]);
    var names = try regionNames(&a.r.space, a.r.e, &a.r.users, &a.pool);
    defer names.deinit();
    var compared: usize = 0;
    for (0..4) |i| {
        var s: Pooled = .{};
        try s.init(4, 64, 8, 2 + 24);
        defer s.deinit();
        try s.r.users.reserve(@intCast(i), need[i] - 1, 0); // alone: first fit at block 0
        try eq(@as(u32, 0), s.pool.regions[i].?.start);
        var cs: Capture = .{ .u = &s.r.users, .trace = &s.r.trace };
        var o: [1]u32 = undefined;
        try schedule.run(Capture, &cs, rs[i .. i + 1], &o);
        var sn = try regionNames(&s.r.space, s.r.e, &s.r.users, &s.pool);
        defer sn.deinit();
        try t.expect(a.pool.regions[i].?.start != 0);
        for (0..5) |li| {
            try glm.serve_rec.expectSame(&names, a.r.trace.events.items[ca.starts[i][li]..ca.ends[i][li]], &sn, s.r.trace.events.items[cs.starts[i][li]..cs.ends[i][li]]);
            compared += ca.ends[i][li] - ca.starts[i][li];
        }
        try eq(a.pool.capacityOf(a.pool.regions[i].?), a.slotState(i).capacity);
        try eq(a.slotState(i).capacity, s.slotState(i).capacity);
    }
    std.debug.print("shared_pool recorded C4 small regions: {d} front launches together = alone (regions at blocks {d},{d},{d},{d} vs 0; capacities {d}/{d}/{d}/{d}; State0/1 512 boundary and the 2048 rows)\n", .{ compared, a.pool.regions[0].?.start, a.pool.regions[1].?.start, a.pool.regions[2].?.start, a.pool.regions[3].?.start, a.slotState(0).capacity, a.slotState(1).capacity, a.slotState(2).capacity, a.slotState(3).capacity });
}

test "shared_pool recorded compaction: the moved region's rows copied front first in non-overlapping pieces, then re-pointed" {
    var a: Pooled = .{};
    try a.init(4, 64, 8, 22);
    defer a.deinit();
    try a.r.users.reserve(0, P.tokensOf(3) - 1, 0); // [0, 3)
    try a.r.users.reserve(1, P.tokensOf(3) - 1, 0); // [3, 6)
    try a.r.users.reserve(2, 2560, 0); // [6, 17): 11 blocks, the Engine's capacity
    _ = a.pool.release(1);
    try a.r.f.flush();
    a.r.trace.reset();
    try a.r.users.reserve(3, P.tokensOf(6) - 1, 0); // 8 free in runs of 3 and 5: slot 2 slides 6 -> 3
    try a.r.f.flush();
    try eq(@as(u64, 1), a.pool.compactions);
    try eq(@as(u64, 1), a.pool.moves);
    try eq(P.Region{ .start = 3, .blocks = 11 }, a.pool.regions[2].?);
    try eq(P.Region{ .start = 14, .blocks = 6 }, a.pool.regions[3].?);
    // every pool buffer: latent rows [768, 768 + 2808) <- [1536, 1536 + 2808) in pieces of 768; pooled keys 192s
    var covered = std.StringHashMap([2]u64).init(gpa);
    defer covered.deinit();
    var copies: usize = 0;
    var last_src: std.StringHashMap(u64) = .init(gpa);
    defer last_src.deinit();
    for (a.r.trace.events.items) |ev| {
        try t.expectEqualStrings("torch.copy_", ev.name);
        const dst = ev.vals[0].t;
        const src = ev.vals[1].t;
        const dn = a.r.space.name(dst.ptr).?;
        const sn = a.r.space.name(src.ptr).?;
        try t.expectEqualStrings(dn.path, sn.path);
        try t.expect(std.mem.startsWith(u8, dn.path, "pool."));
        const row_bytes: u64 = dst.shape[1] * 2; // bf16
        const pooled_keys = std.mem.indexOf(u8, dn.path, ".index.") != null and std.mem.endsWith(u8, dn.path, ".2");
        const per: u64 = if (pooled_keys) P.block / 4 else P.block;
        try eq(dst.shape, src.shape);
        try t.expect(dn.off + dst.shape[0] * row_bytes <= sn.off); // no overlap: the destination ends at or before the source
        try eq(3 * per * row_bytes, sn.off - dn.off); // the shift: 3 blocks
        const g = try last_src.getOrPut(dn.path); // front first: each piece's source after the one before
        if (g.found_existing) try t.expect(sn.off > g.value_ptr.*);
        g.value_ptr.* = sn.off;
        const c = try covered.getOrPut(dn.path);
        if (!c.found_existing) {
            try eq(3 * per * row_bytes, dn.off); // the first piece lands on the new first block
            c.value_ptr.* = .{ dn.off, 0 };
        }
        c.value_ptr.*[1] += dst.shape[0];
        copies += 1;
    }
    var it = covered.iterator();
    var buffers: usize = 0;
    while (it.next()) |kv| {
        const pooled_keys = std.mem.indexOf(u8, kv.key_ptr.*, ".index.") != null and std.mem.endsWith(u8, kv.key_ptr.*, ".2");
        try eq(if (pooled_keys) @as(u64, 11 * 64) else @as(u64, P.tokensOf(11)), kv.value_ptr.*[1]);
        buffers += 1;
    }
    var pool_buffers: usize = 0;
    const Count = struct {
        fn visit(n: *usize, _: []const u8, _: *Buf) anyerror!void {
            n.* += 1;
        }
    };
    try a.pool.walk(&pool_buffers, Count.visit);
    try eq(pool_buffers, buffers);
    const st = a.slotState(2);
    try eq(a.pool.base.kc[0].ptr + 3 * P.block * a.pool.base.kc[0].inner() * 2, st.kc[0].ptr);
    try eq(@as(usize, 2560), st.capacity);
    // the moved slot's next launches equal today's own State's, relative to its new base
    var b: dev.Rec = undefined;
    try b.initRows(4, 64, 8);
    defer b.deinit();
    a.r.trace.reset();
    const row = rows(4)[2];
    var ao: [1]u32 = undefined;
    var bo: [1]u32 = undefined;
    try a.r.users.round(&.{row}, &ao);
    try b.users.round(&.{row}, &bo);
    try a.r.f.flush();
    try b.f.flush();
    var names = try regionNames(&a.r.space, a.r.e, &a.r.users, &a.pool);
    defer names.deinit();
    try glm.serve_rec.expectSame(&names, a.r.trace.events.items, &b.space, b.trace.events.items);
    std.debug.print("shared_pool recorded compaction: 1 move (slot 2, blocks 6->3, 11 blocks): {d} copies over {d} pool buffers, front first, no overlap, all region rows; its next round = own State's ({d} launches)\n", .{ copies, buffers, a.r.trace.events.items.len });
}

// ---- the backend's reserve and release ------------------------------------------------------------------------------

const Back = users.Users(glm.afd_users.Users);

fn backOf(x: *Pooled, n: u8) Back {
    @memset(&x.r.users.generation, 0);
    return .{ .steps = &x.r.users, .count = n, .capacity = 2560, .reach = 2560 };
}

fn idsOf(buf: []u32, seed: u32) []u32 {
    for (buf, 0..) |*id, i| id.* = seed + @as(u32, @intCast(i % 97));
    return buf;
}

const Flag = struct {
    fn check(ctx: *anyopaque) bool {
        const p: *bool = @ptrCast(@alignCast(ctx));
        return p.*;
    }
};

/// The recorded transport's take, 1 ms slower: a long request stays live while a test looks at the queue; its post
/// keeps the recorder's first 128 headers (Rec.headers) and counts the rest.
const Slow = struct {
    fn post(p: *anyopaque, node: *cuda.xchg.Node, hdr: *const [cuda.xchg.proto.HDR_BYTES]u8, pos: usize, R: usize) anyerror!void {
        const r: *dev.Rec = @ptrCast(@alignCast(p));
        if (r.posts < r.headers.len) return dev.Rec.post(p, node, hdr, pos, R);
        r.posts += 1;
    }
    fn take(p: *anyopaque, node: *cuda.xchg.Node, pos: usize, R: usize) anyerror!void {
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        return dev.Rec.take(p, node, pos, R);
    }
};

test "shared_pool backend: a full pool refuses one request (no poisoning, slot unchanged); release on finish returns its blocks" {
    var x: Pooled = .{};
    try x.init(2, 96, 96, 4);
    defer x.deinit();
    var b = backOf(&x, 2);
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    var ia: [96]u32 = undefined;
    var ib: [300]u32 = undefined;
    var cancel_a = false;
    var sa = try lanes.Stream.init(gpa, .{ .id = "A", .prompt = idsOf(&ia, 1000), .max_new = 600, .drafts = false, .cancel_check = .{ .ptr = &cancel_a, .check = Flag.check } });
    defer sa.deinit(gpa);
    var sb = try lanes.Stream.init(gpa, .{ .id = "B", .prompt = idsOf(&ib, 2000), .max_new = 16, .drafts = false });
    defer sb.deinit(gpa);
    try core.addStream(&sa); // 697 rows: 3 blocks of 4
    try eq(@as(u32, 1), x.pool.freeBlocks());
    const gen = b.slots[1].generation;
    try t.expectError(error.PoolFull, core.addStream(&sb)); // 317 rows: 2 blocks
    try t.expect(b.failed == null);
    try t.expect(b.slots[1].owner == null and b.slots[1].generation == gen);
    try t.expect(x.r.users.at.failure() == null and x.pool.regions[1] == null);
    for (0..3) |_| try core.step(); // A goes on: the refusal touched nothing of it
    try eq(@as(usize, 4), sa.emitted().len);
    cancel_a = true;
    while (core.activeCount() > 0) try core.step();
    try eq(@as(u32, 4), x.pool.freeBlocks()); // cancelled: exactly its 3 blocks back
    var sc = try lanes.Stream.init(gpa, .{ .id = "C", .prompt = idsOf(&ib, 3000), .max_new = 16, .drafts = false });
    defer sc.deinit(gpa);
    try core.addStream(&sc);
    while (core.activeCount() > 0) try core.step();
    try eq(lanes.stream.Reason.length, sc.reason);
    try eq(@as(usize, 16), sc.emitted().len);
    try eq(x.pool.blocks, x.pool.freeBlocks()); // finished: exactly its blocks back
    try eq(x.pool.reserved_blocks, x.pool.released_blocks);
    try eq(@as(u64, 2), x.pool.reserves);
    std.debug.print("shared_pool backend: PoolFull refused B alone (failed=null, slot unchanged), A cancelled -> 3 blocks back, C finished 16 tokens -> 2 blocks back; reserved {d} = released {d}\n", .{ x.pool.reserved_blocks, x.pool.released_blocks });
}

test "shared_pool backend: a failed round fails every owner and every region returns to the pool" {
    var x: Pooled = .{};
    try x.init(2, 96, 96, 8);
    defer x.deinit();
    var b = backOf(&x, 2);
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    var ia: [96]u32 = undefined;
    var ib: [130]u32 = undefined;
    var sa = try lanes.Stream.init(gpa, .{ .id = "A", .prompt = idsOf(&ia, 1000), .max_new = 64, .drafts = false });
    defer sa.deinit(gpa);
    var sb = try lanes.Stream.init(gpa, .{ .id = "B", .prompt = idsOf(&ib, 2000), .max_new = 64, .drafts = false });
    defer sb.deinit(gpa);
    try core.addStream(&sa);
    try core.addStream(&sb);
    try eq(@as(u32, 8 - 1 - 1), x.pool.freeBlocks());
    x.r.fail_take = x.r.takes + 1; // the next exchange's take fails
    var failed = false;
    while (core.activeCount() > 0) core.step() catch {
        failed = true;
        break;
    };
    try t.expect(failed);
    try t.expect(b.failed != null);
    b.backend().release(&sa); // what the lane host's failAll does through the core
    b.backend().release(&sb);
    try eq(x.pool.blocks, x.pool.freeBlocks());
    try eq(x.pool.reserved_blocks, x.pool.released_blocks);
    var sc = try lanes.Stream.init(gpa, .{ .id = "C", .prompt = idsOf(&ia, 3000), .max_new = 4, .drafts = false });
    defer sc.deinit(gpa);
    try t.expectError(error.ExpertFailed, core.addStream(&sc)); // sticky: refused before any reservation
    try eq(x.pool.blocks, x.pool.freeBlocks());
    std.debug.print("shared_pool backend failure: sticky {s}, both regions released ({d} blocks), a later request refused before reserving\n", .{ @errorName(b.failed.?), x.pool.released_blocks });
}

// ---- the lane host: a request that does not fit waits, in order, over recorded real users -------------------------

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: usize = 0,
    done: ?api.Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |x| b.tokens += x.len,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn state(b: *@This()) struct { usize, ?api.Reason } {
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        return .{ b.tokens, b.done };
    }
    fn wait(b: *@This()) !api.Reason {
        for (0..60_000) |_| {
            if (b.state()[1]) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.WaitTimeout;
    }
};

test "shared_pool lane host over recorded real users: a request the pool cannot hold waits in order, never fails, then runs" {
    var x: Pooled = .{};
    try x.init(2, 96, 96, 11);
    defer x.deinit();
    x.r.a.wire.?.host.?.take = Slow.take;
    x.r.a.wire.?.host.?.post = Slow.post;
    var b = backOf(&x, 2);
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    const admit: api.Admit = .{ .ctx = &x.pool, .fits = P.Pool.admitFits };
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .context_window = 2560, .drafts = "off", .greedy_users = true, .admit = admit });
    try host.start();
    var up = true;
    defer if (up) host.stop();
    var ia: [96]u32 = undefined;
    var ib: [300]u32 = undefined;
    var ic: [40]u32 = undefined;
    const long: api.Request = .{ .prompt = idsOf(&ia, 1000), .max_tokens = 2400, .drafts = false }; // 10 blocks of 11
    const wide: api.Request = .{ .prompt = idsOf(&ib, 2000), .max_tokens = 16, .drafts = false }; // 2 blocks
    const small: api.Request = .{ .prompt = idsOf(&ic, 3000), .max_tokens = 8, .drafts = false }; // 1 block, after B
    const e = host.engine();
    var a: Box = .{};
    var bb: Box = .{};
    var c: Box = .{};
    try e.submit(1, &long, .{ .ctx = &a, .event = Box.event });
    for (0..60_000) |_| {
        if (a.state()[0] > 0) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try e.submit(2, &wide, .{ .ctx = &bb, .event = Box.event });
    try e.submit(3, &small, .{ .ctx = &c, .event = Box.event });
    std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    try eq(@as(usize, 0), bb.state()[0]); // a lane is free, the pool is not: B waits
    try eq(@as(?api.Reason, null), bb.state()[1]);
    try eq(@as(usize, 0), c.state()[0]); // C would fit, but waits behind B (first in, first out)
    try eq(@as(?api.Reason, null), a.state()[1]);
    e.cancel(1);
    try eq(api.Reason.cancelled, try a.wait());
    try eq(api.Reason.length, try bb.wait());
    try eq(api.Reason.length, try c.wait());
    host.stop();
    up = false;
    try eq(@as(usize, 16), bb.state()[0]);
    try eq(@as(usize, 8), c.state()[0]);
    try t.expect(b.failed == null);
    try eq(x.pool.blocks, x.pool.freeBlocks());
    try eq(x.pool.reserved_blocks, x.pool.released_blocks);
    try eq(@as(u64, 3), x.pool.reserves);
    std.debug.print("shared_pool lane host: B (2 blocks) waited beside A (3 of 4), C queued behind B; A cancelled, B 16 and C 8 tokens; no failure; reserved {d} = released {d} blocks\n", .{ x.pool.reserved_blocks, x.pool.released_blocks });
}

test "shared_pool off: no pool, no reservation, concurrent_users's bytes unchanged" {
    var r: dev.Rec = undefined;
    try r.initRows(4, 64, 8);
    defer r.deinit();
    const before = try r.users.bytes();
    try r.users.reserve(0, 100, 100); // a no-op without a pool
    try t.expect(r.users.pool == null);
    try eq(before, try r.users.bytes());
    try eq(@as(usize, 2560), r.e.st.capacity);
}
