//! user_sampling (local integration; Claude Opus 5.5 via Hermes) CPU tests (step test-user_sampling): TF_AFD_USERS_SAMPLING's backend
//! rules over recorded real users (concurrent_users's dev.Rec: the real native Users over the real engine Users and recorded
//! CUDA launches; shared_pool's pool): the switch off refuses a sampled stream at its claim before any reservation; on, each
//! row's draw is the Engine's keyed draw under its own request's sampling at its own token positions (the recorder's
//! picker reads the Engine's sampling at each draw), the Engine's sampling is null again after every draw, greedy rows
//! draw greedy; a sampled stream cancelled inside a round and a failed round with a sampled row release every slot's
//! sampling and exactly their pool blocks. Recorder only: no CUDA numerics.
//! Scaffolding: shared_pool's afd_pool_test.zig (Pooled; Claude Opus 5.5 via Hermes) over concurrent_users's afd_users_device_test.zig
//! (GPT-6.1 Sol via Hermes) after Ash Hart's cuda_afd_engine_test.zig; NOTICE applies (Ash Hart / TensorFold; Mia
//! (@MiaAI_lab), whose union cover qualification_45c10bcf's draw carries).
const std = @import("std");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const users = @import("glm_users");
const dev = @import("afd_users_device_test.zig");
const P = glm.afd_pool;
const t = std.testing;
const gpa = t.allocator;
const eq = t.expectEqual;
const Back = users.Users(glm.afd_users.Users);

/// The recorder's picker: a sampled draw (positions given) answers a token keyed only by the Engine's sampling AT THAT
/// DRAW and the position (100 + (seed * 31 + position * 7) % 997), logging both; a greedy draw answers 7.
const Pick = struct {
    e: *glm.engine.Engine,
    seeds: [4096]u64 = undefined,
    positions: [4096]u64 = undefined,
    n: usize = 0,
    greedy: usize = 0,
    fail_seed: ?u64 = null,
    fn token(seed: u64, position: u64) u32 {
        return @intCast(100 + (seed *% 31 +% position *% 7) % 997);
    }
    fn pick(p: *anyopaque, _: glm.trace.T, positions: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        const x: *Pick = @ptrCast(@alignCast(p));
        if (positions) |pos| {
            const s = x.e.sampling orelse return error.SampledWithoutSampling;
            if (x.fail_seed) |f| if (s.seed == f) return error.MockDeviceFault;
            for (pos, out) |q, *o| {
                o.* = token(s.seed, q);
                if (x.n < x.seeds.len) {
                    x.seeds[x.n] = s.seed;
                    x.positions[x.n] = q;
                    x.n += 1;
                }
            }
        } else {
            x.greedy += 1;
            @memset(out, 7);
        }
        if (probs) |q| @memset(q, 0);
    }
    fn drawsOf(x: *const Pick, seed: u64, buf: []u64) []u64 {
        var k: usize = 0;
        for (x.seeds[0..x.n], x.positions[0..x.n]) |s, q| if (s == seed and k < buf.len) {
            buf[k] = q;
            k += 1;
        };
        return buf[0..k];
    }
};

/// shared_pool's Pooled (afd_pool_test.zig) with TF_GLM_SAMPLING=1's nucleus rows placed beside the Engine's temporaries.
const Rig = struct {
    r: dev.Rec = undefined,
    pool: P.Pool = undefined,
    pick: Pick = undefined,
    fn init(x: *Rig, n: u8, blocks: u32) !void {
        try x.r.initRows(n, 96, 96);
        errdefer x.r.deinit();
        try x.r.e.useSampling(gpa); // afd_attn_gpu's TF_GLM_SAMPLING=1
        x.r.e.sampled_top_k = true;
        if (x.r.e.nuc) |*nu| try glm.views.walk(&nu.s, glm.views.Path.init("tmp.nucleus"), &x.r.space, glm.trace.placeAll);
        P.detachTokens(&x.r.e.st);
        x.pool = try P.Pool.init(gpa, &x.r.m, @as(usize, blocks) * P.block, x.r.e.buf.rows, true, x.r.e.st.capacity);
        errdefer x.pool.deinit(gpa);
        try x.pool.walk(&x.r.space, glm.trace.placeAll);
        try x.r.users.usePool(&x.pool);
        x.pick = .{ .e = x.r.e };
        x.r.trace.picker = .{ .ctx = &x.pick, .pick = Pick.pick };
    }
    fn deinit(x: *Rig) void {
        x.pool.deinit(gpa);
        x.r.deinit();
    }
    fn back(x: *Rig, n: u8, sampled: bool) Back {
        @memset(&x.r.users.generation, 0);
        return .{ .steps = &x.r.users, .count = n, .capacity = 2560, .reach = 2560, .sampled = sampled };
    }
};

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

fn keyed(seed: u64) lanes.Sampling {
    return .{ .seed = seed, .temperature = 0.8, .top_k = 0, .top_p = 0.9, .min_p = 0.0 };
}

fn run(core: *lanes.Engine) !void {
    while (core.activeCount() > 0) try core.step();
}

test "user_sampling backend: the switch off refuses a sampled stream at its claim (GreedyOnly) before any reservation; on, each row's draws are its own request's (seed, token position), greedy rows draw greedy, and the Engine's sampling is null between draws" {
    var x: Rig = .{};
    try x.init(2, 8);
    defer x.deinit();
    var ia: [40]u32 = undefined;
    var ib: [130]u32 = undefined; // past one 96-row chunk: the first draw after a two-chunk prompt
    {
        var off = x.back(2, false);
        var cfg = try lanes.Config.init(gpa, off.facts(), 1, 0);
        defer cfg.deinit(gpa);
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, off.backend(), clock.clock());
        defer core.deinit();
        var s = try lanes.Stream.init(gpa, .{ .id = "S", .prompt = idsOf(&ia, 1000), .max_new = 8, .drafts = false, .sampling = keyed(41) });
        defer s.deinit(gpa);
        try t.expectError(error.GreedyOnly, core.addStream(&s));
        try eq(x.pool.blocks, x.pool.freeBlocks()); // refused before any reservation
        try t.expect(off.failed == null and off.slots[0].owner == null);
    }
    var b = x.back(2, true);
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    var sa = try lanes.Stream.init(gpa, .{ .id = "A", .prompt = idsOf(&ib, 2000), .max_new = 12, .drafts = false, .sampling = keyed(41) });
    defer sa.deinit(gpa);
    var sg = try lanes.Stream.init(gpa, .{ .id = "G", .prompt = idsOf(&ia, 1000), .max_new = 12, .drafts = false });
    defer sg.deinit(gpa);
    try core.addStream(&sa);
    try core.addStream(&sg);
    try run(&core);
    try eq(@as(usize, 12), sa.emitted().len);
    try eq(@as(usize, 12), sg.emitted().len);
    // A: every token is the keyed answer at its own position (prompt length 130 onward), drawn under seed 41
    for (sa.emitted(), 0..) |tok, i| try eq(Pick.token(41, 130 + i), tok);
    var buf: [64]u64 = undefined;
    const ps = x.pick.drawsOf(41, &buf);
    try eq(@as(usize, 12), ps.len);
    for (ps, 0..) |q, i| try eq(@as(u64, 130 + i), q);
    // G: greedy, today's draw (7) every token
    for (sg.emitted()) |tok| try eq(@as(u32, 7), tok);
    try t.expect(x.r.e.sampling == null); // restored after every draw
    for (x.r.users.sampling) |s| try t.expect(s == null); // released: no slot keeps a request's sampling
    try eq(x.pool.blocks, x.pool.freeBlocks());
    // alone: the same request in a fresh generation, alone, draws the same tokens (keyed by seed and position only)
    var s2 = try lanes.Stream.init(gpa, .{ .id = "A2", .prompt = idsOf(&ib, 2000), .max_new = 12, .drafts = false, .sampling = keyed(41) });
    defer s2.deinit(gpa);
    try core.addStream(&s2);
    try run(&core);
    try t.expectEqualSlices(u32, sa.emitted(), s2.emitted());
    std.debug.print("user_sampling backend: off GreedyOnly before reserving; on: A 12 keyed draws at positions 130..141 under seed 41, G 12 greedy, alone = together, Engine sampling null after, every block back\n", .{});
}

test "user_sampling backend: a sampled stream cancelled inside a round is skipped before its draw; its slot's sampling and exactly its blocks come back; its greedy peer goes on as alone" {
    var x: Rig = .{};
    try x.init(2, 8);
    defer x.deinit();
    var b = x.back(2, true);
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    var ia: [40]u32 = undefined;
    var ib: [60]u32 = undefined;
    var cancel_a = false;
    var sa = try lanes.Stream.init(gpa, .{ .id = "A", .prompt = idsOf(&ib, 2000), .max_new = 64, .drafts = false, .sampling = keyed(77), .cancel_check = .{ .ptr = &cancel_a, .check = Flag.check } });
    defer sa.deinit(gpa);
    var sg = try lanes.Stream.init(gpa, .{ .id = "G", .prompt = idsOf(&ia, 1000), .max_new = 16, .drafts = false });
    defer sg.deinit(gpa);
    try core.addStream(&sa);
    try core.addStream(&sg);
    const before = x.pool.freeBlocks();
    try t.expect(x.r.users.sampling[0] != null or x.r.users.sampling[1] != null); // A's slot holds its sampling
    for (0..3) |_| try core.step();
    var buf: [128]u64 = undefined;
    const n_before = x.pick.drawsOf(77, &buf).len;
    x.r.cancel_in_take = &cancel_a; // the cancel lands inside the next round's exchange: after A's launches, before its draw
    try core.step();
    x.r.cancel_in_take = null;
    try run(&core);
    const n_after = x.pick.drawsOf(77, &buf).len;
    try eq(n_before, n_after); // no draw for A once cancelled inside the round
    try t.expect(sa.emitted().len < 64);
    try eq(@as(usize, 16), sg.emitted().len);
    for (sg.emitted()) |tok| try eq(@as(u32, 7), tok);
    for (x.r.users.sampling) |s| try t.expect(s == null);
    try eq(x.pool.blocks, x.pool.freeBlocks());
    try t.expect(before < x.pool.blocks);
    try t.expect(x.r.e.sampling == null);
    try eq(x.pool.reserved_blocks, x.pool.released_blocks);
    std.debug.print("user_sampling backend cancel: A cancelled inside a round after {d} keyed draws (none after), {d} tokens out; G 16 greedy; every slot's sampling cleared; blocks {d}/{d} back; reserved {d} = released {d}\n", .{ n_before, sa.emitted().len, x.pool.freeBlocks(), x.pool.blocks, x.pool.reserved_blocks, x.pool.released_blocks });
}

test "user_sampling backend: a failed round with a sampled row (its draw faults) fails every owner; every region returns, no slot keeps a sampling, later claims refused" {
    var x: Rig = .{};
    try x.init(2, 8);
    defer x.deinit();
    var b = x.back(2, true);
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    var ia: [40]u32 = undefined;
    var ib: [60]u32 = undefined;
    var sa = try lanes.Stream.init(gpa, .{ .id = "A", .prompt = idsOf(&ib, 2000), .max_new = 64, .drafts = false, .sampling = keyed(99) });
    defer sa.deinit(gpa);
    var sb = try lanes.Stream.init(gpa, .{ .id = "B", .prompt = idsOf(&ia, 1000), .max_new = 64, .drafts = false, .sampling = keyed(98) });
    defer sb.deinit(gpa);
    try core.addStream(&sa);
    try core.addStream(&sb);
    for (0..2) |_| try core.step();
    x.pick.fail_seed = 99; // A's next draw faults inside the round, after the settle
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
    for (x.r.users.sampling) |s| try t.expect(s == null);
    try t.expect(x.r.e.sampling == null); // the faulted draw's sampling restored too
    var sc = try lanes.Stream.init(gpa, .{ .id = "C", .prompt = idsOf(&ia, 3000), .max_new = 4, .drafts = false, .sampling = keyed(5) });
    defer sc.deinit(gpa);
    try t.expectError(error.MockDeviceFault, core.addStream(&sc)); // sticky: refused before any reservation
    try eq(x.pool.blocks, x.pool.freeBlocks());
    std.debug.print("user_sampling backend failure: sticky {s}, both regions released ({d} blocks), no slot sampling left, Engine sampling null, a later sampled claim refused before reserving\n", .{ @errorName(b.failed.?), x.pool.released_blocks });
}
