//! ping_pong scheduler contract. GPT-6.1 Sol (OpenAI Codex), via Hermes.
const std = @import("std");
const glm = @import("glm_cuda");
const pp = glm.afd_users_pp;
test {
    _ = @import("ping_pong_rec_test.zig");
}
const dsv = glm.afd_users_dsv;
const t = std.testing;

const drafts = [_]u32{ 2, 3, 4 };
fn win(slot: u8, width: usize) dsv.Window {
    return .{ .row = .{ .key = .{ .slot = slot, .generation = 1 }, .token = 1, .position = 11 }, .drafts = drafts[0 .. width - 1] };
}

test "ping_pong switch default off, prerequisites and malformed spelling" {
    try t.expect(!try pp.parse(null, false, false, 1, 0));
    try t.expect(!try pp.parse("0", false, false, 1, 0));
    try t.expect(try pp.parse("1", true, true, 2, 0));
    try t.expectError(error.PingPongNeedsSharedDrafts, pp.parse("1", false, true, 2, 0));
    try t.expectError(error.PingPongNeedsPairs, pp.parse("1", true, false, 2, 0));
    try t.expectError(error.PingPongNeedsInflight, pp.parse("1", true, true, 1, 0));
    try t.expectError(error.PingPongNeedsInflight, pp.parse("1", true, true, 2, 3));
    for ([_][]const u8{ "", "true", "2", "01" }) |s| try t.expectError(error.BadAfdUsersPingPong, pp.parse(s, true, true, 2, 0));
}

test "ping_pong canonical prompt-lanes4/3 handshake retains profile and permits only two decode credits" {
    try t.expect(try pp.parse("1", true, true, 4, 4));
    try t.expect(try pp.parse("1", true, true, 3, 3));
    try t.expectError(error.PingPongNeedsInflight, pp.parse("1", true, true, 2, 4));
    try t.expectError(error.PingPongLanesUnsupported, pp.parse("1", true, true, 2, 2));
}

test "ping_pong split C2 1+1 C4 2+2, balances rows and maps physical slots" {
    const wins = [_]dsv.Window{ win(0, 4), win(1, 1), win(2, 4), win(3, 1) };
    const p = try pp.split(&wins);
    try t.expectEqual(@as(usize, 2), p.groups[0].len);
    try t.expectEqual(@as(usize, 2), p.groups[1].len);
    try t.expectEqual(@as(usize, 5), p.groups[0].rows);
    try t.expectEqual(@as(usize, 5), p.groups[1].rows);
    const two = try pp.split(wins[0..2]);
    try t.expectEqual(@as(usize, 1), two.groups[0].len);
    try t.expectEqual(@as(usize, 1), two.groups[1].len);
    const lone = try pp.split(wins[0..1]);
    try t.expectEqual(@as(usize, 0), lone.groups[1].len);
    const full = [_]dsv.Window{ win(0, 4), win(1, 4), win(2, 4), win(3, 4) };
    const max = try pp.split(&full);
    try t.expectEqual(@as(usize, 8), max.groups[0].rows);
    try t.expectEqual(@as(usize, 8), max.groups[1].rows);
}

const Sim = struct {
    turn: glm.afd_pairs.Turn = .{ .count = 3, .most = 2 },
    groups: [2]pp.Group = undefined,
    vals: [2][8]u32 = @splat(@splat(0)),
    starts: [4]u32 = .{ 11, 23, 31, 47 },
    kept: [4]usize = @splat(0),
    pending: usize = 0,
    max_pending: usize = 0,
    fronts_while_pending: usize = 0,
    requests: usize = 0,
    takes: usize = 0,
    failed: bool = false,
    closed: bool = false,
    behind: bool = false,
    fault: ?usize = null,
    cancel: ?*bool = null,

    pub fn alive(s: *Sim) !void {
        if (s.failed) return error.InjectedFailure;
    }
    pub fn fail(s: *Sim) void {
        s.failed = true;
        s.pending = 0;
    }
    pub fn begin(s: *Sim, p: *const pp.Plan) !bool {
        s.groups = p.groups;
        return s.behind;
    }
    pub fn open(s: *Sim, slot: u1, wins: []const dsv.Window, _: usize) !void {
        var at: usize = 0;
        for (wins) |w| {
            var value = s.starts[w.row.key.slot];
            for (0..w.rows()) |j| {
                value = value *% 5 +% (if (j == 0) w.row.token else w.drafts[j - 1]);
                s.vals[slot][at] = value;
                at += 1;
            }
        }
    }
    pub fn advance(s: *Sim, slot: u1, _: []const dsv.Window, R: usize, li: *usize) !?usize {
        if (li.* == 3) return null;
        if (s.pending > 0) s.fronts_while_pending += 1;
        for (s.vals[slot][0..R]) |*v| v.* = v.* *% 3 +% 7 +% @as(u32, @intCast(li.*));
        const out = li.*;
        li.* += 1;
        return out;
    }
    pub fn issue(s: *Sim, slot: u1, _: usize, li: usize) !void {
        try s.turn.request(slot, li);
        s.pending += 1;
        s.requests += 1;
        s.max_pending = @max(s.max_pending, s.pending);
        if (s.cancel) |flag| if (s.requests == 1) {
            flag.* = true;
        };
        if (s.fault) |at| if (s.requests == at) return error.InjectedFailure;
    }
    pub fn complete(s: *Sim, slot: u1, R: usize, li: usize) !void {
        try s.turn.take(slot, li);
        for (s.vals[slot][0..R]) |*v| v.* = v.* *% 11 +% 13;
        s.pending -= 1;
        s.takes += 1;
    }
    pub fn tailDraw(s: *Sim, slot: u1, wins: []const dsv.Window, _: usize, out: []u32) !void {
        var offset: usize = 0;
        for (wins) |w| {
            defer offset += w.rows();
            if (w.row.cancelled()) {
                @memset(out[offset .. offset + w.rows()], 0);
                continue;
            }
            @memcpy(out[offset .. offset + w.rows()], s.vals[slot][offset .. offset + w.rows()]);
            if (w.row.sampling) |sample| out[offset] +%= @intCast(sample.seed +% w.row.position +% 1);
            if (w.row.mask.len > 0) out[offset] = @intCast(@ctz(w.row.mask[0]));
        }
    }
    pub fn commit(s: *Sim, w: dsv.Window) !void {
        s.kept[w.row.key.slot] += 1;
    }
    pub fn end(s: *Sim) !void {
        if (!s.turn.done() or s.pending != 0) return error.OutOfStep;
        s.closed = true;
    }
};

fn serial(s: *const Sim, wins: []const dsv.Window, out: []u32) void {
    var at: usize = 0;
    for (wins) |w| {
        var history = s.starts[w.row.key.slot];
        for (0..w.rows()) |j| {
            history = history *% 5 +% (if (j == 0) w.row.token else w.drafts[j - 1]);
            var value = history;
            for (0..3) |li| value = (value *% 3 +% 7 +% @as(u32, @intCast(li))) *% 11 +% 13;
            out[at] = value;
            at += 1;
        }
    }
}

test "ping_pong CPU numerical scheduler C2/C4 on=off per window, solo=together, 16 rows never >8 a micro-batch" {
    const wins = [_]dsv.Window{ win(0, 4), win(1, 4), win(2, 4), win(3, 4) };
    for ([_]usize{ 2, 4 }) |n| {
        var s: Sim = .{};
        var out: [16]u32 = undefined;
        var expected: [16]u32 = undefined;
        serial(&s, wins[0..n], expected[0 .. n * 4]);
        try pp.run(Sim, &s, wins[0..n], out[0 .. n * 4]);
        try t.expectEqualSlices(u32, expected[0 .. n * 4], out[0 .. n * 4]);
        try t.expect(s.fronts_while_pending > 0 and s.max_pending == 2 and s.closed);
        try t.expectEqual(@as(usize, 6), s.requests);
        try t.expectEqual(s.requests, s.takes);
    }
}

fn cancelled(ctx: ?*anyopaque) bool {
    return @as(*bool, @ptrCast(@alignCast(ctx.?))).*;
}

test "ping_pong cancel with one micro-batch issued drains every promised exchange, no cancelled draw or commit" {
    var flag = false;
    var wins = [_]dsv.Window{ win(0, 1), win(1, 1) };
    wins[1].row.cancel_ctx = &flag;
    wins[1].row.cancel_check = cancelled;
    var s: Sim = .{ .cancel = &flag };
    var out: [2]u32 = undefined;
    try pp.run(Sim, &s, &wins, &out);
    try t.expectEqual(@as(u32, 0), out[1]);
    try t.expectEqual(@as(usize, 1), s.kept[0]);
    try t.expectEqual(@as(usize, 0), s.kept[1]);
    try t.expectEqual(@as(usize, 6), s.requests);
    try t.expectEqual(s.requests, s.takes);
    try t.expect(s.closed);
}

test "ping_pong failure after A0, B0 or last issue closes issuance, no draws or commits, clears results" {
    const wins = [_]dsv.Window{ win(0, 1), win(1, 1) };
    for ([_]usize{ 1, 2, 6 }) |at| {
        var s: Sim = .{ .fault = at };
        var out = [_]u32{ 99, 99 };
        try t.expectError(error.InjectedFailure, pp.run(Sim, &s, &wins, &out));
        try t.expect(s.failed and !s.closed and s.pending == 0);
        try t.expectEqualSlices(u32, &.{ 0, 0 }, &out);
        try t.expectEqual(@as(usize, 0), s.kept[0] + s.kept[1]);
        try t.expectEqual(at, s.requests);
    }
}

test "ping_pong adjacent pair tail barrier keeps FIFO credit and numerical identity" {
    const wins = [_]dsv.Window{ win(0, 3), win(1, 2) };
    var s: Sim = .{ .behind = true };
    var out: [5]u32 = undefined;
    var expected: [5]u32 = undefined;
    serial(&s, &wins, &expected);
    try pp.run(Sim, &s, &wins, &out);
    try t.expectEqualSlices(u32, &expected, &out);
    try t.expect(s.closed and s.max_pending == 2);
}

test "ping_pong sampled and schema rows remain one row with own seed/position/mask beside drafted windows" {
    var wins = [_]dsv.Window{ win(0, 3), win(1, 1), win(2, 1), win(3, 2) };
    wins[1].row.sampling = .{ .seed = 123, .temperature = 0.7, .top_k = 0, .top_p = 1, .min_p = 0 };
    wins[2].row.mask = &.{8};
    var s: Sim = .{};
    var expected: [7]u32 = undefined;
    serial(&s, &wins, &expected);
    expected[3] +%= 123 + 12;
    expected[4] = 3;
    var out: [7]u32 = undefined;
    try pp.run(Sim, &s, &wins, &out);
    try t.expectEqualSlices(u32, &expected, &out);
    wins[1].drafts = &drafts;
    try t.expectError(error.UsersDraftsUnsupported, pp.split(&wins));
}

const fixture = @import("shared_verify_rec_test.zig");
test "ping_pong additional copies exact served bytes, no allocation when off" {
    var rec: fixture.Rec = undefined;
    try rec.init(4, .drafted);
    defer rec.deinit();
    try t.expectEqual(@as(u64, 0), try pp.bytes(&rec.m, false));
    rec.a.paired = true;
    rec.a.inflight = 2;
    const before = try rec.users.bytes();
    try rec.users.usePingPong();
    const after = try rec.users.bytes();
    try t.expectEqual(try pp.bytes(&rec.m, true), after.total - before.total);
    std.debug.print("ping_pong copies fixture={d} B (cut5), OFF=0 B\n", .{after.total - before.total});
}

const served_fixture = @import("shared_verify_test.zig");
test "ping_pong LaneHost C2 pool/interleave greedy drafted=undrafted, on=off, solo=together" {
    try served_fixture.checkPingPongReplies(2, false);
}
test "ping_pong LaneHost C4 pool/interleave greedy drafted=undrafted, on=off, solo=together" {
    try served_fixture.checkPingPongReplies(4, false);
}
test "ping_pong LaneHost sampled keyed row beside drafted rows" {
    try served_fixture.checkPingPongReplies(4, true);
}
test "ping_pong LaneHost lone to shared to lone transitions" {
    try served_fixture.checkPingPongTransition(false);
}
test "ping_pong LaneHost cancel after A0 issued releases both slots" {
    try served_fixture.checkPingPongTransition(true);
}
test "ping_pong LaneHost failure after A0 issued latches and releases every slot" {
    try served_fixture.checkPingPongFailure();
}

test "ping_pong partition exhaustive widths1..4 for C2/C3/C4, stable cap and once-only mapping" {
    for (1..5) |a| for (1..5) |b| for (1..5) |c| for (1..5) |d| {
        const wins = [_]dsv.Window{ win(0, a), win(1, b), win(2, c), win(3, d) };
        for ([_]usize{ 2, 3, 4 }) |n| {
            const p = try pp.split(wins[0..n]);
            var seen: [4]bool = @splat(false);
            for (p.groups) |g| {
                try t.expect(g.rows <= 8 and g.len <= 2);
                for (g.index[0..g.len]) |i| {
                    try t.expect(!seen[i]);
                    seen[i] = true;
                }
            }
            for (seen[0..n]) |s| try t.expect(s);
            const again = try pp.split(wins[0..n]);
            for (p.groups, again.groups) |x, y| try t.expectEqualSlices(usize, x.index[0..x.len], y.index[0..y.len]);
        }
    };
}

test "ping_pong original slice and pair use the same halves route at every <=8 row count" {
    const ae = glm.afd_expert;
    const ready = [_]bool{ true, false, true };
    const suh = [_]?u64{ null, null, null };
    const node: ae.g53.Node = .{ .set = undefined, .plan = undefined, .max_rows = 2048, .ready = &ready, .suh = &suh };
    for (1..9) |R| for (0..ready.len) |pos| {
        const before: ae.Forward = .{ .fseq = 1, .rows = R, .mode = .slice, .first = 0, .count = 2 };
        var after = before;
        after.mode = .pair;
        after.slot = 1;
        try t.expectEqual(ae.rt.route(&node, null, .{}, pos, before), ae.rt.route(&node, null, .{}, pos, after));
        try t.expectEqual(ae.rt.route(null, null, .{}, pos, before), ae.rt.route(null, null, .{}, pos, after));
    };
}

test "ping_pong served-shape extra device memory exactly404992B" {
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, @embedFile("afd_attn_fixture"), .{});
    defer parsed.deinit();
    var m = try glm.model.Model.parse(t.allocator, parsed.value.object.get("config_json").?.string, 1, 0);
    m.mtp_layers = 1;
    try t.expectEqual(@as(u64, 404992), try pp.bytes(&m, true));
}

const envcall = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};
test "ping_pong with batched_drafts C2/C4 LaneHost pool/interleave replies exact against off and solo" {
    try @import("shared_verify_test.zig").checkPingPongComposition(2, false, true);
    try @import("shared_verify_test.zig").checkPingPongComposition(4, false, true);
}

test "ping_pong actual serve plan default-off, C2/C4 with pool/interleave and dependency refusals before load" {
    try @import("env_harness.zig").detach();
    const ns = [_][*:0]const u8{ "TF_AFD_USERS", "TF_AFD_USERS_DRAFTS", "TF_AFD_MTP", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE", "TF_AFD_USERS_SHARED", "TF_AFD_USERS_SHARED_DRAFTS", "TF_AFD_USERS_PING_PONG", "TF_AFD_PAIRS", "TF_AFD_INFLIGHT", "TF_AFD_LANES" };
    var old: [ns.len]?[:0]u8 = @splat(null);
    for (ns, &old) |n, *o| o.* = if (std.c.getenv(n)) |v| try t.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer for (ns, old) |n, o| {
        if (o) |v| {
            _ = envcall.setenv(n, v, 1);
            t.allocator.free(v);
        } else _ = envcall.unsetenv(n);
    };
    for (ns) |n| _ = envcall.unsetenv(n);
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]u8{ 2, 4 }) |users| {
        const num: [:0]const u8 = if (users == 2) "2" else "4";
        const flags = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", num, "--temperature", "0" });
        _ = envcall.setenv("TF_AFD_USERS", num, 1);
        for ([_][*:0]const u8{ "TF_AFD_MTP", "TF_AFD_USERS_DRAFTS", "TF_AFD_USERS_SHARED", "TF_AFD_USERS_SHARED_DRAFTS", "TF_AFD_USERS_INTERLEAVE" }) |n| _ = envcall.setenv(n, "1", 1);
        _ = envcall.setenv("TF_AFD_POOL_TOKENS", "655360", 1);
        _ = envcall.unsetenv("TF_AFD_USERS_PING_PONG");
        const config = "{\"max_position_embeddings\":1048576}";
        const off = try served.plan(a, flags, config);
        try t.expect(!off.users_ping_pong);
        _ = envcall.setenv("TF_AFD_USERS_PING_PONG", "0", 1);
        try t.expect(!(try served.plan(a, flags, config)).users_ping_pong);
        _ = envcall.setenv("TF_AFD_USERS_PING_PONG", "1", 1);
        _ = envcall.unsetenv("TF_AFD_PAIRS");
        _ = envcall.unsetenv("TF_AFD_INFLIGHT");
        try t.expectError(error.PingPongNeedsPairs, served.plan(a, flags, config));
        _ = envcall.setenv("TF_AFD_PAIRS", "1", 1);
        _ = envcall.unsetenv("TF_AFD_INFLIGHT");
        try t.expectError(error.PingPongNeedsInflight, served.plan(a, flags, config));
        _ = envcall.setenv("TF_AFD_INFLIGHT", "2", 1);
        const on = try served.plan(a, flags, config);
        try t.expect(on.users_ping_pong and on.users_shared_drafts and on.pool > 0 and on.interleave != null);
        try t.expectEqual(users, on.users);
        try t.expectEqual(@as(u8, 8), on.stageRows());
        _ = envcall.setenv("TF_AFD_LANES", "4", 1);
        _ = envcall.unsetenv("TF_AFD_INFLIGHT");
        const canonical = try served.plan(a, flags, config);
        try t.expect(canonical.users_ping_pong and canonical.users_shared_drafts);
        try t.expectEqual(@as(u8, 8), canonical.stageRows());
        _ = envcall.setenv("TF_AFD_INFLIGHT", "2", 1);
        try t.expectError(error.BadAfdLanes, served.plan(a, flags, config));
        _ = envcall.setenv("TF_AFD_LANES", "0", 1);
        _ = envcall.setenv("TF_AFD_USERS_SHARED_DRAFTS", "0", 1);
        try t.expectError(error.PingPongNeedsSharedDrafts, served.plan(a, flags, config));
    }
}
