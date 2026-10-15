//! Recorded real CUDA users: buffer operands/State identity, exact R=C wire counts, fail fencing; no CUDA numerics.
//! Fixture scaffolding follows Ash Hart's cuda_afd_engine_test.zig; inherited NOTICE credits apply.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const schedule = glm.afd_users_round;
const t = std.testing;
const gpa = t.allocator;
const proto = cuda.xchg.proto;
pub const Rec = struct {
    parsed: std.json.Parsed(std.json.Value),
    m: glm.model.Model,
    w: glm.views.Weights,
    e: *glm.engine.Engine,
    a: glm.afd_attn.Afd,
    at: glm.afd_engine.Attn,
    space: glm.trace.Space,
    trace: glm.trace.Trace,
    f: glm.fwd.Fwd,
    users: glm.afd_users.Users,
    region: []align(4096) u8,
    status: []u32,
    node: cuda.xchg.Node,
    posts: usize = 0,
    takes: usize = 0,
    fail_take: ?usize = null,
    cancel_in_take: ?*bool = null,
    headers: [128]proto.Header = undefined,
    fn init(r: *Rec, n: u8) !void {
        try r.initRows(n, 64, 8);
    }
    pub fn initRows(r: *Rec, n: u8, prefill_rows: usize, max_rows: u32) !void {
        r.posts = 0;
        r.takes = 0;
        r.fail_take = null;
        r.cancel_in_take = null;
        r.parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
        r.m = try glm.model.Model.parse(gpa, r.parsed.value.object.get("config_json").?.string, 1, 0);
        try r.m.cut(5);
        r.m.mtp_layers = 0;
        r.w = try glm.views.rank(gpa, &r.m);
        r.e = try glm.engine.Engine.create(gpa, &r.m, &r.w, 2560, 8, prefill_rows, true);
        r.a = try glm.afd_attn.Afd.init(&r.m, prefill_rows);
        r.space = .{ .gpa = gpa };
        try glm.views.walk(&r.w, glm.views.Path.init("w"), &r.space, glm.trace.placeAll);
        try r.e.walk(&r.space, glm.trace.placeAll);
        try r.e.walkTmp(&r.space, glm.trace.placeAll);
        r.e.tmp.space = &r.space;
        try r.space.place("b.afd_recv", &r.a.afd_recv);
        r.trace = glm.trace.Trace.init(gpa);
        r.f = glm.fwd.Fwd.recording(&r.m, &r.trace);
        r.at = try glm.afd_engine.Attn.init(r.e, &r.a);
        r.users = try glm.afd_users.Users.init(gpa, &r.at, &r.f, n);
        try r.users.walk(&r.space, glm.trace.placeAll);
        const settings: proto.Settings = .{ .layout = r.a.layout, .max_rows = max_rows, .ports = 1 };
        r.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast((proto.Region{ .slot = settings.slotBytes() }).total()));
        @memset(r.region, 0);
        r.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(r.status, 0);
        r.node = try cuda.xchg.Node.init(gpa, t.io, .attn, .{ .settings = settings, .warn_failures = true }, .{ .region = r.region, .region_dev = @intFromPtr(r.region.ptr), .status = r.status, .status_dev = @intFromPtr(r.status.ptr) });
        r.a.wire = .{ .node = &r.node, .host = .{ .ctx = r, .post = post, .take = take } };
        const positions = [_]usize{ 511, 2047, 513, 2048 };
        for (0..n) |i| {
            r.users.generation[i] = 1;
            const st = if (i == 0) &r.e.st else &r.users.states[i - 1];
            st.pos = positions[i];
            st.parity = @intCast(i % 2);
        }
    }
    pub fn deinit(r: *Rec) void {
        r.node.deinit();
        gpa.free(r.region);
        gpa.free(r.status);
        r.users.deinit(gpa);
        r.trace.deinit();
        r.space.deinit();
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.parsed.deinit();
    }
    pub fn post(p: *anyopaque, _: *cuda.xchg.Node, h: *const [proto.HDR_BYTES]u8, pos: usize, n: usize) !void {
        const r: *Rec = @ptrCast(@alignCast(p));
        const hdr = try proto.Header.read(h);
        try t.expectEqual(@as(i32, @intCast(n)), hdr.rows);
        try t.expectEqual(@as(i32, @intCast(pos)), hdr.layer);
        r.headers[r.posts] = hdr;
        r.posts += 1;
    }
    pub fn take(p: *anyopaque, node: *cuda.xchg.Node, _: usize, _: usize) !void {
        const r: *Rec = @ptrCast(@alignCast(p));
        r.takes += 1;
        if (r.cancel_in_take) |flag| flag.* = true;
        if (r.fail_take == r.takes) node.fail(.{ .why = .silent, .peer = 1, .seq = @intCast(r.posts) });
    }
};
fn rows(n: u8) [4]schedule.Row {
    const pos = [_]usize{ 511, 2047, 513, 2048 };
    var out: [4]schedule.Row = undefined;
    for (0..n) |i| out[i] = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = @intCast(10 + i), .position = pos[i] };
    return out;
}

// Delegates to the REAL device methods, capturing each user's front operands without changing the production trace.
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
fn recorded(n: u8) !void {
    var together: Rec = undefined;
    try together.init(n);
    defer together.deinit();
    var serial: Rec = undefined;
    try serial.init(n);
    defer serial.deinit();
    var a: Capture = .{ .u = &together.users, .trace = &together.trace };
    var b: Capture = .{ .u = &serial.users, .trace = &serial.trace };
    const rs = rows(n);
    var out: [4]u32 = undefined;
    try schedule.run(Capture, &a, rs[0..n], out[0..n]);
    for (rs[0..n], 0..) |r, i| try schedule.run(Capture, &b, &.{r}, out[i..][0..1]);
    for (0..n) |i| for (0..5) |li| try glm.serve_rec.expectSame(&together.space, together.trace.events.items[a.starts[i][li]..a.ends[i][li]], &serial.space, serial.trace.events.items[b.starts[i][li]..b.ends[i][li]]);
    try t.expectEqual(@as(usize, 2), together.posts);
    try t.expectEqual(@as(usize, 2 * n), serial.posts);
    for (together.headers[0..together.posts], 0..) |h, i| {
        try t.expectEqual(@as(i32, n), h.rows);
        try t.expectEqual(@as(i32, @intCast(i)), h.layer);
        try t.expectEqual(proto.Mode.eager, h.mode);
    }
    for (0..n) |i| {
        try t.expect(together.users.saved[i].x.ptr != together.e.buf.x.ptr);
        for (i + 1..n) |j| try t.expect(together.users.saved[i].x.ptr != together.users.saved[j].x.ptr);
        try t.expectEqual(rs[i].position + 1, together.users.position(rs[i].key));
    }
    std.debug.print("concurrent_users real recording C{d}: {d} combined exchanges vs {d} independent; {d} fronts exact op/grid/operand/buffer/scalar identity, no hardware numerics\n", .{ n, together.posts, serial.posts, n * 5 });
}
test "concurrent_users real CUDA C2 own State front operands match serial, wire R2 exact" {
    try recorded(2);
}
test "concurrent_users real CUDA C4 same mapping, own State front operands match serial, wire R4 exact" {
    try recorded(4);
}
fn failure(take_no: usize) !void {
    var r: Rec = undefined;
    try r.init(2);
    defer r.deinit();
    const rs = rows(2);
    var out: [2]u32 = undefined;
    r.fail_take = take_no;
    try t.expectError(error.ExpertFailed, r.users.round(rs[0..2], &out));
    try r.f.flush();
    for (r.trace.events.items) |ev| try t.expect(!std.mem.eql(u8, ev.name, "torch.topk") and !std.mem.eql(u8, ev.name, "torch.cpu") and !std.mem.eql(u8, ev.name, "torch.fill_"));
    for (rs[0..2]) |row| try t.expectEqual(row.position, r.users.position(row.key));
    const posts = r.posts;
    r.trace.reset();
    try t.expectError(error.ExpertFailed, r.users.round(rs[0..2], &out));
    try t.expectError(error.ExpertFailed, r.users.prefill(.{ .slot = 0, .generation = 2 }, &.{ 1, 2 }));
    try r.f.flush();
    try t.expectEqual(@as(usize, 0), r.trace.events.items.len);
    try t.expectEqual(posts, r.posts);
}
test "concurrent_users real transport first-layer failure: sticky no new sends/sample/commit" {
    try failure(1);
}
test "concurrent_users real transport last-layer failure: sticky no new sends/sample/commit" {
    try failure(2);
}

test "concurrent_users full262144 MTP-loaded extra State and row workspace forecast metadata only" {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    defer parsed.deinit();
    var m = try glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
    m.mtp_layers = 1;
    const served = @import("afd_attn_users.zig");
    const c2 = try served.extra(gpa, &m, 262208, 2048, 2);
    const c4 = try served.extra(gpa, &m, 262208, 2048, 4);
    try t.expectEqual(3 * c2.states, c4.states);
    try t.expect(c4.rows > c2.rows);
    std.debug.print("concurrent_users projection full262144 MTP_loaded=1 C2 states={d} rows={d} total={d}; C4 states={d} rows={d} total={d}\n", .{ c2.states, c2.rows, c2.total, c4.states, c4.rows, c4.total });
}

pub const OwnLog = struct {
    values: [256]schedule.Observation = undefined,
    n: usize = 0,
    fn note(p: *anyopaque, value: schedule.Observation) void {
        const log: *OwnLog = @ptrCast(@alignCast(p));
        std.debug.assert(log.n < log.values.len);
        log.values[log.n] = value;
        log.n += 1;
    }
    pub fn attach(log: *OwnLog, r: *Rec) void {
        r.users.observations = true;
        r.users.observer = .{ .ctx = log, .note = note };
    }
};

test "concurrent_users_r2 real R2 commit ownership snapshots and settled State, observer OFF calls zero and operands exact" {
    var a: Rec = undefined;
    try a.init(2);
    defer a.deinit();
    var b: Rec = undefined;
    try b.init(2);
    defer b.deinit();
    var log: OwnLog = .{};
    log.attach(&a);
    var off: OwnLog = .{};
    off.attach(&b);
    b.users.observations = false;
    const rs = rows(2);
    var ao: [2]u32 = undefined;
    var bo: [2]u32 = undefined;
    try a.users.round(rs[0..2], &ao);
    try b.users.round(rs[0..2], &bo);
    try a.f.flush();
    try b.f.flush();
    try glm.serve_rec.expectSame(&a.space, a.trace.events.items, &b.space, b.trace.events.items);
    try t.expectEqual(@as(usize, 0), off.n);
    try t.expectEqual(@as(usize, 6), log.n);
    for (0..2) |i| {
        const before = log.values[2 * i];
        const after = log.values[2 * i + 1];
        try t.expectEqual(.commit_before, before.event);
        try t.expectEqual(.commit_after, after.event);
        try t.expectEqual(@as(u8, @intCast(i)), before.slot);
        try t.expectEqual(before.slot, before.state);
        try t.expectEqual(@as(u64, 1), after.generation);
        try t.expectEqual(rs[i].position, before.position);
        try t.expectEqual(before.position + 1, after.position);
        try t.expectEqual(@as(u1, 1 - before.parity), after.parity);
        const settled = log.values[4 + i];
        try t.expectEqual(.round_settled, settled.event);
        try t.expectEqual(after.position, settled.position);
        try t.expectEqual(@as(usize, 2), settled.commits);
    }
}

test "concurrent_users_r2 actual transport failure before commit snapshots leave both positions unchanged" {
    for (1..3) |take_no| {
        var r: Rec = undefined;
        try r.init(2);
        defer r.deinit();
        var log: OwnLog = .{};
        log.attach(&r);
        r.fail_take = take_no;
        const rs = rows(2);
        var out: [2]u32 = undefined;
        try t.expectError(error.ExpertFailed, r.users.round(rs[0..2], &out));
        try t.expectEqual(@as(usize, 2), log.n);
        for (log.values[0..log.n], rs[0..2]) |value, row| {
            try t.expectEqual(.failure_before_commit, value.event);
            try t.expectEqual(row.position, value.position);
            try t.expectEqual(@as(usize, 0), value.commits);
        }
    }
}

test "concurrent_users_r2 real cancelled row no commit, unaffected peer settles, retirement generation reuse" {
    var r: Rec = undefined;
    try r.init(2);
    defer r.deinit();
    var log: OwnLog = .{};
    log.attach(&r);
    var rs = rows(2);
    var flag = false;
    const Check = struct {
        fn yes(ctx: ?*anyopaque) bool {
            const value: *bool = @ptrCast(@alignCast(ctx.?));
            return value.*;
        }
    };
    rs[0].cancel_ctx = &flag;
    rs[0].cancel_check = Check.yes;
    r.cancel_in_take = &flag;
    var out: [2]u32 = undefined;
    try r.users.round(rs[0..2], &out);
    try t.expectEqual(@as(usize, 6), log.n);
    try t.expectEqual(.cancel_before_sample, log.values[0].event);
    try t.expectEqual(.cancel_before_commit, log.values[1].event);
    try t.expectEqual(@as(u8, 1), log.values[2].slot);
    try t.expectEqual(.commit_before, log.values[2].event);
    try t.expectEqual(rs[0].position, r.users.position(rs[0].key));
    try t.expectEqual(rs[1].position + 1, r.users.position(rs[1].key));
    r.users.observeRelease(rs[0].key, "cancelled");
    r.users.release(rs[0].key);
    try t.expectEqual(.cancel_retired, log.values[6].event);
    try t.expectEqual(.retire, log.values[7].event);
    _ = try r.users.prefill(.{ .slot = 0, .generation = 2 }, &.{ 1, 2, 3 });
    try t.expectEqual(.prefill_before, log.values[8].event);
    try t.expectEqual(.prefill_after, log.values[9].event);
    try t.expectEqual(@as(u64, 2), log.values[9].generation);
    try t.expectEqual(@as(usize, 3), log.values[9].position);
    try t.expectEqual(rs[1].position + 1, r.users.position(rs[1].key));
}

test "concurrent_users_r2 real native recording runs the original 96-prompt 16-token warm state machine" {
    const lanes = @import("lanes");
    const backend = @import("glm_users");
    var r: Rec = undefined;
    try r.initRows(2, 96, 96);
    defer r.deinit();
    @memset(&r.users.generation, 0);
    var b: backend.Users(glm.afd_users.Users) = .{ .steps = &r.users, .count = 2, .capacity = 2560, .reach = 2560 };
    var ids: [96]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1000 + i);
    var warm = try lanes.Stream.init(gpa, .{ .id = "warm", .prompt = &ids, .max_new = 16, .drafts = false });
    defer warm.deinit(gpa);
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    try core.addStream(&warm);
    while (core.activeCount() > 0) try core.step();
    try @import("afd_attn_users.zig").warmComplete(warm.emitted(), warm.reason, r.users.prefills, r.users.rounds, r.at.forwards, r.users.exchanges, 2, r.at.failure() != null);
    try t.expectEqual(ids.len + @as(usize, 15), r.users.position(.{ .slot = 0, .generation = 1 }));
    std.debug.print("concurrent_users_r2 actual warm fixture prompt96 emitted{d} prefills{d} rounds{d} forwards{d} exchanges{d} reason{s}; recorder only\n", .{ warm.emitted().len, r.users.prefills, r.users.rounds, r.at.forwards, r.users.exchanges, @tagName(warm.reason) });
}

test "concurrent_users_r2 warm receipt native completion refuses every incomplete count and failure" {
    const warm = @import("afd_attn_users.zig");
    const tokens: [16]u32 = @splat(7);
    try warm.warmComplete(&tokens, .length, 1, 15, 16, 630, 42, false);
    try t.expectError(error.WarmIncomplete, warm.warmComplete(tokens[0..15], .length, 1, 15, 16, 630, 42, false));
    try t.expectError(error.WarmIncomplete, warm.warmComplete(&tokens, .cancelled, 1, 15, 16, 630, 42, false));
    try t.expectError(error.WarmIncomplete, warm.warmComplete(&tokens, .length, 0, 15, 16, 630, 42, false));
    try t.expectError(error.WarmIncomplete, warm.warmComplete(&tokens, .length, 1, 14, 16, 630, 42, false));
    try t.expectError(error.WarmIncomplete, warm.warmComplete(&tokens, .length, 1, 15, 15, 630, 42, false));
    try t.expectError(error.WarmIncomplete, warm.warmComplete(&tokens, .length, 1, 15, 16, 629, 42, false));
    try t.expectError(error.WarmIncomplete, warm.warmComplete(&tokens, .length, 1, 15, 16, 630, 0, false));
    try t.expectError(error.WarmIncomplete, warm.warmComplete(&tokens, .length, 1, 15, 16, 630, 42, true));
}

test {
    _ = @import("afd_users_retire_test.zig"); // concurrent_users_r4: failed-round and cancel release labels
    _ = @import("structured_users_users_test.zig"); // structured_users: structured lanes with TF_AFD_USERS=2
}
