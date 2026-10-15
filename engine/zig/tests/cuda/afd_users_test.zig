//! concurrent_users CPU fixtures: exact integer State/data oracle, not CUDA numerical or model-output certification.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const glm = @import("glm_cuda");
const users = @import("glm_users");
const scheduling = glm.afd_users_round;
const Key = scheduling.Key;
const Row = scheduling.Row;
const gpa = std.testing.allocator;
const expect = std.testing.expect;
const eq = std.testing.expectEqual;
const envcall = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

test "concurrent_users real device and served path compile without GPU" {
    _ = &glm.afd_users.Users.round;
    _ = &glm.afd_users.Users.prefill;
    _ = &@import("afd_attn_users.zig").run;
    _ = &@import("afd_attn_serve.zig").admit;
    _ = &scheduling.parse;
    _ = @import("afd_users_device_test.zig");
}

test {
    _ = @import("afd_users_fill_test.zig"); // prefill_interleave: the users backend a step a turn through the real LaneHost
    _ = @import("afd_users_fill_device_test.zig"); // prefill_interleave: the recorded device Users, stepwise = whole, interleaved
}

test "concurrent_users switch exact tokens and serve flags reject unsupported requests before weights" {
    try eq(@as(u8, 1), try scheduling.parse(null));
    for ([_][]const u8{ "1", "2", "4" }, [_]u8{ 1, 2, 4 }) |v, n| try eq(n, try scheduling.parse(v));
    for ([_][]const u8{ "", "0", "3", "8", "02", "2 ", " 2", "true" }) |v| try std.testing.expectError(error.BadAfdUsers, scheduling.parse(v));
    const old = if (std.c.getenv("TF_AFD_USERS")) |p| try gpa.dupeSentinel(u8, std.mem.span(p), 0) else null;
    defer {
        if (old) |v| {
            _ = envcall.setenv("TF_AFD_USERS", v, 1);
            gpa.free(v);
        } else {
            _ = envcall.unsetenv("TF_AFD_USERS");
        }
    }
    _ = envcall.setenv("TF_AFD_USERS", "2", 1);
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const flags = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--no-drafts", "--drafts", "0", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    const p = try served.plan(a, flags, cfg);
    try eq(@as(u8, 2), p.users);
    try eq(@as(usize, 262208), p.capacity);
    try eq(@as(u8, 2), (try served.plan(a, try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read" }), cfg)).users);
    try std.testing.expectError(error.GreedyOnly, served.plan(a, try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--no-drafts", "--temperature", "1" }), cfg));
    try std.testing.expectError(error.UsersContext, served.plan(a, try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--no-drafts", "--context", "262145" }), cfg));
    _ = envcall.setenv("TF_AFD_USERS", "3", 1);
    try std.testing.expectError(error.BadAfdUsers, served.plan(a, flags, cfg));
}

const Gate = struct {
    entered: std.atomic.Value(bool) = .init(false),
    open: std.atomic.Value(bool) = .init(false),
};
fn waitFlag(flag: *std.atomic.Value(bool)) !void {
    for (0..3000) |_| {
        if (flag.load(.acquire)) return;
        const ts: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    return error.WaitTimeout;
}
const Got = struct {
    n: usize = 0,
    reason: api.Reason = .stop,
    done: std.atomic.Value(bool) = .init(false),
    fn event(p: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const g: *Got = @ptrCast(@alignCast(p));
        switch (ev.*) {
            .tokens => |ts| g.n += ts.len,
            .finished => |end| {
                g.reason = end.reason;
                g.done.store(true, .release);
            },
            else => {},
        }
    }
    fn sink(g: *Got) api.Sink {
        return .{ .ctx = g, .event = event };
    }
};
test "concurrent_users actual LaneHost queued cancellation and two live users, no cancelled prefill, later ordinary reuse" {
    var r: Run = .{};
    try r.init(2);
    defer r.deinit();
    var gate: Gate = .{};
    r.fake.gate = &gate;
    var host = api.LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 2, .context_window = 262144, .drafts = "off", .greedy_users = true });
    try host.start();
    var up = true;
    defer if (up) host.stop();
    defer gate.open.store(true, .release);
    const engine = host.engine();
    var a: Got = .{};
    var b: Got = .{};
    var c: Got = .{};
    var d: Got = .{};
    const req: api.Request = .{ .prompt = &.{ 1, 2, 3, 4 }, .max_tokens = 4, .drafts = false };
    try engine.submit(1, &req, a.sink());
    try waitFlag(&gate.entered);
    try engine.submit(2, &req, b.sink());
    try engine.submit(3, &req, c.sink());
    engine.cancel(3);
    gate.open.store(true, .release);
    try waitFlag(&a.done);
    try waitFlag(&b.done);
    try waitFlag(&c.done);
    try eq(api.Reason.cancelled, c.reason);
    try eq(@as(usize, 0), c.n);
    try eq(@as(usize, 4), a.n);
    try eq(@as(usize, 4), b.n);
    try engine.submit(4, &req, d.sink());
    try waitFlag(&d.done);
    host.stop();
    up = false;
    try eq(@as(usize, 4), d.n);
    try eq(@as(usize, 3), r.fake.prefills);
    var combined = false;
    for (r.fake.log.items) |e| if (e.kind == .exchange and e.rows == 2) {
        combined = true;
    };
    try expect(combined);
}

const Fake = struct {
    const State = struct { generation: u64 = 0, position: usize = 0, parity: u1 = 0, value: u32 = 0 };
    states: [4]State = @splat(.{}),
    hidden: [4]u32 = @splat(0),
    gathered: [4]u32 = @splat(0),
    result: [4]u32 = @splat(0),
    log: std.ArrayList(struct { kind: enum { begin, front, gather, exchange, scatter, sample, commit }, slot: u8, generation: u64, layer: usize, rows: usize, value: u32 }) = .empty,
    failed: bool = false,
    fail_layer: ?usize = null,
    fail_settle: bool = false,
    fail_after_sample: bool = false,
    cancel: ?*bool = null,
    cancel_layer: ?usize = null,
    gate: ?*Gate = null,
    prefills: usize = 0,
    releases: usize = 0,
    rounds: usize = 0,

    fn note(f: *Fake, kind: @FieldType(@TypeOf(f.log.items[0]), "kind"), row: ?Row, li: usize, n: usize, value: u32) !void {
        try f.log.append(gpa, .{ .kind = kind, .slot = if (row) |r| r.key.slot else 255, .generation = if (row) |r| r.key.generation else 0, .layer = li, .rows = n, .value = value });
    }
    fn deinit(f: *Fake) void {
        f.log.deinit(gpa);
    }
    pub fn prefill(f: *Fake, key: Key, ids: []const u32) !u32 {
        try f.alive();
        var v: u32 = 17;
        for (ids) |x| v = v *% 31 +% x;
        f.states[key.slot] = .{ .generation = key.generation, .position = ids.len, .value = v };
        f.prefills += 1;
        return v % 997;
    }
    pub fn position(f: *Fake, key: Key) usize {
        std.debug.assert(f.states[key.slot].generation == key.generation);
        return f.states[key.slot].position;
    }
    pub fn release(f: *Fake, key: Key) void {
        std.debug.assert(f.states[key.slot].generation == key.generation);
        f.releases += 1;
    }
    pub fn round(f: *Fake, rows: []const Row, out: []u32) !void {
        f.rounds += 1;
        return scheduling.run(Fake, f, rows, out);
    }
    pub fn alive(f: *Fake) !void {
        if (f.failed) return error.ExpertFailed;
    }
    pub fn begin(f: *Fake, rows: []const Row) !void {
        if (f.gate) |gate| {
            gate.entered.store(true, .release);
            try waitFlag(&gate.open);
            f.gate = null;
        }
        for (rows) |row| {
            try eq(row.key.generation, f.states[row.key.slot].generation);
            try eq(row.position, f.position(row.key));
        }
        try f.note(.begin, null, 0, rows.len, 0);
    }
    pub fn open(f: *Fake, row: Row, _: usize) !void {
        const st = f.states[row.key.slot];
        f.hidden[row.key.slot] = st.value +% row.token +% @as(u32, @intCast(st.position));
    }
    pub fn layerCount(_: *Fake) usize {
        return 6;
    }
    pub fn isMoe(_: *Fake, li: usize) bool {
        return li >= 2;
    }
    pub fn front(f: *Fake, row: Row, _: usize, li: usize) !void {
        const st = f.states[row.key.slot];
        // Unequal positions/parities and DSA block/pool boundaries enter each user's OWN front.
        f.hidden[row.key.slot] = f.hidden[row.key.slot] *% @as(u32, @intCast(li + 3)) +% st.parity +% @as(u32, @intCast(st.position / 512 + st.position / 4));
        try f.note(.front, row, li, 1, f.hidden[row.key.slot]);
    }
    pub fn gather(f: *Fake, row: Row, i: usize) !void {
        f.gathered[i] = f.hidden[row.key.slot];
        try f.note(.gather, row, 0, 1, f.gathered[i]);
    }
    pub fn exchange(f: *Fake, li: usize, n: usize) !void {
        try f.note(.exchange, null, li, n, 0);
        if (f.cancel_layer == li) if (f.cancel) |flag| {
            flag.* = true;
        };
        if (f.fail_layer == li) {
            f.failed = true;
            return error.ExpertFailed;
        }
        for (f.gathered[0..n], f.result[0..n]) |x, *out| {
            const expert0 = x *% 3;
            const expert1 = x *% 5;
            const shared = x *% 7;
            out.* = (expert0 +% expert1) +% shared;
        }
    }
    pub fn scatter(f: *Fake, row: Row, i: usize, n: usize) !void {
        f.hidden[row.key.slot] +%= f.result[i];
        try f.note(.scatter, row, 0, n, f.hidden[row.key.slot]);
    }
    pub fn tail(_: *Fake, _: Row, _: usize) !void {}
    pub fn settle(f: *Fake) !void {
        if (f.fail_settle) {
            f.failed = true;
            return error.ExpertFailed;
        }
        try f.alive();
    }
    pub fn sample(f: *Fake, row: Row, _: usize) !u32 {
        const token = f.hidden[row.key.slot] % 997;
        try f.note(.sample, row, 0, 1, token);
        if (f.fail_after_sample) f.failed = true;
        return token;
    }
    pub fn commit(f: *Fake, row: Row, _: usize) !void {
        const st = &f.states[row.key.slot];
        st.value = f.hidden[row.key.slot];
        st.position += 1;
        st.parity = 1 - st.parity;
        try f.note(.commit, row, 0, 1, st.value);
    }
};

const Back = users.Users(Fake);
const Run = struct {
    fake: Fake = .{},
    back: Back = undefined,
    cfg: lanes.Config = undefined,
    core: lanes.Engine = undefined,
    clock: lanes.fake.FixedClock = .{},
    fn init(r: *Run, n: u8) !void {
        r.back = .{ .steps = &r.fake, .count = n, .capacity = 262208, .reach = 262208 };
        r.cfg = try lanes.Config.init(gpa, r.back.facts(), 1, 0);
        r.core = lanes.Engine.init(gpa, &r.cfg, r.back.backend(), r.clock.clock());
    }
    fn deinit(r: *Run) void {
        r.core.deinit();
        r.cfg.deinit(gpa);
        r.fake.deinit();
    }
};

fn stream(id: []const u8, len: usize, max_new: u32, cancel: ?*bool) !lanes.Stream {
    const ids = try gpa.alloc(u32, len);
    defer gpa.free(ids);
    for (ids, 0..) |*x, i| x.* = @intCast(i % 19 + id[0]);
    const Check = struct {
        fn check(ctx: *anyopaque) bool {
            const p: *bool = @ptrCast(@alignCast(ctx));
            return p.*;
        }
    };
    return lanes.Stream.init(gpa, .{ .id = id, .prompt = ids, .max_new = max_new, .drafts = false, .cancel_check = if (cancel) |c| .{ .ptr = c, .check = Check.check } else null });
}

fn together(n: u8) !void {
    var r: Run = .{};
    try r.init(n);
    defer r.deinit();
    var solo: [4]Run = @splat(.{});
    var ss: [4]lanes.Stream = undefined;
    var ts: [4]lanes.Stream = undefined;
    const lengths = [_]usize{ 511, 2047, 513, 2048 };
    const names = [_][]const u8{ "A", "B", "C", "D" };
    var made: usize = 0;
    defer for (0..made) |i| {
        ss[i].deinit(gpa);
        ts[i].deinit(gpa);
        solo[i].deinit();
    };
    for (0..n) |i| {
        try solo[i].init(2);
        ss[i] = try stream(names[i], lengths[i], @intCast(i + 3), null);
        ts[i] = try stream(names[i], lengths[i], @intCast(i + 3), null);
        made += 1;
        try solo[i].core.addStream(&ss[i]);
    }
    // Reordered arrival and physical-slot mapping: D,C,B,A rather than reference order.
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        try r.core.addStream(&ts[i]);
    }
    try expect(r.cfg.family_streams and r.cfg.family_width == 1 and r.cfg.max_copy == 0 and !r.cfg.family_mtp);
    while (r.core.activeCount() > 0) {
        try r.core.step();
        for (0..n) |j| if (solo[j].core.activeCount() > 0) try solo[j].core.step();
    }
    for (0..n) |j| {
        try std.testing.expectEqualSlices(u32, ss[j].emitted(), ts[j].emitted());
        try eq(ss[j].reason, ts[j].reason);
        try eq(@as(usize, j + 3), ts[j].emitted().len);
    }
    var begins: usize = 0;
    var exchanges: usize = 0;
    var widths: [5]usize = @splat(0);
    for (r.fake.log.items) |e| {
        if (e.kind == .begin) {
            begins += 1;
            widths[e.rows] += 1;
        }
        if (e.kind == .exchange) {
            exchanges += 1;
            try expect(e.rows >= 1 and e.rows <= n);
        }
    }
    try eq(begins * 4, exchanges);
    try expect(widths[n] > 0);
    try eq(@as(usize, n), r.fake.prefills);
    try eq(@as(usize, n), r.fake.releases);
}

test "concurrent_users C2 together=independent serial, unequal positions/parities/DSA boundaries/lengths and reordered arrivals" {
    try together(2);
}
test "concurrent_users C4 same mapping, exactly one R4 exchange/layer and uneven retirement" {
    try together(4);
}

test "concurrent_users active cancellation during combined exchange preserves other user and ordinary slot reuse" {
    var r: Run = .{};
    try r.init(2);
    defer r.deinit();
    var flag = false;
    var a = try stream("A", 513, 5, &flag);
    defer a.deinit(gpa);
    var b = try stream("B", 511, 5, null);
    defer b.deinit(gpa);
    try r.core.addStream(&a);
    try r.core.addStream(&b);
    const old = try r.back.keyOf(&a);
    r.fake.cancel = &flag;
    r.fake.cancel_layer = 2;
    try r.core.step();
    try eq(lanes.stream.Reason.cancelled, a.reason);
    try eq(@as(usize, 1), a.emitted().len);
    try eq(@as(usize, 2), b.emitted().len);
    try eq(@as(usize, 513), r.fake.states[old.slot].position);
    var c = try stream("C", 2048, 4, null);
    defer c.deinit(gpa);
    try r.core.addStream(&c);
    const key = try r.back.keyOf(&c);
    try eq(old.slot, key.slot);
    try eq(old.generation + 1, key.generation);
    r.back.releaseKey(old);
    try eq(key, try r.back.keyOf(&c));
    while (r.core.activeCount() > 0) try r.core.step();
    try eq(@as(usize, 5), b.emitted().len);
    try eq(@as(usize, 4), c.emitted().len);
}

test "concurrent_users EOS retirement releases only its user generation" {
    var r: Run = .{};
    try r.init(2);
    defer r.deinit();
    var a = try stream("A", 512, 8, null);
    defer a.deinit(gpa);
    var b = try stream("B", 514, 4, null);
    defer b.deinit(gpa);
    var first: [1]u32 = .{0};
    var oracle: Fake = .{};
    defer oracle.deinit();
    first[0] = try oracle.prefill(.{ .slot = 0, .generation = 1 }, a.prompt());
    a.eos = &first;
    try r.core.addStream(&a);
    try r.core.addStream(&b);
    try eq(lanes.stream.Reason.stop, a.reason);
    try eq(@as(usize, 1), r.core.activeCount());
    while (r.core.activeCount() > 0) try r.core.step();
    try eq(@as(usize, 4), b.emitted().len);
}

fn failCase(layer: ?usize, settle: bool, after_sample: bool) !void {
    var r: Run = .{};
    try r.init(2);
    defer r.deinit();
    var a = try stream("A", 511, 4, null);
    defer a.deinit(gpa);
    var b = try stream("B", 2048, 4, null);
    defer b.deinit(gpa);
    try r.core.addStream(&a);
    try r.core.addStream(&b);
    r.fake.fail_layer = layer;
    r.fake.fail_settle = settle;
    r.fake.fail_after_sample = after_sample;
    try std.testing.expectError(error.ExpertFailed, r.core.step());
    try eq(@as(usize, 1), a.emitted().len);
    try eq(@as(usize, 1), b.emitted().len);
    for (r.fake.log.items) |e| {
        try expect(e.kind != .commit);
        if (!after_sample) try expect(e.kind != .sample);
    }
    const calls = r.fake.log.items.len;
    try std.testing.expectError(error.ExpertFailed, r.core.step());
    var c = try stream("C", 10, 4, null);
    defer c.deinit(gpa);
    try std.testing.expectError(error.ExpertFailed, r.core.addStream(&c));
    try eq(calls, r.fake.log.items.len);
    try eq(@as(usize, 2), r.fake.prefills);
}

test "concurrent_users failure at first MoE: zero later sends/samples/commits, sticky all users" {
    try failCase(2, false, false);
}
test "concurrent_users failure at last MoE: zero samples/commits, sticky all users" {
    try failCase(5, false, false);
}
test "concurrent_users failure observed at settle: zero samples/commits" {
    try failCase(null, true, false);
}
test "concurrent_users failure observed after sample: no next sample or commit" {
    try failCase(null, false, true);
}

test "concurrent_users waiting cancellation refuses prefill without allocation/reset and new user still works" {
    var r: Run = .{};
    try r.init(2);
    defer r.deinit();
    var flag = true;
    var a = try stream("A", 8, 4, &flag);
    defer a.deinit(gpa);
    try std.testing.expectError(error.Cancelled, r.core.addStream(&a));
    try eq(@as(usize, 0), r.fake.prefills);
    var b = try stream("B", 8, 4, null);
    defer b.deinit(gpa);
    try r.core.addStream(&b);
    while (r.core.activeCount() > 0) try r.core.step();
    try eq(@as(usize, 4), b.emitted().len);
}

test "concurrent_users unsupported sampled and MTP requests refuse before device calls; no-free-slot preserves owners" {
    var r: Run = .{};
    try r.init(2);
    defer r.deinit();
    var a = try stream("A", 8, 4, null);
    defer a.deinit(gpa);
    a.drafts = true;
    try std.testing.expectError(error.UsersDraftsUnsupported, r.core.addStream(&a));
    a.drafts = false;
    a.sampling = .{ .temperature = 1, .seed = 7 };
    try std.testing.expectError(error.GreedyOnly, r.core.addStream(&a));
    try eq(@as(usize, 0), r.fake.prefills);
    a.sampling = null;
    var b = try stream("B", 8, 4, null);
    defer b.deinit(gpa);
    var c = try stream("C", 8, 4, null);
    defer c.deinit(gpa);
    try r.core.addStream(&a);
    try r.core.addStream(&b);
    try std.testing.expectError(error.NoFreeSlot, r.core.addStream(&c));
    try eq(@as(usize, 2), r.fake.prefills);
    while (r.core.activeCount() > 0) try r.core.step();
}
