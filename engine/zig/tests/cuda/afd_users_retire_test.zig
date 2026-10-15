//! concurrent_users_r4: a failed round's and a cancelled client's releases through the real LaneHost, users backend and recorded
//! device (afd_users_device_test.zig's Rec), as TF_AFD_USERS_TRACE=1 labels them; no CUDA numerics.
//! Ash Hart authored TensorFold and its lane core; Rec/OwnLog are concurrent_users_r2's (GPT-6.1 Sol via Hermes); this file Claude Opus 5.5 via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const device = @import("afd_users_device_test.zig");
const Rec = device.Rec;
const OwnLog = device.OwnLog;
const Back = @import("glm_users").Users(glm.afd_users.Users);
const proto = cuda.xchg.proto;
const Event = @FieldType(glm.afd_users_round.Observation, "event");
const t = std.testing;
const gpa = t.allocator;

/// The served users path on Rec: the real LaneHost over the real users backend, its observations in `log`.
const Served = struct {
    r: Rec,
    log: OwnLog,
    wire: Wire2,
    back: Back,
    cfg: lanes.Config,
    clock: lanes.fake.FixedClock,
    core: lanes.Engine,
    host: api.LaneHost,
    stopped: bool,
    fn init(s: *Served, trace: bool) !void {
        try s.r.initRows(2, 64, 64);
        errdefer s.r.deinit();
        @memset(&s.r.users.generation, 0); // fresh slots, as the served warm-up's
        s.log = .{};
        s.log.attach(&s.r);
        s.r.users.observations = trace; // TF_AFD_USERS_TRACE=1, or unset: the observer then sees no call
        s.wire = .{ .r = &s.r };
        s.r.a.wire.?.host = .{ .ctx = &s.wire, .post = Wire2.post, .take = Wire2.take };
        s.back = .{ .steps = &s.r.users, .count = 2, .capacity = 2560, .reach = 2560 };
        s.cfg = try lanes.Config.init(gpa, s.back.facts(), 1, 0);
        s.clock = .{};
        s.core = lanes.Engine.init(gpa, &s.cfg, s.back.backend(), s.clock.clock());
        s.host = api.LaneHost.init(gpa, t.io, &s.core, .{ .lanes = 2, .context_window = 2560, .drafts = "off", .greedy_users = true });
        s.stopped = false;
    }
    fn deinit(s: *Served) void {
        s.wire.open.store(true, .release); // never leave the engine thread held at the gate
        if (!s.stopped) s.host.stop();
        s.core.deinit();
        s.cfg.deinit(gpa);
        s.r.deinit();
    }
    fn stop(s: *Served) void {
        s.host.stop();
        s.stopped = true;
    }
    fn ask(s: *Served, c: *Client, req: *const api.Request) !void {
        c.engine = s.host.engine();
        try c.engine.?.submit(c.id, req, c.sink());
    }
    fn count(s: *const Served, event: Event) usize {
        var n: usize = 0;
        for (s.log.values[0..s.log.n]) |v| n += @intFromBool(v.event == event);
        return n;
    }
    fn find(s: *const Served, from: usize, event: Event) !usize {
        for (s.log.values[from..s.log.n], from..) |v, i| if (v.event == event) return i;
        return error.ObservationMissing;
    }
};

/// Rec's exchange with two seams on R2 exchanges (1-based): expert 1 dies at the `fail_at`-th, a gate holds the `hold_at`-th.
const Wire2 = struct {
    r: *Rec,
    fail_at: usize = 0,
    hold_at: usize = 0,
    rows: usize = 0,
    r2: usize = 0,
    entered: std.atomic.Value(bool) = .init(false),
    open: std.atomic.Value(bool) = .init(false),
    fn post(p: *anyopaque, node: *cuda.xchg.Node, h: *const [proto.HDR_BYTES]u8, pos: usize, n: usize) !void {
        const w: *Wire2 = @ptrCast(@alignCast(p));
        w.rows = n;
        w.r2 += @intFromBool(n == 2);
        try Rec.post(w.r, node, h, pos, n);
    }
    fn take(p: *anyopaque, node: *cuda.xchg.Node, pos: usize, n: usize) !void {
        const w: *Wire2 = @ptrCast(@alignCast(p));
        try Rec.take(w.r, node, pos, n);
        if (w.rows != 2) return;
        if (w.r2 == w.hold_at) {
            w.entered.store(true, .release);
            try waitTrue(&w.open);
        }
        if (w.r2 == w.fail_at) node.fail(.{ .why = .dead, .peer = 1, .seq = @intCast(w.r.posts) });
    }
};

fn waitTrue(flag: *std.atomic.Value(bool)) !void {
    for (0..60_000) |_| {
        if (flag.load(.acquire)) return;
        const ts: std.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    return error.WaitTimeout;
}

/// One request's sink; `cancel_at` cancels it from the engine thread, between rounds, once it holds that many tokens.
const Client = struct {
    id: api.Id,
    cancel_at: usize = 0,
    engine: ?api.Engine = null,
    tokens: usize = 0,
    reason: ?api.Reason = null,
    message: []const u8 = "",
    done: std.atomic.Value(bool) = .init(false),
    fn event(p: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const c: *Client = @ptrCast(@alignCast(p));
        switch (ev.*) {
            .tokens => |ts| {
                c.tokens += ts.len;
                if (c.cancel_at > 0 and c.tokens >= c.cancel_at) {
                    c.cancel_at = 0; // as lane_host_test.zig's CancelPrefill cancels from a hook on the engine thread
                    c.engine.?.cancel(c.id);
                }
            },
            .finished => |end| {
                c.reason = end.reason;
                c.message = end.message;
                c.done.store(true, .release);
            },
            else => {},
        }
    }
    fn sink(c: *Client) api.Sink {
        return .{ .ctx = c, .event = event };
    }
};

fn promptIds(comptime n: usize, base: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*x, i| x.* = base + @as(u32, @intCast(i));
    return out;
}

/// qualification_57a13e98 group 4 on the recorder: SHORT (21) and LONG (60) share R2 rounds; expert 1 dies at the second R2 round's last
/// exchange. `text` gets what clients and the wire saw (finish reasons, messages, tokens, every posted header).
fn failedRound(s: *Served, text: *std.ArrayList(u8)) !void {
    s.wire.fail_at = 4; // two MoE layers a round
    var short = promptIds(21, 1000);
    var long = promptIds(60, 2000);
    const rs: api.Request = .{ .prompt = &short, .max_tokens = 8, .drafts = false };
    const rl: api.Request = .{ .prompt = &long, .max_tokens = 8, .drafts = false };
    var a: Client = .{ .id = 1 };
    var b: Client = .{ .id = 2 };
    try s.ask(&a, &rs);
    try s.ask(&b, &rl);
    try s.host.start();
    try waitTrue(&a.done);
    try waitTrue(&b.done);
    s.stop();
    for ([_]*const Client{ &a, &b }) |c| try text.print(gpa, "client {d} finished {s} \"{s}\" tokens {d}\n", .{ c.id, @tagName(c.reason.?), c.message, c.tokens });
    for (s.r.headers[0..s.r.posts]) |h| try text.print(gpa, "wire fseq {d} rows {d} layer {d} mode {s} layers {d} extra {any}\n", .{ h.fseq, h.rows, h.layer, @tagName(h.mode), h.layers, h.extra });
}

test "concurrent_users_r4 real LaneHost: expert 1 dies in an R2 round, both generations failure_retired, clients failed ExpertFailed" {
    var s: Served = undefined;
    try s.init(true);
    defer s.deinit();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try failedRound(&s, &text);
    std.debug.print("concurrent_users_r4-CLIENT-WIRE trace=1 wyhash={x}\n{s}", .{ std.hash.Wyhash.hash(0, text.items), text.items });
    const v = s.log.values[0..s.log.n];
    const first = try s.find(0, .failure_before_commit);
    for (v[first..]) |x| std.debug.print("concurrent_users_r4 OWN event={s} slot={d} generation={d} state={d} position={d} parity={d} round={d} commits={d}\n", .{ @tagName(x.event), x.slot, x.generation, x.state, x.position, x.parity, x.round, x.commits });
    try t.expectEqual(@as(usize, 4), s.wire.r2); // the failure ends R2 sends: the second round's last exchange was the last
    try t.expectEqual(first + 6, v.len);
    try t.expectEqual(@as(usize, 22), v[first].position); // SHORT's pre-commit State after one committed R2 round
    try t.expectEqual(@as(usize, 61), v[first + 1].position); // LONG's
    const want = [_]Event{ .failure_before_commit, .failure_before_commit, .failure_retired, .retire, .failure_retired, .retire };
    const slots = [_]u8{ 0, 1, 0, 0, 1, 1 };
    for (v[first..], want, slots) |x, event, slot| {
        try t.expectEqual(slot, x.slot);
        try t.expectEqual(x.slot, x.state);
        try t.expectEqual(@as(u64, 1), x.generation);
        try t.expectEqual(@as(usize, 0), x.commits);
        try t.expectEqual(v[first + slot].position, x.position); // retired at the pre-commit State: nothing committed
        try t.expectEqual(v[first + slot].parity, x.parity);
        try t.expectEqual(event, x.event);
    }
    try t.expectEqual(@as(usize, 2), s.count(.failure_retired));
    try t.expectEqual(@as(usize, 0), s.count(.cancel_retired));
    try t.expectEqualStrings(
        \\client 1 finished failed "ExpertFailed" tokens 2
        \\client 2 finished failed "ExpertFailed" tokens 2
        \\
    , text.items[0..std.mem.indexOf(u8, text.items, "wire ").?]);
}

test "concurrent_users_r4 TF_AFD_USERS_TRACE unset: a failed round calls no observer, clients and wire equal the traced run's" {
    var on: Served = undefined;
    try on.init(true);
    defer on.deinit();
    var off: Served = undefined;
    try off.init(false);
    defer off.deinit();
    var a: std.ArrayList(u8) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(gpa);
    try failedRound(&on, &a);
    try failedRound(&off, &b);
    std.debug.print("concurrent_users_r4-CLIENT-WIRE trace=unset wyhash={x}\n", .{std.hash.Wyhash.hash(0, b.items)});
    try t.expectEqual(@as(usize, 0), off.log.n);
    try t.expectEqual(@as(usize, 2), on.count(.failure_before_commit));
    try t.expectEqualStrings(a.items, b.items);
}

test "concurrent_users_r4 real LaneHost: a client cancelled between R2 rounds is cancel_retired; the slot's next generation reuses it" {
    var s: Served = undefined;
    try s.init(true);
    defer s.deinit();
    var short = promptIds(21, 1000);
    var long = promptIds(60, 2000);
    const rs: api.Request = .{ .prompt = &short, .max_tokens = 4, .drafts = false };
    const rl: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var a: Client = .{ .id = 1, .cancel_at = 2 }; // after its first R2 round: LaneHost.takeCancels -> Engine.discard
    var b: Client = .{ .id = 2 };
    var c: Client = .{ .id = 3 };
    try s.ask(&a, &rs);
    try s.ask(&b, &rl);
    try s.host.start();
    try waitTrue(&a.done);
    try waitTrue(&b.done);
    try s.ask(&c, &rs);
    try waitTrue(&c.done);
    s.stop();
    try t.expectEqual(api.Reason.cancelled, a.reason.?);
    try t.expectEqual(@as(usize, 2), a.tokens);
    try t.expectEqual(api.Reason.length, b.reason.?);
    try t.expectEqual(@as(usize, 4), b.tokens);
    try t.expectEqual(api.Reason.length, c.reason.?);
    try t.expectEqual(@as(usize, 4), c.tokens);
    try t.expectEqual(@as(usize, 1), s.count(.cancel_retired));
    try t.expectEqual(@as(usize, 0), s.count(.failure_retired));
    const v = s.log.values[0..s.log.n];
    const i = try s.find(0, .cancel_retired);
    try t.expectEqual(@as(u8, 0), v[i].slot);
    try t.expectEqual(@as(u64, 1), v[i].generation);
    try t.expectEqual(@as(usize, 22), v[i].position); // its committed State, unchanged by the cancel
    try t.expectEqual(.retire, v[i + 1].event);
    try t.expectEqual(v[i].slot, v[i + 1].slot);
    try t.expectEqual(v[i].generation, v[i + 1].generation);
    for (v[i + 2 ..]) |x| try t.expect(!(x.slot == 0 and x.generation == 1)); // nothing of the retired generation after
    const j = try s.find(i, .prefill_before);
    try t.expectEqual(@as(u8, 0), v[j].slot); // C takes the cancelled slot as its next generation
    try t.expectEqual(@as(u64, 2), v[j].generation);
    try t.expectEqual(.prefill_after, v[j + 1].event);
    try t.expectEqual(@as(u64, 2), v[j + 1].generation);
    try t.expectEqual(@as(usize, 21), v[j + 1].position);
}

test "concurrent_users_r4 real LaneHost: a client cancelled inside an R2 round is cancel_retired; its peer commits beside it" {
    var s: Served = undefined;
    try s.init(true);
    defer s.deinit();
    s.wire.hold_at = 3; // the second R2 round's first exchange
    var short = promptIds(21, 1000);
    var long = promptIds(60, 2000);
    const rs: api.Request = .{ .prompt = &short, .max_tokens = 4, .drafts = false };
    const rl: api.Request = .{ .prompt = &long, .max_tokens = 4, .drafts = false };
    var a: Client = .{ .id = 1 };
    var b: Client = .{ .id = 2 };
    try s.ask(&a, &rs);
    try s.ask(&b, &rl);
    try s.host.start();
    try waitTrue(&s.wire.entered);
    a.engine.?.cancel(1); // the client leaves while the round's exchange is in flight
    s.wire.open.store(true, .release);
    try waitTrue(&a.done);
    try waitTrue(&b.done);
    s.stop();
    try t.expectEqual(api.Reason.cancelled, a.reason.?);
    try t.expectEqual(@as(usize, 2), a.tokens);
    try t.expectEqual(api.Reason.length, b.reason.?);
    try t.expectEqual(@as(usize, 4), b.tokens);
    try t.expectEqual(@as(usize, 1), s.count(.cancel_retired));
    try t.expectEqual(@as(usize, 0), s.count(.failure_retired));
    const v = s.log.values[0..s.log.n];
    const k = try s.find(0, .cancel_before_sample);
    const want = [_]Event{ .cancel_before_sample, .cancel_before_commit, .commit_before, .commit_after, .round_settled, .round_settled, .cancel_retired, .retire };
    const slots = [_]u8{ 0, 0, 1, 1, 0, 1, 0, 0 };
    for (v[k..][0..want.len], want, slots) |x, event, slot| {
        try t.expectEqual(event, x.event);
        try t.expectEqual(slot, x.slot);
        if (slot == 0) try t.expectEqual(@as(usize, 22), x.position); // the cancelled State never moves
    }
    try t.expectEqual(@as(usize, 62), v[k + 3].position); // the peer committed beside it
}
