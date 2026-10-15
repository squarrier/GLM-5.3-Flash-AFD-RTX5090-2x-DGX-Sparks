//! structured_users: structured output with TF_AFD_USERS=2 through the real LaneHost, lane core, users backend and users device on
//! the recorder (afd_users_device_test.zig's Rec), with the masked adapter in its recorded mode (cuda_mask_dev
//! Dev.recorded): two lanes with one, both and neither structured; a matcher per request; retirement (stop),
//! cancel_retired, failure_retired and a grammar dead end. The device work is MOCK (recorded launches; a masked draw's
//! candidate is the lowest eligible id of the STAGED words; a plain draw answers a fixed token by lane); the matcher is a
//! FAKE script. No CUDA numerics.
//! Ash Hart authored TensorFold and its lane core; Rec/OwnLog are concurrent_users_r2's (GPT-6.1 Sol via Hermes) and the LaneHost
//! seams concurrent_users_r4's; this file Claude Opus 5.5 via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const device = @import("afd_users_device_test.zig");
const Rec = device.Rec;
const OwnLog = device.OwnLog;
const Back = @import("glm_users").Users(glm.afd_users.Users);
const md = glm.sample.mask_dev;
const proto = cuda.xchg.proto;
const Event = @FieldType(glm.afd_users_round.Observation, "event");
const t = std.testing;
const gpa = t.allocator;

/// FAKE matcher: step i (accepted tokens so far) allows `steps[i]` (an empty step: a dead end); terminated once every
/// step is accepted. Counts its fills and closes (a matcher closed exactly once by its stream's end).
const Script = struct {
    rules: lanes.grammar.Rules = undefined,
    steps: []const []const u32,
    /// the step whose (mask-allowed) token the matcher still refuses at commit (a matcher/mask disagreement)
    refuse_at: usize = std.math.maxInt(usize),
    accepted: usize = 0,
    fills: usize = 0,
    closes: usize = 0,
    fn init(x: *Script, width: usize) void {
        x.rules = .{ .ptr = x, .width = width, .vtable = &.{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close } };
    }
    fn of(p: *anyopaque) *Script {
        return @ptrCast(@alignCast(p));
    }
    fn fill(p: *anyopaque, words: []u32) !void {
        const x = of(p);
        x.fills += 1;
        @memset(words, 0);
        if (x.accepted < x.steps.len) for (x.steps[x.accepted]) |tok| {
            words[tok / 32] |= @as(u32, 1) << @as(u5, @intCast(tok % 32));
        };
    }
    fn accept(p: *anyopaque, token: u32) !bool {
        const x = of(p);
        if (x.accepted == x.refuse_at) return false;
        if (x.accepted >= x.steps.len or std.mem.indexOfScalar(u32, x.steps[x.accepted], token) == null) return false;
        x.accepted += 1;
        return true;
    }
    fn rollback(p: *anyopaque, n: usize) !void {
        const x = of(p);
        if (n > x.accepted) return error.BadRollback;
        x.accepted -= n;
    }
    fn terminated(p: *anyopaque) bool {
        const x = of(p);
        return x.accepted >= x.steps.len;
    }
    fn close(p: *anyopaque) void {
        of(p).closes += 1;
    }
};

/// MOCK device answers. A plain draw (the recorder's picker): `plain[lane]` on a lane's own logits row, `first` on the
/// Engine's prompt-pass row. A masked draw (the adapter's emulator): the lowest eligible id of the STAGED words.
const Mock = struct {
    r: *Rec,
    plain: [2]u32 = .{ 70, 90 },
    first: u32 = 50,
    plain_draws: [3]usize = @splat(0),
    masked: usize = 0,
    fn picker(m: *Mock) glm.trace.Picker {
        return .{ .ctx = m, .pick = pick };
    }
    fn pick(p: *anyopaque, logits: glm.trace.T, _: ?[]const u64, out: []u32, probs: ?[]f64) !void {
        const m: *Mock = @ptrCast(@alignCast(p));
        var lane: usize = 2;
        for (m.r.users.saved[0..2], 0..) |s, i| {
            if (s.logits.ptr == logits.ptr) lane = i;
        }
        m.plain_draws[lane] += 1;
        @memset(out, if (lane < 2) m.plain[lane] else m.first);
        if (probs) |x| @memset(x, 0);
    }
    fn emulator(m: *Mock) md.Emulator {
        return .{ .ctx = m, .candidate = candidate };
    }
    fn candidate(p: *anyopaque, x: *const md.Dev, _: glm.trace.T) !md.Candidate {
        const m: *Mock = @ptrCast(@alignCast(p));
        m.masked += 1;
        for (x.host[0..x.words], 0..) |w, i| if (w != 0) return .{ .id = @intCast(i * 32 + @ctz(w)), .bits = 0x3f80_0000 };
        return error.EmptyStaged;
    }
};

/// Rec's exchange with concurrent_users_r4's failure seam on R2 exchanges (1-based): expert 1 dies at the `fail_at`-th.
const Wire2 = struct {
    r: *Rec,
    fail_at: usize = 0,
    rows: usize = 0,
    r2: usize = 0,
    fn post(p: *anyopaque, node: *cuda.xchg.Node, h: *const [proto.HDR_BYTES]u8, pos: usize, n: usize) !void {
        const w: *Wire2 = @ptrCast(@alignCast(p));
        w.rows = n;
        w.r2 += @intFromBool(n == 2);
        try Rec.post(w.r, node, h, pos, n);
    }
    fn take(p: *anyopaque, node: *cuda.xchg.Node, pos: usize, n: usize) !void {
        const w: *Wire2 = @ptrCast(@alignCast(p));
        try Rec.take(w.r, node, pos, n);
        if (w.rows == 2 and w.r2 == w.fail_at) node.fail(.{ .why = .dead, .peer = 1, .seq = @intCast(w.r.posts) });
    }
};

/// The served users path on Rec: the real LaneHost over the real users backend; with `masked`, the recorded adapter on
/// the sampler and the host's structured hook (each structured request takes the next script: its own matcher).
const Served = struct {
    r: Rec,
    log: OwnLog,
    dev: md.Dev,
    mock: Mock,
    wire: Wire2,
    back: Back,
    cfg: lanes.Config,
    clock: lanes.fake.FixedClock,
    core: lanes.Engine,
    host: api.LaneHost,
    scripts: []Script,
    given: usize,
    stopped: bool,
    fn init(s: *Served, masked: bool, scripts: []Script) !void {
        try s.r.initRows(2, 64, 64);
        errdefer s.r.deinit();
        @memset(&s.r.users.generation, 0); // fresh slots, as the served warm-up's
        s.log = .{};
        s.log.attach(&s.r);
        s.r.users.observations = true;
        s.mock = .{ .r = &s.r };
        s.r.trace.picker = s.mock.picker();
        s.dev = try md.Dev.recorded(gpa, &s.r.m, &s.r.space); // placed in both modes: the same Space either way
        s.dev.emulate = s.mock.emulator();
        if (masked) s.r.e.smp.masked_dev = &s.dev;
        s.wire = .{ .r = &s.r };
        s.r.a.wire.?.host = .{ .ctx = &s.wire, .post = Wire2.post, .take = Wire2.take };
        s.back = .{ .steps = &s.r.users, .count = 2, .capacity = 2560, .reach = 2560 };
        s.cfg = try lanes.Config.init(gpa, s.back.facts(), 1, 0);
        s.clock = .{};
        s.core = lanes.Engine.init(gpa, &s.cfg, s.back.backend(), s.clock.clock());
        s.host = api.LaneHost.init(gpa, t.io, &s.core, .{ .lanes = 2, .context_window = 2560, .drafts = "off", .greedy_users = true, .structures = masked });
        if (masked) s.host.structured = .{ .ctx = s, .rules = rules };
        s.scripts = scripts;
        s.given = 0;
        s.stopped = false;
    }
    fn deinit(s: *Served) void {
        if (!s.stopped) s.host.stop();
        s.core.deinit();
        s.cfg.deinit(gpa);
        s.dev.close();
        s.r.deinit();
    }
    fn stop(s: *Served) void {
        s.host.stop();
        s.stopped = true;
    }
    fn rules(ctx: *anyopaque, _: *const api.Request) anyerror!*lanes.grammar.Rules {
        const s: *Served = @ptrCast(@alignCast(ctx));
        if (s.given >= s.scripts.len) return error.NoScript;
        const x = &s.scripts[s.given];
        s.given += 1;
        x.init(s.r.m.vocab);
        return &x.rules;
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
    /// The recorded events that read or write lane `slot`'s own logits row, in order (its row's draws and its head).
    fn laneEvents(s: *Served, slot: usize, out: *std.ArrayList(glm.trace.Event)) !void {
        try s.r.trace.flush();
        const at = s.r.users.saved[slot].logits.ptr;
        for (s.r.trace.events.items) |e| {
            for (e.vals) |v| if (v == .t and v.t.ptr == at) {
                try out.append(gpa, e);
                break;
            };
        }
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

/// One request's sink: its tokens, its end; `cancel_at` cancels it from the engine thread once it holds that many.
const Client = struct {
    id: api.Id,
    cancel_at: usize = 0,
    engine: ?api.Engine = null,
    buf: [32]u32 = @splat(0),
    tokens: usize = 0,
    reason: ?api.Reason = null,
    message: []const u8 = "",
    done: std.atomic.Value(bool) = .init(false),
    fn event(p: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const c: *Client = @ptrCast(@alignCast(p));
        switch (ev.*) {
            .tokens => |ts| {
                for (ts) |x| {
                    if (c.tokens < c.buf.len) c.buf[c.tokens] = x;
                    c.tokens += 1;
                }
                if (c.cancel_at > 0 and c.tokens >= c.cancel_at) {
                    c.cancel_at = 0;
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
    fn got(c: *const Client) []const u32 {
        return c.buf[0..@min(c.tokens, c.buf.len)];
    }
};

fn promptIds(comptime n: usize, base: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*x, i| x.* = base + @as(u32, @intCast(i));
    return out;
}

fn eqEvents(x: []const glm.trace.Event, y: []const glm.trace.Event) !void {
    try t.expectEqual(x.len, y.len);
    for (x, y) |p, q| try t.expectEqualDeep(p, q);
}

/// A tensor the same but for a temporary's address: python's temporaries are placed by op and occurrence
/// (cuda_tmp.zig's `tmp.<op>.<n>`), so the masked draws of a structured peer shift the numbering of a plain row's later
/// temporaries; `row` (the lane's own logits row) must match exactly.
fn eqTensor(p: glm.trace.T, q: glm.trace.T, row: u64) !void {
    if (p.ptr == row or q.ptr == row) try t.expectEqual(p.ptr, q.ptr);
    try t.expectEqual(p.dtype, q.dtype);
    try t.expectEqual(p.dims, q.dims);
    try t.expectEqualSlices(usize, &p.shape, &q.shape);
    try t.expectEqualSlices(usize, &p.stride, &q.stride);
}

fn eqVals(x: []const glm.trace.Val, y: []const glm.trace.Val, row: u64) !void {
    try t.expectEqual(x.len, y.len);
    for (x, y) |p, q| {
        try t.expectEqual(std.meta.activeTag(p), std.meta.activeTag(q));
        switch (p) {
            .t => try eqTensor(p.t, q.t, row),
            .list => try eqVals(p.list, q.list, row),
            else => try t.expectEqualDeep(p, q),
        }
    }
}

/// The same launches on a row: names, grids, arguments, constants and values, temporaries' addresses aside (eqTensor).
fn eqRowEvents(x: []const glm.trace.Event, y: []const glm.trace.Event, row: u64) !void {
    try t.expectEqual(x.len, y.len);
    for (x, y) |p, q| {
        try t.expectEqual(p.op, q.op);
        try t.expectEqualStrings(p.name, q.name);
        try t.expectEqual(p.grid, q.grid);
        try t.expectEqual(p.args.len, q.args.len);
        try t.expectEqual(p.consts.len, q.consts.len);
        try eqVals(p.vals, q.vals, row);
    }
}

const json: api.Structure = .{ .kind = .json };

test "structured_users users: two plain lanes beside the installed adapter (neither structured) record every event and token of the adapter-less run" {
    var n: Served = undefined;
    try n.init(false, &.{});
    defer n.deinit();
    var m: Served = undefined;
    try m.init(true, &.{});
    defer m.deinit();
    var p_ids = promptIds(21, 1000);
    var q_ids = promptIds(30, 3000);
    const rp: api.Request = .{ .prompt = &p_ids, .max_tokens = 5, .drafts = false };
    const rq: api.Request = .{ .prompt = &q_ids, .max_tokens = 5, .drafts = false };
    var cs = [_]Client{ .{ .id = 1 }, .{ .id = 2 }, .{ .id = 1 }, .{ .id = 2 } };
    for ([_]*Served{ &n, &m }, 0..) |s, k| {
        try s.ask(&cs[2 * k], &rp);
        try s.ask(&cs[2 * k + 1], &rq);
        try s.host.start();
        try waitTrue(&cs[2 * k].done);
        try waitTrue(&cs[2 * k + 1].done);
        s.stop();
    }
    for (0..2) |i| {
        try t.expectEqual(api.Reason.length, cs[i].reason.?);
        try t.expectEqualSlices(u32, cs[i].got(), cs[i + 2].got());
    }
    try t.expectEqualSlices(u32, &.{ 50, 70, 70, 70, 70 }, cs[0].got());
    try t.expectEqualSlices(u32, &.{ 50, 90, 90, 90, 90 }, cs[1].got());
    try n.r.trace.flush();
    try m.r.trace.flush();
    try eqEvents(n.r.trace.events.items, m.r.trace.events.items);
    try t.expectEqual(@as(usize, 0), m.mock.masked);
    try t.expectEqual(@as(u64, 0), m.dev.stats.staged);
    try t.expect(m.wire.r2 > 0 and m.wire.r2 == n.wire.r2); // the same shared R2 rounds
    std.debug.print("structured_users users neither: {d} events equal with and without the adapter, R2 exchanges {d}, tokens {any} / {any}\n", .{ m.r.trace.events.items.len, m.wire.r2, cs[2].got(), cs[3].got() });
}

test "structured_users users: one structured and one plain lane share R2 rounds; the structured row draws under its own mask, the plain row's events and tokens are a plain pair's" {
    var n: Served = undefined;
    try n.init(false, &.{});
    defer n.deinit();
    var p_ids = promptIds(21, 1000);
    var q_ids = promptIds(30, 3000);
    const rp: api.Request = .{ .prompt = &p_ids, .max_tokens = 6, .drafts = false };
    const rq: api.Request = .{ .prompt = &q_ids, .max_tokens = 6, .drafts = false };
    var a: Client = .{ .id = 1 };
    var b: Client = .{ .id = 2 };
    try n.ask(&a, &rp);
    try n.ask(&b, &rq);
    try n.host.start();
    try waitTrue(&a.done);
    try waitTrue(&b.done);
    n.stop();
    const steps = [_][]const u32{ &.{ 600, 11 }, &.{12}, &.{ 13, 900 }, &.{ 14, 15 } };
    var scripts = [_]Script{.{ .steps = &steps }};
    var m: Served = undefined;
    try m.init(true, &scripts);
    defer m.deinit();
    const rs: api.Request = .{ .prompt = &q_ids, .max_tokens = 6, .drafts = false, .structure = json };
    var c: Client = .{ .id = 1 };
    var d: Client = .{ .id = 2 };
    try m.ask(&c, &rp);
    try m.ask(&d, &rs);
    try m.host.start();
    try waitTrue(&c.done);
    try waitTrue(&d.done);
    m.stop();
    // the structured lane: its first token and each round's row among its own script's tokens, then its grammar's end
    try t.expectEqualSlices(u32, &.{ 11, 12, 13, 14 }, d.got());
    try t.expectEqual(api.Reason.stop, d.reason.?);
    try t.expectEqual(@as(usize, 4), m.mock.masked);
    try t.expectEqual(@as(usize, 4), scripts[0].fills);
    try t.expectEqual(@as(usize, 1), scripts[0].closes);
    try t.expectEqual(@as(usize, 0), m.mock.plain_draws[1]); // never a plain draw on the structured lane's row
    // the plain lane: the same tokens and the same events on its row as in the plain pair
    try t.expectEqual(api.Reason.length, c.reason.?);
    try t.expectEqualSlices(u32, a.got(), c.got());
    var pe: std.ArrayList(glm.trace.Event) = .empty;
    defer pe.deinit(gpa);
    var qe: std.ArrayList(glm.trace.Event) = .empty;
    defer qe.deinit(gpa);
    try n.laneEvents(0, &pe);
    try m.laneEvents(0, &qe);
    try t.expect(pe.items.len > 0);
    try t.expectEqual(n.r.users.saved[0].logits.ptr, m.r.users.saved[0].logits.ptr);
    try eqRowEvents(pe.items, qe.items, m.r.users.saved[0].logits.ptr);
    // and the structured lane's masked draws never touched the plain lane's row
    var se: std.ArrayList(glm.trace.Event) = .empty;
    defer se.deinit(gpa);
    try m.laneEvents(1, &se);
    for (se.items) |e| for (e.vals) |v| if (v == .t) try t.expect(v.t.ptr != m.r.users.saved[0].logits.ptr);
    try t.expect(m.wire.r2 >= 3); // shared R2 forwards while both lanes were live
    try t.expectEqual(@as(u64, 4), m.dev.stats.staged);
    std.debug.print("structured_users users one-structured: structured {any} (stop), plain {any} = plain pair's; plain-row events {d} equal; masked draws {d}; R2 exchanges {d}\n", .{ d.got(), c.got(), qe.items.len, m.mock.masked, m.wire.r2 });
}

test "structured_users users: two structured lanes in shared R2 rounds, each row under its own matcher's mask" {
    const one = [_][]const u32{ &.{21}, &.{ 22, 40 }, &.{23} };
    const two = [_][]const u32{ &.{ 31, 32 }, &.{33}, &.{34}, &.{ 35, 36 } };
    var scripts = [_]Script{ .{ .steps = &one }, .{ .steps = &two } };
    var m: Served = undefined;
    try m.init(true, &scripts);
    defer m.deinit();
    var p_ids = promptIds(21, 1000);
    var q_ids = promptIds(30, 3000);
    const r1: api.Request = .{ .prompt = &p_ids, .max_tokens = 8, .drafts = false, .structure = json };
    const r2: api.Request = .{ .prompt = &q_ids, .max_tokens = 8, .drafts = false, .structure = json };
    var a: Client = .{ .id = 1 };
    var b: Client = .{ .id = 2 };
    try m.ask(&a, &r1);
    try m.ask(&b, &r2);
    try m.host.start();
    try waitTrue(&a.done);
    try waitTrue(&b.done);
    m.stop();
    try t.expectEqualSlices(u32, &.{ 21, 22, 23 }, a.got());
    try t.expectEqualSlices(u32, &.{ 31, 33, 34, 35 }, b.got());
    for ([_]*const Client{ &a, &b }) |c| try t.expectEqual(api.Reason.stop, c.reason.?);
    try t.expectEqual(@as(usize, 3), scripts[0].fills);
    try t.expectEqual(@as(usize, 4), scripts[1].fills);
    for (scripts) |x| try t.expectEqual(@as(usize, 1), x.closes);
    try t.expectEqual(@as(usize, 7), m.mock.masked);
    try t.expectEqual(@as(usize, 0), m.mock.plain_draws[0] + m.mock.plain_draws[1] + m.mock.plain_draws[2]);
    try t.expect(m.wire.r2 >= 4);
    try t.expectEqual(@as(usize, 2), m.count(.retire));
    std.debug.print("structured_users users both-structured: {any} and {any} (stop, stop), masked draws {d}, R2 exchanges {d}\n", .{ a.got(), b.got(), m.mock.masked, m.wire.r2 });
}

test "structured_users users: a structured lane's stop retires its matcher; its slot's next generation (plain) draws plain, a later structured request starts its own matcher afresh" {
    const first = [_][]const u32{ &.{51}, &.{52} };
    const later = [_][]const u32{ &.{61}, &.{62}, &.{63} };
    var scripts = [_]Script{ .{ .steps = &first }, .{ .steps = &later } };
    var m: Served = undefined;
    try m.init(true, &scripts);
    defer m.deinit();
    var s_ids = promptIds(21, 1000);
    var p_ids = promptIds(30, 3000);
    var q_ids = promptIds(25, 5000);
    const rs: api.Request = .{ .prompt = &s_ids, .max_tokens = 8, .drafts = false, .structure = json };
    const rp: api.Request = .{ .prompt = &p_ids, .max_tokens = 10, .drafts = false };
    const rq: api.Request = .{ .prompt = &q_ids, .max_tokens = 3, .drafts = false };
    var s: Client = .{ .id = 1 };
    var p: Client = .{ .id = 2 };
    var q: Client = .{ .id = 3 };
    var l: Client = .{ .id = 4 };
    try m.ask(&s, &rs);
    try m.ask(&p, &rp);
    try m.host.start();
    try waitTrue(&s.done);
    const masked_at_stop = m.mock.masked;
    try m.ask(&q, &rq); // takes the retired slot 0 (generation 2), plain
    try waitTrue(&q.done);
    try m.ask(&l, &rs); // a later structured request: slot 0 again (generation 3), the next script
    try waitTrue(&l.done);
    try waitTrue(&p.done);
    m.stop();
    try t.expectEqualSlices(u32, &.{ 51, 52 }, s.got());
    try t.expectEqual(api.Reason.stop, s.reason.?);
    try t.expectEqual(@as(usize, 1), scripts[0].closes);
    try t.expectEqual(@as(usize, 2), scripts[0].fills); // nothing of the retired matcher after its stop
    try t.expectEqualSlices(u32, &.{ 50, 70, 70 }, q.got()); // slot 0's plain generation: plain draws on its row
    try t.expectEqual(api.Reason.length, q.reason.?);
    try t.expectEqualSlices(u32, &.{ 61, 62, 63 }, l.got()); // a fresh matcher from its first step
    try t.expectEqual(@as(usize, 1), scripts[1].closes);
    try t.expectEqual(masked_at_stop + 3, m.mock.masked);
    try t.expectEqual(api.Reason.length, p.reason.?);
    try t.expectEqual(@as(usize, 4), m.count(.retire)); // S, Q and L on slot 0 (generations 1-3), P on slot 1
    try t.expectEqual(@as(usize, 0), m.count(.cancel_retired) + m.count(.failure_retired));
    for (m.back.slots[0..2]) |slot| try t.expect(slot.owner == null and !slot.deferred);
    std.debug.print("structured_users users retire: structured {any} stop, slot reuse plain {any}, later structured {any} (fresh matcher), retire labels {d}\n", .{ s.got(), q.got(), l.got(), m.count(.retire) });
}

test "structured_users users: a structured lane cancelled between R2 rounds is cancel_retired and its matcher closed once; its peer goes on; the slot's next structured generation starts afresh" {
    const first = [_][]const u32{ &.{71}, &.{72}, &.{73}, &.{74}, &.{75} };
    const later = [_][]const u32{ &.{81}, &.{82} };
    var scripts = [_]Script{ .{ .steps = &first }, .{ .steps = &later } };
    var m: Served = undefined;
    try m.init(true, &scripts);
    defer m.deinit();
    var s_ids = promptIds(21, 1000);
    var p_ids = promptIds(30, 3000);
    const rs: api.Request = .{ .prompt = &s_ids, .max_tokens = 8, .drafts = false, .structure = json };
    const rp: api.Request = .{ .prompt = &p_ids, .max_tokens = 5, .drafts = false };
    var s: Client = .{ .id = 1, .cancel_at = 2 };
    var p: Client = .{ .id = 2 };
    var c: Client = .{ .id = 3 };
    try m.ask(&s, &rs);
    try m.ask(&p, &rp);
    try m.host.start();
    try waitTrue(&s.done);
    try waitTrue(&p.done);
    try m.ask(&c, &rs);
    try waitTrue(&c.done);
    m.stop();
    try t.expectEqual(api.Reason.cancelled, s.reason.?);
    try t.expectEqualSlices(u32, &.{ 71, 72 }, s.got());
    try t.expectEqual(@as(usize, 1), scripts[0].closes);
    try t.expectEqual(api.Reason.length, p.reason.?);
    try t.expectEqualSlices(u32, &.{ 50, 90, 90, 90, 90 }, p.got());
    try t.expectEqualSlices(u32, &.{ 81, 82 }, c.got());
    try t.expectEqual(api.Reason.stop, c.reason.?);
    try t.expectEqual(@as(usize, 1), scripts[1].closes);
    try t.expectEqual(@as(usize, 1), m.count(.cancel_retired));
    try t.expectEqual(@as(usize, 0), m.count(.failure_retired));
    const v = m.log.values[0..m.log.n];
    var i: usize = 0;
    while (v[i].event != .cancel_retired) i += 1;
    try t.expectEqual(@as(u8, 0), v[i].slot);
    try t.expectEqual(@as(u64, 1), v[i].generation);
    try t.expectEqual(Event.retire, v[i + 1].event);
    var j = i;
    while (!(v[j].event == .prefill_before and v[j].slot == 0)) j += 1;
    try t.expectEqual(@as(u64, 2), v[j].generation); // the next structured request reuses the cancelled slot
    for (m.back.slots[0..2]) |slot| try t.expect(slot.owner == null and !slot.deferred);
    std.debug.print("structured_users users cancel: structured {any} cancel_retired (slot 0 gen 1), peer {any}, next structured {any} on slot 0 gen 2 (fresh matcher)\n", .{ s.got(), p.got(), c.got() });
}

test "structured_users users: a failed R2 round fails both lanes failure_retired; the structured lane's matcher closes once and no slot keeps grammar state" {
    const steps = [_][]const u32{ &.{91}, &.{92}, &.{93}, &.{94}, &.{95}, &.{96} };
    var scripts = [_]Script{.{ .steps = &steps }};
    var m: Served = undefined;
    try m.init(true, &scripts);
    defer m.deinit();
    m.wire.fail_at = 4; // two MoE layers a round: the second R2 round's last exchange
    var s_ids = promptIds(21, 1000);
    var p_ids = promptIds(60, 2000);
    const rs: api.Request = .{ .prompt = &s_ids, .max_tokens = 8, .drafts = false, .structure = json };
    const rp: api.Request = .{ .prompt = &p_ids, .max_tokens = 8, .drafts = false };
    var s: Client = .{ .id = 1 };
    var p: Client = .{ .id = 2 };
    try m.ask(&s, &rs);
    try m.ask(&p, &rp);
    try m.host.start();
    try waitTrue(&s.done);
    try waitTrue(&p.done);
    m.stop();
    for ([_]*const Client{ &s, &p }) |c| try t.expectEqual(api.Reason.failed, c.reason.?);
    try t.expectEqualSlices(u32, &.{ 91, 92 }, s.got()); // the first token and the first R2 round's
    try t.expectEqual(@as(usize, 1), scripts[0].closes);
    try t.expectEqual(@as(usize, 2), m.count(.failure_retired));
    try t.expectEqual(@as(usize, 2), m.count(.retire));
    try t.expectEqual(@as(usize, 0), m.count(.cancel_retired));
    for (m.back.slots[0..2]) |slot| try t.expect(slot.owner == null and !slot.deferred);
    try t.expect(m.r.at.deferred_logits == null);
    std.debug.print("structured_users users failure: both failed \"{s}\", structured {any}; failure_retired {d}, matcher closes {d}\n", .{ s.message, s.got(), m.count(.failure_retired), scripts[0].closes });
}

test "structured_users users: a structured lane whose grammar dead-ends leaves its R2 round alone (failed, failure_retired); its plain peer finishes as in a plain pair" {
    var n: Served = undefined;
    try n.init(false, &.{});
    defer n.deinit();
    var s_ids = promptIds(21, 1000);
    var p_ids = promptIds(30, 3000);
    const rp: api.Request = .{ .prompt = &p_ids, .max_tokens = 5, .drafts = false };
    const rq: api.Request = .{ .prompt = &s_ids, .max_tokens = 5, .drafts = false };
    var a: Client = .{ .id = 1 };
    var b: Client = .{ .id = 2 };
    try n.ask(&a, &rq);
    try n.ask(&b, &rp);
    try n.host.start();
    try waitTrue(&a.done);
    try waitTrue(&b.done);
    n.stop();
    const steps = [_][]const u32{ &.{41}, &.{} };
    var scripts = [_]Script{.{ .steps = &steps }};
    var m: Served = undefined;
    try m.init(true, &scripts);
    defer m.deinit();
    const rs: api.Request = .{ .prompt = &s_ids, .max_tokens = 5, .drafts = false, .structure = json };
    var s: Client = .{ .id = 1 };
    var p: Client = .{ .id = 2 };
    try m.ask(&s, &rs);
    try m.ask(&p, &rp);
    try m.host.start();
    try waitTrue(&s.done);
    try waitTrue(&p.done);
    m.stop();
    try t.expectEqual(api.Reason.failed, s.reason.?);
    try t.expectEqualSlices(u32, &.{41}, s.got());
    try t.expectEqual(@as(usize, 1), scripts[0].closes);
    try t.expectEqual(api.Reason.length, p.reason.?);
    try t.expectEqualSlices(u32, b.got(), p.got());
    try t.expectEqual(@as(usize, 1), m.count(.failure_retired));
    try t.expectEqual(@as(usize, 1), m.mock.masked); // the dead end never reached the device
    std.debug.print("structured_users users dead end: structured {any} failed alone (failure_retired), peer {any} = plain pair's\n", .{ s.got(), p.got() });
}

test "structured_users users: a structured lane whose matcher refuses its drawn token at commit leaves the shared round alone (failed, failure_retired); its plain peer finishes as in a plain pair" {
    var n: Served = undefined;
    try n.init(false, &.{});
    defer n.deinit();
    var s_ids = promptIds(21, 1000);
    var p_ids = promptIds(30, 3000);
    const rp: api.Request = .{ .prompt = &p_ids, .max_tokens = 5, .drafts = false };
    const rq: api.Request = .{ .prompt = &s_ids, .max_tokens = 5, .drafts = false };
    var a: Client = .{ .id = 1 };
    var b: Client = .{ .id = 2 };
    try n.ask(&a, &rq);
    try n.ask(&b, &rp);
    try n.host.start();
    try waitTrue(&a.done);
    try waitTrue(&b.done);
    n.stop();
    const steps = [_][]const u32{ &.{41}, &.{42}, &.{43} };
    var scripts = [_]Script{.{ .steps = &steps, .refuse_at = 1 }};
    var m: Served = undefined;
    try m.init(true, &scripts);
    defer m.deinit();
    const rs: api.Request = .{ .prompt = &s_ids, .max_tokens = 5, .drafts = false, .structure = json };
    var s: Client = .{ .id = 1 };
    var p: Client = .{ .id = 2 };
    try m.ask(&s, &rs);
    try m.ask(&p, &rp);
    try m.host.start();
    try waitTrue(&s.done);
    try waitTrue(&p.done);
    m.stop();
    try t.expectEqual(api.Reason.failed, s.reason.?);
    try t.expectEqualSlices(u32, &.{41}, s.got()); // 42 drawn under its mask, refused at commit: never emitted
    try t.expectEqual(@as(usize, 1), scripts[0].closes);
    try t.expectEqual(@as(usize, 2), m.mock.masked); // the first token's and the refused round's
    try t.expectEqual(api.Reason.length, p.reason.?);
    try t.expectEqualSlices(u32, b.got(), p.got());
    try t.expectEqual(@as(usize, 1), m.count(.failure_retired));
    try t.expectEqual(@as(usize, 0), m.count(.cancel_retired));
    for (m.back.slots[0..2]) |slot| try t.expect(slot.owner == null and !slot.deferred);
    std.debug.print("structured_users users rejected at commit: structured {any} failed \"{s}\" alone (failure_retired), peer {any} = plain pair's\n", .{ s.got(), s.message, p.got() });
}
