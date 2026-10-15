//! qualification_4be15bbf's gates for tf-glm-afd attn serve's lane side (afd_serve_lanes.zig over glm.afd_serve), on afd_attn_fail.zig's
//! three nodes in one process (the experts on fn_xchg's CPU kernels, the attention's forward recorded), each request
//! through api.LaneHost and the lane core as the server sends it:
//! - G-trace: one served request's launches (sha256 by buffer name, as glm_cuda_offtrace_test.zig digests) and exchange
//!   messages (every request posted, in order) equal the tool's own run of the same prompt (prefill, serial decode);
//! - G-dead-half: a half that dies, or one that goes silent, mid-forward fails the in-flight request within the
//!   exchange's timeouts (the harness's 300 ms silent rule), a later request fails at once with nothing launched or
//!   sent, the monitor stops the server once the lane host is idle, and every thread joins (nothing hangs).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const api = @import("engine_api");
const fail = @import("afd_attn_fail.zig");
const sl = @import("afd_serve_lanes.zig");

const xchg = cuda.xchg;
const ae = glm.afd_expert;
const trace = glm.trace;
const testing = std.testing;
const gpa = testing.allocator;
const ms = std.time.ns_per_ms;
const Sha = std.crypto.hash.sha2.Sha256;

fn put(h: *Sha, comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    h.update(std.fmt.bufPrint(&buf, fmt, args) catch unreachable);
}

fn addr(h: *Sha, space: *const trace.Space, x: u64) void {
    if (space.name(x)) |p| put(h, "@{s}+{d}", .{ p.path, p.off }) else put(h, "#{d}", .{x});
}

fn val(h: *Sha, space: *const trace.Space, v: trace.Val) void {
    switch (v) {
        .t => |t| {
            addr(h, space, t.ptr);
            put(h, " {t} {d} {any} {any};", .{ t.dtype, t.dims, t.shape, t.stride });
        },
        .list => |l| {
            put(h, "[{d}", .{l.len});
            for (l) |u| val(h, space, u);
            h.update("]");
        },
        .int => |x| put(h, "i{d};", .{x}),
        .f32 => |x| put(h, "f{d};", .{x}),
        .bool => |x| put(h, "b{};", .{x}),
        .none => h.update("none;"),
    }
}

/// The launches' digest, addresses by buffer name (glm_cuda_offtrace_test.zig's), so two harnesses' traces compare.
fn digest(space: *const trace.Space, events: []const trace.Event) [64]u8 {
    var h = Sha.init(.{});
    for (events) |e| {
        put(&h, "{t} {s} {any}|", .{ e.op, e.name, e.grid });
        for (e.args) |a| {
            put(&h, "{s}=", .{a.name});
            switch (a.value) {
                .ptr => |p| {
                    put(&h, "{s}", .{p.ty});
                    addr(&h, space, p.addr);
                },
                .i32 => |x| put(&h, "i32 {d}", .{x}),
                .f32 => |x| put(&h, "f32 {x}", .{@as(u32, @bitCast(x))}),
                .u64 => |x| put(&h, "u64 {d}", .{x}),
            }
            h.update(",");
        }
        for (e.consts) |c| {
            put(&h, "{s}:", .{c.name});
            if (c.int) |x| put(&h, "{d}", .{x});
            if (c.f32) |x| put(&h, "f{x}", .{@as(u32, @bitCast(x))});
            h.update(",");
        }
        for (e.vals) |v| val(&h, space, v);
        h.update("\n");
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.bytesToHex(d, .lower);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn nap(n: u64) void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(n) };
    _ = std.c.nanosleep(&ts, null);
}

/// One request's events as the lane thread sends them.
const Got = struct {
    tokens: [64]u32 = undefined,
    n: usize = 0,
    reason: api.Reason = .stop,
    msg: [320]u8 = undefined,
    msg_len: usize = 0,
    at: u64 = 0,
    done: std.atomic.Value(bool) = .init(false),

    fn sink(g: *Got) api.Sink {
        return .{ .ctx = g, .event = event };
    }

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .tokens => |t| for (t) |x| {
                if (g.n < g.tokens.len) g.tokens[g.n] = x;
                g.n += 1;
            },
            .finished => |f| {
                g.reason = f.reason;
                g.msg_len = @min(f.message.len, g.msg.len);
                @memcpy(g.msg[0..g.msg_len], f.message[0..g.msg_len]);
                g.at = nowNs();
                g.done.store(true, .release);
            },
            else => {},
        }
    }

    /// Its end within `most` ns, else error.Hung.
    fn wait(g: *Got, most: u64) !void {
        const t0 = nowNs();
        while (!g.done.load(.acquire)) {
            if (nowNs() - t0 > most) return error.Hung;
            nap(5 * ms);
        }
    }
};

/// tf-glm-afd attn serve's lane side over the harness's attention node: one recorded Fwd for the stream (as attn serve
/// keeps one), the served calls, qualification_qualification_e91d6b91's backend, the words and the lane host.
const Served = struct {
    f: glm.fwd.Fwd = undefined,
    s: glm.afd_serve.Serve = undefined,
    b: sl.Bound = undefined,
    back: sl.Backend = undefined,
    words: sl.Words = undefined,
    ln: sl.Lanes = .{ .gpa = gpa },
    up: bool = false,

    fn start(x: *Served, h: *fail.Harness) !void {
        x.f = h.fwd();
        x.s = .{ .at = &h.at, .f = &x.f };
        x.b = .{ .s = &x.s };
        const cap = h.e.st.capacity;
        x.back = .{ .steps = &x.b, .capacity = cap, .reach = sl.reachOf(cap, h.m.denseLimit()) };
        x.words = .{ .node = h.attn() };
        try x.ln.start(testing.io, &x.back, .{ .lanes = 1, .context_window = @intCast(cap - sl.slack), .drafts = "off" }, &x.words);
        x.up = true;
    }

    fn ask(x: *Served, id: api.Id, req: *const api.Request, g: *Got) !void {
        try x.ln.host.engine().submit(id, req, g.sink());
    }

    fn stop(x: *Served) void {
        if (!x.up) return;
        x.up = false;
        x.ln.stop();
    }
};

/// The monitor's stop in a test: counted, as the binary's SIGTERM would stop Ash's server.
const Stopped = struct {
    fired: std.atomic.Value(u32) = .init(0),

    fn fire(ctx: ?*anyopaque) void {
        const s: *Stopped = @ptrCast(@alignCast(ctx.?));
        _ = s.fired.fetchAdd(1, .acq_rel);
    }
};

fn prompt40(base: u32) [40]u32 {
    var p: [40]u32 = undefined;
    for (&p, 0..) |*v, i| v.* = base + @as(u32, @intCast(i));
    return p;
}

test "G-trace (qualification_4be15bbf): a served request's launches and exchange messages equal the tool's own run of the same prompt (prefill, then serial decode)" {
    const prompt = prompt40(300);
    const n = 6;
    // the tool's run (attn's prefill and Attn.decode, one Fwd: what tf-glm-afd attn first, teacher and bench drive)
    var ht: fail.Harness = .{ .io = testing.io, .fault = fail.no_fault };
    try ht.start();
    defer ht.free();
    var f = ht.fwd();
    var out: [n]u32 = undefined;
    _ = try ht.at.decode(&f, try ht.at.prefill(&f, &prompt), n, &out, false);
    try f.flush();
    fail.settle();
    const tool_d = digest(&ht.space, ht.t.events.items);
    var tool_x: [32]u8 = undefined;
    ht.ah.sent.final(&tool_x);
    const tool_c = ht.counts();
    const tool_launches = ht.t.events.items.len;
    ht.stop();
    // the served request: the same prompt and n tokens through the lane host (greedy, EOS ignored, drafts off)
    var hs: fail.Harness = .{ .io = testing.io, .fault = fail.no_fault };
    try hs.start();
    defer hs.free();
    var x: Served = .{};
    try x.start(&hs);
    defer x.stop();
    var got: Got = .{};
    const req: api.Request = .{ .prompt = &prompt, .max_tokens = n, .drafts = false };
    try x.ask(1, &req, &got);
    try got.wait(10 * std.time.ns_per_s);
    x.stop();
    try x.f.flush();
    fail.settle();
    const served_d = digest(&hs.space, hs.t.events.items);
    var served_x: [32]u8 = undefined;
    hs.ah.sent.final(&served_x);
    const served_c = hs.counts();
    hs.stop();
    std.debug.print("G-trace: the tool {d} launches {s}, {d} requests; served {d} launches {s}, {d} requests, {d} tokens ({t}); exchange messages {x} / {x}\n", .{ tool_launches, tool_d[0..16], tool_c.rung, hs.t.events.items.len, served_d[0..16], served_c.rung, got.n, got.reason, tool_x[0..8], served_x[0..8] });
    try testing.expectEqual(api.Reason.length, got.reason);
    try testing.expectEqual(@as(usize, n), got.n);
    try testing.expectEqualSlices(u32, &out, got.tokens[0..n]);
    try testing.expectEqual(tool_launches, hs.t.events.items.len);
    try testing.expectEqualStrings(&tool_d, &served_d);
    try testing.expectEqualSlices(u8, &tool_x, &served_x);
    try testing.expectEqual(tool_c, served_c);
    try testing.expectEqual(@as(u32, n * fail.layers), served_c.rung); // the prompt's one chunk and n - 1 steps
    try testing.expectEqual(@as(u64, 0), hs.ah.wrong);
    try testing.expectEqual(@as(usize, 40 + n - 1), hs.e.st.pos);
}

/// G-dead-half's run: a 40-token request of 8 under `fault` fails; then a second one; the monitor and the joins.
fn deadHalf(fault: [2]?ae.Fault, want: xchg.Failure, emitted: usize, least: u64, most: u64) !void {
    var h: fail.Harness = .{ .io = testing.io, .fault = fault };
    try h.start();
    defer h.free();
    var x: Served = .{};
    try x.start(&h);
    defer x.stop();
    var stopped: Stopped = .{};
    var mon: sl.Monitor = .{ .node = h.attn(), .engine = x.ln.host.engine(), .grace_ns = 5 * std.time.ns_per_s, .stop = Stopped.fire, .stop_ctx = &stopped };
    mon.listening.store(true, .release); // as attn serve's on_listen sets it
    try mon.start();
    defer mon.finish();
    const prompt = prompt40(500);
    var got: Got = .{};
    const t0 = nowNs();
    try x.ask(1, &.{ .prompt = &prompt, .max_tokens = 8, .drafts = false }, &got);
    try got.wait(10 * std.time.ns_per_s); // nothing hangs: the request ends within the exchange's timeouts
    const took = got.at - t0;
    const failed = h.at.failure() orelse return error.NoFailure;
    // the monitor stops the server once the failed request is answered (the lane host idle)
    const s0 = nowNs();
    while (stopped.fired.load(.acquire) == 0 and nowNs() - s0 < 3 * std.time.ns_per_s) nap(5 * ms);
    // a later request fails at once: nothing launched, nothing sent
    try x.f.flush();
    const launches = h.t.events.items.len;
    const before = h.counts();
    var again: Got = .{};
    const t1 = nowNs();
    try x.ask(2, &.{ .prompt = &prompt, .max_tokens = 8, .drafts = false }, &again);
    try again.wait(2 * std.time.ns_per_s);
    const refused_in = again.at - t1;
    mon.finish();
    x.stop();
    try x.f.flush();
    fail.settle();
    const after = h.counts();
    var buf: [160]u8 = undefined;
    std.debug.print("G-dead-half: {s}: the request failed after {d} tokens in {d} ms ({s}); the monitor stopped the server {d} ms after the failure; the next request refused in {d} ms ({s})\n", .{ glm.afd_engine.describe(failed, &buf), got.n, took / ms, got.msg[0..got.msg_len], mon.waited_ns / ms, refused_in / ms, again.msg[0..again.msg_len] });
    h.stop();
    try testing.expectEqual(api.Reason.failed, got.reason);
    try testing.expectEqual(emitted, got.n);
    try testing.expectEqual(want.why, failed.why);
    try testing.expectEqual(want.peer, failed.peer);
    try testing.expectEqual(want.seq, failed.seq);
    try testing.expect(took >= least and took < most);
    try testing.expectEqual(@as(u32, 1), stopped.fired.load(.acquire));
    try testing.expect(mon.waited_ns < 1000 * ms); // idle once the failed request was answered: not the 5 s grace
    try testing.expectEqual(api.Reason.failed, again.reason);
    try testing.expect(std.mem.indexOf(u8, again.msg[0..again.msg_len], "an AFD expert half failed") != null);
    try testing.expect(refused_in < 500 * ms);
    try testing.expectEqual(launches, h.t.events.items.len);
    try testing.expectEqual(before, after);
    try testing.expectEqual(@as(u64, 0), h.ah.wrong);
}

test "G-dead-half (qualification_4be15bbf): expert 0 dies at request 4 (the first decode step's second MoE layer: mid-forward); the served request fails at once, the next is refused, the monitor stops the server, nothing hangs" {
    try deadHalf(.{ .{ .what = .exit, .at = 4 }, null }, .{ .why = .dead, .peer = 0, .seq = 4, .gave_up = 0, .breach = 0 }, 1, 0, 1000 * ms);
}

test "G-dead-half (qualification_4be15bbf): expert 1 goes silent from request 6 (the second decode step's second MoE layer: mid-forward); the served request fails at the watcher's 300 ms, the next is refused, the monitor stops the server, nothing hangs" {
    try deadHalf(.{ null, .{ .what = .silent, .at = 6 } }, .{ .why = .silent, .peer = 1, .seq = 6, .gave_up = 0, .breach = 0 }, 2, 300 * ms, 2000 * ms);
}
