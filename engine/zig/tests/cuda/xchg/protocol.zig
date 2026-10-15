//! The exchange's protocol on the CPU against fake peers (qualification_12f4270a, qualification_3df82c97): three nodes in one process, real control links on loopback, fake proxies between the regions, fn_xchg.cu's kernels on CPU threads.
//! Exact counts: requests, replies, values, proxy posts and status words; a silent expert, a dead one, a breach and a failed write each fail the request in bounds, and the attention goes on.
const std = @import("std");
const cuda = @import("cuda");
const xchg = cuda.xchg;
const proto = xchg.proto;
const lab = @import("trace_lab.zig");
const fake = @import("fake.zig");
const Cpu = @import("trace_cpu.zig").Cpu;
const net = std.Io.net;
const trace = @import("trace.zig");

const gpa = std.testing.allocator;
const layout: proto.Layout = .{ .d = 64, .s = 9, .bf16 = true };
const experts = 16;
const max_rows = 32;
const ms = std.time.ns_per_ms;
const no_faults: [2]fake.Faults = .{ .{}, .{} };

const Harness = struct {
    io: std.Io,
    regions: [3][]align(4096) u8 = undefined,
    status: [3][]u32 = undefined,
    nodes: [3]xchg.Node = undefined,
    openers: [3]fake.Opener = undefined,
    cpus: [3]Cpu = .{ .{}, .{}, .{} },
    server: cuda.link.Server = undefined,
    threads: [2]?std.Thread = .{ null, null },
    ends: [2]lab.ExpertEnd = .{ .{ .how = .lost, .why = "never ran" }, .{ .how = .lost, .why = "never ran" } },
    expert_opts: [2]lab.ExpertOpts = undefined,
    bufs: lab.AttnBufs = undefined,
    fseq: i32 = 0,
    /// qualification_17c70e37: each fake proxy's pause after a stripe
    stripe_pause_ns: u64 = 0,
    recorder: ?*trace.Recorder = null,
    monitor: ?@import("trace_node.zig").Monitor = null,

    fn config(rules: xchg.watcher.Rules, give_up_ns: u64) xchg.Config {
        return .{ .settings = .{ .layout = layout, .max_rows = max_rows, .ports = 1 }, .rules = rules, .give_up_ns = give_up_ns, .beat_ns = 50 * ms, .setup_timeout_ms = 10_000, .warn_failures = true };
    }

    /// qualification_17c70e37: `cfg` with 8 stripes of at least 256 B (this harness's messages are 0.6-8 KiB: the small ones stay one write)
    fn striped(cfg: xchg.Config) xchg.Config {
        var c = cfg;
        c.settings.stripes = 8;
        c.settings.stripe_min = 256;
        return c;
    }

    /// Three nodes up: the experts join from their threads (then run their loops), the attention attends here.
    fn start(h: *Harness, cfg: xchg.Config, expert_cfg: [2]xchg.Config, faults: [3][2]fake.Faults, eopts: [2]lab.ExpertOpts) !void {
        h.expert_opts = eopts;
        h.recorder = try trace.Recorder.create(gpa, @intFromPtr(h), if (cfg.run_watcher) 200 else 201);
        for (&h.cpus, 0..) |*c, i| c.observer = h.recorder.?.writer(@intCast(i));
        const total = (proto.Region{ .slot = cfg.settings.slotBytes() }).total();
        for (0..3) |i| {
            h.regions[i] = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(total));
            @memset(h.regions[i], 0);
            h.status[i] = try gpa.alloc(u32, proto.STATUS_WORDS);
            @memset(h.status[i], 0);
            h.openers[i] = .{ .gpa = gpa, .faults = faults[i], .stripe_pause_ns = h.stripe_pause_ns, .recorder = h.recorder, .owner = @intCast(i) };
            const role: xchg.Role = if (i == 0) .attn else .{ .expert = @intCast(i - 1) };
            const mem: xchg.Memory = .{ .region = h.regions[i], .region_dev = @intFromPtr(h.regions[i].ptr), .status = h.status[i], .status_dev = @intFromPtr(h.status[i].ptr) };
            h.nodes[i] = try xchg.Node.init(gpa, h.io, role, if (i == 0) cfg else expert_cfg[i - 1], mem);
        }
        h.server = try cuda.link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
        const master = try net.IpAddress.parse("127.0.0.1", h.server.port());
        for (0..2) |k| h.threads[k] = try std.Thread.spawn(.{}, expertMain, .{ h, k, master });
        h.bufs = try lab.AttnBufs.init(h.engine(), gpa, layout, max_rows);
        try h.nodes[0].attend(h.server, h.openers[0].opener());
        h.monitor = .{ .node = h.attn(), .writer = h.recorder.?.writer(7) };
        try h.monitor.?.start();
    }

    fn expertMain(h: *Harness, k: usize, master: net.IpAddress) void {
        const n = &h.nodes[k + 1];
        n.join(master, h.openers[k + 1].opener()) catch |e| {
            h.ends[k] = .{ .how = .lost, .why = @errorName(e) };
            return;
        };
        h.cpus[k+1].observe(50, n);
        h.ends[k] = lab.expertLoop(gpa, .{ .cpu = &h.cpus[k + 1] }, n, h.expert_opts[k]) catch |e| .{ .how = .lost, .why = @errorName(e) };
        h.cpus[k+1].observe(51, n);
        if (h.ends[k].how == .die) n.deinit(); // the process is gone: its link and proxy close
    }

    fn attn(h: *Harness) *xchg.Node {
        return &h.nodes[0];
    }

    fn engine(h: *Harness) lab.Engine {
        return .{ .cpu = &h.cpus[0] };
    }

    fn opts(h: *Harness) lab.AttnOpts {
        _ = h;
        return .{ .layout = layout, .experts = experts, .rows = &rows_list, .repeats = 4, .prompt_from = 16, .verbose = false };
    }

    /// Exchanges of one row until the exchange fails (at most `most`): how many succeeded, and the failing one's time from post to return.
    fn untilFailure(h: *Harness, most: usize) !Until {
        return h.untilFailureAt(most, 1);
    }

    const Until = struct { ok: u64, fail_ns: u64 };

    /// `untilFailure` with exchanges of `rows` rows (qualification_17c70e37: large enough to go in stripes).
    fn untilFailureAt(h: *Harness, most: usize, rows: usize) !Until {
        var ok: u64 = 0;
        while (ok < most) {
            h.fseq += 1;
            const t0 = h.attn().nowNs();
            const wrong = try lab.exchange(h.engine(), h.attn(), &h.bufs, h.opts(), h.fseq, rows, true);
            const took = h.attn().nowNs() - t0;
            if (h.attn().failure() != null) return .{ .ok = ok, .fail_ns = took };
            try std.testing.expectEqual(@as(u64, 0), wrong);
            ok += 1;
        }
        return error.NoFailure;
    }

    const Counts = struct { rung: u32, requests: [2]u64, replies: [2]?u64, send_kept: bool };

    /// The attention's two send slots, copied (to compare after a failure).
    fn sendCopy(h: *Harness) ![]u8 {
        const reg = h.attn().reg;
        return gpa.dupe(u8, h.regions[0][@intCast(reg.send(0))..][0..@intCast(2 * reg.slot)]);
    }

    /// After a failure: 100 ms for anything still rung to cross, then the attention's doorbell, each pair's requests,
    /// each live expert's replies (null for the dead one), and whether the send slots still hold `before`.
    fn settle(h: *Harness, dead: ?usize, before: []const u8) Counts {
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * ms };
        _ = std.c.nanosleep(&ts, null);
        const reg = h.attn().reg;
        var c: Counts = .{ .rung = h.attn().doorbell(), .requests = undefined, .replies = undefined, .send_kept = std.mem.eql(u8, before, h.regions[0][@intCast(reg.send(0))..][0..before.len]) };
        for (0..2) |k| {
            c.requests[k] = h.attn().proxies[k].?.ops();
            c.replies[k] = if (dead != null and dead.? == k) null else h.nodes[k + 1].proxies[0].?.ops();
        }
        return c;
    }

    fn finishTrace(h: *Harness) void {
        const r = h.recorder orelse return;
        r.joined = true; r.dump();
        gpa.destroy(r); h.recorder = null;
    }

    /// The attention's clean end, then every node down (an expert that died is down already).
    fn stop(h: *Harness) void {
        h.cpus[0].observe(60, h.attn());
        h.nodes[0].stopExperts();
        for (h.threads) |t| if (t) |x| x.join();
        h.cpus[0].observe(61, h.attn());
        if (h.monitor) |*m| m.join();
        h.nodes[0].deinit();
        for (1..3) |i| if (h.ends[i - 1].how != .die) h.nodes[i].deinit();
        h.finishTrace();
        h.bufs.deinit(h.engine(), gpa);
        h.server.close();
        for (h.regions) |r| gpa.free(r);
        for (h.status) |s| gpa.free(s);
    }
};

const rows_list = [_]usize{ 1, 2, 3, 4, 7, 8, 16, 31, 32 };

fn expertOpts(silent_at: ?u32, exit_at: ?u32) lab.ExpertOpts {
    return .{ .layout = layout, .experts = experts, .max_rows = max_rows, .silent_at = silent_at, .exit_at = exit_at };
}

test "exact over 1-32 rows with values new every forward: requests, replies, posts and status counted exactly" {
    var h: Harness = .{ .io = std.testing.io };
    const cfg = Harness.config(.{}, 20 * std.time.ns_per_s);
    try h.start(cfg, .{ cfg, cfg }, .{ no_faults, no_faults, no_faults }, .{ expertOpts(null, null), expertOpts(null, null) });
    const rep = lab.sweep(h.engine(), h.attn(), &h.bufs, h.opts(), &h.fseq) catch |e| {
        h.stop();
        return e;
    };
    var ops: [4]u64 = undefined;
    for (0..2) |k| ops[k] = h.attn().proxies[k].?.ops();
    for (1..3) |i| ops[i + 1] = h.nodes[i].proxies[0].?.ops();
    const status_clear = std.mem.allEqual(u32, h.status[0], 0);
    const rung = h.attn().doorbell();
    h.stop();
    try std.testing.expectEqual(@as(u64, rows_list.len * 4), rep.exchanges);
    try std.testing.expectEqual(@as(u64, 0), rep.wrong);
    try std.testing.expectEqual(@as(?xchg.Failure, null), rep.failed);
    try std.testing.expectEqual([4]u64{ 36, 36, 36, 36 }, ops);
    try std.testing.expect(status_clear);
    try std.testing.expectEqual(@as(u32, 36), rung);
    var rows_sum: u64 = 0;
    for (rows_list) |r| rows_sum += 4 * r;
    for (h.ends) |e| {
        try std.testing.expectEqual(.stopped, e.how);
        try std.testing.expectEqual(@as(u64, 36), e.requests);
        try std.testing.expectEqual(rows_sum, e.rows);
        try std.testing.expectEqual(@as(u64, 0), e.wrong);
    }
}

test "a silent expert fails its request at the watcher's 300 ms, not the GPU's 5 s; three later requests fail at once and send nothing; the attention goes on" {
    var h: Harness = .{ .io = std.testing.io };
    const cfg = Harness.config(.{ .silent_ns = 300 * ms }, 5 * std.time.ns_per_s);
    try h.start(cfg, .{ cfg, cfg }, .{ no_faults, no_faults, no_faults }, .{ expertOpts(null, null), expertOpts(5, null) });
    const r = h.untilFailure(100) catch |e| {
        h.stop();
        return e;
    };
    const f = h.attn().failure().?;
    const before = try h.sendCopy();
    defer gpa.free(before);
    const after = try lab.afterFailure(h.io, h.engine(), h.attn(), &h.bufs, h.opts(), 3, 50 * ms, &h.fseq);
    const c = h.settle(null, before);
    h.stop();
    try std.testing.expectEqual(@as(u64, 4), r.ok);
    try std.testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 5, .gave_up = 0, .breach = 0 }, f);
    try std.testing.expect(r.fail_ns >= 300 * ms and r.fail_ns < 2000 * ms);
    try std.testing.expectEqual(@as(u64, 3), after.fast);
    // nothing crossed after request 5: the doorbell stayed, each pair carried 1-5, expert 0 answered 1-5 and expert 1 1-4
    try std.testing.expectEqual(Harness.Counts{ .rung = 5, .requests = .{ 5, 5 }, .replies = .{ 5, 4 }, .send_kept = true }, c);
    try std.testing.expectEqual(.stopped, h.ends[0].how);
    try std.testing.expectEqual(@as(u64, 5), h.ends[0].requests);
    try std.testing.expectEqual(@as(u64, 0), h.ends[0].wrong);
    try std.testing.expectEqual(.stopped, h.ends[1].how);
    try std.testing.expectEqual(@as(u64, 4), h.ends[1].requests);
}

test "a dead expert (link closed, proxy gone) fails its request at once, long before the 5 s silence rule; nothing is sent after" {
    var h: Harness = .{ .io = std.testing.io };
    const cfg = Harness.config(.{ .silent_ns = 5 * std.time.ns_per_s }, 10 * std.time.ns_per_s);
    try h.start(cfg, .{ cfg, cfg }, .{ no_faults, no_faults, no_faults }, .{ expertOpts(null, 3), expertOpts(null, null) });
    const r = h.untilFailure(100) catch |e| {
        h.stop();
        return e;
    };
    const f = h.attn().failure().?;
    const before = try h.sendCopy();
    defer gpa.free(before);
    const after = try lab.afterFailure(h.io, h.engine(), h.attn(), &h.bufs, h.opts(), 3, 50 * ms, &h.fseq);
    const c = h.settle(0, before);
    h.stop();
    try std.testing.expectEqual(@as(u64, 2), r.ok);
    try std.testing.expectEqual(xchg.Failure{ .why = .dead, .peer = 0, .seq = 3, .gave_up = 0, .breach = 0 }, f);
    try std.testing.expect(r.fail_ns < 1000 * ms);
    try std.testing.expectEqual(@as(u64, 3), after.fast);
    // the live expert 1 got requests 1-3 and nothing after; the send slots were not rewritten
    try std.testing.expectEqual(Harness.Counts{ .rung = 3, .requests = .{ 3, 3 }, .replies = .{ null, 3 }, .send_kept = true }, c);
    try std.testing.expectEqual(.die, h.ends[0].how);
    try std.testing.expectEqual(@as(u64, 2), h.ends[0].requests);
    try std.testing.expectEqual(.stopped, h.ends[1].how);
    try std.testing.expectEqual(@as(u64, 3), h.ends[1].requests);
    try std.testing.expectEqual(@as(u64, 0), h.ends[1].wrong);
}

test "without the watcher the GPU's bounded wait gives up once at 200 ms, and never spins again" {
    var h: Harness = .{ .io = std.testing.io };
    var cfg = Harness.config(.{ .silent_ns = 60 * std.time.ns_per_s, .beat_stale_ns = 60 * std.time.ns_per_s }, 200 * ms);
    cfg.run_watcher = false;
    try h.start(cfg, .{ cfg, cfg }, .{ no_faults, no_faults, no_faults }, .{ expertOpts(null, null), expertOpts(3, null) });
    const r = h.untilFailure(100) catch |e| {
        h.stop();
        return e;
    };
    const f = h.attn().failure().?;
    const before = try h.sendCopy();
    defer gpa.free(before);
    const after = try lab.afterFailure(h.io, h.engine(), h.attn(), &h.bufs, h.opts(), 3, 50 * ms, &h.fseq);
    const f2 = h.attn().failure().?;
    const c = h.settle(null, before);
    h.stop();
    try std.testing.expectEqual(@as(u64, 2), r.ok);
    try std.testing.expectEqual(xchg.Failure{ .why = .gave_up, .peer = 1, .seq = 3, .gave_up = 1, .breach = 0 }, f);
    try std.testing.expect(r.fail_ns >= 200 * ms and r.fail_ns < 2000 * ms);
    try std.testing.expectEqual(@as(u64, 3), after.fast);
    try std.testing.expectEqual(f, f2); // no second give-up: later waits return before they spin
    try std.testing.expectEqual(Harness.Counts{ .rung = 3, .requests = .{ 3, 3 }, .replies = .{ 3, 2 }, .send_kept = true }, c);
}

test "a reply flag past its seq is a breach, failed at once; a failed write in a pair is the proxy's failure" {
    {
        var h: Harness = .{ .io = std.testing.io };
        const cfg = Harness.config(.{}, 10 * std.time.ns_per_s);
        // expert 1 is rank 1 of its pair: its proxy writes request 4's reply flag as 5
        try h.start(cfg, .{ cfg, cfg }, .{ no_faults, no_faults, .{ .{}, .{ .breach_at = 4 } } }, .{ expertOpts(null, null), expertOpts(null, null) });
        const r = h.untilFailure(100) catch |e| {
            h.stop();
            return e;
        };
        const f = h.attn().failure().?;
        h.stop();
        try std.testing.expectEqual(@as(u64, 3), r.ok);
        try std.testing.expectEqual(xchg.Failure{ .why = .breach, .peer = 1, .seq = 4, .gave_up = 0, .breach = 1 }, f);
        try std.testing.expect(r.fail_ns < 1000 * ms);
    }
    {
        var h: Harness = .{ .io = std.testing.io };
        const cfg = Harness.config(.{}, 10 * std.time.ns_per_s);
        // the attention is rank 0 of its pair with expert 1: that proxy fails to write request 6
        try h.start(cfg, .{ cfg, cfg }, .{ .{ .{ .fail_at = 6 }, .{} }, no_faults, no_faults }, .{ expertOpts(null, null), expertOpts(null, null) });
        const r = h.untilFailure(100) catch |e| {
            h.stop();
            return e;
        };
        const f = h.attn().failure().?;
        h.stop();
        try std.testing.expectEqual(@as(u64, 5), r.ok);
        try std.testing.expectEqual(xchg.Failure{ .why = .proxy, .peer = 1, .seq = 6, .gave_up = 0, .breach = 0 }, f);
        try std.testing.expect(r.fail_ns < 1000 * ms);
    }
}

test "the settings must agree: an expert with another layout is refused on both ends" {
    var h: Harness = .{ .io = std.testing.io };
    const cfg = Harness.config(.{}, 10 * std.time.ns_per_s);
    var other = cfg;
    other.settings.layout.s = 8;
    try std.testing.expectError(error.SettingsDiffer, h.start(cfg, .{ cfg, other }, .{ no_faults, no_faults, no_faults }, .{ expertOpts(null, null), expertOpts(null, null) }));
    // the attention refused before its pairs: close what is up (links, then the listener's backlog), then the experts' ends
    for (h.nodes[0].links) |l| if (l) |x| x.close();
    h.nodes[0].links = .{ null, null };
    h.server.close();
    for (h.threads) |t| if (t) |x| x.join();
    for (1..3) |i| h.nodes[i].deinit();
    h.nodes[0].deinit();
    h.finishTrace();
    h.bufs.deinit(h.engine(), gpa);
    for (h.regions) |r| gpa.free(r);
    for (h.status) |s| gpa.free(s);
    try std.testing.expectEqualStrings("SettingsDiffer", h.ends[1].why);
}

test "qualification_17c70e37: in 8 stripes (each fake pausing 200 us after a stripe), exact over 1-32 rows: the same counts as one write, the larger messages taken stripe by stripe, the small ones as before" {
    // the attention's takes over 3 blocks of 3 parts a stripe, expert 0's over 2 blocks: each block's units in turn
    var h: Harness = .{ .io = std.testing.io, .stripe_pause_ns = 200_000, .cpus = .{ .{ .blocks = 3, .parts = 3 }, .{ .blocks = 2 }, .{} } };
    const cfg = Harness.striped(Harness.config(.{}, 20 * std.time.ns_per_s));
    try h.start(cfg, .{ cfg, cfg }, .{ no_faults, no_faults, no_faults }, .{ expertOpts(null, null), expertOpts(null, null) });
    const rep = lab.sweep(h.engine(), h.attn(), &h.bufs, h.opts(), &h.fseq) catch |e| {
        h.stop();
        return e;
    };
    var ops: [4]u64 = undefined;
    for (0..2) |k| ops[k] = h.attn().proxies[k].?.ops();
    for (1..3) |i| ops[i + 1] = h.nodes[i].proxies[0].?.ops();
    const status_clear = std.mem.allEqual(u32, h.status[0], 0);
    const striped = [3]u64{ h.cpus[0].striped, h.cpus[1].striped, h.cpus[2].striped };
    h.stop();
    try std.testing.expectEqual(@as(u64, rows_list.len * 4), rep.exchanges);
    try std.testing.expectEqual(@as(u64, 0), rep.wrong);
    try std.testing.expectEqual(@as(?xchg.Failure, null), rep.failed);
    try std.testing.expectEqual([4]u64{ 36, 36, 36, 36 }, ops);
    try std.testing.expect(status_clear);
    for (h.ends) |e| {
        try std.testing.expectEqual(.stopped, e.how);
        try std.testing.expectEqual(@as(u64, 36), e.requests);
        try std.testing.expectEqual(@as(u64, 0), e.wrong);
    }
    // replies of 8+ rows (2 KiB fp32 at 8, bf16 from 16) and requests of 8+ rows go in stripes; 1-7 rows stay one write
    try std.testing.expectEqual([3]u64{ 16, 16, 16 }, striped);
}

test "qualification_17c70e37: in stripes, a silent expert still fails its request at the watcher's 300 ms and nothing is sent after; a stripe's word past its seq is a breach" {
    {
        var h: Harness = .{ .io = std.testing.io, .stripe_pause_ns = 100_000 };
        const cfg = Harness.striped(Harness.config(.{ .silent_ns = 300 * ms }, 5 * std.time.ns_per_s));
        try h.start(cfg, .{ cfg, cfg }, .{ no_faults, no_faults, no_faults }, .{ expertOpts(null, null), expertOpts(5, null) });
        const r = h.untilFailureAt(100, 16) catch |e| {
            h.stop();
            return e;
        };
        const f = h.attn().failure().?;
        const before = try h.sendCopy();
        defer gpa.free(before);
        const after = try lab.afterFailure(h.io, h.engine(), h.attn(), &h.bufs, h.opts(), 3, 50 * ms, &h.fseq);
        const c = h.settle(null, before);
        const striped = h.cpus[0].striped;
        h.stop();
        try std.testing.expectEqual(@as(u64, 4), r.ok);
        try std.testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 5, .gave_up = 0, .breach = 0 }, f);
        try std.testing.expect(r.fail_ns >= 300 * ms and r.fail_ns < 2000 * ms);
        try std.testing.expectEqual(@as(u64, 3), after.fast);
        try std.testing.expectEqual(Harness.Counts{ .rung = 5, .requests = .{ 5, 5 }, .replies = .{ 5, 4 }, .send_kept = true }, c);
        try std.testing.expectEqual(@as(u64, 5), striped); // the four answered and the silent one; the three after it are one row
    }
    {
        var h: Harness = .{ .io = std.testing.io, .stripe_pause_ns = 100_000 };
        const cfg = Harness.striped(Harness.config(.{}, 10 * std.time.ns_per_s));
        // expert 1 is rank 1 of its pair: its proxy writes every word of request 4's reply as 5
        try h.start(cfg, .{ cfg, cfg }, .{ no_faults, no_faults, .{ .{}, .{ .breach_at = 4 } } }, .{ expertOpts(null, null), expertOpts(null, null) });
        const r = h.untilFailureAt(100, 16) catch |e| {
            h.stop();
            return e;
        };
        const f = h.attn().failure().?;
        const striped = h.cpus[0].striped;
        h.stop();
        try std.testing.expectEqual(@as(u64, 3), r.ok);
        try std.testing.expectEqual(xchg.Failure{ .why = .breach, .peer = 1, .seq = 4, .gave_up = 0, .breach = 1 }, f);
        try std.testing.expect(r.fail_ns < 1000 * ms);
        try std.testing.expectEqual(@as(u64, 4), striped);
    }
}

test "qualification_17c70e37: the settings must agree on stripes: an expert without them is refused (SettingsDiffer), and a node refuses 1 or 17 stripes" {
    var h: Harness = .{ .io = std.testing.io };
    const cfg = Harness.striped(Harness.config(.{}, 10 * std.time.ns_per_s));
    const plain = Harness.config(.{}, 10 * std.time.ns_per_s);
    try std.testing.expectError(error.SettingsDiffer, h.start(cfg, .{ cfg, plain }, .{ no_faults, no_faults, no_faults }, .{ expertOpts(null, null), expertOpts(null, null) }));
    for (h.nodes[0].links) |l| if (l) |x| x.close();
    h.nodes[0].links = .{ null, null };
    h.server.close();
    for (h.threads) |t| if (t) |x| x.join();
    for (1..3) |i| h.nodes[i].deinit();
    h.nodes[0].deinit();
    h.finishTrace();
    h.bufs.deinit(h.engine(), gpa);
    for (h.regions) |r| gpa.free(r);
    for (h.status) |s| gpa.free(s);
    try std.testing.expectEqualStrings("SettingsDiffer", h.ends[1].why);
    var region: [64]u8 align(16) = undefined;
    var status: [proto.STATUS_WORDS]u32 = undefined;
    for ([_]u32{ 1, 17 }) |k| {
        var bad = plain;
        bad.settings.stripes = k;
        bad.settings.max_rows = 1;
        bad.settings.layout = .{ .d = 1, .s = 1 };
        const big = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast((proto.Region{ .slot = bad.settings.slotBytes() }).total()));
        defer gpa.free(big);
        try std.testing.expectError(error.Invalid, xchg.Node.init(gpa, std.testing.io, .attn, bad, .{ .region = big, .region_dev = 0, .status = &status, .status_dev = 0 }));
    }
    _ = &region;
}
