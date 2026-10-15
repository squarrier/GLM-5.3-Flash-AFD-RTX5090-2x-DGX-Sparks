//! G-loop (qualification_cec3c04c): the expert role's per-forward loop (glm afd_expert.forwards) over native RoCE exchange, three nodes in one
//! process as qualification_fe808d66's protocol.zig runs them (fake proxies copy at the ring, fn_xchg's kernels on CPU threads), with
//! tf-xchg's lab half as the stand-in for the experts' compute. The attention sends python 2.15's forwards: the first
//! request's header opens it, then one request a MoE layer, each layer's two replies taken before the next goes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const ae = glm.afd_expert;
const xchg = cuda.xchg;
const proto = xchg.proto;
const lab = @import("xchg/lab.zig");
const fake = @import("xchg/fake.zig");
const Cpu = @import("xchg/cpu.zig").Cpu;
const net = std.Io.net;

const gpa = std.testing.allocator;
pub const layout: proto.Layout = .{ .d = 64, .s = 9 };
pub const experts = 16;
pub const max_rows = 32;
pub const layers = 42; // python 2.15's MoE layers, 3..44
pub const ms = std.time.ns_per_ms;

/// The expert's device side on fn_xchg's CPU kernels: take into a host landing, the stand-in half (experts [lo, hi)
/// of each row's picks, fp32 in slot order), post routed[:R]; each computed layer's position and R kept (qualification_c2d4ee2e: and
/// its pair window's slot and fseq, its request's sections and its reply hashed).
pub const CpuExpert = struct {
    cpu: Cpu = .{},
    k: u32,
    landing: []align(16) u8 = &.{},
    routed: []align(16) u8 = &.{},
    seen: std.ArrayList([4]usize) = .empty, // (position, R, slot, fseq) of each compute
    sums: std.ArrayList([2]u64) = .empty, // Wyhash of the request past its header, and of the reply, each compute

    fn init(k: u32, rows: usize) !CpuExpert {
        const landing = try gpa.alignedAlloc(u8, .fromByteUnits(16), proto.wireBytes(layout.req(rows)));
        errdefer gpa.free(landing);
        return .{ .k = k, .landing = landing, .routed = try gpa.alignedAlloc(u8, .fromByteUnits(16), layout.rep(rows, false)) };
    }

    pub fn deinit(e: *CpuExpert) void {
        gpa.free(e.landing);
        gpa.free(e.routed);
        e.seen.deinit(gpa);
        e.sums.deinit(gpa);
    }

    pub fn take(e: *CpuExpert, n: *const xchg.Node, R: usize) !void {
        try e.cpu.take(n, &.{e.landing[0..proto.wireBytes(layout.req(R))]});
    }

    pub fn compute(e: *CpuExpert, pos: usize, f: ae.Forward) !void {
        const h = lab.half(experts, e.k);
        lab.expertRef(layout, e.landing, f.rows, h.lo, h.hi, false, e.routed);
        try e.seen.append(gpa, .{ pos, f.rows, f.slot, @intCast(f.fseq) });
        const req = e.landing[proto.HDR_BYTES..proto.wireBytes(layout.req(f.rows))];
        try e.sums.append(gpa, .{ std.hash.Wyhash.hash(0, req), std.hash.Wyhash.hash(0, e.routed[0..layout.rep(f.rows, false)]) });
    }

    pub fn post(e: *CpuExpert, n: *const xchg.Node, R: usize) !void {
        try e.cpu.post(n, e.routed[0..layout.rep(R, false)]);
    }

    pub fn sync(_: *CpuExpert, _: *const xchg.Node) !void {} // the CPU kernels ran in order on this thread
};

pub const Harness = struct {
    io: std.Io,
    regions: [3][]align(4096) u8 = undefined,
    status: [3][]u32 = undefined,
    nodes: [3]xchg.Node = undefined,
    openers: [3]fake.Opener = undefined,
    cpus: [3]Cpu = .{ .{}, .{}, .{} },
    server: cuda.link.Server = undefined,
    threads: [2]?std.Thread = .{ null, null },
    experts: [2]CpuExpert = undefined,
    ends: [2]ae.End = .{ .{ .lost = "never ran" }, .{ .lost = "never ran" } },
    stats: [2]ae.Stats = .{ .{}, .{} },
    posted: [2]u64 = .{ 0, 0 }, // replies each expert's proxy sent, when its loop ended
    down: [2]bool = .{ false, false }, // the expert's process is gone (its node closed)
    attn_down: bool = false,
    stopped: bool = false,
    bufs: lab.AttnBufs = undefined,
    fault: [2]?ae.Fault = .{ null, null }, // each expert's --fault
    silent_ns: u64 = 10 * std.time.ns_per_s, // the attention's watcher: a reply not landed this long fails it
    pairs: bool = false, // qualification_c2d4ee2e: the halves' TF_AFD_PAIRS (their settings and their loop's Opts)
    attn_pairs: ?bool = null, // qualification_c2d4ee2e: the attention's settings' pairs (null: the halves')
    rows: usize = max_rows, // qualification_c2d4ee2e: the wire's rows (the settings' max_rows; 2,048 for G-trace-pairs' chunks)
    inflight: u8 = 1, // qualification_c533375f: the halves' TF_AFD_INFLIGHT (their settings; their loop is the same)
    attn_inflight: ?u8 = null, // qualification_c533375f: the attention's settings' inflight (null: the halves')
    seq: bool = false, // qualification_969df694: the halves' TF_GLM_EXL3_SEQ (their settings' seq=1)
    attn_seq: ?bool = null, // qualification_969df694: the attention's settings' seq (null: the halves')
    lanes: u8 = 0, // qualification_9e49592e: the halves' TF_AFD_LANES (their settings' lanes=N, their loop's Opts)
    attn_lanes: ?u8 = null, // qualification_9e49592e: the attention's settings' lanes (null: the halves')

    fn config(give_up_ns: u64, rows: usize) xchg.Config {
        return .{ .settings = .{ .layout = layout, .max_rows = rows, .ports = 1 }, .give_up_ns = give_up_ns, .beat_ns = 50 * ms, .setup_timeout_ms = 10_000, .warn_failures = true };
    }

    /// Three nodes up: the experts join from their threads, then run python's loop; the attention attends here.
    pub fn start(h: *Harness, expert_give_up_ns: u64) !void {
        var cfg = config(20 * std.time.ns_per_s, h.rows);
        cfg.rules.silent_ns = h.silent_ns;
        cfg.settings.pairs = h.attn_pairs orelse h.pairs;
        cfg.settings.inflight = h.attn_inflight orelse h.inflight;
        cfg.settings.seq = h.attn_seq orelse h.seq;
        cfg.settings.lanes = h.attn_lanes orelse h.lanes;
        var ecfg = config(expert_give_up_ns, h.rows);
        ecfg.settings.pairs = h.pairs;
        ecfg.settings.inflight = h.inflight;
        ecfg.settings.seq = h.seq;
        ecfg.settings.lanes = h.lanes;
        const total = @max(cfg.settings.region().total(), ecfg.settings.region().total()); // qualification_9e49592e: N slots a source with lanes
        for (0..3) |i| {
            h.regions[i] = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(total));
            @memset(h.regions[i], 0);
            h.status[i] = try gpa.alloc(u32, proto.STATUS_WORDS);
            @memset(h.status[i], 0);
            h.openers[i] = .{ .gpa = gpa };
            const role: xchg.Role = if (i == 0) .attn else .{ .expert = @intCast(i - 1) };
            const mem: xchg.Memory = .{ .region = h.regions[i], .region_dev = @intFromPtr(h.regions[i].ptr), .status = h.status[i], .status_dev = @intFromPtr(h.status[i].ptr) };
            h.nodes[i] = try xchg.Node.init(gpa, h.io, role, if (i == 0) cfg else ecfg, mem);
        }
        for (0..2) |k| h.experts[k] = try CpuExpert.init(@intCast(k), h.rows);
        h.server = try cuda.link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
        const master = try net.IpAddress.parse("127.0.0.1", h.server.port());
        for (0..2) |k| h.threads[k] = try std.Thread.spawn(.{}, expertMain, .{ h, k, master });
        h.bufs = try lab.AttnBufs.init(h.engine(), gpa, layout, h.rows);
        try h.nodes[0].attend(h.server, h.openers[0].opener());
    }

    /// Expert k's process: join, python's forwards; an exit-3 end closes its node, as the process's exit would (a
    /// --fault exit's _exit too).
    fn expertMain(h: *Harness, k: usize, master: net.IpAddress) void {
        const n = &h.nodes[k + 1];
        n.join(master, h.openers[k + 1].opener()) catch |e| {
            h.ends[k] = .{ .err = e };
            return;
        };
        h.ends[k] = ae.forwards(&h.experts[k], n, .{ .layers = layers, .max_rows = h.rows, .fault = h.fault[k], .pairs = h.pairs, .lanes = h.lanes }, &h.stats[k]);
        h.posted[k] = n.proxies[0].?.ops();
        if (h.ends[k].code() == cuda.watch.lost_exit_code) {
            n.deinit();
            h.down[k] = true;
        }
    }

    pub fn attn(h: *Harness) *xchg.Node {
        return &h.nodes[0];
    }

    pub fn engine(h: *Harness) lab.Engine {
        return .{ .cpu = &h.cpus[0] };
    }

    /// One forward of R rows from the attention: python's header on every request (the first's opens the forward;
    /// `bad` replaces it), each layer's replies taken and checked against the stand-in. Stops after `upto` layers (the
    /// attention's process gone mid-forward) or at a failure. Returns the wrong values.
    pub fn forward(h: *Harness, fseq: i32, R: usize, bad: ?proto.Header, upto: usize) !u64 {
        const e = h.engine();
        const rb = layout.rep(R, false);
        const wire = proto.wireBytes(layout.req(R));
        var wrong: u64 = 0;
        for (0..upto) |pos| {
            const hd: proto.Header = if (pos == 0 and bad != null) bad.? else .{ .fseq = fseq, .rows = @intCast(R), .layer = @intCast(pos), .mode = if (R > 8) .chunk else .eager, .layers = layers };
            lab.fillRequest(layout, experts, hd, h.bufs.req.host);
            try e.post(h.attn(), h.bufs.req, wire);
            try e.take(h.attn(), &h.bufs.rep, &.{ rb, rb });
            if (h.attn().failure() != null) return wrong;
            for (0..2) |k| {
                const hh = lab.half(experts, @intCast(k));
                lab.expertRef(layout, h.bufs.req.host, R, hh.lo, hh.hi, false, h.bufs.ref);
                wrong += lab.wrongValues(h.bufs.rep[k].host[0..rb], h.bufs.ref[0..rb], 4);
            }
        }
        return wrong;
    }

    /// Requests the attention rang and each pair carried.
    pub const Counts = struct { rung: u32, requests: [2]u64 };

    pub fn counts(h: *Harness) Counts {
        return .{ .rung = h.attn().doorbell(), .requests = .{ h.attn().proxies[0].?.ops(), h.attn().proxies[1].?.ops() } };
    }

    /// The attention's clean end (unless it is gone already), then every node down; once.
    pub fn stop(h: *Harness) void {
        if (h.stopped) return;
        h.stopped = true;
        if (!h.attn_down) h.nodes[0].stopExperts();
        for (h.threads) |t| if (t) |x| x.join();
        h.threads = .{ null, null };
        if (!h.attn_down) h.nodes[0].deinit();
        h.attn_down = true;
        for (0..2) |k| if (!h.down[k]) h.nodes[k + 1].deinit();
        h.down = .{ true, true };
    }

    /// After `stop` (a test's defer): the memory the nodes used.
    pub fn free(h: *Harness) void {
        h.stop();
        for (&h.experts) |*e| e.deinit();
        h.bufs.deinit(h.engine(), gpa);
        h.server.close();
        for (h.regions) |r| gpa.free(r);
        for (h.status) |s| gpa.free(s);
    }

    /// The attention's process gone (no stop frame): its links and proxies close; then the experts' threads end.
    pub fn attnGone(h: *Harness) void {
        h.nodes[0].deinit();
        h.attn_down = true;
        for (h.threads) |t| if (t) |x| x.join();
        h.threads = .{ null, null };
    }
};

pub fn settle() void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * ms };
    _ = std.c.nanosleep(&ts, null);
}

const rows_list = [_]usize{ 1, 2, 3, 7, 8, 16, 32, 5 };

test "G-loop: one CPU turn and 42 replies a forward, R the first header's for every layer, counts exact, a stop exits 0" {
    var h: Harness = .{ .io = std.testing.io };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    var wrong: u64 = 0;
    for (rows_list, 1..) |R, f| wrong += try h.forward(@intCast(f), R, null, layers);
    settle();
    const c = h.counts();
    const failed = h.attn().failure();
    h.stop();
    const F = rows_list.len;
    try std.testing.expectEqual(@as(u64, 0), wrong);
    try std.testing.expectEqual(@as(?xchg.Failure, null), failed);
    try std.testing.expectEqual(@as(u32, F * layers), c.rung);
    try std.testing.expectEqual([2]u64{ F * layers, F * layers }, c.requests);
    try std.testing.expectEqual([2]u64{ F * layers, F * layers }, h.posted);
    var rows_sum: u64 = 0;
    for (rows_list) |r| rows_sum += layers * r;
    for (h.ends, h.stats, h.experts) |e, s, x| {
        try std.testing.expectEqual(ae.End.stopped, e);
        try std.testing.expectEqual(@as(u8, 0), e.code());
        try std.testing.expectEqual(ae.Stats{ .turns = F, .forwards = F, .layers = F * layers, .rows = rows_sum }, s);
        try std.testing.expectEqual(F * layers, x.seen.items.len);
        for (x.seen.items, 0..) |pr, i| { // forward i / 42's layer at position i % 42, with that forward's R
            try std.testing.expectEqual(i % layers, pr[0]);
            try std.testing.expectEqual(rows_list[i / layers], pr[1]);
        }
    }
}

test "G-loop: a bad first header is a breach: the expert exits 3 and posts nothing for it; the attention fails the request (dead)" {
    var h: Harness = .{ .io = std.testing.io };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const wrong = try h.forward(1, 4, null, layers);
    // forward 2 opens at layer 1: python next_forward's "no forward's first"
    _ = try h.forward(2, 4, .{ .fseq = 2, .rows = 4, .layer = 1, .mode = .eager, .layers = layers }, layers);
    settle();
    const failed = h.attn().failure();
    const c = h.counts();
    h.stop();
    try std.testing.expectEqual(@as(u64, 0), wrong);
    try std.testing.expectEqual(proto.Why.dead, failed.?.why);
    try std.testing.expectEqual([2]u64{ layers + 1, layers + 1 }, c.requests); // the bad request crossed to both
    try std.testing.expectEqual(@as(u32, layers + 1), c.rung);
    try std.testing.expectEqual([2]u64{ layers, layers }, h.posted); // and neither answered it
    for (h.ends, h.stats, h.experts) |e, s, x| {
        try std.testing.expectEqual(ae.End{ .breach = error.WrongLayer }, e);
        try std.testing.expectEqual(@as(u8, 3), e.code());
        try std.testing.expectEqual(@as(u64, 2), s.turns); // its CPU saw the second forward's header, then stopped
        try std.testing.expectEqual(@as(u64, 1), s.forwards);
        try std.testing.expectEqual(@as(usize, layers), x.seen.items.len); // nothing computed for it
    }
}

test "G-loop: the attention's EOF between forwards ends each expert in exit 3 at its next wait" {
    var h: Harness = .{ .io = std.testing.io };
    try h.start(20 * std.time.ns_per_s);
    defer h.free();
    const wrong = try h.forward(1, 8, null, layers);
    settle();
    h.attnGone();
    h.stop();
    try std.testing.expectEqual(@as(u64, 0), wrong);
    try std.testing.expectEqual([2]u64{ layers, layers }, h.posted);
    for (h.ends, h.stats) |e, s| {
        try std.testing.expect(e == .lost);
        try std.testing.expectEqual(@as(u8, 3), e.code());
        try std.testing.expectEqual(@as(u64, 1), s.turns);
        try std.testing.expectEqual(@as(u64, 1), s.forwards);
    }
}

test "G-loop: the attention gone mid-forward: each expert's next take gives up (300 ms), it exits 3, nothing is posted after" {
    var h: Harness = .{ .io = std.testing.io };
    try h.start(300 * ms);
    defer h.free();
    const wrong = try h.forward(1, 2, null, layers);
    _ = try h.forward(2, 3, null, 10); // layers 0-9 of forward 2, then the attention's process is gone
    settle();
    h.attnGone();
    h.stop();
    try std.testing.expectEqual(@as(u64, 0), wrong);
    try std.testing.expectEqual([2]u64{ layers + 10, layers + 10 }, h.posted); // forward 2's layers 0-9, nothing after
    for (h.ends, h.stats, h.experts) |e, s, x| {
        try std.testing.expect(e == .failed); // the take of forward 2's layer 10 gave up; every later one returned at once
        try std.testing.expectEqual(proto.Why.gave_up, e.failed.why);
        try std.testing.expectEqual(@as(u32, 1), e.failed.gave_up);
        try std.testing.expectEqual(@as(u8, 3), e.code());
        try std.testing.expectEqual(@as(u64, 2), s.turns);
        try std.testing.expectEqual(@as(u64, 1), s.forwards);
        try std.testing.expectEqual(2 * layers, x.seen.items.len); // the loop ran forward 2's 42 layers, posting none after 9
    }
}

test "G-loop: --fault exit:50 on expert 0: it answers 49 requests and dies once request 50 has landed; the attention fails on it (dead), the other expert exits 3" {
    var h: Harness = .{ .io = std.testing.io, .fault = .{ .{ .what = .exit, .at = 50 }, null } };
    try h.start(300 * ms);
    defer h.free();
    const wrong = try h.forward(1, 4, null, layers);
    _ = try h.forward(2, 4, null, layers); // its 8th request, seq 50, fails
    settle();
    const failed = h.attn().failure();
    h.attnGone(); // the attention exits 2 on it
    h.stop();
    try std.testing.expectEqual(@as(u64, 0), wrong);
    try std.testing.expectEqual(proto.Why.dead, failed.?.why);
    try std.testing.expectEqual([2]u32{ 0, 50 }, [2]u32{ failed.?.peer, failed.?.seq });
    try std.testing.expectEqual(ae.End.die, h.ends[0]);
    try std.testing.expectEqual(@as(u8, 3), h.ends[0].code());
    try std.testing.expectEqual(ae.Stats{ .turns = 2, .forwards = 1, .layers = 49, .rows = 4 * 49 }, h.stats[0]);
    try std.testing.expectEqual([2]u64{ 49, 50 }, h.posted); // nothing from expert 0 for request 50
    try std.testing.expectEqual(@as(u8, 3), h.ends[1].code());
}

test "G-loop: --fault silent:30 on expert 1: 29 replies, then none with its link and heartbeat up; the attention fails on it (silent, 300 ms); the expert exits 3 once the attention is gone" {
    var h: Harness = .{ .io = std.testing.io, .fault = .{ null, .{ .what = .silent, .at = 30 } }, .silent_ns = 300 * ms };
    try h.start(2 * std.time.ns_per_s); // the attention's silence is seen before expert 0's next take gives up
    defer h.free();
    const wrong = try h.forward(1, 2, null, layers); // fails at its 30th request
    const failed = h.attn().failure();
    settle();
    h.attnGone();
    h.stop();
    try std.testing.expectEqual(@as(u64, 0), wrong);
    try std.testing.expectEqual(proto.Why.silent, failed.?.why);
    try std.testing.expectEqual([2]u32{ 1, 30 }, [2]u32{ failed.?.peer, failed.?.seq });
    try std.testing.expect(h.ends[1] == .lost); // idle until the attention's EOF
    try std.testing.expectEqual(@as(u8, 3), h.ends[1].code());
    try std.testing.expectEqual(ae.Stats{ .turns = 1, .forwards = 0, .layers = 29, .rows = 2 * 29 }, h.stats[1]);
    try std.testing.expectEqual([2]u64{ 30, 29 }, h.posted);
    try std.testing.expectEqual(@as(u8, 3), h.ends[0].code());
}

test "check-weights' list equals qualification_2b1188e0's rank-h weights.json for the cut model's MoE layers 3-7, both halves: paths, shapes, dtypes, bytes, address tables, the width tables' digests (TF_GLM_EXL3_CHECKPOINT, TF_GLM_qualification_2b1188e0_RUN; else skipped)" {
    const dir = std.mem.span(std.c.getenv("TF_GLM_EXL3_CHECKPOINT") orelse return error.SkipZigTest);
    const run = std.mem.span(std.c.getenv("TF_GLM_qualification_2b1188e0_RUN") orelse return error.SkipZigTest);
    const cmd = @import("afd_expert.zig");
    var c = try cmd.config(gpa, dir);
    c.run = 8; // qualification_2b1188e0's: the backbone's first 8 layers (`--layers 8`)
    var ix = try glm.index.open(gpa, dir);
    defer ix.deinit();
    for ([_]u32{ 0, 1 }) |hh| {
        const p = try std.fmt.allocPrint(gpa, "{s}/rank{d}/weights.json", .{ run, hh });
        defer gpa.free(p);
        const text = try cmd.readFile(gpa, p);
        defer gpa.free(text);
        var r = try cmd.Ref.init(gpa, text);
        defer r.deinit();
        var x = try ae.half(gpa, &c, &ix.names, hh);
        defer x.deinit();
        try std.testing.expectEqualSlices(u32, &.{ 3, 4, 5, 6, 7 }, x.layers);
        var list: std.ArrayList(ae.Entry) = .empty;
        defer list.deinit(gpa);
        const tab = try gpa.alloc(u8, 36 * @as(usize, x.experts(0).count));
        defer gpa.free(tab);
        var t: cmd.Tally = .{};
        var widths: usize = 0;
        for (0..x.layers.len) |pos| {
            list.clearRetainingCapacity();
            try ae.entries(gpa, &x, pos, &list);
            const tb = x.tables(pos, 0, tab); // the width tables are bytes of the host's (the pointer tables addresses)
            for (list.items) |e| {
                const width = e.field == .gate_k2 or e.field == .up_k2 or e.field == .down_k2;
                const sum = if (width) try ae.digest(e, tb.bytes[@intCast(e.off - tb.off)..][0..@intCast(e.bytes)]) else null;
                widths += @intFromBool(width);
                t.add(&r, e, r.check(e, sum, width or ae.isTable(e)));
            }
        }
        t.finish(&r, x.layers);
        std.debug.print("qualification_2b1188e0 rank{d}: {d} equal, {d} differ, {d} not in it, {d} it has that we lack, {d} outside; {d} width tables by digest; first wrong {s}\n", .{ hh, t.equal, t.differ, t.extra, t.lacking, t.outside, widths, t.first[0..t.first_len] });
        try std.testing.expect(t.ok());
        try std.testing.expectEqual(@as(u64, 5 * 876), t.equal);
        try std.testing.expectEqual(@as(usize, 15), widths);
    }
}

test "the expert command compiles for the host: serve, the selftest and the check (they run on a GPU)" {
    _ = &@import("afd_expert.zig").command;
    _ = @import("afd_expert.zig");
}

test {
    _ = @import("afd_expert_pairs.zig"); // qualification_c2d4ee2e: the halves' prefill pairs over this harness
    _ = @import("afd_expert_seq.zig"); // qualification_969df694: SEQ's seq=1 in the settings at the handshake
    _ = @import("afd_expert_lanes.zig"); // qualification_9e49592e: the halves' prompt chunk lanes over this harness
}
