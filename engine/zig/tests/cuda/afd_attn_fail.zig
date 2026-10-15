//! G-fail (qualification_0fa2bee9, qualification_25364d42 PLAN §4): the AFD attention engine's steps over native RoCE exchange with a silent and a dead expert, three
//! nodes in one process as qualification_fe808d66's protocol.zig runs them (fake proxies copy at the ring, fn_xchg's kernels on CPU
//! threads), each expert running qualification_cec3c04c's per-forward loop (afd_expert.forwards) over the lab's stand-in half; the
//! attention's forward is recorded, its exchanges are real. Also the wire audit (TF_AFD_CHECK) end to end.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const xchg = cuda.xchg;
const proto = xchg.proto;
const ae = glm.afd_expert;
const ag = glm.afd_engine;
const afd = glm.afd_attn;
const audit = glm.afd_audit;
const lab = @import("xchg/trace_lab.zig");
const fake = @import("xchg/fake.zig");
const Cpu = @import("xchg/trace_cpu.zig").Cpu;
const net = std.Io.net;
const causal = @import("xchg/trace.zig");

const gpa = std.testing.allocator;
const fixture = @embedFile("forward_exl3_fixture");
const ms = std.time.ns_per_ms;
const experts_n = 288;
const max_rows = 64; // the wire's: the engine's prompt chunks
pub const layers = 2; // the cut model's MoE layers (3 and 4)
pub const no_fault: [2]?ae.Fault = .{ null, null };

/// The expert's device side on fn_xchg's CPU kernels (as qualification_cec3c04c's G-loop): take into a landing, the stand-in half
/// (experts [lo, hi) of each row's picks, fp32), post routed[:R]; each layer's audit record, by forward and position.
const CpuExpert = struct {
    cpu: Cpu = .{},
    k: u32,
    l: proto.Layout,
    landing: []align(16) u8,
    routed: []align(16) u8,
    at: [2]usize = .{ 0, 0 }, // the forward and position being computed
    records: [32][layers]audit.Layer = undefined,

    fn init(k: u32, l: proto.Layout) !CpuExpert {
        const landing = try gpa.alignedAlloc(u8, .fromByteUnits(16), proto.wireBytes(l.req(max_rows)));
        return .{ .k = k, .l = l, .landing = landing, .routed = try gpa.alignedAlloc(u8, .fromByteUnits(16), l.rep(max_rows, false)) };
    }

    fn deinit(e: *CpuExpert) void {
        gpa.free(e.landing);
        gpa.free(e.routed);
    }

    pub fn take(e: *CpuExpert, n: *const xchg.Node, R: usize) !void {
        const bytes = n.received(n.src(0), e.cpu.state[1]+%1, proto.HDR_BYTES);
        if (proto.Header.read(bytes[0..proto.HDR_BYTES])) |hdr| {
            e.cpu.observer.fseq = hdr.fseq; e.cpu.observer.layer = hdr.layer; e.cpu.observer.rows = hdr.rows;
        } else |_| e.cpu.observer.put(82, @splat(0)); // observer decode must not introduce a new error path
        e.cpu.observe(54, n);
        try e.cpu.take(n, &.{e.landing[0..proto.wireBytes(e.l.req(R))]});
        e.cpu.observe(55, n);
    }

    pub fn compute(e: *CpuExpert, pos: usize, f: ae.Forward) !void {
        e.cpu.observer.put(56, .{pos, @intCast(f.fseq), f.rows, 0, 0, 0, 0, 0});
        const h = lab.half(experts_n, e.k);
        lab.expertRef(e.l, e.landing, f.rows, h.lo, h.hi, false, e.routed);
        e.at = .{ @intCast(f.fseq), pos };
        e.cpu.observer.put(57, .{pos, @intCast(f.fseq), f.rows, 0, 0, 0, 0, 0});
    }

    pub fn post(e: *CpuExpert, n: *const xchg.Node, R: usize) !void {
        const rb = e.l.rep(R, false);
        if (e.at[0] < e.records.len) e.records[e.at[0]][e.at[1]] = audit.expertLayer(e.l, R, e.landing, e.routed[0..rb]);
        try e.cpu.post(n, e.routed[0..rb]);
    }

    pub fn sync(_: *CpuExpert, _: *const xchg.Node) !void {}
};

/// The attention's exchange on fn_xchg's CPU kernels: python's request for the provider's header (the lab's rows,
/// picks and weights), both replies taken and checked against the stand-in; the audit's ring filled as a GPU's copies.
/// qualification_c533375f: a request kept per slot (the posts' order, `reqs[posts % LANES_MAX]`), so a reply is checked against its own
/// request with two in flight (qualification_9e49592e: up to four).
const AttnHost = struct {
    cpu: Cpu = .{},
    l: proto.Layout,
    a: *afd.Afd,
    reqs: [proto.LANES_MAX][]align(16) u8,
    rep: [2][]align(16) u8,
    ref: []align(16) u8,
    posts: usize = 0,
    takes: usize = 0,
    /// qualification_c533375f: the most requests in flight at once (posted, not yet taken)
    most: usize = 0,
    wrong: u64 = 0,
    /// flips a byte of expert 1's reply as the attention keeps it, at this forward and position (the audit's check)
    corrupt: ?[2]usize = null,
    /// qualification_4be15bbf's G-trace: every request the attention posted, in order (the exchange messages, digested)
    sent: std.crypto.hash.sha2.Sha256 = .init(.{}),
    /// qualification_0f991257's G-trace-pairs: every message by window and position (a request past its header, each half's reply)
    msgs: ?*std.ArrayList(Msg) = null,

    fn host(h: *AttnHost) afd.Host {
        return .{ .ctx = h, .post = post, .take = take };
    }

    fn post(ctx: *anyopaque, node: *xchg.Node, hdr: *const [proto.HDR_BYTES]u8, pos: usize, R: usize) anyerror!void {
        const h: *AttnHost = @ptrCast(@alignCast(ctx));
        const req = h.reqs[h.posts % proto.LANES_MAX];
        h.posts += 1;
        h.most = @max(h.most, h.posts - h.takes);
        const ph = try proto.Header.read(hdr); // the original generator's one validation, at the original counter edge
        h.cpu.observer.fseq = ph.fseq; h.cpu.observer.layer = ph.layer; h.cpu.observer.rows = ph.rows;
        h.cpu.observer.seq = h.cpu.state[0]+%1; h.cpu.observer.slot = node.reg.slotOf(h.cpu.observer.seq);
        h.cpu.observe(16, node);
        lab.fillRequest(h.l, experts_n, ph, req);
        @memcpy(req[0..proto.HDR_BYTES], hdr);
        const wire = proto.wireBytes(h.l.req(R));
        h.sent.update(req[0..wire]);
        if (h.msgs) |m| try m.append(gpa, .{ .fseq = h.a.seq, .pos = pos, .what = 0, .sha = Msg.of(req[proto.HDR_BYTES..wire]) });
        try h.cpu.post(node, req[0..wire]);
        h.cpu.observe(17, node);
        if (h.a.ring) |*r| if (R <= r.rows) @memcpy(r.at(pos)[0][0..wire], req[0..wire]);
    }

    fn take(ctx: *anyopaque, node: *xchg.Node, pos: usize, R: usize) anyerror!void {
        const h: *AttnHost = @ptrCast(@alignCast(ctx));
        const req = h.reqs[h.takes % proto.LANES_MAX];
        if (proto.Header.read(req[0..proto.HDR_BYTES])) |ph| {
            h.cpu.observer.fseq = ph.fseq; h.cpu.observer.layer = ph.layer; h.cpu.observer.rows = ph.rows;
        } else |_| h.cpu.observer.put(82, @splat(0));
        h.takes += 1;
        const rb = h.l.rep(R, false);
        h.cpu.observe(18, node);
        try h.cpu.take(node, &.{ h.rep[0][0..rb], h.rep[1][0..rb] });
        h.cpu.observe(19, node);
        if (node.failure() != null) return;
        for (0..2) |k| {
            const hh = lab.half(experts_n, @intCast(k));
            lab.expertRef(h.l, req, R, hh.lo, hh.hi, false, h.ref);
            h.wrong += lab.wrongValues(h.rep[k][0..rb], h.ref[0..rb], 4);
            if (h.msgs) |m| try m.append(gpa, .{ .fseq = h.a.seq, .pos = pos, .what = @intCast(1 + k), .sha = Msg.of(h.rep[k][0..rb]) });
        }
        if (h.corrupt) |c| if (c[0] == h.a.seq and c[1] == pos) {
            h.rep[1][5] ^= 1;
        };
        if (h.a.ring) |*r| if (R <= r.rows) {
            const s = r.at(pos);
            @memcpy(s[1][0..rb], h.rep[0][0..rb]);
            @memcpy(s[2][0..rb], h.rep[1][0..rb]);
        };
    }
};

/// qualification_0f991257: one exchange message of a window at a position: its request past the header (what 0) or half k's reply
/// (what 1 + k), digested.
pub const Msg = struct {
    fseq: u32,
    pos: usize,
    what: u8,
    sha: [32]u8,

    fn of(bytes: []const u8) [32]u8 {
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
        return d;
    }

    pub fn less(_: void, a: Msg, b: Msg) bool {
        if (a.fseq != b.fseq) return a.fseq < b.fseq;
        if (a.pos != b.pos) return a.pos < b.pos;
        return a.what < b.what;
    }
};

pub const Harness = struct { // pub: qualification_4be15bbf's served gates (afd_serve_fail.zig) run on it
    io: std.Io,
    regions: [3][]align(4096) u8 = undefined,
    status: [3][]u32 = undefined,
    nodes: [3]xchg.Node = undefined,
    openers: [3]fake.Opener = undefined,
    server: cuda.link.Server = undefined,
    threads: [2]?std.Thread = .{ null, null },
    experts: [2]CpuExpert = undefined,
    ends: [2]ae.End = .{ .{ .lost = "never ran" }, .{ .lost = "never ran" } },
    stats: [2]ae.Stats = .{ .{}, .{} },
    posted: [2]u64 = .{ 0, 0 },
    down: [2]bool = .{ false, false },
    fault: [2]?ae.Fault,
    stopped: bool = false,
    // the attention: the cut model recorded, B3's Engine, the provider, the engine
    parsed: std.json.Parsed(std.json.Value) = undefined,
    m: glm.model.Model = undefined,
    w: glm.views.Weights = undefined,
    e: *glm.engine.Engine = undefined,
    a: afd.Afd = undefined,
    space: glm.trace.Space = undefined,
    t: glm.trace.Trace = undefined,
    ah: AttnHost = undefined,
    ring: []u8 = &.{},
    at: ag.Attn = undefined,
    win: audit.Window = .{},
    rec: [layers]audit.Layer = undefined,
    /// G-wire: the attention node on the GPU path's config (Afd.nodeConfig) instead of this harness's own
    gpu_cfg: bool = false,
    /// qualification_0f991257: TF_AFD_PAIRS=1 on all three nodes: the settings carry pairs, the halves serve them (afd_expert.Opts.pairs,
    /// qualification_c2d4ee2e's: where the tree has it), the attention pairs a prompt's chunks (B's copies placed last, B's audit ring)
    pairs: bool = false,
    pp: glm.afd_pairs.Pairs = undefined,
    pring: []u8 = &.{},
    /// qualification_c533375f: TF_AFD_INFLIGHT on all three nodes (2 with `pairs`): the settings carry inflight=2, a pair keeps two
    /// requests in flight
    inflight: u8 = 1,
    /// qualification_17c70e37: TF_AFD_STRIPES on all three nodes (the attention's through Afd.nodeConfig), in 256-B stripes: this wire's
    /// messages are a few KiB
    stripes: u32 = 0,
    /// qualification_9e49592e: TF_AFD_LANES=N on all three nodes (with `pairs`): the settings carry inflight=N lanes=N (N slots a source),
    /// the halves serve lane groups, the attention runs a prompt's chunks in lanes (lanes 1..N-1's copies placed last,
    /// their audit rings)
    lanes: u8 = 0,
    ln: glm.afd_lanes.Lanes = undefined,
    lring: []u8 = &.{},
    recorder: ?*causal.Recorder = null,
    monitor: ?@import("xchg/trace_node.zig").Monitor = null,
    identity: u32 = 0,

    fn config(l: proto.Layout, silent_ns: u64, give_up_ns: u64, pairs: bool, inflight: u8, lanes: u8) xchg.Config {
        return .{ .settings = .{ .layout = l, .max_rows = max_rows, .ports = 1, .pairs = pairs, .inflight = inflight, .lanes = lanes }, .rules = .{ .silent_ns = silent_ns }, .give_up_ns = give_up_ns, .beat_ns = 50 * ms, .setup_timeout_ms = 10_000, .warn_failures = true };
    }

    /// qualification_17c70e37: `c` in the harness's 256-B stripes when `own` (the attention's on gpu_cfg come from Afd.nodeConfig)
    fn striped(h: *const Harness, c: xchg.Config, own: bool) xchg.Config {
        var s = c;
        if (own and h.stripes >= 2) {
            s.settings.stripes = h.stripes;
            s.settings.stripe_min = 256;
        }
        return s;
    }

    /// Expert k's loop options: the fault, and qualification_c2d4ee2e's pairs where the tree's halves have them (qualification_9e49592e: the lanes).
    fn opts(h: *const Harness, k: usize) ae.Opts {
        var o: ae.Opts = .{ .layers = layers, .max_rows = max_rows, .fault = h.fault[k] };
        if (comptime @hasField(ae.Opts, "pairs")) o.pairs = h.pairs;
        if (comptime @hasField(ae.Opts, "lanes")) o.lanes = h.lanes;
        return o;
    }

    pub fn start(h: *Harness) !void {
        h.recorder = try causal.Recorder.create(gpa, @intFromPtr(h), h.identity);
        h.parsed = try std.json.parseFromSlice(std.json.Value, gpa, fixture, .{});
        h.m = try glm.model.Model.parse(gpa, h.parsed.value.object.get("config_json").?.string, 1, 0);
        try h.m.cut(5);
        h.w = try glm.views.rank(gpa, &h.m);
        h.e = try glm.engine.Engine.create(gpa, &h.m, &h.w, 2560, 8, max_rows, false);
        h.a = try afd.Afd.init(&h.m, max_rows);
        h.space = .{ .gpa = gpa };
        try glm.views.walk(&h.w, glm.views.Path.init("w"), &h.space, glm.trace.placeAll);
        try h.e.walk(&h.space, glm.trace.placeAll);
        try h.e.walkTmp(&h.space, glm.trace.placeAll);
        h.e.tmp.space = &h.space;
        try h.space.place("b.afd_recv", &h.a.afd_recv);
        h.t = glm.trace.Trace.init(gpa);
        const l = h.a.layout;
        var ring = afd.Ring.of(l, max_rows, null);
        h.ring = try gpa.alloc(u8, ring.total(layers));
        ring.bytes = h.ring;
        h.a.ring = ring;
        const rb = l.rep(max_rows, false);
        h.ah = .{ .l = l, .a = &h.a, .reqs = undefined, .rep = .{ try gpa.alignedAlloc(u8, .fromByteUnits(16), rb), try gpa.alignedAlloc(u8, .fromByteUnits(16), rb) }, .ref = try gpa.alignedAlloc(u8, .fromByteUnits(16), rb) };
        for (&h.ah.reqs) |*r| r.* = try gpa.alignedAlloc(u8, .fromByteUnits(16), proto.wireBytes(l.req(max_rows)));
        // three nodes: the experts join from their threads and run python's loop; the attention attends here (G-wire:
        // with the config tf-glm-afd attn builds on a GPU, Afd.nodeConfig from its flags)
        h.a.paired = h.pairs;
        h.a.inflight = if (h.lanes > 0) h.lanes else h.inflight; // qualification_9e49592e: N lanes keep N in flight
        h.a.lanes = h.lanes;
        h.a.stripes = if (h.stripes >= 2) .{ .k = h.stripes, .min = 256 } else .{}; // qualification_17c70e37: nodeConfig carries it (gpu_cfg)
        const cfg = h.striped(if (h.gpu_cfg) blk: {
            var c = h.a.nodeConfig(max_rows, 1, 0.3, 5, true); // --max-rows, one --dev, --silent-s 0.3 --give-up-s 5
            c.setup_timeout_ms = 10_000;
            c.warn_failures = true;
            break :blk c;
        } else config(l, 300 * ms, 5 * std.time.ns_per_s, h.pairs, h.a.inflight, h.lanes), !h.gpu_cfg);
        const total = cfg.settings.region().total(); // qualification_9e49592e: N slots a source with lanes
        for (0..3) |i| {
            h.regions[i] = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(total));
            @memset(h.regions[i], 0);
            h.status[i] = try gpa.alloc(u32, proto.STATUS_WORDS);
            @memset(h.status[i], 0);
            h.openers[i] = .{ .gpa = gpa, .recorder = h.recorder, .owner = @intCast(i) };
            const role: xchg.Role = if (i == 0) .attn else .{ .expert = @intCast(i - 1) };
            const mem: xchg.Memory = .{ .region = h.regions[i], .region_dev = @intFromPtr(h.regions[i].ptr), .status = h.status[i], .status_dev = @intFromPtr(h.status[i].ptr) };
            h.nodes[i] = try xchg.Node.init(gpa, h.io, role, if (i == 0) cfg else h.striped(config(l, 300 * ms, 2 * std.time.ns_per_s, h.pairs, h.a.inflight, h.lanes), true), mem);
        }
        h.ah.cpu.observer = h.recorder.?.writer(0);
        for (0..2) |k| {
            h.experts[k] = try CpuExpert.init(@intCast(k), l);
            h.experts[k].cpu.observer = h.recorder.?.writer(@intCast(k+1));
        }
        h.server = try cuda.link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
        const master = try net.IpAddress.parse("127.0.0.1", h.server.port());
        for (0..2) |k| h.threads[k] = try std.Thread.spawn(.{}, expertMain, .{ h, k, master });
        try h.nodes[0].attend(h.server, h.openers[0].opener());
        h.monitor = .{ .node = h.attn(), .writer = h.recorder.?.writer(7) };
        try h.monitor.?.start();
        h.a.wire = .{ .node = &h.nodes[0], .host = h.ah.host() };
        h.at = try ag.Attn.init(h.e, &h.a);
        h.at.check = .{ .ctx = h, .forward = checked, .layers = &h.rec };
        if (h.pairs and h.lanes == 0) { // qualification_0f991257: attend passed with pairs on all three nodes
            h.pp = glm.afd_pairs.Pairs.init(&h.at);
            try h.pp.walk(&h.space, glm.trace.placeAll);
            var pr = afd.Ring.of(l, max_rows, null);
            h.pring = try gpa.alloc(u8, pr.total(layers));
            pr.bytes = h.pring;
            h.pp.ring = pr;
            h.pp.agreed = true;
            h.at.pairs = &h.pp;
        }
        if (h.lanes > 0) { // qualification_9e49592e: attend passed with lanes=N on all three nodes
            h.ln = try glm.afd_lanes.Lanes.init(&h.at);
            try h.ln.walk(&h.space, glm.trace.placeAll);
            var lr = afd.Ring.of(l, max_rows, null);
            const per = lr.total(layers);
            h.lring = try gpa.alloc(u8, (h.lanes - 1) * per);
            for (h.ln.rings[0 .. h.lanes - 1], 0..) |*rg, k| {
                lr.bytes = h.lring[k * per ..][0..per];
                rg.* = lr;
            }
            h.ln.agreed = true;
            h.at.lanes = &h.ln;
        }
    }

    /// The audit's attention half against each expert's records of the same forward, as python's _finish_check.
    fn checked(ctx: *anyopaque, seq: u32, R: usize, mine: []const audit.Layer) anyerror!void {
        const h: *Harness = @ptrCast(@alignCast(ctx));
        h.win.forward(R, mine, .{ &h.experts[0].records[seq], &h.experts[1].records[seq] });
    }

    /// Expert k's process: join, qualification_cec3c04c's forwards; an exit-3 end closes its node, as the process's exit would.
    fn expertMain(h: *Harness, k: usize, master: net.IpAddress) void {
        const n = &h.nodes[k + 1];
        n.join(master, h.openers[k + 1].opener()) catch |e| {
            h.ends[k] = .{ .err = e };
            return;
        };
        h.experts[k].cpu.observe(50, n);
        h.ends[k] = ae.forwards(&h.experts[k], n, h.opts(k), &h.stats[k]);
        h.experts[k].cpu.observe(51, n);
        h.posted[k] = n.proxies[0].?.ops();
        if (h.ends[k].code() == cuda.watch.lost_exit_code) {
            n.deinit();
            h.down[k] = true;
        }
    }

    pub fn attn(h: *Harness) *xchg.Node {
        return &h.nodes[0];
    }

    /// A recorded Fwd for the next step (its trace emptied).
    pub fn fwd(h: *Harness) glm.fwd.Fwd {
        h.t.reset();
        return glm.fwd.Fwd.recording(&h.m, &h.t);
    }

    /// The attention's doorbell and the requests each pair carried.
    pub fn counts(h: *Harness) struct { rung: u32, requests: [2]u64 } {
        return .{ .rung = h.attn().doorbell(), .requests = .{ h.attn().proxies[0].?.ops(), h.attn().proxies[1].?.ops() } };
    }

    /// The attention's clean end, then every node down (an expert that exited is down already); once.
    pub fn stop(h: *Harness) void {
        if (h.stopped) return;
        h.stopped = true;
        h.ah.cpu.observe(60, h.attn());
        h.nodes[0].stopExperts();
        for (h.threads) |t| if (t) |x| x.join();
        h.ah.cpu.observe(61, h.attn());
        if (h.monitor) |*m| m.join();
        h.nodes[0].deinit();
        for (0..2) |k| if (!h.down[k]) h.nodes[k + 1].deinit();
        if (h.recorder) |r| { r.joined = true; r.dump(); gpa.destroy(r); h.recorder = null; }
    }

    pub fn free(h: *Harness) void {
        h.stop();
        for (&h.experts) |*e| e.deinit();
        h.server.close();
        for (h.regions) |r| gpa.free(r);
        for (h.status) |s| gpa.free(s);
        for (h.ah.reqs) |r| gpa.free(r);
        for (h.ah.rep) |r| gpa.free(r);
        gpa.free(h.ah.ref);
        gpa.free(h.ring);
        gpa.free(h.pring);
        gpa.free(h.lring);
        h.t.deinit();
        h.space.deinit();
        h.e.destroy(gpa);
        h.w.deinit(gpa);
        h.parsed.deinit();
    }
};

pub fn settle() void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * ms };
    _ = std.c.nanosleep(&ts, null);
}

fn has(t: *const glm.trace.Trace, name: []const u8) bool {
    for (t.events.items) |e| if (std.mem.eql(u8, e.name, name)) return true;
    return false;
}

/// The failing step: steps until one fails (at most `most`); how many succeeded, its time, and its trace's sample.
fn untilFailure(h: *Harness, most: usize) !struct { ok: usize, ns: u64, sampled: bool } {
    var tok: u32 = 11;
    for (0..most) |i| {
        var f = h.fwd();
        const t0 = h.attn().nowNs();
        tok = h.at.step(&f, tok) catch |err| {
            try f.flush();
            if (err != error.ExpertFailed) return err;
            return .{ .ok = i, .ns = h.attn().nowNs() - t0, .sampled = has(&h.t, "torch.topk") or has(&h.t, "torch.fill_") };
        };
    }
    return error.NoFailure;
}

/// After the failure: `n` more steps and a prompt, each refused at once with nothing launched or sent.
fn refused(h: *Harness, n: usize) !void {
    const before = h.counts();
    for (0..n + 1) |i| {
        var f = h.fwd();
        const t0 = h.attn().nowNs();
        const r = if (i < n) h.at.step(&f, 7) else h.at.prefill(&f, &.{ 1, 2, 3 });
        try std.testing.expectError(error.ExpertFailed, r);
        try f.flush();
        try std.testing.expectEqual(@as(usize, 0), h.t.events.items.len);
        try std.testing.expect(h.attn().nowNs() - t0 < 50 * ms);
    }
    try std.testing.expectEqual(before, h.counts());
}

test "G-fail: a healthy run: a 40-row prompt chunk and 6 decode steps, every reply exact, the audit clean; a corrupted reply is the audit's one mismatch" {
    var h: Harness = .{ .io = std.testing.io, .fault = no_fault };
    try h.start();
    defer h.free();
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(100 + i);
    var f = h.fwd();
    var tok = try h.at.prefill(&f, &prompt);
    for (0..6) |i| {
        if (i == 3) h.ah.corrupt = .{ 5, 1 }; // forward 5 (the 4th step): the attention's copy of expert 1's layer-1 reply
        f = h.fwd();
        tok = try h.at.step(&f, tok);
    }
    settle();
    const c = h.counts();
    const failed = h.attn().failure();
    h.stop();
    try std.testing.expectEqual(@as(?xchg.Failure, null), failed);
    try std.testing.expectEqual(@as(u64, 0), h.ah.wrong);
    try std.testing.expectEqual(@as(u32, 7 * layers), c.rung);
    try std.testing.expectEqual([2]u64{ 7 * layers, 7 * layers }, c.requests);
    try std.testing.expectEqual([2]u64{ 7 * layers, 7 * layers }, h.posted);
    for (h.ends, h.stats) |e, s| {
        try std.testing.expectEqual(ae.End.stopped, e);
        try std.testing.expectEqual(ae.Stats{ .turns = 7, .forwards = 7, .layers = 7 * layers, .rows = (40 + 6) * layers }, s);
    }
    try std.testing.expectEqual(@as(usize, 46), h.e.st.pos);
    var buf: [256]u8 = undefined;
    const line = h.win.line(&buf);
    std.debug.print("G-fail healthy: {s}\n", .{line});
    try std.testing.expect(std.mem.startsWith(u8, line, "afd-check: window complete, 7 forwards, 1 mismatches total, rows checked [1, 40]; first mismatch: layer 1 peer 1 out: "));
}

test "G-wire (qualification_cd073cc0): the GPU path's exchange config (Afd.nodeConfig from tf-glm-afd attn's flags) attends both experts over the fake proxy; a 40-row chunk and 3 steps with the audit, every reply exact, the window clean" {
    var h: Harness = .{ .io = std.testing.io, .fault = no_fault, .gpu_cfg = true };
    try h.start();
    defer h.free();
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(200 + i);
    var f = h.fwd();
    var tok = try h.at.prefill(&f, &prompt);
    for (0..3) |_| {
        f = h.fwd();
        tok = try h.at.step(&f, tok);
    }
    settle();
    const c = h.counts();
    const failed = h.attn().failure();
    var tb: [256]u8 = undefined;
    const settings = try h.attn().cfg.settings.text(&tb);
    std.debug.print("G-wire: {s}; watcher silent {d} ms, give-up {d} ms\n", .{ settings, h.attn().cfg.rules.silent_ns / ms, h.attn().cfg.give_up_ns / ms });
    h.stop();
    try std.testing.expectEqual(@as(?xchg.Failure, null), failed);
    try std.testing.expectEqual(@as(u64, 0), h.ah.wrong);
    try std.testing.expectEqual(@as(u32, 4 * layers), c.rung);
    try std.testing.expectEqual([2]u64{ 4 * layers, 4 * layers }, h.posted);
    for (h.ends) |e| try std.testing.expectEqual(ae.End.stopped, e);
    var buf: [256]u8 = undefined;
    const line = h.win.line(&buf);
    std.debug.print("G-wire: {s}\n", .{line});
    try std.testing.expect(std.mem.startsWith(u8, line, "afd-check: window complete, 4 forwards, 0 mismatches"));
}

test "qualification_17c70e37: G-wire in stripes (TF_AFD_STRIPES=8/256 through Afd.nodeConfig): a 40-row chunk and 3 steps with the audit, every reply exact, the window clean, the counts G-wire's; the chunk's messages taken stripe by stripe" {
    var h: Harness = .{ .io = std.testing.io, .fault = no_fault, .gpu_cfg = true, .stripes = 8 };
    try h.start();
    defer h.free();
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(200 + i);
    var f = h.fwd();
    var tok = try h.at.prefill(&f, &prompt);
    for (0..3) |_| {
        f = h.fwd();
        tok = try h.at.step(&f, tok);
    }
    settle();
    const c = h.counts();
    const failed = h.attn().failure();
    var tb: [256]u8 = undefined;
    const settings = try h.attn().cfg.settings.text(&tb);
    std.debug.print("G-wire stripes: {s}; takes in stripes: attention {d}, experts {d} {d}\n", .{ settings, h.ah.cpu.striped, h.experts[0].cpu.striped, h.experts[1].cpu.striped });
    const st = [3]u64{ h.ah.cpu.striped, h.experts[0].cpu.striped, h.experts[1].cpu.striped };
    h.stop();
    try std.testing.expectEqual([2]u64{ 8, 256 }, [2]u64{ h.attn().cfg.settings.stripes, h.attn().cfg.settings.stripe_min }); // from Afd.nodeConfig
    try std.testing.expectEqual(@as(?xchg.Failure, null), failed);
    try std.testing.expectEqual(@as(u64, 0), h.ah.wrong);
    try std.testing.expectEqual(@as(u32, 4 * layers), c.rung);
    try std.testing.expectEqual([2]u64{ 4 * layers, 4 * layers }, h.posted);
    for (h.ends) |e| try std.testing.expectEqual(ae.End.stopped, e);
    // a take goes in stripes when its messages reach 8 stripes of 256 B: the chunk's at 40 rows, a step's at 1 row if as large
    const l = h.a.layout;
    const in = struct {
        fn s(bytes: usize) u64 {
            return @intFromBool(proto.stripesOf(8, 256, bytes) > 1);
        }
    };
    const want_attn = layers * (in.s(l.rep(40, false)) + 3 * in.s(l.rep(1, false)));
    const want_expert = layers * (in.s(proto.wireBytes(l.req(40))) + 3 * in.s(proto.wireBytes(l.req(1))));
    try std.testing.expectEqual([3]u64{ want_attn, want_expert, want_expert }, st);
    try std.testing.expect(st[0] >= layers and st[1] >= layers); // the chunk's two layers at least
    var buf: [256]u8 = undefined;
    const line = h.win.line(&buf);
    std.debug.print("G-wire stripes: {s}\n", .{line});
    try std.testing.expect(std.mem.startsWith(u8, line, "afd-check: window complete, 4 forwards, 0 mismatches"));
}

test "G-fail: a silent expert (expert 1 from request 5) fails its step at the watcher's 300 ms, before its sample and commit; nothing is sent after; later steps refuse before any launch; the live expert's count is exact" {
    var h: Harness = .{ .io = std.testing.io, .fault = .{ null, .{ .what = .silent, .at = 5 } } };
    try h.start();
    defer h.free();
    const r = try untilFailure(&h, 10);
    const failed = h.at.failure().?;
    try refused(&h, 3);
    settle();
    const c = h.counts();
    var buf: [128]u8 = undefined;
    std.debug.print("G-fail silent: {s} after {d} steps, {d} ms\n", .{ ag.describe(failed, &buf), r.ok, r.ns / ms });
    h.stop();
    try std.testing.expectEqual(@as(usize, 2), r.ok); // requests 1-4: steps 1 and 2
    try std.testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 5, .gave_up = 0, .breach = 0 }, failed);
    try std.testing.expect(r.ns >= 300 * ms and r.ns < 2000 * ms);
    try std.testing.expect(!r.sampled);
    try std.testing.expectEqual(@as(usize, 2), h.e.st.pos); // the failed step committed nothing
    try std.testing.expectEqual(@as(u64, 0), h.ah.wrong);
    try std.testing.expectEqual(@as(u32, 5), c.rung); // request 6 (its second layer) was not sent
    try std.testing.expectEqual([2]u64{ 5, 5 }, c.requests);
    try std.testing.expectEqual([2]u64{ 5, 4 }, h.posted); // expert 0 answered all 5 it got, expert 1 the first 4
    try std.testing.expect(h.ends[0] == .failed and h.ends[0].failed.why == .gave_up); // request 6 never came: its take gave up
    try std.testing.expectEqual(ae.End.stopped, h.ends[1]); // silent until the attention's stop frame
}

test "G-fail: a dead expert (expert 0 exits at request 3) fails its step at once; nothing is sent after; later steps refuse before any launch; the live expert saw exactly 3 requests" {
    var h: Harness = .{ .io = std.testing.io, .fault = .{ .{ .what = .exit, .at = 3 }, null } };
    try h.start();
    defer h.free();
    const r = try untilFailure(&h, 10);
    const failed = h.at.failure().?;
    try refused(&h, 3);
    settle();
    const c = h.counts();
    var buf: [128]u8 = undefined;
    std.debug.print("G-fail dead: {s} after {d} steps, {d} ms\n", .{ ag.describe(failed, &buf), r.ok, r.ns / ms });
    h.stop();
    try std.testing.expectEqual(@as(usize, 1), r.ok);
    try std.testing.expectEqual(xchg.Failure{ .why = .dead, .peer = 0, .seq = 3, .gave_up = 0, .breach = 0 }, failed);
    try std.testing.expect(r.ns < 1000 * ms);
    try std.testing.expect(!r.sampled);
    try std.testing.expectEqual(@as(usize, 1), h.e.st.pos);
    try std.testing.expectEqual(@as(u32, 3), c.rung);
    try std.testing.expectEqual([2]u64{ 3, 3 }, c.requests);
    try std.testing.expectEqual(ae.End.die, h.ends[0]);
    try std.testing.expectEqual([2]u64{ 2, 3 }, h.posted); // expert 1 answered exactly requests 1-3
}

test {
    _ = @import("afd_serve_fail.zig"); // qualification_4be15bbf: G-trace and G-dead-half on the served stream, over this file's Harness
    _ = @import("afd_attn_pairs.zig"); // qualification_0f991257: the pair scheduler against the halves (G-trace-pairs' 3-node form, a dead half)
    _ = @import("afd_attn_lanes.zig"); // qualification_9e49592e: the lanes against the halves (G-trace-lanes' 3-node form, a dead half)
}
