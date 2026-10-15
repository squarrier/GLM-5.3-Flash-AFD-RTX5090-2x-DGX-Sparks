//! The served AFD attention stream's lane side (qualification_4be15bbf), for tf-glm-afd attn serve (afd_attn_serve.zig) and its CPU
//! gates: qualification_qualification_e91d6b91's lane backend (native/glm_cuda_lanes.zig: one stream, greedy, drafts off) over the served calls on
//! afd_engine.Attn (glm.afd_serve.Serve), each call with the CUDA context current on the lane host's thread; the
//! words a failed exchange fails a request with; the monitor that stops the server once the exchange failed (Ash on
//! #214: a failed expert fails the request, never a hang; the role then ends with tf-glm-afd's exit 2); and the RTX
//! 5090's forecast for a form at a served context.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const gl = @import("glm_lanes");

const xchg = cuda.xchg;
const Allocator = std.mem.Allocator;

/// The Engine's capacity over the served window (the CUDA GLM's rule, native/glm_cuda.zig: the window + 64).
pub const slack = 64;
/// The served window when --context is not given (the CUDA GLM's default; Scott serves ~262K, never 1M-class).
pub const default_context: i64 = 262_144;
/// GPU memory a form must leave free at its served context (qualification_4be15bbf's rule for the 5090).
pub const headroom: u64 = 2 << 30;
/// The RTX 5090's memory as nvidia-smi reports it on attention-host (qualification_e8c93577's host facts: 32,607 MiB).
pub const card_5090: u64 = 32_607 << 20;
/// The CUDA context, the Triton set and the modules on the device (qualification_cd073cc0's figure; qualification_e8c93577 measured within it).
pub const fixed: u64 = 1 << 30;

/// The served window: --context (0: the model's), else 262,144 within the model's, never past the model's (native/cuda.zig's contextWindow).
pub fn window(requested: ?i64, native: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default_context, native) else default_context;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}

/// config.json's max_position_embeddings (text_config's first), 0 when it names none.
pub fn nativeWindow(a: Allocator, config: []const u8) i64 {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, config, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

/// The positions a sequence may reach: the capacity with the DSA scratch on a long context (qualification_qualification_c3ccc258), else the dense limit.
pub fn reachOf(capacity: usize, dense_limit: usize) usize {
    const long = capacity > dense_limit and glm.prompt.DsaScratch != void;
    return if (long) capacity else @min(capacity, dense_limit);
}

/// What one form holds on the 5090 at a capacity: the packed weights, the Engine's device block (KV and index caches,
/// the prompt Buffers, the DSA scratch), KDA's wide scratch, the exchange's device block, and the fixed term; qualification_9e49592e:
/// with TF_AFD_LANES=N, lanes 1..N-1's copies (cuda_afd_lanes.Lanes.attach's block, reserved at the open).
pub const Forecast = struct {
    weights: u64,
    engine: u64,
    pbuf: u64,
    dsa: u64,
    scratch: u64,
    exchange: u64,
    fixed: u64 = fixed,
    lanes: u64 = 0,

    pub fn total(x: Forecast) u64 {
        return x.weights + x.engine + x.scratch + x.exchange + x.fixed + x.lanes;
    }

    /// Whether it leaves `headroom` of `card` free.
    pub fn fits(x: Forecast, card: u64) bool {
        return x.total() + headroom <= card;
    }

    pub fn line(x: Forecast, card: u64, buf: []u8) []const u8 {
        if (x.lanes > 0) return std.fmt.bufPrint(buf, "{d:.2} GiB: weights {d:.2}, the Engine {d:.2} (prompt Buffers {d:.2}, DSA scratch {d:.2}), KDA scratch {d:.2}, exchange {d:.2}, prompt lanes {d:.2}, context and modules {d:.2}; the card {d:.2} GiB leaves {d:.2} ({s} the 2 GiB headroom)", .{ gib(x.total()), gib(x.weights), gib(x.engine), gib(x.pbuf), gib(x.dsa), gib(x.scratch), gib(x.exchange), gib(x.lanes), gib(x.fixed), gib(card), gib(card -| x.total()), if (x.fits(card)) "within" else "PAST" }) catch "forecast";
        return std.fmt.bufPrint(buf, "{d:.2} GiB: weights {d:.2}, the Engine {d:.2} (prompt Buffers {d:.2}, DSA scratch {d:.2}), KDA scratch {d:.2}, exchange {d:.2}, context and modules {d:.2}; the card {d:.2} GiB leaves {d:.2} ({s} the 2 GiB headroom)", .{ gib(x.total()), gib(x.weights), gib(x.engine), gib(x.pbuf), gib(x.dsa), gib(x.scratch), gib(x.exchange), gib(x.fixed), gib(card), gib(card -| x.total()), if (x.fits(card)) "within" else "PAST" }) catch "forecast";
    }
};

pub fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

/// The forecast for the Engine `e` (its device block sized on a CPU), the provider `a`, `weights` packed bytes and
/// prompt chunks of `max_rows` rows (kda._wide_scratch's q fp32 and y bf16, as afd_attn_gpu allocates them); qualification_9e49592e:
/// `a.lanes` = N adds N - 1 lanes' copies (the pairs' B copies with two in flight each, Lanes.bytes; qualification_47b9bf3a: with the
/// provider's MTP position, each lane's MTP residual and position too).
pub fn forecast(e: *glm.engine.Engine, a: *const glm.afd_attn.Afd, weights: u64, max_rows: usize) !Forecast {
    const eb = try e.deviceBytes();
    var ex: u64 = 0;
    for ([_]glm.views.Buf{ a.afd_recv, a.w.req, a.w.hdr }) |b| ex += std.mem.alignForward(u64, b.bytes(), 256); // Afd.attach's block
    var prompt_lanes: u64 = 0;
    if (a.lanes > 0) {
        var c = glm.afd_lanes_mtp.copies(&e.pbuf, &e.st, a);
        var ar: glm.buffers.Arena = .{};
        try glm.views.walk(&c, glm.views.Path.init("lane"), &ar, glm.buffers.Arena.visit);
        prompt_lanes = (a.lanes - 1) * @as(u64, ar.used);
    }
    return .{ .weights = weights, .engine = eb.total, .pbuf = eb.pbuf, .dsa = eb.dsa, .scratch = max_rows * e.m.ll() * 128 * 6, .exchange = ex, .lanes = prompt_lanes };
}

/// The forecast at `capacity` for model `m` (world 1, its views unbound: sizes only), `weights` and `max_rows`; qualification_9e49592e:
/// `n_lanes` TF_AFD_LANES' N (0: off).
pub fn forecastAt(gpa: Allocator, m: *const glm.model.Model, weights: u64, capacity: usize, max_rows: usize, n_lanes: u8) !Forecast {
    var w = try glm.views.rank(gpa, m);
    defer w.deinit(gpa);
    const e = try glm.engine.Engine.create(gpa, m, &w, capacity, 8, max_rows, capacity > m.denseLimit());
    defer e.destroy(gpa);
    if (try glm.afd_sampling.fromEnv()) try e.useSampling(gpa);
    var a = try glm.afd_attn.Afd.initWith(m, max_rows, m.mtp_layers > 0); // qualification_4be15bbf lab: the MTP head with TF_AFD_MTP=1
    if (n_lanes > 0) {
        a.inflight = n_lanes;
        a.lanes = n_lanes;
    }
    return forecast(e, &a, weights, max_rows);
}

/// The largest window (a multiple of 64, at most 262,144) whose capacity fits `card` with the headroom; 0: none.
pub fn largestWindow(gpa: Allocator, m: *const glm.model.Model, weights: u64, max_rows: usize, card: u64, n_lanes: u8) !usize {
    var lo: usize = 0; // fits (0: nothing yet)
    var hi: usize = @as(usize, @intCast(default_context)) / 64 + 1; // in 64-row steps; past the cap
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if ((try forecastAt(gpa, m, weights, mid * 64 + slack, max_rows, n_lanes)).fits(card)) lo = mid else hi = mid;
    }
    return lo * 64;
}

/// The lane thread's CUDA context: the lane host steps rounds on its own thread, and CUDA binds per thread
/// (native/cuda.zig's Host.bind).
threadlocal var bound: ?*const cuda.Context = null;

/// The served calls with the context current on the calling thread (null on a CPU: recorded); `len` as the Serve's.
pub const Bound = struct {
    s: *glm.afd_serve.Serve,
    ctx: ?*const cuda.Context = null,
    len: usize = 0,
    /// prompt passes run (the warm-up's among them)
    prefills: u64 = 0,

    drafts: bool = false,
    sampled: bool = false,

    pub fn samplingEnabled(b: *const Bound) bool {
        return b.sampled;
    }

    pub fn configureSampling(b: *Bound, p: ?lanes.sampling.Sampling) void {
        if (!b.sampled) return;
        const enabled = p != null and p.?.temperature > 0;
        b.s.at.e.sampling = if (enabled) .{ .seed = p.?.seed, .temperature = p.?.temperature, .top_k = p.?.top_k, .top_p = p.?.top_p, .min_p = p.?.min_p } else null;
        b.s.at.e.stats = .{};
        b.s.at.absorb = b.drafts and !enabled;
    }

    fn bind(b: *Bound) !void {
        const c = b.ctx orelse return;
        if (bound == c) return;
        try c.makeCurrent();
        bound = c;
    }

    pub fn prefill(b: *Bound, ids: []const u32) !u32 {
        try b.bind();
        defer b.len = b.s.len;
        b.prefills += 1;
        return b.s.prefill(ids);
    }

    pub fn step(b: *Bound, token: u32) !u32 {
        try b.bind();
        defer b.len = b.s.len;
        return b.s.step(token);
    }

    pub fn installStructured(b: *Bound, endpoint: lanes.structured_control.Endpoint) !void {
        try b.bind();
        try b.s.installStructured(endpoint);
    }
    pub fn clearStructured(b: *Bound) void {
        b.s.clearStructured();
        b.len = b.s.len;
    }
    pub fn maskedWidth(b: *Bound) usize { return b.s.maskedWidth(); }
    pub fn maskedWords(b: *Bound) usize {
        return b.s.maskedWords();
    }
    pub fn checkMasked(b: *Bound) !void {
        try b.bind();
        try b.s.checkMasked();
    }
    pub fn prefillToLogits(b: *Bound, ids: []const u32) !void {
        try b.bind();
        defer b.len = b.s.len;
        b.prefills += 1;
        try b.s.prefillToLogits(ids);
    }
    pub fn pickFirst(b: *Bound, mask: []const u32) !u32 {
        try b.bind();
        defer b.len = b.s.len;
        return b.s.pickFirst(mask);
    }
    pub fn stepMasked(b: *Bound, token: u32, mask: []const u32) !u32 {
        try b.bind();
        defer b.len = b.s.len;
        return b.s.stepMasked(token, mask);
    }

    pub fn release(b: *Bound) void {
        b.s.release();
        b.len = b.s.len;
    }
};

test "grammar_tests_c5 C06 C09 actual Bound AFD committed compute clone provider and drain faults" {
    try @import("grammar_tests_c5_afd_fault.zig").run(Bound);
}

test "grammar_tests_c3 C06 C07 actual Bound AFD unknown sampler fault poisons and refuses same backend" {
    try @import("grammar_tests_c3_afd_fault.zig").run(Bound);
}

test "grammar_tests_c C08 C09 actual scheduler tapped small finish selection empty cancellation" {
    try @import("grammar_tests_c_finish.zig").run(Bound);
}
test "grammar_tests_c C08 actual cancelled Pairs rejects buffered post-frontier work" {
    try @import("grammar_tests_c_pairs.zig").cutoffWitness(Bound);
}
test "grammar_tests_c C09 configured Lanes eight real action-boundary cancellations" {
    try @import("grammar_tests_c_lanes.zig").boundaries(Bound);
}
test "grammar_tests_c C08 Pairs eight real action-boundary cancellations" {
    try @import("grammar_tests_c_pairs.zig").boundaries(Bound);
}
test "grammar_tests_c C09 configured Lanes2 remains refused at real settings and constructor" {
    try @import("grammar_tests_c_lanes.zig").refusal();
}
test "grammar_tests_c C09 actual Bound Serve Attn configured Lanes3 and4 drain and reuse" {
    try @import("grammar_tests_c_lanes.zig").run(Bound);
}

test "grammar_tests_c C08 actual Bound Serve Attn Pairs permitted requests drain and reuse" {
    try @import("grammar_tests_c_pairs.zig").run(Bound);
}

test "grammar_tests_c C06 C07 actual Bound Serve Attn three-role chunk cancellation and selection" {
    try @import("grammar_tests_c_afd_chunks.zig").run(Bound);
}

test "grammar_tests_c7 C10 C11 actual AFD Lanes.start LaneHost owned request slices and peer READY frame allocation" {
    try @import("grammar_tests_c7_afd_host.zig").run(Bound, Lanes, Words);
}

test "grammar_mask actual Bound Serve Attn masked first and step: forbidden winners removed before topk, staged first" {
    try @import("grammar_mask_test.zig").forbidden(Bound);
}
test "grammar_mask plain with the adapter installed and an all-allowed structured request equal plain, every event" {
    try @import("grammar_mask_test.zig").identity(Bound);
}
test "grammar_mask empty, fill-error and cancel cleanup with valid next structured and plain requests" {
    try @import("grammar_mask_test.zig").cleanup(Bound);
}
test "grammar_mask a device fault after staging poisons the adapter; plain requests unaffected" {
    try @import("grammar_mask_test.zig").poison(Bound);
}
test "grammar_mask a generated thinking marker activates the NEXT pick" {
    try @import("grammar_mask_test.zig").marker(Bound);
}
test "grammar_mask scope refusals and mask plan shapes" {
    try @import("grammar_mask_test.zig").scope(Bound);
}
test "grammar_mask request-owned Rules through the real-library adapter shape (FAKE tfg-shaped matcher) in the actual chain" {
    try @import("grammar_mask_test.zig").owned(Bound);
}
test "grammar_mask host resolution over the kernel's CPU reference equals the oracle" {
    try @import("grammar_mask_test.zig").exact();
}
test "grammar_mask_2 device gate: each row lands before its draws (qualification_2acf601e row 5 drew row 4's winner 27956, not 121622)" {
    try @import("grammar_mask_2_gate_test.zig").late(Bound);
}
test "grammar_mask device gate on the recorded chain (MOCK device) and its failure on a device ignoring the mask" {
    try @import("grammar_mask_test.zig").gateCpu(Bound);
}
test "grammar_mask startup (grammar_join_2, narrowed by structured_users): TF_AFD_GRAMMAR_MASK=1 with TF_AFD_USERS above 1 goes on" {
    try @import("grammar_mask_test.zig").startupUsers();
}
test "grammar_mask startup (grammar_join_2, narrowed by structured_users): TF_AFD_GRAMMAR_MASK=1 with TF_GLM_SAMPLING=1 goes on" {
    try @import("grammar_mask_test.zig").startupSampling();
}
test "grammar_mask startup (grammar_join_2): without TF_AFD_GRAMMAR_MASK nothing is refused (unset: no change)" {
    try @import("grammar_mask_test.zig").startupUnset();
}
test "grammar_mask startup (grammar_join_3, narrowed by structured_users): TF_AFD_GRAMMAR_SERVE=1 with TF_AFD_USERS above 1 or TF_GLM_SAMPLING=1 serves with the mask, refuses without it (SERVE unset: no change)" {
    try @import("grammar_mask_test.zig").startupServe();
}
test "structured_users sampled: a fixed-seed sweep of 12,000 masked draws on a 4,099-token vocabulary never draws a disallowed token" {
    try @import("structured_users_sampled_test.zig").sweep();
}
test "structured_users sampled: 480 masked draws on GLM's 154,880-token vocabulary never draw a disallowed token" {
    try @import("structured_users_sampled_test.zig").glmVocab();
}
test "structured_users sampled: the recorded masked nucleus draw: grammar_mask between the widening and the temperature; identity, empty and disallowed refusals" {
    try @import("structured_users_sampled_test.zig").recorded();
}
test "structured_users sampled: a sampled structured stream through the actual AFD Bound chain draws its script's tokens at the plain sampled positions" {
    try @import("structured_users_sampled_test.zig").chain(Bound);
}

/// qualification_qualification_e91d6b91's lane backend over the served calls: one stream, greedy, drafts off (no rank 1: its mirror tells no one).
pub const Backend = gl.Lanes(Bound);

/// A request's words when this server fails or refuses it: the exchange's failure (an expert half died or went
/// silent), else the CUDA GLM's refusals. Read on the lane thread; the server copies the words at once.
pub const Words = struct {
    node: ?*const xchg.Node,
    buf: [320]u8 = undefined,

    pub fn explain(ctx: ?*anyopaque, err: anyerror) ?[]const u8 {
        const w: *Words = @ptrCast(@alignCast(ctx.?));
        const n = w.node orelse return gl.explain(null, err);
        const x = n.failure() orelse return gl.explain(null, err);
        var d: [160]u8 = undefined;
        return std.fmt.bufPrint(&w.buf, "an AFD expert half failed ({s}; {s}): this request fails, and the attention role stops (tf-glm-afd exit 2)", .{ glm.afd_engine.describe(x, &d), @errorName(err) }) catch "an AFD expert half failed";
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Once the exchange failed: when the server listens and the lane host holds no request (the failed one answered), or
/// `grace_ns` after the failure, `stop` runs once (the binary's: SIGTERM to itself, Ash's server's stop), so the role
/// ends with the exchange's line and exit 2 and the lab's units stop; never a hang.
pub const Monitor = struct {
    node: *const xchg.Node,
    engine: api.Engine,
    grace_ns: u64,
    stop: *const fn (ctx: ?*anyopaque) void,
    stop_ctx: ?*anyopaque = null,
    period_ns: u64 = 20 * std.time.ns_per_ms,
    listening: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    fired: std.atomic.Value(bool) = .init(false),
    /// from the failure seen to the stop
    waited_ns: u64 = 0,
    thread: ?std.Thread = null,

    pub fn start(m: *Monitor) !void {
        m.thread = try std.Thread.spawn(.{}, loop, .{m});
    }

    /// The monitor stopped and joined (whether or not it fired).
    pub fn finish(m: *Monitor) void {
        m.done.store(true, .release);
        if (m.thread) |t| t.join();
        m.thread = null;
    }

    fn loop(m: *Monitor) void {
        var seen: ?u64 = null;
        while (!m.done.load(.acquire)) {
            const ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(m.period_ns) };
            _ = std.c.nanosleep(&ts, null);
            if (seen == null) {
                if (m.node.failure() == null) continue;
                seen = nowNs();
            }
            if (!m.listening.load(.acquire)) continue;
            var st: api.Status = .{};
            var none: [0]u32 = .{};
            m.engine.status(&st, &none);
            const waited = nowNs() - seen.?;
            if ((st.running == 0 and st.waiting == 0) or waited >= m.grace_ns) {
                m.waited_ns = waited;
                var d: [160]u8 = undefined;
                std.debug.print("INFO attn serve: {s}: the in-flight request failed; the server stops (exit 2) after {d} ms\n", .{ glm.afd_engine.describe(m.node.failure().?, &d), waited / std.time.ns_per_ms });
                m.fired.store(true, .release);
                m.stop(m.stop_ctx);
                return;
            }
        }
    }
};

/// The lane core and the lane host over a backend (native/cuda.zig's Host.serve, one lane): `start` begins the host's
/// thread; `stop` joins it (cancelling what is left) and frees them.
pub const Lanes = struct {
    gpa: Allocator,
    cfg: lanes.Config = undefined,
    clock: lanes.backend.WallClock = undefined,
    core: lanes.Engine = undefined,
    host: api.LaneHost = undefined,
    /// grammar_server: the attention role's Rules for structured requests (TF_AFD_GRAMMAR_SERVE=1), given to the host before it starts.
    structured: ?api.LaneHost.Structured = null,

    pub fn start(l: *Lanes, io: std.Io, back: *Backend, info: api.Info, words: *Words) !void {
        return l.startWith(io, back, info, words, null);
    }

    /// With a driver for lone drafted streams (qualification_4be15bbf lab: afd_serve_drafts.Lone), set before the host's thread starts.
    pub fn startWith(l: *Lanes, io: std.Io, back: *Backend, info: api.Info, words: *Words, lone: ?api.Lone) !void {
        l.cfg = try lanes.Config.init(l.gpa, back.facts(), 1, 0);
        errdefer l.cfg.deinit(l.gpa);
        l.clock = .{ .io = io };
        l.core = lanes.Engine.init(l.gpa, &l.cfg, back.backend(), l.clock.clock());
        errdefer l.core.deinit();
        l.host = api.LaneHost.init(l.gpa, io, &l.core, info);
        l.host.explain = .{ .ctx = words, .text = Words.explain };
        l.host.lone = lone;
        l.host.structured = l.structured;
        try l.host.start();
    }

    pub fn stop(l: *Lanes) void {
        l.host.stop();
        l.core.deinit();
        l.cfg.deinit(l.gpa);
    }
};

test "grammar_tests N1 AFD actual Bound Serve Attn masked-first constructor" {
    var s: glm.afd_serve.Serve = undefined;
    var b: Bound = .{ .s = &s };
    var back: Backend = .{ .steps = &b, .capacity = 2560, .reach = 2560 };
    try std.testing.expect(back.backend().vtable.first_masked != null);
    try @import("grammar_tests_afd_test.zig").run(Bound);
    try @import("grammar_tests_afd_alloc.zig").run(Bound);
}

test "the served window: 262,144 by default within the model's 1,048,576, --context, 0 for the model's, never past it" {
    try std.testing.expectEqual(@as(i64, 262_144), try window(null, 1_048_576));
    try std.testing.expectEqual(@as(i64, 131_008), try window(131_008, 1_048_576));
    try std.testing.expectEqual(@as(i64, 1_048_576), try window(0, 1_048_576));
    try std.testing.expectError(error.PastNative, window(2_000_000, 1_048_576));
    try std.testing.expectError(error.Negative, window(-1, 1_048_576));
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    try std.testing.expectEqual(@as(i64, 1_048_576), nativeWindow(arena.allocator(), "{\"max_position_embeddings\": 1048576}"));
    try std.testing.expectEqual(@as(i64, 0), nativeWindow(arena.allocator(), "{}"));
}
