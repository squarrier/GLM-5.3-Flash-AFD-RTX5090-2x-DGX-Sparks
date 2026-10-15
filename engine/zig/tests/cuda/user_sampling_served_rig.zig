//! user_sampling (local integration; Claude Opus 5.5 via Hermes): test-user_sampling_served's rig — structured_entry's served boot (the verb's entry
//! `afd_attn_serve.run` on the recorder, which hands TF_AFD_USERS > 1 to `afd_attn_users.run`; plan(), the switches the
//! lab sets, the REAL GLM-5.3 tokenizer directory and grammar_abi's libtfgrammar) with a MOCK keyed draw. A recorded sampled
//! call answers qualification_45c10bcf's host draw (`nucleus_draw.drawRow`: Gumbel-max over uniform(seed, position, id) after the top_k,
//! top_p and min_p cuts) over a fixed synthetic distribution, under the Engine's Sampling AT THAT CALL and the call's own
//! position — the parameters and keys the device draw would be handed — so a reply moves with exactly what the real
//! draw reads. Greedy calls keep structured_entry's answer (`ok`); a masked sampled call keeps structured_entry's preference among the STAGED
//! words. Every draw is logged in order (greedy, keyed or masked; the seed and position of a sampled one).
//! Rig: structured_entry's (Claude Opus 5.5 via Hermes) after concurrent_users_r2's Rec (GPT-6.1 Sol via Hermes) and Ash Hart's
//! cuda_afd_engine_test.zig; inherited NOTICE credits apply (Ash Hart / TensorFold; Mia (@MiaAI_lab), whose union cover
//! qualification_45c10bcf's draw carries; the XGrammar contributors (MLC, Apache-2.0) through grammar_abi's library).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const server = @import("server");
const served = @import("afd_attn_serve.zig");
const cli = @import("tf_glm_afd.zig");
const g2 = @import("afd_attn_grammar.zig");
const hx = @import("user_sampling_served_http.zig");
const md = glm.sample.mask_dev;
const nd = glm.nucleus_draw;
const proto = cuda.xchg.proto;
const t = std.testing;
const gpa = t.allocator;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// the served window and the Engine's capacity (window + afd_serve_lanes' slack), as plan() computes them
pub const window = 2496;
pub const capacity = 2560;
/// one prompt chunk holds the warm-up's 96 ids (structured_entry's)
const prefill_rows = 96;

pub fn case(want: []const u8) bool {
    const v = std.c.getenv("user_sampling_CASE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), want);
}

fn env(n: [:0]const u8) ![]const u8 {
    const v = std.c.getenv(n) orelse {
        std.debug.print("user_sampling needs {s} (zrun --fix srvabi: grammar_server's real-library mounts)\n", .{n});
        return error.MissingEnv;
    };
    return std.mem.span(v);
}

fn set(n: [:0]const u8, v: [:0]const u8) void {
    _ = setenv(n, v, 1);
}

// ---- the MOCK device ------------------------------------------------------------------------------------------------

pub const DrawKind = enum { greedy, keyed, masked };
pub const Draw = struct { kind: DrawKind, seed: u64 = 0, position: u64 = 0 };

/// the synthetic distribution's words (single ids of this tokenizer) and their fixed logits; end-of-text last
const words_text = [_][]const u8{ " the", " sky", " is", " blue", " and", " clear", " light", " today", ",", ".", " bright" };
const words_logit = [_]f64{ 2.0, 1.6, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.2, -0.6 };
const eos_logit: f64 = -3.0;
const max_cand = words_text.len + 1;

pub const Mock = struct {
    dev: *md.Dev,
    trace: *glm.trace.Trace,
    engine: *glm.engine.Engine,
    text: server.model_text.Text,
    /// structured_entry's: end-of-text first, then " } ] , : { [ 1 (a string closes first, a value before a new one opens)
    order: [8]u32,
    eos: []const u32,
    plain: u32,
    ids: [max_cand]u32 = undefined,
    logits: [max_cand]f64 = undefined,
    n: usize = 0,
    staged_seen: u64 = 0,
    plain_draws: usize = 0,
    keyed_draws: usize = 0,
    masked_sampled: usize = 0,
    masked_greedy: usize = 0,
    resets: usize = 0,
    lock: std.atomic.Mutex = .unlocked,
    log: std.ArrayList(Draw) = .empty,
    /// user_sampling fail case: a keyed draw under this seed at this position fails as a device fault would (null: never)
    fail_at: ?struct { seed: u64, position: u64 } = null,
    failed: usize = 0,

    fn of(p: *anyopaque) *Mock {
        return @ptrCast(@alignCast(p));
    }
    pub fn picker(m: *Mock) glm.trace.Picker {
        return .{ .ctx = m, .pick = pick };
    }
    pub fn emulator(m: *Mock) md.Emulator {
        return .{ .ctx = m, .candidate = candidate };
    }
    /// the words that encode to one id each (and end-of-text), with their logits
    pub fn vocab(m: *Mock, a: std.mem.Allocator) !void {
        for (words_text, words_logit) |s, l| {
            const ids = try m.text.encode(a, s, false);
            if (ids.len != 1) continue;
            m.ids[m.n] = ids[0];
            m.logits[m.n] = l;
            m.n += 1;
        }
        if (m.n < 8) return error.MockVocabulary;
        m.ids[m.n] = m.eos[0];
        m.logits[m.n] = eos_logit;
        m.n += 1;
    }
    fn eligible(w: []const u32, id: u32) bool {
        return id / 32 < w.len and (w[id / 32] >> @as(u5, @intCast(id % 32))) & 1 == 1;
    }
    fn blank(m: *Mock, id: u32) bool {
        var buf: [512]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const s = m.text.decode(fba.allocator(), &.{id}) catch return false;
        return s.len == 0 or std.mem.indexOfAny(u8, s, " \t\r\n") != null;
    }
    fn prefer(m: *Mock, w: []const u32) !u32 {
        for (m.eos) |e| if (eligible(w, e)) return e;
        for (m.order) |id| if (eligible(w, id)) return id;
        var lowest: ?u32 = null;
        for (w, 0..) |w0, i| {
            var x = w0;
            while (x != 0) : (x &= x - 1) {
                const id: u32 = @intCast(i * 32 + @ctz(x));
                if (lowest == null) lowest = id;
                if (!m.blank(id)) return id;
            }
        }
        return lowest orelse error.EmptyStaged;
    }
    /// qualification_45c10bcf's host draw over the synthetic row at `position` under `s`: values logit * (1 / max(t, 1e-6)) (ATen's f64
    /// division form), masses floor(exp(v - max) * 2^40), one rank holding every candidate
    pub fn keyed(m: *const Mock, s: nd.Sampling, position: u64) !u32 {
        const n = m.n;
        var w: [2 + 3 * max_cand]u64 = undefined;
        const inv = 1.0 / @max(s.temperature, 1e-6);
        var top = -std.math.inf(f64);
        for (m.logits[0..n]) |l| top = @max(top, l * inv);
        var total: u64 = 0;
        w[0] = 0;
        for (0..n) |j| {
            const v = m.logits[j] * inv;
            const mass: u64 = @intFromFloat(@floor(@exp(v - top) * 0x1p40));
            total += mass;
            w[2 + j] = @bitCast(v);
            w[2 + n + j] = @bitCast(@as(i64, m.ids[j]));
            w[2 + 2 * n + j] = @bitCast(@as(i64, @intCast(mass)));
        }
        w[1] = total;
        const p: nd.Packet = .{ .words = w[0 .. 2 + 3 * n], .world = 1, .rows = 1, .count = n };
        const cuts = [_]u64{ 0, n };
        var work: [max_cand]nd.Cand = undefined;
        const d = (try nd.drawRow(p, nd.Shards.of(&cuts), 0, position, s, &work)) orelse return error.MockNotCovered;
        return d.token;
    }
    fn hold(m: *Mock) void {
        while (!m.lock.tryLock()) std.Thread.yield() catch {};
    }
    fn note(m: *Mock, d: Draw) void {
        m.hold();
        defer m.lock.unlock();
        m.log.append(gpa, d) catch {};
    }
    /// keeps the recorder's memory bounded over a long served run (only the trace's own bookkeeping is dropped)
    fn trim(m: *Mock) void {
        if (m.trace.events.items.len > 200_000) {
            m.trace.reset();
            m.resets += 1;
        }
    }
    fn pick(p: *anyopaque, _: glm.trace.T, positions: ?[]const u64, out: []u32, probs: ?[]f64) !void {
        const m = of(p);
        if (positions) |pos| {
            const s = m.engine.sampling orelse return error.MockSampledWithoutSampling;
            if (m.dev.stats.staged != m.staged_seen) {
                m.staged_seen = m.dev.stats.staged;
                m.masked_sampled += 1;
                @memset(out, try m.prefer(m.dev.host[0..m.dev.words]));
                m.note(.{ .kind = .masked, .seed = s.seed, .position = pos[0] });
            } else {
                if (m.fail_at) |x| if (s.seed == x.seed and pos[0] == x.position) {
                    m.failed += 1;
                    return error.MockDeviceFault;
                };
                for (pos, 0..) |q, i| out[i] = try m.keyed(s, q);
                m.keyed_draws += 1;
                m.note(.{ .kind = .keyed, .seed = s.seed, .position = pos[0] });
            }
        } else {
            m.plain_draws += 1;
            @memset(out, m.plain);
            m.note(.{ .kind = .greedy });
        }
        if (probs) |x| @memset(x, 0);
        m.trim();
    }
    fn candidate(p: *anyopaque, x: *const md.Dev, _: glm.trace.T) !md.Candidate {
        const m = of(p);
        m.staged_seen = x.stats.staged;
        m.masked_greedy += 1;
        const id = try m.prefer(x.host[0..x.words]);
        m.note(.{ .kind = .greedy });
        m.trim();
        return .{ .id = id, .bits = 0x3f80_0000 };
    }
    /// the draws logged so far (a copy, under the lock)
    pub fn draws(m: *Mock, a: std.mem.Allocator) ![]Draw {
        m.hold();
        defer m.lock.unlock();
        return a.dupe(Draw, m.log.items);
    }
};

// ---- the proxy engine: structured_entry's recorded rig ---------------------------------------------------------------------------

const Rig = struct {
    parsed: std.json.Parsed(std.json.Value) = undefined,
    m: glm.model.Model = undefined,
    w: glm.views.Weights = undefined,
    e: *glm.engine.Engine = undefined,
    a: glm.afd_attn.Afd = undefined,
    at: glm.afd_engine.Attn = undefined,
    space: glm.trace.Space = undefined,
    trace: glm.trace.Trace = undefined,
    f: glm.fwd.Fwd = undefined,
    region: []align(4096) u8 = &.{},
    status: []u32 = &.{},
    node: cuda.xchg.Node = undefined,
    dev: md.Dev = undefined,
    posts: usize = 0,
    takes: usize = 0,

    /// `pooled`: slot 0's caches detached as afd_attn_gpu does for TF_AFD_POOL_TOKENS; `sampling`: TF_GLM_SAMPLING=1's
    /// nucleus and top_k (afd_attn_gpu's useSampling), before the temporaries are placed
    fn init(r: *Rig, pooled: bool, sampling: bool) !void {
        r.parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
        r.m = try glm.model.Model.parse(gpa, r.parsed.value.object.get("config_json").?.string, 1, 0);
        try r.m.cut(5);
        r.m.mtp_layers = 0;
        r.w = try glm.views.rank(gpa, &r.m);
        r.e = try glm.engine.Engine.create(gpa, &r.m, &r.w, capacity, 8, prefill_rows, true);
        if (sampling) {
            try r.e.useSampling(gpa);
            r.e.sampled_top_k = true;
        }
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
        const settings: proto.Settings = .{ .layout = r.a.layout, .max_rows = prefill_rows, .ports = 1 };
        r.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast((proto.Region{ .slot = settings.slotBytes() }).total()));
        @memset(r.region, 0);
        r.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(r.status, 0);
        r.node = try cuda.xchg.Node.init(gpa, t.io, .attn, .{ .settings = settings, .warn_failures = true }, .{ .region = r.region, .region_dev = @intFromPtr(r.region.ptr), .status = r.status, .status_dev = @intFromPtr(r.status.ptr) });
        r.a.wire = .{ .node = &r.node, .host = .{ .ctx = r, .post = post, .take = take } };
        if (pooled) glm.afd_pool.detachTokens(&r.e.st);
        r.dev = try md.Dev.recorded(gpa, &r.m, &r.space);
        r.e.smp.masked_dev = &r.dev; // TF_AFD_GRAMMAR_MASK=1's adapter on the sampler, as afd_attn_gpu installs it
    }
    fn deinit(r: *Rig) void {
        r.dev.close();
        r.node.deinit();
        gpa.free(r.region);
        gpa.free(r.status);
        r.trace.deinit();
        r.space.deinit();
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.parsed.deinit();
    }
    fn post(p: *anyopaque, _: *cuda.xchg.Node, _: *const [proto.HDR_BYTES]u8, _: usize, _: usize) !void {
        const r: *Rig = @ptrCast(@alignCast(p));
        r.posts += 1;
    }
    fn take(p: *anyopaque, _: *cuda.xchg.Node, _: usize, _: usize) !void {
        const r: *Rig = @ptrCast(@alignCast(p));
        r.takes += 1;
    }
};

// ---- one boot: the switches, plan(), the rig, the verb's entry on its own thread, the loopback port ------------------

pub const Words = []const [2][:0]const u8;
const all_names = [_][:0]const u8{ "TF_AFD_USERS", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE", "TF_AFD_GRAMMAR_MASK", "TF_AFD_GRAMMAR_SERVE", "TF_AFD_GRAMMAR_LIB", "TF_AFD_GRAMMAR_LIB_SHA256", "TF_GLM_SAMPLING", "TF_AFD_USERS_SAMPLING" };

pub const Boot = struct {
    rig: Rig = .{},
    mock: Mock = undefined,
    hf: *server.hf_text.HfText = undefined,
    arena: std.heap.ArenaAllocator = undefined,
    port: u16 = 0,
    port_text: [8]u8 = undefined,
    thread: std.Thread = undefined,
    result: anyerror!usize = 0,
    done: std.atomic.Value(bool) = .init(false),
    /// the server answered /health: serve.run's SIGTERM handler is installed (stop may raise it)
    listening: bool = false,
    flags: cli.Flags = .{},
    on: served.On = undefined,
    words: Words = &.{},
    users: u8 = 1,
    /// the recorder's placed device bytes after the rig's placement (weights, the Engine, its temporaries): the CPU
    /// form of the G-mem comparison (switch on = off)
    placed_bytes: u64 = 0,

    /// `words`: the attention line's switches, set before plan() and the role's reads, as the lab does
    pub fn start(b: *Boot, words: Words) !void {
        for (all_names) |n| _ = unsetenv(n);
        b.words = words;
        b.arena = std.heap.ArenaAllocator.init(gpa);
        const a = b.arena.allocator();
        const dir = try env("TF_grammar_server_MODEL");
        set("TF_AFD_GRAMMAR_LIB", try a.dupeSentinel(u8, try env("TF_grammar_server_LIB"), 0));
        set("TF_AFD_GRAMMAR_LIB_SHA256", try a.dupeSentinel(u8, try env("TF_grammar_server_LIB_SHA256"), 0));
        for (words) |kv| set(kv[0], kv[1]);
        var pooled = false;
        var sampling = false;
        var masked = false;
        for (words) |kv| {
            if (std.mem.eql(u8, kv[0], "TF_AFD_POOL_TOKENS")) pooled = true;
            if (std.mem.eql(u8, kv[0], "TF_GLM_SAMPLING") and std.mem.eql(u8, kv[1], "1")) sampling = true;
            if (std.mem.eql(u8, kv[0], "TF_AFD_GRAMMAR_MASK") and std.mem.eql(u8, kv[1], "1")) masked = true;
        }
        b.port = 20000 + @as(u16, @intCast(@mod(std.c.getpid(), 20000)));
        const port_s = try std.fmt.bufPrint(&b.port_text, "{d}", .{b.port});
        // zlv1.sh's SARGS (--host --port --name --context --temperature 0 --drafts 0), as structured_entry's boot
        const args = try a.dupe([]const u8, &.{ "--host", "127.0.0.1", "--port", port_s, "--name", hx.name, "--context", "2496", "--temperature", "0", "--drafts", "0" });
        b.flags = .{ .role = .attn, .verb = .serve, .model = dir, .serve_args = args };
        const config = try std.Io.Dir.cwd().readFileAlloc(t.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(16 << 20));
        const plan = try served.plan(a, b.flags, config);
        try t.expectEqual(@as(usize, capacity), plan.capacity);
        b.users = plan.users;
        const grammar = try g2.fromEnv(masked);
        try b.rig.init(pooled, sampling);
        for (b.rig.space.entries.items) |x| b.placed_bytes += x.len;
        var problem: []const u8 = "";
        b.hf = try server.hf_text.HfText.load(gpa, t.io, dir, a, &problem);
        const text = b.hf.text();
        var order: [8]u32 = undefined;
        for ([_][]const u8{ "\"", "}", "]", ",", ":", "{", "[", "1" }, &order) |s, *id| {
            const ids = try text.encode(a, s, false);
            try t.expectEqual(@as(usize, 1), ids.len);
            id.* = ids[0];
        }
        const plain_ids = try text.encode(a, "ok", false);
        b.mock = .{ .dev = &b.rig.dev, .trace = &b.rig.trace, .engine = b.rig.e, .text = text, .order = order, .eos = text.eosIds(), .plain = plain_ids[0] };
        try b.mock.vocab(a);
        b.rig.trace.picker = b.mock.picker();
        b.rig.dev.emulate = b.mock.emulator();
        b.on = .{ .at = &b.rig.at, .f = &b.rig.f, .ctx = null, .node = null, .plan = plan, .load_ms = 1234.5, .grammar = grammar };
        std.debug.print("user_sampling boot: users {d} pool {d} interleave {any} sampling {} grammar {} window {d} capacity {d}; words", .{ plan.users, plan.pool, plan.interleave, sampling, grammar != null, plan.window, plan.capacity });
        for (words) |kv| std.debug.print(" {s}={s}", .{ kv[0], kv[1] });
        std.debug.print("; port {d}; keyed ids {any}\n", .{ b.port, b.mock.ids[0..b.mock.n] });
        b.thread = try std.Thread.spawn(.{}, run, .{b});
        var tries: usize = 0;
        while (tries < 6000) : (tries += 1) {
            if (b.done.load(.acquire)) break;
            var scratch = std.heap.ArenaAllocator.init(gpa);
            defer scratch.deinit();
            const raw = hx.get(scratch.allocator(), b.port, "/health") catch {
                std.Io.sleep(t.io, .fromMilliseconds(50), .awake) catch {};
                continue;
            };
            if (@import("grammar_server_http.zig").status(raw) == 200) {
                b.listening = true;
                return;
            }
            std.Io.sleep(t.io, .fromMilliseconds(50), .awake) catch {};
        }
        if (b.done.load(.acquire)) b.thread.join();
        std.debug.print("user_sampling boot: the server never answered /health: {any}\n", .{b.result});
        return error.ServerNotUp;
    }

    fn run(b: *Boot) void {
        b.result = served.run(gpa, t.io, b.flags, b.on);
        b.done.store(true, .release);
    }

    /// Ash's server's stop (served.stopUsersServer: SIGTERM to this process), then the entry's own end (structured_entry's)
    pub fn stop(b: *Boot) !usize {
        if (!b.listening) return error.ServerNotUp; // no handler installed: a SIGTERM would end this process
        served.stopUsersServer(null);
        b.thread.join();
        b.mock.log.deinit(gpa);
        b.hf.deinit();
        b.rig.deinit();
        b.arena.deinit();
        for (all_names) |n| _ = unsetenv(n);
        return b.result;
    }
};

// ---- the case's record: every reply and the boot's words, for scripts/check_served.py's cross-boot comparisons ----------

fn jsonStr(out: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    try out.append(a, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(a, "\\\""),
        '\\' => try out.appendSlice(a, "\\\\"),
        '\n' => try out.appendSlice(a, "\\n"),
        '\r' => try out.appendSlice(a, "\\r"),
        '\t' => try out.appendSlice(a, "\\t"),
        0...8, 11, 12, 14...31 => try out.print(a, "\\u{x:0>4}", .{c}),
        else => try out.append(a, c),
    };
    try out.append(a, '"');
}

/// `user_sampling_OUT/<case>.json` (nothing when user_sampling_OUT is unset): the case, its boot's words, every reply in order
pub fn record(a: std.mem.Allocator, case_name: []const u8, words: Words, replies: []const hx.Reply, facts: []const u8) !void {
    const dir = std.c.getenv("user_sampling_OUT") orelse return;
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\"case\":");
    try jsonStr(&out, a, case_name);
    try out.appendSlice(a, ",\"words\":{");
    for (words, 0..) |kv, i| {
        if (i > 0) try out.append(a, ',');
        try jsonStr(&out, a, kv[0]);
        try out.append(a, ':');
        try jsonStr(&out, a, kv[1]);
    }
    try out.print(a, "}},\"facts\":{s},\"replies\":[", .{if (facts.len > 0) facts else "{}"});
    for (replies, 0..) |r, i| {
        if (i > 0) try out.appendSlice(a, ",\n");
        try out.appendSlice(a, "{\"label\":");
        try jsonStr(&out, a, r.label);
        try out.print(a, ",\"kind\":\"{s}\",\"stream\":{},\"status\":{d},\"finish\":", .{ @tagName(r.kind), r.stream, r.status });
        try jsonStr(&out, a, r.finish);
        try out.appendSlice(a, ",\"content\":");
        try jsonStr(&out, a, r.content);
        try out.appendSlice(a, ",\"usage\":");
        try jsonStr(&out, a, r.usage);
        try out.appendSlice(a, ",\"message\":");
        try jsonStr(&out, a, r.message);
        try out.appendSlice(a, ",\"token_sha\":");
        try jsonStr(&out, a, r.token_sha);
        try out.appendSlice(a, ",\"sampling\":");
        try jsonStr(&out, a, r.sampling);
        try out.print(a, ",\"drafted\":{?d},\"accepted\":{?d},\"prompt_tokens\":{d},\"completion_tokens\":{d},\"seconds\":{d},\"prefill_seconds\":{d}}}", .{ r.drafted, r.accepted, r.prompt_tokens, r.completion_tokens, r.seconds, r.prefill_seconds });
    }
    try out.appendSlice(a, "]}\n");
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}.json", .{ std.mem.span(dir), case_name }, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    var at: usize = 0;
    while (at < out.items.len) {
        const n = std.c.write(fd, out.items[at..].ptr, out.items.len - at);
        if (n <= 0) return error.CannotWrite;
        at += @intCast(n);
    }
}
