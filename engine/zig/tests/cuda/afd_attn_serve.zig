//! tf-glm-afd attn serve (qualification_4be15bbf; qualification_e8c93577 §8 item 6, qualification_96ad5727 §8 item 5): the AFD attention role on a GPU, served by Ash's
//! server (its routes, the lane core and api.LaneHost, as tensorfold-native serves the CUDA GLM) over qualification_qualification_e91d6b91's one-stream
//! lane backend and qualification_qualification_e91d6b91's serve calls on afd_engine.Attn (glm.afd_serve), so each forward drives both expert halves
//! through the exchange exactly as `attn first` and `attn bench` do. The halves stay `tf-glm-afd expert` processes,
//! unchanged; they read their own environment. One stream, greedy, no prompt cache; drafts with `--drafts POLICY` and
//! TF_AFD_MTP=1 on the lab branch only (afd_serve_drafts.zig: qualification_0256caaf's MTP head under AFD), off by default. The endpoints are qualification_qualification_e91d6b91-serve's: /health, /v1/models, /metrics, /v1/chat/completions.
//! - `attn serve [tool flags] -- [tensorfold serve flags]`: the tool's --model is the served model; --context sets the
//!   Engine's capacity (the window + 64, the CUDA GLM's rule) and is checked against the RTX 5090 before the load
//!   (the forecast, 2 GiB of headroom); `--dry` prints that forecast on a CPU and the largest window the form fits.
//! - A half that dies or goes silent fails the in-flight request within the exchange's timeouts (silent 10 s, give-up
//!   20 s); the monitor then stops the server, and the role ends `AFD FAILED <why> peer <k> seq <n>`, exit 2.
//! - draft_shared: TF_AFD_USERS_SHARED=1 and TF_AFD_USERS_DRAFTS=1 compose in one plan; draft_shared's edits: Claude Opus 5.5
//!   (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const api = @import("engine_api");
const server = @import("server");
const gl = @import("glm_lanes");
const cli = @import("tf_glm_afd.zig");
const files = @import("afd_expert.zig");
const sl = @import("afd_serve_lanes.zig");
const sd = @import("afd_serve_drafts.zig"); // qualification_4be15bbf lab: drafts (qualification_0256caaf's MTP head under AFD)
const drafted = @import("afd_attn_drafts.zig"); // qualification_21e21bc0: TF_AFD_MTP's model
const g2 = @import("afd_attn_grammar.zig"); // grammar_server: TF_AFD_GRAMMAR_SERVE's grammar source (default off)
const grammar_abi = @import("grammar_abi"); // grammar_abi's loader (-Dgram2abi) or the stub

const xchg = cuda.xchg;
const Allocator = std.mem.Allocator;

/// What the GPU path sizes before its load: Ash's serve flags, the served window and the Engine's capacity; shared_pool's
/// `pool`: TF_AFD_POOL_TOKENS (0: off), the users' shared cache tokens; prefill_interleave's `interleave`: TF_AFD_USERS_INTERLEAVE;
/// user_sampling's `sampled`: TF_AFD_USERS_SAMPLING=1 (sampled requests served by the users); user_drafts's `users_drafts`;
/// shared_forward's `users_shared`: TF_AFD_USERS_SHARED=1 (each users round one forward over every row). shared_verify's
/// `users_shared_drafts`: TF_AFD_USERS_SHARED_DRAFTS=1 (every user's drafted window verified in that one forward).
pub const Plan = struct {
    args: server.cli.Args,
    window: i64,
    capacity: usize,
    drafts: sd.Drafts = .{},
    users: u8 = 1,
    pool: usize = 0,
    interleave: ?glm.afd_users_round.Interleave = null,
    sampled: bool = false,
    users_drafts: bool = false,
    users_shared: bool = false,
    users_shared_drafts: bool = false,
    users_ping_pong: bool = false,
    users_batched_drafts: bool = false, // batched_drafts: draft steps only, default off

    /// shared_forward's stages' rows: the users (stage 1), or shared_verify's cap (a drafted shared forward's rows at most).
    pub fn stageRows(p: Plan) u8 {
        return if (p.users_shared_drafts) @intCast(glm.afd_users_dsv.max_rows) else p.users;
    }
};

/// shared_verify: TF_AFD_USERS_SHARED_DRAFTS, read once before anything loads (unset: today's plan); 1 needs
/// TF_AFD_USERS_SHARED=1 and TF_AFD_USERS_DRAFTS=1 (which need TF_AFD_USERS 2 or 4, --drafts POLICY, TF_AFD_MTP=1);
/// anything but unset, 0 or 1 refused. Each refusal names it.
pub fn usersPingPong(shared_drafts: bool) !bool {
    const v = std.c.getenv("TF_AFD_USERS_PING_PONG") orelse return false;
    const value = std.mem.span(v);
    const asked = glm.afd_users_pp.parse(value, shared_drafts, true, 2, 0) catch |e| {
        std.debug.print("INFO attn serve: TF_AFD_USERS_PING_PONG={s}: {t}\n", .{ value, e });
        return e;
    };
    if (!asked) return false;
    const paired = try glm.afd_pairs.fromEnv();
    const lanes = try glm.afd_lanes.fromEnv(paired);
    const inflight = if (lanes != 0) lanes else try glm.afd_pairs.inflightFromEnv(paired);
    return glm.afd_users_pp.parse(value, shared_drafts, paired, inflight, lanes) catch |e| {
        std.debug.print("INFO attn serve: TF_AFD_USERS_PING_PONG=1: {t}; needs shared drafts, pairs and two decode credits; wire2 or unchanged prompt lanes3/4\n", .{e});
        return e;
    };
}

pub fn usersBatchedDrafts(shared_drafts: bool) !bool {
    const v = std.c.getenv("TF_AFD_USERS_BATCHED_DRAFTS") orelse return false;
    return glm.afd_users_bd.parse(std.mem.span(v), shared_drafts) catch |err| {
        std.debug.print("INFO attn serve: TF_AFD_USERS_BATCHED_DRAFTS={s}: unset/0 off, 1 needs TF_AFD_USERS_SHARED_DRAFTS=1 ({s})\n", .{ std.mem.span(v), @errorName(err) });
        return err;
    };
}

pub fn usersSharedDrafts(shared: bool, users_drafts: bool) !bool {
    const v = std.c.getenv("TF_AFD_USERS_SHARED_DRAFTS") orelse return false;
    const value = std.mem.span(v);
    return glm.afd_users_dsv.sharedDraftsFor(value, shared, users_drafts) catch |e| {
        switch (e) {
            error.BadAfdUsersSharedDrafts => std.debug.print("INFO attn serve: TF_AFD_USERS_SHARED_DRAFTS={s}: unset, 0 or 1 only\n", .{value}),
            error.UsersSharedDraftsNeedsShared => std.debug.print("INFO attn serve: TF_AFD_USERS_SHARED_DRAFTS=1: drafted shared verify needs TF_AFD_USERS_SHARED=1\n", .{}),
            error.UsersSharedDraftsNeedsDrafts => std.debug.print("INFO attn serve: TF_AFD_USERS_SHARED_DRAFTS=1: drafted shared verify needs TF_AFD_USERS_DRAFTS=1 (with --drafts POLICY and TF_AFD_MTP=1)\n", .{}),
        }
        return e;
    };
}

/// shared_forward: TF_AFD_USERS_SHARED, read once before anything loads (unset: today's plan); 1 needs TF_AFD_USERS 2 or 4;
/// anything but unset, 0 or 1 refused. Each refusal names it. draft_shared: with TF_AFD_USERS_DRAFTS=1 both compose (a lone
/// user drafts by user_drafts's loop, every lane-core round is the shared undrafted forward).
pub fn usersShared(users: u8) !bool {
    const v = std.c.getenv("TF_AFD_USERS_SHARED") orelse return false;
    const value = std.mem.span(v);
    return glm.afd_users_round.sharedFor(value, users) catch |e| {
        switch (e) {
            error.BadAfdUsersShared => std.debug.print("INFO attn serve: TF_AFD_USERS_SHARED={s}: unset, 0 or 1 only\n", .{value}),
            error.UsersSharedNeedsUsers => std.debug.print("INFO attn serve: TF_AFD_USERS_SHARED=1: one forward over every user's rows needs TF_AFD_USERS=2 or 4\n", .{}),
        }
        return e;
    };
}

/// user_sampling: TF_AFD_USERS_SAMPLING, read once before anything loads (unset: nothing else is read, today's plan); 1 needs
/// TF_AFD_USERS 2 or 4 and TF_GLM_SAMPLING=1 (qualification_45c10bcf's keyed draw on the attention host); anything but unset, 0 or 1 refused.
pub fn usersSampling(users: u8) !bool {
    const v = std.c.getenv("TF_AFD_USERS_SAMPLING") orelse return false;
    const value = std.mem.span(v);
    const asked = glm.afd_users_round.parseSampling(value) catch |e| {
        std.debug.print("INFO attn serve: TF_AFD_USERS_SAMPLING={s}: unset, 0 or 1 only\n", .{value});
        return e;
    };
    if (!asked) return false;
    return glm.afd_users_round.samplingFor(value, users, try glm.afd_sampling.fromEnv()) catch |e| {
        switch (e) {
            error.UsersSamplingNeedsUsers => std.debug.print("INFO attn serve: TF_AFD_USERS_SAMPLING=1: sampled users need TF_AFD_USERS=2 or 4\n", .{}),
            error.UsersSamplingNeedsGlmSampling => std.debug.print("INFO attn serve: TF_AFD_USERS_SAMPLING=1 needs TF_GLM_SAMPLING=1 (qualification_45c10bcf's keyed draw on the attention host)\n", .{}),
            else => {},
        }
        return e;
    };
}

/// Ash's `tensorfold serve` flags (after `--`, the tool's --model their model), the window and the capacity; refused
/// in words: a malformed flag, drafts (qualification_0256caaf's: the lab branch), more than one stream, a window past the model's.
/// user_drafts: TF_AFD_USERS_DRAFTS=1 lets `--drafts POLICY` (TF_AFD_MTP=1) serve with TF_AFD_USERS 2 or 4.
pub fn plan(a: Allocator, f: cli.Flags, config: []const u8) !Plan {
    const users = try glm.afd_users_round.fromEnv();
    const pool = try glm.afd_pool.fromEnv(); // shared_pool: read once, before anything loads
    const interleave = try glm.afd_users_round.interleaveFromEnv(users); // prefill_interleave: TF_AFD_USERS_INTERLEAVE, before any load
    const users_sampling = try usersSampling(users); // user_sampling: TF_AFD_USERS_SAMPLING, before any load
    const users_drafts = try glm.afd_users_round.draftsFromEnv(users); // user_drafts: TF_AFD_USERS_DRAFTS, before any load
    const users_shared = try usersShared(users); // shared_forward: TF_AFD_USERS_SHARED, before any load (draft_shared: drafts compose)
    const users_shared_drafts = try usersSharedDrafts(users_shared, users_drafts); // shared_verify: needs both, before any load
    const users_ping_pong = try usersPingPong(users_shared_drafts);
    const users_batched_drafts = try usersBatchedDrafts(users_shared_drafts);
    const argv = try a.alloc([]const u8, f.serve_args.len + 1);
    argv[0] = f.model;
    @memcpy(argv[1..], f.serve_args);
    var u: server.cli.Usage = .{};
    const args = server.cli.parse(a, argv, &u) catch |e| {
        if (e == error.Usage) std.debug.print("INFO attn serve: tensorfold serve's flags: {s}\n", .{u.message});
        return error.ServeFlags;
    };
    const drafts = try sd.parse(args.drafts); // qualification_4be15bbf lab: on needs TF_AFD_MTP=1
    if (users > 1) {
        if (drafts.on() and !users_drafts) return error.UsersDraftsUnsupported;
        if (users_drafts and !drafts.on()) { // user_drafts: the switch drafts with the single-user path's policy
            std.debug.print("INFO attn serve: TF_AFD_USERS_DRAFTS=1 needs --drafts POLICY (and TF_AFD_MTP=1)\n", .{});
            return error.UsersDraftsNeedPolicy;
        }
        // Opted-in server defaults to drafts-off; an explicit request draft:true is refused in chat and backend.
        if (args.temperature) |temp| if (temp != 0) return error.GreedyOnly;
        if (server.cli.parallelFixed(args.parallel) and server.cli.parallel(args.parallel).? != users) return error.UsersParallel;
    } else if (server.cli.parallelFixed(args.parallel) and server.cli.parallel(args.parallel).? > 1) {
        std.debug.print("INFO attn serve: --parallel {s}: the AFD attention role serves one stream\n", .{args.parallel});
        return error.OneStream;
    }
    const w = sl.window(args.context, sl.nativeWindow(a, config)) catch |e| {
        std.debug.print("INFO attn serve: --context: {t} (a token count, 0 for the model's window, never past it)\n", .{e});
        return error.ServeContext;
    };
    if (users > 1 and w > 262144) return error.UsersContext;
    if (pool > 0) { // shared_pool: the users' caches from one pool that holds at least one full-window request
        if (users < 2) {
            std.debug.print("INFO attn serve: TF_AFD_POOL_TOKENS={d}: the pool serves TF_AFD_USERS=2 or 4 only\n", .{pool});
            return error.PoolNeedsUsers;
        }
        const least = glm.afd_pool.blocksFor(glm.afd_pool.need(@intCast(w), 0)) * glm.afd_pool.block;
        if (pool < least) {
            std.debug.print("INFO attn serve: TF_AFD_POOL_TOKENS={d}: under one full-window request's region ({d} tokens at window {d})\n", .{ pool, least, w });
            return error.PoolTooSmall;
        }
    }
    return .{ .args = args, .window = w, .capacity = @as(usize, @intCast(w)) + sl.slack, .drafts = drafts, .users = users, .pool = pool, .interleave = interleave, .sampled = users_sampling, .users_drafts = users_drafts, .users_shared = users_shared, .users_shared_drafts = users_shared_drafts, .users_ping_pong = users_ping_pong, .users_batched_drafts = users_batched_drafts };
}

/// The 5090's forecast for the load's `weights` at the plan's capacity, printed; null (refused, in words) when it does
/// not leave the 2 GiB headroom of the device's free memory now.
pub fn admit(gpa: Allocator, m: *const glm.model.Model, weights: u64, p: Plan, max_rows: usize, ctx: *const cuda.Context, n_lanes: u8) !?sl.Forecast {
    const x = try sl.forecastAt(gpa, m, weights, p.capacity, max_rows, n_lanes);
    const mem = try ctx.memInfo();
    if (p.users > 1 and p.pool > 0) { // shared_pool: the Engine less slot 0's caches, the per-request States, the rows, the pool
        var q = try @import("afd_attn_users.zig").pooled(gpa, m, p.capacity, max_rows, p.users, p.pool);
        @import("afd_attn_users.zig").withDrafts(&q, m, p.users, p.users_drafts); // user_drafts: the backlog in the rows (+ 0 unset)
        @import("afd_attn_users.zig").withShared(&q, m, p.stageRows(), p.users_shared); // shared_forward: the stages in the rows (+ 0 unset; shared_verify: at its cap's rows)
        try @import("afd_attn_users.zig").withPingPong(&q, m, p.users_ping_pong);
        @import("afd_attn_users.zig").withBatchedDrafts(&q, m, p.users, p.users_batched_drafts);
        const total = x.total() - q.detached + q.total;
        std.debug.print("FORECAST shared_pool users={d} window={d} pool_tokens={d} base={d} detached={d} states={d} rows={d} pool={d} total={d} headroom={d} free={d}\n", .{ p.users, p.window, p.pool, x.total(), q.detached, q.states, q.rows, q.pool, total, sl.headroom, mem.free });
        if (total + sl.headroom > mem.free) return null;
        std.debug.print("FORECAST attn serve: window {d}, capacity {d}: shared_pool users={d} pool_tokens={d} base={d} detached={d} states={d} rows={d} pool={d} total={d} headroom={d} free={d}\n", .{ p.window, p.capacity, p.users, p.pool, x.total(), q.detached, q.states, q.rows, q.pool, total, sl.headroom, mem.free });
        return x;
    }
    if (p.users > 1) {
        var slots = try @import("afd_attn_users.zig").extra(gpa, m, p.capacity, max_rows, p.users);
        @import("afd_attn_users.zig").withDrafts(&slots, m, p.users, p.users_drafts); // user_drafts: the backlog in the rows (+ 0 unset)
        @import("afd_attn_users.zig").withShared(&slots, m, p.stageRows(), p.users_shared); // shared_forward: the stages in the rows (+ 0 unset; shared_verify: at its cap's rows)
        try @import("afd_attn_users.zig").withPingPong(&slots, m, p.users_ping_pong);
        @import("afd_attn_users.zig").withBatchedDrafts(&slots, m, p.users, p.users_batched_drafts);
        const total = x.total() + slots.total;
        std.debug.print("FORECAST concurrent_users users={d} window={d} base={d} states={d} rows={d} total={d} headroom={d} free={d}\n", .{ p.users, p.window, x.total(), slots.states, slots.rows, total, sl.headroom, mem.free });
        if (total + sl.headroom > mem.free) return null;
        std.debug.print("FORECAST attn serve: window {d}, capacity {d}: concurrent_users users={d} base={d} states={d} rows={d} total={d} headroom={d} free={d}\n", .{ p.window, p.capacity, p.users, x.total(), slots.states, slots.rows, total, sl.headroom, mem.free });
        return x;
    }
    var buf: [512]u8 = undefined;
    std.debug.print("FORECAST attn serve: window {d}, capacity {d}: {s}; free now {d:.2} GiB\n", .{ p.window, p.capacity, x.line(mem.total, &buf), sl.gib(mem.free) });
    if (x.total() + sl.headroom > mem.free) {
        std.debug.print("INFO attn serve: refused: the form needs {d:.2} GiB and 2 GiB of headroom, {d:.2} GiB is free: lower --context\n", .{ sl.gib(x.total()), sl.gib(mem.free) });
        return null;
    }
    return x;
}

/// What afd_attn_gpu.run hands the verb: the node over both halves, its Fwd, the context and the exchange.
pub const On = struct {
    at: *glm.afd_engine.Attn,
    f: *glm.fwd.Fwd,
    ctx: ?*const cuda.Context,
    node: ?*xchg.Node,
    plan: Plan,
    load_ms: f64,
    forecast: ?sl.Forecast = null,
    /// grammar_server: TF_AFD_GRAMMAR_SERVE=1's library pair as read at the role's open; null: structured output refused.
    grammar: ?g2.Env = null,
};

/// The verb's own stop (Ash's server stops on SIGTERM), the port it listens on, and its exit hook's node.
var monitor: ?*sl.Monitor = null;
var guard: ?*xchg.Node = null;
var hooked = false;

pub fn usersOnListen(port: u16) void {
    onListen(port);
}
/// structured_entry: the users entry's LaneHost memory source, the single-user entry's own (readMemory)
pub fn usersReadMemory(ctx: ?*anyopaque, reset_peak: bool) ?api.Memory {
    return readMemory(ctx, reset_peak);
}
pub fn stopUsersServer(ctx: ?*anyopaque) void {
    stopServer(ctx);
}
pub fn setUsersMonitor(m: ?*sl.Monitor, node: ?*xchg.Node) void {
    monitor = m;
    guard = node;
    if (m != null and !hooked) {
        _ = atexit(atExit);
        hooked = true;
    }
}

fn onListen(_: u16) void {
    if (monitor) |m| m.listening.store(true, .release);
}

fn stopServer(_: ?*anyopaque) void {
    std.posix.raise(.TERM) catch {};
}

extern "c" fn atexit(f: *const fn () callconv(.c) void) c_int;

/// Ash's server ends the process itself (exit 0) when replies are still open 2 s after its stop: then the experts
/// get their stop frames (a clean stop), or after a failed exchange the role's line and exit 2.
fn atExit() callconv(.c) void {
    const n = guard orelse return;
    guard = null;
    if (n.failure()) |x| {
        var lb: [256]u8 = undefined;
        std.debug.print("{s}\n", .{cli.fromFailure(.attn, x).line(&lb)});
        std.c._exit(2);
    }
    n.stopExperts();
}

/// generation_config.json's sampling, then the serve flags over it (server/main.zig's, Ash's). user_sampling: the users entry's
/// defaults too when TF_AFD_USERS_SAMPLING=1, so a request resolves to the same sampling in both entries.
pub fn sampling(a: Allocator, io: std.Io, dir: []const u8, args: server.cli.Args) !?server.json.Value {
    const json = server.json;
    const out = try json.newObject(a);
    const path = try std.fs.path.join(a, &.{ dir, "generation_config.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20))) |bytes| {
        if ((try json.parse(a, bytes)) == .ok) {
            const cfg = (try json.parse(a, bytes)).ok;
            for ([_][]const u8{ "temperature", "top_k", "top_p", "min_p" }) |k| if (cfg.field(k)) |v| try out.put(a, k, v);
            if (cfg.get("do_sample")) |d| if (d == .bool) {
                if (!d.bool) try out.put(a, "temperature", .{ .float = 0 }) else if (out.get("temperature") == null) try out.put(a, "temperature", .{ .float = 1 });
            };
        }
    } else |_| {}
    const sampled_on = try glm.afd_sampling.fromEnv();
    if (sampled_on and args.temperature == null) try out.put(a, "temperature", .{ .float = 0 });
    if (args.temperature) |t| try out.put(a, "temperature", .{ .float = t });
    if (args.top_p) |t| try out.put(a, "top_p", .{ .float = t });
    if (args.top_k) |t| try out.put(a, "top_k", try json.intValue(a, t));
    if (args.min_p) |t| try out.put(a, "min_p", .{ .float = t });
    return if (out.count() > 0) json.Value{ .object = out } else null;
}

fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const u = cuda.usage(reset_peak);
    return .{ .active = u.device, .cache = 0, .peak = u.peak };
}

fn env(name: []const u8) ?[]const u8 {
    const m = cli.environ orelse return null;
    return m.get(name);
}

/// `attn serve` on the attached node: the warm-up (both halves), the lane host, then Ash's server until SIGTERM, or
/// until the monitor stops it after a failed exchange (error.ExpertFailed: the caller's AFD line, exit 2). Gives the
/// requests served.
pub fn run(gpa: Allocator, io: std.Io, f: cli.Flags, on: On) !usize {
    if (on.plan.users > 1) return @import("afd_attn_users.zig").run(gpa, io, f, on);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const args = on.plan.args;
    server.log.init(io, false);
    var problem: []const u8 = "";
    const text = server.hf_text.HfText.load(gpa, io, f.model, a, &problem) catch |e| {
        std.debug.print("INFO attn serve: the tokenizer and template: {s}\n", .{if (problem.len > 0) problem else @errorName(e)});
        return e;
    };
    defer text.deinit();
    var glm_text: server.glm_text.GlmText = .{ .inner = text.text(), .clear = server.glm_text.clearThinking(env("TF_GLM_CLEAR_THINKING")) catch return error.ClearThinking };
    var s: glm.afd_serve.Serve = .{ .at = on.at, .f = on.f };
    var steps: sl.Bound = .{ .s = &s, .ctx = on.ctx, .sampled = try glm.afd_sampling.fromEnv(), .drafts = on.plan.drafts.on() };
    // the warm-up, as the CUDA GLM's open runs one: 96 fixed ids and 16 greedy tokens through both halves
    var warm: [gl.warm_tokens]u32 = undefined;
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const d = on.plan.drafts;
    on.at.absorb = d.on(); // qualification_4be15bbf lab: a drafted stream's prompt absorbs into the MTP head
    if (d.on()) {
        const r = try sd.warmUp(&steps, d, &warm);
        std.debug.print("INFO attn serve: warm-up drafts {s}: {d} verify forwards, drafted {d}, accepted {d}\n", .{ d.spec, r.rounds, r.drafted, r.accepted });
    } else try gl.warmUp(sl.Bound, &steps, &warm);
    const warm_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
    std.debug.print("INFO attn serve: warm-up: {d} prompt tokens and {d} greedy tokens through both halves in {d:.0} ms: {any}\n", .{ gl.warm_prompt, gl.warm_tokens, warm_ms, warm });
    const cap = on.at.e.st.capacity;
    const reach = sl.reachOf(cap, on.at.e.m.denseLimit());
    const dreach = try sd.reachOf(on.at.e.m.denseLimit(), reach); // qualification_4be15bbf lab: a drafted stream's reach
    var back: sl.Backend = .{ .steps = &steps, .capacity = cap, .reach = reach };
    var fb: [256]u8 = undefined;
    const fc = if (on.forecast) |x| x.line(sl.card_5090, &fb) else "none (recorded)";
    const startup = try std.fmt.allocPrint(a, "GLM-5.3-Flash AFD attention on CUDA (world 1; the routed experts on two tf-glm-afd expert halves over native RoCE exchange): {s}, drafts {s} (drafted reach {d}), one stream, no prompt cache; capacity {d} (window {d} + {d}), positions reached {d}, prompt chunks of {d} rows; the 5090's forecast {s}", .{ if (steps.sampled) "sampling opt-in (greedy default; sampled drafts off)" else "greedy", if (d.on()) d.spec else "off", dreach, cap, on.plan.window, sl.slack, reach, on.at.e.prefill_rows, fc });
    var words: sl.Words = .{ .node = on.node };
    var lone: sd.Lone = .{ .gpa = gpa, .back = &back, .d = d, .reach = dreach };
    const sampled = try sampling(a, io, f.model, args);
    // grammar_server: with TF_AFD_GRAMMAR_SERVE=1, one grammar source for this server: the server opens its compiler on the
    // first structured request, the lane host takes each structured request's Rules from it, and it is closed after
    // the lane host stopped (its deferred close runs after the lane host's)
    var grammar_src: ?g2.Served = if (on.grammar) |env_pair| g2.Served.init(gpa, io, env_pair, on.at.e.m.vocab, if (comptime g2.has_abi) &grammar_abi.trusted.global else {}) else null;
    defer if (grammar_src) |*g| g.close() catch |e| std.debug.print("INFO attn serve: grammar_server: the grammar compiler stays open: {s} (a matcher still lives)\n", .{@errorName(e)});
    var ln: sl.Lanes = .{ .gpa = gpa };
    if (grammar_src) |*g| ln.structured = g.structured();
    try ln.startWith(io, &back, .{ .lanes = 1, .context_window = @intCast(on.plan.window), .startup = startup, .drafts = if (d.on()) d.spec else "off", .structures = grammar_src != null }, &words, if (d.on()) lone.driver() else null);
    var lanes_up = true;
    defer if (lanes_up) ln.stop();
    if (on.ctx != null) ln.host.memory = .{ .read = readMemory };
    if (env("TENSORFOLD_MAX_QUEUED")) |q| ln.host.max_queued = std.fmt.parseInt(u32, std.mem.trim(u8, q, " "), 10) catch return error.MaxQueued;
    var mon: sl.Monitor = .{ .node = undefined, .engine = ln.host.engine(), .grace_ns = @intFromFloat((f.give_up_s + 5) * std.time.ns_per_s), .stop = stopServer };
    if (on.node) |n| {
        mon.node = n;
        monitor = &mon;
        try mon.start();
        guard = n;
        if (!hooked) hooked = atexit(atExit) == 0;
    }
    const started = std.Io.Clock.awake.now(io).toNanoseconds() - @as(i96, @intFromFloat(on.load_ms * 1e6));
    const code = server.serve.run(gpa, io, args, .{
        .engine = ln.host.engine(),
        .text = glm_text.text(),
        .served = server.hub.servedName(args.name, args.model, f.model),
        .model_dir = f.model,
        .sampling = sampled,
        .environ = cli.environ,
        .grammars = if (grammar_src) |*g| g.source() else null,
        .started = started,
        .on_listen = onListen,
    });
    // the monitor, then the lane host (its thread joined, a request left cancelled); the role's own end follows
    mon.finish();
    monitor = null;
    ln.stop();
    lanes_up = false;
    guard = null;
    const served = steps.prefills -| 1;
    std.debug.print("INFO attn serve: the server stopped (code {d}): {d} requests served\n", .{ code, served });
    if (on.at.failure() != null) return error.ExpertFailed;
    if (code != 0) return error.ServeFailed;
    return served;
}

/// `attn serve --dry` on a CPU over the checkpoint's headers (or qualification_765330ce's header mirror): the form the switches pick
/// (TF_GLM_DENSE, TF_GLM_F8), its forecast at the plan's capacity against the RTX 5090, and the largest window that
/// form fits with the 2 GiB headroom (a multiple of 64, at most 262,144). Nothing loads; no GPU, no network.
pub fn dry(gpa: Allocator, f: cli.Flags, config: []const u8) !cli.Outcome {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try plan(a, f, config);
    var m = try drafted.model(gpa, config, f.layers); // no MTP under AFD (engine.py:499-502), but with TF_AFD_MTP=1 (qualification_21e21bc0)
    var ix = try glm.index.open(gpa, f.model);
    defer ix.deinit();
    var c = try files.config(gpa, f.model);
    c.run = @intCast(m.layers);
    var ld = try glm.afd_load.loadWith(gpa, &m, &c, &ix.names, m.mtp_layers > 0);
    defer ld.deinit();
    const d = try glm.afd_load.dense(gpa, &ld, &ix, &m);
    const form = if (d == null) "BF16" else if (d.?.f8_saved == null) (if (d.?.mtp8 == null) "q4, BF16 head" else "q4, BF16 head, FP8 MTP") else if (d.?.mtp8 == null) "q4, FP8 head" else "q4, FP8 head, FP8 MTP"; // mtp_fp8: TF_GLM_MTP_FP8=1
    const weights = ld.deviceBytes();
    const lanes_n = try glm.afd_lanes.fromEnv(try glm.afd_pairs.fromEnv()); // qualification_9e49592e: TF_AFD_LANES' copies in the forecast (BadAfdLanes)
    const x = try sl.forecastAt(gpa, &m, weights, p.capacity, f.max_rows, lanes_n);
    if (p.users > 1 and p.pool > 0) { // shared_pool: the pooled forecast on the header mirrors, and the largest pool that fits
        const served = @import("afd_attn_users.zig");
        var q = try served.pooled(gpa, &m, p.capacity, f.max_rows, p.users, p.pool);
        served.withDrafts(&q, &m, p.users, p.users_drafts); // user_drafts: the backlog in the rows (+ 0 unset)
        served.withShared(&q, &m, p.stageRows(), p.users_shared); // shared_forward: the stages in the rows (+ 0 unset; shared_verify: at its cap's rows)
        try served.withPingPong(&q, &m, p.users_ping_pong);
        served.withBatchedDrafts(&q, &m, p.users, p.users_batched_drafts);
        const total = x.total() - q.detached + q.total;
        const fixed = total - q.pool;
        const most = try served.largestPool(gpa, &m, p.capacity, sl.card_5090 -| sl.headroom -| fixed);
        std.debug.print("FORECAST shared_pool dry users={d} window={d} capacity={d} pool_tokens={d} base={d} detached={d} states={d} rows={d} pool={d} total={d} headroom={d} card={d} largest_pool={d} MTP_loaded={}\n", .{ p.users, p.window, p.capacity, p.pool, x.total(), q.detached, q.states, q.rows, q.pool, total, sl.headroom, sl.card_5090, most, m.mtp_layers > 0 });
        // four_user_profile join: prefill_interleave's dry line also on the pooled form (its one --dry INFO line wherever the users forecast is)
        if (p.interleave) |il| std.debug.print("INFO prefill_interleave dry interleave={d}:{d}: a new prompt goes in a step of the served pass at a time, {d} decode round(s) of the live users between turns of {d} step(s); no device memory change\n", .{ il.rounds, il.steps, il.rounds, il.steps });
        return .{ .ok = try std.fmt.allocPrint(gpa, "serve --dry shared_pool users {d} window {d} pool {d} total {d} {s}; largest pool {d}; hardware equality UNKNOWN", .{ p.users, p.window, p.pool, total, if (total + sl.headroom <= sl.card_5090) "capacity-fit-projection" else "CAPACITY-LIMITED", most }) };
    }
    if (p.users > 1) {
        var slots = try @import("afd_attn_users.zig").extra(gpa, &m, p.capacity, f.max_rows, p.users);
        @import("afd_attn_users.zig").withDrafts(&slots, &m, p.users, p.users_drafts); // user_drafts: the backlog in the rows (+ 0 unset)
        @import("afd_attn_users.zig").withShared(&slots, &m, p.stageRows(), p.users_shared); // shared_forward: the stages in the rows (+ 0 unset; shared_verify: at its cap's rows)
        try @import("afd_attn_users.zig").withPingPong(&slots, &m, p.users_ping_pong);
        @import("afd_attn_users.zig").withBatchedDrafts(&slots, &m, p.users, p.users_batched_drafts);
        const total = x.total() + slots.total;
        std.debug.print("FORECAST concurrent_users dry users={d} window={d} capacity={d} base={d} states={d} rows={d} total={d} headroom={d} MTP_loaded={}\n", .{ p.users, p.window, p.capacity, x.total(), slots.states, slots.rows, total, sl.headroom, m.mtp_layers > 0 });
        if (p.interleave) |il| std.debug.print("INFO prefill_interleave dry interleave={d}:{d}: a new prompt goes in a step of the served pass at a time, {d} decode round(s) of the live users between turns of {d} step(s); no device memory change\n", .{ il.rounds, il.steps, il.rounds, il.steps });
        return .{ .ok = try std.fmt.allocPrint(gpa, "serve --dry concurrent_users users {d} window {d} total {d} {s}; hardware equality UNKNOWN", .{ p.users, p.window, total, if (total + sl.headroom <= sl.card_5090) "capacity-fit-projection" else "CAPACITY-LIMITED" }) };
    }
    const most = try sl.largestWindow(gpa, &m, weights, f.max_rows, sl.card_5090, lanes_n);
    var buf: [512]u8 = undefined;
    std.debug.print("FORECAST attn serve --dry ({s}{s}, {d} layers): window {d}, capacity {d}: {s}\n", .{ form, if (m.mtp_layers > 0) ", the MTP head (TF_AFD_MTP=1)" else "", m.layers, p.window, p.capacity, x.line(sl.card_5090, &buf) });
    std.debug.print("INFO attn serve --dry: the largest window this form fits on the RTX 5090 with 2 GiB of headroom: {d}\n", .{most});
    return .{ .ok = try std.fmt.allocPrint(gpa, "serve --dry {s} window {d} forecast {d:.2} GiB {s}; largest window {d}", .{ form, p.window, sl.gib(x.total()), if (x.fits(sl.card_5090)) "fits" else "PAST", most }) };
}

test {
    _ = @import("afd_serve_lanes.zig");
    _ = @import("afd_serve_drafts.zig"); // qualification_4be15bbf lab: drafts in serve and G-draft-serve
    _ = &run;
    _ = &admit;
    _ = &dry;
}

test "attn serve's flags: Ash's serve flags after --, the window and capacity; drafts, several streams and a window past the model's refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\": 1048576}";
    const f = try cli.parse(a, .attn, &.{ "serve", "--master", "[::1]:18940", "--model", "/m", "--", "--host", "0.0.0.0", "--port", "18180", "--context", "131008", "--name", "glm-afd" });
    try std.testing.expect(f.verb == .serve and f.prompt == null and f.serve_args.len == 8);
    const p = try plan(a, f, cfg);
    try std.testing.expectEqual(@as(i64, 131_008), p.window);
    try std.testing.expectEqual(@as(usize, 131_072), p.capacity);
    try std.testing.expect(p.args.port == 18180 and std.mem.eql(u8, p.args.host, "0.0.0.0") and std.mem.eql(u8, p.args.model, "/m"));
    const d = try plan(a, try cli.parse(a, .attn, &.{ "serve", "--master", "[::1]:18940", "--model", "/m" }), cfg);
    try std.testing.expectEqual(@as(usize, 262_208), d.capacity); // 262,144 + 64 by default, never 1M-class
    const off = try plan(a, try cli.parse(a, .attn, &.{ "serve", "--master", "[::1]:18940", "--model", "/m", "--", "--drafts", "0", "--parallel", "1" }), cfg);
    try std.testing.expectEqual(@as(usize, 262_208), off.capacity);
    const ex = std.testing.expectError;
    try ex(error.DraftsNeedTfAfdMtp, plan(a, try cli.parse(a, .attn, &.{ "serve", "--master", "[::1]:18940", "--model", "/m", "--", "--drafts", "auto" }), cfg));
    try ex(error.OneStream, plan(a, try cli.parse(a, .attn, &.{ "serve", "--master", "[::1]:18940", "--model", "/m", "--", "--parallel", "2" }), cfg));
    try ex(error.ServeContext, plan(a, try cli.parse(a, .attn, &.{ "serve", "--master", "[::1]:18940", "--model", "/m", "--", "--context", "2000000" }), cfg));
    try ex(error.ServeFlags, plan(a, try cli.parse(a, .attn, &.{ "serve", "--master", "[::1]:18940", "--model", "/m", "--", "--bogus" }), cfg));
    try ex(error.NoMaster, cli.parse(a, .attn, &.{ "serve", "--model", "/m" }));
    try ex(error.ServeArgsAreServes, cli.parse(a, .attn, &.{ "first", "--model", "/m", "--prompt", "p", "--dry", "--", "--port", "1" }));
    try ex(error.ServeAudits, cli.parse(a, .attn, &.{ "serve", "--master", "[::1]:18940", "--model", "/m", "--audit" }));
    const dr = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/m", "--", "--context", "65536" });
    try std.testing.expect(dr.dry and dr.master == null);
    var buf: [512]u8 = undefined;
    try std.testing.expectEqualStrings("attn serve model /m layers all dev mlx5_0 gid 3 max-rows 2048 silent 10s give-up 20s watcher true -- --context 65536 dry", cli.describe(dr, &buf));
}
