//! concurrent_users opt-in served users; the original afd_attn_serve.run path is untouched when TF_AFD_USERS is unset/1.
//! draft_shared: shared_forward's INFO line when user_drafts's drafts are on too; draft_shared's edits: Claude Opus 5.5 (Anthropic; planning
//! and execution), via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const users = @import("glm_users");
const api = @import("engine_api");
const lanes = @import("lanes");
const server = @import("server");
const sl = @import("afd_serve_lanes.zig");
const cli = @import("tf_glm_afd.zig");
const served = @import("afd_attn_serve.zig");
const g2 = @import("afd_attn_grammar.zig"); // structured_users: TF_AFD_GRAMMAR_SERVE's grammar source on the users lanes (default off)
const grammar_abi = @import("grammar_abi"); // grammar_abi's loader (-Dgram2abi) or the stub
const sd = @import("afd_serve_drafts.zig"); // user_drafts: the single-user path's drafted reach (TF_GLM_DRAFT_REACH)
const Back = users.Users(glm.afd_users.Users);

/// Completion is native State-machine evidence, never a count of log rows.
pub fn warmComplete(tokens: []const u32, reason: lanes.stream.Reason, prefills: u64, rounds: u64, forwards: u64, exchanges: u64, moe_layers: usize, failed: bool) !void {
    if (failed or reason != .length or tokens.len != 16 or prefills != 1 or rounds != 15 or forwards != 16 or moe_layers == 0 or exchanges != rounds * moe_layers) return error.WarmIncomplete;
}

pub fn extra(gpa: std.mem.Allocator, m: *const glm.model.Model, capacity: usize, max_rows: usize, count: u8) !glm.afd_users.Bytes {
    var w = try glm.views.rank(gpa, m);
    defer w.deinit(gpa);
    const e = try glm.engine.Engine.create(gpa, m, &w, capacity, 8, max_rows, capacity > m.denseLimit());
    defer e.destroy(gpa);
    var a = try glm.afd_attn.Afd.initWith(m, max_rows, m.mtp_layers > 0);
    var at = try glm.afd_engine.Attn.init(e, &a);
    var t = glm.trace.Trace.init(gpa);
    defer t.deinit();
    var f = glm.fwd.Fwd.recording(m, &t);
    var slots = try glm.afd_users.Users.init(gpa, &at, &f, count);
    defer slots.deinit(gpa);
    return slots.bytes();
}

/// shared_pool: the pooled users' device bytes beside the Engine: the extra States' per-request parts (`states`), the rows,
/// the pool; `detached`: slot 0's capacity-sized caches the Engine no longer holds (out of the base forecast).
pub const Pooled = struct { states: u64, rows: u64, pool: u64, detached: u64, total: u64 };

pub fn pooled(gpa: std.mem.Allocator, m: *const glm.model.Model, capacity: usize, max_rows: usize, count: u8, pool_tokens: usize) !Pooled {
    var w = try glm.views.rank(gpa, m);
    defer w.deinit(gpa);
    const e = try glm.engine.Engine.create(gpa, m, &w, capacity, 8, max_rows, capacity > m.denseLimit());
    defer e.destroy(gpa);
    const detached = try glm.afd_pool.tokenBytes(&e.st);
    var a = try glm.afd_attn.Afd.initWith(m, max_rows, m.mtp_layers > 0);
    var at = try glm.afd_engine.Attn.init(e, &a);
    var t = glm.trace.Trace.init(gpa);
    defer t.deinit();
    var f = glm.fwd.Fwd.recording(m, &t);
    var slots = try glm.afd_users.Users.init(gpa, &at, &f, count);
    defer slots.deinit(gpa);
    var pool = try glm.afd_pool.Pool.init(gpa, m, pool_tokens, e.buf.rows, e.st.index != null, capacity);
    defer pool.deinit(gpa);
    try slots.usePool(&pool);
    const b = try slots.bytes();
    const pb = try pool.bytes();
    return .{ .states = b.states, .rows = b.rows, .pool = pb, .detached = detached, .total = b.total + pb };
}

/// shared_pool: the largest pool (a multiple of 256 tokens) whose buffers fit `room` bytes at `capacity`; 0 when none does.
pub fn largestPool(gpa: std.mem.Allocator, m: *const glm.model.Model, capacity: usize, room: u64) !usize {
    const B = glm.afd_pool.block;
    var lo: usize = 0; // fits (0: none)
    var hi: usize = @intCast(@min(room / B + 1, std.math.maxInt(u32))); // past: a token takes more than one byte
    while (hi - lo > 1) {
        const mid = lo + (hi - lo) / 2;
        var p = try glm.afd_pool.Pool.init(gpa, m, mid * B, 8, capacity > m.denseLimit(), capacity);
        defer p.deinit(gpa);
        if (try p.bytes() <= room) lo = mid else hi = mid;
    }
    return lo * B;
}

/// user_drafts: the users' MTP backlogs beside them (TF_AFD_USERS_DRAFTS=1: a bf16 [rows, hidden] block a slot); 0 off.
pub fn draftBytes(m: *const glm.model.Model, count: u8, on: bool) u64 {
    if (!on) return 0;
    return @as(u64, count) * @as(u64, glm.afd_users_drafts.rows) * @as(u64, m.hidden) * 2;
}

/// user_drafts: a users forecast (extra's or pooled's) with the backlogs in its `rows` and `total`, as Users.bytes() counts
/// them once useDrafts ran (the users' device block); unset: unchanged.
pub fn withDrafts(x: anytype, m: *const glm.model.Model, count: u8, on: bool) void {
    const d = draftBytes(m, count, on);
    x.rows += @intCast(d);
    x.total += @intCast(d);
}

/// shared_forward: the shared round's two stages beside the users (TF_AFD_USERS_SHARED=1: bf16 [count, kda width] and
/// [count, kda heads x 128], each 256-aligned in the users' block, as Users.bytes() counts them); 0 off.
pub fn sharedBytes(m: *const glm.model.Model, count: u8, on: bool) u64 {
    if (!on) return 0;
    const s = glm.afd_users_shared.Stages.init(m, count);
    return std.mem.alignForward(u64, s.proj.bytes(), 256) + std.mem.alignForward(u64, s.out.bytes(), 256);
}

/// shared_forward: a users forecast (extra's or pooled's) with the stages in its `rows` and `total`; unset: unchanged.
pub fn withShared(x: anytype, m: *const glm.model.Model, count: u8, on: bool) void {
    const d = sharedBytes(m, count, on);
    x.rows += @intCast(d);
    x.total += @intCast(d);
}

pub fn withPingPong(x: anytype, m: *const glm.model.Model, on: bool) !void {
    const d = try glm.afd_users_pp.bytes(m, on);
    x.rows += @intCast(d);
    x.total += @intCast(d);
}

/// batched_drafts: per-slot draft hidden stash, zero bytes with the switch off.
pub fn withBatchedDrafts(x: anytype, m: *const glm.model.Model, count: u8, on: bool) void {
    const d = glm.afd_users_bd.bytes(m, count, on);
    x.rows += @intCast(d);
    x.total += @intCast(d);
}

const Host = struct {
    cfg: lanes.Config = undefined,
    clock: lanes.backend.WallClock = undefined,
    core: lanes.Engine = undefined,
    host: api.LaneHost = undefined,
    policy: ?glm.afd_users_round.Interleave = null, // prefill_interleave: TF_AFD_USERS_INTERLEAVE's R:C (null: off)
    lone: ?api.Lone = null, // user_drafts: the users backend's lone driver (TF_AFD_USERS_DRAFTS=1; null: off)
    pub fn start(h: *Host, gpa: std.mem.Allocator, io: std.Io, b: *Back, info: api.Info, words: *sl.Words, structured: ?api.LaneHost.Structured) !void {
        h.cfg = try lanes.Config.init(gpa, b.facts(), b.windowRows(), b.windowRows() - 1); // shared_verify: 4 rows, 3 drafts (off: 1, 0)
        errdefer h.cfg.deinit(gpa);
        h.clock = .{ .io = io };
        h.core = lanes.Engine.init(gpa, &h.cfg, b.backend(), h.clock.clock());
        errdefer h.core.deinit();
        h.host = api.LaneHost.init(gpa, io, &h.core, info);
        h.host.explain = .{ .ctx = words, .text = sl.Words.explain };
        h.host.structured = structured; // structured_users: each structured request's Rules (null: refused, as before)
        h.host.lone = h.lone; // user_drafts
        if (h.policy) |p| h.host.fill_policy = .{ .rounds = p.rounds, .steps = p.steps }; // prefill_interleave
        if (std.c.getenv("TENSORFOLD_MAX_QUEUED")) |v| h.host.max_queued = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.MaxQueued;
        try h.host.start();
    }
    fn stop(h: *Host, gpa: std.mem.Allocator) void {
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(gpa);
    }
};

pub fn run(gpa: std.mem.Allocator, io: std.Io, f: cli.Flags, on: served.On) !usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const args = on.plan.args;
    const count = on.plan.users;
    var slots = try glm.afd_users.Users.init(gpa, on.at, on.f, count);
    defer slots.deinit(gpa);
    slots.ctx = on.ctx;
    // user_drafts (TF_AFD_USERS_DRAFTS=1): the `--drafts` policy's loop for a lone user and a backlog a slot, before attach
    const d = on.plan.drafts;
    if (on.plan.users_drafts) try slots.useDrafts(.{ .auto = d.kind == .auto, .depth = d.policy }, glm.afd_users_drafts.rows);
    // shared_forward (TF_AFD_USERS_SHARED=1): every round one forward over every row; its stages before attach
    if (on.plan.users_shared and !on.plan.users_shared_drafts) try slots.useShared();
    // shared_verify (TF_AFD_USERS_SHARED_DRAFTS=1): every user's drafted window in that one forward; the stages at its cap's rows
    if (on.plan.users_shared_drafts) try slots.useSharedDrafts();
    if (on.plan.users_ping_pong) try slots.usePingPong();
    if (on.plan.users_batched_drafts) try slots.useBatchedDrafts();
    // shared_pool (TF_AFD_POOL_TOKENS): every slot's caches from one pool; slot 0's left the Engine before its attach
    var pool: ?glm.afd_pool.Pool = if (on.plan.pool > 0) try glm.afd_pool.Pool.init(gpa, on.at.e.m, on.plan.pool, on.at.e.buf.rows, on.at.e.st.index != null, on.at.e.st.capacity) else null;
    defer if (pool) |*p| p.deinit(gpa);
    if (pool) |*p| {
        if (on.at.e.st.kc.len > 0 and on.at.e.st.kc[0].shape[0] != 0) return error.PoolSlotZeroAttached;
        try slots.usePool(p);
    }
    if (on.ctx) |ctx| try slots.attach(ctx.d);
    if (pool) |*p| if (on.ctx) |ctx| try p.attach(ctx.d);
    const memory = try slots.bytes();
    std.log.info("concurrent_users allocation users={d} extra_states={d} row_workspace={d} context={d} MTP_loaded={} graphs=off drafts=off hardware_equality=UNKNOWN", .{ count, memory.states, memory.rows, on.plan.window, on.at.e.m.mtp_layers > 0 });
    if (pool) |*p| std.log.info("shared_pool allocation pool_tokens={d} blocks={d} pool_bytes={d} per_request_states={d} context={d}", .{ p.tokens(), p.blocks, try p.bytes(), memory.states, on.plan.window });
    if (on.plan.users_shared) std.log.info("shared_forward allocation users={d} shared_proj_bytes={d} shared_out_bytes={d} stage_bytes={d}", .{ count, slots.stages.?.proj.bytes(), slots.stages.?.out.bytes(), sharedBytes(on.at.e.m, on.plan.stageRows(), true) });
    if (on.plan.users_shared_drafts) std.log.info("shared_verify allocation users={d} stage_rows={d} stage_bytes={d} window_rows={d} max_rows={d}", .{ count, glm.afd_users_dsv.max_rows, sharedBytes(on.at.e.m, on.plan.stageRows(), true), glm.afd_users_dsv.max_width, glm.afd_users_dsv.max_rows });
    // Warm only one owned slot, then reuse the ordinary backend generation counter.
    var back: Back = .{ .steps = &slots, .count = count, .capacity = on.at.e.st.capacity, .reach = sl.reachOf(on.at.e.st.capacity, on.at.e.m.denseLimit()), .sampled = on.plan.sampled };
    if (on.plan.users_drafts) { // user_drafts: a drafted stream's reach, as the single-user path's (TF_GLM_DRAFT_REACH)
        back.drafts = true;
        back.draft_reach = try sd.reachOf(on.at.e.m.denseLimit(), back.reach);
        std.log.info("user_drafts allocation users={d} backlog_rows={d} backlog_bytes={d} drafts={s} drafted_reach={d} mode=A", .{ count, glm.afd_users_drafts.rows, draftBytes(on.at.e.m, count, true), d.spec, back.draft_reach });
    }
    back.shared_drafts = on.plan.users_shared_drafts; // shared_verify: before the warm-up (its one-row rounds take the drafted forward)
    back.ping_pong = on.plan.users_ping_pong;
    if (on.plan.users_ping_pong) std.log.info("ping_pong allocation users={d} copy_bytes={d} micro_rows=8 inflight=2", .{ count, try slots.ping_pong.?.c.bytes() });
    back.batched_drafts = on.plan.users_batched_drafts;
    var ids: [96]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(1000 + i);
    var warm = try lanes.Stream.init(gpa, .{ .id = "warm", .prompt = &ids, .max_new = 16, .drafts = false });
    defer warm.deinit(gpa);
    var cfg = try lanes.Config.init(gpa, back.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var core = lanes.Engine.init(gpa, &cfg, back.backend(), clock.clock());
    defer core.deinit();
    const forwards_before = on.at.forwards;
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    try core.addStream(&warm);
    while (core.activeCount() > 0) try core.step();
    var moe_layers: usize = 0;
    for (on.at.e.w.layers) |layer| if (layer.mlp == null) {
        moe_layers += 1;
    };
    try warmComplete(warm.emitted(), warm.reason, slots.prefills, slots.rounds, on.at.forwards - forwards_before, slots.exchanges, moe_layers, on.at.failure() != null);
    const warm_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
    std.debug.print("INFO attn serve: warm-up: {d} prompt tokens and {d} greedy tokens through both halves in {d:.0} ms: {any}\n", .{ ids.len, warm.emitted().len, warm_ms, warm.emitted() });
    std.debug.print("INFO concurrent_users warm-complete users={d} prompt={d} tokens={d} prefills={d} rounds={d} forwards={d} exchanges={d} moe_layers={d} reason={s} drafts=0 graphs=off\n", .{ count, ids.len, warm.emitted().len, slots.prefills, slots.rounds, on.at.forwards - forwards_before, slots.exchanges, moe_layers, @tagName(warm.reason) });
    server.log.init(io, false);
    var problem: []const u8 = "";
    const text = server.hf_text.HfText.load(gpa, io, f.model, a, &problem) catch |err| {
        std.debug.print("INFO concurrent_users tokenizer: {s}\n", .{if (problem.len > 0) problem else @errorName(err)});
        return err;
    };
    defer text.deinit();
    const env = cli.environ;
    var glm_text: server.glm_text.GlmText = .{ .inner = text.text(), .clear = server.glm_text.clearThinking(if (env) |m| m.get("TF_GLM_CLEAR_THINKING") else null) catch return error.ClearThinking };
    var words: sl.Words = .{ .node = on.node };
    // structured_users: with TF_AFD_GRAMMAR_SERVE=1 (TF_AFD_GRAMMAR_MASK=1), structured requests on every users lane: one grammar
    // source for this server, each request's Rules (its own matcher) from it; closed after the lane host stopped
    var grammar_src: ?g2.Served = if (on.grammar) |env_pair| g2.Served.init(gpa, io, env_pair, on.at.e.m.vocab, if (comptime g2.has_abi) &grammar_abi.trusted.global else {}) else null;
    defer if (grammar_src) |*g| g.close() catch |e| std.debug.print("INFO concurrent_users serve: structured_users: the grammar compiler stays open: {s} (a matcher still lives)\n", .{@errorName(e)});
    var host: Host = .{ .policy = on.plan.interleave, .lone = if (on.plan.users_drafts) back.lone(gpa) else null };
    if (on.plan.interleave) |il| { // prefill_interleave: after the warm-up (a whole pass, as before), the served backend fills a step a turn
        back.interleave = true;
        std.debug.print("INFO prefill_interleave interleave={d}:{d}: a new prompt goes in a step of the served pass at a time (its chunk boundaries and launches unchanged), {d} decode round(s) of the live users between turns of {d} step(s)\n", .{ il.rounds, il.steps, il.rounds, il.steps });
    }
    // four_user_profile join: shared_pool's startup line with the pool (else concurrent_users's), each followed by structured_users's grammar note when served,
    // then prefill_interleave's interleave note when on
    const structured_users_note: []const u8 = if (grammar_src != null) "; structured_users structured output on every lane (TF_AFD_GRAMMAR_SERVE=1: a matcher per request, each row under its own mask)" else "";
    const startup = if (pool) |*p| try std.fmt.allocPrint(a, "shared_pool experimental users {d}: one shared cache pool of {d} tokens ({d} blocks of 256: a contiguous region per request, first fit, compaction; a request waits for room), serial owned prefill, layer-major R=C AFD MoE, greedy/drafts-off/graphs-off; window {d}, pool {d}B, per-request states {d}B rows {d}B; hardware equality UNKNOWN{s}", .{ count, p.tokens(), p.blocks, on.plan.window, try p.bytes(), memory.states, memory.rows, structured_users_note }) else try std.fmt.allocPrint(a, "concurrent_users experimental users {d}: independent State/cache slots, serial owned prefill, layer-major R=C AFD MoE, greedy/drafts-off/graphs-off; window {d}, extra states {d}B rows {d}B; hardware equality UNKNOWN{s}", .{ count, on.plan.window, memory.states, memory.rows, structured_users_note });
    const told0 = if (on.plan.interleave) |il| try std.fmt.allocPrint(a, "{s}; prefill_interleave interleave {d}:{d} (a prompt a step of the served pass at a time between the users' rounds)", .{ startup, il.rounds, il.steps }) else startup;
    // user_sampling: TF_AFD_USERS_SAMPLING=1's note last (nothing when off: today's line)
    const told1 = if (on.plan.sampled) try std.fmt.allocPrint(a, "{s}; user_sampling sampled requests on every lane (TF_AFD_USERS_SAMPLING=1: each row qualification_45c10bcf's keyed draw under its own request's seed and token position, after its own mask)", .{told0}) else told0;
    if (on.plan.sampled) std.debug.print("INFO user_sampling users sampling=on: a sampled request is served on every lane, each row drawn by qualification_45c10bcf's keyed draw under its own request's seed and token position (never slot, row or round); greedy rows unchanged; drafts off\n", .{});
    // user_drafts: the switch's note last (unset: the line as before)
    const told = if (on.plan.users_drafts) try std.fmt.allocPrint(a, "{s}; user_drafts drafts {s} while one user decodes alone (TF_AFD_USERS_DRAFTS=1, mode A; drafted reach {d}; shared rounds undrafted)", .{ told1, d.spec, back.draft_reach }) else told1;
    // shared_forward: the switch's note last (unset: the line as before)
    const told2 = if (on.plan.users_shared) try std.fmt.allocPrint(a, "{s}; shared_forward one forward over every user's rows (TF_AFD_USERS_SHARED=1: row-by-row ops once a round, each user's KDA chain and DSA attention on its own State; undrafted)", .{told}) else told;
    // draft_shared: with user_drafts's drafts too, shared_forward's INFO line says so (shared_forward alone: its line as before); shared_verify: drafted
    const shared_drafts: []const u8 = if (on.plan.users_shared_drafts) "drafted rounds, every user's verify window in the one forward (shared_verify: TF_AFD_USERS_SHARED_DRAFTS=1)" else if (on.plan.users_drafts) "undrafted rounds, a lone user drafts (draft_shared: TF_AFD_USERS_DRAFTS=1)" else "drafts off";
    if (on.plan.users_shared) std.debug.print("INFO shared_forward users shared=on: each round one forward over every user's rows (row-by-row ops once, each user's KDA chain and DSA attention on its own State, one R=C exchange a MoE layer); {s}; stages {d} B\n", .{ shared_drafts, sharedBytes(on.at.e.m, on.plan.stageRows(), true) });
    // shared_verify: the switch's note last (unset: the line as before)
    const told3 = if (on.plan.users_shared_drafts) try std.fmt.allocPrint(a, "{s}; shared_verify drafted shared verify (TF_AFD_USERS_SHARED_DRAFTS=1: every user's pending row and up to {d} MTP drafts in that one forward of at most {d} rows, each kept and drafted on its own State)", .{ told2, glm.afd_users_dsv.max_drafts, glm.afd_users_dsv.max_rows }) else told2;
    if (on.plan.users_shared_drafts) std.debug.print("INFO shared_verify users shared drafts=on: every user's window (pending row + up to {d} drafts) verified in one forward of at most {d} rows, one exchange of all rows a MoE layer; acceptance, rollback and MTP drafts per user on its own State; stages {d} B\n", .{ glm.afd_users_dsv.max_drafts, glm.afd_users_dsv.max_rows, sharedBytes(on.at.e.m, on.plan.stageRows(), true) });
    const told4 = if (on.plan.users_batched_drafts) try std.fmt.allocPrint(a, "{s}; batched_drafts batched MTP drafts (TF_AFD_USERS_BATCHED_DRAFTS=1: one forward per active depth, state and backlog per user)", .{told3}) else told3;
    if (on.plan.users_batched_drafts) std.log.info("batched_drafts allocation users={d} hidden_bytes={d} mtp_rows={d}", .{ count, glm.afd_users_bd.bytes(on.at.e.m, count, true), glm.afd_users_bd.max_rows });
    const admit: ?api.Admit = if (pool) |*p| .{ .ctx = p, .fits = glm.afd_pool.Pool.admitFits } else null;
    try host.start(gpa, io, &back, .{ .lanes = count, .context_window = @intCast(on.plan.window), .drafts = if (on.plan.users_drafts) d.spec else "off", .startup = told4, .greedy_users = true, .users_sampled = on.plan.sampled, .users_drafts = on.plan.users_drafts, .structures = grammar_src != null, .admit = admit }, &words, if (grammar_src) |*g| g.structured() else null);
    var up = true;
    defer if (up) host.stop(gpa);
    var mon: sl.Monitor = .{ .node = undefined, .engine = host.host.engine(), .grace_ns = @intFromFloat((f.give_up_s + 5) * std.time.ns_per_s), .stop = served.stopUsersServer };
    if (on.node) |node| {
        mon.node = node;
        try mon.start();
    }
    defer mon.finish();
    served.setUsersMonitor(&mon, on.node);
    defer served.setUsersMonitor(null, null);
    // structured_entry: the single-user entry's LaneHost memory source (device counts for /metrics and the status route)
    if (on.ctx != null) host.host.memory = .{ .read = served.usersReadMemory };
    // Explicit startup temperature 0 overrides model generation defaults, never an implicit sampling downgrade.
    const sampling = try server.json.newObject(a);
    try sampling.put(a, "temperature", .{ .float = 0 });
    // user_sampling: with TF_AFD_USERS_SAMPLING=1 the single-user entry's defaults (generation_config's, TF_GLM_SAMPLING's
    // temperature 0, the serve flags), so a sampled request resolves to the same seed and parameters in both entries;
    // the default stays greedy (plan() refuses --temperature other than 0 in users mode)
    const defaults: ?server.json.Value = if (on.plan.sampled) try served.sampling(a, io, f.model, args) else .{ .object = sampling };
    // structured_entry (qualification_f245fdbd's 400): the Setup fields the single-user entry passes (afd_attn_serve.run): the checkpoint
    // directory (`.model_dir`: the grammar compiler's tokenizer.json and config.json) and the load's start (`.started`)
    const started = std.Io.Clock.awake.now(io).toNanoseconds() - @as(i96, @intFromFloat(on.load_ms * 1e6));
    const code = server.serve.run(gpa, io, args, .{ .engine = host.host.engine(), .text = glm_text.text(), .served = server.hub.servedName(args.name, args.model, f.model), .model_dir = f.model, .sampling = defaults, .environ = cli.environ, .grammars = if (grammar_src) |*g| g.source() else null, .started = started, .on_listen = served.usersOnListen });
    mon.finish();
    host.stop(gpa);
    up = false;
    if (on.at.failure() != null) return error.ExpertFailed;
    if (code != 0) return error.ServeFailed;
    return @intCast(slots.prefills -| 1);
}
