//! GLM-5.3-Flash on the native CUDA server (qualification_qualification_44ba554e.8): the family `capabilities` lists, its lane backend over the GPU's serve
//! calls, and its open (day 2). The CUDA GLM runs as two TP ranks on two Sparks, each a `tensorfold-native serve` with
//! TF_TP_RANK and TF_TP_MASTER (as tf-glm-forward takes them). Each rank brings up the pair, admits by its own plan's
//! bytes, loads its weights without the MTP layer (drafts off), and warms up (glm_cuda_rank.zig). Rank 0 then serves
//! HTTP over the lane core, its mirror bound to the pair's link; rank 1 opens no port and replays rank 0's words
//! (glm_cuda_pair.zig). Templates: Ash's Metal GLM host and mirror (native/glm_host.zig, families/glm/mirror.zig), and
//! the serve path of Mia's Qwen Zig repo (MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold be3ed85, Apache-2.0).
const std = @import("std");
const api = @import("engine_api");
const cuda = @import("cuda");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const gl = @import("glm_cuda_lanes.zig");
const gp = @import("glm_cuda_pair.zig");
const gr = @import("glm_cuda_rank.zig");
const admit = @import("glm_cuda_admit.zig");
const gd = @import("glm_cuda_drafts.zig");
const native = @import("cuda.zig");
const Allocator = std.mem.Allocator;

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{ "exl3", "mlx-q4g64" };
/// Prompt plus reply tokens a request may use by default (--context overrides it; the checkpoint's window is 1,048,576).
pub const default_context: i64 = 262_144;
/// Prompt rows a prefill call takes at most: python's chunk (decode.prefill's).
pub const prompt_rows: u32 = glm.engine.prefill_rows;
/// The Engine's capacity over the served window (the Metal GLM host's window + 64).
pub const slack = 64;
/// The Engine's prompt chunk rows by default: qualification_e84ee1bf's runnable form (TF_GLM_PREFILL_ROWS overrides it).
pub const default_prefill_rows = 1024;

/// The GPU's serve calls, each armed on the pair's watchdog.
pub const Steps = gp.Armed(glm.serve.Serve);
/// The lane core's backend over them: one stream, greedy, drafts off.
pub const Backend = gl.Lanes(Steps);
pub const explain = gl.explain;

/// The serve settings from the environment, read before any GPU work.
pub const Env = struct {
    rank: u32,
    master: []const u8,
    prefill_rows: usize = default_prefill_rows,
    floor: f64 = admit.floor_gib,
    /// TF_GLM_REACH: the positions a sequence may reach (null: the capacity, with the DSA scratch)
    reach: ?usize = null,
    max_queued: ?u32 = null,
    fatbins: []const u8,
    /// TF_GLM_FORECAST=1: print the rank's forecast and stop (no GPU, no peer)
    forecast_only: bool = false,
    /// TF_GLM_LONG_CONTEXT=0: no index caches, and a sequence reaches the dense limit at most (tf-glm-forward without
    /// --long-context, qualification_f686be52's form); 1 (the default): the index caches, and qualification_qualification_c3ccc258's DSA scratch past the dense limit
    long_context: bool = true,
    /// `serve --drafts` (qualification_qualification_44ba554e.8 day 3; the open sets it from the flag): off by default
    drafts: gd.Drafts = .{},
    /// TF_GLM_DRAFT_REACH: the positions a drafted stream may reach (null: the dense limit; never past the reach)
    draft_reach: ?usize = null,
    /// TF_GLM_EXL3_L2D / TF_GLM_EXL3_PDL (qualification_96d80965): Mia's 0128 in the EXL3 experts, as tf-glm-forward reads them; off
    exl3_sched: glm.exl3.experts.Sched = .{},
    /// rank_fingerprint: TF_GLM_FINGERPRINT=1: the settings end ` fp=1` and the ranks agree their identities before the load; off
    fingerprint: bool = false,
};

fn number(comptime T: type, lookup: *const fn ([:0]const u8) ?[]const u8, name: [:0]const u8, problem: *[]const u8, a: Allocator) !?T {
    const t = std.mem.trim(u8, lookup(name) orelse return null, " ");
    if (t.len == 0) return null;
    return std.fmt.parseInt(T, t, 10) catch {
        problem.* = try std.fmt.allocPrint(a, "{s}={s}: a whole number", .{ name, t });
        return error.Invalid;
    };
}

/// The settings from `lookup` (the environment); null with `problem` set when one is malformed or missing.
pub fn envFrom(a: Allocator, lookup: *const fn ([:0]const u8) ?[]const u8, exe_dir: []const u8, problem: *[]const u8) !?Env {
    const rank = (number(u32, lookup, "TF_TP_RANK", problem, a) catch return null) orelse 0;
    if (rank > 1) {
        problem.* = try std.fmt.allocPrint(a, "TF_TP_RANK={d}: the CUDA GLM-5.3-Flash engine runs two ranks, 0 and 1", .{rank});
        return null;
    }
    const forecast_only = if (lookup("TF_GLM_FORECAST")) |v| std.mem.eql(u8, v, "1") else false;
    const master = lookup("TF_TP_MASTER") orelse if (forecast_only) "" else {
        problem.* = "TF_TP_MASTER is not set: the CUDA GLM-5.3-Flash engine runs as two TP ranks (rank 0's address:port, the same on both)";
        return null;
    };
    var e: Env = .{ .rank = rank, .master = master, .fatbins = lookup("TF_GLM_FATBINS") orelse try std.fs.path.join(a, &.{ exe_dir, "fatbin" }), .forecast_only = forecast_only };
    if (number(usize, lookup, "TF_GLM_PREFILL_ROWS", problem, a) catch return null) |v| e.prefill_rows = v;
    if (e.prefill_rows < 1 or e.prefill_rows > prompt_rows) {
        problem.* = try std.fmt.allocPrint(a, "TF_GLM_PREFILL_ROWS={d}: 1 to {d} prompt rows a chunk", .{ e.prefill_rows, prompt_rows });
        return null;
    }
    e.reach = number(usize, lookup, "TF_GLM_REACH", problem, a) catch return null;
    e.max_queued = number(u32, lookup, "TENSORFOLD_MAX_QUEUED", problem, a) catch return null;
    if (lookup("TF_GLM_LONG_CONTEXT")) |v| {
        const t = std.mem.trim(u8, v, " ");
        if (std.mem.eql(u8, t, "0")) e.long_context = false else if (!std.mem.eql(u8, t, "1") and t.len > 0) {
            problem.* = try std.fmt.allocPrint(a, "TF_GLM_LONG_CONTEXT={s}: 0 or 1", .{t});
            return null;
        }
    }
    e.floor = admit.floorFrom(lookup("TF_GLM_FLOOR_GIB")) orelse {
        problem.* = try std.fmt.allocPrint(a, "TF_GLM_FLOOR_GIB={s}: a number of GiB from 0 to 64", .{lookup("TF_GLM_FLOOR_GIB").?});
        return null;
    };
    e.draft_reach = number(usize, lookup, "TF_GLM_DRAFT_REACH", problem, a) catch return null;
    e.exl3_sched = glm.exl3.experts.Sched.parse(lookup("TF_GLM_EXL3_L2D"), lookup("TF_GLM_EXL3_PDL")) catch {
        problem.* = try std.fmt.allocPrint(a, "TF_GLM_EXL3_L2D={s} TF_GLM_EXL3_PDL={s}: Mia's 0128 switches take 0..64 and 0 or 1", .{ lookup("TF_GLM_EXL3_L2D") orelse "", lookup("TF_GLM_EXL3_PDL") orelse "" });
        return null;
    };
    e.fingerprint = cuda.fingerprint.parse(lookup(cuda.fingerprint.name)) catch {
        problem.* = try std.fmt.allocPrint(a, cuda.fingerprint.name ++ "={s}: 1 or 0 (or unset)", .{lookup(cuda.fingerprint.name).?});
        return null;
    };
    return e;
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The positions a sequence may reach: the capacity when the Engine reserves its DSA scratch (a long context, qualification_qualification_c3ccc258),
/// else the dense limit; never past TF_GLM_REACH.
pub fn reachOf(capacity: usize, dense_limit: usize, long_context: bool, cap: ?usize) usize {
    const scratch = long_context and glm.prompt.DsaScratch != void;
    const r = if (scratch) capacity else @min(capacity, dense_limit);
    return if (cap) |c| @min(r, c) else r;
}

/// One rank's family state: the GPU rank, and on rank 0 the mirror and the lane backend over the armed serve calls.
pub const Glm = struct {
    gpa: Allocator,
    rank: gr.Rank,
    env: Env,
    sender: gp.Sender = undefined,
    steps: Steps = undefined,
    back: Backend = undefined,
    followed: gp.Followed = .{},
    /// rank 1: the driver and context it owns (rank 0's are the lane host's)
    own: ?*native.Gpu = null,
    startup: []u8 = &.{},
    /// rank 0 with drafts on: the lane host's driver for a lone drafted stream
    lone: gd.Lone(Steps) = undefined,

    fn deinit(p: *anyopaque) void {
        const g: *Glm = @ptrCast(@alignCast(p));
        if (g.env.rank == 0) gp.stop(&g.rank.pair) else gp.closeFollower(&g.rank.pair);
        g.rank.deinit();
        if (g.own) |o| {
            o.ctx.deinit();
            o.driver.close();
            g.gpa.destroy(o);
        }
        g.gpa.free(g.startup);
        g.gpa.destroy(g);
    }

    /// native/cuda.zig's LoneRun: a lone drafted stream on rank 0 (`yield` unused: one stream at a time).
    fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, _: *const fn (*anyopaque) bool) anyerror!bool {
        const g: *Glm = @ptrCast(@alignCast(p));
        return g.lone.run(s, hooks, committed);
    }

    /// Rank 1's serve: rank 0's rounds replayed until its stop; 0 then, the watchdog's 3 on a lost peer.
    fn follow(p: *anyopaque) u8 {
        const g: *Glm = @ptrCast(@alignCast(p));
        cuda.watch.installStop();
        std.debug.print("INFO rank 1: following rank 0 ({s}); no HTTP port\n", .{g.env.master});
        gp.follow(glm.serve.Serve, &g.rank.serve, &g.rank.pair, g.gpa, &g.followed, g.env.drafts) catch |e| {
            std.debug.print("FAIL rank 1: following rank 0: {s}\n", .{@errorName(e)});
            return cuda.watch.lost_exit_code;
        };
        const d = g.followed;
        var buf: [160]u8 = undefined;
        const drafted = if (d.decodes > 0) std.fmt.bufPrint(&buf, "; {d} drafted decodes ({d} draft, {d} window, {d} keep)", .{ d.decodes, d.drafts, d.windows, d.keeps }) catch "" else "";
        std.debug.print("PASS rank 1: rank 0 stopped the pair: {d} rounds replayed ({d} begin, {d} step, {d} release){s}\n", .{ d.rounds, d.begins, d.steps, d.releases, drafted });
        return 0;
    }
};

/// What native/cuda.zig serves: rank 0's lane backend, or rank 1's follower.
pub const Loaded = struct {
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    backend: ?lanes.backend.Backend,
    facts: ?lanes.Model,
    max_queued: ?u32,
    startup: []const u8,
    follow: ?*const fn (*anyopaque) u8,
    /// rank 0 with drafts on: the lone drafted stream's driver (native/cuda.zig's LoneRun, called with `ctx`)
    lone: ?*const fn (*anyopaque, *lanes.Stream, *anyopaque, *const fn (*anyopaque) void, *const fn (*anyopaque) bool) anyerror!bool = null,
    /// what the open does with drafts, for the server's start line: "off", or the policy
    drafts: []const u8 = "off",
};

/// The serve settings from this process's environment (the binary's folder holds `fatbin/` by default).
pub fn env(a: Allocator, io: std.Io, problem: *[]const u8) !?Env {
    const exe = std.process.executableDirPathAlloc(io, a) catch ".";
    return envFrom(a, getenv, exe, problem);
}

/// `serve --drafts` (qualification_qualification_44ba554e.8 day 3) as the open takes it: off without the flag, at 0 and under --no-drafts (`on` false);
/// a malformed policy is refused in words (null, `problem` set).
pub fn draftsFrom(a: Allocator, on: bool, spec: ?[]const u8, problem: *[]const u8) !?gd.Drafts {
    const d = try gd.parse(a, spec, problem) orelse return null;
    return if (on) d else gd.Drafts{};
}

/// The rank's forecast alone (TF_GLM_FORECAST=1): its plan and Engine sized from the checkpoint's headers, printed;
/// no GPU, no peer. Always null: the open stops with the forecast as its words.
pub fn forecast(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, window: i64, e: Env, problem: *[]const u8) !?Loaded {
    var r: gr.Rank = .{ .gpa = gpa, .io = io, .o = try options(a, dir, window, e, ""), .d = undefined, .ctx = undefined, .pair = .{ .gpa = gpa, .io = io, .rank = e.rank } };
    defer r.deinit();
    try r.plan();
    const f = try admit.forecast(&r.r.?, r.e.?, e.rank, e.prefill_rows, gpa);
    var buf: [1024]u8 = undefined;
    problem.* = try std.fmt.allocPrint(a, "TF_GLM_FORECAST=1: FORECAST {s} (forecast only: nothing loaded)", .{try admit.describe(f, admit.available(), e.floor, &buf)});
    return null;
}

fn options(a: Allocator, dir: []const u8, window: i64, e: Env, triton: []const u8) !gr.Options {
    return .{ .dir = try a.dupeSentinel(u8, dir, 0), .rank = e.rank, .master = e.master, .capacity = @as(usize, @intCast(window)) + slack, .prefill_rows = e.prefill_rows, .long_context = e.long_context, .triton = triton, .fatbins = e.fatbins, .floor_gib = e.floor, .drafts = e.drafts.on(), .draft_spec = e.drafts.spec, .exl3_sched = e.exl3_sched, .fingerprint = e.fingerprint };
}

/// The load's warm-up on both ranks in lockstep: drafts off, gl.warmUp's 96 ids and 16 greedy tokens over the armed
/// calls; drafts on, the same tokens by the policy's loop (glm_cuda_drafts.warmUp), its counters in the line.
fn warmUp(r: *gr.Rank, e: Env, warm: *[gl.warm_tokens]u32, line: []u8) ![]const u8 {
    if (!e.drafts.on()) {
        var armed: Steps = .{ .steps = &r.serve, .pair = &r.pair };
        try gl.warmUp(Steps, &armed, warm);
        return "greedy tokens";
    }
    const x = try gd.warmUp(glm.serve.Serve, &r.serve, &r.pair, e.drafts, warm);
    return std.fmt.bufPrint(line, "tokens drafted {s} (rounds {d}, drafted {d}, accepted {d})", .{ e.drafts.spec, x.rounds, x.drafted, x.accepted });
}

/// The rank on the GPU of `ctx`: the pair brought up, the weights loaded, the warm-up run on both ranks; then rank 0's
/// lane backend with its mirror bound to the pair's link, or rank 1's follower. `own`: rank 1's driver and context.
pub fn load(a: Allocator, gpa: Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, own: ?*native.Gpu, dir: []const u8, window: i64, triton: []const u8, e: Env, problem: *[]const u8) !?Loaded {
    const g = try gpa.create(Glm);
    g.* = .{ .gpa = gpa, .env = e, .own = own, .rank = .{ .gpa = gpa, .io = io, .o = try options(a, dir, window, e, triton), .d = d, .ctx = ctx, .pair = .{ .gpa = gpa, .io = io, .rank = e.rank } } };
    var ok = false;
    defer if (!ok) {
        g.rank.pair.close();
        g.rank.deinit();
        gpa.destroy(g);
    };
    const r = &g.rank;
    r.init() catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the CUDA GLM-5.3-Flash rank {d} cannot read its kernels (TF_GLM_FATBINS {s}, the Triton set {s}): {s}", .{ e.rank, e.fatbins, triton, @errorName(err) });
        return null;
    };
    r.up() catch |err| {
        problem.* = if (err == error.Refused) r.problem else try std.fmt.allocPrint(a, "the CUDA GLM-5.3-Flash rank {d} could not come up: {s}", .{ e.rank, @errorName(err) });
        // a lost peer during the bring-up: the watchdog's exit 3 (Pair.lost's rule)
        if (err != error.Refused) if (r.pair.watch) |*w| if (cuda.watch.Watch.isLoss(err) or w.peerGone(1000)) w.fail("the other rank is gone (this rank's bring-up failed)");
        return null;
    };
    var warm: [gl.warm_tokens]u32 = undefined;
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var line: [160]u8 = undefined;
    const how = warmUp(r, e, &warm, &line) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the CUDA GLM-5.3-Flash rank {d}'s warm-up failed: {s}", .{ e.rank, @errorName(err) });
        return null;
    };
    try r.stream.?.synchronize();
    const warm_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
    std.debug.print("INFO rank {d}: warm-up: {d} prompt tokens and {d} {s} in {d:.0} ms: {any}\n", .{ e.rank, gl.warm_prompt, gl.warm_tokens, how, warm_ms, warm });
    const cap = r.o.capacity;
    const reach = reachOf(cap, r.r.?.m.denseLimit(), r.o.long_context, e.reach);
    const drafted = gd.reachOf(e.draft_reach, r.r.?.m.denseLimit(), reach);
    var dbuf: [192]u8 = undefined;
    const dtext = if (e.drafts.on()) try std.fmt.bufPrint(&dbuf, "drafts {s} (the MTP layer loaded; a request drafts while its prompt and max_tokens stay within {d} positions)", .{ e.drafts.spec, drafted }) else "drafts off, no MTP layer";
    g.startup = try std.fmt.allocPrint(gpa, "GLM-5.3-Flash on CUDA, TP rank {d} of 2 (master {s}): greedy, {s}; capacity {d} (window {d} + {d}), long context {}, positions reached {d}, prompt chunks of {d} rows; NCCL carries every gather", .{ e.rank, e.master, dtext, cap, window, slack, r.o.long_context, reach, e.prefill_rows });
    std.debug.print("INFO {s}\n", .{g.startup});
    ok = true;
    if (e.rank == 1) return .{ .ctx = g, .deinit = Glm.deinit, .backend = null, .facts = null, .max_queued = null, .startup = g.startup, .follow = Glm.follow };
    g.sender = .{ .gpa = gpa, .pair = &r.pair };
    g.steps = .{ .steps = &r.serve, .pair = &r.pair };
    g.back = .{ .steps = &g.steps, .mirror = g.sender.mirror(), .capacity = cap, .reach = reach };
    g.lone = .{ .gpa = gpa, .back = &g.back, .d = e.drafts, .reach = drafted };
    return .{ .ctx = g, .deinit = Glm.deinit, .backend = g.back.backend(), .facts = g.back.facts(), .max_queued = e.max_queued, .startup = g.startup, .follow = null, .lone = if (e.drafts.on()) Glm.loneRun else null, .drafts = gd.describe(e.drafts) };
}

test "the window is 262,144 by default and prompts go in python's 2,048-row chunks" {
    try std.testing.expectEqual(@as(i64, 262_144), default_context);
    try std.testing.expectEqual(@as(u32, 2048), prompt_rows);
}

const Fake = struct {
    var vars: []const [2][]const u8 = &.{};
    fn lookup(name: [:0]const u8) ?[]const u8 {
        for (vars) |v| if (std.mem.eql(u8, v[0], name)) return v[1];
        return null;
    }
};

test "the rank's settings come from TF_TP_RANK and TF_TP_MASTER, as tf-glm-forward's; bad ones are refused in words" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    Fake.vars = &.{ .{ "TF_TP_RANK", "1" }, .{ "TF_TP_MASTER", "192.0.2.10:18890" }, .{ "TENSORFOLD_MAX_QUEUED", "4" } };
    const e = (try envFrom(a, Fake.lookup, "/opt/tf/bin", &problem)).?;
    try std.testing.expectEqual(@as(u32, 1), e.rank);
    try std.testing.expectEqualStrings("192.0.2.10:18890", e.master);
    try std.testing.expectEqual(@as(usize, 1024), e.prefill_rows);
    try std.testing.expectEqual(@as(?u32, 4), e.max_queued);
    try std.testing.expectEqualStrings("/opt/tf/bin/fatbin", e.fatbins);
    try std.testing.expectEqual(@as(f64, 14), e.floor);
    Fake.vars = &.{.{ "TF_TP_RANK", "0" }};
    try std.testing.expectEqual(@as(?Env, null), try envFrom(a, Fake.lookup, "/x", &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "TF_TP_MASTER is not set") != null);
    Fake.vars = &.{ .{ "TF_TP_RANK", "2" }, .{ "TF_TP_MASTER", "h:1" } };
    try std.testing.expectEqual(@as(?Env, null), try envFrom(a, Fake.lookup, "/x", &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "two ranks") != null);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_PREFILL_ROWS", "4096" } };
    try std.testing.expectEqual(@as(?Env, null), try envFrom(a, Fake.lookup, "/x", &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "1 to 2048") != null);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TENSORFOLD_MAX_QUEUED", "many" } };
    try std.testing.expectEqual(@as(?Env, null), try envFrom(a, Fake.lookup, "/x", &problem));
    try std.testing.expectEqualStrings("TENSORFOLD_MAX_QUEUED=many: a whole number", problem);
    // the long context: on by default, off by TF_GLM_LONG_CONTEXT=0 (qualification_f686be52's dense form), anything else refused
    try std.testing.expect(e.long_context);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_LONG_CONTEXT", "0" } };
    try std.testing.expect(!(try envFrom(a, Fake.lookup, "/x", &problem)).?.long_context);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_LONG_CONTEXT", "yes" } };
    try std.testing.expectEqual(@as(?Env, null), try envFrom(a, Fake.lookup, "/x", &problem));
    try std.testing.expectEqualStrings("TF_GLM_LONG_CONTEXT=yes: 0 or 1", problem);
    // TF_GLM_DRAFT_REACH (qualification_qualification_44ba554e.8 day 3): unset by default, a whole number, refused in words before anything loads
    try std.testing.expectEqual(@as(?usize, null), e.draft_reach);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_DRAFT_REACH", "3000" } };
    try std.testing.expectEqual(@as(?usize, 3000), (try envFrom(a, Fake.lookup, "/x", &problem)).?.draft_reach);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_DRAFT_REACH", "far" } };
    try std.testing.expectEqual(@as(?Env, null), try envFrom(a, Fake.lookup, "/x", &problem));
    try std.testing.expectEqualStrings("TF_GLM_DRAFT_REACH=far: a whole number", problem);
    // Mia's 0128 switches (qualification_96d80965): off unset; read as tf-glm-forward reads them; a bad one refused in words, before anything
    try std.testing.expect(!e.exl3_sched.on());
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_EXL3_L2D", "0" }, .{ "TF_GLM_EXL3_PDL", "1" } };
    try std.testing.expectEqual(glm.exl3.experts.Sched{ .pdl = true }, (try envFrom(a, Fake.lookup, "/x", &problem)).?.exl3_sched);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_EXL3_L2D", "65" } };
    try std.testing.expectEqual(@as(?Env, null), try envFrom(a, Fake.lookup, "/x", &problem));
    try std.testing.expectEqualStrings("TF_GLM_EXL3_L2D=65 TF_GLM_EXL3_PDL=: Mia's 0128 switches take 0..64 and 0 or 1", problem);
}

test "rank_fingerprint: TF_GLM_FINGERPRINT on a rank: off unset and at 0, on at 1 (into the rank's options), anything else refused in words before anything loads" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    Fake.vars = &.{.{ "TF_TP_MASTER", "h:1" }};
    const off = (try envFrom(a, Fake.lookup, "/x", &problem)).?;
    try std.testing.expect(!off.fingerprint);
    try std.testing.expect(!(try options(a, "/m", 4096, off, "/t")).fingerprint);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_FINGERPRINT", "0" } };
    try std.testing.expect(!(try envFrom(a, Fake.lookup, "/x", &problem)).?.fingerprint);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_FINGERPRINT", "1" } };
    const on = (try envFrom(a, Fake.lookup, "/x", &problem)).?;
    try std.testing.expect(on.fingerprint);
    try std.testing.expect((try options(a, "/m", 4096, on, "/t")).fingerprint);
    Fake.vars = &.{ .{ "TF_TP_MASTER", "h:1" }, .{ "TF_GLM_FINGERPRINT", "yes" } };
    try std.testing.expectEqual(@as(?Env, null), try envFrom(a, Fake.lookup, "/x", &problem));
    try std.testing.expectEqualStrings("TF_GLM_FINGERPRINT=yes: 1 or 0 (or unset)", problem);
    _ = &gr.Rank.identity; // the bring-up's identity step (a GPU rank's; compiled here)
    _ = &gr.Rank.fingerprint;
}

test "the reach: the capacity with the DSA scratch (qualification_qualification_c3ccc258) on a long context, the dense limit without, TF_GLM_REACH under both" {
    const scratch = glm.prompt.DsaScratch != void;
    try std.testing.expectEqual(@as(usize, if (scratch) 262_208 else 2051), reachOf(262_208, 2051, true, null));
    try std.testing.expectEqual(@as(usize, 2051), reachOf(262_208, 2051, false, null));
    try std.testing.expectEqual(@as(usize, 2051), reachOf(262_208, 2051, true, 2051));
    try std.testing.expectEqual(@as(usize, 1000), reachOf(1000, 2051, false, null));
}

test "the GPU's serve calls fit the lane backend, and the open's pieces compile" {
    _ = &Backend.backend;
    _ = &Backend.facts;
    _ = &load;
    _ = &forecast;
    _ = &env;
    _ = @import("glm_cuda_lanes_test.zig");
    _ = @import("glm_cuda_pair_test.zig");
    _ = @import("glm_cuda_drafts_test.zig");
    _ = @import("glm_cuda_offtrace_test.zig");
    _ = gd;
    _ = admit;
}
