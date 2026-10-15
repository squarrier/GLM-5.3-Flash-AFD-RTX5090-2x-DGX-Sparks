//! The engine a server drives (``Engine``), and ``LaneHost``: the lane core served on one thread, rounds left to it.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// The server's name for one request, unique while the process lives.
pub const Id = u64;

/// One request's sampling (Python's exact_sampling.Sampling); a request without one decodes greedily.
pub const Sampling = lanes.Sampling;

/// A prompt's reproducible seed (``seed_for``), for requests that name none.
pub const seedFor = lanes.sampling.seedFor;

/// The request's stop strings, checked on the engine's thread after each committed token (the server decodes the tail).
pub const Stop = struct {
    ctx: *anyopaque,
    check: *const fn (ctx: *anyopaque, emitted: []const u32) bool,
};

/// tool_choice "required" or a named function: outside a think block the answer opens a call to an offered tool.
pub const CallGate = struct {
    /// The token that opens a call, the template text before the name, and the mark that ends the name.
    opener: u32,
    lead: []const u8 = "",
    tail: []const u8 = "",
    /// Offered names (empty: any); the gate completes a name its prefix starts.
    names: []const []const u8 = &.{},
    think_open: ?u32 = null,
    think_end: ?u32 = null,
};

/// response_format and the guided_* fields: every token keeps the reply inside this grammar.
pub const Structure = struct {
    kind: enum { json, json_schema, regex, choice, grammar },
    /// The schema (JSON text), regex, choices (a JSON array) or EBNF; empty for any JSON object.
    text: []const u8 = "",
    /// With thinking on, the grammar starts after this token.
    after: ?u32 = null,
    /// Reserved descriptor only; no compiler implementation or structured capability yet.
    compiler: ?*Compiler = null,
    pub const Compiler = opaque {};
    /// grammar_server: the role that owns the structured-output compiler (the AFD attention role with TF_AFD_GRAMMAR_SERVE=1);
    /// the server opens it on its first structured request (release server.zig:204-235) and checks each grammar once.
    pub const Source = struct {
        ctx: *anyopaque,
        /// tokenizer.json's text, config.json's vocab_size, the eos ids: the compiler, or null with `problem` set.
        open: *const fn (ctx: *anyopaque, tokenizer_json: []const u8, vocab: u32, stops: []const u32, problem: *[]const u8) ?*Compiler,
        /// Compiles `kind` (the enum's value: release grammar.zig:56's numbering) over `text` and frees it; false with `err` set.
        check: *const fn (ctx: *anyopaque, c: *Compiler, kind: u32, text: []const u8, err: *[512]u8) bool,
    };
};

/// A reply to decode. The request and every slice in it stay valid until its ``finished`` event.
pub const Request = struct {
    prompt: []const u32,
    max_tokens: u32,
    sampling: ?Sampling = null,
    /// Tokens that end the reply as ``stop`` (the token is delivered); empty when the request ignores EOS.
    eos: []const u32 = &.{},
    stop: ?Stop = null,
    /// false: one token a round and no drafts, the serial reference drafted replies must equal.
    drafts: bool = true,
    /// Background work yields to foreground arrivals.
    background: bool = false,
    /// Prompt prefix lengths worth keeping for a later turn: the rendered history, then shared system blocks.
    history_len: u32 = 0,
    shared_prefixes: []const u32 = &.{},
    /// Where the prompt's prefill chunks start after 0 (Python's PrefillPlan at ``Info.prefill_step``); empty: the engine's own.
    chunks: []const u32 = &.{},
    /// Most reply tokens inside a think block before the engine writes ``think_close`` (0: no limit).
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: ?u32 = null,
    /// The offered tools as JSON text, for drafters that propose a call's structure; empty: no tools.
    tools_json: []const u8 = "",
    /// Set only when ``Info`` says the engine enforces it.
    call: ?CallGate = null,
    structure: ?Structure = null,
    /// Stop a short exact cycle while the think block is open.
    loop_guard: bool = false,
};

pub const Reason = enum { stop, length, cancelled, failed };

/// What a finished reply's rounds did, for its runtime fields and /metrics.
pub const Stats = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    /// The narrowest verify window of the reply's rounds (drafted replies: 2 or more).
    min_rows: u32 = 0,
    prefill_widths: []const u32 = &.{},
    prefill_raised: []const bool = &.{},
    /// The loop period that ended or interrupted the think block, if any.
    loop_period: ?u32 = null,
    /// Prompt prefill duration measured by the engine host, in seconds.
    prefill_seconds: ?f64 = null,
    /// The drafter's own counters as a JSON object, or empty.
    telemetry_json: []const u8 = "",
};

pub const Event = union(enum) {
    /// The prompt is in the cache: ``cached`` of its tokens came from a kept prefix.
    prefilled: u32,
    /// Tokens committed for this request, in order; valid only during the call.
    tokens: []const u32,
    /// The reply ended; nothing follows. ``message`` says why a failed reply failed.
    finished: struct { reason: Reason, stats: Stats = .{}, message: []const u8 = "" },
};

/// Where a request's events go. Called on the engine's thread: it must return at once and not re-enter the engine.
pub const Sink = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, id: Id, event: *const Event) void,
};

/// What the engine serves, fixed once it has loaded.
pub const Info = struct {
    name: []const u8 = "lanes",
    lanes: u32 = 1,
    greedy_users: bool = false, // opt-in concurrent_users: refuse requested sampling/drafts instead of silently downgrading
    /// user_sampling (TF_AFD_USERS_SAMPLING=1): with greedy_users, a sampled request is served (drafts stay refused)
    users_sampled: bool = false,
    /// user_drafts (TF_AFD_USERS_DRAFTS=1): greedy users requests draft while one decodes alone (draft:true accepted)
    users_drafts: bool = false,
    /// Prompt plus reply tokens one request may use (0: no limit).
    context_window: u32 = 0,
    context_fitted: bool = false,
    /// What the engine enforces; the server refuses requests that need more.
    call_gates: bool = false,
    /// The engine observes loop_guard requests in its lane stream.
    loop_guard: bool = false,
    structures: bool = false,
    /// Prompt rows a prefill chunk at most, for the server's chunk starts (0: the engine cuts prompts itself).
    prefill_step: u32 = 0,
    /// A line the server prints once at startup (the engine's memory plan); empty: none.
    startup: []const u8 = "",
    /// What the engine does with drafts, for the start line ("off", or the CUDA GLM's --drafts policy); empty: the
    /// server's on/off from --no-drafts.
    drafts: []const u8 = "",
    /// shared_pool: a shared cache pool's room (upstream v1.0.7's Info.admit): while other streams run, the next queued
    /// request waits until its prompt and reply fit; null: admitted whenever a lane is free.
    admit: ?Admit = null,
};

/// shared_pool: Ash Hart's TensorFold v1.0.7 api.Admit (core/engine_api.zig:157-160), as upstream.
pub const Admit = struct {
    ctx: *anyopaque,
    fits: *const fn (ctx: *anyopaque, prompt: u32, max_tokens: u32) bool,
};

/// A checkpoint family an engine reads: its config ``model_type`` and weight formats, as gate entries name them.
pub const Family = struct { model_type: []const u8, formats: []const []const u8 };

/// What a server asks of the engine it opens: the checkpoint, and the serve flags an engine reads.
pub const Open = struct {
    dir: []const u8,
    model_type: []const u8,
    context: ?i64 = null,
    lanes: u32 = 8,
    /// --parallel named a number: an engine whose memory fits fewer streams refuses instead of serving fewer.
    lanes_fixed: bool = false,
    drafts: bool = true,
    speed_up: ?[]const u8 = null,
    prompt_cache_gib: ?f64 = null,
    prompt_cache_over_cap: bool = false,
    /// --learn: where shared prompt states are kept on disk for later sessions and servers (null: off).
    learn: ?[]const u8 = null,
    /// --learn-gib: what learned states may take on disk, every model and build together.
    learn_gib: f64 = 32,
    /// --device and --segments (CUDA); null: the backend's environment fallback, then its default.
    device: ?u32 = null,
    segments: ?u32 = null,
    /// --drafts: the CUDA GLM's MTP draft policy (0|N|fN|a|a:LOW:HIGH|cN:P|auto); null: its default, 0 (off).
    draft_policy: ?[]const u8 = null,
};

/// An opened engine; ``close`` stops its thread and frees its backend. ``follow``: a follower rank (the CUDA GLM's rank 1)
/// has no engine to serve; main runs follow(ctx) in the server's place, opens no port, and exits with its code.
pub const Opened = struct { engine: Engine, close: *const fn (ctx: *anyopaque) void, ctx: *anyopaque, follow: ?*const fn (ctx: *anyopaque) u8 = null };

pub const Memory = struct { active: u64 = 0, cache: u64 = 0, peak: u64 = 0 };

/// A backend's words for its own refusals (a request it cannot serve); null: the error's name.
pub const Explain = struct {
    ctx: ?*anyopaque = null,
    text: *const fn (ctx: ?*anyopaque, err: anyerror) ?[]const u8,
};

/// A backend's own memory counts for LaneHost; read from HTTP threads, so it must not touch the GPU.
pub const MemorySource = struct {
    ctx: ?*anyopaque = null,
    read: *const fn (ctx: ?*anyopaque, reset_peak: bool) ?Memory,
};

pub const Status = struct {
    /// Requests in prefill or decode, and those waiting for a lane.
    running: u32 = 0,
    waiting: u32 = 0,
    decode_tokens_per_second: f64 = 0,
    prefill_tokens_per_second: f64 = 0,
    /// Requests that gave a lane up to a later one; null when the engine does not count them.
    preemptions: ?u64 = null,
    warming: bool = false,
    /// Live streams written to the caller's buffer: tokens each holds.
    streams: usize = 0,
    /// Generated tokens held by live streams, excluding their prompts.
    generation_tokens: u64 = 0,
};

pub const SubmitError = error{ Closed, Busy };

pub const Engine = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        info: *const fn (ctx: *anyopaque) Info,
        submit: *const fn (ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void,
        /// Ends a request between rounds; its ``finished`` (cancelled) still arrives. Unknown ids are ignored.
        cancel: *const fn (ctx: *anyopaque, id: Id) void,
        status: *const fn (ctx: *anyopaque, out: *Status, stream_tokens: []u32) void,
        /// Device memory in bytes, the peak since the last reset; null when the backend keeps no count.
        memory: *const fn (ctx: *anyopaque, reset_peak: bool) ?Memory,
        /// The engine's queue as a keepalive target, or null when the backend is not Metal.
        keepalive: ?*const fn (ctx: *anyopaque) ?keepalive.Target = null,
        /// One logit per label, and the vocabulary logsumexp. Null until a family scores decisions.
        score: ?*const fn (ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64) error{Failed}!f64 = null,
        /// Whether a request would be taken now, so a full server answers 429 before a stream's 200 (Mia's 0095); null: always.
        admit: ?*const fn (ctx: *anyopaque, background: bool) bool = null,
    };

    pub fn admits(e: Engine, background: bool) bool {
        const f = e.vtable.admit orelse return true;
        return f(e.ctx, background);
    }

    pub fn info(e: Engine) Info {
        return e.vtable.info(e.ctx);
    }
    pub fn submit(e: Engine, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        return e.vtable.submit(e.ctx, id, request, sink);
    }
    pub fn cancel(e: Engine, id: Id) void {
        e.vtable.cancel(e.ctx, id);
    }
    pub fn status(e: Engine, out: *Status, stream_tokens: []u32) void {
        e.vtable.status(e.ctx, out, stream_tokens);
    }
    pub fn memory(e: Engine, reset_peak: bool) ?Memory {
        return e.vtable.memory(e.ctx, reset_peak);
    }
    /// The engine's queue as a keepalive target, or null when the backend is not Metal.
    pub fn keepaliveTarget(e: Engine) ?keepalive.Target {
        const f = e.vtable.keepalive orelse return null;
        return f(e.ctx);
    }

    /// The decision hook, or Unsupported when this engine does not score.
    pub fn score(e: Engine, prompt: []const u32, labels: []const u32, logits: []f64) error{ Failed, Unsupported }!f64 {
        const f = e.vtable.score orelse return error.Unsupported;
        return f(e.ctx, prompt, labels, logits);
    }
};

/// A backend's own driver for a lone greedy stream (the GPU round), which LaneHost runs while the stream is alone.
pub const Lone = struct {
    ctx: *anyopaque,
    sampled: bool = false, // it drives sampled streams too
    /// Prefills `s` and decodes it until it finishes (false) or `hooks.yield` hands it to the lane core (true).
    run: *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: LoneHooks) anyerror!bool,
    /// user_drafts: the lane core decodes the streams this driver may draft undrafted (Stream.lone_drafts, `drafts` off).
    core_undrafted: bool = false,
    /// shared_verify: the lane core drafts exactly the streams this driver may draft (`drafts` = lone_drafts; its shared rounds
    /// verify every stream's drafts in one forward); false: core_undrafted's rule.
    core_drafts: bool = false,
    /// user_drafts: takes back a stream the lane core decoded (it alone is admitted, nothing queued, filling or cancelling)
    /// and decodes it as `run` does after its prompt pass; null: a stream handed over stays in the lane core (upstream).
    resume_lone: ?*const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: LoneHooks) anyerror!bool = null,
    /// user_drafts: `s` can be taken back now (the backend's state for it carried to its pending token).
    resumable: ?*const fn (ctx: *anyopaque, s: *lanes.Stream) bool = null,
};

/// What a lone driver tells the host between rounds: tokens landed; and asks: hand the stream over now?
pub const LoneHooks = struct {
    ctx: *anyopaque,
    committed: *const fn (ctx: *anyopaque) void,
    yield: *const fn (ctx: *anyopaque) bool,
};

/// The lane core served to the HTTP threads (lane_host.zig).
pub const LaneHost = @import("lane_host.zig").LaneHost;
pub const TestHostFixture = if (@import("builtin").is_test) @import("grammar_tests_c_host_test.zig") else struct {};
pub const TestHostTransitions = if (@import("builtin").is_test) @import("grammar_tests_c_host_fixture.zig") else struct {};
pub const TestRequestLease = if (@import("builtin").is_test) @import("grammar_tests_c3_request_lease.zig") else struct {};

/// Exact prompt reuse between requests, for any family (prompt_cache.zig).
pub const prompt_cache = @import("prompt_cache.zig");

/// Learned prompt-cache states on disk (prompt_imprint.zig).
pub const prompt_imprint = @import("prompt_imprint.zig");

/// The idle keepalive's ticker and target contract.
pub const keepalive = @import("keepalive.zig");

test "grammar_tests_c3 C10 actual queued stop and owned RequestLease compiler borrowing" {
    try @import("grammar_tests_c3_request_lease.zig").run();
}

test "grammar_tests_c C10 C11 actual host exhaustive allocation ownership shutdown" {
    try @import("grammar_tests_c_host_test.zig").allocations();
}

test "grammar_tests_c C10 C11 actual host cancel stop failAll with exhausted allocator" {
    try @import("grammar_tests_c_host_drains.zig").drains();
}

test "grammar_tests_c C10 actual host five kinds and same backend plain reuse" {
    try @import("grammar_tests_c_host_drains.zig").kindsAndReuse();
}

test "grammar_tests_c C10 actual submit admission compiler transfer request lifetime" {
    try @import("grammar_tests_c_host_test.zig").run();
}

test {
    _ = @import("lane_host.zig");
    _ = @import("lane_host_reuse_test.zig");
    _ = @import("prompt_cache.zig");
    _ = @import("prompt_imprint.zig");
    _ = @import("keepalive.zig");
}
