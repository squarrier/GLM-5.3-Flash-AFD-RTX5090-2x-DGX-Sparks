//! GLM-5.3-Flash on CUDA behind the lane core (qualification_qualification_44ba554e.8 day 1): one stream, greedy, drafts off, over `Steps` (the GPU's
//! cuda_serve.Serve, or a test's scripted forward). As in the Metal GLM's backend (families/glm/backend.zig, Ash's), each
//! operation's words go to rank 1 before rank 0 runs it, and `replay` runs them there, so both ranks make the same calls.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
pub const mask_frame = @import("glm_cuda_mask_frame.zig");

/// What rank 1 is told before rank 0 runs it: begin (the prompt's ids), step (the token fed), release (no words); with
/// drafts on (glm_cuda_drafts.zig), decode (the pending token and the count) starts qualification_qualification_86ea5170's loop on both ranks, whose
/// draft, window and keep each carry their words.
pub const Op = enum(u32) { begin = 1, step = 2, release = 3, decode = 4, draft = 5, window = 6, keep = 7 };

/// Where an operation's words go: the pair's control link on two Sparks (qualification_qualification_44ba554e.8 day 2), a recorder in tests; none alone.
pub const StructuredDriverCpu = struct {
    open: *const fn (?*anyopaque, []const u32, lanes.structured_control.Endpoint) anyerror!void,
    operation: *const fn (?*anyopaque, lanes.structured_control.Kind, u64, []const u32, []const u32) anyerror!void,
    terminate: *const fn (?*anyopaque, lanes.structured_control.Status) void,
};

pub const Mirror = struct {
    ctx: ?*anyopaque = null,
    send: ?*const fn (ctx: ?*anyopaque, op: Op, words: []const u32) anyerror!void = null,
    chosen_cpu: ?*const fn (ctx: ?*anyopaque, token: u32) anyerror!void = null,
    structured_cpu: ?*const fn (ctx: ?*anyopaque, words: []const u32) anyerror!void = null,
    structured_driver_cpu: ?StructuredDriverCpu = null,

    pub fn tellStructured(m: Mirror, words: []const u32) !void {
        if (m.structured_cpu) |f| {
            if (!@import("builtin").is_test) return error.IncompleteMaskedAdapter;
            return f(m.ctx, words);
        }
        // a peer rank but no structured frame transport (GRAMMAP1's PREPARE/READY/COMMIT is unbuilt): refused
        if (m.send != null) return error.IncompleteMaskedAdapter;
        // no peer (grammar_mask: the AFD attention role, world 1): nothing to tell
    }

    pub fn tell(m: Mirror, op: Op, words: []const u32) !void {
        if (m.send) |f| try f(m.ctx, op, words);
    }
};

/// Rank 1's side: one operation rank 0 sent, run on its own `Steps` (the first token and each draw are the gathered
/// greedy picks, so they match rank 0's without being sent). A drafted decode's frames are glm_cuda_drafts.follow's.
pub fn replay(comptime Steps: type, steps: *Steps, op: Op, words: []const u32) !void {
    switch (op) {
        .begin => _ = try steps.prefill(words),
        .step => {
            if (words.len != 1) return error.MirrorOutOfStep;
            _ = try steps.step(words[0]);
        },
        .release => steps.release(),
        .decode, .draft, .window, .keep => return error.MirrorOutOfStep,
    }
}

/// The load's warm-up, as the Metal GLM host's open runs one: a 96-token prompt (ids 1000-1095) and 16 greedy tokens,
/// then release, so a rank pays its first calls' costs before a request. Both ranks run it in lockstep (fixed ids).
pub const warm_prompt = 96;
pub const warm_tokens = 16;

pub fn warmUp(comptime Steps: type, steps: *Steps, out: *[warm_tokens]u32) !void {
    var ids: [warm_prompt]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    out[0] = try steps.prefill(&ids);
    for (1..warm_tokens) |k| out[k] = try steps.step(out[k - 1]);
    steps.release();
}

/// The lane core's backend over `Steps`: `prefill(ids)` gives the first token, `step(token)` feeds a token and gives the
/// next (drawn and committed), `release()` ends the sequence; `len` is the rows the sequence holds.
pub fn Lanes(comptime Steps: type) type {
    return struct {
        const Self = @This();
        const masked = if (@hasDecl(Steps, "masked_supported")) Steps.masked_supported else @hasDecl(Steps, "checkMasked") and @hasDecl(Steps, "prefillToLogits") and @hasDecl(Steps, "pickFirst") and @hasDecl(Steps, "stepMasked");

        steps: *Steps,
        mirror: Mirror = .{},
        /// rows one sequence may hold (prompt and reply): the Engine's capacity
        capacity: usize,
        /// rows a sequence may reach on this server: the capacity with the DSA scratch (qualification_qualification_c3ccc258), else the dense limit 2,051
        reach: usize,
        owner: ?*Stream = null,
        first_token: u32 = 0,
        first_pending: bool = false,
        epoch: u64 = 0,
        sequence: u64 = 0,
        frame_allocator: std.mem.Allocator = std.heap.page_allocator,
        structured_session: ?*lanes.structured_control.Session = null,
        structured_poisoned: bool = false,

        pub fn backend(b: *Self) be.Backend {
            return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = first, .first_masked = if (masked) firstMasked else null, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
        }

        fn frame(b: *Self, kind: mask_frame.Kind, position: usize, tokens: []const u32, mask: []const u32) !void {
            const r = b.owner.?.rules.?;
            if (b.structured_session != null) {
                const driver = b.mirror.structured_driver_cpu orelse return error.IncompleteMaskedAdapter;
                const operation: lanes.structured_control.Kind = switch (kind) { .first => .first, .step => .step, .begin => return error.ControlShape };
                try driver.operation(b.mirror.ctx, operation, position, tokens, mask);
                return;
            }
            const words = try (mask_frame.Frame{ .kind = kind, .epoch = b.epoch, .sequence = b.sequence, .position = position, .width = r.width, .tokens = tokens, .mask = mask }).encode(b.frame_allocator);
            defer b.frame_allocator.free(words);
            try b.mirror.tellStructured(words);
            b.sequence += 1;
        }

        fn prefillStructured(b: *Self, s: *Stream, prompt: []const u32) !void {
            if (comptime !masked) return error.IncompleteMaskedAdapter;
            try b.steps.checkMasked();
            // structured_users: a sampled structured request (TF_GLM_SAMPLING=1, enabled above) samples among its grammar's tokens
            // (Attn.select: the mask before the nucleus); a control session stays greedy
            if (s.sampling != null and (b.structured_session != null or !@hasDecl(Steps, "configureSampling"))) return error.StructuredSamplingUnsupported;
            if (b.structured_session) |session| {
                if (!@import("builtin").is_test) return error.IncompleteMaskedAdapter;
                if (comptime !@hasDecl(Steps, "installStructured") or !@hasDecl(Steps, "clearStructured")) return error.IncompleteMaskedAdapter;
                const driver = b.mirror.structured_driver_cpu orelse return error.IncompleteMaskedAdapter;
                if (s.rules.?.width != b.steps.maskedWidth()) return error.MaskShape;
                const endpoint = try session.endpoint(0);
                try b.steps.installStructured(endpoint);
                if (s.cancel_check) |check| {
                    session.cancel_ctx = check.ptr;
                    session.cancel_check = check.check;
                }
                b.owner = s;
                s.backend_lease = .entered;
                b.epoch += 1;
                b.sequence = 0;
                b.first_token = 0;
                b.first_pending = false;
                try driver.open(b.mirror.ctx, prompt, endpoint);
                try b.steps.prefillToLogits(prompt);
                s.cached = 0;
                b.first_pending = true;
                return;
            }
            if (b.mirror.send != null and b.mirror.structured_cpu == null) return error.IncompleteMaskedAdapter;
            if (s.rules.?.width != b.steps.maskedWidth()) return error.MaskShape;
            b.owner = s;
            s.backend_lease = .entered;
            b.epoch += 1;
            b.sequence = 0;
            b.first_token = 0;
            b.first_pending = false;
            try b.frame(.begin, 0, prompt, &.{});
            if (comptime @hasDecl(Steps, "configureSampling")) b.steps.configureSampling(s.sampling); // structured_users: as a plain prefill's
            try b.steps.prefillToLogits(prompt);
            s.cached = 0;
            b.first_pending = true;
        }

        fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
            if (comptime !masked) return error.IncompleteMaskedAdapter;
            const b = self(ptr);
            if (b.owner != s or s.rules == null or !b.first_pending) return error.NoDeferredFirst;
            if (position != b.steps.len) return error.PositionMismatch;
            if ((s.rules.?.active and mask.len != b.steps.maskedWords()) or (!s.rules.?.active and mask.len != 0)) return error.MaskShape;
            try b.frame(.first, @intCast(position), &.{}, mask);
            const token = try b.steps.pickFirst(mask);
            if (b.mirror.chosen_cpu) |check| try check(b.mirror.ctx, token);
            b.first_pending = false;
            b.first_token = token;
            return token;
        }

        /// One-token windows of one stream, no draft head: what the round loop may ask.
        pub fn facts(_: *const Self) lanes.Model {
            return .{ .exact_width = 1, .max_streams = 1, .batch_rows = 1, .streams_exact = true };
        }

        fn self(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
            const b = self(ptr);
            if (b.structured_poisoned) return error.ControlPoison;
            if (s.sampling) |p| if (p.temperature > 0) {
                if (comptime @hasDecl(Steps, "samplingEnabled")) {
                    if (!b.steps.samplingEnabled()) return error.GreedyOnly;
                } else return error.GreedyOnly;
            };
            if (b.owner != null) return error.NoFreeSlot;
            const prompt = s.prompt();
            if (prompt.len == 0) return error.EmptyPrompt;
            if (prompt.len + s.max_new > b.capacity) return error.ContextFull;
            if (prompt.len + s.max_new > b.reach) return error.PastDenseLimit;
            if (s.isCancelled()) return error.Cancelled;
            if (s.rules != null) return b.prefillStructured(s, prompt);
            try b.mirror.tell(.begin, prompt);
            b.owner = s;
            errdefer release(ptr, s); // the core releases only a cancelled prefill: a failed one frees its slot here
            s.cached = 0;
            if (comptime @hasDecl(Steps, "configureSampling")) b.steps.configureSampling(s.sampling);
            b.first_token = try b.steps.prefill(prompt);
        }

        fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
            const b = self(ptr);
            if (b.owner != s) return error.UnknownStream;
            if (s.rules != null) {
                if (s.rules.?.active) return error.MissingMask;
                return firstMasked(ptr, s, position, &.{});
            }
            if (position != b.steps.len) {
                release(ptr, s);
                return error.PositionMismatch;
            }
            return b.first_token;
        }

        fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
            return error.Unsupported; // each token comes back to the host every round: no step runs ahead
        }

        fn read(_: *anyopaque, handle: u64) anyerror!u32 {
            return @intCast(handle);
        }

        /// A one-row window: the pending token fed at the sequence's end, the next drawn and the row committed.
        fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const b = self(ptr);
            if (windows.len != 1) return error.WindowOutOfStep;
            const w = windows[0];
            if (b.owner != w.stream) return error.UnknownStream;
            if (w.held != 0 or w.tokens.len != 0 or w.parents != null or w.early) return error.DraftsUnsupported;
            if (w.positions.len != 1 or w.positions[0] != b.steps.len + 1) return error.PositionMismatch;
            if (b.steps.len + 1 > b.reach) return error.PastDenseLimit;
            if (w.stream.rules != null) {
                if (comptime !masked) return error.IncompleteMaskedAdapter;
                const r = w.stream.rules.?;
                if (out.len != 1 or out[0].sampled.len != 1 or out[0].drafts.len != 0) return error.MaskShape;
                if ((r.active and w.masks.len != b.steps.maskedWords()) or (!r.active and w.masks.len != 0)) return error.MaskShape;
                if (b.first_pending) return error.NoDeferredFirst;
                try b.frame(.step, b.steps.len + 1, &.{w.pending}, w.masks);
                const token = try b.steps.stepMasked(w.pending, w.masks);
                if (b.mirror.chosen_cpu) |check| try check(b.mirror.ctx, token);
                out[0].sampled[0] = token;
                return;
            }
            if (w.masks.len != 0) return error.IncompleteMaskedAdapter;
            try b.mirror.tell(.step, &.{w.pending});
            out[0].sampled[0] = try b.steps.step(w.pending);
        }

        /// A one-row window keeps its row (the core asks only when it keeps fewer).
        fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const b = self(ptr);
            for (windows, paths) |w, path| {
                if (b.owner != w.stream) return error.UnknownStream;
                if (path.len != 1 or path[0] != 0) return error.KeepOutOfStep;
            }
        }

        fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
            return error.DraftsUnsupported; // qualification_qualification_44ba554e.7 brings the MTP head
        }

        fn release(ptr: *anyopaque, s: *Stream) void {
            const b = self(ptr);
            if (b.owner != s) return;
            if (s.rules != null) s.backend_lease = .released;
            if (s.rules != null and b.structured_session != null) {
                const session = b.structured_session.?;
                if (b.mirror.structured_driver_cpu) |driver| driver.terminate(b.mirror.ctx, session.status);
                b.structured_poisoned = session.status == .poison;
                session.cancel_ctx = null;
                session.cancel_check = null;
                if (comptime @hasDecl(Steps, "clearStructured")) b.steps.clearStructured();
                b.steps.release();
                b.first_pending = false;
                b.first_token = 0;
                b.owner = null;
                b.structured_session = null;
                return;
            }
            b.first_pending = false;
            b.first_token = 0;
            b.owner = null;
            b.mirror.tell(.release, &.{}) catch |err| std.log.err("glm: rank 1 missed a release: {s}", .{@errorName(err)});
            b.steps.release();
        }
    };
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.GreedyOnly => "the native CUDA GLM-5.3-Flash engine decodes greedily only: send temperature 0 (or serve with --temperature 0)",
        error.ContextFull => "the prompt and max_tokens do not fit this server's --context",
        error.PastDenseLimit => "the prompt and max_tokens pass the positions this CUDA GLM-5.3-Flash server reaches (TF_GLM_REACH, or 2,051 without the DSA scratch): shorten the prompt or lower max_tokens",
        error.NoFreeSlot => "the CUDA GLM-5.3-Flash engine serves one stream at a time: retry shortly",
        else => null,
    };
}
