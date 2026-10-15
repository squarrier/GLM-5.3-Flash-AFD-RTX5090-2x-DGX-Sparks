//! concurrent_users: independent greedy users over one execution-thread Steps, never consecutive rows of one State.
//! Ash Hart authored TensorFold/lane core; Mia (@MiaAI_lab) authored together=solo/exact.py and EXL3 recipes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api"); // user_drafts: the lane host's lone driver (api.Lone)
const be = lanes.backend;
const Stream = lanes.Stream;

const round = @import("glm_cuda").afd_users_round;
const dsv = @import("glm_cuda").afd_users_dsv; // shared_verify: drafted shared verify (TF_AFD_USERS_SHARED_DRAFTS=1)
const pp = @import("glm_cuda").afd_users_pp;
const pp_costs = costs: {
    var c: [pp.total_rows]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = if (i == 0) 19.552 else 1.7 * (19.552 + 6.995 * @as(f64, @floatFromInt(i / 2))) };
    break :costs c;
};
pub const Key = round.Key;
pub const Row = round.Row;

/// shared_verify: a lone stream's window ms by rows and a drafted shared forward's ms by total rows, the depth rule's and the
/// allocation's priors until timed (each stream's own rounds and the shared rounds' overhead are learned at run time):
/// tool_api_fixes's measured row costs (evidence/tool_api_fixes/receipts/c4-projection.json inputs): the 5090's R=1 forward 7.115 ms and
/// 0.141 ms a row, the halves' wait 11.895 ms for one row (qualification_30a87650 C2 18.749 - 6.854) and 6.854 ms a row, launch gaps
/// 0.542 ms: 19.552 + 6.995 (rows - 1).
const dsv_costs = costs: {
    var c: [dsv.max_rows]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = 19.552 + 6.995 * @as(f64, @floatFromInt(i)) };
    break :costs c;
};
/// shared_verify: one chained MTP head step (qualification_a3ab3a09 B's c3:0.85 run, 0.983 ms a step).
const dsv_mtp_step_ms = 0.983;

fn cancelled(ctx: ?*anyopaque) bool {
    const s: *Stream = @ptrCast(@alignCast(ctx.?));
    return s.isCancelled();
}

/// user_sampling: a stream's sampling as qualification_45c10bcf's draw takes it — the single-user Bound.configureSampling's conversion, field for
/// field — or null at temperature 0 or without one (the greedy draw).
pub fn drawOf(p: ?lanes.sampling.Sampling) ?round.Sampling {
    const s = p orelse return null;
    if (!(s.temperature > 0)) return null;
    return .{ .seed = s.seed, .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p };
}

/// Steps owns separate State buffers and executes round's rows layer-major; the backend owns their Stream generations.
/// structured_users: Steps with the masked adapter's calls (prefillDeferred, firstMasked, checkMasked, maskCheck) take structured
/// streams too (TF_AFD_GRAMMAR_MASK=1): the first token waits for its mask, each round's row carries its own mask.
pub fn Users(comptime Steps: type) type {
    return struct {
        const Self = @This();
        /// shared_verify's: `rows` of the slot's last window not yet settled (a drafted window waits for its stream's draft
        /// request or keep), `kept` the rows its last settle kept, the drafts held for its next window
        const Slot = struct { owner: ?*Stream = null, generation: u64 = 0, first: u32 = 0, deferred: bool = false, rows: u8 = 0, kept: u8 = 0, held: [dsv.max_drafts]u32 = @splat(0), held_n: u8 = 0 };
        const masked = @hasDecl(Steps, "prefillDeferred") and @hasDecl(Steps, "firstMasked") and @hasDecl(Steps, "checkMasked") and @hasDecl(Steps, "maskCheck");
        steps: *Steps,
        count: u8,
        capacity: usize,
        reach: usize,
        slots: [4]Slot = @splat(.{}),
        failed: ?anyerror = null,
        /// prefill_interleave: TF_AFD_USERS_INTERLEAVE on: the backend hands out `prefill_step`, a prompt goes in a step a turn
        interleave: bool = false,
        /// user_sampling: TF_AFD_USERS_SAMPLING=1: a sampled stream (finite temperature > 0) is served, each of its draws its own
        /// request's (Row.sampling; its first token's through Steps.configureSampling); off: GreedyOnly, as before
        sampled: bool = false,
        /// user_drafts (TF_AFD_USERS_DRAFTS=1): a lone greedy stream decodes by Steps.drafted (`lone`); the slots of streams
        /// that may draft carry their MTP state through the shared rounds (Steps.track). Off: nothing below is read.
        drafts: bool = false,
        /// user_drafts: prompt plus max_tokens a drafted stream may reach (TF_GLM_DRAFT_REACH, else the dense limit)
        draft_reach: usize = 0,
        /// user_drafts: the lane host's allocator (a lone stream's tokens land in its context)
        gpa: std.mem.Allocator = undefined,
        /// user_drafts counters: lone runs, resumes, hand-overs, drafted rounds, drafts, accepted drafts
        counts: Counts = .{},
        /// shared_verify (TF_AFD_USERS_SHARED_DRAFTS=1): the lane core drafts the streams the lone driver may draft (Lone.core_drafts),
        /// every stream's window in one forward (Steps.roundDrafted), each kept and drafted on its own State. Off: unread.
        shared_drafts: bool = false,
        ping_pong: bool = false,
        batched_drafts: bool = false, // batched_drafts: one MTP forward per depth, default off
        /// shared_verify counters: drafted shared rounds, their rows, windows wider than one row, drafts verified, settles
        dsv_counts: DsvCounts = .{},

        pub const Counts = struct { runs: u64 = 0, resumes: u64 = 0, yields: u64 = 0, rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0 };
        pub const DsvCounts = struct { rounds: u64 = 0, rows: u64 = 0, wide: u64 = 0, drafts: u64 = 0, settles: u64 = 0, requests: u64 = 0, widest: u64 = 0, most_rows: u64 = 0 };
        const can_share_drafts = @hasDecl(Steps, "roundDrafted") and @hasDecl(Steps, "settleDrafted") and @hasDecl(Steps, "draftDrafted");

        /// The lane core's window rows and drafts a round (lanes.Config.init's max_rows, max_draft): shared_verify's pending row
        /// and three drafts; else one row (today's).
        pub fn windowRows(b: *const Self) u32 {
            return if (b.shared_drafts) dsv.max_width else 1;
        }

        pub fn backend(b: *Self) be.Backend {
            // four_user_profile join: prefill_interleave's stepped vtable (interleave on) also carries structured_users's first_masked, as the plain one does
            if (comptime @hasDecl(Steps, "prefillStep")) if (b.interleave)
                return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .prefill_step = prefillStep, .first = first, .first_masked = if (masked) firstMasked else null, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
            return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = first, .first_masked = if (masked) firstMasked else null, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
        }

        pub fn facts(b: *const Self) lanes.Model {
            // shared_verify: windows of a pending row and up to three MTP drafts (or copies), several streams' in one forward of
            // at most eight rows; structured_users's one-row structured streams and the cancel discard as before (single_row_streams)
            if (b.shared_drafts) return .{ .exact_width = dsv.max_width, .mtp = true, .speculate = true, .speculate_early = false, .drafts = dsv.max_drafts, .window_costs = dsv_costs[0..dsv.max_width], .mtp_step_ms = dsv_mtp_step_ms, .streams_exact = true, .hidden_rows = true, .single_row_streams = true, .batch_rows = if (b.ping_pong) pp.total_rows else dsv.max_rows, .max_streams = b.count, .shared_costs = if (b.ping_pong) &pp_costs else &dsv_costs, .draft_streams = true };
            return .{ .exact_width = 1, .max_streams = b.count, .batch_rows = b.count, .hidden_rows = true, .streams_exact = true, .single_row_streams = true };
        }

        fn self(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn alive(b: *const Self) !void {
            if (b.failed) |err| return err;
            if (b.count != 2 and b.count != 4) return error.BadAfdUsers;
        }

        pub fn keyOf(b: *const Self, s: *Stream) !Key {
            for (b.slots[0..b.count], 0..) |slot, i| if (slot.owner == s) return .{ .slot = @intCast(i), .generation = slot.generation };
            return error.UnknownStream;
        }

        /// Stale completion/release handles cannot release a new owner even at the same Stream address.
        pub fn releaseKey(b: *Self, key: Key) void {
            if (key.slot >= b.count) return;
            const slot = &b.slots[key.slot];
            if (slot.owner == null or slot.generation != key.generation) return;
            if (@hasDecl(Steps, "observeRelease")) b.steps.observeRelease(key, slot.owner.?.reason.name());
            b.steps.release(key);
            slot.owner = null;
            slot.deferred = false; // structured_users: a structured owner's deferred first never outlives it
            slot.rows = 0; // shared_verify: its unsettled window and held drafts end with it
            slot.kept = 0;
            slot.held_n = 0;
        }

        fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
            const b = self(ptr);
            const key = try b.claim(s);
            errdefer b.releaseKey(key);
            b.trackFor(key, s, .seed); // user_drafts: a stream that may draft alone carries its MTP state (off: nothing)
            // four_user_profile join: structured_users's structured branch after prefill_interleave's claim (the checks, shared_pool's region, the slot)
            if (comptime masked) if (s.rules != null) {
                // structured_users: the prompt to its last row's logits; the first token waits for the stream's mask (firstMasked)
                s.backend_lease = .entered;
                b.steps.prefillDeferred(key, s.prompt()) catch |err| {
                    b.failed = err;
                    return err;
                };
                b.slots[key.slot].first = 0;
                b.slots[key.slot].deferred = true;
                return;
            };
            b.slots[key.slot].first = b.steps.prefill(key, s.prompt()) catch |err| {
                b.failed = err;
                return err;
            };
        }

        /// The checks a prompt pass passes, then a free slot's next generation owned by `s`.
        fn claim(b: *Self, s: *Stream) !Key {
            try b.alive();
            if (s.sampling) |p| if (p.temperature > 0 or !std.math.isFinite(p.temperature)) {
                // user_sampling: with TF_AFD_USERS_SAMPLING=1 a finite temperature > 0 is served (each row its own keyed draw)
                if (!b.sampled or !std.math.isFinite(p.temperature)) return error.GreedyOnly;
            };
            if (s.drafts and !b.shared_drafts) return error.UsersDraftsUnsupported; // shared_verify: on, the lane core may draft
            if (s.reuse.saved != null) return error.UsersCacheUnsupported;
            if (s.prompt_len == 0) return error.EmptyPrompt;
            if (s.prompt_len > b.capacity or s.max_new > b.capacity - s.prompt_len) return error.ContextFull;
            if (s.prompt_len > b.reach or s.max_new > b.reach - s.prompt_len) return error.PastDenseLimit;
            if (s.isCancelled()) return error.Cancelled;
            if (s.rules) |r| { // structured_users: a structured stream needs the masked adapter, over this vocabulary
                if (comptime !masked) return error.IncompleteMaskedAdapter else {
                    try b.steps.checkMasked();
                    if (r.width != b.steps.maskedWidth()) return error.MaskShape;
                }
            }
            if (b.keyOf(s)) |_| return error.AlreadyOwned else |err| if (err != error.UnknownStream) return err;
            for (b.slots[0..b.count], 0..) |*slot, i| {
                if (slot.owner != null) continue;
                if (slot.generation == std.math.maxInt(u64)) return error.GenerationExhausted;
                // shared_pool: the request's cache region before the slot changes owner; a full pool or a request past it
                // refuses this request only (the lane host's Admit keeps it queued instead), anything else fails all
                if (@hasDecl(Steps, "reserve")) b.steps.reserve(@intCast(i), s.prompt_len, s.max_new) catch |err| {
                    if (err != error.PoolFull and err != error.ContextFull) b.failed = err;
                    return err;
                };
                slot.generation += 1;
                slot.owner = s;
                s.cached = 0;
                // shared_verify: the lane core drafts exactly the streams this backend's slot carries an MTP head for (greedy,
                // unstructured, within the drafted reach: trackFor's rule); every other stream stays one row a round
                if (s.drafts and !(s.lone_drafts and s.rules == null and b.draftable(s) and drawOf(s.sampling) == null)) s.drafts = false;
                // user_sampling: the slot's request sampling for its first token's draw (the switch on only; release clears it)
                if (b.sampled) if (comptime @hasDecl(Steps, "configureSampling")) b.steps.configureSampling(@intCast(i), drawOf(s.sampling));
                return .{ .slot = @intCast(i), .generation = slot.generation };
            }
            return error.NoFreeSlot;
        }

        /// prefill_interleave: one step of the served prompt pass (`Steps.prefillStep`); the first claims a slot as `prefill` does, a
        /// cancel between steps releases it, and a failed step fails every owner as a failed `prefill` does.
        fn prefillStep(ptr: *anyopaque, s: *Stream) anyerror!bool {
            if (comptime !@hasDecl(Steps, "prefillStep")) return error.NoPrefillSteps;
            const b = self(ptr);
            const owned: ?Key = b.keyOf(s) catch null;
            const key = owned orelse try b.claim(s);
            if (owned == null) b.trackFor(key, s, .seed); // user_drafts: as `prefill`, before its first step
            if (owned != null) {
                try b.alive();
                if (s.isCancelled()) {
                    if (@hasDecl(Steps, "observeRelease")) b.steps.observeRelease(key, "cancelled");
                    b.steps.release(key);
                    b.slots[key.slot].owner = null;
                    return error.Cancelled;
                }
            }
            const got = b.steps.prefillStep(key, s.prompt()) catch |err| {
                b.failed = err;
                b.releaseKey(key);
                return err;
            };
            b.slots[key.slot].first = got orelse return false;
            return true;
        }

        fn first(ptr: *anyopaque, s: *Stream, pos: u64) anyerror!u64 {
            const b = self(ptr);
            try b.alive();
            const key = try b.keyOf(s);
            if (pos != b.steps.position(key)) return error.PositionMismatch;
            if (comptime masked) if (b.slots[key.slot].deferred) {
                // structured_users: a structured stream whose grammar is not active yet: the plain draw on its deferred logits
                if (s.rules == null or s.rules.?.active) return error.MissingMask;
                return firstMasked(ptr, s, pos, &.{});
            };
            return b.slots[key.slot].first;
        }

        /// structured_users: a structured stream's first token among `mask`'s (empty: its grammar is not active yet), drawn on the
        /// logits its prompt pass left; the mask is refused before any launch when empty or malformed.
        fn firstMasked(ptr: *anyopaque, s: *Stream, pos: u64, mask: []const u32) anyerror!u64 {
            const b = self(ptr);
            try b.alive();
            const key = try b.keyOf(s);
            const slot = &b.slots[key.slot];
            const r = s.rules orelse return error.NoDeferredFirst;
            if (!slot.deferred) return error.NoDeferredFirst;
            if (pos != b.steps.position(key)) return error.PositionMismatch;
            if ((r.active and mask.len != b.steps.maskedWords()) or (!r.active and mask.len != 0)) return error.MaskShape;
            if (mask.len != 0) try b.steps.maskCheck(mask);
            const token = b.steps.firstMasked(key, mask) catch |err| {
                b.failed = err; // past the checks only device work fails: ALL owners fail, as a failed round's
                return err;
            };
            slot.deferred = false;
            slot.first = token;
            return token;
        }

        fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
            return error.Unsupported;
        }

        fn read(_: *anyopaque, handle: u64) anyerror!u32 {
            return @intCast(handle);
        }

        fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const b = self(ptr);
            try b.alive();
            if (b.shared_drafts) return b.verifyDrafted(windows, out); // shared_verify: every window in one drafted forward
            if (windows.len == 0 or windows.len > b.count or out.len != windows.len) return error.WindowOutOfStep;
            var rows: [4]Row = undefined;
            var mapping: [4]usize = undefined;
            var seen: [4]bool = @splat(false);
            var n: usize = 0;
            // Validate the ENTIRE request before calling Steps or altering any State.
            for (windows, out, 0..) |w, o, wi| {
                const key = try b.keyOf(w.stream);
                if (seen[key.slot]) return error.DuplicateWindow;
                seen[key.slot] = true;
                if (w.held != 0 or w.tokens.len != 0 or w.parents != null or w.early or w.stream.drafts) return error.UsersDraftsUnsupported;
                if (w.positions.len != 1 or o.sampled.len != 1 or o.drafts.len != 0) return error.WindowOutOfStep;
                const pos = b.steps.position(key);
                if (w.positions[0] != pos + 1) return error.PositionMismatch;
                if (pos >= b.reach or pos >= b.capacity) return error.ContextFull;
                if (w.stream.isCancelled()) {
                    out[wi].sampled[0] = 0; // the opted-in core discards a cancelled result, never emits it
                    continue;
                }
                if (w.stream.rules) |r| { // structured_users: a structured row's own mask (an inactive grammar's: none), checked here
                    if (comptime !masked) return error.IncompleteMaskedAdapter else {
                        if ((r.active and w.masks.len != b.steps.maskedWords()) or (!r.active and w.masks.len != 0)) return error.MaskShape;
                        if (b.slots[key.slot].deferred) return error.NoDeferredFirst;
                        if (w.masks.len != 0) try b.steps.maskCheck(w.masks);
                    }
                } else if (w.masks.len != 0) return error.IncompleteMaskedAdapter;
                rows[n] = .{ .key = key, .token = w.pending, .position = pos, .cancel_ctx = w.stream, .cancel_check = cancelled, .mask = w.masks, .sampling = if (b.sampled) drawOf(w.stream.sampling) else null };
                mapping[n] = wi;
                n += 1;
            }
            // Stable physical-slot order, regardless of admission/window order.
            for (0..n) |i| for (i + 1..n) |j| if (rows[j].key.slot < rows[i].key.slot) {
                std.mem.swap(Row, &rows[i], &rows[j]);
                std.mem.swap(usize, &mapping[i], &mapping[j]);
            };
            if (n == 0) return;
            var sampled: [4]u32 = @splat(0);
            b.steps.round(rows[0..n], sampled[0..n]) catch |err| {
                b.failed = err; // ALL owners fail and later prefill/verify refuse before device calls
                return err;
            };
            for (mapping[0..n], sampled[0..n]) |wi, token| out[wi].sampled[0] = token;
        }

        fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const b = self(ptr);
            try b.alive();
            if (windows.len != paths.len) return error.KeepOutOfStep;
            if (b.shared_drafts) return b.keepDrafted(windows, paths); // shared_verify: each window's kept rows settled
            for (windows, paths) |w, path| {
                _ = try b.keyOf(w.stream);
                if (w.stream.isCancelled()) continue;
                if (path.len != 1 or path[0] != 0) return error.KeepOutOfStep;
            }
        }

        fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const b = self(ptr);
            if (b.shared_drafts) return b.draftsDrafted(requests); // shared_verify: each stream's MTP head on its own State
            return error.UsersDraftsUnsupported;
        }

        // -- shared_verify (TF_AFD_USERS_SHARED_DRAFTS=1): the lane core's drafted shared rounds --------------------------------

        /// Every window (its pending row, its held drafts, its copied or forced tokens) checked before any device call,
        /// then one drafted shared forward (Steps.roundDrafted) over every live window in physical slot order. A
        /// window no keep or draft request settled kept every row (the lane core's contract): settled first.
        fn verifyDrafted(b: *Self, windows: []const be.Window, out: []be.Verified) anyerror!void {
            if (comptime !can_share_drafts) return error.UsersSharedDraftsOff else {
                if (windows.len == 0 or windows.len > b.count or out.len != windows.len) return error.WindowOutOfStep;
                for (b.slots[0..b.count], 0..) |*sl, i| if (sl.rows > 0 and sl.owner != null) {
                    try b.settleTo(.{ .slot = @intCast(i), .generation = sl.generation }, sl.rows);
                };
                var wins: [4]dsv.Window = undefined;
                var mapping: [4]usize = undefined;
                var drafts: [4][dsv.max_drafts]u32 = undefined;
                var seen: [4]bool = @splat(false);
                var n: usize = 0;
                var total: usize = 0;
                // Validate the ENTIRE request before calling Steps or altering any State.
                for (windows, out, 0..) |w, o, wi| {
                    const key = try b.keyOf(w.stream);
                    if (seen[key.slot]) return error.DuplicateWindow;
                    seen[key.slot] = true;
                    if (w.parents != null or w.early) return error.TreeWindowsUnsupported;
                    const sl = &b.slots[key.slot];
                    const rows = w.rows();
                    if (rows > dsv.max_width or w.held > sl.held_n) return error.WindowOutOfStep;
                    if (rows > 1 and (!w.stream.drafts or w.masks.len != 0)) return error.UsersDraftsUnsupported;
                    if (w.positions.len != rows or o.sampled.len != rows or o.drafts.len != rows - 1) return error.WindowOutOfStep;
                    const pos = b.steps.position(key);
                    for (w.positions, 0..) |p, j| if (p != pos + 1 + j) return error.PositionMismatch;
                    if (pos + rows > b.reach or pos + rows > b.capacity) return error.ContextFull;
                    total += rows;
                    @memcpy(o.drafts[0..w.held], sl.held[0..w.held]);
                    @memcpy(o.drafts[w.held..], w.tokens);
                    if (w.stream.isCancelled()) {
                        @memset(o.sampled, 0); // the opted-in core discards a cancelled result, never emits it
                        continue;
                    }
                    if (w.stream.rules) |r| { // structured_users: a structured row's own mask (an inactive grammar's: none), checked here
                        if (comptime !masked) return error.IncompleteMaskedAdapter else {
                            if ((r.active and w.masks.len != b.steps.maskedWords()) or (!r.active and w.masks.len != 0)) return error.MaskShape;
                            if (b.slots[key.slot].deferred) return error.NoDeferredFirst;
                            if (w.masks.len != 0) try b.steps.maskCheck(w.masks);
                        }
                    } else if (w.masks.len != 0) return error.IncompleteMaskedAdapter;
                    @memcpy(drafts[n][0 .. rows - 1], o.drafts);
                    wins[n] = .{ .row = .{ .key = key, .token = w.pending, .position = pos, .cancel_ctx = w.stream, .cancel_check = cancelled, .mask = w.masks, .sampling = if (b.sampled) drawOf(w.stream.sampling) else null }, .drafts = drafts[n][0 .. rows - 1] };
                    mapping[n] = wi;
                    n += 1;
                }
                if (total > (if (b.ping_pong) pp.total_rows else dsv.max_rows)) return error.WindowOutOfStep;
                for (windows) |w| b.slots[(try b.keyOf(w.stream)).slot].held_n = 0; // the held drafts were this round's
                // Stable physical-slot order, regardless of admission/window order.
                for (0..n) |i| for (i + 1..n) |j| if (wins[j].row.key.slot < wins[i].row.key.slot) {
                    std.mem.swap(dsv.Window, &wins[i], &wins[j]);
                    std.mem.swap(usize, &mapping[i], &mapping[j]);
                };
                if (n == 0) return;
                var rows_n: usize = 0;
                for (wins[0..n]) |w| rows_n += w.rows();
                var sampled: [pp.total_rows]u32 = @splat(0);
                b.steps.roundDrafted(wins[0..n], sampled[0..rows_n]) catch |err| {
                    b.failed = err; // ALL owners fail and later prefill/verify refuse before device calls
                    return err;
                };
                var r0: usize = 0;
                for (wins[0..n], mapping[0..n]) |w, wi| {
                    const rows = w.rows();
                    defer r0 += rows;
                    @memcpy(out[wi].sampled, sampled[r0 .. r0 + rows]);
                    const sl = &b.slots[w.row.key.slot];
                    sl.rows = 0;
                    sl.kept = 0;
                    if (w.row.cancelled()) continue; // nothing drawn or committed: the core discards it
                    if (rows == 1) sl.kept = 1 else sl.rows = @intCast(rows); // a one-row window's row is committed
                    if (rows > 1) b.dsv_counts.wide += 1;
                    b.dsv_counts.drafts += rows - 1;
                    b.dsv_counts.widest = @max(b.dsv_counts.widest, rows);
                }
                b.dsv_counts.rounds += 1;
                b.dsv_counts.rows += rows_n;
                b.dsv_counts.most_rows = @max(b.dsv_counts.most_rows, rows_n);
            }
        }

        /// The first `n` rows of `key`'s last window stay (Steps.settleDrafted rolls the rest back); a window already
        /// settled (its draft request named its rows, or a one-row window) must be named the same again.
        fn settleTo(b: *Self, key: Key, n: usize) !void {
            if (comptime !can_share_drafts) return error.UsersSharedDraftsOff else {
                const sl = &b.slots[key.slot];
                if (sl.rows == 0) {
                    if (n != sl.kept) return error.KeepOutOfStep;
                    return;
                }
                if (n == 0 or n > sl.rows) return error.KeepOutOfStep;
                b.steps.settleDrafted(key, sl.rows, n) catch |err| {
                    b.failed = err;
                    return err;
                };
                sl.kept = @intCast(n);
                sl.rows = 0;
                b.dsv_counts.settles += 1;
            }
        }

        /// The lane core's keep: each live window's kept prefix (a chain: rows 0 .. k-1) settled.
        fn keepDrafted(b: *Self, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            for (windows, paths) |w, path| {
                const key = try b.keyOf(w.stream);
                if (w.stream.isCancelled()) continue;
                if (path.len == 0 or path.len > dsv.max_width) return error.KeepOutOfStep;
                for (path, 0..) |r, j| if (r != j) return error.TreeWindowsUnsupported;
                const sl = &b.slots[key.slot];
                if ((sl.rows == 0 and path.len != sl.kept) or (sl.rows > 0 and path.len > sl.rows)) return error.KeepOutOfStep;
            }
            for (windows, paths) |w, path| {
                if (w.stream.isCancelled()) continue;
                try b.settleTo(try b.keyOf(w.stream), path.len);
            }
        }

        /// The lane core's draft requests (one batch a round): each stream's last window settled at the rows the request
        /// names, then its MTP head (Steps.draftDrafted) absorbs and chains `depth` drafts held for its next window. A
        /// cancelled or finished stream drafts nothing (it is released after the step).
        fn draftsDrafted(b: *Self, requests: []const be.DraftRequest) anyerror!void {
            if (comptime !can_share_drafts) return error.UsersSharedDraftsOff else {
                try b.alive();
                if (b.batched_drafts) return b.draftsBatched(requests);
                for (requests) |r| {
                    _ = try b.keyOf(r.stream);
                    if (r.early or r.lanes != null or r.ranks) return error.TreeDraftsUnsupported;
                    if (r.depth > dsv.max_drafts) return error.DraftOutOfStep;
                    if (r.rows) |path| {
                        if (path.len != r.follow.len) return error.DraftOutOfStep;
                        for (path, 0..) |x, j| if (x != j) return error.TreeDraftsUnsupported;
                    } else if (r.first == null and r.follow.len != 1) return error.DraftOutOfStep;
                }
                for (requests) |r| {
                    const key = try b.keyOf(r.stream);
                    const sl = &b.slots[key.slot];
                    sl.held_n = 0;
                    if (r.stream.isCancelled()) continue;
                    var one: [1]u32 = undefined;
                    var follow = r.follow;
                    if (r.rows) |path| {
                        if (path.len == 0) continue; // nothing kept: the stream ends this step
                        try b.settleTo(key, path.len);
                    } else {
                        one[0] = if (r.first) |f| switch (f) {
                            .handle => |h| @intCast(h),
                            .value => |v| v,
                        } else r.follow[0];
                        follow = &one;
                    }
                    if (r.stream.finished) continue;
                    if (!r.stream.drafts) return error.UsersDraftsUnsupported;
                    if (r.position != b.steps.position(key) + 1) return error.PositionMismatch;
                    b.steps.draftDrafted(key, follow, r.rows != null, r.position, sl.held[0..r.depth]) catch |err| {
                        b.failed = err;
                        return err;
                    };
                    sl.held_n = @intCast(r.depth);
                    b.dsv_counts.requests += 1;
                }
            }
        }

        /// batched_drafts: collect all live requests with durable per-slot first-token storage, then batch their MTP steps.
        fn draftsBatched(b: *Self, requests: []const be.DraftRequest) !void {
            if (comptime !@hasDecl(Steps, "draftsBatched")) return error.UsersBatchedDraftsOff else {
                const bd = @import("glm_cuda").afd_users_bd;
                if (requests.len > b.count) return error.DraftOutOfStep;
                var seen: [4]bool = @splat(false);
                for (requests) |r| {
                    const key = try b.keyOf(r.stream);
                    if (seen[key.slot]) return error.DuplicateWindow;
                    seen[key.slot] = true;
                    if (r.early or r.lanes != null or r.ranks) return error.TreeDraftsUnsupported;
                    if (r.depth > dsv.max_drafts) return error.DraftOutOfStep;
                    if (r.stream.isCancelled() or r.stream.finished) continue;
                    if (!r.stream.drafts) return error.UsersDraftsUnsupported;
                    const sl = &b.slots[key.slot];
                    var position = b.steps.position(key);
                    if (r.rows) |path| {
                        if (path.len != r.follow.len or path.len > dsv.max_width) return error.DraftOutOfStep;
                        for (path, 0..) |x, j| if (x != j) return error.TreeDraftsUnsupported;
                        if (path.len == 0) continue;
                        if (sl.rows > 0) {
                            if (path.len > sl.rows) return error.KeepOutOfStep;
                            position += path.len;
                        } else if (path.len != sl.kept) return error.KeepOutOfStep;
                    } else if (r.first == null and r.follow.len != 1) return error.DraftOutOfStep;
                    if (r.position != position + 1) return error.PositionMismatch;
                }
                var batch: [4]bd.Request = undefined;
                var ones: [4][1]u32 = undefined;
                var n: usize = 0;
                for (requests) |r| {
                    const key = try b.keyOf(r.stream);
                    const sl = &b.slots[key.slot];
                    sl.held_n = 0;
                    if (r.stream.isCancelled() or r.stream.finished) continue;
                    var follow = r.follow;
                    if (r.rows) |path| {
                        if (path.len == 0) continue;
                        try b.settleTo(key, path.len);
                    } else {
                        ones[key.slot][0] = if (r.first) |f| switch (f) {
                            .handle => |h| @intCast(h),
                            .value => |v| v,
                        } else r.follow[0];
                        follow = &ones[key.slot];
                    }
                    batch[n] = .{ .row = .{ .key = key, .token = 0, .position = b.steps.position(key), .cancel_ctx = r.stream, .cancel_check = cancelled }, .follow = follow, .verified = r.rows != null, .position = r.position, .out = sl.held[0..r.depth] };
                    n += 1;
                }
                if (n == 0) return;
                b.steps.draftsBatched(batch[0..n]) catch |err| {
                    b.failed = err;
                    for (b.slots[0..b.count]) |*sl| sl.held_n = 0;
                    return err;
                };
                for (batch[0..n]) |r| if (!r.row.cancelled()) {
                    b.slots[r.row.key.slot].held_n = @intCast(r.out.len);
                    b.dsv_counts.requests += 1;
                };
            }
        }

        fn release(ptr: *anyopaque, s: *Stream) void {
            const b = self(ptr);
            const key = b.keyOf(s) catch return;
            b.releaseKey(key);
        }

        const can_draft = @hasDecl(Steps, "drafted") and @hasDecl(Steps, "track") and @hasDecl(Steps, "carry") and @hasDecl(Steps, "carried");

        /// user_drafts: `key`'s next prompt pass for a stream that may draft alone: `how` (seed: carried through the
        /// shared rounds; lone: the drafted loop follows at once); every other stream's slot untracked. Off: nothing.
        fn trackFor(b: *Self, key: Key, s: *const Stream, comptime how: @TypeOf(.enum_literal)) void {
            if (comptime !can_draft) return;
            if (!b.drafts) return;
            if (s.lone_drafts and s.rules == null and b.draftable(s)) b.steps.track(key.slot, how) else b.steps.track(key.slot, .off);
        }

        fn draftable(b: *const Self, s: *const Stream) bool {
            return s.prompt_len <= b.draft_reach and s.max_new <= b.draft_reach - s.prompt_len;
        }

        /// user_drafts: the lane host's driver for a lone greedy stream (api.Lone, upstream v1.0.7's yield hand-over): its
        /// prompt pass absorbing into its slot's MTP head, then the `--drafts` policy's drafted rounds on its own State
        /// (Steps.drafted) until it finishes, or an arrival or a cancel hands it to the lane core (the round's kept rows
        /// carried); `resume_lone` takes it back when it decodes alone again. Sampled streams never (sampled false).
        pub fn lone(b: *Self, gpa: std.mem.Allocator) api.Lone {
            b.gpa = gpa;
            // shared_verify: the lane core drafts the streams this driver may draft (core_drafts), in drafted shared rounds
            return .{ .ctx = b, .run = loneRun, .core_undrafted = true, .core_drafts = b.shared_drafts, .resume_lone = loneResume, .resumable = loneResumable };
        }

        fn loneRun(ptr: *anyopaque, s: *Stream, hooks: api.LoneHooks) anyerror!bool {
            if (comptime !can_draft) return error.UsersDraftsOff else {
                const b = self(ptr);
                if (!b.drafts) return error.UsersDraftsOff;
                const key = try b.claim(s);
                b.trackFor(key, s, .lone);
                b.slots[key.slot].first = b.steps.prefill(key, s.prompt()) catch |err| {
                    b.failed = err;
                    b.releaseKey(key);
                    return err;
                };
                _ = s.commit(b.gpa, &.{b.slots[key.slot].first}) catch |err| {
                    b.releaseKey(key);
                    return err;
                };
                b.counts.runs += 1;
                hooks.committed(hooks.ctx);
                if (!s.finished and !(s.lone_drafts and b.draftable(s))) return b.handOver(key, s); // the core's rounds
                return b.loneDecode(key, s, hooks);
            }
        }

        fn loneResume(ptr: *anyopaque, s: *Stream, hooks: api.LoneHooks) anyerror!bool {
            if (comptime !can_draft) return error.UsersDraftsOff else {
                const b = self(ptr);
                const key = try b.keyOf(s);
                b.alive() catch |err| {
                    b.releaseKey(key);
                    return err;
                };
                b.counts.resumes += 1;
                if (b.shared_drafts) { // shared_verify: the drafts the lane core held for its next window are the loop's to make
                    s.dropHeld(b.gpa);
                    b.slots[key.slot].held_n = 0;
                }
                return b.loneDecode(key, s, hooks);
            }
        }

        fn loneResumable(ptr: *anyopaque, s: *Stream) bool {
            if (comptime !can_draft) return false else {
                const b = self(ptr);
                if (!b.drafts or b.failed != null or !s.lone_drafts or s.rules != null or !b.draftable(s)) return false;
                const key = b.keyOf(s) catch return false;
                if (s.context.items.len == 0 or (s.pending orelse return false) != s.context.items[s.context.items.len - 1]) return false;
                if (s.cache_len != b.steps.position(key) or b.slots[key.slot].deferred) return false;
                return b.steps.carried(key);
            }
        }

        /// user_drafts: the stream to the lane core (LaneHost adopts it): its cache settled, `pending` its last token.
        fn handOver(b: *Self, key: Key, s: *Stream) bool {
            s.cache_len = b.steps.position(key);
            s.pending = s.context.items[s.context.items.len - 1];
            b.counts.yields += 1;
            return true;
        }

        /// user_drafts: the drafted rounds from the stream's pending token; its slot released unless it is handed over.
        fn loneDecode(b: *Self, key: Key, s: *Stream, hooks: api.LoneHooks) anyerror!bool {
            if (s.finished or s.isCancelled()) {
                b.releaseKey(key);
                return if (s.finished) false else error.Cancelled;
            }
            const count: usize = 1 + (s.max_new - s.emitted().len); // the pending token, then what is left
            const out = b.gpa.alloc(u32, count) catch |err| {
                b.releaseKey(key);
                return err;
            };
            defer b.gpa.free(out);
            var h: Hooked = .{ .b = b, .key = key, .s = s, .hooks = hooks };
            const hook: Steps.DraftHook = .{ .ctx = &h, .kept = Hooked.kept };
            const r = b.steps.drafted(key, s.context.items[s.context.items.len - 1], count, out, &hook) catch |err| {
                if (h.failed == null) b.failed = err; // past the checks only device work fails: every owner, as a round's
                b.releaseKey(key);
                return h.failed orelse err;
            };
            s.rounds += r.rounds;
            s.drafted += r.drafted;
            s.accepted += r.accepted;
            b.counts.rounds += r.rounds;
            b.counts.drafted += r.drafted;
            b.counts.accepted += r.accepted;
            if (h.failed) |e| {
                b.releaseKey(key);
                return e;
            }
            if (h.yielded) return b.handOver(key, s);
            b.releaseKey(key);
            if (!s.finished and s.isCancelled()) return error.Cancelled;
            return false;
        }

        /// Each drafted round's kept tokens into the stream, then the host told; the stream's end or a cancel ends the
        /// loop; an arrival (`hooks.yield`) ends it with the round's kept rows carried for the lane core's rounds.
        const Hooked = struct {
            b: *Self,
            key: Key,
            s: *Stream,
            hooks: api.LoneHooks,
            failed: ?anyerror = null,
            yielded: bool = false,

            fn kept(p: *anyopaque, tokens: []const u32, _: usize, rows: usize) anyerror!bool {
                const h: *Hooked = @ptrCast(@alignCast(p));
                _ = h.s.commit(h.b.gpa, tokens) catch |e| {
                    h.failed = e;
                    return true;
                };
                h.hooks.committed(h.hooks.ctx);
                if (h.s.finished or h.s.isCancelled()) return true;
                if (!h.hooks.yield(h.hooks.ctx)) return false;
                if (tokens.len != rows) return false; // a cut round is the stream's last (it finished above)
                try h.b.steps.carry(h.key, tokens); // what the loop's next draft would have absorbed
                h.yielded = true;
                return true;
            }
        };
    };
}
