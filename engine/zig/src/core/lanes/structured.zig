//! Structured-only one-row CPU path; the ordinary round loop remains unchanged.
//! structured_users: on a backend whose shared forwards take independent one-row streams (the users backend, TF_AFD_USERS), a
//! structured stream joins other streams: alone it keeps this path; in a shared round (Engine.roundStreams) each
//! structured row carries its own stream's mask (`rows`) and the plain rows none. Claude Opus 5.5 via Hermes.
const std = @import("std");
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;

fn forced(s: *Stream, a: std.mem.Allocator) !?u32 {
    if (s.cutsNext()) return try s.startClose(a);
    return s.popForce();
}
pub fn release(e: *Engine, s: *Stream) void {
    if (s.backend_lease == .entered) {
        s.backend_lease = .released;
        e.backend.release(s);
    }
    s.pending = null;
    s.dropRoundState(e.gpa);
}
/// structured_users: the backend's shared forwards take independent one-row streams (Model.single_row_streams): structured
/// streams may share its rounds, each row drawn under its own stream's mask.
pub fn shared(e: *const Engine) bool {
    return e.cfg.single_row_streams and e.cfg.family_streams;
}
pub fn add(e: *Engine, s: *Stream) !void {
    if (e.live.items.len != 0 and !(shared(e) and e.live.items.len < e.cfg.batch_streams)) return error.NoFreeSlot;
    const hook = e.backend.vtable.first_masked orelse return error.IncompleteMaskedAdapter;
    if (s.drafts or s.proposer != null) return error.DraftsUnsupported;
    if (s.max_new == 0) {
        s.finished = true;
        s.reason = .length;
        return;
    }
    errdefer { release(e, s); s.finished = true; s.reason = .@"error"; }
    try e.backend.prefill(s);
    if (s.isCancelled()) return error.Cancelled;
    const mask = try s.rules.?.next(e.arena.allocator());
    const handle = if (mask.len != 0) try hook(e.backend.ptr, s, s.prompt_len, mask) else try e.backend.first(s, s.prompt_len);
    const token = (try forced(s, e.gpa)) orelse try e.backend.read(handle);
    s.cache_len = s.prompt_len;
    _ = try s.commit(e.gpa, &.{token});
    s.pending = token;
    if (s.finished) {
        release(e, s);
        return;
    }
    try e.live.append(e.gpa, s);
}
pub fn step(e: *Engine) !bool {
    if (e.live.items.len == 0 or e.live.items[0].rules == null) return false;
    if (e.live.items.len != 1) {
        if (shared(e)) return false; // structured_users: a shared round (Engine.roundStreams), each structured row masked
        return error.StructuredStreamsUnsupported;
    }
    const s = e.live.items[0];
    errdefer {
        s.finished = true;
        s.reason = .@"error";
        release(e, s);
        e.live.clearRetainingCapacity();
    }
    if (s.isCancelled()) {
        s.finished = true;
        s.reason = .cancelled;
    } else {
        _ = e.arena.reset(.retain_capacity);
        const mask = try s.rules.?.next(e.arena.allocator());
        const positions = [_]u64{s.cache_len + 1};
        var sampled: [1]u32 = undefined;
        var out = [_]be.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
        const window = [_]be.Window{.{ .stream = s, .pending = s.pending.?, .held = 0, .tokens = &.{}, .parents = null, .positions = &positions, .masks = mask }};
        try e.backend.verify(&window, &out);
        s.cache_len += 1;
        const token = (try forced(s, e.gpa)) orelse sampled[0];
        _ = try s.commit(e.gpa, &.{token});
        s.pending = token;
        s.rounds += 1;
    }
    if (s.finished) {
        release(e, s);
        e.live.clearRetainingCapacity();
    }
    return true;
}

/// structured_users: a shared round's streams that take a row, and each row's mask (empty: unconstrained, a plain stream's row).
pub const Rows = struct { streams: []*Stream, masks: []const []const u32 };

/// structured_users: the row masks of a shared round, one a stream, from each structured stream's own matcher (a cancelled
/// stream's row is skipped by the backend: no mask). A structured stream whose grammar cannot go on (its mask fails,
/// or allows no token) leaves the round alone, finished with an error, and the others' round goes on. Null when no
/// stream of the round is structured: the round exactly as before.
pub fn rows(e: *Engine, streams: []*Stream) !?Rows {
    var any = false;
    for (streams) |s| any = any or s.rules != null;
    if (!any) return null;
    const a = e.arena.allocator();
    const kept = try a.alloc(*Stream, streams.len);
    const masks = try a.alloc([]const u32, streams.len);
    var n: usize = 0;
    for (streams) |s| {
        var mask: []const u32 = &.{};
        if (s.rules) |r| if (!s.isCancelled()) {
            mask = r.next(a) catch |err| {
                end(s, err);
                continue;
            };
            if (r.active and dead(mask)) {
                end(s, error.EmptyMask);
                continue;
            }
        };
        kept[n] = s;
        masks[n] = mask;
        n += 1;
    }
    return .{ .streams = kept[0..n], .masks = masks[0..n] };
}

fn dead(mask: []const u32) bool {
    for (mask) |word| if (word != 0) return false;
    return true;
}

/// structured_users: a structured stream of a shared round ends alone (its grammar refused its row or its token); the step's end
/// releases it (backend release: concurrent_users_r4's failure_retired) and its Rules close with the stream.
pub fn end(s: *Stream, err: anyerror) void {
    std.log.warn("lanes: a structured stream left its shared round: {s}", .{@errorName(err)});
    s.finished = true;
    s.reason = .@"error";
}

/// structured_users: the one-row window's kept path, for a structured stream ended after a shared round's verify.
pub const row0 = [_]u32{0};
