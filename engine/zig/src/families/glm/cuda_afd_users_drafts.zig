//! user_drafts (TF_AFD_USERS_DRAFTS=1, mode A): MTP drafts on the users path. A lone stream decodes by the single-user
//! drafted loop (cuda_afd_drafts.mtpWith / autoWith, the `--drafts` policy, as afd_serve_drafts.decodeWith runs it) on
//! its slot's State. A slot that may draft carries its MTP state through the shared rounds in a backlog: the
//! (final-normed hidden row, next token) pairs its MTP cache has not absorbed; the last pair (the last cache row, the
//! pending token) is the loop's start pair. Rebuilding it later is impossible (the shared rounds keep no hidden rows and
//! the KDA layers' recurrent state cannot replay positions), so it is carried: seeded by a prompt pass or a yielding
//! loop, grown by each shared round, absorbed (cuda_afd_drafts.absorb's own steps) when full and at a resume.
//! Ash Hart authored TensorFold and its lane core's lone driver; Mia (@MiaAI_lab) authored the EXL3 recipe; this file:
//! Claude Opus 5.5 (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const views = @import("cuda_views.zig");
const cc = @import("cuda_compute.zig");
const decode = @import("cuda_decode.zig");
const ad = @import("cuda_afd_drafts.zig");
const Attn = @import("cuda_afd_engine.zig").Attn;
const Fwd = @import("cuda_fwd.zig").Fwd;
const Buf = views.Buf;

/// Pairs a slot's backlog holds by default (auto_decode's backlog rows); full, all but the last are absorbed.
pub const rows: usize = decode.backlog_rows;

/// The drafted loop's policy (afd_serve_drafts.decodeWith's kinds): `auto` takes DepthPolicy.auto at each run.
pub const Policy = struct { auto: bool = false, depth: decode.DepthPolicy = .{} };

/// A slot's next prompt pass: `off` as today (no absorb, no backlog); `seed` absorbs and seeds the backlog with its
/// last row and first token; `lone` absorbs and leaves that pair to the drafted loop that follows at once (the lone
/// driver's opening: the single-user path's calls).
pub const Track = enum { off, seed, lone };

/// Each drafted round's kept tokens to the users backend; true ends the loop after that round.
pub const Hook = struct {
    ctx: *anyopaque,
    kept: *const fn (ctx: *anyopaque, tokens: []const u32, drafts: usize, keep: usize) anyerror!bool,
};

/// A drafted run's counters (decode.Decoded's).
pub const Ran = struct { rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, tokens: usize = 0, ns: u64 = 0 };

/// One slot's carried MTP state.
pub const Backlog = struct {
    hidden: Buf,
    tokens: [rows]u32 = undefined,
    n: usize = 0,
    track: Track = .off,
    /// a lone opening's start pair is the Engine's last_hidden + the first token (nothing ran since)
    head: bool = false,
    flushes: u64 = 0,

    pub fn init(hidden: usize, cap: usize) Backlog {
        return .{ .hidden = Buf.of(.bf16, &.{ @min(cap, rows), hidden }) };
    }

    pub fn capacity(b: *const Backlog) usize {
        return b.hidden.shape[0];
    }

    /// Back to untracked (its owner left or failed): nothing here is read by the next owner.
    pub fn clear(b: *Backlog) void {
        b.n = 0;
        b.track = .off;
        b.head = false;
    }

    /// Tracked, its pairs ending at the pending token: a resume can start from it.
    pub fn carried(b: *const Backlog) bool {
        return b.track != .off and b.n >= 1 and !b.head;
    }
};

/// A `seed` prompt pass ended: its last row (the Engine's last_hidden) and the first token are the first pair.
pub fn seed(x: *Attn, f: *Fwd, b: *Backlog, first: u32) !void {
    try cc.copyDtoD(f, cc.rows(b.hidden, 0, 1), cc.rows(x.e.last_hidden, 0, 1));
    b.tokens[0] = first;
    b.n = 1;
    b.head = false;
}

/// A shared round's row for this slot: its final-normed row (its tail just wrote it) at the backlog's end, before the
/// next row's tail reuses the scratch; its token at the draw, the pair counted at the commit only.
pub fn grow(f: *Fwd, b: *Backlog, fnormed: Buf) !void {
    if (b.n >= b.capacity()) return error.BacklogFull;
    try cc.copyDtoD(f, cc.rows(b.hidden, b.n, 1), cc.rows(fnormed, 0, 1));
}

pub fn drawn(b: *Backlog, token: u32) void {
    if (b.n < b.capacity()) b.tokens[b.n] = token;
}

pub fn committed(b: *Backlog) void {
    b.n += 1;
}

/// All pairs but the last into the slot's MTP cache (the State in the Engine), the last moved to the front; settled.
pub fn absorbHead(x: *Attn, f: *Fwd, b: *Backlog) !void {
    if (b.n < 2) return;
    _ = try ad.absorbRows(x, f, cc.rows(b.hidden, 0, b.n - 1), b.tokens[0 .. b.n - 1]);
    try cc.copyDtoD(f, cc.rows(b.hidden, 0, 1), cc.rows(b.hidden, b.n - 1, 1));
    b.tokens[0] = b.tokens[b.n - 1];
    b.n = 1;
    b.flushes += 1;
}

/// The loop's start: a lone opening's pair is already the Engine's last_hidden; a resume absorbs its pairs but the
/// last, which becomes last_hidden (its token the pending one), exactly the pair the single-user loop starts from.
pub fn start(x: *Attn, f: *Fwd, b: *Backlog, pending: u32) !void {
    if (b.track == .off) return error.NotTracked;
    if (b.head) {
        b.head = false;
        return;
    }
    if (b.n == 0) return error.NotCarried;
    if (b.tokens[b.n - 1] != pending) return error.PendingOutOfStep;
    try absorbHead(x, f, b);
    _ = try cc.clone(f, &x.e.tmp, cc.rows(b.hidden, 0, 1), &x.e.last_hidden);
    b.n = 0;
}

/// A yielding loop's last round: its kept rows (the verify's final-normed rows, Engine.mainHidden) and their tokens,
/// what the loop's next draft would have absorbed, become the backlog.
pub fn carry(x: *Attn, f: *Fwd, b: *Backlog, kept: []const u32) !void {
    if (kept.len == 0 or kept.len > b.capacity() or kept.len > decode.max_rows) return error.CarryRows;
    try cc.copyDtoD(f, cc.rows(b.hidden, 0, kept.len), x.e.mainHidden(kept.len));
    @memcpy(b.tokens[0..kept.len], kept);
    b.n = kept.len;
    b.head = false;
}

/// The drafted loop (afd_serve_drafts.decodeWith's) on the State in the Engine, from its start pair; `hook` hears
/// each round's kept tokens and may end it.
pub fn run(x: *Attn, f: *Fwd, p: Policy, pending: u32, count: usize, out: []u32, hook: *const Hook) !Ran {
    var a: Adapter = .{ .hook = hook };
    const dh: decode.Hook = .{ .ctx = &a, .before = Adapter.before, .kept = Adapter.kept };
    var policy = p.depth;
    const r = if (p.auto) blk: {
        policy = decode.DepthPolicy.auto(x.e.sampling == null or x.e.sampling.?.temperature <= 0.0);
        break :blk try ad.autoWith(x, f, pending, count, out, &policy, false, null, &dh);
    } else try ad.mtpWith(x, f, pending, count, out, &policy, false, null, &dh);
    return .{ .rounds = r.rounds, .drafted = r.drafted, .accepted = r.accepted, .tokens = r.tokens.len, .ns = r.ns };
}

const Adapter = struct {
    hook: *const Hook,

    fn before(_: *anyopaque, _: decode.Hook.Op, _: []const u32) anyerror!void {} // one node: no rank 1 to tell

    fn kept(p: *anyopaque, tokens: []const u32, drafts: usize, keep: usize) anyerror!bool {
        const a: *Adapter = @ptrCast(@alignCast(p));
        return a.hook.kept(a.hook.ctx, tokens, drafts, keep);
    }
};

test "user_drafts backlog: capacity, the carried rule and clear" {
    var b = Backlog.init(64, 8);
    try std.testing.expectEqual(@as(usize, 8), b.capacity());
    try std.testing.expect(!b.carried());
    b.track = .seed;
    try std.testing.expect(!b.carried());
    b.n = 1;
    try std.testing.expect(b.carried());
    b.head = true;
    try std.testing.expect(!b.carried());
    b.clear();
    try std.testing.expect(!b.carried() and b.n == 0 and b.track == .off and !b.head);
    try std.testing.expectEqual(@as(usize, rows), Backlog.init(64, 100_000).capacity());
}
