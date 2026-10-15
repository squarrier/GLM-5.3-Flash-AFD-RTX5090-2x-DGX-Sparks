//! python-0.6's decode loops (glm5_next/cuda/decode.py, ed78d6f: absorb, draft, serial_decode, mtp_decode and
//! DepthPolicy; drafter_choice.auto_decode with no DFlash2 drafter, so every round drafts with the MTP head) over a
//! rank's Engine. A round verifies the pending token and the head's drafts in one window, keeps the rows through the
//! first draft the target did not draw (forward.commit replays the KDA states and shifts the conv windows past the
//! rest), and drafts again from the kept rows (absorb first trims the head's rejected chain from its cache). Drafts
//! only propose: drafted output equals serial output token for token. The loops' shape follows Mia's Flash Next
//! cuda_decode.zig (MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold cbfc429, Apache-2.0); every call, its order
//! and its arguments are python's GLM ones, which tools/zig/glm_mtp_fixtures.py records for cuda_decode_test.zig.
//! `e` is an Engine (cuda_engine.EngineOf): its State, Buffers, sampler, and its forward and MTP steps.
const std = @import("std");
const trace = @import("cuda_trace.zig");
const cc = @import("cuda_compute.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;

/// engine.MAX_ROWS: the widest verify window (a pending token and up to 7 drafts)
pub const max_rows = 8;
/// auto_decode's `cap`: backlog rows the head may owe before it absorbs them anyway
pub const backlog_rows = 256;

/// decode.DepthPolicy: a fixed draft count, or 1-3 by the running acceptance rate (python's floats, f64); a chain
/// stops under `confidence` (decode.draft) when it is above 0. `first` (draft_policy, 0: python's) is the first draft's own
/// floor: a round whose first draw is under it drafts nothing and verifies the pending token alone.
pub const DepthPolicy = struct {
    most: usize = 3,
    fixed: bool = false,
    low: f64 = 0.8,
    high: f64 = 0.9,
    confidence: f64 = 0.0,
    rate: f64 = 0.8,
    first: f64 = 0.0,

    pub fn next(p: *DepthPolicy, drafted: usize, accepted: usize) usize {
        if (p.fixed) return p.most;
        if (drafted > 0) p.rate = 0.875 * p.rate + 0.125 * (@as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(drafted)));
        const d: usize = if (p.rate < p.low) 1 else if (p.rate < p.high) 2 else 3;
        return @max(1, @min(p.most, d));
    }

    /// engine.py's policies for auto_decode's MTP arm: greedy c3:0.35, sampled a 0.6/0.85 (3 drafts at most). draft_policy:
    /// TF_GLM_DRAFT_AUTO=cN:P[:F] replaces the greedy one (autoEnv; refused at the policy's parse when malformed).
    pub fn auto(greedy: bool) DepthPolicy {
        if (greedy) if (autoEnv() catch null) |p| return p;
        return if (greedy) .{ .most = 3, .fixed = true, .confidence = 0.35 } else .{ .most = 3, .low = 0.6, .high = 0.85 };
    }

    /// The chain's floors as draft applies them.
    pub fn floors(p: *const DepthPolicy) Floors {
        return .{ .confidence = p.confidence, .first = p.first };
    }
};

/// decode.draft's stop rule: the running product's floor (`confidence`, 0: none) and the first draw's (`first`).
pub const Floors = struct { confidence: f64 = 0.0, first: f64 = 0.0 };

/// cN:P[:F] (glm_drafts' c form, draft_policy's optional F the first draft's floor): N fixed drafts (capped at MAX_ROWS - 1),
/// the running product's floor P, and F in [0, 1) (0 or absent: python's rule, the first draft always kept).
pub fn parseChain(s: []const u8) !DepthPolicy {
    if (s.len < 2 or s[0] != 'c') return error.Drafts;
    var it = std.mem.splitScalar(u8, s[1..], ':');
    const most = std.fmt.parseInt(usize, it.next().?, 10) catch return error.Drafts;
    const conf = std.fmt.parseFloat(f64, it.next() orelse return error.Drafts) catch return error.Drafts;
    const first = if (it.next()) |x| std.fmt.parseFloat(f64, x) catch return error.Drafts else 0.0;
    if (most == 0 or it.next() != null or !(first >= 0.0 and first < 1.0)) return error.Drafts;
    return .{ .most = @min(most, max_rows - 1), .fixed = true, .confidence = conf, .first = first };
}

/// TF_GLM_DRAFT_AUTO (draft_policy, BadAd84's served draft stop (MiaAI-Lab Qwen3.8-Flash-Next-Single #34) for this engine):
/// unset or empty, python's greedy auto; cN:P[:F] with 0 < P < 1 the greedy auto's chain instead. Anything else is
/// error.BadDraftAuto, which `--drafts auto`'s parse reports before the load.
pub fn autoEnv() !?DepthPolicy {
    const v = std.mem.span(std.c.getenv("TF_GLM_DRAFT_AUTO") orelse return null);
    if (v.len == 0) return null;
    const p = parseChain(v) catch return error.BadDraftAuto;
    if (!(p.confidence > 0.0 and p.confidence < 1.0)) return error.BadDraftAuto;
    if (!told.swap(true, .monotonic)) std.debug.print("INFO drafts: TF_GLM_DRAFT_AUTO={s}: greedy auto drafts up to {d} under the running product {d} (python's c3:0.35 unset), the first draft's floor {d}\n", .{ v, p.most, p.confidence, p.first });
    return p;
}

/// autoEnv's INFO line, once a process
var told = std.atomic.Value(bool).init(false);

/// decode.DecodeResult without the timings: the rounds' depths and keeps as counts by size, the head's steps, and
/// auto_decode's backlog left unabsorbed (python's `pending`: the Engine's backlog rows 0 .. pending - 1).
pub const Decoded = struct {
    tokens: []u32,
    /// verify forwards, the drafts they checked and the drafts kept
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    /// rounds by drafts verified (0 .. 7) and by rows kept (1 .. 8)
    depths: [max_rows]usize = @splat(0),
    keeps: [max_rows + 1]usize = @splat(0),
    /// MTP head forwards: absorb's parts and chained drafts
    mtp_steps: usize = 0,
    pending: usize = 0,
    /// the request's nucleus calls by the gathers they took (1, 2 or 3; from prefill on: Engine.stats, qualification_45c10bcf's)
    gathers: [3]u64 = @splat(0),
    /// python's DecodeResult.stages (ns: the drafts, the verify forwards to the stream drained, the draws, the
    /// commits; stage_names) and seconds (ns, the loop's wall, the stream drained at both ends)
    stages: [4]u64 = @splat(0),
    ns: u64 = 0,
    /// each round's drafts, when the caller gives rows for them (tf-glm-forward's record; null: not kept)
    log: ?*Log = null,
    /// a served stream's round hook (qualification_qualification_44ba554e.8 day 3; null in tf-glm-forward)
    hook: ?*const Hook = null,

    pub const stage_names = [4][]const u8{ "draft", "forward", "sample", "commit" };

    /// Each round's drafts in order (`at[i]`: where round i's start in `drafts`), up to the rows given; with `probs`
    /// (draft_policy) each draft's temperature-1 probability beside it (NaN where its policy reads none).
    pub const Log = struct {
        drafts: []u32,
        at: []usize,
        n: usize = 0,
        rounds: usize = 0,
        probs: ?[]f64 = null,

        pub fn add(l: *Log, d: []const u32, p: []const f64) void {
            if (l.rounds >= l.at.len or l.n + d.len > l.drafts.len) return;
            if (l.probs) |x| if (l.n + d.len > x.len) return;
            l.at[l.rounds] = l.n;
            @memcpy(l.drafts[l.n .. l.n + d.len], d);
            if (l.probs) |x| @memcpy(x[l.n .. l.n + d.len], p[0..d.len]);
            l.n += d.len;
            l.rounds += 1;
        }
    };

    /// The acceptance rate: drafts kept over drafts verified (0 without drafts).
    pub fn acceptance(d: *const Decoded) f64 {
        return if (d.drafted == 0) 0.0 else @as(f64, @floatFromInt(d.accepted)) / @as(f64, @floatFromInt(d.drafted));
    }

    /// Tokens a verify forward commits: engine.py's tokens_per_round, (len(tokens) - 1) / max(rounds, 1).
    pub fn perForward(d: *const Decoded) f64 {
        return @as(f64, @floatFromInt(d.tokens.len -| 1)) / @as(f64, @floatFromInt(@max(d.rounds, 1)));
    }

    fn round(d: *Decoded, drafts: []const u32, probs: []const f64, keep: usize) void {
        if (d.log) |l| l.add(drafts, probs);
        d.count(drafts.len, keep);
    }

    fn count(d: *Decoded, drafts: usize, keep: usize) void {
        d.rounds += 1;
        d.drafted += drafts;
        d.accepted += keep - 1;
        d.depths[drafts] += 1;
        d.keeps[keep] += 1;
    }
};

/// A served stream's hook on the loops below (qualification_qualification_44ba554e.8 day 3, native/glm_cuda_drafts.zig). It launches nothing, so a
/// hooked loop makes the unhooked loop's launches. `before` hears each operation's words before it runs (rank 0 sends
/// them to rank 1, whose own loop checks its words against them); `kept` hears each round's kept tokens after its
/// commit, and true ends the decode there (the stream's own stops, or a cancel).
pub const Hook = struct {
    ctx: *anyopaque,
    before: *const fn (ctx: *anyopaque, op: Op, words: []const u32) anyerror!void,
    kept: *const fn (ctx: *anyopaque, tokens: []const u32, drafts: usize, keep: usize) anyerror!bool,

    /// draft: the first draft's position, the depth asked, the tokens absorbed; window: the rows' tokens; keep: the
    /// window's rows and the rows kept.
    pub const Op = enum { draft, window, keep };
};

/// The hook's `before` for a draft: its position, depth and the tokens the head absorbs.
fn toldDraft(h: ?*const Hook, position: usize, depth: usize, next: []const u32) !void {
    const x = h orelse return;
    var w: [2 + max_rows]u32 = undefined;
    if (next.len > max_rows) return error.DraftRows;
    w[0] = @intCast(position);
    w[1] = @intCast(depth);
    @memcpy(w[2 .. 2 + next.len], next);
    try x.before(x.ctx, .draft, w[0 .. 2 + next.len]);
}

/// Rows [r0, r0 + n) of a 2-D view.
fn sub(x: T, r0: usize, n: usize) T {
    var y = x;
    y.ptr += r0 * x.stride[0] * x.dtype.size();
    y.shape[0] = n;
    return y;
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// torch.cuda.synchronize(): the rank's stream drained (nothing on a recording).
fn drain(f: *Fwd) !void {
    if (f.gpu) |g| try g.stream.synchronize();
}

fn isEos(e: anytype, tok: u32) bool {
    return std.mem.indexOfScalar(u32, e.m.eos[0..e.m.eos_n], tok) != null;
}

/// decode.absorb: the head's rejected chain trimmed from its cache, then (hidden, next) rows in mbuf-sized parts;
/// the last part's logits.
pub fn absorb(e: anytype, f: *Fwd, hidden: T, next: []const u32, steps: *usize) !T {
    const st = &e.st;
    if (st.mtp_drafted > 0) {
        try cc.setMtpLen(f, st, st.mtp_len - st.mtp_drafted);
        st.mtp_drafted = 0;
    }
    const step = (e.mbuf orelse return error.NoMtp).rows;
    var logits: ?T = null;
    var s0: usize = 0;
    while (s0 < next.len) : (s0 += step) {
        const part = next[s0..@min(s0 + step, next.len)];
        logits = try e.mtpStep(f, part, sub(hidden, s0, part.len));
        try cc.setMtpLen(f, st, st.mtp_len + part.len);
        steps.* += 1;
    }
    return logits orelse error.NoRows;
}

/// decode.draft: absorb, then up to `count` drafts chained on the head's own output row, draft j drawn at
/// `position` + j. With a floor (`fl`, above 0) each draw comes with its temperature-1 probability (sample.draft_k
/// candidates a rank), the first draft stays unless under `fl.first` (draft_policy; none then), and the chain stops before a
/// draft that takes the running product under `fl.confidence`, or right after one that does. `probs` (draft_policy) takes each
/// kept draft's probability (NaN without a floor).
pub fn draft(e: anytype, f: *Fwd, hidden: T, next: []const u32, position: usize, count: usize, fl: Floors, out: []u32, probs: ?[]f64, steps: *usize) ![]u32 {
    if (count > out.len) return error.DraftRows;
    if (probs) |ps| if (count > ps.len) return error.DraftRows;
    const st = &e.st;
    var logits = try absorb(e, f, hidden, next, steps);
    var n: usize = 0;
    var chain: f64 = 1.0;
    const confidence = fl.confidence;
    const read = confidence > 0 or fl.first > 0;
    for (0..count) |j| {
        var p: [1]f64 = .{std.math.nan(f64)};
        try e.sampleRows(f, sub(logits, 0, 1), position + j, out[n .. n + 1], if (read) &p else null);
        if (j == 0 and fl.first > 0 and !(p[0] >= fl.first)) break; // draft_policy: under the first draft's floor, no draft
        if (confidence > 0 and j > 0 and chain * p[0] < confidence) break;
        if (probs) |ps| ps[n] = p[0];
        n += 1;
        if (confidence > 0) {
            chain *= p[0];
            if (chain < confidence) break; // a further draft could not pass either: no MTP step for it
        }
        if (j + 1 < count) {
            logits = try e.mtpStep(f, out[n - 1 .. n], cc.rows(e.mbuf.?.fnormed, 0, 1)); // draft_hidden: mbuf.fnormed[0:1]
            try cc.setMtpLen(f, st, st.mtp_len + 1);
            st.mtp_drafted += 1;
            steps.* += 1;
        }
    }
    return out[0..n];
}

/// decode.serial_decode: from `pending` one token a step into `out`, to `count` or an eos (stop_eos).
pub fn serial(e: anytype, f: *Fwd, pending: u32, count: usize, out: []u32, stop_eos: bool) !Decoded {
    if (count == 0 or count > out.len) return error.Count;
    out[0] = pending;
    var n: usize = 1;
    var r: Decoded = .{ .tokens = undefined };
    try drain(f);
    const start = now();
    while (n < count and !(stop_eos and isEos(e, out[n - 1]))) : (n += 1) {
        const t0 = now();
        const logits = try e.forward(f, out[n - 1 .. n]);
        try drain(f);
        const t1 = now();
        try e.sampleRows(f, sub(logits, 0, 1), e.st.pos + 1, out[n .. n + 1], null);
        const t2 = now();
        try cc.commit(f, &e.st, &e.buf, 1, 1);
        r.stages[1] += t1 - t0;
        r.stages[2] += t2 - t1;
        r.stages[3] += now() - t2;
        r.rounds += 1; // python's rounds: len(out) - 1, no depths or keeps
    }
    try drain(f);
    r.ns = now() - start;
    r.tokens = out[0..n];
    r.gathers = e.stats.gathers;
    return r;
}

/// One round's verify: [last, drafts] through the target, its draws at st.pos + 1 .., kept through the first draft
/// it did not draw (or an eos, with stop_eos), then forward.commit of the kept rows. Gives the rows kept.
fn verify(e: anytype, f: *Fwd, last: u32, drafts: []const u32, sampled: *[max_rows]u32, stop_eos: bool, r: *Decoded) !usize {
    var tokens: [max_rows]u32 = undefined;
    if (drafts.len + 1 > tokens.len) return error.DraftRows;
    tokens[0] = last;
    @memcpy(tokens[1 .. 1 + drafts.len], drafts);
    const R = 1 + drafts.len;
    if (r.hook) |h| try h.before(h.ctx, .window, tokens[0..R]);
    const t0 = now();
    const logits = try e.forward(f, tokens[0..R]);
    try drain(f);
    const t1 = now();
    try e.sampleRows(f, sub(logits, 0, R), e.st.pos + 1, sampled[0..R], null);
    var keep: usize = 1;
    for (drafts, 0..) |d, i| {
        if (sampled[i] != d or (stop_eos and isEos(e, sampled[i]))) break;
        keep += 1;
    }
    const t2 = now();
    if (r.hook) |h| try h.before(h.ctx, .keep, &.{ @intCast(R), @intCast(keep) });
    try cc.commit(f, &e.st, &e.buf, R, keep);
    r.stages[1] += t1 - t0;
    r.stages[2] += t2 - t1;
    r.stages[3] += now() - t2;
    return keep;
}

/// decode.mtp_decode: drafts from prefill's last row and the pending token, then rounds of verify and drafts from
/// the kept rows (the policy's depth, at most the tokens left). `out` takes `count` tokens, pending first.
pub fn mtp(e: anytype, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool, log: ?*Decoded.Log) !Decoded {
    return mtpWith(e, f, pending, count, out, policy, stop_eos, log, null);
}

/// mtp with a served stream's hook (null: mtp itself).
pub fn mtpWith(e: anytype, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool, log: ?*Decoded.Log, hook: ?*const Hook) !Decoded {
    if (count == 0 or count > out.len) return error.Count;
    out[0] = pending;
    var n: usize = 1;
    var r: Decoded = .{ .tokens = undefined, .log = log, .hook = hook };
    var dbuf: [max_rows - 1]u32 = undefined;
    var pbuf: [max_rows - 1]f64 = undefined;
    try drain(f);
    const start = now();
    var t0 = now();
    var depth = @min(policy.next(0, 0), count - n);
    if (depth > 0) try toldDraft(hook, e.st.pos + 1, depth, &.{pending});
    var drafts: []u32 = if (depth > 0) try draft(e, f, cc.rows(e.last_hidden, 0, 1), &.{pending}, e.st.pos + 1, depth, policy.floors(), &dbuf, &pbuf, &r.mtp_steps) else dbuf[0..0];
    r.stages[0] += now() - t0;
    var sampled: [max_rows]u32 = undefined;
    while (n < count and !(stop_eos and isEos(e, out[n - 1]))) {
        const keep = try verify(e, f, out[n - 1], drafts, &sampled, stop_eos, &r);
        r.round(drafts, &pbuf, keep);
        const take = @min(keep, count - n);
        @memcpy(out[n .. n + take], sampled[0..take]);
        n += take;
        if (hook) |h| if (try h.kept(h.ctx, sampled[0..take], drafts.len, keep)) break;
        if (n >= count or (stop_eos and isEos(e, sampled[keep - 1]))) break;
        t0 = now();
        depth = @min(policy.next(drafts.len, keep - 1), count - n);
        if (depth > 0) try toldDraft(hook, e.st.pos + 1, depth, sampled[0..keep]);
        drafts = if (depth > 0) try draft(e, f, e.mainHidden(keep), sampled[0..keep], e.st.pos + 1, depth, policy.floors(), &dbuf, &pbuf, &r.mtp_steps) else dbuf[0..0];
        r.stages[0] += now() - t0;
    }
    try drain(f);
    r.ns = now() - start;
    r.tokens = out[0..n];
    r.gathers = e.stats.gathers;
    return r;
}

/// drafter_choice.auto_decode with no DFlash2 drafter (choice None): each round drafts first, from the backlog of
/// committed rows the head has not absorbed (the Engine's `backlog`, python's m_rows: prefill's last row and the
/// pending token, then each round's kept rows), at least one draft; then the verify. The backlog left is `pending`.
pub fn auto(e: anytype, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool, log: ?*Decoded.Log) !Decoded {
    return autoWith(e, f, pending, count, out, policy, stop_eos, log, null);
}

/// auto with a served stream's hook (null: auto itself).
pub fn autoWith(e: anytype, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool, log: ?*Decoded.Log, hook: ?*const Hook) !Decoded {
    if (count == 0 or count > out.len) return error.Count;
    const D = e.m.hidden;
    try e.tmp.made("empty", &e.backlog); // torch.empty((cap, hidden), bfloat16)
    if (e.backlog.shape[0] < max_rows or e.backlog.shape[1] != D) return error.BacklogRows;
    try cc.copyDtoD(f, cc.rows(e.backlog, 0, 1), cc.rows(e.last_hidden, 0, 1));
    var next: [max_rows]u32 = undefined;
    next[0] = pending;
    var owed: usize = 1;
    out[0] = pending;
    var n: usize = 1;
    var r: Decoded = .{ .tokens = undefined, .log = log, .hook = hook };
    var last = [2]usize{ 0, 0 };
    var dbuf: [max_rows - 1]u32 = undefined;
    var pbuf: [max_rows - 1]f64 = undefined;
    var sampled: [max_rows]u32 = undefined;
    try drain(f);
    const start = now();
    while (n < count and !(stop_eos and isEos(e, out[n - 1]))) {
        const t0 = now();
        const depth = @max(1, @min(policy.next(last[0], last[1]), count - n));
        try toldDraft(hook, e.st.pos + 1, depth, next[0..owed]);
        const drafts = try draft(e, f, cc.rows(e.backlog, 0, owed), next[0..owed], e.st.pos + 1, depth, policy.floors(), &dbuf, &pbuf, &r.mtp_steps);
        r.stages[0] += now() - t0;
        owed = 0;
        const keep = try verify(e, f, out[n - 1], drafts, &sampled, stop_eos, &r);
        try cc.copyDtoD(f, cc.rows(e.backlog, owed, keep), e.mainHidden(keep)); // the kept rows join the backlog
        @memcpy(next[owed .. owed + keep], sampled[0..keep]);
        owed += keep;
        last = .{ drafts.len, keep - 1 };
        r.round(drafts, &pbuf, keep);
        const take = @min(keep, count - n);
        @memcpy(out[n .. n + take], sampled[0..take]);
        n += take;
        if (hook) |h| if (try h.kept(h.ctx, sampled[0..take], drafts.len, keep)) break;
    }
    try drain(f);
    r.ns = now() - start;
    r.tokens = out[0..n];
    r.gathers = e.stats.gathers;
    r.pending = owed;
    return r;
}

test "DepthPolicy: python's depths and rate on a fixed run (scripts/depth.py of qualification_qualification_3c2cdcdd); engine.py's auto policies" {
    var p: DepthPolicy = .{};
    const seq = [_][2]usize{ .{ 0, 0 }, .{ 2, 2 }, .{ 2, 2 }, .{ 2, 2 }, .{ 2, 2 }, .{ 2, 2 }, .{ 2, 2 }, .{ 3, 0 }, .{ 1, 1 }, .{ 1, 1 }, .{ 3, 3 } };
    const want = [_]usize{ 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2 };
    for (seq, want) |x, d| try std.testing.expectEqual(d, p.next(x[0], x[1]));
    try std.testing.expectEqual(@as(f64, 0.8636446503922344), p.rate);
    var fixed: DepthPolicy = .{ .fixed = true, .most = 2 };
    try std.testing.expectEqual(@as(usize, 2), fixed.next(3, 0));
    try std.testing.expectEqual(@as(f64, 0.35), DepthPolicy.auto(true).confidence);
    try std.testing.expect(DepthPolicy.auto(true).fixed and !DepthPolicy.auto(false).fixed);
    try std.testing.expectEqual(@as(f64, 0.6), DepthPolicy.auto(false).low);
}

test "the counters: acceptance and tokens a verify forward" {
    var toks = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    var d: Decoded = .{ .tokens = &toks };
    d.count(3, 4);
    d.count(3, 1);
    d.count(2, 2);
    try std.testing.expectEqual(@as(usize, 8), d.drafted);
    try std.testing.expectEqual(@as(usize, 4), d.accepted);
    try std.testing.expectEqual(@as(f64, 0.5), d.acceptance());
    try std.testing.expectEqual(@as(f64, 2.0), d.perForward());
    try std.testing.expectEqual(@as(usize, 2), d.depths[3]);
    try std.testing.expectEqual(@as(usize, 1), d.keeps[4]);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "draft_policy: cN:P[:F] and TF_GLM_DRAFT_AUTO; unset, auto is python's c3:0.35" {
    const t = std.testing;
    const c = try parseChain("c3:0.35");
    try t.expect(c.fixed and c.most == 3 and c.confidence == 0.35 and c.first == 0.0);
    const f = try parseChain("c4:0.5:0.45");
    try t.expect(f.most == 4 and f.confidence == 0.5 and f.first == 0.45);
    try t.expectEqual(@as(usize, max_rows - 1), (try parseChain("c9:0.2")).most);
    for ([_][]const u8{ "c0:0.5", "c3", "c3:x", "c3:0.5:1", "c3:0.5:-0.1", "c3:0.5:0.2:0.1", "f3", "c:0.5" }) |s|
        try t.expectError(error.Drafts, parseChain(s));
    _ = unsetenv("TF_GLM_DRAFT_AUTO");
    try t.expectEqual(@as(?DepthPolicy, null), try autoEnv());
    const py = DepthPolicy.auto(true);
    try t.expect(py.most == 3 and py.confidence == 0.35 and py.first == 0.0);
    _ = setenv("TF_GLM_DRAFT_AUTO", "", 1);
    try t.expectEqual(@as(?DepthPolicy, null), try autoEnv());
    _ = setenv("TF_GLM_DRAFT_AUTO", "c2:0.55:0.5", 1);
    defer _ = unsetenv("TF_GLM_DRAFT_AUTO");
    const a = DepthPolicy.auto(true);
    try t.expect(a.fixed and a.most == 2 and a.confidence == 0.55 and a.first == 0.5);
    try t.expectEqual(@as(f64, 0.6), DepthPolicy.auto(false).low); // the sampled auto is python's either way
    for ([_][*:0]const u8{ "c3:0", "c3:1", "c3:0.5:1.5", "f3", "auto", "0.5" }) |v| {
        _ = setenv("TF_GLM_DRAFT_AUTO", v, 1);
        try t.expectError(error.BadDraftAuto, autoEnv());
        try t.expectEqual(@as(f64, 0.35), DepthPolicy.auto(true).confidence); // a bad value never reaches a decode
    }
}
