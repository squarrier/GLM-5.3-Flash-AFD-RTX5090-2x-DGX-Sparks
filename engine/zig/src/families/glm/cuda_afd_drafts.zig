//! qualification_qualification_86ea5170's decode loops with MTP drafts (cuda_decode.zig's mtp and auto: python-0.6's mtp_decode and auto_decode's MTP
//! arm) on the AFD attention node (qualification_21e21bc0, TF_AFD_MTP=1). A verify window is a backbone forward under AFD (Attn.forward:
//! the slice [0, layers) begun, the forward, the sync and the check) of the pending token and up to 7 drafts, the k + 1
//! rows the exchange's decode rows carry; each MTP head step is one forward of the MTP layer's experts alone
//! (Attn.mtpStep: the slice [layers, 1)), settled before its draw is read. A failed exchange fails the request before any
//! draw from it. Every call, its order and its arguments are cuda_decode's (so python's); DepthPolicy and Decoded are
//! qualification_qualification_86ea5170's. Drafts only propose: drafted output equals serial output token for token (G-draft-equal).
const std = @import("std");
const trace = @import("cuda_trace.zig");
const cc = @import("cuda_compute.zig");
const decode = @import("cuda_decode.zig");
const Attn = @import("cuda_afd_engine.zig").Attn;
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;
const max_rows = decode.max_rows;
pub const Decoded = decode.Decoded;
pub const DepthPolicy = decode.DepthPolicy;

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

fn isEos(x: *const Attn, tok: u32) bool {
    return std.mem.indexOfScalar(u32, x.e.m.eos[0..x.e.m.eos_n], tok) != null;
}

/// Decoded.round (cuda_decode's, private there): the round's drafts (and with the log's `probs`, draft_policy, their
/// probabilities) logged, then counted by depth and keep.
fn round(r: *Decoded, drafts: []const u32, probs: []const f64, keep: usize) void {
    if (r.log) |l| l.add(drafts, probs);
    r.rounds += 1;
    r.drafted += drafts.len;
    r.accepted += keep - 1;
    r.depths[drafts.len] += 1;
    r.keeps[keep] += 1;
}

/// decode.absorb under AFD: the head's rejected chain trimmed from its cache, then (hidden, next) rows in mbuf-sized
/// parts, each one MTP forward; the last part's logits (settled by the caller's draw).
fn absorb(x: *Attn, f: *Fwd, hidden: T, next: []const u32, steps: *usize) !T {
    const st = &x.e.st;
    if (st.mtp_drafted > 0) {
        try cc.setMtpLen(f, st, st.mtp_len - st.mtp_drafted);
        st.mtp_drafted = 0;
    }
    const mb = if (x.e.mbuf) |*b| b else return error.NoMtp;
    var logits: ?T = null;
    var s0: usize = 0;
    while (s0 < next.len) : (s0 += mb.rows) {
        const part = next[s0..@min(s0 + mb.rows, next.len)];
        logits = try x.mtpStep(f, mb, part, sub(hidden, s0, part.len), .eager, null);
        try cc.setMtpLen(f, st, st.mtp_len + part.len);
        steps.* += 1;
    }
    return logits orelse error.NoRows;
}

/// user_drafts: `absorb` for a users slot's carried pairs (the held draft rows trimmed first, then the rows in mbuf-sized
/// parts, each one MTP forward), then the stream drained and the exchange checked: no draw follows.
pub fn absorbRows(x: *Attn, f: *Fwd, hidden: T, next: []const u32) !usize {
    var steps: usize = 0;
    _ = try absorb(x, f, hidden, next, &steps);
    try x.settled(f);
    return steps;
}

/// decode.draft under AFD: absorb, then up to `count` drafts chained on the head's own output row, draft j drawn at
/// `position` + j after its forward settled; the floors (`fl`: the running product's and draft_policy's first draft's) as
/// cuda_decode.draft applies them, each kept draft's probability into `probs` (draft_policy; NaN without a floor).
pub fn draft(x: *Attn, f: *Fwd, hidden: T, next: []const u32, position: usize, count: usize, fl: decode.Floors, out: []u32, probs: ?[]f64, steps: *usize) ![]u32 {
    if (count > out.len) return error.DraftRows;
    if (probs) |ps| if (count > ps.len) return error.DraftRows;
    const e = x.e;
    const st = &e.st;
    var logits = try absorb(x, f, hidden, next, steps);
    var n: usize = 0;
    var chain: f64 = 1.0;
    const confidence = fl.confidence;
    const read = confidence > 0 or fl.first > 0;
    for (0..count) |j| {
        var p: [1]f64 = .{std.math.nan(f64)};
        try x.settled(f); // the MTP forward's sync and check: no draw from a failed exchange
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
            logits = try x.mtpStep(f, &e.mbuf.?, out[n - 1 .. n], cc.rows(e.mbuf.?.fnormed, 0, 1), .eager, null); // draft_hidden
            try cc.setMtpLen(f, st, st.mtp_len + 1);
            st.mtp_drafted += 1;
            steps.* += 1;
        }
    }
    return out[0..n];
}

/// One round's verify: [last, drafts] through the backbone under AFD (synced and checked), its draws at st.pos + 1 ..,
/// kept through the first draft it did not draw (or an eos, with stop_eos), then forward.commit of the kept rows.
fn verify(x: *Attn, f: *Fwd, last: u32, drafts: []const u32, sampled: *[max_rows]u32, stop_eos: bool, r: *Decoded) !usize {
    const e = x.e;
    var tokens: [max_rows]u32 = undefined;
    if (drafts.len + 1 > tokens.len) return error.DraftRows;
    tokens[0] = last;
    @memcpy(tokens[1 .. 1 + drafts.len], drafts);
    const R = 1 + drafts.len;
    const t0 = now();
    const logits = try x.forward(f, tokens[0..R]);
    const t1 = now();
    try e.sampleRows(f, sub(logits, 0, R), e.st.pos + 1, sampled[0..R], null);
    var keep: usize = 1;
    for (drafts, 0..) |d, i| {
        if (sampled[i] != d or (stop_eos and isEos(x, sampled[i]))) break;
        keep += 1;
    }
    const t2 = now();
    try cc.commit(f, &e.st, &e.buf, R, keep);
    r.stages[1] += t1 - t0;
    r.stages[2] += t2 - t1;
    r.stages[3] += now() - t2;
    return keep;
}

fn drain(f: *Fwd) !void {
    if (f.gpu) |g| try g.stream.synchronize();
}

/// decode.mtp_decode under AFD: drafts from prefill's last row and the pending token, then rounds of verify and drafts
/// from the kept rows (the policy's depth, at most the tokens left). `out` takes `count` tokens, pending first.
pub fn mtp(x: *Attn, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool, log: ?*Decoded.Log) !Decoded {
    return mtpWith(x, f, pending, count, out, policy, stop_eos, log, null);
}

/// mtp with a served stream's hook (qualification_4be15bbf, tf-glm-afd attn serve; null: mtp itself): `kept` hears each round's kept
/// tokens and may end the decode, as cuda_decode.mtpWith's; it launches nothing, so the launches are mtp's.
pub fn mtpWith(x: *Attn, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool, log: ?*Decoded.Log, hook: ?*const decode.Hook) !Decoded {
    if (count == 0 or count > out.len) return error.Count;
    const e = x.e;
    out[0] = pending;
    var n: usize = 1;
    var r: Decoded = .{ .tokens = undefined, .log = log };
    var dbuf: [max_rows - 1]u32 = undefined;
    var pbuf: [max_rows - 1]f64 = undefined;
    try drain(f);
    const start = now();
    var t0 = now();
    var depth = @min(policy.next(0, 0), count - n);
    var drafts: []u32 = if (depth > 0) try draft(x, f, cc.rows(e.last_hidden, 0, 1), &.{pending}, e.st.pos + 1, depth, policy.floors(), &dbuf, &pbuf, &r.mtp_steps) else dbuf[0..0];
    r.stages[0] += now() - t0;
    var sampled: [max_rows]u32 = undefined;
    while (n < count and !(stop_eos and isEos(x, out[n - 1]))) {
        const keep = try verify(x, f, out[n - 1], drafts, &sampled, stop_eos, &r);
        round(&r, drafts, &pbuf, keep);
        const take = @min(keep, count - n);
        @memcpy(out[n .. n + take], sampled[0..take]);
        n += take;
        if (hook) |h| if (try h.kept(h.ctx, sampled[0..take], drafts.len, keep)) break;
        if (n >= count or (stop_eos and isEos(x, sampled[keep - 1]))) break;
        t0 = now();
        depth = @min(policy.next(drafts.len, keep - 1), count - n);
        drafts = if (depth > 0) try draft(x, f, e.mainHidden(keep), sampled[0..keep], e.st.pos + 1, depth, policy.floors(), &dbuf, &pbuf, &r.mtp_steps) else dbuf[0..0];
        r.stages[0] += now() - t0;
    }
    try drain(f);
    r.ns = now() - start;
    r.tokens = out[0..n];
    r.gathers = e.stats.gathers;
    return r;
}

/// drafter_choice.auto_decode's MTP arm under AFD: each round drafts first, from the backlog of committed rows the
/// head has not absorbed (prefill's last row and the pending token, then each round's kept rows), at least one draft;
/// then the verify. The backlog left is `pending`.
pub fn auto(x: *Attn, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool, log: ?*Decoded.Log) !Decoded {
    return autoWith(x, f, pending, count, out, policy, stop_eos, log, null);
}

/// auto with a served stream's hook (qualification_4be15bbf; null: auto itself), as cuda_decode.autoWith's `kept`.
pub fn autoWith(x: *Attn, f: *Fwd, pending: u32, count: usize, out: []u32, policy: *DepthPolicy, stop_eos: bool, log: ?*Decoded.Log, hook: ?*const decode.Hook) !Decoded {
    if (count == 0 or count > out.len) return error.Count;
    const e = x.e;
    const D = e.m.hidden;
    try e.tmp.made("empty", &e.backlog); // torch.empty((cap, hidden), bfloat16)
    if (e.backlog.shape[0] < max_rows or e.backlog.shape[1] != D) return error.BacklogRows;
    try cc.copyDtoD(f, cc.rows(e.backlog, 0, 1), cc.rows(e.last_hidden, 0, 1));
    var next: [max_rows]u32 = undefined;
    next[0] = pending;
    var owed: usize = 1;
    out[0] = pending;
    var n: usize = 1;
    var r: Decoded = .{ .tokens = undefined, .log = log };
    var last = [2]usize{ 0, 0 };
    var dbuf: [max_rows - 1]u32 = undefined;
    var pbuf: [max_rows - 1]f64 = undefined;
    var sampled: [max_rows]u32 = undefined;
    try drain(f);
    const start = now();
    while (n < count and !(stop_eos and isEos(x, out[n - 1]))) {
        const t0 = now();
        const depth = @max(1, @min(policy.next(last[0], last[1]), count - n));
        const drafts = try draft(x, f, cc.rows(e.backlog, 0, owed), next[0..owed], e.st.pos + 1, depth, policy.floors(), &dbuf, &pbuf, &r.mtp_steps);
        r.stages[0] += now() - t0;
        owed = 0;
        const keep = try verify(x, f, out[n - 1], drafts, &sampled, stop_eos, &r);
        try cc.copyDtoD(f, cc.rows(e.backlog, owed, keep), e.mainHidden(keep)); // the kept rows join the backlog
        @memcpy(next[owed .. owed + keep], sampled[0..keep]);
        owed += keep;
        last = .{ drafts.len, keep - 1 };
        round(&r, drafts, &pbuf, keep);
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
