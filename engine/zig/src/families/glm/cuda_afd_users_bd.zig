//! batched_drafts: MTP drafts batched by depth, TF_AFD_USERS_BATCHED_DRAFTS=1 (default off).
//! Extends shared_verify's per-stream state, not its verify/acceptance path. Ash Hart authored TensorFold/MTP;
//! Mia (@MiaAI_lab) authored the weights/drafter. @Bizuayeu authored the shared verify design (75131b66,
//! Apache-2.0), reimplemented by shared_verify. This extension: GPT-6.1 Sol (OpenAI Codex), via Hermes.
const std = @import("std");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const cc = @import("cuda_compute.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const layer = @import("cuda_layer.zig");
const dsa = @import("cuda_dsa.zig");
const amtp = @import("cuda_afd_mtp.zig");
const dsv = @import("cuda_afd_users_dsv.zig");
const sharing = @import("cuda_afd_users_shared.zig");
const schedule = @import("cuda_afd_users_round.zig");
const Users = @import("cuda_afd_users.zig").Users;
const Fwd = @import("cuda_fwd.zig").Fwd;
const T = trace.T;
const Key = schedule.Key;
pub const max_rows: usize = 8;

pub fn parse(v: ?[]const u8, shared_drafts: bool) !bool {
    const x = v orelse return false;
    if (std.mem.eql(u8, x, "0")) return false;
    if (!std.mem.eql(u8, x, "1")) return error.BadAfdUsersBatchedDrafts;
    if (!shared_drafts) return error.UsersBatchedDraftsNeedsSharedDrafts;
    return true;
}

pub const Request = struct {
    row: schedule.Row,
    follow: []const u32,
    verified: bool,
    position: usize,
    out: []u32,
};

pub const Stash = struct {
    hidden: views.Buf,
    forwards: u64 = 0,
    rows: u64 = 0,
    draws: u64 = 0,
    pub fn init(m: anytype, count: u8) Stash {
        return .{ .hidden = views.Buf.of(.bf16, &.{ count, m.hidden }) };
    }
};

pub fn bytes(m: anytype, count: u8, on: bool) u64 {
    return if (on) std.mem.alignForward(u64, Stash.init(m, count).hidden.bytes(), 256) else 0;
}

fn state(u: *Users, key: Key) !*bufs.State {
    if (key.slot >= u.count or u.generation[key.slot] != key.generation) return error.StaleGeneration;
    return if (key.slot == 0) &u.at.e.st else &u.states[key.slot - 1];
}

/// Validate every member before any backlog mutation, trim, forward or draw.
pub fn check(u: *Users, requests: []const Request) !void {
    if (u.bd == null) return error.UsersBatchedDraftsOff;
    const v = if (u.dsvs) |*x| x else return error.UsersSharedDraftsOff;
    const d = if (u.drafts) |*x| x else return error.UsersDraftsOff;
    if (requests.len > u.count) return error.DraftOutOfStep;
    var seen: [4]bool = @splat(false);
    for (requests) |r| {
        const st = try state(u, r.row.key);
        const slot = r.row.key.slot;
        if (seen[slot]) return error.DuplicateWindow;
        seen[slot] = true;
        if (r.row.cancelled()) continue;
        const b = &d.backlog[slot];
        const sl = &v.slot[slot];
        if (r.follow.len == 0 or r.out.len > dsv.max_drafts) return error.DraftOutOfStep;
        if (b.track == .off or b.head) return error.NotTracked;
        if (sl.pending != 0) return error.KeepOutOfStep;
        if (b.n > b.capacity() or (sl.in_head and b.n != 1)) return error.BacklogOutOfStep;
        if (st.mtp_drafted > st.mtp_len) return error.MtpOutOfStep;
        if (r.position != st.pos + 1) return error.PositionMismatch;
        const start: usize = @intFromBool(sl.in_head);
        const n = if (r.verified) blk: {
            if (sl.copied == 0 or r.follow.len > sl.copied or sl.base != b.n or r.follow.len > b.capacity() - b.n) return error.DraftOutOfStep;
            break :blk b.n + r.follow.len - start;
        } else blk: {
            if (sl.in_head or b.n == 0 or r.follow.len != 1 or b.tokens[b.n - 1] != r.follow[0]) return error.DraftOutOfStep;
            break :blk b.n;
        };
        if (n == 0) return error.DraftOutOfStep;
        try cc.checkRoom(u.at.e.m, st, n + r.out.len - @intFromBool(r.out.len > 0), st.mtp_len - st.mtp_drafted);
    }
}

const Work = struct { req: Request, start: usize, cursor: usize = 0, n: usize, proposed: [dsv.max_drafts]u32 = @splat(0) };
const Segment = struct { key: Key, next: []const u32, hidden: T };

/// One MTP forward: row-local ops shared; causal DSA on each segment's own MTP cache.
fn forward(u: *Users, segments: []const Segment) !void {
    const e = u.at.e;
    const b = if (e.mbuf) |*x| x else return error.NoMtp;
    const x = e.w.mtp orelse return error.NoMtp;
    const l = &x.layer;
    const f = u.f;
    const D = e.m.hidden;
    var ids: [max_rows]u32 = undefined;
    var R: usize = 0;
    for (segments) |s| {
        const st = try state(u, s.key);
        const n = s.next.len;
        if (n == 0 or R + n > max_rows or R + n > b.rows) return error.Rows;
        try cc.checkRoom(e.m, st, n, st.mtp_len);
        @memcpy(ids[R .. R + n], s.next);
        try cc.copyDtoD(f, cc.rows(b.hin, R, n), s.hidden);
        R += n;
    }
    if (R == 0) return error.Rows;
    try u.at.finish(f);
    f.moe = u.at.a.provider();
    try u.at.a.beginMtp(R, .eager, if (f.gpu) |g| g.stream else null);
    u.at.forwards += 1;
    try cc.stageIds(f, b, ids[0..R]);
    try cc.stagedRecord(f, b);
    try f.tri.embed(b.ids.ptr, cc.table(e.w.embed), b.me.ptr, R, D, 1);
    var r0: usize = 0;
    for (segments) |s| {
        if ((try state(u, s.key)).mtp_len == 0) try cc.zero(f, T.of(b.me, r0 * D * 2, &.{D}));
        r0 += s.next.len;
    }
    try f.tri.rmsnorm(b.me.ptr, D, x.enorm.ptr, b.mcat.ptr, 2 * D, null, e.m.eps, R, D);
    try f.tri.rmsnorm(b.hin.ptr, D, x.hnorm.ptr, b.mcat.ptr + D * 2, 2 * D, null, e.m.eps, R, D);
    const cat = cc.rows(b.mcat, 0, R);
    try f.tri.groupSums(cat.ptr, cat.stride[0], b.mxs.ptr, R, 2 * D);
    try qmm.mm(f, false, cat, x.eh, cc.rows(b.mxs, 0, R), cc.rows(b.mx, 0, R), false, b.sk, e.chip);
    try f.tri.rmsnorm(b.mx.ptr, D, l.in_norm.ptr, b.normed.ptr, D, b.xs.ptr, e.m.eps, R, D);
    var need: ?sharing.Need = null;
    var sparse = false;
    r0 = 0;
    for (segments) |s| {
        const st = try state(u, s.key);
        const v = dsv.segView(b.*, r0, s.next.len);
        const n = try dsv.dsaNeedW(e.m, l, st, &v, amtp.options(st, s.next.len, e.chip, null));
        if (need) |first| if (first.indexed != n.indexed) return error.UsersMixedIndex;
        need = n;
        sparse = sparse or n.sparse;
        r0 += s.next.len;
    }
    try sharing.dsaIn(f, l, b, R, need.?.indexed, sparse, e.chip);
    r0 = 0;
    for (segments) |s| {
        const st = try state(u, s.key);
        const v = dsv.segView(b.*, r0, s.next.len);
        try dsv.dsaSeqW(f, l, st, &v, amtp.options(st, s.next.len, e.chip, null), dsa.scratchOf(f));
        r0 += s.next.len;
    }
    const g = try sharing.dsaOut(f, l, b, R, e.chip);
    try f.tri.residualAdd(b.mx.ptr, b.mx.ptr, g, e.m.world, R, D);
    try f.tri.rmsnorm(b.mx.ptr, D, l.post_norm.ptr, b.normed.ptr, D, b.xs.ptr, e.m.eps, R, D);
    const p = f.moe orelse return error.NoProvider;
    const planes = try p.block(p.ctx, f, l, b, R, .{ .chip = e.chip });
    _ = try amtp.back(f, e.w, b, R, planes, false, e.chip);
    u.at.launched();
    try u.at.settled(f);
    for (segments) |s| {
        const st = try state(u, s.key);
        try cc.setMtpLen(f, st, st.mtp_len + s.next.len);
    }
    try u.at.settled(f);
    u.bd.?.forwards += 1;
    u.bd.?.rows += R;
    std.log.info("batched_drafts MTP forward rows={d} users={d} forward={d}", .{ R, segments.len, u.bd.?.forwards });
}

fn drawn(u: *Users, w: *Work, index: usize, row: usize) !void {
    if (w.req.row.cancelled()) return;
    const b = &u.at.e.mbuf.?;
    try u.at.e.sampleRows(u.f, cc.rows(b.logits, row, 1), w.req.position + index, w.proposed[index .. index + 1], null);
    if (w.req.row.cancelled()) return;
    try cc.copyDtoD(u.f, cc.rows(u.bd.?.hidden, w.req.row.key.slot, 1), cc.rows(b.fnormed, row, 1));
    u.bd.?.draws += 1;
}

/// Absorb waves first, then one shared forward per later depth; publish only after final sync/check.
pub fn run(u: *Users, requests: []const Request) !void {
    try check(u, requests);
    if (requests.len == 0) return;
    errdefer {
        u.at.a.dead = true;
        for (requests) |r| if (r.row.key.slot < u.count) {
            u.drafts.?.backlog[r.row.key.slot].clear();
            u.dsvs.?.slot[r.row.key.slot] = .{};
            @memset(r.out, 0);
        };
    }
    var work: [4]Work = undefined;
    var count: usize = 0;
    for (requests) |r| {
        if (r.row.cancelled()) continue;
        const b = &u.drafts.?.backlog[r.row.key.slot];
        const sl = &u.dsvs.?.slot[r.row.key.slot];
        const st = try state(u, r.row.key);
        const start: usize = @intFromBool(sl.in_head);
        if (r.verified) {
            @memcpy(b.tokens[b.n .. b.n + r.follow.len], r.follow);
            b.n += r.follow.len;
        }
        sl.copied = 0;
        if (st.mtp_drafted > 0) try cc.setMtpLen(u.f, st, st.mtp_len - st.mtp_drafted);
        st.mtp_drafted = 0;
        work[count] = .{ .req = r, .start = start, .n = b.n - start };
        count += 1;
    }
    for (0..count) |i| for (i + 1..count) |j| if (work[j].req.row.key.slot < work[i].req.row.key.slot) std.mem.swap(Work, &work[i], &work[j]);
    while (true) {
        var segments: [4]Segment = undefined;
        var mapping: [4]usize = undefined;
        var ns: usize = 0;
        var R: usize = 0;
        var active: usize = 0;
        for (work[0..count]) |w| if (!w.req.row.cancelled() and w.cursor < w.n) {
            active += 1;
        };
        if (active == 0) break;
        for (work[0..count], 0..) |*w, wi| {
            if (w.req.row.cancelled() or w.cursor == w.n) continue;
            const take = @min(w.n - w.cursor, (max_rows - R) / active);
            active -= 1;
            const b = &u.drafts.?.backlog[w.req.row.key.slot];
            const start = w.start + w.cursor;
            segments[ns] = .{ .key = w.req.row.key, .next = b.tokens[start .. start + take], .hidden = cc.rows(b.hidden, start, take) };
            mapping[ns] = wi;
            ns += 1;
            R += take;
        }
        try forward(u, segments[0..ns]);
        var r0: usize = 0;
        for (segments[0..ns], mapping[0..ns]) |s, wi| {
            const w = &work[wi];
            w.cursor += s.next.len;
            if (w.cursor == w.n and w.req.out.len > 0) try drawn(u, w, 0, r0 + s.next.len - 1);
            r0 += s.next.len;
        }
        try u.at.settled(u.f);
    }
    for (1..dsv.max_drafts) |depth| {
        var segments: [4]Segment = undefined;
        var mapping: [4]usize = undefined;
        var ns: usize = 0;
        for (work[0..count], 0..) |*w, wi| {
            if (w.req.row.cancelled() or w.req.out.len <= depth) continue;
            segments[ns] = .{ .key = w.req.row.key, .next = w.proposed[depth - 1 .. depth], .hidden = cc.rows(u.bd.?.hidden, w.req.row.key.slot, 1) };
            mapping[ns] = wi;
            ns += 1;
        }
        if (ns == 0) break;
        try forward(u, segments[0..ns]);
        for (mapping[0..ns], 0..) |wi, i| {
            const w = &work[wi];
            (try state(u, w.req.row.key)).mtp_drafted += 1;
            try drawn(u, w, depth, i);
        }
        try u.at.settled(u.f);
    }
    for (work[0..count]) |w| {
        if (w.req.row.cancelled()) continue;
        const b = &u.drafts.?.backlog[w.req.row.key.slot];
        if (b.n > 1) try cc.copyDtoD(u.f, cc.rows(b.hidden, 0, 1), cc.rows(b.hidden, b.n - 1, 1));
        b.tokens[0] = b.tokens[b.n - 1];
        b.n = 1;
        u.dsvs.?.slot[w.req.row.key.slot].in_head = true;
        u.dsvs.?.counts.draft_requests += 1;
        u.dsvs.?.counts.absorbed += w.n;
        u.dsvs.?.counts.drafts += w.req.out.len;
    }
    try u.at.settled(u.f);
    for (work[0..count]) |w| if (!w.req.row.cancelled()) @memcpy(w.req.out, w.proposed[0..w.req.out.len]);
}
