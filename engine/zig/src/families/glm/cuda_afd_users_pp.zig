//! ping_pong: <=8-row independent user micro-batches, qualification_c533375f's FIFO two-flight order on one CUDA stream.
//! Ash Hart authored TensorFold/MCDMA; @Bizuayeu authored the shared-window design at TensorFold 75131b66
//! (Apache-2.0); Hugh Madden (@dangerm00se) authored the inherited AFD exchange design. No outside code copied.
//! GPT-6.1 Sol (OpenAI Codex; planning and execution), via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const dsv = @import("cuda_afd_users_dsv.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const afd = @import("cuda_afd_attn.zig");
const pairs = @import("cuda_afd_pairs.zig");

pub const max_rows: usize = 8;
pub const total_rows: usize = 16;

pub fn parse(value: ?[]const u8, shared_drafts: bool, paired: bool, inflight: u8, lanes: u8) !bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (!std.mem.eql(u8, v, "1")) return error.BadAfdUsersPingPong;
    if (!shared_drafts) return error.PingPongNeedsSharedDrafts;
    if (!paired) return error.PingPongNeedsPairs;
    if (lanes != 0 and (lanes < 3 or lanes > 4)) return error.PingPongLanesUnsupported;
    // Decode remains two-credit; a prompt-lanes profile keeps its existing larger wire region.
    if (inflight != (if (lanes == 0) @as(u8, 2) else lanes)) return error.PingPongNeedsInflight;
    return true;
}

pub const Group = struct {
    wins: [2]dsv.Window = undefined,
    index: [2]usize = undefined,
    len: usize = 0,
    rows: usize = 0,
};
pub const Plan = struct {
    groups: [2]Group = .{ .{}, .{} },
    offsets: [4]usize = @splat(0),
    rows: usize = 0,
};

/// <=2 windows per group, minimal row imbalance; stable physical-window order and stable ties.
pub fn split(wins: []const dsv.Window) !Plan {
    if (wins.len == 0 or wins.len > 4) return error.Rows;
    var p: Plan = .{};
    var seen: [4]bool = @splat(false);
    for (wins, 0..) |w, i| {
        _ = try dsv.total(&.{w});
        if (w.row.key.slot >= seen.len or seen[w.row.key.slot]) return error.StaleGeneration;
        seen[w.row.key.slot] = true;
        p.offsets[i] = p.rows;
        p.rows += w.rows();
    }
    var mask: usize = 1;
    if (wins.len > 1) {
        var best: usize = std.math.maxInt(usize);
        for (1..@as(usize, 1) << @intCast(wins.len)) |candidate| {
            if (candidate & 1 == 0 or @popCount(candidate) != (wins.len + 1) / 2) continue;
            var a: usize = 0;
            for (wins, 0..) |w, i| if (candidate & (@as(usize, 1) << @intCast(i)) != 0) {
                a += w.rows();
            };
            const b = p.rows - a;
            if (a > max_rows or b > max_rows) continue;
            const diff = if (a > b) a - b else b - a;
            if (diff < best) {
                best = diff;
                mask = candidate;
            }
        }
        if (best == std.math.maxInt(usize)) return error.Rows;
    }
    for (wins, 0..) |w, i| {
        const slot: usize = if (mask & (@as(usize, 1) << @intCast(i)) != 0) 0 else 1;
        const g = &p.groups[slot];
        g.wins[g.len] = w;
        g.index[g.len] = i;
        g.len += 1;
        g.rows += w.rows();
    }
    return p;
}

/// Generic order, exercised by the real device and CPU stand-ins. Cancellation drains a promised pair.
pub fn run(comptime Device: type, d: *Device, wins: []const dsv.Window, out: []u32) !void {
    const p = try split(wins);
    if (out.len != p.rows or p.groups[1].len == 0) return error.Rows;
    @memset(out, 0);
    try d.alive();
    errdefer {
        @memset(out, 0);
        d.fail();
    }
    const behind = try d.begin(&p);
    for (p.groups, 0..) |g, slot| try d.open(@intCast(slot), g.wins[0..g.len], g.rows);
    var li: [2]usize = @splat(0);
    var at: [2]?usize = .{ null, null };
    at[0] = try d.advance(0, p.groups[0].wins[0..p.groups[0].len], p.groups[0].rows, &li[0]);
    if (at[0]) |l| try d.issue(0, p.groups[0].rows, l);
    at[1] = try d.advance(1, p.groups[1].wins[0..p.groups[1].len], p.groups[1].rows, &li[1]);
    if (at[0] == null and at[1] != null or at[0] != null and at[1] == null) return error.OutOfStep;
    if (at[1]) |l| {
        if (!behind) try d.issue(1, p.groups[1].rows, l);
        var first = true;
        while (true) {
            try d.alive();
            try d.complete(0, p.groups[0].rows, at[0].?);
            if (first and behind) try d.issue(1, p.groups[1].rows, at[1].?);
            first = false;
            at[0] = try d.advance(0, p.groups[0].wins[0..p.groups[0].len], p.groups[0].rows, &li[0]);
            if (at[0]) |next| try d.issue(0, p.groups[0].rows, next);
            try d.alive();
            try d.complete(1, p.groups[1].rows, at[1].?);
            at[1] = try d.advance(1, p.groups[1].wins[0..p.groups[1].len], p.groups[1].rows, &li[1]);
            if (at[0] == null and at[1] == null) break;
            if (at[0] == null or at[1] == null) return error.OutOfStep;
            try d.issue(1, p.groups[1].rows, at[1].?);
        }
    }
    // Both forwards drained before any host draw, commit, MTP or pool retirement.
    var tokens: [max_rows]u32 = undefined;
    for (p.groups, 0..) |g, slot| {
        try d.alive();
        try d.tailDraw(@intCast(slot), g.wins[0..g.len], g.rows, tokens[0..g.rows]);
        var offset: usize = 0;
        for (g.wins[0..g.len], g.index[0..g.len]) |w, original| {
            @memcpy(out[p.offsets[original] .. p.offsets[original] + w.rows()], tokens[offset .. offset + w.rows()]);
            offset += w.rows();
        }
    }
    for (wins) |w| {
        if (w.row.cancelled() or w.rows() != 1) continue;
        try d.alive();
        try d.commit(w);
    }
    try d.end();
}

/// B needs only values that cross a stop. Attention scratch/stages/head are consumed before reuse.
pub const Copies = struct {
    x: views.Buf,
    post: views.Buf,
    comb: views.Buf,
    sy: views.Buf,
    hdr: views.Buf,

    pub fn init(b: *const bufs.Buffers, a: *const afd.Afd) !Copies {
        return .{ .x = freshRows(b.x), .post = freshRows(b.post), .comb = freshRows(b.comb), .sy = freshRows(b.sy orelse return error.NoExl3Scratch), .hdr = pairs.Copies.fresh(a.w.hdr) };
    }
    fn freshRows(b: views.Buf) views.Buf {
        var out = pairs.Copies.fresh(b);
        out.shape[0] = max_rows;
        return out;
    }
    pub fn view(c: Copies, base: bufs.Buffers) bufs.Buffers {
        var b = base;
        b.x = c.x;
        b.post = c.post;
        b.comb = c.comb;
        b.sy = c.sy;
        return b;
    }
    pub fn bytes(c: Copies) !usize {
        var copy = c;
        var arena: bufs.Arena = .{};
        try views.walk(&copy, views.Path.init("ping"), &arena, bufs.Arena.visit);
        return arena.used;
    }
};

pub const State = struct {
    c: Copies,
    ring: ?afd.Ring = null,
    last: ?u32 = null,
    runs: u64 = 0,

    pub fn audit(p: *State, d: *const cuda.Driver, a: *const afd.Afd) !void {
        var r = afd.Ring.of(a.layout, max_rows, null);
        const h = try cuda.HostBuffer.alloc(d, r.total(a.positions()));
        r.bytes = h.bytes;
        r.owned = h;
        p.ring = r;
    }
    pub fn detach(p: *State) void {
        if (p.ring) |*r| if (r.owned) |*h| h.free();
        p.ring = null;
    }
};

/// Forecast from the same concrete buffers as usePingPong; no copies when OFF.
pub fn bytes(m: anytype, on: bool) !u64 {
    if (!on) return 0;
    const b = try bufs.Buffers.init(m, max_rows, 8, false);
    const a = try afd.Afd.initWith(m, max_rows, m.mtp_layers > 0);
    return (try Copies.init(&b, &a)).bytes();
}
