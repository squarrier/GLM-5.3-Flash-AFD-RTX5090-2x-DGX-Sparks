//! ping_pong device binding: independent shared_verify States, existing mode-7 pair wire, qualification_c533375f two-flight credit.
//! Inherited authors: Ash Hart, @Bizuayeu, Mia (@MiaAI_lab), Hugh Madden, tpurtell and Turboderp.
//! GPT-6.1 Sol (OpenAI Codex; planning and execution), via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const pp = @import("cuda_afd_users_pp.zig");
const dsv = @import("cuda_afd_users_dsv.zig");
const pairs = @import("cuda_afd_pairs.zig");
const afd = @import("cuda_afd_attn.zig");
const bufs = @import("cuda_buffers.zig");
const sharing = @import("cuda_afd_users_shared.zig");
const layer = @import("cuda_layer.zig");
const ae = @import("cuda_afd_engine.zig");
const Users = @import("cuda_afd_users.zig").Users;

const Window = struct { seq: u32, rows: usize, pos: usize = 0, pair: afd.Pair };

pub const Device = struct {
    u: *Users,
    base: bufs.Buffers = undefined,
    hdr: @import("cuda_views.zig").Buf = undefined,
    ring: ?afd.Ring = null,
    windows: [2]Window = undefined,
    turn: pairs.Turn = .{ .count = 0, .most = 2 },
    begun: bool = false,

    pub fn alive(d: *Device) !void {
        try d.u.at.alive();
    }

    fn select(d: *Device, slot: u1) void {
        const u = d.u;
        const p = &u.ping_pong.?;
        const a = u.at.a;
        const w = d.windows[slot];
        u.at.e.buf = if (slot == 0) d.base else p.c.view(d.base);
        a.seq = w.seq;
        a.cur = w.rows;
        a.mode = .chunk;
        a.first = 0;
        a.count = a.layers;
        a.pos = w.pos;
        a.pair = w.pair;
        a.lane = null;
        a.w.hdr = if (slot == 0) d.hdr else p.c.hdr;
        a.ring = if (slot == 0) d.ring else p.ring;
    }

    pub fn restore(d: *Device) void {
        if (!d.begun) return;
        const a = d.u.at.a;
        const w = d.windows[1];
        d.u.at.e.buf = d.base;
        a.seq = w.seq;
        a.cur = w.rows;
        a.mode = .eager;
        a.first = 0;
        a.count = a.layers;
        a.pos = w.pos;
        a.pair = null;
        a.lane = null;
        a.w.hdr = d.hdr;
        a.ring = d.ring;
        d.begun = false;
    }

    pub fn begin(d: *Device, p: *const pp.Plan) !bool {
        const u = d.u;
        const a = u.at.a;
        try u.at.finish(u.f);
        try u.at.alive();
        _ = try pp.parse("1", true, a.paired, a.inflight, a.lanes);
        if (u.at.structured_control != null) return error.PingPongControlUnsupported;
        if (a.wire) |w| if (!w.node.cfg.settings.pairs or w.node.cfg.settings.inflight != a.inflight or w.node.cfg.settings.lanes != a.lanes or w.node.reg.slots < 2) return error.SettingsDiffer;
        if (a.pos != a.first and a.pos != a.first + a.count) return error.OutOfStep;
        if (a.layers == 0) return error.NotMoe;
        const saved = &u.ping_pong.?;
        var behind = saved.last != null and saved.last.? == a.seq;
        if (u.at.pairs) |old| behind = behind or (old.last != null and old.last.? == a.seq);
        if (u.at.lanes) |old| behind = behind or (old.last != null and old.last.? == a.seq);
        const first = a.seq +% 1;
        d.base = u.at.e.buf;
        d.hdr = a.w.hdr;
        d.ring = a.ring;
        d.turn = .{ .count = a.layers, .most = 2 };
        d.windows = .{
            .{ .seq = first, .rows = p.groups[0].rows, .pair = .{ .slot = 0, .other = p.groups[1].rows, .first = first } },
            .{ .seq = first +% 1, .rows = p.groups[1].rows, .pair = .{ .slot = 1, .other = p.groups[0].rows, .first = first } },
        };
        d.begun = true;
        u.f.moe = a.provider();
        for (0..2) |slot| {
            d.select(@intCast(slot));
            if (a.host) |h| {
                const st = (u.f.gpu orelse return error.NoStream).stream;
                const n = a.positions() * cuda.xchg.proto.HDR_BYTES;
                a.headers(h.bytes[0..n]);
                try st.d.check(st.d.api.cuMemcpyHtoDAsync_v2(a.w.hdr.ptr, h.bytes.ptr, n, st.handle), "cuMemcpyHtoDAsync");
                try st.synchronize();
            }
            std.log.info("ping_pong micro={d} rows={d} users={d} round={d} seq={d}", .{ slot, p.groups[slot].rows, p.groups[slot].len, u.rounds, d.windows[slot].seq });
        }
        std.log.info("ping_pong decode rows={d} users={d} round={d}", .{ p.rows, p.groups[0].len + p.groups[1].len, u.rounds });
        for (p.groups, 0..) |group, slot| {
            var row0: usize = 0;
            for (group.wins[0..group.len], group.index[0..group.len]) |win, original| {
                const state = if (win.row.key.slot == 0) &u.at.e.st else &u.states[win.row.key.slot - 1];
                std.log.info("ping_pong window={d} slot={d} generation={d} position={d} parity={d} output_row0={d} rows={d} micro={d} micro_row0={d}", .{ original, win.row.key.slot, win.row.key.generation, win.row.position, state.parity, p.offsets[original], win.rows(), slot, row0 });
                row0 += win.rows();
            }
        }
        u.at.forwards += 2;
        return behind;
    }

    pub fn open(d: *Device, slot: u1, wins: []const dsv.Window, R: usize) !void {
        d.select(slot);
        var base: dsv.Dev = .{ .u = d.u };
        try base.dsvOpen(wins, R);
    }

    pub fn advance(d: *Device, slot: u1, wins: []const dsv.Window, R: usize, cursor: *usize) !?usize {
        d.select(slot);
        const u = d.u;
        const e = u.at.e;
        var base: dsv.Dev = .{ .u = u };
        while (cursor.* < e.w.layers.len) {
            try d.alive();
            const li = cursor.*;
            cursor.* += 1;
            const l = &e.w.layers[li];
            try base.dsvIn(wins, R, li);
            var r0: usize = 0;
            for (wins) |w| {
                try base.dsvSeq(w, r0, li);
                r0 += w.rows();
            }
            if (!u.isMoe(li)) {
                try base.dsvOut(R, li);
                continue;
            }
            const b = &e.buf;
            const g = switch (l.kind) {
                .kda => try sharing.kdaOut(u.f, l, b, u.stages.?, R, e.chip),
                .dsa => try sharing.dsaOut(u.f, l, b, R, e.chip),
            };
            const m = e.m;
            try u.f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
            const h = l.ffn_hc orelse return error.PlainLayer;
            try u.f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.post_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
            try u.at.a.route(u.f, l, b, R);
            return li;
        }
        return null;
    }

    pub fn issue(d: *Device, slot: u1, R: usize, li: usize) !void {
        try d.alive();
        d.select(slot);
        const u = d.u;
        try d.turn.request(slot, d.windows[slot].pos);
        try u.at.a.request(u.f, &u.at.e.w.layers[li], &u.at.e.buf, R, .{ .chip = u.at.e.chip });
        d.windows[slot].pos = u.at.a.pos;
        u.exchanges += 1;
        std.log.info("concurrent_users exchange round={d} layer={d} rows={d}", .{ u.rounds, li, R });
    }

    pub fn complete(d: *Device, slot: u1, R: usize, _: usize) !void {
        try d.alive();
        d.select(slot);
        try d.turn.take(slot, d.windows[slot].pos - 1);
        const u = d.u;
        const b = &u.at.e.buf;
        const m = u.at.e.m;
        const planes = try u.at.a.take(u.f, b, R);
        try u.f.tri.hcPost(b.x.ptr, b.x.ptr, planes.g, b.post.ptr, b.comb.ptr, planes.world, R, m.hidden, m.streams);
    }

    pub fn tailDraw(d: *Device, slot: u1, wins: []const dsv.Window, R: usize, out: []u32) !void {
        d.select(slot);
        var base: dsv.Dev = .{ .u = d.u };
        @memset(out, 0);
        try base.dsvTail(wins, R);
        try base.settle();
        var r0: usize = 0;
        for (wins) |w| {
            defer r0 += w.rows();
            if (w.row.cancelled()) continue;
            try d.alive();
            try base.dsvDraw(w, r0, out[r0 .. r0 + w.rows()]);
        }
    }

    pub fn commit(d: *Device, w: dsv.Window) !void {
        var base: dsv.Dev = .{ .u = d.u };
        try base.dsvCommit(w.row, 1, 1);
    }

    pub fn end(d: *Device) !void {
        if (!d.turn.done()) return error.OutOfStep;
        try ae.Attn.settle(d.u.f, d.u.at);
        d.u.ping_pong.?.last = d.windows[1].seq;
        // Prefill pairs may run next: their receive-slot tail barrier must see this pair too.
        if (d.u.at.pairs) |prompt_pairs| prompt_pairs.last = d.windows[1].seq;
        if (d.u.at.lanes) |prompt_lanes| prompt_lanes.last = d.windows[1].seq;
        d.u.ping_pong.?.runs += 1;
        // Two tails are one logical user round, not two allocator rounds.
        d.u.rounds -= 1;
        d.u.dsvs.?.counts.rounds -= 1;
        d.restore();
    }

    pub fn fail(d: *Device) void {
        const u = d.u;
        u.at.a.dead = true;
        if (u.at.a.wire) |w| w.node.fail(.{ .why = .stopped, .peer = 0, .seq = w.node.doorbell() });
        if (u.f.gpu) |g| g.stream.synchronize() catch {};
        u.at.pending = false;
        u.together = false;
        d.restore();
    }
};

pub fn round(u: *Users, wins: []const dsv.Window, out: []u32) !void {
    const p = try pp.split(wins);
    if (out.len != p.rows) return error.Rows;
    if (wins.len == 1) return dsv.round(u, wins, out);
    const v = u.dsvs orelse return error.UsersSharedDraftsOff;
    for (wins) |w| {
        if (w.row.key.slot >= u.count or u.generation[w.row.key.slot] != w.row.key.generation) return error.StaleGeneration;
        const st = if (w.row.key.slot == 0) &u.at.e.st else &u.states[w.row.key.slot - 1];
        if (st.pos != w.row.position) return error.PositionMismatch;
        if (v.slot[w.row.key.slot].pending != 0) return error.Unsettled;
        try @import("cuda_compute.zig").checkRoom(u.at.e.m, st, w.rows(), null);
    }
    var device: Device = .{ .u = u };
    try pp.run(Device, &device, wins, out);
}
