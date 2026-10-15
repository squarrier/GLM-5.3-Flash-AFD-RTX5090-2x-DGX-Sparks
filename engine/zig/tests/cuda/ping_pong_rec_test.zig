//! ping_pong real Users launch/State/wire tests, reusing shared_verify's recorder; no CUDA numerical claim.
//! Inherited authors Ash Hart, @Bizuayeu and Mia; GPT-6.1 Sol (OpenAI Codex), via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const pp = glm.afd_users_pp;
const dsv = glm.afd_users_dsv;
const Rec = @import("shared_verify_rec_test.zig").Rec;
const t = std.testing;
const gpa = t.allocator;
const Range = struct { first: usize, last: usize };

const Watch = struct {
    device: glm.afd_users_pp_device.Device,
    ranges: [2][40]Range = undefined,
    n: [2]usize = @splat(0),
    group: [4]u1 = @splat(0),
    max_pending: usize = 0,
    behind: bool = false,
    fn mark(w: *Watch) !usize {
        try w.device.u.f.flush();
        return w.device.u.f.rec.?.events.items.len;
    }
    fn save(w: *Watch, slot: u1, from: usize) !void {
        w.ranges[slot][w.n[slot]] = .{ .first = from, .last = try w.mark() };
        w.n[slot] += 1;
    }
    pub fn alive(w: *Watch) !void {
        try w.device.alive();
    }
    pub fn fail(w: *Watch) void {
        w.device.fail();
    }
    pub fn begin(w: *Watch, plan: *const pp.Plan) !bool {
        for (plan.groups, 0..) |g, slot| for (g.wins[0..g.len]) |win| {
            w.group[win.row.key.slot] = @intCast(slot);
        };
        w.behind = try w.device.begin(plan);
        return w.behind;
    }
    pub fn open(w: *Watch, slot: u1, wins: []const dsv.Window, R: usize) !void {
        const from = try w.mark();
        try w.device.open(slot, wins, R);
        try w.save(slot, from);
    }
    pub fn advance(w: *Watch, slot: u1, wins: []const dsv.Window, R: usize, li: *usize) !?usize {
        const from = try w.mark();
        const out = try w.device.advance(slot, wins, R, li);
        try w.save(slot, from);
        return out;
    }
    pub fn issue(w: *Watch, slot: u1, R: usize, li: usize) !void {
        const from = try w.mark();
        try w.device.issue(slot, R, li);
        w.max_pending = @max(w.max_pending, w.device.turn.next - w.device.turn.taken);
        try w.save(slot, from);
    }
    pub fn complete(w: *Watch, slot: u1, R: usize, li: usize) !void {
        const from = try w.mark();
        try w.device.complete(slot, R, li);
        try w.save(slot, from);
    }
    pub fn tailDraw(w: *Watch, slot: u1, wins: []const dsv.Window, R: usize, out: []u32) !void {
        const from = try w.mark();
        try w.device.tailDraw(slot, wins, R, out);
        try w.save(slot, from);
    }
    pub fn commit(w: *Watch, win: dsv.Window) !void {
        const from = try w.mark();
        try w.device.commit(win);
        try w.save(w.group[win.row.key.slot], from);
    }
    pub fn end(w: *Watch) !void {
        try w.device.end();
    }
    fn events(w: *Watch, slot: u1) !std.ArrayList(glm.trace.Event) {
        var out: std.ArrayList(glm.trace.Event) = .empty;
        errdefer out.deinit(gpa);
        const ev = w.device.u.f.rec.?.events.items;
        for (w.ranges[slot][0..w.n[slot]]) |r| try out.appendSlice(gpa, ev[r.first..r.last]);
        return out;
    }
};

fn names(r: *Rec, slot: u1) !glm.trace.Space {
    var out: glm.trace.Space = .{ .gpa = gpa };
    errdefer out.deinit();
    if (slot == 1) for (r.space.entries.items) |e| {
        const prefix = "users.ping.";
        if (!std.mem.startsWith(u8, e.path, prefix)) continue;
        const suffix = e.path[prefix.len..];
        if (std.mem.eql(u8, suffix, "hdr")) continue;
        var path: [64]u8 = undefined;
        try out.add(try std.fmt.bufPrint(&path, "b.{s}", .{suffix}), e.buf);
    };
    for (r.space.entries.items) |e| {
        var path: [256]u8 = undefined;
        var size: usize = 0;
        var segs = std.mem.splitScalar(u8, e.path, '.');
        while (segs.next()) |part| {
            const numeric = part.len > 0 and for (part) |c| {
                if (!std.ascii.isDigit(c)) break false;
            } else true;
            if (std.mem.startsWith(u8, e.path, "tmp.") and numeric) continue;
            if (size > 0) {
                path[size] = '.';
                size += 1;
            }
            @memcpy(path[size..][0..part.len], part);
            size += part.len;
        }
        try out.add(path[0..size], e.buf);
    }
    return out;
}

fn enable(r: *Rec) !void {
    r.a.paired = true;
    r.a.inflight = if (r.a.lanes == 0) 2 else r.a.lanes;
    r.node.cfg.settings.pairs = true;
    r.node.cfg.settings.inflight = r.a.inflight;
    try r.users.usePingPong();
    try glm.views.walk(&r.users.ping_pong.?.c, glm.views.Path.init("users.ping"), &r.space, glm.trace.placeAll);
}

fn state(r: *Rec, key: u8) *glm.buffers.State {
    return if (key == 0) &r.e.st else &r.users.states[key - 1];
}
const drafts = [_]u32{ 101, 102, 103 };
fn window(r: *Rec, slot: u8, width: usize) dsv.Window {
    return .{ .row = .{ .key = .{ .slot = slot, .generation = 1 }, .token = 100, .position = state(r, slot).pos }, .drafts = drafts[0 .. width - 1] };
}

const md = glm.sample.mask_dev;
const MaskChoice = struct {
    fn candidate(_: *anyopaque, x: *const md.Dev, _: glm.trace.T) !md.Candidate {
        // Recording fixture only: the real host resolver still validates staged eligibility.
        const p = try md.plan(x.width, x.host[0..x.words]);
        return .{ .id = p.first, .bits = @bitCast(@as(f32, 1)) };
    }
};

fn compare(n: u8, widths: []const usize) !void {
    return compareDraws(n, widths, false);
}

fn compareDraws(n: u8, widths: []const usize, structured: bool) !void {
    var together: Rec = undefined;
    try together.init(n, .drafted);
    defer together.deinit();
    try enable(&together);
    var masked = try md.Dev.recorded(gpa, &together.m, &together.space);
    defer masked.close();
    masked.emulate = .{ .ctx = &together, .candidate = MaskChoice.candidate };
    together.e.smp.masked_dev = &masked;
    const masks = try gpa.alloc(u32, 4 * masked.words);
    defer gpa.free(masks);
    @memset(masks, 0);
    var wins: [4]dsv.Window = undefined;
    for (widths, 0..) |width, i| {
        wins[i] = window(&together, @intCast(i), width);
        if (structured and width == 1) {
            const mask = masks[i * masked.words .. (i + 1) * masked.words];
            mask[0] = @as(u32, 1) << @as(u5, @intCast(4 + i));
            wins[i].row.mask = mask;
        }
    }
    const plan = try pp.split(wins[0..n]);
    const base_ptr = together.e.buf.x.ptr;
    var watch: Watch = .{ .device = .{ .u = &together.users } };
    var out: [16]u32 = undefined;
    try pp.run(Watch, &watch, wins[0..n], out[0..plan.rows]);
    try t.expectEqual(base_ptr, together.e.buf.x.ptr);
    try t.expect(together.a.pair == null and watch.device.turn.done());
    try t.expectEqual(@as(usize, 4), together.posts); // cut5: two MoE positions, A0 B0 A1 B1
    for (together.headers[0..together.posts], 0..) |h, i| {
        const slot = i % 2;
        try t.expectEqual(cuda.xchg.proto.Mode.pair, h.mode);
        try t.expectEqual(@as(i32, @intCast(slot)), h.extra[0]);
        try t.expectEqual(@as(i32, @intCast(plan.groups[slot].rows)), h.rows);
        try t.expect(h.rows <= 8);
        try t.expectEqual(@as(i32, @intCast(i / 2)), h.layer);
        try t.expectEqual(@as(i32, @intCast(plan.groups[1 - slot].rows)), h.extra[1]);
    }
    const expert_pair = try glm.afd_expert.pairs.open(together.headers[0], @intCast(together.a.layers), together.a.positions(), 8);
    _ = try glm.afd_expert.pairs.partner(expert_pair, together.headers[1], @intCast(together.a.layers));
    try glm.afd_expert.pairs.tail(expert_pair, together.headers[3], @intCast(together.a.layers));
    var launches: usize = 0;
    for (plan.groups, 0..) |g, slot| {
        var solo: Rec = undefined;
        try solo.init(n, .drafted);
        defer solo.deinit();
        var solo_mask = try md.Dev.recorded(gpa, &solo.m, &solo.space);
        defer solo_mask.close();
        solo_mask.emulate = .{ .ctx = &solo, .candidate = MaskChoice.candidate };
        solo.e.smp.masked_dev = &solo_mask;
        var solo_out: [8]u32 = undefined;
        try solo.users.roundDrafted(g.wins[0..g.len], solo_out[0..g.rows]);
        try solo.f.flush();
        var a = try watch.events(@intCast(slot));
        defer a.deinit(gpa);
        var sa = try names(&together, @intCast(slot));
        defer sa.deinit();
        var sb = try names(&solo, 0);
        defer sb.deinit();
        try glm.serve_rec.expectSame(&sa, a.items, &sb, solo.trace.events.items);
        launches += a.items.len;
        for (g.wins[0..g.len]) |w| {
            const aa = state(&together, w.row.key.slot);
            const bb = state(&solo, w.row.key.slot);
            try t.expectEqual(bb.pos, aa.pos);
            try t.expectEqual(bb.parity, aa.parity);
        }
    }
    if (structured) {
        try t.expectEqual(@as(usize, 2), masked.stats.masked);
        try t.expectEqual(@as(u32, 5), out[4]);
        try t.expectEqual(@as(u32, 7), out[7]);
    }
    std.debug.print("ping_pong recorded C{d}: {d} per-micro-batch launches exact against alone, mode7/FIFO4/4, independent State positions/parities; <=8 rows\n", .{ n, launches });
}

test "ping_pong recorded real Users schema-mask adapter staged per row beside drafted windows, exact against alone" {
    try compareDraws(4, &.{ 4, 1, 2, 1 }, true);
}

test "ping_pong recorded real Users C2 drafted micro-batches exact against alone" {
    try compare(2, &.{ 4, 3 });
}
test "ping_pong recorded real Users C4 16 logical rows, no 9-16 row kernel path" {
    try compare(4, &.{ 4, 4, 4, 4 });
}
test "ping_pong recorded real Users uneven dense/sparse windows and one-row commit" {
    try compare(4, &.{ 4, 1, 2, 1 });
}

test "ping_pong recorded cross-scheduler pair tail barrier survives subsequent decode pair" {
    var r: Rec = undefined;
    try r.init(2, .drafted);
    defer r.deinit();
    try enable(&r);
    var prompt_pairs = glm.afd_pairs.Pairs.init(&r.at);
    r.at.pairs = &prompt_pairs;
    const base = r.e.buf.x.ptr;
    var out: [2]u32 = undefined;
    for (0..2) |_| {
        const wins = [_]dsv.Window{ window(&r, 0, 1), window(&r, 1, 1) };
        try r.users.roundDrafted(&wins, &out);
        try t.expectEqual(@as(?u32, r.a.seq), prompt_pairs.last);
        try t.expectEqual(prompt_pairs.last, r.users.ping_pong.?.last);
        try t.expectEqual(base, r.e.buf.x.ptr);
        try t.expect(r.a.pair == null);
    }
    try t.expectEqual(@as(u64, 2), r.users.ping_pong.?.runs);
    try t.expectEqual(@as(usize, 8), r.posts);
}

test "ping_pong with batched_drafts recorded real Users drains before batched MTP, private rollback and next pair" {
    const bd = glm.afd_users_bd;
    const Picks = @import("batched_drafts_picker.zig").Picks;
    var r: Rec = undefined;
    try r.init(4, .prompt);
    defer r.deinit();
    try enable(&r);
    try r.users.useBatchedDrafts();
    try r.space.place("users.mtp_hidden", &r.users.bd.?.hidden);
    var picks: Picks = .{ .r = &r };
    r.trace.picker = .{ .ctx = &picks, .pick = Picks.pick };
    var first: [4]u32 = undefined;
    const ids = [_]u32{ 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009 };
    var drafted: [4][3]u32 = undefined;
    var requests: [4]bd.Request = undefined;
    for (0..4) |i| {
        const slot: u8 = @intCast(i);
        const key: glm.afd_users_round.Key = .{ .slot = slot, .generation = 1 };
        r.users.track(slot, .seed);
        first[i] = try r.users.prefill(key, ids[0 .. 7 + i]);
        requests[i] = .{ .row = .{ .key = key, .token = 0, .position = state(&r, slot).pos }, .follow = first[i .. i + 1], .verified = false, .position = state(&r, slot).pos + 1, .out = &drafted[i] };
    }
    try r.users.draftsBatched(&requests);
    const original = r.e.buf.x.ptr;
    var wins: [4]dsv.Window = undefined;
    for (0..4) |i| wins[i] = .{ .row = .{ .key = requests[i].row.key, .token = first[i], .position = state(&r, @intCast(i)).pos }, .drafts = &drafted[i] };
    var verified: [16]u32 = undefined;
    const before = r.posts;
    try r.users.roundDrafted(&wins, &verified);
    try t.expectEqual(@as(usize, 4), r.posts - before);
    try t.expectEqual(original, r.e.buf.x.ptr);
    try t.expect(r.a.pair == null and !r.at.pending);
    const kept = [_]usize{ 2, 4, 3, 1 };
    for (kept, 0..) |keep, i| {
        const slot: u8 = @intCast(i);
        try r.users.settleDrafted(requests[i].row.key, 4, keep);
        requests[i] = .{ .row = .{ .key = requests[i].row.key, .token = 0, .position = state(&r, slot).pos }, .follow = verified[i * 4 .. i * 4 + keep], .verified = true, .position = state(&r, slot).pos + 1, .out = &drafted[i] };
    }
    const after_pair = r.posts;
    try r.users.draftsBatched(&requests);
    // Kept prefixes total10 rows: bounded absorb waves7+3, then the two later draft depths4+4.
    try t.expectEqual(@as(usize, 4), r.posts - after_pair);
    for ([_]i32{ 7, 3, 4, 4 }, 0..) |rows, i| try t.expectEqual(rows, r.hdrs[after_pair + i][1]);
    try t.expectEqual(@as(u64, 7), r.users.bd.?.forwards);
    try t.expectEqual(@as(u64, 1), r.users.ping_pong.?.runs);
    for (0..4) |i| {
        try t.expectEqual(@as(usize, 0), r.users.dsvs.?.slot[i].pending);
        wins[i] = .{ .row = .{ .key = requests[i].row.key, .token = verified[i * 4], .position = state(&r, @intCast(i)).pos }, .drafts = &drafted[i] };
    }
    try r.users.roundDrafted(&wins, &verified);
    try t.expectEqual(@as(u64, 2), r.users.ping_pong.?.runs);
    try t.expectEqual(original, r.e.buf.x.ptr);
}

test "ping_pong recorded canonical lanes3/4 prompt-decode-prompt-decode preserves wire profile and two-credit drain" {
    for ([_]u8{ 3, 4 }) |n| {
        for ([_]bool{ false, true }) |absorb| {
            var r: Rec = undefined;
            try r.initLanes(4, .drafted, n);
            defer r.deinit();
            r.at.absorb = absorb;
            try enable(&r);
            var prompt_lanes = try glm.afd_lanes.Lanes.init(&r.at);
            defer prompt_lanes.detach();
            prompt_lanes.agreed = true;
            try prompt_lanes.walk(&r.space, glm.trace.placeAll);
            r.at.lanes = &prompt_lanes;
            const base = r.e.buf.x.ptr;
            const initial_settings = r.node.cfg.settings;
            var out: [2]u32 = undefined;
            var ids: [256]u32 = undefined;
            for (&ids, 0..) |*id, i| id.* = @intCast(1000 + i);
            var start: usize = 0;
            _ = try prompt_lanes.step(&r.at, &r.f, ids[0 .. @as(usize, n) * r.e.prefill_rows], &start, true);
            const first = [_]dsv.Window{ window(&r, 1, 1), window(&r, 2, 1) };
            var w: Watch = .{ .device = .{ .u = &r.users } };
            const first_barrier = prompt_lanes.last != null and prompt_lanes.last.? == r.a.seq;
            try pp.run(Watch, &w, &first, &out);
            try t.expectEqual(first_barrier, w.behind);
            try t.expectEqual(@as(usize, 2), w.max_pending);
            try t.expectEqual(@as(usize, 2), w.device.turn.most);
            try t.expectEqual(@as(?u32, r.a.seq), prompt_lanes.last);
            start = 0;
            const before_prompt = r.posts;
            const token = try prompt_lanes.step(&r.at, &r.f, ids[0 .. @as(usize, n) * r.e.prefill_rows], &start, true);
            try t.expect(token != null and prompt_lanes.runs == 2);
            for (r.headers[before_prompt..r.posts]) |header| {
                if (header.mode != .lanes) {
                    try t.expect(absorb);
                    try t.expectEqual(cuda.xchg.proto.Mode.slice, header.mode);
                    try t.expectEqual(@as(i32, @intCast(r.a.layers)), header.layer);
                }
            }
            try t.expect(r.a.lane == null and r.a.pair == null);
            const second = [_]dsv.Window{ window(&r, 1, 1), window(&r, 2, 1) };
            var v: Watch = .{ .device = .{ .u = &r.users } };
            const second_barrier = prompt_lanes.last != null and prompt_lanes.last.? == r.a.seq;
            try pp.run(Watch, &v, &second, &out);
            try t.expectEqual(second_barrier, v.behind);
            try t.expect(v.device.turn.done());
            try t.expectEqual(@as(usize, 2), v.max_pending);
            try t.expectEqual(@as(usize, 2), v.device.turn.most);
            try t.expectEqual(@as(?u32, r.a.seq), prompt_lanes.last);
            try t.expectEqual(base, r.e.buf.x.ptr);
            try t.expectEqualDeep(initial_settings, r.node.cfg.settings);
            try t.expectEqual(@as(usize, n), r.node.reg.slots);
            try t.expectEqual(n, r.a.lanes);
            try t.expectEqual(n, r.a.inflight);
            try t.expect(r.a.lane == null and r.a.pair == null);
        }
    }
}

const Fault = struct {
    posts: usize = 0,
    fail_at: ?usize = null,
    flag: ?*bool = null,
    fn post(ctx: *anyopaque, _: *cuda.xchg.Node, _: *const [cuda.xchg.proto.HDR_BYTES]u8, _: usize, _: usize) !void {
        const f: *Fault = @ptrCast(@alignCast(ctx));
        f.posts += 1;
        if (f.flag) |flag| if (f.posts == 1) {
            flag.* = true;
        };
        if (f.fail_at) |at| if (f.posts == at) return error.InjectedFailure;
    }
    fn take(_: *anyopaque, _: *cuda.xchg.Node, _: usize, _: usize) !void {}
};
fn cancel(ctx: ?*anyopaque) bool {
    return @as(*bool, @ptrCast(@alignCast(ctx.?))).*;
}

test "ping_pong recorded real Users failure with A0 issued restores buffer/provider and latches before any commit" {
    var r: Rec = undefined;
    try r.init(2, .drafted);
    defer r.deinit();
    try enable(&r);
    const x = r.e.buf.x.ptr;
    const hdr = r.a.w.hdr.ptr;
    var fault: Fault = .{ .fail_at = 2 };
    r.a.wire.?.host = .{ .ctx = &fault, .post = Fault.post, .take = Fault.take };
    const wins = [_]dsv.Window{ window(&r, 0, 1), window(&r, 1, 1) };
    var out = [_]u32{ 99, 99 };
    try t.expectError(error.InjectedFailure, r.users.roundDrafted(&wins, &out));
    try t.expectEqualSlices(u32, &.{ 0, 0 }, &out);
    try t.expect(r.a.dead and r.a.pair == null);
    try t.expectEqual(x, r.e.buf.x.ptr);
    try t.expectEqual(hdr, r.a.w.hdr.ptr);
    for (wins) |w| try t.expectEqual(w.row.position, state(&r, w.row.key.slot).pos);
    try t.expectError(error.ExpertFailed, r.users.roundDrafted(&wins, &out));
    try t.expectEqual(@as(usize, 2), fault.posts);
}

test "ping_pong recorded real Users cancel with A0 issued drains pair, never commits cancelled slot" {
    var r: Rec = undefined;
    try r.init(2, .drafted);
    defer r.deinit();
    try enable(&r);
    var flag = false;
    var fault: Fault = .{ .flag = &flag };
    r.a.wire.?.host = .{ .ctx = &fault, .post = Fault.post, .take = Fault.take };
    var wins = [_]dsv.Window{ window(&r, 0, 1), window(&r, 1, 1) };
    wins[1].row.cancel_ctx = &flag;
    wins[1].row.cancel_check = cancel;
    var out: [2]u32 = undefined;
    try r.users.roundDrafted(&wins, &out);
    try t.expectEqual(wins[0].row.position + 1, state(&r, 0).pos);
    try t.expectEqual(wins[1].row.position, state(&r, 1).pos);
    try t.expectEqual(@as(usize, 4), fault.posts);
    try t.expect(!r.a.dead and r.a.pair == null);
}
