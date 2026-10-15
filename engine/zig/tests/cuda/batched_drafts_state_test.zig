//! batched_drafts verified-prefix backlog and real State/DSA ownership, CPU recorder only.
const std = @import("std");
const glm = @import("glm_cuda");
const rec = @import("shared_verify_rec_test.zig");
const Picks = @import("batched_drafts_picker.zig").Picks;
const bd = glm.afd_users_bd;
const t = std.testing;
const eq = t.expectEqual;
fn st(r: *rec.Rec, i: usize) *glm.buffers.State {
    return if (i == 0) &r.e.st else &r.users.states[i - 1];
}
const Never = struct {
    fn kept(_: *anyopaque, _: []const u32, _: usize, _: usize) !bool {
        return false;
    }
};

test "batched_drafts real verified-prefix rollback and lone resume: each MTP length matches its kept backbone rows" {
    var r: rec.Rec = undefined;
    try r.init(2, .prompt);
    defer r.deinit();
    try r.users.useBatchedDrafts();
    try r.space.place("users.mtp_hidden", &r.users.bd.?.hidden);
    var picks: Picks = .{ .r = &r };
    r.trace.picker = .{ .ctx = &picks, .pick = Picks.pick };
    const ids = [_]u32{ 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007 };
    var first: [2]u32 = undefined;
    var drafts: [2][3]u32 = undefined;
    var rs: [2]bd.Request = undefined;
    for (0..2) |i| {
        r.users.track(@intCast(i), .seed);
        first[i] = try r.users.prefill(.{ .slot = @intCast(i), .generation = 1 }, ids[0 .. 7 + i]);
        rs[i] = .{ .row = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = 0, .position = st(&r, i).pos }, .follow = first[i .. i + 1], .verified = false, .position = st(&r, i).pos + 1, .out = &drafts[i] };
    }
    try r.users.draftsBatched(&rs);
    var wins: [2]glm.afd_users_dsv.Window = undefined;
    for (0..2) |i| wins[i] = .{ .row = .{ .key = rs[i].row.key, .token = first[i], .position = st(&r, i).pos }, .drafts = &drafts[i] };
    var verified: [8]u32 = undefined;
    try r.users.roundDrafted(&wins, &verified);
    try r.users.settleDrafted(rs[0].row.key, 4, 2);
    try r.users.settleDrafted(rs[1].row.key, 4, 4);
    for (0..2) |i| rs[i] = .{ .row = .{ .key = rs[i].row.key, .token = 0, .position = st(&r, i).pos }, .follow = verified[i * 4 .. i * 4 + (if (i == 0) @as(usize, 2) else 4)], .verified = true, .position = st(&r, i).pos + 1, .out = &drafts[i] };
    try r.users.draftsBatched(&rs);
    for (0..2) |i| {
        try eq(st(&r, i).pos, st(&r, i).mtp_len - st(&r, i).mtp_drafted);
        try t.expect(r.users.dsvs.?.slot[i].in_head);
    }
    var output: [4]u32 = undefined;
    const hook: glm.afd_users.Users.DraftHook = .{ .ctx = &picks, .kept = Never.kept };
    picks.shared = false;
    _ = try r.users.drafted(rs[1].row.key, rs[1].follow[rs[1].follow.len - 1], 4, &output, &hook);
    try t.expect(r.users.dsvs.?.counts.unabsorbed > 0);
    try t.expect(!r.users.dsvs.?.slot[1].in_head);
    std.debug.print("batched_drafts real state: rollback keep 2/4 vs 4/4; MTP trimmed/reabsorbed per slot; lone resume unabsorbs its own start pair\n", .{});
}

test "batched_drafts DSA per segment: own MTP cache/position, dense+sparse mix and row offsets" {
    var r: rec.Rec = undefined;
    try r.init(4, .drafted);
    defer r.deinit();
    try r.users.useBatchedDrafts();
    try r.space.place("users.mtp_hidden", &r.users.bd.?.hidden);
    const pos = [_]usize{ 511, 513, 2048, 2053 };
    var rs: [4]bd.Request = undefined;
    var out: [4][1]u32 = undefined;
    for (0..4) |i| {
        const s = st(&r, i);
        s.pos = pos[i];
        s.mtp_len = pos[i] - 1;
        r.users.track(@intCast(i), .seed);
        const b = &r.users.drafts.?.backlog[i];
        b.n = 1;
        b.tokens[0] = @intCast(20 + i);
        rs[i] = .{ .row = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = 0, .position = s.pos }, .follow = b.tokens[0..1], .verified = false, .position = s.pos + 1, .out = &out[i] };
    }
    try r.f.flush();
    r.trace.reset();
    try r.users.draftsBatched(&rs);
    try r.f.flush();
    var writes: usize = 0;
    var dense: usize = 0;
    var sparse: usize = 0;
    for (r.trace.events.items) |event| {
        if (std.mem.eql(u8, event.name, "_lat_write")) {
            var lc: u64 = 0;
            var ps: u64 = 0;
            var lat: u64 = 0;
            for (event.args) |a| {
                if (std.mem.eql(u8, a.name, "LC")) lc = a.value.ptr.addr;
                if (std.mem.eql(u8, a.name, "POS")) ps = a.value.ptr.addr;
                if (std.mem.eql(u8, a.name, "LAT")) lat = a.value.ptr.addr;
            }
            const s = st(&r, writes);
            try eq(s.mtp_kc.?.ptr, lc);
            try eq(s.mtp_pos_dev.ptr, ps);
            try eq(r.e.mbuf.?.lat.ptr + writes * r.e.mbuf.?.lat.inner() * 2, lat);
            try eq(@as(u32, 1), event.grid[0]);
            writes += 1;
        }
        if (std.mem.eql(u8, event.name, "_dense_chunks")) dense += 1;
        if (std.mem.eql(u8, event.name, "_sparse_chunks")) sparse += 1;
    }
    try eq(@as(usize, 4), writes);
    try t.expect(dense >= 2);
    try eq(@as(usize, 1), sparse);
    std.debug.print("batched_drafts DSA ownership: 4/4 latent writes use own MTP cache/POS and packed row; dense/sparse per-segment options\n", .{});
}
