//! batched_drafts uneven absorb waves and cache/zero-first ownership. CPU recorder, no CUDA arithmetic.
const std = @import("std");
const glm = @import("glm_cuda");
const rec = @import("shared_verify_rec_test.zig");
const bd = glm.afd_users_bd;
const t = std.testing;
const eq = t.expectEqual;

fn state(r: *rec.Rec, i: usize) *glm.buffers.State {
    return if (i == 0) &r.e.st else &r.users.states[i - 1];
}
const Pick = @import("batched_drafts_picker.zig").Picks;

test "batched_drafts uneven backlogs 1/8/9/17: bounded waves, correct finished tails survive later waves, trimmed lengths and zero depth" {
    var r: rec.Rec = undefined;
    try r.init(4, .drafted);
    defer r.deinit();
    try r.users.useBatchedDrafts();
    try r.space.place("users.mtp_hidden", &r.users.bd.?.hidden);
    var picks: Pick = .{ .r = &r };
    r.trace.picker = .{ .ctx = &picks, .pick = Pick.pick };
    const owed = [_]usize{ 1, 8, 9, 17 };
    const depths = [_]usize{ 3, 2, 0, 1 };
    var rs: [4]bd.Request = undefined;
    var outputs: [4][3]u32 = @splat(@splat(0));
    for (0..4) |i| {
        const s = state(&r, i);
        s.pos = 50 + i;
        s.mtp_len = s.pos - owed[i] + 2;
        s.mtp_drafted = 2;
        r.users.track(@intCast(i), .seed);
        const b = &r.users.drafts.?.backlog[i];
        b.n = owed[i];
        for (b.tokens[0..b.n], 0..) |*v, j| v.* = @intCast(20 + 10 * i + j);
        rs[i] = .{ .row = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = 0, .position = s.pos }, .follow = b.tokens[b.n - 1 .. b.n], .verified = false, .position = s.pos + 1, .out = outputs[i][0..depths[i]] };
    }
    const before = r.posts;
    try r.users.draftsBatched(&rs);
    try r.f.flush();
    for (r.hdrs[before..r.posts]) |h| try t.expect(h[1] > 0 and h[1] <= 8);
    try t.expect(r.posts - before >= 7);
    for (0..4) |i| {
        const s = state(&r, i);
        try eq(s.pos, s.mtp_len - s.mtp_drafted);
        try eq(if (depths[i] > 0) depths[i] - 1 else 0, s.mtp_drafted);
        var expected: u32 = @intCast(20 + 10 * i + owed[i] - 1);
        for (outputs[i][0..depths[i]]) |v| {
            expected = Pick.token(expected);
            try eq(expected, v);
        }
    }
    var stash_sources: usize = 0;
    var backbone_sources: usize = 0;
    for (r.trace.events.items) |event| {
        if (!std.mem.eql(u8, event.name, "torch.copy_") or event.vals.len != 2) continue;
        const dst = event.vals[0].t;
        const src = event.vals[1].t;
        if (dst.ptr >= r.users.bd.?.hidden.ptr and dst.ptr < r.users.bd.?.hidden.ptr + r.users.bd.?.hidden.bytes()) {
            try t.expect(src.ptr >= r.e.mbuf.?.fnormed.ptr and src.ptr < r.e.mbuf.?.fnormed.ptr + r.e.mbuf.?.fnormed.bytes());
            stash_sources += 1;
        }
        for (r.users.drafts.?.backlog, 0..) |b, i| if (dst.ptr == b.hidden.ptr and src.ptr == b.hidden.ptr + (owed[i] - 1) * r.m.hidden * 2 and owed[i] > 1) {
            backbone_sources += 1;
        };
    }
    try eq(@as(usize, 6), stash_sources);
    try eq(@as(usize, 3), backbone_sources);
    std.debug.print("batched_drafts uneven backlog: 35 absorbed rows, <=8 per forward, early-completed stash survives; retained backbone pairs distinct; zero-depth stream no drafts\n", .{});
}

test "batched_drafts zero-first per segment: empty cache at nonzero packed offset, no zero on later depth" {
    var r: rec.Rec = undefined;
    try r.init(2, .drafted);
    defer r.deinit();
    try r.users.useBatchedDrafts();
    try r.space.place("users.mtp_hidden", &r.users.bd.?.hidden);
    var out: [2][3]u32 = @splat(@splat(0));
    var rs: [2]bd.Request = undefined;
    for (0..2) |i| {
        const s = state(&r, i);
        s.pos = if (i == 0) 10 else 1;
        s.mtp_len = s.pos - 1;
        r.users.track(@intCast(i), .seed);
        const b = &r.users.drafts.?.backlog[i];
        b.n = 1;
        b.tokens[0] = @intCast(40 + i);
        rs[i] = .{ .row = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = 0, .position = s.pos }, .follow = b.tokens[0..1], .verified = false, .position = s.pos + 1, .out = &out[i] };
    }
    try r.f.flush();
    r.trace.reset();
    try r.users.draftsBatched(&rs);
    try r.f.flush();
    var zeros: usize = 0;
    for (r.trace.events.items) |event| if (std.mem.eql(u8, event.name, "torch.zero_")) {
        const tensor = event.vals[0].t;
        if (tensor.ptr >= r.e.mbuf.?.me.ptr and tensor.ptr < r.e.mbuf.?.me.ptr + r.e.mbuf.?.me.bytes()) {
            try eq(r.e.mbuf.?.me.ptr + r.m.hidden * 2, tensor.ptr);
            zeros += 1;
        }
    };
    try eq(@as(usize, 1), zeros);
}
