//! batched_drafts tests. Recorder tests establish launch/state ownership, not CUDA numerical equality.
//! GPT-6.1 Sol (OpenAI Codex), via Hermes; inherited shared_verify recorder credits retained.
const std = @import("std");
const glm = @import("glm_cuda");
const served = @import("afd_attn_serve.zig");
const rec = @import("shared_verify_rec_test.zig");
const bd = glm.afd_users_bd;
const t = std.testing;
const eq = t.expectEqual;
const gpa = t.allocator;

test {
    _ = @import("shared_verify_test.zig");
    _ = @import("batched_drafts_waves_test.zig");
    _ = @import("batched_drafts_state_test.zig");
    _ = @import("batched_drafts_plan_test.zig");
}
test "batched_drafts host C2/C4 pool + interleave output identity and rollback" {
    try @import("shared_verify_test.zig").checkBatchedHost();
}
test "batched_drafts host lone/shared transitions" {
    try @import("shared_verify_test.zig").checkBatchedTransition();
}
test "batched_drafts host cancellation and failure mid-draft" {
    try @import("shared_verify_test.zig").checkBatchedCancelFailure();
}

test "batched_drafts new plan switch exists, default off" {
    try t.expect(@hasField(served.Plan, "users_batched_drafts"));
    try t.expect(!try bd.parse(null, false));
    try t.expect(!try bd.parse("0", false));
    try t.expect(try bd.parse("1", true));
    try t.expectError(error.UsersBatchedDraftsNeedsSharedDrafts, bd.parse("1", false));
    for ([_][]const u8{ "", "2", "on", "true", "01", " 1", "1 " }) |x| try t.expectError(error.BadAfdUsersBatchedDrafts, bd.parse(x, true));
}

const Picks = @import("batched_drafts_picker.zig").Picks;

fn st(r: *rec.Rec, i: usize) *glm.buffers.State {
    return if (i == 0) &r.e.st else &r.users.states[i - 1];
}

fn enable(r: *rec.Rec) !void {
    try r.users.useBatchedDrafts();
    try r.space.place("users.mtp_hidden", &r.users.bd.?.hidden);
}

fn seed(r: *rec.Rec, n: u8, first: *[4]u32) !void {
    const ids = [_]u32{ 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009 };
    for (0..n) |i| {
        r.users.track(@intCast(i), .seed);
        first[i] = try r.users.prefill(.{ .slot = @intCast(i), .generation = 1 }, ids[0 .. 7 + i]);
    }
    try r.f.flush();
    r.trace.reset();
}

test "batched_drafts recorded C2/C4: one MTP forward per active depth, draft tokens = per-stream keyed row-invariant fixture, own lengths and stash" {
    var equal: usize = 0;
    for ([_]u8{ 2, 4 }) |n| {
        var a: rec.Rec = undefined;
        try a.init(n, .prompt);
        defer a.deinit();
        var b: rec.Rec = undefined;
        try b.init(n, .prompt);
        defer b.deinit();
        try enable(&a);
        var pa: Picks = .{ .r = &a };
        var pb: Picks = .{ .r = &b };
        a.trace.picker = .{ .ctx = &pa, .pick = Picks.pick };
        b.trace.picker = .{ .ctx = &pb, .pick = Picks.pick };
        var fa: [4]u32 = undefined;
        var fb: [4]u32 = undefined;
        try seed(&a, n, &fa);
        try seed(&b, n, &fb);
        var oa: [4][3]u32 = @splat(@splat(0));
        var ob: [4][3]u32 = @splat(@splat(0));
        const depths = [_]usize{ 3, 1, 0, 2 };
        var rs: [4]bd.Request = undefined;
        const before_a = a.posts;
        const before_b = b.posts;
        for (0..n) |i| {
            const key: glm.afd_users_round.Key = .{ .slot = @intCast(i), .generation = 1 };
            rs[n - 1 - i] = .{ .row = .{ .key = key, .token = 0, .position = st(&a, i).pos }, .follow = fa[i .. i + 1], .verified = false, .position = st(&a, i).pos + 1, .out = oa[i][0..depths[i]] };
            try b.users.draftDrafted(key, fb[i .. i + 1], false, st(&b, i).pos + 1, ob[i][0..depths[i]]);
        }
        try a.users.draftsBatched(rs[0..n]);
        try eq(@as(usize, 3), a.posts - before_a);
        try t.expect(b.posts - before_b > a.posts - before_a);
        try eq(@as(i32, n), a.hdrs[before_a][1]);
        try eq(@as(i32, if (n == 4) 2 else 1), a.hdrs[before_a + 1][1]);
        try eq(@as(i32, 1), a.hdrs[before_a + 2][1]);
        for (0..n) |i| {
            try t.expectEqualSlices(u32, ob[i][0..depths[i]], oa[i][0..depths[i]]);
            try eq(st(&b, i).mtp_len, st(&a, i).mtp_len);
            try eq(st(&b, i).mtp_drafted, st(&a, i).mtp_drafted);
            try eq(st(&a, i).pos, st(&a, i).mtp_len - st(&a, i).mtp_drafted);
            try eq(@as(usize, 1), a.users.drafts.?.backlog[i].n);
            try t.expect(a.users.dsvs.?.slot[i].in_head);
            equal += depths[i];
        }
        try eq(@as(u64, 3), a.users.bd.?.forwards);
        try eq(@as(u64, if (n == 4) 32768 else 16384), bd.bytes(&a.m, n, true));
    }
    std.debug.print("batched_drafts recorded: batched = per-stream drafts {d}/{d} keyed row-invariant fixture; C2/C4 headers per depth exact; CUDA numerics NOT RUN\n", .{ equal, equal });
}

test "batched_drafts preflight: invalid final member and duplicate leave every backlog/length/exchange unchanged" {
    var r: rec.Rec = undefined;
    try r.init(2, .prompt);
    defer r.deinit();
    try enable(&r);
    var first: [4]u32 = undefined;
    try seed(&r, 2, &first);
    var out: [2][3]u32 = @splat(@splat(19));
    var rs: [2]bd.Request = undefined;
    for (0..2) |i| rs[i] = .{ .row = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = 0, .position = st(&r, i).pos }, .follow = first[i .. i + 1], .verified = false, .position = st(&r, i).pos + 1, .out = &out[i] };
    const posts = r.posts;
    const len = st(&r, 0).mtp_len;
    rs[1].position += 1;
    try t.expectError(error.PositionMismatch, r.users.draftsBatched(&rs));
    try eq(posts, r.posts);
    try eq(len, st(&r, 0).mtp_len);
    try eq(@as(usize, 1), r.users.drafts.?.backlog[0].n);
    try t.expect(!r.a.dead);
    rs[1] = rs[0];
    try t.expectError(error.DuplicateWindow, r.users.draftsBatched(&rs));
    try eq(posts, r.posts);
}

test "batched_drafts mid-draft failure: no partial publication, dead backend, backlogs invalidated" {
    var r: rec.Rec = undefined;
    try r.init(2, .prompt);
    defer r.deinit();
    try enable(&r);
    var picks: Picks = .{ .r = &r };
    r.trace.picker = .{ .ctx = &picks, .pick = Picks.pick };
    var first: [4]u32 = undefined;
    try seed(&r, 2, &first);
    picks.fail_at = picks.draws + 2;
    var out: [2][3]u32 = @splat(@splat(19));
    var rs: [2]bd.Request = undefined;
    for (0..2) |i| rs[i] = .{ .row = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = 0, .position = st(&r, i).pos }, .follow = first[i .. i + 1], .verified = false, .position = st(&r, i).pos + 1, .out = &out[i] };
    try t.expectError(error.DraftFault, r.users.draftsBatched(&rs));
    try t.expect(r.a.dead);
    for (out) |o| try t.expectEqualSlices(u32, &.{ 0, 0, 0 }, &o);
    const posts = r.posts;
    try t.expectError(error.ExpertFailed, r.users.draftsBatched(&rs));
    try eq(posts, r.posts);
    try eq(@as(usize, 0), r.users.drafts.?.backlog[0].n);
}

test "batched_drafts cancellation during depth: drains cohort, no publication for cancelled stream, survivor continues" {
    var r: rec.Rec = undefined;
    try r.init(2, .prompt);
    defer r.deinit();
    try enable(&r);
    var picks: Picks = .{ .r = &r };
    r.trace.picker = .{ .ctx = &picks, .pick = Picks.pick };
    var first: [4]u32 = undefined;
    try seed(&r, 2, &first);
    picks.cancel_at = picks.draws;
    var out: [2][3]u32 = @splat(@splat(19));
    var rs: [2]bd.Request = undefined;
    for (0..2) |i| rs[i] = .{ .row = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = 0, .position = st(&r, i).pos, .cancel_ctx = if (i == 0) &picks else null, .cancel_check = if (i == 0) Picks.cancelled else null }, .follow = first[i .. i + 1], .verified = false, .position = st(&r, i).pos + 1, .out = &out[i] };
    const before = r.posts;
    try r.users.draftsBatched(&rs);
    try t.expect(picks.flag and !r.a.dead);
    try t.expectEqualSlices(u32, &.{ 19, 19, 19 }, &out[0]);
    try t.expect(out[1][0] != 19 and out[1][1] != 19);
    try eq(@as(i32, 2), r.hdrs[before][1]);
    try eq(@as(i32, 1), r.hdrs[before + 1][1]);
    r.users.release(rs[0].row.key);
    try eq(@as(usize, 0), r.users.drafts.?.backlog[0].n);
}
