//! batched_drafts startup/forecast/default-off contract, actual Plan and real Users.
const std = @import("std");
const glm = @import("glm_cuda");
const served = @import("afd_attn_serve.zig");
const cli = @import("tf_glm_afd.zig");
const rec = @import("shared_verify_rec_test.zig");
const t = std.testing;
extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
extern "c" fn unsetenv([*:0]const u8) c_int;

test "batched_drafts actual plan: unset/0 inert, 1 needs shared drafted verify, users 2/4 pool/interleave and bad spellings" {
    const names = [_][*:0]const u8{ "TF_AFD_USERS", "TF_AFD_MTP", "TF_AFD_USERS_DRAFTS", "TF_AFD_USERS_SHARED", "TF_AFD_USERS_SHARED_DRAFTS", "TF_AFD_USERS_BATCHED_DRAFTS", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE", "TF_AFD_USERS_SAMPLING" };
    try @import("env_harness.zig").detach();
    var old: [names.len]?[:0]u8 = undefined;
    for (names, &old) |n, *o| {
        o.* = if (std.c.getenv(n)) |v| try t.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
        _ = unsetenv(n);
    }
    defer for (names, old) |n, o| {
        if (o) |v| {
            _ = setenv(n, v, 1);
            t.allocator.free(v);
        } else _ = unsetenv(n);
    };
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    _ = setenv("TF_AFD_MTP", "1", 1);
    _ = setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    _ = setenv("TF_AFD_USERS_SHARED", "1", 1);
    _ = setenv("TF_AFD_USERS_SHARED_DRAFTS", "1", 1);
    const cfg = "{\"max_position_embeddings\":1048576}";
    for ([_][*:0]const u8{ "2", "4" }) |n| {
        _ = setenv("TF_AFD_USERS", n, 1);
        _ = unsetenv("TF_AFD_USERS_BATCHED_DRAFTS");
        const flags = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/m", "--", "--drafts", "auto", "--parallel", std.mem.span(n) });
        var p = try served.plan(a, flags, cfg);
        try t.expect(!p.users_batched_drafts and p.stageRows() == 8);
        _ = setenv("TF_AFD_USERS_BATCHED_DRAFTS", "0", 1);
        p = try served.plan(a, flags, cfg);
        try t.expect(!p.users_batched_drafts);
        _ = setenv("TF_AFD_USERS_BATCHED_DRAFTS", "1", 1);
        _ = setenv("TF_AFD_POOL_TOKENS", "655360", 1);
        _ = setenv("TF_AFD_USERS_INTERLEAVE", "1", 1);
        p = try served.plan(a, flags, cfg);
        try t.expect(p.users_batched_drafts and p.users_shared_drafts and p.pool == 655360 and p.interleave != null and p.stageRows() == 8);
        _ = unsetenv("TF_AFD_USERS_SHARED_DRAFTS");
        try t.expectError(error.UsersBatchedDraftsNeedsSharedDrafts, served.plan(a, flags, cfg));
        _ = setenv("TF_AFD_USERS_SHARED_DRAFTS", "1", 1);
        for ([_][*:0]const u8{ "2", "on", "01", "true", "" }) |v| {
            _ = setenv("TF_AFD_USERS_BATCHED_DRAFTS", v, 1);
            try t.expectError(error.BadAfdUsersBatchedDrafts, served.plan(a, flags, cfg));
        }
    }
}

test "batched_drafts allocation and forecast: no bytes off, per-slot stash on only, no verify-buffer growth" {
    for ([_]u8{ 2, 4 }) |n| {
        var r: rec.Rec = undefined;
        try r.init(n, .drafted);
        defer r.deinit();
        const before = try r.users.bytes();
        const rows = r.e.buf.rows;
        const mrows = r.e.mbuf.?.rows;
        try t.expect(r.users.bd == null);
        try t.expectEqual(@as(u64, 0), glm.afd_users_bd.bytes(&r.m, n, false));
        try r.users.useBatchedDrafts();
        const after = try r.users.bytes();
        try t.expectEqual(before.total + glm.afd_users_bd.bytes(&r.m, n, true), after.total);
        try t.expectEqual(rows, r.e.buf.rows);
        try t.expectEqual(mrows, r.e.mbuf.?.rows);
        var forecast = before;
        @import("afd_attn_users.zig").withBatchedDrafts(&forecast, &r.m, n, false);
        try t.expectEqualDeep(before, forecast);
        @import("afd_attn_users.zig").withBatchedDrafts(&forecast, &r.m, n, true);
        try t.expectEqualDeep(after, forecast);
    }
}
