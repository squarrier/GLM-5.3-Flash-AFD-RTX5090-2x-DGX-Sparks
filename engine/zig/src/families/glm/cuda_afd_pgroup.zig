//! TF_GLM_EXL3_PGROUP on the expert halves (qualification_3e1f1105): bertholomus's parallel grouping (glm-dsa-tp4 dbe0773, Apache-2.0)
//! needs one int32 [E] buffer, group_count's counts. It is a block of its own (as X1's): with the switch off nothing is
//! placed and every address is today's; with it on every launch but the grouping keeps its arguments. Both scratches
//! take it, since a half's windows run in order on one stream; only windows of 64+ rows launch groupPar.

const std = @import("std");
const ex3 = @import("cuda_exl3_experts.zig");
const ae = @import("cuda_afd_expert.zig");

/// The block's bytes: counts, int32 [E].
pub fn bytes(experts: usize) usize {
    return 4 * experts;
}

/// Both scratches take the block at `base` (0: off, `group` alone).
pub fn apply(b: *ae.Bufs, base: u64) void {
    b.decode.pg_counts = base;
    b.pref.pg_counts = base;
}

/// The half's INFO line when the switch is in effect (afd_expert.zig prints it once, as it reads the switch).
pub fn line(buf: []u8, half: u32) []const u8 {
    return std.fmt.bufPrint(buf, "INFO afd expert {d}: EXL3 experts: PGROUP (bertholomus's group_count/group_place, TF_GLM_EXL3_PGROUP) on windows of {d}+ rows, the same lists", .{ half, ex3.prompt_rows }) catch "INFO afd expert: EXL3 experts: PGROUP";
}

test "the counts block: nothing in the node's block moves, the scratches carry it, windows of 64+ rows launch groupPar" {
    const k = @import("cuda_exl3_kernels.zig");
    var b = try ae.Bufs.init(4096, 9, 1024, 288, ae.decode_rows, ae.prefill_rows);
    var landing = ae.Landing.init(.{ .d = 4096, .s = 9 }, ae.prefill_rows);
    const off = try ae.place(&b, &landing, 1 << 44);
    const before = b;
    apply(&b, 1 << 46);
    try std.testing.expectEqual(off, try ae.place(&b, &landing, 1 << 44));
    try std.testing.expectEqual(before.pref.members_buf.ptr, b.pref.members_buf.ptr);
    try std.testing.expectEqual(@as(u64, 1 << 46), b.pref.bound(288).counts);
    try std.testing.expectEqual(@as(u64, 1 << 46), b.decode.bound(288).counts);
    try std.testing.expectEqual(@as(usize, 1152), bytes(288));
    const l: ex3.Layer = .{ .gate_ptr = 1, .up_ptr = 2, .down_ptr = 3, .gate_k2 = 4, .up_k2 = 5, .down_k2 = 6, .suh_g = 7, .suh_u = 8, .svh_g = 9, .svh_u = 10, .suh_d = 11, .svh_d = 12, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
    const v = ae.landed(.{ .d = 4096, .s = 9 }, landing.buf.ptr, 2048);
    for ([_]usize{ 1, 8, 63, 64, 2048 }) |R| {
        var ls: k.Launches = .{};
        try ae.launchesSched(&ls, v, l, b.scratch(R), R, 7.0, .{ .optin = 101376 }, .{});
        try std.testing.expectEqualStrings(if (R >= ex3.prompt_rows) k.sym.group_count else k.sym.group, ls.items[0].symbol);
        apply(&b, 0); // off again: today's launch at every R
        var today: k.Launches = .{};
        try ae.launchesSched(&today, v, l, b.scratch(R), R, 7.0, .{ .optin = 101376 }, .{});
        try std.testing.expectEqualStrings(k.sym.group, today.items[0].symbol);
        apply(&b, 1 << 46);
    }
    var buf: [256]u8 = undefined;
    try std.testing.expect(std.mem.startsWith(u8, line(&buf, 1), "INFO afd expert 1: EXL3 experts: PGROUP (bertholomus's"));
}
