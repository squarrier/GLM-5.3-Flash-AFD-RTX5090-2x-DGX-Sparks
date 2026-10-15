//! TF_GLM_EXL3_MMA on the expert halves (qualification_8724e161; bertholomus's grouped_mma over a work list, deepseek-v41-tp2 90b68e0,
//! Apache-2.0): one int32 buffer, the work list of (expert place, 64-row block) pairs, as long as the prompt scratch's
//! widest window allows (mma.workInts). It is a block of its own (as X1's and PGROUP's): with the switch off nothing
//! is placed and every address is today's. Both scratches take it, since a half's windows run in order on one stream:
//! each MMA call writes the list (work_list_kernel) before its two grouped_mma launches read it.

const std = @import("std");
const ex3 = @import("cuda_exl3_experts.zig");
const ae = @import("cuda_afd_expert.zig");

const mma = ex3.mma;

/// The block's bytes: the work list for the widest of the two scratches' windows.
pub fn bytes(b: *const ae.Bufs, experts: usize) usize {
    var most: usize = 0;
    for ([_]*const ae.Scratch{ &b.decode, &b.pref }) |s| most = @max(most, 4 * @as(usize, mma.workInts(@intCast(s.rows), @intCast(s.slots), @intCast(experts))));
    return most;
}

/// Both scratches take the block at `base` (0: off, X1's grouped launches).
pub fn apply(b: *ae.Bufs, base: u64) void {
    b.decode.mma_work = base;
    b.pref.mma_work = base;
}

/// The half's INFO line when the switch is in effect (afd_expert.zig prints it once, as it reads the switch).
pub fn line(buf: []u8, half: u32, p: ex3.Prompt) []const u8 {
    return std.fmt.bufPrint(buf, "INFO afd expert {d}: EXL3 experts: MMA (bertholomus's grouped_mma, exl3_mma, TF_GLM_EXL3_MMA) on X1's windows of {d}+ rows over PGROUP's lists: gate/up {s}, down grouped_mma", .{ half, ex3.prompt_rows, if (p.seq) "SEQ's kernel" else "grouped_mma, the splits folded" }) catch "INFO afd expert: EXL3 experts: MMA";
}

test "the work list block: nothing in the node's block moves, both scratches carry it, X1's 64+-row windows take MMA only with counts and the list" {
    const k = @import("cuda_exl3_kernels.zig");
    var b = try ae.Bufs.init(4096, 9, 1024, 288, ae.decode_rows, ae.prefill_rows);
    var landing = ae.Landing.init(.{ .d = 4096, .s = 9 }, ae.prefill_rows);
    const off = try ae.place(&b, &landing, 1 << 44);
    const before = b;
    apply(&b, 1 << 47);
    ae.pgroup.apply(&b, 1 << 46);
    try std.testing.expectEqual(off, try ae.place(&b, &landing, 1 << 44));
    try std.testing.expectEqual(before.pref.members_buf.ptr, b.pref.members_buf.ptr);
    try std.testing.expectEqual(@as(u64, 1 << 47), b.pref.bound(288).work);
    try std.testing.expectEqual(@as(u64, 1 << 47), b.decode.bound(288).work);
    try std.testing.expectEqual(@as(usize, 4 * mma.workInts(@intCast(ae.prefill_rows), 9, 288)), bytes(&b, 288));
    const l: ex3.Layer = .{ .gate_ptr = 1, .up_ptr = 2, .down_ptr = 3, .gate_k2 = 4, .up_k2 = 5, .down_k2 = 6, .suh_g = 7, .suh_u = 8, .svh_g = 9, .svh_u = 10, .suh_d = 11, .svh_d = 12, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
    const v = ae.landed(.{ .d = 4096, .s = 9 }, landing.buf.ptr, 2048);
    const p: ex3.Prompt = .{ .nfirst = true, .pa = true, .order = true, .xrow = true, .pgroup = true, .mma = true };
    for ([_]usize{ 1, 8, 63, 64, 2048 }) |R| {
        if (R > ae.prefill_rows) continue;
        var ls: k.Launches = .{};
        try ae.launchesX1(&ls, v, l, b.scratch(R), R, 7.0, .{ .optin = 101376 }, .{}, p, 0x90000, 0xf000);
        const mma_on = R >= ex3.prompt_rows;
        try std.testing.expectEqualStrings(if (mma_on) k.sym.group_count else k.sym.group, ls.items[0].symbol);
        try std.testing.expectEqual(mma_on, std.mem.eql(u8, ls.items[2].symbol, k.sym.work_list));
        apply(&b, 0); // the list off: X1 with PGROUP's grouping
        var x1: k.Launches = .{};
        try ae.launchesX1(&x1, v, l, b.scratch(R), R, 7.0, .{ .optin = 101376 }, .{}, p, 0x90000, 0xf000);
        for (x1.slice()) |x| try std.testing.expect(x.image != .exl3_mma_cb1);
        apply(&b, 1 << 47);
    }
    var buf: [256]u8 = undefined;
    try std.testing.expect(std.mem.startsWith(u8, line(&buf, 1, p), "INFO afd expert 1: EXL3 experts: MMA (bertholomus's"));
}
