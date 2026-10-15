//! SEQ on the expert halves (qualification_b3fba1da; Mia's TF_GLM_EXL3_SEQ design, MiaAI-Lab recipe patch 0004, Apache-2.0): with SEQ,
//! gate/up writes one fp32 plane a projection on X1's prompt windows, so the prompt scratch's z needs max(2 I, sk_d D)
//! fp32 a pair, not max(2 sk I, sk_d D). It is cut only when every position's 64+-row windows take that launch; else
//! z stays today's, so a window that falls back to today's split planes still fits.

const std = @import("std");
const ex3 = @import("cuda_exl3_experts.zig");
const ae = @import("cuda_afd_expert.zig");
const views = @import("cuda_views.zig");

const Buf = views.Buf;

/// Whether SEQ's one plane serves every prompt window of the scratch `s`: p.seq (or qualification_8724e161's p.mma, whose gate/up folds
/// its splits into one plane and whose down is GLM's one split) with NFIRST, PF or ORDER on (so the switches stay in
/// effect without a shared suh), every position inside promptTiles (and, for MMA without SEQ, mma.chains: else that
/// window takes X1's split planes), and the 63-row windows (today's split planes, under prompt_rows) within the cut z.
pub fn cuts(p: ex3.Prompt, layers: []const ex3.Layer, s: *const ae.Scratch) bool {
    if (!(p.seq or p.mma) or !(p.nfirst or p.pa or p.order) or s.rows < ex3.prompt_rows) return false;
    for (layers) |l| if (!ex3.promptTiles(l.cb, l.k2_gu, l.k2_d, s.gu, s.dn)) return false;
    if (p.mma and !p.seq) for (layers) |l| if (!ex3.mma.chains(l.dims, l.width, s.gu) or !ex3.mma.chains(l.width, l.dims, s.dn)) return false;
    const I = s.xd.shape[1];
    const D = s.xg.shape[1];
    return 2 * s.gu.sk * I * (ex3.prompt_rows - 1) <= @max(2 * I, s.dn.sk * D) * s.rows;
}

/// The scratch's z at SEQ's size: max(2 I, sk_d D) fp32 for each of its rows x slots pairs.
pub fn cut(s: *ae.Scratch) void {
    const I = s.xd.shape[1];
    const D = s.xg.shape[1];
    s.z = Buf.of(.f32, &.{@max(2 * I, s.dn.sk * D) * s.rows * s.slots});
}

/// The halves' rule before the block is placed: the prompt scratch's z cut when `cuts` holds for every position of
/// `x` (bound at its groups' addresses `base`); one INFO line with SEQ or MMA on. Returns whether it was cut.
pub fn apply(b: *ae.Bufs, x: *const ae.Half, base: []const u64, p: ex3.Prompt, half: u32) !bool {
    if (!(p.seq or p.mma)) return false;
    var layers: [128]ex3.Layer = undefined;
    if (x.layers.len > layers.len or base.len < x.layers.len) return error.SeqLayers; // the role binds pos at base[pos]
    for (0..x.layers.len) |pos| layers[pos] = x.bind(pos, base[pos]);
    const before = b.pref.z.bytes();
    const ok = cuts(p, layers[0..x.layers.len], &b.pref);
    if (ok) cut(&b.pref);
    std.debug.print("INFO afd expert {d}: {s}: the prompt scratch's z {d} MiB -> {d} MiB ({s})\n", .{ half, if (p.seq) "SEQ" else "MMA", before >> 20, b.pref.z.bytes() >> 20, if (ok) "one plane a projection at every position" else "kept: a position or a switch keeps today's planes" });
    return ok;
}

test "SEQ's z: 576 -> 288 MiB at 2,048 rows x 9 slots on GLM's half (I 1,024, D 4,096); kept without SEQ, without NFIRST/PF/ORDER, or outside promptTiles" {
    var s = try ae.Scratch.init(2048, 9, 4096, 1024, 288);
    const l: ex3.Layer = .{ .gate_ptr = 1, .up_ptr = 2, .down_ptr = 3, .gate_k2 = 4, .up_k2 = 5, .down_k2 = 6, .suh_g = 7, .suh_u = 8, .svh_g = 9, .svh_u = 10, .suh_d = 11, .svh_d = 12, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
    const on: ex3.Prompt = .{ .nfirst = true, .pa = true, .order = true, .xrow = true, .seq = true };
    try std.testing.expectEqual(@as(usize, 576 << 20), s.z.bytes());
    try std.testing.expect(cuts(on, &.{ l, l }, &s));
    try std.testing.expect(!cuts(.{ .nfirst = true, .pa = true, .order = true, .xrow = true }, &.{l}, &s)); // SEQ off
    try std.testing.expect(!cuts(.{ .xrow = true, .seq = true }, &.{l}, &s)); // rowrot alone: a layer without a shared suh falls back
    var odd = l;
    odd.k2_gu = .{ 2, 8 };
    try std.testing.expect(!cuts(on, &.{ l, odd }, &s)); // one position outside promptTiles: today's planes there
    const d = try ae.Scratch.init(8, 9, 4096, 1024, 288);
    try std.testing.expect(!cuts(on, &.{l}, &d)); // the decode scratch: under 64 rows, never SEQ's
    var mma = on; // qualification_8724e161: MMA alone (FOLD) cuts it the same; a layer outside MMA's shapes keeps today's planes
    mma.seq = false;
    mma.pgroup = true;
    mma.mma = true;
    try std.testing.expect(cuts(mma, &.{ l, l }, &s));
    var wide = l;
    wide.dims = 4096 + 512; // 288 k tiles: 18-tile chains, not in MMA's 4-tile steps
    try std.testing.expect(!cuts(mma, &.{ l, wide }, &s));
    try std.testing.expect(cuts(on, &.{ l, wide }, &s)); // SEQ's gate/up and X1's down (one split) still fit
    cut(&s);
    try std.testing.expectEqual(@as(usize, 288 << 20), s.z.bytes());
    const P: usize = 2048 * 9; // what each window needs of the cut z: SEQ's plane at 64+ rows, today's four planes under
    try std.testing.expect(4 * 2 * 1024 * P <= s.z.bytes() and 4 * 4096 * P <= s.z.bytes() and 4 * 2 * 4 * 1024 * 63 * 9 <= s.z.bytes());
}
