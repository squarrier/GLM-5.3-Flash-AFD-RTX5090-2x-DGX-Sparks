//! Mia's FP8 head in the dense-q4 form of an EXL3 rank (qualification_qualification_ebed70b7; recipe patch 0002 under 0005, TF_GLM_DENSE=q4): lm_head
//! held as make_f8's e4m3 bytes and fp32 scales a row and 128-column block, made on the GPU from its stored BF16 rows
//! (cuda_pack_x's f8 op, glm_pack.cu's f8 kernels) and multiplied by 0002's _fmm (Tri.matmulF8). kv_b stays BF16 as in
//! the lane, whose latent path reads kv_b only through the absorb (TF_GLM_KVB unset). Off unless TF_GLM_F8=head.
const std = @import("std");
const views = @import("cuda_views.zig");
const pack = @import("cuda_pack.zig");

/// TF_GLM_F8's value: empty (off) or `head`. kv_b is refused: no forward reads its rows as a matmul.
pub fn parse(v: []const u8) !bool {
    if (v.len == 0) return false;
    if (std.mem.eql(u8, v, "head")) return true;
    if (std.mem.indexOf(u8, v, "kv_b") != null) return error.F8KvbIsAbsorbed;
    return error.BadF8Form;
}

/// The switch from the environment; set only with TF_GLM_DENSE=q4:DIR (the caller checks), as the lane's DENSE=q4.
pub fn fromEnv() !bool {
    const v = std.mem.span(std.c.getenv("TF_GLM_F8") orelse return false);
    return parse(v) catch |e| {
        std.log.err("TF_GLM_F8 is head or unset, not {s}{s}", .{ v, if (e == error.F8KvbIsAbsorbed) " (kv_b reaches the forward only through the MLA absorb, which the lane keeps BF16)" else "" });
        return e;
    };
}

/// The plan's head (`w.head.weight`, bf16 [n, k] copied from its stored rows) as make_f8's two buffers made from the
/// same rows: `w.head.weight` e4m3 [n, k] and `w.head.scale` fp32 [n, k/128]. Returns the packed bytes saved; a head
/// that is not such a copy (already FP8, or MLX's triplet) is error.F8HeadNotStoredRows, which the caller reports.
pub fn apply(pk: *pack.Plan) !u64 {
    for (pk.items.items, 0..) |it, i| if (std.mem.eql(u8, it.path, "w.head.weight")) {
        if (it.x != null or it.op != .copy or it.buf.dtype != .bf16 or it.buf.dims != 2 or it.buf.shape[1] % 128 != 0)
            return error.F8HeadNotStoredRows;
        const src = it.op.copy;
        const n = it.buf.shape[0];
        const k = it.buf.shape[1];
        const f = views.F8.of(n, k);
        pk.items.items[i] = .{ .path = it.path, .buf = f.weight, .op = it.op, .home = it.home, .x = .{ .f8 = .{ .src = src, .n = n, .k = k, .part = .weight } } };
        try pk.items.insert(pk.arena.allocator(), i + 1, .{ .path = "w.head.scale", .buf = f.scale, .op = it.op, .home = it.home, .x = .{ .f8 = .{ .src = src, .n = n, .k = k, .part = .scale } } });
        return it.buf.bytes() - f.weight.bytes() - f.scale.bytes();
    };
    return error.NoHeadInPlan;
}

/// bind's step for the FP8 head: `w.head` held B16 becomes the F8 view when the plan makes `w.head.scale` (fp32 [n,
/// k/128], apply's). Otherwise (the switch off, MLX) nothing changes. Returns whether it changed.
pub fn adapt(w: *views.Weights, p: *const pack.Plan) bool {
    const s = p.find("w.head.scale") orelse return false;
    const b = switch (w.head) {
        .b16 => |b| b,
        .q4, .f8 => return false,
    };
    if (s.buf.dtype != .f32 or s.buf.dims != 2 or s.buf.shape[0] != b.n or s.buf.shape[1] * 128 != b.k) return false;
    w.head = .{ .f8 = views.F8.of(b.n, b.k) };
    return true;
}

test "TF_GLM_F8: unset is off, head is on, kv_b and anything else are refused" {
    try std.testing.expect(!try parse(""));
    try std.testing.expect(try parse("head"));
    try std.testing.expectError(error.F8KvbIsAbsorbed, parse("head,kv_b"));
    try std.testing.expectError(error.F8KvbIsAbsorbed, parse("kv_b"));
    try std.testing.expectError(error.BadF8Form, parse("fp8"));
}

test "apply: the head's BF16 copy becomes make_f8's bytes and scales over the same stored rows; the rest stays" {
    var p: pack.Plan = .{ .arena = .init(std.testing.allocator), .groups = 2 };
    defer p.deinit();
    const a = p.arena.allocator();
    const src: @import("cuda_split.zig").Ref = .{ .g = 1, .off = 4096 };
    for ([_][]const u8{ "w.norm", "w.head.weight", "w.draft_head.weight" }) |path|
        try p.items.append(a, .{ .path = path, .buf = views.Buf.of(.bf16, &.{ 256, 512 }), .op = .{ .copy = src }, .home = 1 });
    try std.testing.expectEqual(@as(u64, 256 * 512 * 2 - 256 * 512 - 256 * 4 * 4), try apply(&p));
    const want = [_][]const u8{ "w.norm", "w.head.weight", "w.head.scale", "w.draft_head.weight" };
    try std.testing.expectEqual(want.len, p.items.items.len);
    for (want, p.items.items) |w, it| try std.testing.expectEqualStrings(w, it.path);
    const w = p.items.items[1];
    const s = p.items.items[2];
    try std.testing.expect(w.buf.dtype == .f8e4m3 and w.buf.shape[0] == 256 and w.buf.shape[1] == 512 and w.x.?.f8.part == .weight);
    try std.testing.expect(s.buf.dtype == .f32 and s.buf.shape[1] == 4 and s.x.?.f8.part == .scale and s.x.?.f8.src.off == 4096);
    try std.testing.expect(p.items.items[0].x == null and p.items.items[3].x == null and pack.readsGroup(s, 1) and s.home == 1);
    try std.testing.expectError(error.F8HeadNotStoredRows, apply(&p)); // already FP8: refused, not made twice

    // bind's adapt: the B16 head becomes F8 only with the plan's w.head.scale of the head's shape
    var none: [0]views.Layer = .{};
    var ws: views.Weights = .{ .embed = .{ .b16 = views.Buf.of(.bf16, &.{ 256, 512 }) }, .layers = &none, .norm = views.Buf.of(.bf16, &.{512}), .head = .{ .b16 = views.B16.of(256, 512) }, .mtp = null };
    try std.testing.expect(adapt(&ws, &p) and ws.head == .f8 and ws.head.n() == 256 and ws.head.k() == 512);
    try std.testing.expect(!adapt(&ws, &p)); // once
    var empty: pack.Plan = .{ .arena = .init(std.testing.allocator), .groups = 2 };
    defer empty.deinit();
    ws.head = .{ .b16 = views.B16.of(256, 512) };
    try std.testing.expect(!adapt(&ws, &empty) and ws.head == .b16); // the switch off: no w.head.scale, no change
    p.items.items[2].buf = views.Buf.of(.f32, &.{ 256, 8 }); // a scale of another shape: refused
    try std.testing.expect(!adapt(&ws, &p) and ws.head == .b16);
}
