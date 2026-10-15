//! The MTP draft layer on block FP8 (mtp_fp8): BadAd84's lever (TF_FLASHNEXT_MTP_FP8, PR #36 `e6a35d2` to Mia's one-Spark
//! Qwen3.8-Flash-Next recipe, MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) on GLM's MTP layer at the AFD
//! attention node. TF_GLM_MTP_FP8=1 holds the eight projections the dense-q4 form holds 4-bit there (eh, the DSA's proj,
//! q_b and o, the indexer's kw and qb, the shared expert's gu and down) as Mia's patch 0002 FP8 instead: make_f8's e4m3
//! bytes with an fp32 scale a row and 128-column block (stack_f8 for the stacked ones), made on the GPU from their
//! stored BF16 rows (glm_pack.cu's f8 kernels) and multiplied by 0002's `_fmm`, as the FP8 head is. Only drafts read
//! the MTP layer and the verify keeps the target's tokens, so replies are unchanged. Off (the plan untouched) unless 1,
//! which needs TF_GLM_DENSE=q4:DIR and TF_AFD_MTP=1; q4-mtp.safetensors is then not read.
const std = @import("std");
const views = @import("cuda_views.zig");
const pack = @import("cuda_pack.zig");
const q4 = @import("cuda_pack_q4.zig");

const xo = pack.x_ops;

/// The MTP layer's projections the switch holds on FP8: eh, dsa proj, q_b, o, index kw, qb, shared gu, down.
pub const projections = 8;

/// TF_GLM_MTP_FP8's value: empty or 0 (off), or 1.
pub fn parse(v: []const u8) !bool {
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadMtpFp8;
}

/// The switch from the environment: unset is off; a value but 0 or 1 is refused (BadMtpFp8).
pub fn fromEnv() !bool {
    const v = std.mem.span(std.c.getenv("TF_GLM_MTP_FP8") orelse return false);
    return parse(v) catch |e| {
        std.log.err("TF_GLM_MTP_FP8 is 1 (the MTP layer's projections on block FP8), 0 or unset, not {s}", .{v});
        return e;
    };
}

/// The switch as tf-glm-afd's attention verbs read it before anything loads: 1 needs the MTP layer (TF_AFD_MTP=1).
pub fn checkEnv(mtp: bool) !bool {
    return sound(try fromEnv(), mtp) catch |e| {
        std.log.err("TF_GLM_MTP_FP8=1 holds the MTP layer's projections on FP8: it needs TF_AFD_MTP=1", .{});
        return e;
    };
}

/// `on` against the load: 1 without the MTP layer is refused (MtpFp8NeedsAfdMtp).
pub fn sound(on: bool, mtp: bool) !bool {
    if (on and !mtp) return error.MtpFp8NeedsAfdMtp;
    return on;
}

/// What `apply` made: the projections, their BF16 bytes, and the FP8 bytes with their scales that replace them.
pub const Summary = struct { projections: usize = 0, bf16: u64 = 0, f8: u64 = 0 };

/// Whether a plan item is one of the MTP layer's projections the dense-q4 form holds 4-bit (q4.eligible under w.mtp).
pub fn mtpProjection(it: pack.Item) bool {
    return std.mem.startsWith(u8, it.path, "w.mtp.") and q4.eligible(it);
}

/// The MTP layer's projections in the plan (`<base>.weight`, bf16 [n, k] from one stored tensor or stack_b16's several)
/// as stack_f8's two buffers over the same stored rows: `<base>.weight` e4m3 [n, k] and `<base>.scale` fp32 [n, k/128].
/// Run before the q4 overlay, which then leaves them (no longer BF16). Returns what it made.
pub fn apply(pk: *pack.Plan) !Summary {
    const a = pk.arena.allocator();
    var s: Summary = .{};
    var i: usize = 0;
    while (i < pk.items.items.len) : (i += 1) {
        const it = pk.items.items[i];
        if (!mtpProjection(it)) continue;
        const n = it.buf.shape[0];
        const k = it.buf.shape[1];
        if (k % 128 != 0) return error.F8Width;
        var c: xo.Cat = .{};
        if (it.x) |x| switch (x) {
            .cat => |cat| c = cat,
            else => return error.MtpNotStoredRows,
        } else switch (it.op) {
            .copy => |r| {
                c.srcs[0] = r;
                c.lens[0] = @as(u64, n) * k * 2;
                c.n = 1;
            },
            else => return error.MtpNotStoredRows,
        }
        var bytes: u64 = 0;
        for (c.lens[0..c.n]) |len| bytes += len;
        if (bytes != it.buf.bytes()) return error.MtpNotStoredRows;
        const f = views.F8.of(n, k);
        const base = it.path[0 .. it.path.len - ".weight".len];
        pk.items.items[i] = .{ .path = it.path, .buf = f.weight, .op = it.op, .home = it.home, .x = .{ .f8cat = .{ .cat = c, .k = k, .part = .weight } } };
        try pk.items.insert(a, i + 1, .{ .path = try std.fmt.allocPrint(a, "{s}.scale", .{base}), .buf = f.scale, .op = it.op, .home = it.home, .x = .{ .f8cat = .{ .cat = c, .k = k, .part = .scale } } });
        i += 1;
        s.projections += 1;
        s.bf16 += it.buf.bytes();
        s.f8 += f.weight.bytes() + f.scale.bytes();
    }
    if (s.projections != projections) return error.MtpProjections;
    return s;
}

/// bind's step: each of the MTP layer's B16 projections becomes the F8 view where the plan makes `<path>.scale` (fp32
/// [n, k/128], apply's). With the switch off (no such scale) nothing changes. Returns how many changed.
pub fn adapt(w: *views.Weights, p: *const pack.Plan) usize {
    const x = if (w.mtp) |*m| m else return 0;
    var n = one(&x.eh, "w.mtp.eh", p);
    if (x.layer.dsa) |*d| {
        n += one(&d.proj, "w.mtp.layer.dsa.proj", p) + one(&d.q_b, "w.mtp.layer.dsa.q_b", p) + one(&d.o, "w.mtp.layer.dsa.o", p);
        if (d.index) |*ix| n += one(&ix.kw, "w.mtp.layer.dsa.index.kw", p) + one(&ix.qb, "w.mtp.layer.dsa.index.qb", p);
    }
    if (x.layer.moe) |*mo| if (mo.shared) |*sh| {
        n += one(&sh.gu, "w.mtp.layer.moe.shared.gu", p) + one(&sh.down, "w.mtp.layer.moe.shared.down", p);
    };
    return n;
}

fn one(l: *views.Linear, path: []const u8, p: *const pack.Plan) usize {
    const b = switch (l.*) {
        .b16 => |b| b,
        .q4, .f8 => return 0,
    };
    var name: [96]u8 = undefined;
    const s = p.find(std.fmt.bufPrint(&name, "{s}.scale", .{path}) catch return 0) orelse return 0;
    if (s.buf.dtype != .f32 or s.buf.dims != 2 or s.buf.shape[0] != b.n or s.buf.shape[1] * 128 != b.k) return 0;
    l.* = .{ .f8 = views.F8.of(b.n, b.k) };
    return 1;
}

const testing = std.testing;
const md = @import("cuda_model.zig");
const cfg = @import("config.zig");
const Ref = @import("cuda_split.zig").Ref;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "TF_GLM_MTP_FP8: unset, empty and 0 are off, 1 is on, anything else refused; 1 without TF_AFD_MTP=1 refused" {
    try testing.expect(!try parse(""));
    try testing.expect(!try parse("0"));
    try testing.expect(try parse("1"));
    for ([_][]const u8{ "2", "on", "true", "fp8", " 1", "1 ", "01", "head" }) |v| try testing.expectError(error.BadMtpFp8, parse(v));
    try testing.expect(!try sound(false, false) and !try sound(false, true) and try sound(true, true));
    try testing.expectError(error.MtpFp8NeedsAfdMtp, sound(true, false));
    _ = unsetenv("TF_GLM_MTP_FP8");
    try testing.expect(!try fromEnv());
    try testing.expect(!try checkEnv(false));
    _ = setenv("TF_GLM_MTP_FP8", "0", 1);
    try testing.expect(!try checkEnv(false));
    _ = setenv("TF_GLM_MTP_FP8", "1", 1);
    try testing.expect(try checkEnv(true));
    _ = unsetenv("TF_GLM_MTP_FP8");
}

/// The MTP layer's eight projections (python's paths, rows, columns, stored parts) at world-1 shapes, a backbone DSA
/// layer's proj and the head beside them: none but the eight may change.
const Fixture = struct {
    const mtp = [_]struct { []const u8, usize, usize, u8 }{
        .{ "w.mtp.eh.weight", 64, 256, 1 },                  .{ "w.mtp.layer.dsa.proj.weight", 48, 128, 2 },
        .{ "w.mtp.layer.dsa.q_b.weight", 64, 128, 1 },       .{ "w.mtp.layer.dsa.o.weight", 32, 384, 1 },
        .{ "w.mtp.layer.dsa.index.kw.weight", 40, 128, 2 },  .{ "w.mtp.layer.dsa.index.qb.weight", 32, 128, 1 },
        .{ "w.mtp.layer.moe.shared.gu.weight", 64, 128, 2 }, .{ "w.mtp.layer.moe.shared.down.weight", 32, 256, 1 },
    };

    fn plan(p: *pack.Plan) !void {
        const a = p.arena.allocator();
        try p.items.append(a, .{ .path = "w.layers[3].dsa.proj.weight", .buf = views.Buf.of(.bf16, &.{ 48, 128 }), .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0 });
        var off: u64 = 4096;
        for (mtp, 0..) |t, j| {
            const n = t[1];
            const k = t[2];
            const g: u32 = @intCast(1 + j % 2); // the parts in two stored groups
            if (t[3] == 1) {
                try p.items.append(a, .{ .path = t[0], .buf = views.Buf.of(.bf16, &.{ n, k }), .op = .{ .copy = .{ .g = g, .off = off } }, .home = g });
            } else { // stack_b16 of two parts, the second in the other group: its home the later one
                var c: xo.Cat = .{ .n = 2 };
                c.srcs[0] = .{ .g = g, .off = off };
                c.lens[0] = n / 2 * k * 2;
                c.srcs[1] = .{ .g = 3 - g, .off = off + 65536 };
                c.lens[1] = (n - n / 2) * k * 2;
                try p.items.append(a, .{ .path = t[0], .buf = views.Buf.of(.bf16, &.{ n, k }), .op = .{ .copy = c.srcs[0] }, .home = 2, .x = .{ .cat = c } });
            }
            off += 1 << 20;
        }
        try p.items.append(a, .{ .path = "w.head.weight", .buf = views.Buf.of(.bf16, &.{ 256, 128 }), .op = .{ .copy = .{ .g = 0, .off = 1 << 30 } }, .home = 0 });
    }
};

test "apply: the MTP layer's eight projections become stack_f8's bytes and scales over the same stored rows; the rest stays, and the q4 overlay leaves them" {
    var p: pack.Plan = .{ .arena = .init(testing.allocator), .groups = 3 };
    defer p.deinit();
    try Fixture.plan(&p);
    const before = try p.arena.allocator().dupe(pack.Item, p.items.items);
    const s = try apply(&p);
    try testing.expectEqual(@as(usize, 8), s.projections);
    try testing.expectEqual(before.len + 8, p.items.items.len);
    var bf16: u64 = 0;
    var f8: u64 = 0;
    for (Fixture.mtp) |t| {
        bf16 += t[1] * t[2] * 2;
        f8 += t[1] * t[2] + t[1] * (t[2] / 128) * 4;
    }
    try testing.expectEqual(bf16, s.bf16);
    try testing.expectEqual(f8, s.f8);
    // the first and last items (a backbone layer's proj, the head) untouched; each MTP projection weight then scale
    try testing.expectEqualDeep(before[0], p.items.items[0]);
    try testing.expectEqualDeep(before[before.len - 1], p.items.items[p.items.items.len - 1]);
    for (Fixture.mtp, 0..) |t, j| {
        const w = p.items.items[1 + 2 * j];
        const sc = p.items.items[2 + 2 * j];
        const was = before[1 + j];
        try testing.expectEqualStrings(t[0], w.path);
        try testing.expectEqualStrings(try std.fmt.allocPrint(p.arena.allocator(), "{s}.scale", .{t[0][0 .. t[0].len - ".weight".len]}), sc.path);
        try testing.expect(w.buf.dtype == .f8e4m3 and w.buf.shape[0] == t[1] and w.buf.shape[1] == t[2]);
        try testing.expect(sc.buf.dtype == .f32 and sc.buf.shape[0] == t[1] and sc.buf.shape[1] == t[2] / 128);
        try testing.expect(w.x.?.f8cat.part == .weight and sc.x.?.f8cat.part == .scale);
        try testing.expectEqual(t[3], w.x.?.f8cat.cat.n);
        try testing.expectEqual(was.home, w.home);
        try testing.expectEqual(was.home, sc.home);
        try testing.expectEqual(was.home, xo.home(w.x.?)); // made once its last stored part is loaded
        const src0: Ref = if (was.x) |x| x.cat.srcs[0] else was.op.copy;
        try testing.expectEqual(src0, w.x.?.f8cat.cat.srcs[0]);
        for ([_]u32{ 1, 2 }) |g| try testing.expectEqual(pack.readsGroup(was, g), pack.readsGroup(w, g));
        try testing.expect(!q4.eligible(w) and !q4.eligible(sc) and !mtpProjection(w)); // the q4 overlay passes them by
    }
    try testing.expectError(error.MtpProjections, apply(&p)); // already FP8: none found, nothing made twice
}

test "apply refuses a plan without the eight, or a projection not held as stored BF16 rows" {
    var p: pack.Plan = .{ .arena = .init(testing.allocator), .groups = 3 };
    defer p.deinit();
    try Fixture.plan(&p);
    _ = p.items.orderedRemove(2); // the DSA's proj missing: seven
    try testing.expectError(error.MtpProjections, apply(&p));
    var r: pack.Plan = .{ .arena = .init(testing.allocator), .groups = 3 };
    defer r.deinit();
    try Fixture.plan(&r);
    r.items.items[1].x = .{ .alias = .{ .g = 1, .off = 0 } }; // eh left in place: not stored rows to make
    try testing.expectError(error.MtpNotStoredRows, apply(&r));
}

test "adapt: the MTP layer's B16 views become F8 where the plan makes their scales; the backbone and the head stay" {
    var c: cfg.Config = .{};
    c.run = 4;
    var m = try md.Model.from(&c, .{ .quant = .exl3 }, 1, 0);
    try m.cut(4);
    m.mtp_layers = 1;
    var w = try views.rank(testing.allocator, &m);
    defer w.deinit(testing.allocator);
    var p: pack.Plan = .{ .arena = .init(testing.allocator), .groups = 1 };
    defer p.deinit();
    try testing.expectEqual(@as(usize, 0), adapt(&w, &p)); // the switch off: no scales, nothing changes
    const x = &w.mtp.?;
    const ls = [_]struct { []const u8, *views.Linear }{
        .{ "w.mtp.eh", &x.eh },                                       .{ "w.mtp.layer.dsa.proj", &x.layer.dsa.?.proj },
        .{ "w.mtp.layer.dsa.q_b", &x.layer.dsa.?.q_b },               .{ "w.mtp.layer.dsa.o", &x.layer.dsa.?.o },
        .{ "w.mtp.layer.dsa.index.kw", &x.layer.dsa.?.index.?.kw },   .{ "w.mtp.layer.dsa.index.qb", &x.layer.dsa.?.index.?.qb },
        .{ "w.mtp.layer.moe.shared.gu", &x.layer.moe.?.shared.?.gu }, .{ "w.mtp.layer.moe.shared.down", &x.layer.moe.?.shared.?.down },
    };
    const a = p.arena.allocator();
    for (ls) |t| {
        const f = views.F8.of(t[1].n(), t[1].k());
        try p.items.append(a, .{ .path = try std.fmt.allocPrint(a, "{s}.weight", .{t[0]}), .buf = f.weight, .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0 });
        try p.items.append(a, .{ .path = try std.fmt.allocPrint(a, "{s}.scale", .{t[0]}), .buf = f.scale, .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0 });
    }
    try p.items.append(a, .{ .path = "w.layers[3].dsa.proj.scale", .buf = views.Buf.of(.f32, &.{ w.layers[3].dsa.?.proj.n(), w.layers[3].dsa.?.proj.k() / 128 }), .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0 });
    try testing.expectEqual(@as(usize, 8), adapt(&w, &p));
    for (ls) |t| try testing.expect(t[1].* == .f8);
    try testing.expect(w.layers[3].dsa.?.proj == .b16 and w.head == .b16 and x.layer.dsa.?.kv_k == .b16); // not the MTP's eight
    try testing.expectEqual(@as(usize, 0), adapt(&w, &p)); // once
    x.eh = .{ .b16 = views.B16.of(x.eh.n(), x.eh.k()) };
    p.items.items[1].buf = views.Buf.of(.f32, &.{ x.eh.n(), 1 }); // eh's scale of another shape: eh stays B16
    try testing.expectEqual(@as(usize, 0), adapt(&w, &p));
    try testing.expect(x.eh == .b16);
}
