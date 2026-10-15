//! The CUDA loader's TP split against weights.zig's own loader (the Metal pair's), row for row, on a synthetic checkpoint.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("../../src/families/glm/config.zig");
const wts = @import("../../src/families/glm/weights.zig");
const split = @import("../../src/families/glm/cuda_split.zig");
const synth = @import("../../src/families/glm/cuda_synth.zig");

const io = std.testing.io;
const expectEqual = std.testing.expectEqual;

const Check = struct {
    groups: [][]u8,
    rows: usize = 0, // rows compared

    fn mine(k: *Check, r: split.Ref, n: usize) []const u8 {
        return k.groups[r.g][@intCast(r.off)..][0..n];
    }

    /// A 4-bit matrix (`e` stacked) row by row: the words, the scales and the biases.
    fn q4(k: *Check, m: wts.Q4, o: split.Q4, e: usize) !void {
        try expectEqual(m.n, o.n);
        try expectEqual(m.k, o.k);
        const wr: usize = m.k / 2;
        const sr: usize = m.k / 64 * 2;
        for (0..e * m.n) |r| {
            try std.testing.expectEqualSlices(u8, m.w.addr()[r * wr ..][0..wr], k.mine(o.w, (r + 1) * wr)[r * wr ..]);
            try std.testing.expectEqualSlices(u8, m.s.addr()[r * sr ..][0..sr], k.mine(o.s, (r + 1) * sr)[r * sr ..]);
            try std.testing.expectEqualSlices(u8, m.b.addr()[r * sr ..][0..sr], k.mine(o.b, (r + 1) * sr)[r * sr ..]);
        }
        k.rows += e * m.n;
    }

    fn same(k: *Check, m: wts.Ref, o: split.Ref, n: usize) !void {
        try std.testing.expectEqualSlices(u8, m.addr()[0..n], k.mine(o, n));
    }

    /// weights_fix's hc_pack of our stored [24, 4D] mix equals the Metal buffer.
    fn hcPacked(k: *Check, m: wts.Ref, o: split.Ref) !void {
        const in: []align(1) const u16 = std.mem.bytesAsSlice(u16, k.mine(o, 24 * 16384 * 2));
        const out: [*]align(1) const u16 = @ptrCast(m.addr());
        var i: usize = 0;
        for (0..6) |og| for (0..8) |sgn| for (0..32) |lane| for (0..16) |ii| for (0..4) |tm| for (0..4) |tn| {
            try expectEqual(in[(og * 4 + tm) * 16384 + ii * 1024 + sgn * 128 + lane * 4 + tn], out[i]);
            i += 1;
        };
    }

    /// weights_fix's router_pack of our stored [E, K] bf16 equals the Metal buffer.
    fn routerPacked(k: *Check, m: wts.Ref, o: split.Ref, E: usize, K: usize) !void {
        const in: []align(1) const u16 = std.mem.bytesAsSlice(u16, k.mine(o, E * K * 2));
        const out: [*]align(1) const u16 = @ptrCast(m.addr());
        var i: usize = 0;
        for (0..E / 4) |q| for (0..8) |thr| for (0..K / 32) |ii| for (0..4) |tm| for (0..4) |tn| {
            try expectEqual(in[(4 * q + tn) * K + 32 * ii + 4 * thr + tm], out[i]);
            i += 1;
        };
    }

    /// Our bf16 values widened equal the Metal fp32 buffer (o_norm; the conv taps go through `conv`).
    fn widened(k: *Check, m: wts.Ref, o: split.Ref, n: usize) !void {
        const in: []align(1) const u16 = std.mem.bytesAsSlice(u16, k.mine(o, n * 2));
        const out: [*]align(1) const u32 = @ptrCast(m.addr());
        for (0..n) |i| try expectEqual(@as(u32, in[i]) << 16, out[i]);
    }

    /// Metal's [tap][q | k | v channels] fp32 against our [q | k | v][channel][tap] bf16.
    fn conv(k: *Check, m: wts.Ref, o: split.Ref, C: usize, T: usize) !void {
        const in: []align(1) const u16 = std.mem.bytesAsSlice(u16, k.mine(o, 3 * C * T * 2));
        const out: [*]align(1) const u32 = @ptrCast(m.addr());
        for (0..3) |part| for (0..C) |ch| for (0..T) |tap| {
            try expectEqual(@as(u32, in[(part * C + ch) * T + tap]) << 16, out[tap * 3 * C + part * C + ch]);
        };
    }

    fn layer(k: *Check, c: *const cfg.Config, i: usize, m: *const wts.Layer, o: *const split.Layer) !void {
        const D: usize = c.hidden;
        try expectEqual(m.hc == null, o.hc == null);
        if (m.hc) |hm| for (hm, o.hc.?) |a, b| {
            try k.hcPacked(a.fnp, b.fnp);
            try k.same(a.base, b.base, 24 * 4);
            try k.same(a.scale, b.scale, 3 * 4);
        };
        try k.same(m.in_norm, o.in_norm, D * 2);
        try k.same(m.post_norm, o.post_norm, D * 2);
        switch (m.attn) {
            .kda => |a| {
                const b = o.attn.kda;
                try k.q4(a.in_proj, b.in_proj, 1);
                try k.q4(a.f_b, b.f_b, 1);
                try k.q4(a.g_b, b.g_b, 1);
                try k.q4(a.o_proj, b.o_proj, 1);
                try k.conv(a.conv_w, b.conv, c.kdaWidth(), c.conv);
                try k.same(a.a_log, b.a_log, c.kda_heads * 4);
                try k.same(a.dt_bias, b.dt_bias, c.kdaWidth() * 4);
                try k.widened(a.o_norm, b.o_norm, c.kda_dim);
            },
            .mla => |a| {
                const b = o.attn.mla;
                try k.q4(a.x_proj, b.x_proj, 1);
                try k.q4(a.qr_proj, b.qr_proj, 1);
                try k.q4(a.kv_b, b.kv_b, 1);
                try k.q4(a.o_proj, b.o_proj, 1);
                try k.same(a.q_norm, b.q_norm, c.q_lora * 2);
                try k.same(a.kv_norm, b.kv_norm, c.kv_lora * 2);
                try k.same(a.k_norm_w, b.k_norm_w, c.i_dim * 2);
                try k.same(a.k_norm_b, b.k_norm_b, c.i_dim * 2);
                try k.same(a.ape, b.ape, c.kpool * c.i_dim * 2);
                try k.routerPacked(a.igate, b.igate, c.i_dim, D);
            },
        }
        switch (m.mlp) {
            .dense => |a| {
                try k.q4(a.gate_up, o.mlp.dense.gate_up, 1);
                try k.q4(a.down, o.mlp.dense.down, 1);
            },
            .moe => |a| {
                const b = o.mlp.moe;
                const e: usize = if (i == c.layers) c.experts else c.own[1] - c.own[0];
                try expectEqual(@as(u32, @intCast(e)), b.experts);
                try k.routerPacked(a.router, b.router, c.experts, D);
                try k.same(a.bias, b.bias, c.experts * 4);
                try k.q4(a.sh_gate_up, b.sh_gate_up, 1);
                try k.q4(a.sh_down, b.sh_down, 1);
                try k.q4(a.gate, b.gate, e);
                try k.q4(a.up, b.up, e);
                try k.q4(a.down, b.down, e);
            },
        }
    }
};

fn tmpPath(a: std.mem.Allocator, tmp: std.testing.TmpDir) ![]u8 {
    // std.testing.tmpDir parents the dir at <cwd>/.zig-cache/tmp
    const cwd = try std.process.currentPathAlloc(io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

test "each TP rank's tensors equal the Metal loader's, row for row, its packing undone (and on one rank too)" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmpPath(a, tmp);
    const base = synth.tiny();
    const names = try synth.write(a, dir, &base, 3);
    for ([_][2]u32{ .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 } }) |rr| {
        // the Metal pair's settings as engine.zig makes them: experts by rows, then TP2's heads
        var mc = base;
        if (rr[1] > 1) {
            try cfg.splitRows(&mc, rr[0], rr[1]);
            try cfg.splitHeads(&mc, rr[0], rr[1]);
        }
        const metal = try wts.load(std.testing.allocator, .{}, dir, &mc, 4, false);
        defer {
            metal.deinit();
            std.testing.allocator.destroy(metal);
        }
        const oc = try split.rankConfig(base, rr[0], rr[1]);
        try std.testing.expectEqualDeep(mc, oc);
        var p = try split.plan(std.testing.allocator, &oc, &names, .metal);
        defer p.deinit();
        try expectEqual(names.count(), p.used.count());
        var k: Check = .{ .groups = try synth.execute(a, dir, 3, &p) };
        try k.q4(metal.embed, p.embed, 1);
        try k.q4(metal.head, p.head, 1);
        try k.same(metal.norm, p.norm, base.hidden * 2);
        for (0..oc.run) |i| try k.layer(&oc, i, &metal.layers[i], &p.layers[i]);
        try expectEqual(metal.mtp != null, p.mtp != null);
        try k.layer(&oc, oc.layers, &metal.layers[oc.layers], &p.layers[oc.layers]);
        const mm = metal.mtp.?;
        const om = p.mtp.?;
        try k.q4(mm.eh_proj, om.eh_proj, 1);
        try k.same(mm.enorm, om.enorm, base.hidden * 2);
        try k.same(mm.hnorm, om.hnorm, base.hidden * 2);
        try k.same(mm.norm, om.norm, base.hidden * 2);
        try std.testing.expect(k.rows > 10000);
    }
}

test "the check catches a wrong split: rank 1's plan against the Metal loader's rank 0 differs" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmpPath(a, tmp);
    const base = synth.tiny();
    const names = try synth.write(a, dir, &base, 2);
    var mc = base;
    try cfg.splitRows(&mc, 0, 2);
    try cfg.splitHeads(&mc, 0, 2);
    const metal = try wts.load(std.testing.allocator, .{}, dir, &mc, 4, false);
    defer {
        metal.deinit();
        std.testing.allocator.destroy(metal);
    }
    const oc = try split.rankConfig(base, 1, 2);
    var p = try split.plan(std.testing.allocator, &oc, &names, .metal);
    defer p.deinit();
    var k: Check = .{ .groups = try synth.execute(a, dir, 2, &p) };
    try k.q4(metal.embed, p.embed, 1); // replicated: equal on both ranks
    const m2 = metal.layers[2].mlp.moe;
    const o2 = p.layers[2].mlp.moe;
    try std.testing.expect(!std.mem.eql(u8, m2.gate.w.addr()[0..1024], k.mine(o2.gate.w, 1024))); // the other half of the rows
    try std.testing.expect(!std.mem.eql(u8, metal.head.w.addr()[0..1024], k.mine(p.head.w, 1024))); // the other half of the vocabulary
}

test "the Metal loader's dry plan reads a real checkpoint's headers on both TP2 ranks (TF_GLM_CHECKPOINT, else skipped)" {
    const dir = std.mem.span(std.c.getenv("TF_GLM_CHECKPOINT") orelse return error.SkipZigTest);
    const path = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/config.json", .{dir}, 0);
    defer std.testing.allocator.free(path);
    const f = try mtl.MappedFile.open(path);
    defer f.deinit();
    const base = try cfg.parse(std.testing.allocator, f.bytes[0..f.size]);
    for ([_]u32{ 0, 1 }) |r| {
        var mc = base;
        try cfg.splitRows(&mc, r, 2);
        try cfg.splitHeads(&mc, r, 2);
        const metal = try wts.load(std.testing.allocator, .{}, dir, &mc, 4, true);
        defer {
            metal.deinit();
            std.testing.allocator.destroy(metal);
        }
        try std.testing.expect(metal.mtp != null);
    }
}
