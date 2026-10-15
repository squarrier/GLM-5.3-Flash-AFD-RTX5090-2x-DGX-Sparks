//! The EXL3 split against python-0.6 itself: fixtures_cuda_exl3_split.json (tools/zig/exl3_split_fixtures.py) holds
//! the headers of 078455ff's layers 0-3 and 45 with experts 0-3, and every read python-0.6's split.py RankReader makes
//! of them for each TP2 rank, with the part split_bytes keeps. The plan's copies must be exactly those.
const std = @import("std");
const cfg = @import("config.zig");
const split = @import("cuda_split.zig");
const xs = @import("cuda_exl3_split.zig");

const Span = struct { shard: u64, off: u64, len: u64, row: u64, at: u64, width: u64 };

fn lt(_: void, a: Span, b: Span) bool {
    inline for (.{ "shard", "off", "len", "row", "at", "width" }) |f| {
        if (@field(a, f) != @field(b, f)) return @field(a, f) < @field(b, f);
    }
    return false;
}

fn dtype(s: []const u8) split.DType {
    const map = [_]struct { []const u8, split.DType }{ .{ "BF16", .bf16 }, .{ "F32", .f32 }, .{ "F16", .f16 }, .{ "I16", .i16 }, .{ "I32", .i32 }, .{ "U32", .u32 } };
    for (map) |m| if (std.mem.eql(u8, s, m[0])) return m[1];
    return .other;
}

test "each TP2 rank's EXL3 reads equal python-0.6 split.py's on 078455ff's own headers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fx = (try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures_cuda_exl3_split.json"), .{})).object;
    var names: split.Names = .empty;
    var it = fx.get("tensors").?.object.iterator();
    while (it.next()) |e| {
        const t = e.value_ptr.array.items;
        var s: split.Src = .{ .shard = @intCast(t[0].integer), .off = @intCast(t[1].integer), .len = @intCast(t[2].integer), .dtype = dtype(t[3].string), .rank = @intCast(t[4].array.items.len) };
        for (t[4].array.items, 0..) |d, i| s.shape[i] = @intCast(d.integer);
        try names.put(a, e.key_ptr.*, s);
    }
    var c: cfg.Config = .{};
    c.run = @intCast(fx.get("run").?.integer);
    c.experts = @intCast(fx.get("experts").?.integer);
    const ranks: u32 = @intCast(fx.get("ranks").?.integer);
    for (0..ranks) |r| {
        var p = try xs.plan(std.testing.allocator, &c, &names, @intCast(r), ranks);
        defer p.deinit();
        var want: std.ArrayList(Span) = .empty;
        for (fx.get("spans").?.array.items) |v| {
            const s = v.array.items;
            if (s[0].integer != r) continue;
            try want.append(a, .{ .shard = @intCast(s[1].integer), .off = @intCast(s[2].integer), .len = @intCast(s[3].integer), .row = @intCast(s[4].integer), .at = @intCast(s[5].integer), .width = @intCast(s[6].integer) });
        }
        var got: std.ArrayList(Span) = .empty;
        for (p.copies.items) |cp| try got.append(a, .{ .shard = cp.shard, .off = cp.off, .len = cp.len, .row = cp.row, .at = cp.at, .width = cp.width });
        std.mem.sort(Span, want.items, {}, lt);
        std.mem.sort(Span, got.items, {}, lt);
        try std.testing.expectEqualSlices(Span, want.items, got.items);
        // every tensor read or, for the markers, checked by name; the experts bound and tabled for the kernels
        try std.testing.expectEqual(names.count(), p.used.count());
        try std.testing.expectEqual(@as(u32, 4 * 3 * 2), p.markers); // layers 3 and 45, experts 0-3, gate/up/down
        const base: u64 = 1 << 40;
        const l = p.bind(3, base).?;
        try std.testing.expectEqual(@as(u32, 4096), l.dims);
        try std.testing.expectEqual(@as(u32, 1024), l.width);
        try std.testing.expectEqual([2]u32{ 8, 8 }, l.k2_gu);
        const e = p.experts[3].?;
        var t: [36 * 4]u8 = undefined;
        p.tables(3, base, &t);
        try std.testing.expectEqual(base + e.gate.trellis.off + (2 << 20), std.mem.readInt(u64, t[8..16], .little)); // expert 1: 2 MiB on
        try std.testing.expectEqual(base + e.down.trellis.off, std.mem.readInt(u64, t[64..72], .little));
        try std.testing.expectEqual(@as(u32, 8), std.mem.readInt(u32, t[24 * 4 + 4 * 11 ..][0..4], .little));
        const g = p.bindGlm(3, base).?; // exl3_mm's stacked words: expert e's trellis 2 MiB (K N / 2) past the first
        try std.testing.expectEqual(base + e.gate.trellis.off, g.gt);
        try std.testing.expectEqual(base + e.down.trellis.off, g.dt);
        try std.testing.expectEqual(@as(u32, 1024), g.width);
    }
}
