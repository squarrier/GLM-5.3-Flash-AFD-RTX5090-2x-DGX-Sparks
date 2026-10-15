//! The loader's host checks on a synthetic checkpoint: the index against what was written, the reader against plain preads.
const std = @import("std");
const cfg = @import("config.zig");
const split = @import("cuda_split.zig");
const index = @import("cuda_index.zig");
const read = @import("cuda_read.zig");
const digest = @import("cuda_digest.zig");
const synth = @import("cuda_synth.zig");

const io = std.testing.io;

fn tmpPath(a: std.mem.Allocator, tmp: std.testing.TmpDir) ![]u8 {
    // std.testing.tmpDir parents the dir at <cwd>/.zig-cache/tmp
    const cwd = try std.process.currentPathAlloc(io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

test "the index reads every shard's header as the synthetic checkpoint was written, the vision tensor aside" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmpPath(a, tmp);
    const c = synth.tiny();
    const written = try synth.write(a, dir, &c, 3);
    var ix = try index.open(std.testing.allocator, dir);
    defer ix.deinit();
    try std.testing.expectEqual(@as(usize, 3), ix.shards.len);
    try std.testing.expectEqual(@as(u32, 1), ix.others);
    try std.testing.expectEqual(@as(u32, 0), ix.strays);
    try std.testing.expectEqual(written.count(), ix.names.count());
    var it = written.iterator();
    while (it.next()) |e| try std.testing.expectEqualDeep(e.value_ptr.*, ix.names.get(e.key_ptr.*).?);
}

test "a tensor of no known part counts as unread, so the whole model refuses to load past it" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmpPath(a, tmp);
    const c = synth.tiny();
    _ = try synth.writeExtra(a, dir, &c, 2, &.{"model.mm_projector.weight"});
    std.testing.log_level = .err; // quiet the stray's warning, which this test causes (the runner resets the level each test)
    var ix = try index.open(std.testing.allocator, dir);
    defer ix.deinit();
    try std.testing.expectEqual(@as(u32, 1), ix.strays);
    try std.testing.expectEqual(@as(u32, 1), ix.others);
    var p = try split.plan(std.testing.allocator, &(try split.rankConfig(c, 0, 2)), &ix.names, .python);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 1), ix.unread(&p.used));
}

test "each rank's groups, read through the ring or one read at a time, merged or not, equal plain preads of its plan" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmpPath(a, tmp);
    const c = synth.tiny();
    _ = try synth.write(a, dir, &c, 3);
    var ix = try index.open(std.testing.allocator, dir);
    defer ix.deinit();
    const cases = [_]read.Options{
        .{},
        .{ .ring = false, .gap = 0 },
        .{ .slot_bytes = 16 << 10, .slots = 4, .gap = 0 }, // every large copy cut, sliced copies by whole rows
        .{ .slot_bytes = 64 << 10, .slots = 2, .gap = 1 << 30, .ring = false },
    };
    for ([_]split.Split{ .metal, .python }) |how| for ([_][2]u32{ .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 } }) |rr| {
        const rc = try split.rankConfig(c, rr[0], rr[1]);
        var p = try split.plan(std.testing.allocator, &rc, &ix.names, how);
        defer p.deinit();
        try std.testing.expectEqual(@as(usize, 0), ix.unread(&p.used));
        const want = try synth.execute(a, dir, 3, &p);
        for (cases) |o| {
            var s = try read.schedule(std.testing.allocator, p.copies.items, o);
            defer s.deinit();
            const groups = try a.alloc([]u8, p.groups.items.len);
            for (groups, p.groups.items) |*g, n| g.* = try a.alloc(u8, @intCast(n));
            for (groups) |g| @memset(g, 0);
            var host: read.HostSink = .{ .groups = groups };
            const st = try read.run(std.testing.allocator, ix.shards, &s, host.sink(), o);
            for (want, groups) |w, g| try std.testing.expectEqualSlices(u8, w, g);
            var kept: u64 = 0;
            for (p.copies.items) |cp| kept += cp.kept();
            try std.testing.expectEqual(kept, st.kept);
            try std.testing.expectEqual(s.reads.len, st.reads);
            try std.testing.expect(s.bytes() >= p.sourceBytes());
        }
        // the digests a load would give, group by group from the files, equal the oracle's groups hashed
        const ds = try digest.fromFiles(std.testing.allocator, ix.shards, &p, .{ .slot_bytes = 64 << 10, .slots = 3 }, 3);
        defer std.testing.allocator.free(ds);
        for (want, ds) |w, d| {
            var sha: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(w, &sha, .{});
            try std.testing.expectEqualSlices(u8, &sha, &d.sha);
            try std.testing.expectEqual(@as(u64, w.len), d.bytes);
        }
    };
}

test "a rank reads its own rows: the two TP2 ranks' kept bytes are disjoint halves except what both hold whole" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmpPath(a, tmp);
    const c = synth.tiny();
    const names = try synth.write(a, dir, &c, 2);
    var one = try split.plan(std.testing.allocator, &(try split.rankConfig(c, 0, 1)), &names, .metal);
    defer one.deinit();
    var totals: [2]u64 = .{ 0, 0 };
    for ([_]split.Split{ .metal, .python }, &totals) |how, *total| for ([_]u32{ 0, 1 }) |r| {
        var p = try split.plan(std.testing.allocator, &(try split.rankConfig(c, r, 2)), &names, how);
        defer p.deinit();
        var kept: u64 = 0;
        for (p.copies.items) |cp| kept += cp.kept();
        try std.testing.expect(kept < one.sourceBytes());
        total.* += kept;
    };
    try std.testing.expect(totals[0] > one.sourceBytes()); // the replicated tensors (embedding, mixes, shared and MTP experts) count twice
    try std.testing.expect(totals[1] < totals[0]); // split.py halves the shared and MTP experts too
}
