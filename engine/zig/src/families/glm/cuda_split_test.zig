//! The `.python` split against python-0.6 split.py itself: fixtures_cuda_split.json (by fixtures_cuda_split.py), tensor by tensor.
const std = @import("std");
const split = @import("cuda_split.zig");
const synth = @import("cuda_synth.zig");

const io = std.testing.io;
const Sha256 = std.crypto.hash.sha2.Sha256;

fn tmpPath(a: std.mem.Allocator, tmp: std.testing.TmpDir) ![]u8 {
    // std.testing.tmpDir parents the dir at <cwd>/.zig-cache/tmp
    const cwd = try std.process.currentPathAlloc(io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

fn hex(bytes: []const u8) [64]u8 {
    var sha: [32]u8 = undefined;
    Sha256.hash(bytes, &sha, .{});
    return std.fmt.bytesToHex(sha, .lower);
}

const Place = struct { shard: u32, off: u64, len: u64, name: []const u8 };

fn byOffset(_: void, x: Place, y: Place) bool {
    return if (x.shard != y.shard) x.shard < y.shard else x.off < y.off;
}

/// The tensor a copy reads from: the one whose bytes hold its first byte.
fn owner(places: []const Place, shard: u32, off: u64) ?[]const u8 {
    for (places) |p| if (p.shard == shard and off >= p.off and off < p.off + p.len) return p.name;
    return null;
}

test "each TP2 rank's python slices equal python-0.6 split.py's, tensor by tensor, on the same synthetic checkpoint" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fx = (try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures_cuda_split.json"), .{})).object;
    const shards: u32 = @intCast(fx.get("shards").?.integer);
    const dir = try tmpPath(a, tmp);
    const c = synth.tiny();
    const names = try synth.write(a, dir, &c, shards);
    // the files python-0.6 split are these very bytes
    var files = fx.get("files").?.object.iterator();
    while (files.next()) |f| {
        const bytes = try tmp.dir.readFileAlloc(io, f.key_ptr.*, a, .limited(1 << 30));
        try std.testing.expectEqualStrings(f.value_ptr.string, &hex(bytes));
    }
    var places: std.ArrayList(Place) = .empty;
    var it = names.iterator();
    while (it.next()) |e| try places.append(a, .{ .shard = e.value_ptr.shard, .off = e.value_ptr.off, .len = e.value_ptr.len, .name = e.key_ptr.* });
    std.mem.sort(Place, places.items, {}, byOffset);
    const tensors = fx.get("tensors").?.object; // name: [split.py's rule, rank 0's sha256, rank 1's sha256]
    for ([_]u32{ 0, 1 }) |r| {
        var p = try split.plan(std.testing.allocator, &(try split.rankConfig(c, r, 2)), &names, .python);
        defer p.deinit();
        const groups = try synth.execute(a, dir, shards, &p);
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        for (p.copies.items) |cp| {
            const name = owner(places.items, cp.shard, cp.off) orelse return error.CopyOfNoTensor;
            try std.testing.expect(!(try seen.getOrPut(a, name)).found_existing); // one copy a tensor
            const want = (tensors.get(name) orelse return error.TensorNotInFixture).array.items;
            const got = hex(groups[cp.g][@intCast(cp.dst)..][0..@intCast(cp.kept())]);
            std.testing.expectEqualStrings(want[1 + r].string, &got) catch |err| {
                std.debug.print("rank {d}: {s} ({s} split in python-0.6) differs\n", .{ r, name, want[0].string });
                return err;
            };
        }
        try std.testing.expectEqual(tensors.count(), seen.count()); // every tensor split.py keeps, and no other
    }
}
