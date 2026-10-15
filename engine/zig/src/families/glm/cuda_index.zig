//! A checkpoint folder's index and shard headers as the split reads them: names, dtypes, shapes and byte ranges, no weights.
const std = @import("std");
const core = @import("core");
const split = @import("cuda_split.zig");

const st = core.safetensors;

/// The language model's tensors (`model.language_model.*`, `lm_head.*`), the shard files and what else they hold.
pub const Index = struct {
    arena: std.heap.ArenaAllocator,
    shards: []const [:0]const u8 = &.{},
    sizes: []const u64 = &.{}, // each shard's length
    names: split.Names = .empty,
    others: u32 = 0, // the vision tower's tensors (`model.visual.*`), never read
    other_bytes: u64 = 0,
    strays: u32 = 0, // tensors of no known part: counted as unread, so a renamed tensor is never dropped in silence

    pub fn deinit(ix: *Index) void {
        ix.arena.deinit();
    }

    /// Language-model tensors the plan does not read, and stray ones, logged (the first five) and counted.
    pub fn unread(ix: *const Index, used: *const std.StringHashMapUnmanaged(void)) usize {
        var n: usize = ix.strays;
        var it = ix.names.keyIterator();
        while (it.next()) |k| {
            if (used.contains(k.*)) continue;
            if (n < 5) std.log.err("glm: checkpoint tensor {s} is not read", .{k.*});
            n += 1;
        }
        return n;
    }
};

fn dtypeOf(d: st.DType) split.DType {
    return switch (d) {
        .u32 => .u32,
        .bf16 => .bf16,
        .f32 => .f32,
        .f16 => .f16, // EXL3's scales
        .i16 => .i16, // EXL3's trellises
        .i32 => .i32, // EXL3's codebook markers
        else => .other,
    };
}

/// `len` bytes at `at` of an open file, all of them or an error.
fn preadAll(fd: std.c.fd_t, out: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < out.len) {
        const n = std.c.pread(fd, out.ptr + done, out.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

fn readFile(a: std.mem.Allocator, path: [:0]const u8, most: usize) ![]u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0 or end > most) return error.BadIndex;
    const out = try a.alloc(u8, @intCast(end));
    try preadAll(fd, out, 0);
    return out;
}

/// `dir`'s model.safetensors.index.json and every shard it names (sorted), headers parsed and checked against each file's length.
pub fn open(gpa: std.mem.Allocator, dir: []const u8) !Index {
    var ix: Index = .{ .arena = .init(gpa) };
    errdefer ix.deinit();
    const a = ix.arena.allocator();
    const index_path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors.index.json", .{dir}, 0);
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, try readFile(a, index_path, 1 << 26), .{});
    const map = (doc.object.get("weight_map") orelse return error.BadIndex).object;
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    var it = map.iterator();
    while (it.next()) |kv| try files.put(a, kv.value_ptr.string, {});
    const names = try a.dupe([]const u8, files.keys());
    std.mem.sort([]const u8, names, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    const shards = try a.alloc([:0]const u8, names.len);
    const sizes = try a.alloc(u64, names.len);
    for (names, shards, sizes, 0..) |name, *path, *size, si| {
        path.* = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
        size.* = try addShard(&ix, path.*, @intCast(si));
    }
    ix.shards = shards;
    ix.sizes = sizes;
    return ix;
}

/// One shard's header: its language-model tensors into `names`, the rest counted; returns the file's length.
fn addShard(ix: *Index, path: [:0]const u8, si: u32) !u64 {
    const a = ix.arena.allocator();
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) {
        std.log.err("glm: cannot open shard {s}", .{path});
        return error.FileNotFound;
    }
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 8) return error.BadSafetensors;
    const size: u64 = @intCast(end);
    var head: [8]u8 = undefined;
    try preadAll(fd, &head, 0);
    const hlen = std.mem.readInt(u64, &head, .little);
    if (hlen > size - 8 or hlen > 1 << 28) return error.BadSafetensors;
    const json = try a.alloc(u8, @intCast(hlen));
    try preadAll(fd, json, 8);
    const data = 8 + hlen;
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
    if (doc != .object) return error.BadSafetensors;
    var all = doc.object.iterator();
    while (all.next()) |kv| {
        const k = kv.key_ptr.*;
        if (std.mem.eql(u8, k, "__metadata__") or std.mem.startsWith(u8, k, "model.language_model.") or std.mem.startsWith(u8, k, "lm_head.")) continue;
        if (!std.mem.startsWith(u8, k, "model.visual.")) {
            if (ix.strays < 5) std.log.warn("glm: checkpoint tensor {s} belongs to no part the loader knows", .{k});
            ix.strays += 1;
            continue;
        }
        ix.others += 1;
        if (kv.value_ptr.* == .object) if (kv.value_ptr.object.get("data_offsets")) |o| if (o == .array and o.array.items.len == 2) {
            ix.other_bytes += @intCast(o.array.items[1].integer - o.array.items[0].integer);
        };
    }
    for ([_][]const u8{ "model.language_model.", "lm_head." }) |p| {
        const h = try st.parseHeaderPrefix(a, json, @intCast(size - data), p);
        var hi = h.iterator();
        while (hi.next()) |e| {
            const v = e.value_ptr.*;
            var src: split.Src = .{ .shard = si, .off = data + v.begin, .len = v.end - v.begin, .dtype = dtypeOf(v.dtype), .rank = v.rank };
            @memcpy(src.shape[0..], v.shape[0..]);
            const got = try ix.names.getOrPut(a, e.key_ptr.*);
            if (got.found_existing) {
                std.log.err("glm: tensor {s} is in two shards", .{e.key_ptr.*});
                return error.DuplicateTensor;
            }
            got.value_ptr.* = src;
        }
    }
    return size;
}
