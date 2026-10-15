//! A rank's groups by sha256: from the checkpoint as the plan lands them (no GPU), or read back from the device after a load.
const std = @import("std");
const cuda = @import("cuda");
const split = @import("cuda_split.zig");
const read = @import("cuda_read.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;

/// Group `g`'s length and sha256.
pub const Digest = struct { g: u32, bytes: u64, sha: [32]u8 };

/// One group's bytes in host memory: the sink of a schedule restricted to that group.
const One = struct {
    g: u32,
    buf: []u8,

    fn sink(o: *One) read.Sink {
        return .{ .ctx = o, .put = put, .flush = flush };
    }

    fn put(ctx: *anyopaque, g: u32, dst: u64, bytes: []const u8) anyerror!void {
        const o: *One = @ptrCast(@alignCast(ctx));
        if (g != o.g) return error.WrongGroup;
        @memcpy(o.buf[@intCast(dst)..][0..bytes.len], bytes);
    }

    fn flush(_: *anyopaque) anyerror!void {}
};

fn largest(sizes: []const u64) usize {
    var n: u64 = 0;
    for (sizes) |x| n = @max(n, x);
    return @intCast(n);
}

/// Workers taking groups in turn from the checkpoint: each reads one group into its own buffer and hashes it.
const Files = struct {
    gpa: std.mem.Allocator,
    shards: []const [:0]const u8,
    p: *const split.Plan,
    o: read.Options,
    out: []Digest,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn run(f: *Files) void {
        f.work() catch |err| {
            std.log.err("glm: hashing a group from the checkpoint failed: {t}", .{err});
            f.failed.store(true, .release);
        };
    }

    fn work(f: *Files) !void {
        const buf = try f.gpa.alloc(u8, largest(f.p.groups.items));
        defer f.gpa.free(buf);
        var mine: std.ArrayList(split.Copy) = .empty;
        defer mine.deinit(f.gpa);
        while (!f.failed.load(.acquire)) {
            const g = f.next.fetchAdd(1, .monotonic);
            if (g >= f.out.len) return;
            const n: usize = @intCast(f.p.groups.items[g]);
            mine.clearRetainingCapacity();
            for (f.p.copies.items) |c| if (c.g == g) try mine.append(f.gpa, c);
            @memset(buf[0..n], 0); // the padding between tensors hashes as zeros, as the load leaves it
            var s = try read.schedule(f.gpa, mine.items, f.o);
            defer s.deinit();
            var one: One = .{ .g = @intCast(g), .buf = buf[0..n] };
            _ = try read.run(f.gpa, f.shards, &s, one.sink(), f.o);
            f.out[g] = .{ .g = @intCast(g), .bytes = n, .sha = undefined };
            Sha256.hash(one.buf, &f.out[g].sha, .{});
        }
    }
};

/// Each group read on its own and hashed, `threads` groups at once: what a load puts on the device, with no GPU.
pub fn fromFiles(gpa: std.mem.Allocator, shards: []const [:0]const u8, p: *const split.Plan, o: read.Options, threads: u32) ![]Digest {
    const out = try gpa.alloc(Digest, p.groups.items.len);
    errdefer gpa.free(out);
    var f: Files = .{ .gpa = gpa, .shards = shards, .p = p, .o = o, .out = out };
    try together(threads, &f, Files.run);
    if (f.failed.load(.acquire)) return error.HashFailed;
    return out;
}

/// `run(ctx)` on `threads` threads, the caller's among them, joined.
fn together(threads: u32, ctx: anytype, comptime run: fn (@TypeOf(ctx)) void) !void {
    var workers: [15]?std.Thread = @splat(null);
    const extra = @min(@max(threads, 1) - 1, workers.len);
    for (workers[0..extra]) |*t| t.* = std.Thread.spawn(.{}, run, .{ctx}) catch null;
    run(ctx);
    for (workers[0..extra]) |t| if (t) |th| th.join();
}

/// A batch of downloaded groups hashed on as many threads.
const Batch = struct {
    bufs: [][]u8,
    out: []Digest,
    next: std.atomic.Value(usize) = .init(0),

    fn run(b: *Batch) void {
        while (true) {
            const k = b.next.fetchAdd(1, .monotonic);
            if (k >= b.out.len) return;
            Sha256.hash(b.bufs[k], &b.out[k].sha, .{});
        }
    }
};

/// The device groups downloaded `threads` at a time on this thread (the context's), each batch hashed on `threads` threads.
pub fn fromDevice(gpa: std.mem.Allocator, groups: []const cuda.DeviceBuffer, sizes: []const u64, threads: u32) ![]Digest {
    const t: usize = @min(@max(threads, 1), 16);
    const out = try gpa.alloc(Digest, groups.len);
    errdefer gpa.free(out);
    const bufs = try gpa.alloc([]u8, t);
    defer gpa.free(bufs);
    var made: usize = 0;
    defer for (bufs[0..made]) |b| gpa.free(b);
    while (made < t) : (made += 1) bufs[made] = try gpa.alloc(u8, largest(sizes));
    var g: usize = 0;
    var views: [16][]u8 = undefined;
    while (g < groups.len) {
        const k = @min(t, groups.len - g);
        for (0..k) |j| {
            const n: usize = @intCast(sizes[g + j]);
            views[j] = bufs[j][0..n];
            try groups[g + j].download(0, views[j]);
            out[g + j] = .{ .g = @intCast(g + j), .bytes = n, .sha = undefined };
        }
        var b: Batch = .{ .bufs = views[0..k], .out = out[g..][0..k] };
        try together(@intCast(k), &b, Batch.run);
        g += k;
    }
    return out;
}

/// One sha256 over every group's length and digest in order: a rank's weights in one line.
pub fn combined(ds: []const Digest) [32]u8 {
    var h: Sha256 = .init(.{});
    for (ds) |d| {
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, d.bytes, .little);
        h.update(&len);
        h.update(&d.sha);
    }
    return h.finalResult();
}

test "digests hash a batch on several threads as on one" {
    var bytes: [3][100]u8 = undefined;
    for (&bytes, 0..) |*b, i| @memset(b, @intCast(i + 1));
    var bufs = [_][]u8{ &bytes[0], &bytes[1], &bytes[2] };
    var out: [3]Digest = undefined;
    for (&out, 0..) |*d, i| d.* = .{ .g = @intCast(i), .bytes = 100, .sha = undefined };
    var b: Batch = .{ .bufs = &bufs, .out = &out };
    try together(3, &b, Batch.run);
    for (bytes, out) |x, d| {
        var want: [32]u8 = undefined;
        Sha256.hash(&x, &want, .{});
        try std.testing.expectEqualSlices(u8, &want, &d.sha);
    }
}
