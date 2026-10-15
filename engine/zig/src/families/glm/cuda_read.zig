//! The plan's copies read with O_DIRECT through io_uring, neighbours merged into one read, each landed copy handed on.
const std = @import("std");
const linux = std.os.linux;
const core = @import("core");
const split = @import("cuda_split.zig");

const dio = core.direct_io;

pub const Options = struct {
    slot_bytes: usize = 64 << 20, // one read's most; a larger copy goes as several
    slots: u32 = 8, // reads in flight
    gap: u64 = 64 << 10, // the most unread bytes one read spans between two copies (the Python RankReader's 1 MiB reads 5.5 % more at TP2)
    ring: bool = true, // io_uring where the kernel allows it; else each read completes as it is asked for
};

/// Shard bytes [off, off + len) in one read, carrying the copies [first, first + count).
pub const Read = struct { shard: u32, off: u64, len: u64, first: u32, count: u32 };

/// The copies in read order, cut to fit a slot, and the reads that carry them.
pub const Schedule = struct {
    gpa: std.mem.Allocator,
    copies: []split.Copy,
    reads: []Read,

    pub fn deinit(s: *Schedule) void {
        s.gpa.free(s.copies);
        s.gpa.free(s.reads);
    }

    /// Bytes the reads ask for (merged gaps included).
    pub fn bytes(s: *const Schedule) u64 {
        var n: u64 = 0;
        for (s.reads) |r| n += r.len;
        return n;
    }

    /// Bytes the reads take from the file system: each widened to whole 4 KiB blocks, as O_DIRECT reads it.
    pub fn spanBytes(s: *const Schedule) u64 {
        var n: u64 = 0;
        for (s.reads) |r| n += dio.span(r.off, r.len);
        return n;
    }
};

fn before(_: void, a: split.Copy, b: split.Copy) bool {
    return if (a.shard != b.shard) a.shard < b.shard else a.off < b.off;
}

/// The copies by shard and offset, each cut to fit a slot (a sliced copy by whole rows), neighbours within `gap` merged.
pub fn schedule(gpa: std.mem.Allocator, copies: []const split.Copy, o: Options) !Schedule {
    const cap: u64 = dio.fits(o.slot_bytes);
    var cut: std.ArrayList(split.Copy) = .empty;
    errdefer cut.deinit(gpa);
    for (copies) |c| {
        if (c.len == 0) continue;
        const step: u64 = if (c.row == 0) cap else cap / c.row * c.row;
        if (step == 0) return error.SlotTooSmall;
        var done: u64 = 0;
        while (done < c.len) : (done += step) {
            var p = c;
            p.off = c.off + done;
            p.len = @min(step, c.len - done);
            p.dst = c.dst + (if (c.row == 0) done else done / c.row * c.width);
            try cut.append(gpa, p);
        }
    }
    std.mem.sort(split.Copy, cut.items, {}, before);
    var reads: std.ArrayList(Read) = .empty;
    errdefer reads.deinit(gpa);
    for (cut.items, 0..) |c, i| {
        if (reads.items.len > 0) {
            const r = &reads.items[reads.items.len - 1];
            const end = r.off + r.len;
            if (r.shard == c.shard and c.off < end) return error.OverlappingCopies;
            if (r.shard == c.shard and c.off - end <= o.gap and c.off + c.len - r.off <= cap) {
                r.len = c.off + c.len - r.off;
                r.count += 1;
                continue;
            }
        }
        try reads.append(gpa, .{ .shard = c.shard, .off = c.off, .len = c.len, .first = @intCast(i), .count = 1 });
    }
    return .{ .gpa = gpa, .copies = try cut.toOwnedSlice(gpa), .reads = try reads.toOwnedSlice(gpa) };
}

/// Where landed bytes go: `put` gets each copy's kept bytes; after `flush` returns the slot holding them is reused.
pub const Sink = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, g: u32, dst: u64, bytes: []const u8) anyerror!void,
    flush: *const fn (ctx: *anyopaque) anyerror!void,
};

pub const Stats = struct {
    reads: u64 = 0,
    bytes: u64 = 0, // asked for
    span: u64 = 0, // read, in whole blocks
    kept: u64 = 0, // landed in the groups
    retried: u64 = 0, // short or failed ring reads finished by a plain read
    ns: u64 = 0,
    ring: bool = false,
    direct: bool = false, // every shard opened O_DIRECT

    pub fn gbPerS(s: Stats) f64 {
        return if (s.ns == 0) 0 else @as(f64, @floatFromInt(s.bytes)) / @as(f64, @floatFromInt(s.ns));
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Runner = struct {
    s: *const Schedule,
    files: []dio.File,
    slots: []dio.Buffer,
    sink: Sink,
    stats: Stats = .{},

    /// A read's bytes, then each of its copies' kept bytes (sliced copies compacted in place) to the sink.
    fn land(x: *Runner, r: Read, got: []u8) !void {
        for (x.s.copies[r.first..][0..r.count]) |c| {
            const src = got[@intCast(c.off - r.off)..][0..@intCast(c.len)];
            if (c.row == 0) {
                try x.sink.put(x.sink.ctx, c.g, c.dst, src);
            } else {
                const rows: usize = @intCast(c.len / c.row);
                for (0..rows) |rr| std.mem.copyForwards(u8, src[rr * c.width ..][0..c.width], src[rr * c.row + c.at ..][0..c.width]);
                try x.sink.put(x.sink.ctx, c.g, c.dst, src[0 .. rows * c.width]);
            }
            x.stats.kept += c.kept();
        }
        try x.sink.flush(x.sink.ctx);
        x.stats.reads += 1;
        x.stats.bytes += r.len;
        x.stats.span += dio.span(r.off, r.len);
    }

    /// Each read in turn: the path where io_uring is unavailable.
    fn plain(x: *Runner) !void {
        for (x.s.reads) |r| try x.land(r, try x.files[r.shard].read(x.slots[0], r.off, r.len));
    }

    /// Up to one read per slot in flight; each lands as it completes.
    fn ringed(x: *Runner, ring: *linux.IoUring) !void {
        var pending: [64]?u32 = @splat(null);
        var in_flight: usize = 0;
        var next: usize = 0;
        errdefer while (in_flight > 0) {
            var cq: [64]linux.io_uring_cqe = undefined;
            const n = ring.copy_cqes(&cq, 1) catch break;
            in_flight -= n;
        };
        while (next < x.s.reads.len or in_flight > 0) {
            for (x.slots, 0..) |slot, k| {
                if (next >= x.s.reads.len) break;
                if (pending[k] != null) continue;
                const r = x.s.reads[next];
                const lo = std.mem.alignBackward(u64, r.off, dio.alignment);
                _ = try ring.read(k, x.files[r.shard].fd, .{ .buffer = slot[0..dio.span(r.off, r.len)] }, lo);
                pending[k] = @intCast(next);
                next += 1;
                in_flight += 1;
            }
            _ = try ring.submit();
            var cqes: [64]linux.io_uring_cqe = undefined;
            const n = try ring.copy_cqes(cqes[0..x.slots.len], 1);
            for (cqes[0..n]) |c| {
                const k: usize = @intCast(c.user_data);
                const r = x.s.reads[pending[k].?];
                pending[k] = null;
                in_flight -= 1;
                const lo = std.mem.alignBackward(u64, r.off, dio.alignment);
                const want = r.off - lo + r.len;
                const got = if (c.res >= 0 and @as(u64, @intCast(c.res)) >= want) x.slots[k][@intCast(r.off - lo)..][0..@intCast(r.len)] else blk: {
                    x.stats.retried += 1; // a short or failed read is finished by a plain read of the same span
                    break :blk try x.files[r.shard].read(x.slots[k], r.off, r.len);
                };
                try x.land(r, got);
            }
        }
    }
};

/// Every read of `s` from `shards` into `o.slots` aligned slots, each copy's bytes to `sink`; refused if a read failed.
pub fn run(gpa: std.mem.Allocator, shards: []const [:0]const u8, s: *const Schedule, sink: Sink, o: Options) !Stats {
    if (o.slots == 0 or o.slots > 64) return error.BadSlots;
    const files = try gpa.alloc(dio.File, shards.len);
    defer gpa.free(files);
    var opened: usize = 0;
    defer for (files[0..opened]) |*f| f.close();
    var direct = true;
    while (opened < shards.len) : (opened += 1) {
        files[opened] = try dio.File.open(shards[opened]);
        direct = direct and files[opened].direct;
    }
    const slots = try gpa.alloc(dio.Buffer, o.slots);
    defer gpa.free(slots);
    var made: usize = 0;
    defer for (slots[0..made]) |b| gpa.free(b);
    while (made < o.slots) : (made += 1) slots[made] = try gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), o.slot_bytes);
    var x: Runner = .{ .s = s, .files = files, .slots = slots, .sink = sink };
    x.stats.direct = direct;
    const t0 = nowNs();
    var ring: ?linux.IoUring = if (o.ring) linux.IoUring.init(@intCast(std.math.ceilPowerOfTwoAssert(u32, o.slots)), 0) catch |err| blk: {
        std.log.info("io_uring unavailable ({t}): checkpoint reads run one at a time", .{err});
        break :blk null;
    } else null;
    defer if (ring) |*r| r.deinit();
    if (ring) |*r| {
        x.stats.ring = true;
        try x.ringed(r);
    } else try x.plain();
    x.stats.ns = nowNs() - t0;
    return x.stats;
}

/// Groups in host memory (tests and the CPU tool).
pub const HostSink = struct {
    groups: [][]u8,

    pub fn sink(h: *HostSink) Sink {
        return .{ .ctx = h, .put = put, .flush = flush };
    }

    fn put(ctx: *anyopaque, g: u32, dst: u64, bytes: []const u8) anyerror!void {
        const h: *HostSink = @ptrCast(@alignCast(ctx));
        @memcpy(h.groups[g][@intCast(dst)..][0..bytes.len], bytes);
    }

    fn flush(_: *anyopaque) anyerror!void {}
};

/// Landed bytes counted and dropped: the read path alone, with no memory for the weights.
pub const NullSink = struct {
    bytes: u64 = 0,

    pub fn sink(n: *NullSink) Sink {
        return .{ .ctx = n, .put = put, .flush = flush };
    }

    fn put(ctx: *anyopaque, _: u32, _: u64, bytes: []const u8) anyerror!void {
        const n: *NullSink = @ptrCast(@alignCast(ctx));
        n.bytes += bytes.len;
    }

    fn flush(_: *anyopaque) anyerror!void {}
};

test "a schedule cuts copies to the slot, slices by whole rows and merges neighbours within the gap" {
    const gpa = std.testing.allocator;
    const o: Options = .{ .slot_bytes = 16384, .gap = 100 };
    const cap = dio.fits(o.slot_bytes); // 8192
    const copies = [_]split.Copy{
        .{ .shard = 1, .off = 0, .len = 3 * cap, .g = 0, .dst = 0 }, // cut into three
        .{ .shard = 0, .off = 1000, .len = 10, .g = 1, .dst = 0 },
        .{ .shard = 0, .off = 1050, .len = 10, .g = 1, .dst = 16 }, // 40 bytes on: merged
        .{ .shard = 0, .off = 2000, .len = 10, .g = 1, .dst = 32 }, // 940 bytes on: its own read
        .{ .shard = 2, .off = 0, .len = 3000 * 4, .g = 2, .dst = 0, .row = 3000, .at = 1000, .width = 1000 }, // two rows a cut
    };
    var s = try schedule(gpa, &copies, o);
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 3 + 3 + 2), s.copies.len);
    try std.testing.expectEqual(@as(usize, 2 + 3 + 2), s.reads.len);
    try std.testing.expectEqual(Read{ .shard = 0, .off = 1000, .len = 60, .first = 0, .count = 2 }, s.reads[0]);
    try std.testing.expectEqual(@as(u64, 2 * cap), s.copies[5].dst);
    try std.testing.expectEqual(@as(u64, 2000), s.copies[7].dst); // the second cut's two rows land 2 x 1000 bytes on
    const overlap = [_]split.Copy{ .{ .shard = 0, .off = 0, .len = 10, .g = 0, .dst = 0 }, .{ .shard = 0, .off = 5, .len = 10, .g = 0, .dst = 10 } };
    try std.testing.expectError(error.OverlappingCopies, schedule(gpa, &overlap, o));
}
