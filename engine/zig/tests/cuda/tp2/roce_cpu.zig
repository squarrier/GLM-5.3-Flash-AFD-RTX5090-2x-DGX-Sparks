//! tf-tp2-gather roce-cpu: roce_proxy.c over the real libibverbs with the CPU in fn_roce.cu's place, two ranks over the control link and no GPU.
//! It addresses the region as fn_roce does (send slot, size and doorbell in ctrl, a flag a port, the receive slot), so the proxy, layout and rails run on soft RoCE or the lab's NICs.
const std = @import("std");
const cuda = @import("cuda");
const cases = @import("cases.zig");
const link_mod = cuda.link;
const Watch = cuda.watch.Watch;
const Allocator = std.mem.Allocator;

// roce_proxy.c's entry points, declared as roce.zig declares them
const Proxy = opaque {};
extern fn tf_roce_blob_bytes() u64;
extern fn tf_roce_layout(world: c_int, slot_bytes: u64, out: *[5]u64) c_int;
extern fn tf_roce_open(world: c_int, rank: c_int, names: [*]const [*:0]const u8, ports: c_int, gid_index: c_int, region: *anyopaque, region_bytes: u64, slot_bytes: u64, err: [*]u8, err_len: u64) ?*Proxy;
extern fn tf_roce_blob_of(r: *Proxy, out: *anyopaque) c_int;
extern fn tf_roce_connect(r: *Proxy, peer: *const anyopaque) c_int;
extern fn tf_roce_start(r: *Proxy) c_int;
extern fn tf_roce_failed(r: *Proxy) c_int;
extern fn tf_roce_error(r: *Proxy) [*:0]const u8;
extern fn tf_roce_ops(r: *Proxy) u64;
extern fn tf_roce_close(r: *Proxy) void;

/// fn_roce.cu's geometry: two slots, two ports at most, a flag of 32 words (128 bytes).
const slots = 2;
const ports_max = 2;
const flag_words = 32;

/// The gather the fault replaces: rank 1's 10th.
const fault_at = 10;

/// comm.zig's frame for the RoCE setup: one byte (1: opened), then the blob.
const roce_tag: link_mod.Tag = @enumFromInt(0x524f4345);

pub const Options = struct {
    rank: u32,
    master: std.Io.net.IpAddress,
    iters: usize,
    fault: cases.Flags.Fault = .none,
    sigterm_at: ?f64 = null,
    link_timeout_ms: u32 = 600_000,
};

/// Bytes a rank from 16 to `slot`: GLM decode rows (16 KiB each) up to the slot, and odd unit counts that split unevenly over two ports.
pub fn sizes(slot: u64, out: *[10]u32) []const u32 {
    const all = [_]u32{ 16, 48, 4096, 16016, 16 << 10, 7 * (16 << 10), 16 * (16 << 10), 32 * (16 << 10) };
    var n: usize = 0;
    for (all) |b| if (b <= slot) {
        out[n] = b;
        n += 1;
    };
    if (slot > all[all.len - 1] and slot <= std.math.maxInt(u32)) {
        out[n] = @intCast(slot);
        n += 1;
    }
    return out[0..n];
}

/// fn_roce's flag for (source rank, slot, port), in words from the flags' base.
pub fn flagWord(src: u32, slot: u32, port: usize) usize {
    return ((src * slots + slot) * ports_max + port) * flag_words;
}

fn seconds(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9;
}

/// Opens the proxy on NCCL_IB_HCA's ports, swaps blobs as comm.zig does, connects and starts it; then the gathers. Returns the exit code.
pub fn run(gpa: Allocator, io: std.Io, o: Options) !u8 {
    cuda.watch.installStop();
    const env = cuda.comm.Options.fromEnv();
    var names_buf: [cuda.roce.max_ports][]const u8 = undefined;
    const names = cuda.roce.parseHcas(env.hcas orelse "", &names_buf) catch |e| {
        std.debug.print("FAIL rank {d}: NCCL_IB_HCA ({s}) names no usable port: {t}\n", .{ o.rank, env.hcas orelse "unset", e });
        return 2;
    };
    var zbuf: [cuda.roce.max_ports][64:0]u8 = undefined;
    var zptr: [cuda.roce.max_ports][*:0]const u8 = undefined;
    for (names, 0..) |n, i| zptr[i] = (try std.mem.printSentinel(&zbuf[i], "{s}", .{n}, 0)).ptr;
    var lay: [5]u64 = undefined;
    if (tf_roce_layout(2, env.roce_max, &lay) != 0) {
        std.debug.print("FAIL rank {d}: TF_TP_ROCE_MAX {d} is not a multiple of 4096\n", .{ o.rank, env.roce_max });
        return 2;
    }
    const region = try std.heap.page_allocator.alloc(u8, @intCast(lay[4]));
    defer std.heap.page_allocator.free(region);
    @memset(region, 0);

    var server: ?link_mod.Server = null;
    defer if (server) |s| s.close();
    const link = if (o.rank == 0) blk: {
        server = try link_mod.Server.open(o.master);
        break :blk try server.?.accept(@intCast(o.link_timeout_ms));
    } else try link_mod.Link.join(io, o.master, o.link_timeout_ms);
    defer link.close();
    var text_buf: [160]u8 = undefined;
    const text = try std.fmt.bufPrint(&text_buf, "tf-tp2-gather roce-cpu v1 iters={d} slot={d} ports={d} gid={d} fault={t}", .{ o.iters, env.roce_max, names.len, env.gid_index, o.fault });
    link.agree(gpa, o.rank, text) catch |e| {
        std.debug.print("FAIL rank {d}: the ranks' settings differ ({t}); this rank: {s}\n", .{ o.rank, e, text });
        return 2;
    };

    var err: [256]u8 = @splat(0);
    const opened = tf_roce_open(2, @intCast(o.rank), &zptr, @intCast(names.len), env.gid_index, region.ptr, lay[4], env.roce_max, &err, err.len);
    defer if (opened) |p| tf_roce_close(p);
    if (opened == null) std.debug.print("FAIL rank {d}: RoCE open on {s}: {s}\n", .{ o.rank, env.hcas.?, std.mem.sliceTo(&err, 0) });
    const n = tf_roce_blob_bytes();
    const mine = try gpa.alloc(u8, @intCast(1 + n));
    defer gpa.free(mine);
    @memset(mine, 0);
    if (opened) |p| mine[0] = @intFromBool(tf_roce_blob_of(p, mine[1..].ptr) == 0);
    var theirs: []u8 = undefined;
    if (o.rank == 0) {
        try link.send(roce_tag, mine);
        theirs = try link.expect(gpa, roce_tag);
    } else {
        theirs = try link.expect(gpa, roce_tag);
        try link.send(roce_tag, mine);
    }
    defer gpa.free(theirs);
    if (mine[0] != 1 or theirs.len != mine.len or theirs[0] != 1) {
        std.debug.print("FAIL rank {d}: RoCE did not open on both ranks (this rank {d}, the peer {d})\n", .{ o.rank, mine[0], if (theirs.len > 0) theirs[0] else 0 });
        return 2;
    }
    const proxy = opened.?;
    if (tf_roce_connect(proxy, theirs[1..].ptr) != 0 or tf_roce_start(proxy) != 0) {
        std.debug.print("FAIL rank {d}: RoCE connect or start: {s}\n", .{ o.rank, tf_roce_error(proxy) });
        return 1;
    }
    std.debug.print("INFO rank {d}: proxy up on {s} (GID {d}), slot {d} KiB, region {d} bytes, layout recv {d} flag {d} send {d} ctrl {d}\n", .{ o.rank, env.hcas.?, env.gid_index, env.roce_max >> 10, lay[4], lay[0], lay[1], lay[2], lay[3] });

    var w: Watch = .{ .io = io, .fd = link.fd, .rank = o.rank };
    try w.start();
    // on an error return too: the watch stops before the link closes, so the close is not taken for a loss
    defer w.deinit();
    var g: Gathers = .{ .gpa = gpa, .io = io, .o = o, .proxy = proxy, .region = region, .lay = lay, .slot = env.roce_max, .ports = names.len, .w = &w };
    const rc = g.all() catch |e| {
        if (Watch.isLoss(e) or w.peerGone(1000)) w.fail("the other rank is gone (this rank's link failed)");
        return e;
    };
    if (rc != 0) return rc;
    // a clean end: quiet first, so the other rank closing after its answer is no loss
    w.quiet();
    if (o.rank == 0) {
        try link.send(.stop, "");
        gpa.free(try link.expect(gpa, .ack));
    } else {
        gpa.free(try link.expect(gpa, .stop));
        try link.send(.ack, "");
    }
    return 0;
}

const Gathers = struct {
    gpa: Allocator,
    io: std.Io,
    o: Options,
    proxy: *Proxy,
    region: []u8,
    lay: [5]u64,
    slot: u64,
    ports: usize,
    w: *Watch,
    seq: u32 = 0,

    fn words(g: *const Gathers, off: u64) [*]u32 {
        return @ptrCast(@alignCast(g.region.ptr + @as(usize, @intCast(off))));
    }

    /// Every size `iters` times: this rank's shard into its send slot, the size and the doorbell (fn_roce's steps 1-2), every port's flag, the peer's shard checked.
    fn all(g: *Gathers) !u8 {
        const peer = 1 - g.o.rank;
        const ctrl = g.words(g.lay[3]);
        const flags = g.words(g.lay[1]);
        const most: usize = @intCast(g.slot / 4);
        const mine = try g.gpa.alloc(u32, most);
        defer g.gpa.free(mine);
        const theirs = try g.gpa.alloc(u32, most);
        defer g.gpa.free(theirs);
        const times = try g.gpa.alloc(f64, g.o.iters);
        defer g.gpa.free(times);
        var size_buf: [10]u32 = undefined;
        const t_start = std.Io.Clock.awake.now(g.io).toNanoseconds();
        var wrong: usize = 0;
        for (sizes(g.slot, &size_buf)) |bytes| {
            const n = bytes / 4;
            const t_size = std.Io.Clock.awake.now(g.io).toNanoseconds();
            for (times) |*t| {
                if (g.o.sigterm_at) |at| if (seconds(g.io, t_start) >= at) {
                    std.debug.print("INFO rank {d}: SIGTERM to itself at gather {d}\n", .{ g.o.rank, g.seq });
                    _ = std.c.raise(.TERM);
                };
                g.seq += 1;
                const seq = g.seq;
                const slot = seq % slots;
                cases.fill(mine[0..n], seq, g.o.rank);
                cases.fill(theirs[0..n], seq, peer);
                const send = g.region[@intCast(g.lay[2] + slot * g.slot)..][0..bytes];
                @memcpy(send, std.mem.sliceAsBytes(mine[0..n]));
                var ring = seq;
                var size: u32 = bytes;
                if (g.o.rank == 1 and seq == fault_at) switch (g.o.fault) {
                    .none => {},
                    .size => size = 8,
                    .skip => ring = seq + 2,
                };
                if (g.o.rank == 1 and seq == fault_at and g.o.fault != .none) std.debug.print("INFO rank 1: injected fault {t} at gather {d}: size {d}, doorbell {d}\n", .{ g.o.fault, seq, size, ring });
                g.w.arm(g.w.base_s);
                const t0 = std.Io.Clock.awake.now(g.io).toNanoseconds();
                @atomicStore(u32, &ctrl[4 + slot], size, .monotonic);
                @atomicStore(u32, &ctrl[0], ring, .release);
                for (0..g.ports) |p| {
                    const f = &flags[flagWord(peer, slot, p)];
                    var spins: u64 = 0;
                    while (@atomicLoad(u32, f, .acquire) != seq) : (spins += 1) {
                        if (spins & 1023 == 1023 and tf_roce_failed(g.proxy) != 0) {
                            std.debug.print("FAIL rank {d}: this rank's RoCE proxy failed at gather {d}: {s}\n", .{ g.o.rank, seq, tf_roce_error(g.proxy) });
                            return 1;
                        }
                        std.atomic.spinLoopHint();
                    }
                }
                t.* = @as(f64, @floatFromInt(std.Io.Clock.awake.now(g.io).toNanoseconds() - t0)) / 1e3;
                g.w.disarm();
                const recv = g.region[@intCast(g.lay[0] + (peer * slots + slot) * g.slot)..][0..bytes];
                if (!std.mem.eql(u8, recv, std.mem.sliceAsBytes(theirs[0..n]))) {
                    const got = try g.gpa.alloc(u32, n);
                    defer g.gpa.free(got);
                    @memcpy(std.mem.sliceAsBytes(got), recv);
                    const d = cases.diff(got, theirs[0..n]).?;
                    if (wrong < 5) std.debug.print("FAIL rank {d} gather {d} ({d} B, slot {d}): {d} of {d} words differ, first at {d}\n", .{ g.o.rank, seq, bytes, slot, d.count, n, d.first });
                    wrong += 1;
                }
            }
            // a round is short when the peer was ahead (its flag already there), so the lockstep figure is the wall time a gather
            const lockstep = seconds(g.io, t_size) * 1e6 / @as(f64, @floatFromInt(times.len));
            const s = cases.stats(times);
            std.debug.print("RESULT rank {d} roce-cpu {d} B/rank over {d} port(s): lockstep {d:.1} us a gather | round p50 {d:.1} us p90 {d:.1}\n", .{ g.o.rank, bytes, g.ports, lockstep, s.p50, s.p90 });
        }
        // this rank's last post may trail the peer's flags (they prove the peer's posts, not this rank's): wait for it, a second at most
        var waited: u32 = 0;
        while (tf_roce_ops(g.proxy) < g.seq and waited < 1000) : (waited += 1) std.Io.sleep(g.io, .fromMilliseconds(1), .awake) catch {};
        const ops = tf_roce_ops(g.proxy);
        if (tf_roce_failed(g.proxy) != 0) {
            std.debug.print("FAIL rank {d}: the proxy failed: {s}\n", .{ g.o.rank, tf_roce_error(g.proxy) });
            return 1;
        }
        if (ops != g.seq or wrong > 0) {
            std.debug.print("FAIL rank {d}: {d} of {d} gathers wrong; the proxy posted {d} of {d}\n", .{ g.o.rank, wrong, g.seq, ops, g.seq });
            return 1;
        }
        std.debug.print("PASS rank {d}: {d} gathers over RoCE bit-exact, each slot reused in turn, the proxy posted every one ({d:.1} s)\n", .{ g.o.rank, g.seq, seconds(g.io, t_start) });
        return 0;
    }
};

test "sizes: 16 B to the slot, odd unit counts, the slot itself when larger" {
    var buf: [10]u32 = undefined;
    const d = sizes(512 << 10, &buf);
    try std.testing.expectEqual(@as(usize, 8), d.len);
    try std.testing.expectEqual(@as(u32, 16), d[0]);
    try std.testing.expectEqual(@as(u32, 512 << 10), d[d.len - 1]);
    try std.testing.expect((16016 / 16) % 2 == 1);
    try std.testing.expectEqual(@as(usize, 5), sizes(16 << 10, &buf).len);
    const big = sizes(1 << 20, &buf);
    try std.testing.expectEqual(@as(u32, 1 << 20), big[big.len - 1]);
    for (big) |b| try std.testing.expect(b % 16 == 0);
}

test "flags: fn_roce's addressing matches roce_proxy.c's (source, slot, port), 128 bytes apart" {
    try std.testing.expectEqual(@as(usize, 0), flagWord(0, 0, 0));
    try std.testing.expectEqual(@as(usize, 32), flagWord(0, 0, 1));
    try std.testing.expectEqual(@as(usize, 64), flagWord(0, 1, 0));
    try std.testing.expectEqual(@as(usize, 128), flagWord(1, 0, 0));
    try std.testing.expectEqual(@as(usize, 7 * 32), flagWord(1, 1, 1));
}
