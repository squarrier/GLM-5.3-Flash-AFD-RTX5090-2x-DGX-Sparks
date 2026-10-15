//! The two ranks' communicator: NCCL on every rail plus the one-shot RoCE all-gather for small gathers, every exchange on the caller's stream (graph-capturable).
//! Source: Mia's patch 0006, MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, zig/src/families/flashnext/cuda_comm.zig (Apache-2.0, Copyright 2026 MiaAI-Lab).
//! Changed here: family-agnostic (TF_FLASHNEXT_ROCE, _ROCE_MAX, _L2PF are TF_TP_ROCE, _ROCE_MAX, _L2PF), Options read once, pathOf, allGatherNccl, roceOps, roceLimits.
//! Rank 0 makes the NCCL id and hands it to rank 1 over the control link (link.zig); the launcher names both rails in NCCL_IB_HCA (the PCIe twins) and sets 4 channels.
//! TF_TP_ROCE (default on, 0 off): gathers up to TF_TP_ROCE_MAX a rank (512 KiB) go over roce.zig on NCCL_IB_HCA's ports (GID NCCL_IB_GID_INDEX, default 3).
//! RoCE is used only when both ranks opened it (each says so over the link); larger gathers, broadcasts and point to point stay on NCCL; TF_TP_L2PF (default on) prefetches.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const nccl = @import("nccl.zig");
const roce = @import("roce.zig");
const link_mod = @import("link.zig");
const Link = link_mod.Link;
const Allocator = std.mem.Allocator;

pub const Prefetch = roce.Prefetch;

/// Which transport a gather takes: both ranks make the same choice, since they carve their buffers alike.
pub const Path = enum { roce, nccl };

/// The control-link frame of the RoCE setup: one byte (1: opened) then the rank's connection blob.
const roce_tag: link_mod.Tag = @enumFromInt(0x524f4345);

/// What the communicator turns on; `fromEnv` reads 0006's settings (renamed TF_TP_*), the defaults are hers.
pub const Options = struct {
    /// the one-shot RoCE all-gather for gathers it fits (TF_TP_ROCE; "0" off)
    roce: bool = true,
    /// the next kernels' weights prefetched into the L2 during a gather (TF_TP_L2PF; "0" off)
    l2_prefetch: bool = true,
    /// the largest shard a rank RoCE takes, a multiple of 4096 (TF_TP_ROCE_MAX)
    roce_max: u64 = 512 << 10,
    /// the rails' RDMA devices (NCCL_IB_HCA); null leaves every gather on NCCL
    hcas: ?[]const u8 = null,
    /// the RoCE v2 GID index (NCCL_IB_GID_INDEX)
    gid_index: c_int = 3,
    /// how long a RoCE wait lasts before its kernel traps (fn_roce.cu), unchanged from 0006
    roce_timeout_s: u64 = 600,

    pub fn fromEnv() Options {
        return fromLookup(getenv);
    }

    fn getenv(name: [:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
    }

    /// The options from `lookup` (an environment): unset or malformed numbers keep the defaults, as 0006 parses them.
    pub fn fromLookup(lookup: *const fn ([:0]const u8) ?[]const u8) Options {
        var o: Options = .{};
        if (lookup("TF_TP_ROCE")) |v| o.roce = !std.mem.eql(u8, v, "0");
        if (lookup("TF_TP_L2PF")) |v| o.l2_prefetch = !std.mem.eql(u8, v, "0");
        if (lookup("TF_TP_ROCE_MAX")) |v| o.roce_max = std.fmt.parseInt(u64, v, 10) catch o.roce_max;
        if (lookup("NCCL_IB_GID_INDEX")) |v| o.gid_index = std.fmt.parseInt(c_int, v, 10) catch o.gid_index;
        o.hcas = lookup("NCCL_IB_HCA");
        return o;
    }
};

/// The RoCE all-gather, the L2 prefetch and the driver they were opened with (heap: Comm is copied by value).
const Extra = struct { driver: Driver, r: ?roce.Roce = null, l2: ?roce.L2 = null, prefetch: bool = false };

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    extra: ?*Extra = null,

    /// Join the two-rank communicator on the current CUDA context: rank 0 sends the NCCL id first, rank 1 waits for it.
    pub fn init(gpa: Allocator, link: Link, rank: u32, world: u32, o: Options) !Comm {
        if (world != 2 or rank >= world) return error.UnsupportedWorld;
        var lib = nccl.Library.open() catch |e| {
            std.log.err("cannot open libnccl.so.2 ({s}): two ranks need NCCL", .{@errorName(e)});
            return e;
        };
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(.nccl_id, &id.internal);
        } else {
            const bytes = try link.expect(gpa, .nccl_id);
            defer gpa.free(bytes);
            if (bytes.len != id.internal.len) return error.BadNcclId;
            @memcpy(&id.internal, bytes);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        var c: Comm = .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
        errdefer _ = lib.api.ncclCommDestroy(comm);
        errdefer if (c.extra) |x| freeExtra(x);
        var driver: ?Driver = null;
        if (o.roce or o.l2_prefetch) driver = Driver.open() catch |e| blk: {
            std.log.warn("RoCE / L2 prefetch: no CUDA driver handle ({t}); NCCL alone", .{e});
            break :blk null;
        };
        if (driver) |d| {
            const x = try std.heap.c_allocator.create(Extra);
            x.* = .{ .driver = d };
            c.extra = x;
        }
        // both ranks always say whether they opened RoCE (one rank's setting alone never leaves the other waiting)
        const r = try handshake(gpa, link, rank, if (o.roce) (if (c.extra) |x| &x.driver else null) else null, o);
        if (c.extra) |x| {
            x.r = r;
            if (o.l2_prefetch) {
                x.l2 = roce.L2.init(&x.driver) catch |e| blk: {
                    std.log.warn("L2 prefetch off: {t}", .{e});
                    break :blk null;
                };
                x.prefetch = x.l2 != null;
            }
        } else if (r) |rr| {
            var keep = rr;
            keep.deinit();
        }
        return c;
    }

    /// Opens the RoCE side (none without a driver), tells the peer whether it did, and connects when both did; null when either did not.
    fn handshake(gpa: Allocator, link: Link, rank: u32, d: ?*const Driver, o: Options) !?roce.Roce {
        var r: ?roce.Roce = if (d) |dd| openRoce(dd, rank, o) else null;
        var keep = false;
        defer if (!keep) if (r) |*x| x.deinit();
        const n = roce.Roce.blobBytes();
        const mine = try gpa.alloc(u8, 1 + n);
        defer gpa.free(mine);
        @memset(mine, 0);
        if (r) |*x| {
            mine[0] = 1;
            x.blob(mine[1..]) catch {
                mine[0] = 0;
            };
        }
        var theirs: []u8 = undefined;
        if (rank == 0) {
            try link.send(roce_tag, mine);
            theirs = try link.expect(gpa, roce_tag);
        } else {
            theirs = try link.expect(gpa, roce_tag);
            try link.send(roce_tag, mine);
        }
        defer gpa.free(theirs);
        const peer_ok = theirs.len == 1 + n and theirs[0] == 1;
        if (mine[0] != 1 or !peer_ok) {
            std.log.warn("RoCE all-gather off: this rank {s}, the peer {s}; NCCL carries every gather", .{ if (mine[0] == 1) "opened it" else if (d == null) "has it off (TF_TP_ROCE=0)" else "could not open it", if (peer_ok) "opened it" else "did not" });
            return null;
        }
        // both opened: from here a failure is fatal on both (the peer then times out in its first gather)
        try r.?.connect(theirs[1..]);
        try r.?.start();
        std.log.info("RoCE all-gather on {d} port(s) for gathers up to {d} KiB a rank", .{ r.?.ports, r.?.slot_bytes >> 10 });
        keep = true;
        return r;
    }

    fn openRoce(d: *const Driver, rank: u32, o: Options) ?roce.Roce {
        const hcas = o.hcas orelse {
            std.log.warn("RoCE: NCCL_IB_HCA is not set", .{});
            return null;
        };
        var names: [roce.max_ports][]const u8 = undefined;
        const ports = roce.parseHcas(hcas, &names) catch |e| {
            std.log.warn("RoCE: NCCL_IB_HCA: {t}", .{e});
            return null;
        };
        return roce.Roce.open(d, rank, .{ .ports = ports, .gid_index = o.gid_index, .slot_bytes = o.roce_max, .timeout_s = o.roce_timeout_s }) catch |e| {
            std.log.warn("RoCE: {t}", .{e});
            return null;
        };
    }

    pub fn deinit(c: *Comm) void {
        _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        if (c.extra) |x| freeExtra(x);
        c.extra = null;
    }

    fn freeExtra(x: *Extra) void {
        if (x.r) |*r| r.deinit();
        if (x.l2) |*l| l.deinit();
        x.driver.close();
        std.heap.c_allocator.destroy(x);
    }

    /// True when small gathers go over RoCE.
    pub fn roceOn(c: *const Comm) bool {
        return if (c.extra) |x| x.r != null else false;
    }

    /// The gathers RoCE's proxy has posted so far (0 when RoCE is off).
    pub fn roceOps(c: *const Comm) u64 {
        const x = c.extra orelse return 0;
        const r = x.r orelse return 0;
        return r.ops();
    }

    /// The RoCE ports and the largest shard a rank it takes (0 when RoCE is off).
    pub fn roceLimits(c: *const Comm) struct { ports: u32, slot_bytes: u64 } {
        const x = c.extra orelse return .{ .ports = 0, .slot_bytes = 0 };
        const r = x.r orelse return .{ .ports = 0, .slot_bytes = 0 };
        return .{ .ports = r.ports, .slot_bytes = r.slot_bytes };
    }

    /// The RoCE side's proxy failure or a kernel's timed-out wait, if any, as text in `buf` (the stream traps on its own).
    pub fn roceFailure(c: *const Comm, buf: []u8) ?[]const u8 {
        const x = c.extra orelse return null;
        const r = &(x.r orelse return null);
        if (r.failure()) |f| return f;
        if (r.timedOut()) |t| return std.fmt.bufPrint(buf, "RoCE gather {d} waited past its timeout for rank {d}", .{ t.seq, t.peer }) catch "RoCE gather timed out";
        return null;
    }

    /// The peer is gone: end the communicator's pending collectives so a waiting stream returns (any thread).
    pub fn abort(c: *const Comm) void {
        _ = c.lib.api.ncclCommAbort(c.comm);
    }

    /// The transport allGather picks for these buffers: RoCE when it takes the layout, else NCCL.
    pub fn pathOf(c: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize) Path {
        const x = c.extra orelse return .nccl;
        const r = x.r orelse return .nccl;
        return if (r.takes(send, recv, bytes)) .roce else .nccl;
    }

    /// `count` elements of `dtype` from every rank into `recv` [world * count], rank 0's first, on `stream`.
    pub fn allGather(c: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: abi.Stream) !void {
        return c.allGatherThen(send, recv, count, dtype, stream, .{});
    }

    /// allGather, with `next` (the next kernels' weights) prefetched into the L2 meanwhile when the prefetch is on.
    pub fn allGatherThen(c: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: abi.Stream, next: Prefetch) !void {
        if (c.extra) |x| {
            const pf: Prefetch = if (x.prefetch) next else .{};
            const bytes = count * size(dtype);
            // RoCE takes every size it fits whose layout it can; both ranks carve their buffers alike, so no rank picks the other path
            if (x.r) |*r| if (r.takes(send, recv, bytes)) return r.allGather(send, recv, bytes, stream, pf);
            if (x.l2) |*l| if (pf.bytes > 0) try l.run(pf, stream);
        }
        try c.allGatherNccl(send, recv, count, dtype, stream);
    }

    /// The all-gather on NCCL whatever its size (NCCL's in-place form included: `send` = `recv` + rank * count elements).
    pub fn allGatherNccl(c: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: abi.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream), "ncclAllGather");
    }

    /// NCCL's in-place all-gather (the prompt glue's row halves): never RoCE, on both ranks whatever the size.
    pub fn allGatherInPlace(c: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: abi.Stream) !void {
        try c.allGatherNccl(send, recv, count, dtype, stream);
    }

    /// `bytes` bytes of `buf` from `root` to every rank, in place, on `stream`.
    pub fn broadcast(c: *const Comm, buf: abi.DevicePtr, bytes: usize, root: u32, stream: abi.Stream) !void {
        try c.lib.check(c.lib.api.ncclBroadcast(buf, buf, bytes, .u8, @intCast(root), c.comm, stream), "ncclBroadcast");
    }

    /// Two ranks swap `count` elements in one NCCL group (a send and a receive, the same order on both); the buffers must not overlap.
    pub fn exchange(c: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: abi.Stream) !void {
        const peer: c_int = @intCast(1 - c.rank);
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        c.lib.check(c.lib.api.ncclSend(send, count, dtype, peer, c.comm, stream), "ncclSend") catch |e| {
            _ = c.lib.api.ncclGroupEnd();
            return e;
        };
        c.lib.check(c.lib.api.ncclRecv(recv, count, dtype, peer, c.comm, stream), "ncclRecv") catch |e| {
            _ = c.lib.api.ncclGroupEnd();
            return e;
        };
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    }

    /// One NCCL group of point-to-point pairs with the peer: each `sends[i]` (address, bytes) goes out, each `recvs[i]` comes in, the same order on both.
    pub fn exchangeBytes(c: *const Comm, sends: []const [2]u64, recvs: []const [2]u64, stream: abi.Stream) !void {
        const peer: c_int = @intCast(1 - c.rank);
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        errdefer _ = c.lib.api.ncclGroupEnd();
        for (sends) |sd| if (sd[1] > 0) try c.lib.check(c.lib.api.ncclSend(sd[0], sd[1], .u8, peer, c.comm, stream), "ncclSend");
        for (recvs) |rv| if (rv[1] > 0) try c.lib.check(c.lib.api.ncclRecv(rv[0], rv[1], .u8, peer, c.comm, stream), "ncclRecv");
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    }

    /// The fp32 partials of a [rows, width] block into [world, rows, width] (a tensor-parallel layer's outputs).
    pub fn gatherPartials(c: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, rows: usize, width: usize, stream: abi.Stream) !void {
        try c.allGather(send, recv, rows * width, .f32, stream);
    }

    /// gatherPartials with `next` prefetched meanwhile (allGatherThen).
    pub fn gatherPartialsThen(c: *const Comm, send: abi.DevicePtr, recv: abi.DevicePtr, rows: usize, width: usize, stream: abi.Stream, next: Prefetch) !void {
        try c.allGatherThen(send, recv, rows * width, .f32, stream, next);
    }
};

pub fn size(t: nccl.DataType) usize {
    return switch (t) {
        .i8, .u8 => 1,
        .f16, .bf16 => 2,
        .i32, .u32, .f32 => 4,
        .i64, .u64, .f64 => 8,
    };
}

test "the communicator refuses worlds other than two" {
    // no GPU or NCCL needed: the check runs before the library opens
    const l: Link = .{ .fd = -1 };
    try std.testing.expectError(error.UnsupportedWorld, Comm.init(std.testing.allocator, l, 0, 1, .{}));
    try std.testing.expectError(error.UnsupportedWorld, Comm.init(std.testing.allocator, l, 2, 2, .{}));
}

test "options: 0006's defaults, \"0\" turns a feature off, a malformed number keeps its default" {
    const Env = struct {
        var pairs: []const [2][]const u8 = &.{};
        fn get(name: [:0]const u8) ?[]const u8 {
            for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
            return null;
        }
    };
    Env.pairs = &.{};
    const d = Options.fromLookup(Env.get);
    try std.testing.expect(d.roce and d.l2_prefetch and d.hcas == null);
    try std.testing.expectEqual(@as(u64, 512 << 10), d.roce_max);
    try std.testing.expectEqual(@as(c_int, 3), d.gid_index);
    try std.testing.expectEqual(@as(u64, 600), d.roce_timeout_s);
    Env.pairs = &.{ .{ "TF_TP_ROCE", "0" }, .{ "TF_TP_L2PF", "0" }, .{ "TF_TP_ROCE_MAX", "1048576" }, .{ "NCCL_IB_GID_INDEX", "1" }, .{ "NCCL_IB_HCA", "rocep1s0f0,roceP2p1s0f0" } };
    const o = Options.fromLookup(Env.get);
    try std.testing.expect(!o.roce and !o.l2_prefetch);
    try std.testing.expectEqual(@as(u64, 1 << 20), o.roce_max);
    try std.testing.expectEqual(@as(c_int, 1), o.gid_index);
    try std.testing.expectEqualStrings("rocep1s0f0,roceP2p1s0f0", o.hcas.?);
    Env.pairs = &.{ .{ "TF_TP_ROCE", "1" }, .{ "TF_TP_ROCE_MAX", "big" }, .{ "NCCL_IB_GID_INDEX", "x" } };
    const m = Options.fromLookup(Env.get);
    try std.testing.expect(m.roce);
    try std.testing.expectEqual(@as(u64, 512 << 10), m.roce_max);
    try std.testing.expectEqual(@as(c_int, 3), m.gid_index);
}

test "a communicator without RoCE sends every gather to NCCL" {
    const c: Comm = .{ .lib = undefined, .comm = null, .rank = 0, .world = 2 };
    try std.testing.expectEqual(Path.nccl, c.pathOf(0x1000, 0x10000, 16 << 10));
    try std.testing.expect(!c.roceOn());
    try std.testing.expectEqual(@as(u64, 0), c.roceLimits().slot_bytes);
}
