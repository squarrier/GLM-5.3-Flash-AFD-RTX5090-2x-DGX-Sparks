//! tf-tp2-gather: the two-rank layer (cuda/comm.zig: NCCL on both rails plus the one-shot RoCE all-gather) between two GPUs, weights-free.
//! run: each size 16 KiB to 32 MiB a rank over NCCL and, where it fits, over RoCE, bit-exact and timed; link: the failure path without a GPU; roce-cpu: the RoCE proxy without a GPU; rails; cases.
//! Adapted from Mia's tf-flashnext-comm (patch 0006, zig/tests/cuda/flashnext/comm_test.zig @ cbfc429, Apache-2.0). Exit codes: 0 pass, 1 fail, 2 could not run, 3 lost peer.
const std = @import("std");
const cuda = @import("cuda");
const cases = @import("tp2/cases.zig");
const rails = @import("tp2/rails.zig");
const link_mode = @import("tp2/link_mode.zig");
const roce_cpu = @import("tp2/roce_cpu.zig");
const Comm = cuda.comm.Comm;
const Path = cuda.comm.Path;
const Watch = cuda.watch.Watch;
const net = std.Io.net;
const Allocator = std.mem.Allocator;

const usage =
    \\usage: tf-tp2-gather run RANK MASTER PORT [--set all|glm|mia] [--iters N] [--warmup N] [--rails N] [--soak S] [--sigterm-at T] [--stall-at T]
    \\       tf-tp2-gather link RANK MASTER PORT [--hold S] [--sigterm-at T] [--stall-at T]
    \\       tf-tp2-gather roce-cpu RANK MASTER PORT [--iters N] [--fault size|skip] [--sigterm-at T]
    \\       tf-tp2-gather rails NETDEV [GID]
    \\       tf-tp2-gather cases
    \\
;

/// Seeds of the exactness trials (each a fresh fill of both ranks' blocks).
const seeds = [_]u64{ 0x5eed_0001, 0x5eed_0002 };

/// How long either rank waits for the other at the link (the other may still be starting its container).
const link_timeout_ms: u32 = 600_000;

fn getenv(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return bad();
    const cmd: []const u8 = args[1];
    if (std.mem.eql(u8, cmd, "cases")) return printCases();
    if (std.mem.eql(u8, cmd, "rails")) return railsCmd(init.gpa, init.io, args[2..]);
    const is_run = std.mem.eql(u8, cmd, "run");
    const is_link = std.mem.eql(u8, cmd, "link");
    const is_roce = std.mem.eql(u8, cmd, "roce-cpu");
    if (!is_run and !is_link and !is_roce) return bad();
    if (args.len < 5) return bad();
    const rank = std.fmt.parseInt(u32, args[2], 10) catch return bad();
    const port = std.fmt.parseInt(u16, args[4], 10) catch return bad();
    if (rank > 1) return bad();
    const master = net.IpAddress.parse(args[3], port) catch {
        std.debug.print("MASTER is an IPv4 or IPv6 address, not {s}\n", .{args[3]});
        return 2;
    };
    const flags = cases.Flags.parse(args[5..]) catch return bad();
    if (is_link) {
        link_mode.run(init.gpa, init.io, .{ .rank = rank, .master = master, .hold_s = flags.hold_s, .sigterm_at = flags.sigterm_at, .stall_at = flags.stall_at, .link_timeout_ms = link_timeout_ms }) catch |e| {
            std.debug.print("FAIL rank {d}: {t}\n", .{ rank, e });
            return 1;
        };
        return 0;
    }
    if (is_roce) {
        return roce_cpu.run(init.gpa, init.io, .{ .rank = rank, .master = master, .iters = flags.iters, .fault = flags.fault, .sigterm_at = flags.sigterm_at, .link_timeout_ms = link_timeout_ms }) catch |e| {
            std.debug.print("FAIL rank {d}: {t}\n", .{ rank, e });
            return 1;
        };
    }
    return runGpu(init.gpa, init.io, rank, master, flags);
}

fn bad() u8 {
    std.debug.print("{s}", .{usage});
    return 2;
}

fn printCases() u8 {
    for (cases.glm_cases ++ cases.mia_cases) |c| {
        std.debug.print("{s} [{d}x{d}] {d} B a rank", .{ c.name, c.rows, c.width, cases.bytesOf(c) });
        if (cases.figureFor(c.name)) |f| std.debug.print(" | Mia: {d:.1} us, {s}", .{ f.us, f.what });
        std.debug.print("\n", .{});
    }
    return 0;
}

fn railsCmd(gpa: Allocator, io: std.Io, args: []const [:0]const u8) u8 {
    if (args.len < 1) return bad();
    const gid: ?u32 = if (args.len > 1) std.fmt.parseInt(u32, args[1], 10) catch return bad() else null;
    var r = rails.find(gpa, io, "/sys", args[0], gid) catch |e| {
        std.debug.print("FAIL rails {s}: {t}\n", .{ args[0], e });
        return 1;
    };
    defer r.deinit();
    for (r.slice(), 0..) |x, i| {
        std.debug.print("INFO {s} {s}: rdma {s}, GID {d}", .{ if (i == 0) "link" else "twin", x.netdev, x.rdma, x.gid_index });
        if (rails.ipv4Of(x.gid)) |a| std.debug.print(" (RoCE v2, {d}.{d}.{d}.{d})", .{ a[0], a[1], a[2], a[3] });
        std.debug.print("\n", .{});
    }
    var buf: [256]u8 = undefined;
    std.debug.print("NCCL_IB_HCA={s}\nNCCL_IB_GID_INDEX={d}\nRAILS={d}\n", .{ r.hcaList(&buf) catch "?", r.rails[0].gid_index, r.n });
    return 0;
}

fn runGpu(gpa: Allocator, io: std.Io, rank: u32, master: net.IpAddress, f: cases.Flags) u8 {
    cuda.watch.installStop();
    var driver = cuda.Driver.open() catch |e| {
        std.debug.print("FAIL rank {d}: no CUDA driver ({t})\n", .{ rank, e });
        return 2;
    };
    defer driver.close();
    var ctx = cuda.Context.init(&driver, 0) catch |e| {
        std.debug.print("FAIL rank {d}: no CUDA context ({t})\n", .{ rank, e });
        return 2;
    };
    defer ctx.deinit();
    var r: Run = .{ .gpa = gpa, .io = io, .rank = rank, .f = f, .d = &driver };
    return r.main(&ctx, master) catch |e| {
        std.debug.print("FAIL rank {d}: {t}\n", .{ rank, e });
        return 1;
    };
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn us(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

const Run = struct {
    gpa: Allocator,
    io: std.Io,
    rank: u32,
    f: cases.Flags,
    d: *const cuda.Driver,
    link: cuda.link.Link = undefined,
    comm: Comm = undefined,
    watch: Watch = undefined,
    stream: cuda.Stream = undefined,
    send: cuda.DeviceBuffer = undefined,
    recv: cuda.DeviceBuffer = undefined,
    want: []u32 = &.{},
    got: []u32 = &.{},
    failed: usize = 0,

    fn main(r: *Run, ctx: *const cuda.Context, master: net.IpAddress) !u8 {
        var name_buf: [256]u8 = undefined;
        std.debug.print("INFO rank {d}: {s} sm_{d}, driver {d}\n", .{ r.rank, try ctx.name(&name_buf), try ctx.capability(), try r.d.version() });
        for ([_][:0]const u8{ "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_SOCKET_IFNAME", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "TF_TP_ROCE", "TF_TP_ROCE_MAX", "TF_TP_TIMEOUT_S" }) |n| {
            std.debug.print("INFO rank {d}: {s}={s}\n", .{ r.rank, n, getenv(n) orelse "(unset)" });
        }
        const set = try cases.select(r.f.set);
        const gate = cases.gates(getenv, r.f.rails, cuda.roce.proxyBuilt());
        std.debug.print("INFO rank {d}: RoCE proxy {s}; gates {s}\n", .{ r.rank, if (cuda.roce.proxyBuilt()) "built" else "stub", gate orelse "pass" });
        var server: ?cuda.link.Server = null;
        defer if (server) |s| s.close();
        r.link = if (r.rank == 0) blk: {
            server = try cuda.link.Server.open(master);
            break :blk try server.?.accept(@intCast(link_timeout_ms));
        } else try cuda.link.Link.join(r.io, master, link_timeout_ms);
        defer r.link.close();
        // both ranks run the same table, and both learn whether either failed its gates
        var text_buf: [256]u8 = undefined;
        const roce_wanted = if (getenv("TF_TP_ROCE")) |v| !std.mem.eql(u8, v, "0") else true;
        const head = try cases.settingsText(&text_buf, r.f.set, r.f.iters, r.f.warmup, roce_wanted);
        var full_buf: [320]u8 = undefined;
        const text = try std.fmt.bufPrint(&full_buf, "{s} soak={d} gates={s}", .{ head, r.f.soak_s, if (gate == null) "pass" else "fail" });
        r.link.agree(r.gpa, r.rank, text) catch |e| {
            std.debug.print("FAIL rank {d}: the ranks' settings or gates differ ({t}); this rank: {s}\n", .{ r.rank, e, text });
            return 2;
        };
        if (gate) |why| {
            std.debug.print("FAIL rank {d}: gate: {s}\n", .{ r.rank, why });
            return 2;
        }
        const t_comm = now(r.io);
        r.comm = try Comm.init(r.gpa, r.link, r.rank, 2, cuda.comm.Options.fromEnv());
        defer r.comm.deinit();
        const lim = r.comm.roceLimits();
        std.debug.print("INFO rank {d}: NCCL {d}, joined in {d:.1} ms; RoCE {s} ({d} port(s), up to {d} KiB a rank)\n", .{ r.rank, try r.comm.lib.version(), us(now(r.io) - t_comm) / 1e3, if (r.comm.roceOn()) "on" else "off", lim.ports, lim.slot_bytes >> 10 });
        if (roce_wanted and !r.comm.roceOn()) {
            std.debug.print("FAIL rank {d}: RoCE is on (TF_TP_ROCE) but the all-gather did not open on both ranks\n", .{r.rank});
            return 1;
        }
        r.watch = .{ .io = r.io, .fd = r.link.fd, .rank = r.rank };
        try r.watch.start();
        defer r.watch.deinit();
        return r.work(set) catch |e| {
            // a collective or link error with the other rank gone is the watchdog's case: exit 3, whichever saw it first
            if (Watch.isLoss(e) or r.watch.peerGone(1000)) r.watch.fail("the other rank is gone (this rank's link or collective failed)");
            return e;
        };
    }

    fn work(r: *Run, set: []const cases.Case) !u8 {
        r.stream = try cuda.Stream.init(r.d, true);
        defer r.stream.deinit();
        const most = cases.mostBytes(set);
        r.send = try cuda.DeviceBuffer.alloc(r.d, most);
        defer r.send.free();
        r.recv = try cuda.DeviceBuffer.alloc(r.d, 2 * most);
        defer r.recv.free();
        r.want = try r.gpa.alloc(u32, 2 * most / 4);
        defer r.gpa.free(r.want);
        r.got = try r.gpa.alloc(u32, 2 * most / 4);
        defer r.gpa.free(r.got);
        for (set) |c| try r.case(c);
        try r.sequence();
        if (r.f.soak_s > 0 or r.f.sigterm_at != null or r.f.stall_at != null) try r.soak();
        // the end: rank 0 says stop, rank 1 acknowledges; both watches are quiet first, so the closes are no loss
        r.watch.quiet();
        if (r.rank == 0) {
            try r.link.send(.stop, "");
            r.gpa.free(try r.link.expect(r.gpa, .ack));
        } else {
            r.gpa.free(try r.link.expect(r.gpa, .stop));
            try r.link.send(.ack, "");
        }
        var why_buf: [256]u8 = undefined;
        if (r.comm.roceFailure(&why_buf)) |why| {
            std.debug.print("FAIL rank {d}: RoCE: {s}\n", .{ r.rank, why });
            r.failed += 1;
        }
        if (r.failed > 0) {
            std.debug.print("FAIL rank {d}: {d} check(s) failed\n", .{ r.rank, r.failed });
            return 1;
        }
        std.debug.print("PASS rank {d}: every gather bit-exact and in rank order on both ranks, over {s} ({d} sizes)\n", .{ r.rank, if (r.comm.roceOn()) "NCCL and RoCE" else "NCCL", set.len });
        return 0;
    }

    /// This rank's block (`count` words of `want`) into the send buffer, or in place into its block of recv, on the gather's stream: a legacy-stream copy from pageable memory may land after the gather reads it (qualification_6607f886, an RTX 5090).
    fn put(r: *Run, in_place: bool, count: usize) !void {
        const mine = std.mem.sliceAsBytes(r.want[r.rank * count ..][0..count]);
        if (in_place) try r.recv.uploadAsync(r.rank * count * 4, mine, r.stream.handle) else try r.send.uploadAsync(0, mine, r.stream.handle);
    }

    /// A trial's inputs on the gather's stream: this rank's block, and recv cleared to a marker the gather must overwrite.
    fn stage(r: *Run, count: usize) !void {
        try r.put(false, count);
        try r.recv.fill32(0xdead_beef, r.stream.handle);
    }

    fn gather(r: *Run, p: Path, count: usize) !void {
        return switch (p) {
            .roce => r.comm.allGather(r.send.ptr, r.recv.ptr, count, .f32, r.stream.handle),
            .nccl => r.comm.allGatherNccl(r.send.ptr, r.recv.ptr, count, .f32, r.stream.handle),
        };
    }

    /// Both ranks' received bytes compared over the link (a mismatch fails both); then this rank's words against the expected ones.
    fn check(r: *Run, what: []const u8, c: cases.Case, p: Path) !void {
        const n = 2 * cases.bytesOf(c) / 4;
        try r.recv.download(0, std.mem.sliceAsBytes(r.got[0..n]));
        if (cases.diff(r.got[0..n], r.want[0..n])) |d| {
            std.debug.print("FAIL rank {d} {s} {t} {s}: {d} of {d} words differ, first at {d} (block {d}): got 0x{x:0>8} want 0x{x:0>8}\n", .{ r.rank, c.name, p, what, d.count, n, d.first, d.first / (n / 2), r.got[d.first], r.want[d.first] });
            r.failed += 1;
        }
        r.link.agree(r.gpa, r.rank, std.mem.sliceAsBytes(r.got[0..n])) catch |e| {
            if (e != error.RanksDisagree) return e;
            std.debug.print("FAIL rank {d} {s} {t} {s}: the two ranks hold different bytes\n", .{ r.rank, c.name, p, what });
            r.failed += 1;
        };
    }

    fn case(r: *Run, c: cases.Case) !void {
        const bytes = cases.bytesOf(c);
        const auto = r.comm.pathOf(r.send.ptr, r.recv.ptr, bytes);
        // 0006's routing: the one-shot RoCE gather up to its slot, NCCL above it
        const lim = r.comm.roceLimits().slot_bytes;
        const expect: Path = if (r.comm.roceOn() and bytes <= lim) .roce else .nccl;
        if (auto != expect) {
            std.debug.print("FAIL rank {d} {s}: {d} B a rank routes to {t}, 0006 sends it to {t}\n", .{ r.rank, c.name, bytes, auto, expect });
            r.failed += 1;
        }
        std.debug.print("INFO rank {d} {s}: {d} B a rank, the engine's path {t}\n", .{ r.rank, c.name, bytes, auto });
        try r.arm(c, .nccl);
        if (auto == .roce) try r.arm(c, .roce);
    }

    fn arm(r: *Run, c: cases.Case, p: Path) !void {
        const count = cases.bytesOf(c) / 4;
        r.watch.arm(r.watch.base_s);
        defer r.watch.disarm();
        const ops0 = r.comm.roceOps();
        for (seeds) |seed| {
            cases.fill(r.want[0..count], seed, 0);
            cases.fill(r.want[count .. 2 * count], seed, 1);
            try r.stage(count);
            try r.gather(p, count);
            try r.stream.synchronize();
            try r.check("gather", c, p);
        }
        for (0..r.f.warmup) |_| try r.gather(p, count);
        try r.stream.synchronize();
        var start = try cuda.Event.init(r.d, true);
        defer start.deinit();
        var end = try cuda.Event.init(r.d, true);
        defer end.deinit();
        const host = try r.gpa.alloc(f64, r.f.iters);
        defer r.gpa.free(host);
        const dev = try r.gpa.alloc(f64, r.f.iters);
        defer r.gpa.free(dev);
        for (host, dev) |*h, *g| {
            const t0 = now(r.io);
            try start.record(r.stream);
            try r.gather(p, count);
            try end.record(r.stream);
            try end.synchronize();
            h.* = us(now(r.io) - t0);
            g.* = @as(f64, try cuda.Event.elapsedMs(start, end)) * 1e3;
        }
        // pipelined: the calls back to back, as a graph replays them, one synchronize at the end
        const t0 = now(r.io);
        for (0..r.f.iters) |_| try r.gather(p, count);
        try r.stream.synchronize();
        const piped = us(now(r.io) - t0) / @as(f64, @floatFromInt(r.f.iters));
        try r.check("after timing", c, p);
        if (p == .roce) {
            // this rank's proxy may still be posting the last gather (the peer's flags end the kernel, not this rank's post): a second at most
            const issued = seeds.len + r.f.warmup + 2 * r.f.iters;
            var waited: u32 = 0;
            while (r.comm.roceOps() - ops0 < issued and waited < 1000) : (waited += 1) std.Io.sleep(r.io, .fromMilliseconds(1), .awake) catch {};
            const posted = r.comm.roceOps() - ops0;
            if (posted != issued) {
                std.debug.print("FAIL rank {d} {s} roce: the proxy posted {d} gathers of {d}\n", .{ r.rank, c.name, posted, issued });
                r.failed += 1;
            }
        }
        const h = cases.stats(host);
        const g = cases.stats(dev);
        const b = cases.bytesOf(c);
        std.debug.print("RESULT rank {d} {s} [{d}x{d}] {d} B/rank {t}: host p50 {d:.1} us p90 {d:.1} | gpu p50 {d:.1} us p90 {d:.1} min {d:.1} | piped {d:.1} us/call | busbw gpu {d:.2} GB/s piped {d:.2} GB/s", .{ r.rank, c.name, c.rows, c.width, b, p, h.p50, h.p90, g.p50, g.p90, g.min, piped, cases.busbw(b, g.p50), cases.busbw(b, piped) });
        if (cases.figureFor(c.name)) |fig| std.debug.print(" | Mia {d:.1} us: gpu p50 x{d:.2}", .{ fig.us, g.p50 / fig.us });
        std.debug.print("\n", .{});
    }

    /// 300 back-to-back gathers on the engine's path (every third in NCCL's in-place form), sizes and payloads changing each call (Mia's sequence check).
    fn sequence(r: *Run) !void {
        const shapes = [_][2]usize{ .{ 1, cases.glm_width }, .{ 7, cases.glm_width }, .{ 16, cases.glm_width }, .{ 2, cases.glm_width }, .{ 3, 65 } };
        r.watch.arm(r.watch.base_s);
        defer r.watch.disarm();
        var wrong: usize = 0;
        for (0..300) |i| {
            const sh = shapes[i % shapes.len];
            const count = sh[0] * sh[1];
            cases.fill(r.want[0..count], 0xc0ffee00 + i, 0);
            cases.fill(r.want[count .. 2 * count], 0xc0ffee00 + i, 1);
            const in_place = i % 3 == 2;
            try r.put(in_place, count);
            if (in_place) {
                try r.comm.gatherPartials(r.recv.ptr + r.rank * count * 4, r.recv.ptr, sh[0], sh[1], r.stream.handle);
            } else {
                try r.comm.gatherPartials(r.send.ptr, r.recv.ptr, sh[0], sh[1], r.stream.handle);
            }
            try r.stream.synchronize();
            try r.recv.download(0, std.mem.sliceAsBytes(r.got[0 .. 2 * count]));
            if (!std.mem.eql(u32, r.got[0 .. 2 * count], r.want[0 .. 2 * count])) wrong += 1;
        }
        if (wrong > 0) {
            std.debug.print("FAIL rank {d} sequence: {d} of 300 changing gathers wrong\n", .{ r.rank, wrong });
            r.failed += 1;
        } else std.debug.print("PASS rank {d} sequence: 300 gathers of changing payloads, sizes and forms exact\n", .{r.rank});
    }

    /// Gathers in blocks of 100 (decode rows on the engine's path, every eighth 4 MiB on NCCL) until rank 0 stops them; the failure options act here.
    fn soak(r: *Run) !void {
        const t0 = now(r.io);
        const big = @min(4 << 20, r.send.len) / 4;
        var blocks: u64 = 0;
        while (true) : (blocks += 1) {
            var go: [1]u8 = .{@intFromBool(us(now(r.io) - t0) / 1e6 < r.f.soak_s)};
            if (r.rank == 0) try r.link.send(.round, &go) else {
                const b = try r.link.expect(r.gpa, .round);
                defer r.gpa.free(b);
                go[0] = if (b.len == 1) b[0] else 0;
            }
            if (go[0] == 0) break;
            for (0..100) |i| {
                const s = us(now(r.io) - t0) / 1e6;
                if (r.f.sigterm_at) |t| if (s >= t) {
                    std.debug.print("INFO rank {d}: SIGTERM to itself in soak block {d}\n", .{ r.rank, blocks });
                    _ = std.c.raise(.TERM);
                };
                if (r.f.stall_at) |t| if (s >= t) {
                    std.debug.print("INFO rank {d}: stalling with the link open in soak block {d}\n", .{ r.rank, blocks });
                    while (true) std.Io.sleep(r.io, .fromMilliseconds(1000), .awake) catch {};
                };
                r.watch.arm(r.watch.base_s);
                try r.gather(.roce, cases.glm_width);
                if (i % 8 == 7) try r.gather(.nccl, big);
                try r.stream.synchronize();
                r.watch.disarm();
            }
        }
        std.debug.print("INFO rank {d}: soak done, {d} blocks of 100 in {d:.1} s, RoCE ops {d}\n", .{ r.rank, blocks, us(now(r.io) - t0) / 1e6, r.comm.roceOps() });
    }
};

/// A driver whose copies and fills only count the stream they go on: the staging, checked without a GPU.
const Recorder = struct {
    var stream: cuda.abi.Stream = null;
    var on_stream: usize = 0;
    var elsewhere: usize = 0;
    var legacy: usize = 0;
    var last_dst: cuda.abi.DevicePtr = 0;

    fn copyAsync(dst: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, st: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (st == stream) on_stream += 1 else elsewhere += 1;
        last_dst = dst;
        return cuda.abi.success;
    }

    fn fillAsync(_: cuda.abi.DevicePtr, _: c_uint, _: usize, st: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (st == stream) on_stream += 1 else elsewhere += 1;
        return cuda.abi.success;
    }

    fn copy(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        legacy += 1;
        return cuda.abi.success;
    }

    fn fill(_: cuda.abi.DevicePtr, _: c_uint, _: usize) callconv(.c) cuda.abi.Result {
        legacy += 1;
        return cuda.abi.success;
    }
};

test "a gather's inputs and the cleared receive buffer are written on the gather's stream, none on the legacy stream (qualification_6607f886)" {
    var api: cuda.abi.Api = undefined;
    api.cuMemcpyHtoDAsync_v2 = Recorder.copyAsync;
    api.cuMemsetD32Async = Recorder.fillAsync;
    api.cuMemcpyHtoD_v2 = Recorder.copy;
    api.cuMemsetD32_v2 = Recorder.fill;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    Recorder.stream = @ptrFromInt(0x7a2);
    var want: [64]u32 = @splat(7);
    var r: Run = .{ .gpa = std.testing.allocator, .io = std.testing.io, .rank = 1, .f = .{}, .d = &d, .want = &want };
    r.stream = .{ .d = &d, .handle = Recorder.stream };
    r.send = .{ .d = &d, .ptr = 0x10000, .len = 128 };
    r.recv = .{ .d = &d, .ptr = 0x20000, .len = 256 };
    try r.stage(16); // arm()'s trial: the block, then the marker fill
    try std.testing.expectEqual(@as(usize, 2), Recorder.on_stream);
    try std.testing.expectEqual(@as(cuda.abi.DevicePtr, 0x10000), Recorder.last_dst);
    try r.put(true, 16); // sequence()'s in-place form: rank 1's block of recv
    try std.testing.expectEqual(@as(cuda.abi.DevicePtr, 0x20000 + 16 * 4), Recorder.last_dst);
    for (0..3) |_| try r.put(false, 8);
    try std.testing.expectEqual(@as(usize, 6), Recorder.on_stream);
    try std.testing.expectEqual(@as(usize, 0), Recorder.elsewhere);
    try std.testing.expectEqual(@as(usize, 0), Recorder.legacy);
}

test "the tool's GPU and link paths compile on the host" {
    _ = &runGpu;
    _ = &railsCmd;
    _ = &printCases;
    _ = &Run.main;
    _ = &link_mode.run;
    _ = &roce_cpu.run;
}
