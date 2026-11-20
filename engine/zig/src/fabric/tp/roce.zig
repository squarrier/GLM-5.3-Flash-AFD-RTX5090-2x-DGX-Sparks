// Modified for V4 by Scott Quarrier with AI assistance: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! The RoCE backend between two hosts: the tp_roce kernel stages and rings a doorbell, a proxy thread moves the slot.
const std = @import("std");
const log = @import("log.zig");
const cuda = @import("cuda");
const collective = @import("collective.zig");
const Bootstrap = @import("bootstrap.zig").Bootstrap;
const sock = @import("sock.zig");
const rv = @import("roce_verbs.zig");
const shm = @import("roce_shm.zig");
const guard = @import("roce_guard.zig");
const HostRegister = @import("mailbox.zig").HostRegister;
const posix = std.posix;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Error = collective.Error;

pub const max_blocks = 16;
const threads = 512;
const page = std.heap.page_size_min;
const flag_stride = 128;
// region: ctrl [4 KiB], flags [2 slots][2 HCAs] at 128 B, send [2][slot], recv [2][slot] (mailbox.cu's tp_roce)
const ctrl_off = 0;
const flag_off = 4096;
const send_off = 8192;
// ctrl words (mailbox.cu): doorbell, padded bytes per slot, completed, abort, proxy failure, a failed wait's record
const w_seq = 0;
const w_slot_bytes = 4;
const w_completed = 8;
const w_abort = 12;
const w_proxy_failed = 13;
const w_failed = 16;
/// Model graphs leave sub-millisecond gaps: the proxy stays hot this many idle polls, then naps 20 us at a time.
const idle_spins: u64 = 20_000_000;

const Mode = enum(u32) { gather = 0, sum_f32 = 1, sum_bf16 = 2, exchange = 3 };

/// mailbox.cu's `RoceArgs`, field for field.
const RoceArgs = extern struct {
    in: DevicePtr,
    out: DevicePtr,
    send_slots: DevicePtr,
    recv_slots: DevicePtr,
    flags: DevicePtr,
    ctrl: DevicePtr,
    state: DevicePtr,
    nbytes: u64,
    slot_bytes: u64,
    timeout_ns: u64,
    rank: u32,
    mode: u32,
    n_hca: u32,
    /// TF_TP_ROCE_FAST (config.roce_fast; 0: the classic path)
    opts: u32 = 0,
    /// gather only: device int32 [2] of each rank's bytes (0: every shard is `nbytes`)
    lens: DevicePtr = 0,
    /// Chunked gathers: this op's byte offset into each rank's shard (it moves at most a slot from there; 0 otherwise).
    chunk: u64 = 0,
};

pub const WireKind = enum {
    /// RDMA writes over the HCAs.
    verbs,
    /// The proxy copies into the peer's region in shared memory: the same protocol on one host without RDMA (tests).
    shm,
};

pub const Options = struct {
    wire: WireKind = .verbs,
    hcas: []const rv.HcaSpec = &.{},
    traffic_class: u8 = 0,
    cpu: ?u32 = null,
    max_bytes: usize = 256 * 1024,
    timeout_ns: u64 = 120 * std.time.ns_per_s,
    /// TF_TP_ROCE_FAST: tp_roce's `opts` word (config.roce_fast)
    fast: u32 = 0,
    /// TF_TP_ROCE_BLOCKS: the largest grid (config.roce_blocks)
    blocks: u32 = 16,
    /// The shm wire's stripes, as HCAs: its posts take the verbs wire's split and bounds.
    shm_stripes: u32 = 1,
    /// Tests: sees every shm post's stripes before the copy.
    tap: ?*shm.PostTap = null,
};

const Wire = union(WireKind) {
    verbs: rv.VerbsWire,
    shm: shm.ShmWire,

    fn post(w: *Wire, slot: u32, seq: u32, padded: u32) !void {
        switch (w.*) {
            .verbs => |*v| try v.post(slot, seq, padded),
            .shm => |*s| try s.post(slot, seq, padded),
        }
    }

    fn reap(w: *Wire) !void {
        switch (w.*) {
            .verbs => |*v| try v.reap(),
            .shm => {},
        }
    }

    fn hcas(w: *const Wire) u32 {
        return switch (w.*) {
            .verbs => |v| v.n_hca,
            .shm => |s| s.stripes,
        };
    }
};

pub const Roce = struct {
    me: u32,
    slot_bytes: usize,
    timeout_ns: u64,
    /// TF_TP_ROCE_FAST: the kernel's `opts` word
    fast: u32 = 0,
    /// TF_TP_ROCE_BLOCKS: the largest grid
    max_grid: u32 = 16,
    /// This rank's region (for shm: half of the mapping in `mapping`).
    region: []align(page) u8,
    mapping: []align(page) u8,
    wire: Wire,
    cpu: ?u32,
    // device side (absent in host-only use: tests and `tf-tp-test roce-host`)
    d: ?*const cuda.Driver = null,
    hr: ?HostRegister = null,
    host_unregistered: bool = false, // successful unregister may outlive the close deadline; DLL remains owned
    dev: DevicePtr = 0,
    state: ?cuda.DeviceBuffer = null,
    module: ?cuda.Module = null,
    func: ?cuda.Function = null,
    /// The kernel image takes `RoceArgs.chunk` (it exports tp_roce_chunk_abi): TF_TP_ROCE_CHUNKED_KB needs it.
    chunk_abi: bool = false,
    // the proxy thread
    thread: ?std.Thread = null,
    running: std.atomic.Value(bool) = .init(false),
    proxy_exited: std.atomic.Value(bool) = .init(true),
    close_retained: bool = false,
    last_seq: u32 = 0,
    err_buf: [160]u8 = undefined,
    err_len: usize = 0,
    // host emulation of the kernel's side (tests)
    host_epoch: u32 = 0,
    aborted: std.atomic.Value(bool) = .init(false),

    pub fn regionBytes(slot: usize) usize {
        return send_off + 4 * slot;
    }

    /// Both ranks call it after the bootstrap; any rank failing makes all return `error.RoceUnavailable` (collective).
    pub fn init(d: ?*const cuda.Driver, b: *Bootstrap, image: []const u8, o: Options) !*Roce {
        return initWithin(d,b,image,o,null);
    }
    /// On a partial device/proxy startup failure, hand the exact retained owner to the hybrid parent.
    pub fn initOwned(d: ?*const cuda.Driver,b: *Bootstrap,image: []const u8,o: Options,custody: *?*Roce) !*Roce {
        if(custody.*!=null) return error.Invalid;
        return initWithin(d,b,image,o,custody);
    }
    fn initWithin(d: ?*const cuda.Driver,b: *Bootstrap,image: []const u8,o: Options,custody: ?*?*Roce) !*Roce {
        if (b.world != 2) return error.Unsupported;
        if (o.max_bytes > guard.max_write) return error.WriteTooLarge; // a slot no post could fill in one write
        const slot = std.mem.alignForward(usize, o.max_bytes, 4096);
        const len = regionBytes(slot);
        const r = try std.heap.page_allocator.create(Roce);
        var retained=false;
        errdefer if(!retained) std.heap.page_allocator.destroy(r);
        r.* = .{ .me = b.rank, .slot_bytes = slot, .timeout_ns = o.timeout_ns, .fast = o.fast, .max_grid = o.blocks, .region = undefined, .mapping = undefined, .wire = undefined, .cpu = o.cpu };
        // 1. the regions and the wire, each rank alone; every rank learns whether all did
        const ok = r.openWire(b, o, len) catch |e| blk: {
            log.warn("[tensorfold] tp roce: rank {d} setup failed: {t}", .{ b.rank, e });
            break :blk false;
        };
        errdefer if(ok and !retained) r.closeWire();
        errdefer if(custody) |out| {
            if(r.d!=null or r.thread!=null) {
                retained=true;out.*=r;r.abort();_=r.retainClose(.failed);
            }
        };
        errdefer if(custody==null) if(r.thread) |thread| {
            r.running.store(false,.release);r.abort();thread.join();r.thread=null;
        };
        if (!try agreeAll(b, ok, "RoCE setup")) return error.RoceUnavailable;
        // 2. the verbs connection records, through the bootstrap
        const connected = r.connectWire(b) catch |e| blk: {
            log.warn("[tensorfold] tp roce: rank {d} connect failed: {t}", .{ b.rank, e });
            break :blk false;
        };
        if (!try agreeAll(b, connected, "RoCE connect")) return error.RoceUnavailable;
        // 3. the device side
        if (d) |drv| {
            const dev_ok = r.openDevice(drv, image) catch |e| blk: {
                log.warn("[tensorfold] tp roce: rank {d} device setup failed: {t}", .{ b.rank, e });
                break :blk false;
            };
            if (!try agreeAll(b, dev_ok, "RoCE device setup")) {
                if (dev_ok) r.closeDevice();
                return error.RoceUnavailable;
            }
        }
        r.running.store(true, .release);
        r.proxy_exited.store(false,.release);
        r.thread = try std.Thread.spawn(.{}, proxyLoop, .{r});
        try b.agree("proxy", "RoCE proxy start");
        return r;
    }

    fn agreeAll(b: *Bootstrap, ok: bool, _: []const u8) !bool {
        var blobs: [2][]const u8 = undefined;
        var buf: [2]u8 = undefined;
        const all = try b.allGather(&.{@intFromBool(ok)}, &buf, &blobs);
        return all[0][0] == 1 and all[1][0] == 1;
    }

    fn openWire(r: *Roce, b: *Bootstrap, o: Options, len: usize) !bool {
        switch (o.wire) {
            .verbs => {
                r.mapping = try posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
                r.region = r.mapping;
                errdefer posix.munmap(r.mapping);
                try rv.prepareRegion(r.mapping);
                r.wire = .{ .verbs = try rv.VerbsWire.open(o.hcas, r.region, .{ .send = send_off, .recv = send_off + 2 * r.slot_bytes, .flags = flag_off, .slot = r.slot_bytes }, o.traffic_class) };
            },
            .shm => {
                r.mapping = try shm.sharedPair(b, len);
                r.region = @alignCast(r.mapping[r.me * len ..][0..len]);
                const layout: guard.Layout = .{ .send = send_off, .recv = send_off + 2 * r.slot_bytes, .flags = flag_off, .slot = r.slot_bytes, .len = len };
                r.wire = .{ .shm = .{ .peer = r.mapping[(1 - r.me) * len ..][0..len], .layout = layout, .local = r.region, .stripes = o.shm_stripes, .tap = o.tap } };
            },
        }
        return true;
    }

    fn connectWire(r: *Roce, b: *Bootstrap) !bool {
        switch (r.wire) {
            .shm => return true,
            .verbs => |*v| {
                const mine = v.record();
                var blobs: [2][]const u8 = undefined;
                var buf: [2 * @sizeOf(rv.Record)]u8 = undefined;
                const all = try b.allGather(std.mem.asBytes(&mine), &buf, &blobs);
                var peer: rv.Record = undefined;
                if (all[1 - r.me].len != @sizeOf(rv.Record)) return error.BadRecord;
                @memcpy(std.mem.asBytes(&peer), all[1 - r.me]);
                try v.connect(peer);
                var line: [512]u8 = undefined;
                log.info("[tensorfold] tp roce: rank {d} registered {s}", .{ r.me, v.describe(&line) });
                return true;
            },
        }
    }

    fn openDevice(r: *Roce, d: *const cuda.Driver, image: []const u8) !bool {
        if (image.len == 0) return error.NoKernel;
        r.d = d;
        r.hr = try HostRegister.open();
        try d.check(r.hr.?.register(r.region.ptr, r.region.len, 1 | 2), "cuMemHostRegister"); // PORTABLE | DEVICEMAP
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&r.dev, r.region.ptr, 0), "cuMemHostGetDevicePointer");
        r.state = try cuda.DeviceBuffer.alloc(d, 16);
        try r.state.?.fill8(0, null);
        r.module = try cuda.Module.load(d, image);
        r.func = try r.module.?.function("tp_roce");
        r.chunk_abi = if (r.module.?.function("tp_roce_chunk_abi")) |_| true else |_| false;
        return true;
    }

    fn closeDevice(r: *Roce) void {
        if (r.module) |*m| m.unload();
        if (r.state) |*s| s.free();
        if (r.hr) |*h| {
            if(!r.host_unregistered)_ = h.unregister(r.region.ptr);
            h.lib.close();
        }
        r.module = null;
        r.state = null;
        r.hr = null;
    }

    fn closeWire(r: *Roce) void {
        switch (r.wire) {
            .verbs => |*v| v.close(),
            .shm => {},
        }
        posix.munmap(r.mapping);
    }

    /// Regions a failed transport left registered and mapped (the process exit is their only teardown).
    pub var kept: std.atomic.Value(u32) = .init(0);

    pub const CloseReceipt=@import("roce_close.zig").Receipt;
    fn retainClose(r: *Roce,reason: @import("roce_close.zig").Reason) CloseReceipt {
        if(!r.close_retained) {_=kept.fetchAdd(1,.monotonic);r.close_retained=true;}
        return .{.quiet=false,.retained_bytes=r.region.len,.reason=reason};
    }
    /// Hybrid caller's actual owned stream events, queried under one enclosing deadline. No context synchronize.
    pub fn closeChecked(r: *Roce,until: u64,events: []const cuda.Event) CloseReceipt {
        if(r.close_retained) return r.retainClose(.failed);
        const was_failed=r.failed();
        r.word(w_abort).store(1,.release);r.running.store(false,.release);
        while(r.thread!=null and !r.proxy_exited.load(.acquire)) {
            const remaining_now=sock.nowNs();
            if(remaining_now>=until) return r.retainClose(.proxy);
            sock.sleepNs(@min(20_000,until-remaining_now));
        }
        if(r.thread) |thread| {thread.join();r.thread=null;}
        if(was_failed) return r.retainClose(.failed);
        if(r.d!=null and events.len==0) return r.retainClose(.missing_completion);
        for(events) |ev| while(true) {
            if(sock.nowNs()>=until) return r.retainClose(.deadline);
            const result=ev.d.api.cuEventQuery(ev.handle);
            if(result==cuda.abi.success) break;
            if(result!=cuda.abi.error_not_ready) return r.retainClose(.gpu);
            std.atomic.spinLoopHint();
        };
        const now=sock.nowNs();if(now>=until) return r.retainClose(.deadline);
        const quiet=switch(r.wire) {.verbs=>|*v| v.quiesce(until-now),.shm=>true};
        if(!quiet or sock.nowNs()>=until) return r.retainClose(.wire);
        // A registered region is still owned until the real unregister succeeds.
        // Do this before closeDevice can clear the handle or closeWire can unmap.
        if(r.hr)|*h|{
            if(h.unregister(r.region.ptr)!=cuda.abi.success)return r.retainClose(.gpu);
            r.host_unregistered=true;
            if(sock.nowNs()>=until)return r.retainClose(.deadline);
            h.lib.close();r.hr=null;
        }
        r.closeDevice();r.closeWire();std.heap.page_allocator.destroy(r);
        return .{.quiet=true,.retained_bytes=0,.reason=.closed};
    }

    /// HCAs and GPU quiet, then unregister and unmap; a failed transport's region is kept until exit.
    pub fn deinit(r: *Roce) void {
        const was_failed = r.failed();
        r.word(w_abort).store(1, .release);
        r.running.store(false, .release);
        if (r.thread) |t| t.join();
        const quiet = r.quiesce();
        if (was_failed or !quiet) {
            const why = if (was_failed) "failed" else "did not quiesce";
            log.warn("[tensorfold] tp roce: rank {d}: the transport {s}; region 0x{x} ({d} B) kept registered until exit", .{ r.me, why, @intFromPtr(r.region.ptr), r.region.len });
            _ = kept.fetchAdd(1, .monotonic);
            return;
        }
        r.closeDevice();
        r.closeWire();
        std.heap.page_allocator.destroy(r);
    }

    /// The QPs to error with every posted request reaped, and the GPU's work on the region done.
    fn quiesce(r: *Roce) bool {
        const wire = switch (r.wire) {
            .verbs => |*v| v.quiesce(rv.quiesce_ns),
            .shm => true,
        };
        const gpu = if (r.d) |d| d.api.cuCtxSynchronize() == cuda.abi.success else true;
        var line: [384]u8 = undefined;
        log.info("[tensorfold] tp roce: rank {d} quiesced ({s}): {s}; GPU sync {s}", .{ r.me, if (wire) "drained" else "NOT drained", r.describeLink(&line), if (gpu) "done" else "FAILED" });
        return wire and gpu;
    }

    /// Each HCA's QP state, posts and completions, and what a drain reaped (tf-tp-test's idle check; shm: none).
    pub fn describeLink(r: *const Roce, buf: []u8) []const u8 {
        return switch (r.wire) {
            .verbs => |*v| v.report(buf),
            .shm => "shm wire",
        };
    }

    fn word(r: *const Roce, i: usize) *std.atomic.Value(u32) {
        return @ptrCast(@alignCast(r.region[ctrl_off + 4 * i ..].ptr));
    }

    fn fail(r: *Roce, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(&r.err_buf, fmt, args) catch "proxy failure";
        r.err_len = s.len;
        log.err("[tensorfold] tp roce proxy: {s}", .{s});
        r.word(w_proxy_failed).store(1, .release);
        r.word(w_abort).store(1, .release); // waits on this rank end now instead of at the timeout
        r.halt();
    }

    /// The HCAs stop: nothing of this rank's goes out and none of the peer's writes lands while the ranks go down.
    fn halt(r: *Roce) void {
        switch (r.wire) {
            .verbs => |*v| {
                const ok = v.halt();
                var line: [384]u8 = undefined;
                log.warn("[tensorfold] tp roce: rank {d} halted{s}: {s}", .{ r.me, if (ok) "" else " (a QP refused ERR)", v.report(&line) });
            },
            .shm => {},
        }
    }

    /// Every post of this rank's completed (shm: nothing is ever in flight).
    pub fn linkIdle(r: *const Roce) bool {
        return switch (r.wire) {
            .verbs => |*v| v.idle(),
            .shm => true,
        };
    }

    /// tf-tp-test's `failfast RANK transport`: the proxy's failure path (`fail`), here with no op in flight.
    pub fn inject(r: *Roce) void {
        r.fail("failure injected by tf-tp-test (no RDMA fault)", .{});
    }

    /// The transport failed or was aborted: a device may still write to its region.
    pub fn failed(r: *const Roce) bool {
        if (r.aborted.load(.acquire) or r.word(w_abort).load(.acquire) != 0) return true;
        return r.word(w_proxy_failed).load(.acquire) != 0 or r.word(w_failed).load(.acquire) != 0;
    }

    /// The proxy loop: spin on the doorbell, post each new sequence (at most two pending), reap while idle, then nap.
    fn proxyLoop(r: *Roce) void {
        defer r.proxy_exited.store(true,.release);
        if (r.cpu) |cpu| pin(cpu);
        var idle: u64 = 0;
        while (r.running.load(.monotonic)) {
            if (r.word(w_abort).load(.monotonic) != 0) return;
            const seq = r.word(w_seq).load(.acquire);
            if (seq == r.last_seq) {
                idle += 1;
                if (idle % 64 == 0) r.wire.reap() catch |e| return r.fail("completion: {t}", .{e});
                if (idle >= idle_spins) sock.sleepNs(20 * std.time.ns_per_us) else std.atomic.spinLoopHint();
                continue;
            }
            idle = 0;
            var pending = seq -% r.last_seq;
            if (pending > 2) return r.fail("doorbell skipped {d} ops (posted {d}, now {d})", .{ pending, r.last_seq, seq });
            var s = r.last_seq +% 1;
            while (pending > 0) : ({
                s +%= 1;
                pending -= 1;
            }) {
                const padded = r.word(w_slot_bytes + (s & 1)).load(.acquire);
                if (padded == 0 or padded % 16 != 0 or padded > r.slot_bytes) return r.fail("op {d} of {d} bytes", .{ s, padded });
                r.wire.post(s & 1, s, padded) catch |e| return r.fail("post of op {d}: {t}", .{ s, e });
                r.last_seq = s;
            }
            r.wire.reap() catch |e| return r.fail("completion: {t}", .{e});
        }
    }

    pub fn fits(r: *const Roce, n: usize) bool {
        return n <= r.slot_bytes;
    }

    fn launch(r: *Roce, mode: Mode, in: DevicePtr, out: DevicePtr, n: usize, lens: DevicePtr, stream: collective.Stream) Error!void {
        return r.launchAt(mode, in, out, n, lens, 0, stream);
    }

    fn launchAt(r: *Roce, mode: Mode, in: DevicePtr, out: DevicePtr, n: usize, lens: DevicePtr, chunk: u64, stream: collective.Stream) Error!void {
        if (r.aborted.load(.acquire)) return error.Aborted;
        if (n > r.slot_bytes and (lens == 0 or !r.chunk_abi)) return error.Unsupported;
        if (n == 0) return;
        const d = r.d orelse return error.Unsupported;
        var a: RoceArgs = .{
            .in = in,
            .out = out,
            .send_slots = r.dev + send_off,
            .recv_slots = r.dev + send_off + 2 * r.slot_bytes,
            .flags = r.dev + flag_off,
            .ctrl = r.dev + ctrl_off,
            .state = r.state.?.ptr,
            .nbytes = n,
            .slot_bytes = r.slot_bytes,
            .timeout_ns = r.timeout_ns,
            .rank = r.me,
            .mode = @backingInt(mode),
            .n_hca = r.wire.hcas(),
            .opts = r.fast,
            .lens = lens,
            .chunk = chunk,
        };
        var params = [1]?*anyopaque{@ptrCast(&a)};
        const blocks = @min(@import("mailbox.zig").blocksFor(@min(n, r.slot_bytes)), r.max_grid);
        d.check(d.api.cuLaunchKernel(r.func.?.handle, blocks, 1, 1, threads, 1, 1, 0, stream, &params, null), "cuLaunchKernel tp_roce") catch return error.BackendFailed;
    }

    pub fn allGatherBytes(r: *Roce, s: DevicePtr, o: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return r.launch(.gather, s, o, n, 0, stream);
    }

    /// `allGatherBytes` where rank r posts only lens[r] bytes (device int32 [2], padded to 16, at least 16).
    pub fn allGatherV(r: *Roce, s: DevicePtr, o: DevicePtr, n: usize, lens: DevicePtr, stream: collective.Stream) Error!void {
        if (n > r.slot_bytes) return error.Unsupported; // one op: past a slot is allGatherVChunked's
        if (lens == 0) return error.Invalid;
        return r.launch(.gather, s, o, n, lens, stream);
    }

    /// `allGatherV` past the slot: op c moves at most a slot of each shard from c x slot (empty on both ranks: no op).
    pub fn allGatherVChunked(r: *Roce, s: DevicePtr, o: DevicePtr, n: usize, lens: DevicePtr, stream: collective.Stream) Error!void {
        if (lens == 0) return error.Invalid;
        if (!r.chunk_abi) return error.Unsupported;
        var off: usize = 0;
        while (off < n) : (off += r.slot_bytes) try r.launchAt(.gather, s, o, n, lens, off, stream);
    }

    /// One serialized hybrid frame. The kernel copies this remainder into its own launch arguments.
    pub fn exchangeBefore(r: *Roce,s: DevicePtr,o: DevicePtr,n: usize,stream: collective.Stream,until: u64) Error!void {
        const now=sock.nowNs();if(now>=until or n>guard.max_write) return error.Timeout;
        const original=r.timeout_ns;defer r.timeout_ns=original;
        r.timeout_ns=@min(original,until-now);if(r.timeout_ns==0) return error.Timeout;
        try r.exchangeBytes(s,o,n,stream);
    }
    pub fn exchangeBytes(r: *Roce, s: DevicePtr, o: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return r.launch(.exchange, s, o, n, 0, stream);
    }

    pub fn sum(r: *Roce, s: DevicePtr, o: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        const mode: Mode = switch (t) {
            .f32 => .sum_f32,
            .bf16 => .sum_bf16,
            else => return error.Unsupported,
        };
        return r.launch(mode, s, o, count * t.size(), 0, stream);
    }

    /// Ends this rank's waits and the proxy (the peer learns through the fate channel and aborts its own).
    pub fn abort(r: *Roce) void {
        r.aborted.store(true, .release);
        r.word(w_abort).store(1, .release);
        r.halt();
    }

    pub fn check(r: *Roce) Error!void {
        if (r.word(w_failed).load(.acquire) != 0) {
            log.warn("tp roce: rank {d}'s wait for seq {d} failed ({s} after {d} us; HCA {d}, flag held {d}); doorbell {d}, posted {d}, completed {d}, abort {d}, proxy failed {d}, flags {d} {d}", .{ r.me, r.word(w_failed + 1).load(.acquire), if (r.word(w_failed + 4).load(.acquire) == 1) "aborted" else "timed out", r.word(w_failed + 5).load(.acquire), r.word(w_failed + 3).load(.acquire), r.word(w_failed + 2).load(.acquire), r.word(w_seq).load(.acquire), r.last_seq, r.word(w_completed).load(.acquire), r.word(w_abort).load(.acquire), r.word(w_proxy_failed).load(.acquire), r.flagWord(0), r.flagWord(1) });
            return error.PeerFailed;
        }
        if (r.word(w_proxy_failed).load(.acquire) != 0) return error.BackendFailed;
        if (r.aborted.load(.acquire) or r.word(w_abort).load(.acquire) != 0) return error.Aborted;
    }

    fn flagWord(r: *const Roce, slot: usize) u32 {
        const p: *std.atomic.Value(u32) = @ptrCast(@alignCast(r.region[flag_off + slot * 2 * flag_stride ..].ptr));
        return p.load(.acquire);
    }

    pub fn proxyError(r: *const Roce) []const u8 {
        return r.err_buf[0..r.err_len];
    }

    /// The kernel's side of one op done by the host (tests, `roce-host`): a gather through the same slots and flags.
    pub fn hostGather(r: *Roce, in: []const u8, out: []u8) Error!void {
        return r.hostGatherV(in, out, null);
    }

    /// `hostGather` with variable lengths (`allGatherV`): rank k's first lens[k] bytes land at out + k x n.
    pub fn hostGatherV(r: *Roce, in: []const u8, out: []u8, lens: ?[2]u32) Error!void {
        if (in.len > r.slot_bytes) return error.Invalid;
        return r.hostGatherAt(in, out, lens, 0);
    }

    /// `allGatherVChunked`'s ops done by the host (the kernel's side of each, as `hostGatherV`).
    pub fn hostGatherVChunked(r: *Roce, in: []const u8, out: []u8, lens: [2]u32) Error!void {
        var off: usize = 0;
        while (off < in.len) : (off += r.slot_bytes) try r.hostGatherAt(in, out, lens, off);
    }

    /// One op from shard offset `off`: each rank's bytes past it, at most a slot; an op empty on both ranks is skipped.
    fn hostGatherAt(r: *Roce, in: []const u8, out: []u8, lens: ?[2]u32, off: usize) Error!void {
        const n = in.len;
        if (n == 0 or out.len < 2 * n) return error.Invalid;
        if (lens) |l| if (l[0] > n or l[1] > n) return error.Invalid;
        const full: [2]usize = if (lens) |l| .{ l[0], l[1] } else .{ n, n };
        const mine: usize = @min(full[r.me] -| off, r.slot_bytes);
        const theirs: usize = @min(full[1 - r.me] -| off, r.slot_bytes);
        if (off > 0 and mine == 0 and theirs == 0) return;
        const seq = r.host_epoch +% 1;
        const slot: usize = seq & 1;
        @memcpy(r.region[send_off + slot * r.slot_bytes ..][0..mine], in[off..][0..mine]);
        r.word(w_slot_bytes + slot).store(@intCast(@max(16, std.mem.alignForward(usize, mine, 16))), .release);
        r.word(w_seq).store(seq, .release);
        @memcpy(out[r.me * n + off ..][0..mine], in[off..][0..mine]);
        const t0 = sock.nowNs();
        for (0..r.wire.hcas()) |h| {
            const flag: *std.atomic.Value(u32) = @ptrCast(@alignCast(r.region[flag_off + (slot * 2 + h) * flag_stride ..].ptr));
            while (flag.load(.acquire) != seq) {
                if (r.word(w_abort).load(.monotonic) != 0) return error.Aborted;
                if (sock.nowNs() - t0 > r.timeout_ns) return error.Timeout;
                std.atomic.spinLoopHint();
            }
        }
        @memcpy(out[(1 - r.me) * n + off ..][0..theirs], r.region[send_off + 2 * r.slot_bytes + slot * r.slot_bytes ..][0..theirs]);
        r.host_epoch = seq;
    }
};

test "hybrid close: native typed close retains backing after a late successful event query" {
    const F=struct {fn query(_: cuda.abi.Event) callconv(.c) c_int {sock.sleepNs(5*std.time.ns_per_ms);return 0;}};
    var d: cuda.Driver=undefined;d.api.cuEventQuery=F.query;
    const n=Roce.regionBytes(4096);const mem=try posix.mmap(null,n,.{.READ=true,.WRITE=true},.{.TYPE=.PRIVATE,.ANONYMOUS=true},-1,0);@memset(mem,0);
    const r=try std.heap.page_allocator.create(Roce);r.*=.{.me=0,.slot_bytes=4096,.timeout_ns=100,.region=mem,.mapping=mem,.cpu=null,.wire=.{.shm=.{.peer=mem,.local=mem,.layout=.{.send=send_off,.recv=send_off+8192,.flags=flag_off,.slot=4096,.len=n}}},.d=&d};
    const ev=cuda.Event{.d=&d,.handle=null};const receipt=r.closeChecked(sock.nowNs()+std.time.ns_per_ms,&.{ev});
    try std.testing.expect(!receipt.quiet);
    // Host-only fixture cleanup after proving retained ownership; no CUDA backing exists in this test.
    posix.munmap(mem);std.heap.page_allocator.destroy(r);
}

test "hybrid close: actual constructor retains a proxy owner after final bootstrap mismatch" {
    const t=std.testing;
    const Other=struct {
        fn run(fd: std.c.fd_t) void {
            var b=Bootstrap{.rank=1,.world=2,.timeout_ns=std.time.ns_per_s};b.fds[0]=fd;defer b.close();
            const m=shm.sharedPair(&b,Roce.regionBytes(4096)) catch return;defer posix.munmap(m);
            var slices: [2][]const u8=undefined;var bytes: [2]u8=undefined;
            _=b.allGather(&.{1},&bytes,&slices) catch return;
            _=b.allGather(&.{1},&bytes,&slices) catch return;
            b.agree("wrong-proxy","injected last constructor agreement") catch {};
        }
    };
    var fds: [2]std.c.fd_t=undefined;try t.expectEqual(@as(c_int,0),std.c.socketpair(std.posix.AF.UNIX,std.posix.SOCK.STREAM,0,&fds));
    var b=Bootstrap{.rank=0,.world=2,.timeout_ns=std.time.ns_per_s};b.fds[1]=fds[0];defer b.close();
    const peer=try std.Thread.spawn(.{},Other.run,.{fds[1]});defer peer.join();
    const prior=Roce.kept.load(.acquire);var retained: ?*Roce=null;
    const result=if(comptime @hasDecl(Roce,"initOwned")) Roce.initOwned(null,&b,&.{},.{.wire=.shm,.max_bytes=4096},&retained) else Roce.init(null,&b,&.{},.{.wire=.shm,.max_bytes=4096});
    try t.expectError(error.Mismatch,result);
    try t.expectEqual(prior+1,Roce.kept.load(.acquire));try t.expect(retained!=null);
    // Synthetic host-only cleanup after proving the product retained its exact owner.
    const r=retained.?;r.running.store(false,.release);r.abort();if(r.thread) |thread| thread.join();r.closeWire();std.heap.page_allocator.destroy(r);
}

test "hybrid close: native exchange launch clamps copied timeout to absolute remainder" {
    const F=struct {
        var timeout: u64=0;
        fn launch(_: cuda.abi.Function,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: cuda.abi.Stream,args: ?[*]?*anyopaque,_: ?[*]?*anyopaque) callconv(.c) c_int {
            const a: *const RoceArgs=@ptrCast(@alignCast(args.?[0].?));timeout=a.timeout_ns;return 0;
        }
    };
    var d: cuda.Driver=undefined;d.api.cuLaunchKernel=F.launch;
    var r: Roce=undefined;r.aborted=.init(false);r.slot_bytes=1048576;r.chunk_abi=false;r.d=&d;r.dev=0;r.timeout_ns=100_000_000;r.me=0;r.fast=0;r.max_grid=1;r.wire=.{.shm=.{.stripes=1,.local=undefined,.peer=undefined,.layout=undefined}};
    r.state=.{.d=&d,.ptr=0x1000,.len=16};r.func=.{.d=&d,.handle=@ptrFromInt(7)};
    if(comptime @hasDecl(Roce,"exchangeBefore")) try r.exchangeBefore(1,2,1048576,null,sock.nowNs()+1_000_000) else try r.exchangeBytes(1,2,1048576,null);
    try std.testing.expect(F.timeout>0 and F.timeout<=1_000_000);
    try std.testing.expectEqual(@as(u64,100_000_000),r.timeout_ns);
}

test "hybrid close: failed native RoCE retains backing without unbounded context synchronize" {
    const F=struct {var syncs: usize=0;fn sync() callconv(.c) c_int {syncs+=1;return 0;}};
    var d: cuda.Driver=undefined;d.api.cuCtxSynchronize=F.sync;
    const mem=try std.heap.page_allocator.alignedAlloc(u8,.fromByteUnits(page),Roce.regionBytes(4096));defer std.heap.page_allocator.free(mem);@memset(mem,0);
    const r=try std.heap.page_allocator.create(Roce);defer std.heap.page_allocator.destroy(r);
    r.*=.{.me=0,.slot_bytes=4096,.timeout_ns=100,.region=mem,.mapping=mem,.cpu=null,.wire=.{.shm=.{.peer=mem,.local=mem,.layout=.{.send=send_off,.recv=send_off+8192,.flags=flag_off,.slot=4096,.len=mem.len},.stripes=1,.tap=null}},.d=&d};
    r.word(w_failed).store(1,.release);F.syncs=0;
    if(comptime @hasDecl(Roce,"closeChecked")) {
        const receipt=r.closeChecked(sock.nowNs()+1_000_000,&.{});
        try std.testing.expect(!receipt.quiet);try std.testing.expectEqual(mem.len,receipt.retained_bytes);
    } else r.deinit();
    try std.testing.expectEqual(@as(usize,0),F.syncs);
}

fn pin(cpu: u32) void {
    if (@import("builtin").os.tag != .linux) return;
    var set: std.os.linux.cpu_set_t = @splat(0);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    std.os.linux.sched_setaffinity(0, &set) catch log.warn("tp roce: could not pin the proxy to CPU {d}", .{cpu});
}

test "hybrid close: I3 failed native host unregister preserves exact region and registration owner" {try unregisterTrial(false);}
test "hybrid close: I3 late successful unregister cannot publish quiet beyond deadline" {try unregisterTrial(true);}
fn unregisterTrial(late:bool)!void{
    const t=std.testing;const F=struct{var calls:usize=0;var slow=false;fn query(_:cuda.abi.Event)callconv(.c)c_int{return 0;}fn reg(_:?*anyopaque,_:usize,_:c_uint)callconv(.c)c_int{return 0;}fn unreg(_:?*anyopaque)callconv(.c)c_int{calls+=1;if(slow){sock.sleepNs(5*std.time.ns_per_ms);return 0;}return cuda.abi.error_not_ready;}};F.slow=late;
    var d:cuda.Driver=undefined;d.api.cuEventQuery=F.query;const n=Roce.regionBytes(4096);const mem=try posix.mmap(null,n,.{.READ=true,.WRITE=true},.{.TYPE=.PRIVATE,.ANONYMOUS=true},-1,0);@memset(mem,0);
    const r=try std.heap.page_allocator.create(Roce);r.*=.{.me=0,.slot_bytes=4096,.timeout_ns=100,.region=mem,.mapping=mem,.cpu=null,.wire=.{.shm=.{.peer=mem,.local=mem,.layout=.{.send=send_off,.recv=send_off+8192,.flags=flag_off,.slot=4096,.len=n}}},.d=&d,.hr=.{.lib=try std.DynLib.open("libc.so.6"),.register=F.reg,.unregister=F.unreg}};
    F.calls=0;const receipt=r.closeChecked(sock.nowNs()+@as(u64,if(late)std.time.ns_per_ms else std.time.ns_per_s),&.{.{.d=&d,.handle=null}});
    if(receipt.quiet)return error.TestUnexpectedCleanRelease;
    defer {r.hr.?.lib.close();posix.munmap(mem);std.heap.page_allocator.destroy(r);}
    try t.expectEqual(@as(usize,1),F.calls);try t.expectEqual(n,receipt.retained_bytes);try t.expect(r.hr!=null and r.close_retained);try t.expectEqual(@as(u8,0),mem[n-1]);
    if(late){try t.expectEqual(@import("roce_close.zig").Reason.deadline,receipt.reason);try t.expect(@hasField(Roce,"host_unregistered"));if(comptime @hasField(Roce,"host_unregistered"))try t.expect(r.host_unregistered);}
}
