//! Helios authored this current-native adapter to the released V3 AFD exchange.
//! V3 source: squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks cd50fe4d, engine1fab9035.
//! Ash Hart authored TensorFold/MCDMA; Hugh Madden and T.J. Purtell authored the split;
//! Mia's AI Lab authored the imported native RoCE work. No original credit is removed.
const std = @import("std");
const cuda = @import("cuda");
const routed = @import("cuda_routed.zig");
const X = cuda.xchg;
const proto = X.proto;
/// Actual released V3 prompt kernel launcher; field mapping only, arithmetic unchanged.
pub const g53 = @import("cuda_g53.zig");
/// Borrowed, lifetime-owned resources. Runtime owner must settle before freeing any of them.
/// A request is device-packed, posted to both expert proxies, then taken after shared work.
const Entry = struct { call: routed.Call, destination: routed.HalfDest, sequence: u32 };
pub const Adapter = struct {
    node: *X.Node,
    gpu: *X.Gpu,
    request: cuda.DeviceBuffer,
    header: cuda.HostBuffer,
    done: cuda.Event,
    session: u128,
    capacity: usize,
    sequence: u32 = 0,
    header_at: usize = 0,
    completed: u32 = 0,
    inflight: [proto.LANES_MAX]?Entry = @splat(null),
    count: usize = 0,
    pending: bool = false,
    submitted: bool = false,
    failed: ?anyerror = null,
    last_stream: ?cuda.Stream = null,
    active: ?routed.Call = null,
    destination: ?routed.HalfDest = null,
    streams: [16]?cuda.Stream = @splat(null),
    proofs: [16]?cuda.Event = @splat(null),
    pub fn provider(a: *Adapter) routed.Provider {
        return .{ .ctx = a, .device_ordered = true, .run = run, .finish = finish, .check = check, .settle = settle, .touch = touch };
    }
    fn touch(ctx: *anyopaque, s: cuda.Stream) !void {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        try check(a);
        if (s.d != a.done.d) return error.BadHybridContext;
        for (a.streams) |known| if (known) |old| if (old.handle == s.handle) return;
        for (&a.streams, &a.proofs) |*slot, *proof| if (slot.* == null) {
            proof.* = try cuda.Event.init(s.d, false);
            slot.* = s;
            return;
        };
        return error.TooManyHybridStreams;
    }
    pub fn releaseProofs(a: *Adapter) void {
        std.debug.assert(!a.pending and a.failed == null);
        for (&a.proofs) |*ev| if (ev.*) |*e| {
            e.deinit();
            ev.* = null;
        };
    }
    fn check(ctx: *anyopaque) !void {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        if (a.failed) |err| return err;
        try a.node.check();
    }
    fn clock(a: *Adapter) u64 {
        return a.node.nowNs();
    }
    fn before(a: *Adapter, until: u64) !void {
        try check(a);
        if (a.clock() >= until) return error.Timeout;
    }
    fn run(ctx: *anyopaque, c: routed.Call, d: routed.HalfDest) !routed.ReadyParts {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        return a.post(c, d) catch |err| {
            a.failed = err;
            return err;
        };
    }
    fn post(a: *Adapter, c: routed.Call, d: routed.HalfDest) !routed.ReadyParts {
        try a.before(c.deadline);
        const l = a.node.cfg.settings.layout;
        if (a.count >= @min(proto.LANES_MAX, a.node.reg.slots) or a.session == 0 or a.node.role != .attn or c.rows == 0 or c.rows > a.capacity or c.rows != d.rows or c.hidden != 4096 or c.top_k != 8 or c.native_slot_stride != 9 or c.x_stride_elements < 4096 or l.d != 4096 or l.s != 9 or l.fp8 or l.bf16) return error.BadHybridCall;
        for (a.inflight) |entry| if (entry) |e| {
            const end = d.base + 2 * l.rep(c.rows, false);
            const prior_end = e.destination.base + 2 * l.rep(e.call.rows, false);
            if (d.base < prior_end and e.destination.base < end) return error.HybridFrameStillOwned;
        };
        const bytes = proto.wireBytes(l.req(c.rows));
        if (bytes > a.request.len or bytes > a.node.reg.slot or a.header.bytes.len < proto.HDR_BYTES or a.header.bytes.len % proto.HDR_BYTES != 0) return error.BadHybridStorage;
        if (a.last_stream) |previous| if (previous.handle != c.stream.handle) {
            if (a.count != 0) return error.HybridFrameStillOwned;
            try c.stream.wait(a.done);
        };
        if (a.header_at == a.header.bytes.len) {
            // A wrap may fall BETWEEN pending prompt-lane POSTs. The latest
            // TAKE event then predates their H2D header copies. Fence the actual
            // producer stream here; no TAKE/reply collection is needed for POST
            // completion, and all pending lanes remain owned and outstanding.
            if (a.count != 0) try a.done.record(c.stream);
            // Apply bounded backpressure ONLY at this page reuse boundary,
            // never at every MoE layer and never by enlarging the pinned allocation.
            while (true) {
                try a.before(c.deadline);
                const rc = a.done.d.api.cuEventQuery(a.done.handle);
                try a.before(c.deadline); // a late success cannot renew the call bound
                if (rc == cuda.abi.success) break;
                if (rc != cuda.abi.error_not_ready) return error.HybridCopyFailed;
                std.atomic.spinLoopHint();
            }
            a.header_at = 0;
        }
        const header = a.header.bytes[a.header_at..][0..proto.HDR_BYTES];
        a.header_at += proto.HDR_BYTES;
        a.sequence = try std.math.add(u32, a.sequence, 1);
        if (a.sequence > std.math.maxInt(i32)) return error.BadHybridSequence;
        const h = proto.Header{ .fseq = @intCast(a.sequence), .rows = @intCast(c.rows), .layer = @intCast(c.layer), .mode = if (c.prompt) .chunk else .eager, .layers = 1 };
        h.write(header[0..proto.HDR_BYTES]);
        a.pending = true;
        a.count += 1;
        a.inflight[(a.sequence - 1) % proto.LANES_MAX] = .{ .call = c, .destination = d, .sequence = a.sequence };
        const api = c.stream.d.api;
        const off = l.offsets(c.rows);
        try c.stream.d.check(api.cuMemsetD32Async(a.request.ptr, 0, bytes / 4, c.stream.handle), "V3 defined packet padding");
        try c.stream.d.check(api.cuMemcpyHtoDAsync_v2(a.request.ptr, header.ptr, proto.HDR_BYTES, c.stream.handle), "V3 owned pinned header");
        const x: cuda.abi.Memcpy2D = .{ .src = c.x_bf16, .src_pitch = c.x_stride_elements * 2, .dst = a.request.ptr + off.normed, .dst_pitch = 4096 * 2, .width = 4096 * 2, .height = c.rows };
        try c.stream.d.check(api.cuMemcpy2DAsync_v2(&x, c.stream.handle), "V3 actual-row device input");
        const picks: cuda.abi.Memcpy2D = .{ .src = c.picks_i32, .src_pitch = 9 * 4, .dst = a.request.ptr + off.pick, .dst_pitch = 9 * 4, .width = 9 * 4, .height = c.rows };
        try c.stream.d.check(api.cuMemcpy2DAsync_v2(&picks, c.stream.handle), "V3 ordered native picks");
        const weights: cuda.abi.Memcpy2D = .{ .src = c.weights_f32, .src_pitch = 9 * 4, .dst = a.request.ptr + off.wts, .dst_pitch = 9 * 4, .width = 9 * 4, .height = c.rows };
        try c.stream.d.check(api.cuMemcpy2DAsync_v2(&weights, c.stream.handle), "V3 exact native FP32 weights");
        // The original one post rings both proxies; no complete-half host transaction and no reply wait here.
        a.submitted = true;
        try a.gpu.post(a.node, a.request.ptr, bytes, c.stream);
        a.last_stream = c.stream;
        return .{ .session = a.session, .sequence_first = a.sequence, .sequence_last = a.sequence, .tiles = 1, .buffer_epoch = c.buffer_epoch, .row_map_digest = c.row_map_digest, .rows = c.rows, .ready_event = a.done };
    }
    fn finish(ctx: *anyopaque, c: routed.Call, d: routed.HalfDest, r: routed.ReadyParts) !routed.ReadyParts {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        return a.take(c, d, r) catch |err| {
            a.failed = err;
            return err;
        };
    }
    fn take(a: *Adapter, c: routed.Call, d: routed.HalfDest, r: routed.ReadyParts) !routed.ReadyParts {
        try a.before(c.deadline);
        const entry = a.inflight[a.completed % proto.LANES_MAX] orelse return error.BadHybridCall;
        if (!a.pending or !std.meta.eql(entry.call, c) or !std.meta.eql(entry.destination, d) or r.session != a.session or r.sequence_first != entry.sequence or r.sequence_last != entry.sequence or r.tiles != 1 or r.rows != c.rows or r.buffer_epoch != c.buffer_epoch or !std.mem.eql(u8, &r.row_map_digest, &c.row_map_digest)) return error.BadHybridCall;
        const bytes = a.node.cfg.settings.layout.rep(c.rows, false);
        try a.gpu.take(a.node, &.{ .{ .ptr = d.base, .bytes = bytes }, .{ .ptr = d.base + bytes, .bytes = bytes } }, c.stream);
        try a.done.record(c.stream);
        try a.before(c.deadline);
        a.inflight[a.completed % proto.LANES_MAX] = null;
        a.completed += 1;
        a.count -= 1;
        a.last_stream = c.stream;
        a.pending = a.count != 0;
        return r;
    }
    fn settle(ctx: *anyopaque, until: u64) !routed.Quiescence {
        const a: *Adapter = @ptrCast(@alignCast(ctx));
        if (a.failed != null or a.pending) return .quarantined;
        for (a.streams, a.proofs) |stream, proof| if (stream) |s| try proof.?.record(s);
        if (a.submitted) while (true) {
            a.before(until) catch |err| {
                a.failed = err;
                return .quarantined;
            };
            const rc = a.done.d.api.cuEventQuery(a.done.handle);
            if (rc == cuda.abi.success) {
                a.submitted = false;
                break;
            }
            if (rc != cuda.abi.error_not_ready) {
                a.failed = error.HybridCopyFailed;
                return .quarantined;
            }
            std.atomic.spinLoopHint();
        };
        for (a.proofs) |proof| if (proof) |ev| while (true) {
            a.before(until) catch |err| {
                a.failed = err;
                return .quarantined;
            };
            const rc = ev.d.api.cuEventQuery(ev.handle);
            if (a.clock() >= until) {
                a.failed = error.Timeout;
                return .quarantined;
            }
            if (rc == cuda.abi.success) break;
            if (rc != cuda.abi.error_not_ready) {
                a.failed = error.HybridCopyFailed;
                return .quarantined;
            }
            std.atomic.spinLoopHint();
        };
        return .quiet;
    }
};

/// Current expert-only native kernel owner; direct V3 actual-row weighted return.
pub const ExpertDevice = @import("cuda_hybrid_v3_expert.zig").ExpertDevice;
