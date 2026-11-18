//! Helios authored the current-native expert owner of released V3 actual-row replies.
//! TensorFold/MCDMA: Ash Hart. Released hybrid split: Hugh Madden and T.J. Purtell.
//! Native RoCE: Mia's AI Lab. Direct V3 g53 arithmetic/launch planner remains unchanged.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const native = @import("cuda_hybrid_expert.zig");
const exl3 = @import("cuda_exl3.zig");
const g53 = @import("cuda_g53.zig");
const X = cuda.xchg;
pub const ExpertDevice = struct {
    fallback: native.Worker,
    arena: cuda.DeviceBuffer,
    landing: cuda.DeviceBuffer,
    output: cuda.DeviceBuffer,
    picks8: cuda.DeviceBuffer,
    weights8: cuda.DeviceBuffer,
    combine: cuda.Function,
    capacity: u32,
    prompt_set: ?*const g53.Set,
    prompt_plan: g53.Plan,
    prompt_big: g53.Big,
    pending: bool = false,
    failed: ?anyerror = null,
    fn bigSizes(rows: u32) [4]usize {
        var out: [4]usize = @splat(0);
        for (1..rows + 1) |n| for ([_]bool{ false, true }) |kt3r| {
            const cfg = g53.tunedCfg(@intCast(n), kt3r) catch unreachable;
            const z = g53.bigNeeds(@intCast(n), cfg);
            for (&out, [4]usize{ z.z, z.xd, z.zd, z.xh }) |*a, b| a.* = @max(a.*, b);
        };
        return out;
    }
    fn sizes(rows: u32, prompt: bool) [9]usize {
        const b = if (prompt) bigSizes(rows) else [4]usize{ 0, 0, 0, 0 };
        return .{ (X.proto.Layout{ .d = 4096, .s = 9 }).slotBytes(rows), @as(usize, rows) * 4096 * 4, 128 * 8 * 4, 128 * 8 * 4, if (prompt) g53.Plan.bytes(rows) else 0, b[0], b[1], b[2], b[3] };
    }
    pub fn storageBytes(rows: u32, prompt: bool) usize {
        var out = native.Worker.storageBytes();
        for (sizes(rows, prompt)) |n| out += std.mem.alignForward(usize, n, 256);
        return out;
    }
    pub fn init(d: *const cuda.Driver, w: *const @import("cuda_hybrid_load.zig").ExpertWeights, k: *const exl3.Kernels, router: cuda.grouped.Router, combine: cuda.Function, stream: cuda.Stream, capacity: u32, prompt_set: ?*const g53.Set) !ExpertDevice {
        if (capacity == 0 or capacity > 4096 or (capacity > 128 and prompt_set == null)) return error.BadExpertGeometry;
        var fallback = try native.Worker.init(d, w, k, router, stream);
        errdefer fallback.releaseStorage();
        const shape = sizes(capacity, prompt_set != null);
        var offsets: [shape.len]usize = undefined;
        var total: usize = 0;
        for (shape, 0..) |n, i| {
            offsets[i] = total;
            total += std.mem.alignForward(usize, n, 256);
        }
        const scope = @import("cuda_hybrid_ledger.zig").scopeOwner(.workspace);
        defer scope.end();
        var arena = try cuda.DeviceBuffer.alloc(d, total);
        errdefer arena.free();
        var buffers: [shape.len]cuda.DeviceBuffer = undefined;
        for (&buffers, shape, offsets) |*b, n, off| b.* = .{ .d = d, .ptr = arena.ptr + off, .len = n, .borrowed = true };
        if (prompt_set != null) {
            // g53's last-arrival counters require initial zero, not allocator luck.
            // Initialize once on the same ordered expert stream; no per-call wait.
            std.debug.assert(buffers[4].len % 4 == 0);
            try d.check(d.api.cuMemsetD32Async(buffers[4].ptr, 0, buffers[4].len / 4, stream.handle), "initialize g53 prompt completion counters");
        }
        return .{ .fallback = fallback, .arena = arena, .landing = buffers[0], .output = buffers[1], .picks8 = buffers[2], .weights8 = buffers[3], .combine = combine, .capacity = capacity, .prompt_set = prompt_set, .prompt_plan = if (prompt_set != null) g53.Plan.place(capacity, buffers[4].ptr) else undefined, .prompt_big = .{ .z = buffers[5].ptr, .xd = buffers[6].ptr, .zd = buffers[7].ptr, .xh = buffers[8].ptr, .z_bytes = shape[5], .xd_bytes = shape[6], .zd_bytes = shape[7], .xh_bytes = shape[8] } };
    }
    pub fn releaseStorage(e: *ExpertDevice) void {
        std.debug.assert(!e.pending and e.failed == null);
        e.fallback.releaseStorage();
        e.arena.free();
        e.* = undefined;
    }
    pub fn compute(e: *ExpertDevice, h: X.proto.Header) !void {
        return e.computeCall(h) catch |err| {
            e.failed = err;
            return err;
        };
    }
    fn computeCall(e: *ExpertDevice, h: X.proto.Header) !void {
        if (e.failed) |err| return err;
        const w = &e.fallback;
        const rows = std.math.cast(u32, h.rows) orelse return error.BadHybridCall;
        if (rows == 0 or rows > e.capacity or h.fseq <= 0 or h.layers != 1 or h.layer < 0) return error.BadHybridCall;
        var data: ?exl3.Experts = null;
        for (w.weights.layers) |l| if (l.index == h.layer) {
            data = l.data;
            break;
        };
        const layer = data orelse return error.BadHybridLayer;
        if (w.stopping) |stop| if (stop.load(.acquire)) return error.Cancelled;
        const d = w.stream.d;
        const off = (X.proto.Layout{ .d = 4096, .s = 9 }).offsets(rows);
        e.pending = true;
        if (h.mode != .eager) {
            const set = e.prompt_set orelse return error.MissingV3Prompt;
            // shared_suh means loader-proven common EXL3 input rotation, NOT
            // the local shared expert branch. Retain V3's once-a-row rotation
            // only when every routed gate/up input scale is identical.
            var launches = g53.Launches{};
            try g53.ffn(&launches, .{ .layer = g53.nativeLayer(layer), .shared_suh = if (layer.shx) layer.suh_g else null, .x = e.landing.ptr + off.normed, .xs = 4096, .pick = e.landing.ptr + off.pick, .ps = 9, .wts = e.landing.ptr + off.wts, .ws = 9, .out = e.output.ptr, .rows = rows, .limit = w.weights.config.swiglu_limit, .plan = e.prompt_plan, .big = e.prompt_big }, try g53.tunedCfg(rows, true));
            try set.run(d, w.stream, &launches);
        } else {
            if (rows > 128) return error.BadHybridCall;
            const p: cuda.abi.Memcpy2D = .{ .src = e.landing.ptr + off.pick, .src_pitch = 9 * 4, .dst = e.picks8.ptr, .dst_pitch = 8 * 4, .width = 8 * 4, .height = rows };
            try d.check(d.api.cuMemcpy2DAsync_v2(&p, w.stream.handle), "V3 native ordered picks9 to8");
            const wt: cuda.abi.Memcpy2D = .{ .src = e.landing.ptr + off.wts, .src_pitch = 9 * 4, .dst = e.weights8.ptr, .dst_pitch = 8 * 4, .width = 8 * 4, .height = rows };
            try d.check(d.api.cuMemcpy2DAsync_v2(&wt, w.stream.handle), "V3 exact native routed weights9 to8");
            // Preserve the current native runRouter scratch and its entire native9 FP32 plane. No alias/shrink.
            try native.runRouter(w.kernels, w.router, w.stream, layer, w.scratch, w.plan, w.slots, e.landing.ptr + off.normed, 4096, e.picks8.ptr, rows, 288, 4096, 1024, w.weights.config.swiglu_limit, false, w.output.ptr);
            // Same ordered FP32 FMA kernel formerly used by the head, now once on each expert. No remote-half merge.
            try @import("cuda_glue.zig").combineFunction(e.combine, w.stream, w.output.ptr, true, e.weights8.ptr, e.output.ptr, rows, 4096, 8);
        }
        try w.done.record(w.stream);
    }
    pub fn settle(e: *ExpertDevice, until: u64) !bool {
        if (e.failed) |err| return err;
        if (!e.pending) return true;
        while (tp.sock.nowNs() < until) {
            const rc = e.fallback.stream.d.api.cuEventQuery(e.fallback.done.handle);
            if (rc == cuda.abi.success) {
                // A late driver query cannot renew the original ownership bound.
                if (tp.sock.nowNs() >= until) return false;
                e.pending = false;
                return true;
            }
            if (rc != cuda.abi.error_not_ready) {
                e.failed = error.HybridCopyFailed;
                return error.HybridCopyFailed;
            }
            std.atomic.spinLoopHint();
        }
        return false;
    }
};
