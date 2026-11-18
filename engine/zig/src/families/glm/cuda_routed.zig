//! Optional routed-result ownership. Native tensor ranks are not hybrid contribution planes.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
/// Direct released-V3 packet/exchange adapter; serving construction is separately admitted.
pub const V3 = @import("cuda_hybrid_v3.zig");

pub const Phase = enum(u16) { target, mtp };
pub const Call = struct {
    layer: u32, phase: Phase, native_round: u64, buffer_epoch: u64, row_map_digest: [32]u8,
    rows: usize, hidden: usize, top_k: usize, prompt: bool, x_bf16: u64, x_stride_elements: usize,
    picks_i32: u64, weights_f32: u64, native_slot_stride: usize, stream: cuda.Stream, deadline: u64,
};
pub const HalfDest = struct { base: u64, rows: usize, weights_f32: u64 };
pub const Quiescence = enum { quiet, quarantined };
pub const Provider = struct {
    ctx: *anyopaque,
    /// Producer proves same-stream ordering; close still requires full settle.
    device_ordered: bool = false,
    touch: ?*const fn(*anyopaque,cuda.Stream) anyerror!void=null,
    run: *const fn (*anyopaque,Call,HalfDest) anyerror!ReadyParts,
    /// V3 post/shared/take: enqueue the take only after the local shared work.
    finish: ?*const fn (*anyopaque,Call,HalfDest,ReadyParts) anyerror!ReadyParts = null,
    check: *const fn (*anyopaque) anyerror!void,
    settle: *const fn (*anyopaque,u64) anyerror!Quiescence,
};
pub const ReadyParts = struct {
    session: u128, sequence_first: u64, sequence_last: u64, tiles: u32,
    buffer_epoch: u64, row_map_digest: [32]u8, rows: usize, ready_event: cuda.Event,
};
pub const RoutedResult = union(enum) { local_slots, remote: ReadyParts };
pub const BranchView = struct {
    base: u64, rows: usize, row_stride_bytes: usize, plane_stride_bytes: usize,
    contributors: u32, order: enum { native_tensor, half0_half1_shared }, ready_event: ?cuda.Event = null,
};
/// Per-invocation storage. The half planes and packed epilogue never overlap; twins own separate storage.
pub const Buffer = struct {
    allocator: std.mem.Allocator = std.heap.page_allocator,
    capacity: usize, dims: usize, halves: u64, epilogue: u64, weights: u64, shared_weights: u64,
    result: RoutedResult = .local_slots,
    remote_pending: bool = false,
    provider: ?Provider = null,
    call: ?Call = null,
    epoch: u64 = 0,
    deadline: u64 = std.math.maxInt(u64),

    pub fn check(b: *const Buffer) !void {
        if (b.provider) |p| try p.check(p.ctx);
    }

    /// Native router has finished; compact its unchanged ordered weights without arithmetic.
    pub fn dispatch(b: *Buffer) !void {
        if(b.remote_pending)return error.HybridQuarantined;
        const p=b.provider orelse return error.MissingRoutedProvider;
        try p.check(p.ctx);
        const c=b.call orelse return error.MissingRoutedCall;
        if (c.rows==0 or c.rows>b.capacity or c.hidden!=b.dims or c.top_k!=8 or c.native_slot_stride!=9) return error.BadHybridRows;
        if (!p.device_ordered) if (try p.settle(p.ctx,c.deadline)!=.quiet) return error.HybridQuarantined;
        b.result=.local_slots;b.remote_pending=false;
        const routes: cuda.abi.Memcpy2D=.{.src=c.weights_f32,.src_pitch=9*4,.dst=b.weights,.dst_pitch=8*4,.width=8*4,.height=c.rows};
        try c.stream.d.check(c.stream.d.api.cuMemcpy2DAsync_v2(&routes,c.stream.handle),"hybrid native routed weights stride9 to8");
        const shared: cuda.abi.Memcpy2D=.{.src=c.weights_f32+8*4,.src_pitch=9*4,.dst=b.shared_weights,.dst_pitch=4,.width=4,.height=c.rows};
        try c.stream.d.check(c.stream.d.api.cuMemcpy2DAsync_v2(&shared,c.stream.handle),"hybrid native shared weights contiguous");
        const ready=try p.run(p.ctx,c,.{.base=b.halves,.rows=c.rows,.weights_f32=b.weights});
        try p.check(p.ctx);
        if (ready.rows!=c.rows or ready.buffer_epoch!=c.buffer_epoch or !std.mem.eql(u8,&ready.row_map_digest,&c.row_map_digest) or ready.tiles==0 or ready.sequence_first==0 or ready.sequence_last<ready.sequence_first or ready.sequence_last-ready.sequence_first+1!=ready.tiles) return error.BadRoutedCompletion;
        b.result=.{.remote=ready};b.remote_pending=p.finish!=null;
    }

    /// Actual MoE fill calls this after its shared-expert launches, once per call.
    pub fn complete(b: *Buffer) !void {
        if(!b.remote_pending)return;
        const p=b.provider orelse return error.MissingRoutedProvider;
        const c=b.call orelse return error.MissingRoutedCall;
        const before=switch(b.result){.remote=>|r|r,.local_slots=>return error.NoRemoteResult};
        const finish=p.finish orelse return error.MissingRoutedCompletion;
        try p.check(p.ctx);
        const ready=try finish(p.ctx,c,.{.base=b.halves,.rows=c.rows,.weights_f32=b.weights},before);
        if(ready.session!=before.session or ready.sequence_first!=before.sequence_first or ready.sequence_last!=before.sequence_last or ready.tiles!=before.tiles or ready.rows!=before.rows or ready.buffer_epoch!=before.buffer_epoch or !std.mem.eql(u8,&ready.row_map_digest,&before.row_map_digest))return error.BadRoutedCompletion;
        try p.check(p.ctx);b.result=.{.remote=ready};b.remote_pending=false;
    }
};

/// A tightly packed, nonaliasing piece for the actual native epilogue. Shared remains a third FP32 plane.
pub fn fill(set: glue.Set, stream: cuda.Stream, b: *Buffer, sy: u64, lo: usize, hi: usize) !BranchView {
    try b.complete();
    const ready = switch (b.result) { .remote => |r| r, .local_slots => return error.NoRemoteResult };
    if (hi <= lo or hi > ready.rows or ready.rows > b.capacity or b.dims == 0) return error.BadHybridRows;
    try b.check();
    const n = hi-lo;
    const row = try std.math.mul(usize,b.dims,4);
    const bytes = try std.math.mul(usize,n,row);
    const base = b.epilogue + try std.math.mul(usize,3*lo,row);
    try stream.wait(ready.ready_event);
    try b.check();
    for (0..2) |half| {
        const src = b.halves + (half*ready.rows+lo)*row;
        try stream.d.check(stream.d.api.cuMemcpyDtoDAsync_v2(base+half*bytes,src,bytes,stream.handle),"hybrid durable half to packed piece");
    }
    try glue.combine(set,stream,sy+lo*row,true,b.shared_weights+lo*4,base+2*bytes,n,b.dims,1);
    return .{ .base=base,.rows=n,.row_stride_bytes=row,.plane_stride_bytes=bytes,.contributors=3,.order=.half0_half1_shared,.ready_event=ready.ready_event };
}
