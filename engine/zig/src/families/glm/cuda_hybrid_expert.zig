//! Wire8→native9 adapter. Native EXL3 remains the only routed executor and returns unweighted FP32 slots.
const cuda=@import("cuda");
const std=@import("std");
const transport=@import("cuda_hybrid_transport.zig");
const tp=@import("tp");
const exl3=@import("cuda_exl3.zig");
const experts=@import("cuda_experts.zig");
/// Borrowed exact expert-only weight/kernel owners and bounded device staging. No attention/engine state lives here.
pub const Worker=struct {
    weights: *const @import("cuda_hybrid_load.zig").ExpertWeights,
    kernels: *const exl3.Kernels,router: cuda.grouped.Router,stream: cuda.Stream,
    arena: cuda.DeviceBuffer,input: cuda.DeviceBuffer,output: cuda.DeviceBuffer,slots: Slots,scratch: exl3.Scratch,plan: cuda.grouped.Plan,done: cuda.Event,
    peer: ?transport.FrameIO=null,pending: bool=false,failed: ?anyerror=null,stopping:?*const std.atomic.Value(bool)=null,
    fn storageSizes() [13]usize {
        var sc: [4]usize=@splat(0);
        inline for(.{false,true}) |prompt| inline for(.{false,true}) |shx| {const shape=exl3.scratchBytes(128,4096,1024,prompt,shx);for(&sc,shape) |*n,bytes| n.*=@max(n.*,bytes);};
        const pairs=128*9;
        return .{transport.wire.request_bytes,transport.wire.reply_bytes,128*9*4,128*9*4096*4,sc[0],sc[1],sc[2],sc[3],pairs*4,cuda.grouped.maxItems(pairs,289,16)*3*4,8,pairs*4,((pairs+1023)/1024)*289*4};
    }
    pub fn storageBytes() usize {var n:usize=0;for(storageSizes())|b|n+=std.mem.alignForward(usize,b,256);return n;}
    pub fn init(d: *const cuda.Driver,w: *const @import("cuda_hybrid_load.zig").ExpertWeights,k: *const exl3.Kernels,router: cuda.grouped.Router,stream: cuda.Stream) !Worker {
        const c=w.config;if(!c.exl3 or c.hidden!=4096 or c.experts!=288 or c.topk!=8 or c.moe_inter!=2048 or w.cut.lo!=@as(usize,w.half)*1024 or w.cut.hi!=(@as(usize,w.half)+1)*1024) return error.BadExpertGeometry;
        const sizes=storageSizes();
        var offsets: [sizes.len]usize=undefined;var bytes: usize=0;for(sizes,0..) |n,i| {offsets[i]=bytes;bytes+=std.mem.alignForward(usize,n,256);}
        const memory_scope=@import("cuda_hybrid_ledger.zig").scopeOwner(.workspace);defer memory_scope.end();
        var arena=try cuda.DeviceBuffer.alloc(d,bytes);errdefer arena.free();
        var p: [sizes.len]u64=undefined;for(&p,offsets) |*address,offset| address.*=arena.ptr+offset;
        const done=try cuda.Event.init(d,false);
        return .{.weights=w,.kernels=k,.router=router,.stream=stream,.arena=arena,
            .input=.{.d=d,.ptr=p[0],.len=sizes[0],.borrowed=true},.output=.{.d=d,.ptr=p[1],.len=sizes[1],.borrowed=true},
            .slots=.{.native_picks=p[2],.native_y=p[3]},.scratch=.{.picks=p[4],.xg=p[5],.xu=p[6],.z=p[7]},
            .plan=.{.members=p[8],.items=p[9],.counts=p[10],.rank=p[11],.hist=p[12]},.done=done};
    }
    pub fn releaseStorage(w: *Worker) void {std.debug.assert(!w.pending and w.failed==null);w.done.deinit();w.arena.free();w.*=undefined;}
    pub fn compute(ctx: *anyopaque,h: transport.wire.Header,input: []const u8,output: []u8,until: u64) !void {
        const w: *Worker=@ptrCast(@alignCast(ctx));return w.computeCall(h,input,output,until) catch |err| {w.failed=err;return err;};
    }
    fn computeCall(w: *Worker,h: transport.wire.Header,input: []const u8,output: []u8,until: u64) !void {
        if(w.failed) |err| return err;if(w.pending) return error.HybridQuarantined;
        const n: usize=h.live_rows;const c=w.weights.config;
        if(n==0 or n>128 or h.hidden!=4096 or h.top_k!=8 or h.half!=w.weights.half or input.len!=transport.wire.request_bytes or output.len!=transport.wire.reply_bytes) return error.BadHybridCall;
        if(c.hidden!=4096 or c.topk!=8 or c.experts!=288 or c.moe_inter!=2048 or !c.exl3) return error.BadExpertGeometry;
        const picks_at=128*4096*2;
        if(!std.mem.allEqual(u8,input[n*4096*2..picks_at],0) or !std.mem.allEqual(u8,input[picks_at+n*8*4..],0)) return error.BadUnusedRows;
        for(0..n*8) |i| {const pick=std.mem.readInt(i32,input[picks_at+i*4..][0..4],.little);if(pick<0 or pick>=288) return error.BadHybridPick;}
        if((h.phase==.target and (h.layer>=c.layers or !c.isMoe(h.layer))) or (h.phase==.mtp and (h.layer!=c.layers or c.mtp==0))) return error.BadHybridLayer;
        var found: ?exl3.Experts=null;for(w.weights.layers) |layer| if(layer.index==h.layer) {found=layer.data;break;};
        const data=found orelse return error.BadHybridLayer;
        if(tp.sock.nowNs()>=until) return error.Timeout;if(w.stopping)|s|if(s.load(.acquire))return error.Cancelled;if(w.peer) |p| try p.check(p.ctx);
        @memset(output,0);const d=w.stream.d;w.pending=true;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(w.input.ptr,input.ptr,input.len,w.stream.handle),"hybrid pinned expert input");
        try runRouter(w.kernels,w.router,w.stream,data,w.scratch,w.plan,w.slots,w.input.ptr,4096,w.input.ptr+picks_at,n,288,4096,1024,c.swiglu_limit,h.execution_mode==.prompt,w.output.ptr);
        try d.check(d.api.cuMemcpyDtoHAsync_v2(output.ptr,w.output.ptr,n*8*4096*4,w.stream.handle),"hybrid pinned expert unweighted FP32 output");
        try w.done.record(w.stream);
        while(true) {
            if(tp.sock.nowNs()>=until) return error.Timeout;if(w.stopping)|s|if(s.load(.acquire))return error.Cancelled;if(w.peer) |p| try p.check(p.ctx);
            const rc=d.api.cuEventQuery(w.done.handle);if(rc==cuda.abi.success) {w.pending=false;return;}
            if(rc!=cuda.abi.error_not_ready) try d.result(rc);std.atomic.spinLoopHint();
        }
    }
};
pub const Slots=struct { native_picks: u64,native_y: u64 };
pub fn run(k: *const exl3.Kernels,ek: *const experts.Kernels,stream: cuda.Stream,w: exl3.Experts,scratch: exl3.Scratch,plan: cuda.grouped.Plan,slots: Slots,x: u64,x_stride: usize,picks8: u64,rows: usize,count: usize,dims: usize,width: usize,limit: f32,prompt: bool,out8: u64) !void {
    return runRouter(k,ek.router,stream,w,scratch,plan,slots,x,x_stride,picks8,rows,count,dims,width,limit,prompt,out8);
}
pub fn runRouter(k: *const exl3.Kernels,router: cuda.grouped.Router,stream: cuda.Stream,w: exl3.Experts,scratch: exl3.Scratch,plan: cuda.grouped.Plan,slots: Slots,x: u64,x_stride: usize,picks8: u64,rows: usize,count: usize,dims: usize,width: usize,limit: f32,prompt: bool,out8: u64) !void {
    if(rows==0 or rows>128 or count!=288 or dims!=4096 or width!=1024 or x_stride<dims) return error.BadHybridExpertCall;
    const m: cuda.abi.Memcpy2D=.{.src=picks8,.src_pitch=8*4,.dst=slots.native_picks,.dst_pitch=9*4,.width=8*4,.height=rows};
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m,stream.handle),"hybrid expand routed picks8 to native9");
    for(0..rows) |r| try stream.d.check(stream.d.api.cuMemsetD32Async(slots.native_picks+(r*9+8)*4,@intCast(count),1,stream.handle),"hybrid native shared sentinel");
    try exl3.runRouter(k,router,stream,w,scratch,plan,x,x_stride,slots.native_picks,9,rows,count,dims,width,limit,prompt,slots.native_y,0);
    const y: cuda.abi.Memcpy2D=.{.src=slots.native_y,.src_pitch=9*dims*4,.dst=out8,.dst_pitch=8*dims*4,.width=8*dims*4,.height=rows};
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&y,stream.handle),"hybrid compact unweighted native FP32 slots");
}
