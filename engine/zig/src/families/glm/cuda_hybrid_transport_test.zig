//! CPU device boundary fixture: async copies accept only allocations made via the pinned-host API.
const std=@import("std");const cuda=@import("cuda");const tp=@import("tp");const glm=@import("glm");const t=std.testing;
const F=struct {
    var blocks: std.AutoHashMapUnmanaged(u64,[]align(16)u8)=.empty;
    var pinned: std.AutoHashMapUnmanaged(u64,usize)=.empty;
    var event_destroys: usize=0;
    var cancel_on_record:?*std.atomic.Value(bool)=null;
    var queries: usize=0;var late_query: bool=false;var pinned_copies: usize=0;var native_launches: usize=0;
    var records: [64]usize=undefined;var nrecords: usize=0;
    fn alloc(p: *u64,n: usize) callconv(.c) c_int {const b=t.allocator.alignedAlloc(u8,.@"16",n) catch return 2;p.*=@intFromPtr(b.ptr);blocks.put(t.allocator,p.*,b) catch return 2;return 0;}
    fn free(p: u64) callconv(.c) c_int {const b=blocks.fetchRemove(p) orelse return 1;t.allocator.free(b.value);return 0;}
    fn host(p: *?*anyopaque,n: usize,_: c_uint) callconv(.c) c_int {var ptr: u64=0;const rc=alloc(&ptr,n);if(rc!=0)return rc;p.*=@ptrFromInt(ptr);pinned.put(t.allocator,ptr,n) catch return 2;return 0;}
    fn hfree(p: ?*anyopaque) callconv(.c) c_int {_=pinned.remove(@intFromPtr(p.?));return free(@intFromPtr(p.?));}
    fn isPinned(p: u64,n: usize) bool {var it=pinned.iterator();while(it.next()) |e| if(p>=e.key_ptr.* and p-e.key_ptr.*<=e.value_ptr.* and n<=e.value_ptr.*-(p-e.key_ptr.*)) return true;return false;}
    fn h2d(dst: u64,src: ?*const anyopaque,n: usize,_: cuda.abi.Stream) callconv(.c) c_int {if(!isPinned(@intFromPtr(src.?),n)) return 1;pinned_copies+=1;@memcpy(@as([*]u8,@ptrFromInt(dst))[0..n],@as([*]const u8,@ptrCast(src.?))[0..n]);return 0;}
    fn d2h(dst: ?*anyopaque,src: u64,n: usize,_: cuda.abi.Stream) callconv(.c) c_int {if(!isPinned(@intFromPtr(dst.?),n)) return 1;pinned_copies+=1;@memcpy(@as([*]u8,@ptrCast(dst.?))[0..n],@as([*]const u8,@ptrFromInt(src))[0..n]);return 0;}
    fn copy2d(m: *const cuda.abi.Memcpy2D,_: cuda.abi.Stream) callconv(.c) c_int {
        const src=if(m.src_type==1) @intFromPtr(m.src_host.?) else m.src;
        const dst=if(m.dst_type==1) @intFromPtr(m.dst_host.?) else m.dst;
        if(m.src_type==1 and !isPinned(src,(m.height-1)*m.src_pitch+m.width)) return 1;
        if(m.dst_type==1 and !isPinned(dst,(m.height-1)*m.dst_pitch+m.width)) return 1;
        for(0..m.height) |r| @memcpy(@as([*]u8,@ptrFromInt(dst+r*m.dst_pitch))[0..m.width],@as([*]const u8,@ptrFromInt(src+r*m.src_pitch))[0..m.width]);
        return 0;
    }
    fn memset(p: u64,value: c_uint,n: usize,_: cuda.abi.Stream) callconv(.c) c_int {@memset(@as([*]u32,@ptrFromInt(p))[0..n],value);return 0;}
    fn event(p: *cuda.abi.Event,_: c_uint) callconv(.c) c_int {p.*=@ptrFromInt(1);return 0;}
    fn record(_: cuda.abi.Event,s: cuda.abi.Stream) callconv(.c) c_int {records[nrecords]=@intFromPtr(s);nrecords+=1;if(cancel_on_record)|flag|flag.store(true,.release);return 0;}
    fn destroy(_: cuda.abi.Event) callconv(.c) c_int {event_destroys+=1;return 0;}
    fn query(_: cuda.abi.Event) callconv(.c) c_int {queries+=1;if(late_query) tp.sock.sleepNs(5*std.time.ns_per_ms);return 0;}
    fn launch(f: cuda.abi.Function,gx: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: cuda.abi.Stream,args: ?[*]?*anyopaque,_: ?[*]?*anyopaque) callconv(.c) c_int {
        if(@intFromPtr(f)==77) {native_launches+=1;return 0;}
        if(@intFromPtr(f)==99) {
            const A=struct {fn a(comptime T: type,p: ?[*]?*anyopaque,i: usize) T {return @as(*const T,@ptrCast(@alignCast(p.?[i].?))).*;}};
            const y: [*]const f32=@ptrFromInt(A.a(u64,args,0));const weights: [*]const f32=@ptrFromInt(A.a(u64,args,2));const out: [*]f32=@ptrFromInt(A.a(u64,args,3));
            const dims: usize=@intCast(A.a(i64,args,4));const slots: usize=@intCast(A.a(c_int,args,5));
            for(0..gx) |row| for(0..dims) |col| {var value: f32=0;for(0..slots) |k| value=@mulAdd(f32,y[(row*slots+k)*dims+col],weights[row*slots+k],value);out[row*dims+col]=value;};
            return 0;
        }
        const a: [*]const u64=@ptrCast(@alignCast(args.?[0].?));@memcpy(@as([*]u8,@ptrFromInt(a[1]))[0..1048576],@as([*]const u8,@ptrFromInt(a[0]))[0..1048576]);return 0;
    }
};
const Responses=struct {
    var announced: [2]bool=.{false,false};
    fn announce(ctx: *anyopaque,_: glm.moe.hybridTransport.wire.Header,_: u64) !void {const half: *u8=@ptrCast(@alignCast(ctx));announced[half.*]=true;}
    fn now(_: *anyopaque) u64 {return tp.sock.nowNs();}
    fn check(_: *anyopaque) !void {}
    fn exchange(ctx: *anyopaque,send: []const u8,recv: []u8,_: u64) !void {
        const tr=glm.moe.hybridTransport;const half: *u8=@ptrCast(@alignCast(ctx));
        var h=try tr.wire.Header.decode(send[0..256]);try t.expectEqual(@as(u16,half.*),h.half);@memset(recv,0);
        if(h.kind==.request) {
            h.kind=.idle;h.fragment_payload_bytes=0;
            if(h.fragment_index==0 and !announced[half.*]) {h.native_round=0;h.buffer_epoch=0;h.layer=0;h.phase=.target;h.live_rows=128;h.execution_mode=.decode;h.tile_row_lo=0;h.call_rows=128;h.row_map_sha256=@splat(0);h.bind();}
        } else {
            h.kind=.reply;h.fragment_payload_bytes=@intCast(@min(tr.wire.frame_payload,h.total_payload_bytes-h.payload_offset));
            var off: usize=0;while(off<h.fragment_payload_bytes) : (off+=4) {
                const word=(h.payload_offset+off)/4;const row=word/(8*4096);const k=word/4096%8;
                const v: f32=if(row<h.live_rows) @as(f32,@floatFromInt(k+1))*@as(f32,if(half.*==0) 1 else 10) else 0;
                std.mem.writeInt(u32,recv[256+off..][0..4],@bitCast(v),.little);
            }
        }
        h.encode(recv[0..256]);
    }
};
test "hybrid native IO: actual provider contracts both halves over129 rows into durable FP32 planes" {
    if(comptime !@hasDecl(glm.moe.hybridTransport,"Coordinator")) {
        var r: tp.roce.Roce=undefined;r.aborted=.init(false);r.slot_bytes=1048576;r.chunk_abi=false;try r.exchangeBytes(1,2,16777216,null);return;
    }
    defer F.blocks.deinit(t.allocator);defer F.pinned.deinit(t.allocator);F.blocks=.empty;F.pinned=.empty;
    var d: cuda.Driver=undefined;d.api.cuMemAlloc_v2=F.alloc;d.api.cuMemFree_v2=F.free;d.api.cuMemHostAlloc=F.host;d.api.cuMemFreeHost=F.hfree;d.api.cuMemcpyHtoDAsync_v2=F.h2d;d.api.cuMemcpyDtoHAsync_v2=F.d2h;d.api.cuMemcpy2DAsync_v2=F.copy2d;d.api.cuEventCreate=F.event;d.api.cuEventRecord=F.record;d.api.cuEventDestroy_v2=F.destroy;d.api.cuEventQuery=F.query;d.api.cuLaunchKernel=F.launch;
    var glue: glm.glue.Kernels=undefined;glue.combine=.{.d=&d,.handle=@ptrFromInt(99)};
    var halves=try cuda.DeviceBuffer.alloc(&d,2*129*4096*4);defer halves.free();
    var picks=try cuda.DeviceBuffer.alloc(&d,129*9*4);defer picks.free();@memset(@as([*]u8,@ptrFromInt(picks.ptr))[0..picks.len],0);
    var x=try cuda.DeviceBuffer.alloc(&d,129*4096*2);defer x.free();@memset(@as([*]u8,@ptrFromInt(x.ptr))[0..x.len],0);
    var weights=try cuda.DeviceBuffer.alloc(&d,129*8*4);defer weights.free();
    for(@as([*]f32,@ptrFromInt(weights.ptr))[0..129*8],0..) |*v,i| v.*=@floatFromInt(i%8+1);
    var h0: u8=0;var h1: u8=1;
    const tr=glm.moe.hybridTransport;
    var io0=tr.FrameIO{.ctx=&h0,.now=Responses.now,.check=Responses.check,.exchange=Responses.exchange};var io1=tr.FrameIO{.ctx=&h1,.now=Responses.now,.check=Responses.check,.exchange=Responses.exchange};
    Responses.announced=.{false,false};if(comptime @hasField(tr.FrameIO,"announce")) {io0.announce=Responses.announce;io1.announce=Responses.announce;}
    const c=try tr.Coordinator.init(t.allocator,&d,.{.glue=&glue},.{io0,io1},.{.session=1,.model=@splat(0xAA),.settings=@splat(0xBB),.transaction_timeout_ns=5_000_000_000});
    defer c.releaseStorage();
    const provider=c.provider();F.nrecords=0;
    if(comptime @hasField(@TypeOf(provider),"touch")) {
        try t.expect(provider.touch!=null);try provider.touch.?(provider.ctx,.{.d=&d,.handle=null});try provider.touch.?(provider.ctx,.{.d=&d,.handle=@ptrFromInt(1)});
    } else try t.expect(false);
    const ready=try provider.run(provider.ctx,.{.layer=3,.phase=.target,.native_round=1,.buffer_epoch=2,.row_map_digest=@splat(0x11),.rows=129,.hidden=4096,.top_k=8,.prompt=true,.x_bf16=x.ptr,.x_stride_elements=4096,.picks_i32=picks.ptr,.weights_f32=0,.native_slot_stride=9,.stream=.{.d=&d,.handle=null},.deadline=tp.sock.nowNs()+5_000_000_000},.{.base=halves.ptr,.rows=129,.weights_f32=weights.ptr});
    try t.expectEqual(@as(u32,2),ready.tiles);try t.expectEqual(@as(u64,1),ready.sequence_first);try t.expectEqual(@as(u64,2),ready.sequence_last);
    const output=@as([*]const f32,@ptrFromInt(halves.ptr))[0..2*129*4096];
    for(output[0..129*4096]) |v| try t.expectEqual(@as(f32,204),v);
    for(output[129*4096..]) |v| try t.expectEqual(@as(f32,2040),v);
    try t.expectEqual(glm.moe.routed.Quiescence.quiet,try provider.settle(provider.ctx,tp.sock.nowNs()+1_000_000_000));
    try t.expectEqual(@as(usize,0),F.records[F.nrecords-2]);try t.expectEqual(@as(usize,1),F.records[F.nrecords-1]);
}

test "hybrid native IO: provider settle quarantines a late successful consumer event" {
    defer F.blocks.deinit(t.allocator);defer F.pinned.deinit(t.allocator);F.blocks=.empty;F.pinned=.empty;F.nrecords=0;F.late_query=true;defer F.late_query=false;
    var d: cuda.Driver=undefined;d.api.cuMemAlloc_v2=F.alloc;d.api.cuMemFree_v2=F.free;d.api.cuMemHostAlloc=F.host;d.api.cuMemFreeHost=F.hfree;d.api.cuEventCreate=F.event;d.api.cuEventRecord=F.record;d.api.cuEventDestroy_v2=F.destroy;d.api.cuEventQuery=F.query;
    const tr=glm.moe.hybridTransport;var half: u8=0;const pair=tr.FrameIO{.ctx=&half,.now=Responses.now,.check=Responses.check,.exchange=Responses.exchange};var glue: glm.glue.Kernels=undefined;
    const c=try tr.Coordinator.init(t.allocator,&d,.{.glue=&glue},.{pair,pair},.{.session=1,.model=@splat(1),.settings=@splat(2),.transaction_timeout_ns=std.time.ns_per_s});
    defer {c.failed=null;c.pending=false;c.releaseStorage();} // synthetic-only cleanup; the product refuses any retained release
    const p=c.provider();try p.touch.?(p.ctx,.{.d=&d,.handle=null});
    try t.expectEqual(glm.moe.routed.Quiescence.quarantined,try p.settle(p.ctx,tp.sock.nowNs()+std.time.ns_per_ms));try t.expect(c.failed!=null);
}

test "hybrid native IO: actual selected frame exchange uses pinned copies and bounded event completion" {
    defer F.blocks.deinit(t.allocator);defer F.pinned.deinit(t.allocator);F.blocks=.empty;F.pinned=.empty;F.queries=0;F.pinned_copies=0;
    var d: cuda.Driver=undefined;
    d.api.cuMemAlloc_v2=F.alloc;d.api.cuMemFree_v2=F.free;d.api.cuMemHostAlloc=F.host;d.api.cuMemFreeHost=F.hfree;
    d.api.cuMemcpyHtoDAsync_v2=F.h2d;d.api.cuMemcpyDtoHAsync_v2=F.d2h;d.api.cuEventCreate=F.event;d.api.cuEventRecord=F.record;d.api.cuEventDestroy_v2=F.destroy;d.api.cuEventQuery=F.query;d.api.cuLaunchKernel=F.launch;
    var r: tp.roce.Roce=undefined;r.aborted=.init(false);r.slot_bytes=1048576;r.chunk_abi=false;r.d=&d;r.dev=0;r.timeout_ns=100_000_000;r.me=0;r.fast=0;r.max_grid=1;r.wire=.{.shm=.{.stripes=1,.local=undefined,.peer=undefined,.layout=undefined}};
    const region=try t.allocator.alignedAlloc(u8,.fromByteUnits(4096),8192);defer t.allocator.free(region);@memset(region,0);r.region=region;
    r.state=.{.d=&d,.ptr=0x1000,.len=16};r.func=.{.d=&d,.handle=@ptrFromInt(7)};
    const input=try t.allocator.alloc(u8,1048576);defer t.allocator.free(input);@memset(input,0xA5);
    const output=try t.allocator.alloc(u8,1048576);defer t.allocator.free(output);
    if(comptime @hasDecl(glm.moe.hybridTransport,"NativePair")) {
        var p=try glm.moe.hybridTransport.NativePair.init(&d,&r,.{.d=&d,.handle=null});
        defer p.frames.free();defer p.host.free();defer p.done.deinit();
        const io=p.frameIO();try io.exchange(io.ctx,input,output,tp.sock.nowNs()+100_000_000);
        try t.expectEqualSlices(u8,input,output);
    } else {
        const a=try t.allocator.alloc(u8,1048576);defer t.allocator.free(a);@memcpy(a,input);
        try r.exchangeBefore(@intFromPtr(a.ptr),@intFromPtr(output.ptr),1048576,null,tp.sock.nowNs()+100_000_000);
    }
    try t.expect(F.queries>0);try t.expectEqual(@as(usize,2),F.pinned_copies);
    try t.expectEqual(@as(u32,0),F.blocks.count());
}

test "hybrid native IO: actual peer-control closure poisons pair before GPU consumers" {
    var fds: [2]std.c.fd_t=undefined;try t.expectEqual(@as(c_int,0),std.c.socketpair(std.posix.AF.UNIX,std.posix.SOCK.STREAM,0,&fds));
    defer tp.sock.close(fds[1]);tp.sock.close(fds[0]);
    const region=try t.allocator.alignedAlloc(u8,.fromByteUnits(4096),8192);defer t.allocator.free(region);@memset(region,0);
    var r: tp.roce.Roce=undefined;r.region=region;r.aborted=.init(false);
    var pair: glm.moe.hybridTransport.NativePair=undefined;pair.r=&r;pair.failed=null;pair.control=fds[1];
    const io=pair.frameIO();try t.expectError(error.Closed,io.check(io.ctx));try t.expectError(error.Closed,io.check(io.ctx));
}

fn readWake(pair: *glm.moe.hybridTransport.NativePair,identity: glm.moe.hybridTransport.Identity,sequence: u64,until: u64) !glm.moe.hybridTransport.wire.Header {
    if(comptime @hasDecl(glm.moe.hybridTransport.NativePair,"readWake")) return pair.readWake(identity,0,sequence,until);
    var bytes: [256]u8=undefined;try tp.sock.recvExact(pair.control.?,&bytes,until);return glm.moe.hybridTransport.wire.Header.decode(&bytes);
}
test "hybrid native IO: actual control fd wake binds next role transaction before dispatch" {
    var fds: [2]std.c.fd_t=undefined;try t.expectEqual(@as(c_int,0),std.c.socketpair(std.posix.AF.UNIX,std.posix.SOCK.STREAM,0,&fds));defer tp.sock.close(fds[0]);defer tp.sock.close(fds[1]);
    const region=try t.allocator.alignedAlloc(u8,.fromByteUnits(4096),8192);defer t.allocator.free(region);@memset(region,0);
    var r: tp.roce.Roce=undefined;r.region=region;r.aborted=.init(false);
    var sender: glm.moe.hybridTransport.NativePair=undefined;sender.r=&r;sender.failed=null;sender.control=fds[0];
    var receiver: glm.moe.hybridTransport.NativePair=undefined;receiver.r=&r;receiver.failed=null;receiver.control=fds[1];receiver.call_budget=.{};receiver.turn_until=0;
    const tr=glm.moe.hybridTransport;const identity=tr.Identity{.session=1,.model=@splat(0xAA),.settings=@splat(0xBB),.transaction_timeout_ns=1_000_000_000};
    var h=tr.wire.Header{.session=1,.sequence=1,.native_round=3,.buffer_epoch=7,.layer=3,.phase=.mtp,.half=0,.live_rows=1,.call_rows=1,.model_contract_sha256=identity.model,.row_map_sha256=@splat(0xCC),.settings_sha256=identity.settings};h.bind();
    const Send=struct {p:*tr.NativePair,h:tr.wire.Header,until:u64,failed:?anyerror=null,fn run(s:*@This())void{const io=s.p.frameIO();io.announce.?(io.ctx,s.h,s.until) catch |err| {s.failed=err;};}};
    const until=tp.sock.nowNs()+1_000_000_000;
    var sending=Send{.p=&sender,.h=h,.until=until};const first_thread=try std.Thread.spawn(.{},Send.run,.{&sending});
    const got=readWake(&receiver,identity,1,until);first_thread.join();try t.expect(sending.failed==null);try t.expectEqual(@as(u64,7),(try got).buffer_epoch);try receiver.call_budget.finish(try got);
    h.sequence=2;h.bind();sending=.{.p=&sender,.h=h,.until=until};const second_thread=try std.Thread.spawn(.{},Send.run,.{&sending});defer second_thread.join();
    try t.expectError(error.BadHybridCall,readWake(&receiver,identity,3,until));
}

fn nativeFunctions(p: anytype,d: *cuda.Driver) void {
    inline for(@typeInfo(@TypeOf(p.*)).@"struct".field_names) |name| {
        const T=@TypeOf(@field(p.*,name));if(T==cuda.Function) @field(p.*,name)=.{.d=d,.handle=@ptrFromInt(77)} else if(@typeInfo(T)==.@"struct") nativeFunctions(&@field(p.*,name),d);
    }
}
test "hybrid native IO: actual worker staging uses pinned host input output and frees exact expert scratch" {
    defer F.blocks.deinit(t.allocator);defer F.pinned.deinit(t.allocator);F.blocks=.empty;F.pinned=.empty;F.queries=0;F.nrecords=0;F.native_launches=0;
    var d: cuda.Driver=undefined;d.api.cuMemAlloc_v2=F.alloc;d.api.cuMemFree_v2=F.free;d.api.cuMemHostAlloc=F.host;d.api.cuMemFreeHost=F.hfree;d.api.cuMemcpyHtoDAsync_v2=F.h2d;d.api.cuMemcpyDtoHAsync_v2=F.d2h;d.api.cuMemcpy2DAsync_v2=F.copy2d;d.api.cuMemsetD32Async=F.memset;d.api.cuEventCreate=F.event;d.api.cuEventRecord=F.record;d.api.cuEventDestroy_v2=F.destroy;d.api.cuEventQuery=F.query;d.api.cuLaunchKernel=F.launch;
    var k: glm.exl3.Kernels=undefined;nativeFunctions(&k,&d);k.grouped=.{.{.d=&d,.handle=@ptrFromInt(77)},.{.d=&d,.handle=@ptrFromInt(77)},.{.d=&d,.handle=@ptrFromInt(77)}};
    var router: cuda.grouped.Router=undefined;nativeFunctions(&router,&d);
    const probe: glm.moe.hybridExpert.Worker=undefined;
    var weights: @typeInfo(@TypeOf(probe.weights)).pointer.child=undefined;weights.config=.{.exl3=true};weights.half=0;weights.cut=.{.lo=0,.hi=1024};
    var x3=std.mem.zeroes(glm.exl3.Experts);x3.shx=true;
    var layers=[_]@typeInfo(@TypeOf(weights.layers)).pointer.child{.{.index=3,.data=x3}};weights.layers=&layers;
    var worker=try glm.moe.hybridExpert.Worker.init(&d,&weights,&k,router,.{.d=&d,.handle=null});
    var input=try cuda.HostBuffer.alloc(&d,glm.moe.hybridTransport.wire.request_bytes);defer input.free();@memset(input.bytes,0);
    var output=try cuda.HostBuffer.alloc(&d,glm.moe.hybridTransport.wire.reply_bytes);defer output.free();
    var h=glm.moe.hybridTransport.wire.Header{.session=1,.sequence=1,.native_round=1,.buffer_epoch=1,.layer=3,.phase=.target,.live_rows=1,.call_rows=1};h.bind();
    try glm.moe.hybridExpert.Worker.compute(&worker,h,input.bytes,output.bytes,tp.sock.nowNs()+5_000_000_000);
    try t.expect(!worker.pending);try t.expect(F.native_launches>0);try t.expect(F.queries>0);
    worker.releaseStorage();try t.expectEqual(@as(usize,2),F.blocks.count());try t.expectEqual(F.blocks.count(),F.pinned.count()); // only the caller-owned pinned input/output remain
}

test "hybrid native IO: actual partial role owner cannot publish quiet after close deadline" {
    var d: cuda.Driver=undefined;d.api.cuEventCreate=F.event;d.api.cuEventRecord=F.record;d.api.cuEventDestroy_v2=F.destroy;F.event_destroys=0;F.nrecords=0;
    var peer=glm.moe.hybridForward.Peer{.d=&d,.stream=.{.d=&d,.handle=null},.proof=try cuda.Event.init(&d,false)};
    const receipt=peer.closeChecked(tp.sock.nowNs());
    try t.expect(!receipt.quiet);try t.expect(peer.retained);try t.expectEqual(@as(usize,0),F.event_destroys);
    // CPU-only synthetic event cleanup, deliberately outside the production retained-owner path.
    if(peer.proof) |*ev| ev.deinit();
}

// CPU-only: unchanged native Worker.compute graph and pinned API, cancellation after actual event record.
test "hybrid native IO: I3 cancellation during active worker keeps pending ownership" {
    defer F.blocks.deinit(t.allocator);defer F.pinned.deinit(t.allocator);F.blocks=.empty;F.pinned=.empty;F.queries=0;F.nrecords=0;F.native_launches=0;
    var d:cuda.Driver=undefined;d.api.cuMemAlloc_v2=F.alloc;d.api.cuMemFree_v2=F.free;d.api.cuMemHostAlloc=F.host;d.api.cuMemFreeHost=F.hfree;d.api.cuMemcpyHtoDAsync_v2=F.h2d;d.api.cuMemcpyDtoHAsync_v2=F.d2h;d.api.cuMemcpy2DAsync_v2=F.copy2d;d.api.cuMemsetD32Async=F.memset;d.api.cuEventCreate=F.event;d.api.cuEventRecord=F.record;d.api.cuEventDestroy_v2=F.destroy;d.api.cuEventQuery=F.query;d.api.cuLaunchKernel=F.launch;
    var k:glm.exl3.Kernels=undefined;nativeFunctions(&k,&d);k.grouped=.{.{.d=&d,.handle=@ptrFromInt(77)},.{.d=&d,.handle=@ptrFromInt(77)},.{.d=&d,.handle=@ptrFromInt(77)}};var router:cuda.grouped.Router=undefined;nativeFunctions(&router,&d);
    const W=glm.moe.hybridExpert.Worker;const probe:W=undefined;var weights:@typeInfo(@TypeOf(probe.weights)).pointer.child=undefined;weights.config=.{.exl3=true};weights.half=0;weights.cut=.{.lo=0,.hi=1024};var x3=std.mem.zeroes(glm.exl3.Experts);x3.shx=true;var layers=[_]@typeInfo(@TypeOf(weights.layers)).pointer.child{.{.index=3,.data=x3}};weights.layers=&layers;
    var worker=try W.init(&d,&weights,&k,router,.{.d=&d,.handle=null});var stopping=std.atomic.Value(bool).init(false);if(comptime @hasField(W,"stopping"))worker.stopping=&stopping;
    var input=try cuda.HostBuffer.alloc(&d,glm.moe.hybridTransport.wire.request_bytes);defer input.free();@memset(input.bytes,0);var output=try cuda.HostBuffer.alloc(&d,glm.moe.hybridTransport.wire.reply_bytes);defer output.free();
    var h=glm.moe.hybridTransport.wire.Header{.session=1,.sequence=1,.native_round=1,.buffer_epoch=1,.layer=3,.phase=.target,.live_rows=1,.call_rows=1};h.bind();F.cancel_on_record=&stopping;defer F.cancel_on_record=null;
    const got=W.compute(&worker,h,input.bytes,output.bytes,tp.sock.nowNs()+5_000_000_000);const pending=worker.pending;const failed=worker.failed;
    // Fixture cleanup only: synthetic event API is already complete; product must retain the cancelled owner.
    worker.pending=false;worker.failed=null;worker.releaseStorage();
    try t.expectError(error.Cancelled,got);try t.expect(pending and failed!=null);if(failed)|err|try t.expectEqual(error.Cancelled,err);try t.expect(F.native_launches>0 and stopping.load(.acquire));
}
