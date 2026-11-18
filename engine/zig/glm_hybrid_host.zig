//! Recording the actual MoE fill/provider/forward consumers, never a device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const t = std.testing;
test { _=@import("tp").roce; }

const Record = struct {
    var next_ptr: u64=0x100000000;
    fn alloc(p: *u64,n: usize) callconv(.c) c_int {p.*=next_ptr;next_ptr+=std.mem.alignForward(u64,n,4096);return 0;}
    fn free(_: u64) callconv(.c) c_int {return 0;}
    fn zero(_: u64,_: u8,_: usize) callconv(.c) c_int {return 0;}
    var touches: usize=0;
    var remote_calls: usize = 0;
    var remote_layer: u32 = 0;
    var remote_phase: u16 = 0;
    var remote_rows: usize = 0;
    var combine_calls: usize = 0;
    var combine_slots: i64 = 0;
    var combine_fp32: c_int = 0;
    var put_calls: usize = 0;
    var epilogue_world: c_int=0;
    var epilogue_base: u64=0;
    fn arg(comptime T: type, args: ?[*]?*anyopaque, i: usize) T { return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*; }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        if (@intFromPtr(f)==1) {
            combine_calls+=1;
            combine_fp32=arg(c_int,args,1);
            combine_slots=arg(c_int,args,5);
        }
        if (@intFromPtr(f)==2) put_calls+=1;
        if (@intFromPtr(f)==3) {epilogue_world=arg(c_int,args,7);epilogue_base=arg(u64,args,2);}
        if (@intFromPtr(f)==4) {epilogue_world=arg(c_int,args,5);epilogue_base=arg(u64,args,2);}
        return 0;
    }
    fn memset(_: u64,_: c_uint,_: usize,_: cuda.abi.Stream) callconv(.c) c_int { return 0; }
    fn copy(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) c_int { return 0; }
    fn dcopy(_: u64, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) c_int { return 0; }
    fn wait(_: cuda.abi.Stream, _: cuda.abi.Event, _: c_uint) callconv(.c) c_int { return 0; }
    fn fun(d: *cuda.Driver,id: usize) cuda.Function { return .{ .d=d,.handle=@ptrFromInt(id) }; }
};

test "hybrid provider: actual prompt fill treats absent remote EXL3 as FP32 and keeps shared separate" {
    Record.combine_calls=0; Record.put_calls=0;
    var d: cuda.Driver=undefined;
    d.api.cuLaunchKernel=Record.launch;
    d.api.cuMemcpy2DAsync_v2=Record.copy;
    d.api.cuMemcpyDtoDAsync_v2=Record.dcopy;
    d.api.cuStreamWaitEvent=Record.wait;
    var g: glm.glue.Kernels=undefined;
    inline for (@typeInfo(glm.glue.Kernels).@"struct".field_names) |name| @field(g,name)=Record.fun(&d,7);
    g.combine=Record.fun(&d,1);
    var ek: glm.experts.Kernels=undefined;
    var xk: glm.exl3.Kernels=undefined;
    var scratch=std.mem.zeroes(glm.moe.Scratch);
    scratch.sy=0x400000; scratch.ey=0x500000; scratch.wts=0x600000;
    if (comptime @hasField(glm.moe.Scratch,"hybrid")) {
        const r=glm.moe.routed;
        var buffer=r.Buffer{ .capacity=32,.dims=4096,.halves=0x1000000,.epilogue=0x2000000,.weights=0x3000000,.shared_weights=0x4000000 };
        buffer.result=.{ .remote=.{ .session=1,.sequence_first=1,.sequence_last=1,.tiles=1,.buffer_epoch=1,.row_map_digest=@splat(0xAA),.rows=32,.ready_event=.{.d=&d,.handle=null} } };
        scratch.hybrid=&buffer;
        try doFill(&d,&g,&ek,&xk,scratch);
    } else try doFill(&d,&g,&ek,&xk,scratch);
    try t.expectEqual(@as(usize,0),Record.put_calls);
    try t.expectEqual(@as(usize,1),Record.combine_calls);
    try t.expectEqual(@as(i64,1),Record.combine_slots);
    try t.expectEqual(@as(c_int,1),Record.combine_fp32);
}
fn doFill(d: *cuda.Driver,g: *glm.glue.Kernels,ek: *glm.experts.Kernels,xk: *glm.exl3.Kernels,scratch: glm.moe.Scratch) !void {
    const l: glm.moe.Layer=.{ .router=0,.bias=0,.routed=.{ .format=.nvfp4,.up=0,.down=0,.width=2048,.dims=4096,.experts=288,.limit=7 },.count=288,.shared_gu=.{.bf16=0x700000},.shared_down=.{.bf16=0x800000},.shared=512,.top_k=8,.scale=2.5,.norm=true };
    try glm.moe.fill(.{ .set=.{.glue=g},.experts=ek,.exl3=xk,.put_slot=Record.fun(d,2) },.{.d=d,.handle=null},l,scratch,0x900000,4096,3,20,true,0xA00000);
}

test "hybrid provider: actual nonzero piece HC epilogue consumes three packed planes not TP world" {
    var d: cuda.Driver=undefined;d.api.cuLaunchKernel=Record.launch;
    var modules: glm.forward.Modules=undefined;setFunctions(&modules,&d);modules.index=null;modules.tiled=false;modules.packs=null;
    modules.glue.hc_post=Record.fun(&d,3);
    var w: glm.weights.Weights=.{.gpa=t.allocator,.config=.{},.share=try glm.shard.Share.of(&glm.config.Config{},0,1)};
    var layer: glm.weights.Layer=undefined;layer.ffn=.{.moe=undefined};var layers=[_]glm.weights.Layer{layer};w.layers=&layers;
    var b: glm.forward.Buffers=undefined;b.rows=32;b.prefill=true;b.x=0x100000;b.post=0x200000;b.comb=0x300000;b.hidden=0x400000;
    b.moe=std.mem.zeroes(glm.moe.Scratch);
    var buffer=glm.moe.routed.Buffer{.capacity=32,.dims=4096,.halves=0x1000000,.epilogue=0x2000000,.weights=0x3000000,.shared_weights=0x4000000};
    buffer.result=.{.remote=.{.session=1,.sequence_first=1,.sequence_last=1,.tiles=1,.buffer_epoch=1,.row_map_digest=@splat(0xAA),.rows=32,.ready_event=.{.d=&d,.handle=null}}};
    b.moe.hybrid=&buffer;
    var st: glm.forward.State=undefined;
    const rank=glm.forward.Rank{.w=&w,.b=&b,.st=&st};
    const f=glm.forward.Forward{.d=&d,.m=&modules,.stream=.{.d=&d,.handle=null},.ranks=&.{rank},.exchange=.local};
    Record.epilogue_world=0;
    try f.glueRows(rank,0,.ffn,3,20,0x900000);
    try t.expectEqual(@as(c_int,3),Record.epilogue_world);
    try t.expectEqual(@as(u64,0x2000000+3*3*4096*4),Record.epilogue_base);
    try t.expectEqual(@as(usize,1),w.share.world);
}

test "hybrid provider: worker actual native EXL3 caller expands wire8 to native9 before prompt and decode" {
    var d: cuda.Driver=undefined;d.api.cuLaunchKernel=Record.launch;d.api.cuMemcpy2DAsync_v2=Record.copy;
    d.api.cuMemsetD32Async=Record.memset;
    var modules: glm.forward.Modules=undefined;setFunctions(&modules,&d);
    modules.exl3.grouped=.{Record.fun(&d,7),Record.fun(&d,7),Record.fun(&d,7)};
    var x3=std.mem.zeroes(glm.exl3.Experts);x3.shx=true;
    const stream=cuda.Stream{.d=&d,.handle=null};
    for([_]bool{false,true}) |prompt| {
        if(comptime @hasDecl(glm.moe,"hybridExpert")) {
            try glm.moe.hybridExpert.run(&modules.exl3,&modules.experts,stream,x3,.{.picks=0x7000000,.xg=0,.xu=0,.z=0},.{.members=1,.items=2,.counts=3,.rank=4,.hist=5},.{.native_picks=0x1000000,.native_y=0x2000000},0x3000000,4096,0x4000000,17,288,4096,1024,7,prompt,0x5000000);
        } else try glm.exl3.run(&modules.exl3,&modules.experts,stream,x3,.{.picks=0x7000000,.xg=0,.xu=0,.z=0},.{.members=1,.items=2,.counts=3,.rank=4,.hist=5},0x3000000,4096,0x4000000,8,17,288,4096,1024,7,prompt,0x5000000,0);
    }
}

test "hybrid provider: actual attention Buffers and twin own disjoint durable planes without routed scratch" {
    var d: cuda.Driver=undefined;d.api.cuMemAlloc_v2=Record.alloc;d.api.cuMemFree_v2=Record.free;d.api.cuMemsetD8_v2=Record.zero;
    var w: glm.weights.Weights=.{.gpa=t.allocator,.config=.{.exl3=true},.share=try glm.shard.Share.of(&glm.config.Config{},0,1),.role=.attention};
    var b=try glm.forward.Buffers.init(&d,&w,129,true,1024);defer b.deinit();
    try t.expect(b.moe.hybrid!=null);
    try t.expectEqual(@as(u64,0),b.moe.ey);
    try t.expectEqual(@as(u64,0),b.moe.eact);
    try t.expectEqual(@as(u64,0),b.moe.x3.picks);
    var twin=try glm.forward.Buffers.initTwin(&d,&w,&b,1024);defer twin.deinit();
    try t.expect(twin.moe.hybrid!=null);
    try t.expect(twin.moe.hybrid.?.halves!=b.moe.hybrid.?.halves);
    try t.expect(twin.moe.hybrid.?.epilogue!=b.moe.hybrid.?.epilogue);
    try t.expect(b.moe.hybrid.?.epilogue>=b.moe.hybrid.?.halves+2*129*4096*4);
}

const Framer=if(@hasDecl(glm.moe,"hybridTransport")) struct {
    const tr=glm.moe.hybridTransport;const wire=tr.wire;
    var clock: u64=1;var posts: usize=0;var announced: bool=false;
    fn announce(_: *anyopaque,_: wire.Header,_: u64) !void {announced=true;}
    fn now(_: *anyopaque) u64 {return clock;}
    fn check(_: *anyopaque) !void {}
    fn exchange(_: *anyopaque,send: []const u8,recv: []u8,until: u64) !void {
        try t.expect(send.len<=1048576);try t.expect(clock<until);clock+=1;posts+=1;
        var h=try wire.Header.decode(send[0..256]);
        @memset(recv,0);
        if(h.kind==.request) {
            h.kind=.idle;h.fragment_payload_bytes=0;
            if(h.fragment_index==0 and !announced) {h.native_round=0;h.buffer_epoch=0;h.layer=0;h.phase=.target;h.live_rows=128;h.execution_mode=.decode;h.tile_row_lo=0;h.call_rows=128;h.row_map_sha256=@splat(0);h.bind();}
        } else {
            try t.expectEqual(wire.Kind.idle,h.kind);h.kind=.reply;
            h.fragment_payload_bytes=@intCast(@min(wire.frame_payload,h.total_payload_bytes-h.payload_offset));
            for(recv[256..][0..h.fragment_payload_bytes],0..) |*b,i| b.*=@intCast((h.payload_offset+i)%251);
        }
        h.encode(recv[0..256]);
    }
    fn run() !void {
        const input=try t.allocator.alloc(u8,wire.request_bytes);defer t.allocator.free(input);@memset(input,0x5A);
        const output=try t.allocator.alloc(u8,wire.reply_bytes);defer t.allocator.free(output);
        const tx=try t.allocator.alloc(u8,wire.post_bytes);defer t.allocator.free(tx);
        const rx=try t.allocator.alloc(u8,wire.post_bytes);defer t.allocator.free(rx);
        var ctx: u8=0;clock=1;posts=0;
        var h=wire.Header{.session=123,.sequence=1,.native_round=4,.buffer_epoch=5,.layer=3,.phase=.target,.half=0,.live_rows=128,.row_map_sha256=@splat(0x11),.model_contract_sha256=@splat(0x22),.settings_sha256=@splat(0x33),.execution_mode=.prompt,.tile_row_lo=0,.call_rows=128};h.bind();
        var io=tr.FrameIO{.ctx=&ctx,.now=now,.check=check,.exchange=exchange};announced=false;
        if(comptime @hasField(tr.FrameIO,"announce")) io.announce=announce;
        try tr.transfer(io,h,input,output,tx,rx,100);
        try t.expectEqual(@as(usize,19),posts);
        for(output,0..) |b,i| try t.expectEqual(@as(u8,@intCast(i%251)),b);
    }
} else struct {};

test "hybrid transport: real framing moves full routed tile with native one MiB posts" {
    if(comptime @hasDecl(glm.moe,"hybridTransport")) try Framer.run() else {
        var r: @import("tp").roce.Roce=undefined;r.aborted=.init(false);r.slot_bytes=1048576;r.chunk_abi=false;
        try r.exchangeBytes(1,2,128*8*4096*4,null);
    }
}

fn setFunctions(p: anytype,d: *cuda.Driver) void {
    inline for (@typeInfo(@TypeOf(p.*)).@"struct".field_names) |name| {
        const T=@TypeOf(@field(p.*,name));
        if (T==cuda.Function) @field(p.*,name)=Record.fun(d,7) else if (@typeInfo(T)==.@"struct") setFunctions(&@field(p.*,name),d);
    }
}
const Remote = if (@hasDecl(glm.moe.routed,"Call")) struct {
    fn run(ctx: *anyopaque,call: glm.moe.routed.Call,_: glm.moe.routed.HalfDest) !glm.moe.routed.ReadyParts {
        const d: *cuda.Driver=@ptrCast(@alignCast(ctx));
        Record.remote_calls+=1;Record.remote_layer=call.layer;Record.remote_phase=@intFromEnum(call.phase);Record.remote_rows=call.rows;
        return .{.session=1,.sequence_first=1,.sequence_last=1,.tiles=1,.buffer_epoch=call.buffer_epoch,.row_map_digest=call.row_map_digest,.rows=call.rows,.ready_event=.{.d=d,.handle=null}};
    }
    fn touch(_: *anyopaque,_: cuda.Stream) !void {Record.touches+=1;}
    fn check(_: *anyopaque) !void {}
    fn settle(_: *anyopaque,_: u64) !glm.moe.routed.Quiescence { return .quiet; }
} else struct {};

test "hybrid provider: actual Forward ffn dispatches target and MTP after router with row identity" {
    var d: cuda.Driver=undefined;
    d.api.cuLaunchKernel=Record.launch;d.api.cuMemcpy2DAsync_v2=Record.copy;
    var modules: glm.forward.Modules=undefined;setFunctions(&modules,&d);modules.index=null;modules.tiled=false;modules.packs=null;
    var w: glm.weights.Weights=.{.gpa=t.allocator,.config=.{.exl3=true},.share=try glm.shard.Share.of(&glm.config.Config{},0,1)};
    var b: glm.forward.Buffers=undefined;b.rows=32;b.prefill=true;b.normed=0x900000;b.moe=std.mem.zeroes(glm.moe.Scratch);
    b.moe.pick=0x100000;b.moe.wts=0x200000;b.moe.ey=0x300000;
    var st: glm.forward.State=undefined;st.pos=1234;st.mtp_len=1230;st.arena.buf.ptr=0x100000;
    var buffer=glm.moe.routed.Buffer{.capacity=32,.dims=4096,.halves=0x1000000,.epilogue=0x2000000,.weights=0x3000000,.shared_weights=0x4000000};
    if (comptime @hasField(glm.moe.routed.Buffer,"provider")) buffer.provider=.{.ctx=&d,.run=Remote.run,.check=Remote.check,.settle=Remote.settle};
    b.moe.hybrid=&buffer;
    const rank=glm.forward.Rank{.w=&w,.b=&b,.st=&st};
    const f=glm.forward.Forward{.d=&d,.m=&modules,.stream=.{.d=&d,.handle=null},.ranks=&.{rank},.exchange=.local};
    var l: glm.weights.Layer=undefined;
    var x3=std.mem.zeroes(glm.exl3.Experts);x3.shx=true;
    l.ffn=.{.moe=.{.router=0x1000,.bias=0x2000,.routed=.{.format=.nvfp4,.up=0,.down=0,.width=1024,.dims=4096,.experts=288,.limit=7},.exl3=x3,.count=288,.shared_gu=.{.bf16=0},.shared_down=.{.bf16=0},.shared=512,.top_k=8,.scale=2.5,.norm=true}};
    Record.remote_calls=0;l.index=4;
    try f.ffn(rank,l,17,null);
    try t.expectEqual(@as(usize,1),Record.remote_calls);
    try t.expectEqual(@as(u32,4),Record.remote_layer);
    try t.expectEqual(@as(u8,0),Record.remote_phase);
    try t.expectEqual(@as(usize,17),Record.remote_rows);
    l.index=w.config.layers;
    try f.ffn(rank,l,1,null);
    try t.expectEqual(@as(usize,2),Record.remote_calls);
    try t.expectEqual(@as(u8,1),Record.remote_phase);
}

test "hybrid engine: selected attention role refuses TP before checkpoint or device access" {
    const cfg=glm.moe.hybridConfig.Config{.revision=3,.role=.attention,.session="00000000000000000000000000000001",.model_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="fixture",.setup_ms=30000,.transaction_ms=120000,.close_ms=5000,.served_context=262144,.resident_users=8,.mtp=false,.settings=.{}};
    var opts=glm.engine.Options{.capacity=64,.mtp=false,.settings=.{}};
    if(comptime @hasField(glm.engine.Options,"hybrid")) opts.hybrid=&cfg;
    var d: cuda.Driver=undefined;var ctx: cuda.Context=undefined;var lib: cuda.nccl.Library=undefined;
    try t.expectError(error.BadHybridRole,glm.engine.Engine.init(t.allocator,t.io,&d,&ctx,"/unreachable/checkpoint",&lib,null,0,2,opts));
}

test "hybrid engine: actual provider installation covers prompt verify MTP shared and twin" {
    const r=glm.moe.routed;var hs: [5]r.Buffer=undefined;
    for(&hs) |*h| h.*=.{.capacity=32,.dims=4096,.halves=1,.epilogue=2,.weights=3,.shared_weights=4};
    var d: cuda.Driver=undefined;var e: glm.engine.Engine=undefined;e.w.role=.attention;
    e.bp.moe.hybrid=&hs[0];e.bd.moe.hybrid=&hs[1];e.bm.moe.hybrid=&hs[2];
    var shared: glm.forward.Buffers=undefined;shared.moe.hybrid=&hs[3];e.bs=shared;
    var twin: glm.prompt_pair.Pair=undefined;twin.bp.moe.hybrid=&hs[4];e.pair=twin;
    var provider=r.Provider{.ctx=&d,.run=Remote.run,.check=Remote.check,.settle=Remote.settle};
    Record.touches=0;if(comptime @hasField(r.Provider,"touch")) provider.touch=Remote.touch;
    e.stream=.{.d=&d,.handle=null};e.pieces.side=.{.d=&d,.handle=@ptrFromInt(1)};e.pair.?.pieces.side=e.pieces.side;e.l2pf=null;
    if(comptime @hasDecl(glm.engine.Engine,"installRouted")) try e.installRouted(provider);
    for(&hs) |*h| {try t.expect(h.provider!=null);try t.expectEqual(@intFromPtr(&d),@intFromPtr(h.provider.?.ctx));}
    try t.expectEqual(@as(usize,3),Record.touches);
}

const FailedProvider=struct {fn check(_: *anyopaque) !void {return error.RemoteFailed;}};
test "hybrid engine: actual gather acceptance refuses a failed routed session" {
    var d: cuda.Driver=undefined;var e: glm.engine.Engine=undefined;e.group=null;
    e.routed=.{.ctx=&d,.run=Remote.run,.check=FailedProvider.check,.settle=Remote.settle};
    try t.expectError(error.RemoteFailed,e.checkGathers());
}

const DenyLaunch=struct {
    var calls: usize=0;
    fn launch(_: cuda.abi.Function,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: cuda.abi.Stream,_: ?[*]?*anyopaque,_: ?[*]?*anyopaque) callconv(.c) c_int {calls+=1;return cuda.abi.error_not_ready;}
};
test "hybrid engine: actual draw refuses sticky remote failure before first CUDA launch" {
    var d: cuda.Driver=undefined;d.api.cuLaunchKernel=DenyLaunch.launch;DenyLaunch.calls=0;
    var e: glm.engine.Engine=undefined;e.group=null;e.routed=.{.ctx=&d,.run=Remote.run,.check=FailedProvider.check,.settle=Remote.settle};
    e.structured=false;e.profiling_prompt=null;e.stream=.{.d=&d,.handle=null};e.m.glue.top1=Record.fun(&d,7);e.pairs.ptr=0x100000;
    e.share=try glm.shard.Share.of(&glm.config.Config{},0,1);
    var q: @typeInfo(@TypeOf(e.cur)).pointer.child=undefined;q.sampling=null;e.cur=&q;
    var out: [1]u32=undefined;try t.expectError(error.RemoteFailed,e.draw(0x200000,1,0,0,&out));try t.expectEqual(@as(usize,0),DenyLaunch.calls);
}

test "hybrid provider: actual worker refuses out-of-range wire pick before native execution" {
    const tr=glm.moe.hybridTransport;const a=t.allocator;
    const input=try a.alloc(u8,tr.wire.request_bytes);defer a.free(input);@memset(input,0);
    const output=try a.alloc(u8,tr.wire.reply_bytes);defer a.free(output);
    std.mem.writeInt(i32,input[128*4096*2..][0..4],288,.little);
    var d: cuda.Driver=undefined;d.api.cuLaunchKernel=Record.launch;d.api.cuMemcpy2DAsync_v2=Record.copy;d.api.cuMemsetD32Async=Record.memset;
    var modules: glm.forward.Modules=undefined;setFunctions(&modules,&d);modules.exl3.grouped=.{Record.fun(&d,7),Record.fun(&d,7),Record.fun(&d,7)};
    var h=tr.wire.Header{.session=1,.sequence=1,.native_round=1,.buffer_epoch=1,.layer=3,.phase=.target,.live_rows=1,.call_rows=1};h.bind();
    if(comptime @hasDecl(glm.moe.hybridExpert,"Worker")) {
        var worker: glm.moe.hybridExpert.Worker=undefined;
        var w: @typeInfo(@TypeOf(worker.weights)).pointer.child=undefined;w.config=.{.exl3=true};w.half=0;
        worker.weights=&w;worker.pending=false;worker.failed=null;
        try t.expectError(error.BadHybridPick,glm.moe.hybridExpert.Worker.compute(&worker,h,input,output,std.math.maxInt(u64)));
    } else {
        var x3=std.mem.zeroes(glm.exl3.Experts);x3.shx=true;
        try t.expectError(error.BadHybridPick,glm.moe.hybridExpert.runRouter(&modules.exl3,modules.experts.router,.{.d=&d,.handle=null},x3,.{.picks=0x7000000,.xg=0,.xu=0,.z=0},.{.members=1,.items=2,.counts=3,.rank=4,.hist=5},.{.native_picks=0x1000000,.native_y=0x2000000},0x3000000,4096,@intFromPtr(input.ptr)+128*4096*2,1,288,4096,1024,7,false,0x5000000));
    }
}

const WholeFixture=struct {
    fn getdev(p: *cuda.abi.Device) callconv(.c) c_int {p.*=0;return 0;}
    fn attribute(p: *c_int,_: cuda.abi.DeviceAttribute,_: cuda.abi.Device) callconv(.c) c_int {p.*=0;return 0;}
    fn h2d(_: u64,_: ?*const anyopaque,_: usize,_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
    fn zero(_: u64,_: u8,_: usize,_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
    fn d2h(p: ?*anyopaque,_: u64,n: usize) callconv(.c) c_int {@memset(@as([*]u8,@ptrCast(p.?))[0..n],0);return 0;}
    fn host(p: *?*anyopaque,n: usize,_: c_uint) callconv(.c) c_int {p.*=std.c.malloc(n);return if(p.*==null) 2 else 0;}
    fn hfree(p: ?*anyopaque) callconv(.c) c_int {std.c.free(p);return 0;}
    fn event(p: *cuda.abi.Event,_: c_uint) callconv(.c) c_int {p.*=@ptrFromInt(1);return 0;}
    fn evrecord(_: cuda.abi.Event,_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
    fn evop(_: cuda.abi.Event) callconv(.c) c_int {return 0;}
    fn allg(_: u64,_: u64,_: usize,_: cuda.nccl.DataType,_: cuda.nccl.Comm,_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
    fn init(e: *glm.engine.Engine,d: *cuda.Driver,q: anytype,h: *glm.moe.routed.Buffer,layers: []glm.weights.Layer,cur: []u1,lib: *cuda.nccl.Library) !void {
        d.api.cuLaunchKernel=Record.launch;d.api.cuMemcpy2DAsync_v2=Record.copy;d.api.cuMemcpyDtoDAsync_v2=Record.dcopy;d.api.cuStreamWaitEvent=Record.wait;
        d.api.cuMemcpyHtoDAsync_v2=h2d;d.api.cuStreamSynchronize=sync;d.api.cuMemcpyDtoH_v2=d2h;d.api.cuMemsetD32Async=Record.memset;
        d.api.cuMemHostAlloc=host;d.api.cuMemFreeHost=hfree;d.api.cuEventCreate=event;d.api.cuEventRecord=evrecord;d.api.cuEventQuery=evop;d.api.cuEventDestroy_v2=evop;d.api.cuEventSynchronize=evop;
        d.api.cuCtxGetDevice=getdev;d.api.cuDeviceGetAttribute=attribute;
        d.api.cuMemsetD8Async=zero;
        lib.api.ncclAllGather=allg;
        @memset(std.mem.asBytes(e),0);e.d=d;e.stream=.{.d=d,.handle=null};e.lib=lib;
        e.c=.{.exl3=true,.run=1};e.w=.{.gpa=t.allocator,.config=e.c,.share=try glm.shard.Share.of(&e.c,0,1),.role=.attention};e.w.layers=layers;e.share=e.w.share;
        e.cur=q;@memset(std.mem.asBytes(q),0);q.st.arena.buf.d=d;q.st.arena.buf.ptr=0x100000;q.st.capacity=1024;q.st.ring_rows=1024;q.st.cur=cur;q.st.pending=0;
        @memset(std.mem.asBytes(&e.bd),0);e.bd.rows=32;e.bd.gathered=0x5000000;e.bd.moe.hybrid=h;e.bd.moe.pick=0x110000;e.bd.moe.wts=0x120000;
        e.bd.normed=0x130000;e.bd.hidden=0x140000;e.bd.fnormed=0x150000;
        setFunctions(&e.m,d);e.m.index=null;e.m.packs=null;e.m.tiled=false;e.m.glue.hc_post=Record.fun(d,3);
        h.*=.{.capacity=32,.dims=4096,.halves=0x1000000,.epilogue=0x2000000,.weights=0x3000000,.shared_weights=0x4000000,.provider=.{.ctx=d,.run=Remote.run,.check=Remote.check,.settle=Remote.settle}};
        layers[0]=.{.index=3,.attn_hc=.{.fn_w=0,.base=0,.scale=0},.ffn_hc=.{.fn_w=0,.base=0,.scale=0},.in_norm=.{.bf16=0},.post_norm=.{.bf16=0},.mixer=.{.kda=.{.proj=.{.bf16=0},.fb=0,.gb=0,.o=.{.bf16=0},.conv_w=0,.a_log=0,.dt_bias=0,.norm_w=0,.eps=1e-5,.lower= -5,.heads=64,.dims=4096}},.ffn=undefined};
        layers[0].ffn=.{.moe=.{.router=0,.bias=0,.routed=.{.format=.nvfp4,.up=0,.down=0,.width=2048,.dims=4096,.experts=288,.limit=7},.count=288,.shared_gu=.{.bf16=0},.shared_down=.{.bf16=0},.shared=2048,.top_k=8,.scale=2.5,.norm=true}};
    }
};
test "hybrid provider: actual Engine verify whole forward consumes three remote planes" {
    var d: cuda.Driver=undefined;var lib: cuda.nccl.Library=undefined;var e: glm.engine.Engine=undefined;
    var q: @typeInfo(@TypeOf(e.cur)).pointer.child=undefined;var h: glm.moe.routed.Buffer=undefined;var layers: [1]glm.weights.Layer=undefined;var cur=[_]u1{0};
    try WholeFixture.init(&e,&d,&q,&h,&layers,&cur,&lib);
    var backend=@import("tp").group.Backend{.d=&d,.lib=&lib,.comm=null,.rank=0,.world=1};e.collective=backend.collective();
    Record.epilogue_world=0;Record.epilogue_base=0;
    var out: [1]u32=undefined;try e.verify(&.{1},&out,0);
    try t.expectEqual(@as(c_int,3),Record.epilogue_world);try t.expectEqual(h.epilogue,Record.epilogue_base);
}

test "hybrid provider: actual Engine MTP draft consumes remote FFN residual planes" {
    var d: cuda.Driver=undefined;var lib: cuda.nccl.Library=undefined;var e: glm.engine.Engine=undefined;
    var q: @typeInfo(@TypeOf(e.cur)).pointer.child=undefined;var h: glm.moe.routed.Buffer=undefined;var layers: [1]glm.weights.Layer=undefined;var cur=[_]u1{0};
    try WholeFixture.init(&e,&d,&q,&h,&layers,&cur,&lib);
    var backend=@import("tp").group.Backend{.d=&d,.lib=&lib,.comm=null,.rank=0,.world=1};e.collective=backend.collective();
    e.mtp=true;e.bm=e.bd;e.bm.part_rows=1;e.bm.nch=1;e.m.glue.residual_add=Record.fun(&d,4);
    var ml=layers[0];ml.index=e.c.layers;
    ml.mixer=.{.mla=.{.proj=.{.bf16=0},.q_norm=0,.kv_norm=0,.q_b=.{.bf16=0},.wk=0,.wv=0,.o=.{.bf16=0},.index=.{.kw=.{.bf16=0},.gate=0,.ln_w=0,.ln_b=0,.ape=0,.qb=.{.bf16=0},.heads=32,.d=128},.heads=64,.qk=256,.dv=256,.lw=512,.q_lora=1536,.dims=4096,.eps=1e-5,.dense_limit=2048}};
    e.w.mtp=.{.enorm=0,.hnorm=0,.eh=0,.norm=0,.layer=ml,.draft=.{.w=0,.s=0,.b=0,.n=154880,.k=4096,.npad=154880}};
    Record.epilogue_world=0;Record.epilogue_base=0;
    var out: [1]u32=undefined;try e.draft(false,&.{1},1,&out,null);
    try t.expectEqual(@as(c_int,3),Record.epilogue_world);try t.expectEqual(h.epilogue,Record.epilogue_base);
}

test "hybrid engine: begin refuses sticky routed failure before resetting request state" {
    var d: cuda.Driver=undefined;var lib: cuda.nccl.Library=undefined;var e: glm.engine.Engine=undefined;
    var q: @typeInfo(@TypeOf(e.cur)).pointer.child=undefined;var h: glm.moe.routed.Buffer=undefined;var layers: [1]glm.weights.Layer=undefined;var cur=[_]u1{1};
    try WholeFixture.init(&e,&d,&q,&h,&layers,&cur,&lib);e.routed=.{.ctx=&d,.run=Remote.run,.check=FailedProvider.check,.settle=Remote.settle};
    var rings=[_]u64{0x10000,0x20000};q.st.rings=&rings;
    q.st.pos=17;q.st.mtp_len=12;q.in_head=2;
    try t.expectError(error.RemoteFailed,e.begin(null));
    try t.expectEqual(@as(usize,17),q.st.pos);try t.expectEqual(@as(usize,12),q.st.mtp_len);try t.expectEqual(@as(usize,2),q.in_head);
}

test "hybrid engine: keep refuses sticky failure before committing pending rows" {
    var d: cuda.Driver=undefined;var lib: cuda.nccl.Library=undefined;var e: glm.engine.Engine=undefined;
    var q: @typeInfo(@TypeOf(e.cur)).pointer.child=undefined;var h: glm.moe.routed.Buffer=undefined;var layers: [1]glm.weights.Layer=undefined;var cur=[_]u1{0};
    try WholeFixture.init(&e,&d,&q,&h,&layers,&cur,&lib);e.routed=.{.ctx=&d,.run=Remote.run,.check=FailedProvider.check,.settle=Remote.settle};h.provider=e.routed;
    q.st.pos=17;q.st.pending=3;
    try t.expectError(error.RemoteFailed,e.keep(2));
    try t.expectEqual(@as(usize,17),q.st.pos);try t.expectEqual(@as(usize,3),q.st.pending);
}

test "hybrid engine: zero-depth MTP repair refuses sticky failure before rewind" {
    var d: cuda.Driver=undefined;var lib: cuda.nccl.Library=undefined;var e: glm.engine.Engine=undefined;
    var q: @typeInfo(@TypeOf(e.cur)).pointer.child=undefined;var h: glm.moe.routed.Buffer=undefined;var layers: [1]glm.weights.Layer=undefined;var cur=[_]u1{0};
    try WholeFixture.init(&e,&d,&q,&h,&layers,&cur,&lib);e.routed=.{.ctx=&d,.run=Remote.run,.check=FailedProvider.check,.settle=Remote.settle};e.mtp=true;
    q.st.mtp_len=12;q.in_head=2;
    try t.expectError(error.RemoteFailed,e.draft(false,&.{},0,&.{},null));
    try t.expectEqual(@as(usize,12),q.st.mtp_len);try t.expectEqual(@as(usize,2),q.in_head);
}

test "hybrid engine: cancellation gate refuses failed session before prefix or cancel callbacks" {
    const C=struct {var calls: usize=0;fn cancel(_: ?*anyopaque) bool {calls+=1;return true;}fn reduce(_: u64,_: u64,_: usize,_: cuda.nccl.DataType,_: cuda.nccl.RedOp,_: cuda.nccl.Comm,_: cuda.abi.Stream) callconv(.c) c_int {return 0;}};
    var d: cuda.Driver=undefined;var lib: cuda.nccl.Library=undefined;var e: glm.engine.Engine=undefined;
    var q: @typeInfo(@TypeOf(e.cur)).pointer.child=undefined;var h: glm.moe.routed.Buffer=undefined;var layers: [1]glm.weights.Layer=undefined;var cur=[_]u1{0};
    try WholeFixture.init(&e,&d,&q,&h,&layers,&cur,&lib);e.routed=.{.ctx=&d,.run=Remote.run,.check=FailedProvider.check,.settle=Remote.settle};
    e.shared_buf=.{.d=&d,.ptr=0x10000,.len=8};lib.api.ncclAllReduce=C.reduce;C.calls=0;
    try t.expectError(error.RemoteFailed,e.gate(.{.ptr=null,.check=C.cancel}));try t.expectEqual(@as(usize,0),C.calls);
}

test "hybrid engine: verify refuses failed session before changing pending state" {
    var d: cuda.Driver=undefined;var lib: cuda.nccl.Library=undefined;var e: glm.engine.Engine=undefined;
    var q: @typeInfo(@TypeOf(e.cur)).pointer.child=undefined;var h: glm.moe.routed.Buffer=undefined;var layers: [1]glm.weights.Layer=undefined;var cur=[_]u1{0};
    try WholeFixture.init(&e,&d,&q,&h,&layers,&cur,&lib);e.routed=.{.ctx=&d,.run=Remote.run,.check=FailedProvider.check,.settle=Remote.settle};h.provider=e.routed;
    var backend=@import("tp").group.Backend{.d=&d,.lib=&lib,.comm=null,.rank=0,.world=1};e.collective=backend.collective();q.st.pending=3;
    var out: [1]u32=undefined;try t.expectError(error.RemoteFailed,e.verify(&.{1},&out,0));try t.expectEqual(@as(usize,3),q.st.pending);
}

test "hybrid engine: shared verify refuses failed session before binding or mutating either request" {
    var d: cuda.Driver=undefined;var lib: cuda.nccl.Library=undefined;var e: glm.engine.Engine=undefined;
    const Seq=@typeInfo(@TypeOf(e.cur)).pointer.child;var q: Seq=undefined;var h: glm.moe.routed.Buffer=undefined;var layers: [1]glm.weights.Layer=undefined;var cur=[_]u1{0};
    try WholeFixture.init(&e,&d,&q,&h,&layers,&cur,&lib);e.routed=.{.ctx=&d,.run=Remote.run,.check=FailedProvider.check,.settle=Remote.settle};h.provider=e.routed;
    var backend=@import("tp").group.Backend{.d=&d,.lib=&lib,.comm=null,.rank=0,.world=1};e.collective=backend.collective();
    e.bs=e.bd;var other=q;var slots=[_]?*Seq{&q,&other};e.seqs=.empty;e.seqs.items=&slots;e.seqs.capacity=2;q.st.pending=3;other.st.pending=4;e.slot=7;
    var out: [2]u32=undefined;try t.expectError(error.RemoteFailed,e.verifyShared(&.{.{.slot=0,.ids=&.{1},.out=out[0..1]},.{.slot=1,.ids=&.{2},.out=out[1..2]}}));
    try t.expectEqual(@as(usize,3),q.st.pending);try t.expectEqual(@as(usize,4),other.st.pending);try t.expectEqual(@as(u32,7),e.slot);
}

test "hybrid engine: actual native open consumes selected role config before rank device or model" {
    var tmp=t.tmpDir(.{});defer tmp.cleanup();try tmp.dir.writeFile(t.io,.{.sub_path="role.json",.data="{\"role\":\"expert0\"}"});
    const path=try tmp.dir.realPathFileAlloc(t.io,"role.json",t.allocator);defer t.allocator.free(path);const z=try t.allocator.dupeSentinel(u8,path,0);defer t.allocator.free(z);
    const env=@import("sys").env;_=env.setenv("TF_GLM_HYBRID_CONFIG",z,1);defer _=env.unsetenv("TF_GLM_HYBRID_CONFIG");
    var context: cuda.Context=undefined;
    try t.expectError(error.MissingField,glm.native.open(t.allocator,t.io,&context,"/unreachable/model",null,.{.context=262144,.drafts=true}));
}
