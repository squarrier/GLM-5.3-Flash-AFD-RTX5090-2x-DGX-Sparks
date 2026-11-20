// SPDX-License-Identifier: Apache-2.0
//! Scott and AI assistance I3: actual native memory consumers; CUDA API calls are recording-only CPU fixtures.
const std=@import("std");const t=std.testing;const glm=@import("glm");const cuda=@import("cuda");const m=glm.hybrid_memory;
test { _=glm.native; }
test { _=@import("glm_hybrid_preload_host.zig"); _=@import("glm_hybrid_expert_memory_host.zig"); _=@import("glm_hybrid_packed_memory_host.zig"); _=@import("glm_hybrid_ledger_host.zig"); _=@import("glm_hybrid_owned_gather_host.zig"); }
const F=struct {
    var next:u64=0x100000;var live:usize=0;var device:u64=0;var freed:u64=0;
    fn alloc(p:*u64,n:usize) callconv(.c) c_int {p.*=next;next+=std.mem.alignForward(u64,n+256,256);live+=1;device+=n;return 0;}
    fn free(_:u64) callconv(.c) c_int {live-=1;return 0;}
    fn fill(_:u64,_:u8,_:usize) callconv(.c) c_int {return 0;}
    fn driver() cuda.Driver {next=0x100000;live=0;device=0;freed=0;var d:cuda.Driver=undefined;d.api.cuMemsetD8_v2=fill;d.api.cuMemAlloc_v2=alloc;d.api.cuMemFree_v2=free;return d;}
};
test "I3 native: durable planes use actual packed backing not independently rounded views" {
    var d=F.driver();var c=glm.config.Config{};c.exl3=true;
    const w=glm.weights.Weights{.role=.attention,.gpa=t.allocator,.config=c,.share=try glm.shard.Share.of(&c,0,1)};
    var b=try glm.forward.Buffers.init(&d,&w,32,false,256);defer b.deinit();
    const actual=b.hybrid_arena.?.len;const planned=try m.scratchBytes(32,false);
    try t.expectEqual(actual,planned.total);
    try t.expectEqual(@as(u64,32*4096*5*4+32*9*4),planned.total);
    var twin=try glm.forward.Buffers.initTwin(&d,&w,&b,256);defer twin.deinit();
    try t.expect(twin.hybrid_arena.?.ptr!=b.hybrid_arena.?.ptr);
    try t.expectEqual(b.hybrid_arena.?.len,twin.hybrid_arena.?.len);
}
test "I3 native buffers classify workspace and durable twin planes without charging views twice" {
    var l=glm.hybrid_ledger.Ledger.init(.attention,916);try l.install();defer l.uninstall();var d=F.driver();var c=glm.config.Config{};c.exl3=true;
    const w=glm.weights.Weights{.role=.attention,.gpa=t.allocator,.config=c,.share=try glm.shard.Share.of(&c,0,1)};
    var b=try glm.forward.Buffers.init(&d,&w,32,false,256);var twin=try glm.forward.Buffers.initTwin(&d,&w,&b,256);
    const status=l.status();const actual_planes=b.hybrid_arena.?.len+twin.hybrid_arena.?.len;const actual_workspace=b.arena.buf.len+twin.arena.buf.len;
    twin.deinit();b.deinit();l.finish(true,false);
    try t.expectEqual(actual_planes,status.owners[@intFromEnum(m.Owner.call_planes)].device);try t.expectEqual(actual_workspace,status.owners[@intFromEnum(m.Owner.workspace)].device);
    try t.expectEqual(@as(u64,0),status.unclassified_buffers);try t.expect(l.status().clean);try t.expectEqual(@as(usize,0),F.live);
}
test "I3 native state arenas retain sequence ownership for flat and paged consumers" {
    var l=glm.hybrid_ledger.Ledger.init(.attention,917);try l.install();defer l.uninstall();var d=F.driver();var c=glm.config.Config{};c.exl3=true;
    const w=glm.weights.Weights{.role=.attention,.gpa=t.allocator,.config=c,.share=try glm.shard.Share.of(&c,0,1)};
    var a=try glm.forward.State.init(t.allocator,&d,&w,256);var b=try glm.forward.State.initPaged(t.allocator,&d,&w,256,.{.kv=0,.pages=1,.page_bytes=4096});
    const expected=a.arena.buf.len+b.arena.buf.len;const status=l.status();a.deinit();b.deinit();l.finish(true,false);
    try t.expectEqual(expected,status.owners[@intFromEnum(m.Owner.sequence_state)].device);try t.expect(l.status().clean);try t.expectEqual(@as(usize,0),F.live);
}
test "I3 native Engine snapshot preparation prices device and pinned cache owners once" {
    const C=glm.hybrid_cpu_fixture.F;const env=struct{extern "c" fn setenv([*:0]const u8,[*:0]const u8,c_int)c_int;extern "c" fn unsetenv([*:0]const u8)c_int;};
    defer _=env.unsetenv("TF_GLM_HOST_CACHE");
    for([_]bool{false,true})|host|{
        try t.expectEqual(@as(c_int,0),env.setenv("TF_GLM_HOST_CACHE",if(host)"1"else"0",1));
        var l=glm.hybrid_ledger.Ledger.init(.attention,918);try l.install();defer l.uninstall();var d:cuda.Driver=undefined;const e=try C.engine(&d);
        const budget=4*glm.snapshots.pages.page;try e.prepareSnapshots(budget);const status=l.status();e.deinit();l.finish(true,false);
        const own=status.owners[@intFromEnum(m.Owner.snapshots)];try t.expectEqual(if(host)@as(u64,0)else budget,own.device);try t.expectEqual(if(host)budget else @as(u64,0),own.pinned);try t.expect(l.status().clean);
    }
}

test "I3 actual Pages owns full latent backing and device pooled keys without double spending" {
    const C=glm.hybrid_cpu_fixture.F;const env=struct{extern "c" fn setenv([*:0]const u8,[*:0]const u8,c_int)c_int;extern "c" fn unsetenv([*:0]const u8)c_int;
        fn clear(p:u64,v:u8,n:usize)callconv(.c)c_int{@memset(@as([*]u8,@ptrFromInt(p))[0..n],v);return 0;}
    };
    defer _=env.unsetenv("TF_GLM_KV_HOST");defer _=env.unsetenv("TF_GLM_KV_HOST_SHARED");
    try t.expectEqual(@as(c_int,0),env.setenv("TF_GLM_KV_HOST_SHARED","0",1));
    for([_]bool{false,true})|host|{
        try t.expectEqual(@as(c_int,0),env.setenv("TF_GLM_KV_HOST",if(host)"1"else"0",1));
        var l=glm.hybrid_ledger.Ledger.init(.attention,919);try l.install();defer l.uninstall();var d:cuda.Driver=undefined;const e=try C.engine(&d);d.api.cuMemsetD8_v2=env.clear;
        try glm.pages.ownSequence(e);const layout=try glm.pages.layout(t.allocator,e,0);const p=try glm.pages.Pages.init(t.allocator,e,layout,2,3,true);e.pages=p;
        const latent=if(host)p.host.?.bytes.len else p.arena.?.len;const keys=if(host)p.pk_arena.?.len else 0;const status=l.status();
        e.deinit();l.finish(true,false);const pool=status.owners[@intFromEnum(m.Owner.host_pages)];const pk=status.owners[@intFromEnum(m.Owner.pooled_keys)];
        try t.expectEqual(latent,if(host)pool.pinned else pool.device);try t.expectEqual(keys,pk.device);try t.expectEqual(@as(u64,0),status.unclassified_buffers);try t.expect(l.status().clean);try t.expectEqual(@as(usize,0),C.live);
    }
}

test "I3 native transport frames and coordinator owning buffers exclude borrowed views" {
    const C=glm.hybrid_cpu_fixture.F;const T=glm.moe.hybridTransport;
    const X=struct{fn event(p:*cuda.abi.Event,_:c_uint)callconv(.c)c_int{p.*=null;return 0;}fn drop(_:cuda.abi.Event)callconv(.c)c_int{return 0;}};
    var l=glm.hybrid_ledger.Ledger.init(.attention,922);try l.install();defer l.uninstall();var d:cuda.Driver=undefined;const e=try C.engine(&d);d.api.cuEventCreate=X.event;d.api.cuEventDestroy_v2=X.drop;
    var r:@import("tp").roce.Roce=undefined;var pair=try T.NativePair.init(&d,&r,e.stream);const coord=try T.Coordinator.init(t.allocator,&d,undefined,.{undefined,undefined},.{.session=922,.model=@splat(1),.settings=@splat(2),.transaction_timeout_ns=1000});
    const dev=pair.frames.len+coord.slots.len;const pin=pair.host.bytes.len+coord.request.bytes.len+coord.reply.bytes.len;const status=l.status();
    coord.releaseStorage();pair.releaseStorage();e.deinit();l.finish(true,false);const owned=status.owners[@intFromEnum(m.Owner.transport)];
    try t.expectEqual(dev,owned.device);try t.expectEqual(pin,owned.pinned);try t.expectEqual(@as(u64,0),status.unclassified_buffers);try t.expect(l.status().clean);try t.expectEqual(@as(usize,0),C.live);
}

test "I3 native: selected preload precedes generic checkpoint fallback and propagates refusal" {
    const api=@import("engine_api");const fm=@import("src/native/cuda_family_memory.zig");
    const Selected=struct {
        pub fn preloadWeightBytes(_:std.mem.Allocator,_:std.Io,_:api.Open) !?u64 {return error.SelectedOwnerUnknown;}
    };
    var why:[]const u8="";defer if(why.len>0) t.allocator.free(why);
    try t.expectError(error.SelectedOwnerUnknown,fm.weights(Selected,t.allocator,t.io,.{.dir="/fixture/NEVER_OPENED",.model_type="glm5_next"},&why));
}

// AI-assisted integration: actual attention native and Engine entry stop gates before any metadata/device access.
test "V4 startup: actual GLM Engine refuses latched stop before checkpoint or CUDA" {
    var requested=std.atomic.Value(bool).init(true);
    var o=glm.engine.Options{.capacity=64,.mtp=false,.settings=.{}};
    if(comptime @hasField(glm.engine.Options,"cancel_requested"))o.cancel_requested=&requested;
    try t.expectError(error.Stopped,glm.engine.Engine.init(t.allocator,t.io,undefined,undefined,"/CPU/MISSING_CHECKPOINT99",undefined,undefined,0,1,o));
}
test "V4 startup: actual GLM native open refuses latched stop before selected config or bootstrap" {
    const Env=struct{extern "c" fn setenv([*:0]const u8,[*:0]const u8,c_int)c_int;extern "c" fn unsetenv([*:0]const u8)c_int;};
    try t.expect(std.c.getenv("TF_GLM_HYBRID_CONFIG")==null);try t.expectEqual(@as(c_int,0),Env.setenv("TF_GLM_HYBRID_CONFIG","/CPU/MISSING_SELECTED_CONFIG99",1));defer _=Env.unsetenv("TF_GLM_HYBRID_CONFIG");
    var requested=std.atomic.Value(bool).init(true);
    var o=glm.native.Options{.context=64,.drafts=false};
    if(comptime @hasField(glm.native.Options,"cancel_requested"))o.cancel_requested=&requested;
    try t.expectError(error.Stopped,glm.native.open(t.allocator,t.io,undefined,"/CPU/MISSING_CHECKPOINT99",null,o));
}
test "V4 startup: native family options forward exact sticky token without mutation" {
    const api=@import("engine_api");const options=@import("src/native/cuda_options.zig");
    const Family=struct{pub const Options=struct{cancel_requested:?*const std.atomic.Value(bool)=null,};};
    var requested=std.atomic.Value(bool).init(true);var input=api.Open{.dir="/CPU/NONE",.model_type="glm5_next"};
    if(comptime @hasField(api.Open,"cancel_requested"))input.cancel_requested=&requested;
    var out:Family.Options=.{};var why:[]const u8="";
    try t.expect(options.set(Family,input,&out,&why));try t.expect(out.cancel_requested==&requested);try t.expect(requested.load(.acquire));
}

// AI-assisted integration: actual attention Engine.init, actual hash worker, real OS TERM; CUDA calls record only.
test "V4 startup: attention hash cooperatively aborts before digest success or first CUDA stream" {
    const a=t.allocator;var tmp=t.tmpDir(.{});defer tmp.cleanup();
    const text=try glm.config.textConfig(a,1,0,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const shape=try glm.config.parse(a,config);
    var header=@import("glm_hybrid_memory_fixture.zig").Header{.a=a};defer header.text.deinit(a);try header.expert(shape,true);try header.write(tmp.dir,"");
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);
    const Probe=struct {
        var requested=std.atomic.Value(bool).init(false);var sent=std.atomic.Value(bool).init(false);var completed=std.atomic.Value(usize).init(0);var streams:usize=0;var destroyed:usize=0;
        fn stop(_:std.posix.SIG)callconv(.c)void {requested.store(true,.release);}
        fn block(_:u64)!void {if(!sent.swap(true,.acq_rel))try std.posix.raise(.TERM);}
        fn success()!void {_=completed.fetchAdd(1,.monotonic);return error.CpuAfterDigestBoundary;}
        fn stream(out:*cuda.abi.Stream,_:c_uint)callconv(.c)c_int {streams+=1;out.*=null;return 0;}
        fn sync(_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
        fn destroy(_:cuda.abi.Stream)callconv(.c)c_int{destroyed+=1;return 0;}
    };
    Probe.requested.store(false,.release);Probe.sent.store(false,.release);Probe.completed.store(0,.release);Probe.streams=0;Probe.destroyed=0;
    const action:std.posix.Sigaction=.{.handler=.{.handler=Probe.stop},.mask=std.posix.sigemptyset(),.flags=0};var old:std.posix.Sigaction=undefined;std.posix.sigaction(.TERM,&action,&old);defer std.posix.sigaction(.TERM,&old,null);
    glm.ranks.IdentityTest.before_block=Probe.block;defer glm.ranks.IdentityTest.before_block=null;glm.ranks.IdentityTest.before_success=Probe.success;defer glm.ranks.IdentityTest.before_success=null;
    var d:cuda.Driver=undefined;d.api.cuStreamCreate=Probe.stream;d.api.cuStreamSynchronize=Probe.sync;d.api.cuStreamDestroy_v2=Probe.destroy;
    try t.expectError(error.Stopped,glm.engine.Engine.init(a,t.io,&d,undefined,dir,undefined,undefined,0,1,.{.capacity=64,.mtp=false,.settings=.{},.cancel_requested=&Probe.requested}));
    try t.expect(Probe.sent.load(.acquire));try t.expect(Probe.requested.load(.acquire));try t.expectEqual(@as(usize,0),Probe.completed.load(.acquire));try t.expectEqual(@as(usize,0),Probe.streams);try t.expectEqual(@as(usize,0),Probe.destroyed);
}

test "V4 startup: root consumes attention one-upload cancellation" {try glm.weights.StopTest.afterUpload();}
test "V4 startup: root consumes attention expired entry deadline" {try glm.weights.StopTest.expiredEntry();}

test "V4 startup: actual native Engine options preserve token and absolute setup bound" {try glm.native.StopTest.nativeEngineOptions();}
test "V4 startup: actual Engine setup bound is refused before checkpoint or stream" {
    var o=glm.engine.Options{.capacity=64,.mtp=false,.settings=.{}};if(comptime @hasField(glm.engine.Options,"setup_until"))o.setup_until=0;
    try t.expectError(error.Timeout,glm.engine.Engine.init(t.allocator,t.io,undefined,undefined,"/CPU/NO_CHECKPOINT105",undefined,undefined,0,1,o));
}

test "V4 startup: bootstrap socket readiness consumes TERM while peer never writes" {try @import("tp").sock.StopTest.blockedReadiness();}
test "V4 startup: actual bootstrap refuses captured stop before entry" {try @import("tp").Bootstrap.StopTest.cancelledEntry();}
test "V4 startup: actual bootstrap refuses captured stop before agreement" {try @import("tp").Bootstrap.StopTest.cancelledAgreement();}

test "V4 startup: actual Peer refuses TERM before its event and bootstrap" {try glm.moe.hybridForward.StopTest.peerEntry(false);}
test "V4 startup: actual Session refuses TERM before either peer allocates" {try glm.moe.hybridForward.StopTest.peerEntry(true);}

test "V4 startup: cancelled actual connect owns and closes its unpublished descriptor" {try @import("tp").sock.StopTest.cancelledConnectOwnership();}

test "V4 startup: generic CUDA boundary recognizes only latched Stopped" {
    const opts=@import("src/native/cuda_options.zig");var flag=std.atomic.Value(bool).init(true);
    const yes=if(comptime @hasDecl(opts,"startupStopped"))opts.startupStopped(error.Stopped,&flag)else false;try t.expect(yes);
}
test "V4 startup: generic CUDA boundary recognizes only latched Cancelled" {
    const opts=@import("src/native/cuda_options.zig");var flag=std.atomic.Value(bool).init(true);
    const yes=if(comptime @hasDecl(opts,"startupStopped"))opts.startupStopped(error.Cancelled,&flag)else false;try t.expect(yes);
}
test "V4 startup: generic CUDA boundary never hides retention OOM timeout or unlatched errors" {
    const opts=@import("src/native/cuda_options.zig");var yes=std.atomic.Value(bool).init(true);var no=std.atomic.Value(bool).init(false);
    if(comptime @hasDecl(opts,"startupStopped")){
        for([_]anyerror{error.HybridRetained,error.LoadRetained,error.OutOfMemory,error.Timeout})|err|try t.expect(!opts.startupStopped(err,&yes));
        try t.expect(!opts.startupStopped(error.Stopped,&no));try t.expect(!opts.startupStopped(error.Cancelled,&no));try t.expect(!opts.startupStopped(error.Stopped,null));
    }
}
