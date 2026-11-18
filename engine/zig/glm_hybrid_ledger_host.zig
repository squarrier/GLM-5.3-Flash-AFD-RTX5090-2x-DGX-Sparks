// SPDX-License-Identifier: Apache-2.0
const std=@import("std");const t=std.testing;const glm=@import("glm");const cuda=@import("cuda");
test "I3 memory receipt consumes actual owned status and preserves cancelled/unknown physical scope" {
    try t.expect(@hasField(cuda.memory_receipt.Receipt,"ownership"));
    if(comptime @hasField(cuda.memory_receipt.Receipt,"ownership")){
        var l=glm.hybrid_ledger.Ledger.init(.attention,912);try l.install();defer l.uninstall();
        const scope=l.scope(.gather);defer scope.end();var d=Fake.driver();
        var b=try cuda.DeviceBuffer.alloc(&d,32);var h=try cuda.HostBuffer.alloc(&d,24);
        const live=cuda.memory_receipt.snapshot(0,.load_end,false,123).ownership.?;
        try t.expectEqual(@as(u64,32),live.live_device);try t.expectEqual(@as(u64,24),live.live_pinned);try t.expectEqualStrings("attention",live.role);try t.expectEqual(@as(u128,912),live.incarnation);
        try t.expect(live.driver_private==null and live.verbs_private==null and live.heap_and_checkpoint_residency==null);
        b.free();h.free();l.finish(true,true);const closed=cuda.memory_receipt.snapshot(0,.cleanup_end,false,124).ownership.?;
        try t.expect(closed.closed and closed.cancelled and closed.clean);try t.expectEqual(@as(u64,0),closed.retained_device);
    }
}
const Fake=struct{
    var next:u64=0x100000;var fail:bool=false;
    fn alloc(p:*u64,n:usize) callconv(.c) c_int{p.*=next;next+=n+256;return 0;}
    fn free(_:u64) callconv(.c) c_int{return if(fail)cuda.abi.error_not_ready else 0;}
    fn host(p:*?*anyopaque,n:usize,_:c_uint) callconv(.c) c_int{var ptr:u64=0;_=alloc(&ptr,n);p.*=@ptrFromInt(ptr);return 0;}
    fn hfree(p:?*anyopaque) callconv(.c) c_int{return free(@intFromPtr(p.?));}
    fn driver()cuda.Driver{next=0x100000;fail=false;var d:cuda.Driver=undefined;d.api.cuMemAlloc_v2=alloc;d.api.cuMemFree_v2=free;d.api.cuMemHostAlloc=host;d.api.cuMemFreeHost=hfree;return d;}
};
test "I3 ledger capacity refusal remains sticky after an owning slot is released" {
    var l=glm.hybrid_ledger.Ledger.init(.attention,920);try l.install();defer l.uninstall();var d=Fake.driver();
    const N=struct{fn refused(ctx:*anyopaque)void{const n:*usize=@ptrCast(@alignCast(ctx));n.*+=1;}};var notifications:usize=0;l.refusal_ctx=&notifications;l.refuse=N.refused;
    var owned:[glm.hybrid_memory.ledger_capacity]cuda.DeviceBuffer=undefined;var made:usize=0;defer for(owned[0..made])|*b|b.free();
    for(&owned)|*b|{b.*=try cuda.DeviceBuffer.alloc(&d,1);made+=1;}
    try t.expectError(error.OutOfDeviceMemory,cuda.DeviceBuffer.alloc(&d,1));try t.expectEqual(@as(usize,1),notifications);
    owned[made-1].free();made-=1;const next=Fake.next;
    if(cuda.DeviceBuffer.alloc(&d,1))|accepted|{var b=accepted;b.free();return error.TestExpectedError;}else|err|try t.expectEqual(error.OutOfDeviceMemory,err);
    try t.expectEqual(next,Fake.next);
}

test "I3 ledger refuses in-flight commit after concurrent capacity quarantine" {
    const C=struct{var entered=std.atomic.Value(bool).init(false);var proceed=std.atomic.Value(bool).init(false);var accepted=false;
        fn alloc(p:*u64,n:usize)callconv(.c)c_int{entered.store(true,.release);while(!proceed.load(.acquire))std.Thread.yield() catch {};return Fake.alloc(p,n);}
        fn run(d:*cuda.Driver)void{if(cuda.DeviceBuffer.alloc(d,1))|got|{accepted=true;var b=got;b.free();}else|err|{std.debug.assert(err==error.OutOfDeviceMemory);}}
    };
    var l=glm.hybrid_ledger.Ledger.init(.attention,926);try l.install();defer l.uninstall();var d=Fake.driver();var owned:[glm.hybrid_memory.ledger_capacity-1]cuda.DeviceBuffer=undefined;var made:usize=0;defer for(owned[0..made])|*b|b.free();for(&owned)|*b|{b.*=try cuda.DeviceBuffer.alloc(&d,1);made+=1;}
    C.entered.store(false,.release);C.proceed.store(false,.release);C.accepted=false;d.api.cuMemAlloc_v2=C.alloc;const worker=try std.Thread.spawn(.{},C.run,.{&d});while(!C.entered.load(.acquire))std.Thread.yield() catch {};
    const result=cuda.DeviceBuffer.alloc(&d,1);C.proceed.store(true,.release);worker.join();if(result)|got|{var b=got;b.free();return error.TestExpectedError;}else|err|try t.expectEqual(error.OutOfDeviceMemory,err);
    try t.expect(!C.accepted);try t.expect(l.status().sticky_failure);
}

test "I3 ledger sequence overflow refuses before physical allocation and notifies quarantine" {
    const C=struct{fn notify(ctx:*anyopaque)void{const n:*usize=@ptrCast(@alignCast(ctx));n.*+=1;}};
    var l=glm.hybrid_ledger.Ledger.init(.attention,927);l.sequence=std.math.maxInt(u64);var notified:usize=0;l.refusal_ctx=&notified;l.refuse=C.notify;try l.install();defer l.uninstall();var d=Fake.driver();const next=Fake.next;
    try t.expectError(error.OutOfDeviceMemory,cuda.DeviceBuffer.alloc(&d,16));try t.expectEqual(next,Fake.next);try t.expectEqual(@as(usize,1),notified);l.finish(true,false);try t.expect(!l.status().clean and l.status().sticky_failure);
}
test "I3 ledger concurrent copied release cannot enter driver while the first free is held" {
    const C=struct{var entered=std.atomic.Value(bool).init(false);var proceed=std.atomic.Value(bool).init(false);var calls=std.atomic.Value(usize).init(0);
        fn free(_:u64)callconv(.c)c_int{_=calls.fetchAdd(1,.monotonic);entered.store(true,.release);while(!proceed.load(.acquire))std.Thread.yield() catch {};return 0;}
        fn run(b:*cuda.DeviceBuffer)void{b.free();}
    };
    var l=glm.hybrid_ledger.Ledger.init(.attention,928);try l.install();defer l.uninstall();var d=Fake.driver();d.api.cuMemFree_v2=C.free;C.entered.store(false,.release);C.proceed.store(false,.release);C.calls.store(0,.release);
    var b=try cuda.DeviceBuffer.alloc(&d,16);var stale=b;const thread=try std.Thread.spawn(.{},C.run,.{&b});while(!C.entered.load(.acquire))std.Thread.yield() catch {};stale.free();const pending=l.status();C.proceed.store(true,.release);thread.join();l.finish(true,false);
    try t.expectEqual(@as(usize,1),C.calls.load(.acquire));try t.expectEqual(@as(u64,16),pending.live.device);try t.expect(pending.sticky_failure and !pending.clean);try t.expect(l.status().clean);
}

test "I3 allocation capture: closing inside real allocation cannot miss the in-flight backing" {
    const C=struct{var ledger:*glm.hybrid_ledger.Ledger=undefined;var was_clean=false;
        fn alloc(p:*u64,n:usize)callconv(.c)c_int{ledger.finish(true,false);was_clean=ledger.status().clean;return Fake.alloc(p,n);}
    };
    var l=glm.hybrid_ledger.Ledger.init(.attention,913);C.ledger=&l;C.was_clean=false;try l.install();defer l.uninstall();var d=Fake.driver();d.api.cuMemAlloc_v2=C.alloc;
    if(cuda.DeviceBuffer.alloc(&d,16))|accepted|{var b=accepted;b.free();return error.TestExpectedError;}else|err|try t.expectEqual(error.OutOfDeviceMemory,err);
    try t.expect(!C.was_clean);
}
test "I3 allocation witness rejects stale copied release before the actual driver" {
    const C=struct{var calls:usize=0;var rejecting=false;fn free(_:u64)callconv(.c)c_int{calls+=1;return if(rejecting)cuda.abi.error_not_ready else 0;}};
    var l=glm.hybrid_ledger.Ledger.init(.attention,914);try l.install();defer l.uninstall();var d=Fake.driver();d.api.cuMemFree_v2=C.free;C.calls=0;C.rejecting=false;
    var a=try cuda.DeviceBuffer.alloc(&d,16);var stale=a;const address=a.ptr;a.free();Fake.next=address;var b=try cuda.DeviceBuffer.alloc(&d,16);defer b.free();
    C.rejecting=true;stale.free();C.rejecting=false;try t.expectEqual(@as(usize,1),C.calls);
    try t.expectEqual(@as(u64,16),l.status().live.device);
}
test "I3 hook refusal retains original handle and charge until recognition succeeds" {
    const C=struct{var accepted=false;fn alloc(_: *anyopaque,_:usize)?u64{return 0xCC0000;}fn free(_: *anyopaque,_:u64)bool{return accepted;}};
    var l=glm.hybrid_ledger.Ledger.init(.attention,915);try l.install();defer l.uninstall();var marker:u8=0;C.accepted=false;cuda.memory.setHook(.{.ctx=&marker,.alloc=C.alloc,.free=C.free});defer cuda.memory.setHook(null);
    const before=cuda.usage(false);var b=try cuda.DeviceBuffer.alloc(undefined,16);const address=b.ptr;b.free();
    try t.expectEqual(address,b.ptr);try t.expectEqual(before.device+16,cuda.usage(false).device);try t.expectEqual(@as(u64,16),l.status().hooked_logical_bytes);
    C.accepted=true;b.free();l.finish(true,false);try t.expect(l.status().clean);try t.expectEqual(before.device,cuda.usage(false).device);
}
test "I3 allocation witness: actual native failed free stays charged and typed retained" {
    try t.expect(@hasDecl(glm,"hybrid_ledger"));
    if(comptime @hasDecl(glm,"hybrid_ledger")){
        var d=Fake.driver();var l=glm.hybrid_ledger.Ledger.init(.attention,42);try l.install();defer l.uninstall();
        const scope=l.scope(.weights);defer scope.end();const before=cuda.usage(false);
        var b=try cuda.DeviceBuffer.alloc(&d,16);var h=try cuda.HostBuffer.alloc(&d,32);
        try t.expectEqual(@as(u64,16),l.status().live.device);try t.expectEqual(@as(u64,32),l.status().live.host);try t.expectEqual(@as(u64,32),l.status().live.pinned);
        Fake.fail=true;b.free();h.free();const refused=l.status();
        try t.expectEqual(@as(u64,16),refused.retained.device);try t.expectEqual(@as(u64,32),refused.retained.host);
        try t.expectEqual(before.device+16,cuda.usage(false).device);try t.expectEqual(before.host+32,cuda.usage(false).host);
        try t.expect(!refused.clean and refused.sticky_failure);try t.expect(refused.driver_private==null);
        Fake.fail=false;b.free();h.free();l.finish(true,false);const after=l.status();
        try t.expectEqual(@as(u64,0),after.live.device+after.live.host+after.retained.device+after.retained.host);try t.expect(after.clean and after.sticky_failure);
        try t.expectEqual(before.device,cuda.usage(false).device);try t.expectEqual(before.host,cuda.usage(false).host);
    }
}

const Upload=struct{
    var fail=false;
    fn host(p:*?*anyopaque,n:usize,_:c_uint)callconv(.c)c_int{p.*=std.c.malloc(n);return if(p.*==null)2 else 0;}
    fn free(p:?*anyopaque)callconv(.c)c_int{if(fail)return cuda.abi.error_not_ready;std.c.free(p);return 0;}
    fn copy(_:u64,_:?*const anyopaque,_:usize,_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
    fn event(p:*cuda.abi.Event,_:c_uint)callconv(.c)c_int{p.*=@ptrFromInt(77);return 0;}
    fn record(_:cuda.abi.Event,_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
    fn ev(_:cuda.abi.Event)callconv(.c)c_int{return 0;}
    fn sync(_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
    fn context()callconv(.c)c_int{return 0;}
    fn driver()cuda.Driver{var d=Fake.driver();fail=false;d.api.cuMemHostAlloc=host;d.api.cuMemFreeHost=free;d.api.cuMemcpyHtoDAsync_v2=copy;d.api.cuEventCreate=event;d.api.cuEventRecord=record;d.api.cuEventQuery=ev;d.api.cuEventDestroy_v2=ev;d.api.cuEventSynchronize=ev;d.api.cuStreamSynchronize=sync;d.api.cuCtxSynchronize=context;return d;}
};
test "I3 async custody: actual upload completion releases the original pinned witness" {
    var l=glm.hybrid_ledger.Ledger.init(.attention,941);try l.install();defer l.uninstall();var d=Upload.driver();
    cuda.copy_source.setCurrent(@ptrFromInt(941));defer cuda.copy_source.setCurrent(null);
    const scope=l.scope(.load_staging);defer scope.end();const before=cuda.usage(false);
    const b:cuda.DeviceBuffer=.{.d=&d,.ptr=0xCC000,.len=16,.borrowed=true};
    try b.uploadAsync(0,&.{1,2,3,4},@ptrFromInt(7));try t.expectEqual(@as(u64,4096),l.status().live.pinned);
    try cuda.copy_source.synchronizeStream(&d,@ptrFromInt(7));l.finish(true,false);
    try t.expectEqual(@as(usize,0),cuda.copy_source.retained());try t.expectEqual(before.host,cuda.usage(false).host);
    try t.expect(l.status().clean);try t.expectEqual(l.status().allocations,l.status().releases);
}

test "I3 async custody: failed native free retains quarantine node and original witness" {
    var l=glm.hybrid_ledger.Ledger.init(.attention,942);try l.install();defer l.uninstall();var d=Upload.driver();
    cuda.copy_source.setCurrent(@ptrFromInt(942));defer cuda.copy_source.setCurrent(null);
    const scope=l.scope(.load_staging);defer scope.end();const before=cuda.usage(false);
    var h=try cuda.HostBuffer.alloc(&d,32);cuda.copy_source.retainPinned(&d,h,.{.d=&d,.handle=@ptrFromInt(8)});
    Upload.fail=true;try cuda.copy_source.synchronizeStream(&d,@ptrFromInt(8));const count=cuda.copy_source.retained();const failed=l.status();
    Upload.fail=false;if(count==0)h.free() else try cuda.copy_source.synchronizeStream(&d,@ptrFromInt(8));l.finish(true,false);
    try t.expectEqual(@as(usize,1),count);try t.expectEqual(@as(u64,32),failed.retained.pinned);try t.expect(failed.sticky_failure and !failed.clean);
    try t.expect(l.status().clean);try t.expectEqual(before.host,cuda.usage(false).host);
}

test "I3 async custody: direct Stage disposal keeps failed pinned backing discoverable" {
    var l=glm.hybrid_ledger.Ledger.init(.attention,944);try l.install();defer l.uninstall();var d=Upload.driver();
    cuda.copy_source.setCurrent(@ptrFromInt(944));defer cuda.copy_source.setCurrent(null);
    const scope=l.scope(.load_staging);defer scope.end();const before=cuda.usage(false);
    var s:cuda.copy_source.Stage=.{};_=try s.take(&d,32);var saved=s.buf.?;Upload.fail=true;s.deinit();
    const count=cuda.copy_source.retained();const bad=l.status();Upload.fail=false;
    if(count==0)saved.free() else try cuda.copy_source.synchronizeContext(&d);l.finish(true,false);
    try t.expectEqual(@as(usize,1),count);try t.expectEqual(@as(u64,4096),bad.retained.pinned);try t.expect(l.status().clean);
    try t.expectEqual(before.host,cuda.usage(false).host);
}
