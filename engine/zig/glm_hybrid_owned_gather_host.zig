// SPDX-License-Identifier: Apache-2.0
//! Actual Engine.prepareHostGather/deinit. CPU CUDA boundary adapted byte-for-byte in behavior
//! from I2 Lifetime fixture; native TensorFold contributor and OpenAI Codex assistance credits retained.
const std=@import("std");const t=std.testing;const glm=@import("glm");const cuda=@import("cuda");
extern "c" fn setenv([*:0]const u8,[*:0]const u8,c_int)c_int;extern "c" fn unsetenv([*:0]const u8)c_int;
const F=struct{
    var live:usize=0;var calls:usize=0;var fail_at:usize=std.math.maxInt(usize);
    fn alloc(p:*u64,n:usize)callconv(.c)c_int{calls+=1;if(calls==fail_at)return cuda.abi.error_not_ready;const q=std.c.malloc(@max(n,1)) orelse return 2;p.*=@intFromPtr(q);live+=1;return 0;}
    fn free(p:u64)callconv(.c)c_int{if(p!=0){std.c.free(@ptrFromInt(p));live-=1;}return 0;}
    fn host(p:*?*anyopaque,n:usize,_:c_uint)callconv(.c)c_int{var q:u64=0;const rc=alloc(&q,n);p.*=@ptrFromInt(q);return rc;}
    fn hfree(p:?*anyopaque)callconv(.c)c_int{return free(@intFromPtr(p));}
    fn map(p:*u64,q:?*anyopaque,_:c_uint)callconv(.c)c_int{p.*=@intFromPtr(q);return 0;}
    fn clear(p:u64,v:u32,n:usize)callconv(.c)c_int{@memset(@as([*]u32,@ptrFromInt(p))[0..n],v);return 0;}
    fn sync(_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
    fn module(_:cuda.abi.Module)callconv(.c)c_int{return 0;}
    fn event(_:cuda.abi.Event)callconv(.c)c_int{return 0;}
    fn bind(comptime T:type,p:*T,d:*cuda.Driver)void{
        if(T==*const cuda.Driver){p.*=d;return;}if(T==std.mem.Allocator){p.*=t.allocator;return;}
        switch(@typeInfo(T)){.@"struct"=>if(comptime @typeInfo(T).@"struct".layout!=.@"packed"){inline for(@typeInfo(T).@"struct".field_names)|name|bind(@TypeOf(@field(p.*,name)),&@field(p.*,name),d);},.array=>for(p)|*v|bind(@TypeOf(v.*),v,d),else=>{}}
    }
    fn engine(d:*cuda.Driver)!*glm.engine.Engine{
        d.*=undefined;d.api.cuMemAlloc_v2=alloc;d.api.cuMemFree_v2=free;d.api.cuMemHostAlloc=host;d.api.cuMemFreeHost=hfree;d.api.cuMemHostGetDevicePointer_v2=map;d.api.cuMemsetD32_v2=clear;d.api.cuStreamSynchronize=sync;d.api.cuStreamDestroy_v2=sync;d.api.cuModuleUnload=module;d.api.cuEventDestroy_v2=event;
        const e=try t.allocator.create(glm.engine.Engine);@memset(std.mem.asBytes(e),0);bind(glm.engine.Engine,e,d);e.backend=try t.allocator.create(@typeInfo(@TypeOf(e.backend)).pointer.child);e.capacity=512;e.w.config=.{.layers=0,.kv_lora=16,.i_dim=8,.hidden=64};return e;
    }
};
test "I3 gather ownership: actual native singleton acquisition is once and real engine destructor releases nine backings" {
    try t.expectEqual(@as(c_int,0),setenv("TF_GLM_KV_HOST","1",1));defer _=unsetenv("TF_GLM_KV_HOST");
    F.live=0;F.calls=0;F.fail_at=std.math.maxInt(usize);var d:cuda.Driver=undefined;const e=try F.engine(&d);
    var l=glm.hybrid_ledger.Ledger.init(.attention,77);try l.install();defer l.uninstall();
    try e.prepareHostGather();const g=glm.kv_host.active.?;try t.expectEqual(@as(usize,9),F.live);
    const expected=g.slot_of.len+g.prev_of.len+g.list.len+g.slots.len+g.n.len+g.prev_rows.len+g.prev_n.len+g.cache.len;
    const s=l.status();try t.expectEqual(@as(u64,expected),s.live.device);try t.expectEqual(@as(u64,16),s.live.host);
    try e.prepareHostGather();try t.expectEqual(@as(usize,9),F.calls);
    e.deinit();l.finish(true,false);try t.expectEqual(@as(usize,0),F.live);try t.expect(l.status().clean);
    try t.expect(@hasField(@TypeOf(s),"owners"));
    if(comptime @hasField(@TypeOf(s),"owners"))try t.expectEqual(@as(u64,expected),s.owners[@intFromEnum(glm.hybrid_memory.Owner.gather)].device);
}
