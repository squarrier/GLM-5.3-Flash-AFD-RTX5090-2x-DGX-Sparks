// Modified for V4 by Scott Quarrier with AI assistance: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! Release a stopped native CUDA host in ownership order.
const failed_load = @import("cuda_load_gpu.zig");
const kv_pages = @import("cuda_kv_pool.zig");
pub fn releaseFamily(ctx:*anyopaque,release:*const fn(*anyopaque) void,checked:?*const fn(*anyopaque) bool)bool {
    if(checked)|run|return run(ctx);release(ctx);return true;
}
pub fn close(h: anytype) void {
    h.host.stop();
    if(comptime @hasField(@TypeOf(h.*),"close_gate")) if(h.close_gate) |gate| {
        if(!gate(h.family)) {
            if(h.gpu) |g| {
                if(!g.hybrid_retained) {g.hybrid_retained=true;failed_load.keep(g);}
                @import("cuda_retained_exit.zig").fail("served hybrid close could not prove quiescence");
            }
            return; // keep the host, driver, context and every allocation owner together
        }
    };
    kv_pages.unwatch(&h.kv); // the engine thread is joined: the pool's teardown below reports to no one
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    if (h.recording) |r| r.free(h.gpa);
    failed_load.closeEntry(if (h.gpu) |g| &g.ctx else null) catch {};
    if (h.store) |s| kv_pages.closeStore(h.gpa, s);
    if (h.prompts) |x| x.close(); // its pages back to the pool before the family frees the pool
    if (h.slide) |s| h.slide_free.?(h.gpa, s);
    h.gpa.free(h.dir);
    h.served.close(h.gpa);
    const checked=if(comptime @hasField(@TypeOf(h.*),"release_checked"))h.release_checked else null;
    if(!releaseFamily(h.family,h.release,checked)){
        if(h.gpu)|g| {
            if(!g.hybrid_retained){g.hybrid_retained=true;failed_load.keep(g);}
            @import("cuda_retained_exit.zig").fail("served hybrid release retained allocation owners");
        }
        return; // real failed frees retain the GPU context/driver and host identity
    }
    if (h.gpu) |g| {
        g.ctx.deinit();
        g.driver.close();
        h.gpa.destroy(g);
    }
    h.gpa.free(h.startup);
    h.gpa.destroy(h);
}

const std=@import("std");
const api=@import("engine_api");
const CloseFixture=struct {
    var stops: usize=0;var releases: usize=0;var checked_calls:usize=0;
    const Host=struct {fn stop(_: @This()) void {stops+=1;}};
    const Core=struct {fn deinit(_: @This()) void {releases+=1;}};
    const Cfg=struct {fn deinit(_: @This(),_: std.mem.Allocator) void {releases+=1;}};
    const Rec=struct {fn free(_: *@This(),_: std.mem.Allocator) void {releases+=1;}};
    const Prompt=struct {fn close(_: *@This()) void {releases+=1;}};
    const Served=struct {fn close(_: @This(),_: std.mem.Allocator) void {releases+=1;}};
    fn refuse(_: *anyopaque) bool {return false;}
    fn release(_: *anyopaque) void {releases+=1;}
    fn checked(_: *anyopaque) bool {checked_calls+=1;return false;}
    const Owner=struct {
        host: Host=.{},core: Core=.{},cfg: Cfg=.{},served: Served=.{},gpa: std.mem.Allocator,
        kv: kv_pages.Watch=.{},recording: ?*Rec=null,store: ?*api.prompt_cache.Store=null,prompts: ?*Prompt=null,
        slide: ?*anyopaque=null,slide_free: ?*const fn(std.mem.Allocator,*anyopaque) void=null,
        dir: []u8=&.{},startup: []u8=&.{},gpu: ?*failed_load.Gpu=null,family: *anyopaque,release: *const fn(*anyopaque) void=CloseFixture.release,
        close_gate: ?*const fn(*anyopaque) bool=refuse,
        release_checked: ?*const fn(*anyopaque) bool=null,
    };
};
test "hybrid parent close: failed family quiescence retains native host and all CUDA parents" {
    CloseFixture.stops=0;CloseFixture.releases=0;var marker: u8=0;
    const h=try std.testing.allocator.create(CloseFixture.Owner);h.*=.{.gpa=std.testing.allocator,.family=&marker};
    close(h);
    if(CloseFixture.releases==0) std.testing.allocator.destroy(h);
    try std.testing.expectEqual(@as(usize,1),CloseFixture.stops);
    try std.testing.expectEqual(@as(usize,0),CloseFixture.releases);
}
test "hybrid parent close: I3 post-release retained allocations cannot destroy native CUDA parents" {
    CloseFixture.stops=0;CloseFixture.releases=0;CloseFixture.checked_calls=0;var marker:u8=0;
    const h=try std.testing.allocator.create(CloseFixture.Owner);h.*=.{.gpa=std.testing.allocator,.family=&marker,.close_gate=null,.release_checked=CloseFixture.checked};
    close(h);
    if(CloseFixture.checked_calls==1)std.testing.allocator.destroy(h);
    try std.testing.expectEqual(@as(usize,1),CloseFixture.checked_calls);
    try std.testing.expectEqual(@as(usize,3),CloseFixture.releases);
}
