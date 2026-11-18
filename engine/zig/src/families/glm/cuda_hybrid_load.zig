//! Role-scoped GLM ownership over the native EXL3 channel-cut loader; no second whole engine on experts.
const std=@import("std");
const core=@import("core");
const cuda=@import("cuda");
const config=@import("config.zig");
const weights=@import("cuda_weights.zig");
const exl3=@import("cuda_exl3.zig");
const shard=@import("cuda_shard.zig");
const prefix="model.language_model.";
/// Nonrenewing enclosing startup bound and cooperative TERM at actual I/O/upload boundaries.
pub const Control=struct {
    stopping:?*const std.atomic.Value(bool)=null,until:?u64=null,
    pub fn check(c:Control)!void {
        if(c.stopping)|s|if(s.load(.acquire))return error.Cancelled;
        if(c.until)|at|if(@import("tp").sock.nowNs()>=at)return error.Timeout;
    }
};
pub const Role=weights.Role;
pub const ExpertLayer=struct { index: u32, data: exl3.Experts };
pub const ExpertWeights=struct {
    gpa: std.mem.Allocator, config: config.Config, cut: shard.Span, half: u1,
    layers: []ExpertLayer=&.{},
    buffers: std.ArrayList(cuda.DeviceBuffer)=.empty,
    named: std.ArrayList(weights.Named)=.empty,
    pub fn deinit(w: *ExpertWeights) void {
        for(w.buffers.items) |*b| b.free();
        for(w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);w.named.deinit(w.gpa);w.gpa.free(w.layers);w.*=undefined;
    }
};
/// Only the two native operations the common EXL3 loader needs. No attention/shared/Engine owner is constructed.
const Loader=struct {
    gpa: std.mem.Allocator,d: *const cuda.Driver,w: *ExpertWeights,ck: *core.Checkpoint,
    host: std.ArrayList(u8)=.empty,name_buf: [192]u8=undefined,control:Control=.{},
    pub fn get(L: *Loader,comptime fmt: []const u8,args: anytype) !core.checkpoint.Tensor {
        try L.control.check();return L.ck.get(try std.fmt.bufPrint(&L.name_buf,prefix++fmt,args));
    }
    pub fn put(L: *Loader,comptime fmt: []const u8,args: anytype) !u64 {
        try L.control.check();const ptr=try @import("cuda_upload_owner.zig").keep(L,try cuda.DeviceBuffer.fromHost(L.d,L.host.items),fmt,args);
        L.host.clearRetainingCapacity();try L.control.check();return ptr;
    }
};
/// The real consuming traversal, also exercised with scaled synthetic checkpoint bytes by host tests.
pub fn loadExpertLayers(gpa: std.mem.Allocator,d: *const cuda.Driver,stream: cuda.Stream,ck: *core.Checkpoint,c: config.Config,half: u1,layers: []const u32,with_mtp: bool) !ExpertWeights {
    return loadExpertLayersControlled(gpa,d,stream,ck,c,half,layers,with_mtp,.{});
}
pub fn loadExpertLayersControlled(gpa: std.mem.Allocator,d: *const cuda.Driver,stream: cuda.Stream,ck: *core.Checkpoint,c: config.Config,half: u1,layers: []const u32,with_mtp: bool,control:Control) !ExpertWeights {
    try control.check();_=stream; // native fromHost keeps every borrowed upload until synchronous-copy completion
    if(!c.exl3 or c.moe_inter%256!=0 or c.own[0]!=0 or c.own[1]!=c.experts) return error.BadExpertGeometry;
    const memory_scope=@import("cuda_hybrid_ledger.zig").scopeOwner(.weights);defer memory_scope.end();
    const width=c.moe_inter/2;
    var w=ExpertWeights{.gpa=gpa,.config=c,.half=half,.cut=.{.lo=@as(usize,half)*width,.hi=(@as(usize,half)+1)*width}};
    errdefer w.deinit();
    var L=Loader{.gpa=gpa,.d=d,.w=&w,.ck=ck,.control=control};defer L.host.deinit(gpa);
    var count: usize=@intFromBool(with_mtp and c.mtp>0);
    var prev: ?u32=null;
    for(layers) |i| {if(i>=c.layers or (prev!=null and i<=prev.?)) return error.BadLayer;prev=i;count+=@intFromBool(c.isMoe(i));}
    w.layers=try gpa.alloc(ExpertLayer,count);var at: usize=0;
    for(layers) |i| if(c.isMoe(i)) {w.layers[at]=.{.index=i,.data=try exl3.load(&L,i,c.experts,c.hidden,c.moe_inter,w.cut)};at+=1;ck.release();};
    if(with_mtp and c.mtp>0) {w.layers[at]=.{.index=c.layers,.data=try exl3.load(&L,c.layers,c.experts,c.hidden,c.moe_inter,w.cut)};ck.release();}
    try control.check();return w;
}

test "hybrid load: cancellation at first real upload prevents later checkpoint reads and releases partial owners" {try @import("cuda_hybrid_load_test.zig").controlled(true);}
test "hybrid load: expired enclosing setup admits no checkpoint read or CUDA allocation" {try @import("cuda_hybrid_load_test.zig").controlled(false);}

test "hybrid load: completed expert layer releases real checkpoint pages before the next CUDA upload" {try @import("cuda_hybrid_load_test.zig").reclamation();}
