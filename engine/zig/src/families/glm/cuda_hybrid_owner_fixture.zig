//! Test-only CUDA/verbs boundaries for the real Session/ExpertOwner constructors and role loop.
const std=@import("std");const cuda=@import("cuda");const tp=@import("tp");const cfg=@import("cuda_hybrid_config.zig");const shape=@import("config.zig");const load=@import("cuda_hybrid_load.zig");const x3=@import("cuda_exl3.zig");
pub var d: cuda.Driver=undefined;
pub var blocks=std.atomic.Value(usize).init(0);pub var events=std.atomic.Value(usize).init(0);pub var unbounded=std.atomic.Value(usize).init(0);
var next_stream=std.atomic.Value(usize).init(1);
var controls: [2][2]std.c.fd_t=undefined;var plans: [2][2]std.c.fd_t=undefined;
var sends: [2][2]std.atomic.Value(u64)=undefined;var copied: [2]std.atomic.Value(u32)=undefined;
const A=extern struct {in: u64,out: u64,send: u64,recv: u64,flags: u64,ctrl: u64,state: u64,nbytes: u64,slot: u64,timeout: u64,rank: u32};
fn alloc(p: *u64,n: usize) callconv(.c) c_int {const b=std.c.malloc(@max(n,1)) orelse return 2;p.*=@intFromPtr(b);_=blocks.fetchAdd(1,.monotonic);return 0;}
fn free(p: u64) callconv(.c) c_int {std.c.free(@ptrFromInt(p));_=blocks.fetchSub(1,.monotonic);return 0;}
fn host(p: *?*anyopaque,n: usize,_: c_uint) callconv(.c) c_int {var q: u64=0;const rc=alloc(&q,n);p.*=@ptrFromInt(q);return rc;}
fn hfree(p: ?*anyopaque) callconv(.c) c_int {return free(@intFromPtr(p.?));}
fn event(p: *cuda.abi.Event,_: c_uint) callconv(.c) c_int {p.*=@ptrFromInt(1);_=events.fetchAdd(1,.monotonic);return 0;}
fn eventFree(_: cuda.abi.Event) callconv(.c) c_int {_=events.fetchSub(1,.monotonic);return 0;}
fn record(_: cuda.abi.Event,_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
fn query(_: cuda.abi.Event) callconv(.c) c_int {return 0;}
fn stream(p: *cuda.abi.Stream,_: c_uint) callconv(.c) c_int {p.*=@ptrFromInt(next_stream.fetchAdd(1,.monotonic));return 0;}
fn streamFree(_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
fn streamQuery(_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
fn sync(_: cuda.abi.Stream) callconv(.c) c_int {_=unbounded.fetchAdd(1,.monotonic);return 0;}
fn moduleFree(_: cuda.abi.Module) callconv(.c) c_int {return 0;}
fn h2d(dst: u64,src: ?*const anyopaque,n: usize,_: cuda.abi.Stream) callconv(.c) c_int {@memcpy(@as([*]u8,@ptrFromInt(dst))[0..n],@as([*]const u8,@ptrCast(src.?))[0..n]);return 0;}
fn d2h(dst: ?*anyopaque,src: u64,n: usize,_: cuda.abi.Stream) callconv(.c) c_int {@memcpy(@as([*]u8,@ptrCast(dst.?))[0..n],@as([*]const u8,@ptrFromInt(src))[0..n]);return 0;}
fn launch(f: cuda.abi.Function,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: c_uint,_: cuda.abi.Stream,args: ?[*]?*anyopaque,_: ?[*]?*anyopaque) callconv(.c) c_int {
    const half=@intFromPtr(f)-100;if(half>=2) return 1;
    const a: *const A=@ptrCast(@alignCast(args.?[0].?));if(a.nbytes!=1048576 or a.rank>1) return 1;
    sends[half][a.rank].store(a.in,.release);const until=tp.sock.nowNs()+3*std.time.ns_per_s;
    while(sends[half][1-a.rank].load(.acquire)==0) {if(tp.sock.nowNs()>=until)return 1;std.Thread.yield() catch {};}
    const peer=sends[half][1-a.rank].load(.acquire);@memcpy(@as([*]u8,@ptrFromInt(a.out))[0..1048576],@as([*]const u8,@ptrFromInt(peer))[0..1048576]);_=copied[half].fetchAdd(1,.acq_rel);
    while(copied[half].load(.acquire)!=2) {if(tp.sock.nowNs()>=until)return 1;std.Thread.yield() catch {};}
    return 0;
}
pub fn prepare() !void {
    try std.testing.expectEqual(@as(usize,0),blocks.load(.acquire));try std.testing.expectEqual(@as(usize,0),events.load(.acquire));unbounded.store(0,.release);
    d=undefined;d.api.cuMemAlloc_v2=alloc;d.api.cuMemFree_v2=free;d.api.cuMemHostAlloc=host;d.api.cuMemFreeHost=hfree;d.api.cuEventCreate=event;d.api.cuEventDestroy_v2=eventFree;d.api.cuEventRecord=record;d.api.cuEventQuery=query;
    d.api.cuStreamCreate=stream;d.api.cuStreamDestroy_v2=streamFree;d.api.cuStreamQuery=streamQuery;d.api.cuStreamSynchronize=sync;d.api.cuModuleUnload=moduleFree;
    d.api.cuMemcpyHtoDAsync_v2=h2d;d.api.cuMemcpyDtoHAsync_v2=d2h;d.api.cuLaunchKernel=launch;
    for(0..2) |half| {try std.testing.expectEqual(@as(c_int,0),std.c.socketpair(std.posix.AF.UNIX,std.posix.SOCK.STREAM,0,&controls[half]));try std.testing.expectEqual(@as(c_int,0),std.c.socketpair(std.posix.AF.UNIX,std.posix.SOCK.STREAM,0,&plans[half]));for(&sends[half]) |*s| s.*=.init(0);copied[half]=.init(0);}
}
pub fn config(image: []const u8) cfg.Config {return .{.revision=3,.role=.attention,.session="00000000000000000000000000000001",.model_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",.master=.{"192.0.2.1","192.0.2.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image=image,.setup_ms=3000,.transaction_ms=3000,.close_ms=3000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{}};}
pub fn bootstrap(c: cfg.Config,half: u1,rank: u32,until: u64) !tp.Bootstrap {
    try std.testing.expect(tp.sock.nowNs()<until);var b=tp.Bootstrap{.rank=rank,.world=2,.timeout_ns=@as(u64,c.setup_ms)*std.time.ns_per_ms,.absolute_deadline=until};
    b.fds[1-rank]=controls[half][rank];b.plan_fds[1-rank]=plans[half][rank];return b;
}
pub fn roce(driver: ?*const cuda.Driver,b: *tp.Bootstrap,image: []const u8,opts: tp.roce.Options,custody: *?*tp.roce.Roce) !*tp.roce.Roce {
    const R=tp.roce.Roce;try std.testing.expect(driver==&d and custody.*==null and image.len==1 and opts.traffic_class==104 and opts.max_bytes==1048576);
    const half: usize=if(b.fds[1-b.rank]==controls[0][b.rank]) 0 else 1;
    const bytes=R.regionBytes(1048576);const mem=try std.posix.mmap(null,bytes,.{.READ=true,.WRITE=true},.{.TYPE=.PRIVATE,.ANONYMOUS=true},-1,0);@memset(mem,0);
    const r=try std.heap.page_allocator.create(R);r.*=.{.me=b.rank,.slot_bytes=1048576,.timeout_ns=opts.timeout_ns,.region=mem,.mapping=mem,.cpu=null,.d=driver,.wire=.{.shm=.{.local=mem,.peer=mem,.layout=.{.send=8192,.recv=8192+2*1048576,.flags=4096,.slot=1048576,.len=bytes}}},.state=try cuda.DeviceBuffer.alloc(&d,16),.func=.{.d=&d,.handle=@ptrFromInt(100+half)}};
    return r;
}
pub fn expert(comptime Owner: type) *const fn(*Owner,std.Io,shape.Config,[]const u8,u64) anyerror!cuda.grouped.Router {
    return struct {fn run(o: *Owner,_: std.Io,c: shape.Config,_: []const u8,until: u64) !cuda.grouped.Router {
        try std.testing.expect(tp.sock.nowNs()<until);const half=try o.c.half();o.plans=.{.d=&d,.handle=null};o.kernels=@as(x3.Kernels,undefined);o.kernels.?.mod=.{.d=&d,.handle=null};
        o.weights=load.ExpertWeights{.gpa=o.gpa,.config=c,.half=half,.cut=.{.lo=@as(usize,half)*1024,.hi=(@as(usize,half)+1)*1024}};
        const ls=try o.gpa.alloc(load.ExpertLayer,2);ls[0]=.{.index=3,.data=std.mem.zeroes(x3.Experts)};ls[1]=.{.index=c.layers,.data=std.mem.zeroes(x3.Experts)};o.weights.?.layers=ls;
        return undefined; // native kernel functions are never executed by the explicit close turn
    }}.run;
}
