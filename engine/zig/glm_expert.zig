const std=@import("std");
const glm=@import("glm");
const t=std.testing;

test "hybrid CLI: explicit role JSON binds full effective native settings before load" {
    const cfg=glm.moe.hybridConfig;const a=t.allocator;
    const set=glm.ranks.Settings{.pages=true,.kv8=true,.dense_bits=4,.head_bits=16};
    const text=try std.json.Stringify.valueAlloc(a,.{.revision=@as(u32,3),.role="attention",.session="00000000000000000000000000000001",.model_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",.master=.{"127.0.0.1","127.0.0.1"},.ports=.{@as(u16,29550),@as(u16,29551)},.traffic_class=@as(u8,104),.verbs_device="fixture_mlx5",.roce_image="/fixture/roce.fatbin",.setup_ms=@as(u32,30000),.transaction_ms=@as(u32,120000),.close_ms=@as(u32,5000),.served_context=@as(u32,262144),.resident_users=@as(u32,8),.mtp=true,.settings=set},.{});defer a.free(text);
    var parsed=try cfg.parse(a,text);defer parsed.deinit();
    try parsed.value.attention(a,1,262144,true,set);
    try t.expectError(error.BadHybridRole,parsed.value.attention(a,2,262144,true,set));
    try t.expectError(error.BadHybridContract,parsed.value.attention(a,1,8192,true,set));
    var changed=set;changed.kv8=false;
    try t.expectError(error.BadHybridContract,parsed.value.attention(a,1,262144,true,changed));
    for([_][2]u16{.{65535,29551},.{29550,29557}}) |ports| {
        var bad=parsed.value;bad.ports=ports;const bad_text=try std.json.Stringify.valueAlloc(a,bad,.{});defer a.free(bad_text);
        if(cfg.parse(a,bad_text)) |accepted| {var owned=accepted;owned.deinit();return error.TestExpectedError;} else |err| try t.expectEqual(error.BadHybridContract,err);
    }
    const identity=try parsed.value.identity(a);try t.expectEqual(@as(u128,1),identity.session);try t.expectEqual(@as(u8,0xAA),identity.model[0]);
}

test "hybrid CLI: complete expert argv binds config checkpoint device check before runtime" {
    const cfg=glm.moe.hybridConfig;
    const c=try cfg.Cli.parse(&.{"tf-glm-expert","--config","/fixture/role.json","--checkpoint","/fixture/checkpoint","--device","0","--check"});
    try t.expectEqualStrings("/fixture/role.json",c.config);try t.expectEqualStrings("/fixture/checkpoint",c.checkpoint);try t.expectEqual(@as(c_int,0),c.device);try t.expect(c.check);
    try t.expectError(error.BadExpertArgs,cfg.Cli.parse(&.{"tf-glm-expert","--config","/fixture/role.json"}));
    try t.expectError(error.BadExpertArgs,cfg.Cli.parse(&.{"tf-glm-expert","--config","x","--checkpoint","y","--device","0","--check","--unknown"}));
}

const cuda=@import("cuda");
const MemoryTest=if(@import("builtin").is_test)struct{pub var before_driver:?*const fn()anyerror!void=null;pub var before_identity:?*const fn()anyerror!void=null;}else struct{};
var stopping: std.atomic.Value(bool)=.init(false);
fn stop(_: std.posix.SIG) callconv(.c) void {stopping.store(true,.release);}
pub fn main(init: std.process.Init) !void {
    runMain(init) catch |err| {
        if(stopping.load(.acquire) and (err==error.Stopped or err==error.Cancelled))return;
        return err;
    };
}
fn runMain(init: std.process.Init) !void {
    const action: std.posix.Sigaction=.{.handler=.{.handler=stop},.mask=std.posix.sigemptyset(),.flags=0};std.posix.sigaction(.TERM,&action,null);std.posix.sigaction(.INT,&action,null);
    if(stopping.load(.acquire))return error.Stopped;
    const a=init.gpa;const args=try init.minimal.args.toSlice(init.arena.allocator());const cli=try glm.moe.hybridConfig.Cli.parse(args);
    const text=try std.Io.Dir.cwd().readFileAlloc(init.io,cli.config,a,.limited(65536));defer a.free(text);
    var parsed=try glm.moe.hybridConfig.parse(a,text);var retained=false;defer if(!retained) parsed.deinit();
    const c=parsed.value;_=try c.half();
    if(cli.check) {std.debug.print("CONFIG_ONLY role={s} WIRE3 NO_DEVICE\n",.{@tagName(c.role)});return;}
    if(comptime @import("builtin").is_test)if(MemoryTest.before_identity)|f|try f();
    if(stopping.load(.acquire))return error.Stopped;
    const role=glm.moe.hybridForward;const shape=try role.modelConfigControlled(a,init.io,cli.checkpoint,c,&stopping);
    if(stopping.load(.acquire))return error.Stopped;
    const identity=try c.identity(a);const memory=try a.create(glm.hybrid_ledger.Ledger);var installed=false;defer if(!installed)a.destroy(memory);
    memory.*=glm.hybrid_ledger.Ledger.init(if(c.role==.expert0).expert0 else .expert1,identity.session);try memory.install();installed=true;
    var lifecycle=cuda.memory_receipt.Lifecycle{.rank=@intCast(try c.half())};cuda.memory_receipt.emit(lifecycle.rank,.load_begin,false);
    defer {
        lifecycle.cleanup();
        memory.finish(!retained,stopping.load(.acquire));const final=cuda.memory_owner.snapshot();cuda.memory_receipt.emitCaptured(@intCast(c.half() catch unreachable),.cleanup_end,false,final);
        if(!retained and memory.status().clean){memory.uninstall();a.destroy(memory);}
    }
    const until=@import("tp").sock.nowNs()+|(@as(u64,c.setup_ms)*std.time.ns_per_ms);
    const Gpu=struct {driver: cuda.Driver,ctx: cuda.Context};const gpu=try a.create(Gpu);var driver=false;var context=false;
    defer if(!retained) {lifecycle.cleanup();if(context) gpu.ctx.deinit();if(driver) gpu.driver.close();a.destroy(gpu);};
    if(comptime @import("builtin").is_test)if(MemoryTest.before_driver)|f|try f();
    if(stopping.load(.acquire))return error.Stopped;
    gpu.driver=try cuda.Driver.open();driver=true;gpu.ctx=try cuda.Context.init(&gpu.driver,cli.device);context=true;
    if(stopping.load(.acquire))return error.Stopped;
    var budget_arena=std.heap.ArenaAllocator.init(a);defer budget_arena.deinit();var problem:[]const u8="";
    const pool=(try @import("src/native/cuda_family_memory.zig").pool(budget_arena.allocator(),init.io,&gpu.ctx,&problem)) orelse {std.debug.print("hybrid expert memory budget: {s}\n",.{problem});return error.HybridMemoryBudget;};
    const usage=cuda.usage(false);const held=try std.math.add(u64,usage.device,usage.host);const room=pool.room(held);
    const prior_limit=try cuda.memory.setAllocationLimit(try std.math.add(u64,held,room));defer if(!retained){_=cuda.memory.setAllocationLimit(prior_limit) catch {};};
    role.runExpert(a,init.io,&gpu.ctx,cli.checkpoint,c,shape,&stopping,until,.{.room=room,.unified=pool.unified,.lifecycle=&lifecycle}) catch |err| {if(err==error.HybridRetained) retained=true;return err;};
}

// CPU only: real argv/parser/metadata path up to the driver-open seam.
test "I3 expert CLI: selected observer is installed before driver and clean startup error publishes final ownership" {
    const a=t.allocator;var tmp=t.tmpDir(.{});defer tmp.cleanup();const text=try glm.config.textConfig(a,1,0,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const shape=try glm.config.parse(a,config);
    var header=@import("glm_hybrid_memory_fixture.zig").Header{.a=a};defer header.text.deinit(a);try header.expert(shape,true);try header.write(tmp.dir,"");
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);const dirz=try a.dupeSentinel(u8,dir,0);defer a.free(dirz);const digest=std.fmt.bytesToHex(try glm.ranks.artifact(a,t.io,dir),.lower);
    const h=glm.moe.hybridConfig.Config{.revision=3,.role=.expert1,.session="00000000000000000000000000000001",.model_sha256=&digest,.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="/fixture/NEVER_OPENED",.setup_ms=1000,.transaction_ms=1000,.close_ms=1000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{}};
    const ht=try std.json.Stringify.valueAlloc(a,h,.{});defer a.free(ht);try tmp.dir.writeFile(t.io,.{.sub_path="role.json",.data=ht});const path=try tmp.dir.realPathFileAlloc(t.io,"role.json",a);defer a.free(path);const pathz=try a.dupeSentinel(u8,path,0);defer a.free(pathz);
    const Probe=struct{var final:?cuda.memory_owner.Ownership=null;var phases:[5]cuda.memory_receipt.Phase=undefined;var count:usize=0;
        fn before()!void{const owned=cuda.memory_owner.snapshot() orelse return error.MissingObserver;try t.expectEqualStrings("expert1",owned.role);try t.expect(owned.incarnation!=0 and !owned.closed);return error.CpuDriverBoundary;}
        fn sink(r:cuda.memory_receipt.Receipt)void{phases[count]=r.phase;count+=1;if(r.phase==.cleanup_end)final=r.ownership;}
    };
    Probe.count=0;Probe.final=null;cuda.memory_receipt.Test.sink=Probe.sink;defer cuda.memory_receipt.Test.sink=null;MemoryTest.before_driver=Probe.before;defer MemoryTest.before_driver=null;
    var arena=std.heap.ArenaAllocator.init(a);defer arena.deinit();const argv=[_][*:0]const u8{"tf-glm-expert","--config",pathz.ptr,"--checkpoint",dirz.ptr,"--device","0"};
    try t.expectError(error.CpuDriverBoundary,runMain(.{.gpa=a,.arena=&arena,.io=t.io,.minimal=.{.environ=undefined,.args=.{.vector=&argv}},.environ_map=undefined,.preopens=undefined}));
    try t.expect(cuda.memory_owner.snapshot()==null);try t.expect(Probe.final!=null);try t.expect(Probe.final.?.clean and Probe.final.?.closed);
    try t.expectEqualSlices(cuda.memory_receipt.Phase,&.{.load_begin,.cleanup_begin,.cleanup_end},Probe.phases[0..Probe.count]);
}

// Helios: real TERM at the actual CLI hash boundary; CUDA remains fenced.
test "I3 expert CLI: TERM before checkpoint identity is latched before any driver admission" {
    const a=t.allocator;var tmp=t.tmpDir(.{});defer tmp.cleanup();const text=try glm.config.textConfig(a,1,0,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const shape=try glm.config.parse(a,config);
    var header=@import("glm_hybrid_memory_fixture.zig").Header{.a=a};defer header.text.deinit(a);try header.expert(shape,true);try header.write(tmp.dir,"");
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);const dirz=try a.dupeSentinel(u8,dir,0);defer a.free(dirz);const digest=std.fmt.bytesToHex(try glm.ranks.artifact(a,t.io,dir),.lower);
    const h=glm.moe.hybridConfig.Config{.revision=3,.role=.expert1,.session="00000000000000000000000000000001",.model_sha256=&digest,.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="/fixture/NEVER_OPENED",.setup_ms=1000,.transaction_ms=1000,.close_ms=1000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{}};
    const ht=try std.json.Stringify.valueAlloc(a,h,.{});defer a.free(ht);try tmp.dir.writeFile(t.io,.{.sub_path="role.json",.data=ht});const path=try tmp.dir.realPathFileAlloc(t.io,"role.json",a);defer a.free(path);const pathz=try a.dupeSentinel(u8,path,0);defer a.free(pathz);
    const Probe=struct {
        var prior_handler:bool=false;
        var driver_calls:usize=0;
        fn sentinel(_:std.posix.SIG)callconv(.c)void {prior_handler=true;}
        fn deliver()!void {try std.posix.raise(.TERM);}
        fn before()!void {driver_calls+=1;return error.CpuDriverBoundary;}
    };
    Probe.prior_handler=false;Probe.driver_calls=0;stopping.store(false,.release);defer stopping.store(false,.release);
    const sentinel:std.posix.Sigaction=.{.handler=.{.handler=Probe.sentinel},.mask=std.posix.sigemptyset(),.flags=0};var prior:std.posix.Sigaction=undefined;std.posix.sigaction(.TERM,&sentinel,&prior);defer std.posix.sigaction(.TERM,&prior,null);
    MemoryTest.before_identity=Probe.deliver;defer MemoryTest.before_identity=null;MemoryTest.before_driver=Probe.before;defer MemoryTest.before_driver=null;
    var arena=std.heap.ArenaAllocator.init(a);defer arena.deinit();const argv=[_][*:0]const u8{"tf-glm-expert","--config",pathz.ptr,"--checkpoint",dirz.ptr,"--device","0"};
    try t.expectError(error.Stopped,runMain(.{.gpa=a,.arena=&arena,.io=t.io,.minimal=.{.environ=undefined,.args=.{.vector=&argv}},.environ_map=undefined,.preopens=undefined}));
    try t.expectEqual(@as(usize,0),Probe.driver_calls);
    try t.expect(!Probe.prior_handler);
    try t.expect(cuda.memory_owner.snapshot()==null);
}

// Helios: real TERM inside the real full-identity worker must not publish a digest.
test "I3 expert CLI: TERM inside actual hash worker aborts before digest publication and driver" {
    const a=t.allocator;var tmp=t.tmpDir(.{});defer tmp.cleanup();const text=try glm.config.textConfig(a,1,0,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const shape=try glm.config.parse(a,config);
    var header=@import("glm_hybrid_memory_fixture.zig").Header{.a=a};defer header.text.deinit(a);try header.expert(shape,true);try header.write(tmp.dir,"");
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);const dirz=try a.dupeSentinel(u8,dir,0);defer a.free(dirz);const digest=std.fmt.bytesToHex(try glm.ranks.artifact(a,t.io,dir),.lower);
    const h=glm.moe.hybridConfig.Config{.revision=3,.role=.expert1,.session="00000000000000000000000000000001",.model_sha256=&digest,.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="/fixture/NEVER_OPENED",.setup_ms=1000,.transaction_ms=1000,.close_ms=1000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{}};
    const ht=try std.json.Stringify.valueAlloc(a,h,.{});defer a.free(ht);try tmp.dir.writeFile(t.io,.{.sub_path="role.json",.data=ht});const path=try tmp.dir.realPathFileAlloc(t.io,"role.json",a);defer a.free(path);const pathz=try a.dupeSentinel(u8,path,0);defer a.free(pathz);
    const Probe=struct {
        var sent:std.atomic.Value(bool)=.init(false);
        var completions:std.atomic.Value(usize)=.init(0);
        var later_blocks:std.atomic.Value(usize)=.init(0);
        var driver_calls:usize=0;
        fn block(at:u64)!void {
            if(!sent.swap(true,.acq_rel)) try std.posix.raise(.TERM);
            if(at>0) _=later_blocks.fetchAdd(1,.monotonic);
        }
        fn success()!void {_=completions.fetchAdd(1,.monotonic);}
        fn before()!void {driver_calls+=1;return error.CpuDriverBoundary;}
    };
    Probe.sent.store(false,.release);Probe.completions.store(0,.release);Probe.later_blocks.store(0,.release);Probe.driver_calls=0;
    stopping.store(false,.release);defer stopping.store(false,.release);
    glm.ranks.IdentityTest.before_block=Probe.block;defer glm.ranks.IdentityTest.before_block=null;
    glm.ranks.IdentityTest.before_success=Probe.success;defer glm.ranks.IdentityTest.before_success=null;
    MemoryTest.before_driver=Probe.before;defer MemoryTest.before_driver=null;
    var arena=std.heap.ArenaAllocator.init(a);defer arena.deinit();const argv=[_][*:0]const u8{"tf-glm-expert","--config",pathz.ptr,"--checkpoint",dirz.ptr,"--device","0"};
    try t.expectError(error.Stopped,runMain(.{.gpa=a,.arena=&arena,.io=t.io,.minimal=.{.environ=undefined,.args=.{.vector=&argv}},.environ_map=undefined,.preopens=undefined}));
    try t.expect(Probe.sent.load(.acquire));
    try t.expectEqual(@as(usize,0),Probe.completions.load(.acquire));
    try t.expectEqual(@as(usize,0),Probe.later_blocks.load(.acquire));
    try t.expectEqual(@as(usize,0),Probe.driver_calls);
    try t.expect(cuda.memory_owner.snapshot()==null);
}

fn processExitCase(mode:u2)!void {
    const a=t.allocator;var tmp=t.tmpDir(.{});defer tmp.cleanup();const text=try glm.config.textConfig(a,1,0,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const shape=try glm.config.parse(a,config);
    var header=@import("glm_hybrid_memory_fixture.zig").Header{.a=a};defer header.text.deinit(a);try header.expert(shape,true);try header.write(tmp.dir,"");
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);const dirz=try a.dupeSentinel(u8,dir,0);defer a.free(dirz);const digest=std.fmt.bytesToHex(try glm.ranks.artifact(a,t.io,dir),.lower);
    const h=glm.moe.hybridConfig.Config{.revision=3,.role=.expert1,.session="00000000000000000000000000000001",.model_sha256=&digest,.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="/fixture/NEVER_OPENED",.setup_ms=1000,.transaction_ms=1000,.close_ms=1000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{}};
    const ht=try std.json.Stringify.valueAlloc(a,h,.{});defer a.free(ht);try tmp.dir.writeFile(t.io,.{.sub_path="role.json",.data=ht});const path=try tmp.dir.realPathFileAlloc(t.io,"role.json",a);defer a.free(path);const pathz=try a.dupeSentinel(u8,path,0);defer a.free(pathz);

    // The old boundary tests retain their internal Stopped contract; this calls the public process entry.
    const Probe=struct {
        fn sentinel(_:std.posix.SIG)callconv(.c)void{}
        fn term()!void{try std.posix.raise(.TERM);}
        fn fence()!void{return error.CpuDriverBoundary;}
        fn cancelled()!void{try std.posix.raise(.TERM);return error.Cancelled;}
        fn retained()!void{try std.posix.raise(.TERM);return error.HybridRetained;}
    };
    stopping.store(false,.release);defer stopping.store(false,.release);
    const action:std.posix.Sigaction=.{.handler=.{.handler=Probe.sentinel},.mask=std.posix.sigemptyset(),.flags=0};var prior:std.posix.Sigaction=undefined;var prior_int:std.posix.Sigaction=undefined;
    std.posix.sigaction(.TERM,&action,&prior);defer std.posix.sigaction(.TERM,&prior,null);std.posix.sigaction(.INT,&action,&prior_int);defer std.posix.sigaction(.INT,&prior_int,null);
    MemoryTest.before_identity=if(mode==0)Probe.term else null;defer MemoryTest.before_identity=null;
    MemoryTest.before_driver=switch(mode){0=>Probe.fence,1=>Probe.cancelled,else=>Probe.retained};defer MemoryTest.before_driver=null;
    var arena=std.heap.ArenaAllocator.init(a);defer arena.deinit();const argv=[_][*:0]const u8{"tf-glm-expert","--config",pathz.ptr,"--checkpoint",dirz.ptr,"--device","0"};
    const init:std.process.Init=.{.gpa=a,.arena=&arena,.io=t.io,.minimal=.{.environ=undefined,.args=.{.vector=&argv}},.environ_map=undefined,.preopens=undefined};
    if(mode==2)try t.expectError(error.HybridRetained,main(init))else try main(init);
    try t.expect(stopping.load(.acquire));try t.expect(cuda.memory_owner.snapshot()==null);
}
test "I3 expert CLI: public process entry returns cleanly after TERM before identity" {try processExitCase(0);}
test "I3 expert CLI: public process entry maps clean Cancelled only with the same latch" {try processExitCase(1);}
test "I3 expert CLI: public process entry never masks retained ownership after TERM" {try processExitCase(2);}
