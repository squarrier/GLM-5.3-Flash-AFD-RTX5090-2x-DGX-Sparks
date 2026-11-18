// SPDX-License-Identifier: Apache-2.0
const std=@import("std");const t=std.testing;const glm=@import("glm");
test "I3 expert preload: both channel halves count every expert and MTP once from required headers" {try expertTrial(false,false);}
test "I3 expert preload: canonical scalar mcg markers match native one-element semantics" {try expertTrial(true,false);}
test "I3 expert native budget refuses weight, reserve, cap and workspace shortfall before a stream" {try expertTrial(false,true);}
test "I3 expert entry rejects missing selected header in actual pre-driver modelConfig" {try entryTrial(false);}
test "I3 expert worker entry rejects missing selected header before its first stream" {try entryTrial(true);}
fn entryTrial(worker:bool)!void {
    const a=t.allocator;var tmp=t.tmpDir(.{});defer tmp.cleanup();const text=try glm.config.textConfig(a,1,0,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const c=try glm.config.parse(a,config);
    var f=@import("glm_hybrid_memory_fixture.zig").Header{.a=a};defer f.text.deinit(a);try f.expert(c,true);try f.write(tmp.dir,"model.language_model.layers.1.mlp.experts.287.down_proj.mcg");
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);const digest=std.fmt.bytesToHex(try glm.ranks.artifact(a,t.io,dir),.lower);
    const h=glm.moe.hybridConfig.Config{.revision=3,.role=.expert1,.session="00000000000000000000000000000001",.model_sha256=&digest,.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="/fixture/NEVER_OPENED",.setup_ms=1000,.transaction_ms=1000,.close_ms=1000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{}};
    if(worker){
        const cuda=@import("cuda");const F=struct{var calls:usize=0;fn create(_: *cuda.abi.Stream,_:c_uint)callconv(.c)c_int{calls+=1;return cuda.abi.error_not_ready;}};
        F.calls=0;var d:cuda.Driver=undefined;d.api.cuStreamCreate=F.create;var ctx:cuda.Context=undefined;ctx.d=&d;var stop=std.atomic.Value(bool).init(false);
        var l=glm.hybrid_ledger.Ledger.init(.expert1,1);try l.install();defer l.uninstall();
        const result=glm.moe.hybridForward.runExpert(a,t.io,&ctx,dir,h,c,&stop,std.math.maxInt(u64),.{.room=std.math.maxInt(u64),.unified=true});try t.expectEqual(@as(usize,0),F.calls);try t.expectError(error.MissingHybridTensor,result);
    }else try t.expectError(error.MissingHybridTensor,glm.moe.hybridForward.modelConfig(a,t.io,dir,h));
}
fn expertTrial(scalar:bool,budget:bool)!void{
    const a=t.allocator;var tmp=t.tmpDir(.{});defer tmp.cleanup();
    const text=try glm.config.textConfig(a,1,0,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);
    try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const c=try glm.config.parse(a,config);
    var f=@import("glm_hybrid_memory_fixture.zig").Header{.a=a,.scalar_mcg=scalar};defer f.text.deinit(a);try f.expert(c,true);try f.write(tmp.dir,"");
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);
    var h=glm.moe.hybridConfig.Config{.revision=3,.role=.expert0,.session="00000000000000000000000000000001",.model_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="/fixture/NEVER_OPENED",.setup_ms=1000,.transaction_ms=1000,.close_ms=1000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{}};
    for([_]glm.moe.hybridConfig.Role{.expert0,.expert1})|role|{h.role=role;const p=try glm.hybrid_preload.read(a,t.io,dir,h);
        const expected:u64=288*2*(3*4096*1024/2+6*(4096+1024)+28);
        try t.expectEqual(expected,p.initial.routed);try t.expectEqual(expected,p.load_peak);try t.expectEqual(expected,p.initial.total);
        try t.expectEqual(@as(usize,288*2*3*4),p.required_tensors);
        try t.expect(p.driver_private==null and p.verbs_private==null and p.heap_and_checkpoint_residency==null);
        if(budget){
            const cuda=@import("cuda");const P=@import("src/native/cuda_memory.zig").Pool;const F=struct{var streams:usize=0;fn create(_: *cuda.abi.Stream,_:c_uint)callconv(.c)c_int{streams+=1;return cuda.abi.error_not_ready;}};
            var d:cuda.Driver=undefined;d.api.cuStreamCreate=F.create;var ctx:cuda.Context=undefined;ctx.d=&d;var stop=std.atomic.Value(bool).init(false);
            var l=glm.hybrid_ledger.Ledger.init(if(role==.expert0).expert0 else .expert1,1);try l.install();defer l.uninstall();
            for([_]P{.{.free=64,.total=1024,.reserve=0,.limit=null,.unified=true},.{.free=p.load_peak+1024,.total=p.load_peak+1024,.reserve=p.load_peak,.limit=null,.unified=true},.{.free=p.load_peak+1024,.total=p.load_peak+1024,.reserve=0,.limit=64,.unified=true},.{.free=p.load_peak,.total=p.load_peak,.reserve=0,.limit=null,.unified=true}})|pool|{
                F.streams=0;const result=glm.moe.hybridForward.runExpert(a,t.io,&ctx,dir,h,c,&stop,std.math.maxInt(u64),.{.room=pool.room(0),.unified=pool.unified});try t.expectEqual(@as(usize,0),F.streams);try t.expectError(error.HybridMemoryBudget,result);
            }
        }
    }
}
