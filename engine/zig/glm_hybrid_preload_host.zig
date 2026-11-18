// SPDX-License-Identifier: Apache-2.0
const std=@import("std");const t=std.testing;const glm=@import("glm");const cuda=@import("cuda");const fm=@import("src/native/cuda_family_memory.zig");
extern "c" fn setenv([*:0]const u8,[*:0]const u8,c_int) c_int;
extern "c" fn unsetenv([*:0]const u8) c_int;
test "I3 preload: actual GLM selected metadata refusal cannot fall back to full world" {
    try t.expectEqual(@as(c_int,0),setenv("TF_GLM_HYBRID_CONFIG","/fixture/MISSING_SELECTOR",1));defer _=unsetenv("TF_GLM_HYBRID_CONFIG");
    var why:[]const u8="";defer if(why.len>0)t.allocator.free(why);
    try t.expectError(error.FileNotFound,fm.weights(glm.native,t.allocator,t.io,.{.dir="/fixture/NEVER_LOADED",.model_type="glm5_next",.context=262144,.drafts=true,.kv8=false},&why));
}

fn selected() glm.moe.hybridConfig.Config {return .{.revision=3,.role=.attention,.session="00000000000000000000000000000001",.model_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="/fixture/NEVER_OPENED",.setup_ms=1000,.transaction_ms=1000,.close_ms=1000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{.tiled=false}};}
fn trial(omit:[]const u8,expected:?anyerror,engine:bool,foreign:bool) !void {
    return trialPaged(omit,expected,engine,foreign,false);
}
fn trialPaged(omit:[]const u8,expected:?anyerror,engine:bool,foreign:bool,paged:bool) !void {
    var tmp=t.tmpDir(.{});defer tmp.cleanup();const a=t.allocator;
    const text=try glm.config.textConfig(a,1,0,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);
    try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const c=try glm.config.parse(a,config);
    var f=@import("glm_hybrid_memory_fixture.zig").Header{.a=a};defer f.text.deinit(a);try f.attention(c,true);const resident=f.resident;
    if(foreign)try f.put("model.vision_tower.unrelated_layout","FUTURE_VISION_LAYOUT",&.{1,1,1,1,1,1},2);
    try f.write(tmp.dir,omit);
    var h=selected();h.settings.pages=paged;const htext=try std.json.Stringify.valueAlloc(a,h,.{});defer a.free(htext);try tmp.dir.writeFile(t.io,.{.sub_path="hybrid.json",.data=htext});
    const path=try tmp.dir.realPathFileAlloc(t.io,"hybrid.json",a);defer a.free(path);const z=try a.dupeSentinel(u8,path,0);defer a.free(z);
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);
    inline for(.{.{"TF_GLM_RANK","0"},.{"TF_GLM_WORLD","1"},.{"TF_GLM_MM","0"}})|kv|{try t.expectEqual(@as(c_int,0),setenv(kv[0],kv[1],1));}
    defer { _=unsetenv("TF_GLM_RANK");_=unsetenv("TF_GLM_WORLD");_=unsetenv("TF_GLM_MM");_=unsetenv("TF_GLM_HYBRID_CONFIG");}
    try t.expectEqual(@as(c_int,0),setenv("TF_GLM_HYBRID_CONFIG",z,1));
    try t.expectEqual(@as(c_int,0),setenv("TF_KV_PAGES",if(paged)"1" else "0",1));defer _=unsetenv("TF_KV_PAGES");
    var why:[]const u8="";defer if(why.len>0)a.free(why);
    const raw_draft=@as(u64,c.vocab)*c.hidden/16*9;
    if(engine){
        const Denied=struct{var calls:usize=0;var trace=false;var d:?*cuda.Driver=null;var ledger:?*glm.hybrid_ledger.Ledger=null;var workspace:u64=0;var unclassified:u64=0;
            fn stream(_: *cuda.abi.Stream,_:c_uint) callconv(.c) c_int{calls+=1;if(trace){var b=cuda.DeviceBuffer.alloc(d.?,16) catch return cuda.abi.error_not_ready;defer b.free();const status=ledger.?.status();workspace=status.owners[@intFromEnum(glm.hybrid_memory.Owner.workspace)].device;unclassified=status.unclassified_buffers;}return cuda.abi.error_not_ready;}
        };
        Denied.calls=0;Denied.trace=expected.?==error.NotReady;Denied.workspace=0;Denied.unclassified=0;var l=glm.hybrid_ledger.Ledger.init(.attention,931);try l.install();defer l.uninstall();Denied.ledger=&l;
        var d:cuda.Driver=undefined;const e=try glm.hybrid_cpu_fixture.F.engine(&d);defer e.deinit();Denied.d=&d;d.api.cuStreamCreate=Denied.stream;var ctx:cuda.Context=undefined;var lib:cuda.nccl.Library=undefined;
        const loaded=glm.engine.Engine.init(a,t.io,&d,&ctx,dir,&lib,null,0,1,.{.hybrid=&h,.capacity=glm.native.capacity(h.served_context),.mtp=true,.settings=h.settings});
        try t.expectError(expected.?,loaded);try t.expectEqual(@as(usize,if(Denied.trace)1 else 0),Denied.calls);
        if(Denied.trace){try t.expectEqual(@as(u64,16),Denied.workspace);try t.expectEqual(@as(u64,0),Denied.unclassified);}
    }else{
        const got=fm.weights(glm.native,a,t.io,.{.dir=dir,.model_type="glm5_next",.context=262144,.drafts=true,.kv8=false},&why);
        if(expected)|err|try t.expectError(err,got) else try t.expectEqual(@as(?u64,resident+2*raw_draft),try got);
    }
}
test "I3 preload: actual generic GLM consumer prices attention plus native MTP not shard length" {try trial("",null,false,false);}
test "I3 preload: actual generic consumer resolves BF16 pages before contract fingerprint" {try trialPaged("",null,false,false,true);}
test "I3 preload: native namespace filter ignores unrelated unsupported vision metadata" {try trial("",null,false,true);}
test "I3 preload: a missing native router source refuses rather than lowering its price" {
    inline for(.{"model.language_model.layers.0.mlp.gate.weight","model.language_model.layers.0.hc_attn_scale","model.language_model.layers.0.self_attn.q_proj.weight","model.language_model.layers.1.self_attn.indexer.index_kpool_compress_ape","model.language_model.layers.1.eh_proj.weight","lm_head.weight"})|name|try trial(name,error.MissingHybridTensor,false,false);
}
test "I3 engine preload: actual selected init refuses missing owner before stream or payload load" {try trial("model.language_model.layers.0.mlp.gate.weight",error.MissingHybridTensor,true,false);}

test "I3 preload: actual attention Engine constructor scopes remaining runtime owners" {try trial("",error.NotReady,true,false);}
