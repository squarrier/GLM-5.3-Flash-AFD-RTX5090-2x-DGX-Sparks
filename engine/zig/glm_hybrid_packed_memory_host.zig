// SPDX-License-Identifier: Apache-2.0
const std=@import("std");const t=std.testing;const glm=@import("glm");
test "I3 packed preload: native dense and shared post-tile bytes are separate from uncompressed load peak" {
    const a=t.allocator;var tmp=t.tmpDir(.{});defer tmp.cleanup();
    const text=try glm.config.textConfig(a,1,1,&.{},288,128,4096);defer a.free(text);
    const config=try std.mem.replaceOwned(u8,a,text,"{\"model_type\"","{\"quantization_config\":{\"quant_method\":\"exl3\",\"bits\":4,\"codebook\":\"mcg\"},\"model_type\"");defer a.free(config);
    try tmp.dir.writeFile(t.io,.{.sub_path="config.json",.data=config});const c=try glm.config.parse(a,config);
    var f=@import("glm_hybrid_memory_fixture.zig").Header{.a=a};defer f.text.deinit(a);try f.attention(c,true);try f.write(tmp.dir,"");
    const dir=try tmp.dir.realPathFileAlloc(t.io,".",a);defer a.free(dir);
    const h=glm.moe.hybridConfig.Config{.revision=3,.role=.attention,.session="00000000000000000000000000000001",.model_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="/fixture/NEVER_OPENED",.setup_ms=1000,.transaction_ms=1000,.close_ms=1000,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{.tiled=true,.dense_bits=4,.head_bits=16}};
    const p=try glm.hybrid_preload.read(a,t.io,dir,h);
    try t.expect(@hasField(@TypeOf(p),"resident"));
    if(comptime @hasField(@TypeOf(p),"resident")){
        const dense:u64=3*4096*12288;const shared:u64=3*4096*2048;
        try t.expectEqual(dense*2,p.initial.dense);try t.expectEqual(dense/2+dense/64*2,p.resident.dense);
        try t.expectEqual(shared*2,p.initial.shared);try t.expectEqual(shared/2+shared/64*2,p.resident.shared);
        try t.expect(p.load_peak>p.initial.total and p.initial.total>p.resident.total);
        try t.expectEqual(p.initial.total,p.resident_upper_bound.total);
    }
}
