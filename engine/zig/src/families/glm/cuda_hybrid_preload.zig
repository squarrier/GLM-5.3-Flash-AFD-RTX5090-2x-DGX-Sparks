// SPDX-License-Identifier: Apache-2.0
//! Header-only EXL3 attention preflight. Source schemas: native cuda_weights/cuda_mtp_load,
//! ashhart/TensorFold; EXL3 recipe Mia-AiLab. Adapter: Helios/OpenAI Codex for Scott Quarrier.
const std=@import("std");const core=@import("core");const model=@import("config.zig");const cfg=@import("cuda_hybrid_config.zig");const m=@import("cuda_hybrid_memory.zig");
const st=core.safetensors;const P="model.language_model.";
pub const Plan=struct {role:cfg.Role,initial:m.WeightTotals=. {},resident:m.WeightTotals=. {},resident_upper_bound:m.WeightTotals=. {},load_peak:u64=0,draft_scratch:u64=0,tiling_scratch:u64=0,required_tensors:usize=0,header_binding:[32]u8,scope:[]const u8="selected_weight_peak_bound_not_whole_process_admission",driver_private:?u64=null,verbs_private:?u64=null,heap_and_checkpoint_residency:?u64=null};
// Same bounded metadata-only read as native Metadata.read, with its consuming
// language/head namespace filter applied BEFORE dtype/rank interpretation.
fn readMetadata(a:std.mem.Allocator,io:std.Io,path:[]const u8)!st.Metadata{
    const file=try std.Io.Dir.cwd().openFile(io,path,.{});defer file.close(io);
    const size=std.math.cast(usize,try file.length(io)) orelse return error.BadSafetensors;
    var prefix:[8]u8=undefined;if(try file.readPositionalAll(io,&prefix,0)!=8)return error.BadSafetensors;
    const n=try st.headerSize(std.mem.readInt(u64,&prefix,.little),size);var arena=std.heap.ArenaAllocator.init(a);errdefer arena.deinit();
    const text=try arena.allocator().alloc(u8,n);if(try file.readPositionalAll(io,text,8)!=n)return error.BadSafetensors;
    const names=try st.parseHeaderPrefixes(arena.allocator(),text,size-8-n,&.{P,"lm_head.weight"});
    if(try file.length(io)!=size)return error.BadSafetensors;
    return .{.names=names,.data=8+n,.file_size=size,.arena=arena};
}
fn add(a:u64,b:u64) !u64{return std.math.add(u64,a,b);}fn mul(a:u64,b:u64) !u64{return std.math.mul(u64,a,b);}
fn charge(t:*m.WeightTotals,class:m.WeightClass,n:u64) !void {switch(class){.attention=>t.attention=try add(t.attention,n),.dense=>t.dense=try add(t.dense,n),.shared=>t.shared=try add(t.shared,n),.router=>t.router=try add(t.router,n),.embedding_head=>t.embedding_head=try add(t.embedding_head,n)}t.total=try add(t.total,n);}
fn selected(c:model.Config,h:cfg.Config,name:[]const u8) !?m.WeightClass {
    if(std.mem.eql(u8,name,"lm_head.weight") or std.mem.eql(u8,name,P++"embed_tokens.weight") or std.mem.eql(u8,name,P++"norm.weight"))return if(h.role==.attention) .embedding_head else null;
    if(!std.mem.startsWith(u8,name,P++"layers."))return null;
    const tail=name[(P++"layers.").len..];const end=std.mem.indexOfScalar(u8,tail,'.') orelse return error.HybridTensorName;
    const i=try std.fmt.parseInt(u32,tail[0..end],10);const run=if(h.settings.layers==0)c.layers else h.settings.layers;
    if(i>=run and !(h.mtp and i==c.layers))return null;
    if(h.role!=.attention)return if(std.mem.indexOf(u8,tail,".mlp.experts.")!=null) .attention else null;
    if(std.mem.indexOf(u8,tail,".mlp.experts.")!=null)return null;
    if(std.mem.indexOf(u8,tail,".mlp.shared_experts.")!=null)return .shared;
    if(std.mem.indexOf(u8,tail,".mlp.gate.")!=null)return .router;
    if(std.mem.indexOf(u8,tail,".mlp.")!=null)return .dense;
    if(i==c.layers and (std.mem.indexOf(u8,tail,".eh_proj.")!=null or std.mem.indexOf(u8,tail,".enorm.")!=null or std.mem.indexOf(u8,tail,".hnorm.")!=null or std.mem.indexOf(u8,tail,".shared_head.")!=null))return .embedding_head;
    return .attention;
}
fn residentSize(name:[]const u8,e:st.Entry) !u64 {
    if(e.dtype!=.bf16 and e.dtype!=.f32)return error.UnsupportedHybridWeightFormat;
    var n:u64=1;for(e.shape[0..e.rank])|d|n=try mul(n,d);
    const wide=std.mem.endsWith(u8,name,"_base") or std.mem.endsWith(u8,name,"_scale") or std.mem.endsWith(u8,name,"_conv1d.weight") or std.mem.endsWith(u8,name,".A_log") or std.mem.endsWith(u8,name,".dt_bias") or std.mem.endsWith(u8,name,".e_score_correction_bias") or (e.dtype==.f32 and (std.mem.endsWith(u8,name,"input_layernorm.weight") or std.mem.endsWith(u8,name,"post_attention_layernorm.weight") or std.mem.eql(u8,name,P++"norm.weight")));
    return mul(n,if(wide)4 else @as(u64,2));
}
fn packedSize(name:[]const u8,e:st.Entry,bytes:u64,h:cfg.Config) !u64 {
    if(!h.settings.tiled or e.rank!=2)return bytes;
    var eligible=std.mem.eql(u8,name,"lm_head.weight") or std.mem.endsWith(u8,name,".eh_proj.weight");
    if(std.mem.indexOf(u8,name,".mlp.")!=null and std.mem.indexOf(u8,name,".mlp.gate.")==null)eligible=true;
    inline for(.{"q_proj.weight","k_proj.weight","v_proj.weight","f_a_proj.weight","g_a_proj.weight","b_proj.weight","f_b_proj.weight","g_b_proj.weight","o_proj.weight","q_a_proj.weight","kv_a_proj_with_mqa.weight","q_b_proj.weight","indexer.wk.weight","indexer.weights_proj.weight","indexer.wq_b.weight"})|tail|{if(std.mem.endsWith(u8,name,".self_attn."++tail))eligible=true;}
    if(!eligible)return bytes;
    const bits=if(std.mem.eql(u8,name,"lm_head.weight"))h.settings.head_bits else h.settings.dense_bits;
    if(bits==16)return bytes;
    const elems=try mul(e.shape[0],e.shape[1]);if(e.shape[1]%64!=0)return error.HybridTensorShape;
    const codes=try mul(elems,bits)/8;return add(try mul((try add(codes,255))/256,256),try mul(elems/64,2));
}
pub fn read(a:std.mem.Allocator,io:std.Io,dir:[]const u8,h:cfg.Config) !Plan {

    var arena=std.heap.ArenaAllocator.init(a);defer arena.deinit();const g=arena.allocator();
    const path=try std.fs.path.join(g,&.{dir,"config.json"});const text=try std.Io.Dir.cwd().readFileAlloc(io,path,g,.limited(1<<24));
    var why:core.modelopt.Why=.{};try @import("cuda_exl3.zig").admit(a,text,&why);const c=try model.parse(a,text);
    if(!c.exl3 or c.hidden!=4096 or c.experts!=288 or c.topk!=8 or c.moe_inter!=2048 or h.settings.layers>c.layers or (h.mtp and c.mtp==0))return error.BadExpertGeometry;
    if(!h.settings.tiled and (h.settings.dense_bits!=16 or h.settings.head_bits!=16))return error.PackedNeedsGlmMm;
    var p=Plan{.role=h.role,.header_binding=undefined};var hash=std.crypto.hash.sha2.Sha256.init(.{});hash.update(text);
    const files=try core.checkpoint.shardFiles(g,io,dir);var names:st.Header=.empty;
    for(files)|file|{var md=try readMetadata(a,io,file);defer md.deinit();var it=md.names.iterator();while(it.next())|kv|{
        const class=(try selected(c,h,kv.key_ptr.*)) orelse continue;
        if(names.contains(kv.key_ptr.*))return error.DuplicateTensor;
        try names.put(g,try g.dupe(u8,kv.key_ptr.*),kv.value_ptr.*);
        _=class;
        hash.update(kv.key_ptr.*);hash.update(@tagName(kv.value_ptr.dtype));for(kv.value_ptr.shape[0..kv.value_ptr.rank])|d|{var b:[8]u8=undefined;std.mem.writeInt(u64,&b,d,.little);hash.update(&b);}
    }}
    if(h.role!=.attention){
        p.required_tensors=try @import("cuda_hybrid_schema.zig").experts(&names,c,h);
        const run=if(h.settings.layers==0)c.layers else h.settings.layers;var layers:u64=@intFromBool(h.mtp and c.mtp>0);for(0..run)|i|layers+=@intFromBool(c.isMoe(@intCast(i)));
        if(layers==0)return error.EmptyExpertRole;
        p.initial=try m.weightBytes(if(h.role==.expert0).expert0 else .expert1,&.{},.{.experts=c.experts,.hidden=c.hidden,.inter=c.moe_inter,.layers=layers});p.resident=p.initial;p.resident_upper_bound=p.initial;p.load_peak=p.initial.total;p.header_binding=hash.finalResult();return p;
    }
    var owned=try @import("cuda_hybrid_schema.zig").attention(g,&names,c,h);var it=owned.iterator();while(it.next())|kv|{
        const class=(try selected(c,h,kv.key_ptr.*)).?;const n=try residentSize(kv.key_ptr.*,kv.value_ptr.*);try charge(&p.initial,class,n);try charge(&p.resident,class,try packedSize(kv.key_ptr.*,kv.value_ptr.*,n,h));p.required_tensors+=1;
    }
    if(h.mtp){const nk=try mul(c.vocab,c.hidden);p.draft_scratch=try add(nk/2,try mul(nk/64,4));try charge(&p.initial,.embedding_head,p.draft_scratch);try charge(&p.resident,.embedding_head,p.draft_scratch);}
    p.resident_upper_bound=p.initial;
    // Native tiling owns a largest full-bf16 temporary and a compact destination before
    // the replaced source frees. Do not admit using only post-q4 residency.
    if(h.settings.tiled){const n=@max(c.kdaProj(),@max(2*c.dense_inter,@max(2*c.shared_inter,c.vocab)));p.tiling_scratch=try mul(try mul(n,c.hidden),2);}
    p.load_peak=try add(try add(p.initial.total,p.draft_scratch),try mul(p.tiling_scratch,2));p.header_binding=hash.finalResult();return p;
}
