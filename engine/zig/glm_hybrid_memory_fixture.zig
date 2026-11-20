// SPDX-License-Identifier: Apache-2.0
//! Independent sparse header fixture from native cuda_weights/mpt_load + upstream weights_test schemas.
//! Test data only, no model payload. Credits: TensorFold contributors; OpenAI Codex assistance for Scott.
const std=@import("std");const glm=@import("glm");const t=std.testing;
pub const Header=struct {
    a:std.mem.Allocator, text:std.ArrayList(u8)=.empty, end:u64=0, resident:u64=0,scalar_mcg:bool=false,
    pub fn put(f:*Header,name:[]const u8,dtype:[]const u8,shape:[]const usize,target_size:u64) !void {
        var n:u64=1;for(shape)|x|n*=x;const stored:u64=if(std.mem.eql(u8,dtype,"F32") or std.mem.eql(u8,dtype,"I32"))4 else 2;
        const bytes=n*stored;
        try f.text.print(f.a,"{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[",.{if(f.text.items.len==0)"{" else ",",name,dtype});
        for(shape,0..)|x,i|try f.text.print(f.a,"{s}{d}",.{if(i==0)"" else ",",x});
        try f.text.print(f.a,"],\"data_offsets\":[{d},{d}]}}",.{f.end,f.end+bytes});f.end+=bytes;f.resident+=n*target_size;
    }
    fn row(f:*Header,i:usize,tail:[]const u8,shape:[]const usize,dtype:[]const u8,target:u64) !void {
        const name=try std.fmt.allocPrint(f.a,"model.language_model.layers.{d}.{s}",.{i,tail});defer f.a.free(name);try f.put(name,dtype,shape,target);
    }
    pub fn attention(f:*Header,c:glm.config.Config,mtp:bool) !void {
        const D:usize=c.hidden;const W:usize=c.kdaWidth();const dk:usize=c.kda_dim;
        for(0..c.layers+@as(usize,@intFromBool(mtp)))|i|{
            inline for(.{"input_layernorm.weight","post_attention_layernorm.weight"})|s|try f.row(i,s,&.{D},"BF16",2);
            if(i!=c.layers)inline for(.{"attn","ffn"})|s|{try f.row(i,"hc_"++s++"_fn",&.{24,4*D},"BF16",2);try f.row(i,"hc_"++s++"_base",&.{24},"F32",4);try f.row(i,"hc_"++s++"_scale",&.{3},"F32",4);};
            if(c.kind(@intCast(i))==.kda){
                inline for(.{"q","k","v"})|s|{try f.row(i,"self_attn."++s++"_proj.weight",&.{W,D},"BF16",2);try f.row(i,"self_attn."++s++"_conv1d.weight",&.{W,1,c.conv},"BF16",4);}
                inline for(.{"f_a","g_a"})|s|try f.row(i,"self_attn."++s++"_proj.weight",&.{dk,D},"BF16",2);
                inline for(.{"f_b","g_b"})|s|try f.row(i,"self_attn."++s++"_proj.weight",&.{W,dk},"BF16",2);
                try f.row(i,"self_attn.b_proj.weight",&.{c.kda_heads,D},"BF16",2);try f.row(i,"self_attn.o_proj.weight",&.{D,W},"BF16",2);
                try f.row(i,"self_attn.A_log",&.{c.kda_heads},"F32",4);try f.row(i,"self_attn.dt_bias",&.{W},"F32",4);try f.row(i,"self_attn.o_norm.weight",&.{dk},"BF16",2);
            }else{
                const n=[_]usize{c.q_lora,c.kv_lora,c.mla_heads*c.nope,D,c.i_dim,c.i_heads,c.i_heads*c.i_dim,c.mla_heads*(c.nope+c.v_dim)};
                const k=[_]usize{D,D,c.q_lora,c.mla_heads*c.v_dim,D,D,c.q_lora,c.kv_lora};
                inline for(.{"q_a_proj","kv_a_proj_with_mqa","q_b_proj","o_proj","indexer.wk","indexer.weights_proj","indexer.wq_b","kv_b_proj"},0..)|s,j|try f.row(i,"self_attn."++s++".weight",&.{n[j],k[j]},"BF16",2);
                inline for(.{"q_a_layernorm.weight","kv_a_layernorm.weight","indexer.k_norm.weight","indexer.k_norm.bias"},0..)|s,j|{const ns=[_]usize{c.q_lora,c.kv_lora,c.i_dim,c.i_dim};try f.row(i,"self_attn."++s,&.{ns[j]},"BF16",2);}
                try f.row(i,"self_attn.indexer.index_kpool_compress_gate",&.{c.i_dim,D},"BF16",2);try f.row(i,"self_attn.indexer.index_kpool_compress_ape",&.{c.kpool,c.i_dim},"BF16",2);
            }
            const moe=i==c.layers or c.isMoe(@intCast(i));const width:usize=if(moe)c.shared_inter else c.dense_inter;
            if(moe){try f.row(i,"mlp.gate.weight",&.{c.experts,D},"BF16",2);try f.row(i,"mlp.gate.e_score_correction_bias",&.{c.experts},"BF16",4);}
            inline for(.{"gate_proj","up_proj","down_proj"},0..)|s,j|{const name=try std.fmt.allocPrint(f.a,"mlp.{s}"++s++".weight",.{if(moe)"shared_experts." else ""});defer f.a.free(name);try f.row(i,name,if(j<2)&.{width,D} else &.{D,width},"BF16",2);}
            if(i==c.layers){inline for(.{"enorm.weight","hnorm.weight","shared_head.norm.weight"})|s|try f.row(i,s,&.{D},"BF16",2);try f.row(i,"eh_proj.weight",&.{D,2*D},"BF16",2);}
        }
        try f.put("model.language_model.embed_tokens.weight","BF16",&.{c.vocab,D},2);try f.put("model.language_model.norm.weight","BF16",&.{D},2);try f.put("lm_head.weight","BF16",&.{c.vocab,D},2);
    }
    pub fn write(f:*Header,dir:std.Io.Dir,omit:[]const u8) !void {
        var text=try f.a.dupe(u8,f.text.items);defer f.a.free(text);
        // Name-preserving geometry except a single requested omission renamed away.
        if(omit.len>0){const pos=std.mem.indexOf(u8,text,omit) orelse return error.BadFixture;@memset(text[pos..][0..omit.len],'X');}
        var file=try dir.createFile(t.io,"model.safetensors",.{});defer file.close(t.io);
        var prefix:[8]u8=undefined;std.mem.writeInt(u64,&prefix,@intCast(text.len+1),.little);
        try file.writePositionalAll(t.io,&prefix,0);try file.writePositionalAll(t.io,text,8);try file.writePositionalAll(t.io,"}",8+text.len);
        try file.setLength(t.io,8+text.len+1+f.end);
    }
    pub fn expert(f:*Header,c:glm.config.Config,mtp:bool) !void {
        for(0..c.layers+@as(usize,@intFromBool(mtp)))|i|if(c.isMoe(@intCast(i))){
            for(0..c.experts)|e|inline for(.{"gate_proj","up_proj","down_proj"},0..)|proj,j|{
                const k:usize=if(j<2)c.hidden else c.moe_inter;const n:usize=if(j<2)c.moe_inter else c.hidden;
                inline for(.{"trellis","suh","svh","mcg"},0..)|tail,z|{
                    const name=try std.fmt.allocPrint(f.a,"model.language_model.layers.{d}.mlp.experts.{d}."++proj++"."++tail,.{i,e});defer f.a.free(name);
                    try f.put(name,if(z==0)"I16" else if(z==3)"I32" else "F16",switch(z){0=>&.{k/16,n/16,64},1=>&.{k},2=>&.{n},3=>if(f.scalar_mcg)&.{} else &.{1},else=>unreachable},if(z==3)4 else 2);
                }
            };
        };
    }
};
