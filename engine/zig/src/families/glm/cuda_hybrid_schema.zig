// SPDX-License-Identifier: Apache-2.0
//! Required metadata for the selected native EXL3 role, not a prefix-size census.
//! Schemas from TensorFold cuda_weights/cuda_mtp_load/cuda_exl3, retained attribution;
//! adapter by Helios/OpenAI Codex for Scott Quarrier. No payload validation is claimed.
const std=@import("std");const core=@import("core");const model=@import("config.zig");const cfg=@import("cuda_hybrid_config.zig");
const P="model.language_model.";const Header=core.safetensors.Header;
pub fn experts(names:*const Header,c:model.Config,h:cfg.Config) !usize {
    var b:[256]u8=undefined;const run=if(h.settings.layers==0)c.layers else h.settings.layers;var count:usize=0;
    for(0..run+@as(usize,@intFromBool(h.mtp)))|j|{const i:usize=if(j<run)j else c.layers;if(!c.isMoe(@intCast(i)))continue;
        for(0..c.experts)|e|inline for(.{"gate_proj","up_proj","down_proj"},0..)|proj,z|{
            const k:usize=if(z<2)c.hidden else c.moe_inter;const n:usize=if(z<2)c.moe_inter else c.hidden;
            inline for(.{"trellis","suh","svh","mcg"},0..)|tail,q|{
                const name=try std.fmt.bufPrint(&b,P++"layers.{d}.mlp.experts.{d}."++proj++"."++tail,.{i,e});const entry=names.get(name) orelse return error.MissingHybridTensor;
                const shape:[]const usize=switch(q){0=>&.{k/16,n/16,64},1=>&.{k},2=>&.{n},3=>&.{1},else=>unreachable};
                const dtype:core.safetensors.DType=if(q==0).i16 else if(q==3).i32 else .f16;
                if(entry.dtype!=dtype)return error.UnsupportedHybridWeightFormat;
                if(q==3){var elements:usize=1;for(entry.shape[0..entry.rank])|dim|elements=try std.math.mul(usize,elements,dim);if(elements!=1)return error.HybridTensorShape;}else if(entry.rank!=shape.len or !std.mem.eql(usize,entry.shape[0..entry.rank],shape))return error.HybridTensorShape;
                count+=1;
            }
        };
    }return count;
}
const Required=struct{a:std.mem.Allocator,source:*const Header,owned:Header=.empty};
fn need(names:*Required,name:[]const u8,shape:[]const usize,float32:bool) !void {
    const e=names.source.get(name) orelse return error.MissingHybridTensor;
    if(e.rank!=shape.len or !std.mem.eql(usize,e.shape[0..e.rank],shape))return error.HybridTensorShape;
    if(e.dtype!=.bf16 and !(float32 and e.dtype==.f32))return error.UnsupportedHybridWeightFormat;
    try names.owned.put(names.a,try names.a.dupe(u8,name),e);
}
pub fn attention(a:std.mem.Allocator,input:*const Header,c:model.Config,h:cfg.Config) !Header {
    var required=Required{.a=a,.source=input};const names=&required;
    var b:[256]u8=undefined;const D:usize=c.hidden;const W:usize=c.kdaWidth();const K:usize=c.kda_dim;
    const run=if(h.settings.layers==0)c.layers else h.settings.layers;
    for(0..run+@as(usize,@intFromBool(h.mtp)))|j|{
        const i:usize=if(j<run)j else c.layers;const mtp=i==c.layers;
        inline for(.{"input_layernorm.weight","post_attention_layernorm.weight"})|s|try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}."++s,.{i}),&.{D},true);
        if(!mtp)inline for(.{"attn","ffn"})|s|{
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.hc_"++s++"_fn",.{i}),&.{24,4*D},false);
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.hc_"++s++"_base",.{i}),&.{24},true);
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.hc_"++s++"_scale",.{i}),&.{3},true);
        };
        if(c.kind(@intCast(i))==.kda){
            inline for(.{"q","k","v"})|s|{try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn."++s++"_proj.weight",.{i}),&.{W,D},false);try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn."++s++"_conv1d.weight",.{i}),&.{W,1,c.conv},true);}
            inline for(.{"f_a","g_a"})|s|try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn."++s++"_proj.weight",.{i}),&.{K,D},false);
            inline for(.{"f_b","g_b"})|s|try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn."++s++"_proj.weight",.{i}),&.{W,K},false);
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn.b_proj.weight",.{i}),&.{c.kda_heads,D},false);
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn.o_proj.weight",.{i}),&.{D,W},false);
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn.A_log",.{i}),&.{c.kda_heads},true);
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn.dt_bias",.{i}),&.{W},true);
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn.o_norm.weight",.{i}),&.{K},false);
        }else{
            inline for(.{"q_a_proj","kv_a_proj_with_mqa","q_b_proj","o_proj","indexer.wk","indexer.weights_proj","indexer.wq_b","kv_b_proj"},0..)|s,k|{
                const ns=[_]usize{c.q_lora,c.kv_lora,c.mla_heads*c.nope,D,c.i_dim,c.i_heads,c.i_heads*c.i_dim,c.mla_heads*(c.nope+c.v_dim)};
                const ks=[_]usize{D,D,c.q_lora,c.mla_heads*c.v_dim,D,D,c.q_lora,c.kv_lora};
                try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn."++s++".weight",.{i}),&.{ns[k],ks[k]},false);
            }
            inline for(.{"q_a_layernorm.weight","kv_a_layernorm.weight","indexer.k_norm.weight","indexer.k_norm.bias"},0..)|s,k|{const ns=[_]usize{c.q_lora,c.kv_lora,c.i_dim,c.i_dim};try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn."++s,.{i}),&.{ns[k]},false);}
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn.indexer.index_kpool_compress_gate",.{i}),&.{c.i_dim,D},false);
            try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.self_attn.indexer.index_kpool_compress_ape",.{i}),&.{c.kpool,c.i_dim},false);
        }
        const moe=mtp or c.isMoe(@intCast(i));const width:usize=if(moe)c.shared_inter else c.dense_inter;
        if(moe){try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.mlp.gate.weight",.{i}),&.{c.experts,D},false);try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.mlp.gate.e_score_correction_bias",.{i}),&.{c.experts},true);}
        inline for(.{"gate_proj","up_proj","down_proj"},0..)|s,k|try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.mlp.{s}"++s++".weight",.{i,if(moe)"shared_experts." else ""}),if(k<2)&.{width,D} else &.{D,width},false);
        if(mtp){inline for(.{"enorm.weight","hnorm.weight","shared_head.norm.weight"})|s|try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}."++s,.{i}),&.{D},false);try need(names,try std.fmt.bufPrint(&b,P++"layers.{d}.eh_proj.weight",.{i}),&.{D,2*D},false);}
    }
    try need(names,P++"embed_tokens.weight",&.{c.vocab,D},false);try need(names,P++"norm.weight",&.{D},true);try need(names,"lm_head.weight",&.{c.vocab,D},false);
    return required.owned;
}
