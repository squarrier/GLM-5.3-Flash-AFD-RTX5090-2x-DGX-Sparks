//! WIRE4: phase-batched fixed-endian frames. WIRE3/2 are never reinterpreted; tensor layout ABI stays1.
const std=@import("std");
const routed=@import("cuda_routed.zig");
pub const version:u16=4;
pub const post_bytes: usize=1048576; // native HCA safety ceiling: never raised
pub const frame_payload: usize=post_bytes-256;
pub const tile_rows: usize=128;
pub const request_bytes: usize=tile_rows*4096*2+tile_rows*8*4;
pub const reply_bytes: usize=tile_rows*8*4096*4;
/// Owned staging only; native slots/posts remain post_bytes, not batch_bytes.
pub const batch_bytes:usize=@max(fragments(request_bytes),fragments(reply_bytes))*post_bytes;
// The native lane-price consumer names the selected V3 device dataflow, not the legacy FrameIO codec.
// Cost identity epoch only: protocol versions/layout and 1MiB posts are unchanged.
pub const cost_identity="v3dev4/post1048576/native9/bf16x-fp32w-fp32reply/g53/lanes4/pp2/bmtp/half1024x2";
/// Used by the actual lane-cost shape consumer; inherited geometry prices must miss.
pub fn bindCostShape(base:[]const u8,storage:[]u8)![]const u8 {
    if(base.len>storage.len or @intFromPtr(base.ptr)!=@intFromPtr(storage.ptr))return error.BadCacheBuffer;
    const suffix=try std.fmt.bufPrint(storage[base.len..]," {s}",.{cost_identity});
    return storage[0..base.len+suffix.len];
}
pub const Kind=enum(u16) {idle,request,reply,cancel,close,err};
pub const Status=enum(u32) {ok,cancelled,invalid,failed,unsupported};
pub const Mode=enum(u32) {decode,prompt};
const fields=.{
    .{"magic",0,u32},.{"wire_version",4,u16},.{"kind",6,u16},.{"header_bytes",8,u32},.{"flags",12,u32},
    .{"session",16,u128},.{"sequence",32,u64},.{"native_round",40,u64},.{"buffer_epoch",48,u64},.{"layer",56,u32},.{"phase",60,u16},.{"half",62,u16},
    .{"live_rows",64,u32},.{"hidden",68,u32},.{"top_k",72,u32},.{"packed_input_stride_elements",76,u32},.{"total_payload_bytes",80,u64},.{"status",88,u32},.{"layout_abi",92,u32},
    .{"fragment_index",224,u32},.{"fragment_count",228,u32},.{"payload_offset",232,u64},.{"fragment_payload_bytes",240,u32},.{"execution_mode",244,u32},.{"tile_row_lo",248,u32},.{"call_rows",252,u32},
};
pub const Header=struct {
    magic: u32=0x0AFD4D31,wire_version: u16=version,kind: Kind=.request,header_bytes: u32=256,flags: u32=0,
    session: u128=0,sequence: u64=0,native_round: u64=0,buffer_epoch: u64=0,layer: u32=0,phase: routed.Phase=.target,half: u16=0,
    live_rows: u32=0,hidden: u32=4096,top_k: u32=8,packed_input_stride_elements: u32=4096,total_payload_bytes: u64=0,status: Status=.ok,layout_abi: u32=1,
    model_contract_sha256: [32]u8=@splat(0),row_map_sha256: [32]u8=@splat(0),settings_sha256: [32]u8=@splat(0),call_binding_sha256: [32]u8=@splat(0),
    fragment_index: u32=0,fragment_count: u32=1,payload_offset: u64=0,fragment_payload_bytes: u32=0,execution_mode: Mode=.decode,tile_row_lo: u32=0,call_rows: u32=0,
    pub fn encode(h: Header,out: []u8) void {
        std.debug.assert(out.len==256);@memset(out,0);
        inline for(fields) |f| {
            const v=@field(h,f[0]);
            std.mem.writeInt(f[2],out[f[1]..][0..@sizeOf(f[2])],if(@typeInfo(@TypeOf(v))==.@"enum") @intFromEnum(v) else v,.little);
        }
        @memcpy(out[96..128],&h.model_contract_sha256);@memcpy(out[128..160],&h.row_map_sha256);
        @memcpy(out[160..192],&h.settings_sha256);@memcpy(out[192..224],&h.call_binding_sha256);
    }
    pub fn decode(bytes: []const u8) !Header {
        if(bytes.len!=256) return error.BadHeader;
        var h: Header=.{};
        inline for(fields) |f| {
            const n=std.mem.readInt(f[2],bytes[f[1]..][0..@sizeOf(f[2])],.little);
            const T=@TypeOf(@field(h,f[0]));
            @field(h,f[0])=if(@typeInfo(T)==.@"enum") std.enums.fromInt(T,n) orelse return error.BadHeader else n;
        }
        @memcpy(&h.model_contract_sha256,bytes[96..128]);@memcpy(&h.row_map_sha256,bytes[128..160]);
        @memcpy(&h.settings_sha256,bytes[160..192]);@memcpy(&h.call_binding_sha256,bytes[192..224]);
        if(h.magic!=0x0AFD4D31 or h.header_bytes!=256 or h.flags!=0 or h.layout_abi!=1) return error.BadHeader;
        if(h.wire_version!=version) return error.BadWireVersion;
        if(h.hidden!=4096 or h.top_k!=8 or h.packed_input_stride_elements!=4096 or h.half>1 or h.live_rows==0 or h.live_rows>128 or h.tile_row_lo>h.call_rows or h.live_rows>h.call_rows-h.tile_row_lo or h.session==0 or h.sequence==0 or h.fragment_count==0 or h.fragment_index>=h.fragment_count) return error.BadHeader;
        var bound=h;bound.bind();if(!std.mem.eql(u8,&bound.call_binding_sha256,&h.call_binding_sha256)) return error.BindingMismatch;
        return h;
    }
    pub fn bind(h: *Header) void {
        var hash=std.crypto.hash.sha2.Sha256.init(.{});
        inline for(.{"session","sequence","native_round","buffer_epoch","layer","phase","half","live_rows","hidden","top_k","execution_mode","tile_row_lo","call_rows"}) |name| {
            const value=@field(h.*,name);const T=@TypeOf(value);const I=if(@typeInfo(T)==.@"enum") @typeInfo(T).@"enum".tag_type else T;
            var bytes: [@sizeOf(I)]u8=undefined;std.mem.writeInt(I,&bytes,if(@typeInfo(T)==.@"enum") @intFromEnum(value) else value,.little);hash.update(&bytes);
        }
        hash.update(&h.row_map_sha256);hash.update(&h.model_contract_sha256);hash.update(&h.settings_sha256);hash.final(&h.call_binding_sha256);
    }
};
pub fn fragments(total: usize) usize {return (total+frame_payload-1)/frame_payload;}
pub fn fragment(base: Header,kind: Kind,total: usize,index: usize) Header {
    var h=base;h.kind=kind;h.total_payload_bytes=total;h.fragment_index=@intCast(index);h.fragment_count=@intCast(fragments(total));h.payload_offset=index*frame_payload;
    h.fragment_payload_bytes=if(kind==.idle) 0 else @intCast(@min(frame_payload,total-h.payload_offset));return h;
}
pub fn require(bytes: []const u8,want: Header) !void {
    const got=try Header.decode(bytes);if(got.status!=.ok) return error.PeerFailed;
    var encoded: [256]u8=undefined;want.encode(&encoded);
    if(!std.mem.eql(u8,bytes,&encoded)) return error.BindingMismatch;
}
