//! AI-authored Q1 repair regressions; metadata/clock/work seams only, no devices or sockets.
const std=@import("std");const glm=@import("glm");const tr=glm.moe.hybridTransport;const t=std.testing;
const identity=tr.Identity{.session=1,.model=@splat(0xAA),.settings=@splat(0xBB),.transaction_timeout_ns=1000};
fn header(lo:u32,rows:u32,sequence:u64)tr.wire.Header{var h=tr.wire.Header{.session=1,.sequence=sequence,.native_round=3,.buffer_epoch=7,.layer=3,.live_rows=@min(128,rows-lo),.tile_row_lo=lo,.call_rows=rows,.model_contract_sha256=identity.model,.row_map_sha256=@splat(0xCC),.settings_sha256=identity.settings};h.bind();return h;}
const F=struct{
    h:tr.wire.Header,at:u64=100,remaining:u64=900,delay:u64=0,step:usize=0,ack:bool=false,works:usize=0,work_until:u64=0,fragments:usize=0,advance:u64=0,bad_ack:bool=false,sent_remaining:u64=0,
    fn get(ctx:*anyopaque)*F{return @ptrCast(@alignCast(ctx));}
    fn now(ctx:*anyopaque)u64{return get(ctx).at;}
    fn check(_: *anyopaque)!void{}
    fn receive(ctx:*anyopaque,bytes:[]u8,until:u64)!void{const f=get(ctx);if(f.at>=until)return error.Timeout;if(bytes.len==256){f.h.encode(bytes);f.step=1;}else{try t.expect(f.ack and f.step==1);f.at+=f.delay;std.mem.writeInt(u64,bytes[0..8],f.remaining,.little);f.step=2;}}
    fn send(ctx:*anyopaque,bytes:[]const u8,_:u64)!void{const f=get(ctx);try t.expect(bytes.len==8);try t.expectEqual(@as(u64,0x315447444255464e),std.mem.readInt(u64,bytes[0..8],.little));f.ack=true;}
    fn io(f:*F)tr.ControlIO{return .{.ctx=f,.now=now,.check=check,.recv=receive,.send=send};}
    fn senderSend(ctx:*anyopaque,bytes:[]const u8,_:u64)!void{const f=get(ctx);if(bytes.len==256){const got=try tr.wire.Header.decode(bytes);try t.expectEqualSlices(u8,&f.h.call_binding_sha256,&got.call_binding_sha256);f.step=1;}else{try t.expect(f.ack);f.sent_remaining=std.mem.readInt(u64,bytes[0..8],.little);f.step=2;}}
    fn senderRecv(ctx:*anyopaque,bytes:[]u8,_:u64)!void{const f=get(ctx);try t.expectEqual(@as(usize,1),f.step);f.ack=true;f.at+=f.delay;std.mem.writeInt(u64,bytes[0..8],if(f.bad_ack)0 else 0x315447444255464e,.little);}
    fn sender(f:*F)tr.ControlIO{return .{.ctx=f,.now=now,.check=check,.recv=senderRecv,.send=senderSend};}
    fn exchange(ctx:*anyopaque,send_bytes:[]const u8,recv_bytes:[]u8,until:u64)!void{const f=get(ctx);try t.expectEqual(f.work_until,until);f.fragments+=1;f.at+=f.advance;var h=try tr.wire.Header.decode(send_bytes[0..256]);@memset(recv_bytes,0);if(h.kind==.idle){h.kind=.request;h.fragment_payload_bytes=@intCast(@min(tr.wire.frame_payload,h.total_payload_bytes-h.payload_offset));}else{h.kind=.idle;h.fragment_payload_bytes=0;}h.encode(recv_bytes[0..256]);}
    fn frames(f:*F)tr.FrameIO{return .{.ctx=f,.now=now,.check=check,.exchange=exchange};}
    fn work(ctx:*anyopaque,_:tr.wire.Header,_:[]const u8,_:[]u8,until:u64)!void{const f=get(ctx);f.works+=1;try t.expectEqual(f.work_until,until);}
};
fn pair()tr.NativePair{var p:tr.NativePair=undefined;p.failed=null;p.call_budget=.{};p.turn_until=0;p.control=null;return p;}
test "Q1 repair: actual wake enforces nonrenewing call bound and canonical tile progression"{
    var p=pair();var f=F{.h=header(0,129,1)};
    const first=try p.readWakeOn(f.io(),identity,0,1,5000);try t.expectEqual(@as(u64,1000),p.turn_until);try p.call_budget.finish(first);
    try t.expectEqual(@as(u64,1000),p.turnDeadline(700,1000));f=.{.h=header(128,129,2),.at=700,.remaining=900};
    const last=try p.readWakeOn(f.io(),identity,0,2,5000);try t.expectEqual(@as(u64,1000),p.turn_until);try p.call_budget.finish(last);try t.expect(p.call_budget.activeUntil()==null);
    f=.{.h=header(0,1,3),.at=2000,.remaining=500};_=try p.readWakeOn(f.io(),identity,0,3,5000);try t.expectEqual(@as(u64,2500),p.turn_until);
}
test "Q1 repair: each later remainder can tighten but never extend the first bound"{
    var p=pair();var f=F{.h=header(0,257,1)};const h=try p.readWakeOn(f.io(),identity,0,1,5000);try p.call_budget.finish(h);
    f=.{.h=header(128,257,2),.at=200,.remaining=300};_=try p.readWakeOn(f.io(),identity,0,2,5000);try t.expectEqual(@as(u64,500),p.turn_until);
}
test "Q1 repair: missing later wake expires without a new full transaction allowance"{
    var p=pair();var f=F{.h=header(0,129,1)};const h=try p.readWakeOn(f.io(),identity,0,1,5000);try p.call_budget.finish(h);f=.{.h=header(128,129,2),.at=1000};
    try t.expectError(error.Timeout,p.readWakeOn(f.io(),identity,0,2,5000));try t.expect(p.failed!=null and !f.ack);
}
test "Q1 repair: malformed missing expired and oversized remaining budgets dispatch no work"{
    inline for(.{0,1001})|ns|{var p=pair();var f=F{.h=header(0,1,1),.remaining=ns};try t.expectError(error.BadHybridBudget,p.readWakeOn(f.io(),identity,0,1,5000));try t.expect(p.failed!=null);}
    var p=pair();var f=F{.h=header(0,1,1),.remaining=10,.delay=11};try t.expectError(error.Timeout,p.readWakeOn(f.io(),identity,0,1,5000));try t.expect(p.failed!=null);
    f.at=0;try t.expectError(error.Timeout,p.readWakeOn(f.io(),identity,0,1,5000));
}
test "Q1 repair: gaps overlaps changed identity noncanonical tails and midcall close refuse"{
    inline for(0..6)|kind|{var p=pair();var f=F{.h=header(0,129,1)};const h=try p.readWakeOn(f.io(),identity,0,1,5000);try p.call_budget.finish(h);var bad=header(128,129,2);switch(kind){0=>bad.tile_row_lo=127,1=>bad.tile_row_lo=0,2=>bad.native_round+=1,3=>bad.live_rows=2,4=>bad.sequence=3,5=>{bad.kind=.close;bad.tile_row_lo=0;bad.call_rows=1;},else=>unreachable}bad.bind();f=.{.h=bad};const result=p.readWakeOn(f.io(),identity,0,2,5000);try t.expect(if(result)|_|false else |_|true);try t.expect(p.failed!=null);}
}
test "Q1 repair: sender samples only after causal READY and half1 receives caller remainder"{
    var f=F{.h=header(0,1,1),.at=9_000_000_100,.delay=600};try tr.NativePair.announceOn(f.sender(),f.h,9_000_001_000);try t.expectEqual(@as(u64,300),f.sent_remaining);
    var p=pair();var r=F{.h=f.h,.at=41_000_000_500,.remaining=f.sent_remaining,.delay=100};_=try p.readWakeOn(r.io(),identity,0,1,41_000_005_000);try t.expectEqual(@as(u64,41_000_000_800),p.turn_until);
    f=.{.h=header(0,1,1),.bad_ack=true};try t.expectError(error.BadHybridBudget,tr.NativePair.announceOn(f.sender(),f.h,1000));try t.expectEqual(@as(u64,0),f.sent_remaining);
}
test "Q1 repair: actual wake to worker fragments and work share exact accepted bound"{
    const input=try t.allocator.alloc(u8,tr.wire.request_bytes);defer t.allocator.free(input);const output=try t.allocator.alloc(u8,tr.wire.reply_bytes);defer t.allocator.free(output);const tx=try t.allocator.alloc(u8,tr.wire.post_bytes);defer t.allocator.free(tx);const rx=try t.allocator.alloc(u8,tr.wire.post_bytes);defer t.allocator.free(rx);
    var p=pair();var f=F{.h=header(0,1,1),.remaining=300,.work_until=400};try t.expectEqual(tr.wire.Kind.request,try tr.workerRoleTurnOn(&p,f.io(),f.frames(),identity,0,1,input,output,tx,rx,&f,F.work,5000));try t.expectEqual(@as(usize,1),f.works);try t.expectEqual(tr.wire.fragments(input.len)+tr.wire.fragments(output.len),f.fragments);try t.expect(p.call_budget.activeUntil()==null);
    p=pair();f=.{.h=header(0,1,1),.remaining=300,.work_until=400,.advance=200};try t.expectError(error.Timeout,tr.workerRoleTurnOn(&p,f.io(),f.frames(),identity,0,1,input,output,tx,rx,&f,F.work,5000));try t.expectEqual(@as(usize,0),f.works);try t.expect(p.failed!=null and p.call_budget.pending);
}
