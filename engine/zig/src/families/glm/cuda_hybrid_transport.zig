//! Bounded two-pair transaction framing. The production FrameIO is native RoCE, never a TCP/local fallback.
const std=@import("std");
const cuda=@import("cuda");
const tp=@import("tp");
const routed=@import("cuda_routed.zig");
const glue=@import("cuda_glue.zig");
pub const wire=@import("cuda_hybrid_wire.zig");
pub const FrameIO=struct {
    ctx: *anyopaque,
    announce: ?*const fn(*anyopaque,wire.Header,u64) anyerror!void=null,    now: *const fn(*anyopaque) u64,
    check: *const fn(*anyopaque) anyerror!void,
    exchange: *const fn(*anyopaque,[]const u8,[]u8,u64) anyerror!void,
    batch: ?*const fn(*anyopaque,wire.Header,wire.Kind,[]const u8,[]u8,u64) anyerror!void=null,
};
pub const Identity=struct {session: u128,model: [32]u8,settings: [32]u8,transaction_timeout_ns: u64};
/// Metadata only, on the existing bootstrap fd. Tensor frames remain native RoCE/WIRE4; native posts remain <=1MiB.
pub const ControlIO=struct {
    ctx:*anyopaque,now:*const fn(*anyopaque)u64,check:*const fn(*anyopaque)anyerror!void,
    send:*const fn(*anyopaque,[]const u8,u64)anyerror!void,
    recv:*const fn(*anyopaque,[]u8,u64)anyerror!void,
};
const budget_ready:u64=0x315447444255464e;
/// One nonrenewing local call bound, retained between its ordered tile turns.
pub const CallBudget=struct {
    first:?wire.Header=null,next_row:u32=0,last_sequence:u64=0,until:u64=0,pending:bool=false,
    pub fn activeUntil(b:*const CallBudget)?u64 {
        if(b.first)|h|if(b.pending or b.next_row<h.call_rows)return b.until;
        return null;
    }
    pub fn begin(b:*CallBudget,h:wire.Header,bound:u64)!u64 {
        if(b.pending or h.kind!=.request or h.tile_row_lo>h.call_rows or h.live_rows!=@min(wire.tile_rows,h.call_rows-h.tile_row_lo))return error.BadHybridCall;
        if(h.tile_row_lo==0) {
            if(b.activeUntil()!=null)return error.BadHybridCall;
            b.first=h;b.next_row=0;b.until=bound;
        } else {
            const first=b.first orelse return error.BadHybridCall;
            if(h.tile_row_lo!=b.next_row or h.sequence!=(std.math.add(u64,b.last_sequence,1) catch return error.BadHybridCall))return error.BadHybridCall;
            inline for(.{"session","native_round","buffer_epoch","layer","phase","half","execution_mode","call_rows","model_contract_sha256","row_map_sha256","settings_sha256"})|name|if(!std.meta.eql(@field(h,name),@field(first,name)))return error.BadHybridCall;
            b.until=@min(b.until,bound);
        }
        b.last_sequence=h.sequence;b.pending=true;return b.until;
    }
    pub fn finish(b:*CallBudget,h:wire.Header)!void {
        if(!b.pending or h.sequence!=b.last_sequence or h.tile_row_lo!=b.next_row)return error.BadHybridCall;
        b.next_row=try std.math.add(u32,h.tile_row_lo,h.live_rows);b.pending=false;
    }
};
/// One native routed provider; tiles retire only after their native ordered-FMA contraction is complete.
pub const Coordinator=struct {
    gpa: std.mem.Allocator,d: *const cuda.Driver,set: glue.Set,pairs: [2]FrameIO,identity: Identity,
    request: cuda.HostBuffer,reply: cuda.HostBuffer,frames: []u8,slots: cuda.DeviceBuffer,done: cuda.Event,
    streams: [4]cuda.Stream=undefined,stream_count: usize=0,
    next_sequence: u64=0,pending: bool=false,failed: ?anyerror=null,last_stream: ?cuda.Stream=null,active_deadline: u64=0,
    pub fn init(gpa: std.mem.Allocator,d: *const cuda.Driver,set: glue.Set,pairs: [2]FrameIO,identity: Identity) !*Coordinator {
        const memory_scope=@import("cuda_hybrid_ledger.zig").scopeOwner(.transport);defer memory_scope.end();
        if(identity.session==0 or identity.transaction_timeout_ns==0 or identity.transaction_timeout_ns>3_600_000_000_000) return error.BadHybridIdentity;
        const c=try gpa.create(Coordinator);errdefer gpa.destroy(c);
        var request=try cuda.HostBuffer.alloc(d,wire.request_bytes);errdefer request.free();
        var reply=try cuda.HostBuffer.alloc(d,wire.reply_bytes);errdefer reply.free();
        const frames=try gpa.alloc(u8,2*wire.post_bytes);errdefer gpa.free(frames);
        var slots=try cuda.DeviceBuffer.alloc(d,wire.reply_bytes);errdefer slots.free();
        const done=try cuda.Event.init(d,false);
        c.*=.{.gpa=gpa,.d=d,.set=set,.pairs=pairs,.identity=identity,.request=request,.reply=reply,.frames=frames,.slots=slots,.done=done};return c;
    }
    pub fn provider(c: *Coordinator) routed.Provider {return .{.ctx=c,.run=run,.check=check,.settle=settle,.touch=touch};}
    pub fn releaseStorage(c: *Coordinator) void {
        std.debug.assert(!c.pending and c.failed==null);
        c.done.deinit();c.slots.free();c.gpa.free(c.frames);c.reply.free();c.request.free();const g=c.gpa;g.destroy(c);
    }
    fn touch(ctx: *anyopaque,stream: cuda.Stream) !void {
        const c: *Coordinator=@ptrCast(@alignCast(ctx));try check(c);
        if(stream.d!=c.d) return error.BadHybridContext;
        for(c.streams[0..c.stream_count]) |known| if(known.handle==stream.handle) return;
        if(c.stream_count==c.streams.len) return error.TooManyHybridStreams;
        c.streams[c.stream_count]=stream;c.stream_count+=1;
    }
    fn clock(c: *const Coordinator) u64 {return c.pairs[0].now(c.pairs[0].ctx);}
    fn check(ctx: *anyopaque) !void {
        const c: *Coordinator=@ptrCast(@alignCast(ctx));if(c.failed) |err| return err;
        for(c.pairs) |p| p.check(p.ctx) catch |err| {c.failed=err;return err;};
    }
    fn wait(c: *Coordinator,stream: cuda.Stream,until: u64) !void {
        try c.done.record(stream);c.pending=true;
        while(true) {
            if(c.clock()>=until) return error.Timeout;try check(c);
            const rc=c.d.api.cuEventQuery(c.done.handle);
            if(c.clock()>=until) return error.Timeout;
            if(rc==cuda.abi.success) {c.pending=false;return;}
            if(rc!=cuda.abi.error_not_ready) return error.HybridCopyFailed;std.atomic.spinLoopHint();
        }
    }
    fn settle(ctx: *anyopaque,until: u64) !routed.Quiescence {
        const c: *Coordinator=@ptrCast(@alignCast(ctx));if(c.failed!=null) return .quarantined;
        try check(c);
        const bound=@min(until,if(c.pending) c.active_deadline else c.clock()+|c.identity.transaction_timeout_ns);
        for(c.streams[0..c.stream_count]) |stream| c.wait(stream,bound) catch |err| {c.failed=err;return .quarantined;};
        return .quiet;
    }
    fn run(ctx: *anyopaque,call: routed.Call,dest: routed.HalfDest) !routed.ReadyParts {
        const c: *Coordinator=@ptrCast(@alignCast(ctx));
        return c.runCall(call,dest) catch |err| {c.failed=err;return err;};
    }
    fn runCall(c: *Coordinator,call: routed.Call,dest: routed.HalfDest) !routed.ReadyParts {
        try check(c);
        if(c.pending or call.rows==0 or call.rows!=dest.rows or call.rows>std.math.maxInt(u32) or call.hidden!=4096 or call.top_k!=8 or call.native_slot_stride!=9 or call.x_stride_elements<4096) return error.BadHybridCall;
        const until=@min(call.deadline,c.clock()+|c.identity.transaction_timeout_ns);if(c.clock()>=until) return error.Timeout;
        try touch(c,call.stream);c.active_deadline=until;c.last_stream=call.stream;
        const first=try std.math.add(u64,c.next_sequence,1);
        var tiles: u32=0;var lo: usize=0;
        while(lo<call.rows) : (lo+=wire.tile_rows) {
            const n: usize=@min(wire.tile_rows,call.rows-lo);c.next_sequence=try std.math.add(u64,c.next_sequence,1);tiles+=1;
            try check(c);@memset(c.request.bytes,0);c.pending=true;
            const x: cuda.abi.Memcpy2D=.{.src=call.x_bf16+lo*call.x_stride_elements*2,.src_pitch=call.x_stride_elements*2,.dst_type=1,.dst_host=c.request.bytes.ptr,.dst=0,.dst_pitch=4096*2,.width=4096*2,.height=n};
            try c.d.check(c.d.api.cuMemcpy2DAsync_v2(&x,call.stream.handle),"hybrid pinned native input rows");
            const picks: cuda.abi.Memcpy2D=.{.src=call.picks_i32+lo*9*4,.src_pitch=9*4,.dst_type=1,.dst_host=c.request.bytes.ptr+wire.tile_rows*4096*2,.dst=0,.dst_pitch=8*4,.width=8*4,.height=n};
            try c.d.check(c.d.api.cuMemcpy2DAsync_v2(&picks,call.stream.handle),"hybrid pinned native ordered picks");
            try c.wait(call.stream,until);
            for(0..n*8) |i| {const pick=std.mem.readInt(i32,c.request.bytes[wire.tile_rows*4096*2+i*4..][0..4],.little);if(pick<0 or pick>=288) return error.BadHybridPick;}
            for(c.pairs,0..) |pair,half| {
                var h=wire.Header{.session=c.identity.session,.sequence=c.next_sequence,.native_round=call.native_round,.buffer_epoch=call.buffer_epoch,.layer=call.layer,.phase=call.phase,.half=@intCast(half),.live_rows=@intCast(n),.model_contract_sha256=c.identity.model,.row_map_sha256=call.row_map_digest,.settings_sha256=c.identity.settings,.execution_mode=if(call.prompt) .prompt else .decode,.tile_row_lo=@intCast(lo),.call_rows=@intCast(call.rows)};h.bind();
                try transfer(pair,h,c.request.bytes,c.reply.bytes,c.frames[0..wire.post_bytes],c.frames[wire.post_bytes..],until);
                try check(c);if(!std.mem.allEqual(u8,c.reply.bytes[n*8*4096*4..],0)) return error.BadUnusedRows;
                c.pending=true;try c.d.check(c.d.api.cuMemcpyHtoDAsync_v2(c.slots.ptr,c.reply.bytes.ptr,n*8*4096*4,call.stream.handle),"hybrid pinned unweighted result slots");
                try glue.combine(c.set,call.stream,c.slots.ptr,true,dest.weights_f32+lo*8*4,dest.base+(half*call.rows+lo)*4096*4,n,4096,8);
                try c.wait(call.stream,until); // durable call planes now own this tile; staging may retire
            }
        }
        try check(c);
        return .{.session=c.identity.session,.sequence_first=first,.sequence_last=c.next_sequence,.tiles=tiles,.buffer_epoch=call.buffer_epoch,.row_map_digest=call.row_map_digest,.rows=call.rows,.ready_event=c.done};
    }
};

/// Pinned CPU framing around the unchanged native byte-exchange kernel. This is deliberately not graph-capturable.
pub const NativePair=struct {
    r: *tp.roce.Roce,stream: cuda.Stream,frames: cuda.DeviceBuffer,host: cuda.HostBuffer,done: cuda.Event,
    pending: bool=false,failed: ?anyerror=null,control: ?tp.sock.Fd=null,
    call_budget:CallBudget=.{},turn_until:u64=0,
    pub fn init(d: *const cuda.Driver,r: *tp.roce.Roce,stream: cuda.Stream) !NativePair {
        const memory_scope=@import("cuda_hybrid_ledger.zig").scopeOwner(.transport);defer memory_scope.end();
        var frames=try cuda.DeviceBuffer.alloc(d,2*wire.batch_bytes);errdefer frames.free();
        var host=try cuda.HostBuffer.alloc(d,2*wire.batch_bytes);errdefer host.free();
        const done=try cuda.Event.init(d,false);
        return .{.r=r,.stream=stream,.frames=frames,.host=host,.done=done};
    }
    pub fn frameIO(p: *NativePair) FrameIO {return .{.ctx=p,.now=now,.check=check,.exchange=exchange,.batch=batch,.announce=announce};}
    pub fn releaseStorage(p: *NativePair) void {std.debug.assert(!p.pending and p.failed==null);p.done.deinit();p.host.free();p.frames.free();p.*=undefined;}
    /// Metadata-only wake on the existing native bootstrap plan fd. Tensors never fall back to TCP.
    fn announce(ctx: *anyopaque,h: wire.Header,until: u64) !void {
        const p: *NativePair=@ptrCast(@alignCast(ctx));try check(p);
        try announceOn(p.metadata(),h,until);
    }
    pub fn announceOn(io:ControlIO,h:wire.Header,until:u64)!void {
        try io.check(io.ctx);if(io.now(io.ctx)>=until)return error.Timeout;
        var bytes: [256]u8=undefined;h.encode(&bytes);
        try io.send(io.ctx,&bytes,until);
        var ready:[8]u8=undefined;try io.recv(io.ctx,&ready,until);
        if(std.mem.readInt(u64,&ready,.little)!=budget_ready)return error.BadHybridBudget;
        // Receiver's anchor precedes this sample causally. Never compare host clocks.
        const sampled=io.now(io.ctx);if(sampled>=until)return error.Timeout;
        var remaining:[8]u8=undefined;std.mem.writeInt(u64,&remaining,until-sampled,.little);
        try io.send(io.ctx,&remaining,until);if(io.now(io.ctx)>=until)return error.Timeout;try io.check(io.ctx);
    }
    fn metadata(p:*NativePair)ControlIO {return .{.ctx=p,.now=now,.check=check,.send=sendControl,.recv=recvControl};}
    fn sendControl(ctx:*anyopaque,bytes:[]const u8,until:u64)!void {const p:*NativePair=@ptrCast(@alignCast(ctx));try tp.sock.sendBefore(p.control orelse return error.MissingHybridControl,bytes,until);}
    fn recvControl(ctx:*anyopaque,bytes:[]u8,until:u64)!void {const p:*NativePair=@ptrCast(@alignCast(ctx));try tp.sock.recvExact(p.control orelse return error.MissingHybridControl,bytes,until);}
    pub fn turnDeadline(p:*const NativePair,at:u64,timeout:u64)u64 {
        return p.call_budget.activeUntil() orelse at+|timeout;
    }
    pub fn readWake(p: *NativePair,identity: Identity,half: u1,sequence: u64,until: u64) !wire.Header {
        return p.readWakeOn(p.metadata(),identity,half,sequence,until);
    }
    pub fn readWakeOn(p:*NativePair,io:ControlIO,identity:Identity,half:u1,sequence:u64,until:u64)!wire.Header {
        if(p.failed)|err|return err;
        return p.readWakeInner(io,identity,half,sequence,until) catch |err| {p.failed=err;return err;};
    }
    fn readWakeInner(p: *NativePair,io:ControlIO,identity: Identity,half: u1,sequence: u64,outer: u64) !wire.Header {
        const until=@min(outer,p.call_budget.activeUntil() orelse outer);
        try io.check(io.ctx);if(io.now(io.ctx)>=until) return error.Timeout;
        var bytes: [256]u8=undefined;try io.recv(io.ctx,&bytes,until);
        if(io.now(io.ctx)>=until) return error.Timeout;try io.check(io.ctx);
        const h=try wire.Header.decode(&bytes);
        if(h.status!=.ok or (h.kind!=.request and h.kind!=.close and h.kind!=.cancel) or h.total_payload_bytes!=0 or h.fragment_index!=0 or h.fragment_count!=1 or h.payload_offset!=0 or h.fragment_payload_bytes!=0 or h.session!=identity.session or h.sequence!=sequence or h.half!=half or !std.mem.eql(u8,&h.model_contract_sha256,&identity.model) or !std.mem.eql(u8,&h.settings_sha256,&identity.settings)) return error.BadHybridCall;
        if(h.kind==.request and (h.buffer_epoch==0 or std.mem.allEqual(u8,&h.row_map_sha256,0))) return error.BadHybridCall;
        if(h.kind!=.request and p.call_budget.activeUntil()!=null)return error.BadHybridCall;
        const anchor=io.now(io.ctx);if(anchor>=until)return error.Timeout;
        var ready:[8]u8=undefined;std.mem.writeInt(u64,&ready,budget_ready,.little);try io.send(io.ctx,&ready,until);
        var remaining:[8]u8=undefined;try io.recv(io.ctx,&remaining,until);
        const ns=std.mem.readInt(u64,&remaining,.little);
        const maximum=if(h.kind==.request)identity.transaction_timeout_ns else 3_600_000_000_000;
        if(ns==0 or ns>maximum)return error.BadHybridBudget;
        const bound=@min(until,anchor+|ns);
        p.turn_until=if(h.kind==.request)try p.call_budget.begin(h,bound) else bound;
        if(io.now(io.ctx)>=p.turn_until)return error.Timeout;try io.check(io.ctx);
        return h;
    }
    fn now(_: *anyopaque) u64 {return tp.sock.nowNs();}
    fn check(ctx: *anyopaque) !void {
        const p: *NativePair=@ptrCast(@alignCast(ctx));if(p.failed) |err| return err;
        p.r.check() catch |err| {p.failed=err;return err;};
        if(p.control) |fd| {
            var byte: [1]u8=undefined;const n=std.c.recv(fd,&byte,1,std.posix.MSG.PEEK|std.posix.MSG.DONTWAIT);
            if(n==0 or (n<0 and std.posix.errno(n)!=.AGAIN and std.posix.errno(n)!=.INTR)) {p.failed=error.Closed;return error.Closed;}
        }
    }
    pub fn wait(p: *NativePair,until: u64) !void {
        while(p.pending) {
            if(tp.sock.nowNs()>=until) return error.Timeout;
            try check(p);const rc=p.done.d.api.cuEventQuery(p.done.handle);
            if(rc==cuda.abi.success) {p.pending=false;return;}
            if(rc!=cuda.abi.error_not_ready) return error.HybridCopyFailed;
            std.atomic.spinLoopHint();
        }
    }
    /// Batch only copies and completion boundaries. Every native exchange is still one bounded post,
    /// ordered on this pair's existing stream. No expert work or half transactions overlap.
    fn batch(ctx:*anyopaque,h:wire.Header,kind:wire.Kind,source:[]const u8,dest:[]u8,until:u64)!void {
        const p:*NativePair=@ptrCast(@alignCast(ctx));
        p.batchPhase(h,kind,source,dest,until) catch |err| {p.failed=err;p.r.abort();return err;};
    }
    fn batchPhase(p:*NativePair,h:wire.Header,kind:wire.Kind,source:[]const u8,dest:[]u8,until:u64)!void {
        try check(p);if(tp.sock.nowNs()>=until)return error.Timeout;
        const total= switch(kind){.request=>wire.request_bytes,.reply=>wire.reply_bytes,else=>return error.BadFragment};
        const sending=source.len==total and dest.len==0;
        if(!sending and !(source.len==0 and dest.len==total))return error.BadFrameStorage;
        const count=wire.fragments(total);const bytes=count*wire.post_bytes;
        if(bytes>wire.batch_bytes or p.frames.len!=2*wire.batch_bytes or p.host.bytes.len!=2*wire.batch_bytes)return error.BadFrameStorage;
        try p.wait(until); // prior device/pinned staging must actually retire before reuse
        for(0..count)|i| {
            if(tp.sock.nowNs()>=until)return error.Timeout;try check(p);
            const tx=p.host.bytes[i*wire.post_bytes..][0..wire.post_bytes];@memset(tx,0);
            const send=wire.fragment(h,if(sending)kind else .idle,total,i);send.encode(tx[0..256]);
            if(sending)@memcpy(tx[256..][0..send.fragment_payload_bytes],source[send.payload_offset..][0..send.fragment_payload_bytes]);
        }
        const d=p.stream.d;p.pending=true;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(p.frames.ptr,p.host.bytes.ptr,bytes,p.stream.handle),"hybrid pinned phase upload");
        for(0..count)|i| {
            if(tp.sock.nowNs()>=until)return error.Timeout;try check(p);
            // Same native kernel/API/stream, no larger post, gather mode, renewed deadline or parallel half.
            try p.r.exchangeBefore(p.frames.ptr+i*wire.post_bytes,p.frames.ptr+wire.batch_bytes+i*wire.post_bytes,wire.post_bytes,p.stream.handle,until);
        }
        if(tp.sock.nowNs()>=until)return error.Timeout;try check(p);
        try d.check(d.api.cuMemcpyDtoHAsync_v2(p.host.bytes.ptr+wire.batch_bytes,p.frames.ptr+wire.batch_bytes,bytes,p.stream.handle),"hybrid pinned phase download");
        try p.done.record(p.stream);try p.wait(until);
        // Validate every frame before any reconstructed tensor becomes available to the consumer.
        for(0..count)|i| {
            if(tp.sock.nowNs()>=until)return error.Timeout;try check(p);
            const rx=p.host.bytes[wire.batch_bytes+i*wire.post_bytes..][0..wire.post_bytes];
            const want=wire.fragment(h,if(sending).idle else kind,total,i);try wire.require(rx[0..256],want);
            if(!std.mem.allEqual(u8,rx[256+want.fragment_payload_bytes..],0))return error.BadPadding;
        }
        if(!sending)for(0..count)|i| {
            if(tp.sock.nowNs()>=until)return error.Timeout;try check(p);
            const want=wire.fragment(h,kind,total,i);
            @memcpy(dest[want.payload_offset..][0..want.fragment_payload_bytes],p.host.bytes[wire.batch_bytes+i*wire.post_bytes+256..][0..want.fragment_payload_bytes]);
        };
    }
    fn exchange(ctx: *anyopaque,send: []const u8,recv: []u8,until: u64) !void {
        const p: *NativePair=@ptrCast(@alignCast(ctx));
        p.exchangeFrame(send,recv,until) catch |err| {p.failed=err;p.r.abort();return err;};
    }
    fn exchangeFrame(p: *NativePair,send: []const u8,recv: []u8,until: u64) !void {
        try check(p);if(tp.sock.nowNs()>=until) return error.Timeout;
        if(send.len!=wire.post_bytes or recv.len!=wire.post_bytes) return error.FrameTooLarge;
        try p.wait(until);@memcpy(p.host.bytes[0..wire.post_bytes],send);
        const d=p.stream.d;p.pending=true;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(p.frames.ptr,p.host.bytes.ptr,wire.post_bytes,p.stream.handle),"hybrid pinned frame upload");
        try p.r.exchangeBefore(p.frames.ptr,p.frames.ptr+wire.batch_bytes,wire.post_bytes,p.stream.handle,until);
        try d.check(d.api.cuMemcpyDtoHAsync_v2(p.host.bytes.ptr+wire.batch_bytes,p.frames.ptr+wire.batch_bytes,wire.post_bytes,p.stream.handle),"hybrid pinned frame download");
        try p.done.record(p.stream);try p.wait(until);try check(p);
        @memcpy(recv,p.host.bytes[wire.batch_bytes..][0..wire.post_bytes]);
    }
};

fn step(io: FrameIO,tx: []u8,rx: []u8,send: wire.Header,want: wire.Header,source: []const u8,dest: []u8,until: u64) !void {
    if(io.now(io.ctx)>=until) return error.Timeout;
    try io.check(io.ctx);@memset(tx,0);send.encode(tx[0..256]);
    if(source.len>wire.frame_payload) return error.FrameTooLarge;
    @memcpy(tx[256..][0..source.len],source);
    try io.exchange(io.ctx,tx,rx,until);
    if(io.now(io.ctx)>=until) return error.Timeout;
    try io.check(io.ctx);try wire.require(rx[0..256],want);
    const n=want.fragment_payload_bytes;
    if(dest.len!=n) return error.BadFragment;
    if(!std.mem.allEqual(u8,rx[256+n..],0)) return error.BadPadding;
    @memcpy(dest,rx[256..][0..n]);
}
/// One real expert role turn: consume the actual fd wake, then the opposite native frame transcript.
pub fn workerRoleTurn(control: *NativePair,frames: FrameIO,identity: Identity,half: u1,sequence: u64,input: []u8,output: []u8,tx: []u8,rx: []u8,ctx: *anyopaque,work: WorkFn,until: u64) !wire.Kind {
    return workerRoleTurnOn(control,control.metadata(),frames,identity,half,sequence,input,output,tx,rx,ctx,work,until);
}
pub fn workerRoleTurnOn(control:*NativePair,metadata:ControlIO,frames:FrameIO,identity:Identity,half:u1,sequence:u64,input:[]u8,output:[]u8,tx:[]u8,rx:[]u8,ctx:*anyopaque,work:WorkFn,until:u64)!wire.Kind {
    const h=try control.readWakeOn(metadata,identity,half,sequence,until);
    const bound=control.turn_until;
    if(h.kind==.close) {
        var want=h;want.kind=.idle;control.control=null; // Bootstrap still owns the fd; deliberate close is not peer loss.
        step(frames,tx,rx,h,want,&.{},&.{},bound) catch |err| {control.failed=err;return err;};
        return .close;
    }
    if(h.kind!=.request) return error.BadHybridCall;
    workerTransfer(frames,h,input,output,tx,rx,ctx,work,bound) catch |err| {control.failed=err;return err;};
    try control.call_budget.finish(h);
    try metadata.check(metadata.ctx);return h.kind;
}

pub fn closePeer(control: *NativePair,frames: FrameIO,identity: Identity,half: u1,sequence: u64,tx: []u8,rx: []u8,until: u64) !void {
    var h=wire.Header{.kind=.close,.session=identity.session,.sequence=sequence,.half=half,.live_rows=1,.call_rows=1,.model_contract_sha256=identity.model,.settings_sha256=identity.settings};h.bind();
    const io=control.frameIO();try io.announce.?(io.ctx,h,until);control.control=null;
    var send=h;send.kind=.idle;try step(frames,tx,rx,send,h,&.{},&.{},until);
}

pub const WorkFn=*const fn(*anyopaque,wire.Header,[]const u8,[]u8,u64) anyerror!void;
/// Opposite direction of the same fixed transcript; invoked only after a validated metadata wake.
pub fn workerTransfer(io: FrameIO,h: wire.Header,input: []u8,output: []u8,tx: []u8,rx: []u8,ctx: *anyopaque,work: WorkFn,until: u64) !void {
    if(input.len!=wire.request_bytes or output.len!=wire.reply_bytes or tx.len!=wire.post_bytes or rx.len!=wire.post_bytes) return error.BadFrameStorage;
    if(io.batch)|phase|try phase(io.ctx,h,.request,&.{},input,until) else for(0..wire.fragments(input.len)) |i| {
        const send=wire.fragment(h,.idle,input.len,i);const want=wire.fragment(h,.request,input.len,i);
        try step(io,tx,rx,send,want,&.{},input[want.payload_offset..][0..want.fragment_payload_bytes],until);
    }
    if(io.now(io.ctx)>=until) return error.Timeout;try io.check(io.ctx);
    @memset(output,0);try work(ctx,h,input,output,until);
    if(io.now(io.ctx)>=until) return error.Timeout;try io.check(io.ctx);
    if(io.batch)|phase|try phase(io.ctx,h,.reply,output,&.{},until) else for(0..wire.fragments(output.len)) |i| {
        const send=wire.fragment(h,.reply,output.len,i);const want=wire.fragment(h,.idle,output.len,i);
        try step(io,tx,rx,send,want,output[send.payload_offset..][0..send.fragment_payload_bytes],&.{},until);
    }
}

/// Requests precede replies. IDLE carries the active tensor's fragment identity with zero carried payload.
pub fn transfer(io: FrameIO,h: wire.Header,input: []const u8,output: []u8,tx: []u8,rx: []u8,until: u64) !void {
    if(input.len!=wire.request_bytes or output.len!=wire.reply_bytes or tx.len!=wire.post_bytes or rx.len!=wire.post_bytes) return error.BadFrameStorage;
    try io.check(io.ctx);if(io.now(io.ctx)>=until) return error.Timeout;
    const announce=io.announce orelse return error.MissingHybridControl;try announce(io.ctx,h,until);
    if(io.batch)|phase| {try phase(io.ctx,h,.request,input,&.{},until);try phase(io.ctx,h,.reply,&.{},output,until);return;}
    for(0..wire.fragments(input.len)) |i| {
        const send=wire.fragment(h,.request,input.len,i);const want=wire.fragment(h,.idle,input.len,i);
        try step(io,tx,rx,send,want,input[send.payload_offset..][0..send.fragment_payload_bytes],&.{},until);
    }
    for(0..wire.fragments(output.len)) |i| {
        const send=wire.fragment(h,.idle,output.len,i);const want=wire.fragment(h,.reply,output.len,i);
        try step(io,tx,rx,send,want,&.{},output[want.payload_offset..][0..want.fragment_payload_bytes],until);
    }
}
