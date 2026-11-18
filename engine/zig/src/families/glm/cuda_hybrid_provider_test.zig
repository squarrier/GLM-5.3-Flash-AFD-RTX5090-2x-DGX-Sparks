const std=@import("std");
const t=std.testing;
const tr=@import("glm").moe.hybridTransport;
const tp=@import("tp");
const Channel=struct {
    mutex: std.atomic.Value(bool)=.init(false),data: [2][]u8,epoch: [2]usize=.{0,0},released: [2]usize=.{0,0},
    const Peer=struct {channel: *Channel,index: usize,native: ?*tr.NativePair=null};
    fn now(_: *anyopaque) u64 {return tp.sock.nowNs();}
    fn check(_: *anyopaque) !void {}
    fn announce(ctx: *anyopaque,h: tr.wire.Header,until: u64) !void {const p: *Peer=@ptrCast(@alignCast(ctx));if(p.native) |native| {const nio=native.frameIO();try nio.announce.?(nio.ctx,h,until);}}
    fn lock(c: *Channel) void {while(c.mutex.swap(true,.acquire)) tp.sock.sleepNs(10_000);}
    fn unlock(c: *Channel) void {c.mutex.store(false,.release);}
    fn wait(c: *Channel,until: u64) !void {if(tp.sock.nowNs()>=until) return error.Timeout;c.unlock();tp.sock.sleepNs(10_000);c.lock();}
    fn exchange(ctx: *anyopaque,send: []const u8,recv: []u8,until: u64) !void {
        const p: *Peer=@ptrCast(@alignCast(ctx));const c=p.channel;const me=p.index;const other=1-me;
        c.lock();defer c.unlock();const epoch=c.epoch[me]+1;
        while(c.released[other]+1<epoch) try c.wait(until);
        @memcpy(c.data[me],send);c.epoch[me]=epoch;
        while(c.epoch[other]<epoch) try c.wait(until);
        if(c.epoch[other]!=epoch) return error.BadEpoch;
        @memcpy(recv,c.data[other]);c.released[me]=epoch;
    }
    fn io(p: *Peer) tr.FrameIO {return .{.ctx=p,.now=now,.check=check,.exchange=exchange,.announce=announce};}
};
const Worker=struct {
    control: ?*tr.NativePair=null,io: tr.FrameIO,h: tr.wire.Header,input: []u8,output: []u8,tx: []u8,rx: []u8,until: u64,err: ?anyerror=null,calls: usize=0,
    fn compute(ctx: *anyopaque,h: tr.wire.Header,input: []const u8,output: []u8,until: u64) !void {
        const w: *Worker=@ptrCast(@alignCast(ctx));try t.expect(tp.sock.nowNs()<until);try t.expectEqual(@as(u32,128),h.live_rows);
        for(input) |b| try t.expectEqual(@as(u8,0x13),b);
        for(output,0..) |*b,i| b.*=@intCast(i%251);w.calls+=1;
    }
    fn run(w: *Worker) void {
        if(comptime @hasDecl(tr,"workerRoleTurn")) if(w.control) |control| {
            _=tr.workerRoleTurn(control,w.io,.{.session=w.h.session,.model=w.h.model_contract_sha256,.settings=w.h.settings_sha256,.transaction_timeout_ns=5_000_000_000},0,w.h.sequence,w.input,w.output,w.tx,w.rx,w,compute,w.until) catch |err| {w.err=err;};return;
        };
        const result=if(comptime @hasDecl(tr,"workerTransfer")) tr.workerTransfer(w.io,w.h,w.input,w.output,w.tx,w.rx,w,compute,w.until) else tr.transfer(w.io,w.h,w.input,w.output,w.tx,w.rx,w.until);
        result catch |err| {w.err=err;};
    }
};
test "hybrid peer: both actual transaction directions exchange nineteen bounded frames" {
    const a=t.allocator;const inputs=try a.alloc(u8,2*tr.wire.request_bytes);defer a.free(inputs);@memset(inputs[0..tr.wire.request_bytes],0x13);
    const outputs=try a.alloc(u8,2*tr.wire.reply_bytes);defer a.free(outputs);
    const frames=try a.alloc(u8,6*tr.wire.post_bytes);defer a.free(frames);
    var channel=Channel{.data=.{frames[0..tr.wire.post_bytes],frames[tr.wire.post_bytes..2*tr.wire.post_bytes]}};
    var fds: [2]std.c.fd_t=undefined;try t.expectEqual(@as(c_int,0),std.c.socketpair(std.posix.AF.UNIX,std.posix.SOCK.STREAM,0,&fds));defer tp.sock.close(fds[0]);defer tp.sock.close(fds[1]);
    const region=try a.alignedAlloc(u8,.fromByteUnits(4096),8192);defer a.free(region);@memset(region,0);
    var roce: tp.roce.Roce=undefined;roce.region=region;roce.aborted=.init(false);
    var sender: tr.NativePair=undefined;sender.r=&roce;sender.failed=null;sender.control=fds[0];
    var receiver: tr.NativePair=undefined;receiver.r=&roce;receiver.failed=null;receiver.control=fds[1];
    var left=Channel.Peer{.channel=&channel,.index=0,.native=&sender};var right=Channel.Peer{.channel=&channel,.index=1};
    var h=tr.wire.Header{.session=1,.sequence=1,.native_round=55,.buffer_epoch=8,.layer=3,.phase=.mtp,.half=0,.live_rows=128,.call_rows=128,.model_contract_sha256=@splat(0xAF),.row_map_sha256=@splat(0xBB),.settings_sha256=@splat(0xCC)};h.bind();
    const until=tp.sock.nowNs()+5_000_000_000;
    var worker=Worker{.control=&receiver,.io=Channel.io(&right),.h=h,.input=inputs[tr.wire.request_bytes..],.output=outputs[tr.wire.reply_bytes..],.tx=frames[2*tr.wire.post_bytes..3*tr.wire.post_bytes],.rx=frames[3*tr.wire.post_bytes..4*tr.wire.post_bytes],.until=until};
    const thread=try std.Thread.spawn(.{},Worker.run,.{&worker});
    const result=tr.transfer(Channel.io(&left),h,inputs[0..tr.wire.request_bytes],outputs[0..tr.wire.reply_bytes],frames[4*tr.wire.post_bytes..5*tr.wire.post_bytes],frames[5*tr.wire.post_bytes..],until);
    thread.join();try result;if(worker.err) |err| return err;
    try t.expectEqual(@as(usize,1),worker.calls);try t.expectEqual(@as(usize,19),channel.epoch[0]);try t.expectEqual(channel.epoch[0],channel.epoch[1]);
    var remaining: [1]u8=undefined;const read=std.c.recv(fds[1],&remaining,1,std.posix.MSG.DONTWAIT);try t.expectEqual(@as(isize,-1),read);try t.expectEqual(std.posix.E.AGAIN,std.posix.errno(read));
    for(outputs[0..tr.wire.reply_bytes],0..) |b,i| try t.expectEqual(@as(u8,@intCast(i%251)),b);
}

test "hybrid peer: actual role close acknowledges without dispatching an expert" {
    const a=t.allocator;const frames=try a.alloc(u8,6*tr.wire.post_bytes);defer a.free(frames);
    var channel=Channel{.data=.{frames[0..tr.wire.post_bytes],frames[tr.wire.post_bytes..2*tr.wire.post_bytes]}};
    var fds: [2]std.c.fd_t=undefined;try t.expectEqual(@as(c_int,0),std.c.socketpair(std.posix.AF.UNIX,std.posix.SOCK.STREAM,0,&fds));defer tp.sock.close(fds[0]);defer tp.sock.close(fds[1]);
    const region=try a.alignedAlloc(u8,.fromByteUnits(4096),8192);defer a.free(region);@memset(region,0);
    var roce: tp.roce.Roce=undefined;roce.region=region;roce.aborted=.init(false);
    var sender: tr.NativePair=undefined;sender.r=&roce;sender.failed=null;sender.control=fds[0];
    var receiver: tr.NativePair=undefined;receiver.r=&roce;receiver.failed=null;receiver.control=fds[1];
    var left=Channel.Peer{.channel=&channel,.index=0,.native=&sender};var right=Channel.Peer{.channel=&channel,.index=1};
    var h=tr.wire.Header{.kind=.close,.session=1,.sequence=1,.half=0,.live_rows=1,.call_rows=1,.model_contract_sha256=@splat(0xAA),.settings_sha256=@splat(0xBB)};h.bind();
    const until=tp.sock.nowNs()+5_000_000_000;const identity=tr.Identity{.session=1,.model=h.model_contract_sha256,.settings=h.settings_sha256,.transaction_timeout_ns=5_000_000_000};
    var worker=Worker{.control=&receiver,.io=Channel.io(&right),.h=h,.input=&.{},.output=&.{},.tx=frames[2*tr.wire.post_bytes..3*tr.wire.post_bytes],.rx=frames[3*tr.wire.post_bytes..4*tr.wire.post_bytes],.until=until};
    const thread=try std.Thread.spawn(.{},Worker.run,.{&worker});
    const result=if(comptime @hasDecl(tr,"closePeer")) tr.closePeer(&sender,Channel.io(&left),identity,0,1,frames[4*tr.wire.post_bytes..5*tr.wire.post_bytes],frames[5*tr.wire.post_bytes..],until) else Channel.announce(&left,h,until);
    thread.join();try result;if(worker.err) |err| return err;
    try t.expectEqual(@as(usize,0),worker.calls);try t.expectEqual(@as(usize,1),channel.epoch[0]);try t.expectEqual(channel.epoch[0],channel.epoch[1]);
}
