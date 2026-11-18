// Modified for V4 by Scott Quarrier with Helios: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! The model's text loads on another thread while the engine opens on this one: both come up, or neither is left open.
const std = @import("std");

/// What `both` hands back: the loaded text and the opened engine.
pub fn Both(comptime Load: type, comptime Open: type) type {
    const Text = @typeInfo(@typeInfo(Load).@"fn".return_type.?).error_union.payload;
    const Engine = @typeInfo(@typeInfo(@typeInfo(Open).@"fn".return_type.?).error_union.payload).optional.child;
    return struct { text: Text, engine: Engine };
}

/// Load text on another thread; its error wins, then open's error or refusal, cleaning whichever succeeded.
pub fn both(io: std.Io, load: anytype, load_args: std.meta.ArgsTuple(@TypeOf(load)), open: anytype, open_args: std.meta.ArgsTuple(@TypeOf(open))) !?Both(@TypeOf(load), @TypeOf(open)) {
    var loading = io.async(load, load_args);
    const engine = @call(.auto, open, open_args) catch |e| {
        const text = try loading.await(io);
        text.deinit();
        return e;
    } orelse {
        const text = try loading.await(io);
        text.deinit();
        return null;
    };
    const text = loading.await(io) catch |e| {
        engine.close(engine.ctx);
        return e;
    };
    return .{ .text = text, .engine = engine };
}

// One process latch governs both startup branches. A selected result always retains ownership until handed back or closed.
pub fn bothControlled(io:std.Io,load:anytype,load_args:std.meta.ArgsTuple(@TypeOf(load)),open:anytype,open_args:std.meta.ArgsTuple(@TypeOf(open)),requested:?*const std.atomic.Value(bool)) !?Both(@TypeOf(load),@TypeOf(open)) {
    const flag=requested orelse return both(io,load,load_args,open,open_args);
    if(flag.load(.acquire))return error.Stopped;
    const Result=@typeInfo(@TypeOf(load)).@"fn".return_type.?;
    const Text=@typeInfo(Result).error_union.payload;
    const Event=union(enum){text:Result,stopped:bool};
    const Selection=std.Io.Select(Event);
    const Wait=struct {
        fn stop(source:std.Io,latch:*const std.atomic.Value(bool))bool {
            while(!latch.load(.acquire))std.Io.sleep(source,.fromMilliseconds(10),.awake) catch return false;
            return true;
        }
        fn text(events:*Selection)anyerror!Text {
            return switch(try events.await()){.text=>|result|result,.stopped=>|latched|if(latched)error.Stopped else error.Canceled};
        }
    };
    var buffer:[2]Event=undefined;var events=Selection.init(io,&buffer);
    defer while(events.cancel())|event|switch(event){.text=>|result|{if(result)|text|{text.deinit();}else |_|{}},.stopped=>{}};
    try events.concurrent(.stopped,Wait.stop,.{io,flag});
    events.async(.text,load,load_args);
    const engine=@call(.auto,open,open_args) catch |err| {
        const text=Wait.text(&events) catch |text_err| {return if(text_err==error.Stopped or text_err==error.Canceled or text_err==error.Cancelled)err else text_err;};
        text.deinit();return err;
    } orelse {
        const text=Wait.text(&events) catch |text_err| {if(text_err==error.Stopped or text_err==error.Canceled or text_err==error.Cancelled)return null;return text_err;};
        text.deinit();return null;
    };
    errdefer engine.close(engine.ctx);
    const text=try Wait.text(&events);
    if(flag.load(.acquire)){text.deinit();return error.Stopped;}
    return .{.text=text,.engine=engine};
}

const Fake = struct {
    var closed: u32 = 0;
    var freed: u32 = 0;
    var text: Text = .{};
    const Text = struct {
        fn deinit(_: *Text) void {
            freed += 1;
        }
    };
    const Engine = struct { ctx: *anyopaque, close: *const fn (*anyopaque) void };
    const Outcome = enum { ok, fails, refuses };

    fn reset() void {
        closed = 0;
        freed = 0;
    }

    fn close(_: *anyopaque) void {
        closed += 1;
    }

    fn load(io: std.Io, ms: i64, fails: bool) error{TextFailed}!*Text {
        std.Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
        return if (fails) error.TextFailed else &text;
    }

    fn open(io: std.Io, ms: i64, outcome: Outcome) error{EngineFailed}!?Engine {
        std.Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
        return switch (outcome) {
            .ok => .{ .ctx = &text, .close = close },
            .fails => error.EngineFailed,
            .refuses => null,
        };
    }
};

test "a text that fails after the engine opened closes the engine" {
    const io = std.testing.io;
    Fake.reset();
    try std.testing.expectError(error.TextFailed, both(io, Fake.load, .{ io, 30, true }, Fake.open, .{ io, 0, .ok }));
    try std.testing.expectEqual(@as(u32, 1), Fake.closed);
    try std.testing.expectEqual(@as(u32, 0), Fake.freed);
}

test "an engine that fails or refuses after the text loaded frees the text" {
    const io = std.testing.io;
    Fake.reset();
    try std.testing.expectError(error.EngineFailed, both(io, Fake.load, .{ io, 0, false }, Fake.open, .{ io, 30, .fails }));
    try std.testing.expectEqual(@as(u32, 1), Fake.freed);
    try std.testing.expect((try both(io, Fake.load, .{ io, 0, false }, Fake.open, .{ io, 30, .refuses })) == null);
    try std.testing.expectEqual(@as(u32, 2), Fake.freed);
    try std.testing.expectEqual(@as(u32, 0), Fake.closed);
}

test "both failing reports the text's error, as when it loaded first; both up hands both back" {
    const io = std.testing.io;
    Fake.reset();
    try std.testing.expectError(error.TextFailed, both(io, Fake.load, .{ io, 0, true }, Fake.open, .{ io, 0, .fails }));
    try std.testing.expectError(error.TextFailed, both(io, Fake.load, .{ io, 0, true }, Fake.open, .{ io, 0, .refuses }));
    const up = (try both(io, Fake.load, .{ io, 10, false }, Fake.open, .{ io, 0, .ok })).?;
    try std.testing.expectEqual(&Fake.text, up.text);
    try std.testing.expectEqual(@as(u32, 0), Fake.closed + Fake.freed);
}

// Helios: actual startup join under real Io task cancellation and process TERM; no model/devices.
const t=std.testing;const startup=@This();
const P=struct {
    var stopping=std.atomic.Value(bool).init(false);var fallback=std.atomic.Value(bool).init(false);
    var live=std.atomic.Value(usize).init(0);var joined=std.atomic.Value(usize).init(0);var cancelled=std.atomic.Value(bool).init(false);
    var freed:usize=0;var closed:usize=0;var text:Text=.{};
    const Text=struct{fn deinit(_: *Text)void{freed+=1;}};
    const Engine=struct{ctx:*anyopaque,close:*const fn(*anyopaque)void};const Mode=enum{ok,stopped,unsafe_error};
    fn reset()void{stopping.store(false,.release);fallback.store(false,.release);live.store(0,.release);joined.store(0,.release);cancelled.store(false,.release);freed=0;closed=0;}
    fn signal(_:std.posix.SIG)callconv(.c)void{stopping.store(true,.release);}
    fn finish(_: *anyopaque)void{closed+=1;}
    fn load(io:std.Io,fast:bool)anyerror!*Text {
        _=live.fetchAdd(1,.acq_rel);defer {_=live.fetchSub(1,.acq_rel);_=joined.fetchAdd(1,.acq_rel);}
        if(fast)return &text;
        while(!fallback.load(.acquire))std.Io.sleep(io,.fromMilliseconds(5),.awake) catch |err| {cancelled.store(true,.release);return err;};
        return &text;
    }
    fn open(mode:Mode)anyerror!?Engine {
        switch(mode){.ok=>return .{.ctx=&text,.close=finish},.stopped=>{try std.posix.raise(.TERM);return error.Stopped;},.unsafe_error=>{try std.posix.raise(.TERM);return error.HybridRetained;}}
    }
    fn release(send:bool)void {
        std.Io.sleep(t.io,.fromMilliseconds(20),.awake) catch {};
        if(send)std.posix.raise(.TERM) catch unreachable;
        std.Io.sleep(t.io,.fromMilliseconds(180),.awake) catch {};fallback.store(true,.release);
    }
};
fn invoke(mode:P.Mode,fast:bool)!?startup.Both(@TypeOf(P.load),@TypeOf(P.open)) {
    return if(comptime @hasDecl(startup,"bothControlled"))@field(startup,"bothControlled")(t.io,P.load,.{t.io,fast},P.open,.{mode},&P.stopping)else startup.both(t.io,P.load,.{t.io,fast},P.open,.{mode});
}
fn install()std.posix.Sigaction{const a:std.posix.Sigaction=.{.handler=.{.handler=P.signal},.mask=std.posix.sigemptyset(),.flags=0};var prior:std.posix.Sigaction=undefined;std.posix.sigaction(.TERM,&a,&prior);return prior;}
fn expectStopped(mode:P.Mode)!void {
    if(invoke(mode,false))|maybe|{if(maybe)|up|{up.text.deinit();up.engine.close(up.engine.ctx);}return error.TestExpectedError;}else|err|try t.expectEqual(error.Stopped,err);
}
test "V4 startup join: stopped engine cancels and joins actual async text before return" {
    P.reset();const prior=install();defer std.posix.sigaction(.TERM,&prior,null);const releaser=try std.Thread.spawn(.{},P.release,.{false});defer releaser.join();
    try expectStopped(.stopped);try t.expect(P.cancelled.load(.acquire));try t.expectEqual(@as(usize,1),P.joined.load(.acquire));try t.expectEqual(@as(usize,0),P.live.load(.acquire));try t.expectEqual(@as(usize,0),P.closed);
}
test "V4 startup join: TERM after engine success cancels text and closes exactly once" {
    P.reset();const prior=install();defer std.posix.sigaction(.TERM,&prior,null);const releaser=try std.Thread.spawn(.{},P.release,.{true});defer releaser.join();
    try expectStopped(.ok);try t.expect(P.cancelled.load(.acquire));try t.expectEqual(@as(usize,1),P.joined.load(.acquire));try t.expectEqual(@as(usize,0),P.live.load(.acquire));try t.expectEqual(@as(usize,1),P.closed);
}
test "V4 startup join: latched TERM never masks a retained ownership failure" {
    P.reset();const prior=install();defer std.posix.sigaction(.TERM,&prior,null);const releaser=try std.Thread.spawn(.{},P.release,.{false});defer releaser.join();
    try t.expectError(error.HybridRetained,invoke(.unsafe_error,false));try t.expectEqual(@as(usize,0),P.live.load(.acquire));try t.expectEqual(@as(usize,1),P.joined.load(.acquire));try t.expectEqual(@as(usize,0),P.closed);
}
test "V4 startup join: normal controlled startup transfers both live owners" {
    P.reset();const up=(try invoke(.ok,true)).?;try t.expectEqual(@as(usize,0),P.closed);try t.expectEqual(@as(usize,0),P.freed);up.text.deinit();up.engine.close(up.engine.ctx);try t.expectEqual(@as(usize,1),P.closed);try t.expectEqual(@as(usize,1),P.freed);try t.expectEqual(@as(usize,0),P.live.load(.acquire));
}
