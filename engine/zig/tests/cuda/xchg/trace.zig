//! ping_pong_fixture test-only causal recorder: one writer per stream, scalar copies, dump after every producer joins.
//! GPT-6.1 Sol (OpenAI Codex), via Hermes. CLOCK_MONOTONIC is not a qualification clock override.
const std = @import("std");
const builtin = @import("builtin");
pub const capacity = 4096;
pub const producers = 56;
pub const Event = struct {
    ns: u64,
    kind: u32,
    seq: u32 = 0,
    fseq: i32 = 0,
    layer: i32 = 0,
    rows: i32 = 0,
    slot: u32 = 0,
    data: [8]u64 = @splat(0),
};
pub const Stream = struct { len: usize = 0, overflow: u64 = 0, events: [capacity]Event = undefined };
pub const Recorder = struct {
    id: u64,
    identity: u32,
    born: u64,
    joined: bool = false,
    streams: [producers]Stream = @splat(.{}),

    pub fn create(gpa: std.mem.Allocator, id: u64, identity: u32) !*Recorder {
        const r = try gpa.create(Recorder);
        r.id = id;
        r.identity = identity;
        r.born = now();
        r.joined = false;
        for (&r.streams) |*s| { s.len = 0; s.overflow = 0; }
        return r;
    }
    pub fn writer(r: *Recorder, producer: u32) Writer {
        std.debug.assert(producer < producers);
        return .{ .rec = r, .producer = producer };
    }
    pub fn dump(r: *const Recorder) void {
        std.debug.assert(r.joined);
        std.debug.print("ping_pong_fixture_HARNESS {d} {d} {d}\n", .{r.id, r.identity, r.born});
        for (&r.streams, 0..) |*s, p| {
            if (s.len == 0 and s.overflow == 0) continue;
            std.debug.print("ping_pong_fixture_STREAM {d} {d} {d} {d} {d}\n", .{r.id, r.identity, p, s.len, s.overflow});
            for (s.events[0..s.len]) |e| std.debug.print("ping_pong_fixture_EVENT {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d} {d}\n", .{r.id, r.identity, p, e.ns, e.kind, e.seq, e.fseq, e.layer, e.rows, e.slot, e.data[0], e.data[1], e.data[2], e.data[3], e.data[4], e.data[5], e.data[6], e.data[7]});
        }
    }
};
pub const Writer = struct {
    rec: ?*Recorder = null,
    producer: u32 = 0,
    seq: u32 = 0,
    fseq: i32 = 0,
    layer: i32 = 0,
    rows: i32 = 0,
    slot: u32 = 0,
    pub fn put(w: Writer, kind: u32, data: [8]u64) void {
        const r = w.rec orelse return;
        const s = &r.streams[w.producer];
        if (s.len == capacity) { s.overflow += 1; return; }
        s.events[s.len] = .{ .ns = now(), .kind = kind, .seq = w.seq, .fseq = w.fseq, .layer = w.layer, .rows = w.rows, .slot = w.slot, .data = data };
        s.len += 1;
    }
};
pub fn now() u64 {
    var t: std.c.timespec = undefined;
    std.debug.assert(std.c.clock_gettime(.MONOTONIC, &t) == 0);
    return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
}
fn cRecord(ctx: ?*anyopaque, producer: u32, kind: u32, seq: u32, block: u32, tid: u32, value: u64) callconv(.c) void {
    const r: *Recorder = @ptrCast(@alignCast(ctx orelse return));
    var w = r.writer(producer);
    w.seq = seq;
    w.put(kind, .{block, tid, value, 0, 0, 0, 0, 0});
}
comptime { if (builtin.is_test) @export(&cRecord, .{ .name = "tf_ping_pong_fixture_event" }); }

const T = std.testing;
fn produce(w: Writer, ready: *std.atomic.Value(u32), go: *std.atomic.Value(bool)) void {
    _ = ready.fetchAdd(1, .acq_rel);
    while (!go.load(.acquire)) std.atomic.spinLoopHint();
    for (0..12) |i| w.put(1, .{i, w.producer, 0, 0, 0, 0, 0, 0});
}
test "ping_pong_fixture observer overlapping Harness and peer producers stay distinct and survive joins" {
    const a = try Recorder.create(T.allocator, 101, 1); defer T.allocator.destroy(a);
    const b = try Recorder.create(T.allocator, 202, 3); defer T.allocator.destroy(b);
    var ready: std.atomic.Value(u32) = .init(0);
    var go: std.atomic.Value(bool) = .init(false);
    var ts: [4]std.Thread = undefined;
    const ws = [_]Writer{ a.writer(0), a.writer(1), b.writer(0), b.writer(2) };
    for (ws, &ts) |w, *t| t.* = try std.Thread.spawn(.{}, produce, .{w, &ready, &go});
    while (ready.load(.acquire) != 4) std.atomic.spinLoopHint();
    go.store(true, .release);
    for (ts) |t| t.join();
    a.joined = true; b.joined = true;
    for (ws) |w| {
        const s = &w.rec.?.streams[w.producer];
        try T.expectEqual(@as(usize, 12), s.len);
        for (s.events[0..s.len], 0..) |e, i| {
            try T.expectEqual(@as(u64, i), e.data[0]);
            try T.expectEqual(@as(u64, w.producer), e.data[1]);
            if (i != 0) try T.expect(e.ns >= s.events[i-1].ns);
        }
    }
    try T.expectEqual(@as(usize, 0), a.streams[2].len);
    try T.expectEqual(@as(usize, 0), b.streams[1].len);
}
test "ping_pong_fixture observer copied header and slot scalars do not alias reused buffers" {
    const r = try Recorder.create(T.allocator, 1, 4); defer T.allocator.destroy(r);
    var data = [_]u64{ 3, 1, 5, 17, 2, 44, 55, 0 };
    var w = r.writer(0); w.seq = 3; w.slot = 1;
    w.put(2, data);
    @memset(&data, 99); w.seq = 4; w.slot = 0; w.put(2, data);
    try T.expectEqual(@as(u64, 17), r.streams[0].events[0].data[3]);
    try T.expectEqual(@as(u32, 3), r.streams[0].events[0].seq);
    try T.expectEqual(@as(u32, 1), r.streams[0].events[0].slot);
}
test "ping_pong_fixture observer bounded overflow is explicit not silent truncation" {
    const r = try Recorder.create(T.allocator, 1, 0); defer T.allocator.destroy(r);
    const w = r.writer(0);
    for (0..capacity+3) |_| w.put(3, @splat(0));
    try T.expectEqual(capacity, r.streams[0].len);
    try T.expectEqual(@as(u64, 3), r.streams[0].overflow);
}
test "ping_pong_fixture observer retains all three post failure dead and gave up tuples unchanged" {
    const r = try Recorder.create(T.allocator, 1, 0); defer T.allocator.destroy(r);
    for ([_][8]u64{ .{17,0,3,0,0,3,3,3}, .{1,1,3,1,0,3,3,3} }, 0..) |tuple, p| {
        var w = r.writer(@intCast(p));
        for (0..3) |i| { w.seq = @intCast(i+4); w.put(4, tuple); }
        try T.expectEqual(@as(usize, 3), r.streams[p].len);
        for (r.streams[p].events[0..3]) |e| try T.expectEqual(tuple, e.data);
    }
}
