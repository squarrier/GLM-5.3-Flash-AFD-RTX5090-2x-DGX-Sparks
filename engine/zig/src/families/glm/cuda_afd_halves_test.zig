//! The halves' pair checks on CPU (qualification_c2d4ee2e): TF_AFD_PAIRS's switch, python 2.15's next_forward and next_partner on a
//! pair's two first headers (afd_mcdma.py:475-481, 512-541), WIRE.md §6's words, and B's last header after the pair.
const std = @import("std");
const cuda = @import("cuda");
const ae = @import("cuda_afd_expert.zig");
const hv = @import("cuda_afd_halves.zig");
const proto = cuda.xchg.proto;

const testing = std.testing;

/// A pair's request as the attention sends it (WIRE.md §3): A = fseq 7 of 2,048 rows, B = 8 of 593 (python's 2.6K test).
fn req(slot: i32, fseq: i32, rows: i32, pos: i32, other: i32) proto.Header {
    return .{ .fseq = fseq, .rows = rows, .layer = pos, .mode = .pair, .layers = 42, .extra = .{ slot, other, 7, 0 } };
}

test "TF_AFD_PAIRS: unset or 0 off, 1 on, anything else refused" {
    try testing.expect(!try hv.parse(null));
    try testing.expect(!try hv.parse("0"));
    try testing.expect(try hv.parse("1"));
    for ([_][]const u8{ "", "2", "yes", "true", " 1" }) |v| try testing.expectError(error.BadAfdPairs, hv.parse(v));
}

test "qualification_c533375f: TF_AFD_INFLIGHT on the halves: unset or 1 one in flight, 2 two with the pairs only, anything else refused" {
    try testing.expectEqual(@as(u8, 1), try hv.parseInflight(null, false));
    try testing.expectEqual(@as(u8, 1), try hv.parseInflight(null, true));
    try testing.expectEqual(@as(u8, 1), try hv.parseInflight("1", true));
    try testing.expectEqual(@as(u8, 2), try hv.parseInflight("2", true));
    try testing.expectError(error.BadAfdInflight, hv.parseInflight("2", false)); // without TF_AFD_PAIRS=1
    for ([_][]const u8{ "", "0", "3", "yes", " 2" }) |v| try testing.expectError(error.BadAfdInflight, hv.parseInflight(v, true));
}

test "a pair's first request (A_0): python next_forward's checks and WIRE.md's words; window A runs every position" {
    const p = try hv.open(req(0, 7, 2048, 0, 593), 0, 42, 2048);
    try testing.expectEqual(ae.Forward{ .fseq = 7, .rows = 2048, .mode = .pair, .first = 0, .count = 42 }, p.a);
    try testing.expectEqual(@as(usize, 593), p.b_rows);
    try testing.expectEqual(@as(u32, 84), p.requests());
    try testing.expectEqual(@as(i32, 8), p.bSeq());
    // TF_AFD_MTP=1: 43 positions, word 6 the backbone's 42
    var m = req(0, 7, 2048, 0, 2048);
    m.layers = 43;
    const pm = try hv.open(m, 42, 43, 2048);
    try testing.expectEqual([2]usize{ 0, 42 }, [2]usize{ pm.a.first, pm.a.count });
    // the refusals
    const ex = testing.expectError;
    try ex(error.WrongLayer, hv.open(req(0, 7, 2048, 1, 593), 0, 42, 2048));
    try ex(error.WrongLayerCount, hv.open(req(0, 7, 2048, 0, 593), 0, 43, 2048));
    try ex(error.WrongRows, hv.open(req(0, 7, 2049, 0, 593), 0, 42, 2048));
    try ex(error.WrongRows, hv.open(req(0, 7, 0, 0, 593), 0, 42, 2048));
    try ex(error.WrongPair, hv.open(req(1, 7, 2048, 0, 593), 0, 42, 2048)); // B's first before A's: out of order
    try ex(error.WrongPair, hv.open(req(0, 6, 2048, 0, 593), 0, 42, 2048)); // word 9 not its own fseq
    try ex(error.WrongPair, hv.open(req(0, 7, 2048, 0, 0), 0, 42, 2048)); // no partner rows
    try ex(error.WrongPair, hv.open(req(0, 7, 2048, 0, 2049), 0, 42, 2048));
    try ex(error.WrongPair, hv.open(req(0, 7, 2048, 0, -5), 0, 42, 2048));
    try ex(error.WrongPair, hv.open(req(0, 7, 2048, 0, 593), 43, 42, 2048)); // a count past the positions
    try ex(error.WrongPair, hv.open(req(0, 7, 2048, 0, 593), -1, 42, 2048));
    var lanes = req(0, 7, 2048, 0, 593);
    lanes.extra[3] = 3; // python's lanes word (mode 9's): not a pair's
    try ex(error.WrongPair, hv.open(lanes, 0, 42, 2048));
    var eager = req(0, 7, 8, 0, 8);
    eager.mode = .eager;
    try ex(error.WrongMode, hv.open(eager, 0, 42, 2048));
}

test "a pair's second window (B_0): python next_partner, words 1-10 against A's; then B's forward, slot 1" {
    const p = try hv.open(req(0, 7, 2048, 0, 593), 0, 42, 2048);
    const b = try hv.partner(p, req(1, 8, 593, 0, 2048), 0);
    try testing.expectEqual(ae.Forward{ .fseq = 8, .rows = 593, .mode = .pair, .first = 0, .count = 42, .slot = 1 }, b);
    const ex = testing.expectError;
    try ex(error.WrongPartner, hv.partner(p, req(0, 8, 593, 0, 2048), 0)); // a second slot 0
    try ex(error.WrongPartner, hv.partner(p, req(1, 9, 593, 0, 2048), 0)); // not A's fseq + 1
    try ex(error.WrongPartner, hv.partner(p, req(1, 8, 592, 0, 2048), 0)); // not the rows A announced
    try ex(error.WrongPartner, hv.partner(p, req(1, 8, 593, 0, 2047), 0)); // not A's rows in word 8
    try ex(error.WrongPartner, hv.partner(p, req(1, 8, 593, 1, 2048), 0)); // past position 0
    try ex(error.WrongPartner, hv.partner(p, req(1, 8, 593, 0, 2048), 42)); // another count
    var other = req(1, 8, 593, 0, 2048);
    other.extra[2] = 8; // another pair's first
    try ex(error.WrongPartner, hv.partner(p, other, 0));
    var chunk = req(1, 8, 593, 0, 2048);
    chunk.mode = .chunk; // a stopped pair, then a solo chunk forward where B_0 belongs
    chunk.extra = @splat(0);
    try ex(error.WrongPartner, hv.partner(p, chunk, 0));
    var lay = req(1, 8, 593, 0, 2048);
    lay.layers = 43;
    try ex(error.WrongPartner, hv.partner(p, lay, 0));
    // B's fseq across the 31-bit wrap, as the attention masks it
    var w = req(0, 0x7FFFFFFF, 64, 0, 64);
    w.extra[2] = 0x7FFFFFFF;
    const pw = try hv.open(w, 0, 42, 2048);
    var bw = req(1, 0, 64, 0, 64);
    bw.extra[2] = 0x7FFFFFFF;
    try testing.expectEqual(@as(i32, 0), (try hv.partner(pw, bw, 0)).fseq);
}

test "B's last request after the pair: B's at position n - 1, or WrongPairEnd" {
    const p = try hv.open(req(0, 7, 2048, 0, 593), 0, 42, 2048);
    try hv.tail(p, req(1, 8, 593, 41, 2048), 0);
    const ex = testing.expectError;
    try ex(error.WrongPairEnd, hv.tail(p, req(1, 8, 593, 40, 2048), 0)); // a pair stopped short
    try ex(error.WrongPairEnd, hv.tail(p, req(0, 7, 2048, 41, 593), 0)); // A's, not B's
    var solo = req(1, 9, 64, 41, 0);
    solo.mode = .chunk;
    solo.extra = @splat(0);
    try ex(error.WrongPairEnd, hv.tail(p, solo, 0)); // another forward's request served as the pair's tail
    try ex(error.WrongPairEnd, hv.tail(p, req(1, 8, 593, 41, 2048), 42));
}

test "the headers the halves expect are python 2.15's header_pair words (IMP5 §3): mode 7, slot, other rows, first seq" {
    const p = try hv.open(req(0, 7, 2048, 0, 593), 0, 42, 2048);
    var a: [proto.HDR_BYTES]u8 = undefined;
    p.header(0, 7, 5).write(&a);
    var b: [proto.HDR_BYTES]u8 = undefined;
    p.header(1, 8, 5).write(&b);
    const words = struct {
        fn of(x: *const [proto.HDR_BYTES]u8) [11]i32 {
            var w: [11]i32 = undefined;
            for (&w, 0..) |*v, i| v.* = std.mem.readInt(i32, x[i * 4 ..][0..4], .little);
            return w;
        }
    }.of;
    try testing.expectEqual([11]i32{ proto.MAGIC, 7, 2048, 5, 7, 42, 0, 0, 593, 7, 0 }, words(&a));
    try testing.expectEqual([11]i32{ proto.MAGIC, 8, 593, 5, 7, 42, 0, 1, 2048, 7, 0 }, words(&b));
    try testing.expect(std.mem.allEqual(u8, a[44..], 0) and std.mem.allEqual(u8, b[44..], 0));
}

/// A proxy whose count reaches `target` only at its `after`-th read: the reply the loop posted still being carried.
const Lagging = struct {
    target: u64,
    after: u64,
    reads: u64 = 0,

    const vt: cuda.xchg.Proxy.VTable = .{ .blob = blob, .connect = connect, .start = start, .failure = failure, .ops = ops, .close = close };

    fn blob(_: *anyopaque, _: []u8) anyerror!usize {
        return 0;
    }
    fn connect(_: *anyopaque, _: []const u8) anyerror!void {}
    fn start(_: *anyopaque) anyerror!void {}
    fn failure(_: *anyopaque) ?[]const u8 {
        return null;
    }
    fn ops(ctx: *anyopaque) u64 {
        const l: *Lagging = @ptrCast(@alignCast(ctx));
        l.reads += 1;
        return if (l.reads >= l.after) l.target else l.target - 1;
    }
    fn close(_: *anyopaque) void {}
};

test "qualification_c533375f --fault exit with two in flight: B_0 (the faulty request) landed while A_0's reply is still with the proxy; the half dies only once the proxy has carried it, so every request before the fault is answered" {
    const gpa = testing.allocator;
    const settings: proto.Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 8, .ports = 1, .pairs = true, .inflight = 2 };
    const reg = settings.region();
    const region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(reg.total()));
    defer gpa.free(region);
    @memset(region, 0);
    const status = try gpa.alloc(u32, proto.STATUS_WORDS);
    defer gpa.free(status);
    @memset(status, 0);
    const mem: cuda.xchg.Memory = .{ .region = region, .region_dev = @intFromPtr(region.ptr), .status = status, .status_dev = @intFromPtr(status.ptr) };
    var n = try cuda.xchg.Node.init(gpa, testing.io, .{ .expert = 0 }, .{ .settings = settings, .warn_failures = true }, mem);
    defer n.deinit();
    var lag: Lagging = .{ .target = 1, .after = 50 };
    n.proxies[0] = .{ .ctx = &lag, .vt = &Lagging.vt };
    // the loop posted A_0's reply (its doorbell at 1); B_0, request 2, landed with A_0
    const s = n.src(0);
    req(1, 8, 4, 0, 4).write(region[@intCast(reg.recv(s, reg.slotOf(2)))..][0..proto.HDR_BYTES]);
    @atomicStore(u32, @as(*u32, @ptrCast(@alignCast(region.ptr + reg.flag(s, reg.slotOf(2), 0)))), 2, .release);
    @atomicStore(u32, @as(*u32, @ptrCast(@alignCast(region.ptr + reg.ctrl()))), 1, .release);
    try testing.expectEqual(ae.End.die, ae.fault(&n, .{ .what = .exit, .at = 2 }, 2));
    try testing.expectEqual(@as(u64, 50), lag.reads); // it died once the proxy's count reached the doorbell, not before
}
