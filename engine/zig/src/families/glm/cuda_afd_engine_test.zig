//! qualification_0fa2bee9's checks of the AFD attention engine on a CPU, the launches recorded: a decode step's order (begin, B3's stage
//! and compute through the provider, the sync and the check, then the sample and the commit), a failed exchange failing
//! the step before its sample and refusing the next before any launch, prompt chunks begun MODE_CHUNK, and the refusals.
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const eng = @import("cuda_engine.zig");
const afd = @import("cuda_afd_attn.zig");
const ae = @import("cuda_afd_engine.zig");
const lt = @import("cuda_layer_test.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const testing = std.testing;
const J = std.json.Value;

/// The attention node recorded: the EXL3 fixture's model at world 1 cut to `layers` (no MTP, as python's AFD), B3's
/// Engine and the provider placed at stand-in addresses, and an attention Node (never attended) for its status words.
const Rec = struct {
    parsed: std.json.Parsed(J),
    m: md.Model,
    w: views.Weights,
    e: *eng.Engine,
    a: afd.Afd,
    space: trace.Space,
    t: trace.Trace,
    region: []align(4096) u8,
    status: []u32,
    node: xchg.Node,

    fn init(x: *Rec, gpa: std.mem.Allocator, layers: usize, chunk: usize) !void {
        x.parsed = try std.json.parseFromSlice(J, gpa, lt.fixture_exl3, .{});
        x.m = try md.Model.parse(gpa, x.parsed.value.object.get("config_json").?.string, 1, 0);
        try x.m.cut(layers);
        x.m.mtp_layers = 0;
        x.w = try views.rank(gpa, &x.m);
        x.e = try eng.Engine.create(gpa, &x.m, &x.w, 2560, 8, chunk, false);
        x.a = try afd.Afd.init(&x.m, chunk);
        x.space = .{ .gpa = gpa };
        try views.walk(&x.w, views.Path.init("w"), &x.space, trace.placeAll);
        try x.e.walk(&x.space, trace.placeAll);
        try x.e.walkTmp(&x.space, trace.placeAll);
        x.e.tmp.space = &x.space;
        try x.space.place("b.afd_recv", &x.a.afd_recv);
        x.t = trace.Trace.init(gpa);
        const settings: proto.Settings = .{ .layout = x.a.layout, .max_rows = 8, .ports = 1 };
        const total = (proto.Region{ .slot = settings.slotBytes() }).total();
        x.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(total));
        @memset(x.region, 0);
        x.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(x.status, 0);
        const mem: xchg.Memory = .{ .region = x.region, .region_dev = @intFromPtr(x.region.ptr), .status = x.status, .status_dev = @intFromPtr(x.status.ptr) };
        x.node = try xchg.Node.init(gpa, testing.io, .attn, .{ .settings = settings, .warn_failures = true }, mem);
    }

    fn deinit(x: *Rec, gpa: std.mem.Allocator) void {
        x.node.deinit();
        gpa.free(x.region);
        gpa.free(x.status);
        x.t.deinit();
        x.space.deinit();
        x.e.destroy(gpa);
        x.w.deinit(gpa);
        x.parsed.deinit();
    }
};

/// The exchange's stand-in: the headers posted, the takes counted; the exchange fails at take `fail_take`.
const Stub = struct {
    hdrs: [16]proto.Header = undefined,
    posts: usize = 0,
    takes: usize = 0,
    fail_take: ?usize = null,

    fn host(s: *Stub) afd.Host {
        return .{ .ctx = s, .post = post, .take = take };
    }

    fn post(ctx: *anyopaque, node: *xchg.Node, hdr: *const [proto.HDR_BYTES]u8, pos: usize, R: usize) anyerror!void {
        _ = node;
        const s: *Stub = @ptrCast(@alignCast(ctx));
        const h = try proto.Header.read(hdr);
        if (h.layer != pos or h.rows != R) return error.HeaderNotPos;
        s.hdrs[s.posts] = h;
        s.posts += 1;
    }

    fn take(ctx: *anyopaque, node: *xchg.Node, pos: usize, R: usize) anyerror!void {
        _ = .{ pos, R };
        const s: *Stub = @ptrCast(@alignCast(ctx));
        s.takes += 1;
        if (s.fail_take != null and s.fail_take.? == s.takes) node.fail(.{ .why = .silent, .peer = 1, .seq = @intCast(s.posts) });
    }
};

fn find(ev: []const trace.Event, name: []const u8, last: bool) ?usize {
    var at: ?usize = null;
    for (ev, 0..) |e, i| if (std.mem.eql(u8, e.name, name)) {
        if (!last) return i;
        at = i;
    };
    return at;
}

fn expectHeader(h: proto.Header, fseq: i32, rows: i32, layer: i32, mode: proto.Mode) !void {
    try testing.expectEqual(proto.Header{ .fseq = fseq, .rows = rows, .layer = layer, .mode = mode, .layers = 2 }, h);
}

test "the decode step at world 1: begin, stage and compute through the provider, the check, then the sample and the commit; a failed exchange fails its step before the sample, and the next before any launch" {
    const gpa = testing.allocator;
    var x: Rec = undefined;
    try x.init(gpa, 5, 64); // layers 3 (DSA) and 4 (KDA) are MoE: two exchanges a forward
    defer x.deinit(gpa);
    var s: Stub = .{};
    x.a.wire = .{ .node = &x.node, .host = s.host() };
    var at = try ae.Attn.init(x.e, &x.a);
    var f = fwd.Fwd.recording(&x.m, &x.t);
    _ = try at.step(&f, 11);
    try f.flush();
    var ev = x.t.events.items;
    const collect = find(ev, "afd.collect", true).?;
    const topk = find(ev, "torch.topk", false).?;
    try testing.expect(find(ev, "staged.record", false).? < find(ev, "afd.exchange", false).?);
    try testing.expect(collect < topk and topk < find(ev, "torch.fill_", true).?); // sample, then commit's set_pos
    try testing.expectEqual(@as(usize, 2), s.posts);
    try expectHeader(s.hdrs[0], 1, 1, 0, .eager);
    try expectHeader(s.hdrs[1], 1, 1, 1, .eager);
    try testing.expectEqual(@as(usize, 1), x.e.st.pos);
    // the next step's exchange fails at its second layer: no sample, no commit, the reason kept
    x.t.reset();
    s.fail_take = 4;
    try testing.expectError(error.ExpertFailed, at.step(&f, 12));
    try f.flush();
    ev = x.t.events.items;
    try testing.expect(find(ev, "afd.collect", true) != null);
    for ([_][]const u8{ "torch.float", "torch.topk", "torch.cpu", "torch.fill_" }) |n| try testing.expectEqual(@as(?usize, null), find(ev, n, false));
    try testing.expectEqual(@as(usize, 1), x.e.st.pos);
    try testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 4, .gave_up = 0, .breach = 0 }, at.failure().?);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("afd: failed silent peer 1 seq 4 gave_up 0 breach 0", ae.describe(at.failure().?, &buf));
    // sticky: the next step refuses before any launch; nothing more is posted
    x.t.reset();
    try testing.expectError(error.ExpertFailed, at.step(&f, 12));
    try testing.expectError(error.ExpertFailed, at.prefill(&f, &.{ 1, 2, 3 }));
    try f.flush();
    try testing.expectEqual(@as(usize, 0), x.t.events.items.len);
    try testing.expectEqual(@as(usize, 4), s.posts);
}

test "prompt chunks at world 1: each chunk begun MODE_CHUNK with its rows, committed; the first token after the last chunk's check; a failure in a chunk refuses the next" {
    const gpa = testing.allocator;
    var x: Rec = undefined;
    try x.init(gpa, 5, 64);
    defer x.deinit(gpa);
    var s: Stub = .{};
    x.a.wire = .{ .node = &x.node, .host = s.host() };
    var at = try ae.Attn.init(x.e, &x.a);
    var f = fwd.Fwd.recording(&x.m, &x.t);
    var ids: [100]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + i);
    _ = try at.prefill(&f, &ids); // chunks of 64 and 36 rows
    try f.flush();
    try testing.expectEqual(@as(usize, 4), s.posts);
    try expectHeader(s.hdrs[0], 1, 64, 0, .chunk);
    try expectHeader(s.hdrs[1], 1, 64, 1, .chunk);
    try expectHeader(s.hdrs[2], 2, 36, 0, .chunk);
    try expectHeader(s.hdrs[3], 2, 36, 1, .chunk);
    try testing.expectEqual(@as(usize, 100), x.e.st.pos);
    const ev = x.t.events.items;
    try testing.expect(find(ev, "afd.collect", true).? < find(ev, "torch.topk", false).?);
    _ = try at.step(&f, 7);
    try expectHeader(s.hdrs[4], 3, 1, 0, .eager);
    // a fresh prompt whose first chunk's exchange fails: the second chunk is refused, nothing sampled
    x.t.reset();
    s.fail_take = 7;
    try testing.expectError(error.ExpertFailed, at.prefill(&f, &ids));
    try f.flush();
    try testing.expectEqual(@as(usize, 8), s.posts); // the first chunk's two requests, none of the second's
    try testing.expectEqual(@as(?usize, null), find(x.t.events.items, "torch.topk", false));
    try testing.expectEqual(@as(usize, 64), x.e.st.pos);
}

test "the engine's refusals: world 2, MTP under AFD, the provider of another model or narrower than the engine's windows" {
    const gpa = testing.allocator;
    var x: Rec = undefined;
    try x.init(gpa, 5, 64);
    defer x.deinit(gpa);
    var narrow = try afd.Afd.init(&x.m, 8);
    try testing.expectError(error.Rows, ae.Attn.init(x.e, &narrow));
    var m2 = x.m;
    var other = try afd.Afd.init(&m2, 64);
    try testing.expectError(error.OtherModel, ae.Attn.init(x.e, &other));
    const mbuf = x.e.mbuf;
    x.e.mbuf = x.e.buf;
    try testing.expectError(error.MtpUnderAfd, ae.Attn.init(x.e, &x.a));
    x.e.mbuf = mbuf;
    const w1 = x.e.m;
    var w2 = x.m;
    w2.world = 2;
    x.e.m = &w2;
    try testing.expectError(error.NotWorld1, ae.Attn.init(x.e, &x.a));
    x.e.m = w1;
}
