//! qualification_9051d0c1's checks of the AFD attention role (qualification_25364d42 PLAN §2, §5): the seam (Fwd.moe, cuda_layer.ffnHalf) against both
//! forward fixtures with a stand-in provider; the provider's request bytes against python 2.15's own send (G-wire-A);
//! its launches for a MoE layer's FFN half at world 1, and its refusals. G-trace-A: cuda_afd_trace_test.zig.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const layer = @import("cuda_layer.zig");
const afd = @import("cuda_afd_attn.zig");
const lt = @import("cuda_layer_test.zig");

const proto = cuda.xchg.proto;
const testing = std.testing;
const J = std.json.Value;

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn constOf(e: trace.Event, name: []const u8) ?i64 {
    for (e.consts) |c| if (eql(c.name, name)) return c.int;
    return null;
}

/// A stand-in for the AFD attention's MoE provider (python 2.15's w.afd): one event, three planes at b.gath.
const StandIn = struct {
    calls: usize = 0,

    fn block(ctx: *anyopaque, f: *fwd.Fwd, l: *const views.Layer, b: *const bufs.Buffers, R: usize, opt: layer.Opt) anyerror!fwd.Planes {
        _ = opt;
        const s: *StandIn = @ptrCast(@alignCast(ctx));
        s.calls += 1;
        if (l.moe == null) return error.NotMoe;
        try f.rec.?.call(.ext, "afd.stand_in", &.{ .{ .t = layer.rowsOf(b.normed, R) }, .{ .int = @intCast(R) } });
        return .{ .g = b.gath.ptr, .world = 3 };
    }
};

test "the seam: a MoE provider takes the MoE layers' FFN block, hc_post sums its planes; dense layers unchanged" {
    const gpa = testing.allocator;
    for ([_][]const u8{ lt.fixture_mlx, lt.fixture_exl3 }) |text| {
        const parsed = try std.json.parseFromSlice(J, gpa, text, .{});
        defer parsed.deinit();
        var r = try lt.Rank.init(gpa, parsed.value);
        defer r.deinit(gpa);
        var t = trace.Trace.init(gpa);
        defer t.deinit();
        var s: StandIn = .{};
        var f = fwd.Fwd.recording(&r.m, &t);
        f.moe = .{ .ctx = &s, .block = StandIn.block };
        try layer.ffnHalf(&f, &r.w.layers[3], &r.b, 4, .{});
        const moe_events = t.events.items.len + t.tri_rec.n;
        try layer.ffnHalf(&f, &r.w.layers[0], &r.b, 4, .{}); // a dense layer: the provider is not asked
        try f.flush();
        try testing.expectEqual(@as(usize, 1), s.calls);
        const ev = t.events.items;
        try testing.expectEqual(@as(usize, 4), moe_events);
        try testing.expectEqualStrings("_hc_partial", ev[0].name);
        try testing.expectEqualStrings("_hc_finish", ev[1].name);
        try testing.expectEqualStrings("afd.stand_in", ev[2].name);
        try testing.expectEqualStrings("_hc_post", ev[3].name);
        try testing.expectEqual(@as(?i64, 3), constOf(ev[3], "WORLD"));
        for (ev[3].args) |a| if (eql(a.name, "G")) try testing.expectEqual(r.b.gath.ptr, a.value.ptr.addr);
        const last = ev[ev.len - 1];
        try testing.expectEqualStrings("_hc_post", last.name);
        try testing.expectEqual(@as(?i64, @intCast(r.m.world)), constOf(last, "WORLD"));
        for (ev) |e| try testing.expect(!eql(e.name, "afd.stand_in") or e.op == .ext);
    }
}

/// python 2.15's own afd_mcdma.McdmaTransport publish/header/send on CPU (evidence/qualification_9051d0c1/scripts/wire_ref.py at
/// src@0849f3f), on the inputs `wireInputs` builds: each request's offsets and the sha256 of its bytes.
const WireCase = struct { rows: usize, seq: u32, mode: afd.Mode, pos: usize, pick: usize, wts: usize, end: usize, sha: []const u8 };
const wire_cases = [_]WireCase{
    .{ .rows = 1, .seq = 77, .mode = .eager, .pos = 5, .pick = 8448, .wts = 8704, .end = 8740, .sha = "bf91b9d46e7df64757191762f584af94d5a00ae4000177a2e5bcc41e5bbefb9b" },
    .{ .rows = 7, .seq = 78, .mode = .eager, .pos = 41, .pick = 57600, .wts = 57856, .end = 58108, .sha = "2319eb401c08034ce42509d9667595d6d9c6a9c3f819c98cfc3b8f1e1b7389a4" },
    .{ .rows = 2048, .seq = 79, .mode = .chunk, .pos = 0, .pick = 16777472, .wts = 16851200, .end = 16924928, .sha = "5e9e220e62718daedc1218cf23a31861ade3a7c646e6d775ea1591e4198587cb" },
};

/// wire_ref.py's rows: normed bf16 bits (k 40503 + 7) & 0xFFFF, pick (j 37 + 5) % 288, wts fp32 bits (j 2654435761 + 1) & 0x3FFFFFFF.
const WireInputs = struct { normed: []u16, pick: []i32, wts: []u32 };

fn wireInputs(gpa: std.mem.Allocator, R: usize, D: usize, S: usize) !WireInputs {
    const x: WireInputs = .{ .normed = try gpa.alloc(u16, R * D), .pick = try gpa.alloc(i32, R * S), .wts = try gpa.alloc(u32, R * S) };
    for (x.normed, 0..) |*v, k| v.* = @truncate((k * 40503 + 7) & 0xFFFF);
    for (x.pick, 0..) |*v, j| v.* = @intCast((j * 37 + 5) % 288);
    for (x.wts, 0..) |*v, j| v.* = @truncate((j * 2654435761 + 1) & 0x3FFFFFFF);
    return x;
}

test "G-wire-A: the request for R = 1, 7, 2,048 is python 2.15's afd_mcdma.send, byte for byte (header, offsets, pad)" {
    const gpa = testing.allocator;
    const m = try md.Model.from(&cfg.Config{}, .{ .quant = .exl3 }, 1, 0);
    var a = try afd.Afd.init(&m, 2048);
    try testing.expectEqual(@as(usize, 42), a.layers);
    const D = m.hidden;
    const S = m.top_k + 1;
    const table = try gpa.alloc(u8, a.layers * proto.HDR_BYTES);
    defer gpa.free(table);
    for (wire_cases) |c| {
        const R = c.rows;
        a.seq = c.seq - 1;
        try a.begin(R, c.mode, null);
        a.headers(table); // every MoE layer's row; the request carries row c.pos
        const x = try wireInputs(gpa, R, D, S);
        defer {
            gpa.free(x.normed);
            gpa.free(x.pick);
            gpa.free(x.wts);
        }
        const o = a.layout.offsets(R);
        const wire = proto.wireBytes(o.end);
        const req = try gpa.alloc(u8, wire);
        defer gpa.free(req);
        @memset(req, 0xAA);
        for (afd.pack(a.layout, R, @intFromPtr(table[c.pos * proto.HDR_BYTES ..].ptr), @intFromPtr(x.normed.ptr), @intFromPtr(x.pick.ptr), @intFromPtr(x.wts.ptr))) |p| {
            const dst = req[p.dst..][0..p.bytes];
            if (p.src) |s| @memcpy(dst, @as([*]const u8, @ptrFromInt(s))[0..p.bytes]) else @memset(dst, 0);
        }
        errdefer std.debug.print("rows {d}\n", .{R});
        try testing.expectEqual(proto.Layout.Offsets{ .normed = 256, .pick = c.pick, .wts = c.wts, .end = c.end }, o);
        const words = [_]i32{ proto.MAGIC, @intCast(c.seq), @intCast(R), @intCast(c.pos), @intFromEnum(c.mode), 42, 0, 0, 0, 0, 0 };
        for (words, 0..) |w, i| try testing.expectEqual(w, std.mem.readInt(i32, req[i * 4 ..][0..4], .little));
        try testing.expect(std.mem.allEqual(u8, req[44..proto.HDR_BYTES], 0));
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(x.normed), req[o.normed..o.pick]);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(x.pick), req[o.pick..][0 .. 4 * S * R]);
        try testing.expect(std.mem.allEqual(u8, req[o.pick + 4 * S * R .. o.wts], 0)); // python's zero pad
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(x.wts), req[o.wts..o.end]);
        try testing.expect(std.mem.allEqual(u8, req[o.end..wire], 0)); // the wire's 16-byte tail
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(req[0..o.end], &sha, .{});
        try testing.expectEqualStrings(c.sha, &std.fmt.bytesToHex(sha, .lower));
    }
}

/// The EXL3 fixture's model at world 1 (the attention node), cut to `layers`, with its weights and buffers placed.
const World1 = struct {
    parsed: std.json.Parsed(J),
    m: md.Model,
    w: views.Weights,
    b: bufs.Buffers,
    space: trace.Space,

    fn init(gpa: std.mem.Allocator, layers: usize, x: *World1) !void {
        x.parsed = try std.json.parseFromSlice(J, gpa, lt.fixture_exl3, .{});
        errdefer x.parsed.deinit();
        x.m = try md.Model.parse(gpa, x.parsed.value.object.get("config_json").?.string, 1, 0);
        try x.m.cut(layers);
        x.w = try views.rank(gpa, &x.m);
        errdefer x.w.deinit(gpa);
        x.b = try bufs.Buffers.init(&x.m, 8, 2560, false);
        x.space = .{ .gpa = gpa };
        errdefer x.space.deinit();
        try views.walk(&x.w, views.Path.init("w"), &x.space, trace.placeAll);
        try views.walk(&x.b, views.Path.init("b"), &x.space, trace.placeAll);
    }

    fn deinit(x: *World1, gpa: std.mem.Allocator) void {
        x.space.deinit();
        x.w.deinit(gpa);
        x.parsed.deinit();
    }
};

test "the provider at world 1: moe_route, the exchange, the shared expert, collect; hc_post sums afd_recv's three planes" {
    const gpa = testing.allocator;
    var x: World1 = undefined;
    try World1.init(gpa, 4, &x);
    defer x.deinit(gpa);
    var a = try afd.Afd.init(&x.m, 8);
    try testing.expectEqual(@as(usize, 1), a.layers);
    try x.space.place("b.afd_recv", &a.afd_recv);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var f = fwd.Fwd.recording(&x.m, &t);
    f.moe = a.provider();
    const want = [_][]const u8{ "_hc_partial", "_hc_finish", "_router_part", "_router_sum", "_topk", "afd.exchange", "_bmm", "_reduce", "_swiglu", "_bmm", "_reduce", "afd.collect", "_hc_post" };
    for ([_]usize{ 1, 2, 4, 8 }, 1..) |R, forwards| {
        t.reset();
        try a.begin(R, .eager, null);
        try layer.ffnHalf(&f, &x.w.layers[3], &x.b, R, .{});
        try f.flush();
        const ev = t.events.items;
        errdefer std.debug.print("rows {d}: {d} events\n", .{ R, ev.len });
        try testing.expectEqual(want.len, ev.len);
        for (want, ev) |n, e| try testing.expectEqualStrings(n, e.name);
        const ex = ev[5].vals;
        try testing.expectEqualStrings("b.normed", x.space.name(ex[0].t.ptr).?.path);
        try testing.expectEqual([2]usize{ R, x.m.hidden }, ex[0].t.shape[0..2].*);
        try testing.expectEqualStrings("b.pick", x.space.name(ex[1].t.ptr).?.path);
        try testing.expectEqualStrings("b.wts", x.space.name(ex[2].t.ptr).?.path);
        try testing.expectEqual(@as(i64, @intCast(R)), ex[3].int);
        const co = ev[11].vals;
        try testing.expectEqual(@as(usize, 5), co.len);
        for (co[0..3], 0..) |p, i| {
            const at = x.space.name(p.t.ptr).?;
            try testing.expectEqualStrings("b.afd_recv", at.path);
            try testing.expectEqual(@as(u64, 4 * i * R * x.m.hidden), at.off); // plane i, R rows apart
        }
        try testing.expectEqualStrings("b.sy", x.space.name(co[3].t.ptr).?.path);
        try testing.expectEqual(@as(i64, @intCast(R)), co[4].int);
        try testing.expectEqual(@as(?i64, 3), constOf(ev[12], "WORLD"));
        for (ev[12].args) |g| if (eql(g.name, "G")) try testing.expectEqualStrings("b.afd_recv", x.space.name(g.value.ptr.addr).?.path);
        try testing.expectEqual(@as(usize, 1), a.pos);
        try testing.expectEqual(@as(u32, @intCast(forwards)), a.seq); // one forward each
    }
}

test "the provider's refusals: world 2, MLX, a dense layer, rows other than begin's, a forward stopped between exchanges" {
    const gpa = testing.allocator;
    const w2 = try md.Model.from(&cfg.Config{}, .{ .quant = .exl3 }, 2, 0);
    try testing.expectError(error.NotWorld1, afd.Afd.init(&w2, 8));
    const mlx = try md.Model.from(&cfg.Config{}, .{}, 1, 0);
    try testing.expectError(error.NotExl3, afd.Afd.init(&mlx, 8));
    var x: World1 = undefined;
    try World1.init(gpa, 5, &x); // layers 3 and 4 are MoE
    defer x.deinit(gpa);
    var a = try afd.Afd.init(&x.m, 8);
    try testing.expectEqual(@as(usize, 2), a.layers);
    try testing.expectEqual(@as(?usize, 1), afd.moePos(&x.m, 4));
    try testing.expectEqual(@as(?usize, null), afd.moePos(&x.m, 2));
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var f = fwd.Fwd.recording(&x.m, &t);
    const p = a.provider();
    try a.begin(1, .chunk, null);
    try testing.expectError(error.Rows, a.begin(9, .eager, null));
    try testing.expectError(error.NotPorted, a.begin(1, .pair, null));
    try a.begin(4, .eager, null);
    try testing.expectError(error.NotMoe, p.block(p.ctx, &f, &x.w.layers[0], &x.b, 4, .{}));
    try testing.expectError(error.Rows, p.block(p.ctx, &f, &x.w.layers[3], &x.b, 2, .{}));
    try testing.expectError(error.OutOfStep, p.block(p.ctx, &f, &x.w.layers[4], &x.b, 4, .{})); // layer 4 before layer 3
    _ = try p.block(p.ctx, &f, &x.w.layers[3], &x.b, 4, .{});
    try testing.expectError(error.OutOfStep, a.begin(4, .eager, null)); // stopped after 1 of 2 exchanges: python's die
    _ = try p.block(p.ctx, &f, &x.w.layers[4], &x.b, 4, .{});
    try testing.expectError(error.ExpertFailed, a.begin(4, .eager, null)); // and dead stays dead
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "qualification_969df694: TF_GLM_EXL3_SEQ on the attention: unset, empty or 0 off, 1 on (nodeConfig's settings end in seq=1, after pairs=1 inflight=2), anything else BadExl3Seq; unset, the settings text is qualification_81840057's" {
    defer _ = unsetenv("TF_GLM_EXL3_SEQ");
    _ = unsetenv("TF_GLM_EXL3_SEQ");
    try testing.expect(!try afd.seqFromEnv());
    for ([_][:0]const u8{ "", "0" }) |v| {
        _ = setenv("TF_GLM_EXL3_SEQ", v, 1);
        try testing.expect(!try afd.seqFromEnv());
    }
    _ = setenv("TF_GLM_EXL3_SEQ", "1", 1);
    try testing.expect(try afd.seqFromEnv());
    for ([_][:0]const u8{ "2", "on", "true", "yes", " 1", "01", "-1" }) |v| {
        _ = setenv("TF_GLM_EXL3_SEQ", v, 1);
        try testing.expectError(error.BadExl3Seq, afd.seqFromEnv());
    }
    const m = try md.Model.from(&cfg.Config{}, .{ .quant = .exl3 }, 1, 0);
    var a = try afd.Afd.init(&m, 2048);
    var tb: [256]u8 = undefined;
    const qualification_81840057 = "tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1";
    try testing.expectEqualStrings(qualification_81840057, try a.nodeConfig(2048, 1, 1, 5, true).settings.text(&tb)); // seq unset
    a.exl3_seq = true;
    try testing.expectEqualStrings(qualification_81840057 ++ " seq=1", try a.nodeConfig(2048, 1, 1, 5, true).settings.text(&tb));
    a.paired = true;
    a.inflight = 2;
    try testing.expectEqualStrings(qualification_81840057 ++ " pairs=1 inflight=2 seq=1", try a.nodeConfig(2048, 1, 1, 5, true).settings.text(&tb));
    a.exl3_seq = false;
    try testing.expectEqualStrings(qualification_81840057 ++ " pairs=1 inflight=2", try a.nodeConfig(2048, 1, 1, 5, true).settings.text(&tb));
}

test "qualification_c863b626: the attention's settings with a halves value beside FLIGHT and SEQ: qualification_81840057's text, then pairs=1 inflight=2 seq=1, then halves=<value>" {
    const m = try md.Model.from(&cfg.Config{}, .{ .quant = .exl3 }, 1, 0);
    var a = try afd.Afd.init(&m, 2048);
    var tb: [256]u8 = undefined;
    const qualification_81840057 = "tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1";
    a.paired = true;
    a.inflight = 2;
    a.exl3_seq = true;
    try testing.expectEqualStrings(qualification_81840057 ++ " pairs=1 inflight=2 seq=1", try a.nodeConfig(2048, 1, 1, 5, true).settings.text(&tb));
    for ([_][]const u8{ "g53kt3r", "k4" }) |v| {
        a.halves = v;
        var want: [256]u8 = undefined;
        const w = try std.fmt.bufPrint(&want, "{s} pairs=1 inflight=2 seq=1 halves={s}", .{ qualification_81840057, v });
        try testing.expectEqualStrings(w, try a.nodeConfig(2048, 1, 1, 5, true).settings.text(&tb));
    }
    a.halves = "";
    a.exl3_seq = false;
    a.paired = false;
    a.inflight = 1;
    try testing.expectEqualStrings(qualification_81840057, try a.nodeConfig(2048, 1, 1, 5, true).settings.text(&tb)); // unset: qualification_81840057's
}
