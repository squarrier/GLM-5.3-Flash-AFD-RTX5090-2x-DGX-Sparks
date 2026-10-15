//! qualification_21e21bc0's checks of MTP drafts on the AFD attention node on a CPU, the launches recorded and the exchange a stand-in
//! (cuda_afd_engine_test.zig's way): with TF_AFD_MTP's provider every forward is a slice of the halves' positions (the
//! backbone's [0, L), the MTP head's [L, 1)) that the expert's header check takes; a drafted decode's forwards, rows and
//! positions are qualification_qualification_86ea5170's loops' (absorb, chained drafts, a verify window of k + 1 rows), its tokens the serial decode's;
//! an MTP step's failed exchange fails the request before its draw; without the MTP head the switch is refused.
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const cc = @import("cuda_compute.zig");
const eng = @import("cuda_engine.zig");
const afd = @import("cuda_afd_attn.zig");
const ae = @import("cuda_afd_engine.zig");
const ex = @import("cuda_afd_expert.zig");
const drafts = @import("cuda_afd_drafts.zig");
const glm_decode = @import("cuda_decode.zig");
const lt = @import("cuda_layer_test.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const testing = std.testing;
const J = std.json.Value;

/// The attention node recorded with the MTP head: the EXL3 fixture at world 1 cut to `layers` with its MTP layer kept
/// (as tf-glm-forward's --layers keeps it), B3's Engine with its mbuf, the provider serving the MTP position.
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
        const mtp_layers = x.m.mtp_layers;
        try x.m.cut(layers);
        x.m.mtp_layers = mtp_layers;
        x.w = try views.rank(gpa, &x.m);
        x.e = try eng.Engine.create(gpa, &x.m, &x.w, 2560, 8, chunk, false);
        x.a = try afd.Afd.initWith(&x.m, chunk, true);
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

/// The exchange's stand-in: each post's header (and word 6) and rows kept, the takes counted; the exchange fails at take
/// `fail_take`.
const Stub = struct {
    hdrs: [64]proto.Header = undefined,
    counts: [64]i32 = undefined,
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
        s.counts[s.posts] = std.mem.readInt(i32, hdr[24..28], .little);
        s.posts += 1;
    }

    fn take(ctx: *anyopaque, node: *xchg.Node, pos: usize, R: usize) anyerror!void {
        _ = .{ pos, R };
        const s: *Stub = @ptrCast(@alignCast(ctx));
        s.takes += 1;
        if (s.fail_take != null and s.fail_take.? == s.takes) node.fail(.{ .why = .silent, .peer = 1, .seq = @intCast(s.posts) });
    }

    /// Post i as the expert's first-header check reads it (positions [first, first + count) of `layers`).
    fn forward(s: *const Stub, i: usize, layers: usize) !ex.Forward {
        return ex.first(s.hdrs[i], s.counts[i], layers, 2048);
    }
};

/// Every draw answered with token 5 at probability 1 (each draft is drawn again by the verify: all kept), counted.
const Picks = struct {
    n: usize = 0,

    fn pick(ctx: *anyopaque, logits: trace.T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        _ = .{ logits, at };
        const p: *Picks = @ptrCast(@alignCast(ctx));
        @memset(out, 5);
        if (probs) |x| @memset(x, 1.0);
        p.n += 1;
    }
};

test "TF_AFD_MTP's slices: a backbone step is positions [0, 2) of 3, an MTP head step position 2 alone; the expert's first-header check takes both" {
    const gpa = testing.allocator;
    var x: Rec = undefined;
    try x.init(gpa, 5, 64); // layers 3 (DSA) and 4 (KDA) are MoE, the MTP layer the third position
    defer x.deinit(gpa);
    var s: Stub = .{};
    x.a.wire = .{ .node = &x.node, .host = s.host() };
    var at = try ae.Attn.init(x.e, &x.a);
    try testing.expectEqual(@as(usize, 3), x.a.positions());
    var f = fwd.Fwd.recording(&x.m, &x.t);
    _ = try at.step(&f, 11);
    try testing.expectEqual(@as(usize, 2), s.posts);
    for (0..2) |i| {
        try testing.expectEqual(proto.Header{ .fseq = 1, .rows = 1, .layer = @intCast(i), .mode = .slice, .layers = 3 }, s.hdrs[i]);
        try testing.expectEqual(@as(i32, 2), s.counts[i]);
    }
    try testing.expectEqual(ex.Forward{ .fseq = 1, .rows = 1, .mode = .slice, .first = 0, .count = 2 }, try s.forward(0, 3));
    x.t.reset();
    _ = try at.mtpStep(&f, &x.e.mbuf.?, &.{7}, cc.rows(x.e.last_hidden, 0, 1), .eager, null);
    try at.settled(&f);
    try f.flush();
    try testing.expectEqual(@as(usize, 3), s.posts);
    try testing.expectEqual(proto.Header{ .fseq = 2, .rows = 1, .layer = 2, .mode = .slice, .layers = 3 }, s.hdrs[2]);
    try testing.expectEqual(ex.Forward{ .fseq = 2, .rows = 1, .mode = .slice, .first = 2, .count = 1 }, try s.forward(2, 3));
    var exchanges: usize = 0;
    for (x.t.events.items) |e| exchanges += @intFromBool(std.mem.eql(u8, e.name, "afd.exchange"));
    try testing.expectEqual(@as(usize, 1), exchanges); // the MTP layer's MoE through the provider, nothing local
    // the halves' check refuses a mismatch: a backbone header against a half without the MTP position, and the reverse
    try testing.expectError(error.WrongLayerCount, s.forward(0, 2));
    var off = s.hdrs[0];
    off.mode = .eager;
    off.layers = 2;
    try testing.expectError(error.WrongLayerCount, ex.first(off, 0, 3, 2048));
}

test "a drafted decode under AFD (f3, auto): qualification_qualification_86ea5170's absorb, chained drafts and verify windows of 4 rows; MTP steps one exchange at position 2, verify windows two of 4 rows; tokens equal the serial decode's" {
    const gpa = testing.allocator;
    var x: Rec = undefined;
    try x.init(gpa, 5, 64);
    defer x.deinit(gpa);
    var s: Stub = .{};
    x.a.wire = .{ .node = &x.node, .host = s.host() };
    var at = try ae.Attn.init(x.e, &x.a);
    var picks: Picks = .{};
    x.t.picker = .{ .ctx = &picks, .pick = Picks.pick };
    var f = fwd.Fwd.recording(&x.m, &x.t);
    var ids: [10]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + i);
    at.absorb = true;
    const first = try at.prefill(&f, &ids);
    try testing.expectEqual(@as(usize, 3), s.posts); // the chunk's two exchanges of 10 rows, then the absorb's of 9
    try testing.expectEqual(@as(i32, 10), s.hdrs[0].rows);
    try testing.expectEqual(proto.Header{ .fseq = 2, .rows = 9, .layer = 2, .mode = .slice, .layers = 3 }, s.hdrs[2]);
    try testing.expectEqual(@as(usize, 9), x.e.st.mtp_len);
    var out: [9]u32 = undefined;
    var pol: drafts.DepthPolicy = .{ .most = 3, .fixed = true };
    const d = try drafts.mtp(&at, &f, first, 9, &out, &pol, false, null);
    try testing.expectEqual(@as(usize, 2), d.rounds);
    try testing.expectEqual(@as(usize, 6), d.drafted);
    try testing.expectEqual(@as(usize, 6), d.accepted);
    try testing.expectEqual(@as(usize, 6), d.mtp_steps);
    // after prefill: (absorb 1 row, 2 drafts) at position 2, a verify of 4 rows at 0 and 1, then (absorb 4 rows, 2 drafts), a verify
    const want = [_][2]i32{ .{ 2, 1 }, .{ 2, 1 }, .{ 2, 1 }, .{ 0, 4 }, .{ 1, 4 }, .{ 2, 4 }, .{ 2, 1 }, .{ 2, 1 }, .{ 0, 4 }, .{ 1, 4 } };
    try testing.expectEqual(3 + want.len, s.posts);
    for (want, s.hdrs[3 .. 3 + want.len]) |w, h| try testing.expectEqual(w, [2]i32{ h.layer, h.rows });
    try testing.expectEqual(@as(usize, 10 + 8), x.e.st.pos);
    // the serial decode from the same prompt gives the same tokens (drafts only propose)
    const serial = try gpa.dupe(u32, d.tokens);
    defer gpa.free(serial);
    at.absorb = false;
    try testing.expectEqual(first, try at.prefill(&f, &ids));
    var out2: [9]u32 = undefined;
    try testing.expectEqualSlices(u32, serial, try at.decode(&f, first, 9, &out2, false));
    // auto: engine.py's greedy c3:0.35; every draw at probability 1 keeps 3 drafts a round
    at.absorb = true;
    _ = try at.prefill(&f, &ids);
    var pa = drafts.DepthPolicy.auto(true);
    const da = try drafts.auto(&at, &f, first, 9, &out, &pa, false, null);
    try testing.expectEqualSlices(u32, serial, da.tokens);
    try testing.expectEqual(@as(usize, 2), da.rounds);
    try testing.expectEqual([_]usize{ 0, 0, 0, 2, 0, 0, 0, 0 }, da.depths);
}

test "an MTP step's failed exchange fails the request before its draw; the next forward refuses before any launch" {
    const gpa = testing.allocator;
    var x: Rec = undefined;
    try x.init(gpa, 5, 64);
    defer x.deinit(gpa);
    var s: Stub = .{};
    x.a.wire = .{ .node = &x.node, .host = s.host() };
    var at = try ae.Attn.init(x.e, &x.a);
    var picks: Picks = .{};
    x.t.picker = .{ .ctx = &picks, .pick = Picks.pick };
    var f = fwd.Fwd.recording(&x.m, &x.t);
    const first = try at.prefill(&f, &.{ 1, 2, 3 });
    const draws = picks.n;
    s.fail_take = s.takes + 1; // the absorb's MTP forward
    var out: [5]u32 = undefined;
    var pol: drafts.DepthPolicy = .{ .most = 3, .fixed = true };
    try testing.expectError(error.ExpertFailed, drafts.mtp(&at, &f, first, 5, &out, &pol, false, null));
    try testing.expectEqual(draws, picks.n); // no draw from the failed forward
    const posts = s.posts;
    x.t.reset();
    try testing.expectError(error.ExpertFailed, at.step(&f, 3));
    try f.flush();
    try testing.expectEqual(@as(usize, 0), x.t.events.items.len);
    try testing.expectEqual(posts, s.posts);
}

test "TF_AFD_MTP needs the MTP head: a provider without it refused; an engine with mbuf and a provider without the switch refused" {
    const gpa = testing.allocator;
    var x: Rec = undefined;
    try x.init(gpa, 5, 64);
    defer x.deinit(gpa);
    var m0 = x.m;
    m0.mtp_layers = 0;
    try testing.expectError(error.NoMtp, afd.Afd.initWith(&m0, 64, true));
    var plain = try afd.Afd.init(&x.m, 64);
    try testing.expectError(error.MtpUnderAfd, ae.Attn.init(x.e, &plain));
    try testing.expectError(error.NoMtp, plain.beginMtp(1, .eager, null));
    try testing.expectEqual(@as(usize, 2), plain.positions());
}

/// Each draw logged by the buffer it reads (f: prefill's last row, m: a verify window, d: the MTP head) and its rows,
/// answered 5 (the verify windows' rows counted across draws: 6 at every fifth from the second, a rejection) at
/// probability 0.9; with `low` (draft_policy) every low-th MTP head draw at 0.5.
const Draws = struct {
    e: *const eng.Engine,
    n: usize = 0,
    vrows: usize = 0,
    low: usize = 0,
    dn: usize = 0,
    kinds: [256]u8 = undefined,
    rows: [256]usize = undefined,

    fn pick(ctx: *anyopaque, logits: trace.T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        _ = at;
        const l: *Draws = @ptrCast(@alignCast(ctx));
        const e = l.e;
        if (l.n >= l.kinds.len) return error.TooManyDraws;
        const k: u8 = if (logits.ptr == e.buf.logits.ptr) 'm' else if (logits.ptr == e.mbuf.?.logits.ptr) 'd' else 'f';
        l.kinds[l.n] = k;
        l.rows[l.n] = out.len;
        for (out, 0..) |*o, i| o.* = 5 + @as(u32, @intFromBool(k == 'm' and (l.vrows + i) % 5 == 1));
        if (k == 'm') l.vrows += out.len;
        const low = k == 'd' and l.low > 0 and l.dn % l.low == 0;
        if (k == 'd') l.dn += 1;
        if (probs) |x| @memset(x, if (low) 0.5 else 0.9);
        l.n += 1;
    }
};

/// The two loops a policy runs in.
const Loop = enum { mtp, auto };

/// One policy's decode of 12 tokens after a 10-token prompt: on B3's Engine by cuda_decode (the MTP layer's MoE local,
/// qualification_qualification_86ea5170's loops), and on the attention node by cuda_afd_drafts (the MoE through the provider), on the same answers.
fn sameAsDecode(kind: Loop, policy: drafts.DepthPolicy) !void {
    _ = try sameAsDecodeWith(kind, policy, 0);
}

/// sameAsDecode with every `low`-th head draw at probability 0.5 (draft_policy); returns cuda_decode's counters.
fn sameAsDecodeWith(kind: Loop, policy: drafts.DepthPolicy, low: usize) !drafts.Decoded {
    const gpa = testing.allocator;
    var ids: [10]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(2000 + i);
    var outs: [2][12]u32 = undefined;
    var got: [2]drafts.Decoded = undefined;
    var logs: [2]Draws = undefined;
    var after: [2][3]usize = undefined;
    for (0..2) |side| {
        var x: Rec = undefined;
        try x.init(gpa, 5, 64);
        defer x.deinit(gpa);
        var s: Stub = .{};
        x.a.wire = .{ .node = &x.node, .host = s.host() };
        logs[side] = .{ .e = x.e, .low = low };
        x.t.picker = .{ .ctx = &logs[side], .pick = Draws.pick };
        var f = fwd.Fwd.recording(&x.m, &x.t);
        var pol = policy;
        if (side == 0) {
            const first = try x.e.prefill(&f, &ids, true);
            got[0] = switch (kind) {
                .mtp => try glm_decode.mtp(x.e, &f, first, 12, &outs[0], &pol, false, null),
                .auto => try glm_decode.auto(x.e, &f, first, 12, &outs[0], &pol, false, null),
            };
        } else {
            var at = try ae.Attn.init(x.e, &x.a);
            at.absorb = true;
            const first = try at.prefill(&f, &ids);
            got[1] = switch (kind) {
                .mtp => try drafts.mtp(&at, &f, first, 12, &outs[1], &pol, false, null),
                .auto => try drafts.auto(&at, &f, first, 12, &outs[1], &pol, false, null),
            };
            try testing.expect(s.posts > 0);
        }
        after[side] = .{ x.e.st.pos, x.e.st.mtp_len, x.e.st.mtp_drafted };
    }
    try testing.expectEqualSlices(u32, got[0].tokens, got[1].tokens);
    try testing.expectEqualSlices(u8, logs[0].kinds[0..logs[0].n], logs[1].kinds[0..logs[1].n]);
    try testing.expectEqualSlices(usize, logs[0].rows[0..logs[0].n], logs[1].rows[0..logs[1].n]);
    inline for (.{ "rounds", "drafted", "accepted", "mtp_steps", "pending", "depths", "keeps" }) |name| try testing.expectEqual(@field(got[0], name), @field(got[1], name));
    try testing.expectEqual(after[0], after[1]);
    try testing.expect(got[0].rounds > 1 and (got[0].accepted < got[0].drafted or got[0].depths[0] > 0)); // rejections (or draft_policy's empty rounds) taken
    return got[0];
}

test "the AFD loops are qualification_qualification_86ea5170's: f3, c7:0.7 and auto draw the same rows in the same order as cuda_decode on B3's Engine; tokens, counters and the head's cache equal" {
    try sameAsDecode(.mtp, .{ .most = 3, .fixed = true });
    try sameAsDecode(.mtp, .{ .most = 7, .fixed = true, .confidence = 0.7 });
    try sameAsDecode(.auto, drafts.DepthPolicy.auto(true));
}

test "draft_policy: the first draft's floor in the AFD loops is cuda_decode's: empty rounds verify the pending token alone, the same rows, tokens and counters" {
    // every third head draw at 0.5 under the floor 0.6: those rounds draft nothing; the others chain under 0.7
    const a = try sameAsDecodeWith(.auto, .{ .most = 3, .fixed = true, .confidence = 0.7, .first = 0.6 }, 3);
    try testing.expect(a.depths[0] > 0 and a.drafted > 0);
    const m = try sameAsDecodeWith(.mtp, .{ .most = 7, .fixed = true, .first = 0.6 }, 2);
    try testing.expect(m.depths[0] > 0 and m.drafted > 0);
    _ = try sameAsDecodeWith(.auto, .{ .most = 3, .fixed = true, .confidence = 0.35, .first = 0.95 }, 0); // every round empty
}
