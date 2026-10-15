//! qualification_955b7191's G-cover-120: the AFD attention node's Triton launches at world 1 against the kernel set (kernels.json, with
//! glm_aot.py's world-1 variants) by aot.zig's own rule: python 2.15's (fixtures_cuda_afd_attn.json, R = 1, 2, 4, 8;
//! qualification_0fa2bee9's fixtures_cuda_afd_attn_full.json, every layer with long contexts and a 2,048-row chunk) and the Zig forward's
//! on a CPU through qualification_9051d0c1's provider (layers 0-2, layer 3's FFN half; qualification_0fa2bee9: every layer whole, the DSA block too, at
//! R = 1, 2, 4, 8 from 0, 37 and 2,560, and prompt chunks), world 2 the control.
//! qualification_25364d42's prototype (evidence/qualification_25364d42/proto/qualification_25364d42_world1_cover.zig) made real by qualification_955b7191; extended to DSA and chunks by qualification_53329ce9
const std = @import("std");
const aot = @import("cuda").aot;
const glm = @import("glm_cuda");
const md = glm.model;
const views = glm.views;
const bufs = glm.buffers;
const trace = glm.trace;
const fwd = glm.fwd;
const layer = glm.layer;
const qmm = glm.qmm;
const cc = glm.compute;
const Afd = glm.afd_attn.Afd;

const testing = std.testing;
const spec_text = @embedFile("kernels.json");
const fixture_text = @embedFile("afd_attn_fixture");
const full_text = @embedFile("afd_attn_full_fixture");

/// A spec entry as an aot.json row, and whether glm_aot.py's world-1 rules added it (tp [1]).
pub const Entry = struct { spec: aot.Spec, world1: bool };

pub fn entries(arena: std.mem.Allocator) ![]Entry {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, spec_text, .{});
    const kernels = root.object.get("kernels").?.array.items;
    const out = try arena.alloc(Entry, kernels.len);
    for (kernels, out) |k, *e| {
        const o = k.object;
        const sig = o.get("signature").?.object;
        const attrs = o.get("attrs").?.object;
        var params: std.ArrayList(aot.SpecParam) = .empty;
        for (o.get("params").?.array.items) |pv| {
            const ty = sig.get(pv.string).?.string;
            if (std.mem.eql(u8, ty, "constexpr")) continue;
            const div = if (attrs.get(pv.string)) |a| a.array.items.len > 0 else false;
            try params.append(arena, .{ .name = pv.string, .type = ty, .div16 = div, .nospec = false });
        }
        var consts: std.json.ArrayHashMap(aot.SpecConst) = .{};
        var it = o.get("constexprs").?.object.iterator();
        while (it.next()) |kv| {
            const v = kv.value_ptr.object;
            var c: aot.SpecConst = .{};
            if (v.get("bool")) |bv| {
                c.int = @intFromBool(bv.bool);
            } else if (v.get("int")) |x| {
                c.int = x.integer;
            } else if (v.get("fp64_bits")) |bits| {
                const x: f64 = @bitCast(try std.fmt.parseInt(u64, bits.string[2..], 16));
                c.f32 = @bitCast(@as(f32, @floatCast(x)));
            }
            try consts.map.put(arena, kv.key_ptr.*, c);
        }
        const name = o.get("name").?.string;
        const tp = o.get("tp").?.array.items;
        e.* = .{ .spec = .{ .@"fn" = name, .hash = "", .name = name, .num_warps = 0, .params = params.items, .consts = consts }, .world1 = tp.len == 1 and tp[0].integer == 1 };
    }
    return out;
}

/// Launches by how many variants take them; `world1`: those a world-1 variant takes; misses by kernel and constexprs.
pub const Tally = struct {
    launches: usize = 0,
    one: usize = 0,
    none: usize = 0,
    many: usize = 0,
    world1: usize = 0,
    misses: std.StringArrayHashMapUnmanaged(usize) = .empty,

    fn add(t: *Tally, arena: std.mem.Allocator, set: []const Entry, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) !void {
        t.launches += 1;
        var hits: usize = 0;
        var w1 = false;
        for (set) |e| if (std.mem.eql(u8, e.spec.@"fn", name) and aot.accepts(e.spec, args, consts)) {
            hits += 1;
            w1 = e.world1;
        };
        switch (hits) {
            0 => {
                t.none += 1;
                var key: []const u8 = name;
                for (consts) |c| if (c.int) |v| {
                    key = try std.fmt.allocPrint(arena, "{s} {s}={d}", .{ key, c.name, v });
                };
                for (args) |a| switch (a.value) {
                    .i32 => |v| key = try std.fmt.allocPrint(arena, "{s} {s}:{d}", .{ key, a.name, v }),
                    else => {},
                };
                const g = try t.misses.getOrPut(arena, key);
                if (!g.found_existing) g.value_ptr.* = 0;
                g.value_ptr.* += 1;
            },
            1 => {
                t.one += 1;
                if (w1) t.world1 += 1;
            },
            else => t.many += 1,
        }
    }

    fn events(t: *Tally, arena: std.mem.Allocator, set: []const Entry, tr: *const trace.Trace) !void {
        for (tr.events.items) |e| if (e.op == .triton) try t.add(arena, set, e.name, e.args, e.consts);
    }

    fn print(t: *const Tally, what: []const u8) void {
        std.debug.print("{s}: {d} Triton launches; one variant {d} (a world-1 one {d}), none {d}, several {d}\n", .{ what, t.launches, t.one, t.world1, t.none, t.many });
        var it = t.misses.iterator();
        while (it.next()) |kv| std.debug.print("  no variant: {s} (x{d})\n", .{ kv.key_ptr.*, kv.value_ptr.* });
    }
};

/// The fixture's model (its config.json) at `world`, rank 0, cut to `layers`.
fn modelOf(gpa: std.mem.Allocator, text: []const u8, world: usize, layers: usize) !md.Model {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var m = try md.Model.parse(gpa, parsed.value.object.get("config_json").?.string, world, 0);
    try m.cut(layers);
    return m;
}

/// forward.compute's tail: the streams' mean, the final norm, the head (a prompt chunk's last row only).
fn head(f: *fwd.Fwd, w: *const views.Weights, b: *const bufs.Buffers, R: usize) !void {
    const m = f.m;
    const D = m.hidden;
    try f.tri.streamMean(b.x.ptr, b.hidden.ptr, R, D, m.streams);
    try f.tri.rmsnorm(b.hidden.ptr, D, w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, m.eps, R, D);
    const first = if (b.prefill) R - 1 else 0;
    try qmm.mm(f, b.prefill, cc.rows(b.fnormed, first, R - first), w.head, cc.rows(b.fxs, first, R - first), layer.rowsOf(b.logits, R - first), false, b.sk, .{});
}

/// The Zig forward's launches for R rows at `world` (rank 0), recorded: embed, layers 0-2 whole, layer 3's FFN half
/// (world 2: moe_block and its gather; world 1: qualification_9051d0c1's provider through the seam, hc_post over three planes), stream
/// mean, norm, head.
fn zigLaunches(arena: std.mem.Allocator, set: []const Entry, world: usize, R: usize, t: *Tally) !void {
    const gpa = testing.allocator;
    var m = try modelOf(gpa, fixture_text, world, 4);
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    var b = try bufs.Buffers.init(&m, 8, 2560, false);
    var st = try bufs.State.init(gpa, &m, 2560, 8, false);
    defer st.deinit(gpa);
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try views.walk(&w, views.Path.init("w"), &space, trace.placeAll);
    try views.walk(&b, views.Path.init("b"), &space, trace.placeAll);
    try views.walk(&st, views.Path.init("st"), &space, trace.placeAll);
    var tr = trace.Trace.init(gpa);
    defer tr.deinit();
    var f = fwd.Fwd.recording(&m, &tr);
    const D = m.hidden;
    try f.tri.embed(b.ids.ptr, cc.table(w.embed), b.x.ptr, R, D, m.streams);
    for (w.layers) |*l| switch (l.kind) {
        .kda => try layer.layerForward(&f, l, &st, &b, R, .{ .nch = cc.chunksFor(&st, R), .host_pos = 0 }),
        .dsa => { // the FFN half only here (zigFull runs the DSA block)
            if (world == 1) { // qualification_9051d0c1's provider through the seam (Fwd.moe, layer.ffnHalf): its own launches
                var a = try Afd.init(&m, b.rows);
                try space.place("b.afd_recv", &a.afd_recv);
                f.moe = a.provider();
                defer f.moe = null;
                try a.begin(R, .eager, null);
                try layer.ffnHalf(&f, l, &b, R, .{});
                continue;
            }
            const h = l.ffn_hc.?;
            try f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.post_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, D, m.streams);
            const g = try glm.moe.block(&f, l, &b, R, .{});
            try f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, world, R, D, m.streams);
        },
    };
    try head(&f, &w, &b, R);
    try f.flush();
    try t.events(arena, set, &tr);
}

/// qualification_0fa2bee9: the Zig attention node's launches at world 1 with long contexts (State 6,144 slots, the indexer), through the
/// provider: every layer whole (KDA, the DSA block) at R = 1, 2, 4, 8 from positions 0, 37 and 2,560, each window's head;
/// prompt chunks (the 2,048-row prompt Buffers, MODE_CHUNK) of 2,048 and 512 rows from 0 and of 37 from 2,048, each
/// layer and the head. A chunk's DSA block that refuses it (NotPorted; none since lane A's A4) ends that chunk: `refused`.
fn zigFull(arena: std.mem.Allocator, set: []const Entry, t: *Tally, refused: *usize) !void {
    const gpa = testing.allocator;
    var m = try modelOf(gpa, full_text, 1, 5);
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    var b = try bufs.Buffers.init(&m, 8, 6144, false);
    var pb = try bufs.Buffers.init(&m, 2048, 6144, true);
    var st = try bufs.State.init(gpa, &m, 6144, 8, true);
    defer st.deinit(gpa);
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try views.walk(&w, views.Path.init("w"), &space, trace.placeAll);
    try views.walk(&b, views.Path.init("b"), &space, trace.placeAll);
    try views.walk(&pb, views.Path.init("pb"), &space, trace.placeAll);
    try views.walk(&st, views.Path.init("st"), &space, trace.placeAll);
    var a = try Afd.init(&m, b.rows);
    var pa = try Afd.init(&m, pb.rows);
    try space.place("b.afd_recv", &a.afd_recv);
    try space.place("pb.afd_recv", &pa.afd_recv);
    const fresh = pa; // a refused chunk leaves its forward between exchanges: each chunk's provider as placed
    var tmp: glm.tmp.Tmp = .{ .space = &space };
    var p = glm.prompt.Prompt.init(&m, &tmp, pb.rows);
    var tr = trace.Trace.init(gpa);
    defer tr.deinit();
    var f = fwd.Fwd.recording(&m, &tr);
    for ([_]usize{ 1, 2, 4, 8 }) |R| for ([_]usize{ 0, 37, 2560 }) |pos| {
        st.pos = pos;
        st.parity = @intCast(pos % 2);
        try a.begin(R, .eager, null);
        f.moe = a.provider();
        for (w.layers) |*l| try layer.layerForward(&f, l, &st, &b, R, .{ .nch = cc.chunksFor(&st, R), .host_pos = pos });
        try head(&f, &w, &b, R);
    };
    for ([_][2]usize{ .{ 2048, 0 }, .{ 512, 0 }, .{ 37, 2048 } }) |c| {
        st.pos = c[1];
        st.parity = 0;
        tmp.reset();
        p.reset();
        pa = fresh;
        try pa.begin(c[0], .chunk, null);
        f.moe = pa.provider();
        const done = for (w.layers) |*l| {
            layer.layerForward(&f, l, &st, &pb, c[0], .{ .nch = cc.chunksFor(&st, c[0]), .host_pos = c[1], .prompt = &p }) catch |err| {
                if (err != error.NotPorted or l.kind != .dsa) return err;
                refused.* += 1;
                break false;
            };
        } else true;
        if (done) try head(&f, &w, &pb, c[0]);
    }
    try f.flush();
    try t.events(arena, set, &tr);
}

/// A buffer's dtype in a fixture (its `buffers` or the case's `tmp`) as Triton names a pointer to it.
fn ptrType(fx: std.json.ObjectMap, case: std.json.ObjectMap, buf: []const u8) ![]const u8 {
    const info = fx.get(buf) orelse (case.get("tmp") orelse return error.NoBuffer).object.get(buf) orelse return error.NoBuffer;
    const dt = info.object.get("dtype").?.string;
    const map = [_][2][]const u8{ .{ "bf16", "*bf16" }, .{ "f32", "*fp32" }, .{ "f16", "*fp16" }, .{ "i32", "*i32" }, .{ "i64", "*i64" }, .{ "u8", "*u8" }, .{ "i8", "*i8" } };
    for (map) |p| if (std.mem.eql(u8, p[0], dt)) return p[1];
    return error.UnknownDtype;
}

/// Python 2.15's parameters 0.6's kernels lack (cuda_afd_trace_test.zig's K1, K3, K4): `_bmm`'s SEQ (0); the KV
/// caches' fp8 scale pointer (the cache itself), RS and FP8 (0); the indexer rings' RING (the capacity).
const Dropped = struct { seq: usize = 0, kv8: usize = 0, ring: usize = 0 };

fn kv8Kernel(name: []const u8) bool {
    for ([_][]const u8{ "_lat_write", "_dense_chunks", "_sparse_chunks", "_pool_keys", "_scores" }) |k| if (std.mem.eql(u8, k, name)) return true;
    return false;
}

fn ringKernel(name: []const u8) bool {
    return std.mem.eql(u8, name, "_index_write") or std.mem.eql(u8, name, "_pool_keys");
}

/// python 2.15's launches in a fixture (`rows`: those of R rows only, else every case), read as 0.6's (`Dropped`).
fn pythonLaunches(arena: std.mem.Allocator, set: []const Entry, text: []const u8, rows: ?usize, t: *Tally, dropped: *Dropped) !void {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
    const fx = root.object.get("buffers").?.object;
    var it = root.object.get("cases").?.object.iterator();
    while (it.next()) |kv| {
        const case = kv.value_ptr.object;
        if (rows) |R| if (case.get("rows").?.integer != R) continue;
        for (case.get("events").?.array.items) |ev| {
            const o = ev.object;
            if (!std.mem.eql(u8, o.get("op").?.string, "triton")) continue;
            const name = o.get("fn").?.string;
            var args: [24]aot.Arg = undefined;
            var na: usize = 0;
            var kv8 = false;
            for (o.get("args").?.array.items) |a| {
                const ao = a.object;
                const n = ao.get("name").?.string;
                if (kv8Kernel(name) and (std.mem.eql(u8, n, "LS") or std.mem.eql(u8, n, "PKS"))) {
                    kv8 = true;
                    continue;
                }
                if (ringKernel(name) and std.mem.eql(u8, n, "RING")) {
                    dropped.ring += 1;
                    continue;
                }
                args[na] = if (ao.get("buf")) |bv|
                    aot.ptr(n, try ptrType(fx, case, bv.string), 0x7f0000000000 + @as(u64, @intCast(ao.get("off").?.integer)))
                else if (ao.get("int")) |x|
                    aot.int(n, @intCast(x.integer))
                else
                    aot.float(n, @bitCast(@as(u32, @intCast(ao.get("f32").?.integer))));
                na += 1;
            }
            if (kv8) dropped.kv8 += 1;
            var consts: [16]aot.Const = undefined;
            var nc: usize = 0;
            var ci = o.get("consts").?.object.iterator();
            while (ci.next()) |c| {
                if (c.value_ptr.* == .null) continue;
                const v = c.value_ptr.object;
                if (std.mem.eql(u8, c.key_ptr.*, "SEQ") and std.mem.eql(u8, name, "_bmm")) {
                    if (v.get("int").?.integer != 0) return error.SeqRows;
                    dropped.seq += 1;
                    continue;
                }
                if (kv8 and (std.mem.eql(u8, c.key_ptr.*, "RS") or std.mem.eql(u8, c.key_ptr.*, "FP8"))) {
                    if (std.mem.eql(u8, c.key_ptr.*, "FP8") and v.get("int").?.integer != 0) return error.Fp8;
                    continue;
                }
                consts[nc] = if (v.get("int")) |x| aot.ci(c.key_ptr.*, x.integer) else aot.cf(c.key_ptr.*, @bitCast(@as(u32, @intCast(v.get("f32").?.integer))));
                nc += 1;
            }
            try t.add(arena, set, name, args[0..na], consts[0..nc]);
        }
    }
}

test "G-cover-120: every world-1 launch of the AFD attention node finds exactly one variant (world 2 the control)" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const set = try entries(arena);
    var w2: Tally = .{};
    var w1: Tally = .{};
    var py: Tally = .{};
    var dropped: Dropped = .{};
    for ([_]usize{ 1, 2, 4, 8 }) |R| {
        try zigLaunches(arena, set, 2, R, &w2);
        try zigLaunches(arena, set, 1, R, &w1);
        try pythonLaunches(arena, set, fixture_text, R, &py, &dropped);
    }
    w2.print("Zig, world 2 (the control), R 1,2,4,8");
    w1.print("Zig, world 1 + AFD, R 1,2,4,8");
    py.print("python 2.15 (fixtures_cuda_afd_attn.json), R 1,2,4,8");
    std.debug.print("python 2.15's _bmm launches taken with SEQ=0 dropped: {d}\n", .{dropped.seq});
    for ([_]*const Tally{ &w2, &w1, &py }) |t| {
        try testing.expect(t.launches > 0);
        try testing.expectEqual(t.launches, t.one);
    }
    try testing.expectEqual(@as(usize, 0), w2.world1); // a TP=2 rank needs none of the world-1 variants
    try testing.expect(w1.world1 > 0 and py.world1 > 0);
    try testing.expectEqual(@as(usize, 276), w1.launches); // qualification_9051d0c1's provider: qualification_955b7191's count with afdHalf
}

test "G-cover-120 extended (qualification_0fa2bee9): the DSA block and prompt chunks at world 1, Zig's and python 2.15's, find exactly one variant each" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const set = try entries(arena);
    var zig: Tally = .{};
    var py: Tally = .{};
    var dropped: Dropped = .{};
    var refused: usize = 0;
    try zigFull(arena, set, &zig, &refused);
    try pythonLaunches(arena, set, full_text, null, &py, &dropped);
    zig.print("Zig, world 1 + AFD, long contexts: every layer at R 1,2,4,8 from 0, 37, 2,560; chunks 2,048@0, 512@0, 37@2,048");
    py.print("python 2.15 (fixtures_cuda_afd_attn_full.json), read as 0.6's");
    std.debug.print("dropped: {any}; DSA chunks refused: {d}\n", .{ dropped, refused });
    for ([_]*const Tally{ &zig, &py }) |t| {
        try testing.expect(t.launches > 0);
        try testing.expectEqual(t.launches, t.one);
    }
    try testing.expectEqual(@as(usize, 0), refused); // A4 takes the 2,048-row chunk's DSA block in 512-row blocks
}

// qualification_6057514d: the DSA block at world 1 on qualification_6607f886's long prompt (5,000 tokens: chunks 2,048 @ 0, 2,048 @ 2,048 and 904 @ 4,096,
// LZ2b's prompt B) and the other shapes lane A's G-cover takes at TP=2 (37 @ 2,048 straddles the dense limit, 1,537 @ 0
// ends on a one-row block; decode R 1 and 8 at 5,000 and 5,047), with a trace's dk.Scratch for the sparse partials
// past b.lat_s, as Kernels.dsa holds it on a GPU. The attention node has no MTP layer.
test "G-cover-120 long (qualification_6057514d): the DSA block at world 1 on the long prompt's chunks and steps, each launch one variant" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const set = try entries(arena);
    const gpa = testing.allocator;
    var m = try modelOf(gpa, full_text, 1, 5);
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    var b = try bufs.Buffers.init(&m, 8, 6144, false);
    var pb = try bufs.Buffers.init(&m, 2048, 6144, true);
    var st = try bufs.State.init(gpa, &m, 6144, 8, true);
    defer st.deinit(gpa);
    var x = glm.dsa_kernels.Scratch.init(m.hl(), pb.lat_s.qa.shape[2], pb.rows);
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try views.walk(&w, views.Path.init("w"), &space, trace.placeAll);
    try views.walk(&b, views.Path.init("b"), &space, trace.placeAll);
    try views.walk(&pb, views.Path.init("pb"), &space, trace.placeAll);
    try views.walk(&st, views.Path.init("st"), &space, trace.placeAll);
    try views.walk(&x, views.Path.init("dsa"), &space, trace.placeAll);
    var t: Tally = .{};
    const shapes = [_][3]usize{ .{ 2048, 0, 1 }, .{ 2048, 2048, 1 }, .{ 904, 4096, 1 }, .{ 37, 2048, 1 }, .{ 1537, 0, 1 }, .{ 1, 5000, 0 }, .{ 1, 5047, 0 }, .{ 8, 5000, 0 } };
    for (shapes) |c| for (w.layers) |*l| if (l.kind == .dsa) {
        var tr = trace.Trace.init(gpa);
        defer tr.deinit();
        var f = fwd.Fwd.recording(&m, &tr);
        st.pos = c[1];
        try glm.dsa.partialWith(&f, l, &st, if (c[2] == 1) &pb else &b, c[0], .{ .nch = cc.chunksFor(&st, c[0]), .host_pos = c[1] }, &x);
        try f.flush();
        try t.events(arena, set, &tr);
    };
    t.print("Zig, world 1, the DSA block: chunks 2,048@0, 2,048@2,048, 904@4,096, 37@2,048, 1,537@0; steps R 1@5,000, 1@5,047, 8@5,000");
    try testing.expect(t.launches > 0);
    try testing.expectEqual(t.launches, t.one);
}

// qualification_21e21bc0: the MTP head on the AFD attention node (TF_AFD_MTP=1; python 2.15 has no such path): cuda_afd_mtp.compute
// through the provider at the MTP slice, steps of R = 1, 2, 4, 8 rows at MTP lengths 0, 37 and 2,560 (mbuf), and
// prefill's absorb chunks (the prompt Buffers, its DSA Scratch) of 2,048 and 512 rows from 0 and 37 from 2,048.
test "G-cover-120 MTP (qualification_21e21bc0): the MTP head under AFD at world 1, its steps and prefill's absorb, each launch one variant" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const set = try entries(arena);
    const gpa = testing.allocator;
    var m = try modelOf(gpa, full_text, 1, 5);
    m.mtp_layers = 1; // the cut drops it (python's --layers); TF_AFD_MTP keeps it
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    var mb = try bufs.Buffers.init(&m, 8, 6144, false);
    var pb = try bufs.Buffers.init(&m, 2048, 6144, true);
    var st = try bufs.State.init(gpa, &m, 6144, 8, true);
    defer st.deinit(gpa);
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try views.walk(&w, views.Path.init("w"), &space, trace.placeAll);
    try views.walk(&mb, views.Path.init("mbuf"), &space, trace.placeAll);
    try views.walk(&pb, views.Path.init("pb"), &space, trace.placeAll);
    try views.walk(&st, views.Path.init("st"), &space, trace.placeAll);
    var a = try Afd.initWith(&m, mb.rows, true);
    var pa = try Afd.initWith(&m, pb.rows, true);
    try space.place("mbuf.afd_recv", &a.afd_recv);
    try space.place("pb.afd_recv", &pa.afd_recv);
    var tmp: glm.tmp.Tmp = .{ .space = &space };
    var p = glm.prompt.Prompt.init(&m, &tmp, pb.rows);
    var tr = trace.Trace.init(gpa);
    defer tr.deinit();
    var f = fwd.Fwd.recording(&m, &tr);
    for ([_]usize{ 1, 2, 4, 8 }) |R| for ([_]usize{ 0, 37, 2560 }) |at| {
        st.mtp_len = at;
        try a.beginMtp(R, .eager, null);
        f.moe = a.provider();
        _ = try glm.afd_mtp.compute(&f, &w, &st, &mb, R, .{ .nch = (at + R + cc.chunk - 1) / cc.chunk, .host_pos = at }, true);
    };
    for ([_][2]usize{ .{ 2048, 0 }, .{ 512, 0 }, .{ 37, 2048 } }) |c| {
        st.mtp_len = c[1];
        tmp.reset();
        p.reset();
        try pa.beginMtp(c[0], .chunk, null);
        f.moe = pa.provider();
        _ = try glm.afd_mtp.compute(&f, &w, &st, &pb, c[0], .{ .nch = (c[1] + c[0] + cc.chunk - 1) / cc.chunk, .host_pos = c[1], .prompt = &p }, true);
    }
    try f.flush();
    var t: Tally = .{};
    try t.events(arena, set, &tr);
    t.print("Zig, world 1 + AFD, the MTP head: steps R 1,2,4,8 at MTP lengths 0, 37, 2,560; absorb chunks 2,048@0, 512@0, 37@2,048");
    try testing.expect(t.launches > 0);
    try testing.expectEqual(t.launches, t.one);
}
