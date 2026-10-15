//! G-trace-A (qualification_9051d0c1, qualification_955b7191's fixture) and G-trace-A full (qualification_0fa2bee9; qualification_25364d42 PLAN §5.1): python 2.15's attention node at world 1
//! (src@0849f3f, experts=False, w.comm afd.Solo, w.afd a recorder) event for event against the Zig forward with
//! qualification_9051d0c1's provider. Python 2.15's events are read as 0.6's (`as06`: K1, K2, recorded, not ported); every other event
//! must be equal: op, kernel or call, grid, every argument's buffer, offset, dtype, shape and stride, the constexprs.
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const layer = @import("cuda_layer.zig");
const cc = @import("cuda_compute.zig");
const prompt = @import("cuda_prompt.zig");
const Tmp = @import("cuda_tmp.zig").Tmp;
const afd = @import("cuda_afd_attn.zig");
const qmm = @import("cuda_qmm.zig");
const lt = @import("cuda_layer_test.zig");

const testing = std.testing;
const J = std.json.Value;
/// qualification_955b7191's recording of python 2.15's attention node (layers 0-3, R 1-8 at 0 and 37; qualification_qualification_44ba554e-int c35b57b)
const fixture_attn = @embedFile("fixtures_cuda_afd_attn.json");
/// qualification_0fa2bee9's (tools/zig/glm_afd_fixtures.py attn-full): layers 0-4, long contexts, R 1 and 4 at 0, 37, 2,560; a chunk
const fixture_full = @embedFile("fixtures_cuda_afd_attn_full.json");

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn get(o: J, key: []const u8) J {
    return o.object.get(key) orelse std.debug.panic("fixture: no {s}", .{key});
}

fn int(v: J) i64 {
    return v.integer;
}

/// qualification_955b7191's two differences of python 2.15's attention node from 0.6 at world 1 (evidence/qualification_955b7191/logs/report215-attn.txt),
/// and qualification_0fa2bee9's two in the DSA layers (`kv8`: K3, `ring`: K4; evidence/qualification_0fa2bee9/logs/sig215-full.txt).
const Diffs = struct { seq: usize = 0, solo: usize = 0, kv8: usize = 0, ring: usize = 0 };

/// K3: python 2.15's latent and index cache kernels carry TF_GLM_KV=fp8's parameters (MiaAI-Lab 0038): a scale
/// pointer, the cache itself for a bf16 cache (kv8.parts: "a bf16 cache passes itself as the unused scale argument"),
/// RS its row width and FP8 0, so the bf16 path is 0.6's arithmetic. Each launch's scale argument (equal to its cache
/// argument), RS and FP8 are dropped.
const KvScale = struct { kernel: []const u8, scale: []const u8, cache: []const u8, width: i64 };
const kv8_kernels = [_]KvScale{
    .{ .kernel = "_lat_write", .scale = "LS", .cache = "LC", .width = 512 },
    .{ .kernel = "_dense_chunks", .scale = "LS", .cache = "LC", .width = 512 },
    .{ .kernel = "_sparse_chunks", .scale = "LS", .cache = "LC", .width = 512 },
    .{ .kernel = "_pool_keys", .scale = "PKS", .cache = "PK", .width = 128 },
    .{ .kernel = "_scores", .scale = "PKS", .cache = "PK", .width = 128 },
};
/// K4: the indexer's key and gate rings (2.15: token t at t % RING): with no ring set, RING is the State's capacity and
/// every slot a window writes or reads is below it, so (P + r) % RING is P + r. The RING argument is dropped.
const ring_kernels = [_][]const u8{ "_index_write", "_pool_keys" };

fn argAt(args: []const J, name: []const u8) ?usize {
    for (args, 0..) |a, i| if (eql(get(a, "name").string, name)) return i;
    return null;
}

/// K3 and K4 on one Triton event, in place (`capacity`: the State's slots).
fn dropSwitches(e: *J, capacity: i64, d: *Diffs) !void {
    const name = get(e.*, "fn").string;
    const args = &e.object.getPtr("args").?.array;
    for (kv8_kernels) |k| if (eql(name, k.kernel)) {
        const s = argAt(args.items, k.scale) orelse break; // sparse.py's own _sparse_chunks has none
        const c = argAt(args.items, k.cache) orelse return error.NoCache;
        if (!eql(get(args.items[s], "buf").string, get(args.items[c], "buf").string) or int(get(args.items[s], "off")) != int(get(args.items[c], "off"))) return error.ScaleNotCache;
        _ = args.orderedRemove(s);
        const consts = e.object.getPtr("consts").?;
        const fp8 = consts.object.getPtr("FP8") orelse return error.NoFp8;
        const rs = consts.object.getPtr("RS") orelse return error.NoRs;
        if (int(get(fp8.*, "int")) != 0) return error.Fp8;
        if (int(get(rs.*, "int")) != k.width) return error.RowWidth;
        fp8.* = .null;
        rs.* = .null;
        d.kv8 += 1;
    };
    for (ring_kernels) |k| if (eql(name, k)) {
        const r = argAt(args.items, "RING") orelse return error.NoRing;
        if (int(get(args.items[r], "int")) != capacity) return error.RingNotCapacity;
        _ = args.orderedRemove(r);
        d.ring += 1;
    };
}

/// afd.Solo's all_gather as python logs it: [<p>part[:n] flat, <p>gath[:n]] (recv[:n].copy_(send), the same fp32).
fn isSolo(e: J, p: []const u8) bool {
    const a = get(e, "args").array.items;
    if (a.len != 2) return false;
    const s = get(a[0], "t");
    const r = get(a[1], "t");
    const ss = get(s, "shape").array.items;
    const rs = get(r, "shape").array.items;
    const sb = get(s, "buf").string;
    const rb = get(r, "buf").string;
    const names = std.mem.startsWith(u8, sb, p) and eql(sb[p.len..], "part") and std.mem.startsWith(u8, rb, p) and eql(rb[p.len..], "gath");
    return names and int(get(s, "off")) == 0 and int(get(r, "off")) == 0 and ss.len == 1 and rs.len == 1 and int(ss[0]) == int(rs[0]);
}

/// Python 2.15's events read as 0.6's, in place (K1-K4 recorded, not ported; NOTES): K1, `_bmm`'s constexpr SEQ=0
/// (TF_GLM_SEQ_ROWS off, 0.6's kernel) is dropped; K2, Solo's all_gather is dropped and the `_hc_post` right after it
/// reads <p>part, whose bytes Solo copied into <p>gath (`p`: the Buffers' prefix, b. or pb.); K3 and K4 (`dropSwitches`,
/// `capacity` the State's slots). Any other shape of any is an error; every other event stays.
fn as06(gpa: std.mem.Allocator, events: []J, p: []const u8, capacity: i64, d: *Diffs) ![]J {
    var out: std.ArrayList(J) = .empty;
    errdefer out.deinit(gpa);
    var solo = false;
    const prompt_bufs = eql(p, "pb.");
    const part_name: []const u8 = if (prompt_bufs) "pb.part" else "b.part";
    const gath_name: []const u8 = if (prompt_bufs) "pb.gath" else "b.gath";
    for (events) |*e| {
        const op = get(e.*, "op").string;
        const name = get(e.*, "fn").string;
        if (eql(op, "comm") and eql(name, "all_gather")) {
            if (solo or !isSolo(e.*, p)) return error.NotSolo;
            solo = true;
            d.solo += 1;
            continue;
        }
        if (eql(op, "triton") and eql(name, "_bmm")) {
            const seq = get(e.*, "consts").object.getPtr("SEQ") orelse return error.NoSeq;
            if (int(get(seq.*, "int")) != 0) return error.SeqRows;
            seq.* = .null;
            d.seq += 1;
        }
        if (eql(op, "triton")) try dropSwitches(e, capacity, d);
        if (solo) {
            if (!eql(op, "triton") or !eql(name, "_hc_post")) return error.SoloNotBeforeHcPost;
            for (get(e.*, "args").array.items) |*x| if (eql(get(x.*, "name").string, "G")) {
                const buf = x.object.getPtr("buf") orelse return error.NotSoloG;
                if (!eql(buf.string, gath_name) or int(get(x.*, "off")) != 0) return error.NotSoloG;
                buf.* = .{ .string = part_name };
            };
            solo = false;
        }
        try out.append(gpa, e.*);
    }
    if (solo) return error.SoloAtEnd;
    return out.toOwnedSlice(gpa);
}

/// Buffers of python 2.15's attention node that ours lack: its EXL3 grouping's order (exl3_experts 2.15; the routed
/// experts never run on this node, no event reads it) and the prompt Buffers' att_starts (TF_GLM_PROMPT_DSA's dense
/// blocks' starts, MiaAI-Lab 0009; the switch is off: no event reads it). Ours beyond python's: the routed experts'
/// views (`experts=False`) and a CPU trace's temporaries (named when made).
fn pythonOnly(path: []const u8) bool {
    return std.mem.endsWith(u8, path, ".exl3.order") or eql(path, "pb.att_starts");
}

fn oursOnly(path: []const u8) bool {
    return std.mem.indexOf(u8, path, ".moe.experts.") != null or std.mem.startsWith(u8, path, "tmp.");
}

/// Every buffer of python's is ours with its dtype, shape and bytes, but `pythonOnly`'s (`skip` of them); ours beyond
/// are `oursOnly`. Every difference is printed before the first error.
fn expectBuffers(space: *const trace.Space, want: J, skip: usize) !void {
    var it = want.object.iterator();
    var skipped: usize = 0;
    var missing: usize = 0;
    while (it.next()) |kv| {
        errdefer std.debug.print("buffer {s}\n", .{kv.key_ptr.*});
        const e = for (space.entries.items) |e| {
            if (eql(e.path, kv.key_ptr.*)) break e;
        } else {
            if (pythonOnly(kv.key_ptr.*)) {
                skipped += 1;
            } else {
                std.debug.print("buffer {s}: python's only\n", .{kv.key_ptr.*});
                missing += 1;
            }
            continue;
        };
        try testing.expectEqualStrings(get(kv.value_ptr.*, "dtype").string, @tagName(e.buf.dtype));
        const shape = get(kv.value_ptr.*, "shape").array.items;
        try testing.expectEqual(shape.len, e.buf.dims);
        for (shape, 0..) |s, i| try testing.expectEqual(int(s), @as(i64, @intCast(e.buf.shape[i])));
        try testing.expectEqual(int(get(kv.value_ptr.*, "nbytes")), @as(i64, @intCast(e.buf.bytes())));
    }
    var extra: usize = 0;
    for (space.entries.items) |e| if (want.object.get(e.path) == null and !oursOnly(e.path)) {
        std.debug.print("buffer {s}: not python's\n", .{e.path});
        extra += 1;
    };
    if (missing > 0) return error.MissingBuffer;
    if (extra > 0) return error.ExtraBuffer;
    try testing.expectEqual(skip, skipped);
}

/// forward.compute's tail as cuda_compute.compute runs it (lane B's): the streams' mean, the final norm, the head
/// (a prompt chunk's last row only: python's b.prefill branch).
fn head(f: *fwd.Fwd, w: *const views.Weights, b: *const bufs.Buffers, R: usize, opt: layer.Opt) !void {
    const m = f.m;
    const D = m.hidden;
    try f.tri.streamMean(b.x.ptr, b.hidden.ptr, R, D, m.streams);
    try f.tri.rmsnorm(b.hidden.ptr, D, w.norm.ptr, b.fnormed.ptr, D, b.fxs.ptr, m.eps, R, D);
    const first = if (b.prefill) R - 1 else 0;
    try qmm.mm(f, b.prefill, cc.rows(b.fnormed, first, R - first), w.head, cc.rows(b.fxs, first, R - first), layer.rowsOf(b.logits, R - first), false, b.sk, opt.chip);
}

test "G-trace-A: python 2.15's attention node at world 1, event for event: layers 0-2 whole, layer 3's FFN half through the provider, the head (K1, K2 recorded)" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture_attn, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-forward-trace/1", get(fx, "schema").string);
    try testing.expectEqualStrings("0849f3f", get(fx, "python").string);
    try testing.expectEqualStrings("attn", get(fx, "role").string);
    try testing.expectEqual(@as(i64, 1), int(get(fx, "world")));
    var r = try lt.Rank.init(gpa, fx);
    defer r.deinit(gpa);
    var a = try afd.Afd.init(&r.m, r.b.rows);
    try r.space.place("b.afd_recv", &a.afd_recv);
    try expectBuffers(&r.space, get(fx, "buffers"), 1);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var d: Diffs = .{};
    var n: [3]usize = .{ 0, 0, 0 }; // whole layers, FFN halves, heads
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        const c = kv.value_ptr.*;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        t.reset();
        const R: usize = @intCast(int(get(c, "rows")));
        const block = if (c.object.get("block")) |v| v.string else "layer";
        r.st.pos = @intCast(int(get(c, "pos")));
        r.st.parity = @intCast(int(get(c, "parity")));
        const opt: layer.Opt = .{ .nch = @intCast(int(get(c, "nch"))), .host_pos = r.st.pos };
        var f = fwd.Fwd.recording(&r.m, &t);
        f.moe = a.provider();
        try a.begin(R, .eager, null); // engine.forward's afd.begin, every forward
        const k: usize, const seq: usize, const solo: usize = if (eql(block, "head")) .{ 2, 1, 0 } else if (eql(block, "ffn")) .{ 1, 2, 0 } else .{ 0, 6, 2 };
        if (k == 2) try head(&f, &r.w, &r.b, R, opt) else {
            const l = &r.w.layers[@intCast(int(get(c, "layer")))];
            if (k == 1) try layer.ffnHalf(&f, l, &r.b, R, opt) else try layer.layerForward(&f, l, &r.st, &r.b, R, opt);
        }
        try f.flush();
        n[k] += 1;
        const before = d;
        const want = try as06(gpa, get(c, "events").array.items, "b.", 2560, &d);
        defer gpa.free(want);
        try testing.expectEqual(Diffs{ .seq = before.seq + seq, .solo = before.solo + solo }, d); // qualification_955b7191's counts, case by case
        try lt.expectEventsTmp(&r.space, want, t.events.items, c.object.get("tmp"));
    }
    try testing.expectEqual([3]usize{ 24, 4, 4 }, n);
    try testing.expectEqual(Diffs{ .seq = 156, .solo = 48 }, d); // report215-attn.txt's totals (no DSA: no K3, K4)
}

/// The full fixture's attention node: the cut model at world 1, its weights under python's names, and decode.Engine's
/// long-context buffers as attn-full made them: b (8 rows), st (6,144 slots, the indexer caches), pb (the 2,048-row
/// prompt Buffers), each with its own provider's afd_recv (b.afd_recv, pb.afd_recv), the prompt chunk's temporaries.
const Full = struct {
    m: md.Model,
    w: views.Weights,
    b: bufs.Buffers,
    pb: bufs.Buffers,
    st: bufs.State,
    a: afd.Afd,
    pa: afd.Afd,
    tmp: Tmp = .{},
    p: prompt.Prompt,
    space: trace.Space,

    fn init(x: *Full, gpa: std.mem.Allocator, fx: J) !void {
        x.m = try md.Model.parse(gpa, get(fx, "config_json").string, 1, 0);
        try lt.expectConfig(&x.m, get(fx, "config"));
        try x.m.cut(@intCast(int(get(fx, "layers"))));
        const cap: usize = @intCast(int(get(fx, "capacity")));
        const rows: usize = @intCast(int(get(fx, "max_rows")));
        const prows: usize = @intCast(int(get(fx, "prefill_rows")));
        x.w = try views.rank(gpa, &x.m);
        errdefer x.w.deinit(gpa);
        x.b = try bufs.Buffers.init(&x.m, rows, cap, false);
        x.pb = try bufs.Buffers.init(&x.m, prows, cap, true);
        x.st = try bufs.State.init(gpa, &x.m, cap, rows, get(fx, "long").bool);
        errdefer x.st.deinit(gpa);
        x.a = try afd.Afd.init(&x.m, rows);
        x.pa = try afd.Afd.init(&x.m, prows);
        x.space = .{ .gpa = gpa };
        errdefer x.space.deinit();
        try views.walk(&x.w, views.Path.init("w"), &x.space, lt.placePython);
        try views.walk(&x.b, views.Path.init("b"), &x.space, trace.placeAll);
        try views.walk(&x.st, views.Path.init("st"), &x.space, trace.placeAll);
        try views.walk(&x.pb, views.Path.init("pb"), &x.space, trace.placeAll);
        try x.space.place("b.afd_recv", &x.a.afd_recv);
        try x.space.place("pb.afd_recv", &x.pa.afd_recv);
        x.tmp = .{ .space = &x.space };
        x.p = prompt.Prompt.init(&x.m, &x.tmp, prows);
    }

    fn deinit(x: *Full, gpa: std.mem.Allocator) void {
        x.st.deinit(gpa);
        x.w.deinit(gpa);
        x.space.deinit();
    }
};

/// G-trace-A full's cases by kind: decode windows of KDA and DSA layers, prompt chunks of each kind, the chunk's head;
/// `a4`: DSA prompt chunks refused (error.NotPorted), recorded, not compared: none since lane A's A4.
const FullCounts = struct { kda: usize = 0, dsa: usize = 0, kda_chunk: usize = 0, dsa_chunk: usize = 0, head: usize = 0, a4: usize = 0 };

test "G-trace-A full: python 2.15's attention node at world 1 with long contexts, every layer at R 1 and 4 from 0, 37, 2,560 and on a 2,048-row chunk, event for event (K1-K4 recorded)" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture_full, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-forward-trace/1", get(fx, "schema").string);
    try testing.expectEqualStrings("0849f3f", get(fx, "python").string);
    try testing.expectEqual(@as(i64, 1), int(get(fx, "world")));
    const x = try gpa.create(Full);
    defer gpa.destroy(x);
    try x.init(gpa, fx);
    defer x.deinit(gpa);
    try expectBuffers(&x.space, get(fx, "buffers"), 3);
    const cap = int(get(fx, "capacity"));
    const fresh = .{ x.a, x.pa }; // each case is one forward's layer: its provider as begin leaves it at that layer
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var d: Diffs = .{};
    var n: FullCounts = .{};
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        const c = kv.value_ptr.*;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        t.reset();
        x.tmp.reset(); // python names each case's temporaries afresh ...
        x.p.reset(); // ... and remakes kda's wide scratch
        const R: usize = @intCast(int(get(c, "rows")));
        const chunk = c.object.get("prefill") != null;
        const b = if (chunk) &x.pb else &x.b;
        const a = if (chunk) &x.pa else &x.a;
        a.* = if (chunk) fresh[1] else fresh[0];
        const is_head = c.object.get("block") != null;
        const li: usize = if (is_head) 0 else @intCast(int(get(c, "layer")));
        x.st.pos = @intCast(int(get(c, "pos")));
        x.st.parity = @intCast(int(get(c, "parity")));
        const opt: layer.Opt = .{ .nch = @intCast(int(get(c, "nch"))), .host_pos = x.st.pos, .prompt = if (chunk) &x.p else null };
        var f = fwd.Fwd.recording(&x.m, &t);
        f.moe = a.provider();
        try a.begin(R, if (chunk) .chunk else .eager, null); // decode.prefill's MODE_CHUNK for a chunk
        if (!is_head) a.pos = afd.moePos(&x.m, li) orelse 0;
        if (is_head) try head(&f, &x.w, b, R, opt) else layer.layerForward(&f, &x.w.layers[li], &x.st, b, R, opt) catch |err| {
            if (err != error.NotPorted or !chunk or x.m.kinds[li] != .dsa) return err;
            n.a4 += 1; // a DSA prompt chunk the block refuses
            continue;
        };
        try f.flush();
        const want = try as06(gpa, get(c, "events").array.items, if (chunk) "pb." else "b.", cap, &d);
        defer gpa.free(want);
        try lt.expectEventsTmp(&x.space, want, t.events.items, c.object.get("tmp"));
        const k = if (is_head) &n.head else switch (x.m.kinds[li]) {
            .kda => if (chunk) &n.kda_chunk else &n.kda,
            .dsa => if (chunk) &n.dsa_chunk else &n.dsa,
        };
        k.* += 1;
    }
    std.debug.print("G-trace-A full: {any}; {any}\n", .{ n, d });
    // layers 0-2 and 4 (KDA) and 3 (DSA) at R 1, 4 from 0, 37, 2,560; each layer's 2,048-row chunk from 0 (layer 3's
    // in A4's 512-row blocks, each past the first at pos_dev + r0: python's torch.add) and 37-row one from 2,048, layer
    // 3's 512-row chunk from 0, the chunk's head
    try testing.expectEqual(FullCounts{ .kda = 24, .dsa = 6, .kda_chunk = 8, .dsa_chunk = 3, .head = 1, .a4 = 0 }, n);
    try testing.expectEqual(Diffs{ .seq = 250, .solo = 65, .kv8 = 34, .ring = 18 }, d);
}
