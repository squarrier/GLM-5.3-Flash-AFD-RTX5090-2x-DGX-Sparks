//! The expert role's CPU gates (qualification_cec3c04c): G-trace-E (python 2.15's ExpertNode.compute, qualification_955b7191's fixtures_cuda_afd_expert.json),
//! G-wire-E (the landing views at R = 1, 7, 2,048 against python's land and proto.zig's pins), python next_forward's
//! header checks, python ExpertBuffers' buffers, the half's plan on python-0.6's split fixture (expert groups only;
//! check-weights' paths as qualification_2b1188e0's weights.json names them), its launches against A2's cuda_moe.exl3Launches, the exit
//! rules' codes, and G-bytes on a real checkpoint's headers (TF_GLM_CHECKPOINT).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const split = @import("cuda_split.zig");
const index = @import("cuda_index.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const ex3 = @import("cuda_exl3_experts.zig");
const k3 = @import("cuda_exl3_kernels.zig");
const moe = @import("cuda_moe.zig");
const layer = @import("cuda_layer.zig");
const lt = @import("cuda_layer_test.zig");
const ae = @import("cuda_afd_expert.zig");
const proto = cuda.xchg.proto;

const testing = std.testing;
const J = std.json.Value;
const glm: proto.Layout = .{ .d = 4096, .s = 9 };
const fixture = @embedFile("fixtures_cuda_afd_expert.json");

fn get(o: J, key: []const u8) J {
    return o.object.get(key) orelse std.debug.panic("fixture: no key {s}", .{key});
}

test "G-wire-E: the landing's views at R = 1, 7 and 2,048 are proto.zig's pins and python's land views past the header slot" {
    // python Layout.views(landing, rows) over the bytes past the header slot: (normed, pick, wts) offsets, and req - 256
    const python = [_]struct { usize, [3]usize, usize }{
        .{ 1, .{ 0, 8192, 8448 }, 8484 },
        .{ 7, .{ 0, 57344, 57600 }, 57852 },
        .{ 2048, .{ 0, 16777216, 16850944 }, 16924672 },
    };
    const base: u64 = 1 << 40;
    for (python) |p| {
        const v = ae.landed(glm, base, p[0]);
        try testing.expectEqual(base + proto.HDR_BYTES + p[1][0], v.normed);
        try testing.expectEqual(base + proto.HDR_BYTES + p[1][1], v.pick);
        try testing.expectEqual(base + proto.HDR_BYTES + p[1][2], v.wts);
        try testing.expectEqual(proto.wireBytes(proto.HDR_BYTES + p[2]), v.bytes); // the take's bytes: the request padded to 16
        const o = glm.offsets(p[0]);
        try testing.expectEqual([3]u64{ base + o.normed, base + o.pick, base + o.wts }, [3]u64{ v.normed, v.pick, v.wts });
    }
    try testing.expectEqual(proto.Layout.Offsets{ .normed = 256, .pick = 8448, .wts = 8704, .end = 8740 }, glm.offsets(1)); // proto.zig's pin
    try testing.expectEqual(@as(usize, 8752), ae.landed(glm, 0, 1).bytes);
    for ([_]usize{ 1, 2, 3, 7, 8, 64, 2047, 2048 }) |R| { // every section the kernels read is 256-byte aligned (python ALIGN)
        const v = ae.landed(glm, 0, R);
        try testing.expect(v.normed % 256 == 0 and v.pick % 256 == 0 and v.wts % 256 == 0);
    }
    // the fixture's link.landing is python's: req(2,048) - 256 bytes, ours HDR_BYTES before it
    var l = ae.Landing.init(glm, ae.prefill_rows);
    l.buf.ptr = base;
    try testing.expectEqual(@as(usize, 16924928), l.buf.bytes());
    try testing.expectEqual(base + 256, l.python(glm, ae.prefill_rows).ptr);
    try testing.expectEqual(@as(usize, 16924672), l.python(glm, ae.prefill_rows).bytes());
}

fn hdr(rows: i32, l: i32, mode: proto.Mode, layers: i32) proto.Header {
    return .{ .fseq = 7, .rows = rows, .layer = l, .mode = mode, .layers = layers };
}

test "python next_forward's checks on a forward's first header: layer 0 or a slice's first, 42 layers, rows 1..2,048, the mode" {
    const f = try ae.first(hdr(5, 0, .eager, 42), 0, 42, 2048);
    try testing.expectEqual(ae.Forward{ .fseq = 7, .rows = 5, .mode = .eager, .first = 0, .count = 42 }, f);
    try testing.expectEqual(@as(usize, 42), (try ae.first(hdr(2048, 0, .chunk, 42), 0, 42, 2048)).count);
    const s = try ae.first(hdr(64, 10, .slice, 42), 12, 42, 2048); // a layer slice: positions 10-21
    try testing.expectEqual([2]usize{ 10, 12 }, [2]usize{ s.first, s.count });
    try testing.expectEqual(@as(usize, 1), (try ae.first(hdr(1, 41, .slice, 42), 1, 42, 2048)).count);
    try testing.expectError(error.WrongLayer, ae.first(hdr(1, 1, .eager, 42), 0, 42, 2048));
    try testing.expectError(error.WrongLayerCount, ae.first(hdr(1, 0, .eager, 41), 0, 42, 2048));
    try testing.expectError(error.WrongRows, ae.first(hdr(0, 0, .eager, 42), 0, 42, 2048));
    try testing.expectError(error.WrongRows, ae.first(hdr(2049, 0, .chunk, 42), 0, 42, 2048));
    try testing.expectError(error.WrongRows, ae.first(hdr(-1, 0, .eager, 42), 0, 42, 2048));
    try testing.expectError(error.WrongSlice, ae.first(hdr(8, 40, .slice, 42), 3, 42, 2048));
    try testing.expectError(error.WrongSlice, ae.first(hdr(8, 4, .slice, 42), 0, 42, 2048));
    try testing.expectError(error.WrongSlice, ae.first(hdr(8, -1, .slice, 42), 2, 42, 2048));
    for ([_]proto.Mode{ .pair, .lanes, @enumFromInt(2), @enumFromInt(0) }) |m| try testing.expectError(error.WrongMode, ae.first(hdr(1, 0, m, 42), 0, 42, 2048));
}

/// The fixture's node as ours: python's buffer paths at stand-in addresses (trace.Space), the landing python's
/// `link.landing` past our header slot, each fixture layer's experts as weights.py's views of a TP2 half.
pub const Node = struct {
    space: trace.Space,
    b: ae.Bufs,
    landing: ae.Landing,
    layers: [2]u32,
    w: [2]views.Exl3Experts,

    pub fn init(gpa: std.mem.Allocator, fx: J) !Node {
        const c = get(fx, "config");
        const D: usize = @intCast(get(c, "hidden").integer);
        const slots: usize = @intCast(get(c, "top_k").integer + 1);
        const E: usize = @intCast(get(c, "experts").integer);
        const width: usize = @intCast(@divExact(get(c, "moe_width").integer, 2));
        const top: usize = @intCast(get(fx, "prefill_rows").integer);
        var n: Node = .{ .space = .{ .gpa = gpa }, .b = try ae.Bufs.init(D, slots, width, E, @intCast(get(fx, "max_rows").integer), top), .landing = ae.Landing.init(.{ .d = D, .s = slots }, top), .layers = undefined, .w = undefined };
        errdefer n.space.deinit();
        try views.walk(&n.b, views.Path.init("b"), &n.space, trace.placeAll);
        var py = n.landing.python(.{ .d = D, .s = slots }, top);
        try n.space.place("link.landing", &py);
        n.landing.buf.ptr = py.ptr - proto.HDR_BYTES;
        for (get(fx, "layers").array.items, 0..) |li, k| {
            n.layers[k] = @intCast(li.integer);
            n.w[k] = views.Exl3Experts.of(E, D, width, @intCast(get(c, "bits").integer));
            var p = views.Path.init("w");
            p.push("layers");
            p.pushIndex(n.layers[k]);
            p.push("moe");
            p.push("experts");
            try views.walk(&n.w[k], p, &n.space, trace.placeAll);
        }
        return n;
    }

    pub fn experts(n: *const Node, l: u32) *const views.Exl3Experts {
        return if (n.layers[0] == l) &n.w[0] else &n.w[1];
    }
};

/// python 2.15's buffers that serve only its prompt switches (exl3_experts.order's list, rot_rows' shared suh).
fn switchOnly(p: []const u8) bool {
    return std.mem.endsWith(u8, p, ".order") or std.mem.endsWith(u8, p, ".shared_suh");
}

test "python ExpertBuffers and the half's weights: every buffer the fixture names is ours (dtype, shape, bytes), but 2.15's switch buffers" {
    const parsed = try std.json.parseFromSlice(J, testing.allocator, fixture, .{});
    defer parsed.deinit();
    var n = try Node.init(testing.allocator, parsed.value);
    defer n.space.deinit();
    var it = get(parsed.value, "buffers").object.iterator();
    var theirs: usize = 0;
    var skipped: usize = 0;
    while (it.next()) |kv| {
        errdefer std.debug.print("buffer {s}\n", .{kv.key_ptr.*});
        if (switchOnly(kv.key_ptr.*)) {
            skipped += 1;
            continue;
        }
        const e = for (n.space.entries.items) |e| {
            if (std.mem.eql(u8, e.path, kv.key_ptr.*)) break e;
        } else return error.MissingBuffer;
        try testing.expectEqualStrings(get(kv.value_ptr.*, "dtype").string, @tagName(e.buf.dtype));
        const shape = get(kv.value_ptr.*, "shape").array.items;
        try testing.expectEqual(shape.len, e.buf.dims);
        for (shape, 0..) |s, i| try testing.expectEqual(s.integer, @as(i64, @intCast(e.buf.shape[i])));
        try testing.expectEqual(get(kv.value_ptr.*, "nbytes").integer, @as(i64, @intCast(e.buf.bytes())));
        theirs += 1;
    }
    try testing.expectEqual(@as(usize, 4), skipped); // b.decode.order, b.pref.order, both layers' shared_suh
    try testing.expectEqual(theirs, n.space.entries.items.len); // and we have no other
    // the device block: python's buffers and our landing, 1.23 GiB a node
    var b = n.b;
    var l = n.landing;
    const total = try ae.place(&b, &l, 1 << 32);
    try testing.expectEqual(@as(u64, 1 << 32), b.normed.ptr);
    try testing.expect(l.buf.ptr + l.buf.bytes() <= (1 << 32) + total);
    try testing.expectEqual(@as(usize, 1320342272), total);
    try testing.expectEqual(&b.decode, b.scratch(8));
    try testing.expectEqual(&b.pref, b.scratch(9));
    try testing.expectError(error.Rows, ae.Bufs.init(4096, 9, 1024, 288, 0, 2048));
    // the shared slot python zeroes: slot 8 of each row, one 2-D memset
    const z = ae.sharedSlot(&b.decode, 8, 3);
    try testing.expectEqual(ae.Slot2D{ .dst = b.decode.y.ptr + 8 * 4096 * 4, .pitch = 9 * 4096 * 4, .width = 4096, .height = 3 }, z);
    try testing.expectEqual(ex3.glm_gateup, b.pref.gu);
    try testing.expectEqual(ex3.glm_down, b.pref.dn);
}

/// One argument of python's against ours, as cuda_layer_test's expectVal reads them (no temporaries here).
fn sameVal(space: *const trace.Space, want: J, got: trace.Val) bool {
    const o = want.object;
    if (o.get("t")) |t| {
        const x = switch (got) {
            .t => |x| x,
            else => return false,
        };
        const n = space.name(x.ptr) orelse return false;
        if (!std.mem.eql(u8, n.path, get(t, "buf").string) or n.off != get(t, "off").integer) return false;
        if (!std.mem.eql(u8, get(t, "dtype").string, @tagName(x.dtype))) return false;
        const shape = get(t, "shape").array.items;
        if (shape.len != x.dims) return false;
        for (shape, get(t, "stride").array.items, 0..) |s, d, i| if (s.integer != x.shape[i] or d.integer != x.stride[i]) return false;
        return true;
    }
    if (o.get("int")) |v| return got == .int and got.int == v.integer;
    if (o.get("f32")) |v| return got == .f32 and got.f32 == v.integer;
    if (o.get("bool")) |v| return got == .bool and got.bool == v.bool;
    if (o.get("none") != null) return got == .none;
    return false;
}

/// A prompt window's events (R = 2,048: python 2.15's switches on) against ours (the same calls with them off): every
/// difference, each named, must be the known one. Returns how many arguments differed.
fn expectPrompt(space: *const trace.Space, want: []const J, got: []const trace.Event) !usize {
    try testing.expectEqual(got.len + 1, want.len);
    try testing.expectEqualStrings("exl3_experts.order", get(want[1], "fn").string); // 2.15's extra call: the items' order
    try lt.expectEvents(space, want[0..1], got[0..1]); // group
    try testing.expectEqualStrings("exl3_experts.rot_rows", get(want[2], "fn").string); // 2.15: one rotation of the
    try testing.expectEqualStrings("exl3_experts.rot_in", got[1].name); // rows by the shared suh (ours: by each slot's)
    const rr = get(want[2], "args").array.items; // rot_rows(x, x_stride, shared_suh, xg, R, D)
    for ([_][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 3, 5 }, .{ 4, 7 }, .{ 5, 8 } }) |m| try testing.expect(sameVal(space, rr[m[0]], got[1].vals[m[1]]));
    var differ: usize = 0;
    for ([_][2]usize{ .{ 3, 2 }, .{ 5, 4 } }, [_]bool{ true, false }) |m, gateup| {
        const w = get(want[m[0]], "args").array.items;
        const g = got[m[1]].vals;
        try testing.expectEqualStrings(get(want[m[0]], "fn").string, got[m[1]].name);
        try testing.expectEqual(w.len, g.len);
        for (w, g, 0..) |a, b, i| {
            if (sameVal(space, a, b)) continue;
            differ += 1;
            switch (i) {
                1 => try testing.expect(gateup and sameVal(space, a, g[0])), // python reads xg (rotated once) for up too
                22 => try testing.expect(a.object.get("bool").?.bool), // nfirst
                23 => try testing.expectEqualStrings("b.pref.order", get(get(a, "t"), "buf").string),
                24 => try testing.expect(get(a, "int").integer > 0), // max_items
                25 => try testing.expect(a.object.get("bool").?.bool), // prefetch
                26 => try testing.expect(gateup and a.object.get("bool").?.bool), // xrow: gate/up only
                else => {
                    std.debug.print("grouped argument {d} differs beyond 2.15's switches\n", .{i});
                    return error.UnexpectedDifference;
                },
            }
        }
    }
    try lt.expectEvents(space, want[4..5], got[3..4]); // gateup_epilogue
    try lt.expectEvents(space, want[6..8], got[5..7]); // down_epilogue, _combine
    return differ;
}

test "G-trace-E: python 2.15's ExpertNode.compute, halves 0 and 1, layers 3 and 44, R = 1, 2, 4, 8 and 2,048 (its switches off) equal; on, but for them" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-forward-trace/1", get(fx, "schema").string);
    try testing.expectEqualStrings("expert", get(fx, "role").string);
    var n = try Node.init(gpa, fx);
    defer n.space.deinit();
    const limit: f32 = @bitCast(@as(u32, @intCast(get(get(fx, "config"), "limit").integer)));
    var exact: usize = 0;
    var prompt: usize = 0;
    var off: usize = 0; // 2,048-row windows python recorded with its prompt switches off (qualification_955b7191's `-off` cases, c239f5b)
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        const c = kv.value_ptr.*;
        const R: usize = @intCast(get(c, "rows").integer);
        var t = trace.Trace.init(gpa);
        defer t.deinit();
        try ae.record(&t, n.experts(@intCast(get(c, "layer").integer)), &n.b, ae.landed(glm, n.landing.buf.ptr, R), R, limit);
        const want = get(c, "events").array.items;
        const switched = for (want) |w| {
            if (std.mem.eql(u8, get(w, "fn").string, "exl3_experts.order")) break true;
        } else false;
        if (c.object.get("switches")) |s| {
            try testing.expectEqualStrings("off", s.string);
            try testing.expect(!switched and R >= 64);
            off += 1;
        }
        if (!switched) {
            try lt.expectEvents(&n.space, want, t.events.items);
            try testing.expectEqual(want.len, t.events.items.len);
            exact += 1;
        } else {
            try testing.expect(R >= 64 and get(c, "prompt").bool);
            try testing.expectEqual(@as(usize, 10), try expectPrompt(&n.space, want, t.events.items));
            prompt += 1;
        }
    }
    try testing.expect(off == 0 or off == 4); // every (half, layer) at 2,048 rows once the fixture has them
    try testing.expectEqual(16 + off, exact);
    try testing.expectEqual(@as(usize, 4), prompt);
}

test "G-trace-E with X1 (qualification_ae31a4aa, TF_GLM_EXL3_X1=1): python 2.15's prompt windows, its switches on, equal call for call and argument for argument; its decode windows and switch-off cases as without X1" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    var n = try Node.init(gpa, fx);
    defer n.space.deinit();
    const c = get(fx, "config");
    const D: usize = @intCast(get(c, "hidden").integer);
    // X1's buffers under python 2.15's names: each scratch's order list, and each layer's shared suh (its clone, [D])
    var xb = ae.X1Bufs.init(&n.b, @intCast(get(c, "experts").integer), 2);
    try n.space.place("b.decode.order", &xb.decode_order);
    try n.space.place("b.pref.order", &xb.pref_order);
    var suh: [2]views.Buf = undefined;
    for (&suh, n.layers) |*s, l| {
        s.* = views.Buf.of(.f16, &.{D});
        var name: [64]u8 = undefined;
        try n.space.place(try std.fmt.bufPrint(&name, "w.layers.{d}.moe.experts.shared_suh", .{l}), s);
    }
    // now every buffer the fixture names is ours, the switch buffers too (dtype, shape, bytes)
    var bi = get(fx, "buffers").object.iterator();
    var switch_bufs: usize = 0;
    while (bi.next()) |kv| {
        if (!switchOnly(kv.key_ptr.*)) continue;
        const e = for (n.space.entries.items) |e| {
            if (std.mem.eql(u8, e.path, kv.key_ptr.*)) break e;
        } else return error.MissingBuffer;
        try testing.expectEqualStrings(get(kv.value_ptr.*, "dtype").string, @tagName(e.buf.dtype));
        const shape = get(kv.value_ptr.*, "shape").array.items;
        try testing.expectEqual(shape.len, e.buf.dims);
        for (shape, 0..) |s, i| try testing.expectEqual(s.integer, @as(i64, @intCast(e.buf.shape[i])));
        try testing.expectEqual(get(kv.value_ptr.*, "nbytes").integer, @as(i64, @intCast(e.buf.bytes())));
        switch_bufs += 1;
    }
    try testing.expectEqual(@as(usize, 4), switch_bufs);
    const limit: f32 = @bitCast(@as(u32, @intCast(get(c, "limit").integer)));
    const all: ex3.Prompt = .{ .nfirst = true, .pa = true, .order = true, .xrow = true }; // exl3_generic.switches at 64+
    var prompt: usize = 0;
    var same: usize = 0;
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        const cs = kv.value_ptr.*;
        const R: usize = @intCast(get(cs, "rows").integer);
        const li: u32 = @intCast(get(cs, "layer").integer);
        const k: usize = if (n.layers[0] == li) 0 else 1;
        const py_off = cs.object.get("switches") != null; // python's own run with its switches off (TF_GLM_EXL3_*=0)
        var t = trace.Trace.init(gpa);
        defer t.deinit();
        try ae.recordX1(&t, n.experts(li), &n.b, ae.landed(glm, n.landing.buf.ptr, R), R, limit, if (py_off) .{} else all, .{ .order = xb.order(&n.b, n.b.scratch(R)), .shared_suh = suh[k] });
        const want = get(cs, "events").array.items;
        try lt.expectEvents(&n.space, want, t.events.items);
        try testing.expectEqual(want.len, t.events.items.len);
        if (R >= ex3.prompt_rows and !py_off) {
            try testing.expectEqualStrings("exl3_experts.order", t.events.items[1].name);
            try testing.expectEqualStrings("exl3_experts.rot_rows", t.events.items[2].name);
            prompt += 1;
        } else same += 1;
    }
    try testing.expectEqual(@as(usize, 4), prompt); // halves 0 and 1, layers 3 and 44, at 2,048 rows
    try testing.expect(same == 16 or same == 20);
}

pub const Split = struct {
    arena: std.heap.ArenaAllocator,
    names: split.Names = .empty,
    c: cfg.Config = .{},

    /// fixtures_cuda_exl3_split.json's headers (078455ff's layers 0-3 and 45, experts 0-3) as the split reads them.
    pub fn load() !Split {
        var f: Split = .{ .arena = .init(testing.allocator) };
        errdefer f.arena.deinit();
        const a = f.arena.allocator();
        const fx = (try std.json.parseFromSliceLeaky(J, a, @embedFile("fixtures_cuda_exl3_split.json"), .{})).object;
        var it = fx.get("tensors").?.object.iterator();
        while (it.next()) |e| {
            const t = e.value_ptr.array.items;
            const dt = std.meta.stringToEnum(enum { BF16, F32, F16, I16, I32, U32 }, t[3].string).?;
            var s: split.Src = .{ .shard = @intCast(t[0].integer), .off = @intCast(t[1].integer), .len = @intCast(t[2].integer), .dtype = switch (dt) {
                .BF16 => .bf16, .F32 => .f32, .F16 => .f16, .I16 => .i16, .I32 => .i32, .U32 => .u32,
            }, .rank = @intCast(t[4].array.items.len) };
            for (t[4].array.items, 0..) |d, i| s.shape[i] = @intCast(d.integer);
            try f.names.put(a, e.key_ptr.*, s);
        }
        f.c.run = @intCast(fx.get("run").?.integer);
        f.c.experts = @intCast(fx.get("experts").?.integer);
        return f;
    }
};

test "the half on python-0.6's split fixture: layer 3's expert group only, its reads, tables and check-weights list" {
    var f = try Split.load();
    defer f.arena.deinit();
    var buf: [96]u8 = undefined;
    for ([_]u32{ 0, 1 }) |h| {
        var x = try ae.half(testing.allocator, &f.c, &f.names, h);
        defer x.deinit();
        try testing.expectEqualSlices(u32, &.{3}, x.layers); // the MTP layer's experts (45) are not served
        const e = x.experts(0);
        try testing.expectEqualSlices(u32, &.{e.tables.g}, x.groups);
        try testing.expect(x.plan.experts[45] != null); // the rank's plan has them; the half keeps only its groups
        try testing.expectEqual(@as(usize, 4 * 9), x.copies.len); // trellis, suh and svh of gate, up and down an expert
        for (x.copies) |cp| try testing.expectEqual(e.tables.g, cp.g);
        // a TP2 half of each of 4 experts: three 2 MiB trellises; suh and svh 8 KiB at 4,096 wide, 2 KiB at 1,024
        try testing.expectEqual(@as(u64, 4 * (3 * 2097152 + 3 * 8192 + 3 * 2048)), x.kept());
        try testing.expect(x.bytes() >= x.kept() + 36 * 4 and x.bytes() < x.kept() + 4096 * 16);
        var all: std.ArrayList(ae.Entry) = .empty;
        defer all.deinit(testing.allocator);
        try ae.entries(testing.allocator, &x, 0, &all);
        try testing.expectEqual(@as(usize, 6 + 6 + 3 * 4), all.items.len); // qualification_2b1188e0: 3 + 3 + 6 + 3 E
        try testing.expectEqualStrings("w.layers[3].moe.experts.gate_ptr", ae.path(all.items[0], &buf));
        try testing.expectEqualStrings("w.layers[3].moe.experts.gate_k2", ae.path(all.items[3], &buf));
        try testing.expectEqualStrings("w.layers[3].moe.experts.svh_d", ae.path(all.items[11], &buf));
        try testing.expectEqualStrings("w.layers[3].moe.experts.keep[11]", ae.path(all.items[23], &buf));
        try testing.expectEqual([3]u64{ 256, 64, 64 }, all.items[12].shape); // keep[0]: gate expert 0, qualification_2b1188e0's [256, 64, 64]
        try testing.expectEqual([3]u64{ 64, 256, 64 }, all.items[20].shape); // keep[8]: down expert 0
        try testing.expectEqual([3]u64{ 4, 1024, 0 }, all.items[10].shape); // suh_d [E, 1,024]
        for (all.items[12..]) |en| try testing.expectEqual(@as(u64, 2097152), en.bytes);
        try testing.expectEqual(e.tables.off + 24 * 4, all.items[3].off);
        // the tables at their place in the group as Plan.tables writes them; the entries read them there
        const group = try testing.allocator.alloc(u8, @intCast(x.bytes()));
        defer testing.allocator.free(group);
        @memset(group, 0);
        var tab: [36 * 4]u8 = undefined;
        const t = x.tables(0, 1 << 40, &tab);
        @memcpy(group[@intCast(t.off)..][0..t.bytes.len], t.bytes);
        try testing.expectEqual(@as(u64, (1 << 40) + e.gate.trellis.off), std.mem.readInt(u64, group[@intCast(all.items[0].off)..][0..8], .little));
        try testing.expectEqual(@as(u32, 8), std.mem.readInt(u32, group[@intCast(all.items[4].off)..][0..4], .little)); // up_k2[0]
        const l = x.bind(0, 1 << 40);
        try testing.expectEqual([3]u32{ 4096, 1024, 4 }, [3]u32{ l.dims, l.width, l.count });
        try testing.expectEqual((1 << 40) + all.items[3].off, l.gate_k2);
        // check-weights' lines: qualification_2b1188e0's entry, sha256 of the bytes (none for an address table)
        var aw: std.Io.Writer.Allocating = .init(testing.allocator);
        defer aw.deinit();
        const k2 = all.items[3];
        try ae.writeEntry(&aw.writer, all.items[0], group[@intCast(all.items[0].off)..][0..@intCast(all.items[0].bytes)]);
        try ae.writeEntry(&aw.writer, k2, group[@intCast(k2.off)..][0..@intCast(k2.bytes)]);
        try ae.writeEntry(&aw.writer, all.items[12], null);
        var sum: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(group[@intCast(k2.off)..][0..16], &sum, .{});
        var want: [512]u8 = undefined;
        try testing.expectEqualStrings(try std.fmt.bufPrint(&want,
            \\{{"path": "w.layers[3].moe.experts.gate_ptr", "shape": [4], "dtype": "int64", "sha256": null, "address_table": true, "bytes": 32}}
            \\{{"path": "w.layers[3].moe.experts.gate_k2", "shape": [4], "dtype": "int32", "sha256": "{s}", "bytes": 16}}
            \\{{"path": "w.layers[3].moe.experts.keep[0]", "shape": [256, 64, 64], "dtype": "int16", "sha256": null, "bytes": 2097152}}
            \\
        , .{&std.fmt.bytesToHex(sum, .lower)}), aw.written());
        try testing.expectError(error.EntryBytes, ae.writeEntry(&aw.writer, k2, group[0..4]));
    }
    try testing.expectError(error.BadHalf, ae.half(testing.allocator, &f.c, &f.names, 2));
}

test "TF_AFD_MTP's half (qualification_21e21bc0): the MTP layer's experts (45) one more position, last, on the same expert path; its group read whole" {
    var f = try Split.load();
    defer f.arena.deinit();
    for ([_]u32{ 0, 1 }) |h| {
        var x = try ae.halfWith(testing.allocator, &f.c, &f.names, h, true);
        defer x.deinit();
        var plain = try ae.half(testing.allocator, &f.c, &f.names, h);
        defer plain.deinit();
        try testing.expectEqualSlices(u32, &.{ 3, 45 }, x.layers);
        try testing.expectEqualSlices(u32, &.{ x.experts(0).tables.g, x.experts(1).tables.g }, x.groups);
        try testing.expectEqual(2 * plain.copies.len, x.copies.len); // the same split, a second expert group
        try testing.expectEqual(2 * plain.kept(), x.kept()); // its experts are layer 3's shapes
        const l = x.bind(1, 1 << 40);
        try testing.expectEqual([3]u32{ 4096, 1024, 4 }, [3]u32{ l.dims, l.width, l.count });
    }
    var c = f.c;
    c.mtp = 0; // a checkpoint without the MTP layer
    try testing.expectError(error.NoMtpLayer, ae.halfWith(testing.allocator, &c, &f.names, 0, true));
}

/// Layer views as cuda_moe.exl3Launches binds them (A2's ex3.Layer from the rank's Exl3Experts views).
fn bound(e: views.Exl3Experts) ex3.Layer {
    return .{ .gate_ptr = e.gate_ptr.ptr, .up_ptr = e.up_ptr.ptr, .down_ptr = e.down_ptr.ptr, .gate_k2 = e.gate_k2.ptr, .up_k2 = e.up_k2.ptr, .down_k2 = e.down_k2.ptr, .suh_g = e.suh_g.ptr, .suh_u = e.suh_u.ptr, .svh_g = e.svh_g.ptr, .svh_u = e.svh_u.ptr, .suh_d = e.suh_d.ptr, .svh_d = e.svh_d.ptr, .count = @intCast(e.count), .dims = @intCast(e.dims), .width = @intCast(e.width), .cb = @enumFromInt(e.cb), .k2_gu = e.k2_gu, .k2_d = e.k2_d };
}

test "the half's launches are A2's: cuda_moe.exl3Launches' list on the same buffers, R = 1, 2, 4, 8" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, lt.fixture_exl3, .{});
    defer parsed.deinit();
    var r = try lt.Rank.init(gpa, parsed.value);
    defer r.deinit(gpa);
    const e = r.w.layers[3].moe.?.experts.exl3;
    const s3 = r.b.exl3.?;
    // the rank's buffers as the expert's: its normed and pick rows as the landing's views, its scratch with y = b.ey
    var s = try ae.Scratch.init(r.b.rows, r.m.top_k + 1, e.dims, e.width, e.count);
    inline for (.{ "xg", "xu", "xd", "z", "ids", "count", "members_buf" }) |name| @field(s, name).ptr = @field(s3, name).ptr;
    s.y.ptr = r.b.ey.ptr;
    const sh: ex3.Shared = .{ .optin = 101376 };
    for ([_]usize{ 1, 2, 4, 8 }) |R| {
        var want: k3.Launches = .{};
        try moe.exl3Launches(&want, layer.rowsOf(r.b.normed, R), e, &r.b, R, r.m.limit, sh);
        var got: k3.Launches = .{};
        try ae.launches(&got, .{ .normed = r.b.normed.ptr, .pick = r.b.pick.ptr, .wts = r.b.wts.ptr, .bytes = 0 }, bound(e), &s, R, r.m.limit, sh);
        try testing.expectEqual(@as(usize, 6), got.len);
        try testing.expectEqual(want.len, got.len);
        for (want.slice(), got.slice()) |w, g| {
            try testing.expectEqual(w.image, g.image);
            try testing.expectEqualStrings(w.symbol, g.symbol);
            try testing.expectEqualDeep(w.cfg, g.cfg);
            try testing.expectEqual(w.max_dynamic, g.max_dynamic);
            try testing.expectEqual(w.args.count, g.args.count);
            try testing.expectEqualSlices(u16, w.args.offsets[0..w.args.count], g.args.offsets[0..g.args.count]);
            try testing.expectEqualSlices(u8, w.args.storage[0..w.args.used], g.args.storage[0..g.args.used]);
        }
    }
    try testing.expectError(error.Rows, ae.launches(undefined, undefined, bound(e), &s, 9, 1, sh));
}

test "Mia's 0128 in the half (qualification_96d80965): launchesSched is cuda_moe.exl3LaunchesSched's (tf-glm-forward's, qualification_1b2efbe0's) at each setting; off, launches'; on, only the grouped gate/up and down and the gate/up epilogue change" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, lt.fixture_exl3, .{});
    defer parsed.deinit();
    var r = try lt.Rank.init(gpa, parsed.value);
    defer r.deinit(gpa);
    const e = r.w.layers[3].moe.?.experts.exl3;
    const s3 = r.b.exl3.?;
    var s = try ae.Scratch.init(r.b.rows, r.m.top_k + 1, e.dims, e.width, e.count);
    inline for (.{ "xg", "xu", "xd", "z", "ids", "count", "members_buf" }) |name| @field(s, name).ptr = @field(s3, name).ptr;
    s.y.ptr = r.b.ey.ptr;
    const sh: ex3.Shared = .{ .optin = 101376 };
    const v: ae.Views = .{ .normed = r.b.normed.ptr, .pick = r.b.pick.ptr, .wts = r.b.wts.ptr, .bytes = 0 };
    for ([_]usize{ 1, 2, 4, 8 }) |R| for ([_]ex3.Sched{ .{}, .{ .pdl = true }, .{ .l2d = 4, .pdl = true }, .{ .l2d = 8 } }) |sd| {
        var off: k3.Launches = .{};
        try ae.launches(&off, v, bound(e), &s, R, r.m.limit, sh);
        var want: k3.Launches = .{};
        try moe.exl3LaunchesSched(&want, layer.rowsOf(r.b.normed, R), e, &r.b, R, r.m.limit, sh, sd);
        var got: k3.Launches = .{};
        try ae.launchesSched(&got, v, bound(e), &s, R, r.m.limit, sh, sd);
        try testing.expectEqual(want.len, got.len);
        try testing.expectEqual(off.len, got.len);
        for (want.slice(), got.slice(), off.slice(), 0..) |w, g, o, i| {
            try testing.expectEqual(w.image, g.image);
            try testing.expectEqualStrings(w.symbol, g.symbol);
            try testing.expectEqualDeep(w.cfg, g.cfg);
            try testing.expectEqualSlices(u8, w.args.storage[0..w.args.used], g.args.storage[0..g.args.used]);
            // group, rot_in and the down epilogue (0, 1, 5) are today's at every setting; 2-4 are 0128's when on
            const same = std.mem.eql(u8, o.symbol, g.symbol) and o.image == g.image and std.meta.eql(o.cfg, g.cfg) and std.mem.eql(u8, o.args.storage[0..o.args.used], g.args.storage[0..g.args.used]);
            try testing.expectEqual(!sd.on() or i == 0 or i == 1 or i == 5, same);
        }
        if (sd.on()) {
            try testing.expect(std.mem.startsWith(u8, @tagName(got.slice()[2].image), "exl3_l2d_grouped_cb"));
            try testing.expectEqual(k3.Image.exl3_l2d_experts, got.slice()[3].image);
            try testing.expect(std.mem.startsWith(u8, @tagName(got.slice()[4].image), "exl3_l2d_grouped_cb"));
            for (got.slice()[2..5]) |l| try testing.expectEqual(sd.pdl, l.cfg.pdl);
        }
    };
}

test "how the loop ends: exit 0 on a stop, 3 on a lost attention, a breach or a failed take, 1 on its own error" {
    try testing.expectEqual(@as(u8, 0), (ae.End{ .stopped = {} }).code());
    try testing.expectEqual(@as(u8, 3), (ae.End{ .lost = "eof" }).code());
    try testing.expectEqual(@as(u8, 3), (ae.End{ .breach = error.WrongMode }).code());
    const f: cuda.xchg.Failure = .{ .why = .gave_up, .peer = 1, .seq = 44, .gave_up = 1, .breach = 0 };
    try testing.expectEqual(@as(u8, 3), (ae.End{ .failed = f }).code());
    try testing.expectEqual(@as(u8, 1), (ae.End{ .err = error.CudaFailed }).code());
    try testing.expectEqual(@as(u8, 3), (ae.End{ .die = {} }).code()); // --fault exit
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("afd: failed gave_up peer 1 seq 44 gave_up 1 breach 0 (expert 0 after 1 forwards, 42 layers)", ae.describe(.{ .failed = f }, 0, .{ .forwards = 1, .layers = 42 }, &buf));
    try testing.expectEqualStrings("afd expert 1: protocol breach (WrongMode) after 0 forwards, 0 layers", ae.describe(.{ .breach = error.WrongMode }, 1, .{}, &buf));
    try testing.expectEqualStrings("afd expert 0: --fault exit at request 50: exiting 3 at once after 1 forwards, 49 layers", ae.describe(.die, 0, .{ .forwards = 1, .layers = 49 }, &buf));
}

fn readAll(gpa: std.mem.Allocator, p: [:0]const u8) ![]u8 {
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.NoFile;
    defer _ = std.c.close(fd);
    const buf = try gpa.alloc(u8, 1 << 20);
    errdefer gpa.free(buf);
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n <= 0) return error.NoFile;
    return gpa.realloc(buf, @intCast(n));
}

test "G-bytes: each half of 078455ff's MoE layers 3-44 from its headers is 71.22 GiB, role_bytes.py's (TF_GLM_EXL3_CHECKPOINT, else skipped)" {
    // an EXL3 checkpoint's headers (qualification_765330ce's sparse mirror serves); TF_GLM_CHECKPOINT is the Metal loader's MLX one
    const dir = std.mem.span(std.c.getenv("TF_GLM_EXL3_CHECKPOINT") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    const p = try std.fmt.allocPrintSentinel(gpa, "{s}/config.json", .{dir}, 0);
    defer gpa.free(p);
    const text = try readAll(gpa, p);
    defer gpa.free(text);
    const c = try cfg.parse(gpa, text);
    var ix = try index.open(gpa, dir);
    defer ix.deinit();
    for ([_]u32{ 0, 1 }) |h| {
        var x = try ae.half(gpa, &c, &ix.names, h);
        defer x.deinit();
        try testing.expectEqual(@as(usize, 42), x.layers.len);
        try testing.expectEqual([2]u32{ 3, 44 }, [2]u32{ x.layers[0], x.layers[41] });
        var entries: usize = 0;
        var all: std.ArrayList(ae.Entry) = .empty;
        defer all.deinit(gpa);
        for (0..x.layers.len) |pos| {
            all.clearRetainingCapacity();
            try ae.entries(gpa, &x, pos, &all);
            entries += all.items.len;
        }
        std.debug.print("G-bytes half {d}: {d} MoE layers, kept {d} B ({d:.2} GiB), groups {d} B ({d:.2} GiB), {d} reads, {d} check-weights entries\n", .{ h, x.layers.len, x.kept(), @as(f64, @floatFromInt(x.kept())) / (1 << 30), x.bytes(), @as(f64, @floatFromInt(x.bytes())) / (1 << 30), x.copies.len, entries });
        try testing.expectEqual(@as(u64, 76473040896), x.kept()); // role_bytes.py on qualification_765330ce's map: 71.22 GiB a half
        try testing.expectEqual(@as(usize, 42 * (12 + 3 * 288)), entries); // qualification_2b1188e0's 876 a layer
    }
}
