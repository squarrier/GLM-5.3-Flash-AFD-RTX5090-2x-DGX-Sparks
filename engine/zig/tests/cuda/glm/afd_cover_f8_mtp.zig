//! mtp_fp8's G-cover-120 MTP8: the AFD attention node's Triton launches with the MTP layer's eight projections on FP8
//! (TF_GLM_MTP_FP8=1) against mtp_fp8_w1b's set120 (qualification_955b7191's kernels.json, qualification_7877904d's kernels_f8_w1.json and
//! kernels_f8_mtp_w1.json, the 72 `_fmm` variants evidence/mtp_fp8/f8/f8aot_mtp.py built for sm_120a at the eight shapes), by
//! aot.zig's own rule: exactly one variant a launch. qualification_724c934b's spec is the negative control: it takes every launch
//! with the switch unset and none of the MTP shapes' `_fmm`. Always: python's `_matmul_f8` fixture at the eight shapes
//! against Tri.matmulF8, the shapes the views' of 078455ff's MTP layer. With the mirrors (TF_GLM_EXL3_HEADERS and
//! TF_qualification_7877904d_Q4W1M, else skipped): the attention role in the lane's form (q4, FP8 head, TF_AFD_MTP=1) through
//! afd_load.dense with the switch unset and 1, prefill's MTP absorb and the drafted decode loops; the built aot.json too
//! when TF_qualification_7877904d_SETS names evidence/SETS. `_fmm` is Mia's (MiaAI-Lab, patch 0002); the lever is BadAd84's (PR #36).
const std = @import("std");
const aot = @import("cuda").aot;
const glm = @import("glm_cuda");
const c8 = @import("afd_cover_f8"); // qualification_7877904d's helpers, its own module (build/glm_afd.zig): its tests not run twice

const testing = std.testing;
const base_text = @embedFile("kernels.json");
const f8_text = @embedFile("kernels_f8_w1.json");
const mtp_text = @embedFile("kernels_f8_mtp_w1.json");
const fixture_text = @embedFile("fixtures_triton_f8_mtp_w1.json");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// The MTP layer's eight projections as the views hold them (078455ff at world 1): python's names, the views, fp32 out.
fn mtpViews(x: *glm.views.Mtp) [8]struct { []const u8, *glm.views.Linear, bool } {
    const d = &x.layer.dsa.?;
    const s = &x.layer.moe.?.shared.?;
    return .{ .{ "eh", &x.eh, false }, .{ "dsa.proj", &d.proj, false }, .{ "dsa.q_b", &d.q_b, false }, .{ "dsa.o", &d.o, true }, .{ "index.kw", &d.index.?.kw, false }, .{ "index.qb", &d.index.?.qb, false }, .{ "shared.gu", &s.gu, false }, .{ "shared.down", &s.down, true } };
}

test "G-cover-120 MTP8 (mtp_fp8): the MTP layer's eight projections' _fmm as python launches them at 1-8, 16 and the kits' chunk rows: one variant each in mtp_fp8_w1b's spec, none in qualification_724c934b's" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const old = try std.mem.concat(arena, aot.Spec, &.{ try c8.specEntries(arena, base_text), try c8.specEntries(arena, f8_text) });
    const mtp = try c8.specEntries(arena, mtp_text);
    const set = try std.mem.concat(arena, aot.Spec, &.{ old, mtp });
    const used = try arena.alloc(usize, set.len);
    @memset(used, 0);
    const m = try c8.fixtureModel(testing.allocator);
    try testing.expectEqual(@as(usize, 1), m.mtp_layers);
    var w = try glm.views.rank(testing.allocator, &m);
    defer w.deinit(testing.allocator);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, fixture_text, .{});
    const shapes = root.object.get("shapes").?.object;
    try testing.expectEqual(@as(usize, 8), shapes.count());
    for (mtpViews(&w.mtp.?)) |v| { // python's shapes (config.json) are the views' (Model.parse): N, K, fp32 out
        const s = shapes.get(v[0]).?.array.items;
        try testing.expectEqual(@as(i64, @intCast(v[1].n())), s[0].integer);
        try testing.expectEqual(@as(i64, @intCast(v[1].k())), s[1].integer);
        try testing.expectEqual(v[2], s[2].bool);
    }
    var tr = glm.trace.Trace.init(testing.allocator);
    defer tr.deinit();
    var f = glm.fwd.Fwd.recording(&m, &tr);
    var it = root.object.get("cases").?.object.iterator();
    var cases: usize = 0;
    var launches: usize = 0;
    var reduces: usize = 0;
    while (it.next()) |c| {
        const rest = c.key_ptr.*["fmm/mtp-w1/".len..];
        const slash = std.mem.lastIndexOfScalar(u8, rest, '/').?;
        const s = shapes.get(rest[0..slash]).?.array.items;
        const rows = try std.fmt.parseInt(usize, rest[slash + 2 ..], 10);
        const n: usize = @intCast(s[0].integer);
        const k: usize = @intCast(s[1].integer);
        const at = tr.events.items.len;
        try f.tri.matmulF8(c8.addr(1), k, c8.addr(2), c8.addr(3), c8.addr(4), s[2].bool, c8.addr(5), rows, n, k);
        try f.flush();
        const got = tr.events.items[at..];
        const want = c.value_ptr.array.items;
        errdefer std.debug.print("case {s}\n", .{c.key_ptr.*});
        try testing.expectEqual(want.len, got.len);
        try testing.expectEqual(@as(usize, if (s[3].integer > 1) 2 else 1), got.len); // _fmm, then _reduce past one K slice
        for (want, got) |wv, g| {
            const wo = wv.object;
            try testing.expectEqualStrings(wo.get("fn").?.string, g.name);
            for (wo.get("grid").?.array.items, g.grid) |a, b| try testing.expectEqual(a.integer, @as(i64, b));
            const wa = wo.get("args").?.array.items;
            try testing.expectEqual(wa.len, g.args.len);
            for (wa, g.args) |a, b| {
                try testing.expectEqualStrings(a.object.get("name").?.string, b.name);
                switch (b.value) {
                    .ptr => |p| {
                        try testing.expectEqualStrings(a.object.get("type").?.string, p.ty);
                        try testing.expect(a.object.get("align16").?.bool == (p.addr % 16 == 0));
                    },
                    .i32 => |x| try testing.expectEqual(a.object.get("int").?.integer, @as(i64, x)),
                    else => return error.UnexpectedArg,
                }
            }
            var nc: usize = 0;
            var ci = wo.get("consts").?.object.iterator();
            while (ci.next()) |kv| if (kv.value_ptr.* != .null) {
                const gc = for (g.consts) |x| {
                    if (std.mem.eql(u8, x.name, kv.key_ptr.*)) break x;
                } else return error.MissingConst;
                try testing.expectEqual(kv.value_ptr.object.get("int").?.integer, gc.int.?);
                nc += 1;
            };
            try testing.expectEqual(nc, g.consts.len);
            try testing.expectEqual(@as(usize, 1), c8.hits(set, g.name, g.args, g.consts, used)); // mtp_fp8_w1b: one
            if (std.mem.eql(u8, g.name, "_fmm")) {
                try testing.expectEqual(@as(usize, 0), c8.hits(old, g.name, g.args, g.consts, null)); // qualification_724c934b: none
            } else { // _reduce: qualification_589d8f6f's rows already take it
                try testing.expectEqual(@as(usize, 1), c8.hits(old, g.name, g.args, g.consts, null));
                reduces += 1;
            }
            launches += 1;
        }
        cases += 1;
    }
    try testing.expectEqual(@as(usize, 176), cases); // 8 shapes x 22 row counts
    try testing.expectEqual(@as(usize, 330), launches);
    try testing.expectEqual(@as(usize, 154), reduces); // 7 shapes split K
    try testing.expectEqual(@as(usize, 72), mtp.len); // 8 shapes x (BM 16: M 1, div16, plain; BM 32, 64, 128: div16, plain)
    var captured: usize = 0;
    for (set[old.len..], used[old.len..]) |e, u| if (e.hash.len > 0) { // qualification_7877904d's rule: every captured variant used
        captured += 1;
        if (u == 0) return error.DeadVariant;
    } else try testing.expectEqual(@as(usize, 0), u); // a derived one python's rows never pick (the kits' rows do)
    try testing.expectEqual(@as(usize, 56), captured);
    std.debug.print("G-cover-120 MTP8: the MTP layer's 8 projections at world 1: {d} python launches = Zig's ({d} _reduce), one variant each in mtp_fp8_w1b's spec (487 + {d} _fmm: the {d} captured all used, the {d} derived for the kits' other rows); the _fmm none in qualification_724c934b's\n", .{ launches, reduces, mtp.len, captured, mtp.len - captured });
}

/// qualification_7877904d's kit sequences at their capacities (afd_cover_f8.zig's `seqs`): a prompt of `prompt` tokens, then drafts.
const Seq = struct { capacity: usize, prompt: usize };
const seqs = [_]Seq{ .{ .capacity = 2560, .prompt = 64 }, .{ .capacity = 2560, .prompt = 1024 }, .{ .capacity = 2560, .prompt = 1900 }, .{ .capacity = 6144, .prompt = 5000 } };

/// What a run's load made: q4 projections and files, the MTP layer's FP8, device bytes, the MTP views' forms.
const Load = struct { q4: usize = 0, files: usize = 0, mtp8: ?glm.pack.mtp8.Summary = null, bytes: u64 = 0, forms: [8]std.meta.Tag(glm.views.Linear) = undefined };

fn env(name: [*:0]const u8, value: ?[*:0]const u8) void {
    if (value) |v| _ = setenv(name, v, 1) else _ = unsetenv(name);
}

/// The attention role (TF_AFD_MTP=1) over the mirror in the lane's form through afd_load.dense (TF_GLM_DENSE=q4:DIR,
/// TF_GLM_F8=head, TF_GLM_MTP_FP8 `on` or unset): prefill with the MTP absorb, then afd_drafts.mtp at fixed depths 1-7
/// and auto, after each prompt; every Triton launch into `ts` (one a set) and the MTP shapes' `_fmm` into `mtp_fmm`.
fn run(gpa: std.mem.Allocator, arena: std.mem.Allocator, on: bool, mirror: []const u8, q4dir: []const u8, layers: ?usize, sets: []const c8.Named, ts: []c8.Tally, mtp_fmm: *[8]usize) !Load {
    const text = try c8.readFile(arena, try std.fmt.allocPrint(arena, "{s}/config.json", .{mirror}));
    var out: Load = .{};
    const toks = try arena.alloc(u32, 5000);
    for (toks, 0..) |*t, i| t.* = @intCast(i % 1000 + 1);
    const got = try arena.alloc(u32, 64);
    const dense_v = try std.fmt.allocPrintSentinel(arena, "q4:{s}", .{q4dir}, 0);
    env("TF_GLM_DENSE", dense_v.ptr);
    env("TF_GLM_F8", "head");
    env("TF_GLM_MTP_FP8", if (on) "1" else null);
    defer for ([_][*:0]const u8{ "TF_GLM_DENSE", "TF_GLM_F8", "TF_GLM_MTP_FP8" }) |v| env(v, null);
    var cap_done: usize = 0;
    for (seqs) |s| {
        if (s.capacity == cap_done) continue;
        cap_done = s.capacity;
        var m = try glm.model.Model.parse(gpa, text, 1, 0);
        const keep = m.mtp_layers;
        if (layers) |l| try m.cut(l);
        m.mtp_layers = keep; // the cut drops it (python's --layers); TF_AFD_MTP keeps it
        var w = try glm.views.rank(gpa, &m);
        defer w.deinit(gpa);
        var ix = try glm.index.open(gpa, mirror);
        defer ix.deinit();
        var c = try glm.config.parse(gpa, text);
        c.run = @intCast(m.layers);
        var ld = try glm.afd_load.loadWith(gpa, &m, &c, &ix.names, true);
        defer ld.deinit();
        const d = (try glm.afd_load.dense(gpa, &ld, &ix, &m)).?;
        out.q4 = d.q4.triplets;
        out.files = d.q4.files;
        out.mtp8 = d.mtp8;
        out.bytes = ld.deviceBytes();
        const top = try ld.stand(1 << 36);
        _ = try glm.afd_load.bind(&w, &ld.pk);
        try testing.expectEqual(@as(usize, 0), (try glm.pack.Unbound.of(&w)).count);
        for (mtpViews(&w.mtp.?), &out.forms) |v, *fm| fm.* = std.meta.activeTag(v[1].*);
        var space: glm.trace.Space = .{ .gpa = gpa };
        defer space.deinit();
        space.next = std.mem.alignForward(u64, top, 1 << 36) + (1 << 36);
        const e = try glm.engine.Engine.create(gpa, &m, &w, s.capacity, 8, 2048, s.capacity > m.denseLimit());
        defer e.destroy(gpa);
        try e.walk(&space, glm.trace.placeAll);
        try e.walkTmp(&space, glm.trace.placeAll);
        e.tmp.space = &space;
        var a = try glm.afd_attn.Afd.initWith(&m, 2048, true);
        try space.place("b.afd_recv", &a.afd_recv);
        var at = try glm.afd_engine.Attn.init(e, &a);
        at.absorb = true;
        var tr = glm.trace.Trace.init(gpa);
        defer tr.deinit();
        var f = glm.fwd.Fwd.recording(&m, &tr);
        for (seqs) |q| if (q.capacity == s.capacity) {
            for (1..8) |k| { // fixed depth k: verify windows of k + 1 rows, two rounds
                const p = try at.prefill(&f, toks[0..q.prompt]);
                var pol: glm.afd_drafts.DepthPolicy = .{ .most = k, .fixed = true };
                _ = try glm.afd_drafts.mtp(&at, &f, p, 2 * (k + 1) + 1, got, &pol, false, null);
            }
            const p = try at.prefill(&f, toks[0..q.prompt]);
            var pol = glm.afd_drafts.DepthPolicy.auto(true);
            _ = try glm.afd_drafts.auto(&at, &f, p, 9, got, &pol, false, null);
        };
        try f.flush();
        const mv = mtpViews(&w.mtp.?);
        for (tr.events.items) |ev| if (ev.op == .triton) {
            for (sets, ts) |st, *t| try t.add(arena, st.entries, ev.name, ev.args, ev.consts);
            if (!std.mem.eql(u8, ev.name, "_fmm")) continue;
            var n: ?i64 = null;
            var k: ?i64 = null;
            for (ev.consts) |x| {
                if (std.mem.eql(u8, x.name, "N")) n = x.int;
                if (std.mem.eql(u8, x.name, "K")) k = x.int;
            }
            for (mv, mtp_fmm) |v, *cnt| if (n.? == @as(i64, @intCast(v[1].n())) and k.? == @as(i64, @intCast(v[1].k()))) {
                cnt.* += 1;
            };
        };
    }
    return out;
}

test "G-cover-120 MTP8 forms (mtp_fp8): the attention role with TF_AFD_MTP=1, q4 and the FP8 head, TF_GLM_MTP_FP8 unset and 1, over 078455ff's headers: one variant a launch in mtp_fp8_w1b; qualification_724c934b lacks only the MTP shapes' _fmm" {
    const mirror = std.mem.span(std.c.getenv("TF_GLM_EXL3_HEADERS") orelse return error.SkipZigTest);
    const q4dir = std.mem.span(std.c.getenv("TF_qualification_7877904d_Q4W1M") orelse return error.SkipZigTest);
    const layers: ?usize = if (std.c.getenv("TF_qualification_7877904d_LAYERS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else null;
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const old = try std.mem.concat(arena, aot.Spec, &.{ try c8.specEntries(arena, base_text), try c8.specEntries(arena, f8_text) });
    var sets: std.ArrayList(c8.Named) = .empty;
    try sets.append(arena, .{ .name = "mtp_fp8_w1b spec (kernels.json + kernels_f8_w1.json + kernels_f8_mtp_w1.json)", .entries = try std.mem.concat(arena, aot.Spec, &.{ old, try c8.specEntries(arena, mtp_text) }) });
    try sets.append(arena, .{ .name = "qualification_724c934b spec (kernels.json + kernels_f8_w1.json)", .entries = old });
    if (std.c.getenv("TF_qualification_7877904d_SETS")) |dv| { // the built sets' aot.json rows, as Set.load reads them
        const dir = std.mem.span(dv);
        try sets.append(arena, .{ .name = "mtp_fp8_w1b/set120/aot.json", .entries = try c8.setEntries(arena, try std.fmt.allocPrint(arena, "{s}/mtp_fp8_w1b/set120", .{dir})) });
        try sets.append(arena, .{ .name = "qualification_724c934b/set120/aot.json", .entries = try c8.setEntries(arena, try std.fmt.allocPrint(arena, "{s}/qualification_724c934b/set120", .{dir})) });
    }
    std.debug.print("G-cover-120 MTP8 forms: {d} layers + the MTP layer; drafts f1-f7 and auto after each prompt; sets:", .{layers orelse 45});
    for (sets.items) |s| std.debug.print(" {s} ({d} rows);", .{ s.name, s.entries.len });
    std.debug.print("\n", .{});
    var loads: [2]Load = undefined;
    for ([_]bool{ false, true }, &loads) |on, *ld| {
        const ts = try arena.alloc(c8.Tally, sets.items.len);
        for (ts) |*t| t.* = .{};
        var mtp_fmm: [8]usize = @splat(0);
        ld.* = try run(testing.allocator, arena, on, mirror, q4dir, layers, sets.items, ts, &mtp_fmm);
        std.debug.print("TF_GLM_MTP_FP8 {s}: dense q4 projections {d} from {d} files, MTP on FP8 {any}, {d} B on the device; the MTP views {any}; the MTP shapes' _fmm launches {any}\n", .{ if (on) "1" else "unset", ld.q4, ld.files, ld.mtp8, ld.bytes, ld.forms, mtp_fmm });
        for (sets.items, ts) |s, *t| t.print(s.name);
        ts[0].census(if (on) "mtp8" else "unset");
        for (sets.items, ts, 0..) |_, *t, i| {
            try testing.expect(t.launches > 0);
            if (i % 2 == 0) { // mtp_fp8_w1b (spec, then built rows): exactly one variant each
                try testing.expectEqual(t.launches, t.one);
                continue;
            }
            try testing.expectEqual(@as(usize, 0), t.many); // qualification_724c934b: the unset form whole; with the switch, all but the MTP _fmm
            var lack: usize = 0;
            for (mtp_fmm) |x| lack += x;
            try testing.expectEqual(t.none, lack);
            try testing.expectEqual(t.none, t.miss_fn.get("_fmm") orelse 0);
        }
        for (ld.forms) |fm| try testing.expectEqual(if (on) std.meta.Tag(glm.views.Linear).f8 else .q4, fm);
        for (mtp_fmm) |x| try testing.expect((x > 0) == on); // each of the eight launches _fmm only with the switch
    }
    // the switch: the eight leave the q4 overlay (q4-mtp not read) and their FP8 bytes replace their q4 triplets
    const s8 = loads[1].mtp8.?;
    try testing.expect(loads[0].mtp8 == null and s8.projections == 8);
    try testing.expectEqual(loads[0].q4 - 8, loads[1].q4);
    try testing.expectEqual(loads[0].files - 1, loads[1].files);
    var q4b: u64 = 0;
    var m = try c8.fixtureModel(testing.allocator);
    m.mtp_layers = 1;
    var w = try glm.views.rank(testing.allocator, &m);
    defer w.deinit(testing.allocator);
    for (mtpViews(&w.mtp.?)) |v| {
        const q = glm.views.Q4.of(v[1].n(), v[1].k());
        q4b += q.weight.bytes() + q.scales.bytes() + q.biases.bytes();
    }
    try testing.expectEqual(loads[0].bytes - q4b + s8.f8, loads[1].bytes);
    std.debug.print("TF_GLM_MTP_FP8=1: {d} B of FP8 for the MTP layer's 8 projections in place of {d} B of q4 triplets (BF16 {d} B): +{d} B on the device\n", .{ s8.f8, q4b, s8.bf16, loads[1].bytes - loads[0].bytes });
}
