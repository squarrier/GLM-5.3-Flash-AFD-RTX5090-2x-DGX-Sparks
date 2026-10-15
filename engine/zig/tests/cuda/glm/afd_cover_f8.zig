//! qualification_7877904d's G-cover-120 F8: the AFD attention node's Triton launches at world 1 against the FP8-head set120
//! (evidence/SETS/qualification_724c934b: qualification_955b7191's kernels.json, 478 entries, and kernels_f8_w1.json, the world-1 head's 9 `_fmm`
//! variants evidence/qualification_7877904d/f8/f8aot_w1.py built for sm_120a), by aot.zig's own rule: exactly one variant a launch.
//! qualification_589d8f6f's set (kernels.json alone) is the negative control: the FP8 head finds no `_fmm` there (qualification_e8c93577's
//! MissingTritonVariant). The head (python's fixture against Tri.matmulF8 at N 154,880) always; the attention role's
//! verbs in the BF16, q4 + BF16 head and q4 + FP8 head forms over 078455ff's header mirror and the world-1 triplets
//! when TF_GLM_EXL3_HEADERS and TF_qualification_7877904d_Q4W1 name them (else skipped), and the built aot.json files of both sets when
//! TF_qualification_7877904d_SETS names evidence/SETS. `_fmm` is Mia's (MiaAI-Lab, patch 0002); the cover's form is qualification_955b7191's and qualification_qualification_ebed70b7's.
const std = @import("std");
const aot = @import("cuda").aot;
const glm = @import("glm_cuda");

const testing = std.testing;
const base_text = @embedFile("kernels.json");
const f8_text = @embedFile("kernels_f8_w1.json");
const fixture_text = @embedFile("fixtures_triton_f8_w1.json");
const config_fixture = @embedFile("afd_attn_fixture");

/// A spec's entries (glm_aot.py's kernels.json form) as aot.json rows (aot_row, zig_const); `hash` empty when derived.
pub fn specEntries(arena: std.mem.Allocator, text: []const u8) ![]aot.Spec {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
    const kernels = root.object.get("kernels").?.array.items;
    const out = try arena.alloc(aot.Spec, kernels.len);
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
        const hash = if (o.get("hash")) |h| h.string else "";
        e.* = .{ .@"fn" = name, .hash = hash, .name = name, .num_warps = 0, .params = params.items, .consts = consts };
    }
    return out;
}

/// A built set's rows as aot.zig's Set.load reads DIR/aot.json (no cubins).
pub fn setEntries(arena: std.mem.Allocator, dir: []const u8) ![]aot.Spec {
    const text = try readFile(arena, try std.fmt.allocPrint(arena, "{s}/aot.json", .{dir}));
    const SetJson = struct { kernels: []aot.Spec };
    const p = try std.json.parseFromSliceLeaky(SetJson, arena, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    return p.kernels;
}

pub fn readFile(arena: std.mem.Allocator, path: []const u8) ![]u8 {
    const p = try arena.dupeSentinel(u8, path, 0);
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.NoFile;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    while (true) {
        try out.ensureUnusedCapacity(arena, 1 << 20);
        const room = out.unusedCapacitySlice();
        const n = std.c.read(fd, room.ptr, room.len);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        out.items.len += @intCast(n);
    }
    return out.items;
}

pub fn hits(set: []const aot.Spec, name: []const u8, args: []const aot.Arg, consts: []const aot.Const, used: ?[]usize) usize {
    var n: usize = 0;
    for (set, 0..) |e, i| if (std.mem.eql(u8, e.@"fn", name) and aot.accepts(e, args, consts)) {
        n += 1;
        if (used) |u| u[i] += 1;
    };
    return n;
}

/// A launch's specialization, as aot.zig's `matches` reads it: kernel, constexprs, each argument's type and form.
fn specKey(arena: std.mem.Allocator, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) ![]const u8 {
    var b: std.ArrayList(u8) = .empty;
    try b.appendSlice(arena, name);
    for (consts) |c| if (c.int) |v| try b.print(arena, " {s}={d}", .{ c.name, v }) else if (c.f32) |x| try b.print(arena, " {s}=f{x}", .{ c.name, @as(u32, @bitCast(x)) }) else try b.print(arena, " {s}=?", .{c.name});
    for (args) |a| switch (a.value) {
        .ptr => |p| try b.print(arena, " {s}:{s}{s}", .{ a.name, p.ty, if (p.addr % 16 == 0) "/16" else "" }),
        .i32 => |x| try b.print(arena, " {s}:{s}", .{ a.name, if (x == 1) "1" else if (@mod(x, 16) == 0) "div16" else "plain" }),
        .f32 => try b.print(arena, " {s}:fp32", .{a.name}),
        .u64 => |x| try b.print(arena, " {s}:u64{s}", .{ a.name, if (x % 16 == 0) "/16" else "" }),
    };
    return b.items;
}

/// Launches by how many variants of a set take them, by kernel; the specializations without one (the census).
pub const Tally = struct {
    launches: usize = 0,
    one: usize = 0,
    none: usize = 0,
    many: usize = 0,
    by_fn: std.StringArrayHashMapUnmanaged(usize) = .empty,
    miss_fn: std.StringArrayHashMapUnmanaged(usize) = .empty,
    misses: std.StringArrayHashMapUnmanaged(usize) = .empty,
    seen: std.StringArrayHashMapUnmanaged(usize) = .empty,

    fn bump(arena: std.mem.Allocator, m: *std.StringArrayHashMapUnmanaged(usize), key: []const u8) !void {
        const g = try m.getOrPut(arena, key);
        if (!g.found_existing) g.value_ptr.* = 0;
        g.value_ptr.* += 1;
    }

    pub fn add(t: *Tally, arena: std.mem.Allocator, set: []const aot.Spec, name: []const u8, args: []const aot.Arg, consts: []const aot.Const) !void {
        const key = try specKey(arena, name, args, consts);
        const g = try t.seen.getOrPut(arena, key);
        if (!g.found_existing) g.value_ptr.* = hits(set, name, args, consts, null);
        t.launches += 1;
        try bump(arena, &t.by_fn, name);
        switch (g.value_ptr.*) {
            0 => {
                t.none += 1;
                try bump(arena, &t.miss_fn, name);
                var d: []const u8 = key;
                for (args) |a| switch (a.value) {
                    .i32 => |v| d = try std.fmt.allocPrint(arena, "{s} {s}={d}", .{ d, a.name, v }),
                    else => {},
                };
                try bump(arena, &t.misses, d);
            },
            1 => t.one += 1,
            else => t.many += 1,
        }
    }

    pub fn print(t: *const Tally, what: []const u8) void {
        std.debug.print("{s}: {d} Triton launches ({d} specializations); one variant {d}, none {d}, several {d}\n", .{ what, t.launches, t.seen.count(), t.one, t.none, t.many });
        var it = t.miss_fn.iterator();
        while (it.next()) |kv| std.debug.print("  lacking: {s} x{d}\n", .{ kv.key_ptr.*, kv.value_ptr.* });
        var mi = t.misses.iterator();
        var shown: usize = 0;
        while (mi.next()) |kv| : (shown += 1) if (shown < 12) std.debug.print("    no variant: {s} (x{d})\n", .{ kv.key_ptr.*, kv.value_ptr.* });
    }

    pub fn census(t: *const Tally, what: []const u8) void {
        std.debug.print("  census {s} by fn:", .{what});
        var it = t.by_fn.iterator();
        while (it.next()) |kv| std.debug.print(" {s} {d},", .{ kv.key_ptr.*, kv.value_ptr.* });
        std.debug.print("\n", .{});
    }
};

/// The fixture's model (python 2.15's config.json for 078455ff, its `config_json`) at world 1.
pub fn fixtureModel(gpa: std.mem.Allocator) !glm.model.Model {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_fixture, .{});
    defer parsed.deinit();
    return glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
}

pub fn addr(i: u64) u64 {
    return 0x7f0000000000 + i * 0x10000000;
}

test "G-cover-120 F8 (qualification_7877904d): the world-1 head's _fmm as python launches it at 1-8, 16 and the kit's chunk rows: one variant each in the new set120, none in qualification_589d8f6f's" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const base = try specEntries(arena, base_text);
    const f8 = try specEntries(arena, f8_text);
    const set = try std.mem.concat(arena, aot.Spec, &.{ base, f8 });
    const used = try arena.alloc(usize, set.len);
    @memset(used, 0);
    const m = try fixtureModel(testing.allocator);
    const n = m.vl(); // Linear.of(m, m.vl(), m.hidden): a world-1 rank's head holds every vocabulary row
    const k = m.hidden;
    try testing.expectEqual(@as(usize, 154880), n);
    try testing.expectEqual(@as(usize, 4096), k);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, fixture_text, .{});
    const head = root.object.get("head").?.array.items;
    try testing.expectEqual(@as(i64, @intCast(n)), head[0].integer);
    try testing.expectEqual(@as(i64, @intCast(k)), head[1].integer);
    var it = root.object.get("cases").?.object.iterator();
    var cases: usize = 0;
    var launches: usize = 0;
    var tr = glm.trace.Trace.init(testing.allocator);
    defer tr.deinit();
    var f = glm.fwd.Fwd.recording(&m, &tr);
    while (it.next()) |c| {
        const rows = try std.fmt.parseInt(usize, c.key_ptr.*["fmm/head-w1/r".len..], 10);
        const at = tr.events.items.len;
        try f.tri.matmulF8(addr(1), k, addr(2), addr(3), addr(4), false, addr(5), rows, n, k);
        try f.flush();
        const got = tr.events.items[at..];
        const want = c.value_ptr.array.items;
        errdefer std.debug.print("case {s}\n", .{c.key_ptr.*});
        try testing.expectEqual(want.len, got.len);
        for (want, got) |wv, g| {
            const w = wv.object;
            try testing.expectEqualStrings(w.get("fn").?.string, g.name);
            for (w.get("grid").?.array.items, g.grid) |a, b| try testing.expectEqual(a.integer, @as(i64, b));
            const wa = w.get("args").?.array.items;
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
            var ci = w.get("consts").?.object.iterator();
            while (ci.next()) |kv| if (kv.value_ptr.* != .null) {
                const want_c = kv.value_ptr.object.get("int").?.integer;
                const gc = for (g.consts) |x| {
                    if (std.mem.eql(u8, x.name, kv.key_ptr.*)) break x;
                } else return error.MissingConst;
                try testing.expectEqual(want_c, gc.int.?);
                nc += 1;
            };
            try testing.expectEqual(nc, g.consts.len);
            try testing.expectEqual(@as(usize, 1), hits(set, g.name, g.args, g.consts, used)); // the new set120: one
            try testing.expectEqual(@as(usize, 0), hits(base, g.name, g.args, g.consts, null)); // qualification_589d8f6f: none (qualification_e8c93577)
            launches += 1;
        }
        cases += 1;
    }
    try testing.expectEqual(@as(usize, 22), cases);
    try testing.expectEqual(@as(usize, 22), launches);
    try testing.expectEqual(@as(usize, 9), f8.len); // BM 16: M 1, div16, plain; BM 32, 64, 128: div16, plain
    for (set[base.len..], used[base.len..]) |e, u| if (e.hash.len > 0 and u == 0) return error.DeadVariant;
    // negative controls: B16's 64 columns at the 16-row bucket, and the TP2 rank's head (N 77,440), are no variant
    const r1 = tr.events.items[0];
    const bad = try arena.dupe(aot.Const, r1.consts);
    for (bad) |*x| if (std.mem.eql(u8, x.name, "BLOCK_N")) {
        x.int = 64;
    };
    try testing.expectEqual(@as(usize, 0), hits(set, r1.name, r1.args, bad, null));
    const end = tr.events.items.len;
    try f.tri.matmulF8(addr(1), k, addr(2), addr(3), addr(4), false, addr(5), 1, n / 2, k);
    try f.flush();
    const tp2 = tr.events.items[end];
    try testing.expectEqual(@as(usize, 0), hits(set, tp2.name, tp2.args, tp2.consts, null));
    std.debug.print("G-cover-120 F8: the world-1 head (N {d}, K {d}): {d} python launches = Zig's, one variant each in qualification_724c934b's spec (478 + {d} _fmm), none in qualification_589d8f6f's; negative controls (BLOCK_N 64 at BM 16, the TP2 head N {d}): none\n", .{ n, k, launches, f8.len, n / 2 });
}

/// A form of the attention role's dense weights (qualification_e5a7fc3b's kit: first-a / bench-bf16, first-a-q4b / bench-q4b, first-a-q4
/// / bench-q4).
const Form = enum { bf16, q4b, q4f8 };

/// One kit sequence: a prompt of `prompt` tokens at `capacity`, then the verify windows of 1-8 rows (each committed:
/// all rows, then one), then `steps` greedy steps.
const Seq = struct { capacity: usize, prompt: usize, steps: usize };

/// qualification_e5a7fc3b's kit at its capacities: A (64) and qualification_e84ee1bf's p1024, p1900 at 2,560 (first-a*, bench-*); A again and LZ2b's B
/// (5,000: chunks 2,048, 2,048, 904) at 6,144 (teacher-a's capacity).
const seqs = [_]Seq{ .{ .capacity = 2560, .prompt = 64, .steps = 2 }, .{ .capacity = 2560, .prompt = 1024, .steps = 2 }, .{ .capacity = 2560, .prompt = 1900, .steps = 1 }, .{ .capacity = 6144, .prompt = 64, .steps = 3 }, .{ .capacity = 6144, .prompt = 5000, .steps = 1 } };

pub const Named = struct { name: []const u8, entries: []const aot.Spec };

/// What a form's load bound: projections held q4, the FP8 head's saving, views bound, the head's form.
const Load = struct { q4: usize = 0, f8_saved: ?u64 = null, bound: usize = 0, head: []const u8 = "" };

/// The attention role in `form` over the mirror (tf-glm-afd attn --dry's load: afd_load.load, the dense-q4 overlay,
/// make_f8's plan, stand-ins, bind), its verbs recorded at each capacity; every Triton launch into `ts` (one a set).
fn runForm(gpa: std.mem.Allocator, arena: std.mem.Allocator, form: Form, mirror: []const u8, q4dir: []const u8, layers: ?usize, sets: []const Named, ts: []Tally) !Load {
    const text = try readFile(arena, try std.fmt.allocPrint(arena, "{s}/config.json", .{mirror}));
    var out: Load = .{};
    const toks = try arena.alloc(u32, 5000);
    for (toks, 0..) |*t, i| t.* = @intCast(i % 1000 + 1);
    var cap_done: usize = 0;
    for (seqs) |s| {
        if (s.capacity == cap_done) continue; // one engine a capacity, its sequences in turn
        cap_done = s.capacity;
        var m = try glm.model.Model.parse(gpa, text, 1, 0);
        if (layers) |l| try m.cut(l);
        m.mtp_layers = 0;
        var w = try glm.views.rank(gpa, &m);
        defer w.deinit(gpa);
        var ix = try glm.index.open(gpa, mirror);
        defer ix.deinit();
        var c = try glm.config.parse(gpa, text);
        c.run = @intCast(m.layers);
        var ld = try glm.afd_load.load(gpa, &m, &c, &ix.names);
        defer ld.deinit();
        if (form != .bf16) {
            const q = try glm.pack.q4.overlay(gpa, &ld.pk, &ld.x.groups, &ld.x.copies, &ix, m.layers, false, 0, 1, q4dir);
            out.q4 = q.triplets;
            if (form == .q4f8) out.f8_saved = try glm.pack.f8.apply(&ld.pk);
        }
        const top = try ld.stand(1 << 36);
        out.bound = try glm.afd_load.bind(&w, &ld.pk);
        out.head = @tagName(w.head);
        var space: glm.trace.Space = .{ .gpa = gpa };
        defer space.deinit();
        space.next = std.mem.alignForward(u64, top, 1 << 36) + (1 << 36);
        const e = try glm.engine.Engine.create(gpa, &m, &w, s.capacity, 8, 2048, s.capacity > m.denseLimit());
        defer e.destroy(gpa);
        try e.walk(&space, glm.trace.placeAll);
        try e.walkTmp(&space, glm.trace.placeAll);
        e.tmp.space = &space;
        var a = try glm.afd_attn.Afd.init(&m, 2048);
        try space.place("b.afd_recv", &a.afd_recv);
        var at = try glm.afd_engine.Attn.init(e, &a);
        var tr = glm.trace.Trace.init(gpa);
        defer tr.deinit();
        var f = glm.fwd.Fwd.recording(&m, &tr);
        for (seqs) |q| if (q.capacity == s.capacity) {
            _ = try at.prefill(&f, toks[0..q.prompt]);
            for (1..9) |R| { // the drafted decode's verify windows (k + 1 <= 8 rows), every row kept
                _ = try at.forward(&f, toks[0..R]);
                try glm.compute.commit(&f, &e.st, &e.buf, R, R);
            }
            for (2..9) |R| { // and their first draft refused: one row kept
                _ = try at.forward(&f, toks[0..R]);
                try glm.compute.commit(&f, &e.st, &e.buf, R, 1);
            }
            for (0..q.steps) |_| _ = try at.step(&f, 1);
        };
        try f.flush();
        for (tr.events.items) |ev| if (ev.op == .triton) for (sets, ts) |st, *t| try t.add(arena, st.entries, ev.name, ev.args, ev.consts);
    }
    return out;
}

test "G-cover-120 F8 forms (qualification_7877904d): the attention role at world 1, BF16, q4 + BF16 head, q4 + FP8 head, over 078455ff's headers: one variant a launch in the new set120; qualification_589d8f6f's lacks only _fmm" {
    const mirror = std.mem.span(std.c.getenv("TF_GLM_EXL3_HEADERS") orelse return error.SkipZigTest);
    const q4dir = std.mem.span(std.c.getenv("TF_qualification_7877904d_Q4W1") orelse return error.SkipZigTest);
    const layers: ?usize = if (std.c.getenv("TF_qualification_7877904d_LAYERS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else null;
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const base = try specEntries(arena, base_text);
    const f8 = try specEntries(arena, f8_text);
    var sets: std.ArrayList(Named) = .empty;
    try sets.append(arena, .{ .name = "qualification_724c934b spec (kernels.json + kernels_f8_w1.json)", .entries = try std.mem.concat(arena, aot.Spec, &.{ base, f8 }) });
    try sets.append(arena, .{ .name = "qualification_589d8f6f spec (kernels.json)", .entries = base });
    if (std.c.getenv("TF_qualification_7877904d_SETS")) |d| { // the built sets' aot.json rows, as Set.load reads them
        const dir = std.mem.span(d);
        try sets.append(arena, .{ .name = "qualification_724c934b/set120/aot.json", .entries = try setEntries(arena, try std.fmt.allocPrint(arena, "{s}/qualification_724c934b/set120", .{dir})) });
        try sets.append(arena, .{ .name = "qualification_589d8f6f/set120/aot.json", .entries = try setEntries(arena, try std.fmt.allocPrint(arena, "{s}/qualification_589d8f6f/set120", .{dir})) });
    }
    std.debug.print("G-cover-120 F8 forms: {d} layers; sets:", .{layers orelse 45});
    for (sets.items) |s| std.debug.print(" {s} ({d} rows);", .{ s.name, s.entries.len });
    std.debug.print("\n", .{});
    for ([_]Form{ .bf16, .q4b, .q4f8 }) |form| {
        const ts = try arena.alloc(Tally, sets.items.len);
        for (ts) |*t| t.* = .{};
        const ld = try runForm(testing.allocator, arena, form, mirror, q4dir, layers, sets.items, ts);
        std.debug.print("form {t}: dense q4 projections {d}, lm_head {s} (FP8 saves {d} B), {d} views bound\n", .{ form, ld.q4, ld.head, ld.f8_saved orelse 0, ld.bound });
        for (sets.items, ts) |s, *t| t.print(s.name);
        ts[0].census(@tagName(form));
        const fmm = ts[0].by_fn.get("_fmm") orelse 0;
        try testing.expect(ts[0].launches > 0);
        for (sets.items, ts, 0..) |_, *t, i| {
            if (i % 2 == 0) { // the new set (spec, then built rows): exactly one variant each
                try testing.expectEqual(t.launches, t.one);
                continue;
            }
            try testing.expectEqual(@as(usize, 0), t.many);
            const lack = t.miss_fn.get("_fmm") orelse 0; // qualification_589d8f6f: the FP8 head's _fmm alone, qualification_e8c93577's miss
            try testing.expectEqual(t.none, lack);
            try testing.expectEqual(if (form == .q4f8) fmm else 0, lack);
        }
        switch (form) {
            .q4f8 => {
                try testing.expect(fmm > 0 and ld.f8_saved != null and std.mem.eql(u8, ld.head, "f8"));
            },
            else => try testing.expect(fmm == 0 and std.mem.eql(u8, ld.head, "b16")),
        }
        if (form != .bf16) try testing.expect(ld.q4 > 0);
    }
}
