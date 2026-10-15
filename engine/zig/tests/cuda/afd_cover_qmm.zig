//! qualification_18588626, G-cover CUDA: the qmm_group half of the attention role's cover (qualification_7877904d's G-cover checks Triton launches only).
//! Every `qmm.qmm_group` call an `attn --dry` records goes through qmm_group_cuda's launch (cuda_fwd.qmmGroupLaunch, the
//! GPU path's own) on attention-host's RTX 5090 (12.0, 170 SMs) and on a GB10 (12.1, 48 SMs), at python's tile pick for each
//! (qmm_tiles.group_tile): 0 NotBuilt and 0 Invalid wanted. qualification_96ad5727 met the gap on the 5090 (bench-q4b: NotBuilt at the
//! first decode step): the dry records python's call and stops before the launch, at the GB10's chip. Always: windows
//! of 1-16 rows at world-1 shapes, and the drafted bench --dry over the fixture's config (layers 0-4, TF_AFD_MTP=1: the
//! q4 draft head at world 1, a scripted picker so drafts go 7 deep). With TF_qualification_18588626_CASES and the mirrors: the 3-node
//! kits' attention cases as their DRY prints them (qualification_7877904d's g-cover/cases.json), each drafts case with --dry's stand-in
//! answers and with the scripted ones; TF_qualification_18588626_OUT gets the calls' shapes (python's launch fixture at world 1).
const std = @import("std");
const glm = @import("glm_cuda");
const cli = @import("tf_glm_afd.zig");
const attn = @import("afd_attn.zig");
const bench = @import("afd_attn_bench.zig");
const files = @import("afd_expert.zig");

const testing = std.testing;
const trace = glm.trace;
const T = trace.T;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// A chip as qmm_group_cuda reads it (getCurrentDeviceProperties: major, minor, SMs).
const Chip = struct { name: []const u8, major: u32, minor: u32, sms: u32 };
const chips = [_]Chip{
    .{ .name = "RTX 5090 12.0/170", .major = 12, .minor = 0, .sms = 170 },
    .{ .name = "GB10 12.1/48", .major = 12, .minor = 1, .sms = 48 },
};

/// A recorded call's shape: rows, n, k, x's row stride, the output's type and K slices (python's fixture at world 1).
pub const Shape = struct { m: usize, n: usize, k: usize, x_stride: usize, f32_out: bool, sk: usize };

/// A run's qmm_group calls on each chip: launches by kernel, NotBuilt and Invalid; calls by rows; the shapes.
const Tally = struct {
    a: std.mem.Allocator,
    events: usize = 0,
    calls: usize = 0,
    ext: std.StringArrayHashMapUnmanaged(usize) = .empty, // every extension call by name: the census
    kernels: [chips.len]std.StringArrayHashMapUnmanaged(usize) = .{ .empty, .empty },
    not_built: [chips.len]usize = @splat(0),
    invalid: [chips.len]usize = @splat(0),
    rows: [18]usize = @splat(0), // calls by rows; 17: more than 16
    shapes: std.AutoArrayHashMapUnmanaged(Shape, usize) = .empty,
    off_tile: usize = 0, // calls whose recorded tile is not group_tile's at the dry's chip (the Engine's GB10 default)

    fn bump(a: std.mem.Allocator, m: *std.StringArrayHashMapUnmanaged(usize), k: []const u8) !void {
        try addCount(a, m, k, 1);
    }

    fn addCount(a: std.mem.Allocator, m: *std.StringArrayHashMapUnmanaged(usize), k: []const u8, n: usize) !void {
        const g = try m.getOrPut(a, k);
        if (!g.found_existing) g.value_ptr.* = 0;
        g.value_ptr.* += n;
    }

    fn only(v: trace.Val) !trace.Val {
        if (v != .list or v.list.len != 1) return error.BadCall;
        return v.list[0];
    }

    fn sink(ctx: *anyopaque, events: []const trace.Event) anyerror!void {
        const t: *Tally = @ptrCast(@alignCast(ctx));
        t.events += events.len;
        for (events) |ev| {
            if (ev.op != .ext) continue;
            try bump(t.a, &t.ext, ev.name);
            if (std.mem.eql(u8, ev.name, "qmm.qmm_group")) try t.add(ev.vals);
        }
    }

    /// One recorded `_ext().qmm_group(x, xs, [w], [s], [b], [out], [n], [sk], f32, tile, -1)` through each chip's launch.
    fn add(t: *Tally, v: []const trace.Val) !void {
        if (v.len != 11 or v[10].int != -1) return error.BadCall;
        const x = v[0].t;
        const w = [3]T{ (try only(v[2])).t, (try only(v[3])).t, (try only(v[4])).t };
        const out = (try only(v[5])).t;
        const n: usize = @intCast((try only(v[6])).int);
        const sk: usize = @intCast((try only(v[7])).int);
        const m = x.shape[0];
        t.calls += 1;
        t.rows[@min(m, 17)] += 1;
        if (v[9].int != @as(i64, @intCast(glm.qmm.groupTile(m, 12, 1, 48)))) t.off_tile += 1;
        const s: Shape = .{ .m = m, .n = n, .k = x.shape[1], .x_stride = x.stride[0], .f32_out = v[8].bool, .sk = sk };
        const g = try t.shapes.getOrPut(t.a, s);
        if (!g.found_existing) g.value_ptr.* = 0;
        g.value_ptr.* += 1;
        for (chips, 0..) |c, i| {
            const pick = glm.qmm.groupTile(m, c.major, c.minor, c.sms);
            const l = glm.fwd.qmmGroupLaunch(.{ .major = c.major, .minor = c.minor, .sms = c.sms }, x, v[1].t, w, out, n, sk, v[8].bool, pick) catch |e| {
                switch (e) {
                    error.NotBuilt => t.not_built[i] += 1,
                    error.Invalid => t.invalid[i] += 1,
                }
                continue;
            };
            try bump(t.a, &t.kernels[i], @tagName(l.kernel));
        }
    }

    fn print(t: *const Tally, what: []const u8) void {
        std.debug.print("G-COVER-CUDA {s}: {d} events, {d} qmm_group calls ({d} shapes); rows:", .{ what, t.events, t.calls, t.shapes.count() });
        for (t.rows, 0..) |c, r| if (c > 0) std.debug.print(" {d}{s}:{d}", .{ r, if (r == 17) "+" else "", c });
        std.debug.print("\n", .{});
        for (chips, t.kernels, t.not_built, t.invalid) |c, k, nb, inv| {
            std.debug.print("  {s}: NotBuilt {d}, Invalid {d};", .{ c.name, nb, inv });
            var it = k.iterator();
            while (it.next()) |kv| std.debug.print(" {s} {d},", .{ kv.key_ptr.*, kv.value_ptr.* });
            std.debug.print("\n", .{});
        }
        std.debug.print("  extension calls:", .{});
        var it = t.ext.iterator();
        while (it.next()) |kv| std.debug.print(" {s} {d},", .{ kv.key_ptr.*, kv.value_ptr.* });
        std.debug.print("\n", .{});
    }

    /// The gates: every call launches on both chips, at the tile the dry's chip picks.
    fn expectBuilt(t: *const Tally) !void {
        for (t.not_built, t.invalid) |nb, inv| {
            try testing.expectEqual(@as(usize, 0), nb);
            try testing.expectEqual(@as(usize, 0), inv);
        }
        try testing.expectEqual(@as(usize, 0), t.off_tile);
    }
};

/// The scripted answers (qualification_7877904d's): greedy picks token 0; a draft's draw (its probability asked) 0.9, every third token 1.
const Script = struct {
    drafts: usize = 0,
    wrong: usize = 0,
    fn pick(ctx: *anyopaque, logits: T, positions: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        _ = logits;
        _ = positions;
        const s: *Script = @ptrCast(@alignCast(ctx));
        @memset(out, 0);
        const p = probs orelse return;
        for (out, 0..) |*o, i| {
            s.drafts += 1;
            if (s.drafts % 3 == 0) {
                o.* = 1;
                s.wrong += 1;
            }
            if (i < p.len) p[i] = 0.9;
        }
    }
};

/// `attn --dry` with the sink (and the scripted picker when `script` is set); its outcome must be ok.
fn dryCovered(a: std.mem.Allocator, f: cli.Flags, config: []const u8, tally: *Tally, script: ?*Script) !attn.Dry {
    var pb: [4096]u8 = undefined;
    var t = try attn.tokens(a, if (f.verb == .bench) try bench.decodePrompt(&pb, f.prompt.?, f.bench) else f.prompt.?, f.teacher);
    defer t.deinit(a);
    attn.dry_sink = .{ .ctx = tally, .f = Tally.sink };
    attn.dry_picker = if (script) |s| .{ .ctx = s, .pick = Script.pick } else null;
    defer {
        attn.dry_sink = null;
        attn.dry_picker = null;
    }
    const d = try attn.dry(a, f, &t, config);
    var lb: [256]u8 = undefined;
    std.debug.print("G-COVER-CUDA dry: {s}; {d} forwards, {d} events recorded\n", .{ d.outcome.line(&lb), d.forwards, d.launches });
    try testing.expect(d.outcome == .ok);
    try testing.expectEqual(d.launches, tally.events);
    return d;
}

test "G-cover CUDA (qualification_18588626): windows of 1-16 rows at world-1 shapes launch on the RTX 5090 (tiles 7, 8) and a GB10 (tile 2)" {
    // world-1 projections (n, k, x stride, fp32 out): the draft head (vocab x hidden), the KDA in-projection (64 heads),
    // the shared expert's down, DSA's q_b; K slices as qmm.split_k gives them
    const projs = [_][4]usize{ .{ 154880, 4096, 4096, 0 }, .{ 24896, 4096, 4096, 0 }, .{ 4096, 6144, 6144, 1 }, .{ 16384, 1536, 1536, 0 } };
    var tally: Tally = .{ .a = testing.allocator };
    defer {
        for (&tally.kernels) |*k| k.deinit(testing.allocator);
        tally.shapes.deinit(testing.allocator);
    }
    for (projs) |p| for (1..17) |m| {
        const npad = (p[0] + 127) / 128 * 128;
        const kg = p[1] / 64;
        const x: T = .{ .ptr = 1 << 36, .dtype = .bf16, .dims = 2, .shape = .{ m, p[1], 1, 1, 1 }, .stride = .{ p[2], 1, 1, 1, 1 } };
        const xs: T = .{ .ptr = 2 << 36, .dtype = .f32, .dims = 2, .shape = .{ m, kg, 1, 1, 1 }, .stride = .{ kg, 1, 1, 1, 1 } };
        const ws: T = .{ .ptr = 3 << 36, .dtype = .bf16, .dims = 2, .shape = .{ kg, npad, 1, 1, 1 }, .stride = .{ npad, 1, 1, 1, 1 } };
        const out: T = .{ .ptr = 4 << 36, .dtype = if (p[3] == 1) .f32 else .bf16, .dims = 2, .shape = .{ m, p[0], 1, 1, 1 }, .stride = .{ p[0], 1, 1, 1, 1 } };
        const vals = [_]trace.Val{
            .{ .t = x },                                 .{ .t = xs },
            .{ .list = &.{.{ .t = ws }} },               .{ .list = &.{.{ .t = ws }} },
            .{ .list = &.{.{ .t = ws }} },               .{ .list = &.{.{ .t = out }} },
            .{ .list = &.{.{ .int = @intCast(p[0]) }} }, .{ .list = &.{.{ .int = @intCast(glm.qmm.splitK(p[0], p[1])) }} },
            .{ .bool = p[3] == 1 },                      .{ .int = 0 },
            .{ .int = -1 },
        };
        try tally.add(&vals);
    };
    tally.print("windows 1-16 at world-1 shapes");
    try tally.expectBuilt();
    try testing.expectEqual(@as(usize, 4 * 8), tally.kernels[0].get("group7_bf16").? + tally.kernels[0].get("group7_f32").?);
    try testing.expectEqual(@as(usize, 4 * 8), tally.kernels[0].get("group8_bf16").? + tally.kernels[0].get("group8_f32").?);
    try testing.expectEqual(@as(usize, 4 * 16), tally.kernels[1].get("group_bf16").? + tally.kernels[1].get("group_f32").?);
}

test "G-cover CUDA (qualification_18588626): the drafted bench --dry over the fixture's config (layers 0-4, TF_AFD_MTP=1, drafts 7 deep): every qmm_group call built on the RTX 5090 and a GB10" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try std.process.currentPathAlloc(testing.io, a);
    const out = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    var pj: std.ArrayList(u8) = .empty;
    try pj.appendSlice(a, "{\"prompts\": {\"a\": [");
    for (0..37) |i| try pj.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", 100 + i });
    try pj.appendSlice(a, "], \"b\": [");
    for (0..40) |i| try pj.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", 200 + i });
    try pj.appendSlice(a, "]}}");
    const pfile = try std.fmt.allocPrint(a, "{s}/p.json", .{out});
    try attn.save(pfile, pj.items);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("afd_attn_fixture"), .{});
    const config = root.object.get("config_json").?.string;
    const spec = "decode=a:12,prefill=b,runs=1,warmup=0,skip=1,drafts=f1+f7+c7:0.35+auto";
    const f: cli.Flags = .{ .role = .attn, .verb = .bench, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = pfile, .bench = spec };
    _ = setenv("TF_AFD_MTP", "1", 1);
    defer _ = unsetenv("TF_AFD_MTP");
    var tally: Tally = .{ .a = a };
    var script: Script = .{};
    _ = try dryCovered(a, f, config, &tally, &script);
    tally.print("fixture bench --dry, layers 0-4, TF_AFD_MTP=1, scripted drafts");
    try tally.expectBuilt();
    try testing.expect(tally.calls > 0 and script.drafts > 0 and script.wrong > 0);
    try testing.expect(tally.kernels[0].get("group7_bf16").? == tally.calls); // the q4 draft head: one row, bf16 logits
}

const Case = struct { kit: []const u8, case: []const u8, card: []const u8, gib: i64, env: std.json.ArrayHashMap([]const u8), argv: []const []const u8 };

/// One kit case in one mode: the environment as the kit sets it, the flags through cli.parse, then attn --dry's run (on
/// its own arena: a full-depth bench records millions of events); its counts merged into `all` (on `a`).
fn runCase(a: std.mem.Allocator, c: Case, scripted: bool, all: *Tally) !void {
    var case_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer case_arena.deinit();
    const ca = case_arena.allocator();
    for ([_][:0]const u8{ "TF_AFD_MTP", "TF_GLM_DENSE", "TF_GLM_F8", "TF_AFD_TIME" }) |k| _ = unsetenv(k);
    var it = c.env.map.iterator();
    while (it.next()) |kv| _ = setenv(try ca.dupeSentinel(u8, kv.key_ptr.*, 0), try ca.dupeSentinel(u8, kv.value_ptr.*, 0), 1);
    defer {
        for ([_][:0]const u8{ "TF_AFD_MTP", "TF_GLM_DENSE", "TF_GLM_F8", "TF_AFD_TIME" }) |k| _ = unsetenv(k);
    }
    const argv = try ca.alloc([:0]const u8, c.argv.len);
    for (c.argv, argv) |s, *d| d.* = try ca.dupeSentinel(u8, s, 0);
    const f = try cli.parse(ca, .attn, argv);
    const config = try files.readFile(ca, try std.fmt.allocPrint(ca, "{s}/config.json", .{f.model}));
    var tally: Tally = .{ .a = ca };
    var script: Script = .{};
    _ = try dryCovered(ca, f, config, &tally, if (scripted) &script else null);
    tally.print(try std.fmt.allocPrint(ca, "{s} {s} ({s})", .{ c.kit, c.case, if (scripted) "scripted" else "stand-in" }));
    try tally.expectBuilt();
    // a decode window (bench, teacher) runs the q4 projections and the q4 draft head through qmm_group; `first` is one
    // prompt chunk, all qmm_prefill
    const q4 = c.env.map.get("TF_GLM_DENSE") != null or c.env.map.get("TF_AFD_MTP") != null;
    if (f.verb != .first) try testing.expect((tally.calls > 0) == q4) else try testing.expectEqual(@as(usize, 0), tally.calls);
    if (scripted) try testing.expect(script.drafts > 0 and script.wrong > 0);
    all.events += tally.events;
    all.calls += tally.calls;
    for (0..chips.len) |i| {
        all.not_built[i] += tally.not_built[i];
        all.invalid[i] += tally.invalid[i];
        var ki = tally.kernels[i].iterator(); // keys are @tagName's (static)
        while (ki.next()) |kv| try Tally.addCount(a, &all.kernels[i], kv.key_ptr.*, kv.value_ptr.*);
    }
    for (tally.rows, 0..) |r, i| all.rows[i] += r;
    all.off_tile += tally.off_tile;
    var si = tally.shapes.iterator();
    while (si.next()) |kv| {
        const g = try all.shapes.getOrPut(a, kv.key_ptr.*);
        if (!g.found_existing) g.value_ptr.* = 0;
        g.value_ptr.* += kv.value_ptr.*;
    }
    var ei = tally.ext.iterator();
    while (ei.next()) |kv| try Tally.addCount(a, &all.ext, try a.dupe(u8, kv.key_ptr.*), kv.value_ptr.*);
}

test "G-cover CUDA (qualification_18588626): the 3-node kits' attention cases (TF_qualification_18588626_CASES, over the mirrors) through attn --dry: every qmm_group call built on the RTX 5090 and a GB10" {
    const path = std.mem.span(std.c.getenv("TF_qualification_18588626_CASES") orelse return error.SkipZigTest);
    const only: ?[]const u8 = if (std.c.getenv("TF_qualification_18588626_ONLY")) |v| std.mem.span(v) else null;
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const Cases = struct { cases: []Case };
    const parsed = try std.json.parseFromSliceLeaky(Cases, a, try files.readFile(a, path), .{ .ignore_unknown_fields = true });
    var all: Tally = .{ .a = a };
    var runs: usize = 0;
    for (parsed.cases) |c| {
        if (only) |o| if (std.mem.indexOf(u8, o, c.case) == null) continue;
        if (std.mem.eql(u8, c.argv[0], "check-weights")) { // the load alone: no forward, no launch
            std.debug.print("G-COVER-CUDA {s} {s}: check-weights records no launch\n", .{ c.kit, c.case });
            continue;
        }
        try runCase(a, c, false, &all);
        runs += 1;
        if (c.env.map.get("TF_AFD_MTP") != null) {
            try runCase(a, c, true, &all);
            runs += 1;
        }
    }
    all.print(try std.fmt.allocPrint(a, "all {d} runs", .{runs}));
    try all.expectBuilt();
    try testing.expect(runs > 0 and all.calls > 0);
    // every shape recorded is one of python's world-1 launch fixture's (tools/zig/glm_mm_w1_shapes.json, the input of
    // fixtures_cuda_mm_w1.json): a new shape means rerunning tools/zig/glm_mm_w1_fixtures.py
    const Known = struct { shapes: []const struct { m: usize, n: usize, k: usize, x_stride: usize, f32: bool, sk: usize } };
    const known = try std.json.parseFromSliceLeaky(Known, a, try files.readFile(a, "tools/zig/glm_mm_w1_shapes.json"), .{ .ignore_unknown_fields = true });
    var missing: usize = 0;
    var ki = all.shapes.iterator();
    while (ki.next()) |kv| {
        const s = kv.key_ptr.*;
        var found = false;
        for (known.shapes) |q| found = found or (q.m == s.m and q.n == s.n and q.k == s.k and q.x_stride == s.x_stride and q.f32 == s.f32_out and q.sk == s.sk);
        if (!found) {
            std.debug.print("G-COVER-CUDA: shape m {d} n {d} k {d} x_stride {d} f32 {} sk {d} is not in tools/zig/glm_mm_w1_shapes.json\n", .{ s.m, s.n, s.k, s.x_stride, s.f32_out, s.sk });
            missing += 1;
        }
    }
    std.debug.print("G-COVER-CUDA: {d} shapes recorded, {d} of them not in python's world-1 launch fixture\n", .{ all.shapes.count(), missing });
    try testing.expectEqual(@as(usize, 0), missing);
    const dir = std.mem.span(std.c.getenv("TF_qualification_18588626_OUT") orelse return);
    var j: std.ArrayList(u8) = .empty;
    try j.appendSlice(a, "{\"schema\": \"qualification_7a39d983/1\", \"shapes\": [\n");
    var si = all.shapes.iterator();
    var first = true;
    while (si.next()) |kv| {
        const s = kv.key_ptr.*;
        try j.print(a, "{s} {{\"m\": {d}, \"n\": {d}, \"k\": {d}, \"x_stride\": {d}, \"f32\": {}, \"sk\": {d}, \"calls\": {d}}}", .{ if (first) "" else ",\n", s.m, s.n, s.k, s.x_stride, s.f32_out, s.sk, kv.value_ptr.* });
        first = false;
    }
    try j.appendSlice(a, "\n]}\n");
    try attn.save(try std.fmt.allocPrint(a, "{s}/qmm-shapes.json", .{dir}), j.items);
}
