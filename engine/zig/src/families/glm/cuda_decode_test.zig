//! qualification_qualification_44ba554e.7 day 1's check of the decode loops with MTP drafts (cuda_decode.zig) against python-0.6's: decode.prefill of a
//! 5-token prompt, then serial_decode, mtp_decode (fixed 3 and 7 drafts, adaptive, confidence 0.35) or auto_decode's
//! MTP arm (fixed 3; engine.py's greedy c3:0.35), recorded on a CPU by tools/zig/glm_mtp_fixtures.py over a checkpoint's
//! headers (MLX 4-bit; EXL3) for a TP2 rank cut to layers 0-2 with the MTP head (its layer's blocks are `mark.*`
//! events). A scripted model answers every sample call; the Zig Engine replays the answers, checks its positions
//! against python's, and must make python's launches event for event: the verify windows, commit's KDA replay and conv
//! shift past rejected rows, the head's absorb, chain and trim of rejected drafts. Then its tokens, counters and state.
//! A toy model (drafts right or wrong by a hash) checks drafted == serial over many depths, policies and stops.
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const sample = @import("cuda_sample.zig");
const nucleus = @import("cuda_nucleus.zig");
const eng = @import("cuda_engine.zig");
const decode = @import("cuda_decode.zig");
const lt = @import("cuda_layer_test.zig");
const ct = @import("cuda_compute_test.zig");

const testing = std.testing;
const J = std.json.Value;
const get = ct.get;
const int = ct.int;
pub const E = eng.EngineOf(ct.Marks);
pub const fixture_mlx = @embedFile("fixtures_cuda_decode.json");
const fixture_exl3 = @embedFile("fixtures_cuda_decode_exl3.json");
const fixture_probs = @embedFile("fixtures_cuda_draft_probs.json");
const fixture_sampled = @embedFile("fixtures_cuda_decode_sampled.json");
const fixture_sampled_exl3 = @embedFile("fixtures_cuda_decode_sampled_exl3.json");

/// The rank of a decode fixture: its model (cut, with the MTP head kept), weights, python's decode.Engine (with the
/// nucleus rows, for sampled cases) and the stand-in space.
pub const Rank = struct {
    m: md.Model,
    w: views.Weights,
    e: *E,
    space: trace.Space,

    pub fn init(gpa: std.mem.Allocator, fx: J) !*Rank {
        const r = try gpa.create(Rank);
        errdefer gpa.destroy(r);
        r.m = try md.Model.parse(gpa, get(fx, "config_json").string, int(get(fx, "world")), int(get(fx, "rank")));
        try lt.expectConfig(&r.m, get(fx, "config"));
        const mtp_layers = r.m.mtp_layers;
        try r.m.cut(int(get(fx, "layers")));
        r.m.mtp_layers = mtp_layers;
        r.w = try views.rank(gpa, &r.m);
        errdefer r.w.deinit(gpa);
        r.e = try E.create(gpa, &r.m, &r.w, int(get(fx, "capacity")), int(get(fx, "max_rows")), int(get(fx, "prefill_rows")), get(fx, "long_context").bool);
        errdefer r.e.destroy(gpa);
        try r.e.useSampling(gpa); // tmp.nucleus last but one: python's other buffers keep their names
        r.space = .{ .gpa = gpa };
        errdefer r.space.deinit();
        try views.walk(&r.w, views.Path.init("w"), &r.space, trace.placeAll);
        try r.e.walk(&r.space, trace.placeAll);
        try r.e.walkTmp(&r.space, trace.placeAll);
        r.e.tmp.space = &r.space;
        return r;
    }

    pub fn deinit(r: *Rank, gpa: std.mem.Allocator) void {
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.space.deinit();
        gpa.destroy(r);
    }

    /// A fresh case, as python's: the trace emptied, temporaries named afresh, no kda wide scratch.
    pub fn fresh(r: *Rank, t: *trace.Trace) void {
        t.reset();
        r.e.tmp.reset();
        r.e.prompt.reset();
    }
};

pub const Kind = enum { first, main, draft };

/// Which rows a sample call reads: prefill's clone (the first token), the verify window's, or the MTP head's.
pub fn kindOf(e: *const E, logits: trace.T) Kind {
    if (logits.ptr == e.buf.logits.ptr) return .main;
    if (e.mbuf) |*mb| if (logits.ptr == mb.logits.ptr) return .draft;
    return .first;
}

/// The positions python passes a sample call, from the Engine's state: prefill's len(prompt), a verify window's
/// st.pos + 1 + r, draft j's st.pos + 1 + j (j = the chain's MTP steps so far, st.mtp_drafted).
pub fn position(e: *const E, kind: Kind, r: usize) usize {
    return switch (kind) {
        .first => e.st.pos,
        .main => e.st.pos + 1 + r,
        .draft => e.st.pos + 1 + e.st.mtp_drafted,
    };
}

/// The fixture's sampler calls, answered in order; each call's kind, rows and positions checked against python's.
const Script = struct {
    e: *const E,
    calls: []const J,
    first: u32,
    i: usize = 0,
    in_prefill: bool = true,

    fn pick(ctx: *anyopaque, logits: trace.T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try testing.expectEqual(s.e.sampling != null and s.e.sampling.?.temperature > 0, at != null);
        const kind = kindOf(s.e, logits);
        if (s.in_prefill) { // the prefill's own call (python asserted it: kind first, position n, the toy's truth)
            try testing.expectEqual(Kind.first, kind);
            try testing.expectEqual(@as(usize, 1), out.len);
            out[0] = s.first;
            return;
        }
        if (s.i >= s.calls.len) return error.ExtraSampleCall;
        const c = s.calls[s.i];
        errdefer std.debug.print("sample call {d} ({s})\n", .{ s.i, @tagName(kind) });
        s.i += 1;
        try testing.expectEqualStrings(get(c, "kind").string, @tagName(kind));
        const pos = get(c, "positions").array.items;
        const picks = get(c, "picks").array.items;
        try testing.expectEqual(pos.len, out.len);
        for (pos, picks, out, 0..) |p, k, *o, r| {
            try testing.expectEqual(int(p), position(s.e, kind, r));
            if (at) |x| try testing.expectEqual(@as(u64, @intCast(int(p))), x[r]); // the keyed draw's own positions
            o.* = @intCast(k.integer);
        }
        try testing.expectEqual(c.object.get("probs") != null, probs != null);
        if (probs) |ps| for (get(c, "probs").array.items, ps) |p, *x| {
            x.* = switch (p) {
                .float => |v| v,
                .integer => |v| @floatFromInt(v),
                else => return error.FixtureProb,
            };
        };
    }
};

fn policyOf(spec: []const u8) !decode.DepthPolicy {
    if (spec[0] == 'f') return .{ .most = try std.fmt.parseInt(usize, spec[1..], 10), .fixed = true };
    if (std.mem.eql(u8, spec, "a")) return .{};
    const colon = std.mem.indexOfScalar(u8, spec, ':') orelse return error.Policy;
    if (spec[0] == 'a') { // a:LOW:HIGH (engine.py's sampled auto policy: a:0.6:0.85)
        const two = std.mem.lastIndexOfScalar(u8, spec, ':').?;
        return .{ .low = try std.fmt.parseFloat(f64, spec[colon + 1 .. two]), .high = try std.fmt.parseFloat(f64, spec[two + 1 ..]) };
    }
    return .{ .most = try std.fmt.parseInt(usize, spec[1..colon], 10), .fixed = true, .confidence = try std.fmt.parseFloat(f64, spec[colon + 1 ..]) };
}

fn num(v: J) f64 {
    return switch (v) {
        .float => |x| x,
        .integer => |x| @floatFromInt(x),
        else => std.math.nan(f64),
    };
}

/// A case's exact_sampling.Sampling (top_k 0: the nucleus), or null for greedy.
fn samplingOf(c: J) ?nucleus.Sampling {
    const s = c.object.get("sampling") orelse return null;
    return .{ .seed = @intCast(int(get(s, "seed"))), .temperature = num(get(s, "temperature")), .top_k = @intCast(int(get(s, "top_k"))), .top_p = num(get(s, "top_p")), .min_p = num(get(s, "min_p")) };
}

/// qualification_45c10bcf's first-level nucleus launches for a row window (cuda_nucleus_test.zig's list; one row has no flip).
const nucleus_launches = [_][]const u8{ "torch.float", "nucleus.scale", "nucleus.max", "all_gather", "nucleus.flip", "nucleus.max", "nucleus.mass", "nucleus.sum", "torch.topk", "torch.add", "nucleus.take", "nucleus.take", "nucleus.header", "all_gather", "torch.cpu" };

/// The trace with each nucleus draw, its launches checked against qualification_45c10bcf's list, as one `sample.nucleus` mark on the
/// rows it reads: python's form of the call (glm_mtp_fixtures.nucleus_marks). Other events stay as they are.
fn markNucleus(a: std.mem.Allocator, got: []const trace.Event) ![]const trace.Event {
    var out: std.ArrayList(trace.Event) = .empty;
    var i: usize = 0;
    while (i < got.len) {
        if (!(std.mem.eql(u8, got[i].name, "torch.float") and i + 1 < got.len and std.mem.eql(u8, got[i + 1].name, "nucleus.scale"))) {
            try out.append(a, got[i]);
            i += 1;
            continue;
        }
        const logits = got[i].vals[0].t;
        for (nucleus_launches) |name| {
            if (logits.shape[0] == 1 and std.mem.eql(u8, name, "nucleus.flip")) continue;
            if (i >= got.len) return error.ShortNucleus;
            try testing.expectEqualStrings(name, got[i].name);
            i += 1;
        }
        const vals = try a.alloc(trace.Val, 1);
        vals[0] = .{ .t = logits };
        try out.append(a, .{ .op = .ext, .name = "sample.nucleus", .vals = vals });
    }
    return out.items;
}

/// python's per-round list as counts by size.
fn counts(comptime n: usize, xs: []const J) [n]usize {
    var c: [n]usize = @splat(0);
    for (xs) |x| c[int(x)] += 1;
    return c;
}

/// One decode fixture: python's Config and buffers, then each case's launches, answers, tokens, counters and state.
fn expectDecodes(text: []const u8) !usize {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, text, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-decode-trace/1", get(fx, "schema").string);
    const r = try Rank.init(gpa, fx);
    defer r.deinit(gpa);
    _ = try ct.expectBuffers(&r.space, get(fx, "buffers"));
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var n: usize = 0;
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        const c = kv.value_ptr.*;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        r.fresh(&t);
        r.e.sampling = samplingOf(c);
        defer r.e.sampling = null;
        var f = fwd.Fwd.recording(&r.m, &t);
        var script: Script = .{ .e = r.e, .calls = get(c, "calls").array.items, .first = @intCast(int(get(c, "first"))) };
        t.picker = .{ .ctx = &script, .pick = Script.pick };
        defer t.picker = null;
        var prompt: [16]u32 = undefined;
        const ps = get(c, "prompt").array.items;
        for (ps, prompt[0..ps.len]) |x, *y| y.* = @intCast(x.integer);
        const first = try r.e.prefill(&f, prompt[0..ps.len], true);
        try testing.expectEqual(script.first, first);
        t.reset(); // python logs the loop alone; the temporaries' counts go on from the prefill's
        script.in_prefill = false;
        const count = int(get(c, "count"));
        const stop_eos = get(c, "stop_eos").bool;
        var out: [32]u32 = undefined;
        const loop = get(c, "loop").string;
        var policy = if (std.mem.eql(u8, loop, "serial")) decode.DepthPolicy{} else try policyOf(get(c, "policy").string);
        const d = if (std.mem.eql(u8, loop, "serial"))
            try decode.serial(r.e, &f, first, count, &out, stop_eos)
        else if (std.mem.eql(u8, loop, "mtp"))
            try r.e.mtpDecode(&f, first, count, &out, &policy, stop_eos)
        else
            try r.e.autoDecode(&f, first, count, &out, &policy, stop_eos);
        try f.flush();
        try testing.expectEqual(script.calls.len, script.i);
        const want = get(c, "tokens").array.items;
        try testing.expectEqual(want.len, d.tokens.len);
        for (want, d.tokens) |x, y| try testing.expectEqual(int(x), y);
        try testing.expectEqual(int(get(c, "rounds")), d.rounds);
        try testing.expectEqual(int(get(c, "drafted")), d.drafted);
        try testing.expectEqual(int(get(c, "accepted")), d.accepted);
        if (!std.mem.eql(u8, loop, "serial")) {
            try testing.expectEqual(counts(decode.max_rows, get(c, "depths").array.items), d.depths);
            try testing.expectEqual(counts(decode.max_rows + 1, get(c, "keeps").array.items), d.keeps);
        }
        try testing.expectEqual(int(get(c, "pos")), r.e.st.pos);
        try testing.expectEqual(int(get(c, "mtp_len")), r.e.st.mtp_len);
        try testing.expectEqual(int(get(c, "mtp_drafted")), r.e.st.mtp_drafted);
        try testing.expectEqual(int(get(c, "parity")), r.e.st.parity);
        try testing.expectEqual([3]u64{ 0, 0, 0 }, d.gathers); // a recording draws nothing, so counts no gathers
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const got = if (r.e.sampling != null) try markNucleus(arena.allocator(), t.events.items) else t.events.items;
        try lt.expectEvents(&r.space, get(c, "events").array.items, got);
        n += 1;
    }
    return n;
}

test "MLX 4-bit: python-0.6's serial, MTP-drafted and auto decode loops on a TP2 rank, launch for launch" {
    try testing.expectEqual(@as(usize, 12), try expectDecodes(fixture_mlx));
}

test "EXL3 (BF16 dense, 4-bit draft head): mtp_decode and auto_decode's MTP arm are python-0.6's launches" {
    try testing.expectEqual(@as(usize, 2), try expectDecodes(fixture_exl3));
}

test "MLX 4-bit, sampled (seeded nucleus, top_k 0): every loop's launches, each draw qualification_45c10bcf's at python's positions" {
    try testing.expectEqual(@as(usize, 8), try expectDecodes(fixture_sampled));
}

test "EXL3, sampled: mtp_decode and auto_decode under engine.py's sampled policy (a 0.6/0.85)" {
    try testing.expectEqual(@as(usize, 2), try expectDecodes(fixture_sampled_exl3));
}

test "a greedy draft's probability (decode._probability): python's float64 bits over 1-3 ranks' candidates" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture_probs, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-draft-probability/1", get(fx, "schema").string);
    var rows: usize = 0;
    for (get(fx, "cases").array.items, 0..) |c, i| {
        errdefer std.debug.print("case {d}\n", .{i});
        const cut: sample.Cut = if (c.object.get("top_p")) |p| .{ .top_p = num(p), .top_k = @intCast(int(get(c, "top_k"))) } else .{};
        const vs = get(c, "values").array.items;
        for (vs, get(c, "tokens").array.items, get(c, "chosen").array.items, get(c, "probs").array.items) |vrow, trow, ch, p| {
            var values: [128]f32 = undefined;
            var tokens: [128]i64 = undefined;
            const n = vrow.array.items.len;
            for (vrow.array.items, trow.array.items, 0..) |v, tk, j| {
                values[j] = @bitCast(@as(u32, @intCast(v.integer)));
                tokens[j] = tk.integer;
            }
            const chosen = sample.choose(values[0..n], tokens[0..n]);
            try testing.expectEqual(int(ch), chosen);
            const got: u64 = @bitCast(sample.probabilityCut(values[0..n], tokens[0..n], chosen, cut));
            try testing.expectEqual(@as(u64, @intCast(p.integer)), got);
            rows += 1;
        }
    }
    try testing.expectEqual(@as(usize, 286), rows); // day 1's 193, then 93 of temperature-0 requests' cuts
}

test "numpy's pairwise sum: the 8-accumulator block, the tail, and the halves past 128 values" {
    var a: [300]f64 = undefined;
    for (&a, 0..) |*x, i| x.* = 1.0 / @as(f64, @floatFromInt(i + 1)) + 1e16 * @as(f64, @floatFromInt(i % 3));
    var seq: f64 = 0.0;
    for (a[0..5]) |x| seq += x;
    try testing.expectEqual(seq, sample.pairwiseSum(a[0..5]));
    const r = a[0..8].*;
    const want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7])) + a[8];
    try testing.expectEqual(want, sample.pairwiseSum(a[0..9]));
    try testing.expectEqual(sample.pairwiseSum(a[0..144]) + sample.pairwiseSum(a[144..300]), sample.pairwiseSum(&a));
}

/// A toy model for drafted == serial: the target's token at absolute position p (an eos at `eos_at`), the head right
/// unless a hash of (seed, p) falls under `miss` per mille, its probability by another hash.
const Toy = struct {
    e: *const E,
    seed: u64,
    miss: u64,
    eos_at: ?usize,
    calls: usize = 0,

    fn hash(t: *const Toy, p: usize, salt: u64) u64 {
        var h = std.hash.Wyhash.init(t.seed ^ salt);
        h.update(std.mem.asBytes(&p));
        return h.final();
    }
    fn truth(t: *const Toy, p: usize) u32 {
        if (t.eos_at) |x| if (p == x) return t.e.m.eos[0];
        const v: u32 = @intCast(t.hash(p, 1) % 150000);
        return if (std.mem.indexOfScalar(u32, t.e.m.eos[0..t.e.m.eos_n], v) != null) v + 1 else v;
    }
    fn pick(ctx: *anyopaque, logits: trace.T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        _ = at;
        const t: *Toy = @ptrCast(@alignCast(ctx));
        t.calls += 1;
        const kind = kindOf(t.e, logits);
        for (out, 0..) |*o, r| {
            const p = position(t.e, kind, r);
            o.* = t.truth(p);
            if (kind == .draft and t.hash(p, 2) % 1000 < t.miss) o.* +%= 1 + @as(u32, @intCast(p % 5));
            if (probs) |ps| ps[r] = 0.05 + @as(f64, @floatFromInt(t.hash(p, 3) % 950)) / 1000.0;
        }
    }
};

test "drafted == serial: mtp_decode and auto_decode give serial_decode's tokens at every depth, policy and stop" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture_mlx, .{});
    defer parsed.deinit();
    const r = try Rank.init(gpa, parsed.value);
    defer r.deinit(gpa);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2 };
    const policies = [_]decode.DepthPolicy{ .{ .most = 1, .fixed = true }, .{ .most = 3, .fixed = true }, .{ .most = 7, .fixed = true }, .{}, decode.DepthPolicy.auto(true), .{ .most = 7, .fixed = true, .confidence = 0.2 }, decode.DepthPolicy.auto(false), .{ .most = 3, .fixed = true, .confidence = 0.5, .first = 0.45 }, .{ .most = 7, .fixed = true, .first = 0.6 }, .{ .most = 4, .fixed = true, .confidence = 0.3, .first = 0.9 } }; // draft_policy: the last three with the first draft's floor
    var runs: usize = 0;
    var empty: usize = 0; // draft_policy: rounds whose first draw fell under its floor (no draft, the pending token alone)
    for ([_]u64{ 11, 12, 13 }) |seed| for ([_]u64{ 0, 300, 700, 1000 }) |miss| for ([_]usize{ 1, 2, 6, 19 }) |count| for ([_]?usize{ null, prompt.len + 4 }) |eos_at| {
        var toy: Toy = .{ .e = r.e, .seed = seed, .miss = miss, .eos_at = eos_at };
        t.picker = .{ .ctx = &toy, .pick = Toy.pick };
        defer t.picker = null;
        const stop = eos_at != null;
        var serial_out: [32]u32 = undefined;
        r.fresh(&t);
        var f = fwd.Fwd.recording(&r.m, &t);
        const first = try r.e.prefill(&f, &prompt, true);
        const s = try decode.serial(r.e, &f, first, count, &serial_out, stop);
        for (s.tokens, 0..) |x, i| try testing.expectEqual(toy.truth(prompt.len + i), x);
        const pos_serial = r.e.st.pos;
        var again: [32]u32 = undefined; // the Engine's serialDecode (its callers read the tokens) makes the same reply
        r.fresh(&t);
        f = fwd.Fwd.recording(&r.m, &t);
        _ = try r.e.prefill(&f, &prompt, true);
        try testing.expectEqualSlices(u32, s.tokens, try r.e.serialDecode(&f, first, count, &again, stop));
        try testing.expectEqual(pos_serial, r.e.st.pos);
        for (policies) |p0| for ([_]bool{ false, true }) |auto| {
            errdefer std.debug.print("seed {d} miss {d} count {d} eos {?d} auto {} policy {any}\n", .{ seed, miss, count, eos_at, auto, p0 });
            var policy = p0;
            var out: [32]u32 = undefined;
            r.fresh(&t);
            f = fwd.Fwd.recording(&r.m, &t);
            _ = try r.e.prefill(&f, &prompt, true);
            const d = if (auto) try r.e.autoDecode(&f, first, count, &out, &policy, stop) else try r.e.mtpDecode(&f, first, count, &out, &policy, stop);
            try testing.expectEqualSlices(u32, s.tokens, d.tokens);
            try testing.expect(d.accepted <= d.drafted and d.rounds <= s.rounds);
            try testing.expectEqual(d.tokens.len - 1, @min(d.accepted + d.rounds, count - 1));
            try testing.expect(r.e.st.pos >= pos_serial); // a round past the reply's end commits its rows too
            if (miss == 0 and eos_at == null) try testing.expectEqual(d.drafted, d.accepted); // every draft right: all kept
            if (p0.first > 0) empty += d.depths[0];
            runs += 1;
        };
    };
    try testing.expectEqual(@as(usize, 3 * 4 * 4 * 2 * 10 * 2), runs);
    try testing.expect(empty > 0); // the floor's empty rounds were taken, and kept the serial tokens
}
