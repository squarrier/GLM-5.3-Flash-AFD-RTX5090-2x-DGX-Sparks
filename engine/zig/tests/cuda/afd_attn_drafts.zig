//! tf-glm-afd attn's MTP drafts (qualification_21e21bc0; TF_AFD_MTP=1 on the attention and on both halves, unset: nothing changes).
//! `model`: the attention's model at world 1 cut to --layers as python's AFD cuts it (no MTP, engine.py:499-502), or
//! with the switch the checkpoint's MTP layer kept through the cut (tf-glm-forward's --layers keeps it: config.subset).
//! `bench`: attn bench's `drafts=SPEC` (glm_drafts.zig's forms: N or fN, cN:P, a or a:LOW:HIGH, auto; 0 is the plain
//! bench). Once, the decode prompt drafts off (Attn.step, the head absorbing nothing): the reference. Then each run:
//! the decode prompt's prefill with the head absorbing it, the policy's drafted decode of STEPS + 1 tokens
//! (cuda_afd_drafts: verify windows of k + 1 <= 8 rows over the exchange's decode rows), the prefill prompt.
//! G-draft-equal (qualification_1bae3bb5's): every run's tokens equal the reference's; a run that differs fails the command.
const std = @import("std");
const glm = @import("glm_cuda");
const attn = @import("afd_attn.zig");
const ab = @import("afd_attn_bench.zig");
const gd = @import("glm_drafts.zig");

const ad = glm.afd_drafts;
const Decoded = glm.decode.Decoded;
const Allocator = std.mem.Allocator;

/// The attention node's model (config.json's text) at world 1, cut to `layers`; the MTP layer only with TF_AFD_MTP=1.
pub fn model(gpa: Allocator, config: []const u8, layers: ?u32) !glm.model.Model {
    var m = try glm.model.Model.parse(gpa, config, 1, 0);
    const keep = if (try glm.afd_mtp.fromEnv()) m.mtp_layers else 0;
    _ = try glm.pack.mtp8.checkEnv(keep > 0); // mtp_fp8: TF_GLM_MTP_FP8 refused here, before anything loads, unless sound
    if (layers) |n| try m.cut(n);
    m.mtp_layers = keep;
    return m;
}

/// The INFO line of a TF_AFD_MTP=1 load (nothing without the switch: the outputs as they were).
pub fn info(a: *const glm.afd_attn.Afd, what: []const u8) void {
    if (a.mtp) std.debug.print("INFO attn {s}: TF_AFD_MTP=1: the MTP layer loaded; its routed experts the halves' position {d} (every forward a slice of {d})\n", .{ what, a.layers, a.positions() });
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// The policy's loop over the attention node (bench's fixed length: no eos stop).
fn decodeWith(r: attn.Run, d: gd.Drafts, pending: u32, count: usize, out: []u32, log: ?*Decoded.Log) !Decoded {
    var policy = d.policy;
    return switch (d.kind) {
        .serial => error.DraftsOff,
        .mtp => ad.mtp(r.at, r.f, pending, count, out, &policy, false, log),
        .auto => blk: {
            policy = glm.decode.DepthPolicy.auto(r.at.e.sampling == null or r.at.e.sampling.?.temperature <= 0.0);
            break :blk ad.auto(r.at, r.f, pending, count, out, &policy, false, log);
        },
    };
}

/// glm_drafts.timeLine's line for the attention node, with the run's G-draft-equal.
fn timeLine(ri: usize, label: []const u8, d: gd.Drafts, x: *const Decoded, from: usize, same: bool) void {
    const wall = ms(x.ns);
    const n = x.tokens.len -| 1;
    std.debug.print("TIME attn: bench run {d} ({s}): decode of {d} tokens from position {d}, drafts {s}: {d:.1} ms, {d:.2} tok/s; rounds {d}, drafted {d}, accepted {d}, acceptance {d:.4}, {d:.3} tokens a verify forward, MTP steps {d}; depths {any}; keeps {any}; stages: draft {d:.1} ms, forward {d:.1} ms, sample {d:.1} ms, commit {d:.1} ms; G-draft-equal {s}\n", .{ ri, label, n, from, d.spec, wall, @as(f64, @floatFromInt(n)) * 1e3 / wall, x.rounds, x.drafted, x.accepted, x.acceptance(), x.perForward(), x.mtp_steps, x.depths, x.keeps, ms(x.stages[0]), ms(x.stages[1]), ms(x.stages[2]), ms(x.stages[3]), if (same) "equal" else "DIFFERS" });
}

/// drafts=SPEC's policies: glm_drafts.zig's forms joined by `+` (f3+auto+c7:0.35: one load, one reference), each on.
pub fn policies(s: []const u8, out: []gd.Drafts) ![]gd.Drafts {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, s, '+');
    while (it.next()) |p| {
        if (n == out.len) return error.Drafts;
        out[n] = try gd.parse(p);
        if (!out[n].on()) return if (n == 0 and it.peek() == null) out[0..0] else error.Drafts; // `0` alone: the plain bench
        n += 1;
    }
    return out[0..n];
}

/// The second prompt's gate: its first token and 48 greedy steps (qualification_1bae3bb5's 8-layer G-draft-equal length).
pub const gate = 48;

/// attn bench with drafts on (`spec.drafts`' policies) over --prompt FILE[:NAME]'s prompts (`text`: the spec as given):
/// the reference drafts off on both prompts (STEPS on the decode prompt, `gate` on the prefill prompt), then for each
/// policy its runs, each drafting both. The runs made, or error.DraftNotEqual (once OUT/bench-attn.json is written)
/// when a run's tokens differ from the reference's on either prompt.
pub fn bench(r: attn.Run, spec: ab.Spec, text: []const u8, prompt_flag: []const u8, load_ms: f64) !usize {
    const gpa = r.gpa;
    var pbuf: [8]gd.Drafts = undefined; // draft_policy: up to 8 forms a load (the draft-stop sweep)
    const ds = try policies(spec.drafts, &pbuf);
    if (ds.len == 0) return error.DraftsOff;
    if (!r.at.a.mtp or r.at.e.mbuf == null) return error.DraftsNeedTfAfdMtp; // the halves serve the MTP layer only then
    const dp = try ab.promptIds(gpa, prompt_flag, spec.decode);
    defer gpa.free(dp);
    const pp = try ab.promptIds(gpa, prompt_flag, spec.prefill);
    defer gpa.free(pp);
    const cap = r.at.e.st.capacity;
    const w = glm.decode.max_rows + 1; // a verify window past the last token
    if (dp.len + spec.steps + w > cap or pp.len + gate + w > cap) return error.BenchPastCapacity;
    const out = std.debug.print;
    var js: std.ArrayList(u8) = .empty;
    defer js.deinit(gpa);
    try js.print(gpa, "{{\"role\": \"attn\", \"world\": 1, \"layers\": {d}, \"moe_layers\": {d}, \"mtp\": true, \"capacity\": {d}, \"prefill_rows\": {d}, \"dry\": {}, \"spec\": \"{s}\", \"load_ms\": {d:.1}", .{ r.at.e.m.layers, r.at.a.layers, cap, r.at.e.prefill_rows, r.stream == null, text, load_ms });
    const n = spec.steps + 1;
    const ref = try gpa.alloc(u32, n);
    defer gpa.free(ref);
    const toks = try gpa.alloc(u32, n);
    defer gpa.free(toks);
    var gref: [gate + 1]u32 = undefined;
    var gtoks: [gate + 1]u32 = undefined;
    r.at.absorb = false; // the reference: drafts off, as the plain bench decodes
    var t0 = nowNs();
    ref[0] = try r.at.prefill(r.f, dp);
    const pre0 = try ab.since(r, t0);
    t0 = nowNs();
    for (0..spec.steps) |k| ref[k + 1] = try r.at.step(r.f, ref[k]);
    const wall0 = try ab.since(r, t0);
    gref[0] = try r.at.prefill(r.f, pp);
    for (0..gate) |k| gref[k + 1] = try r.at.step(r.f, gref[k]);
    out("TIME attn: bench reference (drafts off): prefill of {d} tokens ({s}): {d:.3} ms; decode of {d} steps from position {d}: {d:.1} ms, {d:.2} tok/s; first token {d}; {s}: first token {d}, {d} steps\n", .{ dp.len, spec.decode, pre0, spec.steps, dp.len, wall0, @as(f64, @floatFromInt(spec.steps)) * 1e3 / wall0, ref[0], spec.prefill, gref[0], gate });
    try js.print(gpa, ", \"reference\": {{\"prefill_ms\": {d:.3}, \"wall_ms\": {d:.3}, \"out\": ", .{ pre0, wall0 });
    try ab.idsJson(&js, gpa, ref);
    try js.appendSlice(gpa, ", \"gate_out\": ");
    try ab.idsJson(&js, gpa, &gref);
    try js.appendSlice(gpa, "}, \"policies\": [");
    var equal = true;
    var lrows: [8192]u32 = undefined;
    var lat: [4096]usize = undefined;
    var lprobs: [8192]f64 = undefined; // draft_policy: each draft's probability beside it (round_probs)
    for (ds, 0..) |d, di| {
        var eq = true;
        try js.print(gpa, "{s}\n{{\"drafts\": \"{s}\", \"runs\": [", .{ if (di == 0) "" else ",", d.spec });
        for (0..spec.warmup + spec.runs) |ri| {
            const label = if (ri < spec.warmup) "warm-up" else "timed";
            r.at.absorb = true; // prefill's chunks go through the MTP head too (Engine.prefill's absorb)
            t0 = nowNs();
            const first = try r.at.prefill(r.f, dp);
            const pre_ms = try ab.since(r, t0);
            var log: Decoded.Log = .{ .drafts = &lrows, .at = &lat, .probs = &lprobs };
            const x = try decodeWith(r, d, first, n, toks, &log);
            const same = std.mem.eql(u32, ref, x.tokens);
            out("TIME attn: bench run {d} ({s}): prefill of {d} tokens ({s}), the head absorbing it: {d:.3} ms, {d:.1} tok/s; first token {d}\n", .{ ri, label, dp.len, spec.decode, pre_ms, @as(f64, @floatFromInt(dp.len)) * 1e3 / pre_ms, first });
            timeLine(ri, label, d, &x, dp.len, same);
            try js.print(gpa, "{s}\n{{\"run\": {d}, \"warmup\": {}, \"decode\": {{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"first\": {d}, \"steps\": {d}, \"equal\": {}, \"out\": ", .{ if (ri == 0) "" else ",", ri, ri < spec.warmup, spec.decode, dp.len, pre_ms, first, spec.steps, same });
            try ab.idsJson(&js, gpa, x.tokens);
            try gd.record(&js, gpa, gd.After.of(r.at.e), d, &x);
            t0 = nowNs();
            const first2 = try r.at.prefill(r.f, pp);
            const pre2 = try ab.since(r, t0);
            const y = try decodeWith(r, d, first2, gate + 1, &gtoks, null);
            const same2 = std.mem.eql(u32, &gref, y.tokens);
            out("TIME attn: bench run {d} ({s}): prefill of {d} tokens ({s}): {d:.3} ms, {d:.1} tok/s; first token {d}; its {d} drafted tokens (rounds {d}, accepted {d} of {d}) G-draft-equal {s}\n", .{ ri, label, pp.len, spec.prefill, pre2, @as(f64, @floatFromInt(pp.len)) * 1e3 / pre2, first2, gate + 1, y.rounds, y.accepted, y.drafted, if (same2) "equal" else "DIFFERS" });
            try js.print(gpa, "}}, \"prefill\": {{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"first\": {d}, \"gate_equal\": {}, \"gate_out\": ", .{ spec.prefill, pp.len, pre2, first2, same2 });
            try ab.idsJson(&js, gpa, y.tokens);
            try js.appendSlice(gpa, "}}");
            eq = eq and same and same2;
        }
        out("BENCH attn: drafts {s}: {d} runs; G-draft-equal {s}: every run's {d} tokens of {s} and {d} of {s} {s} the reference's (drafts off)\n", .{ d.spec, spec.warmup + spec.runs, if (eq) "PASS" else "FAIL", n, spec.decode, gate + 1, spec.prefill, if (eq) "equal" else "do not all equal" });
        try js.print(gpa, "], \"draft_equal\": {}}}", .{eq});
        equal = equal and eq;
    }
    out("BENCH attn: {d} drafted polic{s} against the reference; G-draft-equal {s}\n", .{ ds.len, if (ds.len == 1) "y" else "ies", if (equal) "PASS" else "FAIL" });
    try js.print(gpa, "\n], \"draft_equal\": {}}}\n", .{equal});
    var pb: [1024]u8 = undefined;
    try attn.save(try std.fmt.bufPrint(&pb, "{s}/bench-attn.json", .{r.out}), js.items);
    if (!equal) return error.DraftNotEqual;
    return ds.len * (spec.warmup + spec.runs);
}

test "TF_AFD_MTP's model: the MTP layer kept through --layers with the switch, dropped without (python's AFD)" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config_json").?.string;
    const m = try model(gpa, config, 5);
    try std.testing.expectEqual(@as(usize, 5), m.layers);
    try std.testing.expectEqual(@as(usize, 0), m.mtp_layers); // TF_AFD_MTP unset in the tests
    const whole = try model(gpa, config, null);
    try std.testing.expectEqual(@as(usize, 0), whole.mtp_layers);
    var full = try glm.model.Model.parse(gpa, config, 1, 0);
    try std.testing.expectEqual(@as(usize, 1), full.mtp_layers); // what the switch keeps
    try full.cut(5);
    try std.testing.expectEqual(@as(usize, 0), full.mtp_layers);
    const d = try gd.parse("auto");
    try std.testing.expect(d.on());
    const s = try ab.parseSpec("decode=p1024:64,prefill=p1900,runs=1,warmup=0,drafts=c7:0.35");
    try std.testing.expectEqualStrings("c7:0.35", s.drafts);
    try std.testing.expectEqualStrings("0", (try ab.parseSpec(ab.default_spec)).drafts);
    var pb: [4]gd.Drafts = undefined;
    const ps = try policies("f3+auto+c7:0.35", &pb);
    try std.testing.expectEqual(@as(usize, 3), ps.len);
    try std.testing.expectEqualStrings("c7:0.35", ps[2].spec);
    try std.testing.expectEqual(@as(f64, 0.35), ps[2].policy.confidence);
    try std.testing.expectEqual(@as(usize, 0), (try policies("0", &pb)).len); // the plain bench
    try std.testing.expectError(error.Drafts, policies("auto+0", &pb));
    try std.testing.expectError(error.Drafts, policies("f1+f2+f3+f4+f5", &pb));
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "attn bench --dry with TF_AFD_MTP=1 (layers 0-4), drafts=f3+auto: one reference, two policies' runs, G-draft-equal on both prompts; unset, refused" {
    const testing = std.testing;
    const J = std.json.Value;
    const cli = @import("tf_glm_afd.zig");
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
    const spec = "decode=a:6,prefill=b,runs=1,warmup=1,skip=1,drafts=f3+auto";
    var pb: [4096]u8 = undefined;
    var t = try attn.tokens(testing.allocator, try ab.decodePrompt(&pb, pfile, spec), null);
    defer t.deinit(testing.allocator);
    const root = try std.json.parseFromSliceLeaky(J, a, @embedFile("afd_attn_fixture"), .{});
    const config = root.object.get("config_json").?.string;
    const f: cli.Flags = .{ .role = .attn, .verb = .bench, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = pfile, .bench = spec };
    const off = try attn.dry(testing.allocator, f, &t, config); // the switch unset: drafts refused
    try testing.expectEqualStrings("DraftsNeedTfAfdMtp", off.outcome.usage);
    _ = setenv("TF_AFD_MTP", "1", 1);
    defer _ = unsetenv("TF_AFD_MTP");
    const d = try attn.dry(testing.allocator, f, &t, config);
    defer testing.allocator.free(d.outcome.ok);
    try testing.expectEqualStrings("bench 4", d.outcome.ok);
    try testing.expect(d.exchanges > d.forwards and d.exchanges < 2 * d.forwards); // backbone forwards 2, MTP steps 1
    const rec = try std.json.parseFromSliceLeaky(J, a, try @import("afd_expert.zig").readFile(a, try std.fmt.allocPrint(a, "{s}/bench-attn.json", .{out})), .{});
    const ps = rec.object.get("policies").?.array.items;
    try testing.expectEqual(@as(usize, 2), ps.len);
    for (ps) |p| {
        try testing.expect(p.object.get("draft_equal").?.bool);
        try testing.expectEqual(@as(usize, 2), p.object.get("runs").?.array.items.len);
    }
    try testing.expect(rec.object.get("draft_equal").?.bool);
    try testing.expectEqual(@as(usize, 7), rec.object.get("reference").?.object.get("out").?.array.items.len);
    try testing.expectEqual(@as(usize, gate + 1), rec.object.get("reference").?.object.get("gate_out").?.array.items.len);
}

test "draft_policy: TF_GLM_DRAFT_AUTO at the policy's parse (refused when malformed); attn bench --dry with it: auto's chain and cN:P:F's recorded, round_probs beside round_drafts, G-draft-equal" {
    const testing = std.testing;
    const J = std.json.Value;
    const cli = @import("tf_glm_afd.zig");
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var pb: [8]gd.Drafts = undefined;
    _ = setenv("TF_GLM_DRAFT_AUTO", "c3:1.5", 1);
    try testing.expectError(error.BadDraftAuto, policies("f3+auto", &pb)); // before any load
    _ = setenv("TF_GLM_DRAFT_AUTO", "c2:0.55:0.5", 1);
    defer _ = unsetenv("TF_GLM_DRAFT_AUTO");
    const ps = try policies("auto+c3:0.5:0.45+f1+f2+f3+f4+f5+c7:0.35", &pb);
    try testing.expectEqual(@as(usize, 8), ps.len);
    try testing.expectEqual(@as(f64, 0.45), ps[1].policy.first);
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
    const spec = "decode=a:6,prefill=b,runs=1,warmup=0,skip=1,drafts=auto+c3:0.5:0.45";
    var nb: [4096]u8 = undefined;
    var t = try attn.tokens(testing.allocator, try ab.decodePrompt(&nb, pfile, spec), null);
    defer t.deinit(testing.allocator);
    const root = try std.json.parseFromSliceLeaky(J, a, @embedFile("afd_attn_fixture"), .{});
    const config = root.object.get("config_json").?.string;
    const f: cli.Flags = .{ .role = .attn, .verb = .bench, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = pfile, .bench = spec };
    _ = setenv("TF_AFD_MTP", "1", 1);
    defer _ = unsetenv("TF_AFD_MTP");
    const d = try attn.dry(testing.allocator, f, &t, config);
    defer testing.allocator.free(d.outcome.ok);
    try testing.expectEqualStrings("bench 2", d.outcome.ok);
    const rec = try std.json.parseFromSliceLeaky(J, a, try @import("afd_expert.zig").readFile(a, try std.fmt.allocPrint(a, "{s}/bench-attn.json", .{out})), .{});
    try testing.expect(rec.object.get("draft_equal").?.bool);
    const want = [2][3]f64{ .{ 2, 0.55, 0.5 }, .{ 3, 0.5, 0.45 } };
    for (rec.object.get("policies").?.array.items, want) |p, w| {
        try testing.expect(p.object.get("draft_equal").?.bool);
        const dr = p.object.get("runs").?.array.items[0].object.get("decode").?.object.get("drafts").?.object;
        const ch = dr.get("chain").?.object;
        try testing.expectEqual(w, [3]f64{ @floatFromInt(ch.get("most").?.integer), ch.get("confidence").?.float, ch.get("first").?.float });
        const rd = dr.get("round_drafts").?.array.items;
        const rp = dr.get("round_probs").?.array.items;
        try testing.expectEqual(rd.len, rp.len);
        for (rd, rp) |x, y| try testing.expectEqual(x.array.items.len, y.array.items.len);
        try testing.expectEqual(@as(i64, @intCast(rd.len)), dr.get("depths").?.array.items[0].integer); // the recording's draws at 0: all empty
    }
}
