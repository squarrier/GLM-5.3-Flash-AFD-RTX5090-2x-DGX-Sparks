//! tf-glm-afd attn bench (qualification_e5a7fc3b): qualification_e84ee1bf's timed test (glm_bench.zig's form) at world 1 over the exchange, after one load.
//! --bench decode=NAME:STEPS,prefill=NAME,runs=R,warmup=W,skip=S,profile=P over --prompt FILE[:NAME]'s `prompts`
//! (default qualification_e84ee1bf's: decode=p1024:512,prefill=p1900,runs=3,warmup=1,skip=16). Each run: the decode prompt's prefill and
//! greedy first token, STEPS greedy steps (Attn.step: begin, the forward, the sync and the check, greedy, commit) each
//! timed to its stream drained, then the prefill prompt's prefill. TIME lines as glm_bench's (`TIME attn: ...`).
//! profile=P: P more steps with each MoE layer marked at the request's post, the shared expert and the replies taken
//! (Afd.mark, the stream drained at each: shares, not a speed). OUT/bench-attn.json holds every number.
const std = @import("std");
const glm = @import("glm_cuda");
const attn = @import("afd_attn.zig");
const drafts = @import("glm_drafts.zig");
/// qualification_21e21bc0: drafts=SPEC's runs (TF_AFD_MTP=1)
const drafted = @import("afd_attn_drafts.zig");

const Allocator = std.mem.Allocator;

pub const Spec = struct {
    decode: []const u8 = "p1024",
    steps: usize = 512,
    prefill: []const u8 = "p1900",
    runs: usize = 3,
    warmup: usize = 1,
    /// the first step (1-based) a decode's median and p90 count (qualification_e84ee1bf: steps 16-512)
    skip: usize = 16,
    profile: usize = 0,
    /// qualification_21e21bc0: drafts=SPEC (glm_drafts.zig's forms; 0: off), with TF_AFD_MTP=1: afd_attn_drafts.zig's runs
    drafts: []const u8 = "0",
};

pub const default_spec = "decode=p1024:512,prefill=p1900,runs=3,warmup=1,skip=16";

pub fn parseSpec(s: []const u8) !Spec {
    var p: Spec = .{};
    var it = std.mem.tokenizeScalar(u8, s, ',');
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse return error.BenchSpec;
        const k = kv[0..eq];
        const v = kv[eq + 1 ..];
        const int = std.fmt.parseInt;
        if (std.mem.eql(u8, k, "decode")) {
            const c = std.mem.lastIndexOfScalar(u8, v, ':');
            p.decode = if (c) |i| v[0..i] else v;
            if (c) |i| p.steps = int(usize, v[i + 1 ..], 10) catch return error.BenchSpec;
        } else if (std.mem.eql(u8, k, "prefill")) p.prefill = v else if (std.mem.eql(u8, k, "runs")) p.runs = try int(usize, v, 10) else if (std.mem.eql(u8, k, "warmup")) p.warmup = try int(usize, v, 10) else if (std.mem.eql(u8, k, "skip")) p.skip = try int(usize, v, 10) else if (std.mem.eql(u8, k, "profile")) p.profile = try int(usize, v, 10) else if (std.mem.eql(u8, k, "drafts")) p.drafts = v else return error.BenchSpec;
    }
    if (p.steps == 0 or p.runs + p.warmup == 0 or p.decode.len == 0 or p.prefill.len == 0) return error.BenchSpec;
    return p;
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Milliseconds since `t0` once the stream has run what was launched (recorded: the CPU's recording time).
pub fn since(r: attn.Run, t0: u64) !f64 {
    if (r.stream) |s| try s.synchronize();
    return @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
}

/// The median and the p90 (nearest rank) of `xs`, sorted in place (glm_bench's).
pub fn stats(xs: []f64) [2]f64 {
    if (xs.len == 0) return .{ 0, 0 };
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const mid = if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
    const k = @min(xs.len - 1, (xs.len * 9 + 9) / 10 - 1);
    return .{ mid, xs[k] };
}

/// The file's named prompt (FILE[:NAME]'s FILE, `prompts`.NAME).
pub fn promptIds(gpa: Allocator, prompt_flag: []const u8, name: []const u8) ![]u32 {
    var pb: [4096]u8 = undefined;
    const t = try attn.tokens(gpa, try named(&pb, prompt_flag, name), null);
    return t.prompt;
}

fn named(buf: []u8, prompt_flag: []const u8, name: []const u8) ![]const u8 {
    const colon = std.mem.lastIndexOfScalar(u8, prompt_flag, ':');
    return std.fmt.bufPrint(buf, "{s}:{s}", .{ if (colon) |c| prompt_flag[0..c] else prompt_flag, name });
}

/// --prompt FILE[:NAME] as the decode prompt's FILE:NAME (the spec's, else qualification_e84ee1bf's p1024), for the run's Tokens.
pub fn decodePrompt(buf: []u8, prompt_flag: []const u8, bench: ?[]const u8) ![]const u8 {
    return named(buf, prompt_flag, (try parseSpec(bench orelse default_spec)).decode);
}

/// The profile's marks: the stream drained at each, the MoE layers' parts summed (post to sent: the request's post
/// and the shared expert; sent to taken: the wait for both halves' replies).
pub const Split = struct {
    r: attn.Run,
    t: [3]u64 = @splat(0),
    shared_ns: u64 = 0,
    wait_ns: u64 = 0,
    layers: usize = 0,

    pub fn at(ctx: *anyopaque, what: glm.afd_attn.At) anyerror!void {
        const s: *Split = @ptrCast(@alignCast(ctx));
        if (s.r.stream) |st| try st.synchronize();
        const now = nowNs();
        s.t[@backingInt(what)] = now;
        switch (what) {
            .post => {},
            .sent => s.shared_ns += now - s.t[0],
            .taken => {
                s.wait_ns += now - s.t[1];
                s.layers += 1;
            },
        }
    }
};

pub fn idsJson(js: *std.ArrayList(u8), gpa: Allocator, xs: []const u32) !void {
    try js.append(gpa, '[');
    for (xs, 0..) |x, i| try js.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", x });
    try js.append(gpa, ']');
}

fn msJson(js: *std.ArrayList(u8), gpa: Allocator, xs: []const f64) !void {
    try js.append(gpa, '[');
    for (xs, 0..) |x, i| try js.print(gpa, "{s}{d:.3}", .{ if (i == 0) "" else ", ", x });
    try js.append(gpa, ']');
}

/// The timed test over `r` (spec from --bench, else qualification_e84ee1bf's); `load_ms`: the bring-up's, for the record. Returns the
/// runs made.
pub fn run(r: attn.Run, bench: ?[]const u8, prompt_flag: []const u8, load_ms: f64) !usize {
    const gpa = r.gpa;
    const spec = try parseSpec(bench orelse default_spec);
    var pol: [8]drafts.Drafts = undefined; // qualification_21e21bc0: drafts on (TF_AFD_MTP=1): the drafted runs and G-draft-equal instead (draft_policy: 8 forms)
    if ((try drafted.policies(spec.drafts, &pol)).len > 0) return drafted.bench(r, spec, bench orelse default_spec, prompt_flag, load_ms);
    const dp = try promptIds(gpa, prompt_flag, spec.decode);
    defer gpa.free(dp);
    const pp = try promptIds(gpa, prompt_flag, spec.prefill);
    defer gpa.free(pp);
    const cap = r.at.e.st.capacity;
    if (dp.len + spec.steps + spec.profile + 1 > cap or pp.len > cap) return error.BenchPastCapacity;
    const out = std.debug.print;
    const dry = r.stream == null;
    var js: std.ArrayList(u8) = .empty;
    defer js.deinit(gpa);
    try js.print(gpa, "{{\"role\": \"attn\", \"world\": 1, \"layers\": {d}, \"moe_layers\": {d}, \"capacity\": {d}, \"prefill_rows\": {d}, \"dry\": {}, \"spec\": \"{s}\", \"load_ms\": {d:.1}, \"runs\": [", .{ r.at.e.m.layers, r.at.a.layers, cap, r.at.e.prefill_rows, dry, bench orelse default_spec, load_ms });
    const toks = try gpa.alloc(u32, spec.steps + 1);
    defer gpa.free(toks);
    const ref = try gpa.alloc(u32, spec.steps + 1);
    defer gpa.free(ref);
    const ms = try gpa.alloc(f64, spec.steps);
    defer gpa.free(ms);
    const sorted = try gpa.alloc(f64, spec.steps);
    defer gpa.free(sorted);
    var same = true;
    var first_p: u32 = 0;
    for (0..spec.warmup + spec.runs) |ri| {
        const label = if (ri < spec.warmup) "warm-up" else "timed";
        var t0 = nowNs();
        toks[0] = try r.at.prefill(r.f, dp);
        const pre_ms = try since(r, t0);
        out("TIME attn: bench run {d} ({s}): prefill of {d} tokens ({s}): {d:.3} ms, {d:.1} tok/s; first token {d}\n", .{ ri, label, dp.len, spec.decode, pre_ms, @as(f64, @floatFromInt(dp.len)) * 1e3 / pre_ms, toks[0] });
        const wall0 = nowNs();
        for (0..spec.steps) |k| {
            t0 = nowNs();
            toks[k + 1] = try r.at.step(r.f, toks[k]);
            ms[k] = try since(r, t0);
        }
        const wall = @as(f64, @floatFromInt(nowNs() - wall0)) / 1e6;
        const from = @min(spec.skip -| 1, spec.steps - 1);
        @memcpy(sorted[0 .. spec.steps - from], ms[from..spec.steps]);
        const st = stats(sorted[0 .. spec.steps - from]);
        out("TIME attn: bench run {d} ({s}): decode of {d} steps from position {d}: median {d:.3} ms, p90 {d:.3} ms (steps {d}-{d}), {d:.2} tok/s; all steps {d:.1} ms, {d:.2} tok/s\n", .{ ri, label, spec.steps, dp.len, st[0], st[1], from + 1, spec.steps, 1e3 / st[0], wall, @as(f64, @floatFromInt(spec.steps)) * 1e3 / wall });
        t0 = nowNs();
        const first2 = try r.at.prefill(r.f, pp);
        const pre2 = try since(r, t0);
        out("TIME attn: bench run {d} ({s}): prefill of {d} tokens ({s}): {d:.3} ms, {d:.1} tok/s; first token {d}\n", .{ ri, label, pp.len, spec.prefill, pre2, @as(f64, @floatFromInt(pp.len)) * 1e3 / pre2, first2 });
        if (ri == 0) {
            @memcpy(ref, toks);
            first_p = first2;
        } else if (!std.mem.eql(u32, ref, toks) or first_p != first2) same = false;
        try js.print(gpa, "{s}\n{{\"run\": {d}, \"warmup\": {}, \"decode\": {{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"first\": {d}, \"steps\": {d}, \"median_ms\": {d:.3}, \"p90_ms\": {d:.3}, \"from_step\": {d}, \"wall_ms\": {d:.3}, \"out\": ", .{ if (ri == 0) "" else ",", ri, ri < spec.warmup, spec.decode, dp.len, pre_ms, toks[0], spec.steps, st[0], st[1], from + 1, wall });
        try idsJson(&js, gpa, toks);
        try js.appendSlice(gpa, ", \"step_ms\": ");
        try msJson(&js, gpa, ms);
        try js.print(gpa, "}}, \"prefill\": {{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"first\": {d}}}}}", .{ spec.prefill, pp.len, pre2, first2 });
    }
    out("BENCH attn: {d} runs; the timed runs' {d} tokens and {s}'s first token {s} the warm-up's\n", .{ spec.warmup + spec.runs, spec.steps + 1, spec.prefill, if (same) "equal" else "DIFFER from" });
    try js.print(gpa, "\n], \"runs_equal\": {}", .{same});
    if (spec.profile > 0) try profile(r, dp, spec, &js);
    try js.appendSlice(gpa, "}\n");
    var pb: [1024]u8 = undefined;
    const path = try std.fmt.bufPrint(&pb, "{s}/bench-attn.json", .{r.out});
    try attn.save(path, js.items);
    return spec.warmup + spec.runs;
}

/// profile=P: the decode prompt again, then P greedy steps with the MoE layers marked (Split); a step's wall against
/// the MoE layers' parts. The experts' own time is in their TIME lines (TF_AFD_TIME=1 on both halves).
fn profile(r: attn.Run, dp: []const u32, spec: Spec, js: *std.ArrayList(u8)) !void {
    var s: Split = .{ .r = r };
    var tok = try r.at.prefill(r.f, dp);
    _ = try since(r, nowNs());
    r.at.a.mark = .{ .ctx = &s, .at = Split.at };
    defer r.at.a.mark = null;
    const t0 = nowNs();
    for (0..spec.profile) |_| tok = try r.at.step(r.f, tok);
    const total = try since(r, t0);
    r.at.a.mark = null;
    const n: f64 = @floatFromInt(spec.profile);
    const ms = struct {
        fn of(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e6;
        }
    }.of;
    const step = total / n;
    const wait = ms(s.wait_ns) / n;
    const shared = ms(s.shared_ns) / n;
    const per: f64 = @floatFromInt(@max(s.layers / spec.profile, 1));
    std.debug.print("TIME attn: bench profile of {d} steps (each MoE layer drained at the post, the shared expert and the replies: shares, not a speed): {d:.3} ms a step; the attention side {d:.3} ms ({d:.1} %), of which the posts and shared experts {d:.3} ms; the wait for both halves' replies {d:.3} ms ({d:.1} %) over {d} MoE layers, {d:.4} ms a layer (the exchange and the experts: the experts' own time is their TIME lines')\n", .{ spec.profile, step, step - wait, 100 * (step - wait) / step, shared, wait, 100 * wait / step, s.layers / spec.profile, wait / per });
    try js.print(r.gpa, ", \"profile\": {{\"steps\": {d}, \"step_ms\": {d:.4}, \"attention_side_ms\": {d:.4}, \"post_and_shared_ms\": {d:.4}, \"wait_ms\": {d:.4}, \"moe_layers\": {d}}}", .{ spec.profile, step, step - wait, shared, wait, s.layers / spec.profile });
}

test "bench's spec (qualification_e84ee1bf's default, the profile) and its stats" {
    const d = try parseSpec(default_spec);
    try std.testing.expectEqualStrings("p1024", d.decode);
    try std.testing.expectEqual(@as(usize, 512), d.steps);
    try std.testing.expectEqualStrings("p1900", d.prefill);
    try std.testing.expectEqual(@as(usize, 3), d.runs);
    try std.testing.expectEqual(@as(usize, 16), d.skip);
    const p = try parseSpec("decode=A:4,prefill=B,runs=1,warmup=0,skip=1,profile=2");
    try std.testing.expectEqual(@as(usize, 4), p.steps);
    try std.testing.expectEqual(@as(usize, 2), p.profile);
    try std.testing.expectError(error.BenchSpec, parseSpec("decode=A:0"));
    try std.testing.expectError(error.BenchSpec, parseSpec("nope=1"));
    var xs = [_]f64{ 5, 1, 4, 2, 3 };
    try std.testing.expectEqual([2]f64{ 3, 5 }, stats(&xs));
    var pb: [64]u8 = undefined;
    try std.testing.expectEqualStrings("p.json:p1024", try decodePrompt(&pb, "p.json", null));
    try std.testing.expectEqualStrings("p.json:A", try decodePrompt(&pb, "p.json:x", "decode=A:4"));
}

test "attn bench --dry (layers 0-4): a warm-up and a timed run of 3 steps, both prompts, the profile's marks, bench-attn.json" {
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
    const spec = "decode=a:3,prefill=b,runs=1,warmup=1,skip=1,profile=2";
    var pb: [4096]u8 = undefined;
    var t = try attn.tokens(testing.allocator, try decodePrompt(&pb, pfile, spec), null);
    defer t.deinit(testing.allocator);
    const root = try std.json.parseFromSliceLeaky(J, a, @embedFile("afd_attn_fixture"), .{});
    const f: cli.Flags = .{ .role = .attn, .verb = .bench, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = pfile, .bench = spec };
    const d = try attn.dry(testing.allocator, f, &t, root.object.get("config_json").?.string);
    defer testing.allocator.free(d.outcome.ok);
    try testing.expectEqualStrings("bench 2", d.outcome.ok);
    // two runs (a: its prefill, 3 steps; b: its prefill), then the profile's prefill and 2 steps
    try testing.expectEqual(@as(u64, 2 * (1 + 3 + 1) + 1 + 2), d.forwards);
    try testing.expectEqual(d.forwards * 2, d.exchanges); // layers 3 and 4 are MoE
    const rec = try std.json.parseFromSliceLeaky(J, a, try @import("afd_expert.zig").readFile(a, try std.fmt.allocPrint(a, "{s}/bench-attn.json", .{out})), .{});
    try testing.expectEqual(@as(usize, 2), rec.object.get("runs").?.array.items.len);
    try testing.expect(rec.object.get("runs_equal").?.bool);
    try testing.expectEqual(@as(i64, 2), rec.object.get("profile").?.object.get("moe_layers").?.integer);
}
