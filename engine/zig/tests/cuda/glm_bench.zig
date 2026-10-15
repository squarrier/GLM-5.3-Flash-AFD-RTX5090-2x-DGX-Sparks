//! tf-glm-forward's `bench` (qualification_e84ee1bf): the Zig GLM's timed runs on two ranks, greedy, one stream, drafts off (prefill
//! without the MTP head's absorb, python's mtp=use_mtp at policy 0; serial decode). One load serves them all, in order:
//! --bench's warm-up runs and then its timed runs, each a prompt's prefill and N greedy steps (decode=NAME:N) and a
//! second prompt's prefill alone (prefill=NAME); the fixed questions (ask=A+B:N, to an eos); a prefill's first token on
//! more prompts (first=A+B); a profiled decode with each block synchronized (profile=N); the decode all-gather's
//! latency (gathers=N). Spans end once the rank's stream has drained; TIME lines go to stderr, the record to
//! OUT/bench-rankR.json. The warm-up's decode keeps each step's shard top 5 (for the margins at a divergence); a
//! timed run's tokens must equal the warm-up's. On --dry (a CPU) the launches are recorded and the memory forecast
//! (`forecast`) is printed first.
//! qualification_358eae44: prefill=A+B+... (several prompts a run, each with its own TIME line, all in the record's `prefills`); busy=N
//! (glm_bench_busy.zig: the GPU-busy share of a decode step, after the record is written once); on --dry, each
//! warm-up prefill's launches a chunk and each step's (LAUNCHES lines, the record's `launches`).
//! With --drafts (qualification_qualification_44ba554e.7 day 2) the decode runs are cuda_decode's loops with MTP drafts (glm_drafts.zig),
//! whose prefills absorb into the head; a run's TIME line and record then carry the counters.
//! qualification_qualification_4ab1ea8f: --drafts A+B+... runs each form's runs in turn on the one load (form F's record in OUT/F, ':' as '-'); rows=R+...
//! (glm_bench_rows.zig) then checks R-row verify windows' logits rows against the serial steps', once, last.
const std = @import("std");
const glm = @import("glm_cuda");
const tool = @import("glm_forward.zig");
const runv = @import("glm_run.zig");
const busyhook = @import("glm_bench_busy.zig");
const count = @import("glm_bench_count.zig");
const drafts = @import("glm_drafts.zig");
const rowcheck = @import("glm_bench_rows.zig");

const Engine = glm.engine.Engine;
const Fwd = glm.fwd.Fwd;
const T = glm.trace.T;
const Buf = glm.views.Buf;
const Args = runv.Run.ArgsT;
const Tokens = runv.Run.TokensT;
const Host = runv.Run.HostT;
const J = std.json.Value;

/// What `bench` runs (--bench KEY=VALUE,...); prompts are names in --prompt FILE's `prompts`.
pub const Spec = struct {
    decode: []const u8 = "p1024",
    steps: usize = 512,
    prefill: []const u8 = "p1900",
    runs: usize = 3,
    warmup: usize = 1,
    /// the first step (1-based) a decode's median and p90 count (the card: steps 16-512)
    skip: usize = 16,
    ask: []const u8 = "",
    ask_steps: usize = 256,
    first: []const u8 = "",
    profile: usize = 0,
    gathers: usize = 0,
    /// qualification_358eae44: decode steps of the GPU-busy profile (an event pair around every launch)
    busy: usize = 0,
    /// qualification_qualification_4ab1ea8f: the R-row logits check's window sizes, R+R+... (glm_bench_rows.zig), on the decode prompt's chain
    rows: []const u8 = "",
};

pub const default_spec = "decode=p1024:512,prefill=p1900,runs=3,warmup=1";

fn nameSteps(v: []const u8, steps: *usize) []const u8 {
    const c = std.mem.lastIndexOfScalar(u8, v, ':') orelse return v;
    steps.* = std.fmt.parseInt(usize, v[c + 1 ..], 10) catch steps.*;
    return v[0..c];
}

pub fn parseSpec(s: []const u8) !Spec {
    var p: Spec = .{};
    var it = std.mem.tokenizeScalar(u8, s, ',');
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse return error.BenchSpec;
        const k = kv[0..eq];
        const v = kv[eq + 1 ..];
        const int = std.fmt.parseInt;
        if (std.mem.eql(u8, k, "decode")) p.decode = nameSteps(v, &p.steps) else if (std.mem.eql(u8, k, "prefill")) p.prefill = v else if (std.mem.eql(u8, k, "runs")) p.runs = try int(usize, v, 10) else if (std.mem.eql(u8, k, "warmup")) p.warmup = try int(usize, v, 10) else if (std.mem.eql(u8, k, "skip")) p.skip = try int(usize, v, 10) else if (std.mem.eql(u8, k, "ask")) p.ask = nameSteps(v, &p.ask_steps) else if (std.mem.eql(u8, k, "first")) p.first = v else if (std.mem.eql(u8, k, "profile")) p.profile = try int(usize, v, 10) else if (std.mem.eql(u8, k, "gathers")) p.gathers = try int(usize, v, 10) else if (std.mem.eql(u8, k, "busy")) p.busy = try int(usize, v, 10) else if (std.mem.eql(u8, k, "rows")) p.rows = v else return error.BenchSpec;
    }
    if (p.steps == 0 or p.runs + p.warmup == 0) return error.BenchSpec;
    return p;
}

/// The watchdog's deadline again (the verb is armed once; a bench outlasts TF_TP_TIMEOUT_S): before each run.
pub const Rearm = struct {
    ctx: ?*anyopaque = null,
    call: ?*const fn (*anyopaque) void = null,

    pub fn now(r: Rearm) void {
        if (r.call) |c| c(r.ctx.?);
    }
};

/// --prompt FILE's named prompts.
pub const Prompts = struct {
    parsed: std.json.Parsed(J),
    gpa: std.mem.Allocator,

    pub fn open(gpa: std.mem.Allocator, spec: []const u8) !Prompts {
        const colon = std.mem.lastIndexOfScalar(u8, spec, ':');
        const text = try tool.slurp(gpa, if (colon) |c| spec[0..c] else spec, 64 << 20);
        defer gpa.free(text);
        return .{ .parsed = try std.json.parseFromSlice(J, gpa, text, .{}), .gpa = gpa };
    }
    pub fn deinit(p: *Prompts) void {
        p.parsed.deinit();
    }
    pub fn ids(p: *Prompts, name: []const u8) ![]u32 {
        const list = (p.parsed.value.object.get("prompts") orelse return error.NoPrompts).object.get(name) orelse return error.NoPrompt;
        const out = try p.gpa.alloc(u32, list.array.items.len);
        for (list.array.items, out) |x, *y| y.* = @intCast(x.integer);
        return out;
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Milliseconds since `t0`, once the rank's stream has run what was launched (on --dry: the CPU's recording time).
fn since(f: *Fwd, t0: u64) !f64 {
    if (f.gpu) |g| try g.stream.synchronize();
    return @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
}

/// The median and the p90 (nearest rank) of `xs` (sorted in place).
fn stats(xs: []f64) [2]f64 {
    if (xs.len == 0) return .{ 0, 0 };
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const mid = if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
    const r = @min(xs.len - 1, (xs.len * 9 + 9) / 10 - 1);
    return .{ mid, xs[r] };
}

fn isEos(e: *const Engine, tok: u32) bool {
    return std.mem.indexOfScalar(u32, e.m.eos[0..e.m.eos_n], tok) != null;
}

fn idsJson(js: *std.ArrayList(u8), gpa: std.mem.Allocator, xs: []const u32) !void {
    try js.append(gpa, '[');
    for (xs, 0..) |x, i| try js.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", x });
    try js.append(gpa, ']');
}

fn msJson(js: *std.ArrayList(u8), gpa: std.mem.Allocator, xs: []const f64) !void {
    try js.append(gpa, '[');
    for (xs, 0..) |x, i| try js.print(gpa, "{s}{d:.3}", .{ if (i == 0) "" else ", ", x });
    try js.append(gpa, ']');
}

/// A decode step: the window's logits, the greedy token over both ranks, the commit (decode.serial_decode's step).
pub fn step(e: *Engine, f: *Fwd, tok: u32, out: *u32) !T {
    const one = [1]u32{tok};
    const logits = try e.forward(f, &one);
    var next: [1]u32 = undefined;
    try glm.sample.greedy(f, &e.smp, logits, &next);
    try glm.compute.commit(f, &e.st, &e.buf, 1, 1);
    out.* = next[0];
    return logits;
}

/// The profile's tap: each block's partials (or b.x) reached, the stream drained, its time to the last mark added to
/// the block's kind (attention KDA or DSA, the dense MLP or the MoE, hc_post, the stream mean and the final norm).
const Prof = struct {
    f: *Fwd,
    last: u64 = 0,
    ns: [7]u64 = @splat(0),
    const names = [7][]const u8{ "attention kda", "attention dsa", "ffn dense", "ffn moe", "hc_post", "mean+norm", "head+sample+commit" };

    fn kind(name: []const u8) usize {
        if (std.mem.endsWith(u8, name, ".kda")) return 0;
        if (std.mem.endsWith(u8, name, ".dsa")) return 1;
        if (std.mem.endsWith(u8, name, ".mlp")) return 2;
        if (std.mem.endsWith(u8, name, ".moe")) return 3;
        if (std.mem.endsWith(u8, name, ".x")) return 4;
        return 5;
    }
    fn mark(p: *Prof, k: usize) !void {
        if (p.f.gpu) |g| try g.stream.synchronize();
        const t = nowNs();
        p.ns[k] += t - p.last;
        p.last = t;
    }
    fn tensor(ctx: *anyopaque, name: []const u8, x: T, pos0: usize, rows: usize, k: ?[]const u8) anyerror!void {
        _ = .{ x, pos0, rows, k };
        const p: *Prof = @ptrCast(@alignCast(ctx));
        try p.mark(kind(name));
    }
};

/// bench: each --drafts form's runs (qualification_qualification_4ab1ea8f: A+B+... in turn, OUT/F each), then rows=R+... once (glm_bench_rows.zig).
pub fn run(e: *Engine, f: *Fwd, a: Args, t: Tokens, h: Host, rearm: Rearm, gpa: std.mem.Allocator) !usize {
    var n: usize = 0;
    var forms = drafts.Forms.of(a.drafts, a.out);
    while (try forms.next()) |x| {
        var b = a;
        b.drafts = x.form;
        b.out = x.out;
        n += try runForm(e, f, b, t, h, rearm, gpa);
    }
    const spec = try parseSpec(if (a.bench.len > 0) a.bench else default_spec);
    if (spec.rows.len > 0) rowcheck.run(e, f, a, spec, h, rearm, gpa) catch |err|
        std.debug.print("WARN rank {d}: bench rows: {t} (the records before it stand)\n", .{ a.rank, err });
    return n;
}

fn runForm(e: *Engine, f: *Fwd, a: Args, t: Tokens, h: Host, rearm: Rearm, gpa: std.mem.Allocator) !usize {
    _ = t;
    const spec = try parseSpec(if (a.bench.len > 0) a.bench else default_spec);
    var p = try Prompts.open(gpa, a.prompt);
    defer p.deinit();
    const dp = try p.ids(spec.decode);
    defer gpa.free(dp);
    var pn: std.ArrayList([]const u8) = .empty; // qualification_358eae44: prefill=A+B+...: each prompt in every run, in order
    defer pn.deinit(gpa);
    var pps: std.ArrayList([]u32) = .empty;
    defer {
        for (pps.items) |x| gpa.free(x);
        pps.deinit(gpa);
    }
    var pit = std.mem.tokenizeScalar(u8, spec.prefill, '+');
    while (pit.next()) |name| {
        try pps.append(gpa, try p.ids(name));
        try pn.append(gpa, name);
    }
    if (pps.items.len == 0) return error.BenchSpec;
    const firsts = try gpa.alloc(u32, pps.items.len);
    defer gpa.free(firsts);
    const R = a.rank;
    const dr = try drafts.parse(a.drafts);
    const out = std.debug.print;
    var js: std.ArrayList(u8) = .empty;
    defer js.deinit(gpa);
    try js.print(gpa, "{{\"rank\": {d}, \"world\": {d}, \"drafts\": \"{s}\", \"layers\": {d}, \"mtp_head\": {}, \"quant\": \"{t}\", \"capacity\": {d}, \"prefill_rows\": {d}, \"dry\": {}, \"spec\": \"{s}\", \"mem_available_gib_start\": {d:.2}, \"runs\": [", .{ R, e.m.world, dr.spec, e.m.layers, e.mbuf != null, e.m.quant, e.st.capacity, e.prefill_rows, a.dry, if (a.bench.len > 0) a.bench else default_spec, tool.memAvailableGib() });
    const toks = try gpa.alloc(u32, spec.steps + 1);
    defer gpa.free(toks);
    const ref = try gpa.alloc(u32, spec.steps + 1);
    defer gpa.free(ref);
    const ms = try gpa.alloc(f64, spec.steps);
    defer gpa.free(ms);
    const sorted = try gpa.alloc(f64, spec.steps);
    defer gpa.free(sorted);
    const counts = try gpa.alloc(usize, spec.steps); // --dry: each warm-up step's launches
    defer gpa.free(counts);
    var lj: std.ArrayList(u8) = .empty; // --dry: the record's `launches`
    defer lj.deinit(gpa);
    var same = true;
    var low_mem: f64 = 1e9;
    for (0..spec.warmup + spec.runs) |ri| {
        rearm.now();
        const warm = ri < spec.warmup;
        const label = if (warm) "warm-up" else "timed";
        const cj: ?*std.ArrayList(u8) = if (f.rec != null and ri == 0) &lj else null;
        var t0 = nowNs();
        toks[0] = try count.prefillOf(e, f, dp, dr.on(), cj, spec.decode, R, gpa);
        const pre_ms = try since(f, t0);
        out("TIME rank {d}: bench run {d} ({s}): prefill of {d} tokens ({s}): {d:.3} ms, {d:.1} tok/s; first token {d}\n", .{ R, ri, label, dp.len, spec.decode, pre_ms, @as(f64, @floatFromInt(dp.len)) * 1e3 / pre_ms, toks[0] });
        var top: std.ArrayList(u8) = .empty;
        defer top.deinit(gpa);
        const wall0 = nowNs();
        var lrows: [4 * 1024]u32 = undefined;
        var lat: [2 * 1024]usize = undefined;
        var dlog: glm.decode.Decoded.Log = .{ .drafts = &lrows, .at = &lat };
        var dry: drafts.DryLog = .{ .e = e, .gpa = gpa };
        defer dry.js.deinit(gpa);
        const dd: ?glm.decode.Decoded = if (dr.on()) blk: {
            if (a.dry and ri == 0) f.rec.?.picker = .{ .ctx = &dry, .pick = drafts.DryLog.pick };
            defer if (f.rec) |x| {
                x.picker = null;
            };
            break :blk try drafts.run(e, f, dr, toks[0], spec.steps + 1, toks, &dlog);
        } else null;
        const after = drafts.After.of(e);
        if (dd == null) for (0..spec.steps) |k| {
            const n0 = count.launches(f);
            t0 = nowNs();
            const logits = try step(e, f, toks[k], &toks[k + 1]);
            ms[k] = try since(f, t0);
            counts[k] = count.launches(f) - n0;
            if (warm) { // the shard's top 5 (python's teacher order), for the margins at a divergence
                const lg = try h.logitsRow(gpa, logits);
                defer gpa.free(lg);
                var five: [5]usize = undefined;
                glm.sample.topkOrder(lg, 5, &five);
                try top.appendSlice(gpa, if (k == 0) "[" else ", [");
                for (five, 0..) |i, j| try top.print(gpa, "{s}[{d}, {d}]", .{ if (j == 0) "" else ", ", i + e.m.rank * e.m.vl(), lg[i] });
                try top.append(gpa, ']');
            }
        };
        const wall = @as(f64, @floatFromInt(nowNs() - wall0)) / 1e6;
        const from = @min(spec.skip -| 1, spec.steps - 1); // skip: the first step (1-based) the median and p90 count
        if (dd) |*d| {
            if (d.tokens.len != spec.steps + 1) return error.DraftedLength;
            @memset(ms, wall / @as(f64, @floatFromInt(spec.steps))); // no steps: a drafted run's mean ms a token
            drafts.timeLine(R, ri, label, dr, d, dp.len);
        }
        @memcpy(sorted[0 .. spec.steps - from], ms[from..spec.steps]);
        const st = stats(sorted[0 .. spec.steps - from]);
        if (dd == null) out("TIME rank {d}: bench run {d} ({s}): decode of {d} steps from position {d}: median {d:.3} ms, p90 {d:.3} ms (steps {d}-{d}), {d:.2} tok/s; all steps {d:.1} ms, {d:.2} tok/s\n", .{ R, ri, label, spec.steps, dp.len, st[0], st[1], from + 1, spec.steps, 1e3 / st[0], wall, @as(f64, @floatFromInt(spec.steps)) * 1e3 / wall });
        if (cj) |j| if (dd == null) try count.stepCounts(j, gpa, R, counts, spec.decode, dp.len); // drafted: no steps
        if (ri == 0) @memcpy(ref, toks) else if (!std.mem.eql(u32, ref, toks)) same = false;
        var pj: std.ArrayList(u8) = .empty; // the run's prefills, in order
        defer pj.deinit(gpa);
        var pre0: f64 = 0;
        var first0: u32 = 0;
        for (pps.items, pn.items, 0..) |pp, name, i| {
            t0 = nowNs();
            const first2 = try count.prefillOf(e, f, pp, dr.on(), cj, name, R, gpa);
            const pre2 = try since(f, t0);
            out("TIME rank {d}: bench run {d} ({s}): prefill of {d} tokens ({s}): {d:.3} ms, {d:.1} tok/s; first token {d}\n", .{ R, ri, label, pp.len, name, pre2, @as(f64, @floatFromInt(pp.len)) * 1e3 / pre2, first2 });
            if (ri == 0) firsts[i] = first2 else if (firsts[i] != first2) same = false;
            if (i == 0) {
                pre0 = pre2;
                first0 = first2;
            }
            try pj.print(gpa, "{s}{{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"first\": {d}}}", .{ if (i == 0) "" else ", ", name, pp.len, pre2, first2 });
        }
        const mem = tool.memAvailableGib();
        low_mem = @min(low_mem, mem);
        try js.print(gpa, "{s}\n{{\"run\": {d}, \"warmup\": {}, \"mem_available_gib\": {d:.2}, \"decode\": {{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"first\": {d}, \"steps\": {d}, \"median_ms\": {d:.3}, \"p90_ms\": {d:.3}, \"from_step\": {d}, \"wall_ms\": {d:.3}, \"out\": ", .{ if (ri == 0) "" else ",", ri, warm, mem, spec.decode, dp.len, pre_ms, toks[0], spec.steps, st[0], st[1], from + 1, wall });
        try idsJson(&js, gpa, toks);
        try js.appendSlice(gpa, ", \"step_ms\": ");
        try msJson(&js, gpa, ms);
        if (warm and dd == null) try js.print(gpa, ", \"shard_top5\": [{s}]", .{top.items});
        if (dd) |*d| try drafts.record(&js, gpa, after, dr, d);
        if (dry.n > 0) try js.print(gpa, ", \"dry_calls\": [{s}]", .{dry.js.items});
        // `prefill`: the first prompt's (qualification_e84ee1bf's record); `prefills`: every prompt's
        try js.print(gpa, "}}, \"prefill\": {{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"first\": {d}}}, \"prefills\": [{s}]}}", .{ pn.items[0], pps.items[0].len, pre0, first0, pj.items });
    }
    out("BENCH rank {d}: {d} runs; the timed runs' {d} tokens {s} the warm-up's; lowest MemAvailable at a run's end {d:.2} GiB\n", .{ R, spec.warmup + spec.runs, spec.steps + 1, if (same) "equal" else "DIFFER from", low_mem });
    try js.print(gpa, "\n], \"runs_equal\": {}, \"ask\": [", .{same});
    try ask(e, f, &p, spec, rearm, &js, gpa, R);
    try js.appendSlice(gpa, "], \"first\": [");
    var n: usize = 0;
    var names = std.mem.tokenizeScalar(u8, spec.first, '+');
    while (names.next()) |name| : (n += 1) {
        rearm.now();
        const q = try p.ids(name);
        defer gpa.free(q);
        const t0 = nowNs();
        const tok = try e.prefill(f, q, false);
        const pms = try since(f, t0);
        out("TIME rank {d}: bench first {s}: prefill of {d} tokens: {d:.3} ms; first token {d}\n", .{ R, name, q.len, pms, tok });
        try js.print(gpa, "{s}{{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"first\": {d}}}", .{ if (n == 0) "" else ", ", name, q.len, pms, tok });
    }
    try js.appendSlice(gpa, "]");
    if (spec.profile > 0) try profile(e, f, dp, spec, rearm, &js, gpa, R);
    if (spec.gathers > 0) try gathers(e, f, spec, rearm, &js, gpa, R);
    if (lj.items.len > 0) try js.print(gpa, ", \"launches\": [{s}]", .{lj.items});
    try js.print(gpa, ", \"mem_available_gib_end\": {d:.2}", .{tool.memAvailableGib()});
    var pb: [1024]u8 = undefined;
    try mkdirs(a.out);
    const path = try std.fmt.bufPrint(&pb, "{s}/bench-rank{d}.json", .{ a.out, R });
    try writeRecord(path, &js, gpa);
    if (spec.busy > 0) { // qualification_358eae44: last, after the record is on disk; a failure here is reported, not fatal
        busyRun(e, f, dp, spec, rearm, &js, gpa, R) catch |err| {
            out("WARN rank {d}: bench busy: {t} (the record before it stands)\n", .{ R, err });
            try js.print(gpa, ", \"busy\": {{\"error\": \"{t}\"}}", .{err});
        };
        try writeRecord(path, &js, gpa);
    }
    return spec.warmup + spec.runs;
}

/// The record so far, closed, at `path` (js itself stays open for more).
fn writeRecord(path: []const u8, js: *std.ArrayList(u8), gpa: std.mem.Allocator) !void {
    try js.appendSlice(gpa, "}\n");
    defer js.items.len -= 2;
    try writeFile(path, js.items);
}

// -- qualification_358eae44: the GPU-busy profile (--dry's launch counts are glm_bench_count.zig's) ----------------------------------

/// busy=N (glm_bench_busy.zig): N greedy steps from the decode prompt with an event pair around each launch, copy and
/// set (no added sync); each step's GPU-busy ms (the pairs' sum) against its wall ms, which ends when the stream drains.
/// The host-bound share is the rest of the wall. On --dry the steps are recorded only.
fn busyRun(e: *Engine, f: *Fwd, dp: []const u32, spec: Spec, rearm: Rearm, js: *std.ArrayList(u8), gpa: std.mem.Allocator, R: u32) !void {
    rearm.now();
    var tok = try e.prefill(f, dp, false);
    const g = f.gpu orelse {
        for (0..spec.busy) |_| _ = try step(e, f, tok, &tok);
        try js.print(gpa, ", \"busy\": {{\"steps\": {d}, \"dry\": true}}", .{spec.busy});
        return;
    };
    const n = spec.busy;
    const xs = try gpa.alloc(f64, 4 * n);
    defer gpa.free(xs);
    const wall, const gpu, const span, const tmp = .{ xs[0..n], xs[n .. 2 * n], xs[2 * n .. 3 * n], xs[3 * n ..] };
    const ops = try gpa.alloc(usize, n);
    defer gpa.free(ops);
    var over: usize = 0;
    const hk = try gpa.create(busyhook.Hook);
    defer gpa.destroy(hk);
    hk.* = .{};
    defer hk.remove();
    try hk.install(g.stream.d);
    for (0..n) |k| {
        const t0 = nowNs();
        _ = try step(e, f, tok, &tok);
        wall[k] = try since(f, t0);
        const s = try hk.take();
        gpu[k] = s.busy_ms;
        span[k] = s.span_ms;
        ops[k] = s.ops;
        over += s.over;
    }
    hk.remove();
    const med = struct {
        fn of(src: []const f64, buf: []f64) f64 {
            @memcpy(buf, src);
            return stats(buf)[0];
        }
    }.of;
    for (tmp, gpu, wall) |*y, b, w| y.* = b / w;
    const share = stats(tmp)[0];
    std.debug.print("TIME rank {d}: bench busy of {d} steps (an event pair around every launch, copy and set; no added sync): wall median {d:.3} ms a step; GPU busy median {d:.3} ms, {d:.1} % of a step's wall (the host-bound share {d:.1} %); span median {d:.3} ms; {d} ops a step (NCCL's all-gathers not covered; {d} ops past the pairs)\n", .{ R, n, med(wall, tmp), med(gpu, tmp), 100 * share, 100 - 100 * share, med(span, tmp), ops[0], over });
    try js.print(gpa, ", \"busy\": {{\"steps\": {d}, \"share_median\": {d:.4}, \"over\": {d}, \"ops\": ", .{ n, share, over });
    try count.countsJson(js, gpa, ops);
    try js.appendSlice(gpa, ", \"wall_ms\": ");
    try msJson(js, gpa, wall);
    try js.appendSlice(gpa, ", \"busy_ms\": ");
    try msJson(js, gpa, gpu);
    try js.appendSlice(gpa, ", \"span_ms\": ");
    try msJson(js, gpa, span);
    try js.append(gpa, '}');
}

/// The fixed questions: each prompt's prefill and greedy steps to an eos or ask's count.
fn ask(e: *Engine, f: *Fwd, p: *Prompts, spec: Spec, rearm: Rearm, js: *std.ArrayList(u8), gpa: std.mem.Allocator, R: u32) !void {
    var it = std.mem.tokenizeScalar(u8, spec.ask, '+');
    var n: usize = 0;
    const toks = try gpa.alloc(u32, spec.ask_steps + 1);
    defer gpa.free(toks);
    const ms = try gpa.alloc(f64, spec.ask_steps);
    defer gpa.free(ms);
    while (it.next()) |name| : (n += 1) {
        rearm.now();
        const q = try p.ids(name);
        defer gpa.free(q);
        var t0 = nowNs();
        toks[0] = try e.prefill(f, q, false);
        const pre = try since(f, t0);
        var k: usize = 0;
        while (k < spec.ask_steps and !(f.rec == null and isEos(e, toks[k]))) : (k += 1) {
            t0 = nowNs();
            _ = try step(e, f, toks[k], &toks[k + 1]);
            ms[k] = try since(f, t0);
        }
        var sorted: [4096]f64 = undefined;
        const m = @min(k, sorted.len);
        @memcpy(sorted[0..m], ms[0..m]);
        const st = stats(sorted[0..m]);
        std.debug.print("TIME rank {d}: bench ask {s}: prefill of {d} tokens: {d:.3} ms; {d} steps, median {d:.3} ms; {d} tokens\n", .{ R, name, q.len, pre, k, st[0], k + 1 });
        try js.print(gpa, "{s}{{\"prompt\": \"{s}\", \"tokens\": {d}, \"prefill_ms\": {d:.3}, \"steps\": {d}, \"median_ms\": {d:.3}, \"out\": ", .{ if (n == 0) "" else ", ", name, q.len, pre, k, st[0] });
        try idsJson(js, gpa, toks[0 .. k + 1]);
        try js.append(gpa, '}');
    }
}

/// A decode with the profile's tap: where a step's time goes, each block synchronized (shares, not a speed).
fn profile(e: *Engine, f: *Fwd, dp: []const u32, spec: Spec, rearm: Rearm, js: *std.ArrayList(u8), gpa: std.mem.Allocator, R: u32) !void {
    rearm.now();
    var tok = try e.prefill(f, dp, false);
    var pr: Prof = .{ .f = f, .last = nowNs() };
    const t0 = nowNs();
    for (0..spec.profile) |_| {
        try pr.mark(6); // the step before's head, sample and commit (the first: the prefill's sync)
        e.tap = .{ .ctx = &pr, .tensor = Prof.tensor };
        defer e.tap = null;
        _ = try step(e, f, tok, &tok);
    }
    try pr.mark(6);
    const total = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
    const per = @as(f64, @floatFromInt(spec.profile));
    std.debug.print("TIME rank {d}: bench profile of {d} steps (each block synchronized: shares, not a speed): {d:.3} ms a step;", .{ R, spec.profile, total / per });
    try js.print(gpa, ", \"profile\": {{\"steps\": {d}, \"ms_a_step\": {d:.3}, \"blocks_ms_a_step\": {{", .{ spec.profile, total / per });
    for (Prof.names, pr.ns, 0..) |name, ns, i| {
        const v = @as(f64, @floatFromInt(ns)) / 1e6 / per;
        std.debug.print(" {s} {d:.3}", .{ name, v });
        try js.print(gpa, "{s}\"{s}\": {d:.3}", .{ if (i == 0) "" else ", ", name, v });
    }
    std.debug.print(" ms\n", .{});
    try js.appendSlice(gpa, "}}");
}

/// The decode window's all-gather alone (b.part's [1, D] fp32 a rank into b.gath), synchronized each time.
fn gathers(e: *Engine, f: *Fwd, spec: Spec, rearm: Rearm, js: *std.ArrayList(u8), gpa: std.mem.Allocator, R: u32) !void {
    rearm.now();
    const d = e.m.hidden;
    const xs = try gpa.alloc(f64, spec.gathers);
    defer gpa.free(xs);
    for (xs) |*x| {
        const t0 = nowNs();
        try f.allGather(T.of(e.buf.part, 0, &.{d}), T.of(e.buf.gath, 0, &.{e.m.world * d}));
        x.* = (try since(f, t0)) * 1e3;
    }
    const st = stats(xs);
    std.debug.print("TIME rank {d}: bench all-gather of {d} fp32 a rank (a decode block's partial): median {d:.1} us, p90 {d:.1} us over {d}\n", .{ R, d, st[0], st[1], spec.gathers });
    try js.print(gpa, ", \"gather_us\": {{\"floats_a_rank\": {d}, \"n\": {d}, \"median\": {d:.2}, \"p90\": {d:.2}}}", .{ d, spec.gathers, st[0], st[1] });
}

pub fn mkdirs(path: []const u8) !void {
    var buf: [4096]u8 = undefined;
    if (path.len + 1 > buf.len) return error.PathTooLong;
    for (path, 0..) |c, i| if (c == '/' and i > 0) {
        @memcpy(buf[0..i], path[0..i]);
        buf[i] = 0;
        _ = std.c.mkdir(buf[0..i :0], 0o755);
    };
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    _ = std.c.mkdir(buf[0..path.len :0], 0o755);
}

pub fn writeFile(path: []const u8, bytes: []const u8) !void {
    var buf: [4096]u8 = undefined;
    if (path.len + 1 > buf.len) return error.PathTooLong;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const fd = std.c.open(buf[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    if (std.c.write(fd, bytes.ptr, bytes.len) != @as(isize, @intCast(bytes.len))) return error.CannotWrite;
}

// -- the memory forecast (--dry), from the checkpoint's headers ---------------------------------------------------

fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1024.0 * 1024.0 * 1024.0);
}

/// Bytes an Arena gives `v` (a Buffers, a State, a sampler scratch): one block, each buffer at a 256-byte boundary.
fn arenaOf(v: anytype, root: []const u8) !u64 {
    var ar: glm.buffers.Arena = .{};
    try glm.views.walk(v, glm.views.Path.init(root), &ar, glm.buffers.Arena.visit);
    return ar.used;
}

const Unbound = struct {
    n: usize = 0,
    bound: usize = 0,
    names: [4][96]u8 = undefined,
    lens: [4]usize = @splat(0),

    fn visit(u: *Unbound, path: []const u8, b: *Buf) anyerror!void {
        if (b.host) return;
        if (b.ptr != 0) {
            u.bound += 1;
            return;
        }
        if (u.n < u.names.len) {
            const l = @min(path.len, u.names[u.n].len);
            @memcpy(u.names[u.n][0..l], path[0..l]);
            u.lens[u.n] = l;
        }
        u.n += 1;
    }
};

/// A rank's device memory at full depth (or the cut), from the plan alone: the packed weights, the load's peak (each
/// stored group read, packed and freed as glm_forward.Run.all does), the Engine's one block (Device.attach) and the
/// tool's KDA scratch; and the views left unbound after pack.bind (stand-in addresses, as --dry places them).
pub fn forecast(r: *tool.Rank, e: *Engine, w: *glm.views.Weights, a: Args, gpa: std.mem.Allocator) !void {
    const pk = &r.pk;
    const groups = r.split.groups();
    const live = try gpa.alloc(bool, groups.len);
    defer gpa.free(live);
    @memset(live, false);
    var stored: u64 = 0;
    var homes: u64 = 0;
    var peak: u64 = 0;
    var peak_stored: u64 = 0;
    for (0..groups.len) |gi| {
        const g: u32 = @intCast(gi);
        var needed = false;
        for (pk.items.items) |it| if (it.home >= g and glm.pack.readsGroup(it, g)) {
            needed = true;
        };
        if (needed) {
            live[gi] = true;
            stored += @max(groups[gi], 256);
        }
        peak_stored = @max(peak_stored, stored);
        homes += @max(pk.homeBytes(g), 256);
        peak = @max(peak, homes + stored);
        for (live, 0..) |*l, s| if (l.* and pk.last(@intCast(s)) <= g) {
            l.* = false;
            stored -= @max(groups[s], 256);
        };
    }
    const parts = [_]u64{ try arenaOf(&e.buf, "b"), try arenaOf(&e.pbuf, "pbuf"), if (e.mbuf) |*x| try arenaOf(x, "mbuf") else 0, try arenaOf(&e.st, "st") };
    var all: glm.buffers.Arena = .{};
    try e.walk(&all, glm.buffers.Arena.visit);
    try e.walkTmp(&all, glm.buffers.Arena.visit);
    const wide: u64 = @as(u64, a.prefill_rows) * e.m.ll() * 128 * 6; // glm_run_gpu's q (fp32) and y (bf16)
    var u: Unbound = .{};
    try glm.views.walk(w, glm.views.Path.init("w"), &u, Unbound.visit);
    const out = std.debug.print;
    out("FORECAST rank {d}: {t}, {d} layers{s}: packed {d:.3} GiB ({d} buffers); after the load the device holds {d:.3} GiB (kept stored groups {d:.3}); the load's peak {d:.3} GiB with stored groups at most {d:.3} GiB at once\n", .{ a.rank, e.m.quant, e.m.layers, if (r.mtp()) " and the MTP layer" else "", gib(pk.bytes()), pk.items.items.len, gib(homes + stored), gib(stored), gib(peak), gib(peak_stored) });
    out("FORECAST rank {d}: Engine block {d:.3} GiB at capacity {d}, prefill rows {d}, long context {} (b {d:.3}, pbuf {d:.3}, mbuf {d:.3}, st {d:.3}, the sampler and temporaries {d:.3}); the tool's KDA wide scratch {d:.3} GiB\n", .{ a.rank, gib(all.used), a.capacity, a.prefill_rows, a.long_context, gib(parts[0]), gib(parts[1]), gib(parts[2]), gib(parts[3]), gib(all.used -| (parts[0] + parts[1] + parts[2] + parts[3])), gib(wide) });
    out("FORECAST rank {d}: run's device memory {d:.3} GiB (the device after the load + the Engine block + the KDA scratch); the CUDA context, modules, NCCL and comm.zig's buffers come on top (measured, not planned here)\n", .{ a.rank, gib(homes + stored + all.used + wide) });
    out("FORECAST rank {d}: views: {d} non-host buffers bound, {d} unbound", .{ a.rank, u.bound, u.n });
    for (u.names[0..@min(u.n, u.names.len)], u.lens[0..@min(u.n, u.names.len)]) |nm, l| out(" {s}", .{nm[0..l]});
    out("\n", .{});
}
