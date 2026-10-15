//! tf-glm-forward's run verbs (qualification_qualification_44ba554e.5 B3, qualification_qualification_7c75d777 PLAN §5) on one TP rank, the other rank running the same verb in lockstep
//! (glm/cuda_ranks.zig: python GlmEngine's bring-up), its weights loaded and packed as check-weights does, then
//! glm/cuda_engine.zig's Engine (python's decode.Engine, eager; greedy, or --temperature's nucleus draw):
//!   first    the prompt's first token (decode.prefill);
//!   teacher  prefill and the MTP head's draws, then --steps teacher-forced decode steps on qualification_6607f886's tokens (its
//!            teacher/P/steps.json): each step's logits row, [1, V/world] fp32 in logits-KK.bin, its top 5 and draws;
//!   dump     the prompt's prefill with qualification_6607f886's recorder taps (glm/cuda_dump.zig): prefill/P/layers.json and its .bins.
//! --dry runs a verb on a CPU over a checkpoint's headers: the views and the Engine at stand-in addresses, every launch
//! recorded (nothing runs) and counted; a block of lane A's not in this tree ends it, which the summary says. First it
//! builds the fwd.Kernels this build's GPU run would pass the rank, and refuses one without .dsa (glm_select's image).
//! Each verb logs its prefill's and each step's time (TIME lines, stderr only; the files are the same without them).
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");
const tool = @import("glm_forward.zig");
const bench = @import("glm_bench.zig");

const Engine = glm.engine.Engine;
const T = glm.trace.T;
const Sha256 = std.crypto.hash.sha2.Sha256;
const J = std.json.Value;

pub const usage =
    \\usage: tf-glm-forward first|teacher|dump OUT --model DIR --rank R --prompt FILE[:NAME] [options]
    \\  first    the prompt's greedy first token on both ranks (rank 0 prints it; OUT/first-NAME.json)
    \\  teacher  then --steps N (48) teacher-forced steps on --teacher STEPS.json's tokens (qualification_6607f886's teacher/NAME): logits,
    \\           top 5 and MTP draws a step, as OUT/teacher/NAME/steps.json
    \\  dump     the prompt's prefill, each layer's block partials and b.x, hidden, fnormed: OUT/prefill/NAME/layers.json
    \\  bench    timed runs, greedy, drafts off (glm_bench.zig): --bench KEY=V,... over FILE's prompts (qualification_e84ee1bf;
    \\           decode=p1024:512,prefill=p1900,runs=3,warmup=1, ask=A+B:N, first=A+B, profile=N, gathers=N): OUT/bench-rankR.json
    \\  FILE: a JSON list of token ids, or qualification_6607f886's results.json (its prompts.NAME)
    \\options: --ranks 2 --master IP:PORT (rank 0's, 127.0.0.1:18890) --layers N (backbone layers; qualification_6607f886's qualification_qualification_5ab55e74: 8)
    \\         --no-mtp --capacity N (2560) --long-context (qualification_6607f886's qualification_qualification_5ab55e74, python's context 6144: --capacity 6152 --long-context)
    \\         --no-mtp-layer (python's TF_GLM_MTP=0, its drafts-off load: the MTP layer neither loaded nor run)
    \\         --prefill-rows N (2048) --triton DIR (the Triton cubins)
    \\         --fatbin FILE (glm_pack) --dry (a CPU: the verb's launches recorded over the checkpoint's headers)
    \\         --temperature T (> 0: python's nucleus draw, top_k off) --top-p P (0.95) --min-p M (0) --seed S (0)
    \\         --drafts 0|N|fN|a[:LOW:HIGH]|cN:P|auto (bench's decode runs: engine.py's request policies, MTP drafts)
    \\
;

const Verb = enum { first, teacher, dump, bench };

pub fn isVerb(s: []const u8) bool {
    return std.meta.stringToEnum(Verb, s) != null;
}

const Args = struct {
    verb: Verb = .first,
    out: []const u8 = ".",
    model: [:0]const u8 = "",
    rank: u32 = 0,
    ranks: u32 = 2,
    layers: ?u32 = null,
    prompt: []const u8 = "",
    teacher: ?[]const u8 = null,
    steps: usize = 48,
    master: []const u8 = "127.0.0.1:18890",
    capacity: usize = 2560,
    long_context: bool = false,
    prefill_rows: usize = glm.engine.prefill_rows,
    mtp: bool = true,
    /// false (--no-mtp-layer): the MTP layer is not loaded either (python's without_mtp)
    mtp_layer: bool = true,
    dry: bool = false,
    triton: ?[]const u8 = null,
    fatbin: ?[]const u8 = null,
    /// --temperature T > 0: python's nucleus draw (top_k off; qualification_45c10bcf) at --top-p, --min-p and --seed; else greedy
    temperature: f64 = 0.0,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,
    seed: u64 = 0,

    /// bench's runs (glm_bench.Spec, KEY=VALUE,...)
    bench: []const u8 = "",
    /// bench's decode policy (glm_drafts.parse): 0 is serial, as before
    drafts: []const u8 = "0",

    pub fn sampling(a: Args) ?glm.nucleus.Sampling {
        return if (a.temperature > 0.0) .{ .seed = a.seed, .temperature = a.temperature, .top_p = a.top_p, .min_p = a.min_p } else null;
    }
};

fn parse(argv: []const [:0]const u8) !Args {
    if (argv.len < 3) return error.Usage;
    var a: Args = .{ .verb = std.meta.stringToEnum(Verb, argv[1]) orelse return error.Usage, .out = argv[2] };
    var i: usize = 3;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (std.mem.eql(u8, k, "--dry")) a.dry = true else if (std.mem.eql(u8, k, "--no-mtp")) a.mtp = false else if (std.mem.eql(u8, k, "--no-mtp-layer")) {
            a.mtp = false;
            a.mtp_layer = false;
        } else if (std.mem.eql(u8, k, "--long-context")) a.long_context = true else {
            if (i + 1 >= argv.len) return error.Usage;
            const v = argv[i + 1];
            i += 1;
            const int = std.fmt.parseInt;
            if (std.mem.eql(u8, k, "--model")) a.model = v else if (std.mem.eql(u8, k, "--rank")) a.rank = try int(u32, v, 10) else if (std.mem.eql(u8, k, "--ranks")) a.ranks = try int(u32, v, 10) else if (std.mem.eql(u8, k, "--layers")) a.layers = try int(u32, v, 10) else if (std.mem.eql(u8, k, "--prompt")) a.prompt = v else if (std.mem.eql(u8, k, "--teacher")) a.teacher = v else if (std.mem.eql(u8, k, "--steps")) a.steps = try int(usize, v, 10) else if (std.mem.eql(u8, k, "--master")) a.master = v else if (std.mem.eql(u8, k, "--capacity")) a.capacity = try int(usize, v, 10) else if (std.mem.eql(u8, k, "--prefill-rows")) a.prefill_rows = try int(usize, v, 10) else if (std.mem.eql(u8, k, "--triton")) a.triton = v else if (std.mem.eql(u8, k, "--fatbin")) a.fatbin = v else if (std.mem.eql(u8, k, "--temperature")) a.temperature = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, k, "--top-p")) a.top_p = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, k, "--min-p")) a.min_p = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, k, "--seed")) a.seed = try int(u64, v, 10) else if (std.mem.eql(u8, k, "--bench")) a.bench = v else if (std.mem.eql(u8, k, "--drafts")) a.drafts = v else return error.Usage;
        }
    }
    if (a.model.len == 0 or a.prompt.len == 0 or a.rank >= a.ranks) return error.Usage;
    if (a.verb == .teacher and a.teacher == null) return error.Usage;
    return a;
}

/// The run's tokens: the prompt (FILE[:NAME]) and, for `teacher`, qualification_6607f886's serial ones (the inputs, then the last next).
const Tokens = struct {
    name: []const u8,
    prompt: []u32,
    serial: []u32 = &.{},
};

fn ids(gpa: std.mem.Allocator, v: J) ![]u32 {
    const xs = v.array.items;
    const out = try gpa.alloc(u32, xs.len);
    for (xs, out) |x, *y| y.* = @intCast(x.integer);
    return out;
}

fn tokens(gpa: std.mem.Allocator, a: Args) !Tokens {
    const colon = std.mem.lastIndexOfScalar(u8, a.prompt, ':');
    const path = if (colon) |c| a.prompt[0..c] else a.prompt;
    const name = if (colon) |c| a.prompt[c + 1 ..] else "prompt";
    const text = try tool.slurp(gpa, path, 64 << 20);
    defer gpa.free(text);
    const p = try std.json.parseFromSlice(J, gpa, text, .{});
    defer p.deinit();
    const list = if (p.value == .array) p.value else (p.value.object.get("prompts") orelse return error.NoPrompts).object.get(name) orelse return error.NoPrompt;
    var t: Tokens = .{ .name = name, .prompt = try ids(gpa, list) };
    if (a.teacher) |s| {
        const st = try tool.slurp(gpa, s, 64 << 20);
        defer gpa.free(st);
        const q = try std.json.parseFromSlice(J, gpa, st, .{});
        defer q.deinit();
        const steps = (q.value.object.get("steps") orelse return error.NoSteps).array.items;
        if (steps.len == 0) return error.NoSteps;
        t.serial = try gpa.alloc(u32, steps.len + 1);
        for (steps, 0..) |x, k| t.serial[k] = @intCast(x.object.get("input").?.integer);
        t.serial[steps.len] = @intCast(steps[steps.len - 1].object.get("next_serial").?.integer);
    }
    return t;
}

/// A digest of what both ranks are asked (verb, tokens, steps), part of the agreed settings: lockstep.
fn request(a: Args, t: Tokens, buf: []u8) ![]const u8 {
    var h: Sha256 = .init(.{});
    h.update(std.mem.sliceAsBytes(t.prompt));
    h.update(std.mem.sliceAsBytes(t.serial));
    h.update(a.bench); // bench's runs (empty for the other verbs: their digests as before)
    if (!a.mtp_layer) h.update("no-mtp-layer");
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.bufPrint(buf, "{t} layers={?d} steps={d} tokens={s}", .{ a.verb, a.layers, a.steps, std.fmt.bytesToHex(d[0..8].*, .lower) });
}

// -- output: qualification_6607f886's files --------------------------------------------------------------------------------------------

/// `s` as a C path in `buf`.
fn cpath(buf: []u8, s: []const u8) ![:0]const u8 {
    if (s.len + 1 > buf.len) return error.PathTooLong;
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

fn mkdirs(path: []const u8) !void {
    var buf: [4096]u8 = undefined;
    for (path, 0..) |c, i| if (c == '/' and i > 0) {
        const z = try cpath(&buf, path[0..i]);
        _ = std.c.mkdir(z, 0o755);
    };
    const z = try cpath(&buf, path);
    _ = std.c.mkdir(z, 0o755);
}

fn writeFile(path: []const u8, bytes: []const u8) !void {
    var buf: [4096]u8 = undefined;
    const z = try cpath(&buf, path);
    const fd = std.c.open(z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    if (std.c.write(fd, bytes.ptr, bytes.len) != @as(isize, @intCast(bytes.len))) return error.CannotWrite;
}

fn dtypeName(d: glm.views.DType) []const u8 {
    return switch (d) {
        .bf16 => "bfloat16",
        .f32 => "float32",
        .f16 => "float16",
        .i32 => "int32",
        .i64 => "int64",
        .i16 => "int16",
        .u8 => "uint8",
        .f8e4m3 => "float8_e4m3fn",
    };
}

fn shapeJson(out: *std.ArrayList(u8), gpa: std.mem.Allocator, shape: []const usize) !void {
    try out.append(gpa, '[');
    for (shape, 0..) |s, i| try out.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", s });
    try out.append(gpa, ']');
}

/// qualification_6607f886's Recorder: each tapped tensor copied down, hashed, its kept rows written to `where`, an entry as python's.
const Recorder = struct {
    gpa: std.mem.Allocator,
    out: []const u8,
    where: []const u8 = "",
    keep: ?[]const [2]usize = null,
    stream: ?cuda.Stream = null,
    seq: usize = 0,
    entries: std.ArrayList(u8) = .empty,
    n: usize = 0,
    total: usize = 0,
    /// the taps' own time (copies, hashes, writes) after the stream drained, for the log's timing lines
    tap_ns: u64 = 0,

    fn tensor(ctx: *anyopaque, name: []const u8, x: T, pos0: usize, rows: usize, kind: ?[]const u8) anyerror!void {
        const r: *Recorder = @ptrCast(@alignCast(ctx));
        if (r.stream) |s| try s.synchronize(); // the forward's work up to the tap, before the tap's own time
        const t0 = nowNs();
        defer r.tap_ns += nowNs() - t0;
        const gpa = r.gpa;
        const shape = x.shape[0..x.dims];
        const dim: usize = if (shape[0] == rows) 0 else if (x.dims > 1 and shape[1] == rows) 1 else 0;
        const bytes = glm.compute.bytes(x);
        const host = try gpa.alloc(u8, bytes);
        defer gpa.free(host);
        if (r.stream) |s| {
            try s.d.check(s.d.api.cuMemcpyDtoH_v2(@ptrCast(host.ptr), x.ptr, bytes), "cuMemcpyDtoH");
        } else @memset(host, 0); // --dry: nothing ran
        var sha: [32]u8 = undefined;
        Sha256.hash(host, &sha, .{});
        const e = &r.entries;
        try e.print(gpa, "{s}\n{{\"name\": \"{s}\", \"pos0\": {d}, \"rows\": {d}, \"row_dim\": {d}, \"shape\": ", .{ if (r.n == 0) "" else ",", name, pos0, rows, dim });
        try shapeJson(e, gpa, shape);
        try e.print(gpa, ", \"dtype\": \"{s}\", \"sha256\": \"{s}\"", .{ dtypeName(x.dtype), std.fmt.bytesToHex(sha, .lower) });
        if (kind) |k| try e.print(gpa, ", \"kind\": \"{s}\"", .{k});
        // the rows kept: every row, or the positions in `keep` (qualification_6607f886's long prompt: its first, chunk-edge and last rows)
        var pick: std.ArrayList(usize) = .empty;
        defer pick.deinit(gpa);
        for (0..rows) |i| if (want(r.keep, pos0 + i)) try pick.append(gpa, i);
        if (pick.items.len > 0) {
            r.seq += 1;
            var fb: [512]u8 = undefined;
            const rel = try std.fmt.bufPrint(&fb, "{s}/{d:0>5}-{s}.bin", .{ r.where, r.seq, name });
            const outer = if (dim == 0) 1 else shape[0];
            const row_bytes = bytes / outer / rows;
            var sub: std.ArrayList(u8) = .empty;
            defer sub.deinit(gpa);
            for (0..outer) |o| for (pick.items) |i| try sub.appendSlice(gpa, host[(o * rows + i) * row_bytes ..][0..row_bytes]);
            var pb: [1024]u8 = undefined;
            try writeFile(try std.fmt.bufPrint(&pb, "{s}/{s}", .{ r.out, rel }), sub.items);
            try e.print(gpa, ", \"saved\": {{\"file\": \"{s}\", \"positions\": [", .{rel});
            for (pick.items, 0..) |i, j| try e.print(gpa, "{s}{d}", .{ if (j == 0) "" else ", ", pos0 + i });
            var s2: [5]usize = undefined;
            @memcpy(s2[0..shape.len], shape);
            s2[dim] = pick.items.len;
            try e.appendSlice(gpa, "], \"shape\": ");
            try shapeJson(e, gpa, s2[0..shape.len]);
            try e.append(gpa, '}');
        }
        try e.append(gpa, '}');
        r.n += 1;
        r.total += 1;
    }

    fn want(keep: ?[]const [2]usize, pos: usize) bool {
        const k = keep orelse return true;
        for (k) |r| if (pos >= r[0] and pos < r[1]) return true;
        return false;
    }

    fn tap(r: *Recorder) glm.dump.Tap {
        return .{ .ctx = r, .tensor = tensor };
    }

    /// The entries so far as a JSON list, and a fresh start.
    fn take(r: *Recorder) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        try out.append(r.gpa, '[');
        try out.appendSlice(r.gpa, r.entries.items);
        try out.appendSlice(r.gpa, "\n]");
        r.entries.clearRetainingCapacity();
        r.n = 0;
        return out.toOwnedSlice(r.gpa);
    }
};

// -- the verbs, the same on a GPU and recorded ------------------------------------------------------------------------

/// What a verb needs beyond the Engine: the logits' bytes on the host (a GPU's; --dry: zeros).
const Host = struct {
    stream: ?cuda.Stream,

    pub fn logitsRow(h: Host, gpa: std.mem.Allocator, x: T) ![]f32 {
        const n = x.shape[1];
        const out = try gpa.alloc(f32, n);
        @memset(out, 0);
        const s = h.stream orelse return out;
        try s.synchronize();
        const raw = try gpa.alloc(u16, n);
        defer gpa.free(raw);
        try s.d.check(s.d.api.cuMemcpyDtoH_v2(@ptrCast(raw.ptr), x.ptr, n * 2), "cuMemcpyDtoH");
        for (raw, out) |b, *f| f.* = @bitCast(@as(u32, b) << 16); // logits[:1].float(): bf16 widened exactly
        return out;
    }
};

/// The log's timing lines (qualification_66fa03c8 §7 item 6), on stderr only, so the saved files are the same with or without them. A
/// span ends once the rank's stream has drained (on --dry: the CPU's recording time).
const Span = struct {
    t0: u64,

    fn start() Span {
        return .{ .t0 = nowNs() };
    }

    fn ms(s: Span, f: *glm.fwd.Fwd) !f64 {
        return msOf(try drained(f) - s.t0);
    }
};

/// The clock once the rank's stream has run what was launched.
fn drained(f: *glm.fwd.Fwd) !u64 {
    if (f.gpu) |g| try g.stream.synchronize();
    return nowNs();
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn msOf(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn first(e: *Engine, f: *glm.fwd.Fwd, a: Args, t: Tokens, gpa: std.mem.Allocator) !u32 {
    const span = Span.start();
    const tok = try e.prefill(f, t.prompt, a.mtp);
    std.debug.print("TIME rank {d}: prefill of {d} tokens: {d:.3} ms\n", .{ a.rank, t.prompt.len, try span.ms(f) });
    var buf: [512]u8 = undefined;
    const g = e.stats.gathers;
    const text = try std.fmt.bufPrint(&buf, "{{\"prompt\": \"{s}\", \"tokens\": {d}, \"first\": {d}, \"rank\": {d}, \"dry\": {}, \"temperature\": {d}, \"top_p\": {d}, \"min_p\": {d}, \"seed\": {d}, \"gathers\": [{d}, {d}, {d}]}}\n", .{ t.name, t.prompt.len, tok, a.rank, a.dry, a.temperature, a.top_p, a.min_p, a.seed, g[0], g[1], g[2] });
    var pb: [1024]u8 = undefined;
    try mkdirs(a.out);
    try writeFile(try std.fmt.bufPrint(&pb, "{s}/first-{s}-rank{d}.json", .{ a.out, t.name, a.rank }), text);
    _ = gpa;
    return tok;
}

fn draws(e: *Engine, f: *glm.fwd.Fwd, hidden: T, next: u32, out: *[3]u32) ![]u32 {
    if (e.mbuf == null) return out[0..0];
    return e.draft(f, hidden, &.{next}, 3, out);
}

fn teacher(e: *Engine, f: *glm.fwd.Fwd, a: Args, t: Tokens, h: Host, rec: *Recorder, gpa: std.mem.Allocator) !usize {
    const toks = t.serial;
    const steps = @min(a.steps, toks.len -| 1); // step k feeds toks[k] and checks toks[k + 1] (qualification_00e296c8: not len - 2)
    var pb: [1024]u8 = undefined;
    const where = try std.fmt.allocPrint(gpa, "teacher/{s}", .{t.name});
    defer gpa.free(where);
    try mkdirs(try std.fmt.bufPrint(&pb, "{s}/{s}", .{ a.out, where }));
    const span = Span.start();
    const tok0 = try e.prefill(f, t.prompt, a.mtp);
    std.debug.print("TIME rank {d}: prefill of {d} tokens: {d:.3} ms\n", .{ a.rank, t.prompt.len, try span.ms(f) });
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    var d: [3]u32 = undefined;
    const before = try draws(e, f, glm.compute.rows(e.last_hidden, 0, 1), toks[0], &d);
    try json.print(gpa, "{{\"prompt\": \"{s}\", \"first\": {d}, \"first_equal\": {}, \"draws_before\": [", .{ t.name, tok0, tok0 == toks[0] });
    for (before, 0..) |x, i| try json.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", x });
    try json.appendSlice(gpa, "], \"steps\": [");
    for (0..steps) |k| {
        rec.where = try std.fmt.bufPrint(&pb, "{s}/s{d:0>2}", .{ where, k });
        var sb: [1024]u8 = undefined;
        try mkdirs(try std.fmt.bufPrint(&sb, "{s}/{s}", .{ a.out, rec.where }));
        const step = Span.start();
        const taps = rec.tap_ns;
        e.tap = rec.tap();
        const logits = try e.forward(f, toks[k .. k + 1]);
        e.tap = null;
        const row0 = try drained(f); // the forward done: the row's copy, file and hash are the host's
        const lg = try h.logitsRow(gpa, logits);
        defer gpa.free(lg);
        var fb: [256]u8 = undefined;
        const file = try std.fmt.bufPrint(&fb, "{s}/logits-{d:0>2}.bin", .{ where, k });
        try writeFile(try std.fmt.bufPrint(&sb, "{s}/{s}", .{ a.out, file }), std.mem.sliceAsBytes(lg));
        var sha: [32]u8 = undefined;
        Sha256.hash(std.mem.sliceAsBytes(lg), &sha, .{});
        const row_ns = nowNs() - row0;
        try glm.compute.commit(f, &e.st, &e.buf, 1, 1);
        const dr = try draws(e, f, e.mainHidden(1), toks[k + 1], &d);
        std.debug.print("TIME rank {d}: step {d} at position {d}: {d:.3} ms (the dump taps {d:.3} ms, the logits row {d:.3} ms)\n", .{ a.rank, k, e.st.pos - 1, try step.ms(f), msOf(rec.tap_ns - taps), msOf(row_ns) });
        const off = e.m.rank * e.m.vl();
        try json.print(gpa, "{s}\n{{\"k\": {d}, \"input\": {d}, \"next_serial\": {d}, \"position\": {d}, \"logits\": {{\"file\": \"{s}\", \"shape\": [1, {d}], \"sha256\": \"{s}\", \"vocab_offset\": {d}}}, \"shard_top5\": [", .{ if (k == 0) "" else ",", k, toks[k], toks[k + 1], e.st.pos - 1, file, lg.len, std.fmt.bytesToHex(sha, .lower), off });
        var top: [5]usize = undefined;
        glm.sample.topkOrder(lg, 5, &top); // the teacher's torch.topk(lg[0], 5), ties in CUDA's order (qualification_66fa03c8 §7 item 3)
        for (top, 0..) |i, j| try json.print(gpa, "{s}[{d}, {d}]", .{ if (j == 0) "" else ", ", i + off, lg[i] });
        try json.appendSlice(gpa, "], \"mtp_draws\": [");
        for (dr, 0..) |x, i| try json.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", x });
        const layers = try rec.take();
        defer gpa.free(layers);
        try json.print(gpa, "], \"layers\": {s}}}", .{layers});
    }
    try json.appendSlice(gpa, "\n]}\n");
    try writeFile(try std.fmt.bufPrint(&pb, "{s}/{s}/steps.json", .{ a.out, where }), json.items);
    return steps;
}

fn dump(e: *Engine, f: *glm.fwd.Fwd, a: Args, t: Tokens, rec: *Recorder, gpa: std.mem.Allocator) !u32 {
    var pb: [1024]u8 = undefined;
    rec.where = try std.fmt.bufPrint(&pb, "prefill/{s}", .{t.name});
    var sb: [1024]u8 = undefined;
    try mkdirs(try std.fmt.bufPrint(&sb, "{s}/{s}", .{ a.out, rec.where }));
    const n = t.prompt.len;
    const keep = [_][2]usize{ .{ 0, 8 }, .{ 2040, 2056 }, .{ 4088, 4104 }, .{ n -| 8, n } };
    rec.keep = if (n <= 256) null else &keep; // qualification_6607f886's rows for a long prompt
    const span = Span.start();
    const taps = rec.tap_ns;
    e.tap = rec.tap();
    // a block of lane A's not in the tree ends the prefill: the layers before it are written, and the stop said
    const got = e.prefill(f, t.prompt, a.mtp);
    e.tap = null;
    const stopped = if (got) |_| false else |err| if (err == error.NotPorted) true else return err;
    if (!stopped) std.debug.print("TIME rank {d}: prefill of {d} tokens: {d:.3} ms (the dump taps {d:.3} ms)\n", .{ a.rank, n, try span.ms(f), msOf(rec.tap_ns - taps) });
    const tok = got catch 0;
    const entries = try rec.take();
    defer gpa.free(entries);
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.print(gpa, "{{\"prompt\": \"{s}\", \"tokens\": {d}, \"first\": {d}, \"chunk_rows\": {d}, \"dense_limit\": {d}, \"stopped\": {s}, \"entries\": {s}}}\n", .{ t.name, n, tok, e.prefill_rows, e.m.denseLimit(), if (stopped) "\"NotPorted: a block of lane A's not in this tree\"" else "null", entries });
    try writeFile(try std.fmt.bufPrint(&sb, "{s}/{s}/layers.json", .{ a.out, rec.where }), json.items);
    if (stopped) return error.NotPorted;
    return tok;
}

// -- --dry: the verb recorded over a checkpoint's headers ----------------------------------------------------------

fn dry(gpa: std.mem.Allocator, a: Args, t: Tokens) !u8 {
    var ix = try glm.index.open(gpa, a.model);
    defer ix.deinit();
    var r = try tool.plans(gpa, .{ .dir = a.model, .rank = a.rank, .ranks = a.ranks, .layers = a.layers, .mtp_layer = a.mtp_layer }, &ix);
    defer r.deinit(gpa);
    if (!a.mtp) r.m.mtp_layers = 0;
    // the fwd.Kernels this build's GPU run passes the rank: refused as there, and without .dsa (qualification_qualification_ac5a9c94 QUESTION 2)
    const gpu = @import("glm_run_gpu.zig");
    const sd = gpu.exl3Sched(a.rank) catch return 1; // 0128's switches, said when on (glm_run_gpu.zig)
    if (gpu.images) {
        const k = gpu.drySet(r.m.quant == .exl3, sd) catch |err| {
            std.debug.print("FAIL rank {d}: --dry: this build's GPU run would refuse the rank: {t}\n", .{ a.rank, err });
            return 1;
        };
        if (gpu.lacksDsa(k)) {
            std.debug.print("FAIL rank {d}: --dry: the GPU run's fwd.Kernels has no .dsa (no glm_select image): its decode would stop with NoDsaKernels from position {d}\n", .{ a.rank, r.m.denseLimit() });
            return 1;
        }
        var kb: [64]u8 = undefined;
        std.debug.print("INFO rank {d}: --dry: the GPU run's fwd.Kernels: {s}\n", .{ a.rank, try gpu.describe(k, &kb) });
    } else std.debug.print("INFO rank {d}: --dry: a host build (no kernel images): its GPU verbs refuse, so no fwd.Kernels\n", .{a.rank});
    var base: u64 = 1 << 36;
    for (r.split.groups(), 0..) |bytes, g| {
        r.pk.place(@intCast(g), base);
        base += std.mem.alignForward(u64, r.pk.homeBytes(@intCast(g)), 1 << 20) + (1 << 20);
        if (!r.pk.keeps(@intCast(g))) continue; // EXL3's experts stay in their stored group
        r.pk.placeKept(@intCast(g), base);
        base += std.mem.alignForward(u64, bytes, 1 << 20) + (1 << 20);
    }
    var w = try glm.views.rank(gpa, &r.m);
    defer w.deinit(gpa);
    // every view bound, none left at address 0 (qualification_d3c2a5a7 §7 item 2): the GPU run's bring-up refuses the same
    const bound = glm.pack.bind(&w, &r.pk) catch |err| {
        std.debug.print("FAIL rank {d}: --dry: binding the forward's views: {t}\n", .{ a.rank, err });
        return 1;
    };
    var space: glm.trace.Space = .{ .gpa = gpa, .next = std.mem.alignForward(u64, base, 1 << 36) + (1 << 36) };
    defer space.deinit();
    const e = try Engine.create(gpa, &r.m, &w, a.capacity, 8, a.prefill_rows, a.long_context);
    defer e.destroy(gpa);
    if (a.sampling()) |s| {
        e.sampling = s;
        try e.useSampling(gpa);
    }
    if (a.verb == .bench) try bench.forecast(&r, e, &w, a, gpa); // qualification_e84ee1bf: the rank's memory, before the stand-ins
    try e.walk(&space, glm.trace.placeAll);
    try e.walkTmp(&space, glm.trace.placeAll);
    try logBytes(e, a.rank, "--dry: ");
    e.tmp.space = &space;
    var tr = glm.trace.Trace.init(gpa);
    defer tr.deinit();
    var f = glm.fwd.Fwd.recording(&r.m, &tr);
    var rec: Recorder = .{ .gpa = gpa, .out = a.out };
    defer rec.entries.deinit(gpa);
    const res: anyerror!usize = switch (a.verb) {
        .first => widen(first(e, &f, a, t, gpa)),
        .teacher => teacher(e, &f, a, t, .{ .stream = null }, &rec, gpa),
        .dump => widen(dump(e, &f, a, t, &rec, gpa)),
        .bench => bench.run(e, &f, a, t, .{ .stream = null }, .{}, gpa),
    };
    try f.flush();
    // TF_GLM_DRY_EVENTS=FILE: the launches as JSON lines, addresses by buffer path (glm_run_events.zig)
    if (std.c.getenv("TF_GLM_DRY_EVENTS")) |p| try @import("glm_run_events.zig").write(gpa, std.mem.span(p), tr.events.items, &space, &w);
    var n = [3]usize{ 0, 0, 0 };
    for (tr.events.items) |x| n[@backingInt(x.op)] += 1;
    const last = if (tr.events.items.len > 0) tr.events.items[tr.events.items.len - 1].name else "-";
    std.debug.print("{t} --dry rank {d}/{d}, {d} layers{s}: {d} views bound; {d} launches recorded (triton {d}, ext {d}, comm {d}), the last {s}; ", .{ a.verb, a.rank, a.ranks, r.m.layers, if (r.m.mtp_layers > 0) " and the MTP head" else "", bound, tr.events.items.len, n[0], n[1], n[2], last });
    if (res) |_| {
        std.debug.print("ran to the end; {d} tensors recorded\n", .{rec.total});
        return 0;
    } else |err| {
        std.debug.print("stopped: {t} ({s})\n", .{ err, if (err == error.NotPorted) "a block of lane A's not in this tree" else "an error" });
        return if (err == error.NotPorted) 0 else 1;
    }
}

pub fn main(init: std.process.Init, argv: []const [:0]const u8) !u8 {
    const gpa = init.gpa;
    const a = parse(argv) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    const t = try tokens(gpa, a);
    defer gpa.free(t.prompt);
    defer if (t.serial.len > 0) gpa.free(t.serial);
    if (a.verb == .teacher and t.serial.len < 2) return error.NoSteps; // one step wants two tokens (qualification_00e296c8, qualification_qualification_7d619f35 §5)
    if (a.dry) return dry(gpa, a, t);
    return @import("glm_run_gpu.zig").run(init, a, t);
}

/// A verb's count as usize (first and dump give the first token).
pub fn widen(x: anyerror!u32) anyerror!usize {
    return @as(usize, try x);
}

/// The Engine's device block, its prompt Buffers' and its DSA scratch's bytes, for G-mem (qualification_qualification_44ba554e.5 B7).
pub fn logBytes(e: *Engine, rank: u32, what: []const u8) !void {
    const b = try e.deviceBytes();
    const mib = struct {
        fn of(x: usize) f64 {
            return @as(f64, @floatFromInt(x)) / (1024.0 * 1024.0);
        }
    }.of;
    std.debug.print("INFO rank {d}: {s}the Engine's device block {d:.1} MiB: the prompt Buffers ({d} rows) {d:.1} MiB, the DSA scratch {d:.1} MiB ({d} B)\n", .{ rank, what, mib(b.total), e.prefill_rows, mib(b.pbuf), mib(b.dsa), b.dsa });
}

pub const Run = struct {
    pub const ArgsT = Args;
    pub const TokensT = Tokens;
    pub const RecorderT = Recorder;
    pub const HostT = Host;
    pub const firstFn = first;
    pub const teacherFn = teacher;
    pub const dumpFn = dump;
    pub const requestFn = request;
};
