//! tf-glm-afd attn (qualification_0fa2bee9; tf_glm_afd.zig hands the `attn` role and `check-audit` here): the AFD attention node at
//! world 1, python 2.15's engine.py attention role (load(world=1, experts=False), no MTP: engine.py:499-502, 678-681)
//! through qualification_0fa2bee9's engine (cuda_afd_engine.zig: B3's Engine, qualification_9051d0c1's provider as Fwd.moe, Afd.begin and Node.check
//! around each forward):
//! - first: the prompt's greedy first token (decode.prefill under AFD: chunks begun MODE_CHUNK), OUT/first-NAME-attn.json;
//! - teacher: then --steps teacher-forced decode steps on --teacher STEPS.json's tokens (qualification_6607f886's), each step's logits row
//!   [1, V] fp32 in OUT/teacher/NAME/logits-KK.bin: tf-glm-forward's steps.json at world 1 (no draws, no taps);
//! - check-weights: qualification_9051d0c1's world-1 plan over the checkpoint's headers, its bytes (qualification_25364d42's 16.62 GiB whole).
//! --audit: TF_AFD_CHECK's attention half (cuda_afd_audit.zig), each decode forward's layers hashed after its sync, a
//! JSON line each in OUT/afd-check-attn.jsonl; `check-audit ATTN E0 E1` is python's window comparison over the files.
//! --dry runs first and teacher on a CPU over config.json: the Engine at stand-in addresses, every launch and exchange
//! recorded (nothing runs; --audit hashes zeros), and the GPU path's fwd.Kernels built at stand-ins. On a GPU the verbs
//! are afd_attn_gpu.zig's (qualification_cd073cc0): the world-1 load, the kernels, the exchange, then the same verbs.
//! qualification_21e21bc0: TF_AFD_MTP=1 keeps the MTP layer and attn bench's drafts=SPEC drafts with it (afd_attn_drafts.zig).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const cli = @import("tf_glm_afd.zig");
const files = @import("afd_expert.zig");
const gpu = @import("afd_attn_gpu.zig");
/// qualification_e5a7fc3b's timed test (attn bench)
const bench = @import("afd_attn_bench.zig");
/// qualification_21e21bc0: TF_AFD_MTP's model (the MTP layer kept) and attn bench's drafts
const drafted = @import("afd_attn_drafts.zig");
/// qualification_4be15bbf: attn serve (Ash's server over the attention role) and its --dry forecast
const serve = @import("afd_attn_serve.zig");

const ag = glm.afd_engine;
const afd = glm.afd_attn;
const audit = glm.afd_audit;
const trace = glm.trace;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const J = std.json.Value;

/// The run's tokens (glm_run.zig's): the prompt (FILE[:NAME]: a JSON list of ids, or qualification_6607f886's results.json's prompts.NAME)
/// and, for `teacher`, the serial ones (each step's input, then the last step's next): from qualification_6607f886's steps.json, or from
/// qualification_1cad2b94's kit teacher.json (FILE[:KEY]: KEY's `inputs` and `next`, KEY the prompt's name when not given).
pub const Tokens = struct {
    name: []const u8,
    prompt: []u32,
    serial: []u32 = &.{},

    pub fn deinit(t: *Tokens, gpa: Allocator) void {
        gpa.free(t.prompt);
        if (t.serial.len > 0) gpa.free(t.serial);
    }
};

fn ids(gpa: Allocator, v: J) ![]u32 {
    const xs = v.array.items;
    const out = try gpa.alloc(u32, xs.len);
    for (xs, out) |x, *y| y.* = @intCast(x.integer);
    return out;
}

pub fn tokens(gpa: Allocator, prompt: []const u8, teacher_path: ?[]const u8) !Tokens {
    const colon = std.mem.lastIndexOfScalar(u8, prompt, ':');
    const name = if (colon) |c| prompt[c + 1 ..] else "prompt";
    const text = try files.readFile(gpa, if (colon) |c| prompt[0..c] else prompt);
    defer gpa.free(text);
    const p = try std.json.parseFromSlice(J, gpa, text, .{});
    defer p.deinit();
    const list = if (p.value == .array) p.value else ((p.value.object.get("prompts") orelse return error.NoPrompts).object.get(name) orelse return error.NoPrompt);
    var t: Tokens = .{ .name = name, .prompt = try ids(gpa, list) };
    errdefer gpa.free(t.prompt);
    if (t.prompt.len == 0) return error.EmptyPrompt;
    if (teacher_path) |tp| {
        const tc = std.mem.lastIndexOfScalar(u8, tp, ':');
        const st = try files.readFile(gpa, if (tc) |c| tp[0..c] else tp);
        defer gpa.free(st);
        const q = try std.json.parseFromSlice(J, gpa, st, .{});
        defer q.deinit();
        if (q.value.object.get("steps")) |sv| { // qualification_6607f886's steps.json
            const steps = sv.array.items;
            if (steps.len == 0) return error.NoSteps; // one step wants two tokens (qualification_00e296c8; qualification_qualification_a12cb1bf's teacher fix)
            t.serial = try gpa.alloc(u32, steps.len + 1);
            for (steps, 0..) |x, k| t.serial[k] = @intCast(x.object.get("input").?.integer);
            t.serial[steps.len] = @intCast(steps[steps.len - 1].object.get("next_serial").?.integer);
        } else { // qualification_1cad2b94's teacher.json: {KEY: {first, inputs, next, positions}}
            const e = (q.value.object.get(if (tc) |c| tp[c + 1 ..] else name) orelse return error.NoTeacherPrompt).object;
            const inputs = e.get("inputs").?.array.items;
            const next = e.get("next").?.array.items;
            if (inputs.len == 0 or next.len != inputs.len) return error.NoSteps;
            t.serial = try gpa.alloc(u32, inputs.len + 1);
            for (inputs, 0..) |x, k| t.serial[k] = @intCast(x.integer);
            t.serial[inputs.len] = @intCast(next[next.len - 1].integer);
        }
    }
    return t;
}

// -- files (glm_run.zig's: lane B's file, not shared) ----------------------------------------------------------------

fn cpath(buf: []u8, s: []const u8) ![:0]const u8 {
    if (s.len + 1 > buf.len) return error.PathTooLong;
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

fn mkdirs(path: []const u8) !void {
    var buf: [4096]u8 = undefined;
    for (path, 0..) |c, i| if (c == '/' and i > 0) {
        _ = std.c.mkdir(try cpath(&buf, path[0..i]), 0o755);
    };
    _ = std.c.mkdir(try cpath(&buf, path), 0o755);
}

fn writeFile(path: []const u8, bytes: []const u8) !void {
    var buf: [4096]u8 = undefined;
    const fd = std.c.open(try cpath(&buf, path), .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    if (bytes.len > 0 and std.c.write(fd, bytes.ptr, bytes.len) != @as(isize, @intCast(bytes.len))) return error.CannotWrite; // an empty slice's ptr is no user address (EFAULT)
}

/// `bytes` at `path`, its directory made first (bench's record).
pub fn save(path: []const u8, bytes: []const u8) !void {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| try mkdirs(path[0..i]);
    try writeFile(path, bytes);
}

// -- the audit's attention half and the window ------------------------------------------------------------------------

/// Attn.check's sink: each checked forward's record as a JSON line (written out at the run's end).
pub const Records = struct {
    gpa: Allocator,
    text: std.ArrayList(u8) = .empty,
    forwards: usize = 0,

    pub fn forward(ctx: *anyopaque, seq: u32, R: usize, layers: []const audit.Layer) anyerror!void {
        const r: *Records = @ptrCast(@alignCast(ctx));
        try audit.writeRecord(r.gpa, &r.text, .attn, seq, R, layers);
        r.forwards += 1;
    }

    pub fn deinit(r: *Records) void {
        r.text.deinit(r.gpa);
    }
};

/// python's _finish_check over the files: each attention record against the experts' of its seq (afd.py:591-656).
pub fn checkAudit(gpa: Allocator, paths: [3][]const u8) !audit.Window {
    var side: [3]std.ArrayList(audit.Record) = .{ .empty, .empty, .empty };
    defer for (&side) |*s| {
        for (s.items) |r| gpa.free(r.layers);
        s.deinit(gpa);
    };
    for (paths, &side, 0..) |p, *s, k| {
        const text = try files.readFile(gpa, p);
        defer gpa.free(text);
        var it = std.mem.tokenizeScalar(u8, text, '\n');
        while (it.next()) |l| {
            const r = try audit.readRecord(gpa, l);
            errdefer gpa.free(r.layers);
            if (@backingInt(r.side) != k) return error.WrongSide;
            try s.append(gpa, r);
        }
    }
    var w: audit.Window = .{};
    for (side[0].items) |a| {
        var theirs: [2]?[]const audit.Layer = .{ null, null };
        for (side[1..], &theirs) |s, *t| for (s.items) |e| if (e.seq == a.seq) {
            t.* = e.layers;
        };
        w.forward(a.rows, a.layers, theirs);
    }
    return w;
}

/// `tf-glm-afd check-audit ATTN.jsonl E0.jsonl E1.jsonl`: the window's line; OK only with no mismatch.
pub fn checkAuditCmd(gpa: Allocator, argv: []const [:0]const u8) cli.Outcome {
    if (argv.len != 3) return .{ .usage = "check-audit needs ATTN E0 E1" };
    const w = checkAudit(gpa, .{ argv[0], argv[1], argv[2] }) catch |err| return .{ .usage = @errorName(err) };
    var buf: [512]u8 = undefined;
    std.debug.print("INFO {s}\n", .{w.line(&buf)});
    if (w.forwards == 0) return .{ .usage = "no-records" };
    if (w.mismatches > 0) return .{ .usage = "audit-mismatch" };
    return .{ .ok = std.fmt.allocPrint(gpa, "check-audit {d} forwards {d} layers", .{ w.forwards, w.layers }) catch "check-audit" };
}

// -- the verbs, the same recorded and on a GPU -------------------------------------------------------------------

/// What a verb runs on: the attention engine, its Fwd, the stream (null: recorded, nothing ran) and where it writes.
pub const Run = struct {
    gpa: Allocator,
    at: *ag.Attn,
    f: *glm.fwd.Fwd,
    stream: ?cuda.Stream,
    out: []const u8,
    t: *const Tokens,
};

pub fn first(r: Run) !u32 {
    const tok = try r.at.prefill(r.f, r.t.prompt);
    var buf: [512]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{{\"prompt\": \"{s}\", \"tokens\": {d}, \"first\": {d}, \"world\": 1, \"dry\": {}}}\n", .{ r.t.name, r.t.prompt.len, tok, r.stream == null });
    var pb: [1024]u8 = undefined;
    try mkdirs(r.out);
    try writeFile(try std.fmt.bufPrint(&pb, "{s}/first-{s}-attn.json", .{ r.out, r.t.name }), text);
    return tok;
}

/// logits[:1].float() on the host: bf16 widened exactly (recorded: zeros).
fn logitsRow(r: Run, x: trace.T) ![]f32 {
    const n = x.shape[1];
    const out = try r.gpa.alloc(f32, n);
    @memset(out, 0);
    const s = r.stream orelse return out;
    try s.synchronize();
    const raw = try r.gpa.alloc(u16, n);
    defer r.gpa.free(raw);
    try s.d.check(s.d.api.cuMemcpyDtoH_v2(@ptrCast(raw.ptr), x.ptr, n * 2), "cuMemcpyDtoH");
    for (raw, out) |b, *v| v.* = @bitCast(@as(u32, b) << 16);
    return out;
}

/// torch.topk(row, 5)'s columns: the largest values, ties to the smaller column.
fn top5(row: []const f32, out: *[5]usize) void {
    for (out, 0..) |*o, j| {
        var best: ?usize = null;
        for (row, 0..) |v, i| {
            if (std.mem.indexOfScalar(usize, out[0..j], i) != null) continue;
            if (best == null or v > row[best.?]) best = i;
        }
        o.* = best orelse 0;
    }
}

/// prefill, then `steps` teacher-forced decode steps (each Attn.forward: begin, the step, the sync and the check; the
/// row read only after), the commit after each: tf-glm-forward's steps.json at world 1 (vocab_offset 0, the whole
/// vocabulary; no MTP draws under AFD; no recorder taps, `layers` empty).
pub fn teacher(r: Run, steps_max: usize) !usize {
    const gpa = r.gpa;
    const toks = r.t.serial;
    const steps = @min(steps_max, toks.len -| 1); // step k feeds toks[k] and checks toks[k + 1] (qualification_qualification_a12cb1bf's fix)
    const e = r.at.e;
    var pb: [1024]u8 = undefined;
    const where = try std.fmt.allocPrint(gpa, "teacher/{s}", .{r.t.name});
    defer gpa.free(where);
    try mkdirs(try std.fmt.bufPrint(&pb, "{s}/{s}", .{ r.out, where }));
    const tok0 = try r.at.prefill(r.f, r.t.prompt);
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.print(gpa, "{{\"prompt\": \"{s}\", \"world\": 1, \"first\": {d}, \"first_equal\": {}, \"draws_before\": [], \"steps\": [", .{ r.t.name, tok0, tok0 == toks[0] });
    for (0..steps) |k| {
        const logits = try r.at.forward(r.f, toks[k .. k + 1]);
        const lg = try logitsRow(r, logits);
        defer gpa.free(lg);
        var fb: [256]u8 = undefined;
        const file = try std.fmt.bufPrint(&fb, "{s}/logits-{d:0>2}.bin", .{ where, k });
        var sb: [1024]u8 = undefined;
        try writeFile(try std.fmt.bufPrint(&sb, "{s}/{s}", .{ r.out, file }), std.mem.sliceAsBytes(lg));
        var sha: [32]u8 = undefined;
        Sha256.hash(std.mem.sliceAsBytes(lg), &sha, .{});
        try glm.compute.commit(r.f, &e.st, &e.buf, 1, 1);
        try json.print(gpa, "{s}\n{{\"k\": {d}, \"input\": {d}, \"next_serial\": {d}, \"position\": {d}, \"logits\": {{\"file\": \"{s}\", \"shape\": [1, {d}], \"sha256\": \"{s}\", \"vocab_offset\": 0}}, \"shard_top5\": [", .{ if (k == 0) "" else ",", k, toks[k], toks[k + 1], e.st.pos - 1, file, lg.len, std.fmt.bytesToHex(sha, .lower) });
        var top: [5]usize = undefined;
        top5(lg, &top);
        for (top, 0..) |i, j| try json.print(gpa, "{s}[{d}, {d}]", .{ if (j == 0) "" else ", ", i, lg[i] });
        try json.appendSlice(gpa, "], \"mtp_draws\": [], \"layers\": []}");
    }
    try json.appendSlice(gpa, "\n]}\n");
    try writeFile(try std.fmt.bufPrint(&pb, "{s}/{s}/steps.json", .{ r.out, where }), json.items);
    return steps;
}

/// A verb's outcome: OK with its count, an exchange that failed (the node's reason: exit 2), or an error of its own.
pub fn outcome(gpa: Allocator, at: *ag.Attn, res: anyerror!usize, what: []const u8) cli.Outcome {
    const n = res catch |err| {
        if (at.failure()) |x| {
            var buf: [160]u8 = undefined;
            std.debug.print("INFO {s}\n", .{ag.describe(x, &buf)});
            return cli.fromFailure(.attn, x);
        }
        std.debug.print("INFO attn {s}: {t}{s}\n", .{ what, err, if (err == error.NotPorted) " (not ported: TF_GLM_LATENT=0's per-head caches, or a begin that is not eager or a chunk)" else "" });
        return .{ .usage = @errorName(err) };
    };
    return .{ .ok = std.fmt.allocPrint(gpa, "{s} {d}", .{ what, n }) catch what };
}

// -- --dry: the verb recorded over config.json ----------------------------------------------------------------------

/// What a --dry run recorded (qualification_f16009a8: its graphs' counts with TF_AFD_GRAPHS=1).
pub const Dry = struct { outcome: cli.Outcome, forwards: u64, launches: usize, exchanges: usize, collects: usize, records: usize, graphs: glm.afd_graphs.Stats = .{} };

/// qualification_18588626: a test's sink for the events a --dry run recorded (afd_cover_qmm.zig's G-cover), and the picker its recorded
/// sample calls answer with (null: --dry's stand-ins, token 0 at probability 0); both null outside that test.
pub const Sink = struct { ctx: *anyopaque, f: *const fn (*anyopaque, []const trace.Event) anyerror!void };
pub var dry_sink: ?Sink = null;
pub var dry_picker: ?trace.Picker = null;

/// qualification_f504f9df: TF_GLM_HALVES_KERNEL on the attention, read once before anything loads. The attention computes nothing with
/// it, but its settings carry the value (Afd.halves), so both halves must run the same prompt-window kernel
/// (SettingsDiffer otherwise); set, one INFO line; a bad value refuses the run (BadHalvesKernel).
pub fn halvesKernel() !glm.halves_kernel.Kernel {
    const k = glm.halves_kernel.fromEnv() catch |e| {
        std.debug.print("INFO attn: TF_GLM_HALVES_KERNEL takes k4 or g53kt3r (or 0, empty, unset)\n", .{});
        return e;
    };
    if (k.on()) std.debug.print("INFO attn: TF_GLM_HALVES_KERNEL={s}: the halves' prompt windows take it; the settings carry halves={s}\n", .{ k.setting(), k.setting() });
    return k;
}

/// `first|teacher --dry` over `config` (config.json's text): the model at world 1 cut to --layers, no MTP (python's AFD);
/// B3's Engine (long context past the dense limit, engine.py:575) and the provider at stand-in addresses, the routed
/// experts unplaced; qualification_0fa2bee9's engine runs the verb, every launch recorded and counted; --audit's records over a ring of
/// host zeros (the plumbing: nothing ran).
pub fn dry(gpa: Allocator, f: cli.Flags, t: *const Tokens, config: []const u8) !Dry {
    const paired = try glm.afd_pairs.fromEnv(); // qualification_0f991257: TF_AFD_PAIRS, read once before anything loads (BadAfdPairs)
    const inflight = try glm.afd_pairs.inflightFromEnv(paired); // qualification_c533375f: TF_AFD_INFLIGHT, after it (BadAfdInflight)
    const seq = try afd.seqFromEnv(); // qualification_969df694: TF_GLM_EXL3_SEQ, after them (BadExl3Seq): the settings' seq only
    const halves = try halvesKernel(); // qualification_f504f9df: TF_GLM_HALVES_KERNEL, read once before anything loads (BadHalvesKernel)
    const stripes = try cuda.xchg.proto.stripesFromEnv("attn --dry"); // qualification_17c70e37: TF_AFD_STRIPES, refused here as on a GPU
    const graphed = try glm.afd_graphs.fromEnv(); // qualification_f16009a8: TF_AFD_GRAPHS, read once before anything loads (BadAfdGraphs)
    const lanes_n = try glm.afd_lanes.fromEnv(paired); // qualification_9e49592e: TF_AFD_LANES, after both (BadAfdLanes)
    var m = try drafted.model(gpa, config, f.layers); // qualification_21e21bc0: the MTP layer only with TF_AFD_MTP=1
    _ = try cuda.xchg.setup.fromEnv("attn --dry"); // qualification_1fee6d78: TF_AFD_SETUP_MS, refused here as on a GPU; set, its INFO line
    var w = try glm.views.rank(gpa, &m);
    defer w.deinit(gpa);
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    // qualification_e5a7fc3b: over a checkpoint (or its header mirror) the views are the load's, bound as the GPU path binds them
    if (try bindDry(gpa, f, &m, &w)) |top| space.next = std.mem.alignForward(u64, top, 1 << 36) + (1 << 36) else try glm.views.walk(&w, glm.views.Path.init("w"), &space, glm.afd_load.skipExperts(*trace.Space, trace.placeAll));
    const e = try glm.engine.Engine.create(gpa, &m, &w, f.capacity, 8, f.max_rows, f.capacity > m.denseLimit());
    defer e.destroy(gpa);
    try e.walk(&space, trace.placeAll);
    try e.walkTmp(&space, trace.placeAll);
    e.tmp.space = &space;
    const eb = try e.deviceBytes(); // what Device.attach reserves on a GPU (LZ2b's memory declaration)
    std.debug.print("INFO attn --dry: the Engine's device block {d} B (the prompt Buffers {d} B, the DSA Scratch {d} B) at capacity {d}\n", .{ eb.total, eb.pbuf, eb.dsa, f.capacity });
    var a = try afd.Afd.initWith(&m, f.max_rows, m.mtp_layers > 0);
    a.halves = halves.setting(); // qualification_f504f9df: the settings both ends would agree carry it
    a.stripes = stripes; // qualification_17c70e37: only the settings carry it; the record (the exchange is one external call a layer) is the same
    if (stripes.k != 0) std.debug.print("INFO attn --dry: TF_AFD_STRIPES={d}/{d}: the settings carry it; the record is unset's\n", .{ stripes.k, stripes.min });
    drafted.info(&a, "--dry");
    a.exl3_seq = seq;
    if (seq) std.debug.print("INFO attn --dry: TF_GLM_EXL3_SEQ=1: the settings carry seq=1 (the halves' SEQ; the attention's launches are unchanged)\n", .{});
    try space.place("b.afd_recv", &a.afd_recv);
    var at = try ag.Attn.init(e, &a);
    var recs: Records = .{ .gpa = gpa };
    defer recs.deinit();
    var ring = afd.Ring.of(a.layout, e.buf.rows, null);
    const ring_bytes = try gpa.alloc(u8, if (f.audit) ring.total(a.positions()) else 0);
    defer gpa.free(ring_bytes);
    const layer_recs = try gpa.alloc(audit.Layer, a.positions());
    defer gpa.free(layer_recs);
    if (f.audit) {
        @memset(ring_bytes, 0);
        ring.bytes = ring_bytes;
        a.ring = ring;
        at.check = .{ .ctx = &recs, .forward = Records.forward, .layers = layer_recs };
    }
    // qualification_0f991257: TF_AFD_PAIRS=1, a prompt of 2+ chunks in pairs: B's copies placed after every other buffer (the settings
    // both ends would agree carry pairs), B's ring of host zeros with --audit; unset, nothing here runs
    var pp: glm.afd_pairs.Pairs = undefined;
    const pring_bytes = try gpa.alloc(u8, if (paired and lanes_n == 0 and f.audit) ring.total(a.positions()) else 0);
    defer gpa.free(pring_bytes);
    if (paired and lanes_n == 0) {
        a.paired = true;
        a.inflight = inflight;
        pp = glm.afd_pairs.Pairs.init(&at);
        pp.agreed = true;
        try pp.walk(&space, trace.placeAll);
        if (f.audit) {
            @memset(pring_bytes, 0);
            var r = afd.Ring.of(a.layout, e.buf.rows, null);
            r.bytes = pring_bytes;
            pp.ring = r;
        }
        at.pairs = &pp;
        std.debug.print("INFO attn --dry: TF_AFD_PAIRS=1: a prompt of 2+ chunks prefills in pairs (B's copies {d} B)\n", .{try pp.bytes()});
        if (inflight > 1) std.debug.print("INFO attn --dry: TF_AFD_INFLIGHT=2: two requests in flight a link inside a pair (B's shared-expert rows among B's copies: {d} B)\n", .{if (pp.c.sy) |s| s.bytes() else 0});
    }
    // qualification_f16009a8: TF_AFD_GRAPHS=1, decode windows and MTP steps through the graphs by key, recorded eagerly, every later forward
    // of a key compared with its first (G-trace); unset, nothing here runs
    var gr = glm.afd_graphs.Graphs.init(gpa, null, "attn --dry");
    defer gr.deinit();
    if (graphed) {
        gr.max = std.math.maxInt(usize);
        at.graphs = &gr;
        std.debug.print("INFO attn --dry: TF_AFD_GRAPHS=1: decode windows and MTP steps through the graphs by key, each key's forwards compared with its first\n", .{});
    }
    // qualification_9e49592e: TF_AFD_LANES=N, a prompt of 2+ chunks in lanes of up to N: lanes 1..N-1's copies placed after every other
    // buffer (the settings both ends would agree carry lanes=N), their rings of host zeros with --audit; unset, nothing
    var ln: glm.afd_lanes.Lanes = undefined;
    const per_ring = ring.total(a.positions());
    const lring_bytes = try gpa.alloc(u8, if (lanes_n > 0 and f.audit) (lanes_n - 1) * per_ring else 0);
    defer gpa.free(lring_bytes);
    if (lanes_n > 0) {
        a.paired = true;
        a.inflight = lanes_n;
        a.lanes = lanes_n;
        ln = try glm.afd_lanes.Lanes.init(&at);
        ln.agreed = true;
        try ln.walk(&space, trace.placeAll);
        if (f.audit) {
            @memset(lring_bytes, 0);
            for (ln.rings[0 .. lanes_n - 1], 0..) |*rg, k| {
                var r = afd.Ring.of(a.layout, e.buf.rows, null);
                r.bytes = lring_bytes[k * per_ring ..][0..per_ring];
                rg.* = r;
            }
        }
        at.lanes = &ln;
        std.debug.print("INFO attn --dry: TF_AFD_LANES={d}: a prompt of 2+ chunks prefills in lanes of up to {d}, {d} requests in flight a link (lanes 1-{d}'s copies {d} B)\n", .{ lanes_n, lanes_n, lanes_n, lanes_n - 1, try ln.bytes() });
    }
    var tr = trace.Trace.init(gpa);
    defer tr.deinit();
    tr.picker = dry_picker; // qualification_18588626: null unless afd_cover_qmm.zig set it
    var fw = glm.fwd.Fwd.recording(&m, &tr);
    // the fwd.Kernels the GPU path would pass (afd_attn_gpu.drySet, at stand-ins): refused as it would refuse
    if (gpu.images) {
        const k = gpu.drySet() catch |err| return .{ .outcome = .{ .usage = @errorName(err) }, .forwards = 0, .launches = 0, .exchanges = 0, .collects = 0, .records = 0 };
        std.debug.print("INFO attn --dry: the GPU path's fwd.Kernels at stand-ins: {s}\n", .{gpu.describe(k)});
    } else std.debug.print("INFO attn --dry: a host build (no kernel images): its GPU verbs refuse, so no fwd.Kernels\n", .{});
    const out = f.out orelse ".";
    const r: Run = .{ .gpa = gpa, .at = &at, .f = &fw, .stream = null, .out = out, .t = t };
    const res: anyerror!usize = switch (f.verb) {
        .first => if (first(r)) |x| @as(usize, x) else |err| err,
        .teacher => teacher(r, f.steps),
        .bench => bench.run(r, f.bench, f.prompt.?, 0),
        else => error.BadVerb,
    };
    try fw.flush();
    if (dry_sink) |s| try s.f(s.ctx, tr.events.items); // qualification_18588626: null unless afd_cover_qmm.zig set it
    var n = [3]usize{ 0, 0, 0 };
    var x = [2]usize{ 0, 0 };
    for (tr.events.items) |ev| {
        n[@backingInt(ev.op)] += 1;
        if (std.mem.eql(u8, ev.name, "afd.exchange")) x[0] += 1;
        if (std.mem.eql(u8, ev.name, "afd.collect")) x[1] += 1;
    }
    std.debug.print("INFO attn {t} --dry: {d} layers ({d} MoE), capacity {d}, chunks of {d}: {d} forwards, {d} launches recorded (triton {d}, ext {d}, comm {d}), {d} exchanges, {d} collects\n", .{ f.verb, m.layers, a.layers, f.capacity, f.max_rows, at.forwards, tr.events.items.len, n[0], n[1], n[2], x[0], x[1] });
    if (graphed) {
        var gb: [320]u8 = undefined;
        std.debug.print("INFO attn --dry: {s}\n", .{gr.line(&gb)});
    }
    if (f.audit) try writeRecords(&recs, out);
    return .{ .outcome = outcome(gpa, &at, res, @tagName(f.verb)), .forwards = at.forwards, .launches = tr.events.items.len, .exchanges = x[0], .collects = x[1], .records = recs.forwards, .graphs = gr.stats };
}

/// --audit's attention records, OUT/afd-check-attn.jsonl (check-audit's first file).
pub fn writeRecords(recs: *const Records, out: []const u8) !void {
    var pb: [1024]u8 = undefined;
    try mkdirs(out);
    try writeFile(try std.fmt.bufPrint(&pb, "{s}/afd-check-attn.jsonl", .{out}), recs.text.items);
    std.debug.print("INFO attn --audit: {d} forwards' records in {s}/afd-check-attn.jsonl\n", .{ recs.forwards, out });
}

// -- check-weights and the GPU ---------------------------------------------------------------------------------------

/// `attn check-weights`: qualification_9051d0c1's world-1 plan (no MTP, the routed experts' groups emptied) over the checkpoint's headers
/// through lane B's EXL3 pack pass (afd_load.load), bound at stand-ins with no view at address 0 (G-load-A): the bytes
/// read and the packed buffers' device bytes. --dry stops there; on a GPU afd_attn_gpu reads and packs them.
pub fn checkWeights(gpa: Allocator, io: std.Io, f: cli.Flags) !cli.Outcome {
    var ix = try glm.index.open(gpa, f.model);
    defer ix.deinit();
    var c = try files.config(gpa, f.model);
    if (f.layers) |n| c.run = @min(n, c.layers);
    var name: [4096]u8 = undefined;
    const config = try files.readFile(gpa, try std.fmt.bufPrint(&name, "{s}/config.json", .{f.model}));
    defer gpa.free(config);
    var m = try drafted.model(gpa, config, if (c.run < c.layers) c.run else null); // qualification_21e21bc0: MTP with TF_AFD_MTP=1
    var ld = try glm.afd_load.loadWith(gpa, &m, &c, &ix.names, m.mtp_layers > 0);
    defer ld.deinit();
    const p = &ld.x;
    const src = p.sourceBytes(); // qualification_25364d42's plan, before the dense-q4 form appends its files
    const mtp_bytes = if (m.mtp_layers > 0) try glm.afd_load.mtpBytes(&c, &ix.names) else 0;
    if (c.run == c.layers and src != glm.afd_load.attention_bytes + mtp_bytes) return .{ .usage = "plan-bytes-not-qualification_99ec72b7" };
    const d = try glm.afd_load.dense(gpa, &ld, &ix, &m);
    _ = try ld.stand(1 << 36);
    var w = try glm.views.rank(gpa, &m);
    defer w.deinit(gpa);
    const bound = try glm.afd_load.bind(&w, &ld.pk);
    std.debug.print("INFO attn check-weights: the world-1 plan, {d} of {d} layers: {d} B read in {d} copies, {d} B in {d} groups (the routed experts' emptied, no MTP); packed: {d} buffers, {d} B on the device, {d} views bound at stand-ins, 0 unbound (the draft head's 3 among them)\n", .{ c.run, c.layers, p.sourceBytes(), p.copies.items.len, p.bytes(), p.groups.items.len, ld.pk.items.items.len, ld.deviceBytes(), bound });
    if (m.mtp_layers > 0) std.debug.print("INFO attn check-weights: TF_AFD_MTP=1: the plan above has the MTP layer too, {d} B of it read (layers.{d}, its routed experts the halves')\n", .{ mtp_bytes, c.layers });
    denseLine(d, "check-weights");
    if (!f.dry) { // the GPU path loads again, over an index the overlay has not touched
        var ix2 = try glm.index.open(gpa, f.model);
        defer ix2.deinit();
        return gpu.checkWeights(gpa, io, f, &m, &c, &ix2);
    }
    return .{ .ok = std.fmt.allocPrint(gpa, "check-weights --dry {d} B read, {d} B packed, {d} views bound{s}{s}", .{ p.sourceBytes(), ld.deviceBytes(), bound, if (d == null) "" else if (d.?.f8_saved == null) ", dense q4" else ", dense q4, FP8 head", if (d != null and d.?.mtp8 != null) ", FP8 MTP" else "" }) catch "check-weights --dry" };
}

/// The dense form's INFO line (qualification_e5a7fc3b): TF_GLM_DENSE=q4:DIR's triplets and TF_GLM_F8=head's saving, or BF16; mtp_fp8's
/// line for the MTP layer on FP8 only when TF_GLM_MTP_FP8=1 made it.
pub fn denseLine(d: ?glm.afd_load.Dense, what: []const u8) void {
    const x = d orelse return std.debug.print("INFO attn {s}: dense BF16 (TF_GLM_DENSE unset), lm_head BF16\n", .{what});
    std.debug.print("INFO attn {s}: dense q4: {d} projections from {d} files ({d} B stored); lm_head {s} ({d} B less)\n", .{ what, x.q4.triplets, x.q4.files, x.q4.bytes, if (x.f8_saved != null) "FP8 (TF_GLM_F8=head)" else "BF16", x.f8_saved orelse 0 });
    if (x.mtp8) |s| std.debug.print("INFO attn {s}: TF_GLM_MTP_FP8=1: the MTP layer's {d} projections on block FP8, {d} B (BF16 {d} B), drafts only; q4-mtp not read\n", .{ what, s.projections, s.f8, s.bf16 });
}

/// --dry over a checkpoint (qualification_e5a7fc3b, the full-depth dry): the views as the GPU path binds them, the world-1 load with
/// the dense form the switches pick, every view bound at stand-ins (none left at 0) from 64 GiB up; returns the next
/// free address. Null without the checkpoint's index (a bare config.json, as the tests' fixture): stand-ins for all.
fn bindDry(gpa: Allocator, f: cli.Flags, m: *const glm.model.Model, w: *glm.views.Weights) !?u64 {
    var name: [4096]u8 = undefined;
    const idx = try std.fmt.bufPrint(&name, "{s}/model.safetensors.index.json", .{f.model});
    var cb: [4096]u8 = undefined;
    if (std.c.access(try cpath(&cb, idx), 0) != 0) return null;
    var ix = try glm.index.open(gpa, f.model);
    defer ix.deinit();
    var c = try files.config(gpa, f.model);
    c.run = @intCast(m.layers);
    var ld = try glm.afd_load.loadWith(gpa, m, &c, &ix.names, m.mtp_layers > 0);
    defer ld.deinit();
    const d = try glm.afd_load.dense(gpa, &ld, &ix, m);
    const top = try ld.stand(1 << 36);
    const bound = try glm.afd_load.bind(w, &ld.pk);
    std.debug.print("INFO attn --dry: the views as the GPU path binds them: the world-1 load ({d} layers), {d} packed buffers, {d} B on the device, {d} views bound at stand-ins, 0 unbound\n", .{ m.layers, ld.pk.items.items.len, ld.deviceBytes(), bound });
    denseLine(d, "--dry");
    return top;
}

/// The `attn` role: check-weights, or first/teacher (--dry: recorded on a CPU; else afd_attn_gpu's GPU path).
pub fn run(gpa: Allocator, io: std.Io, f: cli.Flags) cli.Outcome {
    return go(gpa, io, f) catch |err| .{ .usage = @errorName(err) };
}

fn go(gpa: Allocator, io: std.Io, f: cli.Flags) !cli.Outcome {
    if (f.verb == .@"check-weights") return checkWeights(gpa, io, f);
    if (f.verb == .serve) { // qualification_4be15bbf: no prompt; --dry is the RTX 5090's forecast for the form (afd_attn_serve.zig)
        if (!f.dry) return gpu.run(gpa, io, f, null);
        var cn: [4096]u8 = undefined;
        const text = try files.readFile(gpa, try std.fmt.bufPrint(&cn, "{s}/config.json", .{f.model}));
        defer gpa.free(text);
        return serve.dry(gpa, f, text);
    }
    var pb: [4096]u8 = undefined; // bench: --prompt FILE, the decode prompt by the spec's name
    var t = try tokens(gpa, if (f.verb == .bench) try bench.decodePrompt(&pb, f.prompt.?, f.bench) else f.prompt.?, f.teacher);
    defer t.deinit(gpa);
    if (!f.dry) return gpu.run(gpa, io, f, &t);
    var name: [4096]u8 = undefined;
    const config = try files.readFile(gpa, try std.fmt.bufPrint(&name, "{s}/config.json", .{f.model}));
    defer gpa.free(config);
    return (try dry(gpa, f, &t, config)).outcome;
}

// -- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;

fn tmpPath(a: Allocator, tmp: testing.TmpDir) ![]u8 {
    // std.testing.tmpDir parents the dir at <cwd>/.zig-cache/tmp
    const cwd = try std.process.currentPathAlloc(testing.io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

/// The kit's fixture model (python 2.15's config.json for 078455ff).
fn fixtureConfig(a: Allocator) ![]const u8 {
    const root = try std.json.parseFromSliceLeaky(J, a, @embedFile("afd_attn_fixture"), .{});
    return root.object.get("config_json").?.string;
}

test "attn teacher --dry --audit through qualification_0fa2bee9's engine (layers 0-4): a chunk then 3 steps, steps.json, a record a step, check-audit" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try tmpPath(a, tmp);
    var prompt: [37]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(100 + i);
    var serial = [_]u32{ 7, 8, 9, 10 };
    const t: Tokens = .{ .name = "t37", .prompt = &prompt, .serial = &serial };
    const f: cli.Flags = .{ .role = .attn, .verb = .teacher, .model = "-", .layers = 5, .out = out, .audit = true, .dry = true, .steps = 48, .prompt = "-", .teacher = "-" };
    const d = try dry(testing.allocator, f, &t, try fixtureConfig(a));
    defer testing.allocator.free(d.outcome.ok);
    try testing.expectEqualStrings("teacher 3", d.outcome.ok); // 4 serial tokens: 3 steps (toks.len - 1)
    try testing.expectEqual(@as(u64, 4), d.forwards); // the 37-row chunk, then 3 decode steps
    try testing.expectEqual(@as(usize, 8), d.exchanges); // 4 forwards x layers 3, 4 (MoE)
    try testing.expectEqual(d.exchanges, d.collects);
    try testing.expectEqual(@as(usize, 3), d.records); // the chunk is wider than the ring's 8 rows
    const steps_json = try files.readFile(a, try std.fmt.allocPrint(a, "{s}/teacher/t37/steps.json", .{out}));
    const sj = try std.json.parseFromSliceLeaky(J, a, steps_json, .{});
    const steps = sj.object.get("steps").?.array.items;
    try testing.expectEqual(@as(usize, 3), steps.len);
    for (steps, 0..) |s, k| {
        try testing.expectEqual(@as(i64, @intCast(37 + k)), s.object.get("position").?.integer);
        try testing.expectEqual(@as(usize, 5), s.object.get("shard_top5").?.array.items.len);
        try testing.expectEqual(@as(usize, 0), s.object.get("mtp_draws").?.array.items.len);
    }
    // the experts' records of the same (zero) bytes compare clean; one reply bit flipped is python's one mismatch
    const proto = cuda.xchg.proto;
    var m = try glm.model.Model.parse(a, try fixtureConfig(a), 1, 0);
    try m.cut(5);
    const l: proto.Layout = .{ .d = m.hidden, .s = m.top_k + 1 };
    const zero = try a.alloc(u8, proto.wireBytes(l.req(1)));
    @memset(zero, 0);
    const rep = try a.alloc(u8, 4 * l.d); // a reply of 1 row, fp32
    @memset(rep, 0);
    var paths: [3][:0]const u8 = .{ try std.fmt.allocPrintSentinel(a, "{s}/afd-check-attn.jsonl", .{out}, 0), "", "" };
    for ([_]audit.Side{ .e0, .e1 }, 1..) |side, k| {
        var text: std.ArrayList(u8) = .empty;
        for (2..5) |seq| {
            var mine = [_]audit.Layer{ audit.expertLayer(l, 1, zero, rep), audit.expertLayer(l, 1, zero, rep) };
            if (side == .e1 and seq == 3) mine[1].out[0][0] ^= 1;
            try audit.writeRecord(a, &text, side, @intCast(seq), 1, &mine);
        }
        paths[k] = try std.fmt.allocPrintSentinel(a, "{s}/afd-check-{t}.jsonl", .{ out, side }, 0);
        try writeFile(paths[k], text.items);
    }
    const w = try checkAudit(testing.allocator, .{ paths[0], paths[1], paths[2] });
    try testing.expectEqual(@as(u64, 3), w.forwards);
    try testing.expectEqual(@as(u64, 6), w.layers);
    try testing.expectEqual(@as(u64, 1), w.mismatches);
    try testing.expectEqualStrings("layer 1 peer 1 out", w.first()[0..18]);
    try testing.expectEqualStrings("audit-mismatch", checkAuditCmd(a, &paths).usage);
    try testing.expectEqualStrings("check-audit needs ATTN E0 E1", checkAuditCmd(a, paths[0..2]).usage);
    try testing.expectEqualStrings("WrongSide", checkAuditCmd(a, &.{ paths[1], paths[1], paths[2] }).usage);
}

test "attn first --dry: a 37-token prompt's first token; a 2,049-token one's too, through layer 3's 2,048-row chunk (A4)" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try tmpPath(a, tmp);
    const prompt = try a.alloc(u32, 2049);
    for (prompt, 0..) |*p, i| p.* = @intCast(i % 1000);
    var f: cli.Flags = .{ .role = .attn, .verb = .first, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = "-" };
    const short = try dry(testing.allocator, f, &.{ .name = "s", .prompt = prompt[0..37] }, try fixtureConfig(a));
    defer testing.allocator.free(short.outcome.ok);
    try testing.expectEqualStrings("first 0", short.outcome.ok); // nothing ran: logits of zeros, column 0
    try testing.expectEqual(@as(u64, 1), short.forwards);
    _ = try files.readFile(a, try std.fmt.allocPrint(a, "{s}/first-s-attn.json", .{out}));
    f.audit = true;
    const long = try dry(testing.allocator, f, &.{ .name = "l", .prompt = prompt }, try fixtureConfig(a));
    defer testing.allocator.free(long.outcome.ok);
    try testing.expectEqualStrings("first 0", long.outcome.ok);
    try testing.expectEqual(@as(u64, 2), long.forwards); // chunks of 2,048 rows and 1 row
    try testing.expectEqual(@as(usize, 4), long.exchanges); // 2 forwards x layers 3, 4 (MoE)
    try testing.expectEqual(long.exchanges, long.collects);
    try testing.expectEqual(@as(usize, 1), long.records); // the 1-row chunk's; the 2,048-row one is wider than the ring
    _ = try files.readFile(a, try std.fmt.allocPrint(a, "{s}/first-l-attn.json", .{out}));
}

test "G-dry-long (qualification_cd073cc0): attn first --dry on a 5,000-token prompt at --capacity 6152 (LZ2b's prompt B: chunks 2,048 @ 0, 2,048 @ 2,048, 904 @ 4,096) runs to first 0" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try tmpPath(a, tmp);
    const prompt = try a.alloc(u32, 5000);
    for (prompt, 0..) |*p, i| p.* = @intCast((7 * i) % 1000);
    const f: cli.Flags = .{ .role = .attn, .verb = .first, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = "-", .capacity = 6152 };
    const d = try dry(testing.allocator, f, &.{ .name = "B", .prompt = prompt }, try fixtureConfig(a));
    if (d.outcome != .ok) return error.TestUnexpectedResult; // its INFO line says why
    defer testing.allocator.free(d.outcome.ok);
    try testing.expectEqualStrings("first 0", d.outcome.ok);
    try testing.expectEqual(@as(u64, 3), d.forwards);
    try testing.expectEqual(@as(usize, 6), d.exchanges); // 3 forwards x layers 3, 4 (MoE)
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "qualification_0f991257: attn first --dry --audit with TF_AFD_PAIRS=1 on G-dry-long's prompt (a pair of 2,048 + 2,048, then 904 alone): its first token, forwards and exchanges as unset's, one launch more (B's position); TF_AFD_PAIRS=2 refused before anything loads" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const out = try tmpPath(a, tmp);
    const prompt = try a.alloc(u32, 5000);
    for (prompt, 0..) |*p, i| p.* = @intCast((7 * i) % 1000);
    const f: cli.Flags = .{ .role = .attn, .verb = .first, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = "-", .capacity = 6152, .audit = true };
    const off = try dry(testing.allocator, f, &.{ .name = "B", .prompt = prompt }, try fixtureConfig(a));
    defer testing.allocator.free(off.outcome.ok);
    _ = setenv("TF_AFD_PAIRS", "1", 1);
    defer _ = unsetenv("TF_AFD_PAIRS");
    const on = try dry(testing.allocator, f, &.{ .name = "B", .prompt = prompt }, try fixtureConfig(a));
    defer testing.allocator.free(on.outcome.ok);
    try testing.expectEqualStrings(off.outcome.ok, on.outcome.ok);
    try testing.expectEqual(off.forwards, on.forwards);
    try testing.expectEqual(@as(usize, 6), on.exchanges);
    try testing.expectEqual(on.exchanges, on.collects);
    try testing.expectEqual(off.launches + 1, on.launches);
    try testing.expectEqual(off.records, on.records); // every window here is wider than the 8-row ring
    _ = setenv("TF_AFD_PAIRS", "2", 1);
    try testing.expectError(error.BadAfdPairs, dry(testing.allocator, f, &.{ .name = "B", .prompt = prompt }, try fixtureConfig(a)));
}

test "the teacher's tokens from qualification_6607f886's steps.json or qualification_1cad2b94's kit teacher.json (FILE:KEY, LZ2b's prompts A and B)" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmpPath(a, tmp);
    const pf = try std.fmt.allocPrint(a, "{s}/prompts.json", .{dir});
    try writeFile(pf, "{\"prompts\": {\"A\": [5, 6, 7], \"B\": [8, 9]}}");
    const kit = try std.fmt.allocPrint(a, "{s}/teacher.json", .{dir});
    try writeFile(kit, "{\"short\": {\"first\": 11, \"inputs\": [11, 12, 13], \"next\": [12, 13, 14], \"positions\": [3, 4, 5]}}");
    const qualification_6607f886 = try std.fmt.allocPrint(a, "{s}/steps.json", .{dir});
    try writeFile(qualification_6607f886, "{\"steps\": [{\"input\": 11, \"next_serial\": 12}, {\"input\": 12, \"next_serial\": 13}]}");
    var t = try tokens(testing.allocator, try std.fmt.allocPrint(a, "{s}:A", .{pf}), try std.fmt.allocPrint(a, "{s}:short", .{kit}));
    defer t.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, t.prompt);
    try testing.expectEqualSlices(u32, &.{ 11, 12, 13, 14 }, t.serial);
    var u = try tokens(testing.allocator, try std.fmt.allocPrint(a, "{s}:B", .{pf}), qualification_6607f886);
    defer u.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 11, 12, 13 }, u.serial);
    try testing.expectError(error.NoTeacherPrompt, tokens(testing.allocator, try std.fmt.allocPrint(a, "{s}:A", .{pf}), kit)); // KEY: the prompt's name, A
}
