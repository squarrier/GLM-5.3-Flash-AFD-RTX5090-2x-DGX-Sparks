//! tf-glm-forward's `--drafts` (qualification_qualification_44ba554e.7 day 2): engine.py's request policies (encode_policy and decode_policy, ed78d6f
//! lines 24-75) for bench's decode runs: 0 (serial_decode), N or fN (N MTP drafts, fixed; `f` is qualification_qualification_86ea5170's name for fixed
//! here, never DFlash2), a or a:LOW:HIGH (adaptive, 1-3), cN:P (N fixed, under a confidence floor P; draft_policy's cN:P:F adds
//! the first draft's floor F), and auto (drafter_choice.auto_decode's MTP arm with engine.py's m_policy: greedy c3:0.35,
//! or TF_GLM_DRAFT_AUTO's cN:P[:F] (draft_policy), sampled a 0.6/0.85); N is capped at MAX_ROWS - 1 as decode_policy caps it. The
//! TIME line carries cuda_decode's counters (rounds, drafted, accepted, the
//! acceptance, tokens a verify forward, depths and keeps by size, MTP steps, the nucleus gathers) and python's four
//! stage times. On --dry a picker logs every draw (kind, positions), answered as a recording's (token 0, probability
//! 0), for G-D2-dry: python-0.6's loops on the same answers give the same draws and counters.
const std = @import("std");
const glm = @import("glm_cuda");

const decode = glm.decode;
const Engine = glm.engine.Engine;
const Fwd = glm.fwd.Fwd;
const T = glm.trace.T;

/// A request's draft policy.
pub const Drafts = struct {
    kind: enum { serial, mtp, auto } = .serial,
    policy: decode.DepthPolicy = .{},
    spec: []const u8 = "0",

    pub fn on(d: Drafts) bool {
        return d.kind != .serial;
    }
};

/// engine.encode_policy's forms, as decode_policy reads them (no DFlash2: N and fN are the same fixed policy).
pub fn parse(spec: []const u8) !Drafts {
    const s = std.mem.trim(u8, spec, " ");
    const cap = decode.max_rows - 1;
    if (s.len == 0 or std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "f0")) return .{};
    if (std.mem.eql(u8, s, "auto")) {
        _ = try decode.autoEnv(); // draft_policy: TF_GLM_DRAFT_AUTO, refused here (before any load) when malformed
        return .{ .kind = .auto, .spec = s };
    }
    if (s[0] == 'a') {
        if (s.len == 1) return .{ .kind = .mtp, .spec = s };
        var it = std.mem.splitScalar(u8, s, ':');
        if (!std.mem.eql(u8, it.next().?, "a")) return error.Drafts;
        const low = try std.fmt.parseFloat(f64, it.next() orelse return error.Drafts);
        const high = try std.fmt.parseFloat(f64, it.next() orelse return error.Drafts);
        if (it.next() != null) return error.Drafts;
        return .{ .kind = .mtp, .policy = .{ .low = low, .high = high }, .spec = s };
    }
    if (s[0] == 'c') return .{ .kind = .mtp, .policy = try decode.parseChain(s), .spec = s }; // cN:P, draft_policy's cN:P:F
    const most = try std.fmt.parseInt(usize, if (s[0] == 'f') s[1..] else s, 10);
    if (most == 0) return .{};
    return .{ .kind = .mtp, .policy = .{ .most = @min(most, cap), .fixed = true }, .spec = s };
}

/// --drafts A+B+... (qualification_qualification_4ab1ea8f): each form in turn, with its record's directory: OUT/F (':' as '-'), or OUT for one form.
pub const Forms = struct {
    it: std.mem.TokenIterator(u8, .scalar),
    out: []const u8,
    several: bool,
    buf: [1024]u8 = undefined,

    pub const Form = struct { form: []const u8, out: []const u8 };

    pub fn of(spec: []const u8, out: []const u8) Forms {
        const s = if (std.mem.trim(u8, spec, " ").len == 0) "0" else spec;
        return .{ .it = std.mem.tokenizeScalar(u8, s, '+'), .out = out, .several = std.mem.indexOfScalar(u8, s, '+') != null };
    }

    pub fn next(fs: *Forms) !?Form {
        const form = fs.it.next() orelse return null;
        if (!fs.several) return .{ .form = form, .out = fs.out };
        const dir = try std.fmt.bufPrint(&fs.buf, "{s}/{s}", .{ fs.out, form });
        std.mem.replaceScalar(u8, dir[fs.out.len + 1 ..], ':', '-');
        return .{ .form = form, .out = dir };
    }
};

/// One decode of `count` tokens from `pending` by the policy's loop (python's, no eos stop: bench's fixed length).
pub fn run(e: *Engine, f: *Fwd, d: Drafts, pending: u32, count: usize, out: []u32, log: ?*decode.Decoded.Log) !decode.Decoded {
    return runWith(e, f, d, pending, count, out, log, null);
}

/// run with a served stream's round hook (qualification_qualification_44ba554e.8 day 3, native/glm_cuda_drafts.zig; null: run itself).
pub fn runWith(e: *Engine, f: *Fwd, d: Drafts, pending: u32, count: usize, out: []u32, log: ?*decode.Decoded.Log, hook: ?*const decode.Hook) !decode.Decoded {
    var policy = d.policy;
    return switch (d.kind) {
        .serial => decode.serial(e, f, pending, count, out, false),
        .mtp => decode.mtpWith(e, f, pending, count, out, &policy, false, log, hook),
        .auto => blk: {
            policy = decode.DepthPolicy.auto(e.sampling == null or e.sampling.?.temperature <= 0.0);
            break :blk decode.autoWith(e, f, pending, count, out, &policy, false, log, hook);
        },
    };
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// The TIME line of a drafted decode run (stderr).
pub fn timeLine(R: u32, ri: usize, label: []const u8, d: Drafts, r: *const decode.Decoded, from: usize) void {
    const wall = ms(r.ns);
    const n = r.tokens.len -| 1;
    std.debug.print("TIME rank {d}: bench run {d} ({s}): decode of {d} tokens from position {d}, drafts {s}: {d:.1} ms, {d:.2} tok/s; rounds {d}, drafted {d}, accepted {d}, acceptance {d:.4}, {d:.3} tokens a verify forward, MTP steps {d}; depths {any}; keeps {any}; stages: draft {d:.1} ms, forward {d:.1} ms, sample {d:.1} ms, commit {d:.1} ms; nucleus gathers {any}\n", .{ R, ri, label, n, from, d.spec, wall, @as(f64, @floatFromInt(n)) * 1e3 / wall, r.rounds, r.drafted, r.accepted, r.acceptance(), r.perForward(), r.mtp_steps, r.depths, r.keeps, ms(r.stages[0]), ms(r.stages[1]), ms(r.stages[2]), ms(r.stages[3]), r.gathers });
}

fn list(js: *std.ArrayList(u8), gpa: std.mem.Allocator, xs: anytype) !void {
    try js.append(gpa, '[');
    for (xs, 0..) |x, i| try js.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", x });
    try js.append(gpa, ']');
}

/// The sequence's state right after a decode (bench's next prefill resets it): python's st.pos, mtp_len, mtp_drafted.
pub const After = struct {
    pos: usize,
    mtp_len: usize,
    mtp_drafted: usize,

    pub fn of(e: *const Engine) After {
        return .{ .pos = e.st.pos, .mtp_len = e.st.mtp_len, .mtp_drafted = e.st.mtp_drafted };
    }
};

/// The run's counters, stage times, state after it and each round's drafts for OUT/bench-rankR.json.
pub fn record(js: *std.ArrayList(u8), gpa: std.mem.Allocator, st: After, d: Drafts, r: *const decode.Decoded) !void {
    try js.print(gpa, ", \"drafts\": {{\"spec\": \"{s}\", \"rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"acceptance\": {d:.6}, \"per_forward\": {d:.6}, \"mtp_steps\": {d}, \"wall_ms\": {d:.3}, \"stages_ms\": {{\"draft\": {d:.3}, \"forward\": {d:.3}, \"sample\": {d:.3}, \"commit\": {d:.3}}}, \"pos\": {d}, \"mtp_len\": {d}, \"mtp_drafted\": {d}, \"pending\": {d}, \"depths\": ", .{ d.spec, r.rounds, r.drafted, r.accepted, r.acceptance(), r.perForward(), r.mtp_steps, ms(r.ns), ms(r.stages[0]), ms(r.stages[1]), ms(r.stages[2]), ms(r.stages[3]), st.pos, st.mtp_len, st.mtp_drafted, r.pending });
    try list(js, gpa, r.depths);
    try js.appendSlice(gpa, ", \"keeps\": ");
    try list(js, gpa, r.keeps);
    try js.appendSlice(gpa, ", \"gathers\": ");
    try list(js, gpa, r.gathers);
    try js.appendSlice(gpa, ", \"round_drafts\": [");
    if (r.log) |l| for (0..l.rounds) |i| {
        const end = if (i + 1 < l.rounds) l.at[i + 1] else l.n;
        if (i > 0) try js.appendSlice(gpa, ", ");
        try list(js, gpa, l.drafts[l.at[i]..end]);
    };
    try js.appendSlice(gpa, "]");
    // draft_policy: the policy's chain as run (the benches decode greedy: auto is DepthPolicy.auto(true), TF_GLM_DRAFT_AUTO's
    // when set) and each draft's probability (null: not read)
    const pol = if (d.kind == .auto) decode.DepthPolicy.auto(true) else d.policy;
    try js.print(gpa, ", \"chain\": {{\"most\": {d}, \"confidence\": {d}, \"first\": {d}}}", .{ pol.most, pol.confidence, pol.first });
    if (r.log) |l| if (l.probs) |ps| {
        try js.appendSlice(gpa, ", \"round_probs\": [");
        for (0..l.rounds) |i| {
            const end = if (i + 1 < l.rounds) l.at[i + 1] else l.n;
            try js.appendSlice(gpa, if (i > 0) ", [" else "[");
            for (ps[l.at[i]..end], 0..) |p, j| if (std.math.isNan(p)) try js.print(gpa, "{s}null", .{if (j == 0) "" else ", "}) else try js.print(gpa, "{s}{e}", .{ if (j == 0) "" else ", ", p });
            try js.append(gpa, ']');
        }
        try js.append(gpa, ']');
    };
    try js.append(gpa, '}');
}

/// On --dry: each draw's kind and positions (python's: a verify window's st.pos + 1 + r, draft j's position + j, a
/// sampled draw's own), answered as a recording's.
pub const DryLog = struct {
    e: *const Engine,
    gpa: std.mem.Allocator,
    js: std.ArrayList(u8) = .empty,
    n: usize = 0,

    pub fn pick(ctx: *anyopaque, logits: T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        const l: *DryLog = @ptrCast(@alignCast(ctx));
        const e = l.e;
        const st = &e.st;
        const kind: enum { first, main, draft } = if (logits.ptr == e.buf.logits.ptr) .main else if (e.mbuf != null and logits.ptr == e.mbuf.?.logits.ptr) .draft else .first;
        try l.js.print(l.gpa, "{s}{{\"kind\": \"{t}\", \"positions\": [", .{ if (l.n == 0) "" else ", ", kind });
        for (out, 0..) |*o, r| {
            const p: usize = if (at) |x| @intCast(x[r]) else switch (kind) {
                .first => st.pos,
                .main => st.pos + 1 + r,
                .draft => st.pos + 1 + st.mtp_drafted,
            };
            try l.js.print(l.gpa, "{s}{d}", .{ if (r == 0) "" else ", ", p });
            o.* = 0;
        }
        try l.js.appendSlice(l.gpa, "]}");
        if (probs) |x| @memset(x, 0);
        l.n += 1;
    }
};
