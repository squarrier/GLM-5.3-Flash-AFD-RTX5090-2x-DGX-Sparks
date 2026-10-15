//! user_sampling (local integration; Claude Opus 5.5 via Hermes): test-user_sampling_served — the users server entry (the verb's
//! `afd_attn_serve.run`, which hands TF_AFD_USERS > 1 to `afd_attn_users.run`) and the single-user entry over the REAL
//! GLM-5.3 tokenizer and grammar_abi's libtfgrammar, the recorded proxy engine and a MOCK keyed draw
//! (user_sampling_served_rig.zig): sampled requests with TF_AFD_USERS_SAMPLING=1 at USERS=2 and USERS=4 (pool + interleave),
//! seeded identity alone / beside 1 / beside 3 (greedy, other seeds, structured) and against the single-user boot S
//! (scripts/check_served.py, from each case's user_sampling_OUT record), greedy rows unchanged with the switch on and off,
//! structured + sampled replies valid, sampling_tests_stat's sampling gates 1-11 in their CPU form per row, a client gone
//! mid-stream and a device fault mid-round. One boot per process (user_sampling_CASE: s, u2, u4, u2off, u4off, fail); the
//! startup refusals run without a boot. Every case records first and asserts last, so a RED run keeps its evidence.
//! Rig and requests: structured_entry's served test (Claude Opus 5.5 via Hermes) after concurrent_users_r2 (GPT-6.1 Sol via Hermes) and Ash
//! Hart's tests; sampling_tests_stat's client.py matrix (its bodies); NOTICE credits apply (Ash Hart / TensorFold; Mia
//! (@MiaAI_lab), whose union cover qualification_45c10bcf's draw carries; the XGrammar contributors (MLC, Apache-2.0)).
const std = @import("std");
const rig = @import("user_sampling_served_rig.zig");
const hx = @import("user_sampling_served_http.zig");
const served = @import("afd_attn_serve.zig");
const cli = @import("tf_glm_afd.zig");
const t = std.testing;
const gpa = t.allocator;
const Req = hx.Req;
const Reply = hx.Reply;
const W = [2][:0]const u8;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

pub const refusal = "TF_AFD_USERS supports temperature 0 only; sampled multi-user decode is unsupported";

// ---- the boots ----------------------------------------------------------------------------------------------------------

const single = [_]W{ .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_GRAMMAR_MASK", "1" }, .{ "TF_AFD_GRAMMAR_SERVE", "1" } };
const on2 = [_]W{ .{ "TF_AFD_USERS", "2" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "1" }, .{ "TF_AFD_GRAMMAR_MASK", "1" }, .{ "TF_AFD_GRAMMAR_SERVE", "1" } };
const off2 = [_]W{ .{ "TF_AFD_USERS", "2" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_GRAMMAR_MASK", "1" }, .{ "TF_AFD_GRAMMAR_SERVE", "1" } };
const on4 = [_]W{ .{ "TF_AFD_USERS", "4" }, .{ "TF_AFD_POOL_TOKENS", "720896" }, .{ "TF_AFD_USERS_INTERLEAVE", "1" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "1" }, .{ "TF_AFD_GRAMMAR_MASK", "1" }, .{ "TF_AFD_GRAMMAR_SERVE", "1" } };
const off4 = [_]W{ .{ "TF_AFD_USERS", "4" }, .{ "TF_AFD_POOL_TOKENS", "720896" }, .{ "TF_AFD_USERS_INTERLEAVE", "1" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_GRAMMAR_MASK", "1" }, .{ "TF_AFD_GRAMMAR_SERVE", "1" } };

// ---- the requests -------------------------------------------------------------------------------------------------------

const keeper = "Write two sentences about a lighthouse keeper.";
const long_prompt: []const u8 = blk: {
    const s = "The keeper climbed the stairs at dusk, trimmed the wick and lit the great lamp for the ships. ";
    var out: [s.len * 12]u8 = undefined;
    for (0..12) |i| @memcpy(out[i * s.len ..][0..s.len], s);
    const final = out;
    break :blk &final;
};

/// the seeded identity set: plain sampled requests (whole and streamed, one past a prompt chunk, one unseeded)
fn idSet() [6]Req {
    return .{
        .{ .label = "id-a", .prompt = keeper, .tokens = 64, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":11" },
        .{ .label = "id-a-stream", .prompt = keeper, .tokens = 64, .stream = true, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":11" },
        .{ .label = "id-b", .prompt = "Describe the sky in one line.", .tokens = 48, .sampling = ",\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"seed\":12" },
        .{ .label = "id-c-long", .prompt = long_prompt, .tokens = 40, .stream = true, .sampling = ",\"temperature\":0.6,\"min_p\":0.05,\"seed\":13" },
        .{ .label = "id-u-unseeded", .prompt = "Name a color.", .tokens = 32, .sampling = ",\"temperature\":0.7" },
        .{ .label = "id-d", .prompt = "Count to five.", .tokens = 56, .sampling = ",\"temperature\":1.2,\"top_p\":1.0,\"seed\":14" },
    };
}

/// the neighbours a seeded request is served beside: greedy, sampled with another seed, structured (sampled)
fn neighbours() [3]Req {
    return .{
        .{ .label = "n-greedy", .prompt = "Say ok.", .tokens = 64 },
        .{ .label = "n-sampled", .prompt = "Write a haiku about rain.", .tokens = 64, .sampling = ",\"temperature\":0.9,\"top_p\":0.9,\"seed\":901" },
        .{ .label = "n-schema", .kind = .json_schema, .prompt = "Is the sky blue? Answer as JSON.", .tokens = 96, .sampling = ",\"temperature\":0.7,\"top_p\":0.95,\"seed\":902" },
    };
}

/// structured + sampled: json_schema and json_object, whole and streamed, three settings (structured_entry sp's)
fn structuredSet(a: std.mem.Allocator) ![8]Req {
    const settings = [_][]const u8{ ",\"temperature\":0.7,\"top_p\":0.95", ",\"temperature\":1.0,\"top_p\":0.95", ",\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0" };
    var out: [8]Req = undefined;
    for (&out, 0..) |*r, i| {
        const kind: hx.Kind = if (i % 2 == 0) .json_schema else .json_object;
        r.* = .{ .label = try std.fmt.allocPrint(a, "st-{d}-{s}{s}", .{ i, @tagName(kind), if ((i / 2) % 2 == 1) "-stream" else "" }), .kind = kind, .prompt = if (kind == .json_schema) "Is the sky blue? Answer as JSON." else "Give a JSON object describing the sky.", .tokens = 96, .stream = (i / 2) % 2 == 1, .sampling = try std.fmt.allocPrint(a, "{s},\"seed\":{d}", .{ settings[i % settings.len], 3000 + i }) };
    }
    return out;
}

/// greedy rows (plain and structured) for greedy = solo with the switch on and off
fn greedySet() [3]Req {
    return .{
        .{ .label = "g-plain", .prompt = keeper, .tokens = 48 },
        .{ .label = "g-plain-stream", .prompt = "Describe the sky in one line.", .tokens = 40, .stream = true },
        .{ .label = "g-object", .kind = .json_object, .prompt = "Give a JSON object describing the sky.", .tokens = 96 },
    };
}

// ---- sampling_tests_stat's client matrix (client.py --mode sampled) in its CPU form ------------------------------------------------

const samp_prompt = "Write a detailed story about a lighthouse keeper finding a message in a bottle. No title.";
const temps = [_][]const u8{ "0.6", "1.0", "0.7", "0.9", "0.8" };
/// client.py's 513 on hardware; the CPU form's reply length (the mock's work is per token, the gates are not)
pub const client_tokens = 129;

/// client.py's body(): its member set and order (seed, top_p 0.9, top_k, stream, thinking off), `temp` null omitted
fn clientBody(a: std.mem.Allocator, temp: ?[]const u8, seed: u64, top_k: u32) ![]const u8 {
    const t_part = if (temp) |x| try std.fmt.allocPrint(a, ",\"temperature\":{s}", .{x}) else "";
    return std.fmt.allocPrint(a, ",\"seed\":{d},\"top_p\":0.9,\"top_k\":{d}{s}", .{ seed, top_k, t_part });
}

const Row = struct { label: []const u8, temp: ?[]const u8, stream: bool = false, top_k: u32 = 0 };

/// the 26 rows in client.py's order
fn clientRows(a: std.mem.Allocator) ![]Row {
    var rows: std.ArrayList(Row) = .empty;
    try rows.append(a, .{ .label = "greedy-zero", .temp = "0" });
    try rows.append(a, .{ .label = "greedy-omitted", .temp = null });
    try rows.append(a, .{ .label = "greedy-stream", .temp = "0", .stream = true });
    for (0..3) |i| try rows.append(a, .{ .label = try std.fmt.allocPrint(a, "greedy-speed-{d}", .{i}), .temp = "0" });
    for (temps) |x| {
        try rows.append(a, .{ .label = try std.fmt.allocPrint(a, "sample-{s}", .{x}), .temp = x });
        try rows.append(a, .{ .label = try std.fmt.allocPrint(a, "repeat-{s}", .{x}), .temp = x });
        try rows.append(a, .{ .label = try std.fmt.allocPrint(a, "stream-{s}", .{x}), .temp = x, .stream = true });
    }
    try rows.append(a, .{ .label = "top-k-20", .temp = "0.8", .top_k = 20 });
    try rows.append(a, .{ .label = "repeat-top-k-20", .temp = "0.8", .top_k = 20 });
    for (0..3) |i| try rows.append(a, .{ .label = try std.fmt.allocPrint(a, "sample-speed-{d}", .{i}), .temp = "0.8" });
    return rows.items;
}

/// sampling_tests_stat's `same`: token_sha, content, finish and usage (reasoning: none with thinking off)
fn sameRow(x: Reply, y: Reply) bool {
    return std.mem.eql(u8, x.token_sha, y.token_sha) and std.mem.eql(u8, x.content, y.content) and std.mem.eql(u8, x.finish, y.finish) and std.mem.eql(u8, x.usage, y.usage);
}

/// client.py's request() asserts: 200, token_sha, completion_tokens > 1, seconds > prefill seconds, a finish
fn clientWellFormed(r: Reply) bool {
    return r.status == 200 and r.token_sha.len > 0 and r.completion_tokens > 1 and r.seconds > r.prefill_seconds and r.finish.len > 0;
}

/// One row's client: client.py's matrix on its own connection, sequentially (the rows run concurrently).
const Client = struct {
    k: usize,
    port: u16,
    seed: u64,
    rows: []const Row,
    replies: []Reply = &.{},
    fails: std.ArrayList([]const u8) = .empty,
    arena: std.heap.ArenaAllocator = undefined,
    err: ?anyerror = null,

    fn run(c: *Client) void {
        c.go() catch |e| {
            c.err = e;
        };
    }
    fn fail(c: *Client, comptime fmt: []const u8, args: anytype) void {
        const a = c.arena.allocator();
        const m = std.fmt.allocPrint(a, fmt, args) catch "oom";
        c.fails.append(a, std.fmt.allocPrint(a, "row {d}: {s}", .{ c.k, m }) catch "oom") catch {};
    }
    fn go(c: *Client) !void {
        const a = c.arena.allocator();
        c.replies = try a.alloc(Reply, c.rows.len);
        for (c.rows, c.replies) |row, *out| {
            const req: Req = .{ .label = try std.fmt.allocPrint(a, "c{d}-{s}", .{ c.k, row.label }), .prompt = samp_prompt, .tokens = client_tokens, .stream = row.stream, .sampling = try clientBody(a, row.temp, c.seed, row.top_k) };
            out.* = try hx.send(a, c.port, req);
            const r = out.*;
            if (!clientWellFormed(r)) c.fail("{s} not well formed (HTTP {d} {s} finish {s} completion {d})", .{ row.label, r.status, r.message, r.finish, r.completion_tokens });
            const sampled = if (row.temp) |x| !std.mem.eql(u8, x, "0") else false;
            // gate 2: native sampled counters, explicit integer zeros and sampling "exact"
            if (sampled and !hx.nativeSampled(r)) c.fail("{s} native counters drafted {?d} accepted {?d} sampling {s}", .{ row.label, r.drafted, r.accepted, r.sampling });
        }
        // gate 3: repeat and stream equality; greedy-zero = greedy-omitted = greedy-stream; top-k repeat
        const R = c.replies;
        if (!sameRow(R[0], R[1])) c.fail("greedy-zero != greedy-omitted", .{});
        if (!sameRow(R[0], R[2])) c.fail("greedy-zero != greedy-stream", .{});
        for (0..temps.len) |i| {
            const b = 6 + 3 * i;
            if (!sameRow(R[b], R[b + 1])) c.fail("sample-{s} != repeat", .{temps[i]});
            if (!sameRow(R[b], R[b + 2])) c.fail("sample-{s} != stream", .{temps[i]});
        }
        if (!sameRow(R[21], R[22])) c.fail("top-k-20 != repeat", .{});
    }
};

// ---- gates 4-6 (pairs, G-same, G-det) and gate 8 (AEON-30 greedy) in their CPU form --------------------------------------

/// pairs_probe.py's 12 greedy requests: three prompt sizes x four salted runs, thinking off, not streamed
fn pairsSet(a: std.mem.Allocator, k: usize) ![12]Req {
    var out: [12]Req = undefined;
    const sizes = [_]usize{ 1, 6, 14 };
    for (&out, 0..) |*r, i| {
        const size = sizes[i / 4];
        const run = i % 4 + 1;
        var words: std.ArrayList(u8) = .empty;
        try words.print(a, "[SALT-{d}-r{d}] ", .{ size, run });
        for (0..size) |_| try words.appendSlice(a, "The lamp turned all night over the dark water. ");
        try words.appendSlice(a, "Write a long story.");
        r.* = .{ .label = try std.fmt.allocPrint(a, "p{d}-{d}-r{d}", .{ k, size, run }), .prompt = words.items, .tokens = 33 };
    }
    return out;
}

/// AEON-30's shape in CPU form: 30 distinct greedy questions (no score: the mock has no model)
fn aeonSet(a: std.mem.Allocator) ![30]Req {
    var out: [30]Req = undefined;
    for (&out, 0..) |*r, i| r.* = .{ .label = try std.fmt.allocPrint(a, "aeon-{d:0>2}", .{i}), .prompt = try std.fmt.allocPrint(a, "Question {d}: what is {d} plus {d}? Answer briefly.", .{ i, i * 7, i + 3 }), .tokens = 17 };
    return out;
}

// ---- helpers -------------------------------------------------------------------------------------------------------------

const Log = struct {
    a: std.mem.Allocator,
    all: std.ArrayList(Reply) = .empty,
    fails: std.ArrayList([]const u8) = .empty,
    checks: usize = 0,

    fn keep(l: *Log, r: Reply) !void {
        try l.all.append(l.a, r);
    }
    fn check(l: *Log, ok: bool, comptime fmt: []const u8, args: anytype) !void {
        l.checks += 1;
        if (!ok) {
            const m = try std.fmt.allocPrint(l.a, fmt, args);
            std.debug.print("user_sampling FAIL {s}\n", .{m});
            try l.fails.append(l.a, m);
        }
    }
};

fn show(r: Reply) void {
    std.debug.print("user_sampling reply {s}: HTTP {d} finish {s} tokens {d}+{d} sha {s} sampling {s} drafted {?d} content \"{s}\"{s}{s}\n", .{ r.label, r.status, r.finish, r.prompt_tokens, r.completion_tokens, r.token_sha[0..@min(r.token_sha.len, 12)], r.sampling, r.drafted, r.content[0..@min(r.content.len, 80)], if (r.message.len > 0) " message: " else "", r.message });
}

fn relabel(a: std.mem.Allocator, r: Req, suffix: []const u8) !Req {
    var x = r;
    x.label = try std.fmt.allocPrint(a, "{s}@{s}", .{ r.label, suffix });
    return x;
}

/// the label's request part (before '@')
fn base(label: []const u8) []const u8 {
    return label[0 .. std.mem.indexOfScalar(u8, label, '@') orelse label.len];
}

fn find(rs: []const Reply, label: []const u8) ?Reply {
    for (rs) |r| if (std.mem.eql(u8, base(r.label), label)) return r;
    return null;
}

/// a sampled request's body has a positive temperature
fn sampledReq(r: Req) bool {
    const i = std.mem.indexOf(u8, r.sampling, "\"temperature\":") orelse return false;
    const rest = r.sampling[i + "\"temperature\":".len ..];
    const end = std.mem.indexOfAny(u8, rest, ",}") orelse rest.len;
    const v = std.fmt.parseFloat(f64, rest[0..end]) catch return false;
    return v > 0;
}

fn seedOf(r: Req) ?u64 {
    const i = std.mem.indexOf(u8, r.sampling, "\"seed\":") orelse return null;
    const rest = r.sampling[i + "\"seed\":".len ..];
    const end = std.mem.indexOfAny(u8, rest, ",}") orelse rest.len;
    return std.fmt.parseInt(u64, rest[0..end], 10) catch null;
}

/// qualification_45c10bcf's keys for one seed: the positions its draws took, sorted and deduplicated
fn positionsOf(a: std.mem.Allocator, draws: []const rig.Draw, seed: u64) ![]u64 {
    var out: std.ArrayList(u64) = .empty;
    for (draws) |d| if (d.kind != .greedy and d.seed == seed) {
        if (std.mem.indexOfScalar(u64, out.items, d.position) == null) try out.append(a, d.position);
    };
    std.mem.sort(u64, out.items, {}, std.sort.asc(u64));
    return out.items;
}

/// Keyed by request seed and token position only: a seeded sampled reply's draws are exactly the positions
/// prompt_tokens .. prompt_tokens + n - 1 (n its completion tokens, or one more when its last draw ended it), never
/// a slot's, a row's or a round's.
fn keysExact(a: std.mem.Allocator, draws: []const rig.Draw, r: Reply, seed: u64) !bool {
    const ps = try positionsOf(a, draws, seed);
    if (ps.len == 0) return false;
    const p0: u64 = @intCast(r.prompt_tokens);
    for (ps, 0..) |p, i| if (p != p0 + i) return false;
    const n: u64 = @intCast(r.completion_tokens);
    return ps.len == n or ps.len == n + 1;
}

/// /metrics' requests_running is 0
fn idleNow(a: std.mem.Allocator, port: u16) !bool {
    const v = (try hx.running(a, port)) orelse return false;
    return v == 0;
}

fn facts(a: std.mem.Allocator, b: *rig.Boot, l: *const Log, extra: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"users\":{d},\"port\":{d},\"checks\":{d},\"fails\":{d},\"draws_keyed\":{d},\"draws_masked_sampled\":{d},\"draws_masked_greedy\":{d},\"draws_plain\":{d},\"trace_resets\":{d},\"placed_bytes\":{d}{s}}}", .{ b.users, b.port, l.checks, l.fails.items.len, b.mock.keyed_draws, b.mock.masked_sampled, b.mock.masked_greedy, b.mock.plain_draws, b.mock.resets, b.placed_bytes, extra });
}

fn finish(a: std.mem.Allocator, case_name: []const u8, b: *rig.Boot, l: *Log, extra: []const u8) !void {
    const served_n = b.stop() catch |e| blk: {
        try l.check(false, "{s}: the boot did not close gracefully: {s}", .{ case_name, @errorName(e) });
        break :blk 0;
    };
    try l.check(true, "close", .{}); // gate 10: graceful close (stop returned)
    const f = try facts(a, b, l, try std.fmt.allocPrint(a, ",\"served\":{d}{s}", .{ served_n, extra }));
    try rig.record(a, case_name, b.words, l.all.items, f);
    std.debug.print("user_sampling {s} summary: checks {d}, fails {d}, replies {d}; facts {s}\n", .{ case_name, l.checks, l.fails.items.len, l.all.items.len, f });
    for (l.fails.items) |m| std.debug.print("user_sampling {s} FAILED: {s}\n", .{ case_name, m });
    if (l.fails.items.len > 0) return error.user_sampling_checksfailed;
}

// ---- the users boots (u2, u4: switch on) --------------------------------------------------------------------------------

fn usersOn(case_name: []const u8, words: rig.Words, n: u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var l: Log = .{ .a = a };
    var b: rig.Boot = .{};
    try b.start(words);
    var up = true;
    defer if (up) {
        _ = b.stop() catch {};
    };
    // gate 9 (CPU form): health before
    try l.check(@import("grammar_server_http.zig").status(try hx.get(a, b.port, "/health")) == 200, "health before", .{});

    // 0. greedy references first (G-same's take 1 for this boot, recorded before any sampled request)
    var greedy_solo: [3]Reply = undefined;
    for (greedySet(), &greedy_solo) |r, *s| {
        s.* = try hx.send(a, b.port, r);
        show(s.*);
        try l.keep(s.*);
    }

    // 1. RED -> GREEN: sampled requests at this user count, plain and streamed, are served (not the named 400)
    const red = [_]Req{
        .{ .label = "red-plain", .prompt = keeper, .tokens = 32, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":7001" },
        .{ .label = "red-stream", .prompt = keeper, .tokens = 32, .stream = true, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":7002" },
        .{ .label = "red-object", .kind = .json_object, .prompt = "Give a JSON object describing the sky.", .tokens = 96, .sampling = ",\"temperature\":0.7,\"top_p\":0.95,\"seed\":7003" },
        .{ .label = "red-together", .prompt = "Describe the sky in one line.", .tokens = 32, .sampling = ",\"temperature\":1.0,\"seed\":7004" },
    };
    var refused: usize = 0;
    var served_ok: usize = 0;
    for (red) |r| {
        const x = try hx.send(a, b.port, r);
        show(x);
        try l.keep(x);
        if (x.status == 400 and std.mem.eql(u8, x.message, refusal)) refused += 1;
        const ok = if (r.kind == .plain) hx.plainOk(x) else hx.structuredOk(a, x);
        if (ok and hx.nativeSampled(x)) served_ok += 1;
        std.debug.print("user_sampling RED-CHECK {s} users={d} {s}: HTTP {d} \"{s}\"\n", .{ case_name, n, r.label, x.status, x.message });
    }
    std.debug.print("user_sampling RED/GREEN {s}: sampled requests refused by name {d}/{d}, served well formed {d}/{d}\n", .{ case_name, refused, red.len, served_ok, red.len });
    try l.check(served_ok == red.len, "{s}: sampled served {d}/{d} (refused by name {d})", .{ case_name, served_ok, red.len, refused });
    if (served_ok != red.len) {
        // RED: the rest needs sampled requests served; the greedy references (pairs take 1, AEON-30, sampling_tests_stat's
        // greedy rows) for check_served.py's base comparison, then record what this boot answered and stop here
        for (0..n) |k| for (try pairsSet(a, k)) |r| try l.keep(try hx.send(a, b.port, try relabel(a, r, "take1")));
        for (try aeonSet(a)) |r| try l.keep(try hx.send(a, b.port, r));
        for ([_]Row{ .{ .label = "greedy-zero", .temp = "0" }, .{ .label = "greedy-omitted", .temp = null } }) |row| {
            try l.keep(try hx.send(a, b.port, .{ .label = try std.fmt.allocPrint(a, "c0-{s}", .{row.label}), .prompt = samp_prompt, .tokens = client_tokens, .sampling = try clientBody(a, row.temp, 20261010, 0) }));
        }
        up = false;
        return finish(a, case_name, &b, &l, try std.fmt.allocPrint(a, ",\"red_refused\":{d},\"red_total\":{d},\"stopped_after\":\"red\"", .{ refused, red.len }));
    }

    // 2. seeded identity: each request alone, then beside 1 other (greedy / other seed / structured), and at n=4 beside 3
    const ids = idSet();
    const nb = neighbours();
    var alone: [ids.len]Reply = undefined;
    for (ids, &alone) |r, *s| {
        s.* = try hx.send(a, b.port, try relabel(a, r, "alone"));
        show(s.*);
        try l.keep(s.*);
        try l.check(hx.plainOk(s.*) and hx.nativeSampled(s.*), "{s} alone well formed", .{r.label});
    }
    try l.check(hx.same(alone[0], alone[1]) or (std.mem.eql(u8, alone[0].content, alone[1].content) and std.mem.eql(u8, alone[0].token_sha, alone[1].token_sha)), "id-a whole = streamed", .{});
    var nb_alone: [nb.len]Reply = undefined;
    for (nb, &nb_alone) |r, *s| {
        s.* = try hx.send(a, b.port, try relabel(a, r, "alone"));
        show(s.*);
        try l.keep(s.*);
    }
    var together_same: usize = 0;
    var together_n: usize = 0;
    var in_flight: usize = 0;
    var batches: usize = 0;
    for (ids, alone) |x, ref| {
        for (nb, nb_alone) |y, yref| {
            const pair = [_]Req{ try relabel(a, x, try std.fmt.allocPrint(a, "beside-{s}", .{y.label})), try relabel(a, y, try std.fmt.allocPrint(a, "beside-{s}", .{x.label})) };
            const rs = try hx.together(a, b.port, &pair);
            batches += 1;
            if (hx.overlapped(rs)) in_flight += 1;
            for (rs) |r| {
                show(r);
                try l.keep(r);
            }
            together_n += 1;
            if (hx.same(rs[0], ref)) together_same += 1;
            try l.check(hx.same(rs[0], ref), "{s} beside {s} = alone", .{ x.label, y.label });
            try l.check(if (y.kind == .plain) hx.same(rs[1], yref) else (hx.structuredOk(a, rs[1]) and hx.same(rs[1], yref)), "{s} beside {s} = its alone", .{ y.label, x.label });
        }
        if (n == 4) {
            const quad = [_]Req{ try relabel(a, x, "beside-3"), try relabel(a, nb[0], try std.fmt.allocPrint(a, "beside-3-{s}", .{x.label})), try relabel(a, nb[1], try std.fmt.allocPrint(a, "beside-3-{s}", .{x.label})), try relabel(a, nb[2], try std.fmt.allocPrint(a, "beside-3-{s}", .{x.label})) };
            const rs = try hx.together(a, b.port, &quad);
            batches += 1;
            if (hx.overlapped(rs)) in_flight += 1;
            for (rs) |r| {
                show(r);
                try l.keep(r);
            }
            together_n += 1;
            if (hx.same(rs[0], ref)) together_same += 1;
            try l.check(hx.same(rs[0], ref), "{s} beside 3 = alone", .{x.label});
            for (rs[1..], nb_alone) |r, yref| try l.check(hx.same(r, yref), "{s} beside 3 = its alone", .{base(r.label)});
        }
    }
    std.debug.print("user_sampling identity {s}: seeded sampled together = alone {d}/{d}; batches in flight together {d}/{d} (recorded)\n", .{ case_name, together_same, together_n, in_flight, batches });

    // keyed by (request seed, token position) only: every seeded request's draws at exactly its own positions
    const draws = try b.mock.draws(a);
    for (ids, alone) |r, x| if (seedOf(r)) |seed| try l.check(try keysExact(a, draws, x, seed), "{s}: draws keyed at its own positions (seed {d})", .{ r.label, seed });
    try l.check(try keysExact(a, draws, nb_alone[1], 901), "n-sampled keyed at its own positions", .{});

    // 3. greedy rows unchanged: solo = together beside sampled rows (and = the switch-off boot's: check_served.py)
    for (greedySet(), greedy_solo) |g, ref| {
        const mix = [_]Req{ try relabel(a, g, "beside-sampled"), try relabel(a, ids[0], try std.fmt.allocPrint(a, "beside-{s}", .{g.label})) };
        const rs = try hx.together(a, b.port, &mix);
        for (rs) |r| {
            show(r);
            try l.keep(r);
        }
        try l.check(hx.same(rs[0], ref), "{s} beside a sampled row = solo", .{g.label});
        try l.check(hx.same(rs[1], alone[0]), "id-a beside {s} = alone", .{g.label});
    }
    for (greedySet(), greedy_solo) |g, ref| {
        const again = try hx.send(a, b.port, try relabel(a, g, "again"));
        try l.keep(again);
        try l.check(hx.same(again, ref), "{s} again = solo", .{g.label});
    }

    // 4. structured + sampled: the row's mask before its draw; json_object / json_schema valid n/n, alone and together
    const st = try structuredSet(a);
    var st_ok: usize = 0;
    var st_n: usize = 0;
    const masked_before = b.mock.masked_sampled;
    for (st) |r| {
        const x = try hx.send(a, b.port, try relabel(a, r, "alone"));
        show(x);
        try l.keep(x);
        st_n += 1;
        if (hx.structuredOk(a, x) and hx.nativeSampled(x)) st_ok += 1;
    }
    var i: usize = 0;
    while (i < st.len) : (i += n) {
        var group: [4]Req = undefined;
        for (0..n) |j| group[j] = try relabel(a, st[(i + j) % st.len], "together");
        const rs = try hx.together(a, b.port, group[0..n]);
        for (rs) |x| {
            show(x);
            try l.keep(x);
            st_n += 1;
            if (hx.structuredOk(a, x) and hx.nativeSampled(x)) st_ok += 1;
            try l.check(hx.same(x, find(l.all.items, base(x.label)).?), "{s} together = alone", .{base(x.label)});
        }
    }
    try l.check(st_ok == st_n, "structured + sampled valid {d}/{d}", .{ st_ok, st_n });
    try l.check(b.mock.masked_sampled > masked_before, "masked sampled draws ran (the mask before the draw)", .{});
    const draws2 = try b.mock.draws(a);
    var masked_keyed: usize = 0;
    for (draws2) |d| if (d.kind == .masked and d.seed >= 3000 and d.seed < 3008) {
        masked_keyed += 1;
    };
    try l.check(masked_keyed > 0, "masked draws carried the structured requests' own seeds", .{});
    std.debug.print("user_sampling structured {s}: valid {d}/{d}; masked sampled draws {d} (seeds 3000-3007: {d})\n", .{ case_name, st_ok, st_n, b.mock.masked_sampled - masked_before, masked_keyed });

    // 5. sampling_tests_stat gates in their CPU form, per row: n clients concurrently, each the 26-row matrix with its own seed
    const rows = try clientRows(a);
    try l.check(rows.len == 26, "client rows {d}", .{rows.len});
    var clients: [4]Client = undefined;
    var threads: [4]std.Thread = undefined;
    for (0..n) |k| {
        clients[k] = .{ .k = k, .port = b.port, .seed = 20261010 + k, .rows = rows, .arena = std.heap.ArenaAllocator.init(std.heap.page_allocator) };
        threads[k] = try std.Thread.spawn(.{}, Client.run, .{&clients[k]});
    }
    for (threads[0..n]) |th| th.join();
    defer for (clients[0..n]) |*c| c.arena.deinit();
    var gate1: usize = 0;
    for (clients[0..n]) |*c| {
        if (c.err) |e| try l.check(false, "client {d}: {s}", .{ c.k, @errorName(e) });
        for (c.replies) |r| {
            const x = try hx.copyReply(a, r);
            try l.keep(x);
            if (clientWellFormed(x)) gate1 += 1;
        }
        for (c.fails.items) |m| try l.check(false, "sampling_tests_stat {s}", .{m});
        // gate 1's greedy-zero: every row's equal (and = S's / the switch-off boot's: check_served.py)
        if (c.replies.len > 0 and clients[0].replies.len > 0) try l.check(sameRow(c.replies[0], clients[0].replies[0]), "row {d} greedy-zero = row 0's", .{c.k}) else try l.check(false, "row {d}: no replies", .{c.k});
    }
    try l.check(gate1 == 26 * n, "sampling_tests_stat gate 1: client rows well formed {d}/{d}", .{ gate1, 26 * n });
    std.debug.print("user_sampling sampling_tests_stat {s}: gate 1 rows {d}/{d}\n", .{ case_name, gate1, 26 * n });

    // gates 4-6: pairs (12 greedy x 2 takes per row, rows concurrently), G-det take 2 = take 1; G-same: check_served.py
    var take: [2][4][12]Reply = undefined;
    for (0..2) |tk| {
        for (0..12) |j| {
            var group: [4]Req = undefined;
            for (0..n) |k| group[k] = try relabel(a, (try pairsSet(a, k))[j], try std.fmt.allocPrint(a, "take{d}", .{tk + 1}));
            const rs = try hx.together(a, b.port, group[0..n]);
            for (rs, 0..) |r, k| {
                try l.keep(r);
                take[tk][k][j] = r;
            }
        }
    }
    var pairs_ok: usize = 0;
    var gdet: usize = 0;
    for (0..n) |k| for (0..12) |j| {
        for (0..2) |tk| {
            if (take[tk][k][j].status == 200 and take[tk][k][j].token_sha.len > 0 and take[tk][k][j].finish.len > 0) pairs_ok += 1;
        }
        if (sameRow(take[0][k][j], take[1][k][j])) gdet += 1;
    };
    try l.check(pairs_ok == 24 * n, "gate 4 pairs {d}/{d}", .{ pairs_ok, 24 * n });
    try l.check(gdet == 12 * n, "gate 6 G-det {d}/{d}", .{ gdet, 12 * n });

    // gate 8 (CPU form): AEON-30's shape, 30 greedy requests n at a time, all served
    const aeon = try aeonSet(a);
    var aeon_ok: usize = 0;
    var q: usize = 0;
    while (q < aeon.len) : (q += n) {
        const m = @min(n, aeon.len - q);
        const rs = try hx.together(a, b.port, aeon[q .. q + m]);
        for (rs) |r| {
            try l.keep(r);
            if (r.status == 200 and r.finish.len > 0 and r.completion_tokens > 0) aeon_ok += 1;
        }
    }
    try l.check(aeon_ok == 30, "gate 8 AEON-30 CPU form {d}/30", .{aeon_ok});

    // 6. a sampled client gone mid-stream: the lane, its slot and its region come back; then n requests together = alone
    const cut_req: Req = .{ .label = "cut-sampled", .prompt = keeper, .tokens = 2000, .stream = true, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":5151" };
    const cut_body = try hx.body(a, cut_req);
    const cut_raw = try std.fmt.allocPrint(a, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {d}\r\n\r\n{s}", .{ cut_body.len, cut_body });
    if (hx.askCut(a, b.port, cut_raw, 1500)) |got| {
        std.debug.print("user_sampling cut: the client left after {d} reply bytes\n", .{got.len});
    } else |e| std.debug.print("user_sampling cut: {s}\n", .{@errorName(e)});
    var idle = false;
    for (0..400) |_| {
        if (try idleNow(a, b.port)) {
            idle = true;
            break;
        }
        std.Io.sleep(t.io, .fromMilliseconds(25), .awake) catch {};
    }
    try l.check(idle, "requests_running back to 0 after the cut sampled stream", .{});
    const cut_draws = (try positionsOf(a, try b.mock.draws(a), 5151)).len;
    std.debug.print("user_sampling cut {s}: the cut sampled stream drew {d} of its 2000 tokens before its lane was released\n", .{ case_name, cut_draws });
    try l.check(cut_draws > 0 and cut_draws < 2000, "the cut sampled stream was cancelled mid-stream ({d} draws of 2000)", .{cut_draws});
    var after: [4]Req = undefined;
    const after_src = [_]Req{ ids[0], nb[0], nb[1], nb[2] };
    for (0..n) |k| after[k] = try relabel(a, after_src[k], "after-cut");
    const rs = try hx.together(a, b.port, after[0..n]);
    for (rs, 0..) |r, k| {
        show(r);
        try l.keep(r);
        const ref = if (k == 0) alone[0] else nb_alone[k - 1];
        try l.check(hx.same(r, ref), "{s} after the cut = alone", .{base(r.label)});
    }
    try l.check(hx.overlapped(rs) or n == 1, "after the cut: {d} requests in flight together (no slot or region held by the cut stream)", .{n});

    // gate 9 (CPU form): health after, nothing running, no fault (a last greedy request served)
    try l.check(@import("grammar_server_http.zig").status(try hx.get(a, b.port, "/health")) == 200, "health after", .{});
    const last = try hx.send(a, b.port, try relabel(a, greedySet()[0], "last"));
    try l.keep(last);
    try l.check(hx.same(last, greedy_solo[0]), "last greedy = solo (no fault)", .{});
    try l.check(try idleNow(a, b.port), "requests_running 0 at the end", .{});
    up = false;
    try finish(a, case_name, &b, &l, try std.fmt.allocPrint(a, ",\"red_refused\":{d},\"red_total\":{d},\"identity_together\":{d},\"identity_n\":{d},\"structured_ok\":{d},\"structured_n\":{d},\"gate1\":{d},\"pairs\":{d},\"gdet\":{d},\"aeon\":{d},\"in_flight\":{d},\"batches\":{d}", .{ refused, red.len, together_same, together_n, st_ok, st_n, gate1, pairs_ok, gdet, aeon_ok, in_flight, batches }));
}

// ---- the switch-off boots (u2off, u4off) and the single-user boot S --------------------------------------------------

/// The same greedy requests the switch-on boots send (greedy set, pairs take 1, AEON-30) and the sampled red set, which
/// keeps the named 400 with the switch off.
fn usersOff(case_name: []const u8, words: rig.Words, n: u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var l: Log = .{ .a = a };
    var b: rig.Boot = .{};
    try b.start(words);
    var up = true;
    defer if (up) {
        _ = b.stop() catch {};
    };
    for (greedySet()) |r| {
        const x = try hx.send(a, b.port, r);
        show(x);
        try l.keep(x);
    }
    const sampled = [_]Req{
        .{ .label = "red-plain", .prompt = keeper, .tokens = 32, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":7001" },
        .{ .label = "red-stream", .prompt = keeper, .tokens = 32, .stream = true, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":7002" },
    };
    var refused: usize = 0;
    for (sampled) |r| {
        const x = try hx.send(a, b.port, r);
        show(x);
        try l.keep(x);
        if (x.status == 400 and std.mem.eql(u8, x.message, refusal)) refused += 1;
    }
    try l.check(refused == sampled.len, "switch off: sampled refused by name {d}/{d}", .{ refused, sampled.len });
    for (0..n) |k| for (try pairsSet(a, k)) |r| {
        const x = try hx.send(a, b.port, try relabel(a, r, "take1"));
        try l.keep(x);
    };
    for (try aeonSet(a)) |r| try l.keep(try hx.send(a, b.port, r));
    const g0 = try hx.send(a, b.port, try relabel(a, greedySet()[0], "last"));
    try l.keep(g0);
    try l.check(hx.same(g0, l.all.items[0]), "switch off: greedy last = solo", .{});
    // sampling_tests_stat's greedy rows (greedy-zero, greedy-omitted) for gate 1's reference
    for ([_]Row{ .{ .label = "greedy-zero", .temp = "0" }, .{ .label = "greedy-omitted", .temp = null } }) |row| {
        const x = try hx.send(a, b.port, .{ .label = try std.fmt.allocPrint(a, "c0-{s}", .{row.label}), .prompt = samp_prompt, .tokens = client_tokens, .sampling = try clientBody(a, row.temp, 20261010, 0) });
        try l.keep(x);
    }
    up = false;
    try finish(a, case_name, &b, &l, try std.fmt.allocPrint(a, ",\"off_refused\":{d}", .{refused}));
}

/// The single-user boot S of the same head: every seeded and structured request alone, the neighbours, the greedy
/// rows, the four clients' matrices one after another (seeds 20261010 + k), pairs take 1 and AEON-30.
fn singleUser() !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var l: Log = .{ .a = a };
    var b: rig.Boot = .{};
    try b.start(&single);
    var up = true;
    defer if (up) {
        _ = b.stop() catch {};
    };
    for (greedySet()) |r| {
        const x = try hx.send(a, b.port, r);
        show(x);
        try l.keep(x);
    }
    const red = [_]Req{
        .{ .label = "red-plain", .prompt = keeper, .tokens = 32, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":7001" },
        .{ .label = "red-stream", .prompt = keeper, .tokens = 32, .stream = true, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":7002" },
        .{ .label = "red-object", .kind = .json_object, .prompt = "Give a JSON object describing the sky.", .tokens = 96, .sampling = ",\"temperature\":0.7,\"top_p\":0.95,\"seed\":7003" },
        .{ .label = "red-together", .prompt = "Describe the sky in one line.", .tokens = 32, .sampling = ",\"temperature\":1.0,\"seed\":7004" },
    };
    for (red) |r| {
        const x = try hx.send(a, b.port, r);
        show(x);
        try l.keep(x);
        try l.check(x.status == 200 and hx.nativeSampled(x), "S {s} served", .{r.label});
    }
    const ids = idSet();
    for (ids) |r| {
        const x = try hx.send(a, b.port, try relabel(a, r, "alone"));
        show(x);
        try l.keep(x);
        try l.check(hx.plainOk(x) and hx.nativeSampled(x), "S {s} well formed", .{r.label});
    }
    const draws = try b.mock.draws(a);
    for (ids) |r| if (seedOf(r)) |seed| {
        const x = find(l.all.items, r.label).?;
        try l.check(try keysExact(a, draws, x, seed), "S {s}: draws keyed at its own positions", .{r.label});
    };
    for (neighbours()) |r| {
        const x = try hx.send(a, b.port, try relabel(a, r, "alone"));
        show(x);
        try l.keep(x);
    }
    for (try structuredSet(a)) |r| {
        const x = try hx.send(a, b.port, try relabel(a, r, "alone"));
        show(x);
        try l.keep(x);
        try l.check(hx.structuredOk(a, x) and hx.nativeSampled(x), "S {s} valid", .{r.label});
    }
    const rows = try clientRows(a);
    for (0..4) |k| {
        var c: Client = .{ .k = k, .port = b.port, .seed = 20261010 + k, .rows = rows, .arena = std.heap.ArenaAllocator.init(std.heap.page_allocator) };
        defer c.arena.deinit();
        c.run();
        if (c.err) |e| try l.check(false, "S client {d}: {s}", .{ k, @errorName(e) });
        for (c.replies) |r| try l.keep(try hx.copyReply(a, r));
        for (c.fails.items) |m| try l.check(false, "S sampling_tests_stat {s}", .{m});
    }
    for (0..4) |k| for (try pairsSet(a, k)) |r| try l.keep(try hx.send(a, b.port, try relabel(a, r, "take1")));
    for (try aeonSet(a)) |r| try l.keep(try hx.send(a, b.port, r));
    up = false;
    try finish(a, "s", &b, &l, "");
}

// ---- a device fault in a sampled row's draw mid-round (u4 with the switch on) ---------------------------------------

fn faultMidRound() !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var l: Log = .{ .a = a };
    var b: rig.Boot = .{};
    try b.start(&on4);
    var up = true;
    defer if (up) {
        _ = b.stop() catch {};
    };
    // the faulting request's prompt length, from the same prompt under another seed
    const probe = try hx.send(a, b.port, .{ .label = "fault-probe", .prompt = keeper, .tokens = 8, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":6600" });
    show(probe);
    try l.keep(probe);
    try l.check(probe.status == 200, "fault probe served", .{});
    b.mock.fail_at = .{ .seed = 6666, .position = @as(u64, @intCast(probe.prompt_tokens)) + 5 };
    const mix = [_]Req{
        .{ .label = "fault-sampled", .prompt = keeper, .tokens = 64, .stream = true, .sampling = ",\"temperature\":0.8,\"top_p\":0.9,\"seed\":6666" },
        .{ .label = "fault-greedy", .prompt = "Say ok.", .tokens = 64 },
        .{ .label = "fault-other", .prompt = "Write a haiku about rain.", .tokens = 64, .sampling = ",\"temperature\":0.9,\"top_p\":0.9,\"seed\":6667" },
        .{ .label = "fault-object", .kind = .json_object, .prompt = "Give a JSON object describing the sky.", .tokens = 96 },
    };
    const rs = try hx.together(a, b.port, &mix);
    var ended: usize = 0;
    for (rs) |r| {
        show(r);
        try l.keep(r);
        ended += 1;
    }
    try l.check(b.mock.failed == 1, "the fault fired once in the sampled row's draw ({d})", .{b.mock.failed});
    try l.check(ended == mix.len, "every request of the faulted round ended ({d}/{d})", .{ ended, mix.len });
    const faulted = rs[0];
    try l.check(!(faulted.status == 200 and (std.mem.eql(u8, faulted.finish, "stop") or std.mem.eql(u8, faulted.finish, "length")) and faulted.completion_tokens == 64), "the faulted sampled stream did not finish as served", .{});
    var idle = false;
    for (0..400) |_| {
        if (try idleNow(a, b.port)) {
            idle = true;
            break;
        }
        std.Io.sleep(t.io, .fromMilliseconds(25), .awake) catch {};
    }
    try l.check(idle, "requests_running back to 0 after the fault", .{});
    // concurrent_users's rule: a failed round fails every owner and the backend refuses later work before any device call
    const later = try hx.send(a, b.port, .{ .label = "fault-later", .prompt = "Say ok.", .tokens = 8 });
    show(later);
    try l.keep(later);
    try l.check(!(later.status == 200 and later.completion_tokens == 8 and std.mem.eql(u8, later.finish, "length")), "a later request is refused promptly (sticky failure), never hangs", .{});
    up = false;
    const served_n = b.stop() catch |e| blk: {
        std.debug.print("user_sampling fail: the entry ended {s} after the fault (recorded)\n", .{@errorName(e)});
        break :blk 0;
    };
    const f = try facts(a, &b, &l, try std.fmt.allocPrint(a, ",\"served\":{d},\"fault_fired\":{d}", .{ served_n, b.mock.failed }));
    try rig.record(a, "fail", b.words, l.all.items, f);
    std.debug.print("user_sampling fail summary: checks {d}, fails {d}; facts {s}\n", .{ l.checks, l.fails.items.len, f });
    for (l.fails.items) |m| std.debug.print("user_sampling fail FAILED: {s}\n", .{m});
    if (l.fails.items.len > 0) return error.user_sampling_checksfailed;
}

// ---- the cases ------------------------------------------------------------------------------------------------------------

test "user_sampling u2: TF_AFD_USERS=2 + TF_AFD_USERS_SAMPLING=1 + TF_GLM_SAMPLING=1 + grammar: RED->GREEN, seeded identity alone = beside 1, greedy unchanged, structured+sampled valid, sampling_tests_stat gates per row, a client gone mid-stream" {
    if (!rig.case("u2")) return error.SkipZigTest;
    try usersOn("u2", &on2, 2);
}

test "user_sampling u4: TF_AFD_USERS=4 + pool + interleave + TF_AFD_USERS_SAMPLING=1 + TF_GLM_SAMPLING=1 + grammar: RED->GREEN, seeded identity alone = beside 1 = beside 3, greedy unchanged, structured+sampled valid, sampling_tests_stat gates per row, a client gone mid-stream" {
    if (!rig.case("u4")) return error.SkipZigTest;
    try usersOn("u4", &on4, 4);
}

test "user_sampling u2off: TF_AFD_USERS=2 with the switch unset: sampled refused by name; greedy rows recorded for on = off" {
    if (!rig.case("u2off")) return error.SkipZigTest;
    try usersOff("u2off", &off2, 2);
}

test "user_sampling u4off: TF_AFD_USERS=4 + pool + interleave with the switch unset: sampled refused by name; greedy rows recorded for on = off" {
    if (!rig.case("u4off")) return error.SkipZigTest;
    try usersOff("u4off", &off4, 4);
}

test "user_sampling s: the single-user entry S of the same head (TF_GLM_SAMPLING=1 + grammar): the seeded, structured, greedy and sampling_tests_stat references" {
    if (!rig.case("s")) return error.SkipZigTest;
    try singleUser();
}

test "user_sampling fail: a device fault in a sampled row's draw mid-round at USERS=4 (pool + interleave): every request ends, nothing left running, later work refused promptly" {
    if (!rig.case("fail")) return error.SkipZigTest;
    try faultMidRound();
}

test "user_sampling startup: TF_AFD_USERS_SAMPLING other than unset/0/1 refused; 1 without TF_AFD_USERS >= 2 or without TF_GLM_SAMPLING=1 refused by name; unset or 0 is today's plan" {
    if (!rig.case("refuse")) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = std.mem.span(std.c.getenv("TF_grammar_server_MODEL") orelse return error.MissingEnv);
    const config = try std.Io.Dir.cwd().readFileAlloc(t.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(16 << 20));
    const args = try a.dupe([]const u8, &.{ "--host", "127.0.0.1", "--port", "1", "--name", hx.name, "--context", "2496", "--temperature", "0", "--drafts", "0" });
    const flags: cli.Flags = .{ .role = .attn, .verb = .serve, .model = dir, .serve_args = args };
    const names = [_][:0]const u8{ "TF_AFD_USERS", "TF_GLM_SAMPLING", "TF_AFD_USERS_SAMPLING", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE" };
    const Case = struct { words: []const W, want: ?anyerror, sampled: bool = false };
    const cases = [_]Case{
        .{ .words = &.{ .{ "TF_AFD_USERS", "2" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "2" } }, .want = error.BadAfdUsersSampling },
        .{ .words = &.{ .{ "TF_AFD_USERS", "2" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "on" } }, .want = error.BadAfdUsersSampling },
        .{ .words = &.{ .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "1" } }, .want = error.UsersSamplingNeedsUsers },
        .{ .words = &.{ .{ "TF_AFD_USERS", "1" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "1" } }, .want = error.UsersSamplingNeedsUsers },
        .{ .words = &.{ .{ "TF_AFD_USERS", "2" }, .{ "TF_AFD_USERS_SAMPLING", "1" } }, .want = error.UsersSamplingNeedsGlmSampling },
        .{ .words = &.{ .{ "TF_AFD_USERS", "4" }, .{ "TF_GLM_SAMPLING", "0" }, .{ "TF_AFD_USERS_SAMPLING", "1" } }, .want = error.UsersSamplingNeedsGlmSampling },
        .{ .words = &.{ .{ "TF_AFD_USERS", "2" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "1" } }, .want = null, .sampled = true },
        .{ .words = &.{ .{ "TF_AFD_USERS", "4" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "1" } }, .want = null, .sampled = true },
        .{ .words = &.{ .{ "TF_AFD_USERS", "2" }, .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_USERS_SAMPLING", "0" } }, .want = null },
        .{ .words = &.{ .{ "TF_AFD_USERS", "2" }, .{ "TF_AFD_USERS_SAMPLING", "" } }, .want = null },
        .{ .words = &.{.{ "TF_AFD_USERS", "2" }}, .want = null },
    };
    var ok: usize = 0;
    for (cases) |c| {
        for (names) |nm| _ = unsetenv(nm);
        for (c.words) |kv| _ = setenv(kv[0], kv[1], 1);
        const got = served.plan(a, flags, config);
        const pass = if (c.want) |w| (if (got) |_| false else |e| e == w) else (if (got) |p| p.users >= 2 and planSampled(p) == c.sampled else |_| false);
        if (pass) ok += 1;
        std.debug.print("user_sampling startup", .{});
        for (c.words) |kv| std.debug.print(" {s}={s}", .{ kv[0], kv[1] });
        std.debug.print(": {s} (want {s}{s}) {s}\n", .{ if (got) |_| "planned" else |e| @errorName(e), if (c.want) |w| @errorName(w) else "planned", if (c.sampled) ", sampled" else "", if (pass) "ok" else "WRONG" });
    }
    for (names) |nm| _ = unsetenv(nm);
    std.debug.print("user_sampling startup refusals {d}/{d}\n", .{ ok, cases.len });
    try t.expectEqual(cases.len, ok);
}

/// Plan.sampled when the head has it (a RED head does not: false)
fn planSampled(p: served.Plan) bool {
    if (comptime @hasField(served.Plan, "sampled")) return p.sampled;
    return false;
}
