//! structured_entry (local integration; Claude Opus 5.5 via Hermes): the AFD attention role's SERVED entry end to end on a CPU proxy
//! engine, with the switches qualification_f245fdbd's lab run set. `afd_attn_serve.run` is the verb's entry; it hands TF_AFD_USERS > 1 to
//! `afd_attn_users.run`, the users server entry whose `serve.run` call qualification_f245fdbd found without `.model_dir` (HTTP 400
//! "structured output: the checkpoint has no tokenizer.json" for every structured request). Here both entries run with
//! TF_AFD_GRAMMAR_SERVE=1 (TF_AFD_GRAMMAR_MASK=1) over the REAL GLM-5.3 tokenizer directory (TF_grammar_server_MODEL: grammar_server's
//! fixture checkpoint: tokenizer.json, config.json, the chat template) and grammar_abi's published libtfgrammar
//! (TF_grammar_server_LIB / _LIB_SHA256, a -Dgram2abi build), exactly as the role reads them (TF_AFD_GRAMMAR_LIB/_LIB_SHA256).
//! The proxy engine: afd_users_device_test.zig's recorded rig (Ash Hart's fixture model cut to 5 layers, a recording
//! Fwd, a real native RoCE exchange Node over an in-process transport) under the product's own Attn, Users, Pool, interleave,
//! lane core, LaneHost, grammar source, server and plan(); the device answers are MOCK (a plain draw answers one fixed
//! token; a masked draw, greedy or sampled, takes a fixed preference among the STAGED words the real matcher filled:
//! end-of-text, then the closers, then the lowest eligible non-blank id). The requests are the lab's (structured_users_b's
//! served_probe.py / structured_users_probe.py bodies, prompts, schema, settings and order) over loopback HTTP.
//! One case per process (serve.run's stop flag is process-wide): structured_entry_CASE = u2 | u4 | sp (zig/build/glm_afd.zig's
//! test-structured_entry_served runs each). No CUDA numerics; no network beyond 127.0.0.1.
//! Rig scaffolding: concurrent_users_r2's Rec (GPT-6.1 Sol via Hermes) after Ash Hart's cuda_afd_engine_test.zig; inherited NOTICE
//! credits apply (Ash Hart / TensorFold, the XGrammar contributors (MLC, Apache-2.0) through grammar_abi's library).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const server = @import("server");
const served = @import("afd_attn_serve.zig");
const cli = @import("tf_glm_afd.zig");
const g2 = @import("afd_attn_grammar.zig");
const h = @import("grammar_server_http.zig");
const md = glm.sample.mask_dev;
const proto = cuda.xchg.proto;
const posix = std.posix;
const t = std.testing;
const gpa = t.allocator;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// the served window and the Engine's capacity (window + afd_serve_lanes' slack), as plan() computes them
const window = 2496;
const capacity = 2560;
/// one prompt chunk holds the warm-up's 96 ids (the warm receipt counts one prefill forward)
const prefill_rows = 96;
const name = "glm-5.3-flash-afd";

fn case(want: []const u8) bool {
    const v = std.c.getenv("structured_entry_CASE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), want);
}

fn env(n: [:0]const u8) ![]const u8 {
    const v = std.c.getenv(n) orelse {
        std.debug.print("structured_entry needs {s} (zrun --fix srvabi: grammar_server's real-library mounts)\n", .{n});
        return error.MissingEnv;
    };
    return std.mem.span(v);
}

fn set(n: [:0]const u8, v: [:0]const u8) void {
    _ = setenv(n, v, 1);
}

// ---- the lab's requests (structured_users_b kit: served_probe.py PROMPTS/SCHEMA, structured_users_probe.py PLAIN/SOLO/PAIRS/SAMPLING) -------

const schema_json = "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\",\"enum\":[\"yes\",\"no\"]},\"confidence\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":100},\"sky\":{\"type\":\"object\",\"properties\":{\"color\":{\"type\":\"string\"},\"clear\":{\"type\":\"boolean\"}},\"required\":[\"color\",\"clear\"],\"additionalProperties\":false},\"reasons\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"minItems\":1,\"maxItems\":2}},\"required\":[\"answer\",\"confidence\",\"sky\",\"reasons\"],\"additionalProperties\":false}";
const prompt_object = "Is the sky blue on a clear day? Reply with a short JSON object with an \\\"answer\\\" key (\\\"yes\\\" or \\\"no\\\").";
const prompt_schema = "Is the sky blue on a clear day? Reply in JSON: answer (yes or no), confidence (0-100), sky (its color and whether it is clear) and one or two short reasons.";
const plain_a = "List the eight planets of the solar system in order from the Sun, with one short fact about each.";
const plain_b = "Count from one to forty in words, separated by commas.";
const structured_tokens = 512;
const sampled_tokens = 1024;

const Kind = enum { plain, json_object, json_schema };
const Req = struct {
    label: []const u8,
    kind: Kind,
    prompt: []const u8 = "",
    tokens: u32 = 0,
    stream: bool = false,
    /// a sampling setting's JSON members (",\"temperature\":0.7,...") or "" (greedy: temperature 0, seed 7)
    sampling: []const u8 = "",
};

fn plainA(stream: bool) Req {
    return .{ .label = if (stream) "u-plain-a-stream" else "u-plain-a", .kind = .plain, .prompt = plain_a, .tokens = 192, .stream = stream };
}
fn plainB(stream: bool) Req {
    return .{ .label = if (stream) "u-plain-b-stream" else "u-plain-b", .kind = .plain, .prompt = plain_b, .tokens = 160, .stream = stream };
}
fn schemaReq(stream: bool) Req {
    return .{ .label = if (stream) "u-schema-stream" else "u-schema", .kind = .json_schema, .prompt = prompt_schema, .tokens = structured_tokens, .stream = stream };
}
fn objectReq(stream: bool) Req {
    return .{ .label = if (stream) "u-object-stream" else "u-object", .kind = .json_object, .prompt = prompt_object, .tokens = structured_tokens, .stream = stream };
}

/// served_probe.request_body / structured_users_probe.body: the same members in the same order
fn body(a: std.mem.Allocator, r: Req) ![]u8 {
    const temp: []const u8 = if (r.sampling.len > 0) "" else ",\"temperature\":0,\"seed\":7";
    const fmt: []const u8 = switch (r.kind) {
        .plain => "",
        .json_object => ",\"response_format\":{\"type\":\"json_object\"}",
        .json_schema => try std.fmt.allocPrint(a, ",\"response_format\":{{\"type\":\"json_schema\",\"json_schema\":{{\"name\":\"sky\",\"schema\":{s}}}}}", .{schema_json}),
    };
    return std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}]{s}{s},\"max_tokens\":{d},\"draft\":false,\"chat_template_kwargs\":{{\"enable_thinking\":false}},\"stream\":{s}{s}{s}}}", .{ name, r.prompt, temp, r.sampling, r.tokens, if (r.stream) "true" else "false", if (r.stream) ",\"stream_options\":{\"include_usage\":true}" else "", fmt });
}

// ---- the reply: status, content, finish, usage; whole or SSE (de-chunked) ---------------------------------------------

const Reply = struct {
    label: []const u8,
    kind: Kind,
    status: u16 = 0,
    content: []const u8 = "",
    finish: []const u8 = "",
    usage: []const u8 = "",
    message: []const u8 = "",
    raw: []const u8 = "",
    t0: i96 = 0,
    t1: i96 = 0,
};

fn dechunk(a: std.mem.Allocator, raw: []const u8) ![]const u8 {
    const head_end = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return raw;
    const head = try a.alloc(u8, head_end);
    _ = std.ascii.lowerString(head, raw[0..head_end]);
    const rest = raw[head_end + 4 ..];
    if (std.mem.indexOf(u8, head, "transfer-encoding: chunked") == null) return rest;
    var out: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    while (at < rest.len) {
        const eol = std.mem.indexOfPos(u8, rest, at, "\r\n") orelse break;
        const size_text = std.mem.trim(u8, rest[at..eol], " ");
        const semi = std.mem.indexOfScalar(u8, size_text, ';') orelse size_text.len;
        const size = std.fmt.parseInt(usize, size_text[0..semi], 16) catch break;
        if (size == 0) break;
        const start = eol + 2;
        if (start + size > rest.len) break;
        try out.appendSlice(a, rest[start .. start + size]);
        at = start + size + 2;
    }
    return out.items;
}

fn str(v: ?server.json.Value) []const u8 {
    const x = v orelse return "";
    return switch (x) {
        .string => |s| s,
        else => "",
    };
}

fn parseReply(a: std.mem.Allocator, r: Req, raw: []const u8) !Reply {
    var out: Reply = .{ .label = r.label, .kind = r.kind, .status = h.status(raw), .raw = raw };
    const text = try dechunk(a, raw);
    if (out.status != 200) {
        out.message = switch (server.json.parse(a, text) catch return out) {
            .ok => |v| str((v.get("error") orelse v).get("message")),
            .err => text,
        };
        return out;
    }
    if (!r.stream) {
        const v = switch (try server.json.parse(a, text)) {
            .ok => |v| v,
            .err => return error.NotJson,
        };
        const c = v.get("choices").?.array[0];
        out.content = str(c.get("message").?.get("content"));
        out.finish = str(c.get("finish_reason"));
        if (v.get("usage")) |u| out.usage = try server.json.stringify(a, u, .{});
        return out;
    }
    var content: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line0| {
        const line = std.mem.trimEnd(u8, line0, "\r");
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = line["data: ".len..];
        if (std.mem.eql(u8, data, "[DONE]")) continue;
        const v = switch (server.json.parse(a, data) catch continue) {
            .ok => |v| v,
            .err => continue,
        };
        if (v.get("usage")) |u| if (u == .object) {
            out.usage = try server.json.stringify(a, u, .{});
        };
        const choices = v.get("choices") orelse continue;
        if (choices != .array or choices.array.len == 0) continue;
        const c = choices.array[0];
        if (c.get("delta")) |d| try content.appendSlice(a, str(d.get("content")));
        const fr = str(c.get("finish_reason"));
        if (fr.len > 0) out.finish = fr;
    }
    out.content = content.items;
    return out;
}

/// structured_users_probe's IDENTITY minus the server's token_sha (not in an OpenAI reply): content, finish, the usage counts
fn same(x: Reply, y: Reply) bool {
    return x.status == y.status and std.mem.eql(u8, x.content, y.content) and std.mem.eql(u8, x.finish, y.finish) and std.mem.eql(u8, x.usage, y.usage);
}

// ---- validation (served_probe.validate's rules for these two formats) -------------------------------------------------

fn isObject(a: std.mem.Allocator, text: []const u8) bool {
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return false;
    return v == .object;
}

fn onlyKeys(o: std.json.ObjectMap, keys: []const []const u8) bool {
    if (o.count() != keys.len) return false;
    for (keys) |k| if (o.get(k) == null) return false;
    return true;
}

fn validSky(a: std.mem.Allocator, text: []const u8) bool {
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return false;
    if (v != .object or !onlyKeys(v.object, &.{ "answer", "confidence", "sky", "reasons" })) return false;
    const ans = v.object.get("answer").?;
    if (ans != .string or !(std.mem.eql(u8, ans.string, "yes") or std.mem.eql(u8, ans.string, "no"))) return false;
    const c = v.object.get("confidence").?;
    if (c != .integer or c.integer < 0 or c.integer > 100) return false;
    const sky = v.object.get("sky").?;
    if (sky != .object or !onlyKeys(sky.object, &.{ "color", "clear" })) return false;
    if (sky.object.get("color").? != .string or sky.object.get("clear").? != .bool) return false;
    const rs = v.object.get("reasons").?;
    if (rs != .array or rs.array.items.len < 1 or rs.array.items.len > 2) return false;
    for (rs.array.items) |x| if (x != .string) return false;
    return true;
}

/// A structured reply as gate 3/4/5 count it: HTTP 200, `stop`, parsed, validated.
fn structuredOk(a: std.mem.Allocator, r: Reply) bool {
    if (r.status != 200 or !std.mem.eql(u8, r.finish, "stop")) return false;
    return switch (r.kind) {
        .json_object => isObject(a, r.content),
        .json_schema => validSky(a, r.content),
        .plain => false,
    };
}

// ---- HTTP over loopback (grammar_server_http.zig's ask, at a given port) ----------------------------------------------------

fn ask(a: std.mem.Allocator, port: u16, raw: []const u8) ![]u8 {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    defer _ = posix.system.close(fd);
    const addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = 1 << 24 | 127 };
    if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.ConnectFailed;
    const timeout: posix.timeval = .{ .sec = 600, .usec = 0 };
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch {};
    var at: usize = 0;
    while (at < raw.len) {
        const n = posix.system.write(fd, raw[at..].ptr, raw.len - at);
        if (posix.errno(n) != .SUCCESS) return error.WriteFailed;
        at += @intCast(n);
    }
    var out: std.ArrayList(u8) = .empty;
    var buf: [8192]u8 = undefined;
    while (true) {
        const got = try posix.read(fd, &buf);
        if (got == 0) break;
        try out.appendSlice(a, buf[0..got]);
    }
    return out.toOwnedSlice(a);
}

fn postRaw(a: std.mem.Allocator, port: u16, payload: []const u8) ![]u8 {
    return ask(a, port, try std.fmt.allocPrint(a, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {d}\r\n\r\n{s}", .{ payload.len, payload }));
}

fn now() i96 {
    return std.Io.Clock.awake.now(t.io).toNanoseconds();
}

fn send(a: std.mem.Allocator, port: u16, r: Req) !Reply {
    const t0 = now();
    const raw = try postRaw(a, port, try body(a, r));
    var out = try parseReply(a, r, raw);
    out.t0 = t0;
    out.t1 = now();
    return out;
}

/// Requests released together by one barrier (the probe's pair form), each on its own connection, thread and arena.
const Batch = struct {
    port: u16,
    reqs: []const Req,
    replies: []Reply,
    errs: []?anyerror,
    arenas: []std.heap.ArenaAllocator,
    ready: std.atomic.Value(usize) = .init(0),
    go: std.atomic.Value(bool) = .init(false),

    fn one(b: *Batch, i: usize) void {
        _ = b.ready.fetchAdd(1, .acq_rel);
        while (!b.go.load(.acquire)) std.atomic.spinLoopHint();
        b.replies[i] = send(b.arenas[i].allocator(), b.port, b.reqs[i]) catch |e| blk: {
            b.errs[i] = e;
            break :blk .{ .label = b.reqs[i].label, .kind = b.reqs[i].kind };
        };
    }
};

fn copyReply(a: std.mem.Allocator, r: Reply) !Reply {
    var x = r;
    x.content = try a.dupe(u8, r.content);
    x.finish = try a.dupe(u8, r.finish);
    x.usage = try a.dupe(u8, r.usage);
    x.message = try a.dupe(u8, r.message);
    x.raw = "";
    return x;
}

fn together(a: std.mem.Allocator, port: u16, reqs: []const Req) ![]Reply {
    var b: Batch = .{ .port = port, .reqs = reqs, .replies = try a.alloc(Reply, reqs.len), .errs = try a.alloc(?anyerror, reqs.len), .arenas = try a.alloc(std.heap.ArenaAllocator, reqs.len) };
    @memset(b.errs, null);
    for (b.arenas) |*x| x.* = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer for (b.arenas) |*x| x.deinit();
    const threads = try a.alloc(std.Thread, reqs.len);
    for (threads, 0..) |*th, i| th.* = try std.Thread.spawn(.{}, Batch.one, .{ &b, i });
    while (b.ready.load(.acquire) < reqs.len) std.atomic.spinLoopHint();
    b.go.store(true, .release);
    for (threads) |th| th.join();
    for (b.errs, reqs) |e, r| if (e) |x| {
        std.debug.print("structured_entry request {s}: {s}\n", .{ r.label, @errorName(x) });
        return x;
    };
    for (b.replies) |*r| r.* = try copyReply(a, r.*); // out of the per-thread arenas before they go
    return b.replies;
}

fn overlapped(rs: []const Reply) bool {
    var first_end: i96 = std.math.maxInt(i96);
    var last_start: i96 = 0;
    for (rs) |r| {
        first_end = @min(first_end, r.t1);
        last_start = @max(last_start, r.t0);
    }
    return last_start < first_end;
}

// ---- the MOCK device: plain draws (the recorder's picker) and masked draws (the adapter's emulator) -------------------

const Mock = struct {
    dev: *md.Dev,
    trace: *glm.trace.Trace,
    text: server.model_text.Text,
    /// end-of-text first, then: " } ] , : { [ 1  (single-character ids of this tokenizer): a string closes before
    /// anything else (inside a string every closer is text), a value or container closes before a new one opens
    order: [8]u32,
    eos: []const u32,
    plain: u32,
    staged_seen: u64 = 0,
    plain_draws: usize = 0,
    masked_greedy: usize = 0,
    masked_sampled: usize = 0,
    resets: usize = 0,

    fn of(p: *anyopaque) *Mock {
        return @ptrCast(@alignCast(p));
    }
    fn picker(m: *Mock) glm.trace.Picker {
        return .{ .ctx = m, .pick = pick };
    }
    fn emulator(m: *Mock) md.Emulator {
        return .{ .ctx = m, .candidate = candidate };
    }
    fn eligible(words: []const u32, id: u32) bool {
        return id / 32 < words.len and (words[id / 32] >> @as(u5, @intCast(id % 32))) & 1 == 1;
    }
    fn blank(m: *Mock, id: u32) bool {
        var buf: [512]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const s = m.text.decode(fba.allocator(), &.{id}) catch return false;
        return s.len == 0 or std.mem.indexOfAny(u8, s, " \t\r\n") != null;
    }
    /// end-of-text when allowed, else the first allowed of `order`, else the lowest eligible non-blank id
    fn prefer(m: *Mock, words: []const u32) !u32 {
        for (m.eos) |e| if (eligible(words, e)) return e;
        for (m.order) |id| if (eligible(words, id)) return id;
        var lowest: ?u32 = null;
        for (words, 0..) |w0, i| {
            var w = w0;
            while (w != 0) : (w &= w - 1) {
                const id: u32 = @intCast(i * 32 + @ctz(w));
                if (lowest == null) lowest = id;
                if (!m.blank(id)) return id;
            }
        }
        return lowest orelse error.EmptyStaged;
    }
    /// keeps the recorder's memory bounded over a long served run (only the trace's own bookkeeping is dropped)
    fn trim(m: *Mock) void {
        if (m.trace.events.items.len > 200_000) {
            m.trace.reset();
            m.resets += 1;
        }
    }
    fn pick(p: *anyopaque, _: glm.trace.T, positions: ?[]const u64, out: []u32, probs: ?[]f64) !void {
        const m = of(p);
        if (positions != null and m.dev.stats.staged != m.staged_seen) {
            // structured_users's masked sampled draw: its words were staged for this row just before the nucleus's draw
            m.staged_seen = m.dev.stats.staged;
            m.masked_sampled += 1;
            @memset(out, try m.prefer(m.dev.host[0..m.dev.words]));
        } else {
            m.plain_draws += 1;
            @memset(out, m.plain);
        }
        if (probs) |x| @memset(x, 0);
        m.trim();
    }
    fn candidate(p: *anyopaque, x: *const md.Dev, _: glm.trace.T) !md.Candidate {
        const m = of(p);
        m.staged_seen = x.stats.staged;
        m.masked_greedy += 1;
        const id = try m.prefer(x.host[0..x.words]);
        m.trim();
        return .{ .id = id, .bits = 0x3f80_0000 };
    }
};

// ---- the proxy engine: the recorded rig (concurrent_users_r2's Rec, one Engine, its Attn, a real Node) --------------------------

const Rig = struct {
    parsed: std.json.Parsed(std.json.Value) = undefined,
    m: glm.model.Model = undefined,
    w: glm.views.Weights = undefined,
    e: *glm.engine.Engine = undefined,
    a: glm.afd_attn.Afd = undefined,
    at: glm.afd_engine.Attn = undefined,
    space: glm.trace.Space = undefined,
    trace: glm.trace.Trace = undefined,
    f: glm.fwd.Fwd = undefined,
    region: []align(4096) u8 = &.{},
    status: []u32 = &.{},
    node: cuda.xchg.Node = undefined,
    dev: md.Dev = undefined,
    posts: usize = 0,
    takes: usize = 0,

    /// `pooled`: slot 0's caches detached as afd_attn_gpu does for TF_AFD_POOL_TOKENS; `sampling`: TF_GLM_SAMPLING=1's
    /// nucleus and top_k (afd_attn_gpu's useSampling), before the temporaries are placed
    fn init(r: *Rig, pooled: bool, sampling: bool) !void {
        r.parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
        r.m = try glm.model.Model.parse(gpa, r.parsed.value.object.get("config_json").?.string, 1, 0);
        try r.m.cut(5);
        r.m.mtp_layers = 0;
        r.w = try glm.views.rank(gpa, &r.m);
        r.e = try glm.engine.Engine.create(gpa, &r.m, &r.w, capacity, 8, prefill_rows, true);
        if (sampling) {
            try r.e.useSampling(gpa);
            r.e.sampled_top_k = true;
        }
        r.a = try glm.afd_attn.Afd.init(&r.m, prefill_rows);
        r.space = .{ .gpa = gpa };
        try glm.views.walk(&r.w, glm.views.Path.init("w"), &r.space, glm.trace.placeAll);
        try r.e.walk(&r.space, glm.trace.placeAll);
        try r.e.walkTmp(&r.space, glm.trace.placeAll);
        r.e.tmp.space = &r.space;
        try r.space.place("b.afd_recv", &r.a.afd_recv);
        r.trace = glm.trace.Trace.init(gpa);
        r.f = glm.fwd.Fwd.recording(&r.m, &r.trace);
        r.at = try glm.afd_engine.Attn.init(r.e, &r.a);
        const settings: proto.Settings = .{ .layout = r.a.layout, .max_rows = prefill_rows, .ports = 1 };
        r.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast((proto.Region{ .slot = settings.slotBytes() }).total()));
        @memset(r.region, 0);
        r.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(r.status, 0);
        r.node = try cuda.xchg.Node.init(gpa, t.io, .attn, .{ .settings = settings, .warn_failures = true }, .{ .region = r.region, .region_dev = @intFromPtr(r.region.ptr), .status = r.status, .status_dev = @intFromPtr(r.status.ptr) });
        r.a.wire = .{ .node = &r.node, .host = .{ .ctx = r, .post = post, .take = take } };
        if (pooled) glm.afd_pool.detachTokens(&r.e.st);
        r.dev = try md.Dev.recorded(gpa, &r.m, &r.space);
        r.e.smp.masked_dev = &r.dev; // TF_AFD_GRAMMAR_MASK=1's adapter on the sampler, as afd_attn_gpu installs it
    }
    fn deinit(r: *Rig) void {
        r.dev.close();
        r.node.deinit();
        gpa.free(r.region);
        gpa.free(r.status);
        r.trace.deinit();
        r.space.deinit();
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.parsed.deinit();
    }
    fn post(p: *anyopaque, _: *cuda.xchg.Node, _: *const [proto.HDR_BYTES]u8, _: usize, _: usize) !void {
        const r: *Rig = @ptrCast(@alignCast(p));
        r.posts += 1;
    }
    fn take(p: *anyopaque, _: *cuda.xchg.Node, _: usize, _: usize) !void {
        const r: *Rig = @ptrCast(@alignCast(p));
        r.takes += 1;
    }
};

// ---- one boot: the switches, plan(), the rig, the verb's entry on its own thread, the loopback port ------------------

const Boot = struct {
    rig: Rig = .{},
    mock: Mock = undefined,
    hf: *server.hf_text.HfText = undefined,
    arena: std.heap.ArenaAllocator = undefined,
    port: u16 = 0,
    port_text: [8]u8 = undefined,
    thread: std.Thread = undefined,
    result: anyerror!usize = 0,
    done: std.atomic.Value(bool) = .init(false),
    /// the server answered /health: serve.run's SIGTERM handler is installed (stop may raise it)
    listening: bool = false,
    flags: cli.Flags = .{},
    on: served.On = undefined,

    /// `words`: the attention line's switches (ZL_ADDA's words), set before plan() and the role's reads, as the lab does
    fn start(b: *Boot, words: []const [2][:0]const u8) !void {
        b.arena = std.heap.ArenaAllocator.init(gpa);
        const a = b.arena.allocator();
        const dir = try env("TF_grammar_server_MODEL");
        set("TF_AFD_GRAMMAR_LIB", try a.dupeSentinel(u8, try env("TF_grammar_server_LIB"), 0));
        set("TF_AFD_GRAMMAR_LIB_SHA256", try a.dupeSentinel(u8, try env("TF_grammar_server_LIB_SHA256"), 0));
        for (words) |kv| set(kv[0], kv[1]);
        var pooled = false;
        var sampling = false;
        for (words) |kv| {
            if (std.mem.eql(u8, kv[0], "TF_AFD_POOL_TOKENS")) pooled = true;
            if (std.mem.eql(u8, kv[0], "TF_GLM_SAMPLING") and std.mem.eql(u8, kv[1], "1")) sampling = true;
        }
        b.port = 20000 + @as(u16, @intCast(@mod(std.c.getpid(), 20000)));
        const port_s = try std.fmt.bufPrint(&b.port_text, "{d}", .{b.port});
        // zlv1.sh's SARGS (--host --port --name --context --temperature 0 --drafts 0)
        const args = try a.dupe([]const u8, &.{ "--host", "127.0.0.1", "--port", port_s, "--name", name, "--context", "2496", "--temperature", "0", "--drafts", "0" });
        b.flags = .{ .role = .attn, .verb = .serve, .model = dir, .serve_args = args };
        const config = try std.Io.Dir.cwd().readFileAlloc(t.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(16 << 20));
        const plan = try served.plan(a, b.flags, config);
        try t.expectEqual(@as(usize, capacity), plan.capacity);
        const grammar = try g2.fromEnv(true); // TF_AFD_GRAMMAR_MASK=1 with the adapter installed below
        try b.rig.init(pooled, sampling);
        var problem: []const u8 = "";
        b.hf = try server.hf_text.HfText.load(gpa, t.io, dir, a, &problem);
        const text = b.hf.text();
        var order: [8]u32 = undefined;
        for ([_][]const u8{ "\"", "}", "]", ",", ":", "{", "[", "1" }, &order) |s, *id| {
            const ids = try text.encode(a, s, false);
            try t.expectEqual(@as(usize, 1), ids.len);
            id.* = ids[0];
        }
        const plain_ids = try text.encode(a, "ok", false);
        b.mock = .{ .dev = &b.rig.dev, .trace = &b.rig.trace, .text = text, .order = order, .eos = text.eosIds(), .plain = plain_ids[0] };
        b.rig.trace.picker = b.mock.picker();
        b.rig.dev.emulate = b.mock.emulator();
        b.on = .{ .at = &b.rig.at, .f = &b.rig.f, .ctx = null, .node = null, .plan = plan, .load_ms = 1234.5, .grammar = grammar };
        std.debug.print("structured_entry boot: users {d} pool {d} interleave {any} sampling {} grammar {} window {d} capacity {d}; tokenizer dir {s}; port {d}; order ids {any}; eos {any}; plain {d}\n", .{ plan.users, plan.pool, plan.interleave, sampling, grammar != null, plan.window, plan.capacity, dir, b.port, order, b.mock.eos, b.mock.plain });
        b.thread = try std.Thread.spawn(.{}, run, .{b});
        // the server listens once the warm-up passed: wait for its /health
        var tries: usize = 0;
        while (tries < 6000) : (tries += 1) {
            if (b.done.load(.acquire)) break;
            var scratch = std.heap.ArenaAllocator.init(gpa);
            defer scratch.deinit();
            const raw = ask(scratch.allocator(), b.port, "GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n") catch {
                std.Io.sleep(t.io, .fromMilliseconds(50), .awake) catch {};
                continue;
            };
            if (h.status(raw) == 200) {
                b.listening = true;
                return;
            }
            std.Io.sleep(t.io, .fromMilliseconds(50), .awake) catch {};
        }
        if (b.done.load(.acquire)) b.thread.join();
        std.debug.print("structured_entry boot: the server never answered /health: {any}\n", .{b.result});
        return error.ServerNotUp;
    }

    fn run(b: *Boot) void {
        b.result = served.run(gpa, t.io, b.flags, b.on);
        b.done.store(true, .release);
    }

    /// Ash's server's stop (served.stopUsersServer: SIGTERM to this process), then the entry's own end
    fn stop(b: *Boot) !usize {
        if (!b.listening) return error.ServerNotUp; // no handler installed: a SIGTERM would end this process
        served.stopUsersServer(null);
        b.thread.join();
        b.hf.deinit();
        b.rig.deinit();
        b.arena.deinit();
        for ([_][:0]const u8{ "TF_AFD_USERS", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE", "TF_AFD_GRAMMAR_SERVE", "TF_AFD_GRAMMAR_LIB", "TF_AFD_GRAMMAR_LIB_SHA256", "TF_GLM_SAMPLING" }) |n| _ = unsetenv(n);
        return b.result;
    }
};

fn show(r: Reply) void {
    std.debug.print("structured_entry reply {s}: HTTP {d} finish {s} content \"{s}\" usage {s}{s}{s}\n", .{ r.label, r.status, r.finish, r.content[0..@min(r.content.len, 160)], r.usage, if (r.message.len > 0) " message: " else "", r.message });
}

const Tally = struct {
    structured: usize = 0,
    structured_ok: usize = 0,
    first_bad: ?[]const u8 = null,
    fn add(x: *Tally, a: std.mem.Allocator, r: Reply) void {
        if (r.kind == .plain) return;
        x.structured += 1;
        if (structuredOk(a, r)) x.structured_ok += 1 else if (x.first_bad == null) x.first_bad = r.label;
    }
};

fn expectStructured(a: std.mem.Allocator, tally: *const Tally, replies: []const Reply) !void {
    _ = a;
    if (tally.structured_ok != tally.structured) {
        for (replies) |r| if (r.kind != .plain and r.status != 200) {
            std.debug.print("structured_entry RED-CHECK structured {s}: HTTP {d} \"{s}\"\n", .{ r.label, r.status, r.message });
        };
        std.debug.print("structured_entry structured valid {d}/{d}; first not valid: {s}\n", .{ tally.structured_ok, tally.structured, tally.first_bad orelse "-" });
        return error.StructuredNotServed;
    }
}

// ---- the cases --------------------------------------------------------------------------------------------------------

test "structured_entry u2: the users server entry at TF_AFD_USERS=2 + TF_AFD_GRAMMAR_SERVE=1 over the real GLM tokenizer: structured_users_b gate 4's sequence (solo references, mixed and both pairs x2): every structured reply 200/stop/parsed/validated, paired plain = solo" {
    if (!case("u2")) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var b: Boot = .{};
    try b.start(&.{ .{ "TF_AFD_USERS", "2" }, .{ "TF_AFD_GRAMMAR_MASK", "1" }, .{ "TF_AFD_GRAMMAR_SERVE", "1" } });
    var up = true;
    defer if (up) {
        _ = b.stop() catch {};
    };
    var all: std.ArrayList(Reply) = .empty;
    var tally: Tally = .{};
    // 1 solo references, one at a time (structured_users_probe SOLO)
    const solo_reqs = [_]Req{ plainA(false), plainB(true), schemaReq(false), schemaReq(true), objectReq(false), objectReq(true) };
    var solo: [solo_reqs.len]Reply = undefined;
    for (solo_reqs, &solo) |r, *s| {
        s.* = try send(a, b.port, r);
        show(s.*);
        tally.add(a, s.*);
        try all.append(a, s.*);
    }
    // 2 the pairs (structured_users_probe PAIRS) x REPEATS 2, each pair released together
    const pairs = [_]struct { []const u8, [2]Req }{
        .{ "mixed-1", .{ schemaReq(false), plainA(false) } },     .{ "mixed-2", .{ objectReq(true), plainB(true) } },
        .{ "mixed-3", .{ plainA(false), schemaReq(true) } },      .{ "mixed-4", .{ plainB(true), objectReq(false) } },
        .{ "both-1", .{ schemaReq(false), objectReq(false) } },   .{ "both-2", .{ schemaReq(true), objectReq(true) } },
        .{ "both-3", .{ schemaReq(false), schemaReq(true) } },    .{ "both-4", .{ objectReq(false), schemaReq(true) } },
    };
    var plain_same: usize = 0;
    var plain_paired: usize = 0;
    var in_flight: usize = 0;
    var n_pairs: usize = 0;
    for (0..2) |rep| for (pairs) |p| {
        const rs = try together(a, b.port, &p[1]);
        n_pairs += 1;
        if (overlapped(rs)) in_flight += 1;
        for (rs) |r| {
            std.debug.print("structured_entry pair {s} repeat {d}: ", .{ p[0], rep });
            show(r);
            tally.add(a, r);
            try all.append(a, r);
            if (r.kind == .plain) {
                plain_paired += 1;
                const ref = if (std.mem.startsWith(u8, r.label, "u-plain-a")) solo[0] else solo[1];
                if (same(r, ref)) plain_same += 1 else std.debug.print("structured_entry pair {s}: plain {s} differs from its solo reference\n", .{ p[0], r.label });
            }
        }
    };
    up = false;
    const served_n = try b.stop();
    std.debug.print("structured_entry u2 summary: structured valid {d}/{d} (solo 4 + mixed 8 + both 16); paired plain = solo {d}/{d}; pairs in flight together {d}/{d} (recorded); requests served {d}; draws plain {d} masked {d}; trace resets {d}\n", .{ tally.structured_ok, tally.structured, plain_same, plain_paired, in_flight, n_pairs, served_n, b.mock.plain_draws, b.mock.masked_greedy, b.mock.resets });
    try expectStructured(a, &tally, all.items);
    try t.expectEqual(@as(usize, 28), tally.structured);
    try t.expectEqual(@as(usize, 8), plain_paired);
    try t.expectEqual(plain_paired, plain_same);
    try t.expect(b.mock.masked_greedy > 0);
}

test "structured_entry u4: the users server entry at TF_AFD_USERS=4 + TF_AFD_POOL_TOKENS=720896 + TF_AFD_USERS_INTERLEAVE=1 + TF_AFD_GRAMMAR_SERVE=1 + TF_GLM_SAMPLING=1: the U4G mix (2 structured, 1 sampled, 1 plain greedy) x3: structured valid, greedy = solo, no PoolFull" {
    if (!case("u4")) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var b: Boot = .{};
    try b.start(&.{ .{ "TF_AFD_USERS", "4" }, .{ "TF_AFD_POOL_TOKENS", "720896" }, .{ "TF_AFD_USERS_INTERLEAVE", "1" }, .{ "TF_AFD_GRAMMAR_MASK", "1" }, .{ "TF_AFD_GRAMMAR_SERVE", "1" }, .{ "TF_GLM_SAMPLING", "1" } });
    var up = true;
    defer if (up) {
        _ = b.stop() catch {};
    };
    var all: std.ArrayList(Reply) = .empty;
    var tally: Tally = .{};
    const solo_reqs = [_]Req{ plainA(false), objectReq(false), schemaReq(false) };
    var solo: [solo_reqs.len]Reply = undefined;
    for (solo_reqs, &solo) |r, *s| {
        s.* = try send(a, b.port, r);
        show(s.*);
        tally.add(a, s.*);
        try all.append(a, s.*);
    }
    var greedy_same: usize = 0;
    var sampled_refused: usize = 0;
    var no_pool_full = true;
    var in_flight: usize = 0;
    for (0..3) |round| {
        var sampled = plainB(false);
        sampled.label = "u-sampled";
        sampled.sampling = try std.fmt.allocPrint(a, ",\"temperature\":0.7,\"top_p\":0.95,\"seed\":{d}", .{1000 + round});
        const mix = [_]Req{ objectReq(false), schemaReq(true), sampled, plainA(false) };
        const rs = try together(a, b.port, &mix);
        if (overlapped(rs)) in_flight += 1;
        for (rs) |r| {
            std.debug.print("structured_entry mix round {d}: ", .{round});
            show(r);
            try all.append(a, r);
            if (r.status >= 500) no_pool_full = false;
            if (std.mem.indexOf(u8, r.message, "PoolFull") != null or std.mem.indexOf(u8, r.content, "PoolFull") != null) no_pool_full = false;
            if (std.mem.eql(u8, r.label, "u-sampled")) {
                if (r.status == 400 and std.mem.eql(u8, r.message, "TF_AFD_USERS supports temperature 0 only; sampled multi-user decode is unsupported")) sampled_refused += 1;
                continue;
            }
            tally.add(a, r);
            if (r.kind == .plain) {
                if (same(r, solo[0])) greedy_same += 1 else std.debug.print("structured_entry mix round {d}: plain greedy differs from its solo reference\n", .{round});
            }
        }
    }
    up = false;
    const served_n = try b.stop();
    std.debug.print("structured_entry u4 summary: structured valid {d}/{d} (solo 2 + mix 6); greedy = solo {d}/3; sampled request refused by name {d}/3 (TF_AFD_USERS: temperature 0 only; recorded); no 5xx/PoolFull {}; mixes in flight together {d}/3 (recorded); requests served {d}; draws plain {d} masked {d}; trace resets {d}\n", .{ tally.structured_ok, tally.structured, greedy_same, sampled_refused, no_pool_full, in_flight, served_n, b.mock.plain_draws, b.mock.masked_greedy, b.mock.resets });
    try expectStructured(a, &tally, all.items);
    try t.expectEqual(@as(usize, 8), tally.structured);
    try t.expectEqual(@as(usize, 3), greedy_same);
    try t.expect(no_pool_full);
    try t.expectEqual(@as(usize, 3), sampled_refused);
}

test "structured_entry sp: the single-user entry with TF_GLM_SAMPLING=1 + TF_AFD_GRAMMAR_SERVE=1 over the real GLM tokenizer: structured_users_b gate 5's sequence (24 sampled structured replies, json_schema/json_object, whole/streamed, three settings x 8 seeds, then two repeats): 200/stop/parsed/validated" {
    if (!case("sp")) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var b: Boot = .{};
    try b.start(&.{ .{ "TF_GLM_SAMPLING", "1" }, .{ "TF_AFD_GRAMMAR_MASK", "1" }, .{ "TF_AFD_GRAMMAR_SERVE", "1" } });
    var up = true;
    defer if (up) {
        _ = b.stop() catch {};
    };
    var all: std.ArrayList(Reply) = .empty;
    var tally: Tally = .{};
    const settings = [_][]const u8{ ",\"temperature\":0.7,\"top_p\":0.95", ",\"temperature\":1.0,\"top_p\":0.95", ",\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0" };
    var plan: std.ArrayList(Req) = .empty;
    for (0..24) |i| {
        var r = if (i % 2 == 0) schemaReq((i / 2) % 2 == 1) else objectReq((i / 2) % 2 == 1);
        r.label = try std.fmt.allocPrint(a, "s{d:0>2}-{s}{s}", .{ i, if (i % 2 == 0) "json_schema" else "json_object", if (r.stream) "-stream" else "" });
        r.tokens = sampled_tokens;
        r.sampling = try std.fmt.allocPrint(a, "{s},\"seed\":{d}", .{ settings[i * settings.len / 24], 1000 + i });
        try plan.append(a, r);
    }
    for (0..2) |k| {
        var r = plan.items[k];
        r.label = try std.fmt.allocPrint(a, "{s}-repeat", .{r.label});
        try plan.append(a, r);
    }
    var repeats_same: usize = 0;
    var first: [2]Reply = undefined;
    for (plan.items, 0..) |r, i| {
        const x = try send(a, b.port, r);
        show(x);
        try all.append(a, x);
        if (i < 24) tally.add(a, x);
        if (i < 2) first[i] = x;
        if (i >= 24 and same(x, first[i - 24])) repeats_same += 1;
    }
    up = false;
    const served_n = try b.stop();
    std.debug.print("structured_entry sp summary: sampled structured valid {d}/{d}; same-seed repeats equal {d}/2 (recorded); requests served {d}; draws plain {d} masked sampled {d} masked greedy {d}; trace resets {d}\n", .{ tally.structured_ok, tally.structured, repeats_same, served_n, b.mock.plain_draws, b.mock.masked_sampled, b.mock.masked_greedy, b.mock.resets });
    try expectStructured(a, &tally, all.items);
    try t.expectEqual(@as(usize, 24), tally.structured);
    try t.expect(b.mock.masked_sampled > 0);
}
