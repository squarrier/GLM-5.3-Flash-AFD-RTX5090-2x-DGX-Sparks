//! user_sampling (local integration; Claude Opus 5.5 via Hermes): test-user_sampling_served's requests, replies and loopback HTTP —
//! structured_entry's structured_entry_served_test.zig scaffolding (structured_users_b's probe bodies, the de-chunked whole/SSE reply, the two formats'
//! validation, one barrier for requests released together), plus the reply's native extras sampling_tests_stat's client reads
//! (tensorfold.token_sha / sampling / seconds / prefill_seconds; speculative.drafted / accepted as explicit integers;
//! a stream's from its finish packet), an omitted-temperature body ("-"), a client that goes away mid-stream, and
//! /metrics' requests_running. No network beyond 127.0.0.1.
//! Scaffolding: structured_entry's (Claude Opus 5.5 via Hermes) after concurrent_users_r2's (GPT-6.1 Sol via Hermes) and Ash Hart's tests;
//! inherited NOTICE credits apply (Ash Hart / TensorFold; the XGrammar contributors (MLC, Apache-2.0)).
const std = @import("std");
const server = @import("server");
const h = @import("grammar_server_http.zig");
const posix = std.posix;
const t = std.testing;

pub const name = "glm-5.3-flash-afd";

// ---- requests ---------------------------------------------------------------------------------------------------------

pub const schema_json = "{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\",\"enum\":[\"yes\",\"no\"]},\"confidence\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":100},\"sky\":{\"type\":\"object\",\"properties\":{\"color\":{\"type\":\"string\"},\"clear\":{\"type\":\"boolean\"}},\"required\":[\"color\",\"clear\"],\"additionalProperties\":false},\"reasons\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"minItems\":1,\"maxItems\":2}},\"required\":[\"answer\",\"confidence\",\"sky\",\"reasons\"],\"additionalProperties\":false}";

pub const Kind = enum { plain, json_object, json_schema };
pub const Req = struct {
    label: []const u8,
    kind: Kind = .plain,
    prompt: []const u8 = "",
    tokens: u32 = 64,
    stream: bool = false,
    /// the request's sampling members verbatim (",\"temperature\":0.8,\"seed\":1"); "" sends temperature 0 and seed 7;
    /// "-" sends no temperature and no seed (the server's defaults decide)
    sampling: []const u8 = "",
};

/// structured_entry's body (served_probe.request_body's members in its order), plus "-" for an omitted temperature
pub fn body(a: std.mem.Allocator, r: Req) ![]u8 {
    const temp: []const u8 = if (r.sampling.len == 0) ",\"temperature\":0,\"seed\":7" else if (std.mem.eql(u8, r.sampling, "-")) "" else r.sampling;
    const fmt: []const u8 = switch (r.kind) {
        .plain => "",
        .json_object => ",\"response_format\":{\"type\":\"json_object\"}",
        .json_schema => try std.fmt.allocPrint(a, ",\"response_format\":{{\"type\":\"json_schema\",\"json_schema\":{{\"name\":\"sky\",\"schema\":{s}}}}}", .{schema_json}),
    };
    return std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}]{s},\"max_tokens\":{d},\"draft\":false,\"chat_template_kwargs\":{{\"enable_thinking\":false}},\"stream\":{s}{s}{s}}}", .{ name, r.prompt, temp, r.tokens, if (r.stream) "true" else "false", if (r.stream) ",\"stream_options\":{\"include_usage\":true}" else "", fmt });
}

// ---- replies: status, content, finish, usage, and the native extras (tensorfold, speculative) --------------------------

pub const Reply = struct {
    label: []const u8,
    kind: Kind,
    stream: bool = false,
    status: u16 = 0,
    content: []const u8 = "",
    finish: []const u8 = "",
    usage: []const u8 = "",
    message: []const u8 = "",
    token_sha: []const u8 = "",
    sampling: []const u8 = "",
    /// speculative.drafted / accepted when each is a JSON integer (else null: missing or another type)
    drafted: ?i64 = null,
    accepted: ?i64 = null,
    prompt_tokens: i64 = 0,
    completion_tokens: i64 = 0,
    seconds: f64 = 0,
    prefill_seconds: f64 = 0,
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

/// a JSON integer (never a bool or a float) as i64, else null
fn int(v: ?server.json.Value) ?i64 {
    const x = v orelse return null;
    return x.int64();
}

fn num(v: ?server.json.Value) f64 {
    const x = v orelse return 0;
    return if (x == .bool) 0 else x.float64() orelse 0;
}

/// the extras a whole reply carries at its top level and a stream in its finish packet (openai.zig `extras`)
fn extras(out: *Reply, v: server.json.Value) void {
    if (v.get("tensorfold")) |tf| {
        out.token_sha = str(tf.get("token_sha"));
        out.sampling = str(tf.get("sampling"));
        out.seconds = num(tf.get("seconds"));
        out.prefill_seconds = num(tf.get("prefill_seconds"));
    }
    if (v.get("speculative")) |sp| {
        out.drafted = int(sp.get("drafted"));
        out.accepted = int(sp.get("accepted"));
    }
}

fn counts(out: *Reply, u: server.json.Value) void {
    out.prompt_tokens = int(u.get("prompt_tokens")) orelse 0;
    out.completion_tokens = int(u.get("completion_tokens")) orelse 0;
}

pub fn parseReply(a: std.mem.Allocator, r: Req, raw: []const u8) !Reply {
    var out: Reply = .{ .label = r.label, .kind = r.kind, .stream = r.stream, .status = h.status(raw) };
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
        if (v.get("usage")) |u| {
            out.usage = try server.json.stringify(a, u, .{});
            counts(&out, u);
        }
        extras(&out, v);
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
            counts(&out, u);
        };
        const choices = v.get("choices") orelse continue;
        if (choices != .array or choices.array.len == 0) continue;
        const c = choices.array[0];
        if (c.get("delta")) |d| try content.appendSlice(a, str(d.get("content")));
        const fr = str(c.get("finish_reason"));
        if (fr.len > 0) {
            out.finish = fr;
            extras(&out, v); // sampling_tests_stat: the finish packet's own extras, never an earlier or the usage packet's
        }
    }
    out.content = content.items;
    return out;
}

/// structured_users_probe's IDENTITY plus the server's token_sha: content, finish, the usage counts, the token sha
pub fn same(x: Reply, y: Reply) bool {
    return x.status == y.status and std.mem.eql(u8, x.content, y.content) and std.mem.eql(u8, x.finish, y.finish) and
        std.mem.eql(u8, x.usage, y.usage) and std.mem.eql(u8, x.token_sha, y.token_sha);
}

// ---- validation (served_probe.validate's rules for the two formats) ---------------------------------------------------

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

/// A structured reply as structured_users_b's gates count it: HTTP 200, `stop`, parsed, validated.
pub fn structuredOk(a: std.mem.Allocator, r: Reply) bool {
    if (r.status != 200 or !std.mem.eql(u8, r.finish, "stop")) return false;
    return switch (r.kind) {
        .json_object => isObject(a, r.content),
        .json_schema => validSky(a, r.content),
        .plain => false,
    };
}

/// A plain reply well formed: HTTP 200, stop or length, its usage counted, its token sha present, valid UTF-8.
pub fn plainOk(r: Reply) bool {
    return r.status == 200 and (std.mem.eql(u8, r.finish, "stop") or std.mem.eql(u8, r.finish, "length")) and
        r.completion_tokens > 0 and r.token_sha.len > 0 and std.unicode.utf8ValidateSlice(r.content);
}

/// sampling_tests_stat's require_native_sample: drafted and accepted explicit integer zeros, tensorfold.sampling "exact".
pub fn nativeSampled(r: Reply) bool {
    return r.drafted != null and r.drafted.? == 0 and r.accepted != null and r.accepted.? == 0 and std.mem.eql(u8, r.sampling, "exact");
}

// ---- HTTP over loopback -------------------------------------------------------------------------------------------------

pub fn ask(a: std.mem.Allocator, port: u16, raw: []const u8) ![]u8 {
    return askCut(a, port, raw, null);
}

/// `cut`: close the connection once that many reply bytes arrived (a client that goes away mid-stream)
pub fn askCut(a: std.mem.Allocator, port: u16, raw: []const u8, cut: ?usize) ![]u8 {
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
        if (cut) |c| if (out.items.len >= c) break;
    }
    return out.toOwnedSlice(a);
}

pub fn postRaw(a: std.mem.Allocator, port: u16, payload: []const u8) ![]u8 {
    return ask(a, port, try std.fmt.allocPrint(a, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {d}\r\n\r\n{s}", .{ payload.len, payload }));
}

pub fn get(a: std.mem.Allocator, port: u16, path: []const u8) ![]u8 {
    return ask(a, port, try std.fmt.allocPrint(a, "GET {s} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", .{path}));
}

pub fn now() i96 {
    return std.Io.Clock.awake.now(t.io).toNanoseconds();
}

pub fn send(a: std.mem.Allocator, port: u16, r: Req) !Reply {
    const t0 = now();
    const raw = try postRaw(a, port, try body(a, r));
    var out = try parseReply(a, r, raw);
    out.t0 = t0;
    out.t1 = now();
    return out;
}

/// Requests released together by one barrier, each on its own connection, thread and arena (structured_entry's).
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

pub fn copyReply(a: std.mem.Allocator, r: Reply) !Reply {
    var x = r;
    x.label = try a.dupe(u8, r.label);
    x.content = try a.dupe(u8, r.content);
    x.finish = try a.dupe(u8, r.finish);
    x.usage = try a.dupe(u8, r.usage);
    x.message = try a.dupe(u8, r.message);
    x.token_sha = try a.dupe(u8, r.token_sha);
    x.sampling = try a.dupe(u8, r.sampling);
    return x;
}

pub fn together(a: std.mem.Allocator, port: u16, reqs: []const Req) ![]Reply {
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
        std.debug.print("user_sampling request {s}: {s}\n", .{ r.label, @errorName(x) });
        return x;
    };
    for (b.replies) |*r| r.* = try copyReply(a, r.*);
    return b.replies;
}

pub fn overlapped(rs: []const Reply) bool {
    var first_end: i96 = std.math.maxInt(i96);
    var last_start: i96 = 0;
    for (rs) |r| {
        first_end = @min(first_end, r.t1);
        last_start = @max(last_start, r.t0);
    }
    return last_start < first_end;
}

/// tensorfold:requests_running from /metrics (null: not found)
pub fn running(a: std.mem.Allocator, port: u16) !?i64 {
    const raw = try get(a, port, "/metrics");
    const text = try dechunk(a, raw);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, "tensorfold:requests_running ")) continue;
        const v = std.mem.trim(u8, line["tensorfold:requests_running ".len..], " \r");
        return std.fmt.parseInt(i64, v, 10) catch @as(i64, @intFromFloat(std.fmt.parseFloat(f64, v) catch return null));
    }
    return null;
}
