//! qualification_qualification_44ba554e.8 day 1: the CUDA GLM's lane backend (native/glm_cuda_lanes.zig) under Ash's server over loopback HTTP, with a
//! scripted forward. A chat goes in, is rendered with GLM-5.3's template (the kit's files: TF_GLM_TOKENIZER), and the
//! token stream comes back as the reply and its SSE. Also: a chunked body (0094), and a full server's 429 with
//! Retry-After before a stream's 200 (Mia's 0095).
const std = @import("std");
const server = @import("server");
const api = @import("engine_api");
const lanes = @import("lanes");
const gl = @import("glm_lanes");
const posix = std.posix;
const testing = std.testing;
const gpa = testing.allocator;

/// The scripted forward: the reply's ids in turn; while `hold` is set a step waits (a lane held busy).
const Script = struct {
    reply: []const u32 = &.{},
    len: usize = 0,
    at: usize = 0,
    prompt: [4096]u32 = undefined,
    prompt_len: usize = 0,
    hold: std.atomic.Value(bool) = .init(false),

    pub fn prefill(s: *Script, ids: []const u32) !u32 {
        @memcpy(s.prompt[0..ids.len], ids);
        s.prompt_len = ids.len;
        s.len = ids.len;
        s.at = 0;
        return s.reply[0];
    }

    pub fn step(s: *Script, _: u32) !u32 {
        while (s.hold.load(.acquire)) std.Io.sleep(testing.io, .fromMilliseconds(2), .awake) catch {};
        s.len += 1;
        s.at += 1;
        return s.reply[@min(s.at, s.reply.len - 1)];
    }

    pub fn release(s: *Script) void {
        s.len = 0;
    }
};

/// Ash's server over the lane host, the lane core and the CUDA GLM's backend, listening on a loopback port.
const Stack = struct {
    arena: std.heap.ArenaAllocator,
    script: Script = .{},
    back: gl.Lanes(Script) = undefined,
    cfg: lanes.Config = undefined,
    clock: lanes.backend.WallClock = undefined,
    core: lanes.Engine = undefined,
    host: api.LaneHost = undefined,
    hf: *server.hf_text.HfText = undefined,
    glm: server.glm_text.GlmText = undefined,
    srv: *server.Server = undefined,
    lis: server.listener.Listener = undefined,
    stop: std.atomic.Value(bool) = .init(false),
    thread: std.Thread = undefined,

    fn start(dir: []const u8, reply: []const u8, max_queued: ?u32) !*Stack {
        const io = testing.io;
        const k = try gpa.create(Stack);
        k.* = .{ .arena = .init(gpa) };
        const a = k.arena.allocator();
        var problem: []const u8 = "";
        k.hf = try server.hf_text.HfText.load(gpa, io, dir, a, &problem);
        k.glm = .{ .inner = k.hf.text(), .clear = false };
        const ids = try k.hf.text().encode(a, reply, false);
        k.script.reply = try std.mem.concat(a, u32, &.{ ids, &.{k.hf.text().eosIds()[0]} });
        k.back = .{ .steps = &k.script, .capacity = 262_144 + 64, .reach = 2051 };
        k.cfg = try lanes.Config.init(gpa, k.back.facts(), 1, 0);
        k.clock = .{ .io = io };
        k.core = lanes.Engine.init(gpa, &k.cfg, k.back.backend(), k.clock.clock());
        k.host = api.LaneHost.init(gpa, io, &k.core, .{ .name = "glm-zig", .lanes = 1, .context_window = 262_144, .prefill_step = 2048 });
        k.host.explain = .{ .text = gl.explain };
        k.host.max_queued = max_queued;
        try k.host.start();
        const greedy = (try server.json.parse(a, "{\"temperature\": 0}")).ok; // the serve command's --temperature 0
        k.srv = try server.Server.init(gpa, io, k.host.engine(), k.glm.text(), .{ .served_name = "glm-5.3-flash-zig", .model_ids = &.{"glm-5.3-flash-zig"}, .enable_thinking = true, .default_sampling = greedy, .use_drafts = false }, null);
        k.lis = try server.listener.Listener.open(.{ .ip4 = .loopback(0) });
        k.thread = try std.Thread.spawn(.{}, server.Server.serve, .{ k.srv, k.lis, &k.stop });
        return k;
    }

    fn deinit(k: *Stack) void {
        k.stop.store(true, .release);
        k.thread.join();
        k.lis.close();
        k.srv.deinit();
        k.host.stop();
        k.core.deinit();
        k.cfg.deinit(gpa);
        k.hf.deinit();
        k.arena.deinit();
        gpa.destroy(k);
    }

    /// One HTTP exchange on a new connection (the request closes it): the reply's bytes.
    fn ask(k: *Stack, a: std.mem.Allocator, raw: []const u8) ![]u8 {
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const fd: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(fd);
        const addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, k.lis.port()), .addr = 1 << 24 | 127 };
        if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.ConnectFailed;
        const timeout: posix.timeval = .{ .sec = 20, .usec = 0 };
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
};

fn post(a: std.mem.Allocator, body: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {d}\r\n\r\n{s}", .{ body.len, body });
}

fn kit() ![]const u8 {
    return std.mem.span(std.c.getenv("TF_GLM_TOKENIZER") orelse return error.SkipZigTest);
}

const question = "{\"model\": \"glm-5.3-flash-zig\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}], \"max_tokens\": 32";

fn expectHas(hay: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, hay, needle) == null) {
        std.debug.print("missing {s} in:\n{s}\n", .{ needle, hay });
        return error.Missing;
    }
}

test "a chat through Ash's server and the CUDA GLM's lane backend: GLM-5.3's prompt in, the reply and its SSE out" {
    const k = try Stack.start(try kit(), "Let me check.</think>The answer is 4.", null);
    defer k.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = try k.ask(a, try post(a, question ++ "}"));
    try expectHas(whole, "HTTP/1.1 200 OK");
    try expectHas(whole, "\"reasoning_content\": \"Let me check.\"");
    try expectHas(whole, "\"content\": \"The answer is 4.\"");
    try expectHas(whole, "\"finish_reason\": \"stop\"");
    // the forward got python-0.6's GLM-5.3 prompt (thinking on: the effort line, an opened think block)
    const seen = try k.hf.text().decode(a, k.script.prompt[0..k.script.prompt_len]);
    try testing.expectEqualStrings("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>What is 2+2?<|assistant|><think>", seen);
    const sse = try k.ask(a, try post(a, question ++ ", \"stream\": true}"));
    try expectHas(sse, "HTTP/1.1 200 OK");
    try expectHas(sse, "Content-Type: text/event-stream");
    try expectHas(sse, "\"role\": \"assistant\"");
    try expectHas(sse, "\"reasoning_content\": \"");
    try expectHas(sse, "\"finish_reason\": \"stop\"");
    try expectHas(sse, "data: [DONE]");
    // thinking off: python-0.6's thinking_off prompt
    _ = try k.ask(a, try post(a, question ++ ", \"chat_template_kwargs\": {\"enable_thinking\": false}}"));
    const off = try k.hf.text().decode(a, k.script.prompt[0..k.script.prompt_len]);
    try testing.expectEqualStrings("[gMASK]<sop><|user|>What is 2+2?<|assistant|><think></think>", off);
}

test "a chunked body is read whole (0094); a sampled request is refused in words" {
    const k = try Stack.start(try kit(), "Sure.</think>Four.", null);
    defer k.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const body = question ++ "}";
    const half = body.len / 2;
    const raw = try std.fmt.allocPrint(a, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n{x}\r\n{s}\r\n{x}\r\n{s}\r\n0\r\n\r\n", .{ half, body[0..half], body.len - half, body[half..] });
    const got = try k.ask(a, raw);
    try expectHas(got, "HTTP/1.1 200 OK");
    try expectHas(got, "\"content\": \"Four.\"");
    const sampled = try k.ask(a, try post(a, question ++ ", \"temperature\": 0.7}"));
    try expectHas(sampled, "decodes greedily only");
}

var held_done: std.atomic.Value(bool) = .init(false);

fn hold(k: *Stack, out: *[]u8, a: std.mem.Allocator) void {
    defer held_done.store(true, .release);
    out.* = k.ask(a, post(a, question ++ "}") catch return) catch return;
}

test "a full server answers a stream 429 with Retry-After before its 200 (Mia's 0095), then serves again" {
    const k = try Stack.start(try kit(), "Four.", 0);
    defer k.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    k.script.hold.store(true, .release);
    var first: []u8 = &.{};
    const t = try std.Thread.spawn(.{}, hold, .{ k, &first, std.heap.page_allocator });
    defer std.heap.page_allocator.free(first);
    defer t.join(); // before the stack goes, on any path
    defer k.script.hold.store(false, .release);
    var waited: usize = 0;
    while (k.srv.engine.admits(false)) : (waited += 1) {
        if (waited > 2000) return error.NeverBusy;
        std.Io.sleep(testing.io, .fromMilliseconds(5), .awake) catch {};
    }
    const refused = try k.ask(a, try post(a, question ++ ", \"stream\": true}"));
    try expectHas(refused, "HTTP/1.1 429 Too Many Requests");
    try expectHas(refused, "Retry-After: 5");
    try expectHas(refused, "all serving lanes are busy; retry shortly");
    try testing.expect(std.mem.indexOf(u8, refused, "text/event-stream") == null);
    k.script.hold.store(false, .release);
    while (!held_done.load(.acquire) and waited < 4000) : (waited += 1) std.Io.sleep(testing.io, .fromMilliseconds(5), .awake) catch {};
    try expectHas(first, "HTTP/1.1 200 OK");
    const again = try k.ask(a, try post(a, question ++ ", \"stream\": true}"));
    try expectHas(again, "data: [DONE]");
}
