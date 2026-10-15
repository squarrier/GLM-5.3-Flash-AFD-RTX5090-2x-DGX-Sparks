//! grammar_server test helpers (Claude Opus 5.5 via Hermes): Ash's server on a loopback port with one raw HTTP/1.1 exchange
//! a connection (tests/server/glm_cuda_serve.zig's `ask`), and a byte-level text. Only API the pristine 045dd856 has,
//! so the wire A/B compiles this same file on both trees. CPU only; no device, no network beyond 127.0.0.1.
const std = @import("std");
const server = @import("server");
const api = @import("engine_api");
const posix = std.posix;
const testing = std.testing;
const gpa = testing.allocator;

/// The server's accept loop on 127.0.0.1:<ephemeral> until `finish`.
pub const Http = struct {
    srv: *server.Server = undefined,
    lis: server.listener.Listener = undefined,
    stop: std.atomic.Value(bool) = .init(false),
    thread: std.Thread = undefined,

    pub fn start(h: *Http, srv: *server.Server) !void {
        h.srv = srv;
        h.lis = try server.listener.Listener.open(.{ .ip4 = .loopback(0) });
        h.thread = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, h.lis, &h.stop });
    }

    /// Stops accepting and waits (at most 5 s) for the open connections, as serve.run does before freeing.
    pub fn finish(h: *Http) void {
        h.stop.store(true, .release);
        h.thread.join();
        h.lis.close();
        var waited: u32 = 0;
        while (h.srv.open_connections.load(.acquire) > 0 and waited < 100) : (waited += 1) std.Io.sleep(testing.io, .fromMilliseconds(50), .awake) catch {};
    }

    /// One exchange on a new connection (the request closes it): the reply's raw bytes.
    pub fn ask(h: *Http, a: std.mem.Allocator, raw: []const u8) ![]u8 {
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const fd: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(fd);
        const addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, h.lis.port()), .addr = 1 << 24 | 127 };
        if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.ConnectFailed;
        const timeout: posix.timeval = .{ .sec = 60, .usec = 0 };
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

    pub fn post(h: *Http, a: std.mem.Allocator, path: []const u8, payload: []const u8) ![]u8 {
        return h.ask(a, try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {d}\r\n\r\n{s}", .{ path, payload.len, payload }));
    }

    pub fn get(h: *Http, a: std.mem.Allocator, path: []const u8) ![]u8 {
        return h.ask(a, try std.fmt.allocPrint(a, "GET {s} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", .{path}));
    }
};

/// The status code of a raw reply ("HTTP/1.1 400 Bad Request").
pub fn status(raw: []const u8) u16 {
    if (raw.len < 12) return 0;
    return std.fmt.parseInt(u16, raw[9..12], 10) catch 0;
}

/// The body of a raw reply (after the blank line).
pub fn body(raw: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return "";
    return raw[at + 4 ..];
}

/// A whole (non-stream) reply's JSON body, parsed.
pub fn json(a: std.mem.Allocator, raw: []const u8) !server.json.Value {
    return switch (try server.json.parse(a, body(raw))) {
        .ok => |v| v,
        .err => error.NotJson,
    };
}

/// error.message of a refusal (or a bare {"message"}).
pub fn message(a: std.mem.Allocator, raw: []const u8) ![]const u8 {
    const v = try json(a, raw);
    const e = v.get("error") orelse v;
    return (e.get("message") orelse return error.NoMessage).string;
}

/// choices[0].message.content of a whole chat reply.
pub fn content(a: std.mem.Allocator, raw: []const u8) ![]const u8 {
    const v = try json(a, raw);
    const m = v.get("choices").?.array[0].get("message").?;
    return switch (m.get("content").?) {
        .string => |s| s,
        else => "",
    };
}

pub fn finishReason(a: std.mem.Allocator, raw: []const u8) ![]const u8 {
    const v = try json(a, raw);
    return v.get("choices").?.array[0].get("finish_reason").?.string;
}

/// grammar_server: a byte-level text for the loopback tests (no template).
pub const ByteText = struct {
    eos: []const u32 = &.{},
    pub fn text(t: *ByteText) server.model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) server.model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| {
            if (id < 256) try out.append(a, @intCast(id)) else try out.print(a, "<{d}>", .{id});
        }
        return out.toOwnedSlice(a);
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 154880;
    }
    fn eosIds(ctx: *anyopaque) []const u32 {
        const t: *ByteText = @ptrCast(@alignCast(ctx));
        return t.eos;
    }
    fn render(_: *anyopaque, a: std.mem.Allocator, _: server.json.Value, _: server.model_text.RenderOptions, _: *[]const u8) server.model_text.Error![]u8 {
        return a.dupe(u8, "hi");
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// A chat completion body (greedy, `max_tokens`), with `extra` (",\"response_format\": ...") appended.
pub fn chat(a: std.mem.Allocator, max_tokens: u32, stream: bool, extra: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"glm-5.3-flash\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":{d},\"temperature\":0,\"stream\":{s}{s}}}", .{ max_tokens, if (stream) "true" else "false", extra });
}
