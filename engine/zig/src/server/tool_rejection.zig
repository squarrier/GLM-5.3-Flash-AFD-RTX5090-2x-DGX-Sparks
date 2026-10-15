//! qualification_214a6e4c: name-only rejection logging. @BadAd84 authored Mia PR137's design,
//! 616b840651a428d514607c3ce5524ed5bc4c206d; Apache-2.0, no Python code copied.
//! GPT-6.1 Sol via Hermes authored this default-off native implementation.
const std = @import("std");

pub fn mode(value: ?[]const u8) error{InvalidToolRejectionLog}!bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.InvalidToolRejectionLog;
}

/// An owned JSON string, including quotes, at most 128 UTF-8 bytes.
pub const Name = struct {
    bytes: [128]u8 = undefined,
    len: usize,

    pub fn init(raw: []const u8) Name {
        var n: Name = .{ .len = 1 };
        n.bytes[0] = '"';
        var at: usize = 0;
        while (at < raw.len) {
            const ch = raw[at];
            var escaped: [6]u8 = undefined;
            var take: usize = 1;
            const piece: []const u8 = blk: {
                if (ch == '"' or ch == '\\') {
                    escaped[0] = '\\';
                    escaped[1] = ch;
                    break :blk escaped[0..2];
                }
                const short: ?u8 = switch (ch) {
                    '\n' => 'n',
                    '\r' => 'r',
                    '\t' => 't',
                    '\x08' => 'b',
                    '\x0c' => 'f',
                    else => null,
                };
                if (short) |s| {
                    escaped[0] = '\\';
                    escaped[1] = s;
                    break :blk escaped[0..2];
                }
                if (ch >= 0x80) {
                    const width = std.unicode.utf8ByteSequenceLength(ch) catch 0;
                    if (width > 0 and at + width <= raw.len and std.unicode.utf8ValidateSlice(raw[at..][0..width])) {
                        take = width;
                        break :blk raw[at..][0..width];
                    }
                } else if (ch >= 0x20) break :blk raw[at..][0..1];
                const hex = "0123456789abcdef";
                escaped = .{ '\\', 'u', '0', '0', hex[ch >> 4], hex[ch & 15] };
                break :blk &escaped;
            };
            if (n.len + piece.len + 1 > n.bytes.len) break;
            @memcpy(n.bytes[n.len..][0..piece.len], piece);
            n.len += piece.len;
            at += take;
        }
        n.bytes[n.len] = '"';
        n.len += 1;
        return n;
    }

    pub fn slice(n: *const Name) []const u8 {
        return n.bytes[0..n.len];
    }
};

/// The final request's synchronous receiver; no borrowed name or arguments escape.
pub const Sink = struct {
    ctx: *anyopaque,
    record: *const fn (ctx: *anyopaque, name: Name) void,
    pub fn rejected(s: Sink, raw: []const u8) void {
        s.record(s.ctx, Name.init(raw));
    }
};
