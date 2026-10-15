//! grammar_tests CPU frame skeleton; not the Stage2 prepare/ready/commit wire protocol.
const std = @import("std");
pub const Kind = enum(u32) { begin = 1, first = 2, step = 3 };
pub const header_words = 10;
pub const Frame = struct {
    kind: Kind,
    epoch: u64,
    sequence: u64,
    position: usize,
    width: usize,
    tokens: []const u32,
    mask: []const u32,
    pub fn encode(f: Frame, a: std.mem.Allocator) ![]u32 {
        if (f.width == 0 or f.width > std.math.maxInt(i32) or f.position > std.math.maxInt(u32) or f.tokens.len > std.math.maxInt(u32) or f.mask.len > std.math.maxInt(u32)) return error.MaskFrame;
        if (f.mask.len != 0 and f.mask.len != (f.width + 31) / 32) return error.MaskFrame;
        const w = try a.alloc(u32, header_words + f.tokens.len + f.mask.len);
        w[0] = 1;
        w[1] = @intFromEnum(f.kind);
        w[2] = @truncate(f.epoch);
        w[3] = @truncate(f.epoch >> 32);
        w[4] = @truncate(f.sequence);
        w[5] = @truncate(f.sequence >> 32);
        w[6] = @intCast(f.position);
        w[7] = @intCast(f.width);
        w[8] = @intCast(f.tokens.len);
        w[9] = @intCast(f.mask.len);
        @memcpy(w[header_words..][0..f.tokens.len], f.tokens);
        @memcpy(w[header_words + f.tokens.len ..], f.mask);
        return w;
    }
    pub fn decode(w: []const u32) !Frame {
        if (w.len < header_words or w[0] != 1 or w[7] == 0 or w[7] > std.math.maxInt(i32)) return error.MaskFrame;
        const n: usize = w[8];
        const m: usize = w[9];
        if (n > w.len - header_words or m != w.len - header_words - n) return error.MaskFrame;
        if (m != 0 and m != (@as(usize, w[7]) + 31) / 32) return error.MaskFrame;
        return .{ .kind = std.enums.fromInt(Kind, w[1]) orelse return error.MaskFrame, .epoch = @as(u64, w[2]) | (@as(u64, w[3]) << 32), .sequence = @as(u64, w[4]) | (@as(u64, w[5]) << 32), .position = w[6], .width = w[7], .tokens = w[header_words..][0..n], .mask = w[header_words + n ..] };
    }
};

pub fn replay(comptime Steps: type, steps: *Steps, words: []const u32) !void { _ = try select(Steps, steps, words); }
pub fn select(comptime Steps: type, steps: *Steps, words: []const u32) !?u32 {
    if (comptime !@hasDecl(Steps, "checkMasked")) return error.IncompleteMaskedAdapter;
    try steps.checkMasked();
    const f = try Frame.decode(words);
    if (steps.maskedWidth() != f.width) return error.MaskFrame;
    switch (f.kind) {
        .begin => {
            if (f.tokens.len == 0 or f.position != 0 or f.mask.len != 0) return error.MaskFrame;
            try steps.prefillToLogits(f.tokens);
            return null;
        },
        .first => {
            if (f.tokens.len != 0 or f.position != steps.len) return error.MaskFrame;
            return try steps.pickFirst(f.mask);
        },
        .step => {
            if (f.tokens.len != 1 or f.position != steps.len + 1) return error.MaskFrame;
            return try steps.stepMasked(f.tokens[0], f.mask);
        },
    }
}
