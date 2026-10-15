//! GLM-5.3's prompts as python-0.6 renders them (families/glm5_next/prompts.py `GlmTokenizer`, ed78d6f): earlier turns
//! keep their reasoning (the template's `clear_thinking` passed false unless TF_GLM_CLEAR_THINKING=1), and with thinking
//! off the text has no effort line and a closed think block. Every other call goes to the checkpoint's own text.
const std = @import("std");
const json = @import("json.zig");
const model_text = @import("model_text.zig");
const Allocator = std.mem.Allocator;

pub const model_type = "glm5_next";
const effort_line = "<|system|>Reasoning Effort: Max";
const opened = "<|assistant|><think>";

/// TF_GLM_CLEAR_THINKING as python reads it: unset, empty or "0" is false, "1" true, anything else refused.
pub fn clearThinking(value: ?[]const u8) error{Invalid}!bool {
    const v = std.mem.trim(u8, value orelse "0", " \t\r\n");
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.Invalid;
}

/// python's `thinking_off`: the first effort line goes, and a prompt that ends in an opened think block closes it.
pub fn thinkingOff(a: Allocator, text: []const u8) Allocator.Error![]u8 {
    const at = std.mem.indexOf(u8, text, effort_line);
    const cut = if (at) |i| try std.mem.concat(a, u8, &.{ text[0..i], text[i + effort_line.len ..] }) else try a.dupe(u8, text);
    if (!std.mem.endsWith(u8, cut, opened)) return cut;
    defer a.free(cut);
    return std.mem.concat(a, u8, &.{ cut, "</think>" });
}

pub const GlmText = struct {
    inner: model_text.Text,
    clear: bool,

    pub fn text(g: *GlmText) model_text.Text {
        return .{ .ctx = g, .vtable = &vtable };
    }

    const vtable: model_text.Text.VTable = .{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .special = special };

    fn wrapped(ctx: *anyopaque) model_text.Text {
        const g: *GlmText = @ptrCast(@alignCast(ctx));
        return g.inner;
    }

    fn render(ctx: *anyopaque, a: Allocator, messages: json.Value, options: model_text.RenderOptions, problem: *[]const u8) model_text.Error![]u8 {
        const g: *GlmText = @ptrCast(@alignCast(ctx));
        var o = options;
        o.clear_thinking = o.clear_thinking orelse g.clear;
        const out = try g.inner.vtable.render(g.inner.ctx, a, messages, o, problem);
        return if (options.enable_thinking) out else thinkingOff(a, out);
    }

    fn encode(ctx: *anyopaque, a: Allocator, t: []const u8, add_special: bool) model_text.Error![]u32 {
        const i = wrapped(ctx);
        return i.vtable.encode(i.ctx, a, t, add_special);
    }

    fn decode(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const i = wrapped(ctx);
        return i.vtable.decode(i.ctx, a, ids);
    }

    fn tokenId(ctx: *anyopaque, piece: []const u8) ?u32 {
        const i = wrapped(ctx);
        return i.vtable.token_id(i.ctx, piece);
    }

    fn tokenString(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        const i = wrapped(ctx);
        return i.vtable.token_string(i.ctx, a, id);
    }

    fn vocabSize(ctx: *anyopaque) u32 {
        const i = wrapped(ctx);
        return i.vtable.vocab_size(i.ctx);
    }

    fn eosIds(ctx: *anyopaque) []const u32 {
        const i = wrapped(ctx);
        return i.vtable.eos_ids(i.ctx);
    }

    fn templateSource(ctx: *anyopaque) []const u8 {
        const i = wrapped(ctx);
        return i.vtable.template_source(i.ctx);
    }

    fn special(ctx: *anyopaque, id: u32) bool {
        const i = wrapped(ctx);
        return if (i.vtable.special) |f| f(i.ctx, id) else false;
    }
};

test "thinking off drops the first effort line and closes an opened think block, as python's thinking_off" {
    const a = std.testing.allocator;
    const got = try thinkingOff(a, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>");
    defer a.free(got);
    try std.testing.expectEqualStrings("[gMASK]<sop><|user|>hi<|assistant|><think></think>", got);
    const history = try thinkingOff(a, "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>hi<|assistant|><think></think>ok");
    defer a.free(history);
    try std.testing.expectEqualStrings("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>hi<|assistant|><think></think>ok", history);
    try std.testing.expectEqual(false, try clearThinking(null));
    try std.testing.expectEqual(false, try clearThinking(""));
    try std.testing.expectEqual(true, try clearThinking("1"));
    try std.testing.expectError(error.Invalid, clearThinking("yes"));
}
