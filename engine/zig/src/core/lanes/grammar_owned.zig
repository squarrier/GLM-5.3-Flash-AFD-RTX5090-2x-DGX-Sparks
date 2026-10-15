//! grammar_mask: request-owned Rules over a real grammar library's matcher, shaped to grammar_abi's tfg_owned.Matcher over
//! the release libtfgrammar ABI (evidence/grammar_abi/ABI-CONTRACT.md: Ash Hart's v1.0.6 c080a18e tf_grammar.cc over
//! xgrammar v0.2.8). The request owns the matcher: fill/accept/rollback/terminated reach it unchanged (no prompt is
//! ever fed, the activation marker never reaches it), and the Stream's Rules.close frees it exactly once. A matcher
//! error poisons the Rules, so the request ends; nothing falls back to a plain draw. An all-zero fill is passed on
//! as is: the masked draw refuses it (EmptyMask) before any launch. No library is loaded or called by itself here
//! (grammar_abi's trusted loader and admission own that), and drafts are never asked for (one row, no drafts).
const std = @import("std");
const Rules = @import("grammar.zig").Rules;

/// `M`: grammar_abi's Matcher shape: accept(*M, u32) !bool, rollback(*M, usize) !void, fill(*M, []u32) !void,
/// terminated(*M) !bool, free(*M) !void, and `c.vocab`, the vocabulary its compiler was opened with.
pub fn Owned(comptime M: type) type {
    return struct {
        const Self = @This();
        matcher: M,
        rules: Rules,
        freed: bool = false,
        free_error: ?anyerror = null,
        /// grammar_server: a served request's slot (`create`): Rules.close frees the matcher, then the slot.
        heap: ?std.mem.Allocator = null,

        /// Takes `matcher` (freed here on refusal): its vocabulary must be the model's logits width. `marker`: the
        /// generated token whose pick activates the rules for the NEXT one (GLM's </think>), or null (active at once).
        /// The returned Rules live in `x`, which must outlive the request's Stream.
        pub fn bind(x: *Self, matcher: M, width: usize, marker: ?u32) !*Rules {
            x.* = .{ .matcher = matcher, .rules = .{ .ptr = x, .width = width, .vtable = &vtable, .after = marker, .active = marker == null } };
            if (width == 0 or @as(usize, matcher.c.vocab) != width or (if (marker) |t| t >= width else false)) {
                x.rules.closed = true;
                x.release();
                return error.WidthMismatch;
            }
            return &x.rules;
        }

        const vtable: Rules.VTable = .{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close };

        fn of(p: *anyopaque) *Self {
            return @ptrCast(@alignCast(p));
        }
        fn fill(p: *anyopaque, words: []u32) anyerror!void {
            return of(p).matcher.fill(words);
        }
        fn accept(p: *anyopaque, token: u32) anyerror!bool {
            return of(p).matcher.accept(token);
        }
        fn rollback(p: *anyopaque, n: usize) anyerror!void {
            return of(p).matcher.rollback(n);
        }
        fn terminated(p: *anyopaque) bool {
            const x = of(p);
            return x.matcher.terminated() catch {
                x.rules.poisoned = true; // the next fill refuses (PoisonedRules): the request ends
                return false;
            };
        }
        fn close(p: *anyopaque) void {
            const x = of(p);
            x.release();
            if (x.heap) |gpa| gpa.destroy(x); // Rules.close touches nothing after this call
        }
        fn release(x: *Self) void {
            if (x.freed) return;
            x.freed = true;
            x.matcher.free() catch |err| {
                x.free_error = err;
            };
        }
    };
}

/// One request's Rules from `compiler` (grammar_abi's tfg_owned.Compiler): compile `kind`/`text`, a fresh matcher, the
/// compiled grammar freed right after (the release binding's order: the matcher holds its own reference), then bound.
pub fn request(comptime M: type, x: *Owned(M), compiler: anytype, kind: u32, text: []const u8, err: *[512]u8, width: usize, marker: ?u32) !*Rules {
    var g = try compiler.compile(kind, text, err);
    var m = g.matcher() catch |e| {
        g.free() catch {};
        return e;
    };
    g.free() catch |e| {
        m.free() catch {};
        return e;
    };
    return x.bind(m, width, marker);
}

/// grammar_server: `request` into a slot from `gpa` that the returned Rules' close frees with its matcher (a served request,
/// whose Stream owns the Rules); a refusal frees both before it returns.
pub fn create(comptime M: type, gpa: std.mem.Allocator, compiler: anytype, kind: u32, text: []const u8, err: *[512]u8, width: usize, marker: ?u32) !*Rules {
    const x = try gpa.create(Owned(M));
    const r = request(M, x, compiler, kind, text, err, width, marker) catch |e| {
        gpa.destroy(x);
        return e;
    };
    x.heap = gpa;
    return r;
}
