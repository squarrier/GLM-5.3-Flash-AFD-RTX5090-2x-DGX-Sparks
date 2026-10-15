//! Fake shared-window grammar traversal; NOT a GLM head/held verification adapter.
const std = @import("std");
const Rules = @import("grammar.zig").Rules;
pub const Plan = struct {
    rows: std.ArrayList(u32) = .empty,
    parents: std.ArrayList(i32) = .empty,
    positions: std.ArrayList(u64) = .empty,
    masks: std.ArrayList(u32) = .empty,
    pub fn deinit(p: *Plan, a: std.mem.Allocator) void {
        p.rows.deinit(a); p.parents.deinit(a); p.positions.deinit(a); p.masks.deinit(a);
    }
};
const Walk = struct {
    a: std.mem.Allocator,
    rules: *Rules,
    tokens: []const u32,
    parents: []const i32,
    positions: []const u64,
    out: *Plan,
    fn row(w: *Walk, original: usize, parent: i32) !void {
        if (w.rules.terminated()) return; // never verify a terminated descendant
        const mapped: i32 = @intCast(w.out.rows.items.len);
        const mask = try w.rules.next(w.a);
        defer w.a.free(mask);
        try w.out.rows.append(w.a, @intCast(original));
        try w.out.parents.append(w.a, parent);
        try w.out.positions.append(w.a, w.positions[original]);
        try w.out.masks.appendSlice(w.a, mask);
        for (w.tokens, 1..) |token, child| {
            if (w.parents[child] != original) continue;
            const accepted = w.rules.vtable.accept(w.rules.ptr, token) catch |err| {
                w.rules.poisoned = true;
                return err;
            };
            if (!accepted) continue; // rejected child's descendants are pruned with it
            w.row(child, mapped) catch |err| {
                try w.rules.undo(1); // successful child accept must unwind even on a recursive error
                return err;
            };
            try w.rules.undo(1);
        }
    }
};
pub fn build(a: std.mem.Allocator, rules: *Rules, tokens: []const u32, parents: []const i32, positions: []const u64) !Plan {
    if (parents.len != tokens.len + 1 or positions.len != parents.len or parents[0] != -1) return error.MaskShape;
    for (parents[1..], 1..) |parent, i| if (parent < 0 or parent >= i) return error.MaskShape;
    if (!rules.active or rules.closed or rules.poisoned) return error.PoisonedRules;
    var out = Plan{};
    errdefer out.deinit(a);
    var w = Walk{ .a = a, .rules = rules, .tokens = tokens, .parents = parents, .positions = positions, .out = &out };
    try w.row(0, -1);
    return out;
}
