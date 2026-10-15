//! grammar_tests minimal request-owned Rules contract; no library, compiler, tokenizer or ABI.
const std = @import("std");

pub const Rules = struct {
    ptr: *anyopaque,
    width: usize,
    after: ?u32 = null,
    active: bool = true,
    poisoned: bool = false,
    closed: bool = false,
    vtable: *const VTable,
    pub const VTable = struct {
        fill: *const fn (*anyopaque, []u32) anyerror!void,
        accept: *const fn (*anyopaque, u32) anyerror!bool,
        rollback: *const fn (*anyopaque, usize) anyerror!void,
        terminated: *const fn (*anyopaque) bool,
        close: *const fn (*anyopaque) void,
    };
    pub fn next(r: *Rules, a: std.mem.Allocator) ![]const u32 {
        if (r.closed or r.poisoned) return error.PoisonedRules;
        if (!r.active) return &.{};
        if (r.width == 0 or r.width > std.math.maxInt(i32)) return error.MaskShape;
        const mask = try a.alloc(u32, (r.width + 31) / 32);
        errdefer a.free(mask);
        try r.vtable.fill(r.ptr, mask);
        return mask;
    }
    pub fn follow(r: *Rules, token: u32) !bool {
        if (r.closed or r.poisoned) return error.PoisonedRules;
        if (!r.active) {
            if (r.after == token) r.active = true;
            return true;
        }
        if (token >= r.width) {
            r.poisoned = true;
            return false;
        }
        const accepted = r.vtable.accept(r.ptr, token) catch |err| {
            r.poisoned = true;
            return err;
        };
        if (!accepted) r.poisoned = true;
        return accepted;
    }
    pub fn undo(r: *Rules, n: usize) !void {
        r.vtable.rollback(r.ptr, n) catch |err| {
            r.poisoned = true;
            return err;
        };
    }
    pub fn terminated(r: *Rules) bool {
        return r.active and !r.poisoned and r.vtable.terminated(r.ptr);
    }
    pub fn close(r: *Rules) void {
        if (r.closed) return;
        r.closed = true;
        r.vtable.close(r.ptr);
    }
};
