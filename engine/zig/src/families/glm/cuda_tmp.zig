//! python's temporaries (qualification_qualification_7c75d777 PLAN §4): the tensors torch ops make on the way (`torch.empty`, `.clone()`, `torch.cat`,
//! the sampler's), each a views.Buf its caller keeps. A CPU trace names one when python would make it: `tmp.<op>`,
//! then `tmp.<op>.1`, ... counted by op (tools/zig/glm_step_fixtures.py `_register`), at a fresh stand-in address. On a
//! GPU the buffer is the caller's own device memory and naming it does nothing.
const std = @import("std");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");

pub const Tmp = struct {
    /// a CPU trace's Space (null on a GPU)
    space: ?*trace.Space = null,
    ops: [16][]const u8 = undefined,
    counts: [16]usize = undefined,
    n: usize = 0,

    /// python's counters back to zero (the fixtures name each case's temporaries afresh)
    pub fn reset(t: *Tmp) void {
        t.n = 0;
    }

    fn next(t: *Tmp, op: []const u8) !usize {
        for (t.ops[0..t.n], t.counts[0..t.n]) |o, *c| if (std.mem.eql(u8, o, op)) {
            c.* += 1;
            return c.* - 1;
        };
        if (t.n == t.ops.len) return error.TooManyOps;
        t.ops[t.n] = op;
        t.counts[t.n] = 1;
        t.n += 1;
        return 0;
    }

    /// `b` is the tensor `op` made: on a CPU, placed under python's name.
    pub fn made(t: *Tmp, op: []const u8, b: *views.Buf) !void {
        return t.madeFields(op, &.{""}, &.{b});
    }

    /// The tensors one op made, a field each (torch.topk's values and indices).
    pub fn madeFields(t: *Tmp, op: []const u8, fields: []const []const u8, bs: []const *views.Buf) !void {
        const s = t.space orelse return;
        const n = try t.next(op);
        for (fields, bs) |field, b| {
            var name: [96]u8 = undefined;
            var len = (try std.fmt.bufPrint(&name, "tmp.{s}", .{op})).len;
            if (n > 0) len += (try std.fmt.bufPrint(name[len..], ".{d}", .{n})).len;
            if (field.len > 0) len += (try std.fmt.bufPrint(name[len..], ".{s}", .{field})).len;
            try s.place(name[0..len], b);
        }
    }
};

test "temporaries take python's names, counted by op" {
    var s: trace.Space = .{ .gpa = std.testing.allocator };
    defer s.deinit();
    var t: Tmp = .{ .space = &s };
    var a = views.Buf.of(.f32, &.{4});
    var b = views.Buf.of(.f32, &.{4});
    var c = views.Buf.of(.i64, &.{4});
    try t.made("empty", &a);
    try t.made("empty", &b);
    try t.madeFields("topk", &.{ "values", "indices" }, &.{ &a, &c });
    try std.testing.expectEqualStrings("tmp.empty", s.name(s.entries.items[0].base).?.path);
    try std.testing.expectEqualStrings("tmp.empty.1", s.name(b.ptr).?.path);
    try std.testing.expectEqualStrings("tmp.topk.values", s.name(a.ptr).?.path);
    try std.testing.expectEqualStrings("tmp.topk.indices", s.name(c.ptr).?.path);
    t.reset();
    try t.made("empty", &a);
    try std.testing.expectEqualStrings("tmp.empty", s.name(a.ptr).?.path);
    var gpu: Tmp = .{};
    b.ptr = 4096;
    try gpu.made("clone", &b);
    try std.testing.expectEqual(@as(u64, 4096), b.ptr);
}
