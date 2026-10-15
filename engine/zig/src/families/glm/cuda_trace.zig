//! The forward's launch trace (qualification_qualification_7c75d777 PLAN §4): every Triton launch, extension call and collective in python's order, so a
//! CPU test compares the Zig forward with python-0.6's (tools/zig/glm_forward_fixtures.py) launch by launch. A `Space`
//! names device addresses back to python's buffer paths and byte offsets.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("cuda_triton.zig");
const views = @import("cuda_views.zig");
const aot = cuda.aot;

/// A tensor as an extension call receives it: address, dtype, shape and strides (elements).
pub const T = struct {
    ptr: u64,
    dtype: views.DType,
    dims: u8,
    shape: [5]usize = @splat(1),
    stride: [5]usize = @splat(1),

    /// A contiguous view of `b` with `shape`, `at` bytes into it.
    pub fn of(b: views.Buf, at: usize, shape: []const usize) T {
        var t: T = .{ .ptr = b.ptr + at, .dtype = b.dtype, .dims = @intCast(shape.len) };
        @memcpy(t.shape[0..shape.len], shape);
        var s: usize = 1;
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            t.stride[i] = s;
            s *= shape[i];
        }
        return t;
    }
    /// `b` whole, as python passes a buffer
    pub fn whole(b: views.Buf) T {
        return of(b, 0, b.shape[0..b.dims]);
    }
    /// Rows [0, rows) of a 2-D view whose rows are `stride` elements apart, `cols` wide, `at` bytes into `b`.
    pub fn rows2(b: views.Buf, at: usize, rows: usize, cols: usize, stride: usize) T {
        return .{ .ptr = b.ptr + at, .dtype = b.dtype, .dims = 2, .shape = .{ rows, cols, 1, 1, 1 }, .stride = .{ stride, 1, 1, 1, 1 } };
    }
};

/// An extension call's argument, positional as python passes it.
pub const Val = union(enum) {
    t: T,
    list: []const Val,
    int: i64,
    f32: u32,
    bool: bool,
    none,

    pub fn float(x: f32) Val {
        return .{ .f32 = @bitCast(x) };
    }
};

pub const Op = enum { triton, ext, comm };

pub const Event = struct {
    op: Op,
    name: []const u8,
    grid: [3]u32 = .{ 0, 0, 0 },
    args: []const aot.Arg = &.{},
    consts: []const aot.Const = &.{},
    vals: []const Val = &.{},
};

/// A CPU test's scripted model: what a recorded sample call answers (`logits` the rows sampled; `positions` a sampled
/// draw's absolute positions, null for a greedy one, which reads none; `probs` a draft's probabilities when asked).
pub const Picker = struct {
    ctx: *anyopaque,
    pick: *const fn (ctx: *anyopaque, logits: T, positions: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void,
};

/// The recording sink: Triton launches land in `tri_rec` (cuda_triton.Recorder) and move here, in order, before each
/// extension call or collective and at `flush`.
pub const Trace = struct {
    arena: std.heap.ArenaAllocator,
    events: std.ArrayList(Event) = .empty,
    tri_rec: tri.Recorder = .{},
    /// the answers of recorded sample calls (null: token 0, probability 0)
    picker: ?Picker = null,
    test_fault: if (@import("builtin").is_test) ?*Fault else void = if (@import("builtin").is_test) null else {},

    pub const Fault = struct { name: []const u8, skip: usize = 0, hits: usize = 0, cutoff: usize = 0 };

    /// A recorded sample call's answer: the picker's, else zeros (no token was sampled; never the caller's bytes).
    pub fn picks(t: *Trace, logits: T, positions: ?[]const u64, out: []u32, probs: ?[]f64) !void {
        if (t.picker) |p| return p.pick(p.ctx, logits, positions, out, probs);
        @memset(out, 0);
        if (probs) |x| @memset(x, 0);
    }

    pub fn init(gpa: std.mem.Allocator) Trace {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    }
    pub fn deinit(t: *Trace) void {
        t.arena.deinit();
        t.* = undefined;
    }
    pub fn reset(t: *Trace) void {
        _ = t.arena.reset(.retain_capacity);
        t.events = .empty;
        t.tri_rec.n = 0;
    }

    /// The Triton launches recorded since the last call, appended in order.
    pub fn flush(t: *Trace) !void {
        const a = t.arena.allocator();
        for (t.tri_rec.launches[0..t.tri_rec.n]) |*l| {
            try t.events.append(a, .{ .op = .triton, .name = l.name, .grid = l.grid, .args = try a.dupe(aot.Arg, l.args[0..l.nargs]), .consts = try a.dupe(aot.Const, l.consts[0..l.nconsts]) });
        }
        t.tri_rec.n = 0;
    }

    pub fn call(t: *Trace, op: Op, name: []const u8, vals: []const Val) !void {
        try t.flush();
        try t.events.append(t.arena.allocator(), .{ .op = op, .name = name, .vals = try dupeVals(t.arena.allocator(), vals) });
        if (comptime @import("builtin").is_test) if (t.test_fault) |fault| {
            if (std.mem.eql(u8, name, fault.name)) {
                if (fault.skip > 0) { fault.skip -= 1; return; }
                fault.hits += 1; fault.cutoff = t.events.items.len;
                return error.RecordedPartialFault;
            }
        };
    }
};

fn dupeVals(a: std.mem.Allocator, vals: []const Val) ![]const Val {
    const out = try a.alloc(Val, vals.len);
    for (vals, out) |v, *o| o.* = switch (v) {
        .list => |l| .{ .list = try dupeVals(a, l) },
        else => v,
    };
    return out;
}

/// Python's buffer paths at device addresses: `name(addr)` gives (path, byte offset).
pub const Space = struct {
    gpa: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .empty,
    next: u64 = 1 << 36,

    pub const Entry = struct { path: []u8, base: u64, len: u64, buf: views.Buf };

    pub fn deinit(s: *Space) void {
        for (s.entries.items) |e| s.gpa.free(e.path);
        s.entries.deinit(s.gpa);
        s.* = undefined;
    }

    /// Records `b` under `path` at its address.
    pub fn add(s: *Space, path: []const u8, b: views.Buf) !void {
        try s.entries.append(s.gpa, .{ .path = try s.gpa.dupe(u8, path), .base = b.ptr, .len = b.bytes(), .buf = b });
    }

    /// Stand-in addresses for a CPU test: each buffer at its own 2^36-aligned base (no real memory behind it).
    pub fn place(s: *Space, path: []const u8, b: *views.Buf) !void {
        b.ptr = s.next;
        s.next += 1 << 36;
        try s.add(path, b.*);
    }

    pub fn name(s: *const Space, addr: u64) ?struct { path: []const u8, off: u64 } {
        for (s.entries.items) |e| if (addr >= e.base and addr < e.base + @max(e.len, 1)) return .{ .path = e.path, .off = addr - e.base };
        return null;
    }
};

/// views.walk's visitor that places every buffer in a Space (a CPU test's stand-in addresses).
pub fn placeAll(s: *Space, path: []const u8, b: *views.Buf) anyerror!void {
    try s.place(path, b);
}

test "a Space names addresses by path and offset" {
    var s: Space = .{ .gpa = std.testing.allocator };
    defer s.deinit();
    var a = views.Buf.of(.f32, &.{ 8, 4096 });
    var b = views.Buf.of(.bf16, &.{16});
    try s.place("b.part", &a);
    try s.place("b.x", &b);
    const n = s.name(a.ptr + 4096).?;
    try std.testing.expectEqualStrings("b.part", n.path);
    try std.testing.expectEqual(@as(u64, 4096), n.off);
    try std.testing.expect(s.name(b.ptr + 32) == null);
    const t = T.rows2(a, 16, 4, 128, 4096);
    try std.testing.expectEqual([5]usize{ 4096, 1, 1, 1, 1 }, t.stride);
    try std.testing.expectEqual([5]usize{ 4, 1, 1, 1, 1 }, T.of(b, 0, &.{ 4, 4 }).stride);
}
