//! `tf-glm-forward check-weights DIR --dry --trace FILE` (G-pack-x's trace half, qualification_qualification_ac5a9c94): the rank's pack pass recorded
//! over stand-in addresses (stored group g at (g + 1) << 40; packed buffers where check-weights placed them; EXL3's
//! in-place buffers at their stored bytes), every item's `glm_pack.*` events in the plan's order, and the split's copies,
//! as JSON. evidence/qualification_qualification_ac5a9c94/scripts/trace_cmp.py names each source by checkpoint tensor and compares python-0.6's load.
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const pack = glm.pack;

fn never(_: pack.Kern, _: [3]u32, _: [3]u32, _: *cuda.Args) anyerror!void {
    return error.Unrecorded;
}

fn val(out: *std.ArrayList(u8), gpa: std.mem.Allocator, v: glm.trace.Val) !void {
    switch (v) {
        .int => |x| try out.print(gpa, "{d}", .{x}),
        .list => |l| {
            try out.append(gpa, '[');
            for (l, 0..) |x, i| {
                if (i > 0) try out.appendSlice(gpa, ", ");
                try val(out, gpa, x);
            }
            try out.append(gpa, ']');
        },
        else => try out.appendSlice(gpa, "null"),
    }
}

/// Writes the trace of rank `r` (its packed buffers already placed) to `path`; returns the number of events.
pub fn write(gpa: std.mem.Allocator, r: anytype, shards: []const [:0]const u8, path: []const u8) !usize {
    const groups = r.split.groups();
    const stored = try gpa.alloc(u64, groups.len);
    defer gpa.free(stored);
    for (stored, 0..) |*s, g| s.* = (@as(u64, g) + 1) << 40;
    for (0..groups.len) |g| r.pk.placeKept(@intCast(g), stored[g]);
    var tr = glm.trace.Trace.init(gpa);
    defer tr.deinit();
    var f = glm.fwd.Fwd.recording(&r.m, &tr);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try out.appendSlice(gpa, "{\"schema\": \"glm-pack-trace/1\", \"stored\": \"group g at (g + 1) << 40\", \"shards\": [");
    for (shards, 0..) |s, i| try out.print(gpa, "{s}\"{s}\"", .{ if (i == 0) "" else ", ", s });
    try out.appendSlice(gpa, "],\n\"groups\": [");
    for (groups, 0..) |b, i| try out.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", b });
    try out.appendSlice(gpa, "],\n\"copies\": [");
    for (r.split.copies(), 0..) |c, i| try out.print(gpa, "{s}\n[{d}, {d}, {d}, {d}, {d}, {d}, {d}, {d}]", .{ if (i == 0) "" else ",", c.shard, c.off, c.len, c.g, c.dst, c.row, c.at, c.width });
    try out.appendSlice(gpa, "],\n\"items\": [");
    var events: usize = 0;
    for (r.pk.items.items, 0..) |*it, i| {
        const first = tr.events.items.len;
        try pack.make(&f, .{ .host = never }, it, stored);
        try f.flush();
        try out.print(gpa, "{s}\n{{\"path\": \"{s}\", \"dtype\": \"{s}\", \"shape\": [", .{ if (i == 0) "" else ",", it.path, pack.dtypeName(it.buf.dtype) });
        for (it.buf.shape[0..it.buf.dims], 0..) |s, j| try out.print(gpa, "{s}{d}", .{ if (j == 0) "" else ", ", s });
        try out.print(gpa, "], \"bytes\": {d}, \"ptr\": {d}, \"home\": {d}, \"op\": \"{s}\", \"x\": ", .{ it.buf.bytes(), it.buf.ptr, it.home, @tagName(it.op) });
        if (it.x) |x| try out.print(gpa, "\"{s}\"", .{@tagName(x)}) else try out.appendSlice(gpa, "null");
        if (if (it.x) |x| pack.x_ops.inPlace(x) else null) |k| try out.print(gpa, ", \"kept\": [{d}, {d}]", .{ k.g, k.off });
        try out.appendSlice(gpa, ", \"events\": [");
        for (tr.events.items[first..], 0..) |e, j| {
            try out.print(gpa, "{s}[\"{s}\"", .{ if (j == 0) "" else ", ", e.name });
            for (e.vals) |v| {
                try out.appendSlice(gpa, ", ");
                try val(&out, gpa, v);
            }
            try out.append(gpa, ']');
        }
        try out.appendSlice(gpa, "]}");
        events += tr.events.items.len - first;
    }
    try out.appendSlice(gpa, "]}\n");
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    const fd = std.c.open(z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    if (std.c.write(fd, out.items.ptr, out.items.len) != @as(isize, @intCast(out.items.len))) return error.CannotWrite;
    return events;
}
