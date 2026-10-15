//! tf-glm-forward --dry's launch list as JSON lines (TF_GLM_DRY_EVENTS=FILE; qualification_qualification_44ba554e.5 B7): every recorded event in order,
//! its op, kernel or call, grid, constexprs and arguments, each address named by python's buffer path and offset (the
//! Engine's and its temporaries' in the run's Space, the weights' by their view paths), in the fixtures' form
//! (tools/zig/glm_forward_fixtures.py), so a script can hold a chunk's launches to qualification_qualification_ea348a94's G-trace. Off by default.
const std = @import("std");
const glm = @import("glm_cuda");

const Space = glm.trace.Space;
const Event = glm.trace.Event;
const Val = glm.trace.Val;

/// The weights' paths at their bound addresses (nothing moves).
fn name(s: *Space, path: []const u8, b: *glm.views.Buf) anyerror!void {
    if (b.ptr != 0 and !b.host) try s.add(path, b.*);
}

/// Writes `events` to `path`: one JSON object a line.
pub fn write(gpa: std.mem.Allocator, path: []const u8, events: []const Event, space: *const Space, w: *glm.views.Weights) !void {
    var ws: Space = .{ .gpa = gpa };
    defer ws.deinit();
    try glm.views.walk(w, glm.views.Path.init("w"), &ws, name);
    const n: Names = .{ .a = space, .b = &ws };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    for (events) |e| {
        try out.print(gpa, "{{\"op\": \"{t}\", \"fn\": \"{s}\"", .{ e.op, e.name });
        if (e.op == .triton) {
            try out.print(gpa, ", \"grid\": [{d}, {d}, {d}], \"args\": [", .{ e.grid[0], e.grid[1], e.grid[2] });
            for (e.args, 0..) |a, i| {
                try out.print(gpa, "{s}{{\"name\": \"{s}\", ", .{ if (i == 0) "" else ", ", a.name });
                switch (a.value) {
                    .ptr => |p| {
                        try out.print(gpa, "\"type\": \"{s}\", ", .{p.ty});
                        try n.at(gpa, &out, p.addr);
                    },
                    .i32 => |x| try out.print(gpa, "\"int\": {d}", .{x}),
                    .f32 => |x| try out.print(gpa, "\"f32\": {d}", .{@as(u32, @bitCast(x))}),
                    .u64 => |x| try out.print(gpa, "\"u64\": {d}", .{x}),
                }
                try out.append(gpa, '}');
            }
            try out.appendSlice(gpa, "], \"consts\": {");
            for (e.consts, 0..) |c, i| {
                try out.print(gpa, "{s}\"{s}\": ", .{ if (i == 0) "" else ", ", c.name });
                if (c.int) |x| try out.print(gpa, "{{\"int\": {d}}}", .{x}) else if (c.f32) |x| try out.print(gpa, "{{\"f32\": {d}}}", .{@as(u32, @bitCast(x))}) else try out.appendSlice(gpa, "null");
            }
            try out.appendSlice(gpa, "}}\n");
            continue;
        }
        try out.appendSlice(gpa, ", \"args\": ");
        try vals(gpa, &out, n, e.vals);
        try out.appendSlice(gpa, "}\n");
    }
    var buf: [4096]u8 = undefined;
    if (path.len + 1 > buf.len) return error.PathTooLong;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const fd = std.c.open(buf[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    if (std.c.write(fd, out.items.ptr, out.items.len) != @as(isize, @intCast(out.items.len))) return error.CannotWrite;
}

/// An address by path and byte offset: the run's Space first, then the weights; "?" with the address if neither.
const Names = struct {
    a: *const Space,
    b: *const Space,

    fn at(n: Names, gpa: std.mem.Allocator, out: *std.ArrayList(u8), addr: u64) !void {
        const got = n.a.name(addr) orelse n.b.name(addr) orelse return out.print(gpa, "\"buf\": \"?\", \"addr\": {d}", .{addr});
        try out.print(gpa, "\"buf\": \"{s}\", \"off\": {d}", .{ got.path, got.off });
    }
};

fn vals(gpa: std.mem.Allocator, out: *std.ArrayList(u8), n: Names, xs: []const Val) !void {
    try out.append(gpa, '[');
    for (xs, 0..) |v, i| {
        if (i > 0) try out.appendSlice(gpa, ", ");
        switch (v) {
            .t => |t| {
                try out.appendSlice(gpa, "{\"t\": {");
                try n.at(gpa, out, t.ptr);
                try out.print(gpa, ", \"dtype\": \"{t}\", \"shape\": [", .{t.dtype});
                for (t.shape[0..t.dims], 0..) |s, j| try out.print(gpa, "{s}{d}", .{ if (j == 0) "" else ", ", s });
                try out.appendSlice(gpa, "], \"stride\": [");
                for (t.stride[0..t.dims], 0..) |s, j| try out.print(gpa, "{s}{d}", .{ if (j == 0) "" else ", ", s });
                try out.appendSlice(gpa, "]}}");
            },
            .list => |l| {
                try out.appendSlice(gpa, "{\"list\": ");
                try vals(gpa, out, n, l);
                try out.append(gpa, '}');
            },
            .int => |x| try out.print(gpa, "{{\"int\": {d}}}", .{x}),
            .f32 => |x| try out.print(gpa, "{{\"f32\": {d}}}", .{x}),
            .bool => |x| try out.print(gpa, "{{\"bool\": {}}}", .{x}),
            .none => try out.appendSlice(gpa, "{\"none\": null}"),
        }
    }
    try out.append(gpa, ']');
}
