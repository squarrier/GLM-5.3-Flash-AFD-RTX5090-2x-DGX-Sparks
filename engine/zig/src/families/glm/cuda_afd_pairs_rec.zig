//! qualification_0f991257's record of a paired prefill on a CPU, for its gates (G-trace-pairs, the interleave): each recorded launch given
//! to its chunk by the scheduler's hook (`Log`), a solo prefill's launches cut at its commits (`soloChunks`), and a
//! chunk's launches digested by buffer name, the pair's copies named as the buffers they stand for (`digest`).
const std = @import("std");
const trace = @import("cuda_trace.zig");
const pairs = @import("cuda_afd_pairs.zig");

/// The pair's copies and the buffers they stand for in a solo chunk's launches (qualification_c533375f: B's shared-expert rows with two
/// requests in flight; qualification_47b9bf3a: a lane's MTP residual and MTP position).
pub const renames = [_][2][]const u8{
    .{ "pair.x", "pbuf.x" },             .{ "pair.post", "pbuf.post" },     .{ "pair.comb", "pbuf.comb" },
    .{ "pair.fnormed", "pbuf.fnormed" }, .{ "pair.pos_dev", "st.pos_dev" }, .{ "pair.hdr", "afd.hdr" },
    .{ "pair.sy", "pbuf.sy" },           .{ "pair.mx", "pbuf.mx" },         .{ "pair.mtp_pos_dev", "st.mtp_pos_dev" },
};

pub const Name = struct { path: []const u8, off: u64 };

/// The buffer at `addr` by name, a copy by its original's (qualification_9e49592e: a lane's, `lane1.x` …, as the pair's).
pub fn named(space: *const trace.Space, at: u64) ?Name {
    const n = space.name(at) orelse return null;
    for (renames) |r| if (std.mem.eql(u8, n.path, r[0])) return .{ .path = r[1], .off = n.off };
    if (n.path.len > 6 and std.mem.startsWith(u8, n.path, "lane") and std.ascii.isDigit(n.path[4]) and n.path[5] == '.') {
        for (renames) |r| if (std.mem.eql(u8, n.path[6..], r[0]["pair.".len..])) return .{ .path = r[1], .off = n.off };
    }
    return .{ .path = n.path, .off = n.off };
}

const Sha = std.crypto.hash.sha2.Sha256;

fn put(h: *Sha, comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    h.update(std.fmt.bufPrint(&buf, fmt, args) catch unreachable);
}

fn addr(h: *Sha, space: *const trace.Space, x: u64) void {
    if (named(space, x)) |p| put(h, "@{s}+{d}", .{ p.path, p.off }) else put(h, "#{d}", .{x});
}

fn val(h: *Sha, space: *const trace.Space, v: trace.Val) void {
    switch (v) {
        .t => |t| {
            addr(h, space, t.ptr);
            put(h, " {t} {d} {any} {any};", .{ t.dtype, t.dims, t.shape, t.stride });
        },
        .list => |l| {
            put(h, "[{d}", .{l.len});
            for (l) |u| val(h, space, u);
            h.update("]");
        },
        .int => |x| put(h, "i{d};", .{x}),
        .f32 => |x| put(h, "f{d};", .{x}),
        .bool => |x| put(h, "b{};", .{x}),
        .none => h.update("none;"),
    }
}

/// One launch into `h` (afd_serve_fail.zig's digest, qualification_4be15bbf: op, name, grid, every argument and value by buffer name).
pub fn event(h: *Sha, space: *const trace.Space, e: trace.Event) void {
    put(h, "{t} {s} {any}|", .{ e.op, e.name, e.grid });
    for (e.args) |a| {
        put(h, "{s}=", .{a.name});
        switch (a.value) {
            .ptr => |p| {
                put(h, "{s}", .{p.ty});
                addr(h, space, p.addr);
            },
            .i32 => |x| put(h, "i32 {d}", .{x}),
            .f32 => |x| put(h, "f32 {x}", .{@as(u32, @bitCast(x))}),
            .u64 => |x| put(h, "u64 {d}", .{x}),
        }
        h.update(",");
    }
    for (e.consts) |c| {
        put(h, "{s}:", .{c.name});
        if (c.int) |x| put(h, "{d}", .{x});
        if (c.f32) |x| put(h, "f{x}", .{@as(u32, @bitCast(x))});
        h.update(",");
    }
    for (e.vals) |v| val(h, space, v);
    h.update("\n");
}

/// The launches' digest (hex), in order.
pub fn digest(space: *const trace.Space, events: []const trace.Event) [64]u8 {
    var h = Sha.init(.{});
    for (events) |e| event(&h, space, e);
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.bytesToHex(d, .lower);
}

/// Whether `e` is forward.commit's set_pos (a torch.fill_ of st.pos_dev): where a solo prefill's chunk ends.
pub fn isCommit(space: *const trace.Space, e: trace.Event) bool {
    if (!std.mem.eql(u8, e.name, "torch.fill_") or e.vals.len == 0) return false;
    const t = switch (e.vals[0]) {
        .t => |t| t,
        else => return false,
    };
    const n = space.name(t.ptr) orelse return false;
    return std.mem.eql(u8, n.path, "st.pos_dev");
}

/// A solo prefill's chunks: from `from` (State.reset's end), each chunk's launches through its commit (whose last
/// launch is set_pos's fill), into `out` as [start, end) ranges; gives how many.
pub fn soloChunks(space: *const trace.Space, events: []const trace.Event, from: usize, out: [][2]usize) usize {
    var n: usize = 0;
    var start = from;
    for (events[from..], from..) |e, i| {
        if (!isCommit(space, e)) continue;
        if (n == out.len) return n;
        out[n] = .{ start, i + 1 };
        n += 1;
        start = i + 1;
    }
    return n;
}

/// The scheduler's hook as a test's sink: every step (who, what, position) in order, where its launches start, and
/// the chunk it belongs to (a pair's A and B, then a solo chunk, numbered in prompt order).
pub const Log = struct {
    t: *trace.Trace,
    marks: std.ArrayList(Mark) = .empty,
    gpa: std.mem.Allocator,
    pairs_seen: usize = 0,

    pub const Mark = struct { who: pairs.Who, step: pairs.Step, pos: usize, at: usize, chunk: ?usize };

    pub fn deinit(l: *Log) void {
        l.marks.deinit(l.gpa);
    }

    pub fn hook(l: *Log) pairs.Hook {
        return .{ .ctx = l, .at = at };
    }

    fn at(ctx: *anyopaque, who: pairs.Who, step: pairs.Step, pos: usize) anyerror!void {
        const l: *Log = @ptrCast(@alignCast(ctx));
        try l.t.flush();
        if (who == .a and step == .stage) l.pairs_seen += 1;
        const chunk: ?usize = switch (who) {
            .a => 2 * (l.pairs_seen - 1),
            .b => 2 * (l.pairs_seen - 1) + 1,
            .solo => 2 * l.pairs_seen,
            .pair => null,
        };
        try l.marks.append(l.gpa, .{ .who = who, .step = step, .pos = pos, .at = l.t.events.items.len, .chunk = chunk });
    }

    /// Chunk k's launches (every step's of it, in order), digested by name; and how many.
    pub fn chunkDigest(l: *const Log, space: *const trace.Space, k: usize) struct { hex: [64]u8, n: usize } {
        var h = Sha.init(.{});
        var n: usize = 0;
        const ev = l.t.events.items;
        for (l.marks.items, 0..) |m, i| {
            if (m.chunk != k) continue;
            const end = if (i + 1 < l.marks.items.len) l.marks.items[i + 1].at else ev.len;
            for (ev[m.at..end]) |e| event(&h, space, e);
            n += end - m.at;
        }
        var d: [32]u8 = undefined;
        h.final(&d);
        return .{ .hex = std.fmt.bytesToHex(d, .lower), .n = n };
    }

    /// Chunk k's launches, in order (shallow copies; the caller frees the slice).
    pub fn chunkEvents(l: *const Log, gpa: std.mem.Allocator, k: usize) ![]trace.Event {
        var out: std.ArrayList(trace.Event) = .empty;
        errdefer out.deinit(gpa);
        const ev = l.t.events.items;
        for (l.marks.items, 0..) |m, i| {
            if (m.chunk != k) continue;
            const end = if (i + 1 < l.marks.items.len) l.marks.items[i + 1].at else ev.len;
            try out.appendSlice(gpa, ev[m.at..end]);
        }
        return out.toOwnedSlice(gpa);
    }

    /// The launches of the pair's own steps (B's position), every pair's.
    pub fn own(l: *const Log) usize {
        var n: usize = 0;
        for (l.marks.items, 0..) |m, i| if (m.who == .pair and m.step == .position) {
            n += (if (i + 1 < l.marks.items.len) l.marks.items[i + 1].at else l.t.events.items.len) - m.at;
        };
        return n;
    }

    /// Mark i's launches: from its start to the next mark's.
    fn span(l: *const Log, i: usize) []const trace.Event {
        const ev = l.t.events.items;
        const m = l.marks.items;
        return ev[m[i].at..if (i + 1 < m.len) m[i + 1].at else ev.len];
    }

    /// G-cross (qualification_c533375f): what crosses a step is each window's own, read from the record. Over the windows' steps (stage
    /// to tail) no launch of A names one of B's copies and none of B's names A's buffer it stands for (`crossing`);
    /// and between a window's request and its take no launch of the other window names the window's shared-expert
    /// rows (`sy`: A's, B's), which its take still reads (with two in flight the other window's request runs there).
    /// Gives the first launch that does (its index in the record), or null.
    pub fn crossed(l: *const Log, space: *const trace.Space, sy: [2][]const u8) ?usize {
        const marks = l.marks.items;
        for (marks, 0..) |m, i| {
            if ((m.who != .a and m.who != .b) or m.step == .absorb or m.step == .commit) continue;
            const other: usize = if (m.who == .a) 1 else 0;
            for (l.span(i), m.at..) |e, k| for (crossing) |c| if (names(space, e, c[other])) return k;
        }
        for (marks, 0..) |m, i| {
            if (m.step != .request or (m.who != .a and m.who != .b)) continue;
            const mine: usize = if (m.who == .a) 0 else 1;
            var j = i + 1;
            while (j < marks.len and !(marks[j].who == m.who and marks[j].step == .take and marks[j].pos == m.pos)) : (j += 1) {}
            for (marks[i + 1 .. j], i + 1..) |o, oi| {
                if (o.who == m.who or (o.who != .a and o.who != .b)) continue;
                for (l.span(oi), o.at..) |e, k| if (names(space, e, sy[mine])) return k;
            }
        }
        return null;
    }
};

/// What crosses a step and is each window's own (cuda_afd_pairs.Copies): A's buffer, B's copy of it (qualification_47b9bf3a: a lane's MTP
/// residual and MTP position too; a pair has neither copy).
pub const crossing = [_][2][]const u8{
    .{ "pbuf.x", "pair.x" },   .{ "pbuf.post", "pair.post" },             .{ "pbuf.comb", "pair.comb" }, .{ "pbuf.fnormed", "pair.fnormed" }, .{ "st.pos_dev", "pair.pos_dev" },
    .{ "pbuf.mx", "pair.mx" }, .{ "st.mtp_pos_dev", "pair.mtp_pos_dev" },
};

fn isAt(space: *const trace.Space, x: u64, path: []const u8) bool {
    const n = space.name(x) orelse return false;
    return std.mem.eql(u8, n.path, path);
}

fn valIs(space: *const trace.Space, v: trace.Val, path: []const u8) bool {
    return switch (v) {
        .t => |t| isAt(space, t.ptr, path),
        .list => |l| for (l) |u| {
            if (valIs(space, u, path)) break true;
        } else false,
        else => false,
    };
}

/// Whether launch `e` names buffer `path` (any pointer argument or tensor value inside it), by its own name.
pub fn names(space: *const trace.Space, e: trace.Event, path: []const u8) bool {
    for (e.args) |a| switch (a.value) {
        .ptr => |p| if (isAt(space, p.addr, path)) return true,
        else => {},
    };
    for (e.vals) |v| if (valIs(space, v, path)) return true;
    return false;
}
