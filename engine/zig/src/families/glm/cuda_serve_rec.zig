//! Recording TP ranks for the serve path's CPU checks (qualification_qualification_44ba554e.8): the prompt fixture's model (fixtures_cuda_prompt.json) as
//! rank `rank` of its world, cut to its layers with no MTP head (drafts off), every buffer at a stand-in address; and the
//! launch-trace comparisons, addresses by buffer name (cuda_serve_test.zig; native/glm_cuda_pair_test.zig's two ranks).
const std = @import("std");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const eng = @import("cuda_engine.zig");

const testing = std.testing;
const J = std.json.Value;

/// The prompt fixture (python-0.6's decode.prefill on a TP2 rank, 3 layers): its config_json and Engine settings.
pub const fixture = @embedFile("fixtures_cuda_prompt.json");

fn get(v: J, key: []const u8) J {
    return v.object.get(key).?;
}

fn int(v: J) usize {
    return @intCast(v.integer);
}

/// One recording rank: its model, views, Engine, address space and trace (the parsed fixture lives as long).
pub const Rank = struct {
    fx: std.json.Parsed(J),
    m: md.Model,
    w: views.Weights,
    e: *eng.Engine,
    space: trace.Space,
    t: trace.Trace,

    /// Rank `rank` (null: the fixture's own) of the fixture's model; `capacity` null: the fixture's.
    pub fn init(gpa: std.mem.Allocator, rank: ?usize, capacity: ?usize) !*Rank {
        return initWith(gpa, rank, capacity, false);
    }

    /// `init`, with the model's MTP head kept when `mtp` (drafts on, qualification_qualification_44ba554e.8 day 3: the Engine holds its buffers).
    pub fn initWith(gpa: std.mem.Allocator, rank: ?usize, capacity: ?usize, mtp: bool) !*Rank {
        const r = try gpa.create(Rank);
        errdefer gpa.destroy(r);
        r.fx = try std.json.parseFromSlice(J, gpa, fixture, .{});
        errdefer r.fx.deinit();
        const fx = r.fx.value;
        r.m = try md.Model.parse(gpa, get(fx, "config_json").string, int(get(fx, "world")), rank orelse int(get(fx, "rank")));
        const mtp_layers = r.m.mtp_layers;
        try r.m.cut(int(get(fx, "layers"))); // no MTP head: the cut leaves none
        if (mtp) r.m.mtp_layers = mtp_layers;
        r.w = try views.rank(gpa, &r.m);
        errdefer r.w.deinit(gpa);
        r.e = try eng.Engine.create(gpa, &r.m, &r.w, capacity orelse int(get(fx, "capacity")), int(get(fx, "max_rows")), int(get(fx, "prefill_rows")), get(fx, "long_context").bool);
        errdefer r.e.destroy(gpa);
        r.space = .{ .gpa = gpa };
        try views.walk(&r.w, views.Path.init("w"), &r.space, trace.placeAll);
        try r.e.walk(&r.space, trace.placeAll);
        try r.e.walkTmp(&r.space, trace.placeAll);
        r.e.tmp.space = &r.space;
        r.t = trace.Trace.init(gpa);
        return r;
    }

    pub fn deinit(r: *Rank, gpa: std.mem.Allocator) void {
        r.t.deinit();
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.space.deinit();
        r.fx.deinit();
        gpa.destroy(r);
    }

    /// A Fwd that records this rank's launches.
    pub fn recording(r: *Rank) fwd.Fwd {
        return fwd.Fwd.recording(&r.m, &r.t);
    }

    /// The trace and the Engine's temporaries emptied, as a fresh rank's (a second check on the same rank).
    pub fn clear(r: *Rank) void {
        r.t.reset();
        r.e.tmp.reset();
        r.e.prompt.reset();
    }
};

/// An address as its rank's buffer path and byte offset (python's name for it), so two ranks' temporaries compare.
fn expectAddr(sa: *const trace.Space, x: u64, sb: *const trace.Space, y: u64) !void {
    const p = sa.name(x);
    const q = sb.name(y);
    try testing.expectEqual(p == null, q == null);
    if (p == null) return testing.expectEqual(x, y);
    try testing.expectEqualStrings(p.?.path, q.?.path);
    try testing.expectEqual(p.?.off, q.?.off);
}

fn expectVal(sa: *const trace.Space, x: trace.Val, sb: *const trace.Space, y: trace.Val) !void {
    try testing.expectEqual(std.meta.activeTag(x), std.meta.activeTag(y));
    switch (x) {
        .t => |t| {
            try expectAddr(sa, t.ptr, sb, y.t.ptr);
            try testing.expectEqual(t.dtype, y.t.dtype);
            try testing.expectEqual(t.dims, y.t.dims);
            try testing.expectEqual(t.shape, y.t.shape);
            try testing.expectEqual(t.stride, y.t.stride);
        },
        .list => |l| {
            try testing.expectEqual(l.len, y.list.len);
            for (l, y.list) |u, v| try expectVal(sa, u, sb, v);
        },
        else => try testing.expectEqualDeep(x, y),
    }
}

/// Two traces with the same launches: op, name, grid, arguments (addresses by buffer name), constexprs and values.
pub fn expectSame(sa: *const trace.Space, a: []const trace.Event, sb: *const trace.Space, b: []const trace.Event) !void {
    _ = try compare(sa, a, sb, b, null);
}

/// Two ranks' traces in lockstep: the same launches and collectives in the same order, with the same buffers (by
/// name), shapes, grids and constexprs; scalar values may differ (a rank's own shard offset). Returns how many do, and
/// the launches they are in.
pub fn expectLockstep(sa: *const trace.Space, a: []const trace.Event, sb: *const trace.Space, b: []const trace.Event) !Lockstep {
    var l: Lockstep = .{};
    l.differ = try compare(sa, a, sb, b, &l);
    return l;
}

/// What two lockstep traces differ in: scalar values, and the names of the launches that hold them (the first few).
pub const Lockstep = struct {
    differ: usize = 0,
    names: [4][]const u8 = undefined,
    n: usize = 0,

    fn note(l: *Lockstep, name: []const u8) void {
        for (l.names[0..l.n]) |x| if (std.mem.eql(u8, x, name)) return;
        if (l.n < l.names.len) {
            l.names[l.n] = name;
            l.n += 1;
        }
    }
};

fn scalarsDiffer(x: anytype, y: @TypeOf(x)) bool {
    return !std.meta.eql(x, y);
}

fn compare(sa: *const trace.Space, a: []const trace.Event, sb: *const trace.Space, b: []const trace.Event, loose: ?*Lockstep) !usize {
    var differ: usize = 0;
    try testing.expectEqual(a.len, b.len);
    for (a, b, 0..) |x, y, i| {
        errdefer std.debug.print("launch {d}: {s} against {s}\n", .{ i, x.name, y.name });
        try testing.expectEqual(x.op, y.op);
        try testing.expectEqualStrings(x.name, y.name);
        try testing.expectEqual(x.grid, y.grid);
        try testing.expectEqual(x.args.len, y.args.len);
        const before = differ;
        for (x.args, y.args) |u, v| {
            try testing.expectEqualStrings(u.name, v.name);
            try testing.expectEqual(std.meta.activeTag(u.value), std.meta.activeTag(v.value));
            switch (u.value) {
                .ptr => |p| {
                    try testing.expectEqualStrings(p.ty, v.value.ptr.ty);
                    try expectAddr(sa, p.addr, sb, v.value.ptr.addr);
                },
                else => if (loose != null) {
                    if (scalarsDiffer(u.value, v.value)) differ += 1;
                } else try testing.expectEqualDeep(u.value, v.value),
            }
        }
        try testing.expectEqualDeep(x.consts, y.consts);
        try testing.expectEqual(x.vals.len, y.vals.len);
        for (x.vals, y.vals) |u, v| differ += try compareVal(sa, u, sb, v, loose != null);
        if (differ > before) loose.?.note(x.name);
    }
    return differ;
}

fn compareVal(sa: *const trace.Space, x: trace.Val, sb: *const trace.Space, y: trace.Val, loose: bool) !usize {
    try testing.expectEqual(std.meta.activeTag(x), std.meta.activeTag(y));
    switch (x) {
        .t, .list => {
            try expectVal(sa, x, sb, y);
            return 0;
        },
        else => {
            if (!loose) {
                try testing.expectEqualDeep(x, y);
                return 0;
            }
            return @intFromBool(scalarsDiffer(x, y));
        },
    }
}
