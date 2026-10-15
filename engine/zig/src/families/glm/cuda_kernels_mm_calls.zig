//! python-0.6's recorded qmm and experts extension calls (the JSON of tools/zig/glm_mm_fixtures.py on a CPU and of
//! zig/tests/cuda/glm/mm_oracle.py on a GPU) as the bindings' launches: the CPU fixture test compares them with python's
//! own launches, tf-glm-mm replays the same launches on the GPU against python's outputs.
const std = @import("std");
const mm = @import("cuda_kernels_mm.zig");

const J = std.json.Value;

pub const Error = mm.Error || error{BadCall};

/// A tensor argument: an address (the resolver's), shape and strides in elements.
pub const Ten = struct { ptr: u64, dims: usize, shape: [5]usize = @splat(1), stride: [5]usize = @splat(1) };

/// Up to three launches (experts.cpp plan's wide form), in order.
pub const Launches = struct { n: usize = 0, l: [3]mm.Launch = undefined };

pub fn get(o: J, name: []const u8) Error!J {
    if (o != .object) return error.BadCall;
    return o.object.get(name) orelse error.BadCall;
}

fn int(v: J) Error!i64 {
    const x = try get(v, "int");
    return if (x == .integer) x.integer else error.BadCall;
}

fn usz(v: J) Error!usize {
    const x = try int(v);
    return if (x >= 0) @intCast(x) else error.BadCall;
}

fn flag(v: J) Error!bool {
    const x = try get(v, "bool");
    return if (x == .bool) x.bool else error.BadCall;
}

/// A float argument, carried as its double's bits.
fn real(v: J) Error!f64 {
    const bits: u64 = switch (try get(v, "float")) {
        .integer => |i| @bitCast(i),
        .number_string => |s| std.fmt.parseInt(u64, s, 10) catch return error.BadCall,
        else => return error.BadCall,
    };
    return @bitCast(bits);
}

/// A tensor argument {"t": {...}}; `r.addr(t)` gives its address (a stand-in's, or a device buffer's plus the offset).
pub fn tensor(v: J, r: anytype) !Ten {
    const t = try get(v, "t");
    const shape = (try get(t, "shape")).array.items;
    const stride = (try get(t, "stride")).array.items;
    if (shape.len == 0 or shape.len > 5 or stride.len != shape.len) return error.BadCall;
    var out: Ten = .{ .ptr = try r.addr(t), .dims = shape.len };
    for (shape, stride, 0..) |s, d, i| {
        if (s != .integer or d != .integer or s.integer < 0 or d.integer < 0) return error.BadCall;
        out.shape[i] = @intCast(s.integer);
        out.stride[i] = @intCast(d.integer);
    }
    return out;
}

/// A one-element list (GLM's calls group one weight).
fn only(v: J) Error!J {
    const l = (try get(v, "list")).array.items;
    return if (l.len == 1) l[0] else error.BadCall;
}

fn one(l: mm.Launch) Launches {
    var o: Launches = .{ .n = 1 };
    o.l[0] = l;
    return o;
}

fn maybe(l: ?mm.Launch) Launches {
    return if (l) |x| one(x) else .{};
}

/// experts.cpp run and prefill's arguments: gs, epi, x, slots, w, kg, nb, items, counts, members, out, n, limit, bound.
fn experts(a: []const J, r: anytype) !mm.Experts {
    if (a.len != 14) return error.BadCall;
    const x = try tensor(a[2], r);
    return .{
        .gs = try usz(a[0]), .epi = std.math.cast(u8, try usz(a[1])) orelse return error.BadCall, .x = x.ptr,
        .x_stride = x.stride[0], .slots = try usz(a[3]), .w = (try tensor(a[4], r)).ptr, .kg = try usz(a[5]),
        .nb = try usz(a[6]), .items = (try tensor(a[7], r)).ptr, .counts = (try tensor(a[8], r)).ptr,
        .members = (try tensor(a[9], r)).ptr, .out = (try tensor(a[10], r)).ptr, .n = try usz(a[11]),
        .limit = try real(a[12]), .bound = try usz(a[13]),
    };
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// The launches python's call (qmm qmm_group or qmm_prefill; experts plan, run or prefill) makes on `chip`; `per_sm` the
/// decode experts' resident blocks an SM (gate and up, down) as the wrapper's occupancy query answers.
pub fn launches(call: J, chip: mm.Chip, per_sm: [2]usize, r: anytype) !Launches {
    const f = (try get(call, "fn")).string;
    const a = (try get(call, "args")).array.items;
    if (eql(f, "qmm_group")) {
        if (a.len != 11 or try int(a[10]) != -1) return error.BadCall;
        const w = [3]Ten{ try tensor(try only(a[2]), r), try tensor(try only(a[3]), r), try tensor(try only(a[4]), r) };
        return one(try mm.groupCall(chip, try tensor(a[0], r), try tensor(a[1], r), w, try tensor(try only(a[5]), r), try usz(try only(a[6])), try usz(try only(a[7])), try flag(a[8]), try int(a[9])));
    }
    if (eql(f, "qmm_prefill")) {
        if (a.len != 9) return error.BadCall;
        if (try usz(a[6]) != 64) return error.NotBuilt;
        const w = [3]Ten{ try tensor(a[1], r), try tensor(a[2], r), try tensor(a[3], r) };
        return one(try mm.prefillCall(try tensor(a[0], r), w, try tensor(a[4], r), try usz(a[5]), try flag(a[7]), try int(a[8])));
    }
    if (eql(f, "plan")) {
        if (a.len != 9) return error.BadCall;
        const p: mm.Plan = .{ .members = (try tensor(a[4], r)).ptr, .items = (try tensor(a[5], r)).ptr, .counts = (try tensor(a[6], r)).ptr, .rank = (try tensor(a[7], r)).ptr, .hist = (try tensor(a[8], r)).ptr };
        const ls = try mm.planLaunches((try tensor(a[0], r)).ptr, try usz(a[1]), try usz(a[2]), try usz(a[3]), p);
        var o: Launches = .{ .n = ls.n };
        @memcpy(o.l[0..ls.n], ls.l[0..ls.n]);
        return o;
    }
    if (eql(f, "run")) {
        const e = try experts(a, r);
        return maybe(try mm.runLaunch(e, per_sm[if (e.epi == 2) 0 else 1], chip.sms));
    }
    if (eql(f, "prefill")) return maybe(try mm.prefillExpertsLaunch(try experts(a, r)));
    return error.BadCall;
}

/// The mangled name past its namespace (python's anonymous ones, our tf_glm_* ones): how the fixtures name kernels.
pub fn key(symbol: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, symbol, "_ZN")) return symbol;
    var i: usize = 3;
    var n: usize = 0;
    while (i < symbol.len and std.ascii.isDigit(symbol[i])) : (i += 1) n = n * 10 + (symbol[i] - '0');
    return symbol[@min(symbol.len, i + n)..];
}

/// The kernel of the four images that `k` (a fixture's key) names, if any.
pub fn kernelOf(k: []const u8) ?mm.Kernel {
    for (std.enums.values(mm.Kernel)) |kernel| if (eql(key(kernel.symbol()), k)) return kernel;
    return null;
}

test "keys drop the namespace" {
    try std.testing.expectEqualStrings("11plan_kernelEPKiiiiPiS2_S2_", key(mm.sym.plan));
    try std.testing.expectEqualStrings("11plan_kernelEPKiiiiPiS2_S2_", key("_ZN43_GLOBAL__N__0fe325b1_10_experts_cu_9b8c6efb11plan_kernelEPKiiiiPiS2_S2_"));
    try std.testing.expectEqual(mm.Kernel.down, kernelOf(key(mm.sym.down)).?);
    try std.testing.expect(kernelOf("12group_kernelILi64ELi8ELi128E") == null);
}
