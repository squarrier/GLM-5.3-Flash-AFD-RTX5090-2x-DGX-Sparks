//! The MoE block's launch half against python's calls (qualification_qualification_736508ad). For every MoE case of the forward fixtures (layer 3's
//! MLP half, the MTP layer's moe_block; R = 1, 2, 4, 8), the launches a GPU makes on the stand-in addresses must equal
//! the launches the bindings make from python's recorded calls, argument byte for byte:
//! - MLX: cuda_moe.q4Calls through cuda_experts' launch halves, against A1's reader (cuda_kernels_mm_calls.launches);
//! - EXL3: cuda_moe.exl3Launches (qualification_qualification_44ba554e-exl3's experts.routed), against qualification_qualification_44ba554e-exl3's single-call bindings fed python's six
//!   calls (group, rot_in, grouped, gateup_epilogue, grouped, down_epilogue), as experts.cpp/cu reads them.
const std = @import("std");
const lt = @import("cuda_layer_test.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const layer = @import("cuda_layer.zig");
const mm = @import("cuda_kernels_mm.zig");
const experts = @import("cuda_experts.zig");
const moe = @import("cuda_moe.zig");
const exl3 = @import("cuda_exl3.zig");

const ex3 = exl3.experts;
const k3 = exl3.kernels;
const calls = mm.calls;
const testing = std.testing;
const J = std.json.Value;

fn get(o: J, key: []const u8) J {
    return o.object.get(key) orelse std.debug.panic("fixture: no key {s}", .{key});
}

fn int(v: J) i64 {
    return v.integer;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// Python's tensors at our stand-in addresses: the named buffer's base plus python's byte offset.
const Names = struct {
    space: *const trace.Space,

    pub fn addr(n: Names, t: J) !u64 {
        const buf = get(t, "buf").string;
        for (n.space.entries.items) |e| if (eql(e.path, buf)) return e.base + @as(u64, @intCast(int(get(t, "off"))));
        std.debug.print("python's buffer {s}: not ours\n", .{buf});
        return error.UnknownBuffer;
    }
};

/// The case's MoE views: layer 3's for an MLP half, the MTP layer's for its moe_block; null for a dense or whole layer
/// and for the attention cases (A3's "attn" and "dsa" blocks).
fn moeOf(r: *const lt.Rank, c: J) ?views.Moe {
    const block = (c.object.get("block") orelse return null).string;
    if (eql(block, "moe")) return r.mtp.?.moe;
    if (!eql(block, "mlp")) return null;
    return r.w.layers[@intCast(int(get(c, "layer")))].moe;
}

/// The case's ext events whose fn starts with `prefix`, in order.
fn extEvents(c: J, prefix: []const u8, out: []J) !usize {
    var n: usize = 0;
    for (get(c, "events").array.items) |ev| {
        if (!eql(get(ev, "op").string, "ext") or !std.mem.startsWith(u8, get(ev, "fn").string, prefix)) continue;
        if (n == out.len) return error.ExtraCall;
        out[n] = ev;
        n += 1;
    }
    return n;
}

/// A python experts.* event as cuda_kernels_mm_calls reads a call: fn past "experts.", fp32 values as a double's bits.
fn asCall(a: std.mem.Allocator, ev: J) !J {
    var args = std.json.Array.init(a);
    for (get(ev, "args").array.items) |x| {
        const bits = x.object.get("f32") orelse {
            try args.append(x);
            continue;
        };
        var o: std.json.ObjectMap = .empty;
        try o.put(a, "float", .{ .integer = @bitCast(@as(f64, @as(f32, @bitCast(@as(u32, @intCast(bits.integer)))))) });
        try args.append(.{ .object = o });
    }
    var o: std.json.ObjectMap = .empty;
    try o.put(a, "fn", .{ .string = get(ev, "fn").string["experts.".len..] });
    try o.put(a, "args", .{ .array = args });
    return .{ .object = o };
}

fn sameLaunch(a: mm.Launch, b: mm.Launch) !void {
    try testing.expectEqual(a.kernel, b.kernel);
    try testing.expectEqualDeep(a.cfg, b.cfg);
    try testing.expectEqual(a.args.count, b.args.count);
    try testing.expectEqualSlices(u16, a.args.offsets[0..a.args.count], b.args.offsets[0..b.args.count]);
    try testing.expectEqualSlices(u16, a.args.sizes[0..a.args.count], b.args.sizes[0..b.args.count]);
    try testing.expectEqualSlices(u8, a.args.storage[0..a.args.used], b.args.storage[0..b.args.used]);
}

test "MLX experts: the forward's launch half makes python's plan, gate_up and down launches in every MoE case" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, lt.fixture_mlx, .{});
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var r = try lt.Rank.init(gpa, parsed.value);
    defer r.deinit(gpa);
    const chip: mm.Chip = .{ .major = 12, .minor = 1, .sms = 48 }; // the fixture's GB10 (glm_forward_fixtures _chip)
    const per_sm = [2]usize{ 3, 5 }; // any occupancy answer: both sides read the same one
    const names: Names = .{ .space = &r.space };
    var n: usize = 0;
    var it = get(parsed.value, "cases").object.iterator();
    while (it.next()) |kv| {
        const x = moeOf(&r, kv.value_ptr.*) orelse continue;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        const R: usize = @intCast(int(get(kv.value_ptr.*, "rows")));
        const q = try moe.q4Calls(&r.m, x.experts.q4, &r.b, R);
        var py: [3]J = undefined;
        try testing.expectEqual(@as(usize, 3), try extEvents(kv.value_ptr.*, "experts.", &py));
        const want_plan = try calls.launches(try asCall(arena.allocator(), py[0]), chip, per_sm, names);
        const got_plan = try experts.planLaunches(q.route);
        try testing.expectEqual(want_plan.n, got_plan.n);
        for (want_plan.l[0..want_plan.n], got_plan.l[0..got_plan.n]) |w, g| try sameLaunch(w, g);
        for ([2]experts.Call{ q.gate_up, q.down }, py[1..3]) |call, p| {
            const want = try calls.launches(try asCall(arena.allocator(), p), chip, per_sm, names);
            const got = try experts.launchOf(call, chip, per_sm);
            try testing.expectEqual(want.n, @intFromBool(got != null));
            if (got) |g| try sameLaunch(want.l[0], g);
        }
        n += 1;
    }
    try testing.expectEqual(@as(usize, 8), n);
}

/// The launches qualification_qualification_44ba554e-exl3's binding makes from one of python's exl3_experts calls (experts.cpp and experts.cu read
/// maxm as members.size(1), nexp_max as uids.size(0), x's dtype for rot_in's instance).
fn bind(out: *k3.Launches, ev: J, n: Names, sh: ex3.Shared) !void {
    const f = get(ev, "fn").string["exl3_experts.".len..];
    const a = get(ev, "args").array.items;
    const A = struct {
        a: []const J,
        n: Names,
        fn t(s: @This(), i: usize) !u64 {
            return s.n.addr(get(s.a[i], "t"));
        }
        fn u(s: @This(), i: usize) u32 {
            return @intCast(int(get(s.a[i], "int")));
        }
        fn dim(s: @This(), i: usize, d: usize) u32 {
            return @intCast(int(get(get(s.a[i], "t"), "shape").array.items[d]));
        }
    };
    const x: A = .{ .a = a, .n = n };
    if (eql(f, "group")) return ex3.group(out, try x.t(0), try x.t(1), try x.t(2), try x.t(3), x.u(4), x.u(5), x.u(6), x.dim(3, 1), sh);
    if (eql(f, "rot_in")) return ex3.rotIn(out, try x.t(0), x.u(1), eql(get(get(a[0], "t"), "dtype").string, "bf16"), try x.t(2), try x.t(3), try x.t(4), try x.t(5), try x.t(6), x.u(7), x.u(8), x.u(9), x.u(10));
    if (eql(f, "grouped")) return ex3.grouped(out, .{
        .x0 = try x.t(0), .x1 = try x.t(1), .tp0 = try x.t(2), .tp1 = try x.t(3), .k2_0 = try x.t(4), .k2_1 = try x.t(5),
        .uids = try x.t(6), .ucount = try x.t(7), .members = try x.t(8), .z = try x.t(9), .mats = x.u(10), .K = x.u(11),
        .N = x.u(12), .P = x.u(13), .SK = x.u(14), .slots = x.u(15), .maxm = x.dim(8, 1), .nexp_max = x.dim(6, 0),
        .cb = @enumFromInt(x.u(16)), .nt = x.u(17), .warps = x.u(18), .pf = x.u(19), .lo = x.u(20), .hi = x.u(21),
    });
    if (eql(f, "gateup_epilogue")) {
        const limit: f32 = @bitCast(@as(u32, @intCast(int(get(a[12], "f32")))));
        return ex3.gateupEpilogue(out, try x.t(0), try x.t(1), try x.t(2), try x.t(3), try x.t(4), try x.t(5), x.u(6), x.u(7), x.u(8), x.u(9), x.u(10), x.u(11), limit, @enumFromInt(x.u(13)));
    }
    if (eql(f, "down_epilogue")) return ex3.downEpilogue(out, try x.t(0), try x.t(1), try x.t(2), try x.t(3), x.u(4), x.u(5), x.u(6), x.u(7), x.u(8), x.u(9));
    return error.UnknownCall;
}

fn sameExl3(a: k3.Launch, b: k3.Launch) !void {
    try testing.expectEqual(a.image, b.image);
    try testing.expectEqualStrings(a.symbol, b.symbol);
    try testing.expectEqualDeep(a.cfg, b.cfg);
    try testing.expectEqual(a.max_dynamic, b.max_dynamic);
    try testing.expectEqual(a.args.count, b.args.count);
    try testing.expectEqualSlices(u16, a.args.offsets[0..a.args.count], b.args.offsets[0..b.args.count]);
    try testing.expectEqualSlices(u16, a.args.sizes[0..a.args.count], b.args.sizes[0..b.args.count]);
    try testing.expectEqualSlices(u8, a.args.storage[0..a.args.used], b.args.storage[0..b.args.used]);
}

test "EXL3 routed experts: the forward's launch half makes the launches qualification_qualification_44ba554e-exl3's bindings make from python's calls" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, lt.fixture_exl3, .{});
    defer parsed.deinit();
    var r = try lt.Rank.init(gpa, parsed.value);
    defer r.deinit(gpa);
    const names: Names = .{ .space = &r.space };
    const sh: ex3.Shared = .{ .optin = 101376 }; // a GB10's sharedMemPerBlockOptin; decode windows stay under 48 KiB
    var n: usize = 0;
    var it = get(parsed.value, "cases").object.iterator();
    while (it.next()) |kv| {
        const x = moeOf(&r, kv.value_ptr.*) orelse continue;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        const R: usize = @intCast(int(get(kv.value_ptr.*, "rows")));
        var got: k3.Launches = .{};
        try moe.exl3Launches(&got, layer.rowsOf(r.b.normed, R), x.experts.exl3, &r.b, R, r.m.limit, sh);
        var py: [6]J = undefined;
        try testing.expectEqual(@as(usize, 6), try extEvents(kv.value_ptr.*, "exl3_experts.", &py));
        var want: k3.Launches = .{};
        for (py) |ev| try bind(&want, ev, names, sh);
        try testing.expectEqual(@as(usize, 6), want.len);
        try testing.expectEqual(want.len, got.len);
        for (want.slice(), got.slice()) |w, g| try sameExl3(w, g);
        n += 1;
    }
    try testing.expectEqual(@as(usize, 8), n);
}

test {
    _ = experts;
    _ = moe;
}
