//! check-weights' reference (qualification_cec3c04c; moved from afd_expert.zig by qualification_21e21bc0, unchanged): qualification_2b1188e0's weights.json as the expert
//! half's check-weights compares against it (`--ref`), and the tally of equal, differing and missing entries.
const std = @import("std");
const glm = @import("glm_cuda");
const ae = glm.afd_expert;
const Allocator = std.mem.Allocator;

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// qualification_2b1188e0's weights.json (its `table`: path -> shape, dtype, sha256, bytes, address_table): check-weights' reference.
pub const Ref = struct {
    parsed: std.json.Parsed(std.json.Value),
    table: std.json.ObjectMap,
    covered: [64]u32 = @splat(0), // its routed-expert entries a layer

    pub const Verdict = enum { equal, differ, absent };

    pub fn init(gpa: Allocator, text: []const u8) !Ref {
        const p = try std.json.parseFromSlice(std.json.Value, gpa, text, .{ .allocate = .alloc_always });
        errdefer p.deinit();
        const t = if (p.value == .object) p.value.object.get("table") else null;
        if (t == null or t.? != .object) return error.RefHasNoTable;
        var r: Ref = .{ .parsed = p, .table = t.?.object };
        for (r.table.keys()) |k| if (layerOf(k)) |l| {
            r.covered[l] += 1;
        };
        return r;
    }

    pub fn deinit(r: *Ref) void {
        r.parsed.deinit();
    }

    /// Entry `e` against its path's: dtype, shape, bytes, the address-table flag, and with `data` the digest (`sum`;
    /// null for a table).
    pub fn check(r: *const Ref, e: ae.Entry, sum: ?[32]u8, data: bool) Verdict {
        var buf: [96]u8 = undefined;
        const v = r.table.get(ae.path(e, &buf)) orelse return .absent;
        return if (same(v, e, sum, data)) .equal else .differ;
    }

    fn same(v: std.json.Value, e: ae.Entry, sum: ?[32]u8, data: bool) bool {
        if (v != .object) return false;
        const o = v.object;
        const dt = o.get("dtype") orelse return false;
        if (dt != .string or !eql(dt.string, e.dtype)) return false;
        const sh = o.get("shape") orelse return false;
        if (sh != .array or sh.array.items.len != e.dims) return false;
        for (sh.array.items, e.shape[0..e.dims]) |x, d| if (x != .integer or x.integer != @as(i64, @intCast(d))) return false;
        const by = o.get("bytes") orelse return false;
        if (by != .integer or by.integer != @as(i64, @intCast(e.bytes))) return false;
        const tab = if (o.get("address_table")) |x| x == .bool and x.bool else false;
        if (tab != ae.isTable(e)) return false;
        if (!data) return true;
        const s = o.get("sha256") orelse return false;
        if (sum) |x| return s == .string and eql(s.string, &std.fmt.bytesToHex(x, .lower));
        return s == .null;
    }
};

/// L of a routed-expert path `w.layers[L].moe.experts.…` (L below 64).
fn layerOf(k: []const u8) ?usize {
    const pre = "w.layers[";
    if (!std.mem.startsWith(u8, k, pre)) return null;
    const close = std.mem.indexOfScalarPos(u8, k, pre.len, ']') orelse return null;
    if (!std.mem.startsWith(u8, k[close..], "].moe.experts.")) return null;
    const l = std.fmt.parseInt(usize, k[pre.len..close], 10) catch return null;
    return if (l < 64) l else null;
}

/// check-weights against a ref: entries equal or differing; ours absent from a layer it covers (`extra`) or in one it
/// does not (`outside`); and its entries of our layers that we lack (`lacking`). Passes with some equal, none else.
pub const Tally = struct {
    equal: u64 = 0,
    differ: u64 = 0,
    extra: u64 = 0,
    outside: u64 = 0,
    lacking: u64 = 0,
    seen: [64]u32 = @splat(0),
    first: [96]u8 = undefined,
    first_len: usize = 0,

    pub fn add(t: *Tally, r: *const Ref, e: ae.Entry, v: Ref.Verdict) void {
        const covered = e.layer < 64 and r.covered[e.layer] > 0;
        switch (v) {
            .equal => t.equal += 1,
            .differ => t.differ += 1,
            .absent => if (covered) {
                t.extra += 1;
            } else {
                t.outside += 1;
            },
        }
        if (v != .absent and e.layer < 64) t.seen[e.layer] += 1;
        if ((v == .differ or (v == .absent and covered)) and t.first_len == 0) t.first_len = ae.path(e, &t.first).len;
    }

    pub fn finish(t: *Tally, r: *const Ref, layers: []const u32) void {
        for (layers) |l| if (l < 64 and r.covered[l] > t.seen[l]) {
            t.lacking += r.covered[l] - t.seen[l];
        };
    }

    pub fn ok(t: *const Tally) bool {
        return t.equal > 0 and t.differ == 0 and t.extra == 0 and t.lacking == 0;
    }
};

test "--ref: an entry against qualification_2b1188e0's: equal, a wrong digest, dtype or shape differs, a path it lacks; the tally's verdict" {
    const gpa = std.testing.allocator;
    const k2: ae.Entry = .{ .layer = 3, .field = .gate_k2, .dtype = "int32", .shape = .{ 288, 0, 0 }, .dims = 1, .off = 0, .bytes = 1152 };
    const ptr: ae.Entry = .{ .layer = 3, .field = .gate_ptr, .dtype = "int64", .shape = .{ 288, 0, 0 }, .dims = 1, .off = 0, .bytes = 2304 };
    const keep: ae.Entry = .{ .layer = 3, .field = .keep, .j = 1, .dtype = "int16", .shape = .{ 256, 64, 64 }, .dims = 3, .off = 0, .bytes = 2097152 };
    const data: [1152]u8 = @splat(0);
    const sum = (try ae.digest(k2, &data)).?;
    const text = try std.fmt.allocPrint(gpa,
        \\{{"tensors": 3, "table": {{"w.embed": {{"shape": [1], "dtype": "bfloat16", "sha256": "00", "bytes": 2}},
        \\"w.layers[3].moe.experts.gate_ptr": {{"shape": [288], "dtype": "int64", "sha256": null, "address_table": true, "bytes": 2304}},
        \\"w.layers[3].moe.experts.gate_k2": {{"shape": [288], "dtype": "int32", "sha256": "{s}", "bytes": 1152}},
        \\"w.layers[3].moe.experts.keep[0]": {{"shape": [256, 64, 64], "dtype": "int16", "sha256": "ab", "bytes": 2097152}}}}}}
    , .{&std.fmt.bytesToHex(sum, .lower)});
    defer gpa.free(text);
    var r = try Ref.init(gpa, text);
    defer r.deinit();
    try std.testing.expectEqual(@as(u32, 3), r.covered[3]);
    try std.testing.expectEqual(Ref.Verdict.equal, r.check(k2, sum, true));
    try std.testing.expectEqual(Ref.Verdict.equal, r.check(ptr, null, true));
    try std.testing.expectEqual(Ref.Verdict.differ, r.check(k2, @as([32]u8, @splat(1)), true));
    try std.testing.expectEqual(Ref.Verdict.equal, r.check(k2, null, false)); // a list without bytes: no digest
    var wrong = k2;
    wrong.dtype = "int64";
    try std.testing.expectEqual(Ref.Verdict.differ, r.check(wrong, sum, true));
    wrong = k2;
    wrong.shape[0] = 144;
    try std.testing.expectEqual(Ref.Verdict.differ, r.check(wrong, sum, true));
    try std.testing.expectEqual(Ref.Verdict.absent, r.check(keep, null, false)); // keep[1]: not in it
    var t: Tally = .{};
    t.add(&r, k2, r.check(k2, sum, true));
    t.add(&r, ptr, r.check(ptr, null, true));
    t.finish(&r, &.{3});
    try std.testing.expect(!t.ok()); // it has keep[0], which we lack
    try std.testing.expectEqual(@as(u64, 1), t.lacking);
    var u: Tally = .{};
    u.add(&r, keep, r.check(keep, null, false));
    try std.testing.expectEqual(@as(u64, 1), u.extra);
    try std.testing.expectEqualStrings("w.layers[3].moe.experts.keep[1]", u.first[0..u.first_len]);
    var outside = keep;
    outside.layer = 9;
    u.add(&r, outside, r.check(outside, null, false));
    try std.testing.expectEqual(@as(u64, 1), u.outside);
    try std.testing.expectEqual(@as(?usize, 44), layerOf("w.layers[44].moe.experts.keep[863]"));
    try std.testing.expectEqual(@as(?usize, null), layerOf("w.layers[3].attn.q"));
}
