//! grammar_tests recorded CPU operand contract, not a CUDA mask kernel or device comparator.
const std = @import("std");
const sample = @import("cuda_sample.zig");
const trace = @import("cuda_trace.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

pub const Rank = struct { values: []const f32, ids: []const i64 };
pub const Cpu = struct {
    gpa: std.mem.Allocator,
    width: usize,
    ranks: []const Rank,
    fail: bool = false,
    picks: usize = 0,
    endpoint: ?@import("lanes").structured_control.Endpoint = null,
    local_rank: ?usize = null,
    permit: ?@import("lanes").structured_control.Permit = null,

    pub fn check(c: *const Cpu, f: *const Fwd) !void {
        if (!@import("builtin").is_test or f.gpu != null or f.rec == null) return error.IncompleteMaskedAdapter;
        if (c.width == 0 or c.width > std.math.maxInt(i32) or c.width != f.m.vocab or c.ranks.len == 0 or c.ranks.len != f.m.world or c.width % c.ranks.len != 0) return error.MaskShape;
        if (c.endpoint != null and (c.local_rank == null or c.local_rank.? != f.m.rank or c.endpoint.?.peer != f.m.rank)) return error.MaskShape;
        const seen = try c.gpa.alloc(u32, try words(c.width));
        defer c.gpa.free(seen);
        @memset(seen, 0);
        for (c.ranks) |rank| {
            if (rank.values.len != f.m.vl() or rank.ids.len != rank.values.len) return error.MaskShape;
            for (rank.ids) |id| {
                if (id < 0 or id >= c.width) return error.MaskId;
                const i: usize = @intCast(id);
                const bit = @as(u32, 1) << @as(u5, @intCast(i % 32));
                if (seen[i / 32] & bit != 0) return error.MaskId;
                seen[i / 32] |= bit;
            }
        }
    }
};

pub fn complete(c: *Cpu, candidate: @import("lanes").structured_control.Candidate) !u32 {
    const endpoint = c.endpoint orelse return error.IncompleteMaskedAdapter;
    const permit = c.permit orelse return error.IncompleteMaskedAdapter;
    const result = try endpoint.after(permit, .{ .logits_valid = true, .candidate = candidate });
    switch (result.status) {
        .ok => {}, .cancelled => return error.Cancelled, .empty_mask => return error.EmptyMask,
        .request_error => return error.StructuredRequestError, .poison => return error.ControlPoison,
    }
    if (!result.candidate.valid) return error.EmptyMask;
    return result.candidate.global_id;
}

pub fn words(width: usize) !usize {
    if (width == 0 or width > std.math.maxInt(i32)) return error.MaskShape;
    return (width + 31) / 32;
}

pub fn shape(width: usize, mask: []const u32) !void {
    if (mask.len != 0 and mask.len != try words(width)) return error.MaskShape;
}

pub fn eligible(mask: []const u32, id: i64) bool {
    if (mask.len == 0) return true;
    const t: usize = @intCast(id);
    return mask[t / 32] & (@as(u32, 1) << @as(u5, @intCast(t % 32))) != 0;
}

pub fn allAllowed(width: usize, mask: []const u32) bool {
    if (mask.len == 0) return true;
    for (0..width) |id| if (!eligible(mask, @intCast(id))) return false;
    return true;
}

pub fn pick(c: *Cpu, f: *Fwd, mask: []const u32) !u32 {
    try c.check(f);
    try shape(c.width, mask);
    const t = f.rec.?;
    const mw = try c.gpa.alloc(trace.Val, mask.len);
    defer c.gpa.free(mw);
    for (mw, mask) |*v, word| v.* = .{ .int = word };
    var values: [256]f32 = undefined;
    var ids: [256]i64 = undefined;
    if (c.ranks.len > values.len) return error.MaskShape;
    var valid: usize = 0;
    for (c.ranks, 0..) |rank, r| {
        if (c.endpoint != null and r != c.local_rank.?) continue;
        const mapping = try c.gpa.alloc(trace.Val, rank.ids.len);
        defer c.gpa.free(mapping);
        for (mapping, rank.ids) |*v, id| v.* = .{ .int = id };
        try t.call(.ext, "grammar_tests.cpu.eligibility", &.{ .{ .int = @intCast(r) }, .{ .int = @intCast(c.width) }, .{ .list = mw }, .{ .list = mapping } });
        const row = try c.gpa.alloc(f32, rank.values.len);
        defer c.gpa.free(row);
        const columns = try c.gpa.alloc(usize, rank.values.len);
        defer c.gpa.free(columns);
        var allowed: usize = 0;
        for (rank.values, rank.ids, 0..) |v, id, i| {
            if (!eligible(mask, id)) continue;
            row[allowed] = v;
            columns[allowed] = i;
            allowed += 1;
        }
        const k = @min(allowed, 1); // existing native greedy top-k=1, unchanged
        try t.call(.ext, "grammar_tests.cpu.masked_topk", &.{ .{ .int = @intCast(r) }, .{ .bool = allowed != 0 }, .{ .int = @intCast(k) } });
        if (valid + k > values.len) return error.MaskShape;
        if (k != 0) {
            var order: [1 + sample.margin]usize = undefined;
            sample.topkOrder(row[0..allowed], k, &order);
            for (order[0..k]) |at| {
                const column = columns[at];
                values[valid] = rank.values[column];
                ids[valid] = rank.ids[column];
                valid += 1;
            }
        }
    }
    try t.call(.comm, "grammar_tests.cpu.validity_gather", &.{ .{ .int = @intCast(c.ranks.len) }, .{ .int = @intCast(valid) } });
    if (c.fail) {
        if (c.endpoint) |endpoint| {
            _ = endpoint.quiesce(.poison) catch {};
            return error.ControlPoison;
        }
        return error.GatherFailed;
    }
    if (c.endpoint != null) {
        const candidate = @import("lanes").structured_control.Candidate{ .valid = valid != 0,
            .value_f32_bits = if (valid != 0) @bitCast(values[0]) else 0,
            .global_id = if (valid != 0) @intCast(ids[0]) else 0 };
        const result = try complete(c, candidate);
        try t.call(.ext, "grammar_tests.cpu.choose", &.{.{ .int = result }});
        c.picks += 1;
        return result;
    }
    if (valid == 0) return error.EmptyMask;
    const result = sample.choose(values[0..valid], ids[0..valid]);
    try t.call(.ext, "grammar_tests.cpu.choose", &.{.{ .int = result }});
    c.picks += 1;
    return result;
}
