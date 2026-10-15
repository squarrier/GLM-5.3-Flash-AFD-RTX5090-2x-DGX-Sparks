//! tf-exl3-check's prompt windows (qualification_b3fba1da): X1 and SEQ in the routed cases (TF_GLM_EXL3_X1, TF_GLM_EXL3_SEQ, as the
//! halves read them), and `--bench N` at 1-16 rows and the prompt windows 64, 128, 594, 1,024 and 2,048 rows. With SEQ
//! on, the bench alternates X1's launches and SEQ's run by run, so both arms' medians come from one process.

const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const x3 = glm.exl3;
const ex = x3.experts;

/// The prompt windows `--bench` times besides 1-16 rows (exl3_check_cases.py's routed-bf16 rows from 64).
pub const bench_rows = [_]u32{ 64, 128, 594, 1024, 2048 };

pub fn benched(rows: u32) bool {
    if (rows <= 16) return true;
    for (bench_rows) |r| if (r == rows) return true;
    return false;
}

/// X1's buffers for the routed cases: the order list (python's Scratch.order at the widest case) and the layer's shared
/// suh row (python's test on the group's gate and up suh), on the device or as --dry stand-ins.
pub const X1 = struct {
    p: ex.Prompt = .{},
    ord: u64 = 0,
    shared: ?u64 = null,
    order_buf: ?cuda.DeviceBuffer = null,
    suh_buf: ?cuda.DeviceBuffer = null,
    note: [160]u8 = undefined,
    note_len: usize = 0,

    /// From the group's bytes (host) before they are freed: the shared-suh test, the order list sized for `rows` rows
    /// of `slots` (orderSlots), the images when on. `next`: --dry's stand-in addresses (d null).
    pub fn setup(x: *X1, d: ?*const cuda.Driver, set: ?*x3.kernels.Set, group: []const u8, e: x3.split.Experts, rows: u32, slots: u32, next: *u64) !void {
        if (!x.p.on()) return;
        if (d) |dv| {
            try set.?.loadX1(dv);
            if (x.p.seq) try set.?.loadX1s(dv);
        }
        const D: usize = e.gate.k;
        const n = @as(usize, e.count) * D;
        const g = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, group[e.gate.suh.off..][0 .. 2 * n]));
        const u = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, group[e.up.suh.off..][0 .. 2 * n]));
        var gh = try std.heap.page_allocator.alloc(u16, 2 * n);
        defer std.heap.page_allocator.free(gh);
        for (g, 0..) |v, i| gh[i] = v;
        for (u, 0..) |v, i| gh[n + i] = v;
        const shared = glm.afd_expert.sharedSuh(gh[0..n], gh[n..], D);
        const cap = ex.orderSlots(rows, slots, e.count);
        if (d) |dv| {
            x.order_buf = try cuda.DeviceBuffer.alloc(dv, 4 * @as(usize, cap));
            x.ord = x.order_buf.?.ptr;
            if (shared) {
                x.suh_buf = try cuda.DeviceBuffer.alloc(dv, 2 * D);
                try x.suh_buf.?.upload(0, std.mem.sliceAsBytes(gh[0..D]));
                x.shared = x.suh_buf.?.ptr;
            }
        } else {
            x.ord = next.*;
            next.* += std.mem.alignForward(u64, 4 * @as(u64, cap), 256) + 256;
            if (shared) {
                x.shared = next.*;
                next.* += std.mem.alignForward(u64, 2 * D, 256) + 256;
            }
        }
        const t = std.fmt.bufPrint(&x.note, "nfirst {d}, prefetch {d}, order {d} ({d} slots), rowrot {d} (shared suh {s}), seq {d}", .{ @intFromBool(x.p.nfirst), @intFromBool(x.p.pa), @intFromBool(x.p.order), cap, @intFromBool(x.p.xrow), if (shared) "yes" else "no", @intFromBool(x.p.seq) }) catch "";
        x.note_len = t.len;
    }

    pub fn deinit(x: *X1) void {
        if (x.order_buf) |*b| b.free();
        if (x.suh_buf) |*b| b.free();
        x.order_buf = null;
        x.suh_buf = null;
    }

    /// The kernels' label in BENCH lines.
    pub fn label(p: ex.Prompt, sd: ex.Sched) []const u8 {
        if (p.seq) return "x1+seq";
        if (p.on()) return "x1";
        return if (sd.on()) "0128" else "today";
    }
};

fn isGrouped(l: x3.kernels.Launch) bool {
    return std.mem.startsWith(u8, l.symbol, "_ZN8tf_exl3x14grouped_kernel") or std.mem.startsWith(u8, l.symbol, "_ZN8tf_exl3x18grouped_seq_kernel");
}

/// A part of a routed call: the whole call, gate/up's grouped launch with its epilogue, gate/up's alone, down's alone.
const Part = enum { routed, gu_epi, gate_up, down };

fn partOf(ls: *const x3.kernels.Launches, part: Part) !x3.kernels.Launches {
    if (part == .routed) return ls.*;
    var g: [2]usize = .{ 0, 0 };
    var n: usize = 0;
    for (ls.slice(), 0..) |l, i| if (n < 2 and isGrouped(l)) {
        g[n] = i;
        n += 1;
    };
    if (n != 2) return error.NoGroupedLaunches;
    var out: x3.kernels.Launches = .{};
    switch (part) {
        .gu_epi => for (ls.slice()[g[0] .. g[0] + 2]) |l| out.push(l),
        .gate_up => out.push(ls.slice()[g[0]]),
        .down => out.push(ls.slice()[g[1]]),
        .routed => unreachable,
    }
    for (out.items[0..out.len]) |*l| if (part != .routed) {
        l.cfg.pdl = false; // a launch alone: no programmatic dependency on a kernel that is not there
    };
    return out;
}

/// --bench N: each part N times an arm, the arms alternating run by run (one arm: as before), each run between two
/// events on the stream; medians, p10 and p90 in us, and with two arms their ratio (BENCH-AB).
pub fn bench(gpa: std.mem.Allocator, w: *std.Io.Writer, d: *const cuda.Driver, stream: cuda.Stream, set: x3.kernels.Set, arms: []const *const x3.kernels.Launches, labels: []const []const u8, rows: u32, n: u32, sd: ex.Sched) !void {
    const runs = n * @as(u32, @intCast(arms.len));
    const ev = try gpa.alloc(cuda.Event, 2 * runs);
    defer gpa.free(ev);
    var made: usize = 0;
    defer for (ev[0..made]) |*e| e.deinit();
    while (made < ev.len) : (made += 1) ev[made] = try cuda.Event.init(d, true);
    const us = try gpa.alloc(f64, runs);
    defer gpa.free(us);
    const one = try gpa.alloc(f64, n);
    defer gpa.free(one);
    inline for (@typeInfo(Part).@"enum".field_names) |name| {
        const part = @field(Part, name);
        var lists: [2]x3.kernels.Launches = undefined;
        for (arms, 0..) |a, i| lists[i] = try partOf(a, part);
        for (0..runs) |r| {
            try ev[2 * r].record(stream);
            try set.run(stream, &lists[r % arms.len]);
            try ev[2 * r + 1].record(stream);
        }
        try ev[2 * runs - 1].synchronize();
        for (us, 0..) |*u, r| u.* = @as(f64, try cuda.Event.elapsedMs(ev[2 * r], ev[2 * r + 1])) * 1000;
        var med: [2]f64 = .{ 0, 0 };
        for (0..arms.len) |a| {
            for (one, 0..) |*o, i| o.* = us[i * arms.len + a];
            std.mem.sort(f64, one, {}, std.sort.asc(f64));
            med[a] = one[n / 2];
            try w.print("BENCH routed-bf16 r{d} {s}: median {d:.1} us, p10 {d:.1}, p90 {d:.1} ({d} runs; l2d {d}, pdl {d}, kernels {s})\n", .{ rows, name, one[n / 2], one[n / 10], one[n - 1 - n / 10], n, sd.l2d, @intFromBool(sd.pdl), labels[a] });
        }
        if (arms.len == 2) try w.print("BENCH-AB routed-bf16 r{d} {s}: {s} / {s} = {d:.4} ({d:.1} / {d:.1} us, alternating, {d} runs each)\n", .{ rows, name, labels[1], labels[0], med[1] / med[0], med[1], med[0], n });
        try w.flush();
    }
}

test "the bench's rows: 1-16 and the five prompt windows; the parts of a routed call by its grouped launches" {
    for ([_]u32{ 1, 2, 3, 4, 5, 8, 16, 64, 128, 594, 1024, 2048 }) |r| try std.testing.expect(benched(r));
    for ([_]u32{ 17, 63, 256, 1900, 4096 }) |r| try std.testing.expect(!benched(r));
    const l: ex.Layer = .{ .gate_ptr = 1, .up_ptr = 2, .down_ptr = 3, .gate_k2 = 4, .up_k2 = 5, .down_k2 = 6, .suh_g = 7, .suh_u = 8, .svh_g = 9, .svh_u = 10, .suh_d = 11, .svh_d = 12, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
    const s: ex.Scratch = .{ .xg = 0x10000, .xu = 0x20000, .xd = 0x30000, .z = 0x40000, .y = 0x50000, .ids = 0x60000, .count = 0x70000, .members = 0x80000, .rows = 2048, .slots = 9, .experts = 288, .cfg_gu = ex.glm_gateup, .cfg_d = ex.glm_down };
    for ([_]bool{ false, true }) |seq| {
        var ls: x3.kernels.Launches = .{};
        try ex.routedX1(&ls, 0xd000, 4096, true, 0xe000, null, l, s, 0, 1024, 7.0, .bf16, true, .{ .optin = 101376 }, .{}, .{ .nfirst = true, .pa = true, .order = true, .xrow = true, .seq = seq }, 0x90000, 0xf000);
        const ge = try partOf(&ls, .gu_epi);
        try std.testing.expectEqual(@as(usize, 2), ge.len);
        try std.testing.expect(isGrouped(ge.items[0]));
        try std.testing.expectEqualStrings(x3.kernels.sym.gateup_epilogue, ge.items[1].symbol);
        try std.testing.expectEqual(if (seq) x3.kernels.Image.exl3_x1s_grouped_cb1 else .exl3_x1_grouped_cb1, (try partOf(&ls, .gate_up)).items[0].image);
        try std.testing.expectEqual(x3.kernels.Image.exl3_x1_grouped_cb1, (try partOf(&ls, .down)).items[0].image);
        try std.testing.expectEqual(ls.len, (try partOf(&ls, .routed)).len);
    }
}
