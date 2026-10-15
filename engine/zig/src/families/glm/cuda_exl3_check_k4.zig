//! tf-exl3-check's K4 table (qualification_e4ad04cd; TF_GLM_HALVES_KERNEL=k4 on the tool, TF_EXL3_CHECK_K4=1 on exl3_check.py with
//! python 2.15's tree): layer L's routed experts of a TP2 rank through MiaAI-Lab's EXL3 prompt kernel as python 2.15's
//! halves call it for a prompt window (experts.routed with prompt, pass, order, rowrot), at 2.15's BENCHMARKS §Quality
//! rows up to 2,049 (its 4,096 dropped: the grouping's R x 9 slots x 4 B of shared memory, 147,456 B, is over the
//! GB10's and the 5090's opt-in 101,376 B, in python 2.15 as here), then pass 128, her order off and rowrot off: each
//! case's arrays and scratch bytes as tools/zig/exl3_check_k4.py allocates them, by the bindings (cuda_exl3_k4.zig).

const std = @import("std");
const chk = @import("cuda_exl3_check.zig");
const k4 = @import("cuda_exl3_k4.zig");
const ex = @import("cuda_exl3_experts.zig");
const k3 = @import("cuda_exl3_kernels.zig");

pub const Kind = enum { @"routed-k4", @"routed-k4-p128", @"routed-k4-noorder", @"routed-k4-norowrot" };

/// exl3_check_k4.py's K4: each kind's rows and switches, in its order.
pub const rows = [_]struct { Kind, []const u32, k4.Opts }{
    .{ .@"routed-k4", &.{ 1, 2, 7, 16, 33, 64, 65, 128, 594, 2048, 2049 }, .{} },
    .{ .@"routed-k4-p128", &.{ 7, 594, 2048 }, .{ .pass = 128 } },
    .{ .@"routed-k4-noorder", &.{ 65, 2048 }, .{ .order = false } },
    .{ .@"routed-k4-norowrot", &.{ 65, 2048 }, .{ .xrow = false } },
};

pub const Case = struct {
    kind: Kind,
    rows: u32,
    opts: k4.Opts,

    pub fn id(c: Case, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "{t}-{d}", .{ c.kind, c.rows }) catch unreachable;
    }
};

pub fn table(gpa: std.mem.Allocator) ![]Case {
    var out: std.ArrayList(Case) = .empty;
    errdefer out.deinit(gpa);
    for (rows) |r| for (r[1]) |R| try out.append(gpa, .{ .kind = r[0], .rows = R, .opts = r[2] });
    return out.toOwnedSlice(gpa);
}

/// The case's arrays and scratch as exl3_check_k4.py allocates them: x and pick in; y [P, D] fp32 and xd [P, I] fp16
/// out; experts.Scratch's xg, xu, xd, y, ids, count, members (z is not K4's) and _prompt_buffers' items, counts and
/// order (sized at pass 64, python's smaller pass).
pub fn spec(c: Case, f: chk.Facts) chk.Spec {
    var s: chk.Spec = .{ .limit = f.limit };
    const D: u64 = f.layer.dims;
    const I: u64 = f.layer.width;
    const E: u32 = f.layer.count;
    const R: u64 = c.rows;
    const P = R * chk.slots;
    const maxu = @min(P, E);
    const cap: u64 = k4.promptItems(c.rows, chk.slots, E, k4.passes[0]);
    s.input(chk.arr("x", .bf16, R, D));
    s.input(chk.arr("pick", .i32, R, chk.slots));
    s.output(chk.arr("y", .f32, P, D));
    s.output(chk.arr("xd", .f16, P, I));
    for ([_]chk.Sized{ .{ .name = "xg", .bytes = 2 * P * D }, .{ .name = "xu", .bytes = 2 * P * D }, .{ .name = "xd", .bytes = 2 * P * I }, .{ .name = "y", .bytes = 4 * P * D }, .{ .name = "ids", .bytes = 4 * maxu }, .{ .name = "count", .bytes = 4 }, .{ .name = "members", .bytes = 4 * maxu * R }, .{ .name = "items", .bytes = 12 * cap }, .{ .name = "counts", .bytes = 8 }, .{ .name = "order", .bytes = 4 * cap } }) |x| s.add(x.name, x.bytes);
    return s;
}

/// The listing's first line with K4's mark (exl3_check_cases.py's head[0] + ", K4 (shared suh N)").
pub fn head(w: *std.Io.Writer, layer: u32, rank: u32, f: chk.Facts, trellis: [3]u64, shared: bool) !void {
    var aw: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer aw.deinit();
    try chk.head(&aw.writer, layer, rank, f, trellis);
    const text = aw.written();
    const nl = std.mem.indexOfScalar(u8, text, '\n').?;
    try w.print("{s}, K4 (shared suh {d}){s}", .{ text[0..nl], @intFromBool(shared), text[nl..] });
}

const test_facts: chk.Facts = .{ .layer = .{ .gate_ptr = 0, .up_ptr = 0, .down_ptr = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } }, .gate = .{ 4096, 1024 }, .down = .{ 1024, 4096 }, .limit = 10.0 };

test "K4's table: 18 cases in exl3_check_k4.py's order; the listing equals its dry run on 078455ff (layer 3, rank 0)" {
    const gpa = std.testing.allocator;
    const t = try table(gpa);
    defer gpa.free(t);
    try std.testing.expectEqual(@as(usize, 18), t.len);
    var buf: [96]u8 = undefined;
    try std.testing.expectEqualStrings("routed-k4-1", t[0].id(&buf));
    try std.testing.expectEqualStrings("routed-k4-2049", t[10].id(&buf));
    try std.testing.expectEqualStrings("routed-k4-norowrot-2048", t[17].id(&buf));
    try std.testing.expectEqual(k4.Opts{ .pass = 128 }, t[11].opts);
    // every case's grouping fits the opt-in shared memory of sm_120 and sm_121 (python 2.15's exl3x_group_cuda check;
    // 2.15's 4,096 rows need 147,456 B): the bindings take each at GB10's 101,376 B
    for (t) |c| {
        var ls: k3.Launches = .{};
        try ex.group(&ls, 0, 0, 0, 0, c.rows, chk.slots, 288, c.rows, .{ .optin = 101376 });
    }
    var big: k3.Launches = .{};
    try std.testing.expectError(error.Invalid, ex.group(&big, 0, 0, 0, 0, 4096, chk.slots, 288, 4096, .{ .optin = 101376 }));
    // exl3_check.py --dry with TF_EXL3_CHECK_K4=1 over qualification_765330ce's header mirror (python 2.15 src@0849f3f): its dry.txt
    const want = [_]struct { usize, []const u8 }{
        .{ 0, "routed-k4-1 routed-k4 | in pick:i32[1,9] x:bf16[1,4096] | out xd:f16[9,1024] y:f32[9,4096] | scratch count=4 counts=8 ids=36 items=108 members=36 order=36 xd=18432 xg=73728 xu=73728 y=147456\n" },
        .{ 2, "routed-k4-7 routed-k4 | in pick:i32[7,9] x:bf16[7,4096] | out xd:f16[63,1024] y:f32[63,4096] | scratch count=4 counts=8 ids=252 items=756 members=1764 order=252 xd=129024 xg=516096 xu=516096 y=1032192\n" },
    };
    for (want) |w| {
        var aw: std.Io.Writer.Allocating = .init(gpa);
        defer aw.deinit();
        try chk.line(&aw.writer, t[w[0]].id(&buf), t[w[0]].kind, spec(t[w[0]], test_facts));
        try std.testing.expectEqualStrings(w[1], aw.written());
    }
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try head(&aw.writer, 3, 0, test_facts, .{ 2097152, 2097152, 2097152 }, true);
    try std.testing.expect(std.mem.startsWith(u8, aw.written(), "layer 3 rank 0/2: 288 experts, dims 4096, width 1024, codebook 1, k2 gate/up 8-8, down 8-8, limit 10.0, K4 (shared suh 1)\nlayout trellis bytes an expert: gate=2097152 up=2097152 down=2097152\n"));
}
