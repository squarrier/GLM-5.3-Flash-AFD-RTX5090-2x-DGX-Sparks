//! tf-exl3-check's host half: the case table exl3_check.py runs on one MoE layer's routed experts
//! of a TP2 rank, each case's arrays and scratch bytes by the Zig bindings, the listing both tools' --dry print (diffed
//! line for line), and linear.py's strips re-layout. The cases call python-0.6's EXL3 module (Ash Hart and the
//! TensorFold contributors; format after ExLlamaV3 by turboderp, MIT).

const std = @import("std");
const ex = @import("cuda_exl3_experts.zig");
const gm = @import("cuda_exl3_glm.zig");
const lin = @import("cuda_exl3_linear.zig");

/// GLM's top 8, then the shared expert's slot (pick = E: the kernels skip it).
pub const slots = 9;
pub const fp16_limit: f32 = 7.0;

/// Case kinds in the order exl3_check.py runs them; a tag is the kind's name in the manifest and the PASS lines.
pub const Kind = enum { @"routed-bf16", @"routed-f32", @"routed-wts", @"routed-fp16", @"routed-hot", combine, linear, unpack, dequant, @"glm-mm" };

pub const routed_rows = [_]struct { Kind, []const u32 }{
    .{ .@"routed-bf16", &.{ 1, 2, 3, 4, 5, 8, 16, 64, 128, 256, 594, 1024, 2048 } }, // qualification_b3fba1da: 128 and 594, the bench's
    .{ .@"routed-f32", &.{ 1, 8, 2048 } },
    .{ .@"routed-wts", &.{ 1, 8, 64 } },
    .{ .@"routed-fp16", &.{ 3, 8 } },
    .{ .@"routed-hot", &.{ 64, 2048 } },
};
pub const combine_rows = [_]u32{ 1, 8 };
pub const glm_rows = [_]u32{ 1, 8, 2048 };
pub const linear_m = [_]u32{ 1, 8, 64, 128 };
pub const experts_checked = [_]u32{ 0, 7 };

pub const Proj = enum { gate, up, down };
pub const DType = enum { bf16, f16, f32, i32 };

pub fn size(d: DType) u64 {
    return switch (d) {
        .bf16, .f16 => 2,
        .f32, .i32 => 4,
    };
}

pub fn isRouted(k: Kind) bool {
    return switch (k) {
        .@"routed-bf16", .@"routed-f32", .@"routed-wts", .@"routed-fp16", .@"routed-hot" => true,
        else => false,
    };
}

/// One case of the table.
pub const Case = struct {
    kind: Kind,
    rows: u32 = 0, // R (routed, combine, glm-mm) or M (linear)
    proj: Proj = .gate,
    expert: u32 = 0,
    layout: lin.Layout = .strips,
    out: DType = .f32, // linear's output

    /// The case's id, as exl3_check.py names it.
    pub fn id(c: Case, buf: []u8) []const u8 {
        return (switch (c.kind) {
            .linear => std.fmt.bufPrint(buf, "linear-{t}-e{d}-{d}-{t}-{t}", .{ c.proj, c.expert, c.rows, c.layout, c.out }),
            .unpack => std.fmt.bufPrint(buf, "unpack-{t}-e{d}-{t}", .{ c.proj, c.expert, c.layout }),
            .dequant => std.fmt.bufPrint(buf, "dequant-{t}-e{d}", .{ c.proj, c.expert }),
            else => std.fmt.bufPrint(buf, "{t}-{d}", .{ c.kind, c.rows }),
        }) catch unreachable;
    }
};

/// The case table in exl3_check.py's order.
pub fn table(gpa: std.mem.Allocator) ![]Case {
    var out: std.ArrayList(Case) = .empty;
    errdefer out.deinit(gpa);
    for (routed_rows) |r| for (r[1]) |R| try out.append(gpa, .{ .kind = r[0], .rows = R });
    for (combine_rows) |R| try out.append(gpa, .{ .kind = .combine, .rows = R });
    for ([_]Proj{ .gate, .down }) |p| for (experts_checked) |e| for ([_]lin.Layout{ .strips, .stored }) |l| {
        for (linear_m) |M| for ([_]DType{ .bf16, .f32 }) |d| try out.append(gpa, .{ .kind = .linear, .rows = M, .proj = p, .expert = e, .layout = l, .out = d });
        try out.append(gpa, .{ .kind = .unpack, .proj = p, .expert = e, .layout = l });
    };
    for ([_]Proj{ .gate, .up, .down }) |p| for (experts_checked) |e| try out.append(gpa, .{ .kind = .dequant, .proj = p, .expert = e });
    for (glm_rows) |R| try out.append(gpa, .{ .kind = .@"glm-mm", .rows = R });
    return out.toOwnedSlice(gpa);
}

/// What the cases' shapes depend on: the layer as the plan binds it, one expert's (k, n) a projection, GLM's limit.
pub const Facts = struct {
    layer: ex.Layer,
    gate: [2]u32, // up's are the same
    down: [2]u32,
    limit: f32,

    pub fn kn(f: Facts, p: Proj) [2]u32 {
        return if (p == .down) f.down else f.gate;
    }

    pub fn tiles(f: Facts) error{NoTile}![2]ex.Tile {
        return .{ try ex.defaultConfig(f.layer.dims, f.layer.width, true), try ex.defaultConfig(f.layer.width, f.layer.dims, false) };
    }
};

pub const Arr = struct {
    name: []const u8,
    dtype: DType,
    shape: [2]u64,
    dims: u8 = 2,

    pub fn bytes(a: Arr) u64 {
        var n = size(a.dtype);
        for (a.shape[0..a.dims]) |d| n *= d;
        return n;
    }
};

pub const Sized = struct { name: []const u8, bytes: u64 };

pub fn arr(name: []const u8, dtype: DType, rows: u64, cols: u64) Arr {
    return .{ .name = name, .dtype = dtype, .shape = .{ rows, cols } };
}

fn vec(name: []const u8, dtype: DType, n: u64) Arr {
    return .{ .name = name, .dtype = dtype, .shape = .{ n, 1 }, .dims = 1 };
}

/// A case's arrays, scratch bytes and call parameters, by the Zig bindings alone.
pub const Spec = struct {
    in: [5]Arr = undefined,
    n_in: u8 = 0,
    out: [2]Arr = undefined,
    n_out: u8 = 0,
    scratch: [12]Sized = undefined, // qualification_e4ad04cd: K4's cases hold 10
    n_scratch: u8 = 0,
    x_dtype: DType = .bf16,
    act: ex.Act = .bf16,
    limit: f32 = 0,
    split: [2]u32 = .{ 0, 0 }, // linear: (K splits, warps)

    pub fn input(s: *Spec, a: Arr) void {
        s.in[s.n_in] = a;
        s.n_in += 1;
    }

    pub fn output(s: *Spec, a: Arr) void {
        s.out[s.n_out] = a;
        s.n_out += 1;
    }

    pub fn add(s: *Spec, name: []const u8, bytes: u64) void {
        s.scratch[s.n_scratch] = .{ .name = name, .bytes = bytes };
        s.n_scratch += 1;
    }

    pub fn inputs(s: *const Spec) []const Arr {
        return s.in[0..s.n_in];
    }

    pub fn outputs(s: *const Spec) []const Arr {
        return s.out[0..s.n_out];
    }

    pub fn scratches(s: *const Spec) []const Sized {
        return s.scratch[0..s.n_scratch];
    }

    pub fn scratchOf(s: *const Spec, name: []const u8) u64 {
        for (s.scratches()) |x| if (std.mem.eql(u8, x.name, name)) return x.bytes;
        return 0;
    }
};

/// The case's arrays and scratch as exl3_check.py allocates them, derived from the bindings' own sizing.
pub fn spec(c: Case, f: Facts) error{NoTile}!Spec {
    var s: Spec = .{ .limit = f.limit };
    const D: u64 = f.layer.dims;
    const R: u64 = c.rows;
    const P = R * slots;
    switch (c.kind) {
        .@"routed-bf16", .@"routed-f32", .@"routed-wts", .@"routed-fp16", .@"routed-hot" => {
            s.x_dtype = if (c.kind == .@"routed-fp16") .f16 else .bf16;
            s.act = if (c.kind == .@"routed-f32") .f32 else .bf16;
            if (c.kind == .@"routed-fp16") s.limit = fp16_limit;
            s.input(arr("x", s.x_dtype, R, D));
            s.input(arr("pick", .i32, R, slots));
            s.output(arr("y", .f32, P, D));
            if (c.kind == .@"routed-wts") {
                s.input(arr("wts", .f32, R, slots));
                s.output(arr("out", .f32, R, D));
            }
            const t = try f.tiles();
            const b = ex.Scratch.bytes(c.rows, slots, f.layer, t[0], t[1]);
            for ([_]Sized{ .{ .name = "xg", .bytes = b.xg }, .{ .name = "xu", .bytes = b.xg }, .{ .name = "xd", .bytes = b.xd }, .{ .name = "z", .bytes = b.z }, .{ .name = "y", .bytes = b.y }, .{ .name = "ids", .bytes = b.ids }, .{ .name = "count", .bytes = 4 }, .{ .name = "members", .bytes = b.members } }) |x| s.add(x.name, x.bytes);
        },
        .combine => {
            s.input(arr("y", .f32, P, D));
            s.input(arr("wts", .f32, R, slots));
            s.output(arr("out", .f32, R, D));
        },
        .@"glm-mm" => {
            s.input(arr("x", .bf16, R, D));
            s.input(arr("pick", .i32, R, slots));
            s.input(arr("items", .i32, gm.maxItems(@intCast(P), f.layer.count + 1), 3));
            s.input(vec("counts", .i32, 2));
            s.input(vec("members", .i32, P));
            s.output(arr("y", .f32, P, D));
            const b = gm.Scratch.bytes(c.rows, slots, f.layer.dims, f.layer.width);
            for ([_]Sized{ .{ .name = "xg", .bytes = b.xg }, .{ .name = "xu", .bytes = b.xg }, .{ .name = "xd", .bytes = b.xd }, .{ .name = "z", .bytes = b.z } }) |x| s.add(x.name, x.bytes);
        },
        .linear => {
            const kn = f.kn(c.proj);
            s.split = lin.plan(kn[0], kn[1]);
            s.input(arr("x", .bf16, R, kn[0]));
            s.output(arr("y", c.out, R, kn[1]));
            s.add("xh", 2 * R * kn[0]);
            if (s.split[0] > 1) s.add("z", 4 * @as(u64, s.split[0]) * R * kn[1]);
            s.add("counters", 4 * 8 * @as(u64, kn[1] / 128));
        },
        .unpack, .dequant => {
            const kn = f.kn(c.proj);
            s.output(arr(if (c.kind == .unpack) "w" else "out", .f16, kn[0], kn[1]));
        },
    }
    return s;
}

fn lessArr(_: void, a: Arr, b: Arr) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

fn lessSized(_: void, a: Sized, b: Sized) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

fn arrs(w: *std.Io.Writer, xs: []const Arr) !void {
    for (xs, 0..) |a, i| {
        try w.print("{s}{s}:{t}[", .{ if (i > 0) " " else "", a.name, a.dtype });
        for (a.shape[0..a.dims], 0..) |d, j| try w.print("{s}{d}", .{ if (j > 0) "," else "", d });
        try w.writeAll("]");
    }
}

/// One case as both tools' --dry print it (exl3_check.py's `listing`): names sorted, dtypes and shapes, scratch bytes.
pub fn line(w: *std.Io.Writer, id: []const u8, kind: anytype, s: Spec) !void {
    var in = s.in;
    var out = s.out;
    var sc = s.scratch;
    std.mem.sort(Arr, in[0..s.n_in], {}, lessArr);
    std.mem.sort(Arr, out[0..s.n_out], {}, lessArr);
    std.mem.sort(Sized, sc[0..s.n_scratch], {}, lessSized);
    try w.print("{s} {t} | in ", .{ id, kind });
    try arrs(w, in[0..s.n_in]);
    try w.writeAll(" | out ");
    try arrs(w, out[0..s.n_out]);
    try w.writeAll(" | scratch ");
    if (s.n_scratch == 0) try w.writeAll("-");
    for (sc[0..s.n_scratch], 0..) |x, i| try w.print("{s}{s}={d}", .{ if (i > 0) " " else "", x.name, x.bytes });
    try w.writeAll("\n");
}

/// The listing's first lines: the layer as bound, each projection's trellis bytes an expert, the stacked scales.
pub fn head(w: *std.Io.Writer, layer: u32, rank: u32, f: Facts, trellis: [3]u64) !void {
    const l = f.layer;
    try w.print("layer {d} rank {d}/2: {d} experts, dims {d}, width {d}, codebook {d}, k2 gate/up {d}-{d}, down {d}-{d}, limit {d:.1}\n", .{ layer, rank, l.count, l.dims, l.width, @backingInt(l.cb), l.k2_gu[0], l.k2_gu[1], l.k2_d[0], l.k2_d[1], f.limit });
    try w.print("layout trellis bytes an expert: gate={d} up={d} down={d}\n", .{ trellis[0], trellis[1], trellis[2] });
    const E = l.count;
    try w.print("layout scales: suh_g:f16[{d},{d}] suh_u:f16[{d},{d}] svh_g:f16[{d},{d}] svh_u:f16[{d},{d}] suh_d:f16[{d},{d}] svh_d:f16[{d},{d}]\n", .{ E, f.gate[0], E, f.gate[0], E, f.gate[1], E, f.gate[1], E, f.down[0], E, f.down[1] });
}

/// linear.py strips(): trellis words [k/16, n/16, 8 k2] -> [n/128, k/16, 8, 8 k2], each 128-column block's tiles in k order.
pub fn strips(dst: []u8, src: []const u8, k: u32, n: u32, k2: u32) void {
    const kt: usize = k / 16;
    const nt: usize = n / 16;
    const tb: usize = 16 * @as(usize, k2); // bytes a 16 x 16 tile
    std.debug.assert(src.len == kt * nt * tb and dst.len == src.len and nt % 8 == 0);
    for (0..nt / 8) |b| for (0..kt) |kk| for (0..8) |j| {
        @memcpy(dst[((b * kt + kk) * 8 + j) * tb ..][0..tb], src[(kk * nt + b * 8 + j) * tb ..][0..tb]);
    };
}

const test_facts: Facts = .{ .layer = .{ .gate_ptr = 0, .up_ptr = 0, .down_ptr = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = 288, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } }, .gate = .{ 4096, 1024 }, .down = .{ 1024, 4096 }, .limit = 10.0 };

test "the table: 106 cases in exl3_check.py's order, ids as it names them" {
    const gpa = std.testing.allocator;
    const t = try table(gpa);
    defer gpa.free(t);
    try std.testing.expectEqual(@as(usize, 106), t.len);
    var n: [@typeInfo(Kind).@"enum".field_names.len]u32 = @splat(0);
    for (t) |c| n[@backingInt(c.kind)] += 1;
    try std.testing.expectEqual([_]u32{ 13, 3, 3, 2, 2, 2, 64, 8, 6, 3 }, n);
    var buf: [96]u8 = undefined;
    try std.testing.expectEqualStrings("routed-bf16-1", t[0].id(&buf));
    try std.testing.expectEqualStrings("routed-bf16-128", t[8].id(&buf));
    try std.testing.expectEqualStrings("routed-bf16-594", t[10].id(&buf));
    try std.testing.expectEqualStrings("linear-gate-e0-1-strips-bf16", t[25].id(&buf));
    try std.testing.expectEqualStrings("unpack-down-e7-stored", t[96].id(&buf));
    try std.testing.expectEqualStrings("glm-mm-2048", t[105].id(&buf));
}

test "the listing equals exl3_check.py's dry run on 078455ff (layer 3, rank 0)" {
    const want = [_]struct { Case, []const u8 }{
        .{ .{ .kind = .@"routed-bf16", .rows = 2048 }, "routed-bf16-2048 routed-bf16 | in pick:i32[2048,9] x:bf16[2048,4096] | out y:f32[18432,4096] | scratch count=4 ids=1152 members=2359296 xd=37748736 xg=150994944 xu=150994944 y=301989888 z=603979776\n" },
        .{ .{ .kind = .@"routed-wts", .rows = 1 }, "routed-wts-1 routed-wts | in pick:i32[1,9] wts:f32[1,9] x:bf16[1,4096] | out out:f32[1,4096] y:f32[9,4096] | scratch count=4 ids=36 members=36 xd=18432 xg=73728 xu=73728 y=147456 z=294912\n" },
        .{ .{ .kind = .combine, .rows = 1 }, "combine-1 combine | in wts:f32[1,9] y:f32[9,4096] | out out:f32[1,4096] | scratch -\n" },
        .{ .{ .kind = .linear, .rows = 128, .proj = .down, .expert = 7, .layout = .strips, .out = .bf16 }, "linear-down-e7-128-strips-bf16 linear | in x:bf16[128,1024] | out y:bf16[128,4096] | scratch counters=1024 xh=262144 z=4194304\n" },
        .{ .{ .kind = .unpack, .proj = .down, .expert = 0 }, "unpack-down-e0-strips unpack | in  | out w:f16[1024,4096] | scratch -\n" },
        .{ .{ .kind = .dequant, .proj = .up, .expert = 0 }, "dequant-up-e0 dequant | in  | out out:f16[4096,1024] | scratch -\n" },
        .{ .{ .kind = .@"glm-mm", .rows = 2048 }, "glm-mm-2048 glm-mm | in counts:i32[2] items:i32[1441,3] members:i32[18432] pick:i32[2048,9] x:bf16[2048,4096] | out y:f32[18432,4096] | scratch xd=37748736 xg=150994944 xu=150994944 z=2415919104\n" },
    };
    for (want) |w| {
        var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer aw.deinit();
        var buf: [96]u8 = undefined;
        try line(&aw.writer, w[0].id(&buf), w[0].kind, try spec(w[0], test_facts));
        try std.testing.expectEqualStrings(w[1], aw.written());
    }
}

test "strips: each 128-column block's tiles in k order" {
    var src: [2 * 16 * 32]u8 = undefined; // k 32, n 256, k2 2: 2 x 16 tiles of 32 bytes
    for (0..32) |t| @memset(src[t * 32 ..][0..32], @intCast(t));
    var dst: [src.len]u8 = undefined;
    strips(&dst, &src, 32, 256, 2);
    const order = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 };
    for (order, 0..) |t, i| try std.testing.expectEqual(t, dst[i * 32]);
}
