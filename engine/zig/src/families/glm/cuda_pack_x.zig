//! EXL3's pack ops (qualification_qualification_cec53599), beside cuda_pack.zig's MLX ones and over the same stored groups: stored tensors back to
//! back in one buffer (stack_b16, the KDA conv's torch.cat), stored bytes left in place (experts.prepare keeps each
//! trellis it is given, and its suh and svh stacks are the split's: the group is never freed), a layer's pointer and
//! width tables written there from the host (qualification_765330ce's Plan.tables), qmm.quantize4's draft head (glm_pack.cu's quant4
//! kernels over the stored head rows, then make_q4's layout), and patch 0002's make_f8 and stack_f8 (the FP8 head; the
//! MTP layer on FP8, mtp_fp8). No qualification_765330ce type is named here: qualification_qualification_44ba554e-fwd-b compiles it alone.
const std = @import("std");
const cuda = @import("cuda");
const split = @import("cuda_split.zig");
const pack = @import("cuda_pack.zig");
const fwd = @import("cuda_fwd.zig");

const Ref = split.Ref;

/// Stored tensors back to back: `lens[i]` bytes of `srcs[i]`, in order (python's stack_b16 rows, torch.cat).
pub const Cat = struct { srcs: [6]Ref = undefined, lens: [6]u64 = undefined, n: u8 = 0 };

/// A layer's pointer and width tables (36 E bytes) written at `dst` by `write(ctx, layer, base, out)`, `base` being
/// the address of `dst`'s group: qualification_765330ce's Plan.tables, so the pointers are the trellises' own addresses.
pub const Tables = struct {
    dst: Ref,
    layer: u32,
    bytes: u32,
    ctx: *const anyopaque,
    write: *const fn (ctx: *const anyopaque, layer: usize, base: u64, out: []u8) void,
};

/// qmm.quantize4 of `n` stored bf16 rows of `k` (the head's rank rows), as make_q4 leaves its `part`.
pub const Quant = struct { src: Ref, n: usize, k: usize, part: enum { weight, scales, biases } };

/// patch 0002's make_f8 of `n` stored bf16 rows of `k` (the head's rank rows, qualification_qualification_ebed70b7): e4m3 bytes or fp32 block scales.
pub const F8 = struct { src: Ref, n: usize, k: usize, part: enum { weight, scale } };

/// patch 0002's stack_f8 (mtp_fp8, the MTP layer on FP8): make_f8 of `cat`'s stored bf16 rows of `k`, back to back.
pub const F8Cat = struct { cat: Cat, k: usize, part: enum { weight, scale } };

pub const Op = union(enum) { cat: Cat, alias: Ref, tables: Tables, quant: Quant, f8: F8, f8cat: F8Cat };

/// The last stored group the op reads: its buffer is made once that group is loaded.
pub fn home(x: Op) u32 {
    return switch (x) {
        .cat => |c| catHome(c),
        .alias => |r| r.g,
        .tables => |t| t.dst.g,
        .quant => |q| q.src.g,
        .f8 => |q| q.src.g,
        .f8cat => |q| catHome(q.cat),
    };
}

fn catHome(c: Cat) u32 {
    var g: u32 = 0;
    for (c.srcs[0..c.n]) |r| g = @max(g, r.g);
    return g;
}

pub fn reads(x: Op, g: u32) bool {
    return switch (x) {
        .cat => |c| catReads(c, g),
        .alias => |r| r.g == g,
        .tables => |t| t.dst.g == g,
        .quant => |q| q.src.g == g,
        .f8 => |q| q.src.g == g,
        .f8cat => |q| catReads(q.cat, g),
    };
}

fn catReads(c: Cat, g: u32) bool {
    for (c.srcs[0..c.n]) |r| if (r.g == g) return true;
    return false;
}

/// The stored bytes an op's buffer is (left in place, so their group is kept), else null.
pub fn inPlace(x: Op) ?Ref {
    return switch (x) {
        .alias => |r| r,
        .tables => |t| t.dst,
        else => null,
    };
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn blocks(total: u64) [3]u32 {
    return .{ @intCast((total + 255) / 256), 1, 1 };
}

/// One EXL3 item's copies, table write or launches; `stored[g]` is stored group g's address, `it.buf.ptr` the output.
pub fn make(f: *fwd.Fwd, e: pack.Engine, it: *const pack.Item, x: Op, stored: []const u64) !void {
    const out = it.buf.ptr;
    switch (x) {
        .cat => |c| {
            var at: u64 = 0;
            for (c.srcs[0..c.n], c.lens[0..c.n]) |r, n| {
                try pack.copyBytes(f, e, out + at, stored[r.g] + r.off, n);
                at += n;
            }
            if (at != it.buf.bytes()) return error.CatBytes;
        },
        .alias => {},
        .tables => |t| {
            const at = stored[t.dst.g] + t.dst.off;
            if (f.rec) |r| return r.call(.ext, "glm_pack.tables", &.{ .{ .int = @bitCast(at) }, .{ .int = t.layer }, .{ .int = t.bytes } });
            var buf: [36 * 1024]u8 = undefined;
            if (t.bytes > buf.len) return error.TablesTooBig;
            t.write(t.ctx, t.layer, stored[t.dst.g], buf[0..t.bytes]);
            if (f.gpu) |g| {
                const d = switch (e) {
                    .gpu => |k| k.d,
                    .host => return error.NoKernels,
                };
                try g.stream.synchronize(); // the group's zero fill and uploads are on the stream; this copy is not
                return d.check(d.api.cuMemcpyHtoD_v2(at, &buf, t.bytes), "cuMemcpyHtoD");
            }
            @memcpy(@as([*]u8, @ptrFromInt(at))[0..t.bytes], buf[0..t.bytes]);
        },
        .quant => |q| {
            const npad = (q.n + 127) / 128 * 128;
            var a: cuda.Args = .{};
            a.add(stored[q.src.g] + q.src.off);
            a.add(int(q.n));
            a.add(int(q.k));
            switch (q.part) {
                .weight => {
                    const total: u64 = it.buf.numel();
                    a.add(out);
                    a.add(@as(c_longlong, @intCast(total)));
                    try pack.launch(f, e, .quant4, blocks(total), 256, &a);
                },
                .scales, .biases => {
                    a.add(int(npad));
                    a.add(@as(c_int, @intFromBool(q.part == .biases)));
                    a.add(out);
                    try pack.launch(f, e, .quant4s, blocks(@as(u64, q.k / 64) * npad), 256, &a);
                },
            }
        },
        .f8 => |q| try f8Rows(f, e, stored[q.src.g] + q.src.off, q.n, q.k, q.part == .weight, out),
        .f8cat => |q| { // make_f8 reads each row alone: each part's rows made where the stack puts them
            const per: u64 = if (q.part == .weight) q.k else q.k / 128 * 4; // the output's bytes a row
            var rows: u64 = 0;
            for (q.cat.srcs[0..q.cat.n], q.cat.lens[0..q.cat.n]) |r, len| {
                if (len % (q.k * 2) != 0) return error.CatBytes;
                const n: usize = @intCast(len / (q.k * 2));
                try f8Rows(f, e, stored[r.g] + r.off, n, q.k, q.part == .weight, out + rows * per);
                rows += n;
            }
            if (rows * per != it.buf.bytes()) return error.CatBytes;
        },
    }
}

/// glm_pack.cu's make_f8 of `n` bf16 rows of `k` at `src`: e4m3 bytes (`weight`) or fp32 block scales into `out`.
fn f8Rows(f: *fwd.Fwd, e: pack.Engine, src: u64, n: usize, k: usize, weight: bool, out: u64) !void {
    const total: u64 = if (weight) @as(u64, n) * (k / 4) else @as(u64, n) * (k / 128);
    var a: cuda.Args = .{};
    a.add(src);
    a.add(int(n));
    a.add(int(k));
    a.add(out);
    a.add(@as(c_longlong, @intCast(total)));
    try pack.launch(f, e, if (weight) .f8 else .f8s, blocks(total), 256, &a);
}
