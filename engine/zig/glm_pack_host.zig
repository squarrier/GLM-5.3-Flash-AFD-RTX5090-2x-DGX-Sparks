//! G-layout (qualification_qualification_7c75d777 PLAN §4) on the CPU, in `zig build test`: the pack pass's host reference (glm/cuda_pack_ref.zig)
//! against python-0.6's own packing (fixtures_cuda_pack.json: tools/zig/glm_forward_fixtures.py `pack`, sha256 of
//! kernels/qmm.py pack, stack_q4, kv_b's rows, latent.AbsorbQ4, the conv reshape), and glm_pack.cu emulated on the
//! CPU (zig/tests/cuda/glm_pack_emu.cpp, one call a CUDA thread) through cuda_pack.make against the reference; the
//! experts kernel also against python-0.6's own experts_pack.cu (the tree's checked copy) stacked as experts.make does.
//! TF_GLM_PACK_REAL=<json> adds the real slices' cases (the same schema, inputs read from a checkpoint folder).

const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const pack = glm.pack;
const ref = glm.pack_ref;
const Buf = glm.views.Buf;
const Sha256 = std.crypto.hash.sha2.Sha256;
const J = std.json.Value;

const fixture = @embedFile("src/families/glm/fixtures_cuda_pack.json");

test {
    _ = @import("tests/cuda/glm/pack_f8.zig"); // qualification_qualification_ebed70b7: patch 0002's make_f8 (glm_pack.cu's f8 kernels)
    _ = @import("tests/cuda/glm/pack_mtp8.zig"); // mtp_fp8: its stack_f8 for the MTP layer on FP8 (cuda_pack_x's f8cat)
}

extern fn tf_glm_pack_emu(kernel: c_int, grid: *const [3]c_uint, block: *const [3]c_uint, params: [*]?*anyopaque, sizes: [*]const u16, count: c_uint) c_int;

/// cuda_pack's host engine: each launch through the emulator (kernel numbers as cuda_pack.Kern).
fn emu(kern: pack.Kern, grid: [3]u32, block: [3]u32, args: *cuda.Args) anyerror!void {
    try emuKernel(@backingInt(kern), grid, block, args);
}

fn emuKernel(kernel: c_int, grid: [3]u32, block: [3]u32, args: *cuda.Args) !void {
    const rc = tf_glm_pack_emu(kernel, &grid, &block, args.pointers().?, &args.sizes, @intCast(args.count));
    if (rc != 0) return error.EmulatorRefused;
}

/// One case's stored bytes (a single group) and its packed items, outputs in host memory.
const Case = struct {
    a: std.mem.Allocator,
    stored: []align(16) u8 = &.{},
    items: std.ArrayList(pack.Item) = .empty,
    outs: std.ArrayList([]align(16) u8) = .empty,

    fn group(c: *Case, bytes: usize) !void {
        c.stored = try c.a.alignedAlloc(u8, .@"16", bytes);
    }
    fn add(c: *Case, name: []const u8, buf: Buf, op: pack.Op) !void {
        var b = buf;
        const out = try c.a.alignedAlloc(u8, .@"16", b.bytes());
        @memset(out, 0xA5); // every byte must be written
        b.ptr = @intFromPtr(out.ptr);
        try c.outs.append(c.a, out);
        try c.items.append(c.a, .{ .path = name, .buf = b, .op = op, .home = 0 });
    }
    /// An EXL3 op's buffer (cuda_pack_x.zig), its `op` the stored ref it reads first, as pack.exl3.plan leaves it.
    fn addX(c: *Case, name: []const u8, buf: Buf, x: pack.x_ops.Op) !void {
        try c.add(name, buf, .{ .copy = switch (x) {
            .quant => |q| q.src,
            .cat => |k| k.srcs[0],
            else => return error.UnknownCase,
        } });
        c.items.items[c.items.items.len - 1].x = x;
    }
    /// An MLX triplet [n, k] made into python's Q4 (rows of the stored triplet at w, s, b).
    fn q4(c: *Case, prefix: []const u8, w: u64, s: u64, b: u64, n: usize, k: usize, rows: pack.Rows) !void {
        const v = glm.views.Q4.of(n, k);
        try c.add(try std.fmt.allocPrint(c.a, "{s}weight", .{prefix}), v.weight, .{ .words = .{ .src = .{ .g = 0, .off = w }, .n = n, .k = k, .rows = rows } });
        try c.add(try std.fmt.allocPrint(c.a, "{s}scales", .{prefix}), v.scales, .{ .major = .{ .src = .{ .g = 0, .off = s }, .n = n, .k = k, .rows = rows } });
        try c.add(try std.fmt.allocPrint(c.a, "{s}biases", .{prefix}), v.biases, .{ .major = .{ .src = .{ .g = 0, .off = b }, .n = n, .k = k, .rows = rows } });
    }
};

fn int(o: J, key: []const u8) usize {
    return @intCast(o.object.get(key).?.integer);
}

/// Drawn bf16 made finite as the python fixture does: sign and mantissa kept, the exponent folded into 2^-15 .. 2^0.
fn finite(b: []align(16) u8) void {
    for (std.mem.bytesAsSlice(u16, b)) |*h| h.* = (h.* & 0x807F) | ((0x70 + ((h.* >> 7) & 0xF)) << 7);
}

/// quantize4 (EXL3's draft head) of drawn finite rows, and stack_b16's torch.cat of drawn parts.
fn buildX(c: *Case, o: J, op: []const u8, d: *ref.Draw) !void {
    const k = int(o, "k");
    if (std.mem.eql(u8, op, "quant4")) {
        const n = int(o, "n");
        try c.group(n * k * 2);
        d.halves(c.stored);
        finite(c.stored);
        const h = std.mem.bytesAsSlice(u16, c.stored);
        if (o.object.get("flat")) |fl| { // every flat-th row's first group one value: quantize4's 1e-8 floor
            var r: usize = 0;
            while (r < n) : (r += @intCast(fl.integer)) @memset(h[r * k + 1 ..][0..63], h[r * k]);
        }
        const v = glm.views.Q4.of(n, k);
        for ([_]Buf{ v.weight, v.scales, v.biases }, [_][]const u8{ "weight", "scales", "biases" }, 0..) |buf, leaf, j|
            try c.addX(leaf, buf, .{ .quant = .{ .src = .{ .g = 0, .off = 0 }, .n = n, .k = k, .part = @enumFromInt(j) } });
        return;
    }
    // the parts drawn in order, stored last to first with a 256-byte gap before each: the op's refs carry the order
    const ns = o.object.get("ns").?.array.items;
    var total: usize = 0;
    for (ns) |x| total += @as(usize, @intCast(x.integer)) * k * 2 + 256;
    try c.group(total);
    var cat: pack.x_ops.Cat = .{ .n = @intCast(ns.len) };
    var at = total;
    var rows: usize = 0;
    for (ns, 0..) |x, j| {
        const len = @as(usize, @intCast(x.integer)) * k * 2;
        at -= len;
        d.halves(c.stored[at..][0..len]);
        cat.srcs[j] = .{ .g = 0, .off = at };
        cat.lens[j] = len;
        at -= 256;
        @memset(c.stored[at..][0..256], 0x5A);
        rows += @intCast(x.integer);
    }
    return c.addX("weight", Buf.of(.bf16, &.{ rows, k }), .{ .cat = cat });
}

/// A case's inputs drawn (or read) and its items, in the python fixture's order and names.
fn build(c: *Case, o: J, real: ?*Real) !void {
    const op = o.object.get("op").?.string;
    var d: ref.Draw = .{ .s = if (o.object.get("seed")) |s| @intCast(s.integer) else 0 };
    if (std.mem.eql(u8, op, "quant4") or std.mem.eql(u8, op, "cat")) return buildX(c, o, op, &d);
    if (std.mem.eql(u8, op, "q4") or std.mem.eql(u8, op, "stack") or std.mem.eql(u8, op, "kv_b")) {
        const k = int(o, "k");
        var ns: [8]usize = undefined;
        var parts: usize = 1;
        if (std.mem.eql(u8, op, "stack")) {
            parts = o.object.get("ns").?.array.items.len;
            for (o.object.get("ns").?.array.items, 0..) |x, i| ns[i] = @intCast(x.integer);
        } else ns[0] = if (std.mem.eql(u8, op, "kv_b")) int(o, "heads") * (int(o, "qk") + int(o, "v")) else int(o, "n");
        var n: usize = 0;
        for (ns[0..parts]) |x| n += x;
        const wb = n * k / 2;
        const sb = n * (k / 64) * 2;
        try c.group(wb + 2 * sb);
        if (real) |r| try r.fill(c, o, wb, sb) else {
            var at: [3]usize = .{ 0, 0, 0 }; // each part's rows land after the earlier parts' in every component
            for (ns[0..parts]) |x| {
                d.words(c.stored[at[0]..][0 .. x * k / 2]);
                d.halves(c.stored[wb + at[1] ..][0 .. x * (k / 64) * 2]);
                d.halves(c.stored[wb + sb + at[2] ..][0 .. x * (k / 64) * 2]);
                at[0] += x * k / 2;
                at[1] += x * (k / 64) * 2;
                at[2] += x * (k / 64) * 2;
            }
        }
        if (std.mem.eql(u8, op, "q4")) return c.q4("", 0, wb, wb + sb, n, k, .{});
        if (std.mem.eql(u8, op, "stack")) {
            for (o.object.get("ranges").?.array.items, 0..) |rg, i| {
                const lo: usize = @intCast(rg.array.items[0].integer);
                const hi: usize = @intCast(rg.array.items[1].integer);
                var first: usize = 0;
                for (ns[0..lo]) |x| first += x;
                var rows: usize = 0;
                for (ns[lo..hi]) |x| rows += x;
                try c.q4(try std.fmt.allocPrint(c.a, "r{d}.", .{i}), 0, wb, wb + sb, rows, k, .{ .first = first });
            }
            return;
        }
        const H = int(o, "heads");
        const qk = int(o, "qk");
        const v = int(o, "v");
        const keys: pack.Rows = .{ .block = qk, .stride = qk + v };
        const vals: pack.Rows = .{ .first = qk, .block = v, .stride = qk + v };
        try c.q4("kv_k.", 0, wb, wb + sb, H * qk, k, keys);
        try c.q4("kv_v.", 0, wb, wb + sb, H * v, k, vals);
        for ([_][]const u8{ "k", "v" }, [_]pack.Rows{ keys, vals }, [_]usize{ qk, v }) |kv, r, dim| {
            const src = [_]u64{ 0, wb, wb + sb };
            const width = [_]usize{ k / 8 * 4, k / 64 * 2, k / 64 * 2 };
            for ([_][]const u8{ "w", "s", "b" }, src, width, 0..) |part, off, bytes, j| {
                const buf = if (j == 0) Buf.of(.i32, &.{ H, dim, k / 8 }) else Buf.of(.bf16, &.{ H, dim, k / 64 });
                try c.add(try std.fmt.allocPrint(c.a, "absorb.w{s}{s}", .{ kv, part }), buf, .{ .rows = .{ .src = .{ .g = 0, .off = off }, .n = H * dim, .bytes = bytes, .rows = r } });
            }
        }
        return;
    }
    if (std.mem.eql(u8, op, "conv")) {
        const w = int(o, "w");
        const taps = int(o, "taps");
        try c.group(3 * w * taps * 2);
        if (real) |r| try r.fill(c, o, 0, 0) else d.halves(c.stored);
        return c.add("conv", Buf.of(.bf16, &.{ 3 * w, taps }), .{ .copy = .{ .g = 0, .off = 0 } });
    }
    if (std.mem.eql(u8, op, "f32")) {
        const n = int(o, "n");
        try c.group(4 * n);
        if (real) |r| try r.fill(c, o, 0, 0) else d.words(c.stored);
        return c.add("f32", Buf.of(.f32, &.{n}), .{ .copy = .{ .g = 0, .off = 0 } });
    }
    return error.UnknownCase;
}

/// Each output's sha256 against python's (shape, dtype too), the emulated kernels against the reference: the count.
fn check(gpa: std.mem.Allocator, o: J, real: ?*Real) !usize {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var c: Case = .{ .a = arena.allocator() };
    try build(&c, o, real);
    const name = o.object.get("name").?.string;
    const want = o.object.get("out").?.object;
    if (want.count() != c.items.items.len) {
        std.debug.print("{s}: python has {d} outputs, the pack {d}\n", .{ name, want.count(), c.items.items.len });
        return error.OutputCount;
    }
    var m = try glm.model.Model.from(&glm.Config{}, .{}, 2, 0);
    var f = pack.hostFwd(&m);
    const stored = [_]u64{@intFromPtr(c.stored.ptr)};
    const host = [_][]const u8{c.stored};
    for (c.items.items, c.outs.items) |*it, out| {
        try pack.make(&f, .{ .host = emu }, it, &stored);
        var r: []align(16) u8 = out; // EXL3's ops (quantize4, cat): no host reference, the emulated bytes against python's
        if (it.x == null) {
            r = try c.a.alignedAlloc(u8, .@"16", out.len);
            ref.item(r, it, &host);
        }
        if (!std.mem.eql(u8, r, out)) {
            std.debug.print("{s}.{s}: the emulated glm_pack.cu differs from the host reference\n", .{ name, it.path });
            return error.KernelNotReference;
        }
        const w = want.get(it.path) orelse {
            std.debug.print("{s}: python has no output {s}\n", .{ name, it.path });
            return error.MissingOutput;
        };
        var sha: [32]u8 = undefined;
        Sha256.hash(r, &sha, .{});
        const hex = std.fmt.bytesToHex(sha, .lower);
        const shape = w.object.get("shape").?.array.items;
        var same = shape.len == it.buf.dims and std.mem.eql(u8, w.object.get("dtype").?.string, pack.dtypeName(it.buf.dtype));
        for (shape, 0..) |s, i| same = same and i < it.buf.dims and @as(usize, @intCast(s.integer)) == it.buf.shape[i];
        if (!same or !std.mem.eql(u8, &hex, w.object.get("sha256").?.string)) {
            std.debug.print("{s}.{s}: sha256 {s} (python {s}), shape {any} {t}\n", .{ name, it.path, &hex, w.object.get("sha256").?.string, it.buf.shape[0..it.buf.dims], it.buf.dtype });
            return error.NotPython;
        }
    }
    return c.items.items.len;
}

/// TF_GLM_PACK_REAL's cases: each component of the case read from a checkpoint folder by tensor name and rows.
const Real = struct {
    gpa: std.mem.Allocator,
    ix: glm.index.Index,

    fn read(r: *Real, name: []const u8, lo: usize, rows: usize, out: []u8) !void {
        const s = r.ix.names.get(name) orelse return error.MissingTensor;
        const row = s.len / s.shape[0];
        if ((lo + rows) * row > s.len or out.len != rows * row) return error.BadSlice;
        const fd = std.c.open(r.ix.shards[s.shard], .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd);
        var done: usize = 0;
        while (done < out.len) {
            const n = std.c.pread(fd, out.ptr + done, out.len - done, @intCast(s.off + lo * row + done));
            if (n <= 0) return error.ShortRead;
            done += @intCast(n);
        }
    }

    /// The case's "src": [[name, first row, rows], ...] per component (w, s, b for triplets; parts in order).
    fn fill(r: *Real, c: *Case, o: J, wb: usize, sb: usize) !void {
        const src = o.object.get("src").?.array.items;
        var at: [3]usize = .{ 0, wb, wb + sb };
        for (src) |part| {
            const p = part.array.items;
            const comp: usize = @intCast(p[3].integer); // 0 words (or the plain tensor), 1 scales, 2 biases
            const rows: usize = @intCast(p[2].integer);
            const s = r.ix.names.get(p[0].string) orelse return error.MissingTensor;
            const len = rows * (s.len / s.shape[0]);
            try r.read(p[0].string, @intCast(p[1].integer), rows, c.stored[at[comp]..][0..len]);
            at[comp] += len;
        }
    }
};

test "G-layout: python-0.6's packing of random tensors, from the host reference and glm_pack.cu emulated" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, fixture, .{});
    defer parsed.deinit();
    var outputs: usize = 0;
    for (parsed.value.object.get("cases").?.array.items) |o| outputs += try check(gpa, o, null);
    try std.testing.expect(outputs >= 30);
}

test "G-layout on real slices (TF_GLM_PACK_REAL: the cases' JSON, its \"dir\" a checkpoint folder)" {
    const gpa = std.testing.allocator;
    const path = std.c.getenv("TF_GLM_PACK_REAL") orelse return error.SkipZigTest;
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    const text = try gpa.alloc(u8, 4 << 20);
    defer gpa.free(text);
    const n = std.c.read(fd, text.ptr, text.len);
    if (n <= 0) return error.ShortRead;
    const parsed = try std.json.parseFromSlice(J, gpa, text[0..@intCast(n)], .{});
    defer parsed.deinit();
    var r: Real = .{ .gpa = gpa, .ix = try glm.index.open(gpa, parsed.value.object.get("dir").?.string) };
    defer r.ix.deinit();
    var outputs: usize = 0;
    for (parsed.value.object.get("cases").?.array.items) |o| outputs += try check(gpa, o, &r);
    std.debug.print("G-layout real slices: {d} cases, {d} outputs equal to python-0.6's\n", .{ parsed.value.object.get("cases").?.array.items.len, outputs });
}

/// python-0.6's experts.pack (pack_kernel<2>, emulated) of E experts' [n, k] triplets at `w`, `s`, `b`: [E][n/32][k/64][288].
fn pythonPack(a: std.mem.Allocator, w: []u8, s: []u8, b: []u8, e: usize, n: usize, k: usize) ![]u32 {
    const out = try a.alloc(u32, e * (n / 32) * (k / 64) * 288);
    var args: cuda.Args = .{};
    for ([_][]u8{ w, s, b }) |x| args.add(@intFromPtr(x.ptr));
    args.add(@intFromPtr(out.ptr));
    for ([_]usize{ n, k / 8, k / 64, n / 32 }) |v| args.add(@as(c_int, @intCast(v)));
    try emuKernel(4, .{ @intCast(k / 64), @intCast(n / 32), @intCast(e) }, .{ 288, 1, 1 }, &args);
    return out;
}

test "experts: glm_pack.cu and the host reference equal python-0.6's experts_pack.cu stacked as experts.make" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const E = 3;
    const ni = 64; // an expert's gate and up rows (TP2: its half); down's inputs
    const dm = 128; // the model width: gate and up's inputs, down's rows
    var d: ref.Draw = .{ .s = 77 };
    // stored as the split leaves them: routed gate, up, down [E][rows][...] each w, s, b; shared gate|up and down
    const shapes = [_][2]usize{ .{ ni, dm }, .{ ni, dm }, .{ dm, ni }, .{ 2 * ni, dm }, .{ dm, ni } };
    const counts = [_]usize{ E, E, E, 1, 1 };
    var trips: [5][3][]u8 = undefined;
    var c: Case = .{ .a = a };
    var total: usize = 0;
    for (shapes, counts) |sh, cnt| total += cnt * (sh[0] * sh[1] / 2 + 2 * sh[0] * (sh[1] / 64) * 2);
    try c.group(total);
    var at: usize = 0;
    var refs: [5][3]glm.split.Ref = undefined;
    for (shapes, counts, 0..) |sh, cnt, i| for (0..3) |j| {
        const len = cnt * if (j == 0) sh[0] * sh[1] / 2 else sh[0] * (sh[1] / 64) * 2;
        trips[i][j] = c.stored[at..][0..len];
        refs[i][j] = .{ .g = 0, .off = at };
        if (j == 0) d.words(trips[i][j]) else d.halves(trips[i][j]);
        at += len;
    };
    const T = pack.Trip;
    const half_w = ni * dm / 2;
    const half_s = ni * (dm / 64) * 2;
    const sg: T = .{ .w = refs[3][0], .s = refs[3][1], .b = refs[3][2] };
    const su: T = .{ .w = .{ .g = 0, .off = refs[3][0].off + half_w }, .s = .{ .g = 0, .off = refs[3][1].off + half_s }, .b = .{ .g = 0, .off = refs[3][2].off + half_s } };
    const sd: T = .{ .w = refs[4][0], .s = refs[4][1], .b = refs[4][2] };
    const rt = [3]T{ .{ .w = refs[0][0], .s = refs[0][1], .b = refs[0][2] }, .{ .w = refs[1][0], .s = refs[1][1], .b = refs[1][2] }, .{ .w = refs[2][0], .s = refs[2][1], .b = refs[2][2] } };
    try c.add("up", Buf.of(.i32, &.{ E + 1, ni / 32, dm / 64, 2, 288 }), .{ .experts = .{ .routed = .{ rt[0], rt[1] }, .shared = .{ sg, su }, .mm = 2, .e = E, .n = ni, .k = dm } });
    try c.add("down", Buf.of(.i32, &.{ E + 1, dm / 32, ni / 64, 1, 288 }), .{ .experts = .{ .routed = .{ rt[2], rt[2] }, .shared = .{ sd, sd }, .mm = 1, .e = E, .n = dm, .k = ni } });
    var m = try glm.model.Model.from(&glm.Config{}, .{}, 2, 0);
    var f = pack.hostFwd(&m);
    const stored = [_]u64{@intFromPtr(c.stored.ptr)};
    const host = [_][]const u8{c.stored};
    for (c.items.items, c.outs.items, 0..) |*it, out, which| {
        try pack.make(&f, .{ .host = emu }, it, &stored);
        const r = try a.alignedAlloc(u8, .@"16", out.len);
        ref.item(r, it, &host);
        try std.testing.expectEqualSlices(u8, r, out);
        // python: torch.stack([pack(gate), pack(up)], dim=3) (or pack(down).unsqueeze(3)) of the routed experts then the shared one
        const x = it.op.experts;
        const nb = x.n / 32;
        const kg = x.k / 64;
        const blocks = std.mem.bytesAsSlice(u32, out);
        for (0..x.mm) |mi| for ([_]T{ x.routed[mi], x.shared[mi] }, [_]usize{ E, 1 }, [_]usize{ 0, E }) |t, cnt, first| {
            const wl = cnt * x.n * x.k / 2;
            const sl = cnt * x.n * (x.k / 64) * 2;
            const py = try pythonPack(a, c.stored[t.w.off..][0..wl], c.stored[t.s.off..][0..sl], c.stored[t.b.off..][0..sl], cnt, x.n, x.k);
            for (0..cnt) |e| for (0..nb) |bb| for (0..kg) |g| {
                const got = blocks[((((first + e) * nb + bb) * kg + g) * x.mm + mi) * 288 ..][0..288];
                try std.testing.expectEqualSlices(u32, py[((e * nb + bb) * kg + g) * 288 ..][0..288], got);
            };
        };
        _ = which;
    }
}
