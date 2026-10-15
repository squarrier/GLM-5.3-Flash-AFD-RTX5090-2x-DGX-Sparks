//! python-0.6's nucleus_rows (cuda/sampling.py, ed78d6f) for a TP rank: rows with temperature > 0 and top_k 0, MTP
//! verify windows included. On the GPU, with Ash Hart's torch-op kernels (zig/kernels/cuda/torch_ops): f64 scaling, each
//! rank's row maxima (argmax_f64), the ranks' maxima through one gather, int64 masses and the shard sums (nucleus.cu);
//! then a level's top candidates by topk.cu (never a host sort of a shard), their ids (indexing.cu's lookup), values and
//! masses (its select), packed beside the sums and gathered once. On the host, nucleus_draw.zig with Mia's union cover
//! (patch 0034): NUCLEUS candidates, then WIDER, then whole shards. Each call is recorded (a CPU trace) or launched.
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const cc = @import("cuda_compute.zig");
const sample = @import("cuda_sample.zig");
const nd = @import("nucleus_draw.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;
const Val = trace.Val;
const Buf = views.Buf;
const go = sample.go;
const w64 = sample.w64;

pub const Sampling = nd.Sampling;
pub const Stats = nd.Stats;

/// Rows a call draws at most (a verify window: 1 + drafts).
pub const max_rows = 16;

/// nucleus_rows' rows on the device for windows of up to `rows` rows of a rank's V/world columns (f64 in .i64
/// buffers): scaled logits and masses, argmax columns, the rank's maxima, every rank's (gathered, then transposed) and
/// the ranks', topk's outputs at WIDER, the packet sent and every rank's gathered, and their pinned host copy.
pub const Scratch = struct {
    scaled: Buf,
    mass: Buf,
    col: Buf,
    lmax: Buf,
    maxima: Buf,
    flip: Buf,
    top: Buf,
    values: Buf,
    indices: Buf,
    send: Buf,
    recv: Buf,
    host: Buf,

    pub fn init(m: *const md.Model, rows: usize) Scratch {
        const V = m.vl();
        const k = @min(nd.wider, V);
        const words = nd.Packet.stride(rows, V);
        return .{
            .scaled = Buf.of(.i64, &.{ rows, V }),
            .mass = Buf.of(.i64, &.{ rows, V }),
            .col = Buf.of(.i64, &.{rows}),
            .lmax = Buf.of(.i64, &.{rows}),
            .maxima = Buf.of(.i64, &.{ m.world, rows }),
            .flip = Buf.of(.i64, &.{ rows, m.world }),
            .top = Buf.of(.i64, &.{rows}),
            .values = Buf.of(.f32, &.{ rows, k }),
            .indices = Buf.of(.i64, &.{ rows, k }),
            .send = Buf.of(.i64, &.{words}),
            .recv = Buf.of(.i64, &.{ m.world, words }),
            .host = .{ .dtype = .i64, .dims = 2, .shape = .{ m.world, words, 1, 1, 1 }, .host = true },
        };
    }
};

/// The kernels beyond the sampler's: nucleus.cu and argmax_f64.cu (images torch_nucleus, torch_argmax_f64), and
/// indexing.cu's select from the sampler's own module.
pub const Kernels = struct {
    mods: [2]cuda.Module,
    scale: cuda.Function,
    mass: cuda.Function,
    sum: cuda.Function,
    argmax: cuda.Function,
    select: cuda.Function,
    /// draft_policy (TF_GLM_ROWS_1024=1): the row maxima, masses and integer sums on 1024 threads a row; null: 256, as before
    wide: ?Wide = null,

    pub const Wide = struct { argmax: cuda.Function, mass: cuda.Function, sum: cuda.Function };

    pub fn load(d: *const cuda.Driver, base: *const sample.Kernels) !Kernels {
        return loadImages(d, base, .{ cuda.kernels.torch_nucleus, cuda.kernels.torch_argmax_f64 });
    }

    pub fn loadImages(d: *const cuda.Driver, base: *const sample.Kernels, images: [2][]const u8) !Kernels {
        if (images[0].len == 0 or images[1].len == 0) return error.NotFound; // a host-only build embeds no images
        const wide = try rows1024(); // a malformed switch is refused before any module loads
        var mods: [2]cuda.Module = undefined;
        mods[0] = try cuda.Module.load(d, images[0]);
        errdefer mods[0].unload();
        mods[1] = try cuda.Module.load(d, images[1]);
        errdefer mods[1].unload();
        return .{
            .mods = mods,
            .scale = try mods[0].function("tf_scale_logits_kernel"),
            .mass = try mods[0].function("tf_mass_kernel"),
            .sum = try mods[0].function("tf_integer_row_sum_kernel"),
            .argmax = try mods[1].function("tf_argmax_f64_kernel"),
            .select = try base.mods[2].function("tf_select_axis_kernel"),
            // looked up only when on: images without them (another tree's fatbins) still load with the switch unset
            .wide = if (wide) .{ .argmax = try mods[1].function("tf_argmax_f64_wide_kernel"), .mass = try mods[0].function("tf_mass_wide_kernel"), .sum = try mods[0].function("tf_integer_row_sum_wide_kernel") } else null,
        };
    }

    pub fn unload(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
    }

    /// The row kernels' threads: 1024 with the switch, else 256.
    fn rowThreads(k: *const Kernels) u32 {
        return if (k.wide != null) 1024 else 256;
    }
};

/// TF_GLM_ROWS_1024's value (draft_policy; BadAd84's patches/0010 for Flash Next, MiaAI-Lab qwen-next-single PR #35): unset,
/// empty or 0 off, 1 on (the nucleus draw's row maxima, masses and integer sums on 1024 threads a row, byte-identical);
/// anything else error.BadRows1024.
pub fn parseRows1024(v: []const u8) !bool {
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadRows1024;
}

pub fn rows1024() !bool {
    const v = std.mem.span(std.c.getenv("TF_GLM_ROWS_1024") orelse return false);
    return parseRows1024(v) catch |e| {
        std.log.err("TF_GLM_ROWS_1024 is 1 (the nucleus row kernels on 1024 threads), 0 or unset, not {s}", .{v});
        return e;
    };
}

test "TF_GLM_ROWS_1024: unset, empty and 0 are off, 1 on, anything else refused" {
    try std.testing.expect(!try parseRows1024(""));
    try std.testing.expect(!try parseRows1024("0"));
    try std.testing.expect(try parseRows1024("1"));
    for ([_][]const u8{ "2", "on", "true", "1024", " 1" }) |v| try std.testing.expectError(error.BadRows1024, parseRows1024(v));
}

/// A rank's nucleus sampler: its device rows, kernels (a GPU's), the host's candidate rows, and the ranks' shards.
pub const Nucleus = struct {
    s: Scratch,
    k: ?*const Kernels = null,
    work: []nd.Cand,
    shards: nd.Shards,

    pub fn init(gpa: std.mem.Allocator, m: *const md.Model, rows: usize) !Nucleus {
        if (m.world > nd.max_world or rows > max_rows) return error.SampleShape;
        return .{ .s = Scratch.init(m, rows), .work = try gpa.alloc(nd.Cand, m.world * m.vl()), .shards = nd.Shards.uniform(m.world, m.vl()) };
    }

    pub fn deinit(n: *Nucleus, gpa: std.mem.Allocator) void {
        gpa.free(n.work);
    }
};

fn kernels(n: *const Nucleus) !*const Kernels {
    return n.k orelse error.NoNucleusKernels;
}

fn ext(f: *Fwd, name: []const u8, vals: []const Val) !void {
    try f.rec.?.call(.ext, name, vals);
}

fn f64T(b: Buf, at: usize, shape: []const usize) T {
    return T.of(b, at, shape); // f64 rows in an .i64 buffer: 8 bytes an element, as the kernels read them
}

/// `x.double() / t`: the CPU scalar's reciprocal, then one f64 product an element (ATen's CUDA division).
fn scale(f: *Fwd, n: *const Nucleus, x: T, out: T, temperature: f64) !void {
    const inv = 1.0 / @max(temperature, 1e-6);
    if (f.rec != null) return ext(f, "nucleus.scale", &.{ .{ .t = x }, .{ .int = @bitCast(@as(u64, @bitCast(inv))) }, .{ .t = out } });
    const count = cc.bytes(out) / 8;
    try go(f, (try kernels(n)).scale, .{ sample.blocks(count, 256), 1 }, 256, .{ x.ptr, out.ptr, w64(count), inv });
}

/// `x.max(dim=-1).values` of f64 rows [R, C] (row stride `stride`): argmax_f64's column, then its value.
fn rowMax(f: *Fwd, smp: *const sample.Sampler, n: *const Nucleus, x: T, stride: usize, out: T) !void {
    const R = x.shape[0];
    const C = x.shape[1];
    const col = T.of(n.s.col, 0, &.{R});
    if (f.rec != null) return ext(f, "nucleus.max", &.{ .{ .t = x }, .{ .t = col }, .{ .t = out } });
    const k = try kernels(n);
    if (k.wide) |w| try go(f, w.argmax, .{ R, 1 }, k.rowThreads(), .{ x.ptr, col.ptr, w64(R), w64(C), w64(stride) }) else try go(f, k.argmax, .{ R, 1 }, 256, .{ x.ptr, col.ptr, w64(R), w64(C), w64(stride) });
    try go(f, k.select, .{ R, 1 }, 32, .{ x.ptr, out.ptr, col.ptr, w64(R), w64(C), w64(1), w64(8), w64(1), smp.s.bad.ptr });
}

/// `scaled.gather(1, cols)` and `mass.gather(1, cols)`: [R, V] 8-byte rows at topk's columns [R, c] into `out`.
/// (select copies an element a block: 32 threads for its 8 bytes)
fn take(f: *Fwd, smp: *const sample.Sampler, n: *const Nucleus, x: T, cols: T, out: T) !void {
    if (f.rec != null) return ext(f, "nucleus.take", &.{ .{ .t = x }, .{ .t = cols }, .{ .t = out } });
    const R = cols.shape[0];
    const c = cols.shape[1];
    try go(f, (try kernels(n)).select, .{ R * c, 1 }, 32, .{ x.ptr, out.ptr, cols.ptr, w64(R), w64(x.shape[1]), w64(c), w64(8), w64(c), smp.s.bad.ptr });
}

/// `w.comm.all_gather` of a packet's 64-bit words (as fp32 pairs); one rank: nothing to gather.
fn gatherWords(f: *Fwd, send: T, recv: T) !void {
    var a = send;
    var b = recv;
    a.dtype = .f32;
    b.dtype = .f32;
    a.shape[0] *= 2;
    b.shape[0] *= 2;
    try f.allGather(a, b);
}

/// The call's first half (python's up to `_shares`): `logits.float().double() / t`, every rank's maxima through the
/// gather and the ranks' maximum a row, `floor(exp(scaled - top) * MASS)` and the shard's sums into the packet.
/// structured_users: `widened` works on the widened fp32 rows before the temperature (a grammar mask's -inf); null: none.
fn prefix(f: *Fwd, smp: *const sample.Sampler, n: *const Nucleus, logits: T, temperature: f64, widened: ?Widened) !void {
    const m = f.m;
    const R = logits.shape[0];
    const V = logits.shape[1];
    const s = &n.s;
    const x = T.of(smp.s.float, 0, &.{ R, V });
    try sample.toFloat(f, smp, logits, x);
    if (widened) |w| try w.apply(w.ctx, f, x);
    const scaled = f64T(s.scaled, 0, &.{ R, V });
    try scale(f, n, x, scaled, temperature);
    const lmax = f64T(s.lmax, 0, &.{R});
    try rowMax(f, smp, n, scaled, V, lmax);
    var top = lmax;
    if (m.world > 1) {
        const all = f64T(s.maxima, 0, &.{ m.world, R });
        try gatherWords(f, lmax, T.of(s.maxima, 0, &.{m.world * R}));
        var rows = f64T(s.maxima, 0, &.{ R, m.world }); // one row: [world, 1] is [1, world]
        if (R > 1) { // [world, R] to [R, world]
            rows = f64T(s.flip, 0, &.{ R, m.world });
            if (f.rec != null) try ext(f, "nucleus.flip", &.{ .{ .t = all }, .{ .t = rows } }) else {
                const sm = try sample.kernels(smp);
                try go(f, sm.strided, .{ m.world, 1 }, 256, .{ all.ptr, rows.ptr, w64(m.world), w64(R), w64(8), w64(R * 8), w64(8), w64(8), w64(m.world * 8) });
            }
        }
        top = f64T(s.top, 0, &.{R});
        try rowMax(f, smp, n, rows, m.world, top);
    }
    const mass = T.of(s.mass, 0, &.{ R, V });
    const sums = T.of(s.send, 8, &.{R});
    if (f.rec != null) {
        try ext(f, "nucleus.mass", &.{ .{ .t = scaled }, .{ .t = top }, .{ .t = mass } });
        return ext(f, "nucleus.sum", &.{ .{ .t = mass }, .{ .t = sums } });
    }
    const k = try kernels(n);
    if (k.wide) |w| {
        try go(f, w.mass, .{ R, 1 }, k.rowThreads(), .{ scaled.ptr, top.ptr, mass.ptr, w64(R), w64(V) });
        return go(f, w.sum, .{ R, 1 }, k.rowThreads(), .{ mass.ptr, sums.ptr, w64(R), w64(V) });
    }
    try go(f, k.mass, .{ R, 1 }, 256, .{ scaled.ptr, top.ptr, mass.ptr, w64(R), w64(V) });
    try go(f, k.sum, .{ R, 1 }, 256, .{ mass.ptr, sums.ptr, w64(R), w64(V) });
}

/// nucleus_draw.levels' source: one level's packet ([header, sums, values, ids, masses]) gathered from every rank and
/// copied to the host; `again` copies the last one again once the device is idle (0092).
const Source = struct {
    f: *Fwd,
    smp: *const sample.Sampler,
    n: *const Nucleus,
    rows: usize,
    words: usize = 0,
    count: usize = 0,

    pub fn gather(src: *Source, count: usize) !nd.Packet {
        const f = src.f;
        const m = f.m;
        const s = &src.n.s;
        const R = src.rows;
        const V = m.vl();
        const c = @min(count, V);
        const at = [3]usize{ (1 + R) * 8, (1 + R + R * c) * 8, (1 + R + 2 * R * c) * 8 }; // values, ids, masses
        const scaled = f64T(s.scaled, 0, &.{ R, V });
        const mass = T.of(s.mass, 0, &.{ R, V });
        if (c < V) { // torch.topk(scaled, c): the f32 rows' top c (x * (1/t) keeps their order and ties)
            const cols = T.of(s.indices, 0, &.{ R, c });
            try sample.topk(f, src.smp, T.of(src.smp.s.float, 0, &.{ R, V }), c, T.of(s.values, 0, &.{ R, c }), cols);
            try sample.add(f, src.smp, cols, m.rank * V, T.of(s.send, at[1], &.{ R, c }));
            try take(f, src.smp, src.n, scaled, cols, f64T(s.send, at[0], &.{ R, c }));
            try take(f, src.smp, src.n, mass, cols, T.of(s.send, at[2], &.{ R, c }));
        } else { // whole shards: every column, its id from the rank's table
            try cc.copyDtoD(f, f64T(s.send, at[0], &.{ R, V }), scaled);
            for (0..R) |r| try cc.copyDtoD(f, T.of(s.send, at[1] + r * V * 8, &.{V}), T.whole(src.smp.s.vocab));
            try cc.copyDtoD(f, T.of(s.send, at[2], &.{ R, V }), mass);
        }
        try header(f, src.smp, T.of(s.send, 0, &.{1}));
        src.count = c;
        src.words = nd.Packet.stride(R, c);
        if (m.world > 1) try gatherWords(f, T.of(s.send, 0, &.{src.words}), T.of(s.recv, 0, &.{m.world * src.words}));
        return src.copy();
    }

    /// The gathered packets' `.cpu()` (one rank: its own packet; recording: no rows).
    fn copy(src: *Source) !nd.Packet {
        const f = src.f;
        const s = &src.n.s;
        const n = f.m.world * src.words;
        const from = if (f.m.world > 1) T.of(s.recv, 0, &.{n}) else T.of(s.send, 0, &.{n});
        if (s.host.ptr == 0) return error.NoHostRows;
        const host: [*]u8 = @ptrFromInt(s.host.ptr);
        try cc.toHost(f, from, host[0 .. n * 8]);
        if (f.rec != null) return .{ .words = &.{}, .world = f.m.world, .rows = src.rows, .count = src.count };
        const words: [*]const u64 = @ptrCast(@alignCast(host));
        return .{ .words = words[0..n], .world = f.m.world, .rows = src.rows, .count = src.count };
    }

    pub fn again(src: *Source) !nd.Packet {
        std.debug.print("[tensorfold] nucleus: a gathered id outside its rank's vocabulary; copying the rows again once the device is idle\n", .{});
        try sample.deviceSync(src.f);
        return src.copy();
    }
};

/// The packet's header: the kernels' bad-index word (lookup's and select's), zero-extended; the word then cleared.
fn header(f: *Fwd, smp: *const sample.Sampler, out: T) !void {
    if (f.rec != null) return ext(f, "nucleus.header", &.{ .{ .t = T.whole(smp.s.bad) }, .{ .t = out } });
    const g = f.gpu orelse return error.NoGpu;
    const d = g.stream.d;
    const h = g.stream.handle;
    try d.check(d.api.cuMemsetD32Async(out.ptr, 0, 2, h), "cuMemsetD32Async");
    try d.check(d.api.cuMemcpyDtoDAsync_v2(out.ptr, smp.s.bad.ptr, 4, h), "cuMemcpyDtoDAsync");
    try d.check(d.api.cuMemsetD32Async(smp.s.bad.ptr, 0, 1, h), "cuMemsetD32Async");
}

/// nucleus_rows(logits, positions, sampling, offset=the rank's, gather=w.comm's) on `logits` [R, V/world] bf16: each
/// row's token into `out` (and its share of the mass into `shares`, python's `probs`), the same on every rank; `stats`
/// counts the gathers. Recording makes the first level's launches only, and the trace's picker answers (zeros
/// without one, as the greedy path's).
pub fn nucleusRows(f: *Fwd, smp: *const sample.Sampler, n: *const Nucleus, logits: T, positions: []const u64, s: Sampling, out: []u32, shares: ?[]f64, stats: *Stats) !void {
    return nucleusRowsWith(f, smp, n, logits, positions, s, out, shares, stats, null);
}

/// structured_users: work a caller does on the widened fp32 rows [R, V] (Scratch.float) after `logits.float()` and before the
/// temperature, the maxima, the masses and the candidates (the masked adapter's grammar_mask: a clear bit's column
/// -inf, so it has no mass and is never drawn).
pub const Widened = struct { ctx: *anyopaque, apply: *const fn (ctx: *anyopaque, f: *Fwd, rows: T) anyerror!void };

/// nucleusRows with `widened` between the widening and the temperature (null: nucleusRows itself, the same launches).
pub fn nucleusRowsWith(f: *Fwd, smp: *const sample.Sampler, n: *const Nucleus, logits: T, positions: []const u64, s: Sampling, out: []u32, shares: ?[]f64, stats: *Stats, widened: ?Widened) !void {
    const R = logits.shape[0];
    const V = logits.shape[1];
    if (R == 0 or R > n.s.top.shape[0] or V != f.m.vl() or positions.len != R or out.len < R) return error.SampleShape;
    if (shares) |p| if (p.len < R) return error.SampleShape;
    if (!(s.temperature > 0.0)) return error.NotNucleus;
    try prefix(f, smp, n, logits, s.temperature, widened);
    var src: Source = .{ .f = f, .smp = smp, .n = n, .rows = R };
    if (f.rec) |t| {
        _ = try src.gather(nd.nucleus);
        return t.picks(logits, positions, out[0..R], if (shares) |p| p[0..R] else null);
    }
    var drawn: [max_rows]nd.Drawn = undefined;
    const g = try nd.levels(&src, n.shards, positions, s, n.work, drawn[0..R]);
    stats.add(g, R);
    for (drawn[0..R], 0..) |d, i| out[i] = d.token;
    if (shares) |p| for (drawn[0..R], 0..) |d, i| {
        p[i] = d.share;
    };
}
