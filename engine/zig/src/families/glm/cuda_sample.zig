//! python-0.6's greedy sample_rows (glm5_next/cuda/decode.py, ed78d6f) for a TP rank: torch.topk(logits.float(), 1) on
//! the rank's vocabulary columns, the ids moved to token ids (+ vocab_offset) and packed beside their values, every
//! rank's (value, id) pairs gathered, then the host's np.lexsort((tokens, -values)): the largest value, ties to the
//! smallest id. Each torch op is a call of *Fwd: recorded as python's `torch.<op>` on python's temporaries (`tmp.<op>`:
//! this file's Scratch), or launched with the torch-op kernels Ash Hart wrote for Nemotron (zig/kernels/cuda/torch_ops).
//! On the host, topkOrder lists a row's top k in torch.topk's CUDA order (tf-glm-forward's teacher top 5).
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const cc = @import("cuda_compute.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;
const Tmp = @import("cuda_tmp.zig").Tmp;

const T = trace.T;
const Buf = views.Buf;

/// topk.cu's column tile and threads (Nemotron's cuda_torch_ops.zig)
const topk_tile = 4096;
const topk_threads = 256;

/// exact_sampling.MARGIN: candidates read past top_k, so values tied at the cut resolve by id on the host
pub const margin = 8;
/// sample_rows' candidates a rank for a greedy draft's probability (its 20 competitors and MARGIN)
pub const draft_k = 20 + margin;
/// the gathered candidates a row's host choice reads at most (every rank's draft_k)
const max_candidates = 256;

fn tiles(columns: usize) usize {
    return (columns + topk_tile - 1) / topk_tile;
}

/// launch_shape.h's tf_launch_blocks: a block a `threads` elements, at most 65535.
pub fn blocks(count: usize, threads: usize) usize {
    return @min((count + threads - 1) / threads, 65535);
}

/// sample_rows' temporaries for windows of up to `rows` rows and up to draft_k candidates a row, each named by the
/// torch op that makes it (python's `tmp.<op>`); `cpu` is pinned host memory; `work`, `vocab` and `bad` are the
/// kernels' own (topk.cu's workspace, the rank's token ids that turn columns into ids, lookup's error word).
pub const Scratch = struct {
    float: Buf,
    topk: struct { values: Buf, indices: Buf },
    add: Buf,
    to: Buf,
    cat: Buf,
    empty: Buf,
    cpu: Buf,
    work: Buf,
    vocab: Buf,
    bad: Buf,

    pub fn init(m: *const md.Model, rows: usize) Scratch {
        const V = m.vl();
        const k: usize = @min(V, draft_k);
        const pairs = m.world * rows * 2 * k;
        return .{
            .float = Buf.of(.f32, &.{ rows, V }),
            .topk = .{ .values = Buf.of(.f32, &.{ rows, k }), .indices = Buf.of(.i64, &.{ rows, k }) },
            .add = Buf.of(.i64, &.{ rows, k }),
            .to = Buf.of(.i32, &.{ rows, k }),
            .cat = Buf.of(.f32, &.{ rows, 2 * k }),
            .empty = Buf.of(.f32, &.{pairs}),
            .cpu = .{ .dtype = .f32, .dims = 1, .shape = .{ pairs, 1, 1, 1, 1 }, .host = true },
            .work = Buf.of(.i32, &.{rows * (tiles(V) * 260 + 3)}),
            .vocab = Buf.of(.i64, &.{V}),
            .bad = Buf.of(.i32, &.{1}),
        };
    }
};

/// The torch-op kernels the sampler launches, from the runtime's images (torch_pointwise, _topk, _indexing, _movement).
pub const Kernels = struct {
    mods: [4]cuda.Module,
    to_f32: cuda.Function,
    hist: cuda.Function,
    digit: cuda.Function,
    count: cuda.Function,
    prefix: cuda.Function,
    compact: cuda.Function,
    lookup: cuda.Function,
    narrow: cuda.Function,
    strided: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        const k = cuda.kernels;
        return loadImages(d, .{ k.torch_pointwise, k.torch_topk, k.torch_indexing, k.torch_movement });
    }

    /// From the images a tool embeds itself (tf-glm-forward: torch_pointwise, _topk, _indexing, _movement).
    pub fn loadImages(d: *const cuda.Driver, images: [4][]const u8) !Kernels {
        if (images[1].len == 0) return error.NotFound; // a host-only build embeds no images
        var mods: [4]cuda.Module = undefined;
        var n: usize = 0;
        errdefer for (mods[0..n]) |*x| x.unload();
        for (images) |image| {
            mods[n] = try cuda.Module.load(d, image);
            n += 1;
        }
        return .{
            .mods = mods,
            .to_f32 = try mods[0].function("tf_bf16_to_f32_kernel"),
            .hist = try mods[1].function("tf_topk_f32_histogram_kernel"),
            .digit = try mods[1].function("tf_topk_f32_choose_digit_kernel"),
            .count = try mods[1].function("tf_topk_f32_count_kernel"),
            .prefix = try mods[1].function("tf_topk_f32_prefix_kernel"),
            .compact = try mods[1].function("tf_topk_f32_compact_kernel"),
            .lookup = try mods[2].function("tf_lookup_ids_kernel"),
            .narrow = try mods[2].function("tf_narrow_indices_kernel"),
            .strided = try mods[3].function("tf_strided_copy_kernel"),
        };
    }

    pub fn unload(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
    }
};

/// The sampler: its scratch, on a GPU its kernels, and on a CPU trace the namer of python's temporaries (the scratch's
/// buffers renamed as python makes them; without one they keep the names a walk gave them).
pub const mask_cpu = @import("cuda_mask_cpu.zig");
/// grammar_mask: the real masked adapter (AFD attention, world 1): staged words and Ash Hart's grammar_mask before topk.
pub const mask_dev = @import("cuda_mask_dev.zig");
pub const Sampler = struct { s: Scratch, k: ?*const Kernels = null, tmp: ?*Tmp = null, masked_cpu: ?*mask_cpu.Cpu = null, masked_dev: ?*mask_dev.Dev = null };

pub fn maskedReady(f: *const Fwd, smp: *const Sampler) !void {
    if (smp.masked_dev) |d| {
        if (smp.masked_cpu != null) return error.IncompleteMaskedAdapter; // one adapter, never both
        return d.ready(f);
    }
    const c = smp.masked_cpu orelse return error.IncompleteMaskedAdapter;
    try c.check(f);
}

/// Before a masked step's forward: the device adapter refuses an empty or malformed mask with no launch.
pub fn maskedPrecheck(smp: *const Sampler, mask: []const u32) !void {
    if (smp.masked_dev) |d| try d.precheck(mask);
}

/// CPU-only doubles select before truncation; the device adapter (masked_dev) masks before topk on the GPU.
pub fn greedyMasked(f: *Fwd, smp: *Sampler, logits: T, out: []u32, mask: []const u32) !void {
    try maskedReady(f, smp);
    if (smp.masked_dev) |d| return d.greedy(f, smp, logits, out, mask);
    if (logits.dims != 2 or logits.shape[0] != 1 or logits.shape[1] != f.m.vl() or logits.stride[0] != f.m.vl() or logits.stride[1] != 1 or out.len != 1) return error.MaskShape;
    try mask_cpu.shape(f.m.vocab, mask);
    if (mask_cpu.allAllowed(f.m.vocab, mask)) {
        try greedy(f, smp, logits, out);
        if (smp.masked_cpu.?.endpoint != null) out[0] = try mask_cpu.complete(smp.masked_cpu.?, .{ .valid = true, .global_id = out[0] });
        return;
    }
    try made(smp, "float", &smp.s.float);
    try toFloat(f, smp, logits, T.of(smp.s.float, 0, &.{ 1, f.m.vl() }));
    out[0] = try mask_cpu.pick(smp.masked_cpu.?, f, mask);
}

pub fn made(smp: *Sampler, op: []const u8, b: *Buf) !void {
    if (smp.tmp) |t| try t.made(op, b);
}

/// Once before sampling on a GPU: the rank's token ids (vocab_offset + column) into Scratch.vocab, lookup's word cleared.
pub fn prepare(gpa: std.mem.Allocator, f: *Fwd, smp: *const Sampler) !void {
    if (f.rec != null) return; // setup, not a step's work
    const g = f.gpu orelse return error.NoGpu;
    const V = smp.s.vocab.shape[0];
    const ids = try gpa.alloc(i64, V);
    defer gpa.free(ids);
    for (ids, 0..) |*x, i| x.* = @intCast(f.m.rank * f.m.vl() + i);
    const d = g.stream.d;
    try d.check(d.api.cuMemcpyHtoDAsync_v2(smp.s.vocab.ptr, ids.ptr, V * 8, g.stream.handle), "cuMemcpyHtoDAsync");
    try d.check(d.api.cuMemsetD32Async(smp.s.bad.ptr, 0, 1, g.stream.handle), "cuMemsetD32Async");
    try g.stream.synchronize();
}

/// One launch on the rank's stream; `args` a tuple, each value of its kernel parameter's C type.
pub fn go(f: *Fwd, func: cuda.Function, grid: [2]usize, block: u32, args: anytype) !void {
    const g = f.gpu orelse return error.NoGpu;
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    try cuda.launch.launch(func, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]) }, .block = .{ .x = block } }, g.stream, &a);
}

pub fn kernels(smp: *const Sampler) !*const Kernels {
    return smp.k orelse error.NoSampleKernels;
}

pub fn w64(x: usize) u64 {
    return @intCast(x);
}

/// `logits.float()`: bf16 rows widened, exactly.
pub fn toFloat(f: *Fwd, smp: *const Sampler, x: T, out: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.float", &.{ .{ .t = x }, .{ .t = out } });
    const n = cc.bytes(out) / 4;
    try go(f, (try kernels(smp)).to_f32, .{ blocks(n, 256), 1 }, 256, .{ x.ptr, out.ptr, w64(n) });
}

/// `torch.topk(x, k, dim=-1)` of fp32 rows: values above the k-th in column order, then ties in column order.
pub fn topk(f: *Fwd, smp: *const Sampler, x: T, k: usize, values: T, indices: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.topk", &.{ .{ .t = x }, .{ .int = @intCast(k) }, .{ .int = -1 }, .{ .t = values }, .{ .t = indices } });
    const ks = try kernels(smp);
    const R = w64(x.shape[0]);
    const cols = w64(x.shape[1]);
    const n = w64(tiles(x.shape[1]));
    const hist = smp.s.work.ptr;
    const threshold = hist + R * n * 256 * 4;
    const remaining = threshold + R * 4;
    const greater = remaining + R * 4;
    const equal = greater + R * n * 4;
    const greater_prefix = equal + R * n * 4;
    const equal_prefix = greater_prefix + R * n * 4;
    const greater_total = equal_prefix + R * n * 4;
    const row_bytes = w64(x.stride[0] * 4);
    const kk: u32 = @intCast(k);
    var shift: u32 = 24;
    while (true) : (shift -= 8) {
        try go(f, ks.hist, .{ n, R }, topk_threads, .{ x.ptr, hist, threshold, cols, n, row_bytes, w64(4), shift });
        try go(f, ks.digit, .{ R, 1 }, topk_threads, .{ hist, threshold, remaining, n, kk, shift });
        if (shift == 0) break;
    }
    try go(f, ks.count, .{ n, R }, topk_threads, .{ x.ptr, threshold, greater, equal, cols, n, row_bytes, w64(4) });
    try go(f, ks.prefix, .{ R, 1 }, 1, .{ greater, equal, greater_prefix, equal_prefix, greater_total, n });
    try go(f, ks.compact, .{ n, R }, topk_threads, .{ x.ptr, values.ptr, indices.ptr, threshold, greater_prefix, equal_prefix, greater_total, cols, n, row_bytes, w64(4), kk });
}

/// `ids + vocab_offset` (int64): the rank's columns to token ids, through the rank's id table (lookup, same bits).
pub fn add(f: *Fwd, smp: *const Sampler, ids: T, offset: usize, out: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.add", &.{ .{ .t = ids }, .{ .int = @intCast(offset) }, .{ .t = out } });
    const n = cc.bytes(out) / 8;
    try go(f, (try kernels(smp)).lookup, .{ blocks(n, 256), 1 }, 256, .{ smp.s.vocab.ptr, w64(smp.s.vocab.shape[0]), ids.ptr, out.ptr, w64(n), smp.s.bad.ptr });
}

/// `.to(torch.int32)` of int64 ids: their low words.
pub fn narrow(f: *Fwd, smp: *const Sampler, ids: T, out: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.to", &.{ .{ .t = ids }, .{ .t = out } });
    const n = cc.bytes(out) / 4;
    try go(f, (try kernels(smp)).narrow, .{ blocks(n, 256), 1 }, 256, .{ ids.ptr, out.ptr, w64(n) });
}

/// `torch.cat([values, ids.view(torch.float32)], dim=1)`: [R, k] and [R, k] into [R, 2k] (two strided copies).
fn cat(f: *Fwd, smp: *const Sampler, values: T, ids: T, out: T) !void {
    if (f.rec) |t| return t.call(.ext, "torch.cat", &.{ .{ .list = &.{ .{ .t = values }, .{ .t = ids } } }, .{ .int = 1 }, .{ .t = out } });
    const ks = try kernels(smp);
    const R = w64(values.shape[0]);
    const w = w64(values.shape[1] * 4);
    for ([_]T{ values, ids }, 0..) |x, i| try go(f, ks.strided, .{ R, 1 }, 256, .{ x.ptr, out.ptr + i * w, R, w64(1), w, w, w, 2 * w, w });
}

/// decode.sample_rows with sampling None (greedy) on `logits` [R, V/world] bf16: each row's token, the same on every
/// rank (`out`, R entries). Recording runs nothing: the trace's picker answers (token 0 without one, so a --dry run
/// writes the same files every time).
pub fn greedy(f: *Fwd, smp: *Sampler, logits: T, out: []u32) !void {
    return greedyTop(f, smp, logits, out, null);
}

/// greedy, and with `probs` python's draft form: topk of draft_k candidates a rank (`k = 20 + MARGIN`), the same
/// choice, and each row's temperature-1 probability of its token over every gathered candidate (decode._probability).
pub fn greedyTop(f: *Fwd, smp: *Sampler, logits: T, out: []u32, probs: ?[]f64) !void {
    return greedyCut(f, smp, logits, out, probs, .{});
}

/// A greedy request's top_p and top_k (temperature 0), which decode._probability applies to a draft's probability.
pub const Cut = struct { top_p: f64 = 1.0, top_k: usize = 0 };

/// greedyTop under a greedy request's cut (`probs` only: the choice is the same).
pub fn greedyCut(f: *Fwd, smp: *Sampler, logits: T, out: []u32, probs: ?[]f64, cut: Cut) !void {
    const m = f.m;
    const R = logits.shape[0];
    const V = logits.shape[1];
    const s = &smp.s;
    if (R == 0 or R > s.topk.values.shape[0] or V != s.float.shape[1] or out.len < R) return error.SampleShape;
    if (probs) |p| if (p.len < R) return error.SampleShape;
    const k: usize = if (probs != null) @min(V, draft_k) else 1;
    if (k > s.topk.values.shape[1]) return error.SampleShape;
    try made(smp, "float", &s.float);
    const x = T.of(s.float, 0, &.{ R, V });
    try toFloat(f, smp, logits, x);
    if (smp.tmp) |t| try t.madeFields("topk", &.{ "values", "indices" }, &.{ &s.topk.values, &s.topk.indices });
    const vals = T.of(s.topk.values, 0, &.{ R, k });
    try topk(f, smp, x, k, vals, T.of(s.topk.indices, 0, &.{ R, k }));
    try made(smp, "add", &s.add);
    try add(f, smp, T.of(s.topk.indices, 0, &.{ R, k }), m.rank * m.vl(), T.of(s.add, 0, &.{ R, k }));
    try made(smp, "to", &s.to);
    const ids = T.of(s.to, 0, &.{ R, k });
    try narrow(f, smp, T.of(s.add, 0, &.{ R, k }), ids);
    if (s.cpu.ptr == 0) return error.NoHostRows;
    const host: [*]u8 = @ptrFromInt(s.cpu.ptr);
    if (m.world > 1) {
        var as_f32 = ids;
        as_f32.dtype = .f32;
        try made(smp, "cat", &s.cat);
        try cat(f, smp, vals, as_f32, T.of(s.cat, 0, &.{ R, 2 * k }));
        try made(smp, "empty", &s.empty);
        const got = T.of(s.empty, 0, &.{m.world * R * 2 * k});
        try f.allGather(T.of(s.cat, 0, &.{R * 2 * k}), got);
    }
    try hostRows(f, smp, vals, ids, host);
    if (f.rec) |t| return t.picks(logits, null, out[0..R], if (probs) |p| p[0..R] else null);
    const words: [*]const u32 = @ptrCast(@alignCast(host));
    if (!idsInShards(m.world, m.vl(), words, R, k)) { // Mia's 0092: once more after the device idles, then the request fails
        std.debug.print("[tensorfold] sampler: a gathered id outside its rank's vocabulary; copying the rows again once the device is idle\n", .{});
        try deviceSync(f);
        try hostRows(f, smp, vals, ids, host);
        if (!idsInShards(m.world, m.vl(), words, R, k)) return error.BadTokenId;
    }
    var values: [max_candidates]f32 = undefined;
    var tokens: [max_candidates]i64 = undefined;
    const n = m.world * k;
    if (n > values.len) return error.SampleShape;
    for (0..R) |i| {
        for (0..m.world) |r| {
            for (0..k) |c| {
                const at = slot(m.world, R, k, r, i, c);
                values[r * k + c] = @bitCast(words[at[0]]);
                tokens[r * k + c] = @as(i32, @bitCast(words[at[1]]));
            }
        }
        out[i] = choose(values[0..n], tokens[0..n]);
        if (probs) |p| p[i] = probabilityCut(values[0..n], tokens[0..n], out[i], cut);
    }
}

const ExpFn = *const fn (f64) callconv(.c) f64;
/// glibc's exp, once looked up (its address; 0 before)
var libm_exp = std.atomic.Value(usize).init(0);

fn mathExp(x: f64) callconv(.c) f64 {
    return std.math.exp(x);
}

/// libm's exp, as numpy calls it for float64 where it has no SIMD exp of its own (the Sparks' aarch64): glibc's,
/// opened at run time, because a Zig binary's own `exp` symbol is compiler_rt's, which differs from glibc's in the last
/// bit for some inputs (qualification_qualification_86ea5170 NOTES); std.math.exp where no libm.so.6 opens.
fn libmExp() ExpFn {
    const got = libm_exp.load(.acquire);
    if (got != 0) return @ptrFromInt(got);
    var lib = std.DynLib.open("libm.so.6") catch return &mathExp;
    const f: ExpFn = lib.lookup(ExpFn, "exp") orelse &mathExp; // the library stays open for the process
    libm_exp.store(@intFromPtr(f), .release);
    return f;
}

/// decode._probability for a greedy draw (temperature 1, top_p 1, every gathered candidate): `chosen`'s share of
/// exp(v - max) over the candidates in np.lexsort((tokens, -values)) order, with libm's exp (libmExp) and numpy's
/// pairwise sum; NaN when a value is NaN (np.max's), 0 when `chosen` is not a candidate.
pub fn probability(values: []const f32, tokens: []const i64, chosen: u32) f64 {
    return probabilityCut(values, tokens, chosen, .{});
}

/// decode._probability at temperature 1 under a greedy request's cut: the first top_k candidates (all at 0), then with
/// 0 < top_p < 1 the shares normalized, cut after the first whose running (sequential, np.cumsum) total reaches top_p
/// (np.searchsorted + 1), and normalized again over the rows kept.
pub fn probabilityCut(values: []const f32, tokens: []const i64, chosen: u32, cut: Cut) f64 {
    const n = values.len;
    std.debug.assert(n == tokens.len and n <= max_candidates);
    var order: [max_candidates]usize = undefined;
    for (order[0..n], 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order[0..n], Row{ .values = values, .tokens = tokens }, Row.lt);
    const kk = if (cut.top_k == 0) n else @min(n, cut.top_k);
    var top = -std.math.inf(f64);
    for (order[0..kk]) |i| {
        if (std.math.isNan(values[i])) return std.math.nan(f64);
        top = @max(top, @as(f64, values[i]));
    }
    const exp = libmExp();
    var p: [max_candidates]f64 = undefined;
    for (order[0..kk], p[0..kk]) |i, *x| x.* = exp(@as(f64, values[i]) - top);
    var sum = pairwiseSum(p[0..kk]);
    var keep = kk;
    if (cut.top_p > 0.0 and cut.top_p < 1.0) {
        for (p[0..kk]) |*x| x.* /= sum;
        var run: f64 = 0.0;
        for (p[0..kk], 1..) |x, i| {
            run += x;
            if (run >= cut.top_p) {
                keep = i;
                break;
            }
        }
        sum = pairwiseSum(p[0..keep]);
    }
    for (order[0..keep], p[0..keep]) |i, x| if (tokens[i] == chosen) return x / sum;
    return 0.0;
}

/// One row's candidates in np.lexsort((tokens, -values)) order (`before`'s, NaN last).
const Row = struct {
    values: []const f32,
    tokens: []const i64,
    fn lt(r: Row, a: usize, b: usize) bool {
        return before(r.values[a], r.tokens[a], r.values[b], r.tokens[b]);
    }
};

/// numpy's add.reduce of a contiguous float64 row (pairwise_sum: 8 accumulators up to 128 values, halves above).
pub fn pairwiseSum(a: []const f64) f64 {
    const n = a.len;
    if (n < 8) {
        var res: f64 = 0.0;
        for (a) |x| res += x;
        return res;
    }
    if (n <= 128) {
        var r: [8]f64 = a[0..8].*;
        var i: usize = 8;
        while (i < n - n % 8) : (i += 8) {
            for (&r, a[i .. i + 8]) |*acc, x| acc.* += x;
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < n) : (i += 1) res += a[i];
        return res;
    }
    var half = n / 2;
    half -= half % 8;
    return pairwiseSum(a[0..half]) + pairwiseSum(a[half..]);
}

/// [world, R, 2k]: rank r's row i holds k values, then k ids (one rank: values [R, k], then ids [R, k]).
fn slot(world: usize, R: usize, k: usize, r: usize, i: usize, c: usize) [2]usize {
    return if (world == 1) .{ i * k + c, R * k + i * k + c } else .{ (r * R + i) * 2 * k + c, (r * R + i) * 2 * k + k + c };
}

/// The rows' `.cpu()`: one rank's values, then its ids; on several ranks the gathered pairs.
pub fn hostRows(f: *Fwd, smp: *const Sampler, vals: T, ids: T, host: [*]u8) !void {
    const m = f.m;
    const R = vals.shape[0];
    const k = vals.shape[1];
    if (m.world == 1) { // w.comm is None: values and ids come to the host as they are
        try cc.toHost(f, vals, host[0 .. R * k * 4]);
        return cc.toHost(f, ids, host[R * k * 4 .. 2 * R * k * 4]);
    }
    try cc.toHost(f, T.of(smp.s.empty, 0, &.{ m.world, R, 2 * k }), host[0 .. m.world * R * 2 * k * 4]);
}

/// 0092's check on the gathered pairs: every id inside the vocabulary rows of the rank that sent it.
pub fn idsInShards(world: usize, vl: usize, words: [*]const u32, R: usize, k: usize) bool {
    for (0..R) |i| for (0..world) |r| for (0..k) |c| {
        const id: i64 = @as(i32, @bitCast(words[slot(world, R, k, r, i, c)[1]]));
        const lo: i64 = @intCast(r * vl);
        if (id < lo or id >= lo + @as(i64, @intCast(vl))) return false;
    };
    return true;
}

/// torch.cuda.synchronize(): the whole device idle (0092's wait before the rows are copied again).
pub fn deviceSync(f: *Fwd) !void {
    const g = f.gpu orelse return error.NoGpu;
    const d = g.stream.d;
    try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
}

/// np.lexsort((tokens, -values))[0] over one row's candidates: the largest value, a tie (or a row of NaNs) to the
/// smallest id; NaN sorts after every number, and -0 equals +0.
pub fn choose(values: []const f32, tokens: []const i64) u32 {
    var best: usize = 0;
    for (values[1..], tokens[1..], 1..) |v, t, i| {
        if (before(v, t, values[best], tokens[best])) best = i;
    }
    return @intCast(tokens[best]);
}

fn before(va: f32, ta: i64, vb: f32, tb: i64) bool {
    const na = std.math.isNan(va);
    const nb = std.math.isNan(vb);
    if (na != nb) return nb;
    if (!na and va != vb) return va > vb;
    return ta < tb;
}

/// torch.topk(row, k)'s columns in CUDA's order, k <= 32 (PyTorch 2.13's TensorTopK.cu, then Sort.cu's small sort):
/// those above the k-th largest key, then those equal to it, each in column order, then a 32-slot unstable bitonic sort.
pub fn topkOrder(row: []const f32, k: usize, out: []usize) void {
    std.debug.assert(k >= 1 and k <= 32 and k <= row.len and out.len >= k);
    var top: [32]u32 = undefined; // the k largest keys so far, largest first, repeats kept
    var n: usize = 0;
    for (row) |v| {
        const key = radix(v);
        if (n == k) {
            if (key <= top[k - 1]) continue;
            n -= 1;
        }
        var i = n;
        while (i > 0 and top[i - 1] < key) : (i -= 1) top[i] = top[i - 1];
        top[i] = key;
        n += 1;
    }
    var m: usize = 0;
    for (row, 0..) |v, c| if (radix(v) > top[k - 1]) {
        out[m] = c;
        m += 1;
    };
    for (row, 0..) |v, c| if (m < k and radix(v) == top[k - 1]) {
        out[m] = c;
        m += 1;
    };
    if (k > 1) bitonic(row, out[0..k]); // topk's sorted=True; one value is not sorted
}

/// SortingRadixSelect.cuh's TopKTypeConfig<float>::convert: the bits in value order (+0 above -0), every NaN on top.
fn radix(v: f32) u32 {
    if (std.math.isNan(v)) return 0xffff_ffff;
    const x: u32 = @bitCast(v);
    return x ^ (if (x & 0x8000_0000 != 0) @as(u32, 0xffff_ffff) else 0x8000_0000);
}

/// SortUtils.cuh's bitonicSortKVInPlace on 32 slots (16 threads, two each) with GTOp<float, true>, largest first, the
/// slots past `cols` invalid; each stage's pairs are disjoint, so walking the threads in turn gives the kernel's result.
fn bitonic(row: []const f32, cols: []usize) void {
    var s: Slots = .{};
    for (cols, 0..) |c, i| {
        s.key[i] = row[c];
        s.col[i] = c;
        s.ok[i] = true;
    }
    var size: usize = 2;
    while (size < 32) : (size *= 2) {
        var stride = size / 2;
        while (stride > 0) : (stride /= 2) for (0..16) |t| s.swap(t, stride, t & (size / 2) != 0);
    }
    var stride: usize = 16;
    while (stride > 0) : (stride /= 2) for (0..16) |t| s.swap(t, stride, false);
    @memcpy(cols, s.col[0..cols.len]);
}

const Slots = struct {
    key: [32]f32 = @splat(0),
    col: [32]usize = @splat(0),
    ok: [32]bool = @splat(false),

    /// bitonicSwap: invalid slots sort last; equal valid keys swap when `dir` is false.
    fn swap(s: *Slots, t: usize, stride: usize, dir: bool) void {
        const a = 2 * t - (t & (stride - 1));
        const b = a + stride;
        const gt = (std.math.isNan(s.key[a]) and !std.math.isNan(s.key[b])) or s.key[a] > s.key[b];
        if (((gt and s.ok[a]) or !s.ok[b]) != dir) return;
        std.mem.swap(f32, &s.key[a], &s.key[b]);
        std.mem.swap(usize, &s.col[a], &s.col[b]);
        std.mem.swap(bool, &s.ok[a], &s.ok[b]);
    }
};

test "topkOrder: torch's CUDA order, ties in the bitonic network's order, not by column" {
    var row: [64]f32 = @splat(0);
    var got: [5]usize = undefined;
    topkOrder(&row, 5, &got); // five equal values (tf-glm-forward --dry's logits): the network's order
    try std.testing.expectEqualSlices(usize, &.{ 1, 0, 2, 4, 3 }, &got);
    topkOrder(&row, 1, got[0..1]); // topk 1 is not sorted: the first column
    try std.testing.expectEqual(@as(usize, 0), got[0]);
    row[9] = 2.0;
    row[40] = 1.5;
    row[3] = std.math.nan(f32);
    row[0] = -0.0;
    topkOrder(&row, 5, &got);
    try std.testing.expectEqualSlices(usize, &.{ 3, 9, 40 }, got[0..3]); // NaN first, then by value
    try std.testing.expect(got[3] + got[4] == 3 and got[3] * got[4] == 2); // -0 below +0: the +0 columns 1 and 2 win
}

test "choose: the largest value, ties to the smallest id, NaN last, -0 equal to +0" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    try std.testing.expectEqual(@as(u32, 7), choose(&.{ 1.0, 2.0 }, &.{ 9, 7 }));
    try std.testing.expectEqual(@as(u32, 3), choose(&.{ 2.0, 2.0 }, &.{ 9, 3 }));
    try std.testing.expectEqual(@as(u32, 9), choose(&.{ nan, -inf }, &.{ 4, 9 }));
    try std.testing.expectEqual(@as(u32, 4), choose(&.{ nan, nan }, &.{ 8, 4 }));
    try std.testing.expectEqual(@as(u32, 5), choose(&.{ -0.0, 0.0 }, &.{ 6, 5 }));
}
