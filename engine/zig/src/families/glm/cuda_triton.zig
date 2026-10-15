//! GLM-5.3-Flash's Triton launches: one function per python-0.6 glm5_next/cuda wrapper (ed78d6f), with its grid,
//! arguments and constexprs, so `aot.Set.find` picks the variant Python launched; shapes are a TP=2 rank's.
//! `_merge` and `_sparse_chunks` each have two sources, told apart by their arguments; DFlash2's three are not here.
//! Adapted from zig/src/families/flashnext/cuda_triton.zig by Mia (MiaAI-Lab, @MiaAI_lab), MiaAI-Lab/Qwen3.8-Flash-
//! Dual-DGX-Sparks-TensorFold @ cbfc429 (patch 0004), Apache-2.0: `Tri` and `Recorder` are hers; the GLM wrappers new.

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const p = aot.ptr;

const bf16 = "*bf16";
const f32p = "*fp32";
const i32p = "*i32";
const i64p = "*i64";

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// triton.next_power_of_2
pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(n, 1)) catch unreachable;
}

/// Row tiles of the router and the BF16 matmul: glue.router's and qmm.bucket(min(m, 128))'s 16, 32, 64 or 128.
pub fn bucket(m: usize) usize {
    return if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
}

/// glue.HC_BLOCKS: the hyper-connection projection's partial blocks a row
pub const hc_blocks = 16;
/// glue.ROUTER_KS: the router's K slices when the row width allows them
pub const router_ks = 8;
/// latent.CHUNK, attention.CHUNK, sparse.sparse_attention's CH: keys a chunk program
pub const chunk = 512;
/// sparse.TOPK_POOLS and sparse.POOL: a row's selected pools of 4 tokens
pub const topk_pools = 512;
pub const pool = 4;
/// select_tokens' row width: the 512 pools' tokens, then the incomplete last pool's 3
pub const sparse_width = topk_pools * pool + pool - 1;
/// geometry.MLA_SELECT_ROWS: a prompt chunk's rows scored at once
pub const select_rows = 512;
/// geometry.MLA_PROMPT_ATT_ROWS: a prompt chunk's rows a dense latent attention call takes
pub const prompt_att_rows = 512;
/// sparse.PART_ROWS: the per-head sparse attention's rows a launch
pub const part_rows = 128;

/// latent.head_block: a rank's 32 heads a program in prompt chunks (64 rows and up), 16 in decode windows.
pub fn headBlock(rows: usize) usize {
    return if (rows >= 64) 32 else 16;
}

/// latent.row_block: rows a program of `_absorb_q4`: 1 for decode windows, 16 for prompt chunks.
pub fn rowBlock(rows: usize) usize {
    return if (rows <= 16) 1 else 16;
}

/// latent.chunks_for: 512-key chunks through `length` keys
pub fn chunksFor(length: usize) usize {
    return cdiv(length, chunk);
}

/// sparse.pool_bucket: pools scored for rows pos .. pos + R - 1, the visible ones rounded up to a power of two (at
/// least 1024), at most the capacity's.
pub fn poolBucket(pos: usize, rows: usize, np_max: usize) usize {
    const visible = (pos + rows) / pool + 1;
    return @min(np_max, @max(1024, pow2(visible)));
}

/// qmm.b16_split_k: K slices of a BF16 matmul, fixed by the shape (64-wide steps).
pub fn b16SplitK(n: usize, k: usize) usize {
    const tiles = cdiv(n, 64);
    const steps = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// The checkpoint's dimensions a rank computes with (config.json's text_config; `rank(world)` cuts what TP splits).
pub const Dims = struct {
    hidden: usize = 4096,
    /// hc_mult
    streams: usize = 4,
    /// DSA heads (num_attention_heads), a rank's share
    heads: usize = 64,
    /// KDA heads (linear attention), a rank's share
    kda_heads: usize = 64,
    kda_dim: usize = 128,
    qk_dim: usize = 256,
    v_dim: usize = 256,
    q_lora: usize = 1536,
    kv_lora: usize = 512,
    index_heads: usize = 32,
    index_dim: usize = 128,
    experts: usize = 288,
    top_k: usize = 8,
    /// intermediate_size, moe_intermediate_size and the shared expert's width, a rank's share
    dense: usize = 12288,
    moe: usize = 2048,
    shared: usize = 2048,
    vocab: usize = 154880,
    eps: f32 = 1e-5,
    hc_eps: f32 = 1e-6,
    sinkhorn: usize = 20,
    limit: f32 = 10.0,
    route_scale: f32 = 2.5,

    /// A rank's share at `world` ranks: DSA and KDA heads, the MLP widths and the vocabulary.
    pub fn rank(world: usize) Dims {
        const d: Dims = .{};
        return .{ .heads = d.heads / world, .kda_heads = d.kda_heads / world, .dense = d.dense / world, .moe = d.moe / world, .shared = d.shared / world, .vocab = d.vocab / world };
    }
    /// KDA's conv channels (q | k | v) and its stacked in-projection row (+ f_a, g_a, beta)
    pub fn convDim(d: Dims) usize {
        return 3 * d.kda_heads * d.kda_dim;
    }
    pub fn kdaWidth(d: Dims) usize {
        return d.convDim() + 2 * d.kda_dim + d.kda_heads;
    }
    /// DSA's softmax scale: qk_dim ** -0.5
    pub fn scale(d: Dims) f32 {
        return @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(d.qk_dim))));
    }
};

/// One launch as `aot.Set.run` takes it, copied (names and types are static strings) for the tests.
/// structured_entry: 256 launches between two flushes (was 64): a served USERS=4 round on the recorder records its live rows'
/// layer fronts back to back (no extension call between them; shared_pool's tests flushed around each front instead).
/// Recording only: the events and their order are unchanged (launches still move to the Trace at the same points).
pub const Recorder = struct {
    launches: [256]Saved = undefined,
    n: usize = 0,

    pub const Saved = struct {
        name: []const u8,
        grid: [3]u32,
        args: [16]aot.Arg = undefined,
        nargs: usize,
        consts: [12]aot.Const = undefined,
        nconsts: usize,
    };

    fn add(r: *Recorder, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        if (r.n == r.launches.len) return error.TooManyLaunches;
        const s = &r.launches[r.n];
        s.* = .{ .name = name, .grid = grid, .nargs = args.len, .nconsts = consts.len };
        @memcpy(s.args[0..args.len], args);
        @memcpy(s.consts[0..consts.len], consts);
        r.n += 1;
    }
};

/// glue.embed's table: MLX 4-bit rows (words int32 [V, D/8], scales and biases bf16 [V, D/64]) or bf16 [V, D].
pub const Table = union(enum) { q4: struct { w: u64, s: u64, b: u64 }, b16: u64 };

/// glue.hc_pre's weights: the projection fn [24, S*D] bf16, base [24] and scale [3] fp32.
pub const HcWeights = struct { fn_w: u64, base: u64, scale: u64 };
/// glue.hc_pre's outputs: out [R, D] bf16, xs [R, D/64], post [R, S], comb [R, S*S], part [R, HC_BLOCKS, 32] fp32.
pub const HcOut = struct { out: u64, xs: u64, post: u64, comb: u64, part: u64 };

/// kv_b absorbed into the query and expanded from the latent: the checkpoint's 4-bit rows (latent.AbsorbQ4: words
/// int32, scales and biases bf16) or BF16 [H, qk, L] and [H, v, L] blocks (latent.AbsorbW, EXL3 checkpoints).
pub const Absorb = union(enum) {
    q4: struct { kw: u64, ks: u64, kb: u64, vw: u64, vs: u64, vb: u64 },
    b16: struct { wk: u64, wv: u64 },
};

/// Chunk partials: fp32 outputs `po`, maxima `pm` and sums `pl` (latent.LatentScratch, attention.AttnScratch, or the
/// buffers sparse_attention allocates); `aux`: latent's int32 `dummy` (dense) or attention's bf16 `out`.
pub const Partials = struct { po: u64, pm: u64, pl: u64, aux: u64 = 0 };

/// sparse.index_update's weights (k_norm weight and bias, index_kpool_compress_ape [4, 128], bf16) and caches (index
/// keys, gates, pool keys).
pub const IndexWeights = struct { ln_w: u64, ln_b: u64, ape: u64 };
pub const IndexCaches = struct { ik: u64, ig: u64, pk: u64 };

pub const Tri = struct {
    set: ?*const aot.Set,
    s: cuda.Stream = undefined,
    /// tests: record each launch instead of running it
    rec: ?*Recorder = null,

    fn run(t: Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const) !void {
        const g: [3]u32 = .{ u(grid[0]), u(grid[1]), u(grid[2]) };
        if (t.rec) |r| return r.add(name, g, args, consts);
        try t.set.?.run(t.s, name, g, args, consts);
    }

    // -- embedding, norms, hyper-connections, MoE glue (glue.py) ------------------------------------------------

    /// glue.embed: ids [R] int32 -> out [R, copies * d] bf16, the token's row in each of `copies` streams.
    pub fn embed(t: Tri, ids: u64, table: Table, out: u64, rows: usize, d: usize, copies: usize) !void {
        switch (table) {
            .b16 => |w| try t.run("_embed_b16", .{ rows, d / 64, 1 }, &.{ p("IDS", i32p, ids), p("W", bf16, w), p("OUT", bf16, out) }, &.{ ci("D", d), ci("COPIES", copies) }),
            .q4 => |q| try t.run("_embed", .{ rows, d / 64, 1 }, &.{ p("IDS", i32p, ids), p("W", i32p, q.w), p("S", bf16, q.s), p("B", bf16, q.b), p("OUT", bf16, out) }, &.{ ci("D", d), ci("COPIES", copies) }),
        }
    }

    /// glue.rmsnorm: rows of x (`x_stride` apart) -> out (`o_stride` apart) bf16, each 64-group's fp32 sum into `xs`
    /// when given (SUMS); without it Python passes `out` itself for XS.
    pub fn rmsnorm(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
        const xs_arg = if (xs) |a| p("XS", f32p, a) else p("XS", bf16, out);
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ p("X", bf16, x), int("x_stride", x_stride), p("W", bf16, w), p("OUT", bf16, out), int("o_stride", o_stride), xs_arg, aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", pow2(d)), cb("SUMS", xs != null) });
    }

    /// glue.hc_pre: streams x [R, S*D] bf16 -> the normed collapsed row, its sums, post and comb (Sinkhorn, `iters`).
    pub fn hcPre(t: Tri, x: u64, w: HcWeights, norm_w: u64, o: HcOut, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize, streams: usize) !void {
        try t.run("_hc_partial", .{ rows, hc_blocks, 1 }, &.{ p("X", bf16, x), p("FN", bf16, w.fn_w), p("PART", f32p, o.part) }, &.{ ci("WIDE", streams * d), ci("NB", hc_blocks), ci("SUB", 128) });
        try t.run("_hc_finish", .{ rows, 1, 1 }, &.{ p("X", bf16, x), p("PART", f32p, o.part), p("BASE", f32p, w.base), p("SCALE", f32p, w.scale), p("NW", bf16, norm_w), p("OUT", bf16, o.out), p("XS", f32p, o.xs), p("POST", f32p, o.post), p("COMB", f32p, o.comb), aot.float("eps_norm", eps), aot.float("hc_eps", hc_eps) }, &.{ ci("D", d), ci("S", streams), ci("NB", hc_blocks), ci("ITERS", iters), ci("BLOCK", d) });
    }

    /// glue.hc_post: xout = the streams x [R, S*D] after a block whose fp32 partials [world, R, D] are summed rank 0
    /// first, mixed by post [R, S] and comb [R, S, S] (x and xout may be one buffer).
    pub fn hcPost(t: Tri, x: u64, xout: u64, gathered: u64, post: u64, comb: u64, world: usize, rows: usize, d: usize, streams: usize) !void {
        const block = @min(1024, d);
        try t.run("_hc_post", .{ rows, d / block, 1 }, &.{ p("X", bf16, x), p("XOUT", bf16, xout), p("G", f32p, gathered), p("POST", f32p, post), p("COMB", f32p, comb), int("RS", rows * d) }, &.{ ci("D", d), ci("S", streams), ci("WORLD", world), ci("BLOCK", block) });
    }

    /// glue.residual_add: xout [R, D] bf16 = x + the gathered fp32 partials [world, R, D], rank 0 first.
    pub fn residualAdd(t: Tri, x: u64, xout: u64, gathered: u64, world: usize, rows: usize, d: usize) !void {
        const block = @min(1024, d);
        try t.run("_residual_add", .{ rows, d / block, 1 }, &.{ p("X", bf16, x), p("XOUT", bf16, xout), p("G", f32p, gathered), int("RS", rows * d) }, &.{ ci("D", d), ci("WORLD", world), ci("BLOCK", block) });
    }

    /// glue.stream_mean: out [R, D] bf16 = the mean of the streams x [R, S*D].
    pub fn streamMean(t: Tri, x: u64, out: u64, rows: usize, d: usize, streams: usize) !void {
        const block = @min(1024, d);
        try t.run("_stream_mean", .{ rows, d / block, 1 }, &.{ p("X", bf16, x), p("OUT", bf16, out) }, &.{ ci("D", d), ci("S", streams), ci("BLOCK", block) });
    }

    /// glue.swiglu: out [R, W] bf16 = silu(gate) * up of gu [R, 2W], clamped at `limit`, with its 64-group sums xs.
    pub fn swiglu(t: Tri, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, w: usize) !void {
        const block = std.math.gcd(@as(usize, 512), w);
        if (block < 64) return error.BadShape;
        try t.run("_swiglu", .{ rows, w / block, 1 }, &.{ p("GU", bf16, gu), p("OUT", bf16, out), p("XS", f32p, xs), aot.float("LIMIT", limit) }, &.{ ci("W", w), ci("BLOCK", block) });
    }

    /// glue.router: logits out [M, NE] fp32 (rows `out_stride` apart) = x [M, D] . w [NE, D] (MoE router; pool gate).
    pub fn router(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, part: u64, m: usize, d: usize, ne: usize) !void {
        const bm = bucket(m);
        // K slices need torch's contiguous out (one row, or rows NE apart) and D a multiple of ROUTER_KS * 64
        if (d % (router_ks * 64) != 0 or !(m == 1 or out_stride == ne)) {
            return t.run("_router", .{ cdiv(m, bm), cdiv(ne, 64), 1 }, &.{ p("X", bf16, x), p("W", bf16, w), p("OUT", f32p, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("D", d), ci("NE", ne), ci("BM", bm), ci("BLOCK_E", 64), ci("BK", 64) });
        }
        try t.run("_router_part", .{ cdiv(m, bm), cdiv(ne, 32), router_ks }, &.{ p("X", bf16, x), p("W", bf16, w), p("PART", f32p, part), int("M", m), int("x_stride", x_stride) }, &.{ ci("D", d), ci("NE", ne), ci("BM", bm), ci("BLOCK_E", 32), ci("BK", 64), ci("KS", router_ks) });
        try t.run("_router_sum", .{ cdiv(m * ne, 1024), 1, 1 }, &.{ p("PART", f32p, part), p("OUT", f32p, out), int("total", m * ne) }, &.{ ci("KS", router_ks), ci("BLOCK", 1024) });
    }

    /// glue.select: each row's top_k experts and weights (pick int32, wts fp32 [R, top_k + 1]), then the shared expert.
    pub fn select(t: Tri, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, experts: usize, top_k: usize, scale: f32, norm: bool) !void {
        try t.run("_topk", .{ rows, 1, 1 }, &.{ p("L", f32p, logits), p("BIAS", f32p, bias), p("PICK", i32p, pick), p("WTS", f32p, wts), aot.float("scale", scale) }, &.{ ci("NE", experts), ci("TOPK", top_k), ci("SLOTS", top_k + 1), ci("BLOCK", pow2(experts + 1)), ci("SLOTP", pow2(top_k + 1)), cb("NORM", norm) });
    }

    /// glue.combine: out [R, D] fp32 = sum over slots of wts [R, slots] * y [R, slots, D] (fp32 or bf16).
    pub fn combine(t: Tri, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        const block = @min(1024, d);
        try t.run("_combine", .{ rows, d / block, 1 }, &.{ p("Y", if (y_f32) f32p else bf16, y), p("WTS", f32p, wts), p("OUT", f32p, out) }, &.{ ci("D", d), ci("SLOTS", slots), ci("BLOCK", block) });
    }

    // -- BF16 projections (qmm.py) ------------------------------------------------------------------------------

    /// qmm.group_sums: x [M, K] bf16 (rows `x_stride` apart) -> xs [M, K/64] fp32 sums of each 64-input group.
    pub fn groupSums(t: Tri, x: u64, x_stride: usize, xs: u64, m: usize, k: usize) !void {
        try t.run("_group_sums", .{ m, cdiv(k / 64, 16), 1 }, &.{ p("X", bf16, x), p("XS", f32p, xs), int("x_stride", x_stride) }, &.{ ci("K", k), ci("GB", 16) });
    }

    /// qmm.matmul on BF16 weights (_matmul_b16): out [M, N] (bf16, fp32 with `f32_out`) = x [M, K] (rows `x_stride`
    /// apart) . w [N, K]^T. Split K (b16SplitK) writes fp32 slices into `part` (SK * M * N floats), then `_reduce`.
    pub fn matmulB16(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, f32_out: bool, part: u64, m: usize, n: usize, k: usize) !void {
        if (k % 64 != 0) return error.BadShape;
        const bm = bucket(@min(m, 128));
        const sk = b16SplitK(n, k);
        const oty = if (f32_out) f32p else bf16;
        const part_arg = if (sk > 1) p("PART", f32p, part) else p("PART", oty, out);
        try t.run("_bmm", .{ cdiv(m, bm), cdiv(n, 64), sk }, &.{ p("X", bf16, x), p("W", bf16, w), p("OUT", oty, out), part_arg, int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("BLOCK_N", 64), ci("BK", 64), cb("F32", f32_out) });
        if (sk > 1) try t.run("_reduce", .{ cdiv(m * n, 1024), 1, 1 }, &.{ p("PART", f32p, part), p("OUT", oty, out), int("total", m * n) }, &.{ ci("SK", sk), ci("BLOCK", 1024), cb("F32", f32_out) });
    }

    /// patch 0002's _matmul_f8 (Mia's v1.10; qualification_qualification_ebed70b7): matmulB16 over w [N, K] e4m3 and its fp32 scales [N, K/128],
    /// F8_CONFIG's row buckets with 32 columns a program at 16 rows and 64 past (F8_BN_DECODE, F8_BN), K slices as B16's.
    pub fn matmulF8(t: Tri, x: u64, x_stride: usize, w: u64, s: u64, out: u64, f32_out: bool, part: u64, m: usize, n: usize, k: usize) !void {
        if (k % 128 != 0) return error.BadShape;
        const bm = bucket(@min(m, 128));
        const bn: usize = if (bm == 16) 32 else 64;
        const sk = b16SplitK(n, k);
        const oty = if (f32_out) f32p else bf16;
        const part_arg = if (sk > 1) p("PART", f32p, part) else p("PART", oty, out);
        try t.run("_fmm", .{ cdiv(m, bm), cdiv(n, bn), sk }, &.{ p("X", bf16, x), p("W", "*fp8e4nv", w), p("S", f32p, s), p("OUT", oty, out), part_arg, int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("BLOCK_N", bn), ci("BK", 64), cb("F32", f32_out) });
        if (sk > 1) try t.run("_reduce", .{ cdiv(m * n, 1024), 1, 1 }, &.{ p("PART", f32p, part), p("OUT", oty, out), int("total", m * n) }, &.{ ci("SK", sk), ci("BLOCK", 1024), cb("F32", f32_out) });
    }

    // -- DSA on the latent cache (latent.py, forward.dense_attention) --------------------------------------------

    /// latent.absorb_q: q [R, H, qk] bf16 -> qa [R, H, lw] bf16 through kv_b's key blocks.
    pub fn absorbQ(t: Tri, q: u64, a: Absorb, qa: u64, rows: usize, heads: usize, qk: usize, lw: usize) !void {
        switch (a) {
            .q4 => |w| {
                const rb = rowBlock(rows);
                try t.run("_absorb_q4", .{ heads, lw / 64, cdiv(rows, rb) }, &.{ p("Q", bf16, q), p("WW", i32p, w.kw), p("WS", bf16, w.ks), p("WB", bf16, w.kb), p("QA", bf16, qa), int("R", rows) }, &.{ ci("H", heads), ci("D", qk), ci("LW", lw), ci("RB", rb) });
            },
            .b16 => |w| try t.run("_absorb_q", .{ heads, lw / 32, 1 }, &.{ p("Q", bf16, q), p("WK", bf16, w.wk), p("QA", bf16, qa), int("R", rows) }, &.{ ci("H", heads), ci("D", qk), ci("LW", lw), ci("BN", 32) }),
        }
    }

    /// latent.expand_v: the attention's latent rows [R, H, lw] bf16 -> out [R, H, v] bf16 through kv_b's value blocks.
    pub fn expandV(t: Tri, ol: u64, a: Absorb, out: u64, rows: usize, heads: usize, v_dim: usize, lw: usize) !void {
        switch (a) {
            .q4 => |w| try t.run("_expand_v4", .{ heads, v_dim / 16, 1 }, &.{ p("OL", bf16, ol), p("WW", i32p, w.vw), p("WS", bf16, w.vs), p("WB", bf16, w.vb), p("OUT", bf16, out), int("R", rows) }, &.{ ci("H", heads), ci("DV", v_dim), ci("LW", lw), ci("BN", 16) }),
            .b16 => |w| try t.run("_expand_v", .{ heads, v_dim / 16, 1 }, &.{ p("OL", bf16, ol), p("WV", bf16, w.wv), p("OUT", bf16, out), int("R", rows) }, &.{ ci("H", heads), ci("DV", v_dim), ci("LW", lw), ci("BN", 16) }),
        }
    }

    /// latent.latent_write: lat [R, lw] bf16 rows (`lat_stride` apart) into cache slots pos .. pos + R - 1.
    pub fn latentWrite(t: Tri, lat: u64, lat_stride: usize, cache: u64, pos: u64, rows: usize, lw: usize) !void {
        try t.run("_lat_write", .{ rows, 1, 1 }, &.{ p("LAT", bf16, lat), int("lat_stride", lat_stride), p("LC", bf16, cache), p("POS", i32p, pos) }, &.{ci("LW", lw)});
    }

    /// latent.attention: dense causal attention of qa [R, H, lw] through pos + R - 1 in `nch` chunks, `hb` heads a
    /// program; the partials hold R rows (`s.aux`: the scratch's int32 dummy, CNT of the dense merge).
    pub fn latentAttention(t: Tri, qa: u64, cache: u64, pos: u64, s: Partials, out: u64, rows: usize, heads: usize, lw: usize, nch: usize, hb: usize, scale: f32) !void {
        if (hb != 16 and hb != 32) return error.BadShape;
        try t.run("_dense_chunks", .{ rows, cdiv(heads, hb), nch }, &.{ p("QA", bf16, qa), p("LC", bf16, cache), p("POS", i32p, pos), p("PO", f32p, s.po), p("PM", f32p, s.pm), p("PL", f32p, s.pl), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("CH", chunk), aot.cf("SCALE", scale), ci("HBT", hb), ci("KTT", 32) });
        try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", f32p, s.po), p("PM", f32p, s.pm), p("PL", f32p, s.pl), p("OUT", bf16, out), p("CNT", i32p, s.aux), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("NCH", nch), cb("SPARSE", false) });
    }

    /// forward.dense_attention: latent.attention in blocks of `block_rows`; block_pos[i] holds block i's first position.
    pub fn denseAttention(t: Tri, qa: u64, cache: u64, block_pos: []const u64, s: Partials, out: u64, rows: usize, block_rows: usize, heads: usize, lw: usize, nch: usize, scale: f32) !void {
        const hb = headBlock(rows); // from all R, as Python's
        const row_bytes = heads * lw * 2;
        if (block_pos.len < cdiv(rows, block_rows)) return error.BadShape;
        var r0: usize = 0;
        var i: usize = 0;
        while (r0 < rows) : ({
            r0 += block_rows;
            i += 1;
        }) {
            const n = @min(block_rows, rows - r0);
            try t.latentAttention(qa + r0 * row_bytes, cache, block_pos[i], s, out + r0 * row_bytes, n, heads, lw, nch, hb, scale);
        }
    }

    /// latent.sparse_attention: rows with counts > 0 attend over their selected tokens [R, width] int32 (ascending,
    /// -1 padded); `s` holds cdiv(width, 512) * R * H partials.
    pub fn latentSparse(t: Tri, qa: u64, cache: u64, tokens: u64, counts: u64, out: u64, s: Partials, rows: usize, heads: usize, lw: usize, width: usize, scale: f32) !void {
        const nch = cdiv(width, chunk);
        const hb = headBlock(rows);
        try t.run("_sparse_chunks", .{ rows, cdiv(heads, hb), nch }, &.{ p("QA", bf16, qa), p("LC", bf16, cache), p("TOK", i32p, tokens), p("CNT", i32p, counts), p("PO", f32p, s.po), p("PM", f32p, s.pm), p("PL", f32p, s.pl), int("R", rows) }, &.{ ci("W", width), ci("H", heads), ci("LW", lw), ci("CH", chunk), aot.cf("SCALE", scale), ci("HBT", hb), ci("KTT", 32) });
        try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", f32p, s.po), p("PM", f32p, s.pm), p("PL", f32p, s.pl), p("OUT", bf16, out), p("CNT", i32p, counts), int("R", rows) }, &.{ ci("H", heads), ci("LW", lw), ci("NCH", nch), cb("SPARSE", true) });
    }

    // -- the DSA indexer (sparse.py) ----------------------------------------------------------------------------

    /// sparse.index_update: the window's index keys (k_raw [R, dim] bf16, rows `k_stride` apart) and fp32 gates into
    /// the caches at pos.., then every pool of 4 the window completes.
    pub fn indexUpdate(t: Tri, k_raw: u64, k_stride: usize, gate: u64, w: IndexWeights, c: IndexCaches, pos: u64, rows: usize, dim: usize) !void {
        try t.run("_index_write", .{ rows, 1, 1 }, &.{ p("KR", bf16, k_raw), int("k_stride", k_stride), p("GR", f32p, gate), p("LNW", bf16, w.ln_w), p("LNB", bf16, w.ln_b), p("IK", bf16, c.ik), p("IG", bf16, c.ig), p("POS", i32p, pos), aot.float("eps", 1e-6) }, &.{ci("D", dim)});
        try t.run("_pool_keys", .{ rows / pool + 2, 1, 1 }, &.{ p("IK", bf16, c.ik), p("IG", bf16, c.ig), p("APE", bf16, w.ape), p("PK", bf16, c.pk), p("POS", i32p, pos), int("R", rows) }, &.{ci("D", dim)});
    }

    /// sparse.select_tokens' kernels: SELECT_ROWS rows at a time, pool scores over `np` pools, then the top 512 pools.
    pub fn selectTokens(t: Tri, qi: u64, wts: u64, w_stride: usize, pk: u64, scores: u64, pools: u64, block_pos: []const u64, rows: usize, np: usize, heads: usize, dim: usize) !void {
        if (np < topk_pools) return error.BadShape; // Python's torch top-k path, no kernel
        const block = @min(rows, select_rows);
        if (block_pos.len < cdiv(rows, block)) return error.BadShape;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim))));
        const wscale: f32 = if (heads == 32) @floatCast(1.0 / 5.656854249492381) else @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(heads))));
        var a: usize = 0;
        var i: usize = 0;
        while (a < rows) : ({
            a += block;
            i += 1;
        }) {
            const n = @min(block, rows - a);
            try t.run("_scores", .{ n, cdiv(np, 64), 1 }, &.{ p("QI", bf16, qi + a * heads * dim * 2), p("W", bf16, wts + a * w_stride * 2), int("w_stride", w_stride), p("PK", bf16, pk), p("OUT", f32p, scores), p("POS", i32p, block_pos[i]), int("R", n), int("NP", np), aot.float("scale", scale), aot.float("wscale", wscale) }, &.{ ci("H", heads), ci("HP", @max(16, pow2(heads))), ci("D", dim), ci("BP", 64), ci("RB", 1) });
            try t.run("_select_rows", .{ n, 1, 1 }, &.{ p("S", f32p, scores), p("OUT", i64p, pools + a * topk_pools * 8), int("NP", np), p("POS", i32p, block_pos[i]) }, &.{ ci("K", topk_pools), ci("BLOCK", 1024), cb("VISIBLE", true) });
        }
    }

    // -- per-head caches: TF_GLM_LATENT=0 (sparse.sparse_attention, attention.py) ---------------------------------

    /// sparse.sparse_attention: rows of q [R, H, D] with counts > 0 over their selected tokens in the key and value
    /// caches, PART_ROWS rows a launch; `s` holds cdiv(width, 512) * PART_ROWS * H partials.
    pub fn headSparse(t: Tri, q: u64, kc: u64, vc: u64, tokens: u64, counts: u64, out: u64, s: Partials, rows: usize, heads: usize, dim: usize, width: usize, scale: f32) !void {
        const nch = cdiv(width, chunk);
        var r0: usize = 0;
        while (r0 < rows) : (r0 += part_rows) {
            const n = @min(part_rows, rows - r0);
            const qrow = q + r0 * heads * dim * 2;
            const tok = tokens + r0 * width * 4;
            const cnt = counts + r0 * 4;
            try t.run("_sparse_chunks", .{ n, heads, nch }, &.{ p("Q", bf16, qrow), p("KC", bf16, kc), p("VC", bf16, vc), p("TOK", i32p, tok), p("CNT", i32p, cnt), p("PO", f32p, s.po), p("PM", f32p, s.pm), p("PL", f32p, s.pl) }, &.{ ci("W", width), ci("H", heads), ci("D", dim), ci("CH", chunk), aot.cf("SCALE", scale) });
            try t.run("_sparse_merge", .{ n, heads, 1 }, &.{ p("PO", f32p, s.po), p("PM", f32p, s.pm), p("PL", f32p, s.pl), p("OUT", bf16, out + r0 * heads * dim * 2), p("CNT", i32p, cnt) }, &.{ ci("H", heads), ci("D", dim), ci("NCH", nch), ci("CH", chunk) });
        }
    }

    /// attention.attention: q [R, H, D] over caches filled through pos + R - 1 in `nch` chunks (at most the scratch's),
    /// 16 rows a program, into the scratch's out (`s.aux`) [R, H, D] bf16.
    pub fn headAttention(t: Tri, q: u64, kc: u64, vc: u64, pos: u64, s: Partials, rows: usize, heads: usize, dim: usize, nch: usize, scale: f32) !void {
        if (rows > 16 * 8 or (dim != 128 and dim != 256)) return error.BadShape;
        try t.run("_chunks", .{ cdiv(rows, 16), heads, nch }, &.{ p("Q", bf16, q), p("KC", bf16, kc), p("VC", bf16, vc), p("POS", i32p, pos), p("PO", f32p, s.po), p("PM", f32p, s.pm), p("PL", f32p, s.pl), int("R", rows) }, &.{ ci("H", heads), ci("D", dim), ci("CH", chunk), aot.cf("SCALE", scale), ci("BRT", 16) });
        try t.run("_merge", .{ rows, heads, 1 }, &.{ p("PO", f32p, s.po), p("PM", f32p, s.pm), p("PL", f32p, s.pl), p("OUT", bf16, s.aux), int("R", rows) }, &.{ ci("H", heads), ci("D", dim), ci("NCH", nch) });
    }

    /// attention.kv_write: key and value rows [R, width] bf16 into cache slots pos .. pos + R - 1.
    pub fn kvWrite(t: Tri, kn: u64, vn: u64, kc: u64, vc: u64, pos: u64, rows: usize, width: usize) !void {
        const block = std.math.gcd(@as(usize, 1024), width);
        try t.run("_kv_write", .{ rows, width / block, 1 }, &.{ p("KN", bf16, kn), p("VN", bf16, vn), p("KC", bf16, kc), p("VC", bf16, vc), p("POS", i32p, pos) }, &.{ ci("W", width), ci("BLOCK", block) });
    }

    // -- commit (forward.py) ------------------------------------------------------------------------------------

    /// forward._shift_conv: conv [L, 3, C] (layers `conv_layer` elements apart, in place) takes rows keep .. keep + 2
    /// of [conv; proj rows], proj rows `proj_row` apart within a layer and `proj_layer` across layers.
    pub fn shiftConv(t: Tri, conv: u64, proj: u64, keep: usize, layers: usize, conv_layer: usize, proj_layer: usize, proj_row: usize, channels: usize) !void {
        try t.run("_conv_shift", .{ layers, cdiv(channels, 1024), 1 }, &.{ p("CONV", bf16, conv), p("PROJ", bf16, proj), int("keep", keep), int("conv_layer", conv_layer), int("proj_layer", proj_layer), int("proj_row", proj_row) }, &.{ ci("C", channels), ci("TAPS", 3), ci("BLOCK", 1024) });
    }
};
