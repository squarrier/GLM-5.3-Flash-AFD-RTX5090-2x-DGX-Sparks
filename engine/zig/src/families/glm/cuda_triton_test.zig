//! cuda_triton.zig's wrappers against the Python wrappers' own launches: fixtures_cuda_triton.json, which
//! tools/zig/glm_triton_fixtures.py records on a CPU (python-0.6's glm5_next/cuda wrappers, recorders for kernels).
//! Adapted from the tests of zig/src/families/flashnext/cuda_triton.zig, written by Mia (MiaAI-Lab, @MiaAI_lab) in
//! MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, Apache License 2.0; the GLM cases are new.

const std = @import("std");
const glm = @import("cuda_triton.zig");
const Tri = glm.Tri;
const Recorder = glm.Recorder;
const Dims = glm.Dims;
const Absorb = glm.Absorb;
const Partials = glm.Partials;
const sparse_width = glm.sparse_width;
const prompt_att_rows = glm.prompt_att_rows;
const chunksFor = glm.chunksFor;
const poolBucket = glm.poolBucket;
const b16SplitK = glm.b16SplitK;

const testing = std.testing;
const fixture_text = @embedFile("fixtures_cuda_triton.json");

pub const Fixture = struct {
    parsed: std.json.Parsed(std.json.Value),

    pub fn load(gpa: std.mem.Allocator) !Fixture {
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, gpa, fixture_text, .{}) };
    }
    pub fn deinit(f: *Fixture) void {
        f.parsed.deinit();
    }
    pub fn cases(f: *const Fixture) std.json.ObjectMap {
        return f.parsed.value.object.get("cases").?.object;
    }
    fn case(f: *const Fixture, id: []const u8) ![]const std.json.Value {
        const c = f.cases().get(id) orelse {
            std.debug.print("no fixture case {s}\n", .{id});
            return error.MissingCase;
        };
        return c.array.items;
    }
};

fn jsonInt(v: std.json.Value) i64 {
    return v.integer;
}

/// The recorded launches equal the fixture's: kernel, grid, each runtime argument's type and value by name, and
/// every non-None constexpr.
fn expectLaunches(f: *const Fixture, id: []const u8, rec: *const Recorder) !void {
    const want = try f.case(id);
    errdefer std.debug.print("case {s}\n", .{id});
    try testing.expectEqual(want.len, rec.n);
    for (want, rec.launches[0..rec.n]) |w, got| {
        const o = w.object;
        try testing.expectEqualStrings(o.get("fn").?.string, got.name);
        const grid = o.get("grid").?.array.items;
        for (grid, got.grid) |g, x| try testing.expectEqual(jsonInt(g), @as(i64, x));
        const args = o.get("args").?.array.items;
        try testing.expectEqual(args.len, got.nargs);
        for (args) |arg| {
            const name = arg.object.get("name").?.string;
            const ty = arg.object.get("type").?.string;
            const mine = for (got.args[0..got.nargs]) |x| {
                if (std.mem.eql(u8, x.name, name)) break x;
            } else {
                std.debug.print("{s}: no argument {s}\n", .{ got.name, name });
                return error.MissingArgument;
            };
            errdefer std.debug.print("{s}: argument {s}\n", .{ got.name, name });
            switch (mine.value) {
                .ptr => |x| {
                    try testing.expectEqualStrings(ty, x.ty);
                    try testing.expectEqual(arg.object.get("align16").?.bool, x.addr % 16 == 0);
                },
                .i32 => |x| {
                    try testing.expectEqualStrings(ty, "i32");
                    try testing.expectEqual(jsonInt(arg.object.get("int").?), @as(i64, x));
                },
                .f32 => |x| {
                    try testing.expectEqualStrings(ty, "fp32");
                    try testing.expectEqual(jsonInt(arg.object.get("f32").?), @as(i64, @as(u32, @bitCast(x))));
                },
                .u64 => return error.UnexpectedWord,
            }
        }
        const consts = o.get("consts").?.object;
        var live: usize = 0;
        var it = consts.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* == .null) continue;
            live += 1;
            const name = kv.key_ptr.*;
            const mine = for (got.consts[0..got.nconsts]) |c| {
                if (std.mem.eql(u8, c.name, name)) break c;
            } else {
                std.debug.print("{s}: no constexpr {s}\n", .{ got.name, name });
                return error.MissingConst;
            };
            errdefer std.debug.print("{s}: constexpr {s}\n", .{ got.name, name });
            const v = kv.value_ptr.object;
            if (v.get("int")) |x| try testing.expectEqual(jsonInt(x), mine.int.?);
            if (v.get("f32")) |x| try testing.expectEqual(jsonInt(x), @as(i64, @as(u32, @bitCast(mine.f32.?))));
        }
        try testing.expectEqual(live, got.nconsts);
    }
}

/// Distinct 16-aligned stand-in addresses, 256 MiB apart (types and alignment, not values, select a variant).
fn addr(i: u64) u64 {
    return 0x7f0000000000 + i * 0x10000000;
}

const Ctx = struct {
    f: *const Fixture,
    rec: Recorder = .{},
    checked: usize = 0,
    buf: [96]u8 = undefined,

    fn tri(c: *Ctx) Tri {
        c.rec.n = 0;
        return .{ .set = null, .rec = &c.rec };
    }
    fn expect(c: *Ctx, comptime fmt: []const u8, args: anytype) !void {
        const id = try std.fmt.bufPrint(&c.buf, fmt, args);
        try expectLaunches(c.f, id, &c.rec);
        c.checked += 1;
    }
};

/// tools/zig/glm_triton_cases.py's rows: verify windows (a pending token and up to 7 drafts), prompt chunks and tails
const rows_dec = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8 };
const rows_pre = [_]usize{ 1, 2, 15, 16, 17, 32, 33, 63, 64, 65, 100, 128, 129, 256, 511, 512, 513, 1000, 1024, 2047, 2048 };
const rows_all = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 32, 33, 63, 64, 65, 100, 128, 129, 256, 511, 512, 513, 1000, 1024, 2047, 2048 };
/// latent chunks of a window (Buffers of 8 rows: chunks_for(2560 + 8)) and of a prompt chunk (chunks_for(2560 + 2048))
const nch_dec = [_]usize{ 1, 2, 3, 4, 5, 6 };
const nch_pre = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
/// TF_GLM_LATENT=0's chunks: eager windows to the dense limit, and a 262,144-slot graph's 513
const nch_ab = [_]usize{ 1, 2, 3, 4, 5, 6, 513 };
/// select_tokens' pool buckets: a power of two, a 262,144-slot cache's, an odd capacity's
const pool_buckets = [_]usize{ 1024, 65536, 65537 };
/// KDA layers: 45 less the 11 DSA ones
const kda_layers = 34;

fn contains(xs: []const usize, x: usize) bool {
    return std.mem.indexOfScalar(usize, xs, x) != null;
}

fn glueCases(c: *Ctx) !void {
    const d = Dims.rank(2);
    const D = d.hidden;
    const S = d.streams;
    for (rows_all) |r| {
        for ([_]usize{ S, 1 }) |s| {
            try c.tri().embed(addr(0), .{ .q4 = .{ .w = addr(1), .s = addr(2), .b = addr(3) } }, addr(4), r, D, s);
            try c.expect("embed/q4/r{d}/s{d}", .{ r, s });
            try c.tri().embed(addr(0), .{ .b16 = addr(1) }, addr(4), r, D, s);
            try c.expect("embed/b16/r{d}/s{d}", .{ r, s });
        }
        const dp = addr(5); // the DSA projection [R, q_lora + kv_lora]: q's rows, then the latent's
        const dp_w = d.q_lora + d.kv_lora;
        try c.tri().rmsnorm(dp, dp_w, addr(6), addr(7), d.q_lora, addr(8), d.eps, r, d.q_lora);
        try c.expect("rmsnorm/q/r{d}", .{r});
        try c.tri().rmsnorm(dp + 2 * d.q_lora, dp_w, addr(6), addr(7), d.kv_lora, addr(8), d.eps, r, d.kv_lora);
        try c.expect("rmsnorm/kv/r{d}", .{r});
        try c.tri().rmsnorm(addr(9), D, addr(6), addr(7), D, addr(8), d.eps, r, D);
        try c.expect("rmsnorm/d/r{d}", .{r});
        // the MTP head's [enorm(e) | hnorm(h)] row: the second half, no sums
        try c.tri().rmsnorm(addr(9), D, addr(6), addr(10) + 2 * D, 2 * D, null, d.eps, r, D);
        try c.expect("rmsnorm/half/r{d}", .{r});
        try c.tri().hcPre(addr(11), .{ .fn_w = addr(12), .base = addr(13), .scale = addr(14) }, addr(6), .{ .out = addr(7), .xs = addr(8), .post = addr(15), .comb = addr(16), .part = addr(17) }, d.eps, d.hc_eps, d.sinkhorn, r, D, S);
        try c.expect("hc_pre/r{d}", .{r});
        try c.tri().hcPost(addr(11), addr(11), addr(18), addr(15), addr(16), 2, r, D, S);
        try c.expect("hc_post/r{d}", .{r});
        try c.tri().residualAdd(addr(9), addr(19), addr(18), 2, r, D);
        try c.expect("residual_add/r{d}", .{r});
        try c.tri().streamMean(addr(11), addr(7), r, D, S);
        try c.expect("stream_mean/r{d}", .{r});
        try c.tri().swiglu(addr(20), addr(21), addr(22), d.limit, r, d.dense);
        try c.expect("swiglu/dense/r{d}", .{r});
        try c.tri().swiglu(addr(20), addr(21), addr(22), d.limit, r, d.shared);
        try c.expect("swiglu/shared/r{d}", .{r});
        try c.tri().router(addr(9), D, addr(23), addr(24), d.experts, addr(25), r, D, d.experts);
        try c.expect("router/moe/r{d}", .{r});
        try c.tri().router(addr(9), D, addr(23), addr(24), d.index_dim, addr(25), r, D, d.index_dim);
        try c.expect("router/gate/r{d}", .{r});
        try c.tri().router(addr(9), D, addr(23), addr(24), 2 * d.experts, addr(25), r, D, d.experts);
        try c.expect("router/strided/r{d}", .{r});
        try c.tri().select(addr(24), addr(26), addr(27), addr(28), r, d.experts, d.top_k, d.route_scale, true);
        try c.expect("select/r{d}", .{r});
        try c.tri().combine(addr(29), true, addr(28), addr(30), r, D, d.top_k + 1);
        try c.expect("combine/f32/r{d}", .{r});
        try c.tri().combine(addr(29), false, addr(28), addr(30), r, D, d.top_k + 1);
        try c.expect("combine/bf16/r{d}", .{r});
    }
}

/// glm_triton_cases.Dims.mats: EXL3's BF16 projections a rank runs: (N, K, fp32 out, x row stride)
const Mat = struct { name: []const u8, n: usize, k: usize, f32_out: bool, stride: usize };

fn mats(d: Dims) [16]Mat {
    const kw = d.kdaWidth();
    const kda = d.kda_heads * d.kda_dim;
    return .{
        .{ .name = "kda_proj", .n = kw, .k = d.hidden, .f32_out = false, .stride = d.hidden },
        .{ .name = "kda_fb", .n = kda, .k = d.kda_dim, .f32_out = false, .stride = kw },
        .{ .name = "kda_o", .n = d.hidden, .k = kda, .f32_out = true, .stride = kda },
        .{ .name = "dsa_proj", .n = d.q_lora + d.kv_lora, .k = d.hidden, .f32_out = false, .stride = d.hidden },
        .{ .name = "dsa_qb", .n = d.heads * d.qk_dim, .k = d.q_lora, .f32_out = false, .stride = d.q_lora },
        .{ .name = "dsa_o", .n = d.hidden, .k = d.heads * d.v_dim, .f32_out = true, .stride = d.heads * d.v_dim },
        .{ .name = "idx_kw", .n = d.index_dim + d.index_heads, .k = d.hidden, .f32_out = false, .stride = d.hidden },
        .{ .name = "idx_qb", .n = d.index_heads * d.index_dim, .k = d.q_lora, .f32_out = false, .stride = d.q_lora },
        .{ .name = "mlp_gu", .n = 2 * d.dense, .k = d.hidden, .f32_out = false, .stride = d.hidden },
        .{ .name = "mlp_down", .n = d.hidden, .k = d.dense, .f32_out = true, .stride = d.dense },
        .{ .name = "sh_gu", .n = 2 * d.shared, .k = d.hidden, .f32_out = false, .stride = d.hidden },
        .{ .name = "sh_down", .n = d.hidden, .k = d.shared, .f32_out = true, .stride = d.shared },
        .{ .name = "mtp_eh", .n = d.hidden, .k = 2 * d.hidden, .f32_out = false, .stride = 2 * d.hidden },
        .{ .name = "head", .n = d.vocab, .k = d.hidden, .f32_out = false, .stride = d.hidden },
        .{ .name = "ab_kv_k", .n = d.heads * d.qk_dim, .k = d.kv_lora, .f32_out = false, .stride = d.kv_lora },
        .{ .name = "ab_kv_v", .n = d.heads * d.v_dim, .k = d.kv_lora, .f32_out = false, .stride = d.kv_lora },
    };
}

fn qmmCases(c: *Ctx) !void {
    const d = Dims.rank(2);
    const kw = d.kdaWidth();
    const fa = d.convDim(); // the projection row's f_a columns
    const Input = struct { name: []const u8, x: u64, stride: usize, k: usize };
    for (rows_all) |r| {
        const ins = [_]Input{
            .{ .name = "fa", .x = addr(1) + 2 * fa, .stride = kw, .k = d.kda_dim },
            .{ .name = "kda", .x = addr(1), .stride = d.kda_heads * d.kda_dim, .k = d.kda_heads * d.kda_dim },
            .{ .name = "dsa", .x = addr(1), .stride = d.heads * d.v_dim, .k = d.heads * d.v_dim },
            .{ .name = "mtp", .x = addr(1), .stride = 2 * d.hidden, .k = 2 * d.hidden },
        };
        for (ins) |in| {
            try c.tri().groupSums(in.x, in.stride, addr(2), r, in.k);
            try c.expect("group_sums/{s}/r{d}", .{ in.name, r });
        }
        for (mats(d)) |m| {
            const x = if (std.mem.eql(u8, m.name, "kda_fb")) addr(1) + 2 * fa else addr(1);
            try c.tri().matmulB16(x, m.stride, addr(3), addr(4), m.f32_out, addr(5), r, m.n, m.k);
            try c.expect("bmm/{s}/r{d}", .{ m.name, r });
        }
    }
}

fn convCases(c: *Ctx) !void {
    const d = Dims.rank(2);
    const C = d.convDim();
    const W = d.kdaWidth();
    // a verify window's commit over every KDA layer: proj [L, 8 rows, W]
    for (rows_dec) |r| for (1..r + 1) |keep| {
        try c.tri().shiftConv(addr(0), addr(1), keep, kda_layers, 3 * C, 8 * W, W, C);
        try c.expect("conv/commit/r{d}/k{d}", .{ r, keep });
    };
    // a prompt chunk's layer (the sixth): proj [1, 2048 rows, W], its first R
    for (rows_pre) |r| {
        try c.tri().shiftConv(addr(0) + 5 * 3 * C * 2, addr(2), r, 1, 3 * C, 2048 * W, W, C);
        try c.expect("conv/prefill/r{d}", .{r});
    }
}

fn latentCases(c: *Ctx) !void {
    const d = Dims.rank(2);
    const H = d.heads;
    const L = d.kv_lora;
    const scale = d.scale();
    const Named = struct { name: []const u8, a: Absorb };
    const absorbs = [_]Named{
        .{ .name = "q4", .a = .{ .q4 = .{ .kw = addr(10), .ks = addr(11), .kb = addr(12), .vw = addr(13), .vs = addr(14), .vb = addr(15) } } },
        .{ .name = "b16", .a = .{ .b16 = .{ .wk = addr(10), .wv = addr(13) } } },
    };
    const dense: Partials = .{ .po = addr(20), .pm = addr(21), .pl = addr(22), .aux = addr(23) };
    const sparse: Partials = .{ .po = addr(24), .pm = addr(25), .pl = addr(26) };
    const block_pos = [_]u64{ addr(6), addr(30), addr(31), addr(32) };
    for (rows_all) |r| {
        for (absorbs) |ab| {
            try c.tri().absorbQ(addr(0), ab.a, addr(1), r, H, d.qk_dim, L);
            try c.expect("absorb_q/{s}/r{d}", .{ ab.name, r });
            try c.tri().expandV(addr(2), ab.a, addr(3), r, H, d.v_dim, L);
            try c.expect("expand_v/{s}/r{d}", .{ ab.name, r });
        }
        try c.tri().latentWrite(addr(4), L, addr(5), addr(6), r, L);
        try c.expect("latent_write/r{d}", .{r});
        try c.tri().latentSparse(addr(1), addr(5), addr(7), addr(8), addr(2), sparse, r, H, L, sparse_width, scale);
        try c.expect("latent_sparse/r{d}", .{r});
        // forward.dense_attention: a verify window (Buffers of 8 rows) and a prompt chunk (PROMPT_ATT_ROWS a call)
        if (contains(&rows_dec, r)) for (nch_dec) |nch| {
            try c.tri().denseAttention(addr(1), addr(5), &block_pos, dense, addr(2), r, 8, H, L, nch, scale);
            try c.expect("latent_attention/dec/r{d}/n{d}", .{ r, nch });
        };
        if (contains(&rows_pre, r)) for (nch_pre) |nch| {
            try c.tri().denseAttention(addr(1), addr(5), &block_pos, dense, addr(2), r, prompt_att_rows, H, L, nch, scale);
            try c.expect("latent_attention/pre/r{d}/n{d}", .{ r, nch });
        };
    }
}

fn sparseCases(c: *Ctx) !void {
    const d = Dims.rank(2);
    const kr = d.index_dim + d.index_heads; // the indexer's [k | head weights] projection row
    const block_pos = [_]u64{ addr(8), addr(30), addr(31), addr(32) };
    for (rows_all) |r| {
        try c.tri().indexUpdate(addr(0), kr, addr(1), .{ .ln_w = addr(2), .ln_b = addr(3), .ape = addr(4) }, .{ .ik = addr(5), .ig = addr(6), .pk = addr(7) }, addr(8), r, d.index_dim);
        try c.expect("index_update/r{d}", .{r});
        for (pool_buckets) |np| {
            try c.tri().selectTokens(addr(9), addr(0) + 2 * d.index_dim, kr, addr(7), addr(10), addr(11), &block_pos, r, np, d.index_heads, d.index_dim);
            try c.expect("select_tokens/r{d}/np{d}", .{ r, np });
        }
    }
}

fn abCases(c: *Ctx) !void {
    const d = Dims.rank(2);
    const H = d.heads;
    const scale = d.scale();
    const s: Partials = .{ .po = addr(20), .pm = addr(21), .pl = addr(22), .aux = addr(23) };
    for (rows_all) |r| {
        try c.tri().kvWrite(addr(0), addr(1), addr(2), addr(3), addr(4), r, H * d.qk_dim);
        try c.expect("ab/kv_write/r{d}", .{r});
        try c.tri().headSparse(addr(0), addr(2), addr(3), addr(5), addr(6), addr(7), s, r, H, d.qk_dim, sparse_width, scale);
        try c.expect("ab/sparse/r{d}", .{r});
        if (contains(&rows_dec, r)) for (nch_ab) |nch| {
            try c.tri().headAttention(addr(0), addr(2), addr(3), addr(4), s, r, H, d.qk_dim, nch, scale);
            try c.expect("ab/attention/r{d}/n{d}", .{ r, nch });
        };
    }
}

fn withFixture(comptime group: fn (*Ctx) anyerror!void) !usize {
    var f = try Fixture.load(testing.allocator);
    defer f.deinit();
    var c: Ctx = .{ .f = &f };
    try group(&c);
    return c.checked;
}

test "embedding, norm, hyper-connection and MoE glue launches equal the Python wrappers'" {
    _ = try withFixture(glueCases);
}

test "BF16 projection launches equal qmm.matmul's for every projection of a TP=2 rank, and the group sums" {
    _ = try withFixture(qmmCases);
}

test "conv commit launches equal forward._shift_conv's: a window's commit and a prompt chunk's layer" {
    _ = try withFixture(convCases);
}

test "latent DSA launches equal the Python wrappers': absorb, expand, write, dense windows and chunks, sparse rows" {
    _ = try withFixture(latentCases);
}

test "indexer launches equal sparse.index_update's and select_tokens' at every pool bucket" {
    _ = try withFixture(sparseCases);
}

test "per-head cache launches (TF_GLM_LATENT=0) equal kv_write's, sparse_attention's and attention's" {
    _ = try withFixture(abCases);
}

test "every case of the fixture is replayed, and its generator reached every call site" {
    var f = try Fixture.load(testing.allocator);
    defer f.deinit();
    var c: Ctx = .{ .f = &f };
    inline for (.{ glueCases, qmmCases, convCases, latentCases, sparseCases, abCases }) |group| try group(&c);
    try testing.expectEqual(f.cases().count(), c.checked);
    for (f.parsed.value.object.get("sites").?.array.items) |site| {
        const o = site.object;
        if (o.get("excluded") != null) continue;
        errdefer std.debug.print("site {s}:{d} {s}\n", .{ o.get("file").?.string, o.get("line").?.integer, o.get("kernel").?.string });
        try testing.expect(o.get("launches").?.integer > 0);
    }
}

test "derived shapes follow the Python helpers" {
    try testing.expectEqual(@as(usize, 12576), Dims.rank(2).kdaWidth());
    try testing.expectEqual(@as(usize, 2051), sparse_width);
    try testing.expectEqual(@as(usize, 6), chunksFor(2560 + 8));
    try testing.expectEqual(@as(usize, 9), chunksFor(2560 + 2048));
    try testing.expectEqual(@as(usize, 1024), poolBucket(0, 1, 65537));
    try testing.expectEqual(@as(usize, 65536), poolBucket(262143 - 2048, 2048, 65536));
    try testing.expectEqual(@as(usize, 4), b16SplitK(4096, 8192));
    try testing.expectEqual(@as(usize, 1), b16SplitK(4096, 128));
}
