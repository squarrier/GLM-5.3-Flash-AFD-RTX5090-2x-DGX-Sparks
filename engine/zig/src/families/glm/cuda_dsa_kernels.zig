//! The DSA's own kernel (qualification_qualification_44ba554e.5 A3): sparse.select_tokens' tail, the integer arithmetic python-0.6 (ed78d6f, sparse.py
//! 205-216) does in 21 torch ops after its kernels, as one launch of zig/kernels/cuda/glm_select.cu. Its tokens and
//! counts are python's by construction; `tail` is its arithmetic on the host, held to python's own tail on drawn pools
//! (fixtures_cuda_select.json, tools/zig/glm_dsa_fixtures.py). A4: the blocks' `pos_dev + n` (tf_glm_pos_add), and
//! `Scratch`, a prompt chunk's sparse partials past b.lat_s (python's torch.empty in latent.sparse_attention).
const std = @import("std");
const cuda = @import("cuda");
const views = @import("cuda_views.zig");

/// sparse.TOPK_POOLS and POOL, and a row's tokens: the 512 pools', then the incomplete last pool's 3
pub const topk_pools = 512;
pub const pool = 4;
pub const width = topk_pools * pool + pool - 1;
/// glm_select.cu's threads a block (one a token)
pub const threads = 256;
/// latent.sparse_attention's 512-key chunks of a row's tokens
pub const sparse_chunks = (width + 511) / 512;

/// latent.sparse_attention's partials (po, pm, pl: torch.empty a call) for prompt chunks of up to `rows` rows, where
/// b.lat_s holds too few (a 2,048-row chunk with sparse rows: 640 MiB). Python's capacity plan reserves them as
/// geometry.mla_chunk_scratch's last term; the engine gives them device addresses (views.walk) and Kernels.scratch.
pub const Scratch = struct {
    po: views.Buf,
    pm: views.Buf,
    pl: views.Buf,

    pub fn init(heads: usize, lw: usize, rows: usize) Scratch {
        const n = sparse_chunks * rows * heads;
        return .{ .po = views.Buf.of(.f32, &.{n * lw}), .pm = views.Buf.of(.f32, &.{n}), .pl = views.Buf.of(.f32, &.{n}) };
    }

    pub fn bytes(s: Scratch) usize {
        return s.po.bytes() + s.pm.bytes() + s.pl.bytes();
    }
};

/// glm_select's image loaded (zig/build/glm_fwd_kernels.zig builds and embeds it beside the affine copies).
pub const Kernels = struct {
    module: cuda.Module,
    tail: cuda.Function,
    add: cuda.Function,
    /// the prompt chunks' sparse partials past b.lat_s, at device addresses the engine reserved (null: none)
    scratch: ?*const Scratch = null,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        if (image.len == 0) return error.NotBuilt; // a host build embeds no image
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .tail = try m.function("tf_glm_select_tail"), .add = try m.function("tf_glm_pos_add") };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }

    /// The tail of `rows` rows: pools [rows, 512] int64 and the device position (int32) -> tokens [rows, 2051] and
    /// counts [rows] int32; rows x cdiv(2051, 256) blocks.
    pub fn selectTail(k: *const Kernels, s: cuda.Stream, pools: u64, pos: u64, tokens: u64, counts: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(pools);
        a.add(pos);
        a.add(tokens);
        a.add(counts);
        a.add(@as(u32, @intCast(rows)));
        try cuda.launch.launch(k.tail, launchConfig(rows), s, &a);
    }

    /// `pos_dev + n` (int32 [1]) into `out`: one thread.
    pub fn posAdd(k: *const Kernels, s: cuda.Stream, pos: u64, n: i32, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(pos);
        a.add(n);
        a.add(out);
        try cuda.launch.launch(k.add, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, s, &a);
    }
};

pub fn launchConfig(rows: usize) cuda.launch.Config {
    return .{ .grid = .{ .x = @intCast(rows), .y = (width + threads - 1) / threads }, .block = .{ .x = threads } };
}

/// The kernel's arithmetic on the host: row r (q = pos + r, npool = (q + 1) / 4) takes pool * 4 + i for its 512
/// pools (int64, kept as int32), then npool * 4 + j where that is <= q (else -1), and counts 2048 + those if npool > 512.
pub fn tail(pools: []const i64, pos: i32, rows: usize, tokens: []i32, counts: []i32) void {
    for (0..rows) |r| {
        const q: i64 = @as(i64, pos) + @as(i64, @intCast(r));
        const npool = @divTrunc(q + 1, pool);
        for (0..topk_pools * pool) |i| tokens[r * width + i] = @truncate(pools[r * topk_pools + i / pool] * pool + @as(i64, @intCast(i % pool)));
        var kept: i64 = 0;
        for (0..pool - 1) |j| {
            const t = npool * pool + @as(i64, @intCast(j));
            tokens[r * width + topk_pools * pool + j] = if (t <= q) @intCast(t) else -1;
            kept += @intFromBool(t <= q);
        }
        counts[r] = @intCast(if (npool > topk_pools) topk_pools * pool + kept else 0);
    }
}

const select_fixture = @embedFile("fixtures_cuda_select.json");

/// glm_forward_fixtures.py's `draws`: splitmix64 from a seed
fn splitmix(s: *u64) u64 {
    s.* +%= 0x9E3779B97F4A7C15;
    var z = s.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

test "select_tokens' tail: python-0.6's own torch ops on drawn pools give `tail`'s tokens and counts" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, select_fixture, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("glm-select-tail/1", parsed.value.object.get("schema").?.string);
    var dense: usize = 0;
    var sparse: usize = 0;
    for (parsed.value.object.get("cases").?.array.items) |c| {
        const o = c.object;
        const rows: usize = @intCast(o.get("rows").?.integer);
        const np: u64 = @intCast(o.get("np").?.integer);
        var seed: u64 = @intCast(o.get("seed").?.integer);
        const pools = try gpa.alloc(i64, rows * topk_pools);
        defer gpa.free(pools);
        for (pools) |*p| p.* = @intCast(splitmix(&seed) % np);
        const tokens = try gpa.alloc(i32, rows * width);
        defer gpa.free(tokens);
        var counts: [16]i32 = undefined;
        tail(pools, @intCast(o.get("pos").?.integer), rows, tokens, counts[0..rows]);
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(tokens), &sha, .{});
        try std.testing.expectEqualStrings(o.get("sha256").?.string, &std.fmt.bytesToHex(sha, .lower));
        for (o.get("counts").?.array.items, o.get("tails").?.array.items, 0..) |n, t, r| {
            try std.testing.expectEqual(n.integer, counts[r]);
            for (t.array.items, 0..) |x, j| try std.testing.expectEqual(x.integer, tokens[r * width + topk_pools * pool + j]);
            if (counts[r] == 0) dense += 1 else sparse += 1;
        }
    }
    try std.testing.expect(dense > 0 and sparse > 0); // rows on both sides of the dense limit
    try std.testing.expectEqual(@as(u32, 9), launchConfig(4).grid.y);
}

test "Scratch: python's own reserve for a prompt chunk's sparse partials (geometry.mla_chunk_scratch's last term)" {
    const s = Scratch.init(32, 512, 2048); // a TP2 rank's heads, the latent width, decode.PREFILL_ROWS
    const topk = 2048; // index_topk
    try std.testing.expectEqual(@as(usize, (topk + 515) / 512 * 2048 * 32 * (512 + 2) * 4), s.bytes());
    try std.testing.expectEqual(@as(usize, 671088640), s.po.bytes()); // [5 * 2048 * 32 * 512] fp32, as python's tmp
}
