//! A small synthetic GLM-5.3-Flash checkpoint in MLX's affine 4-bit layout (tests): every tensor its own pseudo-random bytes.
const std = @import("std");
const cfg = @import("config.zig");
const split = @import("cuda_split.zig");

const prefix = "model.language_model.";

/// GLM-5.3-Flash's structure at a size a test writes in a moment: hidden stays 4,096 (weights.zig's hc_pack needs it).
pub fn tiny() cfg.Config {
    var c: cfg.Config = .{
        .vocab = 512,
        .dense_layers = 2,
        .dense_inter = 256,
        .experts = 4,
        .topk = 2,
        .moe_inter = 256,
        .kda_heads = 4,
        .kda_dim = 64,
        .mla_heads = 4,
        .nope = 64,
        .v_dim = 64,
        .q_lora = 128,
        .kv_lora = 128,
        .i_heads = 4,
        .i_dim = 64,
    };
    c.own = .{ 0, c.experts };
    c.inter = .{ 0, c.moe_inter };
    c.run = 4; // layers 0-1 KDA + dense, 2 KDA + MoE, 3 MLA + MoE, and the MTP layer 45 (MLA + MoE)
    return c;
}

const Spec = struct { name: []const u8, dtype: []const u8, shape: [3]usize, rank: u8 };

const Specs = struct {
    a: std.mem.Allocator,
    list: std.ArrayList(Spec) = .empty,

    fn add(s: *Specs, comptime fmt: []const u8, args: anytype, dtype: []const u8, shape: []const usize) !void {
        var sp: Spec = .{ .name = try std.fmt.allocPrint(s.a, fmt, args), .dtype = dtype, .shape = @splat(1), .rank = @intCast(shape.len) };
        @memcpy(sp.shape[0..shape.len], shape);
        try s.list.append(s.a, sp);
    }

    /// A 4-bit matrix [n, k]: weight u32 [n, k/8], scales and biases bf16 [n, k/64].
    fn q4(s: *Specs, comptime fmt: []const u8, args: anytype, n: usize, k: usize) !void {
        try s.add(fmt ++ ".weight", args, "U32", &.{ n, k / 8 });
        try s.add(fmt ++ ".scales", args, "BF16", &.{ n, k / 64 });
        try s.add(fmt ++ ".biases", args, "BF16", &.{ n, k / 64 });
    }
};

/// Every tensor the first `c.run` layers, the MTP layer, the embedding, head and norm need, one vision tensor and `extra`.
fn specs(a: std.mem.Allocator, c: *const cfg.Config, extra: []const []const u8) ![]Spec {
    var s: Specs = .{ .a = a };
    const D: usize = c.hidden;
    try s.q4(prefix ++ "embed_tokens", .{}, c.vocab, D);
    try s.q4("lm_head", .{}, c.vocab, D);
    try s.add(prefix ++ "norm.weight", .{}, "BF16", &.{D});
    try s.add("model.visual.patch_embed.proj.weight", .{}, "BF16", &.{ 2, 2, 2 }); // never read
    for (extra) |x| try s.add("{s}", .{x}, "BF16", &.{ 2, 2, 2 });
    var layers: [cfg.max_layers]u32 = undefined;
    var n: usize = 0;
    for (0..c.run) |i| {
        layers[n] = @intCast(i);
        n += 1;
    }
    layers[n] = c.layers;
    n += 1;
    for (layers[0..n]) |i| {
        const p = prefix ++ "layers.{d}.";
        if (i != c.layers) {
            for ([_][]const u8{ "attn", "ffn" }) |site| {
                try s.add(p ++ "hc_{s}_fn", .{ i, site }, "BF16", &.{ 24, 4 * D });
                try s.add(p ++ "hc_{s}_base", .{ i, site }, "F32", &.{24});
                try s.add(p ++ "hc_{s}_scale", .{ i, site }, "F32", &.{3});
            }
        } else {
            try s.q4(p ++ "eh_proj", .{i}, D, 2 * D);
            for ([_][]const u8{ "enorm", "hnorm", "shared_head.norm" }) |x| try s.add(p ++ "{s}.weight", .{ i, x }, "BF16", &.{D});
        }
        try s.add(p ++ "input_layernorm.weight", .{i}, "BF16", &.{D});
        try s.add(p ++ "post_attention_layernorm.weight", .{i}, "BF16", &.{D});
        const a2 = p ++ "self_attn.";
        if (c.kind(i) == .kda) {
            const W: usize = c.kdaWidth();
            const dk: usize = c.kda_dim;
            for ([_][]const u8{ "q", "k", "v" }) |x| {
                try s.add(a2 ++ "{s}_conv1d.weight", .{ i, x }, "BF16", &.{ W, 1, c.conv });
                try s.q4(a2 ++ "{s}_proj", .{ i, x }, W, D);
            }
            for ([_][]const u8{ "f_a", "g_a" }) |x| try s.q4(a2 ++ "{s}_proj", .{ i, x }, dk, D);
            try s.q4(a2 ++ "b_proj", .{i}, c.kda_heads, D);
            for ([_][]const u8{ "f_b", "g_b" }) |x| try s.q4(a2 ++ "{s}_proj", .{ i, x }, W, dk);
            try s.q4(a2 ++ "o_proj", .{i}, D, W);
            try s.add(a2 ++ "A_log", .{i}, "F32", &.{c.kda_heads});
            try s.add(a2 ++ "dt_bias", .{i}, "F32", &.{W});
            try s.add(a2 ++ "o_norm.weight", .{i}, "BF16", &.{dk});
        } else {
            const H: usize = c.mla_heads;
            try s.q4(a2 ++ "q_a_proj", .{i}, c.q_lora, D);
            try s.q4(a2 ++ "kv_a_proj_with_mqa", .{i}, c.kv_lora, D);
            try s.q4(a2 ++ "indexer.wk", .{i}, c.i_dim, D);
            try s.q4(a2 ++ "indexer.weights_proj", .{i}, c.i_heads, D);
            try s.q4(a2 ++ "q_b_proj", .{i}, H * c.nope, c.q_lora);
            try s.q4(a2 ++ "indexer.wq_b", .{i}, c.i_heads * c.i_dim, c.q_lora);
            try s.q4(a2 ++ "kv_b_proj", .{i}, H * (c.nope + c.v_dim), c.kv_lora);
            try s.q4(a2 ++ "o_proj", .{i}, D, H * c.v_dim);
            try s.add(a2 ++ "q_a_layernorm.weight", .{i}, "BF16", &.{c.q_lora});
            try s.add(a2 ++ "kv_a_layernorm.weight", .{i}, "BF16", &.{c.kv_lora});
            try s.add(a2 ++ "indexer.k_norm.weight", .{i}, "BF16", &.{c.i_dim});
            try s.add(a2 ++ "indexer.k_norm.bias", .{i}, "BF16", &.{c.i_dim});
            try s.add(a2 ++ "indexer.index_kpool_compress_ape", .{i}, "BF16", &.{ c.kpool, c.i_dim });
            try s.add(a2 ++ "indexer.index_kpool_compress_gate", .{i}, "BF16", &.{ c.i_dim, D });
        }
        const m = p ++ "mlp.";
        if (c.isMoe(i)) {
            try s.add(m ++ "gate.weight", .{i}, "BF16", &.{ c.experts, D });
            try s.add(m ++ "gate.e_score_correction_bias", .{i}, "F32", &.{c.experts});
            for ([_][]const u8{ "gate", "up" }) |x| try s.q4(m ++ "shared_experts.{s}_proj", .{ i, x }, c.moe_inter, D);
            try s.q4(m ++ "shared_experts.down_proj", .{i}, D, c.moe_inter);
            for (0..c.experts) |e| {
                for ([_][]const u8{ "gate", "up" }) |x| try s.q4(m ++ "experts.{d}.{s}_proj", .{ i, e, x }, c.moe_inter, D);
                try s.q4(m ++ "experts.{d}.down_proj", .{ i, e }, D, c.moe_inter);
            }
        } else {
            for ([_][]const u8{ "gate", "up" }) |x| try s.q4(m ++ "{s}_proj", .{ i, x }, c.dense_inter, D);
            try s.q4(m ++ "down_proj", .{i}, D, c.dense_inter);
        }
    }
    return s.list.items;
}

fn size(sp: Spec) usize {
    var n: usize = if (std.mem.eql(u8, sp.dtype, "BF16")) 2 else 4;
    for (sp.shape[0..sp.rank]) |d| n *= d;
    return n;
}

/// The byte at `i` of the tensor named `name`: a hash of both, so a misplaced row or byte shows.
pub fn byteOf(name: []const u8, i: usize) u8 {
    return @truncate(std.hash.Wyhash.hash(i / 8, name) >> @intCast(8 * (i % 8)));
}

/// Every byte of a tensor as `byteOf` gives it, eight bytes a hash.
fn fill(name: []const u8, bytes: []u8) void {
    var w: usize = 0;
    while (w * 8 < bytes.len) : (w += 1) {
        var word: [8]u8 = undefined;
        std.mem.writeInt(u64, &word, std.hash.Wyhash.hash(w, name), .little);
        const n = @min(8, bytes.len - w * 8);
        @memcpy(bytes[w * 8 ..][0..n], word[0..n]);
    }
}

fn writeAll(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return error.WriteFailed;
        done += @intCast(n);
    }
}

fn create(a: std.mem.Allocator, dir: []const u8, name: []const u8) !std.c.fd_t {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CreateFailed;
    return fd;
}

/// The checkpoint into `dir` (which exists) as `shards` files and model.safetensors.index.json; returns every tensor's place.
pub fn write(a: std.mem.Allocator, dir: []const u8, c: *const cfg.Config, shards: u32) !split.Names {
    return writeExtra(a, dir, c, shards, &.{});
}

/// `write`, plus bf16 [2, 2, 2] tensors named `extra` (outside the language model: the index must not drop them silently).
pub fn writeExtra(a: std.mem.Allocator, dir: []const u8, c: *const cfg.Config, shards: u32, extra: []const []const u8) !split.Names {
    const all = try specs(a, c, extra);
    var names: split.Names = .empty;
    var map: std.Io.Writer.Allocating = .init(a);
    try map.writer.writeAll("{\"metadata\":{},\"weight_map\":{");
    for (0..shards) |si| {
        const file = try std.fmt.allocPrint(a, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ si + 1, shards });
        var head: std.Io.Writer.Allocating = .init(a);
        try head.writer.writeAll("{\"__metadata__\":{\"format\":\"mlx\"}");
        var off: usize = 0;
        for (all, 0..) |sp, k| if (k % shards == si) {
            try head.writer.print(",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ sp.name, sp.dtype });
            for (sp.shape[0..sp.rank], 0..) |d, j| try head.writer.print("{s}{d}", .{ if (j == 0) "" else ",", d });
            try head.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ off, off + size(sp) });
            off += size(sp);
        };
        try head.writer.writeAll("}");
        while (head.written().len % 8 != 0) try head.writer.writeByte(' ');
        const h = head.written();
        const fd = try create(a, dir, file);
        defer _ = std.c.close(fd);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, h.len, .little);
        try writeAll(fd, &len);
        try writeAll(fd, h);
        off = 8 + h.len;
        for (all, 0..) |sp, k| if (k % shards == si) {
            const bytes = try a.alloc(u8, size(sp));
            fill(sp.name, bytes);
            try writeAll(fd, bytes);
            if (std.mem.startsWith(u8, sp.name, prefix) or std.mem.startsWith(u8, sp.name, "lm_head.")) {
                var src: split.Src = .{ .shard = @intCast(si), .off = off, .len = bytes.len, .dtype = if (sp.dtype[0] == 'U') .u32 else if (sp.dtype[0] == 'B') .bf16 else .f32, .rank = sp.rank };
                @memcpy(src.shape[0..sp.rank], sp.shape[0..sp.rank]);
                try names.put(a, sp.name, src);
            }
            off += bytes.len;
            try map.writer.print("{s}\"{s}\":\"{s}\"", .{ if (k == 0) "" else ",", sp.name, file });
        };
    }
    try map.writer.writeAll("}}");
    const fd = try create(a, dir, "model.safetensors.index.json");
    defer _ = std.c.close(fd);
    try writeAll(fd, map.written());
    return names;
}

/// The plan's groups filled by plain preads, each copy as the split states it: the reader's oracle.
pub fn execute(a: std.mem.Allocator, dir: []const u8, shards: u32, p: *const split.Plan) ![][]u8 {
    const groups = try a.alloc([]u8, p.groups.items.len);
    for (groups, p.groups.items) |*g, n| g.* = try a.alloc(u8, @intCast(n));
    for (groups) |g| @memset(g, 0);
    for (p.copies.items) |c| {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/model-{d:0>5}-of-{d:0>5}.safetensors", .{ dir, c.shard + 1, shards }, 0);
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd);
        const raw = try a.alloc(u8, @intCast(c.len));
        if (std.c.pread(fd, raw.ptr, raw.len, @intCast(c.off)) != @as(isize, @intCast(raw.len))) return error.ShortRead;
        const out = groups[c.g][@intCast(c.dst)..];
        if (c.row == 0) @memcpy(out[0..raw.len], raw) else for (0..raw.len / c.row) |r| @memcpy(out[r * c.width ..][0..c.width], raw[r * c.row + c.at ..][0..c.width]);
    }
    return groups;
}
