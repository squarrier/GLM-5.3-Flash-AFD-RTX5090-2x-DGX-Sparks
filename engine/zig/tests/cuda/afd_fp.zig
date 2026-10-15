//! rank_fingerprint (TF_GLM_FINGERPRINT=1): each AFD role's identity line for the hello (cuda.fingerprint), from what the role
//! loaded. tf-glm-afd embeds every fatbin, so `kernels` lists the images the role's modules were loaded from, by name,
//! plus its Triton set's aot.json (set120 on the attention, set121 on the halves); `exe` covers the binary itself.
//! Reimplemented from @Bizuayeu's 322b4de6 field list (pr/glm53-flash-cuda, Apache-2.0); no code is copied.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const fp = cuda.fingerprint;

const x3 = glm.exl3.kernels;

/// An image a role loaded: its name and the bytes its module was loaded from.
pub const Image = struct { name: []const u8, bytes: []const u8 };

/// A half's fatbins: every EXL3 image its set holds a module of (the six, and 0128's, X1's, SEQ's, MMA's or K4's as
/// loaded), g53_rank when g53kt3r is open, and fn_xchg (xchg.Gpu).
pub fn expertKernels(k: *fp.Kernels, set: *const x3.Set, g53: bool) !void {
    inline for (@typeInfo(x3.Image).@"enum".field_names) |nm| {
        const im = @field(x3.Image, nm);
        if (set.modules[@backingInt(im)] != null) try k.add(nm, x3.bytes(im));
    }
    if (g53) try k.add("g53_rank", cuda.kernels.g53_rank);
    try k.add("fn_xchg", cuda.kernels.fn_xchg);
}

/// The identity line of `role` from `k` (its fatbins so far), the Triton set at `triton`, the checkpoint at `model`, and
/// the GPU's compute capability `sm`; one INFO line names it.
pub fn line(buf: []u8, role: []const u8, k: *fp.Kernels, triton: []const u8, model: []const u8, sm: u32, who: []const u8) ![]const u8 {
    try k.addTriton(triton);
    const l = try fp.local(role, sm, model, k);
    const text = try l.identity().line(buf);
    std.debug.print("INFO {s}: " ++ fp.name ++ "=1: the settings carry fp=1; identity {s}\n", .{ who, text });
    return text;
}

/// Half `half`'s line (afd_expert.serve, after its Gpu opened and before it joins).
pub fn expert(buf: []u8, set: *const x3.Set, g53: bool, triton: []const u8, model: []const u8, sm: u32, half: u32) ![]const u8 {
    var k: fp.Kernels = .{};
    try expertKernels(&k, set, g53);
    var wb: [24]u8 = undefined;
    return line(buf, "expert", &k, triton, model, sm, try std.fmt.bufPrint(&wb, "afd expert {d}", .{half}));
}

/// The attention's line (afd_attn_gpu.run, before it attends): `images`, the fatbins it loaded, by name.
pub fn attn(buf: []u8, images: []const Image, triton: []const u8, model: []const u8, sm: u32) ![]const u8 {
    var k: fp.Kernels = .{};
    for (images) |im| try k.add(im.name, im.bytes);
    try k.add("fn_xchg", cuda.kernels.fn_xchg);
    return line(buf, "attention", &k, triton, model, sm, "attn");
}

/// The attention's embedded fatbins (afd_attn_gpu.zig's `img`; `pack` is --fatbin's bytes when given).
pub const AttnImages = struct {
    pack: []const u8,
    kda: []const u8,
    qmm_group: []const u8,
    qmm_prefill: []const u8,
    experts: []const u8,
    experts_prefill: []const u8,
    select: []const u8,
    pointwise: []const u8,
    topk: []const u8,
    indexing: []const u8,
    movement: []const u8,
    nucleus: []const u8,
    argmax_f64: []const u8,
    grammar_mask: []const u8,
};

/// The fatbins afd_attn_gpu.run loads, by name: the pack pass's, KDA's, Mm's four, DSA select, the sampler's four torch
/// ops; the nucleus pair with sampling on (TF_AFD_SAMPLING), the grammar mask with TF_AFD_GRAMMAR_MASK=1.
pub fn attnImages(out: *[16]Image, im: AttnImages, sampled: bool, masked: bool) []const Image {
    const always = [_]Image{
        .{ .name = "glm_pack", .bytes = im.pack },           .{ .name = "glm_kda", .bytes = im.kda },
        .{ .name = "glm_qmm_group", .bytes = im.qmm_group }, .{ .name = "glm_qmm_prefill", .bytes = im.qmm_prefill },
        .{ .name = "glm_experts", .bytes = im.experts },     .{ .name = "glm_experts_prefill", .bytes = im.experts_prefill },
        .{ .name = "glm_select", .bytes = im.select },       .{ .name = "torch_pointwise", .bytes = im.pointwise },
        .{ .name = "torch_topk", .bytes = im.topk },         .{ .name = "torch_indexing", .bytes = im.indexing },
        .{ .name = "torch_movement", .bytes = im.movement },
    };
    var n: usize = always.len;
    @memcpy(out[0..n], &always);
    if (sampled) {
        out[n] = .{ .name = "torch_nucleus", .bytes = im.nucleus };
        out[n + 1] = .{ .name = "torch_argmax_f64", .bytes = im.argmax_f64 };
        n += 2;
    }
    if (masked) {
        out[n] = .{ .name = "torch_grammar_mask", .bytes = im.grammar_mask };
        n += 1;
    }
    return out[0..n];
}

fn names(k: *const fp.Kernels, out: [][]const u8) [][]const u8 {
    for (out[0..k.n], 0..) |*o, i| o.* = k.lines[i][0..std.mem.indexOfScalar(u8, k.lines[i][0..k.lens[i]], ' ').?];
    return out[0..k.n];
}

test "rank_fingerprint: a half's kernels are the EXL3 images its set loaded, by name, then g53_rank when open and fn_xchg" {
    var set: x3.Set = .{};
    const present: cuda.Module = .{ .d = undefined, .handle = null }; // presence only: never unloaded here
    for (x3.images_all) |im| set.modules[@backingInt(im)] = present;
    set.modules[@backingInt(x3.Image.exl3_x1_experts)] = present;
    var k: fp.Kernels = .{};
    try expertKernels(&k, &set, true);
    var out: [fp.Kernels.max][]const u8 = undefined;
    const got = names(&k, &out);
    const want = [_][]const u8{ "exl3_experts", "exl3_grouped_cb0", "exl3_grouped_cb1", "exl3_grouped_cb2", "exl3_linear", "glm_exl3", "exl3_x1_experts", "g53_rank", "fn_xchg" };
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
    var plain: fp.Kernels = .{};
    try expertKernels(&plain, &set, false);
    try std.testing.expectEqual(want.len - 1, plain.n); // no g53_rank
    try std.testing.expect(!std.mem.eql(u8, &k.digest(), &plain.digest()));
}

test "rank_fingerprint: the attention's fatbins by name: eleven always (--fatbin's bytes as the pack pass's), the nucleus pair with sampling, the grammar mask with the mask" {
    const im: AttnImages = .{ .pack = "own pack", .kda = "", .qmm_group = "", .qmm_prefill = "", .experts = "", .experts_prefill = "", .select = "", .pointwise = "", .topk = "", .indexing = "", .movement = "", .nucleus = "", .argmax_f64 = "", .grammar_mask = "" };
    var lb: [16]Image = undefined;
    const base = attnImages(&lb, im, false, false);
    try std.testing.expectEqual(@as(usize, 11), base.len);
    try std.testing.expectEqualStrings("glm_pack", base[0].name);
    try std.testing.expectEqualStrings("own pack", base[0].bytes);
    try std.testing.expectEqualStrings("torch_movement", base[10].name);
    const all = attnImages(&lb, im, true, true);
    try std.testing.expectEqual(@as(usize, 14), all.len);
    try std.testing.expectEqualStrings("torch_nucleus", all[11].name);
    try std.testing.expectEqualStrings("torch_argmax_f64", all[12].name);
    try std.testing.expectEqualStrings("torch_grammar_mask", all[13].name);
    try std.testing.expectEqualStrings("torch_grammar_mask", attnImages(&lb, im, false, true)[11].name);
}
